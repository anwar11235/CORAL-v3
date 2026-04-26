"""Adaptive Computation Time (ACT) wrapper for CoralInner.

CoralACT wraps CoralInner and manages:
  - ACT carry: recurrent state + step counter + halting flags + current data
  - Carry reset for halted sequences (swap in fresh batch data)
  - Halting decisions: Q-learning with exploration during training
  - Bootstrapped target Q-values via an extra inner forward pass
  - Eval mode: always runs halt_max_steps (no early stopping)

Data flow:
    ACTLossHead.forward(carry, batch)
      → CoralACT.forward(carry, batch)          # one segment
        → CoralInner.forward(inner_carry, data)  # H×L recurrent steps
      ← (new_carry, outputs_dict)
    ← (new_carry, loss, metrics, preds, all_halted)
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn

from coral.models.coral_base import CoralConfig, CoralInner, InnerCarry
from coral.models.coral_v3 import CoralV3Inner, PredMetrics


# ---------------------------------------------------------------------------
# Carry
# ---------------------------------------------------------------------------


@dataclass
class ACTCarry:
    """Outer carry passed between ACT segments.

    Fields:
        inner_carry:   CoralInner's recurrent state (z_H, z_L) — always detached.
        steps:         [B] int32 — how many segments each sequence has run since reset.
        halted:        [B] bool  — True means this sequence halted on the last step.
        current_data:  Dict of [B, ...] tensors — the data batch being processed.
                       For non-halted sequences this is the SAME data as the previous
                       segment; for halted sequences it is the freshly-injected batch.
    """

    inner_carry: InnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]


# ---------------------------------------------------------------------------
# ACT wrapper
# ---------------------------------------------------------------------------


class CoralACT(nn.Module):
    """ACT wrapper around CoralInner.

    Manages the carry lifecycle across segments:
      1. Reset inner_carry and current_data for sequences that halted last step.
      2. Run CoralInner for one segment.
      3. Decide which sequences halt (Q-learning + exploration).
      4. (Training only) Compute bootstrapped target_q_continue via a second
         inner forward pass under no_grad.

    Args:
        config: CoralConfig.  May be passed as a dict (will be coerced).
    """

    def __init__(self, config: CoralConfig) -> None:
        super().__init__()
        if isinstance(config, dict):
            config = CoralConfig(**config)
        self.config = config
        self.inner = CoralInner(config)

    @property
    def puzzle_emb(self):
        """Expose puzzle_emb for the sparse embedding optimizer."""
        return self.inner.puzzle_emb  # AttributeError if puzzle_emb_ndim == 0

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> ACTCarry:
        """Create the initial carry for a new evaluation run.

        All sequences start as halted so that the first forward call injects
        the real batch data and resets the inner carry to H_init / L_init.

        Args:
            batch: A sample batch (used only to determine shapes / dtypes).

        Returns:
            ACTCarry with halted=True for every sequence.
        """
        batch_size = batch["inputs"].shape[0]
        device = batch["inputs"].device
        return ACTCarry(
            inner_carry=self.inner.empty_carry(batch_size, device=device),
            steps=torch.zeros(batch_size, dtype=torch.int32, device=device),
            halted=torch.ones(batch_size, dtype=torch.bool, device=device),
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def forward(
        self,
        carry: ACTCarry,
        batch: Dict[str, torch.Tensor],
    ) -> tuple:
        """Run one ACT segment.

        Args:
            carry: Current ACTCarry from the previous segment.
            batch: New data batch; used to replace halted sequences.

        Returns:
            (new_carry, outputs) where outputs is a dict containing:
                "logits"           — [B, seq_len, vocab_size]
                "q_halt_logits"    — [B] float32
                "q_continue_logits"— [B] float32
                "target_q_continue"— [B] float32  (training only, if halt_max_steps > 1)
        """
        # --- 1. Reset halted sequences ---
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)

        # Step counter restarts from 0 for sequences that just halted
        new_steps = torch.where(carry.halted, torch.zeros_like(carry.steps), carry.steps)

        # Swap in fresh batch data for halted sequences
        new_current_data = {
            k: torch.where(
                carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)),
                batch[k],
                v,
            )
            for k, v in carry.current_data.items()
        }

        # --- 2. Run inner model (one segment = H_cycles × L_cycles steps) ---
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(
            new_inner_carry, new_current_data
        )

        outputs: Dict[str, torch.Tensor] = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
        }

        # --- 3. Halting logic (no gradient needed) ---
        with torch.no_grad():
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps

            # Default: halt only at max_steps (used during eval)
            halted = is_last_step

            if self.training and self.config.halt_max_steps > 1:
                # Q-learning halt signal: halt when q_halt > q_continue
                halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration: with prob halt_exploration_prob, enforce a random
                # minimum number of steps before allowing halting.
                # min_halt_steps = 0 (not drawn) OR Uniform[2, halt_max_steps]
                exploration_mask = torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob
                min_halt_steps = exploration_mask.to(torch.int32) * torch.randint_like(
                    new_steps, low=2, high=self.config.halt_max_steps + 1
                )
                halted = halted & (new_steps >= min_halt_steps)

                # --- 4. Bootstrap target Q for the continue action ---
                # Run an EXTRA inner forward (no grad) to get next-step Q values.
                # inner returns (new_carry, logits, (q_halt, q_continue)); we only need [-1].
                next_q_halt, next_q_continue = self.inner(new_inner_carry, new_current_data)[-1]
                # target_q_continue = sigmoid(q_halt) at last step, else sigmoid(max(q_halt, q_continue))
                target_q_continue = torch.sigmoid(
                    torch.where(
                        is_last_step,
                        next_q_halt,
                        torch.maximum(next_q_halt, next_q_continue),
                    )
                )
                outputs["target_q_continue"] = target_q_continue

        new_carry = ACTCarry(
            inner_carry=new_inner_carry,
            steps=new_steps,
            halted=halted,
            current_data=new_current_data,
        )
        return new_carry, outputs


# ---------------------------------------------------------------------------
# CoralV3ACT — ACT wrapper for CoralV3Inner (Phase 1)
# ---------------------------------------------------------------------------


class CoralV3ACT(nn.Module):
    """ACT wrapper around CoralV3Inner.

    Identical to CoralACT except:
      - Uses CoralV3Inner instead of CoralInner.
      - Handles the optional 4th return value (PredMetrics) from CoralV3Inner
        when use_predictive_coding=True.
      - Forwards epsilon_final, pi_final, and logging scalars through the
        outputs dict so CoralV3LossHead can compute the free energy loss.

    When config.use_predictive_coding=False the behaviour is identical to
    CoralACT (no extra overhead, no extra outputs).

    Args:
        config: CoralConfig (with use_predictive_coding field).
    """

    def __init__(self, config: CoralConfig) -> None:
        super().__init__()
        if isinstance(config, dict):
            config = CoralConfig(**config)
        self.config = config
        self.inner = CoralV3Inner(config)

    @property
    def puzzle_emb(self):
        """Expose puzzle_emb for the sparse embedding optimizer."""
        return self.inner.puzzle_emb  # AttributeError if puzzle_emb_ndim == 0

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> ACTCarry:
        """Create the initial carry for a new evaluation run."""
        batch_size = batch["inputs"].shape[0]
        device = batch["inputs"].device
        return ACTCarry(
            inner_carry=self.inner.empty_carry(batch_size, device=device),
            steps=torch.zeros(batch_size, dtype=torch.int32, device=device),
            halted=torch.ones(batch_size, dtype=torch.bool, device=device),
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    @torch.compiler.disable(recursive=False)
    def forward(
        self,
        carry: ACTCarry,
        batch: Dict[str, torch.Tensor],
    ) -> tuple:
        """Run one ACT segment.

        Returns:
            (new_carry, outputs) where outputs is a dict containing:
                "logits"             — [B, seq_len, vocab_size]
                "q_halt_logits"      — [B] float32
                "q_continue_logits"  — [B] float32
                "target_q_continue"  — [B] float32  (training only)
            When use_predictive_coding=True, outputs additionally contains:
                "epsilon_final"      — [B, seq_len, hidden_size] with grad
                "pi_final"           — [B, seq_len, hidden_size] with grad
                "pred_error_norm"    — scalar (mean over all steps, detached)
                "precision_mean"     — scalar (mean over all steps, detached)

        Note: @torch.compiler.disable prevents dynamo from tracing through this
        function when the outer model is compiled.  The hot transformer kernels
        (H_level, L_level) are compiled as standalone sub-modules in build_model
        and are invoked via their compiled __call__ from this eager context.
        This avoids graph-break / object-identity guard recompile storms caused
        by PredMetrics.moe_lb_loss tensors crossing the disabled moe_losses()
        boundary back into a compiled region (act.py:277 in satisfied-owl run).
        """
        # --- 1. Reset halted sequences ---
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, torch.zeros_like(carry.steps), carry.steps)
        new_current_data = {
            k: torch.where(
                carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)),
                batch[k],
                v,
            )
            for k, v in carry.current_data.items()
        }

        # Determine whether any sequence is about to reach its last ACT segment so
        # that _maybe_record_crystal only fires once per training step instead of
        # halt_max_steps times.  new_steps at this point is still the pre-increment
        # value; add 1 to get the count after this segment completes.
        _steps_before = torch.where(carry.halted, torch.zeros_like(carry.steps), carry.steps)
        _is_last_segment = bool((_steps_before + 1 >= self.config.halt_max_steps).any())

        # --- 2. Run inner model ---
        inner_result = self.inner(new_inner_carry, new_current_data, is_last_segment=_is_last_segment)
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = inner_result[:3]

        outputs: Dict[str, torch.Tensor] = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
        }

        # Unpack prediction / routing / crystal metrics when any mechanism is active
        _any_mechanism = (
            self.config.use_predictive_coding
            or self.config.use_columnar_routing
            or self.config.use_crystallization
        )
        if _any_mechanism:
            pred_metrics: PredMetrics = inner_result[3]
            if pred_metrics.epsilon_final is not None:
                outputs["epsilon_final"] = pred_metrics.epsilon_final  # type: ignore[assignment]
                outputs["pi_final"] = pred_metrics.pi_final  # type: ignore[assignment]
            if pred_metrics.pred_error_norms:
                outputs["prediction_error"] = torch.stack(pred_metrics.pred_error_norms).mean()
                outputs["precision_mean"] = torch.stack(pred_metrics.precision_means).mean()
            if pred_metrics.routing_logits_H is not None:
                outputs["routing_logits_H"] = pred_metrics.routing_logits_H  # type: ignore[assignment]
                outputs["routing_logits_L"] = pred_metrics.routing_logits_L  # type: ignore[assignment]
            if pred_metrics.moe_recon_loss is not None:
                outputs["moe_recon_loss"] = pred_metrics.moe_recon_loss  # type: ignore[assignment]
            if pred_metrics.moe_lb_loss is not None:
                outputs["moe_lb_loss"] = pred_metrics.moe_lb_loss        # type: ignore[assignment]
            if self.config.use_crystallization:
                outputs["moe_passthrough_weight"] = torch.tensor(
                    pred_metrics.moe_passthrough_weight,
                    device=logits.device,
                    dtype=torch.float32,
                )
                if pred_metrics.moe_routing_entropy is not None:
                    outputs["moe_routing_entropy"] = torch.tensor(
                        pred_metrics.moe_routing_entropy,
                        device=logits.device,
                        dtype=torch.float32,
                    )
                if pred_metrics.moe_codebook_util_frac is not None:
                    outputs["moe_codebook_util_frac"] = torch.tensor(
                        pred_metrics.moe_codebook_util_frac,
                        device=logits.device,
                        dtype=torch.float32,
                    )

        # --- 3. Halting logic ---
        with torch.no_grad():
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            halted = is_last_step

            if self.training and self.config.halt_max_steps > 1:
                halted = halted | (q_halt_logits > q_continue_logits)

                exploration_mask = torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob
                min_halt_steps = exploration_mask.to(torch.int32) * torch.randint_like(
                    new_steps, low=2, high=self.config.halt_max_steps + 1
                )
                halted = halted & (new_steps >= min_halt_steps)

                # --- 4. Bootstrap target Q ---
                # Take index 2 explicitly — inner may return 3 or 4 values.
                next_q_halt, next_q_continue = self.inner(new_inner_carry, new_current_data)[2]
                target_q_continue = torch.sigmoid(
                    torch.where(
                        is_last_step,
                        next_q_halt,
                        torch.maximum(next_q_halt, next_q_continue),
                    )
                )
                outputs["target_q_continue"] = target_q_continue

        new_carry = ACTCarry(
            inner_carry=new_inner_carry,
            steps=new_steps,
            halted=halted,
            current_data=new_current_data,
        )
        return new_carry, outputs
