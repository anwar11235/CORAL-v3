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
        return ACTCarry(
            inner_carry=self.inner.empty_carry(batch_size),
            steps=torch.zeros(batch_size, dtype=torch.int32),
            halted=torch.ones(batch_size, dtype=torch.bool),
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
