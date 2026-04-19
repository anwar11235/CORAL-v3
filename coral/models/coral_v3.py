"""CoralV3Inner — CORAL base with optional precision-weighted predictive coding,
sparse columnar routing, and recognition-gated crystallization.

Phase 1 — Predictive coding:
    H predicts L's state; L receives that prediction; H receives the
    precision-weighted prediction error.

Phase 2 — Columnar routing:
    Each ReasoningModule uses ColumnarReasoningModule (index-select, S=8, k=2).

Phase 3 — Crystallization:
    Before each H-cycle (during EVAL only), the recognition network checks
    whether the current state matches a stored codebook entry with high confidence.
    If so, z_L is replaced by the codebook entry and the L-module inner loop is
    skipped entirely.  During TRAINING the full recurrence always runs, and the
    confidence gate is trained via a supervision loss that teaches it to predict
    when bypass would have been safe.

All mechanisms are independently toggled by CoralConfig flags.  When all are
False the forward pass is identical to CoralInner (no overhead).

Dispatch table:
  pc=F, cr=F, cry=F  →  super().forward() (CoralInner, unchanged)
  pc=F, cr=F, cry=T  →  _forward_baseline()
  pc=T, cr=F, cry=*  →  _forward_with_pc()
  pc=F, cr=T, cry=*  →  _forward_with_routing()
  pc=T, cr=T, cry=*  →  _forward_with_pc_and_routing()

Crystallization logic is embedded in every non-baseline path via three helpers:
  _maybe_crystal_bypass_nograd()  — check + execute bypass (eval, non-last h_step)
  _maybe_record_crystal()         — add state to ring buffer (training, after H update)
  _compute_crystal_supervision_loss() — BCE gate loss (1-step-grad, training only)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from coral.models.coral_base import CoralConfig, CoralInner, InnerCarry
from coral.models.columnar import ColumnarReasoningModule
from coral.models.crystallization import (
    CrystallizationBuffer,
    RecognitionNetwork,
    crystallization_diagnostics,
    crystallization_supervision_loss,
)
from coral.models.prediction import PredictionNet, PrecisionNet
from coral.models.transformer_block import TransformerBlockConfig


# ---------------------------------------------------------------------------
# Metrics carrier
# ---------------------------------------------------------------------------


@dataclass
class PredMetrics:
    """Statistics collected during one inner forward pass.

    pred_error_norms and precision_means:
        Detached scalars from every recurrent step (Phase 1 only).

    epsilon_final and pi_final:
        In-graph tensors from the 1-step-grad step; used by CoralV3LossHead
        for the free energy loss (Phase 1).  None when PC is disabled.

    routing_logits_H and routing_logits_L:
        In-graph tensors from the 1-step-grad step; used by CoralV3LossHead
        for the load-balancing loss (Phase 2).  None when routing is disabled.

    crystal_supervision_loss_final:
        In-graph scalar; used by CoralV3LossHead for the crystal gate BCE loss
        (Phase 3, training only).  None when crystallization is disabled or
        during eval.

    crystal_bypass_count:
        Number of H-cycles bypassed via crystallization in the no_grad section
        (always 0 during training).
    """

    pred_error_norms: List[torch.Tensor]
    precision_means: List[torch.Tensor]
    epsilon_final: Optional[torch.Tensor]
    pi_final: Optional[torch.Tensor]
    routing_logits_H: Optional[List[torch.Tensor]] = field(default=None)
    routing_logits_L: Optional[List[torch.Tensor]] = field(default=None)
    crystal_supervision_loss_final: Optional[torch.Tensor] = field(default=None)
    crystal_bypass_count: int = field(default=0)
    crystal_confidence_mean: float = field(default=0.0)
    crystal_reconstruction_error: Optional[torch.Tensor] = field(default=None)
    crystal_target_confidence_mean: Optional[torch.Tensor] = field(default=None)


# ---------------------------------------------------------------------------
# CoralV3Inner
# ---------------------------------------------------------------------------


class CoralV3Inner(CoralInner):
    """CORAL inner model — extends CoralInner with Phase 1/2/3 mechanisms.

    When all flags are False the forward pass is identical to CoralInner.
    Each enabled mechanism adds parameters and changes the forward pass as
    described in the module docstring.
    """

    def __init__(self, config: CoralConfig) -> None:
        super().__init__(config)

        # Gate activation flag — False during bootstrap phase, True afterwards.
        # Set externally by the training loop after the first codebook consolidation.
        # When False, crystallization_supervision_loss is suppressed (gate receives no
        # BCE gradient) so the random codebook cannot poison the confidence head before
        # real codebook entries exist.
        self._crystal_gate_active: bool = config.crystal_bootstrap_steps == 0

        # --- Phase 1: predictive coding ---
        if config.use_predictive_coding:
            dim = config.hidden_size
            self.prediction_net = PredictionNet(h_dim=dim, l_dim=dim)
            self.precision_net = PrecisionNet(dim=dim)

        # --- Phase 2: sparse columnar routing ---
        if config.use_columnar_routing:
            block_cfg = TransformerBlockConfig(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                expansion=config.expansion,
                rms_norm_eps=config.rms_norm_eps,
            )
            self.H_level = ColumnarReasoningModule(
                config=block_cfg,
                num_layers=config.H_layers,
                S=config.num_columns,
                k=config.active_columns,
            )
            self.L_level = ColumnarReasoningModule(
                config=block_cfg,
                num_layers=config.L_layers,
                S=config.num_columns,
                k=config.active_columns,
            )

        # --- Phase 3: recognition-gated crystallization ---
        if config.use_crystallization:
            self.recognition_net = RecognitionNetwork(
                h_dim=config.hidden_size,
                l_dim=config.hidden_size,
                codebook_size=config.codebook_size,
                proj_dim=config.crystal_proj_dim,
            )
            self.crystal_buffer = CrystallizationBuffer(
                capacity=config.crystal_buffer_capacity,
            )

    # ------------------------------------------------------------------
    # Crystallization helpers
    # ------------------------------------------------------------------

    @torch.compiler.disable(recursive=False)
    def _maybe_crystal_bypass_nograd(
        self,
        h_step: int,
        z_H: torch.Tensor,
        z_L: torch.Tensor,
        cos_sin,
    ) -> Tuple[bool, torch.Tensor, torch.Tensor, float]:
        """Attempt crystallization bypass for a non-last H-cycle in eval mode.

        Bypass fires when:
          - use_crystallization is True
          - model is in eval mode (self.training is False)
          - this is NOT the last H-cycle (last H update is reserved for 1-step-grad)
          - mean confidence exceeds the threshold

        When bypass fires:
          - z_L is replaced by the nearest codebook entry
          - H is updated immediately using the bypassed z_L (handling PC/routing)
          - returns True so the caller can `continue` past the L inner loop

        Args:
            h_step: Current H-cycle index.
            z_H:    [B, seq, dim] current H state.
            z_L:    [B, seq, dim] current L state.
            cos_sin: RoPE cache (or None).

        Returns:
            (bypassed, new_z_H, new_z_L, confidence_mean)
            confidence_mean is the mean recognition confidence for this H-cycle
            (0.0 when crystallization is disabled, in training mode, or last H-cycle).
        """
        is_last_h = h_step == self.config.H_cycles - 1
        if not self.config.use_crystallization or self.training or is_last_h:
            return False, z_H, z_L, 0.0

        confidence, nearest_code, _ = self.recognition_net(z_H, z_L)
        conf_mean = confidence.mean().item()
        if conf_mean <= self.config.crystal_confidence_threshold:
            return False, z_H, z_L, conf_mean

        # Bypass: substitute z_L, then update H with the substituted value
        # Cast to z_L's dtype — codebook is fp32; flash-attn requires fp16/bf16.
        z_L = nearest_code.to(z_L.dtype)
        if self.config.use_predictive_coding:
            mu_L = self.prediction_net(z_H)
            epsilon = z_L - mu_L
            pi = self.precision_net(z_L)
            injection = pi * epsilon
        else:
            injection = z_L

        if self.config.use_columnar_routing:
            z_H, _ = self.H_level(z_H, injection, cos_sin=cos_sin)
        else:
            z_H = self.H_level(z_H, injection, cos_sin=cos_sin)

        return True, z_H, z_L, conf_mean

    @torch.compiler.disable(recursive=False)
    def _maybe_record_crystal(
        self,
        z_H: torch.Tensor,
        z_L: torch.Tensor,
        is_last_h: bool = False,
        is_last_segment: bool = False,
    ) -> None:
        """Add current (z_H, z_L) to the crystal buffer during training.

        Only records on the last H-cycle of the last ACT segment to avoid
        redundant recognition-network forward passes and GPU→CPU transfers
        at every segment (would be halt_max_steps × per training step otherwise).

        Called after each non-last H-cycle's H update (still inside no_grad).
        The buffer stores CPU tensors; detachment and CPU transfer happen inside add().
        """
        if not is_last_h or not is_last_segment:
            return
        if not self.config.use_crystallization or not self.training:
            return
        key = self.recognition_net.compute_key(z_H, z_L)   # [B, proj_dim*2]
        pooled_z_L = z_L.mean(dim=1)                        # [B, l_dim]
        self.crystal_buffer.add(key, pooled_z_L)

    @torch.compiler.disable(recursive=False)
    def _compute_crystal_supervision_loss(
        self, z_H: torch.Tensor, z_L_final: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Compute crystallization supervision loss and reconstruction diagnostics.

        Called AFTER the final L update and BEFORE the final H update.

        Returns:
            (bce_loss, mean_reconstruction_error, target_confidence_mean)

            bce_loss is None during the bootstrap phase (gate not active) or
            during eval — gate receives no BCE gradient until _crystal_gate_active
            is True.  Diagnostics are computed whenever crystallization is enabled,
            giving visibility into codebook quality before the gate goes live.

            All three are None when use_crystallization is False.
        """
        if not self.config.use_crystallization:
            return None, None, None

        if self.training and self._crystal_gate_active:
            bce_loss, mean_recon, target_conf = crystallization_supervision_loss(
                self.recognition_net, z_H, z_L_final
            )
            return bce_loss, mean_recon, target_conf

        # Bootstrap phase (training, gate not active) or eval: diagnostics only
        mean_recon, target_conf = crystallization_diagnostics(
            self.recognition_net, z_H, z_L_final
        )
        return None, mean_recon, target_conf

    def consolidate_codebook(self, is_first_consolidation: bool = False) -> Optional[float]:
        """Offline codebook update — call periodically from the training loop.

        Runs k-means-like assignment + EMA on the crystal buffer, then clears it.

        Args:
            is_first_consolidation: When True, requires 80% buffer fill and uses
                                    ema_weight=1.0 (full replace of random init).
                                    Returns None to defer if buffer is not full enough.

        Returns:
            Fraction of codebook entries with at least one assignment, or None
            if the buffer was too small to consolidate.
        """
        if not self.config.use_crystallization:
            return None
        device = str(next(self.recognition_net.parameters()).device)
        usage = self.crystal_buffer.consolidate(
            self.recognition_net,
            device=device,
            is_first_consolidation=is_first_consolidation,
        )
        if usage is not None:
            self.crystal_buffer.clear()
        return usage

    # ------------------------------------------------------------------
    # Forward dispatch
    # ------------------------------------------------------------------

    def forward(
        self,
        carry: InnerCarry,
        batch: Dict[str, torch.Tensor],
        is_last_segment: bool = False,
    ) -> Tuple:
        """Run one segment with the enabled mechanisms.

        Args:
            is_last_segment: True when this is the final ACT segment for at least
                one sequence in the batch.  Controls whether _maybe_record_crystal
                writes to the ring buffer — recording only on the last segment avoids
                halt_max_steps redundant GPU→CPU transfers per training step.

        Returns:
            3-tuple when all mechanisms are disabled (same as CoralInner):
                (new_carry, output, (q_halt, q_continue))
            4-tuple otherwise:
                (new_carry, output, (q_halt, q_continue), pred_metrics)
        """
        pc = self.config.use_predictive_coding
        cr = self.config.use_columnar_routing
        cry = self.config.use_crystallization

        if not pc and not cr and not cry:
            return super().forward(carry, batch)
        elif pc and cr:
            return self._forward_with_pc_and_routing(carry, batch, is_last_segment=is_last_segment)
        elif pc:
            return self._forward_with_pc(carry, batch, is_last_segment=is_last_segment)
        elif cr:
            return self._forward_with_routing(carry, batch, is_last_segment=is_last_segment)
        else:
            # cry=True, pc=False, cr=False
            return self._forward_baseline(carry, batch, is_last_segment=is_last_segment)

    # ------------------------------------------------------------------
    # Forward implementations
    # ------------------------------------------------------------------

    def _forward_baseline(
        self,
        carry: InnerCarry,
        batch: Dict[str, torch.Tensor],
        is_last_segment: bool = False,
    ) -> Tuple[InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], PredMetrics]:
        """Baseline inner forward (no PC, no routing) with crystallization support."""
        cos_sin = self._cos_sin()
        input_embeddings = self._input_embeddings(
            batch["inputs"],
            batch.get("puzzle_identifiers"),
        )

        with torch.no_grad():
            z_H, z_L = carry.z_H, carry.z_L
            crystal_bypassed = 0
            _conf_total = 0.0
            _conf_steps = 0

            for h_step in range(self.config.H_cycles):
                bypassed, z_H, z_L, conf = self._maybe_crystal_bypass_nograd(
                    h_step, z_H, z_L, cos_sin
                )
                if conf > 0.0:
                    _conf_total += conf
                    _conf_steps += 1
                if bypassed:
                    crystal_bypassed += 1
                    continue

                for l_step in range(self.config.L_cycles):
                    is_last_l = (
                        h_step == self.config.H_cycles - 1
                        and l_step == self.config.L_cycles - 1
                    )
                    if not is_last_l:
                        z_L = self.L_level(z_L, z_H + input_embeddings, cos_sin=cos_sin)

                if not (h_step == self.config.H_cycles - 1):
                    z_H = self.H_level(z_H, z_L, cos_sin=cos_sin)
                    self._maybe_record_crystal(
                        z_H, z_L,
                        is_last_h=(h_step == self.config.H_cycles - 2),
                        is_last_segment=is_last_segment,
                    )

        assert not z_H.requires_grad and not z_L.requires_grad

        # 1-step gradient
        z_L = self.L_level(z_L, z_H + input_embeddings, cos_sin=cos_sin)
        crystal_loss_final, crystal_recon_err, crystal_tgt_conf = self._compute_crystal_supervision_loss(z_H, z_L)
        z_H = self.H_level(z_H, z_L, cos_sin=cos_sin)

        new_carry = InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)

        pred_metrics = PredMetrics(
            pred_error_norms=[],
            precision_means=[],
            epsilon_final=None,
            pi_final=None,
            crystal_supervision_loss_final=crystal_loss_final,
            crystal_bypass_count=crystal_bypassed,
            crystal_confidence_mean=_conf_total / _conf_steps if _conf_steps > 0 else 0.0,
            crystal_reconstruction_error=crystal_recon_err,
            crystal_target_confidence_mean=crystal_tgt_conf,
        )
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1]), pred_metrics

    def _forward_with_pc(
        self,
        carry: InnerCarry,
        batch: Dict[str, torch.Tensor],
        is_last_segment: bool = False,
    ) -> Tuple[InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], PredMetrics]:
        """Inner forward with predictive coding (and optional crystallization)."""
        cos_sin = self._cos_sin()
        input_embeddings = self._input_embeddings(
            batch["inputs"],
            batch.get("puzzle_identifiers"),
        )

        pred_error_norms: List[torch.Tensor] = []
        precision_means: List[torch.Tensor] = []

        with torch.no_grad():
            z_H, z_L = carry.z_H, carry.z_L
            crystal_bypassed = 0
            _conf_total = 0.0
            _conf_steps = 0

            for h_step in range(self.config.H_cycles):
                bypassed, z_H, z_L, conf = self._maybe_crystal_bypass_nograd(
                    h_step, z_H, z_L, cos_sin
                )
                if conf > 0.0:
                    _conf_total += conf
                    _conf_steps += 1
                if bypassed:
                    crystal_bypassed += 1
                    continue

                for l_step in range(self.config.L_cycles):
                    is_last_l = (
                        h_step == self.config.H_cycles - 1
                        and l_step == self.config.L_cycles - 1
                    )
                    if not is_last_l:
                        mu_L = self.prediction_net(z_H)
                        z_L = self.L_level(z_L, mu_L + input_embeddings, cos_sin=cos_sin)
                        epsilon = z_L - mu_L
                        pi = self.precision_net(z_L)
                        xi = pi * epsilon
                        pred_error_norms.append(epsilon.norm(dim=-1).mean())
                        precision_means.append(pi.mean())

                if not (h_step == self.config.H_cycles - 1):
                    z_H = self.H_level(z_H, xi, cos_sin=cos_sin)  # type: ignore[possibly-undefined]
                    self._maybe_record_crystal(
                        z_H, z_L,
                        is_last_h=(h_step == self.config.H_cycles - 2),
                        is_last_segment=is_last_segment,
                    )

        assert not z_H.requires_grad and not z_L.requires_grad

        # 1-step gradient
        mu_L = self.prediction_net(z_H)
        z_L = self.L_level(z_L, mu_L + input_embeddings, cos_sin=cos_sin)
        epsilon_final = z_L - mu_L
        pi_final = self.precision_net(z_L)
        xi = pi_final * epsilon_final

        crystal_loss_final, crystal_recon_err, crystal_tgt_conf = self._compute_crystal_supervision_loss(z_H, z_L)

        z_H = self.H_level(z_H, xi, cos_sin=cos_sin)

        pred_error_norms.append(epsilon_final.detach().norm(dim=-1).mean())
        precision_means.append(pi_final.detach().mean())

        new_carry = InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)

        pred_metrics = PredMetrics(
            pred_error_norms=pred_error_norms,
            precision_means=precision_means,
            epsilon_final=epsilon_final,
            pi_final=pi_final,
            crystal_supervision_loss_final=crystal_loss_final,
            crystal_bypass_count=crystal_bypassed,
            crystal_confidence_mean=_conf_total / _conf_steps if _conf_steps > 0 else 0.0,
            crystal_reconstruction_error=crystal_recon_err,
            crystal_target_confidence_mean=crystal_tgt_conf,
        )
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1]), pred_metrics

    def _forward_with_routing(
        self,
        carry: InnerCarry,
        batch: Dict[str, torch.Tensor],
        is_last_segment: bool = False,
    ) -> Tuple[InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], PredMetrics]:
        """Inner forward with columnar routing (and optional crystallization)."""
        cos_sin = self._cos_sin()
        input_embeddings = self._input_embeddings(
            batch["inputs"],
            batch.get("puzzle_identifiers"),
        )

        with torch.no_grad():
            z_H, z_L = carry.z_H, carry.z_L
            crystal_bypassed = 0
            _conf_total = 0.0
            _conf_steps = 0

            for h_step in range(self.config.H_cycles):
                bypassed, z_H, z_L, conf = self._maybe_crystal_bypass_nograd(
                    h_step, z_H, z_L, cos_sin
                )
                if conf > 0.0:
                    _conf_total += conf
                    _conf_steps += 1
                if bypassed:
                    crystal_bypassed += 1
                    continue

                for l_step in range(self.config.L_cycles):
                    is_last_l = (
                        h_step == self.config.H_cycles - 1
                        and l_step == self.config.L_cycles - 1
                    )
                    if not is_last_l:
                        z_L, _ = self.L_level(z_L, z_H + input_embeddings, cos_sin=cos_sin)

                if not (h_step == self.config.H_cycles - 1):
                    z_H, _ = self.H_level(z_H, z_L, cos_sin=cos_sin)
                    self._maybe_record_crystal(
                        z_H, z_L,
                        is_last_h=(h_step == self.config.H_cycles - 2),
                        is_last_segment=is_last_segment,
                    )

        assert not z_H.requires_grad and not z_L.requires_grad

        # 1-step gradient — collect routing logits
        z_L, routing_logits_L = self.L_level(z_L, z_H + input_embeddings, cos_sin=cos_sin)
        crystal_loss_final, crystal_recon_err, crystal_tgt_conf = self._compute_crystal_supervision_loss(z_H, z_L)
        z_H, routing_logits_H = self.H_level(z_H, z_L, cos_sin=cos_sin)

        new_carry = InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)

        pred_metrics = PredMetrics(
            pred_error_norms=[],
            precision_means=[],
            epsilon_final=None,
            pi_final=None,
            routing_logits_H=routing_logits_H,
            routing_logits_L=routing_logits_L,
            crystal_supervision_loss_final=crystal_loss_final,
            crystal_bypass_count=crystal_bypassed,
            crystal_confidence_mean=_conf_total / _conf_steps if _conf_steps > 0 else 0.0,
            crystal_reconstruction_error=crystal_recon_err,
            crystal_target_confidence_mean=crystal_tgt_conf,
        )
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1]), pred_metrics

    def _forward_with_pc_and_routing(
        self,
        carry: InnerCarry,
        batch: Dict[str, torch.Tensor],
        is_last_segment: bool = False,
    ) -> Tuple[InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], PredMetrics]:
        """Inner forward with both predictive coding and columnar routing (and optional crystallization)."""
        cos_sin = self._cos_sin()
        input_embeddings = self._input_embeddings(
            batch["inputs"],
            batch.get("puzzle_identifiers"),
        )

        pred_error_norms: List[torch.Tensor] = []
        precision_means: List[torch.Tensor] = []

        with torch.no_grad():
            z_H, z_L = carry.z_H, carry.z_L
            crystal_bypassed = 0
            _conf_total = 0.0
            _conf_steps = 0

            for h_step in range(self.config.H_cycles):
                bypassed, z_H, z_L, conf = self._maybe_crystal_bypass_nograd(
                    h_step, z_H, z_L, cos_sin
                )
                if conf > 0.0:
                    _conf_total += conf
                    _conf_steps += 1
                if bypassed:
                    crystal_bypassed += 1
                    continue

                for l_step in range(self.config.L_cycles):
                    is_last_l = (
                        h_step == self.config.H_cycles - 1
                        and l_step == self.config.L_cycles - 1
                    )
                    if not is_last_l:
                        mu_L = self.prediction_net(z_H)
                        z_L, _ = self.L_level(z_L, mu_L + input_embeddings, cos_sin=cos_sin)
                        epsilon = z_L - mu_L
                        pi = self.precision_net(z_L)
                        xi = pi * epsilon
                        pred_error_norms.append(epsilon.norm(dim=-1).mean())
                        precision_means.append(pi.mean())

                if not (h_step == self.config.H_cycles - 1):
                    z_H, _ = self.H_level(z_H, xi, cos_sin=cos_sin)  # type: ignore[possibly-undefined]
                    self._maybe_record_crystal(
                        z_H, z_L,
                        is_last_h=(h_step == self.config.H_cycles - 2),
                        is_last_segment=is_last_segment,
                    )

        assert not z_H.requires_grad and not z_L.requires_grad

        # 1-step gradient — collect both PC tensors and routing logits
        mu_L = self.prediction_net(z_H)
        z_L, routing_logits_L = self.L_level(z_L, mu_L + input_embeddings, cos_sin=cos_sin)
        epsilon_final = z_L - mu_L
        pi_final = self.precision_net(z_L)
        xi = pi_final * epsilon_final

        crystal_loss_final, crystal_recon_err, crystal_tgt_conf = self._compute_crystal_supervision_loss(z_H, z_L)

        z_H, routing_logits_H = self.H_level(z_H, xi, cos_sin=cos_sin)

        pred_error_norms.append(epsilon_final.detach().norm(dim=-1).mean())
        precision_means.append(pi_final.detach().mean())

        new_carry = InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)

        pred_metrics = PredMetrics(
            pred_error_norms=pred_error_norms,
            precision_means=precision_means,
            epsilon_final=epsilon_final,
            pi_final=pi_final,
            routing_logits_H=routing_logits_H,
            routing_logits_L=routing_logits_L,
            crystal_supervision_loss_final=crystal_loss_final,
            crystal_bypass_count=crystal_bypassed,
            crystal_confidence_mean=_conf_total / _conf_steps if _conf_steps > 0 else 0.0,
            crystal_reconstruction_error=crystal_recon_err,
            crystal_target_confidence_mean=crystal_tgt_conf,
        )
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1]), pred_metrics
