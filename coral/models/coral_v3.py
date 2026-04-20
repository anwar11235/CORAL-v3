"""CoralV3Inner — CORAL base with precision-weighted predictive coding
and Phase 3b Soft MoE Crystallization.

Phase 1 — Predictive coding:
    H predicts L's state; L receives that prediction; H receives the
    precision-weighted prediction error (free-energy minimisation).

Phase 2 — Columnar routing (STUBBED):
    agate-cuckoo and curly-manatee both collapsed without convergence.
    Dispatch paths that enable columnar routing raise NotImplementedError.
    Reviving columnar routing requires its own redesign; stubs are kept so
    the dispatch-table structure is preserved for future use.

Phase 3b — Soft MoE Crystallization:
    A SpatialMoECodebook routes each L-state through K_modes full-spatial
    codebook experts plus one passthrough expert (the recurrence output z_L_rec).
    The convex combination z_L_out = w_pt*z_L_rec + (1-w_pt)*z_bypass replaces
    the raw L-level output at EVERY H-cycle (no hard bypass, no is_last_h gate).
    During bootstrap (steps 0..crystal_bootstrap_steps) the codebook is masked
    (w_pt forced to 1.0) so random codebook values cannot corrupt z_L.
    After the first spatial k-means consolidation the mask is lifted and end-to-
    end gradient flows to codebook_values via the unweighted reconstruction loss.

Dispatch table:
  pc=F, cr=F, cry=F  →  super().forward() (CoralInner, unchanged, 3-tuple)
  pc=T, cr=F, cry=*  →  _forward_with_pc()        ← ONLY active path
  pc=F, cr=T, …     →  NotImplementedError        (columnar routing stubbed)
  pc=T, cr=T, …     →  NotImplementedError        (columnar routing stubbed)
  pc=F, cr=F, cry=T →  NotImplementedError        (cry without PC unsupported)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from coral.models.coral_base import CoralConfig, CoralInner, InnerCarry
from coral.models.columnar import ColumnarReasoningModule
from coral.models.crystallization import (
    CrystallizationBuffer,
    SpatialMoECodebook,
)
from coral.models.prediction import PredictionNet, PrecisionNet
from coral.models.transformer_block import TransformerBlockConfig


# ---------------------------------------------------------------------------
# Metrics carrier
# ---------------------------------------------------------------------------


@dataclass
class PredMetrics:
    """Statistics collected during one inner forward pass.

    pred_error_norms, precision_means:
        Detached scalars from every recurrent step (Phase 1 only).

    epsilon_final, pi_final:
        In-graph tensors from the 1-step-grad step; used by CoralV3LossHead
        for the free energy loss (Phase 1).  None when PC is disabled.

    routing_logits_H, routing_logits_L:
        Reserved for future columnar routing; always None in Phase 3b.

    moe_recon_loss, moe_lb_loss:
        In-graph scalars from the 1-step-grad step; forwarded to CoralV3LossHead
        as L_recon and L_lb terms.  None during bootstrap, eval, or when
        crystallization is disabled.

    moe_passthrough_weight:
        Mean w_pt (passthrough expert weight) for logging.  1.0 during bootstrap
        (all weight on recurrence).
    """

    pred_error_norms: List[torch.Tensor]
    precision_means: List[torch.Tensor]
    epsilon_final: Optional[torch.Tensor]
    pi_final: Optional[torch.Tensor]
    routing_logits_H: Optional[List[torch.Tensor]] = field(default=None)
    routing_logits_L: Optional[List[torch.Tensor]] = field(default=None)
    moe_recon_loss: Optional[torch.Tensor] = field(default=None)
    moe_lb_loss: Optional[torch.Tensor] = field(default=None)
    moe_passthrough_weight: float = field(default=1.0)


# ---------------------------------------------------------------------------
# CoralV3Inner
# ---------------------------------------------------------------------------


class CoralV3Inner(CoralInner):
    """CORAL inner model — extends CoralInner with Phase 1 and Phase 3b mechanisms.

    When all flags are False the forward pass is identical to CoralInner.
    """

    def __init__(self, config: CoralConfig) -> None:
        super().__init__(config)

        # Bootstrap flag: True until first spatial consolidation.
        # While True, _apply_moe_mixing bypasses the codebook (returns z_L_rec unchanged).
        # Flipped to False by consolidate_codebook() after the first k-means run.
        self._crystal_bootstrap_active: bool = config.crystal_bootstrap_steps > 0

        # --- Phase 1: predictive coding ---
        if config.use_predictive_coding:
            dim = config.hidden_size
            self.prediction_net = PredictionNet(h_dim=dim, l_dim=dim)
            self.precision_net = PrecisionNet(dim=dim)

        # --- Phase 2: sparse columnar routing (STUBBED — instantiation kept for
        #     forward-compatibility with checkpoints that have routing weights) ---
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

        # --- Phase 3b: Soft MoE Crystallization ---
        if config.use_crystallization:
            # total_seq_len includes puzzle-embedding prefix tokens; codebook_values
            # must match the actual z_L sequence dimension during forward.
            self.moe_codebook = SpatialMoECodebook(config, seq_len=self.total_seq_len)
            self.moe_codebook.bootstrap_mask_router(self._crystal_bootstrap_active)
            self.crystal_buffer = CrystallizationBuffer(
                capacity=config.crystal_buffer_capacity,
            )

    # ------------------------------------------------------------------
    # MoE mixing helpers
    # ------------------------------------------------------------------

    def _apply_moe_mixing(
        self,
        z_H: torch.Tensor,
        z_L: torch.Tensor,
        z_L_rec: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Apply Soft MoE blend: z_L_out = w_pt*z_L_rec + (1-w_pt)*z_bypass.

        During bootstrap or when crystallization is disabled, returns z_L_rec
        unchanged (no codebook computation overhead).

        Args:
            z_H:    [B, S, D] — current H-module state.
            z_L:    [B, S, D] — carry z_L from the previous step (NOT z_L_rec).
            z_L_rec:[B, S, D] — raw L-level recurrence output.

        Returns:
            z_L_out: [B, S, D]
            w:       [B, K+1] routing weights, or None if not using codebook.
            z_bypass:[B, S, D] codebook mixture, or None if not using codebook.
        """
        if not self.config.use_crystallization or self._crystal_bootstrap_active:
            return z_L_rec, None, None

        w, z_bypass, _key = self.moe_codebook(z_H, z_L)
        w_pt = w[:, -1]  # [B] passthrough weight
        z_L_out = w_pt[:, None, None] * z_L_rec + (1.0 - w_pt[:, None, None]) * z_bypass
        return z_L_out, w, z_bypass

    def _compute_moe_losses(
        self,
        z_L_final: torch.Tensor,
        w: Optional[torch.Tensor],
        z_bypass: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Delegate to moe_codebook.moe_losses (training only, post-bootstrap)."""
        if not self.config.use_crystallization or not self.training:
            return None, None
        if w is None or z_bypass is None:
            return None, None
        return self.moe_codebook.moe_losses(z_L_final, w, z_bypass)

    @torch.compiler.disable(recursive=False)
    def _maybe_record_crystal(
        self,
        z_H: torch.Tensor,
        z_L: torch.Tensor,
        is_last_h: bool = False,
        is_last_segment: bool = False,
    ) -> None:
        """Add current (z_H, z_L) to the crystal buffer during bootstrap training.

        Only records during bootstrap phase (disabled post-consolidation per spec §3.4).
        Only records on the last H-cycle of the last ACT segment to avoid redundant
        GPU→CPU transfers at every segment.
        """
        if not is_last_h or not is_last_segment:
            return
        if not self.config.use_crystallization or not self.training:
            return
        if not self._crystal_bootstrap_active:
            return  # Buffer disabled post-consolidation

        h_pool = self.moe_codebook.proj_h(z_H).mean(dim=1)   # [B, proj_dim]
        l_pool = self.moe_codebook.proj_l(z_L).mean(dim=1)   # [B, proj_dim]
        key = torch.cat([h_pool, l_pool], dim=-1)             # [B, key_dim]
        pooled_z_L = z_L.mean(dim=1)                          # [B, l_dim]
        self.crystal_buffer.add(key, pooled_z_L, z_L_spatial=z_L)

    def consolidate_codebook(self, is_first_consolidation: bool = False) -> Optional[int]:
        """Run spatial k-means and warm-start codebook_values from the buffer.

        Writes the k_modes cluster centroids (full spatial float tensors) to
        moe_codebook.codebook_values, clears the buffer, and deactivates the
        bootstrap mask so the codebook is live for end-to-end training.

        Args:
            is_first_consolidation: When True, requires ≥80% buffer fill before
                                    proceeding. Returns None to defer if not ready.

        Returns:
            Count of non-empty clusters after k-means, or None if deferred.
        """
        if not self.config.use_crystallization:
            return None

        k_modes = self.config.moe_num_modes

        if is_first_consolidation and self.crystal_buffer.size < int(
            0.8 * self.crystal_buffer.capacity
        ):
            return None

        result = self.crystal_buffer.consolidate_spatial(k_modes)
        if result is None:
            return None

        centroids, utilization = result
        device = self.moe_codebook.codebook_values.device
        dtype = self.moe_codebook.codebook_values.dtype
        self.moe_codebook.codebook_values.data = centroids.to(device=device, dtype=dtype)

        self.crystal_buffer.clear()

        # Deactivate bootstrap mask — router and codebook are now live
        self._crystal_bootstrap_active = False
        self.moe_codebook.bootstrap_mask_router(False)

        return utilization

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

        Returns:
            3-tuple when all mechanisms disabled (same as CoralInner).
            4-tuple otherwise: (new_carry, output, (q_halt, q_continue), pred_metrics)
        """
        pc = self.config.use_predictive_coding
        cr = self.config.use_columnar_routing
        cry = self.config.use_crystallization

        if not pc and not cr and not cry:
            return super().forward(carry, batch)
        elif pc and not cr:
            return self._forward_with_pc(carry, batch, is_last_segment=is_last_segment)
        else:
            raise NotImplementedError(
                "columnar routing disabled pending Phase 2 redesign; "
                "Phase 3b operates on PC-only dispatch (use_predictive_coding=True, "
                "use_columnar_routing=False). cry-only (pc=False, cr=False, cry=True) "
                "is also unsupported — enable use_predictive_coding alongside crystallization."
            )

    # ------------------------------------------------------------------
    # Forward implementation — PC path (only active path in Phase 3b)
    # ------------------------------------------------------------------

    def _forward_with_pc(
        self,
        carry: InnerCarry,
        batch: Dict[str, torch.Tensor],
        is_last_segment: bool = False,
    ) -> Tuple[InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], PredMetrics]:
        """Inner forward with predictive coding and optional Soft MoE crystallization.

        Soft MoE mixing (when use_crystallization=True, post-bootstrap) is applied
        after every L_level call, including the 1-step-grad step. There is no
        is_last_h restriction — soft mixing preserves gradients at every cycle.
        """
        cos_sin = self._cos_sin()
        input_embeddings = self._input_embeddings(
            batch["inputs"],
            batch.get("puzzle_identifiers"),
        )

        pred_error_norms: List[torch.Tensor] = []
        precision_means: List[torch.Tensor] = []

        with torch.no_grad():
            z_H, z_L = carry.z_H, carry.z_L
            xi: Optional[torch.Tensor] = None

            for h_step in range(self.config.H_cycles):
                for l_step in range(self.config.L_cycles):
                    is_last_l = (
                        h_step == self.config.H_cycles - 1
                        and l_step == self.config.L_cycles - 1
                    )
                    if not is_last_l:
                        mu_L = self.prediction_net(z_H)
                        z_L_rec = self.L_level(z_L, mu_L + input_embeddings, cos_sin=cos_sin)
                        z_L, _, _ = self._apply_moe_mixing(z_H, z_L, z_L_rec)
                        epsilon = z_L - mu_L
                        pi = self.precision_net(z_L)
                        xi = pi * epsilon
                        pred_error_norms.append(epsilon.norm(dim=-1).mean())
                        precision_means.append(pi.mean())

                if not (h_step == self.config.H_cycles - 1):
                    z_H = self.H_level(z_H, xi, cos_sin=cos_sin)  # type: ignore[arg-type]
                    self._maybe_record_crystal(
                        z_H, z_L,
                        is_last_h=(h_step == self.config.H_cycles - 2),
                        is_last_segment=is_last_segment,
                    )

        assert not z_H.requires_grad and not z_L.requires_grad

        # 1-step gradient — builds the computation graph for backward
        mu_L = self.prediction_net(z_H)
        z_L_rec = self.L_level(z_L, mu_L + input_embeddings, cos_sin=cos_sin)
        z_L, w, z_bypass = self._apply_moe_mixing(z_H, z_L, z_L_rec)

        epsilon_final = z_L - mu_L
        pi_final = self.precision_net(z_L)
        xi = pi_final * epsilon_final

        moe_recon_loss, moe_lb_loss = self._compute_moe_losses(z_L, w, z_bypass)

        z_H = self.H_level(z_H, xi, cos_sin=cos_sin)

        pred_error_norms.append(epsilon_final.detach().norm(dim=-1).mean())
        precision_means.append(pi_final.detach().mean())

        moe_pt_weight: float = 1.0
        if w is not None:
            moe_pt_weight = float(w[:, -1].mean().item())

        new_carry = InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)

        pred_metrics = PredMetrics(
            pred_error_norms=pred_error_norms,
            precision_means=precision_means,
            epsilon_final=epsilon_final,
            pi_final=pi_final,
            moe_recon_loss=moe_recon_loss,
            moe_lb_loss=moe_lb_loss,
            moe_passthrough_weight=moe_pt_weight,
        )
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1]), pred_metrics
