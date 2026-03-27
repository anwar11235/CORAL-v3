"""CoralV3Inner — CORAL base with optional precision-weighted predictive coding.

This module extends CoralInner (Phase 0) with the Phase 1 mechanism:
precision-weighted predictive coding between the H and L modules.

Behaviour is controlled by CoralConfig.use_predictive_coding:
  False — identical to CoralInner (HRM-equivalent baseline).
  True  — H predicts L's state; L receives that prediction; H receives the
          precision-weighted prediction error.

The information-flow change (when use_predictive_coding=True):

    # Baseline (CoralInner):
    z_L = L_level(z_L, z_H + input_embeddings)
    z_H = H_level(z_H, z_L)

    # CoralV3Inner with predictive coding:
    mu_L = prediction_net(z_H)                     # H's prediction of L's state
    z_L  = L_level(z_L, mu_L + input_embeddings)   # L receives prediction, not raw H
    ε    = z_L - mu_L                               # prediction error
    π    = precision_net(z_L)                       # learned per-dimension precision
    ξ    = π * ε                                    # precision-weighted error
    z_H  = H_level(z_H, ξ)                          # H receives error signal, not raw L

This applies at every recurrent step (both no_grad and 1-step-grad steps).
The prediction loss terms (used by CoralV3LossHead) come only from the final
1-step-grad step, because that is the only pass with a live computation graph.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from coral.models.coral_base import CoralConfig, CoralInner, InnerCarry
from coral.models.prediction import PredictionNet, PrecisionNet


# ---------------------------------------------------------------------------
# Prediction metrics carrier
# ---------------------------------------------------------------------------


@dataclass
class PredMetrics:
    """Prediction-coding statistics collected during one inner forward pass.

    pred_error_norms and precision_means are detached scalars from every
    recurrent step (useful for logging step-level dynamics).

    epsilon_final and pi_final are *in-graph* tensors from the 1-step-grad
    step; they are used by CoralV3LossHead to compute the free energy loss.
    When use_predictive_coding=False both are None.
    """

    pred_error_norms: List[torch.Tensor]   # one scalar per recurrent step (detached)
    precision_means: List[torch.Tensor]    # one scalar per recurrent step (detached)
    epsilon_final: Optional[torch.Tensor]  # [B, seq, dim] — with grad
    pi_final: Optional[torch.Tensor]       # [B, seq, dim] — with grad


# ---------------------------------------------------------------------------
# CoralV3Inner
# ---------------------------------------------------------------------------


class CoralV3Inner(CoralInner):
    """CORAL inner model with optional precision-weighted predictive coding.

    Drop-in replacement for CoralInner:
      - When use_predictive_coding=False the forward pass is identical to
        CoralInner and returns the same 3-tuple.
      - When use_predictive_coding=True the inter-level injection uses the
        predictive coding mechanism described above and returns a 4-tuple
        with an additional PredMetrics.

    prediction_net and precision_net are only created when
    use_predictive_coding=True, so the parameter count is unchanged in
    baseline mode.
    """

    def __init__(self, config: CoralConfig) -> None:
        super().__init__(config)
        if config.use_predictive_coding:
            dim = config.hidden_size
            self.prediction_net = PredictionNet(h_dim=dim, l_dim=dim)
            self.precision_net = PrecisionNet(dim=dim)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        carry: InnerCarry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple:
        """Run one segment with optional predictive coding.

        Args:
            carry: Previous segment's detached carry (z_H, z_L).
            batch: Dict with at minimum "inputs" [B, seq_len] int tokens.

        Returns:
            When use_predictive_coding=False (same as CoralInner):
                (new_carry, output, (q_halt, q_continue))

            When use_predictive_coding=True:
                (new_carry, output, (q_halt, q_continue), pred_metrics)
        """
        if not self.config.use_predictive_coding:
            return super().forward(carry, batch)

        return self._forward_with_pc(carry, batch)

    def _forward_with_pc(
        self,
        carry: InnerCarry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], PredMetrics]:
        """Inner forward with predictive coding active."""
        cos_sin = self._cos_sin()
        input_embeddings = self._input_embeddings(
            batch["inputs"],
            batch.get("puzzle_identifiers"),
        )

        pred_error_norms: List[torch.Tensor] = []
        precision_means: List[torch.Tensor] = []

        # ------------------------------------------------------------------
        # No-grad recurrent steps (all except the final L and H steps).
        # xi is the precision-weighted error passed into H_level.
        # ------------------------------------------------------------------
        with torch.no_grad():
            z_H, z_L = carry.z_H, carry.z_L

            for h_step in range(self.config.H_cycles):
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

                # xi here is from the last executed l_step of this h_step,
                # which is always defined whenever the H update below runs.
                if not (h_step == self.config.H_cycles - 1):
                    z_H = self.H_level(z_H, xi, cos_sin=cos_sin)  # type: ignore[possibly-undefined]

        assert not z_H.requires_grad and not z_L.requires_grad

        # ------------------------------------------------------------------
        # 1-step gradient — only these ops are in the computation graph.
        # ------------------------------------------------------------------
        mu_L = self.prediction_net(z_H)
        z_L = self.L_level(z_L, mu_L + input_embeddings, cos_sin=cos_sin)
        epsilon_final = z_L - mu_L
        pi_final = self.precision_net(z_L)
        xi = pi_final * epsilon_final
        z_H = self.H_level(z_H, xi, cos_sin=cos_sin)

        # Accumulate final-step metrics (detached — for logging only)
        pred_error_norms.append(epsilon_final.detach().norm(dim=-1).mean())
        precision_means.append(pi_final.detach().mean())

        # ------------------------------------------------------------------
        # Outputs
        # ------------------------------------------------------------------
        new_carry = InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)

        pred_metrics = PredMetrics(
            pred_error_norms=pred_error_norms,
            precision_means=precision_means,
            epsilon_final=epsilon_final,
            pi_final=pi_final,
        )

        return new_carry, output, (q_logits[..., 0], q_logits[..., 1]), pred_metrics
