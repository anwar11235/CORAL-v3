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

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from coral.models.coral_base import CoralConfig, CoralInner, InnerCarry
from coral.models.columnar import ColumnarReasoningModule
from coral.models.prediction import PredictionNet, PrecisionNet
from coral.models.transformer_block import TransformerBlockConfig


# ---------------------------------------------------------------------------
# Prediction / routing metrics carrier
# ---------------------------------------------------------------------------


@dataclass
class PredMetrics:
    """Statistics collected during one inner forward pass.

    pred_error_norms and precision_means are detached scalars from every
    recurrent step (useful for logging step-level dynamics).

    epsilon_final and pi_final are *in-graph* tensors from the 1-step-grad
    step; used by CoralV3LossHead for the free energy loss.
    When use_predictive_coding=False both are None.

    routing_logits_H and routing_logits_L are *in-graph* tensors from the
    1-step-grad step; used by CoralV3LossHead for the load-balancing loss.
    When use_columnar_routing=False both are None.
    """

    pred_error_norms: List[torch.Tensor]         # one scalar per recurrent step (detached)
    precision_means: List[torch.Tensor]          # one scalar per recurrent step (detached)
    epsilon_final: Optional[torch.Tensor]        # [B, seq, dim] — with grad
    pi_final: Optional[torch.Tensor]             # [B, seq, dim] — with grad
    routing_logits_H: Optional[List[torch.Tensor]] = field(default=None)  # list of [B, S] per H-layer
    routing_logits_L: Optional[List[torch.Tensor]] = field(default=None)  # list of [B, S] per L-layer


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
        if config.use_columnar_routing:
            # Replace the monolithic ReasoningModules created by CoralInner.__init__
            # with columnar equivalents.  block_cfg must match what CoralInner used.
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

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        carry: InnerCarry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple:
        """Run one segment with optional predictive coding and/or columnar routing.

        Args:
            carry: Previous segment's detached carry (z_H, z_L).
            batch: Dict with at minimum "inputs" [B, seq_len] int tokens.

        Returns:
            When use_predictive_coding=False and use_columnar_routing=False:
                (new_carry, output, (q_halt, q_continue))
            Otherwise:
                (new_carry, output, (q_halt, q_continue), pred_metrics)
        """
        pc = self.config.use_predictive_coding
        cr = self.config.use_columnar_routing

        if pc and cr:
            return self._forward_with_pc_and_routing(carry, batch)
        elif pc:
            return self._forward_with_pc(carry, batch)
        elif cr:
            return self._forward_with_routing(carry, batch)
        else:
            return super().forward(carry, batch)

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

    def _forward_with_routing(
        self,
        carry: InnerCarry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], PredMetrics]:
        """Inner forward with columnar routing active (no predictive coding)."""
        cos_sin = self._cos_sin()
        input_embeddings = self._input_embeddings(
            batch["inputs"],
            batch.get("puzzle_identifiers"),
        )

        # ------------------------------------------------------------------
        # No-grad recurrent steps — routing logits are discarded here.
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
                        z_L, _ = self.L_level(z_L, z_H + input_embeddings, cos_sin=cos_sin)

                if not (h_step == self.config.H_cycles - 1):
                    z_H, _ = self.H_level(z_H, z_L, cos_sin=cos_sin)

        assert not z_H.requires_grad and not z_L.requires_grad

        # ------------------------------------------------------------------
        # 1-step gradient — collect routing logits for the loss.
        # ------------------------------------------------------------------
        z_L, routing_logits_L = self.L_level(z_L, z_H + input_embeddings, cos_sin=cos_sin)
        z_H, routing_logits_H = self.H_level(z_H, z_L, cos_sin=cos_sin)

        # ------------------------------------------------------------------
        # Outputs
        # ------------------------------------------------------------------
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
        )

        return new_carry, output, (q_logits[..., 0], q_logits[..., 1]), pred_metrics

    def _forward_with_pc_and_routing(
        self,
        carry: InnerCarry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], PredMetrics]:
        """Inner forward with both predictive coding and columnar routing active."""
        cos_sin = self._cos_sin()
        input_embeddings = self._input_embeddings(
            batch["inputs"],
            batch.get("puzzle_identifiers"),
        )

        pred_error_norms: List[torch.Tensor] = []
        precision_means: List[torch.Tensor] = []

        # ------------------------------------------------------------------
        # No-grad recurrent steps — routing logits discarded.
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
                        z_L, _ = self.L_level(z_L, mu_L + input_embeddings, cos_sin=cos_sin)
                        epsilon = z_L - mu_L
                        pi = self.precision_net(z_L)
                        xi = pi * epsilon
                        pred_error_norms.append(epsilon.norm(dim=-1).mean())
                        precision_means.append(pi.mean())

                if not (h_step == self.config.H_cycles - 1):
                    z_H, _ = self.H_level(z_H, xi, cos_sin=cos_sin)  # type: ignore[possibly-undefined]

        assert not z_H.requires_grad and not z_L.requires_grad

        # ------------------------------------------------------------------
        # 1-step gradient — collect both PC tensors and routing logits.
        # ------------------------------------------------------------------
        mu_L = self.prediction_net(z_H)
        z_L, routing_logits_L = self.L_level(z_L, mu_L + input_embeddings, cos_sin=cos_sin)
        epsilon_final = z_L - mu_L
        pi_final = self.precision_net(z_L)
        xi = pi_final * epsilon_final
        z_H, routing_logits_H = self.H_level(z_H, xi, cos_sin=cos_sin)

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
            routing_logits_H=routing_logits_H,
            routing_logits_L=routing_logits_L,
        )

        return new_carry, output, (q_logits[..., 0], q_logits[..., 1]), pred_metrics
