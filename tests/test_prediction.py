"""Tests for Phase 1: PredictionNet, PrecisionNet, predictive_coding_loss, CoralV3Inner.

Structural and shape tests run on CPU.
CUDA-only tests require CUDA + flash_attn and are skipped when unavailable.
"""

import pytest
import torch
import torch.nn as nn

from coral.models.prediction import PredictionNet, PrecisionNet
from coral.models.coral_base import CoralConfig, InnerCarry
from coral.models.coral_v3 import CoralV3Inner
from coral.training.losses import predictive_coding_loss


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

HIDDEN = 64
NUM_HEADS = 4
VOCAB = 32
SEQ_LEN = 8
BATCH = 2

SMALL_CFG = dict(
    batch_size=BATCH,
    seq_len=SEQ_LEN,
    vocab_size=VOCAB,
    H_cycles=2,
    L_cycles=2,
    H_layers=2,
    L_layers=2,
    hidden_size=HIDDEN,
    num_heads=NUM_HEADS,
    expansion=4.0,
    rms_norm_eps=1e-5,
    halt_max_steps=4,
    halt_exploration_prob=0.1,
    forward_dtype="float32",  # float32 so CPU tests work
    puzzle_emb_ndim=0,
    num_puzzle_identifiers=0,
)

CUDA_ONLY = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Requires CUDA + flash_attn",
)


def make_v3_model(pc: bool = True, cfg_overrides=None) -> CoralV3Inner:
    cfg = dict(SMALL_CFG)
    cfg["use_predictive_coding"] = pc
    if cfg_overrides:
        cfg.update(cfg_overrides)
    return CoralV3Inner(CoralConfig(**cfg))


def make_batch(B=BATCH, seq=SEQ_LEN, vocab=VOCAB, device="cpu"):
    return {"inputs": torch.randint(0, vocab, (B, seq), device=device)}


def make_carry(model: CoralV3Inner, device="cpu") -> InnerCarry:
    carry = model.empty_carry(BATCH, device=torch.device(device))
    reset_all = torch.ones(BATCH, dtype=torch.bool, device=device)
    return model.reset_carry(reset_all, carry)


# ---------------------------------------------------------------------------
# PredictionNet
# ---------------------------------------------------------------------------


class TestPredictionNet:
    def test_output_shape_equal_dims(self):
        """Output should be [B, seq, l_dim] when h_dim == l_dim."""
        net = PredictionNet(h_dim=HIDDEN, l_dim=HIDDEN)
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN)
        out = net(x)
        assert out.shape == (BATCH, SEQ_LEN, HIDDEN)

    def test_output_shape_different_dims(self):
        """PredictionNet works with h_dim != l_dim (required for future N=3)."""
        h_dim, l_dim = 128, 64
        net = PredictionNet(h_dim=h_dim, l_dim=l_dim)
        x = torch.randn(BATCH, SEQ_LEN, h_dim)
        out = net(x)
        assert out.shape == (BATCH, SEQ_LEN, l_dim)

    def test_no_bias(self):
        """Both linear layers should have no bias (CastedLinear bias=False)."""
        net = PredictionNet(h_dim=HIDDEN, l_dim=HIDDEN)
        assert net.fc1.bias is None
        assert net.fc2.bias is None

    def test_intermediate_dim_is_l_dim_times_2(self):
        """fc1 output features should be l_dim * 2."""
        net = PredictionNet(h_dim=HIDDEN, l_dim=HIDDEN)
        assert net.fc1.weight.shape == (HIDDEN * 2, HIDDEN)
        assert net.fc2.weight.shape == (HIDDEN, HIDDEN * 2)

    def test_differentiable(self):
        """Output should have a grad_fn (in-graph)."""
        net = PredictionNet(h_dim=HIDDEN, l_dim=HIDDEN)
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN)
        out = net(x)
        assert out.grad_fn is not None


# ---------------------------------------------------------------------------
# PrecisionNet
# ---------------------------------------------------------------------------


class TestPrecisionNet:
    def test_output_shape(self):
        """Output shape must match input shape."""
        net = PrecisionNet(dim=HIDDEN)
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN)
        out = net(x)
        assert out.shape == (BATCH, SEQ_LEN, HIDDEN)

    def test_output_always_positive(self):
        """Every element of pi must be > 0 (softplus + eps_min ensures this)."""
        net = PrecisionNet(dim=HIDDEN)
        # Try with random, near-zero, and extreme inputs
        for x in [
            torch.randn(BATCH, SEQ_LEN, HIDDEN),
            torch.zeros(BATCH, SEQ_LEN, HIDDEN),
            torch.full((BATCH, SEQ_LEN, HIDDEN), -100.0),
            torch.full((BATCH, SEQ_LEN, HIDDEN), 100.0),
        ]:
            out = net(x)
            assert (out > 0).all(), f"Precision not positive for input with mean={x.mean():.1f}"

    def test_output_above_eps_min(self):
        """All values should be >= EPS_MIN (0.01)."""
        net = PrecisionNet(dim=HIDDEN)
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN)
        out = net(x)
        assert (out >= PrecisionNet.EPS_MIN).all()

    def test_no_bias(self):
        net = PrecisionNet(dim=HIDDEN)
        assert net.fc1.bias is None
        assert net.fc2.bias is None

    def test_differentiable(self):
        net = PrecisionNet(dim=HIDDEN)
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN)
        out = net(x)
        assert out.grad_fn is not None


# ---------------------------------------------------------------------------
# predictive_coding_loss
# ---------------------------------------------------------------------------


class TestPredictiveCodingLoss:
    def test_returns_two_scalar_tensors(self):
        epsilon = torch.randn(BATCH, SEQ_LEN, HIDDEN)
        pi = torch.rand(BATCH, SEQ_LEN, HIDDEN) + 0.01  # ensure positive
        pred_loss, pi_reg = predictive_coding_loss(epsilon, pi)
        assert pred_loss.shape == ()
        assert pi_reg.shape == ()

    def test_pred_loss_is_non_negative(self):
        """Precision-weighted squared error is always non-negative."""
        epsilon = torch.randn(BATCH, SEQ_LEN, HIDDEN)
        pi = torch.rand(BATCH, SEQ_LEN, HIDDEN) + 0.01
        pred_loss, _ = predictive_coding_loss(epsilon, pi)
        assert pred_loss.item() >= 0

    def test_lambda_scaling(self):
        """Doubling lambda_pred should double pred_loss."""
        epsilon = torch.randn(BATCH, SEQ_LEN, HIDDEN)
        pi = torch.rand(BATCH, SEQ_LEN, HIDDEN) + 0.01
        loss1, _ = predictive_coding_loss(epsilon, pi, lambda_pred=0.1)
        loss2, _ = predictive_coding_loss(epsilon, pi, lambda_pred=0.2)
        assert torch.isclose(loss2, 2 * loss1, rtol=1e-5)

    def test_zero_epsilon_gives_zero_pred_loss(self):
        """When prediction is perfect (ε=0), pred_loss should be 0."""
        epsilon = torch.zeros(BATCH, SEQ_LEN, HIDDEN)
        pi = torch.rand(BATCH, SEQ_LEN, HIDDEN) + 0.01
        pred_loss, _ = predictive_coding_loss(epsilon, pi)
        assert torch.isclose(pred_loss, torch.tensor(0.0), atol=1e-6)

    def test_both_terms_differentiable(self):
        """Both loss terms should support backward."""
        epsilon = torch.randn(BATCH, SEQ_LEN, HIDDEN, requires_grad=True)
        pi = (torch.rand(BATCH, SEQ_LEN, HIDDEN) + 0.01).requires_grad_(True)
        pred_loss, pi_reg = predictive_coding_loss(epsilon, pi)
        (pred_loss + pi_reg).backward()
        assert epsilon.grad is not None
        assert pi.grad is not None


# ---------------------------------------------------------------------------
# CoralV3Inner — structural tests (CPU, no forward pass)
# ---------------------------------------------------------------------------


class TestCoralV3InnerStructure:
    def test_prediction_net_created_when_pc_enabled(self):
        model = make_v3_model(pc=True)
        assert hasattr(model, "prediction_net")
        assert isinstance(model.prediction_net, PredictionNet)

    def test_precision_net_created_when_pc_enabled(self):
        model = make_v3_model(pc=True)
        assert hasattr(model, "precision_net")
        assert isinstance(model.precision_net, PrecisionNet)

    def test_no_prediction_net_when_pc_disabled(self):
        model = make_v3_model(pc=False)
        assert not hasattr(model, "prediction_net")

    def test_no_precision_net_when_pc_disabled(self):
        model = make_v3_model(pc=False)
        assert not hasattr(model, "precision_net")

    def test_extra_params_only_when_pc_enabled(self):
        """PC-enabled model should have more parameters than the baseline."""
        baseline = make_v3_model(pc=False)
        pc_model = make_v3_model(pc=True)
        n_baseline = sum(p.numel() for p in baseline.parameters())
        n_pc = sum(p.numel() for p in pc_model.parameters())
        assert n_pc > n_baseline

    def test_h_init_still_buffer(self):
        model = make_v3_model(pc=True)
        buffer_names = {n for n, _ in model.named_buffers()}
        assert "H_init" in buffer_names

    def test_prediction_net_dim_matches_hidden(self):
        model = make_v3_model(pc=True)
        # fc1 maps hidden → hidden*2
        assert model.prediction_net.fc1.weight.shape[1] == HIDDEN
        assert model.prediction_net.fc1.weight.shape[0] == HIDDEN * 2


# ---------------------------------------------------------------------------
# CoralV3Inner — forward pass tests (CUDA required)
# ---------------------------------------------------------------------------


@CUDA_ONLY
class TestCoralV3InnerForward:
    def _make_model(self, pc: bool):
        cfg = dict(SMALL_CFG)
        cfg["forward_dtype"] = "bfloat16"
        cfg["use_predictive_coding"] = pc
        return CoralV3Inner(CoralConfig(**cfg)).cuda()

    def _make_carry(self, model):
        carry = model.empty_carry(BATCH, device=torch.device("cuda"))
        reset_all = torch.ones(BATCH, dtype=torch.bool, device="cuda")
        return model.reset_carry(reset_all, carry)

    def _make_batch(self):
        return make_batch(device="cuda")

    # --- pc=False: shapes must match CoralInner ---

    def test_pc_false_returns_3_tuple(self):
        model = self._make_model(pc=False)
        carry = self._make_carry(model)
        result = model(carry, self._make_batch())
        assert len(result) == 3

    def test_pc_false_output_shape(self):
        model = self._make_model(pc=False)
        carry = self._make_carry(model)
        _, output, _ = model(carry, self._make_batch())
        assert output.shape == (BATCH, SEQ_LEN, VOCAB)

    def test_pc_false_q_shapes(self):
        model = self._make_model(pc=False)
        carry = self._make_carry(model)
        _, _, (q_halt, q_cont) = model(carry, self._make_batch())
        assert q_halt.shape == (BATCH,)
        assert q_cont.shape == (BATCH,)

    def test_pc_false_carry_detached(self):
        model = self._make_model(pc=False)
        carry = self._make_carry(model)
        new_carry, _, _ = model(carry, self._make_batch())
        assert new_carry.z_H.grad_fn is None
        assert new_carry.z_L.grad_fn is None

    # --- pc=True: correct output shapes and return structure ---

    def test_pc_true_returns_4_tuple(self):
        model = self._make_model(pc=True)
        carry = self._make_carry(model)
        result = model(carry, self._make_batch())
        assert len(result) == 4

    def test_pc_true_output_shape(self):
        model = self._make_model(pc=True)
        carry = self._make_carry(model)
        _, output, _, _ = model(carry, self._make_batch())
        assert output.shape == (BATCH, SEQ_LEN, VOCAB)

    def test_pc_true_q_shapes(self):
        model = self._make_model(pc=True)
        carry = self._make_carry(model)
        _, _, (q_halt, q_cont), _ = model(carry, self._make_batch())
        assert q_halt.shape == (BATCH,)
        assert q_cont.shape == (BATCH,)

    def test_pc_true_carry_detached(self):
        model = self._make_model(pc=True)
        carry = self._make_carry(model)
        new_carry, _, _, _ = model(carry, self._make_batch())
        assert new_carry.z_H.grad_fn is None
        assert new_carry.z_L.grad_fn is None

    def test_pc_true_epsilon_shape(self):
        model = self._make_model(pc=True)
        carry = self._make_carry(model)
        _, _, _, pred_metrics = model(carry, self._make_batch())
        assert pred_metrics.epsilon_final is not None
        assert pred_metrics.epsilon_final.shape == (BATCH, model.total_seq_len, HIDDEN)

    def test_pc_true_pi_shape(self):
        model = self._make_model(pc=True)
        carry = self._make_carry(model)
        _, _, _, pred_metrics = model(carry, self._make_batch())
        assert pred_metrics.pi_final is not None
        assert pred_metrics.pi_final.shape == (BATCH, model.total_seq_len, HIDDEN)

    def test_pc_true_pi_positive(self):
        """pi_final must be strictly positive at inference time."""
        model = self._make_model(pc=True)
        carry = self._make_carry(model)
        _, _, _, pred_metrics = model(carry, self._make_batch())
        assert (pred_metrics.pi_final > 0).all()

    def test_pred_error_is_z_L_minus_mu_L(self):
        """epsilon_final should have a grad_fn — it's z_L - mu_L, both in graph."""
        model = self._make_model(pc=True)
        carry = self._make_carry(model)
        _, _, _, pred_metrics = model(carry, self._make_batch())
        assert pred_metrics.epsilon_final.grad_fn is not None

    def test_pred_error_norms_count(self):
        """There should be H_cycles*L_cycles accumulated norms (all steps including final)."""
        cfg = dict(SMALL_CFG)
        cfg["forward_dtype"] = "bfloat16"
        cfg["use_predictive_coding"] = True
        cfg["H_cycles"] = 2
        cfg["L_cycles"] = 2
        model = CoralV3Inner(CoralConfig(**cfg)).cuda()
        carry = self._make_carry(model)
        _, _, _, pred_metrics = model(carry, self._make_batch())
        # (H*L - 1) no-grad steps + 1 final step = H*L total
        expected = cfg["H_cycles"] * cfg["L_cycles"]
        assert len(pred_metrics.pred_error_norms) == expected

    def test_full_backward_with_pc(self):
        """Full forward + backward should complete without error."""
        model = self._make_model(pc=True)
        carry = self._make_carry(model)
        _, output, (q_halt, q_cont), pred_metrics = model(carry, self._make_batch())
        pred_loss, pi_reg = predictive_coding_loss(
            pred_metrics.epsilon_final,
            pred_metrics.pi_final,
        )
        loss = output.float().sum() + pred_loss + pi_reg
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_prediction_net_gets_grad_in_backward(self):
        """prediction_net and precision_net parameters should receive gradients."""
        model = self._make_model(pc=True)
        carry = self._make_carry(model)
        _, output, _, pred_metrics = model(carry, self._make_batch())
        pred_loss, pi_reg = predictive_coding_loss(
            pred_metrics.epsilon_final,
            pred_metrics.pi_final,
        )
        (output.float().sum() + pred_loss + pi_reg).backward()
        assert model.prediction_net.fc1.weight.grad is not None
        assert model.precision_net.fc1.weight.grad is not None
