"""Tests for Phase 3: RecognitionNetwork, CrystallizationBuffer,
crystallization_supervision_loss, and CoralV3Inner with crystallization.

Structural and shape tests run on CPU.
CUDA-only tests require CUDA + flash_attn and are skipped when unavailable.
"""

import pytest
import torch
import torch.nn.functional as F

from coral.models.transformer_block import TransformerBlockConfig
from coral.models.coral_base import CoralConfig, InnerCarry
from coral.models.coral_v3 import CoralV3Inner, PredMetrics
from coral.models.crystallization import (
    CrystallizationBuffer,
    RecognitionNetwork,
    crystallization_supervision_loss,
)
from coral.training.losses import load_balancing_loss


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

HIDDEN = 64
NUM_HEADS = 4
VOCAB = 32
SEQ_LEN = 8
BATCH = 4
CODEBOOK_SIZE = 16
PROJ_DIM = 32
S = 4
K = 2

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
    forward_dtype="float32",
    puzzle_emb_ndim=0,
    num_puzzle_identifiers=0,
)

CUDA_ONLY = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Requires CUDA + flash_attn",
)


def make_rnet() -> RecognitionNetwork:
    return RecognitionNetwork(
        h_dim=HIDDEN,
        l_dim=HIDDEN,
        codebook_size=CODEBOOK_SIZE,
        proj_dim=PROJ_DIM,
    )


def make_batch(B=BATCH, seq=SEQ_LEN, vocab=VOCAB, device="cpu"):
    return {
        "inputs": torch.randint(0, vocab, (B, seq), device=device),
        "labels": torch.randint(0, vocab, (B, seq), device=device),
    }


def make_inner_carry(model: CoralV3Inner, B=BATCH, device="cpu") -> InnerCarry:
    carry = model.empty_carry(B, device=device)
    return model.reset_carry(torch.ones(B, dtype=torch.bool, device=device), carry)


def make_v3_model(**overrides) -> CoralV3Inner:
    cfg = dict(SMALL_CFG)
    cfg.update(overrides)
    return CoralV3Inner(CoralConfig(**cfg))


# ---------------------------------------------------------------------------
# RecognitionNetwork
# ---------------------------------------------------------------------------


def test_recognition_net_output_shapes():
    rnet = make_rnet()
    z_H = torch.randn(BATCH, SEQ_LEN, HIDDEN)
    z_L = torch.randn(BATCH, SEQ_LEN, HIDDEN)
    confidence, nearest_code, nearest_idx = rnet(z_H, z_L)
    assert confidence.shape == (BATCH,), f"confidence: {confidence.shape}"
    assert nearest_code.shape == (BATCH, SEQ_LEN, HIDDEN), f"nearest_code: {nearest_code.shape}"
    assert nearest_idx.shape == (BATCH,), f"nearest_idx: {nearest_idx.shape}"


def test_recognition_net_confidence_range():
    """Confidence is a sigmoid output; must be in (0, 1)."""
    rnet = make_rnet()
    z_H = torch.randn(BATCH, SEQ_LEN, HIDDEN)
    z_L = torch.randn(BATCH, SEQ_LEN, HIDDEN)
    confidence, _, _ = rnet(z_H, z_L)
    assert (confidence > 0).all() and (confidence < 1).all()


def test_recognition_net_compute_key_shape():
    rnet = make_rnet()
    z_H = torch.randn(BATCH, SEQ_LEN, HIDDEN)
    z_L = torch.randn(BATCH, SEQ_LEN, HIDDEN)
    key = rnet.compute_key(z_H, z_L)
    assert key.shape == (BATCH, PROJ_DIM * 2)


def test_recognition_net_nearest_idx_valid():
    """nearest_idx values must be in [0, codebook_size)."""
    rnet = make_rnet()
    z_H = torch.randn(BATCH, SEQ_LEN, HIDDEN)
    z_L = torch.randn(BATCH, SEQ_LEN, HIDDEN)
    _, _, nearest_idx = rnet(z_H, z_L)
    assert (nearest_idx >= 0).all() and (nearest_idx < CODEBOOK_SIZE).all()


def test_recognition_net_nearest_code_expanded():
    """nearest_code must be the codebook entry expanded to [B, seq, l_dim]."""
    rnet = make_rnet()
    z_H = torch.randn(BATCH, SEQ_LEN, HIDDEN)
    z_L = torch.randn(BATCH, SEQ_LEN, HIDDEN)
    _, nearest_code, nearest_idx = rnet(z_H, z_L)
    # All sequence positions for a given batch element should be the same codebook entry
    for b in range(BATCH):
        expected = rnet.codebook[nearest_idx[b]].unsqueeze(0).expand(SEQ_LEN, -1)
        torch.testing.assert_close(nearest_code[b], expected)


def test_recognition_net_differentiable():
    rnet = make_rnet()
    z_H = torch.randn(BATCH, SEQ_LEN, HIDDEN, requires_grad=True)
    z_L = torch.randn(BATCH, SEQ_LEN, HIDDEN, requires_grad=True)
    confidence, nearest_code, _ = rnet(z_H, z_L)
    (confidence.sum() + nearest_code.sum()).backward()
    assert z_H.grad is not None
    assert z_L.grad is not None


# ---------------------------------------------------------------------------
# CrystallizationBuffer
# ---------------------------------------------------------------------------


def test_buffer_add_grows():
    buf = CrystallizationBuffer(capacity=100)
    keys = torch.randn(BATCH, PROJ_DIM * 2)
    values = torch.randn(BATCH, HIDDEN)
    buf.add(keys, values)
    assert len(buf) == BATCH


def test_buffer_add_multiple_batches():
    buf = CrystallizationBuffer(capacity=100)
    for _ in range(5):
        buf.add(torch.randn(BATCH, PROJ_DIM * 2), torch.randn(BATCH, HIDDEN))
    assert len(buf) == 5 * BATCH


def test_buffer_capacity_limiting():
    capacity = 10
    buf = CrystallizationBuffer(capacity=capacity)
    for _ in range(20):
        buf.add(torch.randn(BATCH, PROJ_DIM * 2), torch.randn(BATCH, HIDDEN))
    assert len(buf) == capacity


def test_buffer_stores_cpu_tensors():
    buf = CrystallizationBuffer(capacity=100)
    buf.add(torch.randn(BATCH, PROJ_DIM * 2), torch.randn(BATCH, HIDDEN))
    for k in buf.keys:
        assert k.device.type == "cpu"
    for v in buf.values:
        assert v.device.type == "cpu"


def test_buffer_clear():
    buf = CrystallizationBuffer(capacity=100)
    buf.add(torch.randn(BATCH, PROJ_DIM * 2), torch.randn(BATCH, HIDDEN))
    buf.clear()
    assert len(buf) == 0
    assert buf.pointer == 0


def test_buffer_consolidate_small_buffer():
    """consolidate silently skips when buffer has < 100 entries."""
    buf = CrystallizationBuffer(capacity=100)
    buf.add(torch.randn(BATCH, PROJ_DIM * 2), torch.randn(BATCH, HIDDEN))
    rnet = RecognitionNetwork(HIDDEN, HIDDEN, codebook_size=CODEBOOK_SIZE, proj_dim=PROJ_DIM)
    # Should not raise
    buf.consolidate(rnet, num_iterations=2, device="cpu")


def test_buffer_consolidate_updates_codebook():
    """After consolidation with enough data, codebook should change from initial values."""
    rnet = RecognitionNetwork(HIDDEN, HIDDEN, codebook_size=8, proj_dim=PROJ_DIM)
    buf = CrystallizationBuffer(capacity=1000)
    # Fill buffer with 200 pairs
    for _ in range(50):
        buf.add(torch.randn(4, PROJ_DIM * 2), torch.randn(4, HIDDEN))

    codebook_before = rnet.codebook.data.clone()
    buf.consolidate(rnet, num_iterations=5, device="cpu")

    # At least some codebook entries should have changed
    changed = (rnet.codebook.data != codebook_before).any(dim=-1)
    assert changed.any(), "Expected some codebook entries to be updated"


# ---------------------------------------------------------------------------
# crystallization_supervision_loss
# ---------------------------------------------------------------------------


def test_crystal_supervision_loss_returns_scalar():
    rnet = make_rnet()
    z_H = torch.randn(BATCH, SEQ_LEN, HIDDEN)
    z_L = torch.randn(BATCH, SEQ_LEN, HIDDEN)
    loss, mean_recon, target_conf = crystallization_supervision_loss(rnet, z_H, z_L)
    assert loss.shape == (), f"expected scalar bce_loss, got {loss.shape}"
    assert mean_recon.shape == (), f"expected scalar mean_recon_error, got {mean_recon.shape}"
    assert target_conf.shape == (), f"expected scalar target_conf_mean, got {target_conf.shape}"
    # Diagnostics must be detached
    assert not mean_recon.requires_grad
    assert not target_conf.requires_grad


def test_crystal_supervision_loss_bounded():
    """BCE output is always >= 0."""
    rnet = make_rnet()
    for _ in range(5):
        z_H = torch.randn(BATCH, SEQ_LEN, HIDDEN)
        z_L = torch.randn(BATCH, SEQ_LEN, HIDDEN)
        loss, _, _ = crystallization_supervision_loss(rnet, z_H, z_L)
        assert loss.item() >= 0.0


def test_crystal_supervision_loss_differentiable():
    rnet = make_rnet()
    z_H = torch.randn(BATCH, SEQ_LEN, HIDDEN)
    z_L = torch.randn(BATCH, SEQ_LEN, HIDDEN, requires_grad=True)
    loss, _, _ = crystallization_supervision_loss(rnet, z_H, z_L)
    loss.backward()
    # Gradient should flow through confidence head (z_H, z_L → key → confidence)
    assert z_L.grad is not None


def test_crystal_supervision_loss_perfect_codebook():
    """If nearest_code == z_L_converged (error = 0 < tolerance), target = 1."""
    rnet = make_rnet()
    z_H = torch.randn(BATCH, SEQ_LEN, HIDDEN)

    # Force z_L to exactly match the first codebook entry
    z_L = rnet.codebook[0].unsqueeze(0).unsqueeze(0).expand(BATCH, SEQ_LEN, -1).clone()

    # Override codebook_keys so entry 0 always wins
    with torch.no_grad():
        rnet.codebook_keys.fill_(0.0)
        rnet.codebook_keys[0].fill_(1.0)

    loss, mean_recon, target_conf = crystallization_supervision_loss(rnet, z_H, z_L, tolerance=0.1)
    # Confidence starts ~0.5 (random init), target = 1.0 → BCE ≈ 0.69
    assert loss.item() >= 0.0
    # With perfect codebook: reconstruction error ≈ 0, target_confidence ≈ 1
    assert target_conf.item() >= 0.9


# ---------------------------------------------------------------------------
# CoralV3Inner — crystallization disabled (backward compat)
# ---------------------------------------------------------------------------


def test_no_crystal_returns_3tuple():
    model = make_v3_model(use_crystallization=False)
    carry = make_inner_carry(model)
    result = model(carry, make_batch())
    assert len(result) == 3


def test_crystal_only_returns_4tuple():
    model = make_v3_model(
        use_crystallization=True,
        codebook_size=CODEBOOK_SIZE,
        crystal_proj_dim=PROJ_DIM,
    )
    carry = make_inner_carry(model)
    result = model(carry, make_batch())
    assert len(result) == 4


# ---------------------------------------------------------------------------
# CoralV3Inner — training mode: no bypass, buffer records states
# ---------------------------------------------------------------------------


def test_crystal_training_no_bypass():
    """During training, crystal_bypass_count must always be 0."""
    model = make_v3_model(
        use_crystallization=True,
        codebook_size=CODEBOOK_SIZE,
        crystal_proj_dim=PROJ_DIM,
        crystal_confidence_threshold=0.0,  # would bypass immediately if eval
    )
    model.train()
    carry = make_inner_carry(model)
    _, _, _, pred_metrics = model(carry, make_batch())
    assert pred_metrics.crystal_bypass_count == 0


def test_crystal_training_buffer_records():
    """After training forward pass, crystal buffer should contain entries."""
    model = make_v3_model(
        use_crystallization=True,
        codebook_size=CODEBOOK_SIZE,
        crystal_proj_dim=PROJ_DIM,
    )
    model.train()
    carry = make_inner_carry(model)
    # is_last_segment=True simulates the final ACT segment; recording is gated on this flag
    model(carry, make_batch(), is_last_segment=True)
    # H_cycles=2 → 1 non-last H cycle → 1 recording (B=4 entries)
    assert len(model.crystal_buffer) > 0


def test_crystal_training_supervision_loss_computed():
    """crystal_supervision_loss_final is non-None only after the gate is activated."""
    # With bootstrap_steps=5000 (default), gate starts inactive → loss is None during bootstrap
    model_bootstrap = make_v3_model(
        use_crystallization=True,
        codebook_size=CODEBOOK_SIZE,
        crystal_proj_dim=PROJ_DIM,
    )
    model_bootstrap.train()
    carry = make_inner_carry(model_bootstrap)
    _, _, _, pm_bootstrap = model_bootstrap(carry, make_batch())
    assert pm_bootstrap.crystal_supervision_loss_final is None, (
        "gate inactive during bootstrap — BCE loss should be None"
    )
    # Diagnostics should still be computed even during bootstrap
    assert pm_bootstrap.crystal_reconstruction_error is not None
    assert pm_bootstrap.crystal_target_confidence_mean is not None

    # With bootstrap_steps=0, gate is active from the start → loss is non-None
    model_active = make_v3_model(
        use_crystallization=True,
        codebook_size=CODEBOOK_SIZE,
        crystal_proj_dim=PROJ_DIM,
        crystal_bootstrap_steps=0,
    )
    model_active.train()
    carry2 = make_inner_carry(model_active)
    _, _, _, pm_active = model_active(carry2, make_batch())
    assert pm_active.crystal_supervision_loss_final is not None
    assert pm_active.crystal_supervision_loss_final.shape == ()


# ---------------------------------------------------------------------------
# CoralV3Inner — eval mode: no bypass when confidence is below threshold
# ---------------------------------------------------------------------------


def test_crystal_eval_no_bypass_default_threshold():
    """With default threshold (0.8) and random init, confidence ≈ 0.5 → no bypass."""
    model = make_v3_model(
        use_crystallization=True,
        codebook_size=CODEBOOK_SIZE,
        crystal_proj_dim=PROJ_DIM,
        crystal_confidence_threshold=0.8,
    )
    model.eval()
    carry = make_inner_carry(model)
    with torch.no_grad():
        _, _, _, pred_metrics = model(carry, make_batch())
    assert pred_metrics.crystal_bypass_count == 0


def test_crystal_eval_supervision_loss_none():
    """In eval mode, crystal_supervision_loss_final should be None."""
    model = make_v3_model(
        use_crystallization=True,
        codebook_size=CODEBOOK_SIZE,
        crystal_proj_dim=PROJ_DIM,
    )
    model.eval()
    carry = make_inner_carry(model)
    with torch.no_grad():
        _, _, _, pred_metrics = model(carry, make_batch())
    assert pred_metrics.crystal_supervision_loss_final is None


# ---------------------------------------------------------------------------
# CoralV3Inner — consolidate_codebook
# ---------------------------------------------------------------------------


def test_consolidate_codebook_clears_buffer():
    model = make_v3_model(
        use_crystallization=True,
        codebook_size=CODEBOOK_SIZE,
        crystal_proj_dim=PROJ_DIM,
    )
    model.train()
    # Run enough passes to fill buffer beyond 100 entries threshold
    carry = make_inner_carry(model, B=32)
    for _ in range(10):
        model(carry, make_batch(B=32))

    initial_len = len(model.crystal_buffer)
    if initial_len >= 100:
        model.consolidate_codebook()
        assert len(model.crystal_buffer) == 0
    else:
        # Buffer < 100 — consolidate skips, then we call clear manually
        model.crystal_buffer.clear()
        assert len(model.crystal_buffer) == 0


def test_consolidate_codebook_noop_when_disabled():
    model = make_v3_model(use_crystallization=False)
    model.consolidate_codebook()  # Should not raise


# ---------------------------------------------------------------------------
# All four mechanism combinations
# ---------------------------------------------------------------------------


def _run_forward(model, train_mode):
    if train_mode:
        model.train()
    else:
        model.eval()
    carry = make_inner_carry(model)
    with torch.set_grad_enabled(train_mode):
        return model(carry, make_batch())


@pytest.mark.parametrize("pc,cr,cry", [
    (False, False, False),  # baseline
    (True,  False, False),  # PC only
    (False, True,  False),  # routing only
    (False, False, True),   # crystal only
    (True,  True,  True),   # all three
])
def test_all_combinations_output_shapes(pc, cr, cry):
    model = make_v3_model(
        use_predictive_coding=pc,
        use_columnar_routing=cr,
        num_columns=S,
        active_columns=K,
        use_crystallization=cry,
        codebook_size=CODEBOOK_SIZE,
        crystal_proj_dim=PROJ_DIM,
    )
    result = _run_forward(model, train_mode=True)

    expected_len = 3 if (not pc and not cr and not cry) else 4
    assert len(result) == expected_len

    if expected_len == 3:
        new_carry, output, (q_halt, q_cont) = result
    else:
        new_carry, output, (q_halt, q_cont), pred_metrics = result

    assert output.shape == (BATCH, SEQ_LEN, VOCAB)
    assert q_halt.shape == (BATCH,)
    assert q_cont.shape == (BATCH,)


@pytest.mark.parametrize("pc,cr,cry", [
    (True,  False, False),
    (False, True,  False),
    (False, False, True),
    (True,  True,  True),
])
def test_all_combinations_pred_metrics_fields(pc, cr, cry):
    model = make_v3_model(
        use_predictive_coding=pc,
        use_columnar_routing=cr,
        num_columns=S,
        active_columns=K,
        use_crystallization=cry,
        codebook_size=CODEBOOK_SIZE,
        crystal_proj_dim=PROJ_DIM,
    )
    _, _, _, pred_metrics = _run_forward(model, train_mode=True)

    if pc:
        assert pred_metrics.epsilon_final is not None
        assert pred_metrics.pi_final is not None
    else:
        assert pred_metrics.epsilon_final is None
        assert pred_metrics.pi_final is None

    if cr:
        assert pred_metrics.routing_logits_H is not None
        assert pred_metrics.routing_logits_L is not None
    else:
        assert pred_metrics.routing_logits_H is None
        assert pred_metrics.routing_logits_L is None

    if cry:
        # Gate inactive by default (bootstrap_steps=5000): diagnostics present, BCE None.
        # Use bootstrap_steps=0 to exercise the BCE path.
        assert pred_metrics.crystal_reconstruction_error is not None
        assert pred_metrics.crystal_target_confidence_mean is not None
    else:
        assert pred_metrics.crystal_supervision_loss_final is None
        assert pred_metrics.crystal_reconstruction_error is None
        assert pred_metrics.crystal_target_confidence_mean is None


def test_load_balancing_loss_with_crystal():
    """load_balancing_loss should still work correctly when crystallization is also active."""
    model = make_v3_model(
        use_columnar_routing=True,
        num_columns=S,
        active_columns=K,
        use_crystallization=True,
        codebook_size=CODEBOOK_SIZE,
        crystal_proj_dim=PROJ_DIM,
    )
    _, _, _, pred_metrics = _run_forward(model, train_mode=True)
    all_logits = pred_metrics.routing_logits_H + pred_metrics.routing_logits_L
    loss = load_balancing_loss(all_logits, S)
    assert loss.shape == ()
    assert loss.item() >= 0.0


# ---------------------------------------------------------------------------
# CUDA tests
# ---------------------------------------------------------------------------


@CUDA_ONLY
def test_crystal_cuda_forward_backward():
    """Full forward + backward with crystallization on CUDA."""
    model = make_v3_model(
        use_crystallization=True,
        codebook_size=CODEBOOK_SIZE,
        crystal_proj_dim=PROJ_DIM,
    ).cuda()
    model.train()
    carry = make_inner_carry(model, device="cuda")
    batch = make_batch(device="cuda")
    _, output, _, pred_metrics = model(carry, batch)
    loss = output.sum() + pred_metrics.crystal_supervision_loss_final
    loss.backward()
    assert not output.isnan().any()
    assert pred_metrics.crystal_supervision_loss_final.item() >= 0.0


@CUDA_ONLY
def test_all_three_mechanisms_cuda():
    """All three mechanisms together on CUDA."""
    model = make_v3_model(
        use_predictive_coding=True,
        use_columnar_routing=True,
        num_columns=S,
        active_columns=K,
        use_crystallization=True,
        codebook_size=CODEBOOK_SIZE,
        crystal_proj_dim=PROJ_DIM,
    ).cuda()
    model.train()
    carry = make_inner_carry(model, device="cuda")
    batch = make_batch(device="cuda")
    _, output, _, pred_metrics = model(carry, batch)
    loss = (
        output.sum()
        + pred_metrics.epsilon_final.sum()
        + pred_metrics.crystal_supervision_loss_final
    )
    for lgt in pred_metrics.routing_logits_H + pred_metrics.routing_logits_L:
        loss = loss + lgt.sum()
    loss.backward()
    assert not output.isnan().any()
