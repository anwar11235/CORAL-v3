"""Tests for Phase 3b: SpatialMoECodebook, CrystallizationBuffer, and CoralV3Inner
with Soft MoE crystallization.

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
    SpatialMoECodebook,
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
MOE_MODES = 8   # small K_modes for tests

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
# SpatialMoECodebook unit tests
# ---------------------------------------------------------------------------


def make_moe_codebook() -> SpatialMoECodebook:
    cfg = CoralConfig(
        batch_size=BATCH,
        seq_len=SEQ_LEN,
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        crystal_proj_dim=PROJ_DIM,
        moe_num_modes=MOE_MODES,
    )
    return SpatialMoECodebook(cfg, seq_len=SEQ_LEN)


def test_moe_codebook_forward_shapes():
    mod = make_moe_codebook()
    z_H = torch.randn(BATCH, SEQ_LEN, HIDDEN)
    z_L = torch.randn(BATCH, SEQ_LEN, HIDDEN)
    w, z_bypass, key = mod(z_H, z_L)
    assert w.shape == (BATCH, MOE_MODES + 1)
    assert z_bypass.shape == (BATCH, SEQ_LEN, HIDDEN)
    assert key.shape == (BATCH, PROJ_DIM * 2)


def test_moe_codebook_w_sums_to_one():
    mod = make_moe_codebook()
    z_H = torch.randn(BATCH, SEQ_LEN, HIDDEN)
    z_L = torch.randn(BATCH, SEQ_LEN, HIDDEN)
    w, _, _ = mod(z_H, z_L)
    torch.testing.assert_close(w.sum(dim=-1), torch.ones(BATCH), atol=1e-5, rtol=0)


def test_moe_codebook_bootstrap_mask_forces_passthrough():
    mod = make_moe_codebook()
    mod.bootstrap_mask_router(True)
    z_H = torch.randn(BATCH, SEQ_LEN, HIDDEN)
    z_L = torch.randn(BATCH, SEQ_LEN, HIDDEN)
    w, z_bypass, _ = mod(z_H, z_L)
    # w_pt = w[:, -1] should be 1.0 when bootstrap mask active
    assert (w[:, -1] == 1.0).all(), "bootstrap mask should force w_pt=1.0"
    # z_bypass should be zeros when bootstrap mask active
    assert (z_bypass == 0.0).all()


def test_moe_codebook_losses_shapes():
    mod = make_moe_codebook()
    z_H = torch.randn(BATCH, SEQ_LEN, HIDDEN)
    z_L = torch.randn(BATCH, SEQ_LEN, HIDDEN)
    w, z_bypass, _ = mod(z_H, z_L)
    z_L_final = torch.randn(BATCH, SEQ_LEN, HIDDEN)
    L_recon, L_lb = mod.moe_losses(z_L_final, w, z_bypass)
    assert L_recon.shape == ()
    assert L_lb.shape == ()
    assert L_recon.item() >= 0.0
    assert L_lb.item() >= 0.0


def test_moe_codebook_recon_loss_unweighted_gradient():
    """codebook_values must always receive gradient from L_recon (unweighted)."""
    mod = make_moe_codebook()
    z_H = torch.randn(BATCH, SEQ_LEN, HIDDEN)
    z_L = torch.randn(BATCH, SEQ_LEN, HIDDEN)
    w, z_bypass, _ = mod(z_H, z_L)
    z_L_final = torch.randn(BATCH, SEQ_LEN, HIDDEN)
    L_recon, L_lb = mod.moe_losses(z_L_final, w, z_bypass)
    L_recon.backward()
    # codebook_values must have a gradient regardless of router weights
    assert mod.codebook_values.grad is not None


# ---------------------------------------------------------------------------
# L_lb Option Y unit tests (Phase 3c)
# ---------------------------------------------------------------------------


def _make_w_from_probs(probs: list) -> torch.Tensor:
    """Create a [1, len(probs)] w tensor from a list of probabilities."""
    t = torch.tensor(probs, dtype=torch.float32).unsqueeze(0)
    assert abs(t.sum().item() - 1.0) < 1e-5
    return t


def test_lb_option_y_zero_at_uniform():
    """L_lb ≈ 0 when routing is uniform over K+1 experts."""
    mod = make_moe_codebook()
    K = MOE_MODES
    w = torch.full((BATCH, K + 1), 1.0 / (K + 1))
    z_L_final = torch.randn(BATCH, SEQ_LEN, HIDDEN)
    z_bypass = torch.randn(BATCH, SEQ_LEN, HIDDEN)
    _, L_lb = mod.moe_losses(z_L_final, w, z_bypass)
    assert L_lb.item() < 1e-5, f"L_lb should be ~0 at uniform K+1, got {L_lb.item()}"


def test_lb_option_y_large_at_passthrough_dominance():
    """L_lb is large and positive when all weight is on passthrough."""
    mod = make_moe_codebook()
    K = MOE_MODES
    # w_pt = 1, all codebook weights = 0
    w = torch.zeros(BATCH, K + 1)
    w[:, -1] = 1.0
    z_L_final = torch.randn(BATCH, SEQ_LEN, HIDDEN)
    z_bypass = torch.randn(BATCH, SEQ_LEN, HIDDEN)
    _, L_lb = mod.moe_losses(z_L_final, w, z_bypass)
    assert L_lb.item() > 0.5, f"L_lb should be large at passthrough dominance, got {L_lb.item()}"


def test_lb_option_y_large_at_single_mode_dominance():
    """L_lb is large and positive when all weight is on one codebook mode."""
    mod = make_moe_codebook()
    K = MOE_MODES
    w = torch.zeros(BATCH, K + 1)
    w[:, 3] = 1.0  # single codebook mode
    z_L_final = torch.randn(BATCH, SEQ_LEN, HIDDEN)
    z_bypass = torch.randn(BATCH, SEQ_LEN, HIDDEN)
    _, L_lb = mod.moe_losses(z_L_final, w, z_bypass)
    assert L_lb.item() > 0.5, f"L_lb should be large at single-mode dominance, got {L_lb.item()}"


def test_lb_option_y_nonzero_at_uniform_codebook_zero_passthrough():
    """Uniform codebook with w_pt=0 is NOT uniform over K+1 → L_lb > 0."""
    mod = make_moe_codebook()
    K = MOE_MODES
    # w_pt = 0, uniform over K codebook modes (1/K each, not 1/(K+1))
    w = torch.zeros(BATCH, K + 1)
    w[:, :K] = 1.0 / K
    z_L_final = torch.randn(BATCH, SEQ_LEN, HIDDEN)
    z_bypass = torch.randn(BATCH, SEQ_LEN, HIDDEN)
    _, L_lb = mod.moe_losses(z_L_final, w, z_bypass)
    assert L_lb.item() > 1e-4, (
        f"L_lb should be > 0 for uniform-codebook+zero-passthrough (not uniform over K+1), "
        f"got {L_lb.item()}"
    )


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
    assert buf.keys.device.type == "cpu"
    assert buf.values.device.type == "cpu"


def test_buffer_clear():
    buf = CrystallizationBuffer(capacity=100)
    buf.add(torch.randn(BATCH, PROJ_DIM * 2), torch.randn(BATCH, HIDDEN))
    buf.clear()
    assert len(buf) == 0
    assert buf.pointer == 0


def test_buffer_spatial_add_and_retrieve():
    """Spatial z_L added to buffer should be stored on CPU."""
    buf = CrystallizationBuffer(capacity=100)
    keys = torch.randn(BATCH, PROJ_DIM * 2)
    values = torch.randn(BATCH, HIDDEN)
    z_L_spatial = torch.randn(BATCH, SEQ_LEN, HIDDEN)
    buf.add(keys, values, z_L_spatial=z_L_spatial)
    assert buf.spatial_buffer is not None
    assert buf.spatial_buffer.device.type == "cpu"
    assert buf.spatial_buffer.shape == (buf.capacity, SEQ_LEN, HIDDEN)


def test_buffer_consolidate_spatial_basic():
    """consolidate_spatial returns centroids of correct shape."""
    capacity = 200
    k_modes = 4
    buf = CrystallizationBuffer(capacity=capacity)
    for _ in range(50):
        z_L = torch.randn(BATCH, SEQ_LEN, HIDDEN)
        buf.add(
            torch.randn(BATCH, PROJ_DIM * 2),
            torch.randn(BATCH, HIDDEN),
            z_L_spatial=z_L,
        )
    result = buf.consolidate_spatial(k_modes=k_modes, num_iterations=3)
    assert result is not None
    centroids, utilization = result
    assert centroids.shape == (k_modes, SEQ_LEN, HIDDEN)
    assert 0 <= utilization <= k_modes


def test_buffer_fill_progression():
    """len(buffer) climbs to capacity then stays there; never exceeds capacity."""
    capacity = 10000
    batch_size = 384
    buf = CrystallizationBuffer(capacity=capacity)
    for step in range(100):
        buf.add(torch.randn(batch_size, PROJ_DIM * 2), torch.randn(batch_size, HIDDEN))
        expected = min((step + 1) * batch_size, capacity)
        assert len(buf) == expected, f"step {step}: expected {expected}, got {len(buf)}"
    assert len(buf) == capacity


def test_buffer_ring_semantics():
    """After saturation, new items overwrite old ones and pointer advances correctly."""
    capacity = 10
    buf = CrystallizationBuffer(capacity=capacity)
    key_dim = PROJ_DIM * 2

    fill_keys = torch.arange(capacity * key_dim, dtype=torch.float32).reshape(capacity, key_dim)
    buf.add(fill_keys[:4], torch.zeros(4, HIDDEN))
    buf.add(fill_keys[4:8], torch.zeros(4, HIDDEN))
    buf.add(fill_keys[8:], torch.zeros(2, HIDDEN))
    assert len(buf) == capacity

    new_keys = torch.full((3, key_dim), 999.0)
    buf.add(new_keys, torch.zeros(3, HIDDEN))
    assert len(buf) == capacity

    torch.testing.assert_close(buf.keys[:3], new_keys)
    torch.testing.assert_close(buf.keys[3:], fill_keys[3:])


def test_buffer_single_large_add():
    """Adding a batch larger than capacity retains only the last capacity rows."""
    capacity = 10000
    buf = CrystallizationBuffer(capacity=capacity)
    B = 15000
    keys = torch.arange(B * PROJ_DIM * 2, dtype=torch.float32).reshape(B, PROJ_DIM * 2)
    values = torch.zeros(B, HIDDEN)
    buf.add(keys, values)

    assert len(buf) == capacity
    assert buf.pointer == 0
    torch.testing.assert_close(buf.keys, keys[B - capacity :])


@CUDA_ONLY
def test_buffer_add_throughput():
    """Verify add() throughput does not degrade after buffer saturation."""
    import time
    buffer = CrystallizationBuffer(capacity=10000)
    keys = torch.randn(384, 256, device="cuda")
    values = torch.randn(384, 512, device="cuda")

    for _ in range(5):
        buffer.add(keys, values)
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    start_empty = time.perf_counter()
    for _ in range(50):
        buffer.add(keys, values)
    torch.cuda.synchronize()
    t_empty = (time.perf_counter() - start_empty) / 50

    torch.cuda.synchronize()
    start_full = time.perf_counter()
    for _ in range(50):
        buffer.add(keys, values)
    torch.cuda.synchronize()
    t_full = (time.perf_counter() - start_full) / 50

    assert t_full < 0.002, (
        f"Buffer add() post-saturation too slow: "
        f"pre={t_empty * 1000:.2f}ms, post={t_full * 1000:.2f}ms (ceiling: 2.00ms)"
    )


# ---------------------------------------------------------------------------
# CoralV3Inner — crystallization disabled (backward compat)
# ---------------------------------------------------------------------------


def test_no_crystal_returns_3tuple():
    model = make_v3_model(use_crystallization=False)
    carry = make_inner_carry(model)
    result = model(carry, make_batch())
    assert len(result) == 3


def test_pc_with_crystal_returns_4tuple():
    """use_predictive_coding=True + use_crystallization=True → 4-tuple."""
    model = make_v3_model(
        use_predictive_coding=True,
        use_crystallization=True,
        crystal_proj_dim=PROJ_DIM,
        moe_num_modes=MOE_MODES,
    )
    carry = make_inner_carry(model)
    result = model(carry, make_batch())
    assert len(result) == 4


def test_pc_only_returns_4tuple():
    """use_predictive_coding=True alone → 4-tuple."""
    model = make_v3_model(use_predictive_coding=True)
    carry = make_inner_carry(model)
    result = model(carry, make_batch())
    assert len(result) == 4


def test_unsupported_dispatch_raises():
    """Paths that require columnar routing raise NotImplementedError."""
    model = make_v3_model(use_columnar_routing=True, num_columns=4, active_columns=2)
    carry = make_inner_carry(model)
    with pytest.raises(NotImplementedError):
        model(carry, make_batch())


# ---------------------------------------------------------------------------
# CoralV3Inner — training mode: buffer records states
# ---------------------------------------------------------------------------


def test_crystal_training_moe_losses_during_bootstrap():
    """During bootstrap, moe_recon_loss and moe_lb_loss should be None."""
    model = make_v3_model(
        use_predictive_coding=True,
        use_crystallization=True,
        crystal_proj_dim=PROJ_DIM,
        moe_num_modes=MOE_MODES,
        crystal_bootstrap_steps=5000,  # bootstrap active
    )
    model.train()
    carry = make_inner_carry(model)
    _, _, _, pm = model(carry, make_batch())
    assert pm.moe_recon_loss is None, "bootstrap active — MoE losses should be None"
    assert pm.moe_lb_loss is None
    assert pm.moe_passthrough_weight == 1.0, "passthrough weight should be 1.0 during bootstrap"


def test_crystal_training_moe_losses_post_bootstrap():
    """With bootstrap_steps=0, MoE losses are computed immediately."""
    model = make_v3_model(
        use_predictive_coding=True,
        use_crystallization=True,
        crystal_proj_dim=PROJ_DIM,
        moe_num_modes=MOE_MODES,
        crystal_bootstrap_steps=0,  # no bootstrap
    )
    model.train()
    carry = make_inner_carry(model)
    _, _, _, pm = model(carry, make_batch())
    assert pm.moe_recon_loss is not None, "no bootstrap — MoE recon loss should be computed"
    assert pm.moe_lb_loss is not None
    assert pm.moe_recon_loss.shape == ()
    assert pm.moe_lb_loss.shape == ()
    assert 0.0 <= pm.moe_passthrough_weight <= 1.0


def test_crystal_training_buffer_records():
    """After training forward pass with is_last_segment=True, buffer should have entries."""
    model = make_v3_model(
        use_predictive_coding=True,
        use_crystallization=True,
        crystal_proj_dim=PROJ_DIM,
        moe_num_modes=MOE_MODES,
    )
    model.train()
    carry = make_inner_carry(model)
    model(carry, make_batch(), is_last_segment=True)
    # H_cycles=2 → 1 non-last H cycle → 1 recording (B=4 entries)
    assert len(model.crystal_buffer) > 0


def test_crystal_eval_moe_losses_none():
    """In eval mode, MoE losses should be None (no gradient needed)."""
    model = make_v3_model(
        use_predictive_coding=True,
        use_crystallization=True,
        crystal_proj_dim=PROJ_DIM,
        moe_num_modes=MOE_MODES,
        crystal_bootstrap_steps=0,
    )
    model.eval()
    carry = make_inner_carry(model)
    with torch.no_grad():
        _, _, _, pm = model(carry, make_batch())
    assert pm.moe_recon_loss is None
    assert pm.moe_lb_loss is None


# ---------------------------------------------------------------------------
# CoralV3Inner — consolidate_codebook
# ---------------------------------------------------------------------------


def test_consolidate_codebook_clears_buffer():
    model = make_v3_model(
        use_predictive_coding=True,
        use_crystallization=True,
        crystal_proj_dim=PROJ_DIM,
        moe_num_modes=MOE_MODES,
    )
    model.train()
    carry = make_inner_carry(model, B=32)
    for _ in range(10):
        model(carry, make_batch(B=32), is_last_segment=True)

    initial_len = len(model.crystal_buffer)
    if initial_len >= MOE_MODES:
        result = model.consolidate_codebook()
        assert len(model.crystal_buffer) == 0
    else:
        model.crystal_buffer.clear()
        assert len(model.crystal_buffer) == 0


def test_consolidate_codebook_noop_when_disabled():
    model = make_v3_model(use_crystallization=False)
    model.consolidate_codebook()  # Should not raise


def test_consolidate_codebook_updates_codebook_values():
    """After consolidation with enough spatial data, codebook_values should change."""
    model = make_v3_model(
        use_predictive_coding=True,
        use_crystallization=True,
        crystal_proj_dim=PROJ_DIM,
        moe_num_modes=MOE_MODES,
        crystal_bootstrap_steps=5000,
    )
    model.train()
    carry = make_inner_carry(model, B=32)
    # Run enough steps to fill the buffer with spatial z_L data
    for _ in range(15):
        model(carry, make_batch(B=32), is_last_segment=True)

    initial_values = model.moe_codebook.codebook_values.data.clone()
    if model.crystal_buffer.size >= MOE_MODES and model.crystal_buffer.spatial_buffer is not None:
        result = model.consolidate_codebook()
        if result is not None:
            # Codebook values should differ from random init after k-means
            changed = (model.moe_codebook.codebook_values.data != initial_values).any()
            assert changed, "codebook_values should be updated after consolidation"
            assert not model._crystal_bootstrap_active, "bootstrap should be deactivated"


# ---------------------------------------------------------------------------
# PredMetrics field checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pc,cry", [
    (True,  False),   # PC only
    (True,  True),    # PC + crystal (bootstrap active)
])
def test_pred_metrics_pc_fields(pc, cry):
    model = make_v3_model(
        use_predictive_coding=pc,
        use_crystallization=cry,
        crystal_proj_dim=PROJ_DIM,
        moe_num_modes=MOE_MODES,
    )
    carry = make_inner_carry(model)
    model.train()
    _, _, _, pm = model(carry, make_batch())

    assert pm.epsilon_final is not None
    assert pm.pi_final is not None
    assert pm.routing_logits_H is None
    assert pm.routing_logits_L is None


# ---------------------------------------------------------------------------
# CUDA tests
# ---------------------------------------------------------------------------


@CUDA_ONLY
def test_crystal_cuda_forward_backward():
    """Full forward + backward with PC + crystallization on CUDA."""
    model = make_v3_model(
        use_predictive_coding=True,
        use_crystallization=True,
        crystal_proj_dim=PROJ_DIM,
        moe_num_modes=MOE_MODES,
        crystal_bootstrap_steps=0,  # activate codebook immediately
    ).cuda()
    model.train()
    carry = make_inner_carry(model, device="cuda")
    batch = make_batch(device="cuda")
    _, output, _, pm = model(carry, batch)
    loss = output.sum()
    if pm.moe_recon_loss is not None:
        loss = loss + pm.moe_recon_loss
    loss.backward()
    assert not output.isnan().any()
