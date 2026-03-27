"""Tests for Phase 2: ColumnarTransformerBlock, ColumnarReasoningModule, load_balancing_loss,
CoralV3Inner with columnar routing.

Structural and shape tests run on CPU.
CUDA-only tests require CUDA + flash_attn and are skipped when unavailable.
"""

import pytest
import torch
import torch.nn.functional as F

from coral.models.transformer_block import TransformerBlockConfig
from coral.models.columnar import ColumnarTransformerBlock, ColumnarReasoningModule
from coral.models.coral_base import CoralConfig, InnerCarry
from coral.models.coral_v3 import CoralV3Inner
from coral.training.losses import load_balancing_loss


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

HIDDEN = 64
NUM_HEADS = 4
VOCAB = 32
SEQ_LEN = 8
BATCH = 4
S = 4
K = 2

BLOCK_CFG = TransformerBlockConfig(
    hidden_size=HIDDEN,
    num_heads=NUM_HEADS,
    expansion=4.0,
    rms_norm_eps=1e-5,
)

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
    reset_all = torch.ones(B, dtype=torch.bool, device=device)
    return model.reset_carry(reset_all, carry)


# ---------------------------------------------------------------------------
# ColumnarTransformerBlock
# ---------------------------------------------------------------------------


def test_columnar_block_output_shape():
    block = ColumnarTransformerBlock(BLOCK_CFG, S=S, k=K)
    x = torch.randn(BATCH, SEQ_LEN, HIDDEN)
    out, logits = block(cos_sin=None, hidden_states=x)
    assert out.shape == (BATCH, SEQ_LEN, HIDDEN), f"expected {(BATCH, SEQ_LEN, HIDDEN)}, got {out.shape}"
    assert logits.shape == (BATCH, S), f"expected {(BATCH, S)}, got {logits.shape}"


def test_columnar_block_routing_logits_shape():
    """routing_logits should be [B, S] regardless of sequence length."""
    block = ColumnarTransformerBlock(BLOCK_CFG, S=S, k=K)
    for seq_len in (1, 4, 16):
        x = torch.randn(BATCH, seq_len, HIDDEN)
        _, logits = block(cos_sin=None, hidden_states=x)
        assert logits.shape == (BATCH, S)


def test_columnar_block_topk_columns_selected():
    """With S=4, k=2: each sample selects exactly 2 columns."""
    block = ColumnarTransformerBlock(BLOCK_CFG, S=S, k=K)
    x = torch.randn(BATCH, SEQ_LEN, HIDDEN)
    _, logits = block(cos_sin=None, hidden_states=x)
    topk_vals, topk_idx = logits.topk(K, dim=-1)
    # All selected indices are valid column indices
    assert (topk_idx >= 0).all() and (topk_idx < S).all()
    # Each sample selects exactly k distinct columns
    for b in range(BATCH):
        assert topk_idx[b].unique().numel() == K


def test_columnar_block_output_dtype_preserved():
    block = ColumnarTransformerBlock(BLOCK_CFG, S=S, k=K)
    x = torch.randn(BATCH, SEQ_LEN, HIDDEN, dtype=torch.float32)
    out, _ = block(cos_sin=None, hidden_states=x)
    assert out.dtype == x.dtype


def test_columnar_block_temperature_clamped():
    """temperature.clamp(0.1, 10.0) is applied; confirm model still runs at extreme values."""
    block = ColumnarTransformerBlock(BLOCK_CFG, S=S, k=K)
    with torch.no_grad():
        block.temperature.fill_(1000.0)
    x = torch.randn(BATCH, SEQ_LEN, HIDDEN)
    out, _ = block(cos_sin=None, hidden_states=x)
    assert not out.isnan().any()


def test_columnar_block_k_equals_s():
    """k == S means all columns are selected for every sample."""
    block = ColumnarTransformerBlock(BLOCK_CFG, S=S, k=S)
    x = torch.randn(BATCH, SEQ_LEN, HIDDEN)
    out, logits = block(cos_sin=None, hidden_states=x)
    assert out.shape == (BATCH, SEQ_LEN, HIDDEN)
    assert logits.shape == (BATCH, S)


# ---------------------------------------------------------------------------
# load_balancing_loss
# ---------------------------------------------------------------------------


def test_lbl_returns_scalar():
    logits = [torch.randn(BATCH, S) for _ in range(4)]
    loss = load_balancing_loss(logits, S)
    assert loss.shape == (), f"expected scalar, got shape {loss.shape}"


def test_lbl_zero_for_uniform():
    """When avg_probs is exactly 1/S the KL loss should be near zero."""
    uniform_logits = torch.zeros(BATCH, S)  # softmax → uniform
    logits = [uniform_logits for _ in range(4)]
    loss = load_balancing_loss(logits, S)
    assert loss.item() < 1e-4, f"expected ~0 for uniform, got {loss.item()}"


def test_lbl_positive_for_nonuniform():
    """When one column dominates, loss should be > 0."""
    # All samples route to column 0: logits[0] >> rest
    skewed = torch.full((BATCH, S), -10.0)
    skewed[:, 0] = 10.0
    logits = [skewed for _ in range(4)]
    loss = load_balancing_loss(logits, S)
    assert loss.item() > 0.1, f"expected positive loss for skewed routing, got {loss.item()}"


def test_lbl_differentiable():
    logits = [torch.randn(BATCH, S, requires_grad=True) for _ in range(3)]
    loss = load_balancing_loss(logits, S)
    loss.backward()
    for l in logits:
        assert l.grad is not None


# ---------------------------------------------------------------------------
# ColumnarReasoningModule
# ---------------------------------------------------------------------------


def test_columnar_module_output_shapes():
    module = ColumnarReasoningModule(BLOCK_CFG, num_layers=2, S=S, k=K)
    x = torch.randn(BATCH, SEQ_LEN, HIDDEN)
    injection = torch.randn(BATCH, SEQ_LEN, HIDDEN)
    out, logits_list = module(hidden_states=x, input_injection=injection, cos_sin=None)
    assert out.shape == (BATCH, SEQ_LEN, HIDDEN)
    assert len(logits_list) == 2, f"expected 2 routing tensors (one per layer), got {len(logits_list)}"
    for lgt in logits_list:
        assert lgt.shape == (BATCH, S)


def test_columnar_module_num_layers():
    for num_layers in (1, 3, 4):
        module = ColumnarReasoningModule(BLOCK_CFG, num_layers=num_layers, S=S, k=K)
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN)
        inj = torch.randn(BATCH, SEQ_LEN, HIDDEN)
        _, logits_list = module(x, inj, cos_sin=None)
        assert len(logits_list) == num_layers


# ---------------------------------------------------------------------------
# CoralV3Inner with columnar routing
# ---------------------------------------------------------------------------


def make_v3_routing_model(cr=True, pc=False, cfg_overrides=None) -> CoralV3Inner:
    cfg = dict(SMALL_CFG)
    cfg["use_columnar_routing"] = cr
    cfg["use_predictive_coding"] = pc
    cfg["num_columns"] = S
    cfg["active_columns"] = K
    cfg["lambda_balance"] = 0.01
    if cfg_overrides:
        cfg.update(cfg_overrides)
    return CoralV3Inner(CoralConfig(**cfg))


def test_v3_routing_forward_shapes():
    model = make_v3_routing_model(cr=True, pc=False)
    carry = make_inner_carry(model)
    batch = make_batch()
    result = model(carry, batch)
    assert len(result) == 4, "expected 4-tuple when routing active"
    new_carry, output, (q_halt, q_cont), pred_metrics = result
    assert output.shape == (BATCH, SEQ_LEN, VOCAB)
    assert q_halt.shape == (BATCH,)
    assert q_cont.shape == (BATCH,)


def test_v3_routing_pred_metrics_routing_logits():
    model = make_v3_routing_model(cr=True, pc=False)
    carry = make_inner_carry(model)
    batch = make_batch()
    _, _, _, pred_metrics = model(carry, batch)
    assert pred_metrics.routing_logits_H is not None
    assert pred_metrics.routing_logits_L is not None
    # H has H_layers=2 blocks, L has L_layers=2 blocks
    assert len(pred_metrics.routing_logits_H) == SMALL_CFG["H_layers"]
    assert len(pred_metrics.routing_logits_L) == SMALL_CFG["L_layers"]
    for lgt in pred_metrics.routing_logits_H + pred_metrics.routing_logits_L:
        assert lgt.shape == (BATCH, S)


def test_v3_routing_no_pc_outputs():
    """With cr=True, pc=False: epsilon_final and pi_final should be None."""
    model = make_v3_routing_model(cr=True, pc=False)
    carry = make_inner_carry(model)
    _, _, _, pred_metrics = model(carry, batch=make_batch())
    assert pred_metrics.epsilon_final is None
    assert pred_metrics.pi_final is None


def test_v3_no_routing_baseline():
    """With cr=False (baseline), forward returns a 3-tuple, unchanged from CoralInner."""
    model = make_v3_routing_model(cr=False, pc=False)
    carry = make_inner_carry(model)
    result = model(carry, make_batch())
    assert len(result) == 3, "expected 3-tuple for baseline (no routing, no PC)"


def test_v3_routing_and_pc_combined():
    """cr=True and pc=True together: all fields populated."""
    model = make_v3_routing_model(cr=True, pc=True)
    carry = make_inner_carry(model)
    result = model(carry, make_batch())
    assert len(result) == 4
    _, _, _, pred_metrics = result
    assert pred_metrics.epsilon_final is not None
    assert pred_metrics.pi_final is not None
    assert pred_metrics.routing_logits_H is not None
    assert pred_metrics.routing_logits_L is not None


def test_v3_routing_parameter_count():
    """Total params with routing should be > monolithic (more columns stored),
    but the topology should satisfy: H+L layer counts are correct."""
    from coral.models.coral_base import CoralInner

    baseline = CoralInner(CoralConfig(**{**SMALL_CFG, "forward_dtype": "float32"}))
    routing_model = make_v3_routing_model(cr=True, pc=False)

    baseline_params = sum(p.numel() for p in baseline.parameters())
    routing_params = sum(p.numel() for p in routing_model.parameters())

    # Columnar model stores S columns per block — total params > monolithic
    assert routing_params > baseline_params, (
        f"routing model ({routing_params}) should have more total params than "
        f"baseline ({baseline_params}) because S columns are stored"
    )


# ---------------------------------------------------------------------------
# CUDA tests
# ---------------------------------------------------------------------------


@CUDA_ONLY
def test_columnar_block_cuda_forward_backward():
    block = ColumnarTransformerBlock(BLOCK_CFG, S=S, k=K).cuda()
    x = torch.randn(BATCH, SEQ_LEN, HIDDEN, device="cuda", requires_grad=True)
    out, logits = block(cos_sin=None, hidden_states=x)
    loss = out.sum() + logits.sum()
    loss.backward()
    assert x.grad is not None
    assert not x.grad.isnan().any()


@CUDA_ONLY
def test_v3_routing_cuda_forward_backward():
    model = make_v3_routing_model(cr=True, pc=False).cuda()
    carry = make_inner_carry(model, device="cuda")
    batch = make_batch(device="cuda")
    new_carry, output, (q_halt, q_cont), pred_metrics = model(carry, batch)
    loss = output.sum() + q_halt.sum() + q_cont.sum()
    for lgt in pred_metrics.routing_logits_H + pred_metrics.routing_logits_L:
        loss = loss + lgt.sum()
    loss.backward()
    # Basic sanity: no NaNs in output
    assert not output.isnan().any()


@CUDA_ONLY
def test_v3_routing_and_pc_cuda():
    """Combined PC + routing forward + backward on CUDA."""
    model = make_v3_routing_model(cr=True, pc=True).cuda()
    carry = make_inner_carry(model, device="cuda")
    batch = make_batch(device="cuda")
    _, output, _, pred_metrics = model(carry, batch)
    loss = (
        output.sum()
        + pred_metrics.epsilon_final.sum()
        + pred_metrics.pi_final.sum()
    )
    for lgt in pred_metrics.routing_logits_H + pred_metrics.routing_logits_L:
        loss = loss + lgt.sum()
    loss.backward()
    assert not output.isnan().any()
