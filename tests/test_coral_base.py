"""Tests for TransformerBlock, ReasoningModule, and CoralInner (Steps 0.3–0.5).

Forward-pass tests require CUDA + flash_attn and are skipped when unavailable.
Structural tests (shapes, buffer vs parameter, init values) run on CPU.
"""

import pytest
import torch
import torch.nn as nn

from coral.models.transformer_block import TransformerBlock, TransformerBlockConfig
from coral.models.reasoning_module import ReasoningModule
from coral.models.coral_base import CoralConfig, CoralInner, InnerCarry

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

HIDDEN = 64   # small hidden size for fast tests (must be divisible by num_heads)
NUM_HEADS = 4
HEAD_DIM = HIDDEN // NUM_HEADS  # 16
VOCAB = 32
SEQ_LEN = 8
BATCH = 2

SMALL_BLOCK_CFG = TransformerBlockConfig(
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
    forward_dtype="float32",   # float32 so CPU tests work (no bfloat16 issues on CPU)
    puzzle_emb_ndim=0,
    num_puzzle_identifiers=0,
)

CUDA_ONLY = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Requires CUDA + flash_attn",
)


def make_model(cfg_overrides=None) -> CoralInner:
    cfg = dict(SMALL_CFG)
    if cfg_overrides:
        cfg.update(cfg_overrides)
    return CoralInner(CoralConfig(**cfg))


def make_batch(B=BATCH, seq=SEQ_LEN, vocab=VOCAB, device="cpu"):
    return {
        "inputs": torch.randint(0, vocab, (B, seq), device=device),
    }


# ---------------------------------------------------------------------------
# TransformerBlock — structural tests (no forward pass)
# ---------------------------------------------------------------------------


class TestTransformerBlockStructure:
    def test_has_self_attn(self):
        block = TransformerBlock(SMALL_BLOCK_CFG)
        assert hasattr(block, "self_attn")

    def test_has_mlp(self):
        block = TransformerBlock(SMALL_BLOCK_CFG)
        assert hasattr(block, "mlp")

    def test_no_norm_parameters(self):
        """RMSNorm is a pure function — no learnable gamma/beta should exist."""
        block = TransformerBlock(SMALL_BLOCK_CFG)
        param_names = [n for n, _ in block.named_parameters()]
        norm_params = [n for n in param_names if "norm" in n.lower()]
        assert norm_params == [], f"Unexpected norm parameters: {norm_params}"

    def test_attn_no_bias(self):
        block = TransformerBlock(SMALL_BLOCK_CFG)
        assert block.self_attn.qkv_proj.bias is None
        assert block.self_attn.o_proj.bias is None

    def test_mlp_no_bias(self):
        block = TransformerBlock(SMALL_BLOCK_CFG)
        assert block.mlp.gate_up_proj.bias is None
        assert block.mlp.down_proj.bias is None


# ---------------------------------------------------------------------------
# ReasoningModule — structural tests
# ---------------------------------------------------------------------------


class TestReasoningModuleStructure:
    def test_layers_count(self):
        blocks = [TransformerBlock(SMALL_BLOCK_CFG) for _ in range(3)]
        mod = ReasoningModule(blocks)
        assert len(mod.layers) == 3

    def test_layers_are_module_list(self):
        blocks = [TransformerBlock(SMALL_BLOCK_CFG) for _ in range(2)]
        mod = ReasoningModule(blocks)
        assert isinstance(mod.layers, nn.ModuleList)


# ---------------------------------------------------------------------------
# CoralInner — structural / init tests (CPU, no forward pass)
# ---------------------------------------------------------------------------


class TestCoralInnerStructure:
    def test_h_init_is_buffer_not_parameter(self):
        model = make_model()
        buffer_names = {n for n, _ in model.named_buffers()}
        param_names = {n for n, _ in model.named_parameters()}
        assert "H_init" in buffer_names, "H_init should be a buffer"
        assert "H_init" not in param_names, "H_init must NOT be a parameter"

    def test_l_init_is_buffer_not_parameter(self):
        model = make_model()
        buffer_names = {n for n, _ in model.named_buffers()}
        param_names = {n for n, _ in model.named_parameters()}
        assert "L_init" in buffer_names, "L_init should be a buffer"
        assert "L_init" not in param_names, "L_init must NOT be a parameter"

    def test_h_init_shape(self):
        model = make_model()
        assert model.H_init.shape == (HIDDEN,)

    def test_l_init_shape(self):
        model = make_model()
        assert model.L_init.shape == (HIDDEN,)

    def test_q_head_weight_zero(self):
        """q_head weight should start at exactly zero."""
        model = make_model()
        assert (model.q_head.weight == 0).all()

    def test_q_head_bias_minus_five(self):
        """q_head bias should start at -5."""
        model = make_model()
        assert model.q_head.bias is not None
        assert torch.allclose(model.q_head.bias, torch.full_like(model.q_head.bias, -5.0))

    def test_h_level_layer_count(self):
        model = make_model()
        assert len(model.H_level.layers) == SMALL_CFG["H_layers"]

    def test_l_level_layer_count(self):
        model = make_model()
        assert len(model.L_level.layers) == SMALL_CFG["L_layers"]

    def test_rotary_emb_present_for_rope(self):
        model = make_model()
        assert hasattr(model, "rotary_emb")

    def test_puzzle_emb_created_when_ndim_nonzero(self):
        """puzzle_emb_ndim > 0 should create a CastedSparseEmbedding."""
        from coral.models.sparse_embedding import CastedSparseEmbedding
        model = make_model({"puzzle_emb_ndim": HIDDEN, "num_puzzle_identifiers": 10})
        assert hasattr(model, "puzzle_emb")
        assert isinstance(model.puzzle_emb, CastedSparseEmbedding)

    def test_puzzle_emb_len_ceiling_div(self):
        """puzzle_emb_len = ceil(puzzle_emb_ndim / hidden_size)."""
        # 32 / 64 = 0.5 → ceil = 1
        model = make_model({"puzzle_emb_ndim": 32, "num_puzzle_identifiers": 10})
        assert model.puzzle_emb_len == 1
        assert model.total_seq_len == SEQ_LEN + 1

    def test_empty_carry_shape(self):
        model = make_model()
        carry = model.empty_carry(BATCH)
        assert carry.z_H.shape == (BATCH, SEQ_LEN, HIDDEN)
        assert carry.z_L.shape == (BATCH, SEQ_LEN, HIDDEN)

    def test_reset_carry_replaces_halted(self):
        """Halted positions should be replaced with init values."""
        model = make_model()
        carry = model.empty_carry(BATCH)
        reset_flag = torch.tensor([True, False])
        new_carry = model.reset_carry(reset_flag, carry)
        assert torch.allclose(new_carry.z_H[0], model.H_init.expand(SEQ_LEN, HIDDEN))
        assert torch.allclose(new_carry.z_H[1], carry.z_H[1])


# ---------------------------------------------------------------------------
# CoralInner — forward pass tests (CUDA required)
# ---------------------------------------------------------------------------


@CUDA_ONLY
class TestCoralInnerForward:
    def _make_model_cuda(self):
        cfg = dict(SMALL_CFG)
        cfg["forward_dtype"] = "bfloat16"
        return CoralInner(CoralConfig(**cfg)).cuda()

    def _make_carry(self, model):
        carry = model.empty_carry(BATCH)
        reset_all = torch.ones(BATCH, dtype=torch.bool, device="cuda")
        return model.reset_carry(reset_all, InnerCarry(
            z_H=carry.z_H.cuda(),
            z_L=carry.z_L.cuda(),
        ))

    def test_output_shape(self):
        """Output logits should be [B, seq_len, vocab_size] — no puzzle tokens."""
        model = self._make_model_cuda()
        carry = self._make_carry(model)
        batch = make_batch(device="cuda")
        _, output, _ = model(carry, batch)
        assert output.shape == (BATCH, SEQ_LEN, VOCAB)

    def test_q_shape(self):
        """q_halt and q_continue should each be [B]."""
        model = self._make_model_cuda()
        carry = self._make_carry(model)
        batch = make_batch(device="cuda")
        _, _, (q_halt, q_continue) = model(carry, batch)
        assert q_halt.shape == (BATCH,)
        assert q_continue.shape == (BATCH,)

    def test_q_is_float32(self):
        """Q-values should always be float32 regardless of forward_dtype."""
        model = self._make_model_cuda()
        carry = self._make_carry(model)
        batch = make_batch(device="cuda")
        _, _, (q_halt, q_continue) = model(carry, batch)
        assert q_halt.dtype == torch.float32
        assert q_continue.dtype == torch.float32

    def test_carry_is_detached(self):
        """New carry tensors should have no grad_fn (detached)."""
        model = self._make_model_cuda()
        carry = self._make_carry(model)
        batch = make_batch(device="cuda")
        new_carry, _, _ = model(carry, batch)
        assert new_carry.z_H.grad_fn is None
        assert new_carry.z_L.grad_fn is None

    def test_output_has_grad_fn(self):
        """Output logits should be part of the computation graph."""
        model = self._make_model_cuda()
        carry = self._make_carry(model)
        batch = make_batch(device="cuda")
        _, output, _ = model(carry, batch)
        assert output.grad_fn is not None, "Output should have a grad_fn (1-step grad)"

    def test_1step_gradient_backward(self):
        """Backward should succeed without error and produce non-None gradients."""
        model = self._make_model_cuda()
        carry = self._make_carry(model)
        batch = make_batch(device="cuda")
        _, output, _ = model(carry, batch)
        loss = output.float().sum()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0, "No parameter gradients after backward"

    def test_only_final_steps_in_graph(self):
        """The assert inside forward() enforces no-grad on carry tensors entering 1-step grad."""
        model = self._make_model_cuda()
        carry = self._make_carry(model)
        batch = make_batch(device="cuda")
        model(carry, batch)
