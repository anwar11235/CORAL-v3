"""Tests for coral/models/common.py and coral/models/layers.py.

These tests verify the core layer primitives for Phase 0: faithful HRM reproduction.
They use CPU-only execution where possible (flash_attn requires CUDA, so Attention
tests are skipped when CUDA is unavailable).
"""

import math
import pytest
import torch
import torch.nn as nn

from coral.models.common import trunc_normal_init_, rms_norm
from coral.models.layers import (
    CastedLinear,
    CastedEmbedding,
    RotaryEmbedding,
    SwiGLU,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ATOL = 1e-4


# ---------------------------------------------------------------------------
# trunc_normal_init_
# ---------------------------------------------------------------------------


class TestTruncNormalInit:
    def test_mean_near_zero(self):
        """Mean of initialized tensor should be approximately 0."""
        t = torch.empty(10_000)
        trunc_normal_init_(t, std=1.0)
        assert abs(t.mean().item()) < 0.05, f"mean too large: {t.mean().item()}"

    def test_std_approximately_correct(self):
        """Std of initialized tensor should be close to the requested std."""
        for requested_std in [0.5, 1.0, 2.0]:
            t = torch.empty(10_000)
            trunc_normal_init_(t, std=requested_std)
            actual_std = t.std().item()
            assert abs(actual_std - requested_std) / requested_std < 0.05, (
                f"std mismatch: requested={requested_std}, actual={actual_std}"
            )

    def test_values_within_truncation_bounds(self):
        """All values should lie within [lower*comp_std, upper*comp_std].

        The clip in the implementation means all values are within the scaled
        truncation region.
        """
        t = torch.empty(10_000)
        std = 1.0
        lower, upper = -2.0, 2.0
        trunc_normal_init_(t, std=std, lower=lower, upper=upper)
        # comp_std >= std, so values must be in [lower*comp_std, upper*comp_std]
        # A sufficient check is that nothing wildly outside the requested range
        assert t.min().item() > lower * 3, f"min too low: {t.min().item()}"
        assert t.max().item() < upper * 3, f"max too high: {t.max().item()}"

    def test_zero_std(self):
        """std=0 should produce an all-zero tensor."""
        t = torch.ones(100)
        trunc_normal_init_(t, std=0.0)
        assert (t == 0).all()

    def test_returns_tensor(self):
        """Should return the same tensor object (in-place)."""
        t = torch.empty(10)
        result = trunc_normal_init_(t, std=1.0)
        assert result is t

    def test_lecun_normal_std(self):
        """LeCun normal (std=1/sqrt(n)) should give correct empirical std."""
        in_features = 512
        std = 1.0 / math.sqrt(in_features)
        t = torch.empty(100_000)
        trunc_normal_init_(t, std=std)
        assert abs(t.std().item() - std) / std < 0.05


# ---------------------------------------------------------------------------
# rms_norm
# ---------------------------------------------------------------------------


class TestRmsNorm:
    def test_unit_rms(self):
        """Output RMS along last dim should be approximately 1."""
        x = torch.randn(4, 16, 512)
        y = rms_norm(x, variance_epsilon=1e-5)
        rms = y.float().square().mean(-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4), (
            f"RMS not close to 1: min={rms.min().item():.6f}, max={rms.max().item():.6f}"
        )

    def test_preserves_dtype_float32(self):
        """Output dtype should match input dtype (float32)."""
        x = torch.randn(2, 8, 64, dtype=torch.float32)
        y = rms_norm(x, variance_epsilon=1e-5)
        assert y.dtype == torch.float32

    def test_preserves_dtype_bfloat16(self):
        """Output dtype should match input dtype (bfloat16)."""
        x = torch.randn(2, 8, 64).to(torch.bfloat16)
        y = rms_norm(x, variance_epsilon=1e-5)
        assert y.dtype == torch.bfloat16

    def test_shape_preserved(self):
        """Output shape should match input shape."""
        x = torch.randn(3, 10, 128)
        y = rms_norm(x, variance_epsilon=1e-5)
        assert y.shape == x.shape

    def test_no_learnable_parameters(self):
        """rms_norm is a pure function — no nn.Module, no parameters."""
        import inspect
        assert callable(rms_norm)
        assert not isinstance(rms_norm, nn.Module)

    def test_epsilon_prevents_division_by_zero(self):
        """Zero input should not produce NaN or Inf."""
        x = torch.zeros(4, 16)
        y = rms_norm(x, variance_epsilon=1e-5)
        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()


# ---------------------------------------------------------------------------
# CastedLinear
# ---------------------------------------------------------------------------


class TestCastedLinear:
    def test_output_shape(self):
        """Output shape should be [..., out_features]."""
        layer = CastedLinear(64, 128, bias=False)
        x = torch.randn(2, 10, 64)
        y = layer(x)
        assert y.shape == (2, 10, 128)

    def test_dtype_casting_bfloat16(self):
        """Weight should be cast to input dtype; output dtype matches input."""
        layer = CastedLinear(64, 32, bias=False)
        x = torch.randn(2, 64).to(torch.bfloat16)
        y = layer(x)
        assert y.dtype == torch.bfloat16

    def test_weight_dtype_is_float32(self):
        """Weight should be stored in float32 (master copy)."""
        layer = CastedLinear(32, 16, bias=False)
        assert layer.weight.dtype == torch.float32

    def test_bias_zero_initialized(self):
        """Bias should start at exactly zero."""
        layer = CastedLinear(32, 16, bias=True)
        assert layer.bias is not None
        assert (layer.bias == 0).all()

    def test_no_bias(self):
        """bias=False → self.bias is None."""
        layer = CastedLinear(32, 16, bias=False)
        assert layer.bias is None

    def test_bias_casting(self):
        """Bias should also be cast to input dtype."""
        layer = CastedLinear(32, 16, bias=True)
        x = torch.randn(4, 32).to(torch.bfloat16)
        y = layer(x)
        assert y.dtype == torch.bfloat16

    def test_weight_init_std(self):
        """Weight empirical std should be approximately 1/sqrt(in_features)."""
        in_features = 256
        layer = CastedLinear(in_features, 1024, bias=False)
        expected_std = 1.0 / math.sqrt(in_features)
        actual_std = layer.weight.std().item()
        assert abs(actual_std - expected_std) / expected_std < 0.05, (
            f"Weight std mismatch: expected≈{expected_std:.4f}, got {actual_std:.4f}"
        )


# ---------------------------------------------------------------------------
# CastedEmbedding
# ---------------------------------------------------------------------------


class TestCastedEmbedding:
    def test_output_shape(self):
        emb = CastedEmbedding(100, 64, init_std=0.1, cast_to=torch.bfloat16)
        indices = torch.randint(0, 100, (2, 10))
        y = emb(indices)
        assert y.shape == (2, 10, 64)

    def test_output_dtype(self):
        emb = CastedEmbedding(100, 64, init_std=0.1, cast_to=torch.bfloat16)
        indices = torch.randint(0, 100, (2, 10))
        y = emb(indices)
        assert y.dtype == torch.bfloat16

    def test_weight_init_std(self):
        std = 1.0 / math.sqrt(512)
        emb = CastedEmbedding(1000, 512, init_std=std, cast_to=torch.float32)
        actual = emb.embedding_weight.std().item()
        assert abs(actual - std) / std < 0.05


# ---------------------------------------------------------------------------
# RotaryEmbedding
# ---------------------------------------------------------------------------


class TestRotaryEmbedding:
    def test_cache_shape(self):
        """cos/sin caches should have shape [max_pos, dim]."""
        dim, max_pos = 64, 512
        rope = RotaryEmbedding(dim=dim, max_position_embeddings=max_pos)
        cos, sin = rope()
        assert cos.shape == (max_pos, dim)
        assert sin.shape == (max_pos, dim)

    def test_cos_sin_values(self):
        """cos^2 + sin^2 should equal 1 element-wise."""
        rope = RotaryEmbedding(dim=64, max_position_embeddings=128)
        cos, sin = rope()
        identity = cos ** 2 + sin ** 2
        assert torch.allclose(identity, torch.ones_like(identity), atol=1e-5)

    def test_not_persistent(self):
        """Caches should NOT appear in the state dict (persistent=False)."""
        rope = RotaryEmbedding(dim=64, max_position_embeddings=64)
        assert "cos_cached" not in rope.state_dict()
        assert "sin_cached" not in rope.state_dict()


# ---------------------------------------------------------------------------
# SwiGLU — intermediate dimension computation
# ---------------------------------------------------------------------------


class TestSwiGLU:
    @pytest.mark.parametrize(
        "hidden_size,expansion,expected_inter",
        [
            (512, 4, 1536),   # round(4*512*2/3)=1365 → 1536
            (256, 4, 768),    # round(4*256*2/3)=682  → 768
            (128, 4, 384),    # round(4*128*2/3)=341  → 384
            (512, 2, 768),    # round(2*512*2/3)=682  → 768
        ],
    )
    def test_intermediate_dim(self, hidden_size, expansion, expected_inter):
        """SwiGLU inter dim should be correctly rounded to next multiple of 256."""
        mlp = SwiGLU(hidden_size=hidden_size, expansion=expansion)
        actual_inter = mlp.down_proj.weight.shape[1]  # down_proj: inter → hidden_size
        assert actual_inter == expected_inter, (
            f"hidden={hidden_size}, expansion={expansion}: "
            f"expected inter={expected_inter}, got {actual_inter}"
        )

    def test_gate_up_proj_shape(self):
        """gate_up_proj should map hidden_size → inter*2."""
        mlp = SwiGLU(hidden_size=512, expansion=4)
        # weight shape is [out_features, in_features]
        assert mlp.gate_up_proj.weight.shape == (1536 * 2, 512)

    def test_down_proj_shape(self):
        """down_proj should map inter → hidden_size."""
        mlp = SwiGLU(hidden_size=512, expansion=4)
        assert mlp.down_proj.weight.shape == (512, 1536)

    def test_output_shape(self):
        """Output shape should equal input shape."""
        mlp = SwiGLU(hidden_size=512, expansion=4)
        x = torch.randn(2, 10, 512)
        y = mlp(x)
        assert y.shape == x.shape

    def test_no_bias(self):
        """SwiGLU should have no bias on any projection."""
        mlp = SwiGLU(hidden_size=64, expansion=4)
        assert mlp.gate_up_proj.bias is None
        assert mlp.down_proj.bias is None

    def test_specific_inter_512_expansion_4(self):
        """Explicit check: hidden=512, expansion=4 → inter=1536."""
        mlp = SwiGLU(hidden_size=512, expansion=4)
        x = torch.randn(2, 8, 512)
        y = mlp(x)
        assert y.shape == (2, 8, 512)
        inter = mlp.down_proj.weight.shape[1]
        assert inter == 1536


# ---------------------------------------------------------------------------
# Attention (requires CUDA + flash_attn)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA and flash_attn")
class TestAttention:
    def _make_attention(self):
        from coral.models.layers import Attention
        return Attention(
            hidden_size=512,
            head_dim=64,
            num_heads=8,
            num_key_value_heads=8,
            causal=False,
        ).cuda().to(torch.bfloat16)

    def _make_rope(self, seq_len: int):
        rope = RotaryEmbedding(dim=64, max_position_embeddings=seq_len).cuda()
        return rope()

    def test_output_shape(self):
        """Attention output should be [B, seq_len, hidden_size]."""
        attn = self._make_attention()
        B, seq_len = 2, 32
        x = torch.randn(B, seq_len, 512, device="cuda", dtype=torch.bfloat16)
        cos_sin = self._make_rope(seq_len)
        y = attn(cos_sin, x)
        assert y.shape == (B, seq_len, 512)

    def test_output_dtype(self):
        """Output dtype should match input dtype."""
        attn = self._make_attention()
        x = torch.randn(2, 16, 512, device="cuda", dtype=torch.bfloat16)
        cos_sin = self._make_rope(16)
        y = attn(cos_sin, x)
        assert y.dtype == torch.bfloat16

    def test_no_bias(self):
        """Attention projections should have no bias."""
        from coral.models.layers import Attention
        attn = Attention(512, 64, 8, 8, causal=False)
        assert attn.qkv_proj.bias is None
        assert attn.o_proj.bias is None

    def test_qkv_proj_shape(self):
        """QKV projection shape: hidden_size → (num_heads + 2*num_kv_heads)*head_dim."""
        from coral.models.layers import Attention
        attn = Attention(512, 64, 8, 8, causal=False)
        # (8 + 2*8) * 64 = 24 * 64 = 1536
        assert attn.qkv_proj.weight.shape == (1536, 512)

    def test_o_proj_shape(self):
        """Output projection: num_heads*head_dim → hidden_size."""
        from coral.models.layers import Attention
        attn = Attention(512, 64, 8, 8, causal=False)
        assert attn.o_proj.weight.shape == (512, 512)
