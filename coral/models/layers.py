"""Layer primitives for CORAL v3 — linear, embedding, attention, and FFN components."""

from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F

from coral.models.common import trunc_normal_init_

# flash_attn is only available on CUDA machines.  Import lazily so the module
# can be loaded on CPU (e.g., for structural tests or imports) without failing.
# The actual flash_attn_func is resolved once at first Attention.forward() call.
def _get_flash_attn_func():
    """Return a flash_attn_func-compatible callable.

    Preference order:
      1. flash_attn_interface (fa3, Hopper GPUs)
      2. flash_attn           (fa2, Ampere and older)
      3. F.scaled_dot_product_attention fallback (CPU / testing)

    The fallback accepts the same signature as flash_attn_func —
    q/k/v shaped [B, seq_len, num_heads, head_dim] — and wraps PyTorch's
    built-in SDPA (which expects [B, num_heads, seq_len, head_dim]).
    Functionally equivalent; used for CPU integration tests only.
    """
    try:
        from flash_attn_interface import flash_attn_func  # type: ignore[import]  # fa3
        return flash_attn_func
    except ImportError:
        pass
    try:
        from flash_attn import flash_attn_func  # type: ignore[import]  # fa2
        return flash_attn_func
    except ImportError:
        pass

    # CPU fallback: wrap F.scaled_dot_product_attention to match flash_attn's interface
    def _sdpa_fallback(q, k, v, causal=False):
        # flash_attn layout: [B, S, H, D] → SDPA layout: [B, H, S, D]
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=causal
        )
        return out.transpose(1, 2)  # back to [B, S, H, D]

    return _sdpa_fallback


CosSin = Tuple[torch.Tensor, torch.Tensor]


def _find_multiple(n: int, k: int) -> int:
    """Return the smallest multiple of k that is >= n."""
    return (-(n // -k)) * k


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split last dim in half and rotate: [-x2, x1]."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors.

    Args:
        q: Query tensor of shape [B, seq_len, num_heads, head_dim].
        k: Key tensor of shape [B, seq_len, num_kv_heads, head_dim].
        cos: Cosine cache of shape [seq_len, head_dim].
        sin: Sine cache of shape [seq_len, head_dim].

    Returns:
        Tuple of (rotated_q, rotated_k) with same shapes and dtypes as inputs.
    """
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)

    # cos/sin: [seq_len, head_dim] → unsqueeze head axis → [seq_len, 1, head_dim]
    q_rot = (q * cos.unsqueeze(-2)) + (_rotate_half(q) * sin.unsqueeze(-2))
    k_rot = (k * cos.unsqueeze(-2)) + (_rotate_half(k) * sin.unsqueeze(-2))

    return q_rot.to(orig_dtype), k_rot.to(orig_dtype)


class CastedLinear(nn.Module):
    """Linear layer that casts its weight to the input dtype on every forward pass.

    Weight is initialized with truncated LeCun normal (std = 1/sqrt(in_features)).
    Optional bias is zero-initialized.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(
            trunc_normal_init_(
                torch.empty(out_features, in_features),
                std=1.0 / (in_features ** 0.5),
            )
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


class CastedEmbedding(nn.Module):
    """Embedding layer that casts its weight to a target dtype on every forward pass.

    Weight is initialized with truncated normal of configurable std.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        init_std: float,
        cast_to: torch.dtype,
    ):
        super().__init__()
        self.cast_to = cast_to
        self.embedding_weight = nn.Parameter(
            trunc_normal_init_(
                torch.empty(num_embeddings, embedding_dim),
                std=init_std,
            )
        )

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return F.embedding(indices, self.embedding_weight.to(self.cast_to))


class RotaryEmbedding(nn.Module):
    """Precomputed RoPE cosine/sine caches.

    Frequencies follow the standard RoPE formula:
        inv_freq[i] = 1 / (base ^ (2i / dim))

    The cache is stored as a non-persistent buffer (not saved in state_dict).
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int,
        base: float = 10000.0,
        device=None,
    ):
        super().__init__()
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        positions = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(positions, inv_freq)  # [max_pos, dim//2]

        # Concatenate to get full head_dim: [max_pos, dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached: torch.Tensor
        self.sin_cached: torch.Tensor
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self) -> CosSin:
        """Return (cos_cached, sin_cached) for full sequence length."""
        return self.cos_cached, self.sin_cached


class Attention(nn.Module):
    """Multi-head self-attention with fused QKV projection and FlashAttention.

    Supports both FlashAttention 2 and 3 (fa3 preferred on Hopper GPUs).
    Non-causal by default (as in HRM). No bias on any projection.
    RoPE is applied to Q and K before attention.
    """

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_heads: int,
        num_key_value_heads: int,
        causal: bool = False,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.output_size = head_dim * num_heads
        self.causal = causal

        # Fused QKV: projects to [Q heads + K heads + V heads] * head_dim
        self.qkv_proj = CastedLinear(
            hidden_size,
            (num_heads + 2 * num_key_value_heads) * head_dim,
            bias=False,
        )
        self.o_proj = CastedLinear(self.output_size, hidden_size, bias=False)

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        B, seq_len, _ = hidden_states.shape

        qkv = self.qkv_proj(hidden_states)
        # Reshape to [B, seq_len, num_heads_total, head_dim]
        qkv = qkv.view(B, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)

        q = qkv[:, :, : self.num_heads]
        k = qkv[:, :, self.num_heads : self.num_heads + self.num_key_value_heads]
        v = qkv[:, :, self.num_heads + self.num_key_value_heads :]

        if cos_sin is not None:
            cos, sin = cos_sin
            q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        # flash_attn_func expects [B, seq_len, num_heads, head_dim]
        flash_attn_func = _get_flash_attn_func()
        attn_out = flash_attn_func(q=q, k=k, v=v, causal=self.causal)
        if isinstance(attn_out, tuple):  # fa3 returns (out, softmax_lse)
            attn_out = attn_out[0]

        attn_out = attn_out.view(B, seq_len, self.output_size)
        return self.o_proj(attn_out)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network with fused gate+up projection.

    intermediate_size = find_multiple(round(expansion * hidden_size * 2/3), 256)

    For hidden_size=512, expansion=4:
        round(4 * 512 * 2/3) = round(1365.33) = 1365 → next multiple of 256 = 1536

    No bias on any projection (matching HRM).
    """

    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)

        # gate and up are concatenated, split in forward
        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down_proj = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)
