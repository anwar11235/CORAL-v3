"""Post-Norm Transformer block — the core building unit for H-level and L-level modules."""

from dataclasses import dataclass

import torch
from torch import nn

from coral.models.common import rms_norm
from coral.models.layers import Attention, SwiGLU, CosSin


@dataclass
class TransformerBlockConfig:
    hidden_size: int
    num_heads: int
    expansion: float
    rms_norm_eps: float = 1e-5


class TransformerBlock(nn.Module):
    """Post-Norm Transformer block.

    Residual connections are normalized AFTER adding the sublayer output:
        x = RMSNorm(x + Attn(x))   # attention sub-layer
        x = RMSNorm(x + FFN(x))    # feed-forward sub-layer

    This is Post-Norm, NOT Pre-Norm.  Pre-Norm would be:
        x = x + Attn(RMSNorm(x))   (norm applied to x before the sublayer)

    No learnable parameters in the normalization — rms_norm is a pure function.
    """

    def __init__(self, config: TransformerBlockConfig) -> None:
        super().__init__()
        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False,
        )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # Post-Norm: normalize the sum of residual + sublayer output
        hidden_states = rms_norm(
            hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
            variance_epsilon=self.norm_eps,
        )
        hidden_states = rms_norm(
            hidden_states + self.mlp(hidden_states),
            variance_epsilon=self.norm_eps,
        )
        return hidden_states
