"""Sparse columnar routing — replaces monolithic TransformerBlocks with S columns + index-select router.

Strategy C (index-select routing) benchmarked at 1.6× overhead vs monolithic on A100
(dim=384, S=8, k=2, B=256, seq=81 Sudoku benchmark, March 2026):

  Strategy             Fwd+Bwd (ms)  Inference (ms)  Peak Memory (MB)
  Monolithic (none)    21.0          6.8             1,321
  A: soft mask         88.4          29.4            4,325
  B: batch reorg       55.9          12.6            1,199
  C: index-select      33.8          9.9             1,188   ← chosen
  Hybrid               136.0         16.8            5,716

S columns, k active per sample:
  - Each column is a TransformerBlock with reduced FFN expansion
  - Router: CastedLinear(dim, S, bias=False) applied to mean-pooled input
  - Learnable temperature (clamped to [0.1, 10.0])
  - index_add_ scatter-back weighted by softmax over top-k logits

ColumnarTransformerBlock   — returns (hidden_states, routing_logits [B, S])
ColumnarReasoningModule    — returns (hidden_states, all_routing_logits: list of [B, S])
"""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from coral.models.layers import CastedLinear, CosSin
from coral.models.transformer_block import TransformerBlock, TransformerBlockConfig


class ColumnarTransformerBlock(nn.Module):
    """Replaces a single TransformerBlock with S smaller columns + index-select router.

    Each column is a full TransformerBlock (same hidden_size, same num_heads) with a
    proportionally reduced FFN expansion:
        col_expansion = max(1.0, expansion * 2 / S)
    For the default expansion=4, S=8 this gives col_expansion=1.0.

    Routing is computed from the mean-pooled input; top-k columns receive the full
    sequence, compute independently, and their outputs are scatter-added back.

    Args:
        config: Full-block TransformerBlockConfig (dim / heads / expansion for the parent block).
        S:      Number of columnar sub-modules.
        k:      Number of active columns per sample per forward call.
    """

    def __init__(self, config: TransformerBlockConfig, S: int = 8, k: int = 2) -> None:
        super().__init__()
        if k > S:
            raise ValueError(f"k={k} must be <= S={S}")
        self.S = S
        self.k = k

        col_expansion = max(1.0, config.expansion * 2 / S)
        col_config = TransformerBlockConfig(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            expansion=col_expansion,
            rms_norm_eps=config.rms_norm_eps,
        )
        self.columns = nn.ModuleList([TransformerBlock(col_config) for _ in range(S)])

        # Lightweight router: single linear, no bias, LeCun init from CastedLinear
        self.router = CastedLinear(config.hidden_size, S, bias=False)

        # Temperature for routing sharpness; clamped to [0.1, 10.0] in forward
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(
        self,
        cos_sin: Optional[CosSin],
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dispatch hidden_states through the top-k columns and scatter results back.

        Args:
            cos_sin:       RoPE (cos, sin) tuple, or None for learned pos encodings.
            hidden_states: [B, seq_len, hidden_size]

        Returns:
            output:         [B, seq_len, hidden_size] — weighted sum of active column outputs.
            routing_logits: [B, S]                   — raw router logits for load-balancing loss.
        """
        B, seq, D = hidden_states.shape

        # Clamp temperature to avoid instability
        temp = self.temperature.clamp(0.1, 10.0)

        # Route from mean-pooled representation
        routing_logits = self.router(hidden_states.mean(dim=1)) / temp  # [B, S]
        topk_vals, topk_idx = routing_logits.topk(self.k, dim=-1)       # [B, k]
        weights = F.softmax(topk_vals, dim=-1)                           # [B, k]

        # Flatten top-k assignments: each sample contributes k entries
        flat_idx = topk_idx.reshape(-1)                                  # [B*k]
        flat_weights = weights.reshape(-1)                               # [B*k]
        sample_idx = (
            torch.arange(B, device=hidden_states.device)
            .unsqueeze(1)
            .expand(B, self.k)
            .reshape(-1)
        )  # [B*k]

        result = torch.zeros_like(hidden_states)  # [B, seq, D]

        for s in range(self.S):
            col_mask = flat_idx == s
            if not col_mask.any():
                continue
            entries = col_mask.nonzero(as_tuple=True)[0]   # indices into flat arrays
            src_samples = sample_idx[entries]               # [n] — which batch samples
            src_weights = flat_weights[entries]             # [n]
            sub_batch = hidden_states[src_samples]          # [n, seq, D]
            col_out = self.columns[s](cos_sin=cos_sin, hidden_states=sub_batch)  # [n, seq, D]
            result.index_add_(
                0,
                src_samples,
                src_weights.unsqueeze(-1).unsqueeze(-1) * col_out,
            )

        return result, routing_logits


class ColumnarReasoningModule(nn.Module):
    """Drop-in replacement for ReasoningModule that uses ColumnarTransformerBlocks.

    Forward pass:
        1. hidden_states = hidden_states + input_injection
        2. For each ColumnarTransformerBlock:
               hidden_states, routing_logits = block(cos_sin, hidden_states)
               collect routing_logits

    Args:
        config:     TransformerBlockConfig for the full block (column width is derived internally).
        num_layers: Number of blocks in this module.
        S:          Number of columns per block.
        k:          Number of active columns per sample.
    """

    def __init__(
        self,
        config: TransformerBlockConfig,
        num_layers: int,
        S: int = 8,
        k: int = 2,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [ColumnarTransformerBlock(config, S=S, k=k) for _ in range(num_layers)]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_injection: torch.Tensor,
        cos_sin: Optional[CosSin] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Run the columnar reasoning module.

        Args:
            hidden_states:   [B, seq_len, hidden_size]
            input_injection: [B, seq_len, hidden_size] — added before the blocks.
            cos_sin:         Optional RoPE (cos, sin).

        Returns:
            hidden_states:      [B, seq_len, hidden_size]
            all_routing_logits: list of [B, S] — one per block, for load-balancing loss.
        """
        hidden_states = hidden_states + input_injection
        all_routing_logits: List[torch.Tensor] = []
        for layer in self.layers:
            hidden_states, routing_logits = layer(cos_sin=cos_sin, hidden_states=hidden_states)
            all_routing_logits.append(routing_logits)
        return hidden_states, all_routing_logits
