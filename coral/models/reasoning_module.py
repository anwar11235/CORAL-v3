"""ReasoningModule — a stack of TransformerBlocks with additive input injection."""

from typing import List, Optional

import torch
from torch import nn

from coral.models.layers import CosSin
from coral.models.transformer_block import TransformerBlock


class ReasoningModule(nn.Module):
    """A stack of TransformerBlocks with an additive input injection at the entry.

    Forward pass:
        1. hidden_states = hidden_states + input_injection   (element-wise add)
        2. hidden_states = block_N(...(block_1(hidden_states))...)

    This is used for both the H-level and L-level modules in HRM.  The injection
    mechanism differs between levels (see coral_base.py), but the module itself is
    identical — the caller decides what to inject.
    """

    def __init__(self, layers: List[TransformerBlock]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_injection: torch.Tensor,
        cos_sin: Optional[CosSin] = None,
    ) -> torch.Tensor:
        """Run the reasoning module.

        Args:
            hidden_states: Current state tensor [B, seq_len, hidden_size].
            input_injection: Tensor added to hidden_states before the blocks [B, seq_len, hidden_size].
            cos_sin: Optional (cos, sin) tuple from RotaryEmbedding.

        Returns:
            Updated hidden_states [B, seq_len, hidden_size].
        """
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(cos_sin=cos_sin, hidden_states=hidden_states)
        return hidden_states
