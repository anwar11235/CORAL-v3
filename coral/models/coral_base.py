"""CORAL base inner model — faithful reproduction of HierarchicalReasoningModel_ACTV1_Inner.

This module implements the core hierarchical reasoning computation without the
ACT (Adaptive Computation Time) wrapper.  The ACT wrapper lives in coral/training/act.py.

Architecture:
  - Two ReasoningModules (H-level and L-level), each with 4 TransformerBlocks
  - Nested recurrence: H_cycles outer × L_cycles inner per segment
  - 1-step gradient: all but the final (L, H) step run under torch.no_grad()
  - Fixed initial states H_init, L_init — buffers, not learned parameters
  - Output from H-module; Q-values from first token of H-module output
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from coral.models.common import trunc_normal_init_
from coral.models.layers import CastedEmbedding, CastedLinear, CosSin, RotaryEmbedding
from coral.models.reasoning_module import ReasoningModule
from coral.models.sparse_embedding import CastedSparseEmbedding
from coral.models.transformer_block import TransformerBlock, TransformerBlockConfig


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class CoralConfig(BaseModel):
    """Configuration for the CORAL base inner model.

    Mirrors HierarchicalReasoningModel_ACTV1Config from HRM exactly.
    """

    batch_size: int
    seq_len: int

    vocab_size: int
    num_puzzle_identifiers: int = 0
    puzzle_emb_ndim: int = 0  # 0 = no puzzle embedding; 512 = 1 prepended token

    H_cycles: int = 2
    L_cycles: int = 2
    H_layers: int = 4
    L_layers: int = 4

    hidden_size: int = 512
    num_heads: int = 8
    expansion: float = 4.0

    pos_encodings: str = "rope"  # "rope" | "learned"
    rope_theta: float = 10000.0

    rms_norm_eps: float = 1e-5
    halt_max_steps: int = 16
    halt_exploration_prob: float = 0.1

    forward_dtype: str = "bfloat16"

    # Phase 1: predictive coding
    use_predictive_coding: bool = False
    lambda_pred: float = 0.1   # weight for precision-weighted prediction error loss
    lambda_pi: float = 0.01    # weight for precision regularisation loss

    # Phase 2: sparse columnar routing
    use_columnar_routing: bool = False
    num_columns: int = 8         # S — number of columnar sub-modules per block
    active_columns: int = 2      # k — active columns per sample per block
    lambda_balance: float = 0.01  # weight for load-balancing KL loss

    # Phase 3: recognition-gated crystallization
    use_crystallization: bool = False
    codebook_size: int = 256              # K — number of codebook entries
    crystal_proj_dim: int = 128           # projection dim for recognition key
    crystal_confidence_threshold: float = 0.8  # bypass fires when mean(confidence) > threshold
    crystal_buffer_capacity: int = 10000  # ring-buffer capacity for consolidation
    crystal_consolidation_interval: int = 10   # epochs between consolidation calls
    lambda_crystal: float = 0.1           # weight for crystallization supervision loss


# ---------------------------------------------------------------------------
# Carry (state between segments)
# ---------------------------------------------------------------------------


@dataclass
class InnerCarry:
    """Recurrent state passed between ACT segments.

    Both tensors are detached after each forward pass — no gradient flows
    between segments (deep supervision).
    """

    z_H: torch.Tensor  # [B, total_seq_len, hidden_size]
    z_L: torch.Tensor  # [B, total_seq_len, hidden_size]


# ---------------------------------------------------------------------------
# Inner model
# ---------------------------------------------------------------------------


class CoralInner(nn.Module):
    """CORAL base inner loop (no ACT wrapper).

    One forward call runs H_cycles × L_cycles recurrent steps, but only the
    very last L-step and H-step are in the computation graph (1-step gradient).
    All prior steps run under torch.no_grad().

    Injection rules:
        L_level: hidden_states = z_L + (z_H + input_embeddings)
        H_level: hidden_states = z_H + z_L

    Output:
        lm_head applied to z_H[:, puzzle_emb_len:]   (skip puzzle tokens)
        q_head  applied to z_H[:, 0]                 (first token only, float32)
    """

    def __init__(self, config: CoralConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype: torch.dtype = getattr(torch, config.forward_dtype)

        # Derived dimensions
        self.embed_scale = math.sqrt(config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        # Ceiling division: how many hidden-sized tokens needed to hold puzzle_emb_ndim values
        self.puzzle_emb_len: int = -(config.puzzle_emb_ndim // -config.hidden_size) if config.puzzle_emb_ndim > 0 else 0
        self.total_seq_len: int = config.seq_len + self.puzzle_emb_len

        # --- I/O projections ---
        self.embed_tokens = CastedEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            init_std=embed_init_std,
            cast_to=self.forward_dtype,
        )
        self.lm_head = CastedLinear(config.hidden_size, config.vocab_size, bias=False)

        # Q-head: special init — weight=0, bias=-5 (keeps Q-values near zero early in training)
        self.q_head = CastedLinear(config.hidden_size, 2, bias=True)
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore[union-attr]

        # --- Puzzle embedding (optional) ---
        # puzzle_emb_len tokens are prepended to every sequence; each token holds hidden_size values.
        # puzzle_emb_ndim may not be a multiple of hidden_size, so the last token can be partial
        # (padded with zeros in _input_embeddings).
        if config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(
                num_embeddings=config.num_puzzle_identifiers,
                embedding_dim=config.puzzle_emb_ndim,
                batch_size=config.batch_size,
                init_std=0,  # Zero-init puzzle embeddings
                cast_to=self.forward_dtype,
            )

        # --- Position encodings ---
        block_cfg = TransformerBlockConfig(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            expansion=config.expansion,
            rms_norm_eps=config.rms_norm_eps,
        )
        if config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=config.hidden_size // config.num_heads,
                max_position_embeddings=self.total_seq_len,
                base=config.rope_theta,
            )
        elif config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                num_embeddings=self.total_seq_len,
                embedding_dim=config.hidden_size,
                init_std=embed_init_std,
                cast_to=self.forward_dtype,
            )
        else:
            raise ValueError(f"Unknown pos_encodings: {config.pos_encodings!r}")

        # --- Reasoning modules ---
        self.H_level = ReasoningModule(
            layers=[TransformerBlock(block_cfg) for _ in range(config.H_layers)]
        )
        self.L_level = ReasoningModule(
            layers=[TransformerBlock(block_cfg) for _ in range(config.L_layers)]
        )

        # --- Fixed initial states (buffers, NOT parameters) ---
        # Initialized with truncated normal std=1; broadcast over batch and sequence dims at runtime.
        self.H_init: torch.Tensor
        self.L_init: torch.Tensor
        self.register_buffer(
            "H_init",
            trunc_normal_init_(torch.empty(config.hidden_size, dtype=self.forward_dtype), std=1.0),
            persistent=True,
        )
        self.register_buffer(
            "L_init",
            trunc_normal_init_(torch.empty(config.hidden_size, dtype=self.forward_dtype), std=1.0),
            persistent=True,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _cos_sin(self) -> Optional[CosSin]:
        """Return the RoPE cache, or None if using learned position embeddings."""
        if hasattr(self, "rotary_emb"):
            return self.rotary_emb()
        return None

    def _input_embeddings(
        self,
        inputs: torch.Tensor,
        puzzle_identifiers: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode token inputs into a scaled embedding sequence.

        Args:
            inputs: Integer token ids [B, seq_len].
            puzzle_identifiers: Per-sample puzzle IDs [B] (used when puzzle_emb_ndim > 0).

        Returns:
            Embedding tensor [B, total_seq_len, hidden_size], scaled by sqrt(hidden_size).
        """
        embedding = self.embed_tokens(inputs.to(torch.int32))  # [B, seq_len, D]

        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)  # [B, puzzle_emb_ndim]
            # Pad to a multiple of hidden_size if needed (last token may be partially filled)
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))
            # Reshape to [B, puzzle_emb_len, hidden_size] and prepend to token sequence
            embedding = torch.cat(
                (puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding),
                dim=-2,
            )

        if self.config.pos_encodings == "learned":
            # Scale by 1/sqrt(2) to maintain forward variance when adding two embeddings
            embedding = 0.707106781 * (
                embedding + self.embed_pos.embedding_weight.to(self.forward_dtype)
            )

        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int, device: torch.device = None) -> InnerCarry:
        """Allocate an uninitialised carry (will be reset on first segment)."""
        return InnerCarry(
            z_H=torch.empty(
                batch_size, self.total_seq_len, self.config.hidden_size,
                dtype=self.forward_dtype, device=device,
            ),
            z_L=torch.empty(
                batch_size, self.total_seq_len, self.config.hidden_size,
                dtype=self.forward_dtype, device=device,
            ),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: InnerCarry) -> InnerCarry:
        """Replace carry with initial states wherever reset_flag is True.

        Args:
            reset_flag: Boolean tensor [B]; True = this sequence was halted and needs reset.
            carry: Current inner carry to update.

        Returns:
            New carry with halted positions replaced by H_init / L_init.
        """
        # H_init / L_init: [hidden_size] → broadcast to [B, total_seq_len, hidden_size]
        flag = reset_flag.view(-1, 1, 1)
        return InnerCarry(
            z_H=torch.where(flag, self.H_init, carry.z_H),
            z_L=torch.where(flag, self.L_init, carry.z_L),
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        carry: InnerCarry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Run one segment (H_cycles × L_cycles recurrent steps) with 1-step gradient.

        Args:
            carry: Previous segment's detached carry (z_H, z_L).
            batch: Dict with at minimum "inputs" [B, seq_len] int tokens.
                   May also include "puzzle_identifiers" [B].

        Returns:
            Tuple of:
                new_carry:  InnerCarry with z_H, z_L both detached
                output:     Logits [B, seq_len, vocab_size] from lm_head(z_H)
                (q_halt, q_continue): float32 tensors [B] for ACT halting
        """
        cos_sin = self._cos_sin()
        input_embeddings = self._input_embeddings(
            batch["inputs"],
            batch.get("puzzle_identifiers"),
        )

        # ------------------------------------------------------------------
        # Run (H_cycles × L_cycles - 1) steps under no_grad.
        # The very last L-step and very last H-step are intentionally skipped
        # here so they can be run outside no_grad to build the computation graph.
        # ------------------------------------------------------------------
        with torch.no_grad():
            z_H, z_L = carry.z_H, carry.z_L

            for h_step in range(self.config.H_cycles):
                for l_step in range(self.config.L_cycles):
                    # Skip the last L-step (reserved for 1-step grad below)
                    is_last_l = (h_step == self.config.H_cycles - 1) and (l_step == self.config.L_cycles - 1)
                    if not is_last_l:
                        z_L = self.L_level(z_L, z_H + input_embeddings, cos_sin=cos_sin)

                # Skip the last H-step (reserved for 1-step grad below)
                if not (h_step == self.config.H_cycles - 1):
                    z_H = self.H_level(z_H, z_L, cos_sin=cos_sin)

        # Both z_H and z_L must be gradient-free at this point
        assert not z_H.requires_grad and not z_L.requires_grad

        # ------------------------------------------------------------------
        # 1-step gradient — ONLY these two ops are in the computation graph.
        # ------------------------------------------------------------------
        z_L = self.L_level(z_L, z_H + input_embeddings, cos_sin=cos_sin)
        z_H = self.H_level(z_H, z_L, cos_sin=cos_sin)

        # ------------------------------------------------------------------
        # Outputs
        # ------------------------------------------------------------------
        # Carry is always detached — no gradient flows between segments
        new_carry = InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())

        # Logits from H-module; strip the puzzle embedding prefix tokens
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]  # [B, seq_len, vocab_size]

        # Q-values from the first token position of z_H, always in float32
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)  # [B, 2]

        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])
