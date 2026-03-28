"""Recognition-gated crystallization — System 1 / System 2 bypass mechanism.

Well-learned patterns are recognised BEFORE L-module computation begins, allowing
the expensive recurrence to be skipped entirely and a stored codebook entry to be
substituted instead.  This mirrors the brain's distinction between deliberate
(System 2) computation and fast pattern retrieval (System 1).

Key design choices:
  - Recognition operates on pooled (mean over sequence) z_H and z_L → compact key
  - Codebook values are mean-pooled converged L-states, expanded back to [B, seq, l_dim]
  - Confidence gate is a small MLP trained via BCE supervision loss during training
  - Bypass ONLY fires during eval; training always runs full recurrence for gradients
  - CrystallizationBuffer stores CPU tensors to avoid long-term GPU memory use;
    consolidation temporarily moves them to the device, then frees them

Components:
  RecognitionNetwork          — computes keys, looks up codebook, predicts bypass confidence
  CrystallizationBuffer       — ring buffer of (key, pooled_z_L) pairs; offline codebook update
  crystallization_supervision_loss — BCE loss that trains the confidence gate
"""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from coral.models.layers import CastedLinear


# ---------------------------------------------------------------------------
# RecognitionNetwork
# ---------------------------------------------------------------------------


class RecognitionNetwork(nn.Module):
    """Lightweight network that decides whether to bypass L-module computation.

    Given the current H-module and L-module states it:
      1. Projects both states to a compact recognition space.
      2. Finds the nearest codebook entry by cosine similarity of the key.
      3. Predicts a bypass confidence score via a small MLP.

    The codebook stores converged L-states (values) and their associated keys.
    Keys are learned jointly with the confidence head during training via the
    crystallization_supervision_loss; values are updated offline via EMA in
    CrystallizationBuffer.consolidate().

    Args:
        h_dim:         Dimension of z_H (H-module state).
        l_dim:         Dimension of z_L (L-module state, also codebook value dim).
        codebook_size: Number of codebook entries (K).
        proj_dim:      Dimension of the recognition projection space.
    """

    def __init__(
        self,
        h_dim: int,
        l_dim: int,
        codebook_size: int = 256,
        proj_dim: int = 128,
    ) -> None:
        super().__init__()
        self.h_dim = h_dim
        self.l_dim = l_dim
        self.codebook_size = codebook_size
        self.proj_dim = proj_dim
        key_dim = proj_dim * 2

        # Project z_H and z_L to a compact recognition space (no bias)
        self.proj_h = CastedLinear(h_dim, proj_dim, bias=False)
        self.proj_l = CastedLinear(l_dim, proj_dim, bias=False)

        # Codebook: values — converged L-states (updated offline via EMA)
        self.codebook = nn.Parameter(torch.randn(codebook_size, l_dim) * 0.01)

        # Codebook keys — matched against recognition key (updated offline + via backprop)
        self.codebook_keys = nn.Parameter(torch.randn(codebook_size, key_dim) * 0.01)

        # Confidence head: predicts bypass safety from context + match quality
        self.confidence_head = nn.Sequential(
            CastedLinear(key_dim + 1, 64, bias=True),
            nn.GELU(),
            CastedLinear(64, 1, bias=True),
        )

    def compute_key(self, z_H: torch.Tensor, z_L: torch.Tensor) -> torch.Tensor:
        """Compute the recognition key by mean-pooling over the sequence dimension.

        Args:
            z_H: [B, seq_len, h_dim]
            z_L: [B, seq_len, l_dim]

        Returns:
            key: [B, proj_dim * 2]
        """
        h_proj = self.proj_h(z_H).mean(dim=1)   # [B, proj_dim]
        l_proj = self.proj_l(z_L).mean(dim=1)   # [B, proj_dim]
        return torch.cat([h_proj, l_proj], dim=-1)  # [B, proj_dim*2]

    def forward(
        self,
        z_H: torch.Tensor,
        z_L: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Look up nearest codebook entry and predict bypass confidence.

        Args:
            z_H: [B, seq_len, h_dim]
            z_L: [B, seq_len, l_dim]

        Returns:
            confidence:   [B]           — probability in (0, 1) that bypass is safe.
            nearest_code: [B, seq_len, l_dim] — codebook value expanded to sequence length.
            nearest_idx:  [B]           — index of the nearest codebook entry.
        """
        key = self.compute_key(z_H, z_L)  # [B, proj_dim*2]

        # Cosine similarity against all codebook keys
        similarities = F.cosine_similarity(
            key.unsqueeze(1),                  # [B, 1, proj_dim*2]
            self.codebook_keys.unsqueeze(0),   # [1, K, proj_dim*2]
            dim=-1,
        )  # [B, K]

        max_similarity, nearest_idx = similarities.max(dim=-1)  # [B]
        nearest_code = self.codebook[nearest_idx]  # [B, l_dim]

        # Expand to sequence length (each sequence position gets the same codebook vector)
        nearest_code = nearest_code.unsqueeze(1).expand(-1, z_L.shape[1], -1)  # [B, seq, l_dim]

        # Confidence prediction from context + match quality
        conf_input = torch.cat(
            [key, max_similarity.unsqueeze(-1)], dim=-1
        )  # [B, proj_dim*2 + 1]
        confidence = torch.sigmoid(self.confidence_head(conf_input).squeeze(-1))  # [B]

        return confidence, nearest_code, nearest_idx


# ---------------------------------------------------------------------------
# CrystallizationBuffer
# ---------------------------------------------------------------------------


class CrystallizationBuffer:
    """Ring buffer that collects (recognition_key, converged_z_L) pairs during training.

    Stored tensors are kept on CPU to avoid long-term GPU memory accumulation.
    Consolidation temporarily moves them to the training device, runs k-means-like
    assignment + EMA codebook updates, then returns.

    This mimics hippocampal replay during sleep-like consolidation in the brain:
    experiences (converged states encountered during training) are replayed to
    update long-term cortical representations (the codebook).

    Args:
        capacity: Maximum number of (key, value) pairs to store.
    """

    def __init__(self, capacity: int = 10000) -> None:
        self.capacity = capacity
        self.keys: list = []    # list of [proj_dim*2] CPU tensors
        self.values: list = []  # list of [l_dim] CPU tensors
        self.pointer: int = 0

    def add(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        """Add a batch of (key, value) pairs to the buffer.

        Args:
            keys:   [B, proj_dim*2] — recognition keys (may be on any device).
            values: [B, l_dim]      — pooled converged L-states (mean over seq dim).
        """
        keys_cpu = keys.detach().cpu()
        values_cpu = values.detach().cpu()
        B = keys_cpu.shape[0]
        for i in range(B):
            if len(self.keys) < self.capacity:
                self.keys.append(keys_cpu[i])
                self.values.append(values_cpu[i])
            else:
                self.keys[self.pointer] = keys_cpu[i]
                self.values[self.pointer] = values_cpu[i]
            self.pointer = (self.pointer + 1) % self.capacity

    def __len__(self) -> int:
        return len(self.keys)

    def consolidate(
        self,
        recognition_net: RecognitionNetwork,
        num_iterations: int = 100,
        device: str = "cpu",
    ) -> Optional[float]:
        """Update the codebook via offline k-means-like assignment + EMA.

        Requires at least 100 stored pairs; silently returns None otherwise.

        Each stored (key, value) pair is assigned to the nearest codebook key
        by cosine similarity.  For each cluster, the codebook entry (both value
        and key) is updated via EMA:
            new_entry = 0.9 * old_entry + 0.1 * cluster_mean

        Args:
            recognition_net: The RecognitionNetwork whose codebook to update.
            num_iterations:  Number of assignment + update rounds.
            device:          Device to use for the computation (should match
                             recognition_net's parameter device).

        Returns:
            Fraction of codebook entries that received at least one assignment
            in the final iteration, or None if the buffer was too small.
        """
        if len(self.keys) < 100:
            return None

        all_keys = torch.stack(self.keys).to(device)      # [N, proj_dim*2]
        all_values = torch.stack(self.values).to(device)  # [N, l_dim]

        K = recognition_net.codebook.shape[0]
        assignments: Optional[torch.Tensor] = None

        with torch.no_grad():
            for _ in range(num_iterations):
                # Assign each experience to nearest codebook key
                sims = F.cosine_similarity(
                    all_keys.unsqueeze(1),                         # [N, 1, key_dim]
                    recognition_net.codebook_keys.unsqueeze(0),    # [1, K, key_dim]
                    dim=-1,
                )  # [N, K]
                assignments = sims.argmax(dim=1)  # [N]

                # EMA update for each codebook entry
                for k_idx in range(K):
                    mask = assignments == k_idx
                    if mask.sum() > 0:
                        mean_value = all_values[mask].mean(dim=0)
                        mean_key = all_keys[mask].mean(dim=0)
                        recognition_net.codebook.data[k_idx] = (
                            0.9 * recognition_net.codebook.data[k_idx] + 0.1 * mean_value
                        )
                        recognition_net.codebook_keys.data[k_idx] = (
                            0.9 * recognition_net.codebook_keys.data[k_idx] + 0.1 * mean_key
                        )

        # Codebook usage: fraction of K entries with at least one assignment
        if assignments is not None:
            used = int(assignments.bincount(minlength=K).gt(0).sum().item())
            return float(used) / K
        return None

    def clear(self) -> None:
        """Empty the buffer (call after consolidation)."""
        self.keys.clear()
        self.values.clear()
        self.pointer = 0


# ---------------------------------------------------------------------------
# Supervision loss
# ---------------------------------------------------------------------------


def crystallization_supervision_loss(
    recognition_net: RecognitionNetwork,
    z_H: torch.Tensor,
    z_L_converged: torch.Tensor,
    tolerance: float = 0.05,
) -> torch.Tensor:
    """Train the confidence gate to predict when crystallization bypass is safe.

    Called after each full L-cycle during training (in the 1-step-grad section
    where z_L_converged is the actual converged L-state from full recurrence).

    The target is 1 if the nearest codebook entry is within tolerance of the
    actual converged state, and 0 otherwise.  This teaches the confidence gate
    to predict bypass accuracy without ever running a bypass during training.

    Args:
        recognition_net:  The RecognitionNetwork.
        z_H:              [B, seq_len, h_dim] — H-state before the final H update.
        z_L_converged:    [B, seq_len, l_dim] — actual converged L-state from full recurrence.
        tolerance:        Mean-squared reconstruction error threshold for "safe" bypass.

    Returns:
        Scalar BCE loss.
    """
    confidence, nearest_code, _ = recognition_net(z_H, z_L_converged)

    # Stop-gradient: the target is not differentiated — it labels whether bypass would be safe
    with torch.no_grad():
        reconstruction_error = (z_L_converged - nearest_code).pow(2).mean(dim=(1, 2))  # [B]
        target_confidence = (reconstruction_error < tolerance).float()

    return F.binary_cross_entropy(confidence, target_confidence)
