"""Phase 3b Soft MoE Crystallization components.

SpatialMoECodebook:
  K_modes full-spatial [seq_len, l_dim] expert templates + 1 passthrough expert.
  Softmax router over K+1 experts; soft blend z_L_out = w_pt*z_L_rec + (1-w_pt)*z_bypass.
  Unweighted L_recon and always-active L_lb prevent passthrough-dominance equilibrium.

CrystallizationBuffer:
  Ring buffer of (recognition_key, pooled_z_L, spatial_z_L) triples on CPU.
  Used during bootstrap (steps 0..crystal_bootstrap_steps) to collect spatial z_L for
  k-means consolidation. Disabled after first consolidation (consolidate_spatial).
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from coral.models.layers import CastedLinear
from coral.models.coral_base import CoralConfig


# ---------------------------------------------------------------------------
# SpatialMoECodebook (Phase 3b)
# ---------------------------------------------------------------------------


class SpatialMoECodebook(nn.Module):
    """Spatially-structured Soft MoE codebook for Phase 3b.

    Replaces the Phase 3a pooled-vector codebook with K_modes
    full-spatial [seq_len, l_dim] expert templates plus one passthrough expert.
    A router MLP computes softmax weights over all K+1=33 experts; the convex
    combination is returned as z_bypass. The soft blend with z_L_rec happens
    in CoralV3Inner (Session 2).

    Reconstruction loss (moe_losses) is UNWEIGHTED — codebook_values always
    receive gradient from L_recon regardless of router weights, making
    w_pt=1.0 an unstable equilibrium (anti-passthrough-dominance mechanism).

    Args:
        config:  CoralConfig — reads hidden_size, crystal_proj_dim, moe_num_modes.
        seq_len: Sequence length of the puzzle (81 for Sudoku-Extreme).
    """

    def __init__(self, config: CoralConfig, seq_len: int) -> None:
        super().__init__()
        l_dim = config.hidden_size
        proj_dim = config.crystal_proj_dim
        key_dim = proj_dim * 2
        K_modes = config.moe_num_modes

        self.K_modes = K_modes
        self.seq_len = seq_len
        self.l_dim = l_dim
        self.key_dim = key_dim

        # Projections — CastedLinear (no bias), same design as Phase 3a recognition key
        self.proj_h = CastedLinear(l_dim, proj_dim, bias=False)
        self.proj_l = CastedLinear(l_dim, proj_dim, bias=False)

        # Spatial mode templates [K_modes, seq_len, l_dim]
        # Overwritten by k-means consolidation at step crystal_bootstrap_steps.
        self.codebook_values = nn.Parameter(
            torch.randn(K_modes, seq_len, l_dim) * 0.02
        )

        # Matching keys for consolidation writeback (pooled key → codebook_keys init)
        self.codebook_keys = nn.Parameter(
            torch.randn(K_modes, key_dim) * 0.02
        )

        # Router MLP: key_dim → 64 → K_modes+1 (standard nn.Linear init)
        self.router_mlp = nn.Sequential(
            nn.Linear(key_dim, 64, bias=True),
            nn.GELU(),
            nn.Linear(64, K_modes + 1, bias=True),
        )

        # Bootstrap mask: when True forward() forces w_pt=1.0 (no codebook usage)
        self._bootstrap_mask_active: bool = False

    def forward(
        self, z_H: torch.Tensor, z_L: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute routing weights, codebook mixture, and recognition key.

        Args:
            z_H: [B, S, D] H-module state.
            z_L: [B, S, D] L-module carry state.

        Returns:
            w:        [B, K_modes+1] softmax routing weights; last entry = passthrough weight.
            z_bypass: [B, S, D]     weighted sum of codebook_values (codebook-only mixture).
            key:      [B, key_dim]  recognition key (for consolidation writeback).
        """
        # Recognition key: mean-pool projected H and L states → [B, key_dim]
        h_pool = self.proj_h(z_H).mean(dim=1)   # [B, proj_dim]
        l_pool = self.proj_l(z_L).mean(dim=1)   # [B, proj_dim]
        key = torch.cat([h_pool, l_pool], dim=-1)  # [B, key_dim]

        if self._bootstrap_mask_active:
            B = z_H.shape[0]
            K = self.K_modes + 1
            w = torch.zeros(B, K, device=z_H.device, dtype=z_H.dtype)
            w[:, -1] = 1.0  # all weight on passthrough; codebook receives none
            z_bypass = torch.zeros(
                B, self.seq_len, self.l_dim, device=z_H.device, dtype=z_H.dtype
            )
            return w, z_bypass, key

        # Route in float32 for numerical stability; cast back to model dtype
        logits = self.router_mlp(key.float())   # [B, K_modes+1]
        w = torch.softmax(logits, dim=-1).to(z_H.dtype)  # [B, K_modes+1]
        w_cb = w[:, : self.K_modes]             # [B, K_modes]

        # Weighted sum of spatial codebook templates → [B, S, D]
        z_bypass = torch.einsum(
            "bk,ksd->bsd", w_cb.float(), self.codebook_values
        ).to(z_H.dtype)

        return w, z_bypass, key

    def moe_losses(
        self,
        z_L_final: torch.Tensor,
        w: torch.Tensor,
        z_bypass: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute MoE auxiliary losses.

        L_recon is UNWEIGHTED (no (1-w_pt) multiplier). codebook_values always
        receive reconstruction gradient regardless of router state.

        L_lb is ALWAYS active (no sg(1-mean(w_pt)) suppressor).

        Args:
            z_L_final: [B, S, D] converged L-state; stop-gradient applied internally.
            w:         [B, K_modes+1] routing weights from forward().
            z_bypass:  [B, S, D] codebook mixture from forward().

        Returns:
            (L_recon, L_lb) — both scalar float tensors.
        """
        # Unweighted reconstruction loss — mean over [B, S, D]
        L_recon = (z_L_final.detach().float() - z_bypass.float()).pow(2).mean()

        # L_lb: non-normalized KL over all K+1 experts (passthrough + K_modes codebook).
        # `w` is a proper softmax distribution, so w.mean(dim=0) is also valid. No
        # normalization needed — penalizes both passthrough dominance and uniform-codebook
        # equilibrium (Option Y per Phase 3c spec §3.1).
        w_mean = w.float().mean(dim=0)           # [K_modes + 1] — includes passthrough
        K_plus_one = float(w_mean.shape[-1])
        target = torch.full_like(w_mean, 1.0 / K_plus_one)
        eps = 1e-10
        L_lb = (w_mean * (torch.log(w_mean + eps) - torch.log(target))).sum()

        return L_recon, L_lb

    def bootstrap_mask_router(self, active: bool) -> None:
        """Toggle the bootstrap passthrough mask.

        When active=True, forward() returns w_pt=1.0 (full passthrough). Used
        during steps 0..crystal_bootstrap_steps before k-means consolidation
        warm-starts codebook_values from actual spatial z_L statistics.
        """
        self._bootstrap_mask_active = active


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

    Uses pre-allocated CPU tensors with vectorised slice assignment so add() runs
    in O(B) without a Python loop per element, preventing throughput degradation
    after buffer saturation.

    Args:
        capacity: Maximum number of (key, value) pairs to store.
    """

    def __init__(self, capacity: int = 10000) -> None:
        self.capacity = capacity
        # Lazily allocated on first add() — we need key_dim and value_dim from inputs.
        self.keys: Optional[torch.Tensor] = None    # [capacity, key_dim]  on CPU
        self.values: Optional[torch.Tensor] = None  # [capacity, value_dim] on CPU
        # Spatial buffer for Phase 3b k-means consolidation. CPU only; never on GPU.
        # Memory: capacity × seq_len × l_dim × 4 bytes ≈ 1.65 GB for default capacity.
        self.spatial_buffer: Optional[torch.Tensor] = None  # [capacity, seq_len, l_dim] float32
        self.pointer: int = 0
        self.size: int = 0  # number of valid entries written, capped at capacity

    def _lazy_init(self, key_dim: int, value_dim: int) -> None:
        if self.keys is None:
            self.keys = torch.zeros(self.capacity, key_dim, dtype=torch.float32)
            self.values = torch.zeros(self.capacity, value_dim, dtype=torch.float32)

    def _lazy_init_spatial(self, seq_len: int, l_dim: int) -> None:
        if self.spatial_buffer is None:
            self.spatial_buffer = torch.zeros(
                self.capacity, seq_len, l_dim, dtype=torch.float32
            )

    @torch._dynamo.disable
    def add(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        z_L_spatial: Optional[torch.Tensor] = None,
    ) -> None:
        """Add a batch of (key, value) pairs via vectorised slice assignment.

        Single GPU→CPU transfer per call; no Python loop over batch elements.

        Args:
            keys:        [B, key_dim]   — recognition keys (may be on any device).
            values:      [B, value_dim] — pooled converged L-states (mean over seq dim).
            z_L_spatial: [B, S, D]     — full per-position L-states for spatial k-means.
                                         Optional; only stored when provided.
        """
        keys_cpu = keys.detach().to(dtype=torch.float32, device="cpu", non_blocking=True)
        values_cpu = values.detach().to(dtype=torch.float32, device="cpu", non_blocking=True)

        B = keys_cpu.shape[0]
        self._lazy_init(keys_cpu.shape[1], values_cpu.shape[1])

        # When B exceeds capacity, only the last capacity entries will be retained.
        # Clip to avoid repeated indices in the assignment below (undefined behavior).
        if B > self.capacity:
            keys_cpu = keys_cpu[-self.capacity :]
            values_cpu = values_cpu[-self.capacity :]
            if z_L_spatial is not None:
                z_L_spatial = z_L_spatial[-self.capacity :]
            B = self.capacity

        # Ring-buffer destination indices: [pointer, pointer+1, ..., pointer+B-1] % capacity
        indices = (torch.arange(B, dtype=torch.long) + self.pointer) % self.capacity

        self.keys[indices] = keys_cpu
        self.values[indices] = values_cpu

        if z_L_spatial is not None:
            z_L_cpu = z_L_spatial.detach().to(
                dtype=torch.float32, device="cpu", non_blocking=True
            )
            self._lazy_init_spatial(z_L_cpu.shape[1], z_L_cpu.shape[2])
            self.spatial_buffer[indices] = z_L_cpu

        self.pointer = int((self.pointer + B) % self.capacity)
        self.size = min(self.size + B, self.capacity)

    def __len__(self) -> int:
        return self.size

    def clear(self) -> None:
        """Reset the buffer (call after consolidation). Keeps tensors allocated for reuse."""
        self.pointer = 0
        self.size = 0
        # Zero spatial buffer to prevent stale entries from influencing the next bootstrap.
        if self.spatial_buffer is not None:
            self.spatial_buffer.zero_()

    def consolidate_spatial(
        self, k_modes: int, num_iterations: int = 100
    ) -> Optional[Tuple[torch.Tensor, int]]:
        """Run Euclidean k-means on buffered spatial z_L to initialise codebook_values.

        Forgy initialisation (k_modes random samples as starting centroids), then
        num_iterations of Lloyd's algorithm on CPU. All operations in float32.

        Args:
            k_modes:        Number of cluster centroids (= moe_num_modes, typically 32).
            num_iterations: Lloyd's iterations (default 100).

        Returns:
            (centroids, utilization) where centroids is [k_modes, seq_len, l_dim]
            float32 CPU tensor and utilization is the count of non-empty clusters
            after the final assignment, or None if the buffer has insufficient data.
        """
        if self.spatial_buffer is None or self.size < k_modes:
            return None

        N = self.size
        all_spatial = self.spatial_buffer[:N]           # [N, S, D]
        S, D = all_spatial.shape[1], all_spatial.shape[2]
        data_flat = all_spatial.view(N, -1).clone()     # [N, S*D] — clone to avoid aliasing

        # Forgy init: pick k_modes random samples as initial centroids
        perm = torch.randperm(N)[:k_modes]
        centroids_flat = data_flat[perm].clone()        # [K, S*D]

        assignments = torch.zeros(N, dtype=torch.long)
        SD = data_flat.shape[1]

        with torch.no_grad():
            for _ in range(num_iterations):
                # Euclidean distance matrix [N, K] via torch.cdist
                dists = torch.cdist(data_flat, centroids_flat)   # [N, K]
                assignments = dists.argmin(dim=1)                # [N]

                # Vectorised centroid update via scatter_add (no Python loop over k)
                new_centroids = torch.zeros(k_modes, SD, dtype=torch.float32)
                counts = torch.zeros(k_modes, dtype=torch.float32)

                new_centroids.scatter_add_(
                    0,
                    assignments.unsqueeze(1).expand(-1, SD),
                    data_flat,
                )
                counts.scatter_add_(0, assignments, torch.ones(N, dtype=torch.float32))

                filled = counts > 0
                new_centroids[filled] /= counts[filled].unsqueeze(1)
                # Keep old centroid for any empty cluster (avoids dead centroids resetting to 0)
                new_centroids[~filled] = centroids_flat[~filled]
                centroids_flat = new_centroids

        utilization = int((assignments.bincount(minlength=k_modes) > 0).sum().item())
        centroids = centroids_flat.view(k_modes, S, D)
        return centroids, utilization
