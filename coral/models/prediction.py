"""Predictive coding components for CORAL v3 Phase 1.

PredictionNet: Maps H-state to a prediction of L-state (mu_L).
PrecisionNet:  Produces per-dimension precision (inverse variance) from L-state.

Both use CastedLinear (no bias) so they inherit the same dtype-casting and
initialisation behaviour as every other linear layer in the model.
"""

import torch
import torch.nn.functional as F
from torch import nn

from coral.models.layers import CastedLinear


class PredictionNet(nn.Module):
    """Maps H-module hidden state to a prediction of L-module hidden state.

    Two-layer MLP: h_dim → l_dim*2 → l_dim.
    CastedLinear (no bias), GELU activation.

    Designed for the general case h_dim != l_dim so it will work for the N=3
    hierarchy without modification.  In the N=2 case both dims equal hidden_size.
    """

    def __init__(self, h_dim: int, l_dim: int) -> None:
        super().__init__()
        self.fc1 = CastedLinear(h_dim, l_dim * 2, bias=False)
        self.fc2 = CastedLinear(l_dim * 2, l_dim, bias=False)

    def forward(self, z_H: torch.Tensor) -> torch.Tensor:
        """Predict L's state from H's state.

        Args:
            z_H: H-module hidden states [B, seq_len, h_dim].

        Returns:
            mu_L: H's prediction of L's state [B, seq_len, l_dim].
        """
        return self.fc2(F.gelu(self.fc1(z_H)))


class PrecisionNet(nn.Module):
    """Produces per-dimension precision (inverse variance) from L-module state.

    Two-layer MLP: dim → dim → dim.
    CastedLinear (no bias), GELU activation, softplus + eps_min output.

    The eps_min = 0.01 floor ensures precision never reaches zero, preventing
    division-by-zero and keeping the free energy loss numerically stable.
    """

    EPS_MIN: float = 0.01

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc1 = CastedLinear(dim, dim, bias=False)
        self.fc2 = CastedLinear(dim, dim, bias=False)

    def forward(self, z_L: torch.Tensor) -> torch.Tensor:
        """Compute precision vector from L's state.

        Args:
            z_L: L-module hidden states [B, seq_len, dim].

        Returns:
            pi: Precision vector [B, seq_len, dim], always > EPS_MIN (i.e. > 0.01).
        """
        return F.softplus(self.fc2(F.gelu(self.fc1(z_L)))) + self.EPS_MIN
