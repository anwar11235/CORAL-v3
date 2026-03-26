"""Core utilities for CORAL v3 — initialization and normalization primitives."""

import math

import torch


def trunc_normal_init_(
    tensor: torch.Tensor,
    std: float = 1.0,
    lower: float = -2.0,
    upper: float = 2.0,
) -> torch.Tensor:
    """JAX-compatible truncated normal initialization (in-place).

    Unlike PyTorch's nn.init.trunc_normal_, this correctly compensates the
    standard deviation so that the initialized tensor has the requested std.
    Based on JAX/Flax's default truncated normal initializer.

    Args:
        tensor: Tensor to initialize in-place.
        std: Desired standard deviation of the (untruncated) base distribution.
        lower: Lower truncation bound in standard-deviation units.
        upper: Upper truncation bound in standard-deviation units.

    Returns:
        The initialized tensor (same object as input).
    """
    with torch.no_grad():
        if std == 0:
            tensor.zero_()
            return tensor

        sqrt2 = math.sqrt(2)

        # CDF values at truncation bounds (using erf representation of Gaussian CDF)
        erf_lower = math.erf(lower / sqrt2)
        erf_upper = math.erf(upper / sqrt2)
        # Probability mass within [lower, upper] for N(0, 1)
        prob_mass = (erf_upper - erf_lower) / 2

        # Standard normal PDF at the truncation bounds
        inv_sqrt_2pi = (2 * math.pi) ** -0.5
        phi_lower = inv_sqrt_2pi * math.exp(-0.5 * lower ** 2)
        phi_upper = inv_sqrt_2pi * math.exp(-0.5 * upper ** 2)

        # Variance of standard normal truncated to [lower, upper]:
        #   Var = 1 - (upper*phi(upper) - lower*phi(lower)) / Z
        #           - ((phi(lower) - phi(upper)) / Z)^2
        # Compensated std so that comp_std * N_trunc has variance = std^2
        trunc_var = (
            1.0
            - (upper * phi_upper - lower * phi_lower) / prob_mass
            - ((phi_lower - phi_upper) / prob_mass) ** 2
        )
        comp_std = std / math.sqrt(trunc_var)

        # Sample via inverse-CDF (erfinv) on the uniform distribution over [erf_lower, erf_upper]
        tensor.uniform_(erf_lower, erf_upper)
        tensor.erfinv_()
        tensor.mul_(sqrt2 * comp_std)
        # Clip to the actual truncation bounds after scaling
        tensor.clamp_(lower * comp_std, upper * comp_std)

    return tensor


def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    """RMSNorm as a pure function with no learnable parameters.

    Computes root-mean-square normalization in float32 precision and casts
    the result back to the input dtype.

    Args:
        hidden_states: Input tensor of any dtype, shape [..., hidden_size].
        variance_epsilon: Small constant added to variance for numerical stability.

    Returns:
        Normalized tensor with the same shape and dtype as input.
    """
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)
