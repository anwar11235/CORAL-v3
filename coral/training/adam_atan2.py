"""Pure PyTorch implementation of AdamATan2 (scale-invariant Adam).

Based on the original adam-atan2 package. Replaces the standard Adam update
m / (sqrt(v) + eps) with atan2(m, sqrt(v)), bounding the update magnitude to
[-pi/2, pi/2] regardless of gradient scale. No bias correction — atan2 is
scale-invariant so bias correction is unnecessary. Uses decoupled weight decay.
"""

from typing import Tuple, Union

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


class AdamATan2(Optimizer):
    """Pure PyTorch implementation of AdamATan2 (scale-invariant Adam).

    Replaces the standard Adam update `m / (sqrt(v) + eps)` with `atan2(m, sqrt(v))`.
    This bounds the update magnitude to [-pi/2, pi/2] regardless of gradient scale.
    Uses decoupled weight decay (like AdamW).
    No bias correction — atan2 is scale-invariant so bias correction is unnecessary.
    """

    def __init__(
        self,
        params,
        lr: Union[float, Tensor] = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 1e-2,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamATan2 does not support sparse gradients")

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = torch.zeros((), dtype=torch.float32, device=p.device)
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state["step"] += 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # Update biased first and second moment estimates (no bias correction)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # atan2 update — scale invariant, magnitude bounded to [-pi/2, pi/2]
                update = torch.atan2(exp_avg, exp_avg_sq.sqrt())

                # Decoupled weight decay
                p.mul_(1 - lr * weight_decay)

                # Apply update
                p.add_(update, alpha=-lr)

        return loss
