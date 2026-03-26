"""Learning rate schedule: linear warmup + cosine decay."""

import math


def cosine_schedule_with_warmup_lr_lambda(
    current_step: int,
    *,
    base_lr: float,
    num_warmup_steps: int,
    num_training_steps: int,
    min_ratio: float = 0.0,
    num_cycles: float = 0.5,
) -> float:
    """Compute the learning rate for a given step.

    Linear warmup from 0 to base_lr over num_warmup_steps, then cosine
    decay to base_lr * min_ratio over the remaining steps.

    Args:
        current_step: Current optimizer step (0-indexed).
        base_lr: Peak learning rate reached at the end of warmup.
        num_warmup_steps: Number of warmup steps.
        num_training_steps: Total number of training steps.
        min_ratio: LR floor as a fraction of base_lr (default 0 → decays to 0).
        num_cycles: Number of cosine cycles (default 0.5 → single half-cosine).

    Returns:
        Absolute LR value for current_step.
    """
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))
    return base_lr * (min_ratio + max(0.0, (1.0 - min_ratio) * cosine_factor))
