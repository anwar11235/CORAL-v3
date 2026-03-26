"""Smoke test — 5 training steps on synthetic data, CPU only, no W&B, no torch.compile.

Run with:
    python scripts/smoke_test.py

Should complete in under 10 seconds and print per-step metrics.
Exit code 0 = pass, non-zero = fail.
"""

import sys
import os
# Ensure the project root is importable when running as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import torch
from torch.utils.data import DataLoader, IterableDataset

# Disable torch.compile for this test
import os
os.environ["DISABLE_COMPILE"] = "1"

from coral.models.coral_base import CoralConfig
from coral.training.act import CoralACT
from coral.training.losses import ACTLossHead, IGNORE_LABEL_ID
from coral.training.scheduler import cosine_schedule_with_warmup_lr_lambda

try:
    from adam_atan2 import AdamATan2
except (ImportError, ModuleNotFoundError):
    from torch.optim import AdamW as AdamATan2
    print("WARNING: adam_atan2 not available, falling back to AdamW")

# ---------------------------------------------------------------------------
# Tiny config
# ---------------------------------------------------------------------------

BATCH = 4
SEQ = 16
VOCAB = 12
NUM_STEPS = 5
LR = 1e-3

CFG = CoralConfig(
    batch_size=BATCH,
    seq_len=SEQ,
    vocab_size=VOCAB,
    hidden_size=64,
    num_heads=2,
    expansion=4.0,
    H_cycles=2,
    L_cycles=2,
    H_layers=1,
    L_layers=1,
    halt_max_steps=2,
    halt_exploration_prob=0.1,
    puzzle_emb_ndim=0,
    forward_dtype="float32",
)


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

class SyntheticDataset(IterableDataset):
    """Yields random batches indefinitely."""

    def __init__(self, num_batches: int):
        self.num_batches = num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = {
                "inputs": torch.randint(0, VOCAB, (BATCH, SEQ)),
                "labels": torch.randint(0, VOCAB, (BATCH, SEQ)),
                "puzzle_identifiers": torch.zeros(BATCH, dtype=torch.int32),
            }
            yield "synthetic", batch, BATCH


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== CORAL v3 Smoke Test ===")
    print(f"Config: hidden={CFG.hidden_size}, heads={CFG.num_heads}, "
          f"H_layers={CFG.H_layers}, L_layers={CFG.L_layers}, "
          f"halt_max={CFG.halt_max_steps}, batch={BATCH}, seq={SEQ}")
    print()

    # Model
    act = CoralACT(CFG)
    model = ACTLossHead(act, loss_type="stablemax_cross_entropy")
    model.train()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    # Optimizer
    optimizer = AdamATan2(model.parameters(), lr=LR, weight_decay=0.1)
    print(f"Optimizer: {type(optimizer).__name__}")

    # Data
    loader = DataLoader(SyntheticDataset(NUM_STEPS), batch_size=None)

    # Train
    carry = None
    losses = []
    t0 = time.time()

    for step, (set_name, batch, gbs) in enumerate(loader, start=1):
        if carry is None:
            carry = model.initial_carry(batch)

        carry, loss, metrics, _, _ = model(carry=carry, batch=batch, return_keys=[])

        scaled_loss = (1.0 / gbs) * loss
        scaled_loss.backward()

        # LR schedule (cosine with warmup)
        lr = cosine_schedule_with_warmup_lr_lambda(
            step, base_lr=LR, num_warmup_steps=2, num_training_steps=NUM_STEPS
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.step()
        optimizer.zero_grad()

        count = max(metrics["count"].item(), 1)
        acc = metrics["accuracy"].item() / count
        losses.append(loss.item())

        print(f"  step {step}/{NUM_STEPS}  loss={loss.item():.4f}  "
              f"acc={acc:.3f}  lr={lr:.2e}  "
              f"halted={carry.halted.sum().item()}/{BATCH}")

    elapsed = time.time() - t0
    print()
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"First loss: {losses[0]:.4f}  Last loss: {losses[-1]:.4f}")

    # Sanity checks
    assert all(l > 0 for l in losses), "All losses should be positive"
    assert not any(torch.isnan(torch.tensor(losses))), "NaN loss detected"
    print()
    print("All checks passed. Smoke test OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
