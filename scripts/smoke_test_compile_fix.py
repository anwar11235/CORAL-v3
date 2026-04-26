"""CPU smoke test: verify no dynamo recompile storms post-consolidation.

Runs a minimal CORAL v3 training loop with crystallization active.
Bootstrap completes at step 50 (crystal_bootstrap_steps=50); the test
then runs 100+ post-consolidation steps and verifies zero dynamo recompile
guard-failure events.

Usage:
    python scripts/smoke_test_compile_fix.py

Expected output:
    [smoke] PASS: 0 dynamo recompile guard events across 110 post-consolidation steps.
"""

import logging
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path so `coral` package is importable when
# this script is run directly (python scripts/smoke_test_compile_fix.py).
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

# Keep cache limit deliberately small so that hitting it is fatal during smoke.
if hasattr(torch._dynamo.config, "recompile_limit"):
    torch._dynamo.config.recompile_limit = 8      # torch 2.7+
else:
    torch._dynamo.config.cache_size_limit = 8     # torch 2.6

# ---------------------------------------------------------------------------
# Capture dynamo WARNING-level messages (recompile alerts, cache exhaustion)
# ---------------------------------------------------------------------------

class _RecompileCounter(logging.Handler):
    def __init__(self):
        super().__init__()
        self.recompile_msgs: list = []

    def emit(self, record):
        msg = self.format(record)
        lower = msg.lower()
        if "recompil" in lower or "cache miss" in lower or "guard fail" in lower or "cache size" in lower:
            self.recompile_msgs.append(msg)

_counter = _RecompileCounter()
_counter.setLevel(logging.WARNING)
for _logger_name in ("torch._dynamo", "torch._inductor"):
    _log = logging.getLogger(_logger_name)
    _log.addHandler(_counter)
    if _log.level == logging.NOTSET or _log.level > logging.WARNING:
        _log.setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Minimal model setup
# ---------------------------------------------------------------------------

from coral.models.coral_base import CoralConfig
from coral.models.coral_v3 import CoralV3Inner
from coral.training.act import CoralV3ACT
from coral.training.losses import CoralV3LossHead

HIDDEN = 64
NUM_HEADS = 4
VOCAB = 32
SEQ_LEN = 8
BATCH = 4
BOOTSTRAP_STEPS = 50
TOTAL_STEPS = BOOTSTRAP_STEPS + 160   # ≥100 post-consolidation steps guaranteed

cfg = CoralConfig(
    batch_size=BATCH,
    seq_len=SEQ_LEN,
    vocab_size=VOCAB,
    H_cycles=2,
    L_cycles=2,
    H_layers=2,
    L_layers=2,
    hidden_size=HIDDEN,
    num_heads=NUM_HEADS,
    expansion=4.0,
    rms_norm_eps=1e-5,
    halt_max_steps=2,
    halt_exploration_prob=0.1,
    forward_dtype="float32",
    puzzle_emb_ndim=0,
    num_puzzle_identifiers=0,
    use_predictive_coding=True,
    use_crystallization=True,
    moe_num_modes=4,
    crystal_proj_dim=16,
    # Buffer capacity 80: 80% threshold = 64 entries.  With halt_max_steps=2,
    # crystal recording fires every 2 outer steps (batch=4 → 2 entries/step),
    # so buffer hits 64 entries by step ~32 → consolidation triggers at
    # step 50 (first eligible) with the buffer sufficiently full.
    crystal_buffer_capacity=80,
    crystal_bootstrap_steps=BOOTSTRAP_STEPS,
    crystal_consolidation_interval=BOOTSTRAP_STEPS,
    lambda_moe_recon=0.1,
    lambda_moe_balance=0.01,
    codebook_size=16,
)


def _make_batch():
    inputs = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    labels = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    return {"inputs": inputs, "labels": labels}


torch.manual_seed(0)
inner_model = CoralV3ACT(cfg)
model = CoralV3LossHead(inner_model, loss_type="softmax_cross_entropy")

# Apply the fix: compile only H_level / L_level, not the whole model.
# Use backend="eager" on CPU (inductor requires a C compiler on Windows);
# on GPU the real training uses the default inductor backend.  The "eager"
# backend still runs dynamo tracing and applies guards — sufficient to verify
# that no recompile storms occur without needing a C toolchain.
_backend = "eager" if not torch.cuda.is_available() else "inductor"
_inner = inner_model.inner
_inner.H_level = torch.compile(_inner.H_level, backend=_backend)
_inner.L_level = torch.compile(_inner.L_level, backend=_backend)

model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

torch._dynamo.reset()

# Snapshot torch._dynamo counters before training
_counters_before = {k: dict(v) for k, v in torch._dynamo.utils.counters.items()}

consolidation_done = False
consolidation_step = TOTAL_STEPS   # tracks when first consolidation succeeded
counter_at_consolidation: dict = {}

print(f"[smoke] Running {TOTAL_STEPS} steps (bootstrap={BOOTSTRAP_STEPS})...")

carry = None
for step in range(1, TOTAL_STEPS + 1):
    batch = _make_batch()

    if carry is None:
        carry = model.initial_carry(batch)

    carry, loss, metrics, _, _ = model(carry=carry, batch=batch, return_keys=[])

    (loss / BATCH).backward()
    optimizer.step()
    optimizer.zero_grad()

    # Consolidation trigger
    should_consolidate = (
        step >= cfg.crystal_bootstrap_steps
        and (step - cfg.crystal_bootstrap_steps) % cfg.crystal_consolidation_interval == 0
    )
    if should_consolidate:
        is_first = not consolidation_done
        usage = inner_model.inner.consolidate_codebook(is_first_consolidation=is_first)
        if usage is not None and not consolidation_done:
            consolidation_done = True
            consolidation_step = step
            counter_at_consolidation = {k: dict(v) for k, v in torch._dynamo.utils.counters.items()}
            print(f"[smoke] step {step}: consolidation done, codebook_util={usage}")

    if step % 20 == 0:
        n_recompile_msgs = len(_counter.recompile_msgs)
        print(f"[smoke] step {step}/{TOTAL_STEPS}  loss={loss.item():.4f}  warning_msgs={n_recompile_msgs}")

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

_counters_after = {k: dict(v) for k, v in torch._dynamo.utils.counters.items()}

def _delta(before, after):
    """Count events that occurred in `after` but not in `before` (nested dicts)."""
    total = 0
    for cat, sub in after.items():
        for key, val in sub.items():
            total += val - before.get(cat, {}).get(key, 0)
    return total

def _post_consol_delta(at_consol, after):
    return _delta(at_consol, after)

total_guard_breaks = _delta(_counters_before, _counters_after)
post_consol_guard_breaks = _post_consol_delta(counter_at_consolidation, _counters_after) if consolidation_done else 0

print(f"\n[smoke] torch._dynamo counters (total new events): {total_guard_breaks}")
print(f"[smoke] torch._dynamo counters (post-consolidation): {post_consol_guard_breaks}")
print(f"[smoke] WARNING-level recompile messages captured:   {len(_counter.recompile_msgs)}")

if _counter.recompile_msgs:
    print("[smoke] Recompile warnings (first 3):")
    for m in _counter.recompile_msgs[:3]:
        print(f"  {m[:300]}")

POST_CONSOLIDATION_STEPS = TOTAL_STEPS - consolidation_step

if POST_CONSOLIDATION_STEPS < 100:
    print(f"\n[smoke] INCONCLUSIVE: consolidation fired at step {consolidation_step}, only {POST_CONSOLIDATION_STEPS} post-consolidation steps (<100). Increase TOTAL_STEPS or reduce crystal_buffer_capacity.")
    sys.exit(2)
elif len(_counter.recompile_msgs) == 0:
    print(f"\n[smoke] PASS: 0 dynamo recompile warnings across {POST_CONSOLIDATION_STEPS} post-consolidation steps.")
    sys.exit(0)
else:
    print(f"\n[smoke] FAIL: {len(_counter.recompile_msgs)} dynamo recompile warning(s) detected.")
    sys.exit(1)
