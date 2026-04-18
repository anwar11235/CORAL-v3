# Phase 3a' — Crystallisation Bootstrap Fix

## Problem Fixed

Prior Phase 1+3 runs (happy-porpoise, impartial-sawfly, dynamic-bulldog) all failed on
performance.  A secondary correctness issue was also present: the crystallisation gate
trained on vacuous targets.  During the pre-consolidation period the random codebook never
matches converged L-states, so `target_confidence = (reconstruction_error < tol).float()`
is always 0.  The gate learns to always output 0 and cannot recover after consolidation
fires because the gradient signal for positive targets is absent.

## Changes

### `coral/models/coral_base.py`
- Added `crystal_bootstrap_steps: int = 5000` to `CoralConfig` (Phase 3 section).
  Controls how many training steps elapse before the first codebook consolidation and
  before gate BCE supervision begins.  `crystal_bootstrap_steps = 0` reproduces prior
  behaviour (gate active immediately, first consolidation fires at step 0 with full replace).

### `coral/models/crystallization.py`
- **`CrystallizationBuffer.consolidate()`**: new `is_first_consolidation: bool = False`
  parameter.  When `True`: requires ≥ 80% buffer fill before proceeding (returns `None`
  to defer if not); uses `ema_weight = 1.0` (full replace of random init) so the random
  codebook is discarded entirely rather than blended in.  When `False`: unchanged EMA
  weight of 0.1.
- **`crystallization_supervision_loss()`**: return type changed from `torch.Tensor` to
  `Tuple[torch.Tensor, torch.Tensor, torch.Tensor]` — `(bce_loss, mean_reconstruction_error,
  target_confidence_mean)`.  The extra two are detached scalars for logging.
- **`crystallization_diagnostics()`**: new function.  Computes reconstruction error and
  target confidence mean without a BCE loss.  Called during the bootstrap phase and at
  eval time so codebook quality is visible before the gate goes live.

### `coral/models/coral_v3.py`
- **`CoralV3Inner.__init__()`**: added `self._crystal_gate_active: bool` attribute.
  Initialised to `True` when `crystal_bootstrap_steps == 0` (backward compat), `False`
  otherwise.  Set externally by the training loop after the first successful consolidation.
- **`PredMetrics`**: added `crystal_reconstruction_error` and `crystal_target_confidence_mean`
  optional tensor fields for per-step diagnostics.
- **`_compute_crystal_supervision_loss()`**: return type changed to
  `Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]`.
  - When `_crystal_gate_active=True` and training: calls `crystallization_supervision_loss`,
    returns BCE loss + diagnostics.
  - Otherwise (bootstrap phase or eval): calls `crystallization_diagnostics`, returns
    `(None, recon_error, target_conf_mean)` — diagnostics without BCE gradient.
  - When `use_crystallization=False`: returns `(None, None, None)`.
- All four `_forward_*()` methods updated to unpack the 3-tuple and populate the new
  `PredMetrics` fields.
- **`consolidate_codebook()`**: new `is_first_consolidation: bool = False` parameter,
  forwarded to `CrystallizationBuffer.consolidate()`.  Buffer is now cleared only when
  consolidation succeeds (not on None return), so a deferred first consolidation doesn't
  lose accumulated pairs.

### `coral/training/act.py`
- `CoralV3ACT.forward()`: forwards `crystal_reconstruction_error` and
  `crystal_target_confidence_mean` from `PredMetrics` into the `outputs` dict so
  the loss head and downstream metrics can read them.

### `coral/training/losses.py`
- `CoralV3LossHead.forward()`: logs `crystal_reconstruction_error` and
  `crystal_target_confidence_mean` as training/eval metrics (alongside existing
  `crystal_bypass_count`, `crystal_confidence_mean`).

### `scripts/train.py`
- `TrainConfig`: added `crystal_bootstrap_steps: int = 5000` and
  `resume_from_checkpoint: Optional[str] = None`.
- `build_model()`: forwards `crystal_bootstrap_steps` and
  `crystal_consolidation_interval` (was already in CoralConfig but not forwarded) to
  `CoralConfig`.
- **`load_warmstart_checkpoint()`**: new function.  Loads a `.pt` state_dict with
  `strict=False`, logs missing and unexpected keys.  Missing keys are expected for new
  modules (RecognitionNetwork, CrystallizationBuffer parameters) when warm-starting from
  a Phase 1 checkpoint.  Optimizer state is not restored — optimizers start fresh.
- **Main loop — bootstrap-phase logic**:
  - `first_consolidation_done: bool` tracks whether the first full-replace consolidation
    has succeeded.  Persists across eval intervals within the run.
  - Consolidation now fires at `step >= crystal_bootstrap_steps` and every
    `crystal_consolidation_interval` steps thereafter (using modular arithmetic on
    `step - crystal_bootstrap_steps`).
  - First consolidation uses `is_first_consolidation=True` (returns `None` to defer if
    buffer < 80% full).  On success, sets `first_consolidation_done = True` and
    `inner._crystal_gate_active = True`.
  - New W&B metrics logged every training step when `use_crystallization=True`:
    - `train/crystal/buffer_fill` — fraction of ring buffer capacity used
    - `train/crystal/first_consolidation_done` — 0.0/1.0 phase indicator
  - New W&B metric logged at each consolidation: `train/crystal/codebook_utilisation_frac`.

### `configs/phase3a_crystal_warmstart.yaml`
- New config file for the validation run.  Sets `use_predictive_coding=true`,
  `use_crystallization=true`, `crystal_bootstrap_steps=5000`,
  `crystal_consolidation_interval=5000`, warm-start from the poetic-giraffe checkpoint.
  All other hyperparameters match poetic-giraffe for clean ablation.

---

## Assumptions

1. The compiled model (`torch.compile`) exposes `._orig_mod` for `load_state_dict`.
   PyTorch 2.x guarantees this via `OptimizedModule`.  If not present, `load_state_dict`
   is called directly on the wrapper (which also works in practice).

2. `state.model.model.inner` navigates `CoralV3LossHead → CoralV3ACT → CoralV3Inner`.
   This path is identical to the existing consolidation call in the prior training loop.

3. The buffer capacity in the config (`crystal_buffer_capacity: 10000`) is the denominator
   for the 80% fill guard.  With `global_batch_size=384` and `crystal_bootstrap_steps=5000`,
   the buffer fills to ~`5000 * (384/halt_max_steps)` ≈ 120 K entries — well above 10 K
   capacity.  The buffer will be full long before step 5000 and the fill guard will be
   satisfied on the first attempt.

---

## Launch Commands (Vast.ai A100 instance)

### 1. Upload the checkpoint

```bash
scp C:\Users\mauha\coral_v3_results\phase1\phase1_best_checkpoint_61pct.pt \
    <user>@<vast-ip>:<port>:/workspace/checkpoints/phase1_best_checkpoint_61pct.pt
```

Or use the Vast.ai web UI to upload.

### 2. SSH into the instance and prepare

```bash
cd /workspace/CORAL-v3
git pull   # make sure you have the Phase 3a changes
pip install -e .
mkdir -p /workspace/checkpoints
```

### 3. Launch training

```bash
OMP_NUM_THREADS=8 python scripts/train.py \
    --config-name phase3a_crystal_warmstart \
    data_path=/workspace/data/sudoku-extreme-1k-aug-1000 \
    2>&1 | tee /workspace/phase3a_crystal_warmstart.log
```

Or with explicit overrides to match exactly:

```bash
OMP_NUM_THREADS=8 python scripts/train.py \
    data_path=/workspace/data/sudoku-extreme-1k-aug-1000 \
    epochs=20000 eval_interval=2000 \
    lr=7e-5 puzzle_emb_lr=7e-5 \
    weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
    +use_predictive_coding=True \
    +use_crystallization=True \
    +crystal_bootstrap_steps=5000 \
    +crystal_consolidation_interval=5000 \
    +resume_from_checkpoint=/workspace/checkpoints/phase1_best_checkpoint_61pct.pt \
    seed=42 \
    2>&1 | tee /workspace/phase3a_crystal_warmstart.log
```

### 4. What to expect at launch

- `[CORAL-v3] Warm-starting from checkpoint: ...` — checkpoint loading begins
- `MISSING  model.inner.recognition_net.*` — expected (new module, ~30 lines)
- `MISSING  model.inner.crystal_buffer.*`   — expected (buffer params if any)
- `[CORAL-v3] Warm-start complete.`
- Training proceeds at target speed (~7–8 it/s with `torch.compile`)
- At step 5000: `[CORAL-v3] Codebook consolidation at step 5000 (first=True, usage=X.XX)`
- `[CORAL-v3] Crystal gate activated — BCE supervision enabled.`
- W&B run name auto-generated (coolname slug, e.g. `brave-falcon`)

### 5. Key W&B metrics to watch

| Metric | Expected trajectory |
|--------|-------------------|
| `train/crystal/buffer_fill` | Ramps 0→1 over first ~100 steps, stays at 1 thereafter |
| `train/crystal/first_consolidation_done` | 0.0 until step 5000, then 1.0 |
| `train/crystal/codebook_utilisation_frac` | At step 5000: aim > 0.30 (success criterion 3) |
| `train/crystal_reconstruction_error` | Should decrease after consolidation as gate trains |
| `eval/all/exact_accuracy` | Should stay near 61%; hard floor 58% (success criterion 1) |
| `eval/all/crystal_bypass_rate` | Should be > 0 by step 20000; hard floor 0.20 (criterion 2) |

---

## Pre-Committed Success Criteria

At step 20000:

1. `eval/all/exact_accuracy ≥ 0.58` (within 3pp of poetic-giraffe 61.07%) — hard threshold.
2. `eval/all/crystal_bypass_rate ≥ 0.20` — bypass firing on ≥ 20% of eval inputs.
3. `train/crystal/codebook_utilisation_frac ≥ 0.30` at most recent consolidation.
4. PC `prediction_error` and `q_halt_accuracy` remain in poetic-giraffe's range at steps 50k–52k equivalent.

**All four pass:** proceed to Phase 3b' (multi-head codebook / decrystallisation design).  
**1 or 4 fail:** crystallisation is interfering — diagnose before adding complexity.  
**2 or 3 fail, 1 passes:** mechanism not engaging but not hurting — gate/buffer/consolidation dynamics issue.
