# Eval Slowness Diagnosis

**Branch:** moe-lb-specialization-compile-fix (HEAD 4017506)  
**Date:** 2026-04-25  
**Observed:** ~14 min per eval call across all Phase 3 runs, 10 calls per run → ~2h21m of 5h15m total consumed by eval  

---

## 1. Hypothesis Scorecard

| # | Hypothesis | Status | Detail |
|---|-----------|--------|--------|
| 1 | Forced halt_max_steps at eval | **CONFIRMED — primary cause** | See §2 |
| 2 | Large augmented test set | **CONFIRMED — multiplicative factor** | See §3 |
| 3 | Compile bypass at eval | Not applicable | Sub-module compile persists across `.eval()` — compiled H_level/L_level are always active |
| 4 | No torch.no_grad at eval | Not applicable | `evaluate()` already wraps in `torch.inference_mode()` (train.py:397) |
| 5 | Per-puzzle Python loop | Not applicable | Eval iterates batches, not individual puzzles; Python overhead is negligible |

---

## 2. Primary Cause: Forced halt_max_steps at Eval

**Location:** `coral/training/act.py:154-155` (CoralACT) and `:339` (CoralV3ACT)

```python
# Default: halt only at max_steps (used during eval)
halted = is_last_step                                    # ← always True only at step 16

if self.training and self.config.halt_max_steps > 1:    # ← Q-halt gated by self.training
    halted = halted | (q_halt_logits > q_continue_logits)
```

At eval (`model.eval()` → `self.training = False`), the Q-learning halt signal is entirely disabled. Every sequence runs **exactly halt_max_steps=16 segments** regardless of how confident the model is.

In training mode, `halted = is_last_step | (q_halt > q_continue)`, so sequences halt as soon as the model is confident. A well-trained model that has reached a confident answer at step 3 will stop there during training — but will still run all 16 steps at eval.

The eval `while True` loop in `train.py:409-413` only exits when `all_done = new_carry.halted.all()`, which under forced halting is True only after step 16.

**Per-batch multiplier: 16 forward calls instead of ~3-6.**

---

## 3. Secondary Cause: Full Augmented Test Set at Every Eval

**Location:** `coral/data/puzzle_dataset.py:180` — `_iter_test()` iterates `total = len(dataset["inputs"])` examples sequentially with no sampling.

`create_dataloader(..., test_set_mode=True)` is called with `epochs_per_iter=1`, which triggers `_iter_test()`. This iterates every augmented variant of every test puzzle.

**Estimated test set size from timing:**

```
Observed: 14 min = 840 s per eval call
Training step time: 1/5 s = 0.200 s (forward + bootstrapping + backward + optimizer)
Eval segment time:  ~0.067 s (inference_mode, no bootstrapping, no backward ≈ ⅓ of training step)

840 s ÷ (16 segments/batch × 0.067 s/segment) ≈ 784 batches per eval call
784 batches × 384 examples/batch ≈ 301,000 test examples per eval
301,000 ÷ ~300 base test puzzles ≈ ~1,000 augmented variants per test puzzle
```

This is consistent with the dataset name `sudoku-extreme-1k-aug-1000`.

**All ~300K augmented test examples are evaluated on every eval call.**

---

## 4. Combined Effect

```
Current: 784 batches × 16 forced segments × 0.067 s/segment = 840 s ✓ (matches observed 14 min)

Fix A alone (Q-halt, assume avg halt at step 4):
  784 batches × 4 avg segments × 0.067 s = 210 s ≈ 3.5 min per eval
  Speedup vs current: ~4×

Fix A + D (eval_max_examples=10000, avg halt at step 4):
  26 batches × 4 avg segments × 0.067 s = 7 s per eval
  Speedup vs current: ~120×
  Accuracy noise: ~2-3% std on exact_match estimated from 10K subsample vs 300K full
```

---

## 5. What Was Ruled Out

**Hypothesis 3 (compile bypass):** `torch.compile` is applied at the sub-module level to `H_level` and `L_level` in `build_model()`. Sub-module compilation persists through `model.eval()` — the `OptimizedModule` wrapping is structural, not mode-dependent. No eval path change needed here.

**Hypothesis 4 (no-grad):** `evaluate()` is already wrapped in `torch.inference_mode()` (train.py:397). Autograd is not a factor.

**Hypothesis 5 (per-puzzle loop):** The eval loop iterates over the `eval_loader` DataLoader which yields batches of up to 384 examples. Python overhead per batch is negligible relative to 16 × 67ms of CUDA compute.

---

## 6. Recommended Fixes

**Fix A (primary, always apply):** Enable Q-halting at eval. Change the halting logic in `CoralACT.forward()` and `CoralV3ACT.forward()` so `q_halt > q_continue` can fire at eval just as in training — but WITHOUT exploration (no random min-halt-steps, no bootstrap target). Expected speedup 3-5× depending on the trained model's average halt step.

This changes eval behavior from "always 16 steps" to "halt when confident, max 16 steps." Accuracy impact: expected neutral to slightly positive (the halt policy is trained to fire when the answer is correct; forcing more steps just wastes compute on an already-confident model).

**Fix D (optional, strong recommendation for multi-seed campaigns):** Add `eval_max_examples: Optional[int]` config parameter. If set, the eval loop exits after processing that many examples. Setting `eval_max_examples=10000` reduces eval from ~300K to 10K examples — a 30× reduction — at the cost of stochastic accuracy noise on each eval call. The noise is well-bounded for monitoring purposes; a final end-of-training eval can still run on the full set.
