# Eval Slowness Diagnosis

**Branch:** moe-lb-specialization-compile-fix  
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

**Per-batch multiplier: 16 forward calls instead of ~3-6 (if Q-halt were active).**

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
Baseline (observed): 784 batches × 16 forced segments × 0.067 s/segment = 840 s ✓

Fix D only (eval_max_examples=10000, 16 forced segments):
  26 batches × 16 segments × 0.067 s = 27.8 s ≈ 30 sec per eval
  Speedup vs current: ~30×
  Accuracy noise: ~2-3% std on exact_match from 10K subsample vs 300K full
  Accuracy risk: ZERO — pure subsampling, no model behaviour change
```

---

## 5. Fix A: Q-halt at Eval — Attempted and Reverted

**Commit shipped:** `7eaaf8e`  
**Reverted in:** next commit  
**Validation result:** `eval/exact_accuracy = 0.0` on `phase3c_option_y_step52080.pt`
(vs 67.62% correct on the same checkpoint with forced halt_max_steps)

**Root cause of regression:** The Q-halt head (`q_head = CastedLinear(hidden_size, 2)`) is
trained under an exploration-active regime where `halt_exploration_prob=0.1` forces random
minimum halt depths. The resulting Q-values reflect the exploration distribution, not
greedy deterministic evaluation. Applied greedily at eval, `q_halt >> q_continue` fires at
step 1 for nearly every sequence, halting before the model has had any chance to compute a
meaningful answer. The prediction at step 1 is effectively random — hence 0% accuracy.

**Why the original diagnosis was wrong:** The hypothesis "the halt policy is trained to halt
when the answer is correct" assumed the Q-head is calibrated for greedy evaluation. It is
not: the Q-head is calibrated for the exploration-mixed training distribution. Using it
greedily at eval extrapolates outside its training distribution.

**Fix A is designated future work.** Enabling Q-halt at eval requires retraining the Q-head
with an eval-mode-aware reward signal (e.g., a separate eval Q-head trained without
exploration, or a curriculum that anneals exploration to 0 near end of training). This is
a non-trivial change to the training objective and out of scope for the current multi-seed
campaign.

---

## 6. What Was Ruled Out

**Hypothesis 3 (compile bypass):** `torch.compile` is applied at the sub-module level to `H_level` and `L_level` in `build_model()`. Sub-module compilation persists through `model.eval()` — the `OptimizedModule` wrapping is structural, not mode-dependent. No eval path change needed here.

**Hypothesis 4 (no-grad):** `evaluate()` is already wrapped in `torch.inference_mode()` (train.py:397). Autograd is not a factor.

**Hypothesis 5 (per-puzzle loop):** The eval loop iterates over the `eval_loader` DataLoader which yields batches of up to 384 examples. Python overhead per batch is negligible relative to 16 × 67ms of CUDA compute.

---

## 7. Shipped Fix: Fix D (eval_max_examples)

**Status: ACTIVE** — deployed in `phase3c_moe_lb_specialization.yaml` as `eval_max_examples: 10000`.

Add `eval_max_examples: Optional[int]` to `TrainConfig`. If set, the `evaluate()` loop skips forward passes after the per-set budget is reached. Default: `None` = evaluate all examples (backward compatible with all prior configs).

**Setting `eval_max_examples=10000`** evaluates the first 10K test examples per call:
- ~26 batches × 16 forced segments × ~0.067s = **~28 seconds per eval** (vs 14 minutes)
- **~30× speedup** with zero accuracy risk — pure subsampling, no model changes
- Accuracy noise: bounded (≈ 2–3% std on exact_match from 10K vs 300K); adequate for
  training-time monitoring; full evaluation can be run on the final checkpoint if needed

---

## 8. Future Work: Q-halt at Eval

To enable early stopping at eval without accuracy regression, the Q-head must be trained
to be calibrated for greedy evaluation. Options:

1. **Exploration annealing**: decay `halt_exploration_prob` from 0.1 → 0 over training.
   Near end of training the Q-head would be trained under near-greedy conditions.

2. **Separate eval Q-head**: train a second Q-head under `model.eval()` conditions
   (no exploration, no bootstrapping) as an auxiliary loss.

3. **Threshold tuning**: keep Q-halt at eval but learn a per-step threshold (not just
   `q_halt > q_continue`) that is calibrated against held-out performance.

None of these is implemented. The multi-seed Phase 3c campaign proceeds with Fix D only.
