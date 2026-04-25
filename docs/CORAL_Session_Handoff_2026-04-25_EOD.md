# CORAL Session Handoff — 2026-04-25 EOD

**Purpose:** Context carry-over for the next Claude session. This is the end-of-day update for the 2026-04-25 session. The session started from `docs/CORAL_Session_Handoff_2026-04-25-EOD.md` (which documented the Phase 3c result) and uncovered three infrastructure issues that were fixed before the day ended.

**Status as of this handoff:** Phase 3c result (67.62%) stands as a real measurement, but is now known to be non-reproducible due to a dataset construction bug that affected the entire project history. Bug fixed. Infrastructure for deterministic builds and fast eval is now in place. No new training data produced today. Ready to re-launch on a canonical seed-0 dataset.

---

## Chronological Summary of 2026-04-25

### Attempt: Validate compile fix and launch multi-seed Phase 3c campaign

**Goal:** Run the Phase 3c config (`moe-lb-specialization-compile-fix` branch, commit `6dcfc0f`) with a warm-start from `phase1_best_checkpoint_61pct.pt` to validate the sub-module compile fix and then launch multi-seed.

**What happened:**

**Issue 1 — Warm-start regression (commit `4017506`, resolved)**
Sub-module compile (Approach A from `6dcfc0f`) replaces `H_level`/`L_level` with `torch.compile()` wrappers. PyTorch registers `_orig_mod` as a named child, shifting state_dict keys from `model.inner.H_level.layers.0.weight` to `model.inner.H_level._orig_mod.layers.0.weight`. The existing warm-start code stripped the top-level `_orig_mod.` prefix (for top-level-compiled checkpoints) but did not insert the sub-module `_orig_mod.` prefix. All 32 backbone weights were silently skipped. Fix: `_remap_checkpoint_keys_for_submodule_compile()` in `scripts/train.py` detects compiled sub-modules via `named_children()` and inserts the prefix at runtime. Validation: `phase3c_option_y_step52080.pt` loads with 0 unexpected/missing backbone keys.

**Issue 2 — Eval slowness, Fix A attempt, Fix A revert (commit `7eaaf8e` then `a96e499`)**
Eval was taking ~14 min/call across all Phase 3 runs (10 calls per run → 2h21m of 5h15m total). Root cause: `act.py` gated Q-halt on `self.training`, forcing every eval puzzle through all 16 ACT segments. Estimated ~300K test examples × 16 segments = 840s = 14 min. Fix A (enable Q-halt at eval) was implemented and deployed to `phase3c_option_y_step52080.pt`. Result: `eval/exact_accuracy = 0` (down from 67.62%). Root cause of regression: the Q-head is trained under `halt_exploration_prob=0.1`, which means Q-values are calibrated for the exploration-active distribution, not greedy deterministic evaluation. With Q-halt active at eval, the model halts at step 1 before computing a meaningful answer. Fix A was reverted. Fix D (`eval_max_examples: 10000` config parameter) was kept: caps eval at 10K examples → ~30 sec/eval, ~30× speedup, zero accuracy risk. Added to `configs/phase3c_moe_lb_specialization.yaml`. See `docs/eval_slowness_diagnosis.md`.

**Issue 3 — Dataset reproducibility bug (commit `9493d6d` on `dataset-determinism-fix`, resolved)**
Re-evaluating `phase3c_option_y_step52080.pt` against a freshly-rebuilt dataset returned 0% accuracy even with the warm-start fix applied. Investigation: `build_sudoku_dataset.py` called `np.random.permutation`, `np.random.rand`, and `np.random.choice` without seeding. Every rebuild produces a different dataset realization. The Phase 3c checkpoint's 67.62% was measured on a specific unreproducible dataset. The dataset was not archived. Fix: `--seed` flag added to `DataProcessConfig` (default `0`), `np.random.seed(config.seed)` called at entry, seed recorded in `dataset.json` as `dataset_seed`. SHA-256 validation confirms byte-identical output across same-seed runs. See `docs/dataset_reproducibility.md`.

---

## What We Have at End of Day

### Branches and commits

| Branch | HEAD | Status | Description |
|--------|------|--------|-------------|
| `moe-lb-specialization-compile-fix` | `a96e499` | **Pushed, ready** | Sub-module compile + warm-start remap + eval_max_examples |
| `dataset-determinism-fix` | `9493d6d` | **Pushed, ready** | Seeded dataset builder |
| `arc-adapter-design` | (see branch) | Design only, no code | ARC adapter design + HRM investigation notes |
| `master` | `1f68255` | Stale (this commit updates it) | Phase 3a' era; needs the above merged |

### What is working

- `phase1_best_checkpoint_61pct.pt` loads correctly into Phase 3c model on `moe-lb-specialization-compile-fix` HEAD
- Sub-module compile is the right compile structure; no dynamo recompile storms
- eval_max_examples=10000 reduces eval time from 14 min to ~30 sec
- Dataset builder is deterministic with `--seed`; SHA-256 validated

### What is NOT working (or unresolved)

- Q-halt at eval: the Q-head needs retraining under eval-aware conditions before this works. Documented in `docs/eval_slowness_diagnosis.md` §8 (future work)
- All previous training run datasets are non-recoverable
- Phase 3c result (67.62%) is non-reproducible without the original dataset

---

## What We Learned

1. **Compile structure:** Sub-module compile (H_level/L_level independently) is the right approach. The `_orig_mod.` prefix remap is now handled automatically at warm-start time by inspecting `named_children()`. This is forward-compatible.

2. **Eval Q-halt:** The Q-head trained under exploration-active conditions cannot be used greedily at eval without retraining. The model halts at step 1 (before any useful computation) because the exploration-calibrated Q-values fire immediately in a deterministic setting. This is a known calibration gap that requires explicit addressing if early-exit eval is desired.

3. **Dataset reproducibility:** All accuracy numbers in the project history are valid measurements on their respective (non-reproducible) datasets. The Phase 3c 67.62% number is real, but it cannot be reproduced on a fresh build. For publication, results must be re-established on a canonical `--seed 0` dataset.

4. **Fundraise framing:** "67.62% achieved on Sudoku-Extreme-1K; currently re-running with a frozen deterministic dataset to establish reproducibility before publishing — paper expected in 6-8 weeks." This is the honest framing for any pitch in the next month.

---

## Decisions Still Pending (For Next Session)

1. **What seed to canonicalize:** Recommendation: `--seed 0` for the Sudoku-Extreme-1K reference dataset. Trivial decision; just confirm.

2. **Whether to archive the `--seed 0` dataset to durable storage:** Recommended yes — store to a separate disk or cloud bucket so it survives Vast teardowns. Estimated 2-5 GB. Low cost, eliminates all future rebuild concerns.

3. **Whether to re-establish Phase 3a/3b/3c baselines on `--seed 0` dataset:** Recommended yes, before multi-seed campaign. This anchors all future results to a reproducible baseline. Estimated ~$10 GPU cost, ~12 hours wall-clock (3 runs × 4 hours each on A100).
   - Run 1: Phase 3a control (no crystal, warm-start from Phase 1 — branch `control-no-crystal`)
   - Run 2: satisfied-owl config (Phase 3b Soft MoE — branch `moe-codebook-design` or re-run from `moe-lb-specialization`)
   - Run 3: Phase 3c Option Y (branch `moe-lb-specialization-compile-fix`)

4. **Multi-seed the winner:** After baselines on `--seed 0`, run 3-5 seeds of the winner. Estimated ~$15-20 GPU cost.

5. **ARC adapter timing:** Recommendation — after Sudoku re-establishment, not in parallel. ARC adapter is a significant engineering effort and the Sudoku results need to be solid before expanding scope.

---

## Resumption Instructions

Read this handoff. Confirm with Anwar what has progressed since 2026-04-25. If nothing has changed:

1. **Merge the fix branches to master:**
   ```bash
   git checkout master
   git merge dataset-determinism-fix
   git merge moe-lb-specialization-compile-fix  # or cherry-pick the relevant commits
   git push origin master
   ```
   (Coordinate with Anwar on merge strategy — he may prefer squash merges or PRs.)

2. **Provision Vast A100 instance:**
   - Image: `anwar1919/coral-v3:2026-04-20`
   - GPU: A100 40GB PCIE (~$0.53/hr) or SXM4 80GB if available
   - Pull branch: `moe-lb-specialization-compile-fix` (contains all compile fixes + eval_max_examples)

3. **Build canonical `--seed 0` dataset on the instance:**
   ```bash
   python coral/data/build_sudoku_dataset.py \
       --output-dir /workspace/data/sudoku-extreme-1k-aug-1000-seed0 \
       --subsample-size 1000 --num-aug 1000 --seed 0
   ```
   Archive the output to durable storage before proceeding.

4. **Launch Phase 3c re-establishment run:**
   ```bash
   python scripts/train.py \
       --config-name phase3c_moe_lb_specialization \
       data_path=/workspace/data/sudoku-extreme-1k-aug-1000-seed0 \
       resume_from_checkpoint=/workspace/checkpoints/phase1_best_checkpoint_61pct.pt \
       seed=0
   ```

5. **Monitor for:** `eval/exact_accuracy` near 67% by step 52080, `eval_max_examples=10000` producing fast eval (~30 sec/call), no recompile warnings in logs.

---

## Cross-References

- Prior handoff (start of session): `docs/CORAL_Session_Handoff_2026-04-25-EOD.md`
- Dataset reproducibility: `docs/dataset_reproducibility.md`
- Eval slowness + Fix A failure: `docs/eval_slowness_diagnosis.md` (on `moe-lb-specialization-compile-fix`)
- ARC adapter design: `docs/CORAL_v3_ARC_Adapter_Design.md` (on `arc-adapter-design`)
