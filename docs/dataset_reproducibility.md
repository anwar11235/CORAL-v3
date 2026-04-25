# Dataset Reproducibility — CORAL v3

**Created:** 2026-04-25  
**Status:** Fix landed on `dataset-determinism-fix` branch, pending merge to `master`

---

## The Bug (Discovered 2026-04-25)

`coral/data/build_sudoku_dataset.py` used `numpy.random` without seeding. Stochastic operations included:

- `np.random.permutation` — digit mapping, band/stack permutations in `shuffle_sudoku()`
- `np.random.rand()` — transpose flag in `shuffle_sudoku()`
- `np.random.choice` — train-set subsampling in `convert_subset()`

**Effect:** Every invocation of the build script produced a different dataset realization — different training/test puzzle subsets, different augmentations. The training split, test split, and every augmented variant changed on each rebuild.

**Discovery:** On 2026-04-25, reloading `phase3c_option_y_step52080.pt` against a freshly-rebuilt dataset yielded `eval/exact_accuracy = 0`. The checkpoint had achieved 67.62% on its original dataset; evaluated against any other realization, it performs at chance.

---

## Impact on Prior Results

All CORAL training runs through Phase 3c are affected:

| Run | Phase | Accuracy | Dataset reproducible? |
|-----|-------|----------|----------------------|
| calculating-caracara | Baseline | 41.2% | No — dataset not archived |
| poetic-giraffe | Phase 1 | 61.07% | No |
| orchid-heron | Baseline (fused) | 54.2% | No |
| jovial-avocet | Phase 3a' | 63.48% | No |
| control-no-crystal | Phase 3a control | 65.58% | No |
| satisfied-owl | Phase 3b | 66.05% | No |
| **phase3c_option_y** | **Phase 3c** | **67.62%** | **No — checkpoint saved, dataset not** |

**What is recoverable:**
- W&B training trajectories (loss curves, metric curves) — intact
- Model checkpoints — intact but tied to their original dataset
- Config YAMLs and commit hashes — intact
- Accuracy numbers as measured on original datasets — valid as single-seed estimates

**What is not recoverable:**
- The exact datasets used for any prior run
- Cross-run comparisons as controlled experiments (different runs may have been on different datasets)

**Implication for the 67.62% claim:** The number is a real measurement on a real dataset, but it cannot be reproduced without the original dataset. It is not suitable as a published benchmark unless re-established on a canonical seeded dataset.

---

## The Fix

**Branch:** `dataset-determinism-fix`, commit `9493d6d`

Changes to `coral/data/build_sudoku_dataset.py`:
1. Added `seed: int = 0` to `DataProcessConfig`
2. Added `np.random.seed(config.seed)` as the first statement of `preprocess_data()` — before any stochastic operation
3. Added `dataset_seed=config.seed` to the `PuzzleDatasetMetadata` written to `dataset.json`

Changes to `coral/data/common.py` and `coral/data/puzzle_dataset.py`:
- Added `dataset_seed: Optional[int] = None` to `PuzzleDatasetMetadata` (backward compatible — old datasets parse with `None`)

**Verified:** Two builds with `--seed 0` produce byte-identical SHA-256 hashes across all `.npy` files. See `tests/test_dataset_determinism.py`.

---

## Project Policy (Going Forward)

1. **Every training run must use a documented seed.** The seed must appear in:
   - The build command (`python coral/data/build_sudoku_dataset.py --seed 0 ...`)
   - The resulting `dataset.json` (`dataset_seed` field)
   - The W&B run config (via the `data_path` pointing to a seeded dataset)

2. **Canonical seed for Sudoku-Extreme-1K:** `--seed 0`. This is the reference dataset for all paper results and multi-seed campaigns.

3. **Dataset archiving:** After building with `--seed 0`, archive the `.npy` files to durable storage (separate disk or cloud bucket) so the dataset survives Vast instance teardowns. This eliminates rebuild dependency entirely.

4. **Multi-seed dataset studies** (if needed): Use `--seed 1`, `--seed 2`, etc. Document which checkpoints were trained on which seed.

5. **No mixing seeds:** Do not warm-start a model trained on `--seed 0` and evaluate it on a dataset built with `--seed 1`. The checkpoint is tied to its training dataset realization.

---

## How to Verify Dataset Reproducibility

```bash
# Build twice with the same seed
python coral/data/build_sudoku_dataset.py \
    --output-dir data/sudoku-extreme-1k-aug-1000-seed0-run1 \
    --subsample-size 1000 --num-aug 1000 --seed 0

python coral/data/build_sudoku_dataset.py \
    --output-dir data/sudoku-extreme-1k-aug-1000-seed0-run2 \
    --subsample-size 1000 --num-aug 1000 --seed 0

# Compare SHA-256 hashes (should be identical)
find data/sudoku-extreme-1k-aug-1000-seed0-run1 -name "*.npy" \
    -exec sha256sum {} \; | sort

find data/sudoku-extreme-1k-aug-1000-seed0-run2 -name "*.npy" \
    -exec sha256sum {} \; | sort
```

Or run the automated test:
```bash
pytest tests/test_dataset_determinism.py -v
```

---

## ARC Adapter Note

The same seeding requirement applies to `coral/data/build_arc_dataset.py` when it is implemented. See `docs/CORAL_v3_ARC_Adapter_Design.md` §8 for the explicit requirement. Failure to seed will reproduce this bug for the ARC pipeline.
