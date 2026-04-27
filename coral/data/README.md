# coral/data

Puzzle dataset loading (`puzzle_dataset.py`) and the Sudoku-Extreme dataset builder.

## Building the Canonical Sudoku-Extreme-1K Dataset

**All four flags below are required.** Omitting any one produces a different dataset
that is incompatible with the canonical checkpoints and step-count expectations.

| Flag | Value | Why required |
|---|---|---|
| `--subsample-size 1000` | 1,000 base puzzles | Defines the "1K" in the dataset name |
| `--num-aug 1000` | 1,000 augmentations per puzzle | Defines the "aug-1000" in the dataset name; gives 1,001 variants per group |
| `--seed 0` | Deterministic build | Required for reproducibility; recorded in `dataset.json` |
| `--output-dir <path>` | Destination directory | Must match `data_path` in your training config |

### One-shot build (recommended)

Use the repo-root `Makefile`:

    make canonical-dataset DATA_DIR=/workspace/data/sudoku-extreme-1k-aug-1000

This runs the correct four-flag command and then verifies `total_groups == 1000`
in the resulting `dataset.json`. If verification fails the make target exits non-zero.

### Manual build

    python coral/data/build_sudoku_dataset.py \
      --output-dir /workspace/data/sudoku-extreme-1k-aug-1000 \
      --subsample-size 1000 \
      --num-aug 1000 \
      --seed 0

### Post-build verification

After building, verify the metadata is correct before launching any training run:

    python - <<'EOF'
    import json, sys
    path = "/workspace/data/sudoku-extreme-1k-aug-1000/train/dataset.json"
    meta = json.load(open(path))
    assert meta["total_groups"] == 1000, \
        f"WRONG DATASET: total_groups={meta['total_groups']}, expected 1000. " \
        f"Did you pass --subsample-size 1000 --num-aug 1000?"
    assert meta.get("dataset_seed") == 0, \
        f"WRONG SEED: dataset_seed={meta.get('dataset_seed')}, expected 0. " \
        f"Did you pass --seed 0?"
    print(f"OK: total_groups={meta['total_groups']}, dataset_seed={meta['dataset_seed']}")
    EOF

Expected output:

    OK: total_groups=1000, dataset_seed=0

**If `total_groups` is not 1000** (e.g., you see 3,831,994), the dataset was built
without `--subsample-size` and/or `--num-aug`. A wrong dataset causes:
- `total_steps` to be 199M instead of 52K (wrong tqdm denominator)
- The dataloader to OOM-kill the worker (61 GB pre-allocation in `_iter_train`)
- Checkpoints trained on this dataset to be incompatible with canonical evals

Delete the directory and rebuild with all four flags.

### Expected output structure

    sudoku-extreme-1k-aug-1000/
    ├── identifiers.json
    ├── train/
    │   ├── dataset.json           ← must contain total_groups=1000, dataset_seed=0
    │   ├── train__inputs.npy
    │   ├── train__labels.npy
    │   ├── train__group_indices.npy
    │   ├── train__puzzle_identifiers.npy
    │   └── train__puzzle_indices.npy
    └── test/
        ├── dataset.json
        ├── test__inputs.npy
        └── ...

### HRM-compatible output (original naming)

Pass `--coral-naming=false` to produce HRM-style `all__` prefixed files:

    python coral/data/build_sudoku_dataset.py \
      --output-dir /workspace/data/sudoku-extreme-1k-aug-1000 \
      --subsample-size 1000 \
      --num-aug 1000 \
      --seed 0 \
      --coral-naming=false

If you built with `--coral-naming=false`, apply the post-build rename manually:

    cd /path/to/data/sudoku-extreme-1k-aug-1000/train
    for f in all__*.npy; do mv "$f" "train__${f#all__}"; done
    cd ../test
    for f in all__*.npy; do mv "$f" "test__${f#all__}"; done

And update the `sets` field in each `dataset.json`:

    sed -i 's/"sets": \["all"\]/"sets": ["train"]/' train/dataset.json
    sed -i 's/"sets": \["all"\]/"sets": ["test"]/' test/dataset.json
