# coral/data

Puzzle dataset loading (`puzzle_dataset.py`) and the Sudoku-Extreme dataset builder.

## Building the Sudoku-Extreme Dataset

The Sudoku-Extreme-1K dataset is generated locally from HRM's dataset builder,
which has been ported into this repo at `coral/data/build_sudoku_dataset.py`
(Apache-2.0, attribution in the file header).

To build:

    python coral/data/build_sudoku_dataset.py \
      --output-dir /path/to/data/sudoku-extreme-1k-aug-1000 \
      --subsample-size 1000 \
      --num-aug 1000

By default `--coral-naming` is enabled, which emits files with `train__` / `test__`
prefixes and sets the `"sets"` field in `dataset.json` correctly. This is what
`coral/data/puzzle_dataset.py` expects.

Expected output structure:

    sudoku-extreme-1k-aug-1000/
    ├── identifiers.json
    ├── train/
    │   ├── dataset.json
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
      --output-dir /path/to/data/sudoku-extreme-1k-aug-1000 \
      --subsample-size 1000 \
      --num-aug 1000 \
      --coral-naming=false

If you built with `--coral-naming=false`, apply the post-build rename manually:

    cd /path/to/data/sudoku-extreme-1k-aug-1000/train
    for f in all__*.npy; do mv "$f" "train__${f#all__}"; done
    cd ../test
    for f in all__*.npy; do mv "$f" "test__${f#all__}"; done

And update the `sets` field in each `dataset.json`:

    sed -i 's/"sets": \["all"\]/"sets": ["train"]/' train/dataset.json
    sed -i 's/"sets": \["all"\]/"sets": ["test"]/' test/dataset.json
