"""Verify that a built dataset matches the canonical Sudoku-Extreme-1K spec.

Usage:
    python scripts/verify_canonical_dataset.py /path/to/sudoku-extreme-1k-aug-1000

Exits 0 on success, 1 on failure. Intended to be called by `make canonical-dataset`.
"""

import json
import sys
from pathlib import Path


def verify(data_dir: str) -> None:
    root = Path(data_dir)
    train_meta_path = root / "train" / "dataset.json"

    if not train_meta_path.exists():
        print(f"ERROR: {train_meta_path} not found — build may have failed.", file=sys.stderr)
        sys.exit(1)

    meta = json.loads(train_meta_path.read_text())
    errors = []

    if meta.get("total_groups") != 1000:
        errors.append(
            f"total_groups={meta.get('total_groups')!r}, expected 1000 "
            f"(did you pass --subsample-size 1000 --num-aug 1000?)"
        )
    if meta.get("dataset_seed") != 0:
        errors.append(
            f"dataset_seed={meta.get('dataset_seed')!r}, expected 0 "
            f"(did you pass --seed 0?)"
        )

    if errors:
        print("DATASET VERIFICATION FAILED:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        print(
            "\nDelete the output directory and rebuild with all four flags:\n"
            "  make canonical-dataset DATA_DIR=<path>\n"
            "  # or manually:\n"
            "  python coral/data/build_sudoku_dataset.py \\\n"
            "    --output-dir <path> --subsample-size 1000 --num-aug 1000 --seed 0",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"OK  total_groups={meta['total_groups']}"
        f"  dataset_seed={meta['dataset_seed']}"
        f"  dataset ready at {root}"
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <data_dir>", file=sys.stderr)
        sys.exit(1)
    verify(sys.argv[1])
