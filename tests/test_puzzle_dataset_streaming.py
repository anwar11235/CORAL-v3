"""Regression tests for _iter_train streaming refactor.

Verifies:
1. New _iter_train produces byte-identical batches to the legacy O(E×N) implementation.
2. Peak memory of the new implementation is bounded by O(N), not O(E×N).
"""

import json
import tracemalloc
from pathlib import Path

import numpy as np
import pytest
import torch

from coral.data.puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata


# ---------------------------------------------------------------------------
# Synthetic dataset builder
# ---------------------------------------------------------------------------

N_GROUPS = 100
PUZZLES_PER_GROUP = 10
SEQ_LEN = 9
SET_NAME = "train"


def _build_synthetic_dataset(tmp_path: Path) -> Path:
    """Write a minimal HRM-format dataset to tmp_path for use in tests."""
    split_dir = tmp_path / "train"
    split_dir.mkdir(parents=True)

    total_puzzles = N_GROUPS * PUZZLES_PER_GROUP
    total_examples = total_puzzles  # 1 example per puzzle

    rng = np.random.default_rng(0)
    inputs = rng.integers(1, 32, size=(total_examples, SEQ_LEN), dtype=np.int32)
    labels = rng.integers(1, 32, size=(total_examples, SEQ_LEN), dtype=np.int32)
    puzzle_identifiers = np.arange(total_puzzles, dtype=np.int32)
    # CSR: each puzzle has 1 example
    puzzle_indices = np.arange(total_puzzles + 1, dtype=np.int64)
    # CSR: each group has PUZZLES_PER_GROUP puzzles
    group_indices = np.arange(0, total_puzzles + 1, PUZZLES_PER_GROUP, dtype=np.int64)

    prefix = f"{SET_NAME}__"
    np.save(split_dir / f"{prefix}inputs.npy", inputs)
    np.save(split_dir / f"{prefix}labels.npy", labels)
    np.save(split_dir / f"{prefix}puzzle_identifiers.npy", puzzle_identifiers)
    np.save(split_dir / f"{prefix}puzzle_indices.npy", puzzle_indices)
    np.save(split_dir / f"{prefix}group_indices.npy", group_indices)

    metadata = {
        "pad_id": 0,
        "ignore_label_id": None,
        "blank_identifier_id": 0,
        "vocab_size": 32,
        "seq_len": SEQ_LEN,
        "num_puzzle_identifiers": total_puzzles,
        "total_groups": N_GROUPS,
        "mean_puzzle_examples": 1.0,
        "sets": [SET_NAME],
        "dataset_seed": 0,
    }
    (split_dir / "dataset.json").write_text(json.dumps(metadata))

    return tmp_path


def _make_config(dataset_path: Path, global_batch_size: int, epochs_per_iter: int, seed: int) -> PuzzleDatasetConfig:
    return PuzzleDatasetConfig(
        seed=seed,
        dataset_path=str(dataset_path),
        global_batch_size=global_batch_size,
        test_set_mode=False,
        epochs_per_iter=epochs_per_iter,
        rank=0,
        num_replicas=1,
    )


def _collect_batches(dataset: PuzzleDataset) -> list:
    """Collect all batches yielded by _iter_train as a list of (set_name, dict, bs) triples.
    dict values are numpy arrays for easy comparison.
    """
    dataset._lazy_load()
    batches = []
    for set_name, batch, bs in dataset._iter_train():
        batches.append((
            set_name,
            {k: v.numpy() for k, v in batch.items()},
            bs,
        ))
    return batches


def _collect_batches_legacy(dataset: PuzzleDataset) -> list:
    dataset._lazy_load()
    batches = []
    for set_name, batch, bs in dataset._iter_train_legacy():
        batches.append((
            set_name,
            {k: v.numpy() for k, v in batch.items()},
            bs,
        ))
    return batches


# ---------------------------------------------------------------------------
# Test 1: byte-identical regression
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("epochs_per_iter", [1, 3, 7])
@pytest.mark.parametrize("global_batch_size", [8, 13])  # 13 is intentionally non-divisor of 100
def test_iter_train_byte_identical(tmp_path, epochs_per_iter, global_batch_size):
    """New _iter_train must produce byte-identical batches to the legacy implementation."""
    dataset_path = _build_synthetic_dataset(tmp_path)

    # Reset global numpy random state to the same value for both runs.
    # _sample_batch uses np.random.choice (global state), so this ensures
    # the within-group example selection is identical.
    np.random.seed(0)
    cfg = _make_config(dataset_path, global_batch_size=global_batch_size,
                       epochs_per_iter=epochs_per_iter, seed=42)
    ds_legacy = PuzzleDataset(cfg, split="train")
    legacy_batches = _collect_batches_legacy(ds_legacy)

    np.random.seed(0)
    cfg2 = _make_config(dataset_path, global_batch_size=global_batch_size,
                        epochs_per_iter=epochs_per_iter, seed=42)
    ds_new = PuzzleDataset(cfg2, split="train")
    new_batches = _collect_batches(ds_new)

    assert len(new_batches) == len(legacy_batches), (
        f"Batch count differs: new={len(new_batches)}, legacy={len(legacy_batches)}"
    )

    for i, (new, old) in enumerate(zip(new_batches, legacy_batches)):
        set_new, batch_new, bs_new = new
        set_old, batch_old, bs_old = old
        assert set_new == set_old, f"batch {i}: set_name mismatch"
        assert bs_new == bs_old, f"batch {i}: effective_bs mismatch"
        for key in ("inputs", "labels", "puzzle_identifiers"):
            assert np.array_equal(batch_new[key], batch_old[key]), (
                f"batch {i}, key '{key}': arrays differ\n"
                f"  new:    {batch_new[key]}\n"
                f"  legacy: {batch_old[key]}"
            )


# ---------------------------------------------------------------------------
# Test 2: multiple __iter__ calls are independent (iters counter advances)
# ---------------------------------------------------------------------------

def test_iter_train_iters_advance(tmp_path):
    """Consecutive __iter__ calls should produce different orderings (different RNG seed)."""
    dataset_path = _build_synthetic_dataset(tmp_path)
    cfg = _make_config(dataset_path, global_batch_size=8, epochs_per_iter=1, seed=7)
    ds = PuzzleDataset(cfg, split="train")
    ds._lazy_load()

    np.random.seed(0)
    batches_1 = _collect_batches(ds)
    np.random.seed(0)
    ds._iters = 0  # reset to simulate a second __iter__ with matching counter
    batches_2 = _collect_batches(ds)
    # Same iters offset → identical
    assert all(
        np.array_equal(b1[1]["inputs"], b2[1]["inputs"])
        for b1, b2 in zip(batches_1, batches_2)
    ), "Same iters counter should produce identical batches"

    # Different iters offset → different
    np.random.seed(0)
    # ds._iters is now 1 from the second collect; run again (iters=2)
    batches_3 = _collect_batches(ds)
    assert not all(
        np.array_equal(b1[1]["inputs"], b3[1]["inputs"])
        for b1, b3 in zip(batches_1, batches_3)
    ), "Different iters counter should produce different batches"


# ---------------------------------------------------------------------------
# Test 3: peak memory profile
# ---------------------------------------------------------------------------

def test_memory_profile(tmp_path, capsys):
    """New implementation peak memory should be << old for large epochs_per_iter."""
    dataset_path = _build_synthetic_dataset(tmp_path)
    EPOCHS = 100
    BATCH_SIZE = 8

    def measure_peak(use_legacy: bool) -> int:
        cfg = _make_config(dataset_path, global_batch_size=BATCH_SIZE,
                           epochs_per_iter=EPOCHS, seed=99)
        ds = PuzzleDataset(cfg, split="train")
        ds._lazy_load()
        np.random.seed(0)
        tracemalloc.start()
        if use_legacy:
            for _ in ds._iter_train_legacy():
                pass
        else:
            for _ in ds._iter_train():
                pass
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak

    peak_legacy = measure_peak(use_legacy=True)
    peak_new    = measure_peak(use_legacy=False)

    # With N=100 groups, E=100 epochs:
    #   legacy peak ≥ N*E*8 bytes = 80 KB (the concatenated group_order array)
    #   new    peak ≤ N*8 bytes   =  800 bytes (one epoch permutation) + overhead
    expected_legacy_floor = N_GROUPS * EPOCHS * 8  # bytes

    with capsys.disabled():
        print(f"\n[memory_profile] epochs_per_iter={EPOCHS}, n_groups={N_GROUPS}")
        print(f"  legacy peak: {peak_legacy:>10,} bytes  (expected >= {expected_legacy_floor:,})")
        print(f"  new    peak: {peak_new:>10,} bytes")
        print(f"  reduction:   {peak_legacy / max(peak_new, 1):.1f}x")

    # New peak should be strictly less than legacy peak
    assert peak_new < peak_legacy, (
        f"New implementation ({peak_new} bytes) should use less memory than "
        f"legacy ({peak_legacy} bytes)"
    )

    # Legacy peak should include at least the group_order array
    assert peak_legacy >= expected_legacy_floor, (
        f"Legacy peak ({peak_legacy} bytes) should be >= N*E*8={expected_legacy_floor} bytes"
    )
