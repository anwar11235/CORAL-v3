"""Tests for deterministic dataset construction (build_sudoku_dataset.py seeding fix).

Root cause context: before 2026-04-25, the Sudoku dataset builder used np.random
without seeding. Each invocation produced a different dataset, making cross-run
comparisons meaningless. These tests guard the determinism fix.

Test A: same seed → byte-identical .npy files (SHA-256 match)
Test B: different seeds → at least one .npy file differs (seed is active)
Test C: seed recorded in dataset.json as 'dataset_seed'

All tests use a small synthetic fixture (15-row CSV, subsample_size=10, num_aug=5)
and mock hf_hub_download to avoid network calls. Total runtime: <2s.
"""

import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from coral.data.build_sudoku_dataset import DataProcessConfig, convert_subset


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_fixture_csv(path: str, n_rows: int) -> None:
    """Write a minimal synthetic Sudoku CSV accepted by convert_subset."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "q", "a", "rating"])
        for i in range(n_rows):
            # q: 81 blanks ('0'); valid as puzzle input
            q = "0" * 81
            # a: 81 digits cycling through 1-9; values 1-9 satisfy the arr<=9 assertion
            a = "".join(str((i + j) % 9 + 1) for j in range(81))
            writer.writerow([f"src{i}", q, a, 100])


def _sha256_npy(directory: str) -> dict:
    """Return {relative_path: sha256_hex} for every .npy file under directory."""
    result = {}
    for root, _, files in os.walk(directory):
        for name in sorted(files):
            if not name.endswith(".npy"):
                continue
            path = os.path.join(root, name)
            rel = os.path.relpath(path, directory)
            with open(path, "rb") as f:
                result[rel] = hashlib.sha256(f.read()).hexdigest()
    return result


def _run_build(seed: int, output_dir: str, train_csv: str, test_csv: str) -> None:
    """Run the build pipeline with a fixed seed and mocked HF download.

    Mirrors exactly what preprocess_data() does after the seeding fix:
      np.random.seed(config.seed)
      convert_subset("train", config)
      convert_subset("test", config)
    """
    def _mock_hf(repo, filename, repo_type=None):
        return train_csv if "train" in filename else test_csv

    cfg = DataProcessConfig(
        output_dir=output_dir,
        subsample_size=10,
        num_aug=5,
        seed=seed,
    )

    with patch("coral.data.build_sudoku_dataset.hf_hub_download", side_effect=_mock_hf):
        np.random.seed(cfg.seed)
        convert_subset("train", cfg)
        convert_subset("test", cfg)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDatasetDeterminism:
    """Guard tests for np.random seeding in build_sudoku_dataset.py."""

    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        # Create fixture CSVs once per test method
        csv_dir = os.path.join(self._tmpdir, "csvs")
        os.makedirs(csv_dir)
        self._train_csv = os.path.join(csv_dir, "train.csv")
        self._test_csv = os.path.join(csv_dir, "test.csv")
        _make_fixture_csv(self._train_csv, n_rows=15)
        _make_fixture_csv(self._test_csv, n_rows=15)

    def teardown_method(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_A_same_seed_byte_identical_output(self):
        """Two runs with seed=0 must produce byte-identical .npy files (SHA-256)."""
        out1 = os.path.join(self._tmpdir, "run1")
        out2 = os.path.join(self._tmpdir, "run2")

        _run_build(seed=0, output_dir=out1,
                   train_csv=self._train_csv, test_csv=self._test_csv)
        _run_build(seed=0, output_dir=out2,
                   train_csv=self._train_csv, test_csv=self._test_csv)

        h1 = _sha256_npy(out1)
        h2 = _sha256_npy(out2)

        assert h1, "No .npy files produced — check build pipeline"
        differing = {k for k in h1 if h1[k] != h2.get(k)}
        assert not differing, (
            f"Same seed=0 produced different output for: {sorted(differing)}"
        )

    def test_B_different_seeds_produce_different_output(self):
        """Seed=0 and seed=1 must produce at least one differing .npy file."""
        out0 = os.path.join(self._tmpdir, "seed0")
        out1 = os.path.join(self._tmpdir, "seed1")

        _run_build(seed=0, output_dir=out0,
                   train_csv=self._train_csv, test_csv=self._test_csv)
        _run_build(seed=1, output_dir=out1,
                   train_csv=self._train_csv, test_csv=self._test_csv)

        h0 = _sha256_npy(out0)
        h1 = _sha256_npy(out1)

        assert h0 and h1, "No .npy files produced"
        differing = {k for k in h0 if h0[k] != h1.get(k)}
        assert differing, (
            "seed=0 and seed=1 produced identical outputs — "
            "seeding may not be controlling stochasticity"
        )

    def test_C_seed_recorded_in_metadata(self):
        """dataset.json must contain 'dataset_seed' equal to the --seed argument."""
        out = os.path.join(self._tmpdir, "seed42")
        _run_build(seed=42, output_dir=out,
                   train_csv=self._train_csv, test_csv=self._test_csv)

        for split in ("train", "test"):
            meta_path = os.path.join(out, split, "dataset.json")
            assert os.path.exists(meta_path), f"Missing {split}/dataset.json"
            with open(meta_path) as f:
                meta = json.load(f)
            assert "dataset_seed" in meta, (
                f"'dataset_seed' not found in {split}/dataset.json"
            )
            assert meta["dataset_seed"] == 42, (
                f"Expected dataset_seed=42 in {split}/dataset.json, got {meta['dataset_seed']}"
            )
