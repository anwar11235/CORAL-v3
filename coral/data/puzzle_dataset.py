"""Puzzle dataset loader compatible with HRM's data format.

Loads datasets produced by HRM's build_sudoku_dataset.py (and similar).
Each dataset split contains .npy memory-mapped arrays:
    {set_name}__inputs.npy               — token ids [N, seq_len]
    {set_name}__labels.npy               — target ids [N, seq_len]
    {set_name}__puzzle_identifiers.npy   — per-puzzle IDs [num_puzzles]
    {set_name}__puzzle_indices.npy       — CSR-style start indices into examples
    {set_name}__group_indices.npy        — CSR-style start indices into puzzle groups

A dataset.json file in the split directory contains PuzzleDatasetMetadata.

Training iteration: randomly shuffle puzzle groups, sample one example per group
per step, pack into global_batch_size batches.

Eval iteration: iterate all examples sequentially without shuffling.
"""

import json
import os
from typing import Iterator, List, Optional, Tuple

import numpy as np
import pydantic
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from coral.training.losses import IGNORE_LABEL_ID


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class PuzzleDatasetMetadata(pydantic.BaseModel):
    """Metadata stored alongside each dataset split."""

    pad_id: int
    ignore_label_id: Optional[int]
    blank_identifier_id: int

    vocab_size: int
    seq_len: int
    num_puzzle_identifiers: int

    total_groups: int
    mean_puzzle_examples: float

    sets: List[str]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class PuzzleDatasetConfig(pydantic.BaseModel):
    seed: int
    dataset_path: str
    global_batch_size: int
    test_set_mode: bool
    epochs_per_iter: int  # Number of epochs to batch per __iter__ call
    rank: int
    num_replicas: int


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------


def _sample_batch(
    rng: np.random.Generator,
    group_order: np.ndarray,
    puzzle_indices: np.ndarray,
    group_indices: np.ndarray,
    start_index: int,
    global_batch_size: int,
) -> Tuple[int, np.ndarray, np.ndarray]:
    """Pack examples from groups into a batch of size ≤ global_batch_size."""
    batch: List[np.ndarray] = []
    batch_puzzle_indices: List[np.ndarray] = []
    current_size = 0

    while (start_index < group_order.size) and (current_size < global_batch_size):
        group_id = group_order[start_index]
        puzzle_id = rng.integers(group_indices[group_id], group_indices[group_id + 1])
        start_index += 1

        puzzle_start = puzzle_indices[puzzle_id]
        puzzle_size = int(puzzle_indices[puzzle_id + 1] - puzzle_start)
        append_size = min(puzzle_size, global_batch_size - current_size)

        batch_puzzle_indices.append(np.full(append_size, puzzle_id, dtype=np.int32))
        batch.append(puzzle_start + np.random.choice(puzzle_size, append_size, replace=False))
        current_size += append_size

    return start_index, np.concatenate(batch), np.concatenate(batch_puzzle_indices)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class PuzzleDataset(IterableDataset):
    """Iterable dataset for HRM-format puzzle datasets.

    Yields (set_name, batch_dict, global_batch_size) tuples.
    batch_dict contains "inputs", "labels", "puzzle_identifiers" as int32 tensors.
    """

    def __init__(self, config: PuzzleDatasetConfig, split: str = "train") -> None:
        super().__init__()
        self.config = config
        self.split = split
        self.metadata = self._load_metadata()

        assert config.global_batch_size % config.num_replicas == 0, (
            f"global_batch_size {config.global_batch_size} must be divisible by "
            f"num_replicas {config.num_replicas}."
        )
        self.local_batch_size = config.global_batch_size // config.num_replicas

        self._data = None
        self._iters = 0

    def _load_metadata(self) -> PuzzleDatasetMetadata:
        metadata_path = os.path.join(self.config.dataset_path, self.split, "dataset.json")
        with open(metadata_path, "r") as f:
            return PuzzleDatasetMetadata(**json.load(f))

    def _lazy_load(self) -> None:
        if self._data is not None:
            return

        # inputs/labels are memory-mapped; indices are loaded into RAM
        mmap_modes = {
            "inputs": "r",
            "labels": "r",
            "puzzle_identifiers": None,
            "puzzle_indices": None,
            "group_indices": None,
        }

        self._data = {}
        for set_name in self.metadata.sets:
            self._data[set_name] = {
                field: np.load(
                    os.path.join(self.config.dataset_path, self.split, f"{set_name}__{field}.npy"),
                    mmap_mode=mmap_mode,
                )
                for field, mmap_mode in mmap_modes.items()
            }

    def _collate(self, batch: dict) -> dict:
        """Convert numpy batch to int32, remap ignore labels, pad if needed, tensorize."""
        batch = {k: v.astype(np.int32) for k, v in batch.items()}

        # Remap dataset-specific ignore id to our IGNORE_LABEL_ID (-100)
        if self.metadata.ignore_label_id is not None:
            batch["labels"][batch["labels"] == self.metadata.ignore_label_id] = IGNORE_LABEL_ID

        # Pad to local_batch_size if the last batch is short
        if batch["puzzle_identifiers"].size < self.local_batch_size:
            pad = self.local_batch_size - batch["puzzle_identifiers"].size
            pad_vals = {
                "inputs": self.metadata.pad_id,
                "labels": IGNORE_LABEL_ID,
                "puzzle_identifiers": self.metadata.blank_identifier_id,
            }
            batch = {
                k: np.pad(v, ((0, pad),) + ((0, 0),) * (v.ndim - 1), constant_values=pad_vals[k])
                for k, v in batch.items()
            }

        return {k: torch.from_numpy(v) for k, v in batch.items()}

    def _iter_test(self) -> Iterator:
        for set_name, dataset in self._data.items():  # type: ignore[union-attr]
            total = len(dataset["inputs"])
            start = 0
            while start < total:
                end = min(total, start + self.config.global_batch_size)
                local_start = start + self.config.rank * self.local_batch_size
                local_end = min(start + (self.config.rank + 1) * self.local_batch_size, end)

                # Resolve puzzle IDs for the local slice
                puzzle_ids = []
                puzzle_idx = np.searchsorted(dataset["puzzle_indices"], local_start, side="right") - 1
                for i in range(local_start, local_end):
                    while (
                        puzzle_idx + 1 < len(dataset["puzzle_indices"])
                        and i >= dataset["puzzle_indices"][puzzle_idx + 1]
                    ):
                        puzzle_idx += 1
                    puzzle_ids.append(puzzle_idx)

                yield set_name, self._collate({
                    "inputs": dataset["inputs"][local_start:local_end],
                    "labels": dataset["labels"][local_start:local_end],
                    "puzzle_identifiers": dataset["puzzle_identifiers"][puzzle_ids],
                }), end - start

                start += self.config.global_batch_size

    def _iter_train(self) -> Iterator:
        for set_name, dataset in self._data.items():  # type: ignore[union-attr]
            self._iters += 1
            rng = np.random.Generator(np.random.Philox(seed=self.config.seed + self._iters))

            # epochs_per_iter concatenated shuffled group orderings
            group_order = np.concatenate([
                rng.permutation(dataset["group_indices"].size - 1)
                for _ in range(self.config.epochs_per_iter)
            ])
            start = 0

            while start < group_order.size:
                start, batch_indices, batch_puzzle_indices = _sample_batch(
                    rng,
                    group_order=group_order,
                    puzzle_indices=dataset["puzzle_indices"],
                    group_indices=dataset["group_indices"],
                    start_index=start,
                    global_batch_size=self.config.global_batch_size,
                )

                effective_bs = batch_puzzle_indices.size
                # Drop the last incomplete batch
                if effective_bs < self.config.global_batch_size:
                    break

                r = self.config.rank
                bs = self.local_batch_size
                yield set_name, self._collate({
                    "inputs": dataset["inputs"][batch_indices[r * bs:(r + 1) * bs]],
                    "labels": dataset["labels"][batch_indices[r * bs:(r + 1) * bs]],
                    "puzzle_identifiers": dataset["puzzle_identifiers"][
                        batch_puzzle_indices[r * bs:(r + 1) * bs]
                    ],
                }), effective_bs

    def __iter__(self) -> Iterator:
        worker_info = get_worker_info()
        assert worker_info is None or worker_info.num_workers == 1, (
            "PuzzleDataset does not support num_workers > 1."
        )
        self._lazy_load()
        if self.config.test_set_mode:
            yield from self._iter_test()
        else:
            yield from self._iter_train()


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def create_dataloader(
    dataset_path: str,
    split: str,
    global_batch_size: int,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 0,
    test_set_mode: bool = False,
    epochs_per_iter: int = 1,
) -> Tuple[DataLoader, PuzzleDatasetMetadata]:
    """Build a DataLoader for a single split."""
    dataset = PuzzleDataset(
        PuzzleDatasetConfig(
            seed=seed,
            dataset_path=dataset_path,
            global_batch_size=global_batch_size,
            test_set_mode=test_set_mode,
            epochs_per_iter=epochs_per_iter,
            rank=rank,
            num_replicas=world_size,
        ),
        split=split,
    )
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=8,
        pin_memory=True,
        persistent_workers=True,
    )
    return loader, dataset.metadata
