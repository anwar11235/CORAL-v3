# CORAL v3 ARC-AGI Adapter — Design Document

**Branch:** arc-adapter-design  
**Date:** 2026-04-25  
**Status:** Pre-implementation design review. Do not implement without sign-off.  
**Reference:** `docs/hrm_arc_integration_notes.md`

---

## 0. Executive Summary

ARC-AGI-1 can be added as a second training/eval target with **no changes to the CORAL model architecture** or training loop. The only new artifacts are a data builder script, one config YAML, and minor tuning decisions. The dominant open question is whether the spatial codebook's buffer memory constraints are acceptable at ARC's 11× longer sequence length, or whether crystallization should be deferred to a separate ARC phase.

---

## 1. What Changes in Config

### 1.1 Automatic (driven by dataset metadata)

`train.py` reads `seq_len`, `vocab_size`, and `num_puzzle_identifiers` from `dataset.json` and passes them directly to `CoralConfig` (train.py:158-160). So these values adapt automatically when a different dataset is provided:

| Parameter | Sudoku-Extreme | ARC-AGI-1 |
|-----------|---------------|-----------|
| `vocab_size` | 11 | **12** (PAD=0, EOS=1, colors 2–11) |
| `seq_len` | 81 | **900** (30×30 max-pad) |
| `num_puzzle_identifiers` | ~1000 | **~960** (official + ConceptARC puzzles) |

No manual config entries needed for these three parameters.

### 1.2 Must be set explicitly in `configs/arc_base.yaml`

**`global_batch_size`**: Reduce from 384. The embedding tensor alone is `B × 900 × 512` elements. With FlashAttention, attention memory is O(seq_len) not O(seq_len²), so the bottleneck is activations during backprop. With 1-step gradient (all but last ACT step under `torch.no_grad()`), this is manageable, but needs profiling. **Starting recommendation: 128.** Tune upward if GPU memory allows.

**`halt_max_steps`**: ARC tasks require more deliberation than Sudoku. HRM uses 16. Start with 16 and increase to 24 if underfitting is observed.

**`crystal_buffer_capacity`**: **Critical constraint** — see §3 below. Must be reduced to ~900 for ARC or crystallization OOMs.

**`crystal_bootstrap_steps`**: Scale with batch size. At batch_size=128, a 5000-step bootstrap accumulates 640K examples — far more than needed. Start at 1000.

**`puzzle_emb_ndim`**: Keep at 512. ARC has ~960 puzzles; the embedding table stays at 960×512 = ~2 MB, which is fine.

**`lr_warmup_steps`**: Keep at 1000 (same as Sudoku).

**No change needed for:** `hidden_size`, `num_heads`, `expansion`, `H_cycles/L_cycles`, `H_layers/L_layers`, `halt_exploration_prob`, `rope_theta`, `rms_norm_eps`, `forward_dtype`, `loss_type`, `lr`, `weight_decay`, `beta1`, `beta2`.

---

## 2. New Code Required

### 2.1 `coral/data/build_arc_dataset.py` (~350 LOC)

The primary new artifact. Produces the identical `.npy` / `dataset.json` format consumed by `PuzzleDataset` — no changes to the dataloader are needed.

**Tokenization:**
```python
PAD_ID = 0       # unused grid space
EOS_ID = 1       # grid boundary marker (placed at end of each grid row or at grid edge)
COLOR_OFFSET = 2 # color 0 → token 2, color 9 → token 11
```

Grid encoding: row-major flatten of 30×30 frame. Original grid is placed at (row_offset, col_offset) within the frame; remaining cells are PAD. EOS marks the actual grid boundary.

**Augmentations to implement (all three from HRM):**

1. **Dihedral group** (8 variants): `dihedral_transform(trans_id)` with `trans_id ∈ {0…7}`. `coral/data/common.py` already has `dihedral_transform()` — reuse it directly.

2. **Color permutation**: Permute colors 1–9 (keep 0/black fixed). Apply the same permutation mapping to both input and output grids of each demonstration pair.

3. **Translational shift**: Random (row_offset, col_offset) within available space. Apply during dataset build; test split uses zero offset.

**Demonstration handling:** Each (input_grid, output_grid) pair stored as a separate row. Use existing CSR `puzzle_indices` / `group_indices` layout — identical to Sudoku builder output.

**Dataset.json contents:**
```json
{
  "pad_id": 0,
  "ignore_label_id": 0,
  "blank_identifier_id": 1,
  "vocab_size": 12,
  "seq_len": 900,
  "num_puzzle_identifiers": <N_puzzles>,
  "total_groups": <N_aug_groups>,
  "mean_puzzle_examples": <float>,
  "sets": ["train__", "test__"]
}
```

**Input data source:** ARC-AGI-1 JSON files from the official ARC prize repository. The builder should accept `--arc-data-dir` pointing to the directory containing `training/` and `evaluation/` subdirectories.

### 2.2 `configs/arc_base.yaml` (~45 LOC)

New config file. Can inherit from `base.yaml` where unchanged. Key overrides:

```yaml
# Derived from dataset metadata — shown for documentation only:
# vocab_size: 12 (set by train.py from dataset.json)
# seq_len: 900   (set by train.py from dataset.json)

global_batch_size: 128
halt_max_steps: 16
puzzle_emb_ndim: 512

# Crystallization — see §3 for rationale on capacity:
crystal_buffer_capacity: 900
crystal_bootstrap_steps: 1000
crystal_consolidation_interval: 5000
moe_num_modes: 64           # see §3.1 open question

lambda_moe_recon: 0.1
lambda_moe_balance: 0.01
```

### 2.3 No other new files required

`PuzzleDataset`, `train.py`, `CoralConfig`, `CoralV3Inner`, `CoralV3ACT`, `CoralV3LossHead` — all unchanged. The model sees a longer token sequence, but nothing in the forward pass is hardcoded to seq_len=81.

---

## 3. What Is Reused Unchanged

| Component | File | Reused? |
|-----------|------|---------|
| Transformer blocks (H/L) | `coral/models/layers.py`, `transformer_block.py` | ✅ unchanged |
| Reasoning module | `coral/models/reasoning_module.py` | ✅ unchanged |
| CoralInner / CoralV3Inner | `coral/models/coral_base.py`, `coral_v3.py` | ✅ unchanged |
| ACT loop | `coral/training/act.py` | ✅ unchanged |
| Loss functions | `coral/training/losses.py` | ✅ unchanged |
| PuzzleDataset / DataLoader | `coral/data/puzzle_dataset.py` | ✅ unchanged |
| Training loop | `scripts/train.py` | ✅ unchanged |
| Optimizer (AdamATan2) | installed package | ✅ unchanged |
| SpatialMoECodebook | `coral/models/crystallization.py` | ✅ unchanged (but K and capacity must be tuned — see §3.1) |
| RoPE positional encoding | `coral/models/layers.py` | ✅ unchanged (covers seq_len=900 with rope_theta=10000) |
| `coral/data/common.py` | dihedral transforms | ✅ reused directly by new builder |

The model architecture is fully agnostic to task domain. The only task-specific parameters (vocab size, sequence length) flow in through the dataset metadata. This is a clean seam.

---

## 4. Open Design Question: Spatial Codebook for ARC

**This section flags a real design decision. Do not resolve by assumption.**

### 4.1 Memory constraint (hard)

`CrystallizationBuffer.spatial_buffer` stores `[capacity, seq_len, l_dim]` float32 on CPU (crystallization.py:202). At seq_len=900, l_dim=512:

| Buffer capacity | CPU RAM |
|----------------|---------|
| 10,000 (Sudoku default) | 18.4 GB ❌ infeasible |
| 4,000 | 7.4 GB ❌ too large |
| **900** | **1.66 GB ✅ matches Sudoku's current footprint** |
| 450 | 0.83 GB ✅ conservative |

**Recommendation A (preferred — no code changes):** Set `crystal_buffer_capacity = 900`. Same RAM as Sudoku's 10K-entry buffer. Fewer k-means points (900 vs 10K) but ARC patterns are more geometrically diverse so the codebook will still benefit from initialization.

**Recommendation B (higher quality, requires code change):** Project `z_L_spatial → [seq_len, proj_dim]` before buffering (using the existing `crystal_proj_dim=128` projection). Buffer stores `[capacity, seq_len, 128]` instead of `[capacity, seq_len, 512]`. At capacity=4000: 1.84 GB. Better coverage, but requires a change to `CrystallizationBuffer._lazy_init_spatial()` and the buffer `add()` call site in `CoralV3Inner`. **This should be done in a dedicated commit, not bundled with the data adapter.**

### 4.2 Number of codebook modes K (tuning question)

Sudoku found ~6 natural L-state modes (`codebook_utilisation_frac = 6/256 ≈ 2.3%` in Phase 3a'). ARC spans vastly more task categories — spatial transformations, counting, symmetry detection, recoloring, pattern completion — and the L-state manifold should have more natural clusters.

| Option | `moe_num_modes` | Rationale |
|--------|-----------------|-----------|
| Conservative | 32 | Same as Phase 3c; minimal risk; may underfit ARC diversity |
| **Recommended** | **64** | 2× Sudoku's K; still tractable; good starting hypothesis |
| Aggressive | 128 | Higher expressivity; longer consolidation; harder to monitor |

**Flag for human decision:** Start at K=64. If `codebook_utilisation_frac` > 0.5 (i.e., most modes are used), increase to 128 in a follow-up run. If `codebook_utilisation_frac` < 0.1 (as in Sudoku's K=32 run), the ARC L-state manifold is simpler than expected and K=32 suffices.

### 4.3 Phased crystallization strategy (recommended)

Given the unknowns above, a two-phase ARC approach is lower risk:

- **ARC Phase 0 (baseline):** Disable crystallization (`use_crystallization: false`). Establish a reliable accuracy baseline on ARC with PC-only CORAL. This also bounds the compute budget (ARC at batch_size=128 with halt_max_steps=16 is expensive).
- **ARC Phase 1 (crystal):** Enable soft MoE with the buffer and K tuned from Phase 0 observations. Warm-start from ARC Phase 0 checkpoint.

---

## 5. Demonstration Pair Encoding — Design Choice

HRM treats each demo as an independent 900-token sequence with no explicit cross-pair attention. The model must infer the rule from the statistical coherence of co-batched pairs.

CORAL has an ACT carry state that persists across segments. Two options for multi-demo ARC puzzles:

**Option A (match HRM):** Each demo is an independent example. Carry resets on halt. Demonstrations from the same puzzle may end up in different batch positions. This is the simpler option and matches HRM exactly.

**Option B (sequential carry):** Force same-puzzle demos to appear consecutively, and reset carry at puzzle boundaries rather than on halt. This lets the model accumulate a "rule hypothesis" across demos. Non-trivial to implement; no evidence it helps; HRM didn't do this.

**Recommendation:** Start with Option A. It requires no changes to the dataloader or ACT loop, and HRM's competitive ARC results suggest it's sufficient.

---

## 6. Implementation Scope Estimate

| Commit | Description | Estimated LOC |
|--------|-------------|---------------|
| 1 | `coral/data/build_arc_dataset.py` — tokenization + augmentations + output format | ~350 |
| 2 | `configs/arc_base.yaml` + initial baseline run (no crystal) | ~45 |
| 3 | Tests: `tests/test_arc_dataset.py` — tokenization roundtrip, augmentation sanity, dataset.json validation | ~120 |
| 4 (optional) | Buffer projection fix for ARC memory efficiency (Recommendation B §4.1) | ~40 |

**Total: 3 commits, ~515 LOC net new. Zero changes to existing files.**

---

## 7. Cleanest Path Forward

The cleanest path is: build `coral/data/build_arc_dataset.py` that emits the identical `.npy` format CORAL already consumes, then launch an ARC Phase 0 baseline run (`use_crystallization=false`) with `configs/arc_base.yaml`. Because `train.py` reads `vocab_size` and `seq_len` from dataset metadata, the model adapts automatically. The only real engineering decision to resolve now is the crystallization buffer memory issue (§4.1) — Recommendation A (set `crystal_buffer_capacity=900`) is the zero-code-change path and should be tried first. K=64 codebook modes is the recommended starting hypothesis for ARC crystallization (§4.2), to be validated by `codebook_utilisation_frac` once the baseline run confirms the ARC task is learnable at all.

---

## 8. Dataset Determinism Requirement (Hard Requirement)

**Added 2026-04-25.** See `docs/dataset_reproducibility.md` for full context.

The Sudoku dataset builder (`coral/data/build_sudoku_dataset.py`) used unseeded `np.random` calls through Phase 3c of CORAL training. Every rebuild produced a different dataset realization. This made cross-run accuracy comparisons methodologically fragile and rendered Phase 3c's 67.62% result non-reproducible. The bug was discovered on 2026-04-25 and fixed in the `dataset-determinism-fix` branch.

**`coral/data/build_arc_dataset.py` MUST follow the same seeding pattern.** Specifically:

- `--seed` CLI argument with default `0`
- `np.random.seed(seed)` called at the very start of the entry point, before any stochastic operation
- All augmentation operations (dihedral transform, color permutation, translational shift) must derive their randomness exclusively from the seeded numpy state — no `random.random()`, no `torch.rand`, no other RNG sources
- Seed value recorded in `dataset.json` under `dataset_seed` (the `PuzzleDatasetMetadata` field is already defined in `coral/data/common.py`)
- SHA-256 validation that two builds with the same seed produce byte-identical `.npy` output (see `tests/test_dataset_determinism.py` for the Sudoku reference implementation of this test)

**Why this matters for ARC specifically:** ARC-AGI-1 has 960 training puzzles × augmentation variants. With unseeded augmentation, each build selects different color permutations and translational shifts. A model checkpoint trained on one realization will appear to perform at chance (0% accuracy) when evaluated against any other realization. This is exactly the failure mode that was discovered for Phase 3c.

Failure to implement seeding in the ARC builder will reproduce the dataset reproducibility issue and block the ARC experiment from producing valid measurements.
