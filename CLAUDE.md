# CORAL v3 — COrtical Reasoning via Abstraction Layers

## What This Project Is

CORAL v3 is a neural architecture for complex reasoning tasks that builds on the Hierarchical Reasoning Model (HRM) architecture, adding three novel mechanisms unified under a variational free energy objective:

1. **Precision-weighted predictive coding** between hierarchy levels (Bayesian attention)
2. **Recognition-gated crystallization** — System 1/System 2 computation bypass via learned codebook
3. **Sparse columnar routing** — Bayesian model selection over specialized sub-modules (index-select, S=8, k=2)

The goal is to match HRM's reasoning accuracy (~99% on Sudoku-Extreme, ~99% on Maze-Hard) with significantly fewer active parameters.

## Key Documents

- `docs/CORAL_v3_Claude_Code_Build_Plan.md` — **Complete build specification**. Read this FIRST before implementing anything. It contains the exact architecture details, phase-by-phase build order, and implementation specs for every component.
- `docs/HRM_Codebase_Analysis.md` — Detailed analysis of HRM's architecture extracted from their source code. Contains every critical dimension, initialization, and design choice.
- `docs/CORAL_v3_Implementation_Plan.md` — Higher-level design rationale, benchmark results, and research context.

## Reference Code

The original HRM implementation is at `../HRM/HRM/` (read-only reference). Key files:
- `../HRM/HRM/models/hrm/hrm_act_v1.py` — HRM model with ACT wrapper
- `../HRM/HRM/models/layers.py` — Transformer block internals (Attention, SwiGLU, RoPE, CastedLinear)
- `../HRM/HRM/models/common.py` — JAX-compatible truncated normal init
- `../HRM/HRM/models/losses.py` — stablemax cross-entropy, ACT loss
- `../HRM/HRM/pretrain.py` — Training loop with deep supervision
- `../HRM/HRM/puzzle_dataset.py` — Dataset loading

**Rules for reference code:**
- READ it to understand exact tensor shapes, edge cases, and initialization details
- NEVER copy code verbatim — write independent implementations that match behavior
- NEVER modify files in ../HRM/

## Build Order

The project is built in phases. Each phase produces a working, testable system. Never break what a previous phase established.

1. **Phase 0**: Faithful HRM reproduction (base Transformer, forward pass, ACT, training loop)
2. **Phase 1**: Predictive coding + precision-weighting between levels
3. **Phase 2**: Sparse columnar routing (index-select, Strategy C)
4. **Phase 3**: Recognition-gated crystallization
5. **Phase 4**: N=3 extension with hierarchical crystallization
6. **Phase 5**: Ablations
7. **Phase 6**: Analysis + figures

## Architecture Specs (Quick Reference)

```
hidden_size: 512
num_heads: 8 (head_dim=64)
expansion: 4 (SwiGLU inter_dim=1536)
H_cycles: 2, L_cycles: 2 (4 steps per segment)
H_layers: 4, L_layers: 4 (Transformer blocks per module)
halt_max_steps: 16 (ACT segments)
pos_encodings: RoPE (base=10000)
forward_dtype: bfloat16
rms_norm_eps: 1e-5
```

Key design choices:
- **Post-Norm** (NOT Pre-Norm): `x = RMSNorm(x + sublayer(x))`
- **RMSNorm has NO learnable parameters** — it's a pure function
- **1-step gradient**: all recurrent steps except the last run under `torch.no_grad()`
- **Initial states are fixed buffers**, not learned parameters
- **Output comes from z_H** (H-module), not z_L
- **Q-values from first token**: `q_head(z_H[:, 0])`
- **L receives**: z_H + input_embeddings (additive injection)
- **H receives**: z_L (additive injection)
- **stablemax** (not softmax) for small-sample experiments, computed in float64
- **AdamATan2** optimizer (scale-invariant Adam)
- **FlashAttention** (fa2 or fa3), non-causal

## Code Style

- Python 3.10+, PyTorch 2.x
- Type hints on function signatures
- Docstrings on classes and non-trivial functions
- Config via pydantic BaseModel classes
- Use dataclasses for carry/state objects
- Keep modules focused — one concept per file
- Tests in tests/ directory, named test_*.py
- Use torch.compile compatibility — avoid .item() calls in forward loops

## Dependencies

```
torch (with CUDA 12.x)
flash-attn (or flash-attn-interface for Hopper GPUs)
adam-atan2
einops
tqdm
pydantic
omegaconf
hydra-core
wandb
numpy
```
## Naming Conventions

Wherever we are implementing analogs from HRM, rename those to reflect CORAL or coral. Do not use HRM in any naming or code.

## Current State (Phases 0–3b complete; Phase 3c code complete 2026-04-21 — ready for training run)

```
coral/
├── models/
│   ├── common.py              ✅ trunc_normal_init_, rms_norm
│   ├── layers.py              ✅ CastedLinear, CastedEmbedding, RotaryEmbedding, Attention, SwiGLU
│   ├── transformer_block.py   ✅ TransformerBlock (Post-Norm), TransformerBlockConfig
│   ├── reasoning_module.py    ✅ ReasoningModule
│   ├── coral_base.py          ✅ CoralInner, CoralConfig, InnerCarry
│   │                             Phase 3b: moe_num_modes, lambda_moe_recon, lambda_moe_balance
│   │                             Phase 3a deprecated fields (crystal_confidence_threshold, lambda_crystal) DELETED
│   ├── coral_v3.py            ✅ CoralV3Inner — Phase 1 + Phase 3b/3c dispatcher (PC-only path active)
│   │                             Phase 3b: SpatialMoECodebook integrated; soft mixing at every H-cycle
│   │                             Phase 3b: PredMetrics updated (moe_recon_loss, moe_lb_loss, moe_passthrough_weight)
│   │                             Phase 3b: columnar routing paths stubbed (NotImplementedError)
│   │                             Phase 3b: consolidate_codebook() uses spatial k-means
│   │                             Phase 3c: PredMetrics gains moe_routing_entropy, moe_codebook_util_frac
│   │                             Phase 3c: _compute_moe_losses returns L_recon during eval (for logging)
│   ├── crystallization.py     ✅ SpatialMoECodebook, CrystallizationBuffer
│   │                             Phase 3b Session 1: SpatialMoECodebook (unweighted L_recon, always-active L_lb)
│   │                             Phase 3b Session 1: CrystallizationBuffer.add() + consolidate_spatial()
│   │                             Phase 3b Session 2: RecognitionNetwork + BCE supervision DELETED
│   │                             PERF: CrystallizationBuffer rewritten with pre-allocated tensors;
│   │                             vectorised add() eliminates Python loop (fixes ~11×/step slowdown)
│   │                             Phase 3c: L_lb Option Y (non-normalized KL over all K+1 experts)
│   │                             Phase 3c: @torch.compiler.disable(recursive=False) on moe_losses()
│   ├── prediction.py          ✅ PredictionNet, PrecisionNet
│   ├── columnar.py            ✅ ColumnarReasoningModule, ColumnarTransformerBlock
│   └── sparse_embedding.py    ✅ CastedSparseEmbedding, CastedSparseEmbeddingSignSGD_Distributed
├── training/
│   ├── losses.py              ✅ stablemax_cross_entropy, softmax_cross_entropy,
│   │                             ACTLossHead, CoralV3LossHead
│   │                             Phase 3b: BCE crystal loss replaced by L_recon + L_lb; new crystal/* metrics
│   │                             Phase 3c: routing_entropy + codebook_utilisation_frac logged train+eval
│   │                             Phase 3c: recon_loss logged independently for eval; lb_loss train-only
│   ├── act.py                 ✅ CoralACT, CoralV3ACT
│   │                             Phase 3b: forwards moe_recon_loss, moe_lb_loss, moe_passthrough_weight
│   │                             Phase 3c: forwards moe_routing_entropy, moe_codebook_util_frac
│   └── scheduler.py           ✅ cosine_schedule_with_warmup_lr_lambda
└── data/
    ├── puzzle_dataset.py      ✅ PuzzleDataset, PuzzleDatasetMetadata, create_dataloader
    ├── common.py              ✅ PuzzleDatasetMetadata, dihedral_transform (ported from HRM, Apache-2.0)
    ├── build_sudoku_dataset.py ✅ Sudoku-Extreme dataset builder (ported from HRM, Apache-2.0)
    │                             --coral-naming flag (default=True) for train__/test__ prefix output
    └── README.md              ✅ Build instructions and post-build workflow
scripts/
└── train.py                   ✅ Hydra-based training entry point
                                  NEW: load_warmstart_checkpoint(), bootstrap-phase consolidation
                                  logic, train/crystal/* W&B metrics
                                  FIX: torch 2.6/2.7 dynamo cache_size_limit/recompile_limit compat
                                  PERF: torch.set_float32_matmul_precision("high") for TF32 on A100
                                  LOG: flat eval/<metric> aliases for W&B cross-run overlay
                                  Phase 3c: consolidation-skipped log message clarified post-bootstrap
configs/
├── base.yaml                  ✅ Default hyperparameters
├── phase3a_crystal_warmstart.yaml  ✅ PC + crystal warm-start config for validation run
├── phase3b_moe_codebook.yaml  ✅ Phase 3b config (satisfied-owl run, normalized L_lb)
└── phase3c_moe_lb_specialization.yaml  ✅ Phase 3c config (Option Y L_lb, same hyperparams as satisfied-owl)
tests/
├── test_layers.py             ✅
├── test_coral_base.py         ✅
├── test_act.py                ✅
├── test_prediction.py         ✅
├── test_columnar.py           ✅
├── test_crystallization.py    ✅  (updated for 3-tuple supervision loss return + 5 new buffer perf tests)
└── test_integration.py        ✅
```

### Key Experimental Results

| Run | W&B ID | Config | Final Accuracy | Status |
|-----|--------|--------|----------------|--------|
| calculating-caracara | kyi7327z | Baseline (pure PyTorch AdamATan2) | 41.2% | Complete |
| defiant-raccoon | xlxm6d3x | Phase 1 (bugged pi_reg) | 14% at 20K | Killed |
| **poetic-giraffe** | **mfno8t1y** | **Phase 1 (fixed pi_reg)** | **61.07%** | **Complete ✓** |
| **orchid-heron** | **jrtvvvi7** | **Baseline (fused AdamATan2)** | **54.2%** | **Complete ✓** |
| agate-cuckoo | v7cmw24l | Phase 2 (routing only, λ_bal=0.01) | 7.3% at 10K | Killed — collapsed |
| curly-manatee | — | Phase 2 (routing only, λ_bal=0.1) | 14.69% at 15K | Killed — collapsed |
| **jovial-avocet** | **hb4bi1fu** | **Phase 3a' (PC + crystal warm-start)** | **63.48%** | **Complete ✓** |
| **satisfied-owl** | — | **Phase 3b (Soft MoE, normalized L_lb)** | **66.05% @ step 52080** | **Complete ✓** |

### Phase 3a' — Completed (jovial-avocet, 2026-04-19)

Key findings:
- bypass_rate peaked at ~0.0012 (step 26040) then self-disabled — gate correctly learned single pooled codebook is too lossy
- codebook_utilisation: 6/256 = 2.34% reproducibly — L-state manifold has ~6 natural modes
- +2.4pp over poetic-giraffe (61.07% → 63.48%) — attribution unresolved (warm-start vs. crystal aux loss)
- Control run (`control-no-crystal` branch) in progress to disambiguate

### Phase 3b — Complete (satisfied-owl, 2026-04-21)

Architecture: Soft MoE Spatial Codebook — K=32 spatial codebook experts + 1 passthrough expert, softmax routing, no binary gate, unweighted reconstruction loss + normalized L_lb.
Result: 66.05% peak @ step 52080, parity-to-slight-win vs no-crystal control (65.58%).
Diagnostic: router converged to near-uniform routing (mean_codebook_weight ≈ 0.025 ≈ uniform 1/32 = 0.031). Normalized L_lb satisfied by uniform codebook routing → no specialization pressure.

### Phase 3c — Code Complete (2026-04-21)

Branch: `moe-lb-specialization` (from moe-codebook-design @ 511793d). 4 commits landed.
Config: `configs/phase3c_moe_lb_specialization.yaml` (same hyperparameters as satisfied-owl, seed=42).

Commit 1 (`601d7f5`): L_lb Option Y — non-normalized KL over all K+1 experts (passthrough + codebook).
  Old: KL(w_cb_norm || uniform(K)) — satisfied by uniform codebook at any passthrough weight.
  New: KL(w_mean || uniform(K+1)) — penalizes passthrough dominance AND uniform-codebook equilibrium.
  + 4 unit tests for Option Y correctness (zero at uniform-K+1, large at degenerate cases).

Commit 2 (`aa0665c`): @torch.compiler.disable(recursive=False) on moe_losses() — prevents dynamo
  recompilation storm (64 recompiles → 30% throughput loss) seen in satisfied-owl at step ~5074.

Commit 3 (`985e33d`): Consolidation log cleanup — post-bootstrap skips now print clear
  "buffer disabled post-bootstrap" message instead of misleading "usage=None".

Commit 4 (`21f5f2f`): New W&B metrics — crystal/routing_entropy and crystal/codebook_utilisation_frac
  for both train and eval. Also eval/crystal/recon_loss (eval now returns L_recon for logging).
  PredMetrics gains moe_routing_entropy, moe_codebook_util_frac fields.

Gate criteria: 150 passed / 48 skipped / 0 failures. Ready for Phase 3c training run.
Success criteria: exact_accuracy ≥ 0.65, routing_entropy > 1.0 post-warmup (specialization signal),
  codebook_utilisation_frac > Phase 3a baseline (6/32 = 0.1875), no eval checkpoint below 0.60.

See `PHASE3A_CHANGES.md` for launch commands and metric tracking guide.