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

## Docker Image

Current production image: `anwar1919/coral-v3:2026-04-20` (also `:latest`)
Previous stable anchor: `anwar1919/coral-v3:2026-04-19` (preserved, do not delete)
Digest: `sha256:46f7946dfbe75a600fb2ab95206f699609d4ec1d091078e3da5366d94f5236f6`

Baked-in fixes (as of 2026-04-20 rebuild):
- flash-attn pinned to 2.7.4.post1 (ABI-compatible with torch 2.6; 2.8.x crashes)
- /workspace created + WORKDIR (all handoff docs assume /workspace)
- pytest pre-installed
- adam-atan2-pytorch==0.3.2 installed (fused CUDA path; avoids pure-PyTorch fallback)

## Current State (Phases 0–3 complete; Phase 3a' complete — jovial-avocet 63.48%)

```
coral/
├── models/
│   ├── common.py              ✅ trunc_normal_init_, rms_norm
│   ├── layers.py              ✅ CastedLinear, CastedEmbedding, RotaryEmbedding, Attention, SwiGLU
│   ├── transformer_block.py   ✅ TransformerBlock (Post-Norm), TransformerBlockConfig
│   ├── reasoning_module.py    ✅ ReasoningModule
│   ├── coral_base.py          ✅ CoralInner, CoralConfig, InnerCarry
│   │                             NEW: crystal_bootstrap_steps field
│   ├── coral_v3.py            ✅ CoralV3Inner — Phase 1/2/3 dispatcher
│   │                             NEW: _crystal_gate_active flag, PredMetrics diagnostic fields
│   │                             FIX: nearest_code.to(z_L.dtype) in bypass (flash-attn fp16/bf16 req)
│   ├── crystallization.py     ✅ RecognitionNetwork, CrystallizationBuffer
│   │                             PERF: CrystallizationBuffer rewritten with pre-allocated tensors;
│   │                             vectorised add() eliminates Python loop (fixes ~11×/step slowdown)
│   │                             PERF: @torch._dynamo.disable on add() (CPU-side op, no compile benefit)
│   │                             NEW: consolidate(is_first_consolidation=), crystallization_diagnostics()
│   │                             CHANGED: crystallization_supervision_loss() now returns 3-tuple
│   ├── prediction.py          ✅ PredictionNet, PrecisionNet
│   ├── columnar.py            ✅ ColumnarReasoningModule, ColumnarTransformerBlock
│   └── sparse_embedding.py    ✅ CastedSparseEmbedding, CastedSparseEmbeddingSignSGD_Distributed
├── training/
│   ├── losses.py              ✅ stablemax_cross_entropy, softmax_cross_entropy,
│   │                             ACTLossHead, CoralV3LossHead
│   │                             NEW: logs crystal_reconstruction_error, crystal_target_confidence_mean
│   ├── act.py                 ✅ CoralACT, CoralV3ACT
│   │                             NEW: forwards crystal diagnostic tensors to outputs dict
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
configs/
├── base.yaml                  ✅ Default hyperparameters
└── phase3a_crystal_warmstart.yaml  ✅ PC + crystal warm-start config for validation run
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

### Phase 3a' — Completed (jovial-avocet, 2026-04-19)

Key findings:
- bypass_rate peaked at ~0.0012 (step 26040) then self-disabled — gate correctly learned single pooled codebook is too lossy
- codebook_utilisation: 6/256 = 2.34% reproducibly — L-state manifold has ~6 natural modes
- +2.4pp over poetic-giraffe (61.07% → 63.48%) — attribution unresolved (warm-start vs. crystal aux loss)
- Control run config ready (`configs/phase3a_control_warmstart.yaml` on `control-no-crystal`, commit `03d32b3`) — awaiting Vast launch to disambiguate

### Phase 3b — Spec Revision 2 complete (2026-04-20); awaiting control result before implementation

Architecture spec: `docs/CORAL_v3_Phase3b_MoE_Codebook_Spec.md` (branch: `moe-codebook-design`, commit `faf8664`)
Design: Soft MoE Spatial Codebook — K=32 spatial codebook experts + 1 passthrough expert, softmax routing, no binary gate, reconstruction loss replaces BCE.
Status: Spec finalized. Implementation blocked on control run result (decision gate §8/§9 Risk 1).
Tightenings (2026-04-20): single-metric Euclidean k-means (Commit 3); hard 10× kill-threshold for L_recon (Risk 3); hard 1.5× action trigger for pred_error stratification (§4.1); stub columnar routing dispatch paths with NotImplementedError (Commit 4 + Risk 5 RESOLVED).
Success criteria: exact_accuracy ≥ 0.65, mean_codebook_weight ≥ 0.15, codebook_utilization observational (K=32 for disambiguation), no eval checkpoint below 0.60.

See `PHASE3A_CHANGES.md` for launch commands and metric tracking guide.