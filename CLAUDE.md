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

## Current State (Phase 0 — Steps 0.1–0.9 complete)

```
coral/
├── models/
│   ├── common.py              ✅ trunc_normal_init_, rms_norm
│   ├── layers.py              ✅ CastedLinear, CastedEmbedding, RotaryEmbedding, Attention, SwiGLU
│   ├── transformer_block.py   ✅ TransformerBlock (Post-Norm), TransformerBlockConfig
│   ├── reasoning_module.py    ✅ ReasoningModule
│   ├── coral_base.py          ✅ CoralInner, CoralConfig, InnerCarry
│   └── sparse_embedding.py    ✅ CastedSparseEmbedding, CastedSparseEmbeddingSignSGD_Distributed
├── training/
│   ├── losses.py              ✅ stablemax_cross_entropy, softmax_cross_entropy, ACTLossHead
│   ├── act.py                 ✅ CoralACT, ACTCarry
│   └── scheduler.py           ✅ cosine_schedule_with_warmup_lr_lambda
└── data/
    └── puzzle_dataset.py      ✅ PuzzleDataset, PuzzleDatasetMetadata, create_dataloader
scripts/
└── train.py                   ✅ Hydra-based training entry point
configs/
└── base.yaml                  ✅ Default hyperparameters
tests/
├── test_layers.py             ✅
├── test_coral_base.py         ✅
└── test_act.py                ✅
```

**Still needed before Phase 1:**
- Step 0.10: Integration tests (test_forward_pass, test_1step_gradient)
- Step 0.11: Validation run on Sudoku-Extreme-1K (target ≥50% accuracy)