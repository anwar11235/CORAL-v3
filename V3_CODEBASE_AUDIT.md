# CORAL v3 Codebase Audit
**Generated:** 2026-04-18  
**Purpose:** Read-only structural audit before adding crystallization extension  
**Scope:** All files in `coral/`, `scripts/`, `configs/`, `tests/`, `docs/`

---

## Section 1: Repository Layout

### Top-Level Directory Structure (2 levels)

```
CORAL-v3/
├── configs/
│   └── base.yaml                   ← single Hydra config file
├── coral/
│   ├── data/
│   │   ├── puzzle_dataset.py
│   │   └── __init__.py
│   ├── models/
│   │   ├── columnar.py             ← Phase 2: ColumnarReasoningModule, ColumnarTransformerBlock
│   │   ├── common.py               ← trunc_normal_init_, rms_norm
│   │   ├── coral_base.py           ← CoralConfig, InnerCarry, CoralInner
│   │   ├── coral_v3.py             ← CoralV3Inner (Phase 1/2/3 dispatcher)
│   │   ├── crystallization.py      ← RecognitionNetwork, CrystallizationBuffer
│   │   ├── layers.py               ← CastedLinear, RotaryEmbedding, Attention, SwiGLU
│   │   ├── prediction.py           ← PredictionNet, PrecisionNet
│   │   ├── reasoning_module.py     ← ReasoningModule (stack of TransformerBlocks)
│   │   ├── sparse_embedding.py     ← CastedSparseEmbedding
│   │   ├── transformer_block.py    ← TransformerBlock, TransformerBlockConfig
│   │   └── __init__.py
│   ├── training/
│   │   ├── act.py                  ← CoralACT, CoralV3ACT (ACT wrappers)
│   │   ├── adam_atan2.py           ← Pure-PyTorch AdamATan2 fallback
│   │   ├── losses.py               ← ACTLossHead, CoralV3LossHead, loss functions
│   │   ├── scheduler.py            ← cosine_schedule_with_warmup_lr_lambda
│   │   └── __init__.py
│   └── __init__.py
├── scripts/
│   ├── plot_precision_dynamics.py  ← figure generation (offline, not a training script)
│   ├── smoke_test.py               ← quick CPU forward pass sanity check
│   ├── train.py                    ← PRIMARY TRAINING ENTRY POINT
│   └── verify_phase2_phase3.py     ← CPU dry-run for Phase 2/3 configs
├── tests/
│   ├── test_act.py
│   ├── test_columnar.py
│   ├── test_coral_base.py
│   ├── test_crystallization.py
│   ├── test_integration.py
│   ├── test_layers.py
│   ├── test_prediction.py
│   └── __init__.py
├── docs/
│   ├── CORAL_v3_Claude_Code_Build_Plan.md
│   ├── CORAL_v3_Handoff_Phase1_3.md
│   ├── CORAL_v3_Handoff_Session2_to_3.md
│   ├── CORAL_v3_Handoff_Session2_to_3_updated.md
│   ├── CORAL_v3_Handoff_Session3_to_4.md
│   ├── CORAL_v3_Implementation_Plan.md
│   └── HRM_Codebase_Analysis.md
├── pyproject.toml
├── requirements.txt
├── CLAUDE.md
└── .gitignore
```

### Entry Points

| Script | Purpose |
|--------|---------|
| `scripts/train.py` | Primary training entry point. Hydra-decorated `main()` at line 476. |
| `scripts/smoke_test.py` | Quick CPU forward pass check; not a full training run. |
| `scripts/verify_phase2_phase3.py` | CPU dry-run validating Phase 2/3 configs build and forward correctly. |
| `scripts/plot_precision_dynamics.py` | Offline figure generation from saved metrics JSON; no GPU needed. |

There is **no dedicated eval script** — evaluation is embedded in the training loop at `scripts/train.py:361–415` and triggered every `eval_interval` steps.

There is **no data build script** in the repo; the dataset is pre-built and pointed to via `data_path` in the config.

### Config System

**Framework:** Hydra 1.3+  
**Config file:** `configs/base.yaml` (the only config file; no multi-run or sweep configs present)  
**Config decorator:** `@hydra.main(config_path="../configs", config_name="base", version_base=None)` at `scripts/train.py:476`  
**Override syntax:** Hydra command-line overrides (`key=value`); boolean flags use `+flag=True`

The config is parsed into a pydantic `TrainConfig` dataclass (`scripts/train.py:53–121`) which validates and type-coerces all values before use. Model hyperparameters are forwarded into `CoralConfig` (a pydantic `BaseModel`) at `scripts/train.py:144–179`.

### Checkpoint Save and Load

**Save location:** `checkpoints/{project_name}/{run_name}/` — auto-constructed at `scripts/train.py:501–504` when `checkpoint_path` is null in config.

**Save function:** `save_checkpoint()` at `scripts/train.py:423–468`.  
- Serializes `state.model.state_dict()` via `torch.save()` at line 443.  
- File naming: `{run_name}_step{state.step}.pt`  
- Retention policy: keeps best (highest `exact_accuracy`) and latest; deletes all others (lines 459–466).  
- Save is triggered at every eval cycle if `RANK == 0` and `checkpoint_path` is set.

**Load:** No automatic resume logic is present in `train.py`. Loading a checkpoint would require manual `model.load_state_dict(torch.load(...))` before calling `main()`. There is no `--resume` flag or warm-start path.

---

## Section 2: Model Architecture

### Top-Level Model Class

The model presented to the training loop is always a **loss head** wrapping an **ACT wrapper** wrapping the **inner model**:

```
CoralV3LossHead           ← coral/training/losses.py:289
  └── CoralV3ACT          ← coral/training/act.py (wraps inner, runs ACT loop)
        └── CoralV3Inner  ← coral/models/coral_v3.py:97
              └── CoralInner  ← coral/models/coral_base.py:109 (base class)
```

When all Phase 1/2/3 flags are `False`, the simpler stack is used:

```
ACTLossHead               ← coral/training/losses.py
  └── CoralACT
        └── CoralInner
```

**`CoralInner.forward()` signature** (`coral/models/coral_base.py:292–296`):
```python
def forward(
    self,
    carry: InnerCarry,
    batch: Dict[str, torch.Tensor],
) -> Tuple[InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
```
Returns `(new_carry, output_logits [B, seq_len, vocab_size], (q_halt [B], q_continue [B]))`.

**`CoralV3Inner.forward()` signature** (`coral/models/coral_v3.py:271–276`):
```python
def forward(
    self,
    carry: InnerCarry,
    batch: Dict[str, torch.Tensor],
    is_last_segment: bool = False,
) -> Tuple:
```
Returns a 3-tuple (same as `CoralInner`) when all mechanisms disabled, or a 4-tuple `(new_carry, output, (q_halt, q_continue), PredMetrics)` when any Phase 1/2/3 is active.

### Hierarchy Structure

There are **two levels**, H (higher abstraction) and L (lower abstraction):

| Level | Class | Layers | State | Role |
|-------|-------|--------|-------|------|
| H | `ReasoningModule` (`reasoning_module.py:12`) | `H_layers` (default 4) TransformerBlocks | `z_H [B, total_seq_len, hidden_size]` | Produces output logits and halt Q-values |
| L | `ReasoningModule` (same class) | `L_layers` (default 4) TransformerBlocks | `z_L [B, total_seq_len, hidden_size]` | Feeds into H; receives H prediction as injection |

**Level communication (baseline, no PC):**
- `z_L` receives `z_H + input_embeddings` as the `hidden_states` injection (`coral_base.py:329`)
- `z_H` receives `z_L` as injection (`coral_base.py:333`)

**Level communication (with Phase 1 PC):**
- `z_L` receives `prediction_net(z_H) + input_embeddings` — H's *prediction* of L's state (`coral_v3.py:418–419`)
- `z_H` receives `pi * epsilon` — precision-weighted prediction error (`coral_v3.py:422, 427`)

`ReasoningModule.forward()` (`reasoning_module.py`) accepts `(hidden_states, injection, cos_sin)` and returns updated hidden states. Injection is added to `hidden_states` before each TransformerBlock.

### Recurrence Structure

**Per-segment loop** (inside a single `CoralInner.forward()` call, `coral_base.py:321–342`):

```
with torch.no_grad():
    for h_step in range(H_cycles):         # 2 outer iterations
        for l_step in range(L_cycles):     # 2 inner iterations
            if not is_last_l:
                z_L = L_level(z_L, ...)    # 3 L-steps under no_grad
        if not is_last_h:
            z_H = H_level(z_H, ...)        # 1 H-step under no_grad

# 1-step gradient:
z_L = L_level(z_L, ...)                   # final L-step (in graph)
z_H = H_level(z_H, ...)                   # final H-step (in graph)
```

With `H_cycles=2, L_cycles=2`, there are 4 recurrent steps per segment. Only the last L-step and last H-step contribute to the loss gradient.

**What persists across ACT segments** (via `InnerCarry`):
- `z_H [B, total_seq_len, hidden_size]` — always detached before being passed to next segment (`coral_base.py:348`)
- `z_L [B, total_seq_len, hidden_size]` — same

**What resets:** When a sequence halts and a new puzzle begins, both `z_H` and `z_L` are reset to their fixed initial buffers (`H_init`, `L_init`) via `reset_carry()` at `coral_base.py:271–286`. The reset is conditional on `reset_flag [B, bool]` passed from the ACT wrapper.

**Outer ACT loop:** Managed by `CoralACT`/`CoralV3ACT` (`training/act.py`). Max segments: `halt_max_steps=16`. Halting is Q-learning based — sequences halt individually; batch processing continues until all halt or max steps reached.

### Tensor Shapes at Each Stage

| Stage | Shape | Notes |
|-------|-------|-------|
| Input tokens | `[B, seq_len]` int | `seq_len=81` for Sudoku |
| Input embeddings | `[B, total_seq_len, hidden_size]` | `total_seq_len = seq_len + puzzle_emb_len` |
| Puzzle embed prefix | `[B, 1, hidden_size]` | 1 token for `puzzle_emb_ndim=512`, prepended |
| `z_H`, `z_L` | `[B, total_seq_len, hidden_size]` | `= [B, 82, 512]` with default config |
| Attention Q/K/V | `[B, total_seq_len, num_heads, head_dim]` | `head_dim = 512/8 = 64` |
| SwiGLU intermediate | `[B, total_seq_len, inter_dim]` | `inter_dim = 1536` (from `expansion=4, hidden_size=512`) |
| Output logits | `[B, seq_len, vocab_size]` | puzzle prefix stripped (`coral_base.py:351`) |
| Q-values | `[B, 2]` float32 | from first token of `z_H` (`coral_base.py:354`) |
| Prediction error `ε` | `[B, total_seq_len, hidden_size]` | Phase 1 only |
| Precision `π` | `[B, total_seq_len, hidden_size]` | Phase 1 only; always `> 0.01` |

`B` is `global_batch_size // world_size` (e.g., 384 on single GPU). The `[B, total_seq_len, hidden_size]` shapes also apply to `mu_L`, `xi` (precision-weighted error), and all intermediate carry tensors.

### Precision-Weighted Predictive Coding (Phase 1)

#### Where prediction errors are computed

Inside `CoralV3Inner._forward_with_pc()` (`coral_v3.py:379`), at two locations:

**Under `torch.no_grad()` (H×L − 1 steps, `coral_v3.py:412–424`):**
```python
mu_L = self.prediction_net(z_H)                      # line 418
z_L = self.L_level(z_L, mu_L + input_embeddings, ...) # line 419
epsilon = z_L - mu_L                                  # line 420  ← prediction error
pi = self.precision_net(z_L)                          # line 421
xi = pi * epsilon                                     # line 422  ← precision-weighted error
pred_error_norms.append(epsilon.norm(dim=-1).mean())  # line 423 (detached, for logging)
```

**In the 1-step-grad section (`coral_v3.py:436–441`):**
```python
mu_L = self.prediction_net(z_H)                      # line 437
z_L = self.L_level(z_L, mu_L + input_embeddings, ...) # line 438
epsilon_final = z_L - mu_L                           # line 439  ← in-graph for loss
pi_final = self.precision_net(z_L)                   # line 440  ← in-graph for loss
xi = pi_final * epsilon_final                        # line 441
```

The same pattern appears in `_forward_with_pc_and_routing()` at `coral_v3.py:551–606`.

#### Where precision is computed

`PrecisionNet` class, `coral/models/prediction.py:44–70`:
```python
class PrecisionNet(nn.Module):
    EPS_MIN: float = 0.01                             # line 54

    def forward(self, z_L: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.fc2(F.gelu(self.fc1(z_L)))) + self.EPS_MIN  # line 70
```

Architecture: Two-layer MLP (`dim → dim → dim`), CastedLinear (no bias), GELU activation, softplus output with `EPS_MIN=0.01` floor. There is **no EMA** — precision is computed fresh on every forward call from the current `z_L`.

#### Where PC loss enters total loss

`CoralV3LossHead.forward()`, `coral/training/losses.py:397–409`:
```python
if "epsilon_final" in outputs and outputs["epsilon_final"] is not None:
    lambda_pred = self.model.config.lambda_pred   # line 399
    lambda_pi = self.model.config.lambda_pi       # line 400
    pred_loss, pi_reg = predictive_coding_loss(
        outputs["epsilon_final"],
        outputs["pi_final"],
        lambda_pred=lambda_pred,
        lambda_pi=lambda_pi,
    )
    total_loss = total_loss + pred_loss + pi_reg  # line 407
```

The PC loss is **unconditionally additive** — there is no warmup or annealing of `lambda_pred` or `lambda_pi`.

#### Precision regularizer form

`coral/training/losses.py:248–250`:
```python
pred_loss = lambda_pred * 0.5 * (pi * epsilon ** 2).sum(dim=-1).mean()
pi_reg    = lambda_pi   * 0.5 * (torch.log(pi + 1e-8) ** 2).sum(dim=-1).mean()
```

- `pred_loss`: `(λ_pred / 2) · Σ π ε²` — precision-weighted squared prediction error
- `pi_reg`: `(λ_pi / 2) · Σ (log π)²` — symmetric log-normal regularizer centered at π=1

This matches the expected form `(λ_π/2)(log π)²`. The `1e-8` epsilon in `log()` provides numerical stability; it does NOT raise the floor (that role belongs to `PrecisionNet.EPS_MIN=0.01`).

**Historical note:** An earlier version used `lambda_pi * (-0.5) * torch.log(pi + 1e-8)` (a one-sided `−log π` term that rewarded large precision). That formula caused precision explosion in run `defiant-raccoon`. The current symmetric `(log π)²` form was introduced in commit `5651f0a` and is the version in the current codebase.

### Gradient Flow

All `detach()` / `no_grad()` calls, with file:line and rationale:

| # | Location | Construct | Rationale |
|---|----------|-----------|-----------|
| 1 | `coral_base.py:321` | `with torch.no_grad():` | 1-step gradient policy — all but the final L+H step run without accumulating graph nodes |
| 2 | `coral_base.py:336` | `assert not z_H.requires_grad and not z_L.requires_grad` | Sanity check verifying no gradient leaked from the no_grad block |
| 3 | `coral_base.py:348` | `InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())` | Deep supervision — no gradient flows between ACT segments; each segment receives only the value of the previous state |
| 4 | `coral_v3.py:151` | `@torch.compiler.disable(recursive=False)` | Compiler directive on `_maybe_crystal_bypass_nograd()` — Python conditionals and `.item()` calls inside the method are incompatible with `torch.compile` graph tracing |
| 5 | `coral_v3.py:209` | `@torch.compiler.disable(recursive=False)` | Same rationale on `_maybe_record_crystal()` — ring buffer operations with Python loops cannot be traced |
| 6 | `coral_v3.py:234` | `@torch.compiler.disable(recursive=False)` | Same rationale on `_compute_crystal_supervision_loss()` |
| 7 | `coral_v3.py:324` | `with torch.no_grad():` | Same as #1, in `_forward_baseline()` |
| 8 | `coral_v3.py:357` | `assert not z_H.requires_grad ...` | Same as #2 |
| 9 | `coral_v3.py:364` | `InnerCarry(z_H=z_H.detach(), ...)` | Same as #3 |
| 10 | `coral_v3.py:395` | `with torch.no_grad():` | Same as #1, in `_forward_with_pc()` |
| 11 | `coral_v3.py:434` | `assert not z_H.requires_grad ...` | Same as #2 |
| 12 | `coral_v3.py:447–448` | `.detach()` on `epsilon_final.norm()` and `pi_final.mean()` | Logging metrics only; detached so they don't influence the backward pass |
| 13 | `coral_v3.py:450` | `InnerCarry(z_H=z_H.detach(), ...)` | Same as #3 |
| 14 | `coral_v3.py:478` | `with torch.no_grad():` | Same as #1, in `_forward_with_routing()` |
| 15 | `coral_v3.py:511` | `assert not z_H.requires_grad ...` | Same as #2 |
| 16 | `coral_v3.py:518` | `InnerCarry(z_H=z_H.detach(), ...)` | Same as #3 |
| 17 | `coral_v3.py:551` | `with torch.no_grad():` | Same as #1, in `_forward_with_pc_and_routing()` |
| 18 | `coral_v3.py:590` | `assert not z_H.requires_grad ...` | Same as #2 |
| 19 | `coral_v3.py:606` | `InnerCarry(z_H=z_H.detach(), ...)` | Same as #3 |
| 20 | `coral_v3.py:603` | `.detach()` on `epsilon_final.norm()` | Logging metric only |
| 21 | `training/act.py:150` | `with torch.no_grad():` | Q-learning target computation — bootstrap values must not propagate gradients back through themselves (standard DQN-style target network logic) |
| 22 | `training/act.py:316` | `with torch.no_grad():` | Same as #21, in `CoralV3ACT` |
| 23 | `training/losses.py:343` | `with torch.no_grad():` | Computing accuracy metrics and binary target labels — these are scalar statistics, not differentiable losses |
| 24 | `crystallization.py:174–175` | `.detach().cpu()` on keys and values in `CrystallizationBuffer.add()` | Ring buffer stores CPU tensors to avoid long-term GPU memory occupation; detach prevents accidental gradient retention in the buffer |
| 25 | `crystallization.py:223` | `with torch.no_grad():` | Offline codebook consolidation — k-means assignment and EMA update; no gradients needed or desired |

---

## Section 3: Training Loop

### Optimiser

Two optimisers are used simultaneously (`scripts/train.py:202–229`):

**Main optimiser** (`train.py:219–227`):
```python
AdamATan2(
    model.parameters(),
    lr=1e-30,          # Near-zero init; scheduler sets effective LR each step
    weight_decay=config.weight_decay,   # default 1.0
    betas=(config.beta1, config.beta2), # default (0.9, 0.95)
)
```
`AdamATan2` is loaded from `adam-atan2-pytorch` (fused CUDA) with automatic fallback to `coral/training/adam_atan2.py` (pure PyTorch). The update rule replaces the standard `m/sqrt(v+ε)` scale with `atan2(m, sqrt(v))`, making the step magnitude bounded by π/2 regardless of gradient scale — implicit gradient clipping.

**Sparse embedding optimiser** (`train.py:208–216`):
```python
CastedSparseEmbeddingSignSGD_Distributed(
    model.puzzle_emb.buffers(),
    lr=1e-30,
    weight_decay=config.puzzle_emb_weight_decay,  # default 0.1
    world_size=world_size,
)
```
Only created when `puzzle_emb_ndim > 0`. Both optimisers start at `lr=1e-30` and have their learning rates set each step by a scheduler.

### LR Schedule

`cosine_schedule_with_warmup_lr_lambda()` from `coral/training/scheduler.py:6–38`.

```
Linear warmup: LR = base_lr * step / lr_warmup_steps     (steps 0 → 1000)
Cosine decay:  LR = base_lr * (min_ratio + (1−min_ratio) * 0.5*(1+cos(π*progress)))
```

- `base_lr = 7e-5` (default, `configs/base.yaml:30`)
- `lr_warmup_steps = 1000` (`base.yaml:32`)
- `lr_min_ratio = 0.1` — LR decays to 10% of base_lr at end of training (`base.yaml:31`)
- `puzzle_emb_lr = 1e-3` for the sparse embedding optimiser (`base.yaml:38`)
- Schedule applied every step at `train.py:276–283`

### Loss Composition

Every term, in order of addition:

| Term | Formula | Coefficient | Applies when |
|------|---------|-------------|--------------|
| `lm_loss` | Cross-entropy (stablemax or softmax) on predicted vs. target tokens | 1.0 | Always |
| `q_halt_loss` | BCE(`q_halt_logits`, `seq_is_correct`) | 0.5 | Always |
| `q_continue_loss` | BCE(`q_continue_logits`, `target_q_continue`) | 0.5 | When `target_q_continue` present in outputs |
| `pred_loss` | `λ_pred · 0.5 · Σ(π · ε²)` | `lambda_pred` (default 0.1) | Phase 1 (`use_predictive_coding=True`) |
| `pi_reg` | `λ_pi · 0.5 · Σ(log π)²` | `lambda_pi` (default 0.01) | Phase 1 |
| `balance_loss` | KL(router empirical dist. ‖ Uniform(S)) | `lambda_balance` (default 0.1) | Phase 2 (`use_columnar_routing=True`) |
| `crystal_loss` | BCE(confidence gate, reconstruction error < tol) | `lambda_crystal` (default 0.1) | Phase 3 training only (`use_crystallization=True`) |

Sources: `coral/training/losses.py:371–458` for all terms.

Base total: `lm_loss + 0.5*(q_halt_loss + q_continue_loss)` (`losses.py:395`)  
With PC: `+ pred_loss + pi_reg` (`losses.py:407`)  
With routing: `+ lambda_balance * balance_loss` (`losses.py:426`)  
With crystallization: `+ lambda_crystal * crystal_loss` (`losses.py:447`)

### Curriculum / Phase Ordering

**Column warmup** (`train.py:232–247`): When `use_columnar_routing=True`, active columns `k` is linearly annealed from `column_warmup_start_k` (default 8, i.e., all columns) down to `active_columns` (default 2) over `column_warmup_steps` (default 10,000) training steps. This is implemented by updating `ColumnarTransformerBlock.k` in-place each step.

**Crystallization codebook consolidation** (`train.py:567–578`): When `use_crystallization=True`, `inner.consolidate_codebook()` is called every `crystal_consolidation_interval` steps (default 5,000). This runs k-means assignment + EMA on the `CrystallizationBuffer` and clears it.

There is **no phase-level activation warmup** for predictive coding (both `lambda_pred` and `lambda_pi` are active from step 1). There is no curriculum on `lambda_balance` or `lambda_crystal`.

### Gradient Clipping, NaN Guards, Mixed Precision

**Gradient clipping:** None explicit. The bounded nature of `AdamATan2`'s update rule (`atan2(m, sqrt(v))`, max magnitude π/2) provides implicit bounding, but there is no `torch.nn.utils.clip_grad_norm_()` call anywhere in `train.py`.

**NaN guards:** None. There is no `torch.isnan()` check or loss scaling logic.

**Mixed precision:**  
- All `CastedLinear` and `CastedEmbedding` layers cast their output to `forward_dtype` (`bfloat16` by default) at `layers.py`.  
- `stablemax_cross_entropy` computes in `float64` for numerical stability (`losses.py:60`).  
- `rms_norm` in `common.py` upscales to `float32` for the norm computation then casts back to the input dtype (`common.py:82–84`).  
- Q-values are explicitly cast to `float32` (`coral_base.py:354`).  
- There is no `torch.autocast()` context — dtype is controlled per-layer via `CastedLinear`.

### Eval Cadence and Metrics

Evaluation runs every `eval_interval` steps (default 2,000, `base.yaml:8`) at `train.py:580–585`.

**Metrics computed** (`losses.py:343–368`, plus phase-specific additions):

| Metric | Meaning |
|--------|---------|
| `accuracy` | Per-token prediction accuracy over valid (non-ignore) positions |
| `exact_accuracy` | Fraction of sequences where every token is correct |
| `q_halt_accuracy` | Whether halt logit sign matches sequence correctness |
| `steps` | Mean number of ACT segments used per sequence |
| `lm_loss`, `q_halt_loss`, `q_continue_loss` | Per-batch loss terms |
| `pred_loss`, `pi_reg` | Phase 1 loss terms |
| `prediction_error`, `precision_mean`, `precision_std` | Phase 1 dynamics |
| `load_balance_loss`, `router_entropy`, `col_{i}_freq` | Phase 2 routing health |
| `crystal_supervision_loss`, `crystal_bypass_count`, `crystal_confidence_mean` | Phase 3 |

Metrics are logged to W&B under `eval/{set_name}/{metric}` keys (`train.py:584`). The `exact_accuracy` metric determines which checkpoint is kept as best.

---

## Section 4: Known-Good Configuration

### Run ID Discrepancy

> **IMPORTANT:** The audit prompt identifies the 61.07% run as W&B ID `xlxm6d3x`. This is incorrect. According to `docs/CORAL_v3_Handoff_Session2_to_3_updated.md:112–113`:
> - W&B ID `xlxm6d3x` = run **defiant-raccoon** = Phase 1 with **bugged** `pi_reg`, killed at 14% accuracy at 20K steps.
> - W&B ID `mfno8t1y` = run **poetic-giraffe** = Phase 1 with **fixed** `pi_reg`, final accuracy **61.07%**.

### poetic-giraffe (W&B mfno8t1y) — the 61.07% run

**Config file:** `configs/base.yaml` with command-line override `+use_predictive_coding=True`

| Parameter | Value | Source |
|-----------|-------|--------|
| `use_predictive_coding` | `True` | CLI override |
| `lr` | `7e-5` | base.yaml + CLI |
| `puzzle_emb_lr` | `7e-5` | CLI override (differs from base.yaml default of `1e-3`) |
| `weight_decay` | `1.0` | base.yaml |
| `puzzle_emb_weight_decay` | `1.0` | CLI override |
| `lambda_pred` | `0.1` | base.yaml default |
| `lambda_pi` | `0.01` | base.yaml default |
| `epochs` | `20000` | base.yaml |
| `eval_interval` | `2000` | base.yaml |
| `hidden_size` | `512` | base.yaml |
| `num_heads` | `8` | base.yaml |
| `H_cycles` / `L_cycles` | `2` / `2` | base.yaml |
| `H_layers` / `L_layers` | `4` / `4` | base.yaml |
| `halt_max_steps` | `16` | base.yaml |
| `forward_dtype` | `bfloat16` | base.yaml |
| `loss_type` | `stablemax_cross_entropy` | base.yaml |
| `AdamATan2` backend | Pure PyTorch (fused not yet available when run launched) | handoff doc |
| Parameters | 27.28M | `docs/CORAL_v3_Handoff_Session2_to_3_updated.md` |

**Launch command** (reconstructed from handoff docs):
```bash
OMP_NUM_THREADS=8 python scripts/train.py \
    data_path=/workspace/data/sudoku-extreme-1k-aug-1000 \
    epochs=20000 eval_interval=2000 \
    lr=7e-5 puzzle_emb_lr=7e-5 \
    weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
    +use_predictive_coding=True
```

**Training stats:**
- Total steps: 52,081 (`poetic_giraffe_train_metrics.json` has 52,081 rows per handoff doc)
- Hardware: A100-SXM4-40GB
- Speed: ~7.6 it/s (pure PyTorch AdamATan2 backend)
- Wall time: ~1.9 hours (52,081 steps / 7.6 it/s)
- Final eval exact accuracy: **61.07%** on Sudoku-Extreme-1K

**Key precision dynamics observed** (`docs/CORAL_v3_Handoff_Session2_to_3_updated.md`):
- Phase transition at step ~2,500–3,000: prediction error collapses from ~25 to <1
- Precision mean spikes from 0.7 to ~0.8 during transition, then settles to 0.6–0.75
- Precision std stable at ~0.2 post-transition (per Session 3 handoff: corrected to 0.01 std post-transition)

**Checkpoint location:** `C:\Users\mauha\coral_v3_results\phase1\phase1_best_checkpoint_61pct.pt` (111 MB, local machine; referenced in handoff doc, not in this repo).

---

## Section 5: Extension Points for Crystallisation

Crystallisation is already partially implemented in this codebase (`coral/models/crystallization.py`, `coral/models/coral_v3.py`). The following documents where a new or extended crystallisation mechanism would plug in.

### Where new per-level mechanisms plug in (parallel to PC module)

**1. Config declaration** — `coral/models/coral_base.py:66–84`  
Add boolean flag and any coefficient to `CoralConfig`. All Phase 1/2/3 flags live here. Example: `use_crystallization: bool = False` at line 78.

**2. Module instantiation** — `coral/models/coral_v3.py:105–145`  
The `CoralV3Inner.__init__()` conditionally builds phase modules. The pattern at lines 109–145:
```python
if config.use_predictive_coding:
    self.prediction_net = PredictionNet(...)
    self.precision_net = PrecisionNet(...)
if config.use_columnar_routing:
    ...
if config.use_crystallization:
    self.recognition_net = RecognitionNetwork(...)
    self.crystal_buffer = CrystallizationBuffer(...)
```
A new mechanism follows the same pattern.

**3. Forward dispatch** — `coral/models/coral_v3.py:291–305`  
The dispatch table at lines 295–305:
```python
if not pc and not cr and not cry:   return super().forward(...)
elif pc and cr:                     return self._forward_with_pc_and_routing(...)
elif pc:                            return self._forward_with_pc(...)
elif cr:                            return self._forward_with_routing(...)
else:                               return self._forward_baseline(...)
```
Adding a new flag requires extending this table and creating a corresponding `_forward_with_*()` method. Note that the 4-way dispatch is already O(2^n) in flags — with a fourth flag it would require up to 8 branches unless combined flags are treated uniformly.

**4. Forward implementation** — `coral/models/coral_v3.py:311–621`  
Each `_forward_with_*()` method follows the same skeleton:
1. `with torch.no_grad():` — run `H_cycles × L_cycles − 1` steps
2. `1-step-grad section` — run final L+H step
3. Return `(new_carry, output, (q_halt, q_continue), PredMetrics)`

New mechanisms insert calls within this skeleton.

**5. `PredMetrics` dataclass** — `coral/models/coral_v3.py:56–89`  
Add a new field to `PredMetrics` for any new in-graph tensor or logging scalar. Example: `crystal_supervision_loss_final: Optional[torch.Tensor]` at line 87.

**6. TrainConfig** — `scripts/train.py:53–121`  
Mirror all new `CoralConfig` fields here so they can be set via Hydra overrides.

**7. `build_model()`** — `scripts/train.py:144–179`  
Forward new `TrainConfig` fields to `CoralConfig`.

### Where codebook-style modules naturally live

`coral/models/crystallization.py` is the canonical home:
- `RecognitionNetwork` (line 36) — key computation, codebook lookup, confidence head
- `CrystallizationBuffer` (line 146) — ring buffer for offline codebook consolidation
- `crystallization_supervision_loss()` (line 264) — BCE gate training loss

A new codebook variant (e.g., a multi-level codebook, a product quantizer, or a VQ-VAE style module) would live in this file or in a new `coral/models/` file imported by `coral_v3.py`.

### Where `forward()` would need modification to consume a crystallised representation

All four `_forward_*()` methods already call `_maybe_crystal_bypass_nograd()` at the start of the `torch.no_grad()` section:

| Method | Call site |
|--------|-----------|
| `_forward_baseline()` | `coral_v3.py:331–338` |
| `_forward_with_pc()` | `coral_v3.py:401–410` |
| `_forward_with_routing()` | `coral_v3.py:484–493` |
| `_forward_with_pc_and_routing()` | `coral_v3.py:558–567` |

`_maybe_crystal_bypass_nograd()` at lines 151–207 substitutes `z_L = nearest_code` (line 193) and then runs H with the substituted z_L (lines 203–205).

A modified crystallisation mechanism would alter `_maybe_crystal_bypass_nograd()` directly, or add a parallel helper. If crystallisation should also affect the 1-step-grad section (e.g., for a differentiable codebook lookup), the modification point is the 1-step-grad block in each `_forward_*()` method — currently only `_compute_crystal_supervision_loss()` (line 248) is called there, before the final H update.

### Where additional logging would be added

**`PredMetrics` fields** (`coral_v3.py:56–89`): Add new scalars/tensors here and populate them in each `_forward_*()` method.

**`CoralV3ACT.forward()`** (`training/act.py`): Aggregates `PredMetrics` across segments and packages them into the `outputs` dict passed to the loss head.

**`CoralV3LossHead.forward()`** (`training/losses.py:326–458`): Reads from `outputs` dict and writes to `metrics` dict. New metric logging follows the pattern at lines 450–454:
```python
for key in ("crystal_bypass_count", "crystal_confidence_mean"):
    if key in outputs:
        val = outputs[key]
        metrics[key] = val.detach() if isinstance(val, torch.Tensor) else val
```

**W&B logging** (`train.py:563–564`): `wandb.log(metrics, step=state.step)` — all keys in `metrics` are automatically logged. No changes needed here.

---

## Section 6: Risk Inventory

### Hardcoded assumptions about level count, hidden dim, or batch shape

| Location | Assumption | Risk |
|----------|-----------|------|
| `coral_base.py:48–51` | `H_cycles=2, L_cycles=2, H_layers=4, L_layers=4` | Parameterised in `CoralConfig`; loop at line 324 uses `self.config.H_cycles`. **Low risk.** |
| `coral_v3.py:183` | `is_last_h = h_step == self.config.H_cycles - 1` | Dynamic — always uses config value. **Low risk.** |
| `coral_v3.py:226–227` | `_maybe_record_crystal` guards `is_last_h` and `is_last_segment` | With `H_cycles=1`, `is_last_h` would be True on step 0, potentially recording every step and overloading the buffer. A third level (N=3 hierarchy) would need this logic revisited. **Medium risk for N≠2.** |
| `prediction.py:29` | `PredictionNet(h_dim, l_dim)` — explicitly designed for h_dim ≠ l_dim | Already generalised for N=3. **Low risk.** |
| `crystallization.py:75` | `self.codebook = nn.Parameter(torch.randn(codebook_size, l_dim) * 0.01)` | Codebook values are shaped `[K, l_dim]`; if H and L have different hidden sizes (N=3), this is `l_dim` not `h_dim`. Consistent with current usage. **Low risk if l_dim is always the target level's dim.** |
| `crystallization.py:231` | `z_L.mean(dim=1)` — pools over sequence dimension | Assumes sequence dimension is always dim 1. True for all current shapes `[B, seq_len, dim]`. **Low risk.** |
| `losses.py:395` | Computes `loss_counts = mask.sum(-1)` — last dimension is token positions | Assumes logits are `[B, seq_len, vocab_size]`. **Low risk.** |
| `base.yaml:6` | `global_batch_size: 384` | Not hardcoded in model code; only in config and dataloader. **Low risk.** |

### Dead branches or disabled mechanisms

| Location | Description | Status |
|----------|-------------|--------|
| `coral_base.py:179` | `elif config.pos_encodings == "learned":` — learned position embedding path | Implemented but never tested; `"rope"` is always used. May have untested edge cases if activated. |
| `train.py:191` | `if "DISABLE_COMPILE" not in os.environ:` | Escape hatch to disable `torch.compile`. Not dead, but a runtime switch. |
| `train.py:25` | `torch._dynamo.config.recompile_limit = 64` | Workaround for variable `k` in columnar routing causing recompile storms. Would be unnecessary if routing is disabled. |
| `training/losses.py:88` | `softmax_cross_entropy()` — alternate loss function | Implemented and selectable via `loss_type`; not used in any known good run. |
| `training/act.py` | `CoralACT` (base) vs `CoralV3ACT` (Phase 1/2/3) | Both present; model selection in `build_model()` at `train.py:181–189` is conditional on flags. If all flags are False, `CoralACT`+`ACTLossHead` is used. This is correct behaviour, not dead code. |
| No commented-out crystallisation stubs found | The crystallisation code is fully wired in (not a stub), but bypass **only fires during eval** (`coral_v3.py:184`). Training always runs full recurrence. | Active code; behaviour is intentional. |

### Code paths assuming deterministic state transitions

| Location | Assumption | Risk for crystallisation |
|----------|-----------|--------------------------|
| `coral_v3.py:184` | `if not self.config.use_crystallization or self.training or is_last_h: return False, ...` | Crystallisation bypass is disabled during training by design. Any extension that makes bypass fire during training would break the 1-step gradient invariant — the `InnerCarry` detach at `coral_v3.py:364/450/518` would detach a bypassed state rather than a recurrently-computed one. This would be a silent correctness change. |
| `training/act.py` Q-learning | DQN bootstrap targets assume the value of the next state is produced by the same model. If crystallisation substitutes a stored codebook entry for `z_L`, the Q-value estimate for the bypassed step comes from a state that was not computed by the current model weights — this is sound during eval (where bypass fires) but would require careful analysis if bypass were ever moved into training. | Medium risk if design changes. |
| `coral_v3.py:183` | `is_last_h = h_step == self.config.H_cycles - 1` — bypass never fires on last H-step | This hardcoded exclusion ensures the final H+L step is always recurrent and can participate in the loss. Any codebook extension that needs to affect the last step would require removing this guard, which would mean the 1-step-grad section never runs for bypassed sequences. | High risk if violated. |
| `crystallization.py:75` | `codebook` is an `nn.Parameter` — updated via backprop AND offline EMA | The codebook receives gradients through the `crystallization_supervision_loss` confidence head training, but the codebook values themselves are updated only offline via `CrystallizationBuffer.consolidate()`. There is no straight-through estimator or VQ-style gradient path through the nearest-code lookup. A new mechanism requiring end-to-end differentiable codebook lookup (e.g., soft nearest-neighbor) would need new gradient infrastructure. | Medium risk. |

### Torch version, Python version, environment-sensitive bits

**Python:** `>=3.10` (`pyproject.toml:9`). Uses Python 3.10+ type hint syntax (union types with `|`, etc.). **Known incompatibility:** `docs/CORAL_v3_Handoff_Session3_to_4.md:80` documents that Python 3.14 Vast.ai images break Hydra's argparse; Python 3.12 is the tested version.

**PyTorch:** `>=2.0` (`pyproject.toml:11`). Uses:
- `torch.compile()` (`train.py:192`) — requires PyTorch 2.0+
- `torch._dynamo.config.recompile_limit` (`train.py:25`) — internal API, may change
- `F.scaled_dot_product_attention()` (`layers.py`) — requires PyTorch 2.0+
- `torch.compiler.disable` decorator (`coral_v3.py:151, 209, 234`) — requires PyTorch 2.x
- FlashAttention 2/3 optional; falls back to `F.scaled_dot_product_attention` when unavailable

**AdamATan2:** Auto-detects `adam-atan2-pytorch` (fused CUDA, requires torch 2.11+cu126 per handoff docs) with fallback to pure-PyTorch `coral/training/adam_atan2.py`. The pure-PyTorch version is ~14% slower.

**`torch.compile` + crystallisation interaction:** This is a documented, unresolved risk. The `@torch.compiler.disable(recursive=False)` decorators on the three crystallisation helper methods (`coral_v3.py:151, 209, 234`) were added specifically because the crystallisation code paths (Python conditionals in `_maybe_crystal_bypass_nograd`, ring buffer operations in `_maybe_record_crystal`) caused `torch.compile` to either hang during tracing or produce incorrect graphs. The decorators prevent those methods from being compiled, at the cost of a graph break at each call site. Any new crystallisation code that uses Python-level control flow, `.item()` calls, or CPU tensor operations must either receive the same decorator treatment or be designed to be fully traceable.

**`torch._dynamo.config.recompile_limit = 64`** (`train.py:25`): Set to suppress recompile-limit warnings from variable sub-batch sizes during columnar routing warmup. This is set globally at module import time and applies to all `torch.compile` invocations in the process.

**Stale `.pyc` files:** `docs/CORAL_v3_Handoff_Session3_to_4.md:81` documents that Python 3.14 `.pyc` files in `__pycache__` can cause silent import errors when switching to Python 3.12. This is an environment hazard when the repo is run on shared compute after being used with a different Python version.

---

## 5-Line Summary

1. **Architecture is clean and well-layered.** The dispatch table in `coral_v3.py:291–305` is the single integration point for new mechanisms; every new Phase follows the same `_forward_with_*()` skeleton and `PredMetrics` contract without touching the base `CoralInner`.

2. **Crystallisation code already exists but has never completed a successful training run.** All Phase 1+3 attempts were killed by severe performance regressions (0.17–1.33 it/s vs. target ~7 it/s); the `@torch.compiler.disable` decorators added in the current codebase are the proposed fix, but no validated run at acceptable speed exists yet.

3. **The 61.07% result belongs to run `poetic-giraffe` (W&B ID `mfno8t1y`), not `xlxm6d3x` (`defiant-raccoon`).** The latter was killed at 14% due to a precision-regularizer sign bug that was subsequently fixed.

4. **The hardcoded `is_last_h` bypass guard in `_maybe_crystal_bypass_nograd()` (`coral_v3.py:183–185`) is a structural invariant:** bypass never fires on the final H-cycle, ensuring the 1-step-grad section always runs. Any crystallisation extension that needs to affect the final step must explicitly remove this guard and reason about what happens to the backward pass.

5. **There is no gradient clipping, no NaN guard, and no checkpoint resume path.** The AdamATan2 bounded-update property provides implicit stability, but a new mechanism (e.g., a differentiable codebook with soft nearest-neighbor gradients) could introduce gradient dynamics outside AdamATan2's assumptions and would benefit from explicit clipping.
