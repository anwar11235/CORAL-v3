# CORAL v3 — Phase 1–3 Handoff Document

---

## Phase 1 Experiment Log (March 27, 2026)

### Bug found and fixed

**Precision regularizer sign bug** (`coral/training/losses.py:247`, commit `5651f0a`)

The original regularizer was:
```python
pi_reg = lambda_pi * (-0.5) * torch.log(pi + 1e-8).sum(dim=-1).mean()
```
`−log(π)` is minimized by pushing `π → ∞`, so the optimizer had a direct
incentive to explode precision rather than control it.  Observed on run
**defiant-raccoon**: `precision_mean` drifted from 0.8 → 3.5+ over 20K
steps, degrading eval accuracy and eventually killing the run.

Fixed to a symmetric log-normal prior centered at `π = 1`:
```python
pi_reg = lambda_pi * 0.5 * (torch.log(pi + 1e-8) ** 2).sum(dim=-1).mean()
```
`(log π)²` has its minimum at `log π = 0` (i.e. `π = 1`) and penalizes
deviation in both directions equally — collapse (`π → 0`) and explosion
(`π → ∞`) incur the same increasing cost.

---

### Infrastructure updates (commits `f3e46a2` → `cd41971`)

| Item | Detail |
|---|---|
| **Fused AdamATan2** | `adam-atan2-pytorch` (fused CUDA) confirmed working on torch 2.11+cu126. ~14% faster: 8.7 it/s vs 7.6 it/s pure PyTorch. `scripts/train.py` now auto-detects with fallback; prints backend at startup. |
| **`build_model` silent failure** | Was always constructing `CoralACT` + `ACTLossHead` regardless of Phase flags — `CoralV3ACT`/`CoralV3LossHead` were never used. Fixed: selects the V3 stack when any of `use_predictive_coding`, `use_columnar_routing`, `use_crystallization` is True. Would have been a completely silent failure in production. |
| **Phase 1-3 config fields** | `TrainConfig` was missing all Phase 1-3 fields; `build_model` was not forwarding them to `CoralConfig`. Both fixed. |
| **Codebook consolidation** | `consolidate_codebook()` existed in `CoralV3Inner` but was never called. Wired into training loop every `crystal_consolidation_interval` steps (default 5000). Logs `[CORAL-v3] Codebook consolidation at step N` and `train/codebook_usage` to W&B. |
| **Column warm-up schedule** | Added linear annealing of active columns `k` from `column_warmup_start_k` (default S=8) down to `active_columns` (default 2) over `column_warmup_steps` (default 5000) steps. `ColumnarTransformerBlock.k` updated in-place each step. Logs `train/active_columns`. |
| **W&B metric names** | Fixed metric key mismatches: `pred_error_norm` → `prediction_error`, `balance_loss` → `load_balance_loss`. Added `precision_std` (from `pi_final`), `crystal_confidence_mean` (accumulated across H-cycles), `crystal_bypass_rate` (eval only), `codebook_usage` (at consolidation). |
| **CPU dry-run verified** | All 4 Phase configs (Phase 2 only, Phase 3 only, Phase 1+2, Phase 1+2+3) verified via `scripts/verify_phase2_phase3.py`. Hydra `+flag=True` override syntax confirmed. 8/8 checks pass. |

---

### Phase 1 runs

| Run | Config | Outcome |
|---|---|---|
| **defiant-raccoon** | Phase 1, bugged `pi_reg` | Precision exploded (0.8 → 3.5+). Eval accuracy peaked at 14% at step 20K. Killed. |
| **poetic-giraffe** | Phase 1, fixed `pi_reg` | Eval accuracy **22.9%** at step 20K vs baseline 17.1% — **+34% relative**. `precision_mean` stable at 0.6–0.75. Still running (~40% complete). |

Baseline (Phase 0, no predictive coding): **17.1%** eval accuracy at step 20K.

---

### Fused optimizer baseline

| Run | Config | Outcome |
|---|---|---|
| **ruby-dalmatian** | Fused AdamATan2, Phase 0 | Tracking similarly to pure-PyTorch baseline (**calculating-caracara**) at step 5K. Killed early after confirming equivalent trajectory — fused backend is a drop-in. |

---

### Next steps when poetic-giraffe completes

**Decision gate:**

- **If final accuracy ≥ 41.2%** (Phase 0 final baseline): Phase 1 is a confirmed improvement. Proceed directly to Phase 2.
- **If final accuracy < 41.2%**: Diagnose before proceeding (check `precision_mean` trajectory, `prediction_error` curve, learning rate).

**Planned run sequence:**

1. **Phase 2** — columnar routing only:
   ```
   python scripts/train.py data_path=... +use_columnar_routing=True
   ```
2. **Phase 1 + 2** — predictive coding + routing combined:
   ```
   python scripts/train.py data_path=... +use_predictive_coding=True +use_columnar_routing=True
   ```
3. **Phase 3** — add crystallization to best Phase 1+2 config:
   ```
   python scripts/train.py data_path=... +use_predictive_coding=True +use_columnar_routing=True +use_crystallization=True
   ```

Use the same base hyperparameters as poetic-giraffe for all Phase 2/3 runs to
ensure clean ablation comparisons.  Monitor `train/active_columns` to confirm
column warm-up is annealing as expected.
