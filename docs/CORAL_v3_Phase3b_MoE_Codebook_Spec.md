# CORAL v3 Phase 3b — Spatially-Structured Soft MoE Codebook Architecture Spec

**Branch:** `moe-codebook-design`  
**Date:** 2026-04-19  
**Status:** Review gate — do not implement until approved

---

## 1. Executive Summary

**Design:** Replace the single-codebook hard-bypass mechanism with a Soft MoE Codebook: K=8 mode experts whose values are full-spatial tensors `[seq_len, l_dim]`, plus one passthrough expert (the recurrence output). A small router MLP computes softmax weights over K+1 experts; the final z_L is the resulting convex combination. No binary gate. No BCE supervision. Codebook values are trainable parameters updated end-to-end via a reconstruction loss.

**Problem solved:** The Phase 3a gate correctly identified that tiling a single pooled vector across 81 sequence positions is too lossy to trust — so it learned to suppress itself. The root cause is not the gating mechanism but the loss of spatial structure in codebook values. This design preserves spatial structure by construction.

**Expected outcomes against four criteria:**

| Criterion | Phase 3a result | Phase 3b target | Justification |
|---|---|---|---|
| `exact_accuracy` | 63.48% | ≥ 65.0% | Soft mixing + reconstruction loss ≥ Phase 3a's BCE auxiliary signal |
| `mean_codebook_weight` (bypass analog) | ~0.0001 | ≥ 0.15 | Spatial fidelity makes codebook safe to use; no self-disable incentive |
| `codebook_utilization` (fraction of K modes used) | 6/256 = 2.3% | ≥ 5/8 = 62.5% | K=8 matched to empirical ~6 modes; near-full coverage expected |
| PC stability (pred_error norm monotone ↓) | Stable throughout | Stable | Soft mixing feeds z_L_mixed into H, preserving PC error signal |

---

## 2. Evidence and Motivation

### 2.1 Root cause of self-disabling bypass (Finding 1)

The bypass substituted `codebook[nearest_idx]` — a single `[l_dim]` vector — tiled to `[seq_len, l_dim]` by `.expand(-1, seq_len, -1)`. For a Sudoku grid with 81 cells, the converged L-state is **not** spatially uniform: each cell holds information specific to its position, its value, and its local constraint neighbourhood. Replacing all 81 positions with the same mean-pooled vector discards precisely the per-position information that differentiates correct from incorrect cell assignments.

The BCE-supervised gate received an honest signal: reconstruction error `‖z_L_converged − nearest_code‖²` was chronically above the `tolerance=0.05` threshold for the vast majority of samples. The target confidence was nearly always 0, so the gate correctly learned `confidence → 0`. The self-disable is not a training instability; it is the expected rational outcome of a gate trained on an accurately-labeled signal.

**The bypass fired at peak ~0.12% of samples and caused a measurable accuracy dip at step 26040.** This is consistent with the gate occasionally over-firing on the rare samples where the pooled approximation happened to be within tolerance — and those samples being genuinely worse for accuracy (the approximation was within MSE tolerance but not within reasoning-quality tolerance).

### 2.2 6-mode collapse is informative, not pathological (Finding 2)

That exactly 6 out of 256 codebook entries are ever assigned across 10 consolidations at 5000-step intervals is a stable empirical finding. The most parsimonious interpretation: **the converged L-state manifold for Sudoku-Extreme, post-PC, has approximately 6 attractor basins.** The remaining 250 entries are dead because there is no data in their Voronoi cells — not because k-means failed to converge.

This has a direct architectural implication: a codebook of K=256 for a 6-mode manifold wastes ~97% of its capacity on uninformed random initialization that never updates. K=8 is sufficient (covers 6 empirical modes with 2 margin entries). This has meaningful consequences for the new design: a smaller, denser codebook with full spatial entries is actually *more* expressive than a large pool of pooled scalars.

### 2.3 Attribution of +2.4pp is unresolved (Finding 3)

Jovial-avocet outperformed poetic-giraffe by 2.41pp (63.48% vs 61.07%) at identical step budget. The control run (`control-no-crystal`, warm-start from same checkpoint, `use_crystallization: false`) will disambiguate between:
- **Hypothesis A:** Fine-tuning a 61% checkpoint for 52k more steps accounts for all the gain (warm-start effect).
- **Hypothesis B:** The BCE auxiliary loss shaped z_L representations toward codebook-compatible attractors, improving the PC error signal.

**The design must be robust to both outcomes.** Phase 3b preserves a reconstruction auxiliary loss even when w_passthrough → 1.0 (model doesn't use codebook), because:
- If Hypothesis A is true: reconstruction loss is zero-cost overhead that does no harm.
- If Hypothesis B is true: reconstruction loss is the mechanism and should be preserved.

If the control run shows Hypothesis A is sufficient, the reconstruction loss weight `lambda_moe_recon` can be set to 0 in ablation without architectural change.

---

## 3. Proposed Architecture

**Decision:** Variant D (Soft MoE over codebook experts), with modifications to address the spatial structure problem. Specifically: codebook values are full spatial tensors `[seq_len, l_dim]`, not pooled vectors; one passthrough expert is included; BCE supervision is removed.

**Rationale over other variants:**
- Variant A (per-head channel-split): splits channels, still tiles each head's code across positions. Does not fix root cause.
- Variant B (soft α mixing, single codebook): fixes the binary gate problem but not the spatial structure. If combined with Variant C it works, but Variant B alone still tiles a pooled vector.
- Variant C (per-position codebook values): fixes spatial structure but doesn't address multi-modal representation or routing dynamics. Variant D subsumes Variant C if the experts each have full spatial entries.
- **Variant D + spatial entries + passthrough expert:** addresses all three findings simultaneously.

### 3.1 Mathematical formulation

**Symbols:**
- B: batch size; S: seq_len; D: l_dim = h_dim = 512; P: proj_dim = 128; key_dim = 2P = 256
- K_modes = 8; K = K_modes + 1 = 9 (8 codebook experts + 1 passthrough expert)

**Learnable parameters (new components):**

```
codebook_values:  R^{K_modes × S × D}      # spatial mode templates
codebook_keys:    R^{K_modes × key_dim}     # matching keys (kept for diagnostics / optional consolidation)
proj_h:           R^{D × P}                 # unchanged from Phase 3a
proj_l:           R^{D × P}                 # unchanged from Phase 3a
router_mlp:       MLP(key_dim → 64 → K)     # replaces confidence_head
```

**Step 1 — Recognition key** (unchanged):
```
h_pool = mean_seq(proj_h(z_H))           # [B, P]
l_pool = mean_seq(proj_l(z_L))           # [B, P]
key    = cat([h_pool, l_pool], dim=-1)   # [B, key_dim]
```

**Step 2 — Routing weights:**
```
logits = router_mlp(key)                 # [B, K]
w      = softmax(logits, dim=-1)         # [B, K], sums to 1
w_cb   = w[:, :K_modes]                 # [B, K_modes] codebook weights
w_pt   = w[:, K_modes]                  # [B]          passthrough weight
```

**Step 3 — L-module recurrence** (ALWAYS runs, same as current):
```
z_L_rec = L_level(z_L, injection, ...)  # [B, S, D]
```

**Step 4 — Codebook mixture:**
```
z_bypass = einsum('bk,ksd->bsd', w_cb, codebook_values)  # [B, S, D]
```

**Step 5 — Soft blend:**
```
z_L_out = w_pt.unsqueeze(-1).unsqueeze(-1) * z_L_rec
        + (1 - w_pt).unsqueeze(-1).unsqueeze(-1) * z_bypass
        = w_pt[:, None, None] * z_L_rec + (1 - w_pt[:, None, None]) * z_bypass
```

**Training loss — reconstruction:**
```
# At 1-step-grad step, z_L_final = z_L_out (with gradient to codebook_values via z_bypass)
# Reconstruction target: what would z_bypass need to look like to match converged z_L?
# We want codebook to track converged states; gradient flows to codebook_values.
# Stop-gradient on z_L_final to prevent encoder collapse (codebook chases current z_L, not vice versa).

L_recon = mean(w_codebook_sum * ||sg(z_L_final) - z_bypass||^2)
```
where `w_codebook_sum = 1 - w_pt` = fraction of output from codebook. This weights the reconstruction loss by how much the model is using the codebook; when w_pt → 1, codebook gets no gradient from reconstruction (consistent — it's not being used).

**Training loss — load balancing (codebook entries only):**
```
w_cb_mean = mean_batch(w_cb)            # [K_modes], mean routing weight per mode
target    = uniform(K_modes) = 1/K_modes
L_lb      = KL(w_cb_mean || target) * sg(1 - mean_batch(w_pt))
```
The `sg(1 - mean_batch(w_pt))` multiplier suppresses the load-balancing loss when the model is primarily using the passthrough expert — we don't want to force codebook diversity when the codebook isn't contributing.

**Total loss addition:**
```
L_crystal = lambda_moe_recon * L_recon + lambda_moe_balance * L_lb
```

### 3.2 Tensor shapes at each step

| Step | Tensor | Shape |
|------|--------|-------|
| Input | z_H | [B, S, D] |
| Input | z_L (carry) | [B, S, D] |
| Recognition key | key | [B, 256] |
| Routing | logits | [B, 9] |
| Routing | w | [B, 9] |
| Codebook weights | w_cb | [B, 8] |
| Passthrough weight | w_pt | [B] |
| Recurrence output | z_L_rec | [B, S, D] |
| Codebook values | codebook_values | [8, 81, 512] |
| Codebook mixture | z_bypass | [B, S, D] |
| Final L state | z_L_out | [B, S, D] |

### 3.3 Answers to open design questions

**1. Partition scheme:** Full-input, no partition. Each of K_modes expert codebooks covers the full `[S, D]` spatial state. Channel-split or position-split would not fix the core problem; they merely re-tile smaller vectors. Orthogonal codebooks are enforced through k-means initialization (modes start at the 6 empirical cluster centroids extended by 2 random init entries), not through an architectural constraint.

**2. Expert structure:** Pure codebook lookup — `codebook_values[k]` is a learned `[S, D]` parameter, no per-expert network. A per-expert network (e.g., small MLP per mode) would add parameters proportional to `K_modes × network_size` and introduce dead-expert risk from more complex gradient dynamics. The empirical ~6-mode structure does not require nonlinear expert computation; a learned template per mode is sufficient.

**3. Routing input:** Joint z_H + z_L (mean-pooled, projected). z_H carries the high-level reasoning context that identifies which problem regime we're in (mode selector); z_L carries the current low-level state that helps identify which attractor we're approaching. Using z_H alone would lose information about where in z_L-space we are; using z_L alone would lose the high-level context.

**4. No-bypass expert:** Yes — the K+1-th expert IS the passthrough (z_L_recurrence). This is cleaner than a separate binary gate for two reasons: (a) no threshold hyperparameter, (b) gradients flow to the router to learn when to prefer recurrence over codebook without BCE supervision. The passthrough expert has no parameters; its "value" is just z_L_rec at runtime.

**5. Supervision:** Reconstruction loss only. BCE supervision is removed entirely. The self-disabling failure in Phase 3a was enabled by BCE: a perfect discriminator can be trained to always output 0 when the codebook is inaccurate. Reconstruction loss does not have this failure mode — it trains the codebook VALUES to become accurate, rather than training a gate to predict that inaccuracy.

### 3.4 Consolidation in the new design

Offline k-means consolidation is retained as an **initialization mechanism only** (first 5000 bootstrap steps), not as the primary update path. During consolidation:
- The existing `CrystallizationBuffer` can still collect `(key, pooled_z_L)` pairs during bootstrap
- First consolidation replaces `codebook_keys` with k-means centroids (K=8, not 256)
- `codebook_values` are initialized from the nearest pooled z_L per mode, then expanded to `[S, D]` via the current segment's per-position means (separate per-position buffer needed; see Implementation Plan)

After first consolidation, backprop is the sole update mechanism for `codebook_values`. EMA is removed; the standard optimizer handles codebook parameter updates. The `CrystallizationBuffer` can be disabled after the first consolidation to avoid the CPU-transfer overhead.

### 3.5 BCE supervision path

Removed entirely. `crystallization_supervision_loss()` will be deprecated (kept for one commit, deleted in the next for clean history). `crystallization_diagnostics()` is also removed (replaced by logging `L_recon` directly).

---

## 4. Interaction with Existing Mechanisms

### 4.1 Interaction with Predictive Coding

Current architecture: `_maybe_crystal_bypass_nograd` fires on non-last H-cycles. When it fires, the PC path reads `mu_L = prediction_net(z_H)`, computes `epsilon = z_L_codebook - mu_L`, and injects `pi × epsilon` into H. This means the codebook z_L participates in the PC error signal.

For Phase 3b: **the same invariant holds.** After `z_L_out = soft_mix(z_bypass, z_L_rec)`, the PC step computes:
```
epsilon = z_L_out - mu_L     # [B, S, D]
pi      = precision_net(z_L_out)
xi      = pi * epsilon        # [B, S, D] — injection into H
```
The `z_L_out` now carries a blend of codebook and recurrence information. When w_pt is high (mostly recurrence), the PC error is identical to Phase 3a. When w_pt is low (mostly codebook), the PC error reflects how well the codebook approximates the prediction target. This is a sensible signal: if the codebook is good, the prediction error should be small.

**No change needed** to PC supervision loss — `epsilon_final`, `pi_final` from the 1-step-grad step flow through `z_L_out`, which includes gradients from both the router and the recurrence path.

**One subtlety:** the PC prediction target `mu_L = prediction_net(z_H)` is trained to predict the recurrence output, not the codebook output. When codebook usage is high, `mu_L` is predicting a mixture of codebook and recurrence — which is a slightly different target than before. This should be self-consistent as both `prediction_net` and `codebook_values` will adapt, but could cause a short adaptation transient at the start of Phase 3b training. Monitor `pred_error_norm` for the first 5000 steps; it may temporarily increase.

### 4.2 Interaction with ACT

Current design: bypass fires only on non-last H-cycles (`is_last_h` check) and only at eval.

For Phase 3b: **soft mixing applies at ALL H-cycles, including the last.** Specifically:
- Every call to `L_level` is followed by `z_L_out = soft_mix(z_bypass, z_L_rec)`
- The `is_last_h` restriction is removed — there is no hard bypass that would prevent gradient flow
- The 1-step-grad invariant is unchanged: only the last (L, H) pair are in the computation graph

The ACT halting logic reads `q_head(z_H[:, 0])`, which does not depend on z_L. Therefore, ACT halting is unaffected by the codebook mixing.

One concern: if the codebook provides a very accurate z_L_out early in an ACT sequence (low h_step), H may converge to a confident halting state earlier. This would manifest as a reduction in mean halt step. This is desirable behavior, not a bug — it is the compute efficiency mechanism in a different form.

### 4.3 Between-segment detach and gradient flow

`InnerCarry` carries `(z_H.detach(), z_L.detach())` between ACT segments. Phase 3b does not change this.

`codebook_values` are `nn.Parameter` — they are not part of the carry. Gradients accumulate to `codebook_values` from every 1-step-grad step across all segments (since segments run sequentially within one outer training step). This is correct: the codebook sees gradient signal from every batch element across all ACT steps.

**Gradients do NOT need to flow across segment boundaries.** The reconstruction loss `‖sg(z_L_final) − z_bypass‖²` is computed within each segment's 1-step-grad pass. The stop-gradient on `z_L_final` prevents the codebook from learning to chase in-graph targets — it learns from the detached converged state. This is safe and consistent with the existing gradient structure.

---

## 5. Failure Modes Analysis

### 5.1 Softmax routing collapse

**Description:** All routing weight concentrates on 1-2 codebook experts, with the remaining K_modes-2 dead. Diagnostic: `entropy(w_cb) < log(2)` (below 1-bit).

**Why this is likely:** Sudoku-Extreme has ~6 natural modes but not all modes appear equally often. Early in training, a few modes dominate the distribution; without encouragement, the router may collapse to those.

**Mitigation:** Load-balancing loss `L_lb` penalizes non-uniform routing over codebook entries. Start with `lambda_moe_balance = 0.01`; if entropy stays below log(3) at step 10K, increase to 0.05.

**Diagnostic metric:** `crystal/routing_entropy` = `mean_batch(entropy(w_cb))`, logged at each eval. Alert if below `log(K_modes / 2) = log(4)` after step 10K.

### 5.2 Passthrough dominance

**Description:** `mean(w_pt) → 1.0`; the model permanently delegates to recurrence and the codebook is never used. This is a valid local minimum (accuracy may be fine) but defeats the purpose.

**Why this happens:** If `L_recon` is large early (codebook is random), the router can minimize loss by routing all weight to passthrough. The reconstruction loss has no effect when w_codebook_sum → 0.

**Mitigation:** Bootstrap phase (first 5000 steps) masks the router gradient and initializes all routing logits to 0 (equal weights = 1/K per expert). After bootstrap, codebook values have been initialized via consolidation and L_recon is meaningful. Additionally, the initialization of `router_mlp` output to near-zero (bias=0, weights small) ensures early routing is approximately uniform.

**Diagnostic metric:** `crystal/mean_passthrough_weight` = `mean_batch(w_pt)`. If this exceeds 0.90 beyond step 20K with `lambda_moe_balance > 0`, investigate.

### 5.3 Codebook reconstruction failure

**Description:** `L_recon` stays high despite training — codebook_values fail to converge to the mode centroids.

**Why this might happen:** If K=8 is too few and the empirical manifold has more than 8 modes, or if the modes are non-stationary during training (z_L distribution shifts as PC improves), reconstruction error stays elevated.

**Mitigation:** (a) K=8 chosen with 2× empirical-mode headroom; should be adequate. (b) Codebook initialization from consolidation at step 5000 gives a warm start on actual mode locations. (c) The EMA consolidation path can be re-enabled as a slow background update if backprop-only proves insufficient — this is a lever we can pull mid-training.

**Diagnostic metric:** `crystal/reconstruction_error` = `mean_batch(‖sg(z_L_final) - z_bypass‖²)`. Target: drops below 0.5 by step 15K (compare to Phase 3a where reconstruction_error was logged but never drove gradient).

### 5.4 Bootstrapping instability from spatial codebook initialization

**Description:** `codebook_values` are initialized randomly. At step 0, `z_bypass = weighted sum of noise`, which would corrupt z_L_out. Even small w_codebook contributions early could destabilize the PC error signal.

**Mitigation:** During bootstrap (steps 0–5000): hard-set `w_pt = 1.0` (router output is masked; passthrough gets all weight). No gradient through router or codebook during bootstrap. After step 5000: consolidation runs, codebook_values get warm-started from actual z_L statistics, then router gradient is unmasked.

This is identical in spirit to the `_crystal_gate_active` flag in Phase 3a, just with a different mechanism.

**Diagnostic metric:** `crystal/bootstrap_active` = bool flag, logged at every eval.

### 5.5 Training/eval distribution shift

**Description:** In Phase 3a, the BCE gate was supervised on training-path z_L (from recurrence after L_level). At eval, the codebook substituted z_L BEFORE L_level ran. This created an inconsistency: the gate was trained on a distribution it would never see at inference.

**Phase 3b status:** This failure mode does NOT apply. Soft mixing uses `z_L_rec` (from L_level), which runs identically at train and eval. The supervision target for `L_recon` is `z_L_final = z_L_out` — the same tensor that will be used at inference. There is no separate "training-path" vs "inference-path" distinction.

The only remaining asymmetry: during bootstrap, router is masked (passthrough=1.0), so the router was not trained on the distribution of z_L_out it will see post-bootstrap. This could cause a transient at step 5000 when the router activates. Monitor `exact_accuracy` at eval steps 5000–15000 for the analogous accuracy dip seen at step 26040 in Phase 3a. If a dip occurs, it should be shallower (soft mixing, not hard substitution) and recover quickly.

**Diagnostic metric:** `exact_accuracy` monotonicity at eval steps 5K–15K. Expect no more than 1pp dip.

---

## 6. Compute / Memory Cost Estimate

**Reference:** jovial-avocet — ~31M params, ~0.14 s/it on A100-40GB, 52k steps in ~4.5 hr.

### Parameter count delta

| Component | Phase 3a | Phase 3b | Delta |
|---|---|---|---|
| proj_h, proj_l | 131,072 | 131,072 | 0 |
| codebook_values | 256 × 512 = 131,072 (pooled) | 8 × 81 × 512 = 331,776 (spatial) | +200,704 |
| codebook_keys | 256 × 256 = 65,536 | 8 × 256 = 2,048 | −63,488 |
| confidence_head / router_mlp | 256×64+64×1 ≈ 16,448 | 256×64+64×9 ≈ 16,960 | +512 |
| **Total crystal module** | ~344,128 | ~481,856 | **+137,728** |
| **% of 31M model** | 1.11% | 1.55% | **+0.44%** |

Net parameter change: +137,728 params ≈ **+0.44%** of the full model. Negligible.

### FLOPs per forward pass delta

The dominant additions per batch step:
1. `einsum('bk,ksd->bsd', w_cb, codebook_values)`: B × K_modes × S × D = 32 × 8 × 81 × 512 ≈ **10.7M FLOPs**
2. Router MLP (256→64→9): B × (256×64 + 64×9) ≈ 32 × **16,960 FLOPs** ≈ 0.54M FLOPs
3. Reconstruction loss: elementwise over [B,S,D] = 32 × 81 × 512 ≈ **1.3M FLOPs**
4. L_module recurrence: always ran in Phase 3a too (bypass was eval-only). No change.

Reference full forward pass: ~8 TransformerBlocks × (attn + SwiGLU) ≈ 8 × (2BST²D + 4BSTD²) with T=81, D=512, B=32 ≈ **~6B FLOPs**. The added operations above total ~12.5M FLOPs = **+0.2%**.

### VRAM delta

- codebook_values bf16: 8 × 81 × 512 × 2 bytes = **663 KB**
- Optimizer states for codebook_values (AdamATan2, 2 moment tensors): ~1.3 MB fp32
- **Total: ~2.0 MB** additional VRAM on a 40GB device. Negligible.

### Throughput impact

The einsum over codebook_values is a B × K_modes batched GEMV — 0.2% additional FLOPs, implemented as a single `torch.einsum` call that cuBLAS can handle efficiently. Expected throughput: **no measurable change from 0.14 s/it.** The CrystallizationBuffer CPU transfers are reduced (only active during bootstrap, then disabled) — Phase 3b may actually be slightly faster at steps > 5000 vs. Phase 3a's ongoing buffer adds.

---

## 7. Implementation Plan

Commits are estimated at ≤200 LOC each. Minimum launch-ready set: Commits 1–5. Commit 6 can follow before the training run begins; Commit 7 is optional.

**Commit 1 — `SpatialMoECodebook` class in `crystallization.py`** *(Hard, ~180 LOC)*
- New class: `SpatialMoECodebook(nn.Module)`
- Parameters: `codebook_values [K_modes, seq_len, l_dim]`, `codebook_keys [K_modes, key_dim]`, `router_mlp`
- `proj_h`, `proj_l` moved here from `RecognitionNetwork` (reused projection logic)
- `forward(z_H, z_L)` → returns `(w [B, K+1], z_bypass [B, S, D], key [B, key_dim])`
- `moe_losses(z_L_final, w, z_bypass)` → returns `(L_recon, L_lb)` — both scalars
- `bootstrap_mask_router(active: bool)` — sets a flag that forces `w_pt = 1.0`
- Keep `CrystallizationBuffer` unchanged for bootstrap consolidation; deprecate `RecognitionNetwork` and both supervision functions

**Commit 2 — `CoralConfig` additions** *(Trivial, ~20 LOC)*
- New fields: `moe_num_modes: int = 8`, `lambda_moe_recon: float = 0.1`, `lambda_moe_balance: float = 0.01`
- Replace `crystal_confidence_threshold` with nothing (removed); remove `lambda_crystal`
- Keep `crystal_bootstrap_steps`, `crystal_consolidation_interval`, `crystal_buffer_capacity` (used for bootstrap consolidation)
- Keep `codebook_size` for `CrystallizationBuffer` consolidation compatibility; the new parameter is `moe_num_modes`

**Commit 3 — `CoralV3Inner` update** *(Moderate, ~120 LOC)*
- Instantiate `SpatialMoECodebook` in place of `RecognitionNetwork` + `confidence_head`
- Add `_apply_moe_mixing(z_H, z_L, z_L_rec)` → `z_L_out` helper — single call site for the soft mix
- Remove `_maybe_crystal_bypass_nograd` (hard bypass gone)
- Remove `is_last_h` check from crystal path — soft mix now applies at every H-cycle
- Update `_compute_crystal_supervision_loss` → `_compute_moe_losses` — returns `(L_recon, L_lb)` instead of `(bce_loss, recon_err, tgt_conf)`
- `_maybe_record_crystal` unchanged (still feeds `CrystallizationBuffer` during bootstrap)
- `crystal_bypass_count` metric renamed `moe_passthrough_weight` (continuous, not integer)

**Commit 4 — `CoralV3LossHead` and `PredMetrics` update** *(Trivial, ~40 LOC)*
- Replace `crystal_supervision_loss_final` field with `moe_recon_loss`, `moe_lb_loss`
- Loss head: `L_crystal = lambda_moe_recon × moe_recon_loss + lambda_moe_balance × moe_lb_loss`
- Update logged W&B metrics: `crystal/recon_loss`, `crystal/lb_loss`, `crystal/mean_passthrough_weight`, `crystal/routing_entropy`, `crystal/codebook_utilization`
- Remove `crystal/bypass_rate` (replaced by `crystal/mean_codebook_weight = 1 - mean_passthrough_weight`)

**Commit 5 — New config `configs/phase3b_moe_codebook.yaml`** *(Trivial, ~50 LOC)*
- Architecture: same as `phase3a_crystal_warmstart.yaml`
- New fields: `moe_num_modes: 8`, `lambda_moe_recon: 0.1`, `lambda_moe_balance: 0.01`
- Warm-start from `phase1_best_checkpoint_61pct.pt` (same checkpoint as Phase 3a — gives stable z_L distribution for bootstrap consolidation)
- Bootstrap: `crystal_bootstrap_steps: 5000` (unchanged)

**Commit 6 — Update `tests/test_crystallization.py`** *(Moderate, ~100 LOC)*
- Remove `RecognitionNetwork` tests; replace with `SpatialMoECodebook` unit tests
- Test: `forward()` output shapes correct; `w` sums to 1; `z_bypass` shape matches `[B,S,D]`
- Test: gradient flows to `codebook_values` from `moe_losses()`
- Test: `bootstrap_mask_router(True)` forces `w_pt ≈ 1.0`
- Test: `moe_losses()` `L_lb` is zero when routing is uniform; non-zero when collapsed

**Commit 7 — Spatial consolidation for `CrystallizationBuffer`** *(Moderate, ~80 LOC — OPTIONAL)*
- `CrystallizationBuffer.add()` extended to also collect per-position z_L (not just pooled mean)
- First consolidation: assigns K_modes cluster centroids using spatial z_L, writes them to `codebook_values`
- This replaces the current "pooled centroid tiled to [S,D]" hack for initialization
- Can be deferred to Phase 3b follow-up; without it, `codebook_values` are initialized to zero and rely on backprop from step 5000 onward (slower warm-start, but correct)

---

## 8. Pre-committed Success Criteria for Phase 3b

**Criterion 1 — `exact_accuracy ≥ 0.650` at the best eval checkpoint**

Justification: jovial-avocet achieved 63.48% final / 63.53% peak. Phase 3b should outperform this if codebook provides any positive contribution beyond warm-start. A clean redesign from the same checkpoint should achieve ≥65% if the mechanism works. If the control run shows that warm-start alone accounts for 63.48%, then ≥65% requires Phase 3b to add ≥1.5pp — still plausible if the soft reconstruction auxiliary helps.

If the control run matches jovial-avocet (≥63%) at 52K steps and Phase 3b matches it too, the mechanism is inert and Phase 3b success should be re-evaluated on bypass_rate criterion instead. This interpretation must be decided before launching Phase 3b.

**Criterion 2 — `mean_codebook_weight ≥ 0.15` at any eval checkpoint after step 15K**

Definition: `mean_codebook_weight = 1 - mean_batch(w_pt)`. This replaces `bypass_rate`. The threshold of 0.15 means at least 15% of the final z_L comes from the codebook mixture on average — well below 50%, so the model has strong recurrence backup. At Phase 3a's peak bypass_rate of ~0.0012, the equivalent `mean_codebook_weight` would have been ~0.001. This criterion requires a 150× improvement.

Justification: If the codebook provides no value and the model correctly learns this, w_pt → 1.0 and `mean_codebook_weight → 0`. Criterion 2 failing at ≥0.15 would indicate the codebook is not learning useful representations — a Phase 3b failure worth diagnosing before Phase 4.

**Criterion 3 — `codebook_utilization ≥ 5/8 = 0.625` at step 10K consolidation**

Definition: fraction of K_modes=8 codebook entries with non-trivial routing weight (mean_batch(w_cb_k) > 0.01). Unlike Phase 3a where utilization was defined by k-means assignment, here utilization is defined by live routing weight.

Justification: K=8 was chosen specifically to match the ~6 empirical modes. All 6+ modes should be active by step 10K once the codebook is initialized. If utilization is below 5/8 (fewer than 5 modes active), routing has collapsed and load-balancing loss needs to be increased.

**Criterion 4 — No single eval checkpoint with `exact_accuracy < 0.600`**

This is a stability criterion. Phase 3a had one non-monotonic point at step 26040 (44.3%), a 2.3pp dip from the previous eval. Phase 3b's soft mixing should not produce such dips because there is no hard substitution event. If any eval checkpoint drops below 60%, it indicates the codebook is actively corrupting z_L — investigate immediately and consider reducing `lambda_moe_recon` or increasing bootstrap_steps.

---

## 9. Risks and Open Questions

**Risk 1 — Control run outcome changes the target**

The control run (`control-no-crystal`, warm-start no crystal) has not yet reported results. If it matches jovial-avocet ≥63%, then Phase 3b's Criterion 1 (≥65%) requires the MoE mechanism to add ≥1.5pp — a non-trivial bar. If the control matches and Phase 3b also matches but doesn't exceed, the researcher needs to decide whether Phase 3b is worth running at all (vs. just investing in Phase 4). **This question should be answered before Phase 3b is launched.** Decision gate: wait for control run to finish; if control ≥63.0%, Criterion 1 becomes ≥65.0%; if control < 61.0%, Criterion 1 becomes ≥64.0%.

**Risk 2 — Spatial codebook is seq_len-coupled**

`codebook_values [K_modes, 81, 512]` is specific to Sudoku's 81-cell sequence. Transfer to other puzzle types (15-puzzle, different grid sizes) requires discarding and re-learning the codebook from scratch, or architectural changes (per-position decoder that takes the mode global vector as input). For Phase 3b on Sudoku this is acceptable. If the architecture is expected to generalize in Phase 5 analysis or Phase 4 N=3 extension, this must be revisited. **The researcher should decide at Phase 4 planning time whether spatial coupling is acceptable or whether a position-agnostic codebook is required.**

**Risk 3 — Reconstruction loss weight `lambda_moe_recon`**

Setting `lambda_moe_recon = 0.1` is carried over from `lambda_crystal = 0.1` in Phase 3a. But the reconstruction loss is computed over `[B, S, D]` (full spatial tensor), while the Phase 3a BCE was a scalar. The effective loss scale is different. An `L_recon` in fp32 units of mean MSE over `[81 × 512]` positions could be O(1)–O(10) depending on initialization; with lambda=0.1, this might dominate task loss early in training.

**Concrete question:** should `lambda_moe_recon` be swept (0.01, 0.05, 0.1) or should we normalize `L_recon` by `S × D`? I recommend computing `L_recon` as the **mean** over `[S, D]` (not sum), so its scale is comparable to a per-token reconstruction error. Launch with `lambda_moe_recon = 0.1`; if reconstruction loss dominates task loss at step 1000 (log ratio > 5×), halve it. This should be monitored closely in the first 2 eval checkpoints.

**Risk 4 — Bootstrap consolidation produces wrong spatial initialisation**

The first consolidation at step 5000 uses `CrystallizationBuffer` which currently stores only pooled (mean over seq) z_L, not full `[S, D]` spatial states. If Commit 7 (spatial consolidation) is deferred, `codebook_values` will not be warm-started from real spatial states — they will remain at random initialization until backprop shapes them. This may delay `mean_codebook_weight` reaching 0.15 by 5000–10000 steps compared to a properly warm-started run.

**Decision required:** should Commit 7 be promoted to the minimum viable set (included before launch), or is the delayed warm-start acceptable? My assessment: Commit 7 is moderately complex (~80 LOC) and the delay in codebook warm-start is a known risk but not a failure mode. If Criterion 2 is met by step 20K regardless, it was not needed. Recommend deferring Commit 7 and monitoring.

**Risk 5 — Phase 2 (columnar routing) interaction is untested**

The forward dispatch in `CoralV3Inner` has four paths: baseline, PC-only, routing-only, PC+routing. Crystallization interacts with all four. Phase 3b modifies `_maybe_crystal_bypass_nograd` and the PCrecord path. The routing-only and PC+routing paths have not been tested with the new soft-mix interface and their `z_L` injection logic differs (they pass `routing_logits_L` from `L_level`). Since Phase 2 (columnar routing) has never been run to convergence (agate-cuckoo and curly-manatee both collapsed), the `pc=F, cr=T` and `pc=T, cr=T` dispatch paths are untested in practice.

The spec does not propose running Phase 3b with columnar routing enabled (Phase 3a also ran `use_columnar_routing: false`). However, Commit 3 must update all four dispatch paths, not just the PC path. **Confirm with the researcher: is the routing-only dispatch path being maintained for future use, or can it be simplified/removed to reduce implementation surface?**
