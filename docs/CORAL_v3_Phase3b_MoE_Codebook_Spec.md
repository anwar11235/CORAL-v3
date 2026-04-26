# CORAL v3 Phase 3b — Spatially-Structured Soft MoE Codebook Architecture Spec

**Branch:** `moe-codebook-design`  
**Date:** 2026-04-19 (revised)  
**Status:** Review gate — do not implement until approved

---

## 1. Executive Summary

**Design:** Replace the single-codebook hard-bypass mechanism with a Soft MoE Codebook: K=32 mode experts whose values are full-spatial tensors `[seq_len, l_dim]`, plus one passthrough expert (the recurrence output). A small router MLP computes softmax weights over K+1=33 experts; the final z_L is the resulting convex combination. No binary gate. No BCE supervision. Codebook values are trainable parameters updated end-to-end via an unweighted reconstruction loss that always provides gradient to codebook entries regardless of router weight.

**Problem solved:** The Phase 3a gate correctly identified that tiling a single pooled vector across 81 sequence positions is too lossy to trust — so it learned to suppress itself. The root cause is not the gating mechanism but the loss of spatial structure in codebook values. This design preserves spatial structure by construction. K=32 is set as an exploratory diagnostic to test whether the Phase 3a 6-code finding reflects a genuinely 6-mode manifold or a projection/init artifact.

**Expected outcomes against four criteria:**

| Criterion | Phase 3a result | Phase 3b target | Justification |
|---|---|---|---|
| `exact_accuracy` | 63.48% | ≥ 65.0% | Soft mixing + reconstruction loss ≥ Phase 3a's BCE auxiliary signal |
| `mean_codebook_weight` (bypass analog) | ~0.0001 | ≥ 0.15 | Spatial fidelity makes codebook safe to use; no self-disable incentive |
| `codebook_utilization` (fraction of K modes used) | 6/256 = 2.3% | Observational — see §8 | K=32 chosen to disambiguate manifold structure, not to hit a utilization target |
| PC stability (pred_error norm monotone ↓) | Stable throughout | Stable | Soft mixing feeds z_L_mixed into H, preserving PC error signal |

---

## 2. Evidence and Motivation

### 2.1 Root cause of self-disabling bypass (Finding 1)

The bypass substituted `codebook[nearest_idx]` — a single `[l_dim]` vector — tiled to `[seq_len, l_dim]` by `.expand(-1, seq_len, -1)`. For a Sudoku grid with 81 cells, the converged L-state is **not** spatially uniform: each cell holds information specific to its position, its value, and its local constraint neighbourhood. Replacing all 81 positions with the same mean-pooled vector discards precisely the per-position information that differentiates correct from incorrect cell assignments.

The BCE-supervised gate received an honest signal: reconstruction error `‖z_L_converged − nearest_code‖²` was chronically above the `tolerance=0.05` threshold for the vast majority of samples. The target confidence was nearly always 0, so the gate correctly learned `confidence → 0`. The self-disable is not a training instability; it is the expected rational outcome of a gate trained on an accurately-labeled signal.

**The bypass fired at peak ~0.12% of samples and caused a measurable accuracy dip at step 26040.** This is consistent with the gate occasionally over-firing on the rare samples where the pooled approximation happened to be within tolerance — and those samples being genuinely worse for accuracy (the approximation was within MSE tolerance but not within reasoning-quality tolerance).

### 2.2 6-mode collapse — informative but not conclusive (Finding 2)

That exactly 6 out of 256 codebook entries are ever assigned across 10 consolidations at 5000-step intervals is a stable empirical finding. **Three interpretations are consistent with this result:**

1. **Manifold genuinely has 6 modes.** The converged L-state post-PC has ~6 attractor basins; the other 250 codes sit in empty regions of key-space because there is simply no data there.
2. **K-means init artifact.** The other 250 codes were initialized in regions of key-space that no real data point reaches — they are *unreachable* by the cosine-similarity assignment, not absent-by-virtue-of-no-data. The 6-code result reflects the density of the projection space, not the manifold.
3. **Undertrained projections.** `proj_h` and `proj_l` may have collapsed the key space to ~6 distinguishable regions regardless of actual manifold structure, making any assignment to >6 bins geometrically impossible.

These cannot be distinguished from the Phase 3a data alone. If we set K=8 based on interpretation #1 and interpretations #2 or #3 are correct, Phase 3b would trivially reproduce the 6-code finding — having encoded the assumption into the architecture, then verified it tautologically.

**Phase 3b launches with K_modes=32 to disambiguate.** Codebook utilization becomes a measured diagnostic:
- If utilization saturates at ~6/32 (≈19%), interpretation #1 is likely correct; future runs can reduce K to 8.
- If utilization scales to 20+/32 (>60%), the Phase 3a finding was a projection or init artifact; the manifold is richer than assumed and Phase 4 design should revisit codebook size.

The cost of K=32 over K=8 is negligible (see §6). The information value of the disambiguation is high.

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

**Decision:** Variant D (Soft MoE over codebook experts), with modifications to address the spatial structure problem. Specifically: codebook values are full spatial tensors `[seq_len, l_dim]`, not pooled vectors; one passthrough expert is included; BCE supervision is removed; reconstruction loss is unweighted (always active regardless of router weights — see §3.1).

**Rationale over other variants:**
- Variant A (per-head channel-split): splits channels, still tiles each head's code across positions. Does not fix root cause.
- Variant B (soft α mixing, single codebook): fixes the binary gate problem but not the spatial structure. If combined with Variant C it works, but Variant B alone still tiles a pooled vector.
- Variant C (per-position codebook values): fixes spatial structure but doesn't address multi-modal representation or routing dynamics. Variant D subsumes Variant C if the experts each have full spatial entries.
- **Variant D + spatial entries + passthrough expert:** addresses all three findings simultaneously.

### 3.1 Mathematical formulation

**Symbols:**
- B: batch size; S: seq_len; D: l_dim = h_dim = 512; P: proj_dim = 128; key_dim = 2P = 256
- K_modes = 32; K = K_modes + 1 = 33 (32 codebook experts + 1 passthrough expert)

**Learnable parameters (new components):**

```
codebook_values:  R^{K_modes × S × D}      # spatial mode templates [32, 81, 512]
codebook_keys:    R^{K_modes × key_dim}     # matching keys (kept for consolidation init)
proj_h:           R^{D × P}                 # unchanged from Phase 3a
proj_l:           R^{D × P}                 # unchanged from Phase 3a
router_mlp:       MLP(key_dim → 64 → K)     # replaces confidence_head; K=33
```

**Step 1 — Recognition key** (unchanged):
```
h_pool = mean_seq(proj_h(z_H))           # [B, P]
l_pool = mean_seq(proj_l(z_L))           # [B, P]
key    = cat([h_pool, l_pool], dim=-1)   # [B, key_dim]
```

**Step 2 — Routing weights:**
```
logits = router_mlp(key)                 # [B, K]   K=33
w      = softmax(logits, dim=-1)         # [B, K], sums to 1
w_cb   = w[:, :K_modes]                 # [B, K_modes=32] codebook weights
w_pt   = w[:, K_modes]                  # [B]              passthrough weight
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
z_L_out = w_pt[:, None, None] * z_L_rec + (1 - w_pt[:, None, None]) * z_bypass
```

**Training loss — reconstruction (unweighted):**
```
L_recon = mean_batch(||sg(z_L_final) - z_bypass||^2)   # mean over [B, S, D]
```

The `w_codebook_sum` multiplier from the original draft is **removed**. `z_bypass` always receives reconstruction gradient regardless of the current router weights. This is the anti-passthrough-dominance mechanism (see below for justification). `sg()` denotes stop-gradient: the codebook learns to match the converged recurrence output, not the other way around.

**Training loss — load balancing (codebook entries only):**
```
w_cb_mean = mean_batch(w_cb)            # [K_modes], mean routing weight per mode
target    = uniform(K_modes) = 1/K_modes
L_lb      = KL(w_cb_mean || target)
```

The `sg(1 - mean_batch(w_pt))` suppressor from the original draft is **removed**. Load-balancing pressure on codebook routing is always active, regardless of passthrough weight.

**Total loss addition:**
```
L_crystal = lambda_moe_recon * L_recon + lambda_moe_balance * L_lb
```

**Anti-passthrough-dominance mechanism — why Option B, why not A/C/D:**

The original draft weighted `L_recon` by `(1 - w_pt)` and suppressed `L_lb` by `sg(1 - mean_batch(w_pt))`. Both multipliers vanish when `w_pt → 1`, making `w_pt = 1.0` a gradient-free stable equilibrium — the Phase 3a self-disable failure reproduced in a new form.

**Option B (unweighted reconstruction) was chosen** over the alternatives:
- **Option A (direct passthrough penalty `relu(mean(w_pt) - w_pt_max)²`):** Adds a hyperparameter `w_pt_max` and creates only a penalized, not structurally-prevented equilibrium. A sufficiently large task-loss benefit from passthrough dominance can still overcome a fixed penalty.
- **Option C (softmax clamp, `w_pt ≤ w_pt_max` architecturally):** Restricts router expressiveness — the model can never fully defer to recurrence even on novel puzzles where the codebook is genuinely unhelpful. Wrong inductive bias.
- **Option D (schedule `w_pt_max` from 0.5 → 1.0 over 10k steps):** Introduces a training schedule hyperparameter with a specific time constant; fragile to different training dynamics.

With Option B: `codebook_values` always receive gradient from `L_recon`, so by the time the router might discover passthrough dominance is viable, the codebook is already tracking real mode locations. A well-trained codebook reduces `L_recon`, incentivizing the router to use it (lower task loss via more accurate `z_L_out`). The combined removal of the `L_lb` suppressor means the router also receives constant pressure toward balanced codebook usage. These two changes together make `w_pt = 1.0` unstable, not just penalized.

The "heavy-handed" concern about Option B (codebook trains even when not used) is actually a feature: the codebook needs to be ready *before* the router starts using it. Weighting gradient by usage creates a chicken-and-egg problem.

### 3.2 Tensor shapes at each step

| Step | Tensor | Shape |
|------|--------|-------|
| Input | z_H | [B, S, D] |
| Input | z_L (carry) | [B, S, D] |
| Recognition key | key | [B, 256] |
| Routing | logits | [B, 33] |
| Routing | w | [B, 33] |
| Codebook weights | w_cb | [B, 32] |
| Passthrough weight | w_pt | [B] |
| Recurrence output | z_L_rec | [B, S, D] |
| Codebook values | codebook_values | [32, 81, 512] |
| Codebook mixture | z_bypass | [B, S, D] |
| Final L state | z_L_out | [B, S, D] |

### 3.3 Answers to open design questions

**1. Partition scheme:** Full-input, no partition. Each of K_modes expert codebooks covers the full `[S, D]` spatial state. Channel-split or position-split would not fix the core problem; they merely re-tile smaller vectors. Codebooks are initialized from k-means centroids on actual z_L spatial states at step 5000 (see §3.4 and Commit 3 below); no architectural orthogonality constraint is imposed.

**2. Expert structure:** Pure codebook lookup — `codebook_values[k]` is a learned `[S, D]` parameter, no per-expert network. A per-expert network (e.g., small MLP per mode) would add parameters proportional to `K_modes × network_size` and introduce dead-expert risk from more complex gradient dynamics. A learned template per mode is sufficient.

**3. Routing input:** Joint z_H + z_L (mean-pooled, projected). z_H carries the high-level reasoning context that identifies which problem regime we're in (mode selector); z_L carries the current low-level state that helps identify which attractor we're approaching. Using z_H alone would lose information about where in z_L-space we are; using z_L alone would lose the high-level context.

**4. No-bypass expert:** Yes — the K+1-th expert IS the passthrough (z_L_recurrence). This is cleaner than a separate binary gate for two reasons: (a) no threshold hyperparameter, (b) gradients flow to the router to learn when to prefer recurrence over codebook without BCE supervision. The passthrough expert has no parameters; its "value" is just z_L_rec at runtime.

**5. Supervision:** Reconstruction loss only. BCE supervision is removed entirely. The self-disabling failure in Phase 3a was enabled by BCE: a perfect discriminator can be trained to always output 0 when the codebook is inaccurate. Reconstruction loss does not have this failure mode — it trains the codebook VALUES to become accurate, rather than training a gate to predict that inaccuracy.

### 3.4 Consolidation in the new design

Offline k-means consolidation is retained as an **initialization mechanism only** (first 5000 bootstrap steps), not as the primary update path. The `CrystallizationBuffer` must store full per-position z_L (shape `[B, S, D]`) during bootstrap — not just pooled means — so that the first consolidation can initialize `codebook_values` to real spatial cluster centroids (see Commit 3 in §7).

After first consolidation at step 5000:
- `codebook_values` are set to the K=32 k-means centroids from the spatial buffer
- Backprop is the sole update mechanism for `codebook_values` going forward
- EMA is removed; the standard optimizer handles codebook parameter updates
- `CrystallizationBuffer` is disabled to avoid ongoing CPU-transfer overhead

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

**One subtlety:** the PC prediction target `mu_L = prediction_net(z_H)` is trained to predict the recurrence output, not the codebook output. When codebook usage is high, `mu_L` is predicting a mixture of codebook and recurrence — a shifted target. The "self-consistency" claim (both `prediction_net` and `codebook_values` will adapt) holds in theory but admits multiple stable equilibria, and training dynamics choose between them stochastically.

**Diagnostic: `pred_error_stratified_by_w_pt`.** At every eval, partition the batch into high-passthrough samples (w_pt > 0.9) and high-codebook samples (w_pt < 0.5) and log `pred_error_norm` separately for each group. If the two groups show diverging `pred_error_norm` trajectories — specifically, if high-codebook samples have consistently higher error than high-passthrough samples after step 20K — `prediction_net` is not adapting to cover the codebook regime. This would indicate a degenerate equilibrium where PC optimizes for recurrence-path inputs only. **Hard trigger:** if `pred_error_norm[w_pt<0.5] / pred_error_norm[w_pt>0.9] > 1.5` at any eval from step 20K onward, reduce `lambda_moe_recon` by half at the next checkpoint restart. This ratio threshold means the PC prediction target and the codebook output have diverged enough that `prediction_net` is no longer a shared signal — exactly the degenerate equilibrium described above. Reducing `lambda_moe_recon` lowers codebook influence and gives `prediction_net` room to re-cover the full z_L distribution.

Monitor `pred_error_norm` overall for the first 5000 steps post-bootstrap (steps 5K–10K); it may temporarily increase as the router activates and shifts the target distribution.

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

**Why this is likely:** Sudoku-Extreme may have only ~6 natural modes (per the Phase 3a finding). With K=32, a near-optimal router might legitimately concentrate weight on 6 entries and ignore 26 — which is not "collapse" in the pathological sense but expected behavior. The `L_lb` load-balancing loss will resist this; the tradeoff between L_lb pressure and natural mode structure is controlled by `lambda_moe_balance`.

**Mitigation:** Start with `lambda_moe_balance = 0.01`; if entropy stays below log(6) at step 10K (fewer than 6 modes meaningfully active), increase to 0.05. If codebook_utilization measurement reveals a genuinely 6-mode manifold (see §2.2 disambiguation), it is acceptable to reduce K to 8 in a follow-up run and relax L_lb.

**Diagnostic metric:** `crystal/routing_entropy` = `mean_batch(entropy(w_cb))`, logged at each eval. Alert if below `log(4)` after step 10K.

### 5.2 Passthrough dominance

**Description:** `mean(w_pt) → 1.0`; the model permanently delegates to recurrence and the codebook is never used. This is a valid local minimum (accuracy may be fine) but defeats the purpose.

**Why the original design allowed this:** The original `L_recon` weighted by `(1 - w_pt)` and the `L_lb` suppressed by `sg(1 - mean_batch(w_pt))` both vanished when `w_pt → 1.0`. This made passthrough dominance a gradient-free stable equilibrium.

**Mitigation (structural, not just monitoring):** Both multipliers are removed. `L_recon` is now unweighted — `codebook_values` always receive reconstruction gradient regardless of router weight (Option B). `L_lb` is always active. The resulting gradient landscape makes `w_pt = 1.0` unstable: `codebook_values` continuously learn real mode locations from `L_recon`, and `L_lb` continually pushes routing weight toward the now-accurate codebook entries. Passthrough dominance can no longer be a stable equilibrium when `lambda_moe_balance > 0` and `lambda_moe_recon > 0`.

**Remaining risk:** If `codebook_values` are poorly initialized (random noise at step 5000), the router's first gradient signals on `L_recon` may still incentivize passthrough before the codebook warms up. This is why Commit 3 (spatial consolidation — mandatory, see §7) is in the minimum viable set: the codebook must be initialized from real z_L cluster centroids before the router activates, not random noise.

**Diagnostic metric:** `crystal/mean_passthrough_weight` = `mean_batch(w_pt)`. Monitor trajectory; a healthy run should show `mean_passthrough_weight` declining from ~1.0 at step 5K to ≤0.85 by step 15K. If still above 0.90 at step 20K despite `lambda_moe_balance > 0`, the codebook initialization at step 5K was likely inadequate — inspect consolidation logs and verify Commit 3 ran correctly.

### 5.3 Codebook reconstruction failure

**Description:** `L_recon` stays high despite training — codebook_values fail to converge to the mode centroids.

**Why this might happen:** If K=32 and the empirical manifold actually has >32 modes (unlikely but possible), or if the modes are non-stationary during training (z_L distribution shifts as PC improves), reconstruction error stays elevated.

**Mitigation:** (a) K=32 is well above the 6 empirically observed modes from Phase 3a; should be adequate for most scenarios. (b) Spatial codebook initialization from consolidation at step 5000 gives a real warm start (unlike Phase 3a's random init). (c) The EMA consolidation path can be re-enabled as a slow background update if backprop-only proves insufficient — this is a lever we can pull mid-training.

**Diagnostic metric:** `crystal/reconstruction_error` = `mean_batch(‖sg(z_L_final) - z_bypass‖²)`. Target: drops below 0.5 by step 15K. If it remains above 1.0 at step 20K, investigate whether k-means initialization was effective or whether the L-state distribution has shifted significantly from the bootstrap-phase observations.

### 5.4 Bootstrapping instability from spatial codebook initialization

**Description:** `codebook_values` are initialized randomly. At step 0, `z_bypass = weighted sum of noise`, which would corrupt z_L_out. Even small w_codebook contributions early could destabilize the PC error signal.

**Mitigation:** During bootstrap (steps 0–5000): hard-set `w_pt = 1.0` (router output is masked; passthrough gets all weight). No gradient through router or codebook during bootstrap. After step 5000: consolidation runs from spatial buffer (Commit 3), `codebook_values` get warm-started from actual z_L spatial statistics, then router gradient is unmasked.

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
| codebook_values | 256 × 512 = 131,072 (pooled) | 32 × 81 × 512 = 1,327,104 (spatial) | +1,196,032 |
| codebook_keys | 256 × 256 = 65,536 | 32 × 256 = 8,192 | −57,344 |
| confidence_head / router_mlp | 256×64+64×1 ≈ 16,448 | 256×64+64×33 ≈ 18,560 | +2,112 |
| **Total crystal module** | ~344,128 | ~1,484,928 | **+1,140,800** |
| **% of 31M model** | 1.11% | 4.79% | **+3.68%** |

Net parameter change: +1,140,800 params ≈ **+3.68%** of the full model. Still negligible relative to model size and well within memory budget.

### FLOPs per forward pass delta

The dominant additions per batch step:
1. `einsum('bk,ksd->bsd', w_cb, codebook_values)`: B × K_modes × S × D = 32 × 32 × 81 × 512 ≈ **42.9M FLOPs**
2. Router MLP (256→64→33): B × (256×64 + 64×33) ≈ 32 × **18,560 FLOPs** ≈ 0.59M FLOPs
3. Reconstruction loss: elementwise over [B,S,D] = 32 × 81 × 512 ≈ **1.3M FLOPs**
4. L_module recurrence: always ran in Phase 3a too (bypass was eval-only). No change.

Reference full forward pass: ~8 TransformerBlocks × (attn + SwiGLU) ≈ **~6B FLOPs**. The added operations above total ~44.8M FLOPs = **+0.7%**. Still negligible.

### VRAM delta

- codebook_values bf16: 32 × 81 × 512 × 2 bytes = **2.65 MB**
- Optimizer states for codebook_values (AdamATan2, 2 moment tensors): ~5.3 MB fp32
- Spatial buffer (bootstrap, steps 0–5K): 10000 × 81 × 512 × 4 bytes ≈ **1.65 GB** on CPU (not GPU)
- **GPU VRAM: ~8 MB additional**. Negligible on 40GB device.

### Throughput impact

The einsum over codebook_values scales linearly with K_modes. At K=32 vs K=8 (4×), the einsum FLOPs increase from ~10.7M to ~42.9M — still only 0.7% of total FLOPs. Expected throughput: **no measurable change from 0.14 s/it.** The spatial buffer CPU transfers (only during bootstrap) add ~81×512 floats per sample per step vs. the Phase 3a pooled 512 floats — 81× larger per-sample transfer, but only during the 5000-step bootstrap window.

---

## 7. Implementation Plan

Commits are estimated at ≤200 LOC each. **Minimum launch-ready set: Commits 1–6.** Commit 3 (spatial consolidation) is mandatory and must land before Commit 4 (`CoralV3Inner` update) since the forward pass depends on proper codebook initialization.

**Commit 1 — `CoralConfig` additions** *(Trivial, ~20 LOC)*
- New fields: `moe_num_modes: int = 32`, `lambda_moe_recon: float = 0.1`, `lambda_moe_balance: float = 0.01`
- Replace `crystal_confidence_threshold` with nothing (removed); remove `lambda_crystal`
- Keep `crystal_bootstrap_steps`, `crystal_consolidation_interval`, `crystal_buffer_capacity` (used for bootstrap consolidation)
- Keep `codebook_size` for `CrystallizationBuffer` consolidation compatibility; the new parameter is `moe_num_modes`

**Commit 2 — `SpatialMoECodebook` class in `crystallization.py`** *(Hard, ~180 LOC)*
- New class: `SpatialMoECodebook(nn.Module)`
- Parameters: `codebook_values [K_modes, seq_len, l_dim]`, `codebook_keys [K_modes, key_dim]`, `router_mlp`
- `proj_h`, `proj_l` moved here from `RecognitionNetwork` (reused projection logic)
- `forward(z_H, z_L)` → returns `(w [B, K+1], z_bypass [B, S, D], key [B, key_dim])`
- `moe_losses(z_L_final, w, z_bypass)` → returns `(L_recon, L_lb)` — both scalars; `L_recon` is unweighted
- `bootstrap_mask_router(active: bool)` — sets a flag that forces `w_pt = 1.0`
- Keep `CrystallizationBuffer` for bootstrap; deprecate `RecognitionNetwork` and both supervision functions

**Commit 3 — Spatial consolidation for `CrystallizationBuffer`** *(Moderate, ~130 LOC)*
- `CrystallizationBuffer.add()` extended to collect full per-position z_L (shape `[B, S, D]`), stored as CPU float32 alongside the existing pooled keys
- Buffer VRAM: no GPU memory (CPU-only); CPU memory: capacity × S × D × 4 bytes = 10000 × 81 × 512 × 4 ≈ 1.65 GB CPU. Acceptable.
- First consolidation at step 5000: runs k-means on spatial z_L (K=32 centroids), initializes `codebook_values` directly from spatial centroids
- k-means initialization: Forgy init (K random samples from buffer as starting centroids), 100 iterations of Lloyd's algorithm on CPU using Euclidean distance on spatial z_L for BOTH cluster assignment and centroid update. The forward-pass router is an MLP and does not use cosine similarity on keys — the split metric in the initial draft was a carry-over from Phase 3a's cosine-similarity routing that no longer applies. `codebook_keys` are still initialized from pooled keys via the existing path (unchanged from prior spec).
- Returns utilization count (how many of K centroids are non-empty after assignment)
- Previous pooled-value path kept for `codebook_keys` initialization (unchanged)
- `CrystallizationBuffer.clear()` now also clears the spatial values buffer

**Commit 4 — `CoralV3Inner` update** *(Moderate, ~120 LOC)*
- Instantiate `SpatialMoECodebook` in place of `RecognitionNetwork` + `confidence_head`
- Add `_apply_moe_mixing(z_H, z_L, z_L_rec)` → `z_L_out` helper — single call site for the soft mix
- Remove `_maybe_crystal_bypass_nograd` (hard bypass gone)
- Remove `is_last_h` check from crystal path — soft mix now applies at every H-cycle
- Update `_compute_crystal_supervision_loss` → `_compute_moe_losses` — returns `(L_recon, L_lb)` instead of `(bce_loss, recon_err, tgt_conf)`
- Update `_maybe_record_crystal` to feed spatial z_L to buffer (requires new `add_spatial()` call)
- `crystal_bypass_count` metric renamed `moe_passthrough_weight` (continuous, not integer)
- Stub the three other dispatch paths (baseline, pc=F/cr=T, pc=T/cr=T) with `raise NotImplementedError('columnar routing disabled pending Phase 2 redesign')` — only the PC dispatch path (pc=T, cr=F) is implemented. Rationale: Phase 2 columnar routing never converged (agate-cuckoo and curly-manatee both collapsed); maintaining four paths means writing tests and carrying correctness burden for code paths with no validation signal. Stubs can be replaced if columnar routing is ever revived with its own redesign.

**Commit 5 — `CoralV3LossHead` and `PredMetrics` update** *(Trivial, ~40 LOC)*
- Replace `crystal_supervision_loss_final` field with `moe_recon_loss`, `moe_lb_loss`
- Loss head: `L_crystal = lambda_moe_recon × moe_recon_loss + lambda_moe_balance × moe_lb_loss`
- Update logged W&B metrics: `crystal/recon_loss`, `crystal/lb_loss`, `crystal/mean_passthrough_weight`, `crystal/routing_entropy`, `crystal/codebook_utilization`, `crystal/pred_error_high_pt`, `crystal/pred_error_high_cb`
- Remove `crystal/bypass_rate` (replaced by `crystal/mean_codebook_weight = 1 - mean_passthrough_weight`)

**Commit 6 — New config `configs/phase3b_moe_codebook.yaml`** *(Trivial, ~50 LOC)*
- Architecture: same as `phase3a_crystal_warmstart.yaml`
- New fields: `moe_num_modes: 32`, `lambda_moe_recon: 0.1`, `lambda_moe_balance: 0.01`
- Warm-start from `phase1_best_checkpoint_61pct.pt` (same checkpoint as Phase 3a)
- Bootstrap: `crystal_bootstrap_steps: 5000` (unchanged)

**Commit 7 — Update `tests/test_crystallization.py`** *(Moderate, ~100 LOC — can land before run but not blocking)*
- Remove `RecognitionNetwork` tests; replace with `SpatialMoECodebook` unit tests
- Test: `forward()` output shapes correct; `w` sums to 1; `z_bypass` shape matches `[B, S, D]`
- Test: gradient flows to `codebook_values` from `moe_losses()` regardless of router weights (verifies Option B unweighted reconstruction)
- Test: `bootstrap_mask_router(True)` forces `w_pt ≈ 1.0`
- Test: `L_lb` is near-zero when routing is uniform; non-zero when collapsed to single expert
- Test: spatial consolidation initializes `codebook_values` to non-random values (sanity check on Commit 3)

---

## 8. Pre-committed Success Criteria for Phase 3b

**Criterion 1 — `exact_accuracy ≥ 0.650` at the best eval checkpoint**

Justification: jovial-avocet achieved 63.48% final / 63.53% peak. Phase 3b should outperform this if codebook provides any positive contribution beyond warm-start. A clean redesign from the same checkpoint should achieve ≥65% if the mechanism works. If the control run shows that warm-start alone accounts for 63.48%, then ≥65% requires Phase 3b to add ≥1.5pp — still plausible if the soft reconstruction auxiliary helps.

If the control run matches jovial-avocet (≥63%) at 52K steps and Phase 3b matches it too, the mechanism is inert and Phase 3b success should be re-evaluated on bypass_rate criterion instead. This interpretation must be decided before launching Phase 3b.

**Criterion 2 — `mean_codebook_weight ≥ 0.15` at any eval checkpoint after step 15K**

Definition: `mean_codebook_weight = 1 - mean_batch(w_pt)`. This replaces `bypass_rate`. The threshold of 0.15 is anchored as follows: at K=32, the uniform prior routing floor is 1/(K+1) = 1/33 ≈ 0.030. A `mean_codebook_weight` of 0.15 is 5× the uniform floor — well above a router that has learned nothing and is routing diffusely across all 33 experts. Below 5× uniform, codebook usage is consistent with noise; above it, the router has learned something about when to use the codebook.

Justification: If the codebook provides no value and the model correctly learns this, w_pt → 1.0 and `mean_codebook_weight → 0`. Criterion 2 failing at ≥0.15 would indicate the codebook is not learning useful representations or the router is not being incentivized to use it — both are Phase 3b failures worth diagnosing before Phase 4.

**Criterion 3 — `codebook_utilization`: observational, not pass/fail**

Definition: fraction of K_modes=32 codebook entries with non-trivial routing weight (mean_batch(w_cb_k) > 0.01) at any eval after step 15K.

This criterion is observational because the goal of K=32 is disambiguation, not hitting a predetermined utilization number. The two informative outcomes are:
- **Utilization saturates near ~6/32 (≈19%):** interpretation #1 from §2.2 is likely correct — the manifold has ~6 modes. Future runs should reduce K to 8 for efficiency.
- **Utilization scales to 20+/32 (>60%):** the Phase 3a 6-code finding was an artifact of projection collapse or k-means init. The manifold is richer than assumed. Phase 4 design should consider a larger or more structured codebook.

Either outcome is informative. The run fails on this dimension only if utilization cannot be measured (e.g., routing entropy is zero — all weight on passthrough), which is covered by Criterion 2.

**Criterion 4 — No single eval checkpoint with `exact_accuracy < 0.600`**

This is a stability criterion. Phase 3a had one non-monotonic point at step 26040 (44.3%), a 2.3pp dip from the previous eval. Phase 3b's soft mixing should not produce such dips because there is no hard substitution event. If any eval checkpoint drops below 60%, it indicates the codebook is actively corrupting z_L — investigate immediately and consider reducing `lambda_moe_recon` or increasing bootstrap_steps.

---

## 9. Risks and Open Questions

**Risk 1 — Control run outcome changes the target**

The control run (`control-no-crystal`, warm-start no crystal) has not yet reported results. If it matches jovial-avocet ≥63%, then Phase 3b's Criterion 1 (≥65%) requires the MoE mechanism to add ≥1.5pp — a non-trivial bar. If the control matches and Phase 3b also matches but doesn't exceed, the researcher needs to decide whether Phase 3b is worth running at all (vs. just investing in Phase 4). **This question should be answered before Phase 3b is launched.** Decision gate: wait for control run to finish; if control ≥63.0%, Criterion 1 becomes ≥65.0%; if control < 61.0%, Criterion 1 becomes ≥64.0%.

**Risk 2 — Spatial codebook is seq_len-coupled**

`codebook_values [K_modes, 81, 512]` is specific to Sudoku's 81-cell sequence. Transfer to other puzzle types (15-puzzle, different grid sizes) requires discarding and re-learning the codebook from scratch, or architectural changes (per-position decoder that takes the mode global vector as input). For Phase 3b on Sudoku this is acceptable. If the architecture is expected to generalize in Phase 5 analysis or Phase 4 N=3 extension, this must be revisited. **The researcher should decide at Phase 4 planning time whether spatial coupling is acceptable or whether a position-agnostic codebook is required.**

**Risk 3 — Reconstruction loss weight `lambda_moe_recon`**

Setting `lambda_moe_recon = 0.1` is carried over from `lambda_crystal = 0.1` in Phase 3a. But the reconstruction loss is now computed over the full spatial tensor `[B, S, D]` and is unweighted. `L_recon` as mean MSE over `[81 × 512]` positions early in training (before codebook warm-start) could be O(1)–O(10); with lambda=0.1, this might dominate task loss before the codebook converges.

**Concrete question:** I recommend computing `L_recon` as the **mean** over `[S, D]` (not sum), so its scale is comparable to a per-token reconstruction error. Launch with `lambda_moe_recon = 0.1`. HARD RULE: if L_recon > 10× task loss at the step 5K eval (first post-bootstrap checkpoint), kill the run, halve `lambda_moe_recon` to 0.05, relaunch. Do NOT attempt to ride it out — 5000 steps of gradient from a scale-mismatched auxiliary loss will have pushed the model into a region later evals cannot cleanly recover from. Cost of a $2 relaunch is less than the cost of misinterpreting a corrupted run. Monitor the `crystal/recon_loss` vs `train/task_loss` ratio at the first two eval checkpoints regardless.

**Risk 4 — Spatial consolidation correctness (resolved structurally)**

This risk was listed in the original spec as a concern about deferring Commit 7. It is now resolved: Commit 3 (spatial consolidation) is part of the minimum viable launch set and must land before the training run. The `CrystallizationBuffer` will collect full `[B, S, D]` spatial z_L during bootstrap, and the first consolidation at step 5000 will initialize `codebook_values` from real k-means cluster centroids. There is no deferred version.

The remaining implementation risk is correctness: verify that the k-means assignment uses spatial z_L (not pooled keys) for centroid computation, and that `codebook_values` are correctly written to the SpatialMoECodebook parameter after consolidation. Commit 7 (test suite) includes a regression test for this.

**Risk 5 — Phase 2 (columnar routing) interaction is untested**

The forward dispatch in `CoralV3Inner` has four paths: baseline, PC-only, routing-only, PC+routing. Crystallization interacts with all four. Phase 3b modifies `_maybe_crystal_bypass_nograd` and the PCrecord path. The routing-only and PC+routing paths have not been tested with the new soft-mix interface and their `z_L` injection logic differs (they pass `routing_logits_L` from `L_level`). Since Phase 2 (columnar routing) has never been run to convergence (agate-cuckoo and curly-manatee both collapsed), the `pc=F, cr=T` and `pc=T, cr=T` dispatch paths are untested in practice.

The spec does not propose running Phase 3b with columnar routing enabled (Phase 3a also ran `use_columnar_routing: false`). **RESOLVED (2026-04-20 review):** columnar routing dispatch paths will be stubbed with `NotImplementedError` in Commit 4. Phase 2 is not being revived in Phase 3b. Future columnar routing work will require its own redesign, at which point the stubs can be replaced.
