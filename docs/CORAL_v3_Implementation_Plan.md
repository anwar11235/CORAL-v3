# CORAL v3 — Full Implementation Plan
## Neuroscience-Aligned Architecture Built on HRM
## March 26, 2026

---

## Executive Summary

CORAL v3 is a ground-up rebuild of CORAL on top of HRM's proven architecture, adding three novel mechanisms — precision-weighted predictive coding, neuroscience-aligned crystallization (recognition-gated computation bypass), and sparse columnar routing — unified under a variational free energy objective.

**Target:** Match HRM's ~99% on Sudoku-Extreme and ~99% on Maze-Hard with 15–18M parameters (vs HRM's 27M), while demonstrating interpretable precision dynamics, emergent crystallization, and column specialization.

**Base Architecture:** HRM's Transformer-based H/L modules, 1-step gradient approximation, deep supervision, Q-learning ACT, stablemax, Adam-atan2.

**Novel Additions (our IP):**
1. Precision-weighted prediction errors between levels (Bayesian attention)
2. Recognition-gated crystallization (System 1/System 2 bypass)
3. Sparse columnar routing (Bayesian model selection over sub-modules)

**Hierarchy:** Start N=2, extend to N=3 for crystallization analysis.

---

## Phase 0: HRM Reproduction (Weeks 1–2)

### Goal
Faithful reproduction of HRM's Sudoku-Extreme results as our baseline. This validates the infrastructure before adding any novel mechanisms.

### 0.1 Study HRM's Open-Source Code

HRM code is public at `github.com/sapientinc/HRM`. Tasks:
- Clone and study the codebase thoroughly
- Identify exact architecture details not fully specified in the paper:
  - Transformer block internals (number of attention heads, FFN expansion ratio, etc.)
  - Exact dimensions of H and L modules (the paper says ~27M total)
  - How `x_tilde` (input embedding) is combined with `z_L` and `z_H` (element-wise addition)
  - Deep supervision segment count `N_supervision` used for each benchmark
  - Q-learning hyperparameters (epsilon, Mmax, Mmin distribution)
  - Data augmentation details for Sudoku (band permutations, digit permutations)
  - stablemax implementation details
  - Adam-atan2 implementation or reference
  - Post-Norm architecture specifics (where RMSNorm is placed)
  - LeCun Normal truncated initialization details
  - Learning rate and warmup schedule

### 0.2 Reproduce Training Infrastructure

Build a clean training codebase that matches HRM exactly:

```
coral_v3/
├── config/
│   ├── sudoku_extreme.yaml
│   ├── maze_hard.yaml
│   └── arc_agi.yaml
├── models/
│   ├── hrm_base.py          # Faithful HRM reproduction
│   ├── transformer_block.py  # Shared Transformer block (Llama-style)
│   ├── input_encoder.py      # fI: token embedding
│   └── output_head.py        # fO: softmax/stablemax head
├── training/
│   ├── trainer.py            # Deep supervision + 1-step gradient loop
│   ├── act.py                # Q-learning adaptive computation time
│   ├── optimizer.py          # Adam-atan2 wrapper
│   └── losses.py             # Seq2seq loss + Q-learning loss
├── data/
│   ├── sudoku.py             # Sudoku-Extreme dataset + augmentation
│   ├── maze.py               # Maze-Hard dataset
│   └── arc.py                # ARC-AGI dataset + augmentation
├── utils/
│   ├── logging.py            # W&B integration
│   └── checkpointing.py
└── scripts/
    ├── train.py
    ├── evaluate.py
    └── reproduce_hrm.py      # Exact HRM reproduction script
```

### 0.3 Key Implementation Details for HRM Reproduction

#### 1-Step Gradient (Critical)
This is the single most important piece to get right. The forward pass runs N*T-1 steps under `torch.no_grad()`, then one final L-step and one final H-step with gradients enabled. This gives O(1) memory regardless of N*T depth.

```python
def hrm_forward(zH, zL, x_tilde, N, T, L_net, H_net):
    with torch.no_grad():
        for i in range(N * T - 1):
            zL = L_net(zL + zH + x_tilde)    # element-wise add, then Transformer
            if (i + 1) % T == 0:
                zH = H_net(zH + zL)           # element-wise add, then Transformer
    
    # 1-step grad — only these two ops are in the computation graph
    zL = L_net(zL + zH + x_tilde)
    zH = H_net(zH + zL)
    
    return zH, zL
```

#### Deep Supervision
Multiple forward passes (segments) per training step. Each segment produces a loss. States are detached between segments. This is NOT the same as simply training for more steps — it provides more frequent gradient signal to the H-module.

```python
z = z_init  # fixed truncated normal
for segment in range(M):
    z, y_hat = hrm_forward(*z, x_tilde, N, T, L_net, H_net)
    loss = seq2seq_loss(y_hat, y_true) + q_learning_loss(...)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    z = (z[0].detach(), z[1].detach())  # break computation graph
```

#### Q-Learning ACT
The halting mechanism uses Q-learning, not a simple sigmoid threshold. The Q-head predicts halt/continue values, and a randomized policy decides when to stop. This is more sophisticated than CORAL v2's approach and handles the exploration/exploitation tradeoff.

### 0.4 Reproduction Success Criteria

| Benchmark | HRM Reported | Our Reproduction Target |
|-----------|-------------|------------------------|
| Sudoku-Extreme (1K samples) | 55% | ≥50% |
| Sudoku-Extreme-Full | ~99% | ≥90% |
| Maze-Hard (1K samples) | 74.5% | ≥65% |

Note: Exact reproduction may not be possible due to undisclosed hyperparameters. The goal is to get close enough that we can attribute subsequent improvements to our mechanisms, not to implementation differences. If we can match within ~5-10%, that's sufficient.

### 0.5 Deliverables
- [ ] Working HRM reproduction with training + evaluation scripts
- [ ] W&B dashboard showing training curves matching HRM's reported dynamics
- [ ] Saved baseline checkpoints for comparison
- [ ] Documentation of any deviations from HRM's reported setup

---

## Phase 1: Predictive Coding + Precision-Weighting (Weeks 2–3)

### Goal
Add prediction error computation and precision-weighting between H and L modules. This is the lightest-touch modification — it changes how information flows between modules without altering the modules themselves.

### 1.1 Prediction Error Computation

Currently in HRM, the H-module receives z_L via element-wise addition:
```python
zH_input = zH + zL  # raw state addition
zH = H_net(zH_input)
```

With predictive coding, the H-module generates a prediction of what z_L *should* be, and only the error (surprise) propagates:

```python
# H-module now has a prediction network
mu_L = prediction_net(zH)          # H predicts L's state
epsilon = zL - mu_L                # prediction error
zH_input = zH + epsilon            # only surprise propagates up
zH = H_net(zH_input)
```

Similarly, the L-module receives a top-down signal from H. Currently:
```python
zL_input = zL + zH + x_tilde
```

With predictive coding:
```python
# The prediction mu_L also serves as the top-down signal to L
zL_input = zL + mu_L + x_tilde     # L receives H's prediction, not raw H state
# The error (zL - mu_L) drives H's update, not L's
```

This is a subtle but important change. The L-module receives the *prediction* (what H expects), not H's raw state. The H-module receives the *error* (what H didn't expect), not L's raw state. This focuses the H-module on what's novel/surprising, which is the core principle of predictive coding.

### 1.2 Prediction Network Architecture

The prediction network should be lightweight — it's mapping from H-dim to L-dim:

```python
class PredictionNet(nn.Module):
    def __init__(self, h_dim, l_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(h_dim, l_dim * 2, bias=False),
            nn.GELU(),
            nn.Linear(l_dim * 2, l_dim, bias=False),
        )
        # Initialize near-identity if h_dim == l_dim
    
    def forward(self, zH):
        return self.net(zH)
```

If H and L have the same dimension (as in standard HRM), this is a simple nonlinear projection. If dimensions differ (which they will at N=3), this handles the dimensional transformation naturally.

### 1.3 Precision-Weighting

Add a precision network at the interface between H and L:

```python
class PrecisionNet(nn.Module):
    def __init__(self, dim, eps_min=0.01):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.GELU(),
            nn.Linear(dim, dim, bias=False),
        )
        self.eps_min = eps_min
    
    def forward(self, z):
        return F.softplus(self.net(z)) + self.eps_min
```

The precision-weighted error becomes:
```python
pi = precision_net(zL)                    # per-dimension precision
epsilon = zL - mu_L                        # raw prediction error
xi = pi * epsilon                          # precision-weighted error
zH_input = zH + projection_up(xi)         # weighted error drives H update
```

### 1.4 Modified Free Energy Loss

Add precision-related terms to the training objective:

```python
def predictive_coding_loss(epsilon, pi, lambda_pred=0.1, lambda_pi=0.01):
    # Precision-weighted prediction error (encourages accurate predictions)
    pred_loss = 0.5 * (pi * epsilon ** 2).sum(dim=-1).mean()
    
    # Precision regularizer (prevents collapse/explosion)
    pi_reg = -0.5 * torch.log(pi + 1e-8).sum(dim=-1).mean()
    
    return lambda_pred * pred_loss + lambda_pi * pi_reg
```

Total loss per segment:
```python
L = seq2seq_loss + q_learning_loss + predictive_coding_loss
```

### 1.5 What NOT to Change
- Do NOT change the Transformer blocks themselves
- Do NOT change the 1-step gradient mechanism
- Do NOT change deep supervision or ACT
- Do NOT change optimizer or learning rate

The only changes are: (a) how information flows between H and L, and (b) additional loss terms.

### 1.6 Success Criteria
- [ ] Accuracy within 2% of Phase 0 baseline (precision-weighting shouldn't hurt, may slightly help)
- [ ] Precision values evolving over reasoning cycles (not staying flat)
- [ ] W&B charts showing precision dynamics: early-diffuse, late-focused pattern
- [ ] Prediction error norms decreasing over cycles within each segment (H is learning to predict L)

### 1.7 Deliverables
- [ ] `models/prediction_net.py` — prediction and precision networks
- [ ] Modified `hrm_base.py` → `coral_v3.py` with predictive coding integration
- [ ] Modified `losses.py` with free energy terms
- [ ] W&B logging for: prediction error norms per cycle, precision values per cycle, precision loss

---

## Phase 2: Neuroscience-Aligned Crystallization (Weeks 3–5)

### Goal
Implement recognition-gated computation bypass — the System 1/System 2 mechanism where well-learned patterns skip the expensive L-module recurrence entirely.

### 2.1 Architecture Components

#### Recognition Network
A lightweight network that examines the current state *before* L-module computation and decides whether to bypass it.

```python
class RecognitionNetwork(nn.Module):
    """
    Fast pattern matcher that decides whether to bypass L-module computation.
    Deliberately small — must be much cheaper than the computation it gates.
    """
    def __init__(self, h_dim, l_dim, x_dim, codebook_size=256, proj_dim=128):
        super().__init__()
        # Project inputs to a compact recognition space
        self.proj_h = nn.Linear(h_dim, proj_dim, bias=False)
        self.proj_l = nn.Linear(l_dim, proj_dim, bias=False)
        self.proj_x = nn.Linear(x_dim, proj_dim, bias=False)
        
        # Codebook of known L-module convergence states
        self.codebook = nn.Parameter(torch.randn(codebook_size, l_dim) * 0.01)
        self.codebook_keys = nn.Parameter(torch.randn(codebook_size, proj_dim * 3) * 0.01)
        
        # Confidence gate
        self.confidence_head = nn.Sequential(
            nn.Linear(proj_dim * 3 + 1, 64),  # +1 for distance feature
            nn.GELU(),
            nn.Linear(64, 1),
        )
    
    def forward(self, zH, zL, x_tilde):
        # Project to recognition space
        h_proj = self.proj_h(zH)
        l_proj = self.proj_l(zL)
        x_proj = self.proj_x(x_tilde)  # Note: x_tilde is sequence; may need pooling
        recognition_key = torch.cat([h_proj, l_proj, x_proj], dim=-1)
        
        # Find nearest codebook entry by key similarity
        similarities = F.cosine_similarity(
            recognition_key.unsqueeze(-2),       # [B, seq, 1, proj*3]
            self.codebook_keys.unsqueeze(0),      # [1, 1, K, proj*3]
            dim=-1
        )  # [B, seq, K]
        
        nearest_idx = similarities.argmax(dim=-1)  # [B, seq]
        max_similarity = similarities.max(dim=-1).values  # [B, seq]
        nearest_code = self.codebook[nearest_idx]  # [B, seq, l_dim]
        
        # Confidence = f(recognition_context, match_quality)
        confidence_input = torch.cat([recognition_key, max_similarity.unsqueeze(-1)], dim=-1)
        confidence = torch.sigmoid(self.confidence_head(confidence_input).squeeze(-1))  # [B, seq]
        
        return confidence, nearest_code, nearest_idx
```

**Note on sequence handling:** HRM operates on token sequences (flattened grids). The recognition network operates per-token — each position in the sequence independently decides whether its local computation can be bypassed. This mirrors how the brain can crystallize sub-patterns (e.g., recognizing a solved block in Sudoku) while still deliberating on others.

#### Crystallization Buffer
Stores experiences for offline codebook learning (the "consolidation" process).

```python
class CrystallizationBuffer:
    """
    Stores (context, converged_state) pairs from training for codebook updates.
    Mimics hippocampal replay during consolidation.
    """
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.keys = []      # recognition keys (context)
        self.values = []    # converged L-states
        self.pointer = 0
    
    def add(self, keys, values):
        """Add batch of experiences. keys: [B, proj*3], values: [B, l_dim]"""
        for k, v in zip(keys, values):
            if len(self.keys) < self.capacity:
                self.keys.append(k.detach())
                self.values.append(v.detach())
            else:
                self.keys[self.pointer] = k.detach()
                self.values[self.pointer] = v.detach()
            self.pointer = (self.pointer + 1) % self.capacity
    
    def consolidate(self, recognition_net, num_iterations=100):
        """
        Offline codebook update. Call periodically (e.g., every N epochs).
        Uses k-means-like assignment + EMA update.
        """
        if len(self.keys) < 100:
            return
        
        all_keys = torch.stack(self.keys)
        all_values = torch.stack(self.values)
        
        for _ in range(num_iterations):
            # Assign each experience to nearest codebook entry
            sims = F.cosine_similarity(
                all_keys.unsqueeze(1),
                recognition_net.codebook_keys.unsqueeze(0),
                dim=-1
            )
            assignments = sims.argmax(dim=1)
            
            # Update codebook entries via EMA
            for k in range(recognition_net.codebook.shape[0]):
                mask = assignments == k
                if mask.sum() > 0:
                    mean_value = all_values[mask].mean(dim=0)
                    mean_key = all_keys[mask].mean(dim=0)
                    recognition_net.codebook.data[k] = (
                        0.9 * recognition_net.codebook.data[k] + 0.1 * mean_value
                    )
                    recognition_net.codebook_keys.data[k] = (
                        0.9 * recognition_net.codebook_keys.data[k] + 0.1 * mean_key
                    )
```

### 2.2 Integration with HRM Forward Pass

```python
def coral_v3_forward(zH, zL, x_tilde, N, T, L_net, H_net, 
                      prediction_net, precision_net, recognition_net,
                      crystal_buffer=None, training=True,
                      crystal_threshold=0.8):
    
    crystal_stats = {'bypassed': 0, 'total': 0, 'confidences': []}
    
    with torch.no_grad():
        for i in range(N * T - 1):
            cycle_start = (i % T == 0)
            
            # --- CRYSTALLIZATION CHECK (at start of each L-cycle) ---
            if cycle_start and not training:  # bypass only during inference
                confidence, nearest_code, idx = recognition_net(zH, zL, x_tilde)
                mean_conf = confidence.mean()
                crystal_stats['confidences'].append(mean_conf.item())
                crystal_stats['total'] += 1
                
                if mean_conf > crystal_threshold:
                    # BYPASS: skip this entire L-cycle
                    zL = nearest_code
                    crystal_stats['bypassed'] += 1
                    # Skip remaining T-1 inner steps
                    # (need to handle loop index — see implementation note)
                    continue
            
            # --- STANDARD L-MODULE UPDATE ---
            # Predictive coding: L receives H's prediction
            mu_L = prediction_net(zH)
            zL = L_net(zL + mu_L + x_tilde)
            
            # --- H-MODULE UPDATE (every T steps) ---
            if (i + 1) % T == 0:
                # Precision-weighted error drives H update
                epsilon = zL - mu_L
                pi = precision_net(zL)
                xi = pi * epsilon
                zH = H_net(zH + xi)  # error, not raw state
                
                # --- RECORD FOR CONSOLIDATION (during training) ---
                if training and crystal_buffer is not None:
                    key = recognition_net.compute_key(zH, zL, x_tilde)
                    crystal_buffer.add(key, zL)  # store converged state
    
    # --- 1-STEP GRAD (always runs, never bypassed) ---
    mu_L = prediction_net(zH)
    zL = L_net(zL + mu_L + x_tilde)
    epsilon = zL - mu_L
    pi = precision_net(zL)
    xi = pi * epsilon
    zH = H_net(zH + xi)
    
    return zH, zL, crystal_stats
```

**Implementation Note:** The loop structure above is simplified. In practice, crystallization bypass needs to skip T-1 remaining inner steps when it fires at the start of a cycle. This requires restructuring the loop as nested (outer cycles × inner T steps) rather than flat, which is actually cleaner anyway:

```python
for cycle in range(N):
    # Crystallization check at cycle start
    if can_crystallize(cycle, training):
        conf, code, idx = recognition_net(zH, zL, x_tilde)
        if conf.mean() > threshold:
            zL = code
            zH = H_net(zH + recognition_net.compute_error(zH, code))
            continue  # skip inner loop entirely
    
    # Inner loop: T steps of L-module
    for t in range(T):
        mu_L = prediction_net(zH)
        zL = L_net(zL + mu_L + x_tilde)
    
    # H-module update
    epsilon = zL - mu_L
    pi = precision_net(zL)
    xi = pi * epsilon
    zH = H_net(zH + xi)
```

### 2.3 Confidence Gate Training

The confidence gate learns during training by comparing what full computation produces vs. what the codebook would have produced:

```python
def crystallization_supervision_loss(recognition_net, zH, zL_converged, x_tilde,
                                       tolerance=0.05):
    """
    Train the confidence gate to accurately predict when bypass is safe.
    Called after each full L-cycle (where we have the converged z_L).
    """
    confidence, nearest_code, _ = recognition_net(zH, zL_converged, x_tilde)
    
    # Would crystallization have been accurate?
    reconstruction_error = (zL_converged - nearest_code).pow(2).mean(dim=-1)
    target_confidence = (reconstruction_error < tolerance).float()
    
    # Binary cross-entropy: confidence should predict bypass safety
    loss = F.binary_cross_entropy(confidence, target_confidence)
    
    return loss
```

### 2.4 Consolidation Schedule

Codebook consolidation (the offline update) happens periodically:

```python
# In training loop
if epoch % consolidation_interval == 0:  # e.g., every 10 epochs
    crystal_buffer.consolidate(recognition_net)
    crystal_buffer.clear()  # fresh buffer after consolidation
```

### 2.5 Phased Activation

Crystallization should NOT be active from the start. The model needs to learn to reason before it can learn to recognize what it's already learned.

| Phase | Epochs | Crystallization Status |
|-------|--------|----------------------|
| A (Foundation) | 1–100 | OFF — no recognition network training, no buffer recording |
| B (Learning to Recognize) | 100–300 | RECORDING — buffer collects converged states, confidence gate trains, but bypass never fires (even at inference) |
| C (Active Crystallization) | 300+ | FULL — bypass fires during inference when confidence exceeds threshold |

This mirrors the brain: you can't consolidate knowledge you haven't learned yet.

### 2.6 Success Criteria
- [ ] Accuracy matches Phase 1 during training (crystallization doesn't fire during training)
- [ ] At inference time with crystallization enabled, accuracy is within 1% of non-crystallized inference
- [ ] Crystallization rate > 0% (some cycles are being bypassed)
- [ ] Inference speedup proportional to crystallization rate
- [ ] Codebook usage > 10% (codes are being used, not all dead)
- [ ] Confidence gate is well-calibrated (high confidence → low reconstruction error)

### 2.7 Deliverables
- [ ] `models/crystallization.py` — RecognitionNetwork, CrystallizationBuffer
- [ ] Modified forward pass with crystallization integration
- [ ] Consolidation training loop
- [ ] W&B logging: crystallization rate, confidence distribution, codebook usage, reconstruction error, inference speedup
- [ ] Analysis script: which patterns crystallize first, crystallization rate vs. training epoch

---

## Phase 3: Sparse Columnar Routing (Weeks 4–6)

### Goal
Replace each monolithic Transformer block with S smaller Transformer "columns" and a learned router that selects k active columns per forward pass. This is where parameter reduction happens.

### 3.0 Routing Strategy Decision — Benchmarked on A100

We benchmarked three routing strategies on our A100-SXM4-40GB (March 26, 2026).
Configuration: dim=384, S=8, k=2, B=256, seq=81 (L1-scale Sudoku).

| Strategy | Fwd+Bwd (ms) | Inference (ms) | Peak Memory (MB) |
|----------|-------------|---------------|------------------|
| Monolithic (no routing) | 21.0 | 6.8 | 1,321 |
| A: All columns + soft mask | 88.4 | 29.4 | 4,325 |
| B: Batch reorg | 55.9 | 12.6 | 1,199 |
| **C: Index-select (CHOSEN)** | **33.8** | **9.9** | **1,188** |
| Hybrid (A train, k-only infer) | 136.0 | 16.8 | 5,716 |

**Decision: Use Strategy C (index-select routing) for both training and inference.**

Rationale:
- Only 1.6× slower than monolithic during training — acceptable overhead for 8-column specialization
- Real compute savings during both training AND inference (not just inference)
- Lower memory than monolithic (sub-batches are smaller)
- Strategy A is 4.2× slower — unacceptable
- Hybrid is worst of all worlds (136ms training)
- Strategy C is 2.6× faster than A during training, 3× faster at inference

### 3.1 Columnar Transformer Block (Index-Select Implementation)

```python
class ColumnarTransformerBlock(nn.Module):
    """
    Replaces a single Transformer block with S smaller columns + router.
    Uses index-select routing: flattens top-k assignments, runs each column
    once on its gathered sub-batch, scatters results back with weights.
    
    Benchmarked at 1.6× monolithic overhead on A100 (dim=384, S=8, k=2, B=256).
    """
    def __init__(self, full_dim, n_heads, S=8, k=2, ffn_mult=4):
        super().__init__()
        self.S = S
        self.k = k
        
        # S independent Transformer columns with reduced FFN
        col_ffn = max(1, ffn_mult * 2 // S)
        self.columns = nn.ModuleList([
            TransformerBlock(
                dim=full_dim,
                n_heads=n_heads,
                ffn_mult=col_ffn,
                # ... other HRM Transformer params (RoPE, GeLU, RMSNorm, etc.)
            )
            for _ in range(S)
        ])
        
        # Lightweight router
        self.router = nn.Linear(full_dim, S, bias=False)
        
        # Temperature for routing sharpness (annealed during training)
        self.temperature = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x):
        """
        x: [B, seq_len, dim]
        Returns: (output, logits) where logits are [B, S] for load-balancing loss
        """
        B, seq, D = x.shape
        
        # Compute routing weights from pooled input
        logits = self.router(x.mean(dim=1)) / self.temperature  # [B, S]
        topk_vals, topk_idx = logits.topk(self.k, dim=-1)       # [B, k]
        weights = F.softmax(topk_vals, dim=-1)                    # [B, k]
        
        # Flatten top-k assignments: each sample appears k times
        flat_idx = topk_idx.reshape(-1)        # [B*k] — which column
        flat_weights = weights.reshape(-1)      # [B*k] — what weight
        sample_idx = torch.arange(B, device=x.device) \
                          .unsqueeze(1).expand(B, self.k).reshape(-1)  # [B*k]
        
        # Dispatch to columns and scatter back
        result = torch.zeros_like(x)  # [B, seq, D]
        
        for s in range(self.S):
            col_mask = (flat_idx == s)
            if not col_mask.any():
                continue
            
            entries = col_mask.nonzero(as_tuple=True)[0]
            src_samples = sample_idx[entries]
            src_weights = flat_weights[entries]
            
            sub_batch = x[src_samples]                      # [n, seq, D]
            col_out = self.columns[s](sub_batch)             # [n, seq, D]
            weighted_out = src_weights.unsqueeze(-1).unsqueeze(-1) * col_out
            result.index_add_(0, src_samples, weighted_out)
        
        return result, logits
```

**Why the Python loop over S is acceptable:** Each iteration runs a full Transformer block on a contiguous sub-batch — the GPU kernel launch overhead is negligible compared to the actual matmul compute. The benchmark confirms this: Strategy C (which uses this exact loop) is only 1.6× slower than a single monolithic Transformer, despite routing overhead. The alternative (avoiding the loop entirely via padding + masking) would waste compute on padded positions.

### 3.3 Load-Balancing Loss

```python
def load_balancing_loss(all_logits, S):
    """
    Encourage uniform column usage across the batch.
    all_logits: list of [B, S] tensors from each routing decision.
    """
    # Average routing distribution across batch and all decisions
    avg_probs = torch.stack([F.softmax(l, dim=-1) for l in all_logits]).mean(dim=(0, 1))  # [S]
    
    # KL from uniform
    uniform = torch.ones(S, device=avg_probs.device) / S
    loss = S * (avg_probs * torch.log(avg_probs / uniform + 1e-8)).sum()
    
    return loss
```

### 3.3 Parameter Accounting

The key parameter reduction comes from the FFN layers, which dominate Transformer param count.

**HRM baseline (monolithic):** If HRM uses FFN expansion ratio 4 (standard SwiGLU):
- Per block FFN: 3 × dim × (4 × dim) = 12 × dim² (gate + up + down projections)
- Attention: ~4 × dim² (Q, K, V, O projections)
- Total per block: ~16 × dim²

**CORAL with S=8 columns, col_ffn=1 each (from `ffn_mult * 2 // S = 4*2//8 = 1`):**
- Per column FFN: 3 × dim × dim = 3 × dim²
- Per column attention: ~4 × dim² (same heads, same dim)
- Per column total: ~7 × dim²
- k=2 active columns compute: 14 × dim² (vs 16 × dim² monolithic) — ~12% compute saving per step
- All S=8 columns stored: 56 × dim² total params — 3.5× more params stored
- Router overhead: negligible (dim × S = dim × 8)

**The parameter story for the paper is NOT "fewer total params stored."** It's:
1. **Fewer active params per step** (k/S = 25% of column params active)
2. **Specialization** — each column learns different reasoning sub-strategies
3. **Combined with crystallization** — bypass skips ALL columns, not just some

**Revised parameter targets:**

| Component | Params | Active per step |
|-----------|--------|----------------|
| Input encoder | ~1M | 1M |
| L-module (8 columns) | ~10M | ~2.5M (k=2 of S=8) |
| H-module (8 columns) | ~10M | ~2.5M (k=2 of S=8) |
| Prediction net | ~0.5M | 0.5M |
| Precision net | ~0.3M | 0.3M |
| Recognition net + codebook | ~0.5M | 0.5M |
| Router + halting + output | ~0.5M | 0.5M |
| **Total** | **~23M** | **~7.8M** |

This gives a clean story: "23M total parameters, but only 7.8M active per reasoning step (34% utilization), vs HRM's 27M fully active." The effective model size during inference is 3.5× smaller than HRM.

**Alternative for aggressive compression:** Use S=8, k=2 but with dim/2 column width + projection in/out. This halves total params but adds projection overhead. Benchmark before committing.

### 3.5 Interaction with Crystallization

When crystallization provides a partial recognition (confidence is moderate, not high enough to bypass), the recognized codebook entry can bias routing:

```python
def recognition_informed_routing(self, x, crystal_confidence, crystal_idx):
    """
    Use partial crystallization to bias column selection.
    crystal_confidence: [B] confidence from recognition net
    crystal_idx: [B] nearest codebook index
    """
    # Standard routing logits
    logits = self.router(x.mean(dim=1)) / self.temperature  # [B, S]
    
    # Learned bias: which columns are associated with which codebook entries
    # crystal_routing_bias: [codebook_size, S] learned association matrix
    bias = self.crystal_routing_bias[crystal_idx]  # [B, S]
    
    # Blend: high confidence → more bias, low confidence → pure routing
    alpha = crystal_confidence.unsqueeze(-1)  # [B, 1]
    biased_logits = logits + alpha * bias
    
    return biased_logits
```

This creates the graceful spectrum: no crystallization → pure routing; partial crystallization → informed routing; full crystallization → bypass.

### 3.6 Phased Activation

| Phase | Columns Active | Temperature |
|-------|---------------|-------------|
| A (Foundation) | All S (k=S) | High (soft routing, all columns learn) |
| B (Specialization) | Warm down k: S → target k | Anneal temperature down |
| C (Sparse) | Target k (e.g., 2) | Low (sharp routing) |

### 3.7 Success Criteria
- [ ] Accuracy within 3% of Phase 1 with sparse routing active
- [ ] Router entropy decreasing during Phase B (columns specializing)
- [ ] No column collapse (all columns used at least 5% of the time)
- [ ] Actual inference speedup with top-k-only column execution
- [ ] Parameter count at target (15–18M total, ~7–9M active)

### 3.8 Deliverables
- [ ] `models/columnar_transformer.py` — ColumnarTransformerBlock
- [ ] `models/router.py` — routing logic with load-balancing
- [ ] Modified `coral_v3.py` using columnar blocks
- [ ] W&B logging: per-column activation frequency, router entropy, load-balance loss

---

## Phase 4: N=3 Extension (Weeks 5–7)

### Goal
Extend from N=2 to N=3 levels to demonstrate hierarchical crystallization — different levels crystallizing at different rates, mirroring the expertise development literature.

### 4.1 Three-Level Architecture

```
Level 3 (Strategic, slowest):  updates every T1 × T2 steps
    ↕ prediction error / precision-weighted
Level 2 (Tactical, medium):   updates every T1 steps  
    ↕ prediction error / precision-weighted
Level 1 (Operational, fastest): updates every step
```

With T1=T2=8: Level 1 does 64 steps per complete cycle, Level 2 does 8, Level 3 does 1.

### 4.2 Dimensional Hierarchy

Following the PR (Participation Ratio) insight from HRM's brain correspondence section, higher levels should have higher dimensionality for representational flexibility:

| Level | Dim | Update Frequency | Role |
|-------|-----|-----------------|------|
| L1 (fast) | 384 | Every step | Local constraint checking |
| L2 (medium) | 512 | Every 8 steps | Regional strategy |
| L3 (slow) | 512 | Every 64 steps | Global planning |

**Note:** This is REVERSED from CORAL v2's progressive compression (d1 > d2 > d3). HRM's PR analysis shows the high-level module has HIGHER dimensionality (PR=89.95 for H vs 30.22 for L). The brain uses higher-dimensional representations at higher levels for greater cognitive flexibility. We should follow the neuroscience, not the compression prior.

However, for parameter efficiency with sparse routing, the lower levels (which run more often) should have the cheaper columns. This tension resolves naturally: L1 has lower dim (384) but runs frequently, L3 has higher dim (512) but runs rarely. Total compute is balanced.

### 4.3 Hierarchical Crystallization

Each level gets its own RecognitionNetwork and codebook:

```python
class HierarchicalCrystallization:
    def __init__(self, dims, codebook_sizes):
        self.recognizers = nn.ModuleList([
            RecognitionNetwork(dims[l+1], dims[l], x_dim, codebook_sizes[l])
            for l in range(len(dims) - 1)
        ])
        self.buffers = [CrystallizationBuffer() for _ in range(len(dims) - 1)]
```

The prediction is that crystallization rates will show a hierarchy:
- L1 crystallizes first and most (local patterns are frequent and low-variance)
- L2 crystallizes later and less (tactical patterns are more context-dependent)
- L3 crystallizes last and least (strategic patterns are most variable)

This is the expertise progression: L1 chunks → L2 chunks of chunks → L3 meta-strategies.

### 4.4 Modified Forward Pass for N=3

```python
def forward_n3(z1, z2, z3, x_tilde, T1, T2, nets, training):
    for outer_cycle in range(K):
        # L3 crystallization check
        if can_crystal(3, training):
            # ... check if entire outer cycle can be bypassed
        
        for mid_cycle in range(T2):
            # L2 crystallization check  
            if can_crystal(2, training):
                # ... check if inner T1 steps can be bypassed
            
            for inner_step in range(T1):
                # L1 crystallization check
                if can_crystal(1, training):
                    # ... check if this single step can be bypassed
                
                # L1 update
                mu_1 = predict_net_2to1(z2)
                z1 = L1_net(z1 + mu_1 + x_tilde)
            
            # L2 update
            eps_1 = z1 - mu_1
            pi_1 = precision_net_1(z1)
            xi_1 = pi_1 * eps_1
            mu_2 = predict_net_3to2(z3)
            z2 = L2_net(z2 + xi_1_projected + mu_2)
        
        # L3 update
        eps_2 = z2 - mu_2
        pi_2 = precision_net_2(z2)
        xi_2 = pi_2 * eps_2
        z3 = L3_net(z3 + xi_2_projected)
```

### 4.5 1-Step Gradient for N=3

The 1-step gradient extends naturally:
```python
# All but final step under no_grad (same principle as N=2)
with torch.no_grad():
    # ... all cycles except the very last inner step of the very last cycle

# 1-step grad: final L1 step → final L2 step → final L3 step
z1 = L1_net(z1 + mu_1 + x_tilde)
z2 = L2_net(z2 + xi_1 + mu_2)
z3 = L3_net(z3 + xi_2)
```

Gradient path: Output head → z3 → z2 → z1 → input embedding. Still O(1) memory.

### 4.6 Success Criteria
- [ ] N=3 accuracy ≥ N=2 accuracy on all benchmarks
- [ ] Hierarchical crystallization pattern visible (L1 > L2 > L3 rates)
- [ ] PR analysis showing dimensionality hierarchy (like HRM's brain correspondence)
- [ ] Adaptive computation: easy problems bypass more levels than hard problems

### 4.7 Deliverables
- [ ] `models/coral_v3_n3.py` — three-level architecture
- [ ] Analysis scripts for hierarchical crystallization visualization
- [ ] PR (Participation Ratio) computation comparing N=2 and N=3

---

## Phase 5: Ablations + Paper Figures (Weeks 6–8)

### 5.1 Ablation Matrix

All ablations run at N=2 for clean comparison with HRM:

| Variant | Description | Tests |
|---------|-------------|-------|
| HRM-reproduced | Our faithful HRM reproduction | Baseline |
| CORAL-full (N=2) | All mechanisms active | Full architecture |
| CORAL-no-precision | Remove precision-weighting (π=1) | Bayesian attention contribution |
| CORAL-no-crystal | Disable crystallization | Crystallization value |
| CORAL-no-sparse | All columns active (k=S) | Sparsity contribution |
| CORAL-no-PC | Remove predictive coding (raw state passing) | Predictive coding contribution |
| CORAL-full (N=3) | Three-level hierarchy | Deeper hierarchy value |
| CORAL-N3-no-crystal | N=3 without crystallization | Isolate N=3 vs crystallization |

### 5.2 Paper Figures

**Figure 1: Architecture Diagram**
CORAL v3 architecture showing: H/L Transformer modules with prediction error pathway, recognition network + codebook, sparse columnar router.

**Figure 2: Main Results Table**
Accuracy comparison on Sudoku-Extreme, Maze-Hard, ARC-AGI across all baselines.

**Figure 3: Precision Dynamics**
Per-cycle precision evolution showing early-diffuse → late-focused pattern. Compare with/without precision-weighting.

**Figure 4: Crystallization Emergence**
(a) Crystallization rate over training epochs per level (N=3), showing L1 > L2 > L3 ordering.
(b) Inference speedup vs. crystallization rate.
(c) Accuracy with/without crystallization bypass at inference.

**Figure 5: Column Specialization**
(a) Router entropy over training showing specialization.
(b) Per-column activation frequency heatmap across problem types.
(c) What different columns compute (qualitative analysis if possible).

**Figure 6: Hierarchical Crystallization (N=3 Showcase)**
The "expertise progression" figure showing how different levels develop crystallized patterns at different rates. This is the neuroscience money shot.

**Figure 7: Ablation Results**
Bar chart showing accuracy contribution of each mechanism.

**Figure 8: Adaptive Computation**
(a) Mean halting steps vs problem difficulty
(b) Mean crystallization rate vs problem difficulty (easy → more bypass)

### 5.3 Analysis Scripts
- [ ] `analysis/precision_dynamics.py` — extract and plot per-cycle precision
- [ ] `analysis/crystallization_rates.py` — per-level crystallization over training
- [ ] `analysis/column_specialization.py` — router analysis
- [ ] `analysis/participation_ratio.py` — PR computation for brain correspondence
- [ ] `analysis/generate_paper_figures.py` — all figures from checkpoints

---

## Phase 6: Paper Revision (Week 8+)

### 6.1 Updated Paper Structure

```
1. Introduction (revise to reflect HRM-based approach)
2. Related Work (add MoE, VQ-VAE differentiation)
3. Background
   3.1 Variational Free Energy
   3.2 HRM Architecture (concise summary)
4. The CORAL Architecture
   4.1 Predictive Coding Between Levels (precision-weighted errors)
   4.2 Recognition-Gated Crystallization (System 1/System 2)
   4.3 Sparse Columnar Routing
   4.4 Unified Free Energy Objective
   4.5 N=3 Extension
5. Experiments
   5.1 Setup + Baselines
   5.2 Main Results
   5.3 Ablation Study
   5.4 Analysis
       - Precision Dynamics
       - Crystallization Emergence
       - Hierarchical Crystallization (N=3)
       - Column Specialization
       - Adaptive Computation + Crystallization Interaction
6. Discussion
   6.1 Why the Unified Objective Matters
   6.2 Brain Correspondence
   6.3 Limitations
7. Conclusion
```

### 6.2 Updated Patent Claims

Revise crystallization claims to cover the recognition-gated bypass mechanism:
- **New claim**: A method for adaptive computation bypass in recurrent reasoning comprising: a recognition network that matches current problem state against a learned codebook BEFORE committing to recurrent computation; a confidence gate that evaluates match quality; and bypass of recurrent computation when confidence exceeds a threshold, with the codebook learned through offline consolidation of converged recurrent states.

This is patentably distinct from both VQ-VAE (which compresses mid-stream, not pre-stream) and from early stopping / ACT (which decides when to stop, not whether to start).

---

## Timeline Summary

| Week | Phase | Deliverable |
|------|-------|-------------|
| 1–2 | Phase 0 | HRM reproduction, validated baseline |
| 2–3 | Phase 1 | Predictive coding + precision-weighting integrated |
| 3–5 | Phase 2 | Crystallization mechanism implemented + validated |
| 4–6 | Phase 3 | Sparse columnar routing implemented + validated |
| 5–7 | Phase 4 | N=3 extension with hierarchical crystallization |
| 6–8 | Phase 5 | Full ablation suite + paper figures |
| 8+ | Phase 6 | Paper revision + patent update |

**Note:** Phases 2 and 3 can partially overlap since they modify different parts of the architecture. Phase 4 depends on Phases 1–3 being stable.

**Total estimated time: 8–10 weeks to submission-ready paper.**

---

## Hardware Requirements

- **Primary GPU:** A100-SXM4-40GB (benchmarked) or A100-80GB
  - 1-step gradient means memory is NOT the bottleneck
  - Index-select routing uses LESS memory than monolithic (1,188MB vs 1,321MB at B=256)
  - Larger GPU enables bigger batch sizes for better utilization
- **Routing overhead budget:** 1.6× per Transformer block call (benchmarked)
  - HRM's no_grad loop dominates wall-clock time (N*T-1 steps)
  - The 1-step grad portion (1 L-step + 1 H-step) is negligible
  - Net training time impact: ~1.6× per epoch vs monolithic HRM (acceptable)
- **Estimated training time per run:**
  - Phase 0 (HRM reproduction): ~24–48 hours
  - Phase 1 (predictive coding, no routing change): ~24–48 hours
  - Phase 2 (crystallization, bypass during inference only): ~24–48 hours
  - Phase 3 (sparse routing at 1.6× overhead): ~36–72 hours
  - Phase 4 (N=3): ~48–96 hours (3 levels × more steps per cycle)
  - Phase 5 (ablations, ~8 runs): ~2–3 weeks total
- **Total estimated GPU cost:** ~$600–1200 at Vast.ai A100-40GB rates ($0.48/hr)

---

## Risk Register

| Risk | Impact | Likelihood | Mitigation |
|------|--------|-----------|------------|
| HRM reproduction fails to match reported numbers | High | Medium | Study their code carefully; email authors if stuck |
| Precision-weighting hurts accuracy | Medium | Low | Easy to ablate; keep lambda_pred small |
| Crystallization never activates | Medium | Medium | This time it's recognition-gated, not entropy-gated; lower confidence threshold if needed |
| ~~Sparse routing causes instability~~ | ~~Medium~~ | **De-risked** | Benchmarked on A100; Strategy C confirmed 1.6× overhead, stable. Load-balancing + temp annealing for specialization. |
| N=3 doesn't improve over N=2 | Medium | Medium | N=3 is a bonus, not the main result; paper works with N=2 alone |
| Total params don't meaningfully reduce | Medium | Low | Parameter accounting shows 23M total / 7.8M active — clear story vs HRM's 27M fully active |
| NeurIPS deadline too tight | High | Medium | Could target ICML 2027 or ICLR 2027 instead; theoretical contribution is timeless |
| Index-select routing overhead at N=3 | Medium | Low | Each level's columns run independently; benchmark at dim=384 already covers the most frequently updated level |

---

## Key Architectural Decisions Log

| Decision | Rationale |
|----------|-----------|
| Build on HRM, not from scratch | HRM's proven training recipe (1-step grad, deep supervision, ACT) solves the depth problem that killed CORAL v2 |
| Keep Transformer blocks | Self-attention is critical for relational reasoning on grids; precision-weighting supplements, doesn't replace it |
| Crystallization bypasses BEFORE computation | Brain-aligned; provides real compute savings; patentably distinct from VQ-VAE |
| Codebook learns offline (consolidation) | Brain-aligned; avoids differentiability issues; cleaner training |
| Confidence gate, not entropy gate | Learned gate is more flexible; operates on inputs, not posteriors |
| Higher dim at higher levels (for N=3) | Follows HRM's PR findings and neuroscience; reverses CORAL v2's compression assumption |
| Start N=2, extend N=3 | Clean ablation against HRM at N=2; N=3 for crystallization hierarchy story |
| **Index-select routing (Strategy C)** | **Benchmarked on A100: 1.6× monolithic overhead (33.8ms vs 21.0ms fwd+bwd), lower memory (1,188MB vs 1,321MB). Strategy A (all-columns) was 4.2× slower. Hybrid was worst at 136ms. Strategy C gives real compute savings during BOTH training and inference, not just inference.** |

---

## Appendix A: Routing Benchmark Results (March 26, 2026)

Run on NVIDIA A100-SXM4-40GB. Full benchmark tested four configs; most relevant results below.

### Config: dim=384, n_heads=8, S=8, k=2, B=256, seq=81 (L1-scale Sudoku)

| Strategy | Params | Fwd (ms) | Fwd+Bwd (ms) | Inference (ms) | Peak Mem (MB) |
|----------|--------|----------|-------------|---------------|--------------|
| Monolithic (no routing) | 2,360,064 | 6.81 | 21.01 | 6.81 | 1,320.8 |
| A: All columns + soft mask | 8,266,752 | 29.45 | 88.43 | 29.39 | 4,325.0 |
| B: Batch reorg | 8,266,752 | 14.01 | 55.90 | 12.58 | 1,199.1 |
| **C: Index-select** | **8,266,752** | **10.14** | **33.77** | **9.89** | **1,188.1** |
| Hybrid (A train, k-only infer) | 14,692,352 | 44.42 | 135.98 | 16.84 | 5,715.9 |

### Config: dim=512, n_heads=8, S=8, k=2, B=256, seq=81 (HRM-scale Sudoku)

| Strategy | Params | Fwd (ms) | Fwd+Bwd (ms) | Inference (ms) | Peak Mem (MB) |
|----------|--------|----------|-------------|---------------|--------------|
| Monolithic | 4,198,400 | 7.11 | 22.65 | 7.01 | 1,707.0 |
| A: All columns + soft mask | 14,692,352 | 35.15 | 107.45 | 35.01 | 6,073.6 |
| B: Batch reorg | 14,692,352 | 15.81 | 63.41 | 14.02 | 1,581.1 |
| **C: Index-select** | **14,692,352** | **11.64** | **38.73** | **11.19** | **1,562.6** |

**Key takeaway:** Strategy C scales well. At dim=512 it's 1.7× monolithic — similar ratio to dim=384. The overhead is consistent and acceptable.
