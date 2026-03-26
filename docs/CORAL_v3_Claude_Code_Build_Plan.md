# CORAL v3 — Claude Code Build Plan
## Comprehensive implementation specification for autonomous coding

---

## PROJECT OVERVIEW

CORAL v3 is a neural architecture for complex reasoning tasks. It builds on top of the Hierarchical Reasoning Model (HRM) architecture, adding three novel mechanisms: precision-weighted predictive coding, recognition-gated crystallization, and sparse columnar routing.

The build proceeds in 7 phases. Each phase produces a working, testable system. No phase should break what the previous phase established.

**Reference material available:**
- `HRM/` — The original HRM repo (READ ONLY, do not modify). Use this to verify exact implementation details (tensor shapes, edge cases, initialization, etc.)
- `docs/HRM_Codebase_Analysis.md` — Detailed analysis of HRM's architecture with every critical spec extracted
- `docs/CORAL_v3_Implementation_Plan.md` — Full implementation plan with design rationale

---

## REPO STRUCTURE

```
CORAL-v3/
├── README.md
├── pyproject.toml
├── requirements.txt
├── configs/
│   ├── base.yaml                  # Shared defaults
│   ├── sudoku_extreme_1k.yaml     # Sudoku-Extreme 1K experiment
│   ├── maze_hard_1k.yaml          # Maze-Hard 1K experiment
│   └── arc_agi.yaml               # ARC-AGI experiment
├── coral/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── common.py              # trunc_normal_init_, rms_norm
│   │   ├── layers.py              # CastedLinear, CastedEmbedding, RotaryEmbedding, Attention, SwiGLU
│   │   ├── transformer_block.py   # Post-Norm Transformer block
│   │   ├── reasoning_module.py    # ReasoningModule wrapping N Transformer blocks
│   │   ├── hrm_base.py            # Faithful HRM reproduction (Phase 0 deliverable)
│   │   ├── prediction.py          # PredictionNet, PrecisionNet (Phase 1)
│   │   ├── crystallization.py     # RecognitionNetwork, CrystallizationBuffer (Phase 3)
│   │   ├── columnar.py            # ColumnarTransformerBlock, index-select router (Phase 2)
│   │   ├── coral_v3.py            # Full CORAL v3 model integrating all mechanisms (Phase 4)
│   │   └── sparse_embedding.py    # CastedSparseEmbedding (from HRM, for puzzle embeddings)
│   ├── training/
│   │   ├── __init__.py
│   │   ├── losses.py              # stablemax_cross_entropy, ACT loss, free energy losses
│   │   ├── act.py                 # ACT wrapper with Q-learning halting
│   │   ├── trainer.py             # Training loop with deep supervision
│   │   └── scheduler.py           # Cosine LR with warmup
│   ├── data/
│   │   ├── __init__.py
│   │   └── puzzle_dataset.py      # Dataset loading (compatible with HRM data format)
│   └── analysis/
│       ├── __init__.py
│       ├── precision_dynamics.py
│       ├── crystallization_rates.py
│       ├── column_specialization.py
│       └── participation_ratio.py
├── scripts/
│   ├── train.py                   # Main training entry point
│   ├── evaluate.py                # Evaluation script
│   └── build_dataset.py           # Dataset preparation (calls HRM's build scripts)
├── tests/
│   ├── test_transformer_block.py
│   ├── test_forward_pass.py
│   ├── test_1step_gradient.py
│   ├── test_act.py
│   ├── test_columnar_routing.py
│   └── test_crystallization.py
└── docs/
    ├── HRM_Codebase_Analysis.md
    └── CORAL_v3_Implementation_Plan.md
```

---

## PHASE 0: FAITHFUL HRM REPRODUCTION

### Goal
Implement a clean, independently-written version of HRM that produces identical behavior. Validate by training on Sudoku-Extreme-1K and matching HRM's reported accuracy.

### Step 0.1: Core Utilities (`coral/models/common.py`)

Implement:
- `trunc_normal_init_(tensor, std, lower, upper)` — JAX-compatible truncated normal initialization. NOT PyTorch's nn.init.trunc_normal_ (which is mathematically incorrect). See HRM's `models/common.py` for the exact implementation with compensated std.
- `rms_norm(hidden_states, variance_epsilon)` — RMSNorm as a pure function (no learnable parameters). Cast to float32 for the computation, cast back to input dtype. See HRM's `models/layers.py`.

### Step 0.2: Layer Primitives (`coral/models/layers.py`)

Implement these exactly matching HRM's specs:

**CastedLinear:**
- Weight initialized with `trunc_normal_init_(std=1/sqrt(in_features))` (LeCun normal)
- Optional bias (zero-initialized when present)
- Forward: `F.linear(input, weight.to(input.dtype), bias.to(input.dtype))`

**CastedEmbedding:**
- Weight initialized with `trunc_normal_init_(std=init_std)`
- Forward: `F.embedding(input, weight.to(cast_to_dtype))`

**RotaryEmbedding:**
- Standard RoPE implementation
- Precompute cos/sin caches for max_position_embeddings
- dim = hidden_size // num_heads = 64
- base = 10000.0
- Returns (cos_cached, sin_cached) tuple

**Attention:**
- Fused QKV projection: `CastedLinear(hidden_size, (num_heads + 2*num_kv_heads) * head_dim, bias=False)`
- Output projection: `CastedLinear(output_size, hidden_size, bias=False)`
- Full MHA (num_kv_heads = num_heads = 8), NOT grouped query attention
- Non-causal attention (causal=False)
- RoPE applied to Q and K before attention
- Uses FlashAttention (flash_attn_func) — support both fa2 and fa3
- Head dim = 64

**SwiGLU:**
- intermediate_size = find_multiple(round(expansion * hidden_size * 2/3), 256)
  - For hidden_size=512, expansion=4: round(4*512*2/3) = 1365 → 1536 (next multiple of 256)
- gate_up_proj: `CastedLinear(hidden_size, inter * 2, bias=False)` — fused gate and up projection
- down_proj: `CastedLinear(inter, hidden_size, bias=False)`
- Forward: `down_proj(silu(gate) * up)` where `gate, up = gate_up_proj(x).chunk(2, dim=-1)`

### Step 0.3: Transformer Block (`coral/models/transformer_block.py`)

**Post-Norm architecture** (this is different from the more common Pre-Norm):

```python
def forward(self, cos_sin, hidden_states):
    # Self Attention with Post-Norm
    hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin, hidden_states), eps)
    # FFN with Post-Norm  
    hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), eps)
    return hidden_states
```

Note: The norm wraps the RESIDUAL (x + sublayer(x)), not just the sublayer input.

Config: rms_norm_eps = 1e-5

### Step 0.4: Reasoning Module (`coral/models/reasoning_module.py`)

A simple wrapper that:
1. Adds input_injection to hidden_states (element-wise addition)
2. Passes through a list of Transformer blocks sequentially

```python
def forward(self, hidden_states, input_injection, cos_sin):
    hidden_states = hidden_states + input_injection
    for layer in self.layers:
        hidden_states = layer(cos_sin=cos_sin, hidden_states=hidden_states)
    return hidden_states
```

### Step 0.5: HRM Inner Model (`coral/models/hrm_base.py`)

This is the core model without the ACT wrapper.

**Components:**
- `embed_tokens`: CastedEmbedding(vocab_size, 512, init_std=1/sqrt(512), cast_to=bfloat16)
- `lm_head`: CastedLinear(512, vocab_size, bias=False)
- `q_head`: CastedLinear(512, 2, bias=True) — special init: weight=0, bias=-5
- `puzzle_emb`: CastedSparseEmbedding (if puzzle_emb_ndim > 0)
- `rotary_emb`: RotaryEmbedding(dim=64, max_pos=seq_len+puzzle_emb_len, base=10000)
- `H_level`: ReasoningModule with 4 Transformer blocks
- `L_level`: ReasoningModule with 4 Transformer blocks
- `H_init`: Buffer (not parameter!) — trunc_normal_(shape=[512], std=1)
- `L_init`: Buffer (not parameter!) — trunc_normal_(shape=[512], std=1)

**Input encoding:**
```python
embedding = embed_tokens(input)  # [B, seq_len, 512]
# Prepend puzzle embedding if configured
embedding = cat(puzzle_emb.view(-1, puzzle_emb_len, 512), embedding)
# Scale
return sqrt(512) * embedding
```

**Forward pass (1-step gradient):**
```python
def forward(self, carry, batch):
    input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
    
    with torch.no_grad():
        z_H, z_L = carry.z_H, carry.z_L
        for h_step in range(H_cycles):              # H_cycles = 2
            for l_step in range(L_cycles):           # L_cycles = 2
                if not (h_step == H_cycles-1 and l_step == L_cycles-1):
                    z_L = self.L_level(z_L, z_H + input_embeddings, cos_sin=cos_sin)
            if not (h_step == H_cycles-1):
                z_H = self.H_level(z_H, z_L, cos_sin=cos_sin)
    
    # 1-step grad (ONLY these two ops build the computation graph)
    z_L = self.L_level(z_L, z_H + input_embeddings, cos_sin=cos_sin)
    z_H = self.H_level(z_H, z_L, cos_sin=cos_sin)
    
    # Outputs
    new_carry = InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
    output = self.lm_head(z_H)[:, puzzle_emb_len:]   # Output from H, skip puzzle tokens
    q_logits = self.q_head(z_H[:, 0]).to(float32)     # Q from first token of H
    
    return new_carry, output, (q_logits[..., 0], q_logits[..., 1])
```

**CRITICAL details:**
- L_level input_injection = `z_H + input_embeddings` (H-state plus input encoding)
- H_level input_injection = `z_L` (just L-state, no input)
- Output logits come from `z_H` (H-module), NOT z_L
- Q-values come from first token position: `z_H[:, 0]`
- Carry is detached after forward (no gradient flows between segments)
- forward_dtype is bfloat16

### Step 0.6: ACT Wrapper (`coral/training/act.py`)

Wraps the inner model to handle:
1. Resetting carry for halted sequences (swap in new data from batch)
2. Running the inner model (one segment)
3. Halting decisions via Q-learning
4. Computing target Q-values for the continue action (bootstrapped)

Key behaviors:
- Training: halt when q_halt > q_continue AND steps >= min_steps, OR steps >= halt_max_steps
- Exploration: with prob 0.1, min_steps is sampled from Uniform(2, halt_max_steps+1)
- Evaluation: always run halt_max_steps segments (no early stopping)
- Target Q-continue requires an EXTRA forward pass through the inner model (for bootstrapping)
- Halted sequences get replaced with fresh samples from the batch

See HRM's `hrm_act_v1.py` for exact implementation.

### Step 0.7: Losses (`coral/training/losses.py`)

**stablemax_cross_entropy:**
```python
def s(x, epsilon=1e-30):
    return torch.where(x < 0, 1/(1-x+epsilon), x + 1)

def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x / s_x.sum(dim=dim, keepdim=True))

def stablemax_cross_entropy(logits, labels, ignore_index=-100):
    logprobs = log_stablemax(logits.to(torch.float64))  # NOTE: float64!
    ...
```

**ACT Loss:**
- total_loss = lm_loss + 0.5 * (q_halt_loss + q_continue_loss)
- lm_loss = stablemax_cross_entropy(logits, labels)
- q_halt_loss = BCE_with_logits(q_halt_logits, is_sequence_correct.float())
- q_continue_loss = BCE_with_logits(q_continue_logits, target_q_continue)
- Loss is normalized by global_batch_size: `(1/global_batch_size * loss).backward()`

### Step 0.8: Training Loop (`coral/training/trainer.py` and `scripts/train.py`)

**Deep supervision loop (the outer training loop):**
```python
for set_name, batch, global_batch_size in train_loader:
    batch = {k: v.cuda() for k, v in batch.items()}
    
    if carry is None:
        carry = model.initial_carry(batch)
    
    # One forward pass = one segment of ACT
    carry, loss, metrics, _, _ = model(carry=carry, batch=batch, return_keys=[])
    
    ((1 / global_batch_size) * loss).backward()
    
    # Gradient sync if distributed
    # Apply optimizer with LR schedule
    optimizer.step()
    optimizer.zero_grad()
```

Note: There is NO explicit segment loop in the training code. Each call to `model(carry, batch)` runs one segment. The ACT wrapper handles halting and carry management. Halted sequences get fresh data from the next batch call. The dataloader provides data continuously.

**Optimizer:** AdamATan2 (scale-invariant Adam variant)
- `pip install adam-atan2`
- Separate optimizer for puzzle embeddings: CastedSparseEmbeddingSignSGD

**LR Schedule:** Cosine with linear warmup
```python
def cosine_with_warmup(step, base_lr, warmup_steps, total_steps, min_ratio):
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return base_lr * (min_ratio + (1-min_ratio) * 0.5 * (1 + cos(pi * progress)))
```

**torch.compile:** Enabled by default. Critical for performance.

### Step 0.9: Data Pipeline (`coral/data/puzzle_dataset.py`)

Use HRM's dataset format directly. Their `build_sudoku_dataset.py` produces the data.
The dataset is an IterableDataset that:
- Memory-maps input/label arrays
- Groups examples by puzzle (for augmentation-aware batching)
- Shuffles groups per epoch
- Packs groups into fixed-size global batches
- Handles distributed (multi-GPU) splitting

For Phase 0, we can reuse HRM's dataset build scripts and just implement a compatible loader.

### Step 0.10: Tests

Write tests to validate the reproduction:
- `test_transformer_block.py`: Compare our block output vs HRM's for identical inputs
- `test_forward_pass.py`: Compare full forward pass output shape and gradient flow
- `test_1step_gradient.py`: Verify only the final step has gradients, all prior steps don't
- `test_act.py`: Verify halting logic, carry reset, Q-value computation

### Step 0.11: Validation

Train on Sudoku-Extreme-1K following HRM's README:
```bash
python scripts/train.py data_path=data/sudoku-extreme-1k-aug-1000 epochs=20000 eval_interval=2000 lr=7e-5 weight_decay=1.0
```
Target: ≥50% accuracy (HRM reports 55%)

---

## PHASE 1: PREDICTIVE CODING + PRECISION-WEIGHTING

### Goal
Add prediction error computation and precision-weighting between H and L modules. This is the first novel mechanism.

### What Changes
ONLY the information flow between H and L changes. The Transformer blocks, training recipe, ACT, losses — all stay identical.

### New Files
- `coral/models/prediction.py`

### Modifications
- `coral/models/coral_v3.py` (new file, extends hrm_base.py with predictive coding)
- `coral/training/losses.py` (add prediction error loss, precision regularizer)

### Implementation

**PredictionNet:** Small MLP that maps from H-state to L-state space.
```python
class PredictionNet(nn.Module):
    def __init__(self, h_dim, l_dim):
        self.net = nn.Sequential(
            CastedLinear(h_dim, l_dim * 2, bias=False),
            nn.GELU(),
            CastedLinear(l_dim * 2, l_dim, bias=False),
        )
    def forward(self, z_H):
        return self.net(z_H)
```

**PrecisionNet:** Produces per-dimension precision vector.
```python
class PrecisionNet(nn.Module):
    def __init__(self, dim, eps_min=0.01):
        self.net = nn.Sequential(
            CastedLinear(dim, dim, bias=False),
            nn.GELU(),
            CastedLinear(dim, dim, bias=False),
        )
        self.eps_min = eps_min
    def forward(self, z):
        return F.softplus(self.net(z)) + self.eps_min
```

**Modified forward pass:**
```python
# Where HRM does:
#   z_L = L_level(z_L, z_H + input_embeddings)
#   z_H = H_level(z_H, z_L)
#
# CORAL v3 does:
#   mu_L = prediction_net(z_H)                    # H predicts L's state
#   z_L = L_level(z_L, mu_L + input_embeddings)   # L receives prediction, not raw H
#   epsilon = z_L - mu_L                           # prediction error
#   pi = precision_net(z_L)                        # learned precision
#   xi = pi * epsilon                              # precision-weighted error
#   z_H = H_level(z_H, xi)                         # H receives error, not raw L
```

**Additional losses:**
```python
pred_loss = 0.5 * (pi * epsilon**2).sum(dim=-1).mean()   # weighted prediction error
pi_reg = -0.5 * torch.log(pi + 1e-8).sum(dim=-1).mean()  # precision regularizer
# Added to total loss with small coefficients (lambda_pred=0.1, lambda_pi=0.01)
```

### Validation
- Accuracy within 2% of Phase 0 baseline
- Precision values evolving over cycles (not flat)
- Prediction error norms decreasing within segments

---

## PHASE 2: SPARSE COLUMNAR ROUTING

### Goal
Replace each monolithic Transformer block with S=8 smaller column blocks + index-select router. This is where active parameter reduction happens.

### What Changes
The internal structure of each ReasoningModule changes. Each Transformer block becomes a ColumnarTransformerBlock. Everything else stays the same.

### New Files
- `coral/models/columnar.py`

### Implementation

**ColumnarTransformerBlock** — uses index-select routing (Strategy C, benchmarked at 1.6× overhead on A100):

```python
class ColumnarTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, S=8, k=2, expansion=4, rms_norm_eps=1e-5):
        self.S = S
        self.k = k
        col_ffn_expansion = max(1, expansion * 2 // S)  # reduced FFN per column
        
        self.columns = nn.ModuleList([
            TransformerBlock(dim, n_heads, col_ffn_expansion, rms_norm_eps)
            for _ in range(S)
        ])
        self.router = CastedLinear(dim, S, bias=False)
        self.temperature = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, cos_sin, hidden_states):
        B, seq, D = hidden_states.shape
        logits = self.router(hidden_states.mean(dim=1)) / self.temperature
        topk_vals, topk_idx = logits.topk(self.k, dim=-1)
        weights = F.softmax(topk_vals, dim=-1)
        
        # Index-select dispatch
        flat_idx = topk_idx.reshape(-1)
        flat_weights = weights.reshape(-1)
        sample_idx = torch.arange(B, device=hidden_states.device).unsqueeze(1).expand(B, self.k).reshape(-1)
        
        result = torch.zeros_like(hidden_states)
        for s in range(self.S):
            col_mask = (flat_idx == s)
            if not col_mask.any():
                continue
            entries = col_mask.nonzero(as_tuple=True)[0]
            src_samples = sample_idx[entries]
            src_weights = flat_weights[entries]
            sub_batch = hidden_states[src_samples]
            col_out = self.columns[s](cos_sin=cos_sin, hidden_states=sub_batch)
            result.index_add_(0, src_samples, src_weights.unsqueeze(-1).unsqueeze(-1) * col_out)
        
        return result  # Note: also need to return logits for load-balancing loss
```

**Load-balancing loss:**
```python
def load_balancing_loss(all_logits, S):
    avg_probs = torch.stack([F.softmax(l, dim=-1) for l in all_logits]).mean(dim=(0,1))
    return S * (avg_probs * torch.log(avg_probs * S + 1e-8)).sum()
```

**Phased activation:**
- Phase A: k=S (all columns active, soft routing, temperature high)
- Phase B: Anneal k from S down to target (e.g., 2), lower temperature
- Phase C: k=target, low temperature

### Validation
- Accuracy within 3% of Phase 1
- Router entropy decreasing during Phase B
- No column collapse (all columns ≥5% usage)
- Parameter count: ~23M total, ~7.8M active per step

---

## PHASE 3: RECOGNITION-GATED CRYSTALLIZATION

### Goal
Implement System 1/System 2 bypass — well-learned patterns skip L-module computation entirely via codebook lookup.

### What Changes
Crystallization logic wraps the inner loop. During training, it records converged states and trains the confidence gate. During inference, it gates whether to bypass L-module cycles.

### New Files
- `coral/models/crystallization.py`

### Implementation

**RecognitionNetwork:**
- Small projections of z_H, z_L, x into a compact recognition space
- Codebook of L-module converged states (values) with corresponding keys
- Confidence head: MLP that predicts whether bypass is safe
- Forward: returns (confidence, nearest_code, nearest_idx)

**CrystallizationBuffer:**
- Stores (recognition_key, converged_z_L) pairs from training
- Consolidation method: k-means-like assignment + EMA update of codebook entries
- Called periodically (e.g., every 10 epochs)

**Confidence gate training:**
- After each full L-cycle in training, compare converged z_L with nearest codebook entry
- Target: BCE(confidence, ||z_L_converged - nearest_code||² < tolerance)
- This trains the gate to predict when bypass would be accurate

**Integration with forward pass:**
- Training: always run full L-module, record to buffer, compute crystallization supervision loss
- Inference: check confidence before each L-cycle; if high, substitute codebook entry and skip

**Phased activation:**
- Phase A (early training): crystallization completely OFF
- Phase B (mid training): recording ON, confidence gate trains, bypass never fires
- Phase C (late training + inference): bypass fires during inference when confidence > threshold

### Validation
- Training accuracy identical to Phase 2 (crystallization doesn't fire during training)
- Inference accuracy within 1% when crystallization bypass is enabled
- Crystallization rate > 0% on well-learned patterns
- Codebook usage > 10%

---

## PHASE 4: FULL INTEGRATION + N=3

### Goal
Integrate all mechanisms into a clean model and extend to N=3 hierarchy levels.

### N=3 Architecture
```
Level 3 (Strategic, slowest):  updates every T1 × T2 steps
    ↕ prediction error / precision-weighted
Level 2 (Tactical, medium):   updates every T1 steps
    ↕ prediction error / precision-weighted  
Level 1 (Operational, fastest): updates every step
```

Dimensions: L1=384, L2=512, L3=512 (higher levels have more representational capacity, following HRM's PR findings).

Each level gets: its own ReasoningModule (with columnar routing), its own PredictionNet/PrecisionNet pair, its own RecognitionNetwork + codebook.

1-step gradient extends naturally: no_grad on all but the final step at each level.

### Validation
- N=3 accuracy ≥ N=2 on all benchmarks
- Hierarchical crystallization visible (L1 > L2 > L3 rates)

---

## PHASE 5: ABLATIONS

Run these variants (all at N=2 unless noted):

| Variant | What's Changed |
|---------|---------------|
| HRM-reproduced | Phase 0 baseline |
| CORAL-full (N=2) | All mechanisms |
| CORAL-no-precision | π=1 everywhere |
| CORAL-no-crystal | Crystallization disabled |
| CORAL-no-sparse | k=S (all columns active) |
| CORAL-no-PC | Raw additive injection (no prediction errors) |
| CORAL-full (N=3) | Three-level hierarchy |

---

## PHASE 6: ANALYSIS + FIGURES

Generate from best checkpoints:
1. Precision dynamics over reasoning cycles
2. Crystallization rate per level over training
3. Column specialization (router entropy, activation frequency)
4. Adaptive computation (halting steps vs problem difficulty)
5. Participation Ratio (brain correspondence)
6. Inference speedup from crystallization

---

## CRITICAL IMPLEMENTATION NOTES

### Things That Must Match HRM Exactly
1. Post-Norm RMSNorm (no learnable parameters)
2. SwiGLU with inter_dim rounded to multiple of 256
3. FlashAttention (fa2 or fa3) with RoPE, non-causal
4. 1-step gradient: no_grad on all but final L-step and H-step
5. Carry detached between segments (deep supervision)
6. Q-learning ACT with exploration (prob=0.1)
7. stablemax in float64 for small-sample experiments
8. AdamATan2 optimizer
9. Truncated LeCun Normal init (JAX-compatible, NOT PyTorch's trunc_normal_)
10. Initial states are fixed buffers, not parameters
11. Output from H-module, Q from first token of z_H
12. bfloat16 forward pass with torch.compile

### Things That Are Novel (Our IP)
1. PredictionNet between levels (H predicts L, error drives H)
2. PrecisionNet modulating prediction errors (Bayesian attention)
3. Index-select columnar routing (S=8, k=2, benchmarked at 1.6× overhead)
4. Recognition-gated crystallization (codebook bypass before computation)
5. Offline consolidation (codebook learns from converged states)
6. Confidence gate (learned prediction of bypass safety)
7. Free energy objective wrapping HRM's existing losses
8. N=3 extension with hierarchical crystallization

### Dependencies
```
torch (with CUDA)
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

### Hardware
- A100-40GB or A100-80GB
- torch.compile enabled
- bfloat16 training
- Single GPU sufficient for 1K-sample experiments
