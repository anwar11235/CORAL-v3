# HRM Codebase Analysis — Exact Architecture Details for CORAL v3
## Extracted from github.com/sapientinc/HRM source code
## March 26, 2026

---

## 1. Architecture Dimensions (from hrm_v1.yaml)

```yaml
hidden_size: 512
num_heads: 8          # → head_dim = 512/8 = 64
expansion: 4          # SwiGLU FFN expansion
H_cycles: 2           # N in the paper
L_cycles: 2           # T in the paper  
H_layers: 4           # Transformer blocks in H-module
L_layers: 4           # Transformer blocks in L-module
halt_max_steps: 16    # Mmax for ACT (deep supervision segments)
halt_exploration_prob: 0.1
pos_encodings: rope
puzzle_emb_ndim: 512  # = hidden_size
```

**Critical finding: H_cycles=2, L_cycles=2, NOT the paper's "N cycles of T steps."**
This means the effective depth per segment is: H_cycles × L_cycles = 4 steps total.
With halt_max_steps=16 segments, total depth = 16 × 4 = 64 steps.
The paper's Figure 3 shows ~64 step indices, confirming this.

**NOT** H_cycles=8 × L_cycles=8 = 64 as we assumed from the paper's defaults!

---

## 2. Transformer Block (Post-Norm)

```python
# From hrm_act_v1.py — HierarchicalReasoningModel_ACTV1Block
def forward(self, cos_sin, hidden_states):
    # Post-Norm (NOT pre-norm!)
    hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), eps)
    hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), eps)
    return hidden_states
```

Post-norm means: `x = RMSNorm(x + Attn(x))` then `x = RMSNorm(x + FFN(x))`
This is different from pre-norm (which is `x = x + Attn(Norm(x))`).

**No learnable scale/bias in RMSNorm** — it's a pure function, not a module:
```python
def rms_norm(hidden_states, variance_epsilon):
    variance = hidden_states.float().square().mean(-1, keepdim=True)
    return (hidden_states.float() * torch.rsqrt(variance + variance_epsilon)).to(input_dtype)
```

---

## 3. Attention

```python
# Fused QKV projection (no bias)
self.qkv_proj = CastedLinear(hidden_size, (num_heads + 2*num_kv_heads) * head_dim, bias=False)
self.o_proj = CastedLinear(output_size, hidden_size, bias=False)

# Uses FlashAttention (fa2 or fa3)
# Non-causal attention (causal=False in config)
# RoPE applied to Q and K
```

Key: full MHA (not GQA) — `num_key_value_heads = num_heads = 8`

---

## 4. SwiGLU FFN

```python
class SwiGLU:
    def __init__(self, hidden_size, expansion):
        inter = find_multiple(round(expansion * hidden_size * 2/3), 256)
        # expansion=4, hidden_size=512: inter = round(4*512*2/3) = 1365 → rounded up to 1536 (multiple of 256)
        
        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)  # 512 → 3072
        self.down_proj = CastedLinear(inter, hidden_size, bias=False)         # 1536 → 512

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)
```

SwiGLU inter dim = 1536 (for hidden_size=512, expansion=4).
Total FFN params per block: 512×3072 + 1536×512 = 1,572,864 + 786,432 = 2,359,296
Total attention params per block: 512×(512+2×512) = 512×1536 (qkv) + 512×512 (o) = 786,432 + 262,144 = 1,048,576
Total per block: ~3.4M

---

## 5. Weight Initialization — Truncated LeCun Normal

```python
# NOT PyTorch's trunc_normal_ — custom JAX-compatible implementation
def trunc_normal_init_(tensor, std=1.0, lower=-2.0, upper=2.0):
    # Mathematically correct truncated normal (JAX-style)
    # Compensates std for truncation
    ...
```

**Linear layers:** std = 1/sqrt(in_features) — this is LeCun normal
**Embedding:** std = 1/sqrt(hidden_size)
**Q-head:** zero init weights, bias=-5 (starts near zero for Q-values)

---

## 6. Initial States

```python
# Fixed truncated normal, NOT learned parameters
self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(hidden_size), std=1), persistent=True)
self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(hidden_size), std=1), persistent=True)
```

Shape is [hidden_size], broadcast to [batch, seq_len, hidden_size].
These are BUFFERS (not parameters) — they don't receive gradients.
They're initialized once and stay fixed throughout training.

---

## 7. Input Encoding

```python
def _input_embeddings(self, input, puzzle_identifiers):
    # Token embedding
    embedding = self.embed_tokens(input)  # [B, seq_len, hidden_size]
    
    # Puzzle embedding (optional, per-puzzle learnable embedding)
    if puzzle_emb_ndim > 0:
        puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
        # Reshape and prepend to sequence
        embedding = cat(puzzle_embedding.view(-1, puzzle_emb_len, hidden_size), embedding)
    
    # Position encoding (RoPE applied in attention, not here)
    # Or learned position embedding added here
    
    # Scale by sqrt(hidden_size)
    return sqrt(hidden_size) * embedding
```

The puzzle embedding is a per-puzzle-ID learnable vector prepended to the sequence.
This is how HRM handles few-shot learning for ARC — each puzzle gets its own embedding.
For Sudoku, puzzle_identifiers map different augmentations of the same puzzle to the same ID.

---

## 8. The Forward Pass — EXACT Implementation

```python
def forward(self, carry, batch):
    input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
    
    with torch.no_grad():
        z_H, z_L = carry.z_H, carry.z_L
        
        for _H_step in range(H_cycles):          # H_cycles = 2
            for _L_step in range(L_cycles):        # L_cycles = 2
                # Skip the very last L-step (saved for 1-step grad)
                if not ((_H_step == H_cycles-1) and (_L_step == L_cycles-1)):
                    z_L = L_level(z_L, z_H + input_embeddings)
            
            # Skip the very last H-step (saved for 1-step grad)  
            if not (_H_step == H_cycles-1):
                z_H = H_level(z_H, z_L)
    
    # 1-step grad — ONLY these two ops are in the computation graph
    z_L = L_level(z_L, z_H + input_embeddings)
    z_H = H_level(z_H, z_L)
    
    # Output from H-module
    output = lm_head(z_H)
    
    # Q-values for ACT (from first token of z_H)
    q_logits = q_head(z_H[:, 0])
    
    return new_carry, output, (q_halt, q_continue)
```

**CRITICAL DETAILS:**

1. **L_level input injection = z_H + input_embeddings** (element-wise add of H-state and input)
2. **H_level input injection = z_L** (just the L-state, no input added)
3. The reasoning module does: `hidden_states = hidden_states + input_injection` then passes through Transformer layers
4. So effectively: L gets z_L + (z_H + input_emb), H gets z_H + z_L
5. **Output comes from z_H**, not z_L (via lm_head on H-module output)
6. **Q-values come from z_H[:, 0]** — first token position only
7. **No prediction errors, no precision weighting** — pure additive injection

---

## 9. ACT (Adaptive Computation Time) — Deep Supervision Loop

```python
# From the ACT wrapper's forward():
def forward(self, carry, batch):
    # Reset carry for halted sequences (swap in new data)
    new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
    new_current_data = {k: where(carry.halted, batch[k], v) ...}
    
    # Forward inner model (one segment = H_cycles × L_cycles steps)
    new_inner_carry, logits, (q_halt, q_continue) = self.inner(new_inner_carry, new_current_data)
    
    # Halting logic
    steps += 1
    halted = (steps >= halt_max_steps) | (q_halt > q_continue)
    
    # Exploration: with prob 0.1, force minimum steps
    min_halt_steps = random * randint(2, halt_max_steps+1)
    halted = halted & (steps >= min_halt_steps)
    
    # Target Q for continue action (bootstrapped)
    if training:
        next_q_halt, next_q_continue = self.inner(new_inner_carry, new_current_data)[-1]
        target_q_continue = sigmoid(where(is_last_step, next_q_halt, max(next_q_halt, next_q_continue)))
```

**Key insight:** Each "segment" is ONE call to self.inner(), which runs H_cycles×L_cycles=4 recurrent steps.
halt_max_steps=16 means up to 16 segments = 64 total recurrent steps.

During training, halted sequences get REPLACED with fresh samples from the batch.
This is how they handle variable computation — different samples in the batch may be on different segments.

During eval, always runs halt_max_steps=16 segments (no early stopping).

---

## 10. Training Loop Details

```python
# From pretrain.py:

# Optimizer: AdamATan2 (scale-invariant Adam variant)
# from adam_atan2 import AdamATan2
optimizer = AdamATan2(
    model.parameters(),
    lr=0,  # Set by scheduler
    weight_decay=config.weight_decay,
    betas=(config.beta1, config.beta2)
)

# LR Schedule: cosine with warmup
# lr ramps linearly from 0 to base_lr over warmup steps
# then decays via cosine to lr * min_ratio

# Loss: stablemax_cross_entropy (default) or softmax_cross_entropy
# Normalized by global_batch_size: ((1/global_batch_size) * loss).backward()

# Total loss = lm_loss + 0.5 * (q_halt_loss + q_continue_loss)
# lm_loss = stablemax_cross_entropy(logits, labels)
# q_halt_loss = BCE(q_halt_logits, is_correct)  
# q_continue_loss = BCE(q_continue_logits, target_q_continue)

# torch.compile is enabled by default (disable with DISABLE_COMPILE env var)
```

**Default hyperparameters from config/cfg_pretrain.yaml (likely, based on README):**

From README command lines:
- Sudoku-Extreme-1k: lr=7e-5 (single GPU) or lr=1e-4 (8 GPU), weight_decay=1.0, epochs=20000
- ARC: default lr (not specified, probably from cfg_pretrain.yaml)
- global_batch_size=384 (single GPU) or default (8 GPU)

---

## 11. Stablemax (NOT softmax)

```python
def s(x, epsilon=1e-30):
    return torch.where(x < 0, 1/(1-x+epsilon), x + 1)

def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x / torch.sum(s_x, dim=dim, keepdim=True))

def stablemax_cross_entropy(logits, labels):
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)  # NOTE: float64!
    ...
```

Stablemax is used instead of softmax for small-sample experiments. It runs in float64.
For full dataset training, they switch to standard softmax_cross_entropy.

---

## 12. Parameter Count Breakdown

Per Transformer block (hidden_size=512, num_heads=8, expansion=4):
- QKV projection: 512 × 1536 = 786,432
- O projection: 512 × 512 = 262,144
- SwiGLU gate_up: 512 × 3072 = 1,572,864
- SwiGLU down: 1536 × 512 = 786,432
- **Total per block: 3,407,872**

Architecture totals:
- L_level: 4 blocks × 3,407,872 = 13,631,488
- H_level: 4 blocks × 3,407,872 = 13,631,488
- Token embedding: vocab_size × 512
- LM head: 512 × vocab_size
- Q head: 512 × 2 + 2 (bias)
- Puzzle embedding: varies
- **~27M total** (matching paper's claim)

---

## 13. Key Differences from CORAL v2

| Aspect | HRM (actual) | CORAL v2 |
|--------|-------------|----------|
| Module type | Transformer (4 layers each) | 2-layer MLP |
| Attention | Full MHA with FlashAttention + RoPE | None (precision replaces attention) |
| Normalization | Post-Norm RMSNorm (no learnable params) | Pre-norm (likely) |
| FFN | SwiGLU with expansion 4 | GELU MLP |
| Gradient | 1-step (O(1) memory) | Full BPTT (O(T) memory) |
| Deep supervision | Yes (16 segments with detached carry) | No |
| ACT | Q-learning with exploration | Sigmoid halting |
| H×L steps | 2×2=4 per segment, 16 segments=64 total | K_max×T=8×8=64 total |
| Input to L | z_L + (z_H + input_emb) | z_L + mu_L + input_emb |
| Input to H | z_H + z_L | z_H + error |
| Output from | z_H (H-module) | z_1 (L-module) |
| Init states | Fixed truncated normal buffers | Learned parameters |
| Loss | stablemax_CE + Q-learning | CE + pred_error + halt_reg |
| Optimizer | AdamATan2 (scale-invariant) | Adam |
| Precision | bfloat16 forward | float32 |
| Compile | torch.compile enabled | torch.compile failed |

---

## 14. What CORAL v3 Must Match Exactly

To reproduce HRM, these must be identical:
1. ✅ Post-Norm RMSNorm (no learnable params)
2. ✅ 4 Transformer layers per module with SwiGLU
3. ✅ Full MHA with FlashAttention + RoPE (non-causal)
4. ✅ 1-step gradient (no_grad on all but final step)
5. ✅ Deep supervision with carry detach between segments
6. ✅ Q-learning ACT with exploration
7. ✅ Truncated LeCun Normal initialization (JAX-style)
8. ✅ Fixed initial states (buffers, not parameters)
9. ✅ AdamATan2 optimizer
10. ✅ Stablemax cross-entropy in float64
11. ✅ bfloat16 forward pass
12. ✅ torch.compile
13. ✅ Input injection: L gets z_H + input_emb, H gets z_L
14. ✅ Output from H-module, Q from first token of z_H
15. ✅ Puzzle embedding prepended to sequence

---

## 15. Where CORAL v3 Diverges (Our Novel Mechanisms)

These are the ONLY changes from HRM:

### A. Prediction Error Between Levels
- Instead of H getting raw z_L, H gets precision-weighted error: ξ = π ⊙ (z_L - predict(z_H))
- Instead of L getting raw z_H, L gets H's prediction: predict(z_H)
- Adds: PredictionNet, PrecisionNet (small MLPs)
- Adds: prediction error loss + precision regularizer to training objective

### B. Recognition-Gated Crystallization
- Before each L-cycle, recognition net checks if pattern is known
- If confident, bypass L computation entirely with codebook lookup
- Codebook learns offline from converged states
- Active only during inference (training always runs full computation)

### C. Sparse Columnar Routing (Index-Select, Strategy C)
- Each Transformer block replaced by S=8 smaller columns + router
- Router selects k=2 columns per sample via index-select dispatch
- Load-balancing loss prevents column collapse
- 1.6× overhead benchmarked on A100

### D. Unified Free Energy Objective
- Wraps HRM's existing stablemax_CE + Q-learning loss
- Adds: prediction error term, precision regularizer, load-balance term, crystallization supervision
