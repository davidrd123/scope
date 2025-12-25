# How FlashAttention Does RoPE

> **What this is:** An explainer for Rotary Position Embeddings (RoPE) - how they work mathematically, how FlashAttention implements them, and why fusing RoPE into attention is both attractive and tricky.
> **Context:** We're exploring Level 5 optimization (fused operations). RoPE is a natural fusion target because it's memory-bound. But it's more complex than `score_mod` because it modifies Q/K **before** the dot product, not scores **after**.
> **Updated:** 2025-12-25

---

## Overview

### What is RoPE?

RoPE (Rotary Position Embedding) encodes **position information** into attention without explicit position embeddings. Instead of adding position embeddings to tokens, RoPE **rotates** Q and K vectors based on their position.

The key insight: when you rotate Q and K by position-dependent angles, the dot product `Q_i · K_j` naturally depends on the **relative** position `i - j`. This gives the model position awareness without any extra parameters.

### Why RoPE Matters for Us

1. **Almost every modern LLM uses RoPE** (Llama, Mistral, Qwen, etc.)
2. **It's applied before attention** - Q and K get rotated, then we compute attention
3. **It's a separate kernel today** - memory round-trip between RoPE and attention
4. **Fusion opportunity** - if we rotate Q/K during attention tile loading, we save bandwidth

### The Punchline

RoPE fusion is attractive but fundamentally different from `score_mod`:

| Aspect | score_mod | RoPE fusion |
|--------|-----------|-------------|
| **Modifies** | Attention scores (after QK^T) | Q and K vectors (before QK^T) |
| **Hook point** | Inside attention loop, after score compute | During tile loading / prologue |
| **FA4 support** | Yes (`score_mod` parameter) | No (would need `q_mod`/`k_mod`) |

---

## The Math

### Complex Rotation View

The elegant way to understand RoPE: treat pairs of dimensions as complex numbers, then rotate.

For position `m` and pair index `d`, you form a complex number from a pair of real dims. The exact pairing depends on the RoPE layout:
```
# GPT-J / interleaved:
z = x[2d] + i·x[2d+1]

# GPT-NeoX / split-halves (FlashAttention default):
z = x[d] + i·x[d + dim/2]

z_rot = z · e^(i·m·θ_d)        # Rotate by position-dependent angle
                               # θ_d = 1 / base^(2d/dim)
```

The angle `θ_d` decreases for higher dimensions - low dimensions rotate fast (high frequency), high dimensions rotate slow (low frequency). This gives the model multi-scale position sensitivity.

### The Actual Formula

In practice, we don't use complex numbers. The 2x2 rotation matrix is:

```
[cos(mθ)  -sin(mθ)]   [x0]     [x0·cos - x1·sin]
[sin(mθ)   cos(mθ)] × [x1]  =  [x0·sin + x1·cos]
```

So for each pair `(x0, x1)`:
```python
out0 = x0 * cos - x1 * sin
out1 = x0 * sin + x1 * cos
```

That's the entire kernel math. Simple!

### Why Relative Positions Fall Out

When Q (at position i) and K (at position j) are both rotated, their dot product becomes:
```
Q_rot · K_rot = f(Q, K, i-j)
```

The absolute positions `i` and `j` cancel; only the relative position `i-j` matters. This is why RoPE gives relative position awareness "for free."

---

## FA's Implementation

FlashAttention packages commonly have a two-layer RoPE implementation:

### Layer 1: Python Interface (`flash_attn/layers/rotary.py`)

**`RotaryEmbedding` class** - the main module:
```python
class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000.0, interleaved=False, scale_base=None):
        # Precompute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2) / dim))

        # Cache cos/sin tables
        self._cos_cached = None
        self._sin_cached = None
```

**Key insight:** The cos/sin tables are **precomputed and cached**. For position `m` and dimension index `d`:
```python
θ_d = inv_freq[d]           # 1 / base^(2d/dim)
cos[m, d] = cos(m * θ_d)
sin[m, d] = sin(m * θ_d)
```

**Calling conventions:**
```python
# Apply to separate Q, K
q = apply_rotary_emb(q, cos, sin)
k = apply_rotary_emb(k, cos, sin)

# Apply to packed QKV (often 1 kernel instead of 2 when `qkv` is contiguous and Q/K share cos/sin)
qkv = apply_rotary_emb_qkv_(qkv, cos, sin)  # inplace

# Apply only to K (e.g., when Q doesn't need rotation)
kv = apply_rotary_emb_kv_(kv, cos, sin)  # inplace
```

### Layer 2: Kernel Implementation (often `flash_attn/ops/triton/rotary.py`)

The actual GPU kernel. Key parts:

**Grid:** One block per (head_chunk, seq_chunk, batch)
```python
grid = lambda META: (
    triton.cdiv(nheads, META["BLOCK_H"]),  # heads
    triton.cdiv(seqlen, META["BLOCK_M"]),   # sequence positions
    batch                                    # batch dimension
)
```

**Core computation (GPT-NeoX style, non-interleaved):**
```python
# Load first half and second half of head dimension
x0 = tl.load(X, mask=mask).to(tl.float32)
x1 = tl.load(X + ROTARY_DIM_HALF * stride_x_headdim, mask=mask).to(tl.float32)

# Rotate
o0 = x0 * cos - x1 * sin
o1 = x0 * sin + x1 * cos

# Store back
tl.store(OUT, o0, mask=mask)
tl.store(OUT + ROTARY_DIM_HALF * stride_out_headdim, o1, mask=mask)
```

**Position offset handling:**
```python
if not IS_SEQLEN_OFFSETS_TENSOR:
    rm_cs = rm + SEQLEN_OFFSETS  # scalar offset (common case)
else:
    rm_cs = rm + tl.load(SEQLEN_OFFSETS + pid_batch)  # per-batch offset
```

---

## Two Styles: GPT-NeoX vs GPT-J

RoPE can pair dimensions in two ways:

### GPT-NeoX Style (Split Halves)

```
head_dim = 128
x0 = x[0:64]     # First half
x1 = x[64:128]   # Second half
```

Pairs: `(dim 0, dim 64), (dim 1, dim 65), ..., (dim 63, dim 127)`

**This is the default** in FlashAttention (`interleaved=False`).

### GPT-J Style (Interleaved)

```
head_dim = 128
x0 = x[0::2]     # Even dims: 0, 2, 4, ...
x1 = x[1::2]     # Odd dims:  1, 3, 5, ...
```

Pairs: `(dim 0, dim 1), (dim 2, dim 3), ..., (dim 126, dim 127)`

**Less common** but some models use it (`interleaved=True`).

### Why Two Styles?

Historical accident. Both are mathematically equivalent - they just pair different dimensions. The NeoX style is slightly more hardware-friendly because it's two contiguous loads.

---

## KV Cache Integration

The clever part of FA's RoPE: **`seqlen_offsets`** handles KV cache seamlessly.

### The Problem

In inference with KV cache:
- **Q** has position = `cache_length` (we're generating new token(s))
- **K** has positions = `0, 1, 2, ..., cache_length` (the whole sequence)

But wait - the cached K values were **already rotated** when they were first computed. We only need to rotate the **new** K (and the new Q).

### The Solution

```python
apply_rotary_emb(
    q,
    cos, sin,
    seqlen_offsets=cache_length,  # Q starts at position cache_length
)

apply_rotary_emb(
    k_new,
    cos, sin,
    seqlen_offsets=cache_length,  # New K also at position cache_length
)
# Cached K was already rotated - don't touch it
```

The kernel handles this with position arithmetic:
```python
# Effective position = block position + offset
rm_cs = rm + SEQLEN_OFFSETS

# Index into cos/sin at the correct position
COS = COS + (rm_cs[:, None] * ROTARY_DIM_HALF + rk_half[None, :])
```

### Variable-Length Sequences (varlen)

For batched sequences of different lengths:
```python
apply_rotary_emb(
    x,  # (total_seqlen, nheads, headdim) - packed
    cos, sin,
    cu_seqlens=cu_seqlens,  # (batch + 1,) - cumulative lengths
    max_seqlen=max_seqlen,
)
```

Each sequence in the batch can have a different length, and the kernel handles the indexing.

---

## Why RoPE is Separate Today

### Memory Traffic Pattern

Current flow:
```
Q in HBM → Load → RoPE kernel → Store → Q' in HBM
K in HBM → Load → RoPE kernel → Store → K' in HBM
Q' in HBM → Load ──┐
K' in HBM → Load ──┼─→ Attention kernel → Store → O in HBM
V in HBM → Load ───┘
```

**Problem:** Q and K make a full round-trip to HBM between RoPE and attention. For memory-bound workloads, this is wasted bandwidth.

### Why Not Just Fuse?

Several reasons RoPE is tricky to fuse into attention:

1. **Different offsets for Q vs K:**
   - In KV-cache scenarios, Q might be at position 100 while K spans positions 0-100
   - The attention kernel would need per-tensor position offset handling

2. **cos/sin tables:**
   - Either pass as extra tensors to attention
   - Or recompute on-the-fly (trades compute for memory)
   - Both add complexity to the kernel

3. **Partial rotation:**
   - RoPE might only rotate the first `rotary_dim` dimensions
   - Rest of head_dim passes through unchanged
   - Adds conditional logic

4. **No hook point:**
   - FA4's `score_mod` runs **after** QK^T
   - RoPE needs to run **before** QK^T, during tile loading
   - Would need a new `q_mod`/`k_mod` callback mechanism

---

## Fusion Opportunity: Where Would We Inject RoPE?

### The Ideal Injection Point

In FA4's attention loop (pseudocode):
```python
for kv_tile in kv_tiles:
    # === PROLOGUE: Load tiles ===
    Q_tile = load_tile(Q)
    K_tile = load_tile(K)
    V_tile = load_tile(V)

    # === HERE: Apply RoPE to Q_tile, K_tile in registers ===
    Q_tile = rotate(Q_tile, cos_q, sin_q)  # No HBM round-trip!
    K_tile = rotate(K_tile, cos_k, sin_k)

    # === COMPUTE ===
    S = Q_tile @ K_tile.T  # Scores (in SRAM)
    S = S / sqrt(d)
    S = score_mod(S)       # <-- Our KV-bias lives here
    P = softmax(S)
    O += P @ V_tile
```

The rotation happens **in registers** right after loading, before any SRAM storage. Zero extra memory traffic.

### What Would We Need?

1. **Callback mechanism:**
   ```python
   @cute.jit
   def q_mod(q_tile, positions, cos, sin):
       # Rotate Q tile by positions
       return rotated_q

   @cute.jit
   def k_mod(k_tile, positions, cos, sin):
       # Rotate K tile by positions
       return rotated_k
   ```

2. **Position tracking:**
   - Q tile positions (current generation position)
   - K tile positions (varies across the KV sequence)

3. **cos/sin table access:**
   - Pass tables as aux_tensors
   - Or embed in closure (if positions are fixed)

### Reference Implementation

For inspiration, look at how the Triton kernel already does it:
```python
# From rotary.py kernel
x0 = tl.load(X, mask=mask).to(tl.float32)
x1 = tl.load(X + ROTARY_DIM_HALF * stride_x_headdim, mask=mask).to(tl.float32)
o0 = x0 * cos - x1 * sin
o1 = x0 * sin + x1 * cos
```

This same logic would go in `q_mod`/`k_mod`, just operating on tiles instead of full tensors.

---

## Questions & Next Steps

### What's Unclear

1. **How does FA4 structure its tile loading?**
   - Need to trace `flash_fwd_sm100.py` to find the exact injection point
   - Where are Q/K tiles in registers vs shared memory?

2. **Position metadata flow:**
   - How do tile positions get tracked through the attention loop?
   - Is there existing infrastructure we can piggyback on?

3. **Compile-time vs runtime:**
   - Can we make position offsets compile-time constants (like `score_mod` closure variables)?
   - Or do they need to be runtime tensors?

### Opportunities

1. **Learning project:** Modify FA4 to add `q_mod`/`k_mod` callbacks
   - Even if we don't get speedup, we learn the internals
   - Document what we find

2. **Measure first:** How much time does RoPE actually take?
   - Recent B300 profiles show `rope_apply` ~0.11ms/call
   - That's small compared to overall self_attn (~several ms)
   - Fusion might not be a big FPS lever by itself

3. **Alternative: Recompute RoPE:**
   - Instead of loading precomputed cos/sin tables, compute on-the-fly
   - Trades ALU for memory bandwidth
   - Might be worth it on memory-bound workloads

### Next Explainer

This naturally leads to:
- **Explainer #5: TMA and Memory Loading** - How FA4 loads tiles, where hooks could go
- **Explainer #1: How FA4 Attention Works** - The overall flow that RoPE fusion would slot into

---

## References

### Code Files

| File | Description |
|------|-------------|
| `flash_attn/layers/rotary.py` | Python interface (`RotaryEmbedding`, `apply_rotary_emb*`). Path depends on your installed `flash_attn`; locate via `python -c "import flash_attn.layers.rotary as r; print(r.__file__)"`. |
| `flash_attn/ops/triton/rotary.py` | Triton kernel in many `flash_attn` versions; locate via `python -c "import flash_attn.ops.triton.rotary as r; print(r.__file__)"`. |
| `vendored/rope/` | Reference implementations from vLLM, transformers |

### Papers

| Paper | What It Covers |
|-------|----------------|
| [RoFormer (Su et al., 2021)](https://arxiv.org/abs/2104.09864) | Original RoPE paper |
| [XPos (Sun et al., 2022)](https://arxiv.org/abs/2212.10554) | Scale-based extension (FA supports via `scale_base`) |

### Related Docs

- [03-score-mod.md](03-score-mod.md) - How we inject KV-bias (score_mod is after QK^T)
- `notes/FA4/b300/optimization-ladder.md` - Where RoPE fusion fits in the optimization hierarchy
- `notes/FA4/b300/level5-level6-resources.md` - Resources for Level 5 work
