# Rotary Position Embeddings (RoPE) for Video

> **Location:** `src/scope/core/pipelines/krea_realtime_video/modules/model.py`
> **Functions:** `rope_apply`, `causal_rope_apply`, `triton_rope_fused_3way`
> **Note:** Conceptual explainer — treat the referenced code + model config as the source of truth for exact dimension splits and frequency tables.

---

## What Problem Does This Solve?

Transformers are **permutation invariant** - they can't distinguish token order without explicit position information. For video, we need to encode:

1. **Temporal position:** Which frame is this token from?
2. **Spatial position:** Where in the frame (height, width)?

RoPE (Rotary Position Embedding) encodes position by **rotating** the query and key vectors, so that the attention score naturally depends on relative position.

---

## The Core Idea

Instead of adding position embeddings to the input, RoPE **rotates** Q and K vectors:

```python
# Standard attention:
score = Q @ K.T

# RoPE attention:
score = rotate(Q, pos_q) @ rotate(K, pos_k).T
#       ──────────────────────────────────────
#       The rotation encodes position!
```

The key insight: rotating both Q and K by their positions means the dot product depends on the **relative** position:

```
rotate(Q, pos_q) @ rotate(K, pos_k).T = f(Q, K, pos_q - pos_k)
                                              ^^^^^^^^^^^^^^^^
                                              relative position!
```

---

## The Rotation

For a 2D vector, rotation by angle θ is:

```
[x']   [cos θ  -sin θ] [x]
[y'] = [sin θ   cos θ] [y]
```

RoPE applies this to pairs of dimensions in the embedding:

```
For head_dim = 128:
  - Dimensions [0,1] rotate by θ₀
  - Dimensions [2,3] rotate by θ₁
  - ...
  - Dimensions [126,127] rotate by θ₆₃
```

Each dimension pair rotates at a **different frequency**, encoding position at multiple scales.

---

## 3D RoPE for Video

Video has 3 position axes: (frame, height, width). The embedding dimensions are split:

RoPE partitions the per-head embedding (`head_dim`) into three groups (T/H/W) and applies a separate rotary frequency table for each axis. The exact split is model-specific; consult `rope_apply`/`causal_rope_apply` for the concrete partition used by the current checkpoint/config.

### Frequency Table

```python
# Loaded from model config (see model.yaml / model.py):
rope_theta = ...
max_rope_freq_table_seq_len = ...

# Frequencies for each dimension:
freqs = 1.0 / (rope_theta ** (torch.arange(0, dim, 2) / dim))
# = [1.0, 0.95, 0.90, 0.86, ..., 0.004]
#   High freq (changes fast)      Low freq (changes slow)

# Position-specific angles:
angles = positions.unsqueeze(-1) * freqs.unsqueeze(0)
# Shape: [num_positions, dim/2]
```

---

## Interleaved vs Stacked Format

There are two ways to arrange the rotation pairs:

### Interleaved (what Wan2.1 uses):
```
[x₀, y₀, x₁, y₁, x₂, y₂, ...]
 └──┘    └──┘    └──┘
 pair0   pair1   pair2
```

### Stacked:
```
[x₀, x₁, x₂, ..., y₀, y₁, y₂, ...]
 └───first half───┘└───second half──┘
```

The interleaved format is more cache-friendly for the rotation operation.

---

## The Implementation

### Standard RoPE Apply

```python
def rope_apply(x, grid_sizes, freqs):
    """
    Apply 3D RoPE to tensor x.

    Args:
        x: [batch, seq_len, num_heads, head_dim]
        grid_sizes: [batch, 3] - (frames, height, width) for each sample
        freqs: [3, max_seq_len, dim//6, 2] - precomputed cos/sin for each axis

    Returns:
        Rotated x with same shape
    """
    # Split into 3 parts for F, H, W
    x_f, x_h, x_w = x.split([dim_f, dim_h, dim_w], dim=-1)

    # Get cos/sin for each position on each axis
    cos_f, sin_f = get_rope_cos_sin(freqs[0], frame_positions, ...)
    cos_h, sin_h = get_rope_cos_sin(freqs[1], height_positions, ...)
    cos_w, sin_w = get_rope_cos_sin(freqs[2], width_positions, ...)

    # Apply rotation to each part
    x_f = apply_rotary(x_f, cos_f, sin_f)
    x_h = apply_rotary(x_h, cos_h, sin_h)
    x_w = apply_rotary(x_w, cos_w, sin_w)

    return torch.cat([x_f, x_h, x_w], dim=-1)


def apply_rotary(x, cos, sin):
    """Rotate pairs of dimensions."""
    # Interleaved format: [x0, y0, x1, y1, ...]
    x0 = x[..., 0::2]  # Even indices
    x1 = x[..., 1::2]  # Odd indices

    # Rotation:
    # x0' = x0 * cos - x1 * sin
    # x1' = x0 * sin + x1 * cos
    x0_new = x0 * cos - x1 * sin
    x1_new = x0 * sin + x1 * cos

    # Interleave back
    return torch.stack([x0_new, x1_new], dim=-1).flatten(-2)
```

---

## Causal RoPE: Handling Streaming

In streaming generation, frames are generated incrementally. The position encoding must account for:

1. **Absolute position:** Frame 5 is always frame 5, even if we're only now generating it
2. **Cache consistency:** Cached K values must have the same positions as when they were computed

```python
def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    """
    Apply RoPE with frame offset for causal (streaming) generation.

    The start_frame offset ensures that frame N always gets position N,
    regardless of which generation step we're on.
    """
    # Frame positions are offset by start_frame
    frame_positions = torch.arange(num_frames) + start_frame
    #                                            ^^^^^^^^^^^^
    #                                            Key for streaming!

    # Height/width positions are always 0-indexed (within frame)
    height_positions = torch.arange(h)
    width_positions = torch.arange(w)

    # ... rest is same as standard rope_apply
```

### Why This Matters

Without offset:
```
Generation step 1: frames 0-2 get positions [0, 1, 2]
Generation step 2: frames 3-5 get positions [0, 1, 2]  ← WRONG!
                   (should be [3, 4, 5])
```

With offset:
```
Generation step 1: start_frame=0, frames 0-2 get positions [0, 1, 2]
Generation step 2: start_frame=3, frames 3-5 get positions [3, 4, 5]  ← Correct!
```

---

## Triton Fused 3-Way Kernel

The standard implementation requires 3 separate passes (F, H, W). The Triton kernel fuses them:

```python
@triton.jit
def rope_fused_3way_kernel(
    x_ptr, freqs_f_ptr, freqs_h_ptr, freqs_w_ptr,
    out_ptr,
    F, H, W, D, start_frame,
    ...
):
    """
    Fused RoPE kernel that applies all 3 axes in one pass.

    Each thread handles one (batch, seq, head) position.
    """
    # Load x values for this position
    x = tl.load(x_ptr + offsets)

    # Compute position on each axis
    seq_idx = (program_id % seq_len)
    f_idx = seq_idx // (H * W) + start_frame
    h_idx = (seq_idx // W) % H
    w_idx = seq_idx % W

    # Load precomputed cos/sin for each axis
    cos_f, sin_f = load_freq(freqs_f_ptr, f_idx, dim_f)
    cos_h, sin_h = load_freq(freqs_h_ptr, h_idx, dim_h)
    cos_w, sin_w = load_freq(freqs_w_ptr, w_idx, dim_w)

    # Apply rotations (fused)
    x_f = rotate(x[:dim_f], cos_f, sin_f)
    x_h = rotate(x[dim_f:dim_f+dim_h], cos_h, sin_h)
    x_w = rotate(x[dim_f+dim_h:], cos_w, sin_w)

    # Store result
    tl.store(out_ptr + offsets, concat(x_f, x_h, x_w))
```

### Performance
The fused Triton path exists to reduce extra passes and memory traffic. Actual speedups depend heavily on shape/layout, compile settings, and GPU generation; treat any single-number claim as non-portable.

The fusion eliminates redundant memory reads/writes between the 3 axes.

---

## Frequency Table Caching

Computing cos/sin for every position is expensive. We precompute and cache:

```python
@lru_cache(maxsize=32)
def get_rope_cos_sin(freqs, positions, dtype, device):
    """
    Cached computation of cos/sin tables.

    Cache key: (freqs_id, positions_tuple, dtype, device)
    """
    angles = positions.unsqueeze(-1) * freqs.unsqueeze(0)
    cos = angles.cos()
    sin = angles.sin()
    return cos.to(dtype=dtype, device=device), sin.to(dtype=dtype, device=device)
```

The cache is keyed by position tuple, so different sequence lengths reuse computation.

---

## Visual: 3D Position Encoding

```
Frame 0:                    Frame 1:
┌─────────────────┐         ┌─────────────────┐
│ (0,0,0) (0,0,1) │         │ (1,0,0) (1,0,1) │
│ (0,1,0) (0,1,1) │         │ (1,1,0) (1,1,1) │
│ (0,2,0) (0,2,1) │         │ (1,2,0) (1,2,1) │
└─────────────────┘         └─────────────────┘
     ↑                           ↑
   (f,h,w)                     (f,h,w)

Each position gets a unique rotation:
- (0,0,0) → rotate by θ(f=0) ⊕ θ(h=0) ⊕ θ(w=0)
- (1,2,1) → rotate by θ(f=1) ⊕ θ(h=2) ⊕ θ(w=1)

The ⊕ is "combine" - each axis affects different dimensions.
```

---

## Why RoPE Over Learned Embeddings?

| Aspect | Learned Embeddings | RoPE |
|--------|-------------------|------|
| **Extrapolation** | Poor (only trained positions) | Good (generalizes to longer sequences) |
| **Memory** | O(max_seq_len × dim) | O(dim) for frequencies |
| **Relative position** | Requires special design | Built-in |
| **Training stability** | Can have issues | Stable (no learned params) |

For video generation with variable lengths and streaming, RoPE's extrapolation and relative position awareness are crucial.

---

## Related

- **Original RoPE paper:** [RoFormer](https://arxiv.org/abs/2104.09864)
- **Parent doc:** [`../krea-architecture.md`](../krea-architecture.md)
- **Causal attention:** [`causal-attention-masks.md`](causal-attention-masks.md)
