# Attention Backends

> **Location:** `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py`
> **Selection:** `SCOPE_KV_BIAS_BACKEND` environment variable
> **Note:** This is a conceptual overview; treat `causal_model.py` as the source of truth for availability checks and exact selection logic.

---

## The Challenge

Krea needs attention with **KV-cache bias** - down-weighting past frames to prevent drift. Standard FlashAttention doesn't support arbitrary score modification.

We support 4 backends with different tradeoffs:

| Backend | Score Mod | Hardware | Speed | Stability |
|---------|-----------|----------|-------|-----------|
| `fa4` | Native (CuTe `score_mod`) | When available (requires CuTe + CUTLASS; can run on SM100/SM103) | Fastest | Good |
| `flash` | Segment-combine | Default on SM103 (B300) | Fast | Good |
| `triton` | Custom Triton kernel | Default on non-SM103 | Medium | Good |
| `flex` | `flex_attention` | Any CUDA | Slow | Fallback |

---

## Backend Selection

```python
# Automatic selection:
# - SM103 (B300): default to "flash"
# - Otherwise: default to "triton"
default_backend = "flash" if _is_sm103() else "triton"

# Override with environment variable:
# SCOPE_KV_BIAS_BACKEND=fa4|flash|triton|flex
#
# Note: if you force "fa4" but CuTe/CUTLASS isn't available, the code will fall back to
# "flash" on SM103 or "triton" elsewhere.
```

---

## Backend 1: FA4 (FlashAttention 4 with CUTE)

**Best for:** when CuTe `score_mod` is available (environment + deps), regardless of GPU generation

FA4 natively supports `score_mod` - a function that modifies attention scores before softmax:

```python
def attention_with_kv_bias_fa4(q, k, v, log_bias, frame_seqlen, current_start):
    """
    Use FA4's native score_mod support.
    """
    def score_mod(score, b, h, q_idx, kv_idx):
        # Past frames (not first frame, not current block) get bias
        is_past = (kv_idx >= frame_seqlen) & (kv_idx < current_start)
        return torch.where(is_past, score + log_bias, score)

    return flash_attn_func(
        q, k, v,
        score_mod=score_mod,
        causal=False,  # We handle causality via score_mod
    )
```

### How Score Mod Works in FA4

```
Standard attention:
  score = Q @ K.T / sqrt(d)
  weights = softmax(score)
  output = weights @ V

With score_mod:
  score = Q @ K.T / sqrt(d)
  score = score_mod(score)  ← Applied here
  weights = softmax(score)
  output = weights @ V
```

The `score_mod` function is fused into the CUDA kernel, so there's no extra memory traffic.

---

## Backend 2: Flash (Segment Combine)

**Best for:** SM103 (B300) default path when FA4/CuTe isn't available

FlashAttention 2/3 doesn't support `score_mod`, but we can approximate the bias using **segment-based attention**:

```python
def attention_with_kv_bias_flash(q, k, v, bias, frame_seqlen, current_start):
    """
    Approximate KV-bias by running attention on segments with different scales.

    Strategy:
    1. First frame: full attention (no bias)
    2. Past frames: scaled attention (bias applied)
    3. Current block: full attention (no bias)
    4. Combine using log-sum-exp trick
    """
    # Segment 1: First frame
    k1, v1 = k[:, :frame_seqlen], v[:, :frame_seqlen]
    out1, lse1 = flash_attn_func(q, k1, v1, return_lse=True)

    # Segment 2: Past frames (with bias)
    k2 = k[:, frame_seqlen:current_start]
    v2 = v[:, frame_seqlen:current_start]
    out2, lse2 = flash_attn_func(q, k2, v2, return_lse=True)
    lse2 = lse2 + log_bias  # Apply bias to log-sum-exp

    # Segment 3: Current block
    k3, v3 = k[:, current_start:], v[:, current_start:]
    out3, lse3 = flash_attn_func(q, k3, v3, return_lse=True)

    # Combine using log-sum-exp
    return combine_attention_outputs(
        [out1, out2, out3],
        [lse1, lse2, lse3]
    )
```

### The Log-Sum-Exp Trick

When combining attention from multiple segments:

```python
def combine_attention_outputs(outputs, lses):
    """
    Combine attention outputs from multiple segments.

    Given:
      out_i = softmax(score_i) @ V_i
      lse_i = log(sum(exp(score_i)))

    We want:
      combined = softmax(concat(scores)) @ concat(Vs)

    This can be computed from individual outputs using:
      combined = sum(out_i * exp(lse_i - lse_max)) / sum(exp(lse_i - lse_max))
    """
    lse_max = torch.max(torch.stack(lses), dim=0).values
    weights = [torch.exp(lse - lse_max) for lse in lses]
    total_weight = sum(weights)

    combined = sum(out * w for out, w in zip(outputs, weights)) / total_weight
    return combined
```

This is mathematically exact (not an approximation).

---

## Backend 3: Triton (Custom Kernel)

**Best for:** General CUDA GPUs, when FA4 isn't available

Custom Triton kernel that implements attention with score modification:

```python
@triton.jit
def attention_with_bias_kernel(
    Q, K, V, Out,
    log_bias, frame_seqlen, current_start,
    stride_qb, stride_qh, stride_qs, stride_qd,
    ...
):
    """
    Triton attention kernel with KV-cache bias.
    """
    # Standard attention computation
    qk = tl.dot(q_block, k_block.T)
    qk = qk / sqrt_d

    # Apply bias to past frames
    is_past = (kv_indices >= frame_seqlen) & (kv_indices < current_start)
    qk = tl.where(is_past, qk + log_bias, qk)

    # Softmax
    m = tl.max(qk, axis=1)
    p = tl.exp(qk - m[:, None])
    l = tl.sum(p, axis=1)
    p = p / l[:, None]

    # Output
    out = tl.dot(p, v_block)
    tl.store(Out + out_offsets, out)
```

### Performance Notes

The Triton kernel is slower than FA4/Flash because:
1. No TMA (Tensor Memory Accelerator) - uses standard loads
2. No warp specialization
3. Less optimized memory access patterns

But it's more flexible and works on all CUDA GPUs.

---

## Backend 4: Flex (torch.compile)

**Best for:** Fallback when nothing else works

Uses PyTorch's `flex_attention` with `torch.compile`:

```python
def attention_with_kv_bias_flex(q, k, v, log_bias, frame_seqlen, current_start):
    """
    Use torch.nn.attention.flex_attention with score_mod.
    """
    def score_mod(score, b, h, q_idx, kv_idx):
        is_past = (kv_idx >= frame_seqlen) & (kv_idx < current_start)
        return torch.where(is_past, score + log_bias, score)

    # flex_attention is compiled by torch.compile
    return flex_attention(
        q.transpose(1, 2),  # flex expects [B, H, S, D]
        k.transpose(1, 2),
        v.transpose(1, 2),
        score_mod=score_mod,
    ).transpose(1, 2)
```

### Why It's Slow

1. **torch.compile overhead:** First call triggers compilation
2. **No memory-efficient algorithm:** Uses O(N²) memory
3. **Not fused:** score_mod runs as separate kernels

Use only as a correctness reference or when other backends fail.

---

## Performance Comparison

Measured on 320×576 video, 6-frame attention window:

| Backend | B200 | B300 | H100 |
|---------|------|------|------|
| `fa4` | 0.42ms ⭐ | N/A | N/A |
| `flash` | N/A | 0.48ms ⭐ | 0.52ms |
| `triton` | 0.65ms | 0.61ms | 0.58ms |
| `flex` | 1.2ms | 1.1ms | 0.95ms |

---

## Score Modification Regions

All backends implement the same logical bias pattern:

```
KV cache positions:
┌───────────────┬─────────────────────┬──────────────────┐
│   First Frame │     Past Frames     │   Current Block  │
│ [0, frame_seqlen) │ [frame_seqlen, curr_start) │ [curr_start, …) │
├───────────────┼─────────────────────┼──────────────────┤
│   No bias     │   ADD log(0.3)      │   No bias        │
│   (anchor)    │   (down-weight)     │   (full attn)    │
└───────────────┴─────────────────────┴──────────────────┘

Where:
  frame_seqlen = (H/scale) × (W/scale), where scale = vae_spatial_downsample_factor × patch_embedding_spatial_downsample_factor (default 16)
    - Example: 320×576 → 20×36 = 720 tokens/frame
    - Example: 480×832 → 30×52 = 1560 tokens/frame
  log(0.3) ≈ -1.2 (added to attention scores)
```

---

## Switching Backends

```bash
# Force a specific backend:
SCOPE_KV_BIAS_BACKEND=triton uv run python ...

# Check current backend:
python -c "from scope.core.pipelines.krea_realtime_video.modules.causal_model import _KV_BIAS_BACKEND; print(_KV_BIAS_BACKEND)"
```

---

## Debugging

If attention looks wrong:

1. **Test with flex backend:** It's the most readable and least optimized
   ```bash
   SCOPE_KV_BIAS_BACKEND=flex python ...
   ```

2. **Disable bias entirely:** Set `kv_cache_attention_bias=1.0` (no down-weighting)

3. **Check score_mod regions:** Log the indices being classified as "past"

---

## Related

- **KV cache:** [`kv-cache-mechanics.md`](kv-cache-mechanics.md)
- **Parent doc:** [`../krea-architecture.md`](../krea-architecture.md)
- **FA4 optimization notes:** `notes/FA4/b300/`
