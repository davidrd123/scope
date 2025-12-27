# Block-wise Causal Attention Masks

> **Location:** `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py`
> **Functions:** `_prepare_blockwise_causal_attn_mask`, `_prepare_blockwise_causal_attn_mask_i2v`
> **Note:** Conceptual explainer — token counts like “tokens per frame” are configuration-dependent. Treat the referenced functions as the source of truth.

---

## What Problem Does This Solve?

Standard causal attention allows each token to attend to all previous tokens:

```
Token-level causal (too fine-grained for video):
          t0  t1  t2  t3  t4  t5  ...
    t0    ✓   ✗   ✗   ✗   ✗   ✗
    t1    ✓   ✓   ✗   ✗   ✗   ✗
    t2    ✓   ✓   ✓   ✗   ✗   ✗
    t3    ✓   ✓   ✓   ✓   ✗   ✗
    ...
```

But video frames are generated in **blocks** (e.g., 3 frames at a time). Within a block, all tokens should attend to each other:

```
Block-wise causal (what we want):
          [Block 0] [Block 1] [Block 2]
[Block 0]    ✓         ✗         ✗
[Block 1]    ✓         ✓         ✗
[Block 2]    ✓         ✓         ✓
```

---

## The Mask Structure

For a video with 9 frames (3 blocks of 3 frames):

```
          Frame 0    Frame 1    Frame 2    Frame 3    Frame 4    Frame 5    ...
          (Block 0)  (Block 0)  (Block 0)  (Block 1)  (Block 1)  (Block 1)
Frame 0      ✓          ✓          ✓          ✗          ✗          ✗
Frame 1      ✓          ✓          ✓          ✗          ✗          ✗
Frame 2      ✓          ✓          ✓          ✗          ✗          ✗
Frame 3      ✓          ✓          ✓          ✓          ✓          ✓
Frame 4      ✓          ✓          ✓          ✓          ✓          ✓
Frame 5      ✓          ✓          ✓          ✓          ✓          ✓
...
```

Key properties:
- All tokens in Block 0 can attend to all tokens in Block 0
- All tokens in Block 1 can attend to all tokens in Block 0 AND Block 1
- Future blocks are masked (✗)

---

## Implementation

```python
def _prepare_blockwise_causal_attn_mask(
    device,
    num_frames,
    frame_seqlen,        # Tokens per frame = (H/scale)×(W/scale) (e.g., 720 at 320×576 with scale=16)
    num_frame_per_block, # Frames per block (e.g., 3)
    local_attn_size=-1,  # Local window in frames (-1 = global)
):
    """
    Create a block-wise causal attention mask.

    Returns a function mask_mod(b, h, q_idx, kv_idx) -> bool
    that returns True if attention is allowed.
    """
    total_seq_len = num_frames * frame_seqlen
    block_size = num_frame_per_block * frame_seqlen

    # Precompute: for each position, what's the end of its block?
    # ends[i] = last position (exclusive) that position i can attend to
    ends = torch.zeros(total_seq_len, device=device, dtype=torch.long)

    for block_start in range(0, total_seq_len, block_size):
        block_end = min(block_start + block_size, total_seq_len)
        ends[block_start:block_end] = block_end

    def mask_mod(b_idx, h_idx, q_idx, kv_idx):
        """
        Returns True if query at q_idx can attend to key at kv_idx.
        """
        if local_attn_size == -1:
            # Global: attend to current block and all previous
            return kv_idx < ends[q_idx]
        else:
            # Local: only attend within window
            local_start = ends[q_idx] - local_attn_size * frame_seqlen
            return (kv_idx < ends[q_idx]) & (kv_idx >= local_start)

    return mask_mod
```

---

## Visual: Block Boundaries

```
Sequence positions (9 frames, 3 per block, 4 tokens per frame for illustration):

Position:  0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 ...
Frame:     0  0  0  0  1  1  1  1  2  2  2  2  3  3  3  ...
Block:     ────Block 0────  ────Block 0────  ────Block 0────  ────Block 1────

ends[]:    12 12 12 12 12 12 12 12 12 12 12 12 24 24 24 ...
           └─────────────────────────────────┘ └─────────────
           Block 0 positions can see up to 12   Block 1 can see up to 24
```

---

## Local Attention Window

With `local_attn_size=6` (6 frames):

```
Without local attention (global):
Block 5 can attend to: [Block 0, Block 1, Block 2, Block 3, Block 4, Block 5]

With local_attn_size=6:
Block 5 can attend to: [Block 3, Block 4, Block 5]
                       └── 6 frames back ──┘

Memory savings: O(N²) → O(N × window)
```

### Implementation Detail

```python
def mask_mod(b_idx, h_idx, q_idx, kv_idx):
    block_end = ends[q_idx]
    block_start = block_end - local_attn_size * frame_seqlen

    return (kv_idx < block_end) & (kv_idx >= block_start)
    #       ─────────────────     ────────────────────────
    #       Standard causal       Local window constraint
```

---

## I2V (Image-to-Video) Mask

For image-conditioned generation, the first frame is special:

```python
def _prepare_blockwise_causal_attn_mask_i2v(...):
    """
    I2V mask: first frame attends only to itself.

    Used when:
    - First frame is a reference image (not generated)
    - It shouldn't condition on other frames
    """
    # First frame: only self-attention
    ends[:frame_seqlen] = frame_seqlen

    # Subsequent blocks: standard block-wise causal
    for block_start in range(frame_seqlen, total_seq_len, block_size):
        ends[block_start:block_start + block_size] = block_start + block_size
```

Visual:

```
Standard T2V:                    I2V:
┌───────────────────────┐        ┌───────────────────────┐
│ [F0][F0][F0]          │        │ [F0]    ✗    ✗        │ ← F0 isolated
│ [F0][F1][F1]          │        │ [F0][F1][F1]          │
│ [F0][F1][F2]          │        │ [F0][F1][F2]          │
└───────────────────────┘        └───────────────────────┘
F0 attends to all                F0 attends only to F0
```

---

## Mask Caching

Block masks are expensive to create. We cache them:

```python
@functools.lru_cache(maxsize=16)
def get_cached_block_mask(num_frames, frame_seqlen, num_frame_per_block, local_attn_size, device):
    return _prepare_blockwise_causal_attn_mask(
        device, num_frames, frame_seqlen, num_frame_per_block, local_attn_size
    )
```

Cache key includes all parameters that affect the mask shape.

---

## Integration with flex_attention

The mask function is used with `torch.nn.attention.flex_attention`:

```python
from torch.nn.attention import flex_attention
from torch.nn.attention.flex_attention import create_block_mask

# Create block mask (sparse representation for efficiency)
block_mask = create_block_mask(
    mask_mod,
    B=batch_size,
    H=num_heads,
    Q_LEN=seq_len,
    KV_LEN=seq_len,
    device=device,
)

# Use in attention
output = flex_attention(query, key, value, block_mask=block_mask)
```

The `block_mask` is a sparse representation that flex_attention uses to skip masked blocks entirely.

---

## Mask During KV Cache Recomputation

During recomputation, we fill the cache with context frames. The mask is different:

```python
# During recompute: all context frames use global attention
# (they're all "in the past" relative to what we're about to generate)

recompute_mask = _prepare_blockwise_causal_attn_mask(
    num_frames=num_context_frames,
    local_attn_size=-1,  # Global attention for context
)
```

After recomputation, the mask is cleared:

```python
model.block_mask = None  # Generation uses KV cache, not mask
```

---

## Why Block-wise Instead of Token-wise?

| Aspect | Token-wise Causal | Block-wise Causal |
|--------|-------------------|-------------------|
| **Parallelism** | Each token sequential | Entire block parallel |
| **Temporal consistency** | Frame-internal order matters | Frames are atomic units |
| **Cache efficiency** | Fine-grained updates | Block-level updates |
| **Semantic match** | Doesn't match video structure | Matches how frames are generated |

Video diffusion naturally operates on frame blocks. Block-wise causality aligns the attention pattern with the generation pattern.

---

## Debugging Masks

To visualize a mask:

```python
import matplotlib.pyplot as plt

# Create mask
mask_fn = _prepare_blockwise_causal_attn_mask(
    device="cpu",
    num_frames=9,
    frame_seqlen=4,  # Small for visualization
    num_frame_per_block=3,
)

# Evaluate on all positions
seq_len = 9 * 4
mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
for q in range(seq_len):
    for kv in range(seq_len):
        mask[q, kv] = mask_fn(0, 0, q, kv)

# Plot
plt.imshow(mask, cmap='Blues')
plt.xlabel('Key position')
plt.ylabel('Query position')
plt.title('Block-wise Causal Mask')
plt.savefig('mask.png')
```

---

## Related

- **KV cache:** [`kv-cache-mechanics.md`](kv-cache-mechanics.md)
- **Attention backends:** [`attention-backends.md`](attention-backends.md)
- **Parent doc:** [`../krea-architecture.md`](../krea-architecture.md)
