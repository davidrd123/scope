# KV Cache Mechanics

> **KV update/eviction:** `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py` (`CausalWanSelfAttention.forward`)
> **Cache setup:** `src/scope/core/pipelines/wan2_1/blocks/setup_caches.py` (`SetupCachesBlock`)
> **Cache recompute:** `src/scope/core/pipelines/krea_realtime_video/blocks/recompute_kv_cache.py` (`RecomputeKVCacheBlock`)
> **Note:** Conceptual explainer — exact tensor shapes/sizes depend on config (resolution, heads, cache window). Use the linked code as the source of truth.

---

## What Problem Does This Solve?

In autoregressive generation, each new frame needs to attend to all previous frames. Without caching:

```
Frame 0: Compute K, V for frame 0
Frame 1: Recompute K, V for frame 0, compute for frame 1
Frame 2: Recompute K, V for frames 0, 1, compute for frame 2
...
Frame N: Recompute K, V for frames 0..N-1, compute for frame N
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
         O(N²) redundant work!
```

With KV caching:

```
Frame 0: Compute K, V for frame 0, store in cache
Frame 1: Load cached K, V for frame 0, compute for frame 1, append to cache
Frame 2: Load cached K, V for frames 0-1, compute for frame 2, append to cache
...
Frame N: Load cached K, V for frames 0..N-1, compute for frame N
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
         O(1) cache lookup!
```

---

## Cache Structure

Each attention layer maintains its own cache:

```python
kv_cache = {
    "k": torch.zeros(batch, max_cache_size, num_heads, head_dim),
    "v": torch.zeros(batch, max_cache_size, num_heads, head_dim),
    "global_end_index": 0,   # Total tokens ever processed
    "local_end_index": 0,    # Current position in cache buffer
}

# Full model has one cache per layer:
kv_caches = [kv_cache for _ in range(num_layers)]  # One cache per transformer block
```

### Frame Sequence Length (`frame_seqlen`)

Each **latent frame** is flattened into `frame_seqlen` tokens:

```
frame_seqlen = (H/scale) × (W/scale)
scale = vae_spatial_downsample_factor × patch_embedding_spatial_downsample_factor  # default 16
```

Examples:
- 320×576 → 20×36 = 720 tokens/frame
- 480×832 → 30×52 = 1560 tokens/frame

### Why Two Indices?

| Index | Meaning | Example |
|-------|---------|---------|
| `global_end_index` | Total tokens seen across all time | After 10 frames: `10 × frame_seqlen` |
| `local_end_index` | Current write position in cache buffer | Moves as we append/evict within the fixed cache window |

The difference matters for:
- **RoPE:** Uses `global_end_index` for position encoding (absolute position)
- **Cache writes:** Uses `local_end_index` for buffer position

---

## Cache Eviction

The cache has finite size. When full, oldest tokens are evicted:

```
Before eviction (cache full):
┌────────────────────────────────────────────────────────────┐
│ [sink] [frame1] [frame2] [frame3] [frame4] [frame5] [new] │
└────────────────────────────────────────────────────────────┘
         ^^^^^^^^
         will be evicted

After eviction:
┌────────────────────────────────────────────────────────────┐
│ [sink] [frame2] [frame3] [frame4] [frame5] [new] [empty]  │
└────────────────────────────────────────────────────────────┘
```

### Sink Tokens

If `sink_size > 0`, the first `sink_size` frames are **never evicted**. In code, this is expressed in **tokens**:

```python
# In CausalWanSelfAttention.forward:
sink_tokens = sink_size * frame_seqlen  # sink_size is in frames

# During eviction:
num_evicted = num_new_tokens + local_end_index - cache_size
num_to_roll = local_end_index - num_evicted - sink_tokens

# Shift non-sink tokens left:
kv_cache["k"][:, sink_tokens:sink_tokens + num_to_roll] = \
    kv_cache["k"][:, sink_tokens + num_evicted:local_end_index].clone()
```

**Why keep the first frame?**
- It anchors the video's initial state
- Prevents drift from the starting point
- Acts as a "ground truth" reference

---

## The Recomputation Problem

Even with caching, errors accumulate:

```
Ground truth:  [Frame 0] → [Frame 1] → [Frame 2] → [Frame 3]
                   ↓           ↓           ↓           ↓
Generated:     [Frame 0] → [Frame 1'] → [Frame 2''] → [Frame 3''']
                              ↑             ↑             ↑
                           small err    bigger err    drift!
```

Each generation step adds a bit of error. After many frames, the model is conditioning on increasingly corrupted context.

### Solution: Periodic Recomputation

Before generating each new block, we:
1. Take the **decoded frames** from the previous block
2. **Re-encode** them through the VAE
3. Run a **forward pass** to fill a fresh cache

```
Recompute flow:
                                    Fresh cache
                                        │
[Frame 0] + [Decoded Frame N-2, N-1] ──►│──► Generate Frame N
                │                       │
                └── Re-encoded ─────────┘
                    (clean latents)
```

### Why Re-encode?

The cached latents have accumulated errors. By going through:
```
Latents → Decode → Pixels → Encode → Fresh Latents
```

We "round-trip" through the VAE, which acts as a regularizer. The decoded pixels are more stable than the drifted latents.

---

## RecomputeKVCacheBlock in Detail

```python
class RecomputeKVCacheBlock:
    def __call__(self, components, state):
        if current_start_frame == 0:
            # First block: just initialize empty buffers
            state.set("context_frame_buffer", torch.zeros(...))
            state.set("decoded_frame_buffer", torch.zeros(...))
            return components, state

        # ═══════════════════════════════════════════════════════
        # Step 1: Gather context frames
        # ═══════════════════════════════════════════════════════

        if current_start_frame < kv_cache_num_frames:
            # Early in video: use original first frame
            context = torch.cat([first_context_frame, context_buffer])
        else:
            # Later: re-encode first frame from decoded buffer
            decoded_first = decoded_frame_buffer[:, :1]
            reencoded_first = vae.encode_to_latent(decoded_first)
            context = torch.cat([reencoded_first, context_buffer])

        # ═══════════════════════════════════════════════════════
        # Step 2: Initialize fresh KV cache
        # ═══════════════════════════════════════════════════════

        kv_cache = initialize_kv_cache(
            batch_size=1,
            num_layers=num_layers,
            cache_size=max_cache_size,
            num_heads=num_heads,
            head_dim=head_dim,
        )

        # ═══════════════════════════════════════════════════════
        # Step 3: Create block mask for context
        # ═══════════════════════════════════════════════════════

        # Context frames use global attention (not causal)
        # because they're all "in the past"
        block_mask = model._prepare_blockwise_causal_attn_mask(
            num_frames=num_context_frames,
            local_attn_size=-1,  # -1 = global attention
        )

        # ═══════════════════════════════════════════════════════
        # Step 4: Forward pass to fill cache (no denoising)
        # ═══════════════════════════════════════════════════════

        generator(
            noisy_image_or_video=context,
            timestep=torch.zeros(...),  # t=0 means "no noise"
            kv_cache=kv_cache,
            block_mask=block_mask,
        )

        # ═══════════════════════════════════════════════════════
        # Step 5: Clear block mask for generation
        # ═══════════════════════════════════════════════════════

        model.block_mask = None

        state.set("kv_cache", kv_cache)
        return components, state
```

---

## KV Cache Attention Bias

Even with recomputation, older frames can dominate attention. Solution: **down-weight past frames**.

```python
# Attention scores are modified:
score = Q @ K.T / sqrt(d)

# With bias:
score = Q @ K.T / sqrt(d) + bias_mask
```

The `bias_mask` is:

```
                    K positions
              [frame0] [past] [current]
Q positions    ─────────────────────────
[current]      [  0  ] [log(b)] [  0  ]

Where b = kv_cache_attention_bias (default 0.3)
      log(0.3) ≈ -1.2
```

### Visual Intuition

```
Without bias (kv_cache_attention_bias=1.0):
┌─────────────────────────────────────────────┐
│ Frame 0  │  Past frames  │  Current block  │
│  ████    │  ████████████ │    ████████    │
│  100%    │     100%      │      100%      │
└─────────────────────────────────────────────┘

With bias (kv_cache_attention_bias=0.3):
┌─────────────────────────────────────────────┐
│ Frame 0  │  Past frames  │  Current block  │
│  ████    │  ░░░░░░░░░░░░ │    ████████    │
│  100%    │      30%      │      100%      │
└─────────────────────────────────────────────┘
                 ↑
         Reduced attention weight
```

**Why not bias Frame 0?**
- Frame 0 is the "anchor" - we want to maintain consistency with the start
- Only intermediate frames get downweighted

---

## Implementation: Score Modification

```python
def create_kv_bias_score_mod(log_scale, frame_seqlen, current_block_start):
    """
    Returns a score_mod function for flex_attention / FA4.

    Regions:
    - Region 1: First frame [0, frame_seqlen) → no bias
    - Region 2: Past frames [frame_seqlen, current_block_start) → ADD bias (negative)
    - Region 3: Current block [current_block_start, ...) → no bias
    """
    def score_mod(score, b_idx, h_idx, q_idx, kv_idx):
        is_past_frame = (kv_idx >= frame_seqlen) & (kv_idx < current_block_start)
        return torch.where(is_past_frame, score + log_scale, score)

    return score_mod
```

The `log_scale = log(kv_cache_attention_bias)` is added to attention scores, which is exactly equivalent to scaling `exp(score)` by `kv_cache_attention_bias` for those keys before softmax normalization.

---

## Memory Layout Over Time

```
Call 1 (frames 0-2):
┌──────────────────────────────────────────────────────────────┐
│ [frame0][frame1][frame2][  empty  ][  empty  ][  empty  ]    │
│    ↑                 ↑                                       │
│  sink              local_end_index=3×frame_seqlen            │
└──────────────────────────────────────────────────────────────┘

Call 2 (frames 3-5), after recompute:
┌──────────────────────────────────────────────────────────────┐
│ [frame0*][frame0-2*][frame3][frame4][frame5][  empty  ]      │
│    ↑                                    ↑                    │
│  sink (re-encoded)                  local_end_index          │
└──────────────────────────────────────────────────────────────┘
      * = recomputed from decoded frames

Call 5, after eviction:
┌──────────────────────────────────────────────────────────────┐
│ [frame0*][frame9-11*][frame12][frame13][frame14]             │
│    ↑          ↑                             ↑                │
│  sink   (frames 3-8 evicted)           local_end_index       │
│         kept most recent                                     │
└──────────────────────────────────────────────────────────────┘
```

---

## Tuning Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `kv_cache_num_frames` | 3 | How many frames to include as context during recompute |
| `local_attn_size` | 6 | Local attention window in frames (-1 = global) |
| `kv_cache_attention_bias` | 0.3 | Down-weight factor for past frames (0-1) |
| `sink_size` | (model-dependent) | Frames to never evict (if >0, the earliest frames are “anchors”) |
| `SCOPE_KV_CACHE_RECOMPUTE_EVERY` | 1 | Recompute KV cache every N blocks (higher can be faster but increases drift risk) |

### Tradeoffs

| Higher `kv_cache_num_frames` | More context for quality | More memory, slower recompute |
| Higher `local_attn_size` | More temporal coherence | More memory, quadratic attention |
| Lower `kv_cache_attention_bias` | Less drift from past | May lose temporal consistency |

---

## Related

- **Parent doc:** [`../krea-architecture.md`](../krea-architecture.md)
- **Attention backends:** [`attention-backends.md`](attention-backends.md)
- **Causal masks:** [`causal-attention-masks.md`](causal-attention-masks.md)
- **VAE streaming:** [`vae-streaming.md`](vae-streaming.md)
