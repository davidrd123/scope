# VAE Streaming Decode

> **Wrapper (what the pipeline calls):** `src/scope/core/pipelines/wan2_1/vae/wan.py` (`WanVAEWrapper.encode_to_latent`, `WanVAEWrapper.decode_to_pixel`, `clear_cache`)
> **Streaming internals:** `src/scope/core/pipelines/wan2_1/vae/modules/vae.py` (`WanVAE_.stream_encode`, `WanVAE_.stream_decode`, `CausalConv3d`, `Decoder3d`)
> **Decode block:** `src/scope/core/pipelines/wan2_1/blocks/decode.py` (`DecodeBlock`)
> **Note:** Conceptual explainer — treat the linked implementations as the source of truth for cache layout and first-call behavior.

---

## What Problem Does This Solve?

Standard VAE decode requires the **entire latent sequence** to produce output:

```
Standard VAE:
  [latent_0, latent_1, latent_2, ..., latent_N] → VAE → [frame_0, ..., frame_M]
                                                         ↑
                                                   Must wait for all latents!
```

For streaming video, we want to decode **incrementally**:

```
Streaming VAE:
  [latent_0, latent_1, latent_2] → VAE → [frame_0, frame_1, frame_2, ...]
  [latent_3, latent_4, latent_5] → VAE → [frame_3, frame_4, frame_5, ...]
                                 ↑
                          Process blocks as they arrive
```

The challenge: 3D convolutions look at **neighboring frames**. How do we handle the temporal boundary?

---

## Causal 3D Convolution

The key innovation is **causal padding** - the convolution only looks at past and current frames, never future:

```
Standard 3D Conv (kernel_size=3 in time):
  Output[t] = f(Input[t-1], Input[t], Input[t+1])
                                      ^^^^^^^^
                                      Needs future frame!

Causal 3D Conv:
  Output[t] = f(Input[t-2], Input[t-1], Input[t])
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
              Only past and current!
```

### Implementation (Simplified)

```python
class CausalConv3d(nn.Conv3d):
    """
    3D convolution with causal padding in time dimension.
    """
    def forward(self, x, cache_x=None):
        # x: [B, C, T, H, W]
        time_pad = self._causal_time_pad
        if cache_x is not None and time_pad > 0:
            x = torch.cat([cache_x.to(x.device), x], dim=2)
            time_pad = max(time_pad - cache_x.shape[2], 0)

        # Causal time padding: left-pad only (no future frames)
        if time_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, time_pad, 0))

        # Spatial padding is either implicit (cuDNN) or explicit, depending on config.
        return super().forward(x)
```

---

## Visual: Causal Padding

For a kernel with temporal size 3:

```
Standard padding (non-causal):
  Input:  [    ] [F0] [F1] [F2] [    ]
                  ↓    ↓    ↓
  Output:        [O0] [O1] [O2]

  O1 = f(F0, F1, F2)  ← Sees future frame F2!

Causal padding:
  Input:  [pad] [pad] [F0] [F1] [F2]
                       ↓    ↓    ↓
  Output:             [O0] [O1] [O2]

  O0 = f(pad, pad, F0)  ← Only sees F0
  O1 = f(pad, F0, F1)   ← Only sees F0, F1
  O2 = f(F0, F1, F2)    ← Only sees F0, F1, F2
```

---

## Streaming State (What Actually Persists)

The Krea pipeline keeps a single `components.vae` instance alive across calls. When `use_cache=True`, streaming state lives *inside* the VAE implementation:

- `WanVAE_.first_batch`: toggles special first-call behavior for `stream_encode`/`stream_decode`.
- `WanVAE_._feat_map` / `WanVAE_._enc_feat_map`: per-`CausalConv3d` feature caches (lists).
- `CACHE_T = 2`: how many timesteps to cache per causal conv (kernel size 3 ⇒ cache 2).
- `_conv_idx` / `_enc_conv_idx`: an index that walks the cache list in lockstep with conv usage.

You generally **don’t pass an explicit cache object around**; you reset streaming state via `components.vae.clear_cache()` (e.g., on hard cuts / `init_cache=True`).

---

## Decoder Architecture

The VAE decoder has multiple layers with different temporal resolutions:

```
Decoder3d:
  ├── Initial conv (no upsample)
  │     └── CausalConv3d(kernel=3,3,3)
  │
  ├── Up Block 1 (upsample 2× spatial)
  │     ├── ResBlock with CausalConv3d
  │     └── Upsample2d
  │
  ├── Up Block 2 (upsample 2× spatial + 2× temporal)
  │     ├── ResBlock with CausalConv3d
  │     └── Upsample3d  ← Temporal upsampling!
  │
  ├── Up Block 3 (upsample 2× spatial + 2× temporal)
  │     ├── ResBlock with CausalConv3d
  │     └── Upsample3d
  │
  └── Final conv → RGB output
```

### Temporal Upsampling

```
Input:  [L0] [L1] [L2]        (3 latent frames)
         ↓    ↓    ↓
After 2× upsample (interpolate):
        [F0] [F1] [F2] [F3] [F4] [F5]   (6 output frames)
              ↑              ↑
           Interpolated between L0-L1 and L1-L2
```

The total temporal upsampling is 4× (two 2× upsample layers).

---

## Wrapper API (What Pipeline Calls)

The pipeline uses `WanVAEWrapper` methods:

- `decode_to_pixel(latent, use_cache=True) -> video`
  - `latent`: `[B, T_latent, 16, H_lat, W_lat]`
  - `video`: `[B, T_px, 3, H, W]` in `[-1, 1]`
  - `use_cache=True` dispatches to `WanVAE_.stream_decode(...)` and preserves caches.

- `encode_to_latent(pixel, use_cache=True) -> latent`
  - `pixel`: `[B, 3, T_px, H, W]`
  - `latent`: `[B, T_latent, 16, H_lat, W_lat]`
  - `use_cache=True` uses `WanVAE_.stream_encode(...)`.
  - `use_cache=False` uses a one-off explicit cache (important when re-encoding frames without disturbing streaming state).

---

## Useful Environment Variables

- `WANVAE_STREAM_DECODE_MODE=chunk|loop` (default: `chunk`)
- `WANVAE_DECODE_CHANNELS_LAST_3D=1` / `WANVAE_ENCODE_CHANNELS_LAST_3D=1` (optional layout tweaks)
- `WANVAE_RESAMPLE_ENSURE_CONTIGUOUS=1` (avoid resample slowpaths that create non-contiguous tensors)
- `WANVAE_CONV3D_IMPLICIT_SPATIAL_PADDING=1` (default on): prefer implicit spatial padding in conv3d to reduce `F.pad` overhead

---

## First Call Special Case

`WanVAE_.stream_encode` and `WanVAE_.stream_decode` both have a special path on `first_batch=True`:

- They clear the relevant caches (`clear_cache_encode` / `clear_cache_decode`).
- They run the first timestep separately (to seed causal conv caches) and then process the remaining timesteps.

On the *encode* side for V2V inputs, `PreprocessVideoBlock` also requests an extra frame for the first block (`target_num_frames += 1`) to match the VAE’s streaming temporal structure.

---

## Compression Ratios

| Dimension | Compression | Example |
|-----------|-------------|---------|
| Temporal | 4× | 12 frames → 3 latent frames |
| Spatial (H) | 8× | 320 pixels → 40 latent |
| Spatial (W) | 8× | 576 pixels → 72 latent |
| Channels | 3 → 16 | RGB → 16 latent channels |

Total compression: 12 × 320 × 576 × 3 = 6.6M values → 3 × 40 × 72 × 16 = 138K values (**48× compression**)

---

## Memory Considerations

Streaming decode is more memory-stable than full decode:

```
Full decode (all at once):
  Peak memory grows with total T

Streaming decode (3 latent frames at a time):
  Peak memory is bounded by chunk T (+ fixed caches)
```

The tradeoff: streaming has cache overhead and can't parallelize across the full sequence.

---

## Integration with Pipeline

In the Krea pipeline, decode uses `DecodeBlock` (Wan2.1 shared blocks):

```python
video = components.vae.decode_to_pixel(block_state.latents, use_cache=True)
block_state.output_video = video
```

The streaming cache is internal to `components.vae`. `SetupCachesBlock` resets it on hard cuts (`init_cache=True`) via `components.vae.clear_cache()`.

---

## Debugging Streaming Issues

Common issues and diagnostics:

### 1. Temporal Discontinuity at Block Boundaries

**Symptom:** Visible seam every 12 frames (3 latent frames × 4 upsample)

**Common causes:**
- A hard cut (`init_cache=True`) cleared the VAE cache between blocks.
- Switching decode to `use_cache=False` (intentionally or accidentally) changes boundary behavior.
- Shape changes (resolution / dtype / device) forced reinit in surrounding blocks.

**Debug:**
```python
# Check whether the VAE is treating this as a "first batch"
print("vae.first_batch:", getattr(components.vae.model, "first_batch", None))
```

### 2. First Block Looks Different

**Symptom:** First 12 frames have different quality/style

**Cause:** `first_batch=True` first-call path (cache seeding) plus different upstream noise/control flow

**Debug:**
```python
# Compare first vs steady-state
print(f"First call latents shape: {block_state.latents.shape}")
print(f"First call output shape: {block_state.output_video.shape}")
```

### 3. Memory Growth Over Time

**Symptom:** VRAM usage increases with each block

**Cause:** Usually not the VAE feature caches (they are fixed-size); more often upstream activations or accumulation outside the VAE.

**Debug tip:** for VAE-only sanity, set `WANVAE_STREAM_DECODE_MODE=loop` and confirm outputs match the default chunked path (they should be equivalent).

---

## Related

- **Encode path:** Similar structure but in reverse (causal conv caches for encode)
- **V2V:** Uses `encode_to_latent` before decode
- **KV cache recompute:** [`kv-cache-mechanics.md`](kv-cache-mechanics.md)
- **Parent doc:** [`../krea-architecture.md`](../krea-architecture.md)
