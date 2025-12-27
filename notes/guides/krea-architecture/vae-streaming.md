# VAE Streaming Decode

> **Location:** `src/scope/core/pipelines/krea_realtime_video/modules/vae.py`
> **Classes:** `WanVAE`, `CausalConv3d`, `Decoder3d`
> **Note:** Conceptual explainer — for exact cache sizes/padding behavior, follow the linked implementation.

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

### Implementation

```python
class CausalConv3d(nn.Conv3d):
    """
    3D convolution with causal padding in time dimension.
    """
    def __init__(self, in_channels, out_channels, kernel_size, ...):
        super().__init__(in_channels, out_channels, kernel_size, ...)

        # Padding for (D, H, W)
        # D (depth/time): pad only on the left (past)
        # H, W: symmetric padding
        self._padding = (
            0, kernel_size[2] // 2,  # W: symmetric
            0, kernel_size[1] // 2,  # H: symmetric
            kernel_size[0] - 1, 0,   # D: left only (causal)
        )

    def forward(self, x, cache=None):
        """
        Args:
            x: [B, C, T, H, W] - input tensor
            cache: [B, C, T_cache, H, W] - cached frames from previous call

        Returns:
            output, new_cache
        """
        if cache is not None:
            # Prepend cached frames
            x = torch.cat([cache, x], dim=2)

        # Apply causal padding
        x = F.pad(x, self._padding)

        # Standard conv (no padding in forward, we did it manually)
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

## Streaming State (Cache)

For streaming, we need to remember the last few frames across calls:

```python
class StreamingVAEState:
    """Cache for streaming decode."""
    def __init__(self):
        self.conv_caches = {}  # Per-layer conv caches
        self.first_call = True

    def get_cache(self, layer_name):
        return self.conv_caches.get(layer_name)

    def set_cache(self, layer_name, cache):
        self.conv_caches[layer_name] = cache
```

### How Much to Cache?

For a kernel with temporal size K, we need to cache K-1 frames:

```
Kernel size 3: cache 2 frames
  Call 1: [pad, pad, F0, F1, F2] → process → cache [F1, F2]
  Call 2: [F1, F2, F3, F4, F5]   → process → cache [F4, F5]
```

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

## Stream Decode Implementation

```python
class WanVAE:
    def stream_decode(self, latents, state=None):
        """
        Decode latents to video frames, supporting streaming.

        Args:
            latents: [B, C, T, H, W] - latent frames to decode
            state: StreamingVAEState - cache from previous call

        Returns:
            frames: [B, 3, T*4, H*8, W*8] - decoded video
            state: Updated streaming state
        """
        if state is None:
            state = StreamingVAEState()

        x = latents

        # Initial conv
        cache = state.get_cache("conv_in")
        x = self.conv_in(x, cache=cache)
        state.set_cache("conv_in", x[:, :, -2:])  # Cache last 2 frames

        # Up blocks
        for i, block in enumerate(self.up_blocks):
            # ResBlock convs
            cache = state.get_cache(f"block_{i}")
            x = block.resblock(x, cache=cache)
            state.set_cache(f"block_{i}", x[:, :, -2:])

            # Upsample (spatial and/or temporal)
            x = block.upsample(x)

        # Final conv
        cache = state.get_cache("conv_out")
        x = self.conv_out(x, cache=cache)
        state.set_cache("conv_out", x[:, :, -2:])

        return x, state
```

---

## First Call Special Case

The first decode call needs special handling:

```python
def stream_decode(self, latents, state=None):
    if state is None or state.first_call:
        # First call: need extra input frame for proper alignment
        # This is because the VAE expects (1 + 4k) frames on first encode
        # and we need to match that structure on decode

        # Pad with zeros or repeat first frame
        if latents.shape[2] == num_latent_frames:
            latents = torch.cat([
                latents[:, :, :1],  # Repeat first frame
                latents
            ], dim=2)

        state.first_call = False
```

This is why `PreprocessVideoBlock` adds an extra frame on the first call.

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

Streaming decode is more memory-efficient than full decode:

```
Full decode (all at once):
  Peak memory = O(max(all_intermediate_activations))
  For 81 frames: ~8GB activation memory

Streaming decode (3 latent frames at a time):
  Peak memory = O(max(block_intermediate_activations))
  For 3 latent frames: ~1GB activation memory
```

The tradeoff: streaming has cache overhead and can't parallelize across the full sequence.

---

## Integration with Pipeline

In the Krea pipeline, streaming decode is used via `DecodeBlock`:

```python
class DecodeBlock:
    def __call__(self, components, state):
        latents = state.get("latents")
        vae_state = state.get("vae_decode_state")

        # Stream decode
        frames, vae_state = components.vae.stream_decode(
            latents,
            state=vae_state
        )

        state.set("output_video", frames)
        state.set("vae_decode_state", vae_state)

        return components, state
```

The `vae_decode_state` persists across pipeline calls, maintaining temporal continuity.

---

## Debugging Streaming Issues

Common issues and diagnostics:

### 1. Temporal Discontinuity at Block Boundaries

**Symptom:** Visible seam every 12 frames (3 latent frames × 4 upsample)

**Cause:** Cache not being passed correctly

**Debug:**
```python
# Log cache sizes
for name, cache in state.conv_caches.items():
    print(f"{name}: {cache.shape if cache is not None else 'None'}")
```

### 2. First Block Looks Different

**Symptom:** First 12 frames have different quality/style

**Cause:** First-call padding behavior

**Debug:**
```python
# Compare first vs steady-state
print(f"First call latents shape: {latents.shape}")
print(f"First call output shape: {output.shape}")
```

### 3. Memory Growth Over Time

**Symptom:** VRAM usage increases with each block

**Cause:** Caches not being trimmed

**Debug:**
```python
# Check cache sizes aren't growing
initial_size = sum(c.numel() for c in state.conv_caches.values())
# ... run several blocks ...
final_size = sum(c.numel() for c in state.conv_caches.values())
assert initial_size == final_size, "Cache is growing!"
```

---

## Related

- **Encode path:** Similar structure but in reverse (causal conv caches for encode)
- **V2V:** Uses `encode_to_latent` before decode
- **Parent doc:** [`../krea-architecture.md`](../krea-architecture.md)
