# Frame Processor Routing

> **Location:** `src/scope/server/frame_processor.py`
> **Class:** `FrameProcessor`

---

## The Journey of a Frame

```
WebRTC Track                    FrameProcessor                      Pipeline
     │                               │                                  │
     │  VideoFrame (yuv420p)         │                                  │
     ├──────────────────────────────►│                                  │
     │                               │  .put() → frame_buffer           │
     │                               │                                  │
     │                               │  [F0, F1, F2, F3, F4, F5, ...]   │
     │                               │                                  │
     │                               │  .prepare_chunk(12)              │
     │                               │  ├── uniform sample              │
     │                               │  └── [F0, F2, F4, F6, ...]       │
     │                               │                                  │
     │                               │  Route based on vace_enabled:    │
     │                               │  ├── False → call_params["video"]│
     │                               │  └── True  → call_params["vace_input_frames"]
     │                               │                                  │
     │                               │          list[Tensor]            │
     │                               ├─────────────────────────────────►│
     │                               │                                  │
```

---

## Step 1: Frame Arrival (`.put()`)

WebRTC delivers frames via `VideoProcessingTrack`:

```python
# src/scope/server/tracks.py:51
class VideoProcessingTrack:
    async def input_loop(self):
        while self.input_task_running:
            input_frame = await self.track.recv()
            self.frame_processor.put(input_frame)  # ◄── Frames arrive here
```

The `put()` method appends to a thread-safe buffer:

```python
# src/scope/server/frame_processor.py:374
def put(self, frame: VideoFrame) -> bool:
    """Add frame to buffer for later processing."""
    with self.frame_buffer_lock:
        self.frame_buffer.append(frame)
        return True
```

**Frame format at this stage:** `VideoFrame` (av library), typically YUV420p.

---

## Step 2: Mode Detection

When the stream starts, mode is set from `initialParameters`:

```python
# src/scope/server/frame_processor.py:219
self._video_mode = (initial_parameters or {}).get("input_mode") == "video"
```

This determines:
- How many input frames to request
- Whether to call `prepare_chunk()`
- Where to route the frames

---

## Step 3: Uniform Sampling (`.prepare_chunk()`)

The pipeline requests N frames per block. But the buffer may have more (or fewer) frames depending on timing.

**Solution:** Uniform sampling across the entire buffer.

```python
# src/scope/server/frame_processor.py:1612
def prepare_chunk(self, chunk_size: int) -> list[torch.Tensor]:
    """
    Sample frames uniformly from the buffer.

    Example with buffer_len=8, chunk_size=4:
      step = 8/4 = 2.0
      indices = [0, 2, 4, 6]  (uniformly distributed)
      Returns frames at positions 0, 2, 4, 6
      Removes frames 0-6 from buffer
    """
    step = len(self.frame_buffer) / chunk_size
    indices = [round(i * step) for i in range(chunk_size)]

    video_frames = [self.frame_buffer[i] for i in indices]

    # Drop all frames up to and including last sampled
    last_idx = indices[-1]
    for _ in range(last_idx + 1):
        self.frame_buffer.popleft()

    # Convert VideoFrame → Tensor (1, H, W, C)
    tensor_frames = []
    for vf in video_frames:
        tensor = torch.from_numpy(
            vf.to_ndarray(format="rgb24")
        ).float().unsqueeze(0)
        tensor_frames.append(tensor)

    return tensor_frames
```

### Visual: Uniform Sampling

```
Buffer: [F0, F1, F2, F3, F4, F5, F6, F7]  (8 frames)
Request: chunk_size=4

step = 8/4 = 2.0
indices = [0, 2, 4, 6]

Sampled:  [F0,     F2,     F4,     F6    ]
           ↑       ↑       ↑       ↑
           0       2       4       6

After sampling:
Buffer: [F7]  (frames 0-6 removed)
```

---

## Step 4: Routing Decision

The key branch that determines V2V mechanism:

```python
# src/scope/server/frame_processor.py:1348
if video_input is not None:
    vace_enabled = getattr(pipeline, "vace_enabled", False)
    if vace_enabled:
        # VACE path: frames become conditioning
        call_params["vace_input_frames"] = video_input
    else:
        # Latent-init path: frames become init latents
        call_params["video"] = video_input
```

**This is the fork in the road** between Mechanism A (latent-init) and Mechanism B (VACE).

---

## The Cadence Problem

### Why Uniform Sampling?

WebRTC frames arrive at variable rates (network jitter, camera timing). The pipeline needs exactly N frames per block.

FrameProcessor effectively does **both**:
1. **Wait until at least N frames are available** (it sleeps/retries if the buffer is short)
2. **If the buffer has > N frames**, uniformly sample N frames across the buffer and drop frames up to the last sampled index

Uniform sampling ensures temporal coverage but introduces **variable stride >= 1.0** (i.e., skipped frames) when the buffer grows:

```
Scenario A: Buffer has 12 frames, need 12
  stride = 1.0 → sample every frame

Scenario B: Buffer has 18 frames, need 12
  stride = 1.5 → sample ~every 1.5 frames (some skipped)
```

### The Jitter Consequence

If generation is slightly slower than capture:
- Buffer grows → stride increases → temporal aliasing
- Motion can look "jumpy" because frame spacing isn't constant

If generation is slightly faster than capture:
- FrameProcessor blocks waiting for frames (it does not sample with `stride < 1.0`)
- Output can appear low-FPS / “frozen” simply because there are no new input frames to drive the next chunk

---

## Frame Count Requirements

The pipeline specifies how many frames it needs via `prepare()`:

```python
# src/scope/core/pipelines/defaults.py
if kwargs.get("video") is not None:
    if video_input_size is None:
        video_input_size = calculate_video_input_size(components_config)
    return Requirements(input_size=video_input_size)
```

**Why 12 frames?**
- `num_frame_per_block = 3` (latent frames to generate)
- `vae_temporal_downsample_factor = 4` (VAE compression)
- 3 × 4 = 12 input frames → 3 latent frames

**Important:** this requirement does *not* account for the VAE’s first-batch `(1 + 4k)` behavior (see next section). In practice, the first block will be resampled to 13 frames inside the pipeline even though the processor typically supplies 12.

---

## First-Block Special Case

The first block needs an extra frame:

```python
# src/scope/core/pipelines/wan2_1/blocks/preprocess_video.py:55
if current_start_frame == 0:
    # First block: need 1 extra frame for VAE alignment
    target_num_frames = num_frame_per_block * vae_downsample + 1
    # = 3 * 4 + 1 = 13 frames
```

This is because the VAE expects `(1 + 4k)` frames on the first encode call.

### The Edge Case

Because `prepare()` requests 12 frames, the FrameProcessor will typically provide 12 frames for the first block. But `PreprocessVideoBlock` targets 13 frames on the first block, so it resamples/duplicates to hit 13. This can introduce a subtle temporal artifact on the first block.

---

## Tensor Format Through the Pipeline

| Stage | Format | Shape |
|-------|--------|-------|
| `VideoFrame` (av) | YUV420p | varies |
| After `to_ndarray("rgb24")` | uint8 RGB | `(H, W, 3)` |
| After `torch.from_numpy().float()` | float32 | `(H, W, 3)` |
| After `.unsqueeze(0)` | float32 | `(1, H, W, 3)` |
| `prepare_chunk` returns | `list[Tensor]` | `[(1,H,W,3), ...]` |
| After `PreprocessVideoBlock` | float32 | `(B, C, T, H, W)` |
| After VAE encode | float32/bf16 | `(B, F, C, H/8, W/8)` (VAE returns BFCHW) |

---

## Debugging Frame Flow

### Log buffer sizes:
```python
# Add to prepare_chunk:
print(f"Buffer: {len(self.frame_buffer)}, requested: {chunk_size}")
print(f"Stride: {len(self.frame_buffer) / chunk_size:.2f}")
print(f"Indices: {indices}")
```

### Check for temporal jitter:
```python
# Log time between sampled frames
times = [vf.time for vf in video_frames]
deltas = [t2 - t1 for t1, t2 in zip(times[:-1], times[1:])]
print(f"Frame deltas: {deltas}")  # Should be ~constant
```

### Verify routing:
```python
# Add before pipeline call:
vace_enabled = getattr(pipeline, "vace_enabled", False)
print(f"vace_enabled: {vace_enabled}")
print(f"call_params keys: {list(call_params.keys())}")
# Should see either "video" or "vace_input_frames", not both
```

---

## Common Issues

### 1. "Video mode but no frames"

**Symptom:** Pipeline runs but `video` kwarg is None

**Cause:** Buffer empty when `prepare_chunk` called

**Debug:** Check frame arrival rate vs generation rate

### 2. "First block looks different"

**Symptom:** First chunk looks different from steady-state output

**Cause:** First-block +1 frame adjustment + fresh caches

**Debug:** Compare `output.shape` (and/or latent frame counts) between first and subsequent blocks

### 3. "Motion looks wrong speed"

**Symptom:** Generated motion faster/slower than input

**Cause:** Stride mismatch between input FPS and generation FPS

**Debug:** Log the stride in `prepare_chunk`, should be ~1.0

---

## Related

- **Latent mixing:** [`latent-noise-mixing.md`](latent-noise-mixing.md)
- **V2V overview:** [`../v2v-mechanisms.md`](../v2v-mechanisms.md)
- **VAE streaming:** [`../krea-architecture/vae-streaming.md`](../krea-architecture/vae-streaming.md)
