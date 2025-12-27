# Video-to-Video Mechanisms in Scope

> **Date:** 2025-12-27
> **Status:** Draft (conceptual; verify against code)
> **Applies to:** All Wan2.1-based pipelines (1.3B and 14B)

---

## Executive Summary

Scope has two distinct ways to use an incoming video stream:

1. **Latent-init V2V ("normal V2V")**: Encode input frames to VAE latents, mix in noise, then denoise. This is what the 1.3B realtime pipelines do in video mode.

2. **VACE-conditioned "V2V editing"**: Generate from noise (T2V latent path) but inject a VACE context encoded from input frames (and/or ref images + masks) during denoising.

**Krea realtime 14B** already supports the first path (latent-init V2V) in its pipeline design. What it's missing today is the second path (VACE) for 14B (wiring + artifacts + load-time toggle). See `notes/proposals/vace-14b-integration.md` for the implementation plan.

---

## Performance Gotchas (Why V2V Can “Look Slow”)

Even when the model is fast, V2V can appear low-FPS in the UI for reasons that are not core denoise/decode throughput:

1. **Output FPS is capped by input FPS.** The server deliberately paces output at `min(input_fps, pipeline_fps)` for temporal correctness (see `FrameProcessor.get_output_fps()` in `src/scope/server/frame_processor.py`).
2. **First block frame-count special case.** In video mode, `PreprocessVideoBlock` will target `num_frame_per_block * vae_temporal_downsample_factor + 1` frames for the first block (`current_start_frame == 0`) (see `src/scope/core/pipelines/wan2_1/blocks/preprocess_video.py`). If fewer frames are available, it will resample/duplicate to hit the target.

For a compute-only baseline (to separate “pipeline throughput” from “input rate / buffering”), see the V2V triage experiment card in `notes/FA4/b300/experiments.md`.

---

## Mental Model: "Input Mode" vs "Pipeline Mode"

There are two "modes" that can be easy to conflate:

### Input Mode (Control-Plane)
Frontend decides whether you're feeding a video stream into the backend.

- Frontend sends `initialParameters.input_mode: "video"` when you choose video mode (`frontend/src/pages/StreamPage.tsx:1088`)
- Backend `FrameProcessor` stores `_video_mode` based on that (`src/scope/server/frame_processor.py:209`)

### Pipeline Mode (Data-Plane)
Wan2.1 pipelines infer "video vs text" based on whether `video` is present in kwargs:

- `resolve_input_mode(kwargs)` returns `"video"` only if `kwargs.get("video") is not None` (`src/scope/core/pipelines/defaults.py:22`)

**Key distinction:** VACE uses video input **without** passing it as `video`, so the pipeline stays in "text mode" while still consuming video frames.

---

## Common Top-to-Bottom Plumbing (Both Mechanisms)

### 1. Frontend Starts a Stream
- Chooses `inputMode` (`frontend/src/pages/StreamPage.tsx:1075`) and sends:
  - `initialParameters.input_mode` to backend
  - In video mode, also sends `noise_scale` / `noise_controller` in `initialParameters`

### 2. WebRTC Track + FrameProcessor Lifecycle
- A `VideoProcessingTrack` owns a `FrameProcessor` (`src/scope/server/tracks.py:18`)
- Incoming WebRTC frames are fed into the processor via `FrameProcessor.put()` (`src/scope/server/tracks.py:51`)
- A worker thread runs `FrameProcessor.process_chunk()` in a loop (`src/scope/server/frame_processor.py:759`)

### 3. How FrameProcessor Decides It Needs N Input Frames
- If the active pipeline implements `prepare()`, FrameProcessor calls it each chunk (`src/scope/server/frame_processor.py:1314`)
- In input video mode, FrameProcessor pre-signals the pipeline with `prepare_params["video"] = True`
- Most realtime pipelines implement `prepare()` via `prepare_for_mode(...)`, which returns `Requirements(input_size=...)` in video mode

### 4. Frame Collection + Shape
- `FrameProcessor.prepare_chunk(chunk_size)` samples `chunk_size` frames uniformly from the buffer
- Converts each `VideoFrame` to a tensor shaped `(1, H, W, C)` float32 (`src/scope/server/frame_processor.py:1612`)
- The result passed into pipelines is typically a `list[torch.Tensor]` of those `(1,H,W,C)` frames

### 5. Routing to One of the Two V2V Mechanisms
FrameProcessor checks `pipeline.vace_enabled` and routes the buffered frames accordingly (`src/scope/server/frame_processor.py:1348`):

```python
if pipeline.vace_enabled:
    call_params["vace_input_frames"] = video_input  # VACE path
else:
    call_params["video"] = video_input              # Latent-init V2V path
```

**That single branch is the "fork in the road."**

---

## Mechanism A: Latent-Init V2V

*Video → VAE latents → mix noise → denoise*

This is the "classic" video2video approach, implemented generically for Wan2.1 pipelines.

### A.1 Entry Condition
Pipeline is called with a non-None `video` kwarg (FrameProcessor sets it when `vace_enabled` is false).

### A.2 Pipeline Requests Frames
`prepare_for_mode(...)` returns an input-size requirement of `num_frame_per_block * vae_temporal_downsample_factor` (`src/scope/core/pipelines/defaults.py:120`).

For typical configs: `3 * 4 = 12` raw frames per chunk.

### A.3 Video Preprocessing (CPU frames → BCTHW on GPU, normalized)

In the modular block stack, `AutoPreprocessVideoBlock` routes based on the presence of `video`:

1. `PreprocessVideoBlock` converts the `list[(1,H,W,C)]` frames into a single tensor `video: [B, C, F, H, W]` on the pipeline device/dtype, resizes to target resolution, and normalizes to `[-1, 1]`
2. `NoiseScaleControllerBlock` optionally adjusts `noise_scale` based on motion, and can trigger cache resets when noise scale changes

### A.4 Cache Setup
`SetupCachesBlock` sets/refreshes:
- KV cache size + attention window parameters
- Cross-attention cache
- Clears VAE cache when doing a hard reset

### A.5 Convert Video → Latents (Core of "Normal V2V")

`AutoPrepareLatentsBlock` chooses `PrepareVideoLatentsBlock` when `video` exists:

```python
# PrepareVideoLatentsBlock (src/scope/core/pipelines/wan2_1/blocks/prepare_video_latents.py:16)
latents = components.vae.encode_to_latent(video)
noise = torch.randn_like(latents)
noisy_latents = noise * noise_scale + latents * (1 - noise_scale)
```

This is the mechanical equivalent of "init from video latents then add noise" you see in many ComfyUI workflows.

### A.6 Denoising + Decode
- `DenoiseBlock` runs the iterative denoising loop, and (if `noise_scale` is set) adjusts the denoise schedule start
- `DecodeBlock` turns denoised latents into pixel frames via the streaming VAE decode

### A.7 What Knobs Matter Here

| Knob | Effect |
|------|--------|
| `noise_scale` | "How strongly to preserve input" (lower preserves more) |
| `noise_controller` | Motion-aware auto-adjust for noise_scale |
| Cache knobs | `manage_cache`, hard cuts / `init_cache` — V2V uses caches heavily for temporal stability |

---

## Mechanism B: VACE-Conditioned "V2V Editing"

*Video → VACE context → guide denoise; latents from noise*

This is **not** "init-from-video"; it's "generate-from-noise but guided by video/ref images/masks".

### B.1 Entry Condition
- The pipeline must be loaded with VACE enabled, so `pipeline.vace_enabled == True` (set at init-time by the VACE mixin)
- FrameProcessor then routes incoming frames to `vace_input_frames` instead of `video`

### B.2 Why the Pipeline Stays in "Text Mode"
`resolve_input_mode()` only checks for `video`, not `vace_input_frames`. So when you feed `vace_input_frames` without `video`, the pipeline:
- Still preprocesses video (because blocks look at `vace_input_frames`)
- But does **not** take the "PrepareVideoLatentsBlock" route

This is intentional: VACE wants the model to denoise fresh latents while injecting conditioning.

### B.3 Preprocess Step Still Happens
`AutoPreprocessVideoBlock` has triggers for `vace_input_frames`. It runs `PreprocessVideoBlock`, which preprocesses `vace_input_frames` into `[B, C, F, H, W]`.

### B.4 Latents Come From Noise (Not Video)
Because `video` is absent, `AutoPrepareLatentsBlock` chooses `PrepareLatentsBlock` — so the denoiser starts from random latents like T2V.

### B.5 VACE Context Encoding (Core of This Mechanism)

Pipelines that support VACE include `VaceEncodingBlock` in their block list. It runs when either:
- `vace_ref_images` is provided (reference-only mode), or
- `vace_input_frames` is provided (conditioning mode)

In conditioning mode:
1. `VaceEncodingBlock` encodes frames through the VAE using `vace_encode_frames`
2. Encodes masks with `vace_encode_masks`
3. Concatenates them with `vace_latent`

The end product is a **96-channel VACE context** (conceptually: latent features + mask features) that is passed into denoising.

### B.6 Denoising Uses vace_context
- `DenoiseBlock` passes `vace_context` into the generator each step
- The generator forwards it down to the model call
- The underlying model is wrapped with `CausalVaceWanModel` when VACE is enabled at load time

### B.7 What Knobs Matter Here

| Knob | Effect |
|------|--------|
| `vace_ref_images` | Reference images for conditioning |
| `vace_input_frames` | Per-chunk video conditioning |
| `vace_input_masks` | Mask for selective conditioning |
| `vace_context_scale` | Conditioning strength |

**Note:** "Normal V2V" knobs like `noise_scale` generally do not apply in the same way, because this path doesn't initialize latents from video.

---

## Where Krea 14B Fits

### Krea 14B Does Have Latent-Init V2V in the Codebase
- Krea pipeline declares support for "video" mode (`src/scope/core/pipelines/schema.py:322`)
- Its modular blocks include the same `AutoPreprocessVideoBlock` and `AutoPrepareLatentsBlock` routing
- FrameProcessor will route to `video` as long as `pipeline.vace_enabled` is false

**Mechanically, Krea can do the same "1.3B-style V2V" path.**

### What Krea 14B Is Missing Today Is VACE
- Krea load params do not include `vace_enabled` (`src/scope/server/schema.py:352`)
- `PipelineManager` never calls `_configure_vace()` for `krea-realtime-video`
- `_get_vace_checkpoint_path()` is hardcoded to the 1.3B module
- Krea artifacts don't include any VACE module downloads
- Krea pipeline doesn't use the VACE mixin nor include `VaceEncodingBlock`

**This is exactly what `notes/proposals/vace-14b-integration.md` is about.**

---

## Comparison: Scope vs ComfyUI-WanVideoWrapper (Why ComfyUI Feels “Battle Tested”)

ComfyUI-WanVideoWrapper is optimized around **offline / batch** sampling where you have the whole video (or at least a stable window) available. Scope is optimized around **realtime streaming** where:

- Input frames arrive asynchronously over WebRTC
- Generation runs in small blocks
- Temporal coherence depends on KV-cache + VAE streaming caches

Those constraints make “V2V that looks stable” significantly harder in Scope, even if the core idea (init latents + denoise) is the same.

### ComfyUI “classic V2V” = init latents + `denoise_strength` (timestep-aligned)

The core V2V path in ComfyUI is:

1. **Encode input frames to latents** (the init for v2v)
   - `WanVideoEncode` encodes a batch of frames to VAE latents and returns a `LATENT` dict with `"samples"`:
     - `/root/ComfyUI-WanVideoWrapper/nodes.py:2154` (`WanVideoEncode.encode`)

2. **Sampler starts from those latents and adds noise at the correct timestep**
   - `WanVideoSampler` accepts optional `samples` (“init Latents to use for video2video process”) and `denoise_strength`
   - When `denoise_strength < 1.0`, it chooses a start step and enables `add_noise_to_samples`, then mixes init latents with noise as a function of the timestep:
     - `/root/ComfyUI-WanVideoWrapper/nodes_sampler.py:718` (reads `samples["samples"]`)
     - `/root/ComfyUI-WanVideoWrapper/nodes_sampler.py:728` (mixes init latents with noise based on `latent_timestep`)

That “noise aligned to the timestep where sampling begins” is a big part of why ComfyUI V2V feels predictable and is easier to tune with a single knob.

### Scope “classic V2V” = init latents + `noise_scale` (not timestep-aligned)

Scope’s latent-init V2V is mechanically similar (encode → mix noise → denoise), but the mixing is driven by a simpler knob:

- `PrepareVideoLatentsBlock` always creates fresh Gaussian noise and mixes with a fixed `noise_scale`:
  - `src/scope/core/pipelines/wan2_1/blocks/prepare_video_latents.py:16`
- `DenoiseBlock` then runs a discrete `denoising_step_list` and adjusts *only the first step* when `noise_scale` is present:
  - `src/scope/core/pipelines/wan2_1/blocks/denoise.py:283`

This works, but it’s not the same contract as “denoise_strength + scheduler start step”, and it hasn’t had the same amount of tuning iteration.

**Important footgun:** because only the *first* timestep is mutated, short step lists can become **non-monotonic** (e.g. a 2-step list like `[1000, 750]` can become `[600, 750]` at `noise_scale=0.7`). If you’re seeing “glitchy” behavior, it’s worth validating that your effective `denoising_step_list` stays in the intended order for the scheduler (see `src/scope/core/pipelines/wan2_1/blocks/denoise.py:283`).

### ComfyUI also stacks additional stabilization tools that Scope doesn’t (yet)

Depending on workflow, ComfyUI wrapper commonly layers in:

- Context windows + overlap (for long sequences)
- Masking (noise masks / conditioning masks)
- Noise augmentation on the reference video
- Color matching between windows
- Extra “first frame / end frame” special casing for some modes

Scope’s realtime pipelines intentionally keep per-chunk work small, and rely on caching for continuity instead of window stitching.

---

## Why Scope V2V Can Look “Glitchy” (Most Likely Code-Level Causes)

These are the highest-suspicion areas where Scope differs from “battle tested” V2V implementations and can plausibly create visible artifacts (jitter, cadence weirdness, sudden discontinuities).

### 1) Input cadence: uniform sampling across a variable-length buffer

`FrameProcessor.prepare_chunk()` does **uniform sampling across the entire current buffer**, then drops everything up to the last sampled frame:

- `src/scope/server/frame_processor.py:1612`

If capture FPS and generation FPS aren’t perfectly matched, buffer length fluctuates → the effective stride between sampled frames fluctuates → **temporal jitter / aliasing** in the V2V input.

This is great for “keep latency bounded” but not great for “V2V looks like it’s tracking the real input video”.

### 2) First-batch VAE semantics: requires `1 + 4k` input frames and can yield an extra latent

WanVAE streaming encode has a special first-batch path:

- `src/scope/core/pipelines/wan2_1/vae/modules/vae.py:821` (`stream_encode`)

This is why `PreprocessVideoBlock` adds an extra frame only for the first block:

- `src/scope/core/pipelines/wan2_1/blocks/preprocess_video.py:55`

Consequence: the first encode in a V2V sequence can yield **one more latent frame** than steady state.

There is precedent that this creates a V2V-only edge case:

- RewardForcing’s pipeline has a custom `PrepareNextBlock` that subtracts 1 frame only for the first V2V block (“we do not understand why this is necessary yet!”):
  - `src/scope/core/pipelines/reward_forcing/blocks/prepare_next.py:1`

If other pipelines hit the same underlying behavior but don’t have the adjustment, you can end up with **off-by-one frame indexing** in `current_start_frame`, which can manifest as cadence glitches or cache drift.

### 3) Cache resets mid-stream can look like hard cuts

In video mode, Scope can reset KV/VAE caches when:

- `init_cache=True` is requested (hard cut / first call / mode transitions), or
- conditioning changes (outside transitions) and `manage_cache=True`
  - `src/scope/core/pipelines/wan2_1/blocks/setup_caches.py:63`

If this triggers frequently (e.g., prompt edits), it can look like discontinuities even if the intent was “keep the stream continuous”.

### 4) Resampling-by-rounding can duplicate or skip frames

`preprocess_video()` resamples to `target_num_frames` using `linspace(...).round().long()` indexing:

- `src/scope/core/pipelines/wan2_1/blocks/preprocess_video.py:72`

That’s a pragmatic way to “always have enough frames”, but it can introduce repeated frames, especially around the first-block +1 behavior.

---

## Practical Debug Checklist (If You Want to Confirm This Quickly)

If you want to validate whether the above matches what you’re seeing, the fastest checks are:

1. **Is input cadence stable?**
   - Log the indices / time deltas between frames chosen by `prepare_chunk()` and see if stride changes chunk-to-chunk.

2. **Does the first V2V chunk produce a different number of output frames than steady state?**
   - Compare `output.shape[0]` from the pipeline in video mode on the first chunk vs subsequent chunks.

3. **Do prompt changes trigger a visible discontinuity?**
   - Correlate cache reset logs (hard cuts) with visible glitches.

If you want, I can add lightweight instrumentation (guarded by an env var) to log per-chunk frame counts + `current_start_frame` + cache-reset reasons so we can get a concrete “glitch signature” before changing behavior.

## Comparison Table

| Aspect | Latent-Init V2V (Mechanism A) | VACE Conditioning (Mechanism B) |
|--------|-------------------------------|--------------------------------|
| **Entry signal** | `video` kwarg non-None | `vace_input_frames` kwarg |
| **Latent source** | VAE-encoded input video | Random noise (like T2V) |
| **How input affects output** | Directly sets init latents | Guides attention via conditioning |
| **Key knob** | `noise_scale` | `vace_context_scale` |
| **Semantic** | "Re-render this video" | "Follow this motion/structure" |
| **Krea 14B support** | Yes (exists today) | No (needs VACE-14B integration) |

---

## When to Use Which

| Use Case | Best Mechanism |
|----------|----------------|
| "Re-render this video in anime style" | **Latent-Init V2V** (structure from init latents) |
| "Generate a robot walking, following this motion reference" | **VACE** (structure from conditioning, content from text) |
| "Style transfer with high fidelity to original" | **Latent-Init V2V** + low `noise_scale` |
| "Motion transfer to completely different subject" | **VACE** |
| "Smooth transitions between prompts" | **Latent-Init V2V** (uses KV cache for temporal coherence) |

---

## Related Documentation

| Document | Contents |
|----------|----------|
| `notes/guides/krea-architecture.md` | Full Krea pipeline architecture |
| `notes/guides/vace-architecture-explainer.md` | How VACE works in Scope |
| `notes/proposals/vace-14b-integration.md` | Implementation plan for VACE on 14B |
| `src/scope/core/pipelines/wan2_1/blocks/prepare_video_latents.py` | Latent-init V2V implementation |
| `src/scope/core/pipelines/wan2_1/vace/blocks/vace_encoding.py` | VACE encoding implementation |
