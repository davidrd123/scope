# Offline Timeline Rendering (Krea Realtime) + Transition Smoothing

This is a detailed engineering report describing how Scope can take a Timeline JSON export (e.g. `timeline-YYYY-MM-DD.json`) and render it offline at higher resolution / higher quality, with prompt continuity and smooth transitions.

It’s written as “what exists in this codebase, what we needed, what we changed, and why those changes map to the behavior you want”.

## 1) Goal / Problem Statement

You want:
- **Offline** rendering (non-real-time is fine).
- **Higher quality** than the interactive realtime stream, mainly via:
  - higher resolution
  - more inference steps (the Krea model card suggests `num_inference_steps=6` for offline quality)
  - optional cache/consistency tuning (KV cache bias, KV cache recomputation window)
- To drive the render from a Scope **Timeline JSON** (prompt program).
- To preserve “single-take continuity” (identity/style/motion) while prompts change over time.

## 2) Key Constraints / Observations

### 2.1 Timeline JSON time semantics
Scope’s Timeline stores `startTime`/`endTime` as **seconds**. A timeline is essentially a “prompt schedule over time”.

For offline rendering, you need to convert time → frame index:
- `frame_idx = round(time_seconds * fps)`
- total frames ≈ `ceil(duration_seconds * fps)`

Important: if you set `--fps` higher, you are asking the model to **generate more frames** for the same duration (more compute), not just re-encode the same frames at a different playback speed.

### 2.2 Krea pipeline output granularity
The Scope-integrated Krea pipeline is autoregressive and produces video in fixed “blocks”:
- In `src/scope/core/pipelines/krea_realtime_video/model.yaml`, `num_frame_per_block: 3`.
- Each pipeline call produces **3 frames** (unless it’s the very first block and VAE streaming behavior forces special-casing at the latent level; in practice, output comes out as a short chunk).

So the offline renderer needs to “loop” pipeline calls until enough frames are generated.

### 2.3 Prompt continuity strategy (“anchor clause”)
The “anchor clause” pattern is not a special timeline feature; it’s a prompting strategy:
- Include an anchor (subject identity + styling + scene constants) in **every** prompt text.
- Then append the changing “action / beat / camera / mood” clauses per segment.

This is compatible with the timeline format as-is: the anchor is just repeated text.

Concrete template (recommended):
```text
ANCHOR: [identity + outfit + environment + lens + lighting + style tags].
BEAT: [what changes this segment].
```

Example across two timeline segments:
```text
ANCHOR: a handcrafted stop-motion puppet yeti, navy scarf, dawn-lit miniature set, 35mm, tungsten practicals, Rankin/Bass animagic.
BEAT: title card, static wide shot, “THE DAWN OF MAN”.
```
```text
ANCHOR: a handcrafted stop-motion puppet yeti, navy scarf, dawn-lit miniature set, 35mm, tungsten practicals, Rankin/Bass animagic.
BEAT: the sun rises behind jagged felt mountains; the puppet enters frame, looking up in awe.
```

The key idea is: **the anchor should be stable and repeated verbatim**, while the beat changes.

### 2.4 Smooth prompt changes: transitions vs hard cuts
Hard prompt cuts are the fastest way to introduce temporal artifacts. A smoother strategy is:
- transition over a few blocks, e.g. 4–8 blocks
- using embedding interpolation (linear/slerp) rather than “sudden re-encode”

Scope already contains the machinery to do this:
- frontend emits `transitionSteps` and `temporalInterpolationMethod` per timeline segment (see below)
- server parameters support a `transition` object
- the Wan2.1 block graph includes an **EmbeddingBlendingBlock** that implements temporal transitions

## 3) What Exists in the Repo (Relevant Pieces)

### 3.1 Timeline export format (frontend)
Export happens in:
- `frontend/src/components/PromptTimeline.tsx`

Key details:
- Filename pattern: `timeline-${YYYY-MM-DD}.json`
- `version: "2.1"` includes `settings.inputMode` plus additional pipeline knobs (quantization, kvCacheAttentionBias, LoRAs).

Timeline prompt blocks can contain:
- `startTime`, `endTime`
- `prompts: [{text, weight}, ...]` (preferred)
- optional `transitionSteps`, `temporalInterpolationMethod` (present for live prompts created via the UI)

### 3.2 Server-side “transition” schema
API schema includes:
- `src/scope/server/schema.py` → `PromptTransition`
  - `target_prompts`
  - `num_steps`
  - `temporal_interpolation_method`

The server frame processor is responsible for *keeping a transition alive* until it’s done.

### 3.3 Krea Realtime pipeline structure (Scope)
Pipeline class:
- `src/scope/core/pipelines/krea_realtime_video/pipeline.py` → `KreaRealtimeVideoPipeline`

It uses a modular block graph:
- `src/scope/core/pipelines/krea_realtime_video/modular_blocks.py`
  - includes `TextConditioningBlock`, `EmbeddingBlendingBlock`, `DenoiseBlock`, etc.

The pipeline keeps a persistent state object:
- `self.state = PipelineState()` in `KreaRealtimeVideoPipeline.__init__`

That means an offline renderer can:
- instantiate the pipeline
- call it repeatedly with updated parameters
- and get “single-take” continuity from the stateful caches

### 3.4 Existing offline scripts (but not timeline-driven)
There are per-pipeline test scripts like:
- `src/scope/core/pipelines/krea_realtime_video/test.py`

They demonstrate how to call the pipeline in a loop and export an MP4, but they do not parse timeline JSON nor implement segment/transition orchestration.

## 4) Integration Option We Chose: Offline Timeline Renderer CLI

We implemented a CLI that consumes a Scope timeline JSON export and produces an MP4 offline.

### 4.1 New CLI entrypoint
- `src/scope/cli/render_timeline.py`
- wired via `pyproject.toml` as `render_timeline = "scope.cli.render_timeline:main"`

### 4.2 “Dry run” validation mode (no model load)
To validate that a timeline parses and to see the resolved render plan **without** loading models:
```bash
uv run render_timeline timeline.json out.mp4 --dry-run
```

This prints a JSON summary (duration, target frames, effective resolution/steps, segment list) and exits.

### 4.3 What it supports (current scope)
- Pipeline: **only** `settings.pipelineId == "krea-realtime-video"` (for now)
- Mode: **text-to-video** only (timeline `inputMode: "text"`)
- Reads settings from the timeline:
  - resolution, seed
  - denoisingSteps → `denoising_step_list`
  - manageCache
  - quantization
  - kvCacheAttentionBias
  - LoRAs (+ merge strategy)
- Offline knobs via flags:
  - `--height/--width`, `--fps`, `--seed`
  - `--num-inference-steps` (converted into a longer timestep schedule)
  - `--denoising-steps` (explicit timesteps)
  - `--kv-cache-attention-bias`
  - `--kv-cache-num-frames` (overrides `kv_cache_num_frames` in `model.yaml`)
  - `--quantization fp8_e4m3fn|none`
  - `--compile/--no-compile`
  - `--no-transitions` to hard cut

### 4.4 Render loop mechanics
Core idea:
- Compute total frames = `ceil(timeline_end_time * fps)`
- For each pipeline call:
  - determine which timeline segment we’re currently in based on `produced_frames / fps`
  - if we crossed into a new segment:
    - if transitions enabled: send `transition={target_prompts, num_steps, temporal_interpolation_method}`
    - else: set `prompts=target_prompts` (hard cut)
  - call pipeline and append frames
  - stop when total frames reached

This mirrors how the realtime server keeps a prompt program alive, except the “clock” is the offline loop.

## 5) Transition Smoothing: How It Works End-to-End

### 5.1 Where temporal transitions are implemented
Temporal smoothing is implemented in:
- `src/scope/core/pipelines/wan2_1/blocks/embedding_blending.py`
- `src/scope/core/pipelines/blending.py` → `EmbeddingBlender`

Mechanism:
- `TextConditioningBlock` encodes prompt embeddings (but intentionally does **not** re-emit embeddings if prompts are unchanged).
- `EmbeddingBlendingBlock`:
  - performs spatial blending across multiple prompt weights
  - and can start a **temporal transition** between the previous embedding and the new embedding
  - it advances that transition one step per pipeline call
  - it sets `state.set("_transition_active", True/False)` for orchestration

Why orchestrators can observe `_transition_active`:
- `KreaRealtimeVideoPipeline` retains a persistent `PipelineState` (`self.state`) across calls.
- That means both the realtime server and the offline CLI can poll `pipeline.state["_transition_active"]` after each call.

### 5.2 Why orchestration must “keep sending transition”
A transition is not a single operation; it’s a **multi-call** operation:
- each call consumes the next interpolated embedding in the transition queue
- after `num_steps` calls, the queue is empty, and the transition completes

So the orchestrator must:
- keep passing `transition` until the pipeline reports `_transition_active == False`
- then clear `transition`

This is the pattern used by the realtime server `FrameProcessor` and replicated in the offline CLI.

## 6) Bugs Found + Fixes

### 6.1 Critical bug: transitions were being canceled every call

Symptom:
- You’d start a transition, but it would never actually “progress” smoothly across calls.
- It would snap, or re-start, or effectively behave like there is no temporal smoothing.

Root cause:
- `EmbeddingBlendingBlock` previously detected “conditioning changed” via a signature compare:
  - it stored a signature when embeddings were present
  - but on later calls (when `TextConditioningBlock` correctly stopped emitting embeddings), the signature became `None`
  - signature changes caused the block to think conditioning changed again
  - it then canceled active transitions (`cancel_transition`) repeatedly

Fix:
- In `src/scope/core/pipelines/wan2_1/blocks/embedding_blending.py` we changed “conditioning_changed” semantics to:
  - treat conditioning as changed **only when new embeddings are actually provided**

This matches the upstream contract: embeddings are only present when prompts genuinely changed.

### 6.2 Orchestration bug: clearing transition but leaving prompts stale

Issue:
- After a transition completes, the orchestrator clears the `transition` parameter.
- But if `prompts` is still the old value in the orchestrator’s parameter dict, you can get:
  - confusion/debuggability issues
  - edge cases where a later cache reset might fall back to old prompts
  - potential “bounce-back” if any internal prompt-tracking state is reset

Fixes:
- Offline CLI: `src/scope/cli/render_timeline.py` now copies `transition.target_prompts` into `parameters["prompts"]` when the transition completes, then removes `transition`.
- Realtime server: `src/scope/server/frame_processor.py` now does the same, persisting the final target prompts into `self.parameters["prompts"]` before clearing `transition`.

## 7) Quality Knobs: What to Change Offline (Practical Guidance)

### 7.1 More inference steps
In this codebase:
- “number of inference steps” corresponds to `len(denoising_step_list)` in `DenoiseBlock`.

The CLI provides:
- `--num-inference-steps 6` → converts into a longer timestep schedule between the first and last timesteps you’re using.
- `--denoising-steps 1000,850,700,550,400,250` → explicit control.

### 7.2 Higher resolution
Use `--height/--width`. Constraint:
- must be divisible by 16 (matches latent/channel layout and WAN components’ downsample factors).

### 7.3 KV cache attention bias
`kv_cache_attention_bias` is passed into `DenoiseBlock` and down into the generator:
- lower values reduce reliance on past frames in cache
- can mitigate error accumulation / repetitive motion
- but may weaken long-range temporal coherence

### 7.4 KV cache recomputation window (`kv_cache_num_frames`)
Krea pipeline includes a KV cache recomputation block:
- `src/scope/core/pipelines/krea_realtime_video/blocks/recompute_kv_cache.py`
- configured by `kv_cache_num_frames` in `model.yaml`

Larger windows can improve stability, but increase compute and memory.
The CLI allows overriding this value via `--kv-cache-num-frames`.

### 7.5 Quantization vs quality
FP8 (`fp8_e4m3fn`) reduces VRAM and can improve throughput.
For absolute quality, you may prefer no quantization (if VRAM allows), but this is model/hardware dependent.

### 7.6 Performance planning (estimates + how to measure)
There isn’t a single “true” FPS number because runtime depends on:
- GPU + attention kernel choice + FP8 vs BF16
- resolution (pixels)
- number of denoising steps (`len(denoising_step_list)`)
- cache recomputation settings (`kv_cache_num_frames`) and how often cache resets happen

Two useful ways to reason about it:

1) **Work estimate (no hardware assumptions)**
- Total frames to generate: `ceil(duration_seconds * fps)`
- Rough number of pipeline calls: `ceil(total_frames / 3)` (Krea defaults to 3 frames per call)

2) **Scaling heuristic (relative to your own baseline)**
If you measure a baseline FPS at some settings, offline settings often scale roughly like:
```text
fps_new ≈ fps_base * (steps_base / steps_new) * (pixels_base / pixels_new)
```
This isn’t exact, but it’s usually directionally correct for planning.

To measure on your machine:
- Run a short offline render (5–10 seconds) and observe wall time.
- Or start from `src/scope/core/pipelines/krea_realtime_video/test.py` and adapt.

Reference (vendor claim):
- `notes/krea_rt.md` states Krea Realtime achieves ~11 fps at 4 steps on a single B200 (model card claim). Treat this as a ballpark, not a guarantee.

## 8) Future Enhancement: V2V “Refine Pass”

A practical offline workflow for “higher res + higher detail” while preserving continuity:
1. T2V timeline render at lower/medium resolution (fast preview).
2. Upscale the resulting video (classic SR/upscaling).
3. Run **video-to-video** through Krea with a low “strength” (~0.3) and more steps to re-detail.

In this codebase, the V2V analogue of “strength” is effectively `noise_scale`:
- In `PrepareVideoLatentsBlock`, the input latents are mixed with noise using `noise_scale`.
- Lower `noise_scale` preserves more of the input video.

CLI V2V support is not implemented yet, but the pipeline already supports V2V mode; it would require:
- loading an input video
- chunking frames to meet `prepare_for_mode()` requirements
- calling pipeline with `video=...` and `noise_scale=...`
- optionally applying the same timeline prompt schedule over time

## 9) Files Changed / Added (Implementation Landmarks)

Added:
- `src/scope/cli/render_timeline.py`
- `src/scope/cli/__init__.py`
- `tests/test_render_timeline_cli.py`

Wired:
- `pyproject.toml` entrypoint `render_timeline`

Fixed transition behavior:
- `src/scope/core/pipelines/wan2_1/blocks/embedding_blending.py`
- `src/scope/server/frame_processor.py`

## 10) Quick Usage Examples

Offline render from a timeline export:
```bash
uv run render_timeline timeline-2025-12-19.json out.mp4
```

Higher quality (more steps) + higher resolution:
```bash
uv run render_timeline timeline-2025-12-19.json out.mp4 \
  --height 720 --width 1280 \
  --num-inference-steps 6 \
  --fps 24
```

Disable transitions (hard cuts):
```bash
uv run render_timeline timeline-2025-12-19.json out.mp4 --no-transitions
```

## Appendix: How we investigated (high-level)
This is the non-private reasoning path that led to the implementation:
1. Locate the exact timeline export schema in the frontend and confirm which knobs are already serialized.
2. Confirm the Krea pipeline is stateful (persistent `PipelineState`) and supports multi-call generation.
3. Trace transitions end-to-end: timeline segment fields → server schema → frame processor lifecycle → modular blocks.
4. Find and fix the temporal transition cancelation bug in `EmbeddingBlendingBlock` caused by “missing embeddings” on subsequent calls.
5. Fix orchestrator lifecycle so that when a transition completes and is cleared, `prompts` is also updated to the final target to avoid stale state.
