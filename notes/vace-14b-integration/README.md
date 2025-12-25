# VACE-14B Integration (Krea Realtime Video)

This folder is the handoff packet for adding **VACE-14B** support to the **Krea 14B** pipeline (`krea-realtime-video`).

**Primary goal:** make it easy for the next engineer to (1) understand how VACE is wired in this repo, (2) implement the missing Krea wiring safely, and (3) validate correctness + performance/memory impact.

## Where to start

- Implementation plan (canonical): `notes/vace-14b-integration/plan.md`
- Chronological context / prior assumptions: `notes/vace-14b-integration/work-log.md`
- Background research (architecture + upstream availability): `notes/vace-14b-integration/research/architecture-research.md`
- Repo-wide VACE overview (how it works today): `notes/guides/vace-architecture-explainer.md`

## What already exists (today)

- **Model wrapper (pipeline-agnostic):** `src/scope/core/pipelines/wan2_1/vace/models/causal_vace_model.py` (`CausalVaceWanModel`)
- **Pipeline mixin:** `src/scope/core/pipelines/wan2_1/vace/mixin.py` (`VACEEnabledPipeline._init_vace`)
- **Per-chunk encoder block:** `src/scope/core/pipelines/wan2_1/vace/blocks/vace_encoding.py` (`VaceEncodingBlock`)
- **Server routing for “video mode”:** `src/scope/server/frame_processor.py` routes incoming frames to:
  - `video` when VACE is **disabled**
  - `vace_input_frames` when `pipeline.vace_enabled` is **True**

## What’s missing specifically for Krea 14B

Minimum “VACE-14B works” wiring items:

- **Artifacts:** add the 14B VACE module to `src/scope/server/pipeline_artifacts.py`
- **Checkpoint selection:** `src/scope/server/pipeline_manager.py:_get_vace_checkpoint_path()` is hardcoded to the 1.3B module today; Krea needs the 14B path
- **Load params:** `src/scope/server/schema.py:KreaRealtimeVideoLoadParams` is missing `vace_enabled`
- **Pipeline init:** `src/scope/core/pipelines/krea_realtime_video/pipeline.py` must:
  - inherit `VACEEnabledPipeline`
  - call `_init_vace()` **before** projection fusing + LoRA
  - fuse projections on both `blocks` and `vace_blocks`
  - clear `vace_ref_images` when not provided (prevents stale ref reuse)
- **Block graph:** insert `VaceEncodingBlock` into `src/scope/core/pipelines/krea_realtime_video/modular_blocks.py`

## Critical semantics (don’t skip this)

- **VACE is a load-time toggle.** `pipeline.vace_enabled` is set by `_init_vace()` and is used by the server to route frames. If you load Krea without VACE, you cannot “turn it on” later without reloading the pipeline.
- **VACE-enabled “video mode” is not normal V2V.** When VACE is enabled, the server routes frames to `vace_input_frames` (conditioning), not `video` (noisy-latent initialization). That means Krea stays on the T2V latent path while using video frames only as VACE conditioning.
- **You cannot combine normal V2V and VACE V2V editing** in a single run (the server explicitly routes to one path or the other).

## Biggest decision points / risks

- **VRAM + perf:** VACE-14B module is ~6.1GB and adds extra compute (VACE blocks run inside the forward). Defaulting `vace_enabled=False` for Krea is likely necessary.
- **Quantization policy:** Krea defaults to FP8 quantization; decide whether VACE blocks should also be quantized (memory) or kept bf16 (quality/stability).
- **KV-cache recompute semantics:** Krea’s `RecomputeKVCacheBlock` calls the generator without `vace_context`; decide whether that’s correct for your VACE use-cases.

## Validation

- Validation checklist + commands live in `notes/vace-14b-integration/plan.md`.
