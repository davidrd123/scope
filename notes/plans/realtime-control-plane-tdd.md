# Realtime Control Plane - TDD Implementation Plan

**Created**: 2025-12-24
**Status**: Scaffold complete; integrate into server runtime next

## Context

Architecture doc v1.2 is complete (`notes/realtime_video_architecture.md`) and addresses the v1.1 review feedback (`notes/research/2025-12-23/realtime-architecture-v1.1-review.md`).

We now have a **tested, pure-Python control plane scaffold** (`src/scope/realtime/` + `tests/realtime/`) that encodes the chunk-boundary semantics, continuity snapshot keys, and LoRA edge-triggering.

The next step is to **integrate these abstractions into the existing server runtime** instead of building a parallel “driver loop” that duplicates `FrameProcessor`.

## What We're Building

### Phase 0 (Done): Control Plane (TDD - Pure Python)

```
src/scope/realtime/
├── __init__.py
├── control_state.py      # ControlState, WorldState, CompiledPrompt dataclasses
├── control_bus.py        # ControlBus, ControlEvent, EventType, ApplyMode
├── pipeline_adapter.py   # PipelineAdapter (kwargs mapping, continuity, edge-trigger)
└── generator_driver.py   # GeneratorDriver (tick loop, event application)

tests/realtime/
├── __init__.py
├── conftest.py           # FakePipeline fixture
├── test_control_state.py
├── test_control_bus.py
├── test_pipeline_adapter.py
└── test_generator_driver.py
```

### Key Test Cases (from architecture spec)

**ControlBus:**
- Event ordering (deterministic at chunk boundary):
  lifecycle → snapshot/restore → style → world → prompt/transition → params
- `NEXT_BOUNDARY` vs `IMMEDIATE_IF_PAUSED` filtering
- History tracking with chunk index (choose one representation and test to it):
  - Option A: annotate `ControlEvent.applied_chunk_index`
  - Option B: store an `AppliedEvent{event, applied_chunk_index}` log entry

**PipelineAdapter:**
- `kwargs_for_call()` produces correct kwargs
- `lora_scales` edge-triggered (only included when changed)
- `capture_continuity()` reads from `pipeline.state` keys
- `restore_continuity()` writes to `pipeline.state` keys
- `negative_prompt` is *not* forwarded to Scope/KREA pipeline kwargs (field can exist for future backends)
- `transition` is pipeline-native (Scope) dict:
  `{"target_prompts": [...], "num_steps": 4, "temporal_interpolation_method": "linear"}`
- `init_cache` is always explicit in kwargs (adapter/driver decides; no reliance on stale state)

**GeneratorDriver:**
- Events applied only at chunk boundaries
- Snapshot captures control + continuity
- Restore sets `_is_prepared = True` (no accidental cache reset)
- `transition` dict passed through to pipeline

### Clarifications (From Repo “Reality Check” Critique)

The proposal in `notes/realtime_video_architecture.md` largely **sits on top of what the repo already does** today:

- **`FrameProcessor` is the existing “GeneratorDriver”**: worker thread loop, chunk gating, parameter merge, pause/reset, transition lifecycle, LoRA one-shot updates, output queue.
- **`pipeline.state` is the continuity store**: context buffers + `current_start_frame` already live in state keys (not pipeline attributes).
- **WebRTC data channel is the existing control plane**: `WebRTCManager → VideoProcessingTrack → FrameProcessor.update_parameters(...)`.

Main gaps to call out explicitly in future doc revisions and implementation planning:

- **Multi-session isolation risk**: `WebRTCManager` supports multiple sessions, but `PipelineManager` appears singleton-ish (one pipeline instance). If multiple `FrameProcessor` threads call the same pipeline concurrently, `pipeline.state` and cache continuity can be cross-contaminated.
- **Input readiness is first-class in video mode**: `FrameProcessor.process_chunk()` gates on `pipeline.prepare()` requirements and waits for enough input frames.
- **FrameBus/Timeline are aspirational today**: runtime is transient queues (output queue), not a retention layer for scrubbing/branch previews.
- **Schema vocabulary mismatch**: pipeline config uses `denoising_steps`; runtime control uses `denoising_step_list`. UI/schema-default mapping needs an explicit translation layer.
- **LoRA merge mode caps “live knobs”**: `PERMANENT_MERGE` implies no runtime updates; `RUNTIME_PEFT` enables them at reduced FPS.

### Phase 1: Integrate Into `FrameProcessor` (Start Here)

Goal: **reuse existing server threading + WebRTC wiring**, and move the *tested control semantics* into the runtime.

Target shape:

- `FrameProcessor` owns:
  - `ControlBus` (deterministic ordering at chunk boundaries)
  - `PipelineAdapter` (kwargs mapping, continuity snapshot keys, LoRA edge-trigger)
  - a “current state” object (either `ControlState` or a small wrapper that preserves the current flexible dict-based kwargs model)

Key decisions to make before refactor:

- **Thread safety**: `ControlBus` is not thread-safe; either protect with a lock or keep using the existing `queue.Queue` and translate queued dict updates → events at the top of `process_chunk()`.
- **Merge semantics**: current runtime merges dict updates and effectively gives “last write wins” per key; event application must preserve this.
- **Unknown params**: `ControlState` is currently a minimal typed surface; the server runtime supports many additional kwargs (`noise_scale`, `manage_cache`, `vace_ref_images`, Spout config, etc.). Decide whether to:
  - add a `passthrough_params: dict[str, Any]` to `ControlState`, or
  - keep the dict as source-of-truth and use the control plane only for the special-case semantics.

TDD approach:

- Add **characterization tests** for current `FrameProcessor.process_chunk()` semantics (pause, init_cache/reset_cache, lora_scales one-shot, transition lifecycle, input gating).
- Refactor incrementally until the same tests pass, then swap implementation details to use `ControlBus`/`PipelineAdapter`.

### Phase 2: Snapshot/Restore (Server-Visible)

Add snapshot/restore endpoints or messages (design choice: WebRTC-only vs REST+WebRTC), implemented via `PipelineAdapter.capture_continuity()` / `.restore_continuity()` plus control state capture.

### Phase 3: Step Mode API (Optional / Product Choice)

Expose “generate one chunk” control for dev console/testing:

- In T2V: step always advances.
- In V2V: step must define behavior when insufficient input frames (stall vs reuse vs error).

### Phase 4: Smoke Harness (Real GPU, not CI)

Single integration check with real KREA pipeline (skip by default):
- Verify state keys exist and are correct types
- Verify continuity restore doesn't hard-reset unexpectedly
- Manual visual check for seamless continuation

Implementation suggestion:
- Either a `pytest` test behind a marker (e.g. `@pytest.mark.gpu_smoke`) that is skipped unless explicitly enabled, or
- A `scripts/realtime_smoke.py` runner that prints state keys and writes a short clip.

### Phase 3: Build On Top

- Branching/rollout (sequential unless multi-worker)
- UI/streaming integration
- Timeline layer

## Files to Reference

- `notes/realtime_video_architecture.md` - Full spec (v1.2)
- `notes/research/2025-12-23/realtime-architecture-v1.1-review.md` - 5Pro feedback
- `src/scope/core/pipelines/krea_realtime_video/` - Existing pipeline to integrate with
- `src/scope/server/frame_processor.py` - Existing runtime driver loop to refactor/extend

## Git State

- Branch: `feature/stream-recording`
- Scaffold commits:
  - `8d93448` - realtime control plane TDD plan
  - `0655010` - Scaffold realtime control plane with tests
  - `6d07931` - Add smoke harness script

## Next Action

Start Phase 1 with TDD:
1. Add characterization tests around `FrameProcessor.process_chunk()`
2. Introduce `ControlBus` + `PipelineAdapter` into `FrameProcessor` behind a small translation layer
3. Preserve existing behavior, then iterate towards the cleaner control-state model
