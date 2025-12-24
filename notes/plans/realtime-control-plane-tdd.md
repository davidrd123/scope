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

#### Key Decisions (RESOLVED)

1. **Thread safety**: Keep `queue.Queue` as cross-thread boundary. Translate dict updates → events inside `process_chunk()`. Worker thread remains single owner of pipeline state.

2. **Unknown params**: Keep `self.parameters: dict` as source-of-truth. Use control-plane abstractions only for semantics that matter (ordering, LoRA edge-triggering, snapshot continuity later). No `passthrough_params` needed in Phase 1.

3. **Drain-all semantics**: Drain ALL pending queue entries at top of each chunk (not just one). Treat queue as "mailbox since last boundary" - cleaner semantic, reduces dropped updates.

   **⚠️ INTENTIONAL BEHAVIOR CHANGE**: Current `FrameProcessor` drains at most 1 update per chunk. New behavior drains ALL pending updates. Edge case: 10 rapid updates during one chunk → old behavior applies them over 10 chunks; new behavior applies all at once. This is intentional and usually better (commit all pending at boundary). Characterization tests should catch regressions.

#### Implementation Flow

```
┌─────────────────────────────────────────────────────────────┐
│ INGEST (top of process_chunk)                               │
├─────────────────────────────────────────────────────────────┤
│ 1. Drain ALL queue entries (while not empty)                │
│ 2. Merge with "last write wins" → merged_dict               │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ TRANSLATE (dict → typed events)                             │
├─────────────────────────────────────────────────────────────┤
│ Use EXISTING EventTypes only:                               │
│   if "paused" in merged      → PAUSE/RESUME                 │
│   if "prompts"/"transition"  → SET_PROMPT                   │
│   if "lora_scales"           → SET_LORA_SCALES              │
│   if "base_seed"             → SET_SEED                     │
│   if "denoising_step_list"   → SET_DENOISE_STEPS            │
│   remaining keys             → merge into self.parameters   │
│                                (no new event type needed)   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ ORDER + APPLY                                               │
├─────────────────────────────────────────────────────────────┤
│ control_bus.drain_pending() → deterministic order           │
│ Apply each event → mutate self.parameters, self.paused, etc │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ CALL PIPELINE (existing logic, but use PipelineAdapter)     │
├─────────────────────────────────────────────────────────────┤
│ - reset_cache consumed HERE (after pause check, like today) │
│ - PipelineAdapter.kwargs_for_call() handles:                │
│     - init_cache (always explicit)                          │
│     - lora_scales edge-triggering                           │
└─────────────────────────────────────────────────────────────┘
```

#### Ordering Nuance

Keep `reset_cache` consumption AFTER the pause check (current behavior). If `reset_cache` is sent while paused, it doesn't clear output / pop until generation resumes. Event application can set `self.parameters["reset_cache"] = True`, but only act on it in the "call pipeline" section when not paused.

#### Characterization Tests (DONE - 6 tests passing)

- `test_init_cache_true_first_call_then_false`
- `test_reset_cache_clears_output_queue_and_forces_init_cache`
- `test_lora_scales_is_one_shot`
- `test_transition_completion_clears_transition_and_updates_prompts`
- `test_new_prompts_without_transition_clears_stale_transition`
- `test_video_mode_requires_input_frames_before_call`

#### What FrameProcessor Will Own (After Integration)

- `control_bus: ControlBus` (deterministic ordering at chunk boundaries)
- `adapter: PipelineAdapter` (kwargs mapping, LoRA edge-trigger; continuity for Phase 2)
- `self.parameters: dict` (source-of-truth for all params, including passthrough)

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
- Key commits:
  - `0655010` - Scaffold realtime control plane with 84 tests
  - `6d07931` - Smoke harness script (validated on B300)
  - `2a6d9c5` - Characterization tests (6 passing) + updated plan

## Test Coverage

- **84 tests** for control plane abstractions (`tests/realtime/`)
- **6 tests** characterizing FrameProcessor behavior (`tests/test_frame_processor_characterization.py`)

## Next Action

**Phase 1 Integration** (decisions resolved, ready to implement):

1. Add `control_bus: ControlBus` and `adapter: PipelineAdapter` to `FrameProcessor.__init__()`
2. Modify `process_chunk()`:
   - Drain ALL queue entries → merge with last-write-wins
   - Translate to typed events (PAUSE/RESUME, SET_PROMPT, SET_LORA_SCALES, SET_SEED, SET_DENOISE_STEPS)
   - Remaining keys merge directly into `self.parameters`
   - `control_bus.drain_pending()` for ordering
   - Apply events → mutate `self.parameters`, `self.paused`, etc.
3. Replace manual `init_cache` / `lora_scales` handling with `adapter.kwargs_for_call()`
4. Ensure all 6 characterization tests still pass
