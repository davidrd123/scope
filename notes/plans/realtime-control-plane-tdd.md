# Realtime Control Plane - TDD Implementation Plan

**Created**: 2025-12-24
**Status**: Ready to scaffold

## Context

Architecture doc v1.2 is complete (`notes/realtime_video_architecture.md`) and addresses all 5Pro review feedback. Now implementing the Control Plane with TDD.

## What We're Building

### Phase 1: Control Plane (TDD - Pure Python)

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

### Phase 2: Smoke Harness (Real GPU, not CI)

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

## Git State

- Branch: `feature/stream-recording`
- Last commit: `8d93448` - realtime control plane TDD plan
- Existing uncommitted changes in `src/scope/core/pipelines/` (leave for now)

## Next Action

Scaffold the `src/scope/realtime/` and `tests/realtime/` directories with:
1. Dataclass definitions from architecture doc
2. Test files with key test cases
3. Minimal implementations to make structure clear
