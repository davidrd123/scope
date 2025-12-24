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
- Event ordering: lifecycle → style → world → prompt → params
- `NEXT_BOUNDARY` vs `IMMEDIATE_IF_PAUSED` filtering
- History tracking with chunk index

**PipelineAdapter:**
- `kwargs_for_call()` produces correct kwargs
- `lora_scales` edge-triggered (only included when changed)
- `capture_continuity()` reads from `pipeline.state` keys
- `restore_continuity()` writes to `pipeline.state` keys

**GeneratorDriver:**
- Events applied only at chunk boundaries
- Snapshot captures control + continuity
- Restore sets `_is_prepared = True` (no accidental cache reset)
- `transition` dict passed through to pipeline

### Phase 2: Smoke Harness (Real GPU, not CI)

Single integration test with real KREA pipeline:
- Verify state keys exist and are correct types
- Verify continuity restore doesn't hard-reset unexpectedly
- Manual visual check for seamless continuation

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
- Last commit: `6b77399` - B300/WAN2.1 docs
- Uncommitted experimental code in `src/scope/core/pipelines/` (leave for now)

## Next Action

Scaffold the `src/scope/realtime/` and `tests/realtime/` directories with:
1. Dataclass definitions from architecture doc
2. Test files with key test cases
3. Minimal implementations to make structure clear
