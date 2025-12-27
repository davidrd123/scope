# Server-Side Session Recorder

> Status: Draft (revised per review01 + review02 findings)
> Date: 2025-12-26, revised 2025-12-27
> Reviews:
> - `notes/proposals/server-side-session-recorder/review01.md`
> - `notes/proposals/server-side-session-recorder/review02.md`
> - `notes/proposals/server-side-session-recorder/review03a.md`

## Revision Summary

### Rev 3 (2025-12-27) — review03a integration

Added an explicit “executable contract” for the MVP:
- Scope (what is and isn’t recorded)
- Definition of Done (acceptance checks)
- File contract + precedence rules (so replay doesn’t churn)
- Chunk-timebase guidance (to avoid wall-clock drift)

### Rev 2 (2025-12-27) — review02 fixes

| Issue | Fix |
|-------|-----|
| **A** | `event.event_type` → `event.type` (attribute name mismatch) |
| **B** | Status comparison: `"loaded"` (lowercase .value), field `_error_message` not `_error` |
| **C** | Baseline prompt: prefer `parameters["prompts"]`; compiler output is not a string (handle `.prompts`/`.prompt`/`.positive` shapes) |
| **D** | Note: render_timeline.py manual timelines still default to 100.0 (future cleanup) |
| **Hook** | Use existing soft-cut internal state (`_soft_transition_original_bias`) directly |
| **Hard cut** | Record when `reset_cache=True` is actually passed as `init_cache=True` (avoid confusing initial warmup init_cache with a user hard cut) |

### Rev 1 (2025-12-27) — review01 fixes

| Risk | Issue | Fix |
|------|-------|-----|
| **A** | `_compiled_prompt` is a compiler object, not a string | Extract baseline prompt from `.prompts[0]` / `.positive[0]` (shape-dependent) |
| **B** | Weight scale mixed (1.0 vs 100.0) | Standardized on 1.0 everywhere |
| **C** | Non-ControlBus prompt changes missed | Added fallback edge detection |
| **E** | Soft cut restore semantics lost | Added `restoreBias`/`restoreWasSet` fields |
| **§3-5** | render_timeline.py support missing | Added schema + loop implementation |

---

## Purpose

Capture all control events at the server level, regardless of source (CLI, API, frontend). This complements the frontend-only approach which only sees UI-driven changes.

## MVP Spec (Executable Contract)

This is the “freeze the semantics” section: once this is written down, implementation becomes wiring and tests instead of contract churn.

### Scope (MVP)

- **Input mode:** text-only (`settings.inputMode = "text"`). Offline renderer rejects video/VACE today.
- **Record (MVP):**
  - prompt updates (including `transition` payloads)
  - hard cuts (`reset_cache=True` → exported as `initCache=true`)
  - soft cuts (`_rcp_soft_transition` → exported as `softCut`)
- **Not recorded (MVP):** LoRA scale changes, seed/denoise step changes, style/world updates (unless they also manifest as a prompt event).

To expand from “timeline recorder” → “full session recorder”, the first high-impact addition is recording and replaying `SET_LORA_SCALES` (otherwise style swaps can diverge visually).

### Definition of Done (MVP acceptance checks)

1. **Hard cut replay fidelity**  
   If a recorded segment has `initCache: true`, offline replay calls the pipeline with `init_cache=True` for **exactly one** pipeline call at that boundary (one-shot, not sticky).

2. **Soft cut replay fidelity**  
   If a recorded segment has `softCut: {bias, chunks, restoreBias, restoreWasSet}`, offline replay:
   - immediately sets `kv_cache_attention_bias=bias`
   - keeps it for exactly `chunks` pipeline calls
   - then restores to `restoreBias` if `restoreWasSet=true`, else restores to “unset” (delete the kwarg)

3. **Transition replay fidelity**  
   If a segment includes transition metadata, offline replay passes a `transition` dict until the pipeline signals completion (`pipeline.state["_transition_active"] == False`), then:
   - clears `transition`, and
   - sets `prompts = transition.target_prompts` (so subsequent segments hold the new prompt).

4. **Stop is async, but path is observable**  
   `POST /api/v1/realtime/session-recording/stop` returns immediately, and `GET /api/v1/realtime/session-recording/status` eventually includes `last_timeline_path`.

5. **No prompt changes still yields a valid timeline**  
   Starting and stopping recording without any prompt changes produces a replayable timeline. (This implies baseline prompt capture must be pipeline-facing: transition target → `parameters["prompts"]` → `pipeline.state["prompts"]` (warmup fallback) → compiler fallback, and the start endpoint should fail loudly if none exist.)

### File contract + precedence rules (MVP)

- **Version:** include `version` (string). Optionally add `primaryTimebase: "chunk"` for clarity; keep `startTime/endTime` as human-facing secondary timebase.
- **Prompt weights:** replay uses recorded weights; if absent, default to `1.0` (match server/runtime conventions).
- **Precedence:**
  - `initCache=true` should be treated as a boundary cut; do not initiate a transition at that boundary.
  - `softCut` is orthogonal and may coexist with either cut or transition.

### Scheduling (offline replay)

For fidelity, canonical replay scheduling should be chunk-based (`startChunk/endChunk`). Time-based scheduling (`startTime/endTime`) is a fallback and can drift under stalls/pauses. A practical renderer UX is `--timebase auto|chunk|time` (default `auto`).

### Optional measurability hook (recommended)

Add an opt-in debug mode where the recorder logs (or stores) the exact per-call pipeline kwargs around each recorded event boundary. This makes “record → replay” validation a simple diff of call sequences.

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   CLI       │     │  Frontend   │     │   API       │
│ playlist nav│     │  Timeline   │     │  Direct     │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │
       └───────────────────┼───────────────────┘
                           │
                           ▼
                 ┌─────────────────────┐
                 │   frame_processor   │
                 │  ┌───────────────┐  │
                 │  │SessionRecorder│◄─┼── captures ALL events
                 │  └───────────────┘  │    (inside process_chunk)
                 └─────────────────────┘
```

## Events to Capture

| Event | Source | Data |
|-------|--------|------|
| Prompt change | `prompts` in control message | prompt text, weight |
| Transition start | `transition` in control message | target_prompts, num_steps, method |
| Transition-only | `transition` without `prompts` | use target_prompts as new segment |
| Hard cut | `reset_cache` in control message | flag (record when `reset_cache=True` is actually passed as `init_cache=True`) |
| Soft cut | `_rcp_soft_transition` reserved key | temp_bias, num_chunks (may occur without prompt change) |

## Implementation

### 1. SessionRecorder Class

```python
# src/scope/server/session_recorder.py

from dataclasses import dataclass, field
from datetime import datetime
import json
import time
from pathlib import Path

@dataclass
class ControlEvent:
    """A single control event during recording."""
    chunk_index: int  # Pipeline chunk when event occurred (primary timebase)
    wall_time: float  # Seconds since recording started (secondary)

    # Prompt (may be None for cut-only events)
    prompt: str | None = None
    prompt_weight: float = 1.0  # FIXED: Use 1.0 consistently (matches pipeline warmup)

    # Transition
    transition_steps: int | None = None
    transition_method: str | None = None  # "linear" or "slerp"

    # Cuts (can occur with or without prompt change)
    hard_cut: bool = False
    soft_cut_bias: float | None = None
    soft_cut_chunks: int | None = None
    # ADDED: Soft cut restore semantics (per review01 Risk E)
    soft_cut_restore_bias: float | None = None  # Original bias to restore (None = was unset)
    soft_cut_restore_was_set: bool = False  # True if bias was explicitly set before soft cut


@dataclass
class SessionRecording:
    """Container for a complete recording session."""
    events: list[ControlEvent] = field(default_factory=list)

    # Timing - chunk-based is primary for offline fidelity
    start_chunk: int = 0
    end_chunk: int | None = None  # Set on stop()

    # Wall-clock secondary (affected by stalls/pauses)
    start_wall_time: float | None = None
    end_wall_time: float | None = None  # Set on stop() - fixes duration bug

    # Pipeline info from pipeline_manager
    pipeline_id: str | None = None
    load_params: dict = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        return self.start_wall_time is not None and self.end_wall_time is None

    @property
    def duration_seconds(self) -> float:
        """Wall-clock duration (frozen after stop)."""
        if self.start_wall_time is None:
            return 0.0
        end = self.end_wall_time if self.end_wall_time is not None else time.monotonic()
        return end - self.start_wall_time

    @property
    def duration_chunks(self) -> int:
        """Chunk-based duration (more precise for offline render)."""
        if self.end_chunk is None:
            return 0
        return self.end_chunk - self.start_chunk


class SessionRecorder:
    """Records control events during a streaming session.

    Thread-safety: All mutation methods are called from within process_chunk(),
    serialized by the mailbox merge. Never call directly from FastAPI threads.

    For status reads from FastAPI, use get_status_snapshot() which returns an
    atomic dict snapshot (safe under GIL).
    """

    def __init__(self):
        self._recording: SessionRecording | None = None
        self._last_prompt: str | None = None  # Track for cut-only events
        # Thread-safe status snapshot (atomic dict replacement under GIL)
        self._status_snapshot: dict = {"is_recording": False}

    @property
    def is_recording(self) -> bool:
        return self._recording is not None and self._recording.is_active

    def start(
        self,
        chunk_index: int,
        pipeline_id: str,
        load_params: dict,
        baseline_prompt: str | None = None,
        baseline_weight: float = 1.0,
    ) -> None:
        """Start recording. Called from process_chunk via reserved key.

        Args:
            chunk_index: Current pipeline chunk index
            pipeline_id: Required - must not be None (fail loudly if unknown)
            load_params: Pipeline load parameters snapshot
            baseline_prompt: Current effective prompt (from transition target or prompts)
            baseline_weight: Prompt weight
        """
        if not pipeline_id:
            raise ValueError("pipeline_id is required for session recording")

        wall_time = time.monotonic()
        self._recording = SessionRecording(
            start_chunk=chunk_index,
            start_wall_time=wall_time,
            pipeline_id=pipeline_id,
            load_params=load_params.copy(),
        )

        # Record baseline prompt at t=0 to avoid empty timelines
        # (Even if user doesn't change prompt during recording, we have initial state)
        if baseline_prompt is not None:
            self._recording.events.append(ControlEvent(
                chunk_index=chunk_index,
                wall_time=0.0,  # Relative to start
                prompt=baseline_prompt,
                prompt_weight=baseline_weight,
                # No transition at t=0 - we capture current state, not replay past transition
            ))
            self._last_prompt = baseline_prompt
        else:
            self._last_prompt = None

        self._update_status_snapshot()

    def record_event(
        self,
        chunk_index: int,
        wall_time: float,
        prompt: str | None = None,
        prompt_weight: float = 1.0,  # FIXED: Use 1.0 consistently (review01 Risk B)
        transition_steps: int | None = None,
        transition_method: str | None = None,
        hard_cut: bool = False,
        soft_cut_bias: float | None = None,
        soft_cut_chunks: int | None = None,
        soft_cut_restore_bias: float | None = None,  # ADDED: review01 Risk E
        soft_cut_restore_was_set: bool = False,  # ADDED: review01 Risk E
    ) -> None:
        """Record a control event. Called from process_chunk."""
        if not self.is_recording:
            return

        # Track last prompt for cut-only events
        if prompt is not None:
            self._last_prompt = prompt

        # For cuts without prompt change, synthesize with last known prompt
        effective_prompt = prompt
        if prompt is None and (hard_cut or soft_cut_bias is not None):
            effective_prompt = self._last_prompt

        self._recording.events.append(ControlEvent(
            chunk_index=chunk_index,
            wall_time=wall_time - self._recording.start_wall_time,
            prompt=effective_prompt,
            prompt_weight=prompt_weight,
            transition_steps=transition_steps,
            transition_method=transition_method,
            hard_cut=hard_cut,
            soft_cut_bias=soft_cut_bias,
            soft_cut_chunks=soft_cut_chunks,
            soft_cut_restore_bias=soft_cut_restore_bias,
            soft_cut_restore_was_set=soft_cut_restore_was_set,
        ))

    def stop(self, chunk_index: int) -> SessionRecording | None:
        """Stop recording and return the recording. Called from process_chunk."""
        if not self.is_recording:
            return None

        self._recording.end_chunk = chunk_index
        self._recording.end_wall_time = time.monotonic()

        recording = self._recording
        self._recording = None
        self._last_prompt = None
        self._update_status_snapshot()
        return recording

    def _update_status_snapshot(self) -> None:
        """Update thread-safe status snapshot. Called after any state change."""
        rec = self._recording
        if rec is None:
            self._status_snapshot = {"is_recording": False}
        else:
            self._status_snapshot = {
                "is_recording": rec.is_active,
                "start_chunk": rec.start_chunk,
                "duration_seconds": rec.duration_seconds,
                "events_count": len(rec.events),
            }

    def get_status_snapshot(self) -> dict:
        """Thread-safe status read for FastAPI endpoints.

        Returns an atomic dict snapshot (GIL guarantees atomic pointer read).
        """
        return self._status_snapshot

    def export_timeline(self, recording: SessionRecording) -> dict:
        """Convert recording to render_timeline.py compatible format."""
        segments = []

        for i, event in enumerate(recording.events):
            # Skip events without prompt (pure metadata events)
            if event.prompt is None:
                continue

            # End time: next prompt event or recording end
            end_chunk = recording.end_chunk or recording.start_chunk
            end_time = recording.duration_seconds
            for j in range(i + 1, len(recording.events)):
                if recording.events[j].prompt is not None:
                    end_chunk = recording.events[j].chunk_index
                    end_time = recording.events[j].wall_time
                    break

            segment = {
                "startTime": event.wall_time,
                "endTime": end_time,
                "startChunk": event.chunk_index - recording.start_chunk,
                "endChunk": end_chunk - recording.start_chunk,
                "prompts": [{"text": event.prompt, "weight": event.prompt_weight}],
            }

            # Transition info
            if event.transition_steps is not None and event.transition_steps > 0:
                segment["transitionSteps"] = event.transition_steps
            if event.transition_method:
                segment["temporalInterpolationMethod"] = event.transition_method

            # Hard cut → initCache
            # NOTE: render_timeline.py does NOT yet support initCache
            if event.hard_cut:
                segment["initCache"] = True

            # Soft cut - EXTENDED per review01 Risk E for restore fidelity
            # NOTE: render_timeline.py does NOT yet support softCut
            if event.soft_cut_bias is not None:
                segment["softCut"] = {
                    "bias": event.soft_cut_bias,
                    "chunks": event.soft_cut_chunks or 2,
                    # ADDED: Restore semantics for faithful replay
                    "restoreBias": event.soft_cut_restore_bias,  # None means "was unset"
                    "restoreWasSet": event.soft_cut_restore_was_set,
                }

            segments.append(segment)

        # Build settings from load_params (dynamic, not hardcoded)
        lp = recording.load_params

        return {
            "version": "1.0",
            "exportedAt": datetime.utcnow().isoformat() + "Z",
            "recording": {
                "durationSeconds": recording.duration_seconds,
                "durationChunks": recording.duration_chunks,
                "startChunk": recording.start_chunk,
                "endChunk": recording.end_chunk,
            },
            "settings": {
                "pipelineId": recording.pipeline_id,
                "resolution": {
                    "height": lp.get("height", 480),
                    "width": lp.get("width", 832),
                },
                "seed": lp.get("seed"),
                "kvCacheAttentionBias": lp.get("kv_cache_attention_bias", 0.3),
                "denoisingSteps": lp.get("denoising_step_list"),
                "quantization": lp.get("quantization"),
                "loras": lp.get("loras"),
                # Map to render_timeline.py's expected field name
                "loraMergeStrategy": lp.get("lora_merge_mode"),
            },
            "prompts": segments,
        }

    def save(self, recording: SessionRecording, path: Path) -> Path:
        """Save timeline to JSON file."""
        timeline = self.export_timeline(recording)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(timeline, indent=2))
        return path
```

### 2. Integration with FrameProcessor

**Key design:**
1. Route start/stop through reserved keys, handle in `process_chunk()` for thread-safety
2. **Record from ControlBus events, not `merged_updates`** - keys get popped before we'd see them
3. Detect hard cuts on **edge** (`"reset_cache" in merged_updates`), not persistent state
4. Use `peek_status_info()` to avoid clearing error state, or gate on LOADED

```python
# In src/scope/server/frame_processor.py

import time
from datetime import datetime
from pathlib import Path
from .session_recorder import SessionRecorder

class FrameProcessor:
    def __init__(self, pipeline_manager, ...):
        # ... existing init
        self.session_recorder = SessionRecorder()
        self._pipeline_manager = pipeline_manager  # Need reference for status
        self._last_recording_path: Path | None = None

    def _get_current_effective_prompt(self) -> tuple[str | None, float]:
        """Get current effective prompt for baseline recording.

	        Priority (FIXED review02-C):
	        1. Transition target prompts (if transition active)
	        2. Current prompts from parameters (PREFERRED - this is what pipeline sees)
	        3. Prompts from `pipeline.state` (fallback when parameters omit prompts; e.g. warmup)
	        4. Compiled prompt from style layer (LAST RESORT - complex type)

        Returns:
            (prompt_text, weight) or (None, 1.0)
        """
        # If transition is active, use target as the "current" prompt
        transition = self.parameters.get("transition")
        if transition and "target_prompts" in transition:
            targets = transition["target_prompts"]
            if targets:
                return targets[0].get("text"), targets[0].get("weight", 1.0)

        # PREFERRED: Use current prompts from parameters (what pipeline actually sees)
        # FIXED review02-C: This is more reliable than compiler outputs
	        prompts = self.parameters.get("prompts")
	        if prompts:
	            return prompts[0].get("text"), prompts[0].get("weight", 1.0)

	        # Fallback: pipeline.state may still hold a prompt (e.g. warmup prompt) even if
	        # frame_processor.parameters does not include "prompts" yet.
	        try:
	            pipeline = self.pipeline_manager.get_pipeline()
	        except Exception:
	            pipeline = None
	        if pipeline is not None and hasattr(pipeline, "state"):
	            state = getattr(pipeline, "state", None)
	            state_prompts = None
	            if state is not None and hasattr(state, "get"):
	                state_prompts = state.get("prompts")
	            elif state is not None:
	                state_prompts = getattr(state, "values", {}).get("prompts")
	            if state_prompts:
	                return state_prompts[0].get("text"), state_prompts[0].get("weight", 1.0)

	        # LAST RESORT: compiled prompt from style layer
	        # NOTE: There are (at least) two "CompiledPrompt" shapes in this repo:
	        # - prompt_compiler.CompiledPrompt: `.prompts` is list[PromptEntry] (has `.text`/`.weight`), plus `.prompt` convenience str
	        # - control_state.CompiledPrompt: `.positive` is list[dict] with {"text","weight"}
        if hasattr(self, "_compiled_prompt") and self._compiled_prompt:
            compiled = self._compiled_prompt

            # prompt_compiler.CompiledPrompt: list[PromptEntry] (or dicts in some codepaths)
            prompts = getattr(compiled, "prompts", None)
            if isinstance(prompts, list) and prompts:
                first_item = prompts[0]
                if hasattr(first_item, "text"):
                    return getattr(first_item, "text", None), getattr(first_item, "weight", 1.0)
                if isinstance(first_item, dict):
                    return first_item.get("text"), first_item.get("weight", 1.0)

            # control_state.CompiledPrompt: list[dict]
            positive = getattr(compiled, "positive", None)
            if isinstance(positive, list) and positive:
                first_item = positive[0]
                if isinstance(first_item, dict):
                    return first_item.get("text"), first_item.get("weight", 1.0)

            # Fallback: some compiler objects expose `.prompt` as a convenience string.
            prompt_str = getattr(compiled, "prompt", None)
            if isinstance(prompt_str, str) and prompt_str.strip():
                return prompt_str, 1.0

        return None, 1.0

    def _handle_session_recording_commands(self, merged_updates: dict) -> None:
        """Handle session recording start/stop. Called inside process_chunk."""

        # Start recording
        if "_rcp_session_recording_start" in merged_updates:
            merged_updates.pop("_rcp_session_recording_start")

            # Use peek (non-mutating) or gate on LOADED to avoid clearing errors
            status = self._pipeline_manager.peek_status_info()
            # FIXED review02-B: compare to "loaded" (lowercase .value), not "LOADED"
            if status.get("status") != "loaded":
                logger.warning("Cannot start recording: pipeline not loaded")
                return

            # Snapshot runtime parameters
            runtime_params = {
                "height": self.parameters.get("height"),
                "width": self.parameters.get("width"),
                "seed": self.parameters.get("seed"),
                "kv_cache_attention_bias": self.parameters.get("kv_cache_attention_bias"),
                "denoising_step_list": self.parameters.get("denoising_step_list"),
                **status.get("load_params", {}),
            }

            # Get baseline prompt for t=0 (avoids empty timelines)
            baseline_prompt, baseline_weight = self._get_current_effective_prompt()

            pipeline_id = status.get("pipeline_id")
            if not pipeline_id:
                logger.error("Cannot start recording: unknown pipeline_id")
                return

            self.session_recorder.start(
                chunk_index=self.chunk_index,
                pipeline_id=pipeline_id,
                load_params=runtime_params,
                baseline_prompt=baseline_prompt,
                baseline_weight=baseline_weight,
            )
            logger.info(f"Session recording started at chunk {self.chunk_index}")

        # Stop recording
        if "_rcp_session_recording_stop" in merged_updates:
            merged_updates.pop("_rcp_session_recording_stop")
            recording = self.session_recorder.stop(self.chunk_index)
            if recording:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
                path = Path.home() / ".daydream-scope" / "recordings" / f"session_{timestamp}.timeline.json"
                saved_path = self.session_recorder.save(recording, path)
                logger.info(f"Session recording saved: {saved_path}")
                self._last_recording_path = saved_path

    def _record_control_events_from_bus(
        self,
        applied_events: list,  # List of ControlBus events that were applied
        hard_cut_executed: bool,  # RENAMED review02: True if init_cache was actually passed to pipeline
        soft_cut_bias: float | None = None,
        soft_cut_chunks: int | None = None,
        soft_cut_restore_bias: float | None = None,  # ADDED: review01 Risk E
        soft_cut_restore_was_set: bool = False,  # ADDED: review01 Risk E
        # ADDED per review01 Risk C: fallback prompt from parameter edge detection
        fallback_prompt: str | None = None,
        fallback_prompt_weight: float = 1.0,
    ) -> None:
        """Record control events from ControlBus (not merged_updates).

        IMPORTANT: We record from applied ControlBus events because by the time
        we'd check merged_updates, the 'prompts' and 'transition' keys have been
        popped during translation to events.

        ADDED per review01 Risk C: Also accepts fallback_prompt for cases where
        prompt changes happen via style/LoRA recompilation that don't produce
        SET_PROMPT events.

        Args:
            applied_events: ControlBus events that were applied this chunk
            hard_cut_executed: True if init_cache was actually passed to pipeline
                               (FIXED review02: record execution, not request)
            soft_cut_bias: Captured from _rcp_soft_transition before it was popped
            soft_cut_chunks: Captured from _rcp_soft_transition before it was popped
            soft_cut_restore_bias: Original bias before soft cut (for restore)
            soft_cut_restore_was_set: Whether bias was explicitly set before soft cut
            fallback_prompt: Prompt from parameter edge detection (Risk C mitigation)
            fallback_prompt_weight: Weight for fallback prompt
        """
        if not self.session_recorder.is_recording:
            return

        wall_time = time.monotonic()
        recorded_prompt_event = False

        # Record SET_PROMPT events from ControlBus
        for event in applied_events:
            # FIXED review02-A: attribute is .type, not .event_type
            if event.type == EventType.SET_PROMPT:
                payload = event.payload or {}

                # Extract prompt (prefer explicit prompts, then transition target)
                prompt = None
                prompt_weight = 1.0
                if "prompts" in payload and payload["prompts"]:
                    prompt = payload["prompts"][0].get("text")
                    prompt_weight = payload["prompts"][0].get("weight", 1.0)
                elif "transition" in payload:
                    trans = payload["transition"]
                    if "target_prompts" in trans and trans["target_prompts"]:
                        prompt = trans["target_prompts"][0].get("text")
                        prompt_weight = trans["target_prompts"][0].get("weight", 1.0)

                # Extract transition metadata
                transition_steps = None
                transition_method = None
                if "transition" in payload:
                    trans = payload["transition"]
                    transition_steps = trans.get("num_steps")
                    transition_method = trans.get("temporal_interpolation_method")

                if prompt is not None:
                    self.session_recorder.record_event(
                        chunk_index=self.chunk_index,
                        wall_time=wall_time,
                        prompt=prompt,
                        prompt_weight=prompt_weight,
                        transition_steps=transition_steps,
                        transition_method=transition_method,
                        hard_cut=hard_cut_executed,  # RENAMED review02
                        soft_cut_bias=soft_cut_bias,
                        soft_cut_chunks=soft_cut_chunks,
                        soft_cut_restore_bias=soft_cut_restore_bias,
                        soft_cut_restore_was_set=soft_cut_restore_was_set,
                    )
                    recorded_prompt_event = True
                    # Clear edge flags after first use
                    hard_cut_executed = False
                    soft_cut_bias = None
                    soft_cut_chunks = None

        # ADDED per review01 Risk C: Fallback for prompt changes that don't go through SET_PROMPT
        # (e.g., style changes causing recompilation, LoRA edge triggers)
        if not recorded_prompt_event and fallback_prompt is not None:
            self.session_recorder.record_event(
                chunk_index=self.chunk_index,
                wall_time=wall_time,
                prompt=fallback_prompt,
                prompt_weight=fallback_prompt_weight,
                hard_cut=hard_cut_executed,
                soft_cut_bias=soft_cut_bias,
                soft_cut_chunks=soft_cut_chunks,
                soft_cut_restore_bias=soft_cut_restore_bias,
                soft_cut_restore_was_set=soft_cut_restore_was_set,
            )
            recorded_prompt_event = True
            hard_cut_executed = False
            soft_cut_bias = None

        # If hard/soft cut occurred without a prompt change, record cut-only event
        if not recorded_prompt_event and (hard_cut_executed or soft_cut_bias is not None):
            self.session_recorder.record_event(
                chunk_index=self.chunk_index,
                wall_time=wall_time,
                prompt=None,  # Will use _last_prompt in recorder
                hard_cut=hard_cut_executed,
                soft_cut_bias=soft_cut_bias,
                soft_cut_chunks=soft_cut_chunks,
                soft_cut_restore_bias=soft_cut_restore_bias,
                soft_cut_restore_was_set=soft_cut_restore_was_set,
            )

    def _handle_soft_transition(self, merged_updates: dict) -> tuple[float | None, int | None, float | None, bool]:
        """Handle soft transition reserved key.

        Returns (bias, chunks, restore_bias, restore_was_set) for recording.

        UPDATED review02: Use existing internal state directly instead of approximating.
        The FrameProcessor already tracks _soft_transition_original_bias and
        _soft_transition_original_bias_was_set - use those for fidelity.
        """
        soft_cut_bias = None
        soft_cut_chunks = None
        restore_bias = None
        restore_was_set = False

        if "_rcp_soft_transition" in merged_updates:
            soft = merged_updates.pop("_rcp_soft_transition")
            soft_cut_bias = soft.get("temp_bias")
            soft_cut_chunks = soft.get("num_chunks")

            # FIXED review02: Use existing internal state directly for restore target
            # (The actual soft-cut logic already tracks this - hook into it)
            # NOTE: This assumes we're reading BEFORE the soft-cut is applied
            # If reading AFTER, use _soft_transition_original_bias directly
            if hasattr(self, "_soft_transition_original_bias"):
                restore_bias = self._soft_transition_original_bias
                restore_was_set = getattr(self, "_soft_transition_original_bias_was_set", False)
            else:
                # Fallback if internal state not yet initialized
                current_bias = self.parameters.get("kv_cache_attention_bias")
                restore_was_set = current_bias is not None
                restore_bias = current_bias

            # ... apply soft transition logic (existing code) ...

        return soft_cut_bias, soft_cut_chunks, restore_bias, restore_was_set

    def _detect_prompt_edge(self, prev_prompt: str | None) -> tuple[str | None, float]:
        """Detect prompt changes from parameter edge (Risk C fallback).

        Called to catch prompt changes that don't go through ControlBus,
        such as style/LoRA triggered recompilation.

        Returns (prompt, weight) if changed, (None, 1.0) otherwise.
        """
        current_prompt, weight = self._get_current_effective_prompt()
        if current_prompt != prev_prompt and current_prompt is not None:
            return current_prompt, weight
        return None, 1.0

    def process_chunk(self, ...):
        # ... existing mailbox merge into merged_updates ...

        # 1. Handle recording commands FIRST (before any keys are consumed)
        self._handle_session_recording_commands(merged_updates)

        # 2. Capture soft cut params BEFORE translation (they get popped)
        # NOTE: Hard cut is now recorded AFTER pipeline call (see below)
        soft_cut_bias, soft_cut_chunks, restore_bias, restore_was_set = \
            self._handle_soft_transition(merged_updates)

        # ADDED per review01 Risk C: Snapshot prompt BEFORE processing
        # to detect changes that don't go through ControlBus
        prev_prompt = self.session_recorder._last_prompt if self.session_recorder.is_recording else None

        # 3. Translate to ControlBus events (pops prompts, transition, etc.)
        # ... existing translation logic ...

        # 4. Drain and apply ControlBus events
        applied_events = self.control_bus.drain_pending(...)
        for event in applied_events:
            # ... existing event application ...

        # 5. ADDED per review01 Risk C: Detect prompt edge (fallback for non-ControlBus changes)
        fallback_prompt, fallback_weight = None, 1.0
        if self.session_recorder.is_recording:
            fallback_prompt, fallback_weight = self._detect_prompt_edge(prev_prompt)

        # 6. Existing hard cut handling - note if we're about to pass init_cache
        reset_cache = self.parameters.pop("reset_cache", None)
        # Record a "hard cut" only when explicitly requested (reset_cache=True).
        # Do NOT treat the initial warmup/init path (init_cache=True when !is_prepared) as a hard cut.
        hard_cut_executed = bool(reset_cache)

        # 7. Pipeline call
        # NOTE: FrameProcessor init_cache semantics are:
        #   init_cache = (not self.is_prepared) if reset_cache is None else bool(reset_cache)
        # ... existing pipeline call ...

        # 8. Record events AFTER pipeline call
        # FIXED review02: Record hard cut when init_cache is actually passed to pipeline,
        # not just when reset_cache arrives. This is the moment it truly takes effect.
        self._record_control_events_from_bus(
            applied_events,
            hard_cut_executed=hard_cut_executed,
            soft_cut_bias=soft_cut_bias,
            soft_cut_chunks=soft_cut_chunks,
            soft_cut_restore_bias=restore_bias,
            soft_cut_restore_was_set=restore_was_set,
            fallback_prompt=fallback_prompt,
            fallback_prompt_weight=fallback_weight,
        )

        # ... rest of process_chunk ...
```

**Note:** Requires adding `PipelineManager.peek_status_info()`:
```python
# In src/scope/server/pipeline_manager.py

def peek_status_info(self) -> dict:
    """Non-mutating status read (does NOT clear errors).

    FIXED review02-B: Use .value (lowercase) for status, and _error_message field.
    """
    with self._lock:
        return {
            "status": self._status.value,  # e.g. "loaded", "error" (lowercase)
            "pipeline_id": self._pipeline_id,
            "load_params": self._load_params.copy() if self._load_params else {},
            "error": self._error_message,  # FIXED: field is _error_message, not _error
        }
```

### 3. API Endpoints

```python
# In src/scope/server/app.py

@app.post("/api/v1/realtime/session-recording/start")
async def start_session_recording(
    webrtc_manager: WebRTCManager = Depends(get_webrtc_manager),
):
    """Start recording control events for timeline export.

    Routes through reserved key for thread-safe handling in process_chunk.
    """
    session = get_active_session(webrtc_manager)
    # Route through update_parameters, not direct call
    apply_control_message(session, {"_rcp_session_recording_start": True})
    return {"status": "recording_started"}


@app.post("/api/v1/realtime/session-recording/stop")
async def stop_session_recording(
    webrtc_manager: WebRTCManager = Depends(get_webrtc_manager),
):
    """Request recording stop. Client should poll /status for completion.

    Routes through reserved key. Non-blocking - actual save happens in process_chunk.
    """
    session = get_active_session(webrtc_manager)
    # Route through update_parameters
    apply_control_message(session, {"_rcp_session_recording_stop": True})
    return {"status": "stop_requested"}


@app.get("/api/v1/realtime/session-recording/status")
async def get_session_recording_status(
    webrtc_manager: WebRTCManager = Depends(get_webrtc_manager),
):
    """Check session recording status. Poll after stop to get saved path.

    Uses thread-safe snapshot - never reads _recording directly from FastAPI thread.
    """
    session = get_active_session(webrtc_manager)
    fp = session.video_track.frame_processor

    # Thread-safe read via atomic dict snapshot (GIL guarantees atomic pointer read)
    snapshot = fp.session_recorder.get_status_snapshot()

    response = {
        "is_recording": snapshot.get("is_recording", False),
        "duration_seconds": snapshot.get("duration_seconds", 0),
        "start_chunk": snapshot.get("start_chunk"),
        "events_count": snapshot.get("events_count", 0),
    }

    # Include last saved path if available (after stop completes)
    if fp._last_recording_path:
        response["last_timeline_path"] = str(fp._last_recording_path)

    return response
```

**Client usage pattern:**
```python
# Stop and poll for completion
client.post("/api/v1/realtime/session-recording/stop")

# Poll until recording stops and path appears
while True:
    status = client.get("/api/v1/realtime/session-recording/status").json()
    if not status["is_recording"] and "last_timeline_path" in status:
        print(f"Saved: {status['last_timeline_path']}")
        break
    time.sleep(0.05)
```

### 4. CLI Integration

```python
# In video_cli.py playlist nav, add 'R' key for recording toggle

elif ch == "R":
    if recording_active:
        # Request stop
        client.post("/api/v1/realtime/session-recording/stop")
        click.echo("  ⏹ Stopping session recording...")

        # Poll for completion (with timeout)
        for _ in range(20):  # 1 second max
            time.sleep(0.05)
            status = client.get("/api/v1/realtime/session-recording/status").json()
            if not status.get("is_recording") and "last_timeline_path" in status:
                path = status["last_timeline_path"]
                click.echo(f"  ✓ Saved: {path}")
                break
        else:
            click.echo("  ⚠ Stop requested but save not confirmed")

        recording_active = False
    else:
        r = client.post("/api/v1/realtime/session-recording/start")
        if r.status_code == 200:
            click.echo("  ⏺ Session recording started")
            recording_active = True
```

## Output Format

Compatible with `render_timeline.py` (with noted limitations):

```json
{
  "version": "1.0",
  "exportedAt": "2025-12-26T14:30:52Z",
  "recording": {
    "durationSeconds": 45.2,
    "durationChunks": 180,
    "startChunk": 1024,
    "endChunk": 1204
  },
  "settings": {
    "pipelineId": "krea-realtime-video",
    "resolution": {"height": 480, "width": 832},
    "seed": 42,
    "kvCacheAttentionBias": 0.3,
    "denoisingSteps": [1000, 750, 500, 250],
    "quantization": "fp8",
    "loras": [{"path": "/path/to/lora.safetensors", "scale": 1.0}],
    "loraMergeStrategy": "permanent_merge"
  },
  "prompts": [
    {
      "startTime": 0.0,
      "endTime": 8.5,
      "startChunk": 0,
      "endChunk": 34,
      "prompts": [{"text": "A serene forest at dawn", "weight": 1.0}]
    },
    {
      "startTime": 8.5,
      "endTime": 15.2,
      "startChunk": 34,
      "endChunk": 61,
      "prompts": [{"text": "A busy city street", "weight": 1.0}],
      "transitionSteps": 4,
      "temporalInterpolationMethod": "slerp"
    },
    {
      "startTime": 15.2,
      "endTime": 22.0,
      "startChunk": 61,
      "endChunk": 88,
      "prompts": [{"text": "A quiet bedroom", "weight": 1.0}],
      "initCache": true
    },
    {
      "startTime": 22.0,
      "endTime": 30.0,
      "startChunk": 88,
      "endChunk": 120,
      "prompts": [{"text": "A peaceful garden", "weight": 1.0}],
      "softCut": {
        "bias": 0.8,
        "chunks": 3,
        "restoreBias": 0.3,
        "restoreWasSet": true
      }
    }
  ]
}
```

## Workflow

```
1. Start stream (UI or CLI)
2. video-cli playlist load prompts.txt
3. video-cli playlist nav
4. Press 'R' to start session recording
5. Navigate with arrows, use s/t/h/x as needed
6. Press 'R' to stop
   → Saves ~/.daydream-scope/recordings/session_2025-12-26_143052.timeline.json
7. render-timeline session.timeline.json output.mp4 --preset quality
```

## render_timeline.py Implementation (per review01 sections 3-5)

The review identified concrete changes needed in `render_timeline.py` to support replay.

### Schema Changes

```python
# In src/scope/cli/render_timeline.py

class TimelineSoftCut(BaseModel):
    """Soft cut parameters for temporary KV bias override."""
    model_config = ConfigDict(extra="ignore")
    bias: float
    chunks: int = 2
    # ADDED per review01 Risk E: Restore semantics
    restoreBias: float | None = None  # None means "was unset"
    restoreWasSet: bool = False


class TimelineSegment(BaseModel):
    model_config = ConfigDict(extra="ignore")
    # ... existing fields ...
    initCache: bool | None = None  # ADDED: Hard cut support
    softCut: TimelineSoftCut | None = None  # ADDED: Soft cut support
```

### Hard Cut (initCache) Replay

```python
# In render loop - segment boundary handling

# State for one-shot init_cache
pending_init_cache = False

# On segment change:
if current_segment_id != last_segment_id:
    # ... existing prompt/transition updates ...

    # ADDED: Hard cut support
    if active_segment.initCache:
        pending_init_cache = True

    last_segment_id = current_segment_id

# Before pipeline call:
if pending_init_cache:
    parameters["init_cache"] = True

output = pipeline(**parameters)

# After pipeline call:
if pending_init_cache:
    parameters.pop("init_cache", None)
    pending_init_cache = False
```

### Soft Cut (softCut) Replay

```python
# State machine for soft cuts (mirrors realtime FrameProcessor)
soft_active: bool = False
soft_chunks_remaining: int = 0
soft_temp_bias: float | None = None
soft_restore_bias: float | None = None
soft_restore_was_set: bool = False

# On segment change - start soft cut if specified:
if active_segment.softCut:
    sc = active_segment.softCut
    # Clamp values (match realtime rules)
    temp_bias = max(0.01, min(1.0, sc.bias))
    chunks = max(1, min(10, sc.chunks))

    # Capture restore target ONLY if not already in soft transition
    # (re-trigger doesn't clobber original restore target)
    if not soft_active:
        soft_restore_bias = sc.restoreBias
        soft_restore_was_set = sc.restoreWasSet

    # Start/restart soft transition
    soft_active = True
    soft_chunks_remaining = chunks
    soft_temp_bias = temp_bias
    parameters["kv_cache_attention_bias"] = temp_bias

# After each pipeline call - decrement and restore:
if soft_active:
    soft_chunks_remaining -= 1
    if soft_chunks_remaining <= 0:
        # Restore original bias
        if soft_restore_was_set and soft_restore_bias is not None:
            parameters["kv_cache_attention_bias"] = soft_restore_bias
        else:
            # Restore to "unset" - pop the key entirely
            parameters.pop("kv_cache_attention_bias", None)
        soft_active = False
```

### Dry-Run Output (Optional Enhancement)

Include initCache/softCut in the plan output for debugging:

```python
# In dry-run mode, add to segment info:
if segment.initCache:
    plan_entry["initCache"] = True
if segment.softCut:
    plan_entry["softCut"] = {
        "bias": segment.softCut.bias,
        "chunks": segment.softCut.chunks,
    }
```

---

## Limitations & Follow-ups

### Remaining Fidelity Gaps (post-review01)

1. **Time-based vs chunk-based scheduling** — The renderer uses wall-time for segment selection, but soft cuts are chunk-based. For high-fidelity replay, add optional "chunk timebase mode" using `startChunk/endChunk`.

2. **`num_frame_per_block` assumption** — Dry-run assumes 3 frames per pipeline call. Should read from model config for accuracy.

### Chunk-based vs Wall-clock Timing

- **Chunk-based** (`startChunk`/`endChunk`): Primary for offline render fidelity. Maps directly to pipeline iterations.
- **Wall-clock** (`startTime`/`endTime`): Secondary, affected by GPU stalls/pauses. Useful for human-readable durations.

`render_timeline.py` could use chunk indices for frame-accurate replay, or wall-clock for natural timing (with potential drift).

## Known Limitations (MVP)

Updated per review01 + review02:

| Limitation | Status | Notes |
|------------|--------|-------|
| **ControlBus event.type** | ✅ FIXED | Was `event.event_type`, now `event.type` (review02-A) |
| **Status comparison** | ✅ FIXED | Use `"loaded"` lowercase, field `_error_message` (review02-B) |
| **Baseline prompt type** | ✅ FIXED | Prefer `parameters["prompts"]`; compiler output is not a string (handle `.prompts`/`.prompt`/`.positive` shapes) (review02-C) |
| **Weight scale** | ✅ FIXED | Recorder uses 1.0; note render_timeline.py manual timelines still default 100.0 (review02-D) |
| **Non-ControlBus prompt changes** | ✅ FIXED | Added fallback edge detection (review01-C) |
| **Soft cut restore semantics** | ✅ FIXED | Use internal state `_soft_transition_original_bias` directly (review02) |
| **Hard cut timing** | ✅ FIXED | Record when `reset_cache=True` is actually passed as `init_cache=True`, not on request arrival (review02) |
| **LoRA/style changes** | ⚠️ NOT RECORDED | Style switches won't replay correctly without LoRA scale events (review02 gap) |
| **V2V/VACE sessions** | ⚠️ NOT SUPPORTED | render_timeline.py rejects `inputMode != "text"` (review02 gap) |
| **Recording start mid-transition** | ⚠️ MVP scope | Records target prompt at t=0 (no transition metadata) |
| **Multi-prompt blending** | ⚠️ MVP scope | Currently captures first prompt only |
| **Pipeline reload mid-recording** | ⚠️ MVP scope | Recording continues with stale pipeline_id |

### Edge Cases to Handle

| Edge Case | Behavior |
|-----------|----------|
| Start when no pipeline loaded | Fails with warning, returns early |
| Start when no prompt set | Records baseline as None; timeline will have empty first segment |
| Start mid-transition | Records target prompt at t=0 (without transition metadata) |
| Stop while paused | Works - stop processed on next mailbox drain |
| Pipeline reload mid-recording | Not handled - recording continues with stale pipeline_id |
| Multiple sessions connected | `get_active_session` errors; recording is per-session |
| Save failure (permissions) | Logged as error; `_last_recording_path` not updated |

## Open Questions

1. **Frontend sync?** Could send WebSocket message when server recording starts/stops so frontend can show indicator.

2. **Multiple prompts (blending)?** Currently captures first prompt only. Could extend to capture full prompt list with weights.

3. **Download endpoint?** Current design returns filesystem path (fine for local use). Could add `/api/v1/realtime/session-recording/download` for remote clients.

## Related Files

- `src/scope/server/frame_processor.py` - Main integration point
- `src/scope/server/app.py` - API endpoints
- `src/scope/cli/video_cli.py` - CLI key binding
- `src/scope/cli/render_timeline.py` - Offline renderer (needs initCache/softCut support)
