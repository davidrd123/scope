# Server-Side Session Recorder

> Status: Draft (hardened per DeepResearch review)
> Date: 2025-12-26
> Review: `notes/research/2025-12-26/session_recorder/oai_5pro_01.md`

## Purpose

Capture all control events at the server level, regardless of source (CLI, API, frontend). This complements the frontend-only approach which only sees UI-driven changes.

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
| Hard cut | `reset_cache` in control message | flag (may occur without prompt change) |
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
    prompt_weight: float = 100.0

    # Transition
    transition_steps: int | None = None
    transition_method: str | None = None  # "linear" or "slerp"

    # Cuts (can occur with or without prompt change)
    hard_cut: bool = False
    soft_cut_bias: float | None = None
    soft_cut_chunks: int | None = None


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
        prompt_weight: float = 100.0,
        transition_steps: int | None = None,
        transition_method: str | None = None,
        hard_cut: bool = False,
        soft_cut_bias: float | None = None,
        soft_cut_chunks: int | None = None,
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

            # Soft cut
            # NOTE: render_timeline.py does NOT yet support softCut
            if event.soft_cut_bias is not None:
                segment["softCut"] = {
                    "bias": event.soft_cut_bias,
                    "chunks": event.soft_cut_chunks or 2,
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

        Priority:
        1. Transition target prompts (if transition active)
        2. Current prompts
        3. Compiled prompt from style layer

        Returns:
            (prompt_text, weight) or (None, 1.0)
        """
        # If transition is active, use target as the "current" prompt
        transition = self.parameters.get("transition")
        if transition and "target_prompts" in transition:
            targets = transition["target_prompts"]
            if targets:
                return targets[0].get("text"), targets[0].get("weight", 1.0)

        # Otherwise use current prompts
        prompts = self.parameters.get("prompts")
        if prompts:
            return prompts[0].get("text"), prompts[0].get("weight", 1.0)

        # Fallback: compiled prompt from style layer (if available)
        if hasattr(self, "_compiled_prompt") and self._compiled_prompt:
            return self._compiled_prompt, 1.0

        return None, 1.0

    def _handle_session_recording_commands(self, merged_updates: dict) -> None:
        """Handle session recording start/stop. Called inside process_chunk."""

        # Start recording
        if "_rcp_session_recording_start" in merged_updates:
            merged_updates.pop("_rcp_session_recording_start")

            # Use peek (non-mutating) or gate on LOADED to avoid clearing errors
            status = self._pipeline_manager.peek_status_info()
            if status.get("status") != "LOADED":
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
        hard_cut_requested: bool,  # Edge flag from merged_updates
        soft_cut_bias: float | None = None,
        soft_cut_chunks: int | None = None,
    ) -> None:
        """Record control events from ControlBus (not merged_updates).

        IMPORTANT: We record from applied ControlBus events because by the time
        we'd check merged_updates, the 'prompts' and 'transition' keys have been
        popped during translation to events.

        Args:
            applied_events: ControlBus events that were applied this chunk
            hard_cut_requested: True if "reset_cache" was in merged_updates (edge, not persistent)
            soft_cut_bias: Captured from _rcp_soft_transition before it was popped
            soft_cut_chunks: Captured from _rcp_soft_transition before it was popped
        """
        if not self.session_recorder.is_recording:
            return

        wall_time = time.monotonic()
        recorded_prompt_event = False

        # Record SET_PROMPT events from ControlBus
        for event in applied_events:
            if event.event_type == EventType.SET_PROMPT:
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
                        hard_cut=hard_cut_requested,
                        soft_cut_bias=soft_cut_bias,
                        soft_cut_chunks=soft_cut_chunks,
                    )
                    recorded_prompt_event = True
                    # Clear edge flags after first use
                    hard_cut_requested = False
                    soft_cut_bias = None
                    soft_cut_chunks = None

        # If hard/soft cut occurred without a prompt change, record cut-only event
        if not recorded_prompt_event and (hard_cut_requested or soft_cut_bias is not None):
            self.session_recorder.record_event(
                chunk_index=self.chunk_index,
                wall_time=wall_time,
                prompt=None,  # Will use _last_prompt in recorder
                hard_cut=hard_cut_requested,
                soft_cut_bias=soft_cut_bias,
                soft_cut_chunks=soft_cut_chunks,
            )

    def _handle_soft_transition(self, merged_updates: dict) -> tuple[float | None, int | None]:
        """Handle soft transition reserved key. Returns (bias, chunks) for recording."""
        soft_cut_bias = None
        soft_cut_chunks = None

        if "_rcp_soft_transition" in merged_updates:
            soft = merged_updates.pop("_rcp_soft_transition")
            soft_cut_bias = soft.get("temp_bias")
            soft_cut_chunks = soft.get("num_chunks")
            # ... apply soft transition logic ...

        return soft_cut_bias, soft_cut_chunks

    def process_chunk(self, ...):
        # ... existing mailbox merge into merged_updates ...

        # 1. Handle recording commands FIRST (before any keys are consumed)
        self._handle_session_recording_commands(merged_updates)

        # 2. Capture edge flags BEFORE translation (they get popped)
        hard_cut_requested = "reset_cache" in merged_updates
        soft_cut_bias, soft_cut_chunks = self._handle_soft_transition(merged_updates)

        # 3. Translate to ControlBus events (pops prompts, transition, etc.)
        # ... existing translation logic ...

        # 4. Drain and apply ControlBus events
        applied_events = self.control_bus.drain_pending(...)
        for event in applied_events:
            # ... existing event application ...

        # 5. Record events AFTER application (from ControlBus, not merged_updates)
        self._record_control_events_from_bus(
            applied_events,
            hard_cut_requested,
            soft_cut_bias,
            soft_cut_chunks,
        )

        # ... rest of process_chunk (pause check, pipeline call, etc.) ...
```

**Note:** Requires adding `PipelineManager.peek_status_info()`:
```python
# In src/scope/server/pipeline_manager.py

def peek_status_info(self) -> dict:
    """Non-mutating status read (does NOT clear errors)."""
    return {
        "status": self._status.name,
        "pipeline_id": self._pipeline_id,
        "load_params": self._load_params.copy() if self._load_params else {},
        "error": self._error,  # Return but don't clear
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
      "prompts": [{"text": "A serene forest at dawn", "weight": 100.0}]
    },
    {
      "startTime": 8.5,
      "endTime": 15.2,
      "startChunk": 34,
      "endChunk": 61,
      "prompts": [{"text": "A busy city street", "weight": 100.0}],
      "transitionSteps": 4,
      "temporalInterpolationMethod": "slerp"
    },
    {
      "startTime": 15.2,
      "endTime": 22.0,
      "startChunk": 61,
      "endChunk": 88,
      "prompts": [{"text": "A quiet bedroom", "weight": 100.0}],
      "initCache": true
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

## Limitations & Follow-ups

### render_timeline.py Does NOT Yet Support:

1. **`initCache` (hard cuts)** - Would need:
   ```python
   if segment.initCache:
       parameters["init_cache"] = True
       output = pipeline(**parameters)
       parameters.pop("init_cache", None)
   ```

2. **`softCut`** - Would need temporary `kv_cache_attention_bias` adjustment for N chunks

These are separate milestones. Current implementation records the events, but faithful offline replay requires render_timeline.py enhancements.

### Chunk-based vs Wall-clock Timing

- **Chunk-based** (`startChunk`/`endChunk`): Primary for offline render fidelity. Maps directly to pipeline iterations.
- **Wall-clock** (`startTime`/`endTime`): Secondary, affected by GPU stalls/pauses. Useful for human-readable durations.

`render_timeline.py` could use chunk indices for frame-accurate replay, or wall-clock for natural timing (with potential drift).

## Known Limitations (MVP)

Per hardening review, these are explicit scope boundaries:

| Limitation | Notes |
|------------|-------|
| **Parameter changes beyond prompts/cuts** | Denoise steps, seeds, LoRA scale changes are NOT captured unless they result in SET_PROMPT events |
| **Soft cut replay** | Recorded but not replayable until `render_timeline.py` supports it. Even then, needs restore-target nuance (`soft_cut_restore_bias`, `soft_cut_restore_was_set`) |
| **Hard cut replay** | Recorded as `initCache`, ignored by renderer until implemented |
| **Recording start mid-transition** | We record transition target prompt as baseline (no transition at t=0) - approximation |
| **Multi-prompt blending** | Currently captures first prompt only |
| **World state / style changes** | Only captured if they result in prompt recompilation (SET_PROMPT) |

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
