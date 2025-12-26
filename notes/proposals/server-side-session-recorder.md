# Server-Side Session Recorder

> Status: Draft (updated with Codex feedback)
> Date: 2025-12-26

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

    Thread-safety: All methods are called from within process_chunk(),
    serialized by the mailbox merge. Never call directly from FastAPI threads.
    """

    def __init__(self):
        self._recording: SessionRecording | None = None
        self._last_prompt: str | None = None  # Track for cut-only events

    @property
    def is_recording(self) -> bool:
        return self._recording is not None and self._recording.is_active

    def start(
        self,
        chunk_index: int,
        pipeline_id: str,
        load_params: dict,
    ) -> None:
        """Start recording. Called from process_chunk via reserved key."""
        self._recording = SessionRecording(
            start_chunk=chunk_index,
            start_wall_time=time.monotonic(),
            pipeline_id=pipeline_id,
            load_params=load_params.copy(),
        )
        self._last_prompt = None

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
        return recording

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

**Key design: Route start/stop through reserved keys, handle in process_chunk() for thread-safety.**

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
        self._recording_stop_requested: bool = False

    def _handle_session_recording_commands(self, merged_updates: dict) -> None:
        """Handle session recording start/stop. Called inside process_chunk."""

        # Start recording
        if "_rcp_session_recording_start" in merged_updates:
            merged_updates.pop("_rcp_session_recording_start")
            status = self._pipeline_manager.get_status_info()

            # Snapshot runtime parameters (not just load_params)
            runtime_params = {
                "height": self.parameters.get("height"),
                "width": self.parameters.get("width"),
                "seed": self.parameters.get("seed"),
                "kv_cache_attention_bias": self.parameters.get("kv_cache_attention_bias"),
                "denoising_step_list": self.parameters.get("denoising_step_list"),
                # Include load_params for LoRA info etc
                **status.get("load_params", {}),
            }

            self.session_recorder.start(
                chunk_index=self.chunk_index,
                pipeline_id=status.get("pipeline_id", "unknown"),
                load_params=runtime_params,
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
                # Store path for API to retrieve
                self._last_recording_path = saved_path

    def _record_control_events(
        self,
        merged_updates: dict,
        soft_cut_bias: float | None = None,
        soft_cut_chunks: int | None = None,
    ) -> None:
        """Record control events if session recording is active.

        Args:
            merged_updates: The merged control updates
            soft_cut_bias: Passed from _handle_soft_transition (already popped)
            soft_cut_chunks: Passed from _handle_soft_transition (already popped)
        """
        if not self.session_recorder.is_recording:
            return

        wall_time = time.monotonic()
        prompt = None
        prompt_weight = 100.0
        transition_steps = None
        transition_method = None
        hard_cut = "reset_cache" in merged_updates

        # Prompt from explicit prompts field
        if "prompts" in merged_updates:
            prompts = merged_updates["prompts"]
            if prompts:
                prompt = prompts[0].get("text", "")
                prompt_weight = prompts[0].get("weight", 100.0)

        # Transition - may have target_prompts even without explicit prompts
        if "transition" in merged_updates:
            trans = merged_updates["transition"]
            transition_steps = trans.get("num_steps")
            transition_method = trans.get("temporal_interpolation_method")
            # If no explicit prompt but transition has target_prompts, use those
            if prompt is None and "target_prompts" in trans:
                target = trans["target_prompts"]
                if target:
                    prompt = target[0].get("text", "")
                    prompt_weight = target[0].get("weight", 100.0)

        # Record if we have anything interesting
        if prompt is not None or hard_cut or soft_cut_bias is not None:
            self.session_recorder.record_event(
                chunk_index=self.chunk_index,
                wall_time=wall_time,
                prompt=prompt,
                prompt_weight=prompt_weight,
                transition_steps=transition_steps,
                transition_method=transition_method,
                hard_cut=hard_cut,
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
        # ... existing mailbox merge ...

        # Handle recording commands FIRST (serialized with other updates)
        self._handle_session_recording_commands(merged_updates)

        # Handle soft transition BEFORE it gets popped, capture for recording
        soft_cut_bias, soft_cut_chunks = self._handle_soft_transition(merged_updates)

        # ... existing parameter handling ...

        # Record events - pass soft cut info since it was already popped
        self._record_control_events(merged_updates, soft_cut_bias, soft_cut_chunks)

        # ... rest of process_chunk ...
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
    """Check session recording status. Poll after stop to get saved path."""
    session = get_active_session(webrtc_manager)
    fp = session.video_track.frame_processor
    rec = fp.session_recorder._recording

    response = {
        "is_recording": fp.session_recorder.is_recording,
        "duration_seconds": rec.duration_seconds if rec else 0,
        "duration_chunks": (fp.chunk_index - rec.start_chunk) if rec else 0,
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

## Open Questions

1. **Frontend sync?** Could send WebSocket message when server recording starts/stops so frontend can show indicator.

2. **Multiple prompts (blending)?** Currently captures first prompt only. Could extend to capture full prompt list with weights.

3. **Download endpoint?** Current design returns filesystem path (fine for local use). Could add `/api/v1/realtime/session-recording/download` for remote clients.

## Related Files

- `src/scope/server/frame_processor.py` - Main integration point
- `src/scope/server/app.py` - API endpoints
- `src/scope/cli/video_cli.py` - CLI key binding
- `src/scope/cli/render_timeline.py` - Offline renderer (needs initCache/softCut support)
