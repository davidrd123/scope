# Server-Side Session Recorder

> Status: Draft
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
                 │  └───────────────┘  │
                 └─────────────────────┘
```

## Events to Capture

| Event | Source | Data |
|-------|--------|------|
| Prompt change | `prompts` in control message | prompt text, weight |
| Transition start | `transition` in control message | target_prompts, num_steps, method |
| Hard cut | `reset_cache` in control message | (flag only) |
| Soft cut | `_rcp_soft_transition` reserved key | temp_bias, num_chunks |
| Soft cut restore | Auto at chunk boundary | (flag only) |

## Implementation

### 1. SessionRecorder Class

```python
# src/scope/server/session_recorder.py

from dataclasses import dataclass, field
from datetime import datetime
import time
import json
from pathlib import Path

@dataclass
class ControlEvent:
    """A single control event during recording."""
    timestamp: float  # Seconds since recording started
    chunk_index: int  # Pipeline chunk when event occurred

    # Prompt
    prompt: str | None = None
    prompt_weight: float = 100.0

    # Transition
    transition_steps: int | None = None
    transition_method: str | None = None  # "linear" or "slerp"

    # Cuts
    hard_cut: bool = False
    soft_cut_bias: float | None = None
    soft_cut_chunks: int | None = None


@dataclass
class SessionRecording:
    """Container for a complete recording session."""
    events: list[ControlEvent] = field(default_factory=list)
    start_time: float | None = None
    start_chunk: int = 0
    settings_snapshot: dict = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        return self.start_time is not None

    @property
    def duration(self) -> float:
        if self.start_time is None:
            return 0.0
        return time.monotonic() - self.start_time


class SessionRecorder:
    """Records control events during a streaming session."""

    def __init__(self):
        self._recording: SessionRecording | None = None

    @property
    def is_recording(self) -> bool:
        return self._recording is not None and self._recording.is_active

    def start(self, settings: dict, chunk_index: int = 0) -> None:
        """Start recording. Captures settings snapshot."""
        self._recording = SessionRecording(
            start_time=time.monotonic(),
            start_chunk=chunk_index,
            settings_snapshot=settings.copy(),
        )

    def record_event(
        self,
        chunk_index: int,
        prompt: str | None = None,
        prompt_weight: float = 100.0,
        transition_steps: int | None = None,
        transition_method: str | None = None,
        hard_cut: bool = False,
        soft_cut_bias: float | None = None,
        soft_cut_chunks: int | None = None,
    ) -> None:
        """Record a control event."""
        if not self.is_recording:
            return

        self._recording.events.append(ControlEvent(
            timestamp=self._recording.duration,
            chunk_index=chunk_index,
            prompt=prompt,
            prompt_weight=prompt_weight,
            transition_steps=transition_steps,
            transition_method=transition_method,
            hard_cut=hard_cut,
            soft_cut_bias=soft_cut_bias,
            soft_cut_chunks=soft_cut_chunks,
        ))

    def stop(self) -> SessionRecording | None:
        """Stop recording and return the recording."""
        if not self.is_recording:
            return None

        recording = self._recording
        self._recording = None
        return recording

    def export_timeline(self, recording: SessionRecording) -> dict:
        """Convert recording to render_timeline.py compatible format."""
        segments = []
        duration = recording.duration

        for i, event in enumerate(recording.events):
            if event.prompt is None:
                continue

            # End time is next prompt event or recording end
            end_time = duration
            for j in range(i + 1, len(recording.events)):
                if recording.events[j].prompt is not None:
                    end_time = recording.events[j].timestamp
                    break

            segment = {
                "startTime": event.timestamp,
                "endTime": end_time,
                "prompts": [{"text": event.prompt, "weight": event.prompt_weight}],
            }

            # Transition info
            if event.transition_steps is not None and event.transition_steps > 0:
                segment["transitionSteps"] = event.transition_steps
            if event.transition_method:
                segment["temporalInterpolationMethod"] = event.transition_method

            # Hard cut → initCache (for offline renderer)
            if event.hard_cut:
                segment["initCache"] = True

            # Soft cut (optional, needs render_timeline.py support)
            if event.soft_cut_bias is not None:
                segment["softCut"] = {
                    "bias": event.soft_cut_bias,
                    "chunks": event.soft_cut_chunks or 2,
                }

            segments.append(segment)

        # Build settings from snapshot
        settings = recording.settings_snapshot

        return {
            "version": "1.0",
            "exportedAt": datetime.utcnow().isoformat() + "Z",
            "recordingDuration": duration,
            "settings": {
                "pipelineId": "krea-realtime-video",
                "resolution": {
                    "height": settings.get("height", 480),
                    "width": settings.get("width", 832),
                },
                "seed": settings.get("seed"),
                "kvCacheAttentionBias": settings.get("kv_cache_attention_bias", 0.3),
                "denoisingSteps": settings.get("denoising_step_list"),
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

```python
# In src/scope/server/frame_processor.py

class FrameProcessor:
    def __init__(self, ...):
        # ... existing init
        self.session_recorder = SessionRecorder()

    def start_session_recording(self) -> None:
        """Start recording control events."""
        settings = {
            "height": self.parameters.get("height"),
            "width": self.parameters.get("width"),
            "seed": self.parameters.get("seed"),
            "kv_cache_attention_bias": self.parameters.get("kv_cache_attention_bias"),
            "denoising_step_list": self.parameters.get("denoising_step_list"),
        }
        self.session_recorder.start(settings, self.chunk_index)

    def stop_session_recording(self) -> Path | None:
        """Stop recording and save timeline."""
        recording = self.session_recorder.stop()
        if recording is None:
            return None

        # Save alongside video recordings
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        path = Path.home() / ".daydream-scope" / "recordings" / f"session_{timestamp}.timeline.json"
        return self.session_recorder.save(recording, path)
```

### 3. Hook into Control Message Handler

In the `_handle_realtime_control_parameters` method (or wherever prompts are processed):

```python
# When prompts change
if "prompts" in updates:
    prompt_text = updates["prompts"][0].get("text", "")
    prompt_weight = updates["prompts"][0].get("weight", 100.0)

    # Check for transition
    transition = updates.get("transition")
    transition_steps = None
    transition_method = None
    if transition:
        transition_steps = transition.get("num_steps")
        transition_method = transition.get("temporal_interpolation_method")

    self.session_recorder.record_event(
        chunk_index=self.chunk_index,
        prompt=prompt_text,
        prompt_weight=prompt_weight,
        transition_steps=transition_steps,
        transition_method=transition_method,
        hard_cut="reset_cache" in updates,
    )

# When soft cut triggers
if "_rcp_soft_transition" in updates:
    soft = updates["_rcp_soft_transition"]
    self.session_recorder.record_event(
        chunk_index=self.chunk_index,
        soft_cut_bias=soft.get("temp_bias"),
        soft_cut_chunks=soft.get("num_chunks"),
    )
```

### 4. API Endpoints

```python
# In src/scope/server/app.py

@app.post("/api/v1/realtime/session-recording/start")
async def start_session_recording(
    webrtc_manager: WebRTCManager = Depends(get_webrtc_manager),
):
    """Start recording control events for timeline export."""
    session = get_active_session(webrtc_manager)
    fp = session.video_track.frame_processor
    fp.start_session_recording()
    return {"status": "recording_started"}


@app.post("/api/v1/realtime/session-recording/stop")
async def stop_session_recording(
    webrtc_manager: WebRTCManager = Depends(get_webrtc_manager),
):
    """Stop recording and export timeline."""
    session = get_active_session(webrtc_manager)
    fp = session.video_track.frame_processor
    path = fp.stop_session_recording()

    if path is None:
        return {"status": "not_recording"}

    return {
        "status": "recording_stopped",
        "timeline_path": str(path),
    }


@app.get("/api/v1/realtime/session-recording/status")
async def get_session_recording_status(
    webrtc_manager: WebRTCManager = Depends(get_webrtc_manager),
):
    """Check if session recording is active."""
    session = get_active_session(webrtc_manager)
    fp = session.video_track.frame_processor
    return {
        "is_recording": fp.session_recorder.is_recording,
        "duration": fp.session_recorder._recording.duration if fp.session_recorder.is_recording else 0,
    }
```

### 5. CLI Integration

```python
# In video_cli.py playlist nav, add 'R' key for recording toggle

elif ch == "R":
    if recording_active:
        r = client.post("/api/v1/realtime/session-recording/stop")
        if r.status_code == 200:
            data = r.json()
            click.echo(f"  ⏹ Session recording stopped: {data.get('timeline_path')}")
            recording_active = False
    else:
        r = client.post("/api/v1/realtime/session-recording/start")
        if r.status_code == 200:
            click.echo("  ⏺ Session recording started")
            recording_active = True
```

## Output Format

Compatible with `render_timeline.py`:

```json
{
  "version": "1.0",
  "exportedAt": "2025-12-26T14:30:52Z",
  "recordingDuration": 45.2,
  "settings": {
    "pipelineId": "krea-realtime-video",
    "resolution": {"height": 480, "width": 832},
    "seed": 42,
    "kvCacheAttentionBias": 0.3,
    "denoisingSteps": [1000, 750, 500, 250]
  },
  "prompts": [
    {
      "startTime": 0.0,
      "endTime": 8.5,
      "prompts": [{"text": "A serene forest at dawn", "weight": 100.0}]
    },
    {
      "startTime": 8.5,
      "endTime": 15.2,
      "prompts": [{"text": "A busy city street", "weight": 100.0}],
      "transitionSteps": 4,
      "temporalInterpolationMethod": "slerp"
    },
    {
      "startTime": 15.2,
      "endTime": 22.0,
      "prompts": [{"text": "A quiet bedroom", "weight": 100.0}],
      "initCache": true
    }
  ]
}
```

## Workflow

```
1. Start stream
2. video-cli playlist load prompts.txt
3. video-cli playlist nav
4. Press 'R' to start session recording
5. Navigate with arrows, use s/t/h/x as needed
6. Press 'R' to stop
   → Saves ~/.daydream-scope/recordings/session_2025-12-26_143052.timeline.json
7. render-timeline session.timeline.json output.mp4 --preset quality
```

## render_timeline.py Changes Needed

To support `initCache` (hard cuts):

```python
# In render loop, when entering a new segment:
if segment.initCache:
    parameters["init_cache"] = True
    # Run one pipeline call with init_cache, then remove it
    output = pipeline(**parameters)
    parameters.pop("init_cache", None)
```

## Open Questions

1. **Sync with browser recording?** Could add WebSocket message when server recording starts/stops so frontend can sync its recording.

2. **Chunk-based vs time-based?** Currently using wall-clock time. Could also track chunk indices for more precise offline replay.

3. **Multiple prompts (blending)?** Current spec captures first prompt only. Could extend to capture full prompt list with weights.

## Related Files

- `src/scope/server/frame_processor.py` - Main integration point
- `src/scope/server/app.py` - API endpoints
- `src/scope/cli/video_cli.py` - CLI key binding
- `src/scope/cli/render_timeline.py` - Offline renderer (needs initCache support)
