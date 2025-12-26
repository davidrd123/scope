# Proposal: Session Recording with Timeline Export

> Status: Implemented (MVP), follow-ups pending
> Date: 2025-12-26

## Problem

Right now we have the *pieces*, but not the “bundle”:

- We can **record the WebRTC stream in the browser** (via `MediaRecorder`).
- We can **edit/export a prompt timeline** in the UI (download a timeline JSON).
- We can **offline re-render** a timeline via `src/scope/cli/render_timeline.py`.

What’s missing is the “one button → two artifacts” workflow:
- When you stop a recording, you also want a **timeline export clipped to that exact recording window**, so you can re-render the same “performance” at higher quality without manual bookkeeping.

## Proposal

When recording, also export a timeline JSON that represents the same session window.

On stop, export both:
1. **Video file** — the realtime capture (as today; browser download)
2. **Timeline JSON** — re-renderable with `src/scope/cli/render_timeline.py` at any quality preset

## MVP Implementation (Frontend-only)

Shipped behavior (as of 2025-12-26):

- When a recording stops, the UI downloads a second file: `${filenameBase}.timeline.json`.
- The exported timeline is **clipped** to the recording window and **normalized** to start at `startTime=0`.
- Segment boundaries are derived from prompt startTime events (“hold last prompt until next change”).
- If recording starts mid-segment, the first exported segment does **not** replay that segment’s transition at `t=0`.
- The export includes extra `recording` metadata; `render_timeline.py` ignores unknown fields (`extra="ignore"`).

Implementation locations:

- `frontend/src/hooks/useStreamRecorder.ts` — adds `onRecordingSaved` and locks a deterministic `filenameBase` at recording start.
- `frontend/src/pages/StreamPage.tsx` — builds the clipped timeline payload at stop time and downloads it when the video save completes.

## Use Case

```
1. Creative exploration session (realtime, 480p, fast)
   - Load playlist, navigate prompts
   - Experiment with soft cuts, transitions, timing
   - Find the "performance" you like

2. Hit record, do the performance

3. Stop recording → get:
   - session_2025-12-26_143052.webm/mp4      (realtime capture)
   - session_2025-12-26_143052.timeline.json (timeline export, clipped + normalized)

4. Re-render at quality:
   python -m scope.cli.render_timeline session.timeline.json output_hq.mp4 --preset quality
```

## Timeline Format

Already exists in `src/scope/cli/render_timeline.py` and is already used by the UI export in `frontend/src/components/PromptTimeline.tsx`.

### Key mapping (what we should capture)

| Live Event | Timeline Representation |
|------------|------------------------|
| Prompt applied | New `TimelineSegment` starts |
| Transition mode | `transitionSteps`, `temporalInterpolationMethod` |
| Time passes | Segment `endTime` extends until next prompt event |

### Important nuance: “hard cut” is not “no transition”

- `transitionSteps: 0` / `--no-transitions` only disables **embedding interpolation**.
- A realtime **hard cut** is a **cache reset** (`reset_cache`) which becomes `init_cache` in the pipeline call (see `src/scope/server/frame_processor.py`).

If we want offline re-render to reproduce hard cuts, the timeline needs an explicit cache-reset signal (e.g. a per-segment `initCache: true` or `resetCache: true`), and `render_timeline.py` needs to apply it.

### Example Output

```json
{
  "version": "1.0",
  "exportedAt": "2025-12-26T14:30:52Z",
  "recordingDuration": 45.2,
  "settings": {
    "pipelineId": "krea-realtime-video",
    "resolution": {"height": 480, "width": 832},
    "seed": 42,
    "kvCacheAttentionBias": 0.3
  },
  "prompts": [
    {
      "startTime": 0.0,
      "endTime": 8.5,
      "prompts": [{"text": "A serene forest at dawn, mist rising", "weight": 100.0}]
    },
    {
      "startTime": 8.5,
      "endTime": 15.2,
      "prompts": [{"text": "A busy city street at noon, crowds moving", "weight": 100.0}],
      "transitionSteps": 4,
      "temporalInterpolationMethod": "slerp"
    },
    {
      "startTime": 15.2,
      "endTime": 22.0,
      "prompts": [{"text": "A quiet bedroom at night, moonlight", "weight": 100.0}],
      "transitionSteps": 0,
      "initCache": true
    }
  ]
}
```

## Implementation

There are two implementation paths. The recommended MVP is frontend-only (no backend changes).

### Option A (Recommended MVP): Frontend-only export on stop

**Why:** recording is already client-side; the timeline already exists client-side; we can export both artifacts together without introducing server state.

Where to implement:
- Recording: `frontend/src/hooks/useStreamRecorder.ts`
- Timeline export: `frontend/src/pages/StreamPage.tsx` (record button stop handler)

What to add:
- On recording start: capture `recordingStartTimeSec` from timeline playback (or `performance.now()` baseline).
- On recording stop: export a **clipped timeline**:
  - include only segments overlapping `[t0, t1]`
  - clamp segment boundaries to `[t0, t1]`
  - subtract `t0` so the timeline starts at `startTime=0`
- Download it as `${filenameBase}.timeline.json` alongside the video file.

This solves the core workflow even before we support “hard cut” or “soft cut” faithfully.

### Option B (Future): Server-side event capture

This path is useful if we later move recording server-side, or if we decide we want the timeline to reflect *exactly* what the generator applied (including reserved keys / control-plane details).

Sketch only (not implemented):

```python
# src/scope/server/session_recorder.py

@dataclass
class RecordedEvent:
    timestamp: float          # Seconds since recording started
    prompt: str | None
    transition_steps: int | None
    transition_method: str | None
    hard_cut: bool
    soft_cut_bias: float | None
    soft_cut_chunks: int | None

class SessionRecorder:
    def __init__(self):
        self.events: list[RecordedEvent] = []
        self.start_time: float | None = None
        self.settings_snapshot: dict = {}

    def start(self, settings: dict):
        """Called when recording begins."""
        self.start_time = time.monotonic()
        self.events = []
        self.settings_snapshot = settings.copy()

    @property
    def is_recording(self) -> bool:
        return self.start_time is not None

    def record_event(
        self,
        prompt: str | None = None,
        transition_steps: int | None = None,
        transition_method: str | None = None,
        hard_cut: bool = False,
        soft_cut_bias: float | None = None,
        soft_cut_chunks: int | None = None,
    ):
        """Record a control event with current timestamp."""
        if self.start_time is None:
            return  # Not recording

        self.events.append(RecordedEvent(
            timestamp=time.monotonic() - self.start_time,
            prompt=prompt,
            transition_steps=transition_steps,
            transition_method=transition_method,
            hard_cut=hard_cut,
            soft_cut_bias=soft_cut_bias,
            soft_cut_chunks=soft_cut_chunks,
        ))

    def stop(self) -> float:
        """Called when recording ends. Returns duration."""
        if self.start_time is None:
            return 0.0
        duration = time.monotonic() - self.start_time
        self.start_time = None
        return duration

    def export_timeline(self, duration: float) -> dict:
        """Convert recorded events to TimelineFile format."""
        segments = []

        for i, event in enumerate(self.events):
            if event.prompt is None:
                continue

            # End time is next event's start, or recording end
            if i + 1 < len(self.events):
                end_time = self.events[i + 1].timestamp
            else:
                end_time = duration

            segment = {
                "startTime": event.timestamp,
                "endTime": end_time,
                "text": event.prompt,
            }

            if event.transition_steps is not None and event.transition_steps > 0:
                segment["transitionSteps"] = event.transition_steps
            if event.transition_method:
                segment["temporalInterpolationMethod"] = event.transition_method
            if event.soft_cut_bias is not None:
                segment["softCut"] = {
                    "bias": event.soft_cut_bias,
                    "chunks": event.soft_cut_chunks or 2,
                }

            segments.append(segment)

        return {
            "version": "1.0",
            "exportedAt": datetime.utcnow().isoformat() + "Z",
            "recordingDuration": duration,
            "settings": {
                "pipelineId": "krea-realtime-video",
                "resolution": {
                    "height": self.settings_snapshot.get("height", 480),
                    "width": self.settings_snapshot.get("width", 832),
                },
                "seed": self.settings_snapshot.get("seed"),
                "kvCacheAttentionBias": self.settings_snapshot.get("kv_cache_attention_bias", 0.3),
            },
            "prompts": segments,
        }
```

### Gaps to solve either way

- **Hard cuts:** require an explicit cache reset signal in the timeline + support in `render_timeline.py` (since “no transition” isn’t a cache reset).
- **Soft cuts:** optional; needs a timeline field and offline application semantics (likely “temporarily lower `kv_cache_attention_bias` for N pipeline calls/chunks”).

### `render_timeline.py` Enhancements (optional, but needed for faithful hard cuts)

To support cache resets and soft cuts in offline rendering:

```python
class TimelineSegment(BaseModel):
    # ... existing fields
    initCache: bool | None = None  # True = reset cache at this segment boundary
    softCut: dict | None = None    # {"bias": 0.1, "chunks": 4}
```

Then in the render loop:
- if `initCache` is true when entering a segment, call the pipeline once with `init_cache=True` (mirrors server hard cut behavior)
- apply soft cut by temporarily adjusting `kv_cache_attention_bias`

## File Outputs

Recording session produces:
```
Downloads in browser (MVP path):
├── session_2025-12-26_143052.webm/mp4       # Realtime video
└── session_2025-12-26_143052.timeline.json  # Timeline (for re-render)

(If we later do server-side recording, we can also write under `~/.daydream-scope/recordings/`.)
```

## Quality Presets Reference

From `render_timeline.py`:

| Preset | Resolution | Steps | VRAM | Use Case |
|--------|-----------|-------|------|----------|
| preview | 320x576 | 4 | ~32GB | Quick iteration |
| standard | 480x832 | 4 | ~40GB | Balanced |
| quality | 480x832 | 6 | ~48GB | Offline quality |
| highres | 720x1280 | 4 | ~48GB | HD output |
| max | 720x1280 | 6 | 80GB+ | Maximum quality |

## Edge Cases

| Case | Behavior |
|------|----------|
| No prompt changes during recording | Single segment spanning full duration |
| Recording started mid-session | First event is current prompt at t=0 |
| Very short segments (<0.5s) | Keep them (user intended quick cuts) |
| Overlapping soft cut + transition | Record both, let renderer apply both (if supported) |

## Future Enhancements

1. **Edit timeline before re-render** - Adjust timing, prompts in JSON
2. **Timeline preview** - Visualize segments before rendering
3. **Merge timelines** - Combine multiple recording sessions
4. **Import from other tools** - Convert from video editing timeline formats
5. **Transition prompts** - Include `>` prefixed prompts (per other proposal)

## Open Questions

1. **Soft cut in offline mode**: Should `render_timeline` support soft cuts, or just transitions?
   - Could simulate by varying `kv_cache_attention_bias` per segment

2. **Initial prompt**: If recording starts with no prompt set, what to do?
   - Option A: Require prompt before recording
   - Option B: Use placeholder "unset" segment

3. **LoRA switches**: Record LoRA/style changes too?
   - Would need `TimelineSegment.loras` field

4. **Resolution mismatch**: Record at 480p, render at 720p - aspect ratio handling?

## Implementation Order

**MVP (frontend-only)**
1. Export a clipped/normalized timeline on recording stop (ties `useStreamRecorder` + `PromptTimeline.handleExport`)
2. Match filenames (`recording-...` ↔ `recording-....timeline.json`)
3. (Optional) Add `recordingDuration` and other metadata fields (renderer ignores extras)

**Follow-ups (only if needed)**
4. Add `initCache`/`softCut` support to `src/scope/cli/render_timeline.py`
5. If we ever do server-side recording: add server-side event capture and export (Option B)

## Related

- `src/scope/cli/render_timeline.py` - Existing offline renderer
- `frontend/src/hooks/useStreamRecorder.ts` - Browser recording
- `frontend/src/components/PromptTimeline.tsx` - Timeline export format + download
- `src/scope/server/frame_processor.py` - Hard cut / cache reset mapping (`reset_cache` → `init_cache`)
- Transition prompts proposal - Could be captured in timeline too
