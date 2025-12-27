# Output Frame History & Timeline Scrubbing

> Status: Draft
> Date: 2025-12-26
> Updated: 2025-12-27
> Reviews:
> - `notes/proposals/frame-buffer-scrubbing/review01.md`
> Related:
> - `notes/realtime_video_architecture.md` (FrameBus / ChunkStore)
> - `notes/proposals/session-recording-timeline-export.md`
> - `notes/proposals/server-side-session-recorder.md`

## Summary

Add a server-side, in-memory **output frame history** (ring buffer keyed by `chunk_index`) so the UI can scrub/replay recent generated output **without re-rendering**. This complements the session recorder (control events) by storing **pixels** for instant preview.

## Problem

Today, generated frames mostly flow through a **destructive** queue to WebRTC and are gone. There's no way to:
- Scroll back through what was generated
- Replay a segment without re-rendering
- Visually preview a branch point before forking
- Scrub a timeline of actual output (not just control events)

The session recording proposals capture **control events** for offline re-render, but don't store **actual frames** for instant playback.

## Solution

Add a per-session output frame history buffer (ring buffer) that stores rendered frames, enabling:
- Instant scrubbing through generated history
- Replay without GPU cost
- Visual branch point selection
- Timeline thumbnails / filmstrip UI

## Architecture
### Naming (important)

`FrameProcessor.frame_buffer` already exists and refers to the **input** video buffer (`VideoFrame` deque). This proposal adds a separate **output** retention layer for generated frames and intentionally avoids the name `frame_buffer` in code to prevent collisions.

### Proposed frame flow (producer-side capture)

```
Pipeline
  └─► FrameProcessor.process_chunk()
        ├─► output_queue (destructive) ──► WebRTC
        └─► output_frame_history (ring buffer) ──► REST (scrub/thumbnails) + snapshot previews
```

### Canonical frame format (MVP)

Capture frames **after** `FrameProcessor` normalizes the pipeline output (this is the same format used by `/api/v1/realtime/frame/latest`):

- dtype: `uint8`
- range: `[0..255]`
- shape: `(T, H, W, C)` where `T = output.shape[0]` (do not assume a fixed frames-per-chunk)
- per-frame addressing: `(chunk_index, frame_index)` where `frame_index ∈ [0, T)`

### Data model + buffer sketch

Store output by **chunk** internally (FrameBus/ChunkStore style) and expose per-frame access via `(chunk_index, frame_index)`.

```python
# src/scope/server/output_frame_history.py (sketch)

from dataclasses import dataclass
from collections import deque
import threading
import torch
from typing import Any


@dataclass(frozen=True)
class OutputChunk:
    chunk_index: int
    created_at: float
    frames: torch.Tensor  # (T, H, W, C) uint8, CPU

    # Minimal, pipeline-facing metadata for UI/debugging (optional)
    prompts: list[dict] | None = None
    compiled_prompt_text: str | None = None
    base_seed: int | None = None
    init_cache: bool | None = None  # True if init_cache was passed for this chunk

    # Optional pre-encoded thumbnail to avoid repeated encode-on-read
    thumbnail_jpeg: bytes | None = None


class OutputFrameHistory:
    def __init__(self, max_chunks: int = 240, max_memory_mb: int = 512):
        self._chunks: deque[OutputChunk] = deque(maxlen=max_chunks)
        self._by_chunk: dict[int, OutputChunk] = {}
        self._lock = threading.Lock()
        self._max_bytes = max_memory_mb * 1024 * 1024
        self._bytes = 0

    # Important: avoid holding _lock during heavy JPEG encode/decode work.
    def add_chunk(self, chunk: OutputChunk) -> None: ...
    def get_chunk(self, chunk_index: int) -> OutputChunk | None: ...
    def get_frame(self, chunk_index: int, frame_index: int) -> torch.Tensor | None: ...
    def get_range(self, start_chunk: int, end_chunk: int) -> list[OutputChunk]: ...
    def get_thumbnails(self, every_n_chunks: int = 10) -> list[dict[str, Any]]: ...
    def get_stats(self) -> dict[str, Any]: ...
    def clear(self) -> None: ...
```

### Lifecycle + indexing

- `chunk_index` in `OutputChunk` refers to the chunk index used for that pipeline call (capture `self.chunk_index` **before** incrementing).
- Clear `output_frame_history` on session stop and on snapshot restore (MVP) to avoid ambiguous/duplicate `chunk_index` keys.
- `init_cache` (captured from `call_params["init_cache"]`) can be surfaced in UI as a “hard cut / reset continuity” marker.

## API Endpoints

```python
# In src/scope/server/app.py (sketch)
#
# Follow the existing `/api/v1/realtime/frame/latest` pattern:
# - `session = get_active_session(webrtc_manager)`
# - `vt.initialize_output_processing()`
# - `fp = vt.frame_processor`
#
# Avoid `base64-in-JSON` for full-resolution scrubbing ranges (payload bloat).
# Use JSON for metadata + binary endpoints for images.
# (Encoding helpers are illustrative; MVP can start with PNG only to match `/frame/latest`.)
#
# Multi-session: `get_active_session()` currently raises if multiple sessions are connected.
# Either keep that constraint for scrubbing endpoints, or add an explicit `session_id` query param.


@app.get("/api/v1/realtime/frames/stats")
async def get_output_frame_history_stats(
    webrtc_manager: WebRTCManager = Depends(get_webrtc_manager),
):
    """Get output frame history statistics."""
    session = get_active_session(webrtc_manager)
    vt = session.video_track
    vt.initialize_output_processing()
    fp = vt.frame_processor
    history = fp.output_frame_history
    return history.get_stats()


@app.get("/api/v1/realtime/frames/metadata")
async def get_output_frame_metadata(
    start_chunk: int,
    end_chunk: int,
    max_chunks: int = 200,
    webrtc_manager: WebRTCManager = Depends(get_webrtc_manager),
):
    """Get chunk metadata for a scrub range (no pixels).

    Server clamps `max_chunks` to prevent pathological requests.
    """
    session = get_active_session(webrtc_manager)
    vt = session.video_track
    vt.initialize_output_processing()
    fp = vt.frame_processor
    history = fp.output_frame_history

    chunks = history.get_range(start_chunk, end_chunk)[:max_chunks]
    return {
        "chunks": [
            {
                "chunk_index": c.chunk_index,
                "created_at": c.created_at,
                "num_frames": int(c.frames.shape[0]),
                "prompts": c.prompts,
                "base_seed": c.base_seed,
                "init_cache": c.init_cache,
            }
            for c in chunks
        ],
        "chunk_range": history.get_stats().get("chunk_range"),
    }


@app.get("/api/v1/realtime/frames/{chunk_index}")
async def get_output_frame_image(
    chunk_index: int,
    frame_index: int = 0,
    format: str = "png",  # "png" | "jpeg"
    webrtc_manager: WebRTCManager = Depends(get_webrtc_manager),
):
    """Get a specific generated frame as image bytes."""
    session = get_active_session(webrtc_manager)
    vt = session.video_track
    vt.initialize_output_processing()
    fp = vt.frame_processor
    history = fp.output_frame_history

    frame = history.get_frame(chunk_index, frame_index)
    if frame is None:
        raise HTTPException(404, f"Frame not found: chunk={chunk_index}, frame={frame_index}")

    # Encode without touching output_queue (non-destructive), similar to `/frame/latest`.
    if format == "png":
        frame_chw = frame.permute(2, 0, 1).contiguous()
        png_bytes = encode_png(frame_chw).numpy().tobytes()
        return Response(content=png_bytes, media_type="image/png")
    else:
        jpeg_bytes = encode_jpeg(frame.permute(2, 0, 1).contiguous(), quality=85).numpy().tobytes()
        return Response(content=jpeg_bytes, media_type="image/jpeg")


@app.get("/api/v1/realtime/frames/thumbnails")
async def get_output_frame_thumbnails(
    every_n_chunks: int = 10,
    max_count: int = 50,
    webrtc_manager: WebRTCManager = Depends(get_webrtc_manager),
):
    """Get a small filmstrip for timeline UI (OK to use base64 here).

    Must clamp `max_count`. Payload is intended for thumbnails only.
    """
    session = get_active_session(webrtc_manager)
    vt = session.video_track
    vt.initialize_output_processing()
    fp = vt.frame_processor
    history = fp.output_frame_history

    thumbs = history.get_thumbnails(every_n_chunks=every_n_chunks)[:max_count]
    return {"thumbnails": thumbs, "chunk_range": history.get_stats().get("chunk_range")}
```

## Integration with FrameProcessor

```python
# In src/scope/server/frame_processor.py

from .output_frame_history import OutputFrameHistory, OutputChunk

class FrameProcessor:
    def __init__(self, ...):
        # ... existing init

        # Output frame history for scrubbing (do not confuse with `self.frame_buffer` input deque)
        self.output_frame_history = OutputFrameHistory(
            max_chunks=int(os.environ.get("OUTPUT_FRAME_HISTORY_MAX_CHUNKS", 240)),
            max_memory_mb=int(os.environ.get("OUTPUT_FRAME_HISTORY_MAX_MB", 512)),
        )

    def process_chunk(self, ...):
        # ... existing chunk processing

        # Capture the chunk index BEFORE incrementing (chunk_index is rewound on snapshot restore).
        chunk_index = self.chunk_index

        # Capture point (MVP): after FrameProcessor normalizes pipeline output to CPU uint8.
        # output: (T, H, W, C) uint8 on CPU, where T = output.shape[0] (do not assume 3).
        if output is not None:
            chunk = OutputChunk(
                chunk_index=chunk_index,
                created_at=time.time(),
                frames=output,  # (T, H, W, C) uint8, CPU
                prompts=self.parameters.get("prompts"),
                compiled_prompt_text=(self._compiled_prompt.prompt if self._compiled_prompt else None),
                base_seed=self.parameters.get("base_seed"),
                init_cache=call_params.get("init_cache"),
            )
            self.output_frame_history.add_chunk(chunk)

        # ... rest of processing (WebRTC send, etc.)
```

## Integration with Snapshots

Snapshots are the natural UI surface for “branch point selection”. For a good UX, snapshot responses should include a small preview image (so the user can see what they’re forking from).

```python
# In src/scope/server/frame_processor.py (sketch)
#
# Note: snapshot responses are sent over the WebRTC data channel as JSON
# (`snapshot_response_callback`), so any image bytes must be base64 encoded.

def _create_snapshot(self) -> Snapshot:
    snapshot = ...  # existing snapshot logic

    # Option A (MVP): derive a small thumbnail from the latest output frame.
    # frame: (H, W, C) uint8 on CPU (already stored for `/frame/latest`).
    frame = self.get_latest_frame()
    snapshot.preview_jpeg_base64 = encode_thumbnail_base64(frame)

    return snapshot


def _restore_snapshot(self, snapshot_id: str) -> bool:
    ok = ...  # existing restore logic
    if ok:
        # MVP semantics: snapshot restore rewinds `chunk_index`, so clear output history to avoid
        # duplicate / ambiguous `chunk_index` keys. (Future: branching timelines.)
        self.output_frame_history.clear()
    return ok
```

## CLI Commands

```bash
# Existing (today): latest frame only
video-cli frame --out ./latest.png

# Proposed (new): add a `frames` group aligned with REST endpoints
video-cli frames stats
video-cli frames get 150 --frame-index 0 --format png --out ./frame_150.png
video-cli frames export --start-chunk 100 --end-chunk 200 --out-dir ./frames/
video-cli frames filmstrip --every-n-chunks 10 --max-count 50 --out ./filmstrip.png
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OUTPUT_FRAME_HISTORY_MAX_CHUNKS` | 240 | Maximum chunks to keep (upper bound; memory limit may evict sooner) |
| `OUTPUT_FRAME_HISTORY_MAX_MB` | 512 | Memory limit in MB (raw frames + optional thumbnails) |
| `OUTPUT_FRAME_HISTORY_STORE_THUMBNAILS` | 1 | If set, store a small per-chunk JPEG thumbnail at write-time |
| `OUTPUT_FRAME_HISTORY_THUMBNAIL_QUALITY` | 60 | JPEG quality for stored thumbnails |

## Memory Considerations

MVP stores CPU `uint8` RGB frames. Memory scales with:

`bytes_per_frame ≈ H * W * 3`

At `480x832`: `~1.2 MB/frame` (raw).

If your pipeline outputs `T` frames per chunk, then:
- bytes per chunk ≈ `T * bytes_per_frame`
- bytes retained ≈ `num_chunks_retained * T * bytes_per_frame` (before accounting for eviction)

**Recommendation (MVP):** store raw `uint8` frames + optional small JPEG thumbnails, and rely on `OUTPUT_FRAME_HISTORY_MAX_MB` for predictable bounds.

## Use Cases

### 1. Scrubbing Timeline

```javascript
// Fetch metadata for the visible timeline window (no pixels).
const { chunks } = await fetch(
  `/api/v1/realtime/frames/metadata?start_chunk=${visibleStart}&end_chunk=${visibleEnd}`
).then(r => r.json());

// When the playhead moves, fetch just the needed frame as image bytes.
async function showFrame(chunkIndex, frameIndex = 0) {
  const r = await fetch(
    `/api/v1/realtime/frames/${chunkIndex}?frame_index=${frameIndex}&format=jpeg`
  );
  const url = URL.createObjectURL(await r.blob());
  previewImg.src = url;
}
```

### 2. Branch Point Preview

```javascript
// Option A: include a thumbnail in the snapshot response (data channel JSON)
branchPreview.src = `data:image/jpeg;base64,${snapshot.preview_jpeg_base64}`;

// Option B: fetch from output frame history by chunk index
await showFrame(snapshot.chunk_index, 0);
```

### 3. Instant Replay

```javascript
// Replay a range by fetching frames on demand.
// Note: for efficiency, we may add a range streaming/zip export endpoint later.
const { chunks } = await fetch(
  `/api/v1/realtime/frames/metadata?start_chunk=${replayStart}&end_chunk=${replayEnd}`
).then(r => r.json());

for (const c of chunks) {
  await showFrame(c.chunk_index, 0);
  await new Promise(r => setTimeout(r, 1000 / 16));
}
```

### 4. Filmstrip Thumbnails

```javascript
// Load filmstrip for timeline overview
const { thumbnails } = await fetch(
  `/api/v1/realtime/frames/thumbnails?every_n_chunks=20&max_count=50`
).then(r => r.json());

filmstrip.render(thumbnails);
```

## Relationship to Other Proposals

| Doc | What it stores | Use case |
|-----|---------------|----------|
| `notes/realtime_video_architecture.md` | FrameBus / ChunkStore concept | Canonical abstraction this proposal should align with |
| `notes/proposals/session-recording-timeline-export.md` | Control events | Offline replay/re-render at any quality |
| `notes/proposals/server-side-session-recorder.md` | Control events | Offline replay/re-render at any quality |
| **This proposal** | Output pixels (recent, in-memory) | Instant scrub/replay/branch preview |

These are complementary:
- Control event recording → durable “what we asked for” (replayable)
- Output frame history → fast “what we saw” (scrubbable), no GPU cost

## Implementation Order

1. **OutputFrameHistory core** - chunk-keyed ring buffer + memory clamp
2. **FrameProcessor capture** - add chunks after CPU `uint8` conversion
3. **REST endpoints** - stats, metadata range, per-frame bytes, thumbnails
4. **Snapshot preview + restore semantics** - thumbnail in response + clear-on-restore (MVP)
5. **CLI additions** - `video-cli frames ...`
6. **Frontend timeline** - filmstrip + scrub playback

## Open Questions

1. **Snapshot restore semantics:** clear history (MVP) vs truncate vs branching timelines?
2. **Thumbnail strategy:** encode at write-time (store bytes) vs encode on read? Downscale method + lock scope.
3. **Range retrieval:** per-frame fetch (simple) vs batch/stream endpoints for efficient replay/export?
4. **Eviction policy:** FIFO vs prioritize snapshot chunks / hard cuts?
5. **Multi-session:** keep `get_active_session()` constraints vs add `session_id` query param?

## Related Files

- `src/scope/server/frame_processor.py` - Main integration point
- `src/scope/server/app.py` - REST endpoints
- `src/scope/server/tracks.py` - `initialize_output_processing()` pattern
- `src/scope/cli/video_cli.py` - CLI integration points
- `notes/realtime_video_architecture.md` - FrameBus / ChunkStore design
- `notes/proposals/server-side-session-recorder.md` - Control event recording
