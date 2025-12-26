# Frame Buffer & Timeline Scrubbing

> Status: Draft
> Date: 2025-12-26
> Related: `notes/realtime_video_architecture.md` (FrameBus design), `notes/proposals/server-side-session-recorder.md`

## Problem

Currently, generated frames flow directly to WebRTC and are gone. There's no way to:
- Scroll back through what was generated
- Replay a segment without re-rendering
- Visually preview a branch point before forking
- Scrub a timeline of actual output (not just control events)

The session recording proposals capture **control events** for offline re-render, but don't store **actual frames** for instant playback.

## Solution

A server-side frame buffer (ring buffer) that stores rendered frames, enabling:
- Instant scrubbing through generated history
- Replay without GPU cost
- Visual branch point selection
- Timeline thumbnails / filmstrip UI

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Frame Flow                                  │
│                                                                  │
│  Pipeline ──► FrameProcessor ──► FrameBuffer ──► WebRTC         │
│                                       │                          │
│                                       ├──► REST API (scrub)      │
│                                       ├──► Thumbnails            │
│                                       └──► Snapshot attachment   │
└─────────────────────────────────────────────────────────────────┘
```

### FrameBuffer Class

```python
# src/scope/server/frame_buffer.py

from dataclasses import dataclass, field
from collections import deque
from threading import RLock
import torch
import numpy as np
from PIL import Image
import io
import base64

@dataclass
class BufferedFrame:
    """A single buffered frame with metadata."""
    chunk_index: int
    frame_index: int  # Within chunk (0, 1, 2 for 3-frame chunks)
    timestamp: float  # Wall-clock when generated

    # Frame data (one of these, based on storage mode)
    tensor: torch.Tensor | None = None  # GPU tensor [C, H, W]
    numpy: np.ndarray | None = None     # CPU numpy [H, W, C]
    jpeg_bytes: bytes | None = None     # Compressed JPEG

    # Metadata
    prompt: str | None = None
    seed: int | None = None

    def to_pil(self) -> Image.Image:
        """Convert to PIL Image regardless of storage format."""
        if self.tensor is not None:
            # GPU tensor → PIL
            arr = self.tensor.cpu().permute(1, 2, 0).numpy()
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
            return Image.fromarray(arr)
        elif self.numpy is not None:
            return Image.fromarray(self.numpy)
        elif self.jpeg_bytes is not None:
            return Image.open(io.BytesIO(self.jpeg_bytes))
        raise ValueError("No frame data available")

    def to_base64_jpeg(self, quality: int = 85) -> str:
        """Convert to base64 JPEG for API responses."""
        if self.jpeg_bytes is not None:
            return base64.b64encode(self.jpeg_bytes).decode()
        pil = self.to_pil()
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=quality)
        return base64.b64encode(buf.getvalue()).decode()

    @property
    def memory_bytes(self) -> int:
        """Approximate memory usage."""
        if self.tensor is not None:
            return self.tensor.numel() * self.tensor.element_size()
        elif self.numpy is not None:
            return self.numpy.nbytes
        elif self.jpeg_bytes is not None:
            return len(self.jpeg_bytes)
        return 0


class FrameBuffer:
    """Ring buffer for rendered frames with scrubbing support.

    Thread-safety: All public methods are thread-safe via RLock.
    Storage modes:
    - "gpu": Keep tensors on GPU (fastest access, highest VRAM)
    - "cpu": Move to CPU numpy (lower VRAM, still fast)
    - "jpeg": Compress to JPEG (lowest memory, decode overhead)
    """

    def __init__(
        self,
        max_frames: int = 300,  # ~100 chunks at 3 frames/chunk = ~25 sec at 4 chunks/sec
        max_memory_mb: int = 512,
        storage_mode: str = "cpu",  # "gpu", "cpu", or "jpeg"
        jpeg_quality: int = 90,
    ):
        self.max_frames = max_frames
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.storage_mode = storage_mode
        self.jpeg_quality = jpeg_quality

        self._buffer: deque[BufferedFrame] = deque(maxlen=max_frames)
        self._lock = RLock()
        self._chunk_index_map: dict[int, list[BufferedFrame]] = {}  # chunk_index → frames

        # Stats
        self._total_frames_added = 0
        self._total_frames_evicted = 0

    def add_frame(
        self,
        frame: torch.Tensor,  # [C, H, W] or [H, W, C]
        chunk_index: int,
        frame_index: int,
        timestamp: float,
        prompt: str | None = None,
        seed: int | None = None,
    ) -> None:
        """Add a frame to the buffer."""
        with self._lock:
            # Convert based on storage mode
            if self.storage_mode == "gpu":
                stored = BufferedFrame(
                    chunk_index=chunk_index,
                    frame_index=frame_index,
                    timestamp=timestamp,
                    tensor=frame.clone(),
                    prompt=prompt,
                    seed=seed,
                )
            elif self.storage_mode == "cpu":
                if frame.dim() == 3 and frame.shape[0] in (3, 4):
                    arr = frame.cpu().permute(1, 2, 0).numpy()
                else:
                    arr = frame.cpu().numpy()
                arr = (arr * 255).clip(0, 255).astype(np.uint8)
                stored = BufferedFrame(
                    chunk_index=chunk_index,
                    frame_index=frame_index,
                    timestamp=timestamp,
                    numpy=arr,
                    prompt=prompt,
                    seed=seed,
                )
            else:  # jpeg
                if frame.dim() == 3 and frame.shape[0] in (3, 4):
                    arr = frame.cpu().permute(1, 2, 0).numpy()
                else:
                    arr = frame.cpu().numpy()
                arr = (arr * 255).clip(0, 255).astype(np.uint8)
                pil = Image.fromarray(arr)
                buf = io.BytesIO()
                pil.save(buf, format="JPEG", quality=self.jpeg_quality)
                stored = BufferedFrame(
                    chunk_index=chunk_index,
                    frame_index=frame_index,
                    timestamp=timestamp,
                    jpeg_bytes=buf.getvalue(),
                    prompt=prompt,
                    seed=seed,
                )

            # Evict if needed (deque handles max_frames automatically)
            if len(self._buffer) == self.max_frames:
                evicted = self._buffer[0]
                self._remove_from_index(evicted)
                self._total_frames_evicted += 1

            self._buffer.append(stored)
            self._add_to_index(stored)
            self._total_frames_added += 1

            # Memory-based eviction
            self._evict_if_over_memory()

    def _add_to_index(self, frame: BufferedFrame) -> None:
        """Add frame to chunk index."""
        if frame.chunk_index not in self._chunk_index_map:
            self._chunk_index_map[frame.chunk_index] = []
        self._chunk_index_map[frame.chunk_index].append(frame)

    def _remove_from_index(self, frame: BufferedFrame) -> None:
        """Remove frame from chunk index."""
        if frame.chunk_index in self._chunk_index_map:
            frames = self._chunk_index_map[frame.chunk_index]
            if frame in frames:
                frames.remove(frame)
            if not frames:
                del self._chunk_index_map[frame.chunk_index]

    def _evict_if_over_memory(self) -> None:
        """Evict oldest frames if over memory limit."""
        while self._current_memory_bytes > self.max_memory_bytes and self._buffer:
            evicted = self._buffer.popleft()
            self._remove_from_index(evicted)
            self._total_frames_evicted += 1

    @property
    def _current_memory_bytes(self) -> int:
        """Current memory usage."""
        return sum(f.memory_bytes for f in self._buffer)

    def get_frame(self, chunk_index: int, frame_index: int = 0) -> BufferedFrame | None:
        """Get a specific frame by chunk and frame index."""
        with self._lock:
            frames = self._chunk_index_map.get(chunk_index, [])
            for f in frames:
                if f.frame_index == frame_index:
                    return f
            return None

    def get_chunk(self, chunk_index: int) -> list[BufferedFrame]:
        """Get all frames for a chunk."""
        with self._lock:
            return list(self._chunk_index_map.get(chunk_index, []))

    def get_range(
        self,
        start_chunk: int,
        end_chunk: int,
        frame_index: int | None = None,
    ) -> list[BufferedFrame]:
        """Get frames in a chunk range (for scrubbing).

        Args:
            start_chunk: Start chunk index (inclusive)
            end_chunk: End chunk index (inclusive)
            frame_index: If set, only return this frame index per chunk
        """
        with self._lock:
            result = []
            for chunk_idx in range(start_chunk, end_chunk + 1):
                frames = self._chunk_index_map.get(chunk_idx, [])
                if frame_index is not None:
                    frames = [f for f in frames if f.frame_index == frame_index]
                result.extend(frames)
            return sorted(result, key=lambda f: (f.chunk_index, f.frame_index))

    def get_latest(self, n: int = 1) -> list[BufferedFrame]:
        """Get the N most recent frames."""
        with self._lock:
            return list(self._buffer)[-n:]

    def get_thumbnails(
        self,
        every_n_chunks: int = 10,
        frame_index: int = 0,
    ) -> list[BufferedFrame]:
        """Get thumbnail frames for filmstrip display."""
        with self._lock:
            chunks = sorted(self._chunk_index_map.keys())
            sampled = chunks[::every_n_chunks]
            result = []
            for chunk_idx in sampled:
                frames = self._chunk_index_map.get(chunk_idx, [])
                for f in frames:
                    if f.frame_index == frame_index:
                        result.append(f)
                        break
            return result

    @property
    def chunk_range(self) -> tuple[int, int] | None:
        """Get (oldest_chunk, newest_chunk) or None if empty."""
        with self._lock:
            if not self._chunk_index_map:
                return None
            chunks = self._chunk_index_map.keys()
            return (min(chunks), max(chunks))

    @property
    def frame_count(self) -> int:
        """Number of frames in buffer."""
        with self._lock:
            return len(self._buffer)

    def clear(self) -> None:
        """Clear all frames."""
        with self._lock:
            self._buffer.clear()
            self._chunk_index_map.clear()

    def get_stats(self) -> dict:
        """Get buffer statistics."""
        with self._lock:
            return {
                "frame_count": len(self._buffer),
                "chunk_range": self.chunk_range,
                "memory_mb": self._current_memory_bytes / (1024 * 1024),
                "max_memory_mb": self.max_memory_bytes / (1024 * 1024),
                "storage_mode": self.storage_mode,
                "total_added": self._total_frames_added,
                "total_evicted": self._total_frames_evicted,
            }
```

## API Endpoints

```python
# In src/scope/server/app.py

@app.get("/api/v1/realtime/frames/range")
async def get_frame_range(
    start_chunk: int,
    end_chunk: int,
    frame_index: int = 0,
    format: str = "jpeg",  # "jpeg" or "metadata"
    webrtc_manager: WebRTCManager = Depends(get_webrtc_manager),
):
    """Get frames in a chunk range for scrubbing.

    Returns list of frames with base64 JPEG data or metadata only.
    """
    session = get_active_session(webrtc_manager)
    buffer = session.video_track.frame_processor.frame_buffer

    frames = buffer.get_range(start_chunk, end_chunk, frame_index)

    if format == "metadata":
        return {
            "frames": [
                {
                    "chunk_index": f.chunk_index,
                    "frame_index": f.frame_index,
                    "timestamp": f.timestamp,
                    "prompt": f.prompt,
                }
                for f in frames
            ]
        }
    else:
        return {
            "frames": [
                {
                    "chunk_index": f.chunk_index,
                    "frame_index": f.frame_index,
                    "timestamp": f.timestamp,
                    "data": f.to_base64_jpeg(),
                }
                for f in frames
            ]
        }


@app.get("/api/v1/realtime/frames/thumbnails")
async def get_thumbnails(
    every_n_chunks: int = 10,
    max_count: int = 30,
    webrtc_manager: WebRTCManager = Depends(get_webrtc_manager),
):
    """Get thumbnail filmstrip for timeline UI."""
    session = get_active_session(webrtc_manager)
    buffer = session.video_track.frame_processor.frame_buffer

    thumbnails = buffer.get_thumbnails(every_n_chunks=every_n_chunks)[:max_count]

    return {
        "thumbnails": [
            {
                "chunk_index": f.chunk_index,
                "timestamp": f.timestamp,
                "data": f.to_base64_jpeg(quality=60),  # Lower quality for thumbnails
            }
            for f in thumbnails
        ],
        "chunk_range": buffer.chunk_range,
    }


@app.get("/api/v1/realtime/frames/{chunk_index}")
async def get_frame_at_chunk(
    chunk_index: int,
    frame_index: int = 0,
    webrtc_manager: WebRTCManager = Depends(get_webrtc_manager),
):
    """Get a specific frame by chunk index."""
    session = get_active_session(webrtc_manager)
    buffer = session.video_track.frame_processor.frame_buffer

    frame = buffer.get_frame(chunk_index, frame_index)
    if frame is None:
        raise HTTPException(404, f"Frame not found: chunk={chunk_index}, frame={frame_index}")

    return {
        "chunk_index": frame.chunk_index,
        "frame_index": frame.frame_index,
        "timestamp": frame.timestamp,
        "prompt": frame.prompt,
        "data": frame.to_base64_jpeg(),
    }


@app.get("/api/v1/realtime/frames/stats")
async def get_frame_buffer_stats(
    webrtc_manager: WebRTCManager = Depends(get_webrtc_manager),
):
    """Get frame buffer statistics."""
    session = get_active_session(webrtc_manager)
    buffer = session.video_track.frame_processor.frame_buffer
    return buffer.get_stats()
```

## Integration with FrameProcessor

```python
# In src/scope/server/frame_processor.py

from .frame_buffer import FrameBuffer

class FrameProcessor:
    def __init__(self, ...):
        # ... existing init

        # Frame buffer for scrubbing
        self.frame_buffer = FrameBuffer(
            max_frames=int(os.environ.get("FRAME_BUFFER_MAX_FRAMES", 300)),
            max_memory_mb=int(os.environ.get("FRAME_BUFFER_MAX_MB", 512)),
            storage_mode=os.environ.get("FRAME_BUFFER_MODE", "cpu"),
        )

    def process_chunk(self, ...):
        # ... existing chunk processing

        # After frames are generated, add to buffer
        if output_frames is not None:
            current_prompt = self._get_current_effective_prompt()[0]
            for i, frame in enumerate(output_frames):
                self.frame_buffer.add_frame(
                    frame=frame,
                    chunk_index=self.chunk_index,
                    frame_index=i,
                    timestamp=time.time(),
                    prompt=current_prompt,
                    seed=self.parameters.get("seed"),
                )

        # ... rest of processing (WebRTC send, etc.)
```

## Integration with Snapshots

Attach frame data to snapshots for visual branch point selection:

```python
# In snapshot creation

def create_snapshot(self, ...) -> Snapshot:
    # ... existing snapshot logic

    # Attach recent frames for preview
    recent_frames = self.frame_buffer.get_latest(n=3)
    snapshot.preview_frames = [f.to_base64_jpeg(quality=70) for f in recent_frames]

    return snapshot
```

## CLI Commands

```bash
# Get buffer stats
video-cli frames stats

# Export frames as images
video-cli frames export --start 100 --end 200 --out ./frames/

# Get frame at specific chunk
video-cli frames get 150 --out frame_150.png

# Export filmstrip
video-cli frames filmstrip --every 10 --out filmstrip.png
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `FRAME_BUFFER_MAX_FRAMES` | 300 | Maximum frames to keep |
| `FRAME_BUFFER_MAX_MB` | 512 | Memory limit in MB |
| `FRAME_BUFFER_MODE` | "cpu" | Storage mode: "gpu", "cpu", "jpeg" |
| `FRAME_BUFFER_JPEG_QUALITY` | 90 | JPEG quality for compressed mode |

## Memory Considerations

At 480x832 resolution:
- Raw tensor (fp32): ~4.8 MB/frame
- CPU numpy (uint8): ~1.2 MB/frame
- JPEG (quality 90): ~50-100 KB/frame

| Mode | 300 frames | 1000 frames |
|------|-----------|-------------|
| GPU (fp32) | 1.4 GB VRAM | 4.8 GB VRAM |
| CPU (uint8) | 360 MB RAM | 1.2 GB RAM |
| JPEG (q90) | 15-30 MB RAM | 50-100 MB RAM |

**Recommendation:** Default to "cpu" mode. Use "jpeg" for long sessions or limited RAM.

## Use Cases

### 1. Scrubbing Timeline

```javascript
// Frontend: fetch frames for visible timeline range
const frames = await fetch(
  `/api/v1/realtime/frames/range?start_chunk=${visibleStart}&end_chunk=${visibleEnd}`
).then(r => r.json());

// Display in timeline
frames.forEach(f => {
  timeline.addFrame(f.chunk_index, `data:image/jpeg;base64,${f.data}`);
});
```

### 2. Branch Point Preview

```javascript
// Before forking, show user what they're branching from
const frame = await fetch(`/api/v1/realtime/frames/${snapshotChunk}`).then(r => r.json());
branchPreview.src = `data:image/jpeg;base64,${frame.data}`;
```

### 3. Instant Replay

```javascript
// Replay last N chunks without re-rendering
const frames = await fetch(
  `/api/v1/realtime/frames/range?start_chunk=${replayStart}&end_chunk=${replayEnd}`
).then(r => r.json());

let i = 0;
const playback = setInterval(() => {
  if (i >= frames.length) {
    clearInterval(playback);
    return;
  }
  canvas.drawImage(frames[i].data);
  i++;
}, 1000 / 16);  // 16 FPS playback
```

### 4. Filmstrip Thumbnails

```javascript
// Load filmstrip for timeline overview
const thumbnails = await fetch(
  `/api/v1/realtime/frames/thumbnails?every_n_chunks=20&max_count=50`
).then(r => r.json());

filmstrip.render(thumbnails);
```

## Relationship to Other Proposals

| Proposal | What it stores | Use case |
|----------|---------------|----------|
| `session-recording-timeline-export.md` | Control events | Offline re-render at any quality |
| `server-side-session-recorder.md` | Control events | Same, server-side |
| **This proposal** | Actual frames | Instant scrub/replay/branch preview |

These are complementary:
- Control event recording → faithful re-render at higher quality
- Frame buffer → instant visual feedback, no GPU cost

## Implementation Order

1. **Core FrameBuffer class** - ring buffer, storage modes, eviction
2. **FrameProcessor integration** - add frames after generation
3. **Basic REST endpoints** - `/frames/range`, `/frames/{chunk}`
4. **Stats endpoint** - buffer monitoring
5. **Thumbnail endpoint** - filmstrip support
6. **CLI commands** - export, stats
7. **Snapshot integration** - attach preview frames
8. **Frontend timeline** - visual scrubbing UI

## Open Questions

1. **GPU vs CPU default?** CPU is safer (no VRAM pressure), but GPU is faster for frequent access.

2. **Frame index granularity?** Store all 3 frames per chunk, or just keyframe (frame 0)?

3. **Eviction policy?** Currently FIFO. Could do LRU or priority-based (keep branch points longer).

4. **WebSocket streaming?** For live filmstrip updates, could push new thumbnails via WebSocket instead of polling.

5. **Disk spill?** For very long sessions, could spill oldest frames to disk instead of evicting.

## Related Files

- `src/scope/server/frame_processor.py` - Main integration point
- `src/scope/server/app.py` - REST endpoints
- `notes/realtime_video_architecture.md` - Original FrameBus design
- `notes/proposals/server-side-session-recorder.md` - Control event recording
