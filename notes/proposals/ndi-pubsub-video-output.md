# NDI / Pub-Sub Video Output

> Status: Draft (Updated with architecture review)
> Date: 2025-12-26
> Updated: 2025-12-27

## Summary

Add NDI (Network Device Interface) output to enable network-based video streaming with multiple subscribers. This enables:

1. **Remote consumption** — Stream from GPU box to local TouchDesigner/OBS
2. **Multi-consumer** — Multiple apps can subscribe to the same stream
3. **V2V pipelines** — Feed output to secondary video-to-video models
4. **Cross-platform** — Works Mac ↔ Windows ↔ Linux

This proposal intentionally separates:

- **MVP multi-output**: producer-side fan-out (Spout + NDI) from `FrameProcessor.process_chunk()`; no broadcaster/ring-buffer required.
- **End-state pub-sub**: a non-destructive ring buffer when we truly need multiple independent consumers/cursors (e.g., multiple WebRTC clients).

## Current Architecture

```
Pipeline → output_queue (destructive) → WebRTC → Single Browser
                                            ↓
                                      Spout (Windows, local only)
```

**Key observation:** Spout is currently fed from `FrameProcessor.get()` (consumer-side), not `process_chunk()` (producer-side). This means **no WebRTC consumer = no Spout output**.

### What exists today (in code)

Scope already has the architectural shape this proposal wants: **`FrameProcessor` is the choke point where “frames become outputs.”**

- **WebRTC video (destructive, single-consumer)**
  - `src/scope/server/frame_processor.py`: `FrameProcessor.process_chunk()` enqueues frames into `self.output_queue`
  - `src/scope/server/tracks.py`: `VideoProcessingTrack.recv()` consumes via `FrameProcessor.get()`
- **REST latest frame (non-destructive, latest-only)**
  - `src/scope/server/frame_processor.py`: `latest_frame_cpu` + `get_latest_frame()` (returns a clone)
  - `src/scope/server/app.py`: `GET /api/v1/realtime/frame/latest` encodes the latest frame as PNG
- **Spout output (Windows)**
  - `src/scope/server/frame_processor.py`: background sender thread (`_spout_sender_loop`) and `spout_sender_queue`
  - **But:** frames are enqueued to Spout inside `FrameProcessor.get()` (i.e., on WebRTC consumption)

### Key gaps / mismatches to address

These are the concrete mismatches between “pub-sub outputs” and how the current server works:

- **Spout is consumer-coupled**: if WebRTC is paused, slow, or absent, Spout output is throttled/starved.
- **`output_queue` is destructive by design**: one consumer drains frames; “true” multi-consumer needs a ring buffer or per-consumer queues.
- **REST/CLI control surface is curated**: there’s no generic `/api/v1/realtime/parameters` endpoint, so the CLI can’t toggle arbitrary output params today.
- **Session scoping is per-WebRTC-session today**: each WebRTC session creates its own `FrameProcessor`, while REST endpoints generally assume a single active session.
- **Schema has Spout but no NDI**: `Parameters.spout_sender` / `HardwareInfoResponse.spout_available` exist; there’s no `ndi_sender` / `ndi_available` yet.
- **Format/convert costs must be explicit**: NDI typically wants BGRX/BGRA/UYVY; conversion should happen off the generation thread with drop-if-full backpressure.

**Limitations:**
- WebRTC queue is destructive (one consumer)
- Spout is **consumer-coupled** to WebRTC (stops when WebRTC stops/pauses)
- Spout is local-only (shared GPU memory)
- No network streaming for external apps
- No multi-consumer broadcast
- No generic REST endpoint to toggle outputs from CLI

## Proposed Architecture

### MVP (no broadcaster yet)

This is the minimal shape needed to add NDI and make Spout “real fan-out”:

```
Pipeline → FrameProcessor.process_chunk() → output_queue (WebRTC)
                                   ├──→ spout_sender_queue (Spout thread)
                                   ├──→ NDISender queue (NDI thread)
                                   └──→ latest_frame_cpu (REST latest)
```

### End-state (true pub-sub / independent cursors)

```
Pipeline → frame_broadcaster (non-destructive)
                ↓
    ┌───────────┼───────────┬──────────────┐
    ↓           ↓           ↓              ↓
 WebRTC      Spout        NDI           REST
 (browser)  (local Win)  (network)    (polling)
```

### Key Changes

1. **Producer-side fan-out** — Feed Spout/NDI from `process_chunk()`, not from `get()`
2. **NDI Sender** — New output sink (queue + background thread, drop-if-full semantics)
3. **Optional Frame Broadcaster** — Only when we need independent cursors/backpressure (e.g., multiple WebRTC consumers)

### Two Implementation Paths

**Path 1: Minimal MVP (Recommended First)**
- Add NDI as peer output sink alongside Spout
- **Prerequisite:** Move Spout enqueue from `get()` to `process_chunk()` (producer-side)
- Feed both from `process_chunk()` after CPU uint8 conversion
- No full pub-sub needed yet (NDI itself supports multiple receivers)

**Path 2: Full Pub-Sub (Future)**
- Replace `output_queue` with `FrameBroadcaster` (ring buffer)
- Each output gets independent cursor with backpressure handling
- Only needed when we want:
  - Multiple WebRTC consumers sharing one generator
  - Robust multi-sink timing guarantees
  - "Seek/scrub" style output history

### Prerequisite: Decouple Spout from WebRTC

Before (or alongside) adding NDI, move Spout to producer-side:

```python
# Current (consumer-coupled): `FrameProcessor.get()` enqueues Spout when WebRTC reads.
def get(self) -> torch.Tensor | None:
    frame = self.output_queue.get_nowait()
    if self.spout_sender_enabled and self.spout_sender is not None:
        self.spout_sender_queue.put_nowait(frame.numpy())
    return frame

# Target (producer-coupled): publish outputs in `process_chunk()` after CPU uint8 conversion.
output = (
  (output * 255.0).clamp(0, 255).to(dtype=torch.uint8).contiguous().detach().cpu()
)

for frame in output:
    self.output_queue.put_nowait(frame)  # WebRTC
    if self.spout_sender_enabled and self.spout_sender is not None:
        self.spout_sender_queue.put_nowait(frame.numpy())  # Spout (drop if full)
    if self.ndi_sender is not None:
        self.ndi_sender.send_frame(frame.numpy())  # NDI (drop if full)
```

This is the ideal hook point: stable CPU tensor buffer suitable for all outputs.

## NDI Overview

**NDI (Network Device Interface)** by Vizrt/NewTek:
- Industry standard for IP video in broadcast/live production
- Low latency (~1-2 frames over LAN)
- Auto-discovery (devices find each other)
- Multiple receivers per sender
- Supports alpha channel, audio, metadata

### Platform Support

| Platform | NDI SDK | Python Bindings |
|----------|---------|-----------------|
| Windows | Full | `ndi-python`, `python-ndi` |
| macOS | Full | `ndi-python`, `python-ndi` |
| Linux | Full (x86_64) | `ndi-python`, `python-ndi` |

### Receiver Support

| Application | NDI Support | Use Case |
|-------------|-------------|----------|
| TouchDesigner | Native (NDI In TOP) | Real-time compositing, effects |
| Unreal Engine | Plugin (NDI SDK) | Virtual production, LED walls |
| Unity | Plugin (NDI for Unity) | Real-time environments |
| OBS | Plugin (obs-ndi) | Streaming, recording |
| vMix | Native | Live production |
| Resolume | Native | VJ, projection mapping |
| FFmpeg | Via `libndi` | Transcoding, pipelines |

**Virtual production note:** NDI into Unreal/Unity enables AI video as a texture source for LED wall content, virtual sets, or real-time material inputs. Teammates have Unreal experience if this becomes a focus.

## Implementation

### Phase 1: NDI Sender (MVP)

Add NDI output alongside existing Spout:

```python
# src/scope/server/ndi/sender.py

# Note: treat this as pseudocode. The exact API differs between Python NDI bindings;
# the important part is the semantics (queue + background thread + drop-if-full).

import NDIlib as ndi
import numpy as np
from queue import Empty, Full, Queue
from threading import Thread

class NDISender:
    def __init__(self, name: str = "DaydreamScope"):
        self.name = name
        self.queue: Queue = Queue(maxsize=2)
        self._running = False
        self._thread: Thread | None = None
        self._sender = None

    def start(self):
        if not ndi.initialize():
            raise RuntimeError("NDI initialization failed")

        send_settings = ndi.SendCreate()
        send_settings.ndi_name = self.name
        self._sender = ndi.send_create(send_settings)

        self._running = True
        self._thread = Thread(target=self._send_loop, daemon=True)
        self._thread.start()

    def _send_loop(self):
        video_frame = ndi.VideoFrameV2()

        while self._running:
            try:
                frame = self.queue.get(timeout=0.1)
            except Empty:
                continue

            # Frame arrives as RGB uint8 (HWC). Convert to BGRX in this thread
            # to avoid blocking the generation thread.
            h, w, c = frame.shape
            bgrx = np.zeros((h, w, 4), dtype=np.uint8)
            bgrx[:, :, 0] = frame[:, :, 2]  # B
            bgrx[:, :, 1] = frame[:, :, 1]  # G
            bgrx[:, :, 2] = frame[:, :, 0]  # R
            bgrx[:, :, 3] = 255             # X (opaque)

            video_frame.xres = w
            video_frame.yres = h
            video_frame.FourCC = ndi.FOURCC_VIDEO_TYPE_BGRX
            video_frame.data = bgrx.tobytes()
            video_frame.line_stride_in_bytes = w * 4

            ndi.send_send_video_v2(self._sender, video_frame)

    def send_frame(self, frame: np.ndarray):
        """Non-blocking frame send. Drops if queue full."""
        try:
            self.queue.put_nowait(frame)
        except Full:
            pass  # Drop frame if queue full

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._sender:
            ndi.send_destroy(self._sender)
        ndi.destroy()
```

### Phase 2: Integration with FrameProcessor

Integration points (all in `src/scope/server/frame_processor.py`), mirroring the existing Spout pattern:

1. Add NDI fields to `FrameProcessor.__init__` (at minimum `ndi_sender` + enabled flag; queue/thread can live inside `NDISender` or mirror Spout’s `*_queue` + background loop).
2. In `FrameProcessor.start()`: pop and apply `ndi_sender` from `self.parameters` (like `spout_sender`).
3. In `FrameProcessor.update_parameters()`: intercept `ndi_sender` and apply immediately (don’t wait for a chunk boundary).
4. In `FrameProcessor.process_chunk()`: after CPU uint8 conversion, publish frames to Spout/NDI and `output_queue` from the same producer-side loop, with independent drop/backpressure policies.
5. In `FrameProcessor.stop()`: stop/join the NDI thread and destroy the sender (avoid deadlocks when `stop()` is called from the worker thread).

Important: to keep NDI/Spout independent, don’t gate sink publishing on `FrameProcessor.get()` or WebRTC consumption (each sink should drop frames if it can’t keep up).

### Phase 3: Configuration Hooks

Follow Spout's pattern for config handling:

```python
# FrameProcessor.start()
if "ndi_sender" in self.parameters:
    ndi_config = self.parameters.pop("ndi_sender")
    self._update_ndi_sender(ndi_config)

# FrameProcessor.update_parameters()
if "ndi_sender" in parameters:
    ndi_config = parameters.pop("ndi_sender")
    self._update_ndi_sender(ndi_config)
# queue remaining parameters (mailbox semantics)
```

**Environment variables:**
```bash
SCOPE_NDI_ENABLED=1
SCOPE_NDI_NAME="DaydreamScope"
```

### Phase 4: Pub-Sub Frame Broadcaster (Future)

For true multi-consumer with backpressure handling:

```python
class FrameBroadcaster:
    """Non-destructive frame distribution to multiple consumers."""

    def __init__(self, buffer_size: int = 30):
        self.buffer = collections.deque(maxlen=buffer_size)
        self.consumers: dict[str, Consumer] = {}
        self.frame_index = 0
        self._lock = threading.Lock()

    def publish(self, frame: np.ndarray):
        """Add frame to buffer, notify consumers."""
        with self._lock:
            self.buffer.append((self.frame_index, frame))
            self.frame_index += 1

        for consumer in self.consumers.values():
            consumer.notify()

    def subscribe(self, consumer_id: str, start_from: str = "latest") -> Consumer:
        """Register a new consumer."""
        consumer = Consumer(self, consumer_id, start_from)
        self.consumers[consumer_id] = consumer
        return consumer

    def unsubscribe(self, consumer_id: str):
        self.consumers.pop(consumer_id, None)


class Consumer:
    """Independent frame reader with own cursor."""

    def __init__(self, broadcaster: FrameBroadcaster, id: str, start_from: str):
        self.broadcaster = broadcaster
        self.id = id
        self.read_index = 0 if start_from == "beginning" else broadcaster.frame_index
        self._event = threading.Event()

    def get_next_frame(self, timeout: float = 1.0) -> np.ndarray | None:
        """Get next frame, waiting if necessary."""
        # ... implementation

    def notify(self):
        self._event.set()
```

## Use Cases

### 1. Remote GPU → Local TouchDesigner

```
Remote GPU Box (Linux)              Local Mac
┌─────────────────┐                ┌─────────────────┐
│  Krea Realtime  │                │  TouchDesigner  │
│  NDI Sender     │──── LAN ──────▶│  NDI In TOP     │
│  "DaydreamScope"│                │                 │
└─────────────────┘                └─────────────────┘
```

In TouchDesigner:
1. Add NDI In TOP
2. Select "DaydreamScope" from discovered sources
3. Use output for effects, mixing, etc.

### 2. V2V Secondary Pass

```
GPU 1                               GPU 2
┌─────────────────┐                ┌─────────────────┐
│  Krea Realtime  │                │  V2V Model      │
│  (text-to-video)│──── NDI ──────▶│  (refinement)   │
│                 │                │                 │
└─────────────────┘                └─────────────────┘
                                           │
                                           ▼
                                   Final Output
```

### 3. Multi-Display / Multi-App

```
                    ┌─────────────────┐
              ┌────▶│  Browser (WebRTC)│
              │     └─────────────────┘
              │     ┌─────────────────┐
GPU Box ──────┼────▶│  TouchDesigner  │
              │     └─────────────────┘
              │     ┌─────────────────┐
              └────▶│  OBS (recording)│
                    └─────────────────┘
```

### 4. Tidal Integration (Audio-Reactive)

```
                    ┌─────────────────┐
Krea Realtime ─────▶│  TouchDesigner  │◀──── Audio Analysis
     (NDI)          │  (effects/mix)  │
                    └────────┬────────┘
                             │
                             ▼ (NDI or Spout)
                    ┌─────────────────┐
                    │  Projector/LED  │
                    └─────────────────┘
```

## Dependencies

**Python NDI bindings:**
```bash
pip install ndi-python
# or
pip install python-ndi
```

**NDI SDK:**
- Auto-bundled with some Python packages
- Or install from https://ndi.video/tools/

## API Design

### REST Strategy Decision

Two options for the REST surface:

**Option A: Dedicated Endpoints (matches current CLI style)**

```
POST /api/v1/realtime/output/ndi/start  (body: {name, groups?, ...})
POST /api/v1/realtime/output/ndi/stop
GET  /api/v1/realtime/output/ndi/status
```

Internally routes through:
```python
session = get_active_session(webrtc_manager)
apply_control_message(session, {"ndi_sender": {...}})
```

**Option B: Generic Parameters Endpoint (more powerful)**

```
POST /api/v1/realtime/parameters  (body: any Parameters fields)
```

This unlocks CLI control for *any* parameter (spout, ndi, vace, etc.) without adding endpoints per feature. Currently **CLI can't toggle Spout** because there's no such endpoint.

Trade-off: less "curated" API surface, but more useful for agent automation.

**Recommendation:** Start with Option A for NDI, but consider adding the generic endpoint in parallel—it solves existing limitations (Spout CLI) and scales better.

### CLI

```bash
# Enable NDI output
video-cli output ndi start --name "DaydreamScope"
video-cli output ndi stop
video-cli output ndi status
```

### Schema Integration

NDI should be in `Parameters` (output transport), **not** pipeline load params (model-specific):

```python
# schema.py
class NDIConfig(BaseModel):
    enabled: bool = Field(default=False)
    name: str = Field(default="DaydreamScope")
    groups: list[str] = Field(default_factory=list)

class Parameters(BaseModel):
    # ... existing fields
    ndi_sender: NDIConfig | None = None

class HardwareInfoResponse(BaseModel):
    # ... existing fields
    ndi_available: bool = False  # Optional but helpful
```

## Performance Considerations

- **Bandwidth:** 1080p30 BGRX ≈ 250 MB/s uncompressed (NDI uses SpeedHQ compression)
- **Latency:** ~1-3 frames over gigabit LAN
- **CPU:** NDI encoding uses ~5-10% CPU per stream
- **Memory:** Frame copy for each output (~5MB per frame at 1080p)

## Open Questions

- [x] **Frame format:** RGB → BGRX conversion in NDI thread (not generation thread)
- [ ] **Resolution:** Match pipeline output or allow scaling?
- [ ] **Audio:** Include audio channel? (for future audio-reactive features)
- [ ] **Discovery:** Use NDI groups for organization?
- [ ] **Compression:** Use NDI|HX for lower bandwidth? (adds latency)
- [ ] **Dynamic resolution:** Handle pipeline reload with different width/height?

## Additional Considerations

### Pause semantics

- Current pause controls both generation and playback (tied together)
- Concretely, `apply_control_message` calls `VideoProcessingTrack.pause()` and forwards `paused` to `FrameProcessor.update_parameters()`
- NDI/Spout should follow **generation pause**, not browser playback pause
- Today they're tied, so NDI stops when paused (reasonable default)
- If you ever want "keep generating but pause browser," you'd need to separate those

### Thread-safety and lifecycle

NDI needs the same lifecycle management as Spout:
- Stop thread cleanly in `FrameProcessor.stop()`
- Release NDI sender instance
- Avoid deadlocks if `stop()` called from worker vs external thread

### Session architecture note

WebRTC sessions each create their own `FrameProcessor`. Pub-sub here is **within one FrameProcessor** (one stream → many outputs), unless you refactor to "one generator shared across sessions."

## Implementation Phases

1. **Phase 0 (Prerequisite):** Move Spout enqueue from `get()` to `process_chunk()` (producer-side)
2. **Phase 1 (MVP):** Basic NDI sender alongside existing outputs, fed from `process_chunk()`
3. **Phase 2:** Configuration via schema + env vars + `update_parameters()` hooks
4. **Phase 3:** REST API + CLI integration (Option A or B)
5. **Phase 4:** Pub-sub frame broadcaster (only when needed for multi-consumer)
6. **Phase 5:** NDI receiver (input from external sources)

## References

- [NDI SDK](https://ndi.video/tools/)
- [ndi-python](https://pypi.org/project/ndi-python/)
- [TouchDesigner NDI](https://docs.derivative.ca/NDI)
- [OBS NDI Plugin](https://github.com/obs-ndi/obs-ndi)

## Supporting Materials

- [`ndi-pubsub-video-output/oai_5pro01.md`](ndi-pubsub-video-output/oai_5pro01.md) — Architecture review and gap analysis
