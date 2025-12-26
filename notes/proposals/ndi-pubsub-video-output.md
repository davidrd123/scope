# NDI / Pub-Sub Video Output

> Status: Draft
> Date: 2025-12-26

## Summary

Add NDI (Network Device Interface) output to enable network-based video streaming with multiple subscribers. This enables:

1. **Remote consumption** — Stream from GPU box to local TouchDesigner/OBS
2. **Multi-consumer** — Multiple apps can subscribe to the same stream
3. **V2V pipelines** — Feed output to secondary video-to-video models
4. **Cross-platform** — Works Mac ↔ Windows ↔ Linux

## Current Architecture

```
Pipeline → output_queue (destructive) → WebRTC → Single Browser
                ↓
          Spout (Windows, local only)
```

**Limitations:**
- WebRTC queue is destructive (one consumer)
- Spout is local-only (shared GPU memory)
- No network streaming for external apps
- No multi-consumer broadcast

## Proposed Architecture

```
Pipeline → frame_broadcaster (non-destructive)
                ↓
    ┌───────────┼───────────┬──────────────┐
    ↓           ↓           ↓              ↓
 WebRTC      Spout        NDI           REST
 (browser)  (local Win)  (network)    (polling)
```

### Key Changes

1. **Frame Broadcaster** — Replace destructive queue with pub-sub pattern
2. **NDI Sender** — New output consumer for network streaming
3. **Consumer Registry** — Each output has independent read cursor

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

| Application | NDI Support |
|-------------|-------------|
| TouchDesigner | Native (NDI In TOP) |
| OBS | Plugin (obs-ndi) |
| vMix | Native |
| Resolume | Native |
| FFmpeg | Via `libndi` |
| VLC | Plugin |

## Implementation

### Phase 1: NDI Sender (MVP)

Add NDI output alongside existing Spout:

```python
# src/scope/server/ndi/sender.py

import NDIlib as ndi
import numpy as np
from queue import Queue
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
            except:
                continue

            # Frame is HWC uint8 numpy array
            h, w, c = frame.shape
            video_frame.xres = w
            video_frame.yres = h
            video_frame.FourCC = ndi.FOURCC_VIDEO_TYPE_BGRX
            video_frame.data = frame.tobytes()
            video_frame.line_stride_in_bytes = w * 4

            ndi.send_send_video_v2(self._sender, video_frame)

    def send_frame(self, frame: np.ndarray):
        """Non-blocking frame send. Drops if queue full."""
        try:
            self.queue.put_nowait(frame)
        except:
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

```python
# In frame_processor.py

class FrameProcessor:
    def __init__(self, ...):
        # ... existing init
        self.ndi_sender: NDISender | None = None

    def _initialize_ndi(self, config: NDIConfig):
        if config.enabled:
            self.ndi_sender = NDISender(name=config.name)
            self.ndi_sender.start()

    def _output_frame(self, frame: np.ndarray):
        # Existing: WebRTC queue
        self.output_queue.put(frame)

        # Existing: Spout (Windows)
        if self.spout_sender:
            self.spout_sender_queue.put(frame.copy())

        # NEW: NDI
        if self.ndi_sender:
            self.ndi_sender.send_frame(frame.copy())
```

### Phase 3: Configuration

```python
# In schema.py

class NDIConfig(BaseModel):
    enabled: bool = Field(default=False)
    name: str = Field(default="DaydreamScope")
    # Optional: groups for organization
    groups: list[str] = Field(default_factory=list)
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

### REST Endpoints

```
POST /api/v1/realtime/output/ndi/start
POST /api/v1/realtime/output/ndi/stop
GET  /api/v1/realtime/output/ndi/status
```

### CLI

```bash
# Enable NDI output
video-cli output ndi start --name "DaydreamScope"
video-cli output ndi stop
video-cli output ndi status
```

### Load Params

```python
pipeline_manager.load(
    pipeline_id="krea-realtime-video",
    ndi_sender=NDIConfig(enabled=True, name="DaydreamScope"),
    # ...
)
```

## Performance Considerations

- **Bandwidth:** 1080p30 BGRX ≈ 250 MB/s uncompressed (NDI uses SpeedHQ compression)
- **Latency:** ~1-3 frames over gigabit LAN
- **CPU:** NDI encoding uses ~5-10% CPU per stream
- **Memory:** Frame copy for each output (~5MB per frame at 1080p)

## Open Questions

- [ ] **Frame format:** RGB vs BGRX vs YUV? (NDI prefers BGRX/UYVY)
- [ ] **Resolution:** Match pipeline output or allow scaling?
- [ ] **Audio:** Include audio channel? (for future audio-reactive features)
- [ ] **Discovery:** Use NDI groups for organization?
- [ ] **Compression:** Use NDI|HX for lower bandwidth? (adds latency)

## Implementation Phases

1. **Phase 1 (MVP):** Basic NDI sender alongside existing outputs
2. **Phase 2:** Configuration via schema + env vars
3. **Phase 3:** REST API + CLI integration
4. **Phase 4:** Pub-sub frame broadcaster for true multi-consumer
5. **Phase 5:** NDI receiver (input from external sources)

## References

- [NDI SDK](https://ndi.video/tools/)
- [ndi-python](https://pypi.org/project/ndi-python/)
- [TouchDesigner NDI](https://docs.derivative.ca/NDI)
- [OBS NDI Plugin](https://github.com/obs-ndi/obs-ndi)
