## REST + CLI smoke testing commands

These are manual + automated checks for the `/api/v1/realtime/*` REST control surface and `video-cli`.

Important notes:
- REST control targets the **single** WebRTC session with `pc.connectionState == "connected"`. If **0** or **>1** sessions are connected, REST returns an error.
- `pause` means **two things**: track/output pause (`VideoProcessingTrack.pause`) + generation pause (`FrameProcessor.paused`). REST does both via `apply_control_message()`.
- `step` generates **one chunk** even while generation is paused. In V2V mode, step will still wait for required input frames.

### 1) Run tests (local)

```bash
uv run pytest -q \
  tests/test_realtime_rest_control.py \
  tests/test_realtime_rest_api.py \
  tests/test_video_cli.py

# optional: run everything
uv run pytest -q
```

### 2) Start server + create a WebRTC session

In one terminal:

```bash
uv run daydream-scope
```

Then open the browser UI (`http://localhost:8000`) and start a stream so a WebRTC session is created and connected.

### 3) Drive via CLI (recommended)

```bash
# configure URL if remote
export VIDEO_API_URL="http://localhost:8000"

uv run video-cli state
uv run video-cli prompt "a dog portrait"
uv run video-cli pause
uv run video-cli step
uv run video-cli frame --out notes/research/2025-12-24/incoming/rest_endpoint/test_out/dog_frame.png
uv run video-cli run --chunks 3
uv run video-cli frame --out notes/research/2025-12-24/incoming/rest_endpoint/test_out/frame.png
```

### 4) Drive via curl (equivalent)

```bash
curl -sS http://localhost:8000/api/v1/realtime/state
curl -sS -X POST http://localhost:8000/api/v1/realtime/pause
curl -sS -X POST "http://localhost:8000/api/v1/realtime/run?chunks=3"
curl -sS -X POST http://localhost:8000/api/v1/realtime/step
curl -sS -X PUT http://localhost:8000/api/v1/realtime/prompt \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"a dog portrait"}'
curl -sS http://localhost:8000/api/v1/realtime/frame/latest \
  --output notes/research/2025-12-24/incoming/rest_endpoint/test_out/frame.png
```
