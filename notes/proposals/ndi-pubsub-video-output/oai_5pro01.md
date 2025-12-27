## What the proposal is aiming for (and what you already have)

The proposal’s “Pipeline → frame_broadcaster → {WebRTC, Spout, NDI, REST}” picture is directionally correct for Scope, because **FrameProcessor is already the choke point where “frames become outputs.”**

Today you effectively have three output surfaces already:

1. **WebRTC video**

   * `FrameProcessor.process_chunk()` produces `output` and enqueues **per-frame** into `self.output_queue` (destructive queue).
   * `VideoProcessingTrack.recv()` consumes via `FrameProcessor.get()`.

2. **REST “latest frame”** (non-destructive)

   * `FrameProcessor.process_chunk()` writes `latest_frame_cpu` (clone of last frame)
   * `/api/v1/realtime/frame/latest` reads it via `fp.get_latest_frame()` and encodes PNG.

3. **Spout output** (Windows)

   * You already have a Spout sender queue + thread (`spout_sender_queue`, `_spout_sender_loop`)
   * **But** frames are currently enqueued to Spout *inside* `FrameProcessor.get()` (i.e., on WebRTC consumption).

So conceptually you already have fan-out, but it is **not a true producer-side fan-out** yet.

---

## Alignment with the NDI proposal

### 1) “NDI Sender alongside Spout” matches your current patterns

Your current Spout integration establishes a strong precedent:

* “output enabled” lives inside `FrameProcessor`
* `FrameProcessor.update_parameters()` intercepts output config keys (`spout_sender`, `spout_receiver`) and applies immediately (out-of-band of the chunk boundary)
* the actual IO work happens in a dedicated background thread (good for realtime)

**NDI fits naturally into the same structure**: `NDISender`, queue, thread, start/stop hooks, and config plumbing.

### 2) The proposal’s output list matches real consumers you already have

The proposal’s list (WebRTC, Spout, REST polling) is already present in code. Adding NDI is conceptually “add one more sink.”

### 3) The “pub-sub” justification is real, because today Spout is coupled to WebRTC consumption

The proposal’s motivation—multiple independent consumers—matches a real architectural limitation in current code (detailed below).

---

## Key gaps / mismatches you should address before (or while) adding NDI

### Gap A: Spout output is **consumer-coupled**, not producer-coupled

Right now:

* `process_chunk()` enqueues frames → `output_queue`
* `VideoProcessingTrack.recv()` calls `FrameProcessor.get()`
* `FrameProcessor.get()` pops one frame from `output_queue` and **then** enqueues it to `spout_sender_queue`

This means:

* **No WebRTC consumer = no Spout output**, even if the pipeline is generating frames.
* If WebRTC is paused, slow, or backpressured, Spout output becomes indirectly throttled / starved.
* This is *the* architectural mismatch with the proposal’s “fan-out broadcaster” concept.

> If you implement NDI by copying the Spout pattern (enqueue in `get()`), you’ll reproduce the same coupling (NDI stops when WebRTC stops).

**Recommendation:** treat Spout and NDI as producer-side sinks fed from `process_chunk()`, not from `get()`.

---

### Gap B: `output_queue` is destructive and single-consumer by design

The proposal is explicit about multi-consumer; your current `output_queue` is:

* a single cursor
* destructive (any consumer drains frames)
* dynamically resized in `process_chunk()` (with a lock that protects resize/flush but not consumer reads)

This is fine for the current “one WebRTC stream” assumption, but it contradicts the “true multi-consumer broadcast” goal unless you add:

* a pub-sub ring buffer, or
* per-consumer queues, or
* separate output fan-out pipeline.

---

### Gap C: REST/CLI surface doesn’t have a generic “update parameters” endpoint

Spout is controllable from the frontend because the data channel sends `spout_sender` / `spout_receiver` in `Parameters`.

But your **CLI can’t currently toggle Spout** (or any arbitrary parameter) because:

* REST endpoints are “curated” controls (pause/run/step/prompt/cuts/style/world/playlist)
* there is **no** `/api/v1/realtime/parameters` or similar “send arbitrary parameters” endpoint
* `video_cli.py` only calls these curated endpoints

The proposal’s suggested REST endpoints `/realtime/output/ndi/start|stop|status` are consistent with your current CLI pattern: **dedicated endpoints per action**.

If you want output toggles via CLI, you either:

* add dedicated output endpoints (proposal), or
* add a generic parameter update endpoint and expose it in CLI.

---

### Gap D: No schema/docs/config for NDI yet

Today your API schema has:

* `Parameters.spout_sender`, `Parameters.spout_receiver`
* `HardwareInfoResponse.spout_available`
* no NDI equivalents

So proposal Phase 2/3 (schema/config/docs) is a real missing piece.

---

### Gap E: Format + conversion costs aren’t accounted for relative to current RGB pipeline

Your pipeline outputs frames as **RGB uint8** on CPU (after normalization), because:

* WebRTC uses `VideoFrame.from_ndarray(..., format="rgb24")`
* Spout sender converts to RGBA (adds alpha) internally

NDI commonly wants **BGRX/BGRA/UYVY/etc** (proposal uses `BGRX`).

So you need a conversion strategy that doesn’t accidentally:

* double-copy frames per output,
* or add latency by blocking the generation thread.

The proposal notes this as an “open question,” but in your current implementation it becomes very concrete.

---

## Integration points for output fan-out in the current code

### Best “hook point” for output fan-out: `FrameProcessor.process_chunk()` after CPU uint8 conversion

You have a clean stage boundary here:

```py
output = (
  (output * 255.0).clamp(0,255).to(uint8).contiguous().detach().cpu()
)
```

At this point, you have a stable CPU tensor buffer suitable for:

* WebRTC enqueue (current behavior)
* updating `latest_frame_cpu` (current behavior)
* producer-side output fan-out (Spout, NDI, file recording, etc.)

**This is the ideal place to publish frames to additional outputs**.

### Where not to attach NDI (if you want independence): `FrameProcessor.get()`

Attaching NDI to `get()` couples NDI output to the WebRTC consumer, repeating today’s Spout coupling.

### Where output config currently lives (and where NDI should mirror): `FrameProcessor.update_parameters()` + `FrameProcessor.start()`

You already support:

* initial parameters: `VideoProcessingTrack(initial_parameters)` → `FrameProcessor.start()` pops `spout_sender` and `spout_receiver`
* runtime parameters: `apply_control_message` → `FrameProcessor.update_parameters()` intercepts output keys immediately

NDI should follow the same pattern for consistency with:

* WebRTC initialParameters
* data-channel updates
* future REST output endpoints (which can call `apply_control_message`)

---

## Recommended design choices (two viable paths)

### Path 1: Minimal NDI MVP without full pub-sub (fastest, least invasive)

You can get NDI working with minimal architecture change:

1. **Add an `NDISender` implementation** (queue + thread, drop-if-full semantics).

   * Mirror Spout sender: small queue (size 2–4), background thread sends frames, stop on shutdown.

2. **Add NDI config to `Parameters`** (parallel to `spout_sender`):

   * `ndi_sender: NDIConfig | None`
   * fields: `enabled`, `name`, maybe `groups`, maybe `video_format` / `fourcc`

3. **Handle `ndi_sender` config in FrameProcessor**:

   * In `start()`: pop `ndi_sender` from initial params and apply
   * In `update_parameters()`: intercept and apply immediately (don’t queue)

4. **Feed NDI from `process_chunk()`**, not `get()`:

   * For each frame, enqueue to NDI sender queue (non-blocking, drop if full)
   * Conversion to BGRX happens in NDI thread (so generation thread stays hot)

5. **(Strongly recommended) Move Spout enqueue to producer side too**

   * Same reason: decouple from WebRTC consumption.
   * This aligns your current Spout with the proposal’s “fan-out” story, even before a broadcaster exists.

This gives you:

* WebRTC: still destructive queue
* Spout: producer-coupled
* NDI: producer-coupled
* REST: latest frame

It is *not* full pub-sub, but it achieves **multi-output** in practice (and NDI itself can have multiple receivers).

---

### Path 2: Full “FrameBroadcaster” pub-sub (matches proposal’s end state)

If you truly want “multiple independent consumers, each with cursor/backpressure,” you need a shared buffer abstraction.

The cleanest way to do this in your current architecture is:

* Replace or supplement `output_queue` with a `FrameBroadcaster` (ring buffer)
* Provide a `Consumer` handle for each output:

  * WebRTC track consumer
  * Spout sender consumer
  * NDI sender consumer
  * REST “latest” (can remain as shortcut) or REST “poll by index” consumer

**Important nuance with your current server**:
WebRTC sessions each create their own `FrameProcessor` today. The REST API also assumes a single active session (`get_active_session`). So “pub-sub” here should be **within one FrameProcessor** first (one stream, many outputs), unless you decide to refactor to “one generator shared across sessions.”

This path is a bigger change but yields:

* true independent pacing for each output
* clearer semantics for pause/step/snapshot too (potentially)

---

## REST + CLI surface: what integrates cleanly with your existing patterns

### Option A: Add dedicated NDI endpoints (matches proposal + current CLI style)

This matches how `video_cli.py` works today (purpose-built endpoints).

Suggested endpoints in `app.py` (proposal-style):

* `POST /api/v1/realtime/output/ndi/start` (body: `{name, groups?, ...}`)
* `POST /api/v1/realtime/output/ndi/stop`
* `GET  /api/v1/realtime/output/ndi/status`

Implementation-wise, you can keep things consistent by having these endpoints call:

* `session = get_active_session(webrtc_manager)`
* `apply_control_message(session, {"ndi_sender": {...}})`

…which routes through `FrameProcessor.update_parameters()` and keeps one control path.

Then CLI can add:

```bash
video-cli output ndi start --name "DaydreamScope"
video-cli output ndi stop
video-cli output ndi status
```

### Option B: Add a generic “send parameters” REST endpoint (more powerful)

If you add:

* `POST /api/v1/realtime/parameters` with a JSON body = `Parameters` (or loose dict)

…then CLI can become a general automation tool for *any* parameter, including spout/ndi/vace/etc, without adding endpoints per feature.

This would also solve the current limitation: **CLI can’t toggle Spout today**.

Given you already have `apply_control_message(session, msg)` as shared logic, this endpoint can be a thin wrapper.

**Trade-off:** slightly less “safe/curated” API surface, but much more useful for agent automation (and your CLI is explicitly “designed for agent automation”).

---

## Config / schema integration points (what to change and where)

### 1) `schema.py` (Pydantic)

Add:

* `class NDIConfig(BaseModel): enabled: bool, name: str, groups: list[str] = [] ...`
* Extend `Parameters`:

  * `ndi_sender: NDIConfig | None`
* Extend `HardwareInfoResponse`:

  * `ndi_available: bool` (optional but helpful)

### 2) `docs/api/parameters.md`

Add a section like Spout:

```js
sendParameters({
  ndi_sender: { enabled: true, name: "DaydreamScope" }
})
```

Also note:

* format expectations (RGB → BGRX conversion happens internally)
* platform requirements (NDI runtime present)

### 3) `docs/api/load.md`

I would **not** put NDI in pipeline load params unless you have a strong reason.

Reason: pipeline load params are clearly model/pipeline-specific (height/width/seed/quantization/vace/loras).
NDI is an **output transport** tied to the active session/stream, not the diffusion model. That aligns with `Parameters`, not `load_params`.

(If you later have a “headless mode: run server, auto-start NDI even without WebRTC,” then environment variables or server config makes more sense than pipeline load params.)

---

## Additional “gotchas” to consider (these matter in practice)

### 1) Pause semantics: generation pause vs playback pause

Your current pause control intentionally updates both:

* playback (`vt.pause(...)`)
* generation (`fp.paused` via queued parameters / control bus)

That’s good, but note:

* If you ever want “keep generating but pause browser playback,” you’d need to separate those.
* NDI/Spout should probably follow **generation pause**, not browser playback pause.

Today they are tied, so NDI will stop when paused (reasonable default).

### 2) Resolution changes while outputs are enabled

Spout sender tries to handle name/size changes on config updates, and it also updates width/height in `send()`.

For NDI you should decide:

* allow dynamic resolution changes mid-stream (likely yes)
* if yes, ensure the sender thread updates frame metadata per frame (proposal does)
* also consider updating config if pipeline reloads with different width/height, because your outputs otherwise won’t be recreated unless a config message is resent.

### 3) Thread-safety and lifecycle

FrameProcessor currently handles:

* stopping spout threads on `stop()`
* releasing spout sender/receiver

NDI needs the same:

* stop thread cleanly
* release NDI sender instance
* avoid deadlocks if stop() is called from worker thread vs external thread

### 4) Throughput: avoid per-frame heavy conversion on the generation thread

If you do RGB→BGRX conversion inside `process_chunk()` for every frame, you’re spending CPU time on the worker thread that should be mostly “pipeline call + enqueue.”

Better:

* enqueue raw RGB frames to NDI thread
* convert there (and drop frames if consumer can’t keep up)

This is consistent with your Spout design (IO thread).

---

## Bottom line: what I would change first to make the proposal “true” in your current architecture

1. **Decouple Spout from WebRTC consumption**
   Move Spout enqueue from `FrameProcessor.get()` into `process_chunk()` (producer-side).
   This immediately makes your current “fan-out” story real.

2. **Add NDI as a peer output sink inside FrameProcessor**
   Mirror Spout’s config+thread structure. Feed it from `process_chunk()`.

3. **Choose REST surface strategy**

   * If you want minimal and consistent with current CLI: add `/realtime/output/ndi/*` endpoints.
   * If you want the CLI to be truly automation-friendly: add a generic `/realtime/parameters` endpoint (and then NDI/spout toggles become just parameter updates).

4. **Only build FrameBroadcaster when you truly need independent cursors/backpressure**
   You don’t need full pub-sub just to get NDI working; you need it when you want:

   * multiple WebRTC consumers sharing one generator stream, or
   * robust multi-sink timing guarantees, or
   * “seek/scrub” style output history beyond `latest_frame_cpu`.

If you want, I can sketch what the smallest “OutputSink” abstraction would look like in `FrameProcessor` (so Spout/NDI/REST recording all plug in cleanly) without committing to a full broadcaster rewrite.
