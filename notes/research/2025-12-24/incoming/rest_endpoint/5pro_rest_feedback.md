Below is a focused review of the “REST endpoints that reuse the existing WebRTC control path” plan, specifically around race conditions / thread-safety, `FrameProcessor` queue semantics, pause/resume/step correctness, and some concrete integration suggestions.

I’m basing this directly on the code you shared in:

* `src/scope/server/webrtc.py` (session + data-channel control path)
* `src/scope/server/tracks.py` (`VideoProcessingTrack` pause/output behavior + lazy `FrameProcessor`)
* `src/scope/server/frame_processor.py` (mailbox merge, pause/step semantics, output queue)
* `src/scope/realtime/control_bus.py` (deterministic boundary ordering)

---

## 1) Biggest correctness issue: “pause” must hit **two** mechanisms

Right now “pause” is *two different things*:

1. **Playback pause** (output freezing)

   * `VideoProcessingTrack.pause(paused: bool)` toggles `VideoProcessingTrack._paused`
   * also flushes `frame_processor.output_queue` when pausing to freeze immediately

2. **Generation pause** (stop calling the pipeline)

   * `FrameProcessor.process_chunk()` only stops generating when `self.paused == True`
   * `self.paused` is driven by the *event translation* of `"paused"` in `merged_updates`:

     ```py
     if "paused" in merged_updates:
         paused_val = merged_updates.pop("paused")
         enqueue(PAUSE) / enqueue(RESUME)
     ```
   * and then applied at chunk boundary by `ControlBus`

**Why this matters for REST design:**
If your REST `POST /api/pause` *only* calls `session.video_track.pause(True)` (as your approach summary suggests), generation can keep running. In fact, while paused, `VideoProcessingTrack.recv()` explicitly *still outputs new frames if they appear*:

```py
if paused:
    frame_tensor = self.frame_processor.get()
    if frame_tensor is not None:
        frame = VideoFrame.from_ndarray(...)
    else:
        frame = self._last_frame
```

So, if generation continues, your “paused” output will continue to advance (defeats pause semantics).

✅ **Recommendation**
Make pause/resume REST endpoints always do **both**:

* `video_track.pause(True|False)` (freeze/resume output immediately)
* `frame_processor.update_parameters({"paused": True|False})` (pause/resume generation at boundary)

Even better: wrap these into **one helper** so no caller can forget half of it:

```py
def set_paused(session: Session, paused: bool):
    vt = session.video_track
    vt.initialize_output_processing()
    vt.pause(paused)  # output behavior
    vt.frame_processor.update_parameters({"paused": paused})  # generation behavior
```

---

## 2) `frame_processor` lazy init: REST must not “fire-and-forget” into `None`

### What exists today

`VideoProcessingTrack.frame_processor` is created lazily in `recv()`:

```py
def initialize_output_processing(self):
    if not self.frame_processor:
        self.frame_processor = FrameProcessor(...)
        self.frame_processor.start()
```

But REST endpoints may be called before the outbound WebRTC sender has pulled the first frame (i.e., before `recv()` ever runs).

### The WebRTC data channel already has this weakness

In `webrtc.py`, you currently do:

```py
if session.video_track and hasattr(session.video_track, "frame_processor"):
    session.video_track.frame_processor.update_parameters(data)
```

`hasattr(...)` is always true (attribute exists), even when `frame_processor is None`. So early messages can throw and hit the exception handler.

✅ **Recommendation**
For REST endpoints (and honestly also for the WebRTC handler), do **one of**:

* **Option A (minimal, consistent):** always call `video_track.initialize_output_processing()` before touching `frame_processor`
* **Option B (more robust):** initialize the output processing eagerly at session creation (in `handle_offer`) so both REST and data-channel control never race it.

If you keep lazy init, REST handlers should do:

```py
vt = session.video_track
vt.initialize_output_processing()
fp = vt.frame_processor
if fp is None or not fp.running:
    # return 409/503: "processor not ready"
```

---

## 3) `FrameProcessor.parameters_queue` dropping updates is a real problem for REST controls

### Current behavior

`FrameProcessor.update_parameters()` uses a bounded queue:

```py
self.parameters_queue = queue.Queue(maxsize=8)

try:
    self.parameters_queue.put_nowait(parameters)
except queue.Full:
    logger.info("Parameter queue full, dropping parameter update")
    return False
```

But the worker thread is designed with “mailbox semantics” (drain all, last-write-wins) and expects to eventually see updates.

### Why this is risky for REST endpoints

For REST calls like:

* `POST /api/pause`
* `POST /api/step`
* `POST /api/run`

…it is **not acceptable** to “maybe drop” the command. If the queue is full at that instant, you can end up with:

* output paused, generation not paused (if the `"paused": True` update dropped)
* `step` request dropped (no chunk generated, REST call lies / hangs)
* prompt update dropped (state mismatches UI/agent)

✅ **Recommendation: make the queue “mailbox-like” for control**
If you want mailbox semantics, implement mailbox semantics:

* On `queue.Full`, discard **older** pending updates to make room for the newest update.
* You can do that by draining 1 item (or draining all) before `put_nowait()` again.

Example pattern:

```py
except queue.Full:
    try:
        self.parameters_queue.get_nowait()  # drop oldest
    except queue.Empty:
        pass
    self.parameters_queue.put_nowait(parameters)  # try again
```

Or, more aggressively (true mailbox):

* drain everything
* enqueue only the latest update

This aligns with the “drain all and merge” design in `process_chunk()`.

**Special note for `_rcp_step`:** it’s *additive* (increments `_pending_steps`). If you collapse multiple step requests into one, prefer sending `_rcp_step: N` rather than N separate updates.

---

## 4) Subtle race: output queue resizing can defeat `VideoProcessingTrack.pause()` flush

When pausing, you flush:

```py
while True:
    try:
        self.frame_processor.output_queue.get_nowait()
    except queue.Empty:
        break
```

But `FrameProcessor.process_chunk()` can *replace* the queue object:

```py
old_queue = self.output_queue
self.output_queue = queue.Queue(maxsize=target)
while not old_queue.empty():
    frame = old_queue.get_nowait()
    self.output_queue.put_nowait(frame)
```

If pause-flush runs after `self.output_queue` is swapped but before frames are transferred, you flush the new (empty) queue, then transfer repopulates it — and “stale frames” survive the pause.

✅ **Recommendation**
Put an explicit lock around **any** mutation/replacement of `output_queue`, and take the same lock during pause flush.

* Add `self.output_queue_lock = threading.Lock()` in `FrameProcessor`
* Wrap the resize block and any external flushes with it

Or: move “flush output queue on pause” into the worker thread at PAUSE event application time (single-threaded ownership of the queue object).

---

## 5) STEP semantics: make REST `POST /api/step` send the same “reserved key” as WebRTC

You already have a great internal contract:

* WebRTC `"type": "step"` → translated to `{"_rcp_step": True}`
* `FrameProcessor` consumes `_rcp_step` and generates exactly one chunk even if paused

✅ **Recommendation**
Make REST `POST /api/step` do the same thing:

* Ensure output pause is active (so playback freezes and step output is “one-shot” visible)
* Ensure generation pause is active
* Send `_rcp_step`

**Best practice:** send **one combined update** so pause + step cannot be split/dropped independently:

```py
fp.update_parameters({"paused": True, "_rcp_step": 1})
vt.pause(True)
```

(You can swap the ordering; the important point is *both happen* and pause+step are in the same update dict if possible.)

### Also: step completion acknowledgement

Right now step completion is only reported via `snapshot_response_callback` as:

```py
{"type": "step_response", "chunk_index": ..., "success": ..., "error": ...}
```

That’s perfect for WebRTC notifications, but REST probably wants a synchronous response.

✅ **Recommendation**
Add a small “wait for step” bridge:

* Store a thread-safe `queue.Queue()` (or `threading.Event` + payload) on the session/controller object for step responses
* Temporarily wrap/tee the `snapshot_response_callback` to also push step responses into that queue
* REST handler awaits it with timeout (`asyncio.to_thread(q.get)` + `asyncio.wait_for`)

This avoids polling `chunk_index` and gives you reliable “step finished” semantics.

---

## 6) `/api/run` with optional `chunks`: you already have a clean mapping

Your code already supports `_rcp_step` as an integer:

```py
if isinstance(step_val, int) and not isinstance(step_val, bool):
    step_count = max(1, step_val)
self._pending_steps += step_count
```

So you can map:

* `POST /api/run` **without** chunks → resume continuous generation

  * `{"paused": False}` and `vt.pause(False)`

* `POST /api/run` **with** chunks=N → generate exactly N chunks *while staying paused*

  * `{"paused": True, "_rcp_step": N}` and `vt.pause(True)`

This matches the CLI “run N chunks” concept nicely without inventing a new control surface.

The only caveat: if your “run N chunks” is expected to *temporarily unpause then re-pause*, this implementation doesn’t do that — it does “N step-chunks while paused”. But given your step design, that’s probably what you want anyway.

---

## 7) `/api/state`: define what “paused” means and avoid mismatched sources

There are two “paused” values:

* `VideoProcessingTrack._paused` (playback/output behavior)
* `FrameProcessor.paused` (generation behavior)

They can diverge if any code path forgets to set both (and REST is exactly where that can happen unless you enforce the helper).

✅ **Recommendation**
Return both, or define a single authoritative meaning:

* If CLI means “generator paused”, report `FrameProcessor.paused`
* If CLI means “viewer paused”, report `VideoProcessingTrack._paused`
* If you want a single boolean, consider:

  * `paused = fp.paused` as the “truth”
  * and treat output pause as derived / best-effort

I’d personally return:

```json
{
  "paused": true,
  "paused_generation": true,
  "paused_output": true,
  "chunk_index": 123,
  "prompt": "..."
}
```

…at least during bring-up, so you can detect drift immediately.

### Prompt extraction

“Current prompt” most realistically lives in:

* `FrameProcessor.parameters.get("prompts")`

But it’s a list of `{text, weight}` items, not a single string.

✅ **Recommendation**
For `/api/state`, choose a stable projection:

* If exactly one prompt, return its `.text`
* If multiple, either join (with weights) or return list in a separate field

---

## 8) `/api/frame/latest`: don’t call `FrameProcessor.get()` from REST

Calling `frame_processor.get()` will **consume** a frame from `output_queue`, which will change what the WebRTC stream outputs (and can create weird “skips”).

✅ **Recommendation**
Expose a *non-destructive* “latest frame” buffer.

Two good options:

### Option A: store latest frame inside `FrameProcessor` (recommended)

When a chunk is produced, set:

* `self.latest_frame_cpu = output[-1]` (or the most recent frame)
* protected by a small lock

Then REST `/api/frame/latest` can encode that tensor without touching `output_queue`.

### Option B: reuse `VideoProcessingTrack._last_frame`

This only works if WebRTC is actively pulling frames (because `_last_frame` is set in `recv()`).

It also needs a lock if you want to be strict (since REST and `recv()` can interleave).

---

## 9) “Single active session” selection: use `connectionState == "connected"` (or clearly define “active”)

Your current helper:

```py
get_active_session_count:
  count sessions where pc.connectionState not in ["closed", "failed"]
```

This includes “new”, “connecting”, and “disconnected”.

✅ **Recommendation**
For REST control, pick one deterministic definition, e.g.:

* “active = connected”
* or “active = connected OR connecting” (if you want to allow early calls)

And implement selection as:

* filter sessions by that predicate
* if len != 1 → 409 with a clear error
* return session id in the response for debugging

Also consider a cleanup policy for “disconnected” sessions that never close.

---

## 10) Integration suggestion: don’t duplicate the WebRTC message translation logic

You already have a mini-protocol in the data channel handler:

* `{"type":"step"}` → reserved keys
* `{"type":"restore_snapshot"}` → reserved keys
* pop `"type"` otherwise
* call `video_track.pause(...)` if `"paused"` present
* forward to `frame_processor.update_parameters(...)`

✅ **Recommendation**
Put that logic in one shared function and reuse it from:

* WebRTC data-channel handler
* REST endpoints

Example shape (conceptually):

```py
def apply_control_message(session, msg: dict):
    vt = session.video_track
    vt.initialize_output_processing()

    if "paused" in msg:
        vt.pause(msg["paused"])

    msg = translate_protocol_to_reserved_keys(msg)

    vt.frame_processor.update_parameters(msg)
```

This is the easiest way to prevent REST controls from subtly diverging from WebRTC controls.

---

# Bottom line

Your plan is viable *if* you address these key points:

1. **Pause must update both output + generation** (or you will “pause” while still generating)
2. **REST must ensure `FrameProcessor` exists** (lazy init otherwise drops/errors)
3. **Don’t allow control updates to be dropped** by `parameters_queue` for critical commands
4. **Avoid consuming `output_queue`** for `/api/frame/latest`; add a non-destructive latest-frame buffer
5. **Make session selection deterministic** (`connected` vs “not closed/failed”)
6. **Share the control-message translation code** between WebRTC and REST paths

If you want, I can also propose a minimal patch layout (where to add helpers + what tiny fields/locks to add) that keeps the “no new control surface” requirement while making the REST controls reliable.
