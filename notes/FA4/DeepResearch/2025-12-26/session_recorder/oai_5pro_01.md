Here’s a “pre-implementation hardening” review of **`notes/proposals/server-side-session-recorder.md`**, specifically against the *actual* control flow you outlined:

REST/CLI → `app.py` → `webrtc.apply_control_message()` → `FrameProcessor.update_parameters()` mailbox → `FrameProcessor.process_chunk()` (reserved keys, ControlBus ordering, pipeline call) → offline `render_timeline.py` (transitions yes; `initCache`/`softCut` no).

I’m going to focus on **edge cases, ordering conflicts, thread-safety, and timeline format compatibility**. I’ll also suggest concrete changes to the proposal so implementation doesn’t fight the current architecture.

---

## 1) Biggest correctness issue: where/when you record prompt + transition

### What the proposal currently implies

The integration sketch suggests recording based on `merged_updates` (after some handling), and extracting prompt/transition out of it.

### Why this is a trap in your current `process_chunk()`

In `FrameProcessor.process_chunk()` today, you:

1. Drain `parameters_queue` into `merged_updates`
2. Consume reserved keys (`_rcp_step`, `_rcp_soft_transition`, snapshot keys, etc.)
3. Translate certain keys into ControlBus events, *popping them from `merged_updates`*:

   * `paused`
   * `prompts`
   * `transition`
   * `lora_scales`, etc.

So if you call `_record_control_events(merged_updates, ...)` **after** translation-to-events, you’ll often find:

* **no `prompts` key** (it’s been popped)
* **no `transition` key** (it’s been popped)

Result: you will silently **miss the main events you wanted most**.

### Hardened recommendation

Record prompt/transition from the **applied ControlBus events**, not from `merged_updates`.

Concretely:

* After you compute `events = self.control_bus.drain_pending(...)`
* While you iterate and apply them (or immediately after), detect `EventType.SET_PROMPT` and read `event.payload` (which still contains `prompts`/`transition`).

This also automatically captures prompt changes injected by:

* `_rcp_world_state` auto-compile (it injects prompts before translation → ends up as SET_PROMPT)
* `_rcp_set_style` recompile (same)

✅ This makes the recorder match “what actually applied at the boundary,” which is exactly what you want.

---

## 2) “Hard cut repeats forever while paused” edge case

### The pitfall

If you detect hard cuts by checking `reset_cache in self.parameters`, you will see it persist until the next pipeline call (because `reset_cache` is popped later, right before pipeline call). While paused (and no step), you return early and never pop it.

If your recorder logic is “if reset_cache in parameters, record hard_cut” you’ll record **the same hard cut repeatedly** on every loop iteration while paused.

### Hardened rule

Only record hard cuts on the **incoming edge**, not on persistent state.

Best source:

* `hard_cut_requested = "reset_cache" in merged_updates` (the drained mailbox update for this chunk boundary)

That ensures exactly-once recording of a hard cut command, even if the actual pipeline call is delayed.

---

## 3) You need an “initial segment” event on `start()` or you’ll export empty timelines

### Current proposal behavior

`SessionRecorder.start()` just starts a recording and clears `_last_prompt`. It does **not** record an initial event.

### What breaks

If the user:

* starts recording,
* doesn’t change prompt during the recording,
* stops recording

Then `recording.events` has no prompt events → `export_timeline()` produces **no segments**, and offline render either fails or renders nothing.

### Hardened behavior: record baseline at start

When handling `_rcp_session_recording_start` inside `process_chunk()`, you should snapshot the “current effective prompt” and record it as the first event at `wall_time=0`.

What is “current effective prompt” in your system?

* If a transition is active, the “segment prompt” is effectively the **transition target** (because the current prompt in `self.parameters["prompts"]` can remain the old one during transition-only mode).
* Otherwise, use `self.parameters.get("prompts")`.

So, at start:

* Determine initial prompt from:

  1. `transition.target_prompts[0]` if `transition` exists
  2. else `prompts[0]` if prompts exist
  3. else (optional) `self._compiled_prompt` if you want style-layer fallback

Also: consistent with your frontend timeline-export note: if recording starts mid-segment, you **should not** replay the segment’s transition at t=0. So for the initial event, **do not include transitionSteps/method**, even if a transition is currently active.

---

## 4) Thread-safety: the proposed `/status` endpoint reads recorder internals unsafely

### Problem

The proposal’s status endpoint accesses:

* `fp.session_recorder._recording`
* `fp._last_recording_path`

Those are mutated from the worker thread inside `process_chunk()`. FastAPI endpoints run in the server thread. You risk:

* torn reads / inconsistent state
* rare exceptions if you read while list append happens
* subtle bugs (status says is_recording but durations mismatch)

### Hardened approach

Keep the “write-only from worker thread” invariant, but expose a **thread-safe snapshot** for status queries.

Two easy options:

**Option A: a lock inside SessionRecorder**

* `SessionRecorder` has `self._lock = threading.Lock()`
* Worker thread holds lock during start/record/stop updates
* `/status` holds lock during snapshot read

**Option B: atomic snapshot dict**

* Worker thread maintains `self._status_snapshot: dict`
* Updates it by *replacing the entire dict* (atomic pointer swap under GIL)
* `/status` reads `snapshot = fp.session_recorder.status_snapshot` without locks

If you want minimal friction, Option B is often enough.

Either way: stop exposing `_recording` directly from REST.

---

## 5) PipelineManager.get_status_info() has side effects that can bite you

Your `PipelineManager.get_status_info()` clears error state when status is ERROR (“returned once, then cleared, status reset to NOT_LOADED”). If you call it during recording start, you may accidentally:

* clear a load error
* reset status transitions
* make debugging harder

### Hardened recommendation

Add a non-mutating “peek” method, e.g.

* `PipelineManager.peek_status_info()` that never clears errors/resets state.

Or: in recorder start, only call status when status is LOADED; otherwise start should fail with a clear message (“pipeline not loaded”).

This is worth documenting explicitly in the proposal because otherwise you’ll get “why did my error disappear?” reports.

---

## 6) Timeline schema compatibility: you’re *mostly* fine, but tighten a few things

### Good news

`render_timeline.py` models have `extra="ignore"`, so adding segment keys like:

* `initCache`
* `softCut`
* `startChunk`/`endChunk`
  won’t break parsing.

### But: `settings.pipelineId` is required

`TimelineSettings.pipelineId: str` (not optional). Your export currently uses:

* `pipeline_id: str | None` in `SessionRecording`
* and then `"pipelineId": recording.pipeline_id`

If it’s `None`, offline render will fail validation. Even if “it should never happen,” harden it anyway:

* enforce pipeline loaded at start; or
* default to `os.getenv("PIPELINE")` (if that’s your selection mechanism); or
* default to a known pipeline id (but better to fail loudly)

### Also tighten naming consistency

Your server export uses:

* `recording: { durationSeconds, durationChunks, ... }`

Frontend proposal uses:

* `recordingDuration` (older) and mentions “extra metadata”.

It’s fine to differ, but if you want fewer downstream tools:

* adopt one canonical metadata shape across both server and frontend exports.

Recommended canonical:

```json
"recording": {
  "durationSeconds": ...,
  "durationChunks": ...,
  "startChunk": ...,
  "endChunk": ...
}
```

(Which your server-side already does.)

### Prompt weight mismatch isn’t fatal, but document it

REST endpoints use weight `1.0`. Timeline defaults use `100.0`. If you’re only ever using single-prompt segments, it’s irrelevant. If you later capture multi-prompt blending, it matters.

Harden doc with:

* “We store weights exactly as applied to the pipeline (often 1.0 from REST/CLI).”
* If you *want* timeline parity with frontend exports, you could scale, but only do that if you’re sure weight semantics match.

---

## 7) Soft cut semantics: timeline needs more than “bias/chunks” for faithful replay later

Your current soft transition implementation has nuance:

* It can restore to an *explicit bias* if provided in the same message
* Or restore to “unset” if the key wasn’t set originally
* It cancels if explicit `kv_cache_attention_bias` arrives mid-soft-transition

If you want offline replay later, recording only:

```json
softCut: { bias, chunks }
```

is not enough to reproduce restore behavior perfectly.

### Hardened MVP stance

It’s totally fine to record softCut as metadata for now (renderer ignores), but the proposal should explicitly acknowledge:

* “This is not sufficient to replay restore semantics later; we’d need to also record restore target / whether bias was ‘unset’.”

Minimal extra fields you might want in recorded event:

* `soft_cut_restore_bias: float | null`
* `soft_cut_restore_was_set: bool`
  …but I’d only add that if you’re serious about implementing renderer support soon.

---

## 8) Segment boundary rules: define them precisely (or you’ll get weird exports)

Right now export logic says:

* new segment starts whenever a recorded event has a prompt (including synthesized prompt for cut-only events)
* segment ends at next prompt-bearing event or recording end

That’s reasonable, but harden it with explicit rules:

### Recommended boundary rules

A recorded event should start a new segment if **any** of these are true:

1. prompt text changes (or prompt list changes)
2. transition starts (even if prompt is derived from target_prompts)
3. hard cut occurs (even if prompt unchanged)
4. soft cut occurs (even if prompt unchanged) — *optional; you may decide soft cut is a modifier, not a boundary*

If you treat soft cut as a segment boundary, you’ll get more segments (and better replay potential). If not, you need a way to attach softCut timing that doesn’t rely on segment boundaries (i.e., separate events list). The proposal currently chooses boundary, which is fine for now.

### Also add deduping

You will inevitably get no-op events like:

* set prompt to same prompt again
* transition events to same target prompt

You probably want:

* do not create a new segment if prompt unchanged **and** no hard/soft cut flag present **and** no transition params present

---

## 9) API ergonomics: start/stop should report “queued vs applied”, and status should report errors

Right now the proposal endpoints always return `recording_started` / `stop_requested`.

Harden it:

* If `apply_control_message()` returns False (queue saturation or fp not ready), return `503` with detail.
* Status should include:

  * `is_recording`
  * `start_chunk`
  * `current_chunk`
  * `duration_seconds`
  * `events_count`
  * `last_saved_path` (optional)
  * `last_error` (if save failed)

Also: consider adding a **download** endpoint early if you ever control Scope remotely (even over LAN). Returning filesystem paths only works for “local machine runs server and client.”

---

## 10) Concrete implementation sketch that fits current `FrameProcessor.process_chunk()` ordering

This is the minimal change that avoids the prompt/transition popping issue:

1. After draining `parameters_queue` into `merged_updates`, handle:

* `_rcp_session_recording_start/stop` (consume)
* `_rcp_soft_transition` (consume; capture the bias/chunks for recording-edge)
* store `hard_cut_requested = "reset_cache" in merged_updates`

2. Continue existing logic: translate to events, apply events.

3. Before the “if paused and not step_requested: return” early return, do recording:

* If recording started this boundary: record initial prompt baseline at wall_time=0
* For each applied `EventType.SET_PROMPT`:

  * derive prompt from payload (prefer prompts else transition.target_prompts)
  * record transition metadata if present
* If `hard_cut_requested` and no SET_PROMPT event: record cut-only event using last prompt
* If softCut triggered and no SET_PROMPT: record cut-only event using last prompt

This ensures you capture events even while paused (because it runs before pause return), but avoids repeated recordings by relying on “edge flags” from `merged_updates`, not persistent `self.parameters`.

---

## 11) Update the proposal doc with a short “Known limitations” list that matches reality

Add a section like:

* **Not captured (MVP):** parameter changes beyond prompts/transition/cuts (denoise steps, seeds, LoRAs, world state, style changes) unless they result in SET_PROMPT events.
* **Soft cut replay:** recorded but not replayable until `render_timeline.py` supports it; even then needs restore-target nuance.
* **Hard cut replay:** recorded as `initCache`, ignored by renderer until implemented.
* **Recording start mid-transition:** we record transition target prompt as baseline and do not replay transition at t=0 (approximation).

This will prevent “why doesn’t my offline render match exactly?” confusion.

---

## 12) Quick checklist of additional edge cases to explicitly cover in the doc

* Start recording when:

  * no pipeline loaded
  * no prompt set yet
  * transition active with no prompts field (transition-only mode)
* Stop recording while:

  * paused (no chunk progress)
  * pipeline call in-flight (stop processed next boundary)
* Multiple sessions connected (current `get_active_session` will error)
* Pipeline reload mid-recording (should fail or end recording)
* Long sessions → events list size (should be fine, but document)
* Save failure (permissions, path missing) and how API reports it

---

## Suggested “doc edits” summary (high signal)

If you only change a few things in `server-side-session-recorder.md`, make them these:

1. **Recording source of truth:** record prompt/transition from **ControlBus SET_PROMPT events** (not `merged_updates`).
2. **Record baseline at start:** always record an initial prompt event at wall_time=0 (derived from transition target if present).
3. **Avoid repeated hard cuts:** record hard_cut only when `"reset_cache" in merged_updates` (edge), not when present in `self.parameters`.
4. **Thread-safe status:** don’t read `_recording` from FastAPI thread; expose a snapshot.
5. **PipelineManager peek:** don’t call `get_status_info()` if it clears errors; use a peek or gate on LOADED.
6. **Ensure `pipelineId` is never None** in exported timeline.

---

If you want, I can also rewrite the relevant sections of `server-side-session-recorder.md` (keeping your structure) into an “implementation-ready” version with the corrected ordering and explicit edge-case behavior—without changing the core idea.
