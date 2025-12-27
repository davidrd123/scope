Here’s a vet of `notes/proposals/server-side-session-recorder.md` against the *actual* control-plane + renderer code you shared, with emphasis on what’s missing for end-to-end replay (especially `initCache`/`softCut`), and the concrete work needed to support it.

## High-level verdict

The proposal is **directionally correct and feasible**:

* **Tap point is right:** recording inside `FrameProcessor.process_chunk()` is the only place that sees the “merged truth” across REST, WebRTC data channel, playlist helpers, dev console, etc.
* **Thread-safety strategy is right:** routing start/stop through reserved keys and executing recorder mutation on the worker thread matches how snapshot/restore/step/soft-transition are handled today.
* **Semantic framing is right:** “chunk index is the primary timebase” matches your deterministic chunk-boundary application model (ControlBus + pipeline calls).

But there are a few **correctness bugs in the proposal’s pseudocode** (field names, types, status enums), and a couple of **completeness gaps** that matter if you want faithful replays (LoRA/style changes; timebase drift; video-mode).

The biggest practical gap remains exactly what you called out:

> **Offline replay (`cli/render_timeline.py`) currently drops/ignores `initCache` and `softCut`** because the schema doesn’t define them and Pydantic is `extra="ignore"`. Even if the server records them perfectly, the renderer will silently throw them away and replays will diverge.

---

## What the proposal gets right (aligned with current code)

### 1) Hard cut semantics as an edge-triggered event

Your realtime semantics in `FrameProcessor` are:

* REST/playlist sends `reset_cache=True`
* `FrameProcessor` later does `reset_cache = self.parameters.pop("reset_cache", None)`
* and passes `init_cache` for that call only.

So recording “hard cut” as an **event** (not sticky state) is correct.

### 2) Soft cut semantics live in FrameProcessor and are chunk-based

Your realtime implementation of `_rcp_soft_transition`:

* applies a temporary `kv_cache_attention_bias` immediately,
* lasts for **N chunks (pipeline calls)**,
* restores to either:

  * the previous explicit bias value, or
  * “unset” (delete the key).

So recording soft cut as `(temp_bias, num_chunks, restore target)` is correct, and your revised proposal adding `restoreBias` / `restoreWasSet` is the right direction.

### 3) Recording prompt changes via the ControlBus is the correct conceptual approach

In your current `FrameProcessor.process_chunk()`:

* `merged_updates["prompts"]` / `merged_updates["transition"]` get **popped** and translated into `ControlBus` events.
* If you record from `merged_updates` *after translation*, you’ll miss them.

So “record from applied ControlBus events” is the correct strategy.

---

## Correctness issues in the proposal that need fixing

These are not conceptual disagreements—they’re “this won’t work as written” issues.

### A) ControlBus event attribute name mismatch (`event.event_type` vs `event.type`)

Your `ControlEvent` dataclass (in `scope/realtime/control_bus.py`) defines:

* `type: EventType`
* `payload: dict`

But the proposal’s pseudocode checks `event.event_type == EventType.SET_PROMPT`.

That needs to be `event.type == EventType.SET_PROMPT`.

### B) PipelineManager status / peek API mismatches

Current `PipelineManager.get_status_info()` returns:

* `"status": current_status.value` (e.g. `"loaded"`, `"error"`, etc.)
* and it **clears error state** and may reset to NOT_LOADED when read in ERROR state.

The proposal introduces `peek_status_info()` and then compares `status.get("status") != "LOADED"`. That’s inconsistent with both the current return shape and the enum values:

* If you keep using `.value`, compare to `"loaded"`.
* If you switch to `.name`, compare to `"LOADED"`.

Also, the proposal’s `peek_status_info()` example references `self._error`, but the current code uses `self._error_message`.

**Bottom line:** you do need a non-mutating status read, but it must be implemented with the existing lock and with the correct field names/values.

### C) Baseline prompt extraction still isn’t type-safe as described

You correctly identified the risk in `review01.md`: `_compiled_prompt` is **not** a simple string prompt.

But in the proposal revision, `_get_current_effective_prompt()` returns `self._compiled_prompt.positive`, and in `control_state.py` that `positive` field is a **list of dicts**, not a single string:

```py
class CompiledPrompt:
    positive: list[dict]
```

So returning it as `prompt_text: str | None` is still wrong.

Also, your server `FrameProcessor` code (the snippet you provided) treats the compiler output as having `.prompts`, `.prompt`, and prompt items with `.to_dict()`, which doesn’t match `control_state.CompiledPrompt` at all. That suggests there may be **two different “CompiledPrompt” shapes in the codebase** (or an outdated annotation), which makes the proposal’s baseline logic extra fragile unless it uses the *actual* pipeline-facing `parameters["prompts"]`.

**Practical fix:** for baseline prompt, prefer:

1. `self.parameters.get("transition")["target_prompts"]` (if present)
2. else `self.parameters.get("prompts")`
3. only then consult compiler outputs, but normalize them into the same `[{text, weight}]` shape.

### D) “Standardized on 1.0 everywhere” is not currently true in offline code

Even if the recorder emits weight `1.0` explicitly (so replay works for recorded timelines), `render_timeline.py` still defaults prompt weights to `100.0` for segments defined via `text`, and its `TimelinePromptItem` default is `100.0`.

That’s not fatal for recorded files (because you include weights), but it *is* a source of drift between “manual offline timelines” and realtime semantics.

---

## Completeness gaps (beyond initCache/softCut) you should decide on

These aren’t blockers for an MVP, but they will produce “why doesn’t replay match?” questions later.

### 1) LoRA/style changes aren’t recorded (yet)

In realtime, style changes can trigger:

* prompt recompilation *and*
* `lora_scales` one-shot updates (zero others, set active style scale).

Your recorder proposal currently records only prompt/transition + cuts. That means a replay may match text but **miss the LoRA scale switching**, which can be a huge visual mismatch.

If you want fidelity for “style swaps”, you likely need to record either:

* `SET_LORA_SCALES` events (full list), or
* high-level “active style changed” events (and have offline renderer replicate LoRA behavior).

### 2) Chunk-based vs time-based scheduling mismatch

The proposal exports both:

* `startTime/endTime` (wall clock)
* `startChunk/endChunk` (chunk timebase)

…but `render_timeline.py` currently selects segments strictly by `current_time = produced_frames / fps`, i.e. wall clock.

If your realtime session had pauses, stalls, or variable throughput, wall-time scheduling is inherently going to drift.

### 3) Video/V2V sessions won’t replay in offline renderer

`render_timeline.py` currently rejects `inputMode != "text"`. A server-side recording taken from a V2V/VACE flow can’t be replayed offline with the current CLI.

---

## The key gap: why recordings won’t fully replay today

### What happens today

Even if your server exports a timeline segment like:

```json
{
  "prompts": [{"text":"...", "weight":1.0}],
  "initCache": true,
  "softCut": {"bias":0.1, "chunks":2, "restoreBias":0.3, "restoreWasSet": true}
}
```

`render_timeline.py` will:

* parse the file with Pydantic models that do **not** include `initCache` or `softCut`,
* with `extra="ignore"`, those fields are silently dropped,
* and the render loop never applies `init_cache=True` nor temporary bias overrides.

So the offline renderer will always behave like “continuous cache, continuous bias,” which means:

* hard cuts won’t happen,
* soft cut responsiveness changes won’t happen,
* and replay will “morph” more than realtime did.

That’s the central missing piece for meaningful session recorder output.

---

## What’s involved to support the proposal end-to-end

Below is the concrete work breakdown, in the order that tends to reduce thrash.

### 1) Implement the recorder module

Add `src/scope/server/session_recorder.py` as proposed.

Key implementation choices to make explicit:

* Store events in chunk index + relative wall time (you already do).
* Decide whether to record:

  * only first prompt, or
  * full `prompts` list (recommended if blending matters).
* Decide whether to record LoRA scale events now or later.

### 2) Integrate recorder into `FrameProcessor.process_chunk()`

You’ll need to add **new reserved keys** (similar to snapshot/restore/step):

* `_rcp_session_recording_start`
* `_rcp_session_recording_stop`

and handle them in the worker thread *before* pause early-return.

Also, don’t duplicate your existing soft-cut logic—**hook into it** to capture:

* effective `temp_bias`, `num_chunks` (after clamp/coerce),
* and restore target from the internal state (`_soft_transition_original_bias`, `_soft_transition_original_bias_was_set`) rather than approximating from `self.parameters`.

For hard cuts, you can capture an edge when you see `reset_cache` arrive, but for best fidelity, consider recording the hard cut **when you actually pass `init_cache=True` to the pipeline** (because that’s the moment it truly takes effect).

### 3) Add `PipelineManager.peek_status_info()` (non-mutating)

You need a status read that:

* uses the same lock as other accesses,
* does **not** clear error state,
* returns the pipeline id + load params needed for export metadata.

And you should align the returned `"status"` string with what callers expect (either `.value` strings like `"loaded"` or `.name` strings like `"LOADED"`).

### 4) Add REST endpoints in `server/app.py`

Add:

* `POST /api/v1/realtime/session-recording/start`
* `POST /api/v1/realtime/session-recording/stop`
* `GET /api/v1/realtime/session-recording/status`

These should use `apply_control_message()` and return 503 if it fails to enqueue (like your other control endpoints).

For status, don’t read mutable recorder state directly from FastAPI threads; use the “atomic snapshot dict” approach you described.

### 5) Add CLI affordance (optional but useful)

In `video_cli.py` playlist nav loop, add a key binding (e.g. `R`) to start/stop and poll status.

(Your proposal’s pattern works; just make sure you don’t assume stop completes immediately—because it completes on the worker thread at the next mailbox drain.)

---

## The offline renderer changes required to replay initCache and softCut

This is the “recordings won’t replay until this exists” part.

### A) Update the schema in `cli/render_timeline.py`

Add a model for soft cut and fields on `TimelineSegment`:

```py
class TimelineSoftCut(BaseModel):
    model_config = ConfigDict(extra="ignore")
    bias: float
    chunks: int = 2
    restoreBias: float | None = None
    restoreWasSet: bool = False

class TimelineSegment(BaseModel):
    model_config = ConfigDict(extra="ignore")
    ...
    initCache: bool | None = None
    softCut: TimelineSoftCut | None = None
```

Also update the `--dry-run` path (which currently inspects raw dicts) if you want debugging output to reflect these fields.

### B) Implement one-shot `init_cache` replay

In the render loop:

* When you enter a segment with `initCache: true`, set a local `pending_init_cache = True`.
* On the *next* pipeline call, pass `init_cache=True` once.
* Then clear it so it’s not sticky.

This mirrors realtime `reset_cache → init_cache` semantics.

### C) Implement a soft-cut state machine in the renderer

You need renderer-local state similar to the server’s:

* `soft_active`
* `soft_chunks_remaining`
* `soft_temp_bias`
* `soft_restore_bias`
* `soft_restore_was_set`

On segment boundary:

* if `segment.softCut` exists:

  * clamp bias to `[0.01, 1.0]`,
  * clamp chunks to `[1, 10]`,
  * (re)start countdown,
  * set `parameters["kv_cache_attention_bias"] = temp_bias`.

After each pipeline call:

* decrement chunks,
* when it hits 0:

  * if `restoreWasSet` and `restoreBias` is not None → restore to that value,
  * else pop the key to represent “unset”.

This is the minimum necessary to replay the server’s semantics with reasonable fidelity.

### D) Decide precedence rules explicitly

Your realtime playlist helper effectively makes some combinations mutually exclusive (e.g., transition not used when hard_cut is true).

Offline renderer should define what it does if a segment includes combinations like:

* `initCache: true` + `transitionSteps > 0`
* `softCut` + a transition

For fidelity, the safest rule is: **do what the recorded segment says**, even if it’s a weird combo—because if it exists in a recording, it happened.

But you may still want guardrails (log warnings) for combinations you consider nonsensical.

### E) (Strongly recommended) Add chunk-based segment scheduling mode

Because your recorder exports `startChunk/endChunk`, the best way to avoid wall-time drift is to let the renderer optionally schedule by chunk index (pipeline call count) rather than by `produced_frames / fps`.

This can be incremental:

* If segments include `startChunk`, select active segment using `call_index` (pipeline iteration count) instead of `current_time`.
* Keep the current time-based mode as default for older timeline files.

---

## Summary: what’s needed to make “record → replay” actually work

Right now, even a perfect server-side recorder will produce timeline JSON that the offline renderer partially ignores. To close the loop:

1. **Server-side**: implement recorder + integrate into `FrameProcessor` + add `peek_status_info` + add REST endpoints.
2. **Offline**: extend `render_timeline.py` schema and render loop to actually replay:

   * `initCache` as one-shot `init_cache=True`
   * `softCut` as a temporary bias override + restore state machine
3. **Fidelity next step (optional but high leverage)**: support chunk-based scheduling using exported `startChunk/endChunk`.

If you want, I can also point out the *lowest-risk implementation strategy* for the `FrameProcessor` integration (where to hook recording without perturbing current semantics), but the core blockers for replay are the renderer changes above.
