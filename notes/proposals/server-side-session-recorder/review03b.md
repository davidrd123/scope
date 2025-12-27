Got it — I’m updating the prep plan and buildable spec according to your latest instructions.


Below is a “prep pass” that turns the current proposal into an **executable contract** with a clear **Definition of Done**, a **frozen file contract**, and **gating replay requirements**. I’m going to treat the default path as:

* **Scope = MVP “prompt timeline recorder”** (prompts/transition + hard/soft cuts, **text-mode only**), because that’s what’s already ~8–9/10 and gets you to “buildable + verifiable” with minimal churn.
* I’ll still include an explicit “delta” section for **full control-plane session recorder** so the upgrade path is unambiguous.

---

# 1) Freeze scope + Definition of Done

## Decision

✅ **Pick:** **MVP Prompt Timeline Recorder**
**Records:** prompt changes + transitions + hard cuts + soft cuts
**Mode:** **text-only** (T2V).
**Goal:** produce a timeline JSON that can be replayed by `render_timeline.py` **without silent divergence**.

## Explicit Out of Scope (MVP)

These can occur during recording, but **are not guaranteed to replay faithfully** unless/until you do the “full session recorder” delta:

* **LoRA/style switching fidelity** (`EventType.SET_LORA_SCALES`, `_rcp_set_style`)
* **Seed changes** (`EventType.SET_SEED`)
* **Denoising step changes** (`EventType.SET_DENOISE_STEPS`)
* **World state updates** (`_rcp_world_state`) *as a semantic event* (prompts induced by it *will* be recorded if they become `SET_PROMPT`)
* **Video/V2V** (`input_mode == "video"`, VACE, etc.)

This is important because today `FrameProcessor` does style changes by emitting both:

* `merged_updates["prompts"] = ...` (→ `SET_PROMPT` event) **and**
* `merged_updates["lora_scales"] = ...` (→ `SET_LORA_SCALES` event)

If MVP ignores `SET_LORA_SCALES`, a style swap session will “look right” textually but replay wrong visually.

---

## Definition of Done (MVP)

All items must be true to call it “ready to build + yields faithful replays”.

### Server-side recording

* [ ] `src/scope/server/session_recorder.py` implemented as a pure recorder class (no FastAPI/thread coupling).
* [ ] `FrameProcessor.process_chunk()` integrates recorder:

  * [ ] Start/stop handled via **reserved keys** on the **worker thread**:

    * `_rcp_session_recording_start`
    * `_rcp_session_recording_stop`
  * [ ] Prompt/transition events recorded from the **drained ControlBus events** list (`events = self.control_bus.drain_pending(...)`).
  * [ ] Soft cut recorded using the **actual clamped** `temp_bias` and `num_chunks`, and restore target derived from existing FrameProcessor state:

    * `_soft_transition_original_bias`
    * `_soft_transition_original_bias_was_set`
  * [ ] Hard cut recorded when it is **executed**, i.e. when the pipeline call is made with `init_cache=True` **due to** `reset_cache=True` (not just “init_cache was true because first call”).
* [ ] `PipelineManager.peek_status_info()` exists and is **non-mutating** (does not clear error state; uses `.value` strings like `"loaded"`; returns `_error_message`).

### API surface

* [ ] REST endpoints exist:

  * `POST /api/v1/realtime/session-recording/start`
  * `POST /api/v1/realtime/session-recording/stop`
  * `GET  /api/v1/realtime/session-recording/status`
* [ ] `stop` is async (returns immediately); `status` eventually returns `last_timeline_path`.

### Offline replay gating (must ship with MVP)

* [ ] `src/scope/cli/render_timeline.py` **parses and replays**:

  * `initCache` → one-shot `init_cache=True`
  * `softCut` → temporary bias state machine with restore semantics
* [ ] Drift prevention:

  * [ ] Add **chunk-based segment scheduling** (or at minimum a flag) so you can choose chunk timebase as canonical and avoid time-based drift.

---

# 2) Acceptance checks (3–5)

These are written so you can implement them either as automated tests (preferred) or as a short manual validation script.

## AC1 — “No prompt changes” still yields a valid timeline

**Setup**

1. Start stream and ensure pipeline is loaded.
2. Set prompt once (`PUT /api/v1/realtime/prompt`).
3. Start recording.
4. Generate N chunks (or wait for N chunks).
5. Stop recording and poll status until path returned.

**Expected**

* Timeline JSON exists.
* `prompts` array contains **at least 1 segment**, starting at chunk/time 0 (baseline).
* That segment has a prompt matching the current effective prompt at start (from `parameters["prompts"]` or transition target).

## AC2 — Hard cut replay matches realtime semantics (one-shot)

**Setup**

1. Start recording.
2. Apply a prompt.
3. Apply a hard cut (`POST /api/v1/realtime/hard-cut`, optionally with a new prompt).
4. Stop recording → get timeline.
5. Replay with updated `render_timeline.py`.

**Expected**

* Exported timeline contains a segment with `"initCache": true` at the boundary corresponding to the cut.
* In replay, `render_timeline.py` passes `init_cache=True` for **exactly one** pipeline call at that boundary (then clears it).

## AC3 — Soft cut replay matches realtime semantics (temporary override + restore)

**Setup**

1. Start recording.
2. Apply prompt A, generate a few chunks.
3. Trigger soft cut for `num_chunks=K`, `temp_bias=B` (optionally with prompt B).
4. Generate >K chunks.
5. Stop recording → replay.

**Expected**

* Exported segment contains:

  * `softCut.bias == clamped(B)`
  * `softCut.chunks == clamped(K)`
  * `softCut.restoreBias` / `restoreWasSet` consistent with FrameProcessor’s internal restore target.
* Replay applies `kv_cache_attention_bias=B` for exactly **K pipeline calls**, then restores:

  * if `restoreWasSet=true`: restore to `restoreBias`
  * else: pop the key (“unset”)

## AC4 — Transition replay semantics match realtime (including transition completion)

**Setup**

1. Start recording.
2. Apply prompt with transition (via whatever path produces `transition` payload; playlist nav with transition on is fine).
3. Generate enough chunks for transition to complete.
4. Stop recording → replay.

**Expected**

* Timeline segment includes:

  * `transitionSteps` and `temporalInterpolationMethod`
* Replay sets `parameters["transition"]` on entry and then:

  * keeps sending transition until pipeline signals it completed (`pipeline.state["_transition_active"] == False`)
  * then sets `parameters["prompts"] = target_prompts` and removes transition (mirrors server logic)

## AC5 — Stop is async and yields a saved path

**Setup**

1. Start recording.
2. Stop recording via endpoint.
3. Poll status.

**Expected**

* `stop` returns immediately.
* Within bounded polls, `status.is_recording == false` and `status.last_timeline_path` exists and is readable JSON.

---

# 3) Freeze the exported timeline file contract

This is the “lock it down” section to insert near the top of `notes/proposals/server-side-session-recorder.md` as the normative contract.

## Versioning

**Set:**

* `"version": "1.1"` for the MVP that includes `initCache`, `softCut`, and chunk scheduling fields.

## Primary timebase

**Canonical timebase = chunk-based** for fidelity.

* `startChunk/endChunk` are authoritative for scheduling and cut durations.
* `startTime/endTime` are secondary (human readability, legacy, approximate).

## Top-level schema

```jsonc
{
  "version": "1.1",
  "exportedAt": "RFC3339 UTC string",
  "recording": {
    "durationSeconds": 12.34,
    "durationChunks": 56,
    "startChunk": 1234,          // absolute chunk index at start
    "endChunk": 1290             // absolute chunk index at stop
  },
  "settings": {
    "pipelineId": "krea-realtime-video",
    "inputMode": "text",         // MUST be "text" for MVP
    "resolution": { "height": 480, "width": 832 },

    "seed": 42,                  // base seed (best-effort; see notes)
    "denoisingSteps": [1000,750,500,250],
    "manageCache": true,
    "kvCacheAttentionBias": 0.3,

    // Optional (load-time only for MVP; runtime LoRA events out-of-scope)
    "quantization": "fp8",
    "loras": [ { "path": "...", "scale": 1.0, "mergeMode": "permanent_merge" } ],
    "loraMergeStrategy": "permanent_merge"
  },
  "prompts": [
    {
      "startTime": 0.0,
      "endTime": 1.23,
      "startChunk": 0,
      "endChunk": 5,

      "prompts": [ { "text": "…", "weight": 1.0 } ],

      "transitionSteps": 4,
      "temporalInterpolationMethod": "linear",

      "initCache": true,

      "softCut": {
        "bias": 0.1,
        "chunks": 2,
        "restoreBias": 0.3,
        "restoreWasSet": true
      }
    }
  ]
}
```

## Segment semantics

Each entry in `prompts[]` is a **segment boundary** (something changed in control semantics), not necessarily a “prompt text changed”.

### Required fields per segment

* `startChunk`, `endChunk` (int, relative to recording start; non-decreasing; `startChunk < endChunk`)
* `startTime`, `endTime` (float seconds relative to start; non-decreasing)
* Either:

  * `prompts: [{text, weight}, …]` (preferred), or
  * `text: "..."` (legacy; if present, renderer will synthesize a prompt list)

### Optional fields

* `transitionSteps` (int > 0)
* `temporalInterpolationMethod` (`"linear"` or `"slerp"`)
* `initCache` (bool)
* `softCut` (object as below)

## `initCache` semantics

If `segment.initCache == true`, offline replay MUST:

* pass `init_cache=True` **for the first pipeline call** after entering this segment,
* then clear it (one-shot).

## `softCut` semantics

If `segment.softCut` exists:

* Clamp:

  * `bias ∈ [0.01, 1.0]`
  * `chunks ∈ [1, 10]`
* On entry:

  * set `kv_cache_attention_bias=bias`
  * start/restart a countdown for `chunks` pipeline calls
  * restore target:

    * if not already in soft transition, use `restoreBias/restoreWasSet`
    * if already in soft transition, do **not** clobber restore target
* After each pipeline call:

  * decrement countdown
  * when countdown hits 0:

    * if `restoreWasSet=true` and `restoreBias != null` → restore to that value
    * else → pop key to represent “unset”

## Precedence rules (hard cut vs transition vs soft cut)

The file is authoritative. If combinations exist, replay them as recorded:

* `initCache` does **not** disable transitions.
* `softCut` does **not** disable transitions.
* If both exist, apply both.

(You can log warnings for “weird combos”, but don’t rewrite semantics.)

---

# 4) Identify the exact event surface to record

## MVP event surface

Record only the following:

1. **Prompt/transition changes**
   Source: `EventType.SET_PROMPT` in `events = self.control_bus.drain_pending(...)`
   Payload shapes observed in code:

   * `{"prompts": [...], "transition": {...}}`
   * `{"prompts": [...]}`
   * `{"transition": {...}}` (transition-only)

2. **Hard cut**
   Source: `reset_cache=True` (popped from `self.parameters`) → results in pipeline call with `init_cache=True` and output queue flush.
   **Record when executed**, i.e. in the same `process_chunk` iteration where:

   * `reset_cache` was present **and truthy**
   * and the pipeline call was actually made

3. **Soft cut**
   Source: `_rcp_soft_transition` reserved key (popped from `merged_updates`)
   **Record using FrameProcessor’s post-clamp state**, including restore semantics:

   * `temp_bias` / `num_chunks` after clamp/coerce
   * `restoreBias = self._soft_transition_original_bias`
   * `restoreWasSet = self._soft_transition_original_bias_was_set`

### Critical mismatch to fix in the proposal text

The proposal’s baseline prompt extraction must NOT assume `control_state.CompiledPrompt.positive`.

In your actual runtime:

* `FrameProcessor._compiled_prompt` is produced by `prompt_compiler.TemplateCompiler.compile(...)`,
* and `app.py` reads `fp._compiled_prompt.prompt` (which exists on `prompt_compiler.CompiledPrompt`).

So the safe baseline extraction order should be:

1. `self.parameters["transition"]["target_prompts"][0]` (if present)
2. `self.parameters["prompts"][0]` (if present)
3. `self._compiled_prompt.prompt` (if present)
4. else None

(And treat weights in the same `[{text, weight}]` shape everywhere.)

---

## Full session recorder delta (future)

If you choose “full control-plane recorder”, add these event types:

* `EventType.SET_LORA_SCALES` → record payload and export as segment-level `loraScales` or as an `events[]` stream.
* `EventType.SET_SEED` → export seed changes.
* `EventType.SET_DENOISE_STEPS` → export denoising step list changes.
* Optionally: `_rcp_set_style`, `_rcp_world_state` as semantic events (even if they also produce prompt changes).

And **offline replay must apply** these changes at the same boundaries (including one-shot `lora_scales` injection like server does).

---

# 5) Make replay support real (gating): required `render_timeline.py` updates

These are “must do” for MVP to avoid silent divergence.

## A) Schema updates (Pydantic models)

Add to `TimelineSegment`:

* `initCache: bool | None = None`
* `softCut: TimelineSoftCut | None = None`
* `startChunk: int | None = None`
* `endChunk: int | None = None`

Add:

```py
class TimelineSoftCut(BaseModel):
    model_config = ConfigDict(extra="ignore")
    bias: float
    chunks: int = 2
    restoreBias: float | None = None
    restoreWasSet: bool = False
```

## B) Render-loop changes: initCache (one-shot)

Maintain `pending_init_cache = False`.

On segment change:

* if `active_segment.initCache`: `pending_init_cache = True`

Before pipeline call:

* if `pending_init_cache`: pass `init_cache=True` for that call only
  After pipeline call:
* clear `pending_init_cache`

## C) Render-loop changes: softCut state machine

Maintain:

* `soft_active`
* `soft_chunks_remaining`
* `soft_restore_bias`
* `soft_restore_was_set`

On segment change when `active_segment.softCut` present:

* clamp bias/chunks
* if not `soft_active`: capture restore target from segment’s `restoreBias/restoreWasSet`
* set `kv_cache_attention_bias=temp_bias`
* set `soft_chunks_remaining=chunks`, `soft_active=True`

After each pipeline call:

* decrement
* when 0: restore or pop per contract

## D) Chunk-based segment scheduling

Add a mode switch (flag or auto):

* **Time mode** (current behavior): segment chosen by `produced_frames/fps >= segment.startTime`
* **Chunk mode**: segment chosen by `call_index >= segment.startChunk`

In chunk mode:

* Maintain `call_index += 1` each pipeline call.
* Stop condition should be:

  * `call_index >= max_end_chunk` (from segments), or
  * `call_index >= recording.durationChunks` if you start parsing `recording`.

This is the key to avoid time drift and make “chunk is primary” actually true.

---

# 6) Wire safely + measurably: FrameProcessor + PipelineManager

## New reserved keys

* `_rcp_session_recording_start`
* `_rcp_session_recording_stop`

These must be handled inside `process_chunk()` **before**:

* translation pops keys into ControlBus,
* and before the pause early-return.

## Non-mutating `peek_status_info()`

Must:

* take `_lock`
* return:

  * `"status": self._status.value` (e.g., `"loaded"`)
  * `"pipeline_id": self._pipeline_id`
  * `"load_params": copy`
  * `"error": self._error_message`
* **must not clear** error state / reset status (unlike `get_status_info()`)

## Recording source-of-truth

* Prompt/transition: from `events = self.control_bus.drain_pending(...)`
* Soft cut: from `_rcp_soft_transition` reserved key handling (capture clamped values + restore target)
* Hard cut: from `reset_cache` execution path (when pipeline call uses it)

---

# 7) The one thing I’d change in the proposal before building

**Fix the baseline prompt + compiled prompt type confusion** explicitly in the proposal text.

Right now the proposal’s rev notes mention `control_state.CompiledPrompt.positive`, but the actual server is using `prompt_compiler.CompiledPrompt` (with `.prompts`, `.prompt`, `.to_pipeline_kwargs()`), and `app.py` already assumes `.prompt`.

So the contract should say:

> Baseline prompt comes from the **pipeline-facing** `parameters["prompts"]` / transition targets first. Compiler output is a last-resort fallback and must be normalized into the same `[{text, weight}]` shape.

That one change prevents a bunch of “it worked in review but breaks in real code” churn.

---

If you want, I can format the above as a ready-to-paste block into `notes/proposals/server-side-session-recorder.md` (with headings matching your doc style), but everything needed to lock scope + DoD + file contract + gating replay changes is already here.
