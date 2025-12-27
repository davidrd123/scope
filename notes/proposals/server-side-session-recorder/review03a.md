Got it — I’m updating the plan according to your instructions.


You’re *very* close—your 5-step “prep pass” is basically the remaining gap between **a good idea** and **a buildable contract**.

At this point the *hard parts* (semantics alignment: edge-triggered hard cut, soft-cut restore nuance, ControlBus ordering vs merged updates, Pydantic dropping fields) are understood. What’s left is mostly: (a) freezing the contract so implementation doesn’t churn, and (b) adding the renderer support so you can validate end-to-end immediately.

Below is what I’d consider the “executable contract + DoD” you can paste into the proposal (or a new `MVP SPEC` section), plus the few extra “gotchas” worth locking down.

---

## Where you are, in concrete terms

### MVP prompt timeline recorder

If you implement:

* recorder + FrameProcessor integration (with correct event field names + baseline prompt normalization),
* `PipelineManager.peek_status_info()`,
* `render_timeline.py` replay for `initCache` + `softCut` (+ chunk scheduling),

…then you’re at “ready to build and validate” for MVP. The semantics aren’t fuzzy anymore; they’re just not encoded yet.

### Full session recorder

You’re not *blocked*—you just need to explicitly expand the recorded event surface + offline replay semantics (LoRA scales being the biggest “visual mismatch” culprit). That’s why it feels ~6/10: it’s not unclear, it’s just **not specified**.

---

## The prep pass as an executable contract

### 1) Freeze scope + DoD

Put a single, explicit scope statement at the top:

**MVP scope:**

* **Input mode:** text-only (`inputMode = "text"`).
* **Record:** prompt sets (including transitions), hard cuts, soft cuts.
* **Do not record (MVP):** LoRA scale changes, seed/denoise step changes, style/world updates (unless they happen to manifest as a prompt event).

Then **3–5 acceptance checks** that are testable, not vibes:

**Acceptance checks (MVP)**

1. **Hard cut replay fidelity:**
   If a recorded segment has `initCache: true`, offline replay calls the pipeline with `init_cache=True` **for exactly the first pipeline call at that boundary**, and **never “sticks”** beyond that call.

2. **Soft cut replay fidelity:**
   If a recorded segment has `softCut: {bias, chunks, restoreBias, restoreWasSet}`, offline replay:

   * immediately sets `kv_cache_attention_bias=bias`,
   * keeps it for exactly `chunks` pipeline calls,
   * then restores to:

     * `restoreBias` if `restoreWasSet=true`, else
     * “unset” (delete the kwarg) if `restoreWasSet=false`.

3. **Transition replay fidelity:**
   If a segment includes a transition, offline replay sends a `transition` dict until the pipeline signals completion (via `_transition_active`), then clears `transition` and sets `prompts=target_prompts`.

4. **Stop is async, but path is observable:**
   `POST /session-recording/stop` returns immediately (e.g. `{status:"stop_requested"}`), and `GET /session-recording/status` eventually includes `last_timeline_path` once saved.

5. **No prompt changes still produces a valid timeline:**
   Starting and stopping recording without any prompt changes produces a timeline file that still parses and replays (because a baseline segment was captured *or* the schema allows an empty prompt list but valid metadata).

Those are enough to prevent semantic churn.

---

### 2) Freeze the file contract

This is the part that prevents you from “shipping JSON that looks right but replays wrong.”

#### Version + timebase

Add these explicitly:

* `version`: e.g. `"scope-timeline-1"` (or `"1.0"` if you prefer).
* `primaryTimebase`: `"chunk"` for fidelity.
* Keep `startTime/endTime` as optional/human-facing secondary.

#### Segment contract

For MVP, lock the segment fields you actually plan to replay:

```json
{
  "startChunk": 0,
  "endChunk": 34,

  "startTime": 0.0,
  "endTime": 8.5,

  "prompts": [{"text": "…", "weight": 1.0}],

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
```

#### Precedence rules (lock these)

This is worth writing down explicitly so you don’t debug “why does replay morph?” later.

I’d freeze these rules for MVP:

* **Hard cut overrides transition initiation.**
  If `initCache=true`, treat the boundary as a cut: apply the new prompt directly and do **not** initiate a transition for that boundary.
  (This matches your playlist helper’s behavior: transition is only built when `not hard_cut`.)

* **Soft cut is orthogonal.**
  Soft cut can apply alongside either a cut or a transition (and does in your current API).

* **Prompt weights:** replay uses exact recorded weights. If absent, default to `1.0` (match REST schema).

#### Canonical scheduling mode

If you care about fidelity, explicitly state:

* **Canonical replay scheduling uses `startChunk/endChunk`.**
* Time-based scheduling is fallback for older timelines or manual edits.

---

### 3) Make replay support real

This is your gating step and you’re exactly right to treat it as such.

Minimum changes in `src/scope/cli/render_timeline.py`:

* Extend `TimelineSegment` schema to include:

  * `initCache: bool | None`
  * `softCut: TimelineSoftCut | None`

* Implement:

  * one-shot `init_cache=True` (pending flag) on boundary,
  * soft cut state machine (chunks remaining + restore semantics),
  * **chunk-based segment selection mode** (`--timebase auto|chunk|time`).

If you do nothing else, do this—because otherwise recordings will “appear to work” but diverge silently.

---

### 4) Identify the exact event surface to record

For MVP, lock the list exactly as you wrote:

* **Record:**

  * `SET_PROMPT` (including transition payload)
  * hard cut edge (`reset_cache` → exported as `initCache`)
  * soft cut reserved key (`_rcp_soft_transition` → exported as `softCut`)

And lock what you do *not* record in MVP.

For full session recorder, list the next set explicitly:

* `SET_LORA_SCALES` (high priority)
* `SET_SEED`
* `SET_DENOISE_STEPS`
* optionally: style/world updates (either as explicit events or as “derived prompt + lora changes” events)

This is the line between “timeline recorder” and “session recorder.”

---

### 5) Wire safely + measurably

Also exactly right.

A couple details worth locking in so implementation doesn’t drift:

* **Recording happens on the worker thread** (inside `process_chunk()`), triggered via reserved keys:

  * `_rcp_session_recording_start`
  * `_rcp_session_recording_stop`

* **Record from applied events**, not raw merged dicts:

  * `events = self.control_bus.drain_pending(..., chunk_index=self.chunk_index)`
  * record based on `event.type == EventType.SET_PROMPT` (note: it’s `.type`, not `.event_type`)

* **Hard cut and soft cut must be captured before they’re consumed**, because:

  * `reset_cache` is popped later (`self.parameters.pop("reset_cache")`)
  * `_rcp_soft_transition` is popped and applied immediately

* **PipelineManager status access must not mutate error state**:

  * implement `peek_status_info()` under the lock
  * use it for recording metadata capture

* **Measurability hook (strongly recommended):**

  * Add an optional debug mode where recorder also logs (or stores) the exact per-call kwargs you send to the pipeline for the chunk(s) around an event.
    That lets you validate “record → replay” by comparing call sequences.

---

## One extra “gotcha” I’d add to your prep pass

### Baseline prompt capture should be pipeline-facing, not compiler-facing

Even after you fix the `CompiledPrompt` mismatch, the least ambiguous baseline is:

* If `parameters["transition"]` exists, baseline prompt = `transition.target_prompts`
* Else baseline prompt = `parameters["prompts"]`
* Only if both missing, fallback to style/compiler outputs (and normalize into `[{text, weight}]`)

That prevents you from binding the recorder contract to the internals of the style layer compiler types.

---

## What “ready to approach it” looks like after the prep pass

Once those 5 steps are done, you’ll have:

* a frozen JSON schema that **offline can actually replay** (no more silent drops),
* a frozen precedence model (hard cut vs transition, soft cut orthogonality),
* a small, testable event surface for MVP,
* and an end-to-end validation path: record a short session → replay → compare per-call control kwargs.

At that point, implementation becomes “wiring + tests,” not “debating semantics.”

If you want, I can draft the exact `MVP SPEC` section text for `server-side-session-recorder.md` (Scope + DoD + file contract + precedence + timebase) in a form you can paste verbatim—so the proposal itself becomes the executable contract.
