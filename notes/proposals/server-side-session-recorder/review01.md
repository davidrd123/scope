Below is a feasibility / risk review of `notes/proposals/server-side-session-recorder.md`, followed by the concrete (non-implementation) code changes needed to make `src/scope/cli/render_timeline.py` **actually replay** `initCache` (hard cuts) and `softCut` (temporary KV bias) in a way that matches **current realtime** semantics.

---

## 1) Server-side session recorder: feasibility and risks

### Overall feasibility

The core architecture is sound:

* **Correct tap-point:** capturing inside `FrameProcessor.process_chunk()` is the right place if you want to see *all* control sources (frontend, CLI, direct API). It’s “after merge, before/after application,” which is exactly where realtime truth lives.
* **Thread-safety direction is right:** routing start/stop via reserved keys and handling them on the worker thread avoids FastAPI-thread mutation hazards.
* **Timebase choice is good:** using **chunk index as primary** and wall-time as secondary is aligned with how the generator actually advances (`chunk_index += 1` at the end of `process_chunk()`).

So: **yes, feasible**.

### Key risks / alignment issues to address

#### Risk A — “baseline prompt” extraction mismatch

Your proposal’s `_get_current_effective_prompt()` suggests:

* transition target prompt (good)
* current prompts (good)
* “compiled prompt from style layer” via `self._compiled_prompt` (⚠️ type mismatch)

In the real `FrameProcessor.__init__` snippet you included, `_compiled_prompt` is a `CompiledPrompt | None`, and `CompiledPrompt` (from `realtime/control_state.py`) has fields like `positive`, `negative`, `lora_scales`. It is *not* a string prompt.

**Implication:** baseline recording could serialize an object or `None`, or just be wrong.

**What to change (conceptually):**

* baseline should use `self._compiled_prompt.positive` (or whatever is the “effective positive prompt string” that ultimately gets turned into `prompts=[{text, weight}]`).

#### Risk B — prompt weight scale inconsistency (1.0 vs 100.0)

You have mixed defaults:

* Pipeline warmup uses `weight: 1.0` (`WARMUP_PROMPT` in `krea_realtime_video/pipeline.py`)
* Timeline schema in `render_timeline.py` defaults weight to **100.0**
* Recorder dataclasses default weight to **100.0**, but extraction defaults to **1.0**

If the embedding blender normalizes weights internally, this won’t matter. If it doesn’t, offline replay will drift from realtime.

**Recommendation:** record and replay **exactly what realtime passes** (`payload["prompts"][0]["weight"]`), and avoid inventing new defaults in the recorder. If you must default, default to *the same value used when prompts are constructed elsewhere in the stack*.

#### Risk C — event completeness depends on ControlBus semantics

The proposal says “record from applied ControlBus events, not merged_updates.”

That’s correct in principle (because keys like `prompts`/`transition` may be popped/translated), but it assumes:

* **prompt changes always produce** `EventType.SET_PROMPT`
* and that event payload always contains enough info to reconstruct prompt+transition metadata

If there are other pathways (style changes causing recompilation, LoRA edge triggers, etc.) that don’t map to `SET_PROMPT`, you’ll miss them.

**Mitigation idea (design-level):**

* record from **both**: ControlBus events (authoritative) plus a “fallback” from parameter edge detection (for rare paths).
* or define a single “applied control snapshot” event that includes the resolved call kwargs.

#### Risk D — hard cut capture must be “edge-triggered”

Realtime hard cut behavior is edge-triggered:

* `reset_cache` is popped (`self.parameters.pop("reset_cache", None)`)
* output queue flushed
* `init_cache` passed for that call only

So your recorder must treat a hard cut as an **event**, not a state.

Your proposed approach (“capture `hard_cut_requested = "reset_cache" in merged_updates`”) is directionally right, but only if you do it **before** the key is popped/consumed.

#### Risk E — soft cut restore semantics are more subtle than the timeline format

Realtime soft cut implementation has nuanced restore rules:

* It distinguishes whether `kv_cache_attention_bias` was **explicitly set** vs “unset → use default”
* It supports re-triggering without clobbering the original restore target
* It cancels if an explicit bias update arrives mid-transition

Your proposed timeline format records only:

```json
"softCut": {"bias": X, "chunks": N}
```

That is enough to “do something,” but it cannot perfectly reproduce:

* “restore to <unset>” vs “restore to original explicit”
* cancellation / override semantics

You already called this out in the proposal; it’s the biggest fidelity gap for offline replay.

#### Risk F — pipeline reload mid-recording

You already note it: recording continues with stale `pipeline_id` / stale `load_params`.

This is fine for MVP, but it’s the kind of thing that produces “mysterious offline mismatch” later.

---

## 2) What must change in `render_timeline.py` to replay `initCache` and `softCut`

Right now, `render_timeline.py` **ignores** these fields because:

* `TimelineSegment` doesn’t define them
* Pydantic is configured `extra="ignore"`, so the JSON fields are dropped

### Goal: match realtime behavior

From your realtime snippets:

#### Realtime hard cut

* Control-plane: `reset_cache=True` in control message
* FrameProcessor:

  * flushes output queue (display correctness)
  * passes `init_cache=True` to pipeline for that call

#### Realtime soft cut

* Control-plane: `_rcp_soft_transition={temp_bias, num_chunks}`
* FrameProcessor:

  * immediately sets `kv_cache_attention_bias=temp_bias`
  * keeps it for `num_chunks` chunk iterations
  * then restores either:

    * the previous explicit bias value, **or**
    * deletes the key (restore to `<unset>` / default)

Offline renderer needs to mimic those semantics at the level it can control:

* it can pass `init_cache` as a kwarg for a given pipeline call
* it can temporarily override `kv_cache_attention_bias` across N pipeline calls

---

## 3) Required schema changes in `render_timeline.py`

### A) Extend `TimelineSegment`

Add two new optional fields to the Pydantic model:

* `initCache: bool | None = None`
* `softCut: TimelineSoftCut | None = None` (or `dict | None` if you want to keep it lightweight)

Recommended (explicit model for validation):

```py
class TimelineSoftCut(BaseModel):
    model_config = ConfigDict(extra="ignore")
    bias: float
    chunks: int = 2

class TimelineSegment(BaseModel):
    model_config = ConfigDict(extra="ignore")
    ...
    initCache: bool | None = None
    softCut: TimelineSoftCut | None = None
```

**Why this is important:**

* You get validation (chunks is int, bias is float)
* You can clamp values consistently with realtime rules (more on that below)

### B) Dry-run output should include these fields

In `--dry-run` mode, the “plan” is built from raw dicts. If you want dry-run to reflect new features, include:

* whether a segment has `initCache`
* whether a segment has `softCut` and what it resolves to

This isn’t required for functionality, but it’s extremely helpful for debugging replay fidelity.

---

## 4) Required render-loop changes for `initCache` replay

### Current behavior (offline)

* Renderer changes prompts/transition at segment boundaries
* It does **not** pass `init_cache` explicitly

Pipeline still does a first-call init implicitly via `handle_mode_transition()`, but there is no way to replay mid-session hard cuts.

### Required behavior

When entering a segment that specifies `initCache: true`, the renderer must:

* pass `init_cache=True` to the pipeline **for the first pipeline call after the boundary**
* and then return to normal behavior (do not keep `init_cache=True` sticky)

### Where to hook it

In the main loop, there is already a “segment boundary” check:

```py
if current_segment_id != last_segment_id:
    ... update prompts/transition ...
    last_segment_id = current_segment_id
```

That is the correct place to stage a **one-shot** `init_cache` for the next call.

### Minimal logic you need (conceptually)

* Maintain a local flag like `pending_init_cache = False`
* On segment change:

  * if `active_segment.initCache` is true → `pending_init_cache = True`
* Right before `output = pipeline(**parameters)`:

  * if `pending_init_cache`: pass `init_cache=True` for that call only
  * then clear `pending_init_cache = False`

**Alignment note:** this matches realtime where `reset_cache` affects *the next* call.

### What about `--no-transitions`?

Realtime semantics: hard cut is independent of transition smoothing.

So offline should treat these as orthogonal:

* if `--no-transitions`, you still apply `init_cache` if requested
* if transitions are enabled, you still apply `init_cache` if requested

Do **not** silently convert `initCache` into “disable transitions.”

---

## 5) Required render-loop changes for `softCut` replay

### The key mapping

Realtime “soft cut” is:

* temporary override of `kv_cache_attention_bias`
* for N *chunks* (pipeline calls)
* restore afterwards (with nuanced “unset vs set” semantics)

Offline renderer has the same basic structure:

* each iteration of the loop is one pipeline call producing ~`num_frame_per_block` frames (currently assumed 3)

So: **softCut.chunks maps naturally to “number of pipeline calls.”**

### A) Add soft-transition state to the renderer loop

You’ll need state variables very similar to realtime `FrameProcessor`:

* `soft_active: bool`
* `soft_chunks_remaining: int`
* `soft_temp_bias: float | None`
* `soft_original_bias: float | None`
* `soft_original_bias_was_set: bool`

Even if you simplify restore semantics, you still need at least:

* whether you’re in a soft cut
* how many calls remain
* what to restore

### B) Start a soft cut when you enter a segment that has `softCut`

At the same segment-change hook where you update prompts/transition:

If `active_segment.softCut` exists:

* parse `bias` and `chunks`
* apply the **same clamping rules** as realtime to avoid downstream numeric issues:

  * `bias = clamp(float(bias), 0.01, 1.0)`
  * `chunks = clamp(int(chunks), 1, 10)` (realtime uses max 10)
* if not already in soft transition:

  * save restore target (“original bias”)
* (re)start countdown
* set `parameters["kv_cache_attention_bias"] = bias` immediately

### C) Re-entrancy behavior should match realtime

Realtime rule:

* if already in soft transition, don’t overwrite the original restore target (unless explicitly overridden)

Offline can mirror this:

* If `soft_active` is false: capture original restore target
* If `soft_active` is true: do **not** change restore target, but do restart countdown and update temp bias

That gives you stable “restore” behavior when soft cuts are spammed.

### D) Decrement and restore *after each pipeline call*

Realtime decrements at the end of `process_chunk()`, after generation, at the “chunk boundary.”

Offline should do the same pattern:

* after `output = pipeline(...)`:

  * if `soft_active`: `soft_chunks_remaining -= 1`
  * if it hits 0: restore bias and clear state

### E) Restore semantics: what you can match vs what you can’t (without timeline changes)

#### What you can match today

You can match “restore to whatever the bias was before soft cut started” if the renderer always carries a base bias in `parameters`.

However, realtime distinguishes:

* restore to explicit previous bias
* restore to `<unset>` (delete the key)

Offline currently always sets:

```py
parameters["kv_cache_attention_bias"] = float(kv_cache_attention_bias)
```

So it never represents “unset” anyway.

#### If you want closer realtime alignment

You have two options:

1. **Represent “unset” in offline renderer**

   * If timeline `settings.kvCacheAttentionBias` is omitted and CLI doesn’t override, don’t set it in `parameters` at all.
   * Let the pipeline/config default apply.
   * Then your soft cut restoration can truly “pop” the key back to unset.

2. **Extend timeline format**

   * Add fields like:

     * `softCut: { bias, chunks, restoreBias?, restoreWasSet? }`
   * Your server-side recorder already has the internal state to write these two bits (`_soft_transition_original_bias` and `_soft_transition_original_bias_was_set`).
   * Then offline can restore exactly like realtime (including `<unset>`).

Given your own notes (“needs restore-target nuance”), option (2) is the fidelity path.

---

## 6) Important ambiguity: time-based segments vs chunk-based softCut duration

You already called this out, and it matters.

### Current renderer chooses segments by time:

```py
current_time = produced_frames / fps
... if current_time >= next_segment.startTime: advance ...
```

But your server-side recorder can export chunk indices (`startChunk`, `endChunk`) and records soft cuts in chunk counts.

### What this means for replay fidelity

* **softCut duration should be chunk-based** (pipeline calls), not wall-time.
* Segment selection being time-based is “okay,” but it can drift if:

  * the recorded `startTime` reflects realtime stalls/pauses
  * offline uses different FPS than realtime capture
  * `num_frame_per_block` differs from the renderer’s assumption (currently hardcoded to 3)

### Recommended follow-up for high-fidelity replay

Add an optional “chunk timebase mode” to the renderer:

* If segments contain `startChunk/endChunk`, allow rendering by “pipeline call index” instead of wall time.
* Soft cuts, hard cuts, and prompt boundaries all become precise in the same domain.

This is not required to *support* `initCache/softCut`, but it’s the main way to make the server-side recorder’s “chunk is primary” claim actually pay off in offline replay.

---

## 7) Summary of concrete code touch points in `render_timeline.py`

To support `initCache` and `softCut` replay (without implementing here), you will need edits in:

1. **Schema**

   * `TimelineSegment`: add `initCache`, `softCut`
   * add `TimelineSoftCut` model (recommended)

2. **Dry-run planner (optional but recommended)**

   * include `initCache` / `softCut` info per segment in the printed plan

3. **Render loop**

   * introduce `pending_init_cache` one-shot behavior at segment boundaries
   * introduce soft-transition state machine:

     * start on segment boundary when softCut present
     * apply temp bias immediately
     * decrement per pipeline call
     * restore when countdown ends
     * handle re-trigger without clobbering restore target

4. **(Optional fidelity improvements)**

   * support “unset bias” in parameters
   * and/or support chunk-based segment scheduling using `startChunk/endChunk`

---

If you want, I can also point out exactly where `render_timeline.py` currently hardcodes assumptions that will subtly affect replay correctness once you start caring about chunk-precise behavior (e.g., `frames_per_call = 3` in dry-run vs potentially reading `num_frame_per_block` from model config).
