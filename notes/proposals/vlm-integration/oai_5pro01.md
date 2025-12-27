Here’s feedback on `notes/proposals/vlm-integration.md`, focused on clarity, feasibility, and alignment with what’s already in the repo (Gemini integration, REST/CLI shape, and the roadmap/spec docs). I’m not implementing anything—this is purely review + risk/consistency callouts.

## Overall assessment

The proposal is directionally solid and mostly aligned with the repo’s “CLI-first / REST over WebRTC control-plane” architecture:

* It correctly identifies **three distinct VLM capabilities** and separates them cleanly (prompt compilation vs frame description vs image editing).
* It correctly anchors “context editing” on the **decoded RGB buffer → VAE re-encode → KV cache recompute** mechanism that exists in `recompute_kv_cache.py`.
* The implementation order (frame analysis → validation spike → image editing) matches the risk profile and lines up with `notes/realtime-roadmap.md` + `notes/capability-roadmap.md`.

Where it needs tightening is: **endpoint/CLI canonicalization**, **what frame is being analyzed/edited (latest vs anchor)**, **thread-safety/race semantics**, and **model/SDK response parsing details** (especially for image editing).

---

## Clarity review

### What’s clear and good

* The “three capabilities, one integration” table is a great framing.
* The “Mechanism” section for image editing is succinct and accurately points to the exact edit surface in `get_context_frames()` (re-encoding from `decoded_frame_buffer[:, :1]`).
* The proposal calls out the **validation spike** explicitly, which is important given “context editing” is still speculative in the roadmap/specs.

### Where clarity breaks down / needs explicitness

1. **Which frame is “the frame” for describe/edit?**

   * `Frame analysis` implies “current output frame” (likely `FrameProcessor.get_latest_frame()` via `/api/v1/realtime/frame/latest`).
   * `Image editing` implies the **anchor frame** (the one that will be re-encoded into KV cache), which is *not* the same as “latest” and is not currently exposed via REST.
   * If a user calls `video-cli frame` (current implementation) they get `/frame/latest` (latest output), but if they call `video-cli edit` per the plan, you’ll be editing `decoded_frame_buffer[:, :1]` (oldest frame in the decoded buffer window). That mismatch will confuse everyone unless the doc explicitly distinguishes:

     * “latest output frame” vs
     * “anchor/context frame used for recompute”

2. **The `confidence` field in describe responses**

   * The plan’s response example includes `"confidence": 0.85`.
   * Gemini (as currently used in this repo) won’t give you a calibrated confidence score out of the box. Unless you’re planning to have the model *self-report* confidence (which is not reliable), this field will either be fake or brittle.
   * Recommendation for doc clarity: either remove it, make it optional/`null`, or rename it to something honest like `self_reported_confidence` / `heuristic_score`.

3. **Step → describe sequencing**

   * The proposed agent loop does: `step` then immediately `describe-frame`.
   * In the actual server, `/api/v1/realtime/step` queues a step (`status="step_queued"`). It does not wait for completion.
   * That means `describe-frame` can easily analyze the *previous* frame unless the agent also waits for `chunk_index` to advance or polls `/frame/latest` until it changes.
   * This is a doc-level mismatch with real semantics and should be called out.

---

## Alignment with existing REST conventions

### What exists today

`src/scope/server/app.py` already uses a clear namespace:

* Control endpoints: `/api/v1/realtime/*` (state/run/pause/step/prompt/frame/latest/world/style/hard-cut/soft-cut)
* Gemini text features:

  * `POST /api/v1/realtime/world/change`
  * `POST /api/v1/prompt/jiggle` (note: **not under `/realtime/`**)

### Inconsistencies to resolve in the proposal/doc set

1. **Endpoint path drift across docs**

   * Your ambiguity note already flags this, and it’s real:

     * `vlm-integration.md` proposes `/api/v1/realtime/frame/describe` and `/api/v1/realtime/frame/edit`.
     * older specs (`context-editing-spec.md`, `notes/reference/cli-implementation.md`) use `/api/frame/describe`, `/api/edit`, etc.
   * The *current* repo has standardized on `/api/v1/realtime/...` for the realtime control plane, so the proposal should declare that as canonical and explicitly mark the older `/api/*` paths as legacy/outdated.

2. **`/api/v1/prompt/jiggle` vs `/api/v1/realtime/...`**

   * In `app.py`, `prompt/jiggle` sits at `/api/v1/prompt/jiggle` while world-change is `/api/v1/realtime/world/change`.
   * The proposal treats prompt jiggle as part of “realtime VLM integration”. That’s conceptually true, but the pathing is inconsistent.
   * If you’re not going to move it, the plan should at least note: “jiggle is currently not under `/realtime/`”.

3. **GET+POST variants for describe**

   * Proposal suggests:

     * `GET /api/v1/realtime/frame/describe`
     * `POST /api/v1/realtime/frame/describe` (with custom prompt)
   * That’s fine, but it’s worth documenting the exact semantics:

     * GET = default system prompt, analyze latest frame
     * POST = override prompt, analyze latest frame (or provided frame?)
   * Otherwise implementers will diverge.

---

## Alignment with existing CLI conventions

### What exists today

`src/scope/cli/video_cli.py` is:

* click-based
* JSON output only
* points at `/api/v1/realtime/*` for state/run/pause/step/prompt/frame/latest/world/style/list

### Gaps vs proposal

* The proposal wants:

  * `video-cli describe-frame`
  * `video-cli edit`
* The current CLI (in the excerpt) does not include either, and it also doesn’t expose:

  * `world change` (natural language)
  * `prompt jiggle`
  * `hard-cut` / `soft-cut` (even though server supports them)
* This isn’t a problem, but the proposal should explicitly say:

  * where these commands will live (top-level vs subgroup, e.g. `video-cli frame describe`)
  * which endpoints they hit (and how they relate to the “older CLI spec” in `notes/reference/cli-implementation.md` which uses legacy paths)

Also: because of the “latest vs anchor” mismatch, the proposal should either:

* introduce `video-cli frame --anchor` (as the older spec did), or
* explicitly say `video-cli edit` edits the “context anchor frame,” not the latest output frame, and provide a way to preview it.

---

## Feasibility notes

### Frame analysis is feasible, but decide “frame source”

You already have the key infrastructure:

* `FrameProcessor.latest_frame_cpu` + `get_latest_frame()` provides a **non-destructive** CPU uint8 frame.
* `/api/v1/realtime/frame/latest` already returns PNG bytes based on that.

So frame analysis can be implemented without touching GPU buffers at all.

What’s missing in the proposal:

* Decide whether describe analyzes:

  * latest output frame (easy, low-risk), or
  * anchor/context frame (harder, requires new access path)

Given your roadmap (“Phase 8 VLM feedback loop”) and existing endpoint (`frame/latest`), the “least risky first step” is:

* **describe latest output frame**.

But if your agent loop wants to evaluate *what’s being conditioned* (the context anchor), you’ll eventually want anchor access too. The plan should acknowledge this split.

### Image editing is feasible, but it’s substantially riskier than the doc implies

The doc’s mechanism is correct, but the integration complexity is under-described:

1. **Thread safety / race conditions**

   * The decoded buffer lives in pipeline state on GPU, and generation happens in a worker thread (`FrameProcessor`).
   * Editing that buffer from an HTTP request handler is a race unless you enforce:

     * “must be paused” **and**
     * a lock/handshake with the worker thread (or schedule the edit through the control bus / chunk boundary)
   * The proposal mentions none of this, but your `context-editing-spec.md` correctly calls out “must be paused to edit”.

2. **Recompute cadence can delay propagation**

   * `recompute_kv_cache.py` has an opt-in performance knob `SCOPE_KV_CACHE_RECOMPUTE_EVERY`.
   * If that is set > 1, edits won’t propagate on “next chunk” reliably.
   * The proposal assumes “next recompute will encode the edit” — that may be false in steady-state if skipping is enabled.
   * This is an important risk to document because it changes the UX (“why didn’t my edit stick?”).

3. **Buffer readiness + early-session behavior**

   * Early in the run, `get_context_frames()` uses `first_context_frame` + `context_frame_buffer` and only later re-encodes from `decoded_frame_buffer`.
   * Editing decoded buffer too early may do nothing (or not be used yet).
   * The proposal should say when the edit surface becomes “active” (e.g., once `current_start_frame - num_frame_per_block >= kv_cache_num_frames`).

4. **Value range / dtype / shape invariants**

   * The buffer is likely `bfloat16` on GPU; Gemini requires PIL/bytes on CPU.
   * You need strict invariants:

     * same H/W
     * same channel order
     * expected range (likely [0,1] floats in the decoded buffer path)
   * If Gemini returns an image with different dimensions or mode, you need to handle or reject it; otherwise you can crash the pipeline.

---

## Alignment with existing Gemini integration

### What’s aligned

* Using `src/scope/realtime/gemini_client.py` as the single integration hub is consistent with how prompt compilation/world-change/jiggle are done.
* The plan’s class structure (`GeminiFrameAnalyzer`, `GeminiFrameEditor`) matches the existing style (`GeminiWorldChanger`, etc.).

### What’s missing / inconsistent with established patterns

1. **Image parsing robustness**

   * Your `gemini-cookbook.md` strongly recommends:

     * “Try `part.as_image()` first, then fallback to decoding `inline_data`.”
   * The proposed `GeminiFrameEditor.edit()` sketch only checks `inline_data`. That’s likely to be brittle across model variants.

2. **Model routing / capabilities**

   * Cookbook calls out substring-based routing as brittle (e.g., `"flash-image"` in model name).
   * The proposal doesn’t mention how model IDs will be configured/selected for:

     * vision description (flash/pro vision)
     * flash image edit
   * At minimum the doc should acknowledge:

     * “we will likely need a small registry / config mapping model_id → capability”
       even if you don’t implement the full registry yet.

3. **Sync calls inside async endpoints**

   * Existing `world/change` endpoint calls Gemini synchronously inside an `async def`. That’s already the pattern, but it can block the event loop.
   * Frame describe/edit will be called more frequently than world-change in agent loops, so the blocking risk is higher.
   * The proposal should call out that these are potentially slow and may need thread offload if responsiveness matters.

4. **Rate limiting**

   * `gemini_client.py` has a simple global `_rate_limit()` already.
   * The proposal doesn’t mention whether describe/edit should use the same limiter, or have their own, or be gated by server state (paused / cooldown).

---

## Missing steps the proposal should include

These aren’t “implement now” requests—just doc-level gaps that matter for feasibility and avoiding surprises:

1. **Define canonical REST paths and update the older specs**

   * State explicitly: “All realtime VLM endpoints live under `/api/v1/realtime/*`.”
   * Mark `/api/edit`-style endpoints in older docs as legacy.

2. **Define “latest frame” vs “anchor frame”**

   * Add a short section:

     * “describe-frame analyzes latest output frame” (v1)
     * “edit modifies anchor frame used for recompute” (v2)
   * If you want both, include an explicit `target` parameter (`latest|anchor`) or separate endpoints.

3. **Add an “anchor frame fetch” plan**

   * Editing UX needs a way to preview the anchor frame.
   * Your older `cli-implementation.md` had `frame --anchor`. The proposal should at least mention this requirement.

4. **Paused-state requirement + locking semantics**

   * For editing, document: “must be paused” and “edit is applied atomically relative to chunk processing.”
   * Otherwise you’ll get heisenbugs.

5. **Describe/edit should return chunk context**

   * Include `chunk_index` in responses (you already do in examples).
   * But also clarify whether the index is:

     * “current chunk produced” vs
     * “chunk at which edit will take effect”.

6. **Agent loop needs a “wait-for-step” strategy**

   * Either:

     * document polling on `chunk_index`, or
     * document that the CLI should wait until the step completes before returning (if you ever change semantics).
   * Right now it’s queue-based, so agents must wait.

7. **Test plan**

   * For frame analysis: unit test that endpoint returns 400/404 correctly when no frame exists; mock Gemini client.
   * For editing: CPU-only validation spike test is good; add doc note about how to run it and what “success” looks like.
   * Mention how to run tests without Gemini key (skip/xfail/mocks).

---

## Risk register

If you want a compact “risks” section in the proposal (recommended), these are the big ones:

* **Model access / allowlisting**: Flash image edit models may not be available everywhere (you already hint at this).
* **Event loop blocking**: describe-frame may be called frequently; sync Gemini calls inside async FastAPI could stall other endpoints.
* **Frame mismatch UX**: editing `decoded_frame_buffer[:, :1]` won’t match `/frame/latest`.
* **KV recompute skipping** (`SCOPE_KV_CACHE_RECOMPUTE_EVERY`) can make edits appear “not applied”.
* **Early-run no-op**: editing before decoded buffer participates in recompute may do nothing.
* **Invariant enforcement**: edited frame must keep shape/range; otherwise pipeline may crash or produce garbage.

---

## Suggested small doc edits (non-code) to make it “implementation-ready”

1. Add a **“Canonical API + CLI naming”** section at the top that declares:

   * API base paths
   * CLI commands
   * how these relate to older docs (deprecated paths)

2. Add a **“Frame targets”** section:

   * latest output frame vs anchor/context frame
   * which endpoints operate on which

3. Replace `confidence` with something implementable or optional.

4. Add a **“Paused + atomicity”** note under Image Editing:

   * “edits are only supported while paused” (or if you want to support live edits, document the scheduling semantics explicitly)

5. Add one paragraph noting the recompute skip knob and its impact on edit latency.

---

If you want, I can also produce a “diff-style” set of doc edits (bullet-by-bullet replacements) that only changes `notes/proposals/vlm-integration.md` text to resolve these inconsistencies—still without implementing any code.
