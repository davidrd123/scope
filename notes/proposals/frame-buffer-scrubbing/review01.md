Below is a vet of `notes/proposals/frame-buffer-scrubbing.md` against the **current** backend + CLI you pasted (FrameProcessor/VideoProcessingTrack/WebRTC/app.py/video_cli.py). I’m not implementing anything—this is strictly “proposal correctness + alignment” and concrete edits you should make.

---

## 1) Major naming and object-model conflicts

### A. `FrameProcessor.frame_buffer` is already the **input** deque

In `src/scope/server/frame_processor.py`, `self.frame_buffer = deque(...)` is the **video input buffer** consumed by `prepare_chunk()` and protected by `frame_buffer_lock`.

But the proposal introduces a `FrameBuffer` and then says:

```py
buffer = session.video_track.frame_processor.frame_buffer
```

That is a hard mismatch: today that field is the input deque of `VideoFrame` objects, not a “history of generated output frames”.

**Edit the proposal:**

* Rename the proposed output pixel buffer to something that cannot collide with:

  * `FrameProcessor.frame_buffer` (input),
  * pipeline continuity keys like `context_frame_buffer` / `decoded_frame_buffer`,
  * and the already-used term “FrameBus” in `notes/realtime_video_architecture.md`.

Concrete options that align better:

* `OutputFrameBuffer`
* `FrameHistory`
* `RenderedFrameRing`
* `ChunkStore` (if you store by chunk)

And explicitly rename the existing input buffer in code terms (proposal-level language) to: `input_frame_deque` / `input_video_buffer` to prevent confusion.

### B. Terminology conflict with `FrameBus` design doc

`notes/realtime_video_architecture.md` already defines a **FrameBus** concept (publish + ring buffer indexed by `chunk_index`). The proposal’s “FrameBuffer” overlaps that purpose but uses different semantics (per-frame objects + base64 JPEG conversion).

**Edit the proposal:**

* Either:

  1. Reframe this proposal as implementing the *FrameBus/ChunkStore* concept (store by chunk packets), **or**
  2. Add a “Relationship to FrameBus” section that states clearly why this is *not* FrameBus and how the two coexist.

Right now, it reads like a parallel competing design.

---

## 2) Data-format mismatches with actual FrameProcessor output

### A. Proposal assumes float tensors in `[0..1]`, but the server produces CPU `uint8` `[0..255]`

In `FrameProcessor.process_chunk()` today:

* `output = pipeline(**call_params)` returns a tensor (likely float in `[0..1]`)
* then FrameProcessor converts it to:

```py
output = (
  (output * 255.0)
  .clamp(0, 255)
  .to(dtype=torch.uint8)
  .contiguous()
  .detach()
  .cpu()
)
```

So **after** this point, frames are CPU `uint8` shaped `(T, H, W, C)`.

But the proposal’s `FrameBuffer.add_frame()` does:

* treat tensor as float and multiply by 255 again
* assumes tensor might be `[C, H, W]`
* converts using PIL

**Edit the proposal:**

* Define the canonical format you will store, and match the real runtime:

  * **Recommended canonical storage** (given current code): CPU `uint8` in `(H, W, C)` (RGB24).
* Adjust conversion logic in the proposal text accordingly:

  * No `*255` scaling if the capture point is after FrameProcessor’s conversion.
  * Don’t assume CHW.

### B. Channel order mismatch: proposal mixes CHW/HWC

Current output queue frames are pushed as `(H, W, C)` tensors (`VideoProcessingTrack.recv()` calls `frame_tensor.numpy()` and passes `format="rgb24"`).

Proposal uses mixed assumptions:

* `tensor: torch.Tensor | None  # GPU tensor [C, H, W]`
* `numpy: np.ndarray | None     # CPU numpy [H, W, C]`

**Edit the proposal:**

* Pick one internal representation for stored frames to avoid constant transposes:

  * If you keep CPU, keep `(H, W, C) uint8` everywhere (matches WebRTC + Spout output paths).
* If you *do* offer GPU mode later, specify exactly where you intercept (see next section).

### C. `3 frames per chunk` is not a safe assumption

The proposal hardcodes:

* “3-frame chunks”
* `frame_index` is 0,1,2

But the server’s chunk output count is literally:

```py
num_frames = output.shape[0]
```

So it can vary by pipeline/config.

**Edit the proposal:**

* Replace “3 frames per chunk” with:

  * “`frames_per_chunk = output.shape[0]` at runtime; frame_index is `0..frames_per_chunk-1`.”
* Include `frames_per_chunk` in buffer stats and in API responses so clients can render correctly.

---

## 3) “GPU storage mode” is not currently aligned with the real capture point

The proposal offers `storage_mode: "gpu" | "cpu" | "jpeg"`.

But today, by the time frames exist as “frames” (and not model latents), they are already moved to CPU `uint8` inside `process_chunk()`.

**Implication:** a “gpu” mode would require a different capture point (before `.cpu()`), and you’d have to decide what you’re storing:

* float frames on GPU (expensive VRAM),
* or convert to `uint8` on GPU and keep that (still VRAM, but less),
* or store latents (not directly scrub-viewable).

**Edit the proposal:**

* Either drop `"gpu"` from MVP storage modes, **or**
* Add an explicit “Capture point” section:

  * MVP: capture **after** conversion (CPU uint8)
  * Future GPU mode: capture **before** `.cpu()`, define exact dtype/shape

Right now it implies you can switch to GPU mode without changing where the data is produced.

---

## 4) Metadata mismatches: “prompt” and “seed” aren’t single fields

### A. Prompt metadata is not a single string in current system

Today “the prompt” may come from:

* explicit `prompts` list in `self.parameters` (list of dicts),
* a `transition`,
* style compilation (`_compiled_prompt` from `TemplateCompiler`),
* and world state auto-recompilation.

So “current_prompt = self._get_current_effective_prompt()[0]” in the proposal is fictional; there is no such function in the code you pasted.

**Edit the proposal:**

* Store `prompts` as the canonical list (the thing you actually pass to the pipeline):

  * e.g. `prompts: list[dict] | None` per chunk/frame
* Optionally store:

  * `compiled_prompt_text` for debugging (you already do similar in `Snapshot`)
  * `active_style_name`
* If you really want a single string for UI: define it as “best-effort preview string” derived from the first positive prompt item, but don’t pretend it’s ground truth.

### B. Seed key in FrameProcessor is `base_seed`, not `seed`

`process_chunk()` handles `base_seed` via events and sets `self.parameters["base_seed"]`.

Proposal records:

```py
seed=self.parameters.get("seed")
```

That doesn’t match.

**Edit the proposal:**

* Record `base_seed` (and possibly also pipeline load `seed` from `load_params` if you need it).
* If there’s a concept like “effective seed after offsets” in your pipelines, specify it explicitly (otherwise you’ll lie to downstream tools).

### C. Hard-cut semantics: `reset_cache` vs `init_cache`

The proposal’s frame buffering is mostly independent, but since you’re building a scrub timeline you’ll almost certainly want to mark hard cuts.

In current code:

* `reset_cache` is a control-plane request
* `init_cache` is what gets passed to the pipeline
* also, `init_cache` is automatically true on the first chunk (`not self.is_prepared`)

**Edit the proposal:**

* Add a per-chunk metadata flag like:

  * `init_cache_passed: bool`
* Don’t equate “hard cut happened” with “reset_cache key arrived”; it’s whether `init_cache=True` was actually passed on the call (your session recorder proposal already corrected this—mirror that insight here).

---

## 5) Threading + lifecycle considerations that are missing or understated

### A. Lifecycle: clear buffer on stop / hard cut?

Your FrameProcessor has a clear `stop()` path and also handles `reset_cache` (hard cut) by flushing the **output queue**.

Proposal doesn’t specify:

* when the frame history buffer is cleared (session end, pipeline reload, hard cut?)
* how it behaves across snapshot restore (restoring chunk_index backward)

**Edits to proposal:**

* Specify buffer lifecycle explicitly:

  * On `FrameProcessor.stop()`: clear history
  * On pipeline unload/reload: clear history (or segment it)
  * On snapshot restore: either:

    * keep frames (timeline has branches), or
    * truncate frames newer than restored chunk_index
      Pick one and document it.

### B. Snapshot restore rewinds `chunk_index`

`_restore_snapshot()` sets:

```py
self.chunk_index = snapshot.chunk_index
```

If you have a buffer indexed by chunk_index, rewinding creates ambiguity:

* Do you overwrite existing chunk entries?
* Do you support “branching” histories?
* Do you allow duplicate chunk_index keys?

**Edit the proposal:**

* Add an explicit plan for restore semantics:

  * MVP option: on restore, clear output history and restart buffering from that point.
  * Future option: branch graph / multi-timeline (but that’s larger).

Right now the proposal implies a simple linear time axis that will break on restore.

### C. Locking is fine, but account for high-frequency readers + writers

An RLock around every encode/decode step can become expensive if:

* worker thread writes frames continuously,
* UI scrubbing hits range endpoints rapidly,
* thumbnails endpoint is polled.

**Edit the proposal:**

* Clarify that encoding/compression should happen:

  * either at write-time (store bytes),
  * or at read-time (encode on request),
  * and what lock scope is (lock only around buffer structure, not heavy encoding).

Even at proposal level, call out “avoid holding lock during JPEG encode”.

---

## 6) API endpoint shape: fits the pattern, but response format is risky

### A. Alignment with existing REST patterns

You’ve already established a good pattern in `/api/v1/realtime/frame/latest`:

* `get_active_session()`
* `vt.initialize_output_processing()`
* use FrameProcessor getters
* return binary bytes (`image/png`)

The proposal endpoints don’t mention `initialize_output_processing()`, and they point at the wrong attribute (`frame_buffer`).

**Edit the proposal:**

* Explicitly mirror the `/frame/latest` endpoint flow and error handling.

### B. Base64-in-JSON for many frames will be huge

Returning large ranges as:

```json
{"frames":[{"data":"<base64...>"}]}
```

will:

* inflate payload size,
* stress CPU for encoding,
* increase latency,
* and may hit server/proxy limits quickly.

**Edit the proposal:**

* Make `format=metadata` the default for range queries, or require an explicit `format=jpeg`.
* Add pagination / limits:

  * `max_frames`, `max_chunks`, or server-side clamp with clear error
* Consider a binary endpoint variant (even if not in MVP):

  * `GET /frames/{chunk_index}.jpg?frame_index=...`
  * or a zip stream for export use cases

Even if you keep base64 for convenience, document the limit and the intended usage (thumbnails only).

### C. Multi-session behavior

`get_active_session()` throws if multiple sessions are connected. That’s already a known constraint.

**Edit the proposal:**

* Mention that scrubbing endpoints follow the same single-active-session assumption, or introduce `session_id` query param for parity with WebRTC manager logic.

---

## 7) CLI section is aspirational and not aligned with current `video_cli.py`

The proposal lists:

```bash
video-cli frames stats
video-cli frames export ...
```

But the current CLI:

* has `frame` (singular) and no `frames` group
* does not have snapshot/restore implemented (it returns “not_implemented”)
* uses `/api/v1/realtime/frame/latest` and writes raw bytes

**Edit the proposal:**

* Call out CLI work as “new commands to add” and match CLI conventions:

  * add a `@cli.group()` called `frames`
  * keep binary output behavior consistent (write bytes to disk)
* Avoid implying it already exists.

---

## 8) Memory estimates: directionally right, but the stated “GPU fp32” case is misleading

Given current code, the buffer will almost certainly store:

* CPU `uint8` `(H,W,C)` frames (≈ H*W*3 bytes)

So the “GPU fp32” estimate is only relevant if you redesign the capture point and store float frames.

**Edit the proposal:**

* Add a footnote: “GPU mode requires capturing pre-CPU conversion and is not part of MVP.”
* Provide estimates for **actual likely MVP**:

  * CPU `uint8` stored as torch tensor views vs clones
  * JPEG bytes (quality tradeoffs)

Also important nuance (proposal should mention):

* If you store per-frame **views** into a chunk tensor, you may retain memory for the entire chunk tensor until *all* views are freed—this impacts eviction predictability.

---

## 9) Recommended concrete edits to make to the proposal document

If you want a quick “patch list” for `frame-buffer-scrubbing.md`, here are the highest-impact edits:

1. **Rename** the proposed `FrameBuffer` to `OutputFrameBuffer` (or similar) and fix all attribute references (`fp.output_frame_buffer`, not `fp.frame_buffer`).
2. Add a section: **“Canonical frame format (MVP)”** = CPU `uint8` RGB `(H,W,C)` captured after FrameProcessor normalization.
3. Replace all “3 frames per chunk” assumptions with `num_frames = output.shape[0]`.
4. Replace `prompt: str` with `prompts: list[dict] | None` (and optionally `compiled_prompt_text`).
5. Replace `seed` with `base_seed` (and optionally include `load_params.seed` separately).
6. Add “snapshot restore semantics” for the buffer (clear/truncate/branch).
7. Update API section to:

   * mirror `/frame/latest` initialization pattern
   * add pagination/limits
   * caution against base64 range payloads
8. Clarify that `"gpu"` storage mode is **future**, not aligned with current capture point.

---

## 10) One “alignment choice” you should make explicitly

Right now you have two overlapping conceptual designs:

* **FrameBus (chunk packets)** from `notes/realtime_video_architecture.md`
* **FrameBuffer (per-frame objects + encoding)** in this proposal

To avoid long-term drift, I’d recommend the proposal explicitly choose one of these as the “canonical abstraction”:

* If you want scrubbing + thumbnails: **store by chunk** (FrameBus/ChunkStore) and derive thumbnails lazily.
* If you want random access per frame: still store by chunk internally, but expose `(chunk_index, frame_index)` addressing.

Either way, make the proposal’s terminology match the architecture doc, or explicitly justify divergence.

---

If you want, I can also draft a revised version of `notes/proposals/frame-buffer-scrubbing.md` (text-only edits) that incorporates the fixes above—still no code.
