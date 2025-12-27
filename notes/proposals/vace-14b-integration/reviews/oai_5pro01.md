Below is a vet of `notes/proposals/vace-14b-integration.md` against the **current** code you provided. I’m focusing on the specific axes you asked for: VACE toggling semantics, Krea wiring, artifacts, schema/load params, block graph ordering, and KV-cache recompute behavior.

---

## What the plan gets right (matches current code)

### VACE is load-time, not per-request

This is accurate given current implementation:

* `VACEEnabledPipeline._init_vace()` only enables VACE when **`config.vace_path` is set at pipeline init** (`wan2_1/vace/mixin.py`).
* The server routes incoming frames based on **`pipeline.vace_enabled`** (`server/frame_processor.py` uses `getattr(pipeline, "vace_enabled", False)` and switches `video` vs `vace_input_frames`).
* So “toggle requires reload” is correct: enabling/disabling VACE is a **pipeline reload concern**, not a runtime “Parameters” concern.

### “VACE video mode is not normal V2V”

Correct and already embodied in current routing + block triggers:

* When VACE is enabled, `frame_processor` routes buffered frames to `vace_input_frames`, not `video`.
* `AutoPrepareLatentsBlock` only switches to V2V latent path on `"video"`, not `"vace_input_frames"` (`auto_prepare_latents.py`).
* Therefore VACE-enabled “video mode” keeps the pipeline on the **T2V latent path**, using the input frames as conditioning only.

### You really do need `VaceEncodingBlock` before `denoise`

This matches how every other VACE-enabled pipeline is wired:

* `longlive`, `streamdiffusionv2`, `reward_forcing` all insert `("vace_encoding", VaceEncodingBlock)` before `("denoise", DenoiseBlock)`.

---

## Hard mismatches: plan vs current code

These are “plan says do X” and **the repo currently does not**.

### 1) Artifacts: Krea doesn’t download VACE-14B today

**Current state**

* `server/pipeline_artifacts.py` defines only:

  * `VACE_ARTIFACT` = `Wan2_1-VACE_module_1_3B_bf16.safetensors`
* `krea-realtime-video` artifacts list does **not** include any VACE module.

**Plan is correct to add a 14B artifact**, but note the operational consequence:

* The artifact list is **static per pipeline**. Adding the 6.1GB module means it will download whenever Krea pipeline artifacts are fetched—**even if `vace_enabled=False`**. If you care about conditional downloads, that’s an extra engineering step (not in the plan) because current artifact plumbing doesn’t look load-param dependent.

### 2) Server load params: Krea load schema lacks `vace_enabled`

**Current state**

* `LongLiveLoadParams` and `StreamDiffusionV2LoadParams` already have `vace_enabled: bool = True`.
* `KreaRealtimeVideoLoadParams` has **no** `vace_enabled`.

So the plan’s Step 2 is still required and not implemented.

### 3) Pipeline manager: Krea load branch never configures VACE

**Current state**

* `_get_vace_checkpoint_path()` is hardcoded to the **1.3B** module:

  * `"WanVideo_comfy/Wan2_1-VACE_module_1_3B_bf16.safetensors"`
* Krea load branch (`elif pipeline_id == "krea-realtime-video": ...`) does **not** check `vace_enabled` and does **not** call `_configure_vace()` at all.

So the plan’s Step 3 is still required and not implemented.

Also: the plan says “teach pipeline_manager to select 14B VACE module.” You must do that **without breaking existing 1.3B pipelines**. Current `_get_vace_checkpoint_path()` has no parameterization, so “just change it to 14B” would silently break LongLive/StreamDiffusion/RewardForcing.

### 4) Krea pipeline does not implement VACE mixin or ordering

**Current state**

* `KreaRealtimeVideoPipeline` inherits: `class KreaRealtimeVideoPipeline(Pipeline, LoRAEnabledPipeline)`
* It does not include `VACEEnabledPipeline`
* It never calls `_init_vace()`

The plan’s Step 4–5 is still required and not implemented.

### 5) Krea modular blocks omit `VaceEncodingBlock`

**Current state**

* `krea_realtime_video/modular_blocks.py` includes:

  * `("recompute_kv_cache", RecomputeKVCacheBlock)`,
  * then `("denoise", DenoiseBlock)`
* There is **no** `("vace_encoding", VaceEncodingBlock)`.

So the plan’s Step 7 is still required and not implemented.

---

## Subtle but important mismatches / incorrect assumptions

These are places where the plan is directionally right but misses a key detail in how the current code actually behaves.

### A) `_init_vace()` device handling can create device-mismatch in Krea’s current init flow

This is the biggest “gotcha” I see that the plan doesn’t fully account for.

In `VACEEnabledPipeline._init_vace()`:

* The wrapper (`CausalVaceWanModel`) initially creates `blocks` and `vace_blocks` on the **same device as the base model blocks** (it copies device from `causal_wan_model.blocks[0]`).
* Then the mixin does:

  * `vace_wrapped_model.vace_patch_embedding.to(device=device, dtype=dtype)`
  * `vace_wrapped_model.vace_blocks.to(device=device, dtype=dtype)`

That only moves **VACE-specific modules**, not the base blocks.

So: if you call `_init_vace(..., device=torch.device("cuda"), ...)` while the base model is still on CPU, you will end up with:

* base blocks on CPU
* `vace_blocks` (and patch embedding) on GPU

That’s a runtime error waiting to happen.

**Why this matters specifically for Krea:**

* Krea’s init currently appears designed to do a lot of work (fuse projections, init LoRAs) *before* the eventual move/quantization step.
* If the base model is still CPU at that point, calling `_init_vace` with `device=cuda` is unsafe.

**What you should add to the plan:**

* When wiring Krea, ensure `_init_vace` uses the **current model device**, not the “target device.”

  * E.g. pass `device=next(generator.model.parameters()).device` at the time you call `_init_vace`.
* Or adjust `_init_vace` itself to default to the base model device if `device` is `None` / not provided.

The plan mentions OOM concerns from moving VACE to GPU before FP8 quantization, but it doesn’t explicitly call out the **device split hazard**.

### B) State hygiene: plan mentions `vace_ref_images`, but `vace_input_frames` can also go stale

Plan correctly calls out the `vace_ref_images` one-shot issue (LongLive clears it in pipeline state).

But Krea’s `_generate()` currently only clears:

* `transition` if missing
* `video` if missing

It does **not** clear:

* `vace_ref_images`
* `vace_input_frames`
* `vace_input_masks`

Even though `frame_processor` clears `vace_ref_images` from its own parameter store after sending once, the pipeline `PipelineState` will retain whatever was last set unless explicitly cleared.

This is important because:

* If you ever run a chunk where no new `vace_input_frames` arrives (mode switch, buffering gap, etc.), stale values could be reused.

**Recommendation:** update the plan to include “clear `vace_input_frames` (and masks) from state if not present in kwargs,” not just `vace_ref_images`.

### C) `_configure_vace()` is partly misaligned with the *actual* runtime surface

The plan already hints this, but it’s worth making explicit:

* `_configure_vace(config, load_params)` currently tries to read:

  * `ref_images` and `vace_context_scale` from `load_params`
* But the server runtime parameter surface uses:

  * `vace_ref_images` and `vace_context_scale` in `Parameters`

So `_configure_vace` is mostly about enabling the module (`config["vace_path"]`) and the rest is currently either unused or inconsistent naming.

**Practical consequence:** don’t block Krea VACE enablement on making `ref_images` work in load params; the important part is `vace_path` + VaceEncodingBlock + model wrapper.

---

## Block graph ordering: plan is mostly right, but watch Krea’s unique recompute block

### What’s correct

Placing `VaceEncodingBlock` between `auto_prepare_latents` and `denoise` matches all other pipelines.

### What the plan should additionally acknowledge

Krea inserts `recompute_kv_cache` *before* `denoise`.

You can place `vace_encoding`:

* **before** recompute (as plan suggests), or
* **after** recompute (still before denoise)

Functionally, if recompute does not consume `vace_context`, either ordering is fine.

But it matters for the KV-recompute decision below.

---

## KV-cache recompute: the plan’s “Option B” is underspecified relative to current Krea behavior

### Current behavior (ground truth)

`RecomputeKVCacheBlock` recomputes KV cache by calling:

```py
components.generator(
  noisy_image_or_video=context_frames,
  conditional_dict={"prompt_embeds": conditioning_embeds},
  timestep=context_timestep,
  kv_cache=...,
  crossattn_cache=...,
  current_start=...
)
```

No `vace_context`, no `vace_context_scale`.

### Why simply “passing `vace_context`” is not straightforward

There are two distinct VACE regimes:

1. **Reference-image-only (R2V)**

   * `vace_context` is derived from reference images and is stable across chunks.
   * Passing it into recompute is plausible and semantically coherent.

2. **VACE V2V editing (`vace_input_frames` per chunk)**

   * `vace_context` is derived from *current chunk* conditioning frames (depth/flow/frames).
   * KV recompute, however, feeds **context frames from buffers** (past frames).
   * You generally do *not* have matching “conditioning inputs” for those buffered frames unless you store them.

So “Option B: pass `vace_context` during recompute” needs to clarify:

* Is it intended only for reference-image-only mode?
* If not, what conditioning frames correspond to the buffered context frames?

### What’s missing from the plan (suggested addition)

If you want recompute to be VACE-consistent in V2V-editing mode, you’d need extra plumbing, e.g.:

* store a parallel history buffer of `vace_input_frames` / masks aligned with `context_frame_buffer`, or
* explicitly define that recompute uses **no VACE** by design (and accept that it’s an approximation).

### A practical recommendation aligned with current code shape

A plan-aligned but code-realistic policy could be:

* **Default:** keep recompute as-is (no vace) to avoid complexity
* **Upgrade path:** if `vace_ref_images` is present (reference-only), pass `vace_context` and `vace_context_scale` into recompute (since that context is stable and meaningful)

That keeps “Option B” from accidentally doing something incoherent for per-frame conditioning.

---

## Additional “missing steps” the plan doesn’t call out but you likely need

### 1) Compile path won’t touch `vace_blocks`

Krea has an opt-in `compile` mode that compiles either:

* `generator.model` (reduce-overhead), or
* each block in `generator.model.blocks`

But `CausalVaceWanModel` runs the VACE pathway via `self.vace_blocks` and `block.forward_vace(...)`.

If you care about perf parity when `compile=True`, you may want to also compile:

* `for block in generator.model.vace_blocks: block.compile(...)`

Not required for correctness, but a likely perf cliff vs baseline Krea.

### 2) Projection fusing needs to cover VACE blocks too

The plan calls this out and it is correct, but I’ll restate the concrete delta:

Current Krea code fuses only:

* `for block in generator.model.blocks: block.self_attn.fuse_projections()`

With VACE enabled, you must fuse:

* `generator.model.blocks` **and**
* `generator.model.vace_blocks` (guarded with `hasattr`)

---

## Concrete delta list: what is currently missing vs plan (by file)

### `src/scope/server/pipeline_artifacts.py`

* Missing: `VACE_14B_ARTIFACT` definition
* Missing: adding it to `"krea-realtime-video"` artifacts list
* Optional extra not in plan: conditional artifact download based on `vace_enabled` (requires new mechanism)

### `src/scope/server/schema.py`

* Missing: `vace_enabled: bool` in `KreaRealtimeVideoLoadParams` (plan calls this out)

### `src/scope/server/pipeline_manager.py`

* Missing: a way to select **14B** VACE module for Krea while keeping **1.3B** for other pipelines
* Missing: in the Krea branch, read `load_params["vace_enabled"]` and call `_configure_vace()` when enabled

### `src/scope/core/pipelines/krea_realtime_video/pipeline.py`

* Missing: `VACEEnabledPipeline` mixin
* Missing: call to `_init_vace()` in `__init__`
* Missing: ordering fixes (VACE wrapping must occur **before** fusing + before `_init_loras`)
* Missing: fuse projections for `vace_blocks`
* Missing: state hygiene for `vace_ref_images` (and realistically `vace_input_frames` too)

### `src/scope/core/pipelines/krea_realtime_video/modular_blocks.py`

* Missing: import and insertion of `VaceEncodingBlock`
* Missing: block name entry, likely `"vace_encoding"` after `auto_prepare_latents`

### `src/scope/core/pipelines/krea_realtime_video/blocks/recompute_kv_cache.py`

* Current: recompute ignores `vace_context` entirely
* Plan: calls this a decision point (correct), but needs more precise semantics as described above

---

## Bottom line

The plan is broadly aligned with the repo’s VACE architecture and with how 1.3B pipelines work, but it needs a few updates to match **current Krea realities**:

1. **Device alignment during `_init_vace`** is a real integration hazard that the plan doesn’t explicitly surface.
2. **State hygiene should cover `vace_input_frames` as well**, not just `vace_ref_images`.
3. The KV-recompute “Option B” needs a more explicit definition (especially for VACE V2V editing), otherwise it’s easy to implement something wrong-but-silent.
4. If you care about Krea’s `compile=True` perf, consider compiling `vace_blocks` too (not in plan).

If you want, I can rewrite the integration plan sections (Steps 3–8) into an “as-implemented” checklist that bakes in the above corrections (device selection + state clearing + recompute policy).
