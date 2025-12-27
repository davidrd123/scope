## Vetting `notes/proposals/style-swap-mode.md` against current code + API docs

This review is based strictly on the *selected* code/docs you provided (FrameProcessor/style registry, style endpoints, LoRA plumbing, pipeline load params, prompt compiler instruction sheet lookup, CLI, and API docs).

---

## What the proposal says that is already true today

### 1) Style switching is “real” in `FrameProcessor` (prompt recompile + one-shot `lora_scales` emission)

Confirmed.

* `FrameProcessor.start()` loads manifests into a `StyleRegistry` from **`Path("styles")`** (repo/CWD-local).
  Evidence: `frame_processor.py` `start()` loads `styles_dir = Path("styles")` and calls `self.style_registry.load_from_directory(styles_dir)`.

* `_rcp_set_style` handling:

  * looks up the manifest in `fp.style_registry`
  * updates `fp.style_manifest`
  * recompiles the prompt using `self.prompt_compiler.compile(self.world_state, self.style_manifest)`
  * **emits `lora_scales` only on “edge trigger”** (`style_changed`), and the emitted list sets:

    * the active style’s LoRA to `lora_default_scale`
    * all other styles’ LoRAs to `0.0`
      Evidence: `frame_processor.py` `_rcp_set_style` block (your excerpt around lines ~900–1069).

So the proposal’s “style switch compiles prompts + emits one-shot `lora_scales`” claim is accurate.

---

### 2) The REST + CLI surfaces for styles exist, and they forward into `_rcp_set_style`

Confirmed.

* CLI:

  * `video-cli style list` → `GET /api/v1/realtime/style/list`
  * `video-cli style set <name>` → `PUT /api/v1/realtime/style` with `{"name": name}`
  * `video-cli style get` → `GET /api/v1/realtime/state` and prints `active_style` + `compiled_prompt`
    Evidence: `src/scope/cli/video_cli.py` excerpt (style list/set/get).

* Server:

  * `PUT /api/v1/realtime/style` forwards `{"_rcp_set_style": request.name}` via `apply_control_message(...)`
  * `GET /api/v1/realtime/style/list` enumerates `fp.style_registry.list_styles()`
    Evidence: `src/scope/server/app.py` excerpt.

So the proposal’s “CLI/REST endpoints exist and drive FrameProcessor style switching” is accurate.

---

### 3) Style endpoints are session-dependent: they require an active WebRTC session

Confirmed (and the proposal already calls this out as a sharp edge).

* `get_active_session()` selects sessions where `pc.connectionState == "connected"`.
* It raises:

  * `"No active WebRTC session"` if zero
  * `"Multiple active sessions (...) specify session_id"` if more than one
    Evidence: `src/scope/server/webrtc.py` excerpt.

So any “style swap mode” behavior is not a global server setting in practice; it’s applied to the currently active session/video track.

---

### 4) Runtime LoRA scale updates exist and are routed through the expected chain

Confirmed.

End-to-end chain is consistent with your architecture section:

* FrameProcessor emits `lora_scales` → enqueued as `EventType.SET_LORA_SCALES` (deterministic ordering at chunk boundaries).
  Evidence: `frame_processor.py` excerpt showing it enqueues `EventType.SET_LORA_SCALES`.

* Pipeline call path handles runtime scale updates:

  * In the Krea pipeline, `lora_scales` is intercepted and `_handle_lora_scale_updates(...)` is called.
  * If `manage_cache` is enabled, it triggers `init_cache=True` (cache reset).
    Evidence: `src/scope/core/pipelines/krea_realtime_video/pipeline.py` excerpt.

* LoRAEnabledPipeline delegates to LoRAManager:

  * `LoRAEnabledPipeline._handle_lora_scale_updates()` calls `LoRAManager.update_adapter_scales(...)`.
    Evidence: `src/scope/core/pipelines/wan2_1/lora/mixin.py` excerpt.

* LoRAManager routes per adapter merge mode:

  * It builds a `path -> merge_mode` map from `loaded_adapters`
  * Groups scale updates by merge_mode
  * Dispatches to the correct strategy (`PeftLoRAStrategy` etc.)
    Evidence: `src/scope/core/pipelines/wan2_1/lora/manager.py` excerpt.

This validates the proposal’s core premise: the runtime update mechanism exists; it’s the pipeline-load/preload and path-identity problems that stop it from working by default.

---

### 5) Docs correctly state the key constraint: runtime `lora_scales` requires `runtime_peft` at load time

Confirmed.

* `docs/api/parameters.md` explicitly says runtime `lora_scales` updates require `lora_merge_mode: "runtime_peft"` at load.
* `docs/api/load.md` shows `lora_merge_mode` and defaults it to `"permanent_merge"`.

That aligns with current pipeline defaults in code (see below).

---

## What the proposal says that is NOT true yet (mismatches vs current code)

These are the “style swap mode” enabling behaviors that are still missing from implementation.

### A) `STYLE_SWAP_MODE=1` is not implemented (no code reads it)

Confirmed mismatch.

In the provided code excerpts:

* `PipelineManager._apply_load_params()` does not check `STYLE_SWAP_MODE`.
* `FrameProcessor.start()` does not check it.
* Nothing in the shown paths reacts to that env var.

So the proposal’s “Activation” section is aspirational right now.

---

### B) PipelineManager does not preload style LoRAs from manifests

Confirmed mismatch.

`PipelineManager._apply_load_params()`:

* defaults `lora_merge_mode = "permanent_merge"`
* only uses `load_params["loras"]` and `load_params["lora_merge_mode"]`
* does **not** scan style manifests / registry
  Evidence: `src/scope/server/pipeline_manager.py` excerpt.

So today, style switching can emit `lora_scales`, but the pipeline will usually **not** have those adapters loaded.

This matches `notes/issues/multi-lora-hot-switching.md`’s stated “current limitation.”

---

### C) The proposal’s “multi-dir style discovery” is not implemented

Confirmed mismatch.

Today:

* `FrameProcessor.start()` uses only `Path("styles")`.
* `StyleRegistry.load_from_directory(...)` loads manifests from the directory you pass it—there’s no helper for `~/.daydream-scope/styles`, no env override, no precedence merging.

So proposal sections about:

* `SCOPE_STYLES_DIRS`
* default dirs `[./styles, ~/.daydream-scope/styles]`
  are not true yet.

---

### D) Manifest iteration is not sorted/deterministic

Confirmed mismatch.

`StyleRegistry.load_from_directory()` iterates `directory.rglob("manifest.yaml")` without sorting.
Evidence: `src/scope/realtime/style_manifest.py` excerpt.

This matters more once you introduce:

* “later dirs win” override semantics, and/or
* a “default style = first loaded” policy

---

### E) Canonical LoRA path normalization is not implemented

Confirmed mismatch.

Today:

* `FrameProcessor` emits `lora_scales` using `manifest.lora_path` **as-is**.
* Pipeline preload/load uses `load_params.loras[].path` **as-is**.
* LoRA updates are matched by exact `path` string identity.

So the proposal’s “canonicalize once and use everywhere” requirement is correct, but it’s not implemented yet.

---

### F) “Warn + skip missing LoRAs” policy conflicts with current runtime_peft loader

Confirmed mismatch.

`PeftLoRAStrategy.load_adapters_from_list()`:

* raises `RuntimeError` when a LoRA file is missing (on `FileNotFoundError`)
* does not warn-and-skip
  Evidence: `src/scope/core/pipelines/wan2_1/lora/strategies/peft_lora.py` excerpt.

So if PipelineManager starts preloading from manifests without filtering, **one missing file will brick pipeline load**.

---

### G) `lora_scales` de-duping by path is not implemented in FrameProcessor

Confirmed mismatch.

`FrameProcessor` builds `lora_updates` by iterating styles and appending an entry per style manifest; it doesn’t dedupe.
Evidence: `frame_processor.py` `_rcp_set_style` excerpt.

---

### H) `STYLE_DEFAULT` / initial activation semantics aren’t implemented

Confirmed mismatch.

`FrameProcessor` starts with `style_manifest = None`; no env var is read; no “initial style set” event is emitted automatically.
Evidence: `frame_processor.py` `__init__` + `start()` excerpt.

---

## Missing details / assumptions / sharp edges (things the proposal should explicitly acknowledge)

Some of these are already mentioned in the proposal’s “Sharp edges / UX gaps” section; a couple are additional.

### 1) Style set API can “succeed” even when the style doesn’t exist

Confirmed.

* `PUT /api/v1/realtime/style` forwards `_rcp_set_style` without validating the style name.
* `FrameProcessor` logs a warning (`Style not found in registry`) but otherwise no-ops.
* The REST endpoint returns `"style_set"` as long as the control message was applied to the session.
  Evidence: `app.py` and `frame_processor.py`.

**Practical impact:** users can get a successful CLI/API response even though nothing changed.

---

### 2) Instruction sheet loading is hardcoded to repo-local `styles/<name>/instructions.md`

Confirmed.

`prompt_compiler._load_instruction_sheet()` computes:

`<repo-root>/styles/<style.name>/instructions.md`

…based on `Path(__file__).parent.parent.parent.parent / "styles" / style.name`.

Evidence: `src/scope/realtime/prompt_compiler.py` excerpt.

Implications:

* Even if you implement multi-dir style discovery, instruction sheets won’t follow it unless you update this logic.
* Also: `StyleManifest` has an `instruction_sheet_path` field, but `_load_instruction_sheet` (in the excerpt) doesn’t use it—so the manifest’s path field is effectively ignored today.

---

### 3) Cache reset semantics are “conditional,” not guaranteed

Confirmed.

* In the Krea pipeline, LoRA scale updates trigger `init_cache=True` **only if** `manage_cache` is enabled.
  Evidence: `krea_realtime_video/pipeline.py`.

FrameProcessor does **not** force:

* `manage_cache=True`, or
* `reset_cache=True`
  on style switch.

So the proposal’s “clean style switches depend on manage_cache staying enabled” is a real constraint, not just a recommendation.

---

### 4) Path-matching footgun is worse than “relative vs absolute”

Confirmed, and worth emphasizing.

Because equality is string-based:

* `rat.safetensors`
* `./models/lora/rat.safetensors`
* `/abs/path/to/rat.safetensors`
  will all be treated as **different adapters** for update matching unless you canonicalize.

This can lead to “LoRA switch” logs while the actual adapter never changes.

---

### 5) Minor behavioral gotcha: style with `lora_path = None` can produce misleading “LoRA switch” logging

This is present in the current FrameProcessor logic.

If the selected style has no `lora_path`:

* `active_lora_path = None`
* the loop still zeros out all other styles’ LoRAs
* and the log line uses `active_lora_path` + `new_style.lora_default_scale`, which can print something like:

  * “LoRA switch: None @ 0.85, zeroed …”
    even though nothing was set active.

Evidence: the `_rcp_set_style` loop + log formatting in your excerpt.

Not catastrophic, but it’s a real “paper cut” if you start using style swap mode heavily.

---

## Mismatches between schema and docs (documentation gaps)

### Per-LoRA `merge_mode` exists in schema/code but is not documented in `docs/api/load.md`

Confirmed.

* `server/schema.py` defines `LoRAConfig.merge_mode` (optional per-LoRA override).
* `docs/api/load.md` documents only:

  * `loras[].path`
  * `loras[].scale`
  * global `lora_merge_mode`

So the capability exists but is currently undocumented.

---

## Bottom line (what this means for “style swap mode” today)

As the proposal’s own “Current behavior” section states (and the code confirms):

`video-cli style set <name>` will **definitely**:

1. switch the active `StyleManifest`
2. recompile prompts
3. emit a `lora_scales` update (edge-triggered)

…but it will **not actually change the LoRA effect** unless all of the following are true:

1. The pipeline was loaded with `lora_merge_mode="runtime_peft"` (docs + code support this).
2. The pipeline already has *all* target style LoRAs loaded.
3. The `path` strings used for preload match exactly the `path` strings emitted on style switch.

Right now, the missing piece is (2) + (3): PipelineManager does not do manifest-driven preload, and there is no canonicalization strategy shared between “load” and “emit.”

---

## Extra “proposal-to-implementation” notes you may want to add to `style-swap-mode.md`

If you want the proposal to be maximally aligned with the current code realities, I’d explicitly add:

* A callout that `_load_instruction_sheet(...)` ignores `StyleManifest.instruction_sheet_path` and is hardcoded to repo-local `styles/<name>/instructions.md`.
* A note that `FrameProcessor` style discovery is **CWD-relative** (`Path("styles")`), while instruction sheet lookup is **code-location-relative** (`Path(__file__).../styles/...`).
* The “selected style has no lora_path” misleading log edge case.

Everything else major in the proposal’s “Implementation Status” section and the prior review (`oai_5pro01.md`) is consistent with the code excerpts you supplied.
