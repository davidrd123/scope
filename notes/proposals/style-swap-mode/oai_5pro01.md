Here’s the “is this doc true yet?” pass on `notes/proposals/style-swap-mode.md`, checked against the code + API docs you included.

I’m going to separate:

* what the proposal **claims**
* what the code/docs **actually do today**
* the **gaps/mismatches** (where the proposal is not yet true)
* a few **missing details / sharp edges** the proposal still doesn’t spell out, but the current code behavior makes relevant.

---

## What is already true in the current implementation

### 1) Style switching is “real” at the FrameProcessor layer (prompt recompiles + one-shot LoRA updates)

The pipeline-side “style” concept is implemented as control messages processed inside `FrameProcessor.process_chunk()`:

* `_rcp_set_style`:

  * looks up the style in `fp.style_registry`
  * sets `fp.style_manifest`
  * (re)creates the compiler via `create_compiler(new_style)` when the style *changes*
  * recompiles prompts
  * and **emits a one-shot `lora_scales` update** *only when the style changes*

This matches the proposal’s architecture description: style switching is driven by `FrameProcessor` via `_rcp_set_style`.

### 2) The runtime LoRA update mechanism exists, and it is keyed by `path` string identity

Your path in the proposal (“FrameProcessor → pipeline call → LoRAEnabledPipeline._handle_lora_scale_updates → LoRAManager.update_adapter_scales → PeftLoRAStrategy”) matches how the code is structured:

* `FrameProcessor` passes `lora_scales` into the pipeline call **only once** (it pops it out of `self.parameters` right before the call).
* `LoRAEnabledPipeline._handle_lora_scale_updates()` routes scale updates into `LoRAManager.update_adapter_scales()`.
* `PeftLoRAStrategy.update_adapter_scales()` matches updates by:

  * `("adapter_name", adapter_name)` or
  * **`("path", path)`**
* and it’s **exact string match**, not file identity.

So the proposal’s “path string identity is critical” premise is correct.

### 3) The REST and CLI surfaces for styles exist, but they are session-dependent

* `video-cli style list/set/get` exists and hits `/api/v1/realtime/style/*`.
* Server endpoints:

  * `PUT /api/v1/realtime/style` forwards `_rcp_set_style`
  * `GET /api/v1/realtime/style/list` enumerates `fp.style_registry.list_styles()`

So the UX described (“`video-cli style set rat`”) is plausible *if* the underlying LoRA side is wired correctly (see below).

### 4) Docs already state the key constraint: runtime LoRA updates require `runtime_peft` at load time

`docs/api/parameters.md` explicitly says runtime `lora_scales` updates require `lora_merge_mode: "runtime_peft"` at load time. That’s consistent with how the LoRA strategies are designed.

---

## Where the proposal is **not yet true** in the current code

### A) `STYLE_SWAP_MODE=1` does nothing today (no code reads it)

The proposal’s central “Activation” section implies server behavior changes when `STYLE_SWAP_MODE=1`.

In the code you shared:

* `PipelineManager._apply_load_params()` **always** defaults `lora_merge_mode = "permanent_merge"` unless explicitly overridden in `load_params`.
* There is **no** conditional logic shown that checks `STYLE_SWAP_MODE` and forces `runtime_peft` or preloads style LoRAs.

So as-written, the proposal reads like a user-facing feature flag, but it’s not implemented as a flag anywhere in the provided code path.

### B) PipelineManager does **not** preload style LoRAs from manifests

The proposal’s “Pipeline load (PipelineManager)” behavior is the crux:

> Discover style manifests → extract LoRAs → preload all with runtime_peft → set scales 0 → merge with explicit load params…

Current reality from `PipelineManager._apply_load_params()`:

* It only uses what’s in `load_params`:

  * `loras = load_params.get("loras", None)`
  * `_lora_merge_mode = load_params.get("lora_merge_mode", "permanent_merge")`
* It does **not**:

  * scan `styles/*/manifest.yaml`
  * build a preload list
  * force runtime_peft
  * dedupe
  * skip missing

So the key enabling behavior described in the proposal is still missing.

**Implication:** Even though `FrameProcessor` emits `lora_scales` updates, the pipeline typically won’t have those adapters loaded (or won’t be in runtime-peft mode), so “style switch” will compile prompts but **not actually change LoRA**.

This aligns with `notes/issues/multi-lora-hot-switching.md` (“style switching compiles prompts correctly but LoRA doesn’t actually change; pipeline loads 1 LoRA with permanent_merge”).

### C) Style discovery is still hardcoded to `Path("styles")` and is not multi-dir / override-aware

The proposal says styles come from:

1. `./styles`
2. `~/.daydream-scope/styles` (user overrides)
3. env override `SCOPE_STYLES_DIRS`

Current code reality:

* `FrameProcessor.start()` (per your included excerpt description) loads manifests from **local `Path("styles")`**.
* `tests/test_style_integration.py` also assumes `Path("styles")`.

There is no `get_style_dirs()` helper yet, and nothing is described/implemented to load from `~/.daydream-scope/styles`.

So the proposal’s directory policy is aspirational.

### D) Manifest loading is not deterministic (unsorted `rglob`)

The proposal says:

> manifest file iteration must be sorted for determinism

Reality:

* `StyleRegistry.load_from_directory()` uses `for manifest_path in directory.rglob("manifest.yaml"):` with **no sort**.
* Default style selection is “first registered” (`_default_style` is set once), so nondeterministic file iteration can lead to nondeterministic default selection if you ever rely on it.

So this is not yet true.

### E) Canonical LoRA path handling is not implemented (and path mismatch can silently break switching)

The proposal requires canonicalization:

* expand `~`
* resolve relative paths (relative to models/lora or models root)
* `.resolve()`

Reality today:

* `styles/rat/manifest.yaml` uses:

  ```yaml
  lora_path: rat_21_step5500.safetensors
  ```

  i.e. **relative filename only**.
* `TemplateCompiler` and `FrameProcessor` pass `StyleManifest.lora_path` through **as-is** when building `lora_scales` updates.
* `PeftLoRAStrategy.load_adapters_from_list()` stores the `path` in `loaded_adapters` as the string it was given (no normalization shown at that layer), and scale updates match by string.

So unless the pipeline was loaded with **that exact same path string**, scale updates won’t match.

This is the single most important “proposal not yet true” gap because even if you implement preload, you must preload with the same canonical string you emit in updates — otherwise switching will log but do nothing.

### F) “Warn + skip missing LoRAs” is not how runtime_peft loading behaves today

The proposal says missing LoRAs should be skipped with a warning in style-swap discovery.

Reality:

* `PeftLoRAStrategy.load_adapters_from_list()` raises `RuntimeError` on `FileNotFoundError`.
* So if PipelineManager starts preloading from manifests without filtering, **one missing file bricks the entire pipeline load**.

So the proposal’s missing-file policy is not implemented and conflicts with current runtime-peft behavior.

### G) Deduping `lora_scales` by path is not implemented

Proposal wants a deduped, canonical list.

Reality:

* `FrameProcessor` generates `lora_updates` by iterating all styles and appending one entry per style manifest with a `lora_path`.
* If two styles share the same `lora_path`, you’ll emit duplicates.

Technically, `PeftLoRAStrategy.update_adapter_scales()` collapses duplicates into a dict while building `scale_map` (last value wins), so this is mostly “noisy not fatal,” but it’s still “proposal not yet true.”

### H) `STYLE_DEFAULT` / initial style semantics are not implemented

Proposal says optional env var `STYLE_DEFAULT=<name>` selects initial style, else none active.

Reality:

* `FrameProcessor` starts with `style_manifest = None`.
* There’s no “set default style on connect” or “send initial lora_scales automatically” behavior shown.
* Style activation is edge-triggered by `_rcp_set_style` only.

So “initial style semantics” described by the proposal are not yet real.

---

## Mismatches / missing details vs API docs

### 1) Docs cover runtime LoRA updates, but not style swap mode or style endpoints

* `docs/api/parameters.md` documents `lora_scales` updates and states the runtime_peft requirement. Good.
* `docs/api/load.md` documents `loras` and `lora_merge_mode`, defaulting to permanent_merge. Good.

But neither doc currently explains:

* `/api/v1/realtime/style` endpoints
* style manifests / styles directory behavior
* how style switching interacts with LoRA loading
* a “STYLE_SWAP_MODE” feature flag

If `STYLE_SWAP_MODE` becomes real, you’ll need doc updates so users understand why FPS drops (runtime_peft), and what the server will preload automatically.

### 2) Per-LoRA merge_mode exists in schema/code, but docs only describe global `lora_merge_mode`

In server schema, `LoRAConfig` includes `merge_mode`. In `LoRAManager.load_adapters_from_list()`, per-LoRA merge_mode is supported.

But `docs/api/load.md` only documents:

* `loras[].path`
* `loras[].scale`
* global `lora_merge_mode`

So the proposal’s mention of “per-LoRA merge_mode explicit” is not aligned with docs today. (It’s feasible in code, but currently undocumented.)

---

## Additional sharp edges the proposal still doesn’t fully cover

These are places where the proposal might be “conceptually right,” but the current code behavior creates user-visible gotchas unless the doc or implementation addresses them.

### 1) Style endpoints are tied to an “active WebRTC session”

Both `/api/v1/realtime/style` and `/api/v1/realtime/style/list` operate by:

* finding an active session (`get_active_session(webrtc_manager)`)
* applying a control message into that session’s `FrameProcessor`

So “style swap mode” isn’t a “server setting” that works pre-connection; it’s per active session/video track.

The proposal doesn’t mention:

* what happens with 0 sessions (API errors)
* what happens with >1 session (which one is “active”?)

### 2) `PUT /api/v1/realtime/style` does not validate the style exists

Today, `set_active_style` just forwards `_rcp_set_style`. In `FrameProcessor`, if style isn’t found, it logs a warning but otherwise does nothing.

Net effect:

* the API/CLI can report success (“style_set”) even though nothing changed.
* That’s a UX mismatch with the proposal’s “immediately changes style” expectation.

### 3) Instruction sheet loading also assumes repo-local `styles/<name>/instructions.md`

`prompt_compiler._load_instruction_sheet()` builds the style directory as repo-root `/styles/<style.name>` via `Path(__file__).parent.../styles/style.name`.

If you do implement multi-dir styles (`~/.daydream-scope/styles`), instruction sheet lookup must also follow the same directory resolution rules, or LLM compiler behavior will be inconsistent across install locations.

The proposal’s “shared style dirs” checklist doesn’t mention this extra place.

### 4) Cache semantics are still “best effort”

The proposal notes:

* “LoRA scale updates trigger a cache reset inside WAN pipelines when manage_cache=True”
* “manage_cache should remain enabled”

But in current FrameProcessor style switching:

* it does not force `reset_cache=True`
* it does not force `manage_cache=True`

So “clean style switches” are dependent on downstream pipeline behavior + the client not disabling manage_cache.

If you want the proposal to be reliably true, you either need:

* documented requirement (“don’t disable manage_cache”), and/or
* explicit server behavior (set reset_cache on style change, or always force init_cache when LoRA updates present regardless of manage_cache)

---

## Practical bottom line

`notes/proposals/style-swap-mode.md` is a good *spec* for what you want, but today:

* The **style layer** (manifests, registry, `_rcp_set_style`, prompt recompile, one-shot `lora_scales`) is implemented.
* The **pipeline-side enabling piece** (preload all style LoRAs + force runtime_peft + canonical path identity + missing file policy) is **not implemented**.
* The **path identity requirement** is currently a real footgun because manifests and pipeline load paths are not canonicalized.

So in current code, “style set” mostly changes prompts (and logs a “LoRA switch” update), but the LoRA effect will **only** change if:

1. the pipeline was loaded with `lora_merge_mode="runtime_peft"`, **and**
2. all style LoRAs you intend to switch between were loaded into that pipeline, **and**
3. the `path` strings match exactly between:

   * the loaded adapters’ stored `"path"` values and
   * the emitted `lora_scales[].path` strings.

---

## If you want a single “proposal vs reality” checklist

### Implemented (true today)

* ✅ `_rcp_set_style` triggers prompt compilation and edge-triggered `lora_scales` emission
* ✅ Runtime LoRA scale updates exist and are keyed by `path` string identity
* ✅ CLI + REST endpoints exist for style list/set (session-dependent)

### Not implemented / mismatched (not true today)

* ❌ `STYLE_SWAP_MODE` flag (no behavioral change)
* ❌ PipelineManager manifest-driven style LoRA preload
* ❌ Forcing `runtime_peft` in a mode
* ❌ Multi-dir style discovery / override precedence
* ❌ Sorted manifest loading for determinism
* ❌ Canonical LoRA path normalization used consistently (load + emit)
* ❌ “Warn + skip missing LoRAs” policy
* ❌ Deduped `lora_scales` list
* ❌ `STYLE_DEFAULT` initial activation semantics

### Missing doc/UX details

* ⚠️ Style endpoints require an active WebRTC session; “active session” ambiguity
* ⚠️ Style set API does not validate style existence (can succeed but no-op)
* ⚠️ Instruction sheet lookup must follow style dirs once multi-dir exists
* ⚠️ Cache reset semantics aren’t enforced on style switch

---

If you want, I can also turn this into a compact “status annotations” patch on the proposal itself (e.g., add an “As of 2025-12-27: NOT IMPLEMENTED YET” section with links to the exact functions/places that still need code), but the list above is the concrete delta between spec and code.
