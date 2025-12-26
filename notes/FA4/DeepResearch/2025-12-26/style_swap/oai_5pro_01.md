Below is a “red pen” review of `notes/proposals/style-swap-mode.md`, focused on *alignment with the current implementation* (FrameProcessor-driven style switching + PipelineManager-driven LoRA loading) and the *edge cases that will bite you in practice*. I’ve also included a hardened, drop-in rewrite of the proposal (with a couple of concrete code-shape tweaks) so you can paste it back into the repo.

---

## Key alignment gaps with current implementation

### 1) Style discovery directory mismatch (`~/.daydream-scope/styles` vs `./styles`)

**Proposal:** scans `~/.daydream-scope/styles/`
**Reality:** `FrameProcessor.start()` loads manifests from **local `styles/`** (relative path), and tests assume that too (`tests/test_style_integration.py` uses `Path("styles")`).

**Impact:** You can end up with:

* Pipeline preloads LoRAs from `~/.daydream-scope/styles`
* FrameProcessor lists styles from `./styles`
* `video-cli style set rat` compiles prompts correctly but sends `lora_scales` for a LoRA path the pipeline never loaded (or with a different string path) → *no visible style change*.

**Hardening fix:** Define a single “styles dirs” resolution function and use it in **both** places:

* FrameProcessor manifest loading
* PipelineManager style-swap LoRA discovery

Suggested default: `["./styles", "~/.daydream-scope/styles"]` with clear precedence rules (user dir overrides built-ins).

---

### 2) Path string identity is *critical* for runtime LoRA scale updates

Runtime updates are keyed by `path`, and the code compares **strings**, not file identity:

* `FrameProcessor` emits `{"path": manifest.lora_path, "scale": ...}`
* `LoRAManager.update_adapter_scales()` matches updates against `loaded_adapters[*]["path"]`
* `PeftLoRAStrategy.update_adapter_scales()` uses `("path", path)` as the lookup key

**If PipelineManager loads `/abs/models/lora/rat.safetensors` but the manifest contains `models/lora/rat.safetensors`**, the update won’t match and the scale won’t change.

**Hardening fix:** Canonicalize paths *once* and use the canonical form everywhere:

* Expand `~`
* Resolve relative paths (ideally relative to `models/lora` or a known models root)
* Use `Path(...).resolve()` to normalize

And make sure the canonical path is what:

* gets passed into pipeline load `loras=[{"path": canonical, ...}]`
* gets stored/used by the style manifest registry (or at least emitted by FrameProcessor when building `lora_scales`)

This is the single most common “it logs LoRA switch but nothing changes” failure mode.

---

### 3) Missing LoRA file in *any* discovered manifest can hard-fail pipeline load

`PeftLoRAStrategy.load_adapters_from_list()` raises on `FileNotFoundError` and turns it into a `RuntimeError`. So if style-swap discovery blindly adds every `manifest.lora_path`, one bad manifest bricks style-swap mode.

**Hardening fix (policy decision needed):**

* **Best-effort mode (recommended):** skip missing paths with a warning and keep the server usable.
* **Strict mode:** fail fast, but then document it loudly and provide debugging output listing the missing paths.

Right now the proposal is implicitly “strict” but doesn’t say so, and will surprise you.

---

### 4) “Active style at default scale on load” is not implemented today

Current behavior:

* LoRA scale changes are **edge-triggered** on `_rcp_set_style` in `FrameProcessor`
* Nothing sets an initial style automatically unless the client does it

So the proposal’s “active style at its default scale, others at 0.0” requires **one** of:

* Pipeline load sets initial scales (in `loras` list) for a chosen default style, OR
* FrameProcessor sends an initialization `lora_scales` update after loading manifests, OR
* Frontend/CLI always calls `style set <default>` after connect

**Hardening fix:** Explicitly define “initial style semantics” (and where it’s applied). Otherwise you’ll have inconsistent “sometimes starts styled, sometimes starts base model”.

---

### 5) Duplicate `lora_path` across styles → duplicate `lora_scales` entries

You already noted this ambiguity. Current FrameProcessor logic emits one update per style, not per unique LoRA path.

This is usually “harmless but noisy” because duplicates tend to have the same scale, but it has two costs:

* More data sent / processed
* Harder debugging (“why do I see the same path 6 times?”)

**Hardening fix:** Deduplicate updates by canonical `path` before enqueueing `lora_scales`.

---

### 6) Determinism: manifest load order is not guaranteed

`StyleRegistry.load_from_directory()` uses `directory.rglob("manifest.yaml")` without sorting. That can vary by filesystem/OS, which affects:

* which style becomes default (`_default_style` is the first registered)
* order returned by `list_styles()`

If style-swap mode relies on “default style” for initial activation, this becomes nondeterministic.

**Hardening fix:** Sort manifest paths before loading.

---

### 7) Cache semantics: style switch triggers `init_cache` only if `manage_cache` is True

Good news: `KreaRealtimeVideoPipeline._generate()` already flips `init_cache=True` when `lora_scales` is present and `manage_cache` is enabled (default True).

But if a client sets `manage_cache=false`, style switches may look glitchy because you’re applying a different adapter against a KV cache built under the old style.

**Hardening fix:** In style-swap mode, consider enforcing a cache reset on style change (either by:

* also sending `reset_cache=True` from FrameProcessor on `_rcp_set_style`, or
* ignoring `manage_cache` for LoRA scale updates and always forcing `init_cache` on LoRA updates)

At minimum, the proposal should document that “manage_cache must remain enabled for clean style switches”.

---

## Recommended hardening changes to the proposal

### A) Unify style directory resolution

Add one function used by both pipeline loading and FrameProcessor. Example policy:

* Default search path list:

  1. `Path("styles")` (repo/dev + packaged built-ins if present)
  2. `Path.home()/".daydream-scope"/"styles"` (user overrides / user-installed styles)
* Precedence: later dirs override earlier for **same style name**.

Also expose override env var (optional): `SCOPE_STYLES_DIRS=/path/a:/path/b`

### B) Canonicalize LoRA paths and use the canonical string everywhere

Decide how to interpret manifest `lora_path`:

* If absolute → use it (after `expanduser` + `resolve`)
* If relative → resolve relative to `models/lora` (or some configured model root)

Then:

* PipelineManager loads adapters using canonical paths
* FrameProcessor emits canonical paths for `lora_scales`

### C) Best-effort missing-file handling (don’t brick the whole mode)

Skip missing LoRAs with a warning, and (ideally) return a list of “styles with missing lora” in `/api/v1/realtime/style/list` so the UI can show them as disabled.

### D) Deduplicate by path

Both in “preload list” and in `lora_scales` updates.

### E) Define initial style semantics

Pick one and document it:

* “Starts with all style LoRAs at 0.0; first `style set` activates.”
* or “Starts with default style active unless `STYLE_DEFAULT` is set.”

---

## Drop-in hardened rewrite of `notes/proposals/style-swap-mode.md`

You can paste this over the existing file; it keeps the spirit but aligns with the actual wiring and closes the sharp edges above.

````md
# Style Swap Mode

> Status: Draft (hardened)
> Date: 2025-12-26

## Summary

Enable instant style switching via `video-cli style set <name>` by:

1. Preloading all style LoRAs at pipeline load time
2. Forcing LoRA merge strategy to `runtime_peft` (required for runtime scale updates)
3. Switching styles by emitting `lora_scales` updates (active LoRA at default scale, others at 0.0)

This aligns with the current architecture:
- Styles live in `FrameProcessor` (StyleRegistry + `_rcp_set_style` → `lora_scales`)
- LoRAs are loaded at pipeline init via `PipelineManager` load params (`loras`, `_lora_merge_mode`)
- Runtime LoRA scale updates flow through:
  FrameProcessor → pipeline call (`lora_scales`) → LoRAEnabledPipeline._handle_lora_scale_updates → LoRAManager → PEFT scaling

## Why this is needed

Runtime LoRA switching only works if:
1. Pipeline was loaded with `lora_merge_mode: runtime_peft`
2. All target LoRAs are already loaded into the pipeline

Default is `permanent_merge` for FPS, which bakes the LoRA into the weights and prevents runtime changes.

## Goals

- `video-cli style set rat` immediately changes the active LoRA effect without restarting the server
- Avoid pipeline reloads during creative iteration
- Keep behavior robust across dev (`./styles`) and user-installed styles (`~/.daydream-scope/styles`)

## Non-goals

- Hot-loading brand-new LoRA files into an already-running pipeline (future work)
- Seamless visual blending between two LoRAs (future work; prompt transitions exist separately)

## Activation

Set env var at server start:

```bash
STYLE_SWAP_MODE=1 uv run python -m scope.server
````

Optional:

* `STYLE_DEFAULT=<name>`: choose initial active style (otherwise: no style active until set)
* `SCOPE_STYLES_DIRS=/path/a:/path/b`: override style manifest search paths
* `SCOPE_PRELOAD_LORAS=rat,tmnt,...`: restrict preload set (advanced / optional)

## Style discovery and directory policy

We must keep `FrameProcessor` style listing and pipeline preload discovery in sync.

Default style dirs (in order):

1. `./styles` (repo/dev built-ins)
2. `~/.daydream-scope/styles` (user overrides)

If a style name exists in multiple dirs, the later dir wins.

Implementation detail: manifest file iteration should be sorted for determinism.

## Critical requirement: canonical LoRA paths

Runtime updates match adapters by the *exact `path` string*.

Therefore we must canonicalize LoRA paths once and use the canonical string everywhere:

* Pipeline preload `loras=[{"path": <canonical>, ...}]`
* FrameProcessor `lora_scales=[{"path": <canonical>, ...}]`

Canonicalization rules:

* expand `~`
* resolve relative paths against `models/lora` (or configured models root)
* call `.resolve()` to normalize

If this is not done, style switching may log success but have no effect.

## Behavior

When `STYLE_SWAP_MODE=1`:

### Pipeline load (PipelineManager)

1. Discover style manifests (same dirs as FrameProcessor)
2. Extract all unique LoRA paths from manifests (canonicalize + dedupe)
3. Filter missing files (warn + skip; do not hard-fail the pipeline by default)
4. Force `_lora_merge_mode = "runtime_peft"`
5. Preload all discovered LoRAs:

   * default scale: 0.0
   * per-LoRA merge_mode: "runtime_peft" (explicit)
6. Merge with any explicitly requested LoRAs from load params:

   * dedupe by canonical path
   * keep explicit per-LoRA merge_mode if provided

### Runtime style switching (FrameProcessor)

On `_rcp_set_style`:

* Recompile prompts for the selected style
* Emit `lora_scales` updates (one-shot) that:

  * set the selected style’s LoRA path to `lora_default_scale`
  * set all other *style* LoRA paths to 0.0
  * dedupe updates by path

Cache semantics:

* LoRA scale updates trigger a cache reset inside WAN pipelines when `manage_cache=True`
* For clean style switches, `manage_cache` should remain enabled

## Failure modes and logging

* If `STYLE_SWAP_MODE=1` but no valid style LoRAs are discovered:

  * pipeline loads normally (no-op)
  * log an INFO explaining why (no styles dir / no manifests / all missing LoRAs)
* If a style manifest references a missing LoRA:

  * warn and skip preloading that LoRA
  * style switching will still compile prompts, but LoRA effect won’t change

Suggested logs:

* Discovered styles + dirs used
* Number of unique LoRAs preloaded + how many skipped
* On style set: active style name, active path, scale, number of paths zeroed

## Tradeoffs

| Mode                                 | FPS   | Runtime Switching |
| ------------------------------------ | ----- | ----------------- |
| `permanent_merge` (default)          | ~100% | No                |
| `STYLE_SWAP_MODE=1` (`runtime_peft`) | ~50%  | Yes               |

## Testing plan

1. Unit test: style discovery returns deterministic ordering and deduped LoRA list
2. Unit test: path canonicalization yields identical strings for preload + lora_scales
3. Integration test: with style swap enabled, switching styles emits `lora_scales` once and does not resend on same-style set
4. Negative test: missing LoRA file is skipped and does not crash pipeline load

```

---

## Extra concrete code-level recommendations (to make the doc “true”)

These are the minimum “make it real” hardening points implied by the rewrite:

1) **Sort manifest paths**
- In `StyleRegistry.load_from_directory()`: iterate `sorted(directory.rglob("manifest.yaml"))`

2) **Shared style dirs function**
- Put it in `scope/realtime/style_manifest.py` (or a small new module) and call it from:
  - `FrameProcessor.start()` (instead of `Path("styles")`)
  - `PipelineManager` style-swap handling

3) **Canonicalize LoRA paths**
- Create a helper: `canonicalize_lora_path(raw: str, models_dir: Path) -> str`
- Ensure FrameProcessor emits canonical paths (not raw manifest values)

4) **Deduplicate `lora_scales` by path in FrameProcessor**
- Build a dict `path -> scale` then convert to list once

5) **Missing LoRA policy**
- In discovery, check `Path(canonical).exists()`. Skip + warn (unless you intentionally want strict mode).

---

If you want, I can also propose an *exact* patch shape for:
- `StyleRegistry.load_from_directory` (sorting + multi-dir load + override behavior)
- a `discover_style_loras()` that returns `(loras, style_name_to_lora_path)` using canonical paths
- the smallest safe change to `FrameProcessor` to emit canonical/deduped `lora_scales`

…but the big win for this review is: **make directory + path canonicalization explicit**. Without that, style swap mode will be “works on my machine” and then silently no-op for everyone else.
```
