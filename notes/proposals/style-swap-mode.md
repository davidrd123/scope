# Style Swap Mode

> Status: Draft (reviewed against codebase)
> Date: 2025-12-26, reviewed 2025-12-27
> Reviews:
> - `notes/proposals/style-swap-mode/oai_5pro01.md`
> - `notes/proposals/style-swap-mode/oai_5pro02.md`

## Implementation Status

### What's working today (✅)

| Feature | Status |
|---------|--------|
| `_rcp_set_style` triggers prompt recompile + one-shot `lora_scales` | ✅ Implemented |
| Runtime LoRA scale updates keyed by `path` string identity | ✅ Implemented |
| CLI + REST endpoints for style list/set | ✅ Implemented (session-dependent) |
| `runtime_peft` merge mode supports scale updates | ✅ Implemented |
| Multi-dir style discovery (`./styles` + `~/.daydream-scope/styles`) | ✅ Implemented |
| Sorted manifest loading (deterministic) | ✅ Implemented |
| Canonical LoRA path normalization (for style swap) | ✅ Implemented |
| Instruction sheet lookup follows style dirs | ✅ Implemented |
| `STYLE_SWAP_MODE=1` manifest-driven LoRA preload + `runtime_peft` forcing | ✅ Implemented |

### What's NOT implemented yet (❌)

| Feature | Gap |
|---------|-----|
| `STYLE_DEFAULT` initial activation | No initial style on connect |

### Current behavior

Today, `video-cli style set <name>` will:
- ✅ Recompile prompts for the new style
- ✅ Emit `lora_scales` update
- ❌ **NOT actually change the LoRA** unless:
  1. Pipeline was loaded with `lora_merge_mode="runtime_peft"`, AND
  2. All style LoRAs were preloaded into the pipeline, AND
  3. The `path` strings match exactly (load vs emit)

This is documented in `notes/issues/multi-lora-hot-switching.md`.

---

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
```

Optional:

- `STYLE_DEFAULT=<name>`: choose initial active style (otherwise: no style active until set)
- `SCOPE_STYLES_DIRS=/path/a:/path/b`: override style manifest search paths
- `SCOPE_PRELOAD_LORAS=rat,tmnt,...`: restrict preload set (advanced / optional)

## Style discovery and directory policy

We must keep `FrameProcessor` style listing and pipeline preload discovery in sync.

Default style dirs (in order):

1. `./styles` (repo/dev built-ins)
2. `~/.daydream-scope/styles` (user overrides)

If a style name exists in multiple dirs, the later dir wins.

Implementation detail: manifest file iteration must be sorted for determinism.

## Critical requirement: canonical LoRA paths

Runtime updates match adapters by the *exact `path` string*.

Therefore we must canonicalize LoRA paths once and use the canonical string everywhere:

- Pipeline preload `loras=[{"path": <canonical>, ...}]`
- FrameProcessor `lora_scales=[{"path": <canonical>, ...}]`

Canonicalization rules:

- expand `~`
- resolve relative paths against `models/lora` (or configured models root)
- call `.resolve()` to normalize

If this is not done, style switching may log success but have no effect.

## Behavior

When `STYLE_SWAP_MODE=1`:

### Pipeline load (PipelineManager)

1. Discover style manifests (same dirs as FrameProcessor)
2. Extract all unique LoRA paths from manifests (canonicalize + dedupe)
3. Filter missing files (warn + skip; do not hard-fail the pipeline by default)
4. Force `_lora_merge_mode = "runtime_peft"`
5. Preload all discovered LoRAs:
   - default scale: 0.0
   - per-LoRA merge_mode: "runtime_peft" (explicit)
6. Merge with any explicitly requested LoRAs from load params:
   - dedupe by canonical path
   - keep explicit per-LoRA merge_mode if provided

### Runtime style switching (FrameProcessor)

On `_rcp_set_style`:

- Recompile prompts for the selected style
- Emit `lora_scales` updates (one-shot) that:
  - set the selected style's LoRA path to `lora_default_scale`
  - set all other *style* LoRA paths to 0.0
  - dedupe updates by path

Cache semantics:

- LoRA scale updates trigger a cache reset inside WAN pipelines when `manage_cache=True`
- For clean style switches, `manage_cache` should remain enabled

## Failure modes and logging

**Desired behavior:**

- If `STYLE_SWAP_MODE=1` but no valid style LoRAs are discovered:
  - pipeline loads normally (no-op)
  - log an INFO explaining why (no styles dir / no manifests / all missing LoRAs)
- If a style manifest references a missing LoRA:
  - warn and skip preloading that LoRA
  - style switching will still compile prompts, but LoRA effect won't change

**Current behavior:**

- `PeftLoRAStrategy.load_adapters_from_list()` still raises `RuntimeError` on `FileNotFoundError` if a missing path reaches it.
- In `STYLE_SWAP_MODE=1`, PipelineManager now filters missing LoRAs (warn + skip) before loading adapters, so style-swap preload won't brick the load.
- Outside style-swap mode, a missing explicitly requested LoRA can still fail pipeline load.

Suggested logs:

- Discovered styles + dirs used
- Number of unique LoRAs preloaded + how many skipped
- On style set: active style name, active path, scale, number of paths zeroed

## Tradeoffs

| Mode                                 | FPS   | Runtime Switching |
| ------------------------------------ | ----- | ----------------- |
| `permanent_merge` (default)          | ~100% | No                |
| `STYLE_SWAP_MODE=1` (`runtime_peft`) | ~50%  | Yes               |

## Implementation checklist

### 1. Shared style directory resolution

Add `get_style_dirs()` to `scope/realtime/style_manifest.py`:

```python
def get_style_dirs() -> list[Path]:
    """Return style directories in precedence order (later wins)."""
    if custom := os.environ.get("SCOPE_STYLES_DIRS"):
        return [Path(p).expanduser() for p in custom.split(":")]
    return [
        Path("styles"),  # repo/dev built-ins
        Path.home() / ".daydream-scope" / "styles",  # user overrides
    ]
```

Use in both:
- `FrameProcessor.start()` (instead of hardcoded `Path("styles")`)
- `PipelineManager` style-swap discovery

### 2. Path canonicalization helper

```python
def canonicalize_lora_path(raw: str, models_root: Path | None = None) -> str:
    """Canonicalize LoRA path for consistent matching."""
    p = Path(raw).expanduser()
    if not p.is_absolute() and models_root:
        p = models_root / p
    return str(p.resolve())
```

### 3. Sort manifest paths in StyleRegistry

In `StyleRegistry.load_from_directory()`:
```python
for manifest_path in sorted(directory.rglob("manifest.yaml")):
```

### 4. Deduplicate lora_scales in FrameProcessor

Build `dict[path, scale]` then convert to list:
```python
scales_by_path = {}
for style_name in self.style_registry.list_styles():
    manifest = self.style_registry.get(style_name)
    if manifest and manifest.lora_path:
        canonical = canonicalize_lora_path(manifest.lora_path)
        scale = manifest.lora_default_scale if style_name == active else 0.0
        scales_by_path[canonical] = scale

lora_scales = [{"path": p, "scale": s} for p, s in scales_by_path.items()]
```

### 5. Skip missing LoRAs with warning

In discovery:
```python
canonical = canonicalize_lora_path(manifest.lora_path)
if not Path(canonical).exists():
    logger.warning(f"Style '{manifest.name}': LoRA not found at {canonical}, skipping")
    continue
```

## Testing plan

1. Unit test: style discovery returns deterministic ordering and deduped LoRA list
2. Unit test: path canonicalization yields identical strings for preload + lora_scales
3. Integration test: with style swap enabled, switching styles emits `lora_scales` once and does not resend on same-style set
4. Negative test: missing LoRA file is skipped and does not crash pipeline load

## Sharp edges / UX gaps (from review)

### Style endpoints require active WebRTC session

`/api/v1/realtime/style/*` endpoints call `get_active_session(webrtc_manager)`:
- With 0 sessions: API errors
- With >1 sessions: ambiguous which is "active"

Style swap is not a "server setting" — it's per active session/video track.

### Style set API doesn't validate style exists

`PUT /api/v1/realtime/style` forwards `_rcp_set_style` without checking if the style is registered. If style isn't found:
- FrameProcessor logs warning but does nothing
- API returns success ("style_set") even though nothing changed

**UX mismatch:** user thinks style changed, but it didn't.

### Instruction sheet lookup must follow style dirs

`prompt_compiler._load_instruction_sheet()` is hardcoded to `<repo-root>/styles/<style.name>/instructions.md` (code-location-relative).

If multi-dir style discovery is implemented, instruction sheet lookup must follow the same resolution rules, or LLM compiler behavior will be inconsistent.

Additional sharp edges:
- `StyleManifest.instruction_sheet_path` exists but is currently ignored by `_load_instruction_sheet()`.
- Style manifest discovery is currently CWD-relative (`Path("styles")`) while instruction sheet lookup is code-location-relative; this can diverge if Scope is run from outside the repo root.

### Cache reset semantics not enforced

The proposal says LoRA scale updates trigger cache reset when `manage_cache=True`. But:
- FrameProcessor style switching does NOT force `reset_cache=True`
- FrameProcessor does NOT force `manage_cache=True`

Clean style switches depend on:
- Downstream pipeline behavior
- Client not disabling `manage_cache`

**Options to fix:**
1. Document requirement ("don't disable manage_cache")
2. Force `reset_cache=True` on style change
3. Force `init_cache=True` when LoRA updates present

### Selected style without `lora_path` can produce misleading logging

If the selected style has `lora_path = null`, the current `_rcp_set_style` logic can still zero out other styles’ LoRAs and log something like `LoRA switch: None @ 0.85, zeroed …` even though no LoRA was activated.

Mitigation: treat `active_lora_path is None` as “no LoRA activation”; only emit `lora_scales` when an active LoRA path exists (or make the log explicit: “zeroed others, no active LoRA”).

## Documentation gap (worth fixing if we implement this)

Per-LoRA `merge_mode` exists in code (`server/schema.py` supports `loras[].merge_mode`) but is not documented in `docs/api/load.md`. This matters for style swap mode because manifest-driven preload may want to stamp `merge_mode: "runtime_peft"` per adapter.

---

## Related files

- `src/scope/server/pipeline_manager.py` - Pipeline load params
- `src/scope/realtime/style_manifest.py` - StyleRegistry and manifest loading
- `src/scope/server/frame_processor.py` - Style switching logic (`_rcp_set_style`)
- `src/scope/core/pipelines/wan2_1/lora/manager.py` - LoRA merge strategies
- `src/scope/realtime/prompt_compiler.py` - Instruction sheet loading
