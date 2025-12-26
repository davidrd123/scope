# Style Swap Mode

> Status: Draft
> Date: 2025-12-26

## Problem

LoRA switching at runtime only works if:
1. Pipeline was loaded with `lora_merge_mode: "runtime_peft"`
2. All target LoRAs were pre-loaded at pipeline startup

By default, we use `permanent_merge` for max FPS, which bakes the LoRA into the model - no runtime switching possible.

## Solution

Environment flag `STYLE_SWAP_MODE=1` that:
1. At pipeline load time, scans `~/.daydream-scope/styles/` for all style manifests
2. Collects all unique LoRA paths from those manifests
3. Forces `lora_merge_mode: "runtime_peft"`
4. Pre-loads all LoRAs (active style at its default scale, others at 0.0)

Then `video-cli style set <name>` works instantly.

## Usage

```bash
# Start server with style swap enabled
STYLE_SWAP_MODE=1 uv run python -m scope.server

# Or export for session
export STYLE_SWAP_MODE=1
uv run python -m scope.server
```

Then in CLI:
```bash
video-cli style list          # Shows all available styles
video-cli style set hidari    # Instant switch
video-cli style set rat       # Instant switch
video-cli style set akira     # Instant switch
```

## Implementation

### 1. Style Discovery Function

```python
# src/scope/realtime/style_manifest.py

def discover_style_loras(styles_dir: Path | None = None) -> list[dict]:
    """Scan styles directory and return all LoRA configs.

    Returns list of {"path": "...", "scale": 0.0} for each unique LoRA.
    Can be called before pipeline load (no FrameProcessor needed).
    """
    if styles_dir is None:
        styles_dir = Path.home() / ".daydream-scope" / "styles"

    if not styles_dir.exists():
        return []

    loras = {}  # path -> manifest (to dedupe)
    for manifest_path in styles_dir.rglob("manifest.yaml"):
        try:
            manifest = StyleManifest.from_yaml(manifest_path)
            if manifest.lora_path and manifest.lora_path not in loras:
                loras[manifest.lora_path] = manifest
        except Exception:
            continue

    return [
        {"path": path, "scale": 0.0, "merge_mode": "runtime_peft"}
        for path in loras.keys()
    ]
```

### 2. Pipeline Manager Integration

```python
# src/scope/server/pipeline_manager.py

import os
from scope.realtime.style_manifest import discover_style_loras

class PipelineManager:
    def _apply_common_load_params(self, config, load_params, ...):
        # ... existing code ...

        # Style swap mode: override with runtime_peft + all style LoRAs
        if os.environ.get("STYLE_SWAP_MODE", "").lower() in ("1", "true", "yes"):
            style_loras = discover_style_loras()
            if style_loras:
                logger.info(
                    f"STYLE_SWAP_MODE: Loading {len(style_loras)} style LoRAs "
                    f"with runtime_peft mode"
                )
                lora_merge_mode = "runtime_peft"
                # Merge with any explicitly requested LoRAs
                existing_paths = {l.get("path") for l in (loras or [])}
                for sl in style_loras:
                    if sl["path"] not in existing_paths:
                        loras = loras or []
                        loras.append(sl)

        # ... rest of existing code ...
```

### 3. Set Active Style on Load

When pipeline loads with style swap mode, we should also set the initial active style's LoRA to its default scale (not 0.0):

```python
# In FrameProcessor initialization or first style set

def _initialize_style_swap_scales(self):
    """Set the default style's LoRA to active scale, others to 0."""
    if not os.environ.get("STYLE_SWAP_MODE"):
        return

    default_style = self.style_registry.get_default()
    if default_style and default_style.lora_path:
        # Build scale updates: active at default_scale, others at 0
        lora_updates = []
        for style_name in self.style_registry.list_styles():
            manifest = self.style_registry.get(style_name)
            if manifest and manifest.lora_path:
                scale = manifest.lora_default_scale if manifest == default_style else 0.0
                lora_updates.append({"path": manifest.lora_path, "scale": scale})

        if lora_updates:
            self.parameters["lora_scales"] = lora_updates
```

## Tradeoffs

| Mode | FPS | Runtime Switching |
|------|-----|-------------------|
| Default (`permanent_merge`) | ~100% | No - reload required |
| `STYLE_SWAP_MODE=1` (`runtime_peft`) | ~50% | Yes - instant |

## Logging

When `STYLE_SWAP_MODE=1`:
```
INFO: STYLE_SWAP_MODE enabled
INFO: Discovered 6 style LoRAs: hidari, akira, rat, graffito, kaiju, tmnt
INFO: Loading with runtime_peft mode for instant switching
INFO: Active style: hidari @ 1.0, others @ 0.0
```

When switching:
```
INFO: Style switch: hidari → rat
INFO: LoRA scales: rat @ 1.0, hidari @ 0.0, akira @ 0.0, ...
```

## Future Enhancements

1. **UI toggle** - Settings panel option to enable style swap mode (requires pipeline reload)
2. **Partial preload** - Only preload a subset of styles to reduce VRAM
3. **Hot-add styles** - Load new LoRAs at runtime without preloading (if PEFT supports it)

## Related

- `src/scope/server/pipeline_manager.py` - Pipeline load params
- `src/scope/realtime/style_manifest.py` - StyleRegistry and manifest loading
- `src/scope/server/frame_processor.py` - Style switching logic
- `src/scope/core/pipelines/wan2_1/lora/manager.py` - LoRA merge strategies
