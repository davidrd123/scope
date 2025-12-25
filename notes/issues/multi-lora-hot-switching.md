# Issue: Multi-LoRA Hot-Switching

**Status**: Planned
**Priority**: Medium
**Blocks**: Full style switching experience

---

## Problem

Currently, the pipeline loads a single LoRA at startup (selected via GUI) using `permanent_merge` mode. This means:

1. Only one LoRA is available per session
2. Style switching via CLI/API compiles prompts correctly but **LoRA doesn't actually change**
3. User must restart the server to switch LoRAs

## Goal

Enable switching between multiple pre-loaded LoRAs at runtime by adjusting their scales (active=0.85, inactive=0.0).

---

## Current State

```
GUI selects LoRA → Pipeline loads 1 LoRA with permanent_merge → Cannot adjust at runtime
```

Log evidence:
```
Loading 1 LoRA(s) with permanent_merge strategy
```

## Desired State

```
Config specifies LoRAs → Pipeline loads N LoRAs with runtime_peft → Scale adjustable via API
```

---

## Affected Components

| Component | Change Needed |
|-----------|---------------|
| `pipeline_manager.py` | Load multiple LoRAs from config/manifests |
| `schema.py` | Possibly extend LoadParams |
| GUI (frontend) | Either lock LoRA selector or make it multi-select |
| `frame_processor.py` | Already done - zeros out inactive LoRAs on style switch |
| Style manifests | Already done - have `lora_path` fields |

---

## Design Options

### Option A: Environment Variable Override

```bash
SCOPE_PRELOAD_LORAS=rat,tmnt,yeti,hidari uv run python -m scope.server.app
```

- Pipeline manager reads env var
- Loads all specified LoRAs with `runtime_peft`
- GUI LoRA selector disabled/hidden when env var set
- **Pro**: Simple, explicit, doesn't break existing workflow
- **Con**: Requires restart to change LoRA set

### Option B: Auto-Load from Style Manifests

```python
# In pipeline_manager.py or frame_processor.py
def get_loras_from_styles():
    registry = StyleRegistry()
    registry.load_from_directory(Path("styles"))
    return [m.lora_path for m in registry._manifests.values() if m.lora_path]
```

- Server scans `styles/*/manifest.yaml` at startup
- Collects all unique `lora_path` entries
- Loads them all with `runtime_peft`
- **Pro**: Automatic, style-driven
- **Con**: May load LoRAs user doesn't want; slower startup

### Option C: Server Config File

```yaml
# config/loras.yaml
preload:
  - path: rat_21_step5500.safetensors
    scale: 0.0
  - path: tmnt_21_step9500.safetensors
    scale: 0.0
  - path: yeti_10bucket_v3_14b_8500steps.safetensors
    scale: 0.0
  - path: hidari_v3_14b_step8250.safetensors
    scale: 0.0
merge_mode: runtime_peft
```

- Explicit config file
- GUI reads config and shows which LoRAs are available
- **Pro**: Clear, version-controllable
- **Con**: Another config file to manage

### Option D: GUI Multi-Select

- Modify frontend to allow selecting multiple LoRAs
- Send list to backend on pipeline load
- **Pro**: User controls exactly what's loaded
- **Con**: Requires frontend changes; more complex UX

---

## Recommended Approach

**Start with Option A (env var)** for quick validation:

1. Add `SCOPE_PRELOAD_LORAS` env var support to `pipeline_manager.py`
2. When set, override GUI selection with env var list
3. Force `runtime_peft` mode when multiple LoRAs specified
4. Add visual indicator in GUI that "LoRA selection locked by config"

Then consider Option C (config file) for production polish.

---

## Implementation Sketch

```python
# pipeline_manager.py

def _get_lora_configs(self, load_params: dict) -> tuple[list[dict], str]:
    """Get LoRA configs, respecting env var override."""

    preload_env = os.environ.get("SCOPE_PRELOAD_LORAS")
    if preload_env:
        # Env var overrides GUI selection
        lora_names = [n.strip() for n in preload_env.split(",")]
        lora_dir = get_models_dir() / "lora"
        loras = []
        for name in lora_names:
            # Find matching file
            matches = list(lora_dir.glob(f"{name}*.safetensors"))
            if matches:
                loras.append({"path": str(matches[0]), "scale": 0.0})

        if len(loras) > 1:
            # Multiple LoRAs require runtime_peft
            return loras, "runtime_peft"
        return loras, load_params.get("lora_merge_mode", "permanent_merge")

    # Fall back to GUI-provided config
    return load_params.get("loras", []), load_params.get("lora_merge_mode", "permanent_merge")
```

---

## Testing Plan

1. Start server with `SCOPE_PRELOAD_LORAS=rat,tmnt`
2. Verify log shows: `Loading 2 LoRA(s) with runtime_peft strategy`
3. Connect via GUI, verify generation works
4. Switch styles via CLI: `video-cli style set tmnt`
5. Verify log shows: `LoRA switch: tmnt... @ 0.85, zeroed 1 others`
6. Verify visual output changes to TMNT style

---

## Open Questions

1. **Memory impact**: 4 LoRAs at 600MB each = 2.4GB extra VRAM. Acceptable?
2. **Startup time**: `runtime_peft` is slower to load than `permanent_merge`. How much?
3. **GUI behavior**: Disable LoRA selector entirely, or show read-only indicator?
4. **Default LoRA**: When server starts, which LoRA is active (scale > 0)?

---

## Related

- Phase 6a: Style layer scaffolding (done)
- `frame_processor.py`: LoRA zero-out logic (done)
- `styles/*/manifest.yaml`: LoRA paths defined (done)
