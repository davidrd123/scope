# Offline Timeline Renderer Guide

## Quick Start

```bash
# List available presets
uv run render_timeline --list-presets

# Render with a preset
uv run render_timeline timeline.json out.mp4 --preset quality

# Preview what will render (no model loading)
uv run render_timeline timeline.json out.mp4 --preset highres --dry-run
```

---

## Presets

| Preset | Resolution | Steps | KV Cache Frames | Quantization | VRAM |
|--------|-----------|-------|-----------------|--------------|------|
| `preview` | 320×576 | 4 | 3 | FP8 | ~32GB |
| `standard` | 480×832 | 4 | 3 | FP8 | ~40GB |
| `quality` | 480×832 | 6 | 3 | FP8 | ~48GB |
| `highres` | 720×1280 | 4 | 2 | FP8 | ~48GB |
| `max` | 720×1280 | 6 | 3 | None | ~80GB+ |

**Notes:**
- `highres` reduces KV cache frames (2 vs 3) to fit higher resolution in memory
- `max` disables quantization for maximum quality, requires high-end GPU
- All presets use `kv_cache_attention_bias: 0.3`

---

## Priority Order

**CLI flags > preset > timeline defaults**

This means:
- `--preset highres` overrides whatever resolution is in your timeline
- `--height 480 --width 832` overrides the preset's resolution
- If no preset and no CLI flag, uses timeline's value

---

## All CLI Parameters

### Required
| Parameter | Description |
|-----------|-------------|
| `timeline` | Input timeline JSON file |
| `output` | Output .mp4 path |

### Quality Settings
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--preset` | None | Quality preset (preview/standard/quality/highres/max) |
| `--height` | From preset/timeline | Output height (must be divisible by 16) |
| `--width` | From preset/timeline | Output width (must be divisible by 16) |
| `--num-inference-steps` | From preset/timeline | Number of denoising steps (converted to linear schedule) |
| `--denoising-steps` | From preset/timeline | Explicit timesteps (e.g., `1000,850,700,550,400,250`) |
| `--quantization` | From preset/timeline | `fp8_e4m3fn` or `none` |

### KV Cache Settings
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--kv-cache-attention-bias` | 0.3 | How much to attend to past frames (lower = less reliance) |
| `--kv-cache-num-frames` | 3 | Frames stored for KV cache recomputation (2-5) |

### Rendering Settings
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--fps` | 16 | Output FPS (16 is native, no benefit going higher) |
| `--seed` | From timeline or 42 | Random seed for reproducibility |
| `--device` | `cuda` | Torch device (e.g., `cuda`, `cuda:0`) |
| `--compile` / `--no-compile` | Auto | Enable/disable torch.compile (auto-enabled on H100) |

### Transition Settings
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--transition-default-steps` | 4 | Transition steps when segment omits transitionSteps |
| `--transition-default-method` | `slerp` | Interpolation method (`linear` or `slerp`) |
| `--no-transitions` | False | Hard cuts between segments (no smooth transitions) |

### Other
| Parameter | Description |
|-----------|-------------|
| `--list-presets` | Show available presets and exit |
| `--dry-run` | Print render plan as JSON without loading models |
| `--verbose` / `-v` | Enable verbose logging |

---

## Examples

### Basic Preset Usage
```bash
# Fast preview
uv run render_timeline timeline.json preview.mp4 --preset preview

# Balanced quality
uv run render_timeline timeline.json standard.mp4 --preset standard

# Offline quality (6 steps)
uv run render_timeline timeline.json quality.mp4 --preset quality

# 720p output
uv run render_timeline timeline.json highres.mp4 --preset highres

# Maximum quality (needs 80GB+ VRAM)
uv run render_timeline timeline.json max.mp4 --preset max
```

### Override Preset Values
```bash
# Use quality preset but with 8 steps
uv run render_timeline timeline.json out.mp4 --preset quality --num-inference-steps 8

# Use highres preset but with 3 KV cache frames
uv run render_timeline timeline.json out.mp4 --preset highres --kv-cache-num-frames 3

# Use standard preset but disable quantization
uv run render_timeline timeline.json out.mp4 --preset standard --quantization none
```

### Custom Settings (No Preset)
```bash
# Fully custom configuration
uv run render_timeline timeline.json out.mp4 \
  --height 480 --width 832 \
  --num-inference-steps 8 \
  --kv-cache-num-frames 3 \
  --kv-cache-attention-bias 0.25 \
  --quantization fp8_e4m3fn
```

### Debugging
```bash
# See what would render without loading models
uv run render_timeline timeline.json out.mp4 --preset quality --dry-run

# Verbose output
uv run render_timeline timeline.json out.mp4 --preset standard -v

# Hard cuts (no transitions) for debugging
uv run render_timeline timeline.json out.mp4 --preset standard --no-transitions
```

---

## Tuning Tips

### Resolution vs Memory
- Higher resolution needs more VRAM
- If OOM, reduce `--kv-cache-num-frames` (trade temporal coherence for memory)
- Or use `--quantization fp8_e4m3fn` if not already

### Steps vs Speed
- More steps = better quality, linear slowdown
- 4 steps: realtime/preview
- 6 steps: Krea's offline recommendation
- 8+ steps: untested, may improve quality

### KV Cache
- `kv_cache_num_frames`: Higher = better temporal coherence, more VRAM
- `kv_cache_attention_bias`: Lower = less reliance on past frames (can reduce error accumulation)

### FPS
- 16 FPS is native (model trained at this granularity)
- Higher FPS just generates more frames for the same duration (more compute, no quality benefit)

---

## See Also
- `notes/krea-offline-render-tuning.md` - Detailed tuning knobs documentation
- `notes/offline-timeline-renderer-report.md` - Implementation details
