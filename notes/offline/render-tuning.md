# Krea Offline Render Tuning Guide

## Core Architecture Constraints

### Frame Generation
- **3 frames per pipeline call** (hardcoded in model weights via `num_frame_per_block`)
- **16 FPS native** - no benefit going higher, model trained at this temporal granularity
- VAE temporal factor (4x) is internal latent representation, not output multiplier

### Attention Window
- `local_attn_size: 6` = 2 blocks of context (6 frames)
- `kv_cache_num_frames: 3` = frames stored for KV cache recomputation
- Designed as sliding window: current block + 1 previous block visible

---

## Tuning Knobs

### 1. Resolution (`--height`, `--width`)
Must be divisible by 16.

| Resolution | Aspect | VRAM (FP8) | Use Case |
|------------|--------|------------|----------|
| 320×576 | 16:9 | ~32GB | Preview, fast iteration |
| 480×832 | ~16:9 | ~40GB | Standard quality |
| 720×1280 | 16:9 | ~48-80GB | High quality final |

**Tradeoff:** Higher res = more VRAM, slower. May need to reduce `kv_cache_num_frames` to fit.

### 2. Denoising Steps (`--denoising-steps` or `--num-inference-steps`)
Controls quality vs speed.

| Steps | Schedule | Use Case |
|-------|----------|----------|
| 4 | `1000,750,500,250` | Realtime/preview (Krea default) |
| 6 | `1000,850,700,550,400,250` | Offline quality (Krea examples max out here) |
| 8+ | Untested | May improve quality, needs verification |

**Tradeoff:** More steps = better quality, linear slowdown. Krea examples use 6 for offline; >6 not benchmarked.

### 3. KV Cache Num Frames (`--kv-cache-num-frames`)
Context window for temporal coherence.

| Value | Effect |
|-------|--------|
| 2 | Less memory, enables higher res, more temporal drift |
| 3 | Default balance |
| 4-5 | Better coherence, more VRAM, may block higher res |

**Key insight:** "If your goal is higher res, it is totally rational to spend temporal coherence budget here" - reduce cache frames to enable resolution.

### 4. KV Cache Attention Bias (`--kv-cache-attention-bias`)
How much to attend to past frames.

| Value | Effect |
|-------|--------|
| 0.3 | Default, tested value |
| Lower | Less reliance on past (reduces error accumulation, looping) |
| Higher | Stronger coherence but may accumulate artifacts |

**Recommendation:** Keep at 0.3 unless debugging specific issues.

### 5. Quantization (`--quantization`)
Memory vs precision.

| Value | Effect |
|-------|--------|
| `fp8_e4m3fn` | ~50% VRAM reduction, slight quality loss |
| `none` | Full precision, needs ~2x VRAM |

**Recommendation:** Use FP8 unless you have 80GB+ VRAM and want maximum quality.

### 6. Compile (`--compile`)
Torch compilation for speed.

- Auto-enabled on H100/Hopper GPUs
- Helps hold FPS while raising resolution
- First run slower (compilation), subsequent runs faster

---

## Preset Definitions

### `preview`
Fast iteration, low VRAM requirement.
```
height: 320
width: 576
denoising_steps: [1000, 750, 500, 250]  # 4 steps
kv_cache_attention_bias: 0.3
kv_cache_num_frames: 3
quantization: fp8_e4m3fn
```
**VRAM:** ~32GB | **Speed:** Fastest

### `standard`
Balanced quality and speed.
```
height: 480
width: 832
denoising_steps: [1000, 750, 500, 250]  # 4 steps
kv_cache_attention_bias: 0.3
kv_cache_num_frames: 3
quantization: fp8_e4m3fn
```
**VRAM:** ~40GB | **Speed:** Fast

### `quality`
Offline quality render, more steps.
```
height: 480
width: 832
denoising_steps: [1000, 850, 700, 550, 400, 250]  # 6 steps
kv_cache_attention_bias: 0.3
kv_cache_num_frames: 3
quantization: fp8_e4m3fn
```
**VRAM:** ~48GB | **Speed:** 1.5x slower than standard

### `highres`
Higher resolution, trade cache for memory.
```
height: 720
width: 1280
denoising_steps: [1000, 750, 500, 250]  # 4 steps
kv_cache_attention_bias: 0.3
kv_cache_num_frames: 2  # Reduced for memory
quantization: fp8_e4m3fn
```
**VRAM:** ~48GB | **Speed:** Slower (more pixels)

### `max`
Maximum quality, needs high-end GPU.
```
height: 720
width: 1280
denoising_steps: [1000, 850, 700, 550, 400, 250]  # 6 steps
kv_cache_attention_bias: 0.3
kv_cache_num_frames: 3
quantization: none  # Full precision
```
**VRAM:** ~80GB+ | **Speed:** Slowest

---

## Workflow Recommendations

### Preview-then-Refine (from 5pro notes)
1. Generate continuous take at `preview` or `standard` preset
2. Review, iterate on prompts/timing
3. Final render at `quality` or `highres` preset
4. Optional: V2V refine pass with low strength for extra detail

### Memory-Constrained (< 48GB)
- Use `preview` or `standard` preset
- If need higher res: reduce `kv_cache_num_frames` to 2
- Always use FP8 quantization

### Quality-Maximizing (80GB+ VRAM)
- Use `max` preset
- Can try `kv_cache_num_frames: 4` for better coherence
- Disable quantization for full precision

---

## Reference Values

From Krea model card and testing:
- **11 FPS on B200** with 4 steps (realtime benchmark)
- **kv_cache_attention_bias: 0.3** is the tested default
- **kv_cache_num_frames: 3** balances coherence and memory
- **FP8 quantization** recommended for most use cases

---

## Sources
- `src/scope/core/pipelines/krea_realtime_video/model.yaml`
- `src/scope/core/pipelines/krea_realtime_video/test.py`
- `notes/krea_rt.md` (Krea model card)
- `notes/5pro_chat.md` and `notes/5pro_chat_02.md` (tuning analysis)
- `notes/offline-timeline-renderer-report.md` (Codex implementation report)
