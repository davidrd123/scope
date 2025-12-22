# Offline Timeline Renderer for Krea Realtime Video

## Overview
Create a Python CLI script that takes a timeline JSON file and renders it to a video file using the Krea Realtime Video pipeline (text-to-video mode).

## Files to Create

### 1. `src/scope/cli/__init__.py`
Empty file to create the new `cli` package.

### 2. `src/scope/cli/render_timeline.py`
Main CLI script (~200 lines).

**Key components:**
- `argparse` CLI with arguments: `timeline` (input JSON), `output` (output .mp4), `--fps`, `--quantization`, `--height`, `--width`, `--device`, `--compile`, `-v`
- Timeline JSON parser using Pydantic models
- Frame calculation: `frames = int(timestamp * fps)`, default 16 FPS
- Pipeline initialization using settings from timeline JSON
- Rendering loop that generates 3 frames per pipeline call
- Progress bar with `tqdm`
- Video export using `diffusers.utils.export_to_video`

**Rendering logic:**
```python
for segment in timeline.prompts:
    num_frames = int((segment.endTime - segment.startTime) * fps)
    num_calls = (num_frames + 2) // 3  # 3 frames per call

    prompts = [{"text": p.text, "weight": p.weight} for p in segment.prompts]

    for _ in range(num_calls):
        output = pipeline(prompts=prompts, kv_cache_attention_bias=0.3)
        outputs.append(output.detach().cpu())

video = torch.cat(outputs)
export_to_video(video.numpy(), output_path, fps=fps)
```

## Files to Modify

### 3. `pyproject.toml`
Add entry point:
```toml
[project.scripts]
render_timeline = "scope.cli.render_timeline:main"
```

## Reference Files
- `src/scope/core/pipelines/krea_realtime_video/test.py` - Pipeline usage pattern
- `src/scope/core/pipelines/krea_realtime_video/pipeline.py` - `KreaRealtimeVideoPipeline` class
- `src/scope/server/download_models.py` - CLI argparse pattern

## Usage
```bash
# Basic usage
uv run render_timeline timeline.json output.mp4

# With FP8 quantization (32GB VRAM)
uv run render_timeline timeline.json output.mp4 --quantization fp8_e4m3fn

# Custom resolution
uv run render_timeline timeline.json output.mp4 --height 480 --width 832
```

## Validation
- Check `pipelineId` is `"krea-realtime-video"`
- Validate segments are ordered and non-overlapping
- Verify all segments have at least one prompt

## Out of Scope (for now)
- Transition handling (`transitionSteps`, `temporalInterpolationMethod`) - would require embedding blending integration
- Video-to-video mode - requires input video processing
- LoRA loading - additional complexity
