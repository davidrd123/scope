from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Literal

try:
    from pydantic import BaseModel, ConfigDict, Field

    _HAS_PYDANTIC = True
except ModuleNotFoundError:  # pragma: no cover
    BaseModel = object  # type: ignore
    ConfigDict = dict  # type: ignore

    def Field(default=None, **_kwargs):  # type: ignore
        return default

    _HAS_PYDANTIC = False

logger = logging.getLogger(__name__)

TemporalInterpolationMethod = Literal["linear", "slerp"]

# Quality presets for offline rendering
# Priority: CLI flags > timeline settings > preset defaults
PRESETS: dict[str, dict] = {
    "preview": {
        "description": "Fast iteration, low VRAM (~32GB)",
        "height": 320,
        "width": 576,
        "denoising_steps": [1000, 750, 500, 250],
        "kv_cache_attention_bias": 0.3,
        "kv_cache_num_frames": 3,
        "quantization": "fp8_e4m3fn",
    },
    "standard": {
        "description": "Balanced quality and speed (~40GB)",
        "height": 480,
        "width": 832,
        "denoising_steps": [1000, 750, 500, 250],
        "kv_cache_attention_bias": 0.3,
        "kv_cache_num_frames": 3,
        "quantization": "fp8_e4m3fn",
    },
    "quality": {
        "description": "Offline quality, more steps (~48GB)",
        "height": 480,
        "width": 832,
        "denoising_steps": [1000, 850, 700, 550, 400, 250],
        "kv_cache_attention_bias": 0.3,
        "kv_cache_num_frames": 3,
        "quantization": "fp8_e4m3fn",
    },
    "highres": {
        "description": "720p, reduced cache for memory (~48GB)",
        "height": 720,
        "width": 1280,
        "denoising_steps": [1000, 750, 500, 250],
        "kv_cache_attention_bias": 0.3,
        "kv_cache_num_frames": 2,
        "quantization": "fp8_e4m3fn",
    },
    "max": {
        "description": "Maximum quality, needs 80GB+ VRAM",
        "height": 720,
        "width": 1280,
        "denoising_steps": [1000, 850, 700, 550, 400, 250],
        "kv_cache_attention_bias": 0.3,
        "kv_cache_num_frames": 3,
        "quantization": None,
    },
}


class TimelinePromptItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str = Field(..., description="Prompt text")
    weight: float = Field(default=100.0, description="Prompt weight")


class TimelineSegment(BaseModel):
    model_config = ConfigDict(extra="ignore")

    startTime: float
    endTime: float
    prompts: list[TimelinePromptItem] | None = None
    text: str | None = None
    transitionSteps: int | None = None
    temporalInterpolationMethod: TemporalInterpolationMethod | None = None

    def prompt_items(self) -> list[dict]:
        if self.prompts:
            return [p.model_dump() for p in self.prompts]
        if self.text and self.text.strip():
            return [{"text": self.text, "weight": 100.0}]
        return []


class TimelineResolution(BaseModel):
    model_config = ConfigDict(extra="ignore")

    height: int
    width: int


class TimelineLoRA(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str | None = None
    path: str
    scale: float = 1.0
    mergeMode: Literal["permanent_merge", "runtime_peft"] | None = None


class TimelineSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    pipelineId: str
    inputMode: Literal["text", "video"] | None = None
    resolution: TimelineResolution | None = None
    seed: int | None = None
    denoisingSteps: list[int] | None = None
    manageCache: bool | None = None
    quantization: str | None = None
    kvCacheAttentionBias: float | None = None
    loras: list[TimelineLoRA] | None = None
    loraMergeStrategy: Literal["permanent_merge", "runtime_peft"] | None = None


class TimelineFile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    prompts: list[TimelineSegment]
    settings: TimelineSettings
    version: str | None = None
    exportedAt: str | None = None


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a Scope timeline JSON export to an MP4 (offline/high-quality)."
        )
    )
    parser.add_argument(
        "timeline",
        type=Path,
        nargs="?",
        default=None,
        help="Input timeline JSON",
    )
    parser.add_argument(
        "output", type=Path, nargs="?", default=None, help="Output .mp4 path"
    )

    parser.add_argument("--fps", type=int, default=16, help="Output FPS")
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Override output height (must be divisible by 16)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Override output width (must be divisible by 16)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override seed")

    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help=(
            "Override denoising step count (Krea model card suggests 6 for offline quality). "
            "Converted into a linear timestep schedule."
        ),
    )
    parser.add_argument(
        "--denoising-steps",
        type=str,
        default=None,
        help="Override denoising timesteps (comma-separated, e.g. 1000,850,700,550,400,250)",
    )
    parser.add_argument(
        "--kv-cache-attention-bias",
        type=float,
        default=None,
        help="Override kv_cache_attention_bias (e.g. 0.3)",
    )
    parser.add_argument(
        "--kv-cache-num-frames",
        type=int,
        default=None,
        help="Override Krea kv_cache_num_frames (>=1) used for KV cache recomputation",
    )
    parser.add_argument(
        "--manage-cache",
        type=lambda s: s.lower() in ("1", "true", "yes", "y", "on"),
        default=None,
        help="Override manage_cache (true/false)",
    )
    parser.add_argument(
        "--quantization",
        choices=["fp8_e4m3fn", "none"],
        default=None,
        help="Override quantization (default: from timeline)",
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable torch.compile (default: auto on H100/Hopper)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device (e.g. cuda, cuda:0)",
    )

    parser.add_argument(
        "--transition-default-steps",
        type=int,
        default=4,
        help="Default transition steps when a segment omits transitionSteps",
    )
    parser.add_argument(
        "--transition-default-method",
        choices=["linear", "slerp"],
        default="slerp",
        help="Default temporal interpolation method when a segment omits it",
    )
    parser.add_argument(
        "--no-transitions",
        action="store_true",
        help="Disable prompt transition smoothing (hard cuts between segments)",
    )

    parser.add_argument(
        "--preset",
        choices=list(PRESETS.keys()),
        default=None,
        help="Quality preset (preview, standard, quality, highres, max). CLI flags override preset values.",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="Show available presets and exit",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse timeline and print the resolved render plan without loading any models",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    return parser.parse_args(argv)


def _parse_denoising_steps(value: str) -> list[int]:
    steps: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        steps.append(int(part))
    if not steps:
        raise ValueError("denoising steps list is empty")
    return steps


def _build_linear_timestep_schedule(
    num_steps: int, *, start: int, end: int
) -> list[int]:
    if num_steps <= 0:
        raise ValueError("num_inference_steps must be > 0")
    if num_steps == 1:
        return [int(start)]
    step = (start - end) / (num_steps - 1)
    return [int(round(start - i * step)) for i in range(num_steps)]


def _validate_resolution(height: int, width: int) -> None:
    scale = 16
    if height % scale != 0 or width % scale != 0:
        raise ValueError(
            f"height/width must be divisible by {scale} (got {height}x{width})"
        )


def _validate_timeline_segments(segments: list[TimelineSegment]) -> None:
    if not segments:
        raise ValueError("timeline has no segments")

    last_start: float | None = None
    last_end: float | None = None
    eps = 1e-6

    for idx, segment in enumerate(segments):
        start = (
            segment.get("startTime") if isinstance(segment, dict) else segment.startTime
        )
        end = segment.get("endTime") if isinstance(segment, dict) else segment.endTime

        if start is None or end is None:
            raise ValueError(f"segment {idx} is missing startTime/endTime")

        if start < 0 or end < 0:
            raise ValueError(
                f"segment {idx} has negative times: start={start} end={end}"
            )
        if end <= start:
            raise ValueError(
                f"segment {idx} has non-positive duration: start={start} end={end}"
            )

        if last_start is not None and start + eps < last_start:
            raise ValueError(
                f"segments are not sorted by startTime: segment {idx} starts at {start} before previous {last_start}"
            )

        if last_end is not None and start + eps < last_end:
            raise ValueError(
                f"segments overlap: segment {idx} starts at {start} before previous ends at {last_end}"
            )

        last_start = float(start)
        last_end = float(end)


def _count_prompt_items(segment: object) -> int:
    if isinstance(segment, dict):
        prompts = segment.get("prompts")
        if isinstance(prompts, list):
            return len(prompts)
        text = segment.get("text")
        if isinstance(text, str) and text.strip():
            return 1
        return 0

    try:
        return len(segment.prompt_items())  # type: ignore[attr-defined]
    except Exception:
        return 0


def render_timeline(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_logging(args.verbose)

    if args.list_presets:
        print("Available presets:\n")
        for name, preset in PRESETS.items():
            print(f"  {name:12} {preset['description']}")
            print(f"               {preset['height']}x{preset['width']}, {len(preset['denoising_steps'])} steps")
        return 0

    if args.timeline is None:
        raise SystemExit("error: timeline argument is required")
    if args.output is None:
        raise SystemExit("error: output argument is required")

    timeline_raw = json.loads(args.timeline.read_text())
    if args.dry_run:
        settings_raw = timeline_raw.get("settings") if isinstance(timeline_raw, dict) else None
        if not isinstance(settings_raw, dict):
            raise SystemExit("Invalid timeline: missing settings")

        pipeline_id = settings_raw.get("pipelineId")
        input_mode = settings_raw.get("inputMode", "text")
        if pipeline_id != "krea-realtime-video":
            raise SystemExit(
                f"Only pipelineId='krea-realtime-video' is supported right now (got {pipeline_id!r})"
            )
        if input_mode and input_mode != "text":
            raise SystemExit(
                f"Only text mode is supported right now (got inputMode={input_mode!r})"
            )

        prompts_raw = timeline_raw.get("prompts")
        if not isinstance(prompts_raw, list) or not prompts_raw:
            raise SystemExit("Timeline has no prompts")

        segments_raw = sorted(prompts_raw, key=lambda s: (s or {}).get("startTime", 0))
        try:
            _validate_timeline_segments(segments_raw)  # type: ignore[arg-type]
        except ValueError as e:
            raise SystemExit(f"Invalid timeline: {e}") from e

        end_time = max(float(s.get("endTime", 0)) for s in segments_raw)
        fps = int(args.fps)
        if fps <= 0:
            raise SystemExit("--fps must be > 0")
        target_frames = int(math.ceil(end_time * fps))

        # Resolve settings: CLI flags > timeline settings > preset defaults
        preset = PRESETS.get(args.preset, {}) if args.preset else {}

        resolution_raw = settings_raw.get("resolution") or {}
        height = args.height or resolution_raw.get("height") or preset.get("height")
        width = args.width or resolution_raw.get("width") or preset.get("width")
        if height is None or width is None:
            raise SystemExit(
                "Missing resolution: provide --height/--width, --preset, or include settings.resolution in timeline"
            )
        _validate_resolution(int(height), int(width))

        denoising_steps = (
            settings_raw.get("denoisingSteps")
            or preset.get("denoising_steps")
            or [1000, 750, 500, 250]
        )
        if args.denoising_steps is not None:
            denoising_steps = _parse_denoising_steps(args.denoising_steps)
        elif args.num_inference_steps is not None:
            denoising_steps = _build_linear_timestep_schedule(
                args.num_inference_steps,
                start=int(denoising_steps[0]),
                end=int(denoising_steps[-1]),
            )

        manage_cache = (
            args.manage_cache
            if args.manage_cache is not None
            else settings_raw.get("manageCache", True)
        )
        kv_cache_attention_bias = (
            args.kv_cache_attention_bias
            if args.kv_cache_attention_bias is not None
            else settings_raw.get("kvCacheAttentionBias")
            or preset.get("kv_cache_attention_bias", 0.3)
        )
        kv_cache_num_frames = args.kv_cache_num_frames or preset.get("kv_cache_num_frames")

        quantization = settings_raw.get("quantization")
        if args.quantization == "none":
            quantization = None
        elif args.quantization == "fp8_e4m3fn":
            quantization = "fp8_e4m3fn"
        elif args.quantization is None and "quantization" in preset:
            quantization = preset["quantization"]

        frames_per_call = 3  # krea-realtime-video defaults (num_frame_per_block)
        est_calls = int(math.ceil(target_frames / frames_per_call))
        plan = {
            "preset": args.preset,
            "pipelineId": pipeline_id,
            "inputMode": input_mode,
            "durationSeconds": end_time,
            "fps": fps,
            "targetFrames": target_frames,
            "estimatedPipelineCalls": est_calls,
            "resolution": {"height": int(height), "width": int(width)},
            "seed": args.seed if args.seed is not None else settings_raw.get("seed", 42),
            "denoising_step_list": denoising_steps,
            "manage_cache": bool(manage_cache),
            "quantization": quantization,
            "kv_cache_attention_bias": float(kv_cache_attention_bias),
            "kv_cache_num_frames": kv_cache_num_frames or 3,
            "segments": [
                {
                    "startTime": float(s.get("startTime", 0)),
                    "endTime": float(s.get("endTime", 0)),
                    "numPromptItems": _count_prompt_items(s),
                    "transitionSteps": s.get("transitionSteps"),
                    "temporalInterpolationMethod": s.get(
                        "temporalInterpolationMethod"
                    ),
                }
                for s in segments_raw
            ],
        }
        print(json.dumps(plan, indent=2))
        return 0

    if not _HAS_PYDANTIC:
        raise SystemExit(
            "pydantic is required for non --dry-run execution. "
            "Install project dependencies (e.g. via `uv sync`)."
        )

    timeline = TimelineFile.model_validate(timeline_raw)

    settings = timeline.settings
    if settings.pipelineId != "krea-realtime-video":
        raise SystemExit(
            f"Only pipelineId='krea-realtime-video' is supported right now (got {settings.pipelineId!r})"
        )

    if settings.inputMode and settings.inputMode != "text":
        raise SystemExit(
            f"Only text mode is supported right now (got inputMode={settings.inputMode!r})"
        )

    if not timeline.prompts:
        raise SystemExit("Timeline has no prompts")

    segments = sorted(timeline.prompts, key=lambda s: s.startTime)
    try:
        _validate_timeline_segments(segments)
    except ValueError as e:
        raise SystemExit(f"Invalid timeline: {e}") from e
    end_time = max(s.endTime for s in segments)
    if end_time <= 0:
        raise SystemExit("Timeline duration is 0s")

    fps = int(args.fps)
    if fps <= 0:
        raise SystemExit("--fps must be > 0")

    target_frames = int(math.ceil(end_time * fps))
    logger.info("Timeline duration=%.3fs target_frames=%d fps=%d", end_time, target_frames, fps)

    # Resolve render settings: CLI flags > timeline settings > preset defaults
    preset = PRESETS.get(args.preset, {}) if args.preset else {}

    height = (
        args.height
        or (settings.resolution.height if settings.resolution else None)
        or preset.get("height")
    )
    width = (
        args.width
        or (settings.resolution.width if settings.resolution else None)
        or preset.get("width")
    )
    seed = args.seed if args.seed is not None else (settings.seed or 42)
    manage_cache = (
        args.manage_cache
        if args.manage_cache is not None
        else (settings.manageCache if settings.manageCache is not None else True)
    )

    if height is None or width is None:
        raise SystemExit(
            "Missing resolution: provide --height/--width, --preset, or include settings.resolution in timeline"
        )
    _validate_resolution(height, width)

    denoising_steps = (
        settings.denoisingSteps
        or preset.get("denoising_steps")
        or [1000, 750, 500, 250]
    )
    if args.denoising_steps is not None:
        denoising_steps = _parse_denoising_steps(args.denoising_steps)
    elif args.num_inference_steps is not None:
        denoising_steps = _build_linear_timestep_schedule(
            args.num_inference_steps, start=denoising_steps[0], end=denoising_steps[-1]
        )

    kv_cache_attention_bias = (
        args.kv_cache_attention_bias
        if args.kv_cache_attention_bias is not None
        else settings.kvCacheAttentionBias
        or preset.get("kv_cache_attention_bias", 0.3)
    )

    kv_cache_num_frames = args.kv_cache_num_frames or preset.get("kv_cache_num_frames")

    quantization = settings.quantization
    if args.quantization == "none":
        quantization = None
    elif args.quantization == "fp8_e4m3fn":
        quantization = "fp8_e4m3fn"
    elif args.quantization is None and "quantization" in preset:
        quantization = preset["quantization"]

    # Lazy imports: keep module import light, and make failures more actionable.
    try:
        import torch
        from diffusers.utils import export_to_video
        from omegaconf import OmegaConf

        from scope.core.config import get_model_file_path, get_models_dir
        from scope.core.pipelines.krea_realtime_video.pipeline import (
            KreaRealtimeVideoPipeline,
        )
        from scope.core.pipelines.utils import Quantization
    except Exception as e:
        raise SystemExit(
            "Required ML dependencies are missing. Ensure Scope dependencies are installed "
            "(torch, diffusers, omegaconf, etc)."
        ) from e

    device = torch.device(args.device)

    # Load + optionally override model config (e.g., kv_cache_num_frames)
    model_config = None
    if kv_cache_num_frames is not None:
        if kv_cache_num_frames < 1:
            raise SystemExit("--kv-cache-num-frames must be >= 1")
        import scope.core.pipelines.krea_realtime_video.pipeline as krea_pipeline_module

        model_yaml_path = Path(krea_pipeline_module.__file__).parent / "model.yaml"
        model_config = OmegaConf.load(model_yaml_path)
        model_config["kv_cache_num_frames"] = int(kv_cache_num_frames)

    loras = []
    if settings.loras:
        for lora in settings.loras:
            lora_dict: dict[str, object] = {"path": lora.path, "scale": lora.scale}
            if lora.mergeMode is not None:
                lora_dict["merge_mode"] = lora.mergeMode
            loras.append(lora_dict)

    config_dict: dict[str, object] = {
        "model_dir": str(get_models_dir()),
        "generator_path": str(
            get_model_file_path("krea-realtime-video/krea-realtime-video-14b.safetensors")
        ),
        "text_encoder_path": str(
            get_model_file_path("WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors")
        ),
        "tokenizer_path": str(get_model_file_path("Wan2.1-T2V-1.3B/google/umt5-xxl")),
        "vae_path": str(get_model_file_path("Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")),
        "height": int(height),
        "width": int(width),
        "seed": int(seed),
    }
    if model_config is not None:
        config_dict["model_config"] = model_config
    if loras:
        config_dict["loras"] = loras
        config_dict["_lora_merge_mode"] = settings.loraMergeStrategy or "permanent_merge"

    config = OmegaConf.create(config_dict)

    compile_enabled = args.compile
    if compile_enabled is None and device.type == "cuda" and torch.cuda.is_available():
        name = torch.cuda.get_device_name(device.index or 0).lower()
        compile_enabled = any(x in name for x in ("h100", "hopper"))

    quantization_enum = None
    if quantization == Quantization.FP8_E4M3FN or quantization == "fp8_e4m3fn":
        quantization_enum = Quantization.FP8_E4M3FN

    logger.info(
        "Loading pipeline=%s device=%s compile=%s quantization=%s resolution=%dx%d steps=%s",
        settings.pipelineId,
        device,
        compile_enabled,
        quantization_enum,
        width,
        height,
        denoising_steps,
    )
    pipeline = KreaRealtimeVideoPipeline(
        config,
        quantization=quantization_enum,
        compile=bool(compile_enabled),
        device=device,
        dtype=torch.bfloat16,
    )

    # Rendering loop (mimics server FrameProcessor lifecycle for transitions)
    parameters: dict[str, object] = {
        "manage_cache": bool(manage_cache),
        "kv_cache_attention_bias": float(kv_cache_attention_bias),
        "denoising_step_list": list(denoising_steps),
    }

    outputs: list["torch.Tensor"] = []
    produced_frames = 0
    seg_idx = 0
    active_segment = segments[0]

    # Initialize prompts with the first segment
    first_prompts = active_segment.prompt_items()
    if not first_prompts:
        raise SystemExit("First timeline segment has no prompts")
    parameters["prompts"] = first_prompts

    last_segment_id = (active_segment.startTime, active_segment.endTime)

    while produced_frames < target_frames:
        current_time = produced_frames / fps

        # Advance segments by startTime so gaps behave like "hold last prompt"
        # until the next segment begins.
        while (
            seg_idx + 1 < len(segments)
            and current_time >= segments[seg_idx + 1].startTime
        ):
            seg_idx += 1
            active_segment = segments[seg_idx]

        current_segment_id = (active_segment.startTime, active_segment.endTime)
        if current_segment_id != last_segment_id:
            target_prompts = active_segment.prompt_items()
            if not target_prompts:
                logger.warning(
                    "Skipping empty prompt segment idx=%d start=%.3f end=%.3f",
                    seg_idx,
                    active_segment.startTime,
                    active_segment.endTime,
                )
            else:
                transition_steps = (
                    active_segment.transitionSteps
                    if active_segment.transitionSteps is not None
                    else int(args.transition_default_steps)
                )
                temporal_method: TemporalInterpolationMethod = (
                    active_segment.temporalInterpolationMethod
                    if active_segment.temporalInterpolationMethod is not None
                    else args.transition_default_method
                )

                if args.no_transitions or transition_steps <= 0:
                    parameters.pop("transition", None)
                    parameters["prompts"] = target_prompts
                    logger.info(
                        "Cut -> segment idx=%d start=%.3f end=%.3f",
                        seg_idx,
                        active_segment.startTime,
                        active_segment.endTime,
                    )
                else:
                    parameters["transition"] = {
                        "target_prompts": target_prompts,
                        "num_steps": int(transition_steps),
                        "temporal_interpolation_method": temporal_method,
                    }
                    logger.info(
                        "Transition -> segment idx=%d start=%.3f end=%.3f steps=%d method=%s",
                        seg_idx,
                        active_segment.startTime,
                        active_segment.endTime,
                        transition_steps,
                        temporal_method,
                    )

            last_segment_id = current_segment_id

        output = pipeline(**parameters)
        outputs.append(output.detach().cpu())
        produced_frames += int(output.shape[0])

        # Keep sending transition until the pipeline signals it completed (like server FrameProcessor).
        if "transition" in parameters and hasattr(pipeline, "state"):
            if not bool(pipeline.state.get("_transition_active", False)):
                transition = parameters.get("transition") or {}
                target_prompts = transition.get("target_prompts")
                if isinstance(target_prompts, list) and target_prompts:
                    parameters["prompts"] = target_prompts
                parameters.pop("transition", None)

        if produced_frames % (fps * 5) < output.shape[0]:
            logger.info(
                "Progress: %d/%d frames (%.1fs/%.1fs)",
                produced_frames,
                target_frames,
                produced_frames / fps,
                end_time,
            )

    video = torch.cat(outputs, dim=0)[:target_frames]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(video.contiguous().numpy(), args.output, fps=fps)
    logger.info("Wrote %s (%d frames @ %dfps)", args.output, target_frames, fps)
    return 0


def main() -> None:
    raise SystemExit(render_timeline())


if __name__ == "__main__":  # pragma: no cover
    main()
