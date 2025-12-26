import atexit
import json
import logging
import os
import time
from collections import defaultdict
from typing import Any

import torch
from diffusers.modular_pipelines import (
    ModularPipelineBlocks,
    PipelineState,
)
from diffusers.modular_pipelines.modular_pipeline_utils import (
    ComponentSpec,
    ConfigSpec,
    InputParam,
    OutputParam,
)

logger = logging.getLogger(__name__)

_PROFILE_DENOISE_STEPS = os.getenv("PROFILE_DENOISE_STEPS", "0") == "1"
_PROFILE_DENOISE_STEPS_JSON = os.getenv("PROFILE_DENOISE_STEPS_JSON")
_denoise_step_cpu_ms = defaultdict(float)
_denoise_step_gpu_ms = defaultdict(float)
_denoise_step_counts = defaultdict(int)
_denoise_step_meta: dict[str, object] = {}


def _should_profile_denoise_steps() -> bool:
    if not _PROFILE_DENOISE_STEPS:
        return False
    if hasattr(torch, "compiler") and hasattr(torch.compiler, "is_compiling"):
        if torch.compiler.is_compiling():
            return False
    return True


def _denoise_step_profile_report() -> None:
    if not _denoise_step_counts:
        return

    total_cpu_ms = sum(_denoise_step_cpu_ms.values())
    total_gpu_ms = sum(_denoise_step_gpu_ms.values())
    logger.info("=== Denoise Step Profiling Report ===")
    for name, cpu_ms in sorted(_denoise_step_cpu_ms.items(), key=lambda kv: -kv[1]):
        calls = _denoise_step_counts.get(name, 0)
        gpu_ms = _denoise_step_gpu_ms.get(name, 0.0)
        cpu_pct = (100.0 * cpu_ms / total_cpu_ms) if total_cpu_ms > 0 else 0.0
        gpu_pct = (100.0 * gpu_ms / total_gpu_ms) if total_gpu_ms > 0 else 0.0
        cpu_per_call = (cpu_ms / calls) if calls else 0.0
        gpu_per_call = (gpu_ms / calls) if calls else 0.0
        logger.info(
            "  %s: CPU %.1fms (%.1f%%) GPU %.1fms (%.1f%%) [%d calls, CPU %.2fms/call, GPU %.2fms/call]",
            name,
            cpu_ms,
            cpu_pct,
            gpu_ms,
            gpu_pct,
            calls,
            cpu_per_call,
            gpu_per_call,
        )
    logger.info("  TOTAL: CPU %.1fms GPU %.1fms", total_cpu_ms, total_gpu_ms)

    if _PROFILE_DENOISE_STEPS_JSON:
        meta = dict(_denoise_step_meta)
        meta.setdefault("torch", torch.__version__)
        meta.setdefault("cuda", torch.version.cuda)
        if torch.cuda.is_available():
            try:
                meta.setdefault("cuda_device", torch.cuda.get_device_name(0))
                meta.setdefault("cuda_capability", list(torch.cuda.get_device_capability(0)))
            except Exception:
                pass
        payload = {
            "meta": meta,
            "total_cpu_ms": total_cpu_ms,
            "total_gpu_ms": total_gpu_ms,
            "steps": {
                name: {
                    "cpu_ms": _denoise_step_cpu_ms.get(name, 0.0),
                    "gpu_ms": _denoise_step_gpu_ms.get(name, 0.0),
                    "calls": int(_denoise_step_counts.get(name, 0)),
                }
                for name in sorted(_denoise_step_counts.keys())
            },
        }
        try:
            with open(_PROFILE_DENOISE_STEPS_JSON, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
            logger.info("Wrote denoise step profile JSON: %s", _PROFILE_DENOISE_STEPS_JSON)
        except Exception as e:
            logger.warning("Failed writing denoise step profile JSON (%s): %s", _PROFILE_DENOISE_STEPS_JSON, e)


atexit.register(_denoise_step_profile_report)


def reset_denoise_step_profile() -> None:
    """Clear accumulated denoise-step profiling counters."""
    _denoise_step_cpu_ms.clear()
    _denoise_step_gpu_ms.clear()
    _denoise_step_counts.clear()
    _denoise_step_meta.clear()


class _ProfileDenoiseStep:
    __slots__ = ("name", "_cpu_start", "_start", "_end")

    def __init__(self, name: str):
        self.name = name
        self._cpu_start = None
        self._start = None
        self._end = None

    def __enter__(self):
        if not _should_profile_denoise_steps():
            return self
        self._cpu_start = time.perf_counter()
        if torch.cuda.is_available():
            self._start = torch.cuda.Event(enable_timing=True)
            self._end = torch.cuda.Event(enable_timing=True)
            self._start.record()
        return self

    def __exit__(self, *_exc):
        if self._cpu_start is None:
            return

        if self._start is not None and self._end is not None:
            self._end.record()
            self._end.synchronize()
            _denoise_step_gpu_ms[self.name] += self._start.elapsed_time(self._end)

        _denoise_step_cpu_ms[self.name] += (time.perf_counter() - self._cpu_start) * 1000.0
        _denoise_step_counts[self.name] += 1


class DenoiseBlock(ModularPipelineBlocks):
    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("generator", torch.nn.Module),
            ComponentSpec("scheduler", torch.nn.Module),
        ]

    @property
    def expected_configs(self) -> list[ConfigSpec]:
        return [
            ConfigSpec("patch_embedding_spatial_downsample_factor", 2),
            ConfigSpec("vae_spatial_downsample_factor", 8),
        ]

    @property
    def description(self) -> str:
        return "Denoise block that performs iterative denoising"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "height",
                type_hint=int,
                description="Height of the video",
            ),
            InputParam(
                "width",
                type_hint=int,
                description="Width of the video",
            ),
            InputParam(
                "kv_cache_attention_bias",
                default=1.0,
                type_hint=float,
                description="Controls how much to rely on past frames in the cache during generation",
            ),
            # The following should be converted to intermediate inputs to denote that they can come from other blocks
            # and can be modified since they are also listed under intermediate outputs. They are included as inputs for now
            # because of what seems to be a bug where intermediate inputs cannot be simplify accessed in block state via
            # block_state.<intermediate_input>
            InputParam(
                "latents",
                required=True,
                type_hint=torch.Tensor,
                description="Noisy latents to denoise",
            ),
            InputParam(
                "current_denoising_step_list",
                required=True,
                type_hint=torch.Tensor,
                description="Current list of denoising steps",
            ),
            InputParam(
                "conditioning_embeds",
                required=True,
                type_hint=torch.Tensor,
                description="Conditioning embeddings used to condition denoising",
            ),
            InputParam(
                "current_start_frame",
                required=True,
                type_hint=int,
                description="Current starting frame index for current block",
            ),
            InputParam(
                "start_frame",
                type_hint=int | None,
                description="Starting frame index that overrides current_start_frame",
            ),
            InputParam(
                "kv_cache",
                required=True,
                type_hint=list[dict],
                description="Initialized KV cache",
            ),
            InputParam(
                "crossattn_cache",
                required=True,
                type_hint=list[dict],
                description="Initialized cross-attention cache",
            ),
            InputParam(
                "generator",
                required=True,
                description="Random number generator",
            ),
            InputParam(
                "noise_scale",
                type_hint=float | None,
                description="Amount of noise added to video",
            ),
            InputParam(
                "vace_context",
                default=None,
                type_hint=torch.Tensor | None,
                description="VACE context that provides visual conditioning",
            ),
            InputParam(
                "vace_context_scale",
                default=1.0,
                type_hint=float,
                description="Scaling factor for VACE hint injection",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "latents",
                type_hint=torch.Tensor,
                description="Denoised latents",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> tuple[Any, PipelineState]:
        block_state = self.get_block_state(state)

        scale_size = (
            components.config.vae_spatial_downsample_factor
            * components.config.patch_embedding_spatial_downsample_factor
        )
        frame_seq_length = (block_state.height // scale_size) * (
            block_state.width // scale_size
        )

        noise = block_state.latents
        batch_size = noise.shape[0]
        num_frames = noise.shape[1]
        denoising_step_list = block_state.current_denoising_step_list.clone()

        conditional_dict = {"prompt_embeds": block_state.conditioning_embeds}

        start_frame = block_state.current_start_frame
        if block_state.start_frame is not None:
            start_frame = block_state.start_frame

        end_frame = start_frame + num_frames

        if block_state.noise_scale is not None:
            # Higher noise scale -> more denoising steps, more intense changes to input
            # Lower noise scale -> less denoising steps, less intense changes to input
            denoising_step_list[0] = int(1000 * block_state.noise_scale) - 100
        if _should_profile_denoise_steps():
            _denoise_step_meta.update(
                {
                    "batch_size": int(batch_size),
                    "num_frames": int(num_frames),
                    "latent_dtype": str(noise.dtype),
                    "latent_device": str(noise.device),
                    "num_denoising_steps": int(denoising_step_list.numel()),
                    "kv_cache_attention_bias": float(block_state.kv_cache_attention_bias),
                }
            )

        # Avoid per-step allocations for timestep tensors (saves small but frequent overhead).
        timestep = torch.empty(
            [batch_size, num_frames],
            device=noise.device,
            dtype=torch.int64,
        )
        next_timestep_tensor = torch.empty(
            [batch_size * num_frames],
            device=noise.device,
            dtype=torch.long,
        )
        random_noise = None

        # Denoising loop
        for index, current_timestep in enumerate(denoising_step_list):
            timestep.fill_(int(current_timestep))

            if index < len(denoising_step_list) - 1:
                with _ProfileDenoiseStep("generator"):
                    _, denoised_pred = components.generator(
                        noisy_image_or_video=noise,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=block_state.kv_cache,
                        crossattn_cache=block_state.crossattn_cache,
                        current_start=start_frame * frame_seq_length,
                        current_end=end_frame * frame_seq_length,
                        kv_cache_attention_bias=block_state.kv_cache_attention_bias,
                        vace_context=block_state.vace_context,
                        vace_context_scale=block_state.vace_context_scale,
                    )

                next_timestep = denoising_step_list[index + 1]
                # Create noise with same shape and properties as denoised_pred
                flattened_pred = denoised_pred.flatten(0, 1)
                with _ProfileDenoiseStep("randn"):
                    if random_noise is None or random_noise.shape != flattened_pred.shape:
                        random_noise = torch.empty_like(flattened_pred)
                    random_noise.normal_(generator=block_state.generator)
                with _ProfileDenoiseStep("scheduler_add_noise"):
                    next_timestep_tensor.fill_(int(next_timestep))
                    noise = components.scheduler.add_noise(
                        flattened_pred,
                        random_noise,
                        next_timestep_tensor,
                    ).unflatten(0, denoised_pred.shape[:2])
            else:
                with _ProfileDenoiseStep("generator"):
                    _, denoised_pred = components.generator(
                        noisy_image_or_video=noise,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=block_state.kv_cache,
                        crossattn_cache=block_state.crossattn_cache,
                        current_start=start_frame * frame_seq_length,
                        current_end=end_frame * frame_seq_length,
                        kv_cache_attention_bias=block_state.kv_cache_attention_bias,
                        vace_context=block_state.vace_context,
                        vace_context_scale=block_state.vace_context_scale,
                    )

        block_state.latents = denoised_pred

        self.set_block_state(state, block_state)
        return components, state
