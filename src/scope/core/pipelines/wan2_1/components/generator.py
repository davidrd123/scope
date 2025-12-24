# Modified from https://github.com/guandeh17/Self-Forcing
import atexit
import inspect
import json
import logging
import os
import time
import types
from collections import defaultdict

import torch

from scope.core.pipelines.utils import load_state_dict

from .scheduler import FlowMatchScheduler, SchedulerInterface

logger = logging.getLogger(__name__)

_PROFILE_GENERATOR_STEPS = os.getenv("PROFILE_GENERATOR_STEPS", "0") == "1"
_PROFILE_GENERATOR_STEPS_JSON = os.getenv("PROFILE_GENERATOR_STEPS_JSON")
_generator_step_cpu_ms = defaultdict(float)
_generator_step_gpu_ms = defaultdict(float)
_generator_step_counts = defaultdict(int)
_generator_step_meta: dict[str, object] = {}


def _should_profile_generator_steps() -> bool:
    if not _PROFILE_GENERATOR_STEPS:
        return False
    if hasattr(torch, "compiler") and hasattr(torch.compiler, "is_compiling"):
        if torch.compiler.is_compiling():
            return False
    return True


def _generator_step_profile_report() -> None:
    if not _generator_step_counts:
        return

    total_cpu_ms = sum(_generator_step_cpu_ms.values())
    total_gpu_ms = sum(_generator_step_gpu_ms.values())

    logger.info("=== Generator Profiling Report ===")
    for name, cpu_ms in sorted(_generator_step_cpu_ms.items(), key=lambda kv: -kv[1]):
        calls = _generator_step_counts.get(name, 0)
        gpu_ms = _generator_step_gpu_ms.get(name, 0.0)
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

    if _PROFILE_GENERATOR_STEPS_JSON:
        meta = dict(_generator_step_meta)
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
                    "cpu_ms": _generator_step_cpu_ms.get(name, 0.0),
                    "gpu_ms": _generator_step_gpu_ms.get(name, 0.0),
                    "calls": int(_generator_step_counts.get(name, 0)),
                }
                for name in sorted(_generator_step_counts.keys())
            },
        }
        try:
            with open(_PROFILE_GENERATOR_STEPS_JSON, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
            logger.info("Wrote generator step profile JSON: %s", _PROFILE_GENERATOR_STEPS_JSON)
        except Exception as e:
            logger.warning("Failed writing generator step profile JSON (%s): %s", _PROFILE_GENERATOR_STEPS_JSON, e)


atexit.register(_generator_step_profile_report)


def reset_generator_step_profile() -> None:
    """Clear accumulated generator profiling counters."""
    _generator_step_cpu_ms.clear()
    _generator_step_gpu_ms.clear()
    _generator_step_counts.clear()
    _generator_step_meta.clear()


class _ProfileGeneratorStep:
    __slots__ = ("name", "_cpu_start", "_start", "_end")

    def __init__(self, name: str):
        self.name = name
        self._cpu_start = None
        self._start = None
        self._end = None

    def __enter__(self):
        if not _should_profile_generator_steps():
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
            _generator_step_gpu_ms[self.name] += self._start.elapsed_time(self._end)

        _generator_step_cpu_ms[self.name] += (time.perf_counter() - self._cpu_start) * 1000.0
        _generator_step_counts[self.name] += 1


def filter_causal_model_cls_config(causal_model_cls, config):
    # Filter config to only include parameters accepted by the model's __init__
    sig = inspect.signature(causal_model_cls.__init__)
    config = {k: v for k, v in config.items() if k in sig.parameters}
    return config


class WanDiffusionWrapper(torch.nn.Module):
    def __init__(
        self,
        causal_model_cls,
        model_name="Wan2.1-T2V-1.3B",
        timestep_shift=8.0,
        local_attn_size=-1,
        sink_size=0,
        model_dir: str | None = None,
        generator_path: str | None = None,
        generator_model_name: str | None = None,
        **model_kwargs,
    ):
        super().__init__()

        # Use provided model_dir or default to "wan_models"
        model_dir = model_dir if model_dir is not None else "wan_models"
        model_path = os.path.join(model_dir, f"{model_name}/")

        if generator_path:
            config_path = os.path.join(model_path, "config.json")
            with open(config_path) as f:
                config = json.load(f)

            config.update({"local_attn_size": local_attn_size, "sink_size": sink_size})
            # Merge in additional model-specific kwargs (e.g., vace_in_dim for VACE models)
            config.update(model_kwargs)

            state_dict = load_state_dict(generator_path)
            # Handle case where the dict with required keys is nested under a specific key
            # eg state_dict["generator"]
            if generator_model_name is not None:
                state_dict = state_dict[generator_model_name]

            # Remove 'model.' prefix if present (from wrapped models)
            if all(k.startswith("model.") for k in state_dict.keys()):
                state_dict = {
                    k.replace("model.", "", 1): v for k, v in state_dict.items()
                }

            with torch.device("meta"):
                self.model = causal_model_cls(
                    **filter_causal_model_cls_config(causal_model_cls, config)
                )

            # HACK!
            # Store freqs shape before it becomes problematic
            freqs_shape = (
                self.model.freqs.shape if hasattr(self.model, "freqs") else None
            )

            # Move model to CPU first to materialize all buffers and parameters
            self.model = self.model.to_empty(device="cpu")
            # Then load the state dict weights
            # Use strict=False to allow partial loading (e.g., VACE model with non-VACE checkpoint)
            self.model.load_state_dict(state_dict, assign=True, strict=False)

            # HACK!
            # Reinitialize self.freqs properly on CPU (it's not in state_dict)
            if freqs_shape is not None and hasattr(self.model, "freqs"):
                # Get model dimensions to recreate freqs
                d = self.model.dim // self.model.num_heads

                # From Wan2.1 model.py
                def rope_params(max_seq_len, dim, theta=10000):
                    assert dim % 2 == 0
                    freqs = torch.outer(
                        torch.arange(max_seq_len),
                        1.0
                        / torch.pow(
                            theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)
                        ),
                    )
                    freqs = torch.polar(torch.ones_like(freqs), freqs)
                    return freqs

                self.model.freqs = torch.cat(
                    [
                        rope_params(1024, d - 4 * (d // 6)),
                        rope_params(1024, 2 * (d // 6)),
                        rope_params(1024, 2 * (d // 6)),
                    ],
                    dim=1,
                )
        else:
            from_pretrained_config = {
                "local_attn_size": local_attn_size,
                "sink_size": sink_size,
            }
            # Merge in additional model-specific kwargs (e.g., vace_in_dim for VACE models)
            from_pretrained_config.update(model_kwargs)
            self.model = causal_model_cls.from_pretrained(
                model_path,
                **filter_causal_model_cls_config(
                    causal_model_cls,
                    from_pretrained_config,
                ),
            )

        self.model.eval()
        self.model.requires_grad_(False)

        # For non-causal diffusion, all frames share the same timestep
        self.uniform_timestep = False

        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)

        # self.seq_len = 1560 * local_attn_size if local_attn_size != -1 else 32760 # [1, 21, 16, 60, 104]
        self.seq_len = (
            1560 * local_attn_size if local_attn_size > 21 else 32760
        )  # [1, 21, 16, 60, 104]
        self.post_init()

    def _convert_flow_pred_to_x0(
        self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        see derivations https://chatgpt.com/share/67bf8589-3d04-8008-bc6e-4cf1a24e2d0e
        """
        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = [
            x.double().to(flow_pred.device)
            for x in [flow_pred, xt, self.scheduler.sigmas, self.scheduler.timesteps]
        ]

        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1
        )
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    @staticmethod
    def _convert_x0_to_flow_pred(
        scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert x0 prediction to flow matching's prediction.
        x0_pred: the x0 prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = (x_t - x_0) / sigma_t
        """
        # use higher precision for calculations
        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = [
            x.double().to(x0_pred.device)
            for x in [x0_pred, xt, scheduler.sigmas, scheduler.timesteps]
        ]
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1
        )
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)

    def _call_model(self, *args, **kwargs):
        # HACK!
        # __call__() and forward() accept *args, **kwargs so inspection doesn't tell us anything
        # As a workaround we inspect the internal _forward_inference() function to determine what the accepted params are
        # This allows us to filter out params that might not work with the underlying CausalWanModel impl
        sig = inspect.signature(self.model._forward_inference)
        accepted = {
            name: value for name, value in kwargs.items() if name in sig.parameters
        }
        return self.model(*args, **accepted)

    def forward(
        self,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: dict,
        timestep: torch.Tensor,
        kv_cache: list[dict] | None = None,
        crossattn_cache: list[dict] | None = None,
        current_start: int | None = None,
        current_end: int | None = None,
        classify_mode: bool | None = False,
        concat_time_embeddings: bool | None = False,
        clean_x: torch.Tensor | None = None,
        aug_t: torch.Tensor | None = None,
        cache_start: int | None = None,
        kv_cache_attention_bias: float = 1.0,
        vace_context: torch.Tensor | None = None,
        vace_context_scale: float = 1.0,
    ) -> torch.Tensor:
        prompt_embeds = conditional_dict["prompt_embeds"]

        # [B, F] -> [B]
        if self.uniform_timestep:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep

        if _should_profile_generator_steps():
            _generator_step_meta.update(
                {
                    "noisy_shape": list(noisy_image_or_video.shape),
                    "noisy_dtype": str(noisy_image_or_video.dtype),
                    "noisy_device": str(noisy_image_or_video.device),
                    "timestep_shape": list(timestep.shape),
                    "timestep_dtype": str(timestep.dtype),
                    "prompt_embeds_shape": list(prompt_embeds.shape),
                    "prompt_embeds_dtype": str(prompt_embeds.dtype),
                    "prompt_embeds_device": str(prompt_embeds.device),
                    "kv_cache_present": bool(kv_cache is not None),
                    "kv_cache_len": int(len(kv_cache)) if kv_cache is not None else 0,
                    "crossattn_cache_present": bool(crossattn_cache is not None),
                    "crossattn_cache_len": int(len(crossattn_cache)) if crossattn_cache is not None else 0,
                    "current_start": int(current_start) if current_start is not None else None,
                    "current_end": int(current_end) if current_end is not None else None,
                    "cache_start": int(cache_start) if cache_start is not None else None,
                    "kv_cache_attention_bias": float(kv_cache_attention_bias),
                }
            )

        logits = None
        # X0 prediction
        if kv_cache is not None:
            with _ProfileGeneratorStep("call_model_kv_cache"):
                flow_pred = self._call_model(
                    noisy_image_or_video.permute(0, 2, 1, 3, 4),
                    t=input_timestep,
                    context=prompt_embeds,
                    seq_len=self.seq_len,
                    kv_cache=kv_cache,
                    crossattn_cache=crossattn_cache,
                    current_start=current_start,
                    current_end=current_end,
                    cache_start=cache_start,
                    kv_cache_attention_bias=kv_cache_attention_bias,
                    vace_context=vace_context,
                    vace_context_scale=vace_context_scale,
                ).permute(0, 2, 1, 3, 4)
        else:
            if clean_x is not None:
                # teacher forcing
                with _ProfileGeneratorStep("call_model_teacher_forcing"):
                    flow_pred = self._call_model(
                        noisy_image_or_video.permute(0, 2, 1, 3, 4),
                        t=input_timestep,
                        context=prompt_embeds,
                        seq_len=self.seq_len,
                        clean_x=clean_x.permute(0, 2, 1, 3, 4),
                        aug_t=aug_t,
                        vace_context=vace_context,
                        vace_context_scale=vace_context_scale,
                    ).permute(0, 2, 1, 3, 4)
            else:
                if classify_mode:
                    with _ProfileGeneratorStep("call_model_classify"):
                        flow_pred, logits = self._call_model(
                            noisy_image_or_video.permute(0, 2, 1, 3, 4),
                            t=input_timestep,
                            context=prompt_embeds,
                            seq_len=self.seq_len,
                            classify_mode=True,
                            register_tokens=self._register_tokens,
                            cls_pred_branch=self._cls_pred_branch,
                            gan_ca_blocks=self._gan_ca_blocks,
                            concat_time_embeddings=concat_time_embeddings,
                            vace_context=vace_context,
                            vace_context_scale=vace_context_scale,
                        )
                        flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
                else:
                    with _ProfileGeneratorStep("call_model_no_cache"):
                        flow_pred = self._call_model(
                            noisy_image_or_video.permute(0, 2, 1, 3, 4),
                            t=input_timestep,
                            context=prompt_embeds,
                            seq_len=self.seq_len,
                            vace_context=vace_context,
                            vace_context_scale=vace_context_scale,
                        ).permute(0, 2, 1, 3, 4)

        with _ProfileGeneratorStep("convert_flow_pred_to_x0"):
            pred_x0 = self._convert_flow_pred_to_x0(
                flow_pred=flow_pred.flatten(0, 1),
                xt=noisy_image_or_video.flatten(0, 1),
                timestep=timestep.flatten(0, 1),
            ).unflatten(0, flow_pred.shape[:2])

        if logits is not None:
            return flow_pred, pred_x0, logits

        return flow_pred, pred_x0

    def get_scheduler(self) -> SchedulerInterface:
        """
        Update the current scheduler with the interface's static method
        """
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise, scheduler
        )
        scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0, scheduler
        )
        scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0, scheduler
        )
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        """
        A few custom initialization steps that should be called after the object is created.
        Currently, the only one we have is to bind a few methods to scheduler.
        We can gradually add more methods here if needed.
        """
        self.get_scheduler()
