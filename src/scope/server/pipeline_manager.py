"""Pipeline Manager for lazy loading and managing ML pipelines."""

import asyncio
import gc
import logging
import os
import threading
from enum import Enum
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def _is_env_true(var_name: str) -> bool:
    return os.getenv(var_name, "").strip().lower() in ("1", "true", "yes", "on")


def _maybe_disable_fp8_quantization(pipeline_id: str, quantization: Any) -> Any:
    if not _is_env_true("SCOPE_DISABLE_FP8_QUANTIZATION"):
        return quantization

    if quantization == "fp8_e4m3fn":
        logger.warning(
            "FP8 quantization requested for %s but disabled by SCOPE_DISABLE_FP8_QUANTIZATION=1; "
            "running unquantized.",
            pipeline_id,
        )
        return None

    return quantization


def get_device() -> torch.device:
    """Get the appropriate device (CUDA if available, CPU otherwise)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PipelineNotAvailableException(Exception):
    """Exception raised when pipeline is not available for processing."""

    pass


class PipelineStatus(Enum):
    """Pipeline loading status enumeration."""

    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"


class PipelineManager:
    """Manager for ML pipeline lifecycle."""

    def __init__(self):
        self._status = PipelineStatus.NOT_LOADED
        self._pipeline = None
        self._pipeline_id = None
        self._load_params = None
        self._error_message = None
        self._lock = threading.RLock()  # Single reentrant lock for all access

    @property
    def status(self) -> PipelineStatus:
        """Get current pipeline status."""
        return self._status

    @property
    def pipeline_id(self) -> str | None:
        """Get current pipeline ID."""
        return self._pipeline_id

    @property
    def error_message(self) -> str | None:
        """Get last error message."""
        return self._error_message

    def get_pipeline(self):
        """Get the loaded pipeline instance (thread-safe)."""
        with self._lock:
            if self._status != PipelineStatus.LOADED or self._pipeline is None:
                raise PipelineNotAvailableException(
                    f"Pipeline not available. Status: {self._status.value}"
                )
            return self._pipeline

    def get_status_info(self) -> dict[str, Any]:
        """Get detailed status information (thread-safe).

        Note: If status is ERROR, the error message is returned once and then cleared
        to prevent persistence across page reloads.
        """
        with self._lock:
            # Capture current state before clearing
            current_status = self._status
            error_message = self._error_message
            pipeline_id = self._pipeline_id
            load_params = self._load_params

            # Capture loaded LoRA adapters if pipeline exposes them
            loaded_lora_adapters = None
            if self._pipeline is not None and hasattr(
                self._pipeline, "loaded_lora_adapters"
            ):
                loaded_lora_adapters = getattr(
                    self._pipeline, "loaded_lora_adapters", None
                )

            # If there's an error, clear it after capturing it
            # This ensures errors don't persist across page reloads
            if self._status == PipelineStatus.ERROR and error_message:
                self._error_message = None
                # Reset status to NOT_LOADED after error is retrieved
                self._status = PipelineStatus.NOT_LOADED
                self._pipeline_id = None
                self._load_params = None

            # Return the captured state (with error status if it was an error)
            return {
                "status": current_status.value,
                "pipeline_id": pipeline_id,
                "load_params": load_params,
                "loaded_lora_adapters": loaded_lora_adapters,
                "error": error_message,
            }

    def peek_status_info(self) -> dict[str, Any]:
        """Get detailed status information without mutating manager state.

        Unlike get_status_info(), this does NOT clear errors or reset pipeline state.
        Intended for recorder start gating and other non-destructive status reads.
        """
        with self._lock:
            loaded_lora_adapters = None
            if self._pipeline is not None and hasattr(
                self._pipeline, "loaded_lora_adapters"
            ):
                loaded_lora_adapters = getattr(
                    self._pipeline, "loaded_lora_adapters", None
                )

            load_params = self._load_params
            if isinstance(load_params, dict):
                load_params = load_params.copy()

            return {
                "status": self._status.value,
                "pipeline_id": self._pipeline_id,
                "load_params": load_params,
                "loaded_lora_adapters": loaded_lora_adapters,
                "error": self._error_message,
            }

    async def get_pipeline_async(self):
        """Get the loaded pipeline instance (async wrapper)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_pipeline)

    async def get_status_info_async(self) -> dict[str, Any]:
        """Get detailed status information (async wrapper)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_status_info)

    async def load_pipeline(
        self, pipeline_id: str | None = None, load_params: dict | None = None
    ) -> bool:
        """
        Load a pipeline asynchronously.

        Args:
            pipeline_id: ID of pipeline to load. If None, uses PIPELINE env var.
            load_params: Pipeline-specific load parameters.

        Returns:
            bool: True if loaded successfully, False otherwise.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._load_pipeline_sync_wrapper, pipeline_id, load_params
        )

    def _load_pipeline_sync_wrapper(
        self, pipeline_id: str | None = None, load_params: dict | None = None
    ) -> bool:
        """Synchronous wrapper for pipeline loading with proper locking."""

        if pipeline_id is None:
            pipeline_id = os.getenv("PIPELINE", "longlive")

        with self._lock:
            # Normalize None to empty dict for comparison
            current_params = self._load_params or {}
            new_params = load_params or {}

            # If already loaded with same type and same params, return success
            if (
                self._status == PipelineStatus.LOADED
                and self._pipeline_id == pipeline_id
                and current_params == new_params
            ):
                logger.info(
                    f"Pipeline {pipeline_id} already loaded with matching parameters"
                )
                return True

            # If a different pipeline is loaded OR same pipeline with different params, unload it first
            if self._status == PipelineStatus.LOADED and (
                self._pipeline_id != pipeline_id or current_params != new_params
            ):
                self._unload_pipeline_unsafe()

            # If already loading, someone else is handling it
            if self._status == PipelineStatus.LOADING:
                logger.info("Pipeline already loading by another thread")
                return False

            # Mark as loading
            self._status = PipelineStatus.LOADING
            self._error_message = None

        # Release lock during slow loading operation
        logger.info(f"Loading pipeline: {pipeline_id}")

        try:
            # Load the pipeline synchronously (we're already in executor thread)
            pipeline = self._load_pipeline_implementation(pipeline_id, load_params)

            # Hold lock while updating state with loaded pipeline
            with self._lock:
                self._pipeline = pipeline
                self._pipeline_id = pipeline_id
                self._load_params = load_params
                self._status = PipelineStatus.LOADED

            logger.info(f"Pipeline {pipeline_id} loaded successfully")
            return True

        except Exception as e:
            from .models_config import get_models_dir

            models_dir = get_models_dir()
            error_msg = f"Failed to load pipeline {pipeline_id}: {e}"
            logger.error(
                f"{error_msg}. If this error persists, consider removing the models "
                f"directory '{models_dir}' and re-downloading models."
            )

            # Hold lock while updating state with error
            with self._lock:
                self._status = PipelineStatus.ERROR
                self._error_message = error_msg
                self._pipeline = None
                self._pipeline_id = None
                self._load_params = None

            return False

    def _get_vace_checkpoint_path(self) -> str:
        """Get the path to the VACE module checkpoint.

        Returns:
            str: Path to VACE module checkpoint file (contains only VACE weights)
        """
        from .models_config import get_model_file_path

        return str(
            get_model_file_path(
                "WanVideo_comfy/Wan2_1-VACE_module_1_3B_bf16.safetensors"
            )
        )

    def _configure_vace(self, config: dict, load_params: dict | None = None) -> None:
        """Configure VACE support for a pipeline.

        Adds vace_path to config and optionally extracts VACE-specific parameters
        from load_params (ref_images, vace_context_scale).

        Args:
            config: Pipeline configuration dict to modify
            load_params: Optional load parameters containing VACE settings
        """
        config["vace_path"] = self._get_vace_checkpoint_path()
        logger.debug(f"_configure_vace: Using VACE checkpoint at {config['vace_path']}")

        # Extract VACE-specific parameters from load_params if present
        if load_params:
            ref_images = load_params.get("ref_images", [])
            if ref_images:
                config["ref_images"] = ref_images
                config["vace_context_scale"] = load_params.get(
                    "vace_context_scale", 1.0
                )
                logger.info(
                    f"_configure_vace: VACE parameters from load_params: "
                    f"ref_images count={len(ref_images)}, "
                    f"vace_context_scale={config.get('vace_context_scale', 1.0)}"
                )

    def _apply_load_params(
        self,
        config: dict,
        load_params: dict | None,
        default_height: int,
        default_width: int,
        default_seed: int = 42,
        pipeline_id: str | None = None,
    ) -> None:
        """Extract and apply common load parameters (resolution, seed, LoRAs) to config.

        Args:
            config: Pipeline config dict to update
            load_params: Load parameters dict (may contain height, width, seed, loras, lora_merge_mode)
            default_height: Default height if not in load_params
            default_width: Default width if not in load_params
            default_seed: Default seed if not in load_params
        """
        height = default_height
        width = default_width
        seed = default_seed
        loras = None
        lora_merge_mode = "permanent_merge"

        if load_params:
            height = load_params.get("height", default_height)
            width = load_params.get("width", default_width)
            seed = load_params.get("seed", default_seed)
            loras = load_params.get("loras", None)
            lora_merge_mode = load_params.get("lora_merge_mode", lora_merge_mode)

        style_swap_mode = _is_env_true("STYLE_SWAP_MODE")
        if style_swap_mode and pipeline_id not in (None, "passthrough"):
            from scope.realtime.style_manifest import (
                StyleRegistry,
                canonicalize_lora_path,
                get_style_dirs,
            )

            requested_style_names: set[str] | None = None
            if raw := os.getenv("SCOPE_PRELOAD_LORAS"):
                requested_style_names = {s.strip() for s in raw.split(",") if s.strip()}

            registry = StyleRegistry()
            registry.load_from_style_dirs()

            style_lora_paths: list[str] = []
            missing_count = 0
            for style_name in registry.list_styles():
                if requested_style_names and style_name not in requested_style_names:
                    continue
                manifest = registry.get(style_name)
                if not manifest or not manifest.lora_path:
                    continue
                canonical = canonicalize_lora_path(manifest.lora_path)
                if not canonical:
                    continue
                if not Path(canonical).exists():
                    missing_count += 1
                    logger.warning(
                        "STYLE_SWAP_MODE=1: style '%s' LoRA not found at %s, skipping",
                        style_name,
                        canonical,
                    )
                    continue
                if canonical not in style_lora_paths:
                    style_lora_paths.append(canonical)

            style_loras = [
                {"path": p, "scale": 0.0, "merge_mode": "runtime_peft"}
                for p in style_lora_paths
            ]

            explicit_loras: list[dict[str, Any]] = []
            if isinstance(loras, list):
                for lora_cfg in loras:
                    if not isinstance(lora_cfg, dict):
                        continue
                    canonical = canonicalize_lora_path(lora_cfg.get("path"))
                    if not canonical:
                        continue
                    if not Path(canonical).exists():
                        missing_count += 1
                        logger.warning(
                            "STYLE_SWAP_MODE=1: requested LoRA not found at %s, skipping",
                            canonical,
                        )
                        continue
                    normalized = dict(lora_cfg)
                    normalized["path"] = canonical
                    explicit_loras.append(normalized)

            loras_by_path: dict[str, dict[str, Any]] = {}
            for lora_cfg in style_loras:
                loras_by_path[lora_cfg["path"]] = lora_cfg
            for lora_cfg in explicit_loras:
                path = lora_cfg.get("path")
                if isinstance(path, str) and path:
                    loras_by_path[path] = lora_cfg

            loras = list(loras_by_path.values()) or None
            if loras:
                logger.info(
                    "STYLE_SWAP_MODE=1: preloading %d LoRA(s) (styles_dir=%s, missing=%d)",
                    len(loras),
                    [str(p) for p in get_style_dirs()],
                    missing_count,
                )
            else:
                logger.info(
                    "STYLE_SWAP_MODE=1: no LoRAs discovered to preload (styles_dir=%s, missing=%d)",
                    [str(p) for p in get_style_dirs()],
                    missing_count,
                )

            if lora_merge_mode != "runtime_peft":
                logger.info(
                    "STYLE_SWAP_MODE=1: forcing lora_merge_mode runtime_peft (was %s)",
                    lora_merge_mode,
                )
            lora_merge_mode = "runtime_peft"

        config["height"] = height
        config["width"] = width
        config["seed"] = seed
        if loras:
            config["loras"] = loras
        # Pass merge_mode directly to mixin, not via config
        config["_lora_merge_mode"] = lora_merge_mode

    def _unload_pipeline_unsafe(self):
        """Unload the current pipeline. Must be called with lock held."""
        if self._pipeline:
            logger.info(f"Unloading pipeline: {self._pipeline_id}")

        # Change status and pipeline atomically
        self._status = PipelineStatus.NOT_LOADED
        self._pipeline = None
        self._pipeline_id = None
        self._load_params = None
        self._error_message = None

        # Cleanup resources
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("CUDA cache cleared")
            except Exception as e:
                logger.warning(f"CUDA cleanup failed: {e}")

    def _load_pipeline_implementation(
        self, pipeline_id: str, load_params: dict | None = None
    ):
        """Synchronous pipeline loading (runs in thread executor)."""
        if pipeline_id == "streamdiffusionv2":
            from scope.core.pipelines import (
                StreamDiffusionV2Pipeline,
            )

            from .models_config import get_model_file_path, get_models_dir

            models_dir = get_models_dir()
            config = OmegaConf.create(
                {
                    "model_dir": str(models_dir),
                    "generator_path": str(
                        get_model_file_path(
                            "StreamDiffusionV2/wan_causal_dmd_v2v/model.pt"
                        )
                    ),
                    "text_encoder_path": str(
                        get_model_file_path(
                            "WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors"
                        )
                    ),
                    "tokenizer_path": str(
                        get_model_file_path("Wan2.1-T2V-1.3B/google/umt5-xxl")
                    ),
                }
            )

            # Configure VACE support if enabled in load_params (default: True)
            # Note: VACE is not available for StreamDiffusion in video mode (enforced by frontend)
            vace_enabled = True
            if load_params:
                vace_enabled = load_params.get("vace_enabled", True)

            if vace_enabled:
                self._configure_vace(config, load_params)
            else:
                logger.info("VACE disabled by load_params, skipping VACE configuration")

            # Apply load parameters (resolution, seed, LoRAs) to config
            self._apply_load_params(
                config,
                load_params,
                default_height=512,
                default_width=512,
                default_seed=42,
                pipeline_id=pipeline_id,
            )

            quantization = None
            if load_params:
                quantization = load_params.get("quantization", None)
                quantization = _maybe_disable_fp8_quantization(pipeline_id, quantization)
                load_params["quantization"] = quantization

            pipeline = StreamDiffusionV2Pipeline(
                config,
                quantization=quantization,
                device=torch.device("cuda"),
                dtype=torch.bfloat16,
            )
            logger.info("StreamDiffusionV2 pipeline initialized")
            return pipeline

        elif pipeline_id == "passthrough":
            from scope.core.pipelines import PassthroughPipeline

            # Use load parameters for resolution, default to 512x512
            height = 512
            width = 512
            if load_params:
                height = load_params.get("height", 512)
                width = load_params.get("width", 512)

            pipeline = PassthroughPipeline(
                height=height,
                width=width,
                device=get_device(),
                dtype=torch.bfloat16,
            )
            logger.info("Passthrough pipeline initialized")
            return pipeline

        elif pipeline_id == "longlive":
            from scope.core.pipelines import LongLivePipeline

            from .models_config import get_model_file_path, get_models_dir

            models_dir = get_models_dir()
            config = OmegaConf.create(
                {
                    "model_dir": str(models_dir),
                    "generator_path": str(
                        get_model_file_path("LongLive-1.3B/models/longlive_base.pt")
                    ),
                    "lora_path": str(
                        get_model_file_path("LongLive-1.3B/models/lora.pt")
                    ),
                    "text_encoder_path": str(
                        get_model_file_path(
                            "WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors"
                        )
                    ),
                    "tokenizer_path": str(
                        get_model_file_path("Wan2.1-T2V-1.3B/google/umt5-xxl")
                    ),
                }
            )

            # Configure VACE support if enabled in load_params (default: True)
            vace_enabled = True
            if load_params:
                vace_enabled = load_params.get("vace_enabled", True)

            if vace_enabled:
                self._configure_vace(config, load_params)
            else:
                logger.info("VACE disabled by load_params, skipping VACE configuration")

            # Apply load parameters (resolution, seed, LoRAs) to config
            self._apply_load_params(
                config,
                load_params,
                default_height=320,
                default_width=576,
                default_seed=42,
                pipeline_id=pipeline_id,
            )

            quantization = None
            if load_params:
                quantization = load_params.get("quantization", None)
                quantization = _maybe_disable_fp8_quantization(pipeline_id, quantization)
                load_params["quantization"] = quantization

            pipeline = LongLivePipeline(
                config,
                quantization=quantization,
                device=torch.device("cuda"),
                dtype=torch.bfloat16,
            )
            logger.info("LongLive pipeline initialized")
            return pipeline

        elif pipeline_id == "krea-realtime-video":
            from scope.core.pipelines import (
                KreaRealtimeVideoPipeline,
            )

            from .models_config import get_model_file_path, get_models_dir

            config = OmegaConf.create(
                {
                    "model_dir": str(get_models_dir()),
                    "generator_path": str(
                        get_model_file_path(
                            "krea-realtime-video/krea-realtime-video-14b.safetensors"
                        )
                    ),
                    "text_encoder_path": str(
                        get_model_file_path(
                            "WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors"
                        )
                    ),
                    "tokenizer_path": str(
                        get_model_file_path("Wan2.1-T2V-1.3B/google/umt5-xxl")
                    ),
                    "vae_path": str(
                        get_model_file_path("Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")
                    ),
                }
            )

            # Apply load parameters (resolution, seed, LoRAs) to config
            self._apply_load_params(
                config,
                load_params,
                default_height=512,
                default_width=512,
                default_seed=42,
                pipeline_id=pipeline_id,
            )

            quantization = None
            if load_params:
                quantization = load_params.get("quantization", None)
                quantization = _maybe_disable_fp8_quantization(pipeline_id, quantization)
                load_params["quantization"] = quantization

            # torch.compile is opt-in for non-Hopper GPUs because it can increase
            # startup latency and has architecture-specific sharp edges (especially
            # when quantization is enabled).
            compile_env = os.getenv("SCOPE_COMPILE_KREA_PIPELINE")
            compile_explicit = compile_env is not None and compile_env != ""
            if compile_explicit:
                compile_enabled = compile_env.lower() in ("1", "true", "yes", "on")
            else:
                # Only compile diffusion model for hopper by default.
                compile_enabled = any(
                    x in torch.cuda.get_device_name(0).lower() for x in ("h100", "hopper")
                )

            # Known issue: FP8 (torchao) + torch.compile has been brittle on some architectures;
            # avoid by default, but allow explicit opt-in (and keep enabled on SM100 where it is
            # a large steady-state win).
            if compile_enabled and quantization is not None:
                allow_env = os.getenv("SCOPE_COMPILE_KREA_PIPELINE_ALLOW_QUANTIZATION", "")
                allow_compile_with_quantization = allow_env.lower() in ("1", "true", "yes", "on")
                try:
                    is_sm100 = torch.cuda.get_device_capability(0) == (10, 0)
                except Exception:
                    is_sm100 = False

                if not (allow_compile_with_quantization or is_sm100):
                    logger.info(
                        "Disabling torch.compile for krea-realtime-video (quantization=%s). "
                        "Set SCOPE_COMPILE_KREA_PIPELINE_ALLOW_QUANTIZATION=1 to override.",
                        quantization,
                    )
                    compile_enabled = False

            pipeline = KreaRealtimeVideoPipeline(
                config,
                quantization=quantization,
                compile=compile_enabled,
                device=torch.device("cuda"),
                dtype=torch.bfloat16,
            )
            logger.info("krea-realtime-video pipeline initialized")
            return pipeline

        elif pipeline_id == "reward-forcing":
            from scope.core.pipelines import (
                RewardForcingPipeline,
            )

            from .models_config import get_model_file_path, get_models_dir

            config = OmegaConf.create(
                {
                    "model_dir": str(get_models_dir()),
                    "generator_path": str(
                        get_model_file_path("Reward-Forcing-T2V-1.3B/rewardforcing.pt")
                    ),
                    "text_encoder_path": str(
                        get_model_file_path(
                            "WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors"
                        )
                    ),
                    "tokenizer_path": str(
                        get_model_file_path("Wan2.1-T2V-1.3B/google/umt5-xxl")
                    ),
                    "vae_path": str(
                        get_model_file_path("Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")
                    ),
                }
            )

            # Configure VACE support if enabled in load_params (default: True)
            vace_enabled = True
            if load_params:
                vace_enabled = load_params.get("vace_enabled", True)

            if vace_enabled:
                self._configure_vace(config, load_params)
            else:
                logger.info("VACE disabled by load_params, skipping VACE configuration")

            # Apply load parameters (resolution, seed, LoRAs) to config
            self._apply_load_params(
                config,
                load_params,
                default_height=320,
                default_width=576,
                default_seed=42,
                pipeline_id=pipeline_id,
            )

            quantization = None
            if load_params:
                quantization = load_params.get("quantization", None)
                quantization = _maybe_disable_fp8_quantization(pipeline_id, quantization)
                load_params["quantization"] = quantization

            pipeline = RewardForcingPipeline(
                config,
                quantization=quantization,
                device=torch.device("cuda"),
                dtype=torch.bfloat16,
            )
            logger.info("RewardForcing pipeline initialized")
            return pipeline

        else:
            raise ValueError(f"Invalid pipeline ID: {pipeline_id}")

    def unload_pipeline(self):
        """Unload the current pipeline (thread-safe)."""
        with self._lock:
            self._unload_pipeline_unsafe()

    def is_loaded(self) -> bool:
        """Check if pipeline is loaded and ready (thread-safe)."""
        with self._lock:
            return self._status == PipelineStatus.LOADED
