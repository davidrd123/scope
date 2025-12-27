"""Unified Wan VAE wrapper with streaming and batch encoding/decoding."""

import atexit
import json
import logging
import os
import time
from collections import defaultdict

import torch

from .constants import WAN_VAE_LATENT_MEAN, WAN_VAE_LATENT_STD
from .modules.vae import _video_vae

# Default filename for standard Wan2.1 VAE checkpoint
DEFAULT_VAE_FILENAME = "Wan2.1_VAE.pth"

logger = logging.getLogger(__name__)

_PROFILE_WANVAE_DECODE = os.getenv("PROFILE_WANVAE_DECODE", "0") == "1"
_PROFILE_WANVAE_DECODE_JSON = os.getenv("PROFILE_WANVAE_DECODE_JSON")
_WANVAE_DECODE_CHANNELS_LAST_3D = os.getenv("WANVAE_DECODE_CHANNELS_LAST_3D", "0") == "1"
_wanvae_decode_cpu_ms = defaultdict(float)
_wanvae_decode_gpu_ms = defaultdict(float)
_wanvae_decode_counts = defaultdict(int)
_wanvae_decode_meta: dict[str, object] = {}

_PROFILE_WANVAE_ENCODE = os.getenv("PROFILE_WANVAE_ENCODE", "0") == "1"
_PROFILE_WANVAE_ENCODE_JSON = os.getenv("PROFILE_WANVAE_ENCODE_JSON")
_WANVAE_ENCODE_CHANNELS_LAST_3D = os.getenv("WANVAE_ENCODE_CHANNELS_LAST_3D", "0") == "1"
_wanvae_encode_cpu_ms = defaultdict(float)
_wanvae_encode_gpu_ms = defaultdict(float)
_wanvae_encode_counts = defaultdict(int)
_wanvae_encode_meta: dict[str, object] = {}


def _should_profile_wanvae_decode() -> bool:
    if not _PROFILE_WANVAE_DECODE:
        return False
    if hasattr(torch, "compiler") and hasattr(torch.compiler, "is_compiling"):
        if torch.compiler.is_compiling():
            return False
    return True


def _wanvae_decode_profile_report() -> None:
    if not _wanvae_decode_counts:
        return

    total_cpu_ms = sum(_wanvae_decode_cpu_ms.values())
    total_gpu_ms = sum(_wanvae_decode_gpu_ms.values())
    logger.info("=== WanVAE Decode Profiling Report ===")
    for name, cpu_ms in sorted(_wanvae_decode_cpu_ms.items(), key=lambda kv: -kv[1]):
        calls = _wanvae_decode_counts.get(name, 0)
        gpu_ms = _wanvae_decode_gpu_ms.get(name, 0.0)
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

    if _PROFILE_WANVAE_DECODE_JSON:
        meta = dict(_wanvae_decode_meta)
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
                    "cpu_ms": _wanvae_decode_cpu_ms.get(name, 0.0),
                    "gpu_ms": _wanvae_decode_gpu_ms.get(name, 0.0),
                    "calls": int(_wanvae_decode_counts.get(name, 0)),
                }
                for name in sorted(_wanvae_decode_counts.keys())
            },
        }
        try:
            with open(_PROFILE_WANVAE_DECODE_JSON, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
            logger.info("Wrote WanVAE decode profile JSON: %s", _PROFILE_WANVAE_DECODE_JSON)
        except Exception as e:
            logger.warning(
                "Failed writing WanVAE decode profile JSON (%s): %s",
                _PROFILE_WANVAE_DECODE_JSON,
                e,
            )


atexit.register(_wanvae_decode_profile_report)


def _should_profile_wanvae_encode() -> bool:
    if not _PROFILE_WANVAE_ENCODE:
        return False
    if hasattr(torch, "compiler") and hasattr(torch.compiler, "is_compiling"):
        if torch.compiler.is_compiling():
            return False
    return True


def _wanvae_encode_profile_report() -> None:
    if not _wanvae_encode_counts:
        return

    total_cpu_ms = sum(_wanvae_encode_cpu_ms.values())
    total_gpu_ms = sum(_wanvae_encode_gpu_ms.values())
    logger.info("=== WanVAE Encode Profiling Report ===")
    for name, cpu_ms in sorted(_wanvae_encode_cpu_ms.items(), key=lambda kv: -kv[1]):
        calls = _wanvae_encode_counts.get(name, 0)
        gpu_ms = _wanvae_encode_gpu_ms.get(name, 0.0)
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

    if _PROFILE_WANVAE_ENCODE_JSON:
        meta = dict(_wanvae_encode_meta)
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
                    "cpu_ms": _wanvae_encode_cpu_ms.get(name, 0.0),
                    "gpu_ms": _wanvae_encode_gpu_ms.get(name, 0.0),
                    "calls": int(_wanvae_encode_counts.get(name, 0)),
                }
                for name in sorted(_wanvae_encode_counts.keys())
            },
        }
        try:
            with open(_PROFILE_WANVAE_ENCODE_JSON, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
            logger.info("Wrote WanVAE encode profile JSON: %s", _PROFILE_WANVAE_ENCODE_JSON)
        except Exception as e:
            logger.warning(
                "Failed writing WanVAE encode profile JSON (%s): %s",
                _PROFILE_WANVAE_ENCODE_JSON,
                e,
            )


atexit.register(_wanvae_encode_profile_report)


def reset_wanvae_decode_profile() -> None:
    """Clear accumulated WanVAEWrapper.decode_to_pixel profiling counters."""
    _wanvae_decode_cpu_ms.clear()
    _wanvae_decode_gpu_ms.clear()
    _wanvae_decode_counts.clear()
    _wanvae_decode_meta.clear()


def reset_wanvae_encode_profile() -> None:
    """Clear accumulated WanVAEWrapper.encode_to_latent profiling counters."""
    _wanvae_encode_cpu_ms.clear()
    _wanvae_encode_gpu_ms.clear()
    _wanvae_encode_counts.clear()
    _wanvae_encode_meta.clear()


class _ProfileWanVAEDecode:
    __slots__ = ("name", "_cpu_start", "_start", "_end")

    def __init__(self, name: str):
        self.name = name
        self._cpu_start = None
        self._start = None
        self._end = None

    def __enter__(self):
        if not _should_profile_wanvae_decode():
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
            _wanvae_decode_gpu_ms[self.name] += self._start.elapsed_time(self._end)

        _wanvae_decode_cpu_ms[self.name] += (time.perf_counter() - self._cpu_start) * 1000.0
        _wanvae_decode_counts[self.name] += 1


class _ProfileWanVAEEncode:
    __slots__ = ("name", "_cpu_start", "_start", "_end")

    def __init__(self, name: str):
        self.name = name
        self._cpu_start = None
        self._start = None
        self._end = None

    def __enter__(self):
        if not _should_profile_wanvae_encode():
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
            _wanvae_encode_gpu_ms[self.name] += self._start.elapsed_time(self._end)

        _wanvae_encode_cpu_ms[self.name] += (time.perf_counter() - self._cpu_start) * 1000.0
        _wanvae_encode_counts[self.name] += 1


class WanVAEWrapper(torch.nn.Module):
    """Unified VAE wrapper for Wan2.1 models.

    This VAE supports both streaming (cached) and batch encoding/decoding modes.
    Normalization is always applied during encoding for consistent latent distributions.
    """

    def __init__(
        self,
        model_dir: str = "wan_models",
        model_name: str = "Wan2.1-T2V-1.3B",
        vae_path: str | None = None,
    ):
        super().__init__()

        # Determine paths with priority: explicit vae_path > model_dir/model_name default
        if vae_path is None:
            vae_path = os.path.join(model_dir, model_name, DEFAULT_VAE_FILENAME)

        self.register_buffer(
            "mean", torch.tensor(WAN_VAE_LATENT_MEAN, dtype=torch.float32)
        )
        self.register_buffer(
            "std", torch.tensor(WAN_VAE_LATENT_STD, dtype=torch.float32)
        )
        self.z_dim = 16

        self.model = (
            _video_vae(
                pretrained_path=vae_path,
                z_dim=self.z_dim,
            )
            .eval()
            .requires_grad_(False)
        )

    def _get_scale(self, device: torch.device, dtype: torch.dtype) -> list:
        """Get normalization scale parameters on the correct device/dtype."""
        return [
            self.mean.to(device=device, dtype=dtype),
            1.0 / self.std.to(device=device, dtype=dtype),
        ]

    def _apply_encoding_normalization(
        self, latent: torch.Tensor, scale: list
    ) -> torch.Tensor:
        """Apply normalization to encoded latents."""
        if isinstance(scale[0], torch.Tensor):
            return (latent - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(
                1, self.z_dim, 1, 1, 1
            )
        return (latent - scale[0]) * scale[1]

    def _create_encoder_cache(self) -> list:
        """Create a fresh encoder feature cache."""
        return [None] * 55

    def encode_to_latent(
        self,
        pixel: torch.Tensor,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Encode video pixels to latents.

        Args:
            pixel: Input video tensor [batch, channels, frames, height, width]
            use_cache: If True, use streaming encode (maintains cache state).
                      If False, use batch encode with a temporary cache.

        Returns:
            Latent tensor [batch, frames, channels, height, width]
        """
        if _should_profile_wanvae_encode():
            _wanvae_encode_meta.update(
                {
                    "use_cache": bool(use_cache),
                    "pixel_shape": list(pixel.shape),
                    "pixel_dtype": str(pixel.dtype),
                    "pixel_device": str(pixel.device),
                    "WANVAE_ENCODE_CHANNELS_LAST_3D": bool(_WANVAE_ENCODE_CHANNELS_LAST_3D),
                }
            )

        with _ProfileWanVAEEncode("prep_memory_format"):
            if _WANVAE_ENCODE_CHANNELS_LAST_3D and pixel.is_cuda and pixel.ndim == 5:
                pixel = pixel.contiguous(memory_format=torch.channels_last_3d)

        with _ProfileWanVAEEncode("get_scale"):
            device, dtype = pixel.device, pixel.dtype
            scale = self._get_scale(device, dtype)

        if use_cache:
            with _ProfileWanVAEEncode("stream_encode"):
                latent = self.model.stream_encode(pixel)
            with _ProfileWanVAEEncode("normalize"):
                # stream_encode returns unnormalized.
                latent = self._apply_encoding_normalization(latent, scale)
        else:
            with _ProfileWanVAEEncode("encode_with_cache"):
                latent = self._encode_with_cache(pixel, scale, self._create_encoder_cache())

        with _ProfileWanVAEEncode("post_permute"):
            # [batch, channels, frames, h, w] -> [batch, frames, channels, h, w]
            return latent.permute(0, 2, 1, 3, 4)

    def _encode_with_cache(
        self, x: torch.Tensor, scale: list, feat_cache: list
    ) -> torch.Tensor:
        """Encode using an explicit cache without affecting internal streaming state.

        This follows the approach from the spike branch where the cache is passed
        explicitly, allowing one-time encodes for operations like first-frame
        re-encoding without clearing the streaming cache.
        """
        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4

        for i in range(iter_):
            conv_idx = [0]
            if i == 0:
                out = self.model.encoder(
                    x[:, :, :1, :, :],
                    feat_cache=feat_cache,
                    feat_idx=conv_idx,
                )
            else:
                out_ = self.model.encoder(
                    x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i, :, :],
                    feat_cache=feat_cache,
                    feat_idx=conv_idx,
                )
                out = torch.cat([out, out_], 2)

        mu, _ = self.model.conv1(out).chunk(2, dim=1)
        # Apply normalization
        return self._apply_encoding_normalization(mu, scale)

    def decode_to_pixel(
        self, latent: torch.Tensor, use_cache: bool = True
    ) -> torch.Tensor:
        """Decode latents to video pixels.

        Args:
            latent: Latent tensor [batch, frames, channels, height, width]
            use_cache: If True, use streaming decode (maintains cache state).
                      If False, use batch decode (clears cache before/after).

        Returns:
            Video tensor [batch, frames, channels, height, width] in range [-1, 1]
        """
        if _should_profile_wanvae_decode():
            _wanvae_decode_meta.update(
                {
                    "use_cache": bool(use_cache),
                    "latent_shape": list(latent.shape),
                    "latent_dtype": str(latent.dtype),
                    "latent_device": str(latent.device),
                    "WANVAE_DECODE_CHANNELS_LAST_3D": bool(_WANVAE_DECODE_CHANNELS_LAST_3D),
                }
            )

        with _ProfileWanVAEDecode("prep_permute_cast"):
            # [batch, frames, channels, h, w] -> [batch, channels, frames, h, w]
            zs = latent.permute(0, 2, 1, 3, 4)
            zs = zs.to(torch.bfloat16).to("cuda")
            if _WANVAE_DECODE_CHANNELS_LAST_3D and zs.is_cuda and zs.ndim == 5:
                zs = zs.contiguous(memory_format=torch.channels_last_3d)

        with _ProfileWanVAEDecode("get_scale"):
            device, dtype = latent.device, latent.dtype
            scale = self._get_scale(device, dtype)

        with _ProfileWanVAEDecode("stream_decode" if use_cache else "decode"):
            if use_cache:
                output = self.model.stream_decode(zs, scale)
            else:
                output = self.model.decode(zs, scale)

        with _ProfileWanVAEDecode("postprocess"):
            output = output.float().clamp_(-1, 1)
            # [batch, channels, frames, h, w] -> [batch, frames, channels, h, w]
            output = output.permute(0, 2, 1, 3, 4)

        return output

    def clear_cache(self):
        """Clear encoder/decoder cache for next sequence."""
        self.model.first_batch = True
