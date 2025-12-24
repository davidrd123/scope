import atexit
import json
import os
import time
import traceback
from collections import defaultdict

import torch
from diffusers.modular_pipelines import SequentialPipelineBlocks
from diffusers.modular_pipelines.modular_pipeline_utils import InsertableDict
from diffusers.utils import logging as diffusers_logging

from ..wan2_1.blocks import (
    AutoPrepareLatentsBlock,
    AutoPreprocessVideoBlock,
    DecodeBlock,
    DenoiseBlock,
    EmbeddingBlendingBlock,
    PrepareNextBlock,
    SetTimestepsBlock,
    SetupCachesBlock,
    TextConditioningBlock,
)
from .blocks import PrepareContextFramesBlock, RecomputeKVCacheBlock

logger = diffusers_logging.get_logger(__name__)

_PROFILE_PIPELINE_BLOCKS = os.getenv("PROFILE_PIPELINE_BLOCKS", "0") == "1"
_PROFILE_PIPELINE_BLOCKS_JSON = os.getenv("PROFILE_PIPELINE_BLOCKS_JSON")
_profile_block_cpu_ms = defaultdict(float)
_profile_block_gpu_ms = defaultdict(float)
_profile_block_counts = defaultdict(int)


def reset_pipeline_block_profile() -> None:
    """Clear accumulated per-block profiling counters."""
    _profile_block_cpu_ms.clear()
    _profile_block_gpu_ms.clear()
    _profile_block_counts.clear()


def _should_profile_pipeline_blocks() -> bool:
    if not _PROFILE_PIPELINE_BLOCKS:
        return False
    if hasattr(torch, "compiler") and hasattr(torch.compiler, "is_compiling"):
        if torch.compiler.is_compiling():
            return False
    return True


def _pipeline_profile_report() -> None:
    if not _profile_block_counts:
        return

    total_cpu_ms = sum(_profile_block_cpu_ms.values())
    total_gpu_ms = sum(_profile_block_gpu_ms.values())
    logger.info("=== Pipeline Block Profiling Report ===")
    for name, cpu_ms in sorted(_profile_block_cpu_ms.items(), key=lambda kv: -kv[1]):
        calls = _profile_block_counts.get(name, 0)
        gpu_ms = _profile_block_gpu_ms.get(name, 0.0)
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

    if _PROFILE_PIPELINE_BLOCKS_JSON:
        meta = {"torch": torch.__version__, "cuda": torch.version.cuda}
        if torch.cuda.is_available():
            try:
                meta.update(
                    {
                        "cuda_device": torch.cuda.get_device_name(0),
                        "cuda_capability": list(torch.cuda.get_device_capability(0)),
                    }
                )
            except Exception:
                pass
        payload = {
            "meta": meta,
            "total_cpu_ms": total_cpu_ms,
            "total_gpu_ms": total_gpu_ms,
            "blocks": {
                name: {
                    "cpu_ms": _profile_block_cpu_ms.get(name, 0.0),
                    "gpu_ms": _profile_block_gpu_ms.get(name, 0.0),
                    "calls": int(_profile_block_counts.get(name, 0)),
                }
                for name in sorted(_profile_block_counts.keys())
            },
        }
        try:
            with open(_PROFILE_PIPELINE_BLOCKS_JSON, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
            logger.info("Wrote pipeline block profile JSON: %s", _PROFILE_PIPELINE_BLOCKS_JSON)
        except Exception as e:
            logger.warning("Failed writing pipeline block profile JSON (%s): %s", _PROFILE_PIPELINE_BLOCKS_JSON, e)


atexit.register(_pipeline_profile_report)


class _ProfilePipelineBlock:
    __slots__ = ("name", "_cpu_start", "_start", "_end")

    def __init__(self, name: str):
        self.name = name
        self._cpu_start = None
        self._start = None
        self._end = None

    def __enter__(self):
        if not _should_profile_pipeline_blocks():
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
            # Wait for block work to finish (profiling only; breaks async overlap).
            self._end.synchronize()
            elapsed_gpu_ms = self._start.elapsed_time(self._end)
            _profile_block_gpu_ms[self.name] += elapsed_gpu_ms

        elapsed_cpu_ms = (time.perf_counter() - self._cpu_start) * 1000.0
        _profile_block_cpu_ms[self.name] += elapsed_cpu_ms
        _profile_block_counts[self.name] += 1


# Main pipeline blocks with multi-mode support (text-to-video and video-to-video)
# AutoPreprocessVideoBlock: Routes to video preprocessing when 'video' input provided
# AutoPrepareLatentsBlock: Routes to PrepareVideoLatentsBlock or PrepareLatentsBlock
ALL_BLOCKS = InsertableDict(
    [
        ("text_conditioning", TextConditioningBlock),
        ("embedding_blending", EmbeddingBlendingBlock),
        ("set_timesteps", SetTimestepsBlock),
        ("auto_preprocess_video", AutoPreprocessVideoBlock),
        ("setup_caches", SetupCachesBlock),
        ("auto_prepare_latents", AutoPrepareLatentsBlock),
        ("recompute_kv_cache", RecomputeKVCacheBlock),
        ("denoise", DenoiseBlock),
        ("decode", DecodeBlock),
        ("prepare_context_frames", PrepareContextFramesBlock),
        ("prepare_next", PrepareNextBlock),
    ]
)


class KreaRealtimeVideoBlocks(SequentialPipelineBlocks):
    block_classes = list(ALL_BLOCKS.values())
    block_names = list(ALL_BLOCKS.keys())

    @torch.no_grad()
    def __call__(self, pipeline, state):
        for block_name, block in self.sub_blocks.items():
            try:
                if _should_profile_pipeline_blocks():
                    with _ProfilePipelineBlock(block_name):
                        pipeline, state = block(pipeline, state)
                else:
                    pipeline, state = block(pipeline, state)
            except Exception as e:
                error_msg = (
                    f"\nError in block: ({block_name}, {block.__class__.__name__})\n"
                    f"Error details: {str(e)}\n"
                    f"Traceback:\n{traceback.format_exc()}"
                )
                logger.error(error_msg)
                raise
        return pipeline, state
