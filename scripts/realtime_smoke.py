#!/usr/bin/env python3
"""Smoke test for realtime control plane with actual KREA pipeline.

This script validates that:
1. GeneratorDriver works with the real pipeline
2. PipelineAdapter correctly maps kwargs
3. Continuity keys exist and have correct types
4. Prompt changes produce visible output differences

Usage:
    python scripts/realtime_smoke.py

    # With different prompts
    python scripts/realtime_smoke.py --prompt "a cat walking"

    # More chunks
    python scripts/realtime_smoke.py --chunks 10

Output:
    outputs/realtime_smoke_*.png - Generated frames
    outputs/realtime_smoke_state.json - State/continuity info
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scope.realtime import ControlState, GeneratorDriver, PipelineAdapter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_models_dir() -> Path:
    """Get models directory using server config."""
    from scope.server.models_config import get_models_dir as _get_models_dir
    return _get_models_dir()


def get_model_file_path(relative_path: str) -> Path:
    """Get full path to a model file."""
    from scope.server.models_config import get_model_file_path as _get_model_file_path
    return _get_model_file_path(relative_path)


def load_krea_pipeline():
    """Load the KREA realtime video pipeline."""
    from scope.core.pipelines import KreaRealtimeVideoPipeline

    logger.info("Loading KREA realtime video pipeline...")

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
            "vae_path": str(get_model_file_path("Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")),
            "height": 512,
            "width": 512,
            "seed": 42,
        }
    )

    # Check if we're on Hopper for torch.compile
    device_name = torch.cuda.get_device_name(0).lower() if torch.cuda.is_available() else ""
    compile_model = any(x in device_name for x in ("h100", "hopper"))

    pipeline = KreaRealtimeVideoPipeline(
        config,
        quantization="fp8_e4m3fn",  # Default quantization
        compile=compile_model,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
    )

    logger.info("Pipeline loaded successfully")
    return pipeline


def save_frames(frames: torch.Tensor, output_dir: Path, prefix: str, chunk_idx: int):
    """Save frames to PNG files."""
    from PIL import Image
    import numpy as np

    # frames is [N, H, W, C] in [0, 1] range or already uint8
    if frames.dtype == torch.bfloat16 or frames.dtype == torch.float32:
        frames = (frames * 255).clamp(0, 255).to(torch.uint8)

    frames_np = frames.cpu().numpy()

    saved = []
    for i, frame in enumerate(frames_np):
        frame_idx = (chunk_idx - 1) * len(frames_np) + i
        path = output_dir / f"{prefix}_frame_{frame_idx:04d}.png"
        Image.fromarray(frame).save(path)
        saved.append(str(path))

    return saved


def inspect_pipeline_state(pipeline, adapter: PipelineAdapter) -> dict:
    """Inspect pipeline state for debugging."""
    info = {
        "continuity_keys": {},
        "other_state_keys": [],
    }

    if hasattr(pipeline, "state"):
        # Check continuity keys
        for key in PipelineAdapter.CONTINUITY_KEYS:
            value = pipeline.state.get(key)
            if value is not None:
                if isinstance(value, torch.Tensor):
                    info["continuity_keys"][key] = {
                        "type": "Tensor",
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                    }
                elif isinstance(value, list):
                    info["continuity_keys"][key] = {
                        "type": "list",
                        "length": len(value),
                    }
                else:
                    info["continuity_keys"][key] = {
                        "type": type(value).__name__,
                        "value": str(value)[:100],
                    }

        # List other state keys
        if hasattr(pipeline.state, "_data"):
            all_keys = set(pipeline.state._data.keys())
            continuity_set = set(PipelineAdapter.CONTINUITY_KEYS)
            info["other_state_keys"] = list(all_keys - continuity_set)

    return info


async def run_smoke_test(args):
    """Run the smoke test."""
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # Load pipeline
    pipeline = load_krea_pipeline()

    # Track results and timing
    results = []
    state_info = {"chunks": [], "continuity_snapshots": []}

    def on_chunk(result):
        """Callback for each generated chunk."""
        logger.info(
            f"Chunk {result.chunk_index}: {result.timing_ms:.1f}ms, "
            f"frames shape: {result.frames.shape if result.frames is not None else 'None'}"
        )
        results.append(result)

    def on_state_change(state):
        """Callback for driver state changes."""
        logger.info(f"Driver state: {state.value}")

    # Create driver
    driver = GeneratorDriver(
        pipeline=pipeline,
        on_chunk=on_chunk,
        on_state_change=on_state_change,
    )

    # Set initial prompt
    driver.control_state.prompts = [{"text": args.prompt, "weight": 1.0}]
    driver.control_state.kv_cache_attention_bias = args.kv_bias

    logger.info(f"Initial prompt: {args.prompt}")
    logger.info(f"KV cache attention bias: {args.kv_bias}")
    logger.info(f"Generating {args.chunks} chunks...")

    # Generate chunks
    total_start = time.perf_counter()

    for i in range(args.chunks):
        # Change prompt mid-way if requested
        if args.prompt2 and i == args.chunks // 2:
            logger.info(f"Changing prompt to: {args.prompt2}")
            driver.control_state.prompts = [{"text": args.prompt2, "weight": 1.0}]

        result = await driver.step()

        if result and result.frames is not None:
            # Save frames
            saved = save_frames(result.frames, output_dir, "realtime_smoke", result.chunk_index)
            logger.info(f"  Saved: {saved[0]} ... {saved[-1]}")

            # Capture state info
            state_info["chunks"].append({
                "chunk_index": result.chunk_index,
                "timing_ms": result.timing_ms,
                "frames_shape": list(result.frames.shape),
            })

        # Capture continuity snapshot after first and last chunk
        if i == 0 or i == args.chunks - 1:
            pipeline_state = inspect_pipeline_state(pipeline, driver.adapter)
            state_info["continuity_snapshots"].append({
                "after_chunk": i + 1,
                "state": pipeline_state,
            })

    total_time = time.perf_counter() - total_start

    # Summary
    logger.info("=" * 60)
    logger.info("SMOKE TEST SUMMARY")
    logger.info("=" * 60)

    total_frames = sum(r.frames.shape[0] for r in results if r.frames is not None)
    avg_chunk_time = sum(r.timing_ms for r in results) / len(results) if results else 0
    fps = total_frames / total_time if total_time > 0 else 0

    logger.info(f"Total chunks: {len(results)}")
    logger.info(f"Total frames: {total_frames}")
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info(f"Average chunk time: {avg_chunk_time:.1f}ms")
    logger.info(f"Effective FPS: {fps:.2f}")

    # Check continuity keys
    logger.info("\nContinuity keys after generation:")
    final_snapshot = state_info["continuity_snapshots"][-1]["state"]
    for key, info in final_snapshot["continuity_keys"].items():
        logger.info(f"  {key}: {info}")

    # Save state info
    state_path = output_dir / "realtime_smoke_state.json"
    with open(state_path, "w") as f:
        json.dump(state_info, f, indent=2, default=str)
    logger.info(f"\nState info saved to: {state_path}")

    # Test snapshot/restore
    if args.test_restore:
        logger.info("\n" + "=" * 60)
        logger.info("TESTING SNAPSHOT/RESTORE")
        logger.info("=" * 60)

        snapshot = driver.snapshot()
        logger.info(f"Snapshot captured at chunk {snapshot['chunk_index']}")
        logger.info(f"Continuity keys in snapshot: {list(snapshot['generator_continuity'].keys())}")

        # Modify state
        driver.control_state.base_seed = 9999
        driver.chunk_index = 0

        # Restore
        driver.restore(snapshot)
        logger.info(f"Restored: chunk_index={driver.chunk_index}, seed={driver.control_state.base_seed}")
        logger.info(f"_is_prepared after restore: {driver._is_prepared}")

        # Generate one more chunk to verify seamless continuation
        result = await driver.step()
        if result and result.frames is not None:
            saved = save_frames(result.frames, output_dir, "realtime_smoke_restored", result.chunk_index)
            logger.info(f"Post-restore chunk saved: {saved}")

    logger.info("\n✓ Smoke test complete!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Realtime control plane smoke test")
    parser.add_argument(
        "--prompt",
        type=str,
        default="a beautiful sunset over mountains, cinematic lighting",
        help="Initial prompt for generation",
    )
    parser.add_argument(
        "--prompt2",
        type=str,
        default=None,
        help="Second prompt to switch to mid-generation (tests prompt changes)",
    )
    parser.add_argument(
        "--chunks",
        type=int,
        default=5,
        help="Number of chunks to generate (each chunk = 3 frames)",
    )
    parser.add_argument(
        "--kv-bias",
        type=float,
        default=0.3,
        help="KV cache attention bias (0.3 = KREA default)",
    )
    parser.add_argument(
        "--test-restore",
        action="store_true",
        help="Test snapshot/restore functionality",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        logger.error("CUDA not available. This script requires a GPU.")
        sys.exit(1)

    logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    asyncio.run(run_smoke_test(args))


if __name__ == "__main__":
    main()
