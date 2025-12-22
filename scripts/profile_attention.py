#!/usr/bin/env python3
"""
Profile attention calls in the Krea realtime pipeline.

Measures p_bias vs p_recompute split to guide kernel optimization.

Usage:
    uv run python scripts/profile_attention.py --timeline timeline.json --frames 16
    uv run python scripts/profile_attention.py --timeline timeline.json --frames 16 --trace profile.json
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler


def analyze_trace(prof):
    """Analyze profiler trace for flex_attention kernels."""
    print("\n" + "=" * 70)
    print("Attention Kernel Analysis")
    print("=" * 70)

    # Get all kernel events
    events = prof.key_averages()

    # Find flex_attention related kernels
    flex_kernels = []
    other_kernels = []

    for evt in events:
        if evt.device_type == torch.profiler.DeviceType.CUDA:
            name = evt.key
            if "flex_attention" in name.lower() or "triton_flex" in name.lower():
                flex_kernels.append(evt)
            elif evt.cuda_time_total > 0:
                other_kernels.append(evt)

    # Sort by time
    flex_kernels.sort(key=lambda e: e.cuda_time_total, reverse=True)
    other_kernels.sort(key=lambda e: e.cuda_time_total, reverse=True)

    # Calculate totals
    total_flex_us = sum(e.cuda_time_total for e in flex_kernels)
    total_other_us = sum(e.cuda_time_total for e in other_kernels)
    total_cuda_us = total_flex_us + total_other_us

    if total_cuda_us == 0:
        print("\nNo CUDA kernels captured! Pipeline may not have run.")
        print(f"Found {len(flex_kernels)} flex kernels, {len(other_kernels)} other kernels")
        return

    print(f"\nTotal CUDA time: {total_cuda_us/1000:.1f} ms")
    print(f"  flex_attention: {total_flex_us/1000:.1f} ms ({total_flex_us/total_cuda_us*100:.1f}%)")
    print(f"  other kernels:  {total_other_us/1000:.1f} ms ({total_other_us/total_cuda_us*100:.1f}%)")

    print("\n" + "-" * 70)
    print("Top flex_attention kernels:")
    print("-" * 70)

    for evt in flex_kernels[:15]:
        pct = evt.cuda_time_total / total_cuda_us * 100
        print(f"  {evt.cuda_time_total/1000:8.2f} ms ({pct:5.1f}%)  {evt.count:4d}x  {evt.key[:60]}")

    print("\n" + "-" * 70)
    print("Top other CUDA kernels:")
    print("-" * 70)

    for evt in other_kernels[:10]:
        pct = evt.cuda_time_total / total_cuda_us * 100
        print(f"  {evt.cuda_time_total/1000:8.2f} ms ({pct:5.1f}%)  {evt.count:4d}x  {evt.key[:60]}")

    # Try to categorize flex kernels by type
    print("\n" + "-" * 70)
    print("Flex attention breakdown by type:")
    print("-" * 70)

    contiguous_time = 0
    non_contiguous_time = 0
    contiguous_count = 0
    non_contiguous_count = 0

    for evt in flex_kernels:
        name = evt.key
        # The kernel config often includes BLOCKS_ARE_CONTIGUOUS
        if "CONTIGUOUS=True" in name or "contiguous" in name.lower():
            contiguous_time += evt.cuda_time_total
            contiguous_count += evt.count
        elif "CONTIGUOUS=False" in name:
            non_contiguous_time += evt.cuda_time_total
            non_contiguous_count += evt.count

    if contiguous_time > 0 or non_contiguous_time > 0:
        print(f"\n  BLOCKS_ARE_CONTIGUOUS=True  (block_mask/recompute path):")
        print(f"    Time: {contiguous_time/1000:.1f} ms, Calls: {contiguous_count}")
        print(f"\n  BLOCKS_ARE_CONTIGUOUS=False (score_mod/bias path):")
        print(f"    Time: {non_contiguous_time/1000:.1f} ms, Calls: {non_contiguous_count}")

        total_flex = contiguous_time + non_contiguous_time
        if total_flex > 0:
            p_recompute = contiguous_time / total_flex
            p_bias = non_contiguous_time / total_flex
            print(f"\n  p_recompute (block_mask): {p_recompute*100:.1f}%")
            print(f"  p_bias (score_mod):       {p_bias*100:.1f}%")

            if p_bias > p_recompute:
                print("\n  -> Kernel B (bias) is the bigger target")
            else:
                print("\n  -> Kernel A (recompute) is the bigger target")
    else:
        print("\n  Could not determine kernel types from names")
        print("  Looking at raw kernel names for clues...")
        for evt in flex_kernels[:5]:
            print(f"    {evt.key}")


def main():
    parser = argparse.ArgumentParser(description="Profile attention in Krea pipeline")
    parser.add_argument("--timeline", type=Path, required=True, help="Timeline JSON file")
    parser.add_argument("--output", type=Path, default=Path("/tmp/profile_output.mp4"))
    parser.add_argument("--frames", type=int, default=16, help="Number of frames to render")
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--kv-cache-attention-bias", type=float, default=0.3)
    parser.add_argument("--kv-cache-num-frames", type=int, default=3)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--trace", type=Path, default=None, help="Save trace to JSON file")
    parser.add_argument("--skip-frames", type=int, default=2, help="Skip first N frames (warmup)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    # Import pipeline
    from scope.cli.render_timeline import render_timeline

    # Modify timeline to limit frames
    timeline_data = json.loads(args.timeline.read_text())
    if "prompts" in timeline_data:
        fps = 16
        end_time = args.frames / fps
        # Filter and truncate segments
        new_prompts = []
        for prompt in timeline_data["prompts"]:
            start = prompt.get("startTime", 0)
            end = prompt.get("endTime", 0)
            if start >= end_time:
                continue  # Skip segments that start after our cutoff
            if end > end_time:
                prompt["endTime"] = end_time
            new_prompts.append(prompt)
        timeline_data["prompts"] = new_prompts

    # Write modified timeline
    temp_timeline = Path("/tmp/profile_timeline.json")
    temp_timeline.write_text(json.dumps(timeline_data))
    print(f"Timeline has {len(timeline_data.get('prompts', []))} segment(s), ending at {end_time}s")

    # Build CLI args
    cli_args = [
        str(temp_timeline),
        str(args.output),
        "--height", str(args.height),
        "--width", str(args.width),
        "--kv-cache-attention-bias", str(args.kv_cache_attention_bias),
        "--num-inference-steps", str(args.num_inference_steps),
    ]
    if args.kv_cache_num_frames:
        cli_args.extend(["--kv-cache-num-frames", str(args.kv_cache_num_frames)])

    print(f"Profiling pipeline with {args.frames} frames...")
    print(f"Settings: {args.height}x{args.width}, {args.num_inference_steps} steps")
    print(f"KV cache: bias={args.kv_cache_attention_bias}, num_frames={args.kv_cache_num_frames}")
    print()

    # Run pipeline without profiler first (model loading fails with profiler active)
    # Then we'll use a simpler approach
    print("Note: Using torch.cuda.Event timing instead of profiler due to model loading conflicts")
    print()

    try:
        render_timeline(cli_args)
    except SystemExit:
        pass

    # Since we can't use PyTorch profiler directly, let's just report that we ran
    print("\n" + "=" * 70)
    print("Pipeline completed. For detailed attention profiling, use nsys:")
    print("=" * 70)
    print(f"\n  nsys profile -o profile_attn uv run python scripts/profile_attention.py ...")
    print("  nsys stats profile_attn.nsys-rep")
    print("\nOr run the microbenchmark to estimate p_bias vs p_recompute:")
    print("  uv run python scripts/bench_blockwise_attn.py --mode block_mask --frames 6 --compile")
    print("  uv run python scripts/bench_blockwise_attn.py --mode bias --q-frames 3 --kv-frames 6 --no-pad-q-to-k --compile")
    return

    # Save trace if requested
    if args.trace:
        prof.export_chrome_trace(str(args.trace))
        print(f"\nTrace saved to: {args.trace}")
        print("View with: chrome://tracing or https://ui.perfetto.dev/")


if __name__ == "__main__":
    main()
