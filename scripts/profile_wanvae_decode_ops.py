#!/usr/bin/env python3
"""
Operator-level (CUDA) profiling for WanVAE stream_decode().

This is a focused companion to `scripts/profile_krea_pipeline_ops.py` that isolates
VAE decode, which is a large share of the B300 steady-state budget.

Example (B300/cu130 env):

  TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas \
  WANVAE_STREAM_DECODE_MODE=chunk \
  .venv-b300-cu130-decode/bin/python scripts/profile_wanvae_decode_ops.py \
    --height 320 --width 576 --t 3 \
    --dtype bf16 \
    --iters 50 --pre-iters 10 \
    --cudnn-benchmark \
    --channels-last-3d \
    --with-stack \
    --summary outputs/b300_wanvae_decode_ops.md

Note: If the Markdown summary shows all-zero `self_cuda_ms`, increase `--iters`
(fast paths sometimes need more kernel launches before CUDA timings appear).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Operator-level profiler for WanVAE stream_decode")
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--t", type=int, default=3, help="Number of frames decoded per call")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp16"],
        default="bf16",
        help="Compute dtype for the VAE decode profile.",
    )
    parser.add_argument("--iters", type=int, default=10, help="Number of profiled calls")
    parser.add_argument("--pre-iters", type=int, default=10, help="Warm calls outside profiler")
    parser.add_argument(
        "--cudnn-benchmark",
        action="store_true",
        help="Enable torch.backends.cudnn.benchmark (can improve conv performance).",
    )
    parser.add_argument(
        "--stream-decode-mode",
        choices=["chunk", "loop"],
        default=None,
        help="Set WANVAE_STREAM_DECODE_MODE for this run.",
    )
    parser.add_argument(
        "--channels-last-3d",
        action="store_true",
        help="Set WANVAE_DECODE_CHANNELS_LAST_3D=1 for this run.",
    )
    parser.add_argument(
        "--implicit-spatial-padding",
        type=int,
        choices=[0, 1],
        default=None,
        help="Set WANVAE_CONV3D_IMPLICIT_SPATIAL_PADDING (default: leave env as-is).",
    )
    parser.add_argument(
        "--upsample-force-fp32",
        action="store_true",
        help="Set WANVAE_UPSAMPLE_FORCE_FP32=1 for this run.",
    )
    parser.add_argument(
        "--record-shapes",
        action="store_true",
        help="Enable record_shapes in the profiler (more detail, more overhead).",
    )
    parser.add_argument(
        "--row-limit",
        type=int,
        default=40,
        help="Rows to print in the profiler table output.",
    )
    parser.add_argument(
        "--with-stack",
        action="store_true",
        help="Capture call stacks in the trace (Kineto verbose mode). This can add overhead.",
    )
    parser.add_argument(
        "--stack-n",
        type=int,
        default=6,
        help="Stack depth for grouped-by-stack summaries (used with --with-stack).",
    )
    parser.add_argument(
        "--stack-key",
        action="append",
        default=None,
        help="Repeatable: print a grouped-by-stack summary for this op key (e.g. --stack-key aten::cudnn_convolution).",
    )
    parser.add_argument(
        "--stack-limit",
        type=int,
        default=15,
        help="How many stack groups to print per --stack-key.",
    )
    parser.add_argument(
        "--stack-include",
        action="append",
        default=None,
        help="Repeatable: only keep stack groups whose frames include this substring.",
    )
    parser.add_argument(
        "--stack-exclude",
        action="append",
        default=None,
        help="Repeatable: drop stack groups whose frames include this substring.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Optional: write a Markdown summary (top ops + grouped-by-stack highlights).",
    )
    parser.add_argument(
        "--chrome-trace",
        type=Path,
        default=None,
        help="Optional: write a Chrome trace JSON (can be large).",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional: write aggregated key_averages() + stack groups to JSON.",
    )
    return parser.parse_args()


def _cuda_unavailable_hint(torch) -> str:
    return (
        "CUDA not available.\n"
        f"- torch: {torch.__version__} (cuda={torch.version.cuda})\n"
        "If NVML is wedged (nvidia-smi errors), a host driver restart/reboot may be required."
    )


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _evt_device_us(evt) -> float:
    # On newer PyTorch versions this is `device_time_total`, not `cuda_time_total`.
    return float(getattr(evt, "device_time_total", 0.0) or 0.0)


def _stack_strs(evt) -> list[str]:
    return [str(frame) for frame in getattr(evt, "stack", [])]


def _stack_group_matches(
    stack_frames: list[str],
    include: list[str] | None,
    exclude: list[str] | None,
) -> bool:
    if include:
        if not any(pat in frame for pat in include for frame in stack_frames):
            return False
    if exclude:
        if any(pat in frame for pat in exclude for frame in stack_frames):
            return False
    return True


def main() -> int:
    args = _parse_args()

    # Set env toggles before importing the VAE modules (they read env at import time).
    if args.stream_decode_mode is not None:
        os.environ["WANVAE_STREAM_DECODE_MODE"] = str(args.stream_decode_mode)
    if args.channels_last_3d:
        os.environ["WANVAE_DECODE_CHANNELS_LAST_3D"] = "1"
    if args.implicit_spatial_padding is not None:
        os.environ["WANVAE_CONV3D_IMPLICIT_SPATIAL_PADDING"] = str(args.implicit_spatial_padding)
    if args.upsample_force_fp32:
        os.environ["WANVAE_UPSAMPLE_FORCE_FP32"] = "1"

    import torch
    from torch.profiler import ProfilerActivity, profile

    if not torch.cuda.is_available():
        raise SystemExit(_cuda_unavailable_hint(torch))

    torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    if args.height % 8 != 0 or args.width % 8 != 0:
        raise SystemExit("height/width must be divisible by 8 (WanVAE latent downsample).")

    from scope.core.config import get_model_file_path
    from scope.core.pipelines.wan2_1.vae.wan import WanVAEWrapper

    vae_path = Path(get_model_file_path("Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"))

    print(f"torch: {torch.__version__} (cuda={torch.version.cuda}, cudnn={torch.backends.cudnn.version()})")
    print(f"device: {torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)}")
    print(f"cudnn.benchmark: {torch.backends.cudnn.benchmark}")
    print(f"vae_path: {vae_path}")
    print(f"WANVAE_STREAM_DECODE_MODE: {os.getenv('WANVAE_STREAM_DECODE_MODE')}")
    print(f"WANVAE_DECODE_CHANNELS_LAST_3D: {os.getenv('WANVAE_DECODE_CHANNELS_LAST_3D')}")
    print(f"WANVAE_CONV3D_IMPLICIT_SPATIAL_PADDING: {os.getenv('WANVAE_CONV3D_IMPLICIT_SPATIAL_PADDING')}")
    print(f"WANVAE_UPSAMPLE_FORCE_FP32: {os.getenv('WANVAE_UPSAMPLE_FORCE_FP32')}")
    print("")

    vae = WanVAEWrapper(vae_path=str(vae_path)).to(device="cuda", dtype=dtype).eval()

    latent_h = args.height // 8
    latent_w = args.width // 8
    latent = torch.randn(
        args.batch,
        args.t,
        16,
        latent_h,
        latent_w,
        device="cuda",
        dtype=dtype,
    )
    zs = latent.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
    scale = vae._get_scale(device=latent.device, dtype=latent.dtype)

    if os.getenv("WANVAE_DECODE_CHANNELS_LAST_3D", "0") == "1":
        zs = zs.contiguous(memory_format=torch.channels_last_3d)
    else:
        zs = zs.contiguous()

    # Warmup outside the profiler to reduce noise / hit steady-state kernels.
    with torch.inference_mode():
        for _ in range(int(args.pre_iters)):
            _ = vae.model.stream_decode(zs, scale)
        torch.cuda.synchronize()

    try:
        experimental_config = torch._C._profiler._ExperimentalConfig(verbose=True)
    except Exception:
        experimental_config = None

    t0 = time.perf_counter()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_stack=bool(args.with_stack),
        record_shapes=bool(args.record_shapes),
        experimental_config=experimental_config if args.with_stack else None,
    ) as prof:
        with torch.inference_mode():
            for _ in range(int(args.iters)):
                _ = vae.model.stream_decode(zs, scale)
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    print(f"Profiled wall time: {dt:.3f}s")
    print("")
    print(prof.key_averages().table(sort_by="device_time_total", row_limit=int(args.row_limit)))

    events_for_summary = list(prof.key_averages())
    total_self_device_us = sum(
        float(getattr(evt, "self_device_time_total", 0.0) or 0.0)
        for evt in events_for_summary
    )
    if total_self_device_us <= 0.0:
        print(
            "\nWARNING: CUDA timings were all zero. This can happen on fast paths with too few\n"
            "kernel launches captured by the profiler. Re-run with a higher `--iters` (e.g. 50).\n"
        )
    top_events = sorted(
        events_for_summary,
        key=lambda e: float(getattr(e, "self_device_time_total", 0.0) or 0.0),
        reverse=True,
    )[: min(len(events_for_summary), 25)]

    summary_lines: list[str] = []
    stack_groups: dict[str, list[dict]] | None = None

    if args.summary is not None:
        summary_lines.append("# WanVAE stream_decode op profile\n\n")
        summary_lines.append("## Meta\n")
        summary_lines.append(f"- torch: `{torch.__version__}` (cuda `{torch.version.cuda}`)\n")
        summary_lines.append(
            f"- device: `{torch.cuda.get_device_name(0)}` cc={torch.cuda.get_device_capability(0)}\n"
        )
        summary_lines.append(f"- height,width: `{args.height}x{args.width}`\n")
        summary_lines.append(f"- latent_shape: `{list(zs.shape)}`\n")
        summary_lines.append(f"- dtype: `{args.dtype}`\n")
        summary_lines.append(f"- iters: `{args.iters}` (pre-iters `{args.pre_iters}`)\n")
        summary_lines.append(f"- cudnn.benchmark: `{bool(torch.backends.cudnn.benchmark)}`\n")
        summary_lines.append(f"- WANVAE_STREAM_DECODE_MODE: `{os.getenv('WANVAE_STREAM_DECODE_MODE')}`\n")
        summary_lines.append(f"- WANVAE_DECODE_CHANNELS_LAST_3D: `{os.getenv('WANVAE_DECODE_CHANNELS_LAST_3D')}`\n")
        summary_lines.append(
            f"- WANVAE_CONV3D_IMPLICIT_SPATIAL_PADDING: `{os.getenv('WANVAE_CONV3D_IMPLICIT_SPATIAL_PADDING')}`\n"
        )
        summary_lines.append(f"- WANVAE_UPSAMPLE_FORCE_FP32: `{os.getenv('WANVAE_UPSAMPLE_FORCE_FP32')}`\n")
        summary_lines.append(f"- profiled_wall_time_s: `{dt:.3f}`\n")
        summary_lines.append("\n## Top ops (self CUDA)\n")
        summary_lines.append("| self_cuda_ms | pct | calls | key |\n")
        summary_lines.append("|---:|---:|---:|---|\n")
        for evt in top_events:
            self_us = float(getattr(evt, "self_device_time_total", 0.0) or 0.0)
            pct = (100.0 * self_us / total_self_device_us) if total_self_device_us else 0.0
            summary_lines.append(
                f"| {self_us/1e3:,.3f} | {pct:,.2f}% | {evt.count:,} | `{evt.key}` |\n"
            )

    if args.with_stack:
        stack_keys = (
            args.stack_key
            if args.stack_key is not None
            else [
                "aten::cudnn_convolution",
                "aten::_convolution",
                "aten::upsample_trilinear3d",
                "aten::native_group_norm",
            ]
        )
        stack_avgs = prof.key_averages(group_by_stack_n=int(args.stack_n))

        print("")
        print(f"Grouped-by-stack summary (group_by_stack_n={args.stack_n}):")
        stack_groups = {}
        include_pats = args.stack_include or []
        exclude_pats = args.stack_exclude or []
        if include_pats:
            print(f"- stack_include: {include_pats}")
        if exclude_pats:
            print(f"- stack_exclude: {exclude_pats}")

        if args.summary is not None and (include_pats or exclude_pats):
            summary_lines.append("\n## Stack filters\n")
            summary_lines.append(f"- stack_include: `{include_pats}`\n")
            summary_lines.append(f"- stack_exclude: `{exclude_pats}`\n")

        for key in stack_keys:
            rows = [evt for evt in stack_avgs if evt.key == key]
            if include_pats or exclude_pats:
                rows = [
                    evt
                    for evt in rows
                    if _stack_group_matches(_stack_strs(evt), include_pats, exclude_pats)
                ]
            if not rows:
                print(f"- {key}: (no events)")
                stack_groups[key] = []
                if args.summary is not None:
                    summary_lines.append(f"\n## Stack groups: `{key}`\n\n(no events)\n")
                continue

            rows.sort(key=_evt_device_us, reverse=True)
            print(f"- {key}: top {min(len(rows), args.stack_limit)}")
            stack_groups[key] = []

            if args.summary is not None:
                summary_lines.append(f"\n## Stack groups: `{key}`\n")
                total_device_us = sum(_evt_device_us(evt) for evt in rows)
                total_calls = sum(int(evt.count) for evt in rows)
                summary_lines.append(
                    f"\nFiltered totals: device_ms={total_device_us/1e3:.3f}, calls={total_calls}\n"
                )

            for evt in rows[: int(args.stack_limit)]:
                device_us = _evt_device_us(evt)
                self_device_us = float(getattr(evt, "self_device_time_total", 0.0) or 0.0)
                cpu_us = float(getattr(evt, "cpu_time_total", 0.0) or 0.0)
                print(
                    f"  count={evt.count:<7} "
                    f"device_ms={device_us/1e3:>8.3f} "
                    f"self_device_ms={self_device_us/1e3:>8.3f} "
                    f"cpu_ms={cpu_us/1e3:>8.3f}"
                )
                frames = _stack_strs(evt)
                for frame in frames[: int(args.stack_n)]:
                    print(f"    {frame}")
                stack_groups[key].append(
                    {
                        "count": int(evt.count),
                        "cpu_time_total_us": cpu_us,
                        "device_time_total_us": device_us,
                        "self_device_time_total_us": self_device_us,
                        "stack": frames,
                    }
                )
                if args.summary is not None:
                    summary_lines.append(
                        f"\n- count={evt.count} device_ms={device_us/1e3:.3f} "
                        f"self_device_ms={self_device_us/1e3:.3f} cpu_ms={cpu_us/1e3:.3f}\n"
                    )
                    for frame in frames[: int(args.stack_n)]:
                        summary_lines.append(f"  - `{frame}`\n")

    if args.json is not None:
        events = []
        for evt in prof.key_averages():
            device_time_total = _evt_device_us(evt)
            if device_time_total <= 0:
                continue
            events.append(
                {
                    "key": evt.key,
                    "count": evt.count,
                    "cpu_time_total_us": evt.cpu_time_total,
                    "device_time_total_us": device_time_total,
                    "self_device_time_total_us": getattr(evt, "self_device_time_total", 0.0),
                }
            )
        _write_json(
            args.json,
            {
                "meta": {
                    "torch_version": torch.__version__,
                    "torch_cuda_version": torch.version.cuda,
                    "device_name": torch.cuda.get_device_name(0),
                    "device_cc": list(torch.cuda.get_device_capability(0)),
                    "height": args.height,
                    "width": args.width,
                    "t": args.t,
                    "iters": args.iters,
                    "pre_iters": args.pre_iters,
                    "dtype": args.dtype,
                    "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                    "wanvae_stream_decode_mode": os.getenv("WANVAE_STREAM_DECODE_MODE"),
                    "wanvae_decode_channels_last_3d": os.getenv("WANVAE_DECODE_CHANNELS_LAST_3D"),
                    "wanvae_conv3d_implicit_spatial_padding": os.getenv(
                        "WANVAE_CONV3D_IMPLICIT_SPATIAL_PADDING"
                    ),
                    "wanvae_upsample_force_fp32": os.getenv("WANVAE_UPSAMPLE_FORCE_FP32"),
                    "record_shapes": bool(args.record_shapes),
                    "with_stack": bool(args.with_stack),
                    "stack_n": args.stack_n if args.with_stack else None,
                    "stack_keys": args.stack_key if args.with_stack else None,
                    "stack_limit": args.stack_limit if args.with_stack else None,
                    "stack_include": args.stack_include if args.with_stack else None,
                    "stack_exclude": args.stack_exclude if args.with_stack else None,
                    "profiled_wall_time_s": dt,
                },
                "events": events,
                "stack_groups": stack_groups,
            },
        )
        print(f"\nWrote JSON: {args.json}")

    if args.summary is not None:
        _write_text(args.summary, "".join(summary_lines))
        print(f"Wrote summary: {args.summary}")

    if args.chrome_trace is not None:
        args.chrome_trace.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(args.chrome_trace))
        print(f"Wrote Chrome trace: {args.chrome_trace}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
