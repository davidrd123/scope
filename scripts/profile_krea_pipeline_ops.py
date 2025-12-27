#!/usr/bin/env python3
"""
Operator-level (CUDA) profiling for the Krea realtime pipeline.

This complements the lightweight CUDA-event block profilers by answering:
  - Which *ops/kernels* dominate GPU time inside denoise/transformer?
  - Is time going to attention vs norms vs GEMMs vs decode convs?

Typical B300 usage (cu130 env):

  SCOPE_KV_BIAS_BACKEND=fa4 \
  TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas \
  DISABLE_FLEX_ATTENTION_COMPILE=1 \
  WANVAE_STREAM_DECODE_MODE=chunk \
  PYTHONPATH=src .venv-b300-cu130-decode/bin/python scripts/profile_krea_pipeline_ops.py \
    --height 320 --width 576 \
    --quantization none \
    --kv-cache-attention-bias 0.3 \
    --iters 1 \
    --json outputs/b300_cu130_ops_profile.json

To profile the *compiled* attention blocks (useful for seeing fusion effects), add:

  --compile

And optionally set a specific compile mode:

  --compile-mode reduce-overhead
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Operator-level profiler for Krea pipeline")
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--iters", type=int, default=1, help="Number of profiled pipeline calls")
    parser.add_argument("--pre-iters", type=int, default=1, help="Warm iterations outside profiler")
    parser.add_argument("--kv-cache-attention-bias", type=float, default=0.3)
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile for the diffusion attention blocks (same as the server's SCOPE_COMPILE_KREA_PIPELINE=1).",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        default=None,
        help="Optional torch.compile mode (sets SCOPE_TORCH_COMPILE_MODE for this run).",
    )
    parser.add_argument(
        "--kv-bias-backend",
        choices=["auto", "fa4", "flash", "triton", "flex"],
        default="auto",
        help="Set SCOPE_KV_BIAS_BACKEND for this run (read at import time).",
    )
    parser.add_argument(
        "--quantization",
        choices=["fp8_e4m3fn", "none"],
        default="none",
    )
    parser.add_argument("--prompt", type=str, default="a majestic sunset")
    parser.add_argument(
        "--cudnn-benchmark",
        action="store_true",
        help="Enable torch.backends.cudnn.benchmark (can improve conv performance).",
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
        default=5,
        help="Stack depth for grouped-by-stack summaries (used with --with-stack).",
    )
    parser.add_argument(
        "--stack-key",
        action="append",
        default=None,
        help="Repeatable: print a grouped-by-stack summary for this op key (e.g. --stack-key aten::copy_). "
        "Defaults to aten::copy_, aten::_to_copy, aten::to, and aten::fill_.",
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
        help="Repeatable: only keep stack groups whose frames include this substring. "
        "Example: --stack-include CausalWanSelfAttention",
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
        help="Optional: write aggregated key_averages() to JSON.",
    )
    return parser.parse_args()


def _cuda_unavailable_hint(torch) -> str:
    import os as _os

    dev_checks = []
    for path in ("/dev/nvidiactl", "/dev/nvidia0", "/dev/nvidia-uvm"):
        try:
            fd = _os.open(path, _os.O_RDWR)
            _os.close(fd)
            dev_checks.append(f"{path}=rw_ok")
        except FileNotFoundError:
            dev_checks.append(f"{path}=missing")
        except PermissionError:
            dev_checks.append(f"{path}=rw_denied")
        except Exception as e:
            dev_checks.append(f"{path}=err:{type(e).__name__}")

    return (
        "CUDA not available.\n"
        f"- torch: {torch.__version__} (cuda={torch.version.cuda})\n"
        f"- /dev access: {', '.join(dev_checks)}\n"
        "If /dev/* is rw_denied, you likely don't have GPU *compute* permissions "
        "(device-cgroup / container missing GPU access). If NVML is wedged "
        "(nvidia-smi errors), a host driver restart/reboot may be required."
    )


def _maybe_set_default_env() -> None:
    # Conservative defaults for SM103 (B300).
    try:
        import torch

        is_sm103 = (
            torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (10, 3)
        )
    except Exception:
        is_sm103 = False

    if not is_sm103:
        return

    os.environ.setdefault("DISABLE_FLEX_ATTENTION_COMPILE", "1")
    os.environ.setdefault("WANVAE_STREAM_DECODE_MODE", "chunk")
    os.environ.setdefault("WANVAE_DECODE_CHANNELS_LAST_3D", "1")
    os.environ.setdefault("WANVAE_RESAMPLE_ENSURE_CONTIGUOUS", "1")
    os.environ.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda-12.9/bin/ptxas")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


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

    if args.kv_bias_backend != "auto":
        os.environ["SCOPE_KV_BIAS_BACKEND"] = args.kv_bias_backend
    if args.compile_mode is not None:
        os.environ["SCOPE_TORCH_COMPILE_MODE"] = args.compile_mode

    _maybe_set_default_env()

    import torch
    from omegaconf import OmegaConf
    from torch.profiler import ProfilerActivity, profile

    from scope.core.config import get_model_file_path, get_models_dir
    from scope.core.pipelines.krea_realtime_video.pipeline import KreaRealtimeVideoPipeline
    from scope.core.pipelines.utils import Quantization

    if not torch.cuda.is_available():
        raise SystemExit(_cuda_unavailable_hint(torch))

    torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)

    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"Compute capability: {torch.cuda.get_device_capability(0)}")
    print(f"torch: {torch.__version__} (cuda={torch.version.cuda})")
    print(f"cudnn.benchmark: {torch.backends.cudnn.benchmark}")
    print(f"compile={bool(args.compile)}")
    if os.getenv("SCOPE_TORCH_COMPILE_MODE"):
        print(f"SCOPE_TORCH_COMPILE_MODE={os.getenv('SCOPE_TORCH_COMPILE_MODE')}")
    print(f"SCOPE_KV_BIAS_BACKEND={os.getenv('SCOPE_KV_BIAS_BACKEND')}")
    print(f"DISABLE_FLEX_ATTENTION_COMPILE={os.getenv('DISABLE_FLEX_ATTENTION_COMPILE')}")
    print(f"WANVAE_STREAM_DECODE_MODE={os.getenv('WANVAE_STREAM_DECODE_MODE')}")
    print(f"WANVAE_DECODE_CHANNELS_LAST_3D={os.getenv('WANVAE_DECODE_CHANNELS_LAST_3D')}")
    print(f"WANVAE_RESAMPLE_ENSURE_CONTIGUOUS={os.getenv('WANVAE_RESAMPLE_ENSURE_CONTIGUOUS')}")
    print(f"TRITON_PTXAS_PATH={os.getenv('TRITON_PTXAS_PATH')}")
    if args.with_stack:
        print(f"with_stack=True (stack_n={args.stack_n})")

    quantization = Quantization.FP8_E4M3FN if args.quantization == "fp8_e4m3fn" else None

    config = OmegaConf.create(
        {
            "model_dir": str(get_models_dir()),
            "generator_path": str(
                get_model_file_path("krea-realtime-video/krea-realtime-video-14b.safetensors")
            ),
            "text_encoder_path": str(
                get_model_file_path("WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors")
            ),
            "tokenizer_path": str(get_model_file_path("Wan2.1-T2V-1.3B/google/umt5-xxl")),
            "vae_path": str(get_model_file_path("Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")),
            "model_config": OmegaConf.load(
                Path(__file__).resolve().parents[1] / "src/scope/core/pipelines/krea_realtime_video/model.yaml"
            ),
            "height": args.height,
            "width": args.width,
        }
    )

    pipeline = KreaRealtimeVideoPipeline(
        config,
        quantization=quantization,
        compile=bool(args.compile),
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
    )

    prompts = [{"text": args.prompt, "weight": 100}]

    if args.pre_iters:
        print(f"Pre-warm (outside profiler): {args.pre_iters} iteration(s)")
        for _ in range(args.pre_iters):
            pipeline(prompts=prompts, kv_cache_attention_bias=args.kv_cache_attention_bias)
        torch.cuda.synchronize()

    print(f"Profiling: {args.iters} iteration(s)")
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    experimental_config = None
    if args.with_stack:
        # `with_stack=True` alone isn't enough in some Kineto builds; verbose mode
        # ensures stack frames are recorded in the trace.
        try:
            experimental_config = torch._C._profiler._ExperimentalConfig(verbose=True)
        except Exception:
            experimental_config = None

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_stack=bool(args.with_stack),
        experimental_config=experimental_config,
    ) as prof:
        for _ in range(args.iters):
            pipeline(prompts=prompts, kv_cache_attention_bias=args.kv_cache_attention_bias)
        torch.cuda.synchronize()

    dt = time.perf_counter() - t0
    print(f"Profiled wall time: {dt:.3f}s")
    print("")
    # On newer PyTorch versions the profiler exposes GPU time as `device_time_total`
    # (not `cuda_time_total`). The table renderer will label it as CUDA for CUDA devices.
    print(prof.key_averages().table(sort_by="device_time_total", row_limit=args.row_limit))

    summary_lines: list[str] = []
    stack_groups: dict[str, list[dict]] | None = None

    events_for_summary = list(prof.key_averages())
    total_self_device_us = sum(
        float(getattr(evt, "self_device_time_total", 0.0) or 0.0)
        for evt in events_for_summary
    )
    top_events = sorted(
        events_for_summary,
        key=lambda e: float(getattr(e, "self_device_time_total", 0.0) or 0.0),
        reverse=True,
    )[: min(len(events_for_summary), 25)]

    if args.summary is not None:
        summary_lines.append("# Krea pipeline op profile\n\n")
        summary_lines.append("## Meta\n")
        summary_lines.append(f"- torch: `{torch.__version__}` (cuda `{torch.version.cuda}`)\n")
        summary_lines.append(
            f"- device: `{torch.cuda.get_device_name(0)}` cc={torch.cuda.get_device_capability(0)}\n"
        )
        summary_lines.append(f"- compile: `{bool(args.compile)}`\n")
        summary_lines.append(f"- kv_cache_attention_bias: `{args.kv_cache_attention_bias}`\n")
        summary_lines.append(f"- SCOPE_KV_BIAS_BACKEND: `{os.getenv('SCOPE_KV_BIAS_BACKEND')}`\n")
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
                "aten::copy_",
                "aten::_to_copy",
                "aten::to",
                "aten::fill_",
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

            rows.sort(
                key=_evt_device_us,
                reverse=True,
            )
            print(f"- {key}: top {min(len(rows), args.stack_limit)}")
            stack_groups[key] = []
            if args.summary is not None:
                summary_lines.append(f"\n## Stack groups: `{key}`\n")
                total_device_us = sum(_evt_device_us(evt) for evt in rows)
                total_calls = sum(int(evt.count) for evt in rows)
                summary_lines.append(
                    f"\nFiltered totals: device_ms={total_device_us/1e3:.3f}, calls={total_calls}\n"
                )
            for evt in rows[: args.stack_limit]:
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
                    "iters": args.iters,
                    "pre_iters": args.pre_iters,
                    "kv_cache_attention_bias": args.kv_cache_attention_bias,
                    "kv_bias_backend": os.getenv("SCOPE_KV_BIAS_BACKEND"),
                    "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                    "disable_flex_attention_compile": os.getenv("DISABLE_FLEX_ATTENTION_COMPILE"),
                    "wanvae_stream_decode_mode": os.getenv("WANVAE_STREAM_DECODE_MODE"),
                    "triton_ptxas_path": os.getenv("TRITON_PTXAS_PATH"),
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
