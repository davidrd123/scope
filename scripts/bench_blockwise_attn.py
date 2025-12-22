#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math

import torch
from torch.nn.attention.flex_attention import flex_attention

from scope.core.pipelines.krea_realtime_video.modules.causal_model import (
    get_block_mask,
    get_sdpa_mask,
)


def _parse_dtype(value: str) -> torch.dtype:
    value = value.lower()
    if value == "bf16":
        return torch.bfloat16
    if value == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {value}")


def _pad_to_multiple(length: int, multiple: int) -> int:
    return int(math.ceil(length / multiple) * multiple)


def _pad_qkv(q: torch.Tensor, target_len: int) -> torch.Tensor:
    if q.shape[2] >= target_len:
        return q
    pad = target_len - q.shape[2]
    pad_shape = list(q.shape)
    pad_shape[2] = pad
    padding = torch.zeros(pad_shape, device=q.device, dtype=q.dtype)
    return torch.cat([q, padding], dim=2)


def _make_qkv(batch: int, heads: int, length: int, head_dim: int, dtype, device):
    q = torch.randn((batch, heads, length, head_dim), device=device, dtype=dtype)
    k = torch.randn((batch, heads, length, head_dim), device=device, dtype=dtype)
    v = torch.randn((batch, heads, length, head_dim), device=device, dtype=dtype)
    return q, k, v


def _time_cuda(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def run_block_mask(args) -> None:
    dtype = _parse_dtype(args.dtype)
    device = torch.device(args.device)

    q_len = args.frames * args.frame_seqlen
    padded_len = _pad_to_multiple(q_len, args.alignment)

    q, k, v = _make_qkv(args.batch, args.heads, q_len, args.head_dim, dtype, device)
    q = _pad_qkv(q, padded_len).contiguous()
    k = _pad_qkv(k, padded_len).contiguous()
    v = _pad_qkv(v, padded_len).contiguous()

    block_mask = get_block_mask(
        str(device),
        num_frames=args.frames,
        frame_seqlen=args.frame_seqlen,
        num_frame_per_block=args.num_frame_per_block,
        local_attn_size=args.local_attn_size,
    )

    attn = flex_attention
    if args.compile:
        attn = torch.compile(attn, dynamic=False, mode="max-autotune-no-cudagraphs")

    def _call():
        out = attn(
            query=q,
            key=k,
            value=v,
            block_mask=block_mask,
            kernel_options={"BLOCKS_ARE_CONTIGUOUS": True},
        )
        return out[:, :, :q_len]

    avg_ms = _time_cuda(_call, args.warmup, args.iters)
    tokens = args.batch * q_len
    tokens_per_s = tokens / (avg_ms / 1e3)

    print("mode=block_mask")
    print(f"shape: B={args.batch} H={args.heads} L={q_len} D={args.head_dim}")
    print(f"padded_len={padded_len} alignment={args.alignment}")
    print(f"avg_ms={avg_ms:.3f} tokens/s={tokens_per_s:.1f}")

    if args.check:
        if q_len > args.max_check_tokens:
            print("check skipped: L too large for SDPA")
            return
        mask = get_sdpa_mask(
            str(device),
            num_frames=args.frames,
            frame_seqlen=args.frame_seqlen,
            num_frame_per_block=args.num_frame_per_block,
            local_attn_size=args.local_attn_size,
            dtype=torch.bool,
        )
        out_ref = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False
        )
        out_test = _call()
        max_err = (out_test - out_ref[:, :, :q_len]).abs().max().item()
        mean_err = (out_test - out_ref[:, :, :q_len]).abs().mean().item()
        print(f"check max_err={max_err:.3e} mean_err={mean_err:.3e}")


def run_bias(args) -> None:
    dtype = _parse_dtype(args.dtype)
    device = torch.device(args.device)

    q_frames = args.q_frames if args.q_frames is not None else args.frames
    kv_frames = args.kv_frames if args.kv_frames is not None else args.frames

    q_len = q_frames * args.frame_seqlen
    kv_len = kv_frames * args.frame_seqlen
    padded_q_len = _pad_to_multiple(q_len, args.alignment)
    padded_k_len = _pad_to_multiple(kv_len, args.alignment)
    if args.pad_q_to_k:
        padded_len = _pad_to_multiple(max(q_len, kv_len), args.alignment)
        padded_q_len = padded_len
        padded_k_len = padded_len

    q, _, _ = _make_qkv(args.batch, args.heads, q_len, args.head_dim, dtype, device)
    _, k, v = _make_qkv(args.batch, args.heads, kv_len, args.head_dim, dtype, device)
    q = _pad_qkv(q, padded_q_len).contiguous()
    k = _pad_qkv(k, padded_k_len).contiguous()
    v = _pad_qkv(v, padded_k_len).contiguous()

    log_scale = math.log(args.kv_cache_attention_bias)
    frame_seqlen_tensor = torch.as_tensor(
        args.frame_seqlen, dtype=torch.int32, device=device
    )
    cache_current_block_start = kv_len - args.frame_seqlen * args.num_frame_per_block
    cache_current_block_start_tensor = torch.as_tensor(
        cache_current_block_start, dtype=torch.int32, device=device
    ).squeeze()
    log_scale_tensor = torch.as_tensor(log_scale, dtype=dtype, device=device)

    def score_mod(score, b_idx, h_idx, q_idx, kv_idx):
        return torch.where(
            (kv_idx >= frame_seqlen_tensor)
            & (kv_idx < cache_current_block_start_tensor),
            score + log_scale_tensor,
            score,
        )

    attn = flex_attention
    if args.compile:
        attn = torch.compile(attn, dynamic=False, mode="max-autotune-no-cudagraphs")

    def _call():
        out = attn(
            query=q,
            key=k,
            value=v,
            score_mod=score_mod,
        )
        return out[:, :, :q_len]

    avg_ms = _time_cuda(_call, args.warmup, args.iters)
    tokens = args.batch * q_len
    tokens_per_s = tokens / (avg_ms / 1e3)

    print("mode=bias")
    print(f"shape: B={args.batch} H={args.heads} Lq={q_len} Lk={kv_len} D={args.head_dim}")
    print(
        f"padded_q_len={padded_q_len} padded_k_len={padded_k_len} alignment={args.alignment}"
    )
    print(f"kv_cache_attention_bias={args.kv_cache_attention_bias}")
    print(f"avg_ms={avg_ms:.3f} tokens/s={tokens_per_s:.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Microbench blockwise/bias attention")
    parser.add_argument("--mode", choices=["block_mask", "bias"], default="block_mask")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--q-frames", type=int, default=None)
    parser.add_argument("--kv-frames", type=int, default=None)
    parser.add_argument("--frame-seqlen", type=int, default=1560)
    parser.add_argument("--num-frame-per-block", type=int, default=3)
    parser.add_argument("--local-attn-size", type=int, default=-1)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--alignment", type=int, default=128)
    parser.add_argument(
        "--pad-q-to-k",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pad Q to K length for bias path (default: true)",
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--max-check-tokens", type=int, default=1024)
    parser.add_argument("--kv-cache-attention-bias", type=float, default=0.3)

    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    if args.mode == "block_mask":
        run_block_mask(args)
    else:
        run_bias(args)


if __name__ == "__main__":
    main()
