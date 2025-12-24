#!/usr/bin/env python3
"""
Smoke test for FA4/CuTe `score_mod` on the current GPU.

This isolates whether failures come from:
- score_mod plumbing itself (identity)
- basic arithmetic in score_mod (add const)
- predicate ops in score_mod (kv_idx comparisons + where)

Usage:
  TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas uv run python scripts/test_fa4_score_mod_smoke.py
"""

from __future__ import annotations

import inspect
import math
import operator
import sys
from pathlib import Path

import torch


def _extend_flash_attn_path_for_score_mod() -> None:
    import flash_attn as _flash_attn

    try:
        base_path = Path(__file__).resolve()
    except Exception:
        return
    for parent in base_path.parents:
        vendored = parent / "vendored" / "flash_attn_cute_score_mod" / "flash_attn"
        if vendored.is_dir():
            vendored_str = str(vendored)
            if vendored_str not in _flash_attn.__path__:
                _flash_attn.__path__.insert(0, vendored_str)
            cute_mod = sys.modules.get("flash_attn.cute")
            if cute_mod is not None and hasattr(cute_mod, "__path__"):
                vendored_cute = str(vendored / "cute")
                if vendored_cute not in cute_mod.__path__:
                    cute_mod.__path__.insert(0, vendored_cute)
            sys.modules.pop("flash_attn.cute.interface", None)
            return
        for repo_dir in ("flash-attention", "flash-attention.bak"):
            candidate = parent / repo_dir / "flash_attn"
            if candidate.is_dir():
                candidate_str = str(candidate)
                if candidate_str not in _flash_attn.__path__:
                    _flash_attn.__path__.insert(0, candidate_str)
                cute_mod = sys.modules.get("flash_attn.cute")
                if cute_mod is not None and hasattr(cute_mod, "__path__"):
                    candidate_cute = str(candidate / "cute")
                    if candidate_cute not in cute_mod.__path__:
                        cute_mod.__path__.insert(0, candidate_cute)
                sys.modules.pop("flash_attn.cute.interface", None)
                return


def _run_one(name: str, fn) -> bool:
    print(f"\n=== {name} ===")
    try:
        fn()
        print("PASS")
        return True
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {e}")
        return False


def main() -> int:
    print("============================================================")
    print("FA4/CUTE score_mod Smoke Test")
    print("============================================================")
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return 2
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"Compute capability: {torch.cuda.get_device_capability(0)}")

    _extend_flash_attn_path_for_score_mod()
    from flash_attn.cute.interface import _flash_attn_fwd

    if "score_mod" not in inspect.signature(_flash_attn_fwd).parameters:
        print("ERROR: _flash_attn_fwd has no score_mod parameter in this environment")
        return 2

    import cutlass.cute as cute

    B, H, Lq, Lk, D = 1, 16, 64, 128, 128
    q = torch.randn(B, Lq, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, Lk, H, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, Lk, H, D, device="cuda", dtype=torch.bfloat16)

    def call(score_mod):
        _flash_attn_fwd(q, k, v, score_mod=score_mod, causal=False, return_lse=False)

    @cute.jit
    def score_mod_identity(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        return tSrS_ssa

    _log_bias = math.log(0.3)

    @cute.jit
    def score_mod_add_const(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        bias_tensor = cute.full_like(tSrS_ssa, _log_bias)
        return tSrS_ssa + bias_tensor

    _frame_seqlen = 32
    _block_start = 96

    @cute.jit
    def score_mod_conditional(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        frame_seqlen_tensor = cute.full_like(kv_idx, _frame_seqlen)
        block_start_tensor = cute.full_like(kv_idx, _block_start)
        cond_ge = operator.ge(kv_idx, frame_seqlen_tensor)
        cond_lt = operator.lt(kv_idx, block_start_tensor)
        mask = cond_ge & cond_lt
        bias_tensor = cute.full_like(tSrS_ssa, _log_bias)
        return cute.where(mask, tSrS_ssa + bias_tensor, tSrS_ssa)

    ok = True
    ok &= _run_one("identity", lambda: call(score_mod_identity))
    ok &= _run_one("add const", lambda: call(score_mod_add_const))
    ok &= _run_one("conditional where", lambda: call(score_mod_conditional))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
