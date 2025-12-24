#!/usr/bin/env python3
"""
Test FA4/CUTE score_mod for KV-cache bias (Kernel B replacement).

This is a standalone test that validates:
1. CUTE score_mod compiles and runs
2. Output matches FlexAttention reference within tolerance
3. Performance comparison (optional)

Usage:
    uv run python scripts/test_fa4_kv_bias.py
"""

import sys
import os
import math
import inspect
from pathlib import Path
import torch

def _extend_flash_attn_path_for_score_mod() -> None:
    import flash_attn as _flash_attn
    import sys

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

import cutlass
import cutlass.cute as cute
import operator

_extend_flash_attn_path_for_score_mod()
from flash_attn.cute.interface import _flash_attn_fwd
if "score_mod" not in inspect.signature(_flash_attn_fwd).parameters:
    raise SystemExit(
        "flash_attn.cute.interface._flash_attn_fwd() does not support score_mod. "
        "Ensure the vendored CuTe sources are present (vendored/flash_attn_cute_score_mod) "
        "or a local flash-attention checkout is available."
    )

# Try to import FlexAttention for reference
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    HAS_FLEX = True
except ImportError:
    HAS_FLEX = False
    print("Warning: FlexAttention not available, skipping reference comparison")


# =============================================================================
# CUTE score_mod for KV-cache bias
# =============================================================================

def make_kv_bias_score_mod(frame_seqlen: int, block_start: int, log_bias: float):
    """
    Factory that creates a score_mod with constants captured in closure.

    This avoids aux_tensors, keeping vec_size=2 for better performance.

    Bias rule (3 regions):
      - Region 1: First frame (kv_idx < frame_seqlen) → NO BIAS
      - Region 2: Past frames (frame_seqlen ≤ kv_idx < block_start) → +log_bias
      - Region 3: Current block (kv_idx ≥ block_start) → NO BIAS
    """
    # Closure constants (compile-time)
    _frame_seqlen = frame_seqlen
    _block_start = max(0, int(block_start))
    _log_bias = log_bias

    @cute.jit
    def score_mod_kv_bias(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        frame_seqlen_tensor = cute.full_like(kv_idx, _frame_seqlen)
        block_start_tensor = cute.full_like(kv_idx, _block_start)
        cond_ge = operator.ge(kv_idx, frame_seqlen_tensor)
        cond_lt = operator.lt(kv_idx, block_start_tensor)
        past_frame_mask = cond_ge & cond_lt

        # Add log_bias only to Region 2
        bias_tensor = cute.full_like(tSrS_ssa, _log_bias)
        return cute.where(past_frame_mask, tSrS_ssa + bias_tensor, tSrS_ssa)

    return score_mod_kv_bias


# =============================================================================
# FlexAttention reference (for correctness check)
# =============================================================================

def make_flex_score_mod(frame_seqlen: int, block_start: int, log_bias: float):
    """Create FlexAttention score_mod for reference."""
    frame_seqlen_t = torch.tensor(frame_seqlen, dtype=torch.int32)
    block_start_t = torch.tensor(block_start, dtype=torch.int32)
    log_bias_t = torch.tensor(log_bias, dtype=torch.float32)

    def score_mod(score, b_idx, h_idx, q_idx, kv_idx):
        return torch.where(
            (kv_idx >= frame_seqlen_t) & (kv_idx < block_start_t),
            score + log_bias_t,
            score,
        )
    return score_mod


# =============================================================================
# Test functions
# =============================================================================

def test_fa4_basic(B=1, H=16, Lq=64, Lk=128, D=128, dtype=torch.bfloat16):
    """Test basic FA4 forward pass with score_mod."""
    print(f"\n=== Basic FA4 test: B={B}, H={H}, Lq={Lq}, Lk={Lk}, D={D} ===")

    device = "cuda"

    # Create test tensors (FA4 expects [B, L, H, D])
    q = torch.randn(B, Lq, H, D, device=device, dtype=dtype)
    k = torch.randn(B, Lk, H, D, device=device, dtype=dtype)
    v = torch.randn(B, Lk, H, D, device=device, dtype=dtype)

    # Test parameters
    frame_seqlen = Lk // 4  # First 1/4 is "first frame"
    block_size = Lk // 4    # Last 1/4 is "current block"
    block_start = Lk - block_size
    kv_cache_attention_bias = 0.3
    log_bias = math.log(kv_cache_attention_bias)

    print(f"  frame_seqlen={frame_seqlen}, block_start={block_start}, log_bias={log_bias:.4f}")

    # Create score_mod
    score_mod = make_kv_bias_score_mod(frame_seqlen, block_start, log_bias)

    # Run FA4 forward
    print("  Running FA4 forward...")
    try:
        out, lse = _flash_attn_fwd(
            q, k, v,
            score_mod=score_mod,
            causal=False,
            return_lse=True,
        )
        print(f"  Output shape: {out.shape}")
        print(f"  Output dtype: {out.dtype}")
        print(f"  Output range: [{out.min().item():.4f}, {out.max().item():.4f}]")
        print("  FA4 forward: PASS")
        return True
    except Exception as e:
        print(f"  FA4 forward: FAIL - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fa4_vs_flex(B=1, H=16, Lq=64, Lk=128, D=128, dtype=torch.bfloat16):
    """Test FA4 output matches FlexAttention reference."""
    if not HAS_FLEX:
        print("\n=== Skipping FlexAttention comparison (not available) ===")
        return True

    print(f"\n=== FA4 vs FlexAttention: B={B}, H={H}, Lq={Lq}, Lk={Lk}, D={D} ===")

    device = "cuda"

    # Create test tensors
    q = torch.randn(B, Lq, H, D, device=device, dtype=dtype)
    k = torch.randn(B, Lk, H, D, device=device, dtype=dtype)
    v = torch.randn(B, Lk, H, D, device=device, dtype=dtype)

    # Test parameters
    frame_seqlen = Lk // 4
    block_size = Lk // 4
    block_start = Lk - block_size
    kv_cache_attention_bias = 0.3
    log_bias = math.log(kv_cache_attention_bias)

    print(f"  frame_seqlen={frame_seqlen}, block_start={block_start}, log_bias={log_bias:.4f}")

    # FA4 path
    score_mod_cute = make_kv_bias_score_mod(frame_seqlen, block_start, log_bias)
    out_fa4, _ = _flash_attn_fwd(
        q, k, v,
        score_mod=score_mod_cute,
        causal=False,
        return_lse=True,
    )

    # FlexAttention path (needs [B, H, L, D] layout)
    score_mod_flex = make_flex_score_mod(frame_seqlen, block_start, log_bias)
    q_flex = q.transpose(1, 2).contiguous()  # [B, H, Lq, D]
    k_flex = k.transpose(1, 2).contiguous()  # [B, H, Lk, D]
    v_flex = v.transpose(1, 2).contiguous()  # [B, H, Lk, D]

    out_flex = flex_attention(q_flex, k_flex, v_flex, score_mod=score_mod_flex)
    out_flex = out_flex.transpose(1, 2)  # Back to [B, L, H, D]

    # Compare
    diff = (out_fa4 - out_flex).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"  Max diff: {max_diff:.6f}")
    print(f"  Mean diff: {mean_diff:.6f}")

    # Tolerance for bf16
    rtol = 0.02
    atol = 0.01
    if torch.allclose(out_fa4, out_flex, rtol=rtol, atol=atol):
        print("  Comparison: PASS")
        return True
    else:
        print(f"  Comparison: FAIL (rtol={rtol}, atol={atol})")
        # Show where differences are largest
        flat_diff = diff.flatten()
        top_k = torch.topk(flat_diff, min(5, flat_diff.numel()))
        print(f"  Top 5 diffs: {top_k.values.tolist()}")
        return False


def test_real_shape():
    """Test with real Krea shape: B=1, H=16, Lq=4680, Lk=9360, D=128."""
    print("\n=== Real shape test: B=1, H=16, Lq=4680, Lk=9360, D=128 ===")

    B, H, Lq, Lk, D = 1, 16, 4680, 9360, 128
    dtype = torch.bfloat16
    device = "cuda"

    # Real parameters
    frame_seqlen = 1560  # 30 * 52 for typical resolution
    num_frame_per_block = 1
    block_size = frame_seqlen * num_frame_per_block
    block_start = Lk - block_size
    kv_cache_attention_bias = 0.3
    log_bias = math.log(kv_cache_attention_bias)

    print(f"  frame_seqlen={frame_seqlen}, block_size={block_size}")

    # Create test tensors (FA4 layout: [B, L, H, D])
    q = torch.randn(B, Lq, H, D, device=device, dtype=dtype)
    k = torch.randn(B, Lk, H, D, device=device, dtype=dtype)
    v = torch.randn(B, Lk, H, D, device=device, dtype=dtype)

    # ===== FA4 Benchmark =====
    score_mod_cute = make_kv_bias_score_mod(frame_seqlen, block_size, log_bias)

    # Warmup
    print("  FA4 warmup (JIT compilation)...")
    for _ in range(3):
        out, _ = _flash_attn_fwd(q, k, v, score_mod=score_mod_cute, causal=False, return_lse=True)
    torch.cuda.synchronize()

    # Benchmark FA4
    import time
    torch.cuda.synchronize()
    start = time.perf_counter()
    n_iters = 50
    for _ in range(n_iters):
        out, _ = _flash_attn_fwd(q, k, v, score_mod=score_mod_cute, causal=False, return_lse=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    fa4_ms = (elapsed / n_iters) * 1000
    print(f"  FA4 + score_mod: {fa4_ms:.3f} ms/iter")

    # ===== FlexAttention Benchmark =====
    if HAS_FLEX:
        # FlexAttention needs [B, H, L, D] layout
        q_flex = q.transpose(1, 2).contiguous()
        k_flex = k.transpose(1, 2).contiguous()
        v_flex = v.transpose(1, 2).contiguous()
        score_mod_flex = make_flex_score_mod(frame_seqlen, block_start, log_bias)

        # Warmup
        print("  FlexAttention warmup...")
        for _ in range(3):
            out_flex = flex_attention(q_flex, k_flex, v_flex, score_mod=score_mod_flex)
        torch.cuda.synchronize()

        # Benchmark FlexAttention
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_iters):
            out_flex = flex_attention(q_flex, k_flex, v_flex, score_mod=score_mod_flex)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        flex_ms = (elapsed / n_iters) * 1000
        print(f"  FlexAttention: {flex_ms:.3f} ms/iter")
        print(f"  FA4 speedup vs Flex: {flex_ms / fa4_ms:.2f}x")

    print(f"\n  === Summary ===")
    print(f"  FA4 + score_mod: {fa4_ms:.3f} ms")
    print(f"  Triton Kernel B: ~1.022 ms (from tune_kernel_b.py)")
    print(f"  FA4 speedup vs Triton: {1.022 / fa4_ms:.2f}x")

    return True


def main():
    print("=" * 60)
    print("FA4/CUTE KV-bias score_mod Test")
    print("=" * 60)

    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return 1

    print(f"CUDA device: {torch.cuda.get_device_name()}")
    print(f"Compute capability: {torch.cuda.get_device_capability()}")

    results = []

    # Test 1: Basic FA4 forward
    results.append(("Basic FA4", test_fa4_basic()))

    # Test 2: Compare with FlexAttention
    results.append(("FA4 vs Flex", test_fa4_vs_flex()))

    # Test 3: Real shape
    results.append(("Real shape", test_real_shape()))

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        all_pass = all_pass and passed

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
