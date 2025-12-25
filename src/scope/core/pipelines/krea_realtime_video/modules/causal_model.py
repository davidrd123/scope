# Modified from https://github.com/krea-ai/realtime-video
import atexit
import functools
import logging
import math
import os
import subprocess
from collections import defaultdict

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
)

from scope.core.pipelines.wan2_1.modules.attention import attention

logger = logging.getLogger(__name__)

def _is_sm103() -> bool:
    """Return True on B300 (SM103)."""
    try:
        return (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability(0) == (10, 3)
        )
    except Exception:
        return False


# KV-cache bias backend selection
# SCOPE_KV_BIAS_BACKEND:
# - fa4:   FA4/CUTE score_mod (fastest when available; requires CUTLASS DSL)
# - flash: Bias via FlashAttention segment-combine (no score_mod; SM103 default)
# - triton: Triton Kernel B (SM100 default)
# - flex:  torch.nn.attention.flex_attention fallback
_env_kv_bias_backend = os.getenv("SCOPE_KV_BIAS_BACKEND")
_KV_BIAS_BACKEND = (
    _env_kv_bias_backend.lower()
    if _env_kv_bias_backend
    else ("flash" if _is_sm103() else "triton")
)

# FA4 + CUTE score_mod: 1.89x faster than Triton Kernel B on B200
_fa4_available = False
_fa4_fwd = None
_fa4_fwd_opaque = None
_fa4_score_mod_cache = {}
_fa4_bias_tripped = False

if _KV_BIAS_BACKEND == "fa4":
    try:
        import inspect
        import operator
        import sys
        from pathlib import Path

        import flash_attn as _flash_attn

        def _extend_flash_attn_path_for_score_mod() -> None:
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

                    # If `flash_attn.cute` was already imported (likely from the wheel),
                    # make sure *its* module search path also prefers the vendored code.
                    cute_mod = sys.modules.get("flash_attn.cute")
                    if cute_mod is not None and hasattr(cute_mod, "__path__"):
                        vendored_cute = str(vendored / "cute")
                        if vendored_cute not in cute_mod.__path__:
                            cute_mod.__path__.insert(0, vendored_cute)

                    # Force re-import of CuTe modules so we don't keep cached wheel modules.
                    for mod_name in list(sys.modules.keys()):
                        if mod_name == "flash_attn.cute" or mod_name.startswith("flash_attn.cute."):
                            sys.modules.pop(mod_name, None)
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
                        for mod_name in list(sys.modules.keys()):
                            if mod_name == "flash_attn.cute" or mod_name.startswith("flash_attn.cute."):
                                sys.modules.pop(mod_name, None)
                        return

        _extend_flash_attn_path_for_score_mod()
        from flash_attn.cute.interface import _flash_attn_fwd as _fa4_fwd
        # CuTe uses DLPack + Python glue which torch.compile / Dynamo cannot safely trace.
        # Keep the call opaque so we can still compile surrounding regions.
        def _fa4_fwd_opaque(*args, **kwargs):
            return _fa4_fwd(*args, **kwargs)

        try:
            import torch._dynamo  # type: ignore

            _fa4_fwd_opaque = torch._dynamo.disable(_fa4_fwd_opaque)  # type: ignore[attr-defined]
        except Exception:
            pass
        if "score_mod" not in inspect.signature(_fa4_fwd).parameters:
            raise ImportError(
                "FA4/CUTE score_mod requires the vendored (or local) flash-attention CuTe sources; "
                "_flash_attn_fwd() lacks score_mod in this environment."
            )
        import cutlass
        import cutlass.cute as cute
        _fa4_available = True
        logger.info("FA4/CUTE score_mod enabled for KV-cache attention bias (1.89x faster than Triton)")
    except Exception as e:
        fallback_backend = "flash" if _is_sm103() else "triton"
        logger.warning(f"FA4/CUTE not available, falling back to {fallback_backend}: {e}")
        _KV_BIAS_BACKEND = fallback_backend

# Triton Kernel B: fast fallback for KV-cache bias path
USE_TRITON_KERNEL_B = _KV_BIAS_BACKEND in ("triton", "fa4", "flash")  # fa4/flash fall back to triton
_triton_kernel_b = None

if USE_TRITON_KERNEL_B:
    try:
        from scope.core.kernels import triton_kernel_b as _triton_kernel_b
        logger.info("Triton Kernel B available for KV-cache attention bias")
    except ImportError as e:
        logger.warning(f"Triton Kernel B not available: {e}")
        USE_TRITON_KERNEL_B = False


def _get_fa4_score_mod(frame_seqlen: int, block_start: int, log_bias: float):
    """
    Get or create a CUTE score_mod for FA4 KV-cache bias.

    Uses closure constants to avoid aux_tensors (keeps vec_size=2 for better perf).
    Caches compiled score_mods by (frame_seqlen, block_start, log_bias).

    Note: `block_start` should be `max(0, Lk - block_size)` computed on the Python
    side for the current KV cache length. Capturing it as a closure constant avoids
    using `seqlen_info.seqlen_k` (runtime values) inside the score_mod, which can be
    brittle across cutlass-dsl versions.
    """
    if not _fa4_available:
        return None

    # Round log_bias to avoid cache explosion
    cache_key = (int(frame_seqlen), int(block_start), round(log_bias, 6))
    if cache_key in _fa4_score_mod_cache:
        return _fa4_score_mod_cache[cache_key]

    # Closure constants (compile-time)
    _frame_seqlen = frame_seqlen
    _block_start = max(0, int(block_start))
    _log_bias = log_bias

    @cute.jit
    def score_mod_kv_bias(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        frame_seqlen_tensor = cute.full_like(kv_idx, _frame_seqlen)
        block_start_tensor = cute.full_like(kv_idx, _block_start)
        bias_tensor = cute.full_like(tSrS_ssa, _log_bias)
        # Apply bias only to Region 2:
        #   kv_idx in [frame_seqlen, block_start)
        # Avoid boolean-and on SSA values (can trigger MLIR dominance issues in some cutlass-dsl builds).
        biased = cute.where(
            operator.ge(kv_idx, frame_seqlen_tensor),
            tSrS_ssa + bias_tensor,
            tSrS_ssa,
        )
        return cute.where(operator.lt(kv_idx, block_start_tensor), biased, tSrS_ssa)

    _fa4_score_mod_cache[cache_key] = score_mod_kv_bias
    return score_mod_kv_bias


try:
    import torch._dynamo  # type: ignore

    _get_fa4_score_mod = torch._dynamo.disable(_get_fa4_score_mod)  # type: ignore[attr-defined]
except Exception:
    pass


_flash_bias_fa4_fwd = None
_flash_bias_tripped = False


def _get_fa4_fwd():
    global _flash_bias_fa4_fwd
    if _flash_bias_fa4_fwd is not None:
        return _flash_bias_fa4_fwd
    try:
        import inspect
        from flash_attn.cute.interface import _flash_attn_fwd

        if "return_lse" not in inspect.signature(_flash_attn_fwd).parameters:
            # Older flash-attn wheels expose a CuTe _flash_attn_fwd without `return_lse`.
            # In that case, fall back to the compiled varlen kernel which returns LSE.
            _flash_bias_fa4_fwd = None
            return None

        _flash_bias_fa4_fwd = _flash_attn_fwd
    except Exception:
        _flash_bias_fa4_fwd = None
    return _flash_bias_fa4_fwd


_cu_seqlens_cache: dict[tuple[str, int, int], torch.Tensor] = {}


def _get_cu_seqlens(device: torch.device, batch_size: int, seqlen: int) -> torch.Tensor:
    key = (str(device), int(batch_size), int(seqlen))
    cached = _cu_seqlens_cache.get(key)
    if cached is not None:
        return cached
    cu = torch.arange(
        0,
        (batch_size + 1) * seqlen,
        step=seqlen,
        device=device,
        dtype=torch.int32,
    )
    _cu_seqlens_cache[key] = cu
    return cu


def _flash_attn_with_lse(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, softmax_scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute attention output + LSE for segment-combine KV-cache bias.

    Returns:
        out: [B, Lq, H, D]
        lse: [B, Lq, H] (float32)
    """
    fa4_fwd = _get_fa4_fwd()
    if fa4_fwd is not None:
        out, lse = fa4_fwd(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            causal=False,
            return_lse=True,
        )
        if lse is None:
            raise RuntimeError("FA4 _flash_attn_fwd returned lse=None with return_lse=True")
        if lse.dim() == 3:
            # [B, H, Lq] -> [B, Lq, H]
            lse = lse.transpose(1, 2)
        elif lse.dim() == 2:
            # [H, total_q] -> [B, Lq, H]
            b, lq = q.shape[:2]
            lse = lse.transpose(0, 1).reshape(b, lq, -1)
        else:
            raise RuntimeError(f"Unexpected lse shape from FA4: {tuple(lse.shape)}")
        return out, lse

    from flash_attn.flash_attn_interface import _flash_attn_varlen_forward

    b, lq, _h, _d = q.shape
    lk = k.shape[1]
    q_flat = q.reshape(b * lq, q.shape[2], q.shape[3])
    k_flat = k.reshape(b * lk, k.shape[2], k.shape[3])
    v_flat = v.reshape(b * lk, v.shape[2], v.shape[3])
    cu_seqlens_q = _get_cu_seqlens(q.device, b, lq)
    cu_seqlens_k = _get_cu_seqlens(q.device, b, lk)
    out_flat, softmax_lse, _s_dmask, _rng_state = _flash_attn_varlen_forward(
        q_flat,
        k_flat,
        v_flat,
        cu_seqlens_q,
        cu_seqlens_k,
        lq,
        lk,
        0.0,  # dropout_p
        softmax_scale,
        False,  # causal
    )
    out = out_flat.reshape(b, lq, q.shape[2], v.shape[3])
    lse = softmax_lse.transpose(0, 1).reshape(b, lq, -1)
    return out, lse


def _merge_out_lse(
    out_a: torch.Tensor,
    lse_a: torch.Tensor,
    out_b: torch.Tensor,
    lse_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Merge two attention results from disjoint KV segments.

    Shapes:
      out_*: [B, Lq, H, D]
      lse_*: [B, Lq, H] (float32)
    """
    lse_a = lse_a.to(torch.float32)
    lse_b = lse_b.to(torch.float32)
    lse_new = torch.logaddexp(lse_a, lse_b)
    w_a = torch.exp(lse_a - lse_new)
    w_b = torch.exp(lse_b - lse_new)
    out_new = out_a.to(torch.float32) * w_a.unsqueeze(-1) + out_b.to(torch.float32) * w_b.unsqueeze(-1)
    return out_new, lse_new


def _kv_bias_flash_combine(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    frame_seqlen: int,
    current_block_start: int,
    log_bias: float,
) -> torch.Tensor:
    """
    Exact KV-bias attention using FlashAttention segment combine.

    This reproduces the score_mod rule:
      add log_bias for kv_idx in [frame_seqlen, current_block_start)
    without requiring a score_mod-capable kernel.
    """
    lk = int(k.shape[1])
    if lk == 0 or log_bias == 0.0:
        return attention(q, k, v)

    frame_end = max(0, min(int(frame_seqlen), lk))
    block_start = max(0, min(int(current_block_start), lk))
    if block_start <= frame_end:
        # No biased region; matches the score_mod condition always being False.
        return attention(q, k, v)

    softmax_scale = 1.0 / math.sqrt(q.shape[-1])

    out_accum = None
    lse_accum = None
    for seg_start, seg_end, seg_log_w in (
        (0, frame_end, 0.0),
        (frame_end, block_start, float(log_bias)),
        (block_start, lk, 0.0),
    ):
        if seg_end <= seg_start:
            continue
        out_seg, lse_seg = _flash_attn_with_lse(q, k[:, seg_start:seg_end], v[:, seg_start:seg_end], softmax_scale)
        if seg_log_w != 0.0:
            lse_seg = lse_seg + seg_log_w
        if out_accum is None:
            out_accum = out_seg
            lse_accum = lse_seg
            continue
        out_accum, lse_accum = _merge_out_lse(out_accum, lse_accum, out_seg, lse_seg)

    assert out_accum is not None
    return out_accum.to(dtype=q.dtype)


# Simple profiler for timing operations
# Note: Profiling is incompatible with torch.compile - disable if compile is enabled
_PROFILE_ENABLED = os.getenv("PROFILE_ATTENTION", "0") == "1"
_profile_times = defaultdict(float)
_profile_counts = defaultdict(int)
_profile_call_count = 0


def reset_attention_profile() -> None:
    """Reset accumulated attention profiling state (times/counts/call counter)."""
    _profile_times.clear()
    _profile_counts.clear()
    global _profile_call_count
    _profile_call_count = 0


def _should_profile():
    """Check if profiling should run (disabled during torch.compile tracing)."""
    if not _PROFILE_ENABLED:
        return False
    # torch.compiler.is_compiling() returns True during dynamo tracing
    if hasattr(torch, "compiler") and hasattr(torch.compiler, "is_compiling"):
        if torch.compiler.is_compiling():
            return False
    return True


def profile_report():
    """Print profiling report."""
    if not _profile_times:
        return
    total = sum(_profile_times.values())
    logger.info("=== Attention Profiling Report ===")
    for name, time_ms in sorted(_profile_times.items(), key=lambda x: -x[1]):
        count = _profile_counts[name]
        pct = 100 * time_ms / total if total > 0 else 0
        logger.info(f"  {name}: {time_ms:.1f}ms ({pct:.1f}%) [{count} calls, {time_ms/count:.2f}ms/call]")
    logger.info(f"  TOTAL: {total:.1f}ms")

    # Top-level transformer block split (does not double-count nested profiled regions).
    top_total = sum(_profile_times.get(name, 0.0) for name in ("self_attn", "cross_attn", "ffn"))
    if top_total > 0:
        logger.info("=== Transformer Block Split (top-level) ===")
        for name in ("self_attn", "cross_attn", "ffn"):
            t = _profile_times.get(name, 0.0)
            logger.info(f"  {name}: {t:.1f}ms ({100 * t / top_total:.1f}%)")

    # Compute p_bias vs p_recompute
    # KV-bias can run via multiple backends (FA4 / Flash segment-combine / Triton / flex fallback).
    # Aggregate them for the "p_bias" summary.
    bias_time = sum(
        _profile_times.get(name, 0.0)
        for name in (
            "self_attn_kv_bias_fa4",
            "self_attn_kv_bias_flash",
            "self_attn_kv_bias",
        )
    )
    recompute_time = _profile_times.get("self_attn_block_mask", 0.0)
    plain_time = _profile_times.get("self_attn_kv_plain", 0.0)
    attn_total = bias_time + recompute_time + plain_time
    if attn_total > 0:
        logger.info("=== p_bias vs p_recompute ===")
        logger.info(f"  p_bias (Kernel B):     {100 * bias_time / attn_total:.1f}%")
        logger.info(f"  p_recompute (Kernel A): {100 * recompute_time / attn_total:.1f}%")
        logger.info(f"  p_plain (FA path):     {100 * plain_time / attn_total:.1f}%")

    # Helpful nested breakdown: how much of self_attn is KV-bias vs other work.
    self_attn_total = _profile_times.get("self_attn", 0.0)
    if self_attn_total > 0:
        other_time = self_attn_total - (bias_time + recompute_time + plain_time)
        logger.info("=== self_attn Breakdown (nested) ===")
        logger.info(f"  kv_bias_total:   {bias_time:.1f}ms ({100 * bias_time / self_attn_total:.1f}% of self_attn)")
        logger.info(f"  block_mask:      {recompute_time:.1f}ms ({100 * recompute_time / self_attn_total:.1f}% of self_attn)")
        logger.info(f"  plain_kv:        {plain_time:.1f}ms ({100 * plain_time / self_attn_total:.1f}% of self_attn)")
        logger.info(f"  other_in_self:   {other_time:.1f}ms ({100 * other_time / self_attn_total:.1f}% of self_attn)")


# Register atexit handler to print profiling report on exit
atexit.register(profile_report)


class _ProfileBlock:
    """Context manager for profiling a code block."""
    __slots__ = ('name', 'start_event', 'end_event')

    def __init__(self, name: str):
        self.name = name
        self.start_event = None
        self.end_event = None

    def __enter__(self):
        if _should_profile():
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
        return self

    def __exit__(self, *args):
        if self.start_event is not None:
            self.end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = self.start_event.elapsed_time(self.end_event)
            _profile_times[self.name] += elapsed_ms
            _profile_counts[self.name] += 1


from .model import (
    USE_TRITON_ROTARY,
    USE_TRITON_ROPE_FUSED,
    WAN_CROSSATTENTION_CLASSES,
    MLPProj,
    WanLayerNorm,
    WanRMSNorm,
    get_rope_cos_sin,
    rope_apply,
    rope_params,
    sinusoidal_embedding_1d,
    triton_apply_rotary,
    triton_rope_fused_3way,
)

_FLEX_ATTENTION_EAGER = flex_attention
_FLEX_ATTENTION_COMPILED = None
_FLEX_ATTENTION_USE_EAGER = False
_FLEX_ATTENTION_DISABLE_COMPILE = os.getenv("DISABLE_FLEX_ATTENTION_COMPILE", "0") == "1"


def _maybe_set_triton_ptxas_path() -> None:
    if os.getenv("TRITON_PTXAS_PATH"):
        return

    candidates = [
        "/usr/local/cuda-13.1/bin/ptxas",
        "/usr/local/cuda-13.0/bin/ptxas",
        "/usr/local/cuda-12.9/bin/ptxas",
        "/usr/local/cuda/bin/ptxas",
        "/usr/local/cuda-12.8/bin/ptxas",
    ]

    def supports_sm103(ptxas_path: str) -> bool:
        try:
            proc = subprocess.run(
                [ptxas_path, "--help"],
                capture_output=True,
                text=True,
                check=False,
                timeout=2,
            )
        except Exception:
            return False
        return proc.returncode == 0 and "sm_103" in (proc.stdout or "")

    # Prefer a ptxas that explicitly supports SM103 (B300).
    for candidate in candidates:
        if os.path.exists(candidate) and supports_sm103(candidate):
            os.environ["TRITON_PTXAS_PATH"] = candidate
            logger.info("Set TRITON_PTXAS_PATH=%s (supports SM103)", candidate)
            return

    # Fall back to the first ptxas we can find.
    for candidate in candidates:
        if os.path.exists(candidate):
            os.environ["TRITON_PTXAS_PATH"] = candidate
            logger.info("Set TRITON_PTXAS_PATH=%s", candidate)
            return


def _flex_attention_first_call(*args, **kwargs):
    """
    Lazily compile flex_attention and fall back to eager if compilation fails.

    This makes B300/SM103 bringup less brittle: toolchain mismatches (e.g., old ptxas)
    can surface as Inductor/Triton compilation failures.
    """
    global _FLEX_ATTENTION_COMPILED, _FLEX_ATTENTION_USE_EAGER, flex_attention

    if _FLEX_ATTENTION_USE_EAGER or _FLEX_ATTENTION_DISABLE_COMPILE:
        flex_attention = _FLEX_ATTENTION_EAGER
        return _FLEX_ATTENTION_EAGER(*args, **kwargs)

    _maybe_set_triton_ptxas_path()
    if _FLEX_ATTENTION_COMPILED is None:
        _FLEX_ATTENTION_COMPILED = torch.compile(
            _FLEX_ATTENTION_EAGER, dynamic=False, mode="max-autotune-no-cudagraphs"
        )

    try:
        out = _FLEX_ATTENTION_COMPILED(*args, **kwargs)
    except Exception as e:
        _FLEX_ATTENTION_USE_EAGER = True
        flex_attention = _FLEX_ATTENTION_EAGER
        logger.warning("flex_attention compile failed; falling back to eager: %s", e)
        return _FLEX_ATTENTION_EAGER(*args, **kwargs)

    flex_attention = _FLEX_ATTENTION_COMPILED
    return out


flex_attention = _flex_attention_first_call

# Constants for flex_attention operations
FLEX_ATTENTION_ALIGNMENT = 128
KV_CACHE_ATTENTION_BIAS_DISABLED = 1.0


def _pad_tensor_for_flex_attention(
    tensor: torch.Tensor, target_length: int, pad_dim: int = 1
) -> torch.Tensor:
    """Pad tensor to target_length along pad_dim. Returns original if no padding needed."""
    current_length = tensor.shape[pad_dim]
    padded_length = target_length - current_length

    if padded_length <= 0:
        return tensor

    pad_shape = list(tensor.shape)
    pad_shape[pad_dim] = padded_length

    padding = torch.zeros(pad_shape, device=tensor.device, dtype=tensor.dtype)
    return torch.cat([tensor, padding], dim=pad_dim)


def rope_params_riflex(max_seq_len, dim, theta=10000, k=0, L_test=None):
    assert dim % 2 == 0
    # Use float32 (complex64) instead of float64 (complex128) to reduce cast overhead
    omega = 1.0 / torch.pow(theta, torch.arange(0, dim, 2, dtype=torch.float32).div(dim))
    if k is not None:
        print("Doing riflex w/ ltest", L_test)
        omega[k - 1] = 0.9 * 2 * torch.pi / L_test
    freqs = torch.outer(torch.arange(max_seq_len), omega)

    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@functools.lru_cache(maxsize=32)
def get_sdpa_mask(
    device: str,
    num_frames: int = 21,
    frame_seqlen: int = 1560,
    num_frame_per_block: int = 1,
    local_attn_size: int = -1,
    dtype: torch.dtype = torch.bool,
):
    """
    Create an attention mask tensor for torch.nn.functional.scaled_dot_product_attention

    Args:
        device: Device to create the mask on
        num_frames: Number of frames
        frame_seqlen: Sequence length per frame
        num_frame_per_block: Number of frames per block
        local_attn_size: Local attention window size (-1 for global)
        dtype: Data type for the mask (torch.bool for masking, torch.float for additive)

    Returns:
        torch.Tensor: Attention mask of shape (seq_len, seq_len)
                     - True/1.0 for allowed attention
                     - False/-inf for masked attention
    """
    print("Generating SDPA attention mask")
    total_length = num_frames * frame_seqlen

    # Right padding to get to a multiple of 128
    padded_length = math.ceil(total_length / 128) * 128 - total_length
    full_length = total_length + padded_length

    # Create the ends array (same logic as original)
    ends = torch.zeros(full_length, device=device, dtype=torch.long)

    frame_indices = torch.arange(
        start=0,
        end=total_length,
        step=frame_seqlen * num_frame_per_block,
        device=device,
    )

    for tmp in frame_indices:
        end_idx = min(tmp + frame_seqlen * num_frame_per_block, full_length)
        ends[tmp:end_idx] = end_idx

    # Create q_idx and kv_idx coordinate matrices
    q_indices = torch.arange(full_length, device=device).unsqueeze(
        1
    )  # Shape: (seq_len, 1)
    kv_indices = torch.arange(full_length, device=device).unsqueeze(
        0
    )  # Shape: (1, seq_len)

    # Apply the attention logic
    if local_attn_size == -1:
        # Global attention within blocks + diagonal
        mask = (kv_indices < ends[q_indices]) | (q_indices == kv_indices)
    else:
        # Local attention within blocks + diagonal
        local_window_start = ends[q_indices] - local_attn_size * frame_seqlen
        mask = ((kv_indices < ends[q_indices]) & (kv_indices >= local_window_start)) | (
            q_indices == kv_indices
        )

    if dtype == torch.bool:
        return mask
    elif dtype == torch.float32 or dtype == torch.float16:
        # Convert to additive mask (0.0 for attend, -inf for mask)
        return mask.float() * 0.0 + (~mask).float() * float("-inf")
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


@functools.lru_cache(maxsize=32)
def get_block_mask(
    device: str,
    num_frames: int = 21,
    frame_seqlen: int = 1560,
    num_frame_per_block=3,
    local_attn_size=-1,
):
    total_length = num_frames * frame_seqlen

    # we do right padding to get to a multiple of 128
    padded_length = math.ceil(total_length / 128) * 128 - total_length

    ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

    # Block-wise causal mask will attend to all elements that are before the end of the current chunk
    frame_indices = torch.arange(
        start=0,
        end=total_length,
        step=frame_seqlen * num_frame_per_block,
        device=device,
    )

    for tmp in frame_indices:
        ends[tmp : tmp + frame_seqlen * num_frame_per_block] = (
            tmp + frame_seqlen * num_frame_per_block
        )

    def attention_mask(b, h, q_idx, kv_idx):
        if local_attn_size == -1:
            return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
        else:
            return (
                (kv_idx < ends[q_idx])
                & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))
            ) | (q_idx == kv_idx)

    block_mask = create_block_mask(
        attention_mask,
        B=None,
        H=None,
        Q_LEN=total_length + padded_length,
        KV_LEN=total_length + padded_length,
        _compile=False,
        device=device,
    )
    return block_mask


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    """Optimized RoPE using cached sin/cos (no float64, no complex math).

    Uses Triton fused 3-way kernel when available (Step 2).
    Falls back to Triton rotary (Step 0) or PyTorch.
    """
    # --- Fused Triton 3-way RoPE fast path (D=128 only) ---
    if USE_TRITON_ROPE_FUSED and x.is_cuda and x.ndim == 4 and x.shape[-1] == 128:
        try:
            return triton_rope_fused_3way(x, grid_sizes, freqs, start_frame=int(start_frame), inplace=None)
        except Exception:
            if os.environ.get("SCOPE_TRITON_ROPE_FUSED_STRICT", "0") == "1":
                raise
            pass

    n, c = x.size(2), x.size(3) // 2

    # split freqs (freqs is complex: real=cos, imag=sin)
    freqs_split = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    if x.size(0) == 1:
        if isinstance(grid_sizes, torch.Tensor):
            f, h, w = (int(v) for v in grid_sizes[0].tolist())
        else:
            f, h, w = (int(v) for v in grid_sizes[0])
        seq_len = f * h * w

        cos, sin = get_rope_cos_sin(
            freqs_split, f, h, w, start_frame, x.dtype, x.device, c
        )
        cos_fa = cos.squeeze(1)
        sin_fa = sin.squeeze(1)

        if USE_TRITON_ROTARY:
            if seq_len == x.shape[1]:
                return triton_apply_rotary(x, cos_fa, sin_fa, interleaved=True)
            out = x.clone()
            out[:, :seq_len] = triton_apply_rotary(
                x[:, :seq_len], cos_fa, sin_fa, interleaved=True
            )
            return out

        x_i = x[0, :seq_len].reshape(seq_len, n, -1, 2)
        x0, x1 = x_i[..., 0], x_i[..., 1]
        x0_new = x0 * cos - x1 * sin
        x1_new = x0 * sin + x1 * cos
        x_i = torch.stack([x0_new, x1_new], dim=-1).flatten(2).unsqueeze(0)
        if seq_len == x.shape[1]:
            return x_i
        out = x.clone()
        out[:, :seq_len] = x_i
        return out

    out = x.clone()
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        cos, sin = get_rope_cos_sin(
            freqs_split, f, h, w, start_frame, x.dtype, x.device, c
        )

        if USE_TRITON_ROTARY:
            cos_fa = cos.squeeze(1)
            sin_fa = sin.squeeze(1)
            out[i : i + 1, :seq_len] = triton_apply_rotary(
                x[i : i + 1, :seq_len], cos_fa, sin_fa, interleaved=True
            )
            continue

        x_i = x[i, :seq_len].reshape(seq_len, n, -1, 2)
        x0, x1 = x_i[..., 0], x_i[..., 1]
        x0_new = x0 * cos - x1 * sin
        x1_new = x0 * sin + x1 * cos
        out[i, :seq_len] = torch.stack([x0_new, x1_new], dim=-1).flatten(2)
    return out


class CausalWanSelfAttention(nn.Module):
    def __init__(
        self, dim, num_heads, local_attn_size=-1, sink_size=0, qk_norm=True, eps=1e-6
    ):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.frame_seq_length = 1560  # Default value, can be updated dynamically
        self.max_attention_size = (
            32760 if local_attn_size == -1 else local_attn_size * self.frame_seq_length
        )
        self.fused_projections = False

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    @torch.no_grad()
    def fuse_projections(self):
        # if not self.is_cross_attention:
        if self.fused_projections:
            return
        concatenated_weights = torch.cat(
            [self.q.weight.data, self.k.weight.data, self.v.weight.data]
        )
        concatenated_bias = torch.cat(
            [self.q.bias.data, self.k.bias.data, self.v.bias.data]
        )
        out_features, in_features = concatenated_weights.shape
        with torch.device("meta"):
            self.to_qkv = torch.nn.Linear(in_features, out_features, bias=True)
        self.to_qkv.load_state_dict(
            {"weight": concatenated_weights, "bias": concatenated_bias},
            strict=True,
            assign=True,
        )
        self.fused_projections = True

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        block_mask,
        kv_cache=None,
        current_start=0,
        cache_start=None,
        kv_cache_attention_bias=1.0,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            block_mask (BlockMask)
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if cache_start is None:
            cache_start = current_start

        # query, key, value function
        # @torch.compile(dynamic=True, mode="max-autotune-no-cudagraphs")
        def qkv_fn(x):
            if self.fused_projections:
                # print("Using fused projections")
                q, k, v = self.to_qkv(x).chunk(3, dim=-1)
                q = self.norm_q(q).view(b, s, n, d)
                k = self.norm_k(k).view(b, s, n, d)
                v = v.view(b, s, n, d)
            else:
                q = self.norm_q(self.q(x)).view(b, s, n, d)
                k = self.norm_k(self.k(x)).view(b, s, n, d)
                v = self.v(x).view(b, s, n, d)
            return q, k, v

        if _should_profile():
            with _ProfileBlock("qkv_projection"):
                q, k, v = qkv_fn(x)
        else:
            q, k, v = qkv_fn(x)

        if kv_cache is None or block_mask is not None:
            # if it is teacher forcing training?
            # is_tf = (s == seq_lens[0].item() * 2)
            is_tf = False
            if is_tf:
                print("Teacher forcing training")
                q_chunk = torch.chunk(q, 2, dim=1)
                k_chunk = torch.chunk(k, 2, dim=1)
                roped_query = []
                roped_key = []
                # rope should be same for clean and noisy parts
                for ii in range(2):
                    rq = rope_apply(q_chunk[ii], grid_sizes, freqs).type_as(v)
                    rk = rope_apply(k_chunk[ii], grid_sizes, freqs).type_as(v)
                    roped_query.append(rq)
                    roped_key.append(rk)

                roped_query = torch.cat(roped_query, dim=1)
                roped_key = torch.cat(roped_key, dim=1)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [
                        roped_query,
                        torch.zeros(
                            [q.shape[0], padded_length, q.shape[2], q.shape[3]],
                            device=q.device,
                            dtype=v.dtype,
                        ),
                    ],
                    dim=1,
                )

                padded_roped_key = torch.cat(
                    [
                        roped_key,
                        torch.zeros(
                            [k.shape[0], padded_length, k.shape[2], k.shape[3]],
                            device=k.device,
                            dtype=v.dtype,
                        ),
                    ],
                    dim=1,
                )

                padded_v = torch.cat(
                    [
                        v,
                        torch.zeros(
                            [v.shape[0], padded_length, v.shape[2], v.shape[3]],
                            device=v.device,
                            dtype=v.dtype,
                        ),
                    ],
                    dim=1,
                )

                if _should_profile():
                    with _ProfileBlock("self_attn_block_mask"):
                        attn_out = flex_attention(
                            query=padded_roped_query.transpose(2, 1),
                            key=padded_roped_key.transpose(2, 1),
                            value=padded_v.transpose(2, 1),
                            block_mask=block_mask,
                        )
                else:
                    attn_out = flex_attention(
                        query=padded_roped_query.transpose(2, 1),
                        key=padded_roped_key.transpose(2, 1),
                        value=padded_v.transpose(2, 1),
                        block_mask=block_mask,
                    )
                if padded_length > 0:
                    attn_out = attn_out[:, :, :-padded_length]
                x = attn_out.transpose(2, 1)

            else:
                roped_query = rope_apply(q, grid_sizes, freqs).type_as(v)
                roped_key = rope_apply(k, grid_sizes, freqs).type_as(v)

                local_end_index = roped_key.shape[1]
                kv_cache["k"][:, :local_end_index] = roped_key
                kv_cache["v"][:, :local_end_index] = v

                kv_cache["global_end_index"] = local_end_index
                kv_cache["local_end_index"] = local_end_index

                use_dense_attention_for_block_mask = False
                if block_mask is not None and _FLEX_ATTENTION_DISABLE_COMPILE and _is_sm103():
                    # On SM103 + torch 2.9 / triton 3.5, `torch.compile(flex_attention)` hard-aborts
                    # with an LLVM tcgen05 intrinsic error. Eager flex_attention is extremely slow
                    # for block masks (materializes full scores).
                    #
                    # For the Krea recompute path, the block_mask is often "single block" (i.e.
                    # num_frames <= num_frame_per_block), which is equivalent to dense attention.
                    try:
                        # Avoid Tensor.item() (graph break) by deriving frame count from the
                        # token sequence length. In this path, s == num_frames * frame_seq_length.
                        frame_seqlen = int(getattr(self, "frame_seq_length", 0))
                        num_frames = int(s) // frame_seqlen if frame_seqlen > 0 else 0
                        num_frame_per_block = int(getattr(self, "num_frame_per_block", 1))
                        use_dense_attention_for_block_mask = num_frames <= num_frame_per_block
                    except Exception:
                        use_dense_attention_for_block_mask = False

                if use_dense_attention_for_block_mask:
                    if _should_profile():
                        with _ProfileBlock("self_attn_block_mask"):
                            x = attention(roped_query, roped_key, v)
                    else:
                        x = attention(roped_query, roped_key, v)
                else:
                    padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                    padded_roped_query = torch.cat(
                        [
                            roped_query,
                            torch.zeros(
                                [q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                device=q.device,
                                dtype=v.dtype,
                            ),
                        ],
                        dim=1,
                    )

                    padded_roped_key = torch.cat(
                        [
                            roped_key,
                            torch.zeros(
                                [k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                device=k.device,
                                dtype=v.dtype,
                            ),
                        ],
                        dim=1,
                    )

                    padded_v = torch.cat(
                        [
                            v,
                            torch.zeros(
                                [v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                device=v.device,
                                dtype=v.dtype,
                            ),
                        ],
                        dim=1,
                    )

                    if _should_profile():
                        with _ProfileBlock("self_attn_block_mask"):
                            attn_out = flex_attention(
                                query=padded_roped_query.transpose(2, 1).contiguous(),
                                key=padded_roped_key.transpose(2, 1).contiguous(),
                                value=padded_v.transpose(2, 1).contiguous(),
                                block_mask=block_mask,
                                kernel_options={
                                    "BLOCKS_ARE_CONTIGUOUS": True,
                                },
                            )
                    else:
                        attn_out = flex_attention(
                            query=padded_roped_query.transpose(2, 1).contiguous(),
                            key=padded_roped_key.transpose(2, 1).contiguous(),
                            value=padded_v.transpose(2, 1).contiguous(),
                            block_mask=block_mask,
                            kernel_options={
                                "BLOCKS_ARE_CONTIGUOUS": True,
                            },
                        )
                    if padded_length > 0:
                        attn_out = attn_out[:, :, :-padded_length]
                    x = attn_out.transpose(2, 1)
        else:
            # frame_seqlen = math.prod(grid_sizes[0][1:]).item() # torch compile doesn't like this
            frame_seqlen = self.frame_seq_length
            current_start_frame = current_start // frame_seqlen
            if _should_profile():
                with _ProfileBlock("rope_apply"):
                    roped_query = causal_rope_apply(
                        q, grid_sizes, freqs, start_frame=current_start_frame
                    ).type_as(v)
                    roped_key = causal_rope_apply(
                        k, grid_sizes, freqs, start_frame=current_start_frame
                    ).type_as(v)
            else:
                roped_query = causal_rope_apply(
                    q, grid_sizes, freqs, start_frame=current_start_frame
                ).type_as(v)
                roped_key = causal_rope_apply(
                    k, grid_sizes, freqs, start_frame=current_start_frame
                ).type_as(v)

            current_end = current_start + roped_query.shape[1]
            sink_tokens = self.sink_size * frame_seqlen
            # If we are using local attention and the current KV cache size is larger than the local attention size, we need to truncate the KV cache
            kv_cache_size = kv_cache["k"].shape[1]
            num_new_tokens = roped_query.shape[1]
            if (
                self.local_attn_size != -1
                and (current_end > kv_cache["global_end_index"])
                and (num_new_tokens + kv_cache["local_end_index"] > kv_cache_size)
            ):
                if _should_profile():
                    with _ProfileBlock("cache_eviction"):
                        # Calculate the number of new tokens added in this step
                        # Shift existing cache content left to discard oldest tokens
                        # Clone the source slice to avoid overlapping memory error
                        num_evicted_tokens = (
                            num_new_tokens + kv_cache["local_end_index"] - kv_cache_size
                        )
                        num_rolled_tokens = (
                            kv_cache["local_end_index"] - num_evicted_tokens - sink_tokens
                        )
                        kv_cache["k"][:, sink_tokens : sink_tokens + num_rolled_tokens] = (
                            kv_cache["k"][
                                :,
                                sink_tokens + num_evicted_tokens : sink_tokens
                                + num_evicted_tokens
                                + num_rolled_tokens,
                            ].clone()
                        )
                        kv_cache["v"][:, sink_tokens : sink_tokens + num_rolled_tokens] = (
                            kv_cache["v"][
                                :,
                                sink_tokens + num_evicted_tokens : sink_tokens
                                + num_evicted_tokens
                                + num_rolled_tokens,
                            ].clone()
                        )
                        # Insert the new keys/values at the end
                        local_end_index = (
                            kv_cache["local_end_index"]
                            + current_end
                            - kv_cache["global_end_index"]
                            - num_evicted_tokens
                        )
                        local_start_index = local_end_index - num_new_tokens
                        kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                        kv_cache["v"][:, local_start_index:local_end_index] = v
                else:
                    num_evicted_tokens = (
                        num_new_tokens + kv_cache["local_end_index"] - kv_cache_size
                    )
                    num_rolled_tokens = (
                        kv_cache["local_end_index"] - num_evicted_tokens - sink_tokens
                    )
                    kv_cache["k"][:, sink_tokens : sink_tokens + num_rolled_tokens] = (
                        kv_cache["k"][
                            :,
                            sink_tokens + num_evicted_tokens : sink_tokens
                            + num_evicted_tokens
                            + num_rolled_tokens,
                        ].clone()
                    )
                    kv_cache["v"][:, sink_tokens : sink_tokens + num_rolled_tokens] = (
                        kv_cache["v"][
                            :,
                            sink_tokens + num_evicted_tokens : sink_tokens
                            + num_evicted_tokens
                            + num_rolled_tokens,
                        ].clone()
                    )
                    local_end_index = (
                        kv_cache["local_end_index"]
                        + current_end
                        - kv_cache["global_end_index"]
                        - num_evicted_tokens
                    )
                    local_start_index = local_end_index - num_new_tokens
                    kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                    kv_cache["v"][:, local_start_index:local_end_index] = v
            else:
                if _should_profile():
                    with _ProfileBlock("cache_update"):
                        # Assign new keys/values directly up to current_end
                        local_end_index = (
                            kv_cache["local_end_index"]
                            + current_end
                            - kv_cache["global_end_index"]
                        )
                        local_start_index = local_end_index - num_new_tokens
                        kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                        kv_cache["v"][:, local_start_index:local_end_index] = v
                else:
                    local_end_index = (
                        kv_cache["local_end_index"] + current_end - kv_cache["global_end_index"]
                    )
                    local_start_index = local_end_index - num_new_tokens
                    kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                    kv_cache["v"][:, local_start_index:local_end_index] = v

            kv_start_idx = max(0, local_end_index - self.max_attention_size)
            cached_k = kv_cache["k"][:, kv_start_idx:local_end_index]
            cached_v = kv_cache["v"][:, kv_start_idx:local_end_index]

            if kv_cache_attention_bias != KV_CACHE_ATTENTION_BIAS_DISABLED:
                # Use flex_attention with bias to mitigate error accumulation in past frames
                # log_scale in (0, 1]: smaller values = less attention to past frame tokens
                log_scale = math.log(kv_cache_attention_bias)

                # Exclude first frame and current block from bias
                cache_len = local_end_index - kv_start_idx
                cache_current_block_start = (
                    cache_len - frame_seqlen * self.num_frame_per_block
                )

                q_len = roped_query.shape[1]
                block_size = frame_seqlen * self.num_frame_per_block

                x = None

                global _fa4_bias_tripped
                if _fa4_available and (not _fa4_bias_tripped) and _KV_BIAS_BACKEND == "fa4":
                    # FA4 + CUTE score_mod: 1.89x faster than Triton Kernel B
                    # FA4 expects layout: [B, L, H, D] (already correct)
                    try:
                        score_mod_cute = _get_fa4_score_mod(
                            frame_seqlen,
                            block_start=cache_current_block_start,
                            log_bias=log_scale,
                        )
                        if _should_profile():
                            with _ProfileBlock("self_attn_kv_bias_fa4"):
                                # FA4/CuTe can fail to infer "leading dim" from a B=1 view
                                # when K/V come from a larger cache tensor (stride(0) reflects
                                # the full cache, not the sliced window). Normalizing the
                                # batch dimension stride avoids a costly clone/copy.
                                q_fa4 = roped_query
                                k_fa4 = cached_k
                                v_fa4 = cached_v
                                if k_fa4.shape[0] == 1:
                                    q_fa4 = q_fa4[0].unsqueeze(0)
                                    k_fa4 = k_fa4[0].unsqueeze(0)
                                    v_fa4 = v_fa4[0].unsqueeze(0)
                                x, _ = (_fa4_fwd_opaque or _fa4_fwd)(
                                    q_fa4,
                                    k_fa4,
                                    v_fa4,
                                    score_mod=score_mod_cute,
                                    causal=False,
                                    return_lse=False,
                                )
                        else:
                            q_fa4 = roped_query
                            k_fa4 = cached_k
                            v_fa4 = cached_v
                            if k_fa4.shape[0] == 1:
                                q_fa4 = q_fa4[0].unsqueeze(0)
                                k_fa4 = k_fa4[0].unsqueeze(0)
                                v_fa4 = v_fa4[0].unsqueeze(0)
                            x, _ = (_fa4_fwd_opaque or _fa4_fwd)(
                                q_fa4,
                                k_fa4,
                                v_fa4,
                                score_mod=score_mod_cute,
                                causal=False,
                                return_lse=False,
                            )
                    except Exception as e:
                        _fa4_bias_tripped = True
                        logger.warning(
                            "FA4/CUTE KV-bias failed; falling back to Triton: %s",
                            e,
                        )

                elif _KV_BIAS_BACKEND == "flash":
                    global _flash_bias_tripped
                    if not _flash_bias_tripped:
                        try:
                            if _should_profile():
                                with _ProfileBlock("self_attn_kv_bias_flash"):
                                    x = _kv_bias_flash_combine(
                                        roped_query,
                                        cached_k,
                                        cached_v,
                                        frame_seqlen=frame_seqlen,
                                        current_block_start=cache_current_block_start,
                                        log_bias=log_scale,
                                    )
                            else:
                                x = _kv_bias_flash_combine(
                                    roped_query,
                                    cached_k,
                                    cached_v,
                                    frame_seqlen=frame_seqlen,
                                    current_block_start=cache_current_block_start,
                                    log_bias=log_scale,
                                )
                        except Exception as e:
                            _flash_bias_tripped = True
                            logger.warning(
                                "KV-bias flash backend failed; falling back to Triton: %s",
                                e,
                            )

                if x is None and USE_TRITON_KERNEL_B and _triton_kernel_b is not None:
                    # Triton Kernel B: no padding needed
                    # Input: (B, L, H, D) -> (B, H, L, D)
                    if _should_profile():
                        with _ProfileBlock("transpose_contiguous"):
                            Q_t = roped_query.transpose(2, 1).contiguous()
                            K_t = cached_k.transpose(2, 1).contiguous()
                            V_t = cached_v.transpose(2, 1).contiguous()
                        with _ProfileBlock("self_attn_kv_bias"):
                            x = _triton_kernel_b(
                                Q=Q_t,
                                K=K_t,
                                V=V_t,
                                frame_seqlen=frame_seqlen,
                                current_block_start=cache_current_block_start,
                                log_bias=log_scale,
                            ).transpose(2, 1)  # (B, H, L, D) -> (B, L, H, D)
                    else:
                        Q_t = roped_query.transpose(2, 1).contiguous()
                        K_t = cached_k.transpose(2, 1).contiguous()
                        V_t = cached_v.transpose(2, 1).contiguous()
                        x = _triton_kernel_b(
                            Q=Q_t,
                            K=K_t,
                            V=V_t,
                            frame_seqlen=frame_seqlen,
                            current_block_start=cache_current_block_start,
                            log_bias=log_scale,
                        ).transpose(2, 1)  # (B, H, L, D) -> (B, L, H, D)

                if x is None:
                    # Fallback: flex_attention with padding
                    kv_len = cached_k.shape[1]
                    target_padded_length = (
                        math.ceil(max(q_len, kv_len) / FLEX_ATTENTION_ALIGNMENT)
                        * FLEX_ATTENTION_ALIGNMENT
                    )

                    padded_roped_query = _pad_tensor_for_flex_attention(
                        roped_query, target_padded_length, pad_dim=1
                    )
                    padded_k = _pad_tensor_for_flex_attention(
                        cached_k, target_padded_length, pad_dim=1
                    )
                    padded_v = _pad_tensor_for_flex_attention(
                        cached_v, target_padded_length, pad_dim=1
                    )

                    # Convert scalars to tensors to avoid ShapeAsConstantBuffer dtype issues during compilation
                    # This is critical when using torch.compile with flex_attention
                    frame_seqlen_tensor = torch.as_tensor(
                        frame_seqlen, dtype=torch.int32, device=roped_query.device
                    )
                    cache_current_block_start_tensor = torch.as_tensor(
                        cache_current_block_start, dtype=torch.int32, device=roped_query.device
                    ).squeeze()
                    log_scale_tensor = torch.as_tensor(
                        log_scale, dtype=roped_query.dtype, device=roped_query.device
                    )

                    def score_mod(score, b_idx, h_idx, q_idx, kv_idx):
                        # Apply bias only to past frames (exclude first frame and current block)
                        return torch.where(
                            (kv_idx >= frame_seqlen_tensor)
                            & (kv_idx < cache_current_block_start_tensor),
                            score + log_scale_tensor,
                            score,
                        )

                    if _should_profile():
                        with _ProfileBlock("self_attn_kv_bias"):
                            x = flex_attention(
                                query=padded_roped_query.transpose(2, 1).contiguous(),
                                key=padded_k.transpose(2, 1).contiguous(),
                                value=padded_v.transpose(2, 1).contiguous(),
                                score_mod=score_mod,
                            )[:, :, :q_len].transpose(2, 1)
                    else:
                        x = flex_attention(
                            query=padded_roped_query.transpose(2, 1).contiguous(),
                            key=padded_k.transpose(2, 1).contiguous(),
                            value=padded_v.transpose(2, 1).contiguous(),
                            score_mod=score_mod,
                        )[:, :, :q_len].transpose(2, 1)
            else:
                # Use original Flash/Sage Attention path when bias is disabled (1.0)
                # This preserves the original behavior and avoids flex_attention overhead
                if _should_profile():
                    with _ProfileBlock("self_attn_kv_plain"):
                        x = attention(roped_query, cached_k, cached_v)
                else:
                    x = attention(roped_query, cached_k, cached_v)

            kv_cache["global_end_index"] = current_end
            kv_cache["local_end_index"] = local_end_index

        # output
        if _should_profile():
            with _ProfileBlock("output_projection"):
                x = x.flatten(2)
                x = self.o(x)
        else:
            x = x.flatten(2)
            x = self.o(x)
        return x


class CausalWanAttentionBlock(nn.Module):
    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        local_attn_size=-1,
        sink_size=0,
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(
            dim, num_heads, local_attn_size, sink_size, qk_norm, eps
        )
        self.norm3 = (
            WanLayerNorm(dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](
            dim, num_heads, (-1, -1), qk_norm, eps
        )
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        block_mask,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=None,
        kv_cache_attention_bias=1.0,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        # assert e[0].dtype == torch.float32

        # self-attention
        _do_profile = _should_profile()
        if _do_profile:
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        y = self.self_attn(
            (
                self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen))
                * (1 + e[1])
                + e[0]
            ).flatten(1, 2),
            seq_lens,
            grid_sizes,
            freqs,
            block_mask,
            kv_cache,
            current_start,
            cache_start,
            kv_cache_attention_bias,
        )

        if _do_profile:
            end.record()
            torch.cuda.synchronize()
            _profile_times["self_attn"] += start.elapsed_time(end)
            _profile_counts["self_attn"] += 1

        # with amp.autocast(dtype=torch.float32):
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(
            1, 2
        )

        # cross-attention
        if _do_profile:
            start2 = torch.cuda.Event(enable_timing=True)
            end2 = torch.cuda.Event(enable_timing=True)
            start2.record()

        x = x + self.cross_attn(
            self.norm3(x), context, context_lens, crossattn_cache=crossattn_cache
        )

        if _do_profile:
            end2.record()
            torch.cuda.synchronize()
            _profile_times["cross_attn"] += start2.elapsed_time(end2)
            _profile_counts["cross_attn"] += 1

        # ffn
        if _do_profile:
            start3 = torch.cuda.Event(enable_timing=True)
            end3 = torch.cuda.Event(enable_timing=True)
            start3.record()

        y = self.ffn(
            (
                self.norm2(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen))
                * (1 + e[4])
                + e[3]
            ).flatten(1, 2)
        )
        # with amp.autocast(dtype=torch.float32):
        x = x + (
            y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[5]
        ).flatten(1, 2)

        if _do_profile:
            end3.record()
            torch.cuda.synchronize()
            _profile_times["ffn"] += start3.elapsed_time(end3)
            _profile_counts["ffn"] += 1

            global _profile_call_count
            _profile_call_count += 1
            if _profile_call_count % 1000 == 0:
                profile_report()
        return x


class CausalHead(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, F, 1, C]
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = self.head(
            self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1])
            + e[0]
        )
        return x


class CausalWanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = ["patch_size", "cross_attn_norm", "qk_norm", "text_dim"]
    _no_split_modules = ["WanAttentionBlock"]
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        model_type="t2v",
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        local_attn_size=-1,
        sink_size=0,
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
    ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            local_attn_size (`int`, *optional*, defaults to -1):
                Window size for temporal local attention (-1 indicates global attention)
            sink_size (`int`, *optional*, defaults to 0):
                Size of the attention sink, we keep the first `sink_size` frames unchanged when rolling the KV cache
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ["t2v", "i2v"]
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim)
        )

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = "t2v_cross_attn" if model_type == "t2v" else "i2v_cross_attn"
        self.blocks = nn.ModuleList(
            [
                CausalWanAttentionBlock(
                    cross_attn_type,
                    dim,
                    ffn_dim,
                    num_heads,
                    local_attn_size,
                    sink_size,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                )
                for _ in range(num_layers)
            ]
        )

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6)),
                # rope_params_riflex(1024, d - 4 * (d // 6), ),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
            ],
            dim=1,
        )

        if model_type == "i2v":
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = False

        self.block_mask = None

        self.num_frame_per_block = 1
        self.independent_first_frame = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device,
        num_frames: int = 21,
        frame_seqlen: int = 1560,
        num_frame_per_block=1,
        local_attn_size=-1,
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        block_mask = get_block_mask(
            str(device), num_frames, frame_seqlen, num_frame_per_block, local_attn_size
        )
        return block_mask

    @staticmethod
    def _prepare_teacher_forcing_mask(
        device: torch.device | str,
        num_frames: int = 21,
        frame_seqlen: int = 1560,
        num_frame_per_block=1,
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        # debug
        DEBUG = False
        if DEBUG:
            num_frames = 9
            frame_seqlen = 256

        total_length = num_frames * frame_seqlen * 2

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        clean_ends = num_frames * frame_seqlen
        # for clean context frames, we can construct their flex attention mask based on a [start, end] interval
        context_ends = torch.zeros(
            total_length + padded_length, device=device, dtype=torch.long
        )
        # for noisy frames, we need two intervals to construct the flex attention mask [context_start, context_end] [noisy_start, noisy_end]
        noise_context_starts = torch.zeros(
            total_length + padded_length, device=device, dtype=torch.long
        )
        noise_context_ends = torch.zeros(
            total_length + padded_length, device=device, dtype=torch.long
        )
        noise_noise_starts = torch.zeros(
            total_length + padded_length, device=device, dtype=torch.long
        )
        noise_noise_ends = torch.zeros(
            total_length + padded_length, device=device, dtype=torch.long
        )

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        attention_block_size = frame_seqlen * num_frame_per_block
        frame_indices = torch.arange(
            start=0,
            end=num_frames * frame_seqlen,
            step=attention_block_size,
            device=device,
            dtype=torch.long,
        )

        # attention for clean context frames
        for start in frame_indices:
            context_ends[start : start + attention_block_size] = (
                start + attention_block_size
            )

        noisy_image_start_list = torch.arange(
            num_frames * frame_seqlen,
            total_length,
            step=attention_block_size,
            device=device,
            dtype=torch.long,
        )
        noisy_image_end_list = noisy_image_start_list + attention_block_size

        # attention for noisy frames
        for block_index, (start, end) in enumerate(
            zip(noisy_image_start_list, noisy_image_end_list, strict=False)
        ):
            # attend to noisy tokens within the same block
            noise_noise_starts[start:end] = start
            noise_noise_ends[start:end] = end
            # attend to context tokens in previous blocks
            # noise_context_starts[start:end] = 0
            noise_context_ends[start:end] = block_index * attention_block_size

        def attention_mask(b, h, q_idx, kv_idx):
            # first design the mask for clean frames
            clean_mask = (q_idx < clean_ends) & (kv_idx < context_ends[q_idx])
            # then design the mask for noisy frames
            # noisy frames will attend to all clean preceeding clean frames + itself
            C1 = (kv_idx < noise_noise_ends[q_idx]) & (
                kv_idx >= noise_noise_starts[q_idx]
            )
            C2 = (kv_idx < noise_context_ends[q_idx]) & (
                kv_idx >= noise_context_starts[q_idx]
            )
            noise_mask = (q_idx >= clean_ends) & (C1 | C2)

            eye_mask = q_idx == kv_idx
            return eye_mask | clean_mask | noise_mask

        block_mask = create_block_mask(
            attention_mask,
            B=None,
            H=None,
            Q_LEN=total_length + padded_length,
            KV_LEN=total_length + padded_length,
            _compile=False,
            device=device,
        )

        if DEBUG:
            print(block_mask)
            try:
                imageio = __import__("imageio")
                np = __import__("numpy")
                cv2 = __import__("cv2")
                flex_mod = __import__(
                    "torch.nn.attention.flex_attention",
                    fromlist=["create_mask"],
                )
                create_mask = getattr(flex_mod, "create_mask")
            except Exception as e:
                logger.warning("DEBUG mask dump skipped (missing deps): %s", e)
            else:
                mask = create_mask(
                    attention_mask,
                    B=None,
                    H=None,
                    Q_LEN=total_length + padded_length,
                    KV_LEN=total_length + padded_length,
                    device=device,
                )
                mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
                imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255.0 * mask))

        return block_mask

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_i2v(
        device: torch.device | str,
        num_frames: int = 21,
        frame_seqlen: int = 1560,
        num_frame_per_block=4,
        local_attn_size=-1,
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [N latent frame] ... [N latent frame]
        The first frame is separated out to support I2V generation
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(
            total_length + padded_length, device=device, dtype=torch.long
        )

        # special handling for the first frame
        ends[:frame_seqlen] = frame_seqlen

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=frame_seqlen,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device,
        )

        for idx, tmp in enumerate(frame_indices):
            ends[tmp : tmp + frame_seqlen * num_frame_per_block] = (
                tmp + frame_seqlen * num_frame_per_block
            )

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return (
                    (kv_idx < ends[q_idx])
                    & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))
                ) | (q_idx == kv_idx)

        block_mask = create_block_mask(
            attention_mask,
            B=None,
            H=None,
            Q_LEN=total_length + padded_length,
            KV_LEN=total_length + padded_length,
            _compile=False,
            device=device,
        )

        # if not dist.is_initialized() or dist.get_rank() == 0:
        # print(
        #     f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
        # print(block_mask)

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    def _forward_inference(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        kv_cache: dict = None,
        crossattn_cache: dict = None,
        current_start: int = 0,
        cache_start: int = 0,
        kv_cache_attention_bias: float = 1.0,
    ):
        r"""
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one by one (1560 tokens each)

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """

        if self.model_type == "i2v":
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y, strict=False)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x]
        )
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)
        """
        torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])
        """

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x)
        )
        e0 = (
            self.time_projection(e)
            .unflatten(1, (6, self.dim))
            .unflatten(dim=0, sizes=t.shape)
        )
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack(
                [
                    torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                    for u in context
                ]
            )
        )

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask,
        )
        # print("Block mask in forward : ", self.block_mask)

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)

            return custom_forward

        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                        "kv_cache_attention_bias": kv_cache_attention_bias,
                    }
                )
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x,
                    **kwargs,
                    use_reentrant=False,
                )
            else:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "crossattn_cache": crossattn_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                        "kv_cache_attention_bias": kv_cache_attention_bias,
                    }
                )
                x = block(x, **kwargs)

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def _forward_train(
        self,
        x,
        t,
        context,
        seq_len,
        clean_x=None,
        aug_t=None,
        clip_fea=None,
        y=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == "i2v":
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # Construct blockwise causal attn mask
        if self.block_mask is None:
            if clean_x is not None:
                if self.independent_first_frame:
                    raise NotImplementedError()
                else:
                    self.block_mask = self._prepare_teacher_forcing_mask(
                        device,
                        num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2]
                        * x.shape[-1]
                        // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                    )
            else:
                if self.independent_first_frame:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask_i2v(
                        device,
                        num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2]
                        * x.shape[-1]
                        // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size,
                    )
                else:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask(
                        device,
                        num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2]
                        * x.shape[-1]
                        // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size,
                    )

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y, strict=False)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x]
        )
        x = [u.flatten(2).transpose(1, 2) for u in x]

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(
            [
                torch.cat(
                    [u, u.new_zeros(1, seq_lens[0] - u.size(1), u.size(2))], dim=1
                )
                for u in x
            ]
        )

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x)
        )
        e0 = (
            self.time_projection(e)
            .unflatten(1, (6, self.dim))
            .unflatten(dim=0, sizes=t.shape)
        )
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack(
                [
                    torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                    for u in context
                ]
            )
        )

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        if clean_x is not None:
            clean_x = [self.patch_embedding(u.unsqueeze(0)) for u in clean_x]
            clean_x = [u.flatten(2).transpose(1, 2) for u in clean_x]

            seq_lens_clean = torch.tensor(
                [u.size(1) for u in clean_x], dtype=torch.long
            )
            assert seq_lens_clean.max() <= seq_len
            clean_x = torch.cat(
                [
                    torch.cat(
                        [u, u.new_zeros(1, seq_lens_clean[0] - u.size(1), u.size(2))],
                        dim=1,
                    )
                    for u in clean_x
                ]
            )

            x = torch.cat([clean_x, x], dim=1)
            if aug_t is None:
                aug_t = torch.zeros_like(t)
            e_clean = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, aug_t.flatten()).type_as(x)
            )
            e0_clean = (
                self.time_projection(e_clean)
                .unflatten(1, (6, self.dim))
                .unflatten(dim=0, sizes=t.shape)
            )
            e0 = torch.cat([e0_clean, e0], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask,
        )

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)

            return custom_forward

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x,
                    **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

        if clean_x is not None:
            x = x[:, x.shape[1] // 2 :]

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def forward(self, *args, **kwargs):
        result = self._forward_inference(*args, **kwargs)
        # if kwargs.get('kv_cache', None) is not None:
        # else:
        #     result = self._forward_train(*args, **kwargs)

        return result

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist(), strict=False):
            u = u[: math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum("fhwpqrc->cfphqwr", u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size, strict=False)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
