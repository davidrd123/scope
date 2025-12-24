# Modified from https://github.com/guandeh17/Self-Forcing
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import logging
import os
import platform
from pathlib import Path

import torch
try:
    import flash_attn as _flash_attn
except ModuleNotFoundError:  # Optional dependency (e.g. B300 cu130 experiments)
    _flash_attn = None

logger = logging.getLogger(__name__)


def _extend_flash_attn_path() -> None:
    if _flash_attn is None:
        return
    try:
        base_path = Path(__file__).resolve()
    except Exception:
        return
    for parent in base_path.parents:
        candidate = parent / "flash-attention" / "flash_attn"
        if candidate.is_dir():
            candidate_str = str(candidate)
            if candidate_str not in _flash_attn.__path__:
                _flash_attn.__path__.insert(0, candidate_str)
            break


flash_attn_func = None
flash_attn = None
FLASH_ATTN_2_AVAILABLE = False
if _flash_attn is not None:
    _extend_flash_attn_path()
    flash_attn_func = _flash_attn.flash_attn_func
    flash_attn = _flash_attn
    FLASH_ATTN_2_AVAILABLE = True


def is_hopper_gpu():
    if not torch.cuda.is_available():
        return False
    device_name = torch.cuda.get_device_name(0).lower()
    return "h100" in device_name or "hopper" in device_name

def is_b200_gpu():
    if not torch.cuda.is_available():
        return False
    device_name = torch.cuda.get_device_name(0).lower()
    return "b200" in device_name


def is_blackwell_gpu():
    if not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability(0)
    return major >= 10


flash_attn_4_varlen_func = None
FLASH_ATTN_4_AVAILABLE = False
if os.getenv("DISABLE_FLASH_ATTENTION_4", "0") == "0":
    if _flash_attn is not None:
        try:
            from flash_attn.cute import flash_attn_varlen_func as flash_attn_4_varlen_func

            FLASH_ATTN_4_AVAILABLE = is_blackwell_gpu()
        except Exception:
            pass

FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn_interface

    FLASH_ATTN_3_AVAILABLE = is_hopper_gpu()
except ModuleNotFoundError:
    pass

if not FLASH_ATTN_3_AVAILABLE and platform.system() != "Windows":
    try:
        from kernels import get_kernel

        flash_attn_3_hub = get_kernel(
            "kernels-community/flash-attn3", revision="fake-ops-return-probs"
        )
        flash_attn_interface = flash_attn_3_hub
        FLASH_ATTN_3_AVAILABLE = is_hopper_gpu()
    except Exception:
        pass

sageattn_func = None
SAGEATTN_AVAILABLE = False
# Do not try to load SageAttention on Hopper GPUs because at the moment
# loading SageAttention 2.2.0 in the sage module causes static on a H100
# Do not try to load SageAttention on B200 GPUs because at the moment
# SageAttention 2.2.0 is not supported on B200 GPUs
if not is_hopper_gpu() and not is_b200_gpu():
    from .sage import SAGEATTN_AVAILABLE, sageattn_func

import warnings

__all__ = [
    "flash_attention",
    "attention",
    "sageattn_func",
    "SAGEATTN_AVAILABLE",
]

logger.info(f"Attention backends: FA2={FLASH_ATTN_2_AVAILABLE}, FA3={FLASH_ATTN_3_AVAILABLE}, FA4={FLASH_ATTN_4_AVAILABLE}, Sage={SAGEATTN_AVAILABLE}")

# Track which attention backend was used (for one-time runtime logging)
_attention_backend_logged = False


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    if not FLASH_ATTN_3_AVAILABLE and not FLASH_ATTN_4_AVAILABLE:
        assert flash_attn_func is not None
        return flash_attn_func(q, k, v)
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == "cuda" and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor([lq] * b, dtype=torch.int32).to(
            device=q.device, non_blocking=True
        )
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens, strict=False)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor([lk] * b, dtype=torch.int32).to(
            device=k.device, non_blocking=True
        )
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens, strict=False)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens, strict=False)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None:
        if version == 4 and not FLASH_ATTN_4_AVAILABLE:
            warnings.warn(
                "Flash attention 4 (cute) is not available, use flash attention 3/2 instead."
            )
        if version == 3 and not FLASH_ATTN_3_AVAILABLE:
            warnings.warn(
                "Flash attention 3 is not available, use flash attention 2 instead."
            )

    # apply attention
    global _attention_backend_logged
    if (version is None or version == 4) and FLASH_ATTN_4_AVAILABLE:
        if not _attention_backend_logged:
            logger.info("Using Flash Attention 4 (CUTE) for Blackwell/SM100+")
            _attention_backend_logged = True
        window_size_fa4 = (None, None) if window_size == (-1, -1) else window_size
        # FA4 (CUTE) doesn't support deterministic parameter
        x = flash_attn_4_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens])
            .cumsum(0, dtype=torch.int32)
            .to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens])
            .cumsum(0, dtype=torch.int32)
            .to(q.device, non_blocking=True),
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size_fa4,
        )
        if isinstance(x, tuple):
            x = x[0]
        x = x.unflatten(0, (b, lq))
    elif (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        if not _attention_backend_logged:
            logger.info("Using Flash Attention 3 for Hopper/SM90")
            _attention_backend_logged = True
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens])
            .cumsum(0, dtype=torch.int32)
            .to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens])
            .cumsum(0, dtype=torch.int32)
            .to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic,
        ).unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        if not _attention_backend_logged:
            logger.info("Using Flash Attention 2")
            _attention_backend_logged = True
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens])
            .cumsum(0, dtype=torch.int32)
            .to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens])
            .cumsum(0, dtype=torch.int32)
            .to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
        ).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
    # og_dtype=torch.bfloat16,
):
    global _attention_backend_logged
    if SAGEATTN_AVAILABLE:
        if not _attention_backend_logged:
            logger.info("Using SageAttention")
            _attention_backend_logged = True
        attn_mask = None

        og_dtype = q.dtype
        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = sageattn_func(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p
        )

        out = out.transpose(1, 2).contiguous().to(og_dtype)
        return out

    elif FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE or FLASH_ATTN_4_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if not _attention_backend_logged:
            logger.info("Using PyTorch SDPA (no optimized attention available)")
            _attention_backend_logged = True
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                "Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance."
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p
        )

        out = out.transpose(1, 2).contiguous()
        return out
