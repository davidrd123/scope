"""
TorchAO compatibility patches.

As of 2025-12-26, `torchao.quantization.Float8Tensor` (created by `quantize_`)
does not implement `aten.as_strided.default` in torchao v0.14.1 / v0.15.0.

This can break `torch.compile` in graphs that hit AOTAutograd aliasing logic
(which can emit `.as_strided(...)` even if model code never calls it directly).

This module provides a PerTensor-only monkeypatch that registers
`aten.as_strided.default` for the quantization Float8Tensor wrapper.
"""

from __future__ import annotations

import functools
import logging

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

logger = logging.getLogger(__name__)

aten = torch.ops.aten


@functools.lru_cache(maxsize=1)
def patch_torchao_quantization_float8_as_strided() -> bool:
    """
    Patch TorchAO quantization Float8Tensor to support `aten.as_strided.default`.

    Safety: only supports per-tensor scale (`scale.numel() == 1`). For per-row /
    per-block scaling, `as_strided` semantics can be ambiguous.
    """
    try:
        from torchao.quantization.quantize_.workflows.float8.float8_tensor import (
            Float8Tensor,
        )
    except Exception as e:
        logger.debug("TorchAO not available (cannot patch Float8Tensor): %s", e)
        return False

    # If upstream already implements it, do nothing.
    try:
        table = getattr(Float8Tensor, "_ATEN_OP_TABLE", {}).get(Float8Tensor, {})
        if aten.as_strided.default in table:
            return True
    except Exception:
        # If TorchAO internals changed, fall through and try registering anyway.
        pass

    implements = Float8Tensor.implements

    @implements(aten.as_strided.default)
    def _float8_as_strided(func, types, args, kwargs):
        self = args[0]
        size = args[1]
        stride = args[2]
        storage_offset = args[3] if len(args) > 3 else kwargs.get("storage_offset", 0)

        if self.scale.numel() != 1:
            raise NotImplementedError(
                "as_strided on torchao.quantization.Float8Tensor is only supported for "
                "per-tensor scale (scale.numel() == 1)."
            )

        new_qdata = aten.as_strided.default(self.qdata, size, stride, storage_offset)

        new_rank = len(size)
        new_scale = (
            self.scale.reshape([1] * new_rank) if self.scale.ndim != new_rank else self.scale
        )
        new_block_size = list(size)

        new = Float8Tensor(
            new_qdata,
            new_scale,
            new_block_size,
            self.mm_config,
            self.act_quant_kwargs,
            self.kernel_preference,
            self.dtype,
        )
        return return_and_correct_aliasing(func, args, kwargs, new)

    logger.warning(
        "Applied TorchAO quantization Float8Tensor `aten.as_strided.default` monkeypatch "
        "(PerTensor-only). Remove this once upstream TorchAO adds as_strided support."
    )
    return True

