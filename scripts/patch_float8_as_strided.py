"""
Monkeypatch: Add aten.as_strided.default support to torchao.quantization.Float8Tensor

This patches the missing op that blocks torch.compile + FP8 quantization.
Only safe for per-tensor scale (scale.numel() == 1).

Usage:
    import scripts.patch_float8_as_strided  # Apply patch at import time

    # Then proceed with normal quantization + compile flow
    from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig
    quantize_(model, Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()))
    compiled = torch.compile(model)

Based on precedent from:
- torchao/float8/float8_ops.py (training float8 with _assert_tensorwise_scale)
- torchao/dtypes/nf4tensor.py (NF4 with stride constraints)
"""

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

aten = torch.ops.aten


def _patch_float8tensor_as_strided():
    """Register aten.as_strided.default for torchao.quantization.Float8Tensor."""
    try:
        from torchao.quantization.quantize_.workflows.float8.float8_tensor import (
            Float8Tensor,
        )
    except ImportError as e:
        print(f"[patch_float8_as_strided] Could not import Float8Tensor: {e}")
        return False

    # Check if already patched
    if hasattr(Float8Tensor, "_as_strided_patched"):
        print("[patch_float8_as_strided] Already patched, skipping")
        return True

    # Get the implements decorator from Float8Tensor (inherited from TorchAOBaseTensor)
    implements = Float8Tensor.implements

    @implements(aten.as_strided.default)
    def float8_as_strided(func, types, args, kwargs):
        """
        as_strided for Float8Tensor.

        Only safe for per-tensor scale where scale.numel() == 1.
        For per-row/per-block scaling, the mapping from elements to scales
        becomes ambiguous under arbitrary stride transforms.
        """
        self = args[0]
        size = args[1]
        stride = args[2]
        storage_offset = args[3] if len(args) > 3 else 0

        # Safety check: only allow per-tensor scale
        if self.scale.numel() != 1:
            raise NotImplementedError(
                f"as_strided on Float8Tensor is only supported for per-tensor scale "
                f"(scale.numel() == 1), but got scale with {self.scale.numel()} elements. "
                f"Per-row/per-block scaling makes as_strided semantics ambiguous."
            )

        # Apply as_strided to the underlying qdata
        new_qdata = aten.as_strided.default(self.qdata, size, stride, storage_offset)

        # Reshape scale to match new rank if needed (for per-tensor, scale is scalar-like)
        new_rank = len(size)
        if self.scale.ndim != new_rank:
            new_scale = self.scale.reshape([1] * new_rank)
        else:
            new_scale = self.scale

        # For per-tensor scale, block_size spans the whole tensor
        new_block_size = list(size)

        return return_and_correct_aliasing(
            func,
            args,
            kwargs,
            Float8Tensor(
                new_qdata,
                new_scale,
                new_block_size,
                self.mm_config,
                self.act_quant_kwargs,
                self.kernel_preference,
                self.dtype,
            ),
        )

    # Mark as patched
    Float8Tensor._as_strided_patched = True
    print("[patch_float8_as_strided] Successfully patched Float8Tensor with aten.as_strided.default")
    return True


# Apply patch on import
_patch_float8tensor_as_strided()
