"""Flash Attention CUTE (CUDA Template Engine) implementation."""

__version__ = "0.1.0"

import cutlass.cute as cute

# Compatibility shims for older nvidia-cutlass-dsl builds (e.g. 4.1.0).
# Newer flash-attention CUTE code expects these helpers.
if not hasattr(cute, "make_rmem_tensor") and hasattr(cute, "make_fragment"):
    cute.make_rmem_tensor = cute.make_fragment
if not hasattr(cute, "make_rmem_tensor_like") and hasattr(cute, "make_fragment_like"):
    cute.make_rmem_tensor_like = cute.make_fragment_like

from .interface import (
    flash_attn_func,
    flash_attn_varlen_func,
)

from flash_attn.cute.cute_dsl_utils import cute_compile_patched

# Patch cute.compile to optionally dump SASS
cute.compile = cute_compile_patched


__all__ = [
    "flash_attn_func",
    "flash_attn_varlen_func",
]
