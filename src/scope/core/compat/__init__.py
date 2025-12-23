"""
Compatibility utilities for cross-GPU support.

This module provides compatibility layers for:
- SM103 (B300) vs SM100 (B200) GPU support
- Backend selection for attention kernels
"""

from scope.core.compat.sm103 import (
    get_compute_capability,
    is_sm100,
    is_sm103,
    is_blackwell,
    is_hopper,
    patch_cutlass_for_sm103,
    get_recommended_backend,
    check_fa4_availability,
    check_flex_attention_availability,
    get_capability_info,
)

__all__ = [
    "get_compute_capability",
    "is_sm100",
    "is_sm103",
    "is_blackwell",
    "is_hopper",
    "patch_cutlass_for_sm103",
    "get_recommended_backend",
    "check_fa4_availability",
    "check_flex_attention_availability",
    "get_capability_info",
]
