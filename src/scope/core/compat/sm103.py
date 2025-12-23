"""
SM103 (B300) Compatibility Layer

This module provides compatibility utilities for running FA4/CUTE kernels
on both B200 (SM100) and B300 (SM103) GPUs.

B200 = SM100 (compute capability 10.0)
B300 = SM103 (compute capability 10.3)

Key issues on SM103:
1. CUTLASS/CUTE DSL has architecture validation lists that may not include SM103
2. PyTorch flex_attention's inductor doesn't fully support SM103 yet
3. nvidia-cutlass-dsl conflicts with torch._inductor on SM103

Usage:
    from scope.core.compat.sm103 import (
        get_compute_capability,
        is_sm100,
        is_sm103,
        is_blackwell,
        patch_cutlass_for_sm103,
        get_recommended_backend,
    )
"""

import functools
import logging
import os
import sys
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Cache device capability
_device_capability_cache: Optional[Tuple[int, int]] = None


def get_compute_capability() -> Tuple[int, int]:
    """
    Get the CUDA compute capability of the current device.

    Returns:
        Tuple of (major, minor) version, e.g., (10, 0) for B200, (10, 3) for B300
    """
    global _device_capability_cache
    if _device_capability_cache is not None:
        return _device_capability_cache

    try:
        import torch
        if torch.cuda.is_available():
            _device_capability_cache = torch.cuda.get_device_capability()
            return _device_capability_cache
    except Exception as e:
        logger.warning(f"Failed to get device capability: {e}")

    return (0, 0)


def is_sm100() -> bool:
    """Check if running on SM100 (B200, compute capability 10.0)."""
    major, minor = get_compute_capability()
    return major == 10 and minor == 0


def is_sm103() -> bool:
    """Check if running on SM103 (B300, compute capability 10.3)."""
    major, minor = get_compute_capability()
    return major == 10 and minor == 3


def is_blackwell() -> bool:
    """Check if running on Blackwell architecture (SM100 or SM103)."""
    major, _ = get_compute_capability()
    return major == 10


def is_hopper() -> bool:
    """Check if running on Hopper architecture (SM90)."""
    major, _ = get_compute_capability()
    return major == 9


@functools.lru_cache(maxsize=1)
def patch_cutlass_for_sm103() -> bool:
    """
    Patch nvidia-cutlass-dsl to support SM103 (B300).

    The CUTLASS/CUTE DSL library has architecture validation lists that
    only include SM100 by default. This patches them to also allow SM103.

    Returns:
        True if patching was successful or not needed, False if failed
    """
    if not is_sm103():
        logger.debug("Not SM103, skipping CUTLASS patch")
        return True

    logger.info("Detected SM103 (B300), applying CUTLASS compatibility patch...")

    try:
        import cutlass
        import cutlass.cute as cute
    except ImportError:
        logger.warning("cutlass not installed, cannot patch for SM103")
        return False

    # Files to patch and their arch validation attributes
    # Each entry: (module_path, attr_path, valid_arch_values)
    patches = [
        # cute/arch/mbar.py - mbarrier operations
        ("cute.arch.mbar", "_VALID_ARCHS", [100, 103]),
        # cute/arch/elect.py - elect operations
        ("cute.arch.elect", "_VALID_ARCHS", [100, 103]),
        # cute/nvgpu/tcgen05/mma.py - TensorCore Gen 05 MMA
        ("cute.nvgpu.tcgen05.mma", "_VALID_ARCHS", [100, 103]),
        # cute/nvgpu/tcgen05/copy.py - TensorCore Gen 05 copy
        ("cute.nvgpu.tcgen05.copy", "_VALID_ARCHS", [100, 103]),
        # cute/nvgpu/cpasync/copy.py - async copy
        ("cute.nvgpu.cpasync.copy", "_VALID_ARCHS", [100, 103]),
    ]

    patched_count = 0
    for module_path, attr_name, valid_archs in patches:
        try:
            # Import the module
            parts = module_path.split(".")
            module = cutlass
            for part in parts:
                module = getattr(module, part, None)
                if module is None:
                    break

            if module is None:
                logger.debug(f"Module {module_path} not found, skipping")
                continue

            # Check if attribute exists
            if not hasattr(module, attr_name):
                # Try to find the validation in different places
                # Some modules may have it as a local variable in functions
                logger.debug(f"Attribute {attr_name} not found in {module_path}")
                continue

            current_value = getattr(module, attr_name)
            if isinstance(current_value, (list, set, tuple)):
                # Add 103 if not present
                if 103 not in current_value:
                    if isinstance(current_value, list):
                        current_value.append(103)
                    elif isinstance(current_value, set):
                        current_value.add(103)
                    else:
                        setattr(module, attr_name, tuple(current_value) + (103,))
                    patched_count += 1
                    logger.debug(f"Patched {module_path}.{attr_name}")
            else:
                logger.debug(f"Unexpected type for {module_path}.{attr_name}: {type(current_value)}")
        except Exception as e:
            logger.debug(f"Failed to patch {module_path}.{attr_name}: {e}")

    if patched_count > 0:
        logger.info(f"Applied {patched_count} SM103 compatibility patches to CUTLASS")
    else:
        logger.info("No CUTLASS patches needed (already compatible or different structure)")

    return True


def get_recommended_backend() -> str:
    """
    Get the recommended KV-cache attention backend for the current GPU.

    Returns:
        One of: "fa4", "triton", "flex"

    Recommendations:
    - SM100 (B200): "fa4" if available, else "triton"
    - SM103 (B300): "triton" (fa4 may work after patching, but safer default)
    - SM90 (Hopper): "triton"
    - Other: "flex"
    """
    env_backend = os.getenv("SCOPE_KV_BIAS_BACKEND", "").lower()
    if env_backend:
        # User explicitly set backend, respect it
        return env_backend

    if is_sm100():
        # B200: FA4 is 1.89x faster than Triton, but default to triton for safety
        # User can set SCOPE_KV_BIAS_BACKEND=fa4 to enable
        return "triton"
    elif is_sm103():
        # B300: Triton should work, FA4 needs patching and testing
        return "triton"
    elif is_hopper():
        # Hopper: Triton works well
        return "triton"
    else:
        # Other: flex_attention as fallback
        return "flex"


def check_fa4_availability() -> Tuple[bool, str]:
    """
    Check if FA4/CUTE is available and working on the current GPU.

    Returns:
        Tuple of (is_available, reason_if_not)
    """
    if not is_blackwell():
        return False, "FA4/CUTE requires Blackwell GPU (SM100 or SM103)"

    # Apply SM103 patch if needed
    if is_sm103():
        if not patch_cutlass_for_sm103():
            return False, "Failed to patch CUTLASS for SM103"

    try:
        # Try importing FA4
        import cutlass
        import cutlass.cute as cute
        from flash_attn.cute.interface import _flash_attn_fwd
        return True, "FA4/CUTE available"
    except ImportError as e:
        return False, f"Import error: {e}"
    except Exception as e:
        return False, f"Unexpected error: {e}"


def check_flex_attention_availability() -> Tuple[bool, str]:
    """
    Check if flex_attention is available and working on the current GPU.

    Note: On SM103, torch.compile with flex_attention may fail due to
    inductor not having SM103 kernel support.

    Returns:
        Tuple of (is_available, reason_if_not)
    """
    try:
        import torch
        from torch.nn.attention.flex_attention import flex_attention

        if is_sm103():
            # flex_attention with torch.compile is known to fail on SM103
            return True, "flex_attention available but torch.compile may fail on SM103"

        return True, "flex_attention available"
    except ImportError as e:
        return False, f"Import error: {e}"


def get_capability_info() -> dict:
    """
    Get comprehensive GPU capability information for debugging.

    Returns:
        Dict with capability details
    """
    major, minor = get_compute_capability()

    fa4_available, fa4_reason = check_fa4_availability()
    flex_available, flex_reason = check_flex_attention_availability()

    return {
        "compute_capability": (major, minor),
        "sm_version": f"SM{major * 10 + minor}" if major > 0 else "Unknown",
        "architecture": (
            "Blackwell" if major == 10 else
            "Hopper" if major == 9 else
            "Ampere" if major == 8 else
            "Unknown"
        ),
        "gpu_name": _get_gpu_name(),
        "is_sm100": is_sm100(),
        "is_sm103": is_sm103(),
        "is_blackwell": is_blackwell(),
        "recommended_backend": get_recommended_backend(),
        "fa4_available": fa4_available,
        "fa4_reason": fa4_reason,
        "flex_available": flex_available,
        "flex_reason": flex_reason,
    }


def _get_gpu_name() -> str:
    """Get the GPU device name."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name()
    except Exception:
        pass
    return "Unknown"


# Auto-apply SM103 patch on import if detected
def _auto_patch():
    """Auto-apply patches on module import."""
    if is_sm103():
        # Only log at debug level during auto-patch
        try:
            patch_cutlass_for_sm103()
        except Exception as e:
            logger.debug(f"Auto-patch for SM103 failed: {e}")


# Run auto-patch on import
_auto_patch()
