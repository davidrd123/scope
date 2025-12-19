#!/usr/bin/env python3
"""
Patch nvidia-cutlass-dsl to support SM103 (B300/B300A) architecture.

FA4 (Flash Attention 4) with CUTE DSL only officially supports sm_100a/sm_100f.
The B300 uses sm_103, which requires patching the arch validation checks.

Usage:
    python scripts/patch_cutlass_sm103.py

Or:
    uv run python scripts/patch_cutlass_sm103.py
"""

import os
import re
import sys
from pathlib import Path


def find_venv_site_packages():
    """Find the site-packages directory for the current venv."""
    # Try common locations
    candidates = [
        Path(".venv/lib/python3.12/site-packages"),
        Path(".venv/lib/python3.11/site-packages"),
        Path(".venv/lib/python3.10/site-packages"),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Fallback to sys.path
    for p in sys.path:
        if "site-packages" in p:
            return Path(p)

    return None


def patch_file(filepath: Path, search: str, replace: str) -> bool:
    """Patch a file by replacing search pattern with replace string."""
    if not filepath.exists():
        return False

    content = filepath.read_text()
    if search in content and replace not in content:
        new_content = content.replace(search, replace)
        filepath.write_text(new_content)
        return True
    return False


def main():
    site_packages = find_venv_site_packages()
    if not site_packages:
        print("ERROR: Could not find site-packages directory")
        sys.exit(1)

    cutlass_base = site_packages / "nvidia_cutlass_dsl/python_packages/cutlass"

    if not cutlass_base.exists():
        print("ERROR: nvidia-cutlass-dsl not installed")
        sys.exit(1)

    print(f"Patching cutlass at: {cutlass_base}")

    # Files that need arch list patches
    patch_patterns = [
        # Pattern: add sm_103 variants after sm_100f
        ('"sm_100f",', '"sm_100f", "sm_103", "sm_103a",'),
        # Pattern: add sm_103 variants in multi-line lists ending with sm_100f
        ('"sm_100f",\n    ]', '"sm_100f",\n        "sm_103",\n        "sm_103a",\n    ]'),
        ('"sm_100f",\n        ]', '"sm_100f",\n            "sm_103",\n            "sm_103a",\n        ]'),
        # Pattern: add sm_103 to lists ending with sm_100a (MX format ops)
        ('"sm_100a",\n    ]', '"sm_100a",\n        "sm_103",\n        "sm_103a",\n    ]'),
    ]

    files_to_patch = [
        cutlass_base / "cute/arch/mbar.py",
        cutlass_base / "cute/arch/elect.py",
        cutlass_base / "cute/nvgpu/tcgen05/mma.py",
        cutlass_base / "cute/nvgpu/tcgen05/copy.py",
        cutlass_base / "cute/nvgpu/cpasync/copy.py",
    ]

    patched_count = 0
    for filepath in files_to_patch:
        if not filepath.exists():
            print(f"  SKIP: {filepath.name} (not found)")
            continue

        file_patched = False
        for search, replace in patch_patterns:
            if patch_file(filepath, search, replace):
                file_patched = True

        if file_patched:
            print(f"  PATCHED: {filepath.relative_to(cutlass_base)}")
            patched_count += 1
        else:
            # Check if already patched
            content = filepath.read_text()
            if "sm_103" in content:
                print(f"  OK: {filepath.relative_to(cutlass_base)} (already patched)")
            else:
                print(f"  SKIP: {filepath.relative_to(cutlass_base)} (no changes needed)")

    print(f"\nPatched {patched_count} files for SM103 support")
    print("\nTo test FA4:")
    print('  uv run python -c "from flash_attn.cute import flash_attn_func; print(\'FA4 OK\')"')


if __name__ == "__main__":
    main()
