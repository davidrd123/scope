#!/bin/bash
# Patch nvidia-cutlass-dsl for SM103 (B300) support
# Requires: nvidia-cutlass-dsl==4.1.0
#
# Usage: ./scripts/patch_cutlass_sm103.sh [venv_path]
#
# This script patches CUTLASS DSL to allow sm_103a architecture
# so FA4/CUTE can run on B300 GPUs.

set -e

VENV_PATH="${1:-.venv}"
CUTLASS_PATH="$VENV_PATH/lib/python3.12/site-packages/nvidia_cutlass_dsl/python_packages/cutlass"
PYTHON_BIN="$VENV_PATH/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="python"
fi

# Check if running on B300
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    if [[ ! "$GPU_NAME" =~ "B300" ]]; then
        echo "Warning: This script is intended for B300 GPUs, detected: $GPU_NAME"
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

if [ ! -d "$CUTLASS_PATH" ]; then
    echo "Error: CUTLASS path not found: $CUTLASS_PATH"
    echo ""
    echo "Make sure nvidia-cutlass-dsl==4.1.0 is installed:"
    echo "  uv pip install nvidia-cutlass-dsl==4.1.0"
    exit 1
fi

# Check version
CUTLASS_VERSION=$(
    "$PYTHON_BIN" -c "import importlib.metadata as m; print(m.version('nvidia-cutlass-dsl'))" 2>/dev/null || echo "unknown"
)
if [ "$CUTLASS_VERSION" != "4.1.0" ]; then
    echo "Warning: Expected nvidia-cutlass-dsl==4.1.0, found $CUTLASS_VERSION"
    echo "This script was tested with 4.1.0 and may not work with other versions."
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "Patching nvidia-cutlass-dsl for SM103 (B300) support..."
echo "  CUTLASS path: $CUTLASS_PATH"
echo ""

# 1. Patch impl_utils.py
IMPL_FILE="$CUTLASS_PATH/impl_utils.py"
if [ -f "${IMPL_FILE}.bak" ]; then
    echo "  impl_utils.py already has backup, skipping backup"
else
    cp "$IMPL_FILE" "${IMPL_FILE}.bak"
fi

cat > "$IMPL_FILE" << 'IMPL_EOF'
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# PATCHED: Added SM103 support for B300 GPUs
# See: notes/FA4/B300-FA4-PATCHES.md

def check_value_in(
    value, possible_values: list, value_description: str, prefix=""
) -> None:
    # SM103 patch: allow sm_103a wherever sm_100a/sm_100f is allowed
    if value == 'sm_103a' and value_description == 'arch':
        if 'sm_100a' in possible_values or 'sm_100f' in possible_values:
            possible_values = list(possible_values) + ['sm_103a']
    if value not in possible_values:
        err_msg = prefix
        if err_msg != "":
            err_msg += ": "
        err_msg += f"invalid {value_description}, got {value}, must be one of {possible_values}"
        raise ValueError(err_msg)


def check_type_in(ty, possible_types: list, type_description: str, prefix="") -> None:
    if not isinstance(ty, type):
        ty = type(ty)
    if ty not in possible_types:
        err_msg = prefix
        if err_msg != "":
            err_msg += ": "
        err_msg += f"invalid type for {type_description}, got {ty}, must be one of {possible_types}"
        raise TypeError(err_msg)
IMPL_EOF
echo "  ✓ Patched: impl_utils.py"

# 2. Patch tcgen05/mma.py
MMA_FILE="$CUTLASS_PATH/cute/nvgpu/tcgen05/mma.py"
if [ -f "${MMA_FILE}.bak" ]; then
    echo "  tcgen05/mma.py already has backup, restoring before patching"
    cp "${MMA_FILE}.bak" "$MMA_FILE"
else
    cp "$MMA_FILE" "${MMA_FILE}.bak"
fi
sed -i 's/"sm_100a",/"sm_100a", "sm_103a",/g' "$MMA_FILE"
sed -i 's/"sm_100f",/"sm_100f", "sm_103a",/g' "$MMA_FILE"
echo "  ✓ Patched: tcgen05/mma.py"

# 3. Patch tcgen05/copy.py
TCGEN_COPY="$CUTLASS_PATH/cute/nvgpu/tcgen05/copy.py"
if [ -f "${TCGEN_COPY}.bak" ]; then
    cp "${TCGEN_COPY}.bak" "$TCGEN_COPY"
else
    cp "$TCGEN_COPY" "${TCGEN_COPY}.bak"
fi
sed -i 's/"sm_100a",/"sm_100a", "sm_103a",/g' "$TCGEN_COPY"
sed -i 's/"sm_100f",/"sm_100f", "sm_103a",/g' "$TCGEN_COPY"
echo "  ✓ Patched: tcgen05/copy.py"

# 4. Patch cpasync/copy.py
CPASYNC_COPY="$CUTLASS_PATH/cute/nvgpu/cpasync/copy.py"
if [ -f "${CPASYNC_COPY}.bak" ]; then
    cp "${CPASYNC_COPY}.bak" "$CPASYNC_COPY"
else
    cp "$CPASYNC_COPY" "${CPASYNC_COPY}.bak"
fi
sed -i 's/"sm_100a",/"sm_100a", "sm_103a",/g' "$CPASYNC_COPY"
sed -i 's/"sm_100f",/"sm_100f", "sm_103a",/g' "$CPASYNC_COPY"
echo "  ✓ Patched: cpasync/copy.py"

# 5. Clear pycache
find "$CUTLASS_PATH" -name "*.pyc" -delete 2>/dev/null || true
find "$CUTLASS_PATH" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
echo "  ✓ Cleared: __pycache__"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  SM103 patches applied successfully!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "To verify FA4 works:"
echo ""
echo "  export TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas"
echo "  python -c \"from flash_attn.cute.interface import _flash_attn_fwd; print('FA4 OK')\""
echo ""
echo "To restore original files:"
echo "  cp ${IMPL_FILE}.bak $IMPL_FILE"
echo "  cp ${MMA_FILE}.bak $MMA_FILE"
echo "  cp ${TCGEN_COPY}.bak $TCGEN_COPY"
echo "  cp ${CPASYNC_COPY}.bak $CPASYNC_COPY"
echo ""
