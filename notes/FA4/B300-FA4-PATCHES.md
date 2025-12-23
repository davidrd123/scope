# FA4/CUTE on B300 (SM103) - Patch Guide

**Date**: 2025-12-23
**Status**: ✅ FA4 basic works, ❌ score_mod needs version matching

## Summary

FA4/CUTE can run on B300 (SM103) after patching `nvidia-cutlass-dsl` to accept the `sm_103a` architecture. Without patches, CUTLASS DSL rejects SM103 because it only allows `sm_100a` and `sm_100f`.

### Results After Patching

| Kernel | B300 (SM103) | Notes |
|--------|--------------|-------|
| **FA4 basic (causal)** | 0.074 ms | ✅ 13x faster than Triton |
| **Triton Kernel B** | 0.977 ms | ✅ Baseline |
| **flex_attention** | 1.094 ms | ✅ Works |
| **FA4 with score_mod** | ❌ | Version mismatch |

## Prerequisites

1. **CUDA 12.9** installed (for `ptxas` SM103 support)
2. **nvidia-cutlass-dsl==4.1.0** (required by flash_attn==2.8.3)
3. **TRITON_PTXAS_PATH** set to CUDA 12.9's ptxas

```bash
# Install correct cutlass version
uv pip install nvidia-cutlass-dsl==4.1.0

# Set ptxas path
export TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas
```

## Patches Required

### 1. Patch `impl_utils.py` (arch validation function)

This is the core fix - patches the `check_value_in` function to allow `sm_103a` wherever `sm_100a/sm_100f` is allowed.

**File**: `.venv/lib/python3.12/site-packages/nvidia_cutlass_dsl/python_packages/cutlass/impl_utils.py`

```python
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

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
```

### 2. Patch `tcgen05/mma.py` (TensorCore MMA ops)

Add `sm_103a` to all `admissible_archs` lists.

**File**: `.venv/lib/python3.12/site-packages/nvidia_cutlass_dsl/python_packages/cutlass/cute/nvgpu/tcgen05/mma.py`

```bash
sed -i 's/"sm_100a",/"sm_100a", "sm_103a",/g' mma.py
sed -i 's/"sm_100f",/"sm_100f", "sm_103a",/g' mma.py
```

### 3. Patch `tcgen05/copy.py` (TensorCore copy ops)

**File**: `.venv/lib/python3.12/site-packages/nvidia_cutlass_dsl/python_packages/cutlass/cute/nvgpu/tcgen05/copy.py`

```bash
sed -i 's/"sm_100a",/"sm_100a", "sm_103a",/g' copy.py
sed -i 's/"sm_100f",/"sm_100f", "sm_103a",/g' copy.py
```

### 4. Patch `cpasync/copy.py` (async copy ops)

**File**: `.venv/lib/python3.12/site-packages/nvidia_cutlass_dsl/python_packages/cutlass/cute/nvgpu/cpasync/copy.py`

```bash
sed -i 's/"sm_100a",/"sm_100a", "sm_103a",/g' copy.py
sed -i 's/"sm_100f",/"sm_100f", "sm_103a",/g' copy.py
```

### 5. Clear Python cache

```bash
find .venv/lib/python3.12/site-packages/nvidia_cutlass_dsl -name "*.pyc" -delete
find .venv/lib/python3.12/site-packages/nvidia_cutlass_dsl -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
```

## Automated Patch Script

Save as `scripts/patch_cutlass_sm103.sh`:

```bash
#!/bin/bash
# Patch nvidia-cutlass-dsl for SM103 (B300) support
# Requires: nvidia-cutlass-dsl==4.1.0

set -e

VENV_PATH="${1:-.venv}"
CUTLASS_PATH="$VENV_PATH/lib/python3.12/site-packages/nvidia_cutlass_dsl/python_packages/cutlass"

if [ ! -d "$CUTLASS_PATH" ]; then
    echo "Error: CUTLASS path not found: $CUTLASS_PATH"
    exit 1
fi

echo "Patching nvidia-cutlass-dsl for SM103 support..."

# 1. Patch impl_utils.py
IMPL_FILE="$CUTLASS_PATH/impl_utils.py"
cp "$IMPL_FILE" "${IMPL_FILE}.bak"

cat > "$IMPL_FILE" << 'EOF'
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
# PATCHED: Added SM103 support for B300 GPUs

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
EOF
echo "  Patched: impl_utils.py"

# 2. Patch tcgen05/mma.py
MMA_FILE="$CUTLASS_PATH/cute/nvgpu/tcgen05/mma.py"
cp "$MMA_FILE" "${MMA_FILE}.bak"
sed -i 's/"sm_100a",/"sm_100a", "sm_103a",/g' "$MMA_FILE"
sed -i 's/"sm_100f",/"sm_100f", "sm_103a",/g' "$MMA_FILE"
echo "  Patched: tcgen05/mma.py"

# 3. Patch tcgen05/copy.py
TCGEN_COPY="$CUTLASS_PATH/cute/nvgpu/tcgen05/copy.py"
cp "$TCGEN_COPY" "${TCGEN_COPY}.bak"
sed -i 's/"sm_100a",/"sm_100a", "sm_103a",/g' "$TCGEN_COPY"
sed -i 's/"sm_100f",/"sm_100f", "sm_103a",/g' "$TCGEN_COPY"
echo "  Patched: tcgen05/copy.py"

# 4. Patch cpasync/copy.py
CPASYNC_COPY="$CUTLASS_PATH/cute/nvgpu/cpasync/copy.py"
cp "$CPASYNC_COPY" "${CPASYNC_COPY}.bak"
sed -i 's/"sm_100a",/"sm_100a", "sm_103a",/g' "$CPASYNC_COPY"
sed -i 's/"sm_100f",/"sm_100f", "sm_103a",/g' "$CPASYNC_COPY"
echo "  Patched: cpasync/copy.py"

# 5. Clear pycache
find "$CUTLASS_PATH" -name "*.pyc" -delete 2>/dev/null || true
find "$CUTLASS_PATH" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
echo "  Cleared: __pycache__"

echo ""
echo "SM103 patches applied successfully!"
echo ""
echo "To verify:"
echo "  TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas python -c \\"
echo "    \"from flash_attn.cute.interface import _flash_attn_fwd; print('FA4 OK')\""
```

## Verification

```bash
# Set environment
export TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas

# Test FA4 import
python -c "from flash_attn.cute.interface import _flash_attn_fwd; print('FA4 imported')"

# Benchmark
python -c "
import torch
from flash_attn.cute.interface import _flash_attn_fwd
import time

B, H, S, D = 1, 16, 4680, 128
q = torch.randn(B, S, H, D, device='cuda', dtype=torch.bfloat16)
k = torch.randn(B, S, H, D, device='cuda', dtype=torch.bfloat16)
v = torch.randn(B, S, H, D, device='cuda', dtype=torch.bfloat16)

# Warmup
for _ in range(5):
    _flash_attn_fwd(q, k, v, causal=True, softmax_scale=1.0/D**0.5)
torch.cuda.synchronize()

# Benchmark
start = time.perf_counter()
for _ in range(50):
    _flash_attn_fwd(q, k, v, causal=True, softmax_scale=1.0/D**0.5)
torch.cuda.synchronize()
print(f'FA4: {(time.perf_counter()-start)/50*1000:.3f} ms')
"
```

## What Works vs What Doesn't

### ✅ Works on B300 after patches

1. **FA4 basic attention** (causal, non-causal)
   - Forward pass: ✅
   - Performance: 0.074ms (13x faster than Triton)

2. **All Blackwell tcgen05 ops**
   - MMA (matrix multiply-accumulate)
   - TMA copy operations
   - Async copy operations

### ❌ Doesn't work yet

1. **FA4 with score_mod**
   - The `flash-attention.bak` repo (with score_mod) expects different cutlass-dsl API
   - `FastDivmodDivisor` not in cutlass-dsl 4.1.0
   - Need matched versions of flash-attn wheel + cutlass-dsl

2. **FA4 backward pass**
   - Not tested on B300
   - May need additional patches

## Version Compatibility Matrix

| flash-attn | cutlass-dsl | score_mod | B300 Status |
|------------|-------------|-----------|-------------|
| 2.8.3 wheel | 4.1.0 | ❌ | ✅ Basic works after patches |
| flash-attention.bak | 4.1.0 | ✅ | ❌ API mismatch |
| flash-attention.bak | ??? | ✅ | 🔍 Need to find matching version |

## Next Steps

1. **Find matching cutlass-dsl for flash-attention.bak**
   - Check what version of cutlass-dsl exports `FastDivmodDivisor`
   - May need cutlass-dsl 4.2.x or 4.3.x with different patches

2. **Integrate FA4 into daydream-scope**
   - Add `SCOPE_KV_BIAS_BACKEND=fa4` option
   - Auto-detect SM103 and apply patches at runtime
   - Fallback to Triton if FA4 fails

3. **Benchmark full pipeline**
   - Test FA4 in render_timeline on B300
   - Compare FPS: FA4 vs Triton vs flex_attention

## Files Modified

On B300 machine, these files were patched:

```
.venv/lib/python3.12/site-packages/nvidia_cutlass_dsl/python_packages/cutlass/
├── impl_utils.py                    # check_value_in SM103 support
└── cute/nvgpu/
    ├── tcgen05/
    │   ├── mma.py                   # admissible_archs += sm_103a
    │   └── copy.py                  # admissible_archs += sm_103a
    └── cpasync/
        └── copy.py                  # admissible_archs += sm_103a
```

Backups created with `.bak` extension.

## References

- [NVIDIA CUDA 12.9 SM103 Support](https://docs.nvidia.com/cuda/cuda-features-archive/index.html)
- [Triton SM103 Issue](https://github.com/triton-lang/triton/issues/8473)
- [Blackwell Compatibility Guide](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/)
