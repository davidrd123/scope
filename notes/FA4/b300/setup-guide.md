# B300 (SM103) Setup Guide

**Date**: 2025-12-23
**Status**: ✅ WORKING

## Quick Setup for B300 Machines

```bash
# 1. Install CUDA 12.9 (has SM103 support)
sudo apt update && sudo apt install -y cuda-toolkit-12-9

# 2. Add ptxas path to shell profile (RECOMMENDED - works with uv run)
echo 'export TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas' >> ~/.bashrc
source ~/.bashrc

# 3. Verify it works
uv run python scripts/triton_sdpa.py --kernel-b
uv run daydream-scope  # Full server
```

### Alternative: Per-command (no persistence)
```bash
TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas uv run daydream-scope
```

### Note on `.venv/bin/activate`
Adding to `.venv/bin/activate` only works when manually activating the venv with `source .venv/bin/activate`. It does NOT work with `uv run` which bypasses activate scripts. Use `~/.bashrc` instead for `uv run` compatibility.

## Why This Is Needed

- **CUDA 12.8's ptxas** doesn't support `sm_103a` (B300's architecture)
- **Triton 3.4.0** bundles an older ptxas that also doesn't support SM103
- **CUDA 12.9** added SM103 support
- Setting `TRITON_PTXAS_PATH` tells Triton to use the system ptxas instead of bundled

## Verified Working on B300

| Component | Status | Performance |
|-----------|--------|-------------|
| **Triton Kernel B** | ✅ PASS | 0.977 ms |
| **flex_attention** | ✅ PASS | 1.095 ms |
| **RoPE fused** | ✅ PASS | 0.029 ms |
| **Correctness tests** | ✅ ALL PASS | |

**Triton Kernel B is 10.8% faster than flex_attention on B300!**

## Environment Details

```
GPU: NVIDIA B300 SXM6 AC
Compute Capability: (10, 3) = SM103
CUDA Toolkit: 12.9 (for ptxas)
PyTorch: 2.8.0+cu128
Triton: 3.4.0
Python: 3.12
```

## What Works on B300

| Feature | B200 (SM100) | B300 (SM103) | Notes |
|---------|--------------|--------------|-------|
| Triton Kernel B | ✅ | ✅ | 0.977 ms (with CUDA 12.9 ptxas) |
| flex_attention | ✅ | ✅ | 1.095 ms |
| RoPE fused | ✅ | ✅ | 0.029 ms |
| FA4/CUTE basic | ✅ | ✅ | **0.074 ms** (13x faster!) - needs patches |
| FA4/CUTE score_mod | ✅ | ❌ | Version mismatch, see B300-FA4-PATCHES.md |
| daydream-scope | ✅ 20 FPS | ✅ 8.8 FPS | Baseline, room for optimization |

## FA4/CUTE on B300 (Optional)

FA4 basic attention works on B300 after patching `nvidia-cutlass-dsl`. It's **13x faster** than Triton for causal attention!

```bash
# 1. Install correct cutlass version
uv pip install nvidia-cutlass-dsl==4.1.0

# 2. Apply SM103 patches
./scripts/patch_cutlass_sm103.sh

# 3. Verify FA4 works
python -c "from flash_attn.cute.interface import _flash_attn_fwd; print('FA4 OK')"
```

**Note**: FA4 with `score_mod` (for Kernel B equivalent) doesn't work yet due to version mismatches. See `notes/FA4/b300/fa4-patches.md` for details.

## Troubleshooting

### Error: `ptxas fatal: Value 'sm_103a' is not defined`

**Cause**: Using old ptxas that doesn't know SM103

**Fix**:
```bash
# Verify CUDA 12.9 is installed
/usr/local/cuda-12.9/bin/ptxas --help | grep sm_103

# Set the env var
export TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas
```

### Error: `NoValidChoicesError` for flex_attention

**Cause**: torch.compile failing on SM103

**Fix**: The TRITON_PTXAS_PATH fix should resolve this. If not, the SM103 compat layer in `src/scope/core/compat/sm103.py` can help gate compilation.

## Files Modified/Created

1. `.venv/bin/activate` - Added `TRITON_PTXAS_PATH`
2. `src/scope/core/compat/sm103.py` - SM103 compatibility utilities
3. `src/scope/core/compat/__init__.py` - Compat module init
4. `notes/FA4/b300/investigation.md` - Full investigation log
5. `notes/FA4/b300/fa4-patches.md` - FA4/CUTE SM103 patch documentation
6. `scripts/patch_cutlass_sm103.sh` - Automated CUTLASS patching script

### Files Patched for FA4 (by patch script)

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

## Next Steps

1. **Improve B300 FPS** (currently 8.8 FPS vs B200's 20 FPS)
   - Profile to find bottlenecks
   - Test with FA4 basic attention in pipeline

2. **FA4 score_mod on B300**
   - Find matching cutlass-dsl version for flash-attention.bak
   - Or wait for upstream FA4 wheel with score_mod support

3. **Performance tuning**
   - Block sizes may benefit from B300-specific tuning
   - SM103 has 148 SMs vs B200's 160 SMs
