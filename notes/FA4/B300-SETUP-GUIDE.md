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

| Feature | B200 (SM100) | B300 (SM103) |
|---------|--------------|--------------|
| Triton Kernel B | ✅ | ✅ (with CUDA 12.9 ptxas) |
| flex_attention | ✅ | ✅ |
| RoPE fused | ✅ | ✅ |
| FA4/CUTE score_mod | ✅ | ❓ Needs separate venv with cutlass-dsl |

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
4. `notes/FA4/b300-investigation.md` - Full investigation log

## Next Steps (If Needed)

1. **FA4/CUTE on B300**: Requires separate venv with nvidia-cutlass-dsl (conflicts with pip flash-attn)
2. **Performance tuning**: Block sizes may benefit from B300-specific tuning
3. **Full pipeline test**: Run `render_timeline` on B300
