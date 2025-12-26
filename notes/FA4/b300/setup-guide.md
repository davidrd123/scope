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

## Recommended: Two Virtual Environments (B200 vs B300)

The repo default environment (`torch==2.8.0+cu128`, `cuda==12.8`) is “good enough to run” on SM103 once you point Triton at a newer `ptxas`,
but it is **not an SM103-native runtime stack** (cuDNN/cuBLAS/etc are still the cu128 bundle).

Given B300’s perf issue appears dominated by **`denoise` + `decode`** (not just attention), it’s worth having a clean way to experiment with a
newer CUDA-runtime PyTorch build without destabilizing the B200 environment.

### How to do it with `uv`

`uv` supports selecting the environment directory via `UV_PROJECT_ENVIRONMENT` (this is already used by the app setup code).

Create two environments:

```bash
# Baseline (keep current lockfile / cu128 stack) – good for B200
UV_PROJECT_ENVIRONMENT=.venv-b200 uv sync

# Experimental (for B300) – start from the lock, then override torch
UV_PROJECT_ENVIRONMENT=.venv-b300 uv sync
```

Run commands against a chosen env:

```bash
UV_PROJECT_ENVIRONMENT=.venv-b300 TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas uv run daydream-scope
UV_PROJECT_ENVIRONMENT=.venv-b200 uv run daydream-scope
```

### “SM103-native” stack (what it means)

To be truly SM103-native, you want:

- **PyTorch built with CUDA >= 12.9 / 13.x** (e.g. `cu129` / `cu130` wheels if available)
- A CUDA toolkit `ptxas` that recognizes `sm_103` (already handled via `TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas`)

Why this might matter:

- Decode is heavy **3D conv** (`WanVAE_.stream_decode`), which is typically **cuDNN**-dominated.
- It’s plausible `torch==+cu128`’s bundled cuDNN doesn’t have the best SM103 kernels yet, even if GEMM looks great.

#### Update (2025-12-24): cu130 makes decode ~4× faster

We now have direct evidence that the runtime stack is the lever on B300:

- `torch 2.8.0+cu129` (CUDA 12.9, cuDNN 9.10): `stream_decode(t=3)` ~`760ms/call`
- `torch 2.9.0+cu130` (CUDA 13.0, cuDNN 9.13): `stream_decode(t=3)` ~`194ms/call` (**~3.9× faster**)

Logs:
- `outputs/b300_cu129_vae_stream_decode_bench.log`
- `outputs/b300_cu130_vae_stream_decode_bench.log`

This strongly suggests the “8.8 FPS” issue is largely a **cuDNN/conv3d stack** issue, not attention.

#### Update (2025-12-24): cu130 + FlashAttention restores end-to-end FPS

On torch `2.9.0+cu130` (triton `3.5.0`), `torch.compile(flex_attention)` currently hard-aborts on SM103
(`LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.tcgen05.wait.st`). For stability:

- Set `DISABLE_FLEX_ATTENTION_COMPILE=1`
- Ensure `flash_attn` is installed, otherwise KV-bias falls back to slow paths (can look like ~`1 FPS`)
- If you see `torchao` warnings like “Skipping import of cpp extensions…”, install a torch 2.9-compatible torchao (per torchao’s matrix: `torchao==0.14.1`) or just run `scripts/b300_env_fix_cu130.sh` (it does a best-effort torchao upgrade).

Install FlashAttention in the cu130 env (note: builds a large CUDA extension, ~1GB):

```bash
uv pip install -p .venv-b300-cu130-decode/bin/python wheel ninja
uv pip install -p .venv-b300-cu130-decode/bin/python --no-deps --no-build-isolation --no-binary flash-attn flash-attn==2.8.3
```

Reference benchmark result on B300 (`320x576`, FP8, bias=0.3):
- `outputs/b300_cu130_fp8_bias03_flashattn.log` → **~13.3–13.5 FPS**

### Running Daydream on B300 (cu130 env)

This avoids colliding with the shared `.venv` and sets the key SM103 env vars:

```bash
./scripts/setup_b300_cu130_env.sh .venv-b300-cu130-decode  # one-time (or after uv sync clobbers torch)
./scripts/run_daydream_b300.sh
```

If the cu130 env ever gets clobbered back to cu128 (e.g. by `uv sync`), repair it with:

```bash
./scripts/b300_env_fix_cu130.sh .venv-b300-cu130-decode
```

Practical experiment:

1) Keep `.venv-b200` unchanged (working baseline).
2) In `.venv-b300`, install a CUDA 12.9/13 build of PyTorch if/when available, then re-run the block profile at `320x576` and compare `decode` + `denoise`.

Note: wheel availability / exact index URLs change quickly; check your preferred PyTorch index for `cu129`/`cu130` before pinning anything.

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

### Decode-Only Quick Check (works without the full pipeline)

This is a cheap discriminator when you’re unsure whether the box/runtime is “good” for SM103.

```bash
uv venv .venv-b300-cu130-decode --python 3.12
uv pip install -p .venv-b300-cu130-decode/bin/python --index-url https://download.pytorch.org/whl/cu130 torch==2.9.0+cu130
uv pip install -p .venv-b300-cu130-decode/bin/python einops numpy

PYTHONPATH=src .venv-b300-cu130-decode/bin/python scripts/bench_wanvae_stream_decode.py \
  --height 320 --width 576 --t 3 --cudnn-benchmark
```

## What Works on B300

| Feature | B200 (SM100) | B300 (SM103) | Notes |
|---------|--------------|--------------|-------|
| Triton Kernel B | ✅ | ✅ | 0.977 ms (with CUDA 12.9 ptxas) |
| flex_attention | ✅ | ✅ | 1.095 ms |
| RoPE fused | ✅ | ✅ | 0.029 ms |
| FA4/CUTE basic | ✅ | ✅ | **0.074 ms** (13x faster!) - needs patches |
| FA4/CUTE score_mod | ✅ | ❌ | Version mismatch, see B300-FA4-PATCHES.md |
| daydream-scope | ✅ 20 FPS | ✅ 8.8 FPS (baseline) / ✅ ~13.3 FPS (cu130+flash-attn) | End-to-end sensitive to runtime stack |

## FA4/CUTE on B300 (Optional)

FA4 basic attention works on B300 after patching `nvidia-cutlass-dsl`. It's **13x faster** than Triton for causal attention!

> ⚠️ IMPORTANT: `nvidia-cutlass-dsl` conflicts with PyTorch Inductor in this repo.
>
> Symptom: `torch.compile` can fail with `NoValidChoicesError` for `flex_attention` because `nvidia-cutlass-dsl` installs a top-level `cutlass` module that shadows `torch._inductor`’s internal cutlass utilities.
>
> Recommendation: use a separate venv for FA4/CUTE experiments, and uninstall FA4 deps before running the normal pipeline.
>
> Details: `notes/FA4/b300/investigation.md` (Issue 2).

```bash
# 1. Install dependencies for flash_attn.cute (CuTe)
uv pip install cuda-python nvidia-cutlass-dsl==4.1.0

# 2. Apply SM103 patches (pass your venv dir if not using `.venv`)
PATH=.venv/bin:$PATH ./scripts/patch_cutlass_sm103.sh .venv

# 3. Verify FA4 works
PATH=.venv/bin:$PATH python -c "from flash_attn.cute.interface import _flash_attn_fwd; print('FA4 OK')"
```

To uninstall the FA4 deps:

```bash
uv pip uninstall nvidia-cutlass-dsl cuda-python cuda-bindings cuda-pathfinder
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
