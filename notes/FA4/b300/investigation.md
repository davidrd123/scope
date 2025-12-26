# B300 (SM103) Investigation Notes

## Environment
- **GPU**: NVIDIA B300 SXM6 AC → now testing on **B200**
- **Compute Capability**: SM103 (10.3) for B300, **SM100 (10.0) for B200**
- **PyTorch**: 2.8.0+cu128
- **CUDA**: 12.8 (historical on B300)
- **Python**: 3.12.3

## Update: SM103 requires newer `ptxas` (CUDA 12.9+)

The earlier B300 failures included:
- `ptxas fatal: Value 'sm_103a' is not defined for option 'gpu-name'`
- `NoValidChoicesError: target: flex_attention` during `torch.compile` autotune

These can happen when Triton/Inductor is assembling with an older `ptxas` (e.g., CUDA 12.8).

**Fix:** ensure Triton uses a CUDA toolkit whose `ptxas` supports SM103, e.g. CUDA **12.9+**.

On the machine backing this repo, CUDA 12.9 is installed and supports SM103:
```bash
/usr/local/cuda-12.9/bin/ptxas --help | rg "sm_103"
```

Recommended env var (set before importing anything that triggers Triton compilation):
```bash
export TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas
```

Note: Triton often uses its bundled `ptxas` by default (inside the Python wheel), which may be
too old for SM103. The above override forces the system toolkit `ptxas` instead.

If `torch.compile(flex_attention, ...)` still fails after this, *then* it’s likely a genuine SM103 support gap in the PyTorch/Triton stack.

### B200 vs B300
- B200 = SM100 (officially supported by PyTorch flex_attention and FA4/CUTE)
- B300 = SM103 (NOT supported, requires patches)

---

## B200 (SM100) Test Results ✅

### Compiled flex_attention WORKS on SM100
```bash
# This works on B200, fails on B300:
flex_attention = torch.compile(_flex_attention, dynamic=False, mode='max-autotune-no-cudagraphs')
```

Autotune output shows kernel selection working:
```
AUTOTUNE flex_decoding(1x8x1x64x64, ...)
  triton_flex_decoding_1 0.0061 ms 100.0%
```

### Attention Backend Status on B200
- FA2: ✅ Available
- FA3: ❌ (H100 only)
- FA4: ✅ Available (local flash-attention repo)
- SageAttn: ❌ (no cp312 wheel)

### nvidia-cutlass-dsl Present but No Conflict on SM100
Even with nvidia-cutlass-dsl 4.3.3 installed, flex_attention compilation succeeds.
The conflict only manifests on SM103 (B300).

### Conclusion
**SM100 (B200) works out of the box with PyTorch 2.8.** No patches needed for flex_attention.
SM103 (B300) requires either:
1. Skip torch.compile for flex_attention on SM103
2. Wait for PyTorch to add SM103 kernel support

### FA4 Version Conflict on B200
The local `flash-attention/` repo breaks FA4! Here's why:

1. `attention.py` has `_extend_flash_attn_path()` that inserts local repo into path
2. Local repo's cute requires `FastDivmodDivisor` (cutlass-dsl >= 4.3.3)
3. Pip flash_attn's cute requires `cutlass.utils.ampere_helpers` (cutlass-dsl 4.1.0)

**Fix**: Rename or remove local flash-attention repo:
```bash
mv /root/scope/flash-attention /root/scope/flash-attention.bak
```

Then FA4 will use the pip-installed version which works with cutlass-dsl 4.1.0.

---

## Issue 1: FA4 (CUTE) - RESOLVED FOR BENCHMARKS

### Problem
FA4 (Flash Attention 4 via CUTE DSL) only officially supports sm_100a/sm_100f, not sm_103.

### Solution
Created `scripts/patch_cutlass_sm103.py` that patches nvidia-cutlass-dsl to add sm_103 support:
- Patches arch validation lists in `cute/arch/mbar.py`, `cute/arch/elect.py`
- Patches `cute/nvgpu/tcgen05/mma.py`, `cute/nvgpu/tcgen05/copy.py`, `cute/nvgpu/cpasync/copy.py`

### Result
FA4 benchmarks work after patching:
```
Config          Batch  Seq    Heads  Dim    Time (ms)    TFLOPS
Small           2      512    8      64     0.046        11.59
Medium          2      1024   8      64     0.042        51.05
Large           2      2048   8      128    0.043        404.03
XL              1      4096   8      128    0.046        739.98
```

---

## Issue 2: nvidia-cutlass-dsl Conflicts with PyTorch Inductor (cutlass shadowing)

### Problem
When FA4 deps are installed, `nvidia-cutlass-dsl` provides a `cutlass` module that shadows PyTorch's internal cutlass utilities in `torch._inductor`.

### Symptom
```
NoValidChoicesError: target: flex_attention
...
AttributeError: module 'cutlass' has no attribute 'CACHE_FILE'
```

### Update (2025-12-26)
This is no longer a hard blocker for the main pipeline: **FA4 KV-bias can coexist with regional `torch.compile`** on B300/cu130 by keeping CuTe calls opaque to Dynamo and disabling flex_attention compilation on SM103 (see `notes/FA4/b300/session-state.md` and `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py`).

### Mitigations / Workarounds
If you need to compile `flex_attention` (or hit Inductor compilation failures that trace to `torch._inductor.codegen.cuda.cutlass_utils`), uninstall FA4 deps:
```bash
uv pip uninstall nvidia-cutlass-dsl cuda-python cuda-bindings cuda-pathfinder
```

If you’re running FA4 KV-bias with regional compile, prefer:
- `DISABLE_FLEX_ATTENTION_COMPILE=1` on SM103 (avoid tcgen05 LLVM aborts)
- Leave CuTe calls opaque to Dynamo (already done in the code path above)

---

## Issue 3: flex_attention on B300 - UNDER INVESTIGATION

### Problem
Even without FA4 deps, `render_timeline` fails on B300 with `NoValidChoicesError` for `flex_attention`.

### Root Cause Hypothesis
Most likely: Triton/Inductor is using an older `ptxas` that doesn’t recognize `sm_103a`, so all candidate kernels fail to assemble and autotune surfaces `NoValidChoicesError`.

Secondary possibility (if using CUDA 12.9+ `ptxas` doesn’t fix it): PyTorch 2.8’s inductor lacks complete SM103 kernel support for flex_attention’s `"max-autotune-no-cudagraphs"` mode.

### What We Tried

#### Attempt 1: TORCH_COMPILE_DISABLE=1
**Result**: Different error - `'tuple' object has no attribute 'unflatten'`

**Analysis**: When torch.compile is disabled, something in the attention path returns a tuple instead of a tensor. The code at `causal_model.py:675` does:
```python
x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)
```
Where `y = self.self_attn(...)`. So `self_attn` returns a tuple when uncompiled.

**Key insight**: flex_attention itself works fine uncompiled (tested independently), so the tuple must come from elsewhere in the model.

#### Attempt 2: Direct flex_attention test
```python
TORCH_COMPILE_DISABLE=1 python -c "
from torch.nn.attention.flex_attention import flex_attention
q = torch.randn(1, 8, 64, 64, dtype=torch.bfloat16, device='cuda')
out = flex_attention(q, k, v)
print(type(out))  # <class 'torch.Tensor'> - works!
"
```
**Result**: Works fine, returns tensor.

### Hypotheses to Test Next

1. **Something else in `self_attn` returns tuple**: The `CausalWanSelfAttention.forward()` method may have conditional paths that return different types

2. **PyTorch version mismatch**: flex_attention API may have changed between versions, and the code assumes compiled behavior

3. **SM103-specific codepath**: There may be architecture detection that triggers different behavior on B300

4. **The `attention()` fallback path**: Line 571 has `x = attention(roped_query, cached_k, cached_v)` as fallback - this might return tuple

### Code Locations to Check
- `causal_model.py:562-579` - flex_attention call and return
- `causal_model.py:571` - `attention()` fallback function
- `attention.py` - the `attention()` function implementation
- Check if `kv_cache_attention_bias` affects which path is taken

---

## Key Files

### Modified for FA4 Support
- `pyproject.toml` - Added cp312 flash-attn wheels, fa4 dependency group
- `.python-version` - Changed to 3.12
- `scripts/patch_cutlass_sm103.py` - SM103 arch patches

### Relevant Source Files
- `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py` - flex_attention usage
- `src/scope/core/pipelines/wan2_1/modules/attention.py` - attention backends (FA2/FA3/FA4)

---

## LATEST FINDINGS (Claude session - save state)

### Local flash-attention Directory Found!
There's a cloned `flash-attention` repo at `/root/scope/flash-attention/` that gets injected into the path via `_extend_flash_attn_path()` in attention.py:
```python
for parent in base_path.parents:
    candidate = parent / "flash-attention" / "flash_attn"
    if candidate.is_dir():
        _flash_attn.__path__.insert(0, candidate_str)
```

This local repo has its own `flash_attn/cute/` directory which may work differently!

### Current Attention Backend Status on B300 (Python 3.12)
```
SAGEATTN_AVAILABLE: False  (no cp312 wheel)
FLASH_ATTN_2_AVAILABLE: True
FLASH_ATTN_3_AVAILABLE: False (H100 only)
FLASH_ATTN_4_AVAILABLE: True  (from local flash-attention repo!)
```

**KEY INSIGHT**: FA4 shows as available because the LOCAL `flash-attention/flash_attn/cute/` is being loaded, NOT the pip-installed package. The local version might have different dependencies or behavior!

### The Actual Error Flow
1. Without FA4 deps: `render_timeline` fails with `NoValidChoicesError` for `flex_attention` in torch.compile
2. With `TORCH_COMPILE_DISABLE=1`: Different error - `'tuple' object has no attribute 'unflatten'`
3. The tuple error comes from `y = self.self_attn(...)` at causal_model.py:658
4. With `DISABLE_FLASH_ATTENTION_4=1`: **SAME flex_attention error** - this is NOT about FA4!

### ROOT CAUSE IDENTIFIED
The problem is in `causal_model.py` lines 26-28:
```python
flex_attention = torch.compile(
    flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs"
)
```
This compiles flex_attention at MODULE IMPORT TIME. PyTorch 2.8's inductor doesn't have SM103 kernel support for flex_attention's max-autotune mode.

**This is a PyTorch + B300/SM103 compatibility issue, NOT an FA4 issue!**

### Next Investigation Steps
1. **Check local flash-attention cute interface** - does it return tuple?
2. **Remove/rename local flash-attention dir** to use only pip package
3. **Check if FA4 path is being taken** despite cutlass-dsl not installed
4. **The `attention()` function** falls back to flash_attn when sage not available

### Working Commands
```bash
# Check attention backends
uv run python -c "from scope.core.pipelines.wan2_1.modules.attention import *; print(f'FA4: {FLASH_ATTN_4_AVAILABLE}')"

# Test without torch.compile
TORCH_COMPILE_DISABLE=1 uv run render_timeline timeline.json out.mp4 --preset standard

# Uninstall FA4 deps (if needed)
uv pip uninstall nvidia-cutlass-dsl cuda-python cuda-bindings cuda-pathfinder
```

---

## POTENTIAL FIX

Modify `causal_model.py` to conditionally compile flex_attention:
```python
import torch

def is_sm103():
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        return major == 10 and minor == 3
    return False

from torch.nn.attention.flex_attention import flex_attention as _flex_attention

# Don't compile on SM103 (B300) - inductor doesn't support it yet
if not is_sm103():
    flex_attention = torch.compile(
        _flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs"
    )
else:
    flex_attention = _flex_attention  # Use uncompiled version
```

OR use environment variable:
```python
import os
if os.getenv("DISABLE_FLEX_ATTENTION_COMPILE", "0") == "0":
    flex_attention = torch.compile(...)
else:
    flex_attention = _flex_attention
```

**CAVEAT**: When we tested with `TORCH_COMPILE_DISABLE=1` (global), we got:
```
'tuple' object has no attribute 'unflatten'
```
So uncompiled flex_attention may also have issues - need to debug why `self_attn` returns tuple.

---

## Next Steps

1. **Try the fix above** - modify causal_model.py to skip compile on SM103

2. **Test uncompiled flex_attention** - might have the tuple issue still

3. **Check if H100 works**: Compare behavior on different GPU

4. **Debug the tuple return**: If uncompiled fails, trace where tuple comes from

---

## 2025-12-23 Analysis: FA4 + score_mod SM103 Compatibility

### Overview
Analysis of what's needed to run the FA4/CUTE `score_mod` path (Kernel B) on B300 (SM103).

### Current State on B200 (SM100)
- **Kernel B with FA4/CUTE score_mod**: 0.54ms (1.89x faster than Triton)
- **Kernel B with Triton**: 1.02ms
- Backend selection: `SCOPE_KV_BIAS_BACKEND=fa4|triton|flex`

### CUTE Files Touched by `_flash_attn_fwd` with `score_mod`

The FA4 score_mod path invokes these files when `compute_capability == 10`:

```
flash_attn/cute/
├── interface.py           # Entry point - _flash_attn_fwd()
├── flash_fwd_sm100.py     # FlashAttentionForwardSm100 class
│   ├── imports tcgen05 (mma, copy)
│   ├── imports cpasync (copy)
│   └── uses apply_score_mod_inner from softmax.py
├── softmax.py             # SoftmaxSm100, apply_score_mod_inner()
│   └── score_mod is called here with (batch_idx, head_idx, q_idx, kv_idx)
├── blackwell_helpers.py   # tcgen05 gemm operations
├── mma_sm100_desc.py      # MMA instruction descriptors
└── pipeline.py            # Pipeline utilities
```

### nvidia-cutlass-dsl Files Requiring SM103 Patches

The existing `scripts/patch_cutlass_sm103.py` patches:

| File | Purpose | Patch Type |
|------|---------|------------|
| `cute/arch/mbar.py` | mbarrier operations | Add 103 to `_VALID_ARCHS` |
| `cute/arch/elect.py` | elect operations | Add 103 to `_VALID_ARCHS` |
| `cute/nvgpu/tcgen05/mma.py` | TensorCore Gen 05 MMA | Add 103 to `_VALID_ARCHS` |
| `cute/nvgpu/tcgen05/copy.py` | TensorCore Gen 05 copy | Add 103 to `_VALID_ARCHS` |
| `cute/nvgpu/cpasync/copy.py` | Async copy | Add 103 to `_VALID_ARCHS` |

**Verdict**: The patch covers ALL tcgen05/cpasync modules used by the score_mod path.

### New SM103 Compatibility Layer

Created: `src/scope/core/compat/sm103.py`

Features:
1. **Auto-detection**: `is_sm100()`, `is_sm103()`, `is_blackwell()`
2. **Auto-patching**: `patch_cutlass_for_sm103()` - patches cutlass-dsl on import
3. **Backend recommendation**: `get_recommended_backend()` - returns safe defaults
4. **Capability info**: `get_capability_info()` - debug helper

Usage:
```python
from scope.core.compat.sm103 import (
    is_sm103, patch_cutlass_for_sm103, get_recommended_backend
)

# Auto-detect and get safe backend
backend = get_recommended_backend()  # "triton" on SM103, "fa4" or "triton" on SM100
```

### SM103 Adaptation Risks

| Component | Risk Level | Notes |
|-----------|------------|-------|
| Triton Kernel B | **Low** | JIT compiles to target GPU, may need retuning |
| Triton RoPE v2 | **Low** | Same as above |
| FA4/CUTE score_mod | **Medium** | Needs patching, untested on real SM103 |
| flex_attention (compiled) | **High** | torch._inductor lacks SM103 kernels |
| FA4 → Triton fallback | **Low** | Already implemented in causal_model.py |

### Recommended Testing on B300

```bash
# 1. Check device capability
python -c "import torch; print(torch.cuda.get_device_capability())"
# Expected: (10, 3)

# 2. Check capability info from new compat module
python -c "from scope.core.compat.sm103 import get_capability_info; import pprint; pprint.pprint(get_capability_info())"

# 3. Test Triton Kernel B (should work)
SCOPE_KV_BIAS_BACKEND=triton uv run python scripts/triton_sdpa.py --kernel-b

# 4. Test FA4 score_mod (after auto-patching)
SCOPE_KV_BIAS_BACKEND=fa4 uv run python -c "
from scope.core.compat.sm103 import patch_cutlass_for_sm103
patch_cutlass_for_sm103()
from flash_attn.cute.interface import _flash_attn_fwd
import torch
q = torch.randn(1, 64, 8, 128, dtype=torch.bfloat16, device='cuda')
k = torch.randn(1, 256, 8, 128, dtype=torch.bfloat16, device='cuda')
v = torch.randn(1, 256, 8, 128, dtype=torch.bfloat16, device='cuda')
out, _ = _flash_attn_fwd(q, k, v, causal=False)
print('FA4 basic test passed:', out.shape)
"

# 5. Force PTX JIT to detect missing PTX coverage
CUDA_FORCE_PTX_JIT=1 SCOPE_KV_BIAS_BACKEND=fa4 uv run python -c "
from flash_attn.cute.interface import _flash_attn_fwd
print('PTX JIT test passed')
"
```

### Questions to Resolve

1. **Does SM103 share the same tcgen05 instruction set as SM100?**
   - Likely yes (same Blackwell family), but untested
   - PTX should JIT to SM103 if cubin doesn't match

2. **Will Triton kernels need retuning for B300?**
   - B300 may have different memory bandwidth, L2 size, SM count
   - Block sizes tuned for B200 may not be optimal

3. **Can we use FA4 as primary on SM103?**
   - Only after validation on real B300 hardware
   - Default should remain Triton until confirmed

---

## 2025-12-23: BREAKTHROUGH - Triton Works on B300!

### The Fix: CUDA 12.9 ptxas

**Root cause**: CUDA 12.8's ptxas doesn't support `sm_103a`. Triton 3.4.0 bundles an older ptxas.

**Solution**: Install CUDA 12.9 and point Triton to its ptxas:

```bash
# Install CUDA 12.9 (has SM103 support)
sudo apt install cuda-toolkit-12-9

# Set the env var (added to .venv/bin/activate)
export TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas
```

### Verified Working on B300 (SM103)

```
GPU: NVIDIA B300 SXM6 AC
Compute Capability: (10, 3) = SM103

Kernel B Correctness Tests: ALL PASS
- Test 1 (B=1, H=1, Lq=64, Lk=128, D=16): PASS (max error: 0.0039)
- Test 2 (B=1, H=4, Lq=128, Lk=256, D=64): PASS (max error: 0.0039)
- Test 3 (B=1, H=16, Lq=4680, Lk=9360, D=128): PASS (max error: 0.0020)

Kernel B Benchmark (4680x9360, H=16, D=128):
- flex_attention: 1.095 ms
- Triton Kernel B: 0.977 ms (10.8% faster)
```

### B300 Environment Setup

```bash
# In .venv/bin/activate, add:
export TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas
```

### Updated Status

| Component | B200 (SM100) | B300 (SM103) |
|-----------|--------------|--------------|
| Triton Kernel B | ✅ Works | ✅ Works (with CUDA 12.9 ptxas) |
| flex_attention | ✅ Works | ✅ Works |
| FA4/CUTE score_mod | ✅ Works | ❓ Needs cutlass-dsl |
| RoPE fused | ✅ Works | ✅ Works (0.029 ms) |

### RoPE Benchmark on B300

```
shape: B=1 L=4680 H=16 D=128 dtype=bf16
PyTorch (prefix only): 0.091 ms
Triton rotary (prefix only): 0.067 ms
rope_apply (end-to-end): 0.030 ms
triton_rope_fused_3way (direct): 0.029 ms
Correctness: max_err=0.031250 mean_err=0.000595
```

### What's Left for Full B300 Support

1. ~~**Test RoPE v2 kernel** on B300~~ ✅ DONE
2. **Full pipeline test** (`render_timeline`) on B300
3. **FA4/CUTE** requires separate venv with nvidia-cutlass-dsl (conflicts with pip flash-attn)
4. **Performance tuning** - block sizes may benefit from B300-specific tuning

---

## Git Commits
- `701afd6` - Add Python 3.12 support and FA4 (CUTE) dependencies for Blackwell
- `28aa3d9` - Update .gitignore with video files and cloned repos
- `b76bc4b` - Fix fa4 dependency group to use Python version markers
- `858ddf2` - Add warning about nvidia-cutlass-dsl conflict with PyTorch inductor
