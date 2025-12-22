# B300 (SM103) Investigation Notes

## Environment
- **GPU**: NVIDIA B300 SXM6 AC → now testing on **B200**
- **Compute Capability**: SM103 (10.3) for B300, **SM100 (10.0) for B200**
- **PyTorch**: 2.8.0+cu128
- **CUDA**: 12.8
- **Python**: 3.12.3

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

## Issue 2: nvidia-cutlass-dsl Conflicts with PyTorch Inductor - BLOCKER

### Problem
When FA4 deps are installed, `nvidia-cutlass-dsl` provides a `cutlass` module that shadows PyTorch's internal cutlass utilities in `torch._inductor`.

### Symptom
```
NoValidChoicesError: target: flex_attention
...
AttributeError: module 'cutlass' has no attribute 'CACHE_FILE'
```

### Workaround
Uninstall FA4 deps before running normal pipelines:
```bash
uv pip uninstall nvidia-cutlass-dsl cuda-python cuda-bindings cuda-pathfinder
```

### Status
FA4 can only be used in isolation (benchmarks). Cannot coexist with render_timeline.

---

## Issue 3: flex_attention on B300 - UNDER INVESTIGATION

### Problem
Even without FA4 deps, `render_timeline` fails on B300 with `NoValidChoicesError` for `flex_attention`.

### Root Cause Hypothesis
PyTorch 2.8's inductor doesn't have complete SM103 kernel support for flex_attention's "max-autotune-no-cudagraphs" compilation mode.

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

## Git Commits
- `701afd6` - Add Python 3.12 support and FA4 (CUTE) dependencies for Blackwell
- `28aa3d9` - Update .gitignore with video files and cloned repos
- `b76bc4b` - Fix fa4 dependency group to use Python version markers
- `858ddf2` - Add warning about nvidia-cutlass-dsl conflict with PyTorch inductor
