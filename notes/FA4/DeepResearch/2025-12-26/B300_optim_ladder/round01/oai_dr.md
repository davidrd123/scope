# PyTorch 2.9 Optimization on NVIDIA Blackwell B300: Critical Issues and Solutions

PyTorch 2.9 users targeting NVIDIA Blackwell B300 (SM103) GPUs face three critical challenge areas: **CUDAGraphs output overwriting errors with torch.compile**, **Float8Tensor dispatch limitations in TorchAO**, and **severe Conv3d/BF16 performance regressions with cuDNN**. This report curates authoritative links, issue IDs, and excerpts for each area with actionable workarounds.

---

## AREA 1: torch.compile modes and CUDAGraphs issues

The `reduce-overhead` mode enables CUDAGraph Trees for reduced Python overhead, while `max-autotune-no-cudagraphs` provides autotuning without CUDA graphs—critical for dynamic shapes or when CUDAGraph errors persist.

### The "CUDAGraphs output overwritten" error

This runtime error occurs when outputs from prior CUDAGraph invocations are accessed after being overwritten. The core issue stems from CUDAGraph Trees reusing memory across graph captures.

**Key GitHub Issues:**

| Issue | Title | Status | URL |
|-------|-------|--------|-----|
| #144961 | CUDAGraph outputs overwritten by subsequent run | Open | https://github.com/pytorch/pytorch/issues/144961 |
| #148439 | [cudagraph_trees] RuntimeError during Megatron-LM training | Closed | https://github.com/pytorch/pytorch/issues/148439 |
| #158551 | CUDAGraphs RuntimeError despite clone and cudagraph_mark_step_begin | Open | https://github.com/pytorch/pytorch/issues/158551 |
| #141171 | cudagraph_trees error message suggestion doesn't always work | Open | https://github.com/pytorch/pytorch/issues/141171 |
| #146569 | torch.compile with fullgraph=True causes overwritten variable error | Open | https://github.com/pytorch/pytorch/issues/146569 |

**Error Pattern (from Issue #144961):**
```python
@torch.compile(mode="reduce-overhead")
def my_model(x):
    y = torch.matmul(x, x)
    return y

x = torch.randn(10, 10, device="cuda")
y1 = my_model(x)
y2 = my_model(x)
print(y1)  # RuntimeError: Error: accessing tensor output of CUDAGraphs 
           # that has been overwritten by a subsequent run.
```

### Recommended workarounds for CUDAGraph errors

The official PyTorch 2.9 CUDAGraph Trees documentation (https://docs.pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html) recommends these solutions:

```python
# Workaround 1: Mark step boundaries explicitly
for iteration in range(num_iters):
    torch.compiler.cudagraph_mark_step_begin()  # Mark new iteration
    output = compiled_model(input)

# Workaround 2: Clone outputs outside torch.compile
y1 = my_model(x).clone()  # Clone before subsequent calls

# Workaround 3: Enable input mutation support
torch._inductor.config.triton.cudagraph_support_input_mutation = True

# Workaround 4: Disable CUDAGraphs entirely
model = torch.compile(model, mode="max-autotune-no-cudagraphs")
# OR via environment variable:
# TORCHINDUCTOR_CUDAGRAPHS=0
```

**Key Insight from Issue #121861 (PR #123231):** Input mutations fall into four categories: (a) no mutation, (b) mutation on parameters/buffers, (c) mutation on cudagraph-recorded tensors, (d) mutation on non-cudagraph tensors. Only types (a), (b), (c) support CUDAGraph—type (d) will silently fail or error.

### Blackwell SM103 tracking and torch.compile status

**Main Tracking Issue #145949:** https://github.com/pytorch/pytorch/issues/145949
- Labels: `module: build`, `module: cuda`
- Status: Open (active development)

**Key Blackwell PRs:**
- PR #145436: Full Family Blackwell Support codegen
- PR #141724: Add support for blackwell codegen  
- PR #145602: Add Blackwell support to SDPA
- PR #145746: 128-bit vectorization for Blackwell

**Issue #159779 (CUDA 13.0/SM103 Support):** "CUDA 13.0 supports all NVIDIA architectures from Turing through Blackwell. Featuring **sm_110** (Jetson Thor), **sm_103** (B300/GB300), **sm_121** (DGX SPARK/DIGITS)."

**Critical Note:** CUDA 12.8 first introduced sm_12x and sm_10x arch support. PyTorch 2.9+cu130 requires building from source with appropriate `TORCH_CUDA_ARCH_LIST` for full SM103 optimization.

---

## AREA 2: TorchAO float8 and PyTorch 2.9 status

### Version compatibility matrix

**Source: pytorch/ao#2919** (https://github.com/pytorch/ao/issues/2919)

| TorchAO Version | Built With PyTorch | Python API Support |
|-----------------|-------------------|-------------------|
| **0.15.0dev (nightly)** | 2.10.0dev | 2.10.0, **2.9.0**, 2.8.0 |
| **0.14.1** | **2.9.0** | **2.9.0**, 2.8.0, 2.7.1 |
| 0.13.0 | 2.8.0 | 2.8.0, 2.7.1, 2.6.0 |

**Recommendation:** Use **TorchAO 0.14.1** for PyTorch 2.9.0 compatibility.

### Float8Tensor dispatch architecture and unimplemented ops

Float8Tensor uses `__torch_dispatch__` to intercept aten ops and route to optimized kernels. The dispatch mechanism checks `FLOAT8_OPS_TABLE` for supported operations.

**Dispatch Pattern (from pytorch/ao#391):**
```python
# Float8Tensor dispatch in float8_tensor.py
if func in FLOAT8_OPS_TABLE:
    return FLOAT8_OPS_TABLE[func](func, args, kwargs)
raise NotImplementedError(f"attempting to run {func}, this is not supported")
```

**Known Dispatch Error Patterns:**

| Issue | Error Type | URL |
|-------|-----------|-----|
| pytorch/ao#565 | `NotImplementedError: attempting to run aten.addmm, not supported` | https://github.com/pytorch/ao/issues/565 |
| pytorch/ao#890 | `aten.permute.default unimplemented` | https://github.com/pytorch/ao/issues/890 |
| ComfyUI#59 | `aten._has_compatible_shallow_copy_type unimplemented` | https://github.com/kijai/ComfyUI-HunyuanVideoWrapper/issues/59 |

**Note on aten.as_strided:** While Float8Tensor does not natively implement `aten.as_strided`, the related NF4Tensor (torchao/dtypes/nf4tensor.py) shows the restricted implementation pattern:
```python
f"aten.as_strided(NF4Tensor) only support continuous stride={make_contiguous_strides_for(size)} but got stride={stride}"
```

Ops not in `FLOAT8_OPS_TABLE` will raise `NotImplementedError`. The workaround is to ensure all operations in the model are supported or to dequantize before unsupported ops.

### quantize_ → Float8Tensor workflow under torch.compile

**Source: pytorch/ao#574 (RFC: Float8 Inference)** - https://github.com/pytorch/ao/issues/574

```python
from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig, PerRow

# Quantization workflow
quantize_(model, Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()))

# CRITICAL: torch.compile required for competitive performance
model = torch.compile(model)
```

**Performance Without Compile (pytorch/ao#685):**

| Mode | Time | Memory |
|------|------|--------|
| PyTorch FP16 | 3.14ms | 664MB |
| TorchAO FP8 (compile=False) | **7.17ms** | **1.2GB** |
| TorchAO FP8 (compile=True) | **2.61ms** | 724MB |

**Key Quote:** "We heavily rely on the compile stack to generate efficient and fused casting code. We require torch.compile for competitive performance."

### Version migration: Float8Tensor v1 → v2

**Issue pytorch/ao#2649** documents the migration from AffineQuantizedTensor (v1) to Float8Tensor (v2):

```python
# Deprecation warning in TorchAO 0.13.0+
warnings.warn(
    "Config Deprecation: version 1 of Float8WeightOnlyConfig is deprecated, "
    "please use version 2, see https://github.com/pytorch/ao/issues/2649"
)
```

**Use version=2 (default) for all new Float8 configurations.**

---

## AREA 3: Conv3d/cuDNN performance on Blackwell (inference focus)

### Critical PyTorch 2.9 regressions affecting all recent GPUs

**SEVERE WARNING:** PyTorch 2.9.x has documented **4x-40x performance regressions** and **OOM errors** for Conv3d with BF16/FP16 that affect H100, H200, and by extension Blackwell GPUs.

| Issue | Problem | Impact | URL |
|-------|---------|--------|-----|
| #168167 | Conv3D/BF16 40,000x slower (40s vs 1ms) | Severe | https://github.com/pytorch/pytorch/issues/168167 |
| #166643 | BF16 conv3d uses 3x more memory than FP32 | Memory | https://github.com/pytorch/pytorch/issues/166643 |
| #166122 | 4x AMP performance regression | Performance | https://github.com/pytorch/pytorch/issues/166122 |
| #166790 | cuDNN selects algorithm requiring 27GB workspace | OOM | https://github.com/pytorch/pytorch/issues/166790 |

**Excerpt from Issue #168167:** "When running a simple nn.Conv3d projection benchmark on H200 with torch.bfloat16, the forward pass becomes **~40 seconds in PyTorch 2.9.1**, while the exact same code finishes in **~1 ms on PyTorch 2.8**."

**Root Cause:** cuDNN algorithm selection in PyTorch 2.9 with cuDNN 9.10.2 picks suboptimal kernels for BF16 Conv3d.

### When cuDNN falls back to slow_conv_dilated3d

cuDNN fallback to the slow native implementation occurs under these conditions:

1. **Dilation > 1**: cuDNN lacks optimized dilated 3D convolution
2. **Grouped/Depthwise Conv3d**: Issue #37406 documents slow performance
3. **cudnn.deterministic=True with dilation**: Issue #28777 shows 10x-80x slowdown
4. **Unsupported kernel/stride combinations**

**Detection via profiler (Issue #32370):**
```
slow_conv_dilated3d    67.05%    93.521us  # Indicates cuDNN fallback
```

### Recommended memory layouts: channels_last_3d (NDHWC)

**PR #48430** enabled channels_last_3d support requiring cuDNN ≥ 8.0.5.

**PyTorch Documentation** (https://pytorch.org/docs/stable/generated/torch.nn.utils.convert_conv3d_weight_memory_format.html):
> "NDHWC (channels_last_3d) conversion for convolution in cuDNN is beneficial to run convolution in NDHWC, even in cases where we have to apply permutation to input tensors."

**Inference Configuration:**
```python
import torch
import torch.nn as nn

# Convert model to channels_last_3d for optimal cuDNN performance
model.eval()
model = model.to(memory_format=torch.channels_last_3d)
# OR more targeted:
model = nn.utils.convert_conv3d_weight_memory_format(model, torch.channels_last_3d)

# Convert input as well
input = input.to(memory_format=torch.channels_last_3d)

with torch.no_grad():
    output = model(input)
```

### Environment variables and cuDNN optimization flags

**Critical Settings for PyTorch 2.9 Conv3d:**

```python
import os
import torch

# Limit cuDNN workspace to prevent OOM (cuDNN 8.0.5+)
os.environ['CUDNN_CONV_WSCAP_DBG'] = str(8 * 1024**3)  # 8GB limit

# PyTorch backend settings for inference
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True       # Enable for fixed input sizes
torch.backends.cudnn.benchmark_limit = 4    # Limit algorithm search attempts
torch.backends.cudnn.allow_tf32 = True      # Enable TF32 for Ampere+
torch.backends.cuda.matmul.allow_tf32 = True
```

**Issue #49207** documents `CUDNN_CONV_WSCAP_DBG`: "Starting from v8.0.5, cuDNN allows specifying the maximum workspace size by using CUDNN_CONV_WSCAP_DBG environmental variable."

### Blackwell-specific cuDNN considerations

**NVIDIA Blackwell RTX Software Migration Guide** (https://forums.developer.nvidia.com/t/software-migration-guide-for-nvidia-blackwell-rtx-gpus-a-guide-to-cuda-12-8-pytorch-tensorrt-and-llama-cpp/321330):
- Driver R570+ required for Blackwell
- CUDA 12.8 is first version with native Blackwell support (sm_100, sm_120)
- cuDNN 9.x recommended

**cuDNN Kernel Selection Issue** (https://forums.developer.nvidia.com/t/cudnn-only-selects-sm80-kernels-on-blackwell-devices/338923):
> "Both PyTorch and Jax/XLA (and thus probably cuDNN) only select sm80 conv kernels on a 5060ti even though sm100 kernels should be available."

**NVIDIA cuDNN Bug Report** (https://forums.developer.nvidia.com/t/cudnn-bug-report-conv3d-performance-regression-with-bfloat16-float16-on-h100/355210) confirms this is a known upstream cuDNN issue affecting H100/H200 and Blackwell.

---

## Summary: Recommended configuration for PyTorch 2.9 on Blackwell B300

```python
import os
import torch
import torch.nn as nn

# Environment setup (before torch import in production)
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
os.environ['CUDNN_CONV_WSCAP_DBG'] = str(8 * 1024**3)  # Limit cuDNN workspace

# PyTorch backend configuration
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True  # For fixed input sizes
torch.backends.cudnn.benchmark_limit = 4
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

# For torch.compile with CUDAGraphs
torch._inductor.config.triton.cudagraph_support_input_mutation = True

# Model setup for inference
model.eval()
model = model.to(memory_format=torch.channels_last_3d)  # For Conv3d
model = torch.compile(model, mode="max-autotune-no-cudagraphs")  # Avoid CUDAGraph issues

# TorchAO Float8 (requires torch.compile)
from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig
quantize_(model, Float8DynamicActivationFloat8WeightConfig())
model = torch.compile(model)  # Required for Float8 performance
```

**Version Requirements:**

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| PyTorch | 2.7+ | **2.8.x** (avoid 2.9.x for Conv3d/BF16) |
| TorchAO | 0.13.0 | **0.14.1** (for PyTorch 2.9) |
| cuDNN | 8.0.5+ | **9.10+** |
| CUDA | 12.8 | **12.8+** (native Blackwell) |
| Driver | R570+ | Latest |

**Critical Recommendation:** For Conv3d/BF16 inference workloads, consider using **PyTorch 2.8.x** until the cuDNN regressions in 2.9.x are resolved upstream.
