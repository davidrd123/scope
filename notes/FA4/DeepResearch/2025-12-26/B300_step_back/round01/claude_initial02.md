# B300 (SM103) GPU optimization documentation gaps: Primary source findings

The B300/GB300 (SM103, compute capability 10.3) faces several critical software compatibility gaps as of December 2025. **Triton 3.5.1 fixes the most severe blocking issue**, while cuDNN 9.15+ addresses Conv3d regressions. SM103 requires CUDA 12.9+ and often PyTorch nightly for full functionality. Below are the authoritative sources and actionable recommendations for each documentation gap.

---

## Priority 1: Triton tcgen05 LLVM failure is fixed in 3.5.1

The "LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.tcgen05.wait.st" failure was a critical SM103 regression in Triton 3.5.0, now **resolved in Triton 3.5.1** (released November 12, 2025).

### Authoritative sources

| Resource | URL |
|----------|-----|
| **Primary issue** | https://github.com/triton-lang/triton/issues/8473 |
| **Duplicate with error trace** | https://github.com/triton-lang/triton/issues/8481 |
| **Fix PR** | https://github.com/triton-lang/triton/pull/8045 |
| **LLVM upstream fix** | https://github.com/llvm/llvm-project/commit/8a2dd2b |
| **Release 3.5.1 notes** | https://github.com/triton-lang/triton/releases/tag/v3.5.1 |

### Key excerpts

From issue #8473 (opened October 17, 2025):
> "The release 3.4 did support sm103 as long as ptxas shipped with CUDA 12.9 or newer is used. After 3.4, #7725 broke sm103 support... Unfortunately, the release 3.5 was cut off between these commits."

From PR #8045:
> "Some changes in #7725 that introduced more uses of NVVM ops broke sm_103 support... Until tcgen05 ops are properly supported by NVPTX, I propose to partially revert those changes."

**Root cause**: LLVM's `hasTcgen05Instructions()` method in NVPTXSubtarget.h only returned `true` for SM100/SM101, excluding SM103.

### Version compatibility matrix

| Triton Version | SM103 Status |
|----------------|--------------|
| 3.4.0 | ✅ Works with CUDA 12.9+ ptxas |
| **3.5.0** | ❌ **BROKEN** - tcgen05 LLVM error |
| **3.5.1** | ✅ **FIXED** |
| main branch | ✅ Fixed via LLVM bump |

### PyTorch Inductor / flex_attention issues

- **Issue #160385**: FlexAttention backward pass TTGIR errors on B200 (related Blackwell architecture)
- **Issue #145949**: PyTorch Blackwell tracking umbrella issue
- **Issue #146518**: Triton upgrade tracking for Blackwell support

### Actionable recommendations

```bash
# Primary fix: upgrade Triton
pip install triton==3.5.1

# Alternative: use PyTorch nightly (includes fixed Triton)
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu130

# If stuck on older Triton, use system ptxas
export TRITON_PTXAS_PATH=$(which ptxas)
```

---

## Priority 2: cuDNN 9.15+ fixes Conv3d regression

No explicit "Conv3d" optimizations are documented for Blackwell in cuDNN release notes. However, **cuDNN 9.15.0+ resolves the BF16/FP16 Conv3d memory regression** reported in PyTorch 2.9.0.

### Authoritative sources

| Resource | URL |
|----------|-----|
| **cuDNN Release Notes** | https://docs.nvidia.com/deeplearning/cudnn/backend/latest/release-notes.html |
| **PyTorch 2.9.1 Release** | https://github.com/pytorch/pytorch/releases/tag/v2.9.1 |
| **PyTorch Conv3d Issue** | https://github.com/pytorch/pytorch/issues/166643 |
| **cuDNN Archive** | https://developer.nvidia.com/cudnn-archive |

### Key findings by cuDNN version

**cuDNN 9.13.0**:
> "Performance of **grouped convolution** has been improved on **Blackwell-architecture data center GPUs**"

**cuDNN 9.15.0**:
> "The performance of matrix multiplication and convolution operations within the runtime fusion engines has been enhanced on Blackwell-architecture GPUs"

**cuDNN 9.17.0**:
> "Performance for FP16 and BF16 has been further optimized on Blackwell-architecture GPUs (compute capability 10.0 and 12.0)"
> "Performance for FP8 matmul and convolutions has been significantly optimized on the GeForce RTX 5090 GPU"

### BF16/FP16 Conv3d regression fix

PyTorch issue #166643 documented that `F.conv3d` with bfloat16 inputs consumed **~3x more memory** than float32 (8.5 GB vs 3.4 GB on H100) in PyTorch 2.9.0.

**From PyTorch 2.9.1 release notes**:
> "If you are impacted please install **nvidia-cudnn package version 9.15+ from pypi**."

### Actionable recommendations

```bash
# Install cuDNN 9.15+ to fix Conv3d regression
pip install nvidia-cudnn-cu12>=9.15

# For latest Blackwell optimizations
pip install nvidia-cudnn-cu12>=9.17
```

**Note**: SM103 (compute capability 10.3) is not explicitly mentioned in cuDNN release notes. Optimizations reference "Blackwell-architecture data center GPUs" (SM100) and compute capability 12.0 (SM120 consumer GPUs).

---

## Priority 3: CUDA 12.9 first added sm_103 ptxas support

**CUDA Toolkit 12.9** is the first version with official ptxas support for sm_103, including all three architecture variants.

### Authoritative sources

| Resource | URL |
|----------|-----|
| **CUDA 12.9 Release Notes** | https://docs.nvidia.com/cuda/archive/12.9.0/cuda-toolkit-release-notes/index.html |
| **CUDA 13.0 Release Notes** | https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html |
| **CUDA 13.1 Release Notes** | https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html |
| **Family-specific Architecture Blog** | https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features |

### Architecture variant distinctions

| Target | Meaning | Compatibility |
|--------|---------|---------------|
| **sm_103** | Standard target | PTX forward-compatible to later CCs; cubin compatible within major CC |
| **sm_103f** | Family-specific | Compatible with all 10.x GPUs (SM100, SM103) |
| **sm_103a** | Architecture-specific | **Only runs on SM103** - no forward compatibility |

From the official NVIDIA blog:
> "Family-specific features are similar to architecture-specific features, except that they are supported by devices of more than one minor compute capability. All devices within a family share the same major compute capability version."

### CUDA version timeline

| CUDA Version | sm_103 Status | Key Changes |
|--------------|---------------|-------------|
| Pre-12.8 | Not available | — |
| 12.8 | Experimental (libraries only) | Some MathDx libraries added experimental support |
| **12.9** | **First official support** | Added sm_103, sm_103f, sm_103a |
| 12.9.1 | Stable | Fixed address calculation miscompilation |
| 13.0 | Full support | Thor renamed sm_101→sm_110 |
| 13.1 | Optimized | Removed kernel loading overhead for CC 10.3 |

### Known issues

**MMA Miscompilation (CUDA 12.8-12.9)**:
> "We observed miscompilation issues with kernels that use Ampere-style MMA (e.g., mma.m16n8k16 for FP16)... affecting SM80, SM90, and SM100 architectures."
> **Workaround**: Compile with `-Xptxas -O0`

**cuBLAS 13.1 bug fix**:
> "Removed unnecessary overhead related to loading kernels on GPUs with compute capability 10.3"

### Actionable recommendations

- Use **CUDA 12.9.1+** for stable SM103 support
- Use **CUDA 13.1** for optimal performance (kernel loading overhead fixed)
- For architecture-specific Tensor Core features, use `-arch=sm_103a`
- For cross-Blackwell compatibility, use `-gencode arch=compute_100f,code=sm_100`

---

## Priority 4: TorchAO 0.14.1 is the correct version for torch 2.9.0

**torchao 0.15.0 is a nightly/dev version** built against torch 2.10.0dev. For stable PyTorch 2.9.0, use **torchao 0.14.1**.

### Authoritative sources

| Resource | URL |
|----------|-----|
| **Compatibility Matrix** | https://github.com/pytorch/ao/issues/2919 |
| **Float8 FSDP Utils** | https://docs.pytorch.org/ao/stable/_modules/torchao/float8/fsdp_utils.html |
| **Releases** | https://github.com/pytorch/ao/releases |
| **MXFP8 Training Blog** | https://pytorch.org/blog/accelerating-2k-scale-pre-training-up-to-1-28x-with-torchao-mxfp8-and-torchtitan-on-crusoe-b200-cluster/ |

### Compatibility matrix

| torchao version | torch version (binary built with) | torch version (Python API) |
|----------------|-----------------------------------|---------------------------|
| 0.15.0dev (nightly) | 2.10.0dev | 2.10.0, 2.9.0, 2.8.0 |
| **0.14.1** | **2.9.0** | **2.9.0**, 2.8.0, 2.7.1 |
| 0.13.0 | 2.8.0 | 2.8.0, 2.7.1, 2.6.0 |

### Float8Tensor and as_strided handling

Float8Tensor explicitly handles `aten.as_strided.default` in FSDP contexts. From `torchao/float8/fsdp_utils.py`:

```python
_ops_to_preserve_subclass = {
    torch.ops.aten.as_strided.default,  # Explicitly handled
    torch.ops.aten.view.default,
    torch.ops.aten.clone.default,
    # ... other ops
}
```

No specific Float8Tensor as_strided issues documented in GitHub issues. Quantized tensor subclasses generally require contiguous strides.

### FP8 on Blackwell status

| Feature | Status |
|---------|--------|
| **MXFP8 Training** | ✅ Production-ready (torchao 0.14.1+) |
| **NVFP4** | ⚠️ Prototype only |
| **Hardware support** | SM100 (B200), SM103 (B300), SM120 (RTX 50xx) |

Performance on B200 clusters: **1.22x-1.28x speedup vs BF16** with MXFP8.

### Actionable recommendations

```bash
# For PyTorch 2.9.0+cu130
pip install torchao==0.14.1

# MXFP8 quantization example
from torchao.quantization import quantize_
from torchao.prototype.mx_formats import MXFPInferenceConfig
model = quantize_(model, MXFPInferenceConfig(block_size=32))
```

Requirements: **Python 3.10+** (Python 3.9 dropped in PyTorch 2.9), **CUDA 12.8+** for full Blackwell support.

---

## Priority 5: cuDNN SDPA backend supports Blackwell via PyTorch 2.7+

PyTorch 2.7 added Blackwell SDPA support with cuDNN 9.7.0+. The backend is enabled via environment variable or context manager.

### Authoritative sources

| Resource | URL |
|----------|-----|
| **PyTorch Blackwell SDPA PR** | https://github.com/pytorch/pytorch/pull/145602 |
| **Blackwell Tracking Issue** | https://github.com/pytorch/pytorch/issues/145949 |
| **torch.backends docs** | https://docs.pytorch.org/docs/stable/backends.html |
| **SDPBackend enum** | https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.SDPBackend.html |
| **cuDNN SDPA Blog** | https://developer.nvidia.com/blog/accelerating-transformers-with-nvidia-cudnn-9 |

### How to enable cuDNN SDPA

**Environment variable**:
```bash
export TORCH_CUDNN_SDPA_ENABLED=1
```

**Python API**:
```python
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

# Global enable
torch.backends.cuda.enable_cudnn_sdp(True)

# Check availability
torch.backends.cuda.can_use_cudnn_attention(params, debug=True)

# Context manager for explicit backend selection
with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
    output = F.scaled_dot_product_attention(query, key, value)
```

### Performance vs FlashAttention

From cuDNN documentation:
- cuDNN SDPA up to **2x faster** than PyTorch eager in BF16
- cuDNN SDPA up to **3x faster** in FP8
- On H100: cuDNN provides up to **75% speedup over FlashAttentionV2**

**Blackwell-specific optimizations (cuDNN 9.10+)**:
- Arbitrary head dimension support for prefill/training
- Improved SDPA fprop/bprop for small problem sizes with causal masking
- Significantly improved **Paged Attention** performance

### Known issues

**Critical**: FP8 SDPA on SM100 Blackwell contains a **deadlock** that can cause kernel hang with large problem sizes or multiple simultaneous kernels. Fix planned for future cuDNN release.

### Architecture support matrix

| Architecture | Compute Capability | cuDNN SDPA | PyTorch SDPA |
|--------------|-------------------|------------|--------------|
| Hopper | SM90 | ✅ Default for H100+ | ✅ |
| Blackwell Datacenter | SM100, SM103 | ✅ cuDNN 9.5+ | ✅ PyTorch 2.7+ |
| Blackwell GeForce | SM120 | ✅ | ✅ PyTorch 2.7+ |

### Actionable recommendations

- Use **PyTorch 2.7+** with **CUDA 12.8+** for Blackwell SDPA
- Avoid FP8 SDPA on SM100 until deadlock fix is released
- Enable via `torch.backends.cuda.enable_cudnn_sdp(True)` or `TORCH_CUDNN_SDPA_ENABLED=1`

---

## Priority 6: ThunderKittens supports SM100 (B200), not SM103 (B300)

ThunderKittens has **explicit B200 (SM100) support** but **no documented SM103 (B300) support**. B300 may work with B200 kernels due to architecture similarity, but this is unverified.

### Authoritative sources

| Resource | URL |
|----------|-----|
| **Main Repository** | https://github.com/HazyResearch/ThunderKittens |
| **Blackwell Branch** | https://github.com/HazyResearch/ThunderKittens/tree/blackwell |
| **B200 Attention Kernel** | https://github.com/HazyResearch/ThunderKittens/blob/blackwell/kernels/attn/b200/b200.cu |
| **Hazy Research Blog** | https://hazyresearch.stanford.edu/blog/2025-03-15-tk-blackwell |

### Current support status

| GPU | Architecture | Status |
|-----|--------------|--------|
| B200 | SM100 | ✅ Officially supported with dedicated kernels |
| B300 | SM103 | ⚠️ **Not explicitly supported** |

From the Hazy Research blog (March 2025):
> "Thanks again to Together AI for collaborating with us and helping us get running with **NVIDIA B200's** to write these kernels"

All kernel directories are named `B200/` or `b200/` with no SM103 variants.

### Available Blackwell kernels

- **Attention Forward/Backward**: `kernels/attn/b200/b200.cu`
- **BF16 GEMM**: `kernels/matmul/B200/matmul.cu`
- **FP8 GEMM**: `kernels/matmul/FP8_B200/matmul.cu`

### Performance claims (B200)

- **Attention**: Near-cuDNN speeds, up to **2x faster than FlashAttention-3 on H100**
- **BF16 GEMM**: At or near cuBLAS speeds
- **FP8 GEMM**: At or near cuBLAS speeds

### Actionable recommendations

For B300 (SM103):
1. B200 kernels **may work** due to shared Blackwell architecture
2. Requires CUDA 13.0+ with SM103 support
3. Use PyTorch nightly to avoid PTXAS errors
4. May need recompilation with `-arch=sm_103a`
5. Open a GitHub issue requesting SM103 support if needed

---

## Priority 7: SageAttention supports Python 3.12, SM103 has toolchain issues

SageAttention officially supports **Python 3.9+** (including 3.12). **SM103 (B300) has known toolchain compatibility issues** requiring PyTorch nightly.

### Authoritative sources

| Resource | URL |
|----------|-----|
| **GitHub Repository** | https://github.com/thu-ml/SageAttention |
| **PyPI** | https://pypi.org/project/sageattention/ |
| **Windows Wheels** | https://github.com/woct0rdho/SageAttention/releases |
| **SageAttention3 Blackwell** | https://github.com/thu-ml/SageAttention/tree/main/sageattention3_blackwell |
| **B200 Support Request** | https://github.com/thu-ml/SageAttention/issues/322 |

### Python version support

- **Official requirement**: Python ≥3.9
- **Prebuilt Windows wheels** use Python Stable ABI (ABI3), supporting **Python 3.9 through 3.13**

### CUDA version requirements by architecture

| GPU Architecture | CUDA Required |
|------------------|---------------|
| Blackwell (RTX 50xx) / SageAttention2++ | **≥12.8** |
| Ada FP8 (RTX 40xx) | ≥12.4 |
| Hopper FP8 (H100) | ≥12.3 |
| Ampere (RTX 30xx, A100) | ≥12.0 |

### Blackwell architecture support

| Architecture | GPUs | SageAttention Status |
|--------------|------|---------------------|
| SM120 | RTX 5090, 5080, 5070 | ✅ Full support (SageAttention2++, SageAttention3) |
| SM100 | B200, GB200 | ❌ **Not yet supported** (Issues #237, #322 open) |
| SM103 | B300 | ⚠️ Toolchain issues with torch stable |

**SageAttention3** (FP4 kernels) currently only supports SM120 (RTX 5090), achieving **1038 TOPS** (~5x speedup over FlashAttention).

### Installation for Python 3.12 + CUDA 13.x

```bash
# Standard installation
pip install sageattention==2.2.0 --no-build-isolation

# Windows prebuilt wheels (from woct0rdho fork)
pip install sageattention-2.2.0+cu130torch2.9.0.post3-cp39-abi3-win_amd64.whl

# For SM103 (B300), use PyTorch nightly to avoid PTXAS errors
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu130
```

### Key GitHub issues

| Issue | Title | Status |
|-------|-------|--------|
| #237 | Support SageAttentionV3 on Other Blackwell GPUs (SM100/B200) | Open |
| #319 | Windows11 Python 3.12 CUDA 13.0 PyTorch 2.9.1 Compilation failed | Open |
| #322 | Support for B200? | Open |

### Actionable recommendations

- **Python 3.12**: ✅ Fully supported
- **SM103 (B300)**: Use PyTorch nightly; stable torch 2.9.1+cu130 has PTXAS issues
- **SM100 (B200/GB200)**: Not yet supported; monitor GitHub issues #237 and #322
- **SM120 (RTX 50xx)**: Full support with CUDA 12.8+

---

## Summary of actionable recommendations for B300 optimization

| Component | Action | Version |
|-----------|--------|---------|
| **Triton** | Upgrade to fix tcgen05 LLVM error | **3.5.1** |
| **cuDNN** | Install to fix Conv3d regression | **9.15+** |
| **CUDA Toolkit** | Required for SM103 ptxas support | **12.9.1+** (13.1 for optimal) |
| **torchao** | Use stable version for torch 2.9.0 | **0.14.1** |
| **PyTorch** | Use nightly for best SM103 support | **nightly/cu130** |
| **cuDNN SDPA** | Enable for optimized attention | `TORCH_CUDNN_SDPA_ENABLED=1` |
| **ThunderKittens** | B200 kernels may work; no official SM103 | Blackwell branch |
| **SageAttention** | Works with Python 3.12; SM103 needs nightly | **2.2.0** |

**Critical insight**: Many SM103-specific issues stem from the fact that SM103 support was added later than SM100. CUDA 12.9 first added SM103 support, but optimal functionality requires CUDA 13.1 and PyTorch nightly builds that include Triton 3.5.1+.
