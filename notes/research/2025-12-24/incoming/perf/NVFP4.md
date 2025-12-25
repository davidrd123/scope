# NVFP4 on NVIDIA Blackwell (SM100)
Practical adoption guide for coding agents

> **Status:** Speculative / Reference Material
> **Relevance to B300 work:** Low (current bottleneck is conv3d, not GEMMs)
> **When this becomes relevant:**
> - After VAE decode bottleneck is solved and GEMMs become limiting
> - For KV cache compression to reduce memory bandwidth
> - NVIDIA FP4 GEMM competition (running ~2 weeks from 2025-12-25)
>
> **Key insight:** NVFP4 does NOT help FA4 score_mod (confirmed in TL;DR below)

This document is meant to be dropped into the repo as a living reference for evaluating and adopting NVFP4 (NVIDIA microscaled FP4) in our Blackwell-targeted inference stack.

## TL;DR

- NVFP4 is a 4-bit E2M1 value format with microscaling.
- It uses a shared FP8 (E4M3) scale per 16-value micro-block, plus a per-tensor FP32 scalar. This dual scaling is a key reason it preserves accuracy better than simpler FP4 variants.  
- Blackwell 5th-gen Tensor Cores accelerate NVFP4 and can handle grouping, scaling, and 4-bit matrix ops in hardware.  
- NVFP4 is most valuable for GEMM-heavy parts of transformers (QKV projections, MLP, output projection) and possibly for KV cache storage and bandwidth if your runtime supports it.
- It is not a meaningful optimization lever for FA4 score_mod. Score_mod is mostly elementwise work and not a Tensor Core GEMM bottleneck.

## What NVFP4 is

### Numeric format and scaling
NVFP4 uses:
- Element format: FP4 E2M1 (1 sign, 2 exponent, 1 mantissa)  
- Microscaling: 1 shared FP8 E4M3 scaling factor per 16 values (micro-block)  
- Additional scaling: a second-level FP32 scalar applied per tensor  

NVIDIA describes this as a "two-level micro-block scaling strategy" where E4M3 scaling is applied to each 16-value micro-block, plus a per-tensor FP32 scalar. The goal is to reduce quantization error while keeping the memory and compute benefits of 4-bit.  

### Relation to MXFP4 and plain FP4
NVIDIA compares three 4-bit formats on Blackwell:
- Plain FP4 (E2M1) plus a software scaling factor  
- MXFP4 with a shared power-of-two scale per 32-value block  
- NVFP4 with a shared FP8 scale per 16-value block  

This smaller block size and higher precision scale encoding are central to NVFP4's accuracy story.

## Where NVFP4 helps in transformers

### Highest leverage targets
1) GEMMs
- QKV projection GEMMs
- MLP GEMMs
- Output projection GEMM
These are the largest Tensor Core consumers and often memory bandwidth limited at scale.

2) KV cache storage and bandwidth (decode path)
If your decode is KV bandwidth bound, storing K and V in NVFP4 can reduce memory footprint and memory traffic. Some runtimes already advertise NVFP4/BF16 KV cache support, which is a strong signal this is viable in production stacks.

### Low leverage targets
- Softmax, normalization, and similar numerically sensitive reductions
- Attention score_mod logic
Most of these are not dominated by large matmul throughput and often need higher precision for stability.

## Decision tree for adoption

Use this quick decision tree to decide what to prototype first.

### Step 0: Is the target GPU Blackwell datacenter class (compute capability 10.x)?
- If no: NVFP4 is a non-starter.
- If yes: continue.

### Step 1: What is our bottleneck?
A) GEMM-heavy (MLP and projections dominate)
- Start with weight-only or W4A8 style quantization paths that use NVFP4 GEMMs.

B) Decode is KV cache bandwidth bound
- Investigate NVFP4 KV cache support in your runtime.
- Prototype KV cache compression and attention kernels that can consume NVFP4 K/V.

C) Prefill dominated by attention compute
- NVFP4 may help indirectly by accelerating QKV and MLP, but attention itself usually wants higher precision for the softmax path.
- Consider whether the attention implementation supports low precision K/V without destabilizing logits.

### Step 2: Choose integration path based on desired effort

Path A: Runtime adoption (fastest path)
- TensorRT Model Optimizer or LLM Compressor for quantization
- Deploy with TensorRT-LLM, vLLM (early support), and watch for SGLang support

Path B: PyTorch prototyping path (fast iteration)
- Use FlashInfer NVFP4 quantization utilities for packing tensors and producing scale factors
- Use an existing backend (FlashInfer, vLLM integration) that can consume the packed format

Path C: Custom kernel path (maximum control, highest effort)
- CUTLASS SM100 blockscaled GEMMs
- Triton block scaled matmul tutorial as a reference implementation
- CuTe kernels only if we are ready to manage packing and scale layouts carefully

## Integration paths and what to evaluate

### Path A: TensorRT Model Optimizer + TensorRT-LLM (and vLLM)
NVIDIA recommends quantizing models to NVFP4 using TensorRT Model Optimizer or LLM Compressor, then deploying with TensorRT-LLM or vLLM (early NVFP4 support), with SGLang support upcoming.

Why this is attractive:
- It is the most "drop-in" way to test end-to-end wins.
- It includes attention and KV cache engineering that is hard to replicate quickly.

Agent tasks:
- Identify whether our model family and serving requirements are supported by the runtime.
- Measure throughput and latency for prefill and decode.
- Measure memory headroom and KV cache footprint.

### Path B: FlashInfer NVFP4 quantization utilities
FlashInfer exposes a direct API:

`flashinfer.fp4_quantization.nvfp4_quantize(a, a_global_sf, sfLayout=..., do_shuffle=False, sf_vec_size=16, enable_pdl=None)`

- Input `a` is fp16 or bf16 with shape [M, K]
- `a_global_sf` is a float32 scalar tensor of shape [1]
- `sf_vec_size` defaults to 16
- It returns a packed quantized tensor and a scale-factor tensor

Notes:
- The quantized tensor is packed with shape [M, K/2] with dtype FLOAT4_E2M1X2, meaning two FP4 values per byte.
- Some backends require scale-factor shuffling for tensor B scale factors (do_shuffle), which matters for interoperability.

Agent tasks:
- Prototype quantization for a single GEMM operand (for example MLP weights).
- Verify shape constraints. Expect K to be a multiple of sf_vec_size in many implementations.
- Ensure downstream kernel path can consume the packed tensor and scale factors.

### Path C: Custom kernels (CUTLASS, Triton, and CuTe)
This path is for when we need:
- custom fusion,
- custom layouts,
- or want to integrate NVFP4 into our own kernels.

#### CUTLASS
CUTLASS defines a blockscaled type:
- `nv_float4_t<float_e2m1_t>`: E2M1 element type with `float_ue4m3_t` scale factor and vector size 16.

CUTLASS also documents Blackwell SM100 `tcgen05.mma` instructions that support new 4, 6, and 8-bit float types with and without scale factors.

Practical starting point:
- Use the CUTLASS Blackwell narrow precision examples, especially the NVFP4 BF16 GEMM example:
  - `examples/72_blackwell_narrow_precision_gemm/72a_blackwell_nvfp4_bf16_gemm.cu`

Agent tasks:
- Build and run the example on SM100.
- Confirm build flags target sm100 correctly.
- Use it as a canonical reference for packing, scale layout, and kernel configuration.

#### Triton
Triton provides a block scaled matmul tutorial that supports nvfp4 and highlights a key issue:
- scale factor access must be contiguous for low latency in the tensor core MMA inner loop
- devices supporting PTX 8.7+ can use block scaled matmul instructions

Practical starting point:
- Run the tutorial benchmark and treat it as a sanity check for toolchain readiness:
  - `python 10-block-scaled-matmul.py --format nvfp4`

Agent tasks:
- Run Triton tutorial on B200/B300 and capture baseline performance.
- Compare to cuBLASLt NVFP4 GEMM if available in your environment.

#### CuTe kernels
CuTe is evolving quickly for Blackwell, but NVFP4 in CuTe-based kernels means:
- you still must manage packed FP4 data and scale tensors correctly
- you must match the layout expectations of the blockscaled MMA instructions

Recommendation:
- Only attempt this after we have a clear win from Path A or Path B and we have a concrete kernel we want to own.

## KV cache: what "NVFP4/BF16 KV cache" usually means

If a runtime advertises NVFP4/BF16 KV cache, it typically means:
- K and/or V are stored in a compressed NVFP4 representation (plus scales)
- compute may use a mixed path (for example dequantize into bf16 fragments or use blockscaled ops directly)
- not every kernel is NVFP4 end-to-end, and many numerically sensitive parts remain bf16 or fp16

Agent tasks:
- Determine what is NVFP4 and what remains bf16.
- Measure decode speed and memory usage for long context workloads.
- Validate quality: perplexity, downstream task accuracy, or generation quality.

## FlashAttention v4 and score_mod: where NVFP4 fits

For our current FA4 score_mod work:
- NVFP4 does not directly accelerate score_mod.
- If we later decide to compress KV cache, NVFP4 could matter for the attention backend, but that is a larger change than a score_mod function.

If we do explore NVFP4 KV cache with attention:
- treat it as an alternative attention backend project, not a score_mod project
- keep the softmax and logits path in higher precision unless the backend has proven stability and quality

## Implementation checklist (agent friendly)

### Hardware and runtime checks
- Confirm GPU compute capability is 10.x (Blackwell datacenter) for NVFP4 hardware acceleration.
- Confirm driver, CUDA toolkit, and compiler support for FP4 and blockscaled instructions.
- Confirm your framework versions match the NVFP4 support claims (TensorRT-LLM, vLLM, Triton, FlashInfer).

### Data representation and layout checks
- Confirm micro-block size and scale layout expectations:
  - NVFP4 uses 16-value blocks with FP8 E4M3 scales, plus a per-tensor FP32 scalar.
- Confirm packed storage format:
  - expect FP4 to be packed (2 values per byte) for memory efficiency.
- Confirm K dimension constraints:
  - many blockscaled formats require K to be a multiple of the scale vector size (often 16).

### Kernel selection and correctness
- Choose a reference implementation first:
  - CUTLASS example for NVFP4 BF16 GEMM
  - Triton block scaled matmul tutorial
- Build correctness tests:
  - compare GEMM results to bf16 reference for representative shapes
  - measure quantization error distribution per layer
- For LLMs, run end-to-end quality checks (perplexity, eval suite, or golden-output diffs).

### Performance validation
- Always separate prefill and decode metrics.
- Profile memory bandwidth and Tensor Core utilization.
- Compare:
  - bf16 baseline
  - fp8 baseline (if available)
  - nvfp4 candidate
- Track memory footprint changes, especially KV cache.

### Rollout safety
- Keep a switch to disable NVFP4 quickly (runtime flag).
- Keep a mixed precision fallback (bf16 or fp16) for problematic layers.
- Use canary prompts and regression tests.

## Minimal code snippets

### FlashInfer NVFP4 quantization
```python
import torch
from flashinfer.fp4_quantization import nvfp4_quantize, SfLayout

# a: [M, K] fp16 or bf16
a = torch.randn(1024, 4096, device="cuda", dtype=torch.bfloat16)

# global scale factor: [1] float32
a_global_sf = torch.tensor([1.0], device="cuda", dtype=torch.float32)

aq, aq_sf = nvfp4_quantize(
    a,
    a_global_sf,
    sfLayout=SfLayout.layout_128x4,
    do_shuffle=False,
    sf_vec_size=16,
)

# aq is packed: shape [M, K/2]
# aq_sf holds scale factors (shape depends on layout)
```

### Triton tutorial quick run
```bash
python 10-block-scaled-matmul.py --format nvfp4 --bench
```

### CUTLASS example pointer
Look for the NVFP4 BF16 example in CUTLASS:
- `examples/72_blackwell_narrow_precision_gemm/72a_blackwell_nvfp4_bf16_gemm.cu`

## Common pitfalls

- Treating NVFP4 as "just another dtype" and ignoring scale tensors and layout requirements.
- Ignoring the scale factor memory layout. Scale access is in the hot inner loop, and poor layout can erase gains.
- Quantizing numerically sensitive ops without guardrails (softmax logits, some norms).
- Benchmarking only prefill or only decode. NVFP4 can shift bottlenecks.
- Not pinning down what is supported on which runtime. "Early support" can mean limited model coverage or partial feature coverage.

## References (URLs in code blocks)

NVIDIA NVFP4 intro blog:
```text
https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/
```

CUTLASS docs:
```text
https://docs.nvidia.com/cutlass/media/docs/cpp/fundamental_types.html
https://docs.nvidia.com/cutlass/media/docs/cpp/blackwell_functionality.html
```

CUTLASS example file (source in NVIDIA/cutlass):
```text
examples/72_blackwell_narrow_precision_gemm/72a_blackwell_nvfp4_bf16_gemm.cu
```

Triton block scaled matmul tutorial:
```text
https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html
```

TensorRT-LLM releases (look for NVFP4/BF16 KV cache and cuBLASLt NVFP4 backend notes):
```text
https://github.com/NVIDIA/TensorRT-LLM/releases
```

FlashInfer NVFP4 quantize docs:
```text
https://docs.flashinfer.ai/generated/flashinfer.fp4_quantization.nvfp4_quantize.html
```

Transformer Engine NVFP4 notes:
```text
https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html
https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/common.html
```
