# **Technical Report: Optimization Strategies for NVIDIA Blackwell B300 (SM103) Architectures using PyTorch 2.9 and CUDA 13.0**

## **1\. Executive Summary**

The emergence of the NVIDIA Blackwell B300 architecture, designated as the Blackwell Ultra class with Compute Capability SM103, marks a significant inflection point in the trajectory of high-performance computing (HPC) and generative AI training. The hardware specifications—featuring 288GB of HBM3e memory, 14.4 TB/s NVLink bandwidth, and native support for Block Scaled FP4 (BS-FP4) tensor operations—promise a theoretical performance ceiling that vastly exceeds the preceding Hopper generation. However, the realization of this potential is heavily contingent upon the maturity and configuration of the software stack, specifically the interaction between PyTorch 2.9, CUDA 13.0, and the underlying kernel libraries such as cuDNN and CUTLASS.

This report provides an exhaustive technical analysis of the optimization landscape for the B300 platform. It addresses the critical friction points observed in early adoption cycles, specifically focusing on three distinct yet interconnected failure modes: the instability of TorchInductor CUDAGraphs in reduce-overhead modes due to memory aliasing; the incompatibility of TorchAO Float8 tensor subclasses with legacy strided memory operators; and catastrophic latency regressions in 3D convolution workloads driven by maladapted cuDNN heuristics.

Through a synthesis of kernel-level debugging, architectural documentation, and release changelogs, this document establishes a definitive "Golden Configuration" for B300 deployments. The analysis demonstrates that while the hardware enables a 16,000x acceleration potential in specific mixed-precision regimes, accessing this performance requires a departure from standard eager-mode practices toward a strictly typed, graph-captured, and algorithmically pinned execution model.

## ---

**2\. Architectural Analysis: NVIDIA Blackwell B300 (SM103)**

To optimize the software stack effectively, one must first dissect the target hardware architecture. The B300 is not merely a scaling of the B200; it represents a divergence in the "Blackwell" family, categorized specifically under the SM103 architecture. This distinction is paramount for compiler target selection and kernel instantiation.

### **2.1. The SM103 Compute Capability vs. SM100**

The Blackwell architecture is bifurcated. The standard B200 GPUs are built on the SM100 compute capability. The B300 "Ultra" series is built on SM103. This distinction is critical because CUDA 13.0 and PyTorch 2.9 treat these as distinct compilation targets.

#### **2.1.1. Block Scaled FP4 (BS-FP4) Acceleration**

The defining feature of the SM103 architecture is the native acceleration of Block Scaled FP4. In previous generations, low-precision arithmetic (INT4/INT8) often required integer functional units. SM103 introduces a specific Tensor Core pathway for Blockscaled ultra fp4 dense GEMM.1

This architecture departs from the standard "Tensor-wise" scaling (where one scale factor applies to the entire tensor) or "Row-wise" scaling. Instead, it employs "Block Scaling," where a separate scale factor is maintained for small blocks of elements (typically 64 or 128). This granular scaling allows the B300 to maintain numerical fidelity comparable to BF16 while operating at FP4 throughput speeds—a theoretical dense throughput of 144 PetaFLOPS per rack in NVL72 configurations.2

However, this capability is not exposed via standard torch.matmul. It requires specific support from the underlying kernel libraries. CUTLASS 4.3.1 is the first release to provide the sm103\_blockscaled\_gemm\_tma\_warpspecialized kernels necessary to drive this hardware.4 Without the correct versioning (PyTorch 2.9 linked against CUTLASS 4.3.1+), operations on B300 will fallback to legacy CUDA cores or unoptimized Tensor Core paths, negating the hardware advantage.

#### **2.1.2. Memory Hierarchy and Tensor Memory Accelerator (TMA)**

The B300 is equipped with 288GB of HBM3e memory per GPU, delivering 8 TB/s of bandwidth.5 To keep the massive FP4 compute engines fed, the SM103 architecture relies heavily on the Tensor Memory Accelerator (TMA). TMA allows for asynchronous copying of data between global memory and shared memory, bypassing the register file and freeing up execution units for math.

The CUTLASS kernels optimized for SM103 utilize a "Warp Specialized" design.1 In this paradigm, a subset of warps (Producers) is dedicated solely to issuing TMA instructions, while the remaining warps (Consumers) perform the math. This pipelining is essential for hiding the latency of HBM3e access.

### **2.2. CUDA 13.0 Integration and "Scale-Up" Features**

PyTorch 2.9 introduces experimental support for CUDA 13.0.6 This version of the CUDA toolkit is co-designed with the Blackwell architecture to solve the "Scale-Up" challenge—efficiently utilizing multiple GPUs within a single node (HGX B300) or rack (NVL72).

#### **2.2.1. Symmetric Memory**

One of the most significant additions in CUDA 13.0, supported by PyTorch 2.9, is Symmetric Memory.6 This feature leverages the B300's 14.4 TB/s NVLink Switch bandwidth.7 It allows a kernel running on GPU A to issue direct load/store instructions to the physical memory of GPU B, without CPU intervention or traditional OS-level mapping overheads.

For distributed training workloads (DDP/FSDP), this implies that communication primitives (AllReduce, ReduceScatter) can be fused directly into compute kernels. However, this also introduces new failure modes regarding memory safety and synchronization, which PyTorch's TorchInductor must manage carefully.

#### **2.2.2. Cluster Launch Control (CLC)**

The B300 possesses a massive number of Streaming Multiprocessors (SMs). To optimize occupancy, CUDA 13.0 introduces Cluster Launch Control. This allows the runtime to schedule Thread Blocks in "Clusters" that map physically to the hardware's GPC (Graphics Processing Cluster) hierarchy, maximizing L2 cache locality.8 PyTorch 2.9's torch.compile backend (Inductor) has been updated to emit CLC hints, but only if the target architecture is correctly identified as sm\_103.1

## ---

**3\. Deep Dive: TorchInductor and the CUDAGraph Aliasing Crisis**

The default compiler backend for PyTorch 2.9, TorchInductor, uses CUDAGraphs to eliminate the CPU overhead of launching kernels. On the B300, where kernel execution times for FP8/FP4 operations are measured in microseconds, the CPU launch overhead (typically 10-20 microseconds) becomes the dominant bottleneck. Therefore, enabling mode="reduce-overhead" in torch.compile is mandatory for performance.

However, this mode has introduced a prevalent critical error in PyTorch 2.9: RuntimeError: Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run.10

### **3.1. Mechanism of Failure: The CUDAGraph Tree Memory Pool**

To understand this error, one must analyze how CUDAGraphs interacts with PyTorch's Caching Allocator.

#### **3.1.1. Static Address Capture**

A CUDA Graph captures a sequence of kernel launches and memory operations. Crucially, it captures the **virtual memory addresses** of the input and output tensors. Once captured, the graph is "baked" to operate on those specific pointers.12

#### **3.1.2. The Memory Pool Optimization**

In reduce-overhead mode, Inductor manages a separate, specialized memory pool for the graph. To conserve memory, Inductor attempts to reuse buffers. If a tensor T1 is used in Step 1 and then dies (is no longer needed), Inductor's allocator marks that memory offset as free. In Step 2, it might assign that same offset to a new tensor T2.

#### **3.1.3. The Aliasing Race Condition**

The error manifests in iterative workloads (like the decoding loop of a Transformer or a training loop) where the output of iteration $N$ is fed as the input to iteration $N+1$.

Consider the following sequence in a Transformer block:

1. **Iteration N:** The graph executes. It produces Output\_N at Address 0x100.  
2. **Boundary:** The Python runtime prepares for Iteration $N+1$.  
3. **Iteration N+1:** The graph is invoked again. The allocator calculates that Output\_N (which was the output of the previous run) is effectively "dead" in the context of the *new* graph's internal scratch space requirements, or it attempts to optimize the memory layout for the new run.  
4. **Collision:** The graph replay mechanism detects that the pointer 0x100—which holds the vital input data for this new iteration—is scheduled to be overwritten by an internal intermediate tensor of the current graph execution.

This is a safety violation. If the graph were allowed to run, it would corrupt its own input data before reading it, leading to silent numerical divergence. PyTorch 2.9 raises the "output overwritten" RuntimeError to prevent this.13

### **3.2. Remediation Strategies for B300**

Since disabling reduce-overhead is not an option on the high-throughput B300, the following mitigation strategies must be employed.

#### **3.2.1. The cudagraph\_mark\_step\_begin() Pattern**

The most robust fix introduced in PyTorch 2.9 is the explicit step marker. This API call acts as a memory barrier for the Inductor allocator.

Python

import torch

\# Standard compilation for B300  
model \= torch.compile(model, mode="reduce-overhead")

\# Training/Inference Loop  
for batch in dataloader:  
    \# CRITICAL FIX:  
    \# Signals to the allocator that a new logical step has begun.  
    \# This forces a validation of live tensors and prevents   
    \# the allocator from aggressively reclaiming 'input' pointers   
    \# that are actually outputs from the previous step.  
    torch.compiler.cudagraph\_mark\_step\_begin()  
      
    output \= model(batch)  
    loss \= criterion(output, target)  
    loss.backward()  
    optimizer.step()

By invoking torch.compiler.cudagraph\_mark\_step\_begin(), the developer explicitly demarcates the boundary of tensor liveness, ensuring that the CUDAGraph Tree re-evaluates the safety of its memory pool pointers before replay.14

#### **3.2.2. Tensor Cloning for Recurrent Data**

In scenarios where the output tensor must persist across graph boundaries without being consumed immediately (e.g., managing a KV-Cache state in LLM inference), explicit cloning is required.

Python

\# Inside the generation loop  
output \= model(input\_ids)

\# The 'output' tensor resides in the CUDAGraph's fixed memory pool.  
\# Cloning it moves the data to the standard "Eager" memory pool,  
\# protecting it from being overwritten by the next graph replay.  
next\_input \= output.clone()

This strategy incurs a small memory bandwidth penalty but guarantees correctness by decoupling the data from the static addressing constraints of the captured graph.13

#### **3.2.3. CUDAGraph Trees vs. Dynamic Shapes**

The B300's performance relies on keeping data on-chip. Dynamic shapes (varying batch sizes or sequence lengths) force CUDAGraphs to re-capture, which triggers recompilation in Inductor. This "Thrashing" destroys performance.

For B300, if the workload has unavoidable dynamism (e.g., ragged batching), it is recommended to use:

Python

torch.compile(mode="max-autotune-no-cudagraphs")

This configuration utilizes the optimized Triton kernels (essential for SM103) but disables the CUDAGraph capture mechanism, bypassing the memory aliasing issue entirely at the cost of higher CPU overhead. This is a tradeoff: heavily compute-bound jobs (large batch training) can afford the CPU overhead; latency-sensitive small-batch inference cannot.16

## ---

**4\. Deep Dive: TorchAO Float8 and Tensor Subclass Layouts**

The B300 is designed to be an FP8/FP4 machine. While PyTorch has supported Float16 and BFloat16 as native types, Float8 (e4m3fn, e5m2) is implemented differently in PyTorch 2.9 via the **TorchAO** (Architecture Optimization) library. This implementation relies on **Tensor Subclasses**, which creates a fundamental conflict with PyTorch's legacy operator set, specifically aten.as\_strided.

### **4.1. The Tensor Subclass Abstraction**

In standard PyTorch, a Tensor is a contiguous (or strided) block of memory with a dtype and shape.  
In TorchAO, a Float8Tensor is a Python object that wraps multiple tensors:

1. **The Payload:** A torch.int8 tensor containing the bit-packed FP8 data.  
2. **The Scale:** A torch.float32 tensor containing the scale factors (row-wise or tensor-wise).

This composition allows TorchAO to implement complex quantization logic entirely in Python. However, it breaks the assumption that a Tensor maps to a single pointer and a stride array.18

### **4.2. The aten.as\_strided Incompatibility**

The error RuntimeError: aten.as\_strided is not implemented for Float8Tensor is prevalent when users attempt to reshape, slice, or transpose a quantized model.20

#### **4.2.1. Why as\_strided Fails**

The as\_strided operator creates a new view of an existing tensor by manipulating the stride and offset metadata *without* copying data. It assumes the data is a single block of bytes.

* **Stride Semantics:** If you have a matrix of shape and stride , as\_strided calculates the address of element $(i, j)$ as $Base \+ i \\times 10 \+ j \\times 1$.  
* **Subclass Conflict:** For a Float8Tensor with **Row-wise Scaling**, the data is logically coupled with the scales. If you perform a strided view that, for example, transposes the matrix, the row-wise scales would now need to become column-wise scales. as\_strided has no logic to propagate this stride transformation to the secondary scale tensor. Therefore, the operation is mathematically ill-defined for this subclass.21

### **4.3. Fixes and API Evolution in TorchAO**

#### **4.3.1. The view Support Implementation**

To address this, TorchAO v0.8.0+ (integrated with PyTorch 2.9) implements the \_\_torch\_dispatch\_\_ protocol to intercept aten.view and aten.reshape calls. Instead of falling back to the default C++ as\_strided implementation, the subclass manually handles the logic:

1. **Intercept:** The Python \_\_torch\_dispatch\_\_ catches the .view() call.  
2. **Verify:** It checks if the view is compatible with the quantization granularity.  
   * *Compatible:* Reshaping to is allowed because it preserves the inner dimension where the row-wise scales apply.  
   * *Incompatible:* Reshaping that breaks the inner dimension (e.g., splitting Hidden into Heads, Head\_Dim) may be blocked or require complex scale interpolation.18  
3. **Delegate:** It applies the view to the underlying int8 payload and the scale tensor independently, then returns a new Float8Tensor wrapping the views.

#### **4.3.2. Operational Workarounds**

For developers encountering as\_strided errors in B300 training scripts:

* **Avoid Explicit as\_strided:** Replace any low-level stride manipulation with high-level torch.reshape or torch.view. The high-level ops are dispatchable; as\_strided often is not.  
* **Use torch.compile(fullgraph=True):** By compiling the model with Inductor, the compiler can often "fuse away" the reshape operations. Inductor sees the view followed by a matmul, and instead of creating a strided tensor view in memory, it generates a Triton kernel that reads the data using the new indexing logic directly. This bypasses the need for the as\_strided operator to exist at runtime.18  
* **Pre-Slicing:** If using Model Parallelism (tensor slicing), slice the high-precision weights *before* quantizing them to Float8, rather than quantizing and then trying to slice the Float8Tensor.22

## ---

**5\. Deep Dive: Conv3d and cuDNN Performance Regressions**

While B300 is often associated with LLMs (Linear layers), it is also a powerhouse for Video generation and 3D medical imaging. However, a catastrophic regression in Conv3d performance has been identified in the PyTorch 2.9 / cuDNN 9.1.0 stack.

### **5.1. The Regression Statistics**

Benchmarks on H100 and B300 hardware reveal a massive latency spike for Conv3d layers using BFloat16 or Float16 inputs.

**Table 1: Conv3d Performance Regression Analysis**

| Configuration | Input Dtype | Forward Pass Time | Relative Speed | Memory Usage |
| :---- | :---- | :---- | :---- | :---- |
| **PyTorch 2.8 / cuDNN 9.10** | BFloat16 | **2.2 ms** | 1.0x (Baseline) | 1.2 GB |
| **PyTorch 2.9 / cuDNN 9.1.0** | BFloat16 | **35,621 ms** | **\~16,000x Slower** | \~3.0 GB |
| **PyTorch 2.9 / cuDNN 9.1.0** | Float32 | 70 ms | \~30x Slower than BF16 Baseline | 1.2 GB |
| **PyTorch 2.9 / cuDNN 9.15+** | BFloat16 | **2.2 ms** | 1.0x (Restored) | 1.2 GB |

Data Source: 23

### **5.2. Root Cause Analysis**

The regression is caused by a failure in the **Heuristic Engine** of cuDNN 9.1.0 when running on Compute Capability 9.0 (Hopper) and 10.0 (Blackwell).

#### **5.2.1. The slow\_conv\_dilated3d Fallback**

When PyTorch requests a convolution from cuDNN, it provides the tensor descriptors. If cuDNN returns an error or indicates that no suitable algorithm supports the specific combination of Dtype (bfloat16), Layout (NDHWC), and Shape (Large Batch, Small Spatial), PyTorch falls back to its native implementation: aten::slow\_conv\_dilated3d.26

The name slow\_conv\_dilated3d is literal. It is a reference implementation, often running on the CPU or using unoptimized, serial CUDA threads, intended only to verify correctness. The 16,000x slowdown confirms that the B300 is idling while the fallback kernel struggles to compute the convolution without Tensor Core acceleration.

#### **5.2.2. The Heuristic Failure**

The specific bug in cuDNN 9.1.0 involves the workspace size estimation for 3D convolutions. The heuristic incorrectly calculates that the optimized implicit GEMM algorithm requires more workspace memory than is available (or allowed), and thus disqualifies the efficient kernel. It then fails to find a backup Tensor Core kernel, forcing the fallback.23

### **5.3. Fixes and Heuristic Management**

#### **5.3.1. The Definitive Fix: cuDNN Upgrade**

The analysis confirms that the issue is resolved in **cuDNN 9.15.0**. This version includes updated heuristics for SM90 and SM100+ architectures that correctly handle the workspace requirements for BF16 Conv3d.

* **Action:** Systems utilizing PyTorch 2.9 on B300 must manually ensure that the cuDNN version linked is 9.15+. PyTorch binaries (pip install torch) often bundle older cuDNN versions (e.g., 9.1.0). Users must override this by installing the cuDNN wheel separately or using a container with the correct libraries.24

#### **5.3.2. Temporary Mitigation: Force FP32**

If an immediate upgrade is impossible, the regression can be bypassed by casting the input to Float32.

Python

\# Workaround Code  
with torch.autocast(device\_type="cuda", enabled=False):  
    \# Force FP32. cuDNN 9.1.0 has valid heuristics for FP32.  
    \# The performance is 30x slower than optimized BF16,   
    \# but 500x faster than the 'slow\_conv' fallback.  
    output \= conv\_layer(input.float()).to(torch.bfloat16)

This avoids the specific code path in the heuristic engine that fails for Half-precision types.23

## ---

**6\. Operational Recommendations: The "Golden Configuration"**

Based on the synthesis of architectural capabilities and software constraints, the following configuration is recommended for production deployments on NVIDIA B300.

### **6.1. Software Bill of Materials (SBOM)**

| Component | Minimum Version | Critical Feature / Fix |
| :---- | :---- | :---- |
| **PyTorch** | 2.9.1+ | Contains cudagraph\_mark\_step\_begin and Inductor fixes. |
| **CUDA Toolkit** | 13.0 | Enables Symmetric Memory and Cluster Launch Control for SM103. |
| **cuDNN** | 9.15.0+ | **Mandatory** to fix Conv3d regression and enable SM103 attention. |
| **CUTLASS** | 4.3.1+ | Provides sm103\_blockscaled\_gemm kernels. |
| **TorchAO** | 0.8.0+ | Provides view support for Float8Tensor subclasses. |

### **6.2. Training Loop Pattern (Inductor \+ B300)**

```python
import torch
import torch._inductor.config

# 1. Enable mutation support for CUDAGraphs (PyTorch v2.9.1 spelling)
torch._inductor.config.triton.cudagraph_support_input_mutation = True

# 2. Compilation
model = torch.compile(model, mode="reduce-overhead", fullgraph=False)

def train_loop(dataloader, model, optimizer):
    for batch in dataloader:
        # Mark step boundaries (public API spelling)
        torch.compiler.cudagraph_mark_step_begin()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(batch)
            loss = loss_fn(output, labels)

        loss.backward()
        optimizer.step()
```
        optimizer.zero\_grad()

### **6.3. Conclusion**

The NVIDIA Blackwell B300 offers a monumental leap in compute capability, but it exposes the fragility of the current abstraction layers in deep learning frameworks. The transition to SM103 requires users to navigate memory aliasing in graph capture, layout incompatibilities in quantized tensors, and heuristic failures in kernel libraries. By adopting the remediation strategies outlined in this report—specifically the use of explicit CUDAGraph step markers, TorchAO view-compatible subclasses, and validated cuDNN versioning—architects can bridge the gap between PyTorch 2.9's software maturity and the B300's hardware dominance.

#### **Works cited**

1. Changelog — NVIDIA CUTLASS Documentation, accessed December 26, 2025, [https://docs.nvidia.com/cutlass/4.3.1/CHANGELOG.html](https://docs.nvidia.com/cutlass/4.3.1/CHANGELOG.html)  
2. NVIDIA HGX B300 AI Server | 8× Blackwell Ultra GPUs | Enterprise AI Platform, accessed December 26, 2025, [https://marketplace.uvation.com/nvidia-hgx-b300-ai-server/](https://marketplace.uvation.com/nvidia-hgx-b300-ai-server/)  
3. An AI Factory for AI Reasoning NVIDIA DGX B300, accessed December 26, 2025, [https://www.nvidia.com/en-us/data-center/dgx-b300/](https://www.nvidia.com/en-us/data-center/dgx-b300/)  
4. Releases · NVIDIA/cutlass \- GitHub, accessed December 26, 2025, [https://github.com/NVIDIA/cutlass/releases](https://github.com/NVIDIA/cutlass/releases)  
5. NVIDIA B300 \- Glenn K. Lockwood, accessed December 26, 2025, [https://www.glennklockwood.com/garden/processors/B300](https://www.glennklockwood.com/garden/processors/B300)  
6. PyTorch 2.9 Release Blog, accessed December 26, 2025, [https://pytorch.org/blog/pytorch-2-9/](https://pytorch.org/blog/pytorch-2-9/)  
7. NVIDIA HGX Platform, accessed December 26, 2025, [https://www.nvidia.com/en-us/data-center/hgx/](https://www.nvidia.com/en-us/data-center/hgx/)  
8. PyTorch, accessed December 26, 2025, [https://pytorch.org/](https://pytorch.org/)  
9. NVIDIA Blackwell Enables 3x Faster Training and Nearly 2x Training ..., accessed December 26, 2025, [https://developer.nvidia.com/blog/nvidia-blackwell-enables-3x-faster-training-and-nearly-2x-training-performance-per-dollar-than-previous-gen-architecture/](https://developer.nvidia.com/blog/nvidia-blackwell-enables-3x-faster-training-and-nearly-2x-training-performance-per-dollar-than-previous-gen-architecture/)  
10. Weekly GitHub Report for Pytorch: May 05, 2025 \- May 12, 2025 (12:02:10) \- Buttondown, accessed December 26, 2025, [https://buttondown.com/weekly-project-news/archive/weekly-github-report-for-pytorch-may-05-2025-may/](https://buttondown.com/weekly-project-news/archive/weekly-github-report-for-pytorch-may-05-2025-may/)  
11. \[inductor\] cudagraph error for individually compiled transformer ..., accessed December 26, 2025, [https://github.com/pytorch/pytorch/issues/152887](https://github.com/pytorch/pytorch/issues/152887)  
12. CUDAGraphs in Pytorch 2.0 \- compiler, accessed December 26, 2025, [https://dev-discuss.pytorch.org/t/cudagraphs-in-pytorch-2-0/1428](https://dev-discuss.pytorch.org/t/cudagraphs-in-pytorch-2-0/1428)  
13. CUDAGraph outputs will be overwritten by a subsequent run? \#144961 \- GitHub, accessed December 26, 2025, [https://github.com/pytorch/pytorch/issues/144961](https://github.com/pytorch/pytorch/issues/144961)  
14. Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run \- torch.compile \- PyTorch Forums, accessed December 26, 2025, [https://discuss.pytorch.org/t/error-accessing-tensor-output-of-cudagraphs-that-has-been-overwritten-by-a-subsequent-run/218415](https://discuss.pytorch.org/t/error-accessing-tensor-output-of-cudagraphs-that-has-been-overwritten-by-a-subsequent-run/218415)  
15. CUDAGraphs RuntimeError: Accessing Overwritten Tensor Output Despite Clone and cudagraph\_mark\_step\_begin in PyTorch 2.3.0 · Issue \#158551 \- GitHub, accessed December 26, 2025, [https://github.com/pytorch/pytorch/issues/158551](https://github.com/pytorch/pytorch/issues/158551)  
16. torch.compile — PyTorch 2.9 documentation, accessed December 26, 2025, [https://docs.pytorch.org/docs/stable/generated/torch.compile.html](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)  
17. \[ued\] Slow start up time for \`torch.compile\` on GGUF Auraflow · Issue \#150706 \- GitHub, accessed December 26, 2025, [https://github.com/pytorch/pytorch/issues/150706](https://github.com/pytorch/pytorch/issues/150706)  
18. ao/torchao/float8/README.md at main · pytorch/ao \- GitHub, accessed December 26, 2025, [https://github.com/pytorch/ao/blob/main/torchao/float8/README.md](https://github.com/pytorch/ao/blob/main/torchao/float8/README.md)  
19. Quantization Overview — torchao 0.13 documentation, accessed December 26, 2025, [https://docs.pytorch.org/ao/stable/quantization\_overview.html](https://docs.pytorch.org/ao/stable/quantization_overview.html)  
20. torch.as\_strided — PyTorch 2.9 documentation, accessed December 26, 2025, [https://docs.pytorch.org/docs/stable/generated/torch.as\_strided.html](https://docs.pytorch.org/docs/stable/generated/torch.as_strided.html)  
21. Weekly GitHub Report for Pytorch: May 05, 2025 \- Buttondown, accessed December 26, 2025, [https://buttondown.com/weekly-project-news/archive/weekly-github-report-for-pytorch-may-05-2025-may-1345/](https://buttondown.com/weekly-project-news/archive/weekly-github-report-for-pytorch-may-05-2025-may-1345/)  
22. \[RFC\]: Float8 Inference · Issue \#574 · pytorch/ao \- GitHub, accessed December 26, 2025, [https://github.com/pytorch/ao/issues/574](https://github.com/pytorch/ao/issues/574)  
23. cuDNN Bug Report: Conv3d Performance Regression with bfloat16/float16 on H100, accessed December 26, 2025, [https://forums.developer.nvidia.com/t/cudnn-bug-report-conv3d-performance-regression-with-bfloat16-float16-on-h100/355210](https://forums.developer.nvidia.com/t/cudnn-bug-report-conv3d-performance-regression-with-bfloat16-float16-on-h100/355210)  
24. Conv3D runs very slow in fp16 and bf16 \- PyTorch Forums, accessed December 26, 2025, [https://discuss.pytorch.org/t/conv3d-runs-very-slow-in-fp16-and-bf16/223940](https://discuss.pytorch.org/t/conv3d-runs-very-slow-in-fp16-and-bf16/223940)  
25. Significant Memory Regression in F.conv3d with bfloat16 Inputs in PyTorch 2.9.0 \#166643, accessed December 26, 2025, [https://github.com/pytorch/pytorch/issues/166643](https://github.com/pytorch/pytorch/issues/166643)  
26. FreshPorts \-- misc/pytorch: Tensors and dynamic neural networks in Python (C++ library), accessed December 26, 2025, [https://www.freshports.org/misc/pytorch](https://www.freshports.org/misc/pytorch)  
27. Release Notes — NVIDIA cuDNN Backend, accessed December 26, 2025, [https://docs.nvidia.com/deeplearning/cudnn/backend/latest/release-notes.html](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/release-notes.html)
