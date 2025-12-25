Here is a comprehensive Markdown transcription of the document. I have formatted it to serve as high-quality technical notes, correcting OCR artifacts and spacing.

Where diagrams or graphs appeared in the original PDF, I have inserted **[Visual Note]** blocks that describe the content and data visualization in detail.

***

# Getting Memory-bound Kernels to Speed-of-Light

**Authors:** Wentao Guo, Ted Zadouri, Tri Dao
**Source Repo:** [Dao-AILab/quack](https://github.com/Dao-AILab/quack)
**Date:** July 10, 2025

> **[Visual Note: Header Image]**
> The document features a 3D render of a white duck riding a golden rocket into space, labeled "QUACK" with a speedometer showing "GB/s 3000". This symbolizes the project's goal: high-velocity memory throughput.

To make GPUs "go brrr" for both model training and inference, one has to optimize both compute-bound kernels (e.g., matmul, attention) and memory-bound kernels (pretty much everything else, such as elementwise, normalization, loss).

**Matmul** and **attention** are already some of the most heavily optimized subroutines that we have. Here we instead focus on memory-bound kernels, where most of the time is spent on memory access (IO) instead of on actual computation. By understanding and exploiting the thread and memory hierarchy on modern accelerators, we can get these kernels close to speed-of-light (as fast as theoretically possible).

Thanks to the recent **CuTe-DSL**, we can do so right in the comfort of an ergonomic Python environment, without having to touch CUDA C or C++.

> **[Visual Note: Performance Graphs]**
> A set of four line charts comparing Memory Bandwidth (GB/s) vs. Batch Size/Reduction Dim.
> *   **Comparison:** QuACK (Ours) vs. torch.compile vs. Liger Kernel vs. cuDNN.
> *   **Kernels:** RMSNorm (FP32/BF16), Softmax (FP32/BF16), Cross-Entropy (FP32/BF16).
> *   **Trend:** The "QuACK" (blue line) consistently stays near the top (approx. 3000 GB/s), maintaining stability as dimensions increase. Other methods often show significant drops or lower peaks.

## Arithmetic Intensity and The Roofline
For memory-bound kernels, the ratio between the number of Floating-point Operations (FLOPs) consumed and the number of bytes transferred is small (such ratio is called **Arithmetic Intensity**). Once a kernel’s arithmetic intensity enters the memory-bound regime, the kernel's throughput is determined by how many bytes/second the kernel can deliver rather than by FLOPs/second the kernel computes.

Arithmetic intensity of a memory-bound softmax kernel is $O(1)$.

> **[Visual Note: Roofline Model Graph]**
> A log-log graph showing "Performance [FLOPs]" vs "Arithmetic Intensity [FLOP/byte]".
> *   **The Slope:** The diagonal line represents the memory-bound region (throughput limited by bandwidth).
> *   **The Plateau:** The flat horizontal line represents the compute-bound region (limited by hardware FLOPs).
> *   **Data Point:** A specific point is highlighted with an Arithmetic Intensity of 0.52, placing it firmly in the memory-bound (diagonal) section.

Within memory-bound kernels, elementwise activation is usually easier to deal with—it is inherently perfectly parallel as there are no dependencies across the elements. However, **reduction operations** are also prevalent in DL operators such as Softmax and RMSNorm, and they require an aggregation of all values. A parallel associative reduction algorithm will execute $O(\log(\text{\#reduced dim}))$ rounds of partial reduction across threads in different spatiality where our knowledge of GPU memory hierarchy would help.

> **[Visual Note: Reduction Tree Diagram]**
> A schematic showing a parallel maximum reduction tree.
> *   **Level 1:** Pairs of numbers (e.g., [3,1], [7,0]) are compared to find the max.
> *   **Level 2:** The results of the previous comparisons are paired and compared again.
> *   **Result:** This continues until a single maximum value (7) remains. This visualizes the $O(\log N)$ reduction steps.

In this note, we describe how we can leverage the GPU memory hierarchy to implement efficient reduction kernels using **CuTe DSL**. We focus on 3 kernels: **RMSNorm, Softmax, and Cross Entropy Loss**.

To hit "GPU speed-of-light throughput," we need 2 ingredients:
1.  **Global memory coalesced load/store**
2.  **Hardware-aware reduction strategy**

We also explain **Cluster Reduction**, a feature on Nvidia Hopper (H100) GPUs, and how it helps with very large reductions.

---

## GPU Memory Hierarchy

Before writing code, we must understand the Hopper architecture (e.g., H100). The CUDA execution hierarchy spans four tiers:

1.  **Threads:** Groups of 32 make a Warp.
2.  **Thread Blocks:** Groups of warps inside a Streaming Multiprocessor (SM). Shared Memory (SMEM) is unified (192-256 KB).
3.  **Thread Block Clusters (New in H100):** Up to 16 thread blocks on neighboring SMs. They can access each other's shared memory via **Distributed Shared Memory (DSMEM)**. This avoids costly global memory round-trips.
4.  **Grid:** The full kernel execution.

### Execution Granularity Meets Memory Hierarchy

| Execution Granularity | Operating Memory | Description |
| :--- | :--- | :--- |
| **Threads** | Registers (1st tier) | Each thread can own up to 255 registers. |
| **Warps** | Registers (1st tier) | 32 consecutive threads. Threads within a warp can fetch registers from others via **warp shuffle**. |
| **Thread Blocks** | Shared Memory (2nd tier) | Up to 1024 threads (32 warps). All threads on the same SM can read/write the same unified Shared Memory. |
| **Thread Block Clusters** | **Distributed Shared Memory** (3rd tier) | Neighboring blocks (up to 16) communicate via dedicated SM-to-SM network. |
| **Grids** | Global Memory | All threads access HBM (High Bandwidth Memory). |

> **[Visual Note: Memory Latency Chart]**
> A graph plotting Latency (ns) vs Test Size (KB).
> *   **Registers:** Lowest latency, highest bandwidth (>100 TB/s).
> *   **Shared Memory (SMEM):** ~10-20 ns latency, ~20-30 TB/s bandwidth.
> *   **L2 Cache:** Significant jump in latency (~150-200 ns).
> *   **DRAM (HBM):** Highest latency (~400 ns), lowest bandwidth (3.35 TB/s on H100).
> *   **Takeaway:** We must perform reductions as high up the pyramid (locally) as possible to avoid the HBM bottleneck.

---

## Hardware-Aware Load & Store Strategy

For memory-bound kernels, the HBM's 3.35 TB/s is the bottleneck. We must maximize **Memory Coalescing**.

*   **Vectorization:** On H100, each thread should hold a multiple of 128 bits (e.g., 4x FP32 or 8x BF16).
*   **Method:**
    1.  Asynchronously load from GMEM to SMEM.
    2.  Vectorize load from SMEM to Registers.
    3.  Compute/Reduce.
    4.  Store directly to GMEM.

> **[Visual Note: Coalesced Access]**
> A diagram showing a strip of memory addresses. Arrows indicate that a warp accesses a contiguous block of memory (0 to 384 bytes) simultaneously, rather than scattered random access. This "vectorizes" transactions into single large chunks.

**Code Snippet: Load Implementation (CuTe DSL)**

```python
# blkX: logical id -> address
blkX = ...

# allocate shared memory for the input vectors
smem = cutlass.utils.SmemAllocator()
sX = smem.allocate_tensor(gX.element_type, ...)

# declare copy atoms for memory copy (128 bits)
copy_atom_load_X_async = cute.make_copy_atom(
    cute.nvgpu.cpasync.CopyG2SOp(),
    gX.element_type, num_bits_per_copy=128)

# partition inputs in gmem and smem
tXgX = thr_copy_X_async.partition_S(blkX)
tXsX = thr_copy_X_async.partition_S(sX)

# allocate registers
tXrX = cute.make_fragment_like(tXgX)

# 1. Async copy GMEM -> SMEM
cute.copy(copy_atom_load_X_async, tXgX, tXsX)

# 2. Wait for copy to finish
cute.arch.cp_async_commit_group()
cute.arch.cp_async_wait_group(0)

# 3. Copy SMEM -> Registers
cute.autovec_copy(tXsX, tXrX)
x = tXrX.load()
```

---

## Hardware-Aware Reduction Strategy

We reduce values from top to bottom, matching the memory hierarchy tiers.

| Execution Granularity | Operating Memory | Reduction Strategy |
| :--- | :--- | :--- |
| **Threads** | Registers | Thread reduction |
| **Warps** | Registers | Warp reduction |
| **Thread Blocks** | Shared Memory | Block reduction |
| **Clusters** | Distributed SMEM | Cluster reduction |

### 1. Thread Reduction
Each thread reduces the multiple vectorized values it loaded locally.

```python
# usage
max_x = x.reduce(cute.ReductionOp.MAX, init_val=float('-inf'),
                 reduction_profile=0)
```

### 2. Warp Reduction
A synchronous "butterfly" shuffle allows threads in a warp to read each other's registers.

> **[Visual Note: Butterfly Warp Reduction]**
> A schematic depicting the "XOR warp shuffle."
> *   **Iteration 0:** Threads swap data with neighbors distance 1 away.
> *   **Iteration 1:** Threads swap with neighbors distance 2 away.
> *   **Iteration 2:** Distance 4, and so on.
> *   Result: Every thread in the warp ends up with the fully reduced value.

```python
@cute.jit
def warp_reduce(val, op, width=32):
    for i in range(int(math.log2(width))):
        # Read from another thread's register
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val
```

### 3. Block Reduction
Warps inside a block coordinate via Shared Memory (SMEM).
1.  First thread of each warp writes its result to a **Reduction Buffer** in SMEM.
2.  **Barrier Synchronization.**
3.  Top-laned threads read from the buffer and perform a final reduction.

> **[Visual Note: Block Reduction Flow]**
> 1.  **Write:** Each Warp (blue rectangles) writes its result to a specific slot in SMEM (checkered orange box).
> 2.  **Read:** A single warp reads all those partial results back from SMEM.
> 3.  **Finalize:** That single warp performs a Butterfly Shuffle to get the final block value.

```python
@cute.jit
def block_reduce(val, op, reduction_buffer, init_val=0.0):
    lane_idx = cute.arch.lane_idx()
    # ... calculation of row/col indices ...

    if lane_idx == 0:
        # Write warp-reduced value to SMEM
        reduction_buffer[row_idx, col_idx] = val

    cute.arch.barrier() # Sync

    block_reduce_val = init_val
    if lane_idx < warps_per_row:
        # Read back from buffer
        block_reduce_val = reduction_buffer[row_idx, lane_idx]

    # Warp reduce the values read from buffer
    return warp_reduce(block_reduce_val, op)
```

### 4. Cluster Reduction (New)
Threads in a cluster communicate via **Distributed Shared Memory (DSMEM)** over the SM-to-SM network.

> **[Visual Note: Cluster Reduction]**
> *   **Left Side (Sender):** Warps in one Thread Block write their data into the Shared Memory of *other* Thread Blocks in the cluster (represented by red arrows crossing block boundaries).
> *   **Right Side (Receiver):** A single warp reads the data that was deposited into its local SMEM by all the other blocks, then reduces it.
> *   **Mechanism:** Uses `mapa.shared::cluster` instructions.

**Key Steps:**
1.  **Map Pointers:** Use `set_block_rank` to map local SMEM pointers to remote block addresses.
2.  **Remote Store:** Use `st.async.shared::cluster` to write partial results to a peer's SMEM.
3.  **Barrier:** `mbarrier` ensures all data has arrived.
4.  **Local Reduce:** The target block reduces the values now present in its SMEM.

```python
# Simplified flow
if lane_idx < cluster_n:
    # Write to remote SMEM
    store_shared_remote(val, ..., peer_cta_rank=lane_idx)

# Wait for all warps in cluster
cute.arch.mbarrier_wait(mbar_ptr, phase=0)

# Final reduction loop
# ... load from buffer and warp_reduce ...
```

---

## Benchmarks & Results

The team benchmarked on **NVIDIA H100 80GB (HBM3)**.
*   **Baselines:** Torch.compile (PyTorch 2.7.1), Liger Kernel, cuDNN.
*   **Metric:** Memory Throughput (GB/s).

### Throughput Analysis
*   **Peak Performance:** The implementation achieves **3.01 TB/s** (approx 90% of HBM3 peak) for large reduction dimensions (262k).
*   **Comparison:** This is nearly **50% higher** than torch.compile (1.89 TB/s).
*   **Stability:** The "QuACK" implementation maintains ~3 TB/s consistently. Liger and others degrade significantly at large sizes (e.g., 65k input).

### Why Cluster Reduction Matters (The Liger Case)
Liger kernel performance drops from ~3.0 TB/s to ~2.0 TB/s when input size goes from 32k to 65k.
*   **Reason:** Massive **register spilling**. The kernel runs out of registers trying to handle 65k elements per SM, forcing data back to HBM (slow).
*   **Cluster Solution:** By grouping 16 SMs, QuACK handles $16 \times 32k = 0.5M$ elements without reloading from GMEM. It creates a "Mega SM".

> **[Visual Note: NCU Profiling]**
> *   **Our Implementation:** Shows a balanced diagram where HBM (HBM3) is fully saturated (1.51 TB/s read + write). The "Sol" (Speed of Light) is achieved.
> *   **Liger Fail Case:** The memory workload chart shows "Register Spills" in assembly (LDL instructions) and cascaded writes back to HBM, clogging the bandwidth with unnecessary traffic.

### Comparison with Torch.Compile
Torch.compile generates a Triton kernel.
*   **Inefficiency:** It uses 2 global memory loads (one for max/sum, one for final calculation).
*   **QuACK:** Uses 1 load (or optimized reloading from registers/SMEM).
*   **Result:** Triton achieves ~2.0 TB/s (2/3rds of peak) due to the extra load.

---

## Conclusion

Hitting "speed-of-light" confirms that carefully handcrafted CuTe kernels can squeeze every byte from the hardware. However, this comes at a cost:
*   **Productivity vs. Performance:** There is a trade-off between the ease of Python/Torch and the control of CUDA C++.
*   **Pareto Frontier:** CuTe DSL offers a sweet spot—Python productivity with CUDA-like control.

> **[Visual Note: Pareto Frontier Graph]**
> A conceptual curve showing "Productivity" (Y-axis) vs "Performance" (X-axis).
> *   **Torch:** High Productivity, Lower Performance.
> *   **Triton:** Middle ground.
> *   **CUDA/PTX:** Low Productivity, Max Performance.
> *   **CuTe DSL:** Attempts to push the curve outward, offering high performance with better ergonomics.

**Future:** We believe efficient GPU kernel development can be automated ("LLM.compile") using templates like these for load/store and reduction strategies.

---

## Appendix: TV Layout
(Thread Value Layout for coalesced access)

**Tunable Hyperparameters:**
*   `thread_per_row`: Threads per row.
*   `num_threads`: Threads per block.
*   `cluster_n`: Size of thread block cluster.
*   `vecsize`: Elements per vectorized load.

**Derived Constants & Layouts:**
The layout ensures `(vecsize x cols_per_block, 1)` stride for threads, maximizing contiguous memory access.
