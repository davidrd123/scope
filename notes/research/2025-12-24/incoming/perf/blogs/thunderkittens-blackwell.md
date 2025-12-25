Here is the fully integrated, maximally informative version of the document. I have combined the structural clarity and visual descriptions from the PDF analysis with the active hyperlinks and specific code details from the web source.

***

# ThunderKittens Now on Blackwells!

**Authors:** [Benjamin Spector](https://benjaminfspector.com/), [Aaryan Singhal](https://www.aaryan-singhal.com/), [Dan Fu](https://danfu.org/), [Chris Ré](https://cs.stanford.edu/people/chrismre/)
**Source:** [Hazy Research Blog](https://hazyresearch.stanford.edu/blog/2025-03-15-tk-blackwell)
**Date:** March 15, 2025

> **[Visual Note: Header Image]**
> The cover image features a photorealistic black kitten wearing a dark, futuristic tactical helmet with glowing orange electrical energy emanating from the "ears" of the helmet. The kitten is climbing out of a stone well.

**Quick Links:**
*   [**GEMM Kernels Source Code**](https://github.com/HazyResearch/ThunderKittens/tree/blackwell/kernels/matmul)
*   [**Attention Kernel Source Code**](https://github.com/HazyResearch/ThunderKittens/blob/blackwell/kernels/attn/b200/b200.cu)
*   [**ThunderKittens Blog (TK v2)**](https://hazyresearch.stanford.edu/blog/2024-10-29-tk2)

---

With our collaborators at Together AI, we’ve been having fun playing around with some NVIDIA Blackwell GPUs over the past few weeks and reading about all the exciting new features. The cool thing is – turns out the new features, from 5th-generation tensor cores, to Tensor Memory and CTA pairs, fit pretty well into TK’s (ThunderKittens) existing tile-based abstractions. It’s all about dataflow!

Today, we’re releasing a few new kernels for the NVIDIA Blackwell architecture, written in ThunderKittens:

*   **BF16 and FP8 ThunderKittens GEMM kernels:** Running at or near cuBLAS speeds, and up to **2x faster** than cuBLAS GEMMs on H100.
*   **Attention forwards and backwards:** Both running at near-cuDNN speeds on B200, and up to **2x faster** than FA3 on H100.

In the remainder of this blog, we’re going to take a deep dive into how we use the new hardware features for these kernels, as well as how we adapt kernels written for the NVIDIA Hopper architecture to the new NVIDIA Blackwell architecture. By the end of this blog, you’ll learn all about these new features and how they make attention go vroom!

## It's All About the Dataflow

In our experience, writing performant kernels on NVIDIA Blackwell GPUs feels a lot more like programming a **dataflow-machine** than writing traditional (circa ~2022) CUDA kernels. It’s all about loading in enough data at a high enough throughput to keep the tensor cores hot.

*   **H100:** The main mechanism was using warp-specialized TMA loads to asynchronously fetch data while the tensor cores did computation (e.g., in attention, asynchronously loading the next tiles of $K$ and $V$ while computing the current $QK^T$ tiles and online softmax).
*   **B200:** The tensor cores now have **2–2.5x the power** of those on the H100. To fully utilize that compute, we need to load a lot more data at once. Luckily, the new hardware features make it easier to build deeper pipelines.

## Matrix Multiplication

Of all of the kernels one can run, a matrix multiply kernel has the *least* excuse for bubbles in the data pipeline. With a little care, one can eliminate just about all of them.

> **[Visual Note: Pipeline Efficiency Comparison]**
> A visualization comparing "Tensor Pipe PM Sampling" (Performance Monitor Sampling).
> *   **Top Bar (TK):** A solid blue bar with almost no gaps, indicating continuous throughput.
> *   **Bottom Bar (cuBLAS):** A blue bar interrupted by frequent black gaps ("bubbles"), indicating stalls in the pipeline.

Our new matrix multiplication kernel has a few tricks up its sleeve that are different from Hopper:

1.  **Threadblock Clusters:** We launch threadblock clusters to take advantage of the CTA pair mechanism. This increases reuse and reduces bandwidth requirements on shared memory.
2.  **Producer Warps:** We reserve two producer warps to launch the matrix multiplies for each consumer warpgroup. Consumers no longer launch their own matrix multiplies!
3.  **Direct Signaling:** MMA (Matrix Multiply-Accumulate) instructions directly signal to the producer load warps that pipeline stages are freed and ready to be filled.
4.  **Output Pipelining:** Producers signal consumers that output accumulators are finished and ready. Consumers pipeline output accumulators into registers, into shared memory, and then out into HBM.
    *   *Detail:* We even serialize the consumer warpgroups and force one to load tensor memory into registers and signal the producers before the other can load its tensor memory, so that these loads are pipelined, too.
5.  **Persistent Kernel:** We adopt a persistent kernel to pipeline the next inputs while previous outputs are written out.
    *   *Detail:* In fact, we can even launch the next matrix multiply accumulate block while the previous is still in tensor memory.

**Result:** There is only **one bubble** in the whole tensor pipeline: when the first consumer warpgroup reads its output accumulator into registers. We think this takes about 140 ns every few hundred microseconds; the rest is all tensor cores.

## Attention

One important optimization turns out to be launching the AV MMA’s (Attention-Value Matrix Mul) from the *previous* iteration of the attention loop while starting the QK MMA of the *current* iteration, and loading the K and V tiles of the *next* iteration.

**Pseudocode Structure:**

```cpp
// Producer warpgroup
if (warpgroup::is_producer()) {
    if (warpgroup::warpid() == 0) {
        // do QK.T matmul
    }
    if (warpgroup::warpid() == 1) {
        // do AV matmul
    }
    if (warpgroup::warpid() == 2) {
        // load next K
    }
    if (warpgroup::warpid() == 3) {
        // load next V
    }
}
// Consumer warpgroups
else {
    // Do O online softmax while signaling next AV
}
```

## Hardware Feature Deep Dive

Below, we’ll go into each of the new hardware features that enable these pipelines in more detail.

### 1. Fifth-Generation Tensor Cores

A major new feature of the B200 is larger, faster tensor cores. These run around ~2–2.5x faster than the tensor cores in the H100, and they’re a major source of speedups on the new generation of hardware.

For review: tensor cores are on-chip compute units that compute large GEMMs: the computation $D = A @ B + D$, where $A$, $B$, and $D$ are all matrices. In this blog, we’ll use an $M \times N \times K$ notation for GEMMs, meaning that $A$ has shape $M \times K$, $B$ has shape $K \times N$, and $D$ has shape $M \times N$.

The B200 tensor cores aren’t just faster than the H100 tensor cores – they’re also much larger. From our microbenchmarking, they seem to behave like **$128 \times 128$ systolics**.

*   **Sizing Requirements:** To get full FLOP utilization, you want $M$ and $N$ to be **128** (or larger multiples of 128).
*   **Performance Hit:** Smaller values of $M$ and $N$ run at the corresponding fraction of 128; e.g., a $64 \times 64 \times 64$ GEMM will run at one-quarter the FLOP rate of a $128 \times 128 \times 64$ GEMM.
    *   *Context:* This is a bit of a departure from the H100, where smaller GEMM shapes were enough to max out the tensor cores.

**ThunderKittens Abstraction:**

```cpp
using namespace kittens;
tt<float, 128, 128> d; // 128 x 128 FP32 tensor memory tile
__shared__ st_bf<128, 64> a, b; // 128 x 64 BF16 shared tile
__shared__ semaphore sem; // semaphore (mbarrier)

// init...
mma<transpose::N, transpose::T>(d, a, b, sem); // do it
```

### 2. Tensor Memory

There is a new layer of the memory hierarchy: **Tensor Memory**. These are registers specifically allocated for tensor core instructions.

*   **Capacity:** This gives us an extra **256KB of register memory** to play with, in addition to the 256KB we had already, and up to 227KB of shared memory.
*   **Benefit:** This is especially useful for building more complex dataflow pipelines, which are always trading off the degree of preloading we can accomplish, vs. how much SRAM is available on each SM. Tensor memory gives us more room to work!

**Example: Attention Backwards Storage**
In the TK attention kernel, we make extensive use of tensor memory, especially during the backwards pass (which has a higher footprint due to the need to store gradient values):

```cpp
// The tensor memory used by a warpgroup in attention backwards.
struct wg_tmem_t {
    tt<float, 64, 128> &kg;
    tt<float, 64, 128> &vg;
    tt<float, 64, 64>  &sb;
    tt<float, 64, 64>  &dp;
    tt<bf16,  64, 64>  &pb_bf;
    tt<bf16,  64, 64>  &dp_bf;
    semaphore *mma_sem;
};
```

### 3. CTA Pairs

There’s another nice set of abstractions that allows for deeper coordination between different CUDA thread blocks – **CTA pairs** (Cooperative Thread Array Pairs).

*   **History:** In classic (pre-2022) CUDA, thread blocks executed independently. Starting with Hopper, thread blocks in the same “cluster” were able to coordinate. CTA pairs are deeply related to these notions.
*   **Definition:** Two CTAs operating within the same cluster, scheduled on the same or adjacent SMs.
*   **Capability:** Two CTAs can coordinate to execute tensor core instructions, accessing the Tensor Memory of *both* of the CTAs within the CTA pair. (See the [NVIDIA PTX guide](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-cta-pair) for more details).

**ThunderKittens Abstraction:**
You’ve actually already seen the ThunderKittens abstraction for CTA pairs in this very blog post! If the `ncta` variable in the `mma` template is set to `2`, we can have two CTAs on a single SM coordinate to do a larger GEMM:

```cpp
using namespace kittens;
template<int trans_a, int n_trans_b, ducks::tt::all D, 
         ducks::st_descriptor::input A, ducks::st_descriptor::input B, 
         int acc=1, int ncta=1>
__device__ static inline void mma(D &d, const A &a, const B &b, semaphore &sem);
```

## Conclusion

We hope you enjoyed this whirlwind of new features and how you can use them to write blazing fast attention kernels!

**Resources:**
*   [Attention Kernel (Code)](https://github.com/HazyResearch/ThunderKittens/blob/blackwell/kernels/attn/b200/b200.cu)
*   [BF16 GEMM Kernel (Code)](https://github.com/HazyResearch/ThunderKittens/blob/blackwell/kernels/matmul/B200/matmul.cu)
*   [FP8 GEMM Kernel (Code)](https://github.com/HazyResearch/ThunderKittens/blob/blackwell/kernels/matmul/FP8_B200/matmul.cu)
*   [ThunderKittens Blog](https://hazyresearch.stanford.edu/blog/2024-10-29-tk2)

If you’d like to learn more about these kernels or work with us on developing the next set of updates, please reach out to Ben or Dan!

*   **Ben:** [bfs@stanford.edu](mailto:bfs@stanford.edu)
*   **Dan:** [danfu@ucsd.edu](mailto:danfu@ucsd.edu)

And thanks again to Together AI for collaborating with us and helping us get running with NVIDIA B200’s to write these kernels!

Finally, if you'd like to learn how to build kernels like this and contribute to the cutting edge, please reach out to work with us! Dan is building the kernels team at Together AI and is [actively hiring](https://job-boards.greenhouse.io/togetherai/jobs/4188119007), and we are always looking for great academic collaborators! Come build with us!
