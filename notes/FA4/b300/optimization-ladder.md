# B300 Optimization Ladder: Where We Are & What's Above

> **Context:** The hierarchy of GPU optimization work, mapped to the B300 journey.
> **Updated:** 2025-12-25

---

## The Ladder

### Level 1: Use the Tools
*"I called the library"*

- Use PyTorch ops, maybe torch.compile
- Call FlashAttention, hope it works
- Accept whatever FPS you get

**Example:** "I used FlashAttention and got 8.8 FPS on B300"

---

### Level 2: Profile and Understand
*"I know where the time goes"*

- Build profiling infrastructure
- Identify bottlenecks with data
- Understand the gap between expectation and reality

**What we did:**
- Added `PROFILE_PIPELINE_BLOCKS`, `PROFILE_ATTENTION` infrastructure
- Discovered block breakdown: denoise 62%, decode 25%, recompute 12%
- Found attention was only 27% of self-attention time
- Identified QKV + RoPE as bigger than expected

**Example:** "Profiling showed VAE decode was 760ms on cu129 - that's the bottleneck"

---

### Level 3: Write Custom Kernels
*"I wrote code that runs on the GPU"*

- Triton kernels that work correctly
- Beat naive implementations
- Understand thread blocks, shared memory, tiling

**What we did:**
- Wrote Triton Kernel B for KV-bias attention
- Beat flex_attention by 10.7% (1.02ms vs 1.14ms)
- Learned Triton autotuning (BLOCK_M, BLOCK_N, warps, stages)

**Example:** "My Triton kernel handles the KV-bias pattern correctly and beats PyTorch's flex_attention"

---

### Level 4: Beat the Libraries
*"My specialized kernel beats FlashAttention on this workload"*

- Custom kernels that outperform established libraries
- Exploit problem structure they can't assume
- Understand why the library isn't optimal for your case

**What we did:**
- Integrated FA4/CUTE with score_mod for KV-bias
- 1.89x faster than our own Triton kernel (0.54ms vs 1.02ms)
- Diagnosed SM103 runtime issues (cu130 requirement)
- Found Triton's SM103 fallback was catastrophic (scalar kernel → 1 FPS)

**Example:** "FA4 with custom score_mod beats both our Triton kernel and standard FlashAttention because we exploit the KV-bias structure"

---

## ✅ YOU ARE HERE: Between Level 4 and 5

Current state:
- B300: **19.0 FPS** (cu130 + FA4 + compile) vs starting point of **8.8 FPS**
- Closed the gap with B200 (20 FPS)
- Solved the "white whale" mystery (runtime stack + backend selection)
- Have documented, reproducible optimizations

What's missing for Level 5:
- Kernels are still separate (RoPE, Attention, Projections)
- Not exploiting Blackwell-specific features (TMA, warp specialization)
- torch.compile integration is partial (some modes abort on SM103)

---

### Level 5: Fused Operations
*"One kernel does what used to be four"*

Instead of:
```
RoPE kernel        → 0.38ms
QKV projection     → X ms (GEMM)
Attention kernel   → 0.54ms
Output projection  → Y ms (GEMM)
────────────────────────────
4 kernel launches, 4 memory round-trips
```

Fused:
```
Load Q,K,V once
  → RoPE in registers (no store/load)
  → Attention in SRAM
  → Output projection fused
────────────────────────────
1 kernel launch, 1 memory round-trip
```

**Why it matters:** Memory bandwidth is often the bottleneck, not compute. Each kernel launch reads from and writes to global memory. Fusion eliminates intermediate round-trips.

**Concrete opportunity for B300:**
- RoPE is currently 0.38ms × 2 (Q and K) = 0.76ms
- If fused into attention, that's ~0.76ms saved per forward pass
- At 4 denoise steps + 1 recompute = 5 passes = **~3.8ms saved per frame**
- That's potentially **+2-3 FPS** on top of current 19 FPS

**How to get there:**
- FA4/CUTE supports custom prologue/epilogue
- Inject RoPE computation into the attention kernel's data loading phase
- Requires understanding CUTE's tile iterators and register layout

---

### Level 6: Architecture-Specific Optimization
*"I'm using Blackwell features that didn't exist last year"*

**Warp Specialization (Hopper/Blackwell):**
```
Warp 0-3:  Load data (TMA - Tensor Memory Accelerator)
Warp 4-7:  Compute (Tensor Cores)
Warp 8-11: Store results

All running simultaneously in a software pipeline
```

**TMA (Tensor Memory Accelerator):**
- Hardware unit that loads tensor tiles directly into shared memory
- Async, overlapped with compute
- No thread involvement during transfer

**What this looks like in code:**
```python
# Level 4 (what we have)
q = tl.load(Q_ptr + offsets)  # Threads load cooperatively

# Level 6 (Blackwell-optimized)
tma_load(Q_smem, Q_gmem_desc)  # Hardware loads async
barrier.wait()                  # Compute on previous tile
                                # while next tile loads
```

**Concrete opportunity for B300:**
- B300 is SM103 (Blackwell)
- TMA and warp specialization are available
- Current kernels don't use them

---

### Level 7: Novel Algorithms
*"I invented a new way to compute this"*

This is FlashAttention-level contribution:
- FlashAttention: Tiling + recomputation to avoid O(n²) memory
- Flash-Decoding: Parallel KV-cache attention for long sequences
- Ring Attention: Distributed attention across devices

**What it would look like for our problem:**
- New algorithm for streaming video attention that exploits temporal structure
- Maybe: predictive KV-cache that skips redundant computation
- Maybe: frame-delta attention that only recomputes what changed

---

### Level 8: Production at Scale
*"Millions of users run my kernel"*

- Battle-tested, handles edge cases
- Optimized for multiple GPU generations
- Documented, maintained, upstreamed

**Examples:**
- FlashAttention (Tri Dao)
- cuBLAS/cuDNN (NVIDIA)
- Triton compiler itself

---

## The Path Forward

### Immediate (Level 4→5): Fuse RoPE + Attention

**Effort:** Medium (days, not weeks)
**Impact:** +2-3 FPS estimated

Steps:
1. Understand CUTE's prologue mechanism
2. Implement RoPE as a score_mod prologue (apply before softmax)
3. Or: Implement as a custom Q/K loading hook
4. Benchmark, validate correctness

### Medium-term (Level 5→6): TMA/Warp Specialization

**Effort:** High (weeks)
**Impact:** Unknown, potentially significant

Steps:
1. Study ThunderKittens / FlashAttention-3 implementations
2. Understand TMA descriptors and async barriers
3. Rewrite attention kernel with producer/consumer warp split
4. Profile to ensure compute/memory overlap

### Stretch (Level 6→7): Novel Algorithm

**Effort:** Research-level
**Impact:** Could be transformative

Ideas:
- Temporal attention sparsity (skip unchanged regions)
- Predictive KV-cache (speculate on next frame's keys)
- Frame-delta encoding (diff-based attention updates)

---

## Reference: Current B300 Performance Stack

```
Layer                          Time        Status
─────────────────────────────────────────────────
VAE Decode                     ~194ms      ✅ Fixed (cu130)
Denoise (4 steps)              ~528ms      Partially optimized
├── Self-Attention             ~56%
│   ├── KV-bias kernel         ~22%        ✅ FA4 score_mod
│   ├── QKV projection         ~21%        Stock GEMM
│   └── RoPE                   ~15%        ✅ Triton fused
├── Cross-Attention            ~22%        Stock
└── FFN                        ~22%        Stock
Recompute KV-Cache             ~102ms      Stock
Text Conditioning              ~9ms        Stock
─────────────────────────────────────────────────
Total per frame (3 frames)     ~850ms      → ~19 FPS (compile)
```

---

## Summary

| Level | Description | B300 Status |
|-------|-------------|-------------|
| 1 | Use the tools | ✅ Done |
| 2 | Profile and understand | ✅ Done |
| 3 | Write custom kernels | ✅ Done (Triton) |
| 4 | Beat the libraries | ✅ Done (FA4/CUTE) |
| 5 | Fused operations | 🎯 **Next target** |
| 6 | Architecture-specific | Future |
| 7 | Novel algorithms | Research |
| 8 | Production at scale | N/A |

You're at **Level 4.5** - you've beaten the libraries on your specific workload, diagnosed platform issues, and have a working optimized system. The path to Level 5 (fused RoPE + Attention) is visible and achievable.
