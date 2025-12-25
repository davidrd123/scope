# Resources for Level 5-6 Optimization

> **Goal:** Map out what we have and what we need to reach Level 5 (fused operations) and Level 6 (Blackwell-specific features like TMA, warp specialization).
> **Updated:** 2025-12-25

---

## What We Already Have

### Local Resources (In This Repo)

| Resource | Location | Relevance |
|----------|----------|-----------|
| **ThunderKittens Blackwell blog** | `notes/research/2025-12-24/incoming/perf/blogs/thunderkittens-blackwell.md` | Level 6: TMA, CTA pairs, warp specialization in attention |
| **Warp Specialization blog** | `notes/research/2025-12-24/incoming/perf/blogs/warp-specialization.md` | Level 6: Triton's built-in warp spec (`num_consumer_groups`) |
| **tcgen05 for dummies** | `notes/research/2025-12-24/incoming/perf/blogs/gau-nerst-tcgen05.md` | Level 6: Low-level Blackwell tensor core programming |
| **FA4/CUTE source** | `.venv/lib/python3.12/site-packages/flash_attn/cute/` | Level 5: The code we're already using |
| **FA RoPE implementation** | `.venv/lib/python3.12/site-packages/flash_attn/layers/rotary.py` | Level 5: Reference for fused RoPE |

### Key FA4/CUTE Files to Study

```
flash_attn/cute/
├── flash_fwd_sm100.py      # 98KB - Blackwell forward pass (STUDY THIS)
├── flash_fwd.py            # 92KB - General forward pass
├── blackwell_helpers.py    # 30KB - SM100-specific helpers
├── interface.py            # 24KB - The API entry point
├── mma_sm100_desc.py       # 11KB - MMA descriptors for SM100
├── softmax.py              # 12KB - Online softmax implementation
├── tile_scheduler.py       # 26KB - How tiles are scheduled
└── pipeline.py             # 7KB  - Pipeline stages
```

---

## Level 5: Fused RoPE + Attention

### The Opportunity

Current flow (3 separate kernels):
```
RoPE(Q) → 0.38ms → global memory
RoPE(K) → 0.38ms → global memory
Load Q,K → Attention → 0.54ms
─────────────────────────────────
Total: 1.30ms, 3 kernel launches
```

Fused flow (1 kernel):
```
Load Q,K → RoPE in registers → Attention → Store
─────────────────────────────────────────────────
Target: ~0.60-0.70ms, 1 kernel launch
```

### Resources for Level 5

#### 1. FA's RoPE Implementation
```python
# .venv/lib/python3.12/site-packages/flash_attn/layers/rotary.py
# This is what FA already does for RoPE - study the interface
```

**What to look for:**
- How they apply rotary embeddings
- Whether there's a fused version
- How cos/sin tables are passed

#### 2. FA4/CUTE score_mod Mechanism
```python
# You're already using this for KV-bias!
# The question: can score_mod also do Q/K modification before attention?
```

**Key insight from ThunderKittens blog:**
> "One important optimization turns out to be launching the AV MMA's from the *previous* iteration while starting the QK MMA of the *current* iteration, and loading the K and V tiles of the *next* iteration."

This suggests the pipeline has distinct phases where you could inject RoPE.

#### 3. CUTE Prologue/Epilogue
```python
# flash_attn/cute/flash_fwd.py and flash_fwd_sm100.py
# Look for: how tiles are loaded before attention
# The "prologue" is where you'd inject RoPE
```

**Search for:**
- `load_q`, `load_k`, `load_v` functions
- Any existing support for Q/K preprocessing
- Tile loading patterns

### Concrete Steps for Level 5

1. **Read `flash_attn/layers/rotary.py`** - Understand FA's RoPE interface
2. **Read `flash_attn/cute/interface.py`** - Look for prelogue/callback hooks
3. **Read `flash_attn/cute/flash_fwd_sm100.py`** - Find where Q/K are loaded
4. **Prototype:** Try applying RoPE inside `score_mod` (even if semantically wrong) to see if the mechanism works
5. **Fork flash_attn/cute** - Add a `q_mod`/`k_mod` callback similar to `score_mod`

### External Resources for Level 5

| Resource | URL | What It Teaches |
|----------|-----|-----------------|
| FlashAttention-3 paper | https://arxiv.org/abs/2407.08608 | Low-level optimizations, pipeline structure |
| Tri Dao's blog posts | https://tridao.me/ | Design decisions behind FA |
| CUTLASS Python DSL docs | https://github.com/NVIDIA/cutlass/tree/main/python | How CUTE DSL works |

---

## Level 6: Blackwell-Specific (TMA, Warp Specialization)

### The Opportunity

Current: Standard kernel execution
```
All warps do the same thing, take turns
Load → Compute → Store (sequential)
```

Warp Specialized:
```
Warp 0-1: Producer (TMA loads, async)
Warp 2-5: Consumer (Tensor Core compute)
Warp 6-7: Output (store results)

All running simultaneously!
```

### Key Blackwell Features

#### 1. TMA (Tensor Memory Accelerator)
From tcgen05 blog:
> "TMA can issue loads of arbitrary sizes, using only 1 thread. In PTX, TMA corresponds to `cp.async.bulk` (1D tile) and `cp.async.bulk.tensor` (1D to 5D tile) instructions."

**What it replaces:**
- Cooperative thread loads (`tl.load` in Triton)
- Manual address calculation

**Files to study:**
- `flash_attn/cute/flash_fwd_sm100.py` - Already uses TMA
- `gau-nerst-tcgen05.md` - Tutorial on TMA setup

#### 2. 5th Gen Tensor Cores (tcgen05)
From tcgen05 blog:
> "They seem to behave like 128×128 systolics. To get full FLOP utilization, you want M and N to be 128 (or larger multiples of 128)."

**Implication:** Tile sizes matter more on Blackwell. 64×64 runs at 1/4 the rate of 128×128.

**Files to study:**
- `flash_attn/cute/mma_sm100_desc.py` - MMA descriptors
- `flash_attn/cute/blackwell_helpers.py` - SM100 helpers

#### 3. Warp Specialization in Triton
From warp-specialization blog:
```python
@triton.autotune(
    configs=[
        triton.Config({...},
            num_consumer_groups=2,      # Enable warp spec!
            num_buffers_warp_spec=3,
        ),
    ],
)
```

**This is available in Triton 3.2+ / PyTorch 2.6+**

**What to check:**
- Does our Triton version support this?
- Can we add it to our existing Triton kernels?

#### 4. CTA Pairs
From ThunderKittens blog:
> "Two CTAs can coordinate to execute tensor core instructions, accessing the Tensor Memory of *both* of the CTAs within the CTA pair."

This is more advanced - likely Level 7 territory.

### Concrete Steps for Level 6

1. **Audit our Triton version:**
   ```bash
   python -c "import triton; print(triton.__version__)"
   ```
   Need 3.2+ for warp specialization

2. **Read `flash_fwd_sm100.py` carefully:**
   - How does it use TMA?
   - How does it structure the warp groups?
   - What's the pipeline depth?

3. **Study ThunderKittens attention kernel:**
   - https://github.com/HazyResearch/ThunderKittens/blob/blackwell/kernels/attn/b200/b200.cu
   - This is the state-of-the-art for Blackwell attention

4. **Try Triton warp specialization:**
   - Add `num_consumer_groups=2` to our Triton kernel
   - Benchmark impact

5. **Profile TMA usage:**
   - Is our current FA4 path using TMA optimally?
   - `nsys` profiling could reveal TMA stalls

### External Resources for Level 6

| Resource | URL | What It Teaches |
|----------|-----|-----------------|
| **ThunderKittens B200 attention** | https://github.com/HazyResearch/ThunderKittens/blob/blackwell/kernels/attn/b200/b200.cu | Reference implementation |
| **ThunderKittens GEMM** | https://github.com/HazyResearch/ThunderKittens/tree/blackwell/kernels/matmul | Blackwell GEMM patterns |
| **tcgen05 repo** | https://github.com/gau-nernst/learn-cuda/tree/main/02e_matmul_sm100/ | Working Blackwell matmul |
| **Triton persistent GEMM tutorial** | https://github.com/triton-lang/triton/blob/release/3.2.x/python/tutorials/09-persistent-matmul.py | Warp spec in Triton |
| **NVIDIA PTX Guide - tcgen05** | https://docs.nvidia.com/cuda/parallel-thread-execution/#tensorcore-5th-generation-instructions | Official reference |
| **CUTLASS CTA Pair docs** | https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-cta-pair | CTA pair mechanism |

---

## Learning Path

### For Level 5 (Fused RoPE + Attention)

```
Week 1:
├── Read flash_attn/layers/rotary.py
├── Read flash_attn/cute/interface.py (focus on score_mod)
├── Trace how Q/K are loaded in flash_fwd_sm100.py
└── Prototype: RoPE inside score_mod (even if wrong)

Week 2:
├── Fork flash_attn/cute locally
├── Add q_mod/k_mod callbacks
├── Benchmark fused vs separate
└── Validate correctness
```

### For Level 6 (Blackwell-Specific)

```
Week 1:
├── Read ThunderKittens attention kernel (b200.cu)
├── Read tcgen05 blog in detail
├── Understand TMA + mbarrier pattern
└── Check Triton version for warp spec support

Week 2:
├── Profile current FA4 path with nsys
├── Identify TMA stalls or bubbles
├── Try Triton num_consumer_groups
└── Compare with ThunderKittens perf

Week 3+:
├── Consider ThunderKittens integration
├── Or: write custom CUDA kernel using tcgen05 patterns
└── This is research-level work
```

---

## Key Insight from ThunderKittens

The pseudocode from their blog is the blueprint for Level 6:

```cpp
// This is the structure of a Blackwell attention kernel
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
else {
    // Consumer: online softmax while signaling next AV
}
```

**Four things happening in parallel:**
1. QK matmul (current iteration)
2. AV matmul (previous iteration)
3. K load (next iteration)
4. V load (next iteration)

This is the "dataflow machine" model they describe.

---

## Summary: What to Pull In

### High Priority (Level 5)

| Resource | Action |
|----------|--------|
| `flash_attn/cute/*.py` | Study the interface for adding Q/K preprocessing |
| `flash_attn/layers/rotary.py` | Understand FA's RoPE implementation |
| Your existing `score_mod` code | Starting point for fused ops |

### Medium Priority (Level 6)

| Resource | Action |
|----------|--------|
| ThunderKittens b200.cu | Study as reference implementation |
| tcgen05 tutorial | Learn low-level Blackwell patterns |
| Triton 3.2 release notes | Check warp spec API |

### Nice to Have

| Resource | Action |
|----------|--------|
| FlashAttention-3 paper | Understand design decisions |
| CUTLASS DSL docs | Deeper CUTE understanding |
| nsys profiling | Find actual bottlenecks in current path |
