# Resources for Level 5-6 Optimization

> **Goal:** Map out what we have and what we need to reach Level 5 (fused operations) and Level 6 (Blackwell-specific features like TMA, warp specialization).
> **Updated:** 2025-12-25

**What this is:** a curated reading list + pointers for “Level 5/6” exploration.  
**What this is not:** the current performance truth (use `notes/FA4/b300/session-state.md`) or the measurement protocol (use `notes/FA4/b300/investigation-runbook.md`).  
When you try something, capture it as a 1-card experiment in `notes/FA4/b300/experiments.md`.

**Portability note:** avoid hardcoding `.venv/...` paths. Prefer repo-local sources (e.g. `vendored/flash_attn_cute_score_mod/`) or locate installed packages via Python.

**Reality check (so this doc stays useful):**
- The FA4 SM10x path we use for KV-bias already exercises “Level 6-ish” mechanisms (TMA/mbarriers/warp specialization/tcgen05+TMEM).
- The point of the “Level 5/6” workstream here is (a) to understand those mechanisms well enough to reason about profiles and failures, and (b) to apply similar thinking to the remaining hotspots (QKV/projections, glue copies, decode), or to future custom kernels.

---

## What We Already Have

### Local Resources (In This Repo)

| Resource | Location | Relevance |
|----------|----------|-----------|
| **ThunderKittens Blackwell blog** | `notes/research/2025-12-24/incoming/perf/blogs/thunderkittens-blackwell.md` | Level 6: TMA, CTA pairs, warp specialization in attention |
| **Warp Specialization blog** | `notes/research/2025-12-24/incoming/perf/blogs/warp-specialization.md` | Level 6: Triton's built-in warp spec (`num_consumer_groups`) |
| **tcgen05 for dummies** | `notes/research/2025-12-24/incoming/perf/blogs/gau-nerst-tcgen05.md` | Level 6: Low-level Blackwell tensor core programming |
| **FA4/CUTE source (vendored)** | `vendored/flash_attn_cute_score_mod/flash_attn/cute/` | Level 5/6: the FA4/CuTe code we’re actually editing/using for KV-bias |
| **FlashAttention RoPE implementation (installed)** | Locate via: `python -c "import flash_attn.layers.rotary as r; print(r.__file__)"` | Level 5: reference for rotary interfaces / tables |

### Start Here (Code-Grounded Reading)

- Blackwell/SM100 mental model: `notes/FA4/explainers/02-blackwell-path.md`
- Tile scheduling + pipelines: `notes/FA4/explainers/06-tile-scheduling.md`
- Online softmax (and where scale/bias interacts): `notes/FA4/explainers/07-online-softmax.md`
- Paged KV + KV-loading backends: `notes/FA4/explainers/09-paged-kv.md`

### Key FA4/CUTE Files to Study

```
vendored/flash_attn_cute_score_mod/flash_attn/cute/
├── flash_fwd_sm100.py      # SM100 (B200) forward pass; closest Blackwell reference (B300 is SM103)
├── flash_fwd.py            # General forward pass
├── blackwell_helpers.py    # SM100 helpers (useful conceptually; not guaranteed identical on SM103)
├── interface.py            # API entry point (score_mod plumbing lives here)
├── mma_sm100_desc.py       # MMA descriptors for SM100
├── softmax.py              # Online softmax implementation
├── tile_scheduler.py       # Tile scheduling / swizzling
└── pipeline.py             # Pipeline staging
```

---

## Level 5: Fused RoPE + Attention

### The Opportunity

**Important framing:** FA/FA4 already fuse the *core attention math* (QKᵀ → online softmax → P×V) into one kernel.  
When we talk about “Level 5 fusion” here, we mostly mean pulling **RoPE + attention-adjacent glue** (casts/copies/layout fixes) closer to (or inside) that kernel so we reduce launches and global-memory round-trips.

Current flow (conceptually, multiple kernels + global-memory round-trips):
```
QKV projection (GEMM)
RoPE(Q/K) (separate step today)
Attention (KV-bias path uses FA4 score_mod)
Layout/cast/copy glue (often shows as `aten::copy_` / `aten::to`)
```

Fused flow (aspirational):
```
Load Q,K → RoPE in registers → Attention → Store
```

**Reality check:** in recent B300 cu130 drilldowns, `rope_apply` is relatively small compared to the remaining self-attn overhead (QKV projection + “other_in_self”). So RoPE+attn fusion is a great *learning* target, but may not be a standalone FPS lever.

### Resources for Level 5

#### 1. FA's RoPE Implementation
```python
# Locate the installed file first:
#   python -c "import flash_attn.layers.rotary as r; print(r.__file__)"
# Then study the interface + implementation.
```

**What to look for:**
- How they apply rotary embeddings
- Whether there's a fused version
- How cos/sin tables are passed

#### 2. FA4/CUTE score_mod Mechanism
```python
# You're already using this for KV-bias!
# Important: score_mod modifies attention *scores* after the QK dot-product.
# It cannot implement RoPE (RoPE modifies Q/K vectors before the dot-product).
# If we fuse RoPE, it’s via a Q/K load/prologue hook (or a new q_mod/k_mod-style callback).
```

**Key insight from ThunderKittens blog:**
> "One important optimization turns out to be launching the AV MMA's from the *previous* iteration while starting the QK MMA of the *current* iteration, and loading the K and V tiles of the *next* iteration."

This suggests the pipeline has distinct phases where you could inject RoPE.

#### 3. CUTE Prologue/Epilogue
```python
# vendored/flash_attn_cute_score_mod/flash_attn/cute/flash_fwd.py
# vendored/flash_attn_cute_score_mod/flash_attn/cute/flash_fwd_sm100.py
# Look for: how tiles are loaded before attention
# The "prologue" is where you'd inject RoPE
```

**Search for:**
- `load_q`, `load_k`, `load_v` functions
- Any existing support for Q/K preprocessing
- Tile loading patterns

### Concrete Steps for Level 5

1. **Read `flash_attn/layers/rotary.py`** - Understand FA's RoPE interface
2. **Read `vendored/flash_attn_cute_score_mod/flash_attn/cute/interface.py`** - Look for prologue/callback hooks and how aux tensors are passed
3. **Read `vendored/flash_attn_cute_score_mod/flash_attn/cute/flash_fwd_sm100.py`** - Find where Q/K are loaded
4. **Prototype (safe):** use `score_mod` to validate callback plumbing (e.g. apply a tiny constant bias) and confirm it changes outputs as expected
5. **Prototype (real fusion):** fork FA4/CuTe and add an explicit Q/K preprocessing hook (conceptually `q_mod`/`k_mod`) that runs during load/prologue

### External Resources for Level 5

| Resource | URL | What It Teaches |
|----------|-----|-----------------|
| FlashAttention-3 paper | https://arxiv.org/abs/2407.08608 | Low-level optimizations, pipeline structure |
| Tri Dao's blog posts | https://tridao.me/ | Design decisions behind FA |
| CUTLASS Python DSL docs | https://github.com/NVIDIA/cutlass/tree/main/python | How CUTE DSL works |

---

## Level 6: Blackwell-Specific (TMA, Warp Specialization)

### The Opportunity

**Mental model (simplified):** Standard kernel execution
```
All warps do the same thing, take turns
Load → Compute → Store (sequential)
```

**Mental model (simplified):** Warp specialized
```
Warp 0-1: Producer (TMA loads, async)
Warp 2-5: Consumer (Tensor Core compute)
Warp 6-7: Output (store results)

All running simultaneously!
```

**In this repo:** FA4’s SM10x path already looks like this (but with more granular roles: softmax warps, correction warps, a load warp, etc.). The “Level 6” opportunity is less “does TMA exist?” and more “can we avoid fallbacks and tune the remaining bottlenecks without breaking correctness?”

### Key Blackwell Features

#### 1. TMA (Tensor Memory Accelerator)
From tcgen05 blog:
> "TMA can issue loads of arbitrary sizes, using only 1 thread. In PTX, TMA corresponds to `cp.async.bulk` (1D tile) and `cp.async.bulk.tensor` (1D to 5D tile) instructions."

**What it replaces:**
- Cooperative thread loads (`tl.load` in Triton)
- Manual address calculation

**Files to study:**
- `vendored/flash_attn_cute_score_mod/flash_attn/cute/flash_fwd_sm100.py` - Study how the kernel uses Blackwell-ish mechanisms
- `notes/research/2025-12-24/incoming/perf/blogs/gau-nerst-tcgen05.md` - Tutorial on TMA setup

#### 2. 5th Gen Tensor Cores (tcgen05)
From tcgen05 blog:
> "They seem to behave like 128×128 systolics. To get full FLOP utilization, you want M and N to be 128 (or larger multiples of 128)."

**Implication (rule of thumb):** tile sizes and major modes matter more on Blackwell; treat shape/tiling claims as “verify by measurement” rather than guarantees.

**Files to study:**
- `vendored/flash_attn_cute_score_mod/flash_attn/cute/mma_sm100_desc.py` - MMA descriptors
- `vendored/flash_attn_cute_score_mod/flash_attn/cute/blackwell_helpers.py` - SM100 helpers

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
- On SM103, some Triton/Inductor tcgen05 paths can hard-abort; treat warp-spec experiments as “learning first,” with an escape hatch back to stable configs (`notes/FA4/b300/session-state.md`).

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

2. **Read `vendored/flash_attn_cute_score_mod/flash_attn/cute/flash_fwd_sm100.py` carefully:**
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
├── Read vendored/flash_attn_cute_score_mod/flash_attn/cute/interface.py (focus on score_mod)
├── Trace how Q/K are loaded in vendored/flash_attn_cute_score_mod/flash_attn/cute/flash_fwd_sm100.py
└── Prototype: use score_mod to validate plumbing (e.g. tiny constant bias) and confirm it changes outputs

Week 2:
├── Fork vendored/flash_attn_cute_score_mod locally
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
| `vendored/flash_attn_cute_score_mod/flash_attn/cute/*.py` | Study the interface for adding Q/K preprocessing |
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
