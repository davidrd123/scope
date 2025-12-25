# B300 Optimization Ladder: Where We Are & What's Above

> **Context:** The hierarchy of GPU optimization work, mapped to the B300 journey.
> **Updated:** 2025-12-25

**Learning-first note:** This is intentionally a *learning ladder*, not just a perf plan. It’s totally fine to bounce between rungs as long as we keep writing down (1) what we tried, (2) what we measured, and (3) what we learned.
Use `notes/FA4/b300/experiments.md` as the default place to capture those “one-change” experiment cards.

**Ground truth sources (when numbers disagree):**
- `notes/FA4/b300/session-state.md` (current reproducible best-known configs + caveats)
- `notes/FA4/b300/investigation-runbook.md` (how we measure and decide)
- Recent drilldown artifacts under `outputs/` (e.g. `outputs/b300_*_drilldown_*.log/json`)

---

## The Ladder

### Level 1: Use the Tools
*"I called the library"*

- Use PyTorch ops, maybe torch.compile
- Call FlashAttention, hope it works
- Accept whatever FPS you get

**Example:** "I ran the default stack on B300 and got ~8.8 FPS at `320x576`."

---

### Level 2: Profile and Understand
*"I know where the time goes"*

- Build profiling infrastructure
- Identify bottlenecks with data
- Understand the gap between expectation and reality

**What we did:**
- Added `PROFILE_PIPELINE_BLOCKS`, `PROFILE_ATTENTION` infrastructure
- Established that (a) the repo-default stack is decode/cuDNN-bound on B300, and (b) on a cu130 stack, decode improves and the transformer becomes the bottleneck
- Measured block-level shares (representative cu130 runs are usually “denoise dominates; decode second; recompute is non-trivial” — exact % varies by config and whether profilers are enabled)
- Measured that the **KV-bias kernel is only a slice of self-attn time**, and its share depends on backend (rule of thumb: `flash` segment-combine can be ~35–40% of `self_attn`, while `fa4` score_mod often drops it closer to ~20–30%); the remainder is dominated by QKV projection + other attention work + glue ops

**Example:** "Profiling showed VAE decode was 760ms on cu129 - that's the bottleneck"

---

### Level 3: Write Custom Kernels
*"I wrote code that runs on the GPU"*

- Triton kernels that work correctly
- Beat naive implementations
- Understand thread blocks, shared memory, tiling

**What we did:**
- Wrote Triton Kernel B for KV-bias attention
- In some microbenchmarks, beat flex_attention (useful learning milestone)
- Learned Triton autotuning (BLOCK_M, BLOCK_N, warps, stages)
- Learned the hard lesson that **microbench wins can evaporate end-to-end** if the backend is forced into a fallback path (on SM103 + current Triton, the “triton KV-bias backend” can be catastrophically slow)

**Example:** "My Triton kernel handles the KV-bias pattern correctly and beats PyTorch's flex_attention"

---

### Level 4: Bend the Libraries
*"I extended a fast library to fit a weird pattern"*

- Understand library constraints (supported masks, varlen, score mods, hardware arch support)
- Use the “fast path” primitives (FlashAttention/FA4/CuTe) but adapt them to our requirements (KV-bias + streaming + caching)
- Build pragmatic guardrails (version detection, backend selection, fallbacks) so the system is usable, not just “fast in one script”

**What we did:**
- Integrated **FA4/CUTE `score_mod`** for the KV-bias path so we don’t pay segment-combine overhead
- On recent B300 cu130 runs, KV-bias dropped from ~`0.9ms/call` (flash segment-combine) to ~`0.4ms/call` (FA4 score_mod)
- Diagnosed that the biggest “B300 mystery” wasn’t attention at all — it was the runtime stack (cuDNN/conv3d decode behavior) and backend selection hazards on SM103

**Example:** "I used FA4’s fast kernel, but customized it to directly support KV-bias, and then verified it actually moves end-to-end numbers."

---

## ✅ YOU ARE HERE: Between Level 4 and 5

Current state:
- B300: **~15 FPS** in typical cu130 end-to-end runs, with **best-case ~19 FPS** in the benchmark when `--compile` is viable (config-dependent)
- Solved the “white whale” mystery: the original ~8.8 FPS wasn’t an attention backend cap; it was dominated by runtime stack + decode/cuDNN behavior and SM103 backend pitfalls
- Have documented, reproducible optimizations

What's missing for Level 5:
- RoPE is still separate from attention, and attention-adjacent glue (casts/copies/layout fixes) is still sizable
- Not exploiting Blackwell-specific features (TMA, warp specialization)
- torch.compile integration is partial (some modes abort on SM103)

---

### Level 5: Fuse Adjacent Work (RoPE + Attention Glue)
*"Fewer launches and less memory traffic"*

**Important framing:** FlashAttention / FA4 already fuse the *core attention math* (QKᵀ → online softmax → P×V) into a single kernel.  
When we say “Level 5 fusion” here, we mostly mean fusing **the extra per-step work around attention** (RoPE, layout/cast/copy glue, small elementwise ops) so we reduce kernel launches and global-memory round-trips *inside the transformer block*.

Current shape (conceptual; there are often extra glue kernels in between):
```
QKV projection (GEMM)
RoPE(Q/K) (separate kernel today)      → ~0.11ms/call (B300 cu130 drilldown example)
Attention (already fused math)         → KV-bias portion ~0.41ms/call (FA4 score_mod example)
Layout/cast/copy glue                  → often shows up as `aten::copy_` / `aten::to`
Output projection (GEMM)
```

Level 5 target (realistic near-term fusion goals):
```
RoPE applied during Q/K load/prologue (no separate RoPE kernel)
Attention outputs produced in the most downstream-friendly layout/dtype we can manage
Fewer intermediate stores/loads and fewer “glue” kernels
```

**Why it matters:** Memory bandwidth is often the bottleneck, not compute. Each kernel launch reads from and writes to global memory. Fusion eliminates intermediate round-trips.

**Reality check (important):** on recent B300 cu130 profiles, **RoPE is already relatively small** (example: ~`0.11ms/call`). So “fuse RoPE into attention” is a great *learning* project, but it may not be a huge FPS lever by itself. The bigger remaining chunk is typically the “other_in_self” part of self-attn (QKV projection + non-bias attention work + glue ops).

**Also important:** “Fuse output projection into attention” is a different class of work (fused MHA / fused linear layers) and is *not* the default shape of FA/FA4 kernels. Treat that as a stretch topic, not the first Level 5 milestone here.

**How to get there:**
- FA4/CUTE supports customization hooks (prologue/load/epilogue style patterns)
- If we fuse RoPE, it’s by rotating **Q/K vectors during load/prologue** (before the dot-product), not by score_mod (score_mod changes scores, RoPE changes vectors)
- Expect lots of “try/measure/learn” iterations; the output of each iteration should be a small lesson + a reproducible benchmark

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

**Effort:** Medium-to-high (this can turn into real R&D depending on how deep we go)
**Impact:** Unknown; likely modest unless we also reduce memory traffic / conversions / projections

Steps:
1. Pick a *single* representative benchmark (canonical `320x576`, bias `0.3`, cu130 env) and record the baseline
2. Identify the “why” for fusion (e.g. reduce kernel launches, reduce `aten::copy_`, remove redundant transposes)
3. Prototype a minimal fusion target (e.g. “RoPE during Q/K load”) and validate correctness first
4. Benchmark, then write down the lesson (what moved, what didn’t, what got harder)

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

Representative recent cu130 drilldown signal (for intuition; always re-measure):

- Block-level time is dominated by `denoise`, then `decode`, then `recompute_kv_cache` (see `outputs/b300_cu130_*_drilldown_blocks_profile.json`)
- Inside the transformer (example `outputs/b300_cu130_none_bias0.3_drilldown_perf.log`):
  - `self_attn` is the largest bucket
  - KV-bias share depends on backend (roughly ~35–40% of `self_attn` on `flash`, ~20–30% on `fa4`); **the remaining “other_in_self” is still the majority**
  - `rope_apply` is present but relatively small in recent runs

Takeaway: after the cu130 decode fix + FA4 KV-bias, the next wins are more likely to come from reducing the remaining self-attn overhead (QKV projection / non-bias attention / data movement) and expanding safe `torch.compile` coverage, rather than “RoPE fusion alone.”

---

## Summary

| Level | Description | B300 Status |
|-------|-------------|-------------|
| 1 | Use the tools | ✅ Done |
| 2 | Profile and understand | ✅ Done |
| 3 | Write custom kernels | ✅ Done (Triton) |
| 4 | Bend the libraries | ✅ Done (FA4/CUTE) |
| 5 | Fused operations | 🎯 Next target (as learning) |
| 6 | Architecture-specific | Future |
| 7 | Novel algorithms | Research |
| 8 | Production at scale | N/A |

You're at **Level 4.5**: you’ve diagnosed platform issues, have a working optimized stack, and have enough instrumentation to safely explore deeper optimizations. Level 5 (fusion) is doable as a learning project, but the right mindset is “small experiments + clean measurements + write down lessons,” not “guaranteed FPS.”
