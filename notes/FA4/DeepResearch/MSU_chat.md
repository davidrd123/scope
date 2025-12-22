# 5pro

Below is a practical “on-ramp” doc for building the custom kernel that the talk slide is hinting at (specialized block-autoregressive attention kernels), with enough grounding in what Krea Realtime is actually doing, why FlexAttention is currently required, and what concrete kernel target(s) would let you remove it.

---

# Replacing FlexAttention in Krea Realtime: Kernel Research Notes and Build Plan

## 0) Why this is a real, high leverage target

Krea Realtime 14B’s long-form stability relies on two inference-time techniques:

1. **KV cache recomputation with block-causal masking**
   They periodically recompute the KV cache for the sliding context window, and they do it with a **block-causal mask**. ([Krea][1])

2. **KV cache attention bias**
   During sampling, they apply a **negative attention bias on past-frame tokens**, while leaving current-frame self-attention unbiased. ([Krea][1])

In their own writeup they explicitly say both of these currently force them onto **PyTorch FlexAttention**, and they call out a “huge gap” versus kernels like **FlashAttention 4** or **SageAttention**, plus additional overhead from first-frame re-encoding. ([Krea][1])

So the “custom kernel” project is not speculative. It is directly aligned with their identified bottleneck: “we need this quality trick, but FlexAttention is expensive.” ([Krea][1])

## 1) The exact mask you are trying to accelerate

Krea’s causal structure is not “token causal” inside the block. It is:

* **Bidirectional inside a block**
* **Causal between blocks**

They define it plainly: “All frames within a block attend to each other; all frames attend to all past frames.” ([Krea][1])

That means the attention mask is block-lower-triangular at the block level, but **each block is dense**.

### Key observation for kernel design

If your tokens are ordered by time (frames) and frames are grouped into blocks of `F` frames, and each frame has `S` tokens, then:

* Block index of token `i` is: `b(i) = floor(i / (F*S))`
* The block-causal rule is: allow attention iff `b(k) <= b(q)`

That is an extremely structured sparsity pattern.

It is not “arbitrary masking,” it is “prefix-of-blocks.”

That structure is the opening to write something faster than FlexAttention, because a specialized kernel can avoid per-element mask logic and avoid “mixed block” handling for most tiles.

## 2) What FlexAttention is buying you, and why it is slow here

FlexAttention is PyTorch’s general “programmable attention” mechanism: you provide a mask and/or score modification and it generates a kernel that honors it, often using **BlockMask** to exploit block-sparsity. ([PyTorch Documentation][2])

It is powerful, but generality costs you:

* mask classification and handling partial tiles
* overhead from score modification paths
* worse kernels than FA4/FA3 in many real inference shapes

There is precedent that for specific sparsity patterns, a custom Triton implementation can outperform FlexAttention by a lot. ([GitHub][3])

So the core play is: keep the *semantics* (mask + bias), but replace the *engine*.

## 3) Two kernel targets, not one

Think of this as two deliverables. They can be done independently, and each removes one FlexAttention dependency.

### Kernel A: Block-causal attention for KV recomputation (no bias)

Krea recomputes KV cache “by passing clean latent frames through the model with a block-causal mask.” ([Krea][1])

This is the nastier one because it needs the block-causal mask at scale.

### Kernel B: Attention with “past-frame negative bias” (for sampling)

They apply “negative attention bias to tokens of past frames during sampling,” and they say this also requires FlexAttention. ([Krea][1])

This one may be simpler because in sampling you often only compute attention for the newly generated block against a KV cache, so the mask side is simpler. The differentiator is “additive score bias per key token group.”

If you implement Kernel B first, you can often switch most sampling passes back onto FA4/FA3-style fast paths plus your bias.

## 4) The most promising implementation paths

You basically have three routes. I would plan them in this order because it maximizes “time to first win.”

### Route 1: Win without a new kernel by exploiting structure (fastest to test)

Goal: reduce FlexAttention overhead by ensuring the block mask is almost entirely “full or empty tiles,” not partial.

The “Sliding Tile Attention” writeup (different project, same lesson) calls out how **mixed blocks** (partially masked tiles) cause a lot of overhead and require extra logic. ([Hao AI Lab][4])

For block-causal masking, partial tiles mainly come from **frame boundaries not aligning to your kernel tile sizes**.

Concrete tricks:

* **Pad `S` tokens per frame** up to a multiple of your KV/Q tile sizes so that inter-frame boundaries align to tiles.
* Ensure “block boundary” aligns to a whole number of tiles.

If you can get to “most tiles are fully valid or fully invalid,” FlexAttention’s BlockMask path gets much closer to its best case.

Also, for mask creation overhead, use direct BlockMask construction ideas rather than expensive generic mask building. vLLM documents building BlockMask more directly (for its own attention) as a performance strategy. ([PyTorch Documentation][5])

This route is not the final solution, but it can buy you time and give you profiling clarity.

### Route 2: Triton custom kernel for block-causal prefix-of-blocks (portable across GPUs)

This is the classic “I want control, I want it to run on Hopper and Blackwell, I want to iterate fast.”

Your block-causal rule `b(k) <= b(q)` implies that for a query tile, valid keys are a contiguous prefix of the K sequence, with the cutoff determined by the query tile’s block index.

So the kernel can:

1. Compute `q_block = floor(q_start / (F*S))`
2. Compute `k_end = min(Lk, (q_block+1)*F*S)`
3. Loop K/V tiles from 0 to `k_end` only
4. Do a streaming softmax (max, sumexp) and accumulate output

No per-element mask needed inside the inner loop for almost all tiles.

Then for bias (Kernel B), the kernel just adds:

* bias = `0` if key token in current block
* bias = `-beta` if key token in past blocks

That bias is piecewise constant by block, so you can apply it at tile granularity most of the time.

This is exactly the kind of specialization FlexAttention cannot assume.

### Route 3: FlashAttention “CUTE DSL” as the fastest path on B200 (SM100)

There is an explicit precedent for using FlashAttention’s CUTE DSL path to implement FlexAttention-like behavior. Colfax has a guide literally titled along the lines of “FlexAttention in FlashAttention CUTE DSL.” ([Colfax Research][6])

This matters because Krea specifically calls out the performance gap versus FA4. ([Krea][1])

If you are targeting B200 performance, this is likely the endgame:

* implement the mask and score mods in the FlashAttention CUTE ecosystem
* keep FA4-class performance characteristics

Even if you still use FlexAttention-style APIs at the Python level, you can swap out the backend.

## 5) What to measure first (so you do not optimize the wrong thing)

Before writing kernels, isolate two timings:

1. **Sampling step attention time** (with bias enabled)
2. **KV recomputation attention time** (with block-causal mask)

Krea calls recomputation expensive and gives a complexity discussion; it is easy for it to dominate once you are generating long sequences. ([Krea][1])
They also say the bias requires FlexAttention and has significant overhead. ([Krea][1])

So you want a profiler breakdown like:

* % time in flex_attention forward
* how often recomputation happens per generated frames
* whether bias path or recompute path is the bigger tax in your target “Scope streaming” workload

## 6) Concrete “first kernel” spec (what you actually implement)

### Inputs you will need at runtime

* `F`: frames per block
* `S`: tokens per frame
* `block_bias_beta` (float, maybe per step)
* `Lk`: total key length (context tokens)
* mapping from token index → block id (but you can compute block id via integer division if layout is consistent)

### Kernel A behavior (recompute)

For each query token `q`:

* allow keys `k` iff `floor(k/(F*S)) <= floor(q/(F*S))`

No bias.

### Kernel B behavior (sampling bias)

For each query token in current block:

* allow all keys in cache plus current block (usually no “future” exists in this call)
* add bias:

  * 0 for keys in current block
  * `-beta` for keys in past blocks

This matches the description: “negative attention bias to tokens of past frames during sampling, while keeping unmodified attention in the self-attention to the tokens of the current frame.” ([Krea][1])

## 7) How to avoid the biggest hidden trap: “mixed tiles”

This is the performance killer in most masked attention kernels.

If `S` is not tile-aligned, then some tiles straddle frame boundaries, meaning a tile contains keys from two different blocks. Then you need per-element logic inside a tile, which is expensive.

Two practical approaches:

* **Padding approach:** pad tokens per frame to next multiple of your tile size (common choice: 128). Then the block boundaries line up with tile boundaries.
* **Layout approach:** reorder tokens so that frames are contiguous and aligned, even if you pad.

The Sliding Tile Attention discussion makes the general point: GPU efficiency comes from masks that avoid mixed blocks and allow async pipelining. ([Hao AI Lab][4])

Your mask is simpler than STA. You can take advantage of that.

## 8) Where PyTorch BlockMask options may still help

Even if you stay inside FlexAttention for a while, PyTorch’s BlockMask kernel options exist specifically to let the runtime assume more structure, for example contiguity of blocks.

Your block-causal pattern is contiguous at the block level, so if you do not exploit that, you are leaving performance on the table.

This is not “the kernel replacement,” but it is a good interim improvement and also a way to validate your understanding of the block structure.

## 9) Suggested build sequence for fastest progress

### Step 1: Reproduce the mask and bias in a tiny harness

* Generate random Q/K/V tensors with the same shapes as your real run (capture them from a single forward pass in Krea or Scope).
* Implement reference outputs using FlexAttention.
* Lock dtype and scaling (BF16 or FP16).

### Step 2: Implement Kernel B first (bias only)

Why:

* It removes FlexAttention from the per-step sampling hot path.
* It is conceptually simpler than recomputation.

If you get this right, you likely get a visible FPS win without touching recomputation yet.

### Step 3: Implement Kernel A (block-causal recompute)

This is the bigger win for long sequences, because recomputation is called out as expensive but necessary. ([Krea][1])

### Step 4: Add “alignment mode” (padding)

Add an option to pad tokens so block boundaries align to tiles. Measure the speedup. Keep it if it is a net win.

### Step 5: Backend dispatch

* Hopper (H100): Triton kernel
* Blackwell (B200): FlashAttention CUTE DSL kernel (or Triton if good enough)

Krea’s own positioning is that B200 is their headline perf target (11 fps at 4 steps). ([Krea][1])
So a B200-optimized backend is especially credible.

## 10) If you want the “research punchline”

Krea has already told you the bottleneck and why: block-causal KV recomputation and attention bias both require FlexAttention today, and FlexAttention is materially slower than FA4/Sage. ([Krea][1])

The kernel you want is not “general masking,” it is a specialized, structured attention:

* block-causal prefix-of-blocks masking for recomputation
* piecewise-constant past-frame bias for sampling

That structure is exactly what lets a custom Triton kernel (or a FlashAttention CUTE DSL extension) beat FlexAttention, consistent with broader evidence that specialization can outperform FlexAttention for specific patterns. ([GitHub][3])

---

## If you tell me one thing, I can make this even sharper

If you paste just one profiler snippet (or even just the Q/K/V shapes for the slowest attention call, plus `F`, `S`, and context length `N`), I can help you choose tile sizes and decide whether “padding to avoid mixed tiles” will be a big win or a waste.

No need for full code, just shapes and which phase (sampling bias vs recomputation).

[1]: https://www.krea.ai/blog/krea-realtime-14b "Krea Realtime 14B: Real-Time, Long-Form AI Video Generation"
[2]: https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html?utm_source=chatgpt.com "torch.nn.attention.flex_attention"
[3]: https://github.com/pytorch/pytorch/issues/141129 "[FlexAttention] Block sparse mask is significantly slower than sdpa and handmade triton implementation  · Issue #141129 · pytorch/pytorch · GitHub"
[4]: https://hao-ai-lab.github.io/blogs/sta/ "Fast Video Generation with Sliding Tile Attention | Hao AI Lab @ UCSD"
[5]: https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html "torch.nn.attention.flex_attention — PyTorch 2.9 documentation"
[6]: https://research.colfax-intl.com/a-users-guide-to-flexattention-in-flash-attention-cute-dsl/ "A User’s Guide to FlexAttention in FlashAttention CuTe DSL – Colfax Research"


# Me

Do you want to give me a set of a guide or a rundown of Cute DSL? And I'll be targeting the B200 only. But, yeah, I'll be working on the profiler tomorrow. Yeah, so, and then just the merits and relative ease of the Custom Tracking Kernel or the Fast Attention Cute DSL. And so once I start working toward it. So reproduce the mask and bias in a tiny harness. So if I, looking at step two, implement kernel B only, just what might that look like? Would you advise trying in Triton or Cute DSL first?

# 5pro

Yes. I’m going to assume by “Qt DSL” you mean **FlashAttention’s CUTE DSL path** (the SM100 Blackwell backend built on CUTLASS/CUTE), not the Qt UI framework.

Since you are **B200-only**, you have a rare advantage: you can bias your whole approach toward the fastest SM100-native path without worrying about portability.

## Quick rundown of FlashAttention CUTE DSL for B200

### What it is

* **CUTE** is NVIDIA’s C++ template system for describing tiled tensor layouts and MMA pipelines.
* FlashAttention uses a **CUTE DSL backend** to generate high-performance attention kernels, with a dedicated SM100 path for Blackwell.
* The user-facing entry points are in `flash_attn.cute.interface`, plus internal classes like `FlashAttentionForwardSm100`. ([Colfax Research][1])

### Why it matters for your problem

Krea Realtime uses FlexAttention today because it needs:

* a **custom mask** (block-causal KV recomputation), and
* a **custom score modification** (negative bias on past-frame tokens),
  and FlexAttention is the mechanism that lets you express those kinds of variants. The cost is that FlexAttention is slower than FA4-class kernels. ([PyTorch][2])

CUTE DSL gives you a middle path:

* still express mask_mod / score_mod (FlexAttention semantics),
* but run it on a FlashAttention-style kernel backend (CUTE). ([Colfax Research][1])

### Practical gotchas on B200

* The CUTE SM100 backend is evolving and can hit assertion limits related to SM100 TMEM capacity for certain shapes. ([GitHub][3])
* Some users report that CUTE interface return values (like LSE) are not always available in the way the “vanilla” interface is. ([GitHub][4])

Those two facts strongly shape your decision about “Triton vs CUTE first.”

---

## Triton vs CUTE DSL: merits and relative ease

Here is the honest trade:

### Option 1: Triton custom kernel

Best for:

* fastest iteration
* easiest debugging
* “I want a working kernel in a harness quickly”

Pros

* Python-level dev loop, no C++ extension build.
* Easy to add exactly your bias rule and early-exit structure.
* Great for proving correctness and plumbing into Scope.

Cons

* Likely not FA4-level performance on B200 for all shapes.
* You become responsible for tuning, numerics, and edge cases.

### Option 2: CUTE DSL (FlashAttention CUTE path)

Best for:

* maximizing performance on SM100
* reusing the FlashAttention scheduling and math pipeline

Pros

* Highest ceiling on B200.
* You can express your bias as a `score_mod` and keep everything inside the FlashAttention machinery. ([Colfax Research][1])

Cons

* More build friction (flash-attn from source, cutlass-dsl).
* Debugging is harder than Triton.
* Some instability and shape limitations are still showing up in the issue tracker for SM100. ([GitHub][3])

### Option 3: “FlexAttention with CUTE backend” (the stealth best first move on B200)

This is worth calling out explicitly:

If you can get **FlexAttention to dispatch to the CUTE backend**, you might get 80 percent of the win without writing a full custom kernel. The Colfax guide shows exactly this concept: define `score_mod` / `mask_mod`, then run through the CUTE DSL interface, and it also describes a Torch-integrated route using `torch.compile(flex_attention)` with options that force the Flash/CUTE backend. ([Colfax Research][1])

This is extremely aligned with your “kernel B only” goal because kernel B is fundamentally just a score modification.

---

## What kernel B is, in concrete terms

Kernel B is: attention during sampling where keys from “past frames” get a constant negative bias, and “current frame” keys do not.

Mathematically (per head):

* logits = QKᵀ * scale + bias(k)
* bias(k) = 0 if key token is in current block, else -beta

You are not changing Q/K/V shapes. You are changing the score computation and softmax weighting.

This makes kernel B dramatically easier than kernel A (block-causal recompute), because it is:

* no complicated mask topology (you usually attend to everything in the cache anyway)
* a simple piecewise constant score shift

---

## What implementing “kernel B only” might look like

You have three practical implementations. Because you are B200-only, I’d think of them as a ladder.

### Path B1: Implement kernel B as a CUTE DSL score_mod (recommended first on B200)

This is the smallest amount of “kernel work” with the highest performance upside.

Concept:

* Keep using attention normally
* Inject a CUTE DSL `score_mod` that adds `-beta` when `kv_idx < cutoff`

The Colfax article shows what a CUTE DSL `score_mod` looks like, how it operates on `TensorSSA`, and how you pass it into `_flash_attn_fwd`. ([Colfax Research][1])

What your score_mod needs:

* `cutoff` = first key index of the current block (so keys < cutoff are “past”)
* `beta` as a scalar
* return `score - beta` for past keys, else `score`

Important performance note:

* This is a cheap score_mod because it depends only on indices and constants, not aux tensors.
* CUTE can vectorize and broadcast this kind of modification more efficiently than “arbitrary bias lookup.” ([Colfax Research][1])

Risk:

* You need to confirm your environment can compile and run the SM100 CUTE attention for your exact head_dim, seqlens, and GQA config without hitting SM100 TMEM assertions. ([GitHub][3])

### Path B2: Implement kernel B as a dedicated CUTE kernel variant (best performance, most work)

This is “edit FlashAttention CUTE forward” territory:

* Add parameters `cutoff`, `beta`
* Apply the bias in the score pipeline
* Keep everything else identical

You do this if:

* score_mod overhead is still too high
* you need absolute peak B200 throughput
* you are ready to live inside flash-attn’s kernel code

This can be very fast, but it is real kernel engineering work.

### Path B3: Triton kernel B (best for correctness and integration speed)

You implement a fused attention forward for the sampling phase:

* Streaming softmax over K tiles
* Add a constant bias per tile based on its key range

A clean trick here: because the bias is piecewise constant, you can apply it at tile granularity for most tiles. Only “boundary tiles” that straddle cutoff need per-element branching, and you can reduce those by padding or alignment.

---

## So, Triton or CUTE first?

Given your constraints (B200-only) and your stated goal (replace FlexAttention because it is slow), I’d recommend a staged approach:

### Recommendation

1. **Try CUTE DSL score_mod first (B1).**

   * It is the fastest route to “FlexAttention semantics with FlashAttention-style performance” on SM100. ([Colfax Research][1])
2. If you hit shape limitations or instability (TMEM assertions, missing return values), fall back to **Triton kernel B (B3)** to keep moving.
3. If Triton works but is not fast enough, come back and do **dedicated CUTE kernel variant (B2)**.

This sequence keeps you from spending days tuning Triton only to later realize you could have gotten most of the win with a score_mod running on the CUTE backend.

---

## Minimal harness for kernel B (what to build first)

You want a tiny “truth harness” that you can run outside Scope and outside Krea.

Inputs:

* q: [B, Hq, Lq, D]
* k,v: [B, Hkv, Lk, D]
* cutoff: int (index separating past vs current keys)
* beta: float

Reference implementation:

* compute logits = q @ k.T * scale
* add bias: logits[:, :, :, :cutoff] -= beta
* softmax + matmul with v

Then compare against:

* your CUTE score_mod path, or Triton kernel path

Correctness checks:

* max abs error
* relative error
* check sensitivity at different beta values (beta small and beta large)

---

## One subtle but important detail about “bias only”

Because the bias is a constant shift applied to a subset of keys, it changes the relative mass between the past and current sets.

That means you cannot approximate it by:

* masking, or
* post-scaling the output

It must be applied at the logits stage, before softmax.

This is why kernel B is a good first target: it is a pure “score modification” feature.

---

## Actionable next steps for you when you profile

When you profile tomorrow, capture these exact facts for the slowest FlexAttention call:

* B, Hq, Hkv
* Lq and Lk at the moment of sampling
* D (head_dim)
* dtype (bf16 vs fp16)
* whether you are doing varlen (cu_seqlens) or fixed
* cutoff policy (how you define “current block” boundary)

With just those, you can immediately decide:

* whether CUTE can handle your shape cleanly
* what tile sizes make sense if you go Triton
* whether padding to avoid “mixed cutoff tiles” is worth it

If you paste the shapes and how you define cutoff, I can sketch the exact CUTE score_mod logic you need (index comparison, scalar-to-ssa handling) and also the exact Triton kernel structure for the same semantics.

[1]: https://research.colfax-intl.com/a-users-guide-to-flexattention-in-flash-attention-cute-dsl/ "A User’s Guide to FlexAttention in FlashAttention CuTe DSL – Colfax Research"
[2]: https://pytorch.org/blog/flexattention/?utm_source=chatgpt.com "FlexAttention: The Flexibility of PyTorch with ..."
[3]: https://github.com/Dao-AILab/flash-attention/issues/1959 "[cute, fwd, sm100] Issue on B200 cute forward with Gemma3 · Issue #1959 · Dao-AILab/flash-attention · GitHub"
[4]: https://github.com/Dao-AILab/flash-attention/issues/1782 "cute.interface _flash_attn_fwd fucntion doesn't provide LSE tensor in the return value · Issue #1782 · Dao-AILab/flash-attention · GitHub"

# Me

Okay, so if path B1 works, what might be the range of improvement? If previously when I was kind of profiling during a run of CRIA real-time, it looked like FlexiTension was a bottleneck compared to FlashTension4, and then like FlexiTension accounted for like 60 to maybe 65 percent of the total time.

# 5pro

You can get a pretty tight bound on this with Amdahl’s law, and it also makes clear what numbers you need from tomorrow’s profiler run.

Let:

* **p** = fraction of total step time currently spent in FlexAttention (you saw ~0.60 to 0.70)
* **s** = speedup factor of “that attention work” after switching to a CUTE DSL score_mod path (B1)

Then the end-to-end speedup is:

[
\text{Speedup} = \frac{1}{(1-p) + \frac{p}{s}}
]

## 1) Hard ceiling, even with a perfect kernel

If you made the FlexAttention portion infinitely fast, the best you can do is:

[
\text{Max Speedup} = \frac{1}{1-p}
]

So:

* If **p = 0.60**, max is **2.5x**
* If **p = 0.65**, max is **2.86x**
* If **p = 0.70**, max is **3.33x**

That’s the “nothing else matters” upper bound.

## 2) Realistic range if B1 turns FlexAttention into “FA4-ish with a tiny tax”

The whole question becomes: what is **s** in practice?

You mentioned “FlexAttention bottleneck compared to FlashAttention4.” In many real cases, getting off FlexAttention can be worth something like **1.5x to 4x** on the attention portion depending on shape, masking complexity, and whether you were hitting mixed tiles.

Here are end-to-end speedups for common scenarios.

### If p = 0.60

* s = 1.5 → speedup = 1 / (0.40 + 0.60/1.5) = 1 / (0.40 + 0.40) = **1.25x**
* s = 2 → 1 / (0.40 + 0.30) = **1.43x**
* s = 3 → 1 / (0.40 + 0.20) = **1.67x**
* s = 4 → 1 / (0.40 + 0.15) = **1.82x**

### If p = 0.65

* s = 1.5 → 1 / (0.35 + 0.433) = **1.28x**
* s = 2 → 1 / (0.35 + 0.325) = **1.48x**
* s = 3 → 1 / (0.35 + 0.217) = **1.76x**
* s = 4 → 1 / (0.35 + 0.163) = **1.95x**

### If p = 0.70

* s = 1.5 → 1 / (0.30 + 0.467) = **1.30x**
* s = 2 → 1 / (0.30 + 0.35) = **1.54x**
* s = 3 → 1 / (0.30 + 0.233) = **1.88x**
* s = 4 → 1 / (0.30 + 0.175) = **2.11x**

So if your experience is “FlexAttention is dramatically slower than FA4,” the realistic end-to-end range to keep in mind is often:

* **~1.5x to ~2.1x overall** (if attention portion improves 2x to 4x and p is 0.6 to 0.7)

And the absolute ceiling is **2.5x to 3.33x** depending on p.

## 3) One crucial caveat: B1 only helps if it replaces most of the FlexAttention time you measured

Earlier we framed B1 as “Kernel B only,” meaning “sampling-time attention with bias.” But Krea also uses FlexAttention for KV recomputation with block-causal masking. If your profiled 60 to 70 percent includes both:

* sampling attention with bias, and
* recomputation attention with mask,

then B1 might only remove a subset of that p.

Example: suppose FlexAttention is 70 percent total time, but only half of that is “bias attention” that B1 replaces. Then the replaced fraction is p’ = 0.35, not 0.70.

If p’ = 0.35:

* s = 3 gives speedup = 1 / (0.65 + 0.35/3) = 1 / (0.65 + 0.117) = **1.30x**
* s = 4 gives speedup = 1 / (0.65 + 0.0875) = **1.36x**

So you really want to know tomorrow:

* What fraction of total time is spent in “bias attention” calls vs “recompute mask attention” calls?

That tells you whether B1 is a huge win by itself or just a meaningful first step.

## 4) What I’d expect from B1 specifically

B1 is appealing because your “bias” is piecewise constant. In the best case it behaves like:

* “FA4 attention” plus
* a very cheap index compare and add (or tile-level compare and add)

So in many shapes the overhead could be small relative to the cost of QK and softmax, meaning B1 might get you close to FA4 throughput. But you should assume some tax until proven, because score_mod paths can sometimes inhibit certain fusion tricks.

The right way to quantify s fast is a microbenchmark:

* Run one representative attention call under:

  1. FlexAttention implementation of the bias
  2. CUTE DSL score_mod implementation of the same bias
  3. Plain FA4 without bias (for an upper bound)

Then compute:

* s1 = Flex / CUTE-scoremod
* s2 = CUTE-scoremod / plain-FA4 (the overhead tax)

## 5) Converting that to FPS intuition

If your current FPS is `fps_old`, then:

[
fps_{new} \approx fps_{old} \times \text{Speedup}
]

So if you are at 10 FPS and you hit:

* 1.5x overall, you go to ~15 FPS
* 2.0x overall, you go to ~20 FPS

That’s why it is worth doing, especially if p is really 0.65 to 0.70.

## What to capture in tomorrow’s profiler to make this precise

For each FlexAttention hotspot, record:

* total self time
* call count per second
* shapes: B, heads, Lq, Lk, head_dim
* whether the call is “bias sampling” or “recompute block-causal”
* dtype
* whether varlen is used

Then we can compute:

* p_bias, p_recompute separately
* the best-case impact of B1 alone
* whether your next kernel should be “recompute mask” instead

If you paste a single profiler table with those call sites and percentages, I can plug it into the formula and give you a tight expected improvement band (and identify whether B1 is your biggest lever or whether recomputation will dominate after you fix bias).
