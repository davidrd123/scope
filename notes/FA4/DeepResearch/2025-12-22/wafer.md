You’re doing the *right kind* of work here: you’ve turned “we need a custom kernel” into (1) **precise semantics**, (2) **real shapes**, (3) **microbench harness**, (4) **integration + feature flag**, and (5) **profiling-driven reprioritization**. That’s the whole game in kernel land.

Below are the main takeaways and what I’d change / do next.

---

## What’s strongest about the journey so far

### 1) The A/B split was the correct architectural move

You separated:

* **Kernel A** = block-causal recompute (mask structure problem)
* **Kernel B** = KV-cache attention with piecewise-constant bias (score_mod problem)

…and you didn’t “cargo cult” the idea that they’re equally valuable. You pivoted once the data showed where the time is.

### 2) You hit the “padding trap” early and fixed it

The observation that your `Lq=4680` bias benchmark was effectively doing *9472* work because of padding is exactly the kind of silent benchmark distortion that wastes weeks if you don’t catch it.

Adding `--pad-q-to-k/--no-pad-q-to-k` was a very high-signal change.

### 3) You learned the real lesson behind FlexAttention’s advantage

Your Kernel A attempt basically re-derived a core truth:

> “Runtime masking in a generic loop rarely beats compile-time block sparsity / structured skipping.”

Your “skip tile” attempt getting worse is *expected* because Triton can’t actually “early break” a `tl.range()` loop the way you want; it still executes the loop body and you pay divergence + predication costs.

That is exactly why FlexAttention + BlockMask is hard to beat on structured sparsity unless you also get **compile-time skipping** (multi-kernel specialization or block-sparse iteration).

### 4) You shipped: Kernel B integrated, feature-flagged, measured

This is the big one: you didn’t stop at “microbench win,” you:

* made it default
* kept a fallback
* pinned a B200 config
* checked correctness on real shapes
* validated real-world FPS movement

That’s the right bar.

### 5) Your profiling writeup is honest and actionable

The “nested timers double-count” note is important, and your fine-grained breakdown is the first time your roadmap becomes *inevitable* instead of speculative.

---

## The one thing I’d tighten immediately: benchmark consistency

There’s a small “smell” in the log: the **flex bias baseline** is reported as:

* ~0.779 ms in the earlier no-pad run (Iteration 5),
* but later Kernel B says flex is 1.144 ms (same nominal shape).

That doesn’t mean anything is wrong, but it’s a sign that **one of these is not comparable** (different kernel selected, different dtype, different padding, different env flags, different warmup discipline, different “steady state” criteria, etc.).

What I’d do (without adding tons of process):

* Make a single command that prints a JSON blob:

  * `{git_sha, torch_ver, triton_ver, cuda_ver, dtype, B,H,D,Lq,Lk,padded_q,padded_k, warmup_iters, iters, mean_ms, p50_ms, p95_ms}`
* Always record both:

  * **kernel-only** timing (just the attention kernel wrapper)
  * **callsite timing** (what causal_model actually does around it)

That will stop “phantom wins” from creeping in.

---

## Interpreting your profiling: what it *really* says

Your fine-grained `self_attn` breakdown is the most important section in the whole doc:

* `self_attn_kv_bias`: **27.4% of self_attn**
* `qkv_projection`: **20.8%**
* `rope_apply`: **15.8%**
* everything else smaller

Two immediate implications:

### A) Kernel B is worth optimizing further, but expectations should be Amdahl-based

A 10–12% speedup on something that’s ~27% of `self_attn` translates to only ~3% faster `self_attn` (ballpark).

So if you see 20% FPS swings in tiny clips, you’re probably also capturing:

* padding removal changing real FLOPs,
* reduced compile/autotune overhead,
* different warmup regimes,
* or pipeline scheduling effects.

Kernel B can still be a big deal (especially if FA4/CUTE is *much* faster), but now you have the lens to judge claims.

### B) Kernel A is *performance*-deprioritized, but may still matter for “dependency removal”

Your measured `p_recompute` is small enough that chasing Kernel A purely for speed doesn’t make sense.

But: if your real goal is “**no FlexAttention in the stack**,” then you still need *some* strategy for Kernel A even if it’s not faster.

That’s where your “two dense calls trick” becomes valuable: it’s not a performance play, it’s a **simplification / backend unification** play.

---

## What I would do next (in order)

### 1) Use Nsight Compute on Kernel B *now* (Wafer helps here)

Not because “profiling is good,” but because you need exactly one binary decision:

**Is Kernel B compute-bound (tensor core limited) or memory-bound (K/V bandwidth + softmax traffic)?**

From that, the next optimization path changes:

* If **compute-bound**:

  * the only real escape hatch is a more optimized backend (FA4/CUTE, wgmma-quality kernels).
* If **memory-bound**:

  * there may still be Triton headroom via better staging / block pointers / fewer rereads / better block shapes.

This is the point where Wafer (as an NCU report explorer) is actually useful.

### 2) Try to unlock FA4/CUTE for Kernel B, but treat it as a *measurement*, not a belief

Given your updated Amdahl picture, FA4/CUTE is only worth the hassle if it beats your Triton kernel by a meaningful factor.

So the goal isn’t “port Kernel B to CUTE.”
The goal is: **get one FA4/CUTE score_mod path running on the exact (Lq,Lk,H,D) shape and compare.**

If it’s only ~5–15% faster than Triton, you probably stop there.
If it’s ~1.5–2×, you have your next win.

### 3) RoPE: if you do Triton Phase 3, fuse Q+K in one kernel

Right now RoPE is paying:

* launch cost twice (Q and K),
* memory traffic twice,
* and it’s still separated from attention.

If you write a Triton kernel anyway, the highest-leverage *first version* is:

* one kernel that takes **two pointers** (Q and K) and applies the same cos/sin to both in one pass.

Even if the per-element math is identical, you cut kernel launch overhead and improve cache locality for the cos/sin loads.

(Then measure. If it’s not a win, you stop before doing the harder “3-way lookup in-kernel” work.)

### 4) Consider a “drop FlexAttention” plan that doesn’t require beating it

If you still want to remove FlexAttention dependency:

* **Kernel A via “two dense calls”** is the most pragmatic:

  * block 0: (4680×4680)
  * block 1: (4680×9360)

Route both to FA4.
Even if it’s slightly slower than Flex+BlockMask, the total end-to-end hit is tiny if Kernel A is ~3–4% of `self_attn`.

This is the kind of trade you can justify if packaging/maintainability matters.

### 5) Don’t ignore qkv_projection: it’s “quietly” huge

Since `qkv_projection` is ~21% of `self_attn`, you have two realistic options:

* **Backend upgrade** (if available in your stack):

  * more optimized GEMM path (cublasLt epilogues, TE kernels, etc.)
* **Fusion** (harder, bigger win):

  * fuse QKV projection + (maybe) RoPE into a single producer so Q/K don’t get written then reread.

Even small % gains there are meaningful because the call count is so high.

---

## Two specific improvements to your doc that will pay off later

### Add a “Golden Shape Table” that’s explicitly versioned

Right now you have great shape derivations, but I’d add a single canonical “these are the shapes we optimize for” list like:

* Resolution → tokens/frame
* (Lq,Lk) for bias path in steady-state
* dtype
* layout `[B,L,H,D]` vs `[B,H,L,D]`

…and label it “**DO NOT CHANGE WITHOUT UPDATING BASELINES**.”

### Make “perf wins” always report two numbers

For any change, report:

1. **microbench kernel time**
2. **in-situ callsite time** (inside causal_model path)

This will prevent you from over-optimizing kernels that are no longer the bottleneck because Python/dispatch/packing dominates.

---

## Bottom line

* The pivot away from Kernel A and toward Kernel B + RoPE/QKV was justified and *correct*.
* Kernel B being “only” ~10% faster is still a solid step, but your profiling says the next wins must come from:

  * **either** a much faster attention backend (FA4/CUTE-class)
  * **or** attacking the **non-attention** costs (QKV + RoPE), ideally with fusion or reduced memory traffic.
* Kernel A is only worth touching again if your objective is **eliminating FlexAttention**, not if the objective is speed.

If you paste just one Nsight Compute summary for **Kernel B (Triton) vs Flex bias path** (even just the top-line: achieved occupancy, SM throughput, DRAM/L2 throughput, register count), I can tell you whether Triton has real headroom left or whether CUTE/FA4 is the only path to a big jump.
