# DeepResearch → FA4/Krea kernel work (TL;DR + mapping)

Goal: replace Krea Realtime’s FlexAttention-heavy self-attn paths (block-causal recompute + KV-cache bias) with a custom kernel/back-end that is closer to FA4-class performance.

Sources
- [`MSU_chat.md`](MSU_chat.md) (kernel targets + suggested sequence)
- [`wafer.md`](wafer.md) (profiling workflow/tooling note)

## The relevant takeaways (actionable)

1) The “why”: FlexAttention is required today because Krea needs **(A)** block-causal KV recomputation masks and **(B)** a **score_mod** that biases down past-frame tokens during sampling; both are slower than FA4/Sage-class kernels.

2) Split the kernel project into two deliverables:
- **Kernel A (recompute):** block-causal attention (dense within block, causal across blocks), no bias.
- **Kernel B (sampling):** add a piecewise-constant logits bias for “past frames” vs “current block” (score_mod), typically with KV-cache shapes (often `Lq != Lk`).

3) Sequence recommendation (B200-first):
- Try **B1: “score_mod on a FlashAttention/CUTE backend”** first (closest to “FA4-ish + tiny tax” without writing a full custom kernel).
- Fall back to Triton for correctness/iteration speed if CUTE is unstable for target shapes.
- Do Kernel A after Kernel B, because recompute masks are the harder/uglier case.

4) Biggest performance trap: **mixed tiles / misalignment**
- Your mask/bias boundaries should align to kernel tile boundaries as much as possible.
- Padding/alignment can help eliminate mixed tiles, but it can also increase raw FLOPs; measure the trade.

5) Measure impact with Amdahl’s law, but only after splitting time by path:
- `p_bias`: fraction of time spent in “bias sampling” attention calls.
- `p_recompute`: fraction of time spent in recomputation (block_mask) attention calls.
- B-only work (Kernel B1/B3) only accelerates `p_bias`, not necessarily all of `p_bias + p_recompute`.

## How this maps onto the repo (what’s already aligned)

Existing hot paths
- `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py`
  - Block-causal recompute uses `flex_attention(..., block_mask=...)`
  - KV-cache bias uses `flex_attention(..., score_mod=...)`
  - Bias disabled (`kv_cache_attention_bias == 1.0`) falls back to `attention(...)` (FA4/FA2/SDPA path)

FA4 integration you already have
- `src/scope/core/pipelines/wan2_1/modules/attention.py`
  - Picks FA4 (CUTE) on SM100+ and guards tuple returns from CUTE interfaces.

Microbench harness (matches “build a tiny harness first”)
- `scripts/bench_blockwise_attn.py`
  - `--mode block_mask` ≈ recompute-style block-causal path
  - `--mode bias` ≈ sampling-time score_mod path
  - `--no-pad-q-to-k` is important to avoid hiding true `Lq` scaling (padding can dominate runtime)

Kernel prototyping (apprentice track)
- `scripts/triton_sdpa.py` + [`kernel-dev-log.md`](../kernel-dev-log.md)
  - M1–M3 correctness is already logged as PASS.
  - M4 (beating flex_attention) is running into exactly the DeepResearch warning: **runtime masking** can lose vs FlexAttention’s **compile-time block sparsity**.

## Concrete “next facts to capture” (so kernel work stays targeted)

From a single real Krea Realtime run, capture for the top attention callsites:
- `B, H, D`, dtype, `Lq` and `Lk`, and whether it’s recompute (`block_mask`) or bias (`score_mod`)
- Call counts per second for each path (`p_bias` vs `p_recompute`)
- For bias: exact cutoff definition (first frame excluded, current block excluded)

Profiling tooling note
- [`wafer.md`](wafer.md) is only about improving the Nsight Compute (`ncu`) loop; it doesn’t replace `nsys` / PyTorch profiler for end-to-end bottleneck attribution.

