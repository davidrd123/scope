# Split-K and Segment-Combine in FA (and why `score_mod` helps)

> **Explainer #11** — Why some attention backends “split” the KV dimension (Split-K), what “segment combine” is, and how this relates to our KV-bias workload (Flash vs FA4 score_mod).
> **Updated:** 2025-12-25

---

## Overview

In an ideal world, attention is one fused kernel:

```
O = softmax(QK^T) V
```

In practice, implementations sometimes break this into “segments” along the KV axis (N blocks) and combine partial results.

This explainer is meant to connect:

- The conceptual reason Split-K exists (parallelism / occupancy / cache behavior)
- What “segment combine” means at the math level
- Why our KV-bias pattern can make certain backends pay more combine overhead
- Why FA4/CuTe `score_mod` is attractive: it lets us express KV-bias **inside** the main kernel, avoiding some extra combine glue

---

## Split-K: The idea

Instead of one program computing a full `(M_block × N_total)` attention accumulation, you can:

1. Split the KV range into `S` segments
2. Compute partial outputs for each segment independently
3. Combine them into the final output

This can improve throughput when:

- N is large and you want more parallel CTAs.
- Some tiles are imbalanced (causal patterns, varlen).
- You want better L2 reuse patterns by restricting each CTA’s working set.

---

## Segment combine: what gets combined?

Softmax makes combining non-trivial.

For a query row `i`, define a segment `s` producing:

- `m_s = max_j S_ij over j in segment s`
- `l_s = sum_j exp(S_ij - m_s) over j in segment s`
- `o_s = sum_j exp(S_ij - m_s) * V_j over j in segment s`  (unnormalized output numerator)

To combine segments, you need the global max:

```
m = max_s m_s
```

Then:

```
l = sum_s l_s * exp(m_s - m)
o = sum_s o_s * exp(m_s - m)
O = o / l
```

That is “segment combine”: rescale each segment’s partial sums by `exp(m_s - m)` and then normalize.

This looks a lot like online softmax (explainer #7), but at the granularity of “segments” rather than “tiles inside one CTA”.

---

## Where this shows up in our workload

Our streaming KV-bias attention has two backend-dependent stories:

- **Flash path**: implements KV-bias via a more indirect route (segment-ish handling / combine glue), which can make “KV-bias compute” a larger fraction of self-attention time.
- **FA4 score_mod path**: applies bias inside the core kernel, so the bias logic does not require extra segment combine work.

We’ve observed this empirically in profiling: the “KV-bias slice” is smaller when the bias is expressed via FA4 `score_mod` than when it’s expressed via Flash segment-combine style logic.

This doc is the missing conceptual bridge that explains *why* those profiles look different.

---

## What this explainer should eventually include (Phase 2 work)

To make this explainer “complete”, we should pin down:

1. Which exact FlashAttention code path we use for “flash backend” in our repo when KV-bias is enabled.
2. Where the segment/split happens (API flags / internal dispatch).
3. Where the combine kernel runs and what tensors it reads/writes.
4. Why that combine overhead grows (or becomes more visible) for our shapes.

The most useful deliverable would be:

- A minimal pseudo-callgraph: “front door → forward kernel → (optional) combine kernel”
- A tensor lifecycle diagram of what gets written to global memory in the split/segment path

---

## Questions & Opportunities

1. Can we keep the bias logic entirely within the main kernel across all GPUs (SM90/SM100/SM103)?
2. For SM103 specifically, what are the safe fallbacks when `score_mod` isn’t available?
3. Is it ever worth using split-K for our exact streaming pattern, or is it always overhead?

---

## References

- Explainer #3: `notes/FA4/explainers/03-score-mod.md`
- Explainer #7: `notes/FA4/explainers/07-online-softmax.md`
- `vendored/flash_attn_cute_score_mod/flash_attn/cute/interface.py` (dispatch surface)

