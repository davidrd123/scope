# FA4 Backward Pass on SM100 (dQ, dK, dV)

> **Explainer #10** — How FA4 computes the backward pass on Blackwell (SM100): dQ/dK/dV, how LSE is used, and where `score_mod_bwd` / `mask_mod` fit.
> **Updated:** 2025-12-25

---

## Overview

The forward pass computes:

```
S = QK^T (scaled, biased, masked)
P = softmax(S)
O = P V
```

The backward pass needs gradients:

```
dQ, dK, dV given dO (and forward intermediates like LSE / row_max / row_sum)
```

FA4 has architecture-specific backward kernels. For SM100, the implementation lives in:

- `vendored/flash_attn_cute_score_mod/flash_attn/cute/flash_bwd_sm100.py`

This explainer is a starter: it lays down the “map” (which tensors, which phases, which warps) so a future deep dive can fill in the math and the exact loop structure.

---

## What forward outputs are needed for backward?

Most flash-attention backprop derivations use:

- `O` (output)
- `P` (softmax probabilities) *or* a way to reconstruct them
- `LSE` per row (log-sum-exp), or equivalently `row_max` + `row_sum`

In FA4/CuTe, LSE is written (optionally) during/after the forward correction loop and is used by backward kernels to avoid recomputing normalization.

See explainer #7 for how LSE relates to online softmax:
- `notes/FA4/explainers/07-online-softmax.md`

---

## Kernel structure (SM100 backward)

`FlashAttentionBackwardSm100` defines:

- Tile sizes (`tile_m`, `tile_n`)
- Warp specialization roles
- TMEM layout/offsets for intermediate accumulators (SM100 uses TMEM heavily)

At a high level, backward needs to compute:

1. `dV = P^T dO`
2. `dP = dO V^T`
3. `dS = dP ⊙ softmax_grad(S)` (involves row-wise sums and LSE)
4. `dQ = dS K`
5. `dK = dS^T Q`

The kernel pipelines loads of Q/K/V/dO and reductions for dQ/dK/dV.

---

## Warp specialization (observed in code)

In `flash_bwd_sm100.py`, the backward kernel allocates 16 warps (512 threads), with roles like:

- reduce warps (dQ accumulation reduction)
- compute warps (main math for dP/dS and partial dQ/dK/dV)
- MMA warp (tcgen05 issue)
- load warp (TMA/cp.async loads)
- epilogue warp (writeback)
- empty warp (spare / scheduling)

The exact grouping differs from forward, but the principle is the same:
overlap load/compute/reduce/writeback.

---

## Where `score_mod_bwd` and `mask_mod` fit

Forward customization:
- `score_mod` modifies scores before softmax
- masks turn invalid entries into `-inf`

Backward must be consistent:

- If `score_mod` changed `S`, backward must incorporate the derivative of the modifier where applicable.
- If `mask_mod` masked entries, backward must not propagate gradients through masked entries.

SM100 backward exposes:
- `score_mod_bwd` (a backward-side companion to `score_mod`)
- `mask_mod` (used for sparse/structured masking cases)

This is likely one of the hardest “correctness” zones when extending FA4: custom score logic must have the right backward behavior or you silently train the wrong model (not our main use-case today, but still important for completeness).

---

## Why we care (even for inference)

Even though our current project is inference-first:

- The forward path often writes LSE and supports backward metadata.
- Understanding backward constraints helps avoid breaking assumptions when we patch vendored FA4/CuTe code (e.g. changing what gets written or when).

---

## Questions & Opportunities

1. What forward intermediates does our current build actually write (LSE on/off), and does that affect backward availability?
2. How does `score_mod_bwd` relate to `apply_score_mod_bwd_inner` in `softmax.py`?
3. If we extend FA4 with a hypothetical `q_mod/k_mod` hook (for RoPE fusion), what backward changes would be required?

---

## References

- `vendored/flash_attn_cute_score_mod/flash_attn/cute/flash_bwd_sm100.py`
- `vendored/flash_attn_cute_score_mod/flash_attn/cute/flash_bwd_sm90.py`
- `vendored/flash_attn_cute_score_mod/flash_attn/cute/softmax.py`
- Explainer #7: `notes/FA4/explainers/07-online-softmax.md`

