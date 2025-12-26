# FA4 Backward Pass on SM100 (dQ, dK, dV)

> **What this is:** A code-grounded explainer for FA4’s Blackwell (SM100) backward: how the kernel computes `dQ`, `dK`, `dV`, what extra forward-side stats it needs, and exactly where `mask_mod` / `score_mod_bwd` plug in.
> **Updated:** 2025-12-25

---

## Overview

Forward computes:

```
S = QK^T
logits = scale_and_optional_score_mod(S)
logits = apply_masks(logits)
P = softmax(logits)
O = P V
```

The backward pass needs gradients:

```
dQ, dK, dV given dO (and forward intermediates like LSE / row_max / row_sum)
```

FA4 has architecture-specific backward kernels. For SM100, the implementation lives in:

- `vendored/flash_attn_cute_score_mod/flash_attn/cute/flash_bwd_sm100.py`

This explainer follows the code: it explains the exact tiles, warp roles, pipelines, and the places where custom hooks are applied.

---

## The 3-kernel backward pipeline (interface → preprocess → main → postprocess)

In this repo, `_flash_attn_bwd` (see `vendored/.../cute/interface.py`) runs three steps:

1. **Preprocess:** compute “row stats” needed for `dS` cheaply
   - `dPsum = (O * dO).sum(dim=-1)` (per query row)
   - `lse_log2 = lse * log2(e)` (convert natural-log LSE to log2 domain)
   - zero out `dq_accum` (float32 accumulator buffer)

   This is why the SM100 backward kernel subtracts `LSE` directly in an `exp2(...)` computation: it is not the raw forward `lse`, it is already `lse_log2`.

2. **Main backward (SM100):** compute
   - `dK`, `dV` directly (or into fp32 accumulators for GQA)
   - `dq_accum` (float32) using a specialized reduction path

3. **Postprocess:** cast/scale accumulators into final outputs
   - `dq = cast(dq_accum) * softmax_scale`
   - for GQA (`qhead_per_kvhead > 1`), also cast/scale `dk_accum` and `dv_accum`

The SM100-specific kernel described below is step (2): `FlashAttentionBackwardSm100` in `flash_bwd_sm100.py`.

---

## Core math (what the kernel computes)

The backward pass can be expressed as:

```
dP = dO V^T
dS = P ⊙ (dP - dPsum)          # dPsum = sum_j dP_ij * P_ij = dot(dO_i, O_i)

dV = P^T dO
dK = dS^T Q
dQ = dS K
```

Two important “implementation facts” from this repo:

- The kernel **reconstructs `P`** from `(S, lse_log2)` (it does not materialize `P` from forward).
- The kernel uses `dPsum` as a provided input (computed in preprocess), so it never needs a per-row reduction over `dP ⊙ P` inside the main kernel.

For scaling:

- If there is no `score_mod`, `S` is the unscaled dot product and softmax uses `S * softmax_scale`.
- The kernel computes `dS` in “logits space” and applies `softmax_scale` later:
  - `dK` is scaled at store time for MHA, or during postprocess for GQA.
  - `dQ` is scaled during postprocess (`dq_accum → dq`).

---

## Tile schedule: KV-major (outer loop is `n_block`)

`FlashAttentionBackwardSm100` schedules tiles by `n_block` (KV blocks):

- The tile scheduler produces work items keyed by `(n_block, head_idx, batch_idx, ...)`.
- For each `n_block` tile, the kernel loops over all contributing `m_block` tiles.

This matches the outputs:

- `dK` and `dV` are naturally written per KV tile (`n_block` major).
- `dQ` is naturally written per Q tile (`m_block` major), so the kernel writes into an intermediate `dq_accum` buffer and uses a separate reduction path.

Also note: SM100 backward currently asserts “no varlen” in `flash_bwd_sm100.py` (`mCuSeqlens*` / `mSeqUsed*` are not supported in this kernel yet).

---

## Warp specialization (SM100)

The kernel uses 16 warps (512 threads) with fixed roles (see `FlashAttentionBackwardSm100.__init__`):

- **Reduce warps:** `reduce_warp_ids = (0, 1, 2, 3)` (reduce/store `dq_accum`)
- **Compute warps:** `compute_warp_ids = (4..11)` (turn `S → P`, compute `dS`, apply hooks/masks)
- **MMA warp:** `mma_warp_id = 12` (UMMA GEMMs into TMEM)
- **Load warp:** `load_warp_id = 13` (TMA loads of Q/K/V/dO/LSE/dPsum)
- **Epilogue warp:** `epi_warp_id = 14` (currently mostly a helper / placeholder)
- **Empty warp:** `empty_warp_id = 15` (parked)

The interesting part is how these roles communicate: SM100 heavily uses TMEM and pipeline barriers to overlap the phases.

---

## Pipelines + TMEM reuse (why the code is structured this way)

### TMEM allocation and overlap

The kernel explicitly reuses TMEM buffers (see offsets in `__init__`):

- `S` and `P` share TMEM (`tmem_S_offset == tmem_P_offset == 0`)
- `dP`, `dQ`, and `dS` share TMEM (`tmem_dP_offset == tmem_dQ_offset == tmem_dS_offset`)
- `dV` and `dK` each have their own TMEM regions

This is essential because SM100 TMEM is capacity-limited (512 columns).

### The key “handshake” pipelines

Inside `kernel(...)`, the code creates several pipelines. The ones that define the overall algorithm are:

- `pipeline_S_P`: MMA warp produces `S`/`P` in TMEM, compute warps consume it
- `pipeline_dP`: MMA warp produces `dP` in TMEM, compute warps consume it
- `pipeline_dS`: compute warps produce `dS` (smem and/or TMEM), MMA warp consumes it
- `pipeline_dKV`: MMA warp produces `dV`/`dK` in TMEM, compute warps run the epilogue stores
- `pipeline_dQ`: MMA warp produces `dQ` accumulations in TMEM, reduce warps consume and reduce to `dq_accum`

The result is a “ping-pong” structure:

- UMMA writes → async threads read/transform → UMMA consumes the transformed result → async threads store.

---

## Walkthrough: what each warp group actually does

This section is the concrete map from `flash_bwd_sm100.py`.

### 1) Load warp (`load_warp_id == 13`): bring Q/K/V/dO + row stats into smem

The load warp uses TMA pipelines to stage:

- `Q` + `K` together for the first `m_block` (the code uses `extra_tx_count` so one barrier covers both)
- `LSE_log2` (from preprocess) in `pipeline_LSE`
- `dO` + `V` similarly, and `dPsum` in `pipeline_dPsum`

In the dense case, it preloads the first `m_block`, then streams the rest of `m_block_min+1 .. m_block_max`.

### 2) MMA warp (`mma_warp_id == 12`): the GEMM engine

The MMA warp defines GEMMs that match the backward equations, but with a key transpose:

- **Score computation uses `S = K @ Q.T`**, so the score tile is `(tile_n, tile_m)` (KV rows, Q cols).

The GEMMs in the `mma(...)` method correspond to:

- `mma_qk_fn`: `S = K @ Q.T` → writes `tStS` (TMEM)
- `mma_dov_fn`: `dP = V @ dO.T` → writes `tdPtdP` (TMEM)
- `mma_pdo_fn`: `dV = P.T @ dO` → accumulates `tdVtdV` (TMEM)
- `mma_dsq_fn`: `dK += dS.T @ Q` → accumulates `tdKtdK` (TMEM)
- `mma_dsk_fn`: `dQ = dS @ K` → writes `tdQtdQ` (TMEM)

The prologue does:

1. compute first `S`
2. compute first `dP`
3. wait for `P` (from compute warps) and compute first `dV`

Then the main loop iterates over the remaining `m_block`s, repeatedly:

- compute next `S`
- consume `dS` (from compute warps) and compute `dK`/`dQ`
- compute next `dP`
- wait for `P` and update `dV`

The ordering of `dK` vs `dQ` depends on `use_smem_dS_for_mma_dK` (a deterministic+causal optimization): if `dS` is in smem for one path and in TMEM for another, the kernel swaps the order to keep correctness.

### 3) Compute warps (`compute_warp_ids == 4..11`): `S → P` and `dP → dS`

This is `compute_loop(...)`. For each `(n_block, m_block)`:

1. **Read `S` from TMEM → RMEM**
2. **Optionally apply `score_mod`**, then **apply mask** (same order as forward)
3. **Compute `P = exp(S - LSE)`** and write `P` back to TMEM
4. **Read `dP` from TMEM**, subtract per-row `dPsum`, multiply by `P` to form `dS`
5. **Optionally apply `score_mod_bwd`** to transform gradients back to “pre-mod scaled score” space
6. Write `dS` to smem (and optionally TMEM) and signal `pipeline_dS` so the MMA warp can consume it

Masking uses the backward-specific helper:

- `AttentionMask(..., swap_AB=True)`
- `mask.apply_mask_sm100_transposed(...)`

This matches the transpose convention in backward (`S = K @ Q.T`), and it’s also where block-sparse “full block vs partial block” shortcuts are handled (`is_full_block`, `check_m_boundary`).

### 4) Reduce warps (`reduce_warp_ids == 0..3`): reduce/store `dq_accum`

The MMA warp produces `dQ` fragments into TMEM. The reduce warps consume those fragments via `pipeline_dQ` and perform a bulk reduce-add into global `dq_accum`.

This is implemented in `dQacc_reduce(...)` and uses:

- TMEM→RMEM loads
- staging into smem
- `copy_utils.cpasync_reduce_bulk_add_f32(...)` into the `dq_accum` buffer

Deterministic mode uses semaphores (`mdQ_semaphore`) to enforce ordering.

### 5) Epilogue: store `dK` and `dV` (and scale as needed)

For MHA, the epilogue stores `dV` and `dK` from TMEM to global memory. In the non-TMA store path, `epilogue_dKV(...)` explicitly multiplies `dK` by `softmax_scale` before casting.

For GQA, the kernel uses fp32 accumulators and the Python interface postprocesses them (including scaling `dK` by `softmax_scale`).

---

## `score_mod` / `score_mod_bwd` / `mask_mod` (exactly how they fit)

### Forward-side `score_mod` inside backward

Even in backward, the kernel may need to apply the forward `score_mod` because it reconstructs `P` from `S`.

In `compute_loop`, when `score_mod` is present:

- It saves a copy of the **unscaled** score tile (`tSrS_pre`) before applying anything.
- It calls `apply_score_mod(...)`, which uses `apply_score_mod_inner(..., transpose_indices=True)` so that
  the callback sees `(q_idx, kv_idx)` in the same convention as forward even though the score tile is transposed.

### `mask_mod`

Masking is applied after `score_mod`, matching forward ordering:

- `mask.apply_mask_sm100_transposed(..., mask_mod=self.mask_mod, mask_causal=..., mask_local=...)`

This is the same “masking backend split” discussed in explainer #8, but using the backward-specific transposed masking function.

### `score_mod_bwd`

If `score_mod` changes the logits, backward must map gradients back to the “pre-mod” score space. That’s what `score_mod_bwd` is for.

In `compute_loop`, after computing `dS = P ⊙ (dP - dPsum)`, the code does:

- `apply_score_mod_bwd_inner(grad_tensor=dS, score_tensor=tSrS_pre, transpose_indices=True)`

Per the helper’s docstring (`softmax.py`), this transforms `dlogits` in place into `d(scaled_scores)` consistent with the forward `score_mod` graph.

---

## Why we care (even for inference)

This project is inference-first, but the backward explainer is still useful because:

- The same “row stats” concepts (row max / row sum / LSE) are what make **segment-combine** correct (Explainer #11).
- Some “small” forward changes (e.g. whether LSE is written, or how scores are modified) can silently break the assumptions that backward relies on.
- If we ever want to upstream a customization (like a score modifier) into a training setting, **`score_mod_bwd` correctness is non-negotiable**.

In other words: you don’t need backward to ship FPS improvements, but you *do* want to avoid “patching the kernel into a corner”.

---

## Questions & Opportunities

1. SM100 backward currently asserts no varlen (`cu_seqlens_*` / `seqused_*`). What would it take to thread varlen through scheduler + masking paths?
2. `pack_gqa` is forced off on SM100 bwd in the interface today. Is that fundamental or just “not wired up yet”?
3. What forward intermediates does our build actually write (LSE on/off), and does that affect backward availability or segment-combine plumbing?
4. How does `score_mod_bwd` relate to `apply_score_mod_bwd_inner` in `softmax.py` (and what are the minimal safe `score_mod_bwd` patterns for additive biases)?
5. If we ever extend FA4 with a hypothetical `q_mod/k_mod` hook (for RoPE fusion), what backward changes would be required?

---

## References

- `vendored/flash_attn_cute_score_mod/flash_attn/cute/flash_bwd_sm100.py`
- `vendored/flash_attn_cute_score_mod/flash_attn/cute/flash_bwd_sm90.py` (contrast: different arch, different constraints)
- `vendored/flash_attn_cute_score_mod/flash_attn/cute/interface.py` (preprocess/postprocess + `lse_log2` / `dPsum`)
- `vendored/flash_attn_cute_score_mod/flash_attn/cute/softmax.py` (`apply_score_mod_inner`, `apply_score_mod_bwd_inner`)
- `vendored/flash_attn_cute_score_mod/flash_attn/cute/mask.py` (`apply_mask_sm100_transposed`)
- Explainer #8: `notes/FA4/explainers/08-masking-and-mask_mod.md`
- Explainer #7: `notes/FA4/explainers/07-online-softmax.md`
