# Paged KV in FA4/CUTE (When `use_tma_KV=False`)

> **Explainer #9** — What “paged KV” means in FA4, how it loads K/V without TMA, and why this path exists.
> **Updated:** 2025-12-25

---

## Overview

FA4’s SM100 forward kernel has two broad KV loading strategies:

1. **TMA KV** (preferred when possible): TMA loads directly into shared memory.
2. **Paged KV** (non-TMA): K/V are stored in pages and loaded via `cp.async`-style tiled copies.

You can see this reflected in the forward kernel’s warp roles:

- Normal TMA: `load_warp_ids = (14,)`, `empty_warp_ids = (15,)`
- Paged KV: KV loading uses more producer resources (often `load_warp_ids = (14, 15)`), because it’s not TMA-driven.

Key sources:
- `vendored/flash_attn_cute_score_mod/flash_attn/cute/paged_kv.py`
- `vendored/flash_attn_cute_score_mod/flash_attn/cute/flash_fwd_sm100.py`
- `vendored/flash_attn_cute_score_mod/flash_attn/cute/copy_utils.py`

---

## What “Paged KV” is (Conceptually)

Paged KV is an inference-oriented memory layout for long KV caches:

- K/V are stored in fixed-size “pages” (chunks).
- A per-sequence **page table** maps logical token indices → physical pages.
- The kernel loads the required pages for each `n_block` into shared memory.

This is common when:

- You need to manage KV cache growth without reallocating large contiguous buffers.
- You want to reuse the same paged KV layout across multiple kernels.

---

## The PagedKVManager (Code Map)

`paged_kv.py` defines a `PagedKVManager` dataclass that encapsulates:

- `mPageTable`: the per-batch mapping from logical page index → page id
- `mK_paged`, `mV_paged`: the paged K/V storage tensors
- `page_size_divmod`: fast division/mod (token index → (page_idx, page_offset))
- tiled copy objects (`gmem_tiled_copy_KV`, `gmem_thr_copy_KV`) using `cpasync.CopyG2SOp`

The control flow is roughly:

1. `load_page_table(n_block)`: for rows in the N tile, compute which page each row belongs to and cache `(page, offset)` in registers.
2. `load_KV(n_block, sX, K_or_V)`: for each row, issue async copies from the correct page into shared memory.

---

## load_page_table: logical token → (page, offset)

For each “row” of the N tile:

1. Compute `row_idx = n_block * n_block_size + row`
2. Compute `page_idx, page_offset = divmod(row_idx + leftpad_k, page_size)`
3. Read `page = page_table[page_idx]` (if in bounds)
4. Store `page` and `page_offset` into small register tensors for reuse during the load loop

This is a key pattern: compute address mapping once, then reuse it for K and V loads.

---

## load_KV: loading into shared memory

`load_KV`:

- Partitions the shared-memory destination tensor (`sX`) to a 2D view.
- Uses a tiled copy atom based on `cpasync.CopyG2SOp`.
- For V, it “finesses” the layout to transpose V into the layout expected by the MMA/softmax path.

Important detail (from the code):
- For V out-of-bounds lanes, it may explicitly clear/zero-fill smem to avoid consuming garbage in later stages.
- For K, it often relies on masking on the score side (so clearing is not required).

---

## Why this matters for our project

Even if our current workload uses contiguous KV (not paged KV), this explainer matters because:

- The “non-TMA KV path” is a real-world fallback path that can show up depending on configuration.
- The “extra load warps” behavior can change occupancy and interact with our own backend selection/perf expectations.
- If we ever adopt paged KV caching for streaming video, this becomes directly relevant.

---

## Questions & Opportunities

1. Under what conditions does FA4 disable TMA KV (and why)?
2. How does paged KV interact with:
   - split-K (segment combine)
   - varlen scheduling
   - block-sparse masks
3. What are the perf tradeoffs (L2 behavior, bandwidth, staging) between TMA KV vs paged KV?

---

## References

- `vendored/flash_attn_cute_score_mod/flash_attn/cute/paged_kv.py`
- `vendored/flash_attn_cute_score_mod/flash_attn/cute/flash_fwd_sm100.py`
- `vendored/flash_attn_cute_score_mod/flash_attn/cute/copy_utils.py`
- Explainer #5: `notes/FA4/explainers/05-tma-memory-loading.md`
- Explainer #6: `notes/FA4/explainers/06-tile-scheduling.md`

