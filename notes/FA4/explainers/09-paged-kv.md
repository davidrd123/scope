# Paged KV in FA4/CUTE (When `use_tma_KV=False`)

> **What this is:** A code-grounded explainer for FA4’s *paged KV* support on SM100, with a focus on the non‑TMA loading path (`use_tma_KV=False` / “paged_kv_non_tma”).
> **Updated:** 2025-12-25

---

## Overview

FA4’s SM100 forward kernel supports two *KV layouts* and two *KV loading backends*:

### KV layouts
- **Contiguous KV:** K/V are laid out as a single contiguous sequence (training-style).
- **Paged KV:** K/V are stored in fixed-size pages, and a per-sequence *page table* maps logical pages → physical pages (inference cache-style).

### KV loading backends (SM100)
- **TMA KV (`use_tma_KV=True`)**: loads K/V into smem using TMA.
- **Non‑TMA paged KV (`use_tma_KV=False`)**: loads K/V into smem using `cp.async` (`PagedKVManager`).

This explainer focuses on the *non‑TMA* paged KV backend, but it’s easiest to understand by also seeing where the TMA path fits.

Key sources:
- `vendored/flash_attn_cute_score_mod/flash_attn/cute/paged_kv.py`
- `vendored/flash_attn_cute_score_mod/flash_attn/cute/flash_fwd_sm100.py`
- `vendored/flash_attn_cute_score_mod/flash_attn/cute/copy_utils.py`
- `vendored/flash_attn_cute_score_mod/flash_attn/cute/interface.py`

---

## When do we hit the non‑TMA paged KV path?

In this repo, the flag that drives the backend choice is set in the Python interface:

- In `interface.py`, when `page_table is not None`, we read `page_size = k.shape[1]`.
- On SM100 we construct `FlashAttentionForwardSm100(..., paged_kv_non_tma=page_size not in [None, 128])`.

**SM10x note:** the interface dispatch is major-only (`compute_capability == 10`), so SM103 (B300) also routes through `FlashAttentionForwardSm100`. The paged-KV behavior described here is the same codepath on SM100 vs SM103; the practical differences are usually toolchain/ptxas/codegen constraints, not the Python-level dispatch.

So:

- `page_table is None` → contiguous KV (unrelated to this explainer).
- `page_table is not None` **and** `page_size == 128` → paged KV + **TMA KV** (fast path).
- `page_table is not None` **and** `page_size != 128` → paged KV + **non‑TMA** backend (`PagedKVManager`).

Why: the TMA paged KV path assumes `page_size == n_block_size` (typically 128), while the non‑TMA path is written to handle general `page_size` via `divmod`.

Additional constraints worth knowing:

- `page_table` is only supported on SM100 in this codebase (`interface.py` asserts it’s not supported on SM90).
- `page_table` is not supported with `cu_seqlens_k` (assert in `interface.py`).
- Block sparsity + paged KV is explicitly unsupported on SM100 (`flash_fwd_sm100.py` raises).
- Non‑TMA paged KV disallows irregular head dims (`flash_fwd_sm100.py` asserts when `use_tma_KV=False`).

---

## The Concept: What “Paged KV” means here

Paged KV is an inference-oriented KV-cache layout:

- K/V are stored in fixed-size “pages” (chunks).
- A per-sequence **page table** maps logical **page indices** → physical page ids.
- The kernel loads the pages needed for each `n_block` tile into shared memory.

This is common when:

- You need to manage KV cache growth without reallocating large contiguous buffers.
- You want to reuse the same paged KV layout across multiple kernels.

---

## Data Layout (as seen by `flash_fwd_sm100.py`)

At the Python API level (see `interface.py`), paged KV inputs are:

- `page_table`: shape `(batch, max_num_pages_per_seq)`, dtype `int32`
- `k`: shape `(num_pages, page_size, num_head_kv, head_dim)`
- `v`: shape `(num_pages, page_size, num_head_kv, head_dim_v)`

Inside `flash_fwd_sm100.py`, the kernel transposes to layouts that match the SM100 MMA/TMA expectations:

- `mK` becomes `(page_size, d, h_k, num_pages)` (comment in `flash_fwd_sm100.py`)
- `mV` becomes `(d, page_size, h_k, num_pages)` (it additionally transposes V so `d` is the first dimension)

Given a logical token index `tok` (0-based, within the KV sequence):

```
page_idx    = (tok + leftpad_k) // page_size
page_offset = (tok + leftpad_k) %  page_size
page_id     = page_table[batch, page_idx]     # physical page id

K[tok, d] lives at mK[page_offset, d, head, page_id]
V[d, tok] lives at mV[d, page_offset, head, page_id]
```

On SM100, `seqlen_k` is typically provided via `seqused_k` (aka `mSeqUsedK`) so the kernel knows how many KV tokens are actually valid, even if the page table capacity is larger.

---

## Backend A (for context): paged KV with TMA (`page_size == 128`)

When `use_tma_KV=True`, the SM100 load warp uses TMA for KV.

The key bit for paged KV is that the load code reads:

```
page_id = mPageTable[batch_idx, n_block]
```

and passes it down to `load_KV(...)` as `page_idx=page_id`.

Inside `FlashAttentionForwardSm100.load_KV`, the TMA source tile selection is:

- `tXgX_cur = tXgX[None, block]` for contiguous KV
- `tXgX_cur = tXgX[None, 0, page_id]` for paged KV

That `0` is the “block within the page”, and it only works because this path assumes `page_size == n_block_size`
(the file even comments: “Currently we assume that page_size == n_block_size so we index into tXgX with block = 0”).

If `page_size != 128`, the interface flips to the non‑TMA backend below.

---

## Backend B: `PagedKVManager` (non‑TMA paged KV)

`paged_kv.py` defines a `PagedKVManager` dataclass that encapsulates:

- `mPageTable`: the per-batch mapping from logical page index → page id
- `mK_paged`, `mV_paged`: the paged K/V storage tensors
- `page_size_divmod`: fast division/mod (token index → (page_idx, page_offset))
- tiled copy objects (`gmem_tiled_copy_KV`, `gmem_thr_copy_KV`) using `cpasync.CopyG2SOp`

The control flow is roughly:

1. `load_page_table(n_block)`: for rows in the N tile, compute which page each row belongs to and cache `(page_id, page_offset)` in registers.
2. `load_KV(n_block, sX, K_or_V)`: for each row, issue `cp.async` copies from the correct page into shared memory.

---

## `PagedKVManager.create(...)`: copy tiling setup

The manager sets up a fixed “universal” 128-bit async copy:

- `universal_copy_bits = 128`
- `async_copy_elems = universal_copy_bits // dtype.width` (e.g. 8 for FP16/BF16)
- `atom_async_copy = cpasync.CopyG2SOp(cache_mode=GLOBAL)`

It also fixes:

- `gmem_threads_per_row = 8`

The intent (comment in code) is: *8 threads × 128 bits = 128 bytes = 1 cache line*.

Threads are arranged in a 2D layout:

```
thr_layout = (num_threads // gmem_threads_per_row, gmem_threads_per_row)
```

and each thread gets a slice: `gmem_thr_copy_KV = gmem_tiled_copy_KV.get_slice(thread_idx)`.

Finally, it computes how many “row entries” each thread must track:

```
page_entry_per_thread = n_block_size * gmem_threads_per_row // num_threads
```

In the common non‑TMA configuration:

- `load_warp_ids = (14, 15)` → `num_threads = 64`
- `n_block_size = 128`, `gmem_threads_per_row = 8`
- so `page_entry_per_thread = 16`

and each thread allocates:

- `tPrPage[16]` (page ids)
- `tPrPageOffset[16]` (page offsets)

---

## `load_page_table(n_block)`: logical row → (page_id, page_offset)

`load_page_table` computes the page mapping for the rows of the current `n_block` tile and stores it in registers.

For each local entry `i` that this thread owns:

1. Compute the tile-local row coordinate:
   - `row = (i * num_threads + thread_idx) // gmem_threads_per_row`
2. Convert to a global KV token index:
   - `row_idx = n_block * n_block_size + row`
3. Compute `(page_idx, page_offset)` using `FastDivmod`:
   - `page_idx, page_offset = page_size_divmod.divmod(row_idx + leftpad_k)`
4. If this row is in-bounds (`row_idx < seqlen_k`), load the physical page id:
   - `page_id = mPageTable[page_idx]`
5. Cache:
   - `tPrPage[i] = page_id`
   - `tPrPageOffset[i] = page_offset`

This is a deliberate structure: do the `divmod + page_table` lookup once, then reuse it for both K and V.

In `paged_kv.py`, the literal shape of the loop is:

```python
for i in range(page_entry_per_thread):
    row = (i * num_threads + thread_idx) // gmem_threads_per_row
    row_idx = n_block * n_block_size + row

    page_idx, page_offset = page_size_divmod.divmod(row_idx + leftpad_k)
    is_valid = row_idx < seqlen_k
    page_id = mPageTable[page_idx] if is_valid else 0

    tPrPage[i] = page_id
    tPrPageOffset[i] = page_offset
```

(The actual code has a slightly more complex `is_valid` predicate to handle edge cases where the
thread→row mapping might overrun `n_block_size` under unusual `num_threads` choices.)

---

## `load_KV(n_block, sX, K_or_V)`: async load K or V into smem

`load_KV` does the actual global → shared loads.

High-level behavior:

- It “finesses” the shared-memory tensor into a 2D `(rows, cols)` view (`sX_pi`) so the tiled copy can treat it like a matrix.
- For `K_or_V == "V"`, it transposes `sX_pi` because V is stored/consumed with a different major mode (V has `d` as the first dimension in `flash_fwd_sm100.py`).
- It iterates rows, uses cached `(page_id, page_offset)` to compute the correct source pointer, and issues `cp.async` copies for each head-dimension chunk.

Important detail (from the code):
- If a row is out-of-bounds (`should_load == False`), it **does not** issue any `cp.async` loads.
- In that case, it **does** explicitly clear V’s destination smem (via `fill_swizzled(..., 0)`), because leaving stale/NaN data can leak into `P @ V` even when `P` is zeroed.
- It does **not** clear K in this path because the kernel will mask out the corresponding scores anyway.

### How the source pointer is formed (this is the “paged” part)

After `PagedKVManager.create(...)`, the manager keeps per-head views:

- `mK_paged`: shape `(page_size, d, num_pages)` (K is indexed by `(page_offset, d, page_id)`)
- `mV_paged`: shape `(d, page_size, num_pages)` (V is indexed by `(d, page_offset, page_id)`)

Inside `load_KV`, for a given row `m` (token row within the tile), the code does:

```python
page_id     = tPrPage[m]
page_offset = tPrPageOffset[m]

mX_paged_cur = (
    mK_paged[page_offset, None, page_id]   # K  (None = take full d dimension)
    or
    mV_paged[None, page_offset, page_id]   # V  (None = take full d dimension)
)
```

Then it uses `cute.tiled_divide(..., (async_copy_elems,))` so that each `cute.copy(...)` emits one
128-bit `cp.async` transaction.

### The row/col bounds checks

Two separate “are we allowed to touch memory?” checks exist:

- In `load_page_table`: `row_idx < seqlen_k` guards the page table access.
- In `load_KV`: `should_load = (row_in_tile < (seqlen_k - n_block * n_block_size))` guards the K/V global loads.

The split is intentional: the page table might be smaller and cheaper to touch than K/V, and K/V loads are the ones
that must never go out of bounds.

---

## How it plugs into the SM100 kernel (warp roles + pipeline)

The SM100 kernel uses “warp roles” to split work across warps. KV loading is handled by `load_warp_ids`.

In `FlashAttentionForwardSm100.__init__`:

- Default (TMA KV): `load_warp_ids = (14,)`, `empty_warp_ids = (15,)`
- Non‑TMA KV: `load_warp_ids = (14, 15)`, `empty_warp_ids = ()`

So disabling TMA effectively “spends” the previously empty warp on KV loading.

In `FlashAttentionForwardSm100.make_and_init_load_kv_pipeline`:

- TMA KV uses `PipelineTmaUmma` with a producer group sized to `len(load_warp_ids)`.
- Non‑TMA KV uses `PipelineAsyncUmma` with a producer group sized to `len(load_warp_ids) * WARP_SIZE`
  (i.e., all threads in the load warps participate as producers).

In the persistent load loop (`flash_fwd_sm100.py`), the non‑TMA sequence per `n_block` looks like:

1. `paged_kv_manager.load_page_table(n_block)`
2. `load_K(...)` → calls `paged_kv_manager.load_KV(..., "K")`, then `cp_async_commit_group()` and signals the mbarrier
3. `load_V(...)` → calls `paged_kv_manager.load_KV(..., "V")`, then `cp_async_commit_group()` and signals the mbarrier

This is why the “paged KV non‑TMA” backend is not just “a different pointer math” — it also changes how many warps are
producers and which pipeline type is used.

---

## Why this matters for our project

Even if our current workload uses contiguous KV (not paged KV), this explainer matters because:

- The “non-TMA KV path” is a real-world fallback path that can show up depending on configuration.
- The “extra load warps” behavior can change occupancy and interact with our own backend selection/perf expectations.
- If we ever adopt paged KV caching for streaming video, this becomes directly relevant.

---

## Questions & Opportunities

1. For `page_size != 128`, do we want to tune `n_block_size` (tile N) to reduce cross-page mixing, or is the current `divmod` path “good enough”?
2. Can we reduce repeated `mPageTable` traffic (e.g., cache `page_id` per `n_block` when `page_size == n_block_size`) without complicating the general path?
3. When `page_size == 128`, is TMA always better than the `cp.async` backend, or are there corner cases (alignment, residency, small batch) where non‑TMA wins?
4. Is the V zero-fill strictly necessary for correctness (NaN avoidance), or can we guarantee “safe” values another way?

---

## References

- `vendored/flash_attn_cute_score_mod/flash_attn/cute/paged_kv.py`
- `vendored/flash_attn_cute_score_mod/flash_attn/cute/flash_fwd_sm100.py`
- `vendored/flash_attn_cute_score_mod/flash_attn/cute/copy_utils.py`
- `vendored/flash_attn_cute_score_mod/flash_attn/cute/interface.py`
- `vendored/flash_attn_cute_score_mod/flash_attn/cute/seqlen_info.py`
- Explainer #5: `notes/FA4/explainers/05-tma-memory-loading.md`
- Explainer #6: `notes/FA4/explainers/06-tile-scheduling.md`
