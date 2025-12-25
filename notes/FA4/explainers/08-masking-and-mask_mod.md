# Masking and mask_mod in FA4/CUTE

> **What this is:** An explainer for how FA4 applies seqlen/causal/local masking inside the fused kernel, and what `mask_mod` really is (and isn’t).
> **Updated:** 2025-12-25

---

## Overview

FA4 has two distinct “customization surfaces” that often get conflated:

- `score_mod`: modify **scores** (after `QK^T` and scaling, before softmax)
- `mask_mod`: decide whether a (q, kv) pair is **valid** (a boolean mask; invalid entries become `-inf`)

This explainer is about masking:

- Built-in masking modes: seqlen bounds, causal, local window
- The “R2P trick” (bit ops that compile to a dedicated instruction) used in some masking paths
- What `mask_mod` receives (indices) and how it is applied (elementwise, inside the kernel)
- Why there are effectively two masking “backends” in-tree:
  - `AttentionMask.apply_mask(...)` (SM80/SM90-style kernels, used by `flash_fwd.py`)
  - `AttentionMask.apply_mask_sm100(...)` + `apply_mask_sm100_transposed(...)` (SM100 kernels, used by `flash_fwd_sm100.py` / `flash_bwd_sm100.py`)

Key sources:
- `vendored/flash_attn_cute_score_mod/flash_attn/cute/mask.py`
- `vendored/flash_attn_cute_score_mod/flash_attn/cute/mask_definitions.py`

---

## The Concept: What Masking Means Inside the Kernel

The attention kernel computes a tile of scores:

```
S_tile = Q_tile @ K_tile^T   # shape [tile_m, tile_n]
```

Masking is implemented as:

```
S_tile[r, c] = -inf   if (r, c) is invalid
S_tile[r, c] = S_tile[r, c] otherwise
```

So masking must happen **before softmax** (because `softmax(-inf) = 0`).

### Why masking is not “just a score_mod”

You *can* implement masking by returning `-inf` in a `score_mod`, but FA4 treats masking specially:

- It has optimized paths for causal/local bounds that avoid per-element function calls.
- It can apply fast bitmask operations (R2P) for in-bounds columns.

Use `mask_mod` when you need “valid/invalid”; use `score_mod` when you need “continuous bias”.

---

## The `AttentionMask` API (Mental Model)

In `mask.py`, FA4 defines an `AttentionMask` dataclass with an `apply_mask(...)` method.

At a high level, it chooses among:

1. **No mask** (fast path): do nothing
2. **mask_mod-only**: apply a user-provided boolean predicate per element
3. **causal/local**: apply bounds computed from `(m_block, n_block, seqlens, window sizes)`

One non-obvious rule from the code: `mask_mod` is only evaluated when `mask_causal == mask_local == False`.
If you pass `mask_mod` with `mask_causal=True`, the kernel takes the causal branch and ignores `mask_mod`.

### Indices: what “row” and “col” mean

Masking operates on **global token indices**:

- `global_row_idx = m_block * tile_m + row_in_tile`
- `global_col_idx = n_block * tile_n + col_in_tile` (plus any per-thread column offset)

FA4 computes these using CuTe’s “identity tensor” + the thread’s MMA partitioning view.

---

## mask_mod: What it is and how it’s called

`mask_mod` is a callback that returns a boolean-like value (a CuTe SSA boolean, not a Python `bool`):

```python
mask_value = mask_mod(
    batch_idx,
    head_idx,
    q_idx,     # global query token index
    kv_idx,    # global key token index
    aux_tensors,
)
```

Then the kernel sets:

```
S[q_idx, kv_idx] = S[q_idx, kv_idx] if mask_value else -inf
```

FA4 provides examples of CuTe-compiled `mask_mod` functions in `mask_definitions.py` (e.g. `cute_document_mask`, `get_cute_sliding_window_mask`, `cute_prefix_lm_mask`).

### Practical implications

- `mask_mod` is **elementwise** and can be expensive if used broadly.
- Prefer built-in masking (causal/local/seqlen) when possible.
- If you only need a simple “band” mask, local/causal is usually faster than `mask_mod`.

---

## Built-in Masks: seqlen, causal, local

FA4 supports:

- **seqlen mask**: mask out tokens beyond the actual sequence length (important for varlen/padded batches)
- **causal mask**: enforce `kv_idx <= q_idx` (modulo offsets/splits)
- **local mask**: restrict to a window around `q_idx` (left/right window sizes)

### The “causal_row_offset” pattern (why it looks weird)

In the causal path, the kernel computes a per-row column limit:

```
col_limit_right = row_idx + causal_row_offset
```

and masks columns `>= col_limit_right`.

In `mask.py` the “offset” includes the tile origin and the thread’s column offset so that
the comparison can stay in the **tile-local** coordinate system (no per-element `global_kv_idx` math).

The exact offset depends on how FA4 tiles in N and how it aligns Q/K seqlens.

---

## The Implementation (Walkthrough of `mask.py`)

This section uses the variable names from `vendored/flash_attn_cute_score_mod/flash_attn/cute/mask.py`.

One meta-point before diving in: `mask_seqlen`, `mask_causal`, `mask_local`, and even `mask_mod` are passed as
`cutlass.Constexpr[...]` parameters, and most branches are guarded by `if const_expr(...)`.
So the JIT generates *different* kernels for different masking configurations; this is not “one runtime-branchy kernel”.

### 1) How FA4 gets per-element (row, col) coordinates

All masking paths start by creating an “identity score tile” and then asking the MMA partitioning
logic “which (row, col) coordinates does this thread own?”

On SM80/SM90-style kernels (`apply_mask`):

- `cS = cute.make_identity_tensor((tile_m, tile_n))`
- `tScS_mn = utils.make_acc_tensor_mn_view(thr_mma.partition_C(cS), transpose=swap_AB)`
  - runtime coordinates for this thread’s accumulator fragment
- `t0ScS_mn = utils.make_acc_tensor_mn_view(thr_mma.get_slice(0).partition_C(cS), transpose=swap_AB)`
  - compile-time-known coordinates (slice 0), used to make comparisons unrollable

On SM100 (`apply_mask_sm100`), the kernel does the same MMA partitioning, but then remaps the coordinates
through the TMA/TMEM load mapping:

- `tScS = thr_mma.partition_C(cS)`
- `tScS_t2r = thr_tmem_load.partition_D(tScS)`

`tScS_t2r[i]` tells you which (row, col) coordinate corresponds to `acc_S[i]` for this thread.

### 2) `AttentionMask.apply_mask(...)` (SM80/SM90)

`apply_mask` has three high-level branches:

One small but important guardrail (used in both SM90 and SM100 codepaths): the kernel sometimes
represents “no valid n-tiles” by passing a negative `n_block`, so `mask.py` clamps:

```
if n_block < 0:
  n_block = 0
```

This is a workaround for edge cases like “completely masked out rows where `n_block_max = 0`”.

#### A) seqlen-only (`mask_seqlen=True`, no causal/local, no `mask_mod`)

Key setup:

- `thr_col_offset = tScS_mn[0][COL]`
- `seqlenk_col_limit = seqlen_k - n_block * tile_n - thr_col_offset`

Then for each column fragment `c`, it checks:

```
oob = t0ScS_mn[0, c][COL] >= seqlenk_col_limit
```

and if `oob`, masks **that entire column** to `-inf` for all `r` in the fragment.

Note: there is an R2P fast path in `mask.py`, but this branch currently hard-disables it:
`r2p = const_expr(False and not self.swap_AB)`.

#### B) `mask_mod` (FlexAttention-style elementwise mask)

Taken only when `mask_causal == mask_local == False` and `mask_mod is not None`.

Core loop shape (simplified):

```
for r:
  global_row_idx = tScS_mn[r, 0][0] + m_block * tile_m
  for col:
    col_idx_local = t0ScS_mn[0, col][1]
    global_col_idx = thr_col_offset + col_idx_local + n_block * tile_n
    cond = mask_mod(batch, head, global_row_idx, global_col_idx, aux_tensors)
    acc_S_mn[r, col] = acc_S_mn[r, col] if cond else -inf
```

Two details that matter in practice:

- **SSA conversion:** `mask_mod` is invoked with SSA scalars (`utils.scalar_to_ssa`) and returns an SSA boolean-like value, which is then converted back (`utils.ssa_to_scalar`) before masking.
- **Wrapping indices for aux tensor safety:** if `mask_seqlen` and `aux_tensors` are present and `fastdiv_mods` is provided, FA4 computes:
  - `row_for_mod = global_row_idx % seqlen_q`
  - `col_for_mod = global_col_idx % seqlen_k`

  so that `mask_mod` can safely index `aux_tensors` even when the tile extends past the actual sequence length.
  After calling `mask_mod`, FA4 still hard-masks true out-of-bounds elements to `-inf`.

#### C) causal/local (no `mask_mod` evaluation)

Taken whenever `mask_causal` or `mask_local` is true. There are two sub-cases:

- **`swap_AB == False` (normal forward `S = QK^T`)**
- **`swap_AB == True` (masking a transposed accumulator view; bounds are applied as “row limits per column”)**

For the normal (`swap_AB == False`) causal mask:

- `causal_row_offset = 1 + seqlen_k - n_block * tile_n - seqlen_q - thr_col_offset`
- `row_idx = tScS_mn[r, 0][0] + m_block * tile_m` (or PackGQA-adjusted)
- `col_limit_right = row_idx + causal_row_offset`
- if `mask_seqlen`: `col_limit_right = min(col_limit_right, seqlenk_col_limit)`

Then FA4 masks columns where `col_idx_local >= col_limit_right`.
When possible, it uses:

- `mask_r2p(acc_S_mn[r, None], col_limit_right, arch=90, rank1=True)`

to avoid an explicit `(r,c)` loop.

If you rewrite this back into global indices, it’s the usual causal condition with an offset:

- `global_kv = n_block * tile_n + thr_col_offset + col_idx_local`
- `global_q = row_idx`
- allow if `global_kv <= global_q + (seqlen_k - seqlen_q)`

The `+1` inside `causal_row_offset` exists because the implementation masks `col_idx_local >= col_limit_right`
(so “allowed” means `col_idx_local < col_limit_right`).

For the normal local mask, it similarly computes `col_limit_left` / `col_limit_right` and masks anything outside:

```
col_idx_local < col_limit_left or col_idx_local >= col_limit_right
```

PackGQA note: when `qhead_per_kvhead_packgqa != 1`, FA4 avoids recomputing the `// qhead_per_kvhead_packgqa`
division per `(r,thread)` by shuffling a precomputed `mma_m_idx` across threads in the same row.

For the `swap_AB == True` sub-case, the logic is the same idea but applied “sideways”:

- Iterate columns `c`
- Treat `col0 = t0ScS_mn[0, c][COL]` as the tile-local KV coordinate for that column
- Compute a **row** limit (`row_limit_top`, and optionally `row_limit_bot` for local)
- Mask rows based on comparisons against `t0ScS_mn[r, 0][ROW]`

In code (causal case, simplified):

```
row_limit_top =
  tile_m                         if (mask_seqlen and col0 >= seqlenk_col_limit)
  col0 - causal_row_offset        otherwise

mask if row_idx < row_limit_top
```

### 3) The R2P helpers: `mask_r2p` and `mask_r2p_transposed`

`mask_r2p(X, col_limit, arch, rank1)` is a bitmask-based masking helper.

Important implementation details:

- It chunks columns in groups of **24** (not 32) because `mask >> i` is “wrong” for `i == 31` in the chosen lowering.
- It uses `cutlass.range_constexpr(...)` so the compiler can lower it to the dedicated R2P instruction.
- For `arch == 90`, it transforms the limit:

  ```
  col_limit_transformed = col_limit // 8 * 2 + min(col_limit % 8, 2)
  ```

  to match the non-trivial lane-to-column mapping of the SM90 fragment layout (see comments in the code).

`mask_r2p_transposed` is the analogous helper for the transposed/backward mapping.
It bakes in an assumption of `num_wg = 2` and transforms `row_limit_top` to match the layout produced by
the SM100 TMEM copy patterns.

### 4) `AttentionMask.apply_mask_sm100(...)` (SM100 forward)

SM100 uses `tScS_t2r = thr_tmem_load.partition_D(tScS)` and typically treats `acc_S` as a rank-1 fragment,
so masking is written in terms of `acc_S[i]` and `tScS_t2r[i]`.

Structure mirrors the SM90 version:

- seqlen-only: `mask_r2p(acc_S, seqlenk_col_limit, arch=100, rank1=True)`
- `mask_mod`-only: elementwise mask using:
  - `global_row = tScS_t2r[0][0] + m_block * tile_m` (row is constant for this fragment)
  - `global_col = col_coord + n_block * tile_n` (per `i`)
- causal/local: compute `col_limit_right` / `col_limit_left` and apply R2P or loops

Callsite: `flash_fwd_sm100.py` builds a `mask_fn = partial(mask.apply_mask_sm100, ...)` and passes it into the
mainloop (optionally also building a `mask_fn_none` for full blocks in the block-sparse case).

### 5) `AttentionMask.apply_mask_sm100_transposed(...)` (SM100 backward masking)

This is used in `flash_bwd_sm100.py` when masking a “transposed” score fragment (comments in the code call this
“Backward pass: mask S = K @ Q.T”).

Key parameters beyond the forward version:

- `tScS_t2r`, `t0ScS_t2r`: coordinate tensors for the TMEM→RMEM mapping
- `is_full_block`: for block-sparse backward, skip `mask_mod` for full blocks
- `check_m_boundary`: skip `seqlen_q` boundary checks for non-final `m_block`s

The block-sparse path is explicitly spelled out:

- **full blocks:** skip `mask_mod`, only apply seqlen masking
- **partial blocks:** apply `mask_mod` elementwise, then apply seqlen masking

The causal/local path computes a “row limit” instead of a “col limit” and can use:

- `mask_r2p_transposed(acc_S, row_limit_top, num_rep)`

to avoid an explicit loop.

The core quantities in the causal/local branch (names from `mask.py`) are:

- `thr_col_offset = tScS_t2r[0][COL]`
- `seqlenk_col_limit = seqlen_k - n_block * tile_n - thr_col_offset`
- `thr_row_offset = tScS_t2r[0][ROW]`
- `seqlenq_row_limit = seqlen_q - m_block * tile_m - thr_row_offset`
- `causal_offset = seqlenq_row_limit - seqlenk_col_limit`

Then:

- causal: `row_limit_top = causal_offset` (or `tile_m` if the entire K tile is OOB)
- local: `row_limit_top = causal_offset - window_size_right`, `row_limit_bot = causal_offset + window_size_left`

---

## Key Design Decisions

- **Compile-time specialization:** masking knobs are `Constexpr`, so unused branches are compiled away instead of becoming runtime divergence.
- **Don’t pay for masking on “interior tiles”:** the forward kernels usually only call masking on (1) the last `n_block` for K-residue (`mask_seqlen=True`) and (2) the near-diagonal blocks for causal/local. If `mask_mod is None`, most tiles do no masking work at all.
- **Keep comparisons tile-local:** offsets like `seqlenk_col_limit` and `causal_row_offset` are designed so masking is often a compare against a tile-local `col_idx_local`, avoiding per-element `global_kv` math.
- **Use R2P when it maps cleanly:** `mask_r2p` / `mask_r2p_transposed` encode layout-specific transformations so a simple “limit” becomes a fast bitmask predicate.
- **Make `mask_mod` safe by construction:** the optional `% seqlen` wrapping exists because `mask_mod` may execute on out-of-bounds coordinates; masking happens after the callback returns.
- **Backward block-sparse shortcuts:** `is_full_block` and `check_m_boundary` exist to avoid elementwise work when the scheduler already knows a tile is fully valid / fully in-bounds.

---

## The R2P trick (why it exists)

Masking can become a bottleneck if you do:

```python
for r in rows:
  for c in cols:
    if c >= limit: S[r, c] = -inf
```

FA4 uses a bitmask approach in some cases to generate a mask with shifts and apply it quickly.
In `mask.py`, this is described as compiling down to an “R2P” instruction (a hardware-supported path for turning bit masks into predicates / predicates into per-lane effects).

The core helper is:
- `mask_r2p(...)` for “rank1” or matrix-shaped fragments
- `mask_r2p_transposed(...)` for certain transposed layouts

This is one of the reasons “masking is special” vs a generic `score_mod`.

---

## How this connects to our KV-bias work

Our KV-bias is not a boolean “valid/invalid” — it is a continuous bias applied to a region.
So it belongs in `score_mod`, not `mask_mod`.

However, understanding `mask_mod` matters because:

- Some “KV-bias-like” patterns can be approximated as a local window (mask out far history).
- If we ever change the conditioning strategy (e.g. strict truncation), `mask_mod`/local masks might be relevant.

---

## Questions & Opportunities

1. `apply_mask` disables R2P in the seqlen-only branch; can we re-enable it (or is it a compiler/codegen issue worth leaving alone)?
2. If we ever need a non-rectangular region mask (e.g. document masks), does `mask_mod` overhead dominate, or does it remain amortized because only edge tiles call masking?
3. `mask_mod` is mutually exclusive with causal/local in `mask.py`. Do we ever need “causal + custom mask” (and if so, should we encode it purely in `mask_mod`)?

---

## References

- `vendored/flash_attn_cute_score_mod/flash_attn/cute/mask.py`
- `vendored/flash_attn_cute_score_mod/flash_attn/cute/mask_definitions.py`
- Mask callsites:
  - `vendored/flash_attn_cute_score_mod/flash_attn/cute/flash_fwd.py`
  - `vendored/flash_attn_cute_score_mod/flash_attn/cute/flash_fwd_sm100.py`
  - `vendored/flash_attn_cute_score_mod/flash_attn/cute/flash_bwd_sm100.py`
- Explainer #3: `notes/FA4/explainers/03-score-mod.md`
- Explainer #7: `notes/FA4/explainers/07-online-softmax.md`
