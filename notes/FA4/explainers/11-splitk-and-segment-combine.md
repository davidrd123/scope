# Split-K and Segment-Combine (Two Places You’ll See It in This Repo)

> **Explainer #11** — “Segment combine” shows up twice in our stack:
> 1) as a **library optimization** (Split-KV inside FA4/CuTe), and
> 2) as a **functional trick** (our KV-bias “flash” backend splits KV into segments and combines them).
> **Updated:** 2025-12-25

---

## Overview

Attention is “one equation”:

```
O = softmax(QK^T) V
```

But there are two different reasons you might compute it in *pieces* and then **combine** the pieces:

1. **Split-KV for parallelism (library optimization)**  
   One attention call, but KV blocks are partitioned across `num_splits` CTAs, producing `(out_partial, lse_partial)` that a **combine kernel** merges.

2. **Segment-combine for KV-bias (functional workaround)**  
   Our KV-bias rule (“add `log_bias` for a KV index range”) can be implemented by running attention on multiple **disjoint KV segments** and then combining the results using the same log-sum-exp math.

The unifying idea is: **softmax is non-linear**, so “combine” must be done via **LSE** (log-sum-exp), not by naive summation.

Key files:
- FA4 Split-KV dispatch + combine: `vendored/flash_attn_cute_score_mod/flash_attn/cute/interface.py`
- Split partitioning: `vendored/flash_attn_cute_score_mod/flash_attn/cute/block_info.py`
- Combine kernel implementation: `vendored/flash_attn_cute_score_mod/flash_attn/cute/flash_fwd_combine.py`
- Our KV-bias segment-combine backend: `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py`

---

## The Concept: Combining Softmax Results via LSE

Suppose the KV axis is split into disjoint segments `s ∈ {0..S-1}`.
For a given query row, segment `s` computes:

- `O_s = softmax(logits_s) V_s` (softmax **within** segment `s`)
- `LSE_s = log(Σ_j exp(logits_s[j]))` (a scalar per query row, float32)

Then the full attention over the union of segments is:

```
LSE = log( Σ_s exp(LSE_s) )
w_s = exp(LSE_s - LSE)
O   = Σ_s w_s * O_s
```

This is what “segment combine” means in practice:
- Use **logaddexp** to combine normalizers.
- Use **exp(LSE_s - LSE)** as weights for the already-normalized `O_s`.

This is the “one-row” version of online softmax (see Explainer #7), but performed at the granularity of “segments”.

---

## Part A: Split-KV inside FA4/CuTe (Library Optimization)

### 1) How split-KV is enabled

FA4’s Python entrypoint takes a `num_splits` argument:
- `vendored/flash_attn_cute_score_mod/flash_attn/cute/interface.py::_flash_attn_fwd(...)`

It chooses Split-KV when `num_splits > 1`:

- It allocates:
  - `out_partial`: shape `(num_splits, B, Lq, H, Dv)`, dtype `float32`
  - `lse_partial`: shape `(num_splits, B, H, Lq)`, dtype `float32`
- It launches the SM100 forward kernel with `is_split_kv=True`, so each CTA works on a `split_idx`.
- Then it calls `_flash_attn_fwd_combine(out_partial, lse_partial.transpose(-1, -2), out, ...)`.

The transpose is not “math”; it’s layout plumbing:
- The forward kernel naturally writes LSE as `[B, H, Lq]`.
- The combine kernel expects `[B, Lq, H]` (and similarly for the `num_splits`-prefixed version), so the interface passes a transposed view.

### 2) How KV blocks are partitioned per split

The partitioning is computed by `BlockInfo.get_n_block_min_max(...)`:
- `vendored/flash_attn_cute_score_mod/flash_attn/cute/block_info.py`

When `is_split_kv=True`, it divides the per-tile `n_block` range into `num_splits` contiguous chunks:

```
num_n_blocks_per_split = ceil((n_block_max - n_block_min) / num_splits)
n_block_min = n_block_min + split_idx * num_n_blocks_per_split
n_block_max = min(n_block_min + num_n_blocks_per_split, n_block_max)
```

So every `(m_block, head, batch)` tile is replicated `num_splits` times, and each split processes a slice of KV blocks.

### 3) What the SM100 forward kernel produces per split

Inside `FlashAttentionForwardSm100`:
- `vendored/flash_attn_cute_score_mod/flash_attn/cute/flash_fwd_sm100.py`

The tile scheduler includes `split_idx` in the tile index, and the load path uses:

- `n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block, split_idx, num_splits)`

Each split computes a **valid** softmax over only its KV-block subset, and it stores:

- `out_partial[split_idx, ...]` = `O_s` (the segment-normalized output)
- `lse_partial[split_idx, ...]` = `LSE_s` (natural log LSE for the segment; float32; `-inf` if the split is empty)

### 4) How the combine kernel merges splits

The actual combine kernel is `FlashAttentionForwardCombine`:
- `vendored/flash_attn_cute_score_mod/flash_attn/cute/flash_fwd_combine.py`
- wrapped by `vendored/flash_attn_cute_score_mod/flash_attn/cute/interface.py::_flash_attn_fwd_combine(...)`

The combine kernel does, per `(m_block, head, batch)`:

1. Load all `lse_partial[s]` for each row into shared memory.
2. Compute `lse_max = max_s lse_partial[s]`.
3. Compute `lse_sum = log(Σ_s exp(lse_partial[s] - lse_max)) + lse_max` and optionally store it to the final `lse`.
4. Compute normalized weights `w_s = exp(lse_partial[s] - lse_sum)` and store them back into shared memory.
5. Stream `out_partial[s]` for each split and accumulate:
   ```
   out = Σ_s w_s * out_partial[s]
   ```

This is the exact same combine math as the “concept” section above; the kernel just uses `exp2` + `LOG2_E` internally for speed.

**Why Split-KV isn’t “free”:**
- The forward kernel must write `out_partial` + `lse_partial` to global memory.
- The combine kernel reads those partials back and writes the final `out` (and possibly `lse`).
- You only want Split-KV when the extra parallelism outweighs that extra memory traffic + extra kernel launch.

---

## Part B: Segment-Combine in Our KV-Bias “Flash” Backend (Functional Workaround)

Our KV-bias rule is:

> add `log_bias` to logits where `kv_idx` is in `[frame_seqlen, current_block_start)`

When we don’t have a `score_mod`-capable kernel, we implement this by splitting KV into **three disjoint segments** and combining them:
- `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py::_kv_bias_flash_combine(...)`

Segments:
1. `[0, frame_end)` — unbiased
2. `[frame_end, block_start)` — biased by `log_bias`
3. `[block_start, lk)` — unbiased

For each segment we compute `(out_seg, lse_seg)` via `_flash_attn_with_lse(...)` (either FA4 `_flash_attn_fwd(return_lse=True)` or the FA2 varlen kernel as fallback).

Then we apply the bias *at the segment level* by shifting the segment’s LSE:

```
if seg_log_w != 0:
    lse_seg = lse_seg + seg_log_w
```

This works because adding a constant `c` to **all** logits in a segment multiplies the segment’s exp-sum by `exp(c)`,
so the segment LSE shifts by `+c`.

Finally we merge segments using:
- `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py::_merge_out_lse(...)`

which is literally:

```
lse_new = logaddexp(lse_a, lse_b)
w_a = exp(lse_a - lse_new)
w_b = exp(lse_b - lse_new)
out_new = w_a * out_a + w_b * out_b
```

This is “segment combine” in its simplest (two-segment) form, repeated for up to three segments.

**Why it costs more:** it’s multiple attention calls + extra elementwise combine work, which is exactly the overhead that `score_mod` avoids.

---

## Why FA4 `score_mod` Helps (In One Sentence)

With `SCOPE_KV_BIAS_BACKEND=fa4`, we express the KV-bias as an in-kernel `score_mod`, so we compute the *correct logits* in a single attention call — no KV segmentation, no extra attention calls, no logaddexp combine.

See:
- `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py` (`score_mod_kv_bias`, backend selection)
- Explainer #3: [`03-score-mod.md`](03-score-mod.md)

---

## Cost Model (Why Combine Shows Up in Profiles)

This is the “so what does it cost?” section — useful when you’re staring at a profiler and wondering why combine is a big deal.

### A) Split-KV inside FA4 (library optimization)

Split-KV pays for parallelism with extra memory traffic and an extra kernel:

- Forward kernel writes:
  - `out_partial`: `num_splits × B × Lq × H × Dv` **float32**
  - `lse_partial`: `num_splits × B × H × Lq` **float32**
- Combine kernel reads those partials back and writes:
  - final `out` (and optionally final `lse`)

Rule of thumb:

- Split-KV is worth it when KV is large enough that extra CTAs improve occupancy/latency *more* than the extra write/read traffic hurts.
- It’s **not** free even if compute scales nicely, because you’re explicitly materializing partial outputs.

### B) Segment-combine for KV-bias (functional workaround)

Segment-combine is “pay extra to express a score modification without score_mod”.

For `S` segments (we use up to 3):

- You run **S attention calls** (separate kernels).
  - Q is effectively re-used S times (loaded/computed per call).
  - K/V are streamed per segment (total KV length sums to the original KV, but launches and Q overhead multiply).
- You run **combine math** (logaddexp + weighted sum) over `B×Lq×H×Dv`.

What this means in practice:

- Even if the total `Lk` across segments equals the original `Lk`, you still pay:
  - launch overhead × S
  - Q-side overhead × S
  - combine overhead (often memory-bound)

This is why `score_mod` tends to win when available: it keeps the whole operation inside a single attention kernel.

---

## References

- Split-KV dispatch + combine call: `vendored/flash_attn_cute_score_mod/flash_attn/cute/interface.py`
- Split partitioning: `vendored/flash_attn_cute_score_mod/flash_attn/cute/block_info.py`
- SM100 forward kernel split handling (`split_idx`, per-split LSE write): `vendored/flash_attn_cute_score_mod/flash_attn/cute/flash_fwd_sm100.py`
- Combine kernel algorithm (`lse_max`, weights, O accumulation): `vendored/flash_attn_cute_score_mod/flash_attn/cute/flash_fwd_combine.py`
- KV-bias segment-combine backend: `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py`
- Related: Explainer #7 [`07-online-softmax.md`](07-online-softmax.md)
