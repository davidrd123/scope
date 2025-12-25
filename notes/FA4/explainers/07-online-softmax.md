# Online Softmax in FA4/CUTE

> **What this is:** An explainer for how FlashAttention 4 computes softmax incrementally as KV tiles stream through, without materializing the full attention matrix.
> **Context:** After understanding tile scheduling (#6) and memory loading (#5), this explainer covers the core algorithmic innovation that makes FlashAttention possible: computing softmax row-wise as tiles arrive.
> **Updated:** 2025-12-25

---

## Overview

### The Problem with Standard Softmax

Standard attention computes:
```
Attention(Q, K, V) = softmax(QK^T / √d) × V
```

Softmax requires:
1. **Row max** - `max_j(S_ij)` for numerical stability
2. **Row sum** - `Σ_j exp(S_ij - max)`

Both require seeing **all elements** in a row before computing the output. For long sequences, this means materializing the full N×N attention matrix in memory.

### The Online Solution

FlashAttention's insight: maintain **running statistics** as tiles stream through:

```
For each KV tile:
  1. Compute partial scores: S_partial = Q @ K_tile.T
  2. Update running max: max_new = max(max_old, max(S_partial))
  3. Rescale previous work: O = O * exp(max_old - max_new)
  4. Compute exp: P_partial = exp(S_partial - max_new)
  5. Update running sum: sum_new = sum_old * exp(max_old - max_new) + sum(P_partial)
  6. Accumulate: O += P_partial @ V_tile
```

After all tiles: `O_final = O / sum`

### What This Explainer Covers

1. The online softmax algorithm in detail
2. Why we use `exp2` instead of `exp`
3. How FA4 tracks row max and row sum
4. The rescale threshold optimization
5. Computing the final logsumexp (LSE)

---

## The Algorithm

### Standard Softmax (Infeasible)

For row `i` of the attention matrix:

```python
# Requires full row in memory
S_i = [S_i0, S_i1, ..., S_iN]  # N scores
m = max(S_i)                    # Global max
P_i = exp(S_i - m)              # Stable exp
l = sum(P_i)                    # Sum for normalization
A_i = P_i / l                   # Normalized attention
O_i = A_i @ V                   # Output
```

**Problem:** Storing `S_i` requires O(N) memory per row, O(M×N) total.

### Online Softmax (FlashAttention)

Process in tiles, maintaining running statistics:

```python
m = -inf  # Running max
l = 0     # Running sum
O = 0     # Running output

for tile in KV_tiles:
    S_tile = Q @ K_tile.T          # Partial scores
    m_tile = max(S_tile)           # Tile max
    m_new = max(m, m_tile)         # Update global max

    # Rescale factor for previous work
    scale = exp(m - m_new)

    # Update running statistics
    P_tile = exp(S_tile - m_new)   # Stable exp for this tile
    l = l * scale + sum(P_tile)    # Rescaled sum
    O = O * scale + P_tile @ V_tile  # Rescaled output

    m = m_new  # Update max

# Finalize
O = O / l
```

**Key insight:** The `scale = exp(m_old - m_new)` factor corrects previous work when a new max is discovered.

---

## Why exp2 Instead of exp

### The Mathematical Trick

GPUs have fast `exp2` (2^x) but slower `exp` (e^x). Since:
```
exp(x) = exp2(x * log2(e))
```

We can rewrite everything in base-2:

```python
scale_log2 = softmax_scale * log2(e)  # Precompute

# Instead of: exp(S * scale - max * scale)
# We compute: exp2(S * scale_log2 - max * scale_log2)
```

**Note on `score_mod`:** When a custom `score_mod` is enabled, FA4/CuTe applies `softmax_scale` *before* calling the score modifier (so the modifier sees the conventional scaled scores). In that case the exp2 path typically uses `scale_log2 = log2(e)` (change-of-base only), since the scores are already scaled.

### In the Code

```python
# softmax.py lines 91-93
if cutlass.const_expr(is_first):
    row_max_cur_scaled = row_max_cur * scale_log2
    acc_S_row_exp = utils.exp2f(acc_S_row * scale_log2 - row_max_cur_scaled)
```

The `exp2f` utility uses the PTX `ex2.approx` instruction:

```python
# utils.py lines 286-300
def exp2f(x: cute.TensorSSA | Float32) -> cute.TensorSSA | Float32:
    if const_expr(isinstance(x, cute.TensorSSA)):
        res = cute.make_fragment(x.shape, Float32)
        res.store(x)
        for i in cutlass.range_constexpr(cute.size(x.shape)):
            res[i] = cute.arch.exp2(res[i])
        return res.load()
    else:
        return cute.arch.exp2(x)
```

---

## FA4's Softmax Implementation

### The Softmax Class

```python
# softmax.py lines 20-42
@dataclass
class Softmax(ParamsBase):
    scale_log2: Float32      # softmax_scale * log2(e) (or log2(e) if scores are pre-scaled)
    num_rows: int            # Rows tracked per thread
    row_max: cute.Tensor     # Running max per row
    row_sum: cute.Tensor     # Running sum per row
    arch: int = 80           # GPU architecture
    softmax_scale: Float32   # Original scale (used when score_mod wants scaled scores)

    def reset(self) -> None:
        self.row_max.fill(-Float32.inf)
        self.row_sum.fill(0.0)
```

### The online_softmax Method

The core algorithm, processing one row at a time:

```python
# softmax.py lines 54-112
@cute.jit
def online_softmax(self, acc_S, is_first=False, check_inf=True):
    acc_S_mn = make_acc_tensor_mn_view(acc_S)  # Reshape to (M, N)
    row_scale = cute.make_fragment_like(self.row_max)

    for r in cutlass.range(cute.size(row_max), unroll_full=True):
        acc_S_row = acc_S_mn[r, None].load()  # One row of scores

        # Step 1: Find new max
        row_max_cur = fmax_reduce(
            acc_S_row,
            init_val=row_max[r] if not is_first else None
        )
        row_max_cur = warp_reduce(row_max_cur, fmax, width=4)  # Quad reduction

        # Handle -inf (empty row)
        if check_inf:
            row_max_cur = 0.0 if row_max_cur == -inf else row_max_cur

        # Step 2: Compute exp and scale factor
        if is_first:
            # First tile: no rescaling needed
            row_max_cur_scaled = row_max_cur * scale_log2
            acc_S_row_exp = exp2f(acc_S_row * scale_log2 - row_max_cur_scaled)
            acc_S_row_sum = fadd_reduce(acc_S_row_exp)
            row_scale[r] = 1.0
        else:
            # Later tiles: rescale previous work
            row_max_prev = row_max[r]
            row_max_cur_scaled = row_max_cur * scale_log2
            acc_S_row_exp = exp2f(acc_S_row * scale_log2 - row_max_cur_scaled)

            # Scale factor to correct previous accumulation
            row_scale[r] = exp2f((row_max_prev - row_max_cur) * scale_log2)

            # Add to rescaled previous sum
            acc_S_row_sum = fadd_reduce(
                acc_S_row_exp,
                init_val=row_sum[r] * row_scale[r]
            )

        # Step 3: Update running statistics
        row_max[r] = row_max_cur
        row_sum[r] = acc_S_row_sum
        acc_S_mn[r, None].store(acc_S_row_exp)  # Store P for PV matmul

    return row_scale  # Caller uses this to rescale O
```

### Rescaling the Output

After each tile, the output accumulator must be rescaled:

```python
# softmax.py lines 149-160
@cute.jit
def rescale_O(self, acc_O, row_scale):
    acc_O_mn = make_acc_tensor_mn_view(acc_O)
    for r in cutlass.range(cute.size(row_scale), unroll_full=True):
        acc_O_mn[r, None].store(acc_O_mn[r, None].load() * row_scale[r])
```

---

## SM100-Specific Optimizations

### SoftmaxSm100 Class

Blackwell has specific optimizations:

```python
# softmax.py lines 163-185
@dataclass
class SoftmaxSm100(Softmax):
    rescale_threshold: float = 0.0  # Skip rescale if max change is small

    @staticmethod
    def create(scale_log2, rescale_threshold=0.0, softmax_scale=None):
        num_rows = 1  # SM100 keeps running stats for 1 row per thread
        arch = 100
        row_max = cute.make_rmem_tensor(num_rows, Float32)
        row_sum = cute.make_rmem_tensor(num_rows, Float32)
        return SoftmaxSm100(...)
```

### The Rescale Threshold Optimization

When the max doesn't change much, skip rescaling:

```python
# softmax.py lines 187-205
@cute.jit
def update_row_max(self, acc_S_row, is_first):
    if is_first:
        row_max_new = self._compute_row_max(acc_S_row)
        row_max_safe = row_max_new if row_max_new != -inf else 0.0
        acc_scale = 0.0
    else:
        row_max_old = self.row_max[0]
        row_max_new = self._compute_row_max(acc_S_row, init_val=row_max_old)
        row_max_safe = row_max_new if row_max_new != -inf else 0.0
        acc_scale_ = (row_max_old - row_max_safe) * self.scale_log2

        acc_scale = exp2f(acc_scale_)

        # OPTIMIZATION: If max changed by less than threshold, don't rescale
        if self.rescale_threshold > 0.0:
            if acc_scale_ >= -self.rescale_threshold:
                row_max_new = row_max_old
                row_max_safe = row_max_old
                acc_scale = 1.0  # No rescaling

    self.row_max[0] = row_max_new
    return row_max_safe, acc_scale
```

**Why this helps:** Rescaling requires reading and rewriting the output accumulator. If the max hasn't changed significantly, `exp(old - new) ≈ 1.0` and rescaling is wasted work.

### Packed Float Operations

SM100 uses packed 2-element operations for efficiency:

```python
# softmax.py lines 217-229
@cute.jit
def scale_subtract_rowmax(self, acc_S_row, row_max):
    row_max_scaled = row_max * self.scale_log2
    for i in cutlass.range(0, cute.size(acc_S_row.shape), 2, unroll_full=True):
        # Process 2 elements at once
        acc_S_row[i], acc_S_row[i + 1] = fma_packed_f32x2(
            (acc_S_row[i], acc_S_row[i + 1]),
            (self.scale_log2, self.scale_log2),
            (-row_max_scaled, -row_max_scaled),
        )
```

---

## Reduction Operations

### Row-Wise Max (fmax_reduce)

Find max across a row of scores:

```python
# utils.py lines 340-387
@cute.jit
def fmax_reduce(x, init_val=None, arch=80):
    if arch < 100 or cute.size(x.shape) % 8 != 0:
        # SM80/90 (and irregular sizes): unrolled 2-input fmax reduction
        ...
    else:
        # SM100 + multiples of 8: force 3-input fmax for better throughput
        res = cute.make_fragment(x.shape, Float32)
        res.store(x)

        # Process 8 elements at a time with 3-input max
        local_max_0 = (
            fmax(init_val, res[0], res[1]) if init_val is not None else fmax(res[0], res[1])
        )
        local_max = [
            local_max_0,
            fmax(res[2], res[3]),
            fmax(res[4], res[5]),
            fmax(res[6], res[7]),
        ]
        for i in range(8, cute.size(x.shape), 8):
            local_max[0] = fmax(local_max[0], res[i], res[i + 1])  # 3-input!
            local_max[1] = fmax(local_max[1], res[i + 2], res[i + 3])
            local_max[2] = fmax(local_max[2], res[i + 4], res[i + 5])
            local_max[3] = fmax(local_max[3], res[i + 6], res[i + 7])

        # Final reduction
        local_max[0] = fmax(local_max[0], local_max[1])
        return fmax(local_max[0], local_max[2], local_max[3])
```

### Row-Wise Sum (fadd_reduce)

Sum exp values across a row:

```python
# utils.py lines 391-428
@cute.jit
def fadd_reduce(x, init_val=None, arch=80):
    if arch >= 100 and cute.size(x.shape) % 8 == 0:
        # SM100: Use packed f32x2 adds
        res = cute.make_fragment(x.shape, Float32)
        res.store(x)

        local_sum_0 = (
            add_packed_f32x2((init_val, 0.0), (res[0], res[1]))
            if init_val is not None
            else (res[0], res[1])
        )
        local_sum = [local_sum_0, (res[2], res[3]), (res[4], res[5]), (res[6], res[7])]

        for i in range(8, cute.size(x.shape), 8):
            local_sum[0] = add_packed_f32x2(local_sum[0], (res[i + 0], res[i + 1]))
            local_sum[1] = add_packed_f32x2(local_sum[1], (res[i + 2], res[i + 3]))
            ...

        # Final reduction
        local_sum[0] = add_packed_f32x2(local_sum[0], local_sum[1])
        local_sum[2] = add_packed_f32x2(local_sum[2], local_sum[3])
        local_sum[0] = add_packed_f32x2(local_sum[0], local_sum[2])
        return local_sum[0][0] + local_sum[0][1]
    else:
        # SM80/90: Use standard reduce
        return x.reduce(cute.ReductionOp.ADD, init_val or 0.0, 0)
```

### Warp-Level Reduction

After thread-local reduction, combine across threads:

```python
# utils.py lines 152-166
@cute.jit
def warp_reduce(val, op, width=WARP_SIZE):
    if isinstance(val, cute.TensorSSA):
        # Vector: reduce each element
        res = cute.make_fragment(val.shape, val.dtype)
        res.store(val)
        for i in range(cute.size(val.shape)):
            res[i] = warp_reduce(res[i], op, width)
        return res.load()
    else:
        # Scalar: butterfly shuffle
        for i in range(int(math.log2(width))):
            val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val
```

---

## Finalization and LSE

### Computing the Final Output

After all tiles are processed, the SM80/90 path calls `Softmax.finalize()` to compute the final `1 / row_sum` scale and produce LSE for backward.

```python
# softmax.py (Softmax.finalize)
@cute.jit
def finalize(self, final_scale=1.0, sink_val=None):
    row_sum = self.row_sum
    row_max = self.row_max
    scale_log2 = self.scale_log2

    # Quad reduction for row_sum (wasn't done during online softmax)
    row_sum.store(utils.warp_reduce(row_sum.load(), operator.add, width=4))

    row_scale = cute.make_fragment_like(row_max, Float32)

    for r in cutlass.range(cute.size(row_sum), unroll_full=True):
        # Optional: add sink token contribution
        if sink_val is not None:
            LOG2_E = math.log2(math.e)
            row_sum[r] += utils.exp2f(sink_val * LOG2_E - row_max[r] * scale_log2)

        # Handle zero/NaN sum
        acc_O_mn_row_is_zero_or_nan = row_sum[r] == 0.0 or row_sum[r] != row_sum[r]

        # Final scale: 1/sum
        row_scale[r] = (
            cute.arch.rcp_approx(row_sum[r] if not acc_O_mn_row_is_zero_or_nan else 1.0)
        ) * final_scale

        # Compute LSE (logsumexp) for backward pass
        row_sum_cur = row_sum[r]
        LN2 = math.log(2.0)
        row_sum[r] = (
            (row_max[r] * scale_log2 + utils.log2f(row_sum_cur)) * LN2
            if not acc_O_mn_row_is_zero_or_nan
            else -Float32.inf
        )

    return row_scale
```

**SM100 note:** The SM100 forward kernel does not call `Softmax.finalize()`; final scaling (and optional LSE writeback) is done in the correction loop after receiving the final `row_sum`/`row_max` from the softmax warps.

### The Logsumexp (LSE)

LSE is needed for the backward pass:
```
LSE = log(Σ exp(S_ij)) = max + log(Σ exp(S_ij - max))
```

In log2 space:
```python
# row_max is in original scale, row_sum is sum of exp2(...)
# Converting back:
LSE = (row_max * scale_log2 + log2(row_sum)) * ln(2)
```

---

## Integration in the Kernel

### Warp Specialization

In FA4/SM100, softmax runs in specialized warps:

```python
# flash_fwd_sm100.py lines 148-154
self.softmax0_warp_ids = (0, 1, 2, 3)    # Softmax for stage 0
self.softmax1_warp_ids = (4, 5, 6, 7)    # Softmax for stage 1
self.correction_warp_ids = (8, 9, 10, 11) # Correction warps
```

### The Softmax Step Flow

```python
# flash_fwd_sm100.py (simplified from lines 1908-1961)
# Called per KV tile

# 1. Update row max
row_max, acc_scale = softmax.update_row_max(tSrS_t2r.load(), is_first)

# 2. Publish the rescale factor for correction warps (non-first tiles)
if not is_first:
    sScale[thread_idx + stage * m_block_size] = acc_scale  # exp(old_max - new_max) in base-2

# 3. Notify correction warps that the scale / row_max update is ready
cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_softmax_corr_full_offset + stage)

# 4. Scale and subtract row max
softmax.scale_subtract_rowmax(tSrS_t2r, row_max)

# 5. Apply exp2 and convert to P
softmax.apply_exp2_convert(tSrS_t2r, tSrP_r2t, ...)

# 6. Update row sum (after correction has consumed the scale, to keep the pipeline ordered)
cute.arch.mbarrier_wait(
    mbar_ptr + self.mbar_softmax_corr_empty_offset + stage, si_corr_producer_phase
)
softmax.update_row_sum(tSrS_t2r.load(), acc_scale, is_first)
```

### Correction Warps

Correction warps do two distinct things:

1. **Inter-tile rescale:** for each non-first KV block, optionally rescale the running `O` accumulator by `acc_scale = exp(old_max - new_max)` (skip when `acc_scale == 1.0`).
2. **Finalize:** after the last KV block, apply `1 / row_sum` (and optional sink token) and optionally write LSE to global memory.

```python
# flash_fwd_sm100.py (simplified)
# --- inter-tile rescale loop ---
scale = sScale[tidx + stage * m_block_size]  # acc_scale written by softmax_step
should_rescale = cute.arch.vote_ballot_sync(scale < 1.0) != 0
if should_rescale:
    self.correction_rescale(..., scale)

cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage)
cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_softmax_corr_empty_offset + (1 - stage))

# --- finalization ---
row_sum = sScale[tidx + stage * m_block_size]  # overwritten by softmax warps after the last block
row_max = sScale[tidx + stage * m_block_size + m_block_size * 2]  # only if LSE/sink enabled

# Handle sink token if present
if sink_val is not None:
    if row_max == -Float32.inf:
        row_max = sink_val * (LOG2_E / softmax_scale_log2)
        row_sum = 1.0
    else:
        row_sum += utils.exp2f(sink_val * LOG2_E - row_max * softmax_scale_log2)

acc_O_mn_row_is_zero_or_nan = row_sum == 0.0 or row_sum != row_sum
scale = cute.arch.rcp_approx(row_sum if not acc_O_mn_row_is_zero_or_nan else 1.0)
self.correction_epilogue(..., scale, ...)

# Optional LSE writeback (later in correction_loop):
lse = (row_max * softmax_scale_log2 + utils.log2f(row_sum)) * LN2
```

---

## Key Design Decisions

### 1. Why Track Max Separately?

Could combine max and sum into LSE, but separate tracking:
- Allows cheaper rescaling (`exp(old_max - new_max)`)
- Enables the rescale threshold optimization
- Easier to handle special cases (-inf, NaN)

### 2. Why Quad Reduction (width=4)?

```python
row_max_cur = warp_reduce(row_max_cur, fmax, width=4)
```

In the generic `Softmax.online_softmax` path, the accumulator layout splits a logical row across a 4-lane group (“quad”). The code uses `warp_reduce(..., width=4)` because (for the layouts used here) 4 lanes collectively cover the full logical row that needs a single max/sum. This is about the kernel’s lane↔fragment mapping, not a hardware “SM100 rule” (and the SM100 path uses a different per-thread stats layout).

### 3. Why Store row_max and row_sum in Shared Memory?

```python
# flash_fwd_sm100.py lines 1739-1745
sScale[tidx + stage * m_block_size] = softmax.row_sum[0]
sScale[tidx + stage * m_block_size + m_block_size * 2] = softmax.row_max[0]
```

Softmax warps compute these; correction warps and epilogue warps need them. On SM100, the same `sScale[...]` slot is also used earlier to publish the per-block `acc_scale` used for rescaling `O` (the correction loop distinguishes “acc_scale” vs “row_sum/row_max” by which `mbar_softmax_corr_full` signal it is handling).

### 4. Why the Check for Zero/NaN?

```python
acc_O_mn_row_is_zero_or_nan = row_sum[r] == 0.0 or row_sum[r] != row_sum[r]
```

Empty rows (all masked) have sum=0. Using 1.0 instead prevents division by zero:
```python
row_scale[r] = cute.arch.rcp_approx(row_sum[r] if not invalid else 1.0)
```

---

## Numerical Stability Analysis

### The Problem

Naive: `softmax(S) = exp(S) / sum(exp(S))`

If `S[i] = 1000`, then `exp(1000) = inf`. Overflow!

### The Solution

Shift by max: `softmax(S) = exp(S - max) / sum(exp(S - max))`

Now the largest exp argument is 0, so `exp(0) = 1`. No overflow.

### Why Online Works

The key insight: even though we don't know the final max when processing early tiles, we can correct later:

```
Tile 1: max=100, compute P1 = exp(S1 - 100)
Tile 2: max=200 (new global max!)
        P1_corrected = P1 * exp(100 - 200) = P1 * exp(-100) ≈ 0

        P2 = exp(S2 - 200)
```

Early tiles get "scaled down" when larger scores arrive. This is numerically stable because we're multiplying by small numbers, not dividing.

---

## Questions for Further Investigation

1. **What's the optimal rescale threshold?**
   - Currently uses 8.0 for FP16, 0.0 for FP8
   - How was this tuned?

2. **E2E (emulated exp2) optimization:**
   - Code shows `ex2_emulation_2` path
   - When is emulation faster than hardware exp2?

3. **Multi-row processing:**
   - SM80/90 Softmax processes `num_rows > 1`
   - SM100 uses `num_rows = 1` in `SoftmaxSm100` (one running-stat row per thread)
   - Why the difference?

---

## References

### Code Files

| File | Description |
|------|-------------|
| `vendored/flash_attn_cute_score_mod/flash_attn/cute/softmax.py` | Softmax and SoftmaxSm100 classes |
| `vendored/flash_attn_cute_score_mod/flash_attn/cute/utils.py` | Reduction operations (fmax_reduce, fadd_reduce, warp_reduce) |
| `vendored/flash_attn_cute_score_mod/flash_attn/cute/flash_fwd.py` | Online softmax usage (SM80/90 paths) |
| `vendored/flash_attn_cute_score_mod/flash_attn/cute/flash_fwd_sm100.py` | SM100 integration (softmax_step + correction loop) |

### Papers

| Paper | What It Covers |
|-------|----------------|
| [FlashAttention](https://arxiv.org/abs/2205.14135) | Original online softmax algorithm |
| [FlashAttention-2](https://arxiv.org/abs/2307.08691) | Improved parallelization |
| [Online Normalizer Calculation](https://arxiv.org/abs/1805.02867) | Theoretical foundation |

### Related Explainers

| # | Topic | Relevance |
|---|-------|-----------|
| 5 | [TMA and Memory Loading](05-tma-memory-loading.md) | How tiles arrive |
| 6 | [Tile Scheduling](06-tile-scheduling.md) | Order tiles are processed |
| 3 | [score_mod](03-score-mod.md) | Modifying scores before softmax |
