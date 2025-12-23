# Phase 3 – Step 3 (Triton RoPE Optimization) Spec

## Status (2025-12-23)

**NOT STARTED** - Planning doc for future work.

**Baseline:** Step 2 v2 kernel achieves **20.2 FPS** with BLOCK_M=8, BLOCK_H=2.

---

## Goals

1. **Tune v2 kernel** - find optimal BLOCK_M/BLOCK_H for B200
2. **Fuse Q+K RoPE** - halve launch overhead (one kernel call instead of two)
3. **Micro-optimizations** - reduce wrapper overhead further

---

## 1. BLOCK_M/BLOCK_H Tuning

Current defaults: `BLOCK_M=8`, `BLOCK_H=2` (copied from FlashAttention rotary)

### Tuning space

| BLOCK_M | BLOCK_H | Threads | Notes |
|---------|---------|---------|-------|
| 8 | 2 | 16×M | Current default |
| 16 | 2 | 32×M | More tokens per program |
| 32 | 2 | 64×M | May hit register pressure |
| 8 | 4 | 32×M | More heads per program |
| 16 | 4 | 64×M | Balanced |

### Benchmark commands

```bash
# Baseline (current)
SCOPE_TRITON_ROPE_FUSED_STRICT=1 \
uv run python scripts/bench_triton_rope.py --pad-to-multiple 128 --iters 100

# Sweep BLOCK_M
for M in 8 16 32; do
  echo "=== BLOCK_M=$M ==="
  SCOPE_TRITON_ROPE_FUSED_BLOCK_M=$M \
  uv run python scripts/bench_triton_rope.py --pad-to-multiple 128 --iters 100
done

# Sweep BLOCK_H
for H in 2 4 8; do
  echo "=== BLOCK_H=$H ==="
  SCOPE_TRITON_ROPE_FUSED_BLOCK_H=$H \
  uv run python scripts/bench_triton_rope.py --pad-to-multiple 128 --iters 100
done
```

### What to measure

- Kernel time (direct)
- End-to-end `rope_apply` time
- Full pipeline FPS (if kernel time changes significantly)

---

## 2. Fuse Q+K RoPE

### Current flow

```
rope_apply(q, grid_sizes, freqs)  # kernel launch #1
rope_apply(k, grid_sizes, freqs)  # kernel launch #2
```

Both calls use:
- Same `grid_sizes` (f, h, w)
- Same `freqs` (same cos/sin tables)
- Same `start_frame`

### Proposed fused flow

```python
def rope_apply_qk(q, k, grid_sizes, freqs, start_frame=0):
    """Apply RoPE to both q and k in a single kernel launch."""
    # Single kernel processes both tensors
    # Halves: launch overhead, cos/sin table loads, index computation
```

### Kernel design options

**Option A: Stack Q+K along batch dim**
```python
# Stack: [B, L, H, D] + [B, L, H, D] -> [2*B, L, H, D]
qk = torch.cat([q, k], dim=0)
out = rope_fused_3way(qk, ...)
q_out, k_out = out.chunk(2, dim=0)
```
- Pro: No kernel changes needed
- Con: Extra concat/chunk overhead, may not fit in registers

**Option B: Dedicated Q+K kernel**
```python
@triton.jit
def rope_fused_3way_qk_kernel(
    Q_ptr, K_ptr, Q_OUT_ptr, K_OUT_ptr, ...
):
    # Load Q and K for same (batch, token, head)
    # Apply same cos/sin to both
    # Store both outputs
```
- Pro: True fusion, one set of index math
- Con: More kernel code to maintain

**Option C: Multi-tensor launch (if Triton supports)**
- Use Triton's multi-tensor capabilities if available

### Expected gains

- ~50% reduction in kernel launch overhead
- ~50% reduction in cos/sin table cache lookups
- Real impact: probably 0.01-0.03ms per attention layer

---

## 3. Micro-optimizations

### Wrapper overhead (already low, but check)

- `get_rope_axis_tables()` cache hit rate
- `_as_int3()` conversion overhead
- Tail copy when `seq_len < L`

### Potential PTX-level wins

- Check if Triton generates optimal `LDG.128` for cos/sin loads
- Verify no register spills with NCU

---

## 4. Integration with FA4/CUTE

Now that cuda-python is unblocked, consider:

- Can CUTE `score_mod` absorb RoPE? (probably not worth it)
- Does FA4 have its own RoPE path we should use instead?

---

## Files

- Kernel: `src/scope/core/kernels/triton_rope_fused.py`
- Benchmark: `scripts/bench_triton_rope.py`
- Correctness: `scripts/test_rope_correctness.py`

---

## Acceptance Criteria

- [ ] Find optimal BLOCK_M/BLOCK_H (may be status quo)
- [ ] Evaluate Q+K fusion ROI (implement if >5% win)
- [ ] Document final tuning in phase3-triton-rope.md
