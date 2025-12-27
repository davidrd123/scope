# Phase 3 Triton RoPE Plan

## Context

- Baseline is Phase 2.5 (cached cos/sin). Phase 2.6 "broadcast" regressed FPS and was rolled back.
- Goal of Phase 3: fuse all RoPE math into a single Triton kernel to avoid intermediate tensors and reduce launch overhead.

## Data Layout and Mapping (must match current semantics)

- `x` shape: `[B, L, H, D]` where `D = 2*C` and pairs are contiguous: `x[..., 2*j]`, `x[..., 2*j+1]`.
- Token to grid mapping:
  - `seq_len = f * h * w`
  - `f_idx = start_frame + t // (h * w)`
  - `h_idx = (t % (h * w)) // w`
  - `w_idx = t % w`
- `freqs` is complex and split into 3 column chunks:
  - `C0 = C - 2 * (C // 3)` (time), `C1 = C // 3` (height), `C2 = C // 3` (width)
  - D=128 -> C=64 -> chunks 22/21/21
  - Current Python impl concatenates these chunks (not multiply).
- `start_frame` only offsets the time axis.
- Tail tokens (`t >= seq_len`) must be preserved; kernel should not write them.

## Incremental Plan

### Step 0 (optional baseline) - COMPLETE (2025-12-23)

**Status:** Integrated and working.

Used FlashAttention's Triton rotary kernel with pre-materialized cos/sin.

**Files changed:**
- Copied: `vendored/rope/flash_attn_triton_rotary.py` → `src/scope/core/kernels/triton_rotary.py`
- Modified: `rope_apply()` and `causal_rope_apply()` to use Triton when available
- Feature flag: `SCOPE_DISABLE_TRITON_ROTARY=1` to force PyTorch fallback

**Benchmark results:**

| Component | Time (ms) | Notes |
|-----------|-----------|-------|
| PyTorch rope_apply | 0.200 | Baseline |
| Direct Triton kernel | 0.101 | **1.98x faster** |
| Integrated rope_apply | 0.183 | Only 8.5% faster (overhead) |

**Overhead breakdown:**

| Step | Time (ms) |
|------|-----------|
| Prep (slice+squeeze) | 0.003 |
| Triton kernel | 0.101 |
| Post (squeeze+cat) | 0.025 |
| **Expected total** | 0.129 |
| **Actual integrated** | 0.183 |
| **Unexplained gap** | 0.054 |

The 0.054ms gap comes from Python loop overhead: `grid_sizes.tolist()`, cache lookup, `freqs.split()`, `torch.stack()`.

**Correctness:**
- FP32: Exact match (diff < 1e-7)
- BF16: Max diff 0.03125 (1 bf16 ULP), 99.5% within 0.01
- FA uses fp32 internally = better numerics than our PyTorch path

**Next:** Reduce integration overhead OR proceed to Step 1 custom kernel.

#### Step 0.5: Wrapper overhead cleanup (do this before writing a new kernel)

The Step 0 result shows the FA rotary kernel is already ~2× faster, but we lose most of the win to wrapper overhead and tensor assembly. Before investing in a custom Triton kernel, remove the obvious Python + `torch.cat/stack` costs so we can re-measure the “true” upside of Step 1.

Concrete checklist:
- Special-case **`B==1`** (Krea realtime steady-state) and avoid per-sample `tolist()` loops + `torch.stack(output)`.
- Avoid `torch.cat` when `seq_len == x.shape[1]` (common in causal path); `cat` forces an allocate+copy even with an empty tail.
- Only `clone()` the full tensor when `seq_len < L` (padding case); otherwise return the kernel output directly.
- Optional follow-up: inference-only **in-place** rotation (guarded by `not torch.is_grad_enabled()`), if we’re comfortable with `rope_apply` mutating `q/k`.

### Step 1 (lowest risk)
Simple Triton kernel that rotates `x` using pre-materialized `cos/sin`:
- Wrapper builds `cos/sin[seq_len, C]` using `get_rope_cos_sin()`.
- Kernel only does rotation and masked store for `t < seq_len`.
- Validate vs `rope_apply` / `causal_rope_apply`.

**Current status:** Step 1 is already wired via `triton_apply_rotary` in:
- `src/scope/core/pipelines/krea_realtime_video/modules/model.py`
- `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py`

**Benchmark harness:** `scripts/bench_triton_rope.py`

**Step 1 microbench (B200, bf16):**
- Shape: `f=3, h=30, w=52`, `B=1`, `H=16`, `D=128`, `seq_len=4680`
- Correctness vs PyTorch reference: `max_err=0.03125`, `mean_err=0.000595`
- PyTorch (cached cos/sin): **0.097 ms**
- Triton (flash-attn rotary): **0.129 ms**
- Result: Triton is slower here; Step 1 is not a win.

### Step 2 (fuse 3-way lookup) - COMPLETE (2025-12-23)

**Status:** v2 kernel fixes regression. **20.2 FPS** (matches/exceeds Step 0 baseline).

**Files created/modified:**
- Created: `src/scope/core/kernels/triton_rope_fused.py` (v1 + v2 kernels + wrapper)
- Modified: `model.py` and `causal_model.py` to add fused fast path
- Feature flags:
  - `SCOPE_DISABLE_TRITON_ROPE_FUSED=1` - disable fused path
  - `SCOPE_TRITON_ROPE_FUSED_STRICT=1` - crash instead of fallback
  - `SCOPE_TRITON_ROPE_FUSED_IMPL=v1|v2` - kernel selection (default v2)
  - `SCOPE_TRITON_ROPE_FUSED_BLOCK_M/BLOCK_H` - v2 tuning (default 8/2)

**v1 (regressed):**
- Padded chunk sizes 22/21/21 → 32/32/32 = 96 pairs (50% extra work)
- Full pipeline: 20 → 17.8 FPS (11% regression)

**v2 (fixed):**
- Key insight: C=64 is already power-of-2
- Uses unified `tl.arange(0, 64)` with masks to select from 3 axis tables
- FlashAttn-style tiling: BLOCK_M=8, BLOCK_H=2
- Uses `tl.split/tl.join` for rotation (proven in triton_rotary.py)

**Benchmark (padded L=4736):**
| Implementation | rope_apply time | vs Step 0 |
|----------------|-----------------|-----------|
| Step 0 (FA rotary) | 0.153 ms | baseline |
| v1 (padded) | 0.125 ms | 1.22x |
| **v2 (unified 64)** | **0.069 ms** | **2.22x** |

**Full pipeline:** 20.2 FPS (v2 default)

**Block tuning (300 iters, pad-to-multiple=128):**
- Sweeped `BLOCK_M ∈ {8,16}`, `BLOCK_H ∈ {1,2}`.
- Results were extremely close; best was **M=8, H=2**:
  - `rope_apply` ~0.066 ms, `triton_rope_fused_3way` ~0.065 ms.
- Decision: keep default `BLOCK_M=8`, `BLOCK_H=2`.

**Docs:**
- [`phase3-triton-rope-step2.md`](../phase3-triton-rope-step2.md) - Step 2 spec
- [`Phase3_01.md`](../DeepResearch/2025-12-22/Phase3_01.md) - GPT-5 Pro v1 draft
- [`Phase3_02.md`](../DeepResearch/2025-12-22/Phase3_02.md) - GPT-5 Pro v2 fix diagnosis
- [`Phase3_03.md`](../DeepResearch/2025-12-22/Phase3_03.md) - GPT-5 Pro v2 alternative impl

### Step 3 (optimize access + launch) - PLANNED

**Status:** Planning doc ready at [`phase3-triton-rope-step3.md`](../phase3-triton-rope-step3.md)

**Goals:**
1. Tune BLOCK_M/BLOCK_H for v2 kernel (sweep 8/16/32)
2. Fuse Q+K RoPE in single kernel (halve launch overhead)
3. Micro-optimizations (wrapper overhead, PTX-level)

**Spec:** [`phase3-triton-rope-step3.md`](../phase3-triton-rope-step3.md)

## Reference Artifacts (vendored)

- `vendored/rope/flash_attn_triton_rotary.py`
- `vendored/rope/vllm_rotary_embedding_common.py`
- `vendored/rope/vllm_rotary_embedding_base.py`
- `vendored/rope/transformers_modeling_llama.py`

## Profiling Notes (Kernel B, B200)

NCU on the **real shape** (B=1, H=16, Lq=4680, Lk=9360, D=128) using `scripts/tune_kernel_b.py` shows:
- Compute (SM) throughput: **~39%**
- Memory throughput: **~12%** (DRAM ~0.7%)
- L2 hit rate: **~87%**
- Regs/thread: **178**
- Dynamic shared mem: **~82 KB**
- Waves/SM: **4**

Interpretation: Kernel B is **latency/occupancy limited**, not memory-bound. Further Kernel B tuning likely has limited upside. Focus should move to RoPE fusion.

**Important:** NCU runs on `scripts/triton_sdpa.py --kernel-b` can profile the small test shape (grid size = 1). Use `scripts/tune_kernel_b.py` to ensure a full grid for meaningful metrics.

## Nsight Systems (NSYS) Notes

NVTX range name for Triton Kernel B benchmark:
- `kernel_b_triton_bench` (added in `scripts/triton_sdpa.py`)

Recommended capture:
```
nsys profile --trace=cuda,nvtx,osrt \
  --capture-range=nvtx --nvtx-capture=kernel_b_triton_bench \
  --capture-range-end=stop-shutdown \
  --force-overwrite=true -o /root/scope/nsys_kernel_b_nvtx \
  uv run python scripts/triton_sdpa.py --kernel-b

nsys stats --force-export=true /root/scope/nsys_kernel_b_nvtx.nsys-rep
```

If NVTX still does not show up:
- Run the NVTX smoke test (simple `range_push`/`range_pop`) to verify.
- Use a kernel-only harness (e.g., `scripts/tune_kernel_b.py`) without capture-range to avoid correctness tests in the trace.

## Acceptance Criteria

- Correctness: `torch.allclose` vs current `rope_apply` / `causal_rope_apply` for start_frame=0 and >0.
- Performance: Step 1 should at least match Phase 2.5; Step 2 should improve further without regressions.
- Stability: no layout assumptions broken (watch SAGEATTN path if enabled).
