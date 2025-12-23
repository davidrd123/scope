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

### Step 2 (fuse 3-way lookup) - IMPLEMENTED, REGRESSION (2025-12-23)

**Status:** Implemented and integrated, but causes FPS regression (20 → 17.8 FPS). Disabled by default pending fix.

**Files created/modified:**
- Created: `src/scope/core/kernels/triton_rope_fused.py` (kernel + wrapper)
- Modified: `model.py` and `causal_model.py` to add fused fast path
- Feature flags:
  - `SCOPE_DISABLE_TRITON_ROPE_FUSED=1` - disable fused path (use this for now)
  - `SCOPE_TRITON_ROPE_FUSED_STRICT=1` - crash instead of fallback
  - `SCOPE_TRITON_ROPE_FUSED_BLOCK_L/NUM_WARPS/NUM_STAGES` - tuning

**What was fixed:**
- Clone issue: Uses `empty_like()` + tiny tail copy instead of `x.clone()` for padded case
- Power-of-2: Pads chunk sizes (22/21/21 → 32/32/32) with masks for Triton `arange`

**Root cause of regression:**
Power-of-2 padding overhead. Processing 96 pairs instead of 64 = **50% more work**.

**Microbench (misleading):**
| Case | Step 0 | Step 2 | Speedup |
|------|--------|--------|---------|
| Unpadded (L=4680) | 0.126 ms | 0.121 ms | 4% |
| Padded (L=4736) | 0.165 ms | 0.124 ms | 25% |

But full pipeline: **20 → 17.8 FPS** (11% regression).

**Fix options:**
1. Channel tiling - process chunks in power-of-2 tiles (e.g., 16 at a time) with loop
2. Match FA's approach - they handle non-power-of-2 with BLOCK_K tiling
3. Hardcode D=128 unrolled path without padding

**Spec + design doc:**
- `notes/FA4/phase3-triton-rope-step2.md`

**GPT-5 Pro draft:**
- `notes/FA4/DeepResearch/2025-12-22/Phase3_01.md`

### Step 3 (optimize access + launch)
- Tune `BLOCK_L` and launch params for target shapes (likely 128/256, warps 4/8, stages 2).
- Consider fusing Q+K in a single kernel to avoid duplicate cos/sin loads and launch overhead.
- Consider storing freqs as real cos/sin (fp32) to avoid complex handling in Triton.

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
