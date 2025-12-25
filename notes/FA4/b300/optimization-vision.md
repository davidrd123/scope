# B300 Optimization Vision

> **Purpose:** Strategic roadmap for B300 performance optimization
> **Created:** 2025-12-25 | **Updated:** 2025-12-25
> **Current:** ~15 FPS (fp8) / ~16.7 FPS (qnone) @ 320×576 | **Target:** 24+ FPS (real-time)

---

## Executive Summary

We've achieved a **70% improvement** (8.8 → 15 FPS) through the cu130 runtime stack and FA4 score_mod optimization. The remaining bottleneck is **attention** (~79% of transformer time). This document lays out the strategic options for the next phase.

### Progress Log

| Date | Milestone | FPS | Key Change |
|------|-----------|-----|------------|
| 2025-12-24 | Baseline | 8.8 | repo default stack |
| 2025-12-24 | cu130 + segment-combine | 13.5 | cu130 fixed VAE decode |
| 2025-12-25 | **cu130 + FA4 score_mod** | **15.0** | FA4 KV-bias working (Option A) |

---

## 1. Where We Are

### Performance Baseline
| Metric | Before | cu130 + flash | cu130 + FA4 | Target |
|--------|--------|---------------|-------------|--------|
| FPS @ 320×576 | 8.8 | 13.5 | **15.0** | 24+ |
| VAE decode/frame | 760ms | 194ms | 194ms | - |
| Transformer share | ~46% | ~79% | ~79% | - |

### Backend Comparison (2025-12-25)
| Backend | FPS | Notes |
|---------|-----|-------|
| `SCOPE_KV_BIAS_BACKEND=fa4` | **15.01** | FA4 with score_mod (current best) |
| `SCOPE_KV_BIAS_BACKEND=flash` | 13.47 | segment-combine (3 FA calls) |
| `SCOPE_KV_BIAS_BACKEND=triton` | 1.07 | Triton Kernel B (broken on SM103) |

For Daydream-style runs on B300 we typically use `quantization none`. In that configuration, `SCOPE_KV_BIAS_BACKEND=fa4` reached ~`16.7 FPS` (vs ~`14.9` on `flash`) at `320x576`.

### Bottleneck Breakdown (Profiled 2025-12-25)

```
Pipeline Total: ~6000ms for 6 iterations
├── Transformer (denoise + recompute_kv): ~79%
│   └── Within Transformer:
│       ├── self_attn:  ~56%  ─┐
│       ├── cross_attn: ~22%  ├── ATTENTION (self+cross): ~78%
│       └── ffn:        ~22%  ─┘
└── VAE decode: ~20% (FIXED by cu130)
```

**Key insight:** Attention (self + cross) is ~78% of transformer time. Optimizing attention directly impacts the majority of end-to-end latency.

Nested self-attn detail (B300 cu130, `kv_cache_attention_bias=0.3`):
- With `SCOPE_KV_BIAS_BACKEND=flash`, KV-bias is ~38% of `self_attn` (~0.91ms/call).
- With `SCOPE_KV_BIAS_BACKEND=fa4`, KV-bias drops to ~22% of `self_attn` (~0.42ms/call); `other_in_self` becomes the majority.

### New Signal: Op-level GPU Profile (Copies + Elementwise + FP8 GEMMs)

An operator-level profile of one steady-state pipeline call (see `scripts/profile_krea_pipeline_ops.py`) shows a large amount of GPU time in:
- `aten::copy_` / `aten::to` / `aten::_to_copy` (dtype conversions + copies)
- FP8 GEMMs (`aten::_scaled_mm`)
- many elementwise kernels

Implication: after FA4 KV-bias is in place, the next wins may come from reducing **conversion/memory traffic** and improving **FP8 fastpaths**, not just swapping attention kernels.

---

## 2. Strategic Options

### Option A: Fix FA4 score_mod ✅ COMPLETE

**What:** Get FlashAttention-4 working with native score_mod for KV-bias, eliminating the segment-combine overhead.

**Result:** +1.5 FPS (13.5 → 15.0), **11% improvement** over segment-combine

**Fixes required (by Codex, 2025-12-25):**
1. Removed `import imageio` debug imports (cutlass-dsl AST preprocessor imports everything)
2. Fixed K/V slice stride normalization (`[0].unsqueeze(0)` views)
3. Disabled FA4 CuTe for non-bias attention when `SCOPE_KV_BIAS_BACKEND=fa4` to avoid mixing module variants

**Best config:**
```bash
SCOPE_KV_BIAS_BACKEND=fa4 \
TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas \
DISABLE_FLEX_ATTENTION_COMPILE=1 \
WANVAE_STREAM_DECODE_MODE=chunk
```

**Files changed:**
- `src/scope/core/pipelines/wan2_1/modules/attention.py` (+6 lines)
- `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py` (imageio removal)

---

### Option B: ThunderKittens Attention (High Impact, High Effort)

**What:** Integrate ThunderKittens kernels optimized for Blackwell's 128×128 tensor cores.

**Why it matters:**
- Blog claims 1400+ TFLOPS on B200 (near theoretical max)
- Designed specifically for Blackwell wgmma/tcgen05
- Already targets 128×128 tile sizes that match our head_dim=128

**Potential gain:**
- Could improve all attention ops (42.7% + 27.7% + 8.4% = 79%)
- If 2x faster attention: +50% overall speedup → ~22 FPS

**Effort:** High (new dependency, integration work, SM103 compat testing)

**Risk:** Research-grade code; may need patches for SM103

---

### Option C: cuDNN Attention Backend (Medium Impact, Low Effort)

**What:** Use PyTorch's cuDNN attention backend instead of FlashAttention.

**Why it matters:**
- cuDNN 9.13 (in cu130) has Blackwell-optimized attention
- Already integrated in PyTorch 2.9
- May "just work" with minimal code changes

**Potential gain:** Unknown, needs benchmarking

**Effort:** Low (configuration change, benchmark)

**Risk:** May not support all attention patterns (block masks, score_mod)

---

### Option D: Reduce KV Recompute ❌ REJECTED (Glitches)

**Status:** Rejected for real usage — confirmed visible glitches.

**Why not:** KREA's blog explicitly states KV Cache Recomputation is *essential* for preventing error accumulation in autoregressive video generation. They tried cheaper alternatives and rejected them:

> "Despite many attempts to devise cheaper solutions, we found these two techniques to be crucial for long, stable generations."

See `notes/krea/blog-krea-realtime-14b.md` for full rationale.

**What:** Change caching policy to reduce `recompute_kv_cache` frequency.

**Measured impact:**
- `recompute_kv_cache` is ~14% of pipeline time
- `SCOPE_KV_CACHE_RECOMPUTE_EVERY=2`: +12% FPS (15→16.8)
- **But breaks quality**: user observed visible glitches in Daydream on B300

**Experimental knob (for curiosity only):**

```bash
# Recompute KV cache every N blocks (default 1 = current behavior)
export SCOPE_KV_CACHE_RECOMPUTE_EVERY=2
```

Measured (B300 cu130, `320x576`, fp8, `kv_cache_attention_bias=0.3`, `SCOPE_KV_BIAS_BACKEND=fa4`):
- `SCOPE_KV_CACHE_RECOMPUTE_EVERY=1`: ~`15.0 FPS`
- `SCOPE_KV_CACHE_RECOMPUTE_EVERY=2`: ~`16.8 FPS` (alternates “fast/slow” iterations; **glitches**)

Shortcut: `scripts/bench_b300_recompute_cadence.sh` sweeps cadence values and writes logs/JSON under `outputs/`.

---

### Option E: torch.compile Transformer Blocks (Medium Impact, Medium Effort)

**What:** Apply regional compilation to transformer forward pass (not flex_attention).

**Why it matters:**
- PyTorch/Diffusers blog shows significant gains from regional compile
- Can fuse norms, projections, activations
- Avoid full model compile (graph breaks, long compile times)

**Current blocker:**
- `DISABLE_FLEX_ATTENTION_COMPILE=1` needed to avoid tcgen05 LLVM abort
- Need to compile around flex_attention, not through it
- **As of 2025-12-25:** `--compile` fails with fp8 (`torchao` Float8Tensor missing `aten.as_strided` under AOTAutograd) and FA4 score_mod can break Dynamo tracing via DLpack FakeTensor paths. Treat compile as experimental until those are resolved.

**Potential gain:**
- ffn + projections + rope = ~17% of transformer
- 20-30% speedup on these ops → +1-2 FPS

**Effort:** Medium (need to structure compilation carefully)

**Risk:** SM103 compiler issues; graph breaks; compile time overhead

---

### Option F: SageAttention (Unknown Impact, Low Effort)

**What:** Install and benchmark SageAttention package.

**Current state:** Not installed (`No module named 'sageattention'`)

**Potential gain:** Unknown, claims competitive with FlashAttention

**Effort:** Low (pip install, benchmark)

**Risk:** May not support SM103; limited documentation

---

## 3. Recommended Path

### ✅ Completed

1. **FA4 score_mod** (Option A) — DONE
   - Result: 13.5 → 15.0 FPS (+11%)
   - `SCOPE_KV_BIAS_BACKEND=fa4` is now the default best path

### Phase 1: Quick Wins (Next)

2. **Benchmark cuDNN attention** (Option C)
   - Lowest effort, may yield additional gains
   - Command: Set `TORCH_CUDNN_SDPA_ENABLED=1` or similar

2b. **Re-test quantization choice on B300**
   - B300 has ample VRAM; FP8 is not mandatory.
   - Early A/B indicates `--quantization none` can be faster than `fp8_e4m3fn` on this stack (likely due to conversion/scaling overhead).

3. **Install SageAttention** (Option F)
   - Quick test to see if it's viable
   - If works: free performance

4. **Profile with different kv_cache_attention_bias values**
   - Current: 0.3
   - Test: 1.0 (bypass bias path entirely)
   - Understand cost/benefit of bias feature

### Phase 2: Attention Optimization

5. **Evaluate ThunderKittens** (Option B)
   - Clone, build, benchmark on SM103
   - If works: could improve all attention ops (42.7% self_attn)

### Phase 3: Compile

6. **Regional torch.compile** (Option E)
   - Compile transformer blocks excluding flex_attention
   - Requires careful scoping to avoid tcgen05

---

## 6. 60-Minute Autonomous Loop (Next)

Goal: convert “`other_in_self` is big” into a concrete lever (GEMM fastpaths vs compiler fusion vs layout overhead).

1) **Lock a baseline (FA4, steady-state)**
   - Run `scripts/bench_b300_kv_bias_backends.sh` with `BACKENDS="fa4"` at `320x576`, fp8, bias `0.3` and record FPS + warmup time.
   - Run `PROFILE_ATTENTION=1` once (accept lower FPS) to capture:
     - Transformer Block Split (top-level): `self_attn`/`cross_attn`/`ffn`
     - self_attn Breakdown (nested): `kv_bias_total` vs `other_in_self`

2) **Fix `torchao` ABI mismatch in the cu130 env**
   - In `.venv-b300-cu130-decode`, install a PyPI `torchao` wheel (default `TORCHAO_VERSION=0.15.0`) to remove:
     - `Skipping import of cpp extensions due to incompatible torch version ...`
   - Re-run the same baseline + profiler and compare:
     - `qkv_projection` and `output_projection` ms/call (should improve if FP8 fastpaths were missing)

3) **Try “regional compile” without compiling flex_attention**
   - Run `scripts/profile_krea_pipeline_blocks.py --compile` with `DISABLE_FLEX_ATTENTION_COMPILE=1` and `SCOPE_KV_BIAS_BACKEND=fa4`.
   - If it fails, capture the exact error and decide whether we need finer-grained compilation (compile FFN/projections only).

4) **Write down what moved**
   - Update `notes/FA4/b300/session-state.md` with:
     - before/after FPS
     - before/after `qkv_projection` + `output_projection` timings
     - whether `--compile` helped or failed on SM103

## 4. Success Metrics

| Milestone | FPS Target | Key Change | Status |
|-----------|------------|------------|--------|
| Baseline | 8.8 | repo default | ✅ |
| cu130 stack | 13.5 | VAE decode fixed | ✅ |
| **FA4 score_mod** | **15.0** | Option A complete | ✅ **CURRENT** |
| Phase 1 | 17-18 | cuDNN/Sage + bias tuning | ⏳ Next |
| Phase 2 | 20-22 | ThunderKittens | Planned |
| Phase 3 | 24+ | Compile + algorithmic | Planned |

---

## 5. Key Files Reference

| Purpose | File |
|---------|------|
| This vision | `notes/FA4/b300/optimization-vision.md` |
| Session state | `notes/FA4/b300/session-state.md` |
| External docs | `notes/FA4/b300/blackwell-docs.md` |
| Investigation runbook | `notes/FA4/b300/investigation-runbook.md` |
| Profiling script | `scripts/profile_krea_pipeline_blocks.py` |
| Attention code | `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py` |
| FA4 backend fix | `src/scope/core/pipelines/wan2_1/modules/attention.py` |

### Benchmark Artifacts
| Backend | Log | JSON |
|---------|-----|------|
| FA4 | `outputs/b300_cu130_fp8_e4m3fn_bias0.3_kvbias_fa4.log` | `outputs/b300_cu130_fp8_e4m3fn_bias0.3_kvbias_fa4.json` |
| Flash | `outputs/b300_cu130_fp8_e4m3fn_bias0.3_kvbias_flash.log` | - |
| Triton | `outputs/b300_cu130_fp8_e4m3fn_bias0.3_kvbias_triton.log` | - |

---

## 6. Open Questions

1. **What's the quality impact of `kv_cache_attention_bias=1.0`?** (bypasses bias entirely)
2. **Can we compile transformer blocks without triggering tcgen05?** (need to scope carefully)
3. **Is ThunderKittens SM103-ready?** (blog only mentions B200/SM100)
4. **What's the theoretical max FPS at 320×576?** (need to calculate based on compute/memory limits)

---

## Appendix: Blog References

See `notes/FA4/b300/blackwell-docs.md` Section 6 for detailed blog summaries:
- ThunderKittens on Blackwell (attention optimization)
- FlexDecoding (inference-time attention)
- PyTorch + Diffusers (regional compilation)
- GPU sync elimination patterns
