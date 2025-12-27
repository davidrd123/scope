# "other_in_self" Breakdown: Stack-Attributed Analysis

> Status: Partial (coarse + stack snapshots recorded; needs periodic refresh)
> Priority: **High** — can't optimize what we haven't measured
> Date: 2025-12-27
> Sources:
> - External references + constraints: [`claude01.md`](../DeepResearch/2025-12-26/B300_step_back/doc_ref_guide/claude01.md)
> - “Don’t guess, measure” decision rules: [`5pro_rp02.md`](../DeepResearch/2025-12-26/B300_step_back/round01/5pro_rp02.md)

---

## Execution Sequence (Do This First!)

The fastest "first execution" sequence to prevent spending weeks on a gorgeous kernel that targets the wrong thing:

1. **Fill `layout-contracts.md`** for exactly one representative shape end-to-end
2. **Do ONE stack-attributed `other_in_self` run** and fill this breakdown table
3. **THEN pick** between: pack-only kernel, RoPE+pack kernel, or KV-write kernel

Don't skip to step 3. The breakdown data determines which kernel is worth building.

---

## Decision Rules (From Profiling Data)

> **Key insight**: The breakdown results dictate which kernel to build first.

| If this dominates... | Then your first kernel is... | Rationale |
|----------------------|------------------------------|-----------|
| `layout_copies` (transpose, contiguous, _to_copy) | **Pack kernel** | Layout transforms are pure overhead |
| `rope_apply` | **Fused RoPE + pack kernel** | Combine RoPE compute with layout fix |
| `cache_write` (K/V writes) | **Focus on cache layout first** | Write path often bandwidth-bound |
| QKV projection | **Not a kernel target** — it's cuBLAS | Don't compete with cuBLAS |

**These are decision rules, not assumptions.** The point of this doc is to stop guessing.

---

## Purpose

The profiler shows "other_in_self" as the majority of self_attn time after FA4 KV-bias optimization. This doc breaks down exactly what's in that bucket using stack-attributed profiling.

---

## How to Generate This Data

### Step 0: Get the coarse breakdown (cheap)

Enable `PROFILE_ATTENTION=1` and record the report at exit. This tells you how much `self_attn` time is KV-bias vs everything else, and also records sub-blocks like `qkv_projection`, `rope_apply`, and `cache_update` when enabled.

> Note: `PROFILE_ATTENTION=1` is **not compatible with `--compile`** for the self-attn regions (the profiling blocks get pruned during tracing). Run without `--compile` to get the nested breakdown numbers.

**Recommended command (B300 baseline, BF16, bias=0.3, FA4; no compile):**
```bash
OUT_PREFIX=outputs/b300_cu130_none_bias0.3_drilldown_2025-12-27_clean \
ITERS=4 SKIP=1 \
scripts/profile_b300_denoise_drilldown.sh
```

**Artifacts (latest):**
- `outputs/b300_cu130_none_bias0.3_drilldown_2025-12-27_resume_perf.log`
- `outputs/b300_cu130_none_bias0.3_drilldown_2025-12-27_resume_perf.json`
- `outputs/b300_cu130_none_bias0.3_drilldown_2025-12-27_resume_blocks_profile.json`
- `outputs/b300_cu130_none_bias0.3_drilldown_2025-12-27_resume_denoise_steps.json`
- `outputs/b300_cu130_none_bias0.3_drilldown_2025-12-27_resume_generator_steps.json`
- `outputs/b300_cu130_none_bias0.3_drilldown_2025-12-27_resume_vae_decode.json`
- `outputs/b300_cu130_none_bias0.3_drilldown_2025-12-27_resume_vae_decode_inner.json`

**Previous snapshot (same config, earlier run):**
- `outputs/b300_cu130_none_bias0.3_drilldown_2025-12-27_clean_perf.log`

**Observed `PROFILE_ATTENTION` report (latest, from `*_resume_perf.log`, profiled iterations only):**

| Component | Total (ms) | Calls | ms/call | Notes |
|-----------|-----------:|------:|--------:|------|
| `self_attn` | 675.0 | 600 | 1.13 | Includes KV-bias + all glue |
| `self_attn_kv_bias_fa4` | 195.6 | 480 | 0.41 | KV-bias portion only |
| `self_attn_block_mask` | 32.8 | 120 | 0.27 | Recompute / block-mask path |
| `qkv_projection` | 165.3 | 600 | 0.28 | cuBLAS GEMM(s) + norms |
| `rope_apply` | 53.5 | 480 | 0.11 | Triton fused RoPE enabled |
| `cache_update` | 28.5 | 480 | 0.06 | KV-cache writes / indices |
| `output_projection` | 65.3 | 600 | 0.11 | cuBLAS GEMM |

**Derived (nested) breakdown (latest):**
- `other_in_self` = **446.7ms** (**66.2%** of `self_attn`)
- `kv_bias_total` = **195.6ms** (**29.0%** of `self_attn`)
- `block_mask` = **32.8ms** (**4.9%** of `self_attn`)

**Remainder estimate (why Level 5 exists):**
- `other_in_self` includes `qkv_projection` + `rope_apply` + `cache_update` + `output_projection`, **plus** additional overhead not explicitly labeled.
- Remainder ≈ `446.7ms - (165.3 + 53.5 + 28.5 + 65.3)ms = 134.1ms` (≈**30%** of `other_in_self`) → this is the “what is it?” bucket we need stack attribution for.

### Step 1: Stack-attributed op profiling (to find the real copy sources)

```bash
DISABLE_FLEX_ATTENTION_COMPILE=1 \
WANVAE_STREAM_DECODE_MODE=chunk \
WANVAE_DECODE_CHANNELS_LAST_3D=1 \
WANVAE_RESAMPLE_ENSURE_CONTIGUOUS=1 \
SCOPE_KV_BIAS_BACKEND=fa4 \
SCOPE_ENABLE_FA4_VARLEN=1 \
PYTHONPATH=src \
.venv-b300-cu130-triton351/bin/python scripts/profile_krea_pipeline_ops.py \
  --height 320 --width 576 \
  --iters 1 --pre-iters 1 \
  --kv-cache-attention-bias 0.3 \
  --kv-bias-backend fa4 \
  --quantization none \
  --cudnn-benchmark \
  --compile \
  --with-stack --stack-n 12 \
  --stack-key aten::copy_ \
  --stack-key aten::contiguous \
  --stack-key aten::clone \
  --stack-key aten::_to_copy \
  --stack-key aten::fill_ \
  --stack-include CausalWanSelfAttention \
  --stack-filter-top 40 \
  --summary outputs/b300_cu130_triton351_ops_profile_selfattn_compile_fa4_varlen_stack_topops.md \
  --json outputs/b300_cu130_triton351_ops_profile_selfattn_compile_fa4_varlen_stack_topops.json \
  |& tee outputs/b300_cu130_triton351_ops_profile_selfattn_compile_fa4_varlen_stack_topops.log
```

Filter output for self-attn-related stacks (look for call stacks that include `CausalWanSelfAttention.forward`).

**Kickoff op+stack snapshot (whole pipeline, 1 iteration):**
- `outputs/b300_cu130_ops_profile_stack.md`
- `outputs/b300_cu130_ops_profile_stack.json`

Quick read:
- Top CUDA self time is dominated by `aten::addmm` (GEMMs).
- FA4 attention appears as `kernel_cutlass_kernel_flash_attncuteflash_fwd_sm100...`.
- `aten::copy_`/`aten::fill_` are present; the largest `aten::copy_` stack groups in this snapshot point at VAE Conv3d decode, while attention-related copies exist but are smaller and require stack filtering.

**Self-attn–scoped stack filter (latest, compile on; CausalWanSelfAttention-only):**
- `outputs/b300_cu130_ops_profile_selfattn_compile_fa4_stack_2025-12-27_resume.md`
- `outputs/b300_cu130_ops_profile_selfattn_compile_fa4_stack_2025-12-27_resume.json`

Key findings (filtered to `CausalWanSelfAttention` stacks):

| Op key | device_ms (total) | calls | Notes |
|--------|------------------:|------:|------|
| `aten::copy_` | 5.305 | 328 | Stacks point at RoPE call sites in `causal_model.py:1190–1196` (likely tail-preservation copy in `rope_fused_3way`, not a dtype cast) |
| `aten::contiguous` | (no events) | 0 | Good: no obvious layout fixups attributed here in this config |
| `aten::clone` | (no events) | 0 | |
| `aten::_to_copy` | (no events) | 0 | |
| `aten::to` | (no events) | 0 | |
| `aten::fill_` | ~0.000 | 8 | Small counters/indices (near-zero on GPU) |

Notable from the **stack-filtered top-op table** in the same artifact:
- The bulk of time is still in GEMMs (NVJET / `aten::mm` / `aten::addmm`) and the FA4/CuTe forward kernel.
- Non-trivial “glue” survives compile inside self-attn stacks:
  - `aten::copy_`: **5.305ms** total
  - `direct_copy_kernel_cuda` (elementwise copy): **4.421ms** total
  - `Memcpy DtoD`: **0.884ms** total

**Quick A/B (sanity check):** enabling the existing K-side fused path (`SCOPE_ROPE_K_TO_CACHE=1`) was a very small/noisy win in a short run
(`iters=6,skip=2`: `~33.41 → ~33.55 FPS`), suggesting K-side cache-write fusion is **not** the dominant remaining overhead.

**Previous snapshot (triton351 env):**
- `outputs/b300_cu130_triton351_ops_profile_selfattn_compile_fa4_varlen_stack_topops.md`
- `outputs/b300_cu130_triton351_ops_profile_selfattn_compile_fa4_varlen_stack_topops.json`
- `outputs/b300_cu130_triton351_ops_profile_selfattn_compile_fa4_varlen_stack_topops.log`

#### Scalar sync cleanup (fixed)

We observed a high-volume `Memcpy DtoH (Device -> Pinned)` + `aten::item` pattern inside `CausalWanSelfAttention` stacks.
Root cause: `initialize_kv_cache(...)` stored `kv_cache["global_end_index"]` / `kv_cache["local_end_index"]` as **1-element CUDA tensors**, and these
ended up used in Python control-flow / slicing, triggering implicit `.item()` conversions and GPU→CPU sync.

Fix: keep the index tensors on **CPU** (still tensors, so the object identity stays stable across iterations for compile/cudagraph friendliness).

Evidence:
- Before: `outputs/b300_cu130_triton351_ops_profile_selfattn_item_stack_2025-12-27.md`
  - `Memcpy DtoH (Device -> Pinned)`: present (1280 calls)
  - `aten::item`: non-zero device time (sync)
- After: `outputs/b300_cu130_triton351_ops_profile_selfattn_item_stack_cpuindices_2025-12-27.md`
  - `Memcpy DtoH (Device -> Pinned)`: **(no events)**
  - `aten::item`: still present, but **device_ms=0.000** (no CUDA sync)

**Historical snapshot (pre-triton351 / different config):**
- `outputs/b300_cu130_ops_profile_selfattn_stack.md` (shows `aten::contiguous` / `aten::clone` present under self-attn stacks)

### Hypothesis to validate (don’t assume): fused projections vs separate projections

Fused projections can introduce strided Q/K/V views. In some configs this has correlated with additional copies/contiguity work, but this is **config-dependent** (compile mode, env, and attention backend).

If you want to test this today, do an A/B in the same env with:
- Baseline: fused projections ON (default)
- Change: `SCOPE_DISABLE_FUSED_PROJECTIONS=1`

Then compare:
- `PROFILE_ATTENTION` (no compile): `other_in_self` remainder and `cache_update` time
- Stack-filtered op profile (compile on): presence/absence of `contiguous`/`clone` and changes in `aten::copy_` stacks

---

## Breakdown Template

Fill in after profiling run:

### Total self_attn Time

| Component | Time (ms) | % of self_attn | Calls | Notes |
|-----------|-----------|----------------|-------|-------|
| **kv_bias_attention** | ? | ? | ? | FA4 score_mod path |
| **other_in_self** | ? | ? | ? | Everything else |
| **Total** | ? | 100% | | |

### other_in_self Decomposition

| Op / Stack Group | Time (ms) | % of other_in_self | Call Count | Stack Signature |
|------------------|-----------|-------------------|------------|-----------------|
| QKV projection | ? | ? | ? | `linear → ...` |
| Q/K RoPE application | ? | ? | ? | `rope → ...` |
| K cache write | ? | ? | ? | `cache → ...` |
| V cache write | ? | ? | ? | `cache → ...` |
| Transpose (Q→attn layout) | ? | ? | ? | `transpose → ...` |
| Transpose (K→attn layout) | ? | ? | ? | `transpose → ...` |
| Contiguous calls | ? | ? | ? | `contiguous → ...` |
| _to_copy (dtype) | ? | ? | ? | `_to_copy → ...` |
| Output projection | ? | ? | ? | `linear → ...` |
| Other | ? | ? | ? | |

---

## Top Stack Groups (Ranked by Time)

After profiling, paste the top 10 stack groups here:

```
# Example format:
1. [XX.X ms, YY%] aten::linear
   Stack: attention.py:123 → self_attn → qkv_proj

2. [XX.X ms, YY%] aten::transpose
   Stack: attention.py:145 → self_attn → reshape_for_attn

3. ...
```

---

## Fusion Candidates

Based on the breakdown above:

| Candidate Fusion | Ops to Combine | Expected Savings | Difficulty |
|------------------|----------------|------------------|------------|
| QKV + RoPE | `linear` + `rope` | ? ms | Hard (GEMM epilogue) |
| RoPE + K cache write | `rope` + `cache_write` | ? ms | Medium |
| RoPE + transpose | `rope` + `transpose` | ? ms | Easy |
| Layout elimination | Multiple `transpose`/`contiguous` | ? ms | Medium |

---

## Key Questions to Answer

1. [ ] **Is QKV fused or separate?** — Affects fusion strategy
2. [ ] **Where do transposes come from?** — Layout mismatch between proj and attn?
3. [ ] **Are contiguous calls avoidable?** — Can we change upstream layout?
4. [ ] **RoPE precision** — Does it use fp32 intermediates?
5. [ ] **Cache write pattern** — Append or scatter? Affects TMA viability.

---

## Snapshot Log (Evidence)

Keep these around so future work can cite concrete artifacts instead of “vibes”:

- Coarse (no compile) `PROFILE_ATTENTION` snapshot: `outputs/b300_cu130_none_bias0.3_drilldown_2025-12-27_clean_perf.log`
- Stack-attributed self-attn (compile on; FA4 varlen; triton351 env): `outputs/b300_cu130_triton351_ops_profile_selfattn_compile_fa4_varlen_stack_topops.md`
- Older stack-attributed self-attn (compile on; ensure-contig run): `outputs/b300_cu130_triton351_ops_profile_selfattn_compile_ensurecontig_stack.md`
- Older self-attn stack filter (pre-triton351 / different config; shows contiguous/clone present): `outputs/b300_cu130_ops_profile_selfattn_stack.md`

---

## Next Steps

1. [ ] Run profiling command above
2. [ ] Fill in breakdown tables
3. [ ] Identify top 3 fusion candidates
4. [ ] Cross-reference with `layout-contracts.md`
5. [ ] Propose specific kernel scope

---

## Raw Profile Output

Paste raw output here for reference:

```
[Paste profile output after running]
```
