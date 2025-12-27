# "other_in_self" Breakdown: Stack-Attributed Analysis

> Status: Stub (needs profiling run to fill in)
> Priority: **High** — can't optimize what we haven't measured
> Date: 2025-12-26
> Sources:
> - External references + constraints: `notes/FA4/DeepResearch/2025-12-26/B300_step_back/doc_ref_guide/claude01.md`
> - “Don’t guess, measure” decision rules: `notes/FA4/DeepResearch/2025-12-26/B300_step_back/round01/5pro_rp02.md`

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

### Step 1: Stack-attributed op profiling (to find the real copy sources)

```bash
DISABLE_FLEX_ATTENTION_COMPILE=1 \
WANVAE_STREAM_DECODE_MODE=chunk \
TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas \
SCOPE_KV_BIAS_BACKEND=fa4 \
.venv-b300-cu130-decode/bin/python scripts/profile_krea_pipeline_ops.py \
  --height 320 --width 576 \
  --iters 1 --pre-iters 1 \
  --kv-cache-attention-bias 0.3 \
  --kv-bias-backend fa4 \
  --quantization none \
  --cudnn-benchmark \
  --with-stack --stack-n 12 \
  --stack-key aten::contiguous \
  --stack-key aten::transpose \
  --stack-key aten::_to_copy \
  --stack-key aten::copy_ \
  --stack-key aten::fill_ \
  --summary outputs/b300_other_in_self_summary.md
```

Filter output for self-attn-related stacks (look for call stacks that include `CausalWanSelfAttention.forward`).

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

## Comparison Points

### Before Patch-Embed Fix

| Metric | Value |
|--------|-------|
| aten::copy_ calls | ~35,000 |
| aten::fill_ calls | ~35,000 |
| Total self_attn | ? ms |
| other_in_self | ? ms |

### After Patch-Embed Fix (Current)

| Metric | Value |
|--------|-------|
| aten::copy_ calls | ~9,600 |
| aten::fill_ calls | ~9,600 |
| Total self_attn | ? ms |
| other_in_self | ? ms |

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
