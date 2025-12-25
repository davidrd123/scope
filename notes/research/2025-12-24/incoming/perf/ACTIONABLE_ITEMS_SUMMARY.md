# Actionable Items Extracted from Perf Chat Logs

> **Generated:** 2025-12-24
> **Source files:** 3 ChatGPT conversation exports (see `chat/` subfolder)
> **Purpose:** Consolidated list of performance recommendations to verify against codebase

## Lifecycle: incoming → actionable → verified → docs

```
notes/research/.../incoming/perf/
├── chat/           # Raw captures (archival, speculative)
├── blogs/          # Saved blog posts (reference material)
└── ACTIONABLE_ITEMS_SUMMARY.md   # THIS FILE: Testing TODO list

notes/FA4/b300/
└── blackwell-docs.md   # Only VERIFIED learnings graduate here
```

**Status meanings in this file:**
- ✅ Done = Tested and verified, should be in blackwell-docs.md
- ⚠️ Partial = Partially tested, needs more work
- ❓ Need to verify = Not yet tested (speculative)
- ❌ Blocked = Cannot test due to external issue

**When to graduate items:** Once an item is ✅ verified, add the finding to `notes/FA4/b300/blackwell-docs.md` and optionally mark it here as "graduated".

---

## File 1: CuTe DSL Guide

**Type:** Reference material (not actionable items)

**Useful for:**
- score_mod function signature: `def score_mod(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors)`
- TensorSSA semantics and `cute.where`, `cute.full_like` patterns
- Examples: causal masking, sliding window, ALiBi, block-diagonal

**Key insight (verified in codebase):**
- `aux_tensors` forces `vec_size=1` (less vectorization) → prefer constant-closure approach
- This matches our implementation in `causal_model.py` which uses closure constants

---

## File 2: Improving B300 (SM103) Performance

### Verified Facts
- [x] B300 = SM103 (CC 10.3), B200 = SM100 (CC 10.0)
- [x] Triton Kernel B works on SM103 (JIT compiles)
- [x] `TRITON_PTXAS_PATH` fix needed for SM103

### Implementation Steps (from chat)

| Step | Description | Status |
|------|-------------|--------|
| Step 0 | Make `score_mod` importable via `flash-attention.bak` or vendored | ✅ Done (vendored path in causal_model.py) |
| Step 1 | Add FA4 score_mod entrypoint in attention.py | ✅ Done |
| Step 2 | Implement CuTe `score_mod` for KV bias with constant-closure | ✅ Done (see `_get_fa4_score_mod`) |
| Step 3 | Microbench + correctness harness | ⚠️ Partial (scripts exist) |
| Step 4 | Integrate behind feature flag | ✅ Done (`SCOPE_KV_BIAS_BACKEND`) |
| Step 5 | Nsight validation | ❌ Blocked (CUPTI errors) |

### Remaining Recommendations to Verify

1. **FA4 wheel compatibility on SM103**
   - Chat suggests checking if FA4 wheel has PTX coverage for 10.3
   - Current status: Using `flash-attn==2.8.3` + vendored CuTe sources

2. **cuDNN optimization for SM103**
   - Chat doesn't mention this, but our investigation found cu130 cuDNN is key
   - **This is the actual bottleneck** (not attention)

---

## File 3: Fix Chunk Padding (RoPE Regression)

### Diagnosis (verified in code)
- RoPE Step-2 regression caused by power-of-2 padding
- Real chunks: C0/C1/C2 = 22/21/21 pairs
- Padded chunks: 32/32/32 = +50% lanes → register pressure

### Implementation Steps

| Step | Description | Status |
|------|-------------|--------|
| Step 0 | Baseline measurement | ✅ Done |
| Step 1 | Replace kernel with v2 (C=64 fixed) | ❓ Need to verify |
| Step 2 | Update wrapper, delete C*_PAD path | ❓ Need to verify |

### Debug Commands (from chat)
```bash
# Force fused path, crash on fallback
SCOPE_TRITON_ROPE_FUSED_STRICT=1 \
SCOPE_DISABLE_TRITON_ROTARY=1 \
uv run python scripts/test_rope_correctness.py --pad-to-multiple 128

# Benchmark fused vs wrapper
SCOPE_TRITON_ROPE_FUSED_STRICT=1 \
uv run python scripts/bench_triton_rope.py --pad-to-multiple 128

# Debug print shapes
SCOPE_TRITON_ROPE_FUSED_DEBUG=1 \
SCOPE_TRITON_ROPE_FUSED_STRICT=1 \
uv run python scripts/bench_triton_rope.py --pad-to-multiple 128
```

---

## Cross-Check Against Current Codebase

### What the chats got RIGHT
1. SM103/SM100 compatibility analysis
2. FA4 score_mod constant-closure approach
3. `TRITON_PTXAS_PATH` fix for SM103
4. Backend selection hierarchy (FA4 → Triton Kernel B → flex_attention)

### What the chats MISSED (discovered through actual profiling)
1. **cuDNN/conv3d is the real B300 bottleneck** (not attention)
2. **cu130 stack provides 4x decode speedup** (760ms → 194ms)
3. **VAE streaming decode mode** (`WANVAE_STREAM_DECODE_MODE=chunk`)
4. **DISABLE_FLEX_ATTENTION_COMPILE=1** needed for torch 2.9/SM103

### Priority Order (Based on Actual Impact)

1. **cu130 runtime stack** — 70% FPS improvement (8.8 → 15 FPS)
2. **FlashAttention installation in cu130 env** — prevents 1 FPS fallback
3. **VAE chunk decode mode** — minor improvement
4. **cuDNN benchmark** — minor improvement
5. **FA4 score_mod tuning** — marginal (attention is not the bottleneck)

---

## Recommendations for Future LLM Chats

When getting perf recommendations from LLMs:
1. Always profile first to identify actual bottlenecks
2. LLMs tend to focus on attention optimization (flashy) but miss runtime/cuDNN issues
3. Provide block-level timing breakdown as context
4. Verify SM-specific recommendations on actual hardware
