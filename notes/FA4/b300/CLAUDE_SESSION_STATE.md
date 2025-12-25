# Claude Session State — B300 Investigation (2025-12-25)

> **Purpose:** Handoff document for context compaction. Resume from here.

---

## Current Status

**B300 @ 320×576:** ~14.8–15.0 FPS (up from 8.8 baseline = +70%)

**Branch:** `feature/stream-recording`
**Latest commit:** `60a544a` (pushed) - Add B300 optimization docs and incoming research materials

---

## What Was Accomplished This Session

### 1. Blog Analysis and Documentation
- Reviewed 16 blog posts in `notes/research/2025-12-24/incoming/perf/blogs/`
- Added Section 6 to `blackwell-docs.md`: Quick reference to all blogs with key insights
- Added Section 7 to `blackwell-docs.md`: Codebase correlation analysis

### 2. Key Findings from Codebase Correlation

| Category | Status | Notes |
|----------|--------|-------|
| Tensor core 128×128 | ✓ Aligned | head_dim=128 for both 14B and 1.3B |
| flex_attention max-autotune | ✓ Aligned | Uses max-autotune-no-cudagraphs |
| non_blocking transfers | ✓ Aligned | attention.py uses non_blocking=True |
| GPU sync elimination | ⚠️ Opportunity | `.item()` calls in KV cache hot path |
| Regional compilation | ❓ Not implemented | Could reduce cold start |

### 3. Research Organization
- Moved 3 ChatGPT exports to `incoming/perf/chat/`
- Added lifecycle framing to `ACTIONABLE_ITEMS_SUMMARY.md`:
  - `incoming/` = raw, speculative
  - `ACTIONABLE_ITEMS_SUMMARY.md` = testing TODO list
  - `blackwell-docs.md` = verified, graduated learnings
- Added status note to `NVFP4.md` (speculative, relevant for NVIDIA FP4 GEMM competition)

### 4. Current Bottleneck Analysis (cu130 stack)

```
Total: ~6076 ms (6 calls)
├── denoise: 3951 ms (65%) ← TRANSFORMER, needs deeper breakdown
│   └── generator: 3946 ms (99.9% of denoise)
│       └── call_model_kv_cache: 4635 ms
│           ├── Attention (self+cross)? → UNKNOWN
│           ├── Linear/MLP GEMMs?       → UNKNOWN
│           └── Norms, RoPE?            → UNKNOWN
├── decode: 1242 ms (20%) ← VAE conv3d, FIXED by cu130
├── recompute_kv_cache: 842 ms (14%)
└── text_conditioning: 35 ms (0.6%)
```

**Key insight:** After cu130 fixed decode (760ms→194ms), denoise/transformer is now the #1 bottleneck at 65%. The blogs focusing on attention optimization ARE relevant now, but we need a deeper profile to know if it's attention or linear GEMMs.

---

## Key Files to Know

| File | Purpose |
|------|---------|
| `notes/FA4/b300/blackwell-docs.md` | External docs + blog refs + codebase correlation (Sections 6-7 are new) |
| `notes/FA4/b300/session-state.md` | Current B300 status + repro commands |
| `notes/research/2025-12-24/incoming/perf/ACTIONABLE_ITEMS_SUMMARY.md` | Testing TODO list with lifecycle framing |
| `notes/research/2025-12-24/incoming/perf/blogs/` | 16 saved blog posts |
| `notes/research/2025-12-24/incoming/perf/chat/` | 3 ChatGPT exports (moved here) |
| `outputs/b300_cu130_fp8_bias03_drilldown_*.json` | Profiling artifacts |

---

## Pending Investigation

**Need to answer:** What's inside the 3951ms `generator()` call?
- Is it attention (self+cross)? → ThunderKittens, FlexAttention blogs apply
- Is it linear/MLP GEMMs? → NVFP4 applies
- Is it RoPE/norms? → Triton fusion applies

**How to get this:** Run with `PROFILE_ATTENTION=1` or add op-level timing to generator.

---

## Commits Pushed This Session

```
60a544a Add B300 optimization docs and incoming research materials
```

---

## Quick Resume Commands

```bash
# Check current status
cat notes/FA4/b300/session-state.md | head -50

# View blog reference
cat notes/FA4/b300/blackwell-docs.md | grep -A 100 "## 6) Blackwell Optimization"

# Run B300 benchmark
./scripts/run_daydream_b300.sh

# Profile with attention breakdown
PROFILE_ATTENTION=1 ./scripts/run_daydream_b300.sh
```

---

## Next Steps (When Resuming)

1. **Get transformer-internal breakdown** - Profile what's inside `generator()` (attention vs linear vs other)
2. **If attention dominates** - Evaluate ThunderKittens or cuDNN attention kernels
3. **If linear dominates** - Consider NVFP4 for GEMMs (NVIDIA competition running ~2 weeks)
4. **Test `.item()` elimination** - The GPU sync pattern identified in KV cache code
