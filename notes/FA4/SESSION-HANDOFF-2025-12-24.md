# Session Handoff - 2025-12-24

**Last updated:** End of session 2 (notes reorg + architecture review)

## Session 2 Accomplishments

1. **Notes reorganization:**
   - `notes/FA4/` - organized by topic (b300/, rope/, fa4/, b200/)
   - `notes/research/` - organized by date (2025-12-23/)
   - `notes/realtime_video_architecture.md` - elevated to root as primary doc

2. **B300 investigation runbook:** `notes/FA4/b300/investigation-runbook.md`
   - Systematic tests for H1-H5 hypotheses
   - Specific commands, output locations, decision tree

3. **Architecture review:**
   - `notes/realtime_video_architecture.md` (v1.1, 1800+ lines)
   - `notes/research/2025-12-23/realtime-architecture-v1.1-review.md` (5Pro feedback)

## Current State

### B300 Mystery (UNSOLVED)
- **B200**: ~20 FPS at 320x576, 4 steps
- **B300**: 8.8 FPS at same settings (2x slower)
- Kernel optimizations show expected speedups in microbenchmarks but FPS doesn't change

### What Was Tried

**Codex LSE Fix (2025-12-24):**
- Hypothesis: Flash segment-combine was silently failing due to missing `return_lse` support
- Fix applied: `_get_fa4_fwd()` checks for `return_lse`, uses `_flash_attn_varlen_forward` as fallback
- **Result: Still 8.8 FPS** - hypothesis was wrong, mystery remains

**Other attempts (all still 8.8 FPS):**
- Enable FA4 for main attention
- `SCOPE_KV_CACHE_ATTENTION_BIAS=1.0`
- MAX_FRAME_RATE 8→30
- Switch FA4↔Triton backend
- LSE fallback fix

### Conclusion
The B300 8.8 FPS issue is **not** an attention backend problem. Something else in the pipeline is the bottleneck, but we don't know what.

## Git State

### Committed
- `fe7082a` - Add code churn context to B300 investigation
- `9e4c7b3` - Clean up B300 investigation notes - facts vs speculation
- `6559377` - Document B300 8.8 FPS mystery in kernel-dev-log

### Uncommitted (Codex changes)
```
M notes/FA4/phase4-fa4-score-mod.md
M scripts/test_fa4_kv_bias.py
M src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py
?? scripts/test_fa4_score_mod_smoke.py
?? vendored/flash_attn_cute_score_mod/
```

Key uncommitted changes:
1. **causal_model.py**: Vendored path priority, trip-breaker, LSE fallback fix
2. **vendored/flash_attn_cute_score_mod/**: 34 FA4/CUTE files from flash-attention.bak
3. **test_fa4_score_mod_smoke.py**: New smoke test for score_mod patterns

## Next Step: H100 Fallback

B300 mystery is unsolved. Pragmatic path forward is H100:
- B200 spot instances unavailable (GPU mode hackathon eating capacity)
- H100 is available and should work without SM103 complications
- No patches needed (H100 is SM90, not SM103)
- No CUDA 12.9 ptxas override needed

```bash
# On H100 - just run it
uv run daydream-scope
```

Expected: H100 should work at decent FPS with standard code paths.

## Hackathon Context
- Deadline: January 9th
- Goal: Real-time video generation for teammates to experience
- B200 was ideal ($0.95/hr, 20 FPS), now unavailable
- B300 ($1.24/hr) stuck at 8.8 FPS despite all attempts
- **H100 is the pragmatic fallback**

## Key Documentation (Read Order)

### 1. Quick Start
- **This file** (`SESSION-HANDOFF-2025-12-24.md`) - Current state and next steps

### 2. Understanding the Journey
- `notes/FA4/kernel-dev-log.md` - Full optimization journey, B200 wins, B300 mystery
  - Read "B300 (SM103) INVESTIGATION" section for the unsolved problem
  - Read "FINE-GRAINED PROFILING" for where time goes in the pipeline

### 3. B300-Specific (if revisiting)
- `notes/FA4/b300/fa4-patches.md` - SM103 patches for nvidia-cutlass-dsl
- Environment: `TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas`

### 4. Code
- `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py` - Attention backends
- `scripts/test_fa4_kv_bias.py` - FA4 test script
- `vendored/flash_attn_cute_score_mod/` - Vendored FA4 files (uncommitted)

## Next Steps (Priority Order)

1. **Spin up H100** - Verify it runs at decent FPS with `uv run daydream-scope`
2. **If H100 works** - Use it for hackathon, defer B300 investigation
3. **Document H100 baseline** - What FPS do we get?
4. **Get teammates demo-ready** - Hackathon deadline Jan 9

### If someone wants to continue B300 investigation:

**Primary resource:** `notes/FA4/b300/investigation-runbook.md` - systematic tests with specific commands.

**The key insight:** Attention backend changes don't move FPS. The bottleneck is elsewhere.

**Quick start:**
```bash
export GPU_TAG=h100  # or b200, b300
export OUT_DIR=/tmp/gpu-investigation/$GPU_TAG
mkdir -p $OUT_DIR

uv run python scripts/profile_krea_pipeline_blocks.py \
  --iters 20 --skip 3 --profile-blocks \
  --profile-blocks-json $OUT_DIR/block-profile.json
```

**Hypotheses (see runbook for full tests):**
| ID | Hypothesis | Quick check |
|----|------------|-------------|
| H1 | Pacing/backpressure limit | gpu_ms << wall_ms? |
| H2 | CPU bound | Low GPU util, high CPU? |
| H3 | Thermal/power throttling | High util, low clocks? |
| H4 | Shared GPU (MIG/vGPU) | Other processes visible? |
| H5 | Kernel fallback (cuBLAS SM103) | GEMMs slow? |

**Additional B300-specific concerns:**
- Memory hierarchy - L2 eviction patterns may differ from B200
- Driver/runtime overhead - CUDA 12.9 SM103 support is fresh

## Architecture v1.2 Action Items (from 5Pro review)

1. **Pipeline Adapter Contract** - map ControlState to KREA kwargs, handle continuity via `pipeline.state` keys (not direct attributes)
2. **Wire ControlBus to GeneratorDriver** - events as source of truth, drain at chunk boundaries
3. **Implement prompt ramps** - pick engine-native (EmbeddingBlendingBlock) or control-layer approach
4. **Clarify rollouts sequential** - single pipeline can't truly parallelize

## Next Priorities

1. **H100 fallback** - verify it works for hackathon (Jan 9 deadline)
2. **Architecture v1.2** - address 5Pro feedback
3. **B300 investigation** - run runbook if time permits

## Uncommitted (Intentional)

```
 M scripts/test_fa4_kv_bias.py
 M src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py
```
These are experimental Codex changes (LSE fallback) that didn't fix B300.

## Claude Code Version
See `notes/NOTES-INDEX.md` (“Environment Note”) for the Claude Code version/pinning caveat.
