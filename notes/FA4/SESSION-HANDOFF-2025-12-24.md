# Session Handoff - 2025-12-24

## Current State

### The Mystery
- **B200**: ~20 FPS at 320x576, 4 steps
- **B300**: 8.8 FPS at same settings (2x slower)
- Kernel optimizations show expected speedups in microbenchmarks but FPS doesn't change

### What Codex Just Found
Codex identified that the "flash segment-combine" path was **silently failing** on B300:
- Needs `return_lse=True` support
- Installed `_flash_attn_fwd` didn't support it
- Exception → trip-breaker → **silently fell back to Triton Kernel B**
- Triton Kernel B is slow on B300 (1.6ms vs 0.38ms FA4)

**Fix applied:** `_get_fa4_fwd()` now checks for `return_lse` support and uses `_flash_attn_varlen_forward` as fallback.

### Currently Testing
```bash
SCOPE_KV_BIAS_BACKEND=flash TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas uv run daydream-scope
```
If FPS improves from 8.8 → Codex found the culprit.

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

## Parallel Workstream: H100 Fallback

B200 spot instances unavailable (GPU mode hackathon eating capacity). H100 is available and should work:
- No SM103 patches needed (H100 is SM90)
- No CUDA 12.9 ptxas override needed
- Standard paths that worked before B300 work

```bash
# On H100 - just run it
uv run daydream-scope
```

## Hackathon Context
- Deadline: January 9th
- Goal: Real-time video generation for teammates to experience
- B200 was ideal ($0.95/hr, 20 FPS), now unavailable
- B300 ($1.24/hr) running at 8.8 FPS - testing Codex fix now
- H100 fallback if B300 fix doesn't work

## Key Files
- `notes/FA4/kernel-dev-log.md` - Full journey, B300 investigation section
- `notes/FA4/B300-FA4-PATCHES.md` - SM103 patch guide
- `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py` - Attention backends

## Next Steps (Priority Order)

1. **Check test result** - Did `SCOPE_KV_BIAS_BACKEND=flash` fix B300 FPS?
2. **If yes** - Commit Codex changes, B300 is usable
3. **If no** - Spin up H100, verify it works at decent FPS
4. **Either way** - Get teammates demo-ready for hackathon

## Claude Code Version
Locked at 2.0.62 (see notes/TODO-next-session.md for unlock instructions)
