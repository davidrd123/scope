# B300 Session State - 2025-12-23 (Updated)

## Current Status

**B300 is at 8.8 FPS - suspiciously constant regardless of changes made**

## Key Discovery This Session

The `flash-attention` symlink was shadowing the working FA4 package. Removed it, FA4 now imports correctly:
- `FLASH_ATTN_4_AVAILABLE: True`
- FA4 benchmarks show 0.38ms for KV-cache attention (vs 1.6ms Triton)

**BUT FPS stays at exactly 8.8 regardless of:**
- Enabling FA4 for main attention
- Setting `SCOPE_KV_CACHE_ATTENTION_BIAS=1.0` (bypasses Triton Kernel B)
- Changing codec MAX_FRAME_RATE from 8 to 30

This suggests a **hard limiter we haven't found yet**.

## Two Codex Agents Working

### Codex1: FA4 score_mod fix (preferred approach)
- Patching `flash-attention.bak` to replace `FastDivmodDivisor` with custom `FastDivmod`
- Files modified: `fast_math.py`, `tile_scheduler.py`, `paged_kv.py`, `flash_fwd.py`, `flash_fwd_sm100.py`, `flash_fwd_combine.py`, `flash_bwd_sm100.py`
- **Status: Still working**

### Codex2: Segment-combine fallback
- Added `_kv_bias_flash_combine()` - splits KV into 3 segments, runs FA4 on each, merges with LSE
- Auto-detects SM103 → defaults to "flash" backend
- **Status: Complete, merged into causal_model.py**

## Current Code State

| Component | Status |
|-----------|--------|
| `flash-attention` symlink | Missing (removed) |
| `flash-attention.bak/` | Has Codex1's FastDivmod patches (incomplete) |
| `causal_model.py` | Has Codex2's flash-combine + Codex1's path extension |
| `webrtc.py` | MAX_FRAME_RATE changed 8→30 |
| `pipeline.py` | Added SCOPE_KV_CACHE_ATTENTION_BIAS env var |

## Backend Selection on B300

```python
_KV_BIAS_BACKEND = "flash" if _is_sm103() else "triton"
```

- `fa4`: FA4 with score_mod (needs Codex1 to finish + symlink)
- `flash`: Segment-combine (Codex2, currently active on B300)
- `triton`: Triton Kernel B (slow on B300: 1.6ms)
- `flex`: flex_attention fallback

## To Test When Codex1 Finishes

```bash
# Recreate symlink
ln -s flash-attention.bak flash-attention

# Run with FA4 score_mod
SCOPE_KV_BIAS_BACKEND=fa4 TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas uv run daydream-scope
```

## To Test Current State (Codex2's approach)

```bash
TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas uv run daydream-scope
```

## Mystery: Why Always 8.8 FPS?

Investigated but not found:
- Codec cap (was 8, changed to 30 - no effect)
- KV-cache bias path (bypassed with bias=1.0 - no effect)
- Attention backend (FA4 vs Triton - no effect on FPS)

Possible unexplored causes:
- FPS calculation/reporting bug
- Some synchronization in frame_processor.py
- Input frame rate limiting
- Something in the actual pipeline execution path

## Files to Investigate

- `/root/scope/src/scope/server/frame_processor.py` - FPS calculation at line 296
- `/root/scope/src/scope/server/tracks.py` - Frame rate control
- The actual pipeline call path that generates frames

## Environment

```bash
export TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas
# Optional overrides:
export SCOPE_KV_BIAS_BACKEND=flash  # or fa4, triton
export SCOPE_KV_CACHE_ATTENTION_BIAS=1.0  # disable bias for plain FA4
export PROFILE_ATTENTION=1  # enable profiling
```
