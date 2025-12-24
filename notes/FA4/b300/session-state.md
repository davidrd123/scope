# B300 Session State - 2025-12-24 (Updated)

## Current Status

**B300 is ~8.8 FPS at `320x576` (reference resolution) and the number is stable across iterations.**

**Key update:** This no longer looks like an “invisible FPS cap” caused by WebRTC/codec pacing or CPU stalls. A block-level CUDA-event profile shows GPU time ≈ wall time, and the dominant blocks are `denoise` and `decode` (not the KV-bias attention microkernel).

See also: `notes/FA4/b300/investigation-runbook.md` (“Ground Truth” section).

## New Evidence (Zoom-Out Block Profile)

Artifact: `outputs/b300_pipeline_blocks_profile.json` (generated via `PROFILE_PIPELINE_BLOCKS=1`).

Per-block GPU time breakdown (4 pipeline calls):

| Block | GPU ms | Share (of total GPU) | Per-call |
|------:|-------:|----------------------:|---------:|
| `denoise` | ~4208 | ~46% | ~1052 ms |
| `decode` | ~3010 | ~33% | ~752 ms |
| `recompute_kv_cache` | ~1218 | ~13% | ~305 ms |
| `text_conditioning` | ~655 | ~7% | ~164 ms |

**Interpretation:** End-to-end throughput is largely compute-bound inside `denoise` (+ `decode`), so swapping KV-bias backends (Triton Kernel B vs FA4 vs segment-combine) won’t materially move FPS unless it changes time inside `denoise` (or reduces work in `decode` / `recompute_kv_cache`).

## Key Discovery This Session

The `flash-attention` symlink was shadowing the working FA4 package. Removed it, FA4 now imports correctly:
- `FLASH_ATTN_4_AVAILABLE: True`
- FA4 benchmarks show 0.38ms for KV-cache attention (vs 1.6ms Triton)

**BUT FPS stays at exactly 8.8 regardless of:**
- Enabling FA4 for main attention
- Setting `SCOPE_KV_CACHE_ATTENTION_BIAS=1.0` (bypasses Triton Kernel B)
- Changing codec MAX_FRAME_RATE from 8 to 30

This originally suggested a “hard limiter”, but the block profile above points to a more mundane explanation: **the slow path is dominated by non-attention work** (`denoise` / `decode` / `recompute_kv_cache`).

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

## Repo/Packaging Note (Reproducibility)

- `flash-attention.bak/` is gitignored and not available to RepoPrompt / other machines by default.
- If/when we want the FA4 score_mod path reproducible in-repo, prefer vendoring the minimal CuTe sources (already present at `vendored/flash_attn_cute_score_mod/`) rather than relying on `flash-attention.bak/`.

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

## Next Questions (Now That It Looks Compute-Bound)

Priority is to drill into the *dominant* blocks:

- Within `denoise`: how much is attention vs other ops (conv/linear/norm/rope/etc.)?
- Within `decode`: is VAE decode unusually slow on SM103 (cuDNN algo selection / kernel fallback / precision path)?
- Can we reduce `recompute_kv_cache` work without hurting quality (knobs like cache lengths / frames)?

Practical note: `torch.profiler` CUDA timelines appear broken on this environment (CUPTI errors). Prefer CUDA-event based probes (`PROFILE_PIPELINE_BLOCKS`, `PROFILE_ATTENTION`) or external `nsys` if available.

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
