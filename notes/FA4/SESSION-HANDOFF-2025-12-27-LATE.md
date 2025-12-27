# Session Handoff: 2025-12-27 Late Evening

## What We Just Did

### Documentation Deep Dives Created

**`notes/guides/krea-architecture/`** (6 complete files):
- `adaln-modulation.md` - AdaLN / timestep conditioning
- `kv-cache-mechanics.md` - Eviction, recomputation, bias
- `rope-embeddings.md` - 3D RoPE for video
- `attention-backends.md` - FA4/Flash/Triton/Flex
- `causal-attention-masks.md` - Block-wise causality
- `vae-streaming.md` - Causal 3D conv, streaming decode

**`notes/guides/v2v-mechanisms/`** (2 of 3 files done):
- `README.md` - Index (complete)
- `frame-processor-routing.md` - WebRTC → buffer → pipeline (complete)
- `latent-noise-mixing.md` - `noise_scale` semantics + schedule adjustment (draft)

### `noise_scale` Semantics (Drafted)

See `notes/guides/v2v-mechanisms/latent-noise-mixing.md` for the current writeup:
- Scope: explicit latent/noise mixing + a denoise schedule tweak (`denoising_step_list[0] = int(1000*noise_scale) - 100`)
- ComfyUI: typically schedule-aligned “strength” semantics (not 1:1)

---

## Context Discussed

- UI V2V “low FPS” can be **input-rate limited**: output pacing is `min(input_fps, pipeline_fps)` (see `FrameProcessor.get_output_fps()` in `src/scope/server/frame_processor.py`). Compute-only V2V baselines can be much faster (see `notes/FA4/b300/experiments.md` and the `outputs/b300_v2v_*` receipts).
- VACE-14B proposal (`notes/proposals/vace-14b-integration.md`) adds conditioning path, not latent-init
- Two V2V mechanisms: latent-init (Mechanism A) vs VACE (Mechanism B)
- FrameProcessor routing: `vace_enabled` determines which kwarg gets frames

---

## Pending Tasks

1. Test style swap: `STYLE_SWAP_MODE=1 ./scripts/run_daydream_b300.sh`
2. Measure style swap performance
3. Post cohort project page

---

## Uncommitted Files

```
notes/guides/krea-architecture/   (new folder)
notes/guides/v2v-mechanisms.md    (new overview)
notes/guides/v2v-mechanisms/      (new folder)
notes/guides/krea-architecture.md (untracked absolute symlink; don’t commit as-is)
notes/FA4/SESSION-HANDOFF-2025-12-27-LATE.md (this file)
```

---

## Resume Instructions

1. Finish `notes/guides/v2v-mechanisms/latent-noise-mixing.md`
2. Consider committing the guides
3. Or switch to style swap testing / B300 optimization
