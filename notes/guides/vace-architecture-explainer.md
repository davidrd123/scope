# VACE Architecture Explainer

> **Purpose:** Get someone up to speed on how VACE works in this codebase
> **Last Updated:** 2025-12-25
> **Status:** Current for 1.3B pipelines; Krea 14B integration pending

---

## Table of Contents

1. [What is VACE?](#what-is-vace)
2. [Architecture Overview](#architecture-overview)
3. [Code Map](#code-map)
4. [How Pipelines Integrate VACE](#how-pipelines-integrate-vace)
5. [The VACE Mixin Pattern](#the-vace-mixin-pattern)
6. [Block Graph and Data Flow](#block-graph-and-data-flow)
7. [What's Missing for Krea 14B](#whats-missing-for-krea-14b)
8. [Open Questions](#open-questions)

---

## What is VACE?

**VACE** (in this repo) = "Video All-In-One Creation and Editing"

It's a technique for conditioning video generation on:
- **Reference images** ("generate video that looks like this image") — style, character consistency
- **Structural guidance** (depth maps, pose, flow, scribbles) — per-frame control

Think of it as ControlNet for video, but designed for the Wan2.1 architecture.

### Key Insight

VACE works by creating a **parallel processing path** for conditioning information. Reference images/conditioning maps are encoded through "VACE blocks" that run alongside the main transformer blocks, producing "hints" that get injected into the main path.

```
Main Path:           [Block 0] → [Block 1] → [Block 2] → ... → [Block N] → Output
                         ↑           ↑           ↑
VACE Path:           [VACE 0] → [VACE 1] → [VACE 2] → ...
                         ↑
                    [Reference Images / Conditioning Maps]
```

---

## Architecture Overview

### Three Layers

1. **Model Layer** — `CausalVaceWanModel`
   - Wraps any CausalWanModel (LongLive, StreamDiffusion, Krea, etc.)
   - Creates parallel VACE blocks
   - Handles hint injection during forward pass

2. **Pipeline Layer** — `VACEEnabledPipeline` mixin
   - Handles VACE initialization at pipeline load time
   - Loads VACE weights from checkpoint
   - Exposes `vace_enabled` flag

3. **Block Layer** — `VaceEncodingBlock`
   - Runs per-chunk during generation
   - Encodes reference images / conditioning maps
   - Produces `vace_context` for the denoising block

### Weight Files

| Model Size | File | Size | Location |
|------------|------|------|----------|
| 1.3B | `Wan2_1-VACE_module_1_3B_bf16.safetensors` | ~2 GB | `Kijai/WanVideo_comfy` |
| 14B | `Wan2_1-VACE_module_14B_bf16.safetensors` | ~6.1 GB | `Kijai/WanVideo_comfy` |

The weight files contain:
- `vace_patch_embedding` — Conv3d that projects conditioning to model dimension
- `vace_blocks` — Parallel transformer blocks for processing conditioning

---

## Code Map

```
src/scope/core/pipelines/wan2_1/vace/
├── __init__.py                    # Exports: CausalVaceWanModel, load_vace_weights_only, VACEEnabledPipeline
├── mixin.py                       # VACEEnabledPipeline mixin class
│
├── models/
│   ├── causal_vace_model.py       # CausalVaceWanModel — the wrapper that adds VACE to any CausalWan
│   └── attention_blocks.py        # Factory functions that create VACE-aware block classes
│
├── blocks/
│   ├── __init__.py
│   └── vace_encoding.py           # VaceEncodingBlock — per-chunk encoding block
│
└── utils/
    ├── encoding.py                # vace_encode_frames, vace_encode_masks, vace_latent, load_and_prepare_reference_images
    └── weight_loader.py           # load_vace_weights_only
```

### Key Files to Understand

| File | What It Does | When to Read |
|------|--------------|--------------|
| `mixin.py` | `_init_vace()` method for pipeline init | Understanding how pipelines load VACE |
| `causal_vace_model.py` | Model wrapper, forward pass with hints | Understanding hint injection |
| `vace_encoding.py` | Per-chunk encoding block | Understanding runtime conditioning |
| `encoding.py` | VAE encoding utilities | Understanding how images become latents |

---

## How Pipelines Integrate VACE

### Current VACE-Enabled Pipelines

| Pipeline | Has VACE | Notes |
|----------|----------|-------|
| LongLive (1.3B) | Yes | Full implementation |
| StreamDiffusionV2 (1.3B) | Yes | Full implementation |
| RewardForcing (1.3B) | Yes | Full implementation |
| **Krea (14B)** | **No** | **Missing — needs integration** |

### Integration Pattern (LongLive Example)

**Step 1: Class Declaration** — Add mixin to inheritance
```python
# src/scope/core/pipelines/longlive/pipeline.py:36
class LongLivePipeline(Pipeline, LoRAEnabledPipeline, VACEEnabledPipeline):
```

**Step 2: Init VACE Before LoRA** — Critical ordering!
```python
# src/scope/core/pipelines/longlive/pipeline.py:87-89
generator.model = self._init_vace(
    config, generator.model, device=device, dtype=dtype
)
```

**Step 3: State Hygiene** — Clear stale refs
```python
# src/scope/core/pipelines/longlive/pipeline.py:221-222
if "vace_ref_images" not in kwargs:
    self.state.set("vace_ref_images", None)
```

**Step 4: Block Graph** — Add VaceEncodingBlock before denoise
```python
# src/scope/core/pipelines/longlive/modular_blocks.py:43
("vace_encoding", VaceEncodingBlock),
("denoise", DenoiseBlock),
```

**Step 5: Pipeline Manager** — Configure VACE path
```python
# src/scope/server/pipeline_manager.py:426-432
if vace_enabled:
    self._configure_vace(config, load_params)
```

---

## The VACE Mixin Pattern

`VACEEnabledPipeline` is a mixin class that provides `_init_vace()`:

```python
# Simplified from mixin.py

class VACEEnabledPipeline:
    vace_enabled: bool = False
    vace_path: str | None = None

    def _init_vace(self, config, model, device, dtype):
        vace_path = config.get("vace_path")
        if not vace_path:
            return model  # VACE disabled, return unchanged

        # 1. Wrap model with VACE
        vace_model = CausalVaceWanModel(model, vace_in_dim=96)

        # 2. Move VACE components to device
        vace_model.vace_patch_embedding.to(device, dtype)
        vace_model.vace_blocks.to(device, dtype)

        # 3. Load VACE weights
        load_vace_weights_only(vace_model, vace_path)

        self.vace_enabled = True
        return vace_model
```

### Critical Ordering

VACE must be initialized **BEFORE** LoRA because:
1. VACE wraps the model, creating new blocks (`vace_blocks`)
2. LoRA wraps the VACE-wrapped model
3. Final structure: `LoRA(VACE(BaseModel))`

If you do it backwards, LoRA adapters don't see the VACE blocks.

---

## Block Graph and Data Flow

### LongLive Block Order (with VACE)

```
1. text_conditioning        → Encode text prompt
2. embedding_blending       → Blend prompt embeddings
3. set_timesteps           → Set denoising schedule
4. auto_preprocess_video   → Preprocess input video (if V2V mode)
5. setup_caches            → Initialize KV caches
6. set_transformer_blocks_local_attn_size
7. auto_prepare_latents    → Prepare initial latents
8. recache_frames          → Recache if needed
9. vace_encoding           → **ENCODE VACE CONTEXT** ← This is where VACE happens
10. denoise                → Run denoising (hints injected here)
11. clean_kv_cache
12. decode                 → VAE decode
13. prepare_recache_frames
14. prepare_next           → Prepare for next chunk
```

### Krea Block Order (no VACE)

```
1. text_conditioning
2. embedding_blending
3. set_timesteps
4. auto_preprocess_video
5. setup_caches
6. auto_prepare_latents
7. recompute_kv_cache      → Krea-specific cache recompute
8. denoise
9. decode
10. prepare_context_frames
11. prepare_next
```

Note: Krea has `recompute_kv_cache` instead of `recache_frames`, and is missing `vace_encoding`.

---

## What's Missing for Krea 14B

### Checklist

| Component | Status | File | Change |
|-----------|--------|------|--------|
| 14B VACE artifact | Missing | `pipeline_artifacts.py` | Add `Wan2_1-VACE_module_14B_bf16.safetensors` |
| 14B checkpoint path | Hardcoded to 1.3B | `pipeline_manager.py:216-228` | Update `_get_vace_checkpoint_path()` |
| VACEEnabledPipeline mixin | Missing | `krea.../pipeline.py:42` | Add to class inheritance |
| `_init_vace()` call | Missing | `krea.../pipeline.py:~82` | Call before fusing/LoRA |
| Fusing order | Fuses before VACE | `krea.../pipeline.py:85-86` | Move fusing after `_init_vace()` |
| VACE blocks fusing | Not handled | `krea.../pipeline.py` | Fuse `model.vace_blocks` too |
| State hygiene | Missing | `krea.../pipeline.py:~228` | Clear `vace_ref_images` |
| VaceEncodingBlock | Not in block list | `krea.../modular_blocks.py` | Insert before `denoise` |
| `_configure_vace()` call | Missing | `pipeline_manager.py:458-528` | Add like other pipelines |
| `vace_enabled` param | Missing | `server/schema.py` | Add to `KreaRealtimeVideoLoadParams` |

### Critical Ordering Issue

Current Krea (wrong for VACE):
```python
# Line 85-86 — fuses BEFORE VACE would exist
for block in generator.model.blocks:
    block.self_attn.fuse_projections()

# Line 89 — LoRA init (VACE should be before this)
generator.model = self._init_loras(config, generator.model)
```

Correct order:
```python
# 1. Init VACE (creates vace_blocks)
generator.model = self._init_vace(config, generator.model, device, dtype)

# 2. Fuse ALL blocks (original + VACE)
for block in generator.model.blocks:
    block.self_attn.fuse_projections()
if hasattr(generator.model, 'vace_blocks'):
    for block in generator.model.vace_blocks:
        block.self_attn.fuse_projections()

# 3. Then LoRA
generator.model = self._init_loras(config, generator.model)
```

---

## Open Questions

### Architecture

1. **VACE block fusing** — Do VACE blocks have `self_attn.fuse_projections()`?
   - Need to check `attention_blocks.py` to confirm structure
   - If not, the fusing loop needs a guard

2. **14B weight shape validation** — Does `vace_patch_embedding.weight` have the expected shape `(5120, 96, 1, 2, 2)` for 14B?
   - 1.3B has dim=1536, 14B has dim=5120
   - Need to verify checkpoint before loading

3. **KV cache recompute path** — Does Krea's `recompute_kv_cache.py` need `vace_context` passed through?
   - LongLive doesn't have this block
   - May need to extend recompute to handle VACE

### Operational

4. **Memory impact** — VACE-14B adds ~6.1 GB
   - Recommend `vace_enabled=False` as default
   - Need to test memory on target hardware

5. **One-shot semantics** — Should VACE ref images persist across chunks?
   - LongLive clears `vace_ref_images` when not provided (one-shot)
   - Krea should do the same to avoid re-encoding stale refs

### API

6. **REST API surface** — How should VACE be exposed?
   - `vace_ref_images` already routed in `frame_processor.py:1085`
   - Need to verify Krea path is wired

---

## Quick Reference

### To Enable VACE on a Pipeline

1. Add `VACEEnabledPipeline` to class inheritance
2. Call `self._init_vace()` before fusing and LoRA
3. Fuse `vace_blocks` if present
4. Clear `vace_ref_images` in `_generate()` when not provided
5. Add `VaceEncodingBlock` to block graph before `denoise`
6. Add VACE artifact to `pipeline_artifacts.py`
7. Update `pipeline_manager.py` to call `_configure_vace()`

### To Debug VACE Issues

1. Check `vace_enabled` on pipeline: `pipeline.vace_enabled`
2. Check if VACE blocks exist: `hasattr(pipeline.components.generator.model, 'vace_blocks')`
3. Check vace_context is being passed: add logging in `VaceEncodingBlock.__call__`
4. Check hints are being injected: add logging in `CausalVaceWanModel._forward_inference`

---

## Related Documentation

- VACE paper: [ali-vilab/VACE](https://github.com/ali-vilab/VACE)
- Integration plan: `notes/vace-14b-integration/plan.md`
- Capability roadmap: `notes/capability-roadmap.md`
