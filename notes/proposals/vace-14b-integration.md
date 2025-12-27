# VACE-14B Integration Plan

**Created**: 2024-12-24
**Last Updated**: 2025-12-27
**Status**: Ready to implement
**Goal**: Add VACE (reference image conditioning + V2V conditioning) support to the Krea 14B pipeline
**Review**: `notes/proposals/vace-14b-integration/reviews/oai_5pro01.md`

---

## Executive Summary

VACE-14B weights exist upstream and are downloadable. This is **engineering + wiring work**, not blocked on missing weights. The main tasks are:
1. Add artifact for VACE-14B module download
2. Teach `pipeline_manager.py` to select the **14B** VACE module for Krea (it’s hardcoded to 1.3B today)
3. Add `vace_enabled` to `KreaRealtimeVideoLoadParams` (default should likely be **False** for VRAM)
4. Wire `VACEEnabledPipeline` mixin into Krea (VACE before projection fusing + LoRA, and avoid CPU/GPU device-split hazards)
5. Add `VaceEncodingBlock` to Krea modular blocks + clear `vace_*` inputs in state to avoid stale conditioning
6. Decide (and document) KV-cache recompute + quantization policy for VACE-14B (and whether to compile `vace_blocks` when `compile=True`)

---

## 1. Background

### 1.1 Current State

| Pipeline | Base Model | VACE Support | Notes |
|----------|------------|--------------|-------|
| **Krea Realtime** | Wan2.1-T2V-14B | **No** | CausVid distillation; supports text+video modes; VACE missing |
| **LongLive** | Wan2.1-T2V-1.3B | Yes | Has VACE wired |
| **StreamDiffusionV2** | Wan2.1-T2V-1.3B | Yes | Has VACE wired |
| **Reward Forcing** | Wan2.1-T2V-1.3B | Yes | Has VACE wired |

Krea is the **only 14B pipeline** in the codebase, and it's the only one without VACE.

### 1.2 Architecture Comparison (Verified)

| Model | dim | num_layers | num_heads | head_dim | ffn_dim |
|------:|----:|-----------:|----------:|---------:|--------:|
| Wan2.1-T2V-1.3B | 1536 | 30 | 12 | 128 | 8960 |
| Wan2.1-T2V-14B  | 5120 | 40 | 40 | 128 | 13824 |

**Key implications:**
- VACE weights are **not portable** between 1.3B and 14B (different `dim`, `num_layers`)
- VACE injects at `range(0, num_layers, 2)` = **every other layer starting from 0**:
  - **1.3B (30 layers)**: 15 VACE blocks at [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28]
  - **14B (40 layers)**: 20 VACE blocks at [0,2,4,6,...,36,38]
- `head_dim=128` is identical, so attention kernel behavior should be similar

**Source**: `causal_vace_model.py:79-81`

### 1.3 Upstream VACE-14B Availability (Verified)

| Source | File | Size | Notes |
|--------|------|------|-------|
| `Kijai/WanVideo_comfy` | `Wan2_1-VACE_module_14B_bf16.safetensors` | **6.1 GB** | Module-only, verified exists |
| `Wan-AI/Wan2.1-VACE-14B` | Full checkpoint (7 shards) | ~75 GB | Contains base weights too |

**Recommendation**: Use the module-only file from Kijai (6.1 GB vs 75 GB).

**Verified**: File exists, uploaded 7 months ago, SHA256: `66a4bd41ec0fc58f1ff6d1313e06cd9a4c24ab60171a5846937536f8d4de6a65`

---

## 1.4 Repo Semantics You Must Internalize (Krea-specific)

These are the “gotchas” that will bite you if you treat VACE as “just another model flag”.

### VACE is a load-time toggle in this repo

- `VACEEnabledPipeline._init_vace()` enables VACE when (and only when) `config.vace_path` is set.
- The server routes incoming frames based on `pipeline.vace_enabled` (not on request parameters).
- Practically: enabling/disabling VACE requires **reloading** the pipeline.

Key server code path: `src/scope/server/frame_processor.py` routes:
- `video_input → video` when VACE is disabled
- `video_input → vace_input_frames` when VACE is enabled

### VACE-enabled “video mode” is not the same as normal V2V

- `AutoPrepareLatentsBlock` switches to V2V latents only when the input is named `video`.
- When VACE is enabled, the server routes frames to `vace_input_frames`, so Krea stays on the **text/T2V latent path** and uses video frames only as conditioning.
- The server explicitly does **not** support combining normal V2V and VACE V2V editing simultaneously.

## 2. Implementation Steps

### Step 1: Add VACE-14B Artifact

**File**: `src/scope/server/pipeline_artifacts.py`

```python
# Add new artifact constant
VACE_14B_ARTIFACT = HuggingfaceRepoArtifact(
    repo_id="Kijai/WanVideo_comfy",
    files=["Wan2_1-VACE_module_14B_bf16.safetensors"],
)

# Add to krea-realtime-video artifacts list
"krea-realtime-video": [
    WAN_1_3B_ARTIFACT,
    UMT5_ENCODER_ARTIFACT,
    VACE_14B_ARTIFACT,  # <-- Add this
    HuggingfaceRepoArtifact(...),  # existing 14B config
    HuggingfaceRepoArtifact(...),  # existing Krea checkpoint
],
```

Note: artifact downloads are currently **static per pipeline** (not conditional on load params). Adding the 6.1GB module to Krea’s artifacts means it will download as part of Krea setup even when `vace_enabled=False`. If conditional downloads matter, that’s an additional feature (not covered here).

### Step 2: Update the Server Load API (vace_enabled for Krea)

**File**: `src/scope/server/schema.py`

Add `vace_enabled: bool` to `KreaRealtimeVideoLoadParams`.

**Recommendation:** default `False` for Krea, because VACE-14B adds major VRAM + runtime overhead.

Why this is required:
- `pipeline_manager.py` needs a load-time signal to decide whether to set `config["vace_path"]`.
- `_init_vace()` reads `config.vace_path` at pipeline init and sets `pipeline.vace_enabled`.
- The server routes frames based on `pipeline.vace_enabled`.

### Step 3: Update pipeline_manager.py (select 14B VACE module + wire Krea)

**File**: `src/scope/server/pipeline_manager.py`

Two required changes:

1) **Select the correct VACE checkpoint for the model size**
- `_get_vace_checkpoint_path()` is hardcoded to the 1.3B module today.
- Krea must use the 14B module: `WanVideo_comfy/Wan2_1-VACE_module_14B_bf16.safetensors`.
- Do **not** simply change the existing `_get_vace_checkpoint_path()` to 14B; that would break 1.3B pipelines. Instead, parameterize selection (e.g. per `pipeline_id` or per base model size).

2) **Actually call `_configure_vace()` for Krea**, guarded by `load_params["vace_enabled"]`
- LongLive/StreamDiffusion/RewardForcing already do this; Krea does not.

Note: `_configure_vace()` currently also looks for `ref_images` and `vace_context_scale` in `load_params`. Runtime “Parameters” use `vace_ref_images` / `vace_context_scale`, so treat load-param `ref_images` as optional follow-up (add to schema + define semantics, or remove from `_configure_vace()`).

### Step 4: Add VACEEnabledPipeline mixin to Krea

**File**: `src/scope/core/pipelines/krea_realtime_video/pipeline.py`

```python
from ..wan2_1.vace.mixin import VACEEnabledPipeline

class KreaRealtimeVideoPipeline(Pipeline, LoRAEnabledPipeline, VACEEnabledPipeline):
    ...
```

### Step 5: Wire VACE in `__init__` (ordering + fusing + LoRA)

**File**: `src/scope/core/pipelines/krea_realtime_video/pipeline.py`

**Current code (roughly):**
```python
generator = WanDiffusionWrapper(...)
for block in generator.model.blocks:
    block.self_attn.fuse_projections()
generator.model = self._init_loras(...)
... quantize / move to device ...
```

**Required ordering for VACE:**
```python
generator = WanDiffusionWrapper(...)

# VACE must be applied before fusing and before LoRA.
#
# IMPORTANT: `_init_vace()` moves ONLY VACE modules to the passed (device, dtype).
# In Krea today the base model is still on CPU at this point, so passing the
# *target* CUDA device here can split the model across CPU/GPU.
#
# Pass the *current* base-model device/dtype (likely CPU) to keep everything aligned.
base_device = next(generator.model.parameters()).device
base_dtype = next(generator.model.parameters()).dtype
generator.model = self._init_vace(config, generator.model, base_device, base_dtype)

# Fuse projections after VACE wraps/replaces blocks.
for block in generator.model.blocks:
    block.self_attn.fuse_projections()
if hasattr(generator.model, "vace_blocks"):
    for block in generator.model.vace_blocks:
        block.self_attn.fuse_projections()

# LoRA must be outermost wrapper.
generator.model = self._init_loras(config, generator.model)
```

Quantization note (Krea defaults to FP8):
- `_init_vace()` moves VACE modules to `device`/`dtype` before loading weights.
- On memory-constrained GPUs, loading VACE bf16 weights on GPU *before* FP8 quantization may OOM.
- If you hit this, keep VACE on CPU during init (as above) so the later `generator.to(...)` / FP8 `quantize_(..., device=...)` moves the full wrapped model in one pass.

Also required: add LongLive-style state hygiene so `vace_ref_images` and per-chunk conditioning (`vace_input_frames`, `vace_input_masks`) don’t persist across chunks when not provided.

Optional perf parity: when `compile=True` and using per-block compilation, also compile `generator.model.vace_blocks` (not just `generator.model.blocks`) to avoid a VACE-only perf cliff.

### Step 6 (Optional, for parity): Add Config Schema Fields

**File**: `src/scope/core/pipelines/schema.py`

LongLive/StreamDiffusion/RewardForcing include `ref_images` + `vace_context_scale` on their config classes; Krea does not. Adding these improves parity and schema metadata, but is not strictly required for “VACE-14B loads and runs”.

Important: runtime VACE uses `vace_ref_images` (pipeline inputs), not `ref_images` (config). Decide whether you want to map `ref_images → vace_ref_images` as a convenience.

### Step 7: Add VaceEncodingBlock to Krea modular blocks

**File**: `src/scope/core/pipelines/krea_realtime_video/modular_blocks.py`

Insert `VaceEncodingBlock` so that `vace_context` exists before `denoise`.

Recommended placement for flexibility: after `auto_prepare_latents` and before `recompute_kv_cache` / `denoise`.

### Step 8 (Decision): KV-cache recompute semantics for VACE

**File**: `src/scope/core/pipelines/krea_realtime_video/blocks/recompute_kv_cache.py`

Krea’s recompute block currently calls the generator without `vace_context` / `vace_context_scale`. Decide whether that is correct for your use-cases:

- **Option A (simplest): keep it as-is.** This matches other cache recache paths and avoids extra complexity.
- **Option B (scoped): pass `vace_context` during recompute for reference-only mode.** This is coherent when conditioning comes from `vace_ref_images` (stable across chunks). For VACE V2V editing (`vace_input_frames` per chunk), “passing vace_context during recompute” is underspecified unless you also buffer conditioning aligned to the context-frame buffer.

This decision can affect both quality and performance; document whichever choice you make.

## 3. Known Pitfalls

### 3.1 VAE Path Mismatch

This used to be a pitfall, but **Krea already sets `vae_path` explicitly** in `src/scope/server/pipeline_manager.py` (points at the 1.3B VAE artifact).

Don’t remove that path unless you also update artifacts and verify `WanVAEWrapper`’s default lookup behavior.

### 3.2 Device/dtype split hazard during `_init_vace()` in Krea

`VACEEnabledPipeline._init_vace()` moves only the VACE-specific modules (`vace_patch_embedding`, `vace_blocks`) to the provided `(device, dtype)`. In Krea’s current init flow, the base model is still on CPU when you’re doing ordering-sensitive operations (wrap → fuse projections → init LoRAs), so passing the *target* CUDA device into `_init_vace()` can split the model across CPU/GPU and crash at runtime.

**Mitigation:** call `_init_vace()` using the current base-model device/dtype (typically CPU), then let the existing `generator.to(...)` / FP8 `quantize_(..., device=...)` move the full wrapped model onto the target device.

### 3.3 Projection Fusing After VACE

Krea uses projection fusing for attention (`block.self_attn.fuse_projections()`).

VACE wrapping replaces attention blocks, so fusing must happen **after** `_init_vace()` and must cover both:
- `generator.model.blocks`
- `generator.model.vace_blocks` (when present)

### 3.4 VACE conditioning inputs can go stale without state clearing

Even though the server clears one-shot parameters from its mailbox, the pipeline `PipelineState` persists keys until explicitly cleared. For Krea, ensure `_generate()` clears these when absent from kwargs:
- `vace_ref_images` (reference images)
- `vace_input_frames` and `vace_input_masks` (per-chunk conditioning video)

Otherwise a chunk that arrives without new VACE inputs can silently reuse old conditioning.

### 3.5 Server Routing Changes Semantics

When VACE is enabled, server “video mode” input frames are routed to `vace_input_frames` (conditioning), not `video` (V2V latents). This is intentional but easy to miss.

### 3.6 Quantization Policy (FP8 by default on Krea)

Krea defaults to FP8 quantization at load time. VACE-14B adds large bf16 weights; decide whether you want:
- VACE blocks to also be quantized (memory/stability tradeoffs), or
- VACE blocks to remain bf16 while the base model is quantized (mixed-dtype + VRAM tradeoffs).

### 3.7 Distilled Checkpoint Compatibility

Open question: Does the Krea distilled checkpoint (`krea-realtime-video-14b.safetensors`) preserve the same block structure that our VACE wrapper expects?

**Risk**: Distillation can sometimes change internal structure.

**Mitigation**: Run shape validation in `load_vace_weights_only()` - it will error if shapes don't match.

---

## 4. Validation Checklist

Before declaring VACE-14B working:

- [ ] VACE-14B artifact downloads successfully
- [ ] `pipeline_manager.py` selects the 14B VACE module for Krea (not the 1.3B module)
- [ ] Loading Krea with VACE enabled sets `pipeline.vace_enabled == True`
- [ ] In server “video mode”, frames route to `vace_input_frames` when VACE is enabled
- [ ] VACE module keys match our filter (`vace_blocks.*`, `vace_patch_embedding.*`)
- [ ] Number of VACE blocks = 20 (for 40 layers, every other layer [0,2,4,...,38])
- [ ] Shape check: `vace_patch_embedding.weight` is `(5120, 96, 1, 2, 2)`
- [ ] Projection fusing still works after VACE block replacement
- [ ] VAE loads without path errors
- [ ] Minimal R2V generation produces reasonable output
- [ ] `vace_ref_images` is “one-shot” (doesn’t silently persist across chunks)
- [ ] `vace_input_frames` / `vace_input_masks` don’t silently persist across chunks when not provided
- [ ] Memory footprint acceptable (14B + VACE + FP8)

---

## 5. Files to Modify

| File | Change |
|------|--------|
| `src/scope/server/pipeline_artifacts.py` | Add VACE-14B artifact |
| `src/scope/server/pipeline_manager.py` | Select 14B module for Krea (without breaking 1.3B) + call `_configure_vace()` for Krea when enabled |
| `src/scope/server/schema.py` | Add `vace_enabled` to `KreaRealtimeVideoLoadParams` (recommend default `False`) |
| `src/scope/core/pipelines/krea_realtime_video/pipeline.py` | Add mixin, call `_init_vace()` before fuse/LoRA (avoid device split), fuse `vace_blocks`, clear `vace_ref_images`/`vace_input_frames`/`vace_input_masks`, optionally compile `vace_blocks` |
| `src/scope/core/pipelines/krea_realtime_video/modular_blocks.py` | Insert `VaceEncodingBlock` |
| `src/scope/core/pipelines/schema.py` | (Optional) Add `ref_images` / `vace_context_scale` to `KreaRealtimeVideoConfig` |
| `src/scope/core/pipelines/krea_realtime_video/blocks/recompute_kv_cache.py` | (Decision) If recompute should pass VACE context |

---

## 6. Verification Commands

### Check VACE module shapes locally
```bash
python - <<'PY'
from safetensors.torch import load_file

path = "~/.daydream-scope/models/WanVideo_comfy/Wan2_1-VACE_module_14B_bf16.safetensors"
sd = load_file(path, device="cpu")
print("num_keys:", len(sd))

w = sd.get("vace_patch_embedding.weight")
print("vace_patch_embedding.weight:", None if w is None else (tuple(w.shape), w.dtype))

# Expect: (5120, 96, 1, 2, 2) for 14B
PY
```

### Verify VAE location
```bash
ls -lh ~/.daydream-scope/models/Wan2.1-T2V-*/Wan2.1_VAE.pth
```

### Check 14B config
```bash
cat ~/.daydream-scope/models/Wan2.1-T2V-14B/config.json | jq '{dim, num_heads, num_layers, ffn_dim}'
```

---

## 7. Reference: How VACE Loading Works (Verified from Source)

From `src/scope/core/pipelines/wan2_1/vace/`:

### `mixin.py` - VACEEnabledPipeline
1. `_init_vace(config, model, device, dtype)` wraps model with `CausalVaceWanModel`
2. Moves VACE components to correct device/dtype
3. Calls `load_vace_weights_only()` to load weights

### `models/causal_vace_model.py` - CausalVaceWanModel
1. VACE layers: `range(0, num_layers, 2)` = **every other layer starting from 0**
2. Creates `vace_patch_embedding` Conv3D: `(vace_in_dim, dim, patch_size, patch_size)`
3. Creates `vace_blocks` ModuleList with VACE attention blocks
4. 1.3B: 15 VACE blocks, 14B: 20 VACE blocks

### `utils/weight_loader.py` - load_vace_weights_only
1. Auto-detects checkpoint type (module-only vs full)
2. Filters for keys: `vace_blocks.*`, `vace_patch_embedding.*`
3. Validates `vace_patch_embedding.weight` shape matches model
4. Errors if shape mismatch or weights all zeros

The mixin is already implemented and tested with 1.3B pipelines. For 14B, the differences are:
- Larger `dim` (5120 vs 1536) affects weight shapes
- More VACE blocks (20 vs 15) due to deeper model

---

## 8. Next Actions

1. Implement **Step 1** (artifact) and verify the 14B VACE module downloads.
2. Implement **Step 2–3** (server schema + pipeline manager) so Krea can be loaded with VACE enabled (and default stays off).
3. Implement **Step 4–5–7** (Krea pipeline + block graph) and verify `pipeline.vace_enabled` + basic R2V.
4. Decide **Step 8** (KV recompute semantics) and the quantization policy; document the choice.
5. Validate on target hardware: VRAM headroom + perf impact + qualitative output.

---

## Supporting Materials

| File | Contents |
|------|----------|
| [`vace-14b-integration/work-log.md`](vace-14b-integration/work-log.md) | Chronological context, prior assumptions |
| [`vace-14b-integration/research/`](vace-14b-integration/research/) | Architecture research, upstream availability |
| [`../guides/vace-architecture-explainer.md`](../guides/vace-architecture-explainer.md) | Repo-wide VACE overview (how it works today) |
