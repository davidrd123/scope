# VACE-14B Integration Plan

**Created**: 2024-12-24
**Status**: Ready to implement
**Goal**: Add VACE (reference image conditioning) support to Krea 14B pipeline

---

## Executive Summary

VACE-14B weights exist upstream and are downloadable. This is **engineering + wiring work**, not blocked on missing weights. The main tasks are:
1. Add artifact for VACE-14B module download
2. Wire `VACEEnabledPipeline` mixin into Krea pipeline
3. Add `VaceEncodingBlock` to modular blocks
4. Handle VAE path resolution (currently downloads to 1.3B folder only)

---

## 1. Background

### 1.1 Current State

| Pipeline | Base Model | VACE Support | Notes |
|----------|------------|--------------|-------|
| **Krea Realtime** | Wan2.1-T2V-14B | **No** | CausVid distillation, T2V only |
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

### Step 2: Add VACEEnabledPipeline Mixin to Krea

**File**: `src/scope/core/pipelines/krea_realtime_video/pipeline.py`

```python
from ..wan2_1.vace.mixin import VACEEnabledPipeline

class KreaRealtimeVideoPipeline(Pipeline, LoRAEnabledPipeline, VACEEnabledPipeline):
    ...
```

### Step 3: Wire VACE in __init__

**File**: `src/scope/core/pipelines/krea_realtime_video/pipeline.py`

**Current code (lines 75-91):**
```python
generator = WanDiffusionWrapper(...)          # Line 75-82
for block in generator.model.blocks:          # Line 85-86
    block.self_attn.fuse_projections()
generator.model = self._init_loras(...)       # Line 89
if quantization == Quantization.FP8_E4M3FN:   # Line 91+
```

**Modified order (VACE before fusing, fusing handles both block types):**
```python
# 1. Create generator (unchanged)
generator = WanDiffusionWrapper(...)

# 2. VACE wrapping BEFORE fusing (creates new blocks that inherit fuse_projections)
generator.model = self._init_vace(config, generator.model, device, dtype)

# 3. Fuse projections on ALL blocks (regular AND VACE)
for block in generator.model.blocks:
    block.self_attn.fuse_projections()
if hasattr(generator.model, 'vace_blocks') and generator.model.vace_blocks:
    for block in generator.model.vace_blocks:
        block.self_attn.fuse_projections()

# 4. LoRA init (unchanged)
generator.model = self._init_loras(config, generator.model)

# 5. Quantization (unchanged)
if quantization == Quantization.FP8_E4M3FN:
    ...
```

**Why this order:**
- VACE creates new blocks via factory pattern (inherits from Krea's block class)
- New blocks start with separate Q, K, V (unfused)
- Fusing must happen AFTER VACE wrapping to fuse the new blocks
- LoRA must be last (wraps outermost)

**Verified from:**
- `mixin.py:58-64`: `_init_vace(config, model, device, dtype)` signature
- `attention_blocks.py:24,125`: Factory inherits from pipeline's block class
- `longlive/pipeline.py:87-89`: Shows VACE before LoRA pattern

### Step 3b: Add Config Schema Fields

**File**: `src/scope/core/pipelines/schema.py` (lines 322-369)

Add VACE fields to `KreaRealtimeVideoConfig` (matching LongLive pattern from lines 221-231):

```python
class KreaRealtimeVideoConfig(BasePipelineConfig):
    # ... existing fields ...

    # VACE (optional reference image conditioning) - NEW
    ref_images: list[str] | None = Field(
        default=None,
        description="List of reference image paths for VACE conditioning",
    )
    vace_context_scale: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Scaling factor for VACE hint injection (0.0 to 2.0)",
    )
```

**Note**: `vace_path` and `vace_in_dim` are lower-level and read from `model.yaml` or passed via `base_model_kwargs`, not the user-facing config schema.

### Step 4: Add VaceEncodingBlock to Modular Blocks

**File**: `src/scope/core/pipelines/krea_realtime_video/modular_blocks.py`

```python
from ..wan2_1.vace.blocks import VaceEncodingBlock

# Add before denoise block in the blocks list:
("vace_encoding", VaceEncodingBlock),
```

### Step 5: Update pipeline_manager.py

**File**: `src/scope/server/pipeline_manager.py`

Add VACE configuration for Krea pipeline (similar to how LongLive does it).

---

## 3. Known Pitfalls

### 3.1 VAE Path Mismatch

**Problem**: `pipeline_artifacts.py` downloads `Wan2.1_VAE.pth` only to the 1.3B folder, but `WanVAEWrapper(model_name="Wan2.1-T2V-14B")` looks in the 14B folder by default.

**Solutions** (pick one):
1. Add `Wan2.1_VAE.pth` to 14B artifact list (downloads twice, wastes space)
2. Set `vae_path` explicitly to the 1.3B VAE file in Krea config
3. Symlink 1.3B VAE to 14B folder
4. Modify `WanVAEWrapper` to fall back to shared location

**Recommended**: Option 2 (explicit `vae_path`) - cleanest, no duplicate downloads.

### 3.2 Projection Fusing After VACE

VACE wrapping replaces attention blocks with `VACECrossAttention`. If Krea uses projection fusing (for FA4/CUTE), this may need to be re-run after VACE initialization.

**Check**: Look at how projection fusing is done in Krea `__init__` and ensure it happens after `_init_vace()`.

### 3.3 Distilled Checkpoint Compatibility

Open question: Does the Krea distilled checkpoint (`krea-realtime-video-14b.safetensors`) preserve the same block structure that our VACE wrapper expects?

**Risk**: Distillation can sometimes change internal structure.

**Mitigation**: Run shape validation in `load_vace_weights_only()` - it will error if shapes don't match.

---

## 4. Validation Checklist

Before declaring VACE-14B working:

- [ ] VACE-14B artifact downloads successfully
- [ ] VACE module keys match our filter (`vace_blocks.*`, `vace_patch_embedding.*`)
- [ ] Number of VACE blocks = 20 (for 40 layers, every other layer [0,2,4,...,38])
- [ ] Shape check: `vace_patch_embedding.weight` is `(5120, 96, 1, 2, 2)`
- [ ] Projection fusing still works after VACE block replacement
- [ ] VAE loads without path errors
- [ ] Minimal R2V generation produces reasonable output
- [ ] Memory footprint acceptable (14B + VACE + FP8)

---

## 5. Files to Modify

| File | Change |
|------|--------|
| `src/scope/server/pipeline_artifacts.py` | Add VACE-14B artifact |
| `src/scope/core/pipelines/krea_realtime_video/pipeline.py` | Add mixin, VACE wiring, guarded fusing |
| `src/scope/core/pipelines/krea_realtime_video/modular_blocks.py` | Add VaceEncodingBlock |
| `src/scope/core/pipelines/schema.py` | Add `ref_images`, `vace_context_scale` to KreaRealtimeVideoConfig |
| `src/scope/core/pipelines/krea_realtime_video/model.yaml` | Add `vace_path` pointing to 14B module |

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

1. **Verify VACE-14B module exists on HuggingFace** (download test)
2. **Implement Step 1**: Add artifact
3. **Implement Steps 2-4**: Wire mixin and blocks
4. **Test**: Run Krea with VACE enabled, check for errors
5. **Validate**: Generate sample R2V output
