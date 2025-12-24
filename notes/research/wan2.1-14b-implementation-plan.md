# VACE-14B Implementation Plan

**Created**: 2024-12-24
**Context**: Ready to implement VACE support for Krea 14B pipeline

---

## Current State (What We Just Discovered)

From local filesystem check:
- **No VACE-14B module downloaded yet** - `WanVideo_comfy/` only has the text encoder
- **No VAE in 14B folder** - `Wan2.1-T2V-14B/` only has `config.json`
- **VAE lives in 1.3B folder** - `Wan2.1-T2V-1.3B/Wan2.1_VAE.pth` (507MB)

---

## External Dependencies Needed

### 1. VACE-14B Module Weights
**Need to find and download**: `Wan2_1-VACE_module_14B_bf16.safetensors`

Likely location: `Kijai/WanVideo_comfy` on HuggingFace (same repo as 1.3B module)

**Action**: Check HuggingFace for exact filename and add to artifacts

### 2. VAE Situation (Verify)
Current observation: Krea 14B probably uses the 1.3B VAE via path override or symlink.

**Action**:
- Check if `vae_path` is explicitly set in Krea config/runtime
- Or verify both model sizes share the same VAE (checksum upstream)
- Simplest fix: Add explicit `vae_path` to Krea config pointing to shared VAE

---

## Implementation Steps

### Step 1: Add VACE-14B Artifact
```python
# In src/scope/server/pipeline_artifacts.py
VACE_14B_ARTIFACT = HuggingfaceRepoArtifact(
    repo_id="Kijai/WanVideo_comfy",
    files=["Wan2_1-VACE_module_14B_bf16.safetensors"],
)

# Add to krea-realtime-video artifacts list
```

### Step 2: Add VACEEnabledPipeline Mixin to Krea
```python
# In src/scope/core/pipelines/krea_realtime_video/pipeline.py

from ..wan2_1.vace.mixin import VACEEnabledPipeline

class KreaRealtimeVideoPipeline(Pipeline, LoRAEnabledPipeline, VACEEnabledPipeline):
    ...
```

### Step 3: Wire VACE in __init__
In `KreaRealtimeVideoPipeline.__init__()`, after generator creation but before quantization:

```python
# After: generator = WanDiffusionWrapper(...)
# Before: if quantization == Quantization.FP8_E4M3FN:

generator.model = self._init_vace(config, generator.model, device, dtype)

# Note: May need to re-run projection fusing after VACE wrapping
# because VACE replaces the attention blocks
```

### Step 4: Add VaceEncodingBlock to Modular Blocks
```python
# In src/scope/core/pipelines/krea_realtime_video/modular_blocks.py

from ..wan2_1.vace.blocks import VaceEncodingBlock

# Add before denoise block:
("vace_encoding", VaceEncodingBlock),
```

### Step 5: Update pipeline_manager.py
Add VACE configuration for Krea pipeline (similar to how longlive does it).

---

## Risk Items to Validate

- [ ] VACE-14B module exists at `Kijai/WanVideo_comfy`
- [ ] Key structure matches: `vace_blocks.*`, `vace_patch_embedding.*`
- [ ] 14B has 40 layers → expect 20 vace_blocks (every other layer)
- [ ] Projection fusing still works after VACE block replacement
- [ ] Memory fits: 14B + VACE + FP8 quantization

---

## Files to Modify

1. `src/scope/server/pipeline_artifacts.py` - Add VACE-14B artifact
2. `src/scope/core/pipelines/krea_realtime_video/pipeline.py` - Add mixin + wiring
3. `src/scope/core/pipelines/krea_realtime_video/modular_blocks.py` - Add VaceEncodingBlock
4. `src/scope/server/pipeline_manager.py` - Add VACE config for Krea

---

## First Action After Context Reload

1. Check HuggingFace `Kijai/WanVideo_comfy` for VACE-14B module file
2. If exists, proceed with implementation
3. If not, check `Wan-AI/Wan2.1-VACE-14B` for module extraction

---

## Reference Docs Created This Session

- `src/scope/core/pipelines/krea_realtime_video/docs/architecture-guide.md` - Full pipeline explainer
- `notes/research/wan2.1-14b-integration-observations.md` - Research findings + next steps
- `notes/research/wan2.1-14b-repoprompt-query.md` - (Can delete, research complete)
