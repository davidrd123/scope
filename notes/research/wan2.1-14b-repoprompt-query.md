# Repo Prompt Query: Wan 2.1 14B T2V Integration Research

## Context Document
Include: `notes/research/wan2.1-14b-integration-observations.md`

## Query Prompt

```
I'm researching how to integrate Wan 2.1 14B text-to-video capabilities more broadly across this codebase. Currently, only the Krea Realtime pipeline uses 14B, while LongLive, StreamDiffusion, and other pipelines use the 1.3B model.

I need to understand:

1. **Architecture Differences**: What are the exact architectural differences between Wan 2.1 14B and 1.3B? The code comments mention 1.3B has "weird channel/head configurations" - what specifically is different about 14B?

2. **VACE Compatibility**: VACE (reference image conditioning) only has 1.3B weights. Is it architecturally possible to use VACE with 14B? What would need to change?

3. **Pipeline Portability**: How model-agnostic are the pipeline implementations? Could LongLive or StreamDiffusion be switched to 14B base with just config changes, or would they need new distillation checkpoints?

4. **External Resources Needed**:
   - Does VACE-14B exist anywhere (HuggingFace, GitHub)?
   - Are there 14B versions of LongLive or StreamDiffusion distillations?
   - What does the CausVid paper (Krea's base) say about architecture changes for the distillation?

Please gather all relevant code for understanding the 14B vs 1.3B model architecture, VACE integration points, and pipeline configurability.
```

## Files to Include

### Core Model Configurations
- `src/scope/core/pipelines/krea_realtime_video/model.yaml`
- `src/scope/core/pipelines/longlive/model.yaml`
- `src/scope/core/pipelines/streamdiffusionv2/model.yaml`
- `src/scope/core/pipelines/reward_forcing/model.yaml`

### Causal Model Implementations (Architecture Details)
- `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py` (14B version)
- `src/scope/core/pipelines/longlive/modules/causal_model.py` (1.3B version)
- `src/scope/core/pipelines/krea_realtime_video/modules/model.py`

### Pipeline Entry Points
- `src/scope/core/pipelines/krea_realtime_video/pipeline.py`
- `src/scope/core/pipelines/longlive/pipeline.py`
- `src/scope/core/pipelines/streamdiffusionv2/pipeline.py`

### VACE Implementation
- `src/scope/core/pipelines/wan2_1/vace/mixin.py`
- `src/scope/core/pipelines/wan2_1/vace/models/causal_vace_model.py`
- `src/scope/core/pipelines/wan2_1/vace/utils/weight_loader.py`

### Shared Components
- `src/scope/core/pipelines/wan2_1/components/generator.py` (WanDiffusionWrapper)
- `src/scope/server/pipeline_artifacts.py` (what checkpoints are downloaded)

### Research Document
- `notes/research/wan2.1-14b-integration-observations.md`

## Glob Patterns for Repo Prompt

```
# Model configs
src/scope/core/pipelines/*/model.yaml

# Causal model implementations
src/scope/core/pipelines/*/modules/causal_model.py
src/scope/core/pipelines/*/modules/model.py

# Pipeline entry points
src/scope/core/pipelines/*/pipeline.py

# VACE
src/scope/core/pipelines/wan2_1/vace/**/*.py

# Shared wan2_1 components
src/scope/core/pipelines/wan2_1/components/*.py

# Artifacts
src/scope/server/pipeline_artifacts.py

# Research doc
notes/research/wan2.1-14b-integration-observations.md
```

## Key Code Sections to Highlight

### 1. The 1.3B "Weird Channel" Comment
Look for this pattern in causal_model.py files:
```python
# wan 1.3B model has a weird channel / head configurations and require max-autotune to work with flexattention
# see https://github.com/pytorch/pytorch/issues/133254
```

### 2. Model Loading in WanDiffusionWrapper
`src/scope/core/pipelines/wan2_1/components/generator.py`:
- How config.json is loaded
- How model_name determines architecture
- The `filter_causal_model_cls_config()` function

### 3. VACE Dimension Requirements
`src/scope/core/pipelines/wan2_1/vace/models/causal_vace_model.py`:
- `vace_in_dim` parameter (default 96)
- How VACE hooks into the base model's `dim`

### 4. Attention Dimensions
In any causal_model.py, look for:
```python
self.dim = dim
self.num_heads = num_heads
self.head_dim = dim // num_heads
```

## Web Research Tasks

After gathering codebase context, research these external sources:

1. **HuggingFace Search**:
   - `Wan-AI/Wan2.1-T2V-14B` - full model card
   - Search for "VACE 14B" or "Wan2.1-VACE-14B"
   - Search for "LongLive 14B"

2. **GitHub Search**:
   - `ali-vilab/VACE` - check for 14B support
   - `NVlabs/LongLive` - check for 14B variant
   - `krea-ai/realtime-video` - original Krea implementation

3. **Papers**:
   - CausVid: arxiv.org/abs/2412.07772
   - VACE paper (if separate from Wan)
   - LongLive paper

4. **PyTorch Issue**:
   - github.com/pytorch/pytorch/issues/133254 - understand the 1.3B flex_attention issue

## Expected Outputs

After research, update `notes/research/wan2.1-14b-integration-observations.md` with:

1. **Confirmed architecture specs** for 14B vs 1.3B
2. **VACE-14B availability** (exists / doesn't exist / would need training)
3. **14B distillation options** (what's available, what would need to be created)
4. **Recommended path forward** for 14B integration
