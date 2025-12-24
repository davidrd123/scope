# Wan 2.1 14B T2V Base Model - Observations & Research Plan

**Date**: 2024-12-24
**Updated**: 2024-12-24 (added external research findings)
**Status**: External research complete, actionable next steps identified
**Purpose**: Document what we know from codebase + external sources, plan integration work

---

## 0. Key Findings from External Research

### Architecture Specs Confirmed (from upstream config.json)

| Spec | Wan2.1-T2V-1.3B | Wan2.1-T2V-14B | Notes |
|------|----------------:|---------------:|-------|
| `dim` | 1536 | 5120 | Transformer hidden size |
| `ffn_dim` | 8960 | 13824 | MLP hidden size |
| `num_heads` | 12 | 40 | Attention heads |
| `num_layers` | 30 | 40 | Transformer depth |
| `head_dim` | 128 | 128 | **Identical** - so "weird config" isn't head_dim |

### VACE-14B Exists Upstream!

This was the key blocker assumption that turns out to be wrong:

- **Full model repo**: `Wan-AI/Wan2.1-VACE-14B`
- **Module file**: `Wan2_1-VACE_module_14B_bf16.safetensors` (community packaging)

**Implication**: Adding VACE to Krea 14B is now **artifact + wiring work**, not "requires training".

### VAE Sharing Question (Still Open)

Are the 1.3B and 14B VAEs identical? Scope currently only downloads 1.3B VAE but Krea looks for 14B path. Need to verify via checksum or upstream docs.

---

## 1. Current State Summary

### Pipeline Model Usage

| Pipeline | Base Model | Checkpoint | Notes |
|----------|------------|------------|-------|
| **Krea Realtime** | Wan2.1-T2V-14B | `krea-realtime-video-14b.safetensors` | CausVid distillation |
| **LongLive** | Wan2.1-T2V-1.3B | `longlive_base.pt` + `lora.pt` | NVlabs, 20.7 FPS on H100 |
| **StreamDiffusionV2** | Wan2.1-T2V-1.3B | `wan_causal_dmd_v2v/model.pt` | DMD distillation |
| **Reward Forcing** | Wan2.1-T2V-1.3B | `rewardforcing.pt` | Reward-guided finetuning |

### Key Observation
- **Krea is the only 14B pipeline** in the codebase
- All other pipelines (LongLive, StreamDiffusion, Reward Forcing) use **1.3B**
- VACE module only exists for 1.3B: `Wan2_1-VACE_module_1_3B_bf16.safetensors`

---

## 2. Architecture Differences (From Codebase)

### Model Configs

**Krea (14B)** - `krea_realtime_video/model.yaml`:
```yaml
base_model_name: Wan2.1-T2V-14B
base_model_kwargs:
  timestep_shift: 5.0
num_frame_per_block: 3
kv_cache_num_frames: 3
local_attn_size: 6
```

**LongLive (1.3B)** - `longlive/model.yaml`:
```yaml
base_model_name: Wan2.1-T2V-1.3B
base_model_kwargs:
  timestep_shift: 5.0
  sink_size: 3
num_frame_per_block: 3
local_attn_size: 12  # 2x larger window than Krea
```

### Code Comments About 1.3B

Found in `longlive/modules/causal_model.py`, `streamdiffusionv2/modules/causal_model.py`, `reward_forcing/modules/causal_model.py`:

```python
# wan 1.3B model has a weird channel / head configurations and require max-autotune to work with flexattention
# see https://github.com/pytorch/pytorch/issues/133254
# change to default for other models
```

This suggests 14B has "normal" attention dimensions that don't need the max-autotune workaround.

---

## 3. Questions for Web Research

### Model Architecture Questions

1. **What are the exact architecture specs for Wan2.1-T2V-14B vs 1.3B?**
   - Hidden dimension (dim)
   - Number of heads (num_heads)
   - Number of layers
   - FFN dimension
   - Total parameters breakdown

2. **What's the "weird channel/head configuration" in 1.3B?**
   - Why does it break flex_attention without max-autotune?
   - Does 14B have standard power-of-2 dimensions?

3. **Does VACE work with 14B?**
   - Is there a `Wan2_1-VACE_module_14B` anywhere?
   - Would 1.3B VACE weights transfer to 14B? (Probably not - dim mismatch)

### Available Checkpoints Questions

4. **What 14B-based distilled/finetuned models exist?**
   - Krea Realtime (CausVid distillation) - we have this
   - Any other 14B distillations? (LongLive-14B? StreamDiffusion-14B?)
   - Any 14B LoRAs available?

5. **What are the official Wan-AI 14B model variants?**
   - `Wan-AI/Wan2.1-T2V-14B` - base T2V
   - `Wan-AI/Wan2.1-I2V-14B` - I2V variant?
   - Any VACE-14B?

### Memory & Performance Questions

6. **What's the VRAM footprint for 14B?**
   - FP16/BF16 size
   - FP8 quantized size (Krea uses FP8_E4M3FN)
   - Can it fit on 24GB without quantization?

7. **Inference speed comparison 14B vs 1.3B?**
   - Is the 10x parameter increase proportional to slowdown?
   - What optimizations does Krea/CausVid use to achieve realtime?

### Codebase Integration Questions

8. **Can we port LongLive/StreamDiffusion to 14B base?**
   - Would need 14B-specific distillation checkpoints
   - Or train new distillation from 14B base
   - Code seems model-agnostic (just config changes)

9. **Can we use Krea 14B with VACE?**
   - Would need VACE-14B weights
   - Or architectural modifications to bridge dim mismatch

---

## 4. Artifacts Currently Downloaded

From `pipeline_artifacts.py`:

```python
# Krea downloads:
"Wan-AI/Wan2.1-T2V-14B" -> ["config.json"]  # Just config, not weights!
"krea/krea-realtime-video" -> ["krea-realtime-video-14b.safetensors"]

# Shared:
"Wan-AI/Wan2.1-T2V-1.3B" -> ["config.json", "Wan2.1_VAE.pth", "google"]
"Kijai/WanVideo_comfy" -> ["umt5-xxl-enc-fp8_e4m3fn.safetensors"]
```

**Observations:**
- Krea uses the **1.3B VAE** (same VAE for both models?)
- Text encoder is shared (umt5-xxl)
- 14B base weights aren't downloaded - only the Krea distilled checkpoint
- The `config.json` from 14B is used to load architecture

---

## 5. What's In the 14B config.json?

**NEED TO CHECK**: Read the actual config.json from `Wan-AI/Wan2.1-T2V-14B` repo to get:
- Hidden dim
- Number of heads
- Number of layers
- Any architectural differences from 1.3B

---

## 6. Potential Integration Paths

### Path A: Bring VACE to Krea 14B
- **Blocker**: No VACE-14B weights exist (that we know of)
- **Work needed**: Train VACE adapter for 14B architecture
- **Fallback**: Use 1.3B pipeline with VACE for reference-conditioned work

### Path B: Port LongLive to 14B
- **Requirement**: LongLive-14B distillation checkpoint
- **Code changes**: Minimal - just config + checkpoint path
- **Status**: Unknown if NVlabs has 14B version

### Path C: Use Krea 14B as-is (T2V only)
- **Current state**: Working
- **Limitations**: No VACE, no I2V, T2V only
- **Best for**: Pure text-to-video streaming

### Path D: Create 14B VACE module
- **Work**: Fine-tune VACE for 14B architecture
- **Data**: Would need VACE training data/procedure
- **Complexity**: High

---

## 7. Next Steps for Research

1. **Fetch Wan2.1-T2V-14B config.json** and document exact architecture
2. **Search HuggingFace** for:
   - VACE-14B models
   - LongLive-14B variants
   - Other 14B-based video diffusion models
3. **Read CausVid paper** (arxiv.org/abs/2412.07772) for distillation details
4. **Check Wan-AI GitHub/blog** for model variants and roadmap
5. **Benchmark memory usage** of Krea 14B with different quantization levels

---

## 8. External Resources to Fetch

| Resource | URL | What to Extract |
|----------|-----|-----------------|
| CausVid Paper | arxiv.org/abs/2412.07772 | Distillation method, architecture changes |
| Wan-AI HuggingFace | huggingface.co/Wan-AI | All available models, VACE variants |
| Krea HuggingFace | huggingface.co/krea | Model cards, training details |
| PyTorch Issue #133254 | github.com/pytorch/pytorch/issues/133254 | 1.3B attention dimension issue |
| LongLive Repo | github.com/NVlabs/LongLive | Check for 14B support |
| VACE Paper/Repo | github.com/ali-vilab/VACE | 14B VACE availability |

---

## 9. Summary & Actionable Next Steps

**Current Working State:**
- Krea Realtime Video: 14B, T2V only, works with FP8 quantization
- Other pipelines: 1.3B, have VACE support

**Key Gaps (Updated):**
- ~~No VACE for 14B~~ → **VACE-14B exists upstream, just needs wiring**
- No I2V for 14B (in this codebase)
- No 14B variants of LongLive/StreamDiffusion (would need new distillations)

---

## 10. Concrete Next Steps

### A) Add VACE to Krea 14B (Unblocked!)

1. **Add artifact** for `Wan2_1-VACE_module_14B_bf16.safetensors`:
   ```python
   # In pipeline_artifacts.py
   VACE_14B_ARTIFACT = HuggingfaceRepoArtifact(
       repo_id="Kijai/WanVideo_comfy",  # or wherever it's hosted
       files=["Wan2_1-VACE_module_14B_bf16.safetensors"],
   )
   ```

2. **Add mixin to Krea pipeline**:
   ```python
   # In krea_realtime_video/pipeline.py
   from ..wan2_1.vace.mixin import VACEEnabledPipeline

   class KreaRealtimeVideoPipeline(Pipeline, LoRAEnabledPipeline, VACEEnabledPipeline):
   ```

3. **Wire up in __init__** (after LoRA, before quantization):
   ```python
   generator.model = self._init_vace(config, generator.model, device, dtype)
   ```

4. **Re-run projection fusing** after VACE wrapping (VACE replaces blocks)

5. **Add VaceEncodingBlock** to modular_blocks.py

### B) Verify VAE Sharing

```bash
# Compare checksums
md5sum wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth
md5sum wan_models/Wan2.1-T2V-14B/Wan2.1_VAE.pth  # if exists
```

Or check upstream HuggingFace - if same file, standardize to shared path.

### C) Investigate "Weird Head Config" (Lower Priority)

The 1.3B flex_attention issue isn't head_dim (both are 128). Likely:
- Non-power-of-2 total dim (1536 vs 5120)?
- Num heads (12 vs 40)?
- Some other kernel tuning heuristic?

Reference: PyTorch issue #133254

### D) 14B Distillations for Other Pipelines (Future)

No 14B LongLive/StreamDiffusion found. Would require:
- Training new distillations on 14B base
- Or using Krea 14B as the 14B realtime option (current state)

---

## 11. Risk Checklist for VACE-14B Integration

Before declaring VACE-14B working:

- [ ] VACE module keys match Scope wrapper (`vace_blocks.*`, `vace_patch_embedding.*`)
- [ ] Number of VACE blocks matches `vace_layers = range(0, num_layers, 2)` for 40 layers
- [ ] Shape check passes in `load_vace_weights_only()`
- [ ] Projection fusing still works after VACE block replacement
- [ ] Minimal R2V generation produces reasonable output
- [ ] Memory footprint acceptable (14B + VACE + FP8)
