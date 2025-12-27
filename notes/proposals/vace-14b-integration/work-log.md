# VACE-14B Integration Work Log

Chronological log of work on adding VACE support to Krea 14B pipeline.

For current “do this next” instructions, start with:
- `notes/vace-14b-integration/README.md`
- `notes/vace-14b-integration/plan.md`

---

## 2024-12-24: Research & Planning

### Session 1: Initial Investigation

**Goal**: Understand what's needed to add VACE to Krea 14B

**Findings**:
- Krea is the only 14B pipeline in the codebase
- All other pipelines (LongLive, StreamDiffusionV2, Reward Forcing) use 1.3B and already have VACE
- Initially thought VACE-14B might not exist upstream

**Actions**:
- Created architecture guide for `krea_realtime_video/` pipeline
- Documented I2V code exists but isn't used (Krea is T2V-only)
- Explored VACE implementation in `wan2_1/vace/`

### Session 2: External Research

**Goal**: Verify upstream VACE-14B availability

**Findings** (confirmed via HuggingFace):
- VACE-14B **does exist**:
  - Official: `Wan-AI/Wan2.1-VACE-14B`
  - Module-only: `Kijai/WanVideo_comfy/Wan2_1-VACE_module_14B_bf16.safetensors`
- Architecture verified from `config.json`:
  - 14B: dim=5120, layers=40, heads=40, head_dim=128
  - 1.3B: dim=1536, layers=30, heads=12, head_dim=128
- VAE path mismatch identified: artifacts download VAE to 1.3B folder only

**Actions**:
- Created research observations doc
- Created implementation plan
- Identified 5 implementation steps + pitfalls

### Session 3: Consolidation

**Goal**: Organize research into canonical docs

**Actions**:
- Created `notes/vace-14b-integration/` directory
- Consolidated research + plan into `plan.md`
- Created this work log

**Status**: Ready to begin implementation

### Session 4: Source Code Verification

**Goal**: Verify assumptions against actual source code before implementing

**External patch claimed**:
- VACE layers: `range(num_layers // 4, num_layers, 4)` → 8 blocks for 14B
- `_init_vace(generator)` signature

**Actual source code says** (verified from `wan2_1/vace/`):
- VACE layers: `range(0, num_layers, 2)` = **every other layer** → **20 blocks** for 14B
- `_init_vace(config, model, device, dtype)` signature

**VACE-14B module file**:
- Initial web scrape missed it, but file **DOES exist**:
  - `Kijai/WanVideo_comfy/Wan2_1-VACE_module_14B_bf16.safetensors`
  - Size: 6.1 GB
  - SHA256: `66a4bd41ec0fc58f1ff6d1313e06cd9a4c24ab60171a5846937536f8d4de6a65`

**Files read for verification**:
- `src/scope/core/pipelines/wan2_1/vace/mixin.py` - VACEEnabledPipeline mixin
- `src/scope/core/pipelines/wan2_1/vace/models/causal_vace_model.py` - VACE layer logic
- `src/scope/core/pipelines/wan2_1/vace/utils/weight_loader.py` - Key filtering logic

**Plan corrections applied**:
- VACE blocks: 20 for 14B (not 8), 15 for 1.3B (not 6)
- `_init_vace` signature: `(config, model, device, dtype)` not `(generator)`
- Module file: verified exists with size and hash
- Added guarded `_fuse()` helper for projection fusing

**Status**: Plan verified against source code, ready for implementation

### Session 5: Config Schema & Pipeline Wiring Deep Dive

**Goal**: Find where Krea config is defined, understand projection fusing order

**Findings**:

1. **Config schema location**: `src/scope/core/pipelines/schema.py`
   - `KreaRealtimeVideoConfig` at lines 322-369
   - Does NOT have VACE fields (unlike LongLive, StreamDiffusionV2, RewardForcing)
   - Need to add: `ref_images`, `vace_context_scale`

2. **Projection fusing order matters**:
   - Krea does fusing at lines 85-86 (before LoRA)
   - VACE creates NEW blocks via factory pattern
   - New blocks inherit `fuse_projections()` but start unfused
   - **Correct order**: VACE wrapping → fuse ALL blocks → LoRA

3. **LongLive comparison** (`longlive/pipeline.py`):
   - Line 36: Class inherits `VACEEnabledPipeline`
   - Line 87-89: VACE wrapping before LoRA
   - Does NOT have projection fusing (Krea is unique)

4. **VACE attention blocks** (`attention_blocks.py`):
   - Factory pattern creates classes inheriting from pipeline's block class
   - Both `VaceWanAttentionBlock` and `BaseWanAttentionBlock` inherit `fuse_projections`

**Updated plan with**:
- Correct code ordering (VACE → fuse → LoRA)
- Actual config schema file path
- Fields to add (`ref_images`, `vace_context_scale`)
- `model.yaml` for `vace_path`

**Status**: Plan fully verified, implementation-ready

---

## Next Session: Implementation

Planned work:
1. Verify VACE-14B module downloads from HuggingFace
2. Add artifact to `pipeline_artifacts.py`
3. Wire `VACEEnabledPipeline` mixin into Krea
4. Add `VaceEncodingBlock` to modular blocks
5. Test and validate

---

## 2025-12-25: Repo-wide Integration Audit (What’s Actually Missing)

**Goal**: Verify the plan against current server + pipeline behavior and identify the real missing wiring points for Krea 14B.

**Corrections / clarifications**:
- Krea is **not** “T2V-only” in the repo; it supports text+video modes, but VACE is not wired.
- The VAE “14B path mismatch” is **already mitigated** for Krea by explicitly setting `vae_path` in `src/scope/server/pipeline_manager.py`.

**Key findings (must implement)**:
1. **Server routing depends on `pipeline.vace_enabled`**, not request params:
   - `src/scope/server/frame_processor.py` routes incoming frames to `vace_input_frames` only when VACE was enabled at pipeline init.
2. **Krea is missing the server-side toggle**:
   - Add `vace_enabled` to `src/scope/server/schema.py:KreaRealtimeVideoLoadParams` (likely default `False`).
   - Call `_configure_vace()` in Krea branch of `src/scope/server/pipeline_manager.py` when enabled.
3. **VACE checkpoint path selection is hardcoded to 1.3B**:
   - `src/scope/server/pipeline_manager.py:_get_vace_checkpoint_path()` must select the 14B module for Krea.
4. **Krea pipeline wiring is incomplete**:
   - `src/scope/core/pipelines/krea_realtime_video/pipeline.py` must inherit `VACEEnabledPipeline` and call `_init_vace()` before projection fusing + LoRA.
   - Must fuse projections on `vace_blocks` as well (when present).
   - Must clear `vace_ref_images` in `_generate()` when not provided (one-shot semantics).
5. **Krea block graph is missing VACE encoding**:
   - Insert `VaceEncodingBlock` into `src/scope/core/pipelines/krea_realtime_video/modular_blocks.py` before `denoise`.

**Major decision points (document before coding)**:
- Quantization policy: should VACE-14B blocks also be FP8 quantized (memory) or kept bf16 (quality/stability)?
- KV-cache recompute semantics: `RecomputeKVCacheBlock` currently calls generator without `vace_context`; decide whether that’s acceptable for your VACE use-cases.

<!-- Template for new entries:

## YYYY-MM-DD: [Title]

**Goal**:

**Findings**:
-

**Actions**:
-

**Blockers**:
-

**Status**:

-->
