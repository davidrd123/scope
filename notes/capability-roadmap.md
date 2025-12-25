# Capability Roadmap

> **Purpose:** Track capability features (vs. performance optimization)
> **Created:** 2025-12-25
> **See also:** `notes/FA4/b300/optimization-vision.md` for performance work

---

## Executive Summary

Three capability features in the pipeline:

| Feature | Status | Blocking? | ETA |
|---------|--------|-----------|-----|
| **Style Layer (Phase 6a)** | In Progress | No | Now |
| **VACE-14B Integration** | Ready to Implement | No | Next |
| **Context Editing** | Speculative | Needs validation spike | TBD |

---

## 1. Style Layer (Phase 6a) — IN PROGRESS

**What:** Wire WorldState + StyleManifest + TemplateCompiler into live session and REST API.

**Owner:** Codex (implementation in progress)

**Status:** Plan approved, implementation starting.

### Scope

- Add WorldState/StyleManifest/TemplateCompiler to FrameProcessor
- 4 REST endpoints under `/api/v1/realtime/`:
  - `GET /state` — includes world_state, active_style, compiled_prompt
  - `PUT /world` — replace full WorldState
  - `PUT /style` — set active style by name
  - `GET /style/list` — available styles from registry
- Extend Snapshot to preserve style state
- CLI commands: `video-cli world`, `video-cli style`
- Minimal RAT manifest (`styles/rat/manifest.yaml`)

### Key Design Decisions

1. **Cohesive REST surface** — All under `/api/v1/realtime/`
2. **Full replace, not patch** — PUT replaces entire WorldState
3. **LoRA edge-trigger** — Only send lora_scales when style changes
4. **Prompt precedence** — Explicit prompts win over compiled prompts
5. **Thread-safe** — Atomic replace via `model_copy(update=...)`

### Files

| File | Changes |
|------|---------|
| `src/scope/server/frame_processor.py` | WorldState, StyleManifest, TemplateCompiler; extend Snapshot |
| `src/scope/server/app.py` | 4 REST endpoints; extend RealtimeStateResponse |
| `src/scope/server/schema.py` | Request/response schemas |
| `src/scope/server/webrtc.py` | Forward `_rcp_world_state`, `_rcp_set_style` |
| `src/scope/cli/video_cli.py` | world/style commands |
| `styles/rat/manifest.yaml` | NEW — Minimal RAT style |
| `tests/test_style_integration.py` | NEW — Integration tests |

### Reference

- Plan: `notes/plans/phase6-prompt-compilation.md`
- Architecture: `notes/realtime_video_architecture.md`

---

## 2. VACE-14B Integration — READY TO IMPLEMENT

**What:** Add VACE (reference image conditioning) support to Krea 14B pipeline.

**Status:** Ready to implement — engineering + wiring work, not blocked on weights.

### Why It Matters

- **VACE** = "Video Anything with Controllable Editing"
- Enables reference image conditioning: "generate video that looks like this image"
- Currently available on 1.3B pipelines (LongLive, StreamDiffusionV2, Reward Forcing)
- Krea 14B is the **only 14B pipeline** and the **only one without VACE**

### Upstream Availability

| Source | File | Size |
|--------|------|------|
| `Kijai/WanVideo_comfy` | `Wan2_1-VACE_module_14B_bf16.safetensors` | 6.1 GB |

Verified: File exists, SHA256: `66a4bd41ec0fc58f1ff6d1313e06cd9a4c24ab60171a5846937536f8d4de6a65`

### Implementation Steps

1. **Add VACE-14B Artifact** → `src/scope/server/pipeline_artifacts.py`
2. **Add VACEEnabledPipeline Mixin** → `src/scope/core/pipelines/krea_realtime_video/pipeline.py`
3. **Wire VACE in __init__** (VACE before fusing, fusing handles both block types)
4. **Add Config Schema Fields** → `ref_images`, `vace_context_scale`
5. **Add VaceEncodingBlock** → `modular_blocks.py`
6. **Update PipelineManager** → Select 14B VACE module path (currently hardcoded to 1.3B)
7. **Add `vace_enabled` toggle** → `KreaRealtimeVideoLoadParams` (decide: load-time optional or always-on)

### Files to Modify

| File | Change |
|------|--------|
| `src/scope/server/pipeline_artifacts.py` | Add VACE-14B artifact |
| `src/scope/core/pipelines/krea_realtime_video/pipeline.py` | Add mixin, VACE wiring, guarded fusing |
| `src/scope/core/pipelines/krea_realtime_video/modular_blocks.py` | Add VaceEncodingBlock |
| `src/scope/core/pipelines/schema.py` | Add `ref_images`, `vace_context_scale` to KreaRealtimeVideoConfig |
| `src/scope/server/schema.py` | Add `vace_enabled` to `KreaRealtimeVideoLoadParams` |
| `src/scope/server/pipeline_manager.py` | Update `_get_vace_checkpoint_path()` for 14B; call `_configure_vace` for Krea |

### Known Pitfalls

1. **VAE Path Mismatch** — `Wan2.1_VAE.pth` only in 1.3B folder; set explicit `vae_path` in config
2. **Projection Fusing After VACE** — VACE creates new blocks; fuse AFTER `_init_vace()`
3. **Distilled Checkpoint Compat** — Run shape validation to confirm structure match
4. **PipelineManager gap** — `_get_vace_checkpoint_path()` is hardcoded to 1.3B module; Krea never calls `_configure_vace`
5. **Config plumbing** — `model.yaml` alone won't enable VACE; must plumb `vace_path` through PipelineManager into runtime config

### Open Decision

**VACE loading strategy for Krea:**
- **Option A: Always-on** — VACE always loaded, `ref_images` optional at runtime
- **Option B: Load-time toggle** — Add `vace_enabled` to `KreaRealtimeVideoLoadParams` like other pipelines

Recommend **Option B** to avoid 6 GB memory overhead when VACE not needed.

### Validation Checklist

- [ ] VACE-14B artifact downloads successfully
- [ ] 20 VACE blocks created (40 layers ÷ 2)
- [ ] Shape check: `vace_patch_embedding.weight` is `(5120, 96, 1, 2, 2)`
- [ ] Projection fusing works after VACE block replacement
- [ ] VAE loads without path errors
- [ ] Minimal R2V generation produces output
- [ ] Memory footprint acceptable (14B + VACE + FP8)

### Reference

- Full plan: `notes/vace-14b-integration/plan.md`
- VACE mixin: `src/scope/core/pipelines/wan2_1/vace/mixin.py`
- Example pipeline: `src/scope/core/pipelines/longlive/pipeline.py`

---

## 3. Context Editing — SPECULATIVE

**What:** Edit frames in the pipeline's decoded buffer, triggering KV cache recomputation so changes propagate to future frames.

**Status:** Speculative — needs validation spike before committing to implementation.

### The Insight

KREA's `recompute_kv_cache.py` re-encodes the anchor frame from RGB during KV cache recomputation:

```python
# From recompute_kv_cache.py, get_context_frames()
decoded_first_frame = state.decoded_frame_buffer[:, :1]
reencoded_latent = vae.encode_to_latent(decoded_first_frame)
```

This creates an **edit surface**: modify `decoded_frame_buffer[:, :1]` → next recompute encodes the edit → KV cache reflects the change → future frames "remember" the edit.

### What This Enables

| Operation | Description |
|-----------|-------------|
| Error correction | Remove hallucinated limb, fix identity drift |
| Retroactive insertion | "There should have been a knife on the table" |
| Character modification | Costume change, add injury, fix expression |
| Environment tweaks | Add rain, change lighting, time of day |

### Validation Spike (Hour 1 Test)

Simple test without any image edit model — just tint the anchor frame blue:

```python
# Get decoded buffer
decoded_buffer = pipeline.state.get('decoded_frame_buffer')

# Tint anchor blue
anchor_frame = decoded_buffer[:, :1].clone()
anchor_frame[:, :, 2, :, :] = 1.0  # Max blue
anchor_frame[:, :, 0, :, :] = 0.0  # Zero red
decoded_buffer[:, :1] = anchor_frame

# Continue generating...
```

**Expected results:**

| Outcome | Interpretation | Next Step |
|---------|---------------|-----------|
| Scene goes blue, stays blue | Edit propagates through KV cache | Integrate real edit model |
| Flickers blue then reverts | Model "corrects" back to prior | Try aligning prompt with edit |
| No visible change | Edit not reaching recompute path | Check timing, buffer indices |
| Generation breaks | Edit too aggressive | Try subtler mutation |

### Dependencies

- **nano-banana** (or similar image edit model) for semantic edits
- **VLM integration** for frame description / agent evaluation loop

### Open Questions

- [ ] What's the Python API for nano-banana?
- [ ] Latency expectation: <1s? 1-3s?
- [ ] Frame format: PIL Image? Tensor? Base64?
- [ ] How to handle edits that conflict with prompt?

### Reference

- Full spec: `notes/research/2025-12-24/incoming/context_editing_and_console_spec.md`
- KREA recompute: `src/scope/core/pipelines/krea_realtime_video/components/recompute_kv_cache.py`

---

## Dependency Graph

```
Style Layer (Phase 6a)
    │
    ├── [independent] VACE-14B Integration
    │
    └── [independent] Context Editing
                         │
                         └── depends on: nano-banana / image edit model
                         └── depends on: VLM for agent loop
```

All three features are independent of each other. Style Layer is in progress; VACE-14B is ready; Context Editing needs validation first.

---

## Priority Recommendation

1. **Style Layer** — In progress, let it complete
2. **VACE-14B** — Next up, straightforward engineering work
3. **Context Editing** — Run validation spike before committing to full implementation

---

## Related Performance Work

See `notes/FA4/b300/optimization-vision.md` for performance optimization roadmap:
- Current: 15 FPS @ 320×576 with FA4
- Target: 24+ FPS
- Blocked: torch.compile, recompute cadence
- Next: cuDNN benchmark, SageAttention, torchao fix
