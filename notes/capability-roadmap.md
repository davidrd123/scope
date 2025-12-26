# Capability Roadmap

> **Purpose:** Track capability features (vs. performance optimization)
> **Created:** 2025-12-25
> **See also:** `notes/FA4/b300/optimization-vision.md` for performance work

---

## Executive Summary

Capability features in the pipeline:

| Feature | Status | Blocking? | Proposal |
|---------|--------|-----------|----------|
| **Style Layer (Phase 6a)** | In Progress | No | Below |
| **VACE-14B Integration** | Ready to Implement | No | Below |
| **Server-Side Session Recorder** | Ready to Implement | No | `proposals/server-side-session-recorder.md` |
| **Style Swap Mode** | Ready to Implement | No | `proposals/style-swap-mode.md` |
| **VLM Frame Analysis** | Ready to Implement | No | `proposals/vlm-integration.md` |
| **VLM-Mediated V2V** | Speculative | Needs VLM first | Below |
| **Context Editing** | Speculative | Needs validation spike | Below |
| **Tidal Cycles Integration** | Speculative | Needs Tidal setup | `proposals/tidal-cycles-integration.md` |
| **NDI / Pub-Sub Video Output** | Ready to Implement | No | `proposals/ndi-pubsub-video-output.md` |

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

## 2. VACE-14B Integration — READY TO IMPLEMENT (Not Started)

**What:** Add VACE (reference image conditioning) support to Krea 14B pipeline.

**Status:** Plan complete, implementation NOT STARTED for Krea. Infrastructure exists (used by 1.3B pipelines), but Krea needs wiring + correct artifact + blocks.

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

### What's Already In Place (Good News)

| Component | Status | Location |
|-----------|--------|----------|
| VACEEnabledPipeline mixin | ✓ Exists | `src/scope/core/pipelines/wan2_1/vace/mixin.py` |
| VaceEncodingBlock | ✓ Exists | `src/scope/core/pipelines/wan2_1/vace/blocks/vace_encoding.py` |
| Runtime VACE routing | ✓ Works | `frame_processor.py:1085` routes to `vace_input_frames` when `pipeline.vace_enabled` |
| VAE path handling | ✓ Already set | `pipeline_manager.py:481` sets explicit `vae_path` for Krea |

### What's Missing for Krea

| Gap | Current State | Fix |
|-----|---------------|-----|
| 14B VACE artifact | Only 1.3B module in artifacts | Add `Wan2_1-VACE_module_14B_bf16.safetensors` |
| Checkpoint path | Hardcoded to 1.3B | Update `_get_vace_checkpoint_path()` for 14B |
| Pipeline init | Krea doesn't inherit mixin | Add `VACEEnabledPipeline` + call `_init_vace()` |
| PipelineManager | Never calls `_configure_vace` for Krea | Add `_configure_vace()` call |
| Modular blocks | No `VaceEncodingBlock` in Krea | Add to `modular_blocks.py` |
| Schema fields | No `ref_images`/`vace_context_scale` | Add to `KreaRealtimeVideoConfig` |
| Load params | No `vace_enabled` toggle | Add to `KreaRealtimeVideoLoadParams` |

### Known Pitfalls

1. **VAE Path Mismatch** — `Wan2.1_VAE.pth` only in 1.3B folder; already handled for Krea
2. **Projection Fusing After VACE** — VACE creates new blocks; fuse AFTER `_init_vace()`
3. **Distilled Checkpoint Compat** — Run shape validation to confirm structure match
4. **One-shot semantics** — Add logic to clear stale `vace_ref_images` after single use (like LongLive does at `longlive/pipeline.py:220`), otherwise Krea will keep re-encoding refs

### Open Decision

**VACE loading strategy for Krea:**
- **Option A: Always-on** — VACE always loaded, `ref_images` optional at runtime
- **Option B: Load-time toggle** — Add `vace_enabled` to `KreaRealtimeVideoLoadParams` like other pipelines

Recommend **Option B** to avoid 6 GB memory overhead when VACE not needed.

### Phase 6b Execution Checklist

**1. Decide defaults**
- [ ] Pick `vace_enabled` default for Krea load params (recommend `False` to avoid +6.1GB unless requested)

**2. Artifacts**
- [ ] Add `Wan2_1-VACE_module_14B_bf16.safetensors` to Krea downloads → `pipeline_artifacts.py:43`

**3. PipelineManager wiring**
- [ ] Stop using 1.3B hardcode for Krea → `pipeline_manager.py:216` select 14B filename
- [ ] Add `vace_enabled` handling in Krea load path → `pipeline_manager.py:458` call `_configure_vace()` when enabled

**4. Schema / API surfacing**
- [ ] Add `vace_enabled` to `KreaRealtimeVideoLoadParams` → `server/schema.py:352`
- [ ] Add `ref_images` + `vace_context_scale` to `KreaRealtimeVideoConfig` → `pipelines/schema.py:322`

**5. Krea pipeline model wiring**
- [ ] Make Krea inherit `VACEEnabledPipeline` + call `_init_vace()` before fusing/LoRA → `pipeline.py:42`
- [ ] Update fusing loops to include `model.vace_blocks` when present → `pipeline.py:85`

**6. Krea block graph**
- [ ] Insert `("vace_encoding", VaceEncodingBlock)` before denoise → `modular_blocks.py:152`

**7. State hygiene (avoid re-encoding + stale tensors)**
- [ ] Clear `vace_ref_images` when not provided (LongLive pattern)
- [ ] Clear `vace_input_frames` when absent → `pipeline.py:210`

**8. (Recommended) Cache recompute correctness**
- [ ] Pass `vace_context` + `vace_context_scale` through KV recompute → `recompute_kv_cache.py:16`

**9. Tests (CPU-only)**
- [ ] Krea block list contains `vace_encoding`
- [ ] PipelineManager chooses 14B module path for Krea
- [ ] Schema includes new fields

**10. GPU validation**
- [ ] Weight shape: `vace_patch_embedding.weight == (5120, 96, 1, 2, 2)`
- [ ] Load Krea with `vace_enabled=true`, send `vace_ref_images` once
- [ ] Confirm no re-encoding on later chunks (memory stable)

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

## 4. Server-Side Session Recorder — READY TO IMPLEMENT

**What:** Capture all control events at the server level (CLI, API, frontend) for timeline export and offline re-rendering.

**Status:** Proposal hardened, ready for implementation.

**Proposal:** `notes/proposals/server-side-session-recorder.md`

### Key Features

- Records: prompt changes, transitions, hard cuts, soft cuts
- Timebase: chunk_index (primary) + wall_clock (secondary)
- Output: `~/.daydream-scope/recordings/session_*.timeline.json`
- Compatible with `render_timeline.py` for offline re-rendering

### Implementation Summary

1. `SessionRecorder` class in `src/scope/server/session_recorder.py`
2. Integration via reserved keys (`_rcp_session_recording_start/stop`)
3. API endpoints: `POST /start`, `POST /stop`, `GET /status`
4. CLI: `R` key in playlist nav

---

## 5. Style Swap Mode — READY TO IMPLEMENT

**What:** Enable instant style/LoRA switching without pipeline reload by preloading all LoRAs at startup.

**Status:** Proposal hardened, ready for implementation.

**Proposal:** `notes/proposals/style-swap-mode.md`

### Key Features

- Preload all style LoRAs at pipeline load time
- Force `runtime_peft` merge strategy (required for scale updates)
- Switch styles by setting `lora_scales` (active=1.0, others=0.0)
- ~50% FPS tradeoff for instant switching capability

### Environment Variables

- `STYLE_SWAP_MODE=1` — Enable preloading
- `STYLE_DEFAULT=<name>` — Initial active style
- `SCOPE_STYLES_DIRS=/path` — Additional style directories

---

## 6. VLM Frame Analysis — READY TO IMPLEMENT

**What:** Analyze generated frames with Gemini Vision to get descriptions for agent loops / monitoring.

**Status:** Ready to implement (gemini_client.py exists).

**Proposal:** `notes/proposals/vlm-integration.md`

### API

- Model: `gemini-2.0-flash` or `gemini-2.5-flash`
- Endpoint: `GET/POST /api/v1/realtime/frame/describe`
- CLI: `video-cli describe-frame`

### Use Cases

- Agent feedback loop (evaluate if prompt is being followed)
- Monitoring / quality checks
- Input for VLM-Mediated V2V (see below)

---

## 7. VLM-Mediated Video-to-Video — SPECULATIVE

**What:** Use an external video source (webcam, screen capture) as input, run VLM captioning on it continuously, and use those captions to drive generative video output.

**Status:** Speculative — depends on VLM Frame Analysis being implemented first.

### The Concept

```
External Video → VLM Caption → Prompt Stream → Generative Model → Styled Output
   (webcam)      (Gemini)      (continuous)      (Krea/WAN)       (LoRA style)
```

The VLM acts as a **semantic bridge** — it "watches" the input and "describes" it, and that description drives generation. This is **indirect V2V via text captioning mediation**.

### Creative Applications

| Use Case | Description |
|----------|-------------|
| Style mirroring | Mirror yourself in anime/Rooster&Terry style |
| Movement visualization | Abstract interpretations of live movement |
| Performance art | Live performance → styled reimagining |
| Screen → art | Screen capture → artistic reinterpretation |

### Architecture (Sketch)

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Video Source   │────▶│  VLM Captioner  │────▶│  Prompt Stream  │
│  (webcam/OBS)   │     │  (Gemini Flash) │     │  (continuous)   │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                        ┌─────────────────┐              │
                        │  Generative     │◀─────────────┘
                        │  Pipeline       │
                        │  (Krea + LoRA)  │
                        └────────┬────────┘
                                 │
                        ┌────────▼────────┐
                        │  Styled Output  │
                        │  (WebRTC)       │
                        └─────────────────┘
```

### Open Questions

- [ ] How to ingest external video? (OBS virtual camera? NDI? RTMP?)
- [ ] VLM caption rate? (1 FPS? 4 FPS? On-demand?)
- [ ] Caption → prompt transformation (direct use? template?)
- [ ] Latency budget for real-time feel

### Dependencies

- VLM Frame Analysis (section 6) must be implemented first
- External video capture integration

---

## 8. NDI / Pub-Sub Video Output — READY TO IMPLEMENT

**What:** Add NDI (Network Device Interface) output for network-based video streaming with multiple subscribers.

**Status:** Ready to implement — straightforward addition using `ndi-python`.

**Proposal:** `notes/proposals/ndi-pubsub-video-output.md`

### Why NDI

- **Network streaming** — Stream from remote GPU to local TouchDesigner/OBS
- **Multi-consumer** — Multiple apps subscribe to same stream (unlike WebRTC queue)
- **Cross-platform** — Mac ↔ Windows ↔ Linux
- **Low latency** — 1-2 frames over LAN (designed for live production)
- **Native support** — TouchDesigner, OBS, Resolume, vMix all have NDI built-in

### Current Limitation

```
Pipeline → output_queue (destructive) → WebRTC → Single Browser
                ↓
          Spout (Windows, local only)
```

### Proposed Addition

```
Pipeline → output_queue → WebRTC
    ↓
    ├── Spout (local, Windows)
    └── NDI (network, cross-platform)  ← NEW
```

### Use Cases

1. **Remote GPU → Local TouchDesigner** — NDI over LAN
2. **V2V Secondary Pass** — Feed to another GPU running refinement model
3. **Multi-display** — Same stream to browser, TD, OBS simultaneously
4. **Tidal Integration** — TouchDesigner as audio-reactive effects layer

### Implementation Summary

1. Add `NDISender` class (similar to existing `SpoutSender`)
2. Add `NDIConfig` to schema
3. Clone frames to NDI queue in `_output_frame()`
4. Add REST/CLI controls

---

## 9. Tidal Cycles Integration — SPECULATIVE

**What:** Synchronize live-coded music (Tidal Cycles) with video generation by routing "music intent" from video cues to Tidal parameters.

**Status:** Speculative — requires Tidal setup, MCP server, and OSC bridge.

**Proposal:** `notes/proposals/tidal-cycles-integration.md`

**Research:** `notes/research/2025-12-26/tidal/aoi_5pro_01.md`

### The Concept

```
Video Cues → Music Intent → OSC Bridge → Tidal Cycles → Live Music
  (prompts)   (energy/tension)  (HTTP→OSC)   (patterns)    (SuperDirt)
```

The video system emits "cue + intent" events; the music system translates them into:
- **Continuous steering** via OSC `/ctrl` (fast, safe)
- **Arrangement toggles** via `/solo`, `/hush`, `/muteAll`
- **Pattern rewrites** via MCP `tidal_eval` (powerful, gated)

### Music Intent Vocabulary

| Control | Range | Maps To |
|---------|-------|---------|
| `energy` | 0–1 | Drum density, gain, distortion |
| `tension` | 0–1 | Filter modulation, dissonance, syncopation |
| `density` | 0–1 | Event frequency, layering |
| `space` | 0–1 | Reverb/delay send, tail length |
| `brightness` | 0–1 | Filter cutoff |
| `grit` | 0–1 | Distortion/saturation |
| `focus` | discrete | drums / bass / pads / full |

### Mapping to Video Transitions

| Video Event | Music Response |
|-------------|----------------|
| **Soft cut** (plasticity window) | Parameter morph + Tidal transition (`xfadeIn`, `clutchIn`) |
| **Hard cut** (cache reset) | `/hush` → new pattern bank |
| **Scene change** | Cue sheet lookup → intent update |

### Architecture Layers

1. **Phase 1: OSC Parameter Steering**
   - Tidal listens on `127.0.0.1:6010` for `/ctrl` messages
   - Video box sends intent via HTTP to local bridge
   - Bridge converts to OSC `/ctrl <key> <value>`

2. **Phase 2: Arrangement Controls**
   - Add `/solo`, `/hush`, `/muteAll` for scene boundaries
   - Map hard cuts to hush-then-start patterns

3. **Phase 3: Agent-Mediated Rewrites**
   - MCP server exposes `tidal_eval`, `tidal_get_state`, etc.
   - Claude Code proposes pattern changes
   - Human-in-the-loop approval before applying

### Existing Resources

| Resource | URL |
|----------|-----|
| TidalCycles MCP Server | https://github.com/Benedict/tidal-cycles-mcp-server |
| Tidal OSC Controller Input | https://tidalcycles.org/docs/working-with-patterns/Controller_Input/ |
| Tidal Transitions | https://tidalcycles.org/docs/reference/transitions/ |
| Open Stage Control (UI) | https://club.tidalcycles.org/t/open-stage-control-tidalcycles/1283 |

### Cue Sheet Format (Draft)

```json
{
  "cue_id": "akira_042",
  "prompt_id": "playlist_index_42",
  "transition": { "type": "soft", "cycles": 4 },
  "tags": ["neo_tokyo", "chase", "motorbikes"],
  "music_intent": {
    "energy": 0.85,
    "tension": 0.90,
    "space": 0.15,
    "brightness": 0.70,
    "density": 0.80,
    "grit": 0.55
  },
  "arrangement": {
    "focus": "drums+bass",
    "action": "unsoloAll"
  }
}
```

### Dependencies

- Tidal Cycles + SuperDirt installed on music machine
- OSC bridge (HTTP → OSC) — reference impl in research doc
- MCP server for agent integration (optional, Phase 3)

### Open Questions

- [ ] Network topology: same box or remote? (affects OSC routing)
- [ ] Sync mechanism: manual start or Ableton Link?
- [ ] Control surface for buddy: Open Stage Control or custom?

---

## Dependency Graph

```
Style Layer (Phase 6a) ─────────────────────────────────────────────┐
    │                                                               │
    ├── [independent] VACE-14B Integration                          │
    │                                                               │
    ├── [independent] Server-Side Session Recorder                  │
    │                                                               │
    ├── [independent] Style Swap Mode                               │
    │                                                               │
    ├── [independent] NDI / Pub-Sub Video Output ───────────────────┤
    │                        │                                      │
    │                        └──▶ V2V Secondary Pass                │
    │                                                               │
    ├── [independent] VLM Frame Analysis ───────────────────────────┤
    │                        │                                      │
    │                        └──▶ VLM-Mediated V2V                  │
    │                                  │                            │
    │                                  └── depends on: external     │
    │                                      video capture            │
    │                                                               │
    ├── [independent] Tidal Cycles Integration                      │
    │                        │                                      │
    │                        ├── depends on: Tidal + SuperDirt      │
    │                        └── depends on: OSC bridge             │
    │                                                               │
    └── [independent] Context Editing                               │
                         │                                          │
                         ├── depends on: nano-banana / image edit   │
                         └── depends on: VLM for agent loop ────────┘
```

**Ready now:** Style Layer (in progress), VACE-14B, Session Recorder, Style Swap, VLM Frame Analysis, NDI Output
**Speculative:** VLM-Mediated V2V, Context Editing, Tidal Cycles Integration

---

## Priority Recommendation

### Tier 1: In Progress
1. **Style Layer (Phase 6a)** — Let it complete

### Tier 2: Ready Now (pick based on need)
2. **Server-Side Session Recorder** — Enables offline re-rendering workflow
3. **Style Swap Mode** — Enables live style switching for performances
4. **VACE-14B Integration** — Reference image conditioning
5. **VLM Frame Analysis** — Foundation for agent loops + V2V
6. **NDI / Pub-Sub Video Output** — Network streaming to TouchDesigner/OBS

### Tier 3: Speculative (needs validation/dependencies)
7. **VLM-Mediated V2V** — Depends on VLM Frame Analysis + video capture
8. **Context Editing** — Run validation spike first
9. **Tidal Cycles Integration** — Depends on Tidal setup + OSC bridge

---

## Feature Requests

### Hard Cut Toggle for Playlist Navigation

**Requested:** 2025-12-25
**Status:** REST endpoint done, playlist integration pending

**What:** Add a "hard cut" mode to playlist navigation that resets the KV cache when switching prompts, causing a clean scene transition instead of morphing the current scene.

**Current behavior:** When navigating to a new prompt, the scene smoothly mutates from the current visual to match the new prompt (because the KV cache maintains continuity).

**Desired behavior:** Option to do a "hard cut" where the cache is reset, starting fresh with the new prompt instead of morphing.

**Completed:**
- [x] `POST /api/v1/realtime/hard-cut` endpoint (2025-12-25)
  - Resets KV cache for clean scene transition
  - Optional `prompt` parameter to set new prompt after reset
  - Returns `{"status": "hard_cut_applied", "chunk_index": N}`
- [x] Playlist navigation `?hard_cut=true` parameter (2025-12-25)
  - `POST /api/v1/realtime/playlist/next?hard_cut=true`
  - `POST /api/v1/realtime/playlist/prev?hard_cut=true`
  - `POST /api/v1/realtime/playlist/goto?hard_cut=true`
  - `POST /api/v1/realtime/playlist/apply?hard_cut=true`
- [x] Interactive nav mode `H` key toggle (2025-12-25)
  - `video-cli playlist nav` now supports `H` to toggle hard cut mode
  - When enabled, all transitions (manual and autoplay) do hard cuts
  - Display shows `[✂ HARD CUT]` indicator

**Remaining:**
- [ ] Add `--hard-cut` flag to individual `video-cli playlist` commands
- [ ] Sequence format support (hard cut markers in YAML/JSON)

**GUI reference:** The "Manage Cache" toggle and "Reset Cache" button already exist in the web UI.

---

## Related Performance Work

See `notes/FA4/b300/optimization-vision.md` for performance optimization roadmap:
- Current: 15 FPS @ 320×576 with FA4
- Target: 24+ FPS
- Blocked: torch.compile, recompute cadence
- Next: cuDNN benchmark, SageAttention, torchao fix
