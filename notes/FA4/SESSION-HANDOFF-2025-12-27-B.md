# Session Handoff - 2025-12-27 (Part B)

> **Purpose:** Context preservation before compaction
> **Branch:** `feature/stream-recording`
> **Latest commit:** `b0e069e`

---

## What Was Done This Session

### 1. NDI Pub-Sub Proposal Update

- Read feedback from `notes/proposals/ndi-pubsub-video-output/oai_5pro01.md`
- Updated proposal with architecture review findings:
  - Added "Two Implementation Paths" (MVP vs full pub-sub)
  - Added prerequisite: decouple Spout from WebRTC (move to producer-side)
  - Updated NDI sender code with RGB→BGRX conversion in IO thread
  - Added REST strategy decision (dedicated endpoints vs generic `/parameters`)
  - Updated implementation phases (Phase 0 = Spout prerequisite)
- Commit: `8728fb1`

### 2. Cohort Pitch Update

- Updated `notes/daydream/cohort-pitch.md` with new performance numbers:
  - 2.5x → **3.3x** (later updated to 3.5x after FA4 varlen)
  - FPS: ~22-23 → **~29** (benchmark)
  - Added VAE decode line: ~195ms → ~60ms
- Commit: `5176e1c`

### 3. Style Swap Mode Implementation

- Drafted core helpers in `src/scope/realtime/style_manifest.py`:
  - `get_style_dirs()` - multi-dir style discovery
  - `canonicalize_lora_path()` - resolve bare filenames to absolute paths
  - `StyleRegistry.load_from_style_dirs()` - load from all configured dirs
  - `StyleRegistry.get_all_lora_paths()` - unique canonical paths
  - `StyleRegistry.build_lora_scales_for_style()` - build lora_scales update

- Codex wired up the remaining pieces (commit `3de726c`):
  - `PipelineManager`: `STYLE_SWAP_MODE=1` preloads LoRAs, forces `runtime_peft`
  - `FrameProcessor`: uses `load_from_style_dirs()`, emits canonical `lora_scales`
  - `prompt_compiler`: instruction sheet lookup follows style dirs
  - Proposal doc updated with implementation status

- **Style swap is ready to test:**
  ```bash
  STYLE_SWAP_MODE=1 ./scripts/run_daydream_b300.sh
  # Then: video-cli style set rat
  ```

### 4. B300 Documentation Update

- Updated stale numbers across B300 docs:
  - `session-state.md`: 3.3× → **3.5×**, new block profile table
  - `optimization-ladder.md`: ~22-23 FPS → **~30.7-30.9 FPS**
  - `README.md`: added current best to ground truth
  - Marked "Two Codex Agents" work as completed/historical
- Commit: `b0e069e`

### 5. Reviewed Cohort Project Pages

- Explored `notes/daydream/cohort_docs/` to see what others posted
- Format varies: minimal (thumbnail + paragraph) to detailed (tech architecture)
- User considering waiting to post until style switching demo is ready

---

## Current Performance State (B300)

| Metric | Value |
|--------|-------|
| Best FPS (benchmark) | ~30.7-30.9 |
| Speedup vs baseline | **3.5×** (8.8 → 30.7) |
| With style swap (~50% hit) | ~15 FPS (estimated) |

Block breakdown (current best config):
- **denoise**: ~267 ms/call (~69%)
- **recompute_kv_cache**: ~62 ms/call (~16%)
- **decode**: ~60 ms/call (~15%) — **solved**

---

## Key Documents

| Purpose | Location |
|---------|----------|
| B300 session state | `notes/FA4/b300/session-state.md` |
| Style swap proposal | `notes/proposals/style-swap-mode.md` |
| NDI proposal | `notes/proposals/ndi-pubsub-video-output.md` |
| Cohort pitch | `notes/daydream/cohort-pitch.md` |
| Optimization ladder | `notes/FA4/b300/optimization-ladder.md` |

---

## Style Swap Status

**Implementation:** ✅ Complete (commit `3de726c`)

| Feature | Status |
|---------|--------|
| Multi-dir style discovery | ✅ |
| Canonical LoRA path normalization | ✅ |
| `STYLE_SWAP_MODE=1` preload + runtime_peft | ✅ |
| Instruction sheet follows style dirs | ✅ |
| `STYLE_DEFAULT` initial activation | ❌ Not yet |

**Expected performance:** ~15 FPS with style swap enabled (50% hit from `runtime_peft`)

---

## Pending / Next Steps

1. **Test style swap** — Run with `STYLE_SWAP_MODE=1`, verify switching works
2. **Measure style swap perf** — Confirm ~50% hit, update docs if different
3. **Cohort project page** — Post once style switching demo is ready
4. **Push commits** — Currently 1 commit ahead of origin

---

## Git State

```
Branch: feature/stream-recording
Latest: b0e069e - docs(b300): update to 3.5x baseline (~30.7 FPS)
Ahead of origin by 1 commit
```

Uncommitted:
- `scripts/run_daydream_b300.sh` (Codex changes)
- Various `outputs/` logs
- `notes/proposals/vace-14b-integration.md` (minor edit)
