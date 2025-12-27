# Session Handoff - 2025-12-27

> **Purpose:** Context preservation before compaction
> **Branch:** `feature/stream-recording`
> **Latest commit:** `efbeeb5`

---

## What Was Done This Session

### 1. Documentation Consolidation

- **Created `notes/VISION.md`** - Internal living document (vs external cohort pitch)
- **Created `notes/concepts/narrative-engine.md`** - Phase 2 concepts (world state, trajectories, information topology)
- **Created `notes/concepts/prompt-engineering-workflow.md`** - Loom-like visual behavior R&D
- **Updated `notes/NOTES-INDEX.md`** - Added Concepts and Proposals sections
- **Updated `notes/daydream/cohort-pitch.md`** - Experience-first framing

### 2. Clickable Links in FA4 Docs

- Made `notes/` references clickable across all FA4 documentation (46 files total)
- Pattern: `` `notes/path/file.md` `` → `[file.md](relative/path)`
- Commits: `f18e183`, `2e0a92b`

### 3. New Proposals

- **Created `notes/proposals/multi-gpu-scaling.md`** - Exploratory proposal for pipeline parallelism
  - Based on StreamDiffusionV2 research (58 FPS with 14B on 4×H100)
  - Covers VAE offload, pipeline parallel, temporal, DistriFusion approaches
  - Updated with B300 baseline data: ~22-23 FPS with compile, decode ~35% of time

### 4. Restructured VACE Proposal

- Moved `notes/vace-14b-integration/plan.md` → `notes/proposals/vace-14b-integration.md`
- Kept supporting materials in `notes/proposals/vace-14b-integration/`
- Aligned with other proposals pattern (top-level .md + subdirectory for supporting materials)
- Commit: `efbeeb5`

---

## Codex Progress (B300 Optimization)

### From Codex1 (earlier)
- Disabled fused QKV projections on B300 (`SCOPE_DISABLE_FUSED_PROJECTIONS=1`)
- FPS: 18.09 → 18.53, self_attn: 1.25 → 1.18 ms/call
- Added layout contract dumping (`SCOPE_LAYOUT_CONTRACT_OUT`)

### From Codex (latest - commit 04df73c)
- Added `WANVAE_DECODE_CHANNELS_LAST_3D=1` for VAE decode
- Decode: ~199.9ms → ~195.1ms (-2.4%), FPS: 21.28 → 21.50
- Added guard against `max-autotune*` modes on SM103 with Triton <3.5.1
- **Next suggested:** Upgrade Triton to 3.5.1 in new venv, retry max-autotune

### Codex Recommended Triton Upgrade Plan
```bash
# Create new env (acts as checkpoint)
./scripts/setup_b300_cu130_env.sh .venv-b300-cu130-triton351

# Upgrade Triton
uv pip install -p .venv-b300-cu130-triton351/bin/python --upgrade triton==3.5.1

# Run with new env
B300_ENV_DIR=.venv-b300-cu130-triton351 ./scripts/run_daydream_b300.sh

# Revert is instant: switch back to B300_ENV_DIR=.venv-b300-cu130-decode
```

---

## Current B300 Performance State

| Config | FPS | Notes |
|--------|-----|-------|
| BF16, no compile | ~19-20 | Stable baseline |
| BF16, `--compile` | ~22-23 | Default recommended |
| BF16, `--compile`, channels_last_3d | ~21.5 | Latest |

Block breakdown (compiled):
- **denoise**: ~65%
- **decode**: ~35%

**FP8 is broken** on B300 - produces garbage output. Use `--quantization none` (BF16).

---

## Key Documents

| Purpose | Location |
|---------|----------|
| Internal vision | `notes/VISION.md` |
| B300 session state | `notes/FA4/b300/session-state.md` |
| B300 development plan | `notes/FA4/b300/development-plan.md` |
| Optimization ladder | `notes/FA4/b300/optimization-ladder.md` |
| Proposals index | `notes/NOTES-INDEX.md` (Section 3) |
| Multi-GPU proposal | `notes/proposals/multi-gpu-scaling.md` |
| VACE-14B proposal | `notes/proposals/vace-14b-integration.md` |

---

## Pending / Next Steps

1. **Triton 3.5.1 upgrade** - Codex has a plan ready, would unlock max-autotune modes
2. **VAE decode deep dive** - Conv3d layout/contracts, cuDNN graph capture
3. **Multi-GPU investigation** - Start with VAE offload (simplest approach)

---

## Git State

```
Branch: feature/stream-recording
Latest: efbeeb5 - refactor(proposals): restructure vace-14b-integration
Up to date with origin
```

Untracked outputs from Codex Triton 3.5.1 experiments exist in `outputs/b300_cu130_triton351_*`.
