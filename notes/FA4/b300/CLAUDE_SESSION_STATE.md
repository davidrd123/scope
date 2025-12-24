# Claude Session State — B300 Investigation (2025-12-24)

> **Purpose:** Handoff document for context compaction. Resume from here.

---

## Current Status

**B300 @ 320×576:** ~14.8–15.0 FPS (up from 8.8 baseline = +70%)

**Branch:** `feature/stream-recording`
**Latest commit:** `1a462a2` (pushed)

---

## What Was Accomplished This Session

### 1. B300 Performance Investigation Complete
- Identified root cause: **cuDNN lacking SM103 kernels** in cu128/cu129 stack
- VAE decode (conv3d) was bottleneck, not attention
- cu130 stack provides 4x decode speedup (760ms → 194ms)

### 2. Environment Tooling Created
```bash
./scripts/setup_b300_cu130_env.sh    # One-time setup
./scripts/b300_env_fix_cu130.sh      # Fix after uv sync clobbers
./scripts/run_daydream_b300.sh       # Run with correct env vars
```

### 3. Profiling Infrastructure Added
- `scripts/profile_b300_denoise_drilldown.sh`
- CUDA event profilers in denoise.py, generator.py, wan.py
- Block-level timing: denoise 62%, decode 25%, recompute_kv 12%

### 4. Documentation Updated
- `notes/FA4/b300/session-state.md` — 15 FPS result
- `notes/FA4/b300/investigation-runbook.md` — full findings
- `notes/research/2025-12-24/incoming/perf/ACTIONABLE_ITEMS_SUMMARY.md` — LLM recommendations cross-checked

### 5. Commits Pushed Today
```
1a462a2 Add perf research chat exports with disclaimers and actionable summary
717bf50 Add B300 benchmark artifacts and research notes
d9ea647 Add B300 denoise drilldown profiling and benchmark artifacts
3cd10c6 Add style layer scaffolding (Days 5-7)
55a01c1 Add snapshot/restore and step mode to realtime control plane
e0dff46 Add B300 profiling infrastructure and update docs with 15 FPS result
d8284d9 Add B300/cu130 support: env scripts, VAE profiling, flash backend fallback
```

---

## Key Files to Know

| File | Purpose |
|------|---------|
| `notes/FA4/b300/session-state.md` | Current B300 status + repro commands |
| `notes/FA4/b300/investigation-runbook.md` | Full investigation methodology |
| `.venv-b300-cu130-decode/` | Isolated cu130 env (torch 2.9 + flash-attn) |
| `scripts/run_daydream_b300.sh` | Runs daydream with B300 env vars |

---

## Remaining Work (Not Urgent)

1. **RoPE chunk padding** — `triton_rope_fused.py` still has C*_PAD; chat suggested fixing this but attention isn't the bottleneck
2. **Nsight profiling** — blocked by CUPTI errors on this box
3. **Gap to 20 FPS** — denoise is now 62% of time; may need further cu130 stack optimization

---

## Environment Notes

```bash
# B300 requires these env vars
export TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas
export DISABLE_FLEX_ATTENTION_COMPILE=1  # torch 2.9 + SM103
export WANVAE_STREAM_DECODE_MODE=chunk

# Run with cu130 env
./scripts/run_daydream_b300.sh
```

---

## Quick Resume Commands

```bash
# Check current status
cat notes/FA4/b300/session-state.md | head -50

# Run B300 benchmark
./scripts/run_daydream_b300.sh

# View recent commits
git log --oneline -10
```
