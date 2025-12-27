# FA4 Optimization Map (Timeline, Truth Sources, Workspace)

> **Purpose:** A short “you are here” map for the FA4 / realtime-video optimization work: what’s been done, what’s currently true, where to record new learnings, and where the relevant resources live.
> **Updated:** 2025-12-26

---

## Current Status (as of Dec 25)

**Canonical perf comparisons use:** `320x576`, 4 denoise steps, KV-bias `0.3`, quality-preserving settings.

- **B200 (SM100):** ~`20 FPS` at `320x576` (canonical setting).
- **B300 (SM103):**
  - Repo-default stack: ~`8.8 FPS` at `320x576` (decode/cuDNN-bound).
  - cu130 stack: ~`15 FPS` typical end-to-end, with higher best-case when `torch.compile` is usable.

Source of truth for “what to run right now”: [`session-state.md`](b300/session-state.md).

---

## What Goes Where (so we don’t duplicate)

**Protocol vs state vs notebook vs bibliography (per GPU):**

- **B200 session state (what to run today):** [`session-state.md`](b200/session-state.md)
- **B200 development plan (what to build next):** [`development-plan.md`](b200/development-plan.md)
- **B200 experiments log:** [`experiments.md`](b200/experiments.md)
- **B300 session state (what to run today):** [`session-state.md`](b300/session-state.md)
- **B300 development plan (what to build next):** [`development-plan.md`](b300/development-plan.md)
- **B300 runbook (how to measure / decide):** [`investigation-runbook.md`](b300/investigation-runbook.md)
- **B300 experiments log:** [`experiments.md`](b300/experiments.md)
- **B300 external docs brief:** [`blackwell-docs.md`](b300/blackwell-docs.md)
- **B300 narrative findings:** [`investigation.md`](b300/investigation.md)

**Broader FA4 work:**
- **Entry point + directory map:** [`README.md`](README.md)
- **Shareable writeup (profiling → wins):** [`kernel-optimization-guide.md`](docs/kernel-optimization-guide.md)
- **Long-form chronological log (apprentice track):** [`kernel-dev-log.md`](kernel-dev-log.md)
- **FA4/CuTe internals explainers (learning track):** [`README.md`](explainers/README.md)

---

## Timeline (High Level)

This is intentionally “milestones only.” The detailed blow-by-blow is [`kernel-dev-log.md`](kernel-dev-log.md).

| Date | Milestone | Where it lives |
|------|-----------|----------------|
| 2025-12-22–23 | Added profiling hooks (block + attention splits) to stop guessing | [`kernel-dev-log.md`](kernel-dev-log.md) + [`kernel-optimization-guide.md`](docs/kernel-optimization-guide.md) |
| 2025-12-23 | Triton KV-bias kernel exploration (learning + microbenching) | `scripts/triton_sdpa.py`, [`kernel-dev-log.md`](kernel-dev-log.md) |
| 2025-12-23–24 | RoPE cleanup/fusion work (Triton RoPE) | `notes/FA4/rope/*` |
| 2025-12-24 | B300 “8.8 FPS mystery” framed + systematic runbook | [`investigation-runbook.md`](b300/investigation-runbook.md) |
| 2025-12-24–25 | Key B300 insight: runtime stack dominates decode; cu130 fixes VAE decode | [`session-state.md`](b300/session-state.md) |
| 2025-12-25 | FA4/CuTe `score_mod` KV-bias stabilized for B300; backend guardrails documented | [`session-state.md`](b300/session-state.md), [`optimization-vision.md`](b300/optimization-vision.md) |
| 2025-12-26 | Collected source-backed upstream refs (TorchAO FP8+compile, Conv3d/cuDNN, CUDAGraph step markers) | [`session-state.md`](b300/session-state.md), [`blackwell-docs.md`](b300/blackwell-docs.md), [`claude_dr.md`](DeepResearch/2025-12-26/B300_optim_ladder/round02/claude_dr.md) |

---

## Workspace Map (Code + Artifacts)

- **Benchmark scripts:** `scripts/profile_krea_pipeline_blocks.py`, `scripts/profile_b300_denoise_drilldown.sh`
- **Attention kernel dev harness:** `scripts/triton_sdpa.py`
- **Primary pipeline codepaths:** `src/scope/core/pipelines/krea_realtime_video/` and `src/scope/core/pipelines/wan2_1/`
- **Vendored FA4/CuTe reference (for score_mod work):** `vendored/flash_attn_cute_score_mod/`
- **Profiling outputs (JSON/logs):** `outputs/` (treat as ephemeral; promote conclusions into notes)

---

## How We Measure (Quick Reference)

**End-to-end FPS (canonical):**
```bash
uv run python scripts/profile_krea_pipeline_blocks.py \
  --height 320 --width 576 \
  --iters 6 --skip 2 \
  --kv-cache-attention-bias 0.3
```

**Deeper drilldown (writes JSON under `outputs/`):**
```bash
scripts/profile_b300_denoise_drilldown.sh
```

**Rule:** if a profiler adds synchronizations, treat the output as “breakdown truth,” not peak FPS.

---

## Cross-GPU Experiment Checklist (the “do this every time” part)

When you run an experiment (one change), record a card in the appropriate experiments log:
- B200: [`experiments.md`](b200/experiments.md)
- B300: [`experiments.md`](b300/experiments.md)

**Always capture:**
- GPU + environment (B200/B300, which venv/stack, torch + cuda versions)
- Key knobs (`height/width`, steps, bias, backend env vars, quantization, compile on/off)
- Exact command(s) used
- Baseline you’re comparing against
- Result (FPS + any profiler deltas) + a short “lesson”
- Artifact filenames under `outputs/` (if any)

**Promotion rules (keep docs clean):**
- If an experiment becomes “the thing we should do by default,” summarize it in the GPU’s `session-state.md`.
- If it changes *how we measure/decide*, update [`investigation-runbook.md`](b300/investigation-runbook.md).
- If it’s mainly external justification, add it to [`blackwell-docs.md`](b300/blackwell-docs.md).

---

## “If I’m Starting Today…”

1. Read [`README.md`](README.md) (map of the area).
2. Read [`kernel-optimization-guide.md`](docs/kernel-optimization-guide.md) (the story + why these levers matter).
3. If you’re on B300, follow [`session-state.md`](b300/session-state.md) exactly once to get a stable baseline.
4. When you try a change: record it as a card in the appropriate experiments log ([`experiments.md`](b200/experiments.md) or [`experiments.md`](b300/experiments.md)).
