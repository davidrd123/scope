# Repo Agent Notes

These are guardrails for coding agents working in this repo.

## Collaboration Safety

- Do not run destructive git commands (`git restore`, `git reset`, `git checkout --`, `git clean`) on files outside the current task scope without asking first.
- Do not “clean up” or revert unrelated working-tree changes; instead, scope your edits to the files required for the task and call out unexpected diffs.
- Avoid editing dependency/lock files (`pyproject.toml`, `uv.lock`) unless the user explicitly asks.
- Avoid touching shared virtualenvs (e.g. `.venv`) unless the user explicitly asks; prefer isolated envs for experiments.
- If external documentation (blogs/specs/release notes) would materially speed up or de-risk a task, pause and ask the user to fetch/curate it before proceeding; be explicit about exactly what to collect and in what format (links, excerpts, PDFs, etc.).

## B300 Perf Work

- Canonical perf comparisons use `320x576` resolution and stable, quality-preserving settings (no KV-cache recompute skipping).
- Prefer the B300 scripts under `scripts/` (e.g. `scripts/run_daydream_b300.sh`) so toolchain/env settings don’t collide with other work.

## Orientation (New Agents)

- Cohort context / what others shared: `notes/ecosystem.md`, `notes/daydream/interactive-ai-video-program.md`
- Realtime pipeline architecture: `src/scope/core/pipelines/krea_realtime_video/docs/architecture-guide.md`
- FA4/B200/B300 perf thread entrypoint: `notes/FA4/b300/README.md`
  - Ground truth configs + known issues: `notes/FA4/b300/session-state.md`
  - How we measure: `notes/FA4/b300/investigation-runbook.md`
  - One-change experiment log: `notes/FA4/b300/experiments.md`
- Backend selection / knobs: `notes/FA4/explainers/17-backend-selection-and-knobs.md`
- KV-bias attention codepaths:
  - Bias enabled (`kv_cache_attention_bias < 1.0`): `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py`
  - Bias disabled/plain attention: `src/scope/core/pipelines/wan2_1/modules/attention.py`
- TorchAO FP8 + compile `aten.as_strided.default` gap:
  - Paste-ready upstream issue: `notes/issues/torchao-as-strided-dispatch.md`
  - Local PerTensor-only monkeypatch (experiments): `scripts/patch_float8_as_strided.py`

## Doc Upkeep

- When adding new knobs/benchmarks, update `notes/FA4/b300/session-state.md` and `notes/FA4/b300/README.md` so new agents can orient quickly.
- Prefer “one-change” experiment cards in `notes/FA4/b300/experiments.md` over long narrative dumps.
