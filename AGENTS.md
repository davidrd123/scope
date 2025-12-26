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
