# Research Intake → Distillation Workflow

This repo uses a simple pipeline for handling “raw dumps” of research/specs and turning them into artifacts we actually build from.

## Directory Conventions

- **Incoming (raw dumps)**: `notes/research/YYYY-MM-DD/incoming/`
  - Unedited exports, chat transcripts, long specs, style packets, perf notes.
  - Treat as *append-only*; don’t try to keep these “clean”.
- **Distilled (usable summaries)**: `notes/research/YYYY-MM-DD/`
  - Short, human-scannable summaries that extract decisions, invariants, and next actions.
  - Example: `notes/research/2025-12-24/style-layer-revision-notes.md`
- **Canonical realtime docs** (what we reference while building):
  - `notes/realtime_video_architecture.md` (deep spec)
  - `notes/realtime-roadmap.md` (what we’re doing next + why)
  - `notes/realtime-devlog.md` (what changed + when)
- **Stable “reference” extracts** (code-ish snippets / runbooks):
  - `notes/reference/` (keep these short and directly actionable)
- **Style packs** (assets we run with):
  - `styles/<style_name>/manifest.yaml`
  - `styles/<style_name>/instructions.md`

## Status Labels (Use In Index Tables)

- `UNREAD`: captured but not reviewed
- `TRIAGED`: skimmed; classified; owner + next action identified
- `DISTILLED`: a usable summary exists
- `INTEGRATED`: decisions landed in canonical docs and/or code/tests
- `DEFERRED`: not needed for current milestone
- `OBSOLETE`: replaced by newer source

## Distillation Checklist (Per Incoming Doc)

When converting an incoming doc into something we can build from, capture:

1. **Scope**: what feature axis does it support? (control plane / style / context editing / perf / UI)
2. **Hard invariants**: facts about the existing repo/runtime that must be respected
3. **Contracts**: message formats, state keys, API surfaces
4. **Key decisions**: what we decided, and what we explicitly deferred
5. **Open questions / risks**: multi-session, pause semantics, input readiness, etc.
6. **Action items**: concrete next steps with file pointers
7. **Where integrated**: links to code/tests/docs that reflect the decisions

## Template

Use `notes/research/_distillation_template.md` when creating a new distillation note.

