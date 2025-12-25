# 2025-12-24 Research Index (Incoming → Distilled → Integrated)

This index is the “second pass” map from raw drops in `incoming/` to the smaller set of notes/docs we actually build from.

## Canonical Docs (Used While Building)

- `notes/realtime_video_architecture.md`
- `notes/realtime-roadmap.md`
- `notes/realtime-devlog.md`
- `notes/plans/realtime-control-plane-tdd.md`
- `notes/reference/context-editing-code.md`
- `notes/reference/cli-implementation.md`

## Incoming (Raw)

| Source | Axis | Status | Distilled / Integrated Into |
|---|---|---:|---|
| `notes/research/2025-12-24/incoming/project_knowledge.md` | context editing + UI + perf | DISTILLED | `notes/realtime-roadmap.md`, `notes/realtime-devlog.md`, `notes/reference/context-editing-code.md` |
| `notes/research/2025-12-24/incoming/context_editing_and_console_spec.md` | context editing + dev console | DISTILLED | `notes/realtime-roadmap.md`, `notes/realtime-devlog.md`, `notes/reference/context-editing-code.md` |
| `notes/research/2025-12-24/incoming/interface_specifications.md` | CLI/Web/TUI | DISTILLED | `notes/realtime-roadmap.md`, `notes/realtime-devlog.md`, `notes/reference/cli-implementation.md` |
| `notes/research/2025-12-24/incoming/style/` | style layer | DISTILLED (partial) | `notes/research/2025-12-24/style-layer-revision-notes.md`, `src/scope/realtime/*` (style scaffolding) |
| `notes/research/2025-12-24/incoming/perf/` | perf | TRIAGED | Use as reference; promote only actionable conclusions into `notes/FA4/` or `notes/TODO-next-session.md` |

## Distilled Notes (This Date)

- `notes/research/2025-12-24/style-layer-revision-notes.md` (extracts universal prompting patterns; informs InstructionSheet authoring)
- `notes/research/2025-12-24/rcp-v1.2-feedback.md` (architecture alignment critique; informs integration approach)

## Next Distillation Actions

- Convert at least one real style packet (e.g. RAT or Kaiju) into a runnable style pack under `styles/<name>/` (manifest + instruction sheet).
- Decide how to reconcile REST-centric interface specs with the repo’s current WebRTC-first control surface (document a mapping, don’t implement both at once).
- Promote any “hard” context-edit invariants (buffer keys, edit surface, pause requirements) into `notes/realtime_video_architecture.md` so the mechanism is part of the canonical spec.

