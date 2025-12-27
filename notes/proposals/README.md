# Proposals Index (Readiness + Implementation)

> **Purpose:** Track proposal readiness vs actual repo implementation
> **Updated:** 2025-12-27

## Legend

- **Repo status:** Implemented / Partial / Not started / Doc-only
- **Readiness:** High (start coding now) / Med (some decisions/deps) / Low (research first)
- **Effort:** S/M/L (relative)
- **Risk:** Low/Med/High (unknowns + external deps + perf/VRAM sensitivity)

## Grid

| Proposal | Doc status | Repo status | Readiness | Effort | Risk | Top blocker / next step |
|---|---|---|---|---|---|---|
| [frame-buffer-scrubbing.md](frame-buffer-scrubbing.md) | Draft | Not started | Med | M | Med | Decide ring-buffer budget + add output history + scrub endpoints (session scoping). |
| [gemini-cookbook.md](gemini-cookbook.md) | Reference | Doc-only | N/A | N/A | Low | Use as pattern library (Gemini flows already exist elsewhere in repo). |
| [hardware-control-surface.md](hardware-control-surface.md) | Draft (vetted) | Not started | Med | M | Med | Build “hardware router” process + decide how to target the correct active session. |
| [multi-gpu-scaling.md](multi-gpu-scaling.md) | Exploratory | Not started | Low | L | High | Profile + pick a concrete parallelism strategy before any code churn. |
| [ndi-pubsub-video-output.md](ndi-pubsub-video-output.md) | Draft | Not started | Med | M | Med/High | Choose NDI binding + move output fan-out to `process_chunk()` (producer-side) + add NDI sender thread. |
| [realtime-voice-integration.md](realtime-voice-integration.md) | Draft | Not started | Med | M | Med/High | Pick Phase 1 path (browser STT vs Whisper) and define the control contract. |
| [server-side-session-recorder.md](server-side-session-recorder.md) | Draft (doc lags) | Implemented (MVP) | High | S/M | Med | Expand recorded events (e.g., LoRA scales) + harden multi-session behavior. |
| [session-recording-timeline-export.md](session-recording-timeline-export.md) | Implemented (MVP) | Implemented (MVP) | High | S | Low | Align timeline semantics with server recorder (cuts/transitions fidelity). |
| [style-swap-mode.md](style-swap-mode.md) | Draft (reviewed) | Partial (core mode exists) | High | S | Med | Add `STYLE_DEFAULT` initial activation + tighten “style exists” validation. |
| [tidal-cycles-integration.md](tidal-cycles-integration.md) | Draft | Not started | Med | M | Med/High | Define minimal intent schema + implement the bridge (OSC/MCP) for one happy path. |
| [transition-prompts.md](transition-prompts.md) | Draft | Not started | High | S | Low | Implement `>` parsing + thread transition text into existing transition machinery. |
| [vace-14b-integration.md](vace-14b-integration.md) | Ready to implement | Not started | Med | L | High | Add artifacts + schema + Krea mixin/block wiring; decide VRAM/quant + KV-recompute semantics. |
| [vlm-integration.md](vlm-integration.md) | Draft | Partial (prompt compilation exists; frame analysis NOT started) | Med | M | Med/High | Add frame analysis + image editing plumbing (provider deps + endpoints + UI wiring). |
| Context Editing | Roadmap-only | Not started | Low | M | High | Run validation spike first (tint anchor frame, see if it propagates). See `capability-roadmap.md` §3. |
| VLM-Mediated V2V | Roadmap-only | Not started | Low | L | High | Depends on VLM Frame Analysis + external video capture. See `capability-roadmap.md` §7. |

## Evidence pointers (quick)

- Server-side session recorder wiring: `src/scope/server/app.py`, `src/scope/server/frame_processor.py`, `src/scope/server/session_recorder.py`, `src/scope/cli/render_timeline.py`
- Timeline export on record stop: `frontend/src/pages/StreamPage.tsx`, `frontend/src/hooks/useStreamRecorder.ts`
- Style swap mode: `src/scope/server/pipeline_manager.py`, `src/scope/server/frame_processor.py`
- Prompt compilation (VLM): `src/scope/realtime/gemini_client.py`

## Cross-cutting blockers

- **Session scoping:** many REST endpoints assume a single "active session"; multi-session requires explicit routing. Affects: hardware-control-surface, style endpoints, frame-buffer-scrubbing.
- **Generic parameters:** several proposals want "set arbitrary knobs" without going through a browser data-channel. Affects: hardware-control-surface, NDI.
- **External deps:** NDI + voice + VLM + VACE introduce packaging/keys/weight downloads and perf/VRAM risks.

## Alignment note

This grid tracks implementation status. For detailed specs, checklists, and architecture, see `notes/capability-roadmap.md`.
