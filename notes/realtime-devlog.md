# Realtime Control Plane - Development Log

Chronological record of implementation work and decisions.

---

## 2025-12-25

### Phase 5 Planning: REST Endpoint Architecture

Analyzed how to add REST endpoints for CLI control. Key decision: **hook into existing WebRTC control path** rather than creating parallel control surface.

**Architecture:**
- REST endpoints find active WebRTC session via `WebRTCManager.sessions`
- Call `frame_processor.update_parameters()` (same code path as data channel)
- Browser stays open as "execution backend" / monitor
- No headless mode for hackathon (real-time interactivity needs visual feedback)

**GPT-5 Pro Review** (`notes/research/2025-12-24/incoming/rest_endpoint/5pro_rest_feedback.md`):

Identified 6 issues to fix before implementing REST:

1. **Pause must hit two mechanisms** - `VideoProcessingTrack.pause()` (freezes playback) AND `update_parameters({"paused": True})` (stops generation)
2. **Lazy FrameProcessor init** - REST must call `initialize_output_processing()` before touching `frame_processor`
3. **Queue drops updates** - Bounded queue drops newest on full; should drop oldest (mailbox semantics)
4. **Output queue resize race** - Pause-flush can race with queue resize; need lock
5. **`/api/frame/latest` consumes** - Calling `get()` removes from queue; need non-destructive buffer
6. **Duplicate control logic** - Extract shared `apply_control_message()` for both WebRTC and REST

**Action:** Fix these issues first, then add REST endpoints.

### Phase 5 Implementation Complete

Implemented all fixes and REST endpoints:

**Fixes applied:**
1. `apply_control_message()` helper in `webrtc.py` - shared by WebRTC data channel and REST
2. `get_active_session()` helper - finds single connected session for REST
3. Mailbox queue semantics in `frame_processor.py` - drops oldest on full, not newest
4. `output_queue_lock` + `flush_output_queue()` - prevents resize/flush race
5. `latest_frame_cpu` buffer + `get_latest_frame()` - non-destructive frame reads

**REST endpoints added** (`/api/v1/realtime/`):
- `GET /state` - paused, chunk_index, prompt, session_id
- `POST /pause` - pause generation + playback
- `POST /run` - resume (or `?chunks=N` for N step chunks)
- `POST /step` - generate one chunk while paused
- `PUT /prompt` - set prompt text (JSON body)
- `GET /frame/latest` - PNG image of latest frame

**CLI added** (`video-cli`):
- `video-cli state` - get current state (JSON)
- `video-cli pause` - pause generation
- `video-cli run [--chunks N]` - resume or run N chunks
- `video-cli step` - generate one chunk
- `video-cli prompt "text"` - set prompt
- `video-cli frame --out path.png` - save current frame

Entry point: `pyproject.toml` → `video-cli = "scope.cli.video_cli:main"`

---

## 2025-12-24 (Afternoon)

### Integrated Incoming Specs

Processed and distilled three incoming spec documents:
- `notes/research/2025-12-24/incoming/project_knowledge.md` - Project knowledge base
- `notes/research/2025-12-24/incoming/context_editing_and_console_spec.md` - Context editing mechanism
- `notes/research/2025-12-24/incoming/interface_specifications.md` - CLI/Web UI specs

**Key integration:**
- Updated `notes/realtime-roadmap.md` with two feature axes (prompt compilation + context editing)
- Created `notes/reference/context-editing-code.md` - Gemini integration, edit API, validation test
- Created `notes/reference/cli-implementation.md` - Full CLI implementation with Click
 - Added distillation workflow + index: `notes/research/PROCESS.md`, `notes/research/2025-12-24/INDEX.md`

**Two feature axes clarified:**
1. **Prompt Compilation**: Control what we SAY (WorldState → LLM → prompt)
2. **Context Editing**: Control what model SEES (edit anchor frame → KV cache propagation)

**The mechanism**: Edit `decoded_frame_buffer[:, :1]` → re-encoded during KV cache recompute → future frames "remember" the edit.

**Phases updated:**
- Phase 5: CLI + Core API (next)
- Phase 6: Context Editing
- Phase 7: Interactive UI
- Phase 8: VLM Feedback Loop

---

## 2025-12-24

### Style Layer Review Complete

Reviewed 3 style slices to understand prompt compilation patterns:
- `notes/research/2025-12-24/incoming/style/RAT/` - Rooster & Terry (claymation comedy)
- `notes/research/2025-12-24/incoming/style/Kaiju/` - Japanese tokusatsu films
- `notes/research/2025-12-24/incoming/style/TMNT/` - Graffiti sketchbook animation

**Finding**: 7 universal patterns across all domains:
1. Trigger phrase
2. 3-tier anchor hierarchy (essential/common/situational)
3. First mention pattern: `[descriptor], [Name]`
4. 5-step structural flow (ESTABLISH → STYLE → ACTION → LIGHTING → CONCLUDE)
5. 3 length tiers (quick/standard/complex)
6. "Keep grammar, change nouns" off-road guidance
7. OOD/attention budget concept

**Decision**: InstructionSheet carries the weight. StyleManifest stays minimal (trigger, lora_path, instruction_sheet_path). All domain knowledge lives in prose instruction sheets that become LLM system prompts.

**Implication**: Current scaffolding is mostly right. No schema changes needed. The 7 patterns become a checklist for writing good instruction sheets.

See: `notes/research/2025-12-24/style-layer-revision-notes.md`

### Doc Refactor

Created cleaner doc structure:
- `notes/realtime-roadmap.md` - High-level plan (this is new)
- `notes/realtime-devlog.md` - Development log (this file)
- `notes/realtime_video_architecture.md` - Keep as detailed technical spec

### Next: Interactive Prompt UI

Sketched Phase 5 workflow:
1. Freeform text input ("what should happen?")
2. Style selector (picks trigger + instruction sheet)
3. Compile button (LLM compiles to full prompt)
4. Show compiled prompt (editable)
5. Send to pipeline (existing WebRTC flow)

Need:
- `POST /api/v1/compile-prompt` endpoint
- Gemini Flash API integration
- Simple frontend for prompt input

---

## 2025-12-24 (Earlier)

### Style Layer Scaffolding (Days 5-7)

Added style layer infrastructure:
- `StyleManifest` schema with vocab dictionaries
- `StyleRegistry` for loading/caching manifests from YAML
- `WorldState` schema (domain-agnostic scene representation)
- `PromptCompiler` interface with pluggable backends
- `InstructionSheet` loader

Commit: `3cd10c6 Add style layer scaffolding (Days 5-7)`

### Snapshot/Restore + Step Mode (Phase 2-3)

Live tested on B300:
- Snapshot/restore working via WebRTC data channel
- Step mode generates single chunks while paused
- LRU eviction for snapshot store

Commit: `55a01c1 Add snapshot/restore and step mode to realtime control plane`

---

## 2025-12-23

### FrameProcessor Integration (Phase 1)

Integrated ControlBus into FrameProcessor:
- Drain-all semantics (all pending queue updates applied per chunk)
- Dict→event translation for PAUSE/RESUME, SET_PROMPT, etc.
- chunk_index tracking

Commit: `2c198fc Integrate ControlBus into FrameProcessor (Phase 1)`

### Control Plane Scaffold (Phase 0)

TDD implementation of control plane abstractions:
- `ControlBus` - event queue with deterministic ordering
- `PipelineAdapter` - kwargs mapping, LoRA edge-triggering
- `GeneratorDriver` - async driver (reference implementation)

84 tests passing.

---

## Key Decisions

| Decision | Date | Rationale |
|----------|------|-----------|
| Drain-all queue semantics | 2025-12-23 | Treat queue as "mailbox since last boundary" - cleaner than one-at-a-time |
| Server-side snapshots | 2025-12-24 | GPU tensors can't serialize to JSON for WebRTC |
| V2V step pending until ready | 2025-12-24 | Better UX than immediate fail + retry |
| InstructionSheet carries weight | 2025-12-24 | LLM does thinking; prose more flexible than schemas |
| REST hooks into WebRTC | 2025-12-25 | Reuse existing control path; browser as monitor |
| Fix control path first | 2025-12-25 | GPT-5 Pro review: 6 issues to address before REST |

---

## Test Coverage

- 84 tests: control plane abstractions
- 21 tests: style layer
- 21 tests: FrameProcessor characterization
- **Total: 126 tests**
