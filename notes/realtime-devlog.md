# Realtime Control Plane - Development Log

Chronological record of implementation work and decisions.

---

## 2025-12-24 (Afternoon)

### Integrated Incoming Specs

Processed and distilled three incoming spec documents:
- `incoming/project_knowledge.md` - Project knowledge base
- `incoming/context_editing_and_console_spec.md` - Context editing mechanism
- `incoming/interface_specifications.md` - CLI/Web UI specs

**Key integration:**
- Updated `notes/realtime-roadmap.md` with two feature axes (prompt compilation + context editing)
- Created `notes/reference/context-editing-code.md` - Gemini integration, edit API, validation test
- Created `notes/reference/cli-implementation.md` - Full CLI implementation with Click

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
- `incoming/style/RAT/` - Rooster & Terry (claymation comedy)
- `incoming/style/Kaiju/` - Japanese tokusatsu films
- `incoming/style/TMNT/` - Graffiti sketchbook animation

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

---

## Test Coverage

- 84 tests: control plane abstractions
- 21 tests: style layer
- 21 tests: FrameProcessor characterization
- **Total: 126 tests**
