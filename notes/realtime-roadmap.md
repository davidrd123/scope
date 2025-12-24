# Realtime Control Plane - Roadmap

**Goal**: Interactive real-time video generation with style-aware prompt compilation AND retroactive context editing.

**Timeline**: 16 days to hackathon demo (from incoming specs)

**Current State**: Core infrastructure complete. Need interactive UI and context editing.

---

## Two Feature Axes

The system has two complementary ways to influence generation:

| Axis | What it does | How |
|------|--------------|-----|
| **Prompt Compilation** | Control what we SAY to the model | WorldState → LLM → styled prompt |
| **Context Editing** | Control what the model SEES | Edit anchor frame → KV cache propagation |

Both feed into the same pipeline. Both are needed for full creative control.

---

## Phases Overview

| Phase | Status | Description |
|-------|--------|-------------|
| 0. Control Plane | Done | ControlBus, PipelineAdapter, event ordering |
| 1. FrameProcessor Integration | Done | Drain-all semantics, dict→event translation |
| 2. Snapshot/Restore | Done | Server-side snapshots, LRU eviction |
| 3. Step Mode | Done | Generate single chunks while paused |
| 4. Style Layer | Simplified | Minimal StyleManifest + rich InstructionSheets |
| 5. CLI + Core API | Next | `video-cli` + REST endpoints |
| 6. Context Editing | Planned | Anchor frame editing via Gemini image model |
| 7. Interactive UI | Planned | Web console for prompt tuning + editing |
| 8. VLM Feedback Loop | Planned | Describe-frame for agent evaluation |

---

## What's Working Now

**Execution backend**: WebRTC streaming pipeline
- `FrameProcessor` → `pipeline.call()` → frames via WebRTC
- Control via data channel: pause/resume, prompts, LoRA scales, seed
- Snapshot/restore for branching
- Step mode for frame-by-frame
- **14 fps on B300** (from incoming specs)

**Style layer scaffolding** (tests pass, not wired in):
- `StyleManifest`: trigger phrase, LoRA path, instruction sheet path
- `WorldState`: characters, actions, lighting, camera
- `PromptCompiler`: Template (deterministic), LLM (planned), Cached (memoization)

**Pipeline buffers** (key for context editing):
- `decoded_frame_buffer`: 9 RGB frames (4x temporal downsample)
- `context_frame_buffer`: 2 latent frames
- KV cache: ~14 blocks

---

## Phase 5: CLI + Core API

**Goal**: CLI-first interface for both humans and agents.

### Core Endpoints (MVP)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/state` | Current driver state, chunk count, prompt |
| POST | `/api/step` | Generate one chunk |
| POST | `/api/run` | Start continuous generation |
| POST | `/api/pause` | Pause generation |
| PUT | `/api/prompt` | Set prompt text |
| GET | `/api/frame/latest` | Get current frame as image |

### CLI Commands

```bash
video-cli state                    # Get current state (JSON)
video-cli step                     # Generate one chunk
video-cli run [--chunks N]         # Run N chunks
video-cli pause                    # Pause generation
video-cli prompt "text"            # Set prompt
video-cli frame --out path.png     # Save current frame
video-cli snapshot                 # Create snapshot
video-cli restore <id>             # Restore snapshot
```

All commands return JSON for agent automation.

---

## Phase 6: Context Editing

**Goal**: Edit anchor frame → changes propagate to future frames.

### The Mechanism

```
decoded_frame_buffer[:, :1] (RGB anchor frame)
        │
        ▼ Edit with Gemini image model
        │
        ▼ Next recompute: vae.encode_to_latent(edited_frame)
        │
        ▼ KV cache rebuilt from edited content
        │
        ▼ Future frames "remember" the edit
```

This works because `recompute_kv_cache.py` re-encodes the first decoded frame into latent space during KV cache recomputation.

### Validation Experiment

Before building anything, prove the mechanism:

```python
# Color tint test
decoded_buffer = pipeline.state.get('decoded_frame_buffer')
anchor = decoded_buffer[:, :1].clone()
anchor[:, :, 2, :, :] = 1.0  # Max blue
anchor[:, :, 0, :, :] = 0.0  # Zero red
decoded_buffer[:, :1] = anchor
# Continue generating → should stay blue
```

### New Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/edit` | Edit anchor frame with semantic instruction |
| POST | `/api/edit/preview` | Preview edit without applying |
| GET | `/api/frame/describe` | VLM description of current frame |

### Gemini Integration

```python
async def edit_frame(frame: torch.Tensor, instruction: str) -> torch.Tensor:
    """Apply semantic edit via Gemini 2.5 Flash."""
    input_b64 = tensor_to_base64(frame)
    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash-preview-native-image",
        contents=[{
            "parts": [
                {"inline_data": {"mime_type": "image/png", "data": input_b64}},
                {"text": instruction}
            ]
        }]
    )
    # Extract and return edited image
```

---

## Phase 7: Interactive UI

**Goal**: Web console for prompt tuning and editing.

### Layout (3-panel)

```
┌──────────┬─────────────────────────────┬──────────────┐
│ Sidebar  │ Main                        │ Right Panel  │
│          │                             │              │
│ Status   │ Controls: ▶ ⏸ ⏭ [prompt]   │ Edit Panel   │
│ Chunk    │                             │ - Anchor     │
│ Buffer   │ Frame Preview               │ - Instruction│
│          │ (large)                     │ - Apply      │
│          │                             │              │
│          │ Timeline                    │ Metrics      │
│          │ [frame thumbnails]          │ - Latency    │
│          │                             │ - VRAM       │
└──────────┴─────────────────────────────┴──────────────┘
```

### FastHTML Implementation

Web UI at port 8001, hits API at port 8000. Polling-based initially, upgrade to WebSocket if needed.

---

## Phase 8: VLM Feedback Loop

**Goal**: Agent can evaluate frames and decide next action.

### describe-frame Endpoint

```bash
$ video-cli describe-frame
{"description": "A red ball mid-bounce on grass, wooden fence in background"}
```

### Agent Loop Pattern

```python
def direct_video(goal: str):
    run_cli(f'video-cli prompt "{goal}"')

    for i in range(max_iterations):
        run_cli("video-cli step")
        desc = run_cli("video-cli describe-frame")

        if "character not visible" in desc:
            run_cli('video-cli edit "bring character to center"')
        elif i == 20:  # Narrative beat
            run_cli('video-cli snapshot')
            run_cli('video-cli prompt "character notices something"')
```

---

## Prompt Compilation (Style Layer)

### Minimal StyleManifest

```python
@dataclass
class StyleManifest:
    name: str
    trigger_phrase: str
    lora_path: str
    instruction_sheet_path: str
    negative_prompt: str | None = None
```

### InstructionSheet Checklist (7 Universal Patterns)

1. Trigger phrase usage
2. Anchor hierarchy (essential/common/situational)
3. First mention pattern: `[descriptor], [Name]`
4. Structural flow (ESTABLISH → STYLE → ACTION → LIGHTING → CONCLUDE)
5. Length tiers (quick/standard/complex)
6. Off-road guidance ("keep grammar, change nouns")
7. OOD/attention budget ("one novel thing at a time")

### Compile Endpoint

```
POST /api/v1/compile-prompt
{
  "style": "rat_v1",
  "scene": "Rooster chases Terry around kitchen table"
}

Response:
{
  "compiled_prompt": "Clay-Plastic Pose-to-Pose Animation — ...",
  "tokens": 142
}
```

---

## Execution Backend

**To run**:
```bash
uv run python app.py          # Start server at :8000
# Open browser to localhost:8000
# WebRTC connects, video streams
```

**Control via data channel**:
```javascript
window.dataChannel.send('{"prompts": [{"text": "...", "weight": 1.0}]}');
window.dataChannel.send('{"paused": true}');
window.dataChannel.send('{"type": "step"}');
window.dataChannel.send('{"type": "snapshot_request"}');
```

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2025-12-24 | InstructionSheet carries the weight | LLM does thinking; prose > schema |
| 2025-12-24 | StyleManifest stays minimal | trigger, lora_path, instruction_sheet_path |
| 2025-12-24 | 7 universal patterns = checklist | Documentation, not code slots |
| 2025-12-24 | V2V step: pending until ready | Better UX for creative tool |
| 2025-12-24 | CLI-first interface | Agent automation is primary use case |
| 2025-12-24 | Context editing via anchor frame | Re-encode path in KV cache recompute |

---

## Files Reference

| File | Purpose |
|------|---------|
| `app.py` | FastAPI server, WebRTC endpoints |
| `src/scope/server/frame_processor.py` | Pipeline driver, control integration |
| `src/scope/realtime/style_manifest.py` | StyleManifest + StyleRegistry |
| `src/scope/realtime/prompt_compiler.py` | TemplateCompiler, LLMCompiler (stub) |
| `src/scope/realtime/world_state.py` | WorldState schema |
| `styles/example/manifest.yaml` | Template manifest |
| `styles/example/instructions.md` | Template instruction sheet |

---

## Source Documents

These incoming docs were distilled into this roadmap:

| Doc | Content |
|-----|---------|
| `incoming/project_knowledge.md` | Project knowledge base, Gemini integration code |
| `incoming/context_editing_and_console_spec.md` | Context editing mechanism, CLI/API specs |
| `incoming/interface_specifications.md` | Full CLI implementation, FastHTML web UI |

Key code snippets and runbooks are preserved in `notes/reference/`.
