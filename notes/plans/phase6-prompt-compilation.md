# Phase 6: Prompt Compilation & World State

**Goal**: Interactive prompt compilation with state tracking and LLM-powered refinement.

---

## Core Concept

```
WorldState (JSON)          ← "characters, environment, what's happening"
     ↓
InstructionSheet (prose)   ← "how to translate for this LoRA/style"
     ↓
Gemini Flash               ← "compile to T2V prompt"
     ↓
Compiled prompt            → apply to pipeline
```

---

## WorldState Tracking

### Current State
```json
{
  "characters": [
    {"name": "Rooster", "descriptor": "clay-puppet rooster", "action": "chasing"},
    {"name": "Terry", "descriptor": "nervous clay hen", "action": "fleeing"}
  ],
  "environment": "farmhouse kitchen",
  "lighting": "warm morning light through window",
  "camera": "medium shot, slight dutch angle",
  "mood": "comedic chaos"
}
```

### History
Keep last N states for continuity:
```json
{
  "current": { ... },
  "history": [
    {"chunk_index": 42, "state": { ... }},
    {"chunk_index": 38, "state": { ... }}
  ]
}
```

---

## Interaction Patterns

### 1. Direct State Update
Set WorldState directly:
```
POST /api/v1/world/state
{"characters": [...], "environment": "..."}
```

### 2. Change Request (semantic)
Natural language → WorldState delta:
```
POST /api/v1/world/change
{"instruction": "make Rooster angry instead of playful"}
```
→ LLM interprets, updates WorldState, recompiles

### 3. Jiggle (refinement)
Improve prompt without changing intent:
```
POST /api/v1/prompt/jiggle
{"reason": "more dynamic motion"}
```
→ LLM rewrites prompt, keeps WorldState same

### 4. Compile & Apply
Force recompilation and apply:
```
POST /api/v1/prompt/compile
{"apply": true}
```

### 5. Preview (no apply)
See what would compile without applying:
```
POST /api/v1/prompt/compile
{"apply": false}
```
→ Returns compiled prompt for review

---

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/v1/world/state` | Current WorldState + history |
| PUT | `/api/v1/world/state` | Set WorldState directly |
| POST | `/api/v1/world/change` | Semantic change via LLM |
| POST | `/api/v1/prompt/jiggle` | Refine prompt, keep intent |
| POST | `/api/v1/prompt/compile` | Compile & optionally apply |
| GET | `/api/v1/style/list` | Available styles |
| PUT | `/api/v1/style/active` | Set active style |

---

## Session-Owned State

Each WebRTC session owns:
```python
class SessionPromptState:
    world_state: WorldState
    world_history: list[tuple[int, WorldState]]  # (chunk_index, state)
    active_style: StyleManifest
    compiled_prompt: str | None
    compiler: PromptCompiler
```

Compile happens:
- On explicit `/compile` call
- On WorldState change (if auto-compile enabled)
- At chunk boundaries (optional continuous mode)

---

## Gemini Integration

```python
async def compile_prompt(
    world_state: WorldState,
    instruction_sheet: str,
    style: StyleManifest,
) -> str:
    """Compile WorldState to T2V prompt via Gemini Flash."""
    system = f"""You are a video prompt compiler.

Style: {style.trigger_phrase}

{instruction_sheet}

Output ONLY the compiled prompt, no explanation."""

    user = f"""Compile this scene to a T2V prompt:

{world_state.to_json()}"""

    response = await gemini.generate_content(
        model="gemini-2.0-flash",
        contents=[{"role": "user", "parts": [{"text": user}]}],
        system_instruction=system,
    )
    return response.text.strip()
```

---

## Change Request Flow

```python
async def apply_change(
    current_state: WorldState,
    instruction: str,
    history: list[WorldState],
) -> WorldState:
    """Apply semantic change via LLM."""
    system = """You modify WorldState JSON based on user instructions.
Output ONLY valid JSON, no explanation."""

    user = f"""Current state:
{current_state.to_json()}

Recent history:
{format_history(history)}

User instruction: {instruction}

Output the modified WorldState JSON:"""

    response = await gemini.generate_content(...)
    return WorldState.from_json(response.text)
```

---

## Wire into Existing Path

From Codex's summary:

1. **Session owns state**: `WorldState` + `StyleManifest` + `PromptCompiler`
2. **Compile at chunk boundaries**: Feed `ControlBus` / `FrameProcessor`
3. **Include in `/api/v1/realtime/state`**: Show compiled prompt + active style/world
4. **Convert one real style pack**: `styles/rat_v1/` with manifest + instruction sheet

---

## CLI Extensions

```bash
video-cli world                     # Get current WorldState
video-cli world --set '{"...}'      # Set WorldState
video-cli change "make X happen"    # Semantic change
video-cli jiggle                    # Refine current prompt
video-cli compile                   # Force compile & apply
video-cli compile --preview         # Preview without applying
video-cli style list                # Available styles
video-cli style set rat_v1          # Set active style
```

---

## Implementation Order

1. **Wire existing scaffolding**: Connect `WorldState`, `PromptCompiler`, `StyleManifest` to session
2. **Add `/compile` endpoint**: WorldState → prompt (no LLM yet, use TemplateCompiler)
3. **Add Gemini integration**: Replace template with LLM compilation
4. **Add state tracking**: History, include in `/state` response
5. **Add change/jiggle**: Semantic modification endpoints
6. **Convert RAT style**: Real manifest + instruction sheet under `styles/rat_v1/`
7. **Integration tests**: style/world update → compiled kwargs → pipeline call

---

## Open Questions

- [ ] Auto-compile on WorldState change, or explicit only?
- [ ] How much history to keep? (5 states? 10?)
- [ ] Jiggle: should it show diff or just replace?
- [ ] Preview: return both old and new prompt for comparison?

---

## Related Phases

- **Phase 6b (Context Editing)**: Edit anchor frame visually (separate from prompt)
- **Phase 7 (UI)**: Web console to interact with all of this
- **Phase 8 (VLM Feedback)**: `describe-frame` for agent evaluation loop
