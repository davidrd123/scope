# TUI Director Console Proposal

**Goal**: Real-time keyboard-driven control for live video direction.

---

## Why TUI over Web UI?

| TUI (Textual) | Web UI |
|---------------|--------|
| Zero latency (local) | Network round-trip |
| Works over SSH | Requires browser |
| Keyboard-first (fast) | Mouse-heavy |
| Single terminal window | Separate browser tab |
| Easy to script/automate | Harder to automate |

**Recommendation**: TUI for director, Web UI for observers/clients.

---

## Layout Sketch

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ SCOPE DIRECTOR                                    chunk: 1234  15.2 FPS  ▶  │
├───────────────────────────┬─────────────────────────────────────────────────┤
│ STYLE: [rat]              │ COMPILED PROMPT                                 │
│ ────────────────────────  │ ─────────────────────────────────────────────── │
│ ○ tmnt                    │ Clay-Plastic Pose-to-Pose Animation,            │
│ ● rat                     │ holds a strong key pose, Rooster with           │
│ ○ yeti                    │ exasperated eye roll, Terry with jubilant       │
│ ○ hidari                  │ glee, medium shot, reaction HOLD...             │
│                           │                                                 │
├───────────────────────────┼─────────────────────────────────────────────────┤
│ BEAT: [payoff]            │ WORLD STATE                                     │
│ ────────────────────────  │ ─────────────────────────────────────────────── │
│ [1] setup                 │ scene: aftermath, Rooster singed                │
│ [2] escalation            │ camera: medium                                  │
│ [3] climax                │ action: idle                                    │
│ [4] payoff    ◀──         │ ───────────────────────────────────────────     │
│ [5] reset                 │ ROOSTER: frustrated, slumped                    │
│                           │ TERRY: happy, walking away                      │
├───────────────────────────┼─────────────────────────────────────────────────┤
│ CAMERA: [m]edium          │ QUICK ACTIONS                                   │
│ ────────────────────────  │ ─────────────────────────────────────────────── │
│ [c] close_up              │ [Space] pause/resume                            │
│ [m] medium     ◀──        │ [Enter] step (when paused)                      │
│ [w] wide                  │ [S] snapshot                                    │
│ [l] low_angle             │ [R] restore last snapshot                       │
│                           │ [P] edit prompt (modal)                         │
├───────────────────────────┴─────────────────────────────────────────────────┤
│ > status: running | session: a11cd4b6 | last update: 0.3s ago              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Keyboard Controls

### Global (always active)
| Key | Action |
|-----|--------|
| `Space` | Toggle pause/resume |
| `Enter` | Step one chunk (when paused) |
| `q` | Quit |
| `?` | Help overlay |

### Style Selection (1-9 or Tab to cycle)
| Key | Action |
|-----|--------|
| `Tab` | Cycle to next style |
| `Shift+Tab` | Cycle to previous style |
| `1-9` | Select style by index |

### Beat Selection (always active)
| Key | Action |
|-----|--------|
| `1` | setup |
| `2` | escalation |
| `3` | climax |
| `4` | payoff |
| `5` | reset |

### Camera (c prefix or direct)
| Key | Action |
|-----|--------|
| `c` then `c` | close_up |
| `c` then `m` | medium |
| `c` then `w` | wide |
| `c` then `l` | low_angle |

Or just `m` for medium, `w` for wide when in camera focus.

### Character Quick-Set
| Key | Action |
|-----|--------|
| `r` | Focus Rooster panel |
| `t` | Focus Terry panel |
| Then emotion keys... |

### Emotions (when character focused)
| Key | Action |
|-----|--------|
| `h` | happy |
| `a` | angry |
| `s` | sad/shocked |
| `f` | frustrated |
| `d` | determined |
| `n` | neutral |

### Snapshots
| Key | Action |
|-----|--------|
| `S` | Create snapshot |
| `R` | Restore last snapshot |
| `L` | List snapshots (modal) |

### Prompt Override
| Key | Action |
|-----|--------|
| `P` | Open prompt editor (modal) |
| `Esc` | Cancel and return to auto-compile |

---

## Implementation Stack

```
Textual (Python TUI framework)
    │
    ├── httpx (async HTTP client)
    │   └── Calls existing REST API
    │
    └── websocket (optional)
        └── For real-time state updates
```

### Core Components

```python
# src/scope/cli/director_tui.py

class DirectorApp(App):
    """Main TUI application."""

    BINDINGS = [
        ("space", "toggle_pause", "Pause/Resume"),
        ("enter", "step", "Step"),
        ("q", "quit", "Quit"),
        ("tab", "next_style", "Next Style"),
        ("1", "beat_setup", "Setup"),
        ("2", "beat_escalation", "Escalation"),
        # ...
    ]

class StylePanel(Widget):
    """Style selector with radio buttons."""

class BeatPanel(Widget):
    """Beat selector with number keys."""

class CameraPanel(Widget):
    """Camera intent selector."""

class WorldStatePanel(Widget):
    """Live WorldState display."""

class PromptPanel(Widget):
    """Compiled prompt display."""

class StatusBar(Widget):
    """Bottom status with FPS, chunk, session."""
```

---

## State Sync

**Polling approach** (simple, reliable):
```python
async def poll_state(self):
    while True:
        state = await self.client.get("/api/v1/realtime/state")
        self.update_display(state)
        await asyncio.sleep(0.1)  # 10 Hz
```

**WebSocket approach** (future, lower latency):
```python
async def subscribe_state(self):
    async with websockets.connect(ws_url) as ws:
        async for message in ws:
            self.update_display(json.loads(message))
```

---

## Phased Implementation

### Phase 1: Read-Only Display
- Connect to running server
- Display current state (chunk, prompt, style, world)
- Pause/resume/step controls
- ~2 hours

### Phase 2: Style & Beat Controls
- Style switching (Tab, 1-9)
- Beat selection (1-5)
- Camera quick-keys
- ~2 hours

### Phase 3: Character Editing
- Focus panels (r/t for Rooster/Terry)
- Emotion quick-keys
- Action editing
- ~2 hours

### Phase 4: Polish
- Help overlay
- Snapshot management
- Prompt editor modal
- Error handling
- ~2 hours

---

## Open Questions

1. **Key conflicts**: Beat (1-5) vs Style (1-9) - use modifier?
   - Option A: `Ctrl+1-5` for beats, `1-9` for styles
   - Option B: Mode-based (press `b` to enter beat mode)
   - Option C: Separate panels, keys depend on focus

2. **Character limit**: What if there are 5+ characters?
   - Scrollable list with up/down
   - Or limit to 2-3 "focus" characters

3. **Prompt override**: How long should manual prompt "stick"?
   - Until next world update (current)
   - Until explicitly cleared
   - Timeout-based

4. **FPS display**: Do we have access to actual pipeline FPS?
   - Currently chunk_index delta could approximate it

---

## Entry Point

```bash
# Launch the director TUI
uv run video-cli director

# Or standalone
uv run python -m scope.cli.director_tui
```

---

## Dependencies

```toml
[project.optional-dependencies]
tui = [
    "textual>=0.47.0",
]
```

---

## Related

- Phase 6a: Style layer (done)
- Phase 7: Web UI (separate, for observers)
- Phase 8: VLM feedback (could display in TUI too)
