# Hardware Control Surface Integration

> Status: Draft (vetted against current Scope code)
> Date: 2025-12-27
> Review: `notes/proposals/hardware-control-surface/review01.md`
> Related: `notes/plans/tui-director-console.md`, `notes/proposals/tidal-cycles-integration.md`

## Summary

Integrate physical hardware (Elgato Stream Deck, MIDI controllers) as ergonomic input surfaces for real-time video direction and audio performance.

**Important reality check (today):**
- Realtime control is **REST + WebRTC data-channel** driven.
- `video-cli` is a REST client for a *connected* WebRTC session.
- There is **no** existing "hardware router" module/command in `src/` yet — this proposal describes one to build.

---

## What exists today (authoritative control surfaces)

### Control planes

| Plane | Exists today? | How to use |
|------|---------------|------------|
| REST control (pause/run/step/prompt/world/style/playlist) | ✅ | `src/scope/server/app.py` + `video-cli` |
| WebRTC data-channel parameter updates | ✅ | `docs/api/parameters.md` ("Send Parameters") |
| Snapshot / restore | ✅ (data-channel protocol + reserved keys) | `apply_control_message` supports message `type`; REST endpoints not implemented |
| "Hardware router" process (MIDI/StreamDeck → Scope) | ❌ | Proposed here |
| Device listing (`--list-devices`) | ❌ | Proposed |
| Optional `control` dependency group in `pyproject.toml` | ❌ | Proposed |

### Single-active-session constraint (REST)

Most REST endpoints call `get_active_session()` which requires:
- exactly **one** connected WebRTC session, otherwise REST control fails.

So a hardware control surface must assume:
- a WebRTC session is already established and connected.

**Important implication:** a "router" that opens its *own* WebRTC connection risks creating a second connected session, which will break most REST control (no REST endpoints accept `session_id` today). For any design that needs continuous parameter updates, prefer either:
- adding a server-side "set parameters" endpoint that forwards to `apply_control_message`, or
- running the router inside the same WebRTC client that already owns the data-channel.

---

## Canonical parameter names (don't invent new ones)

### For WebRTC data-channel updates (authoritative)
See: `docs/api/parameters.md` (plus `src/scope/realtime/control_state.py` for additional supported keys)

Common keys:
- `prompts: [{text, weight}]`
- `denoising_step_list: [1000, 750, 500, 250]` (descending ints)
- `noise_scale: float`
- `noise_controller: bool`
- `manage_cache: bool`
- `reset_cache: bool` (one-shot hard cut)
- `kv_cache_attention_bias: float` (supported; not currently listed in `docs/api/parameters.md`)
- `transition: { target_prompts, num_steps, temporal_interpolation_method }`
- `lora_scales: [{path, scale}]` (requires runtime LoRA merge mode at load)
- `spout_sender`, `spout_receiver` (Windows)
- `vace_ref_images`, `vace_context_scale`

### Reserved keys (handled in FrameProcessor; not forwarded to pipeline)
- `_rcp_world_state` (WorldState full replace)
- `_rcp_set_style` (style name)
- `_rcp_soft_transition` (temporary bias override; used by REST `/realtime/soft-cut`)
- `_rcp_step` (step N chunks)
- `_rcp_snapshot_request`, `_rcp_restore_snapshot` (snapshot/restore)

### Data-channel protocol message types supported by `apply_control_message`
- `{ "type": "snapshot_request" }`
- `{ "type": "restore_snapshot", "snapshot_id": "..." }`
- `{ "type": "step" }` (translated into `_rcp_step`)

---

## The Vision

```
Physical Input → (Proposed) Hardware Router → Scope (REST/WebRTC) + Tidal (OSC) + UI sync
```

A single "semantic" control (e.g., `intensity`) can fan out to:
- video parameters (e.g., `noise_scale`, `denoising_step_list`)
- audio intent (e.g., Tidal `energy`)
- UI state (TUI, Stream Deck button highlights)

This is *semantic control* — you control meaning, not one-off knobs.

---

## Hardware Inventory

### User's Current Setup

| Device | Capabilities |
|--------|-------------|
| Elgato Stream Deck | 15/32/64 LCD buttons, dynamic icons |
| 2-octave MIDI keyboard | Keys, velocity sensitivity |
| MIDI dials | Continuous rotation, relative or absolute |
| MIDI sliders/faders | Linear continuous control |

### Recommended Additions (Future)

| Device | Use Case |
|--------|----------|
| MIDI transport (play/stop/rec) | Timeline control |
| Jog wheel | Timeline scrubbing |
| Foot pedal | Hands-free triggers (step, cue advance) |

---

## Control mapping philosophy (aligned to current Scope)

### Discrete vs Continuous

| Control Type | Hardware | Scope mapping (today) |
|--------------|----------|------------------------|
| Discrete | Stream Deck buttons, MIDI notes | REST endpoints for pause/run/step/style/world (full replace) |
| Continuous | MIDI faders/encoders | WebRTC data-channel parameter updates (recommended for smooth control) |
| Momentary | Keys / pedals | REST step/pause or data-channel messages |

### Semantic layers → real parameters

| Semantic control | Video effect (real today) | Audio effect (proposal) |
|---|---|---|
| `intensity` | adjust `noise_scale` and/or choose `denoising_step_list` preset | map to Tidal `energy` |
| `stability` | adjust `kv_cache_attention_bias` | map to Tidal `density`/"tightness" |
| `transition_speed` | `transition.num_steps` OR soft cut duration (`/realtime/soft-cut`) | map to transition/crossfade |
| `scene_cut` | hard cut via `reset_cache: true` or `/realtime/hard-cut` | Tidal hush / pattern swap |

Notes:
- There is no `denoising_steps` scalar parameter; use `denoising_step_list`.
- There is no `kv_cache_bias`; use `kv_cache_attention_bias`.
- "Soft cut" is a *temporary override* via `_rcp_soft_transition` / `/realtime/soft-cut`.

---

## Stream Deck Integration

### Why Stream Deck

- **Dynamic LCD buttons** — show current state, thumbnails, icons
- **Physical tactile feedback** — faster than keyboard for mode switching
- **Visual confirmation** — see what you're about to press

### What Stream Deck can reliably do with *existing* APIs

Without any new Scope endpoints, Stream Deck buttons can call:

| Button | Action | How to send today |
|---|---|---|
| Play/Pause | pause/resume | `POST /api/v1/realtime/pause` and `POST /api/v1/realtime/run` |
| Step | step one chunk | `POST /api/v1/realtime/step` |
| Style buttons | set style | `PUT /api/v1/realtime/style` (`{"name":"rat"}`) |
| Beat/Camera buttons | update WorldState | `PUT /api/v1/realtime/world` (full replace) |
| Hard cut | reset cache | `POST /api/v1/realtime/hard-cut` |
| Soft cut | temporary bias | `POST /api/v1/realtime/soft-cut` |

### Snapshot button (IMPORTANT)

Snapshot/restore is **not available via REST/CLI** today.
A Stream Deck snapshot button would require:
- sending WebRTC data-channel messages (`type: snapshot_request`) OR
- adding REST endpoints for snapshot/restore (future work).

### Button Layout (15-key example)

```
┌─────┬─────┬─────┬─────┬─────┐
│STYLE│STYLE│STYLE│STYLE│STYLE│  Row 1: Style presets (thumbnails)
│ rat │tmnt │yeti │hidr │     │
├─────┼─────┼─────┼─────┼─────┤
│BEAT │BEAT │BEAT │BEAT │BEAT │  Row 2: Beat triggers
│setup│escl │clmx │payf │rest │
├─────┼─────┼─────┼─────┼─────┤
│ CAM │ CAM │ CAM │SNAP │PLAY │  Row 3: Camera + actions
│close│ med │wide │ 📷  │ ▶⏸ │
└─────┴─────┴─────┴─────┴─────┘
```

### Dynamic Updates

Buttons update to show current state:
- Selected style has highlight ring
- Current beat shows arrow indicator
- Play/Pause shows appropriate icon
- Snapshot button could show last snapshot thumbnail

### Timeline/Branching Mode (Future)

When timeline mode is active, buttons could show:
- Branch previews (thumbnail + name)
- Cue points (numbered, with description)
- Loop in/out markers

```
┌─────┬─────┬─────┬─────┬─────┐
│BR A │BR B │BR C │     │     │  Row 1: Branch options
│ 🌙  │ 🌅  │ 🔥  │     │     │
├─────┼─────┼─────┼─────┼─────┤
│CUE 1│CUE 2│CUE 3│CUE 4│CUE 5│  Row 2: Cue points
│intro│build│drop │break│outro│
├─────┼─────┼─────┼─────┼─────┤
│LOOP │LOOP │ ◀◀  │ ▶▶  │ ▶⏸ │  Row 3: Transport
│ IN  │ OUT │     │     │     │
└─────┴─────┴─────┴─────┴─────┘
```

### Implementation

**Library**: `streamdeck` Python package

```python
from StreamDeck.DeviceManager import DeviceManager

class StreamDeckController:
    def __init__(self, router: InputRouter):
        self.router = router
        self.deck = DeviceManager().enumerate()[0]
        self.deck.set_key_callback(self.on_key)

    def on_key(self, deck, key, state):
        if state:  # Key pressed
            action = self.key_map.get(key)
            if action:
                self.router.dispatch(action)

    def update_button(self, key: int, image: bytes, label: str):
        """Update button with new image/state."""
        self.deck.set_key_image(key, image)
```

---

## MIDI Controller Integration

### Continuous controls (recommend WebRTC data-channel)

For smooth faders/encoders (10–60Hz updates), WebRTC data-channel is the best match,
because REST does not currently provide a "set arbitrary parameters" endpoint.

**Practical note:** a standalone router process cannot currently send data-channel messages into the existing active session. To drive continuous controls from a separate process, we'd likely need a REST forwarder endpoint (e.g. `/api/v1/realtime/parameters`) or to run the router inside the WebRTC client.

| Fader | Canonical video key (today) | Notes |
|------|------------------------------|------|
| 1 | `denoising_step_list` | Choose from presets (you cannot send a scalar "steps") |
| 2 | `noise_scale` | float 0–1 |
| 3 | `kv_cache_attention_bias` | float 0.01–1.0 |
| 4 | `transition.num_steps` | affects prompt interpolation (if using `transition`) |

### Discrete MIDI notes (works via REST)

MIDI note-on can call REST endpoints:
- styles via `/api/v1/realtime/style`
- beat/camera via `/api/v1/realtime/world` (full replace)
- pause/run/step via `/api/v1/realtime/*`

### "Beat triggers" and "camera triggers"

These must change `WorldState` fields:
- `world_state.beat` values: `setup|escalation|climax|payoff|reset|transition`
- `world_state.camera` values: `close_up|medium|wide|low_angle|high_angle|tracking|static|establishing`

### Control Assignments

#### Faders/Sliders → Continuous Parameters

| Fader | Video | Audio (Tidal) | Semantic |
|-------|-------|---------------|----------|
| 1 | `denoising_step_list` preset | energy | Intensity |
| 2 | `noise_scale` (0-1) | tension | Chaos |
| 3 | `kv_cache_attention_bias` (0-1) | density | Complexity |
| 4 | `transition.num_steps` | space | Atmosphere |
| 5 | (reserved) | brightness | Tone |
| 6 | (reserved) | grit | Texture |
| 7 | (reserved) | (reserved) | |
| 8 | Master / timeline scrub | master gain | Master |

#### Keys → Discrete Triggers

**Octave 1 (Lower)**: Beats and Actions
```
C  D  E  F  G  A  B  C
│  │  │  │  │  │  │  │
│  │  │  │  │  │  │  └─ Step (when paused)
│  │  │  │  │  │  └──── Pause/Resume
│  │  │  │  │  └─────── Snapshot
│  │  │  │  └────────── Reset beat
│  │  │  └───────────── Payoff beat
│  │  └──────────────── Climax beat
│  └─────────────────── Escalation beat
└────────────────────── Setup beat
```

**Octave 2 (Upper)**: Styles and Camera
```
C  D  E  F  G  A  B  C
│  │  │  │  │  │  │  │
│  │  │  │  │  │  │  └─ (reserved)
│  │  │  │  │  │  └──── Wide camera
│  │  │  │  │  └─────── Medium camera
│  │  │  │  └────────── Close camera
│  │  │  └───────────── Style 4
│  │  └──────────────── Style 3
│  └─────────────────── Style 2
└────────────────────── Style 1
```

**Velocity Sensitivity**: Key velocity could map to transition speed or intensity modifier.

### Shared MIDI with Tidal

If Tidal is running locally, MIDI can route to both:

```
MIDI Controller
      │
      ▼
┌─────────────────┐
│  MIDI Router    │
│  (e.g., JACK,   │
│   IAC Driver)   │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌───────┐
│ Scope │ │ Tidal │
│ Input │ │ (SuperDirt)
│ Router│ │       │
└───────┘ └───────┘
```

Or the Input Router can forward MIDI CCs to Tidal as OSC:

```python
def on_midi_cc(self, channel, cc, value):
    normalized = value / 127.0

    # Route to video API
    if cc == 1:  # Fader 1
        # Choose denoising_step_list preset based on normalized value
        presets = [
            [1000, 750, 500, 250],  # low intensity
            [1000, 850, 700, 550],  # medium
            [1000, 900, 800, 700],  # high
        ]
        preset_idx = int(normalized * (len(presets) - 1))
        self.video_client.set_param("denoising_step_list", presets[preset_idx])

    # Route to Tidal OSC
    if cc in self.tidal_cc_map:
        control_name = self.tidal_cc_map[cc]
        self.osc_client.send_message("/ctrl", [control_name, normalized])
```

### Implementation

**Library**: `mido` or `python-rtmidi`

```python
import mido

class MIDIController:
    def __init__(self, router: InputRouter, port_name: str):
        self.router = router
        self.input = mido.open_input(port_name, callback=self.on_message)

    def on_message(self, msg):
        if msg.type == 'note_on':
            self.router.dispatch_note(msg.note, msg.velocity)
        elif msg.type == 'control_change':
            self.router.dispatch_cc(msg.control, msg.value)
```

---

## Proposed Hardware Router (new component; not in repo yet)

A separate process that:
1. polls `/api/v1/realtime/state` for current state
2. translates hardware input into either:
   - REST calls (discrete actions)
   - (future) generic parameter updates (needs a server forwarder endpoint, or running inside the WebRTC client)
3. (optional) forwards audio intent to Tidal via OSC (see `tidal-cycles-integration.md`)

### Unified Input Router

Central dispatcher that normalizes all input sources:

```python
class InputRouter:
    """Routes input from any source to video/audio/UI."""

    def __init__(self):
        self.video_client = VideoAPIClient()
        self.tidal_client = TidalOSCClient()  # Optional
        self.state = SharedState()
        self.subscribers = []  # UI updaters

    async def dispatch(self, action: Action):
        """Dispatch action to all relevant destinations."""

        # Update shared state
        self.state.apply(action)

        # Route to video
        if action.affects_video:
            await self.video_client.send(action.to_video_command())

        # Route to Tidal
        if action.affects_audio and self.tidal_client:
            self.tidal_client.send(action.to_tidal_command())

        # Notify UI subscribers (TUI, Stream Deck, web)
        for subscriber in self.subscribers:
            subscriber.on_state_change(self.state)

    def dispatch_cc(self, cc: int, value: int):
        """Handle MIDI CC → semantic control."""
        if cc in CC_MAP:
            semantic, video_param, tidal_param = CC_MAP[cc]
            normalized = value / 127.0

            # Fan out to both
            self.dispatch(SemanticControl(
                name=semantic,
                value=normalized,
                video_param=video_param,
                tidal_param=tidal_param
            ))
```

### State sync (today)

`GET /api/v1/realtime/state` returns:
- `paused`, `chunk_index`, `session_id`, `prompt`
- plus style-layer state: `world_state`, `active_style`, `compiled_prompt`

So a shared state model should align to those names (don't invent new ones).

---

## State Synchronization

All input surfaces need to show the same state:

```python
class SharedState:
    """Single source of truth for all UIs."""

    current_style: str
    current_beat: str
    current_camera: str
    intensity: float  # 0-1
    tension: float
    # ... other semantic controls

    is_playing: bool
    chunk_index: int
    fps: float

    # Timeline state (future)
    current_branch: Optional[str]
    timeline_position: float
    cue_points: List[CuePoint]

class StateSubscriber(Protocol):
    def on_state_change(self, state: SharedState) -> None: ...

# Each UI implements StateSubscriber:
# - TUI: update widgets
# - Stream Deck: update button images
# - Web UI: push via WebSocket
```

---

## Implementation Phases

### Phase 1: MIDI → Video Only

- [ ] MIDI CC input (mido)
- [ ] Map faders to existing API params (`noise_scale`, `kv_cache_attention_bias`)
- [ ] Map keys to style/beat/camera (via REST)
- [ ] Basic InputRouter with video-only dispatch

### Phase 2: Stream Deck Integration

- [ ] Stream Deck button events
- [ ] Dynamic button images (current state highlighting)
- [ ] Button map for styles/beats/actions
- [ ] State subscriber for button updates

### Phase 3: Tidal OSC Forwarding

- [ ] MIDI CC → OSC `/ctrl` for Tidal parameters
- [ ] Shared semantic controls (intensity → energy)
- [ ] Sync with Tidal proposal (Intent Bridge)

### Phase 4: Timeline/Branching Controls

- [ ] Stream Deck branch/cue display
- [ ] MIDI fader for timeline scrub
- [ ] Branch switching actions
- [ ] Cue point jumping

### Phase 5: Bidirectional Sync

- [ ] State changes from video/Tidal update hardware
- [ ] Motorized faders (if available) snap to current values
- [ ] Stream Deck shows server-side state

---

## Configuration (proposed; not present today)

If implemented, these files would be new additions:
- `config/midi_map.yaml`
- `config/streamdeck_layout.yaml`

### MIDI Mapping File

```yaml
# config/midi_map.yaml
device: "Arturia KeyStep Pro"  # Or auto-detect

cc_map:
  1:
    semantic: intensity
    video: denoising_step_list  # Will use preset selection
    tidal: energy
  2:
    semantic: chaos
    video: noise_scale
    video_range: [0.0, 1.0]
    tidal: tension
  3:
    semantic: stability
    video: kv_cache_attention_bias
    video_range: [0.01, 1.0]
    tidal: density
  # ...

note_map:
  36: { action: beat, value: setup }      # C1
  38: { action: beat, value: escalation } # D1
  40: { action: beat, value: climax }     # E1
  41: { action: beat, value: payoff }     # F1
  43: { action: beat, value: reset }      # G1
  # ...
```

### Stream Deck Layout File

```yaml
# config/streamdeck_layout.yaml
device: "Stream Deck MK.2"

pages:
  default:
    rows:
      - [style:rat, style:tmnt, style:yeti, style:hidari, null]
      - [beat:setup, beat:escalation, beat:climax, beat:payoff, beat:reset]
      - [camera:close, camera:medium, camera:wide, action:snapshot, action:play_pause]

  timeline:
    rows:
      - [branch:a, branch:b, branch:c, null, null]
      - [cue:1, cue:2, cue:3, cue:4, cue:5]
      - [loop_in, loop_out, rewind, forward, action:play_pause]
```

---

## Entry Points

### What exists today

- Run the server: `daydream-scope` (see `pyproject.toml`)
- Control via CLI (REST):
  - `video-cli state`
  - `video-cli pause`
  - `video-cli run [--chunks N]`
  - `video-cli step`
  - `video-cli prompt ...`
  - `video-cli world ...` (full replace)
  - `video-cli style list|set|get`
  - `video-cli playlist ...`

### What does NOT exist today (proposal)

- `video-cli control` / `--list-devices`

If we add a router, recommended future entrypoint would be something like:
- `video-cli control-surface` (new subcommand), OR
- `python -m scope.cli.control_surface` (new module)

---

## Dependencies (proposed)

`pyproject.toml` currently does NOT define optional dependency groups for control surfaces.

If/when implementing this proposal, we would add:

```toml
[project.optional-dependencies]
control = [
    "mido>=1.3.0",
    "python-rtmidi>=1.5.0",
    "streamdeck>=0.9.0",
    "python-osc>=1.8.0",  # For Tidal
    "Pillow>=10.0.0",  # For Stream Deck images
]
```

---

## Open Questions

1. **Continuous control transport**: add a REST endpoint for arbitrary params, or rely on WebRTC data-channel?
   - Today: WebRTC data-channel is the only "generic parameter update" path, but it's only available to the WebRTC client (not an external router process).

2. **WorldState patching**: do we add `/realtime/world/patch`?
   - Today: `/realtime/world` is full replace.

3. **Snapshot UX**: do we expose snapshot/restore via REST?
   - Today: snapshot/restore is data-channel only; CLI placeholders exist but are not implemented.

4. **MIDI routing**: Should MIDI go through system MIDI router (JACK/IAC) or direct to Python?
   - Pro system router: Tidal gets MIDI natively
   - Pro direct: Lower latency, simpler setup

5. **Fader scaling**: Linear or exponential curves for parameters?
   - `noise_scale`: maybe exponential (more control at low end)
   - `kv_cache_attention_bias`: probably linear

6. **Conflict resolution**: If Stream Deck says "style A" and MIDI key says "style B" simultaneously?
   - Last-write-wins is probably fine
   - Or: MIDI keys are momentary preview, Stream Deck is latch

---

## References

### Internal

- `docs/api/parameters.md` (canonical parameter names)
- `src/scope/realtime/control_state.py` (ControlState fields incl. `kv_cache_attention_bias`)
- `src/scope/server/app.py` (REST realtime endpoints)
- `src/scope/server/webrtc.py` (`apply_control_message` message translation)
- `src/scope/server/frame_processor.py` (reserved keys + soft transition behavior)
- `src/scope/realtime/world_state.py` (beat/camera canonical values)

### External

- Stream Deck SDK: https://developer.elgato.com/documentation/stream-deck/
- Python streamdeck: https://python-elgato-streamdeck.readthedocs.io/
- mido (MIDI): https://mido.readthedocs.io/
- python-rtmidi: https://spotlightkid.github.io/python-rtmidi/
- python-osc: https://python-osc.readthedocs.io/
- TUI Director Console: `notes/plans/tui-director-console.md`
- Tidal Integration: `notes/proposals/tidal-cycles-integration.md`
