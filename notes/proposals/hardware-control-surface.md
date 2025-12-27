# Hardware Control Surface Integration

> Status: Draft
> Date: 2025-12-27
> Related: `tui-director-console.md`, `tidal-cycles-integration.md`

## Summary

Integrate physical hardware (Elgato Stream Deck, MIDI controllers) as ergonomic input surfaces for real-time video direction and audio performance. This creates a unified "performance rig" where one gesture can affect both video and audio simultaneously.

## The Vision

```
Physical Input → Unified Router → Video + Audio + UI Sync
```

A single MIDI fader labeled "intensity" could:
- Increase video `noise_scale` and `denoising_steps`
- Send Tidal `energy` and `tension` values
- Update all UI displays (TUI, Stream Deck icons, web observers)

This is **semantic control** — you're controlling *meaning*, not parameters.

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

## Control Mapping Philosophy

### Discrete vs Continuous

| Control Type | Hardware | Examples |
|--------------|----------|----------|
| **Discrete** | Stream Deck buttons, MIDI keys | Style select, beat trigger, branch switch |
| **Continuous** | MIDI faders, dials | Intensity, camera zoom, crossfade position |
| **Momentary** | MIDI keys (velocity), foot pedal | Step-while-held, preview-while-held |

### Semantic Layers

Instead of mapping hardware directly to low-level parameters, define semantic controls that fan out to multiple destinations:

| Semantic Control | Video Effect | Audio Effect (Tidal) |
|------------------|--------------|----------------------|
| `intensity` | ↑ denoising_steps, ↑ noise_scale | ↑ energy, ↑ gain |
| `tension` | (camera shake?), (color grading?) | ↑ tension, ↑ filter mod |
| `space` | (depth of field?) | ↑ reverb, ↑ stereo width |
| `focus` | camera preset | solo channel |

---

## Stream Deck Integration

### Why Stream Deck

- **Dynamic LCD buttons** — show current state, thumbnails, icons
- **Physical tactile feedback** — faster than keyboard for mode switching
- **Visual confirmation** — see what you're about to press

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

### Control Assignments

#### Faders/Sliders → Continuous Parameters

| Fader | Video | Audio (Tidal) | Semantic |
|-------|-------|---------------|----------|
| 1 | denoising_steps (4-8) | energy | Intensity |
| 2 | noise_scale (0-1) | tension | Chaos |
| 3 | kv_cache_bias (0-1) | density | Complexity |
| 4 | crossfade_duration | space | Atmosphere |
| 5 | (reserved) | brightness | Tone |
| 6 | (reserved) | grit | Texture |
| 7 | (reserved) | (reserved) | |
| 8 | Master / timeline scrub | master gain | Master |

#### Dials/Encoders → Morphing/Fine Control

| Dial | Function |
|------|----------|
| 1 | Camera parameter fine-tune |
| 2 | Character attribute morph |
| 3 | Style blend (between adjacent) |
| 4 | Timeline position (jog) |

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
        self.video_client.set_param("denoising_steps", int(4 + normalized * 4))

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

## Unified Input Router

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
- [ ] Map faders to existing API params (denoising_steps, noise_scale)
- [ ] Map keys to style/beat/camera
- [ ] Basic InputRouter with video-only dispatch
- ~4 hours

### Phase 2: Stream Deck Integration

- [ ] Stream Deck button events
- [ ] Dynamic button images (current state highlighting)
- [ ] Button map for styles/beats/actions
- [ ] State subscriber for button updates
- ~4 hours

### Phase 3: Tidal OSC Forwarding

- [ ] MIDI CC → OSC `/ctrl` for Tidal parameters
- [ ] Shared semantic controls (intensity → energy)
- [ ] Sync with Tidal proposal (Intent Bridge)
- ~2 hours

### Phase 4: Timeline/Branching Controls

- [ ] Stream Deck branch/cue display
- [ ] MIDI fader for timeline scrub
- [ ] Branch switching actions
- [ ] Cue point jumping
- ~4 hours

### Phase 5: Bidirectional Sync

- [ ] State changes from video/Tidal update hardware
- [ ] Motorized faders (if available) snap to current values
- [ ] Stream Deck shows server-side state
- ~4 hours

---

## Configuration

### MIDI Mapping File

```yaml
# config/midi_map.yaml
device: "Arturia KeyStep Pro"  # Or auto-detect

cc_map:
  1:
    semantic: intensity
    video: denoising_steps
    video_range: [4, 8]
    tidal: energy
  2:
    semantic: chaos
    video: noise_scale
    video_range: [0.0, 1.0]
    tidal: tension
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

```bash
# Start input router (connects to running server)
uv run video-cli control

# Or with specific devices
uv run video-cli control --midi "KeyStep Pro" --streamdeck

# List available devices
uv run video-cli control --list-devices
```

---

## Dependencies

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

1. **MIDI routing**: Should MIDI go through system MIDI router (JACK/IAC) or direct to Python?
   - Pro system router: Tidal gets MIDI natively
   - Pro direct: Lower latency, simpler setup

2. **Fader scaling**: Linear or exponential curves for parameters?
   - Denoising steps: probably linear (4-8 is small range)
   - Noise scale: maybe exponential (more control at low end)

3. **Conflict resolution**: If Stream Deck says "style A" and MIDI key says "style B" simultaneously?
   - Last-write-wins is probably fine
   - Or: MIDI keys are momentary preview, Stream Deck is latch

4. **Motorized faders**: Worth supporting for bidirectional sync?
   - Cool but expensive
   - Most MIDI controllers don't have them

5. **Preset management**: Save/recall hardware mappings?
   - Useful for different performance contexts
   - YAML files per "rig profile"

---

## References

- Stream Deck SDK: https://developer.elgato.com/documentation/stream-deck/
- Python streamdeck: https://python-elgato-streamdeck.readthedocs.io/
- mido (MIDI): https://mido.readthedocs.io/
- python-rtmidi: https://spotlightkid.github.io/python-rtmidi/
- python-osc: https://python-osc.readthedocs.io/
- TUI Director Console: `notes/plans/tui-director-console.md`
- Tidal Integration: `notes/proposals/tidal-cycles-integration.md`
