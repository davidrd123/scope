# Tidal Cycles Integration

> Status: Draft
> Date: 2025-12-26
> Research: `notes/research/2025-12-26/tidal/aoi_5pro_01.md`

## Summary

Synchronize live-coded music (Tidal Cycles) with real-time video generation by routing "music intent" derived from video cues to Tidal parameters. This creates a unified audio-visual performance instrument where narrative intent drives both video prompts and musical expression.

## The Vision

```
World State → Narrative Intent → ┬→ Video Prompts → Krea Realtime → Visual Output
                                 │
                                 └→ Music Intent  → Tidal Cycles → Audio Output
```

The same narrative intention layer that compiles world state into video prompts also emits music intent — a small vocabulary of continuous parameters (energy, tension, space) plus discrete actions (solo, hush, pattern switch). This keeps video and music semantically aligned without tight temporal coupling.

## Architecture

### Control Planes

| Plane | Mechanism | Latency | Use Case |
|-------|-----------|---------|----------|
| **Parameter steering** | OSC `/ctrl` → Tidal `cF/cS/cI` | ~10ms | Continuous morphing during soft cuts |
| **Arrangement control** | OSC `/solo`, `/hush`, `/muteAll` | ~10ms | Scene boundary toggles |
| **Pattern rewrites** | MCP `tidal_eval` | ~100ms | Hard cuts, section changes |

### Network Topology

```
┌─────────────────────────────────────────────────────────────────┐
│  VIDEO BOX (Remote GPU)                                         │
│  ┌─────────────────┐     ┌─────────────────┐                   │
│  │  Krea Realtime  │     │  Intent Emitter │                   │
│  │  (video gen)    │     │  (HTTP POST)    │                   │
│  └────────┬────────┘     └────────┬────────┘                   │
│           │                       │                             │
└───────────┼───────────────────────┼─────────────────────────────┘
            │ WebRTC                │ HTTP/WebSocket
            ▼                       ▼
┌───────────────────────────────────────────────────────────────────┐
│  MUSIC BOX (Local)                                                │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────┐ │
│  │  Intent Bridge  │────▶│  Tidal Cycles   │────▶│  SuperDirt  │ │
│  │  (HTTP→OSC)     │ OSC │  (GHCi)         │     │  (audio)    │ │
│  └─────────────────┘     └─────────────────┘     └─────────────┘ │
│           ▲                       ▲                              │
│           │                       │                              │
│  ┌────────┴────────┐     ┌────────┴────────┐                    │
│  │  Buddy Console  │     │  MCP Server     │                    │
│  │  (live tweaks)  │     │  (agent edits)  │                    │
│  └─────────────────┘     └─────────────────┘                    │
└───────────────────────────────────────────────────────────────────┘
```

## Music Intent Vocabulary

Keep it small and composable — these map cleanly to both continuous morphing and discrete switches.

### Continuous Controls (0–1 range)

| Control | Description | Tidal Mapping |
|---------|-------------|---------------|
| `energy` | Overall intensity | Drum density, gain, distortion amount |
| `tension` | Harmonic/rhythmic pressure | Filter modulation depth, dissonance, syncopation |
| `density` | Event frequency | Note count, layering, subdivision |
| `space` | Room/atmosphere | Reverb send, delay feedback, stereo width |
| `brightness` | Tonal character | Filter cutoff, harmonic content |
| `grit` | Texture/edge | Distortion, bit crush, noise |

### Discrete Controls

| Control | Values | Tidal Mapping |
|---------|--------|---------------|
| `focus` | `drums`, `bass`, `pads`, `full` | `/solo` / `/unsolo` channels |
| `action` | `hush`, `muteAll`, `unmuteAll` | Playback controller OSC |

## Mapping to Video Events

| Video Event | Music Response | Implementation |
|-------------|----------------|----------------|
| **Prompt change** | Update intent values | OSC `/ctrl` for each control |
| **Soft cut** (plasticity window) | Morph parameters + Tidal transition | `xfadeIn`, `clutchIn`, `interpolateIn` |
| **Hard cut** (cache reset) | Hush + new pattern bank | `/hush` → `tidal_eval` new patterns |
| **Transition start** | Begin parameter interpolation | Ramp intent values over N chunks |

## Cue Sheet Format

Aligned with video prompt playlists:

```json
{
  "version": "1.0",
  "cues": [
    {
      "cue_id": "akira_001",
      "prompt_id": "playlist_index_1",
      "t_start_s": 0.0,
      "t_end_s": 8.5,
      "transition": {
        "type": "soft",
        "cycles": 4
      },
      "tags": ["neo_tokyo", "night", "neon", "establishing"],
      "music_intent": {
        "energy": 0.3,
        "tension": 0.2,
        "density": 0.4,
        "space": 0.7,
        "brightness": 0.5,
        "grit": 0.2
      },
      "arrangement": {
        "focus": "pads",
        "action": null
      }
    },
    {
      "cue_id": "akira_002",
      "prompt_id": "playlist_index_2",
      "t_start_s": 8.5,
      "t_end_s": 15.2,
      "transition": {
        "type": "hard",
        "cycles": 0
      },
      "tags": ["chase", "motorbikes", "speed"],
      "music_intent": {
        "energy": 0.85,
        "tension": 0.9,
        "density": 0.8,
        "space": 0.15,
        "brightness": 0.7,
        "grit": 0.6
      },
      "arrangement": {
        "focus": "drums+bass",
        "action": "hush"
      }
    }
  ]
}
```

## Implementation Phases

### Phase 1: OSC Parameter Steering (MVP)

**Goal:** Continuous music-video alignment without pattern rewrites.

**Components:**
1. **Parametric Tidal Patch** — Single stable patch that reads controls via `cF`/`cS`
2. **Intent Bridge** — HTTP server on music box that converts JSON → OSC
3. **Intent Emitter** — Video box emits intent on prompt changes

**Tidal Patch Structure:**
```haskell
-- Read external controls
let energy = cF 0.5 "energy"
    tension = cF 0.5 "tension"
    density = cF 0.5 "density"
    space = cF 0.3 "space"
    brightness = cF 0.5 "brightness"
    grit = cF 0.2 "grit"

-- Map to musical parameters
d1 $ sound "bd*4"
   # gain (range 0.5 1.0 energy)
   # lpf (range 200 8000 brightness)
   # room (range 0 0.8 space)
   # crush (range 16 4 grit)
```

**Bridge Endpoint:**
```python
@app.post("/intent")
async def update_intent(intent: MusicIntent):
    for key, value in intent.dict().items():
        osc_client.send_message("/ctrl", [key, float(value)])
    return {"status": "ok"}
```

### Phase 2: Arrangement Controls

**Goal:** Scene boundary actions (hush, solo, unsolo).

**Add to bridge:**
```python
@app.post("/playback/{action}")
async def playback_control(action: str, channel: Optional[int] = None):
    if action == "hush":
        osc_client.send_message("/hush", [])
    elif action == "solo" and channel:
        osc_client.send_message("/solo", [channel])
    # etc.
```

**Map hard cuts:**
- Video hard cut → POST `/playback/hush`
- After hush settles → POST `/playback/unmuteAll` or start new section

### Phase 3: Tidal Transitions

**Goal:** Soft cuts feel musical, not abrupt.

**Tidal transition functions:**
- `xfadeIn n` — Crossfade over n cycles
- `clutchIn n` — Degrade old, restore new over n cycles
- `interpolateIn n` — Morph control values between patterns

**Trigger via MCP or evaluated code:**
```haskell
-- On soft cut (4-cycle transition):
xfadeIn 4 $ d1 $ sound "new_pattern"
```

### Phase 4: Agent-Mediated Rewrites

**Goal:** Claude Code can propose pattern changes with human approval.

**MCP Server:** Use existing `tidal-cycles-mcp-server`:
- `tidal_eval` — Send pattern to channel
- `tidal_hush`, `tidal_solo`, `tidal_silence`
- `tidal_get_state`, `tidal_get_history`

**Workflow:**
1. Agent analyzes scene tags + narrative arc
2. Proposes new pattern (displayed to buddy)
3. Buddy approves → `tidal_eval` with transition
4. Buddy rejects → continue current pattern

**Human-in-the-loop UI:**
```
┌─────────────────────────────────────────────┐
│ Agent Proposal                              │
├─────────────────────────────────────────────┤
│ Scene: "chase intensifies"                  │
│                                             │
│ Suggested d1 pattern:                       │
│   sound "bd(5,8)" # gain 0.9 # crush 6      │
│                                             │
│ [Apply Now] [Apply Next Bar] [Reject]       │
└─────────────────────────────────────────────┘
```

## Integration with Video System

### Reserved Control Keys

Add to `FrameProcessor` reserved key handling:

```python
# _rcp_music_intent — emit music intent to bridge
if "_rcp_music_intent" in merged_updates:
    intent = merged_updates.pop("_rcp_music_intent")
    self._emit_music_intent(intent)

# _rcp_music_action — trigger arrangement action
if "_rcp_music_action" in merged_updates:
    action = merged_updates.pop("_rcp_music_action")
    self._emit_music_action(action)
```

### Playlist Integration

Extend playlist format to include music intent per prompt:

```yaml
# prompts.yaml
- prompt: "Neo-Tokyo skyline at night, neon lights reflecting"
  music:
    energy: 0.3
    tension: 0.2
    space: 0.7
    transition: soft

- prompt: "High-speed motorcycle chase through tunnels"
  music:
    energy: 0.85
    tension: 0.9
    space: 0.15
    transition: hard
```

### CLI Commands

```bash
# Manual intent override
video-cli music intent --energy 0.8 --tension 0.9

# Trigger action
video-cli music hush
video-cli music solo drums

# Load cue sheet
video-cli music cues load akira_cues.json
```

## Sync Considerations

### For MVP: Manual Start

- Start video and music together manually
- Accept small drift (usually imperceptible for ambient/reactive music)
- Cue boundaries re-sync intent

### Future: Ableton Link

Tidal supports Ableton Link for shared tempo/phase:
- All Link-enabled apps share musical timeline
- Quantized transitions land on beat
- Works across network

### Alternative: Chunk-Based Timing

- Video emits chunk index with each intent update
- Music box can interpolate/schedule based on chunk rate (~4 FPS)
- More precise than wall-clock, less complex than Link

## Files to Create

| File | Purpose |
|------|---------|
| `src/scope/integrations/tidal/bridge.py` | FastAPI HTTP→OSC bridge |
| `src/scope/integrations/tidal/client.py` | Client for video box to emit intent |
| `src/scope/integrations/tidal/schema.py` | Pydantic models for intent/actions |
| `examples/tidal/parametric_patch.tidal` | Reference Tidal patch |
| `examples/tidal/akira_cues.json` | Example cue sheet |

## Dependencies

- **Tidal Cycles** — https://tidalcycles.org/
- **SuperDirt** — Audio engine for Tidal
- **python-osc** — OSC client for bridge
- **tidal-cycles-mcp-server** — https://github.com/Benedict/tidal-cycles-mcp-server (optional, Phase 4)

## Open Questions

- [ ] **Network topology:** Same box or remote? Affects OSC routing and latency.
- [ ] **Sync mechanism:** Manual start, Ableton Link, or chunk-based?
- [ ] **Control surface:** Open Stage Control, custom UI, or CLI only?
- [ ] **Pattern banks:** How many pre-composed sections? Scene-tagged or mood-tagged?
- [ ] **Intent derivation:** Manual per-cue, or auto-derived from scene tags via LLM?

## References

- [Tidal OSC Controller Input](https://tidalcycles.org/docs/working-with-patterns/Controller_Input/)
- [Tidal Transitions](https://tidalcycles.org/docs/reference/transitions/)
- [Tidal Playback Controllers](https://userbase.tidalcycles.org/Playback_Controllers.html)
- [TidalCycles MCP Server](https://github.com/Benedict/tidal-cycles-mcp-server)
- [Open Stage Control + Tidal](https://club.tidalcycles.org/t/open-stage-control-tidalcycles/1283)
- [Ableton Link in Tidal](https://userbase.tidalcycles.org/Link_synchronisation.html)
