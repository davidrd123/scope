# Tidal Cycles Integration

> Status: Draft (vetted against current Scope control-plane behavior)
> Date: 2025-12-26 (rev: 2025-12-27)
> Research: `notes/research/2025-12-26/tidal/aoi_5pro_01.md`
> Review: `notes/proposals/tidal-cycles-integration/oai_5pro01.md`
> Related: `notes/proposals/hardware-control-surface.md`

## Summary

Synchronize live-coded music (Tidal Cycles) with real-time video generation by routing **music intent** (continuous parameters + discrete arrangement actions) from the video system to the music system.

**Design goal:** treat audio as a *first-class sibling* of video prompting in the same narrative stack.

This proposal is deliberately staged:
- **Start simple:** precomputed cue sheet + OSC parameter steering (no agent rewrites)
- **Add live control:** same intent interface, driven by runtime events (soft/hard cuts, prompt changes)
- **Only later:** pattern rewrites (MCP / evaluated Tidal code) with human approval

## The Vision

```
World State → Narrative Intent → ┬→ Video Prompts → Krea Realtime → Visual Output
                                 │
                                 └→ Music Intent  → Tidal Cycles → Audio Output
```

The same narrative intention layer that compiles world state into video prompts also emits music intent — a small vocabulary of continuous parameters (energy, tension, space) plus discrete actions (solo, hush, pattern switch). This keeps video and music semantically aligned without tight temporal coupling.

---

## Reality Check: What Exists Today in Scope

Scope realtime control is already chunk-boundary-driven and has a clear "reserved key" mechanism that fits music intent cleanly.

### Control planes that exist today

- **REST endpoints** in `src/scope/server/app.py`:
  - `/api/v1/realtime/hard-cut` → forwards `{ "reset_cache": true }` (+ optional prompt)
  - `/api/v1/realtime/soft-cut` → forwards `{ "_rcp_soft_transition": { temp_bias, num_chunks } }` (+ optional prompt)
  - `/api/v1/realtime/world` → forwards `{ "_rcp_world_state": <WorldState> }`
  - `/api/v1/realtime/style` → forwards `{ "_rcp_set_style": <style_name> }`
- `/api/v1/realtime/run`, `/api/v1/realtime/pause`, `/api/v1/realtime/step` (step uses a reserved key: `_rcp_step`)
- **WebRTC data-channel** messages are translated by `apply_control_message` in `src/scope/server/webrtc.py` (examples: `snapshot_request` → `_rcp_snapshot_request`, `restore_snapshot` → `_rcp_restore_snapshot`, `step` → `_rcp_step`)
- **Reserved keys are consumed inside `FrameProcessor`** and are **not forwarded** to the pipeline — this is the correct pattern for music integration
- **Soft cut semantics are already defined**:
  - `_rcp_soft_transition` temporarily overrides `kv_cache_attention_bias` for N chunks and then restores.
  - Inputs are coerced/clamped (bias ∈ `[0.01, 1.0]`, chunks ∈ `[1, 10]`).
  - An explicit `kv_cache_attention_bias` update arriving during an active soft transition cancels it (so we don’t restore over a user override).

**Implication:** align audio events to the same chunk-boundary semantics and reuse the "reserved keys consumed in FrameProcessor" pattern.

---

## Architecture

### Control Planes

| Plane | Mechanism | Latency | Use Case |
|-------|-----------|---------|----------|
| **Parameter steering** | OSC `/ctrl` → Tidal `cF/cS/cI` | ~10ms | Continuous morphing during soft cuts |
| **Arrangement control** | OSC `/solo`, `/hush`, `/muteAll` | ~10ms | Scene boundary toggles |
| **Pattern rewrites** | MCP `tidal_eval` | ~100ms | Hard cuts, section changes |

### Network Topology (Two-Box, Recommended)

```
┌────────────────────────────────────────────────────────────────────┐
│ VIDEO BOX (Remote GPU)                                             │
│                                                                    │
│ Scope server + realtime pipeline                                   │
│  - REST / data-channel control plane                               │
│  - WorldState + Style + PromptCompiler                             │
│  - (Proposed) MusicIntentEmitter                                   │
│                                                                    │
│                emits JSON intent events (HTTP/WebSocket)           │
└───────────────────────────────┬────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│ MUSIC BOX (Local)                                                  │
│                                                                    │
│ Intent Bridge (HTTP→OSC)                                           │
│  - validates + clamps                                              │
│  - smooths / rate-limits                                           │
│  - logs / replays                                                  │
│                                                                    │
│ Tidal Cycles (GHCi) listens on localhost OSC                       │
│  - /ctrl  <key> <value>                                            │
│  - /hush, /solo, /muteAll, ... (playback controller OSC)           │
│                                                                    │
│ SuperDirt / audio output                                           │
└────────────────────────────────────────────────────────────────────┘
```

**Why keep OSC on localhost:** avoids exposing Tidal's OSC ports to the network. Expose only the Intent Bridge (HTTP) on a trusted LAN/VPN.

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

This proposal aligns to real Scope controls:

| Video Event | Scope Mechanism | Music Response |
|-------------|-----------------|----------------|
| **Prompt change** | `prompts=[...]` | Update intent values (OSC `/ctrl`) |
| **Soft cut** | `_rcp_soft_transition` (REST `/realtime/soft-cut`) | Interpolate controls over N chunks; optionally trigger Tidal transition |
| **Hard cut** | `reset_cache=true` (REST `/realtime/hard-cut`) | `/hush` + (later) pattern bank switch |
| **Style change** | `_rcp_set_style` | Optional: change "palette" (EQ, kit bank, FX macro) |
| **World state update** | `_rcp_world_state` | Derive intent or look up cue metadata |

**Note:** Scope "soft cut" is specifically a temporary bias override in video. Music should interpret that as a "plasticity window" for smoother morphing, not as a hard scene break.

---

## Event Envelope Format

Standardize on one event contract for video→music communication:

```json
{
  "type": "music_intent",        // "music_intent" | "music_action" | "cue"
  "source": "scope",
  "ts_unix_ms": 1735260000000,
  "chunk_index": 1234,           // optional but recommended
  "cue_id": "akira_042",         // optional
  "payload": { ... }
}
```

**Payloads:**

`music_intent`:
```json
{
  "controls": { "energy": 0.7, "tension": 0.4, "density": 0.6, ... },
  "ramp": { "mode": "linear", "seconds": 2.0 }   // optional smoothing hint
}
```

`music_action`:
```json
{ "action": "hush", "focus": "drums" }
```

---

## Alignment with Scope Control-Plane Semantics

### Where the music intent should be emitted

**Goal:** if a prompt/world/style update is applied at a chunk boundary, the corresponding audio update should be emitted at that same boundary.

#### Option A (recommended): Reserved keys consumed in FrameProcessor

Add reserved key handlers analogous to snapshot/step:

```python
# Consumed in FrameProcessor; never forwarded to pipeline
"_rcp_music_intent": { "energy": 0.7, "tension": 0.2, ... }
"_rcp_music_action": { "action": "hush", "focus": "drums" }
"_rcp_music_cue":    { "cue_id": "akira_042", "transition": {...} }
```

These keys:
- Flow through `parameters_queue` (thread-safe)
- Are drained/merged (mailbox semantics)
- Are consumed in `process_chunk()` and **never forwarded to pipeline**

**Important:** do not do network IO in the hot path. Instead:
- `FrameProcessor` enqueues events to a lightweight emitter queue
- A separate emitter thread performs HTTP sends

#### Option B: Emit from the REST layer

Add Scope-side code so `/realtime/soft-cut` and `/realtime/hard-cut` also call a music client directly.

Simpler but: bypasses chunk-boundary ordering and can drift relative to actual generation boundaries.

**Recommendation:** Option A for correct alignment; Option B okay for early experiments.

### Event ordering (recommended)

Keep ordering deterministic so “what you heard” matches “what you saw” when replaying logs:

1. Apply **style** changes (palette)
2. Apply **world state** changes (and any prompt compilation they trigger)
3. Apply **prompt / transition** changes
4. Apply **hard/soft cut** actions (punctuation)
5. Emit **music intent** / **music action** events (derived from the post-update state)

Practical rule: compute the intent vector from the *post-update* WorldState + style, then treat actions (`hush`, `solo`, etc.) as punctuation on top.

---

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

### Phase 0: Offline Scoring Workflow (First 5-10 Minutes of Akira)

**Goal:** Rehearse the control vocabulary and patch without networking complexity.

**Steps:**
1. Write a cue sheet aligned to your prompt playlist (manual values)
2. Run a script on the music machine that "plays" the cue sheet (timestamped or step-advanced)
3. Iterate: adjust cue values, replay, repeat

**Outcome:** A stable Tidal patch + a set of cue values that "works" before adding network integration.

---

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

## MVP Acceptance Checks (Phases 0–2)

Keep this deliberately small; success is “rehearsable” A/V control, not maximal automation.

- **Phase 0 (offline scoring):** cue sheet playback steers a parametric Tidal patch repeatably (no networking, no live rewrites).
- **Phase 1 (bridge + intent):**
  - Sending `POST /intent` updates (`energy`, `tension`, etc.) audibly changes the patch within a short window (≲ 1–2 musical cycles).
  - Bridge rejects unknown keys and clamps out-of-range values.
  - Video-side emitter is non-blocking (no FPS drop; no stalls due to network IO).
- **Phase 2 (arrangement actions):** hard cut → `hush` is reliable; focus changes are deterministic (solo/unsolo behavior is predictable).

Explicitly out of scope for MVP: automated pattern rewrites, Link-based beat quantization, and “agents can freely eval Tidal code”.

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

**Future extension:** attach music intent metadata to playlist entries.

**MVP recommendation:** keep the existing caption-file playlist format unchanged and use a separate cue sheet keyed by `playlist_index` (or by a stable prompt ID/hash). This avoids a playlist-format migration just to get audio control.

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

**Optional (post-MVP):** a `video-cli music ...` wrapper is nice ergonomics, but it’s not required to validate the architecture. Early on, prefer a standalone music-machine script (or `curl`) so we don’t couple audio tooling to WebRTC session state.

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

## Reliability and Safety

- **Do not block realtime generation:** All HTTP sends must be async / background from the video thread
- **Rate-limit:** Cap intent sends (10-30 Hz max) and coalesce updates (mailbox semantics)
- **Allowlist + clamp:** Never accept arbitrary OSC paths from the network
- **Auth:** Optional bearer token for the bridge if exposed beyond localhost
- **Logging:** Log events so you can replay a "performance" offline (ideally keyed by `chunk_index` so it can line up with video timelines)
  - Future: integrate audio-event logging with session timeline export so a single “recording” can replay both video prompts/cuts and music intent.

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/scope/integrations/tidal/schema.py` | Pydantic models: `MusicIntent`, `MusicAction`, event envelope |
| `src/scope/integrations/tidal/client.py` | HTTP client used by video box emitter |
| `src/scope/integrations/tidal/emitter.py` | Background sender (queue + coalesce + retry policy) |
| `src/scope/integrations/tidal/bridge.py` | FastAPI HTTP→OSC bridge (runs on music machine) |
| `notes/proposals/tidal-cycles-integration/examples/parametric_patch.tidal` | Reference parametric patch |
| `notes/proposals/tidal-cycles-integration/examples/akira_cues.json` | Example cue sheet aligned to prompts |
| `scripts/tidal_play_cue_sheet.py` | Offline cue playback (music machine) |
| `scripts/tidal_send_intent.py` | Manual intent override (music machine) |

## Dependencies

- **Tidal Cycles** — https://tidalcycles.org/
- **SuperDirt** — Audio engine for Tidal
- **python-osc** — OSC client for bridge
- **tidal-cycles-mcp-server** — https://github.com/Benedict/tidal-cycles-mcp-server (optional, Phase 4)

## Open Questions (with Recommended Defaults)

1. **Network topology:** Same box or remote?
   - **Default:** Video remote, music local; keep Tidal OSC on localhost; expose only HTTP bridge

2. **Sync mechanism:** Manual, Link, or chunk-based?
   - **Default:** Manual start + cue boundary resync; add `chunk_index` later if needed

3. **Derivation of intent:** Manual vs heuristics vs narrative engine?
   - **Default:** Manual cue sheet for first 5-10 minutes; then add simple tag heuristics; then compile from WorldState later

4. **Pattern rewrites:** When?
   - **Default:** Only after Phase 1-2 feels musically solid; keep human-in-loop; gate to boundaries

5. **Where to emit:** FrameProcessor reserved keys vs REST-side emission?
   - **Default:** FrameProcessor reserved keys + background emitter (best alignment with chunk semantics)

## References

- [Tidal OSC Controller Input](https://tidalcycles.org/docs/working-with-patterns/Controller_Input/)
- [Tidal Transitions](https://tidalcycles.org/docs/reference/transitions/)
- [Tidal Playback Controllers](https://userbase.tidalcycles.org/Playback_Controllers.html)
- [TidalCycles MCP Server](https://github.com/Benedict/tidal-cycles-mcp-server)
- [Open Stage Control + Tidal](https://club.tidalcycles.org/t/open-stage-control-tidalcycles/1283)
- [Ableton Link in Tidal](https://userbase.tidalcycles.org/Link_synchronisation.html)
