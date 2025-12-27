```md
# Tidal Cycles Integration

> Status: Draft (vetted against current Scope control-plane behavior)
> Date: 2025-12-26 (rev: 2025-12-27)
> Research: `notes/research/2025-12-26/tidal/aoi_5pro_01.md`
> Related: `notes/concepts/narrative-engine.md`, `notes/plans/realtime-control-plane-tdd.md`,
>          `notes/proposals/hardware-control-surface.md`

## Summary

Synchronize live-coded music (Tidal Cycles) with real-time video generation by routing **music intent**
(continuous parameters + discrete arrangement actions) from the video system to the music system.

**Design goal:** treat audio as a *first-class sibling* of video prompting in the same narrative stack:

```

World State → Narrative / Intent Layer → ┬→ Video prompts → Scope realtime pipeline → Visual output
│
└→ Music intent  → Tidal Cycles (OSC/MCP)    → Audio output

```

This proposal is deliberately staged:
- **Start simple:** precomputed cue sheet + OSC parameter steering (no agent rewrites).
- **Add live control:** same intent interface, driven by runtime events (soft/hard cuts, prompt changes).
- **Only later:** pattern rewrites (MCP / evaluated Tidal code) with human approval.

---

## Reality check: what exists today in Scope (authoritative)

Scope realtime control is already chunk-boundary-driven and has a clear “reserved key” mechanism that
fits music intent cleanly.

### Control planes that exist today

- **REST endpoints** in `src/scope/server/app.py`:
  - `/api/v1/realtime/hard-cut` → forwards `{ "reset_cache": true }` (+ optional prompt)
  - `/api/v1/realtime/soft-cut` → forwards `{ "_rcp_soft_transition": { temp_bias, num_chunks } }` (+ optional prompt)
  - `/api/v1/realtime/world` → forwards `{ "_rcp_world_state": <WorldState> }`
  - `/api/v1/realtime/style` → forwards `{ "_rcp_set_style": <style_name> }`
  - `/api/v1/realtime/run`, `/pause`, `/step` (step implemented via `_rcp_step`)
- **WebRTC data-channel** messages are translated by `apply_control_message` in `src/scope/server/webrtc.py`:
  - `{ "type":"snapshot_request" }` → `{ "_rcp_snapshot_request": true }`
  - `{ "type":"restore_snapshot", "snapshot_id":"..." }` → `{ "_rcp_restore_snapshot": {...} }`
  - `{ "type":"step" }` → `{ "_rcp_step": true }`
- **Reserved keys are consumed inside `FrameProcessor`** (`src/scope/server/frame_processor.py`) and are **not forwarded**
  to the pipeline. This is the correct pattern for music integration because we don’t want audio-control
  fields to leak into pipeline kwargs.
- **Soft cut semantics are already defined**:
  - `_rcp_soft_transition` temporarily overrides `kv_cache_attention_bias` for N chunks and restores.
  - Inputs are coerced + clamped (bias ∈ [0.01, 1.0], chunks ∈ [1, 10]).
  - Explicit `kv_cache_attention_bias` updates cancel the soft transition.

**Implication for this proposal:** we should align audio events to the same chunk-boundary semantics and reuse the same
“reserved keys consumed in `FrameProcessor`” pattern (or an equivalent chunk-boundary emitter).

---

## The vision

This is an A/V performance instrument: **meaning → parameters**.

- Video side is already treated as a **playlist / edit decision list** (prompts + cuts).
- Music side should mirror that:
  - continuous steering during **soft cuts / plasticity windows**
  - structural edits on **hard cuts / section changes**
  - later, agent-assisted rewrites (human-in-the-loop)

---

## Architecture

### Two-box topology (recommended default)

This matches the current pragmatic reality: remote GPU for video, local music machine for Tidal.

```

┌────────────────────────────────────────────────────────────────────┐
│ VIDEO BOX (Remote GPU)                                             │
│                                                                    │
│ Scope server + realtime pipeline                                    │
│  - REST / data-channel control plane                                │
│  - WorldState + Style + PromptCompiler                              │
│  - (Proposed) MusicIntentEmitter                                    │
│                                                                    │
│                emits JSON intent events (HTTP/WebSocket)            │
└───────────────────────────────────┬────────────────────────────────┘
│
▼
┌────────────────────────────────────────────────────────────────────┐
│ MUSIC BOX (Local)                                                  │
│                                                                    │
│ Intent Bridge (HTTP→OSC)                                           │
│  - validates + clamps                                               │
│  - smooths / rate-limits                                             │
│  - logs / replays                                                   │
│                                                                    │
│ Tidal Cycles (GHCi) listens on localhost OSC                        │
│  - /ctrl  <key> <value>                                             │
│  - /hush, /solo, /muteAll, ... (playback controller OSC)            │
│                                                                    │
│ SuperDirt / audio output                                            │
└────────────────────────────────────────────────────────────────────┘

````

**Why keep OSC on localhost:** avoids exposing Tidal’s OSC ports to the network.
Expose only the Intent Bridge (HTTP) on a trusted LAN/VPN.

---

## Control planes

| Plane | Mechanism | Latency | Best for |
|------|-----------|---------|----------|
| **Parameter steering** | OSC `/ctrl` → Tidal `cF/cS/cI` | low | continuous morphing during soft cuts |
| **Arrangement control** | OSC playback controllers (`/hush`, `/solo`, …) | low | scene boundary toggles, hard cut punctuation |
| **Pattern rewrites** | MCP `tidal_eval` / evaluated Tidal code | higher | hard cuts, section changes, “new motif” edits |

**Key design principle:** treat pattern rewrites as *structural* and gate them (approval + boundaries),
while parameter steering stays *safe* and continuous.

---

## Music intent vocabulary

Keep it small and composable. This is the “semantic surface” you will actually rehearse with.

### Continuous controls (normalized 0–1)

| Control | Meaning | Typical mapping ideas |
|---|---|---|
| `energy` | macro intensity | gain, drum density, distortion amount |
| `tension` | narrative pressure | modulation depth, dissonance, syncopation |
| `density` | event frequency / layering | subdivision, polyphony, extra voices |
| `space` | room / atmosphere | reverb send, delay feedback, stereo width |
| `brightness` | tonal brightness | filter cutoff / harmonic emphasis |
| `grit` | texture / edge | crush / saturation / noise |

**Optional later:** `valence` (-1..1), `focus_amount`, `swing`, `pulse`, etc.  
But do not add these until the first 6 feel controllable.

### Discrete controls

| Control | Values | Mapping |
|---|---|---|
| `focus` | `drums`, `bass`, `pads`, `full` | solo/unsolo logic across channels |
| `action` | `hush`, `muteAll`, `unmuteAll`, `solo`, `unsoloAll` | playback controller OSC |

---

## Mapping to Scope video events

This proposal intentionally aligns to real Scope controls:

| Video event | Scope mechanism today | Music response |
|---|---|---|
| Prompt change | `prompts=[...]` | update intent values (and maybe focus) |
| Soft cut | `_rcp_soft_transition` (REST `/realtime/soft-cut`) | interpolate controls over N chunks/cycles; optionally trigger Tidal transition |
| Hard cut | `reset_cache=true` (REST `/realtime/hard-cut`) | `/hush` + (later) pattern bank switch |
| Style change | `_rcp_set_style` | optional: change “palette” (EQ, kit bank, FX macro) |
| World state update | `_rcp_world_state` | derive intent (future) or look up cue metadata (MVP) |

**Note:** Scope “soft cut” is *specifically* a temporary bias override in video.  
Music should interpret that as a “plasticity window” for smoother morphing, not as a hard scene break.

---

## Cue sheet format

This remains the best “offline rehearsal” artifact. It is also future training data:
“given these tags and transitions, these intent curves worked.”

```json
{
  "version": "1.0",
  "track": "akira_rankin_v0",
  "cues": [
    {
      "cue_id": "akira_001",
      "prompt_id": "playlist_index_1",
      "t_start_s": 0.0,
      "t_end_s": 8.5,
      "transition": { "type": "soft", "cycles": 4 },
      "tags": ["neo_tokyo", "night", "neon", "establishing"],
      "music_intent": {
        "energy": 0.3,
        "tension": 0.2,
        "density": 0.4,
        "space": 0.7,
        "brightness": 0.5,
        "grit": 0.2
      },
      "arrangement": { "focus": "pads", "action": null }
    },
    {
      "cue_id": "akira_002",
      "prompt_id": "playlist_index_2",
      "t_start_s": 8.5,
      "t_end_s": 15.2,
      "transition": { "type": "hard", "cycles": 0 },
      "tags": ["chase", "motorbikes", "speed"],
      "music_intent": {
        "energy": 0.85,
        "tension": 0.9,
        "density": 0.8,
        "space": 0.15,
        "brightness": 0.7,
        "grit": 0.6
      },
      "arrangement": { "focus": "drums+bass", "action": "hush" }
    }
  ]
}
````

---

## Concrete interfaces

### 1) Video → Music machine: Event envelope (recommended)

Rather than multiple ad-hoc endpoints, standardize on one event contract.
This makes later additions (agent proposals, replay, logging) much easier.

```json
{
  "type": "music_intent",        // "music_intent" | "music_action" | "cue"
  "source": "scope",
  "ts_unix_ms": 1735260000000,
  "chunk_index": 1234,           // optional but recommended when available
  "cue_id": "akira_042",         // optional
  "payload": { ... }
}
```

Payloads:

**music_intent**

```json
{
  "controls": {
    "energy": 0.7,
    "tension": 0.4,
    "density": 0.6,
    "space": 0.2,
    "brightness": 0.5,
    "grit": 0.1
  },
  "ramp": { "mode": "linear", "seconds": 2.0 }   // optional smoothing hint
}
```

**music_action**

```json
{ "action": "hush", "focus": "drums" }
```

**cue**

```json
{
  "cue_id": "akira_042",
  "prompt_id": "playlist_index_42",
  "transition": { "type": "soft", "num_chunks": 3 }
}
```

### 2) Music machine: Intent Bridge HTTP API (MVP)

Minimal FastAPI contract:

* `POST /intent` → updates one or more continuous controls (converted to OSC `/ctrl`)
* `POST /action` → triggers discrete playback action (converted to OSC `/hush`, `/solo`, etc.)
* `POST /event` → optional unified endpoint (accepts the envelope above)

Example MVP endpoints (intentionally tiny):

```python
@app.post("/intent")
async def intent_update(req: MusicIntentUpdate): ...

@app.post("/action")
async def action(req: MusicAction): ...
```

Bridge responsibilities:

* **allowlist keys** (`energy`, `tension`, …)
* clamp to valid ranges
* optional ramping / smoothing
* logging + “last known intent”
* do not block Tidal: best-effort OSC send

---

## Alignment with Scope control-plane semantics

### Where the music intent should be emitted

**Goal:** if a prompt/world/style update is applied at a chunk boundary, the corresponding audio update should be emitted at that same boundary (or immediately after) and should be deterministic.

There are two viable designs; both align with today’s architecture:

#### Option A (recommended): reserved keys consumed in `FrameProcessor`

Add reserved key handlers analogous to snapshot/step:

* `_rcp_music_intent` (dict of continuous controls, plus optional ramp hints)
* `_rcp_music_action` (discrete action)
* `_rcp_music_cue` (metadata: cue_id, transition, tags, etc.)

These keys:

* flow through `parameters_queue` (thread-safe)
* are drained/merged (mailbox semantics)
* are consumed in `process_chunk()` and **never forwarded to pipeline**

**Important implementation requirement:** do not do network IO in the hot path.
Instead:

* `FrameProcessor` enqueues events to a lightweight emitter queue
* a separate emitter thread (or async task on the server loop) performs HTTP sends

This preserves realtime FPS and avoids stalls.

#### Option B: emit from the REST layer

Add Scope-side code so `/realtime/soft-cut` and `/realtime/hard-cut` also call a music client directly.

This is simpler conceptually, but:

* it bypasses chunk-boundary ordering and can drift relative to actual generation boundaries
* it’s less complete (world/style changes can come via data-channel too)

So Option B is okay for early experiments, but Option A is the “correct” alignment.

### Event ordering

Scope already cares about ordering (see `notes/plans/realtime-control-plane-tdd.md`).
For music, keep it simple:

1. style change
2. world state update (and derived prompt compile)
3. prompt/transition changes
4. hard/soft cut actions (if present)

Practically: “music intent” should be computed from the *post-update* WorldState + style,
then actions applied as punctuation.

---

## How this connects to the Narrative Engine (future-facing, but staged)

The Narrative Engine notes introduce layers that are ideal *sources* of music intent later:

* **Trajectories:** tension building as a continuous parameter → maps to `tension`, `density`, `brightness`
* **Information topology:** irony/suspense → maps to `tension` and arrangement focus
* **Intent/subtext/surface:** “what should audience feel” → directly maps to the intent vector

**But MVP does not require any of that.**
MVP uses:

* manual cue sheet values
* optional heuristics from simple tags (`["chase"]` → energy↑)

Design requirement now: keep the intent interface stable so the derivation can evolve without changing consumers.

---

## Implementation phases (practical staging)

### Phase 0: Offline scoring workflow (first 5–10 minutes of Akira)

**Goal:** rehearse the control vocabulary and patch without networking complexity.

* Write a cue sheet aligned to your prompt playlist (manual values)
* Run a script on the music machine that “plays” the cue sheet (timestamped or step-advanced)
* Iterate: adjust cue values, replay, repeat

Outcome: a stable patch + a set of cue values that “works.”

### Phase 1: OSC parameter steering (MVP integration)

**Goal:** make the patch steerable by external values.

* Tidal patch reads `cF/cS/cI` controls (energy/tension/density/space/brightness/grit)
* Intent Bridge receives HTTP JSON → sends OSC `/ctrl` to localhost Tidal

Reference bridge behavior (conceptual):

```python
@app.post("/intent")
async def update_intent(intent: MusicIntent):
    for key, value in intent.dict().items():
        osc_client.send_message("/ctrl", [key, float(value)])
    return {"status": "ok"}
```

### Phase 2: Arrangement controls

**Goal:** make hard cuts and focus shifts feel intentional.

Bridge adds:

* `/action` that emits `/hush`, `/muteAll`, `/unmuteAll`, `/solo`, `/unsoloAll`

Mapping:

* video hard cut → `/hush` (and optionally start next section after a beat)
* focus changes (“close-up” / “wide”) → solo/unsolo sets

### Phase 3: Musical soft cuts (transitions)

**Goal:** make “plasticity windows” feel musical, not abrupt.

Use transition functions inside Tidal:

* `xfadeIn n`
* `clutchIn n`
* `interpolateIn n`

Trigger mechanism (choose one):

* manual (buddy runs small snippets)
* evaluated code via MCP/file-mode (Phase 4 scaffolding)

### Phase 4: Agent-mediated pattern rewrites (human-in-the-loop)

**Goal:** allow structural changes at cue boundaries, safely.

* Use an MCP server (e.g. `tidal_eval`, `tidal_hush`, `tidal_get_state/history`)
* Agent proposes pattern changes; buddy approves
* Apply on cue boundary with a transition (avoid “LLM surprise” mid-bar)

---

## Integration points in the video system (Scope)

### Proposed reserved keys

Add (proposal; not present today):

```python
# Consumed in FrameProcessor; never forwarded to pipeline
"_rcp_music_intent": { "energy": 0.7, "tension": 0.2, ... }
"_rcp_music_action": { "action": "hush", "focus": "drums" }
"_rcp_music_cue":    { "cue_id": "akira_042", "transition": {...}, "tags": [...] }
```

### Where to trigger them

* On **style change** (`_rcp_set_style`): optionally send an audio “palette switch”
* On **world state update** (`_rcp_world_state`):

  * MVP: lookup cue sheet by `cue_id` or playlist index and emit stored intent
  * later: compile intent from WorldState + style
* On **soft cut** (`_rcp_soft_transition`): emit a ramp hint to the bridge
* On **hard cut** (`reset_cache`): emit a `/hush` action

---

## CLI and tooling ideas (staged, don’t block MVP)

These are optional conveniences; MVP works without touching `video-cli`.

### Music machine tooling (recommended early)

* `python tools/play_cue_sheet.py cue.json` (offline playback)
* `python tools/send_intent.py --energy 0.8 --tension 0.9` (manual override)
* `python tools/hush.py` (manual punctuation)

### Video machine / unified tooling (later)

Potential future commands:

* `video-cli music intent --energy 0.8 ...`
* `video-cli music hush`
* `video-cli music cues load akira.json`

But note: `video-cli` today is primarily a REST client for a *single active WebRTC session*.
Keeping music tooling independent early avoids fighting session constraints.

---

## Sync considerations

### MVP default: manual start + cue boundary resync

* Start video and music together manually
* Let cue boundaries re-align intent (small drift is tolerable for ambient/reactive scoring)

### Later options (pick only if needed)

* **Chunk-index based scheduling**: include `chunk_index` in events and schedule ramps by chunk count
  (the bridge can “play back” control changes keyed to chunk index)
* **Ableton Link**: possible shared musical timeline; consider only after the basic system feels good

---

## Reliability and safety

* **Do not block realtime generation**: all HTTP sends must be async / background from the video thread.
* **Rate-limit**: cap intent sends (e.g. 10–30 Hz max) and coalesce updates (mailbox semantics).
* **Allowlist + clamp**: never accept arbitrary OSC paths from the network.
* **Auth**: optional bearer token for the bridge if exposed beyond localhost.
* **Logging**: log events so you can replay a “performance” offline.

---

## Files to create (proposal)

| File                                      | Purpose                                                       |
| ----------------------------------------- | ------------------------------------------------------------- |
| `src/scope/integrations/tidal/schema.py`  | Pydantic models: `MusicIntent`, `MusicAction`, event envelope |
| `src/scope/integrations/tidal/client.py`  | HTTP client used by video box emitter                         |
| `src/scope/integrations/tidal/emitter.py` | background sender (queue + coalesce + retry policy)           |
| `src/scope/integrations/tidal/bridge.py`  | FastAPI HTTP→OSC bridge (runs on music machine)               |
| `examples/tidal/parametric_patch.tidal`   | reference parametric patch                                    |
| `examples/tidal/akira_cues.json`          | example cue sheet aligned to prompts                          |
| `tools/play_cue_sheet.py`                 | offline cue playback (music machine)                          |
| `tools/send_intent.py`                    | manual intent override                                        |

---

## Open questions (with recommended defaults)

1. **Network topology:** same box or remote?

   * Default: video remote, music local; keep Tidal OSC on localhost; expose only HTTP bridge.

2. **Sync mechanism:** manual, Link, or chunk-based?

   * Default: manual start + cue boundary resync; add chunk_index later if needed.

3. **Derivation of intent:** manual vs heuristics vs narrative engine compile?

   * Default: manual cue sheet for first 5–10 minutes; then add simple tag heuristics; then compile from WorldState later.

4. **Pattern rewrites:** when?

   * Default: only after Phase 1–2 feels musically solid; keep human-in-loop; gate to boundaries.

5. **Where to emit:** FrameProcessor reserved keys vs REST-side emission?

   * Default: FrameProcessor reserved keys + background emitter (best alignment with chunk semantics).

---

## References

* Tidal controller input: [https://tidalcycles.org/docs/working-with-patterns/Controller_Input/](https://tidalcycles.org/docs/working-with-patterns/Controller_Input/)
* Tidal transitions: [https://tidalcycles.org/docs/reference/transitions/](https://tidalcycles.org/docs/reference/transitions/)
* Playback controllers: [https://userbase.tidalcycles.org/Playback_Controllers.html](https://userbase.tidalcycles.org/Playback_Controllers.html)
* Ableton Link in Tidal: [https://userbase.tidalcycles.org/Link_synchronisation.html](https://userbase.tidalcycles.org/Link_synchronisation.html)
* TidalCycles MCP Server (optional, Phase 4): [https://github.com/Benedict/tidal-cycles-mcp-server](https://github.com/Benedict/tidal-cycles-mcp-server)

```

If you want, I can also produce a **small “Scope alignment appendix”** inside this same doc that lists the exact REST endpoints and the exact reserved keys currently implemented (as a checklist table), but I kept the above focused on design + interfaces + staging so it stays readable as a proposal.
```
