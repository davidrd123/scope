# Review 01 — Hardware Control Surface Proposal

> Date: 2025-12-27
> Note: `notes/proposals/hardware-control-surface.md` now incorporates the fixes described here; the remaining design constraint is how an external router sends continuous parameter updates.

Below is a correctness pass on `notes/proposals/hardware-control-surface.md` against what’s actually in the repo right now (REST + WebRTC data-channel control, `video-cli`, `app.py` endpoints, `apply_control_message`, `FrameProcessor` reserved keys, and the documented parameter contract in `docs/api/parameters.md`).

I’m focusing on **terminology + entry points + parameter names + what’s actually callable today**, and explicitly labeling anything that is *not* currently present.

---

## Key mismatches to fix

### 1) CLI entry points in the proposal don’t exist

**Proposal says:**

* `uv run video-cli control`
* `video-cli control --list-devices`

**Reality:**

* `video-cli` currently has: `state`, `run`, `pause`, `step`, `prompt`, `world`, `style (list/set/get)`, and `playlist (...)`.
* There is **no** `control` command and **no** device listing support in `src/scope/cli/video_cli.py`.

✅ Fix: Change “Entry Points” to either:

* “Run a standalone router script (to be created) that uses REST/WebRTC”, or
* “(Future) add `video-cli control-surface` subcommand”.

---

### 2) Wrong / inconsistent parameter names

**Proposal uses** (examples/tables/code):

* `denoising_steps`
* `kv_cache_bias`
* `crossfade_duration`
* `denoising_steps reset` semantics

**Reality (authoritative names today):**
From `docs/api/parameters.md` and server schema:

* `denoising_step_list` (list of descending ints)
* `kv_cache_attention_bias` (float)
* `transition: { target_prompts, num_steps, temporal_interpolation_method }`
* hard cut is `reset_cache: true`
* soft cut is **reserved key** `_rcp_soft_transition` (or REST `/api/v1/realtime/soft-cut`), not a “crossfade duration” param

✅ Fix: Replace those names everywhere in the proposal.

---

### 3) Snapshot/restore claims are not REST/CLI-available today

**Reality:**

* Snapshot/restore is supported in `FrameProcessor` via reserved keys:

  * `_rcp_snapshot_request`
  * `_rcp_restore_snapshot`
* But **REST endpoints for snapshot/restore are not implemented**, and `video-cli snapshot/restore` explicitly returns `not_implemented`.

Also, `apply_control_message` supports WebRTC protocol “type” messages:

* `{ "type": "snapshot_request" }`
* `{ "type": "restore_snapshot", "snapshot_id": "..." }`

✅ Fix: In the hardware proposal, mark snapshot/restore buttons as:

* **WebRTC data-channel only** (today), or
* “requires adding REST endpoints” (future)

---

### 4) World/beat/camera controls are not patch endpoints

**Reality:**

* `/api/v1/realtime/world` is **full replace**, not patch.
* `WorldState.beat` and `WorldState.camera` exist, but there is no dedicated REST endpoint like `/realtime/beat` or `/realtime/camera`.

✅ Fix: The doc should state that a control surface must:

* read current `world_state` from `/api/v1/realtime/state`,
* modify the field(s),
* send the entire new `world_state` back via `/api/v1/realtime/world`.

(Or in the future, add patch endpoints.)

---

### 5) “Unified InputRouter”, “SharedState”, config files are not in `src/`

**Reality:**

* No `InputRouter` implementation exists in `src/`.
* No `config/midi_map.yaml` or `config/streamdeck_layout.yaml` exists.
* This is fine as a proposal, but the doc currently reads like some of it is already wired.

✅ Fix: label these as **new components/files to be created**.

---

### 6) Optional dependency group `control = [...]` is not in `pyproject.toml`

**Reality:**

* There is **no** `[project.optional-dependencies]` section at all currently.

✅ Fix: change “Dependencies” section wording to “Proposed additions to `pyproject.toml`”.

---

### 7) A standalone router can’t write to the existing WebRTC data-channel

The WebRTC data-channel is owned by the connected client session; a separate process cannot currently inject data-channel messages into that session.

If we want high-rate continuous controls (MIDI faders/encoders) from a standalone router, we likely need either:
- a server-side “set parameters” endpoint that forwards to `apply_control_message`, or
- to run the router inside the same WebRTC client that already owns the data-channel.

Also note: opening a second WebRTC connection risks breaking REST control (REST endpoints don’t accept `session_id` today).

---

## Drop-in replacement: corrected + aligned `hardware-control-surface.md`

This is a revised version of the proposal that keeps the intent, but makes the “what exists today” boundary explicit and fixes names/entry points/references.

You can replace the file contents with the following:

````md
# Hardware Control Surface Integration

> Status: Draft (vetted against current Scope code)
> Date: 2025-12-27
> Related: `notes/plans/tui-director-console.md`, `notes/proposals/tidal-cycles-integration.md`

## Summary

Integrate physical hardware (Elgato Stream Deck, MIDI controllers) as ergonomic input surfaces for real-time video direction and audio performance.

**Important reality check (today):**
- Realtime control is **REST + WebRTC data-channel** driven.
- `video-cli` is a REST client for a *connected* WebRTC session.
- There is **no** existing “hardware router” module/command in `src/` yet — this proposal describes one to build.

---

## What exists today (authoritative control surfaces)

### Control planes

| Plane | Exists today? | How to use |
|------|---------------|------------|
| REST control (pause/run/step/prompt/world/style/playlist) | ✅ | `src/scope/server/app.py` + `video-cli` |
| WebRTC data-channel parameter updates | ✅ | `docs/api/parameters.md` (“Send Parameters”) |
| Snapshot / restore | ✅ (data-channel protocol + reserved keys) | `apply_control_message` supports message `type`; REST endpoints not implemented |
| “Hardware router” process (MIDI/StreamDeck → Scope) | ❌ | Proposed here |
| Device listing (`--list-devices`) | ❌ | Proposed |
| Optional `control` dependency group in `pyproject.toml` | ❌ | Proposed |

### Single-active-session constraint (REST)

Most REST endpoints call `get_active_session()` which requires:
- exactly **one** connected WebRTC session, otherwise REST control fails.

So a hardware control surface must assume:
- a WebRTC session is already established and connected.

---

## Canonical parameter names (don’t invent new ones)

### For WebRTC data-channel updates (authoritative)
See: `docs/api/parameters.md`

Common keys:
- `prompts: [{text, weight}]`
- `denoising_step_list: [1000, 750, 500, 250]` (descending ints)
- `noise_scale: float`
- `noise_controller: bool`
- `manage_cache: bool`
- `reset_cache: bool` (one-shot hard cut)
- `kv_cache_attention_bias: float`
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

A single “semantic” control (e.g., `intensity`) can fan out to:
- video parameters (e.g., `noise_scale`, `denoising_step_list`)
- audio intent (e.g., Tidal `energy`)
- UI state (TUI, Stream Deck button highlights)

This is *semantic control* — you control meaning, not one-off knobs.

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
| `stability` | adjust `kv_cache_attention_bias` | map to Tidal `density`/“tightness” |
| `transition_speed` | `transition.num_steps` OR soft cut duration (`/realtime/soft-cut`) | map to transition/crossfade |
| `scene_cut` | hard cut via `reset_cache: true` or `/realtime/hard-cut` | Tidal hush / pattern swap |

Notes:
- There is no `denoising_steps` scalar parameter; use `denoising_step_list`.
- There is no `kv_cache_bias`; use `kv_cache_attention_bias`.
- “Soft cut” is a *temporary override* via `_rcp_soft_transition` / `/realtime/soft-cut`.

---

## Stream Deck Integration (proposal)

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

---

## MIDI Controller Integration (proposal)

### Continuous controls (recommend WebRTC data-channel)
For smooth faders/encoders (10–60Hz updates), WebRTC data-channel is the best match,
because REST does not currently provide a “set arbitrary parameters” endpoint.

| Fader | Canonical video key (today) | Notes |
|------|------------------------------|------|
| 1 | `denoising_step_list` | Choose from presets (you cannot send a scalar “steps”) |
| 2 | `noise_scale` | float 0–1 |
| 3 | `kv_cache_attention_bias` | float 0.01–1.0 |
| 4 | `transition.num_steps` | affects prompt interpolation (if using `transition`) |

### Discrete MIDI notes (works via REST)
MIDI note-on can call REST endpoints:
- styles via `/api/v1/realtime/style`
- beat/camera via `/api/v1/realtime/world` (full replace)
- pause/run/step via `/api/v1/realtime/*`

### “Beat triggers” and “camera triggers”
These must change `WorldState` fields:
- `world_state.beat` values: `setup|escalation|climax|payoff|reset|transition`
- `world_state.camera` values: `close_up|medium|wide|low_angle|high_angle|tracking|static|establishing`

---

## Proposed Hardware Router (new component; not in repo yet)

A separate process that:
1. polls `/api/v1/realtime/state` for current state
2. translates hardware input into either:
   - REST calls (discrete actions)
   - WebRTC data-channel messages (continuous params + snapshot)
3. (optional) forwards audio intent to Tidal via OSC (see `tidal-cycles-integration.md`)

### State sync (today)
`GET /api/v1/realtime/state` returns:
- `paused`, `chunk_index`, `session_id`, `prompt`
- plus style-layer state: `world_state`, `active_style`, `compiled_prompt`

So a shared state model should align to those names (don’t invent new ones).

---

## Configuration files (proposed; not present today)

If implemented, these files would be new additions:
- `config/midi_map.yaml`
- `config/streamdeck_layout.yaml`

(Or store them under `notes/` or `examples/` if you want them non-runtime.)

---

## Entry points (corrected)

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

## Dependencies (corrected)

`pyproject.toml` currently does NOT define optional dependency groups for control surfaces.

If/when implementing this proposal, we would add something like:

```toml
[project.optional-dependencies]
control = [
  "mido>=1.3.0",
  "python-rtmidi>=1.5.0",
  "python-osc>=1.8.0",  # for Tidal intent forwarding
  "Pillow>=10.0.0",     # for Stream Deck image rendering
  # Stream Deck python library (exact package name TBD during implementation)
]
```

---

## Open Questions (updated)

1. **Continuous control transport**: add a REST endpoint for arbitrary params, or rely on WebRTC data-channel?

   * Today: WebRTC data-channel is the only “generic parameter update” path documented.
2. **WorldState patching**: do we add `/realtime/world/patch`?

   * Today: `/realtime/world` is full replace.
3. **Snapshot UX**: do we expose snapshot/restore via REST?

   * Today: snapshot/restore is data-channel only; CLI placeholders exist but are not implemented.

---

## References (internal + external)

Internal:

* `docs/api/parameters.md` (canonical parameter names)
* `src/scope/server/app.py` (REST realtime endpoints)
* `src/scope/server/webrtc.py` (`apply_control_message` message translation)
* `src/scope/server/frame_processor.py` (reserved keys + soft transition behavior)
* `src/scope/realtime/world_state.py` (beat/camera canonical values)

External:

* Stream Deck SDK: [https://developer.elgato.com/documentation/stream-deck/](https://developer.elgato.com/documentation/stream-deck/)
* mido (MIDI): [https://mido.readthedocs.io/](https://mido.readthedocs.io/)
* python-rtmidi: [https://spotlightkid.github.io/python-rtmidi/](https://spotlightkid.github.io/python-rtmidi/)
* python-osc: [https://python-osc.readthedocs.io/](https://python-osc.readthedocs.io/)

````

---

## Quick “what changed” summary (so you can sanity-check)
- Removed/rewrote the false claim that `video-cli control` exists.
- Corrected parameter names:
  - `denoising_steps` → `denoising_step_list`
  - `kv_cache_bias` → `kv_cache_attention_bias`
  - `crossfade_duration` → `transition.num_steps` / `/realtime/soft-cut` duration
- Marked snapshot/restore as **data-channel only today** (no REST/CLI).
- Explicitly stated `/realtime/world` is **full replace**.
- Marked optional dependencies and config files as **proposed**, not present.

If you want, I can also provide a smaller **unified diff** instead of the full replacement doc, but the above is designed to be copy/paste-able as a corrected, consistent version.
