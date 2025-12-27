This is a solid proposal—clear separation of concerns, staged rollout, and (most importantly) it keeps the *semantic* layer (“music intent”) small and composable, which matches the “meaning-first → derived layers” direction in your narrative-engine notes. 

Below are the main tweaks I’d make so the doc is both *implementable* and *accurate to how Tidal’s control + transition APIs actually behave*, plus a few upgrades that will make Phase 1–2 feel great without having to jump to full pattern rewrites right away.

---

## What’s already strong

* **Intent vocabulary is the right shape**: continuous 0–1 controls + a few discrete actions makes it robust under latency/jitter and easy to drive from “world state” or scene tags.
* **Control planes are staged correctly**: start with `/ctrl` steering, add playback controllers next, save “eval” (MCP) for later.
* **Cue-sheet thinking fits your “semi-offline rehearsal loop”**: you can iterate scoring against a repeatable segment (5–10 minutes of Akira) without needing perfect clock sync.

---

## High-priority corrections / clarifications

### 1) `/ctrl` details (port, payload, `cF` signature)

Tidal’s controller input listens (by default) on **127.0.0.1:6010**, and expects OSC path **`/ctrl`** with **two values**: key + value. `cF` takes a **default value first**, then the key string. ([Tidal Cycles][1])

So this part of your Phase 1 patch is correct in *shape*:

```haskell
let energy = cF 0.5 "energy"
```

…and your bridge idea of sending:

```python
osc_client.send_message("/ctrl", [key, float(value)])
```

is exactly aligned with the docs (python-osc will handle OSC typetags for you). ([Tidal Cycles][1])

**Doc tweak suggestion:** in your “Parameter steering” plane, explicitly state that OSC goes to Tidal’s **control port (6010)**, not SuperDirt’s **audio/synth port (often 57120)**. That separation matters operationally. ([Tidal Club][2])

---

### 2) Playback controllers are real, and they share the same OSC interface

The OSC “playback controller” actions you listed are correct: `/mute`, `/unmute`, `/solo`, `/unsolo`, `/muteAll`, `/unmuteAll`, `/unsoloAll`, `/hush`. They take either a **number** (`/solo 3` → `d3`) or a **string** if you use named patterns. ([TidalCycles Userbase][3])

**Important nuance:** these controls exist on the *same OSC interface used for controller input*, and (by default) also go to **port 6010**. ([Tidal Cycles][4])

So Phase 2 can stay entirely OSC-based without extra plumbing.

---

### 3) Your transitions snippet needs a syntax fix

This is the biggest “copy/paste bug” in the proposal:

You wrote:

```haskell
xfadeIn 4 $ d1 $ sound "new_pattern"
```

But Tidal transitions apply *to an existing running stream* by ID. The canonical usage is:

```haskell
-- changes d1
xfadeIn 1 8 $ s "arpy*8" # n (run 8)
```

where the first argument is the pattern ID (e.g., `1` for `d1`) and the second is the number of cycles for the transition. ([Tidal Cycles][5])

**Doc replacement snippet (copy/paste safe):**

```haskell
-- start something on d1
d1 $ s "bd(3,8) drum*4"

-- soft cut: crossfade d1 over 4 cycles
xfadeIn 1 4 $ s "arpy*8" # n (run 8)

-- soft cut alternative: clutch d1 over 4 cycles
clutchIn 1 4 $ s "[hh*4, odx(3,8)]"
```

([Tidal Cycles][5])

Also: since you want “Apply Next Bar / Apply Now” semantics for the buddy console, look at `jumpIn'`, `jumpMod`, and `waitT`—they’re basically quantized scheduling helpers you can wrap around transitions. ([Tidal Cycles][5])

---

### 4) Ableton Link section: update it to reflect current Tidal versions

Your doc says “Tidal supports Ableton Link” (true), but the operational detail matters:

* The Tidal docs note **native Link integration in Tidal 1.9**, enabled by default there.
* **Tidal 1.10 has Link disabled by default**, and you enable it in the boot config (by uncommenting `streamEnableLink tidal`). ([Tidal Cycles][6])

So I’d tweak the “Future: Ableton Link” section to explicitly call out that version-dependent toggle, and treat Link as “Phase 5+” unless you truly need shared tempo/phase across machines.

---

### 5) “Stop a single pattern” is not an OSC action (today)

You can hush everything via OSC (`/hush`)—great. But **there isn’t a standard OSC `/silence`** to stop a single pattern; if you want to stop one stream, OSC gives you `/mute` (and you can “stop” by muting indefinitely), or you send actual Tidal code (`d1 $ silence`) via eval. ([Tidal Cycles][4])

**Doc tweak suggestion:** in “Arrangement control”, treat `/mute` as your “stop channel” primitive; reserve eval for “replace pattern / true silence”.

---

## Consistency tweaks that will save you pain later

### A) Make `focus` unambiguous

Right now you have:

* Table: `focus` ∈ `drums`, `bass`, `pads`, `full`
* Example cue: `"drums+bass"` (not in the table)

I’d recommend one of these:

**Option 1 (clean JSON):**

```json
"focus": ["drums", "bass"]
```

**Option 2 (Tidal-native):** use *named patterns* and keep `focus` as a single string or list of strings you can pass directly to `/solo` as a string pattern ID. Tidal transitions and playback controllers both support string IDs (e.g. `p "drums"` … `xfade "drums"` … `/solo "drums"`). ([Tidal Cycles][5])

Named patterns are a big ergonomic win because they decouple “instrument meaning” from “d1/d2 numbering”.

---

### B) Replace the latency numbers with “jitter tolerance” language

“~10ms” vs “~100ms” is directionally fine, but in practice:

* OSC is UDP (no delivery guarantees)
* Tidal schedules audio events ahead of time (via `oLatency`)
* control values are sampled at Tidal’s frame timespan

So I’d reword the table as:

* Parameter steering: **low-latency / jitter-tolerant**
* Arrangement control: **near-immediate but musically quantized by context**
* Eval: **higher-latency + human-gated**

If you want responsiveness, consider documenting `cFrameTimespan` explicitly. People often set something like `cFrameTimespan = 1/20` for more responsive external control. ([Tidal Club][2])

---

## One big “Phase 1.5” upgrade: avoid MCP eval by using pattern-bank selection

You can get surprisingly far without MCP by doing **bank selection inside Tidal**, driven by external integer controls:

* Send `/ctrl section 3`
* In Tidal: `pickF (cI 0 "section") [pat0, pat1, pat2, pat3] ...`

The Open Stage Control thread shows exactly this pattern: selecting functions/patterns via `pickF` + `cI`. ([Tidal Club][2])

This has 3 advantages:

1. **Safety:** the agent never “runs code”, it just selects from precomposed materials.
2. **Stability:** fewer failure modes mid-performance.
3. **Musical continuity:** transitions stay in Tidal-land.

Separately, the OSC docs also note `cP` if you want to receive *mini-notation patterns as strings* over `/ctrl` (still safer than arbitrary eval). ([Tidal Cycles][4])

**How I’d reflect this in your proposal:**
Keep MCP eval as Phase 4, but add a “Phase 3 (Banked composition)” that gets you 80% of the value with 20% of the risk.

---

## How this ties to your narrative-engine notes (worth one explicit paragraph in the doc)

Your notes emphasize that music can encode *subtext*—especially the gap between what the audience knows and what characters know, and more generally the “information topology” of the scene. 

A simple way to make that concrete in the proposal:

* `tension` isn’t just “action intensity”
* it can be derived from an “epistemic gap” scalar (audience knowledge – character knowledge)
* `space` can reflect “psychological distance / isolation”
* `brightness` can reflect “moral legibility / clarity” vs ambiguity

You don’t need this in MVP, but naming it early keeps the mapping layer philosophically aligned with what you’re building on the video side.

---

## Extra resources I’d drop into your repo’s research folder

These are the pages I’d treat as “source of truth” for implementation details:

* Tidal **Controller Input** (`/ctrl`, defaults, `cF/cS/cI/cT/cP`). ([Tidal Cycles][1])
* Tidal **Transitions** (correct syntax for `xfadeIn`, `clutchIn`, `jumpIn'`, `waitT`, etc.). ([Tidal Cycles][5])
* Tidal **OSC / Playback controllers** (mute/solo/hush paths and args). ([Tidal Cycles][4])
* Tidal **Multi-User / Native Link** (version-specific Link behavior + `cQuantum`, `cBeatsPerCycle`). ([Tidal Cycles][6])
* The **Workshop tutorial** (quick on-ramp while you’re learning Tidal). ([Tidal Cycles][7])

And if you’re serious about “Claude Code as intermediary”:

* Claude Code docs on adding MCP servers (HTTP recommended; SSE deprecated). ([Claude Code][8])
* Claude help center notes on remote MCP transport/auth support. ([Claude Help Center][9])
* `mcp-proxy` for bridging transport types if you end up needing it in weird deployment topologies. ([GitHub][10])
* The Tidal community post about the **Tidal Cycles MCP Server** (gives you a reality check on how people are actually using it). ([Tidal Club][11])

---

## Net: what I’d change in your proposal text (minimal edits)

1. Fix the transitions snippet (`xfadeIn 1 4 ...`, not `xfadeIn 4 $ d1 ...`). ([Tidal Cycles][5])
2. Update the Ableton Link section with the 1.9 vs 1.10 default behavior. ([Tidal Cycles][6])
3. Make `focus` either a list or explicitly named-pattern based. ([Tidal Cycles][5])
4. Add “Phase 3: bank selection via `cI`/`pickF` (no eval)” as the middle step before MCP. ([Tidal Club][2])
5. Add one paragraph connecting `tension/space/brightness` to the narrative-engine “information topology” idea (so the mapping stays conceptually coherent as it grows). 

If you apply just those, the doc goes from “very good draft” to “ready to implement without surprises.”

[1]: https://tidalcycles.org/docs/working-with-patterns/Controller_Input/ "Controller Input | Tidal Cycles"
[2]: https://club.tidalcycles.org/t/open-stage-control-tidalcycles/1283 "Open Stage Control + TidalCycles = ❤️ - Announcements - Tidal Club"
[3]: https://userbase.tidalcycles.org/Playback_Controllers.html "Playback Controllers - TidalCycles userbase"
[4]: https://tidalcycles.org/docs/configuration/MIDIOSC/osc/ "OSC | Tidal Cycles"
[5]: https://tidalcycles.org/docs/reference/transitions/ "Transitions | Tidal Cycles"
[6]: https://tidalcycles.org/docs/configuration/multiuser-tidal/ "Multi-User Tidal | Tidal Cycles"
[7]: https://tidalcycles.org/docs/patternlib/tutorials/workshop/ "Workshop | Tidal Cycles"
[8]: https://code.claude.com/docs/en/mcp "Connect Claude Code to tools via MCP - Claude Code Docs"
[9]: https://support.claude.com/en/articles/11503834-building-custom-connectors-via-remote-mcp-servers "Building Custom Connectors via Remote MCP Servers | Claude Help Center"
[10]: https://github.com/sparfenyuk/mcp-proxy "GitHub - sparfenyuk/mcp-proxy: A bridge between Streamable HTTP and stdio MCP transports"
[11]: https://club.tidalcycles.org/t/tidal-cycles-mcp-server/5989 "Tidal Cycles MCP Server - Announcements - Tidal Club"
