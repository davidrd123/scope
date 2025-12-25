# Creative Workflows: Explore, Record, Playback

> **Status:** Early ideation
> **Created:** 2025-12-25
> **Context:** Emerged from playlist navigation work + Akira caption exploration

---

## The Insight

Current workflow is purely **exploratory** — the KV cache maintains continuity, so everything morphs from what's on screen. This is great for discovery, but limits us to improvisation.

What if we could:
1. **Explore** — improvise, try paths, discover what works
2. **Record** — capture the good paths as we find them
3. **Playback** — replay sequences with structure, including hard cuts

This is the difference between **jazz improv** and **composition** — and we want both, plus the bridge between them.

---

## Core Concepts

### Explore Mode (Current)

- KV cache flows continuously
- Everything morphs from current state
- Great for discovery
- No memory of "how we got here"

### Record Mode (New)

- Same as explore, but capturing the path
- Marks prompts, timing, and decision points
- Can insert "hard cut here" markers
- Exports to a sequence format

### Playback Mode (New)

- Load a pre-defined sequence
- Execute prompts in order
- Hard cuts reset the cache for clean scene breaks
- Branch points allow choose-your-own-adventure

---

## Hard Cuts: The Atomic Primitive

Everything builds on the ability to do a **hard cut** — a clean scene transition rather than a morph.

### Current Behavior (Morph)
```
[Scene A] → prompt change → [Scene A morphs into Scene B]
```

### Hard Cut Behavior
```
[Scene A] → HARD CUT → [Scene B starts fresh]
```

### Implementation (from capability-roadmap.md)
```
1. Disable "Manage Cache"
2. Reset the cache
3. Re-enable "Manage Cache"
4. Apply the new prompt
```

This is already a feature request. Once implemented, sequences and recording are just layers on top.

---

## Sequence Format (Sketch)

Like musical notation:
- **Prompts** = notes/phrases
- **Hard cuts** = rests / bar lines / key changes
- **Morph (default)** = legato
- **Branches** = alternate takes / variations

### Simple Linear Sequence

```yaml
name: "Akira Opening"
default_duration: 5s

sequence:
  - prompt: "Neo-Tokyo 2019, city lights"
    duration: 8s

  - prompt: "Motorcycle gang, highway chase"

  - hard_cut: true  # or just: "---"

  - prompt: "Tetsuo face to face with Kaneda"

  - prompt: "Tension, neon reflections"
```

### Branching Sequence

```yaml
name: "Akira Choice Point"

scenes:
  - id: confrontation
    prompts:
      - "Tetsuo and Kaneda, standoff"
      - "Tension building"
    next:
      - label: "Follow Tetsuo"
        goto: tetsuo_path
      - label: "Follow Kaneda"
        goto: kaneda_path

  - id: tetsuo_path
    hard_cut: true
    prompts:
      - "Tetsuo's power awakening"
      - "Hospital machines sparking"
    next: ending_a

  - id: kaneda_path
    hard_cut: true
    prompts:
      - "Kaneda on motorcycle, determined"
      - "Racing through neon streets"
    next: ending_b
```

---

## Recording Workflow (Vision)

```
1. Start in explore mode
2. Find a good sequence of prompts
3. Mark "record from here"
4. Continue exploring, system captures:
   - Prompts used
   - Timing between changes
   - When you manually triggered hard cuts
5. Mark "hard cut" at natural scene breaks
6. Export session as sequence file
7. Later: load, replay, share, branch
```

---

## Use Cases

### Choose Your Own Adventure
- Pre-explored branching tree
- User makes choices at branch points
- Each path has been validated to look good

### Music Video / Narrative
- Timed sequence synced to music
- Hard cuts on beat drops
- Prompts evolve with song sections

### Live Performance
- Performer navigates through prepared sequences
- Can diverge into explore mode
- Can return to "safe" recorded paths

### Training / Demo
- Reproducible sequences for showing off capabilities
- "Here's what the system can do" walkthroughs

---

## Relationship to Existing Features

| Feature | Status | How It Connects |
|---------|--------|-----------------|
| Prompt Playlists | Implemented | Linear sequence, no hard cuts yet |
| Autoplay Navigation | Implemented | Timing/auto-advance |
| Hard Cut Toggle | Feature Request | The atomic primitive |
| Style Layer | In Progress | Could be captured in sequences too |
| VACE Reference Images | Implemented (1.3B) | Could be sequence elements |

---

## Open Questions

1. **File format** — YAML? JSON? Custom DSL? Markdown-ish?
2. **Timing model** — duration per prompt? or frame-count based?
3. **Branch UI** — how does user make choices during playback?
4. **Recording granularity** — capture every prompt change, or user-marked waypoints?
5. **Sync to external** — music, timecode, other media?
6. **Sharing** — sequences as shareable artifacts?

---

## Next Steps

1. Implement hard cut primitive (CLI + API)
2. Extend playlist format to include hard cut markers
3. Prototype recording mode
4. Iterate on sequence format based on actual use

---

## Related Documents

### Core Concepts (in this folder)
- This document — explore/record/playback model, sequence format sketch

### Detailed Specs (should probably live here too)
- `notes/plans/tui-director-console.md` — TUI for real-time keyboard-driven directing
  - Keyboard controls, style/beat/camera selection, snapshots
  - Textual-based implementation sketch

- `notes/research/2025-12-24/incoming/context_editing_and_console_spec.md` — Full spec for context editing + CLI
  - Retroactive frame editing via decoded_frame_buffer
  - CLI commands (video-cli step, edit, snapshot, describe-frame)
  - Agent integration patterns
  - **Very comprehensive, 900 lines**

### Research Background
- `notes/research/2025-12-23/krea-realtime-prompt-sequences.md` — Prompt sequence pathways
  - "Anchor clause" pattern (constant subject/setting)
  - Soft transitions vs hard cuts
  - Two-pass workflows (preview then final render)
  - LongLive, StreamingT2V comparisons

### Feature Requests
- `notes/capability-roadmap.md` — Hard Cut Toggle feature request

### Project Context
- `notes/daydream/cohort-pitch.md` — Project vision
- `notes/daydream/interactive-ai-video-program.md` — Cohort program description

---

## Consolidation TODO

Consider moving to `notes/concepts/`:
- [ ] `tui-director-console.md` → `concepts/tui-director.md`
- [ ] `context_editing_and_console_spec.md` → `concepts/context-editing-spec.md`
- [ ] `krea-realtime-prompt-sequences.md` → `concepts/prompt-sequences.md`

This would create a cohesive "concepts" folder for creative/product ideation.
