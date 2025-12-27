# Editorial Intelligence Layer

> Status: Concept (from Patrick, 2025-12-27)
> Contributors: Patrick (TheSwoosh)
> Related: `narrative-engine.md`, `tui-director.md`, `prompt-engineering-workflow.md`

## Problem Statement

Take script syntax (story + scene direction) and chuff it through a pipeline to get a visual representation edited as quickly as possible.

## Core Insight

A good forward-thinking director thinks ahead to the type of coverage they provide their editing team. The coverage and shot selection is inherent in the scene direction itself — the script provides clues and subtext.

**Key principle:** A director who knows how the scene will be cut in the editing room curates their shot list and coverage accordingly.

---

## The Pipeline

```
Script / Scene Direction
        │
        ▼
┌─────────────────────────────┐
│  Agent: Director Brain      │
│  - Ingests scene            │
│  - Extracts subtext/clues   │
│  - Thinks like an editor    │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Coverage Plan              │
│  - Shot types (based on     │
│    scene context)           │
│  - Action → zooms, whip pans│
│  - Dialogue → shot/reverse  │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Editing Logic Library      │◄──── Markdown technique guides
│  - Mood/emotion → technique │      (repo of patterns)
│  - Pacing rules             │
│  - Shot order logic         │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Augmented Prompt Sequence  │
│  - Order matters            │
│  - Linger duration          │
│  - Cut timing               │
│  - Visual cue underscoring  │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  TUI Playlist Execution     │
│  - Real-time generation     │
│  - Soft cuts between shots  │
│  - Interactive override     │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Notes / Refinement Loop    │
│  - Take notes (any form)    │
│  - Apply to prompts         │
│  - Reinsert/reorder shots   │
│  - Re-run the cut           │
└─────────────────────────────┘
```

---

## Editing Logic Library

A repo of markdown files describing contrasting editing techniques:

### Possible Structure

```
styles/editing/
├── pacing/
│   ├── slow-burn.md        # Long takes, minimal cuts
│   ├── frenetic.md         # Quick cuts, jump cuts
│   └── rhythmic.md         # Cuts on beat, musical timing
├── mood/
│   ├── tension.md          # Hold shots, delayed reveals
│   ├── release.md          # Wide shots, breathing room
│   └── intimacy.md         # Close-ups, shallow depth
├── coverage/
│   ├── dialogue.md         # Shot/reverse, two-shot, OTS
│   ├── action.md           # Wide establish, tight inserts, whip pans
│   └── montage.md          # Thematic juxtaposition, rhythm
└── transitions/
    ├── hard-cut.md         # Jarring, intentional discontinuity
    ├── soft-cut.md         # Smooth blend (our innovation)
    ├── match-cut.md        # Visual/thematic continuity
    └── l-cut-j-cut.md      # Audio leads/follows video
```

### Example: `tension.md`

```markdown
# Tension Editing

## When to Use
- Building toward confrontation
- Suspense before reveal
- Character internal conflict

## Shot Guidance
- HOLD longer than comfortable
- Tight on faces (catch micro-expressions)
- Avoid cutting away — let it breathe
- Slow push-in rather than cut

## Linger Duration
- 1.5x normal shot length minimum
- Let the audience squirm

## Cut Timing
- Cut on inhale, not exhale
- Cut before the expected moment
- Subvert rhythm established earlier

## Prompt Augmentation
- Add: "tense stillness", "held breath", "anticipation"
- Avoid: action verbs, movement descriptors
```

---

## Integration Points

### With TUI Director Console

The editing logic influences:
- **Linger duration** — How long TUI stays on each prompt before advancing
- **Cut timing** — When soft cuts trigger (on beat? on action?)
- **Shot order** — Sequence of prompts in playlist

### With Style Layer

Editing technique can be orthogonal to visual style:
- Same "tension" technique could apply to Hidari, TMNT, or Rankin-Bass
- Or certain styles might have preferred editing approaches

### With Prompt Compilation

Editing logic **augments** prompts to underscore visual cues from the script:
- Scene says "confrontation" → editing logic adds "hold", camera says "close"
- Scene says "chase" → editing logic adds "whip pan", "quick cut" metadata

---

## The Notes Loop

Future-proofing for iterative refinement:

1. **Generate initial cut** from script
2. **Watch and take notes** (any form — text, voice, annotations)
3. **Notes go into vault** — associated with specific prompts/shots
4. **Agent applies notes** to corresponding prompts
5. **Reinsert/reorder** shots in playlist
6. **Re-run** the cut with refinements
7. **Repeat** until satisfied

This is the "quick iterative machine that addresses instant previsualization of permuting story."

---

## Relation to Coverage Types

### Action Scene Coverage

| Script Cue | Coverage Response |
|------------|-------------------|
| "fight breaks out" | Wide establish → tight inserts → reaction shots |
| "chase through streets" | Whip pans, dutch angles, POV shots |
| "explosion" | Wide for scale → close for reaction → wide aftermath |

### Dialogue Scene Coverage

| Script Cue | Coverage Response |
|------------|-------------------|
| "heated argument" | Shot/reverse, tightening close-ups |
| "confession" | Single on speaker, reactions sparingly |
| "group discussion" | Two-shots, wide establishing, singles for emphasis |

### Emotional Scene Coverage

| Script Cue | Coverage Response |
|------------|-------------------|
| "moment of realization" | Hold on face, push in slowly |
| "grief" | Wide isolation shot, then close |
| "joy/celebration" | Movement, wide, multiple angles |

---

## Open Questions

- [ ] How does script parsing work? (Screenplay format? Custom DSL? Natural language?)
- [ ] Where does the "Director Brain" agent live? (Claude? Local LLM? Rule-based?)
- [ ] How are notes ingested? (Markdown? Voice transcription? Structured form?)
- [ ] How does this integrate with existing playlist format?

---

## Next Steps

1. Patrick to develop editing logic markdown examples
2. Define script input format (or accept multiple)
3. Prototype "Director Brain" prompt that generates coverage plan
4. Wire coverage plan → TUI playlist with timing metadata

---

## The Vision (Patrick's Words)

> "A quick iterative machine that addresses instant previsualization of permuting story."

Script in → Edited sequence out → Intervene anywhere → Refine with notes → Re-run.

The director stays in control.
