# Internal Vision

> **Purpose:** Living document - actual state, real priorities, where threads lead
> **Pattern:** Log discoveries → synthesize periodically
> **Updated:** 2025-12-26
> **See also:** `capability-roadmap.md` (implementation tracking), `daydream/cohort-pitch.md` (external-facing)

---

## The Thesis

**AI video generators are simulators, not agents.** They compute "what comes next" given context. The interesting work is building interfaces to navigate that state space - not just rolling dice on outputs.

**Interactive continuity:** Video generation with memory, branching, and play. Not isolated clips. Not pre-rendered branches. Actually exploring the possibility space in real-time.

---

## Actual State of Play

### What's Implemented

| Component | Status | Notes |
|-----------|--------|-------|
| Real-time pipeline | 20 FPS B200, 19 FPS B300 | Krea 14B with FA4 attention |
| Step mode | Working | Pause, generate chunk, inspect |
| Prompt transitions | Working | Smooth interpolation |
| Prompt playlist | Working | Navigate caption files |
| CLI tools | Working | `video-cli prompt/step/pause/run` |
| Hard/soft cuts | Working | Cache reset, plasticity window |
| REST API | Working | `/api/v1/realtime/` surface |
| Session recording | Frontend working | Timeline export on stop |

### What's Designed (Not Built)

| Component | Status | Doc |
|-----------|--------|-----|
| WorldState/StyleManifest | In progress | `capability-roadmap.md` §1 |
| Style swap (hot LoRA switching) | Ready | `proposals/style-swap-mode.md` |
| VLM frame analysis | Ready | `proposals/vlm-integration.md` |
| Frame buffer (scrubbing) | Ready | `proposals/frame-buffer-scrubbing.md` |
| Server-side session recorder | Ready | `proposals/server-side-session-recorder.md` |
| VACE-14B | Ready | `capability-roadmap.md` §2 |
| NDI output | Ready | `proposals/ndi-pubsub-video-output.md` |

### What's Speculative (Phase 2+)

| Concept | Doc | Depends on |
|---------|-----|------------|
| Trajectory semantics | `concepts/narrative-engine.md` | WorldState |
| Information topology | `concepts/narrative-engine.md` | WorldState |
| Multi-agent stakeholders | `concepts/narrative-engine.md` | Basic narrative working |
| Intent/subtext layers | `concepts/narrative-engine.md` | WorldState + style |
| Context editing (anchor edits) | `capability-roadmap.md` §3 | Validation spike |
| Tidal integration | `proposals/tidal-cycles-integration.md` | Tidal setup |

---

## Real Priorities (What's Actually Hot)

### Right Now
1. **Interactive control** - step, pause, branch, compare. This is the core loop.
2. **Real-time performance** - 19-24 FPS is the responsive threshold. Already there.
3. **Style switching** - flip between LoRAs to see same content in different styles.

### Next Up
4. **Prompt engineering workflow** - Loom-like iteration to find what works.
5. **VLM feedback** - describe-frame for automated evaluation.
6. **Frame buffer** - scrub back through what was generated.

### Later (Phase 2)
7. **Narrative coherence** - world state that persists, trajectories, information topology.
8. **Multi-agent** - stakeholders debating each beat.

---

## The Discovery Process

Building infrastructure → discovering what's cool → adjusting direction.

### Pattern
1. Build the basic capability (e.g., step mode)
2. Play with it, notice what's interesting
3. Log the insight (see Discovery Log below)
4. Adjust priorities based on what's actually useful
5. Synthesize into this doc periodically

### Current Discoveries

**Interactive control is the unlock.** Before: "generate and hope." Now: "generate, evaluate, adjust, branch." The ability to pause and poke changes everything.

**Style switching has narrative meaning.** LoRA isn't just visual style - it's tonal register. Switching from Rankin-Bass to Mutant Mayhem *means* something. The same scene feels different.

**Prompt engineering is iterative R&D.** "Does this prompt make the prop jiggle?" requires step-by-step evaluation and branching. Not narrative coherence - visual behavior verification.

**FPS threshold matters.** Below ~16 FPS (1x playback), interaction feels sluggish. Above 1.2x, it feels responsive. B300 at 19 FPS crossed into usable territory.

---

## Beyond the Hackathon

This continues after Jan 9.

### Integration with Existing Work
- **217 LoRAs on HuggingFace** - 4 years of trained styles
- **Rooster & Terry** - ongoing animated project, natural test case
- **Prompt engineering knowledge** - what works for which LoRA

### The Broader Thesis
From the Simulators doc: the model doesn't "want" what simulated entities want. It computes what comes next. We're building the interface to explore that space.

**Implication:** The system doesn't need to be "intelligent" about narrative. It needs to be controllable and explorable. Intelligence comes from the human (or agent) navigating the space.

### Long-term Direction
1. **Instrument for creative exploration** - not a pipeline, a tool
2. **Narrative as navigable space** - branch, compare, select
3. **Style as meaning** - LoRA switching as creative choice
4. **Agent-assisted iteration** - VLM in the loop for evaluation

---

## Decision Framework

### What to Work On

**Prioritize based on:**
1. Does it enable new interactions? (high value)
2. Does it make existing interactions faster? (medium value)
3. Does it add complexity without enabling new things? (low value)

**Current ranking:**
1. Style Layer (enables style switching) - in progress
2. VLM frame analysis (enables automated evaluation) - ready
3. Frame buffer (enables scrubbing/comparison) - ready
4. VACE-14B (enables reference conditioning) - ready
5. Narrative engine concepts (enables coherence) - Phase 2

### Trade-offs

| Trade-off | Current Choice | Reason |
|-----------|---------------|--------|
| FPS vs features | FPS first | Need responsive threshold before adding features |
| Stability vs exploration | Stability on B300 | Can't optimize what doesn't run reliably |
| Implement vs design | Implement Style Layer | Need working foundation before Phase 2 |
| Breadth vs depth | Breadth of capabilities | Discovery phase - find what's interesting |

---

## Discovery Log

Log insights as they happen. Synthesize into sections above periodically.

### 2025-12-26
- B300 reached 19 FPS - crossed into responsive territory (1.2x real-time)
- Narrative engine concepts captured in `concepts/narrative-engine.md`
- Prompt engineering workflow is distinct from narrative coherence - different use case
- Created `concepts/prompt-engineering-workflow.md` for Loom-like iteration pattern
- Cohort pitch reframed as experience-first ("pause, fork, compare, continue")

### 2025-12-25
- FA4 attention integrated, 1.89x speedup
- Session recording working on frontend
- REST API surface complete
- Style Layer (Phase 6a) in progress

### 2025-12-24
- B300 investigation - cu130 runtime was the issue, not kernels
- Established FA4 documentation structure
- RoPE fusion working

### 2025-12-22-23
- Project kickoff
- Initial pipeline bring-up
- Profiling infrastructure established

---

## Key Documents

| What | Where |
|------|-------|
| Implementation tracking | `capability-roadmap.md` |
| External pitch | `daydream/cohort-pitch.md` |
| Phase 2 concepts | `concepts/narrative-engine.md` |
| Prompt iteration workflow | `concepts/prompt-engineering-workflow.md` |
| All proposals | `proposals/` |
| FA4 optimization | `FA4/` |
| Master index | `NOTES-INDEX.md` |

---

## Open Threads

Things to think about, not yet prioritized:

1. **Voice input** - speak prompts instead of typing. Natural for creative flow.
2. **Audio/Tidal sync** - music driving generation parameters. Expressive potential.
3. **Mobile camera input** - VACE reference from phone. Live performance angle.
4. **Multi-LoRA blending** - not just switching, but mixing styles. Unexplored.
5. **Agent-driven narrative** - Claude as director, making beat decisions. Phase 2+.
6. **Cross-session memory** - what worked before, carry forward. Far future.

---

*Last synthesized: 2025-12-26*
