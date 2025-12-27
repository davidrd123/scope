# Narrative Engine & Real-Time Video System
## Design Notes - December 2025

---

## Part 1: Conceptual Foundations

### The Pydantic Inversion (Code 2.0)

Traditional software (Code 1.0) embodies left-hemisphere dominance in McGilchrist's framing:
- Explicit schemas
- Rigid type enforcement
- Rejection of anything that doesn't conform to predefined categories
- No tolerance for ambiguity

**The inversion:**
- **Old world:** Structure determines what meaning is permissible
- **New world:** Meaning determines what structure is appropriate

Context-dependent validation means:
- Schema as prior, not law — LLM interprets mismatches semantically
- Semantic coercion over rejection — transform input based on understood intent
- Dynamic schema generation — appropriate structure inferred from context
- Validation as conversation — ambiguity triggers clarification, not failure
- Probabilistic conformity — confidence scores rather than pass/fail

**The schema serves understanding. Understanding does not serve the schema.**

### Non-Engineers Building Differently

The engineering mindset has been the *only* way to build because building required thinking like a computer. Sixty years of software built exclusively by people who passed through that filter.

With sufficiently capable coding agents, people with different orientations might build:
- Systems that prefer ambiguity and ask clarifying questions
- Systems oriented around meaning rather than function
- Systems that feel more like conversation than vending machine
- Solutions to problems engineers didn't think to solve

The architecture LLM buddy holds rigor while you engage in dialogue. Structure is necessary — but now it can be in service of intent rather than constraining it.

---

## Part 2: System Architecture

### World State Representation

**Static Layer** — facts, configuration, what *is*
- Scene location, characters present, relationships, backstory
- Traditional scene graph territory

**Dynamic Layer** — processes, what's *happening*
- Emotional trajectories (not just "angry" but "anger building")
- Intentions in motion (character A trying to convince character B)
- Tension arcs (scene approaching breaking point)
- "Being angry" is state; "escalating toward confrontation" is trajectory with momentum

**Derived Layer** — expression parameters
- Text-to-video prompts generated from static + dynamic
- Tidal parameters for musical accompaniment
- Transition timing responding to narrative beats

### Information Topology

**Ground truth** — what's actually the case in the world

**Per-character belief states** — their model, which can be:
- Wrong
- Incomplete
- Outdated
- Deliberately false

**Audience knowledge state** — what's been rendered/revealed

**Computable from these:**
- Irony level — gap between audience and character knowledge
- Suspense potential — audience knows something bad coming, character doesn't
- Reveal readiness — pressure built up behind a secret

### Chekhov's Gun as State

Information planted but not paid off is narrative debt. Track:
- What's been shown
- What it implies
- Whether it's been activated

Plant without payoff = broken promise. Payoff without plant = deus ex machina.

---

## Part 3: Intent & Subtext Layers

### Hitchcock's Method

Not "what happens in the story" but "what do I want the audience to feel, and when" — then reverse-engineer events, shots, and information flow to produce that.

The supervening intent is emotional/experiential, not narrative.

### Subtext as Engine

The surface (dialogue, action, what's visible) is downstream of what's *not* said. Characters talk about the weather but the scene is about their failing marriage.

**Generative flow:**
1. Emotional intent → 
2. Subtext → 
3. Surface that disguises/reveals subtext → 
4. Rendering choices that modulate the gap

The tension between surface and subtext *is* the energy. A scene where characters say exactly what they mean has no charge.

### Complete Layer Hierarchy

**Intent Layers (top-down)**
- Supervening emotional intent (what should audience feel)
- Subtext (what's actually going on beneath surface)
- Surface (observable pretext)
- Rendered output (prompts, music, timing)

**Ontological Layers**
- World substrate (facts, physics, positions)
- Character internals (beliefs, desires, knowledge — potentially divergent from truth)
- Relationships (between characters, not just individual states)
- Trajectories (things in motion, unresolved intentions)
- Plants/setups awaiting payoff

**Information Topology**
- Ground truth
- Per-character belief states
- Audience knowledge state
- Gaps as active generators of tension

---

## Part 4: Multi-Agent Stakeholder System

### Agent Roles

Each stakeholder responsible for different concerns:

- **Theme agent** — does this beat serve the theme?
- **Character agent** — is this consistent with what this character would do? Holds per-character belief states
- **Pacing agent** — rhythm, tension curves, audience attention
- **Trajectory agent** — what's been set up, what needs payoff
- **Tension agent** — advocates for maintaining/widening information gaps until right moment
- **World/continuity agent** — hard constraints, what's physically possible

### Negotiation Mechanics

**Veto vs preference**
- Hard constraints (character can't be in two places)
- Soft preferences (this beat would be *better* if...)

**Weighted authority by context**
- Climactic emotional scene: character and thematic agents have more weight
- Action sequence: trajectory and world-substrate have more weight
- Genre agent helps adjudicate which mode we're in

**Proposer vs critic roles**
- Not all agents propose — some primarily react
- Trajectory agent: "we need a beat that raises stakes"
- Others evaluate whether proposals satisfy that

**Dissent logging**
- When agent is overruled, record it
- "Theme agent wanted X, was overruled because Y"
- Useful for debugging and intentional subversion

**Tiebreaker**
- Meta-agent or human as arbiter
- System surfaces conflicts it can't resolve

### Rewindable Branching

Agent dialogue becomes metadata on trajectory. Each node:
- World state at time T
- Negotiation that produced this transition
- Dissenting opinions

On rewind/branch:
- See *why* this path went this way
- Explicitly override: "this time, honor the thematic agent's objection"
- Ask: "what would character agent have preferred?"
- Replay negotiation with different weights

Version control for narrative — commits include reasoning, not just diffs.

### Implementation Notes

- Start simple: basic API calls, JSON world state
- Claude Agent SDK or OpenAI SDK for harness
- Multi-model flexibility valuable (Gemini 3 Flash for fast reactive agents)
- Build coordination layer yourself — the negotiation is your architecture
- Skills as progressive disclosure of domain knowledge per agent

---

## Part 5: Output Mapping

### Video Prompt Translation

World state → text prompts that encode:
- What's in frame
- Compositional intent
- Emotional tenor
- Character state/action

Mediated by LoRA selection as tonal register:
- Hidari for handcrafted/deliberate
- Mutant Mayhem for chaotic energy
- Rankin/Bass for cozy/whimsical
- The swap *means* something narratively

### Tidal Integration

Music can express what audience knows that characters don't. Underscore the characters "can't hear."

Mapping possibilities:
- Subtext/irony gaps → musical tension
- Leitmotifs activate based on audience knowledge, not just scene content
- Character internal states as patterns that modulate music
- Tidal is inherently about patterns over time — fits process/trajectory layer

### Editing Logic as Latent Operations

Prompts aren't just descriptions — they're cuts. The sequence *is* the edit.

Interpolating between embeddings during plasticity windows = dissolves at latent level.

Cinema grammar becomes parameters:
- Shot/reverse-shot
- Match cuts
- L-cuts
- Smash cuts
- Soft cuts (via decreased continuity parameter)

Transitions respond to narrative beats, not arbitrary timing.

---

## Part 6: Current Pipeline

### Krea Real-Time + Wan 2.1

- Auto-regressive variant running on B200
- Started at 8.8 fps, now at ~20.4 fps through optimization
- Flash Attention 4, Blackwell-specific tuning
- Replacing Flex Attention with targeted implementations
- 4-6 diffusion steps for real-time

### Workflow

1. Source film segmented by scenes
2. Gemini 3 Flash for captioning (cost-effective at scale)
3. Captions become prompt sequence
4. Real-time TUI for scrolling through prompts
5. LoRA hot-swapping (all loaded, weights at zero except active)
6. Soft cuts via plasticity window / reduced continuity

### Trained LoRAs

- Rankin/Bass (Rudolph 1964)
- Kaiju/Tokusatsu
- Rooster and Terry (student project, squash/stretch)
- TMNT Mutant Mayhem
- Hidari (wood stop-motion)
- (Arcane available from others)

### LoRA Training Candidates

High contrast / bold graphic (survives low fidelity):
- Spider-Verse
- Yellow Submarine
- Fantastic Planet
- Redline

Painterly / textural (imperfection reads as style):
- Loving Vincent
- Belladonna of Sadness
- Tale of Princess Kaguya

Stop motion (blend space with Rankin/Bass and Hidari):
- Kubo and the Two Strings
- Mad God
- Coraline

Rotoscope / liminal:
- Waking Life
- A Scanner Darkly

Outliers:
- Mob Psycho 100 (sakuga bursts)
- The Thief and the Cobbler

### Source Films for Prompt Sequences

Already done:
- 2001: A Space Odyssey
- Akira

Maximum cultural recognition:
- The Matrix
- The Godfather
- Jurassic Park

Anime through non-anime LoRAs:
- End of Evangelion
- Paprika
- Perfect Blue

Texture/atmosphere-heavy:
- Blade Runner
- Apocalypse Now
- Stalker

Music-native (for Tidal integration):
- Fantasia segments
- Interstella 5555
- The Wall

Wild card:
- Koyaanisqatsi

---

## Part 7: Timeline & Next Steps

**16-day hackathon, currently day 4**

**Immediate (days 4-7):**
- Tidal exploration and documentation gathering
- Continue optimization (Codex running in parallel)
- Segment additional films for prompt sequences
- LoRA hot-swapping implementation

**Medium-term (days 8-11):**
- Hook narrative engine concepts into pipeline
- Start with JSON world state + 1-2 agents
- Basic negotiation loop

**Later (days 12-16):**
- Tidal integration with state
- Multi-agent coordination
- Branching/rewind capability

**Principle:** Hold complete system in mind while building one piece. Early decisions implicitly constrain later options.

---

## Appendix: The Simulators Frame

From project file — relevant conceptual background:

GPT (and similar models) as simulators rather than agents or oracles:
- Distinction between simulator (rule) and simulacra (phenomena)
- Model doesn't "want" what simulated entities want
- Prediction orthogonality thesis: predictor can simulate agents optimizing toward any objectives
- Not optimizing for a goal — computing what comes next given context

Implications for narrative engine:
- The system simulates narrative processes, doesn't "want" particular outcomes
- Characters are simulacra with their own (simulated) goals
- System can simulate multiple competing agents simultaneously
- "Direction" comes from constraints and stakeholder negotiation, not built-in objective

---

## Appendix: Key References

- McGilchrist — hemisphere theory, left as emissary not master
- Hitchcock — supervening emotional intent
- Simulators doc — prediction vs agency framing
- Claude Agent SDK — harness for building agents (Claude-only natively, proxies available for multi-model)
- OpenAI Agent SDK — native multi-model via LiteLLM
