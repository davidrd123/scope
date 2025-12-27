# Narrative Engine — Phase 2 Concepts

> **Status:** Vision / Future-facing
> **Depends on:** Phase 1 infrastructure (WorldState, StyleManifest, BranchGraph)
> **Source:** `notes/research/2025-12-26/narrative_engine/claude_01.md`

These concepts layer on top of the current architecture. Don't implement until Phase 1 is stable.

---

## The Core Insight

Current system: WorldState → StyleManifest → Prompt → Render

Missing layers:
1. **Dynamic trajectories** (not just states)
2. **Information topology** (who knows what)
3. **Intent/subtext** (what should audience feel)
4. **Multi-agent negotiation** (stakeholders debate each beat)

---

## 1. Trajectory Semantics

**Current WorldState:** Static snapshots
- `character.emotion = "angry"`

**Proposed:** Trajectories with momentum
- `character.trajectory = "anger_building_toward_confrontation"`
- "Being angry" is state; "escalating toward confrontation" is process

### Schema Extension (Draft)

```python
class Trajectory(BaseModel):
    """A process in motion, not just a state."""
    type: str  # "emotion_arc", "intention", "tension_build"
    current: float  # 0.0 to 1.0, where in the arc
    momentum: float  # rate of change
    target: str | None  # what it's heading toward

class CharacterState(BaseModel):
    name: str
    emotion: str  # current snapshot
    trajectory: Trajectory | None  # process in motion
    intention: str | None  # "trying to convince B"
```

### Value
- Enables "tension building" as a continuous parameter
- Maps naturally to Tidal's pattern-over-time model
- Gives transitions meaning (following a trajectory, not just morphing)

---

## 2. Information Topology

Track divergent knowledge states:

| Layer | What it holds |
|-------|---------------|
| **Ground truth** | What's actually the case |
| **Character belief** | What each character thinks is true (can be wrong, incomplete, outdated) |
| **Audience knowledge** | What's been revealed to the viewer |

### Computed Values

From these layers, derive:

```python
irony_level = audience_knowledge - character_knowledge
# High irony: audience knows the killer is behind the door, character doesn't

suspense_potential = audience_foreshadowing - character_awareness
# High suspense: we know something bad is coming, they don't

reveal_readiness = pressure_built / threshold
# How much tension has accumulated behind a secret
```

### Schema Extension (Draft)

```python
class KnowledgeState(BaseModel):
    """What an entity knows or believes."""
    facts: dict[str, Any]  # key → belief
    confidence: dict[str, float]  # key → how sure

class InformationTopology(BaseModel):
    ground_truth: dict[str, Any]
    character_beliefs: dict[str, KnowledgeState]  # character_name → beliefs
    audience_knowledge: KnowledgeState

    def irony_level(self, character: str, fact_key: str) -> float:
        """How much does audience know that character doesn't?"""
        ...
```

### Value
- Enables Hitchcock's method (orchestrate what audience knows vs characters)
- Suspense becomes a computed, trackable value
- Reveals have setup/payoff mechanics

---

## 3. Chekhov's Gun Tracking

Narrative debt: things planted but not paid off.

| Event | State |
|-------|-------|
| Knife shown on table | Planted |
| Knife used in fight | Paid off |
| Knife never mentioned again | Broken promise |
| Fight uses knife never shown | Deus ex machina |

### Schema Extension (Draft)

```python
class Plant(BaseModel):
    """Something introduced that implies future relevance."""
    id: str
    description: str
    introduced_at: int  # chunk index
    paid_off_at: int | None
    implied_payoff: str | None  # "will be used as weapon"

class NarrativeDebt(BaseModel):
    plants: list[Plant]

    @property
    def unpaid(self) -> list[Plant]:
        return [p for p in self.plants if p.paid_off_at is None]
```

### Value
- System can warn: "you planted X but never paid it off"
- Agent can suggest: "this would be a good moment to use the knife"
- Avoids deus ex machina by tracking what's been established

---

## 4. Intent/Subtext Layers

The Hitchcock method: start with "what should audience feel" and reverse-engineer the surface.

### Layer Hierarchy

```
1. Supervening Intent    "I want the audience to feel dread"
        ↓
2. Subtext               "The marriage is failing"
        ↓
3. Surface               "They discuss the weather"
        ↓
4. Rendered Output       Prompts, camera, music
```

The gap between surface and subtext *is* the energy. A scene where characters say exactly what they mean has no charge.

### Schema Extension (Draft)

```python
class SceneIntent(BaseModel):
    """What the scene should accomplish emotionally."""
    audience_feeling: str  # "dread", "relief", "anticipation"
    subtext: str  # what's really happening
    surface: str  # what's overtly happening
    tension_source: str | None  # where the gap creates energy

class IntentCompiler:
    """Generates surface from intent + subtext."""
    def compile(self, intent: SceneIntent, style: StyleManifest) -> str:
        # Given "audience should feel dread" + "marriage failing" + "discussing weather"
        # Generate: camera angles, lighting, pacing that convey dread
        ...
```

### Value
- Directors think in intent, not prompts
- Same subtext, different surfaces = variation without losing meaning
- Enables "make this scene feel more tense" as a high-level control

---

## 5. Multi-Agent Stakeholder System

Multiple agents debate each beat, each responsible for different concerns:

| Agent | Responsibility |
|-------|---------------|
| **Theme** | Does this serve the theme? |
| **Character** | Would this character do this? Holds per-character beliefs |
| **Pacing** | Rhythm, tension curves, audience attention |
| **Trajectory** | What's been set up, what needs payoff |
| **Tension** | Advocates for maintaining/widening information gaps |
| **World** | Hard constraints, what's physically possible |

### Negotiation Mechanics

**Veto vs Preference:**
- Hard constraints (character can't be in two places) = veto
- Soft preferences (this beat would be *better* if...) = weighted vote

**Weighted Authority by Context:**
- Climactic emotional scene: character + theme agents have more weight
- Action sequence: trajectory + world agents have more weight

**Proposer vs Critic:**
- Not all agents propose; some primarily react
- Trajectory agent: "we need a beat that raises stakes"
- Others evaluate whether proposals satisfy that

**Dissent Logging:**
- When agent is overruled, record it
- "Theme agent wanted X, was overruled because Y"
- Useful for debugging and intentional subversion

### Implementation Notes

- Start simple: basic API calls, JSON world state
- Claude Agent SDK or OpenAI SDK for harness
- Multi-model flexibility (Gemini 3 Flash for fast reactive agents)
- Build coordination layer yourself — the negotiation is the architecture

---

## 6. Rewindable Branching with Reasoning

Extend BranchGraph to include *why* each path went that way.

### Current BranchGraph
- Snapshots at decision points
- Can fork, preview, select
- State only, no reasoning

### Proposed Extension
- Agent dialogue becomes metadata on each node
- Each node stores: world state + negotiation that produced transition + dissenting opinions
- On rewind/branch:
  - See *why* this path went this way
  - "What would character agent have preferred?"
  - Explicitly override: "this time, honor the thematic objection"
  - Replay negotiation with different weights

### Schema Extension (Draft)

```python
class BranchNode(BaseModel):
    # Existing
    snapshot: Snapshot
    children: list[str]  # child node IDs

    # New: reasoning
    negotiation_log: list[AgentVote]
    dissent: list[AgentDissent]
    decision_rationale: str

class AgentVote(BaseModel):
    agent: str  # "theme", "character", etc.
    position: str  # what they advocated
    weight: float  # how much authority in this context

class AgentDissent(BaseModel):
    agent: str
    wanted: str
    overruled_because: str
```

### Value
- Version control for narrative — commits include reasoning, not just diffs
- Can ask: "show me branches where theme agent was overruled"
- Enables intentional subversion with awareness

---

## 7. Output Mapping Extensions

### LoRA as Tonal Register

LoRA selection isn't just visual style — it's meaning:
- Rankin-Bass for cozy/whimsical
- Mutant Mayhem for chaotic energy
- Hidari for handcrafted/deliberate

The swap *means* something narratively. Style changes can mark tonal shifts.

### Editing Logic as Latent Operations

Prompts aren't just descriptions — they're cuts. The sequence is the edit.

Cinema grammar becomes parameters:
- Shot/reverse-shot
- Match cuts
- L-cuts (audio leads visual)
- Smash cuts
- Soft cuts (via decreased continuity parameter)

Transitions respond to narrative beats, not arbitrary timing.

### Tidal Integration (Enhanced)

Music expresses what audience knows that characters don't. The underscore "they can't hear."

Mapping possibilities:
- Subtext/irony gaps → musical tension
- Leitmotifs activate based on audience knowledge, not just scene content
- Character internal states as patterns that modulate music

---

## Implementation Phases

### Phase 1 (Current — In Progress)
- WorldState (static)
- StyleManifest
- TemplateCompiler
- BranchGraph (state only)
- Basic Tidal OSC bridge

### Phase 2 (This Doc)
- Trajectory semantics → extend WorldState
- Information topology → new schema
- Intent/subtext layers → new compilation stage
- Chekhov tracking → new tracking system
- Agent negotiation → full multi-agent system
- Reasoning in branches → extend BranchGraph

### Phase 3 (Future)
- Learning from user choices (which branches get selected)
- Agent self-improvement based on outcomes
- Cross-session narrative memory

---

## Quick Reference: What Enables What

| Concept | Enables |
|---------|---------|
| Trajectories | "Tension building" as continuous parameter |
| Information topology | Hitchcock suspense/irony as computable values |
| Chekhov tracking | Setup/payoff awareness, avoid deus ex machina |
| Intent/subtext | Director-level control ("make this feel tense") |
| Multi-agent | Automated narrative coherence checking |
| Reasoning in branches | "Why did it go this way?" + intentional override |

---

## Related Files

| File | Relationship |
|------|--------------|
| `notes/capability-roadmap.md` | Phase 1 implementation tracking |
| `notes/realtime_video_architecture.md` | Current architecture |
| `notes/proposals/tidal-cycles-integration.md` | Music integration |
| `notes/research/2025-12-26/narrative_engine/claude_01.md` | Source brainstorm |
