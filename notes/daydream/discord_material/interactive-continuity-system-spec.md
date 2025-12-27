# Interactive Continuity System - Cohort Project Spec

Created: 2025-12-21
Updated: 2025-12-21
Status: Multi-world-pack architecture

---

## Strategic Framing

**Cohort demo:** Show the same interactive system running across multiple World Packs. This proves the architecture generalizes - it's a platform, not just one cool project.

**Internal goal:** Everything built here flows into Graffito. The cohort properties are both valid projects AND proving grounds for the infrastructure.

### World Packs Portfolio

| Property | IP Status | LoRA 2.1 | Demo Priority | Notes |
|----------|-----------|----------|---------------|-------|
| **Wobbly Willow Farm** | Clean (original IP) | Uses R&T LoRA | **Primary** | Percy/Sheldon/Daphne - full world pack ready |
| **Rudolph 1964** | Clean (style homage) | Needs training | Secondary | Seasonal, proven quality |
| **Kaiju** | Gray (Shōwa-era homage) | Dataset ready | Optional | Strong aesthetic, IP-careful prompting |
| **Graffito** | Internal only (Mark's IP) | Needs training | Private | The real destination |
| **R&T Style** | Clean (USC permission) | Training now | Style only | LoRA as style engine, not content |

See individual `world-pack-*.md` files for details on each property.

---

## Core Philosophy

**Cool outputs are side effects of pursuing higher utility.**

The interactive demo is not the goal. The goal is a system that:
1. **Explores** narrative/visual state space autonomously
2. **Surfaces** interesting states and failure cases
3. **Enables iteration** on prompts at any paused state
4. **Extracts lessons** that improve World Packs and prompt skills
5. **Recursively self-improves** as learnings feed back into the system

Playable demos, polished clips, even entire films - these emerge as **byproducts** of a generative exploration pipeline that's actually building durable knowledge.

---

## Core Concept

A **continuity-first interactive storytelling** system where an agent **story director** tracks world state and compiles user choices into a **prompt timeline** that renders as a continuous take.

---

## The Problem Being Solved

AI video generates impressive clips but struggles with:
1. **Temporal coherence** - identity/environment drift across shots
2. **Interactive continuity** - changing direction mid-generation without world breaking
3. **State persistence** - the model doesn't "remember" what's in the scene

---

## The Exploration Loop

The system operates in three modes, each feeding the others:

```
┌─────────────────────────────────────────────────────────────────┐
│                    THE EXPLORATION LOOP                         │
│                                                                 │
│   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐    │
│   │   Agents    │ ───▶ │   Explore   │ ───▶ │    Mark     │    │
│   │  traverse   │      │   state     │      │ interesting │    │
│   │   space     │      │   space     │      │   states    │    │
│   └─────────────┘      └─────────────┘      └──────┬──────┘    │
│                                                     │           │
│   ┌─────────────┐      ┌─────────────┐      ┌──────▼──────┐    │
│   │   Update    │ ◀─── │   Extract   │ ◀─── │   Iterate   │    │
│   │   Skills    │      │   lessons   │      │   prompts   │    │
│   │ World Packs │      │             │      │             │    │
│   └─────────────┘      └─────────────┘      └─────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**The video outputs are proofs. The refined World Pack + prompt vocabulary is the durable artifact.**

### Three Modes of Operation

| Mode | Who | Purpose | Output |
|------|-----|---------|--------|
| **Play** | Human | Experience the narrative | Enjoyment, demo |
| **Explore** | Agents | Traverse state space autonomously | Marked interesting states |
| **Refine** | Human + Claude | Iterate prompts at paused states | Improved skills, world pack updates |

### Agent Exploration

Agents traverse the narrative state space autonomously, marking states for human review:

```yaml
exploration_config:
  strategy: breadth_first  # or depth_first, random_walk, reward_guided
  
  marking_criteria:
    - visual_quality: "VLM scores frame aesthetics > threshold"
    - narrative_interest: "unexpected state combination"
    - failure_case: "prompt failed to render intent"
    - edge_case: "unusual but valid state"
    
  on_mark:
    - snapshot_state
    - cache_video_segment
    - log_prompt_and_result
    - queue_for_human_review
```

Agents run overnight. Morning: gallery of interesting states and failure cases to review.

### Refine Mode (Prompt Iteration at State)

When paused at any state, invoke prompt iteration tools:

```
┌─────────────────────────────────────────────────────────────┐
│  STATE: kitchen_choice_A2                                   │
│  Current prompt: "Tony reaches for the door handle..."      │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Render      │  │ Permute     │  │ Compare variants    │  │
│  │ current     │  │ (Claude     │  │ (side by side)      │  │
│  │             │  │  skills)    │  │                     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│                                                             │
│  Variants:                                                  │
│  [A] Original          [B] Front-loaded    [C] Simplified   │
│  [D] More motion vocab [E] Less motion     [F] Alt camera   │
│                                                             │
│  ✓ Mark winner → Update canonical prompt for this beat      │
└─────────────────────────────────────────────────────────────┘
```

Existing prompt permutation skills become integrated tools in the exploration workflow.

### Lesson Extraction → Skills

When iteration finds something that works, capture it:

```yaml
lesson:
  context: "R&T slapstick, object spawn"
  before: "A banana peel appears on the floor"
  after: "A bright yellow banana peel POPS into frame on the wooden floor"
  
  insight: "Object spawns need color + material + action verb"
  
  applies_to:
    - world_packs: [rooster_terry, rudolph]
    - beat_types: [object_spawn]
    
  add_to_skill: prompt-engineering-toolkit/object-spawns.md
```

Over time, exploration builds a **corpus of prompt patterns** feeding back into World Packs and general skills.

### Timeline as Tree (Speculative Branching)

State space is a tree, not a line. System can:
- **Pre-render** probable branches while user is on current path
- **Cache** explored branches for instant scrubbing
- **Rewind** to any branch point and choose differently

```yaml
timeline_tree:
  root: scene_start
  current_node: kitchen_choice_A2
  
  nodes:
    scene_start:
      children: [kitchen_choice]
      
    kitchen_choice:
      type: branch_point
      children: [kitchen_choice_A, kitchen_choice_B]
      
    kitchen_choice_A:
      parent: kitchen_choice
      children: [kitchen_choice_A1, kitchen_choice_A2]
      video_cache: "/cache/kitchen_A_0-10s.mp4"
      prompt_variants: [original, v2_front_loaded, v3_simplified]
      best_variant: v2_front_loaded
      
    kitchen_choice_B:
      parent: kitchen_choice
      children: []  # not yet explored
      video_cache: "/cache/kitchen_B_0-5s.mp4"  # partial, speculative
```

### For Production (Graffito)

This is how you actually develop the show:

1. Define scene in World Logic
2. Agents explore all branches overnight
3. Morning: review marked states, iterate prompts on interesting/broken ones
4. Learnings go into Graffito World Pack
5. Re-run exploration with improved prompts
6. Repeat until quality threshold met
7. Final renders become the actual film

The interactive demo is a **side effect** of building a production pipeline.

---

## The Architecture

**Core Philosophy:** Decouple **Simulation** (deterministic, trustworthy) from **Representation** (generative, hallucinatory). Don't rely on the AI to remember if a door is locked.

### Layer Model

```
┌─────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │  Curated    │  │  Freeform   │  │  Dev Console    │  │
│  │  Choices    │  │  Input      │  │  (State Editor) │  │
│  │  (2-4 opts) │  │  (open box) │  │  (direct access)│  │
│  └─────────────┘  └─────────────┘  └─────────────────┘  │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                    WORLD LOGIC LAYER                    │
│         Characters, locations, props, relationships     │
│         Narrative beats, branch points, state machine   │
│              (Domain-agnostic "truth")                  │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                    STYLE LOGIC LAYER                    │
│         Translates intent → prompt for target LoRA      │
│    "menacing walk" → material vocab + motion vocab      │
│         (Per-world-pack prompt compiler)                │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                    RENDER LAYER                         │
│              LoRA + Scope/Krea pipeline                 │
│                 (Visual output)                         │
└─────────────────────────────────────────────────────────┘
```

**Key insight:** World Logic is portable across styles. Define the narrative once ("protagonist confronts antagonist at the gate, can fight or negotiate"), and the Style Logic layer compiles it differently for each World Pack.

**Style as hot-swappable parameter:** Same state engine, same prompt timeline, same narrative logic - just a different World Pack rendering it. "Flip the style" button demonstrates the architecture's modularity.

### Presentation Modes

| Mode | Interface | Purpose |
|------|-----------|--------|
| **Player** | Curated choices (2-4 options) | Peak experience, vetted for visual/narrative payoff |
| **Creator** | Freeform text input | Explore what the system can handle |
| **Dev Console** | Direct state editor | "The tree is now an elm" - proves architecture is real |

Curated choices are the polished demo. The console proves there's no smoke and mirrors.

### World Logic Layer (Exploded)

```
┌─────────────────────────────────────────────────────────────────┐
│                      WORLD LOGIC LAYER (exploded)               │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    NARRATIVE LOGIC                        │  │
│  │    Genre conventions, beat patterns, arc templates        │  │
│  │    "Slapstick: setup → escalation → payoff → reset"       │  │
│  │    "Noir: tension builds, betrayal, moral ambiguity"      │  │
│  │    Influences: what SHOULD happen for good storytelling   │  │
│  └─────────────────────────┬─────────────────────────────────┘  │
│                            │ shapes                             │
│  ┌─────────────────────────▼─────────────────────────────────┐  │
│  │                  CHARACTER INTERNAL STATE                 │  │
│  │    Emotions, motivations, knowledge, relationships        │  │
│  │    "Rooster: frustrated, doesn't know Terry hid the key"  │  │
│  │    Updates: events + interpretation (slower, stickier)    │  │
│  └─────────────────────────┬─────────────────────────────────┘  │
│                            │ drives                             │
│  ┌─────────────────────────▼─────────────────────────────────┐  │
│  │                  EXTERNAL WORLD STATE                     │  │
│  │    Props, locations, physical positions, visibility       │  │
│  │    "Banana peel: on floor, near door, visible to camera"  │  │
│  │    Updates: actions (fast, concrete)                      │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Different update rates:**

| Layer | Updates when... | Example |
|-------|-----------------|--------|
| External | Every action | "pick up", "drop", "walk to" |
| Internal | Every beat | "sees the peel", "realizes he's been tricked" |
| Narrative | Every arc transition | "setup complete, now escalate" |

**Narrative Logic as genre template:**

Slapstick (R&T):
```yaml
narrative_template: slapstick
beats:
  - setup: establish the gag (banana peel placed)
  - false_safety: character almost avoids it
  - escalation: more peels, or bigger stakes
  - payoff: slip happens, maximum comic timing
  - reset: dignity restored, new gag begins
  
conventions:
  - physical_harm: temporary, no real consequences
  - audience_knows_more: dramatic irony is core
  - rule_of_three: escalate twice, pay off third
```

Emotional drama (Graffito):
```yaml
narrative_template: hero_journey_fragment
beats:
  - ordinary_world: Tony in kitchen, regulated
  - call: sensory overload begins
  - threshold: overwhelm vs. breakthrough choice
  - transformation: flow state achieved (or not)
  
conventions:
  - internal_stakes: emotional regulation is life-or-death
  - magical_realism: graffiti responds to emotional state
  - mentor_wisdom: Grandma's knowledge unlocks path
```

**The narrative layer is what makes curated choices feel "right"** - it knows that in slapstick, the third option should be the payoff, not another setup.

### Four Components

| Component | Role | Implementation |
|-----------|------|----------------|
| **State Engine** (Truth) | Tracks world state in JSON: locations, characters visible, inventory, emotional state | Python/lightweight DB |
| **Resolver** (Director) | Translates state changes into prompt instructions, decides MORPH vs HARD CUT | Rule engine + LLM |
| **Renderer** (Hallucination) | Scope/Krea real-time video with prompt updates at ~2s intervals | WebRTC stream |
| **Perception Unit** (Verifier) | VLM watches stream, confirms hallucination matches truth | Gemini Flash frame sampling |

### Data Flow

```
User Action -> [State Engine] -> [Resolver] -> [Text Prompt] -> [Krea/Scope] -> [User Display]
                                      ^                               |
                                      |                               v
                               [Perception Unit] <---(Sample Frame)---/
```

---

## Transition Strategy

Based on video analysis of LoRA-constrained generation:

| Event Type | Visual Strategy | Implementation |
|------------|-----------------|----------------|
| Local action (pick up item) | MORPH (continuous) | Update prompt text, let pixels flow |
| Object spawn (bird appears) | MORPH (summon) | Inject keywords, accept pop-in as style |
| Location change (enter cave) | HARD CUT (reset) | Send strength:1.0 or blank noise frame before new prompt |
| Dream/magic sequence | MELT (style) | Deliberately allow "bad" morphing |

**Key Insight:** The "stop-motion" aesthetic acts as natural camouflage for frame-to-frame jitter. User brain accepts jerky movement as stylistic intent.

---

## World Pack Format

A **bundle of constraints** that makes a world generatable. See `world-pack-template.md` for the full schema.

**Core components:**
- **Style LoRA** (wan 2.1 for real-time, 2.2 for offline)
- **Style prefix** (opening phrase for every prompt)
- **Material vocabulary** (how things are BUILT in this world)
- **Motion vocabulary** (how things MOVE in this world)
- **Canonical character descriptions** (full description every time)
- **Location descriptions** (reusable environment prompts)
- **Camera language** (shot types that work)
- **Transition rules** (MORPH vs HARD CUT decisions)
- **Reference assets** (training data, style guides)

The LoRA + vocabulary solve the **consistency problem** by constraining the model's imagination.

**Current World Packs:**
- `world-pack-graffito.md` - Internal, most detailed
- `world-pack-rudolph-1964.md` - Public demo ready
- `world-pack-rooster-terry.md` - Needs filling in
- `world-pack-kaiju.md` - Public demo ready

---

## Prompt Timeline Format

```yaml
timeline:
  session: graffito_demo_01
  world_pack: graffito_v2
  
  beats:
    - id: kitchen_anchor
      time: 0-10s
      type: ANCHOR
      prompt: |
        Graffito Mixed-Media Stop-Motion — Tony, a 7-year-old puppet with a 
        PHOTOGRAPHIC CUTOUT face of a young boy with curly hair and expressive 
        green eyes, and a PAINTED PAPER CUTOUT body wearing his signature light 
        blue shirt with a red collar and beige pants, stands in the kitchen...
    
    - id: turn_to_door
      time: 10-20s  
      type: TRANSITION
      prompt: |
        Graffito Mixed-Media Stop-Motion — Tony turns toward the door, his 
        articulated paper body moving with jerky stop-motion steps...
      transition_window: 2s  # blend embeddings during transition
    
    - id: choice_moment
      time: 20-30s
      type: BRANCH_POINT
      trigger: user_choice
      state_check: tony.regulation_level
      options:
        - choice: "Open the door"
          condition: tony.regulation_level > 0.5
          next_beat: door_opens
          state_update:
            location: hallway
            visible_characters: [tony, monk]
        - choice: "Hide under table"
          condition: tony.regulation_level <= 0.5
          next_beat: tony_hides
          state_update:
            tony.position: under_table
            tony.regulation_level: -0.2  # relative change
```

---

## State Schema (Draft)

```json
{
  "world": {
    "location": "kitchen",
    "time_of_day": "evening",
    "weather": null,
    "active_props": ["magic_spray_can", "sketchbook"]
  },
  "characters": {
    "tony": {
      "visible": true,
      "position": "center_frame",
      "emotional_state": "anxious",
      "regulation_level": 0.6,
      "inventory": ["sketchbook"],
      "canonical_description": "Tony, a 7-year-old puppet with a PHOTOGRAPHIC CUTOUT face..."
    },
    "monk": {
      "visible": false,
      "location": "jail",
      "emotional_state": "worried"
    }
  },
  "narrative": {
    "current_beat": "kitchen_anchor",
    "completed_beats": [],
    "branch_history": []
  }
}
```

---

## Perception Unit Logic

**Trigger:** Runs only after major user interactions or every ~5 seconds.

**Blind Spot Timer:** Ignore video feed for ~1.5 seconds after prompt update to allow video to "settle."

**Sample Task:**
```
"The Game State says Tony is holding the 'Magic Spray Can'. 
Look at this frame. Is a spray can visible? 
Is it being held by the character? 
Return TRUE/FALSE only."
```

**Correction Flow:**
1. If FALSE and retries < 3: strengthen prompt with object keywords
2. If FALSE and retries >= 3: log failure, continue (don't block)
3. If critical object: trigger HARD CUT reset

---

## Deliverables (Cohort Project)

1. **Playable vertical slice** with 2-3 branch points
2. **Prompt timeline format** (JSON/YAML spec, documented)
3. **Continuity anchors** convention (what must persist across beats)
4. **World pack** conventions for characters/locations
5. **State engine** prototype (Python, portable)
6. Optional: **Offline render pass** for higher resolution final

---

## Existing Infrastructure

| Tool | Status | Role in System |
|------|--------|----------------|
| Graffito LoRA (v2) | 2.2 trained, 2.1 needed | World locking / style constraint |
| Rudolph LoRA | 2.2 trained, 2.1 needed | World locking / style constraint |
| Kaiju LoRA | 2.2 trained, 2.1 dataset ready | World locking / style constraint |
| Rooster & Terry LoRA | 2.1 training NOW | World locking / style constraint |
| Captioning spec | Proven (Graffito) | Material vocabulary, canonical descriptions |
| Nano Banana Studio | Working | Frame generation, keyframe selection |
| Prompt by API | Working | Batch T2V/I2V via ComfyUI |
| Scope | Learning | Real-time rendering target |

---

## Implementation Phases

### Phase 1: The "Blind" Engine (Text Only)
- Build JSON schema for locations, entities, state
- Script Resolver logic: user input → new state → prompt string
- **Goal:** Play the game in a text terminal to verify logic works

### Phase 2: The Manual Connection
- Connect prompt output to Scope/Krea API
- Test MORPH vs HARD CUT commands
- Verify reset latency (<500ms target)
- **Goal:** See state changes reflected in video

### Phase 3: The Perception Loop
- Set up frame sampling from video stream
- Feed frames to Gemini Flash, pipe output to Resolver
- **Goal:** Game auto-corrects when objects fail to spawn

### Phase 4: Interactive Demo
- Simple UI for branch choices
- 2-3 branch points in a short scene
- Record sessions for cohort demo
- **Goal:** Playable vertical slice

---

## Open Questions

1. **Scope API surface:** What's the actual interface for prompt updates mid-stream?
2. **Latency budget:** How fast can we update prompts without visible discontinuity?
3. **State granularity:** How much detail in state before it becomes unwieldy?
4. **Branch complexity:** How many concurrent state variables before combinatorial explosion?
5. **Perception reliability:** How often does VLM verification fail? Cost at scale?

---

## References

- Gemini session: "Alien Cable" founding technical report
- GPT-5 Pro session: Bundle 2 (Agent Story Director)
- Daydream application text
- Graffito project briefing
- Krea Realtime 14B architecture notes

## Related Files

- `world-pack-template.md` - Empty template for new properties
- `world-pack-wobbly-willow-farm.md` - **Primary demo**, full detail, original IP
- `world-pack-graffito.md` - Internal, full detail
- `world-pack-rudolph-1964.md` - Public demo
- `world-pack-kaiju.md` - Public demo
