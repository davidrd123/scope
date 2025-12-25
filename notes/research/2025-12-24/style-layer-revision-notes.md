# Style Layer Revision Notes

**Date:** 2025-12-24
**Purpose:** Review RAT material, make observations about style/prompting structure

## Source Material

```
incoming/style/RAT/
├── Captioning/
│   ├── RAT_pat_video_v4.md      # Captioning rubric (LLM instruction sheet)
│   └── rat_video_captions.txt   # Example captions
├── Prompting/
│   ├── RAT_Prompting_Guidelines.md   # User-facing prompting guide
│   └── WobblyWillowFarm.md           # Translation rubric for new IP
└── Prompts/
    ├── rat_wild.yaml       # Wild creature prompts
    ├── rat_fury_road.yaml
    ├── rat_fury_road_xpose.yaml
    └── rat_percy.yaml      # Wobbly Willow Farm prompts
```

---

## Observations

### 1. Captioning Rubric (RAT_pat_video_v4.md)

**What it is:** An LLM instruction sheet for generating training captions. NOT a vocab dictionary.

**Notable structures:**

- **Trigger phrase**: `"Clay-Plastic Pose-to-Pose Animation —"` prepended to every caption
- **Mandatory structure** (malleable order): Primary Element → Secondary → Motion → Aesthetic
- **Character-specific verbs**: Rooster uses "snaps, jerks" vs Terry uses "settles, drifts"
- **ALL CAPS for key visuals**: CLEAN SILHOUETTES, RIM LIGHTING, POSE-TO-POSE
- **Appearance variant handling**: Additive, Transformative, Compositional modifiers
- **The Golden Rule**: Translate interpretation → objective physical description
  - NOT: "Rooster acts proud"
  - YES: "Rooster stands with his chest puffed out, silhouette angular"

**Observation:** This is a meta-instruction for how an LLM should think about the domain, not a mapping table.

### 2. Prompting Guidelines (RAT_Prompting_Guidelines.md)

**What it is:** A simplified user-facing guide for writing prompts in the R&T style.

**Notable structures:**

- **Trigger phrase required** with specific punctuation (em dash)
- **Structural flow**: ESTABLISH → STYLE → POSE/ACTION → LIGHTING → CONCLUDE
- **Length tiers**: Quick (40-70 words), Standard (80-130), Complex (140-180)
- **Character shape language**: Rooster=angular, Terry=rounded
- **Anchor hierarchy**:
  - Core anchors (2-3 per prompt): SMOOTH CLAY-PLASTIC SHADERS, CLEAN SILHOUETTES
  - Supporting anchors (1-3): WARM PASTEL PALETTE, RIM LIGHTING
  - Specialized terms (context-specific): ROOSTER'S GADGET, REACTION HOLD
- **Lighting formula**: Name the palette + one key effect
- **"Off-Road" guidance**: Use the style on other subjects by keeping grammar, changing nouns

**Observation:** Shows that vocabulary has hierarchy (required vs optional), and that there's a structural flow to how prompts are built.

### 3. Translation Rubric (WobblyWillowFarm.md)

**What it is:** A guide for adapting the R&T style to a new IP (different characters, same aesthetic).

**Notable structures:**

- **Character dossier**: Maps archetypes to style-appropriate descriptions
  - The Schemer → Percy the Pig (blocky, angular trotters)
  - The Oblivious One → Sheldon the Sheep (fluffy, rounded, S-curves)
  - The Wildcard → Daphne the Duck (zany, long flexible neck)
- **Lighting matrix**: Scenario → complete micro-template
  - Day Exterior: "lit by soft warm key light from a SINGLE LARGE CIRCULAR DISC; pale blue AMBIENT HUE fills the sky; shadows read as SOFT POSTERIZED SHADOW SHAPES"
- **Advanced vocabulary "texture pack"**: Surface/Material, Line/Ink, Motion, Composition terms
- **Contextual vs Global cues**: Some describe specific objects, others describe the whole frame

**Observation:** Shows that styles can have "translation layers" for new content. Character descriptions aren't just names—they're full prose with shape language baked in.

### 4. YAML Prompt Files

**What they are:** Actual prompt batch files with defaults and job lists.

**Notable structure:**

```yaml
defaults:
  t2v_lora_low: rat/rat_v1_wan_low_9250steps.safetensors
  t2v_lora_high: rat/rat_v1_wan_high_3250steps.safetensors
  negative_prompt: |
    photorealistic, photography, complex textures, ...

jobs:
  - title: "RAT-Percy-Fire-01-FuseSetup"
    prompt: |
      Clay-Plastic Pose-to-Pose Animation — A medium wide shot...
```

**Observation:** LoRA paths, negative prompts, and inference settings are all part of the "style" definition. The prompts themselves are 80-200 words and follow the structural flow.

---

## Key Patterns Observed

### Pattern 1: Vocab Has Hierarchy
Not all terms are equal. Some are required, some are selective, some are context-specific. A flat dict doesn't capture this.

### Pattern 2: Lighting Is Templated
Time of day doesn't map to a single word—it maps to a complete micro-template with key light + ambient + graphic effect.

### Pattern 3: Characters Have Prose Descriptions
Character identification isn't just a name. First mentions include full descriptions with shape language. Motion verbs are character-specific.

### Pattern 4: Structural Flow Exists
There's an order to how prompts are built (establish → style → action → lighting → conclude). The order can be adapted based on what's most prominent.

### Pattern 5: Meta-Instructions Drive Quality
The "Golden Rule" (interpretation → physical description) isn't vocab—it's a meta-instruction that guides how to think about the domain.

### Pattern 6: Style ≠ Just LoRA
The "style" includes: trigger phrase, vocab hierarchy, lighting templates, character profiles, structural guidance, negative prompts, and inference settings.

---

## Questions Raised

1. **Is vocab hierarchy universal?** Do all styles have "required" vs "optional" anchors, or is this R&T-specific?

2. **How portable are lighting templates?** Does every style need a lighting matrix, or just styles with strong time-of-day differentiation?

3. **What about styles without characters?** R&T is character-driven. What does a style manifest look like for landscapes, abstract art, etc.?

4. **Should the compiler be style-aware?** The structural flow and Golden Rule suggest the compiler needs domain-specific logic, not just template substitution.

5. **How do translation layers work?** WobblyWillowFarm shows translating archetypes to a style. Is this a common pattern we should support?

6. **What's the minimum viable style definition?** R&T has extensive material. What's the simplest style that still works?

---

## What Our Current Design Handles

- Trigger words ✓
- Vocab dictionaries (flat) ✓
- LoRA path and default scale ✓
- Max prompt tokens ✓
- Priority order (list, not flow) ✓
- InstructionSheet for LLM ✓

## What's Missing or Different

- Vocab hierarchy (required/core/supporting/specialized)
- Lighting as micro-templates
- Character profiles with shape language and motion verbs
- Structural flow (not just priority)
- Meta-instructions (Golden Rule)
- Negative prompts as part of style
- Translation layer for new IPs

---

## Raw Notes for Further Discussion

- The captioning rubric is ~113 lines. That's a lot of instruction. Is our InstructionSheet format rich enough?
- The prompting guidelines are user-facing, separate from the captioning rubric. Two levels of documentation.
- R&T has specific comedy structure (setup/payoff/reaction). Other styles won't have this. Don't over-index.
- The YAML files show that prompt batching is part of the workflow. Integration with Scope pipeline?
- Negative prompts in rat_wild.yaml are extensive (30+ lines). This is clearly tuned.

---

## Kaiju Comparison (Second Slice)

Reviewed: `incoming/style/Kaiju/` - Japanese tokusatsu kaiju films (Godzilla franchise)

### Same Structural Patterns as R&T

| Pattern | R&T | Kaiju |
|---------|-----|-------|
| Trigger phrase | `Clay-Plastic Pose-to-Pose Animation —` | `Japanese Kaiju Film —` |
| ALL CAPS anchors | CLEAN SILHOUETTES, POSE-TO-POSE | SUITMATION, MINIATURE CITYSCAPE |
| First mention | "the angular orange rooster, Rooster" | "the reptilian charcoal-gray kaiju, Godzilla" |
| Structural flow | ESTABLISH → STYLE → POSE → LIGHTING → CONCLUDE | ESTABLISH → KAIJU & SETTING → ACTION & SFX → LIGHTING → CONCLUDE |
| Length tiers | Quick 40-70, Standard 80-130, Complex 140-180 | Quick 50-80, Standard 90-150, Complex 160-200 |
| Anchor hierarchy | Core/Supporting/Specialized | Essential/Common/Situational |
| Off-road guidance | "keep grammar, change nouns" | "keep grammar, change nouns" |

### Different Domain-Specific Content

**R&T domain:**
- Comedy patterns: Setup (A), Payoff (B), Reaction (C)
- Character shape language: angular vs rounded
- Aesthetic vocabulary: SMOOTH CLAY-PLASTIC SHADERS, WARM PASTEL PALETTE

**Kaiju domain:**
- Action patterns: Kaiju-First, Destruction-First, Scale-First
- Era variants: Shōwa, Heisei, Millennium, Reiwa (affects craft vocabulary)
- Material/SFX vocabulary: LATEX RUBBER SUIT, PYROTECHNICS, MINIATURE
- OOD distance concept: low/medium/high with explicit guidelines
- Multi-kaiju handling: primary/secondary/group
- Temporal connectors: "First... Then... Finally..."

### Universal Patterns Confirmed

These appear in BOTH domains, suggesting they're structural rather than content:

1. **Trigger phrase** - domain-specific words, same structural role
2. **Anchor hierarchy** - 3 tiers with different usage frequency
3. **First mention pattern** - descriptor + name
4. **Structural flow** - similar 5-step rhythm
5. **Length tiers** - quick/standard/complex with word counts
6. **Off-road guidance** - same principle
7. **One novel thing at a time** - for OOD/unusual prompts

### Domain-Specific Patterns (NOT universal)

1. **Comedy patterns** - R&T only (setup/payoff/reaction)
2. **Era variants** - Kaiju explicit, R&T implicit (character evolution)
3. **Destruction language** - Kaiju emphasizes "how things break" with material terms
4. **Temporal connectors** - Kaiju uses explicit "First/Then/Finally", R&T less so

### Implication

The style layer should support:
- Trigger phrase (string)
- Anchor hierarchy (3 tiers with frequency guidance)
- Character profiles (descriptor + name pattern)
- Structural flow hints (ordered list of steps)
- Length tiers (3 levels with word counts)
- Off-road/OOD guidance (prose instructions)

But should NOT bake in:
- Comedy patterns (domain-specific)
- Era variants (domain-specific)
- Material vocabulary categories (varies by domain)

---

## TMNT Comparison (Third Slice)

Reviewed: `incoming/style/TMNT/` - Graffiti sketchbook animation style

### Same Structural Patterns (Confirmed Universal)

| Pattern | R&T | Kaiju | TMNT |
|---------|-----|-------|------|
| Trigger phrase | `Clay-Plastic Pose-to-Pose...` | `Japanese Kaiju Film —` | `Graffiti Sketchbook Animation —` |
| Anchor tiers | Core/Supporting/Specialized | Essential/Common/Situational | Essential/Common/Situational |
| First mention | "[descriptor], [Name]" | "[descriptor], [Name]" | "[descriptor], [Name]" |
| Structural flow | 5-step | 5-step | 5-step (ESTABLISH → STYLE → ACTION → LIGHTING → CONCLUDE) |
| Length tiers | Quick/Standard/Complex | Quick/Standard/Complex | Quick 40-70 / Standard 80-140 / Complex 150-200 |
| Off-road guidance | "keep grammar, change nouns" | "keep grammar, change nouns" | "keep grammar, change nouns" |
| OOD concept | implicit | explicit | explicit ("attention budget") |

### TMNT-Specific Patterns (NOT universal)

**"Two Pillars" framework:**
1. Accurate Identification - Rule of Foundational Identity
2. Dynamic Action - specific character motion language

**Shot-type patterns:**
- Setting-First shots: environment dominates
- Character-First shots: character introduced then context
- Object-First shots: object/vehicle dominates

**Identification protocol:**
- More detailed than R&T or Kaiju
- "Rule of Foundational Identity" - each character's base look must be established
- Mask color as primary identifier (Leonardo = blue, Raphael = red, etc.)
- Weapon and personality mapped to motion verbs

**Aesthetic vocabulary:**
- SPRAY PAINT TEXTURE, GRAFFITI LINEWORK, URBAN GRIT
- STREET ART SHADOWS, NEON GLOW, CONCRETE TEXTURE
- Marker/pen stroke artifacts

---

## Final Synthesis: Universal vs Domain-Specific

### Universal Patterns (ALL 3 domains confirm)

These are structural and should be SLOTS in the style layer:

1. **Trigger phrase** - string prepended to all prompts
2. **3-tier anchor hierarchy** - essential (always use), common (usually use), situational (when appropriate)
3. **First mention pattern** - `[descriptor], [Name]` for character introduction
4. **5-step structural flow** - ESTABLISH → STYLE → ACTION → LIGHTING → CONCLUDE
5. **3 length tiers** - quick/standard/complex with word count ranges
6. **"Keep grammar, change nouns"** - off-road/style-transfer guidance
7. **OOD/attention budget** - one novel thing at a time

### Domain-Specific Patterns (VARY by style)

These should come from instruction sheets, NOT baked into code:

- **R&T:** Comedy patterns (setup/payoff/reaction), shape language (angular/rounded)
- **Kaiju:** Era variants (Shōwa/Heisei), material/SFX vocab, temporal connectors
- **TMNT:** Shot-type patterns, identification protocol, urban/street art vocab

### Key Insight

**The style layer should provide SLOTS for universal patterns. Domain-specific content is loaded from InstructionSheet markdown files.**

Current design already does this via `LLMCompiler + InstructionSheet`. The missing piece is making the universal structure more explicit in `StyleManifest`:

```python
# Current: flat vocab dicts
vocab = {"material": {...}, "motion": {...}}

# Proposed: hierarchical anchors
anchors = {
    "essential": ["TRIGGER", "ANCHOR1", "ANCHOR2"],  # always use
    "common": ["ANCHOR3", "ANCHOR4"],                 # usually use
    "situational": ["ANCHOR5", "ANCHOR6"]             # when appropriate
}

# Proposed: length tiers
length_tiers = {
    "quick": (40, 70),
    "standard": (80, 140),
    "complex": (150, 200)
}

# Proposed: structural flow hints
flow = ["ESTABLISH", "STYLE", "ACTION", "LIGHTING", "CONCLUDE"]
```

The LLM still does the actual compilation. These hints just give it guardrails.

---

## Decision: InstructionSheet Carries the Weight (2025-12-24)

**Decided:** Keep StyleManifest minimal. All domain knowledge lives in InstructionSheet prose.

**StyleManifest (minimal):**
```python
@dataclass
class StyleManifest:
    name: str
    trigger_phrase: str
    lora_path: str
    instruction_sheet_path: str
    negative_prompt: str | None = None
```

**InstructionSheet (rich prose):**
- Anchor hierarchy (essential/common/situational)
- Length tiers and word counts
- Structural flow guidance
- Character identification patterns
- Off-road/transfer guidance
- OOD/attention budget rules
- Domain-specific patterns (comedy beats, era variants, shot types, etc.)

**Rationale:** We're leaning on the LLM to do all the thinking. Prose instructions are more flexible than rigid schemas. The 7 universal patterns become a **checklist for writing good instruction sheets**, not code slots.

**Implication:** Current scaffolding is mostly right. We need real instruction sheets that follow the universal patterns, not schema changes.
