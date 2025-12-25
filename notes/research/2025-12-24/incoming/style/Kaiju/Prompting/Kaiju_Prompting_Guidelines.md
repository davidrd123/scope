# Japanese Kaiju Film / Kaiju LoRA Prompting Guidelines

**Status:** v1.1 — Caption‑aligned + era/OOD extensions  
**Owner:** davidrd  
**Model:** Kaiju LoRA (T5 text encoder)

Shared prompt patterns reference: `WorkingSpace/davidrd/OutsideStruct/PromptWriting/Prompt_Writing_Patterns_Shared.md`

Strict Shōwa‑mode cartridge spec (agent‑style, with sanctioned tables): `WorkingSpace/davidrd/Contexts/Kaiju/Prompting/Protocols/Japanese_Kaiju_Film_Prompting_Protocol_v5_2.md`

---

## Overview

This guide teaches you to write prompts in the **style and grammar** of the Kaiju training captions while **strategically simplifying** for effective prompting. The goal is not exhaustive documentation of every miniature and effect, but focused direction for the shot: which kaiju, what they’re doing, what gets destroyed, and how it feels.

**Key Principle:** Use the captions’ cinematic vocabulary (SUITMATION, MINIATURE CITYSCAPE, PYROTECHNICS, etc.) and sentence structure, but selectively. Lead with the main beat, anchor kaiju identity and SFX, and describe one clear moment or micro‑sequence.

Also preserve the T5 optimizations from captioning:

- No markdown in the prompt itself (no `**`, backticks, lists).  
- Avoid quotation marks unless grammatically necessary.  
- Use ALL CAPS for iconic SFX and material terms.  
- Character names are standard capitalized words (Godzilla, Gigan, Minilla).

**How to use this doc:**

- For day‑to‑day prompting: focus on Sections 1–7 and 9, plus the Quick Checklist (Section 14).  
- For deeper control (era, off‑road concepts, term catalog): Sections 11–13 and 15 act as appendices.

---

## 1. Trigger Phrase (Required)

**Format:**

```text
Japanese Kaiju Film — 
```

- Always start with this exact trigger, including the trailing space.  
- Use the em dash (—), not a hyphen (-).  
- Follow immediately with your shot description.

**Why it matters:** This phrase activates the Kaiju‑film style. Without it, you’re more likely to get generic monsters or disaster imagery instead of tokusatsu‑driven miniatures and suit work.

---

## 2. Narrative Priority (What Goes First)

The captions give **most attention to the first thing they describe**. Decide what the shot is really about and open with that:

- The kaiju’s pose / move  
- The destruction / impact  
- The scale contrast vs. humans or vehicles

Three patterns cover most clips:

### 2.1 Pattern A: Kaiju‑First (Pose / Attack / Reaction)

Use when the main beat is what a specific kaiju is doing.

**Structure:**

```text
[Shot type] of [kaiju descriptor + NAME] [primary action] in [miniature setting]. [SFX + materials]. [Lighting / smoke].
```

**Example:**

> Japanese Kaiju Film — In a low‑angle wide shot, the reptilian, charcoal‑gray kaiju, Godzilla, stands atop a MINIATURE CITYSCAPE, raising his arms in an anthropomorphic, triumphant pose. SUITMATION movement sends his PEBBLED‑TEXTURE LATEX RUBBER SUIT swaying as orange PYROTECHNICS burst from a MINIATURE BUILDING below. Thick black smoke billows around him under a hazy, dusk sky.

Use when the kaiju’s pose, gesture, or attack is the core of the shot.

---

### 2.2 Pattern B: Destruction‑First (Impact / Beam / Collapse)

Use when the shot is really about **what gets hit or destroyed**.

**Structure:**

```text
[Shot type] on [miniature target] [impact / destruction]. [Which kaiju / weapon causes it]. [SFX terms + debris behavior]. [Camera or scale note].
```

**Example:**

> Japanese Kaiju Film — A static, high‑angle shot of a dense MINIATURE CITYSCAPE captures a tall MINIATURE OFFICE TOWER as it erupts in a GASOLINE FIREBALL. GODZILLAS BLUE ATOMIC BREATH cuts across the frame as an ANIMATED OPTICAL OVERLAY, slicing through the tower while PLASTER AND BALSA WOOD MINIATURES collapse into CGI PARTICLE DEBRIS. SPARKLER‑LIKE PYROTECHNICS scatter from the impact while the camera shakes to emphasize the destruction.

Use when the key idea is “this thing explodes / collapses / gets sliced by a beam.”

---

### 2.3 Pattern C: Scale‑First (Humans / Vehicles vs. Kaiju)

Use when you want to emphasize **scale and compositing** between kaiju and humans or vehicles.

**Structure:**

```text
[Shot type] from [human / vehicle POV] showing [kaiju presence] beyond. [Compositing / miniature notes]. [Emotional tone].
```

**Example:**

> Japanese Kaiju Film — From a low‑angle shot behind fleeing civilians on a city street, the towering silhouette of Gaira looms over a MINIATURE BUILDING ROW. The humanoid kaiju covered in shaggy GREEN SYNTHETIC FUR is composited behind real foreground actors, his SUITMATION movements slow and menacing. Neon signs flicker as SPARKLER‑LIKE PYROTECHNICS burst from a MINIATURE POWER LINE, bathing the scene in harsh, orange light.

Use when you care most about “humans tiny, kaiju huge” or explicit tokusatsu compositing.

---

## 3. Structural Flow

Aim for this rhythm:

```text
ESTABLISH SHOT → KAIJU & SETTING → ACTION & SFX → LIGHTING & SMOKE → CONCLUDE
```

- **ESTABLISH SHOT:** Shot type + basic location (wide low‑angle on a MINIATURE CITYSCAPE, medium shot in a studio water tank, etc.).  
- **KAIJU & SETTING:** Identify the key kaiju using the caption formula; anchor the miniature environment (airport, harbor, rural hillside).  
- **ACTION & SFX:** One clear beat or micro‑sequence, plus SFX craft (SUITMATION, WIREWORK, PYROTECHNICS, ANIMATED OPTICAL OVERLAY).  
- **LIGHTING & SMOKE:** One main light idea (night firelight, gray overcast, hazy studio backlight) plus smoke/fog if relevant.  
- **CONCLUDE:** End state or a simple continuation (the kaiju holds the pose, the building continues burning, the camera holds on drifting smoke).

You don’t have to keep this order strictly; SFX and lighting can weave through the description. Just hit all beats once.

Optionally, end with a single, plain‑language **thematic capstone** sentence such as “The scene feels like a playful Shōwa‑era wrestling match” or “The moment plays as a grim, Heisei‑style disaster tableau.”

---

## 4. Length Guidelines

Captions run ~75–300 words per clip; prompts can be shorter and more focused.

| Type           | Word Count | Use Case                                      |
| -------------- | ---------: | --------------------------------------------- |
| **Quick Beat** |      50–80 | Simple pose, one beam, or a single explosion |
| **Standard**   |     90–150 | One kaiju vs. city or vs. another kaiju      |
| **Complex**    |    160–200 | Multi‑kaiju clash, multiple impacts, rich set|

Guideline: if you’re describing **more than one full “First / Then / Finally” sequence**, you’re drifting back toward caption‑level detail. Compress to the most important 1–2 beats.

---

## 5. Kaiju Identification

Mirror the caption rules so the LoRA sees familiar patterns.

### 5.1 First Mention

```text
[descriptor phrase], [Name]
```

Pulled from the profiles in the captioning system prompt.

**Examples:**

- “the reptilian, charcoal‑gray kaiju, Godzilla”  
- “the robotic kaiju covered in riveted, SILVER‑PAINTED MOLDED PLASTIC/FIBERGLASS PLATES, Mechagodzilla”  
- “the humanoid kaiju covered in shaggy GREEN SYNTHETIC FUR, Gaira”  
- “the small, chubby gray‑green kaiju with a pug‑like face, Minilla”

### 5.2 Subsequent Mentions

Use the name alone, or a shortened descriptor:

- “Godzilla turns his head toward the burning MINIATURE BUILDING.”  
- “Mechagodzilla hovers above the MINIATURE CITYSCAPE.”  
- “Gaira raises his fur‑covered arms.”  
- “Minilla puffs his cheeks and tries to fire MINILLAS ATOMIC SMOKE RINGS.”

### 5.3 Multi‑Kaiju Scenes & Focus

Multi‑kaiju battles are common, but prompts still need a **primary subject**:

- For two kaiju, pick one as the lead (“Godzilla grapples with Anguirus…”).  
- For three or more, treat them as:
  - One primary (“Godzilla”),  
  - One clearly described secondary (“Anguirus at his side”),  
  - Everyone else as a group (“their allied kaiju celebrate in the background”).

Avoid giving equal, exhaustive detail to four monsters at once. Instead:

- Fully describe one kaiju’s action and expression.  
- Give shorter phrases for others (“Rodan circles overhead”, “King Caesar pumps his SYNTHETIC FUR‑covered arms in victory”).

### 5.4 Humans, Military, and Vehicles

Captions treat humans and machines as **scale cues** and **compositing context**, not co‑equal stars.

Keep them lowercase and concise:

- “soldiers crouch in the reeds”  
- “civilians run down the street”  
- “a tank column advances along the MINIATURE highway”  
- “a MASER CANNON battery fires from the hillside”  
- “the SUPER X hovers above the burning MINIATURE CITYSCAPE”

Use them primarily in Pattern C (Scale‑First): foreground humans / hardware, background kaiju.

---

## 6. Style Anchor Integration

Don’t spam every term. Pick a **focused set of 4–6 ALL‑CAPS anchors** that match your scene.

### 6.1 Core SFX / Craft Anchors (2–3 per prompt)

These define the kaiju‑film look:

- SUITMATION  
- LATEX RUBBER SUIT / PEBBLED‑TEXTURE LATEX RUBBER SUIT  
- MINIATURE CITYSCAPE / PLASTER AND BALSA WOOD MINIATURES  
- SPARKLER‑LIKE PYROTECHNICS / LARGE‑SCALE GASOLINE FIREBALLS  
- WIREWORK / WIRE‑ASSISTED FLIGHT  
- MATTE PAINTING  
- ANIMATED OPTICAL OVERLAY / OPTICAL BEAM EFFECT / COMPOSITED DIGITAL BEAM  
- CGI PARTICLE DEBRIS (for later eras)

**Integration examples:**

- “The Shōwa‑era SUITMATION puppet stomps through a dense MINIATURE CITYSCAPE, kicking up PLASTER AND BALSA WOOD MINIATURE debris.”  
- “GODZILLAS BLUE ATOMIC BREATH, rendered as an ANIMATED OPTICAL OVERLAY, slices across the frame, triggering SPARKLER‑LIKE PYROTECHNICS on the buildings below.”

### 6.2 Era / Tone Anchors (0–2 per prompt)

Use sparingly to suggest era (see Section 12 for a full cheat sheet):

- “Shōwa‑era SUITMATION puppet in a brightly lit MINIATURE CITYSCAPE”  
- “Heisei‑era Godzilla with an ANIMATRONIC HEAD surrounded by massive PYROTECHNICS”

Often it’s enough to describe craft and mood without naming the era explicitly.

### 6.3 Iconic Attacks & Tech (1–3 per prompt)

From the “Key Visuals” list:

- GODZILLAS ATOMIC BREATH (BLUE / RED / PURPLE)  
- KING GHIDORAHS GRAVITY BEAMS  
- MOTHRA'S POISON SCALES  
- MINILLAS ATOMIC SMOKE RINGS  
- TITANOSAURUSS HURRICANE‑FORCE WIND  
- MECHAGODZILLAS SPACE BEAMS  
- OXYGEN DESTROYER, MASER CANNON, SUPER X

Guidelines:

- Pick one primary attack per shot (two max if they interact).  
  - **GOOD (two interacting):** “GODZILLAS BLUE ATOMIC BREATH collides with KING GHIDORAHS GRAVITY BEAMS in a burst of ANIMATED OPTICAL OVERLAYS.”  
  - **BAD (unrelated):** “Godzilla fires GODZILLAS BLUE ATOMIC BREATH while Rodan uses TITANOSAURUSS HURRICANE‑FORCE WIND” — unrelated attacks from mismatched kaiju/abilities.  
- Specify color and rendering style when helpful: “GODZILLAS BLUE ATOMIC BREATH as a COMPOSITED DIGITAL BEAM.”  
- Pair beam names with debris behavior on impact.

**Total anchor count:** aim for **4–6 ALL‑CAPS tokens** total (mix of materials, techniques, and iconic effects).

### 6.4 Destruction & Debris Language

The LoRA cares about **how things break**, not just that they explode. Use caption‑like material language:

- “PLASTER dust erupts”  
- “BALSA WOOD fragments fly”  
- “SPARKLER‑LIKE PYROTECHNICS burst on the suit’s chest”  
- “LARGE‑SCALE GASOLINE FIREBALLS engulf a MINIATURE BUILDING PROP”

Helpful templates:

- Building impact: “A MINIATURE BUILDING PROP erupts in a GASOLINE FIREBALL, shattering its PLASTER facade as BALSA WOOD fragments rain down through a cloud of gray dust.”  
- Beam strike on hillside: “GODZILLAS BLUE ATOMIC BREATH slams into a rocky MINIATURE hillside, triggering a flash, SPARKLER‑LIKE PYROTECHNICS, and a spray of PLASTER debris.”

Use 2–3 destruction phrases per prompt; more becomes noise.

Props such as MASER CANNONS, tanks, helicopters, and the SUPER X can also be **active**, not just scenery—let them recoil, tilt, track targets, or flare their engines so they participate in the destruction.

For especially complex destruction or beam effects, it’s safe to use occasional **technical analogies** from physics or film craft (for example, cracks spreading like a city‑scale Lichtenberg figure, or a GASOLINE FIREBALL blooming like a weapons‑range test explosion) as long as you stay within live‑action tokusatsu language rather than other art mediums (oil painting, pixel art, etc.).

---

## 7. Action Sequencing

### 7.1 Present Tense, Third Person

Match the captions:

- GOOD: “Godzilla raises his arms, then slams his tail into a MINIATURE BRIDGE.”  
- BAD: “Godzilla raised his arms and slammed his tail…”

---

### 7.2 Sequential Beats

Reuse caption connectors:

- “First,” / “Initially,” — starting state  
- “Then,” / “Next,” — progression  
- “Finally,” / “Throughout,” — resolution or ongoing behavior

**Example:**

> First, Mechagodzilla hovers above the MINIATURE CITYSCAPE, arms and legs tucked in. Then, he tilts his head toward the camera and fires MECHAGODZILLAS SPACE BEAMS as ANIMATED OPTICAL OVERLAYS from his visor. Finally, SPARKLER‑LIKE PYROTECHNICS erupt along the rooftops as PLASTER debris collapses into the streets.

Limit yourself to **one micro‑sequence** (1–3 beats). If you have three separate mini‑scenes, you’re back in caption territory.

---

### 7.3 Simultaneous Elements

Use commas, not extra “First/Then,” for simultaneous events:

- “Godzilla strides forward through the MINIATURE harbor, water splashing around his legs, while a MASER CANNON battery fires streaks of blue light from the shore.”

---

### 7.4 Kaiju Wrestling & Combat Choreography

Shōwa‑era fights are **giant rubber‑suit wrestling matches**.

- Emphasize telegraphed, anthropomorphic moves (“boxing‑style punches”, “over‑the‑shoulder throw”).  
- Emphasize stiff, lumbering motion (“slow, deliberate SUITMATION gait”).  
- Describe 1–2 key moves, not a long combo.

Example:

> First, Godzilla locks Anguirus in a clumsy wrestling hold. Then, he pivots and throws the quadrupedal kaiju over his shoulder, sending him crashing into the MINIATURE hillside in a burst of PLASTER dust.

---

### 7.5 Emotional Tone & Body Language

Tone varies by era (Shōwa playful, Heisei serious), but one emotionally clear phrase is enough:

- “Minilla flails his arms in a frantic, toddler‑like tantrum.”  
- “Godzilla’s rounded face tilts down in a moment of weary concern.”  
- “Rodan’s stiff wings beat furiously with panicked urgency.”

Aim for this kind of emotionally specific language, applied to **body language and suit performance** rather than generic labels.

---

## 8. Lighting, Smoke, and Camera

### 8.1 Lighting Formula

Captions describe:

1. Time of day / environment  
2. One key light source (direction, color, quality)  
3. Atmosphere (smoke, haze, weather)

Pattern:

```text
[time of day] scene lit by [primary light source] with [color / quality], [smoke/fog/atmosphere behavior].
```

Examples:

- “a night scene lit by burning MINIATURE BUILDINGS and orange SPARKLER‑LIKE PYROTECHNICS, thick black smoke rolling into the sky”  
- “flat, gray overcast daylight over a MINIATURE AIRPORT, soft light emphasizing PLASTER textures”  
- “harsh studio backlight catching the PEBBLED‑TEXTURE LATEX RUBBER SUIT as dust hangs in the air”

Era cues often ride on lighting (see Section 12).

### 8.2 Smoke, Atmosphere, and Particulate

Smoke and dust define tokusatsu destruction:

- “thick gray practical smoke billows from behind the MINIATURE buildings”  
- “fine PLASTER dust hangs in the air after the collapse”  
- “artificial snow swirls around the SUITMATION puppets”

Use 1–2 phrases per prompt to avoid clutter.

### 8.3 Camera Work

Follow caption grammar: **shot type first**, then optional camera or edit notes.

- Shot types: “wide, low‑angle shot”, “static high‑angle”, “medium shot”, “tight close‑up”, “over‑the‑shoulder”.  
- Camera behaviors: “static camera”, “slow push‑in”, “brief handheld shake”, “quick cut to a close‑up”.

Constraint: describe **at most one explicit camera move**, or keep it static.

---

## 9. What to Simplify vs. Captions

Captions exist to document **everything**. Prompts exist to **steer**.

### 9.1 Always Include

1. `Japanese Kaiju Film —` trigger  
2. Primary kaiju (name + first‑mention descriptor)  
3. Primary beat (pose, attack, destruction, or scale moment)  
4. 2–3 core craft anchors (SUITMATION, MINIATURE CITYSCAPE, PYROTECHNICS, etc.)  
5. Lighting / atmosphere (time of day + key source + smoke/fog)

### 9.2 Include When Relevant

- Shot type and angle (wide low‑angle, static high‑angle, etc.)  
- Human / vehicle presence for scale  
- Era hint (Shōwa vs Heisei)  
- Specific iconic weapons (ATOMIC BREATH, GRAVITY BEAMS, MASER CANNON)  
- Compositing details (MATTE PAINTING background, WIREWORK, optical overlays)

### 9.3 Skip or Compress

- Long chains of micro‑movements (“he turns his head slightly, then slightly more…”).  
- Exhaustive miniature cataloging (every building, road, and tree).  
- Repeating similar SFX phrases once they’re established.  
- Entire edit sequences; keep 1–2 of the strongest cuts or angles.

If a caption has three separate “First / Then / Finally” arcs, pick **the one that defines the shot**.

For multi‑kaiju scenes, compress supporting actions:

- Caption‑style: “Rodan circles, King Caesar dances, Anguirus stomps, Godzilla roars…”  
- Prompt‑style: “Godzilla stands over the fallen KING GHIDORAH as Rodan and King Caesar celebrate in the hazy background.”

---

## 10. Caption → Prompt Examples (Simplified)

10.1–10.3 show compressed “caption flavor → prompt” examples; 10.4 walks through a full caption → prompt transform.

### 10.1 Mechagodzilla Beam Attack (Destruction + SFX)

**Caption flavor:** Mechagodzilla flies as a WIREWORK prop, turns its head, and fires rainbow and red/purple MECHAGODZILLAS SPACE BEAMS from its eyes, with ANIMATED OPTICAL OVERLAYS.

**Prompt:**

> Japanese Kaiju Film — Mechagodzilla, the robotic kaiju covered in riveted SILVER‑PAINTED MOLDED PLASTIC/FIBERGLASS PLATES, flies through a hazy blue sky as a WIREWORK prop. In a low‑angle shot, he tilts his head toward the camera and fires MECHAGODZILLAS SPACE BEAMS from his glowing eyes, rendered as rainbow and red‑purple ANIMATED OPTICAL OVERLAYS. The beams streak forward as SPARKLER‑LIKE PYROTECHNICS erupt below in a MINIATURE CITYSCAPE.

---

### 10.2 Godzilla vs. Anguirus Throw (Kaiju‑First + Wrestling)

**Caption flavor:** Godzilla lifts Anguirus and performs a wrestling‑style throw in a rocky miniature landscape, with dust and PLASTER.

**Prompt:**

> Japanese Kaiju Film — In a static wide shot of a rocky MINIATURE landscape, the reptilian, charcoal‑gray kaiju, Godzilla, hoists the spiky‑shelled quadruped Anguirus over his shoulder. Using an exaggerated SUITMATION wrestling motion, Godzilla pivots and hurls Anguirus into the hillside. The LATEX RUBBER SUIT of Anguirus crashes into the miniature terrain, sending up a billowing cloud of PLASTER dust that briefly obscures both kaiju.

---

### 10.3 Minilla’s Smoke Rings (Character + Parent‑Child Beat)

**Caption flavor:** Minilla tries to fire atomic smoke rings while Godzilla watches.

**Prompt:**

> Japanese Kaiju Film — In a low‑angle medium shot, the small, chubby gray‑green kaiju with a pug‑like face, Minilla, stands in front of Godzilla against a rocky MINIATURE hillside under a dark, cloudy sky. Minilla puffs his cheeks and exhales, firing one of MINILLAS ATOMIC SMOKE RINGS as a swirling white ANIMATED OPTICAL OVERLAY that drifts across the frame. Behind him, the towering SUITMATION figure of Godzilla stands watchfully, his PEBBLED‑TEXTURE LATEX RUBBER SUIT silhouetted against the MATTE PAINTING sky.

---

### 10.4 Full Caption Walkthrough: Minilla’s Smoke Rings

**Full caption (training example):**

> Japanese Kaiju Film — In a low-angle shot on a miniature island set, the small, chubby kaiju with a pug-like face, Minilla, stands in the foreground. He exhales his signature MINILLA'S ATOMIC SMOKE RINGS, which appears as a continuous stream of glowing blue smoke rendered as an ANIMATED OPTICAL OVERLAY. Behind him and towering over him, the larger, charcoal-gray reptilian kaiju, Godzilla, watches intently. The two SUITMATION characters are positioned on a rocky cliffside adorned with miniature jungle trees under a bright blue sky.

**Simplified prompt (for prompting):**

> Japanese Kaiju Film — In a low‑angle medium shot on a MINIATURE island cliff, the small, chubby gray‑green kaiju with a pug‑like face, Minilla, stands in the foreground, puffing his cheeks as he exhales one of MINILLAS ATOMIC SMOKE RINGS. The swirling white ANIMATED OPTICAL OVERLAY smoke ring drifts across the frame while the towering SUITMATION figure of Godzilla watches behind him, framed against a bright MATTE PAINTING sky and scattered MINIATURE jungle trees.

**What changed:**

- ❌ **Dropped extra environmental cataloging**  
  - Removed the fully enumerated “rocky cliffside adorned with miniature jungle trees…” style phrasing.  
  - Kept a compressed version: “MINIATURE island cliff” + “scattered MINIATURE jungle trees.”
- ❌ **Reduced temporal micro‑beats**  
  - Caption implies multiple micro‑moments (“stands… exhales… continuous stream… watches intently”).  
  - Prompt focuses on one clear beat: Minilla puffing his cheeks and firing a single ring that drifts across the frame.
- ✅ **Kept core style anchors**  
  - SUITMATION, MINIATURE, ANIMATED OPTICAL OVERLAY, ATOMIC SMOKE RINGS, MATTE PAINTING all preserved.  
  - Kept Godzilla present but compressed to “towering SUITMATION figure of Godzilla” instead of a full re‑intro.
- ✅ **Aligned with T5 rules**  
  - Converted `MINILLA'S ATOMIC SMOKE RINGS` → `MINILLAS ATOMIC SMOKE RINGS` in the prompt to avoid an apostrophe in the ALL‑CAPS keyphrase.  
  - Avoided quotation marks inside the prompt body.
- ✅ **Clarified narrative priority**  
  - Kept the shot Kaiju‑First, centered on Minilla’s action under Godzilla’s supervision, mirroring Pattern A.

---

## 11. Prompting Off‑Road (Beyond Canon Clips)

To use this LoRA for **original kaiju or other franchises** while keeping the tokusatsu flavor, follow a simple rule: **keep the grammar, change the nouns** in kaiju form.

### 11.1 Keep the Grammar and Craft

- Shot‑first or kaiju‑first structure.  
- Craft terms: SUITMATION, LATEX RUBBER SUIT, MINIATURE CITYSCAPE, SPARKLER‑LIKE PYROTECHNICS, MATTE PAINTING, ANIMATED OPTICAL OVERLAY, etc.

### 11.2 Invent New Kaiju with Familiar Grammar

Structure:

```text
[descriptor phrase], [New Name]
```

Examples:

- “the towering beetle‑like kaiju with jagged SILVER MANDIBLES, Scarablos”  
- “the skeletal dragon kaiju wreathed in BLUE CGI PARTICLE DEBRIS, Arclidon”  
- “the hulking, stone‑skinned golem kaiju with glowing magma cracks, Basalgron”

Drop them into miniature setups:

> “A static low‑angle shot of Scarablos rampaging through a dense MINIATURE CITYSCAPE, SPARKLER‑LIKE PYROTECHNICS bursting along the streets as PLASTER AND BALSA WOOD MINIATURES crumble.”

### 11.3 Other Franchises / Non‑Godzilla Monsters

“Kaiju‑ize” other monsters by:

- Re‑describing them in kaiju grammar (“the giant turtle kaiju with a spiked shell and jet thrusters…”).  
- Embedding them in miniature sets (“stomps through a MINIATURE harbor”, “looms over a MINIATURE TOKYO STREET”).

As long as you respect the trigger, naming grammar, and SUITMATION + MINIATURE + SFX vocabulary, the LoRA should generalize the style.

---

## 12. Shōwa vs Heisei Style Cues

The current caption corpus for this LoRA clearly covers at least **two eras**:

- **Shōwa (1954–1975)** — classic rubber‑suit Godzilla, more playful and anthropomorphic.  
- **Heisei (1984–1995)** — heavier, more menacing Godzilla with ANIMATRONIC HEADS and darker tone.

Use era cues when you want to bias toward one feel or the other, but keep in mind that the LoRA is still grounded in these two families of clips.

### 12.1 Quick Era Cheat Sheet (Dataset‑backed)

- **Shōwa:** SUITMATION with PEBBLED‑TEXTURE LATEX RUBBER SUITS, obvious MINIATURE CITYSCAPE, SPARKLER‑LIKE PYROTECHNICS, simple ANIMATED OPTICAL OVERLAY beams. Tone: campy, wrestling‑like, expressive and sometimes goofy.  
- **Heisei:** Heavier LATEX RUBBER SUITS, ANIMATRONIC HEADS, darker MINIATURE CITYSCAPES with lots of smoke and fire, COMPOSITED DIGITAL BEAMS. Tone: serious, weighty, ominous.

**Visual comparison:**

| Era                 | Suits / SFX                                                     | Beams                       | Tone                         | Example caption cue                                                            |
|---------------------|-----------------------------------------------------------------|-----------------------------|------------------------------|-------------------------------------------------------------------------------|
| **Shōwa**           | Shōwa‑era SUITMATION, PEBBLED‑TEXTURE LATEX RUBBER SUIT, bright MINIATURE sets | ANIMATED OPTICAL OVERLAY    | Playful, wrestling‑style     | “the Shōwa‑era SUITMATION puppet performs a triumphant dance amid MINIATURE ruins” |
| **Heisei**          | Heisei‑era SUITMATION, heavy LATEX RUBBER SUIT, ANIMATRONIC HEADS, thick smoke | COMPOSITED DIGITAL BEAM / RED GODZILLAS ATOMIC BREATH | Dark, epic, imposing          | “the massive, charcoal‑black kaiju, Godzilla, stands in a smoky MINIATURE CITYSCAPE, rendered in Heisei‑era SUITMATION” |

### 12.2 Choosing Between Shōwa and Heisei

Ask:

- **Tone:** light, celebratory, anthropomorphic → Shōwa; grim, looming, catastrophic → Heisei.  
- **Camera & motion:** bouncy dances, exaggerated wrestling, visible suit constraints → Shōwa; slow, ponderous head turns and heavy steps through smoke → Heisei.  
- **Effects:** bright day or simple night with clear MINIATURE sets and OPTICAL OVERLAYS → Shōwa; dense smoke, night city, glowing fires, COMPOSITED DIGITAL BEAMS / RED GODZILLAS ATOMIC BREATH → Heisei.

You don’t always need to name the era; you can also imply it via these cues.

---

## 13. Managing Conceptual Distance (OOD Prompts)

This is the Kaiju analogue of TMNT’s OOD / attention‑budget guidance.

**Core Principle:** Each prompt should revolve around **one clear core idea**. The more “Out‑of‑Distribution” the idea is from classic tokusatsu (suits + miniatures + beams), the more you must simplify everything else.

### 13.1 What Counts as OOD?

Think in distance from:

- Japanese tokusatsu kaiju movies  
- SUITMATION + MINIATURE CITYSCAPE + practical PYROTECHNICS

Low distance (safe):

- New but Godzilla‑like kaiju (dinosaurs, dragons, insects, gorillas).  
- New cities/countries but same miniature destruction style.

**Example (low distance):**

> Japanese Kaiju Film — In a static low‑angle shot, the towering beetle‑like kaiju with jagged SILVER MANDIBLES, Scarablos, rampages through a dense MINIATURE TOKYO STREET at night. SPARKLER‑LIKE PYROTECHNICS burst from shopfronts as PLASTER AND BALSA WOOD MINIATURES crumble around his clawed feet in the hazy streetlight.

Medium distance (manageable):

- Very exotic kaiju (floating crystal shards, abstract energy beasts).  
- Outer‑space or deep‑ocean settings, still shot like miniature sets.  
- New beam types (gravity distortions, time‑freeze beams).

**Example (medium distance):**

> Japanese Kaiju Film — On a rocky MINIATURE hillside, a serpentine energy kaiju formed from swirling BLUE CGI PARTICLE DEBRIS coils around a cliffside shrine. In a wide shot, it fires a gravity‑distortion beam depicted as a COMPOSITED DIGITAL BEAM that warps a row of MINIATURE BUILDINGS below, sending PLASTER debris into the air.

High distance (challenging):

- No giant creature at all.  
- Purely 2D/anime looks.  
- Pure high‑fantasy castles and armies with no miniature/modern cues.

**Example (high distance, challenging):**

> Japanese Kaiju Film — A PHOTOREALISTIC CGI RENDERING of a floating crystal fortress hangs over a medieval walled town at dusk. No kaiju are visible; only a swirling portal of PURPLE CGI PARTICLE DEBRIS in the sky sheds light on the MINIATURE‑like rooftops below.

As distance increases, lean harder on classic anchors to pull it back into distribution.

### 13.2 One Novel Thing at a Time

When going off‑road:

- If the subject is novel (energy serpent, floating island), keep setting, camera, and SFX grammar classic.  
- If the effect is novel (reality warp, time stop), keep kaiju design and miniature city familiar.  
- If the setting is novel (floating island, alien temple), keep Godzilla/kaiju style and destruction classic.

If subject **and** setting **and** effect are all new, either:

- simplify down to one novel axis, or  
- split across multiple prompts.

---

## 14. Quick Reference Checklist

Before finalizing a prompt, check:

- [ ] Starts with `Japanese Kaiju Film —`  
- [ ] Uses a clear Narrative Priority pattern (Kaiju‑First, Destruction‑First, or Scale‑First)  
- [ ] Puts the **reason the shot exists** first (pose, destruction, or scale moment)  
- [ ] Introduces each kaiju as `[descriptor phrase], [Name]` on first mention  
- [ ] Chooses one primary kaiju in multi‑kaiju scenes  
- [ ] Includes 2–3 core craft anchors (SUITMATION, MINIATURE CITYSCAPE, PYROTECHNICS…)  
- [ ] Adds 1–3 supporting terms (MATTE PAINTING, WIREWORK, CGI PARTICLE DEBRIS, etc.)  
- [ ] Uses one primary attack/effect (two max) from the Key Visuals list when needed  
- [ ] Specifies time of day + main light source + basic smoke/fog  
- [ ] Uses present tense, third person  
- [ ] Describes one main beat or micro‑sequence (1–2 “First / Then / Finally” steps)  
- [ ] Has at most one explicit camera move (or static)  
- [ ] Word count is in a sensible range (50–80 quick, 90–150 standard, 160–200 complex)

For OOD prompts:

- [ ] Identified what’s novel (subject / effect / setting)  
- [ ] Simplified everything else back to familiar kaiju grammar

---

## 15. Appendix – Style Terms by Category

A compact catalog of useful terms, adapted from the captioning sysprompt and actual captions.

### 15.1 Essential (use 2–3 per prompt)

- SUITMATION  
- LATEX RUBBER SUIT / PEBBLED‑TEXTURE LATEX RUBBER SUIT  
- MINIATURE CITYSCAPE / MINIATURE BUILDING PROP  
- PLASTER AND BALSA WOOD MINIATURES  
- ANIMATED OPTICAL OVERLAY / OPTICAL BEAM EFFECT  
- COMPOSITED DIGITAL BEAM

### 15.2 Common (use 1–3 selectively)

- SPARKLER‑LIKE PYROTECHNICS  
- LARGE‑SCALE GASOLINE FIREBALLS  
- WIREWORK / WIRE‑ASSISTED FLIGHT  
- MATTE PAINTING  
- CGI PARTICLE DEBRIS  
- ANIMATRONIC HEADS  
- PHOTOREALISTIC CGI RENDERING  
- DIGITAL PARTICLE EFFECTS

### 15.3 Situational (use when clearly visible)

- POWDERY SCALES (MOTHRA'S POISON SCALES)  
- LIQUID SILK (Mothra Larva webs)  
- MINILLAS ATOMIC SMOKE RINGS  
- TITANOSAURUSS HURRICANE‑FORCE WIND  
- MASER CANNON, SUPER X  
- MINIATURE POWER LINE, MINIATURE AIRPORT, MINIATURE HARBOR

Use these when they are obviously present in the imagined shot; avoid sprinkling them randomly.

---

**End of Guidelines**
