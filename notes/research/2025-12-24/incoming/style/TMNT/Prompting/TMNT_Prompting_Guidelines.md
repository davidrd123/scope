# TMNT Mutant Mayhem / WAN 2.2 Prompting Guidelines

**Status:** v1.0 — Caption-aligned structure
**Owner:** davidrd
**Model:** WAN 2.2 (T5 text encoder)

Shared prompt patterns reference: `WorkingSpace/davidrd/OutsideStruct/PromptWriting/Prompt_Writing_Patterns_Shared.md`

---

## Overview

This guide teaches you to write prompts in the **style and grammar** of the TMNT training captions while **strategically simplifying** for effective prompting. The goal is not exhaustive description but focused visual direction.

**Key Principle:** Use the caption's technical vocabulary and sentence structure, but be selective about content. Lead with your beat, include essential style anchors, describe one primary action clearly.

---

## 1. Trigger Phrase (Required)

**Format:**
```
Graffiti Sketchbook Animation —
```

- **Always** start with this exact trigger
- Use em dash (—), not hyphen (-)
- Follow with a space before your content

**Why it matters:** This trigger activates the visual style. Without it, you get generic CG animation.

---

## 2. The Principle of Narrative Priority

**Core Principle:** The most important element—the reason the shot exists—should be described first to dedicate the most attention and detail to it.

The model pays most attention to what comes first in your prompt. Structure your opening based on what matters most for the shot. Choose the pattern that matches your emphasis:

### Pattern A: Setting-First (Establishing Shot)
Use when the primary goal is to establish a location, mood, or atmosphere. The environment is the main character of the shot.

**Structure:**
```
[Shot type] [setting & atmosphere] [style anchors]. [Subject & action (secondary)].
```

**Example:**
> Graffiti Sketchbook Animation — A low-angle wide shot of a rain-slicked alley at midnight, graffiti-covered brick walls receding into darkness. Harsh amber streetlight from above casts deep shadows with heavy CROSS-HATCHING. The scene is rendered with ROUGH PAINT-SWATCH TEXTURES and SCRIBBLED LINEWORK. A lone figure in a hoodie walks slowly in the background, silhouetted.

**Use when:** Location/mood is more important than character action.

### Pattern B: Character-First (Interaction Shot)
Use when the primary goal is to capture a character's performance, emotion, or specific interaction. The character element is the most important part of the shot.

**Structure:**
```
[Shot type] of [character & descriptor] [primary action/emotion] in [setting (secondary)]. [Style & lighting].
```

**Example:**
> Graffiti Sketchbook Animation — A medium close-up of a street magician in a worn vest as he performs a coin vanish, his exaggerated grin shifting to surprise. He stands on a subway platform, fluorescent tubes overhead. His features are rendered with SCRIBBLED GRAPHITE-LIKE OUTLINES and CROSS-HATCHING. Cool blue AMBIENT HUE, ROUGH PAINT-SWATCH TEXTURES on clothing.

**Use when:** Character performance/emotion is the core beat.

### Pattern C: Object-First (Insert Shot)
Use when the primary goal is to focus on a specific object or detail that is critical to the story.

**Structure:**
```
[Shot type] on [focal object/detail] [state/action]. [Supporting context & style (secondary)].
```

**Example:**
> Graffiti Sketchbook Animation — An extreme close-up on a jury-rigged EMF detector wrapped in duct tape, its green LCD screen spiking wildly. The device is held by gloved hands in a dark room. FLUORESCENT GLOW from the screen casts harsh light, rendered with SCRIBBLED LINEWORK and CROSS-HATCHING in the shadows. Deep blue AMBIENT HUE beyond the glow.

**Use when:** A specific object/detail is the narrative focus.

---

**Choosing Your Pattern:**
- Ask: "What is the reason this shot exists?"
- Put that element first
- Make everything else support it

---

## 3. Structural Flow

Every prompt should follow this rhythm:

```
ESTABLISH → STYLE → ACTION → LIGHTING → CONCLUDE
```

**ESTABLISH:** Open with shot type or subject + location
**STYLE:** Integrate 4-6 style anchors naturally throughout
**ACTION:** Describe the primary beat/sequence
**LIGHTING:** One key light + one ambient color
**CONCLUDE:** End state or camera move

This isn't a rigid order—style and lighting weave through action—but hits all these elements.

---

## 4. Length Guidelines

Base length on scene complexity:

| Type | Word Count | Use Case |
|------|-----------|----------|
| **Quick Beat** | 50-80 words | Simple action, expression change, establishing shot |
| **Standard** | 90-150 words | Character intro + action, moderate scene complexity |
| **Complex** | 160-200 words | Multi-character interaction, sequential actions, rich environment |

**Note:** These are shorter than the training captions (which run 75-250+ words) because prompts don't need exhaustive description—just enough to drive the visual output.

---

## 5. Character Identification

Use the caption's naming formula:

### First Mention
```
[Name], [distinguishing visual feature]
```

**Examples from captions:**
- "Leonardo, the turtle in the BLUE MASK"
- "Raphael, the tallest turtle wearing a RED do-rag"
- "Donatello, the slender turtle in the PURPLE MASK and glasses"
- "Michelangelo, the shortest turtle in the ORANGE MASK"

### Subsequent Mentions
Use name alone or repeat the descriptor:
- "Leonardo raises his hand"
- "the turtle in the BLUE MASK glances back"

### Mask Colors (CAPITALIZE)
- BLUE MASK → Leonardo
- RED do-rag → Raphael
- PURPLE MASK → Donatello
- ORANGE MASK → Michelangelo

**For prompting off-road (non-TMNT characters):** Adapt the formula. Instead of "Leonardo, the turtle in the BLUE MASK," try "Marcus, a lanky teenager in a red hoodie" or "the fox mutant with torn ears."

---

## 6. Style Anchor Integration

Don't treat style terms as a checklist to spam. Integrate them naturally into your descriptions.

### Core Anchors (Use 2-3 per prompt)
These are the foundation of the visual style:

- **SCRIBBLED LINEWORK** / **SCRIBBLED GRAPHITE-LIKE OUTLINES**
- **ROUGH PAINT-SWATCH TEXTURES**
- **CROSS-HATCHING** (for shadows/depth)

**Integration examples:**
- "His features are rendered with SCRIBBLED GRAPHITE-LIKE OUTLINES and visible CROSS-HATCHING for shadows."
- "The scene is rendered with ROUGH PAINT-SWATCH TEXTURES and a distinct sketchbook aesthetic."

### Supporting Anchors (Use 1-3 selectively)
Add these based on scene needs:

- **SHALLOW DEPTH OF FIELD** (for focus/miniature effect)
- **STUTTERING FRAME RATE** (snappy motion)
- **HIGH-CONTRAST** (lighting quality)
- **ASYMMETRICAL CHARACTER DESIGN** (imperfect, handcrafted look)

### Specialized Terms (Scene-specific)
- **DOODLED EFFECTS** (energy, impacts, emphasis)
- **FLUORESCENT GLOW** (mutagen, screens, neon)
- **SIMULATED HANDHELD CAMERA** (documentary realism)
- **AMBIENT HUE** (color atmosphere)

**Total:** Aim for 4-6 CAPITALIZED terms integrated across the prompt.

---

## 7. Action Sequencing

### Present Tense, Third Person
All captions use present tense. Maintain this.

**RIGHT:** "He raises his hand, glances back, then smirks."
**WRONG:** "He raised his hand and glanced back."

### Sequential Actions
Use these transition words from the captions:

- **"First,"** / **"Initially,"** → starting state
- **"Then,"** / **"As he moves,"** → progression
- **"Finally,"** / **"Throughout,"** → conclusion/duration

**Example from caption:**
> "First, he glares forward with a tense, wide-eyed look. Then, his mouth twitches and opens slightly to bare his teeth in a brief, silent snarl. Finally, the snarl recedes, and his expression settles into a smug, knowing smirk."

### Simultaneous Actions
Use commas for things happening at once:

**Example:**
> "He raises his hand, his brow furrows, and his mouth opens."

### Beat-First Ordering
Lead with the **reason the shot exists**, then add context.

**GOOD:** "A mutant rat lunges toward the camera, claws extended, in a dark sewer tunnel."
**WEAK:** "In a dark sewer tunnel, there is a mutant rat that lunges toward the camera with claws extended."

The first version puts the action (the beat) up front.

### Emotional Tone & Expression
The TMNT captions have strong personality through specific, expressive emotional descriptors. When describing character expressions or reactions, use precise emotional language that captures both the physical manifestation AND the feeling.

**Describe the physical components of expressions:**
- "His brow furrows and his mouth opens wide"
- "Her eyes widen, pupils dilating"
- "A smile spreads across his face, revealing teeth"

**But also name the emotional quality:**
- "a look of weary resignation"
- "a wide, manic grin"
- "a moment of quiet satisfaction"
- "exasperated disbelief"
- "tense, coiled anticipation"
- "smug confidence"
- "cautious curiosity"

**Examples from captions:**
- "his expression settles into a smug, knowing smirk"
- "a tense, wide-eyed look"
- "wide smile and open arms"
- "tired but determined expressions"

**Apply this to prompts:**

**WEAK (Generic):**
> "The werewolf looks at the camera and opens his mouth."

**STRONG (Emotionally Specific):**
> "The werewolf's expression shifts to one of feral aggression, his lips pulling back to reveal elongated canines in a menacing snarl."

**WEAK (Vague):**
> "The ghost hunter reacts to the EMF spike."

**STRONG (Emotionally Specific):**
> "Her eyes widen in a mix of fear and exhilaration as the EMF meter spikes, mouth opening in breathless anticipation."

Strong emotional descriptors add personality and make characters feel lived-in, not generic. This is especially important when creating new characters for off-road prompting—emotion grounds them in the TMNT world's expressive, character-driven storytelling.

---

## 8. Lighting & Camera Language

### Lighting Formula
Captions consistently describe:
1. **One key light source** (direction, color, quality)
2. **One ambient color** (overall atmosphere)

**Pattern from captions:**
```
"The scene is bathed in a [quality], [color] AMBIENT HUE"
"A [quality] light from [direction/source] casts..."
"[Subject] is illuminated by a [quality] [color] glow from..."
```

**Examples:**
- "The scene is bathed in a dark, blue AMBIENT HUE, punctuated only by the artificial glow from high-visibility clothing."
- "A warm, HIGH-CONTRAST light source from above casts deep shadows."
- "The phone's screen casts a blue glow onto his smiling face."

**Lighting vocabulary:**
- Quality: HIGH-CONTRAST, SOFT DIFFUSED, HARSH DIRECT, FLUORESCENT
- Color: amber, blue, green, red, purple, warm, cool
- Effect: casts shadows, creates halos, bathes the scene, illuminates

### Camera Work
**Shot types:** low-angle, high-angle, medium shot, close-up, wide shot, tight shot, over-the-shoulder

**Camera moves (one max):**
- "The camera slowly pushes in/pulls back/pans left/pans right/tilts up/tilts down"
- "A slow, deliberate push-in from the SIMULATED HANDHELD CAMERA"
- "Throughout the shot, a faint SIMULATED HANDHELD CAMERA SHAKE"

**Constraint:** Describe **one camera move maximum** per prompt. Static is also fine.

---

## 9. What to Simplify (Strategic Selection)

The training captions describe **everything** because that's their job. Prompts should be **selective**.

### ✅ Always Include
1. **Trigger phrase** ("Graffiti Sketchbook Animation —")
2. **Primary subject** (who/what)
3. **Primary action/beat** (the point of the shot)
4. **2-3 core style anchors** (SCRIBBLED LINEWORK, ROUGH TEXTURES, CROSS-HATCHING)
5. **Key lighting** (one source + ambient color)

### ✅ Include When Relevant
6. Shot type (if composition matters)
7. 1-3 supporting style terms (SHALLOW DEPTH OF FIELD, STUTTERING FRAME RATE)
8. Camera move (if motion is important)
9. Character descriptors (if introducing someone)
10. Setting details (if environment drives the mood)

### ❌ Skip or Minimize
- **Exhaustive environmental cataloging** ("...and shadowy buildings line the narrow road, creating a tunnel-like perspective, while parked cars...")
- **Micro-details** that don't drive output ("the raised lettering on the cover which reads NYC SEWER")
- **Multiple sequential beats** (captions often have 3-4; stick to 1-2 for prompts)
- **Redundant style terms** (don't say SCRIBBLED LINEWORK and SCRIBBLED GRAPHITE-LIKE OUTLINES and SCRIBBLED AESTHETIC—pick one)

### Density Priority
**Focus attention budget on:**
1. **Style** (what makes it look like TMNT)
2. **Action** (what's happening)
3. **Subject** (who/what is doing it)
4. **Lighting** (mood/atmosphere)

**Reduce attention on:**
- Background elements unless they're the subject
- Fine-grained choreography unless that's the beat
- Temporal precision ("for 2 seconds," "briefly," etc.)

---

## 10. Real Examples with Simplification

### Example 1: Establishing Shot

**Full Caption (140 words):**
> Graffiti Sketchbook Animation — A low-angle shot establishes a gritty, nocturnal New York City street, rendered with a painterly aesthetic and ROUGH PAINT-SWATCH TEXTURES. The wet pavement reflects the warm, high-contrast glow of overhead streetlights, while parked cars and shadowy buildings line the narrow road, creating a tunnel-like perspective. In the center of the frame, a tiny brown street rat stands frozen on a crosswalk, its small form silhouetted. Far down the street, a large van drives directly toward it, its two bright headlights cutting through the darkness. As the vehicle advances, its headlights intensify, casting a growing glare and a prominent lens flare that threatens to overwhelm the rat's silhouette. The entire scene is defined by a SHALLOW DEPTH OF FIELD, keeping the foreground pavement sharp while the distant elements remain soft and impressionistic.

**Simplified Prompt (95 words):**
> Graffiti Sketchbook Animation — A low-angle shot of a gritty nocturnal city street, wet pavement reflecting warm streetlight glow. A tiny brown rat stands frozen on a crosswalk in the center, silhouetted. Far down the street, a large van drives toward it, bright headlights cutting through darkness and casting a growing glare. The scene is rendered with ROUGH PAINT-SWATCH TEXTURES, SCRIBBLED LINEWORK, and SHALLOW DEPTH OF FIELD that keeps the rat sharp while distant elements blur softly. HIGH-CONTRAST lighting from overhead streetlights, cool blue AMBIENT HUE in shadows.

**What changed:**
- ❌ Removed: "parked cars and shadowy buildings line the narrow road, creating a tunnel-like perspective" (environmental detail)
- ❌ Removed: "prominent lens flare that threatens to overwhelm" (micro-detail)
- ❌ Removed: "painterly aesthetic" (redundant with style terms)
- ✅ Kept: Shot type, subject (rat), action (van approaching), lighting, core style anchors
- ✅ Added: Cool blue AMBIENT HUE (inferred but makes lighting more specific)

---

### Example 2: Character Introduction + Action

**Full Caption (140 words):**
> Graffiti Sketchbook Animation — On a dimly lit city rooftop at night, four mutant turtles gather under a sickly green light cast from a graffiti-covered brick wall. Initially, three of them are grouped together: Raphael, the tallest turtle wearing a RED do-rag, holds a square pizza box with the words My Pizza on it. Next to him stands Donatello, the slender turtle in the PURPLE MASK and glasses, who has his arm around Michelangelo, the shortest turtle in the ORANGE MASK. Across from them, Leonardo, the turtle in the BLUE MASK, approaches with a wide smile and open arms. As he raises a hand, Donatello suddenly pulls out a smartphone, its screen erupting in a bright blue light that illuminates their faces. He then raises the phone high into the air to take a group selfie. The other three turtles immediately crowd in and strike poses, with Leonardo and Michelangelo both throwing up peace signs as they huddle together for the picture. The scene is rendered with SCRIBBLED GRAPHITE-LIKE OUTLINES and a stuttering frame rate, while the background city lights are blurred with a SHALLOW DEPTH OF FIELD.

**Simplified Prompt (110 words):**
> Graffiti Sketchbook Animation — On a dimly lit rooftop at night, four mutant turtles gather for a group selfie. Donatello, the slender turtle in the PURPLE MASK and glasses, raises a smartphone high. Its blue screen glow illuminates the faces of Raphael in the RED do-rag, Michelangelo in the ORANGE MASK, and Leonardo in the BLUE MASK as they crowd in and strike poses, throwing up peace signs. The scene is rendered with SCRIBBLED GRAPHITE-LIKE OUTLINES, ROUGH PAINT-SWATCH TEXTURES, and CROSS-HATCHING. SHALLOW DEPTH OF FIELD blurs the city lights behind them. Sickly green light from the graffiti-covered wall casts shadows.

**What changed:**
- ❌ Removed: Sequential build-up ("Initially, three of them... Next to him... Across from them...") → collapsed to the final state
- ❌ Removed: "the words My Pizza on it" (prop detail)
- ❌ Removed: "with a wide smile and open arms" → focus on the core action (selfie)
- ❌ Removed: "stuttering frame rate" → limited style term count
- ✅ Kept: All four character identifiers, primary action (selfie), smartphone glow, lighting, key style anchors
- ✅ Simplified: Went straight to the beat (group selfie) instead of step-by-step blocking

---

### Example 3: Micro-Expression Focus

**Full Caption (110 words):**
> Graffiti Sketchbook Animation — In a medium shot, a man with a dark goatee is framed by the open window of a vehicle's cab, the scene steeped in a deep blue AMBIENT HUE. He wears a blue collared shirt and a tall, pointed paper-style hat with red stitching. His features are rendered with the distinct sketchbook aesthetic, featuring rough SCRIBBLED GRAPHITE-LIKE OUTLINES and visible CROSS-HATCHING for shadows on his face and arm. His arm rests on the bottom of the window frame. The clip focuses on the subtle but menacing shift in his expression. First, he glares forward with a tense, wide-eyed look. Then, his mouth twitches and opens slightly to bare his teeth in a brief, silent snarl. Finally, the snarl recedes, and his expression settles into a smug, knowing smirk as he continues to stare ahead.

**Simplified Prompt (85 words):**
> Graffiti Sketchbook Animation — A medium shot of a man with a dark goatee framed by a vehicle window, deep blue AMBIENT HUE. He wears a pointed paper hat with red stitching. His expression shifts from a tense glare to a brief snarl, then settles into a smug smirk. His features are rendered with SCRIBBLED GRAPHITE-LIKE OUTLINES and visible CROSS-HATCHING for shadows. His arm rests on the window frame. The scene emphasizes the menacing micro-expressions in the sketchbook style.

**What changed:**
- ❌ Removed: "blue collared shirt" (clothing detail that doesn't add to the beat)
- ❌ Removed: "distinct sketchbook aesthetic" (redundant phrasing)
- ❌ Reduced: "First... Then... Finally..." → compressed to "shifts from X to Y to Z"
- ✅ Kept: Shot type, subject, expression sequence, key style terms, lighting, posture
- ✅ Maintained: The core beat (menacing expression shift)

---

## 11. Prompting Off-Road (Beyond TMNT)

You want to use the **visual style** for other subjects (non-turtles, non-mutants). Here's how:

### Keep the Grammar, Change the Nouns
The caption structure works for anything:

**TMNT Style:**
> "Leonardo, the turtle in the BLUE MASK, leaps across a rooftop..."

**Off-Road Style:**
> "A cyberpunk courier in a neon blue jacket leaps across a rooftop..."

**TMNT Style:**
> "Raphael grips his sai, muscles tensing, as he faces the camera."

**Off-Road Style:**
> "A scarred boxer grips the ropes, muscles tensing, as he faces the crowd."

### Maintain Style Anchors
The visual language stays the same:
- SCRIBBLED LINEWORK
- ROUGH PAINT-SWATCH TEXTURES
- CROSS-HATCHING
- SHALLOW DEPTH OF FIELD
- HIGH-CONTRAST lighting

These terms encode the "graffiti sketchbook" look. Use them for any subject.

### Character Building
Apply the identification formula to original characters:

**Structure:** [Name/role], [unique visual trait]

**Examples:**
- "Kai, a lanky android with exposed wire tendons"
- "The graffiti artist in the tattered yellow hoodie"
- "A mechanical owl with asymmetrical wings and glowing red eyes"

### World Transposition
Want TMNT style but different setting?

**Keep:**
- Trigger phrase
- Style anchors (SCRIBBLED, ROUGH TEXTURES, CROSS-HATCHING)
- Lighting approach (key + ambient)
- Caption grammar (shot-first or subject-first)

**Change:**
- Environment ("cyberpunk alley" not "NYC sewer")
- Characters (your OCs, not turtles)
- Props and details

**Example:**
> Graffiti Sketchbook Animation — A low-angle shot of a neon-lit Tokyo street at night, wet pavement reflecting pink and blue signs. A lone figure in a long coat stands at the center, face obscured by shadow. The scene is rendered with ROUGH PAINT-SWATCH TEXTURES, heavy CROSS-HATCHING in the shadows, and SCRIBBLED LINEWORK defining building edges. SHALLOW DEPTH OF FIELD keeps the figure sharp while distant neon blurs. HIGH-CONTRAST pink key light from signage, deep blue AMBIENT HUE.

Notice: Same structure as TMNT captions, but entirely different world.

---

## 12. Managing Conceptual Distance for Off-Road Prompting

**Core Principle:** Every prompt should be built around **one singular, clear core idea**. The more "conceptually distant" or Out-of-Distribution (OOD) that idea is from the TMNT training data, the more of the model's "Attention Budget" must be dedicated to it, and the simpler everything else must be.

### Understanding Attention Budget

The model is not a machine executing commands—it interprets a creative brief. Asking it to simultaneously innovate on multiple difficult fronts will overload it, causing:
- Cross-talk between ideas
- Stylistic dilution
- Failure to render the most important elements

**By focusing each prompt on ONE thing that's new/difficult, you give the model the best chance to execute it brilliantly.**

### What Counts as OOD?

For TMNT, these are increasingly distant from training data:

**Low Distance (Safe):**
- Different mutant animals (raccoon, bat, frog)
- Different urban locations (warehouse, bridge, rooftop)
- Similar human characters (cops, construction workers, teenagers)

**Medium Distance (Manageable):**
- Non-mutant creatures (werewolves, cryptids)
- Supernatural elements (ghosts, magic effects)
- Different time of day (dawn, daytime)
- Different cities (London, Tokyo, Mexico City)

**High Distance (Challenging):**
- Completely different character types (robots, aliens, fantasy creatures)
- Non-urban settings (forests, deserts, space)
- Abstract/surreal concepts
- Period pieces (medieval, Victorian, 1920s)

### The OOD Prompting Strategy

**When introducing something OOD, simplify everything else.**

#### Principle A: Novel Subject → Familiar Everything Else

**INEFFECTIVE (Too Many OOD Concepts):**
> Graffiti Sketchbook Animation — In a zero-gravity spaceship interior, a futuristic android with holographic wings performs a complex martial arts routine while the camera executes a rapid orbital arc around it.

**Problems:** OOD subject (android) + OOD setting (spaceship) + OOD action (martial arts in zero-g) + OOD camera (orbital). The model will fail at all of them.

**EFFECTIVE (Singular Focus on OOD Subject):**
> Graffiti Sketchbook Animation — A low-angle medium shot of a futuristic android with asymmetrical metal plating and exposed wire joints, standing motionless in a gritty urban alley. The android's single glowing red eye is a large circular lens. The alley has graffiti-covered brick walls and scattered trash. Rendered with SCRIBBLED LINEWORK, ROUGH PAINT-SWATCH TEXTURES, and CROSS-HATCHING in shadows. Harsh amber streetlight from above, cool blue AMBIENT HUE. Static camera.

**Why it works:** Dedicates attention budget to translating "android" into TMNT style. Familiar setting (alley), simple action (standing), static camera.

#### Principle B: Novel Action/Effect → Familiar Everything Else

**INEFFECTIVE (Divided Focus):**
> Graffiti Sketchbook Animation — A detailed miniature cityscape melts into liquid while a complex origami bird flies through and the camera dollies out.

**Problems:** OOD effect (melting) + OOD object (origami bird) + camera move. The melting effect will be weak.

**EFFECTIVE (Singular Focus on OOD Effect):**
> Graffiti Sketchbook Animation — A high-angle wide shot of a handcrafted cityscape diorama, buildings rendered with ROUGH PAINT-SWATCH TEXTURES. First, the city is solid and still. Then, the sharp edges soften and droop, buildings melting downward like wet paint. The structures collapse into swirling pools of color with SCRIBBLED LINEWORK at the edges. Static camera observing the transformation. CROSS-HATCHING in the darker pigments.

**Why it works:** The ONLY thing happening is the melt. No other characters, no camera moves, simple subject. Every token dedicated to that one effect.

#### Principle C: Novel Setting → Familiar Everything Else

**INEFFECTIVE (Too Ambitious):**
> Graffiti Sketchbook Animation — In a lush alien jungle with bioluminescent plants, a robot explorer discovers an ancient artifact while performing acrobatic leaps, camera tracking dynamically.

**Problems:** OOD setting (alien jungle) + OOD character (robot) + complex action (acrobatics) + camera move.

**EFFECTIVE (Singular Focus on OOD Setting):**
> Graffiti Sketchbook Animation — A low-angle wide shot establishing a moonlit desert roadside at night. A weathered gas station sits alone in the frame, its fluorescent tubes casting green-white glow on cracked pavement. Tumbleweeds rest against rusted pumps. The scene is rendered with ROUGH PAINT-SWATCH TEXTURES on wood siding, SCRIBBLED LINEWORK defining the station edges, and heavy CROSS-HATCHING in shadows. Cool silver moonlight as key, warm amber glow from station windows. Static shot, no characters.

**Why it works:** Focus on translating "desert gas station" into TMNT style. No characters to complicate, static camera, simple composition.

### Practical Guidelines for OOD Prompts

**When going off-road, follow this checklist:**

1. **Identify what's OOD:** Is it the character? Setting? Action? Effect?
2. **Make that ONE thing the star:** Put it first, describe it in detail
3. **Simplify everything else:**
   - Use familiar TMNT settings when possible (alleys, rooftops, subways)
   - Keep action simple (standing, walking, single gesture)
   - Use static camera or one simple move
   - Avoid multiple characters unless they're all OOD together
4. **Anchor heavily in style terms:** Use all core anchors (SCRIBBLED, ROUGH, CROSS-HATCHING) to ground the OOD element in TMNT aesthetic
5. **Test incrementally:** Try medium-distance concepts before high-distance ones

### Examples by Distance Level

**Low Distance (Easy OOD):**
> Graffiti Sketchbook Animation — A medium shot of a mutant raccoon in a torn leather vest, crouched on a fire escape at night. His ringed tail hangs down between the grates. Cool silver moonlight casts shadows through the metal bars with CROSS-HATCHING. Rendered with SCRIBBLED LINEWORK on fur and ROUGH PAINT-SWATCH TEXTURES on brick wall. Deep blue AMBIENT HUE. Static camera.

**Medium Distance (Moderate OOD):**
> Graffiti Sketchbook Animation — A low-angle shot of a street magician's hands performing a card flourish, cards fanning with EXTREME SMEAR FRAMES. His fingerless-gloved hands show SCRIBBLED LINEWORK at the edges. Warm golden hour light as key, cool blue AMBIENT HUE in shadows. A small burst of 2D-STYLE DOODLED EFFECTS—hand-drawn sparkles—appears at the moment of the flourish. ROUGH PAINT-SWATCH TEXTURES on vest. Close focus, blurred background.

**High Distance (Challenging OOD):**
> Graffiti Sketchbook Animation — A medium wide shot of a 1920s noir detective in a fedora and trench coat, standing under a streetlamp in a rain-slicked alley. He lights a cigarette, the match flame briefly illuminating his face. The scene is rendered with SCRIBBLED GRAPHITE-LIKE OUTLINES, ROUGH PAINT-SWATCH TEXTURES on his coat, and heavy CROSS-HATCHING in the deep shadows. Warm amber key light from streetlamp above, cool blue AMBIENT HUE from moonlight. SHALLOW DEPTH OF FIELD. Static camera at eye level.

**Notice:** As distance increases, we rely MORE heavily on style anchors and familiar TMNT visual language to ground the foreign concept.

---

## 13. Quick Reference Checklist

Before finalizing your prompt, verify:

- [ ] Starts with "Graffiti Sketchbook Animation —"
- [ ] Uses appropriate Narrative Priority pattern (Setting-First, Character-First, or Object-First)
- [ ] Puts the most important element FIRST (what is the reason this shot exists?)
- [ ] If going OOD: identified what's novel and simplified everything else
- [ ] Identifies characters using name + descriptor (if applicable)
- [ ] Includes 2-3 core style anchors (SCRIBBLED, ROUGH, CROSS-HATCHING)
- [ ] Includes 1-3 supporting style terms (SHALLOW DOF, etc.)
- [ ] Describes one primary action/beat clearly
- [ ] Specifies one key light + one ambient color
- [ ] Uses present tense, third person
- [ ] Has one camera move max (or static)
- [ ] Word count: 50-80 (quick), 90-150 (standard), 160-200 (complex)

---

## 14. Appendix: Common Style Terms by Frequency

**ESSENTIAL (use 2-3 per prompt):**
- SCRIBBLED LINEWORK / SCRIBBLED GRAPHITE-LIKE OUTLINES
- ROUGH PAINT-SWATCH TEXTURES
- CROSS-HATCHING

**COMMON (use 1-2 selectively):**
- HIGH-CONTRAST
- SHALLOW DEPTH OF FIELD
- AMBIENT HUE
- ASYMMETRICAL CHARACTER DESIGN

**SITUATIONAL (use when appropriate):**
- STUTTERING FRAME RATE (motion)
- FLUORESCENT GLOW (screens, mutagen, neon)
- SIMULATED HANDHELD CAMERA (documentary realism)
- DOODLED EFFECTS (impacts, energy)
- HARSH DIRECT / SOFT DIFFUSED (light quality)
- VOLUMETRIC RAYS (atmosphere, beams)
- GRITTY LENS FLARE (optics)
- SCRATCHY HAND-DRAWN ACTION LINES (emphasis)

**SPECIALIZED:**
- VISCOUS, GLOWING GREEN MUTAGEN (specific to TMNT content)
- EXTREME SMEAR FRAMES (extreme fast motion)

---

**End of Guidelines**

*Next revision will incorporate WAN 2.2 model-specific optimization (attention budgets, token weighting, T5 encoder behavior).*
