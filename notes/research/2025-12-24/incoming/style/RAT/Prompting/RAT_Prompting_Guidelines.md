# Rooster & Terry / WAN 2.2 Prompting Guidelines

**Status:** v1.0 — Caption-aligned structure
**Owner:** davidrd
**Model:** WAN 2.2 (T5 text encoder)

---

## Overview

This guide teaches you to write prompts in the **style and grammar** of the *Rooster & Terry* training captions while **strategically simplifying** for effective prompting. The goal is not exhaustive description but focused visual direction that captures the unique comedic timing and clean, stylized aesthetic of the source.

**Key Principle:** Use the caption's technical vocabulary and sentence structure, but be selective. Lead with the gag's function, include essential style anchors, and describe one primary pose or action clearly.

---

## 1. Trigger Phrase (Required)

**Format:**
```
Clay-Plastic Pose-to-Pose Animation —
```

- **Always** start with this exact trigger
- Use em dash (—), not hyphen (-)
- Follow with a space before your content

**Why it matters:** This trigger activates the specific visual style of clean, 3D models animated with sharp, 2D timing.

---

## 2. The Principle of Gag Priority

**Core Principle:** The most important element—the *comedic function* of the shot—should be described first to dedicate the most attention and detail to it. Choose the pattern that matches your emphasis.

### Pattern A: The Setup (Gag Introduction)
Use when the primary goal is to show a character's confident planning, their gadget, or their smug pre-victory pose.

**Structure:**
```
[Shot type] of [character] in a [quality] pose, [implied intent]. [Setting & style anchors].
```

**Example:**
> Clay-Plastic Pose-to-Pose Animation — A low-angle medium shot of Rooster holding a strong key pose of supreme self-satisfaction, his chest puffed forward and his eyes narrowed smugly. He stands on a red-tiled rooftop under a bright sun. The scene is rendered with SMOOTH CLAY-PLASTIC SHADERS, a CLEAN SILHOUETTE, and a WARM PASTEL PALETTE.

**Use when:** The character's posture and planning are the focus.

### Pattern B: The Payoff (Gag Climax)
Use when the primary goal is to capture the moment of chaotic failure—the explosion, the crash, the trap backfiring.

**Structure:**
```
[Subject & action] resulting in [consequence/effect]. [Style & lighting].
```

**Example:**
> Clay-Plastic Pose-to-Pose Animation — A ROOSTER'S GADGET violently backfires, instantly transforming him into a plucked, pink chicken frozen in a shocked REACTION HOLD. The scene is lit with a COOL SATURATED PALETTE with strong RIM LIGHTING. The effect is punctuated by a GRAPHIC ONOMATOPEYA of the word KABOOM.

**Use when:** The cause-and-effect of the gag's failure is the core beat.

### Pattern C: The Reaction (Gag Aftermath)
Use when the primary goal is to focus on a character's exaggerated, held reaction to the payoff.

**Structure:**
```
[Shot type] of [character] frozen in an exaggerated [reaction] pose. [Physical details of the pose]. [Style anchors].
```

**Example:**
> Clay-Plastic Pose-to-Pose Animation — A GROTESQUE EXTREME CLOSE-UP of Rooster frozen in a shocked REACTION HOLD, his eyes bulging to massive proportions and his beak agape. The animation holds this strong key pose for several seconds. Rendered with SMOOTH CLAY-PLASTIC SHADERS and sharp RIM LIGHTING against a deep indigo sky.

**Use when:** A character's over-the-top, static reaction is the punchline.

---

## 3. Structural Flow

Every prompt should follow this rhythm:

```
ESTABLISH → STYLE → POSE/ACTION → LIGHTING → CONCLUDE
```

**ESTABLISH:** Open with shot type or subject + location
**STYLE:** Integrate 4-6 style anchors naturally throughout
**POSE/ACTION:** Describe the primary pose or a single, sharp movement
**LIGHTING:** Note the palette (Warm/Cool) + one key effect (Rim Light/Bloom)
**CONCLUDE:** End state or held pose

---

## 4. Length Guidelines

| Type | Word Count | Use Case |
|------|-----------|----------|
| **Quick Beat** | 40-70 words | Simple pose, establishing shot |
| **Standard** | 80-130 words | Character intro + pose, moderate scene complexity |
| **Complex** | 140-180 words | Multi-character interaction, gag with setup + payoff |

**Note:** Prompts are shorter than training captions. They need just enough detail to drive the visual output.

---

## 5. Character Identification

### First Mention
- **Rooster:** "Rooster, the angular orange rooster"
- **Terry:** "Terry, the rounded white and blue rooster"

### Subsequent Mentions
Use the name alone.

### Shape Language (Crucial for this style)
- **Rooster:** angular, triangular, sharp, tense, compact
- **Terry:** rounded, soft, S-curves, loose, balanced

---

## 6. Style Anchor Integration

Integrate these terms naturally. Don't just list them.

### Core Anchors (Use 2-3 per prompt)
- **SMOOTH CLAY-PLASTIC SHADERS**
- **CLEAN SILHOUETTES**
- **POSE-TO-POSE** (to describe motion or a held pose)
- **SIMPLIFIED GRAPHIC GEOMETRY** (for environments)

### Supporting Anchors (Use 1-3 selectively)
- **ROUNDED EDGES**
- **WARM PASTEL PALETTE** (Day)
- **COOL SATURATED PALETTE** (Night)
- **RIM LIGHTING** (especially at night)
- **SINGLE LARGE CIRCULAR DISC** (for the sun/moon)

### Specialized Terms (Gag-specific)
- **ROOSTER'S GADGET**
- **GRAPHIC ONOMATOPEYA**
- **GROTESQUE EXTREME CLOSE-UP**
- **REACTION HOLD**

**Total:** Aim for 4-6 CAPITALIZED terms integrated across the prompt.

---

## 7. Pose & Performance

### Present Tense, Third Person
**RIGHT:** "He holds a pose."
**WRONG:** "He held a pose."

### Describing the Pose (The "Action" of a Still Moment)
This is the core of the R&T style. Use the character-specific directives from the captions:

**For Rooster (Tension & Energy):**
- "holds a strong key pose"
- "his body is coiled with tension"
- "a sharp, angular silhouette"
- "frozen in a moment of explosive energy"

**For Terry (Ease & Balance):**
- "settles into a relaxed, balanced pose"
- "his body forms a loose, gentle S-curve"
- "a soft, rounded silhouette"
- "stands with his weight distributed evenly"

### Describing Movement (If any)
Use the three-part rhythm:
- **Anticipation:** "He crouches in anticipation..."
- **Pop:** "He snaps upward in a sharp pop..."
- **Hold:** "...settling into a new held pose."

---

## 8. Lighting & Color Language

### Lighting Formula
1.  **Name the Palette** (Day or Night)
2.  **Name one key effect** (Bloom or Rim Light)

**Examples:**
- "The scene is rendered in a WARM PASTEL PALETTE, with a soft diffuse bloom from the sun."
- "The scene is bathed in a COOL SATURATED PALETTE, with strong magenta RIM LIGHTING defining his form."

---

## 9. What to Simplify (Strategic Selection)

### ✅ Always Include
1. Trigger phrase
2. Primary subject (Rooster/Terry) & their shape language
3. Primary pose/action (the point of the shot)
4. 2-3 core style anchors (SMOOTH SHADERS, CLEAN SILHOUETTES, POSE-TO-POSE)
5. Lighting palette (Warm/Cool)

### ❌ Skip or Minimize
- Exhaustive environmental cataloging
- Micro-details (e.g., the specific text on a box)
- Multiple sequential beats (focus on ONE key moment)
- Redundant style terms

---

## 10. Real Examples with Simplification

### Example 1: The Setup

**Full Caption (113 words):**
> Clay-Plastic Pose-to-Pose Animation — Rooster poses triumphantly with a trophy on a rooftop. In a low-angle medium shot against a bright blue sky, Rooster stands on one leg atop the peak of a red-tiled roof at the ROOFTOP FARM CREST. His orange and brown body, rendered in a SMOOTH CLAY-PLASTIC SHADER style, has a distinct angularity and a CLEAN SILHOUETTE. His expression is one of disdainful victory, with heavy, half-closed red eyelids and his beak pointed slightly down. He casually holds a small golden trophy in his left wing. The scene is lit with a WARM PASTEL PALETTE under a large white circular disc for a sun, creating a soft bloom effect. The background is composed of SIMPLIFIED GRAPHIC GEOMETRY, with blocky white clouds and the roof of another small building visible. The entire composition is held in a static POSE-TO-POSE hold, emphasizing his victorious stance.

**Simplified Prompt (78 words):**
> Clay-Plastic Pose-to-Pose Animation — A low-angle medium shot of Rooster, the angular orange rooster, holding a strong key pose of disdainful victory on a rooftop. He stands on one leg, holding a golden trophy. The scene is rendered with SMOOTH CLAY-PLASTIC SHADERS and a CLEAN SILHOUETTE against a bright blue sky. Lit by a WARM PASTEL PALETTE with a soft bloom from a SINGLE LARGE CIRCULAR DISC for the sun. The composition is a static POSE-TO-POSE hold.

### Example 2: The Reaction

**Full Caption (143 words):**
> Clay-Plastic Pose-to-Pose Animation — An obsessive rooster's schemes to outsmart his rival consistently backfire with disastrous results. Set at night on the ROOFTOP FARM CREST, the scene is rendered in a COOL, SATURATED PALETTE with a deep indigo sky. The camera tilts up a large, abstract structure, revealing Rooster in a transformative state, completely covered from head to toe in thick, black, dripping tar. He laboriously pulls himself up and slumps onto a weathervane in a slow POSE-TO-POSE movement, settling into a final REACTION HOLD. The tar drips from his form, catching a strong magenta RIM LIGHTING, and his eyes glow a vibrant green from within the black mass, giving him a dazed and weary expression against the night sky.

**Simplified Prompt (85 words):**
> Clay-Plastic Pose-to-Pose Animation — A medium shot of Rooster in a transformative state, completely covered in thick, black, dripping tar. He slumps onto a weathervane in a final, defeated REACTION HOLD. His tar-covered form is rendered with SMOOTH CLAY-PLASTIC SHADERS, his dazed green eyes glowing from within the black mass. The scene is lit by a COOL SATURATED PALETTE, with strong magenta RIM LIGHTING defining his CLEAN SILHOUETTE against a deep indigo night sky.

---

## 11. Prompting Off-Road (Beyond Rooster & Terry)

To use this style on other subjects, focus on the core aesthetic: **clean, toy-like forms and strong, theatrical poses.**

**Keep the Grammar, Change the Nouns:**
- **R&T Style:** "Rooster, the angular orange rooster, holds a strong key pose..."
- **Off-Road Style:** "A noir detective, a figure of sharp, blocky shapes, holds a strong key pose..."

**Maintain Style Anchors:**
- SMOOTH CLAY-PLASTIC SHADERS
- CLEAN SILHOUETTES
- POSE-TO-POSE
- SIMPLIFIED GRAPHIC GEOMETRY

**Example:**
> Clay-Plastic Pose-to-Pose Animation — A low-angle shot of a noir detective in a fedora, a figure of sharp, angular shapes, holding a strong key pose under a streetlamp. The scene is rendered with SMOOTH CLAY-PLASTIC SHADERS and a CLEAN SILHOUETTE. Lit by a COOL SATURATED PALETTE, with harsh RIM LIGHTING from the streetlamp casting long, graphic shadows. The background is composed of SIMPLIFIED GRAPHIC GEOMETRY.