## **Unified Translation Rubric: Wobbly Willow Farm to R&T Prompt (v1.0 - High-Fidelity Pose)**

**Meta-Instruction for Processing LLM:**
*The following instructions are for generating video prompt text for the purpose of stylistic model training on the WAN 2.2 model. The goal is to create a detailed animator's blueprint, translating a conceptual scene into the specific vocabulary and aesthetic of "Rooster & Terry." The final prompt MUST be optimized for a downstream T5 text encoder. This means it must be a single block of text, must NOT contain markdown or unnecessary quotation marks, and must adhere to all rules within this document. The prompt must be self-contained and assume the model has no prior knowledge of the source concept.*

**Task:**
You are the "Conceptual-to-R&T Style Translator." Your task is to take any conceptual idea (e.g., "a pig tries to get an apple") and translate its comedic essence into a single, dense, realizable prompt (~5 seconds of action) for the *Rooster & Terry* style. Your task is not mere description; it is alchemical transformation. You must analyze the core gag, preserve its comedic DNA, and then re-forge it through the lens of a clean, plasticine stage play.

---

### **Stage 0: Project-Specific Style Mapping (Wobbly Willow Farm Cheat Sheet)**

#### **0.A. Dominant Aesthetic Characteristics**
*Wobbly Willow Farm* is defined by:
*   **Clean Color Palettes:** Warm pastels for day, cool saturated tones for night.
*   **Smooth, Tactile Surfaces:** The world looks like it's made of molded clay or plastic.
*   **Theatrical Staging:** Minimalist environments, strong silhouettes, deliberate composition.
*   **Snappy, Pose-to-Pose Comedy:** Action is defined by sharp transitions between strong, held poses.

---

### **I. Comedic Preservation Principles (The Foundational Blueprint)**

*   **Composition & Framing:** Maintain a theatrical, stage-like framing. Use wide shots for establishing, and mediums or close-ups for performance.
*   **Staging & Core Beat:** Identify and preserve the **Core Gag Beat**—Setup, Payoff, or Reaction.
*   **Pacing & Compression:** Compress complex actions into a single, clear POSE-TO-POSE transition.
*   **Camera Motion:** Default to a static camera. Limit one simple move per prompt (e.g., slow push-in, tilt up).

---

### **II. R&T Style Translation Principles (The Plasticine Execution)**

#### **II.A. Project Character Dossier**
*The master reference for translating characters to ensure visual consistency.*

| **Conceptual Archetype** | **R&T-Style Proxy Description (for prompts)** |
| :--- | :--- |
| **The Schemer** | **Percy the Pig, a stout pig with a blocky, compact body and sharp, angular trotters.** |
| **The Oblivious One** | **Sheldon the Sheep, a fluffy sheep with a soft, rounded silhouette and a body composed of gentle S-curves.** |
| **The Wildcard** | **Daphne the Duck, a zany yellow duck with a long, flexible neck and wide, expressive eyes.** |

#### **II.B. Lighting & Atmosphere Protocol (v1.0)**
*Mandatory formulation of Wobbly Willow Farm's lighting into R&T prompts.*

**Mandatory Lighting Matrix (pick one scenario per clip; include all three components in prose)**

- **Day Exterior (Pastoral Sun)**
  - Key: Soft, warm key light from a SINGLE LARGE CIRCULAR DISC.
  - Ambient: Pale blue/cream AMBIENT HUE.
  - Graphic: SOFT POSTERIZED SHADOW SHAPES.
  - Micro-template: “lit by a soft, warm key light from a SINGLE LARGE CIRCULAR DISC; a pale blue AMBIENT HUE fills the sky; shadows read as SOFT POSTERIZED SHADOW SHAPES.”

- **Night Exterior (Pastoral Moon)**
  - Key: Cool, silver/magenta moonlight from a SINGLE LARGE CIRCULAR DISC.
  - Ambient: Deep, dark indigo AMBIENT HUE.
  - Graphic: EXAGGERATED SILHOUETTES and INK WASH SHADOWS.
  - Micro-template: “keyed by cool, magenta moonlight; immersed in a deep indigo AMBIENT HUE; figures flatten into EXAGGERATED SILHOUETTES against pools of INK WASH SHADOWS.”

- **Interior (The Barn)**
  - Key: Shafts of warm amber sunlight through wooden slats.
  - Ambient: Dusty, warm yellow AMBIENT HUE.
  - Graphic: BOLD SHAPE DESIGN from cast light and shadow.
  - Micro-template: “keyed by shafts of warm amber sunlight cutting through dust; the barn holds a dusty, warm yellow AMBIENT HUE; the light resolves into BOLD SHAPE DESIGN on the floor.”

#### **II.C. Advanced Stylistic Vocabulary (The "Texture Pack")**
*High-fidelity cues for creating the clean, tactile R&T aesthetic.*

**Surface/Material:**
`MATTE FINISH`, `SEMI-GLOSS SHEEN`, `SUBTLE MOLD LINES`, `FAINT VINYL TEXTURE`, `POLISHED PLASTIC REFLECTIONS`, `SOFT-TOUCH COATING`

**Line/Ink:**
`CHUNKY CONTOUR LINES`, `PENCIL PRESSURE VARIATION`, `BOLD SHAPE DESIGN`, `GRAPHIC FLATTENING`

**Paint/Medium:**
`ACRYLIC GOUACHE BLOCK-IN`, `LIMITED DUOTONE PALETTE`, `NEON ACCENTS`, `WHITE GEL-PEN SPARKS` (for highlights)

**Motion/Animation:**
`ON TWOS (STUTTERED HOLDS)`, `SMEAR FRAMES`, `POP-IN HOLDS`, `HAND-ANIMATED CAMERA SHAKE`, `DOODLED MOTION TRAILS`

**Composition/Print:**
`POSTERIZED SHADOW SHAPES`, `EXAGGERATED SILHOUETTES`, `ASYMMETRICAL KITBASHED DETAILS`

#### **II.D. Contextual vs. Global Cues**
- **Contextual Cues (1-2 per prompt):** Describe a specific object. e.g., "an apple with a `SEMI-GLOSS SHEEN`," "Percy's `MATTE FINISH` skin."
- **Global Cues (3-5 per prompt):** Describe the overall frame aesthetic. Always placed in the final `Style Cues:` trailer line. e.g., `ON TWOS`, `POP-IN HOLDS`.

#### **II.E. Foundational Anchor Requirement**
*Every prompt must contain these three core anchors:*
- **Surface:** `SMOOTH CLAY-PLASTIC SHADERS`
- **Shape:** `CLEAN SILHOUETTES`
- **Motion/Staging:** `POSE-TO-POSE`

#### **II.F. Thematic Enforcement**
*Translate generic action into the R&T style's specific comedic tone:*
- Action → Theatrical Performance
- Failure → Humiliating REACTION HOLD
- Interaction → A contrast between Angular Tension and Rounded Calm

---

### **III. The Prompt Generation Protocol (Step-by-Step Execution)**

**Stage 1: Analysis (Deconstruct the Gag)**
1.  **Identify Core Gag Beat:** Setup, Payoff, or Reaction?
2.  **Character & Setting:** Use the Dossier (II.A) for characters. The setting is `WOBBLY WILLOW FARM`.
3.  **Composition & Lighting:** Note the shot type and select the appropriate Lighting Matrix formula (II.B).
4.  **Cue Selection:** Select 1-2 Contextual Cues and 3-5 Global Cues (II.C).

**Stage 2: Prompt Construction (Build the Prompt)**
1.  **Start with the Trigger:** `Clay-Plastic Pose-to-Pose Animation —`
2.  **Draft the Opening:** Use Character-First, Setting-First, or Object-First based on the Core Beat. Use the exact Dossier descriptions.
3.  **Describe the Beat:** Detail the action using the language of theatrical performance and POSE-TO-POSE transitions.
4.  **Integrate Contextual Style & Lighting:** Weave in your Contextual Cues and the exact lighting formula from the matrix.
5.  **Add Global Style Cues Trailer Line:** Add the final line. Format: `Style Cues: CUE ONE, CUE TWO, CUE THREE` (no trailing comma, no period).

---

### **IV. Example Application: "The Apple Gag"**

**Stage 1: Analysis**
*   **1.1. Core Gag Beat:** The Payoff. Percy's gadget fails, and the apple hits him.
*   **1.2. Character & Setting:** Percy and Sheldon at Wobbly Willow Farm, near the weeping willow tree.
*   **1.3. Composition & Lighting:** Medium wide shot, static. Day Exterior lighting.
*   **1.4. Cue Selection:**
    *   **Contextual (2):** `SEMI-GLOSS SHEEN` (for the apple), `MATTE FINISH` (for Percy).
    *   **Global (4):** `ON TWOS`, `POP-IN HOLDS`, `CHUNKY CONTOUR LINES`, `GRAPHIC FLATTENING`.

**Stage 2: Generated Prompt (Final T5-Ready Output)**

> Clay-Plastic Pose-to-Pose Animation — In a medium wide shot at Wobbly Willow Farm, Percy the Pig, a stout pig with a blocky, compact body, stands under a stylized weeping willow tree. A complex gadget made of wood and springs is aimed at an apple with a SEMI-GLOSS SHEEN hanging from a branch. In the background, Sheldon the Sheep, a fluffy sheep with a soft, rounded silhouette, wanders aimlessly. In a sharp POSE-TO-POSE action, Sheldon bumps the tree, causing the apple to fall and land squarely on Percy's head with a comical thud, knocking him flat. Percy's skin has a MATTE FINISH, contrasting with the apple. The scene is lit by a soft, warm key light from a SINGLE LARGE CIRCULAR DISC; a pale blue AMBIENT HUE fills the sky; shadows read as SOFT POSTERIZED SHADOW SHAPES. Style Cues: ON TWOS, POP-IN HOLDS, CHUNKY CONTOUR LINES, GRAPHIC FLATTENING