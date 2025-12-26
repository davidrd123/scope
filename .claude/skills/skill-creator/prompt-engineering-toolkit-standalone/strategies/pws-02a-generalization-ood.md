### **Prompt Writing Strategy 2.1: Generalization Strategies for OOD & In-Domain Prompts**

**Lesson:** The primary goal of a prompt is to guide the AI to a specific creative target. To achieve this, we must use different prompting strategies for "In-Domain" concepts (things the LoRA was trained on) versus "Out-of-Distribution" (OOD) concepts (new ideas). Every prompt should still be built around **one singular, clear core idea.**

**Rationale:** The LoRA has strong, pre-existing biases based on its training data (e.g., PAINTED CARDBOARD is strongly associated with Bronx buildings). For In-Domain subjects, we can leverage this bias with our detailed, core lexicon. For novel, OOD subjects, these strong biases can be a hindrance, causing the model to ignore the new idea and default to what it already knows. To successfully generalize, we must strategically weaken these associations.

---

#### **Strategy A: Prompting for In-Domain Subjects (Using the Core Lexicon)**

**Goal:** Create a classic, high-fidelity "Graffito" shot of the Bronx school. This is a core, In-Domain subject.

* **APPROACH (Use the Full, Reserved Lexicon):** For subjects that are central to the training data, we should use our most detailed and specific reserved tokens. This ensures maximum stylistic cohesion and reliably activates the LoRA's deepest training.  
* **EFFECTIVE PROMPT (Rich & Specific):**  
  Graffito Mixed-Media Stop-Motion: A static, establishing shot presents the grand, decaying facade of an inner-city school. The subject is the imposing school building prop, constructed from \*\*LAYERS OF PAINTED CARDBOARD\*\*. Its brick texture is created from thick, gritty strokes of red and brown \*\*THICK IMPASTO PAINT\*\*, with areas of black paint simulating soot stains. Tiny pigeon props made of \*\*CRUMPLED PAPER\*\* are perched on the windowsills.  
  * **Critique:** This works perfectly because we are "playing the hits." We are using the exact language the LoRA was trained on to get a high-fidelity, stylistically perfect result of a known subject.  
* 

---

#### **Strategy B: Prompting for OOD Subjects (Using Synonymic & Parsimonious Language)**

**Goal:** Create a novel "Graffito"-style landscape of a **snowy mountain cabin**. This is a completely OOD setting.

* **APPROACH (Use Synonymic & Parsimonious Language):** To prevent the model from ignoring the "cabin" and trying to render a "Bronx building in the snow," we must avoid our strongest reserved tokens associated with architecture. We will use a **synonymic lexicon** and a more **parsimonious (concise)** style to describe the scene, giving the AI the freedom to apply the *texture* of "Graffito" to a new *form*.  
* **INEFFECTIVE (Using Core Lexicon for OOD):**  
  Graffito Mixed-Media Stop-Motion: A massive cabin made of \*\*LAYERS OF PAINTED CARDBOARD\*\* with \*\*THICK IMPASTO PAINT\*\* textures sits in the snow.  
  * **Critique:** This prompt is high-risk. The powerful tokens LAYERS OF PAINTED CARDBOARD and THICK IMPASTO PAINT are so strongly tied to the Bronx tenements in the training data that the model may ignore the word "cabin" and generate a snowy tenement building instead.  
*   
* **EFFECTIVE (Using Synonymic & Parsimonious Style):**  
  Graffito Mixed-Media Stop-Motion: A wide, serene shot captures a small, isolated cabin nearly buried in deep snow on a mountainside. The cabin is a simple A-frame structure, its walls built from tiny logs of \*\*painted wood scraps\*\*. The entire world is covered in a thick, pristine blanket of "snow," a landscape sculpted from vast, soft mounds of \*\*white cotton batting\*\*. The jagged peaks of distant mountains are large cutouts of \*\*gesso-primed fiberboard\*\*.  
  * **Critique:** This is successful because it achieves generalization through several key strategies:  
    1. **It avoids the main architectural trigger words.** It uses painted wood scraps instead of PAINTED CARDBOARD.  
    2. **It uses our expanded synonymic lexicon** (gesso-primed fiberboard) to evoke a handcrafted feel without being overly prescriptive.  
    3. **It is concise and focused.** It describes the *essence* of the cabin and the snow, trusting the LoRA to apply its overall textural style to these new shapes.  
  * 

