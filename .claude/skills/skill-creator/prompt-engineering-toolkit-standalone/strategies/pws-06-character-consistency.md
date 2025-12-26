### **Prompt Writing Strategy 6: Enforcing Character Consistency**

**Lesson:** To ensure visual consistency for our main, canonical characters, their full, established visual description **MUST** be included in the Subject component of the prompt whenever they appear in their "hero" costume. For non-canonical attire, we must be explicit in both describing the new and, potentially, negating the old.

**Rationale:** The LoRA has learned a strong visual association for our main characters based on their training data. Simply using a character's name (e.g., "Tony") is not a reliable instruction and can lead to inconsistent or generic results. Providing the full, detailed description acts as a powerful and unambiguous command, forcing the AI to render the character with perfect, on-model accuracy every single time.

---

#### **Principle A: Using the Full "Hero" Costume Description**

**Goal:** A shot of Tony in his standard, recognizable outfit.

* **THE CANONICAL DESCRIPTION (The Anchor):**  
  Tony, a 7-year-old puppet with a \*\*PHOTOGRAPHIC CUTOUT\*\* face of a young boy with curly hair and expressive green eyes, and a \*\*PAINTED PAPER CUTOUT\*\* body wearing his signature light blue shirt with a red collar and beige pants.  
* **INEFFECTIVE (Vague & Unreliable):**  
  Graffito Mixed-Media Stop-Motion: Tony stands in an alleyway.  
  * **Critique:** This is a high-risk prompt. The AI might generate a generic boy, or a character that only vaguely resembles Tony. We are leaving his appearance entirely to chance.  
*   
* **EFFECTIVE (Specific & Reliable):**  
  Graffito Mixed-Media Stop-Motion: A lonely Tony stands on a street corner. The subject is \*\*Tony, a 7-year-old puppet with a PHOTOGRAPHIC CUTOUT face of a young boy with curly hair and expressive green eyes, and a PAINTED PAPER CUTOUT body wearing his signature light blue shirt with a red collar and beige pants.\*\* The scene is a simple, gritty street corner...  
  * **Critique:** This is a professional-grade prompt. By including the full, non-negotiable description in the Subject component, we are guaranteeing that the character will be rendered with perfect visual consistency, exactly as he appears in the training data.  
* 

---

#### **Principle B: Describing Non-Canonical Attire (Negation & Addition)**

**Goal:** A shot of a new character, a DJ, wearing a specific 70s outfit. This is non-canonical.

* **THE GOAL OUTFIT:** A sleeveless denim vest over a thermal shirt, with cargo pants.  
* **INEFFECTIVE (Vague Description):**  
  Graffito Mixed-Media Stop-Motion: A DJ is mixing records. He is wearing a vest and pants.  
  * **Critique:** This is far too generic. The AI will likely default to a random, uninspired outfit. It gives no specific stylistic or material direction.  
*   
* **EFFECTIVE (Detailed & Material-Specific):**  
  Graffito Mixed-Media Stop-Motion: A DJ is mixing records. The subject is a DJ, his uniform iconic: a \*\*heavy, sleeveless denim battle vest made from a dark, weathered, and frayed denim-like paper\*\*. Underneath the vest is a \*\*simple, tight-fitting thermal shirt made from a thin, waffle-textured textile\*\*, with its long sleeves pushed up to the elbows. He wears a pair of \*\*army surplus-style cargo pants made from a heavy, olive-green canvas material\*\*.  
  * **Critique:** This is successful because it applies the same level of material specificity to the new costume as we do to our canonical ones. It uses our synonymic lexicon to build a unique, tangible, and visually interesting outfit from the ground up, ensuring it feels like a real, handcrafted piece from the "Graffito" world.  
* 

---

#### **Principle C: Overriding Strong Biases with Negation (Advanced)**

**Goal:** A shot of Tony, but without his signature red-collar shirt, wearing a simple t-shirt instead.

* **INEFFECTIVE (Simple Replacement \- High Risk):**  
  Graffito Mixed-Media Stop-Motion: Tony, a 7-year-old puppet...wearing a simple white t-shirt.  
  * **Critique:** The LoRA's association between "Tony" and his "light blue shirt with a red collar" might be so strong that it overrides the new instruction and generates him in his hero costume anyway.  
*   
* **EFFECTIVE (Negation \+ Addition \- More Reliable):**  
  Graffito Mixed-Media Stop-Motion: Tony is in his room. The subject is Tony, a 7-year-old puppet... \*\*he is not wearing his usual blue shirt with the red collar\*\*, instead he wears a simple, plain white t-shirt made from a \*\*thin, creased textile\*\*.  
  * **Critique:** By explicitly negating the default item (not wearing his usual blue shirt), we are giving the AI a direct command to break its own bias. This, combined with the clear description of the new item, dramatically increases the chances of a successful, non-canonical costume change. This should be tested and observed.  
* 

