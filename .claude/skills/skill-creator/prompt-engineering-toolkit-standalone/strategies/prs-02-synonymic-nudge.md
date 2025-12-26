### **Prompt Revision Strategy 2: The Synonymic Nudge (For Creative Generalization)**

**Lesson:** The "Synonymic Nudge" is a primary revision strategy used to solve several common creative problems, most notably **overfitting**. The technique involves taking a working prompt and simply swapping out the specific, reserved material and motion tokens from the training captions (e.g., PAINTED CARDBOARD) with their **synonymic equivalents** (e.g., layered paper mache).

**Rationale:** This technique serves four crucial purposes in our creative workflow:

1. **Combating Overfitting:** Its primary use is to break the LoRA's tendency to "snap" all new scenes into the visual orbit of the training data. When every new building starts to look like the Bronx school, this is our tool to force visual variety.  
2. **De-Cluttering the Frame:** Sometimes, a prompt with too many strong, reserved tokens can cause the AI to get confused and "shoe-horn" multiple, unrelated trained concepts into a single shot, leading to a visually chaotic and cluttered result. By swapping some reserved tokens for synonyms, we can simplify the AI's task and encourage it to focus on a single, clearer composition.  
3. **Encouraging Generalization:** It is a powerful way to test and encourage the LoRA to apply its learned *style* (the gritty, handcrafted feel) to new *forms* and *textures*. It's how we discover the full range of what the "Graffito" aesthetic can be.  
4. **Achieving Nuance and Subtlety:** Our expanded synonymic lexicon contains words with slightly different connotations (dense pulp board feels heavier than laminated paper stock). Using these synonyms allows us to achieve more nuanced and specific results that might be difficult to get with the broader, core tokens alone.

---

### **Few-Shot Examples by Application**

#### **Example 1: Fixing an Overfit Prop**

**The Problem:** We are trying to generate a shot of a **vintage rotary telephone**, but because we used PAINTED CARDBOARD, the result looks clunky, rectangular, and too much like a small building from the training data.

* **ORIGINAL PROMPT (Causing Overfitting):**  
  Graffito Mixed-Media Stop-Motion: A vintage rotary telephone. The subject is a telephone prop, its heavy black body constructed from \*\*PAINTED CARDBOARD\*\*.  
  * **Critique:** The powerful token PAINTED CARDBOARD is so strongly associated with architecture that it is distorting the form of the telephone.  
*   
* **REVISED PROMPT (Using Synonymic Nudge):**  
  Graffito Mixed-Media Stop-Motion: A vintage rotary telephone. The subject is a telephone prop, its heavy black body suggesting a form of \*\*sculpted paper pulp\*\* covered in a glossy black \*\*dimensional paint daub\*\*.  
  * **Critique:** This revision is successful because it uses synonyms that are not tied to architecture. **Sculpted paper pulp** and **dimensional paint daub** still evoke a handcrafted, "Graffito" feel, but they give the AI the freedom to create the correct, curved shape of a telephone without being biased by the hard edges of a cardboard box.  
* 

#### **Example 2: Creating a New, In-Style Character**

**The Problem:** We need to create a new character, a **bodega owner**, but we want him to look distinct from Tony and the other characters, who are all made of PAINTED PAPER CUTOUTS.

* **ORIGINAL PROMPT (Risking Visual Repetition):**  
  Graffito Mixed-Media Stop-Motion: A bodega owner. The subject is a man, his body made from \*\*PAINTED PAPER CUTOUTS\*\*.  
  * **Critique:** This will likely produce a character that looks like a variation of Tony. It doesn't encourage the model to create a new type of person.  
*   
* **REVISED PROMPT (Using Synonymic Nudge):**  
  Graffito Mixed-Media Stop-Motion: A bodega owner. The subject is a man, his heavy-set body suggesting a construction of \*\*molded fiber sheets\*\*. He is dressed in a simple shirt made from \*\*creased and weathered textiles\*\*.  
  * **Critique:** This is a much better approach for creating a new character. By using synonyms like **molded fiber sheets** and **weathered textiles**, we are asking the AI to build a new person from a different set of (but stylistically related) materials, ensuring he will have a unique texture and feel while still belonging in the "Graffito" universe.  
* 

#### **Example 3: Adding Nuance to an Environment**

**The Problem:** We want to create a shot of a **tenement stoop**, but we want it to feel particularly solid, heavy, and ancient, more so than our standard cardboard sets.

* **ORIGINAL PROMPT (Good, but Generic):**  
  Graffito Mixed-Media Stop-Motion: Three men sit on a tenement stoop made of \*\*PAINTED CARDBOARD\*\*.  
  * **Critique:** This is a perfectly fine prompt, but "PAINTED CARDBOARD" might result in a lighter, more set-like feel than we want for this specific shot.  
*   
* **REVISED PROMPT (Using Synonymic Nudge for Nuance):**  
  Graffito Mixed-Media Stop-Motion: Three men sit on a tenement stoop. The stoop appears to be sculpted entirely from dimensional, gritty layers of gray \*\*heavy modeling paste\*\* over a core of \*\*dense pulp board\*\*.  
  * **Critique:** This revision is more nuanced and specific. The synonyms **heavy modeling paste** and **dense pulp board** carry a stronger connotation of weight, solidity, and sculptural form than our standard PAINTED CARDBOARD. This is a perfect example of using the synonymic lexicon not just to avoid bias, but to dial in a very specific texture and feeling.

---

### **Modular Appendage for "The Synonymic Nudge" Markdown**

---

### **Advanced Application: The "Cast of Characters" Methodology**

* **Principle:** One of the most powerful and specific use cases for the "Synonymic Nudge" technique is in generating a diverse cast of secondary characters for synthetic data training.  
* **Workflow:** This methodology combines the "Synonymic Nudge" with several other core lessons to create detailed and unique character portraits:  
  * **Frame the prompt as a "Character Portrait"** to focus on a single, isolated figure.  
  * **Dedicate the "Attention Budget" to the Subject component**, focusing on a detailed, iconic, and era-specific costume.  
  * **Apply the "Synonymic Nudge"** to the materials of the character's body and clothing to ensure they are visually distinct from any main, trained characters and from each other.  
  * **Simplify the Scene and Motion** to a simple diorama and a looping, "NPC-style" action.  
*   
* **Example Application:**  
  * **Goal:** Create a Stoop Sitter character that doesn't look like our main character, Tony.  
  * **Action:** We use the "Synonymic Nudge" to swap PAINTED PAPER CUTOUTS (associated with Tony) for **sculpted paper pulp** and **creased and weathered textiles**. This successfully applies the general tool of synonymic prompting to the specific, high-level task of character diversification.

