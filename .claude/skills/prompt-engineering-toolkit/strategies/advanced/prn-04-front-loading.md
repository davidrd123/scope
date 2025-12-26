### **Prompt Revision Strategy: Strategic Front-Loading (Forcing Visual Priority)**

**Lesson:** "Strategic Front-Loading" is an advanced revision strategy used to correct a prompt when the AI is failing to prioritize the most important visual element in the frame. The technique involves **deliberately breaking the standard Subject \+ Scene \+ ... formula** and re-ordering the prompt to place the true "hero" of the shot—whether it's a character, a prop, or an effect—at the absolute beginning of the descriptive paragraph.

**When to Use This Strategy:** This technique should be used when you absolutely need to see a certain phenomenon or concept prioritized above all else in the generation.

**Rationale:** This is a brute-force application of the "Attention Budget" principle. By placing the most complex or important element's description at the very start of the paragraph, we are telling the AI: "This is the most important thing. Get this right before you worry about anything else." It's a deliberate sacrifice of standard architectural structure for a guaranteed, laser-focused result, and it is essential for correcting prompts where the descriptive hierarchy does not match the desired visual hierarchy.

---

### **Few-Shot Example: Correcting a Mismatched Visual Hierarchy**

**The Goal:** A medium shot that is clearly and immediately focused on the disgusted reactions of two characters at a banquet.

* **INEFFECTIVE (Mismatched Hierarchy):**  
  Graffito Mixed-Media Stop-Motion — A medium shot frames a lavish diorama banquet hall constructed from layered PAINTED CARDBOARD and CRUMPLED PAPER, its walls textured with THICK IMPASTO PAINT. Distant GRAFFITI MURALS of stylized figures seem to droop with unease. In the foreground, the Grandma puppet, a PAINTED PAPER CUTOUT with a PHOTOGRAPHIC CUTOUT face contorted in a grimace of disgust, sits at a table. Beside her, the Monk puppet, his PHOTOGRAPHIC CUTOUT face featuring glasses and a beard, leans in with a concerned expression...  
  * **Critique:** This prompt is a classic example of a mismatched hierarchy. The shot is a medium shot of two characters, but the prompt spends its initial, most powerful "Attention Budget" describing the background (the banquet hall). The AI is being told to focus on the environment first, which is a direct contradiction of the desired visual. This can lead to the characters being under-rendered or the AI attempting a wider, establishing shot.  
*   
* **EFFECTIVE (Using Strategic Front-Loading):**  
  Graffito Mixed-Media Stop-Motion: A medium shot frames two characters, Grandma and Monk, reacting with disgust at a grotesque banquet. \*\*\[RE-ORDERED: The Characters are now the Subject\]\*\* The subjects are the Grandma puppet, a PAINTED PAPER CUTOUT with a PHOTOGRAPHIC CUTOUT face contorted in a grimace of disgust, and the Monk puppet, his PHOTOGRAPHIC CUTOUT face showing a concerned expression. \*\*\[Scene with Background Demoted\]\*\* They are seated at a table in the foreground of a lavish diorama banquet hall, its walls constructed from layered PAINTED CARDBOARD. \*\*\[Motion\]\*\* Grandma's articulated paper arm, trembling with a subtle stop-motion wobble, holds a painted spoon over a silver bowl filled with a grotesque mixture of THICK IMPASTO PAINT...  
  * **Critique:** This is a successful revision. It **deliberately breaks the formula** by placing the detailed description of the two main characters *before* the description of the environment. It correctly identifies that in a medium shot of characters, the characters *are* the subject. The background is "demoted" to a secondary element. This forces the AI to dedicate its primary "Attention Budget" to rendering the crucial facial performances first, perfectly aligning the prompt's structure with the desired visual outcome.  
* 

