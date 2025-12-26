### **Prompting Strategy: The "High-Density, Low-Token" Prompt (Parsimony for Img2Vid)**

**Lesson:** Prompts for an **image-to-video (img2vid)** workflow should be radically simplified and focused, using a "high-density, low-token" approach. Unlike text-to-video, where the prompt must build the world from scratch, an img2vid prompt should only describe the **change or motion** that needs to occur. **However, it is important to make minor references to the input or reference image so that the model can better “see” that image and respect it.**

**Rationale:** In an img2vid workflow, the start frame has already done 90% of the work. It has established the subject, the style, the composition, the lighting, and the color palette, effectively pre-spending the AI's "Attention Budget." Our role is no longer that of an architect building a scene, but that of a director on set giving a simple, clear instruction to an actor who is already in costume. Any token in an img2vid prompt that re-describes what is already clearly visible on screen is a distraction. It wastes the AI's limited cognitive resources and risks pulling focus from the primary directive, which is the animation itself. The goal is to **Guide, not Reinvent.**

---

### **Few-Shot Example: Executing a Simple Action from a Start Frame**

**The Goal:** We have a start frame showing Tony standing still in a detailed "Graffito" alley. We want him to perform a simple action: take a step forward.

* **INEFFECTIVE (Txt2Vid approach applied to Img2Vid):**  
  Graffito Mixed-Media Stop-Motion: In a gritty alleyway, a small puppet figure takes a step forward. The subject is Tony, a 7-year-old puppet with a PHOTOGRAPHIC CUTOUT face and a PAINTED PAPER CUTOUT body. The scene is a detailed alley diorama made of LAYERS OF PAINTED CARDBOARD with GRAFFITI MURALS. The lighting is a moody downlight. Tony takes a single, jerky stop-motion step forward.  
  * **Critique:** This is a classic mistake. The prompt wastes its entire "Attention Budget" re-describing the character and the scene, which the AI can already see perfectly in the start frame. This "noise" can confuse the model, causing it to alter the character's appearance, change the lighting, or fail to execute the simple motion command.  
*   
* **EFFECTIVE (High-Density, Low-Token Img2Vid Approach):**  
  Graffito Mixed-Media Stop-Motion: Tony takes a single, heavy-footed, jerky stop-motion step forward.  
  * **Critique:** This is a perfect img2vid prompt. It is lean, focused, and respects the start frame. It trusts that the AI already knows who Tony is and where he is. The prompt dedicates 100% of its "Attention Budget" to the single most important piece of new information: the **quality and direction of the motion**. This gives the AI the best possible chance of executing the desired animation with high fidelity, without introducing unwanted visual changes.  
* 

