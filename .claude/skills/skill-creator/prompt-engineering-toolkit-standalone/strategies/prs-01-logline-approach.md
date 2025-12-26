### **Prompt Revision Strategy 1: The Logline Approach (Core Concept Distillation)**

**Lesson:** "The Logline Approach" is an advanced revision technique used for debugging, concept testing, and creative exploration. It involves stripping a complex prompt down to a single, concise, filmmaking-style **logline** that contains only the most critical action or idea, completely free of our specialized lexicon and structural rules.

**Rationale:** This technique is a deliberate "incision" designed to achieve one primary goal: **to untether the prompt from the specific stylistic biases of a LoRA and see how the foundational WAN2.2 base model interprets the core creative concept.** By removing all the stylistic scaffolding and reducing the scene to its logline, we get a raw, unfiltered look at the model's foundational understanding. This is an invaluable tool for diagnosing why a complex prompt is failing or for generating a wide range of fresh, unexpected starting points from which we can then begin to layer our style back on.

---

### **Few-Shot Example: Distilling a Complex Scene to its Logline** 

**Goal:** A complex, atmospheric shot of the masked Graffito legends bringing color back to a dead, monochrome city.

* **FULL, DETAILED PROMPT (For High-Fidelity Generation):**  
  Graffito Mixed-Media Stop-Motion: In a stark black-and-white fantasy sequence, mysterious masked figures rappel down a building, bringing vibrant, magical color back to a dead city. The subjects are several mysterious figures whose bodies are made of black PAINTED PAPER CUTOUTS. The scene is a gritty city street diorama, constructed from PAINTED CARDBOARD and rendered entirely in high-contrast black and white. The figures descend with fluid, silent, stop-motion movements, rappelling down ropes. As they descend, they aim their spray cans and unleash bursts of intensely vibrant, glowing, multi-colored THICK IMPASTO PAINT. The brilliant paint hits the monochrome walls, creating beautiful GRAFFITI MURALS. The camera is a static, low-angle wide shot. The only color comes from the glowing paint itself. The scene feels like a mythic act of creation.  
* **LOGLINE PROMPT (For Core Concept Validation & Exploration):**  
  Graffito Mixed-Media Stop-Motion: In a black and white city at night, masked ninja-like figures rappel down a building while spraying bright, glowing, colorful graffiti onto the monochrome walls.  
  * **Critique & Application:** This logline version is a powerful tool.  
    1. **It isolates the core idea:** "Ninjas spray color onto a B\&W world." If the AI can't generate a coherent image from this, it will definitely fail with the more complex version. This makes it an excellent **debugging** tool.  
    2. **It is untethered.** Without the specific material tokens, the base model is free to "dream" more broadly about what "graffiti" or a "black and white city" might look like. This can produce unexpected and creatively exciting **starting points**.  
    3. **It provides a path forward.** Once we have a logline generation that we like, we can use it as a foundation. Our next step would be to "build it back up" by re-introducing our other techniques: adding the detailed material lexicon, using synonymic language, or applying advanced aesthetic controls to pull the raw idea back into the specific, high-fidelity "Graffito" universe.  
  *   
* 

