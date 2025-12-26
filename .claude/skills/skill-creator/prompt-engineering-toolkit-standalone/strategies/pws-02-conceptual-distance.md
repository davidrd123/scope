### **Prompt Writing Strategy 2: Managing "Conceptual Distance" & Focusing on a Singular Idea**

**Lesson:** Every prompt, whether for text-to-video or image-to-video, should be built around **one singular, clear core idea or concept.** The more "conceptually distant" or Out-of-Distribution (OOD) that core idea is from the LoRA's training data, the more of the AI's "Attention Budget" must be dedicated to it, and the simpler the other elements of the prompt must be.

**Rationale:** The AI is not a machine that executes a list of commands; it is an artist that interprets a creative brief. Asking it to simultaneously innovate on multiple, conceptually difficult fronts at once (e.g., a novel character *in* a novel location *performing* a novel action) will overload its creative process. This leads to "cross-talk" between ideas, stylistic dilution, or a failure to render the most important elements. By focusing each prompt on a single, clear goal, we give the AI the best possible chance to execute that one thing brilliantly.

---

#### **Principle A: Introducing a Novel (OOD) Subject**

**Goal:** Create a "Graffito"-style sci-fi robot, a concept that is very far from our 1970s Bronx training data.

* **INEFFECTIVE (Too Many OOD Concepts at Once):**  
  Graffito Mixed-Media Stop-Motion: In a surreal zero-gravity environment, a futuristic robot made of PAINTED CARDBOARD performs a complex breakdancing routine as the camera executes a rapid orbital arc around it.  
  * **Critique:** This prompt is asking for too many miracles at once. It has an OOD subject (robot), an OOD setting (zero-gravity), and an OOD action (breakdancing). The AI's "Attention Budget" is split three ways, and it will likely fail at all of them, producing a generic, incoherent result.  
*   
* **EFFECTIVE (Singular Focus on the OOD Subject):**  
  Graffito Mixed-Media Stop-Motion: \*\*A futuristic robot, built from a clunky, boxy assemblage of silver PAINTED CARDBOARD and painted tin, stands motionless in a simple, gritty alleyway.\*\* The robot's single, glowing red eye is a large sculpted bead. The scene is a standard, in-distribution "Graffito" alley. The motion is completely static. The camera is a clean, locked-off medium shot.  
  * **Critique:** This is successful because it dedicates nearly the entire "Attention Budget" to the single, difficult task of translating a "robot" into the "Graffito" style. By placing it in a familiar setting and keeping the action and camera simple, we give the AI the best chance to focus on the one thing that truly matters: the OOD subject.  
* 

---

#### **Principle B: Executing a Novel (OOD) Action or VFX**

**Goal:** Create a surreal, psychedelic shot of a cityscape melting. This is a novel VFX for our world.

* **INEFFECTIVE (Divided Focus):**  
  Graffito Mixed-Media Stop-Motion: A detailed cityscape diorama made of PAINTED CARDBOARD begins to melt. As it melts, a complex paper pigeon flies through the scene, and the camera performs a slow dolly out.  
  * **Critique:** The "melting" VFX is the star of the show, but its "Attention Budget" is being stolen by the complex action of the pigeon and the simultaneous camera move. This will likely result in a weak or poorly-rendered melting effect.  
*   
* **EFFECTIVE (Singular Focus on the OOD VFX):**  
  Graffito Mixed-Media Stop-Motion: \*\*A handcrafted cityscape diorama, its buildings sculpted from solid THICK IMPASTO PAINT, begins to melt and dissolve into a river of liquid color.\*\* First, the city is still and solid. Then, in a surreal stop-motion sequence, the sharp edges of the paint-buildings soften and droop, and the entire diorama slowly collapses into a single, swirling, abstract river of liquid pigment. The camera is a static high-angle wide shot, observing the dissolution.  
  * **Critique:** This works because the *only* thing happening in this prompt is the melt. The subject (the city) and the effect (the melting) are a single, unified idea. The camera is locked off, and there are no other characters or actions to distract the AI. Every token is dedicated to achieving that one, spectacular, singular effect.

