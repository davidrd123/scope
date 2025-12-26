### **Prompt Revision Strategy: Creating Dynamic Motion with Sequential Actions**

**Lesson:** To combat the "slow-motion effect" and create more dynamic, naturally paced motion, a prompt should describe a **sequence of actions from the start to the end of the shot**, rather than a single, isolated action.

**When to Use This Strategy:**

* **To Fix "Slow Motion":** This is the primary tool to use when a generated video feels unnaturally slow, as if a single, simple action is being stretched out over the entire duration.  
* **To Add Kinetic Energy:** When a shot feels static or lacks energy, adding a sequence of small, secondary actions can make it feel more alive and dynamic.  
* **To Choreograph Complex Performance:** This technique is essential for creating a multi-part performance within a single shot, guiding the AI through a clear beginning, middle, and end.

**Rationale:** The AI model will always attempt to fill the duration of the video clip (e.g., 4 seconds) with the actions you provide. If you give it only one simple action ("he swings a sword"), it will often stretch that single action to last the full 4 seconds, resulting in an unnatural slow-motion effect. By providing a sequence of events ("he spins, swings the sword, stops, looks up, and a crow flies by"), you are giving the AI multiple distinct "narrative beats" to animate. This forces it to execute each part of the sequence more quickly, resulting in a more complex, engaging, and naturally paced final video.

---

### **Few-Shot Example (from community feedback):**

**The Goal:** A dynamic shot of a character performing an action.

* **INEFFECTIVE (Singular Action \- Risks Slow Motion):**  
  "...she swings her sword and looks at the building behind her."  
  * **Critique:** This prompt is high-risk for the "slow-motion" problem. The AI is given two simple, connected actions and may stretch the "swing" to last the entire duration, making it feel weak and un-dynamic.  
*   
* **EFFECTIVE (Sequential Actions \- Creates Dynamic Pace):**  
  "...she spins around to face away from the camera, swinging her sword. the camera pans up as she looks up at the building, sun rays shining around it. a crow flies from the right to the left."  
  * **Critique:** This is a much stronger and more reliable prompt. The key to getting good motion is to **describe all the things happening from start to end of the generation.** By breaking the action into a clear multi-part sequence (spin/swing \-\> camera tilts / character looks \-\> crow flies), we are giving the AI several distinct beats to animate within the same duration, forcing a quicker, more natural pace and resulting in a far more complex and dynamic final shot.

