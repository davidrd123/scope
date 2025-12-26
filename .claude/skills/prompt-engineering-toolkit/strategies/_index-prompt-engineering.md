\#\#\# \*\*MASTER SKILL INDEX: The Prompt Engineering Toolkit (New Build)\*\*

\*This document contains a directory of 18 specialized prompt engineering skills, organized by category. When the user provides feedback or a creative challenge, analyze their request to first identify the relevant category, then match their problem to the \`Trigger\` condition of a specific skill. State the name of the recommended skill and request the contents of its associated \`File\` for detailed instructions.\*

\---  
\#\#\# \*\*Category: Core Writing Principles\*\*  
\*(Fundamental strategies for constructing effective prompts from scratch.)\*

\*\*Skill: Mastering the "Attention Budget" & Front-Loading the Narrative\*\*  
\*   \*\*File:\*\* \`pws1\_attention\_budget.md\`  
\*   \*\*Trigger:\*\* When the generated output ignores critical elements, has a confused composition, or fails to execute the main action, especially if the key action was described late in the prompt.  
\*   \*\*Summary:\*\* Ensures all critical elements (subject, action, setting) are established in a concise "logline" at the absolute beginning of a prompt. This gives the AI a clear blueprint before it spends its "attention budget" on finer details.

\*\*Skill: Managing "Conceptual Distance" & Focusing on a Singular Idea\*\*  
\*   \*\*File:\*\* \`pws2\_conceptual\_distance.md\`  
\*   \*\*Trigger:\*\* When a prompt for a novel or Out-of-Distribution (OOD) idea fails, producing a generic or incoherent result. This often happens when asking for too many new things at once.  
\*   \*\*Summary:\*\* Dictates that every prompt should focus on a single core concept. For a highly novel (OOD) idea (like a new character or effect), the other prompt elements (setting, action) must be kept simple and familiar to focus the AI's attention on the one difficult task.

\*\*Skill: Generalization Strategies for OOD & In-Domain Prompts\*\*  
\*   \*\*File:\*\* \`pws2-1\_generalization\_strategies.md\`  
\*   \*\*Trigger:\*\* When a prompt for a new, Out-of-Distribution (OOD) subject (e.g., a cabin) is being "overpowered" by the model's training, resulting in an output that resembles a known subject (e.g., a tenement building).  
\*   \*\*Summary:\*\* To render new (OOD) subjects, this skill involves deliberately avoiding the core, trained lexicon and using synonymic or parsimonious language instead. This prevents the model's strong biases from overriding the new concept and encourages stylistic generalization.

\*\*Skill: Strategic Simplification (The "Clear to a Kid Reader" Principle)\*\*  
\*   \*\*File:\*\* \`pws3\_strategic\_simplification.md\`  
\*   \*\*Trigger:\*\* When a prompt is overly long or detailed, yet the output is weak, confused, or ignores key instructions.  
\*   \*\*Summary:\*\* Improves prompt effectiveness by trimming redundant language, over-description of non-essential elements, and overly granular details. It focuses on maximum impact with the minimum number of tokens, ensuring clarity directs the AI's attention budget.

\*\*Skill: The Aesthetic Control Component\*\*  
\*   \*\*File:\*\* \`pws4-2\_aesthetic\_control.md\`  
\*   \*\*Trigger:\*\* When a prompt needs specific cinematic direction for camera, lighting, or photographic style.  
\*   \*\*Summary:\*\* This skill defines a key component of the prompt architecture, focusing on cinematography. It uses established cinematic terminology for camera movement (e.g., 'dolly out'), lighting (e.g., 'chiaroscuro'), and color grading to give the AI clear, professional-grade visual instructions.

\*\*Skill: The Stylization Component (The "Thematic Capstone")\*\*  
\*   \*\*File:\*\* \`pws4-1\_stylization\_component.md\`  
\*   \*\*Trigger:\*\* When a prompt's technical details are correct, but the output lacks a clear mood, feeling, or artistic point of view.  
\*   \*\*Summary:\*\* This skill defines a final, critical component of the prompt architecture. It involves adding a single "Thematic Capstone" sentence at the end of a prompt to describe the intended emotional atmosphere or artistic impression, providing the AI with the crucial "why" behind the scene.

\*\*Skill: The "Graffito" Lexicon (Material & Motion Translation)\*\*  
\*   \*\*File:\*\* \`pws5\_lexicon.md\`  
\*   \*\*Trigger:\*\* When the output looks too realistic or generic, and has lost the specific, handcrafted, stop-motion aesthetic of the project.  
\*   \*\*Summary:\*\* The cornerstone of stylistic integrity. This skill requires describing all subjects, scenes, and effects not as they are in reality, but as tangible props built from a specific, established "arts-and-crafts" material and motion lexicon to reliably trigger the desired style.

\*\*Skill: Enforcing Character Consistency\*\*  
\*   \*\*File:\*\* \`pws6\_character\_consistency.md\`  
\*   \*\*Trigger:\*\* When a main character is rendered inconsistently, off-model, or in the wrong attire.  
\*   \*\*Summary:\*\* Guarantees character consistency by including the full, canonical visual description in the prompt whenever a main character appears. For non-canonical attire, it uses explicit description and, if necessary, negation of the default costume.

\*\*Skill: Directing Performance (Emotion \+ Physicality)\*\*  
\*   \*\*File:\*\* \`pws7\_directing\_performance.md\`  
\*   \*\*Trigger:\*\* When a character's performance feels lifeless, generic, or the emotion isn't "reading" correctly.  
\*   \*\*Summary:\*\* Directs character performance by leading with a primary emotional state (e.g., 'sad,' 'panicked') and then anchoring it with specific, complementary physical puppet-like actions (e.g., 'slumped shoulders,' 'flailing limbs') to create a stylistically authentic performance.

\*\*Skill: Active Props & Tangible VFX\*\*  
\*   \*\*File:\*\* \`pws8\_active\_props.md\`  
\*   \*\*Trigger:\*\* When props feel like static set dressing or visual effects (e.g., magic, water) feel generic, out-of-style, or lack impact.  
\*   \*\*Summary:\*\* Creates a more dynamic world by describing characters \*actively interacting\* with props. For VFX, it mandates describing them as a tangible, two-stage materialization process (e.g., a liquid state of 'cellophane strips' that solidifies into 'beads') to ground them in the handcrafted reality.

\*\*Skill: Prompting via Technical Analogy (Advanced Vocabulary)\*\*  
\*   \*\*File:\*\* \`pws9\_technical\_analogy.md\`  
\*   \*\*Trigger:\*\* When trying to achieve a highly specific or complex visual effect, form, or style that is difficult to describe with everyday language.  
\*   \*\*Summary:\*\* Achieves complex results efficiently by borrowing precise, niche vocabulary from other professional fields (e.g., 'mycelium network' for organic growth, 'decoupage' for a layered paper style, 'Dutch angle' for a disorienting camera). This taps into dense visual concepts the AI already understands.

\---  
\#\#\# \*\*Category: Advanced Revision & Creative Strategies\*\*  
\*(Techniques for debugging, refining, or creatively reimagining existing prompts.)\*

\*\*Skill: The Logline Approach (Core Concept Distillation)\*\*  
\*   \*\*File:\*\* \`rev1\_logline\_approach.md\`  
\*   \*\*Trigger:\*\* When a complex prompt is failing and the core concept needs to be debugged, or when seeking fresh, unexpected interpretations of an idea.  
\*   \*\*Summary:\*\* A debugging tool that involves stripping a complex prompt down to a single, filmmaking-style logline, free of any special lexicon. This tests the base model's understanding of the core idea in isolation.

\*\*Skill: The Synonymic Nudge (For Creative Generalization)\*\*  
\*   \*\*File:\*\* \`rev2\_synonymic\_nudge.md\`  
\*   \*\*Trigger:\*\* When the model is "overfitting" and producing repetitive visuals, or when a frame is visually cluttered and lacks a clear focus.  
\*   \*\*Summary:\*\* Solves overfitting and forces visual variety by swapping specific, trained keywords (e.g., \`PAINTED CARDBOARD\`) with their synonymic equivalents (e.g., \`sculpted paper pulp\`). This encourages the model to apply its style to new forms.

\*\*Skill: The Poetic Reimagining (Exploring Subtext)\*\*  
\*   \*\*File:\*\* \`rev3\_poetic\_reimagining.md\`  
\*   \*\*Trigger:\*\* When a literal description of a scene fails to capture the desired emotional subtext, or when a creative block requires a high-concept, artistically ambitious approach.  
\*   \*\*Summary:\*\* A creative tool that involves creating a new prompt that is a poetic or metaphorical representation of a scene's \*feeling\*, rather than a literal description of its action (e.g., prompting for 'a city screaming' instead of 'a car nearly hits a boy').

\---  
\#\#\# \*\*Category: Specialized \`Img2Vid\` Techniques\*\*  
\*(A specific workflow for prompts that use a source image.)\*

\*\*Skill: The "Img2Vid Attention Budget" (Parsimonious Action Prompt)\*\*  
\*   \*\*File:\*\* \`img2vid\_parsimonious\_prompt.md\`  
\*   \*\*Trigger:\*\* In an \`img2vid\` workflow, when a foundational command (like 'the camera is static') is being ignored in favor of overly descriptive details.  
\*   \*\*Summary:\*\* For \`img2vid\`, this skill dictates that the prompt should be "parsimonious" (lean), trusting the input image to establish the scene. The text prompt should focus almost exclusively on the desired \*action\* or \*change\*, giving simple, critical instructions maximum signal strength.

\*\*Skill: The "Enforcement Sentence" \- Surgical Style in Parsimonious Prompts\*\*  
\*   \*\*File:\*\* \`img2vid\_enforcement\_sentence.md\`  
\*   \*\*Trigger:\*\* In an \`img2vid\` workflow, a parsimonious prompt successfully creates the basic action but the \*quality\* of the motion is poor (e.g., a "head turn" looks like a flat 2D stretch).  
\*   \*\*Summary:\*\* A refinement technique where, instead of overhauling a lean prompt with detail, you inject a single, concise "Enforcement Sentence" that bundles all necessary stylistic keywords to fix one specific visual problem (e.g., 'Their head turns are executed with a rotoscoped quality...').

\*\*Skill: The "Parsimony-to-Enforcement" Pipeline (MASTER)\*\*  
\*   \*\*File:\*\* \`img2vid\_master\_pipeline.md\`  
\*   \*\*Trigger:\*\* When approaching any new \`img2vid\` task and in need of a reliable, step-by-step workflow.  
\*   \*\*Summary:\*\* The master \`img2vid\` workflow: 1\) Start with a parsimonious prompt to establish the core action and absolute rules. 2\) Diagnose the output for a single visual flaw. 3\) Apply surgical enforcement by injecting a targeted "Enforcement Sentence" to solve that one specific problem.

\---  
\#\#\# \*\*Category: Advanced Conceptual Tools\*\*  
\*(High-level strategies and discovery processes.)\*

\*\*Skill: Cross-Lingual Prompt Enhancement\*\*  
\*   \*\*File:\*\* \`advanced\_cross\_lingual.md\`  
\*   \*\*Trigger:\*\* When a meticulously crafted English prompt is still failing to achieve a highly complex or nuanced Out-of-Distribution (OOD) result.  
\*   \*\*Summary:\*\* An advanced refinement technique that involves translating a "gold-standard" English prompt into a language the model may have a more "native" understanding of (e.g., Chinese). This can potentially bypass linguistic ambiguities to achieve a deeper level of conceptual fidelity.

