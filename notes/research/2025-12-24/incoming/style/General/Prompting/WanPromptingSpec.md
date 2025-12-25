# Technical Specification: Wan Video Model (V2)

### **1. Purpose and Scope**

This document is the definitive technical specification for generating prompts for the **Wan Video Model**. It serves as the official vocabulary and syntax library referenced by the `Creative Prompting Protocol`. Its rules are non-negotiable for ensuring prompt achievability and performance.

### **2. Core Principles**

1.  **Amplify, Do Not Invent:** Faithfully translate user intent. Inject cinematic techniques or styles from this guide only when directly justified by the user's request.
2.  **Ensure Achievability:** All prompts must be scoped to a single, coherent shot. Ambitious concepts must be broken down into individual, achievable moments.
3.  **Adhere to Structure:** All prompts must be constructed using the `Prompt Architecture` and `Technical Vocabulary` defined below.

---

### **3. Prompt Architecture: The Principle of Narrative Priority**

The structure of a prompt is not fixed; it must be adapted to serve the narrative emphasis of the shot. The most important element—the reason the shot exists—should be described first to dedicate the most attention and detail to it.

#### **Pattern A: The Establishing Shot (Setting-First)**
Use when the primary goal is to establish a location, mood, or atmosphere. The environment is the main character of the shot.
*   **Order of Operations:** `[Setting & Atmosphere]` + `[Aesthetic Control & Stylization]` + `[Subject & Action]`
*   **Example Structure:** "A rain-slicked, neon-lit alley at midnight, cinematic film noir style, high contrast lighting. A lone man in a trench coat, his back to the camera, walks slowly away."

#### **Pattern B: The Interaction Shot (Character-First)**
Use when the primary goal is to capture a character's performance, emotion, or a specific interaction. The human element is the most important part of the shot.
*   **Order of Operations:** `[Subject & Action]` + `[Setting & Atmosphere]` + `[Aesthetic Control & Stylization]`
*   **Example Structure:** "A man with downcast eyes and slumped shoulders drops a photograph. He is in a dimly lit, sparsely furnished room. Close-up shot, soft lighting, somber mood."

#### **Pattern C: The Insert Shot (Object-First)**
Use when the primary goal is to focus on a specific object or detail that is critical to the story.
*   **Order of Operations:** `[Focal Object/Detail]` + `[Supporting Context & Aesthetics]`
*   **Example Structure:** "An antique silver locket lies open on a wooden table, revealing a faded photograph inside. Extreme close-up shot, practical lighting from a nearby candle, creating long shadows."

---

### **4. Technical Vocabulary**

This section contains the approved keywords for controlling the visual output of the Wan model.

#### **4.1 Light Source**

| Term | Recommended Keywords |
| :--- | :--- |
| **Sunny Lighting** | `Sunny lighting`, `edge lighting`, `low-contrast`, `warm colors`, `soft lighting`, `side lighting`, `day time` |
| **Artificial Lighting** | `Artificial lighting`, `edge lighting`, `desaturated colors`, `warm colors` |
| **Moonlighting** | `Moonlighting` |
| **Practical Lighting**| `Practical lighting`, `underlighting`, `high contrast lighting`, `night time` |
| **Firelighting** | `Firelighting` |
| **Fluorescent** | `Fluorescent lighting`, `cool colors`, `grayish-blue hue` |
| **Overcast Lighting**| `Overcast lighting`, `soft lighting`, `low contrast`, `desaturated colors`, `cool colors` |
| **Mixed Lighting** | `Mixed lighting`, `high contrast`, `mixed colors`, `cool colors` |

#### **4.2 Shot Size, Angle & Composition**

| Term | Recommended Keywords |
| :--- | :--- |
| **Extreme Close-up**| `Extreme close-up shot` |
| **Close-up** | `Close-up shot` |
| **Medium Shot** | `Medium shot`, `medium close-up shot` |
| **Wide Shot** | `Wide shot`, `medium wide shot`, `wide-angle lens` |
| **Establishing Shot**| `Establishing shot`, `extreme wide shot` |
| **High Angle** | `High angle shot`, `top-down shot` |
| **Low Angle** | `Low angle shot` |
| **Dutch Angle** | `Dutch angle shot` |
| **Over-the-Shoulder**| `Over-the-shoulder shot` |
| **Center Composition**| `Center composition`, `symmetrical composition` |
| **Weighted Comp.** | `Balanced composition`, `left-weighted composition`, `right-weighted composition` |

#### **4.3 Camera Movement**

| Term | Recommended Keywords |
| :--- | :--- |
| **Static** | `Static shot`, `fixed shot` |
| **Push In / Pull Back** | `Camera pushes in`, `camera pulls back`, `dolly in` |
| **Pan / Tilt** | `Camera pans right`, `camera pans left`, `camera tilts up`, `camera tilts down` |
| **Tracking** | `Tracking shot`, `follow-cam perspective` |
| **Handheld Feel** | `Handheld camera`, `motion blur` |

*(This section would continue with any other approved keyword tables, such as for specific visual styles or effects, all following this clean, term-keyword format.)*