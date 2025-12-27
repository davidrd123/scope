# Tidal + Scope Experiments

> Status: Exploratory
> Date: 2025-12-27

Ideas to prototype once the basic OSC wiring is in place. Each experiment is a direction to pull on — some will lead nowhere, some might open new territory.

## How to Read This (Taxonomy + Gates)

These ideas span multiple *classes* of work. Labeling them helps avoid mixing latency-sensitive realtime work with “cool but offline” exploration.

### Experiment Classes

- **Music→Video control:** extract intent/features from music and steer video.
- **Video→Music control:** extract features from video and steer music.
- **Video→Video conditioning:** feed video-derived signals back into video generation (e.g., VACE conditioning / self-conditioning loops).
- **New abstractions / languages:** change *how* we specify transitions/composition (often offline first).

### Latency Sensitivity (Rule of Thumb)

- **Low:** fine to run offline or with >1s delay.
- **Medium:** OK with smoothing; tolerate ~100–500ms control latency.
- **High:** requires tight realtime feel (beat/gesture-aligned); hard to do over network or with heavy inference.

### “Gates” We Use Per Experiment

For each experiment, keep a small card so we can decide quickly whether to continue:
- **MVP prereq:** what must already exist (usually: Phase 1 bridge wiring + logging).
- **Success looks like:** what you can observe/measure.
- **Stop condition:** when to call it and document the lesson.

---

## 1. Pattern Algebra → Embedding Space

**Why interesting:** Tidal's operators (`slow`, `fast`, `jux`, `stack`, `every`) are composable transformations. What if they operated directly on prompt embeddings instead of audio events?

**Experiment card:**
- **Class:** New abstractions / languages (video-side)
- **Latency sensitivity:** Low (offline / non-realtime is fine)
- **MVP prereq:** stable embedding transitions exist (LERP/SLERP) and there’s a hook point to apply an embedding transform.
- **Success looks like:** a short clip where one operator (“slow/fast/stack”) has an obvious, controllable visual effect.
- **Stop condition:** operators are either imperceptible, wildly unstable, or require too much bespoke “semantic direction” tuning to be useful.

**Mapping sketch:**

| Tidal Operator | Embedding Operation |
|----------------|---------------------|
| `slow 2` | Double transition duration (already supported) |
| `fast 2` | Halve transition duration |
| `jux f` | Apply `f` to a mirrored copy, blend both |
| `stack [a, b]` | Weighted sum of embeddings |
| `every n f` | Apply `f` every nth cycle |
| `# speed x` | Scale interpolation rate |
| `rev` | Reverse the interpolation direction? |
| `palindrome` | A→B→A within one cycle |

**How to prototype:**
1. Start with `jux` — it's visually intuitive (left/right split in audio)
2. **Speculative baseline (likely nonsense, but cheap):** take embedding `e`, create `e_mirror = -e`, then blend `0.5 * e + 0.5 * e_mirror`.
3. **More grounded alternative:** choose (or learn) a direction `d` (e.g. "more noir", "more neon") and do `e ± α·d` for a controlled contrast.
4. See what "visual jux" looks like — does it create meaningful contrast without destroying coherence?

**Concept Sliders (more grounded alternative):**

Rather than raw embedding arithmetic, **Concept Sliders** (ECCV 2024) provide pre-learned directional vectors:
- Low-rank LoRA adaptors trained to find directions for concepts (age, style, lighting, expression)
- **Compose additively**, support negative weights
- Real-time audio mapping is trivial: `slider_strength = bass_amplitude * 0.8`
- **SliderSpace** (2025) enables zero-shot extraction without additional training

This may be cleaner than trying to derive meaningful directions from scratch. See: `github.com/rohitgandikota/sliders`

**Open questions:**
- Which embedding dimensions are semantically meaningful to negate?
- Does CLIP embedding space have a "mirror" that makes visual sense?
- Could we learn a `jux` transform that produces complementary visuals?
- Are Concept Sliders compatible with Scope's embedding pipeline?

---

## 2. Bidirectional Feedback (Video → Audio)

**Why interesting:** Tidal supports OSC controller input (`/ctrl`). The video system could send features back, creating a feedback loop where what Tidal "sees" influences what it plays.

**Experiment card:**
- **Class:** Video→Music control (feedback)
- **Latency sensitivity:** Medium (video generation is slower than audio; smoothing is mandatory)
- **MVP prereq:** Phase 1 bridge is working (video can send intent to music reliably) and we can compute at least one cheap video feature.
- **Success looks like:** the music responds to a stable video-derived signal without stalls or runaway oscillation.
- **Stop condition:** feedback feels laggy/unusable even after smoothing, or the loop is unstable (runaway) in normal use.

**How to prototype:**
1. Extract simple frame features from generated video:
   - Dominant hue (0-360)
   - Motion magnitude (frame diff)
   - Edge density (Sobel filter sum)
   - Brightness
2. Send as OSC to Tidal's controller input (`/ctrl`) (exact port is configurable; don’t bake a port number into the design):
   ```
   /ctrl hue 0.7
   /ctrl motion 0.3
   /ctrl edges 0.8
   ```
3. Tidal pattern references these:
   ```haskell
   d1 $ sound "bd*4" # speed (1 + cF 0 "motion")
   ```

**Feedback loop potential:**
- Fast motion → faster tempo → more visual change → more motion (runaway)
- Need dampening or the loop explodes
- Or embrace the chaos — let it find strange attractors

**Smoothing with mass-spring-damper:**

For audio-reactive parameters, use physically intuitive dynamics (TouchDesigner's Lag CHOP implements this):
```
ζ < 1: underdamped (overshoot, oscillate) — bouncy, reactive
ζ = 1: critically damped (fastest without overshoot) — snappy but stable
ζ > 1: overdamped (slow approach) — smooth, laggy
```
The damping ratio ζ is the key tuning knob. Start with ζ ≈ 0.7 for responsive-but-stable feel.

**Open questions:**
- What's the latency budget? `/ctrl` is real-time but video generation isn't
- Should feedback be smoothed (rolling average) or instantaneous?
- What happens when the loop converges vs diverges?
- What damping ratio feels "musical"?

---

## 3. Tension Accumulator

**Why interesting:** Current event-driven approach sends discrete "prompt X at time T". But music has *trajectory* — tension building, releasing. The video system should know where we are in that arc.

**Experiment card:**
- **Class:** Music→Video control (derived features)
- **Latency sensitivity:** Medium (depends on window size; usually seconds-scale is OK)
- **MVP prereq:** we can observe the music stream in a machine-readable way (OSC event proxy or audio feature tap) and we can send a small control to Scope (reserved key or REST).
- **Success looks like:** “tension” rises/falls in a way that matches what you hear, and video responds predictably (without requiring perfect sync).
- **Stop condition:** the signal is too noisy or too delayed to be musically meaningful, or it encourages hacks that compromise video quality.

**Architecture:**
```
Tidal OSC events → Accumulator → Scope
                      ↓
              Rolling state:
              - note_density (notes per cycle, 4-cycle window)
              - pitch_register (weighted average MIDI note)
              - harmonic_density (unique pitch classes)
              - rhythmic_complexity (entropy of onset pattern)
                      ↓
              tension: 0.0 → 1.0
```

**Tension Ribbons (MorpheuS — more musically grounded):**

If MIDI/pitch info is available, the **Spiral Array** model (Herremans & Chew) provides richer tension metrics:
- **Cloud diameter**: dispersion of notes in tonal space (captures dissonance)
- **Cloud momentum**: movement of pitch sets over time
- **Tensile strain**: distance between local and global tonal context

These produce "tension ribbons" — continuous curves that track harmonic tension more precisely than note density alone. Computable per time window for real-time use. See: `dorienherremans.com/tension`

**How to prototype:**
1. Python script between Tidal and Scope
2. Receives OSC from the Tidal→SuperDirt event stream (recommended: make the accumulator a proxy by having Tidal send to it, and it forwards to SuperDirt). Avoid assuming a fixed port in the doc; SuperDirt ports are commonly configured.
3. Maintains rolling stats
4. Forwards to Scope with added `tension` field in envelope
5. Scope uses tension to modulate:
   - Denoising steps (more tension → fewer steps → rawer output?)
   - Blend toward "intense" prompt variant
   - Color temperature shift
   - Motion intensity

**Open questions:**
- What's the right time window? 4 cycles? 8?
- Should tension be auto-derived or manually controllable?
- Can we detect "drop" moments (sudden tension release)?

---

## 4. Arousal-Valence Bridge

**Why interesting:** Instead of mapping low-level audio features to visuals, use emotional space as the bridge. Arousal (energy) and valence (mood) are well-studied in music psychology.

**Experiment card:**
- **Class:** Music→Video control (audio ML features)
- **Latency sensitivity:** Medium–High (depends on model + windowing)
- **MVP prereq:** reliable audio capture from the Tidal/SuperDirt output and a place to run inference (music box is fine).
- **Success looks like:** the A/V corner prompts produce recognizably different “moods”, and the mapping tracks music changes without feeling random.
- **Stop condition:** inference latency/jitter makes it feel disconnected, or the predictions don’t correlate with perceived arousal/valence for your material.

**The mapping:**

| Arousal | Valence | Visual Character |
|---------|---------|------------------|
| High | High | Bright, saturated, fast, expanding |
| High | Low | Dark, saturated, aggressive, contracting |
| Low | High | Soft, warm, slow, gentle motion |
| Low | Low | Muted, cold, minimal motion |

**How to prototype:**
1. Use Essentia's pre-trained models (arousal R²=0.88, valence R²=0.74)
2. Run on audio stream from SuperCollider/SuperDirt
3. Output continuous arousal/valence to Scope
4. Scope blends between 4 "corner" prompts based on position in A-V space

**Corner prompts example:**
```
high_arousal_high_valence = "explosive celebration, bright fireworks, golden light"
high_arousal_low_valence = "violent storm, dark chaos, red lightning"
low_arousal_high_valence = "peaceful meadow, soft sunset, gentle breeze"
low_arousal_low_valence = "foggy void, muted grays, stillness"
```

**Open questions:**
- Essentia runs on audio — how to get SuperDirt's output stream?
- Latency of arousal prediction?
- Is 2D space (A-V) enough or do we need more dimensions?

---

## 5. Visual Pattern Language

**Why interesting:** What if video had its own pattern language that mirrored Tidal's syntax? Not controlling audio, but composing visual transformations with the same combinatorial power.

**Experiment card:**
- **Class:** New abstractions / languages
- **Latency sensitivity:** Low initially (design); High later if you try to run it live
- **MVP prereq:** none beyond clear definitions; do not start by building a parser.
- **Success looks like:** a minimal vocabulary (3–5 ops) that composes cleanly and yields reusable visual “moves”.
- **Stop condition:** it turns into a general scene graph / programming language, or becomes tightly coupled to one model’s quirks.

**Speculative syntax:**
```haskell
-- Hypothetical visual patterns
v1 $ scene "nebula" # density 0.8 # morph (slow 4 sine)

v1 $ stack [
  scene "forest" # layer "foreground",
  scene "mountains" # layer "background" # blur 0.3
]

v1 $ every 4 (# style "glitch") $ scene "cityscape"

v1 $ jux (# hue (+0.5)) $ scene "abstract"  -- complementary colors
```

**How to prototype:**
1. Don't build a parser — just define what the operations *mean*
2. Implement a few transforms:
   - `# density x` → affects denoising steps or CFG
   - `# morph f` → time-varying interpolation curve
   - `# hue x` → post-process color shift (or embedding manipulation)
3. See if the combinatorial approach yields interesting results

**Open questions:**
- Should visual patterns sync to Tidal's clock or run independently?
- What's the equivalent of `sound "bd sd"` — a visual "sample"?
- How do visual patterns layer? Alpha compositing? Latent blending?

---

## 6. MCP as Integration Protocol

**Why interesting:** OSC is low-level (just floats and strings). MCP (Model Context Protocol) is LLM-aware — it can carry structured context, tool calls, semantic queries. The Epidemic Sound MCP server shows this working for music discovery.

**Experiment card:**
- **Class:** New protocol / orchestration (semantic control plane)
- **Latency sensitivity:** High (LLM and tool calls are not “beat realtime” by default)
- **MVP prereq:** the OSC/HTTP bridge path works reliably; MCP must be additive, not a replacement.
- **Success looks like:** richer commands reduce manual glue work without introducing stalls or unsafe “eval arbitrary code” behavior.
- **Stop condition:** latency/complexity outweighs the value, or it creates an unsafe surface area that’s hard to sandbox.

**Potential:**
- Tidal sends structured intent: `{"mood": "building tension", "genre_hint": "ambient", "next_transition": "drop in 4 bars"}`
- Scope's LLM layer interprets this, generates appropriate prompt
- Richer than raw OSC, more semantic

**How to prototype:**
1. Run a local MCP server alongside Scope
2. Tidal (or a bridge) sends MCP tool calls instead of raw OSC
3. MCP server translates to Scope control messages
4. Bonus: conversation history gives context accumulation for free

**Open questions:**
- Is MCP overkill for real-time? Latency?
- How does Tidal call MCP? Through a bridge?
- What tools would the MCP server expose?

---

## 7. VACE Visual Conditioning (Streaming + Self-Conditioning)

**Why interesting:** VACE already has a per-chunk conditioning input (`vace_input_frames`). That opens two powerful directions for Tidal experiments:

- **Live conditioning:** webcam/NDI/capture video conditions the generation while Tidal controls prompt/transition.
- **Self-conditioning loop:** generated frames (or extracted features) become conditioning for the *next* chunk, creating a controllable feedback system.

**Experiment card:**
- **Class:** Video→Video conditioning (optionally combined with Music→Video control)
- **Latency sensitivity:** Medium (generation is chunked; any extra analysis/transform needs smoothing and budget awareness)
- **MVP prereq:** a VACE-enabled run mode + a way to provide a per-chunk video input stream.
- **Success looks like:** conditioning influence is visible and controllable without locking the scene into “sticky” artifacts or collapsing to noise.
- **Stop condition:** the loop is unstable (runaway), over-constrains the model (no novelty), or costs too much FPS for your target use.

**How to prototype:**
1. Start with **live conditioning** (no feedback yet): supply a stable camera feed, keep Tidal patterns simple, and verify the conditioning “takes”.
2. Move to **feature conditioning**: derive a 3‑channel map from frames (edges, depth, motion, segmentation) and feed that instead of raw RGB.
3. Try **self-conditioning** cautiously:
   - Use the *decoded output* as next-chunk conditioning (optionally blurred / edge-extracted).
   - Add dampening: rolling average, thresholding, masks, or periodic "break" events to prevent lock-in.

**NCA stability techniques (from Neural Cellular Automata):**

Generative feedback operates at the edge of chaos. NCA research provides battle-tested stabilization:
- **50% stochastic dropout** on update vectors — prevents lock-in
- **Alpha threshold** — cells below 0.1 considered "dead" and zeroed (prevents noise accumulation)
- **Sample pool training** — mix fresh seeds with previous states (our "periodic break event" is this)

These translate directly to diffusion self-conditioning. The "break event" idea (Tidal triggers `every 16 (trigger "vace_break")`) is essentially sample pool injection.

**Error-recycling (for long sessions):**

From Stable Video Infinity (2025): training assumes clean data, but inference uses self-generated (error-prone) outputs. **Error-recycling fine-tuning** uses the model's own errors as supervisory signals, enabling minute-scale coherent generation without drift. Relevant if we ever fine-tune for self-conditioning.

**Notes / constraints to keep in mind:**
- In Scope’s current server behavior, when VACE is enabled the video input routes to `vace_input_frames` (conditioning) rather than the normal V2V input path; treat VACE as a distinct mode rather than “one more knob”.
- The “interesting version” for Tidal is often **not** raw RGB feedback, but **structured features** (edges/depth/motion) that behave like a controllable visual instrument.

---

## Priority / Energy

Roughly ordered by "can hack on this afternoon" to "needs more thought":

1. **Bidirectional Feedback** — just needs OSC sender in Scope, receiver config in Tidal
2. **Tension Accumulator** — standalone Python script, minimal integration
3. **VACE Visual Conditioning** — compelling, but requires VACE mode + video input stream
4. **Pattern Algebra (jux)** — single experiment in embedding space
5. **Arousal-Valence** — needs Essentia setup, audio routing
6. **Visual Pattern Language** — design exercise first, implementation later
7. **MCP Integration** — architectural, needs more design

---

## Techniques / Building Blocks

Cross-cutting techniques that apply to multiple experiments:

### Interpolation
- **SLERP** (spherical linear interpolation) for hyperspherical latent spaces — essential for Gaussian priors
- **Norm-preserving interpolation (NIN)** solves SLERP degeneracies where norms drift outside training distribution
- For 3+ concepts, use **convex hull methods** to prevent degenerate outputs

### Smoothing / Dynamics
- **Mass-spring-damper**: damping ratio ζ controls feel (underdamped=bouncy, critically damped=snappy, overdamped=smooth)
- **ADSR envelopes**: attack time = transition speed into visual state
- **LFOs**: sine for smooth pulsing, saw for ramps

### Stability (for feedback loops)
- **Stochastic dropout** (50%) on update vectors
- **Alpha/magnitude thresholds** to zero out noise accumulation
- **Sample pool mixing** (fresh seeds + previous states)
- **Error-recycling fine-tuning** for drift prevention in long sessions

### Semantic Control
- **Concept Sliders**: pre-learned LoRA directions for style/lighting/expression
- **SpLiCE**: decompose CLIP embeddings into 10-30 interpretable concepts
- **Arousal-Valence**: 2D emotional space from Essentia models
- **Tension Ribbons**: cloud diameter/momentum/strain from Spiral Array

### Narrative Arcs
Six canonical story shapes (Reagan et al.): rags-to-riches, tragedy, man-in-a-hole, Icarus, Cinderella, Oedipus. Could serve as templates for video generation arcs, with real-time deviation as "surprise" parameter.

---

## See Also

- `tidal-cycles-integration.md` — implementation spec for basic wiring
- `claude01.md` — ecosystem survey (Hydra, Strudel, OSC patterns, prior art)
- `claude02.md` — frontier techniques (concept sliders, NCA, tension models, StreamDiffusion)
- Essentia arousal/valence models: https://essentia.upf.edu/models.html
- Tidal `/ctrl` docs: https://tidalcycles.org/docs/configuration/MIDIOSC/osc
- Concept Sliders: https://github.com/rohitgandikota/sliders
- MorpheuS tension: https://dorienherremans.com/tension
- Neural Cellular Automata: https://distill.pub/2020/growing-ca/
