# Tidal + Scope Experiments

> Status: Exploratory
> Date: 2025-12-27

Ideas to prototype once the basic OSC wiring is in place. Each experiment is a direction to pull on — some will lead nowhere, some might open new territory.

---

## 1. Pattern Algebra → Embedding Space

**Why interesting:** Tidal's operators (`slow`, `fast`, `jux`, `stack`, `every`) are composable transformations. What if they operated directly on prompt embeddings instead of audio events?

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
2. Take embedding `e`, create `e_mirror = -e` (or negate specific principal components)
3. Blend: `0.5 * e + 0.5 * e_mirror`
4. See what "visual jux" looks like — does negation create meaningful contrast?

**Open questions:**
- Which embedding dimensions are semantically meaningful to negate?
- Does CLIP embedding space have a "mirror" that makes visual sense?
- Could we learn a `jux` transform that produces complementary visuals?

---

## 2. Bidirectional Feedback (Video → Audio)

**Why interesting:** Tidal has a `/ctrl` input channel (port 6010). The video system could send features back, creating a feedback loop where what Tidal "sees" influences what it plays.

**How to prototype:**
1. Extract simple frame features from generated video:
   - Dominant hue (0-360)
   - Motion magnitude (frame diff)
   - Edge density (Sobel filter sum)
   - Brightness
2. Send as OSC to Tidal's `/ctrl`:
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

**Open questions:**
- What's the latency budget? `/ctrl` is real-time but video generation isn't
- Should feedback be smoothed (rolling average) or instantaneous?
- What happens when the loop converges vs diverges?

---

## 3. Tension Accumulator

**Why interesting:** Current event-driven approach sends discrete "prompt X at time T". But music has *trajectory* — tension building, releasing. The video system should know where we are in that arc.

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

**How to prototype:**
1. Python script between Tidal and Scope
2. Receives OSC from Tidal (port 57120)
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

## Priority / Energy

Roughly ordered by "can hack on this afternoon" to "needs more thought":

1. **Bidirectional Feedback** — just needs OSC sender in Scope, receiver config in Tidal
2. **Tension Accumulator** — standalone Python script, minimal integration
3. **Pattern Algebra (jux)** — single experiment in embedding space
4. **Arousal-Valence** — needs Essentia setup, audio routing
5. **Visual Pattern Language** — design exercise first, implementation later
6. **MCP Integration** — architectural, needs more design

---

## See Also

- `tidal-cycles-integration.md` — implementation spec for basic wiring
- `claude01.md` — ecosystem research, prior art survey
- Essentia arousal/valence models: https://essentia.upf.edu/models.html
- Tidal `/ctrl` docs: https://tidalcycles.org/docs/configuration/MIDIOSC/osc
