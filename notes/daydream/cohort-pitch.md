# AI Previz Machine

> **tl;dr:** Script goes in, edited sequence comes out — but the director stays in control. Pause, branch, try a different style, refine with notes, re-run. Plus: made Scope 2.5x faster on B300.

---

## The Vision

A quick iterative machine for instant previsualization of permuting story.

```
Script/Scene Direction
        ↓
   Director Brain (coverage planning)
        ↓
   Editing Logic (mood, pacing, shot order)
        ↓
   Prompt Sequence → Real-time Generation
        ↓
   Notes → Refine → Re-run
```

**The director stays in control.** Pause anywhere. Branch into variations. Switch visual styles. Refine with notes. Re-run the cut.

**Demo concept:** Take a scene, generate it in three different LoRA styles (Hidari, TMNT, Rankin-Bass), with music sync — all in real-time.

---

## What's Working (Built in 4 Days)

| Feature | Status |
|---------|--------|
| **B300 performance** | 8.8 → 22+ FPS (2.5x uplift, upstreamable to Scope) |
| **TUI director console** | Switch styles, navigate playlists, step through scenes |
| **Soft cuts** | Smooth transitions between prompts (our innovation) |
| **Style switching** | Same scene, different LoRAs — Hidari Akira is amazing |
| **FA4 + custom kernels** | KV-cache attention bias, Triton kernels, CUTE DSL |

The TUI already enables moving through a movie from different LoRA perspectives in real-time. Soft cuts emerged from this interactive exploration.

---

## The Performance Story (Scope Contribution)

B300 (Daydream's newer GPU) was running at 8.8 FPS — barely usable. B200 hit 20 FPS.

**What we found:**
- Attention was only 27% of the bottleneck
- QKV projections and RoPE were bigger than expected
- CUDA runtime stack issues on B300

**What we did:**
- Custom Triton kernels for attention
- FlashAttention 4 integration with CUTE DSL
- B300 environment fixes (cu130 runtime)

**Results:**

| Metric | Before | After |
|--------|--------|-------|
| B300 FPS | 8.8 | 22+ (2.5x) |
| KV-bias attention | 1.02ms | 0.54ms |

This is upstreamable to Scope — makes B300 actually viable for real-time work.

---

## The Team (Parallel Workstreams)

We're three people now, each pushing a layer:

| Person | Focus | Status |
|--------|-------|--------|
| **Dave** | TUI, real-time interaction, performance | Working — 4 days of solo velocity |
| **Patrick** | Editorial intelligence, script-to-prompt pipeline | Concepts ready, building |
| **John** | Audio/music sync, Tidal Cycles integration | Excited, starting |

Plus AI agents (Codex) maintaining the performance layer.

---

## What's Coming

### Editorial Intelligence (Patrick)

A "Director Brain" that thinks like an editor:

```
Script → Coverage Plan → Editing Logic → Prompt Sequence
```

- Scene direction implies shot types (action → zooms/whip pans, dialogue → shot/reverse)
- Editing logic library (markdown guides for mood, pacing, tension)
- Notes loop for iterative refinement

### Audio/Music Sync (John)

Music drives generation parameters:

- Tidal Cycles or DAW → OSC → video intensity, tension, pacing
- Even just a chill-hop render of a scene as proof of concept
- Beat-aligned transitions

### Branching & Timeline (Dave)

The "playable" layer:

- Fork into variations at any point
- Compare branches side by side
- Timeline scrubbing with hardware controllers

---

## Collaboration Interests

Already exploring with the team, but would love cohort input on:

| Area | What We're Thinking | Want to Explore |
|------|---------------------|-----------------|
| **VLM feedback** | AI watches output, suggests prompt adjustments | Anyone doing vision-language model work? |
| **Voice control** | Speak prompts instead of typing | Real-time speech-to-text integration |
| **Mobile camera** | Phone as reference input (VACE conditioning) | Anyone using live camera feeds? |
| **Hardware controllers** | Stream Deck, MIDI for tactile control | Physical interface design |

Happy to share code, notes, kernel optimizations, or just talk approaches.

---

## Background

**Content library:**
- 217 LoRAs on HuggingFace from 4 years of generation work
- Trained styles: Rankin-Bass, TMNT Mutant Mayhem, Rooster & Terry, Kaiju, Hidari

**Technical depth:**
- Kernel-level understanding of the render pipeline
- FA4, CUTE DSL, Triton, B300/Blackwell optimization
- Prompt engineering framework that actually works

**Team experience:**
- Animation/VFX production pipelines
- Live performance and music technology
- Real-time systems and game development

---

## Resources

| What | Where |
|------|-------|
| Kernel optimization guide | `notes/FA4/docs/kernel-optimization-guide.md` |
| Editorial intelligence concept | `notes/concepts/editorial-intelligence.md` |
| Tidal integration proposal | `notes/proposals/tidal-cycles-integration.md` |
| Hardware control surface | `notes/proposals/hardware-control-surface.md` |
| Capability roadmap (10 features) | `notes/capability-roadmap.md` |
