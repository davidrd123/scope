# AI Previz Machine

> **tl;dr:** Switch the visual language of a scene with a keypress. Iterate while you think, not after. Running on Blackwell GPUs with 2.5x baseline performance.

---

## The Vision

A quick iterative machine for instant previsualization of permuting story.

You're watching a scene generate in real-time. You don't like the mood — press a key, and the whole visual language shifts. Woodblock print style. Classic cel animation. Gritty noir. The scene keeps flowing, but now it *feels* different.

That's what we built.

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

**Demo concept:** One scene, three visual languages, music-synced — switching between them in real-time.

---

## What's Working (Built in 4 Days)

| Feature | What It Feels Like |
|---------|-------------------|
| **Real-time style switching** | Press a key, the visual language changes — same scene, completely different feel |
| **Soft cuts** | Smooth transitions between styles instead of jarring jumps (emerged from interactive use) |
| **TUI director console** | Navigate playlists, step through scenes, switch styles — all keyboard-driven |
| **Blackwell performance** | 2.5x speedup on B200/B300 — fast enough to iterate while you think |

The TUI lets you move through a scene from different stylistic perspectives in real-time. We started with one style, didn't like it, switched — and discovered that smooth transitions between styles (soft cuts) felt better than hard switches. That interaction pattern emerged from the speed.

---

## The Performance Story (Scope Contribution)

Spot instances pushed us to B300 when B200 availability was tight. Blessing in disguise — it forced us to solve Blackwell compatibility broadly.

**The problem:** B300 was running at ~8.8 FPS on the default stack. Barely interactive. You'd wait for frames instead of iterating.

**What we learned:**
- The slowdown wasn't just one thing — runtime stack, decode behavior, kernel selection all mattered
- Attention was only part of the picture; the whole pipeline needed tuning
- Getting next-gen hardware to perform requires understanding the full stack

**What we did:**
- Custom kernels for the attention pattern Scope uses (KV-cache bias)
- Runtime stack tuned for Blackwell architecture
- Removed slow paths in the video encode pipeline

**Results:**

| | Before | After |
|--------|--------|-------|
| Blackwell FPS | ~8.8 | ~22–23 |
| Attention latency | ~0.9ms | ~0.4ms |

The work generalizes to B200/B300 — anywhere Blackwell runs. This is upstreamable to Scope: next-gen hardware support that benefits everyone.

---

## What's Coming

**Editorial intelligence** — A "Director Brain" that thinks like an editor. Scene direction implies shot types (action → zooms/whip pans, dialogue → shot/reverse). Notes loop for iterative refinement. (Patrick's been developing concepts here.)

**Audio/music sync** — Music drives generation parameters via OSC. Beat-aligned transitions, intensity mapping. Even just a chill-hop render of a scene as proof of concept. (John's exploring Tidal Cycles integration.)

**Branching & timeline** — Fork into variations at any point. Compare branches side by side. Timeline scrubbing with hardware controllers.

---

## Collaboration Interests

Already exploring with the team, but would love cohort input on:

| Area | What We're Thinking | Want to Explore |
|------|---------------------|-----------------|
| **VLM feedback** | AI watches output, suggests prompt adjustments | Anyone doing vision-language model work? |
| **Voice control** | Speak prompts instead of typing | Real-time speech-to-text integration |
| **Mobile camera** | Phone as reference input (VACE conditioning) | Anyone using live camera feeds? |
| **Hardware controllers** | Stream Deck, MIDI for tactile control | Physical interface design |

Happy to share code, notes, or just talk approaches.
