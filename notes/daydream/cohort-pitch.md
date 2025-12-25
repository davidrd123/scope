# Interactive Real-Time Video Instrument

> **tl;dr:** Building a creative tool where you can pause AI video generation, branch into multiple directions, compare them, and continue from the best one. Plus: made it 2x faster by writing custom GPU kernels.

---

## What I'm Building

An **interactive instrument** for real-time AI video, not just a pipeline.

Think of it like a video game save system, but for AI generation:
- **Pause** at any point
- **Branch** into multiple variations (different prompts, seeds, styles)
- **Preview** each branch for a few seconds
- **Pick** the one you like and continue from there

```
Generating... → PAUSE → Fork 4 variations → Preview all → Pick best → Continue
```

Plus a **style system** that separates *what happens* from *how it looks*:

```
Same story:                     Different styles:
"character enters menacingly"   → Rankin-Bass stop-motion
                                → TMNT spray-paint graffiti
                                → Rooster & Terry 2D animation
```

---

## Demo: What It Looks Like

**Step Mode** (working now):
```bash
$ video-cli pause
$ video-cli prompt "a fox walking through snow, rankin-bass style"
$ video-cli step          # Generate 3 frames
$ video-cli step          # Generate 3 more
$ video-cli prompt "the fox looks up, surprised"
$ video-cli step
```

**Branching** (in progress):
```bash
$ video-cli pause
$ video-cli snapshot      # Save current state
$ video-cli fork --variations 4 --horizon 6
  # Generates 4 different continuations, 6 chunks each
  # Branch 1: original prompt
  # Branch 2: seed variation
  # Branch 3: higher tension
  # Branch 4: prompt variation
$ video-cli preview       # See all 4 previews
$ video-cli select 2      # Continue from branch 2
$ video-cli run
```

---

## The Performance Story

I got nerdsniped by a performance mystery and ended up learning GPU kernel development.

**The problem:** B300 (the newer GPU) was running at 8.8 FPS while B200 hit 20 FPS.

**The journey:**
1. Added profiling to find where time goes
2. Discovered attention kernel was only 27% of the bottleneck (not what I expected!)
3. Wrote custom Triton kernels
4. Integrated FlashAttention 4 with CUTE DSL (NVIDIA's tensor core library)
5. Found the real B300 issue was the CUDA runtime stack, not the kernels

**Results:**

| What | Before | After |
|------|--------|-------|
| Attention kernel | 1.02ms | 0.54ms (1.89x faster) |
| B200 FPS | ~15 | ~20 |
| B300 FPS | 8.8 | 15 (with cu130 runtime) |

I wrote up the full journey if anyone's curious about GPU kernel optimization: `notes/FA4/docs/kernel-optimization-guide.md`

---

## Architecture (For the Curious)

```
User Input ──▶ WorldState ──▶ StyleManifest ──▶ Pipeline ──▶ Frames
                   │              │                            │
                   │              │         ┌──────────────────┘
                   │              │         │
                   ▼              ▼         ▼
              "character      "rankin-bass   [3 frames]
               enters         stop-motion
               menacingly"    puppet walk"
```

**WorldState**: What's happening (actions, emotions, beats) - no visual style info
**StyleManifest**: How to render for a specific LoRA (vocabulary, triggers, camera language)
**Pipeline**: The optimized Krea 14B with FA4 kernels

The separation means I can describe a scene once and render it multiple ways.

---

## What I'd Love to Explore

### Integration Ideas

| Idea | What It Would Do |
|------|------------------|
| **Voice input** | Speak prompts instead of typing |
| **OSC/audio sync** | Music amplitude → generation parameters (like Frost Bytes!) |
| **VLM feedback** | Claude watches the output and suggests prompt adjustments |
| **Mobile camera** | Phone camera as VACE reference input (like Dreamwalker!) |

### Open Questions

1. **Narrative control**: How much should the system understand about story structure? I have a simple beat system (setup/escalation/payoff/reset) but could go deeper.

2. **Multi-LoRA blending**: Right now it's one style at a time. Has anyone experimented with blending multiple LoRAs in real-time?

3. **Latency budget**: For live performance, what's acceptable? I'm at ~50ms per frame, wondering if that's good enough.

---

## Current Status

**Working:**
- Pipeline at 20 FPS (B200) / 15 FPS (B300)
- Step mode (pause, generate one chunk, inspect)
- Prompt updates with transitions
- CLI tools

**In Progress:**
- StyleManifest schema
- Snapshot/restore with seamless continuation
- Branching UI

**Planned:**
- VLM feedback loop
- VACE-14B integration (reference image conditioning)

---

## Want to Collaborate?

I'm particularly interested in:

1. **Audio/music integration** - If you're doing anything with OSC or beat detection, I'd love to connect WorldState parameters to audio

2. **Voice input** - Has anyone set up real-time speech-to-text for prompt control?

3. **Mobile/camera input** - The VACE integration would let camera feeds drive the generation

4. **Narrative/storytelling** - If you're working with LLMs for story generation, there's a natural connection to the WorldState system

Happy to share code, notes, or just chat about approaches!

---

## Resources

| What | Where |
|------|-------|
| Architecture spec | `notes/realtime_video_architecture.md` |
| Kernel optimization guide | `notes/FA4/docs/kernel-optimization-guide.md` |
| Performance dev log | `notes/FA4/kernel-dev-log.md` |
| Project overview | `notes/PROJECT-OVERVIEW.md` |
