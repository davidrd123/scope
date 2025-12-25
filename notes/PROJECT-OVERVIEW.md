# Project Overview: Real-Time Video Generation Instrument

> **Internal Reference Document**
> **Updated:** 2025-12-25
> **Context:** Daydream Interactive AI Video Program (Dec 22 - Jan 9)

---

## The Big Picture

We're building an **interactive instrument for real-time AI video generation** - not just a pipeline that turns prompts into video, but a creative tool where you can:

- **Iterate in real-time**: Step through generation, adjust prompts, see results immediately
- **Branch and explore**: Pause, fork into multiple directions, compare, pick the best
- **Separate content from style**: Same story beats render differently with different LoRAs
- **Control at multiple levels**: From high-level narrative beats down to raw prompt tokens

```
┌─────────────────────────────────────────────────────────────────┐
│                     THE INSTRUMENT                               │
│                                                                  │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│   │  WorldState │───▶│   Style     │───▶│  Pipeline   │───▶ Video│
│   │  (content)  │    │  Compiler   │    │  (engine)   │        │
│   └─────────────┘    └─────────────┘    └─────────────┘        │
│         ▲                                      │                │
│         │            ┌─────────────┐           │                │
│         └────────────│ Control Bus │◀──────────┘                │
│                      │  (events)   │                            │
│                      └─────────────┘                            │
│                            ▲                                    │
│         ┌──────────────────┼──────────────────┐                │
│         │                  │                  │                │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐         │
│   │ Dev Console │   │  Branch     │   │   Voice/    │         │
│   │ (step mode) │   │  Graph      │   │   VLM       │         │
│   └─────────────┘   └─────────────┘   └─────────────┘         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Two Tracks, One Project

### Track 1: The Instrument (Architecture)

**What:** The interaction system that makes video generation controllable and creative.

**Core primitives:**

| Primitive | Purpose | Status |
|-----------|---------|--------|
| **WorldState** | Domain-agnostic truth (beats, emotions, props) | Designed |
| **StyleManifest** | Per-LoRA prompt vocabulary | Designed |
| **PromptCompiler** | WorldState + Style → Pipeline prompts | Designed |
| **ControlBus** | Event queue with chunk-boundary semantics | Designed |
| **BranchGraph** | DAG of snapshots for fork/resume | Designed |
| **FrameBus** | Ring buffer for frame distribution | Designed |

**Key insight:** The pipeline produces 3-frame chunks. Build the interaction system on chunk boundaries - that's where you can safely pause, branch, and modify state.

### Track 2: The Engine (Performance)

**What:** Making the pipeline fast enough for real-time interaction.

**Achievements:**

| Optimization | Result | Details |
|--------------|--------|---------|
| **FA4/CUTE score_mod** | 1.89x faster attention | 0.54ms vs 1.02ms Triton |
| **RoPE fusion** | 1.26x faster RoPE | Triton kernel, no float64 |
| **B200 tuning** | 20 FPS at 320x576 | Up from ~15 FPS |
| **B300 environment** | 15 FPS with cu130 | Was 8.8 FPS |

**Key insight:** Profiling first. We found attention was only 27% of self-attention time - QKV projection and RoPE were bigger than expected. Optimization is guided by measurement, not assumptions.

### How They Connect

```
Track 1 (Instrument)          Track 2 (Engine)
─────────────────────         ────────────────────
WorldState
    ↓
StyleManifest
    ↓
PromptCompiler
    ↓
ControlState ──────────────▶ Pipeline ◀──── FA4/CUTE kernels
    ↓                              ◀──── RoPE fusion
ControlBus                         ◀──── Triton optimizations
    ↓
BranchGraph
```

The instrument defines *what* you can do (pause, branch, step, style-switch).
The engine determines *how fast* you can do it (20 FPS vs 8 FPS).

Without the engine work, you'd be iterating at 8 FPS - frustratingly slow.
Without the instrument work, you'd have a fast pipeline with no way to control it creatively.

---

## Current Status

### What's Working

| Component | Status | Notes |
|-----------|--------|-------|
| Pipeline (Krea 14B) | ✅ Working | 20 FPS on B200, 15 FPS on B300 |
| FA4 attention | ✅ Integrated | 1.89x faster kernel |
| RoPE optimization | ✅ Integrated | Triton fused kernel |
| REST API (basic) | ✅ Working | `/api/v1/realtime/` endpoints |
| CLI (`video-cli`) | ✅ Working | `prompt`, `step`, `frame` commands |
| Prompt playlist | ✅ Working | Navigate caption files |

### What's Designed (Not Built)

| Component | Status | Doc |
|-----------|--------|-----|
| WorldState | 📋 Designed | `realtime_video_architecture.md` |
| StyleManifest | 📋 Designed | `realtime_video_architecture.md` |
| PromptCompiler | 📋 Designed | `realtime_video_architecture.md` |
| ControlBus | 📋 Designed | `realtime_video_architecture.md` |
| BranchGraph | 📋 Designed | `realtime_video_architecture.md` |
| Full snapshot/restore | 📋 Designed | Needs pipeline.state integration |

### What's Blocked/Deferred

| Component | Status | Reason |
|-----------|--------|--------|
| VACE-14B integration | ⏸️ Ready | Waiting on Style Layer first |
| Context editing | ⏸️ Speculative | Needs validation |
| VLM feedback loop | ⏸️ Future | Phase 8 |

---

## The Build Path

### What We Had (Day 1)

- Krea Realtime pipeline (working, ~15 FPS)
- WebRTC streaming
- Basic prompt input

### What We Built (Days 1-4)

- Kernel profiling infrastructure
- FA4/CUTE integration (1.89x attention speedup)
- RoPE fusion (Triton kernel)
- B300 environment setup (cu130)
- REST API endpoints
- CLI tools

### What's Next (Days 5-16)

**Instrument Track:**

| Priority | Component | Effort | Impact |
|----------|-----------|--------|--------|
| 1 | Step mode polish | Low | Enables iteration |
| 2 | StyleManifest schema | Medium | Enables style switching |
| 3 | PromptCompiler | Medium | Connects World → Style |
| 4 | Snapshot/restore | Medium | Enables branching |
| 5 | BranchGraph | Medium | Full fork/resume |

**Engine Track:**

| Priority | Component | Effort | Impact |
|----------|-----------|--------|--------|
| 1 | torch.compile exploration | Medium | Potential gains |
| 2 | VAE optimization | High | Significant on B300 |
| 3 | Further profiling | Low | Find new targets |

---

## Architecture Deep Dive

### The Semantic Layer Stack

```
┌─────────────────────────────────────────────────────────────────┐
│ PRESENTATION LAYER                                               │
│   User choices, Dev Console, Voice input                        │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ WORLD LOGIC LAYER                                                │
│   Narrative Logic: beat patterns, arc templates                 │
│   Character State: emotions, motivations, knowledge             │
│   External State: props, locations, visibility                  │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ STYLE LOGIC LAYER                                                │
│   StyleManifest: per-LoRA vocabulary                            │
│   PromptCompiler: WorldState → prompts                          │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ RENDER LAYER                                                     │
│   Pipeline + LoRA + FA4 kernels                                 │
└─────────────────────────────────────────────────────────────────┘
```

### The Same Story, Different Styles

This is the key creative capability. Same WorldState, different visual output:

```
WorldState:
  current_action: "character enters menacingly"
  camera_intent: "low angle"
  beat: "tension_build"

            ↓ StyleManifest: Rudolph 1964
"rudolph1964, rankinbass, slow deliberate puppet walk cycle,
 felt texture, miniature set perspective, deep shadows"

            ↓ StyleManifest: TMNT Mutant Mayhem
"tmnt_mayhem, aggressive spray paint style, graffiti energy,
 street-level angle, neon rim lighting"

            ↓ StyleManifest: Rooster & Terry
"rat_lora, 2D animation smear frames, exaggerated silhouette,
 dramatic low camera, film noir shadows"
```

### Event-Driven Control

All control flows through the ControlBus with chunk-boundary semantics:

```
User action ──▶ ControlEvent ──▶ ControlBus ──▶ Apply at chunk boundary
                                     │
                                     ▼
                              ┌─────────────┐
                              │ Event Types │
                              ├─────────────┤
                              │ SET_PROMPT  │
                              │ SET_WORLD   │
                              │ PAUSE       │
                              │ STEP        │
                              │ FORK        │
                              │ RESTORE     │
                              └─────────────┘
```

This makes control:
- **Deterministic**: Events applied in consistent order
- **Debuggable**: Event history is stored
- **Replayable**: Given initial state + events, reproduce session

### Branching Model

```
                    ┌─────────────────────────────────────┐
                    │          Linear Generation          │
                    │  chunk 0 → chunk 1 → chunk 2 → ...  │
                    └─────────────────────────────────────┘
                                        │
                                    [PAUSE]
                                        │
                                        ▼
                    ┌─────────────────────────────────────┐
                    │           Snapshot Created          │
                    │  WorldState + ControlState + buffers│
                    └─────────────────────────────────────┘
                                        │
                           ┌────────────┼────────────┐
                           │            │            │
                        [FORK]       [FORK]       [FORK]
                           │            │            │
                           ▼            ▼            ▼
                    ┌──────────┐ ┌──────────┐ ┌──────────┐
                    │ Branch A │ │ Branch B │ │ Branch C │
                    │ +prompt  │ │ +seed    │ │ +tension │
                    └──────────┘ └──────────┘ └──────────┘
                           │            │            │
                      [rollout]    [rollout]    [rollout]
                           │            │            │
                           ▼            ▼            ▼
                    ┌──────────┐ ┌──────────┐ ┌──────────┐
                    │ Preview  │ │ Preview  │ │ Preview  │
                    │ 6 chunks │ │ 6 chunks │ │ 6 chunks │
                    └──────────┘ └──────────┘ └──────────┘
                                        │
                                   [SELECT B]
                                        │
                                        ▼
                    ┌─────────────────────────────────────┐
                    │       Continue from Branch B        │
                    └─────────────────────────────────────┘
```

---

## Key Files

### Architecture & Design

| File | Description |
|------|-------------|
| `notes/realtime_video_architecture.md` | Full architecture spec (1900+ lines) |
| `notes/capability-roadmap.md` | Feature priorities |
| `notes/realtime-roadmap.md` | 8-phase roadmap |

### Performance & Kernels

| File | Description |
|------|-------------|
| `notes/FA4/kernel-dev-log.md` | Full optimization chronicle |
| `notes/FA4/docs/kernel-optimization-guide.md` | Technical explainer |
| `notes/FA4/b300/session-state.md` | B300 environment setup |

### Code

| File | Description |
|------|-------------|
| `src/scope/core/pipelines/krea_realtime_video/` | Pipeline implementation |
| `src/scope/core/kernels/triton_attention.py` | Triton Kernel B |
| `src/scope/core/kernels/triton_rotary.py` | Triton RoPE |

---

## Integration Opportunities

Ideas from other cohort projects that could enhance this:

| From | Idea | How It Fits |
|------|------|-------------|
| **Frost Bytes** | OSC audio sync | OSC → WorldState.tension, StyleManifest params |
| **Dreamwalker** | Mobile camera input | V2V mode, VACE conditioning |
| **Narrative AI** | Claude as storyteller | VLM → WorldState updates, beat suggestions |
| **Community** | Real-time voice input | Voice → prompt, voice → character emotion |

---

## Open Questions

1. **How much narrative logic?** Keep beats simple (setup/escalation/payoff/reset) or build a real narrative engine?

2. **True KV cache serialization?** Current snapshots use context buffers + recompute. Full cache serialization is faster but heavier.

3. **Multi-LoRA hot-switching?** Current design assumes one LoRA at a time. Multi-LoRA needs blending strategy.

4. **VLM integration depth?** Use for suggestions only, or let it drive generation?

---

## Success Criteria (Jan 9)

**Minimum:**
- Step mode working smoothly
- At least one StyleManifest (e.g., Rudolph 1964)
- Basic snapshot/restore

**Stretch:**
- Full branching (fork, rollout, select)
- Multiple StyleManifests
- Voice input integration

**Demo-worthy:**
- Live session where you pause, branch, compare, continue
- Same scene rendered in multiple styles
- Real-time prompt iteration at 20 FPS
