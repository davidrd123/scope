# Daydream Scope Ecosystem

> **Purpose:** Track projects, integrations, and opportunities in the Scope ecosystem
> **We are:** A fork extending Scope for custom LoRA workflows + multi-stage pipelines
> **Updated:** 2025-12-25

---

## Ecosystem Map

```
                    ┌─────────────────────────────────┐
                    │     Daydream Scope (upstream)   │
                    │  - Krea 14B (T2V)               │
                    │  - LongLive 1.3B (V2V)          │
                    │  - StreamDiffusionV2 (V2V)      │
                    │  - WebRTC + REST API            │
                    └───────────────┬─────────────────┘
                                    │
         ┌──────────────────────────┼──────────────────────────┐
         │                          │                          │
         ▼                          ▼                          ▼
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│   Our Fork      │      │   Frost Bytes   │      │   Dreamwalker   │
│   (this repo)   │      │   (UE5 VJ)      │      │ (Unity Mobile)  │
└─────────────────┘      └─────────────────┘      └─────────────────┘
```

---

## 1. Community Projects

### Frost Bytes (UE5 Virtual Production)

**Creator:** (Daydream program participant)
**Status:** MVP in progress
**Links:** Daydream showcase

**What it does:**
- Live-coded Tidal Cycles music performance
- Aurora Borealis visuals driven by AI
- OSC audio data → prompt synchronization
- UE5 as rendering/compositing layer

**Architecture:**
```
Tidal Cycles → SuperCollider → OSC → UE5
                                      ↓
                              Scope API (RunPod)
                                      ↓
                              WebRTC → UE5 texture
```

**Interesting insights:**
- Embraces video artifacts as artistic features
- Uses HDMI capture for code display overlay
- Latency-managed sync between audio and video

**Integration opportunities:**
- OSC → Style Layer could be a reusable pattern
- UE5 WebRTC ingestion code could be shared

---

### Dreamwalker (Unity Mobile)

**Creator:** jreinjr
**Status:** Active development
**Links:** https://github.com/jreinjr/Dreamwalker

**What it does:**
- Android app for real-time AI video
- Camera capture → Scope → display loop
- Supports VACE, LoRA, prompt control

**Architecture:**
```
Android Camera → Unity WebRTC Client
                        ↓
              Scope Backend (local/RunPod)
                        ↓
              Processed frames → Display
```

**Tech stack:**
- Unity 2022.3 LTS / Unity 6
- WebRTC bidirectional
- REST API for control
- URP rendering

**Features exposed:**
- Pipeline selection
- Noise scale / denoising steps
- Text prompts
- VACE toggle
- LoRA loading

**Integration opportunities:**
- Reference Unity WebRTC implementation
- Mobile-friendly API patterns
- VACE API surface validation

---

## 2. Scope Pipelines (Upstream)

### Krea Realtime (14B) — Our Focus

**Model:** Wan2.1-14B
**Mode:** Text-to-Video only
**Status:** We're optimizing (15 FPS achieved)

**Our extensions:**
- FA4 attention optimization
- Style Layer (Phase 6a)
- VACE-14B integration (Phase 6b planned)
- Custom LoRA support (R&T, etc.)

**Limitations:**
- T2V only (no camera/reference input yet)
- Needs VACE for V2V capability

---

### LongLive (1.3B)

**Model:** Wan2.1-1.3B
**Mode:** Video-to-Video (has VACE)
**Status:** Works upstream, we haven't modified

**Capabilities:**
- Reference image conditioning
- Camera input → styled output
- Lower quality than 14B but interactive

**Potential use:**
- Second stage in our chained pipeline
- Performative/reactive layer on top of 14B output

---

### StreamDiffusionV2 (1.3B)

**Model:** Wan2.1-1.3B
**Mode:** Video-to-Video (has VACE)
**Status:** Works upstream

**Similar to LongLive** — alternative V2V option for chained pipeline.

---

## 3. Parts We Haven't Touched Yet

### WebRTC Pub/Sub Architecture

**Location:** `lib/webrtc.py`, `src/scope/server/webrtc.py`
**Status:** Works, but we haven't verified multi-consumer

**Questions:**
- [ ] Can multiple consumers subscribe to same stream?
- [ ] What's the latency when chaining through second GPU?
- [ ] How to route 14B output → 1.3B input?

---

### OSC Integration

**Status:** Not in Scope core, but Frost Bytes uses it
**Opportunity:** Add OSC → parameter mapping

**Potential implementation:**
```python
# OSC server that maps to Style Layer
osc.map("/audio/amplitude", "world_state.intensity")
osc.map("/audio/frequency", "world_state.color_temp")
```

---

### Audio-Reactive Features

**Status:** Not implemented
**Opportunity:** Beat detection → prompt/parameter changes

**References:**
- Frost Bytes uses Tidal Cycles → OSC
- Could integrate with Style Layer's WorldState

---

## 4. Gemini-Mediated Video-to-Video

**Concept:** Use Gemini Flash as a translation layer to enable indirect V2V on 14B.

```
Video Input → Gemini Flash (vision) → Text Description → 14B T2V → Video Output
```

**Why this works:**
- 14B is T2V only, but Gemini can describe video in real-time
- The "lag" becomes a feature — call and response, not sync
- Enables dialogue between input and output (Tom → Jerry)
- Already proven: 2001 shots → Rankin-Bass Rudolph translation

**Performance architecture:**
```
Monitor A: Video Input          Monitor B: Generated Response
    │                                 ▲
    ▼                                 │
Gemini Flash ──────────────────▶ 14B + LoRAs
(describe action,                (generate in style
 camera, emotion)                 vocabulary)
```

**Translation vocabulary:**
- Action: "enters frame", "looks left", "reaches for"
- Camera: "close-up", "wide shot", "pan right"
- Emotion: "surprised", "scheming", "joyful"
- Material: Rankin-Bass stop-motion, clay animation, etc.

**Use cases:**
- Live performance: Performer on A, AI responds on B
- VJ workflow: Reference clips drive generated visuals
- Character dialogue: Tom actions → Jerry responses
- Style transfer: Any video → your LoRA's aesthetic

**Implementation notes:**
- Gemini client already scaffolded (`src/scope/realtime/gemini_client.py`)
- Could tie into Style Layer's WorldState
- Need to tune prompt templates for consistent translation

---

## 5. Our Multi-Stage Vision

### Option A: Chained Pipeline (14B → 1.3B)
```
GPU 1: 14B T2V ──▶ GPU 2: 1.3B V2V ──▶ Final Output
     │                   │
     │                   └── OSC/audio reactive
     └── Custom LoRAs, Style Layer
```

### Option B: Gemini-Mediated Dialogue
```
Video Input ──▶ Gemini Flash ──▶ 14B T2V ──▶ Response Output
(Monitor A)      (translate)      + LoRAs      (Monitor B)
                                     │
                                     └── Style Layer vocabulary
```

### Option C: Full Stack (Both)
```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT SOURCES                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │ Live Cam │  │ Clips    │  │ OSC/MIDI │  │ Text     │    │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘    │
└───────┼─────────────┼─────────────┼─────────────┼──────────┘
        │             │             │             │
        ▼             ▼             ▼             ▼
┌─────────────────────────────────────────────────────────────┐
│                 TRANSLATION LAYER                            │
│  Gemini Flash: video→text    Style Layer: state→prompt      │
│  OSC mappings: audio→params  WorldState: unified control    │
└───────────────────────────────┬─────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                   GPU 1: 14B T2V                             │
│  Krea Realtime + Custom LoRAs (R&T, Yeti, etc.)             │
│  FA4 optimized @ 15 FPS                                      │
└───────────────────────────────┬─────────────────────────────┘
                                │
                    WebRTC pub/sub
                                │
         ┌──────────────────────┼──────────────────────┐
         │                      │                      │
         ▼                      ▼                      ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  Monitor (you)  │  │ GPU 2: 1.3B V2V │  │ Recording/      │
│                 │  │ Reactive layer  │  │ Broadcast       │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

---

## 5. Collaboration Opportunities

| Project | What We Could Share | What We Could Learn |
|---------|---------------------|---------------------|
| Frost Bytes | Style Layer API, OSC patterns | UE5 integration, artistic artifact handling |
| Dreamwalker | VACE-14B when ready | Unity WebRTC client, mobile API patterns |
| Upstream Scope | Our optimizations (FA4, etc.) | New pipeline features |

---

## 6. Open Questions

1. **Multi-GPU chaining:** How to efficiently route 14B → 1.3B across GPUs?
2. **Latency budget:** What's acceptable for live performance?
3. **OSC standardization:** Common mappings for audio → visual params?
4. **VACE on 14B:** Will quality be noticeably better than 1.3B V2V?

---

## 7. Resources

- Daydream Scope repo: (upstream)
- Dreamwalker: https://github.com/jreinjr/Dreamwalker
- Tidal Cycles: https://tidalcycles.org/
- WebRTC in UE5: (Frost Bytes implementation)
