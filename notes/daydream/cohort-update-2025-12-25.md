# Cohort Update — 2025-12-25

**Project:** Playable real-time AI video (Daydream Scope fork)  
**TL;DR:** I’m building a controllable, real-time video generation “instrument” by (1) making the 14B Krea realtime pipeline fast enough to be interactive and (2) adding a control layer (world state + style manifests + simple APIs) so it’s steerable mid-performance.

---

## Vision

Interactive video should feel like a game instrument: you “play” scene state (characters, mood, camera) and style (LoRAs / instruction sheets) live, and the model responds in real time without needing to stop and re-render.

---

## Progress

- **Performance:** Deep profiling + kernel work around KV-cache causal attention (FA4/CUTE `score_mod` for KV-bias) and RoPE cleanup/fusion.
- **B200 / B300 bring-up:** At canonical settings (`320x576`, 4 denoise steps, KV-bias `0.3`), we’ve seen ~`20 FPS` on B200 and ~`15 FPS` on B300 with an SM103-native runtime stack (Torch `2.9+cu130` / newer cuDNN).
- **Control surface:** REST endpoints for realtime control (`/api/v1/realtime/*`) plus scaffolding for a “Style Layer” (world state + prompt compilation + playlist-style direction).
- **Documentation:** Wrote an explainer for the Krea realtime pipeline and a “profiling → gains” kernel optimization guide.

---

## What I Learned

- **Amdahl’s law is ruthless:** speeding up the KV-bias attention kernel alone doesn’t double FPS; QKV GEMMs + RoPE + VAE decode can dominate depending on GPU + runtime.
- **Runtime stack matters on Blackwell:** on B300 (SM103), decode (conv3d/cuDNN) performance can swing dramatically depending on the CUDA/cuDNN bundle even when model code is unchanged.

---

## Next Up (now → Jan 9)

- **Make the “director controls” feel good:** tighten the Style Layer loop (world change → prompt compile → apply) and add a minimal set of performance-friendly knobs for live steering.
- **Share a short clip + control video:** a 20–30s “here’s how it responds to state/style changes” demo suitable for cohort updates.
- **VACE-14B direction:** decide whether to prototype a chained pipeline (14B stylized generation → 1.3B VACE reactive layer) or go straight into 14B VACE integration.

---

## Asks

- If you’ve shipped on B300/Blackwell: any gotchas on **Torch/cuDNN versions** that materially change conv3d / decode performance?
- If you’re doing interactive storytelling: what’s your “minimum viable” control set (camera / beats / prompts) that feels expressive without UI complexity?

---

## Deep Dives (if you want details)

- Notes map: `notes/NOTES-INDEX.md`
- Perf deep dive (profiling → FA4): `notes/FA4/docs/kernel-optimization-guide.md`
- Krea realtime architecture: `src/scope/core/pipelines/krea_realtime_video/docs/architecture-guide.md`
- What others are building: `notes/ecosystem.md`
