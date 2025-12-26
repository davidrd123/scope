# Daydream Scope Notes Index

> **Purpose:** Navigate this notes directory quickly
> **Updated:** 2025-12-25
> **Context:** Daydream Interactive AI Video Program (Dec 22 - Jan 9)

---

## Quick Start

| What you want | Where to go |
|---------------|-------------|
| Current session state | [FA4/SESSION-HANDOFF-2025-12-24.md](FA4/SESSION-HANDOFF-2025-12-24.md) |
| Project pitch (vision) | [daydream/cohort-pitch.md](daydream/cohort-pitch.md) |
| Project overview (internal map) | [PROJECT-OVERVIEW.md](PROJECT-OVERVIEW.md) |
| Latest cohort-style update | [daydream/cohort-update-2025-12-25.md](daydream/cohort-update-2025-12-25.md) |
| Cohort update template | [daydream/cohort-update-template.md](daydream/cohort-update-template.md) |
| Performance optimization journey | [FA4/kernel-dev-log.md](FA4/kernel-dev-log.md) |
| Perf deep dive (shareable) | [FA4/docs/kernel-optimization-guide.md](FA4/docs/kernel-optimization-guide.md) |
| Perf work map (timeline + where to record things) | [FA4/optimization-map.md](FA4/optimization-map.md) |
| Feature roadmap | [capability-roadmap.md](capability-roadmap.md) |
| What others are building | [ecosystem.md](ecosystem.md) |
| Pipeline architecture explainer | [../src/scope/core/pipelines/krea_realtime_video/docs/architecture-guide.md](../src/scope/core/pipelines/krea_realtime_video/docs/architecture-guide.md) |

---

## 1. Performance Optimization (FA4/)

The kernel optimization campaign - FlashAttention 4, Triton kernels, RoPE fusion.

### Core Documents

| File | Description |
|------|-------------|
| [FA4/kernel-dev-log.md](FA4/kernel-dev-log.md) | **The chronicle** - 1000+ line dev log tracking all optimization work |
| [FA4/optimization-map.md](FA4/optimization-map.md) | Timeline + “truth sources” + where to record experiments/resources |
| [FA4/docs/kernel-optimization-guide.md](FA4/docs/kernel-optimization-guide.md) | Shareable writeup: profiling → KV-bias attention → FA4/CUTE `score_mod` → RoPE → FPS |
| [FA4/README.md](FA4/README.md) | Navigation guide for FA4 directory |
| [FA4/SESSION-HANDOFF-2025-12-24.md](FA4/SESSION-HANDOFF-2025-12-24.md) | Latest session state, what's working, what's blocked |

### B300 (Blackwell SM103) Investigation

| File | Description |
|------|-------------|
| [FA4/b300/session-state.md](FA4/b300/session-state.md) | Current B300 state, env setup, profiling results |
| [FA4/b300/investigation-runbook.md](FA4/b300/investigation-runbook.md) | Systematic debugging guide with 5 hypotheses |
| [FA4/b300/investigation.md](FA4/b300/investigation.md) | Investigation findings |
| [FA4/b300/fa4-patches.md](FA4/b300/fa4-patches.md) | SM103 patches for nvidia-cutlass-dsl |
| [FA4/b300/setup-guide.md](FA4/b300/setup-guide.md) | B300 environment setup |
| [FA4/b300/optimization-vision.md](FA4/b300/optimization-vision.md) | Strategic performance roadmap |
| [FA4/b300/development-plan.md](FA4/b300/development-plan.md) | Concrete B300 plan (priorities + milestones) |
| [FA4/b300/experiments.md](FA4/b300/experiments.md) | One-change experiment cards (hypothesis → command → result → lesson) |

### B200 (Blackwell SM100) Notes

| File | Description |
|------|-------------|
| [FA4/b200/session-state.md](FA4/b200/session-state.md) | Current B200 state + repro commands |
| [FA4/b200/development-plan.md](FA4/b200/development-plan.md) | Concrete B200 plan (priorities + milestones) |
| [FA4/b200/experiments.md](FA4/b200/experiments.md) | One-change experiment cards |
| [FA4/b200/bringup-log.md](FA4/b200/bringup-log.md) | Historical bring-up notes |

### FA4/CuTe Explainers (Learning Track + Phase 3 Playbooks)

| File | Description |
|------|-------------|
| [FA4/explainers/README.md](FA4/explainers/README.md) | Index of all explainers (Phase 1–3) |
| [FA4/explainers/13-optimization-bootstrapping.md](FA4/explainers/13-optimization-bootstrapping.md) | Phase 3: optimization playbook |
| [FA4/explainers/14-blog-patterns-to-experiments.md](FA4/explainers/14-blog-patterns-to-experiments.md) | Phase 3: blog patterns → experiment cards |

### RoPE Optimization

| File | Description |
|------|-------------|
| [FA4/rope/optimization.md](FA4/rope/optimization.md) | RoPE optimization plan and phases |
| [FA4/rope/phase3-triton-rope.md](FA4/rope/phase3-triton-rope.md) | Triton RoPE kernel development |
| [FA4/rope/phase3-step2.md](FA4/rope/phase3-step2.md) | Phase 3 Step 2 details |
| [FA4/rope/phase3-step3.md](FA4/rope/phase3-step3.md) | Phase 3 Step 3 details |

### FA4/CUTE Integration

| File | Description |
|------|-------------|
| [FA4/fa4/phase4-score-mod.md](FA4/fa4/phase4-score-mod.md) | FA4 with CUTE DSL score_mod integration |

### DeepResearch (AI-assisted investigation)

| File | Description |
|------|-------------|
| [FA4/DeepResearch/summary.md](FA4/DeepResearch/summary.md) | Synthesized recommendations |
| [FA4/DeepResearch/MSU_chat.md](FA4/DeepResearch/MSU_chat.md) | Kernel project breakdown + sequencing |

---

## 2. Feature Development

### Roadmaps

| File | Description |
|------|-------------|
| [capability-roadmap.md](capability-roadmap.md) | Feature priorities (Style Layer, VACE-14B, Context Editing) |
| [realtime-roadmap.md](realtime-roadmap.md) | 8-phase roadmap with API specs |
| [ROADMAP.md](ROADMAP.md) | High-level project direction |

### Development Logs

| File | Description |
|------|-------------|
| [realtime-devlog.md](realtime-devlog.md) | Implementation timeline and decisions |
| [TODO-next-session.md](TODO-next-session.md) | Outstanding work items |

### Plans

| File | Description |
|------|-------------|
| [plans/phase6-prompt-compilation.md](plans/phase6-prompt-compilation.md) | Style Layer design |
| [plans/tui-director-console.md](plans/tui-director-console.md) | TUI Director Console design |
| [plans/realtime-control-plane-tdd.md](plans/realtime-control-plane-tdd.md) | Control plane TDD |

### VACE-14B Integration

| File | Description |
|------|-------------|
| [vace-14b-integration/plan.md](vace-14b-integration/plan.md) | VACE-14B integration plan |
| [vace-14b-integration/work-log.md](vace-14b-integration/work-log.md) | Integration work log |

---

## 3. Architecture & Reference

| File | Description |
|------|-------------|
| [realtime_video_architecture.md](realtime_video_architecture.md) | Pipeline architecture design |
| [reference/cli-implementation.md](reference/cli-implementation.md) | CLI code from specs |
| [reference/context-editing-code.md](reference/context-editing-code.md) | Gemini integration patterns |
| [console-commands.md](console-commands.md) | Command reference |

---

## 4. Ecosystem & Community

| File | Description |
|------|-------------|
| [ecosystem.md](ecosystem.md) | Community projects + our multi-stage vision |
| [daydream/interactive-ai-video-program.md](daydream/interactive-ai-video-program.md) | Program info (Dec 22 - Jan 9) |
| [daydream/cohort-pitch.md](daydream/cohort-pitch.md) | Shareable project vision + demo narrative |
| [daydream/cohort-update-2025-12-25.md](daydream/cohort-update-2025-12-25.md) | Shareable progress update (Dec 25) |
| [daydream/cohort-update-template.md](daydream/cohort-update-template.md) | 1-page template for future cohort updates |

---

## 5. Krea Pipeline Specifics

| File | Description |
|------|-------------|
| [krea/realtime.md](krea/realtime.md) | Krea Realtime notes |
| [krea/input-mode-fix.md](krea/input-mode-fix.md) | Input mode fixes |
| [krea/blog-krea-realtime-14b.md](krea/blog-krea-realtime-14b.md) | Krea Realtime 14B blog notes |

---

## 6. Offline Rendering

| File | Description |
|------|-------------|
| [offline/timeline-renderer.md](offline/timeline-renderer.md) | Timeline renderer design |
| [offline/render-timeline-guide.md](offline/render-timeline-guide.md) | Render timeline usage guide |
| [offline/render-tuning.md](offline/render-tuning.md) | Render tuning notes |

---

## 7. Issues & Known Problems

| File | Description |
|------|-------------|
| [issues/multi-lora-hot-switching.md](issues/multi-lora-hot-switching.md) | LoRA hot-switching design |

---

## 8. Research (by date)

### 2025-12-24 (Latest)

| File | Description |
|------|-------------|
| [research/2025-12-24/style-layer-revision-notes.md](research/2025-12-24/style-layer-revision-notes.md) | Style Layer 7-pattern design |
| [research/2025-12-24/VACE-integration.md](research/2025-12-24/VACE-integration.md) | VACE integration notes |

**Incoming (raw, not processed):**
- `research/2025-12-24/incoming/perf/` - Performance blogs and references
- `research/2025-12-24/incoming/skills/RAT/` - RAT character skill pillars
- `research/2025-12-24/incoming/rest_endpoint/` - REST API feedback

### 2025-12-23

| File | Description |
|------|-------------|
| [research/2025-12-23/realtime-architecture-v1.1-review.md](research/2025-12-23/realtime-architecture-v1.1-review.md) | Architecture review |
| [research/2025-12-23/krea-resolution-attention-backends.md](research/2025-12-23/krea-resolution-attention-backends.md) | Resolution vs attention backend analysis |

### Process

| File | Description |
|------|-------------|
| [research/PROCESS.md](research/PROCESS.md) | Raw → distilled → integrated workflow |

---

## Key Achievements (as of Dec 25)

### Performance
- **FA4/CUTE score_mod**: 1.89x faster attention kernel (0.54ms vs 1.02ms Triton)
- **RoPE Phase 3**: Fused kernel, fixed regression, hit 20.2 FPS
- **B200**: Optimized to ~20 FPS at 320x576
- **B300**: 15 FPS achieved with cu130 runtime + FlashAttention (was 8.8 FPS)

### Features
- **REST API** (Phase 5): Complete - `/api/v1/realtime/` endpoints + CLI
- **Style Layer** (Phase 6a): In progress
- **VACE-14B** (Phase 6b): Ready to implement

---

## Directory Structure

```
notes/
├── FA4/                    # Kernel optimization work
│   ├── b300/               # B300 investigation
│   ├── b200/               # B200 bringup
│   ├── rope/               # RoPE optimization
│   ├── fa4/                # FA4/CUTE integration
│   └── DeepResearch/       # AI-assisted research
├── research/               # Dated research & specs
│   └── 2025-12-*/          # By date
├── plans/                  # Future development plans
├── issues/                 # Known problems
├── offline/                # Offline rendering
├── krea/                   # Krea pipeline notes
├── reference/              # Code patterns & guides
├── vace-14b-integration/   # VACE work
└── daydream/               # Program info
```
