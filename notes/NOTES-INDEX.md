# Daydream Scope Notes Index

> **Purpose:** Navigate this notes directory quickly
> **Updated:** 2025-12-26
> **Context:** Daydream Interactive AI Video Program (Dec 22 - Jan 9)

---

## ⚠️ Environment Note

**Claude Code version jump:** Attempted to pin at `0.2.62` but it auto-updated to `0.2.74`. Version pinning via `npm install -g @anthropic-ai/claude-code@0.2.62` did not hold. Be aware of potential behavior/context changes.

---

## ⚠️ Incoming Material (~78 files - Needs Permanent Home)

**Location:** `research/2025-12-24/incoming/` (recount: `find notes/research/2025-12-24/incoming -type f | wc -l`)

### Performance (29 files) → FA4 docs
| Path | Contents |
|------|----------|
| `perf/blogs/` | 15 blog posts: torch-compile, flexattn, thunderkittens-blackwell, warp-specialization, verda-b200-b300, etc. |
| `perf/chat/` | 3 ChatGPT exports: CuTe DSL guide, chunk padding fix, B300 SM103 perf |
| `perf/ref/` | 7 reference docs: torchao-compat, SM103, CUDA 12.9/13.1, Blackwell arch |
| `perf/` | ACTIONABLE_ITEMS_SUMMARY.md, NVFP4.md, cuda_repoprompt_req.md |

### Style Assets (37 files) → guides/ or assets/
| Path | Contents |
|------|----------|
| `style/Akira/` | Prompting guide, captioning (v4 + lexicon), prompts (videodrome, FightClub chapters, 2001) |
| `style/Kaiju/` | Prompting guidelines (v5.4), captioning, prompts (alice wonderland, kant ufo) |
| `style/RAT/` | Prompting guidelines, captioning, prompts (wild, fury road, percy) |
| `style/TMNT/` | Prompting guidelines, captioning |
| `style/General/` | WanPromptingSpec.md, Creative Briefs, PCM interaction paradigm |

### Skills (5 files) → managed skill or daydream/
| Path | Contents |
|------|----------|
| `skills/RAT/` | Skill.md + Pillars I-IV (Constitution, Cast, Art Direction, Story Engine) |

### Specs (3 files) → plans/ or reference/
| Path | Contents |
|------|----------|
| Top-level | project_knowledge.md, context_editing_and_console_spec.md, interface_specifications.md |

### REST Endpoint (4 files) → close out or archive
| Path | Contents |
|------|----------|
| `rest_endpoint/` | testing_cmds.md, 5pro_rest_feedback.md, test images |

**Process:** See [research/PROCESS.md](research/PROCESS.md) for raw → distilled → integrated workflow.

---

## Quick Start

| What you want | Where to go |
|---------------|-------------|
| **Internal vision (why & where)** | [VISION.md](VISION.md) |
| Project session state | [session-state-2025-12-25.md](session-state-2025-12-25.md) |
| FA4 session handoff | [FA4/SESSION-HANDOFF-2025-12-24.md](FA4/SESSION-HANDOFF-2025-12-24.md) |
| B300 session state | [FA4/b300/session-state.md](FA4/b300/session-state.md) |
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

## Current Workstreams (Meta)

- **Perf profiling support (no core changes):** We’re actively improving measurement tooling to produce clearer data without touching the core pipeline (e.g. `scripts/profile_krea_pipeline_ops.py`, `scripts/profile_krea_pipeline_blocks.py`).
- **Feature/proposal consolidation:** Drafts and implementation candidates live under `notes/proposals/` (many were co-developed with Claude while iterating on requirements).

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
| [FA4/explainers/15-scope-to-fa4-call-path.md](FA4/explainers/15-scope-to-fa4-call-path.md) | End-to-end call path from TUI to kernel |
| [FA4/explainers/16-numerics-and-fp8.md](FA4/explainers/16-numerics-and-fp8.md) | KV-bias math, exp2, FP8 tradeoffs |
| [FA4/explainers/17-backend-selection-and-knobs.md](FA4/explainers/17-backend-selection-and-knobs.md) | Complete knobs reference |
| [FA4/explainers/18-debugging-cookbook.md](FA4/explainers/18-debugging-cookbook.md) | Symptom → cause → fix guide |

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

## 2. Concepts (Phase 2 Vision)

Future-facing ideas that build on Phase 1 infrastructure.

| File | Description |
|------|-------------|
| [concepts/narrative-engine.md](concepts/narrative-engine.md) | **Phase 2** - Trajectories, information topology, multi-agent stakeholders, intent/subtext layers |
| [concepts/prompt-engineering-workflow.md](concepts/prompt-engineering-workflow.md) | Loom-like branching for visual behavior R&D (prompt iteration, A/B testing) |
| [concepts/context-editing-spec.md](concepts/context-editing-spec.md) | Context editing mechanism |
| [concepts/creative-workflows.md](concepts/creative-workflows.md) | Creative workflow patterns |
| [concepts/prompt-sequences.md](concepts/prompt-sequences.md) | Prompt sequence design |
| [concepts/tui-director.md](concepts/tui-director.md) | TUI director concept |

---

## 3. Proposals (Ready to Implement)

Concrete implementation proposals with specs and checklists.

| File | Description | Status |
|------|-------------|--------|
| [proposals/style-swap-mode.md](proposals/style-swap-mode.md) | Instant LoRA switching via preload + runtime_peft | Ready |
| [proposals/vlm-integration.md](proposals/vlm-integration.md) | Gemini for frame analysis + image editing | Ready |
| [proposals/frame-buffer-scrubbing.md](proposals/frame-buffer-scrubbing.md) | Ring buffer for instant scrub/replay/branch preview | Ready |
| [proposals/server-side-session-recorder.md](proposals/server-side-session-recorder.md) | Record control events for offline re-render | Ready |
| [proposals/session-recording-timeline-export.md](proposals/session-recording-timeline-export.md) | Timeline export on recording stop | Implemented |
| [proposals/gemini-cookbook.md](proposals/gemini-cookbook.md) | Gemini integration patterns (from comfy_automation) | Reference |
| [proposals/ndi-pubsub-video-output.md](proposals/ndi-pubsub-video-output.md) | NDI network streaming | Ready |
| [proposals/tidal-cycles-integration.md](proposals/tidal-cycles-integration.md) | Music sync via OSC | Speculative |
| [proposals/multi-gpu-scaling.md](proposals/multi-gpu-scaling.md) | Pipeline parallelism for multi-GPU inference | Exploratory |
| [proposals/transition-prompts.md](proposals/transition-prompts.md) | Transition prompt syntax | Draft |

---

## 4. Feature Development

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
| [proposals/vace-14b-integration.md](proposals/vace-14b-integration.md) | VACE-14B integration plan |
| [proposals/vace-14b-integration/work-log.md](proposals/vace-14b-integration/work-log.md) | Integration work log |
| [proposals/vace-14b-integration/research/](proposals/vace-14b-integration/research/) | Architecture research |

---

## 5. Architecture & Reference

| File | Description |
|------|-------------|
| [realtime_video_architecture.md](realtime_video_architecture.md) | Pipeline architecture design |
| [reference/cli-implementation.md](reference/cli-implementation.md) | CLI code from specs |
| [reference/context-editing-code.md](reference/context-editing-code.md) | Gemini integration patterns |
| [console-commands.md](console-commands.md) | Command reference |

---

## 6. Ecosystem & Community

| File | Description |
|------|-------------|
| [ecosystem.md](ecosystem.md) | Community projects + our multi-stage vision |
| [daydream/interactive-ai-video-program.md](daydream/interactive-ai-video-program.md) | Program info (Dec 22 - Jan 9) |
| [daydream/cohort-pitch.md](daydream/cohort-pitch.md) | Shareable project vision + demo narrative |
| [daydream/cohort-update-2025-12-25.md](daydream/cohort-update-2025-12-25.md) | Shareable progress update (Dec 25) |
| [daydream/cohort-update-template.md](daydream/cohort-update-template.md) | 1-page template for future cohort updates |

---

## 7. Krea Pipeline Specifics

| File | Description |
|------|-------------|
| [krea/realtime.md](krea/realtime.md) | Krea Realtime notes |
| [krea/input-mode-fix.md](krea/input-mode-fix.md) | Input mode fixes |
| [krea/blog-krea-realtime-14b.md](krea/blog-krea-realtime-14b.md) | Krea Realtime 14B blog notes |

---

## 8. Offline Rendering

| File | Description |
|------|-------------|
| [offline/timeline-renderer.md](offline/timeline-renderer.md) | Timeline renderer design |
| [offline/render-timeline-guide.md](offline/render-timeline-guide.md) | Render timeline usage guide |
| [offline/render-tuning.md](offline/render-tuning.md) | Render tuning notes |

---

## 9. Issues & Known Problems

| File | Description |
|------|-------------|
| [issues/multi-lora-hot-switching.md](issues/multi-lora-hot-switching.md) | LoRA hot-switching design |

---

## 10. Research (by date)

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

## Key Achievements (as of Dec 26)

### Performance
- **FA4/CUTE score_mod**: 1.89x faster attention kernel (0.54ms vs 1.02ms Triton)
- **RoPE Phase 3**: Fused kernel, fixed regression, hit 20.2 FPS
- **B200**: Optimized to ~20 FPS at 320x576
- **B300**: 15 FPS achieved with cu130 runtime + FlashAttention (was 8.8 FPS)

### Features
- **REST API** (Phase 5): Complete - `/api/v1/realtime/` endpoints + CLI
- **Style Layer** (Phase 6a): Complete (world/style endpoints + `video-cli world|style`)
- **VACE-14B**: Ready to implement

---

## Directory Structure

```
notes/
├── VISION.md               # Internal vision (living doc - why & where)
├── NOTES-INDEX.md          # This file (navigation)
├── capability-roadmap.md   # Implementation tracking (what's being built)
├── FA4/                    # Kernel optimization work
│   ├── b300/               # B300 investigation
│   ├── b200/               # B200 bringup
│   ├── rope/               # RoPE optimization
│   ├── fa4/                # FA4/CUTE integration
│   └── DeepResearch/       # AI-assisted research
├── concepts/               # Phase 2 vision docs (narrative engine, workflows)
├── proposals/              # Implementation proposals (ready to build)
│   └── vace-14b-integration/  # VACE-14B supporting materials
├── research/               # Dated research & specs
│   └── 2025-12-*/          # By date
├── plans/                  # Development plans
├── issues/                 # Known problems
├── offline/                # Offline rendering
├── krea/                   # Krea pipeline notes
├── reference/              # Code patterns & guides
└── daydream/               # Program info & cohort materials
```
