# FA4 Kernel Optimization Notes

This directory tracks the journey of optimizing attention kernels for the Krea Realtime Video pipeline.

## Where to Start

| If you want to... | Start here |
|-------------------|------------|
| Get the map (timeline + “what goes where”) | `optimization-map.md` |
| Get the shareable overview | `docs/kernel-optimization-guide.md` |
| Understand the full journey | `kernel-dev-log.md` |
| Continue from last session | `SESSION-HANDOFF-*.md` (latest date) |
| See B200 current state | `b200/session-state.md` |
| Investigate B300 8.8 FPS issue | `b300/investigation-runbook.md` |
| See B300 optimization roadmap | `b300/optimization-vision.md` |
| Set up a B300 machine | `b300/setup-guide.md` |
| Understand RoPE optimizations | `rope/optimization.md` |
| Work on FA4 score_mod | `fa4/phase4-score-mod.md` |

## Directory Structure

```
notes/FA4/
├── README.md                      # This file
├── optimization-map.md            # Timeline + truth sources + workspace map
├── kernel-dev-log.md              # Main chronicle of all work
├── SESSION-HANDOFF-*.md           # Session handoff docs
├── docs/                          # Shareable deep dives
│   └── kernel-optimization-guide.md
├── explainers/                    # FA4/CuTe learning track (read code → write explainer)
│   └── README.md
│
├── b300/                          # B300 (SM103) specific
│   ├── investigation-runbook.md   # Systematic debugging guide
│   ├── investigation.md           # Investigation notes
│   ├── fa4-patches.md             # CUTLASS DSL patches for SM103
│   ├── experiments.md             # “One-change” experiment cards
│   ├── level5-level6-resources.md # Reading list for deeper kernel work
│   ├── optimization-ladder.md      # Learning ladder framing (corrected + measurement-driven)
│   ├── optimization-vision.md     # Strategic optimization roadmap
│   ├── setup-guide.md             # Environment setup
│   └── session-state.md           # Stashed session state
│
├── b200/                          # B200 (SM100) specific
│   ├── session-state.md           # Current “what to run today”
│   ├── experiments.md             # “One-change” experiment cards
│   └── bringup-log.md             # Historical bring-up notes
│
├── rope/                          # RoPE optimization work
│   ├── optimization.md            # Overview and strategy
│   ├── phase3-triton-rope.md      # Phase 3 main doc
│   ├── phase3-step2.md            # Step 2 details
│   └── phase3-step3.md            # Step 3 details
│
├── fa4/                           # FA4/CUTE score_mod work
│   └── phase4-score-mod.md        # FA4 integration plan
│
└── DeepResearch/                  # External research synthesis
    ├── summary.md                 # Overview
    ├── MSU_chat.md                # MSU analysis
    ├── wafer.md                   # Wafer profiling notes
    └── 2025-12-*/                 # Date-organized research
```

## Key Concepts

- **Kernel A**: Block-causal recompute path (FlexAttention with BlockMask)
- **Kernel B**: KV-cache bias path (Triton kernel, 10.7% faster than FlexAttention)
- **FA4/CUTE**: FlashAttention 4 with CUTLASS DSL - faster but has dependency issues
- **SM103**: B300 compute capability, requires patches for CUTLASS DSL

## Current Status (Dec 2025)

- **B200**: ~20 FPS at 320x576, 4 steps (working well)
- **B300**:
  - ~8.8 FPS on the repo-default cu128 stack
  - ~14.8–15.0 FPS at 320x576 on a cu130 env with FlashAttention installed (see `b300/session-state.md`)
- **H100**: Fallback option for hackathon (Jan 9 deadline)

## Quick Commands

```bash
# Run block profiler for investigation
uv run python scripts/profile_krea_pipeline_blocks.py \
  --iters 20 --skip 3 --profile-blocks

# Next denoise step (split transformer time: self_attn vs cross_attn vs ffn)
PROFILE_ATTENTION=1 \
uv run python scripts/profile_krea_pipeline_blocks.py --iters 6 --skip 2

# Test FA4 on B300
TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas uv run python scripts/test_fa4_kv_bias.py
```
