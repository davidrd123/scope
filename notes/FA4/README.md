# FA4 Kernel Optimization Notes

This directory tracks the journey of optimizing attention kernels for the Krea Realtime Video pipeline.

## Where to Start

| If you want to... | Start here |
|-------------------|------------|
| Understand the full journey | `kernel-dev-log.md` |
| Continue from last session | `SESSION-HANDOFF-*.md` (latest date) |
| Investigate B300 8.8 FPS issue | `b300/investigation-runbook.md` |
| Set up a B300 machine | `b300/setup-guide.md` |
| Understand RoPE optimizations | `rope/optimization.md` |
| Work on FA4 score_mod | `fa4/phase4-score-mod.md` |

## Directory Structure

```
notes/FA4/
├── README.md                      # This file
├── kernel-dev-log.md              # Main chronicle of all work
├── SESSION-HANDOFF-*.md           # Session handoff docs
│
├── b300/                          # B300 (SM103) specific
│   ├── investigation-runbook.md   # Systematic debugging guide
│   ├── investigation.md           # Investigation notes
│   ├── fa4-patches.md             # CUTLASS DSL patches for SM103
│   ├── setup-guide.md             # Environment setup
│   └── session-state.md           # Stashed session state
│
├── b200/                          # B200 (SM100) specific
│   └── bringup-log.md             # Initial bring-up notes
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

## Current Status (Dec 2024)

- **B200**: ~20 FPS at 320x576, 4 steps (working well)
- **B300**: 8.8 FPS at same settings (mystery unsolved - see `b300/investigation-runbook.md`)
- **H100**: Fallback option for hackathon (Jan 9 deadline)

## Quick Commands

```bash
# Run block profiler for investigation
uv run python scripts/profile_krea_pipeline_blocks.py \
  --iters 20 --skip 3 --profile-blocks

# Test FA4 on B300
TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas uv run python scripts/test_fa4_kv_bias.py
```
