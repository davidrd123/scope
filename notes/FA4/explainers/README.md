# FA4/CUTE Explainer Series

> **Goal:** Deep understanding of FlashAttention 4 internals, documented as teachable explainers.
> **Approach:** Read code → Write explainer → Understanding crystallizes → (Maybe) optimize
> **Updated:** 2025-12-25

---

## The Explainers

Each explainer should:
1. Explain what the code does at a conceptual level
2. Walk through the actual implementation
3. Highlight key design decisions
4. Note opportunities or questions that arise

### Phase 1 (Complete): Forward-Pass Mental Model

| # | Topic | Source Files | Status |
|---|-------|--------------|--------|
| 1 | **How FA4 Attention Works** | `flash_fwd.py`, `interface.py` | ✅ DONE |
| 2 | **The Blackwell Path (SM100)** | `flash_fwd_sm100.py`, `blackwell_helpers.py` | ✅ DONE |
| 3 | **score_mod: Custom Attention Logic** | `interface.py`, your KV-bias code | ✅ DONE |
| 4 | **How FA Does RoPE** | `layers/rotary.py`, `ops/triton/rotary.py` | ✅ DONE |
| 5 | **TMA and Memory Loading** | `flash_fwd_sm100.py`, tcgen05 blog | ✅ DONE |
| 6 | **Tile Scheduling and Pipelining** | `tile_scheduler.py`, `pipeline.py` | ✅ DONE |
| 7 | **Online Softmax** | `softmax.py` | ✅ DONE |

### Phase 2 (In Progress): Missing Pieces / “Make It Usable”

This phase fills in the “gaps” that matter when you’re debugging real kernels in the wild or trying to extend FA4 for our workload (KV-bias, streaming, SM103).

| # | Topic | Source Files | Status |
|---|-------|--------------|--------|
| 8 | **Masking and `mask_mod`** | `mask.py`, `mask_definitions.py` | ✅ DONE |
| 9 | **Paged KV (when TMA KV is off)** | `paged_kv.py`, `flash_fwd_sm100.py` | 📝 TODO |
| 10 | **Backward Pass (SM100)** | `flash_bwd_sm100.py`, `softmax.py` | 📝 TODO |
| 11 | **Split-K and Segment-Combine** | `interface.py` + forward combine/reduction paths | 📝 TODO |
| 12 | **SM103 Notes (B300): What Carries Over** | our backend selection + FA4/CuTe constraints | 📝 TODO |

### Explainers Index

| # | Topic | File |
|---|-------|------|
| 1 | How FA4 Attention Works | [01-how-fa4-works.md](01-how-fa4-works.md) |
| 2 | The Blackwell Path (SM100) | [02-blackwell-path.md](02-blackwell-path.md) |
| 3 | score_mod: Custom Attention Logic | [03-score-mod.md](03-score-mod.md) |
| 4 | How FA Does RoPE | [04-rope.md](04-rope.md) |
| 5 | TMA and Memory Loading | [05-tma-memory-loading.md](05-tma-memory-loading.md) |
| 6 | Tile Scheduling and Pipelining | [06-tile-scheduling.md](06-tile-scheduling.md) |
| 7 | Online Softmax | [07-online-softmax.md](07-online-softmax.md) |
| 8 | Masking and `mask_mod` | [08-masking-and-mask_mod.md](08-masking-and-mask_mod.md) |
| 9 | Paged KV | [09-paged-kv.md](09-paged-kv.md) |
| 10 | Backward Pass (SM100) | [10-backward-sm100.md](10-backward-sm100.md) |
| 11 | Split-K and Segment-Combine | [11-splitk-and-segment-combine.md](11-splitk-and-segment-combine.md) |
| 12 | SM103 Notes (B300) | [12-sm103-notes.md](12-sm103-notes.md) |

---

## Source Material

### Code to Study
```
vendored/flash_attn_cute_score_mod/flash_attn/
├── cute/
│   ├── interface.py            # Entry point, score_mod plumbing
│   ├── flash_fwd.py            # General forward pass
│   ├── flash_fwd_sm100.py      # SM100 reference path (closest Blackwell reference in-tree)
│   ├── blackwell_helpers.py    # SM100 utilities
│   ├── softmax.py              # Online softmax
│   ├── tile_scheduler.py       # Tile scheduling
│   └── pipeline.py             # Pipeline stages
└── (layers/rotary.py)          # RoPE lives in the installed FlashAttention package (locate via Python)
```

### Reference Blogs (already in repo)
- `notes/research/2025-12-24/incoming/perf/blogs/thunderkittens-blackwell.md`
- `notes/research/2025-12-24/incoming/perf/blogs/gau-nerst-tcgen05.md`
- `notes/research/2025-12-24/incoming/perf/blogs/warp-specialization.md`

---

## Explainer Template

```markdown
# [Topic]: How [X] Works in FA4

## Overview
What is this? Why does it matter?

## The Concept
High-level explanation without code.

## The Implementation
Walk through the actual code with annotations.

## Key Design Decisions
Why did they do it this way?

## Questions & Opportunities
What's unclear? What could be improved?

## References
Links to source files, papers, blogs.
```

---

## How This Connects to Optimization

The optimization ladder from `optimization-ladder.md`:

```
Level 4 (where we are): Beat the libraries
Level 5 (next):         Fused operations
Level 6 (future):       Blackwell-specific (TMA, warp spec)
```

**The hypothesis:** Deep understanding of FA4 internals will reveal:
1. Where fusion opportunities exist (Level 5)
2. How Blackwell features are currently used (Level 6)
3. What's suboptimal for our specific workload

But the explainers have value even if no optimization falls out - they're documentation of the journey and reference material for the future.
