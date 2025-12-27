# Optimization Philosophy: Why We Climb

> **Context:** Guiding principles for Level-5+ kernel work on B300.
> **Created:** 2025-12-27
> **See also:** `optimization-ladder.md` (Level definitions)

---

## Where This Fits

This document is the **“why”** for how we operate.

- For **current truth / commands / caveats**, start at [`README.md`](README.md) → [`session-state.md`](session-state.md).
- For **how to measure**, see [`investigation-runbook.md`](investigation-runbook.md).
- For **one-change experiment cards**, write in [`experiments.md`](experiments.md).
- For the **“what we learned” distillation**, see [`lessons.md`](lessons.md).
- For any **kernel work**, use [`kernel-experiment-template.md`](kernel-experiment-template.md) as the gate (perf + quality).

## The Core Thesis

**We're here to learn patterns, not just chase FPS.**

FPS is the feedback signal that tells us whether we understood something correctly. It’s not the only goal. The goal is to deeply understand GPU optimization primitives so we can:

1. Apply them to future problems (this project and beyond)
2. Know when *not* to write a custom kernel
3. Design novel algorithms that wouldn't be possible without owning the primitives

---

## What Success Looks Like

| Metric | Anti-pattern | Success pattern |
|--------|--------------|-----------------|
| **Output** | "Got 2% faster" | "Documented why X works and Y doesn't" |
| **Experiment** | Try 5 knobs, ship the winner | One variable, measured, written down |
| **Stopping** | "We could try one more thing..." | "Diminishing returns; here's what we learned" |
| **Knowledge** | Lives in your head | Lives in a Markdown file someone else can read |

---

## The Hunt List

Primitives we're trying to understand deeply:

### 1. TMA (Tensor Memory Accelerator)
- Hardware unit that loads tensor tiles directly into shared memory
- Async, overlapped with compute
- **Pattern:** Producer warps issue TMA loads; consumer warps compute on previous tile

### 2. mbarrier (Async Barriers)
- Coordination primitive for async operations
- arrive/wait semantics with phase tracking
- **Pattern:** Know when data is ready without busy-spinning

### 3. Warp Specialization
- Different warps do different jobs (load / compute / store)
- Software pipelining across warp groups
- **Pattern:** Hide latency by overlapping different stages

### 4. Fused Kernel Design
- Epilogue hooks (do work after GEMM, before store)
- Prologue hooks (do work during load, before compute)
- **Pattern:** Eliminate intermediate stores/loads

### 5. Layout & Alignment Contracts
- What shapes/strides/alignments do the fast paths require?
- When do you silently fall back to slow paths?
- **Pattern:** Design data flow to stay on fast paths

---

## The Experiment Shape

Every experiment should fit this template:

```markdown
## Experiment: [Name]

**Hypothesis:** [What we think will happen and why]

**Setup:**
- Baseline config: [exact config]
- Change: [exactly one thing]
- Measurement: [how we'll know]

**Result:** [What actually happened]

**Lesson:** [What we learned, even if null result]
```

Null results are valuable. "This didn't work because X" is knowledge.

---

## Evidence Policy (Receipts, Not Vibes)

Treat **every claim** as one of:
- **Measured:** We ran it and saved artifacts.
- **Sourced:** It’s a spec/layout constraint with a pinned version + link.
- **Code-audited:** We point at the exact code path (file + symbol) that enforces it.
- **Inferred:** We think it’s true; we label it and include a verification path.

Minimum bar for performance claims:
- Exact command + env (baseline and change), run under comparable machine/GPU conditions.
- A saved artifact in `outputs/` (profile json, summary md, raw logs) with a stable name.
- Version pins (at least: torch, CUDA toolkit, driver, Triton; cuDNN if relevant).

Minimum bar for spec/layout claims:
- Link to the primary source + version (PTX ISA / CUDA Guide / Driver API), and quote or paraphrase narrowly.
- If we’re unsure, write **“inferred”** and include the fastest way to confirm (file path + grep target).

This is the meta-skill: don’t let future you (or future models) get derailed by unmarked speculation.

---

## When to Stop

Signs you should stop and document:

1. **Diminishing returns** — The next experiment would be smaller than measurement noise
2. **Wrong mountain** — You're optimizing something that isn't the bottleneck
3. **Complexity explosion** — The "simple" kernel is now 500 lines of edge cases
4. **The answer is "don't"** — Sometimes the library path is correct

The goal is to *know* these things, not to avoid them. Hitting a wall and understanding why is success.

---

## What Transfers

These patterns apply beyond B300/attention:

| Pattern | This project | Future applications |
|---------|--------------|---------------------|
| TMA | Attention Q/K/V loads | Any tensor-heavy kernel |
| mbarrier | Sync between load/compute | Producer-consumer pipelines |
| Warp spec | Attention warp roles | Any latency-hiding design |
| Fused kernels | RoPE + pack + KV-write | Any "glue elimination" |
| Layout contracts | FA4 requirements | Any library integration |

The specific kernel we write matters less than the patterns we extract.

---

## The Reusable Playbook (Levels 1-5)

The documentation built here isn't just about B300. It's a template:

Levels are defined in `optimization-ladder.md`. Short glossary:
- **1–2:** Profile + build trustworthy measurement.
- **3:** Make the stack/library fast paths applicable (compat + flags).
- **4:** Fix shape/stride/layout mismatches so fast paths stay enabled.
- **5:** Stack-attributed profiling to delete glue (copies, views, repacks).
- **6:** Minimal custom kernel/fusion work with strict validation gates.
- **7:** Algorithm-level changes (may require custom kernels).
- **8:** Productionization (hardening, portability, long-term maintenance).

| Level | Action | Artifact |
|-------|--------|----------|
| 1-2 | Profile → find bottleneck | `investigation-runbook.md` |
| 3 | Check FA4/library compatibility | `layout-contracts.md` |
| 4 | Fix shape/stride mismatches | `layout-contracts.md` |
| 4-5 | Stack-attributed profiling → kill glue | `other-in-self-breakdown.md` pattern |

**Next model, same checklist.** The Level-6 hunting is specific to this architecture; the Level 1-5 loop is general.

This is why we write things down: you're building **tools**, not just knowledge. The runbook that worked for B300 will work for the next GPU with minor edits. The layout-contracts pattern applies to any library integration.

The philosophy doc tells you *why* to climb. The playbook tells you *how* to climb any mountain.

---

## The Mountain Metaphor

> "You climb it to see what's on the other side, not just for the altitude."

The FPS number is the altitude reading. It tells you you're going up. But the view from the top — understanding how these primitives compose — is why you climbed.

Sometimes you discover the mountain isn't worth climbing. That's also valuable. Now you know.

---

## Practical Guardrails

1. **One experiment at a time** — If you can't describe the single variable, you're not ready
2. **Write it down before you forget** — The lesson evaporates faster than you think
3. **Timebox exploration** — "I'll spend 2 hours understanding X" not "I'll figure out X"
4. **Ask for missing docs** — If progress depends on upstream/spec details, stop and ask for doc fetch/curation
5. **Quality sanity gate** — If output looks wrong, stop and debug before measuring perf
6. **Baseline before change** — Always re-measure baseline; it drifts

### B300-Specific Guardrails (Keep Us Honest)

- **Quality over speed:** FP8 is currently off-limits for “real” output; BF16 (`--quantization none`) is canonical. (Details: [`session-state.md`](session-state.md).)
- **Canonical comparisons:** `320x576`, stable quality-preserving settings (no KV-cache recompute skipping), and record the exact command/env.
- **Avoid “perf-only” wins:** if a knob improves FPS but causes visible glitches, keep it documented as debug-only.

---

## The Anti-Goals

Things we're explicitly *not* trying to do:

- Ship a production kernel to millions of users (Level 8)
- Invent a novel algorithm (Level 7) — that might emerge, but it's not the goal
- Hit a specific FPS target **at the expense of quality or understanding**
- Compete with NVIDIA's libraries — we're learning their patterns, not replacing them

---

## Summary

We're doing Level-6 work because:

1. **The patterns transfer** — TMA/mbarrier/warp-spec are the vocabulary of modern GPU kernels
2. **It unblocks understanding** — You can't design novel algorithms without owning the primitives
3. **Small wins compound** — And more importantly, the *understanding* of why they work compounds
4. **Knowing when NOT to write a kernel is valuable** — That's a skill you can only learn by trying

The mountain is real. Climb it for the view.
