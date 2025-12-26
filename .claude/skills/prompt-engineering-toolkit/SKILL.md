---
name: prompt-engineering-toolkit
description: >
  Core prompt engineering techniques for LoRA-based video generation. Use when:
  prompts ignore critical elements, outputs are confused or generic, characters
  are inconsistent, performances feel lifeless, or style drifts toward realism.
  Covers attention budget, conceptual distance, simplification, aesthetic control,
  character consistency, performance direction, and advanced revision strategies.
allowed-tools: Read, Grep, Glob
---

# Prompt Engineering Toolkit

A comprehensive toolkit for architecting and revising prompts for LoRA-based video generation.

## How to Use This Skill

1. **Identify the problem** from user feedback or output analysis
2. **Match to a Trigger** in the index below
3. **Announce** which strategy you're using
4. **Read only that strategy file** (progressive disclosure)

## Strategy Index

### Core Writing Principles

| Strategy | Trigger | File |
|----------|---------|------|
| Attention Budget | Output ignores critical elements, confused composition, main action not executed | [pws-01-attention-budget.md](./strategies/pws-01-attention-budget.md) |
| Conceptual Distance | Novel/OOD idea fails, producing generic or incoherent result | [pws-02-conceptual-distance.md](./strategies/pws-02-conceptual-distance.md) |
| Generalization (OOD) | OOD subject overpowered by model training, looks like known subject | [pws-02a-generalization-ood.md](./strategies/pws-02a-generalization-ood.md) |
| Strategic Simplification | Prompt too long/detailed but output weak, confused, ignores instructions | [pws-03-strategic-simplification.md](./strategies/pws-03-strategic-simplification.md) |
| Stylization Capstone | Technical details correct but output lacks mood/feeling/artistic POV | [pws-04a-stylization-capstone.md](./strategies/pws-04a-stylization-capstone.md) |
| Aesthetic Control | Need specific cinematic direction for camera, lighting, or style | [pws-04b-aesthetic-control.md](./strategies/pws-04b-aesthetic-control.md) |
| Graffito Lexicon | Output looks too realistic/generic, lost handcrafted aesthetic | [pws-05-graffito-lexicon.md](./strategies/pws-05-graffito-lexicon.md) |
| Character Consistency | Main character rendered inconsistently, off-model, wrong attire | [pws-06-character-consistency.md](./strategies/pws-06-character-consistency.md) |
| Directing Performance | Character performance feels lifeless, generic, emotion not reading | [pws-07-directing-performance.md](./strategies/pws-07-directing-performance.md) |
| Active Props & VFX | Props feel static, VFX feel generic/out-of-style/lack impact | [pws-08-active-props-vfx.md](./strategies/pws-08-active-props-vfx.md) |
| Technical Analogy | Trying to achieve highly specific/complex visual effect | [pws-09-technical-analogy.md](./strategies/pws-09-technical-analogy.md) |

### Core Revision Strategies

| Strategy | Trigger | File |
|----------|---------|------|
| Logline Approach | Complex prompt failing, need to debug core concept | [prs-01-logline-approach.md](./strategies/prs-01-logline-approach.md) |
| Synonymic Nudge | Model overfitting, repetitive visuals, cluttered frame | [prs-02-synonymic-nudge.md](./strategies/prs-02-synonymic-nudge.md) |
| Poetic Reimagining | Literal description fails to capture emotional subtext | [prs-03-poetic-reimagining.md](./strategies/prs-03-poetic-reimagining.md) |

### Advanced Revision Toolkit (Prompt_Revision_NEW)

| Strategy | Trigger | File |
|----------|---------|------|
| Implicit Direction | Output stiff, literal, unimaginative despite direct commands | [prn-01-implicit-direction.md](./strategies/advanced/prn-01-implicit-direction.md) |
| Sequential Actions | Output unnaturally slow-motion, lacks kinetic energy | [prn-02-sequential-actions.md](./strategies/advanced/prn-02-sequential-actions.md) |
| Technical Keyword Injection | Descriptive language fails for process-oriented effects | [prn-03-technical-keyword.md](./strategies/advanced/prn-03-technical-keyword.md) |
| Strategic Front-Loading | Output focuses on wrong element (background vs subject) | [prn-04-front-loading.md](./strategies/advanced/prn-04-front-loading.md) |
| Anchor & Vector | Complex OOD action causes model to abandon core aesthetic | [prn-05-anchor-vector.md](./strategies/advanced/prn-05-anchor-vector.md) |
| High-Density Low-Token | Img2vid: model changing character/scene or failing motion | [prn-06-high-density-low-token.md](./strategies/advanced/prn-06-high-density-low-token.md) |
| Unlock Keyword | Desired effect not achieved through direct prompting | [prn-07-unlock-keyword.md](./strategies/advanced/prn-07-unlock-keyword.md) |

### Advanced Techniques

| Technique | Trigger | File |
|-----------|---------|------|
| Cross-Lingual Enhancement | English prompt still failing for complex OOD result | [adv-01-cross-lingual.md](./strategies/adv-01-cross-lingual.md) |

## Full Index Reference

For the complete master index with detailed summaries, see:
[_index-prompt-engineering.md](./strategies/_index-prompt-engineering.md)
