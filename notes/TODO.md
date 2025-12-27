# Pending Tasks

> **Updated:** 2025-12-27 (Session E)
> **Purpose:** Cross-session task tracking

---

## Pending: Merge to Main + Upstream Sync

### Merge feature/stream-recording → main

**Status:** Ready, waiting for team coordination

`feature/stream-recording` is **289 commits ahead** of main, with main having 0 unique commits (clean fast-forward possible).

```bash
# When ready:
git checkout main && git merge feature/stream-recording --ff-only && git push origin main
```

**Why waiting:** Other team members are working on things. Coordinate before pushing.

### Upstream Sync Strategy

We're forked from an upstream repo. As we move fast, we need a strategy to incorporate upstream changes without losing our work.

**Recommended approach:**
1. **Track upstream as a remote:**
   ```bash
   git remote add upstream <upstream-repo-url>
   git fetch upstream
   ```

2. **Periodic rebase or merge from upstream/main:**
   ```bash
   # Option A: Rebase (cleaner history, more conflict-prone)
   git rebase upstream/main

   # Option B: Merge (preserves history, easier conflicts)
   git merge upstream/main
   ```

3. **Before big upstream syncs:**
   - Make sure our main is up to date
   - Create a sync branch: `git checkout -b sync/upstream-YYYY-MM-DD`
   - Resolve conflicts there, test, then merge to main

**Open questions:**
- [ ] What's the upstream repo URL?
- [ ] How often do we want to sync? (weekly? per-release?)
- [ ] Who owns the sync process?

---

## What Was Done This Session

### 1. B300 Optimization Orientation
- Reviewed current state: ~30.7 FPS (3.5× baseline), VAE decode solved
- Discussed Level 5 vs Level 6 work and realistic upside (~5-15% for Level 6 kernel work)
- Reviewed `philosophy.md` + `lessons.md` pairing (yin/yang for learning-first approach)
- Key insight: Level 6 value is in **patterns learned**, not FPS gained

### 2. Documentation Alignment
- Updated `capability-roadmap.md`:
  - Fixed status for Session Recorder (MVP), Style Swap (Partial)
  - Updated performance numbers (15 FPS → 30.7 FPS)
  - Added 4 missing features (§11-14): Transition Prompts, Hardware Control, Frame Buffer Scrubbing, Multi-GPU
  - Reorganized priority tiers (4 tiers now)
- Updated `proposals/README.md`:
  - Added Context Editing and VLM-Mediated V2V rows
  - Clarified VLM integration status
  - Added alignment note pointing to capability-roadmap.md

### 3. Tidal Cycles Proposal Update
- Integrated review from `oai_5pro01.md`
- Added: Reality Check section, Event Envelope Format, Alignment section (Option A vs B)
- Added: Phase 0 (offline scoring), Reliability/Safety section
- Updated: Open Questions with recommended defaults

---

## Pending: Style Swap (Ready to Test)

- [ ] Test style swap with `STYLE_SWAP_MODE=1`
  ```bash
  STYLE_SWAP_MODE=1 ./scripts/run_daydream_b300.sh
  # Then: video-cli style set rat
  ```
- [ ] Measure style swap performance (~50% FPS hit expected from `runtime_peft`)
- [ ] Post cohort project page after style switching demo is ready

---

## Pending: B300 Optimization (Codex Working)

- [ ] Other Codex scrubbing research docs (TMA/mbarrier/tcgen05/FA4 contracts)
- [ ] Fill `other-in-self-breakdown.md` from real stack-attributed profiles
- [ ] Level 5 work: fuse RoPE + attention glue (learning goal, modest FPS upside)

---

## Quick Reference

| Doc | Purpose |
|-----|---------|
| `notes/FA4/b300/philosophy.md` | Why we climb (learning > FPS) |
| `notes/FA4/b300/lessons.md` | What we've actually learned |
| `notes/FA4/b300/optimization-ladder.md` | The levels (currently at 4.5) |
| `notes/proposals/README.md` | Implementation status grid |
| `notes/capability-roadmap.md` | Deep specs + checklists |
| `notes/proposals/tidal-cycles-integration.md` | Just updated with review feedback |

---

## Git State

```
Branch: feature/stream-recording
Uncommitted:
- notes/TODO.md (this file)
- notes/capability-roadmap.md (status updates + new sections)
- notes/proposals/README.md (added rows)
- notes/proposals/tidal-cycles-integration.md (review integration)
```
