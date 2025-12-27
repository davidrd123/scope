# 2025-12-27: “Climbing The Mountain” (Blackwell / SM103 doc pack)

Curated, source-linked notes to “pack our gear” for Level 6 (custom kernel / fusion) work on Blackwell (SM103).

## Verified takeaways (don’t regress these)

- **mbarrier** is a `.b64` (8-byte) object in shared memory with **8-byte alignment** (not 64-byte).
- **CUtensorMap** descriptor objects must have their **address aligned to 64 bytes**.
- `cuTensorMapEncodeTiled` **tensorRank must be 3, 4, or 5** (and `globalStrides[]` are in bytes).

## What’s in here

- `oai_5pro.md`: Primary-source constraint notes with direct NVIDIA doc URLs (tool citation artifacts removed).
- `oai_dr01.md`: Longer synthesis with many footnotes; includes a clearly-marked inferred section for FA4 `score_mod` / KV-bias usage.
- `claude01.md`: Broad reference sweep (PTX, mbarrier, tcgen05, tensormap, patterns).
- `claude03a.md`: “Spec-like” extraction; corrected `cuTensorMapEncodeTiled` rank constraint (3–5).
- `claude04a.md`, `claude04b.md`: Resolves the common **mbarrier vs CUtensorMap alignment** confusion; summarizes `sm_*a` target nuance.
- `claude02.md`, `claude03b.md`: Placeholders (kept for provenance with the original thread naming).

## Where this plugs in

- B300 working set index: `notes/FA4/b300/README.md`
- Blackwell primitives quick-ref: `notes/FA4/b300/blackwell-primitives-cheatsheet.md`
- Level 6 layout contracts: `notes/FA4/b300/layout-contracts.md`
