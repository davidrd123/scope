# Session Handoff - 2025-12-25 (Explainer Series)

> **Purpose:** Preserve context before compaction for seamless continuation.

---

## What We Were Doing

Writing an **explainer series** for FA4/CUTE internals. The goal is understanding-first documentation - the explainers ARE the deliverable, optimization is what might fall out.

## Completed This Session

### Documents Created

1. **`notes/NOTES-INDEX.md`** - Navigation hub for all notes
2. **`notes/PROJECT-OVERVIEW.md`** - Unified view of instrument + engine tracks
3. **`notes/daydream/cohort-pitch.md`** - Shareable project pitch for cohort
4. **`notes/daydream/cohort-update-2025-12-25.md`** - Progress update
5. **`notes/FA4/b300/optimization-ladder.md`** - The "hierarchy of sexy" for GPU optimization
6. **`notes/FA4/b300/level5-level6-resources.md`** - Resource map for Level 5/6
7. **`notes/FA4/explainers/README.md`** - Explainer series roadmap
8. **`notes/FA4/explainers/03-score-mod.md`** - COMPLETED explainer on score_mod
9. **`notes/FA4/explainers/04-rope.md`** - COMPLETED explainer on RoPE
10. **`notes/FA4/explainers/05-tma-memory-loading.md`** - COMPLETED explainer on TMA/memory loading

### Explainer Status

| # | Topic | Status |
|---|-------|--------|
| 1 | How FA4 Attention Works | ✅ DONE |
| 2 | The Blackwell Path (SM100) | ✅ DONE |
| 3 | score_mod: Custom Attention Logic | ✅ DONE |
| 4 | How FA Does RoPE | ✅ DONE |
| 5 | TMA and Memory Loading | ✅ DONE |
| 6 | Tile Scheduling and Pipelining | ✅ DONE |
| 7 | Online Softmax | ✅ DONE |

---

## Completed: Explainer #4 - How FA Does RoPE

**File:** `notes/FA4/explainers/04-rope.md`

### Key Insights Captured

1. **RoPE math:** `out0 = x0 * cos - x1 * sin; out1 = x0 * sin + x1 * cos`
2. **Two styles:** GPT-NeoX (split halves) vs GPT-J (interleaved)
3. **KV cache integration:** `seqlen_offsets` handles position shifts
4. **Why it's separate today:** Different Q/K offsets, cos/sin table needs
5. **Critical insight:** score_mod modifies **scores** (after QK^T), RoPE modifies **Q/K** (before QK^T) - different hook points!

---

## Completed: Explainer #5 - TMA and Memory Loading

**File:** `notes/FA4/explainers/05-tma-memory-loading.md`

### Key Insights Captured

1. **TMA:** Hardware async memory transfers, 1 thread issues, `cp.async.bulk.tensor` PTX
2. **mbarrier:** Phase-based synchronization for producer/consumer
3. **Warp specialization:** 16 warps with roles (softmax 0-7, correction 8-11, MMA 12, epilogue 13, load 14, empty 15)
4. **RoPE fusion answer:** For Blackwell/tcgen05, RoPE would modify shared memory AFTER TMA load, BEFORE MMA reads
5. **Key files studied:** `flash_fwd_sm100.py`, `copy_utils.py`, `pipeline.py`, `blackwell_helpers.py`

### Suggested Next Explainers

Natural follow-ons:
- **#6: Tile Scheduling and Pipelining** - How FA4 manages multi-stage pipeline
- **#7: Online Softmax** - The incremental softmax algorithm
- **#1: How FA4 Attention Works** - Overview/synthesis of the pieces

---

## Context for Continuation

### The Bigger Picture

- User is in **Daydream cohort** (Dec 22 - Jan 9)
- Two tracks: **Instrument** (control/branching) + **Engine** (kernel optimization)
- Currently at **Level 4.5** on the optimization ladder
- Goal: Level 5 (fused operations) as a **learning project**

### The Philosophy

From Codex's feedback (incorporated into optimization-ladder.md):
> "This is intentionally a *learning ladder*, not just a perf plan. It's totally fine to bounce between rungs as long as we keep writing down (1) what we tried, (2) what we measured, and (3) what we learned."

### Key Files to Reference

- Vendored FA4: `vendored/flash_attn_cute_score_mod/flash_attn/cute/`
- Installed FA: `.venv/lib/python3.12/site-packages/flash_attn/`
- Our RoPE: `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py` (search for `rope`)
- Triton RoPE: `src/scope/core/kernels/triton_rotary.py` (our fused version)

---

## Next Steps

1. **Continue with #6** (Tile Scheduling and Pipelining) - how FA4 manages the multi-stage pipeline
2. **Or #7** (Online Softmax) - the incremental softmax algorithm
3. **Or #1** (How FA4 Attention Works) - write the overview as synthesis
4. **Keep the learning-first mindset** - explainers are the win

---

---

## Completed: Explainer #6 - Tile Scheduling and Pipelining

**File:** `notes/FA4/explainers/06-tile-scheduling.md`

### Key Insights Captured

1. **Four scheduler types:** SingleTileScheduler (basic), StaticPersistentTileScheduler (reuses blocks), SingleTileLPTScheduler (L2 optimized), SingleTileVarlenScheduler (variable length)
2. **LPT (Longest Processing Time first):** Processes later M blocks first because they have more work (causal), ensures load balancing
3. **L2 cache swizzle:** Groups heads together so K/V stays in L2 across M block processing
4. **Pipeline state:** `PipelineStateSimple` tracks index (buffer slot) and phase (odd/even iteration)
5. **FastDivmod:** Precomputed magic multiplier for fast integer division on GPU
6. **Interleaved K/V loading:** K0, V0, K1, V1... because compute needs K before V

### Key Code Patterns

```python
# LPT reversal (tile_scheduler.py:347)
block = params.num_block - 1 - block

# L2 swizzle calculation (tile_scheduler.py:283)
swizzle = 1 << log2_floor(size_l2 // size_one_head)

# Pipeline state advance (pipeline.py:96)
self._phase_index += 1  # Encodes both index and phase
```

---

---

## Completed: Explainer #7 - Online Softmax

**File:** `notes/FA4/explainers/07-online-softmax.md`

### Key Insights Captured

1. **Online algorithm:** Maintain running `row_max` and `row_sum`, rescale previous work when new max discovered
2. **exp2 trick:** Use `exp2(x * log2(e))` instead of `exp(x)` - faster on GPU
3. **Rescale threshold (SM100):** Skip rescaling if `exp(old_max - new_max) ≈ 1.0`
4. **Warp specialization:** Softmax warps (0-7), correction warps (8-11) for rescaling O
5. **LSE computation:** `(row_max * scale_log2 + log2(row_sum)) * ln(2)` for backward pass
6. **Quad reduction:** `warp_reduce(..., width=4)` because 4 threads per quad share data

### Core Formula

```python
# Per tile:
scale = exp2((max_old - max_new) * scale_log2)  # Correction factor
P_tile = exp2(S_tile * scale_log2 - max_new * scale_log2)
row_sum = row_sum * scale + sum(P_tile)
O = O * scale + P_tile @ V_tile
```

---

## Completed: Explainer #2 - The Blackwell Path (SM100)

**File:** `notes/FA4/explainers/02-blackwell-path.md`

### Key Insights Captured

1. **tcgen05:** 5th generation tensor core instructions for Blackwell, uses `tcgen05.mma` PTX
2. **Tensor Memory (TMEM):** Dedicated per-warpgroup memory for MMA accumulators, 512 columns capacity
3. **Shared memory descriptors:** 64-bit descriptors encoding layout, swizzle, and addressing for MMA operands
4. **Elect-one pattern:** Single thread issues MMA via `elect.sync` instruction, avoids redundant work
5. **TS mode vs SS mode:** TS reads A operand from TMEM (used for P×V), SS reads both from SMEM (used for Q×K)
6. **Warp specialization on SM100:** 16 warps with dedicated roles (load, MMA, softmax, correction, epilogue)

### Key Code Patterns

```python
# Elect-one for MMA issue (blackwell_helpers.py)
elected, _ = asm(
    "elect.sync _|$0, 0xffffffff;",
    "+p,=r", True, 0
)

# Instruction descriptor (mma_sm100_desc.py)
desc = make_instr_desc(M=64, N=128, sparsity=0, neg_A=0, neg_B=0, trans_B=0)

# Shared memory descriptor (mma_sm100_desc.py)
smem_desc = (base_addr << 4) | (swizzle_type << 62) | (layout_type << 29)
```

---

## Completed: Explainer #1 - How FA4 Attention Works (Overview)

**File:** `notes/FA4/explainers/01-how-fa4-works.md`

### Key Insights Captured

1. **Three architectures:** SM80 (Ampere), SM90 (Hopper), SM100 (Blackwell) with increasing hardware specialization
2. **Kernel structure:** Prologue → Mainloop → Epilogue pattern across all architectures
3. **Mainloop separation:** First tile (boundary checks), causal tiles (mask), full tiles (no mask) for performance
4. **Architecture routing:** `interface.py` dispatches to `FlashAttentionForwardSm80/90/100` based on compute capability
5. **Why online softmax:** Enables single-pass attention, O(n) memory instead of O(n²)
6. **Why warp specialization:** SM100's TMEM/tcgen05 require specific programming patterns

### Synthesis Diagram

```
Entry: interface.py → Dispatch by compute_capability
                    ↓
         ┌─────────────────────────┐
         │  Architecture Classes   │
         │  SM80 / SM90 / SM100    │
         └─────────┬───────────────┘
                   ↓
    ┌──────────────────────────────┐
    │          MAINLOOP            │
    │  S = Q @ K^T → softmax → O   │
    │  (Explainers #5,6,7 details) │
    └──────────────────────────────┘
```

---

## Session Stats (Final)

- **Explainer series COMPLETE** - All 7 explainers written
- Created ~14 new documents total across sessions
- Completed explainers:
  1. How FA4 Attention Works (Overview/Synthesis)
  2. The Blackwell Path (SM100)
  3. score_mod: Custom Attention Logic
  4. How FA Does RoPE
  5. TMA and Memory Loading
  6. Tile Scheduling and Pipelining
  7. Online Softmax
