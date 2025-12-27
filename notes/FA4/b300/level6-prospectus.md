# Level 6 Kernel Work Prospectus

> **Date:** 2025-12-27
> **Baseline (benchmark harness):** ~34.5 FPS (BF16 / `--quantization none` + `--compile`, B300)
> **Purpose:** Plan the next measured Level 6 kernel exploration

---

## Current State (What We Actually Know)

| Metric | Value |
|--------|-------|
| Best-known (harness) | ~34.5 FPS @ `320x576` |
| Frame time (harness) | ~29.0ms/frame |
| Attention split (no-compile drilldown) | `self_attn` dominates denoise |
| Self-attn split (no-compile drilldown) | `other_in_self` dominates after FA4 KV-bias |
| Level 6 upside (speculative) | ~1–3ms/frame (if we delete real overhead) |

**Key insight:** The value of Level 6 is in **measured overhead deleted** *and* **patterns learned** (TMA/mbarriers/warp-spec/TMEM) that transfer to future kernels.

---

## Kernel-Targetable Time (Measured; Don’t Guess)

We should not estimate “ms/frame” by inventing call counts. Instead:

- Use the **`PROFILE_ATTENTION`** report (no compile) to get `ms/call` + `calls`.
- Use the benchmark log (`iter=... frames=...`) to compute **`calls_per_frame = calls_total / frames_total`**.
- Convert to `ms/frame = ms/call * calls_per_frame`.

### Example budget (latest drilldown run)

From `outputs/b300_cu130_none_bias0.3_drilldown_2025-12-27_resume_perf.log` (no compile; `iters=4`, `skip=1`):
- Profiled iters: 3 (`iter=001..003`)
- Frames per iter: 12
- **Frames profiled:** 36

From `notes/FA4/b300/other-in-self-breakdown.md` (same run), call counts:
- `self_attn`: 600 calls → ~16.67 calls/frame
- `rope_apply`: 480 calls → ~13.33 calls/frame
- `cache_update`: 480 calls → ~13.33 calls/frame

Per-frame contributions in that run:
- `rope_apply`: 0.11ms/call × 13.33 ≈ **~1.5ms/frame**
- `cache_update`: 0.06ms/call × 13.33 ≈ **~0.8ms/frame**
- “unlabeled remainder” inside `other_in_self` (see breakdown doc): **~4–5ms/frame** (needs stack attribution to name)

**Interpretation:** Level 6 is only worth it if we can delete **real, repeated overhead** (layout/packing/kv-write) that survives `--compile` and is not already dwarfed by GEMMs / FA4.

**Extra signal (compile+stack):** `outputs/b300_cu130_ops_profile_selfattn_compile_fa4_stack_2025-12-27_resume.md` shows non-trivial
`aten::copy_` + `direct_copy_kernel_cuda` time inside `CausalWanSelfAttention` stacks, pointing at the RoPE call sites. This strongly suggests a
“RoPE on padded tensors” tax (e.g. tail-preservation copies). If that’s correct, the first “Option A” prototype should prioritize **prefix-only**
RoPE/pack (write only active tokens) rather than fusing more compute.

---

## Candidate Projects

### Option A: Fused RoPE + KV-Pack Kernel

**What:** Single kernel that applies RoPE to K and packs into KV-cache layout.

**Why:**
- RoPE is 0.11ms/call, already Triton
- Cache update is 0.11ms/call
- Fusion eliminates intermediate tensor + memory round-trip

**Primitives to learn:**
- TMA for async KV-cache writes (contiguous tile → strided cache)
- Warp specialization if RoPE compute is light enough to overlap with memory

**Difficulty:** Medium
**Expected savings:** Express as **% of (RoPE + cache_update)**, not fixed ms:
- If a fused kernel removes ~25–50% of `rope_apply + cache_update` (≈2.3ms/frame in the example run), that’s **~0.6–1.2ms/frame**.
**Learning value:** High (TMA write patterns, fusion boundary design)

---

### Option B: Post-Projection Pack Kernel

**What:** Kernel that takes strided QKV output from fused projection and packs into attention-ready layout.

**Why:**
- Fused projections can output strided views (config-dependent).
- Some configs show layout fixups (`copy_` / dtype casts / contiguity work) before attention.
- If this overhead exists and is not already optimized away, a single pack kernel can delete it.

**Primitives to learn:**
- TMA for strided reads (non-contiguous input)
- Tile-based transpose + pack

**Difficulty:** Medium-Hard
**Expected savings:** Unknown until we re-measure under today’s “best stack”.
**Learning value:** Medium (strided TMA patterns), High if it deletes multiple kernels.

---

### Option C: Fused LayerNorm + Projection Epilogue

**What:** Fuse the pre-attention LayerNorm into the projection GEMM epilogue.

**Why:**
- LayerNorm shows up in profiles as separate Triton kernels
- CUTLASS epilogue fusion could eliminate the separate pass

**Primitives to learn:**
- CUTLASS epilogue customization
- Online normalization (running mean/var)

**Difficulty:** Hard
**Expected savings:** Potentially meaningful, but only if LN is still a visible hotspot under `--compile`.
**Learning value:** High (CUTLASS epilogue patterns)

---

### Option D: Custom FA4-Compatible Attention Kernel

**What:** Write our own attention kernel that matches FA4's KV-bias contract.

**Why:**
- Full control over layout, fusion opportunities
- Could fuse RoPE + attention in one kernel

**Primitives to learn:**
- Everything: TMA, mbarrier, warp-spec, online softmax, all of it

**Difficulty:** Very Hard (weeks of work)
**Expected savings:** Uncertain (FA4 is already good)
**Learning value:** Maximum (but time cost is high)

---

## Recommended First Project: Option A (Fused RoPE + KV-Pack)

**Rationale:**
1. Clear fusion boundary (RoPE output → cache write)
2. We already measure non-trivial time in `rope_apply` + `cache_update` in the drilldown run
3. TMA write patterns are typically simpler than TMA reads
4. It’s a good learning kernel even if the end-to-end win is modest
5. Can prototype in Triton first, then go CuTe/CUTLASS if warranted

**Concrete “first cut” inside Option A (based on compile+stack):**
- Prioritize deleting RoPE-related copy kernels (e.g. `aten::copy_` / `direct_copy_kernel_cuda`) that appear at the RoPE call sites under
  `CausalWanSelfAttention`. Treat this as a **prefix-only RoPE/pack** problem first, not a “fuse more math” problem.
- Keep `SCOPE_ROPE_K_TO_CACHE=1` as an optional follow-up; the quick A/B so far suggests it’s not the dominant lever by itself.

---

## Prerequisites / Research for Tomorrow

### 1. Layout Contract Review
- [ ] Review `outputs/b300_layout_contract_fuseproj_2025-12-27.json`
- [ ] Understand exact shapes at RoPE → cache boundary
- [ ] Document strides for both source (RoPE output) and dest (KV-cache)

### 2. TMA Basics
- [ ] Read `notes/FA4/b300/blackwell-primitives-cheatsheet.md`
- [ ] Understand TMA descriptor setup for 2D tiles
- [ ] Check if TMA supports the strided patterns we need

### 3. Existing Kernel Inspection
- [ ] Read current Triton RoPE kernel (`rope_fused_3way_kernel_v2`)
- [ ] Read current cache update code path
- [ ] Identify exact fusion point

### 4. Prototype Strategy
- [ ] Start with Triton (faster iteration)
- [ ] Add explicit `tl.store` to cache (no TMA)
- [ ] Measure overhead of fused vs separate
- [ ] Only go to CUTLASS/TMA if Triton version shows promise

### 5. Guardrails (Blast Radius)
- [ ] Ship behind an env flag (e.g. `SCOPE_L6_FUSE_ROPE_CACHE=1`) with a hard fallback to the current path
- [ ] Add a “correctness oracle” (small-shape unit check comparing fused vs reference)
- [ ] Add an opt-in microbench harness that uses **real shapes** from `layout-contracts.md`

---

## Success Criteria

| Outcome | FPS | Learning |
|---------|-----|----------|
| **Home run** | +1.5ms/frame or better | Reusable TMA/warp-spec patterns documented |
| **Solid win** | +0.5–1.5ms/frame | Fusion deletes kernels and stays stable |
| **Learning only** | No FPS change | We can explain exactly why (measured) |
| **Pivot signal** | — | If RoPE+pack fusion is blocked, try Option B |

---

## Files to Reference

| Doc | Purpose |
|-----|---------|
| `notes/FA4/b300/layout-contracts.md` | Tensor shapes at boundaries |
| `notes/FA4/b300/blackwell-primitives-cheatsheet.md` | TMA/mbarrier quick ref |
| `notes/FA4/b300/level5-level6-resources.md` | External links |
| `notes/FA4/b300/other-in-self-breakdown.md` | Profile data |
| `outputs/b300_layout_contract_*.json` | Real layout dumps |

---

## Tomorrow's Kickoff Sequence

1. Re-run a **no-compile drilldown** (`scripts/profile_b300_denoise_drilldown.sh`) and confirm:
   - frames/iter
   - `rope_apply` + `cache_update` `ms/call`
2. Re-run a **compile+stack op profile** scoped to `CausalWanSelfAttention` and confirm what copy/pack overhead survives.
3. Fill/update `layout-contracts.md` for the exact RoPE→cache boundary (one representative shape).
4. If (and only if) there is still a meaningful budget:
   - build a Triton fused prototype behind a flag
   - microbench it on the real shapes
   - integrate into one call site with a fallback
5. Update `experiments.md` with: measurements, decision, and next actions.

Good night!
