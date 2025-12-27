# Phase 4: FA4/CuTe DSL `score_mod` for KV-cache Bias (Kernel B)

## Why this exists

We already have a strong B200 path for Krea Realtime self-attention **KV-cache bias** (“Kernel B”) via a custom Triton kernel.

The next obvious “maybe bigger” win is to see whether **FlashAttention-4 (CuTe DSL)** can run the same bias path faster (or more simply) by using a custom `score_mod` that is inlined into FA4’s SM100 kernel.

This doc is a plan + integration sketch so other models/people can critique it quickly.

## Current repo status (2025-12-23)

### What’s working
- **RoPE Phase 3 Step 2 (v2 fused)** is complete and the default; perf is back to ~20 FPS on B200. See [`phase3-triton-rope.md`](../phase3-triton-rope.md).
- **Kernel B Triton** is integrated in `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py` and beats flex_attention on the real shape.
- **B1: FA4/CUTE score_mod KV-bias** is integrated (opt-in):
  - Backend selector: `SCOPE_KV_BIAS_BACKEND=fa4|triton|flex` (default `triton`)
  - Implementation: `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py`
  - Test harness: `scripts/test_fa4_kv_bias.py`

### Results (B200 / SM100)
- Kernel B latency (steady-state): **~0.54ms (FA4)** vs **~1.02ms (Triton)** → **~1.9× faster** Kernel B.
- End-to-end FPS uplift observed: **~+3–6%** (depends on resolution / workload mix).

### What changed to make “CuTe score_mod” work
- The installed wheel `flash-attn==2.8.3` exposes `flash_attn.cute.flash_attn_varlen_func(...)`, but it **does not accept `score_mod`**.
- We use the newer CuTe interface with `score_mod` support from the vendored CuTe sources:
  - `vendored/flash_attn_cute_score_mod/flash_attn/cute/interface.py`
  - This is injected at runtime by extending `flash_attn.__path__` (see `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py`).
  - KV-bias FA4 path imports and calls `flash_attn.cute.interface._flash_attn_fwd(...)` directly (not the wheel’s `flash_attn.cute.flash_attn_varlen_func`).

## Goal (Phase 4)

Implement and benchmark a FA4/CuTe DSL `score_mod` that matches Krea’s KV-cache bias rule, and optionally integrate it behind a feature flag:

Backend order goal:
1) FA4 `score_mod` (if available + faster)
2) Triton Kernel B (current default)
3) flex_attention fallback

We only flip defaults if FA4 clearly wins and is stable.

## KV-cache bias semantics (what we must exactly reproduce)

In `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py` the flex_attention fallback currently applies:

- `log_bias = log(kv_cache_attention_bias)` (natural log)
- Bias applies for keys where:
  - exclude first frame: `kv_idx >= frame_seqlen`
  - exclude current semantic block: `kv_idx < cache_current_block_start`

Where:
- `frame_seqlen = (height//16) * (width//16)`
- `block_size = frame_seqlen * num_frame_per_block`
- `cache_current_block_start = Lk - block_size` (in token units)

So biased region is `kv_idx ∈ [frame_seqlen, Lk - block_size)`.

## Key insight from CuTe DSL score_mod guide

From [`CuTeDSL_score_mod.md`](../DeepResearch/2025-12-23/CuTeDSL_score_mod.md) + FA4 code:

- `score_mod` is called with a `seqlen_info` object that exposes **runtime** `seqlen_k`.
- Passing `aux_tensors` usually forces `vec_size=1` (less vectorization), so we want **zero aux tensors**.

**Therefore:** we can avoid recompiling per `Lk` and avoid aux tensors by computing `cache_current_block_start` inside the score_mod from `seqlen_info.seqlen_k`.

## Proposed score_mod (in CuTe DSL)

We want an inference-only score modifier:

```
@cute.jit
def score_mod_kv_bias(scores, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
    Lk = seqlen_info.seqlen_k
    current_block_start = Lk - block_size_tokens
    mask = (kv_idx >= frame_seqlen_tokens) & (kv_idx < current_block_start)
    return scores + where(mask, log_bias, 0)
```

Notes:
- `scores` passed to `score_mod` are already multiplied by `softmax_scale` inside FA4 (see `apply_score_mod_inner` in `flash-attention.bak/flash_attn/cute/softmax.py`), so adding a natural-log bias is correct.
- `block_size_tokens = frame_seqlen * num_frame_per_block` is constant for a given pipeline resolution/config → closure constant.
- `log_bias` is constant for a run/config → closure constant.
- We must handle early frames where `Lk < block_size_tokens` by clamping:
  - `current_block_start = max(0, Lk - block_size_tokens)`

## Implementation plan

### Step 0 — Make `score_mod` importable (plumbing)

We need `flash_attn.cute.flash_attn_varlen_func` to accept `score_mod=...` (and maybe `aux_tensors=...`).

Two options:

**Option 0A (preferred for repo tools + reproducibility): use vendored CuTe code**
- ✅ Use `vendored/flash_attn_cute_score_mod/flash_attn/cute/interface.py` (tracked in Git).
- ✅ Import `flash_attn.cute.interface._flash_attn_fwd` (supports `score_mod`) after path injection.

**Option 0B (still useful for fast iteration): use a local clone**
- Keep a local `flash-attention.bak/` checkout and inject it via `flash_attn.__path__` if you prefer.

**Option 0B: upgrade/replace the wheel**
- Install a flash-attn build that exposes the score_mod CuTe interface.
- Higher risk (binary build/ABI) and slower iteration.

### Step 1 — Add a FA4 score_mod entrypoint in Scope

Touchpoint: `src/scope/core/pipelines/wan2_1/modules/attention.py`

- Add a function (or extend `flash_attention`) to accept:
  - `score_mod: Optional[Callable]`
  - `aux_tensors: Optional[list[torch.Tensor]]`
- Only attempt FA4 `score_mod` if:
  - `FLASH_ATTN_4_AVAILABLE` is True
  - the imported `flash_attn_4_varlen_func` signature supports the kwargs (guard with `inspect.signature` or `try/except TypeError`).
- If unsupported, fall back to:
  - Triton Kernel B (preferred), else flex_attention (existing).

### Step 2 — Implement the CuTe `score_mod` for KV bias

Touchpoint: new helper module, recommended:
- `src/scope/core/pipelines/krea_realtime_video/modules/cute_score_mod.py`

Responsibilities:
- Build and cache the `@cute.jit` callable.
- Avoid aux_tensors.
- Capture only constants that should specialize the kernel:
  - `frame_seqlen_tokens` (int)
  - `block_size_tokens` (int)
  - `log_bias` (float32)

Cache key suggestion:
`(frame_seqlen_tokens, block_size_tokens, round(log_bias, 6))`

### Step 3 — Microbench + correctness harness

We need an apples-to-apples benchmark on the true shape:
- `B=1, H=16, D=128`
- Typical: `Lq=4680`, `Lk=9360` (3-frame query, 6-frame KV)

Recommended new script:
- `scripts/bench_fa4_score_mod.py` (or extend `scripts/triton_sdpa.py --kernel-b` with a `--fa4-score-mod` mode)

Comparisons:
1) FA4 `score_mod` (new)
2) Triton Kernel B (current)
3) flex_attention `score_mod` (baseline)

Correctness:
- Compare outputs vs flex_attention reference for small and medium shapes (bf16 tolerance).
- Include “early cache” cases: `Lk` in `{frame_seqlen, 2*frame_seqlen, …, 6*frame_seqlen}`.

Perf:
- Warm up first-call JIT (CuTe compiles on first use).
- Report steady-state average over many iters.

### Step 4 — Integrate into Krea pipeline behind a flag

Touchpoint: `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py`

Add a new backend selector, suggested env var:
- `SCOPE_KV_BIAS_BACKEND=triton|fa4|flex` (default `triton`)

Behavior:
- If `fa4`: attempt FA4 score_mod; if it errors, log once and fall back to `triton` for the rest of the process (“trip breaker”).
- If `triton`: current behavior.

### Step 5 — Nsight validation (optional but recommended)

Once FA4 score_mod runs:
- Use NCU to confirm kernel is on SM100 FA4 path and check occupancy/reg pressure.
- Use NSYS to ensure there aren’t unexpected CPU-side overheads (e.g., repeated compilation).

## How hard is this? (risk / effort)

**Core score_mod logic:** low complexity (hours).

**Plumbing + deps:** medium to high complexity (days), because:
- Our current installed FA4 API doesn’t expose score_mod.
- CuTe DSL versions and signatures vary; we need robust guards.
- `nvidia-cutlass-dsl` can conflict with `torch._inductor` (module shadowing); see [`setup-guide.md`](../b300/setup-guide.md) and [`investigation.md`](../b300/investigation.md) (Issue 2).

**Performance outcome uncertainty:** medium.
- We might get a win if FA4’s SM100 kernel handles this shape better than our Triton implementation.
- But we should expect “maybe marginal” because Kernel B is already good and appears occupancy-limited in profiling.

## Acceptance criteria

- **Correctness:** Matches flex_attention reference within bf16 tolerance across early + steady-state `Lk` values.
- **No silent fallback:** logs clearly indicate which backend is used.
- **Performance:** FA4 score_mod is faster than Triton Kernel B on the real shape, or we keep Triton as default.
- **Stability:** no regressions in non-score_mod paths; failures trip-break back to Triton.

## RepoPrompt bundle (for external review)

**Files to include**
- `src/scope/core/pipelines/wan2_1/modules/attention.py`
- `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py`
- `scripts/triton_sdpa.py`
- `scripts/tune_kernel_b.py`
- [`CuTeDSL_score_mod.md`](../DeepResearch/2025-12-23/CuTeDSL_score_mod.md)
- `vendored/flash_attn_cute_score_mod/flash_attn/cute/interface.py`
- `vendored/flash_attn_cute_score_mod/flash_attn/cute/softmax.py`
- `vendored/flash_attn_cute_score_mod/flash_attn/cute/utils.py`

**Query (paste into RepoPrompt)**

Goal: implement FA4/CuTe DSL `score_mod` for Krea KV-cache bias attention (Kernel B), and benchmark it against Triton Kernel B.

1) Confirm why our current runtime FA4 (`flash-attn==2.8.3`) doesn’t expose `score_mod`, and the cleanest way to enable the newer `flash-attention.bak/flash_attn/cute/interface.py` implementation via Scope’s `_extend_flash_attn_path()` hook.
2) Design the CuTe `score_mod` to reproduce Scope’s bias rule:
   - bias keys where `kv_idx >= frame_seqlen` and `kv_idx < seqlen_info.seqlen_k - block_size_tokens`
   - add `log_bias = log(kv_cache_attention_bias)` (natural log)
   - avoid `aux_tensors` to keep `vec_size=2`
3) Propose exact code changes in:
   - `src/scope/core/pipelines/wan2_1/modules/attention.py` (FA4 score_mod call + guards)
   - `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py` (backend selection + fallback)
4) Propose a microbench harness (new script or extend `scripts/triton_sdpa.py`) that compares:
   - FA4 score_mod vs Triton Kernel B vs flex_attention score_mod
   using the real shape: `B=1,H=16,D=128,Lq=4680,Lk=9360` and several early-cache `Lk` values.
