# Phase 3 – Step 2 (Triton RoPE "3‑Way Lookup Fused") Spec

## Status (2025-12-23)

**IMPLEMENTED BUT REGRESSED**: Kernel works correctly but causes 20 → 17.8 FPS regression.

- **Kernel:** `src/scope/core/kernels/triton_rope_fused.py`
- **Integration:** Fast path in `rope_apply()` and `causal_rope_apply()`
- **Disable flag:** `SCOPE_DISABLE_TRITON_ROPE_FUSED=1` (use this until fixed)

**Root cause:** Power-of-2 padding. Triton `tl.arange()` requires power-of-2 ranges:
- Actual: C0=22, C1=21, C2=21 (64 pairs)
- Padded: 32, 32, 32 (96 pairs) = **50% more work**

**Fix needed:** Channel tiling or different approach to avoid padding overhead.

---

## Goal

Implement a **single Triton kernel** that applies RoPE for Krea Realtime Video by:

- Computing `(f_idx, h_idx, w_idx)` **inside the kernel** (no Python loops for indexing)
- Loading RoPE angles from **three small per-axis tables** (time/height/width)
- Rotating `x[..., 2*j:2*j+2]` in-place or out-of-place **without materializing `(seq_len, C)` cos/sin**
- Preserving tail (`t >= seq_len`) without cloning the whole tensor

Primary target: **B200 (SM100)**, steady-state shapes: `B=1`, `H=16`, `D=128`, `seq_len=f*h*w` with `L` padded to a multiple of 128.

## Current Baseline (what Step 2 replaces)

`rope_apply()` / `causal_rope_apply()` today:

- Split `freqs` into 3 chunks.
- Build `freqs_i` via `expand + cat + reshape` → then `cos/sin = freqs_i.real/imag`.
- Cache `(cos, sin)` in `_ROPE_CACHE` (Phase 2.5).
- Apply rotation (PyTorch) or call `triton_apply_rotary` (Phase 3 Step 0).
- If `seq_len < L`, **clone the entire `x`** to preserve tail and write prefix into it.

### Performance hazard #1: cloning when padded

In Krea, `L` is right-padded to a multiple of 128. Example: `seq_len=4680` → `L=4736` (tail = 56 tokens).

Current behavior clones the full `[B, L, H, D]` tensor (~19MB for bf16 at `B=1,L=4736,H=16,D=128`) just to preserve a ~56-token tail.

**Step 2 must avoid full clones.**

Recommended semantics:
- **Out-of-place:** allocate `out = empty_like(x)`, write prefix in kernel, then copy the tail only:
  - `out[:, seq_len:] = x[:, seq_len:]` (tail copy is tiny; ≤127 tokens typically).
- **In-place (inference):** masked stores to `x` for `t < seq_len`; tail remains untouched.

This tail strategy should also be applied to Step 0 / Step 1 wrappers if we want “free” gains pre-Step 2.

## Semantics to Preserve

### Data layout

All current callsites expect:
- `x` shape: `[B, L, H, D]`
- `D = 2*C`, interleaved pairs: `(2*j, 2*j+1)` are a complex pair

If we later discover a `[B, H, L, D]` layout in some callsite, support it explicitly (wrapper chooses strides + `HN`).

### Token ↔ grid mapping

Let `grid_sizes = (f, h, w)`, `seq_len = f*h*w`, `HW = h*w`.

For token index `t ∈ [0, seq_len)`:
- `f_local = t // HW`
- `rem = t - f_local*HW`
- `h_idx = rem // w`
- `w_idx = rem - h_idx*w`
- `f_idx = start_frame + f_local`

**Only time uses `start_frame`.** Spatial axes always start at 0.

### Freqs chunking

Let `C = D//2`, then:
- `C0 = C - 2*(C//3)` (time)
- `C1 = C//3` (height)
- `C2 = C//3` (width)

For `D=128` → `C=64` → `C0/C1/C2 = 22/21/21`.

## Step 2 Kernel Interface (proposed)

### Inputs

- `x`: `[B, L, H, D]` (`bf16`/`fp16`)
- `freqs`: complex `[max_seq_len, C]` (complex64 preferred)
- `grid_sizes`: `(f, h, w)` (usually `B==1` but keep `B` support)
- `start_frame: int` (0 for `rope_apply`, >0 for `causal_rope_apply`)

### Internal cached tables (per device + freqs identity)

Instead of building `(seq_len, C)` cos/sin, build (and cache) these **float32** tables:

- `cos_f, sin_f`: `[max_seq_len, C0]` (time axis; rows indexed by `f_idx`)
- `cos_h, sin_h`: `[h, C1]` (height axis; rows indexed by `h_idx`)
- `cos_w, sin_w`: `[w, C2]` (width axis; rows indexed by `w_idx`)

Notes:
- Height/width tables do **not** need `max_seq_len` rows; slice to `[:h]`, `[:w]`.
- Cache key should include `freqs.data_ptr()`, device index, and chunk sizes (and/or `freqs.shape`).

### Output semantics

Provide three modes:

1) **In-place (preferred for inference):**
   - kernel writes `x[..., t < seq_len, :]` only
   - returns `x`

2) **Out-of-place, preserve tail (safe default for “functional” semantics):**
   - `out = empty_like(x)`
   - kernel writes prefix into `out`
   - `out[:, seq_len:] = x[:, seq_len:]` (tail copy)
   - returns `out`

3) **Out-of-place, no tail (debug/bench only):**
   - `out = empty_like(x)`
   - kernel writes prefix; tail left uninitialized
   - only valid if downstream never reads tail

The doc in `DeepResearch/2025-12-22/Phase3_01.md` uses `x.clone()` for tail; replace that with “tail copy”.

## Kernel Shape + Launch

Start simple and specialize later.

### Grid

2D grid:
- `pid_bh ∈ [0, B*H)` (batch×head)
- `pid_l ∈ [0, ceil_div(seq_len, BLOCK_L))` (token tiles)

### Work per program

Each program:
- computes `offs_l = pid_l*BLOCK_L + arange(BLOCK_L)`
- computes `f_idx/h_idx/w_idx` for those tokens
- rotates 3 contiguous channel chunks (`C0/C1/C2`)
- masked stores for `offs_l < seq_len`

### Tuning knobs

Start with:
- `BLOCK_L=128` or `256`
- `num_warps=4` or `8`
- `num_stages=2`

If register pressure is high, tile in channel dimension:
- add `BLOCK_C` and loop over channel tiles per chunk.

## Integration Strategy (min risk)

1) Implement kernel + wrapper in a new module:
   - `src/scope/core/kernels/triton_rope_fused.py` (or similar)
2) Add a feature flag (default off initially):
   - `SCOPE_DISABLE_TRITON_ROPE_FUSED=1` to force fallback
   - optionally `SCOPE_TRITON_ROPE_FUSED_STRICT=1` to raise instead of falling back
3) In `rope_apply` / `causal_rope_apply`:
   - fast-path only for `x.is_cuda`, `D==128`, `B==1` initially
   - use in-place mode by default in inference (`not torch.is_grad_enabled()`)
   - otherwise out-of-place + tail copy
4) Keep current Step 0/Phase 2.5 path as fallback.

## Harness Extensions (so we can validate “immediately when it lands”)

### Correctness

Extend `scripts/test_rope_correctness.py` to:
- optionally force fused path via env var (must be set before imports)
- test padded `L` cases that trigger tail logic (e.g. `--pad 56`, `--pad 64`)
- test `start_frame` values `[0, 2]` (already present)
- add a “stress” mode to sweep a few `(f,h,w)` shapes

### Performance

Extend `scripts/bench_triton_rope.py` to benchmark:
- current `rope_apply()` end-to-end (includes wrapper + cache)
- fused wrapper end-to-end (once available)
- direct kernel-only timing (optional)

Add NVTX ranges around the benchmark loops for NSYS capture.

## Acceptance Criteria

- Correctness: match `rope_ref()` within bf16 tolerance (current max err ~0.03125).
- No full-tensor clones for the padded case in inference mode.
- End-to-end `rope_apply/causal_rope_apply` time improves materially vs Phase 2.5/Step 0 wrapper.

## Open Questions

- Can we fuse Q+K RoPE in one kernel call (same indices/tables) to halve wrapper + launch overhead?
- Is `grid_sizes` ever non-uniform across batch for offline runs? (If yes, keep general B loop path.)
- Are there callsites where `x` is `[B, H, L, D]`? If yes, support both layouts in wrapper.
