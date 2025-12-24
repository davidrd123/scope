# Kernel dev log (apprentice track)

Purpose
- Track iterative kernel work (Triton/CUDA) for blockwise causal self-attn.
- Capture shapes, correctness, perf, and next steps.
- DeepResearch mapping + recommended sequencing: `notes/FA4/DeepResearch/summary.md`

Current status (2025-12-23)
- **B1: FA4/CUTE score_mod COMPLETE**: **20.8-21.4 FPS** (+0.6-1.2 FPS, +3-6%)
  - Kernel B: 0.540ms (FA4) vs 1.022ms (Triton) = **1.89x faster**
  - Commit: `c19d828`
  - Env var: `SCOPE_KV_BIAS_BACKEND=fa4|triton|flex` (default: triton)
  - Deps: `nvidia-cutlass-dsl>=4.3.3`, `apache-tvm-ffi`
  - Details: `notes/FA4/phase4-fa4-score-mod.md`
  - Test script: `scripts/test_fa4_kv_bias.py`
- **RoPE Phase 3 Step 2 COMPLETE**: v2 fused kernel achieves **20.2 FPS** (fixed 17.8 FPS regression from v1 padding overhead)
  - v2 uses unified 64-pair arange (no per-chunk padding)
  - 2.2x faster than Step 0 baseline in microbench
  - Tag: `v0.3.0-phase3-step2`
  - Details: `notes/FA4/phase3-triton-rope.md`
- Kernel B (KV-cache bias): Triton integrated + pinned for B200; real runs show ~20% FPS uplift at 320x576; profiling shows `p_bias≈89.5%` of **attention-kernel** time (Kernel A is small).
- Fine-grained `self_attn` breakdown (targeting "what moves the needle"): `self_attn_kv_bias` 27.4%, `qkv_projection` 20.8%, `rope_apply` 15.8%, `self_attn_block_mask` 3.4% → kernel work needs to shift toward RoPE/QKV (and/or a much faster Kernel B backend).
- Next bets (highest upside per effort): (1) ~~RoPE fusion~~ **DONE**, (2) ~~FA4/CUTE score_mod~~ **DONE**, (3) Make FA4 the default backend, (4) RoPE Step 3 tuning (BLOCK_M/H, Q+K fusion), (5) Update pyproject.toml with FA4 deps.

Amdahl cheat sheet (based on latest fine-grained profile; assumes `self_attn≈51%` of total step time)
- 2× `self_attn_kv_bias` (~14% of total) → ~7–8% faster end-to-end
- 2× `qkv_projection` (~11% of total) → ~6% faster end-to-end
- 2× `rope_apply` (~8% of total) → ~4% faster end-to-end

Context (why this exists)
- Krea Realtime’s self-attn uses `torch.nn.attention.flex_attention` for:
  - KV recomputation with a block-causal `block_mask`
  - KV-cache sampling with a piecewise-constant `score_mod` bias on past-frame tokens
- Those FlexAttention paths are currently the main blocker to “FA4-class” performance for Krea Realtime self-attn; cross-attn is already on FA4 in other parts of the stack.
- Primary codepaths:
  - Krea self-attn callsites: `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py`
  - Microbench harness for the same semantics: `scripts/bench_blockwise_attn.py`
  - Triton prototype kernel track: `scripts/triton_sdpa.py`
  - FA4 backend integration (for non-Flex paths): `src/scope/core/pipelines/wan2_1/modules/attention.py`

DeepResearch docs (source of "what to build")
- Kernel project breakdown + sequencing: `notes/FA4/DeepResearch/MSU_chat.md`
- Profiling workflow note (Nsight Compute loop / Wafer): `notes/FA4/DeepResearch/wafer.md`
- Repo-specific synthesis + pointers: `notes/FA4/DeepResearch/summary.md`
- External reviews captured verbatim (useful for onboarding other agents): `notes/FA4/DeepResearch/2025-12-22/MSU.md`, `notes/FA4/DeepResearch/2025-12-22/wafer.md`
- RoPE Phase 3 fix: `notes/FA4/DeepResearch/2025-12-22/Phase3_02.md`, `Phase3_03.md`

Vendored CUTE reference (pulled 2025-12-23)
- `vendored/cutlass-cute/python/CuTeDSL/blackwell/fmha.py` (133KB) - Full FMHA for SM100
- `vendored/cutlass-cute/python/CuTeDSL/blackwell/fmha_bwd.py` (131KB) - FMHA backward
- `vendored/cutlass-cute/python/CuTeDSL/utils/fmha_helpers.py` - Helper utilities
- `vendored/cutlass-cute/cute/tutorial/blackwell/` - C++ SM100 MMA tutorials
- `flash-attention.bak/tests/cute/score_mod_definitions.py` - score_mod examples
- `flash-attention.bak/flash_attn/cute/interface.py` - FA4 CUTE interface (62KB)

Design split (deliverables)
- Kernel A (recompute): block-causal attention (dense within block, causal across blocks), no bias.
- Kernel B (sampling): KV-cache attention with a logits bias for “past frames” vs “current block” (score_mod); often `Lq != Lk`.
- Recommended order (B200-first): Kernel B first (best “time-to-win”), then Kernel A.

Krea Realtime “specialize-friendly” settings (Scope defaults)
- Model-fixed (from `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py`):
  - `dim=2048`, `num_heads=16` → `head_dim=128`
  - `FLEX_ATTENTION_ALIGNMENT=128` (most attention lengths are padded to multiples of 128)
- Pipeline-config (from `src/scope/core/pipelines/krea_realtime_video/model.yaml`):
  - `num_frame_per_block=3` (semantic block size in frames)
  - `local_attn_size=6` (KV cache capacity in frames for Scope’s Krea pipeline)
  - `kv_cache_num_frames=3` (context frames used for KV recomputation)
- Resolution-dependent:
  - `frame_seqlen = (height//16) * (width//16)` (see `src/scope/core/pipelines/krea_realtime_video/blocks/recompute_kv_cache.py`)
  - Example: `480x832` → `(480//16)*(832//16)=30*52=1560` tokens per frame

Derived steady-state shapes (with `frame_seqlen=1560`)
| Parameter | Value | Notes |
|---|---:|---|
| `B` | 1 | single stream |
| `H` | 16 | CausalWanModel default |
| `D` | 128 | `2048 // 16` |
| `frame_seqlen` | 1560 | example `480x832` (resolution-dependent) |
| `num_frame_per_block` | 3 | semantic block size (frames) |
| `block_size` | 4680 | `3 * 1560` tokens |
| local window (`Lk`) | 9360 | `local_attn_size=6` → `6 * 1560` |
| global cap (`Lk`) | 32760 | CausalWanModel default when `local_attn_size=-1` (≈21 frames at 1560) |

Bias cutoff (Kernel B)
- In `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py`, the `score_mod` biases keys where:
  - `kv_idx >= frame_seqlen` (exclude first frame)
  - `kv_idx < (Lk - block_size)` (exclude current block)
- For `Lk=9360` and `block_size=4680`: biased region is `kv_idx ∈ [1560, 4680)` (2 frames).

Microbench snapshot (current)
| Path | Shape | Time/call | Notes |
|---|---|---:|---|
| Kernel A (block_mask / recompute) | `L=9360` (6 frames) | 1.239 ms | flex path, `BLOCK_M=128`, `BLOCK_N=64` |
| Kernel B (bias / sampling) | `Lq=4680`, `Lk=9360` | 0.776 ms | flex path, `BLOCK_M=64`, `BLOCK_N=64` |

Near-term plan (codex2 note; worth keeping)
- Integrate the “~10% better” Triton result for the recompute path first, then focus effort on Kernel B (bias sampling).
- Kernel B candidate that avoids FlexAttention:
  - FA4/CUTE exposes `flash_attn_varlen_func(..., score_mod=..., aux_tensors=...)` in the CUTE interface (see `flash-attention.bak/flash_attn/cute/interface.py`).
  - Existing examples to copy/adapt:
    - `flash-attention.bak/tests/cute/score_mod_definitions.py` (uses comparisons + `cute.where`)
    - `flash-attention.bak/tests/cute/test_score_mod_varlen.py` (varlen test harness)
  - Proposed integration point: replace the bias-path `flex_attention(..., score_mod=...)` at `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py:550` with an FA4/CUTE call when available, keeping FlexAttention as fallback.
  - Use `aux_tensors` (e.g., frame_seqlen, cache_current_block_start, log_scale) to avoid recompiling the kernel each call.
- Add path-level counters so other agents can compute `p_bias` vs `p_recompute` directly:
  - `self_attn_block_mask` (recompute / block_mask path)
  - `self_attn_kv_bias` (bias score_mod path)
  - `self_attn_kv_plain` (bias disabled → `attention(...)` fast path)
- Constraints to remember:
  - Score-mod backward (`score_mod_bwd`) only matters if grads are enabled; inference-only can ignore.
  - FA4/CUTE signatures vary across versions; guard for missing kwargs (ties back to “Robust FA4 Wrapper” TODO).

Gotchas (read before optimizing)
- Mixed tiles / alignment matters: if frame/block boundaries cut through kernel tiles, you pay per-element masking/branching.
- Padding can help alignment but can also hide true scaling and add FLOPs; for microbench, use `scripts/bench_blockwise_attn.py --no-pad-q-to-k` when measuring `Lq != Lk`.
- The M4 result here matches the DeepResearch warning: runtime masking in a generic Triton loop can lose to FlexAttention’s compile-time block sparsity (BlockMask).

What to capture next (so other agents can help without re-deriving context)
- Separate the real Krea Realtime hotpath into `p_bias` vs `p_recompute` (fraction of total step time), with shapes per callsite (`B,H,D,Lq,Lk`, dtype, call count).
- For bias path: record the exact “past vs current block” cutoff rule and whether padding is forcing `Lq == Lk` at runtime.
- Mirror those shapes in `scripts/bench_blockwise_attn.py` (both `--pad-q-to-k` and `--no-pad-q-to-k`) so kernel changes have an apples-to-apples benchmark.

How we work together
- You run on hardware + collect benchmarks/profiles.
- I propose kernel changes + tuning based on your results.
- We iterate with short feedback loops (diffs + metrics).

What you need to learn to help me effectively
- Triton basics: tl.load/tl.store, pointer arithmetic, program IDs, launch grids.
- Matmul tiling: BLOCK_M/BLOCK_N/BLOCK_K; num_warps/num_stages.
- Softmax stability: max-subtraction, exp, normalization; FP16/BF16 nuance.
- Mask logic: causal and blockwise causal (prefix) masks.
- Profiling: Nsight Systems/Compute; interpreting memory vs compute limits.

Apprentice path (minimal skills that unblock me)
1) Read/modify Triton kernels and run a unit test.
2) Add a mask and verify against PyTorch/flex_attention on tiny shapes.
3) Sweep block sizes and record latency + throughput.
4) Capture Nsight summary and report top bottlenecks.

Iteration template (copy for each change)
- Date:
- Goal:
- Kernel version/tag:
- Hardware + stack:
  - GPU:
  - CUDA:
  - PyTorch:
  - Triton:
- Shapes:
  - B, Lq, Lk, H, D:
  - frame_seqlen, num_frame_per_block:
- Correctness:
  - max error:
  - mean error:
- Performance:
  - latency (ms):
  - throughput (tokens/s):
  - vs baseline (%):
- Profiler notes:
  - kernel name:
  - occupancy:
  - mem bw:
- Next steps:
- Questions:

Milestones
- M1: Triton SDPA kernel matches PyTorch SDPA on small shapes.
- M2: Blockwise causal mask matches flex_attention output on small shapes.
- M3: KV cache + bias works and matches on small shapes.
- M4: Beats flex_attention self-attn on B200 for target shapes.

Iteration 1 (block_mask baseline)
- Date: 2025-12-19
- Goal: baseline flex_attention block_mask throughput
- Kernel version/tag: flex_attention (PyTorch)
- Hardware + stack:
  - GPU: B200 (user run)
  - CUDA: unknown
  - PyTorch: unknown
  - Triton: unknown
- Shapes:
  - B=1, H=16, L=9360, D=128
  - frame_seqlen=1560, num_frame_per_block=3, frames=6
  - padded_len=9472, alignment=128
- Performance:
  - avg_ms=1.376
  - tokens/s=6,802,961.2
- Notes:
  - Run without `--compile` flag.

Iteration 2 (block_mask + compile)
- Date: 2025-12-19
- Goal: block_mask with torch.compile enabled
- Kernel version/tag: flex_attention (PyTorch)
- Hardware + stack:
  - GPU: B200 (user run)
  - CUDA: unknown
  - PyTorch: unknown
  - Triton: unknown
- Shapes:
  - B=1, H=16, L=9360, D=128
  - frame_seqlen=1560, num_frame_per_block=3, frames=6
  - padded_len=9472, alignment=128
- Performance:
  - avg_ms=1.376
  - tokens/s=6,801,095.2
- Notes:
  - No change vs non-compile, likely already compiled internally by flex_attention.
  - Warning on exit: `cutlass.CACHE_FILE` missing (non-fatal).

Iteration 3 (bias path + compile)
- Date: 2025-12-19
- Goal: kv_cache bias path throughput
- Kernel version/tag: flex_attention (PyTorch)
- Hardware + stack:
  - GPU: B200 (user run)
  - CUDA: unknown
  - PyTorch: unknown
  - Triton: unknown
- Shapes:
  - B=1, H=16, Lq=4680, Lk=9360, D=128
  - frame_seqlen=1560, num_frame_per_block=3
  - padded_len=9472, alignment=128
- Performance:
  - avg_ms=1.540
  - tokens/s=3,039,514.3
- Notes:
  - Autotune picked `triton_flex_attention_4` with BLOCK_M=64, BLOCK_N=64, num_warps=4, num_stages=3.
  - `BLOCKS_ARE_CONTIGUOUS=False` on bias path; kernel differs from block_mask path.
  - Warning on exit: `cutlass.CACHE_FILE` missing (non-fatal).

Iteration 4 (bias path + compile, Lq=Lk)
- Date: 2025-12-19
- Goal: isolate score_mod cost with Lq=Lk
- Kernel version/tag: flex_attention (PyTorch)
- Hardware + stack:
  - GPU: B200 (user run)
  - CUDA: unknown
  - PyTorch: unknown
  - Triton: unknown
- Shapes:
  - B=1, H=16, Lq=9360, Lk=9360, D=128
  - frame_seqlen=1560, num_frame_per_block=3
  - padded_len=9472, alignment=128
- Performance:
  - avg_ms=1.539
  - tokens/s=6,080,537.4
- Notes:
  - Bias path still slower than block_mask (~11-12%).
  - Warning on exit: `cutlass.CACHE_FILE` missing (non-fatal).

Follow-up observation
- The earlier bias run with Lq=4680 used padded Q length = 9472 (same as K), so compute stayed constant.
- Added a `--pad-q-to-k/--no-pad-q-to-k` flag in `scripts/bench_blockwise_attn.py` to measure true Lq scaling.

Iteration 5 (bias path + compile, no pad Q to K)
- Date: 2025-12-19
- Goal: measure bias path with true Lq scaling
- Kernel version/tag: flex_attention (PyTorch)
- Hardware + stack:
  - GPU: B200 (user run)
  - CUDA: unknown
  - PyTorch: unknown
  - Triton: unknown
- Shapes:
  - B=1, H=16, Lq=4680, Lk=9360, D=128
  - frame_seqlen=1560, num_frame_per_block=3
  - padded_q_len=4736, padded_k_len=9472, alignment=128
- Performance:
  - avg_ms=0.779
  - tokens/s=6,009,398.3
- Notes:
  - Autotune picked `triton_flex_attention_4` with BLOCK_M=64, BLOCK_N=64, num_warps=4, num_stages=3.
  - Bias path runtime roughly halves when Q length halves (as expected).
  - Warning on exit: `cutlass.CACHE_FILE` missing (non-fatal).

Iteration 6 (bias path + compile, shorter K)
- Date: 2025-12-19
- Goal: measure K length sensitivity
- Kernel version/tag: flex_attention (PyTorch)
- Hardware + stack:
  - GPU: B200 (user run)
  - CUDA: unknown
  - PyTorch: unknown
  - Triton: unknown
- Shapes:
  - B=1, H=16, Lq=4680, Lk=4680, D=128
  - frame_seqlen=1560, num_frame_per_block=3
  - padded_q_len=4736, padded_k_len=4736, alignment=128
- Performance:
  - avg_ms=0.402
  - tokens/s=11,641,790.8
- Notes:
  - Reducing K length is the biggest win; 6-frame K -> 3-frame K ~2x speedup.
  - Warning on exit: `cutlass.CACHE_FILE` missing (non-fatal).

Iteration 7 (bias path + compile, bias=1.0)
- Date: 2025-12-19
- Goal: estimate score_mod overhead at bias=1.0
- Kernel version/tag: flex_attention (PyTorch)
- Hardware + stack:
  - GPU: B200 (user run)
  - CUDA: unknown
  - PyTorch: unknown
  - Triton: unknown
- Shapes:
  - B=1, H=16, Lq=4680, Lk=9360, D=128
  - frame_seqlen=1560, num_frame_per_block=3
  - padded_q_len=4736, padded_k_len=9472, alignment=128
- Performance:
  - avg_ms=0.779
  - tokens/s=6,005,355.9
- Notes:
  - Bias=1.0 shows same runtime as bias=0.3 in this path; overhead is dominated by K length and kernel shape.
  - Warning on exit: `cutlass.CACHE_FILE` missing (non-fatal).

Iteration 8 (block_mask + compile, frames=3)
- Date: 2025-12-19
- Goal: recompute cost proxy for kv_cache_num_frames=3
- Kernel version/tag: flex_attention (PyTorch)
- Hardware + stack:
  - GPU: B200 (local run)
  - CUDA: unknown
  - PyTorch: unknown
  - Triton: unknown
- Shapes:
  - B=1, H=16, L=4680, D=128
  - frame_seqlen=1560, num_frame_per_block=3, frames=3
  - padded_len=4736, alignment=128
- Performance:
  - avg_ms=0.543
  - tokens/s=8,621,299.4
- Notes:
  - Autotune picked BLOCK_M=128, BLOCK_N=64, num_warps=8 for this shape.

Iteration 9 (block_mask + compile, frames=2)
- Date: 2025-12-19
- Goal: recompute cost proxy for kv_cache_num_frames=2
- Kernel version/tag: flex_attention (PyTorch)
- Hardware + stack:
  - GPU: B200 (local run)
  - CUDA: unknown
  - PyTorch: unknown
  - Triton: unknown
- Shapes:
  - B=1, H=16, L=3120, D=128
  - frame_seqlen=1560, num_frame_per_block=3, frames=2
  - padded_len=3200, alignment=128
- Performance:
  - avg_ms=0.209
  - tokens/s=14,905,521.3

Iteration 10 (block_mask + compile, frames=1)
- Date: 2025-12-19
- Goal: recompute cost proxy for kv_cache_num_frames=1
- Kernel version/tag: flex_attention (PyTorch)
- Hardware + stack:
  - GPU: B200 (local run)
  - CUDA: unknown
  - PyTorch: unknown
  - Triton: unknown
- Shapes:
  - B=1, H=16, L=1560, D=128
  - frame_seqlen=1560, num_frame_per_block=3, frames=1
  - padded_len=1664, alignment=128
- Performance:
  - avg_ms=0.082
  - tokens/s=19,056,815.8

End-to-end sweep (render_timeline, 5s clip, 80 frames @ 16fps)
- Timeline: `notes/FA4/bench-timeline-5s.json` (single segment, 0-5s)
- Settings: 320x576, 4 denoising steps, quantization=None, FA4 enabled, no transitions
- kv_cache_num_frames=3:
  - real_s=66.47
  - warmup_s=8.25
  - wall_from_ts_s=62.52
- kv_cache_num_frames=2:
  - real_s=72.67
  - warmup_s=16.72
  - wall_from_ts_s=69.76
- kv_cache_num_frames=1:
  - real_s=79.08
  - warmup_s=16.46
  - wall_from_ts_s=75.30
- Notes:
  - End-to-end timings are dominated by model load + FA4 compile/warmup; shorter clips obscure steady-state gains.
  - Warmup time varies with kv_cache_num_frames, likely due to extra autotune shapes.

Triton SDPA Kernel Development
------------------------------

M1: Basic SDPA (2025-12-19)
- Status: PASS
- Kernel: scripts/triton_sdpa.py
- Tests:
  - B=1, H=1, L=16, D=16: max_err=0.000000, PASS
  - B=1, H=4, L=64, D=64: max_err=0.000000, PASS
  - B=2, H=8, L=128, D=64: max_err=0.003906, PASS
- Notes: Standard causal mask matches PyTorch SDPA.

M2: Blockwise Causal (2025-12-19)
- Status: PASS
- Tests:
  - B=1, H=1, L=64, D=16, block_size=16: max_err=0.000000, PASS
  - B=1, H=4, L=128, D=64, block_size=32: max_err=0.000977, PASS
  - B=1, H=8, L=256, D=64, block_size=64: max_err=0.000977, PASS
- Notes: Full attention within block, causal across blocks.

M3: KV Cache + Bias (2025-12-19)
- Status: PASS
- Features added:
  - Separate Q/K sequence lengths (Lq can differ from Lk)
  - Bias tensor addition to attention scores (score_mod)
- Test 3a: KV Cache (Lq < Lk, no mask)
  - B=1, H=1, Lq=32, Lk=64, D=16: PASS
  - B=1, H=4, Lq=64, Lk=128, D=64: PASS
  - B=1, H=8, Lq=128, Lk=256, D=64: PASS
- Test 3b: Bias Addition (Lq == Lk)
  - B=1, H=1, Lq=64, Lk=64, D=16: PASS
  - B=1, H=4, Lq=128, Lk=128, D=64: PASS
  - B=2, H=8, Lq=256, Lk=256, D=64: PASS
- Test 3c: KV Cache + Bias + Blockwise Mask
  - B=1, H=1, Lq=32, Lk=64, D=16, block_size=16: PASS
  - B=1, H=4, Lq=64, Lk=128, D=64, block_size=32: PASS
  - B=1, H=8, Lq=128, Lk=256, D=64, block_size=64: PASS
- Baseline performance:
  - Triton SDPA: 0.041 ms (B=1, H=16, L=1024, D=128)
  - PyTorch SDPA: 0.036 ms
  - Ratio: 0.88x (slower, not yet optimized)

M4: Beat flex_attention on B200 (2025-12-22)
- Status: IN PROGRESS
- Goal: Optimize kernel to beat flex_attention for target shapes
- Target shape: B=1, H=16, L=9360, D=128, block_size=4680 (blockwise causal)

Baseline benchmarks:
  - flex_attention (compiled): 1.867 ms  ← target to beat
  - PyTorch SDPA (with mask):  5.427 ms
  - Triton SDPA (initial):     2.295 ms  (0.81x flex, 2.36x PyTorch)

Optimizations attempted:

1. Autotune expansion (108 configs):
   - Added BLOCK_M: [64, 128, 256], BLOCK_N: [32, 64, 128, 256]
   - Added num_warps: [4, 8, 16], num_stages: [2, 3, 4, 5]
   - Best config found: BLOCK_M=64, BLOCK_N=64, num_warps=8, num_stages=2
   - Result: No improvement (still 2.295 ms)

2. Early exit for masked KV blocks:
   - Added logic to limit loop iterations for causal/blockwise modes
   - Initial implementation used tl.minimum() on loop bound
   - PROBLEM DISCOVERED: Blockwise causal is SLOWER than full attention!
     - Full attention (no mask): 1.703 ms
     - Blockwise causal:         2.297 ms
     - Expected: blockwise should be ~25% faster due to sparsity

Sparsity analysis (L=9360, block_size=4680):
  - 2 semantic blocks: block 0 (positions 0-4679), block 1 (positions 4680-9359)
  - Block 0 queries attend to: block 0 only (50% of KV)
  - Block 1 queries attend to: blocks 0+1 (100% of KV)
  - Total tiles (dense):  21609
  - Total tiles (sparse): 16280
  - Theoretical sparsity: 24.7%
  - Expected speedup:     1.33x

Root cause investigation:
  - Triton's range() with runtime bound may not optimize well
  - Mask computation overhead (integer division for block indices)
  - Memory loads still happen even for masked tiles

Current fix attempt:
  - Changed to compile-time loop bound with runtime skip logic
  - Added conditional mask to skip K/V loads for fully-masked tiles
  - Using `skip_tile = kv_tile_start >= max_k_needed` to avoid loads

Key insight:
  flex_attention uses compile-time block sparsity via create_block_mask().
  It generates specialized code that completely skips masked 128x128 blocks.
  Our kernel does masking at runtime, which has overhead.

Next steps:
  - Test if conditional load skipping helps
  - Consider fusing bias computation (log(0.3) inline vs tensor load)
  - Profile to identify bottleneck (compute vs memory vs overhead)
  - May need to generate specialized kernels for known sparsity patterns

3. Skip-tile optimization FAILED (made things worse):
   - Added conditional load masking with skip_tile flag
   - Result: 2.885 ms (SLOWER than 2.297 ms before)
   - Reason: Runtime conditionals add overhead without skipping iterations
   - Triton still executes all loop iterations, just with masked loads

PIVOT RECOMMENDATION (from DeepResearch):
--------------------------------------

The M4 approach of beating flex_attention on Kernel A (block_mask/recompute) is hitting
the fundamental limitation: runtime masking cannot beat compile-time block sparsity.

DeepResearch recommends a two-kernel approach:

1. **Kernel B first (sampling with bias)** - the easier/faster win:
   - Shape: Lq=4680, Lk=9360 (3 query frames, 6 kv frames)
   - Just piecewise constant bias: 0 for current block, -beta for past
   - Three implementation paths:
     - B1: CUTE DSL score_mod (try first - "FA4-ish with tiny tax")
     - B2: Dedicated CUTE kernel variant (most work, best perf)
     - B3: Triton kernel (fallback for iteration speed)

2. **Kernel A second (block_mask/recompute)** - the harder problem:
   - Needs compile-time sparsity like flex_attention's BlockMask
   - Runtime masking has unavoidable overhead

Microbenchmark baselines (B200, 2025-12-22):
  - block_mask (L=9360, 6 frames, local_attn=6): 1.239 ms
  - bias (Lq=4680, Lk=9360, no pad): 0.776 ms
  - flex_attention picks: BLOCK_M=128, BLOCK_N=64 for block_mask
                          BLOCK_M=64, BLOCK_N=64 for bias

Krea realtime fixed shapes (for specialized kernel):
  - B=1, H=16, D=128
  - frame_seqlen=1560 tokens/frame
  - block_size=4680 (3 frames × 1560)
  - L_local=9360 (6 frames with local_attn_size=6)

Expected improvement from Kernel B (Amdahl's law estimate):
  - If FlexAttention is 60-70% of total time
  - And we achieve 2-4x speedup on bias path
  - End-to-end improvement: ~1.5x to 2.1x
  - Update (post real profiling): this was optimistic for 480x832; see “FINE-GRAINED PROFILING” for current bottleneck shares.

KERNEL B: COMPLETE (2025-12-22)
-------------------------------

Triton Kernel B beats flex_attention by 10.6%!

Benchmark results (Lq=4680, Lk=9360, H=16, D=128):
  - flex_attention: 1.144 ms
  - Triton Kernel B: 1.023 ms
  - Speedup: 1.12x

Correctness tests: ALL PASS
  - Small (B=1, H=1, Lq=64, Lk=128, D=16): max_err=0.004
  - Medium (B=1, H=4, Lq=128, Lk=256, D=64): max_err=0.008
  - Krea target (B=1, H=16, Lq=4680, Lk=9360, D=128): max_err=0.002

Implementation:
  - File: scripts/triton_sdpa.py
  - Kernel: kernel_b_bias_attention (with autotune)
  - Wrapper: triton_kernel_b()
  - Test: uv run python scripts/triton_sdpa.py --kernel-b

KERNEL B: INTEGRATED (2025-12-22)
----------------------------------

Triton Kernel B is now the default for the KV-cache bias path.

Integration details:
  - Commit: 2d132dc "Integrate Triton Kernel B into KV-cache attention path"
  - Location: src/scope/core/kernels/triton_attention.py
  - Integration point: causal_model.py:576-586
  - Feature flag: USE_TRITON_KERNEL_B=1 (default on), set =0 to use flex_attention

Key improvements over flex_attention:
  - 10.7% faster (1.023 ms vs 1.144 ms)
  - No padding needed (Triton handles Lq != Lk natively)
  - Simpler code path (no score_mod compilation overhead)

Fallback: If USE_TRITON_KERNEL_B=0, reverts to flex_attention with padding.

UNDERSTANDING CUTE vs TRITON (Codex synthesis)
----------------------------------------------

**CUTE is not a new algorithm, it's a different backend for the same operations.**

Where each backend fits in causal_model.py:

| Path | Current Backend | CUTE Alternative |
|------|-----------------|------------------|
| Kernel B (bias, line 576) | **Triton** (10.7% win) | flash_attn_varlen_func(..., score_mod=..., aux_tensors=...) |
| Kernel A (recompute, line 407/467) | flex_attention + BlockMask | block-sparse tensors or mask_mod (deeper integration) |
| Plain path (bias=1.0, line 606) | FA2/FA4 via attention() | Already on FA path |

Why CUTE for Kernel B could be even faster:
  - Native CUDA (vs Triton's compilation)
  - aux_tensors avoids recompilation per call
  - Supports true Lq != Lk without padding

Why CUTE for Kernel A is harder:
  - Needs compile-time block sparsity (like BlockMask)
  - Would require rebuilding blockwise path around block-sparse tensors
  - Not a simple backend swap

CUTE is blocked on: `cuda-python` package (ModuleNotFoundError: No module named 'cuda')
Fix: `uv sync --group fa4` (but known conflict with PyTorch inductor)

OPEN QUESTIONS (for future sessions)
------------------------------------

1. **Is Kernel A worth pursuing?**
   - Codex2: "Only if profiling shows recompute still dominates a meaningful slice of total time."
   - Sparsity is modest (~25% in 6-frame/3-frame-block case)
   - flex_attention's compile-time BlockMask is a big structural advantage

2. **Measure p_bias vs p_recompute in real runs**
   - Critical for Amdahl's law: which path dominates?
   - If p_bias >> p_recompute, Kernel B win is mostly captured
   - Add profiling counters: self_attn_block_mask, self_attn_kv_bias, self_attn_kv_plain

3. **Codex "two dense calls" trick for Kernel A**
   - With local_attn_size=6 and num_frame_per_block=3, blockwise-causal is just 2 semantic blocks
   - Could compute as two dense FA calls:
     - Block 0: 4680×4680 (self-attention within current block)
     - Block 1: 4680×9360 (cross-attention to all visible history)
   - Route both to FA2/FA4, skip FlexAttention/BlockMask entirely
   - Avoids the "runtime masking loses to compile-time sparsity" trap

4. **B1: CUTE DSL score_mod**
   - Potentially faster than Triton (native CUDA)
   - Reference patterns: flash-attention.bak/tests/cute/score_mod_definitions.py
   - Blocked on cuda-python deps

CURRENT STATE SUMMARY (2025-12-22)
----------------------------------

| Component | Status | Notes |
|-----------|--------|-------|
| Kernel B (Triton) | ✅ INTEGRATED | Default, 10.7% faster |
| B1 (CUTE score_mod) | ⏸️ BLOCKED | Needs cuda-python |
| Kernel A (Triton) | ❌ FAILED | Runtime masking loses to BlockMask |
| Kernel A (two dense calls) | 📋 PROPOSED | Untested, may bypass BlockMask |
| p_bias vs p_recompute | ❓ UNKNOWN | Needs real-run profiling |

Real-world validation (2025-12-22):
  - 320x576: Before ~15 FPS → After ~18 FPS (**20% speedup**)
  - 480x832: ~7.5 FPS (with Triton Kernel B)
  - Note: Better than 10.7% microbench because bias path runs 4x/frame

Resolution scaling reference:
  | Resolution | tokens/frame | vs 320p | Observed FPS |
  |------------|--------------|---------|--------------|
  | 320x576    | 720          | 1.0x    | ~18 FPS      |
  | 384x672    | 1008         | 1.4x    | ~13 FPS      |
  | 416x736    | 1196         | 1.6x    | ~10.4 FPS    |
  | 480x832    | 1560         | 2.1x    | ~7.5 FPS     |

Why 20% FPS gain > Amdahl prediction (Codex synthesis):
  - Amdahl: 10.7% on 69% share → ~7-8% expected
  - Actual: 20% observed
  - Extra gain from: (1) no padding overhead, (2) less flex_attention compile/dispatch cost
  - Conclusion: We got TWO wins - faster kernel + removed Lq→Lk padding waste
  - Update (post fine-grained profiling): kernel B is a smaller share of `self_attn` than the “kernel-only” view; future Amdahl estimates should use the fine-grained breakdown.

PRIORITY RANKING (Codex consensus, 2025-12-22)
----------------------------------------------
Note: superseded by “UPDATED PRIORITY RANKING (post-fine-grained-profiling)” below.

1. **p_bias vs p_recompute accounting** (highest leverage, lowest effort)
   - Turns every decision from "maybe" into Amdahl calculation
   - Add counters: self_attn_block_mask, self_attn_kv_bias, self_attn_kv_plain

2. **B1: FA4/CUTE score_mod for Kernel B**
   - Could be another step-function if FA4 >> Triton on these shapes
   - Blocked on cuda-python deps

3. **Kernel A optimization** (only if p_recompute is large)
   - Runtime masking loses to compile-time BlockMask (proven in M4)
   - Resolution scaling data suggests Kernel A matters more at higher res
   - "Two dense calls" trick may bypass BlockMask entirely

4. **Kernel A via Triton** (lowest priority)
   - Only meaningful if it beats flex_attention in situ
   - Upside capped by p_recompute fraction

PROFILING RESULTS (2025-12-22)
-------------------------------

Real-world profiling with `PROFILE_ATTENTION=1 uv run daydream-scope`:

**Full breakdown:**
| Component | Time | % of Total | Calls | ms/call |
|-----------|------|------------|-------|---------|
| self_attn (outer) | 73.3s | 51.2% | 34720 | 2.11 |
| self_attn_kv_bias | 25.6s | 17.9% | 27840 | 0.92 |
| ffn | 21.4s | 14.9% | 34720 | 0.62 |
| cross_attn | 19.8s | 13.9% | 34720 | 0.57 |
| self_attn_block_mask | 3.0s | 2.1% | 6880 | 0.44 |

NOTE on `% of Total`:
- `self_attn_kv_bias` / `self_attn_block_mask` are **nested inside** `self_attn (outer)`.
- The profiler prints `% of Total` using `sum(_profile_times.values())`, so these percentages **double-count** nested timers.
- Use:
  - **Top-level share**: compare `self_attn (outer)` vs `cross_attn` vs `ffn`.
  - **Within self_attn breakdown**: compare `self_attn_kv_bias` / `self_attn_block_mask` vs `self_attn (outer)`.

**p_bias vs p_recompute (attention kernel time only):**
| Path | % of Attention Kernel Time |
|------|---------------------------|
| p_bias (Kernel B) | **89.5%** |
| p_recompute (Kernel A) | **10.5%** |
| p_plain (FA path) | 0.0% |

**Key insights:**

1. **Kernel A is NOT worth pursuing** - only 10.5% of attention-kernel time
   - Also small within `self_attn (outer)`: `3.0s / 73.3s ≈ 4.1%` of self-attn time in this profile
   - Confirms the Codex recommendation to deprioritize Kernel A

2. **Kernel B optimization was the right target** - 89.5% of attention kernel time
   - Kernel B is also a meaningful slice of self-attn time: `25.6s / 73.3s ≈ 34.9%`

3. **NEW: Non-attention-call cost dominates self-attention**
   - self_attn outer: 73.3s
   - attention calls (measured inside `self_attn_*` blocks): 25.6s + 3.0s = 28.6s
   - **Gap: 44.7s (61% of self_attn time is outside attention calls)**
   - Most likely sources: QKV projection + RMSNorm, RoPE (`causal_rope_apply`), KV-cache maintenance (evict/shift), output projection.
   - Note: `.transpose(...).contiguous()` for Q/K/V is already *inside* `self_attn_kv_bias` / `self_attn_block_mask` timers in the current instrumentation.

4. **Call count ratio confirms architecture understanding**
   - self_attn_kv_bias: 27840 calls (4x per frame, bias path)
   - self_attn_block_mask: 6880 calls (1x per frame, recompute path)
   - Ratio: 4:1 matches expected bias:recompute split

**Profiling methodology:**
- Environment: `PROFILE_ATTENTION=1`
- Uses CUDA events with torch.cuda.synchronize() per call
- Overhead: ~15-17% FPS reduction during profiling (sync breaks async pipeline)
- Results are relative percentages, valid despite overhead

FINE-GRAINED PROFILING (2025-12-22)
------------------------------------

Breakdown within `self_attn` (159.3s outer, 65000 calls):

| Component | Time | % of self_attn | Calls | ms/call |
|-----------|------|----------------|-------|---------|
| **self_attn_kv_bias** (kernel) | 43.7s | 27.4% | 52160 | 0.84 |
| **qkv_projection** | 33.1s | 20.8% | 65000 | 0.51 |
| **rope_apply** | 25.1s | 15.8% | 52160 | 0.48 |
| output_projection | 13.2s | 8.3% | 65000 | 0.20 |
| transpose_contiguous | 8.6s | 5.4% | 52160 | 0.16 |
| self_attn_block_mask | 5.4s | 3.4% | 12840 | 0.42 |
| cache_update | 3.3s | 2.1% | 52160 | 0.06 |

**Key findings:**

1. **QKV + RoPE dominate non-kernel time: 58.2s** (more than attention kernel at 43.7s!)
   - `qkv_projection`: 33.1s (already fused `to_qkv` GEMM + q/k RMSNorm) - 20.8% of self_attn
   - `rope_apply`: 25.1s (`causal_rope_apply`; float64 upcast + allocations) - 15.8% of self_attn

2. **transpose_contiguous is NOT the bottleneck**: only 8.6s (5.4%)
   - Initial hypothesis was wrong - memory layout conversion is cheap
   - The real cost is compute (QKV projection) and RoPE allocations

3. **Optimization targets (by impact):**
   - QKV + RMSNorm optimization: 33.1s potential
   - RoPE fusion/optimization: 25.1s potential
   - These are **compute-bound**, not memory-copy bound

4. **FA4/CUTE upside is real but bounded (in this profile):**
   - FA4/CUTE can replace the FlexAttention `score_mod` path with a varlen backend (supports true `Lq != Lk`, avoids `max(Lq,Lk)` padding tax).
   - FA4/CUTE attention does **not** fuse QKV projection or RoPE; those remain separate kernels unless we build a new fused operator.
   - Amdahl sanity-check: even a 2× speedup of `self_attn_kv_bias` (27.4% share) is only ~16% faster `self_attn` at this resolution/workload.

5. **cache_update is negligible**: 3.3s (2.1%) - no optimization needed

UPDATED PRIORITY RANKING (post-fine-grained-profiling)
-------------------------------------------------------

1. ~~p_bias vs p_recompute accounting~~ → **DONE** (p_bias=89.5%, p_recompute=10.5%)

2. ~~Memory overhead investigation~~ → **DONE** - identified QKV (33.1s) + RoPE (25.1s) as main targets

3. **RoPE optimization** (HIGH PRIORITY)
   - 25.1s (15.8% of self_attn) - significant but harder to optimize
   - `causal_rope_apply` uses view_as_real + flatten + cat - lots of allocations
   - Likely win: fused RoPE kernel (Triton/CUDA) staying in fp32/bf16 (no float64 upcast)

4. **B1: FA4/CUTE score_mod for Kernel B** (HIGH PRIORITY once deps are unblocked)
   - Goal: replace bias-path FlexAttention `score_mod` with FA4/CUTE `flash_attn_varlen_func(score_mod=..., aux_tensors=...)`.
   - Expectation: measure in situ vs current Triton Kernel B; don’t assume >1.2–1.3× without data.
   - Blocked on cuda-python deps - UNBLOCK THIS FIRST.

5. ~~**Kernel A optimization**~~ → **DEPRIORITIZED**
   - Only 5.4s (3.4% of self_attn), 11% of attention kernel time
   - Max possible gain: ~1% of total time
   - Not worth the engineering effort

6. ~~**transpose_contiguous optimization**~~ → **NOT NEEDED**
   - Only 8.6s (5.4%) - initial hypothesis was wrong
   - Memory layout conversion is cheap on B200

ROPE OPTIMIZATION (2025-12-22)
-------------------------------

Status
- ✅ Phase 1: remove float64 upcast + avoid complex multiply (use sin/cos directly) — `78b835c`
- ✅ Phase 2: cache `(cos, sin)` by `(device, dtype, f, h, w, start_frame, c)` — `eb280ba`
- ✅ Phase 2.5: generate RoPE freqs as complex64 (float32) to reduce cast overhead — `eba19ce`
- Notes: `notes/FA4/rope-optimization.md`
- Post-change profiling snapshot:
  - `rope_apply` timer: 0.48ms → 0.42ms per call (~12.5% faster)
  - Share of `self_attn`: 15.8% → ~14.0%
  - Interpretation: real win but not a step-function; remaining cost is dominated by `freqs_i` materialization and memory traffic.

**Why RoPE before FA4:**
- RoPE is 25.1s (15.8% of self_attn) - significant target
- Pure Triton - no new dependencies, no risk to working setup
- FA4/CUTE has cuda-python dep issues - tackle after banking RoPE win

**Current RoPE implementation (post-Phase-2.5):**
- `rope_apply`: src/scope/core/pipelines/krea_realtime_video/modules/model.py:40-67
- `causal_rope_apply`: src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py:279-312
- Problems:
  1. Still materializes a per-token `(cos, sin)` table via `torch.cat([...]).reshape(...)` (on cache miss).
  2. In streaming, `start_frame` is monotonic, so cache reuse is limited across steps; main guaranteed reuse is within a call (Q and K share the same key).
  3. The concatenated table replicates values across `(f, h, w)` structure; this is memory-traffic-heavy for an otherwise tiny amount of math.

**Optimization strategy:**

1. **Phase 2.6: avoid `freqs_i` materialization (next “no-kernel” win)**
   - Reshape `x[:seq_len]` to `(f, h, w, n, c, 2)` and apply RoPE in three channel chunks (temporal/height/width) using broadcast `cos/sin`.
   - Removes `torch.cat + reshape` entirely and avoids reading a fully materialized `(seq_len, c)` table.
   - See `notes/FA4/rope-optimization.md`.

2. **Triton RoPE kernel**
   - Fuse: load Q/K → apply rotation in registers → store
   - No intermediate tensors (view_as_complex, view_as_real, flatten, cat all gone)
   - Stay in float32/bfloat16 (no float64 upcast)

3. **Expected savings:**
   - Baseline was: 25.1s (0.48ms/call × 52160 calls) in the pre-Phase-1 profile
   - Target: cut RoPE by an additional ~30–60% via caching and/or a fused kernel
   - End-to-end impact depends on post-Phase-1 shares; re-profile first

**Reference implementations:**
- vLLM has Triton RoPE: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/rotary_embedding.py
- xFormers has fused RoPE
- FlashAttention repo has examples

Commits this session:
  - 55eef9d: Add Triton Kernel B: 10.7% faster than flex_attention
  - 2d132dc: Integrate Triton Kernel B into KV-cache attention path
  - 4b5edc5: Add development notes and update gitignore
  - 78b835c: Optimize RoPE: remove float64 upcast, use sin/cos directly
  - eb280ba: Add RoPE cos/sin caching (Phase 2)
  - eba19ce: Use float32 (complex64) for RoPE freqs to reduce cast overhead
  - b12e1cb: Add RoPE optimization doc for review

Notes:
  - Added a tiny Kernel B tuning harness: scripts/tune_kernel_b.py (sweep BLOCK_M/BLOCK_N/warps/stages for B200 shape).
  - First run hit OOM on shared memory for BLOCK_M=128/BLOCK_N=128/warps=8/stages=3; harness now skips invalid configs.

Kernel B tuning results (B200, 2025-12-22):
  Shape: B=1 H=16 Lq=4680 Lk=9360 D=128 dtype=bf16
  frame_seqlen=1560 current_block_start=4680 beta=0.3

  | Config | Time |
  |--------|------|
  | BLOCK_M=64 BLOCK_N=64 warps=8 stages=2 | **1.022 ms** (winner) |
  | BLOCK_M=128 BLOCK_N=64 warps=4 stages=2 | 1.054 ms |
  | BLOCK_M=64 BLOCK_N=64 warps=8 stages=3 | 1.068 ms |
  | BLOCK_M=64 BLOCK_N=64 warps=4 stages=3 | 1.196 ms |
  | BLOCK_M=128 BLOCK_N=128 warps=8 stages=2 | 1.254 ms |

  Pattern: smaller tiles (64×64) + 8 warps + 2 stages wins on B200 for this shape.

  Action: Pinned winning config for B200 (compute capability >= 10) in triton_attention.py.
  Commit: d18e7d7

  Nsight Compute snapshot (B200, real shape; see `notes/FA4/phase3-triton-rope.md`):
  - Shape: B=1 H=16 Lq=4680 Lk=9360 D=128 dtype=bf16
  - SM throughput: ~39%
  - Memory throughput: ~12% (DRAM ~0.7%)
  - L2 hit rate: ~87%
  - Regs/thread: 178
  - Dynamic shared mem: ~82 KB
  - Waves/SM: 4
  - Interpretation: Kernel B looks latency/occupancy-limited (not memory-bound); further pure-Triton tuning likely has limited upside vs “new backend” (FA4/CUTE score_mod) or shifting focus to RoPE/QKV.
  - Important: profiling `scripts/triton_sdpa.py --kernel-b` can hit tiny grids (grid size = 1). Use `scripts/tune_kernel_b.py` for meaningful NCU metrics.

RoPE Phase 2.6 attempt (2025-12-22) - ROLLED BACK:
  Idea: Broadcast per-axis cos/sin to avoid (seq_len, C) materialization
  Result: FPS dropped 18.6-18.8 → 17 (regression)
  Lesson: Traded memory traffic for kernel launch overhead - wrong tradeoff on GPU.
          3 separate rotation blocks + concat added more overhead than cache-based approach.
          Complex broadcast patterns less efficient than simple contiguous ops.
  Conclusion: Cache-based Phase 2.5 is sweet spot before Triton fusion.
  See: notes/FA4/rope-optimization.md for full details.

RoPE Phase 3 planning (2025-12-22):
  Status: Plan written, implementation pending
  Approach: 3-step incremental development
    Step 1: Simple Triton kernel with pre-materialized cos/sin (derisk Triton mechanics)
    Step 2: Fuse 3-way lookup into kernel (compute f_idx/h_idx/w_idx in-kernel)
    Step 3: Optimize memory access patterns
  Target file: src/scope/core/kernels/triton_rope.py (new)
  See: notes/FA4/rope-optimization.md for full plan

RoPE Phase 3 Step 0: FA's Existing Rotary Kernel (2025-12-23)
--------------------------------------------------------------

**Status: COMPLETE - Integrated**

Changed approach: Instead of writing a kernel from scratch, first tried FlashAttention's
existing Triton rotary kernel. It works with our 3-way positional encoding because we
pre-compute the merged cos/sin table.

**Integration:**
- Copied: `vendored/rope/flash_attn_triton_rotary.py` → `src/scope/core/kernels/triton_rotary.py`
- Modified: `rope_apply()` and `causal_rope_apply()` to use Triton kernel when available
- Feature flag: `SCOPE_DISABLE_TRITON_ROTARY=1` to force PyTorch fallback

**Benchmark results:**

| Component | Time (ms) | Notes |
|-----------|-----------|-------|
| PyTorch rope_apply | 0.200 | Baseline |
| Direct Triton kernel | 0.101 | **1.98x faster** than PyTorch |
| Integrated rope_apply | 0.183 | Only 8.5% faster (overhead) |

**Overhead breakdown:**

| Step | Time (ms) |
|------|-----------|
| Prep (slice+squeeze) | 0.003 |
| Triton kernel | 0.101 |
| Post (squeeze+cat) | 0.025 |
| **Expected total** | 0.129 |
| **Actual integrated** | 0.183 |
| **Unexplained overhead** | 0.054 |

The 0.054 ms gap comes from Python loop overhead (`grid_sizes.tolist()`, cache lookup per
batch, `freqs.split()`, `torch.stack()`).

**Correctness:**
- FP32: Exact match (diff < 1e-7)
- BF16: Max diff 0.03125 (1 bf16 ULP)
  - 88.3% of values have diff <= 0.001
  - 99.5% have diff <= 0.01
  - FA uses fp32 internally = better numerics

**Next steps:**
- Option A: Remove Python loop overhead by calling kernel once for all batches (blocked by variable grid_sizes)
- Option B: Pre-squeeze cos/sin in cache (minor, ~0.003ms)
- Option C: Fuse Q+K in one kernel call (requires kernel modification)
- Option D: Move to Step 1 - custom kernel with 3-way lookup fusion

**Files changed:**
- `src/scope/core/kernels/triton_rotary.py` (new)
- `src/scope/core/pipelines/krea_realtime_video/modules/model.py` (USE_TRITON_ROTARY, rope_apply)
- `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py` (causal_rope_apply)

**Detailed doc:** `notes/FA4/phase3-triton-rope.md`

B300 (SM103) INVESTIGATION (2025-12-23/24)
-------------------------------------------

### Code Churn Context

**Dec 22-23:** 12 commits in 2 days on causal_model.py (1,493 → 2,093 lines, +40%)
- Multiple attention backends added (FA4, flash, triton, flex)
- RoPE optimizations with temporary regressions (fixed, ended at ~20.2 FPS on B200)
- Profiling infrastructure added
- All performance testing was on **B200**

**Dec 23:** B300 work started after B200 optimizations complete
- B300 immediately showed 8.8 FPS at same settings where B200 hit ~20 FPS

### Current Status: B300 is 2x slower than B200

**Test conditions (same for both):** 320x576 resolution, 4 denoising steps

| GPU | FPS | Notes |
|-----|-----|-------|
| B200 (SM100) | ~20 FPS | Final state after optimizations |
| B300 (SM103) | 8.8 FPS | 2x slower, unexplained |

### What We Know (Facts):

1. B300 runs at 8.8 FPS at 320x576/4 steps
2. B200 runs at ~18 FPS at same settings
3. Attention kernel microbenchmarks show expected speedups (FA4 0.38ms vs Triton 1.6ms)
4. Changing attention backends on B300 does not change FPS

### What We Don't Know:

1. **Has B200 been re-tested after all the B300 work?** - We may have broken something that affects both
2. **Full pipeline profile on B300** - We have attention kernel profiles, but not full frame breakdown
3. **Where the time is going** - If attention is 4x faster but FPS doesn't change, something else dominates

### What Was Tried (on B300):

| Change | Result |
|--------|--------|
| Enable FA4 for main attention | Still 8.8 FPS |
| `SCOPE_KV_CACHE_ATTENTION_BIAS=1.0` | Still 8.8 FPS |
| MAX_FRAME_RATE 8→30 | Still 8.8 FPS |
| Switch FA4↔Triton backend | Still 8.8 FPS |

### Possible Causes (Unverified):

1. **Something broke during B300 work** - Need to re-test B200 baseline
2. **SM103-specific issue** - B300 may have different perf characteristics
3. **Full pipeline bottleneck** - Something outside attention dominates on B300
4. **Profiling needed** - Full frame profile, not just attention kernels

### Environment for B300:

```bash
export TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas  # Required for SM103
```

### Next Steps:

1. **Re-test B200** - Verify baseline still works at ~18 FPS
2. **Full pipeline profile on B300** - Where does time go per frame?
3. **Compare profiles** - B200 vs B300 at same settings
