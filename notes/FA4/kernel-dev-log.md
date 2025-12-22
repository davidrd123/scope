# Kernel dev log (apprentice track)

Purpose
- Track iterative kernel work (Triton/CUDA) for blockwise causal self-attn.
- Capture shapes, correctness, perf, and next steps.
- DeepResearch mapping + recommended sequencing: `notes/FA4/DeepResearch/summary.md`

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

DeepResearch docs (source of “what to build”)
- Kernel project breakdown + sequencing: `notes/FA4/DeepResearch/MSU_chat.md`
- Profiling workflow note (Nsight Compute loop / Wafer): `notes/FA4/DeepResearch/wafer.md`
- Repo-specific synthesis + pointers: `notes/FA4/DeepResearch/summary.md`

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

Next: Integration into causal_model.py, then Kernel A (block_mask/recompute)
