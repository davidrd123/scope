# B200 bringup log (FA4 + flex_attention)

Navigation
- Current “what to run today”: `notes/FA4/b200/session-state.md`
- Log one-change trials: `notes/FA4/b200/experiments.md`
- Overall map: `notes/FA4/optimization-map.md`

Context
- Goal: get krea-realtime-video to run on Blackwell (B200) while validating FA4/CUTE attention path.
- Prior failure on B300 was `ptxas fatal: Value 'sm_103a' is not defined for option 'gpu-name'` from Triton in flex_attention.

Observations
- On B300, flex_attention (Triton) compile failed with sm_103a; this is a toolchain/ptxas target mismatch rather than a model bug.
- FA4/CUTE exists in local `flash-attention/flash_attn/cute/` and is gated on SM100 in `flash_attn.cute`.
- Attention code now logs availability for FA2/FA3/FA4 and selects FA4 on SM100.
- kv_cache_attention_bias is now set from config before warmup so warmup path matches runtime bias.
- Current run on B200 appears to proceed past model/LoRA load; no immediate ptxas errors observed so far.

Hypotheses
- B200 is SM100; FA4/CUTE targets SM100 and should be a better fit than SM103.
- Triton flex_attention kernels should compile on SM100 if the installed PyTorch/Triton/ptxas stack includes SM100 support.
- If flex_attention still fails, it will likely be a backend/toolchain issue (Triton target support), not a model config issue.

What I am trying
- Continue the same render_timeline run on B200 and watch for: 
  - `flash attn 4 (cute) available True`
  - `Using Flash Attention 4 (CUTE) for Blackwell/SM100` (one-time log)
  - absence of `ptxas fatal ... sm_103a` errors.

What to verify next
- Confirm GPU capability via `torch.cuda.get_device_capability()` is (10, 0) on B200.
- Confirm `flash_attn.cute` import path is active (local repo path in `flash_attn.__path__`).
- If a failure still occurs, capture the exact stack trace and identify whether it is:
  - flex_attention/Triton compile, or
  - flash-attn-cute import/ABI mismatch.

Fallback options if B200 still fails
- Add a runtime flag to disable flex_attention and use SDPA fallback for block masks.
- Force `kv_cache_attention_bias=1.0` to avoid the bias path that uses flex_attention.
- Use FA2 or FA3 if FA4 is unavailable, but expect lower performance.

2025-12-19: B200 run status
- FA4/CUTE initialized and used: `Using Flash Attention 4 (CUTE) for Blackwell/SM100+` appeared in logs.
- Prior error on B200 (now resolved): `flash_attn_varlen_func() got an unexpected keyword argument 'deterministic'` from FA4 path. This suggests the CUTE API signature differs across versions; may need a guard if it reappears.
- Profiling (PROFILE_ATTENTION=1) shows self-attn dominates compute:
  - self_attn ~59% (~6.67 ms/call)
  - cross_attn ~25% (~2.84 ms/call)
  - ffn ~16% (~1.77 ms/call)
- Cross-attn is fast and already on FA4; self-attn is the bottleneck and appears to be on flex_attention (block-mask path).

Implications
- Biggest speedups would come from replacing/optimizing self-attn; cross-attn is not the limiting factor.
- Swapping self-attn to FA4 is non-trivial because FA4 does not accept block masks; would require mask re-encoding or block-packing.
- `torch.compile` may help non-attention parts but won’t change the self-attn backend selection.

Architecture notes: where self-attn uses flex_attention
- Self-attn path (krea realtime, `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py`):
```
if kv_cache is None or block_mask is not None:
    # blockwise causal mask -> flex_attention(block_mask=...)
else:
    if kv_cache_attention_bias != 1.0:
        # flex_attention(score_mod=...) to bias past frames
    else:
        # attention(...) -> FA4/FA2/SDPA depending on backend
```
- Blockwise causal mask is built via `create_block_mask()` using an `ends` array:
  - For query token q in block j: allow all keys kv with `kv_idx < end_of_block_j`
  - This is causal across blocks but *non-causal within a block* (full block attention).

Training vs inference masking
- `_forward_train` builds `self.block_mask` and passes it into blocks.
- `_forward_inference` does not build a mask; it relies on kv_cache.
- `RecomputeKVCacheBlock` temporarily sets `generator.model.block_mask` to rebuild cache, then clears it.

KV cache + bias interaction
- Presets default `kv_cache_attention_bias=0.3`, so inference self-attn uses the bias path -> flex_attention.
- Setting bias to `1.0` skips the score_mod path and calls `attention(...)`, which can use FA4.

Cross-attn vs self-attn backends
- Cross-attn calls `flash_attention(...)` (FA4 on SM100) directly in `krea_realtime_video/modules/model.py`.
- Self-attn is split across the block-mask path and the kv_cache path above.

Why FA4 for self-attn is hard (no custom kernel)
- FA4 supports causal and windowed attention, but not arbitrary block masks or score_mod.
- To keep blockwise causal (full attention within block + all prior blocks), FA4 would need either:
  - a different masking API, or
  - a data layout trick that emulates the mask without duplicating keys/values.
- Without changing the mask, the only easy FA4 path is setting `kv_cache_attention_bias=1.0` (behavior change).

Custom kernel outline (what it would entail)
- Semantics to match:
  - Blockwise causal mask: full attention inside each block, prefix attention to all previous blocks.
  - Optional local_attn_size and sink_size logic.
  - Optional kv_cache_attention_bias (score_mod on past frames).
- API surface:
  - Inputs: Q/K/V in [B, L, H, D] (bf16/fp16), cu_seqlens for varlen.
  - Block size = frame_seqlen * num_frame_per_block.
  - KV cache pointers + current_start offsets (avoid copies).
- Kernel design:
  - Block-sparse/prefix attention: for each Q block, attend K/V up to end_of_block.
  - Fused softmax + scale + output writeback (flash-attn style).
  - Mask logic integrated in block scheduler (skip masked tiles).
  - Bias support: add log_scale to scores for past frames in a specific kv_idx range.
- Implementation path:
  - CUTE/CUTLASS kernel (SM100) or Triton block-sparse kernel.
  - PyTorch extension (C++/CUDA) with a forward API; backward only if training.
  - Runtime dispatch by GPU capability; fallback to flex_attention.
- Engineering cost:
  - Prototype (forward-only, no bias/local window): 1-2 weeks for a kernel engineer.
  - Full parity (bias + local window + training backward): 4-8+ weeks.
- Requires tight benchmarking and numerical validation vs SDPA/flex_attention.

Learning kernel engineering (roadmap)
- Prereqs:
  - Solid C++ and CUDA fundamentals (memory hierarchy, warps, shared memory, occupancy).
  - Linear algebra + numerical stability (softmax, scaling, fp16/bf16 pitfalls).
  - Profiling skills (Nsight Systems/Compute, roofline thinking).
- Suggested learning order:
  1) CUDA basics: vector add, reductions, prefix-sum, layernorm.
  2) Matmul kernels: tiling, shared memory, Tensor Cores, epilogue fusion.
  3) Attention kernels: softmax stability, causal masks, block-sparse patterns.
  4) Performance tuning: kernel launch overhead, memory coalescing, register pressure.
  5) Integration: PyTorch custom ops, dispatch by device capability, testing.
- Tools/libraries to study:
  - Triton (rapid prototyping, easier to iterate).
  - CUTLASS / CUTE (production-grade, used by FA4; higher complexity).
  - FlashAttention codebase (real-world attention kernel patterns).
- Time investment (realistic):
  - 2-4 weeks: comfortable with CUDA basics + profiling simple kernels.
  - 2-3 months: able to write and optimize matmul/softmax kernels.
  - 4-6+ months: capable of custom attention kernels that rival FA2/FA4.

Mini-curriculum (background path toward blockwise attention kernel)
- Phase 0 (week 0-1): Baseline + profiling
  - Goal: become fluent in profiling and reproduce the current bottleneck.
  - Tasks:
    - Run `PROFILE_ATTENTION=1` and log self_attn/cross_attn/ffn timings.
    - Use Nsight Systems to capture one render block and identify top kernels.
    - Build a tiny microbenchmark that calls current self-attn with a small batch.
  - Deliverable: baseline timing table + annotated profile trace.
- Phase 1 (week 1-2): Triton basics
  - Goal: write a working attention-style kernel in Triton.
  - Tasks:
    - Implement a tiled matmul in Triton and benchmark vs torch.matmul.
    - Implement softmax with numerically stable max-subtraction.
    - Implement SDPA for fixed shapes (no mask) and validate outputs.
  - Deliverable: Triton SDPA kernel with unit tests vs PyTorch.
- Phase 2 (week 3-4): Masks + blockwise causal in Triton
  - Goal: express the blockwise causal mask in kernel logic.
  - Tasks:
    - Implement causal mask in Triton SDPA.
    - Add blockwise causal logic using `ends` array (same as create_block_mask).
    - Validate vs current flex_attention outputs on small shapes.
  - Deliverable: Triton blockwise attention kernel that matches flex_attention numerics.
- Phase 3 (week 5-6): KV cache + bias
  - Goal: match inference behavior with kv_cache and kv_cache_attention_bias.
  - Tasks:
    - Accept KV cache pointers and current_start offsets.
    - Implement bias region (`score_mod`) as log_scale on past frames.
    - Benchmark the kernel on 1–2 realistic sizes.
  - Deliverable: Triton kernel that supports KV cache + bias with correctness tests.
- Phase 4 (week 7-10): CUTE/CUTLASS or production hardening
  - Goal: get production-level performance on SM100.
  - Two paths:
    - Triton hardening: autotuning, block sizes, better memory layout.
    - CUTE/CUTLASS port: study FA4 code, implement a minimal blockwise kernel.
  - Deliverable: kernel that beats flex_attention self-attn in wall time.
- Phase 5 (week 11-12): Integration into Scope
  - Goal: safe integration + fallback.
  - Tasks:
    - Runtime dispatch by GPU capability (SM100 -> custom kernel).
    - Fallback to flex_attention or SDPA on other GPUs.
    - Add correctness checks (max/mean error) and perf regression tests.
  - Deliverable: optional backend selection with test coverage and benchmarks.

Recommended reading (names only)
- CUDA Programming Guide (memory hierarchy, occupancy).
- Triton tutorials and language reference.
- CUTLASS/CUTE examples (matmul, softmax).
- FlashAttention v2/v3/v4 source code for real-world kernel patterns.

Project-aligned micro-milestones
- M1: match flex_attention numerics for a tiny blockwise mask.
- M2: add kv_cache + bias correctness (small shapes).
- M3: beat flex_attention on B200 for self-attn microbenchmark.
- M4: integrate and show end-to-end FPS improvement on timeline render.
