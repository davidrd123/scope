# Multi-GPU Scaling for KREA Realtime

> Status: Exploratory / Research
> Date: 2025-12-27
> Confidence: Low (needs investigation)

## Summary

Enable multi-GPU inference for the KREA realtime pipeline to scale FPS beyond single-GPU limits.

Key framing: **multi-GPU only helps if we can overlap work** (or otherwise avoid turning “more devices” into “more copies + more latency”).

StreamDiffusionV2 reports multi-GPU viability (including **~58 FPS with a 14B model on 4× H100**) using pipeline-parallel style orchestration, but we should treat those numbers as *reported* until we reproduce comparable behavior in our stack.

## What We Know

### StreamDiffusionV2 Approach ([paper](https://arxiv.org/abs/2511.07399))

| Strategy | Feasibility | Notes |
|----------|-------------|-------|
| **Pipeline Parallel** | ✅ Proven | Consecutive DiT stages across GPUs |
| **Tensor Parallel** | ⚠️ Likely poor fit | Their claim: spatial activations make comm prohibitive |
| **Sequence Parallel** | ⚠️ Risky | Their claim: unpredictable latency jitter |

Key mechanisms:
- **SLO-aware batching scheduler** - respects latency guarantees
- **Dynamic block scheduler** - reallocates work based on runtime measurements
- **VAE on separate device** - encode/decode don't compete with transformer

Note: their distributed setup uses `torchrun --nproc_per_node=N`, but our **first prototype** (VAE offload) can likely be done in a single process without `torch.distributed`.

### Current KREA Pipeline State

The pipeline assumes single device:
```python
# src/scope/core/pipelines/krea_realtime_video/pipeline.py
generator = generator.to(device=device, dtype=dtype)
text_encoder = text_encoder.to(device=device)
vae = vae.to(device=device, dtype=dtype)
```

No distributed primitives currently wired.

### Current Single-GPU Baseline (B300, 2025-12-27)

| Config | FPS | Notes |
|--------|-----|-------|
| BF16, no compile | ~19-20 | Stable baseline |
| BF16, `--compile` | ~22-23 | Default recommended |
| BF16, `--compile`, channels_last_3d | ~21.5 | Latest (decode: ~195ms) |

Block-level breakdown (compiled):
- **denoise**: ~65% (dominant)
- **decode**: ~35% (post channels_last_3d optimization)

Multi-GPU would need to beat ~23 FPS to be worthwhile on B300.

### Vendored Resources

We have Blackwell-specific distributed GEMM patterns in `vendored/cutlass-cute/python/CuTeDSL/distributed/`:
- `distributed_gemm_all_reduce_blackwell.py`
- `distributed_gemm_reduce_scatter_blackwell.py`
- `all_reduce_one_shot_lamport.py`

These are kernel-level primitives, not pipeline orchestration.

## What We Don't Know

1. **How StreamDiffusionV2 partitions the DiT** - block-level? layer-level?
2. **Communication patterns** - what tensors cross GPU boundaries, how often?
3. **KV cache distribution** - our streaming KV cache adds complexity
4. **Latency vs throughput tradeoff** - pipeline parallel adds per-frame latency
5. **Heterogeneous GPU support** - mixed GPU types?

## Potential Approaches

### Approach A: VAE Offload (Simplest)

Move VAE encode/decode to a second GPU while transformer runs on primary.

**Pros:**
- Minimal code changes
- VAE decode is **~35% of compiled pipeline time** on B300 (post-optimization)
- No transformer modification

**Cons:**
- Limited scaling (2 GPUs max benefit)
- Cross-GPU tensor copies for latents (and possibly decoded frames)
- **Only a win if we overlap** decode with denoise across chunks (i.e., pipeline stages); sequential offload can be neutral/negative
- Layout considerations: VAE benefits from `channels_last_3d` (`WANVAE_DECODE_CHANNELS_LAST_3D=1`), need to ensure cross-GPU copies preserve this

**Upper bound intuition (if perfectly overlapped, ignoring copy overhead):**
- Speedup is capped by the larger stage share: `speedup <= 1 / max(denoise_share, decode_share)`
- With the current compiled breakdown (~65% denoise / ~35% decode), best-case is ~`1/0.65 ≈ 1.54×` (then subtract copy/jitter costs)

**Complexity:** Low

### Approach B: Pipeline Parallel Transformer (StreamDiffusionV2 style)

Split transformer blocks across GPUs. GPU 0 runs blocks 0-N/2, GPU 1 runs N/2-N.

**Pros:**
- Proven by StreamDiffusionV2
- Near-linear scaling claimed

**Cons:**
- Adds per-frame latency (pipeline fill time)
- KV cache needs distribution strategy
- More complex orchestration

**Complexity:** Medium-High

### Approach C: Temporal Parallelism

Different GPUs work on different frames/chunks in the stream.

**Pros:**
- Natural fit for streaming
- Independent work units

**Cons:**
- Consistency across frames?
- KV cache coherence between GPUs?
- May not reduce single-frame latency

**Complexity:** High (unexplored)

### Approach D: DistriFusion-style Patch Parallel

Split spatial patches across GPUs ([DistriFusion paper](https://arxiv.org/abs/2402.19481)).

**Pros:**
- Works for high-resolution
- Proven for diffusion models

**Cons:**
- Communication at attention layers
- May not help at our resolutions (320×576)
- Designed for image, not video

**Complexity:** High

## Prototype Prereqs / “Measure Before Building” (Especially for VAE Offload)

Before writing orchestration code, measure the two things that decide if Approach A is viable:

1) **Peer-to-peer feasibility and copy cost**
- Does `cuda:0 → cuda:1` support P2P? (NVLink vs PCIe matters a lot.)
- What’s the measured time to copy the relevant latent tensors between devices?

2) **Overlap feasibility**
- Can decode run concurrently with denoise without fighting for CPU scheduling / streams?
- Does the server architecture make it easy to pipeline across chunks (queue latents, decode previous while denoising next)?

## Suggested Investigation Path

### Phase 1: Measure (Before Building)

1. Profile single-GPU to understand where time goes at higher resolutions
2. Identify natural partition points (VAE boundary, transformer block boundaries)
3. Estimate communication costs for candidate splits

### Phase 2: Reference Study

1. Read StreamDiffusionV2 multi-GPU implementation
2. Understand their block scheduler
3. Check if their approach transfers to our architecture

### Phase 3: Prototype (Simplest First)

1. **VAE decode offload** - run decode on GPU 1 *with pipelining* (decode chunk N-1 while denoise chunk N), measure impact
2. If beneficial, try encode offload too (or keep encode on GPU 0 if it’s not dominant)
3. Only then consider transformer partitioning

### Phase 4: Production Hardening (If Warranted)

1. Dynamic load balancing
2. Graceful degradation to single-GPU
3. Heterogeneous GPU support

## Open Questions

- Does our streaming KV cache complicate pipeline parallelism?
- What's the minimum GPU memory per device for useful partitioning?
- Does NVLink vs PCIe matter significantly?
- How does `torch.compile` interact with multi-GPU?
  - *Partial answer:* `max-autotune` modes can hard-abort on SM103 with Triton <3.5.1 (tcgen05 LLVM intrinsic). Guards exist in pipeline.py. Multi-GPU would need similar guards per device.

## Non-Goals (For Now)

- Distributed training (inference only)
- Multi-node scaling (single machine, multiple GPUs)
- Automatic partitioning (manual configuration is fine initially)

## References

- [StreamDiffusionV2 Paper](https://arxiv.org/abs/2511.07399)
- [StreamDiffusionV2 GitHub](https://github.com/chenfengxu714/StreamDiffusionV2)
- [DistriFusion](https://hanlab.mit.edu/blog/distrifusion)
- [vLLM Parallelism Docs](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)
- Vendored: `vendored/cutlass-cute/python/CuTeDSL/distributed/README.md`

## Related Files

- `src/scope/core/pipelines/krea_realtime_video/pipeline.py` - current device handling
- `src/scope/core/pipelines/streamdiffusionv2/` - reference pipeline (no multi-GPU yet)
- `notes/FA4/b300/session-state.md` - current single-GPU performance baseline
