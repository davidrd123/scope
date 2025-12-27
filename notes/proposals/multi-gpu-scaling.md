# Multi-GPU Scaling for KREA Realtime

> Status: Exploratory / Research
> Date: 2025-12-27
> Confidence: Low (needs investigation)

## Summary

Enable multi-GPU inference for the KREA realtime pipeline to scale FPS beyond single-GPU limits. StreamDiffusionV2 demonstrates this is viable, achieving **58 FPS with 14B models on 4× H100** using pipeline parallelism.

## What We Know

### StreamDiffusionV2 Approach ([paper](https://arxiv.org/abs/2511.07399))

| Strategy | Feasibility | Notes |
|----------|-------------|-------|
| **Pipeline Parallel** | ✅ Proven | Consecutive DiT stages across GPUs |
| **Tensor Parallel** | ❌ Not viable | Spatial activations = prohibitive communication |
| **Sequence Parallel** | ❌ Unstable | Unpredictable latency jitter |

Key mechanisms:
- **SLO-aware batching scheduler** - respects latency guarantees
- **Dynamic block scheduler** - reallocates work based on runtime measurements
- **VAE on separate device** - encode/decode don't compete with transformer

Their usage: `torchrun --nproc_per_node=N`

### Current KREA Pipeline State

The pipeline assumes single device:
```python
# src/scope/core/pipelines/krea_realtime_video/pipeline.py
generator = generator.to(device=device, dtype=dtype)
text_encoder = text_encoder.to(device=device)
vae = vae.to(device=device, dtype=dtype)
```

No distributed primitives currently wired.

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
- VAE is ~25% of pipeline time on B300
- No transformer modification

**Cons:**
- Limited scaling (2 GPUs max benefit)
- Cross-GPU tensor copies for latents

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

1. **VAE offload** - move decode to GPU 1, measure impact
2. If beneficial, try encode offload too
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
