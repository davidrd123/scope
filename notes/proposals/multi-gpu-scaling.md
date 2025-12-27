# Multi-GPU Scaling for KREA Realtime

> Status: Exploratory / Research
> Date: 2025-12-27
> Confidence: Low (needs investigation)

## Summary

Enable multi-GPU inference for the KREA realtime pipeline to scale FPS beyond single-GPU limits.

Key framing: **multi-GPU only helps if we can overlap work** (or otherwise avoid turning “more devices” into “more copies + more latency”).

Before diving in, clarify which “scaling” you mean:
- **Scale a single stream’s FPS** (hard; needs overlap + careful orchestration).
- **Scale total throughput across streams/users** (easy: run independent sessions pinned to GPUs; little/no model partitioning).

This proposal is primarily about **single-stream FPS**, but many of the measurements apply to throughput scaling too.

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

The B300 baseline moved quickly as we fixed decode slow-path issues. Avoid hard-coding numbers here; treat the B300 “truth” as:
- `notes/FA4/b300/session-state.md` (best-known config + block shares + caveats)

As of 2025-12-27, the best-known **quality-preserving** config on B300 is **~30+ FPS** (BF16 + `--compile`) and the block profile looks like:
- `denoise` dominant
- `recompute_kv_cache` non-trivial
- `decode` no longer dominant at the canonical `320x576` resolution (≈ “solved”)

This matters for multi-GPU: **VAE offload only pays if decode is a large share at the target resolution/settings.**

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

### Approach 0: Multi-Session Scaling (Throughput, Not Single-Stream FPS)

Run independent sessions pinned to different GPUs (no model partitioning). This doesn’t make one stream faster, but it’s the lowest-risk way to “scale on multiple GPUs”.

**Pros:**
- Minimal architectural risk (no cross-GPU KV cache / activations)
- Great for multiple users/streams
- Can often be done by routing “session → GPU” at the server layer

**Cons:**
- Doesn’t help a single stream exceed single-GPU FPS
- Still needs a device assignment/routing story (and graceful fallback)

### Approach A: VAE Offload (Simplest)

Move VAE encode/decode to a second GPU while transformer runs on primary.

**Pros:**
- Minimal code changes
- VAE decode can be a large share on some stacks/resolutions (historically ~25–35% on B300 before the decode fast-path fix)
- No transformer modification

**Cons:**
- Limited scaling (2 GPUs max benefit)
- Cross-GPU tensor copies for latents (and possibly decoded frames)
- **Only a win if we overlap** decode with denoise across chunks (i.e., pipeline stages); sequential offload can be neutral/negative
- Layout considerations: VAE benefits from `channels_last_3d` (`WANVAE_DECODE_CHANNELS_LAST_3D=1`), need to ensure cross-GPU copies preserve this

**Upper bound intuition (if perfectly overlapped, ignoring copy overhead):**
- Speedup is capped by the larger stage share: `speedup <= 1 / max(denoise_share, decode_share)`
- On the current best-known B300 config, decode is no longer dominant at `320x576`, so the theoretical bound is much smaller (and copy/jitter can erase it).

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

## Decision Gates (When to Stop vs Build)

Multi-GPU work can balloon quickly. These gates prevent “endless exploration”:

### Gate 0: Define the objective

Pick one:
- **Lower latency** (time-to-first-frame / time-to-new-prompt)
- **Higher steady-state FPS**
- **Higher total throughput (multi-session)**

Each objective implies different architecture choices and success metrics.

### Gate 1: Amdahl sanity check (is there even headroom?)

If a candidate stage is only `p` share of steady-state time, the absolute best-case speedup from “perfectly overlapping” it is:

`speedup <= 1 / (1 - p)`

Example: if decode is 15% share, perfect overlap caps at ~1.18× *before* copy overhead/jitter.

If the theoretical bound is small, don’t build it unless it also improves latency/UX.

### Gate 2: Measured P2P + overlap

Proceed only if:
- P2P is available (or copy time is still comfortably below the saved compute time), and
- you can demonstrate *real overlap* with a microbenchmark (not just “two devices exist”).

### Gate 3: End-to-end win at equal quality

Only “count” wins that:
- preserve output quality (BF16 baseline), and
- improve an end-to-end metric (steady FPS or latency), not just a microbench.

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

## Testing Program (Concrete)

Treat this as an experiment ladder: each stage produces artifacts (logs, profiles) and has a pass/fail.

### Test 1: Hardware topology + P2P

Artifacts:
- `nvidia-smi topo -m` output (NVLink vs PCIe)
- `torch.cuda.can_access_peer(0, 1)` (and enabling peer access if needed)

Pass:
- P2P is supported and enabled for the target device pair.

### Test 2: Copy microbench (latents and/or frames)

Measure end-to-end copy time for the *actual tensors you’d move*:
- latents: GPU0 → GPU1 (decode input)
- frames: GPU1 → GPU0 (if required by the server plumbing)

Pass:
- Copy time is comfortably below the compute you’re trying to overlap (otherwise you just moved the bottleneck).

### Test 3: Overlap microbench

Goal: prove we can overlap “denoise-like work” on GPU0 with “decode-like work” on GPU1.

Approach:
- Run a representative denoise chunk on GPU0 in one thread/stream
- Run a representative decode call on GPU1 in another thread/stream
- Use CUDA events to measure wall time vs per-device time

Pass:
- Wall time is close to `max(t_denoise, t_decode)` (not `t_denoise + t_decode`).

### Test 4: End-to-end offload prototype (Approach A)

Minimal implementation idea:
- GPU0 produces latents for chunk N and enqueues them
- A background worker on GPU1 decodes chunk N-1 concurrently
- The main loop consumes decoded frames and continues streaming

Pass:
- Measured end-to-end improvement on the canonical harness (and a “looks correct for 1–2 minutes” quality sanity check).

### Test 5: Stress + jitter

Measure jitter under:
- playlist nav events (hard/soft cuts, transitions)
- variable prompt lengths
- sustained run (minutes)

Pass:
- No periodic stalls, no obvious runaway latency, and stable memory usage.

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
- `notes/FA4/b300/session-state.md` - current B300 single-GPU performance baseline (truth source)
