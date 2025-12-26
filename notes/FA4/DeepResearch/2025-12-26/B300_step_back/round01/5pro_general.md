A few “outside” angles jump out from your log. I’ll keep these in the same one-change, reproducible spirit you’re already using.

## 0) First, the architecture sanity check that explains a lot of the weirdness

If any of your confusion is still about “FA4 works on B200/B300 but not on 5090” (or “Blackwell but different behavior”), the clean explanation is: they are different CUDA compute capabilities even though the marketing name overlaps.

* B200 is compute capability **10.0**, B300 is **10.3**. ([NVIDIA Developer][1])
* GeForce RTX 5090 is compute capability **12.0 (sm_120)**. ([NVIDIA Developer][1])

Starting with CUDA 12.9, NVIDIA also introduced “family” vs “architecture-specific” compile targets (the `f` and `a` suffix idea). Family targets are designed to be forward-compatible within a family (for example 10.x), while `a` targets are architecture-specific and not forward-compatible. ([NVIDIA Developer][2])

This matters because it’s very easy for kernels to be built for 10.x (B200/B300) and simply not exist for 12.0 yet, or to require an arch-specific target (`sm_103a` etc) that breaks portability. ([NVIDIA Developer][2])

That’s the “why” behind a lot of “same branding, different fast paths.”

## 1) The “last 5%” lever: try `torch.compile` mode `max-autotune-no-cudagraphs`

You already discovered that `reduce-overhead` is unstable on SM103 because of the CUDAGraph overwritten-output error. That is a known failure mode in PyTorch’s CUDA graphs story. ([GitHub][3])

The thing I would try next is **max-autotune without CUDA graphs**:

* PyTorch explicitly documents `mode="max-autotune-no-cudagraphs"` as “similar to max-autotune but without CUDA graphs.” ([PyTorch Docs][4])
* It also documents that `max-autotune` enables CUDA graphs by default on GPU. ([PyTorch Docs][4])
  So this mode is basically tailored to your exact situation: you want better kernel selection (GEMMs, convs) but CUDA graphs are currently toxic in your workload.

### Experiment card suggestion

**Change (one thing):** `SCOPE_TORCH_COMPILE_MODE=default` → `max-autotune-no-cudagraphs`

**Command:**

```bash
SCOPE_TORCH_COMPILE_MODE=max-autotune-no-cudagraphs \
SCOPE_KV_BIAS_BACKEND=fa4 \
DISABLE_FLEX_ATTENTION_COMPILE=1 \
WANVAE_STREAM_DECODE_MODE=chunk \
TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas \
.venv-b300-cu130-decode/bin/python scripts/profile_krea_pipeline_blocks.py \
  --height 320 --width 576 \
  --iters 6 --skip 2 \
  --kv-cache-attention-bias 0.3 \
  --quantization none \
  --compile \
  --cudnn-benchmark
```

**What I’d watch for:**

* If you get even a small gain, this might be your cleanest path from ~22.8 FPS to 24+ without touching correctness-sensitive code.
* If it regresses, it’s still valuable: it suggests your remaining bottleneck isn’t inductor’s kernel choices but something more structural (layout/copies, attention glue, decode).

## 2) Treat cross-attn as “unoptimized by your KV-bias work” and go after it directly

Your KV-bias work is mostly buying down a slice of *self-attn* cost. Cross-attn is still sitting there, and in a DiT-ish diffusion transformer it can be surprisingly expensive.

Two very practical cross-attn tactics that often outperform “RoPE fusion” in ROI:

### 2a) Force cuDNN SDPA for the attention that can use it (especially cross-attn)

PyTorch exposes explicit toggles to enable/disable cuDNN SDPA kernels. ([PyTorch Docs][5])
It also provides `can_use_cudnn_attention(params, debug=True)` so you can tell why it did not pick cuDNN for a given shape/stride. ([PyTorch Docs][5])

If your cross-attn is going through `F.scaled_dot_product_attention`, you can often force a backend for a run and see if it wins.

**Important gotcha:** the old `torch.backends.cuda.sdp_kernel()` context manager is deprecated, and putting it inside compiled regions can cause Dynamo issues. Newer guidance is to set the backend choice outside the compiled region or use the newer context manager. ([GitHub][6])

**One-change experiment idea:** globally enable cuDNN SDPA and disable flash/mem-efficient SDPA for a run (only if you’re sure your attention code path uses SDPA for the parts you care about).

### 2b) Cache cross-attn K/V projections of the text/context (if you aren’t already)

This is one of those “feels too simple to matter” ideas that sometimes matters a lot in practice, especially when:

* you do multiple denoise steps
* you do KV-cache recomputation passes
* the conditioning context is constant across those steps (text prompt embeddings, style embeddings, etc.)

**The premise:** for cross-attn, the *context* keys and values do not change across denoise steps. You can compute per-block `K_ctx, V_ctx` once and reuse them for every step.

**Why it might be big in your setup:** your pipeline does recompute work and runs multiple transformer passes per frame. Even modest savings per cross-attn call compound.

**Minimal “one change” version:** implement a per-forward cache keyed by `(block_id, ctx_ptr/version, dtype, device, maybe batch size)` and reuse it across steps within a single generation call. That keeps the caching scope conservative (no tricky invalidation across runs).

Pitfalls to explicitly guard:

* CFG (conditional/unconditional contexts are different)
* multiple prompts or prompt changes mid-stream
* attention mask differences (if any)

## 3) Your biggest win so far is a clue: hunt more “Conv3d slow path” cases

Your Conv3d patch-embed rewrite was enormous. That’s a pattern, not a one-off.

I would apply the same lens to the VAE and any other 3D convolutions:

### 3a) Systematically search for `kernel_size[0]==1` Conv3d in decode and rewrite to per-frame Conv2d

Exactly what you did for patch embedding, but applied to decode stacks.

This is often where the “mysterious copy/fill storm” comes from: backend uses an algorithm that requires big implicit transforms/reformats.

### 3b) Memory format experiment: channels_last_3d

For 3D conv-heavy sections, it’s often worth a one-change test:

* set VAE decode modules and inputs to `channels_last_3d`
* measure decode wall time and any extra `aten::contiguous`/`aten::to` that might appear (sometimes it helps, sometimes it makes things worse)

This is worth trying because you already proved you’re fighting backend slow paths, not purely raw FLOPs.

## 4) Make the compile modes work for you, not against you

You hit the CUDAGraph overwrite error in `reduce-overhead`. There are a couple of directions here:

### 4a) If you still want cudagraphs, do it manually (one region, one lifetime)

Instead of Dynamo attempting cudagraph capture across a lot of stateful Python glue, manually capture a very tight region with:

* fixed shapes
* preallocated inputs/outputs
* explicit `copy_()` into static buffers

This can often avoid the “overwritten by subsequent run” problem because you control exactly what tensors live across iterations. The PyTorch docs and issues around cudagraph tree semantics give hints on where the trapdoors are. ([PyTorch Docs][7])

I would only do this after you try `max-autotune-no-cudagraphs`, because manual cudagraphs are fiddly.

### 4b) Use `torch._inductor.list_mode_options()` to see what modes actually flip

PyTorch explicitly recommends this for understanding mode effects. ([PyTorch Docs][4])
It’s a good way to keep your experiment log honest: you can record exactly what changed.

## 5) FP8 is “garbage output” now, but you can still extract a real path forward

Right now you’re treating FP8 as perf-only, which is the correct operational stance. But debugging it can be done in a way that produces useful information quickly:

### 5a) Try weight-only FP8 (leave activations in BF16)

A lot of “everything turned gray/noisy” failures come from activation quantization or scaling errors, not from weights alone.

TorchAO weight-only PTQ explicitly supports FP8 (E4M3FN and E5M2) for weights, ignores activations, and supports per-layer skips. ([NVIDIA Docs][8])

So one clean diagnostic is:

* weight-only FP8 on transformer linear layers
* skip VAE, norms, and any delicate layers
* compare quality

If weight-only looks sane but your current FP8 mode does not, you immediately know where the bug class lives (activation path or scaling path).

### 5b) Use E5M2 as a stability probe

Even if it’s slower, it can be a good “is this precision fundamentally viable” test because exponent range is larger. (If E5M2 works and E4M3FN does not, that’s a strong hint that you’re saturating/underflowing somewhere, or scaling is wrong.)

## 6) Longer-term: ThunderKittens is relevant, but treat it as a controlled integration project

If you decide to go for a big swing, ThunderKittens is legitimately aligned with what you are doing: attention and GEMM kernels designed to exploit Blackwell mechanisms (warp specialization, tensor memory, CTA pairs). ([Hazy Research][9])

Two sober notes:

* The TK blog data is heavily B200-centered, and you are on B300 (CC 10.3), so you should assume “needs verification,” not “drop-in win.” ([Hazy Research][9])
* It’s a real integration effort. I’d only do it after you’ve extracted the lower-risk gains above (compile mode, cross-attn backend, decode slow paths).

## If I had to pick only three next cards

1. **`max-autotune-no-cudagraphs`** compile mode (best chance at the last 5% without correctness risk). ([PyTorch Docs][4])
2. **Force/benchmark cuDNN SDPA** for cross-attn where possible, with `can_use_cudnn_attention(..., debug=True)` logging so failures are explained, not mysterious. ([PyTorch Docs][5])
3. **Conv3d kernel_size[0]==1 hunt in VAE decode** with the same Conv2d-per-frame rewrite pattern that already paid off.

If you want, paste one representative “ops profile with stack” snippet from the current best run (after patch_embed + rms_norm fix). I can point to the next 1–2 call stacks that are most likely to be “another patch_embed-style win,” rather than a bunch of incremental micro-optimizations.

[1]: https://developer.nvidia.com/cuda/gpus "CUDA GPU Compute Capability | NVIDIA Developer"
[2]: https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/ "NVIDIA Blackwell and NVIDIA CUDA 12.9 Introduce Family-Specific Architecture Features | NVIDIA Technical Blog"
[3]: https://github.com/pytorch/pytorch/issues/144961?utm_source=chatgpt.com "CUDAGraph outputs will be overwritten by a subsequent ..."
[4]: https://docs.pytorch.org/docs/stable/generated/torch.compile.html?utm_source=chatgpt.com "torch.compile — PyTorch 2.9 documentation"
[5]: https://docs.pytorch.org/docs/stable/backends.html?utm_source=chatgpt.com "torch.backends"
[6]: https://github.com/pytorch/pytorch/issues/127523?utm_source=chatgpt.com "SDPA memory efficient and flash attention kernels don't ..."
[7]: https://docs.pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html?utm_source=chatgpt.com "CUDAGraph Trees"
[8]: https://docs.nvidia.com/tao/tao-toolkit/latest/text/tao_quant/backends_torchao.html "7. TorchAO Backend (Weight-Only PTQ) — Tao Toolkit"
[9]: https://hazyresearch.stanford.edu/blog/2025-03-15-tk-blackwell "ThunderKittens Now on Blackwells! · Hazy Research"
