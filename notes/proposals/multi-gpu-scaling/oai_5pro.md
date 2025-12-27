Here’s what I was able to dig up (mostly straight from the StreamDiffusionV2 paper + repo instructions), plus a “how to tell if it *actually* worked” checklist you can apply to KREA.

## What StreamDiffusionV2 actually does for multi‑GPU

### 1) Partitioning: **DiT blocks are split across GPUs (block/layer pipeline parallel)**

In their **Scalable pipeline orchestration** section, they explicitly say the **DiT blocks are partitioned across devices** and run in a pipeline-parallel manner. ([ar5iv][1])

So for your “Do they split block-level or layer-level?” question: the paper’s language points to **block-level partitioning** (i.e., ranges of transformer blocks per rank). ([ar5iv][1])

### 2) Communication pattern: **ring pipeline**, activations hop stage-to-stage each micro-step

They describe a **ring structure**: each device processes an input sequence as a “micro-step” and **transmits results to the next stage**. That implies your main cross‑GPU traffic is **the stage boundary activations** (hidden states / latents) each micro-step. ([ar5iv][1])

They also call out that pipeline-parallel inference adds **inter-stage communication + activation traffic**, keeping things effectively memory-bound unless you schedule carefully. ([ar5iv][1])

### 3) How they avoid stalls: **async comm overlap with two CUDA streams**

They explicitly say each GPU uses **two CUDA streams** (compute + comm) and overlaps transfers with computation to hide latency. ([ar5iv][1])

This is directly relevant to KREA if you do either:

* VAE offload with GPU-to-GPU copies, or
* pipeline parallel transformer (you’ll otherwise get “bubble” stalls).

### 4) Why sequence parallel wasn’t their choice (matches your note)

They frame live-streaming as heavily constrained by per-frame deadlines and jitter, and contrast different parallelism options; the key takeaway is they’re leaning on **pipeline parallel orchestration** + scheduling instead of sequence-parallel attention in this setting. ([ar5iv][1])

### 5) Scheduling: SLO-aware batching + multi-GPU pipeline orchestration

They use an **SLO-aware batching scheduler** that adjusts batch size to meet a target stream rate / per-frame deadline while maximizing utilization. ([ar5iv][1])

Then they extend that idea into the multi‑GPU pipeline setting and combine it with their “stream batch / batch denoising” style approach so they still meet streaming constraints while getting throughput scaling. ([ar5iv][1])

### 6) The “Dynamic DiT block scheduler” is *exactly* about VAE-induced imbalance

This is the most directly useful thing for your **Approach A** and for “where to put the VAE.”

They say static partitioning can be unbalanced because **first/last ranks also handle VAE encode/decode in addition to DiT blocks**, causing pipeline stalls. They add a **lightweight inference-time DiT block scheduler** that reallocates blocks between devices based on measured execution time to minimize per-stage latency / bubbles. ([ar5iv][1])

That’s basically a formal confirmation of your intuition: *even if you only split the transformer, the VAE placement can dominate whether scaling looks good.*

### 7) Claims: heterogenous GPU environments + scaling results

The abstract claims it “scales seamlessly across heterogeneous GPU environments,” supports flexible denoising steps, and reports **58.28 FPS with a 14B model on four H100s** (plus TTFF ~0.5s). ([arXiv][2])

## Repo-level “how they run it”

The public repo/docs show a multi-GPU entrypoint using `torchrun`:

```bash
torchrun --nproc_per_node=2 --master_port=29501 streamv2v/inference_pipe.py ...
# --schedule_block  # optional: enable block scheduling
```

That tells you two important implementation facts:

1. They’re using **multi-process distributed** (not single-process multi-device), and
2. There’s a runtime switch that enables their **block scheduling**. ([Hugging Face][3])

## What this implies for your KREA options

### Approach A: VAE offload (your “simplest” path)

StreamDiffusionV2 basically validates the premise: VAE work can create **stage imbalance**, and they explicitly talk about VAE encode/decode living on edge ranks and causing stalls. ([ar5iv][1])

Practical takeaway for KREA:

* **Offloading VAE to GPU1** can help not only because it’s ~25% time, but because it can smooth utilization on the “main” GPU.
* But you must treat **copies as a first-class cost** (async copies + overlap, ideally peer access / NVLink). Their two-stream overlap idea is a good template even for just VAE offload. ([ar5iv][1])

### Approach B: Pipeline-parallel transformer (StreamDiffusionV2 style)

This is the clearest match:

* Partition by **transformer blocks** across devices. ([ar5iv][1])
* Expect **activation sends** between stages every micro-step. ([ar5iv][1])
* Use **async overlap** to avoid bubble stalls. ([ar5iv][1])
* Add a lightweight **dynamic block partitioner** early, because “VAE on the ends” and other per-rank extras will otherwise kill scaling. ([ar5iv][1])

### KV cache distribution (your open question #3)

From the *paper excerpts I could pull*, they don’t spell out “KV tensors crossing GPUs” in the same explicit way they do for stage outputs. What you *can* safely infer from a blocks-partitioned pipeline is:

* **Each stage “owns” KV for its local blocks** (KV is per-layer), so you *should not* need to ship KV across GPUs *if* you keep the pipeline split aligned with block ranges.

I’m calling that out as an inference, not a confirmed quote, because the lines I retrieved don’t explicitly state “KV never crosses GPUs.” (The paper does mention a rolling KV cache design elsewhere, but I wasn’t able to extract the deeper KV details within the tool limits.) ([ar5iv][1])

## If you’re “not sure this succeeded”: how to verify multi‑GPU use in practice

This catches the most common failure mode: you “launched distributed” but everything still runs on GPU0.

### A) VAE offload success checklist

You should see, per frame:

* GPU0 busy during transformer
* GPU1 busy during encode/decode (or at least during decode)

Concrete checks:

1. **nvidia-smi**: both GPUs show utilization spikes and non-trivial memory allocated during steady-state.
2. Log **tensor devices** at the boundaries:

   * Latents produced by VAE encode are on `cuda:1`
   * Latents consumed by transformer are on `cuda:0`
   * Latents sent to decode move back to `cuda:1`
3. Time breakdown: if decode was ~25% before, you should see GPU0 step time drop, *unless* PCIe copies are eating it.

### B) Pipeline-parallel (torchrun) success checklist

If you do `torchrun --nproc_per_node=N`, you should verify:

1. **N processes actually started**:

   * Each process prints `rank`, `local_rank`, `world_size`
2. **Each process binds to a different GPU**:

   * rank 0 → cuda:0, rank 1 → cuda:1, etc.
3. **Each GPU holds only its partition’s weights**

   * Memory per GPU should be ~1/N of transformer weights (plus overhead), not “full model on every GPU.”
4. Throughput/latency signature:

   * Pipeline parallel often improves **steady-state FPS** more than it improves **single-frame latency** (pipeline fill is real). StreamDiffusionV2 explicitly frames its design around meeting deadlines while scaling FPS. ([ar5iv][1])

### C) One tiny “prove it’s distributed” snippet (drop into your entrypoint)

If you have *any* doubt, add something like:

* print `os.environ["RANK"]`, `LOCAL_RANK`, `WORLD_SIZE`
* `torch.cuda.set_device(local_rank)`
* print `torch.cuda.current_device()`, `torch.cuda.get_device_name()`

If those don’t show distinct device assignments per rank, you’re not really multi-GPU.

## How I’d apply this to KREA with minimum thrash

1. **Do Approach A first**, but implement it “like a pipeline stage”:

   * dedicated CUDA stream for copies
   * non-blocking `.to(device, non_blocking=True)`
   * overlap copies with compute when possible
     (This mirrors their explicit “compute + comm streams” idea.) ([ar5iv][1])

2. If A gives you real headroom, move to **Approach B** with an MVP:

   * static split transformer blocks 50/50
   * send activations between ranks
   * measure bubbles
   * then add **dynamic block scheduling** (because KREA will also have “rank 0 does extra stuff” problems just like they describe). ([ar5iv][1])

3. Treat your vendored CUTLASS “distributed GEMM” bits as **useful later**:

   * StreamDiffusionV2’s core scaling story here is *orchestration + pipeline parallel*, not “tensor parallel inside attention/MLP.” ([ar5iv][1])
     Those kernels might become relevant if you later decide to do **intra-block tensor parallelism** inside a stage, but they’re not the first lever.

If you paste a short log snippet from your last attempt (just the launch command + any rank/device prints + a couple FPS numbers), I can tell you very quickly whether you actually got multi-GPU execution or a silent single-GPU fallback.

[1]: https://ar5iv.org/pdf/2511.07399 "[2511.07399] StreamDiffusionV2: A Streaming System for Dynamic and Interactive Video Generation"
[2]: https://arxiv.org/abs/2511.07399 "[2511.07399] StreamDiffusionV2: A Streaming System for Dynamic and Interactive Video Generation"
[3]: https://huggingface.co/jerryfeng/StreamDiffusionV2?utm_source=chatgpt.com "jerryfeng/StreamDiffusionV2"
