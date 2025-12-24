If your target is “higher resolution, stable interactive preview, 15 FPS is fine,” then the picture shifts a lot. You stop caring about squeezing the last 2x out of latency and start caring about two things:

1. **Per-frame memory headroom** (activations + KV cache + VAE) so you can raise resolution without OOM or aggressive offload
2. **A mature attention backend on the target GPU**, because attention dominates at higher token counts

### First, a key correction about Krea vs Wan 2.2

The open-source **Krea Realtime 14B repo is explicitly distilled from Wan 2.1 14B**, not Wan 2.2. ([GitHub][1])

So if you mean “I want Wan 2.2 14B quality specifically,” you have two realistic paths:

* **Interactive path:** use Krea Realtime (Wan 2.1 distilled) for live work, then optionally “finalize” with a slower Wan 2.2 pipeline offline
* **All-in Wan 2.2 path:** use Wan 2.2 directly and invest in acceleration (distilled variants, sparse attention, caching, multi-GPU sharding). That is usually not real-time out of the box. The Wan 2.2 repo is structured like an offline generator, not a streaming real-time system. ([GitHub][2])

### The fastest way to higher res at ~15 FPS is often “bigger single GPU,” not multi-GPU

Multi-GPU sharding can help, but it is engineering-heavy and it can increase jitter. If your main pain is “I cannot push resolution,” a **B200-class GPU is compelling primarily because it gives you a lot more memory headroom** (HGX B200 docs cite ~180GB HBM3e per GPU; datasheets cite ~8 TB/s bandwidth class). ([NVIDIA Images][3])

That extra memory is exactly what you want when video models start paying huge per-frame costs (KV cache, larger latent grids, bigger VAE decode buffers).

### On the attention-kernel question: your assumption is flipped for Krea

You said: “H100 is probably FlashAttention3, maybe B200 has no fast attention unless SageAttention works.”

For **Krea Realtime 14B specifically**, their README says:

* **B200 (recommended): install `flash_attn`**
* **H100 / RTX 5xxx: install their SageAttention wheel**
* They claim **“Multiple attention backends (Flash Attention 4, SageAttention)”**
* They report **11 FPS T2V on B200 with 4 inference steps** ([GitHub][1])

So in Krea’s world, B200 is not “missing fast attention.” It is the preferred target, and they’re explicitly aiming at “Flash Attention 4” there. ([GitHub][1])

Now, zooming out a bit: upstream Dao-AILab flash-attention’s README still frames FlashAttention-3 as Hopper-focused (“optimized for Hopper GPUs like H100”), and there are issues asking about Blackwell support timelines. ([GitHub][4])
That mismatch basically means: **follow Krea’s pinned stack** rather than assuming “pip flash-attn = FA3 = works everywhere.” Krea is telling you what they tested.

Also, even if you end up not using flash-attn at all on Blackwell, NVIDIA’s stack is not empty here: cuDNN has added compute capability 10.0 support and later release notes mention significant Blackwell attention optimizations. ([NVIDIA Docs][5])

### What you can trade off today to push resolution (without needing multi-GPU)

If we stay grounded in knobs that are clearly present in the open Krea repo:

#### 1) Inference steps (biggest speed knob)

Krea’s headline number is **11 FPS on B200 with 4 inference steps**. ([GitHub][1])
If you want to move from 480p-ish to 720p-ish while staying near 15 FPS, your realistic trade is often:

* keep steps very low (4 or even 3)
* accept that “live” quality is preview quality, then refine offline if needed

#### 2) KV cache depth

Krea calls out “Optimized KV cache management (up to 25GB per GPU)” and exposes `kv_cache_num_frames` in their sampling params (example uses 3). ([GitHub][1])

This is a classic resolution enabler:

* **Lower `kv_cache_num_frames`** = less memory, often less compute, but more temporal drift
* **Higher `kv_cache_num_frames`** = better temporal consistency, but it eats memory fast, which blocks higher resolution

If your goal is “higher res at 15 FPS,” it is totally rational to spend temporal coherence budget here.

#### 3) “num_blocks”

Their example params include `num_blocks=9`. ([GitHub][1])
I will not pretend I know the exact internal meaning without reading their model code, but operationally, anything called “num_blocks” in this sort of stack tends to behave like a quality/speed dial. If it is what it sounds like, reducing it should reduce compute per frame.

#### 4) Compilation

They explicitly support `torch.compile` via `DO_COMPILE=true` for better inference performance. ([GitHub][1])
This matters more when you are trying to hold FPS steady while raising resolution, because Python overhead and kernel launch overhead become a bigger fraction of time once you start optimizing the big kernels.

### If you truly want Wan 2.2 14B for quality, what “multi-GPU” looks like there

Wan 2.2’s own repo shows a very different, offline-oriented setup. A few relevant bits:

* They show a single-GPU command for 1280×720 and note it can run on “at least 80GB VRAM.” ([GitHub][2])
* They show a concrete multi-GPU launch (8 GPUs) for Wan2.2-Animate using **FSDP + DeepSpeed Ulysses** flags. ([GitHub][2])

That gives you a realistic template for what scaling Wan 2.2 looks like: sharding plus sequence-parallel style approaches, not “flip one flag in a single FastAPI process.”

Also notice the repo’s “Community Works” section is basically a menu of acceleration strategies (distillation, sparse attention, caching, FP8 quant, sequence parallelism). That is where most of the practical “tradeoffs” live if you insist on Wan 2.2 quality. ([GitHub][2])

### My honest take for your stated goal

Given what you said (higher resolution matters more than hitting 30+ FPS):

* If you can tolerate “Krea-quality” even though it is Wan 2.1 distilled, **run Krea Realtime on B200** and push resolution using:

  * steps fixed low (4)
  * reduce KV cache frames if memory binds
  * enable compile
  * use their recommended attention backend for that GPU ([GitHub][1])

* If you truly want **Wan 2.2 14B** quality, treat it as a separate “final render” path, and lean on:

  * multi-GPU sharding patterns from Wan 2.2 (FSDP + Ulysses-style) ([GitHub][2])
  * distillation / sparse attention / caching work referenced by the Wan 2.2 project itself ([GitHub][2])

Trying to make “Wan 2.2 14B real-time streaming at higher res” happen inside Scope as a single monolithic FastAPI process is the hardest possible route.

### One practical next step that will de-risk everything

Before you change architecture, do one measurement run on H100:

1. Set a target resolution you actually want (say 1280×720).
2. Measure VRAM peak and FPS with your current stack.
3. Then test just two deltas:

   * lower KV cache frames (if applicable in your pipeline)
   * enable compile (if your pipeline allows it)

If you tell me those two numbers (FPS and peak VRAM), I can give you a pretty crisp “this is memory-bound vs compute-bound” call, and whether B200 likely buys you “higher resolution at 15 FPS” without any multi-GPU work.

[1]: https://github.com/krea-ai/realtime-video "GitHub - krea-ai/realtime-video: Krea Realtime 14B. An open-source realtime AI video model."
[2]: https://github.com/Wan-Video/Wan2.2 "GitHub - Wan-Video/Wan2.2: Wan: Open and Advanced Large-Scale Video Generative Models"
[3]: https://images.nvidia.com/aem-dam/Solutions/documents/HGX-B200-PCF-Summary.pdf?utm_source=chatgpt.com "HGX B200 PCF Summary"
[4]: https://github.com/Dao-AILab/flash-attention "GitHub - Dao-AILab/flash-attention: Fast and memory-efficient exact attention"
[5]: https://docs.nvidia.com/deeplearning/cudnn/backend/v9.11.1/release-notes.html?utm_source=chatgpt.com "Release Notes — NVIDIA cuDNN Backend"



----


If the real goal is “many prompts, one continuous take” with continuity (identity, style, motion) and then an offline “final render” at higher res or higher fidelity, there are basically two families of solutions:

1. **True stateful, causal generation**
   The model keeps an internal state (typically KV cache or something equivalent) and you *change the conditioning over time* while the state keeps the world coherent. This is why Krea Realtime feels like directing a single shot.

2. **Chunked generation with continuity scaffolding**
   You render in chunks (A, then B, then C), and you carry continuity across chunk boundaries using overlap, memory modules, or blending tricks. This can work with very high quality non-causal models, but it is never as “one-take” clean as a good causal model unless the scaffolding is strong.

Below are the best concrete pathways I found for the specific thing you want.

---

## Pathway A: Stay in Krea Realtime, but treat it like a “state machine” you can re-render offline

### A1) Prompt sequences with continuity (the important part)

Krea Realtime’s own model card explicitly calls out **interactive prompt changes mid-generation** as a first-class capability. ([Hugging Face][1])
That maps directly to a “prompt program” idea: keep the model state alive, swap prompts on a schedule, and let the state carry continuity.

If you are using it via Diffusers ModularPipeline, the key mechanism is that you keep a persistent `PipelineState()` and iterate block-by-block (exactly like the model card’s examples). ([Hugging Face][1])

A simple “prompt timeline” pattern looks like this (pseudocode style, but matches their block loop structure):

```py
schedule = [
  # (start_block, end_block, prompt)
  (0, 10, "ANCHOR: a specific character, outfit, location, lens, lighting"),
  (10, 20, "ANCHOR + action change: character walks into a new room"),
  (20, 30, "ANCHOR + mood change: rain starts, neon reflections"),
]

def prompt_for_block(b):
    for s, e, p in schedule:
        if s <= b < e:
            return [p]
    return [schedule[-1][2]]

state = PipelineState()
for block_idx in range(num_blocks):
    prompt = prompt_for_block(block_idx)
    state = pipe(state, prompt=prompt, block_idx=block_idx, num_inference_steps=...)
```

Two practical tricks that matter more than people expect:

* **Use an “anchor clause” that never changes.** LongLive (below) independently recommends repeating the subject + setting in every prompt to improve coherence during prompt switches, and the same principle applies here. ([GitHub][2])
* **Do not hard cut prompts.** For the cleanest transitions, do a short transition window where you gradually rewrite from old to new across a few blocks (or interpolate embeddings if you want to go deeper). Scope’s release notes also mention prompt transition control via temporal interpolation, which is basically the UI expression of this idea. ([GitHub][3])

### A2) Offline “final render” inside Krea Realtime

You have two strong options, both supported by the public artifacts:

**Option A: Run more steps and heavier settings offline**
The Krea repo explicitly lists “offline batch sampling for high-quality outputs” and “streaming and batch inference modes.” ([GitHub][4])
The model card examples also show non-realtime settings (for example, `num_inference_steps=6` in the block loop). ([Hugging Face][1])

**Option B: Two-pass refine using Video-to-Video**
Krea’s diffusers examples include **Video-to-Video** with a `strength` parameter (example uses `strength=0.3`) and also a streaming `video_stream` mode that maintains temporal consistency across chunks. ([Hugging Face][1])

That enables a very practical “preview then final” workflow:

1. Generate a continuous low/medium-res take with your full prompt schedule (fast, interactive).
2. Upscale that video (classic video SR, or even simple upscale if you are careful).
3. Feed the upscaled video into Krea V2V with **low strength** and **more steps** to “re-detail” while preserving motion and identity.

This is usually the highest leverage way to “punch up” without losing the continuity you liked in the first place.

### A3) Hardware and kernels note (H100 vs B200)

The Krea model card claims **11 fps at 4 steps on a single NVIDIA B200** for text-to-video. ([Hugging Face][1])
It also recommends `torch.compile`, SageAttention, and FP8 quantization via torchao for optimized inference, plus the ability to disable SageAttention to use Flash Attention 3 through kernels. ([Hugging Face][1])

So the “B200 might lack the right attention fastpath” worry is less scary in practice: their published recipe explicitly contemplates multiple attention backends and optimization routes.

---

## Pathway B: Keep Krea for the “directed one-take preview,” then re-render the final in Wan2.2 at 720p with multi-GPU

If the final output quality target is “Wan2.2 A14B quality at 720p,” the official Wan2.2 repo is the most straightforward place to land because it has:

* 480p and 720p support for the MoE A14B models (T2V and I2V) ([GitHub][5])
* Explicit **multi-GPU inference** examples using `torchrun` (FSDP + DeepSpeed Ulysses flags are shown) ([GitHub][5])
* A note that TI2V-5B supports 720p at 24 fps, and command examples for 1280×720 or 1280×704 depending on task ([GitHub][5])

### The continuity problem (and the honest constraint)

Wan2.2 in its standard form is not the same “stateful prompt stream” mechanism as Krea. So to emulate “many prompts, one take,” you need chunking scaffolding:

* Render chunk 1 with prompt 1.
* For chunk 2, seed continuity using image-to-video anchored on the last frame (or a selected keyframe) of chunk 1, then blend overlaps.

This works, but it is not as clean as true causal prompt streaming. The upside is you get the high-res Wan2.2 look, and you can throw multi-GPU at it. ([GitHub][5])

If you want, I can sketch a concrete chunk/overlap schedule that tends to minimize boundary artifacts (how many overlap frames, how to pick anchors, what to keep constant in the prompt).

---

## Pathway C: Use a different causal model that is explicitly designed for sequential prompts

### C1) LongLive (NVlabs): sequential prompts + causal continuity

LongLive is extremely aligned with your “prompt sequence, continuity preserved” requirement:

* It explicitly says it **accepts sequential user prompts and generates corresponding videos in real time**. ([GitHub][2])
* It uses a KV “recache” mechanism to make prompt switches smoother and more adherent. ([GitHub][2])
* It reports 20.7 FPS on a single H100, with FP8 pushing higher. ([GitHub][2])
* It is open: repo + weights are linked from the project. ([GitHub][2])

Tradeoff: it is a 1.3B model. You will likely not get Wan2.2-A14B “cinematic” fidelity, but you do get a purpose-built sequential prompt mechanism. That can make it a great “narrative backbone generator” that you later upscale/refine.

### C2) StreamingT2V: chunked long-video generation with explicit transition machinery

StreamingT2V is the best “chunked but principled” long video approach I found:

* It is an autoregressive long-video method aimed at smooth transitions and extendable long generation (paper). ([arXiv][6])
* Code and weights were released, and the repo documents VRAM needs (default is heavy, memory-optimized exists). ([GitHub][7])

Tradeoff: it is more of a research system and may be less plug-and-play than Krea, but conceptually it is exactly the “keep continuity across segments” design space.

---

## Pathway D: The “Krea-for-Wan2.2” direction is starting to exist, but it is preview-grade

FastVideo Team has released a **CausalWan2.2 I2V A14B preview** and describes converting a high-quality Wan2.2 MoE into an autoregressive model. ([Hao AI Lab][8])
The model card explicitly warns it is preview quality and inference is unoptimized. ([Hugging Face][9])

This is the most direct answer to “I want Wan2.2 quality but with Krea-style causal prompt streaming.” It is just not mature yet.

---

## A concrete workflow that matches what you said you want

If you want something that works now and preserves the “directable continuous take” feel:

1. **Creative pass (interactive continuity)**

* Use Krea Realtime with a prompt timeline.
* Keep an anchor clause constant.
* Transition prompts over a few blocks instead of hard switching. ([Hugging Face][1])

2. **Offline final (quality/res)**
   Choose one:

* **Krea V2V refine pass** (lowest risk to continuity): upscale video then V2V at low strength with more steps. ([Hugging Face][1])
* **Wan2.2 final render** (highest fidelity/res): chunked I2V anchored on previous chunk keyframes; use multi-GPU if you want to push resolution or reduce offload overhead. ([GitHub][5])
* **LongLive backbone + Wan2.2 refine** (if you care more about narrative continuity than raw per-frame fidelity in the first pass): LongLive for the sequential prompt continuity, then a heavy model for the final look. ([GitHub][2])

---

If you tell me your target final format (for example 1280×720 at 24 fps vs 1920×1080 at 24 fps, and total duration), I can recommend a specific block schedule and a two-pass recipe (including overlap sizes and where prompt transitions should happen) that is optimized for continuity rather than raw benchmark FPS.

[1]: https://huggingface.co/krea/krea-realtime-video/resolve/main/README.md?download=true "huggingface.co"
[2]: https://github.com/NVlabs/LongLive "GitHub - NVlabs/LongLive: LongLive: Real-time Interactive Long Video Generation"
[3]: https://github.com/daydreamlive/scope/releases/tag/v0.1.0a5 "Release v0.1.0a5 · daydreamlive/scope · GitHub"
[4]: https://github.com/krea-ai/realtime-video "GitHub - krea-ai/realtime-video: Krea Realtime 14B. An open-source realtime AI video model."
[5]: https://github.com/Wan-Video/Wan2.2 "GitHub - Wan-Video/Wan2.2: Wan: Open and Advanced Large-Scale Video Generative Models"
[6]: https://arxiv.org/abs/2403.14773?utm_source=chatgpt.com "StreamingT2V: Consistent, Dynamic, and Extendable Long Video Generation from Text"
[7]: https://github.com/Picsart-AI-Research/StreamingT2V "GitHub - Picsart-AI-Research/StreamingT2V: [CVPR 2025] StreamingT2V: Consistent, Dynamic, and Extendable Long Video Generation from Text"
[8]: https://hao-ai-lab.github.io/blogs/fastvideo_causalwan_preview/ "CausalWan-MoE Preview: Applying Self-Forcing Distillation To Wan2.2 | Hao AI Lab @ UCSD"
[9]: https://huggingface.co/FastVideo/CausalWan2.2-I2V-A14B-Preview-Diffusers "FastVideo/CausalWan2.2-I2V-A14B-Preview-Diffusers · Hugging Face"
