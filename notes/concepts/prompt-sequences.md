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
