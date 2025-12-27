# Krea Architecture ↔ B200/B300 Optimization Crossover

> **Date:** 2025-12-27
> **Status:** Living doc (keep it accurate; link receipts)
> **Purpose:** Connect the “how it works” Krea docs to the “how we made it fast” B200/B300 optimization thread, without duplicating either.

---

## How To Use This Doc

- If you’re reading Krea architecture explainers and wonder **“which parts matter for performance?”**: start here.
- If you’re working the B300 ladder and wonder **“what component am I actually optimizing?”**: use the component map below to jump to the right explainer and code path.

**Source of truth:**
- For “what works today on B300”: `notes/FA4/b300/session-state.md`
- For “how we measure”: `notes/FA4/b300/investigation-runbook.md`
- For “what we tried / receipts”: `notes/FA4/b300/experiments.md`
- For pipeline implementation: `src/scope/core/pipelines/krea_realtime_video/`

---

## Canonical Measurement Conventions (So We Don’t Lie To Ourselves)

The optimization thread standardized around:
- **Quality-first:** BF16 (`--quantization none`) is canonical for “real output”.
- **Canonical resolution:** `320x576` (and stable, quality-preserving settings).
- **One change at a time** + **save artifacts** in `outputs/`.

See `notes/FA4/b300/investigation-runbook.md` for the full protocol.

---

## Component ↔ Optimization Map

This table is the “crossover index”: architecture surface area → perf levers → where to measure.

| Component (mental model) | Where in code | What we optimized / learned | How we measure |
|---|---|---|---|
| **KV-bias self-attention** (drift control) | `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py` | Backend selection + “silent fallback” is a recurring footgun; B300 defaults to `flash` segment-combine; FA4/CuTe `score_mod` is fastest when available. See [`attention-backends.md`](attention-backends.md). | `scripts/profile_krea_pipeline_ops.py` (focus `self_attn*`); `notes/FA4/explainers/17-backend-selection-and-knobs.md` |
| **RoPE + KV-cache writes** (glue around attention) | `src/scope/core/pipelines/krea_realtime_video/modules/model.py`, `.../causal_model.py` | A large share of “remaining time” is often glue (copies/views/packing) rather than the attention kernel itself; this is the main Level 6 target (“post-projection pack” concept). See [`rope-embeddings.md`](rope-embeddings.md). | Op-profile with stacks (`scripts/profile_krea_pipeline_ops.py --with-stack`), then document in `notes/FA4/b300/other-in-self-breakdown.md` |
| **KV-cache recomputation** (stability cost) | `src/scope/core/pipelines/krea_realtime_video/blocks/recompute_kv_cache.py` | Recomputation is a meaningful slice of time; wins tend to come from reducing downstream glue and avoiding slow-path attention configs rather than changing the algorithm first. See [`kv-cache-mechanics.md`](kv-cache-mechanics.md). | Block timing (`PROFILE_PIPELINE_BLOCKS=1` / `scripts/profile_krea_pipeline_blocks.py`) |
| **VAE streaming decode** (latents → pixels) | `src/scope/core/pipelines/wan2_1/vae/wan.py`, `src/scope/core/pipelines/wan2_1/vae/modules/vae.py` | Decode has historically contained a lot of dtype/layout/padding overhead; changes often look like “remove glue” (avoid extra casts/copies; prefer fast memory formats). See [`vae-streaming.md`](vae-streaming.md). | `PROFILE_WANVAE_DECODE_INNER=1` + block timing; receipts in `outputs/` |
| **VAE encode (V2V only)** | `src/scope/core/pipelines/wan2_1/vae/wan.py` | Encode is “new cost” in V2V. Offline benchmarks showed UI low-FPS can be input-rate limited, not compute-limited. | `PROFILE_WANVAE_ENCODE=1` (JSON); V2V experiment card in `notes/FA4/b300/experiments.md` |
| **Server pacing / input buffering** (what user perceives as FPS) | `src/scope/server/frame_processor.py`, `src/scope/server/tracks.py` | Output FPS is intentionally `min(input_fps, pipeline_fps)`. A fast pipeline can still "look slow" if capture is slow or the buffer is starved. | Inspect `FrameProcessor.get_output_fps()` + V2V docs in [`notes/guides/v2v-mechanisms/`](../v2v-mechanisms/) (esp. [`frame-processor-routing.md`](../v2v-mechanisms/frame-processor-routing.md)) |
| **LoRA / style swap** (runtime weight patching) | `src/scope/core/pipelines/wan2_1/modules/`, LoRA loading paths | Style swap with `runtime_peft` causes **torch.compile graph breaks** (lesson #6 above). Expected ~50% FPS hit without mitigation. Solution: `custom_op` wrappers (see ComfyUI research below). | `STYLE_SWAP_MODE=1` + FPS measurement; [`notes/research/comfyui-wrapper-techniques.md`](../../research/comfyui-wrapper-techniques.md) |

---

## “What Changed” From The B300 Optimization Thread (Short Version)

This is the minimal set of takeaways worth reflecting back into Krea explainers:

1. **Backend selection is half the battle.** Many “mystery regressions” are really a backend falling back (attention, RoPE, compile modes). Always log which backend actually ran. See `notes/FA4/explainers/17-backend-selection-and-knobs.md`.
2. **Most reliable wins were “delete glue”.** Stack-attributed profiles repeatedly pointed at copies/casts/layout transforms; fixing those often beats chasing new kernels.
3. **FP8 is not currently a quality-safe path on B300.** Even when compile blockers are patched, output quality can be visibly wrong; BF16 remains canonical.
4. **UI FPS ≠ pipeline FPS.** The server paces output at `min(input_fps, pipeline_fps)`; separate “compute throughput” from “stream throughput” when diagnosing.
5. **Level 6 direction is clearer now.** The remaining hotspot tends to be attention-adjacent glue (RoPE/packing/KV-write/layout contracts), not the attention kernel itself; that's why the Level 6 docs focus on layout contracts and `other_in_self`.
6. **torch.compile graph breaks kill perf.** Dynamic operations cause recompilation or eager fallback. The canonical example is **LoRA weight patching**: the compiler traces `weight + lora_A @ lora_B * scale` and bakes in specific weights. Swap LoRA → graph invalid → recompile or fall back. The fix: wrap dynamic ops in `torch.library.custom_op` with a `register_fake` stub that declares output shape without tracing internals. Compiler sees stable signature, actual math runs at runtime with any weights. See ComfyUI research below.

For the "current truth" and longer narrative, see `notes/FA4/b300/README.md` and `notes/FA4/b300/lessons.md`.

---

## External Research: ComfyUI-WanVideoWrapper Techniques

We surveyed `/root/ComfyUI-WanVideoWrapper/` for techniques that might help Scope. Key findings:

| Technique | Relevance | Details |
|-----------|-----------|---------|
| **LoRA custom_op wrappers** | High - solves style swap FPS hit | Wrap LoRA application in `torch.library.custom_op` to avoid graph breaks |
| **FP8 fast matmul** | Medium - could help GEMM-bound workloads | Uses `torch._scaled_mm` with per-layer scales |
| **SageAttention Blackwell** | Low - FA4 likely already optimal | `sageattn3_blackwell` kernel, worth a quick benchmark |

Full analysis: [`notes/research/comfyui-wrapper-techniques.md`](../../research/comfyui-wrapper-techniques.md)

---

## How To Reflect This Back Into Krea Explainers (Suggested Pattern)

When editing or adding Krea explainers, append a small “Performance Hooks” section that includes:
- **Which env vars/knobs affect this component** (and whether they’re read at import time)
- **How to profile it** (block vs op; any stack-attribution recipe)
- **Common failure modes** (fallbacks; shape/layout pitfalls; compile hazards)
- **Links to receipts** (an `experiments.md` card and/or `outputs/...` artifacts)

This keeps architecture docs readable while making them operational for performance work.
