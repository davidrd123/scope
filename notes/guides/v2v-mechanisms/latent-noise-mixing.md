# Latent / Noise Mixing (`noise_scale`)

> **Location:** `src/scope/core/pipelines/wan2_1/blocks/prepare_video_latents.py`, `src/scope/core/pipelines/wan2_1/blocks/denoise.py`
> **Related:** `src/scope/core/pipelines/wan2_1/blocks/noise_scale_controller.py`
> **Status:** Draft (conceptual; verify against code)

---

## What This Controls

In Scope’s “latent-init” video-to-video path (i.e., pipeline called with `video=...`), `noise_scale` controls **how much of the input video is preserved** vs **how much new content is injected**.

At a high level:

```
input frames ──VAE encode──► video latents ──mix noise──► noisy latents ──denoise──► decoded frames
```

---

## Scope Semantics (Two Places)

### 1) Mixing Latents With Noise

In `PrepareVideoLatentsBlock`, Scope mixes per-block noise into the encoded video latents:

```python
# src/scope/core/pipelines/wan2_1/blocks/prepare_video_latents.py
noisy_latents = noise * noise_scale + latents * (1 - noise_scale)
```

Intuition:
- `noise_scale → 0.0`: mostly preserve the encoded video latents (minimal change)
- `noise_scale → 1.0`: mostly ignore the input and generate new content

### 2) Adjusting the Denoising Start Step

In `DenoiseBlock`, Scope also adjusts the **first** timestep in the denoising schedule when `noise_scale` is provided:

```python
# src/scope/core/pipelines/wan2_1/blocks/denoise.py
denoising_step_list[0] = int(1000 * noise_scale) - 100
```

This is a heuristic coupling between “how noisy are the latents” and “how far down the schedule we start”.
Treat it as an implementation detail (it may evolve).

#### Footguns to Watch For (Likely “Glitchy V2V” Contributors)

1) **Negative timesteps are possible**
- With `noise_scale < 0.1`, `int(1000 * noise_scale) - 100` becomes negative (e.g. `noise_scale=0.0 → -100`).
- It’s unclear whether the model/scheduler semantics are well-defined for negative timesteps; avoid this regime unless you’ve validated it.

2) **Short step lists can become non-monotonic**

In video mode, the UI often uses a short list like `[1000, 750]` (for latency reasons). With the heuristic above:
- `noise_scale=0.7` → `denoising_step_list[0]=600`, yielding `[600, 750]`

That goes “less noisy → more noisy” across steps, which is the opposite of the usual diffusion sampling direction and can plausibly show up as flicker/jitter/instability.

Rule of thumb for the `[1000, 750]` case:
- If `noise_scale < 0.85`, the first step becomes `< 750` and the list becomes increasing.

Generalizing for a 2-step list `[1000, second]`:
- you typically want `int(1000 * noise_scale) - 100 >= second` (i.e. `noise_scale ≳ (second + 100) / 1000`, accounting for `int()` rounding).

If you’re seeing V2V glitches, checking this is one of the fastest sanity tests.

---

## `noise_controller` (Motion-Aware Updates)

When enabled, `NoiseScaleControllerBlock` dynamically adjusts `noise_scale` based on motion estimates:

- High motion → lower `noise_scale` (preserve input frames more)
- Low motion → higher `noise_scale` (rely on generation more)

**Cache reset nuance:** when `noise_controller=True`, the block updates `noise_scale` without forcing `init_cache=True` (so you don’t hard-cut just because motion changed). Cache resets from noise changes only happen when `noise_controller=False` and you manually change `noise_scale` while `manage_cache=True`.

See `src/scope/core/pipelines/wan2_1/blocks/noise_scale_controller.py`.

---

## Scope vs ComfyUI (Why Knobs Feel Different)

ComfyUI “V2V strength” knobs often combine:
- how much noise is mixed into init latents, and
- which timestep the denoiser starts from (schedule-aligned noise).

Scope splits (and loosely re-couples) these ideas via:
- explicit linear mixing (`noise * noise_scale + latents * (1-noise_scale)`), and
- a separate schedule tweak of `denoising_step_list[0]`.

If you’re comparing results across systems, the mapping is not 1:1 — validate perceptually and (when possible) with a small offline oracle.

### Concrete ComfyUI Contrast (What “Battle Tested” Usually Means)

In ComfyUI-WanVideoWrapper, “video2video strength” is typically implemented as:

1. Encode input video into init latents (`WanVideoEncode`)
2. Choose a sampling start point from `denoise_strength`
3. Add noise to init latents at the scheduler-aligned timestep (`add_noise_to_samples`)

Pointers:
- Init latents encoding: `/root/ComfyUI-WanVideoWrapper/nodes.py:2154`
- V2V mixing in the sampler: `/root/ComfyUI-WanVideoWrapper/nodes_sampler.py:718` and `/root/ComfyUI-WanVideoWrapper/nodes_sampler.py:728`

The practical difference is: **ComfyUI ties “how much noise” to “where in the schedule”**, while Scope currently does a simpler latent-space blend and a heuristic first-step tweak.

---

## Debug Checklist (Fast)

1. Log the runtime `denoising_step_list` being sent (frontend → server).
2. Log the *effective* first step after the `noise_scale` mutation in `DenoiseBlock`.
3. Confirm the resulting list is in the intended direction (typically high → low).
