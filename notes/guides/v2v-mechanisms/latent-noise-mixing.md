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

---

## `noise_controller` (Motion-Aware Updates)

When enabled, `NoiseScaleControllerBlock` can dynamically adjust `noise_scale` based on motion estimates of the input frames, and can trigger cache resets if the noise scale changes enough to invalidate cached assumptions. See `src/scope/core/pipelines/wan2_1/blocks/noise_scale_controller.py`.

---

## Scope vs ComfyUI (Why Knobs Feel Different)

ComfyUI “V2V strength” knobs often combine:
- how much noise is mixed into init latents, and
- which timestep the denoiser starts from (schedule-aligned noise).

Scope splits (and loosely re-couples) these ideas via:
- explicit linear mixing (`noise * noise_scale + latents * (1-noise_scale)`), and
- a separate schedule tweak of `denoising_step_list[0]`.

If you’re comparing results across systems, the mapping is not 1:1 — validate perceptually and (when possible) with a small offline oracle.

