# Understanding noise_scale (Tutorial)

> **Reference doc:** [`latent-noise-mixing.md`](latent-noise-mixing.md)
> **Purpose:** Conceptual guide for thinking about and tuning V2V strength

---

## The Mental Model

Think of `noise_scale` as answering: **"How much creative freedom does the model get?"**

```
noise_scale = 0.0    "Maximum preservation" (but see schedule footguns below)
              ↓
noise_scale = 0.5    "Use my video as a strong suggestion"
              ↓
noise_scale = 1.0    "Ignore my video, just use the prompt"
```

---

## How It Works (Visual)

```
                            Scope's V2V Pipeline
                            ═══════════════════

Input Video          Noise
     │                 │
     ▼                 ▼
[VAE Encode]      [Generate]
     │                 │
     ▼                 ▼
  latents           noise
     │                 │
     └────────┬────────┘
              │
              ▼
    ┌─────────────────────────────────┐
    │  mixed = noise * scale          │  ◄─── Step 1: Linear mix
    │        + latents * (1 - scale)  │
    └─────────────────────────────────┘
              │
              ▼
    ┌─────────────────────────────────┐
    │  denoising_step_list[0] = ...   │  ◄─── Step 2: First-step heuristic
    └─────────────────────────────────┘
              │
              ▼
         [Denoise]
              │
              ▼
       Output Latents
```

Two things happen:
1. **Mixing:** Your video latents get blended with random noise
2. **Schedule shift:** The denoiser starts earlier or later based on how noisy the mix is

---

## The -100 Offset

The current heuristic is:
```python
# src/scope/core/pipelines/wan2_1/blocks/denoise.py
denoising_step_list[0] = int(1000 * noise_scale) - 100
```

Why subtract 100?

- There isn’t a single “canonical” explanation in the code today — treat this as a **heuristic** rather than a principled mapping.
- If you need to reason about it, treat the following as **inferred hypotheses**, not guaranteed intent:

1. **(Hypothesis) Alignment mismatch:** linear latent mixing doesn’t match the scheduler’s exact noise schedule.

2. **(Hypothesis) Preservation bias:** the offset biases towards a slightly “less destructive” first step for a given `noise_scale`.

3. **(Hypothesis) Safety margin:** it may avoid overly aggressive first steps when the blend is ambiguous.

| noise_scale | Raw mapping | With -100 offset | Effect |
|-------------|-------------|------------------|--------|
| 0.3 | 300 | 200 | Light touch-up |
| 0.5 | 500 | 400 | Moderate restyle |
| 0.8 | 800 | 700 | Heavy transformation |
| 1.0 | 1000 | 900 | Near-full generation |

### Important Footguns (Worth Checking First When V2V Looks “Glitchy”)

1) **Negative timesteps are possible**
- `noise_scale < 0.1` makes the first step negative (e.g. `noise_scale=0.0 → -100`).
- Unless you’ve validated behavior for negative timesteps with your scheduler/model, avoid this regime.

2) **Short step lists can become non-monotonic**

Many video-mode defaults use a short list like `[1000, 750]` for latency. With the heuristic above:
- `noise_scale=0.7` → `[600, 750]` (increasing)

If your scheduler expects timesteps to move consistently from “more noise” to “less noise”, this can be a stability killer.

Rule of thumb for the `[1000, 750]` case:
- keep `noise_scale >= 0.85` so the first step stays `>= 750`

---

## Scope vs ComfyUI

If you're coming from ComfyUI, the knobs feel different:

| Aspect | Scope | ComfyUI |
|--------|-------|---------|
| Parameter | `noise_scale` (0-1) | `denoise_strength` (0-1) |
| Mixing | Linear interpolation | Sigma-scaled addition |
| Schedule | Separate `-100` heuristic | Derived from strength |
| Intuition | "How much noise to add" | "How many steps to run" |

**Key insight:** Same number ≠ same result. `noise_scale=0.7` in Scope and `denoise_strength=0.7` in ComfyUI will look different. Tune by eye, not by number.

---

## Practical Tuning Guide

### By Range

| noise_scale | What You Get |
|-------------|--------------|
| **0.2-0.4** | Subtle style transfer. Motion and composition strongly preserved. Prompt influences color/texture more than structure. |
| **0.5-0.7** | Balanced transformation. Prompt can override major elements. Motion patterns still recognizable. |
| **0.8-1.0** | Near text-to-video. Only vague temporal hints from input. Prompt dominates completely. |

### Common Scenarios

**"I want more of the original video preserved"**
- Lower `noise_scale` (try 0.2-0.3)
- You'll get lighter denoising, more input structure kept

**"The output doesn't match my prompt"**
- Raise `noise_scale` (try 0.6-0.8)
- Gives the model more room to interpret the prompt

**"Motion looks jittery or stuttery"**
- This is usually **not** a noise_scale issue
- Check frame sampling in [`frame-processor-routing.md`](frame-processor-routing.md)
- Look at buffer stride - should be ~1.0

**"First frame looks different from the rest"**
- VAE alignment issue on first block
- See [`../krea-architecture/vae-streaming.md`](../krea-architecture/vae-streaming.md)

---

## Under the Hood: Seed Management

Each block gets a deterministic but unique seed:

```python
block_seed = base_seed + block_state.current_start_frame
rng = torch.Generator(device=...).manual_seed(block_seed)
```

This ensures:
- **Reproducibility:** Same seed → same output
- **Temporal variation:** Blocks don't repeat the same noise pattern
- **Continuity:** Adjacent blocks have related but different noise

---

## When noise_scale is None

For pure text-to-video (no input video), `noise_scale` is None and the schedule adjustment is skipped entirely. The pipeline starts from pure noise at the full schedule.

---

## Motion-Aware Adjustment (noise_controller)

When `noise_controller=True`, the `NoiseScaleControllerBlock` can dynamically adjust `noise_scale` based on motion in the input frames:

- High motion → tends to *lower* `noise_scale` (preserve input frames more)
- Low motion → tends to *raise* `noise_scale` (rely on generation more)

**Cache reset nuance:** when `noise_controller=True`, the block updates `noise_scale` without forcing `init_cache=True` (so you don’t hard-cut just because motion changed). Cache resets from noise changes only happen when `noise_controller=False` and you manually change `noise_scale` while `manage_cache=True`.

See: `src/scope/core/pipelines/wan2_1/blocks/noise_scale_controller.py`

---

## Debugging Checklist

1. **Log effective noise_scale per block** (especially with noise_controller)
2. **Log denoising_step_list after mutation** - confirm it stays in the intended order
3. **If output unstable only with noise_controller** - suspect rapid schedule changes

---

## Related

- **Reference:** [`latent-noise-mixing.md`](latent-noise-mixing.md)
- **Frame flow:** [`frame-processor-routing.md`](frame-processor-routing.md)
- **VAE details:** [`../krea-architecture/vae-streaming.md`](../krea-architecture/vae-streaming.md)
