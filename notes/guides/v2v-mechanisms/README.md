# V2V Mechanisms Deep Dives

This folder contains in-depth explanations of how video-to-video works in Scope.

**Parent doc:** [`../v2v-mechanisms.md`](../v2v-mechanisms.md) (overview)

> Note: These deep dives are conceptual and intended to speed up code navigation. Treat referenced code paths as the source of truth.

## Deep Dives

| File | Topic | One-liner |
|------|-------|-----------|
| [`frame-processor-routing.md`](frame-processor-routing.md) | Frame Flow | WebRTC → buffer → sampling → pipeline kwargs |
| [`latent-noise-mixing.md`](latent-noise-mixing.md) | Noise Mixing | noise_scale math, Scope vs ComfyUI semantics |
| [`latent-noise-mixing-tutorial.md`](latent-noise-mixing-tutorial.md) | Tutorial | Practical guide for tuning noise_scale safely |

## Tutorials

| File | Topic | One-liner |
|------|-------|-----------|
| [`latent-noise-mixing-tutorial.md`](latent-noise-mixing-tutorial.md) | Using noise_scale | Mental models, tuning guide, practical tips |

## Related

- [`../krea-architecture/`](../krea-architecture/) - How the model works
- [`../vace-architecture-explainer.md`](../vace-architecture-explainer.md) - VACE conditioning path
