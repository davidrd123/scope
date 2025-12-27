# Adaptive Layer Normalization (AdaLN)

> **Location:** `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py`
> **Class:** `CausalWanAttentionBlock`
> **Note:** Conceptual explainer — some names/shapes are implementation details and may evolve; treat the referenced class as the source of truth.

---

## What Problem Does This Solve?

A diffusion model needs to behave differently at different noise levels:
- At **t=1000** (pure noise): Make bold structural decisions
- At **t=500** (mid-denoise): Refine structure, add features
- At **t=250** (low noise): Preserve details, make subtle adjustments

But a standard transformer has fixed weights. How do we make it **timestep-aware**?

**Answer:** Modulate the normalization layers based on timestep.

---

## The Mechanism

### Standard LayerNorm
```python
y = LayerNorm(x)  # Normalize to zero-mean, unit-variance
```

### AdaLN (Adaptive LayerNorm)
```python
y = LayerNorm(x) * (1 + scale) + shift
#                  └── learned from timestep
```

The `scale` and `shift` are **predicted from the timestep embedding**, so each noise level can transform features differently.

---

## The 6 Modulation Parameters

Each attention block receives a timestep embedding `e` that gets chunked into 6 parameters:

```python
e = (self.modulation + e).chunk(6)
#    └── learned bias      └── from time_projection(time_embedding(t))
```

| Index | Name | Applied To | Purpose |
|-------|------|------------|---------|
| `e[0]` | shift₁ | Self-attention input | Additive bias before attention |
| `e[1]` | scale₁ | Self-attention input | Multiplicative scaling before attention |
| `e[2]` | gate₁ | Self-attention output | Controls residual contribution |
| `e[3]` | shift₂ | FFN input | Additive bias before FFN |
| `e[4]` | scale₂ | FFN input | Multiplicative scaling before FFN |
| `e[5]` | gate₂ | FFN output | Controls residual contribution |

---

## The Forward Pass

```python
class CausalWanAttentionBlock(nn.Module):
    def forward(self, x, e, ...):
        # Chunk timestep embedding into 6 modulation parameters
        e = (self.modulation + e).chunk(6)

        # ═══════════════════════════════════════════════════════
        # SELF-ATTENTION with AdaLN
        # ═══════════════════════════════════════════════════════

        # Apply AdaLN: normalize, then scale and shift
        attn_input = self.norm1(x) * (1 + e[1]) + e[0]
        #            └── LayerNorm   └── scale    └── shift

        # Run self-attention
        y = self.self_attn(attn_input, ...)

        # Gated residual connection
        x = x + y * e[2]
        #       └── gate controls how much attention output to add

        # ═══════════════════════════════════════════════════════
        # CROSS-ATTENTION (no modulation - text is timestep-invariant)
        # ═══════════════════════════════════════════════════════
        x = x + self.cross_attn(self.norm3(x), context, ...)

        # ═══════════════════════════════════════════════════════
        # FFN with AdaLN
        # ═══════════════════════════════════════════════════════

        # Apply AdaLN
        ffn_input = self.norm2(x) * (1 + e[4]) + e[3]

        # Run FFN
        y = self.ffn(ffn_input)

        # Gated residual
        x = x + y * e[5]

        return x
```

---

## Why `(1 + scale)` Instead of Just `scale`?

The form `(1 + scale)` centers the scaling around 1:

| `scale` value | Effect |
|---------------|--------|
| `0` | Multiply by 1 (identity - no change) |
| `+0.5` | Multiply by 1.5 (amplify 50%) |
| `-0.3` | Multiply by 0.7 (dampen 30%) |

This makes it easy for the model to learn "do nothing" (scale=0) as a default, and deviate from there.

If we used raw `scale`, the model would need to learn `scale=1` for identity, which is less natural for initialization.

---

## Why Gates?

Standard residual connection:
```python
x = x + sublayer(x)  # Always add 100% of sublayer output
```

Gated residual:
```python
x = x + sublayer(x) * gate  # Add (gate × 100)% of sublayer output
```

The gate gives the model **per-layer, per-timestep control** over how much each sublayer contributes:

| Scenario | Typical Gate Behavior |
|----------|----------------------|
| High noise (t=1000) | Strong self-attention gates, weaker FFN gates |
| Low noise (t=250) | Balanced gates, let details through |
| Certain layers | Some layers might learn to "turn off" for certain timesteps |

This is more expressive than fixed residual connections.

---

## Why No Modulation on Cross-Attention?

```python
# Cross-attention has NO scale/shift/gate:
x = x + self.cross_attn(self.norm3(x), context, ...)
```

**Reasoning:**
1. Cross-attention conditions on **text embeddings**, not timestep
2. The text conditioning should be **consistent** across all noise levels
3. The self-attention and FFN modulation provide enough timestep-awareness
4. Fewer parameters = faster training, less overfitting

---

## Visual Flow

```
Input x ─────────────────────────────────────────────────────────────────►
    │                                                                    │
    │         Timestep t                                                 │
    │            │                                                       │
    │            ▼                                                       │
    │    ┌───────────────┐                                               │
    │    │time_embedding │──► e ──► chunk(6)                             │
    │    └───────────────┘         │                                     │
    │                              ▼                                     │
    │              [shift₁, scale₁, gate₁, shift₂, scale₂, gate₂]        │
    │                  │       │      │        │       │      │          │
    │    ┌─────────────┴───────┴──────┼────────┼───────┼──────┼─────┐    │
    │    │        SELF-ATTENTION      │        │       │      │     │    │
    │    │                            │        │       │      │     │    │
    │    │  norm1(x)                  │        │       │      │     │    │
    │    │     │                      │        │       │      │     │    │
    │    │     ▼                      │        │       │      │     │    │
    │    │  ×(1+scale₁) + shift₁ ◄────┘        │       │      │     │    │
    │    │     │                               │       │      │     │    │
    │    │     ▼                               │       │      │     │    │
    │    │  self_attn(...)                     │       │      │     │    │
    │    │     │                               │       │      │     │    │
    │    │     ▼                               │       │      │     │    │
    │    │  × gate₁ ◄──────────────────────────┘       │      │     │    │
    │    │     │                                       │      │     │    │
    │    └─────┼───────────────────────────────────────┼──────┼─────┘    │
    │          │                                       │      │          │
    └──────────┴──► (+) ───────────────────────────────┼──────┼──────────┤
                     │                                 │      │          │
                     │  ┌─────────────────────┐        │      │          │
                     │  │  CROSS-ATTENTION    │        │      │          │
                     │  │  (no modulation)    │        │      │          │
                     │  └─────────────────────┘        │      │          │
                     │          │                      │      │          │
                     └──────────┴──► (+) ──────────────┼──────┼──────────┤
                                      │                │      │          │
                     ┌────────────────┼────────────────┴──────┴────┐     │
                     │          FFN   │                            │     │
                     │                │                            │     │
                     │  norm2(x) ◄────┘                            │     │
                     │     │                                       │     │
                     │     ▼                                       │     │
                     │  ×(1+scale₂) + shift₂                       │     │
                     │     │                                       │     │
                     │     ▼                                       │     │
                     │  ffn(...)                                   │     │
                     │     │                                       │     │
                     │     ▼                                       │     │
                     │  × gate₂                                    │     │
                     │     │                                       │     │
                     └─────┼───────────────────────────────────────┘     │
                           │                                             │
                           └──► (+) ─────────────────────────────────────►
                                                                    Output x
```

---

## Where the 6 Parameters Come From

```python
# In CausalWanModel.__init__:
self.time_embedding = nn.Sequential(
    nn.Linear(freq_dim, dim),
    nn.SiLU(),
    nn.Linear(dim, dim)
)
self.time_projection = nn.Sequential(
    nn.SiLU(),
    nn.Linear(dim, 6 * dim)  # ◄── Projects to 6 × dim
)

# In forward:
t_embed = sinusoidal_embedding_1d(freq_dim, timestep)  # [B, freq_dim]
e = self.time_embedding(t_embed)                        # [B, dim]
e0 = self.time_projection(e).unflatten(1, (6, dim))     # [B, 6, dim]

# Each block receives e0 and chunks it:
# e0[:, 0] = shift₁, e0[:, 1] = scale₁, etc.
```

---

## Intuition: What Does the Model Learn?

Through training, the model learns timestep-specific behaviors:

| Timestep | Learned Behavior (Typical) |
|----------|---------------------------|
| t=1000 | Large scales → amplify signals to cut through noise |
| t=1000 | Strong gates → make bold changes to latents |
| t=500 | Moderate scales → balanced refinement |
| t=250 | Small scales → preserve existing structure |
| t=250 | Weak gates → subtle adjustments only |

The 6 parameters × `num_layers` transformer blocks = **(6 × num_layers) learned knobs** that control how the network behaves at each noise level.

---

## Related

- **DiT Paper:** [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)
- **Parent doc:** [`../krea-architecture.md`](../krea-architecture.md)
