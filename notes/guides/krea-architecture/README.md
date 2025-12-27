# Krea Architecture Deep Dives

This folder contains in-depth explanations of key mechanisms in the Krea Realtime Video Pipeline.

**Primary overview:** [`src/scope/core/pipelines/krea_realtime_video/docs/architecture-guide.md`](../../../src/scope/core/pipelines/krea_realtime_video/docs/architecture-guide.md)

> Note: These guides are intended as conceptual explainers. Exact shapes, constants, and fast-path conditions can vary by config and environment.
> Treat the linked code as the source of truth.

## Deep Dives

| File | Topic | One-liner |
|------|-------|-----------|
| [`adaln-modulation.md`](adaln-modulation.md) | Adaptive LayerNorm | How timestep controls layer behavior |
| [`kv-cache-mechanics.md`](kv-cache-mechanics.md) | KV Cache | Eviction, recomputation, attention bias |
| [`rope-embeddings.md`](rope-embeddings.md) | Rotary Position Embeddings | 3D position encoding for video |
| [`attention-backends.md`](attention-backends.md) | Attention Backends | FA4, Flash, Triton, Flex tradeoffs |
| [`causal-attention-masks.md`](causal-attention-masks.md) | Block-wise Causal Masks | How streaming attention works |
| [`vae-streaming.md`](vae-streaming.md) | VAE Streaming Decode | Causal 3D convolutions |
