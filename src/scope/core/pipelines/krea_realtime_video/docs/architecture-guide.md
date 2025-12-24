# KREA Realtime Video Pipeline: Comprehensive Architecture Guide

This guide provides an in-depth explanation of every component in the `krea_realtime_video` pipeline, a streaming video generation system built on the Wan2.1 14B diffusion model with causal attention and KV caching.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Directory Structure](#2-directory-structure)
3. [Configuration (model.yaml)](#3-configuration-modelyaml)
4. [Main Pipeline (pipeline.py)](#4-main-pipeline-pipelinepy)
5. [Modular Blocks System (modular_blocks.py)](#5-modular-blocks-system-modular_blockspy)
6. [The Causal Model (modules/causal_model.py)](#6-the-causal-model-modulescausal_modelpy)
7. [Standard Model Reference (modules/model.py)](#7-standard-model-reference-modulesmodelpy)
8. [VAE Components (modules/vae.py)](#8-vae-components-modulesvaepy)
9. [Pipeline Blocks](#9-pipeline-blocks)
10. [Data Flow and Execution](#10-data-flow-and-execution)
11. [Performance Optimizations](#11-performance-optimizations)
12. [Usage Example (test.py)](#12-usage-example-testpy)

---

## 1. Overview

### What is this pipeline?

The KREA Realtime Video Pipeline is a **streaming video generation** system that produces video frame-by-frame (or block-by-block) in real-time. Unlike traditional video diffusion models that generate entire videos at once, this pipeline:

1. **Processes frames incrementally**: Generates 3 frames at a time (configurable via `num_frame_per_block`)
2. **Uses KV caching**: Stores key-value pairs from previous frames to avoid recomputation
3. **Maintains temporal coherence**: Uses context frames and attention bias to keep frames consistent
4. **Supports streaming output**: Each call returns the next chunk of video

### Key Innovations

| Feature | Description |
|---------|-------------|
| **Causal Attention** | Each frame block can only attend to previous frames, enabling streaming |
| **KV Cache Recomputation** | Periodically refreshes the cache with context frames to prevent error drift |
| **Attention Bias** | Down-weights attention to older frames to mitigate accumulating errors |
| **Multiple Attention Backends** | FA4/CUTE, FlashAttention, Triton, and flex_attention fallbacks |

### Based On

- **CausVid Paper**: [arxiv.org/abs/2412.07772](https://arxiv.org/abs/2412.07772)
- **Base Model**: Wan2.1-T2V-14B (Alibaba's text-to-video model)

---

## 2. Directory Structure

```
krea_realtime_video/
├── __init__.py                 # Exports KreaRealtimeVideoPipeline
├── pipeline.py                 # Main pipeline class (241 lines)
├── modular_blocks.py           # Block sequence configuration (43 lines)
├── model.yaml                  # Model hyperparameters
├── test.py                     # Example usage/benchmark script
│
├── modules/                    # Core neural network components
│   ├── __init__.py
│   ├── causal_model.py         # Causal attention model (2127 lines) ⭐
│   ├── model.py                # Standard (non-causal) model reference
│   ├── vae.py                  # 3D Causal VAE (779 lines)
│   └── vae_block3.py           # Modular VAE encoder/decoder
│
├── blocks/                     # Pipeline processing blocks
│   ├── __init__.py
│   ├── prepare_context_frames.py   # Manages context frame buffers
│   └── recompute_kv_cache.py       # KV cache refresh logic
│
├── components/                 # Reserved for future components
│   └── __init__.py
│
└── docs/                       # Documentation
    ├── usage.md
    └── examples/
```

---

## 3. Configuration (model.yaml)

The `model.yaml` file defines the core hyperparameters:

```yaml
base_model_name: Wan2.1-T2V-14B      # Base diffusion model
base_model_kwargs:
  timestep_shift: 5.0                 # Noise schedule shift

num_frame_per_block: 3                # Frames generated per call
kv_cache_num_frames: 3                # Context frames for KV cache
local_attn_size: 6                    # Local attention window (frames)

vae_spatial_downsample_factor: 8      # VAE spatial compression
vae_temporal_downsample_factor: 4     # VAE temporal compression
patch_embedding_spatial_downsample_factor: 2  # Patch embedding compression

max_rope_freq_table_seq_len: 1024     # RoPE frequency table size
```

### Key Parameters Explained

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `num_frame_per_block` | 3 | Each pipeline call generates 3 new frames |
| `kv_cache_num_frames` | 3 | The KV cache holds context from 3 frames |
| `local_attn_size` | 6 | Attention can look back 6 frames (local window) |
| Resolution scale | 16x | Total downsampling: 8 (VAE) × 2 (patch) = 16 |

---

## 4. Main Pipeline (pipeline.py)

**Location**: `pipeline.py` (241 lines)

The `KreaRealtimeVideoPipeline` class is the main entry point. It inherits from:
- `Pipeline`: Base pipeline interface
- `LoRAEnabledPipeline`: LoRA adapter support

### 4.1 Class Constants

```python
DEFAULT_DENOISING_STEP_LIST = [1000, 750, 500, 250]  # Multi-step denoising
DEFAULT_KV_CACHE_ATTENTION_BIAS = 0.3                # Past-frame attention weight
WARMUP_PROMPT = [{"text": "a majestic sunset", "weight": 1.0}]
```

### 4.2 Initialization (`__init__`)

The constructor performs these steps in order:

```
1. Validate Resolution
   └── Height/width must be divisible by 16 (8×VAE × 2×patch)

2. Load Generator (Diffusion Model)
   ├── Creates WanDiffusionWrapper with CausalWanModel
   ├── Fuses Q/K/V projections for efficiency
   ├── Initializes LoRA adapters (optional)
   └── Optionally quantizes to FP8_E4M3FN

3. Load Text Encoder
   └── WanTextEncoderWrapper (UMT5-XXL)

4. Load VAE
   └── WanVAEWrapper (3D Causal VAE)

5. Setup Components Manager
   ├── Stores generator, scheduler, vae, text_encoder
   └── Creates EmbeddingBlender

6. Initialize State
   ├── current_start_frame = 0
   ├── manage_cache = True
   └── kv_cache_attention_bias (from config or env)

7. Warmup Runs
   └── Run pipeline ceil(local_attn_size / num_frame_per_block) + 1 times
       to fill KV cache and trigger torch.compile
```

#### Why Warmup?

```python
# Cache fills at: num_frame_per_block frames per iteration
# Cache capacity: local_attn_size frames
# Iterations needed: ceil(local_attn_size / num_frame_per_block) + 1
#   (+1 to exercise the "cache full with eviction" path)
warmup_runs = (local_attn_size // num_frame_per_block) + 1  # = 3
```

This ensures `torch.compile` compiles kernels at steady-state cache size.

### 4.3 Main Methods

#### `prepare(**kwargs) -> Requirements`

Returns input requirements based on current mode (text-to-video or video-to-video).

#### `__call__(**kwargs) -> torch.Tensor`

Main entry point. Handles mode transitions and calls `_generate()`.

```python
def __call__(self, **kwargs) -> torch.Tensor:
    # Handle mode transitions (e.g., switching from T2V to V2V)
    self.first_call, self.last_mode = handle_mode_transition(
        self.state, self.components.vae, self.first_call, self.last_mode, kwargs
    )
    return self._generate(**kwargs)
```

#### `_generate(**kwargs) -> torch.Tensor`

Core generation logic:

```python
def _generate(self, **kwargs) -> torch.Tensor:
    # 1. Handle LoRA scale updates (triggers cache reset)
    if lora_scales is not None:
        self._handle_lora_scale_updates(...)
        kwargs["init_cache"] = True  # Reset cache on LoRA change

    # 2. Update pipeline state with kwargs
    for k, v in kwargs.items():
        self.state.set(k, v)

    # 3. Apply default denoising steps if not provided
    if self.state.get("denoising_step_list") is None:
        self.state.set("denoising_step_list", DEFAULT_DENOISING_STEP_LIST)

    # 4. Apply mode-specific defaults
    apply_mode_defaults_to_state(self.state, self.__class__, mode, kwargs)

    # 5. Execute the modular blocks pipeline
    _, self.state = self.blocks(self.components, self.state)

    # 6. Postprocess and return
    return postprocess_chunk(self.state.values["output_video"])
```

### 4.4 Important State Variables

| State Key | Type | Description |
|-----------|------|-------------|
| `current_start_frame` | int | Index of first frame in current block |
| `manage_cache` | bool | Whether to auto-manage KV cache |
| `kv_cache_attention_bias` | float | Attention weight for past frames (0.3 default) |
| `height`, `width` | int | Video resolution |
| `base_seed` | int | Random seed |
| `denoising_step_list` | list | Timesteps for denoising (e.g., [1000, 750, 500, 250]) |

---

## 5. Modular Blocks System (modular_blocks.py)

**Location**: `modular_blocks.py` (43 lines)

The pipeline uses a **modular blocks architecture** from `diffusers`. Each block is a self-contained processing step.

### 5.1 Block Sequence

```python
ALL_BLOCKS = InsertableDict([
    ("text_conditioning", TextConditioningBlock),      # 1. Encode text prompts
    ("embedding_blending", EmbeddingBlendingBlock),    # 2. Blend embeddings
    ("set_timesteps", SetTimestepsBlock),              # 3. Setup diffusion schedule
    ("auto_preprocess_video", AutoPreprocessVideoBlock),# 4. Encode video input (if V2V)
    ("setup_caches", SetupCachesBlock),                # 5. Initialize KV/cross-attn caches
    ("auto_prepare_latents", AutoPrepareLatentsBlock), # 6. Prepare noisy latents
    ("recompute_kv_cache", RecomputeKVCacheBlock),     # 7. Refresh KV cache ⭐
    ("denoise", DenoiseBlock),                         # 8. Main denoising loop
    ("decode", DecodeBlock),                           # 9. VAE decode to pixels
    ("prepare_context_frames", PrepareContextFramesBlock), # 10. Update buffers ⭐
    ("prepare_next", PrepareNextBlock),                # 11. Prepare for next call
])
```

### 5.2 Block Flow Diagram

```
                    ┌─────────────────────────────────────┐
                    │          Input (prompts, video)     │
                    └─────────────────┬───────────────────┘
                                      ▼
                    ┌─────────────────────────────────────┐
                    │       1. TextConditioningBlock      │
                    │   Encode prompts → text embeddings  │
                    └─────────────────┬───────────────────┘
                                      ▼
                    ┌─────────────────────────────────────┐
                    │       2. EmbeddingBlendingBlock     │
                    │   Blend with LoRA / style embeds    │
                    └─────────────────┬───────────────────┘
                                      ▼
                    ┌─────────────────────────────────────┐
                    │        3. SetTimestepsBlock         │
                    │   Setup diffusion timesteps         │
                    │   [1000, 750, 500, 250]             │
                    └─────────────────┬───────────────────┘
                                      ▼
                    ┌─────────────────────────────────────┐
                    │     4. AutoPreprocessVideoBlock     │
                    │   If video input: encode to latent  │
                    └─────────────────┬───────────────────┘
                                      ▼
                    ┌─────────────────────────────────────┐
                    │        5. SetupCachesBlock          │
                    │   Initialize KV cache, cross-attn   │
                    └─────────────────┬───────────────────┘
                                      ▼
                    ┌─────────────────────────────────────┐
                    │      6. AutoPrepareLatentsBlock     │
                    │   Create noisy latents for frames   │
                    └─────────────────┬───────────────────┘
                                      ▼
                    ┌─────────────────────────────────────┐
                    │      7. RecomputeKVCacheBlock ⭐    │
                    │   Fill cache with context frames    │
                    │   (prevents error accumulation)     │
                    └─────────────────┬───────────────────┘
                                      ▼
                    ┌─────────────────────────────────────┐
                    │          8. DenoiseBlock            │
                    │   For each timestep:                │
                    │     - Forward pass with KV cache    │
                    │     - Scheduler step                │
                    └─────────────────┬───────────────────┘
                                      ▼
                    ┌─────────────────────────────────────┐
                    │          9. DecodeBlock             │
                    │   VAE decode latents → video        │
                    └─────────────────┬───────────────────┘
                                      ▼
                    ┌─────────────────────────────────────┐
                    │   10. PrepareContextFramesBlock ⭐  │
                    │   Update sliding window buffers     │
                    └─────────────────┬───────────────────┘
                                      ▼
                    ┌─────────────────────────────────────┐
                    │        11. PrepareNextBlock         │
                    │   Increment frame counter           │
                    │   Prepare state for next call       │
                    └─────────────────┬───────────────────┘
                                      ▼
                    ┌─────────────────────────────────────┐
                    │       Output (3 video frames)       │
                    └─────────────────────────────────────┘
```

---

## 6. The Causal Model (modules/causal_model.py)

**Location**: `modules/causal_model.py` (2127 lines)

This is the **heart of the streaming architecture**. It implements causal (autoregressive) attention with KV caching.

### 6.1 Architecture Overview

```
CausalWanModel
├── patch_embedding (Conv3d)     # Video → patches
├── text_embedding (MLP)         # Text features → hidden dim
├── time_embedding (MLP)         # Timestep → embedding
├── time_projection (MLP)        # Time → modulation parameters
├── blocks (ModuleList)          # 32× CausalWanAttentionBlock
│   └── CausalWanAttentionBlock
│       ├── norm1 + self_attn (CausalWanSelfAttention)
│       ├── norm3 + cross_attn
│       └── norm2 + ffn
├── head (CausalHead)            # Output projection
└── freqs (RoPE frequencies)     # Rotary position embeddings
```

### 6.2 CausalWanSelfAttention

**Location**: Lines 743-1172

This class implements block-wise causal self-attention with KV caching.

#### Key Attributes

```python
class CausalWanSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, local_attn_size=-1, sink_size=0, ...):
        self.dim = dim                    # Hidden dimension
        self.num_heads = num_heads        # Number of attention heads
        self.head_dim = dim // num_heads  # Dimension per head
        self.local_attn_size = local_attn_size  # Local attention window
        self.sink_size = sink_size        # Sink tokens (kept during eviction)
        self.frame_seq_length = 1560      # Tokens per frame (default)

        # Projections (can be fused)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

        # Query/Key normalization
        self.norm_q = WanRMSNorm(dim)
        self.norm_k = WanRMSNorm(dim)
```

#### Projection Fusion

```python
def fuse_projections(self):
    """Combine Q, K, V into single linear for efficiency."""
    concatenated_weights = torch.cat([
        self.q.weight.data,
        self.k.weight.data,
        self.v.weight.data
    ])
    self.to_qkv = nn.Linear(in_features, out_features)
    self.fused_projections = True
```

#### Forward Pass

The forward method has two main paths:

**Path A: With Block Mask (Training/Initialization)**
```python
if kv_cache is None or block_mask is not None:
    # Apply RoPE
    roped_query = rope_apply(q, grid_sizes, freqs)
    roped_key = rope_apply(k, grid_sizes, freqs)

    # Pad to multiple of 128 for flex_attention
    padded_q = pad_to_128(roped_query)
    padded_k = pad_to_128(roped_key)
    padded_v = pad_to_128(v)

    # Use flex_attention with block mask
    attn_out = flex_attention(
        query=padded_q.transpose(2, 1),
        key=padded_k.transpose(2, 1),
        value=padded_v.transpose(2, 1),
        block_mask=block_mask,
    )
```

**Path B: With KV Cache (Streaming Inference)**
```python
else:
    # Apply causal RoPE with frame offset
    roped_query = causal_rope_apply(q, grid_sizes, freqs, start_frame=current_start_frame)
    roped_key = causal_rope_apply(k, grid_sizes, freqs, start_frame=current_start_frame)

    # Update KV cache
    kv_cache["k"][:, local_start_index:local_end_index] = roped_key
    kv_cache["v"][:, local_start_index:local_end_index] = v

    # Handle cache eviction if needed
    if cache_is_full:
        # Shift cache left, keeping sink tokens
        evict_oldest_tokens()

    # Retrieve cached K, V
    cached_k = kv_cache["k"][:, kv_start_idx:local_end_index]
    cached_v = kv_cache["v"][:, kv_start_idx:local_end_index]

    # Apply attention (with optional bias)
    if kv_cache_attention_bias != 1.0:
        # Use specialized attention with bias
        x = attention_with_kv_bias(...)
    else:
        # Standard Flash Attention
        x = attention(roped_query, cached_k, cached_v)
```

### 6.3 KV Cache Eviction Strategy

When the cache is full, the oldest tokens are evicted:

```python
if num_new_tokens + local_end_index > cache_size:
    # Calculate eviction
    num_evicted = num_new_tokens + local_end_index - cache_size
    num_rolled = local_end_index - num_evicted - sink_tokens

    # Shift cache left (keep sink tokens)
    # [sink][old1][old2][old3][new] → [sink][old2][old3][new][ ]
    kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled] = \
        kv_cache["k"][:, sink_tokens + num_evicted:...].clone()
```

The `sink_size` parameter ensures the first N tokens (representing the first frame) are never evicted.

### 6.4 KV Cache Attention Bias

To mitigate error accumulation in autoregressive generation, past frames receive reduced attention weight:

```python
# log_scale = log(kv_cache_attention_bias)  # e.g., log(0.3) ≈ -1.2
# This is added to attention scores for past frames

def score_mod(score, b_idx, h_idx, q_idx, kv_idx):
    # Region 2: past frames (not first frame, not current block)
    # [first_frame][past_frames][current_block]
    #      ↑           ↑              ↑
    #   no bias    ADD BIAS       no bias

    return torch.where(
        (kv_idx >= frame_seqlen) & (kv_idx < current_block_start),
        score + log_scale,  # Reduce attention
        score,              # Keep original
    )
```

### 6.5 Attention Backend Selection

The code supports multiple attention backends:

```python
# Environment variable: SCOPE_KV_BIAS_BACKEND
# Options: fa4, flash, triton, flex

_KV_BIAS_BACKEND = (
    "flash" if _is_sm103()  # B300 GPUs
    else "triton"           # Default
)
```

| Backend | Description | Best For |
|---------|-------------|----------|
| `fa4` | FA4/CUTE with score_mod | B200 (1.89x faster) |
| `flash` | FlashAttention segment-combine | SM103/B300 |
| `triton` | Triton Kernel B | Default fallback |
| `flex` | torch.nn.attention.flex_attention | Ultimate fallback |

### 6.6 CausalWanAttentionBlock

Wraps self-attention, cross-attention, and FFN:

```python
class CausalWanAttentionBlock(nn.Module):
    def forward(self, x, e, ...):
        # Time modulation: 6 parameters per frame
        e = (self.modulation + e).chunk(6)  # [shift, scale] × 3

        # Self-attention with modulation
        y = self.self_attn(
            (self.norm1(x) * (1 + e[1]) + e[0]),  # scale + shift
            ...
        )
        x = x + y * e[2]  # gate

        # Cross-attention (text conditioning)
        x = x + self.cross_attn(self.norm3(x), context, ...)

        # FFN with modulation
        y = self.ffn(
            (self.norm2(x) * (1 + e[4]) + e[3])
        )
        x = x + y * e[5]  # gate

        return x
```

### 6.7 Attention Mask Generation

#### Block-wise Causal Mask (Standard)

```python
def _prepare_blockwise_causal_attn_mask(device, num_frames, frame_seqlen,
                                        num_frame_per_block, local_attn_size):
    """
    Creates: [block1][block2][block3]...
    Each block can attend to itself + all previous blocks
    """
    # ends[i] = end index of the block containing position i
    ends = torch.zeros(total_length)
    for block_start in range(0, total_length, block_size):
        ends[block_start:block_start + block_size] = block_start + block_size

    def attention_mask(b, h, q_idx, kv_idx):
        if local_attn_size == -1:
            return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
        else:
            # Local attention: only attend within window
            return (kv_idx < ends[q_idx]) & \
                   (kv_idx >= ends[q_idx] - local_attn_size * frame_seqlen)
```

#### I2V Mask (Image-to-Video)

```python
def _prepare_blockwise_causal_attn_mask_i2v(...):
    """
    First frame is independent: [frame1][block1][block2]...
    Used for image-conditioned video generation
    """
    ends[:frame_seqlen] = frame_seqlen  # First frame only attends to itself
    # Subsequent blocks follow standard pattern
```

### 6.8 RoPE (Rotary Position Embedding)

```python
def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    """
    Apply RoPE with frame offset for streaming.

    Supports:
    - Triton fused 3-way kernel (fastest, D=128 only)
    - Triton rotary kernel
    - PyTorch fallback
    """
    if USE_TRITON_ROPE_FUSED and x.shape[-1] == 128:
        return triton_rope_fused_3way(x, grid_sizes, freqs, start_frame)

    # Compute cos/sin from frequency table
    cos, sin = get_rope_cos_sin(freqs_split, f, h, w, start_frame, ...)

    # Apply rotation
    x0, x1 = x[..., 0], x[..., 1]  # Interleaved
    x0_new = x0 * cos - x1 * sin
    x1_new = x0 * sin + x1 * cos
```

### 6.9 CausalWanModel Forward Methods

#### Inference (`_forward_inference`)

```python
def _forward_inference(self, x, t, context, seq_len, kv_cache, crossattn_cache,
                       current_start, cache_start, kv_cache_attention_bias):
    # 1. Patch embedding
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    x = torch.cat([u.flatten(2).transpose(1, 2) for u in x])

    # 2. Time embedding
    e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t))
    e0 = self.time_projection(e).unflatten(1, (6, self.dim))

    # 3. Text embedding
    context = self.text_embedding(torch.stack(context))

    # 4. Process through blocks
    for block_index, block in enumerate(self.blocks):
        x = block(x,
            e=e0,
            kv_cache=kv_cache[block_index],
            crossattn_cache=crossattn_cache[block_index],
            current_start=current_start,
            kv_cache_attention_bias=kv_cache_attention_bias,
            ...
        )

    # 5. Head projection
    x = self.head(x, e)

    # 6. Unpatchify
    return self.unpatchify(x, grid_sizes)
```

#### Training (`_forward_train`)

Similar but includes:
- Block mask construction
- Teacher forcing support
- Gradient checkpointing

---

## 7. Standard Model Reference (modules/model.py)

**Location**: `modules/model.py` (1116 lines)

This contains the non-causal version of the attention mechanism, used for:
- Reference implementation
- Cross-attention classes
- Shared utilities (RoPE, normalization, etc.)

### Key Exports Used by Causal Model

```python
from .model import (
    USE_TRITON_ROTARY,           # Flag for Triton rotary kernel
    USE_TRITON_ROPE_FUSED,       # Flag for fused RoPE
    WAN_CROSSATTENTION_CLASSES,  # T2V and I2V cross-attention
    MLPProj,                     # MLP projection for I2V
    WanLayerNorm,                # Layer normalization
    WanRMSNorm,                  # RMS normalization
    get_rope_cos_sin,            # Cached RoPE computation
    rope_apply,                  # RoPE application
    rope_params,                 # RoPE frequency parameters
    sinusoidal_embedding_1d,     # Timestep embedding
    triton_apply_rotary,         # Triton rotary kernel
    triton_rope_fused_3way,      # Fused 3-way RoPE kernel
)
```

---

## 8. VAE Components (modules/vae.py)

**Location**: `modules/vae.py` (779 lines)

The VAE (Variational Autoencoder) compresses video to/from latent space.

### 8.1 Architecture

```
WanVAE
├── encoder (Encoder3d)
│   ├── CausalConv3d layers
│   ├── Resample (downsample3d)
│   └── Attention blocks
│
├── decoder (Decoder3d)
│   ├── CausalConv3d layers
│   ├── Resample (upsample3d)
│   └── Attention blocks
│
└── standardization (mean/std arrays)
```

### 8.2 Causal 3D Convolution

```python
class CausalConv3d(nn.Conv3d):
    """
    3D convolution with causal masking in time dimension.
    Pads temporally so output only depends on past/current frames.
    """
    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None:
            x = torch.cat([cache_x, x], dim=2)  # Prepend cache
            padding[4] -= cache_x.shape[2]      # Reduce temporal padding
        x = F.pad(x, padding)
        return super().forward(x)
```

### 8.3 Resampling

```python
class Resample(nn.Module):
    """Spatial and temporal up/downsampling."""

    # Modes:
    # - upsample2d: 2× spatial only
    # - upsample3d: 2× spatial + 2× temporal
    # - downsample2d: 0.5× spatial only
    # - downsample3d: 0.5× spatial + 0.5× temporal
```

### 8.4 Compression Factors

| Dimension | Factor | Example |
|-----------|--------|---------|
| Spatial | 8× | 576×320 → 72×40 |
| Temporal | 4× | 12 frames → 3 latent frames |
| Channels | 3→16 | RGB → 16-channel latent |

---

## 9. Pipeline Blocks

### 9.1 PrepareContextFramesBlock

**Location**: `blocks/prepare_context_frames.py` (122 lines)

Manages sliding window buffers for context frames.

#### Purpose

After each generation block, this saves:
1. **Latent frames** (for KV cache recomputation)
2. **Decoded frames** (for re-encoding the first frame)

#### Inputs/Outputs

```python
# Inputs
latents: torch.Tensor           # Denoised latent frames
output_video: torch.Tensor      # Decoded video frames
current_start_frame: int        # Current block index

# Outputs
first_context_frame: torch.Tensor    # First frame (for anchoring)
context_frame_buffer: torch.Tensor   # Sliding window of latents
decoded_frame_buffer: torch.Tensor   # Sliding window of decoded frames
```

#### Logic

```python
def __call__(self, components, state):
    if current_start_frame == 0:
        # Save first frame for anchoring
        first_context_frame = latents[:, :1]

    # Update sliding window (FIFO)
    context_frame_buffer = torch.cat([
        context_frame_buffer,
        latents
    ], dim=1)[:, -max_size:]  # Keep only last max_size frames
```

### 9.2 RecomputeKVCacheBlock

**Location**: `blocks/recompute_kv_cache.py` (271 lines)

**This is critical for preventing error accumulation in streaming generation.**

#### Purpose

Before generating each new block, this:
1. Initializes a fresh KV cache
2. Fills it with clean context frames
3. Prepares attention masks

#### Why Recomputation?

Without recomputation, errors accumulate exponentially:
```
Frame 0 → Frame 3 → Frame 6 → Frame 9 → ...
  ↓         ↓         ↓         ↓
 OK     small err   bigger    drift!
```

With recomputation:
```
[clean frame 0] + [frame 6] + [frame 9] → fresh cache → Frame 12
                                            ↓
                                    errors reset!
```

#### Logic

```python
def __call__(self, components, state):
    if current_start_frame == 0:
        # First block: just initialize empty buffers
        context_frame_buffer = torch.zeros(...)
        decoded_frame_buffer = torch.zeros(...)
        return

    # Get context frames
    context_frames = get_context_frames(components, state)
    # Returns: [first_frame, buffer_frame1, buffer_frame2, ...]

    # Initialize fresh KV cache
    kv_cache = initialize_kv_cache(...)

    # Create attention mask for context
    block_mask = model._prepare_blockwise_causal_attn_mask(
        num_frames=num_context_frames,
        local_attn_size=-1,  # Global attention for recomputation
    )

    # Run forward pass to fill cache (timestep=0, no denoising)
    generator(
        noisy_image_or_video=context_frames,
        timestep=torch.zeros(...),  # No noise
        kv_cache=kv_cache,
    )

    # Clear block mask for subsequent generation
    model.block_mask = None
```

#### Context Frame Strategy

```python
def get_context_frames(components, state):
    if current_start_frame < kv_cache_num_frames:
        # Early in video: use original first frame
        return torch.cat([first_context_frame, context_frame_buffer])
    else:
        # Later: re-encode first frame from decoded buffer
        # (reduces drift from VAE encode/decode cycles)
        decoded_first = decoded_frame_buffer[:, :1]
        reencoded = vae.encode_to_latent(decoded_first)
        return torch.cat([reencoded, context_frame_buffer])
```

---

## 10. Data Flow and Execution

### 10.1 Single Call Flow

```
Input: prompts = [{"text": "a dog running", "weight": 1.0}]
       current_start_frame = 0 (first call)

─────────────────────────────────────────────────────────────

1. TextConditioningBlock
   prompts → text_encoder → conditioning_embeds [1, 512, 4096]

2. EmbeddingBlendingBlock
   (Optional blending with style/LoRA embeddings)

3. SetTimestepsBlock
   denoising_step_list = [1000, 750, 500, 250]

4. AutoPreprocessVideoBlock
   (No-op for T2V mode)

5. SetupCachesBlock
   Initialize kv_cache[32] (one per block)
   Initialize crossattn_cache[32]

6. AutoPrepareLatentsBlock
   Create random latents [1, 3, 16, 40, 72]  # 3 frames
   (batch, frames, channels, H/16, W/16)

7. RecomputeKVCacheBlock
   current_start_frame == 0 → Initialize empty buffers

8. DenoiseBlock
   For t in [1000, 750, 500, 250]:
     noise_pred = generator(latents, t, conditioning_embeds, kv_cache)
     latents = scheduler.step(noise_pred, t, latents)

9. DecodeBlock
   video = vae.decode(latents)  # [1, 3, 3, 320, 576]

10. PrepareContextFramesBlock
    first_context_frame = latents[:, :1]
    context_frame_buffer = latents

11. PrepareNextBlock
    current_start_frame += 3

─────────────────────────────────────────────────────────────

Output: video tensor [3, 320, 576, 3] (T, H, W, C)
```

### 10.2 Multi-Call Streaming

```
Call 1: current_start_frame=0
        └── Generate frames 0-2
            └── Cache filled with frames 0-2

Call 2: current_start_frame=3
        ├── RecomputeKVCache: [frame0] + [frame0-2] → fresh cache
        └── Generate frames 3-5
            └── Cache: [frame0][frame0-2][frame3-5]

Call 3: current_start_frame=6
        ├── RecomputeKVCache: [frame0*] + [frame3-5] → fresh cache
        │   (* re-encoded from decoded buffer)
        └── Generate frames 6-8

...continues...
```

### 10.3 KV Cache Memory Layout

```
KV Cache Structure (per attention block):
{
    "k": tensor[1, max_cache_size, num_heads, head_dim],
    "v": tensor[1, max_cache_size, num_heads, head_dim],
    "global_end_index": int,  # Total tokens seen
    "local_end_index": int,   # Current cache end
}

Memory Evolution:
┌────────────────────────────────────────────────────┐
│ Call 1: [frame0, frame1, frame2, _, _, _, _, _]    │
├────────────────────────────────────────────────────┤
│ Call 2: [recomputed context][frame3, frame4, ...] │
├────────────────────────────────────────────────────┤
│ Call 3: [recomputed context][frame6, frame7, ...] │
└────────────────────────────────────────────────────┘
```

---

## 11. Performance Optimizations

### 11.1 Kernel Optimizations

| Kernel | Purpose | Speedup |
|--------|---------|---------|
| `triton_rope_fused_3way` | Fused RoPE for Q, K, V | ~30% |
| `triton_kernel_b` | KV-cache attention bias | Baseline |
| `FA4/CUTE score_mod` | KV-bias in FlashAttention | 1.89× |
| Projection fusion | Combined Q, K, V linear | ~10% |

### 11.2 FP8 Quantization

```python
if quantization == Quantization.FP8_E4M3FN:
    from torchao.quantization.quant_api import (
        Float8DynamicActivationFloat8WeightConfig,
        quantize_,
    )
    quantize_(generator, Float8DynamicActivationFloat8WeightConfig(...))
```

Benefits:
- ~2× memory reduction
- ~1.5× speedup on supported hardware

### 11.3 Caching

| Cache | Purpose | Implementation |
|-------|---------|----------------|
| RoPE cos/sin | Avoid recomputation | LRU cache (32 entries) |
| FA4 score_mod | Compiled CUTE kernels | Dict keyed by (seqlen, block, bias) |
| cu_seqlens | FlashAttention sequence lengths | Dict by (device, batch, seqlen) |
| Block masks | Attention masks | functools.lru_cache |

### 11.4 Hardware Detection

```python
def _is_sm103() -> bool:
    """Detect B300 GPUs (SM103)."""
    return torch.cuda.get_device_capability(0) == (10, 3)

# Automatic backend selection
_KV_BIAS_BACKEND = "flash" if _is_sm103() else "triton"
```

### 11.5 Profiling

Enable with `PROFILE_ATTENTION=1`:

```python
with _ProfileBlock("self_attn_kv_bias"):
    x = attention_with_bias(...)

# At exit, prints:
# === Attention Profiling Report ===
#   self_attn: 150.2ms (45.3%) [100 calls, 1.50ms/call]
#   cross_attn: 80.1ms (24.2%)
#   ffn: 95.3ms (28.8%)
```

---

## 12. Usage Example (test.py)

### Basic Usage

```python
from scope.core.pipelines.krea_realtime_video import KreaRealtimeVideoPipeline
from scope.core.pipelines.utils import Quantization

# Configuration
config = OmegaConf.create({
    "model_dir": "/path/to/models",
    "generator_path": "krea-realtime-video-14b.safetensors",
    "text_encoder_path": "umt5-xxl-enc-fp8.safetensors",
    "tokenizer_path": "Wan2.1-T2V-1.3B/google/umt5-xxl",
    "vae_path": "Wan2.1_VAE.pth",
    "height": 320,
    "width": 576,
})

# Initialize pipeline (includes warmup)
pipeline = KreaRealtimeVideoPipeline(
    config,
    quantization=Quantization.FP8_E4M3FN,  # For 32GB VRAM
    compile=False,
    device=torch.device("cuda"),
    dtype=torch.bfloat16,
)

# Generate streaming video
outputs = []
for _ in range(27):  # 27 calls × 3 frames = 81 frames
    output = pipeline(
        prompts=[{"text": "A dog running in a field", "weight": 1.0}],
        kv_cache_attention_bias=0.3,
    )
    outputs.append(output.cpu())

# Export
video = torch.concat(outputs)
export_to_video(video.numpy(), "output.mp4", fps=16)
```

### Changing Prompts Mid-Stream

```python
# Prompt 1 for 10 blocks
for _ in range(10):
    output = pipeline(prompts=[{"text": "A sunny beach", "weight": 1.0}])
    outputs.append(output)

# Prompt 2 for next 10 blocks (continues same video)
for _ in range(10):
    output = pipeline(prompts=[{"text": "Sunset on the beach", "weight": 1.0}])
    outputs.append(output)
```

### Resetting the Pipeline

```python
# Force cache reset
output = pipeline(prompts=prompts, init_cache=True)
```

---

## Quick Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SCOPE_KV_CACHE_ATTENTION_BIAS` | 0.3 | Past-frame attention weight |
| `SCOPE_KV_BIAS_BACKEND` | auto | `fa4`, `flash`, `triton`, or `flex` |
| `PROFILE_ATTENTION` | 0 | Enable attention profiling |
| `DISABLE_FLEX_ATTENTION_COMPILE` | 0 | Use eager flex_attention |
| `TRITON_PTXAS_PATH` | auto | Path to ptxas for Triton |

### Key Dimensions

| Quantity | Value | Notes |
|----------|-------|-------|
| Video resolution | 320×576 (default) | Must be divisible by 16 |
| Frames per block | 3 | `num_frame_per_block` |
| Tokens per frame | 1560 | (H/16) × (W/16) = 40×39 ≈ 1560 |
| Model dimension | 2048 | Hidden size |
| Number of heads | 16 | Attention heads |
| Number of layers | 32 | Transformer blocks |
| FFN dimension | 8192 | 4× hidden size |

### Pipeline State Keys

| Key | Type | Description |
|-----|------|-------------|
| `prompts` | list[dict] | Text prompts with weights |
| `current_start_frame` | int | Frame index for current block |
| `kv_cache` | list[dict] | KV cache per layer |
| `kv_cache_attention_bias` | float | 0.3 = 30% attention to past |
| `denoising_step_list` | list[int] | Timesteps (default: [1000, 750, 500, 250]) |
| `latents` | Tensor | Current latent frames |
| `output_video` | Tensor | Decoded video output |
| `context_frame_buffer` | Tensor | Sliding window of past latents |
| `first_context_frame` | Tensor | Anchor frame for recomputation |

---

## Conclusion

The KREA Realtime Video Pipeline is a sophisticated streaming video generation system that achieves real-time performance through:

1. **Causal attention** with block-wise masking
2. **KV caching** with intelligent eviction
3. **Context recomputation** to prevent error drift
4. **Attention bias** to down-weight stale information
5. **Multiple optimized backends** for different hardware

The modular design allows easy customization and extension while maintaining high performance on modern GPUs.
