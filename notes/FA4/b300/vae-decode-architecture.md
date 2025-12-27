# VAE Decode Architecture Map

> Status: Partial (cuDNN constraints documented, code audit pending)
> Priority: Medium — drives Conv3d→Conv2d hunt and cuDNN planning
> Date: 2025-12-26
> Source: [`claude01.md`](../DeepResearch/2025-12-26/B300_step_back/doc_ref_guide/claude01.md)

## Purpose

Annotated decode graph with ops, shapes, and markers for:
- Which ops are slow (from profiling)
- Which Conv3d have `kernel_size[0]==1` (Conv2d candidates)
- Which could fuse (conv+bias+activation)

---

## VAE Decode Flow (as implemented today)

**Locations (code):**
- Wrapper / profiling hooks: `src/scope/core/pipelines/wan2_1/vae/wan.py` (`WanVAEWrapper.decode_to_pixel`)
- Core implementation: `src/scope/core/pipelines/wan2_1/vae/modules/vae.py` (`WanVAE_.stream_decode`)

High-level flow (streaming decode path):

```
Input: latent z [B, C_latent, T, H, W]
  │
  ├─► apply_scale
  │     z = z / scale[1] + scale[0]
  │
  ├─► conv2 (pointwise conv)
  │     x = CausalConv3d(z_dim → z_dim, kernel_size=1)(z)
  │
  ├─► decoder (causal; uses feature caches)
  │     - first batch: decoder_first (t=1) + decoder_rest (t-1) + cat
  │     - later batches:
  │         - loop mode: per-frame decoder (slow)
  │         - chunk mode: decoder over full chunk (fast)
  │
  └─► output [B, 3, T, H', W']
```

Notes:
- `WANVAE_STREAM_DECODE_MODE=chunk` is the preferred mode for performance.
- `WanVAE_.stream_decode` has built-in inner profilers (`PROFILE_WANVAE_DECODE_INNER=1`) which can be used to fill the timing tables below.

---

## Key Building Blocks (from code)

- `CausalConv3d` (`src/scope/core/pipelines/wan2_1/vae/modules/vae.py`)
  - Optionally uses implicit spatial padding (lets cuDNN handle H/W padding; time padding remains explicit for causality).
- `ResidualBlock`
  - Two `CausalConv3d(..., kernel_size=3, padding=1)` (i.e. 3×3×3) + a `shortcut` that is either Identity or `CausalConv3d(..., kernel_size=1)`.
- `Resample`
  - Spatial resample is done per-frame via `nn.Conv2d` after rearranging to `(B*T, C, H, W)`.
  - Temporal resample uses `time_conv = CausalConv3d(..., kernel_size=(3,1,1))` (time-kernel=3, spatial 1×1).
- `AttentionBlock` (inside encoder/decoder at some scales)
  - Per-frame attention: reshapes to `(B*T, C, H, W)`, computes QKV via `nn.Conv2d`, runs `F.scaled_dot_product_attention`.

---

## Layer-by-Layer Breakdown

Fill in from code audit of VAE decoder:

| Layer | Type | Kernel Size | Stride | Time (ms) | Conv2d Candidate? | Fuse Candidate? |
|-------|------|-------------|--------|-----------|-------------------|-----------------|
| conv_in | Conv3d | (?, ?, ?) | (?, ?, ?) | ? | ? | ? |
| mid_block.conv1 | Conv3d | (?, ?, ?) | (?, ?, ?) | ? | ? | ? |
| mid_block.norm | GroupNorm | - | - | ? | - | conv+norm? |
| mid_block.act | SiLU | - | - | ? | - | norm+act? |
| up_block_0.conv | Conv3d | (?, ?, ?) | (?, ?, ?) | ? | ? | ? |
| up_block_0.upsample | Upsample/ConvT | ? | ? | ? | ? | ? |
| ... | ... | ... | ... | ... | ... | ... |
| conv_out | Conv3d | (?, ?, ?) | (?, ?, ?) | ? | ? | ? |

---

## Conv3d → Conv2d Candidates

From a quick code scan, these have `kernel_size[0] == 1` (time-kernel=1):

| Layer | Current | Proposed Change | Expected Savings |
|-------|---------|-----------------|------------------|
| `WanVAE_.conv2` | `CausalConv3d(z_dim→z_dim, kernel=1)` | per-frame `Conv2d(z_dim→z_dim, kernel=1)` | TBD |
| `ResidualBlock.shortcut` | `CausalConv3d(in→out, kernel=1)` | per-frame `Conv2d(in→out, kernel=1)` | TBD |

**Pattern to look for:**
```python
# If kernel_size[0] == 1 and stride[0] == 1:
nn.Conv3d(..., kernel_size=(1, k, k), stride=(1, s, s))
# Can become:
# for t in range(T):
#     out[:, :, t] = conv2d(x[:, :, t])
```

Reality check:
- Most of the heavy decode path uses `kernel_size=3` (3×3×3), so “Conv3d→Conv2d” may be a limited lever in VAE decode compared to where it paid off (patch embedding).
- Still, it’s worth auditing **time-kernel=1** sites because they’re the easiest safe rewrite and can remove unexpected slow paths.

---

## Fusion Candidates

| Fusion | Ops | Pattern | cuDNN Support? |
|--------|-----|---------|----------------|
| conv+bias+relu | Conv3d + bias + ReLU | Standard | Yes |
| conv+groupnorm | Conv3d + GroupNorm | Less common | Maybe |
| conv+silu | Conv3d + SiLU | | Check |
| upsample+conv | Upsample + Conv3d | | Transposed conv? |

---

## cuDNN Graph Planning Notes

For treating decode as a planned subsystem.

### cuDNN Frontend vs Backend

| API | Use When | Code Volume |
|-----|----------|-------------|
| **Frontend** (preferred) | New code, complex graphs | 5-10× less code |
| **Backend** | Legacy, pure C interface | Verbose |

**Frontend advantages:** RAII, error handling, errata filters, autotuning built-in.

### Potential Graph Boundaries

| Scope | Risk | Benefit |
|-------|------|---------|
| Entire decode | High (debug hard) | Maximum fusion |
| Per-resolution block | Medium | Good balance |
| Individual conv+act | Low | Incremental wins |

### cuDNN Constraints for Conv3d (from doc_ref_guide)

| # | Constraint | Details |
|---|------------|---------|
| 1 | **Conv3d uses NCDHW** | 5D tensors: (N, C, D, H, W) for channel-first |
| 2 | **NDHWC for Tensor Cores** | Transform to channels-last for performance |
| 3 | **Channel alignment** | FP16: **multiples of 8**; INT8: multiples of 16 |
| 4 | **128-bit alignment minimum** | 1024-bit alignment is better |
| 5 | **In-place blocked for multi-node** | Input/output UIDs can't match in graphs with >1 node |
| 6 | **Virtual tensors enable fusion** | `set_is_virtual(true)` for intermediates |
| 7 | **3D heuristics fall back** | `CUDNN_HEUR_MODE_B` → `CUDNN_HEUR_MODE_A` for 3D conv |
| 8 | **FFT/Winograd block graph capture** | Filter engines with `SUPPORTS_CUDA_GRAPH_NATIVE_API` |
| 9 | **Runtime compilation overhead** | Engines with `RUNTIME_COMPILATION` have higher init time |

### Frontend API Pattern for Conv3d

```cpp
namespace fe = cudnn_frontend;
auto graph = fe::graph::Graph();
graph.set_io_data_type(fe::DataType_t::HALF)
     .set_compute_data_type(fe::DataType_t::FLOAT);

// 5D input: (N, C, D, H, W) — NCDHW
auto X = graph.tensor(fe::graph::Tensor_attributes()
    .set_dim({N, C, D, H, W})
    .set_stride({C*D*H*W, D*H*W, H*W, W, 1}));  // Contiguous NCDHW

auto conv_options = fe::graph::Conv_fprop_attributes()
    .set_padding({padD, padH, padW})
    .set_stride({strideD, strideH, strideW})
    .set_dilation({dilD, dilH, dilW});

auto Y = graph.conv_fprop(X, W, conv_options);

// Mark intermediate as virtual to enable fusion
Y.set_is_virtual(true);

// Add activation
auto Z = graph.pointwise(Y, fe::graph::Pointwise_attributes()
    .set_mode(fe::PointwiseMode_t::RELU));

graph.validate().build_operation_graph(handle);
graph.create_execution_plans({fe::HeurMode_t::A});  // Use mode A for 3D
graph.build_plans(handle);
```

### Graph Capture Filtering

```cpp
// Filter for graph-capture-compatible engines
auto plans = graph.create_execution_plans({fe::HeurMode_t::A});
plans.filter([](const auto& plan) {
    return plan.get_behavior_notes().count(
        CUDNN_BEHAVIOR_NOTE_SUPPORTS_CUDA_GRAPH_NATIVE_API) > 0;
});
```

### Reference Docs

| Resource | URL |
|----------|-----|
| Frontend Developer Guide | https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/developer/overview.html |
| Backend Graph API | https://docs.nvidia.com/deeplearning/cudnn/backend/v9.7.0/developer/graph-api.html |
| Support Matrix | https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/support-matrix.html |
| cudnn-frontend GitHub | https://github.com/NVIDIA/cudnn-frontend |
| Sample: conv_sample.cpp | `samples/cpp/conv_sample.cpp` in cudnn-frontend repo |

---

## Profiling Data

### Decode Block Timing

From `profile_krea_pipeline_blocks.py`:

| Block | Time (ms) | % of Pipeline |
|-------|-----------|---------------|
| vae_decode | ? | ? |
| vae_decode_inner | ? | ? |

### Decode Op Timing

From `profile_krea_pipeline_ops.py --with-stack`:

| Op | Time (ms) | Call Count | Top Stack |
|----|-----------|------------|-----------|
| aten::conv3d | ? | ? | ? |
| aten::group_norm | ? | ? | ? |
| aten::silu | ? | ? | ? |
| aten::upsample_* | ? | ? | ? |
| aten::copy_ | ? | ? | ? |

---

## Action Items

1. [ ] **Code audit:** Walk through VAE decoder, document each layer
2. [ ] **Profile:** Run ops profiler focused on decode
3. [ ] **Identify Conv2d candidates:** grep for `kernel_size=(1,` or `kernel_size=[1,`
4. [ ] **Test Conv2d rewrite:** One layer, measure impact
5. [ ] **cuDNN graph experiment:** Try planning smallest fusion

---

## References

- VAE source: `src/scope/core/pipelines/wan2_1/vae/`
- Patch-embed Conv2d fix: (commit reference)
- cuDNN frontend: https://github.com/NVIDIA/cudnn-frontend
