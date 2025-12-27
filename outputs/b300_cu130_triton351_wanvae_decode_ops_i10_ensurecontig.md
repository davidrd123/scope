# WanVAE stream_decode op profile

## Meta
- torch: `2.9.0+cu130` (cuda `13.0`)
- device: `NVIDIA B300 SXM6 AC` cc=(10, 3)
- height,width: `320x576`
- latent_shape: `[1, 16, 3, 40, 72]`
- dtype: `bf16`
- iters: `10` (pre-iters `10`)
- cudnn.benchmark: `True`
- WANVAE_STREAM_DECODE_MODE: `chunk`
- WANVAE_DECODE_CHANNELS_LAST_3D: `1`
- WANVAE_CONV3D_IMPLICIT_SPATIAL_PADDING: `1`
- WANVAE_UPSAMPLE_FORCE_FP32: `None`
- profiled_wall_time_s: `0.524`

## Top ops (self CUDA)
| self_cuda_ms | pct | calls | key |
|---:|---:|---:|---|
| 0.000 | 0.00% | 470 | `aten::view` |
| 0.000 | 0.00% | 310 | `aten::div` |
| 0.000 | 0.00% | 460 | `aten::add` |
| 0.000 | 0.00% | 340 | `aten::conv3d` |
| 0.000 | 0.00% | 390 | `aten::convolution` |
| 0.000 | 0.00% | 390 | `aten::_convolution` |
| 0.000 | 0.00% | 390 | `aten::cudnn_convolution` |
| 0.000 | 0.00% | 530 | `aten::reshape` |
| 0.000 | 0.00% | 390 | `aten::add_` |
| 0.000 | 0.00% | 1,030 | `aten::slice` |
| 0.000 | 0.00% | 1,520 | `aten::as_strided` |
| 0.000 | 0.00% | 690 | `aten::clone` |
| 0.000 | 0.00% | 320 | `aten::empty_strided` |
| 0.000 | 0.00% | 690 | `aten::copy_` |
| 0.000 | 0.00% | 320 | `aten::to` |
| 0.000 | 0.00% | 340 | `aten::cat` |
| 0.000 | 0.00% | 710 | `aten::narrow` |
| 0.000 | 0.00% | 370 | `aten::contiguous` |
| 0.000 | 0.00% | 370 | `aten::empty_like` |
| 0.000 | 0.00% | 420 | `aten::empty` |
| 0.000 | 0.00% | 300 | `aten::linalg_vector_norm` |
| 0.000 | 0.00% | 300 | `aten::clamp_min` |
| 0.000 | 0.00% | 300 | `aten::expand_as` |
| 0.000 | 0.00% | 300 | `aten::expand` |
| 0.000 | 0.00% | 600 | `aten::mul` |
