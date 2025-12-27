# WanVAE stream_decode op profile

## Meta
- torch: `2.9.0+cu130` (cuda `13.0`)
- device: `NVIDIA B300 SXM6 AC` cc=(10, 3)
- height,width: `320x576`
- latent_shape: `[1, 16, 3, 40, 72]`
- dtype: `bf16`
- iters: `1` (pre-iters `10`)
- cudnn.benchmark: `True`
- WANVAE_STREAM_DECODE_MODE: `chunk`
- WANVAE_DECODE_CHANNELS_LAST_3D: `1`
- WANVAE_CONV3D_IMPLICIT_SPATIAL_PADDING: `1`
- WANVAE_UPSAMPLE_FORCE_FP32: `None`
- profiled_wall_time_s: `0.427`

## Top ops (self CUDA)
| self_cuda_ms | pct | calls | key |
|---:|---:|---:|---|
| 0.000 | 0.00% | 29 | `aten::view` |
| 0.000 | 0.00% | 31 | `aten::div` |
| 0.000 | 0.00% | 46 | `aten::add` |
| 0.000 | 0.00% | 34 | `aten::conv3d` |
| 0.000 | 0.00% | 39 | `aten::convolution` |
| 0.000 | 0.00% | 39 | `aten::_convolution` |
| 0.000 | 0.00% | 19 | `aten::cudnn_convolution` |
| 0.000 | 0.00% | 33 | `aten::reshape` |
| 0.000 | 0.00% | 19 | `aten::add_` |
| 0.000 | 0.00% | 103 | `aten::slice` |
| 0.000 | 0.00% | 9,798 | `aten::as_strided` |
| 0.000 | 0.00% | 48 | `aten::clone` |
| 0.000 | 0.00% | 32 | `aten::empty_strided` |
| 0.000 | 0.00% | 4,851 | `aten::copy_` |
| 0.000 | 0.00% | 32 | `aten::to` |
| 0.000 | 0.00% | 34 | `aten::cat` |
| 0.000 | 0.00% | 71 | `aten::narrow` |
| 0.000 | 0.00% | 16 | `aten::contiguous` |
| 0.000 | 0.00% | 33 | `aten::empty_like` |
| 0.000 | 0.00% | 61 | `aten::empty` |
| 0.000 | 0.00% | 30 | `aten::linalg_vector_norm` |
| 0.000 | 0.00% | 30 | `aten::clamp_min` |
| 0.000 | 0.00% | 30 | `aten::expand_as` |
| 0.000 | 0.00% | 30 | `aten::expand` |
| 0.000 | 0.00% | 60 | `aten::mul` |

## Stack groups: `aten::cudnn_convolution`

Filtered totals: device_ms=0.000, calls=19

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.029
  - `<built-in method conv3d of type object at 0x7b6325fd6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_0`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.037
  - `<built-in method conv3d of type object at 0x7b6325fd6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_1`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.027
  - `<built-in method conv3d of type object at 0x7b6325fd6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_2`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.024
  - `<built-in method conv3d of type object at 0x7b6325fd6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_3`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.025
  - `<built-in method conv2d of type object at 0x7b6325fd6c40>`
  - `torch/nn/modules/conv.py(530): _conv_forward`
  - `torch/nn/modules/conv.py(547): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Conv2d_0`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(465): forward`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.017
  - `<built-in method conv2d of type object at 0x7b6325fd6c40>`
  - `torch/nn/modules/conv.py(530): _conv_forward`
  - `torch/nn/modules/conv.py(547): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Conv2d_1`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(465): forward`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.028
  - `<built-in method conv3d of type object at 0x7b6325fd6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_4`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.023
  - `<built-in method conv3d of type object at 0x7b6325fd6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_5`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.022
  - `<built-in method conv3d of type object at 0x7b6325fd6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_6`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.022
  - `<built-in method conv3d of type object at 0x7b6325fd6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_7`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.023
  - `<built-in method conv3d of type object at 0x7b6325fd6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_8`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.024
  - `<built-in method conv3d of type object at 0x7b6325fd6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_9`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.021
  - `<built-in method conv3d of type object at 0x7b6325fd6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_10`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.022
  - `<built-in method conv3d of type object at 0x7b6325fd6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_11`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.025
  - `<built-in method conv3d of type object at 0x7b6325fd6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_12`

## Stack groups: `aten::_convolution`

Filtered totals: device_ms=0.000, calls=39

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.055
  - `<built-in method conv3d of type object at 0x7b6325fd6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_0`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.050
  - `<built-in method conv3d of type object at 0x7b6325fd6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_1`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.040
  - `<built-in method conv3d of type object at 0x7b6325fd6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_2`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.035
  - `<built-in method conv3d of type object at 0x7b6325fd6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_3`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.037
  - `<built-in method conv2d of type object at 0x7b6325fd6c40>`
  - `torch/nn/modules/conv.py(530): _conv_forward`
  - `torch/nn/modules/conv.py(547): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Conv2d_0`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(465): forward`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.029
  - `<built-in method conv2d of type object at 0x7b6325fd6c40>`
  - `torch/nn/modules/conv.py(530): _conv_forward`
  - `torch/nn/modules/conv.py(547): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Conv2d_1`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(465): forward`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.040
  - `<built-in method conv3d of type object at 0x7b6325fd6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_4`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.034
  - `<built-in method conv3d of type object at 0x7b6325fd6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_5`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.032
  - `<built-in method conv3d of type object at 0x7b6325fd6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_6`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.041
  - `<built-in method conv3d of type object at 0x7b6325fd6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_7`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.049
  - `<built-in method conv3d of type object at 0x7b6325fd6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_8`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.035
  - `<built-in method conv3d of type object at 0x7b6325fd6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_9`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.032
  - `<built-in method conv3d of type object at 0x7b6325fd6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_10`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.033
  - `<built-in method conv3d of type object at 0x7b6325fd6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_11`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.036
  - `<built-in method conv3d of type object at 0x7b6325fd6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_12`

## Stack groups: `aten::upsample_trilinear3d`

(no events)

## Stack groups: `aten::native_group_norm`

(no events)
