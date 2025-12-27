# Krea pipeline op profile

## Meta
- torch: `2.9.0+cu130` (cuda `13.0`)
- device: `NVIDIA B300 SXM6 AC` cc=(10, 3)
- compile: `True`
- kv_cache_attention_bias: `0.3`
- SCOPE_KV_BIAS_BACKEND: `fa4`
- profiled_wall_time_s: `7.334`

## Top ops (self CUDA)
| self_cuda_ms | pct | calls | key |
|---:|---:|---:|---|
| 135.991 | 18.21% | 200 | `## Call CompiledFxGraph f3xe5wyeoz3mjoa4uwronzbu27nnhqzgktf5hbdcuquz6tflrwl5 ##` |
| 128.409 | 17.20% | 1,200 | `aten::mm` |
| 60.675 | 8.13% | 800 | `nvjet_tst_256x128_64x5_2x2_2cta_h_bz_TNT` |
| 42.351 | 5.67% | 160 | `## Call CompiledFxGraph fed4ov6qe57tzjn5yggz6dav3tn4vnbpeuuzyim6fjrswsfn4qe6 ##` |
| 33.954 | 4.55% | 200 | `nvjet_tst_256x160_64x5_2x1_2cta_v_bz_TNT` |
| 33.780 | 4.52% | 200 | `nvjet_tst_128x256_64x6_2x1_2cta_v_bz_TNT` |
| 33.481 | 4.48% | 160 | `Torch-Compiled Region: 9/1` |
| 33.481 | 4.48% | 160 | `kernel_cutlass_kernel_flash_attncuteflash_fwd_sm100FlashAttentionForwardSm100_object_at__tensor0000o11101213_tensor0000o11101213_tensor0000o10111213_tensor0000o11101213_None_None_None_Non_0` |
| 31.308 | 4.19% | 430 | `aten::addmm` |
| 30.696 | 4.11% | 400 | `nvjet_tst_256x128_64x5_2x2_2cta_h_bz_bias_TNT` |
| 22.904 | 3.07% | 240 | `flash_attn::_flash_attn_forward` |
| 22.904 | 3.07% | 240 | `void flash::flash_fwd_kernel<Flash_fwd_kernel_traits<128, 128, 64, 4, false, false, cutlass::bfloat16_t, Flash_kernel_traits<128, 128, 64, 4, cutlass::bfloat16_t> >, false, false, false, false, false, true, false, false>(flash::Flash_fwd_params)` |
| 14.292 | 1.91% | 40 | `## Call CompiledFxGraph flwczbr2va6dg3efruela4opqksbqbcddxpjzr47fiv62qbso3w7 ##` |
| 10.583 | 1.42% | 40 | `## Call CompiledFxGraph fuerq27apmsb63irvfj2omfonzgokg7zwoewsmzlbgle7oevigjt ##` |
| 6.766 | 0.91% | 200 | `triton_poi_fused_addmm_gelu_view_3` |
| 6.766 | 0.91% | 200 | `triton_poi_fused_addmm_gelu_view_3` |
| 5.472 | 0.73% | 400 | `triton_red_fused__to_copy_add_addmm_mean_mul_pow_rsqrt_view_0` |
| 5.472 | 0.73% | 400 | `triton_red_fused__to_copy_add_addmm_mean_mul_pow_rsqrt_view_0` |
| 5.384 | 0.72% | 10,062 | `aten::copy_` |
| 5.316 | 0.71% | 200 | `triton_red_fused_add_mul_native_layer_norm_view_0` |
| 5.316 | 0.71% | 200 | `triton_red_fused_add_mul_native_layer_norm_view_0` |
| 5.184 | 0.69% | 200 | `triton_red_fused_add_addmm_mul_native_layer_norm_view_2` |
| 5.184 | 0.69% | 200 | `triton_red_fused_add_addmm_mul_native_layer_norm_view_2` |
| 5.128 | 0.69% | 400 | `rope_fused_3way_kernel_v2_0` |
| 5.128 | 0.69% | 400 | `rope_fused_3way_kernel_v2` |

## Stack filters
- stack_include: `['wan2_1/vae']`
- stack_exclude: `[]`

## Stack groups: `aten::cudnn_convolution`

Filtered totals: device_ms=0.076, calls=8

- count=1 device_ms=0.060 self_device_ms=0.049 cpu_ms=0.042
  - `<built-in method conv3d of type object at 0x7f1c8cfd6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_4`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: ResidualBlock_1`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(666): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Decoder3d_0`

- count=1 device_ms=0.008 self_device_ms=0.008 cpu_ms=0.025
  - `<built-in method conv2d of type object at 0x7f1c8cfd6c40>`
  - `torch/nn/modules/conv.py(530): _conv_forward`
  - `torch/nn/modules/conv.py(547): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Conv2d_0`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(465): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: AttentionBlock_0`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(666): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Decoder3d_0`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(875): stream_decode`

- count=1 device_ms=0.005 self_device_ms=0.005 cpu_ms=0.019
  - `<built-in method conv2d of type object at 0x7f1c8cfd6c40>`
  - `torch/nn/modules/conv.py(530): _conv_forward`
  - `torch/nn/modules/conv.py(547): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Conv2d_1`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(465): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: AttentionBlock_0`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(666): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Decoder3d_0`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(875): stream_decode`

- count=1 device_ms=0.003 self_device_ms=0.003 cpu_ms=0.035
  - `<built-in method conv3d of type object at 0x7f1c8cfd6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_0`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(875): stream_decode`
  - `scope/core/pipelines/wan2_1/vae/wan.py(262): decode_to_pixel`
  - `scope/core/pipelines/wan2_1/blocks/decode.py(47): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`
  - `scope/core/pipelines/krea_realtime_video/modular_blocks.py(173): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.049
  - `<built-in method conv2d of type object at 0x7f1c8cfd6c40>`
  - `torch/nn/modules/conv.py(530): _conv_forward`
  - `torch/nn/modules/conv.py(547): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Conv2d_2`
  - `torch/nn/modules/container.py(245): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Sequential_3`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(311): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Resample_0`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(666): forward`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.020
  - `<built-in method conv3d of type object at 0x7f1c8cfd6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_13`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: ResidualBlock_5`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(666): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Decoder3d_0`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.037
  - `<built-in method conv2d of type object at 0x7f1c8cfd6c40>`
  - `torch/nn/modules/conv.py(530): _conv_forward`
  - `torch/nn/modules/conv.py(547): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Conv2d_3`
  - `torch/nn/modules/container.py(245): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Sequential_4`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(311): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Resample_1`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(666): forward`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.449
  - `<built-in method conv2d of type object at 0x7f1c8cfd6c40>`
  - `torch/nn/modules/conv.py(530): _conv_forward`
  - `torch/nn/modules/conv.py(547): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Conv2d_4`
  - `torch/nn/modules/container.py(245): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Sequential_5`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(311): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Resample_2`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(666): forward`

## Stack groups: `aten::convolution`

Filtered totals: device_ms=3.020, calls=39

- count=1 device_ms=0.868 self_device_ms=0.000 cpu_ms=3.217
  - `<built-in method conv3d of type object at 0x7f1c8cfd6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_3`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: ResidualBlock_0`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(666): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Decoder3d_0`

- count=1 device_ms=0.865 self_device_ms=0.000 cpu_ms=3.228
  - `<built-in method conv3d of type object at 0x7f1c8cfd6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_2`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: ResidualBlock_0`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(666): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Decoder3d_0`

- count=1 device_ms=0.674 self_device_ms=0.000 cpu_ms=3.268
  - `<built-in method conv3d of type object at 0x7f1c8cfd6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_1`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(666): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Decoder3d_0`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(875): stream_decode`
  - `scope/core/pipelines/wan2_1/vae/wan.py(262): decode_to_pixel`
  - `scope/core/pipelines/wan2_1/blocks/decode.py(47): __call__`

- count=1 device_ms=0.481 self_device_ms=0.000 cpu_ms=3.386
  - `<built-in method conv3d of type object at 0x7f1c8cfd6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_5`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: ResidualBlock_1`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(666): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Decoder3d_0`

- count=1 device_ms=0.069 self_device_ms=0.000 cpu_ms=0.055
  - `<built-in method conv3d of type object at 0x7f1c8cfd6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_4`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: ResidualBlock_1`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(666): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Decoder3d_0`

- count=1 device_ms=0.041 self_device_ms=0.000 cpu_ms=0.052
  - `<built-in method conv2d of type object at 0x7f1c8cfd6c40>`
  - `torch/nn/modules/conv.py(530): _conv_forward`
  - `torch/nn/modules/conv.py(547): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Conv2d_0`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(465): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: AttentionBlock_0`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(666): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Decoder3d_0`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(875): stream_decode`

- count=1 device_ms=0.014 self_device_ms=0.000 cpu_ms=0.032
  - `<built-in method conv2d of type object at 0x7f1c8cfd6c40>`
  - `torch/nn/modules/conv.py(530): _conv_forward`
  - `torch/nn/modules/conv.py(547): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Conv2d_1`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(465): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: AttentionBlock_0`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(666): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Decoder3d_0`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(875): stream_decode`

- count=1 device_ms=0.009 self_device_ms=0.000 cpu_ms=0.069
  - `<built-in method conv3d of type object at 0x7f1c8cfd6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_0`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(875): stream_decode`
  - `scope/core/pipelines/wan2_1/vae/wan.py(262): decode_to_pixel`
  - `scope/core/pipelines/wan2_1/blocks/decode.py(47): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`
  - `scope/core/pipelines/krea_realtime_video/modular_blocks.py(173): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=3.518
  - `<built-in method conv3d of type object at 0x7f1c8cfd6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_6`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: ResidualBlock_2`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(666): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Decoder3d_0`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=3.176
  - `<built-in method conv3d of type object at 0x7f1c8cfd6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_7`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: ResidualBlock_2`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(666): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Decoder3d_0`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=3.175
  - `<built-in method conv3d of type object at 0x7f1c8cfd6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_8`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: ResidualBlock_3`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(666): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Decoder3d_0`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=3.165
  - `<built-in method conv3d of type object at 0x7f1c8cfd6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_9`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: ResidualBlock_3`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(666): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Decoder3d_0`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=3.143
  - `<built-in method conv3d of type object at 0x7f1c8cfd6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_10`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: ResidualBlock_4`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(666): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Decoder3d_0`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=3.194
  - `<built-in method conv3d of type object at 0x7f1c8cfd6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_11`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: ResidualBlock_4`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(666): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Decoder3d_0`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=6.348
  - `<built-in method conv3d of type object at 0x7f1c8cfd6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_12`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(311): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Resample_0`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(666): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Decoder3d_0`

## Stack groups: `aten::conv3d`

Filtered totals: device_ms=2.965, calls=34

- count=1 device_ms=0.868 self_device_ms=0.000 cpu_ms=3.218
  - `<built-in method conv3d of type object at 0x7f1c8cfd6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_3`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: ResidualBlock_0`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(666): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Decoder3d_0`

- count=1 device_ms=0.865 self_device_ms=0.000 cpu_ms=3.229
  - `<built-in method conv3d of type object at 0x7f1c8cfd6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_2`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: ResidualBlock_0`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(666): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Decoder3d_0`

- count=1 device_ms=0.674 self_device_ms=0.000 cpu_ms=3.268
  - `<built-in method conv3d of type object at 0x7f1c8cfd6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_1`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(666): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Decoder3d_0`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(875): stream_decode`
  - `scope/core/pipelines/wan2_1/vae/wan.py(262): decode_to_pixel`
  - `scope/core/pipelines/wan2_1/blocks/decode.py(47): __call__`

- count=1 device_ms=0.481 self_device_ms=0.000 cpu_ms=3.387
  - `<built-in method conv3d of type object at 0x7f1c8cfd6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_5`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: ResidualBlock_1`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(666): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Decoder3d_0`

- count=1 device_ms=0.069 self_device_ms=0.000 cpu_ms=0.056
  - `<built-in method conv3d of type object at 0x7f1c8cfd6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_4`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: ResidualBlock_1`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(666): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Decoder3d_0`

- count=1 device_ms=0.009 self_device_ms=0.000 cpu_ms=0.071
  - `<built-in method conv3d of type object at 0x7f1c8cfd6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_0`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(875): stream_decode`
  - `scope/core/pipelines/wan2_1/vae/wan.py(262): decode_to_pixel`
  - `scope/core/pipelines/wan2_1/blocks/decode.py(47): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`
  - `scope/core/pipelines/krea_realtime_video/modular_blocks.py(173): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=3.519
  - `<built-in method conv3d of type object at 0x7f1c8cfd6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_6`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: ResidualBlock_2`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(666): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Decoder3d_0`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=3.177
  - `<built-in method conv3d of type object at 0x7f1c8cfd6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_7`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: ResidualBlock_2`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(666): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Decoder3d_0`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=3.176
  - `<built-in method conv3d of type object at 0x7f1c8cfd6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_8`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: ResidualBlock_3`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(666): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Decoder3d_0`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=3.166
  - `<built-in method conv3d of type object at 0x7f1c8cfd6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_9`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: ResidualBlock_3`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(666): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Decoder3d_0`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=3.144
  - `<built-in method conv3d of type object at 0x7f1c8cfd6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_10`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: ResidualBlock_4`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(666): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Decoder3d_0`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=3.195
  - `<built-in method conv3d of type object at 0x7f1c8cfd6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_11`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: ResidualBlock_4`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(666): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Decoder3d_0`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=6.349
  - `<built-in method conv3d of type object at 0x7f1c8cfd6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_12`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(311): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Resample_0`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(666): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Decoder3d_0`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.044
  - `<built-in method conv3d of type object at 0x7f1c8cfd6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_13`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: ResidualBlock_5`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(666): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Decoder3d_0`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=3.206
  - `<built-in method conv3d of type object at 0x7f1c8cfd6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_14`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: ResidualBlock_5`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(666): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Decoder3d_0`

## Stack groups: `aten::conv2d`

Filtered totals: device_ms=0.054, calls=5

- count=1 device_ms=0.041 self_device_ms=0.000 cpu_ms=0.053
  - `<built-in method conv2d of type object at 0x7f1c8cfd6c40>`
  - `torch/nn/modules/conv.py(530): _conv_forward`
  - `torch/nn/modules/conv.py(547): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Conv2d_0`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(465): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: AttentionBlock_0`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(666): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Decoder3d_0`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(875): stream_decode`

- count=1 device_ms=0.014 self_device_ms=0.000 cpu_ms=0.033
  - `<built-in method conv2d of type object at 0x7f1c8cfd6c40>`
  - `torch/nn/modules/conv.py(530): _conv_forward`
  - `torch/nn/modules/conv.py(547): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Conv2d_1`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(465): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: AttentionBlock_0`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(666): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Decoder3d_0`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(875): stream_decode`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.066
  - `<built-in method conv2d of type object at 0x7f1c8cfd6c40>`
  - `torch/nn/modules/conv.py(530): _conv_forward`
  - `torch/nn/modules/conv.py(547): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Conv2d_2`
  - `torch/nn/modules/container.py(245): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Sequential_3`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(311): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Resample_0`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(666): forward`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.053
  - `<built-in method conv2d of type object at 0x7f1c8cfd6c40>`
  - `torch/nn/modules/conv.py(530): _conv_forward`
  - `torch/nn/modules/conv.py(547): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Conv2d_3`
  - `torch/nn/modules/container.py(245): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Sequential_4`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(311): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Resample_1`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(666): forward`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.496
  - `<built-in method conv2d of type object at 0x7f1c8cfd6c40>`
  - `torch/nn/modules/conv.py(530): _conv_forward`
  - `torch/nn/modules/conv.py(547): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Conv2d_4`
  - `torch/nn/modules/container.py(245): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Sequential_5`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(311): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Resample_2`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(666): forward`
