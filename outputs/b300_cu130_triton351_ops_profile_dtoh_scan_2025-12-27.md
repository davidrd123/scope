# Krea pipeline op profile

## Meta
- torch: `2.9.0+cu130` (cuda `13.0`)
- device: `NVIDIA B300 SXM6 AC` cc=(10, 3)
- compile: `True`
- kv_cache_attention_bias: `0.3`
- SCOPE_KV_BIAS_BACKEND: `fa4`
- profiled_wall_time_s: `12.746`

## Top ops (self CUDA)
| self_cuda_ms | pct | calls | key |
|---:|---:|---:|---|
| 82.760 | 12.19% | 600 | `aten::mm` |
| 82.565 | 12.16% | 200 | `## Call CompiledFxGraph fzkr5p2ai4jlimh5nkeoy4kj6gag3djpy7rrfhfz6gvfo74pftp2 ##` |
| 68.430 | 10.08% | 630 | `aten::addmm` |
| 37.472 | 5.52% | 200 | `nvjet_tst_256x128_64x5_2x1_2cta_v_bz_bias_TNT` |
| 34.418 | 5.07% | 160 | `## Call CompiledFxGraph fe3p354nzwxb6oufl3yqqnv5n7oj6ozrogyabdrellckb4pv7r4i ##` |
| 33.643 | 4.95% | 200 | `nvjet_tst_256x160_64x5_2x1_2cta_v_bz_TNT` |
| 33.460 | 4.93% | 200 | `nvjet_tst_128x256_64x6_2x1_2cta_v_bz_TNT` |
| 33.379 | 4.92% | 160 | `Torch-Compiled Region: 9/1` |
| 33.379 | 4.92% | 160 | `kernel_cutlass_kernel_flash_attncuteflash_fwd_sm100FlashAttentionForwardSm100_object_at__tensor0000o11101213_tensor0000o11101213_tensor0000o10111213_tensor0000o11101213_None_None_None_Non_0` |
| 30.345 | 4.47% | 400 | `nvjet_tst_256x128_64x5_2x2_2cta_h_bz_bias_TNT` |
| 18.676 | 2.75% | 200 | `## Call CompiledFxGraph ffw4q4kngccow57pa2mkgee77ogpi55si5mvf26jkdld7scmyvs7 ##` |
| 18.104 | 2.67% | 240 | `## Call CompiledFxGraph fcceaggubatdtr6yp77cg7bal2pxb7hmqu76agprelqh5hx5asm4 ##` |
| 15.657 | 2.31% | 200 | `nvjet_tst_256x128_64x5_2x2_2cta_h_bz_TNT` |
| 10.863 | 1.60% | 4,049 | `aten::copy_` |
| 10.696 | 1.57% | 240 | `FlashAttnVarlenFunc` |
| 10.696 | 1.57% | 240 | `kernel_cutlass_kernel_flash_attncuteflash_fwd_sm100FlashAttentionForwardSm100_object_at__tensor000o111012_tensor000o111012_tensor000o101112_tensorptrbf16gmemalign16odiv81div8_None_tensorp_0` |
| 10.549 | 1.55% | 400 | `triton_red_fused_add_mul_native_layer_norm_view_0` |
| 10.549 | 1.55% | 400 | `triton_red_fused_add_mul_native_layer_norm_view_0` |
| 10.183 | 1.50% | 3,239 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1})` |
| 8.625 | 1.27% | 40 | `## Call CompiledFxGraph fk64ajjwdf2q6mjilnqgjewiklnrsdt2pevlvsuyisl466jtpaa3 ##` |
| 6.735 | 0.99% | 200 | `triton_poi_fused_addmm_gelu_view_1` |
| 6.735 | 0.99% | 200 | `triton_poi_fused_addmm_gelu_view_1` |
| 5.458 | 0.80% | 200 | `## Call CompiledFxGraph f732rfwuj3e43dof7xqnzwq6bnjpadzz7val5ryx64zqiuudjyof ##` |
| 5.055 | 0.74% | 200 | `## Call CompiledFxGraph fuh67mwwyfifvu65poxmdnfvc3zukk5l7ofvfw7wqcnloi5p64op ##` |
| 4.981 | 0.73% | 400 | `rope_fused_3way_kernel_v2_0` |

## Top ops (stack-filtered)
| device_ms | pct | calls | groups | key |
|---:|---:|---:|---:|---|
| 194.100 | 11.91% | 160 | 40 | `Torch-Compiled Region: 0/2` |
| 163.695 | 10.04% | 400 | 4 | `## Call CompiledFxGraph fzkr5p2ai4jlimh5nkeoy4kj6gag3djpy7rrfhfz6gvfo74pftp2 ##` |
| 127.284 | 7.81% | 200 | 40 | `Torch-Compiled Region: 13/1` |
| 88.732 | 5.44% | 160 | 40 | `Torch-Compiled Region: 1/2` |
| 82.760 | 5.08% | 600 | 41 | `aten::mm` |
| 81.130 | 4.98% | 200 | 40 | `Torch-Compiled Region: 19/0` |
| 68.430 | 4.20% | 630 | 128 | `aten::addmm` |
| 68.248 | 4.19% | 320 | 160 | `## Call CompiledFxGraph fe3p354nzwxb6oufl3yqqnv5n7oj6ozrogyabdrellckb4pv7r4i ##` |
| 52.893 | 3.24% | 160 | 40 | `Torch-Compiled Region: 8/1` |
| 50.917 | 3.12% | 160 | 40 | `Torch-Compiled Region: 9/1` |
| 43.596 | 2.67% | 40 | 40 | `Torch-Compiled Region: 0/1` |
| 40.696 | 2.50% | 200 | 40 | `Torch-Compiled Region: 14/1` |
| 37.472 | 2.30% | 200 | 80 | `nvjet_tst_256x128_64x5_2x1_2cta_v_bz_bias_TNT` |
| 36.960 | 2.27% | 400 | 157 | `## Call CompiledFxGraph ffw4q4kngccow57pa2mkgee77ogpi55si5mvf26jkdld7scmyvs7 ##` |
| 36.208 | 2.22% | 480 | 34 | `## Call CompiledFxGraph fcceaggubatdtr6yp77cg7bal2pxb7hmqu76agprelqh5hx5asm4 ##` |
| 33.643 | 2.06% | 200 | 1 | `nvjet_tst_256x160_64x5_2x1_2cta_v_bz_TNT` |
| 33.460 | 2.05% | 200 | 1 | `nvjet_tst_128x256_64x6_2x1_2cta_v_bz_TNT` |
| 33.379 | 2.05% | 160 | 40 | `kernel_cutlass_kernel_flash_attncuteflash_fwd_sm100FlashAttentionForwardSm100_object_at__tensor0000o11101213_tensor0000o11101213_tensor0000o10111213_tensor0000o11101213_None_None_None_Non_0` |
| 30.345 | 1.86% | 400 | 42 | `nvjet_tst_256x128_64x5_2x2_2cta_h_bz_bias_TNT` |
| 21.097 | 1.29% | 800 | 4 | `triton_red_fused_add_mul_native_layer_norm_view_0` |
| 17.136 | 1.05% | 40 | 40 | `Torch-Compiled Region: 1/1` |
| 17.102 | 1.05% | 80 | 80 | `## Call CompiledFxGraph fk64ajjwdf2q6mjilnqgjewiklnrsdt2pevlvsuyisl466jtpaa3 ##` |
| 15.657 | 0.96% | 200 | 40 | `nvjet_tst_256x128_64x5_2x2_2cta_h_bz_TNT` |
| 15.024 | 0.92% | 200 | 40 | `Torch-Compiled Region: 18/0` |
| 13.470 | 0.83% | 400 | 2 | `triton_poi_fused_addmm_gelu_view_1` |
| 12.854 | 0.79% | 190 | 46 | `aten::linear` |
| 10.917 | 0.67% | 400 | 94 | `## Call CompiledFxGraph f732rfwuj3e43dof7xqnzwq6bnjpadzz7val5ryx64zqiuudjyof ##` |
| 10.863 | 0.67% | 3,842 | 73 | `aten::copy_` |
| 10.696 | 0.66% | 240 | 80 | `FlashAttnVarlenFunc` |
| 10.696 | 0.66% | 240 | 80 | `kernel_cutlass_kernel_flash_attncuteflash_fwd_sm100FlashAttentionForwardSm100_object_at__tensor000o111012_tensor000o111012_tensor000o101112_tensorptrbf16gmemalign16odiv81div8_None_tensorp_0` |
| 10.183 | 0.62% | 3,239 | 71 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1})` |
| 9.963 | 0.61% | 800 | 117 | `## Call CompiledFxGraph f4oz2wvcxejxwpdxprvuut3twkm7x5q4nrlvgkq2bxu63do7rkav ##` |
| 9.599 | 0.59% | 400 | 173 | `## Call CompiledFxGraph fuh67mwwyfifvu65poxmdnfvc3zukk5l7ofvfw7wqcnloi5p64op ##` |
| 8.275 | 0.51% | 400 | 2 | `triton_red_fused_add_mul_native_layer_norm_split_unsqueeze_view_1` |
| 8.157 | 0.50% | 40 | 40 | `Torch-Compiled Region: 21/0` |
| 7.662 | 0.47% | 40 | 40 | `Torch-Compiled Region: 22/0` |
| 7.388 | 0.45% | 200 | 40 | `Torch-Compiled Region: 15/1` |
| 7.289 | 0.45% | 20 | 17 | `aten::convolution` |
| 7.289 | 0.45% | 20 | 17 | `aten::_convolution` |
| 6.919 | 0.42% | 13 | 13 | `aten::conv3d` |
| 6.683 | 0.41% | 8 | 8 | `aten::slow_conv_dilated3d` |
| 6.238 | 0.38% | 3,549 | 13 | `aten::fill_` |
| 5.255 | 0.32% | 400 | 86 | `triton_red_fused__to_copy_add_addmm_mean_mul_pow_rsqrt_view_0` |
| 4.981 | 0.31% | 400 | 160 | `Torch-Compiled Region: 3/0` |
| 4.981 | 0.31% | 400 | 165 | `Torch-Compiled Region: 4/1` |
| 4.981 | 0.31% | 400 | 178 | `Torch-Compiled Region: 7/1` |
| 4.981 | 0.31% | 400 | 22 | `rope_fused_3way_kernel_v2_0` |
| 4.981 | 0.31% | 400 | 22 | `rope_fused_3way_kernel_v2` |
| 4.952 | 0.30% | 400 | 160 | `triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_split_view_0` |
| 4.719 | 0.29% | 400 | 160 | `triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_split_view_1` |
| 4.403 | 0.27% | 400 | 2 | `triton_poi_fused_add_addmm_mul_view_2` |
| 3.931 | 0.24% | 40 | 40 | `Torch-Compiled Region: 23/0` |
| 3.931 | 0.24% | 40 | 40 | `Torch-Compiled Region: 15/2` |
| 3.080 | 0.19% | 40 | 40 | `Torch-Compiled Region: 24/0` |
| 2.541 | 0.16% | 400 | 54 | `## Call CompiledFxGraph fy2diovumaytikxgbn47fhbd5ncpe2aly5adrlxorw2ifhz2izcc ##` |
| 1.562 | 0.10% | 80 | 7 | `## Call CompiledFxGraph fz2xsbnhyncv5m3nm7v2bf6vozcxvz5hjd4576llteuddvmlokvy ##` |
| 1.103 | 0.07% | 7 | 7 | `void at::native::vol2col_kernel<c10::BFloat16>(long, c10::BFloat16 const*, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, c10::BFloat16*)` |
| 0.999 | 0.06% | 81 | 2 | `aten::zero_` |
| 0.997 | 0.06% | 80 | 1 | `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul> >(int, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul>)` |
| 0.937 | 0.06% | 185 | 50 | `Memcpy DtoD (Device -> Device)` |

## Stack groups: `aten::item`

Filtered totals: device_ms=0.000, calls=1492

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.002
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanDiffusionWrapper_0`
  - `scope/core/pipelines/krea_realtime_video/blocks/recompute_kv_cache.py(133): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`
  - `scope/core/pipelines/krea_realtime_video/modular_blocks.py(173): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`

- count=5 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.007
  - `<built-in method item of Tensor object at 0x74a4ec374e60>`
  - `torch/_dynamo/guards.py(1632): <lambda>`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_0`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=5 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.008
  - `<built-in method item of Tensor object at 0x74a4ec374e60>`
  - `torch/_dynamo/guards.py(1632): <lambda>`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_1`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=5 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.008
  - `<built-in method item of Tensor object at 0x74a4ec374e60>`
  - `torch/_dynamo/guards.py(1632): <lambda>`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_2`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=5 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.008
  - `<built-in method item of Tensor object at 0x74a4ec374e60>`
  - `torch/_dynamo/guards.py(1632): <lambda>`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_3`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=5 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.008
  - `<built-in method item of Tensor object at 0x74a4ec374e60>`
  - `torch/_dynamo/guards.py(1632): <lambda>`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_4`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=5 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.008
  - `<built-in method item of Tensor object at 0x74a4ec374e60>`
  - `torch/_dynamo/guards.py(1632): <lambda>`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_5`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=5 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.008
  - `<built-in method item of Tensor object at 0x74a4ec374e60>`
  - `torch/_dynamo/guards.py(1632): <lambda>`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_6`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=5 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.008
  - `<built-in method item of Tensor object at 0x74a4ec374e60>`
  - `torch/_dynamo/guards.py(1632): <lambda>`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_7`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=5 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.008
  - `<built-in method item of Tensor object at 0x74a4ec374e60>`
  - `torch/_dynamo/guards.py(1632): <lambda>`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_8`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=5 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.008
  - `<built-in method item of Tensor object at 0x74a4ec374e60>`
  - `torch/_dynamo/guards.py(1632): <lambda>`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_9`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=5 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.008
  - `<built-in method item of Tensor object at 0x74a4ec374e60>`
  - `torch/_dynamo/guards.py(1632): <lambda>`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_10`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=5 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.008
  - `<built-in method item of Tensor object at 0x74a4ec374e60>`
  - `torch/_dynamo/guards.py(1632): <lambda>`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_11`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=5 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.008
  - `<built-in method item of Tensor object at 0x74a4ec374e60>`
  - `torch/_dynamo/guards.py(1632): <lambda>`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_12`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=5 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.008
  - `<built-in method item of Tensor object at 0x74a4ec374e60>`
  - `torch/_dynamo/guards.py(1632): <lambda>`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_13`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

## Stack groups: `aten::_local_scalar_dense`

Filtered totals: device_ms=0.000, calls=1492

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.001
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanDiffusionWrapper_0`
  - `scope/core/pipelines/krea_realtime_video/blocks/recompute_kv_cache.py(133): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`
  - `scope/core/pipelines/krea_realtime_video/modular_blocks.py(173): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`

- count=5 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.003
  - `<built-in method item of Tensor object at 0x74a4ec374e60>`
  - `torch/_dynamo/guards.py(1632): <lambda>`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_0`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=5 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.003
  - `<built-in method item of Tensor object at 0x74a4ec374e60>`
  - `torch/_dynamo/guards.py(1632): <lambda>`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_1`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=5 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.003
  - `<built-in method item of Tensor object at 0x74a4ec374e60>`
  - `torch/_dynamo/guards.py(1632): <lambda>`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_2`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=5 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.003
  - `<built-in method item of Tensor object at 0x74a4ec374e60>`
  - `torch/_dynamo/guards.py(1632): <lambda>`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_3`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=5 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.003
  - `<built-in method item of Tensor object at 0x74a4ec374e60>`
  - `torch/_dynamo/guards.py(1632): <lambda>`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_4`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=5 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.003
  - `<built-in method item of Tensor object at 0x74a4ec374e60>`
  - `torch/_dynamo/guards.py(1632): <lambda>`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_5`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=5 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.003
  - `<built-in method item of Tensor object at 0x74a4ec374e60>`
  - `torch/_dynamo/guards.py(1632): <lambda>`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_6`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=5 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.003
  - `<built-in method item of Tensor object at 0x74a4ec374e60>`
  - `torch/_dynamo/guards.py(1632): <lambda>`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_7`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=5 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.003
  - `<built-in method item of Tensor object at 0x74a4ec374e60>`
  - `torch/_dynamo/guards.py(1632): <lambda>`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_8`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=5 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.003
  - `<built-in method item of Tensor object at 0x74a4ec374e60>`
  - `torch/_dynamo/guards.py(1632): <lambda>`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_9`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=5 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.003
  - `<built-in method item of Tensor object at 0x74a4ec374e60>`
  - `torch/_dynamo/guards.py(1632): <lambda>`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_10`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=5 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.003
  - `<built-in method item of Tensor object at 0x74a4ec374e60>`
  - `torch/_dynamo/guards.py(1632): <lambda>`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_11`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=5 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.003
  - `<built-in method item of Tensor object at 0x74a4ec374e60>`
  - `torch/_dynamo/guards.py(1632): <lambda>`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_12`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=5 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.003
  - `<built-in method item of Tensor object at 0x74a4ec374e60>`
  - `torch/_dynamo/guards.py(1632): <lambda>`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_13`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

## Stack groups: `Memcpy DtoH (Device -> Pinned)`

(no events)
