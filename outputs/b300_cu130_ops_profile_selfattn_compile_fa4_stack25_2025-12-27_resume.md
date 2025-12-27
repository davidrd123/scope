# Krea pipeline op profile

## Meta
- torch: `2.9.0+cu130` (cuda `13.0`)
- device: `NVIDIA B300 SXM6 AC` cc=(10, 3)
- compile: `True`
- kv_cache_attention_bias: `0.3`
- SCOPE_KV_BIAS_BACKEND: `fa4`
- profiled_wall_time_s: `12.486`

## Top ops (self CUDA)
| self_cuda_ms | pct | calls | key |
|---:|---:|---:|---|
| 84.583 | 12.32% | 600 | `aten::mm` |
| 84.515 | 12.31% | 200 | `## Call CompiledFxGraph fzkr5p2ai4jlimh5nkeoy4kj6gag3djpy7rrfhfz6gvfo74pftp2 ##` |
| 69.664 | 10.15% | 630 | `aten::addmm` |
| 38.342 | 5.59% | 200 | `nvjet_tst_256x128_64x5_2x1_2cta_v_bz_bias_TNT` |
| 35.212 | 5.13% | 160 | `## Call CompiledFxGraph fe3p354nzwxb6oufl3yqqnv5n7oj6ozrogyabdrellckb4pv7r4i ##` |
| 34.666 | 5.05% | 200 | `nvjet_tst_256x160_64x5_2x1_2cta_v_bz_TNT` |
| 34.114 | 4.97% | 200 | `nvjet_tst_128x256_64x6_2x1_2cta_v_bz_TNT` |
| 33.507 | 4.88% | 160 | `Torch-Compiled Region: 9/1` |
| 33.507 | 4.88% | 160 | `kernel_cutlass_kernel_flash_attncuteflash_fwd_sm100FlashAttentionForwardSm100_object_at__tensor0000o11101213_tensor0000o11101213_tensor0000o10111213_tensor0000o11101213_None_None_None_Non_0` |
| 30.713 | 4.47% | 400 | `nvjet_tst_256x128_64x5_2x2_2cta_h_bz_bias_TNT` |
| 18.856 | 2.75% | 200 | `## Call CompiledFxGraph ffw4q4kngccow57pa2mkgee77ogpi55si5mvf26jkdld7scmyvs7 ##` |
| 18.326 | 2.67% | 240 | `## Call CompiledFxGraph fcceaggubatdtr6yp77cg7bal2pxb7hmqu76agprelqh5hx5asm4 ##` |
| 15.804 | 2.30% | 200 | `nvjet_tst_256x128_64x5_2x2_2cta_h_bz_TNT` |
| 10.837 | 1.58% | 4,049 | `aten::copy_` |
| 10.681 | 1.56% | 240 | `FlashAttnVarlenFunc` |
| 10.681 | 1.56% | 240 | `kernel_cutlass_kernel_flash_attncuteflash_fwd_sm100FlashAttentionForwardSm100_object_at__tensor000o111012_tensor000o111012_tensor000o101112_tensorptrbf16gmemalign16odiv81div8_None_tensorp_0` |
| 10.575 | 1.54% | 400 | `triton_red_fused_add_mul_native_layer_norm_view_0` |
| 10.575 | 1.54% | 400 | `triton_red_fused_add_mul_native_layer_norm_view_0` |
| 10.160 | 1.48% | 3,308 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1})` |
| 8.755 | 1.28% | 40 | `## Call CompiledFxGraph fk64ajjwdf2q6mjilnqgjewiklnrsdt2pevlvsuyisl466jtpaa3 ##` |
| 6.799 | 0.99% | 200 | `triton_poi_fused_addmm_gelu_view_1` |
| 6.799 | 0.99% | 200 | `triton_poi_fused_addmm_gelu_view_1` |
| 5.488 | 0.80% | 200 | `## Call CompiledFxGraph f732rfwuj3e43dof7xqnzwq6bnjpadzz7val5ryx64zqiuudjyof ##` |
| 5.095 | 0.74% | 200 | `## Call CompiledFxGraph fuh67mwwyfifvu65poxmdnfvc3zukk5l7ofvfw7wqcnloi5p64op ##` |
| 5.007 | 0.73% | 400 | `rope_fused_3way_kernel_v2_0` |

## Stack filters
- stack_include: `['CausalWanSelfAttention']`
- stack_exclude: `[]`

## Top ops (stack-filtered)
Filtered to stack frames matching include=['CausalWanSelfAttention'] exclude=[].

| device_ms | pct | calls | groups | key |
|---:|---:|---:|---:|---|
| 89.900 | 16.12% | 160 | 40 | `Torch-Compiled Region: 1/2` |
| 69.903 | 12.54% | 320 | 184 | `## Call CompiledFxGraph fe3p354nzwxb6oufl3yqqnv5n7oj6ozrogyabdrellckb4pv7r4i ##` |
| 53.842 | 9.66% | 400 | 178 | `aten::addmm` |
| 53.168 | 9.54% | 160 | 40 | `Torch-Compiled Region: 8/1` |
| 51.203 | 9.18% | 160 | 40 | `Torch-Compiled Region: 9/1` |
| 38.342 | 6.88% | 200 | 80 | `nvjet_tst_256x128_64x5_2x1_2cta_v_bz_bias_TNT` |
| 33.507 | 6.01% | 160 | 40 | `kernel_cutlass_kernel_flash_attncuteflash_fwd_sm100FlashAttentionForwardSm100_object_at__tensor0000o11101213_tensor0000o11101213_tensor0000o10111213_tensor0000o11101213_None_None_None_Non_0` |
| 17.378 | 3.12% | 80 | 80 | `## Call CompiledFxGraph fk64ajjwdf2q6mjilnqgjewiklnrsdt2pevlvsuyisl466jtpaa3 ##` |
| 17.328 | 3.11% | 40 | 40 | `Torch-Compiled Region: 1/1` |
| 15.501 | 2.78% | 200 | 98 | `nvjet_tst_256x128_64x5_2x2_2cta_h_bz_bias_TNT` |
| 12.387 | 2.22% | 160 | 58 | `aten::linear` |
| 10.015 | 1.80% | 800 | 538 | `## Call CompiledFxGraph f4oz2wvcxejxwpdxprvuut3twkm7x5q4nrlvgkq2bxu63do7rkav ##` |
| 8.196 | 1.47% | 40 | 40 | `Torch-Compiled Region: 21/0` |
| 7.703 | 1.38% | 40 | 40 | `Torch-Compiled Region: 22/0` |
| 6.227 | 1.12% | 80 | 80 | `## Call CompiledFxGraph fcceaggubatdtr6yp77cg7bal2pxb7hmqu76agprelqh5hx5asm4 ##` |
| 5.309 | 0.95% | 320 | 58 | `aten::copy_` |
| 5.208 | 0.93% | 400 | 160 | `triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_split_view_0` |
| 5.007 | 0.90% | 400 | 160 | `Torch-Compiled Region: 3/0` |
| 5.007 | 0.90% | 400 | 160 | `Torch-Compiled Region: 4/1` |
| 5.007 | 0.90% | 400 | 160 | `Torch-Compiled Region: 7/1` |
| 5.007 | 0.90% | 400 | 188 | `rope_fused_3way_kernel_v2_0` |
| 5.007 | 0.90% | 400 | 188 | `rope_fused_3way_kernel_v2` |
| 4.868 | 0.87% | 195 | 162 | `## Call CompiledFxGraph fuh67mwwyfifvu65poxmdnfvc3zukk5l7ofvfw7wqcnloi5p64op ##` |
| 4.738 | 0.85% | 400 | 160 | `triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_split_view_1` |
| 4.425 | 0.79% | 160 | 58 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1})` |

## Stack groups: `aten::copy_`

Filtered totals: device_ms=5.309, calls=480

- count=8 device_ms=0.133 self_device_ms=0.133 cpu_ms=0.089
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_35`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_35`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanDiffusionWrapper_0`
  - `scope/core/pipelines/wan2_1/blocks/denoise.py(258): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`
  - `scope/core/pipelines/krea_realtime_video/modular_blocks.py(173): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`
  - `scope/core/pipelines/krea_realtime_video/pipeline.py(302): _generate`
  - `scope/core/pipelines/krea_realtime_video/pipeline.py(296): __call__`
  - `profile_krea_pipeline_ops.py(319): main`
  - `profile_krea_pipeline_ops.py(575): <module>`

- count=8 device_ms=0.133 self_device_ms=0.133 cpu_ms=0.103
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_19`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_19`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanDiffusionWrapper_0`
  - `scope/core/pipelines/wan2_1/blocks/denoise.py(258): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`
  - `scope/core/pipelines/krea_realtime_video/modular_blocks.py(173): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`
  - `scope/core/pipelines/krea_realtime_video/pipeline.py(302): _generate`
  - `scope/core/pipelines/krea_realtime_video/pipeline.py(296): __call__`
  - `profile_krea_pipeline_ops.py(319): main`
  - `profile_krea_pipeline_ops.py(575): <module>`

- count=8 device_ms=0.133 self_device_ms=0.133 cpu_ms=0.083
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_0`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
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
  - `nn.Module: WanDiffusionWrapper_0`
  - `scope/core/pipelines/wan2_1/blocks/denoise.py(258): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`
  - `scope/core/pipelines/krea_realtime_video/modular_blocks.py(173): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`
  - `scope/core/pipelines/krea_realtime_video/pipeline.py(302): _generate`
  - `scope/core/pipelines/krea_realtime_video/pipeline.py(296): __call__`
  - `profile_krea_pipeline_ops.py(319): main`
  - `profile_krea_pipeline_ops.py(575): <module>`

- count=8 device_ms=0.133 self_device_ms=0.133 cpu_ms=0.089
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_9`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
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
  - `nn.Module: WanDiffusionWrapper_0`
  - `scope/core/pipelines/wan2_1/blocks/denoise.py(258): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`
  - `scope/core/pipelines/krea_realtime_video/modular_blocks.py(173): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`
  - `scope/core/pipelines/krea_realtime_video/pipeline.py(302): _generate`
  - `scope/core/pipelines/krea_realtime_video/pipeline.py(296): __call__`
  - `profile_krea_pipeline_ops.py(319): main`
  - `profile_krea_pipeline_ops.py(575): <module>`

- count=8 device_ms=0.133 self_device_ms=0.133 cpu_ms=0.097
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_22`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_22`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanDiffusionWrapper_0`
  - `scope/core/pipelines/wan2_1/blocks/denoise.py(258): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`
  - `scope/core/pipelines/krea_realtime_video/modular_blocks.py(173): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`
  - `scope/core/pipelines/krea_realtime_video/pipeline.py(302): _generate`
  - `scope/core/pipelines/krea_realtime_video/pipeline.py(296): __call__`
  - `profile_krea_pipeline_ops.py(319): main`
  - `profile_krea_pipeline_ops.py(575): <module>`

- count=8 device_ms=0.133 self_device_ms=0.133 cpu_ms=0.100
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_39`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_39`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanDiffusionWrapper_0`
  - `scope/core/pipelines/wan2_1/blocks/denoise.py(258): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`
  - `scope/core/pipelines/krea_realtime_video/modular_blocks.py(173): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`
  - `scope/core/pipelines/krea_realtime_video/pipeline.py(302): _generate`
  - `scope/core/pipelines/krea_realtime_video/pipeline.py(296): __call__`
  - `profile_krea_pipeline_ops.py(319): main`
  - `profile_krea_pipeline_ops.py(575): <module>`

- count=8 device_ms=0.133 self_device_ms=0.133 cpu_ms=0.090
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_30`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_30`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanDiffusionWrapper_0`
  - `scope/core/pipelines/wan2_1/blocks/denoise.py(258): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`
  - `scope/core/pipelines/krea_realtime_video/modular_blocks.py(173): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`
  - `scope/core/pipelines/krea_realtime_video/pipeline.py(302): _generate`
  - `scope/core/pipelines/krea_realtime_video/pipeline.py(296): __call__`
  - `profile_krea_pipeline_ops.py(319): main`
  - `profile_krea_pipeline_ops.py(575): <module>`

- count=8 device_ms=0.133 self_device_ms=0.133 cpu_ms=0.091
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_27`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_27`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanDiffusionWrapper_0`
  - `scope/core/pipelines/wan2_1/blocks/denoise.py(258): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`
  - `scope/core/pipelines/krea_realtime_video/modular_blocks.py(173): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`
  - `scope/core/pipelines/krea_realtime_video/pipeline.py(302): _generate`
  - `scope/core/pipelines/krea_realtime_video/pipeline.py(296): __call__`
  - `profile_krea_pipeline_ops.py(319): main`
  - `profile_krea_pipeline_ops.py(575): <module>`

- count=8 device_ms=0.133 self_device_ms=0.133 cpu_ms=0.090
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_20`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_20`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanDiffusionWrapper_0`
  - `scope/core/pipelines/wan2_1/blocks/denoise.py(258): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`
  - `scope/core/pipelines/krea_realtime_video/modular_blocks.py(173): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`
  - `scope/core/pipelines/krea_realtime_video/pipeline.py(302): _generate`
  - `scope/core/pipelines/krea_realtime_video/pipeline.py(296): __call__`
  - `profile_krea_pipeline_ops.py(319): main`
  - `profile_krea_pipeline_ops.py(575): <module>`

- count=8 device_ms=0.133 self_device_ms=0.133 cpu_ms=0.091
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_7`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
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
  - `nn.Module: WanDiffusionWrapper_0`
  - `scope/core/pipelines/wan2_1/blocks/denoise.py(258): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`
  - `scope/core/pipelines/krea_realtime_video/modular_blocks.py(173): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`
  - `scope/core/pipelines/krea_realtime_video/pipeline.py(302): _generate`
  - `scope/core/pipelines/krea_realtime_video/pipeline.py(296): __call__`
  - `profile_krea_pipeline_ops.py(319): main`
  - `profile_krea_pipeline_ops.py(575): <module>`

- count=8 device_ms=0.133 self_device_ms=0.133 cpu_ms=0.089
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_4`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
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
  - `nn.Module: WanDiffusionWrapper_0`
  - `scope/core/pipelines/wan2_1/blocks/denoise.py(258): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`
  - `scope/core/pipelines/krea_realtime_video/modular_blocks.py(173): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`
  - `scope/core/pipelines/krea_realtime_video/pipeline.py(302): _generate`
  - `scope/core/pipelines/krea_realtime_video/pipeline.py(296): __call__`
  - `profile_krea_pipeline_ops.py(319): main`
  - `profile_krea_pipeline_ops.py(575): <module>`

- count=8 device_ms=0.133 self_device_ms=0.133 cpu_ms=0.092
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_10`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
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
  - `nn.Module: WanDiffusionWrapper_0`
  - `scope/core/pipelines/wan2_1/blocks/denoise.py(258): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`
  - `scope/core/pipelines/krea_realtime_video/modular_blocks.py(173): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`
  - `scope/core/pipelines/krea_realtime_video/pipeline.py(302): _generate`
  - `scope/core/pipelines/krea_realtime_video/pipeline.py(296): __call__`
  - `profile_krea_pipeline_ops.py(319): main`
  - `profile_krea_pipeline_ops.py(575): <module>`
