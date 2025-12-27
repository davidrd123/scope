# Krea pipeline op profile

## Meta
- torch: `2.9.0+cu130` (cuda `13.0`)
- device: `NVIDIA B300 SXM6 AC` cc=(10, 3)
- compile: `True`
- kv_cache_attention_bias: `0.3`
- SCOPE_KV_BIAS_BACKEND: `fa4`
- profiled_wall_time_s: `7.533`

## Top ops (self CUDA)
| self_cuda_ms | pct | calls | key |
|---:|---:|---:|---|
| 136.693 | 18.20% | 200 | `## Call CompiledFxGraph f3xe5wyeoz3mjoa4uwronzbu27nnhqzgktf5hbdcuquz6tflrwl5 ##` |
| 129.123 | 17.19% | 1,200 | `aten::mm` |
| 60.976 | 8.12% | 800 | `nvjet_tst_256x128_64x5_2x2_2cta_h_bz_TNT` |
| 42.508 | 5.66% | 160 | `## Call CompiledFxGraph fed4ov6qe57tzjn5yggz6dav3tn4vnbpeuuzyim6fjrswsfn4qe6 ##` |
| 34.222 | 4.56% | 200 | `nvjet_tst_256x160_64x5_2x1_2cta_v_bz_TNT` |
| 33.926 | 4.52% | 200 | `nvjet_tst_128x256_64x6_2x1_2cta_v_bz_TNT` |
| 33.671 | 4.48% | 160 | `Torch-Compiled Region: 9/1` |
| 33.671 | 4.48% | 160 | `kernel_cutlass_kernel_flash_attncuteflash_fwd_sm100FlashAttentionForwardSm100_object_at__tensor0000o11101213_tensor0000o11101213_tensor0000o10111213_tensor0000o11101213_None_None_None_Non_0` |
| 31.484 | 4.19% | 430 | `aten::addmm` |
| 30.876 | 4.11% | 400 | `nvjet_tst_256x128_64x5_2x2_2cta_h_bz_bias_TNT` |
| 23.019 | 3.06% | 240 | `flash_attn::_flash_attn_forward` |
| 23.019 | 3.06% | 240 | `void flash::flash_fwd_kernel<Flash_fwd_kernel_traits<128, 128, 64, 4, false, false, cutlass::bfloat16_t, Flash_kernel_traits<128, 128, 64, 4, cutlass::bfloat16_t> >, false, false, false, false, false, true, false, false>(flash::Flash_fwd_params)` |
| 14.372 | 1.91% | 40 | `## Call CompiledFxGraph flwczbr2va6dg3efruela4opqksbqbcddxpjzr47fiv62qbso3w7 ##` |
| 10.611 | 1.41% | 40 | `## Call CompiledFxGraph fuerq27apmsb63irvfj2omfonzgokg7zwoewsmzlbgle7oevigjt ##` |
| 6.785 | 0.90% | 200 | `triton_poi_fused_addmm_gelu_view_3` |
| 6.785 | 0.90% | 200 | `triton_poi_fused_addmm_gelu_view_3` |
| 5.496 | 0.73% | 400 | `triton_red_fused__to_copy_add_addmm_mean_mul_pow_rsqrt_view_0` |
| 5.496 | 0.73% | 400 | `triton_red_fused__to_copy_add_addmm_mean_mul_pow_rsqrt_view_0` |
| 5.328 | 0.71% | 200 | `triton_red_fused_add_mul_native_layer_norm_view_0` |
| 5.328 | 0.71% | 200 | `triton_red_fused_add_mul_native_layer_norm_view_0` |
| 5.312 | 0.71% | 4,129 | `aten::copy_` |
| 5.200 | 0.69% | 200 | `triton_red_fused_add_addmm_mul_native_layer_norm_view_2` |
| 5.200 | 0.69% | 200 | `triton_red_fused_add_addmm_mul_native_layer_norm_view_2` |
| 5.131 | 0.68% | 400 | `rope_fused_3way_kernel_v2_0` |
| 5.131 | 0.68% | 400 | `rope_fused_3way_kernel_v2` |

## Stack filters
- stack_include: `['CausalWanSelfAttention']`
- stack_exclude: `[]`

## Stack groups: `aten::copy_`

Filtered totals: device_ms=2.518, calls=480

- count=8 device_ms=0.056 self_device_ms=0.056 cpu_ms=0.065
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_23`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_23`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=8 device_ms=0.056 self_device_ms=0.056 cpu_ms=0.076
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_18`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_18`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=8 device_ms=0.056 self_device_ms=0.056 cpu_ms=0.064
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_8`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_8`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=8 device_ms=0.056 self_device_ms=0.056 cpu_ms=0.067
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

- count=8 device_ms=0.056 self_device_ms=0.056 cpu_ms=0.064
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_12`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_12`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=8 device_ms=0.056 self_device_ms=0.056 cpu_ms=0.063
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_37`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_37`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=8 device_ms=0.055 self_device_ms=0.055 cpu_ms=0.063
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

- count=8 device_ms=0.055 self_device_ms=0.055 cpu_ms=0.064
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_28`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_28`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=8 device_ms=0.055 self_device_ms=0.055 cpu_ms=0.067
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

- count=8 device_ms=0.055 self_device_ms=0.055 cpu_ms=0.064
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_1`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_1`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=8 device_ms=0.055 self_device_ms=0.055 cpu_ms=0.064
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_38`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_38`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=8 device_ms=0.055 self_device_ms=0.055 cpu_ms=0.061
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_34`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_34`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=8 device_ms=0.055 self_device_ms=0.055 cpu_ms=0.068
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_3`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_3`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=8 device_ms=0.055 self_device_ms=0.055 cpu_ms=0.065
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_11`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_11`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=8 device_ms=0.055 self_device_ms=0.055 cpu_ms=0.064
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

- count=8 device_ms=0.055 self_device_ms=0.055 cpu_ms=0.074
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_15`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_15`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=8 device_ms=0.055 self_device_ms=0.055 cpu_ms=0.066
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_17`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_17`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=8 device_ms=0.055 self_device_ms=0.055 cpu_ms=0.065
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_5`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_5`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=8 device_ms=0.055 self_device_ms=0.055 cpu_ms=0.065
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_25`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_25`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=8 device_ms=0.055 self_device_ms=0.055 cpu_ms=0.063
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_24`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_24`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=6 device_ms=0.042 self_device_ms=0.042 cpu_ms=0.047
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_33`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_33`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=6 device_ms=0.042 self_device_ms=0.042 cpu_ms=0.060
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_16`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_16`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=6 device_ms=0.042 self_device_ms=0.042 cpu_ms=0.049
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

- count=6 device_ms=0.042 self_device_ms=0.042 cpu_ms=0.047
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_31`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_31`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=6 device_ms=0.042 self_device_ms=0.042 cpu_ms=0.049
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

## Stack groups: `aten::contiguous`

(no events)

## Stack groups: `aten::clone`

(no events)

## Stack groups: `aten::_to_copy`

(no events)

## Stack groups: `aten::fill_`

Filtered totals: device_ms=0.289, calls=160

- count=4 device_ms=0.008 self_device_ms=0.008 cpu_ms=0.036
  - `<built-in method fill_ of Tensor object at 0x7cb3ec3b90e0>`
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

- count=4 device_ms=0.008 self_device_ms=0.008 cpu_ms=0.036
  - `<built-in method fill_ of Tensor object at 0x7cb3ec3b90e0>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_18`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_18`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`

- count=4 device_ms=0.007 self_device_ms=0.007 cpu_ms=0.037
  - `<built-in method fill_ of Tensor object at 0x7cb3ec3b90e0>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_5`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_5`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`

- count=4 device_ms=0.007 self_device_ms=0.007 cpu_ms=0.036
  - `<built-in method fill_ of Tensor object at 0x7cb3ec3b90e0>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_3`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_3`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`

- count=4 device_ms=0.007 self_device_ms=0.007 cpu_ms=0.036
  - `<built-in method fill_ of Tensor object at 0x7cb3ec3b90e0>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_17`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_17`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`

- count=4 device_ms=0.007 self_device_ms=0.007 cpu_ms=0.037
  - `<built-in method fill_ of Tensor object at 0x7cb3ec3b90e0>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_38`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_38`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`

- count=4 device_ms=0.007 self_device_ms=0.007 cpu_ms=0.040
  - `<built-in method fill_ of Tensor object at 0x7cb3ec3b90e0>`
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

- count=4 device_ms=0.007 self_device_ms=0.007 cpu_ms=0.036
  - `<built-in method fill_ of Tensor object at 0x7cb3ec3b90e0>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_12`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_12`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`

- count=4 device_ms=0.007 self_device_ms=0.007 cpu_ms=0.036
  - `<built-in method fill_ of Tensor object at 0x7cb3ec3b90e0>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_15`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_15`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`

- count=4 device_ms=0.007 self_device_ms=0.007 cpu_ms=0.036
  - `<built-in method fill_ of Tensor object at 0x7cb3ec3b90e0>`
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

- count=4 device_ms=0.007 self_device_ms=0.007 cpu_ms=0.038
  - `<built-in method fill_ of Tensor object at 0x7cb3ec3b90e0>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_8`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_8`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`

- count=4 device_ms=0.007 self_device_ms=0.007 cpu_ms=0.038
  - `<built-in method fill_ of Tensor object at 0x7cb3ec3b90e0>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_1`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_1`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`

- count=4 device_ms=0.007 self_device_ms=0.007 cpu_ms=0.037
  - `<built-in method fill_ of Tensor object at 0x7cb3ec3b90e0>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_11`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_11`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`

- count=4 device_ms=0.007 self_device_ms=0.007 cpu_ms=0.039
  - `<built-in method fill_ of Tensor object at 0x7cb3ec3b90e0>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_37`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_37`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`

- count=4 device_ms=0.007 self_device_ms=0.007 cpu_ms=0.038
  - `<built-in method fill_ of Tensor object at 0x7cb3ec3b90e0>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_28`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_28`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`

- count=4 device_ms=0.007 self_device_ms=0.007 cpu_ms=0.048
  - `<built-in method fill_ of Tensor object at 0x7cb3ec3b90e0>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_23`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_23`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`

- count=4 device_ms=0.007 self_device_ms=0.007 cpu_ms=0.037
  - `<built-in method fill_ of Tensor object at 0x7cb3ec3b90e0>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_34`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_34`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`

- count=4 device_ms=0.007 self_device_ms=0.007 cpu_ms=0.037
  - `<built-in method fill_ of Tensor object at 0x7cb3ec3b90e0>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_24`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_24`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`

- count=4 device_ms=0.007 self_device_ms=0.007 cpu_ms=0.037
  - `<built-in method fill_ of Tensor object at 0x7cb3ec3b90e0>`
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

- count=4 device_ms=0.007 self_device_ms=0.007 cpu_ms=0.037
  - `<built-in method fill_ of Tensor object at 0x7cb3ec3b90e0>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_25`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_25`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`

- count=3 device_ms=0.006 self_device_ms=0.006 cpu_ms=0.029
  - `<built-in method fill_ of Tensor object at 0x7cb3ec3b90e0>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_2`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_2`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`

- count=3 device_ms=0.006 self_device_ms=0.006 cpu_ms=0.029
  - `<built-in method fill_ of Tensor object at 0x7cb3ec3b90e0>`
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

- count=3 device_ms=0.006 self_device_ms=0.006 cpu_ms=0.028
  - `<built-in method fill_ of Tensor object at 0x7cb3ec3b90e0>`
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

- count=3 device_ms=0.005 self_device_ms=0.005 cpu_ms=0.028
  - `<built-in method fill_ of Tensor object at 0x7cb3ec3b90e0>`
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

- count=3 device_ms=0.005 self_device_ms=0.005 cpu_ms=0.029
  - `<built-in method fill_ of Tensor object at 0x7cb3ec3b90e0>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_31`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_31`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`

## Stack groups: `aten::transpose`

Filtered totals: device_ms=0.000, calls=160

- count=4 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.008
  - `<built-in function linear>`
  - `torch/nn/modules/linear.py(130): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Linear_6`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_0`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`

- count=4 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.008
  - `<built-in function linear>`
  - `torch/nn/modules/linear.py(130): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Linear_7`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_1`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`

- count=3 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.006
  - `<built-in function linear>`
  - `torch/nn/modules/linear.py(130): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Linear_8`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_2`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`

- count=4 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.008
  - `<built-in function linear>`
  - `torch/nn/modules/linear.py(130): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Linear_9`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_3`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`

- count=4 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.008
  - `<built-in function linear>`
  - `torch/nn/modules/linear.py(130): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Linear_10`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_4`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`

- count=4 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.008
  - `<built-in function linear>`
  - `torch/nn/modules/linear.py(130): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Linear_11`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_5`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`

- count=3 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.011
  - `<built-in function linear>`
  - `torch/nn/modules/linear.py(130): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Linear_12`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_6`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`

- count=3 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.006
  - `<built-in function linear>`
  - `torch/nn/modules/linear.py(130): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Linear_13`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_7`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`

- count=4 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.008
  - `<built-in function linear>`
  - `torch/nn/modules/linear.py(130): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Linear_14`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_8`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`

- count=3 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.006
  - `<built-in function linear>`
  - `torch/nn/modules/linear.py(130): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Linear_15`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_9`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`

- count=3 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.006
  - `<built-in function linear>`
  - `torch/nn/modules/linear.py(130): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Linear_16`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_10`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`

- count=4 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.008
  - `<built-in function linear>`
  - `torch/nn/modules/linear.py(130): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Linear_17`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_11`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`

- count=4 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.008
  - `<built-in function linear>`
  - `torch/nn/modules/linear.py(130): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Linear_18`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_12`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`

- count=2 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.004
  - `<built-in function linear>`
  - `torch/nn/modules/linear.py(130): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Linear_19`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_13`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`

- count=2 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.004
  - `<built-in function linear>`
  - `torch/nn/modules/linear.py(130): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Linear_20`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_14`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`

- count=4 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.008
  - `<built-in function linear>`
  - `torch/nn/modules/linear.py(130): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Linear_21`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_15`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`

- count=3 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.006
  - `<built-in function linear>`
  - `torch/nn/modules/linear.py(130): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Linear_22`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_16`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`

- count=4 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.008
  - `<built-in function linear>`
  - `torch/nn/modules/linear.py(130): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Linear_23`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_17`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`

- count=4 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.008
  - `<built-in function linear>`
  - `torch/nn/modules/linear.py(130): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Linear_24`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_18`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`

- count=3 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.006
  - `<built-in function linear>`
  - `torch/nn/modules/linear.py(130): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Linear_25`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_19`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`

- count=4 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.008
  - `<built-in function linear>`
  - `torch/nn/modules/linear.py(130): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Linear_26`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_20`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.002
  - `<built-in function linear>`
  - `torch/nn/modules/linear.py(130): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Linear_27`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_21`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_21`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`

- count=3 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.006
  - `<built-in function linear>`
  - `torch/nn/modules/linear.py(130): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Linear_28`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_22`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`

- count=4 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.008
  - `<built-in function linear>`
  - `torch/nn/modules/linear.py(130): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Linear_29`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_23`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`

- count=4 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.008
  - `<built-in function linear>`
  - `torch/nn/modules/linear.py(130): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Linear_30`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_24`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
