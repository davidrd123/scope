# Krea pipeline op profile

## Meta
- torch: `2.9.0+cu130` (cuda `13.0`)
- device: `NVIDIA B300 SXM6 AC` cc=(10, 3)
- compile: `True`
- kv_cache_attention_bias: `0.3`
- SCOPE_KV_BIAS_BACKEND: `fa4`
- profiled_wall_time_s: `7.873`

## Top ops (self CUDA)
| self_cuda_ms | pct | calls | key |
|---:|---:|---:|---|
| 136.806 | 18.26% | 200 | `## Call CompiledFxGraph f3xe5wyeoz3mjoa4uwronzbu27nnhqzgktf5hbdcuquz6tflrwl5 ##` |
| 128.840 | 17.20% | 1,200 | `aten::mm` |
| 60.849 | 8.12% | 800 | `nvjet_tst_256x128_64x5_2x2_2cta_h_bz_TNT` |
| 42.494 | 5.67% | 160 | `## Call CompiledFxGraph fed4ov6qe57tzjn5yggz6dav3tn4vnbpeuuzyim6fjrswsfn4qe6 ##` |
| 34.112 | 4.55% | 200 | `nvjet_tst_256x160_64x5_2x1_2cta_v_bz_TNT` |
| 33.878 | 4.52% | 200 | `nvjet_tst_128x256_64x6_2x1_2cta_v_bz_TNT` |
| 33.586 | 4.48% | 160 | `Torch-Compiled Region: 9/1` |
| 33.586 | 4.48% | 160 | `kernel_cutlass_kernel_flash_attncuteflash_fwd_sm100FlashAttentionForwardSm100_object_at__tensor0000o11101213_tensor0000o11101213_tensor0000o10111213_tensor0000o11101213_None_None_None_Non_0` |
| 31.412 | 4.19% | 430 | `aten::addmm` |
| 30.800 | 4.11% | 400 | `nvjet_tst_256x128_64x5_2x2_2cta_h_bz_bias_TNT` |
| 22.953 | 3.06% | 240 | `flash_attn::_flash_attn_forward` |
| 22.953 | 3.06% | 240 | `void flash::flash_fwd_kernel<Flash_fwd_kernel_traits<128, 128, 64, 4, false, false, cutlass::bfloat16_t, Flash_kernel_traits<128, 128, 64, 4, cutlass::bfloat16_t> >, false, false, false, false, false, true, false, false>(flash::Flash_fwd_params)` |
| 14.317 | 1.91% | 40 | `## Call CompiledFxGraph flwczbr2va6dg3efruela4opqksbqbcddxpjzr47fiv62qbso3w7 ##` |
| 10.596 | 1.41% | 40 | `## Call CompiledFxGraph fuerq27apmsb63irvfj2omfonzgokg7zwoewsmzlbgle7oevigjt ##` |
| 6.776 | 0.90% | 200 | `triton_poi_fused_addmm_gelu_view_3` |
| 6.776 | 0.90% | 200 | `triton_poi_fused_addmm_gelu_view_3` |
| 5.476 | 0.73% | 400 | `triton_red_fused__to_copy_add_addmm_mean_mul_pow_rsqrt_view_0` |
| 5.476 | 0.73% | 400 | `triton_red_fused__to_copy_add_addmm_mean_mul_pow_rsqrt_view_0` |
| 5.386 | 0.72% | 10,062 | `aten::copy_` |
| 5.317 | 0.71% | 200 | `triton_red_fused_add_mul_native_layer_norm_view_0` |
| 5.317 | 0.71% | 200 | `triton_red_fused_add_mul_native_layer_norm_view_0` |
| 5.198 | 0.69% | 200 | `triton_red_fused_add_addmm_mul_native_layer_norm_view_2` |
| 5.198 | 0.69% | 200 | `triton_red_fused_add_addmm_mul_native_layer_norm_view_2` |
| 5.138 | 0.69% | 400 | `rope_fused_3way_kernel_v2_0` |
| 5.138 | 0.69% | 400 | `rope_fused_3way_kernel_v2` |

## Stack filters
- stack_include: `['CausalWanSelfAttention']`
- stack_exclude: `[]`

## Stack groups: `aten::copy_`

Filtered totals: device_ms=2.507, calls=477

- count=8 device_ms=0.056 self_device_ms=0.056 cpu_ms=0.075
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

- count=8 device_ms=0.056 self_device_ms=0.056 cpu_ms=0.075
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

- count=8 device_ms=0.056 self_device_ms=0.056 cpu_ms=0.065
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

- count=8 device_ms=0.056 self_device_ms=0.056 cpu_ms=0.071
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

- count=8 device_ms=0.056 self_device_ms=0.056 cpu_ms=0.075
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

- count=8 device_ms=0.056 self_device_ms=0.056 cpu_ms=0.067
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

- count=8 device_ms=0.056 self_device_ms=0.056 cpu_ms=0.074
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_26`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_26`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=8 device_ms=0.056 self_device_ms=0.056 cpu_ms=0.092
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_14`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_14`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=8 device_ms=0.056 self_device_ms=0.056 cpu_ms=0.071
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

- count=8 device_ms=0.055 self_device_ms=0.055 cpu_ms=0.070
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

- count=8 device_ms=0.055 self_device_ms=0.055 cpu_ms=0.071
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

- count=8 device_ms=0.055 self_device_ms=0.055 cpu_ms=0.072
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_36`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_36`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=8 device_ms=0.055 self_device_ms=0.055 cpu_ms=0.066
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

- count=8 device_ms=0.055 self_device_ms=0.055 cpu_ms=0.069
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

- count=8 device_ms=0.055 self_device_ms=0.055 cpu_ms=0.075
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

## Stack groups: `aten::contiguous`

(no events)

## Stack groups: `aten::_to_copy`

(no events)
