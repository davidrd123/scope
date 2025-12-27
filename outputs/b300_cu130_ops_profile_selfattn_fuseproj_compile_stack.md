# Krea pipeline op profile

## Meta
- torch: `2.9.0+cu130` (cuda `13.0`)
- device: `NVIDIA B300 SXM6 AC` cc=(10, 3)
- compile: `True`
- kv_cache_attention_bias: `0.3`
- SCOPE_KV_BIAS_BACKEND: `fa4`
- profiled_wall_time_s: `7.911`

## Top ops (self CUDA)
| self_cuda_ms | pct | calls | key |
|---:|---:|---:|---|
| 137.274 | 18.71% | 200 | `## Call CompiledFxGraph f3xe5wyeoz3mjoa4uwronzbu27nnhqzgktf5hbdcuquz6tflrwl5 ##` |
| 99.094 | 13.51% | 800 | `aten::mm` |
| 54.328 | 7.40% | 430 | `aten::addmm` |
| 38.151 | 5.20% | 200 | `nvjet_tst_256x128_64x5_2x1_2cta_v_bz_bias_TNT` |
| 34.913 | 4.76% | 160 | `## Call CompiledFxGraph fufpjilrjyyvnwvvsz62i635xkhwfgfzhugu7zsfwkdgdsgpaaq6 ##` |
| 34.411 | 4.69% | 200 | `nvjet_tst_256x160_64x5_2x1_2cta_v_bz_TNT` |
| 34.104 | 4.65% | 200 | `nvjet_tst_128x256_64x6_2x1_2cta_v_bz_TNT` |
| 33.754 | 4.60% | 160 | `Torch-Compiled Region: 9/1` |
| 33.754 | 4.60% | 160 | `kernel_cutlass_kernel_flash_attncuteflash_fwd_sm100FlashAttentionForwardSm100_object_at__tensor0000o11101213_tensor0000o11101213_tensor0000o10111213_tensor0000o11101213_None_None_None_Non_0` |
| 30.579 | 4.17% | 400 | `nvjet_tst_256x128_64x5_2x2_2cta_h_bz_TNT` |
| 23.258 | 3.17% | 240 | `flash_attn::_flash_attn_forward` |
| 23.258 | 3.17% | 240 | `void flash::flash_fwd_kernel<Flash_fwd_kernel_traits<128, 128, 64, 4, false, false, cutlass::bfloat16_t, Flash_kernel_traits<128, 128, 64, 4, cutlass::bfloat16_t> >, false, false, false, false, false, true, false, false>(flash::Flash_fwd_params)` |
| 15.569 | 2.12% | 200 | `nvjet_tst_256x128_64x5_2x2_2cta_h_bz_bias_TNT` |
| 14.624 | 1.99% | 40 | `## Call CompiledFxGraph fuva4gvw3zh2y5sspubmmnz5wmo27xaijdgipbcoydxrkst26dcj ##` |
| 8.833 | 1.20% | 40 | `## Call CompiledFxGraph fyx4v4qcffsdemz4232xhgk4syoymveqhhtmdh6kr73qiuazm4yg ##` |
| 8.770 | 1.20% | 10,062 | `aten::copy_` |
| 7.676 | 1.05% | 1,786 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1})` |
| 6.801 | 0.93% | 200 | `triton_poi_fused_addmm_gelu_view_3` |
| 6.801 | 0.93% | 200 | `triton_poi_fused_addmm_gelu_view_3` |
| 5.296 | 0.72% | 200 | `triton_red_fused_add_mul_native_layer_norm_view_0` |
| 5.296 | 0.72% | 200 | `triton_red_fused_add_mul_native_layer_norm_view_0` |
| 5.242 | 0.71% | 200 | `triton_red_fused_add_addmm_mul_native_layer_norm_view_2` |
| 5.242 | 0.71% | 200 | `triton_red_fused_add_addmm_mul_native_layer_norm_view_2` |
| 5.071 | 0.69% | 200 | `## Call CompiledFxGraph fykf4eh4paj6eytlsiqjalfu6saslprgfv7azwixjwgnn36txthc ##` |
| 5.005 | 0.68% | 400 | `rope_fused_3way_kernel_v2_0` |

## Stack filters
- stack_include: `['CausalWanSelfAttention']`
- stack_exclude: `[]`

## Stack groups: `aten::contiguous`

(no events)

## Stack groups: `aten::copy_`

Filtered totals: device_ms=5.623, calls=480

- count=8 device_ms=0.135 self_device_ms=0.135 cpu_ms=0.073
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_6`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `torch/_dynamo/eval_frame.py(791): compile_wrapper`
  - `nn.Module: CausalWanAttentionBlock_6`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2286): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2623): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=8 device_ms=0.134 self_device_ms=0.134 cpu_ms=0.072
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

- count=8 device_ms=0.134 self_device_ms=0.134 cpu_ms=0.071
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

- count=8 device_ms=0.134 self_device_ms=0.134 cpu_ms=0.073
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

- count=8 device_ms=0.133 self_device_ms=0.133 cpu_ms=0.073
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

- count=8 device_ms=0.133 self_device_ms=0.133 cpu_ms=0.074
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

- count=8 device_ms=0.133 self_device_ms=0.133 cpu_ms=0.074
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

- count=8 device_ms=0.133 self_device_ms=0.133 cpu_ms=0.073
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

- count=8 device_ms=0.133 self_device_ms=0.133 cpu_ms=0.074
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

- count=8 device_ms=0.133 self_device_ms=0.133 cpu_ms=0.073
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

- count=8 device_ms=0.133 self_device_ms=0.133 cpu_ms=0.073
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

- count=8 device_ms=0.133 self_device_ms=0.133 cpu_ms=0.074
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

- count=8 device_ms=0.133 self_device_ms=0.133 cpu_ms=0.073
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
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=8 device_ms=0.133 self_device_ms=0.133 cpu_ms=0.073
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

- count=8 device_ms=0.133 self_device_ms=0.133 cpu_ms=0.074
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

## Stack groups: `aten::_to_copy`

(no events)
