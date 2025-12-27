# Krea pipeline op profile

## Meta
- torch: `2.9.0+cu130` (cuda `13.0`)
- device: `NVIDIA B300 SXM6 AC` cc=(10, 3)
- compile: `True`
- kv_cache_attention_bias: `0.3`
- SCOPE_KV_BIAS_BACKEND: `fa4`
- profiled_wall_time_s: `12.767`

## Top ops (self CUDA)
| self_cuda_ms | pct | calls | key |
|---:|---:|---:|---|
| 84.634 | 12.31% | 600 | `aten::mm` |
| 84.486 | 12.29% | 200 | `## Call CompiledFxGraph fzkr5p2ai4jlimh5nkeoy4kj6gag3djpy7rrfhfz6gvfo74pftp2 ##` |
| 69.797 | 10.15% | 630 | `aten::addmm` |
| 38.432 | 5.59% | 200 | `nvjet_tst_256x128_64x5_2x1_2cta_v_bz_bias_TNT` |
| 35.236 | 5.12% | 160 | `## Call CompiledFxGraph fe3p354nzwxb6oufl3yqqnv5n7oj6ozrogyabdrellckb4pv7r4i ##` |
| 34.693 | 5.04% | 200 | `nvjet_tst_256x160_64x5_2x1_2cta_v_bz_TNT` |
| 34.124 | 4.96% | 200 | `nvjet_tst_128x256_64x6_2x1_2cta_v_bz_TNT` |
| 33.508 | 4.87% | 160 | `Torch-Compiled Region: 9/1` |
| 33.508 | 4.87% | 160 | `kernel_cutlass_kernel_flash_attncuteflash_fwd_sm100FlashAttentionForwardSm100_object_at__tensor0000o11101213_tensor0000o11101213_tensor0000o10111213_tensor0000o11101213_None_None_None_Non_0` |
| 30.754 | 4.47% | 400 | `nvjet_tst_256x128_64x5_2x2_2cta_h_bz_bias_TNT` |
| 18.904 | 2.75% | 200 | `## Call CompiledFxGraph ffw4q4kngccow57pa2mkgee77ogpi55si5mvf26jkdld7scmyvs7 ##` |
| 18.372 | 2.67% | 240 | `## Call CompiledFxGraph fcceaggubatdtr6yp77cg7bal2pxb7hmqu76agprelqh5hx5asm4 ##` |
| 15.817 | 2.30% | 200 | `nvjet_tst_256x128_64x5_2x2_2cta_h_bz_TNT` |
| 10.841 | 1.58% | 4,049 | `aten::copy_` |
| 10.724 | 1.56% | 240 | `FlashAttnVarlenFunc` |
| 10.724 | 1.56% | 240 | `kernel_cutlass_kernel_flash_attncuteflash_fwd_sm100FlashAttentionForwardSm100_object_at__tensor000o111012_tensor000o111012_tensor000o101112_tensorptrbf16gmemalign16odiv81div8_None_tensorp_0` |
| 10.596 | 1.54% | 400 | `triton_red_fused_add_mul_native_layer_norm_view_0` |
| 10.596 | 1.54% | 400 | `triton_red_fused_add_mul_native_layer_norm_view_0` |
| 10.165 | 1.48% | 3,308 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1})` |
| 8.826 | 1.28% | 40 | `## Call CompiledFxGraph fk64ajjwdf2q6mjilnqgjewiklnrsdt2pevlvsuyisl466jtpaa3 ##` |
| 6.801 | 0.99% | 200 | `triton_poi_fused_addmm_gelu_view_1` |
| 6.801 | 0.99% | 200 | `triton_poi_fused_addmm_gelu_view_1` |
| 5.499 | 0.80% | 200 | `## Call CompiledFxGraph f732rfwuj3e43dof7xqnzwq6bnjpadzz7val5ryx64zqiuudjyof ##` |
| 5.105 | 0.74% | 200 | `## Call CompiledFxGraph fuh67mwwyfifvu65poxmdnfvc3zukk5l7ofvfw7wqcnloi5p64op ##` |
| 5.014 | 0.73% | 400 | `rope_fused_3way_kernel_v2_0` |

## Stack filters
- stack_include: `['CausalWanSelfAttention']`
- stack_exclude: `[]`

## Top ops (stack-filtered)
Filtered to stack frames matching include=['CausalWanSelfAttention'] exclude=[].

| device_ms | pct | calls | groups | key |
|---:|---:|---:|---:|---|
| 89.918 | 31.12% | 160 | 40 | `Torch-Compiled Region: 1/2` |
| 53.169 | 18.40% | 160 | 40 | `Torch-Compiled Region: 8/1` |
| 51.199 | 17.72% | 160 | 40 | `Torch-Compiled Region: 9/1` |
| 33.508 | 11.60% | 160 | 40 | `kernel_cutlass_kernel_flash_attncuteflash_fwd_sm100FlashAttentionForwardSm100_object_at__tensor0000o11101213_tensor0000o11101213_tensor0000o10111213_tensor0000o11101213_None_None_None_Non_0` |
| 17.446 | 6.04% | 40 | 40 | `Torch-Compiled Region: 1/1` |
| 8.241 | 2.85% | 40 | 40 | `Torch-Compiled Region: 21/0` |
| 7.745 | 2.68% | 40 | 40 | `Torch-Compiled Region: 22/0` |
| 5.310 | 1.84% | 320 | 42 | `aten::copy_` |
| 5.014 | 1.74% | 400 | 160 | `Torch-Compiled Region: 3/0` |
| 4.426 | 1.53% | 160 | 42 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1})` |
| 3.956 | 1.37% | 40 | 40 | `Torch-Compiled Region: 23/0` |
| 3.135 | 1.08% | 40 | 40 | `Torch-Compiled Region: 24/0` |
| 2.548 | 0.88% | 200 | 80 | `Torch-Compiled Region: 4/1` |
| 1.098 | 0.38% | 5 | 4 | `## Call CompiledFxGraph fe3p354nzwxb6oufl3yqqnv5n7oj6ozrogyabdrellckb4pv7r4i ##` |
| 0.884 | 0.31% | 160 | 42 | `Memcpy DtoD (Device -> Device)` |
| 0.668 | 0.23% | 3 | 3 | `## Call CompiledFxGraph fk64ajjwdf2q6mjilnqgjewiklnrsdt2pevlvsuyisl466jtpaa3 ##` |
| 0.503 | 0.17% | 20 | 17 | `## Call CompiledFxGraph fuh67mwwyfifvu65poxmdnfvc3zukk5l7ofvfw7wqcnloi5p64op ##` |
| 0.128 | 0.04% | 10 | 9 | `Torch-Compiled Region: 7/1` |
| 0.061 | 0.02% | 5 | 5 | `## Call CompiledFxGraph f4oz2wvcxejxwpdxprvuut3twkm7x5q4nrlvgkq2bxu63do7rkav ##` |

## Stack groups: `aten::copy_`

Filtered totals: device_ms=5.310, calls=322

- count=8 device_ms=0.134 self_device_ms=0.134 cpu_ms=0.093
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_24`

- count=8 device_ms=0.134 self_device_ms=0.134 cpu_ms=0.094
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_39`

- count=8 device_ms=0.134 self_device_ms=0.134 cpu_ms=0.092
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_36`

- count=8 device_ms=0.133 self_device_ms=0.133 cpu_ms=0.104
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_2`

- count=8 device_ms=0.133 self_device_ms=0.133 cpu_ms=0.094
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_9`

- count=8 device_ms=0.133 self_device_ms=0.133 cpu_ms=0.093
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_34`

- count=8 device_ms=0.133 self_device_ms=0.133 cpu_ms=0.094
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_14`

- count=8 device_ms=0.133 self_device_ms=0.133 cpu_ms=0.094
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_22`

- count=8 device_ms=0.133 self_device_ms=0.133 cpu_ms=0.094
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_21`

- count=8 device_ms=0.133 self_device_ms=0.133 cpu_ms=0.094
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_32`

- count=8 device_ms=0.133 self_device_ms=0.133 cpu_ms=0.092
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_31`

- count=8 device_ms=0.133 self_device_ms=0.133 cpu_ms=0.116
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_8`

- count=8 device_ms=0.133 self_device_ms=0.133 cpu_ms=0.115
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_3`

- count=8 device_ms=0.133 self_device_ms=0.133 cpu_ms=0.105
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_20`

- count=8 device_ms=0.133 self_device_ms=0.133 cpu_ms=0.093
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_37`

## Stack groups: `aten::_to_copy`

(no events)

## Stack groups: `aten::to`

(no events)

## Stack groups: `aten::fill_`

Filtered totals: device_ms=0.000, calls=2

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.001
  - `<built-in method fill_ of Tensor object at 0x76e2a8391680>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_18`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.001
  - `<built-in method fill_ of Tensor object at 0x76e2a8391680>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_28`
