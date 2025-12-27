# Krea pipeline op profile

## Meta
- torch: `2.9.0+cu130` (cuda `13.0`)
- device: `NVIDIA B300 SXM6 AC` cc=(10, 3)
- compile: `True`
- kv_cache_attention_bias: `0.3`
- SCOPE_KV_BIAS_BACKEND: `fa4`
- profiled_wall_time_s: `12.643`

## Top ops (self CUDA)
| self_cuda_ms | pct | calls | key |
|---:|---:|---:|---|
| 84.591 | 12.30% | 600 | `aten::mm` |
| 84.491 | 12.29% | 200 | `## Call CompiledFxGraph fzkr5p2ai4jlimh5nkeoy4kj6gag3djpy7rrfhfz6gvfo74pftp2 ##` |
| 69.790 | 10.15% | 630 | `aten::addmm` |
| 38.441 | 5.59% | 200 | `nvjet_tst_256x128_64x5_2x1_2cta_v_bz_bias_TNT` |
| 35.247 | 5.13% | 160 | `## Call CompiledFxGraph fe3p354nzwxb6oufl3yqqnv5n7oj6ozrogyabdrellckb4pv7r4i ##` |
| 34.671 | 5.04% | 200 | `nvjet_tst_256x160_64x5_2x1_2cta_v_bz_TNT` |
| 34.114 | 4.96% | 200 | `nvjet_tst_128x256_64x6_2x1_2cta_v_bz_TNT` |
| 33.518 | 4.88% | 160 | `Torch-Compiled Region: 9/1` |
| 33.518 | 4.88% | 160 | `kernel_cutlass_kernel_flash_attncuteflash_fwd_sm100FlashAttentionForwardSm100_object_at__tensor0000o11101213_tensor0000o11101213_tensor0000o10111213_tensor0000o11101213_None_None_None_Non_0` |
| 30.740 | 4.47% | 400 | `nvjet_tst_256x128_64x5_2x2_2cta_h_bz_bias_TNT` |
| 18.884 | 2.75% | 200 | `## Call CompiledFxGraph ffw4q4kngccow57pa2mkgee77ogpi55si5mvf26jkdld7scmyvs7 ##` |
| 18.377 | 2.67% | 240 | `## Call CompiledFxGraph fcceaggubatdtr6yp77cg7bal2pxb7hmqu76agprelqh5hx5asm4 ##` |
| 15.806 | 2.30% | 200 | `nvjet_tst_256x128_64x5_2x2_2cta_h_bz_TNT` |
| 10.836 | 1.58% | 4,049 | `aten::copy_` |
| 10.747 | 1.56% | 240 | `FlashAttnVarlenFunc` |
| 10.747 | 1.56% | 240 | `kernel_cutlass_kernel_flash_attncuteflash_fwd_sm100FlashAttentionForwardSm100_object_at__tensor000o111012_tensor000o111012_tensor000o101112_tensorptrbf16gmemalign16odiv81div8_None_tensorp_0` |
| 10.599 | 1.54% | 400 | `triton_red_fused_add_mul_native_layer_norm_view_0` |
| 10.599 | 1.54% | 400 | `triton_red_fused_add_mul_native_layer_norm_view_0` |
| 10.160 | 1.48% | 3,308 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1})` |
| 8.815 | 1.28% | 40 | `## Call CompiledFxGraph fk64ajjwdf2q6mjilnqgjewiklnrsdt2pevlvsuyisl466jtpaa3 ##` |
| 6.796 | 0.99% | 200 | `triton_poi_fused_addmm_gelu_view_1` |
| 6.796 | 0.99% | 200 | `triton_poi_fused_addmm_gelu_view_1` |
| 5.491 | 0.80% | 200 | `## Call CompiledFxGraph f732rfwuj3e43dof7xqnzwq6bnjpadzz7val5ryx64zqiuudjyof ##` |
| 5.096 | 0.74% | 200 | `## Call CompiledFxGraph fuh67mwwyfifvu65poxmdnfvc3zukk5l7ofvfw7wqcnloi5p64op ##` |
| 5.032 | 0.73% | 400 | `rope_fused_3way_kernel_v2_0` |

## Stack filters
- stack_include: `['CausalWanSelfAttention']`
- stack_exclude: `[]`

## Top ops (stack-filtered)
Filtered to stack frames matching include=['CausalWanSelfAttention'] exclude=[].

| device_ms | pct | calls | groups | key |
|---:|---:|---:|---:|---|
| 89.936 | 31.25% | 160 | 40 | `Torch-Compiled Region: 1/2` |
| 53.157 | 18.47% | 160 | 40 | `Torch-Compiled Region: 8/1` |
| 51.186 | 17.78% | 160 | 40 | `Torch-Compiled Region: 9/1` |
| 33.518 | 11.65% | 160 | 40 | `kernel_cutlass_kernel_flash_attncuteflash_fwd_sm100FlashAttentionForwardSm100_object_at__tensor0000o11101213_tensor0000o11101213_tensor0000o10111213_tensor0000o11101213_None_None_None_Non_0` |
| 17.430 | 6.06% | 40 | 40 | `Torch-Compiled Region: 1/1` |
| 8.233 | 2.86% | 40 | 40 | `Torch-Compiled Region: 21/0` |
| 7.737 | 2.69% | 40 | 40 | `Torch-Compiled Region: 22/0` |
| 5.304 | 1.84% | 320 | 44 | `aten::copy_` |
| 5.032 | 1.75% | 400 | 159 | `Torch-Compiled Region: 3/0` |
| 4.418 | 1.53% | 160 | 44 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1})` |
| 3.967 | 1.38% | 40 | 40 | `Torch-Compiled Region: 23/0` |
| 3.119 | 1.08% | 40 | 40 | `Torch-Compiled Region: 24/0` |
| 2.577 | 0.90% | 201 | 81 | `Torch-Compiled Region: 4/1` |
| 0.886 | 0.31% | 160 | 44 | `Memcpy DtoD (Device -> Device)` |
| 0.313 | 0.11% | 25 | 24 | `## Call CompiledFxGraph f4oz2wvcxejxwpdxprvuut3twkm7x5q4nrlvgkq2bxu63do7rkav ##` |
| 0.299 | 0.10% | 3 | 3 | `Torch-Compiled Region: 15/2` |
| 0.227 | 0.08% | 1 | 1 | `## Call CompiledFxGraph fk64ajjwdf2q6mjilnqgjewiklnrsdt2pevlvsuyisl466jtpaa3 ##` |
| 0.219 | 0.08% | 1 | 1 | `## Call CompiledFxGraph fe3p354nzwxb6oufl3yqqnv5n7oj6ozrogyabdrellckb4pv7r4i ##` |
| 0.201 | 0.07% | 8 | 8 | `## Call CompiledFxGraph fuh67mwwyfifvu65poxmdnfvc3zukk5l7ofvfw7wqcnloi5p64op ##` |
| 0.065 | 0.02% | 5 | 5 | `Torch-Compiled Region: 7/1` |

## Stack groups: `aten::copy_`

Filtered totals: device_ms=5.304, calls=324

- count=8 device_ms=0.133 self_device_ms=0.133 cpu_ms=0.093
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_36`

- count=8 device_ms=0.133 self_device_ms=0.133 cpu_ms=0.093
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_31`

- count=8 device_ms=0.133 self_device_ms=0.133 cpu_ms=0.095
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_18`

- count=8 device_ms=0.133 self_device_ms=0.133 cpu_ms=0.094
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_23`

- count=8 device_ms=0.133 self_device_ms=0.133 cpu_ms=0.095
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_11`

- count=8 device_ms=0.133 self_device_ms=0.133 cpu_ms=0.095
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_38`

- count=8 device_ms=0.133 self_device_ms=0.133 cpu_ms=0.101
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_19`

- count=8 device_ms=0.133 self_device_ms=0.133 cpu_ms=0.093
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_12`

- count=8 device_ms=0.133 self_device_ms=0.133 cpu_ms=0.104
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_6`

- count=8 device_ms=0.133 self_device_ms=0.133 cpu_ms=0.108
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_33`

- count=8 device_ms=0.133 self_device_ms=0.133 cpu_ms=0.094
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_1`

- count=8 device_ms=0.133 self_device_ms=0.133 cpu_ms=0.096
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_32`

## Stack groups: `aten::_to_copy`

(no events)

## Stack groups: `aten::to`

(no events)

## Stack groups: `aten::fill_`

Filtered totals: device_ms=0.000, calls=4

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.001
  - `<built-in method fill_ of Tensor object at 0x7a4749913570>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_21`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.001
  - `<built-in method fill_ of Tensor object at 0x7a4749913570>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_10`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.001
  - `<built-in method fill_ of Tensor object at 0x7a4749913570>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_13`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.001
  - `<built-in method fill_ of Tensor object at 0x7a4749913570>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_28`
