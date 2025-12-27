# Krea pipeline op profile

## Meta
- torch: `2.9.0+cu130` (cuda `13.0`)
- device: `NVIDIA B300 SXM6 AC` cc=(10, 3)
- compile: `True`
- kv_cache_attention_bias: `0.3`
- SCOPE_KV_BIAS_BACKEND: `fa4`
- profiled_wall_time_s: `12.145`

## Top ops (self CUDA)
| self_cuda_ms | pct | calls | key |
|---:|---:|---:|---|
| 84.542 | 12.34% | 600 | `aten::mm` |
| 84.508 | 12.33% | 200 | `## Call CompiledFxGraph fzkr5p2ai4jlimh5nkeoy4kj6gag3djpy7rrfhfz6gvfo74pftp2 ##` |
| 69.780 | 10.18% | 630 | `aten::addmm` |
| 38.404 | 5.60% | 200 | `nvjet_tst_256x128_64x5_2x1_2cta_v_bz_bias_TNT` |
| 35.249 | 5.14% | 160 | `## Call CompiledFxGraph fe3p354nzwxb6oufl3yqqnv5n7oj6ozrogyabdrellckb4pv7r4i ##` |
| 34.678 | 5.06% | 200 | `nvjet_tst_256x160_64x5_2x1_2cta_v_bz_TNT` |
| 34.066 | 4.97% | 200 | `nvjet_tst_128x256_64x6_2x1_2cta_v_bz_TNT` |
| 33.504 | 4.89% | 160 | `Torch-Compiled Region: 8/1` |
| 33.504 | 4.89% | 160 | `kernel_cutlass_kernel_flash_attncuteflash_fwd_sm100FlashAttentionForwardSm100_object_at__tensor0000o11101213_tensor0000o11101213_tensor0000o10111213_tensor0000o11101213_None_None_None_Non_0` |
| 30.772 | 4.49% | 400 | `nvjet_tst_256x128_64x5_2x2_2cta_h_bz_bias_TNT` |
| 18.822 | 2.75% | 200 | `## Call CompiledFxGraph ffw4q4kngccow57pa2mkgee77ogpi55si5mvf26jkdld7scmyvs7 ##` |
| 18.381 | 2.68% | 240 | `## Call CompiledFxGraph fcceaggubatdtr6yp77cg7bal2pxb7hmqu76agprelqh5hx5asm4 ##` |
| 15.798 | 2.31% | 200 | `nvjet_tst_256x128_64x5_2x2_2cta_h_bz_TNT` |
| 10.617 | 1.55% | 240 | `FlashAttnVarlenFunc` |
| 10.617 | 1.55% | 240 | `kernel_cutlass_kernel_flash_attncuteflash_fwd_sm100FlashAttentionForwardSm100_object_at__tensor000o111012_tensor000o111012_tensor000o101112_tensorptrbf16gmemalign16odiv81div8_None_tensorp_0` |
| 10.608 | 1.55% | 400 | `triton_red_fused_add_mul_native_layer_norm_view_0` |
| 10.608 | 1.55% | 400 | `triton_red_fused_add_mul_native_layer_norm_view_0` |
| 10.282 | 1.50% | 3,384 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1})` |
| 10.072 | 1.47% | 3,889 | `aten::copy_` |
| 8.798 | 1.28% | 40 | `## Call CompiledFxGraph fk64ajjwdf2q6mjilnqgjewiklnrsdt2pevlvsuyisl466jtpaa3 ##` |
| 6.796 | 0.99% | 200 | `triton_poi_fused_addmm_gelu_view_1` |
| 6.796 | 0.99% | 200 | `triton_poi_fused_addmm_gelu_view_1` |
| 5.498 | 0.80% | 200 | `## Call CompiledFxGraph f732rfwuj3e43dof7xqnzwq6bnjpadzz7val5ryx64zqiuudjyof ##` |
| 5.096 | 0.74% | 200 | `## Call CompiledFxGraph fuh67mwwyfifvu65poxmdnfvc3zukk5l7ofvfw7wqcnloi5p64op ##` |
| 5.083 | 0.74% | 400 | `rope_fused_3way_kernel_v2_0` |

## Stack filters
- stack_include: `['CausalWanSelfAttention']`
- stack_exclude: `[]`

## Top ops (stack-filtered)
Filtered to stack frames matching include=['CausalWanSelfAttention'] exclude=[].

| device_ms | pct | calls | groups | key |
|---:|---:|---:|---:|---|
| 89.098 | 38.08% | 160 | 40 | `Torch-Compiled Region: 1/2` |
| 52.336 | 22.37% | 160 | 40 | `Torch-Compiled Region: 8/1` |
| 33.504 | 14.32% | 160 | 40 | `kernel_cutlass_kernel_flash_attncuteflash_fwd_sm100FlashAttentionForwardSm100_object_at__tensor0000o11101213_tensor0000o11101213_tensor0000o10111213_tensor0000o11101213_None_None_None_Non_0` |
| 17.375 | 7.43% | 40 | 40 | `Torch-Compiled Region: 1/1` |
| 8.199 | 3.50% | 40 | 40 | `Torch-Compiled Region: 23/0` |
| 7.703 | 3.29% | 40 | 40 | `Torch-Compiled Region: 24/0` |
| 4.417 | 1.89% | 160 | 40 | `aten::copy_` |
| 4.417 | 1.89% | 160 | 40 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1})` |
| 3.930 | 1.68% | 40 | 40 | `Torch-Compiled Region: 25/0` |
| 3.123 | 1.33% | 40 | 40 | `Torch-Compiled Region: 26/0` |
| 3.059 | 1.31% | 240 | 120 | `Torch-Compiled Region: 3/0` |
| 2.563 | 1.10% | 200 | 80 | `Torch-Compiled Region: 4/1` |
| 2.024 | 0.87% | 160 | 40 | `Torch-Compiled Region: 9/0` |
| 2.024 | 0.87% | 160 | 40 | `Torch-Compiled Region: 10/1` |
| 0.125 | 0.05% | 5 | 5 | `## Call CompiledFxGraph fuh67mwwyfifvu65poxmdnfvc3zukk5l7ofvfw7wqcnloi5p64op ##` |
| 0.064 | 0.03% | 5 | 5 | `## Call CompiledFxGraph f4oz2wvcxejxwpdxprvuut3twkm7x5q4nrlvgkq2bxu63do7rkav ##` |
| 0.026 | 0.01% | 2 | 2 | `Torch-Compiled Region: 7/1` |
| 0.013 | 0.01% | 1 | 1 | `Torch-Compiled Region: 11/1` |

## Stack groups: `aten::copy_`

Filtered totals: device_ms=4.417, calls=320

- count=4 device_ms=0.111 self_device_ms=0.111 cpu_ms=0.050
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_1`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`

- count=4 device_ms=0.111 self_device_ms=0.111 cpu_ms=0.049
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_10`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`

- count=4 device_ms=0.111 self_device_ms=0.111 cpu_ms=0.049
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_39`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`

- count=4 device_ms=0.111 self_device_ms=0.111 cpu_ms=0.049
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_2`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`

- count=4 device_ms=0.111 self_device_ms=0.111 cpu_ms=0.049
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_33`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`

- count=4 device_ms=0.111 self_device_ms=0.111 cpu_ms=0.051
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_32`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`

- count=4 device_ms=0.111 self_device_ms=0.111 cpu_ms=0.049
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_22`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`

- count=4 device_ms=0.111 self_device_ms=0.111 cpu_ms=0.048
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_3`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`

- count=4 device_ms=0.111 self_device_ms=0.111 cpu_ms=0.049
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_25`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`

- count=4 device_ms=0.111 self_device_ms=0.111 cpu_ms=0.049
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_13`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`

- count=4 device_ms=0.111 self_device_ms=0.111 cpu_ms=0.049
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_19`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`

- count=4 device_ms=0.111 self_device_ms=0.111 cpu_ms=0.061
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_30`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1730): forward`

## Stack groups: `aten::_to_copy`

(no events)

## Stack groups: `aten::to`

(no events)

## Stack groups: `aten::fill_`

Filtered totals: device_ms=0.000, calls=160

- count=4 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.007
  - `<built-in method fill_ of Tensor object at 0x7e7de4a6aa80>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_0`

- count=4 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.006
  - `<built-in method fill_ of Tensor object at 0x7e7de4a6aa80>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_1`

- count=4 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.006
  - `<built-in method fill_ of Tensor object at 0x7e7de4a6aa80>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_2`

- count=4 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.006
  - `<built-in method fill_ of Tensor object at 0x7e7de4a6aa80>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_3`

- count=4 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.006
  - `<built-in method fill_ of Tensor object at 0x7e7de4a6aa80>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_4`

- count=4 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.005
  - `<built-in method fill_ of Tensor object at 0x7e7de4a6aa80>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_5`

- count=4 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.005
  - `<built-in method fill_ of Tensor object at 0x7e7de4a6aa80>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_6`

- count=4 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.006
  - `<built-in method fill_ of Tensor object at 0x7e7de4a6aa80>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_7`

- count=4 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.006
  - `<built-in method fill_ of Tensor object at 0x7e7de4a6aa80>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_8`

- count=4 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.006
  - `<built-in method fill_ of Tensor object at 0x7e7de4a6aa80>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_9`

- count=4 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.006
  - `<built-in method fill_ of Tensor object at 0x7e7de4a6aa80>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_10`

- count=4 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.005
  - `<built-in method fill_ of Tensor object at 0x7e7de4a6aa80>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_11`
