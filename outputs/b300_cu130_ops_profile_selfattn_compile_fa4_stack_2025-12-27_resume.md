# Krea pipeline op profile

## Meta
- torch: `2.9.0+cu130` (cuda `13.0`)
- device: `NVIDIA B300 SXM6 AC` cc=(10, 3)
- compile: `True`
- kv_cache_attention_bias: `0.3`
- SCOPE_KV_BIAS_BACKEND: `fa4`
- profiled_wall_time_s: `12.805`

## Top ops (self CUDA)
| self_cuda_ms | pct | calls | key |
|---:|---:|---:|---|
| 84.508 | 12.31% | 600 | `aten::mm` |
| 84.424 | 12.30% | 200 | `## Call CompiledFxGraph fzkr5p2ai4jlimh5nkeoy4kj6gag3djpy7rrfhfz6gvfo74pftp2 ##` |
| 69.742 | 10.16% | 630 | `aten::addmm` |
| 38.408 | 5.60% | 200 | `nvjet_tst_256x128_64x5_2x1_2cta_v_bz_bias_TNT` |
| 35.248 | 5.13% | 160 | `## Call CompiledFxGraph fe3p354nzwxb6oufl3yqqnv5n7oj6ozrogyabdrellckb4pv7r4i ##` |
| 34.630 | 5.04% | 200 | `nvjet_tst_256x160_64x5_2x1_2cta_v_bz_TNT` |
| 34.080 | 4.96% | 200 | `nvjet_tst_128x256_64x6_2x1_2cta_v_bz_TNT` |
| 33.513 | 4.88% | 160 | `Torch-Compiled Region: 9/1` |
| 33.513 | 4.88% | 160 | `kernel_cutlass_kernel_flash_attncuteflash_fwd_sm100FlashAttentionForwardSm100_object_at__tensor0000o11101213_tensor0000o11101213_tensor0000o10111213_tensor0000o11101213_None_None_None_Non_0` |
| 30.726 | 4.48% | 400 | `nvjet_tst_256x128_64x5_2x2_2cta_h_bz_bias_TNT` |
| 18.891 | 2.75% | 200 | `## Call CompiledFxGraph ffw4q4kngccow57pa2mkgee77ogpi55si5mvf26jkdld7scmyvs7 ##` |
| 18.338 | 2.67% | 240 | `## Call CompiledFxGraph fcceaggubatdtr6yp77cg7bal2pxb7hmqu76agprelqh5hx5asm4 ##` |
| 15.797 | 2.30% | 200 | `nvjet_tst_256x128_64x5_2x2_2cta_h_bz_TNT` |
| 10.835 | 1.58% | 4,049 | `aten::copy_` |
| 10.599 | 1.54% | 240 | `FlashAttnVarlenFunc` |
| 10.599 | 1.54% | 240 | `kernel_cutlass_kernel_flash_attncuteflash_fwd_sm100FlashAttentionForwardSm100_object_at__tensor000o111012_tensor000o111012_tensor000o101112_tensorptrbf16gmemalign16odiv81div8_None_tensorp_0` |
| 10.582 | 1.54% | 400 | `triton_red_fused_add_mul_native_layer_norm_view_0` |
| 10.582 | 1.54% | 400 | `triton_red_fused_add_mul_native_layer_norm_view_0` |
| 10.156 | 1.48% | 3,308 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1})` |
| 8.765 | 1.28% | 40 | `## Call CompiledFxGraph fk64ajjwdf2q6mjilnqgjewiklnrsdt2pevlvsuyisl466jtpaa3 ##` |
| 6.784 | 0.99% | 200 | `triton_poi_fused_addmm_gelu_view_1` |
| 6.784 | 0.99% | 200 | `triton_poi_fused_addmm_gelu_view_1` |
| 5.486 | 0.80% | 200 | `## Call CompiledFxGraph f732rfwuj3e43dof7xqnzwq6bnjpadzz7val5ryx64zqiuudjyof ##` |
| 5.088 | 0.74% | 200 | `## Call CompiledFxGraph fuh67mwwyfifvu65poxmdnfvc3zukk5l7ofvfw7wqcnloi5p64op ##` |
| 5.015 | 0.73% | 400 | `rope_fused_3way_kernel_v2_0` |

## Stack filters
- stack_include: `['CausalWanSelfAttention']`
- stack_exclude: `[]`

## Top ops (stack-filtered)
Filtered to stack frames matching include=['CausalWanSelfAttention'] exclude=[].

| device_ms | pct | calls | groups | key |
|---:|---:|---:|---:|---|
| 89.935 | 31.30% | 160 | 40 | `Torch-Compiled Region: 1/2` |
| 53.175 | 18.51% | 160 | 40 | `Torch-Compiled Region: 8/1` |
| 51.207 | 17.82% | 160 | 41 | `Torch-Compiled Region: 9/1` |
| 33.513 | 11.66% | 160 | 41 | `kernel_cutlass_kernel_flash_attncuteflash_fwd_sm100FlashAttentionForwardSm100_object_at__tensor0000o11101213_tensor0000o11101213_tensor0000o10111213_tensor0000o11101213_None_None_None_Non_0` |
| 17.305 | 6.02% | 40 | 40 | `Torch-Compiled Region: 1/1` |
| 8.162 | 2.84% | 40 | 40 | `Torch-Compiled Region: 21/0` |
| 7.669 | 2.67% | 40 | 40 | `Torch-Compiled Region: 22/0` |
| 5.305 | 1.85% | 320 | 47 | `aten::copy_` |
| 5.015 | 1.75% | 400 | 160 | `Torch-Compiled Region: 3/0` |
| 4.421 | 1.54% | 160 | 47 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1})` |
| 3.915 | 1.36% | 40 | 40 | `Torch-Compiled Region: 23/0` |
| 3.103 | 1.08% | 40 | 40 | `Torch-Compiled Region: 24/0` |
| 2.566 | 0.89% | 201 | 80 | `Torch-Compiled Region: 4/1` |
| 0.884 | 0.31% | 160 | 47 | `Memcpy DtoD (Device -> Device)` |
| 0.439 | 0.15% | 2 | 2 | `## Call CompiledFxGraph fk64ajjwdf2q6mjilnqgjewiklnrsdt2pevlvsuyisl466jtpaa3 ##` |
| 0.384 | 0.13% | 31 | 26 | `## Call CompiledFxGraph f4oz2wvcxejxwpdxprvuut3twkm7x5q4nrlvgkq2bxu63do7rkav ##` |
| 0.195 | 0.07% | 2 | 2 | `Torch-Compiled Region: 15/2` |
| 0.099 | 0.03% | 4 | 4 | `## Call CompiledFxGraph fuh67mwwyfifvu65poxmdnfvc3zukk5l7ofvfw7wqcnloi5p64op ##` |
| 0.013 | 0.00% | 1 | 1 | `Torch-Compiled Region: 7/1` |

## Stack groups: `aten::copy_`

Filtered totals: device_ms=5.305, calls=328

- count=8 device_ms=0.133 self_device_ms=0.133 cpu_ms=0.090
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_25`

- count=8 device_ms=0.133 self_device_ms=0.133 cpu_ms=0.099
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_23`

- count=8 device_ms=0.133 self_device_ms=0.133 cpu_ms=0.092
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_6`

- count=8 device_ms=0.133 self_device_ms=0.133 cpu_ms=0.093
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_3`

- count=8 device_ms=0.133 self_device_ms=0.133 cpu_ms=0.102
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_1`

- count=8 device_ms=0.133 self_device_ms=0.133 cpu_ms=0.094
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_5`

- count=8 device_ms=0.133 self_device_ms=0.133 cpu_ms=0.093
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_28`

- count=8 device_ms=0.133 self_device_ms=0.133 cpu_ms=0.094
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_2`

- count=8 device_ms=0.133 self_device_ms=0.133 cpu_ms=0.093
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_14`

- count=8 device_ms=0.133 self_device_ms=0.133 cpu_ms=0.101
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_4`

- count=8 device_ms=0.133 self_device_ms=0.133 cpu_ms=0.093
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_27`

- count=8 device_ms=0.133 self_device_ms=0.133 cpu_ms=0.099
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_9`

- count=8 device_ms=0.133 self_device_ms=0.133 cpu_ms=0.091
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_15`

- count=8 device_ms=0.133 self_device_ms=0.133 cpu_ms=0.102
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_36`

- count=8 device_ms=0.133 self_device_ms=0.133 cpu_ms=0.092
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_19`

## Stack groups: `aten::_to_copy`

(no events)

## Stack groups: `aten::to`

(no events)

## Stack groups: `aten::fill_`

Filtered totals: device_ms=0.000, calls=8

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.001
  - `<built-in method fill_ of Tensor object at 0x7dd864cf9590>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_13`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.001
  - `<built-in method fill_ of Tensor object at 0x7dd864cf9590>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_32`

- count=2 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.003
  - `<built-in method fill_ of Tensor object at 0x7dd864cf9590>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_8`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.002
  - `<built-in method fill_ of Tensor object at 0x7dd864cf9590>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_29`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.001
  - `<built-in method fill_ of Tensor object at 0x7dd864cf9590>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_39`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.001
  - `<built-in method fill_ of Tensor object at 0x7dd864cf9590>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1190): torch_dynamo_resume_in_forward_at_1190`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_10`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.002
  - `<built-in method fill_ of Tensor object at 0x7dd864cf9590>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1196): torch_dynamo_resume_in_forward_at_1196`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(939): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_11`
