# Krea pipeline op profile

## Meta
- torch: `2.9.0+cu130` (cuda `13.0`)
- device: `NVIDIA B300 SXM6 AC` cc=(10, 3)
- compile: `False`
- kv_cache_attention_bias: `0.3`
- SCOPE_KV_BIAS_BACKEND: `fa4`
- profiled_wall_time_s: `5.415`

## Top ops (self CUDA)
| self_cuda_ms | pct | calls | key |
|---:|---:|---:|---|
| 160.251 | 28.61% | 1,630 | `aten::addmm` |
| 91.872 | 16.40% | 1,200 | `nvjet_tst_256x128_64x5_2x2_2cta_h_bz_bias_TNT` |
| 34.152 | 6.10% | 200 | `nvjet_tst_256x160_64x5_2x1_2cta_v_bz_bias_TNT` |
| 33.602 | 6.00% | 200 | `nvjet_tst_128x256_64x6_2x1_2cta_v_bz_bias_TNT` |
| 33.446 | 5.97% | 160 | `kernel_cutlass_kernel_flash_attncuteflash_fwd_sm100FlashAttentionForwardSm100_object_at__tensor0000o11101213_tensor0000o11101213_tensor0000o10111213_tensor0000o11101213_None_None_None_Non_0` |
| 22.973 | 4.10% | 240 | `flash_attn::_flash_attn_forward` |
| 22.973 | 4.10% | 240 | `void flash::flash_fwd_kernel<Flash_fwd_kernel_traits<128, 128, 64, 4, false, false, cutlass::bfloat16_t, Flash_kernel_traits<128, 128, 64, 4, cutlass::bfloat16_t> >, false, false, false, false, false, true, false, false>(flash::Flash_fwd_params)` |
| 21.644 | 3.86% | 882 | `aten::mul` |
| 21.579 | 3.85% | 810 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1})` |
| 18.601 | 3.32% | 1,825 | `aten::add` |
| 11.548 | 2.06% | 621 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<c10::BFloat16> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<c10::BFloat16> const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<c10::BFloat16> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<c10::BFloat16> const&)::{lambda(int)#1})` |
| 11.384 | 2.03% | 605 | `aten::native_layer_norm` |
| 11.384 | 2.03% | 605 | `void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<c10::BFloat16, float, false>(int, float, c10::BFloat16 const*, c10::BFloat16 const*, c10::BFloat16 const*, float*, float*, c10::BFloat16*)` |
| 6.345 | 1.13% | 600 | `aten::_fused_rms_norm` |
| 6.345 | 1.13% | 600 | `void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<c10::BFloat16, float, true>(int, float, c10::BFloat16 const*, c10::BFloat16 const*, c10::BFloat16 const*, float*, float*, c10::BFloat16*)` |
| 6.037 | 1.08% | 205 | `aten::gelu` |
| 6.037 | 1.08% | 205 | `void at::native::vectorized_elementwise_kernel<4, at::native::GeluCUDAKernelImpl(at::TensorIteratorBase&, at::native::GeluType)::{lambda()#1}::operator()() const::{lambda()#4}::operator()() const::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul> >(int, at::native::GeluCUDAKernelImpl(at::TensorIteratorBase&, at::native::GeluType)::{lambda()#1}::operator()() const::{lambda()#4}::operator()() const::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul>)` |
| 5.689 | 1.02% | 10,142 | `aten::copy_` |
| 5.405 | 0.96% | 601 | `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul>)` |
| 5.147 | 0.92% | 400 | `rope_fused_3way_kernel_v2` |
| 3.198 | 0.57% | 576 | `Memcpy DtoD (Device -> Device)` |
| 2.602 | 0.46% | 1,266 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1})` |
| 2.160 | 0.39% | 1,292 | `aten::_local_scalar_dense` |
| 2.160 | 0.39% | 1,280 | `Memcpy DtoH (Device -> Pinned)` |
| 1.575 | 0.28% | 816 | `aten::sub` |

## Stack filters
- stack_include: `['CausalWanSelfAttention']`
- stack_exclude: `[]`

## Stack groups: `aten::contiguous`

(no events)

## Stack groups: `aten::copy_`

Filtered totals: device_ms=3.164, calls=560

- count=10 device_ms=0.072 self_device_ms=0.072 cpu_ms=0.085
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_36`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_36`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2090): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2427): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanDiffusionWrapper_0`

- count=10 device_ms=0.072 self_device_ms=0.072 cpu_ms=0.074
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_8`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_8`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2090): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2427): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanDiffusionWrapper_0`

- count=10 device_ms=0.072 self_device_ms=0.072 cpu_ms=0.087
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_30`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_30`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2090): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2427): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanDiffusionWrapper_0`

- count=10 device_ms=0.072 self_device_ms=0.072 cpu_ms=0.079
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_22`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_22`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2090): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2427): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanDiffusionWrapper_0`

- count=10 device_ms=0.072 self_device_ms=0.072 cpu_ms=0.077
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_19`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_19`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2090): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2427): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanDiffusionWrapper_0`

- count=10 device_ms=0.072 self_device_ms=0.072 cpu_ms=0.081
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_32`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_32`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2090): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2427): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanDiffusionWrapper_0`

- count=10 device_ms=0.072 self_device_ms=0.072 cpu_ms=0.080
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_20`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_20`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2090): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2427): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanDiffusionWrapper_0`

- count=10 device_ms=0.072 self_device_ms=0.072 cpu_ms=0.082
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_25`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_25`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2090): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2427): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanDiffusionWrapper_0`

- count=10 device_ms=0.072 self_device_ms=0.072 cpu_ms=0.073
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_10`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_10`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2090): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2427): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanDiffusionWrapper_0`

- count=10 device_ms=0.072 self_device_ms=0.072 cpu_ms=0.076
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_6`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_6`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2090): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2427): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanDiffusionWrapper_0`

- count=10 device_ms=0.072 self_device_ms=0.072 cpu_ms=0.084
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_0`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_0`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2090): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2427): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanDiffusionWrapper_0`

- count=10 device_ms=0.072 self_device_ms=0.072 cpu_ms=0.078
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_39`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_39`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2090): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2427): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanDiffusionWrapper_0`

- count=10 device_ms=0.072 self_device_ms=0.072 cpu_ms=0.080
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_18`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_18`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2090): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2427): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanDiffusionWrapper_0`

- count=10 device_ms=0.072 self_device_ms=0.072 cpu_ms=0.078
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_37`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_37`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2090): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2427): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanDiffusionWrapper_0`

- count=10 device_ms=0.072 self_device_ms=0.072 cpu_ms=0.083
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_28`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_28`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2090): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2427): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanDiffusionWrapper_0`

## Stack groups: `aten::fill_`

Filtered totals: device_ms=0.404, calls=240

- count=6 device_ms=0.011 self_device_ms=0.011 cpu_ms=0.047
  - `<built-in method fill_ of Tensor object at 0x7de07c52b4d0>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_19`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_19`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2090): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2427): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=6 device_ms=0.011 self_device_ms=0.011 cpu_ms=0.048
  - `<built-in method fill_ of Tensor object at 0x7de07c52b4d0>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_3`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_3`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2090): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2427): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=6 device_ms=0.011 self_device_ms=0.011 cpu_ms=0.045
  - `<built-in method fill_ of Tensor object at 0x7de07c52b4d0>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_5`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_5`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2090): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2427): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=6 device_ms=0.011 self_device_ms=0.011 cpu_ms=0.045
  - `<built-in method fill_ of Tensor object at 0x7de07c52b4d0>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_16`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_16`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2090): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2427): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=6 device_ms=0.011 self_device_ms=0.011 cpu_ms=0.048
  - `<built-in method fill_ of Tensor object at 0x7de07c52b4d0>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_21`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_21`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2090): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2427): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=6 device_ms=0.011 self_device_ms=0.011 cpu_ms=0.051
  - `<built-in method fill_ of Tensor object at 0x7de07c52b4d0>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_0`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_0`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2090): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2427): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=6 device_ms=0.011 self_device_ms=0.011 cpu_ms=0.045
  - `<built-in method fill_ of Tensor object at 0x7de07c52b4d0>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_13`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_13`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2090): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2427): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=6 device_ms=0.011 self_device_ms=0.011 cpu_ms=0.047
  - `<built-in method fill_ of Tensor object at 0x7de07c52b4d0>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_32`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_32`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2090): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2427): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=6 device_ms=0.011 self_device_ms=0.011 cpu_ms=0.048
  - `<built-in method fill_ of Tensor object at 0x7de07c52b4d0>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_38`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_38`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2090): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2427): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=6 device_ms=0.010 self_device_ms=0.010 cpu_ms=0.046
  - `<built-in method fill_ of Tensor object at 0x7de07c52b4d0>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_17`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_17`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2090): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2427): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=6 device_ms=0.010 self_device_ms=0.010 cpu_ms=0.048
  - `<built-in method fill_ of Tensor object at 0x7de07c52b4d0>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_37`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_37`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2090): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2427): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=6 device_ms=0.010 self_device_ms=0.010 cpu_ms=0.051
  - `<built-in method fill_ of Tensor object at 0x7de07c52b4d0>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_29`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_29`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2090): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2427): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=6 device_ms=0.010 self_device_ms=0.010 cpu_ms=0.045
  - `<built-in method fill_ of Tensor object at 0x7de07c52b4d0>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_11`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_11`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2090): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2427): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=6 device_ms=0.010 self_device_ms=0.010 cpu_ms=0.045
  - `<built-in method fill_ of Tensor object at 0x7de07c52b4d0>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_14`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_14`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2090): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2427): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=6 device_ms=0.010 self_device_ms=0.010 cpu_ms=0.052
  - `<built-in method fill_ of Tensor object at 0x7de07c52b4d0>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_27`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_27`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2090): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2427): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

## Stack groups: `aten::clone`

(no events)
