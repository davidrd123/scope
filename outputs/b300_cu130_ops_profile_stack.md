# Krea pipeline op profile

## Meta
- torch: `2.9.0+cu130` (cuda `13.0`)
- device: `NVIDIA B300 SXM6 AC` cc=(10, 3)
- compile: `False`
- kv_cache_attention_bias: `0.3`
- SCOPE_KV_BIAS_BACKEND: `fa4`
- profiled_wall_time_s: `5.290`

## Top ops (self CUDA)
| self_cuda_ms | pct | calls | key |
|---:|---:|---:|---|
| 151.911 | 26.77% | 1,230 | `aten::addmm` |
| 45.690 | 8.05% | 600 | `nvjet_tst_256x128_64x5_2x2_2cta_h_bz_bias_TNT` |
| 37.829 | 6.67% | 200 | `nvjet_tst_256x128_64x5_2x1_2cta_v_bz_bias_TNT` |
| 34.171 | 6.02% | 200 | `nvjet_tst_256x160_64x5_2x1_2cta_v_bz_bias_TNT` |
| 33.604 | 5.92% | 200 | `nvjet_tst_128x256_64x6_2x1_2cta_v_bz_bias_TNT` |
| 33.405 | 5.89% | 160 | `kernel_cutlass_kernel_flash_attncuteflash_fwd_sm100FlashAttentionForwardSm100_object_at__tensor0000o11101213_tensor0000o11101213_tensor0000o10111213_tensor0000o11101213_None_None_None_Non_0` |
| 22.973 | 4.05% | 240 | `flash_attn::_flash_attn_forward` |
| 22.973 | 4.05% | 240 | `void flash::flash_fwd_kernel<Flash_fwd_kernel_traits<128, 128, 64, 4, false, false, cutlass::bfloat16_t, Flash_kernel_traits<128, 128, 64, 4, cutlass::bfloat16_t> >, false, false, false, false, false, true, false, false>(flash::Flash_fwd_params)` |
| 21.600 | 3.81% | 882 | `aten::mul` |
| 21.546 | 3.80% | 807 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1})` |
| 19.166 | 3.38% | 10,542 | `aten::copy_` |
| 18.520 | 3.26% | 1,825 | `aten::add` |
| 17.854 | 3.15% | 1,626 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1})` |
| 11.482 | 2.02% | 617 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<c10::BFloat16> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<c10::BFloat16> const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<c10::BFloat16> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<c10::BFloat16> const&)::{lambda(int)#1})` |
| 11.392 | 2.01% | 605 | `aten::native_layer_norm` |
| 11.392 | 2.01% | 605 | `void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<c10::BFloat16, float, false>(int, float, c10::BFloat16 const*, c10::BFloat16 const*, c10::BFloat16 const*, float*, float*, c10::BFloat16*)` |
| 6.118 | 1.08% | 600 | `aten::_fused_rms_norm` |
| 6.118 | 1.08% | 600 | `void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<c10::BFloat16, float, true>(int, float, c10::BFloat16 const*, c10::BFloat16 const*, c10::BFloat16 const*, float*, float*, c10::BFloat16*)` |
| 6.045 | 1.07% | 205 | `aten::gelu` |
| 6.045 | 1.07% | 205 | `void at::native::vectorized_elementwise_kernel<4, at::native::GeluCUDAKernelImpl(at::TensorIteratorBase&, at::native::GeluType)::{lambda()#1}::operator()() const::{lambda()#4}::operator()() const::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul> >(int, at::native::GeluCUDAKernelImpl(at::TensorIteratorBase&, at::native::GeluType)::{lambda()#1}::operator()() const::{lambda()#4}::operator()() const::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul>)` |
| 5.401 | 0.95% | 600 | `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul>)` |
| 5.022 | 0.88% | 400 | `rope_fused_3way_kernel_v2` |
| 2.029 | 0.36% | 1,292 | `aten::_local_scalar_dense` |
| 2.029 | 0.36% | 1,280 | `Memcpy DtoH (Device -> Pinned)` |
| 1.496 | 0.26% | 816 | `aten::sub` |

## Stack groups: `aten::contiguous`

- count=5 device_ms=0.133 self_device_ms=0.000 cpu_ms=0.063
  - `<built-in method rms_norm of type object at 0x7217d43d6c40>`
  - `torch/nn/functional.py(2906): rms_norm`
  - `scope/core/pipelines/krea_realtime_video/modules/model.py(237): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanRMSNorm_30`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(913): qkv_fn`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_10`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_10`

- count=5 device_ms=0.132 self_device_ms=0.000 cpu_ms=0.065
  - `<built-in method rms_norm of type object at 0x7217d43d6c40>`
  - `torch/nn/functional.py(2906): rms_norm`
  - `scope/core/pipelines/krea_realtime_video/modules/model.py(237): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanRMSNorm_78`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(913): qkv_fn`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_26`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_26`

- count=5 device_ms=0.132 self_device_ms=0.000 cpu_ms=0.064
  - `<built-in method rms_norm of type object at 0x7217d43d6c40>`
  - `torch/nn/functional.py(2906): rms_norm`
  - `scope/core/pipelines/krea_realtime_video/modules/model.py(237): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanRMSNorm_48`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(913): qkv_fn`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_16`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_16`

- count=5 device_ms=0.132 self_device_ms=0.000 cpu_ms=0.063
  - `<built-in method rms_norm of type object at 0x7217d43d6c40>`
  - `torch/nn/functional.py(2906): rms_norm`
  - `scope/core/pipelines/krea_realtime_video/modules/model.py(237): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanRMSNorm_24`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(913): qkv_fn`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_8`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_8`

- count=5 device_ms=0.132 self_device_ms=0.000 cpu_ms=0.064
  - `<built-in method rms_norm of type object at 0x7217d43d6c40>`
  - `torch/nn/functional.py(2906): rms_norm`
  - `scope/core/pipelines/krea_realtime_video/modules/model.py(237): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanRMSNorm_51`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(913): qkv_fn`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_17`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_17`

- count=5 device_ms=0.131 self_device_ms=0.000 cpu_ms=0.066
  - `<built-in method rms_norm of type object at 0x7217d43d6c40>`
  - `torch/nn/functional.py(2906): rms_norm`
  - `scope/core/pipelines/krea_realtime_video/modules/model.py(237): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanRMSNorm_3`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(913): qkv_fn`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_1`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_1`

- count=5 device_ms=0.131 self_device_ms=0.000 cpu_ms=0.065
  - `<built-in method rms_norm of type object at 0x7217d43d6c40>`
  - `torch/nn/functional.py(2906): rms_norm`
  - `scope/core/pipelines/krea_realtime_video/modules/model.py(237): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanRMSNorm_42`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(913): qkv_fn`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_14`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_14`

- count=5 device_ms=0.131 self_device_ms=0.000 cpu_ms=0.064
  - `<built-in method rms_norm of type object at 0x7217d43d6c40>`
  - `torch/nn/functional.py(2906): rms_norm`
  - `scope/core/pipelines/krea_realtime_video/modules/model.py(237): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanRMSNorm_75`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(913): qkv_fn`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_25`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_25`

- count=5 device_ms=0.131 self_device_ms=0.000 cpu_ms=0.072
  - `<built-in method rms_norm of type object at 0x7217d43d6c40>`
  - `torch/nn/functional.py(2906): rms_norm`
  - `scope/core/pipelines/krea_realtime_video/modules/model.py(237): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanRMSNorm_39`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(913): qkv_fn`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_13`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_13`

- count=5 device_ms=0.131 self_device_ms=0.000 cpu_ms=0.064
  - `<built-in method rms_norm of type object at 0x7217d43d6c40>`
  - `torch/nn/functional.py(2906): rms_norm`
  - `scope/core/pipelines/krea_realtime_video/modules/model.py(237): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanRMSNorm_60`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(913): qkv_fn`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_20`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_20`

- count=5 device_ms=0.131 self_device_ms=0.000 cpu_ms=0.064
  - `<built-in method rms_norm of type object at 0x7217d43d6c40>`
  - `torch/nn/functional.py(2906): rms_norm`
  - `scope/core/pipelines/krea_realtime_video/modules/model.py(237): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanRMSNorm_114`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(913): qkv_fn`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_38`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_38`

- count=5 device_ms=0.131 self_device_ms=0.000 cpu_ms=0.072
  - `<built-in method rms_norm of type object at 0x7217d43d6c40>`
  - `torch/nn/functional.py(2906): rms_norm`
  - `scope/core/pipelines/krea_realtime_video/modules/model.py(237): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanRMSNorm_72`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(913): qkv_fn`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_24`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_24`

- count=5 device_ms=0.131 self_device_ms=0.000 cpu_ms=0.064
  - `<built-in method rms_norm of type object at 0x7217d43d6c40>`
  - `torch/nn/functional.py(2906): rms_norm`
  - `scope/core/pipelines/krea_realtime_video/modules/model.py(237): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanRMSNorm_96`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(913): qkv_fn`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_32`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_32`

- count=5 device_ms=0.131 self_device_ms=0.000 cpu_ms=0.064
  - `<built-in method rms_norm of type object at 0x7217d43d6c40>`
  - `torch/nn/functional.py(2906): rms_norm`
  - `scope/core/pipelines/krea_realtime_video/modules/model.py(237): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanRMSNorm_81`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(913): qkv_fn`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_27`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_27`

- count=5 device_ms=0.131 self_device_ms=0.000 cpu_ms=0.063
  - `<built-in method rms_norm of type object at 0x7217d43d6c40>`
  - `torch/nn/functional.py(2906): rms_norm`
  - `scope/core/pipelines/krea_realtime_video/modules/model.py(237): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanRMSNorm_54`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(913): qkv_fn`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_18`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_18`

## Stack groups: `aten::transpose`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.004
  - `<built-in method transpose of Tensor object at 0x721242aaf890>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2090): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2427): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanDiffusionWrapper_0`
  - `scope/core/pipelines/krea_realtime_video/blocks/recompute_kv_cache.py(133): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`
  - `scope/core/pipelines/krea_realtime_video/modular_blocks.py(173): __call__`

- count=5 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.009
  - `<built-in function linear>`
  - `torch/nn/modules/linear.py(130): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Linear_0`
  - `torch/nn/modules/container.py(245): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Sequential_0`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2090): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2427): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`

- count=5 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.005
  - `<built-in function linear>`
  - `torch/nn/modules/linear.py(130): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Linear_1`
  - `torch/nn/modules/container.py(245): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Sequential_0`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2090): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2427): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`

- count=5 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.006
  - `<built-in function linear>`
  - `torch/nn/modules/linear.py(130): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Linear_2`
  - `torch/nn/modules/container.py(245): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Sequential_1`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2090): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2427): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`

- count=5 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.006
  - `<built-in function linear>`
  - `torch/nn/modules/linear.py(130): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Linear_3`
  - `torch/nn/modules/container.py(245): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Sequential_2`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2090): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2427): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`

- count=5 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.005
  - `<built-in function linear>`
  - `torch/nn/modules/linear.py(130): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Linear_4`
  - `torch/nn/modules/container.py(245): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Sequential_2`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2090): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2427): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`

- count=5 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.006
  - `<built-in function linear>`
  - `torch/nn/modules/linear.py(130): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Linear_5`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(913): qkv_fn`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_0`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_0`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2090): _forward_inference`

- count=5 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.010
  - `<built-in function linear>`
  - `torch/nn/modules/linear.py(130): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Linear_6`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_0`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_0`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2090): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2427): forward`

- count=5 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.007
  - `<built-in function linear>`
  - `torch/nn/modules/linear.py(130): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Linear_7`
  - `scope/core/pipelines/krea_realtime_video/modules/model.py(335): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanT2VCrossAttention_0`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_0`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2090): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2427): forward`

- count=5 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.008
  - `<built-in function linear>`
  - `torch/nn/modules/linear.py(130): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Linear_8`
  - `scope/core/pipelines/krea_realtime_video/modules/model.py(335): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanT2VCrossAttention_0`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_0`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2090): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2427): forward`

- count=5 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.006
  - `<built-in function linear>`
  - `torch/nn/modules/linear.py(130): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Linear_9`
  - `torch/nn/modules/container.py(245): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Sequential_3`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_0`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2090): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2427): forward`

- count=5 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.006
  - `<built-in function linear>`
  - `torch/nn/modules/linear.py(130): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Linear_10`
  - `torch/nn/modules/container.py(245): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Sequential_3`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_0`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2090): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2427): forward`

- count=5 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.007
  - `<built-in function linear>`
  - `torch/nn/modules/linear.py(130): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Linear_11`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(913): qkv_fn`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_1`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_1`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2090): _forward_inference`

- count=5 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.010
  - `<built-in function linear>`
  - `torch/nn/modules/linear.py(130): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Linear_12`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_1`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_1`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2090): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2427): forward`

- count=5 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.007
  - `<built-in function linear>`
  - `torch/nn/modules/linear.py(130): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Linear_13`
  - `scope/core/pipelines/krea_realtime_video/modules/model.py(335): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanT2VCrossAttention_1`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_1`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2090): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2427): forward`

## Stack groups: `aten::copy_`

- count=384 device_ms=0.663 self_device_ms=0.663 cpu_ms=1.533
  - `<built-in method conv3d of type object at 0x7217d43d6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_1`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(666): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Decoder3d_0`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(875): stream_decode`
  - `scope/core/pipelines/wan2_1/vae/wan.py(262): decode_to_pixel`
  - `scope/core/pipelines/wan2_1/blocks/decode.py(47): __call__`

- count=384 device_ms=0.661 self_device_ms=0.661 cpu_ms=1.538
  - `<built-in method conv3d of type object at 0x7217d43d6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_2`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: ResidualBlock_0`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(666): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Decoder3d_0`

- count=384 device_ms=0.400 self_device_ms=0.400 cpu_ms=1.687
  - `<built-in method conv3d of type object at 0x7217d43d6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_3`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: ResidualBlock_0`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(666): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Decoder3d_0`

- count=10 device_ms=0.164 self_device_ms=0.164 cpu_ms=0.082
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_12`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_12`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2090): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2427): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`

- count=10 device_ms=0.164 self_device_ms=0.164 cpu_ms=0.082
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

- count=10 device_ms=0.164 self_device_ms=0.164 cpu_ms=0.092
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

- count=10 device_ms=0.164 self_device_ms=0.164 cpu_ms=0.072
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

- count=10 device_ms=0.164 self_device_ms=0.164 cpu_ms=0.090
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_2`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_2`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2090): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2427): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`

- count=10 device_ms=0.163 self_device_ms=0.163 cpu_ms=0.073
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

- count=10 device_ms=0.163 self_device_ms=0.163 cpu_ms=0.073
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

- count=10 device_ms=0.163 self_device_ms=0.163 cpu_ms=0.077
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_1`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_1`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2090): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2427): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`

- count=10 device_ms=0.163 self_device_ms=0.163 cpu_ms=0.071
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

- count=10 device_ms=0.163 self_device_ms=0.163 cpu_ms=0.073
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

- count=10 device_ms=0.163 self_device_ms=0.163 cpu_ms=0.085
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

- count=10 device_ms=0.163 self_device_ms=0.163 cpu_ms=0.082
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_33`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_33`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2090): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2427): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`

## Stack groups: `aten::_to_copy`

- count=16 device_ms=0.063 self_device_ms=0.000 cpu_ms=0.149
  - `<built-in method double of Tensor object at 0x721242aaf160>`
  - `scope/core/pipelines/wan2_1/components/generator.py(261): _convert_flow_pred_to_x0`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanDiffusionWrapper_0`
  - `scope/core/pipelines/wan2_1/blocks/denoise.py(258): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`
  - `scope/core/pipelines/krea_realtime_video/modular_blocks.py(173): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`
  - `scope/core/pipelines/krea_realtime_video/pipeline.py(256): _generate`
  - `scope/core/pipelines/krea_realtime_video/pipeline.py(250): __call__`
  - `profile_krea_pipeline_ops.py(273): main`

- count=80 device_ms=0.063 self_device_ms=0.000 cpu_ms=1.533
  - `<built-in method tensor of type object at 0x7217d43d6c40>`
  - `scope/core/pipelines/wan2_1/utils.py(7): initialize_kv_cache`
  - `scope/core/pipelines/krea_realtime_video/blocks/recompute_kv_cache.py(133): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`
  - `scope/core/pipelines/krea_realtime_video/modular_blocks.py(173): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`
  - `scope/core/pipelines/krea_realtime_video/pipeline.py(256): _generate`
  - `scope/core/pipelines/krea_realtime_video/pipeline.py(250): __call__`
  - `profile_krea_pipeline_ops.py(273): main`
  - `profile_krea_pipeline_ops.py(445): <module>`

- count=4 device_ms=0.015 self_device_ms=0.000 cpu_ms=0.038
  - `<built-in method double of Tensor object at 0x721242aaf160>`
  - `scope/core/pipelines/wan2_1/components/generator.py(261): _convert_flow_pred_to_x0`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanDiffusionWrapper_0`
  - `scope/core/pipelines/krea_realtime_video/blocks/recompute_kv_cache.py(133): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`
  - `scope/core/pipelines/krea_realtime_video/modular_blocks.py(173): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`
  - `scope/core/pipelines/krea_realtime_video/pipeline.py(256): _generate`
  - `scope/core/pipelines/krea_realtime_video/pipeline.py(250): __call__`
  - `profile_krea_pipeline_ops.py(273): main`

- count=4 device_ms=0.014 self_device_ms=0.000 cpu_ms=0.041
  - `<built-in method to of Tensor object at 0x72145bf73ac0>`
  - `scope/core/pipelines/wan2_1/components/generator.py(261): _convert_flow_pred_to_x0`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanDiffusionWrapper_0`
  - `scope/core/pipelines/wan2_1/blocks/denoise.py(258): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`
  - `scope/core/pipelines/krea_realtime_video/modular_blocks.py(173): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`
  - `scope/core/pipelines/krea_realtime_video/pipeline.py(256): _generate`
  - `scope/core/pipelines/krea_realtime_video/pipeline.py(250): __call__`
  - `profile_krea_pipeline_ops.py(273): main`

- count=4 device_ms=0.012 self_device_ms=0.000 cpu_ms=0.036
  - `<built-in method type_as of Tensor object at 0x721242aafb60>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2090): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2427): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanDiffusionWrapper_0`
  - `scope/core/pipelines/wan2_1/blocks/denoise.py(258): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`
  - `scope/core/pipelines/krea_realtime_video/modular_blocks.py(173): __call__`

- count=4 device_ms=0.008 self_device_ms=0.000 cpu_ms=0.041
  - `<built-in method type of Tensor object at 0x721242aaf8e0>`
  - `scope/core/pipelines/krea_realtime_video/modules/model.py(59): sinusoidal_embedding_1d`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2090): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2427): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanDiffusionWrapper_0`
  - `scope/core/pipelines/wan2_1/blocks/denoise.py(258): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`

- count=3 device_ms=0.005 self_device_ms=0.000 cpu_ms=0.034
  - `<built-in method type_as of Tensor object at 0x721242aafb60>`
  - `scope/core/pipelines/wan2_1/components/scheduler.py(166): add_noise`
  - `scope/core/pipelines/wan2_1/blocks/denoise.py(258): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`
  - `scope/core/pipelines/krea_realtime_video/modular_blocks.py(173): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`
  - `scope/core/pipelines/krea_realtime_video/pipeline.py(256): _generate`
  - `scope/core/pipelines/krea_realtime_video/pipeline.py(250): __call__`
  - `profile_krea_pipeline_ops.py(273): main`
  - `profile_krea_pipeline_ops.py(445): <module>`

- count=1 device_ms=0.004 self_device_ms=0.000 cpu_ms=0.010
  - `<built-in method to of Tensor object at 0x72145bf73ac0>`
  - `scope/core/pipelines/wan2_1/components/generator.py(261): _convert_flow_pred_to_x0`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanDiffusionWrapper_0`
  - `scope/core/pipelines/krea_realtime_video/blocks/recompute_kv_cache.py(133): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`
  - `scope/core/pipelines/krea_realtime_video/modular_blocks.py(173): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`
  - `scope/core/pipelines/krea_realtime_video/pipeline.py(256): _generate`
  - `scope/core/pipelines/krea_realtime_video/pipeline.py(250): __call__`
  - `profile_krea_pipeline_ops.py(273): main`

- count=1 device_ms=0.003 self_device_ms=0.000 cpu_ms=0.011
  - `<built-in method type_as of Tensor object at 0x721242aafb60>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2090): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2427): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanDiffusionWrapper_0`
  - `scope/core/pipelines/krea_realtime_video/blocks/recompute_kv_cache.py(133): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`
  - `scope/core/pipelines/krea_realtime_video/modular_blocks.py(173): __call__`

- count=1 device_ms=0.002 self_device_ms=0.000 cpu_ms=0.015
  - `<built-in method type of Tensor object at 0x721242aaf8e0>`
  - `scope/core/pipelines/krea_realtime_video/modules/model.py(59): sinusoidal_embedding_1d`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2090): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2427): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanDiffusionWrapper_0`
  - `scope/core/pipelines/krea_realtime_video/blocks/recompute_kv_cache.py(133): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.017
  - `<built-in method float of Tensor object at 0x7214702bd900>`
  - `scope/core/pipelines/wan2_1/vae/wan.py(262): decode_to_pixel`
  - `scope/core/pipelines/wan2_1/blocks/decode.py(47): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`
  - `scope/core/pipelines/krea_realtime_video/modular_blocks.py(173): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`
  - `scope/core/pipelines/krea_realtime_video/pipeline.py(256): _generate`
  - `scope/core/pipelines/krea_realtime_video/pipeline.py(250): __call__`
  - `profile_krea_pipeline_ops.py(273): main`
  - `profile_krea_pipeline_ops.py(445): <module>`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.012
  - `<built-in method to of Tensor object at 0x72145bf73ac0>`
  - `scope/core/pipelines/krea_realtime_video/blocks/prepare_context_frames.py(89): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`
  - `scope/core/pipelines/krea_realtime_video/modular_blocks.py(173): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`
  - `scope/core/pipelines/krea_realtime_video/pipeline.py(256): _generate`
  - `scope/core/pipelines/krea_realtime_video/pipeline.py(250): __call__`
  - `profile_krea_pipeline_ops.py(273): main`
  - `profile_krea_pipeline_ops.py(445): <module>`

## Stack groups: `aten::fill_`

- count=80 device_ms=0.991 self_device_ms=0.991 cpu_ms=0.317
  - `<built-in method zero_ of Tensor object at 0x7214637faa30>`
  - `scope/core/pipelines/wan2_1/utils.py(7): initialize_kv_cache`
  - `scope/core/pipelines/krea_realtime_video/blocks/recompute_kv_cache.py(133): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`
  - `scope/core/pipelines/krea_realtime_video/modular_blocks.py(173): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`
  - `scope/core/pipelines/krea_realtime_video/pipeline.py(256): _generate`
  - `scope/core/pipelines/krea_realtime_video/pipeline.py(250): __call__`
  - `profile_krea_pipeline_ops.py(273): main`
  - `profile_krea_pipeline_ops.py(445): <module>`

- count=384 device_ms=0.663 self_device_ms=0.000 cpu_ms=1.936
  - `<built-in method conv3d of type object at 0x7217d43d6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_1`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(666): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Decoder3d_0`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(875): stream_decode`
  - `scope/core/pipelines/wan2_1/vae/wan.py(262): decode_to_pixel`
  - `scope/core/pipelines/wan2_1/blocks/decode.py(47): __call__`

- count=384 device_ms=0.661 self_device_ms=0.000 cpu_ms=1.945
  - `<built-in method conv3d of type object at 0x7217d43d6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_2`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: ResidualBlock_0`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(666): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Decoder3d_0`

- count=384 device_ms=0.400 self_device_ms=0.000 cpu_ms=2.074
  - `<built-in method conv3d of type object at 0x7217d43d6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_3`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: ResidualBlock_0`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(666): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Decoder3d_0`

- count=6 device_ms=0.011 self_device_ms=0.011 cpu_ms=0.046
  - `<built-in method fill_ of Tensor object at 0x721242aad860>`
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

- count=6 device_ms=0.011 self_device_ms=0.011 cpu_ms=0.044
  - `<built-in method fill_ of Tensor object at 0x721242aad860>`
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

- count=6 device_ms=0.011 self_device_ms=0.011 cpu_ms=0.048
  - `<built-in method fill_ of Tensor object at 0x721242aad860>`
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

- count=6 device_ms=0.011 self_device_ms=0.011 cpu_ms=0.047
  - `<built-in method fill_ of Tensor object at 0x721242aad860>`
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

- count=6 device_ms=0.011 self_device_ms=0.011 cpu_ms=0.047
  - `<built-in method fill_ of Tensor object at 0x721242aad860>`
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

- count=6 device_ms=0.011 self_device_ms=0.011 cpu_ms=0.061
  - `<built-in method fill_ of Tensor object at 0x721242aad860>`
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

- count=6 device_ms=0.011 self_device_ms=0.011 cpu_ms=0.045
  - `<built-in method fill_ of Tensor object at 0x721242aad860>`
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

- count=6 device_ms=0.011 self_device_ms=0.011 cpu_ms=0.067
  - `<built-in method fill_ of Tensor object at 0x721242aad860>`
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

- count=6 device_ms=0.011 self_device_ms=0.011 cpu_ms=0.048
  - `<built-in method fill_ of Tensor object at 0x721242aad860>`
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

- count=6 device_ms=0.011 self_device_ms=0.011 cpu_ms=0.046
  - `<built-in method fill_ of Tensor object at 0x721242aad860>`
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

- count=6 device_ms=0.011 self_device_ms=0.011 cpu_ms=0.046
  - `<built-in method fill_ of Tensor object at 0x721242aad860>`
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

## Stack groups: `aten::cat`

- count=8 device_ms=0.197 self_device_ms=0.197 cpu_ms=0.104
  - `<built-in method cat of type object at 0x7217d43d6c40>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2090): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2427): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanDiffusionWrapper_0`
  - `scope/core/pipelines/wan2_1/blocks/denoise.py(258): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`
  - `scope/core/pipelines/krea_realtime_video/modular_blocks.py(173): __call__`

- count=2 device_ms=0.049 self_device_ms=0.049 cpu_ms=0.030
  - `<built-in method cat of type object at 0x7217d43d6c40>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2090): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2427): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanDiffusionWrapper_0`
  - `scope/core/pipelines/krea_realtime_video/blocks/recompute_kv_cache.py(133): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`
  - `scope/core/pipelines/krea_realtime_video/modular_blocks.py(173): __call__`

- count=12 device_ms=0.016 self_device_ms=0.016 cpu_ms=0.172
  - `<built-in method stack of type object at 0x7217d43d6c40>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2090): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2427): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanDiffusionWrapper_0`
  - `scope/core/pipelines/wan2_1/blocks/denoise.py(258): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`
  - `scope/core/pipelines/krea_realtime_video/modular_blocks.py(173): __call__`

- count=1 device_ms=0.015 self_device_ms=0.015 cpu_ms=0.019
  - `<built-in method cat of type object at 0x7217d43d6c40>`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_2`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: ResidualBlock_0`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(666): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Decoder3d_0`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(875): stream_decode`
  - `scope/core/pipelines/wan2_1/vae/wan.py(262): decode_to_pixel`

- count=1 device_ms=0.014 self_device_ms=0.014 cpu_ms=0.017
  - `<built-in method cat of type object at 0x7217d43d6c40>`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_3`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: ResidualBlock_0`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(666): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Decoder3d_0`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(875): stream_decode`
  - `scope/core/pipelines/wan2_1/vae/wan.py(262): decode_to_pixel`

- count=4 device_ms=0.007 self_device_ms=0.007 cpu_ms=0.046
  - `<built-in method cat of type object at 0x7217d43d6c40>`
  - `scope/core/pipelines/krea_realtime_video/modules/model.py(59): sinusoidal_embedding_1d`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2090): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2427): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanDiffusionWrapper_0`
  - `scope/core/pipelines/wan2_1/blocks/denoise.py(258): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`

- count=1 device_ms=0.006 self_device_ms=0.006 cpu_ms=0.021
  - `<built-in method cat of type object at 0x7217d43d6c40>`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_1`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(666): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Decoder3d_0`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(875): stream_decode`
  - `scope/core/pipelines/wan2_1/vae/wan.py(262): decode_to_pixel`
  - `scope/core/pipelines/wan2_1/blocks/decode.py(47): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`
  - `scope/core/pipelines/krea_realtime_video/modular_blocks.py(173): __call__`

- count=3 device_ms=0.004 self_device_ms=0.004 cpu_ms=0.054
  - `<built-in method stack of type object at 0x7217d43d6c40>`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2090): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2427): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanDiffusionWrapper_0`
  - `scope/core/pipelines/krea_realtime_video/blocks/recompute_kv_cache.py(133): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`
  - `scope/core/pipelines/krea_realtime_video/modular_blocks.py(173): __call__`

- count=1 device_ms=0.004 self_device_ms=0.004 cpu_ms=0.077
  - `<built-in method cat of type object at 0x7217d43d6c40>`
  - `scope/core/pipelines/krea_realtime_video/blocks/recompute_kv_cache.py(261): get_context_frames`
  - `scope/core/pipelines/krea_realtime_video/blocks/recompute_kv_cache.py(133): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`
  - `scope/core/pipelines/krea_realtime_video/modular_blocks.py(173): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`
  - `scope/core/pipelines/krea_realtime_video/pipeline.py(256): _generate`
  - `scope/core/pipelines/krea_realtime_video/pipeline.py(250): __call__`
  - `profile_krea_pipeline_ops.py(273): main`
  - `profile_krea_pipeline_ops.py(445): <module>`

- count=1 device_ms=0.001 self_device_ms=0.001 cpu_ms=0.013
  - `<built-in method cat of type object at 0x7217d43d6c40>`
  - `scope/core/pipelines/krea_realtime_video/modules/model.py(59): sinusoidal_embedding_1d`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2090): _forward_inference`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(2427): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanModel_0`
  - `scope/core/pipelines/wan2_1/components/generator.py(314): _call_model`
  - `scope/core/pipelines/wan2_1/components/generator.py(332): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanDiffusionWrapper_0`
  - `scope/core/pipelines/krea_realtime_video/blocks/recompute_kv_cache.py(133): __call__`
  - `torch/utils/_contextlib.py(117): decorate_context`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.023
  - `<built-in method cat of type object at 0x7217d43d6c40>`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_4`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: ResidualBlock_1`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(666): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Decoder3d_0`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(875): stream_decode`
  - `scope/core/pipelines/wan2_1/vae/wan.py(262): decode_to_pixel`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.018
  - `<built-in method cat of type object at 0x7217d43d6c40>`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_5`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: ResidualBlock_1`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(666): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Decoder3d_0`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(875): stream_decode`
  - `scope/core/pipelines/wan2_1/vae/wan.py(262): decode_to_pixel`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.019
  - `<built-in method cat of type object at 0x7217d43d6c40>`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_6`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: ResidualBlock_2`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(666): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Decoder3d_0`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(875): stream_decode`
  - `scope/core/pipelines/wan2_1/vae/wan.py(262): decode_to_pixel`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.018
  - `<built-in method cat of type object at 0x7217d43d6c40>`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_7`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: ResidualBlock_2`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(666): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Decoder3d_0`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(875): stream_decode`
  - `scope/core/pipelines/wan2_1/vae/wan.py(262): decode_to_pixel`

- count=1 device_ms=0.000 self_device_ms=0.000 cpu_ms=0.017
  - `<built-in method cat of type object at 0x7217d43d6c40>`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_8`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: ResidualBlock_3`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(666): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Decoder3d_0`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(875): stream_decode`
  - `scope/core/pipelines/wan2_1/vae/wan.py(262): decode_to_pixel`

## Stack groups: `aten::clone`

- count=5 device_ms=0.133 self_device_ms=0.000 cpu_ms=0.058
  - `<built-in method rms_norm of type object at 0x7217d43d6c40>`
  - `torch/nn/functional.py(2906): rms_norm`
  - `scope/core/pipelines/krea_realtime_video/modules/model.py(237): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanRMSNorm_30`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(913): qkv_fn`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_10`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_10`

- count=5 device_ms=0.132 self_device_ms=0.000 cpu_ms=0.061
  - `<built-in method rms_norm of type object at 0x7217d43d6c40>`
  - `torch/nn/functional.py(2906): rms_norm`
  - `scope/core/pipelines/krea_realtime_video/modules/model.py(237): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanRMSNorm_78`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(913): qkv_fn`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_26`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_26`

- count=5 device_ms=0.132 self_device_ms=0.000 cpu_ms=0.059
  - `<built-in method rms_norm of type object at 0x7217d43d6c40>`
  - `torch/nn/functional.py(2906): rms_norm`
  - `scope/core/pipelines/krea_realtime_video/modules/model.py(237): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanRMSNorm_48`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(913): qkv_fn`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_16`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_16`

- count=5 device_ms=0.132 self_device_ms=0.000 cpu_ms=0.059
  - `<built-in method rms_norm of type object at 0x7217d43d6c40>`
  - `torch/nn/functional.py(2906): rms_norm`
  - `scope/core/pipelines/krea_realtime_video/modules/model.py(237): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanRMSNorm_24`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(913): qkv_fn`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_8`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_8`

- count=5 device_ms=0.132 self_device_ms=0.000 cpu_ms=0.059
  - `<built-in method rms_norm of type object at 0x7217d43d6c40>`
  - `torch/nn/functional.py(2906): rms_norm`
  - `scope/core/pipelines/krea_realtime_video/modules/model.py(237): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanRMSNorm_51`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(913): qkv_fn`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_17`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_17`

- count=5 device_ms=0.131 self_device_ms=0.000 cpu_ms=0.061
  - `<built-in method rms_norm of type object at 0x7217d43d6c40>`
  - `torch/nn/functional.py(2906): rms_norm`
  - `scope/core/pipelines/krea_realtime_video/modules/model.py(237): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanRMSNorm_3`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(913): qkv_fn`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_1`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_1`

- count=5 device_ms=0.131 self_device_ms=0.000 cpu_ms=0.060
  - `<built-in method rms_norm of type object at 0x7217d43d6c40>`
  - `torch/nn/functional.py(2906): rms_norm`
  - `scope/core/pipelines/krea_realtime_video/modules/model.py(237): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanRMSNorm_42`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(913): qkv_fn`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_14`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_14`

- count=5 device_ms=0.131 self_device_ms=0.000 cpu_ms=0.060
  - `<built-in method rms_norm of type object at 0x7217d43d6c40>`
  - `torch/nn/functional.py(2906): rms_norm`
  - `scope/core/pipelines/krea_realtime_video/modules/model.py(237): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanRMSNorm_75`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(913): qkv_fn`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_25`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_25`

- count=5 device_ms=0.131 self_device_ms=0.000 cpu_ms=0.068
  - `<built-in method rms_norm of type object at 0x7217d43d6c40>`
  - `torch/nn/functional.py(2906): rms_norm`
  - `scope/core/pipelines/krea_realtime_video/modules/model.py(237): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanRMSNorm_39`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(913): qkv_fn`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_13`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_13`

- count=5 device_ms=0.131 self_device_ms=0.000 cpu_ms=0.059
  - `<built-in method rms_norm of type object at 0x7217d43d6c40>`
  - `torch/nn/functional.py(2906): rms_norm`
  - `scope/core/pipelines/krea_realtime_video/modules/model.py(237): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanRMSNorm_60`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(913): qkv_fn`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_20`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_20`

- count=5 device_ms=0.131 self_device_ms=0.000 cpu_ms=0.060
  - `<built-in method rms_norm of type object at 0x7217d43d6c40>`
  - `torch/nn/functional.py(2906): rms_norm`
  - `scope/core/pipelines/krea_realtime_video/modules/model.py(237): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanRMSNorm_114`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(913): qkv_fn`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_38`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_38`

- count=5 device_ms=0.131 self_device_ms=0.000 cpu_ms=0.067
  - `<built-in method rms_norm of type object at 0x7217d43d6c40>`
  - `torch/nn/functional.py(2906): rms_norm`
  - `scope/core/pipelines/krea_realtime_video/modules/model.py(237): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanRMSNorm_72`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(913): qkv_fn`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_24`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_24`

- count=5 device_ms=0.131 self_device_ms=0.000 cpu_ms=0.059
  - `<built-in method rms_norm of type object at 0x7217d43d6c40>`
  - `torch/nn/functional.py(2906): rms_norm`
  - `scope/core/pipelines/krea_realtime_video/modules/model.py(237): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanRMSNorm_96`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(913): qkv_fn`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_32`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_32`

- count=5 device_ms=0.131 self_device_ms=0.000 cpu_ms=0.060
  - `<built-in method rms_norm of type object at 0x7217d43d6c40>`
  - `torch/nn/functional.py(2906): rms_norm`
  - `scope/core/pipelines/krea_realtime_video/modules/model.py(237): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanRMSNorm_81`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(913): qkv_fn`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_27`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_27`

- count=5 device_ms=0.131 self_device_ms=0.000 cpu_ms=0.058
  - `<built-in method rms_norm of type object at 0x7217d43d6c40>`
  - `torch/nn/functional.py(2906): rms_norm`
  - `scope/core/pipelines/krea_realtime_video/modules/model.py(237): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: WanRMSNorm_54`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(913): qkv_fn`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(887): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanSelfAttention_18`
  - `scope/core/pipelines/krea_realtime_video/modules/causal_model.py(1534): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalWanAttentionBlock_18`
