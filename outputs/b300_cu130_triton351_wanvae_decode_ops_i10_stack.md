# WanVAE stream_decode op profile

## Meta
- torch: `2.9.0+cu130` (cuda `13.0`)
- device: `NVIDIA B300 SXM6 AC` cc=(10, 3)
- height,width: `320x576`
- latent_shape: `[1, 16, 3, 40, 72]`
- dtype: `bf16`
- iters: `10` (pre-iters `10`)
- cudnn.benchmark: `True`
- WANVAE_STREAM_DECODE_MODE: `chunk`
- WANVAE_DECODE_CHANNELS_LAST_3D: `1`
- WANVAE_CONV3D_IMPLICIT_SPATIAL_PADDING: `1`
- WANVAE_UPSAMPLE_FORCE_FP32: `None`
- profiled_wall_time_s: `3.733`

## Top ops (self CUDA)
| self_cuda_ms | pct | calls | key |
|---:|---:|---:|---|
| 832.342 | 35.61% | 200 | `aten::slow_conv_dilated3d` |
| 682.136 | 29.18% | 132 | `void at::native::vol2col_kernel<c10::BFloat16>(long, c10::BFloat16 const*, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, c10::BFloat16*)` |
| 119.829 | 5.13% | 1,171 | `Command Buffer Full` |
| 116.034 | 4.96% | 33,428 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1})` |
| 77.451 | 3.31% | 48,510 | `aten::copy_` |
| 74.785 | 3.20% | 72 | `nvjet_tst_256x96_64x5_2x1_2cta_v_badd_NNT` |
| 44.824 | 1.92% | 41 | `nvjet_tst_128x192_64x5_4x1_v_badd_NNT` |
| 42.444 | 1.82% | 600 | `aten::mul` |
| 39.451 | 1.69% | 340 | `aten::cat` |
| 35.469 | 1.52% | 310 | `aten::div` |
| 35.469 | 1.52% | 210 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::DivFunctor<c10::BFloat16> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::DivFunctor<c10::BFloat16> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::DivFunctor<c10::BFloat16> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::DivFunctor<c10::BFloat16> > const&)::{lambda(int)#1})` |
| 33.492 | 1.43% | 203 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1})` |
| 19.795 | 0.85% | 190 | `aten::cudnn_convolution` |
| 18.313 | 0.78% | 460 | `aten::add` |
| 12.356 | 0.53% | 30 | `aten::_upsample_nearest_exact2d` |
| 12.356 | 0.53% | 20 | `void at::native::(anonymous namespace)::upsample_nearest2d_out_frame<c10::BFloat16, &at::native::nearest_neighbor_exact_compute_source_index>(c10::BFloat16 const*, c10::BFloat16*, unsigned long, unsigned long, unsigned long, unsigned long, unsigned long, float, float)` |
| 12.255 | 0.52% | 290 | `aten::silu` |
| 12.255 | 0.52% | 196 | `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::silu_kernel(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#6}::operator()() const::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul> >(int, at::native::(anonymous namespace)::silu_kernel(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#6}::operator()() const::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul>)` |
| 12.125 | 0.52% | 12 | `nvjet_tst_512x8_64x3_2x1_v_badd_NNT` |
| 12.116 | 0.52% | 42 | `nvjet_tst_320x192_64x4_2x1_2cta_v_badd_NNT` |
| 11.827 | 0.51% | 152 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<c10::BFloat16> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<c10::BFloat16> const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<c10::BFloat16> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<c10::BFloat16> const&)::{lambda(int)#1})` |
| 8.952 | 0.38% | 203 | `void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul> >(int, at::native::AUnaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul>)` |
| 8.951 | 0.38% | 203 | `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<c10::BFloat16>, std::array<char*, 2ul> >(int, at::native::CUDAFunctorOnSelf_add<c10::BFloat16>, std::array<char*, 2ul>)` |
| 7.094 | 0.30% | 300 | `aten::linalg_vector_norm` |
| 6.759 | 0.29% | 126 | `void at::native::reduce_kernel<128, 4, at::native::ReduceOp<c10::BFloat16, at::native::NormTwoOps<c10::BFloat16, float, c10::BFloat16>, unsigned int, c10::BFloat16, 4, 4> >(at::native::ReduceOp<c10::BFloat16, at::native::NormTwoOps<c10::BFloat16, float, c10::BFloat16>, unsigned int, c10::BFloat16, 4, 4>)` |

## Stack groups: `aten::slow_conv_dilated3d`

Filtered totals: device_ms=980.306, calls=200

- count=10 device_ms=100.596 self_device_ms=36.750 cpu_ms=51.723
  - `<built-in method conv3d of type object at 0x7043fa9d6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_26`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=10 device_ms=76.148 self_device_ms=73.639 cpu_ms=71.295
  - `<built-in method conv3d of type object at 0x7043fa9d6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_32`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=10 device_ms=76.143 self_device_ms=73.633 cpu_ms=72.613
  - `<built-in method conv3d of type object at 0x7043fa9d6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_28`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=10 device_ms=76.126 self_device_ms=73.638 cpu_ms=70.224
  - `<built-in method conv3d of type object at 0x7043fa9d6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_30`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=10 device_ms=75.909 self_device_ms=73.672 cpu_ms=8.572
  - `<built-in method conv3d of type object at 0x7043fa9d6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_31`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=10 device_ms=75.875 self_device_ms=73.647 cpu_ms=8.812
  - `<built-in method conv3d of type object at 0x7043fa9d6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_27`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=10 device_ms=75.873 self_device_ms=73.644 cpu_ms=8.604
  - `<built-in method conv3d of type object at 0x7043fa9d6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_29`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=10 device_ms=73.208 self_device_ms=73.136 cpu_ms=0.693
  - `<built-in method conv3d of type object at 0x7043fa9d6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_33`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(666): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=10 device_ms=51.872 self_device_ms=42.871 cpu_ms=17.723
  - `<built-in method conv3d of type object at 0x7043fa9d6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_22`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=10 device_ms=45.785 self_device_ms=42.871 cpu_ms=16.481
  - `<built-in method conv3d of type object at 0x7043fa9d6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_25`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=10 device_ms=45.758 self_device_ms=42.891 cpu_ms=17.440
  - `<built-in method conv3d of type object at 0x7043fa9d6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_21`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=10 device_ms=45.753 self_device_ms=42.872 cpu_ms=16.953
  - `<built-in method conv3d of type object at 0x7043fa9d6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_23`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=10 device_ms=45.735 self_device_ms=42.858 cpu_ms=16.355
  - `<built-in method conv3d of type object at 0x7043fa9d6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_24`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=10 device_ms=20.537 self_device_ms=5.650 cpu_ms=381.846
  - `<built-in method conv3d of type object at 0x7043fa9d6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_14`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=10 device_ms=16.532 self_device_ms=11.170 cpu_ms=277.482
  - `<built-in method conv3d of type object at 0x7043fa9d6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_16`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=10 device_ms=16.353 self_device_ms=11.167 cpu_ms=414.132
  - `<built-in method conv3d of type object at 0x7043fa9d6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_15`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=10 device_ms=15.969 self_device_ms=11.184 cpu_ms=34.466
  - `<built-in method conv3d of type object at 0x7043fa9d6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_19`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=10 device_ms=15.953 self_device_ms=11.175 cpu_ms=32.897
  - `<built-in method conv3d of type object at 0x7043fa9d6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_17`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=10 device_ms=15.942 self_device_ms=11.174 cpu_ms=32.935
  - `<built-in method conv3d of type object at 0x7043fa9d6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_18`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=10 device_ms=14.240 self_device_ms=4.699 cpu_ms=67.480
  - `<built-in method conv3d of type object at 0x7043fa9d6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_20`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(311): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

## Stack groups: `aten::cudnn_convolution`

Filtered totals: device_ms=55.749, calls=190

- count=10 device_ms=43.923 self_device_ms=8.785 cpu_ms=4.543
  - `<built-in method conv2d of type object at 0x7043fa9d6c40>`
  - `torch/nn/modules/conv.py(530): _conv_forward`
  - `torch/nn/modules/conv.py(547): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Conv2d_4`
  - `torch/nn/modules/container.py(245): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Sequential_2`

- count=10 device_ms=6.304 self_device_ms=6.304 cpu_ms=0.369
  - `<built-in method conv2d of type object at 0x7043fa9d6c40>`
  - `torch/nn/modules/conv.py(530): _conv_forward`
  - `torch/nn/modules/conv.py(547): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Conv2d_3`
  - `torch/nn/modules/container.py(245): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Sequential_1`

- count=10 device_ms=0.880 self_device_ms=0.880 cpu_ms=0.273
  - `<built-in method conv2d of type object at 0x7043fa9d6c40>`
  - `torch/nn/modules/conv.py(530): _conv_forward`
  - `torch/nn/modules/conv.py(547): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Conv2d_2`
  - `torch/nn/modules/container.py(245): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Sequential_0`

- count=10 device_ms=0.419 self_device_ms=0.341 cpu_ms=0.247
  - `<built-in method conv3d of type object at 0x7043fa9d6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_4`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=10 device_ms=0.418 self_device_ms=0.341 cpu_ms=0.209
  - `<built-in method conv3d of type object at 0x7043fa9d6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_5`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=10 device_ms=0.418 self_device_ms=0.341 cpu_ms=0.240
  - `<built-in method conv3d of type object at 0x7043fa9d6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_11`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=10 device_ms=0.417 self_device_ms=0.341 cpu_ms=0.218
  - `<built-in method conv3d of type object at 0x7043fa9d6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_9`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=10 device_ms=0.417 self_device_ms=0.342 cpu_ms=0.238
  - `<built-in method conv3d of type object at 0x7043fa9d6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_2`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=10 device_ms=0.417 self_device_ms=0.340 cpu_ms=0.208
  - `<built-in method conv3d of type object at 0x7043fa9d6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_7`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=10 device_ms=0.417 self_device_ms=0.340 cpu_ms=0.204
  - `<built-in method conv3d of type object at 0x7043fa9d6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_6`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=10 device_ms=0.417 self_device_ms=0.341 cpu_ms=0.220
  - `<built-in method conv3d of type object at 0x7043fa9d6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_3`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=10 device_ms=0.416 self_device_ms=0.339 cpu_ms=0.200
  - `<built-in method conv3d of type object at 0x7043fa9d6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_8`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=10 device_ms=0.415 self_device_ms=0.339 cpu_ms=0.260
  - `<built-in method conv3d of type object at 0x7043fa9d6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_10`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=10 device_ms=0.127 self_device_ms=0.099 cpu_ms=0.247
  - `<built-in method conv3d of type object at 0x7043fa9d6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_12`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(311): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=10 device_ms=0.111 self_device_ms=0.089 cpu_ms=0.266
  - `<built-in method conv3d of type object at 0x7043fa9d6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_1`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(666): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=10 device_ms=0.108 self_device_ms=0.108 cpu_ms=0.193
  - `<built-in method conv3d of type object at 0x7043fa9d6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_13`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=10 device_ms=0.065 self_device_ms=0.065 cpu_ms=0.200
  - `<built-in method conv2d of type object at 0x7043fa9d6c40>`
  - `torch/nn/modules/conv.py(530): _conv_forward`
  - `torch/nn/modules/conv.py(547): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Conv2d_0`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(465): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: AttentionBlock_0`

- count=10 device_ms=0.034 self_device_ms=0.034 cpu_ms=0.150
  - `<built-in method conv2d of type object at 0x7043fa9d6c40>`
  - `torch/nn/modules/conv.py(530): _conv_forward`
  - `torch/nn/modules/conv.py(547): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Conv2d_1`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(465): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: AttentionBlock_0`

- count=10 device_ms=0.028 self_device_ms=0.028 cpu_ms=0.219
  - `<built-in method conv3d of type object at 0x7043fa9d6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_0`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(875): stream_decode`
  - `profile_wanvae_decode_ops.py(255): main`

## Stack groups: `aten::copy_`

Filtered totals: device_ms=79.727, calls=48510

- count=7680 device_ms=9.541 self_device_ms=9.541 cpu_ms=31.403
  - `<built-in method conv3d of type object at 0x7043fa9d6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_20`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(311): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=3840 device_ms=5.362 self_device_ms=4.779 cpu_ms=259.320
  - `<built-in method conv3d of type object at 0x7043fa9d6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_16`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=3840 device_ms=5.202 self_device_ms=4.766 cpu_ms=361.630
  - `<built-in method conv3d of type object at 0x7043fa9d6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_14`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=3840 device_ms=5.186 self_device_ms=4.759 cpu_ms=395.588
  - `<built-in method conv3d of type object at 0x7043fa9d6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_15`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=3840 device_ms=4.786 self_device_ms=4.784 cpu_ms=16.160
  - `<built-in method conv3d of type object at 0x7043fa9d6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_19`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=3840 device_ms=4.778 self_device_ms=4.778 cpu_ms=15.107
  - `<built-in method conv3d of type object at 0x7043fa9d6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_17`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=3840 device_ms=4.768 self_device_ms=4.768 cpu_ms=15.083
  - `<built-in method conv3d of type object at 0x7043fa9d6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_18`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=1920 device_ms=2.915 self_device_ms=2.915 cpu_ms=7.486
  - `<built-in method conv3d of type object at 0x7043fa9d6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_25`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=1920 device_ms=2.880 self_device_ms=2.880 cpu_ms=7.578
  - `<built-in method conv3d of type object at 0x7043fa9d6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_23`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=1920 device_ms=2.876 self_device_ms=2.876 cpu_ms=7.386
  - `<built-in method conv3d of type object at 0x7043fa9d6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_24`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=1920 device_ms=2.872 self_device_ms=2.872 cpu_ms=7.764
  - `<built-in method conv3d of type object at 0x7043fa9d6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_22`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=1920 device_ms=2.867 self_device_ms=2.867 cpu_ms=7.850
  - `<built-in method conv3d of type object at 0x7043fa9d6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_21`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=1920 device_ms=2.604 self_device_ms=2.604 cpu_ms=8.470
  - `<built-in method conv3d of type object at 0x7043fa9d6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_26`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=960 device_ms=2.511 self_device_ms=2.232 cpu_ms=67.829
  - `<built-in method conv3d of type object at 0x7043fa9d6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_28`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=960 device_ms=2.509 self_device_ms=2.230 cpu_ms=66.283
  - `<built-in method conv3d of type object at 0x7043fa9d6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_32`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=960 device_ms=2.487 self_device_ms=2.231 cpu_ms=65.360
  - `<built-in method conv3d of type object at 0x7043fa9d6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_30`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=960 device_ms=2.237 self_device_ms=2.237 cpu_ms=3.794
  - `<built-in method conv3d of type object at 0x7043fa9d6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_31`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=960 device_ms=2.229 self_device_ms=2.229 cpu_ms=3.849
  - `<built-in method conv3d of type object at 0x7043fa9d6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_29`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=960 device_ms=2.229 self_device_ms=2.229 cpu_ms=3.841
  - `<built-in method conv3d of type object at 0x7043fa9d6c40>`
  - `torch/nn/modules/conv.py(699): _conv_forward`
  - `torch/nn/modules/conv.py(716): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(178): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: CausalConv3d_27`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=10 device_ms=1.468 self_device_ms=1.468 cpu_ms=0.063
  - `<built-in function _upsample_nearest_exact2d>`
  - `torch/nn/functional.py(4530): interpolate`
  - `torch/nn/modules/upsampling.py(171): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(220): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Upsample_2`
  - `torch/nn/modules/container.py(245): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=20 device_ms=0.953 self_device_ms=0.953 cpu_ms=0.096
  - `<built-in method clone of Tensor object at 0x7040def2d130>`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: ResidualBlock_12`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(666): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Decoder3d_0`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(875): stream_decode`

- count=20 device_ms=0.953 self_device_ms=0.953 cpu_ms=0.096
  - `<built-in method clone of Tensor object at 0x7040def2d130>`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: ResidualBlock_13`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(666): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Decoder3d_0`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(875): stream_decode`

- count=10 device_ms=0.853 self_device_ms=0.853 cpu_ms=0.057
  - `<built-in function _upsample_nearest_exact2d>`
  - `torch/nn/functional.py(4530): interpolate`
  - `torch/nn/modules/upsampling.py(171): forward`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(220): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Upsample_1`
  - `torch/nn/modules/container.py(245): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`

- count=20 device_ms=0.616 self_device_ms=0.616 cpu_ms=0.165
  - `<built-in method clone of Tensor object at 0x7040def2d130>`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: ResidualBlock_11`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(666): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Decoder3d_0`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(875): stream_decode`

- count=20 device_ms=0.567 self_device_ms=0.567 cpu_ms=0.091
  - `<built-in method clone of Tensor object at 0x7040def2d130>`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(423): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: ResidualBlock_10`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(666): forward`
  - `torch/nn/modules/module.py(1779): _call_impl`
  - `nn.Module: Decoder3d_0`
  - `scope/core/pipelines/wan2_1/vae/modules/vae.py(875): stream_decode`
