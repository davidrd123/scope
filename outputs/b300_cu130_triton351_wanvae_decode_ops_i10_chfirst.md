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
- WANVAE_DECODE_CHANNELS_LAST_3D: `0`
- WANVAE_CONV3D_IMPLICIT_SPATIAL_PADDING: `1`
- WANVAE_UPSAMPLE_FORCE_FP32: `None`
- profiled_wall_time_s: `3.144`

## Top ops (self CUDA)
| self_cuda_ms | pct | calls | key |
|---:|---:|---:|---|
| 1,064.464 | 36.31% | 230 | `aten::slow_conv_dilated3d` |
| 871.872 | 29.74% | 190 | `void at::native::vol2col_kernel<c10::BFloat16>(long, c10::BFloat16 const*, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, c10::BFloat16*)` |
| 161.816 | 5.52% | 50,718 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1})` |
| 112.754 | 3.85% | 60,020 | `aten::copy_` |
| 99.701 | 3.40% | 96 | `nvjet_tst_256x96_64x5_2x1_2cta_v_badd_NNT` |
| 82.342 | 2.81% | 1,254 | `Command Buffer Full` |
| 53.793 | 1.83% | 600 | `aten::mul` |
| 52.511 | 1.79% | 48 | `nvjet_tst_128x192_64x5_4x1_v_badd_NNT` |
| 49.989 | 1.70% | 340 | `aten::cat` |
| 44.971 | 1.53% | 310 | `aten::div` |
| 44.971 | 1.53% | 264 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::DivFunctor<c10::BFloat16> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::DivFunctor<c10::BFloat16> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::DivFunctor<c10::BFloat16> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::DivFunctor<c10::BFloat16> > const&)::{lambda(int)#1})` |
| 42.448 | 1.45% | 255 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1})` |
| 24.026 | 0.82% | 160 | `aten::cudnn_convolution` |
| 23.402 | 0.80% | 460 | `aten::add` |
| 16.412 | 0.56% | 16 | `nvjet_tst_512x8_64x3_2x1_v_badd_NNT` |
| 15.606 | 0.53% | 30 | `aten::_upsample_nearest_exact2d` |
| 15.606 | 0.53% | 25 | `void at::native::(anonymous namespace)::upsample_nearest2d_out_frame<c10::BFloat16, &at::native::nearest_neighbor_exact_compute_source_index>(c10::BFloat16 const*, c10::BFloat16*, unsigned long, unsigned long, unsigned long, unsigned long, unsigned long, float, float)` |
| 15.527 | 0.53% | 290 | `aten::silu` |
| 15.527 | 0.53% | 246 | `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::silu_kernel(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#6}::operator()() const::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul> >(int, at::native::(anonymous namespace)::silu_kernel(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#6}::operator()() const::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul>)` |
| 14.917 | 0.51% | 176 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<c10::BFloat16> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<c10::BFloat16> const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<c10::BFloat16> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<c10::BFloat16> const&)::{lambda(int)#1})` |
| 14.643 | 0.50% | 51 | `nvjet_tst_320x192_64x4_2x1_2cta_v_badd_NNT` |
| 11.345 | 0.39% | 255 | `void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul> >(int, at::native::AUnaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul>)` |
| 11.344 | 0.39% | 255 | `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<c10::BFloat16>, std::array<char*, 2ul> >(int, at::native::CUDAFunctorOnSelf_add<c10::BFloat16>, std::array<char*, 2ul>)` |
| 9.225 | 0.31% | 300 | `aten::linalg_vector_norm` |
| 8.913 | 0.30% | 183 | `void at::native::reduce_kernel<128, 4, at::native::ReduceOp<c10::BFloat16, at::native::NormTwoOps<c10::BFloat16, float, c10::BFloat16>, unsigned int, c10::BFloat16, 4, 4> >(at::native::ReduceOp<c10::BFloat16, at::native::NormTwoOps<c10::BFloat16, float, c10::BFloat16>, unsigned int, c10::BFloat16, 4, 4>)` |
