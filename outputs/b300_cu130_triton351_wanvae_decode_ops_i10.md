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
- profiled_wall_time_s: `2.726`

## Top ops (self CUDA)
| self_cuda_ms | pct | calls | key |
|---:|---:|---:|---|
| 832.322 | 35.50% | 200 | `aten::slow_conv_dilated3d` |
| 682.157 | 29.10% | 132 | `void at::native::vol2col_kernel<c10::BFloat16>(long, c10::BFloat16 const*, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, c10::BFloat16*)` |
| 126.673 | 5.40% | 1,186 | `Command Buffer Full` |
| 116.009 | 4.95% | 33,426 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1})` |
| 77.432 | 3.30% | 48,510 | `aten::copy_` |
| 74.796 | 3.19% | 72 | `nvjet_tst_256x96_64x5_2x1_2cta_v_badd_NNT` |
| 44.840 | 1.91% | 41 | `nvjet_tst_128x192_64x5_4x1_v_badd_NNT` |
| 42.446 | 1.81% | 600 | `aten::mul` |
| 39.446 | 1.68% | 340 | `aten::cat` |
| 35.479 | 1.51% | 310 | `aten::div` |
| 35.479 | 1.51% | 210 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::DivFunctor<c10::BFloat16> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::DivFunctor<c10::BFloat16> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::DivFunctor<c10::BFloat16> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::DivFunctor<c10::BFloat16> > const&)::{lambda(int)#1})` |
| 33.495 | 1.43% | 203 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1})` |
| 19.814 | 0.85% | 190 | `aten::cudnn_convolution` |
| 18.311 | 0.78% | 460 | `aten::add` |
| 12.352 | 0.53% | 30 | `aten::_upsample_nearest_exact2d` |
| 12.352 | 0.53% | 20 | `void at::native::(anonymous namespace)::upsample_nearest2d_out_frame<c10::BFloat16, &at::native::nearest_neighbor_exact_compute_source_index>(c10::BFloat16 const*, c10::BFloat16*, unsigned long, unsigned long, unsigned long, unsigned long, unsigned long, float, float)` |
| 12.257 | 0.52% | 290 | `aten::silu` |
| 12.257 | 0.52% | 196 | `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::silu_kernel(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#6}::operator()() const::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul> >(int, at::native::(anonymous namespace)::silu_kernel(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#6}::operator()() const::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul>)` |
| 12.129 | 0.52% | 42 | `nvjet_tst_320x192_64x4_2x1_2cta_v_badd_NNT` |
| 12.041 | 0.51% | 12 | `nvjet_tst_512x8_64x3_2x1_v_badd_NNT` |
| 11.823 | 0.50% | 152 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<c10::BFloat16> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<c10::BFloat16> const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<c10::BFloat16> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<c10::BFloat16> const&)::{lambda(int)#1})` |
| 8.951 | 0.38% | 203 | `void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul> >(int, at::native::AUnaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul>)` |
| 8.950 | 0.38% | 203 | `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<c10::BFloat16>, std::array<char*, 2ul> >(int, at::native::CUDAFunctorOnSelf_add<c10::BFloat16>, std::array<char*, 2ul>)` |
| 7.086 | 0.30% | 300 | `aten::linalg_vector_norm` |
| 6.753 | 0.29% | 126 | `void at::native::reduce_kernel<128, 4, at::native::ReduceOp<c10::BFloat16, at::native::NormTwoOps<c10::BFloat16, float, c10::BFloat16>, unsigned int, c10::BFloat16, 4, 4> >(at::native::ReduceOp<c10::BFloat16, at::native::NormTwoOps<c10::BFloat16, float, c10::BFloat16>, unsigned int, c10::BFloat16, 4, 4>)` |
