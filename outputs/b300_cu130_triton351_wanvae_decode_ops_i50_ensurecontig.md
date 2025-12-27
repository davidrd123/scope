# WanVAE stream_decode op profile

## Meta
- torch: `2.9.0+cu130` (cuda `13.0`)
- device: `NVIDIA B300 SXM6 AC` cc=(10, 3)
- height,width: `320x576`
- latent_shape: `[1, 16, 3, 40, 72]`
- dtype: `bf16`
- iters: `50` (pre-iters `10`)
- cudnn.benchmark: `True`
- WANVAE_STREAM_DECODE_MODE: `chunk`
- WANVAE_DECODE_CHANNELS_LAST_3D: `1`
- WANVAE_CONV3D_IMPLICIT_SPATIAL_PADDING: `1`
- WANVAE_UPSAMPLE_FORCE_FP32: `None`
- profiled_wall_time_s: `2.681`

## Top ops (self CUDA)
| self_cuda_ms | pct | calls | key |
|---:|---:|---:|---|
| 576.773 | 20.27% | 4,325 | `Command Buffer Full` |
| 369.788 | 13.00% | 1,950 | `aten::cudnn_convolution` |
| 169.081 | 5.94% | 161 | `cutlass3x_sm100_tensorop_s32x256x16_conv3d_fprop_weight_stationary_nq_2d_tiled_bf16_bf16_f32_void_bf16_void_t3xr3xs3_32x256x64_dyn_cga_ndhwc_ndhwc_ndhwc_1sm` |
| 152.291 | 5.35% | 3,000 | `aten::mul` |
| 131.063 | 4.61% | 942 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<c10::BFloat16> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<c10::BFloat16> const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<c10::BFloat16> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<c10::BFloat16> const&)::{lambda(int)#1})` |
| 131.000 | 4.60% | 1,950 | `aten::add_` |
| 127.772 | 4.49% | 1,550 | `aten::div` |
| 127.772 | 4.49% | 729 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::DivFunctor<c10::BFloat16> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::DivFunctor<c10::BFloat16> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::DivFunctor<c10::BFloat16> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::DivFunctor<c10::BFloat16> > const&)::{lambda(int)#1})` |
| 120.209 | 4.23% | 208 | `cutlass3x_sm100_tensorop_s256x128x16implicit_gemm_fprop_bf16_bf16_f32_void_bf16_f32_256x128x64_dyn_cga_ndhwc_ndhwc_ndhwc_align8_2sm_relu_valpha` |
| 119.878 | 4.21% | 705 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1})` |
| 100.474 | 3.53% | 1,500 | `aten::linalg_vector_norm` |
| 100.474 | 3.53% | 705 | `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<c10::BFloat16, at::native::NormTwoOps<c10::BFloat16, float, c10::BFloat16>, unsigned int, c10::BFloat16, 4, 4> >(at::native::ReduceOp<c10::BFloat16, at::native::NormTwoOps<c10::BFloat16, float, c10::BFloat16>, unsigned int, c10::BFloat16, 4, 4>)` |
| 71.819 | 2.52% | 150 | `aten::_upsample_nearest_exact2d` |
| 55.683 | 1.96% | 23 | `void at::native::(anonymous namespace)::upsample_nearest2d_nhwc_out_frame<c10::BFloat16, &at::native::nearest_neighbor_exact_compute_source_index>(c10::BFloat16 const*, c10::BFloat16*, unsigned long, unsigned long, unsigned long, unsigned long, unsigned long, float, float, unsigned long)` |
| 54.955 | 1.93% | 406 | `cutlass3x_sm100_tensorop_s128x128x16implicit_gemm_fprop_bf16_bf16_f32_void_bf16_f32_128x128x64_dyn_cga_ndhwc_ndhwc_ndhwc_align8_2sm_relu_valpha` |
| 54.593 | 1.92% | 2,300 | `aten::add` |
| 48.140 | 1.69% | 2,256 | `Memcpy DtoD (Device -> Device)` |
| 47.559 | 1.67% | 1,700 | `aten::cat` |
| 44.326 | 1.56% | 1,450 | `aten::silu` |
| 44.326 | 1.56% | 681 | `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::silu_kernel(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#6}::operator()() const::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul> >(int, at::native::(anonymous namespace)::silu_kernel(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#6}::operator()() const::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul>)` |
| 32.427 | 1.14% | 705 | `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<c10::BFloat16>, std::array<char*, 2ul> >(int, at::native::CUDAFunctorOnSelf_add<c10::BFloat16>, std::array<char*, 2ul>)` |
| 32.413 | 1.14% | 705 | `void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul> >(int, at::native::AUnaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul>)` |
| 26.194 | 0.92% | 3,450 | `aten::copy_` |
| 25.612 | 0.90% | 963 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1})` |
| 22.103 | 0.78% | 353 | `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul>)` |
