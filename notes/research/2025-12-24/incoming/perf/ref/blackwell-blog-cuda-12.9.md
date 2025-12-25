Title: NVIDIA Blackwell and NVIDIA CUDA 12.9 Introduce Family-Specific Architecture Features

URL Source: https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/

Published Time: 2025-05-01T22:39:39+00:00

Markdown Content:
One of the earliest architectural design decisions that went into the CUDA platform for NVIDIA GPUs was support for backward compatibility of GPU code. This design means that new GPUs should be able to run programs written for previous GPUs without modification. It’s accomplished by two foundational features of CUDA:

*   NVIDIA [Parallel Thread Execution](https://developer.nvidia.com/blog/understanding-ptx-the-assembly-language-of-cuda-gpu-computing/) (PTX) virtual instruction set architecture (ISA)
*   NVIDIA driver that just-in-time (JIT) compiles PTX code at runtime

PTX is the virtual ISA that targets NVIDIA GPUs. You can think of it like assembly code, but rather than being limited to a specific physical chip hardware architecture, it’s designed to be general enough to be forward compatible with future GPU architectures.

Ever since NVIDIA created the CUDA platform to enable developers to write general purpose programs for GPUs, PTX has been an integral part of CUDA. PTX code built for previous GPUs can be JIT compiled by today’s drivers and run on current GPUs without modification.

Here’s an example. It’s a simple piece of code that prints out the GPU name and compute capability and also prints hello from inside the GPU kernel.

`#include <stdio.h>`

`#include <iostream>`

`__global__``void``printfKernel()`

`{`

```printf``(``">>>>>>>>>>>>>>>>>>>>\n"``);`

```printf``(``"HELLO FROM THREAD %d\n"``, threadIdx.x );`

```printf``(``">>>>>>>>>>>>>>>>>>>>\n"``);`

`}`

`int``main(``int``argc,``char``** argv)`

`{`

```cudaDeviceProp deviceProp;`

```cudaGetDeviceProperties(&deviceProp, 0);`

```std::cout << deviceProp.name << std::endl;`

```std::cout <<``"Compute Capability: "``<< deviceProp.major`

```<<``"."``<< deviceProp.minor << std::endl;`

```printfKernel<<<1,1>>>();`

```cudaDeviceSynchronize();`

```std::cout <<``"End Program"``<< std::endl;`

```return``0;`

`}`

When we compile this code with CUDA 12.8 and run it on our system, which has [NVIDIA RTX 4000 Ada](https://www.nvidia.com/en-us/design-visualization/rtx-4000/), we get the following result:

`$ nvcc -o x.device_info device_info.cu`

`$ .``/x``.device_info`

`NVIDIA RTX 4000 Ada Generation`

`Compute Capability: 8.9`

`>>>>>>>>>>>>>>>>>>>>`

`HELLO FROM THREAD 0`

`>>>>>>>>>>>>>>>>>>>>`

`End Program`

As we didn’t specify any compiler flags to NVCC, it uses the lowest PTX target that is supported by this version of the compiler. You can inspect the executable file to see which PTX architecture and which CUDA binary (cubin) architecture is in your code using `cuobjdump` (the output is snipped for brevity):

`$ cuobjdump x.device_info`

`Fatbin elf code:`

`================`

`arch = sm_52`

`>>> snipped <<<`

`Fatbin ptx code:`

`================`

`arch = sm_52`

`>>> snipped <<<`

You see both `ELF`, which means binary, and `PTX`. When you see an output like this, it means both the cubin and the PTX are embedded in the object file. The architecture is `sm_52`, which is [compute capability](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html) (CC) 5.2. The CC is represented by a number _X.Y_, where _X_ is the major revision number and _Y_ is the minor revision number.

Back to the example. The GPU is CC 8.9, as shown by the printed output when running the code, so how is this code able to run on this GPU?

This is where JIT compilation comes into play. The CUDA driver JIT compiles the PTX to run on the CC 8.9 GPU. As long as your code includes PTX generated from an architecture equal to, or prior to, the architecture of your GPU, your code will run properly.

You can verify this by changing the compiler flags slightly. Add the argument `-gencode arch=compute_75,code=compute_75`. This tells NVCC that you want it to build PTX for your application with version `compute_75` (compute capability 7.5), then put that PTX into the executable and verify with `cuobjdump`. For more information about how NVCC builds PTX and binary code, see Figure 1 in [Understanding PTX, the Assembly Language of CUDA GPU Computing](https://developer.nvidia.com/blog/understanding-ptx-the-assembly-language-of-cuda-gpu-computing/).

You can see that it runs properly.

`$ nvcc -gencode arch=compute_75,code=compute_75 -o x.device_info device_info.cu`

`$ .``/x``.device_info`

`NVIDIA RTX 4000 Ada Generation`

`Compute Capability: 8.9`

`>>>>>>>>>>>>>>>>>>>>`

`HELLO FROM THREAD 0`

`>>>>>>>>>>>>>>>>>>>>`

`End Program`

Now, if you change `code=compute_75` to `code=sm_75`, this tells NVCC to build the same PTX as earlier (`arch=compute_75`). However, rather than leaving the PTX in the executable for JIT compilation, NVCC should compile it into a cubin for `SM_75` and put that cubin into the executable. Again, you can verify with `cuobjdump`. The result is as follows:

`$ nvcc -gencode arch=compute_75,code=sm_75 -o x.device_info device_info.cu`

`$ .``/x``.device_info`

`NVIDIA RTX 4000 Ada Generation`

`Compute Capability: 8.9`

`End Program`

If you look carefully, you see that, `“HELLO FROM THREAD 0”` was not printed. We omitted all the error checking code to make the code example cleaner.

If we had included error checking, as you always should in real code, you’d see that the GPU kernel was not launched, and the error message that is returned is, `“No kernel image is available for execution on the device”`. This means that there was no code for the kernel in the application that is compatible with this CC 8.9 device and so the kernel never launched.

The rule of thumb to follow for all CUDA-capable GPUs (with the exception of Tegra, as they follow [different rules](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#binary-compatibility)) up to and including CC 8.9 is the following:

*   [PTX compatibility](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#ptx-compatibility): Any code with PTX of a certain CC will run on GPUs of that CC and any GPU with a later CC.
*   [Cubin compatibility](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#binary-compatibility): Any code with a cubin of a certain CC will run on GPUs of that CC and any later GPU with that same major capability. For example, a GPU with CC 8.6 can run a cubin that was built for CC 8.0. The reverse is not true. If you build a cubin for CC 8.6, it only runs on CC 8.6 and later, not on 8.0.

Architecture-specific feature set introduced in NVIDIA Hopper[](https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/#architecture-specific_feature_set_introduced_in_nvidia_hopper)
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Beginning with the NVIDIA Hopper architecture (CC 9.0), NVIDIA introduced a small and highly specialized set of features that are called _architecture-specific_, which are only guaranteed to exist on a specific target architecture. A majority of these features are related to the use of Tensor Cores.

To use these features, you must either embed PTX or cubin code in your application using the `compute_90a` flag for PTX or `sm_90a` flag for a cubin in your compilation. When building the architecture-specific target using the `a` suffix, the PTX or cubin code is not forward-compatible with any future GPU architecture.

For example, you compile your CUDA kernel with the following NVCC line:

`$ nvcc -gencode arch=compute_90a,code=sm_90a -c kernel.cu`

In this case, your code only loads and runs on devices of CC 9.0. There is no forward-compatibility for either PTX or a cubin when using the architecture-specific `a` suffix.

Family-specific feature set introduced in NVIDIA Blackwell[](https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/#family-specific_feature_set_introduced_in_nvidia_blackwell)
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Beginning with the NVIDIA Blackwell architecture and CUDA 12.9, a new category of feature is introduced: _family-specific_.

Family-specific features are similar to architecture-specific features, except that they are supported by devices of more than one minor compute capability. All devices within a family share the same major compute capability version. Family-specific features are guaranteed to be available in the same family of GPUs, which includes later GPUs of the same major compute capability and higher minor compute capability.

The family-specific compiler target is similar to the architecture-specific target, but instead of the compiler target using an `a` suffix, you use an `f` suffix.

For more information about which GPUs are in the same family, see the [Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities) and the [CUDA Compute Capability](https://developer.nvidia.com/cuda-gpus) page. For more information about which features are part of the family-specific target, see the table in [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#changes-in-ptx-isa-version-8-8).

For example, you compile your CUDA kernel with the following NVCC line, which invokes the family-specific code generation target:

`$ nvcc -gencode arch=compute_100f,code=sm_100 -c kernel.cu`

In this case, you generate architecture-specific cubin code for the `sm_100f` family and your code will only run on devices with compute capability 10.x.

Currently, this is 10.0 and 10.3 compute capability GPUs. If new GPUs are introduced with a 10.x compute capability, the code would be compatible on those GPUs as well because they would be in the `sm_100f` family. In this case, `code=sm_100` and `code=sm_100f` are aliases of each other and will generate the same cubin that will run on devices in the the sm_100f family.

The way to think about these different feature sets in NVCC is as follows:

*   **No suffix:** Your [PTX or cubin compatibility](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#application-compatibility) is the same as it always has been.
*   **f suffix:** Whether you stop at PTX or generate cubin from that code, that code is compatible to run on GPU devices with the same major compute capability version and with an equal or higher minor compute capability version.
*   **`a` suffix:** The code only runs on GPUs of that specific CC and no others.

Developer guidance[](https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/#developer_guidance)
-----------------------------------------------------------------------------------------------------------------------------------------------------------------

Now that we’ve explained how the architecture and family-specific code targets are built with NVCC, we want to offer recommendations for what you should do when building applications.

In general, you should build code that has the opportunity to run on as many architectures as possible. As long as you are not using architecture or family-specific features, you don’t have to include architecture or family-specific targets in your application, and you can continue to build your code as you’ve always done it. Even if you are using libraries that are using architecture or family-specific features, as long as those libraries are distributed in binary form, they will run properly.

So, when do you need to use family or architecture-specific compiler targets?

As mentioned earlier, these targets are used when features are used that are primarily related to Tensor Cores, and specifically programming Tensor Cores through PTX. If you are writing PTX directly and using family or architecture-specific features, then you must build your code with the `f` or the `a` flags respectively, depending on whether the PTX instructions you’re using are in the `f` set of features or whether they’re only in the `a` set of features.

If you want portability across GPUs of different CC, you must include appropriate guards in your code to ensure that there are fallback code paths available when running on different GPU architectures that don’t have those features. Use the following macros to control the code paths based on family and architecture-specific features you are using:

*   `__CUDA_ARCH_FAMILY_SPECIFIC__`
*   `__CUDA_ARCH_SPECIFIC__`

These macros are defined similarly to `__CUDA_ARCH__`. For more information, see the [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#application-compatibility).

For example, if you are building your application and using a header library such as [CUTLASS](https://github.com/NVIDIA/cutlass), or any library that includes CUTLASS, such as [cuBLASDx](https://developer.nvidia.com/cublasdx-downloads), and you’re running your application on CC 9.0 (NVIDIA Hopper) or later, you should build architecture-specific targets for the GPU devices where your code will run.

CUTLASS is specifically designed for high performance and has special code paths that use architecture-specific features to extract maximum performance. These libraries already have fallback paths internally for full compatibility with other architectures.

In other words, you don’t need to worry about having fallback paths using the macros if you’re using libraries.

Putting it into practice[](https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/#putting_it_into_practice)
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Now that we’ve discussed what the architecture and family-specific targets are, and when to use them, we’ll pull everything together.

### General case[](https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/#general_case)

The first thing to determine is whether your code uses architecture or family-specific features. You’re likely to know if you are using these features because you’re either writing PTX directly, or including header libraries like CUTLASS that do. If this is not the case, which is true for a majority of developers, building your application is just like it always has been.

To provide for the best performance and future compatibility, the typical guidance is to build binary code for each architecture where you know your code will run. This provides the best performance.

You should also embed PTX for the newest architecture available to provide the best future compatibility. For example, you might know your code will run on devices of CC 8.0, 9.0, and 10.0. The following code example shows how you can compile binary for those architectures, and also CC 10.0 PTX for future compatibility.

`$ nvcc -gencode arch=compute_80,code=sm_80`

```-gencode arch=compute_90,code=sm_90`

```-gencode arch=compute_100,code=sm_100`

```-gencode arch=compute_100,code=compute_100 -c kernels.cu`

### Family-specific features[](https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/#family-specific_features)

If you choose to optimize your code using specific features that are not portable across different architectures, then you should first determine whether these [features are in the family-specific feature set](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#changes-in-ptx-isa-version-8-8).

If so, then you can build your targets with the `f` suffix, and you will have compatibility within that family. If you want portability to GPUs outside the family, you must include fallback code paths for any code which uses the family-specific features.

Typically, this is done by guarding the family-specific code through conditional macros in your application. Extending the earlier example and also including family-specific features for CC 10.0, your use of NVCC might look like the following code example:

`$ nvcc -gencode arch=compute_80,code=sm_80`

```-gencode arch=compute_90,code=sm_90`

```-gencode arch=compute_100f,code=sm_100`

```-gencode arch=compute_100,code=compute_100 -c kernels.cu`

This provides your code with the ability to run on devices of CC 8.0, 9.0, and 10.0 with family-specific features for 10.0. Through the embedded PTX, your code will run on future devices as well.

Another possible scenario using family-specific features is when you know that your application must take advantage of these features, and the application is only designed to run on the devices in that family. For example if you’ve designed your code to only use features from the `100f` family, and only intend to run it on devices in this family, the building of the application is similar to the following code example:

`$ nvcc  -gencode arch=compute_100f,code=sm_100 -c kernels.cu`

In this case, your code is portable only across the devices in this family.

### Architecture-specific features[](https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/#architecture-specific_features)

If you’ve determined that family-specific features are not sufficient for your application and you must use features in the architecture-specific feature set, you must build with the `a` flag.

Similar to the case of building with `f`, you must determine what kind of code portability you must build into your application by guarding that code with conditional macros inside your application. For the same portability as the previous example, build your code as in the following code example:

`$ nvcc -gencode arch=compute_80,code=sm_80`

```-gencode arch=compute_90,code=sm_90`

```-gencode arch=compute_100a,code=sm_100a`

```-gencode arch=compute_100,code=compute_100 -c kernels.cu`

Your code will have the same compatibility to run on CC 8.0, CC 9.0, CC 10.0, and beyond,

Similarly to designing applications solely for a specific family, you can choose to design and optimize for a specific architecture as well. If you design and write an application that uses architecture-specific features, and you know it doesn’t have to run on any other GPUs, you can build similar to the following code example:

`$ nvcc  -gencode arch=compute_100a,code=sm_100a -c kernels.cu`

Your application will run on CC 10.0 only, with no compatibility to any other GPUs.

Wrapping up[](https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/#wrapping_up)
---------------------------------------------------------------------------------------------------------------------------------------------------

To summarize, here’s a simple flow for how you should think about building your code:

*   Are you writing PTX directly, or calling a library, like CUTLASS, that does? If not, you shouldn’t need to include `f` or `a` flags. Just build your code like you’ve always done.
*   If you are writing PTX, or including a header-only library that does, you’ll need to determine whether that library uses features that are in the architecture or family-specific feature set and if so, determine which `f` or `a` flag to use. You’ll need to consult that library’s documentation to determine the best way to build for your architecture. For example, the [CUTLASS](https://github.com/NVIDIA/cutlass/blob/main/README.md) build instructions specify using the `a` flag when building for devices of CC 9.0 and 10.0

This post covered a lot of ground to show you how to build code that can use architecture– and family-specific features. We want to be clear that using the `a` and `f` suffixes in your NVCC compile line isn’t a magic optimization technique. To use these features specifically, you must be writing PTX directly or calling a library that does.

Download [CUDA 12.9](https://developer.nvidia.com/cuda-downloads) today to start using family and architecture-specific features in your code on Blackwell.

### Acknowledgments[](https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/#acknowledgments)

_Thanks to the following NVIDIA contributors: Cody Addison, Vyas Venkataraman, Rob Armstrong, Girish Bharambe, and Mridula Prakash._
