# Multi‑GPU Scaling for KREA Realtime Pipeline

## Summary

**Goal:** Scale Krea Realtime’s text-to-video diffusion pipeline beyond single-GPU limits by leveraging multiple GPUs in parallel. Current single-GPU performance (\~11 FPS on NVIDIA B200 with 4 steps[\[1\]](https://www.reddit.com/r/StableDiffusion/comments/1ocr3re/krea_realtime_14b_an_opensource_realtime_ai_video/#:~:text=,40GB%2B%20VRAM%20recommended)) can be improved by distributing work. The StreamDiffusionV2 project has already shown near-linear speedups using multi-GPU inference – **achieving \~58 FPS on a 14B model with 4× H100 GPUs**. Adapting similar multi-GPU strategies (primarily pipeline parallelism across the transformer/UNet) could substantially boost Krea Realtime’s throughput without retraining models. This report outlines known approaches, evaluates their feasibility for Krea, and suggests an investigation plan. We focus on splitting the **Transformer** stage (the DiT model) across GPUs, while also considering the **VAE** and scheduler if relevant. We highlight which parts of the codebase assume a single device and would need modification, and incorporate insights from external implementations like StreamDiffusionV2. Practical considerations – such as communication overhead, synchronization, and memory limits – are emphasized over abstract theory.

## What We Know

### StreamDiffusionV2’s Multi‑GPU Approach

StreamDiffusionV2 is a recent system designed for real-time video diffusion that **scales across multiple GPUs**. Key strategies from their paper and code include:

* **Pipeline Parallelism:** They split the diffusion model’s layers (a DiT transformer with \~30 blocks) across GPUs, forming a pipeline. For example, with 2 GPUs, the first half of the layers run on GPU0 and the second half on GPU1. This yielded near-linear FPS scaling in their tests. The pipeline stages operate concurrently on different video frames/denoising steps to keep all GPUs busy. This is a proven approach in their system.

* **Tensor Model Parallelism:** Not used – splitting individual neural layers across GPUs (e.g. splitting matrix multiplies) was deemed impractical due to huge intermediate activations that would need to be communicated each step, negating any speed gain. In real-time streaming, small batch sizes and sequential dependencies make this form of parallelism communication-bound (thus not viable in StreamDiffusionV2).

* **Sequence/Temporal Parallelism:** Not used for core inference – running different diffusion time steps or video frames on different GPUs in parallel was avoided due to synchronization complexity and unpredictable latency jitter. Instead, StreamDiffusionV2’s parallelism is structured (via pipeline parallel) to guarantee per-frame deadlines.

**Performance:** Without specialized optimizations (no TensorRT, no quantization), StreamDiffusionV2 achieves **58.28 FPS with a 14B model on 4× H100** (and 64.5 FPS with a 1.3B model) while keeping first-frame latency around 0.5 s. This confirms multi-GPU inference can drastically improve throughput for large T2V models.

**Key mechanisms from StreamDiffusionV2:**  
\- **SLO-aware batching scheduler** – Merges requests carefully to meet strict real-time latency Service-Level Objectives. In our context, this means the system decides how many diffusion steps or frames to process together without violating the per-frame deadline.  
\- **Dynamic block scheduler** – Adjusts the pipeline partitioning on the fly. StreamDiffusionV2 can redistribute work across GPUs at runtime based on measured load. (Their code has an optional \--schedule\_block mode to enable this for optimal throughput.) This helps handle variability in content or if GPUs have different capabilities.  
\- **VAE on a separate device** – They offload the VAE encoder/decoder to a different GPU than the UNet/transformer. This prevents VAE work (which is around image encoding/decoding) from competing for the same GPU memory and compute as the core diffusion model. By doing VAE encode upfront on GPU1 while GPU0 runs the first diffusion steps, and VAE decode on GPU1 while GPU0 starts the next frame, they achieve overlap. This architectural choice isn’t explicitly stated in the paper, but the pipeline description (VAE encode → diffusion → VAE decode) and known practice suggest dedicating one GPU to VAE can improve throughput.

*How they use multi-GPU:* StreamDiffusionV2 launches with PyTorch’s distributed launcher. For example, running torchrun \--nproc\_per\_node=2 ... inference\_pipe.py spawns two processes each pinned to a GPU. Each process constructs part of the pipeline (e.g., rank 0 might own the first half of the model and rank 1 the second half). They likely utilize torch.distributed to communicate intermediate activations (e.g., the latents after half the layers). The code is organized so that on each iteration, GPU0 computes its part then sends data to GPU1, which computes and sends back, etc., in a streaming fashion. This concurrent orchestration across denoising steps and layers is how near-linear scaling is achieved.

### Current KREA Pipeline (Single-GPU Assumptions)

Krea Realtime’s pipeline is currently written for a single GPU execution. All components – the text encoder, the diffusion model (“generator”), and the VAE – are moved to **one device** at initialization. For example, in src/scope/core/pipelines/krea\_realtime\_video/pipeline.py (feature/stream-recording branch), we see code like:

generator \= generator.to(device=device, dtype=dtype)  
text\_encoder \= text\_encoder.to(device=device)  
vae \= vae.to(device=device, dtype=dtype)

This indicates the entire model and supporting modules reside on the same device. There’s no logic for dividing work across multiple torch.device instances or across multiple processes. The **device placement is hard-coded to a single GPU**, and PyTorch’s distributed utilities (like DistributedDataParallel or pipeline parallel modules) are not used.

**Implication:** Out-of-the-box, the Krea pipeline will only use one GPU even if multiple are available. In fact, documentation suggests a single very large GPU is needed – e.g. a 40 GB VRAM card is recommended for high resolutions[\[2\]](https://github.com/daydreamlive/scope/blob/d42c620648aa9b340e07ab42614b3d9de28eb9ed/src/scope/core/pipelines/krea_realtime_video/docs/usage.md#L9-L17). This simplifies the code but means any multi-GPU support must be added almost from scratch.

**Related internal resources:** There are some low-level primitives in the repo hinting at multi-GPU or multi-node support, but they are not integrated at the pipeline level. Under vendored/cutlass-cute/python/CuTeDSL/distributed/, there are scripts like distributed\_gemm\_all\_reduce\_blackwell.py and distributed\_gemm\_reduce\_scatter\_blackwell.py. These appear to be experimental CUDA kernels (likely targeting NVIDIA Blackwell architecture) for distributed matrix multiplies (using all-reduce or reduce-scatter across GPUs). However, these operate at the **tensor operation level** (GEMM \= matrix multiply) and would require significant engineering to use for the diffusion model’s layers. They are not currently invoked by the pipeline code. In short, the codebase has some building blocks for distributed computation, but no pipeline orchestration that actually splits the model or data across GPUs yet.

## What We Don’t Know (Open Questions)

Despite knowing the general strategies, several unknowns need investigation before proceeding:

1. **Partitioning strategy for the DiT (Transformer) model:** How exactly does StreamDiffusionV2 partition the 14B DiT across GPUs? Is it a simple split at a certain transformer block (e.g., first N layers on GPU0, next N layers on GPU1), or something more granular (e.g., splitting each denoising step’s work)? We need to confirm this from their implementation. For Krea’s model (distilled Wan2.1 14B), which has \~40 transformer layers, an obvious split is 20 layers per GPU for 2 GPUs. But details like attention coupling between layers might affect where the cut can occur cleanly.

2. **Communication overhead and patterns:** In any pipeline parallel setup, activations (latent feature maps) must be passed between devices at partition boundaries every diffusion step. We need to quantify these tensors – e.g., if the latent is 16×320×576 (C×H×W) in half-precision, moving it across PCIe/NVLink every step might cost a few milliseconds. Is that negligible relative to compute? Also, how often do we need to sync or wait? StreamDiffusionV2’s “block scheduler” suggests they measure and adapt to these costs. We should identify what data is sent (probably latents or attention key/value updates) and ensure it won’t bottleneck us. **Async scheduling** (overlapping communication with computation) will be key if communication is significant.

3. **Stateful streaming & KV cache distribution:** Krea’s pipeline likely maintains a **rolling KV cache** for the transformer (similar to a causal LLM or the “sink-token–guided rolling KV cache” mentioned in StreamDiffusionV2). This cache holds key/value tensors for attention to avoid recomputing them for each new frame. In multi-GPU context, how do we manage this? If the transformer is split, each GPU might store the KV for the layers it owns. We need to verify if Krea’s implementation already supports a distributed KV or if it assumes a single memory space. Ensuring that as new frames are generated, the KV cache (which is sequentially updated) remains consistent across GPUs is non-trivial. This could be a complicating factor that StreamDiffusionV2 had to solve (their paper explicitly mentions a rolling KV cache design).

4. **Latency vs. Throughput Trade-off:** Pipeline parallelism can increase throughput (FPS) but typically adds latency for the first output (pipeline fill latency). In streaming, per-frame latency is critical (we can’t buffer too many frames ahead without introducing lag). StreamDiffusionV2 reports \~0.5 s time-to-first-frame at 4×H100, which is acceptable, but if we add more GPUs or slower interconnect, this might grow. We need to quantify how a multi-GPU pipeline would affect frame latency for Krea. There’s a balance: for higher quality (more steps) modes, throughput matters more; for ultra-low-latency mode, maybe we’d disable multi-GPU. We should measure how much *overhead* pipeline parallel introduces (e.g., if using 2 GPUs gives 2× FPS but adds \~1/2 second of lag, is it worth it for our use case?).

5. **Heterogeneous GPU support:** StreamDiffusionV2 is designed to “scale seamlessly across heterogeneous GPU environments” – meaning it can use different GPU models together (e.g., one high-memory, one high-speed) and schedule work appropriately. Can we do the same? In practice, mixing GPUs (say an A100 \+ an RTX 5090\) might complicate synchronization (different compute speeds, memory sizes). Does PyTorch pipeline parallel handle flow control when one stage is slower? It likely will stall the faster GPU unless the block scheduler reallocates work. This is an advanced scenario – for now, we might assume identical GPUs, but it’s worth noting if the solution can extend to hetero-GPU rigs (since our users might have one strong GPU and one weaker GPU, for example).

## Potential Approaches for Multi-GPU Inference

We outline four possible approaches, from simplest to most complex, to utilize multiple GPUs in Krea Realtime:

### **A. VAE Offload (Two-Device Split)** – *Simplest incremental improvement*

Run the **VAE encoder/decoder** on a different GPU from the main diffusion model. In the current pipeline, the VAE compresses each incoming frame to latent space before diffusion, and decompresses the output latents to an image frame. This typically takes \~20–25% of the total pipeline time (based on internal profiling on a B300 GPU). Offloading it could free up time on the main GPU.

* **Pros:** Very straightforward to implement. We can send the video frames to GPU1 for VAE encoding while GPU0 runs the text encoder and diffusion model. Similarly, when the diffusion step finishes, GPU0 can start the next frame’s diffusion while GPU1 decodes the previous output. This parallelizes VAE work with UNet work. Memory-wise, each GPU now holds only either the VAE or the UNet, which is easier on VRAM.

* **Cons:** Only helps if VAE was a significant bottleneck. Scaling is limited – effectively uses at most 2 GPUs (one for UNet, one for VAE). If we have more GPUs, this approach won’t utilize them well. Also, transferring the latent tensors between GPUs each frame is an added overhead (latents are 4× smaller than full images, but still non-trivial size, e.g., 16 channels × 320×576 for each frame). We must copy latents over PCIe/NVLink twice per frame (after encode and before decode). If NVLink is available, that’s manageable; on PCIe it could be \~5–10 ms overhead which slightly impacts latency.

* **Complexity:** **Low.** We mostly need to modify the pipeline to accept two device arguments (one for VAE, one for the rest) and insert .to(device1) for VAE. PyTorch’s built-in ops can handle the transfers. This is a good first experiment to see some multi-GPU gain with minimal risk.

### **B. Pipeline-Parallel Transformer** – *Split the UNet/DiT across GPUs (StreamDiffusionV2 style)*

This is the core approach used by StreamDiffusionV2: divide the transformer’s layers into stages on different GPUs and run them in parallel on a stream of inputs. For instance, with 2 GPUs, the first N layers (e.g., 20\) are on GPU0 and the remaining N layers on GPU1. During inference, while GPU1 is processing the later layers of frame *t*, GPU0 can start processing the first layers of frame *t+1* (pipelining across time). With more GPUs, we could split into more stages (e.g., 4 GPUs with \~10 layers each).

* **Pros:** **Proven effective for increasing throughput.** As seen in StreamDiffusionV2, this can yield nearly linear speedups with more GPUs. For large models like 14B, it makes good use of available memory (each GPU holds a fraction of the model’s weights). This approach directly targets the transformer, which is the main compute-heavy part of the pipeline. If properly balanced, each GPU does roughly equal work every step, keeping all devices busy.

* **Cons:** **Increased complexity in code and runtime.** We will need to introduce inter-process (or at least inter-device) communication every diffusion step for handing off latents from one stage to the next. Implementing this could be done via PyTorch’s distributed RPC or by spawning separate processes (as torchrun does) and using DistributedDataParallel with pipeline partitioning. Either way, our code needs refactoring. Also, pipeline parallelism introduces **startup latency**: the first frame has to go through GPU0 then GPU1, etc., before any output emerges. This “fill pipeline” latency is essentially one extra frame of delay per additional GPU stage. For example, 2-stage pipeline means the first output comes after both stages have processed once. As noted, 0.5 s first-frame latency on 4 GPUs was observed – we have to ensure this is acceptable. Another con: **state synchronization.** If the diffusion model uses any feedback between blocks (e.g., global conditioning), splitting could be tricky. However, most diffusion UNets are feed-forward per step, so splitting at a block boundary should be fine. We must also handle the **KV cache**: each GPU could maintain KV for its layers, which means before generating the next frame, GPU0 and GPU1 each have their own portion of the past keys/values. We’ll need to coordinate to ensure the pipeline uses the updated state correctly (probably the pipeline code will just handle it if each stage’s state stays on that GPU). Lastly, implementing dynamic load balancing (like moving layers between GPUs if one is slower) is complex – initially we can static-partition equally and assume identical GPUs.

* **Complexity:** **Medium to High.** We need to modify the pipeline structure significantly. We may introduce a new class or script (similar to StreamDiffusionV2’s inference\_pipe.py) that launches processes for each partition. Leveraging PyTorch’s existing **Pipeline Parallel** API (available in torch.distributed.pipeline.sync) might help manage micro-batches and back-pressure automatically. Alternatively, a manual implementation using point-to-point CUDA streams or even just synchronous sends might suffice. Either way, testing and debugging distributed code can be challenging. This approach should be attempted after simpler ones prove insufficient.

### **C. Temporal Parallelism (Frame/Chunk Parallel)** – *Parallelize across time dimension*

In a streaming scenario, we could assign different GPUs to work on different segments of the video sequence simultaneously. For example, in a two-GPU setup: while GPU0 is generating frames 1–5, GPU1 could start working on frames 6–10. This is akin to a double-buffering of the video timeline. One could also imagine GPU0 handling odd-numbered frames and GPU1 even-numbered frames, though that might break temporal continuity. A more sensible variant: split the timeline into chunks (each chunk still sequential internally) and generate chunks in parallel, then stitch together.

* **Pros:** This approach treats *each GPU’s work as an independent sub-problem*, which can be easier to implement (no layer-wise partitioning, each GPU runs a full copy of the model on a subset of frames). It could be a natural extension if we want to generate multiple videos or multiple segments concurrently. It might also bypass some of the KV cache complexity, since each GPU could maintain its own cache for the frames it handles (though that means duplicate cache state and potential discontinuities at boundaries).

* **Cons:** For a single continuous video stream, splitting the timeline is problematic – diffusion is an autoregressive process where each frame depends on the previous frame’s output. If GPU1 starts generating frame 6 without frames 1–5 fully decided, you lose causal consistency. We might partition by *blocks of frames* (e.g., GPU0 does frames 1–30, GPU1 does 31–60 in parallel), but then frame 31 won’t condition on frame 30 properly – unless we wait, defeating parallelism. Another idea is one GPU handles the *current* frame’s diffusion steps while another prepares the *next* frame’s initial noise or does some other precomputation. However, there’s not much precomputation possible because each frame’s generation strictly follows from the previous. Thus, true temporal parallelism in a single video stream likely undermines coherence. It’s better suited if we had multiple independent video streams to serve (data parallel inference), which is not our case here.

* **Complexity:** **High and likely not worth it** for one stream. It’s an unexplored approach for diffusion streams because of the sequential dependency. We list it for completeness, but we probably won’t pursue this for real-time video generation (the focus will be on spatial/model parallelism).

### **D. Patch-Based Parallelism (DistriFusion-style)** – *Spatially split the frame*

A recently proposed method called **DistriFusion** (CVPR 2024\) accelerates diffusion inference by slicing high-res images into patches and processing patches on different GPUs in parallel. Essentially, each GPU runs the full model on a portion of the image (e.g., top-half vs bottom-half), and they exchange boundary information to maintain coherence. For video, one could imagine dividing each frame into e.g. left and right halves, generating both halves concurrently, and then combining.

* **Pros:** This directly increases throughput for higher resolution outputs, since each GPU handles fewer pixels. If we ever push beyond our current resolution (320×576 or 480×832), patch-based scaling might be beneficial. It’s “training-free” and can work with existing models. Also, for extremely large diffusion models where even a single layer barely fits on one GPU, splitting activations spatially could be a way to fit the model (though our case is more about speed than memory).

* **Cons:** Diffusion models (especially spatial UNets) have interactions across the whole image via attention and convolution. Splitting into patches means at patch borders you either have to communicate or you get artifacts. DistriFusion’s solution (displaced patch method) reuses feature maps from previous steps to provide context, allowing **asynchronous communication** that overlaps with compute. This is clever but complex. Also, our resolutions might not be high enough to warrant patch splits – at 576px width, splitting into two 288px patches might yield sub-linear speedup once you account for communication and lost efficiency at borders. Additionally, DistriFusion assumed *image* generation (all steps parallel in space); for *video* the temporal consistency adds another layer of complexity. We’d also require NVLink or very fast interconnect to share patch boundaries frequently (the authors note NVLink was important for their multi-3090 setup).

* **Complexity:** **High.** Implementing patch-level pipeline parallelism would involve rewriting the model’s forward pass to condition on ghost regions or to stitch outputs. Unless we target very high resolutions that single GPUs cannot handle at required FPS, this is likely not our first choice. It’s an interesting reference if we hit resolution or quality ceilings in the future.

## Suggested Investigation Path

To proceed pragmatically, we propose a phased approach:

**Phase 1: Measurement (Profile the Baseline)**  
Before writing any multi-GPU code, gather precise performance data on the existing single-GPU pipeline. Use a high-end GPU (e.g., H100 or RTX 6000 Ada) at various resolutions (e.g., 320×576 vs 480×832) and frame rates to see where time is spent per frame. Key things to measure: \- Time spent in **VAE encode**, **UNet (transformer)** by denoising step, and **VAE decode**. This will tell us how much offloading VAE might help (if VAE is 5 ms of a 40 ms frame, it’s 12.5%; if it’s 15 ms, it’s much bigger). The mention that VAE is \~25% on B300 suggests a notable chunk – worth offloading.  
\- GPU utilization during streaming – is the GPU fully busy or are there idle gaps? Idle gaps could indicate where a second GPU could be inserted to overlap work (for example, if GPU is waiting on CPU or I/O at some stage).  
\- Memory usage – ensure that splitting model across two GPUs will indeed allow us to double up frames or steps. If currently a 40GB GPU is 90% utilized, putting half the model on one 24GB GPU and half on another might be possible.  
\- Communication estimates – e.g., record the size of the latent tensor (generator output) per step. This will be the data sent between GPUs in pipeline parallel. Calculate how many bytes per second that would be at target FPS and see if PCIe (\~12 GB/s) or NVLink (\~50 GB/s) can handle it.

**Phase 2: Reference Study (Learn from Prior Art)**  
Dive deeper into StreamDiffusionV2’s code to answer the unknowns: \- How do they implement the multi-GPU pipeline? Identify the parts in their inference\_pipe.py (multi-GPU entry point) where the model is split. Check if they use PyTorch’s built-in pipeline parallel or a custom approach. For example, do they create the model and then manually move some layers to cuda:0 and others to cuda:1? Or do they instantiate separate model objects per process? Understanding this will guide our design (e.g., whether to use multi-process with torchrun or try multi-device in one process – the former is more likely).  
\- Examine the **block scheduler** implementation if present. See how it monitors the throughput of each stage and redistributes blocks. This might be complex to adopt immediately, but we can at least design our code to allow manual re-partitioning (like a config that says how many layers on each GPU) and maybe later automate it.  
\- Look at how they handle **KV cache and synchronization**. If their pipeline keeps a rolling cache, see if each GPU process keeps local cache slices. Also check if any modifications to the model were required (e.g., did they replace any torch.nn.DataParallel usage with something else?).  
\- Review other systems: e.g., vLLM’s docs on parallelism to see how they combine pipeline and tensor parallel for large language models[\[3\]](https://docs.vllm.ai/en/v0.8.0/serving/distributed_serving.html#:~:text=vLLM%20supports%20distributed%20tensor,LM%27s%20tensor%20parallel%20algorithm)[\[4\]](https://github.com/vllm-project/vllm/issues/27239#:~:text=Heterogeneous%20TP%20per%20Pipeline%20Stage,would%20let%20users%20combine) – while LLMs differ, the principles of balancing work and syncing might inspire our implementation. Also check if Hugging Face Diffusers or others have a multi-GPU inference mode (there was a discussion[\[5\]](https://huggingface.co/docs/diffusers/v0.21.0/en/training/distributed_inference#:~:text=Distributed%20inference%20with%20multiple%20GPUs,a%20GPU%20to%20each) suggesting splitting pipeline by rank manually).

**Phase 3: Prototyping (Implement Incrementally)**  
Start with the least disruptive changes and progressively tackle more complex ones:

1. **VAE Offload Prototype:** Modify the pipeline to accept an optional second device for the VAE. For testing, use two GPUs of similar capability. Measure the FPS and latency improvement. If we see, say, a 15–20% boost in FPS, that’s a good sign. Also measure the overhead of transferring latents between GPUs – ensure it doesn’t eat all the gains. If the improvement is minor (or negative), we may need to tweak (e.g., maybe only offload decode or encode, not both, to reduce transfers). This will also shake out any hidden assumptions in code (e.g., if the pipeline assumed everything on one device for some operations).

2. **Basic Pipeline Parallel (2 GPUs):** Implement a simple static split of the transformer. For example, move the latter half of generator’s layers to cuda:1. This could be done by manually slicing the torch modules. An easier route: run two separate processes – one creates the pipeline and loads the model normally (on GPU0) except it truncates after layer N; the other loads the model and starts from layer N+1 to end (on GPU1). Use a RPC or socket to send tensors from process 0 to 1 and back. This is crude but can test functionality. Alternatively, use torch.distributed.pipeline which might do micro-batch automatically if we wrap the model appropriately. The goal of the prototype is to confirm we can get both GPUs working on the same frame sequence and to measure speedup. We’ll likely hard-code for a single-stream, two-stage pipeline.

   * Important: ensure that the visual output is still correct (it should be, but any mismatch indicates a bug in how we split or sync).

   * Also, measure the new first-frame latency and per-frame latency. Compare with phase 1 baseline.

3. **Scaling to 3–4 GPUs / Flexible Partition:** If 2-GPU pipeline shows good results, extend the design to more GPUs (if available). This likely means allowing the number of pipeline stages to be configurable and splitting layers accordingly. We might also integrate better with PyTorch Distributed at this point (using collective communications instead of manual). If available, test on a server with 4 GPUs (e.g., 4× A100) to see if we approach linear scaling like StreamDiffusionV2. Keep an eye on whether any GPU is running ahead or starving – if so, adjust the layer distribution.

Only after these prototypes prove worthwhile should we integrate deeply into the product.

**Phase 4: Production Hardening**  
If we decide multi-GPU support is a go for Krea Realtime, we’ll need to polish the implementation for real-world use: \- Implement a **dynamic scheduler** (if necessary) akin to StreamDiffusionV2’s block scheduler. This could mean monitoring the inference time of each stage every few frames and migrating a layer to a faster/slower GPU if imbalance is detected. Or simpler, provide multiple preset partitions (for different hardware combinations or step counts) and choose appropriate one. \- Ensure **graceful fallback** to single-GPU. The pipeline should detect if only one GPU is present or if multi-GPU is disabled, and still run as before. We may also allow toggling multi-GPU via config or UI, since not all users have multi-GPU and some might prefer consistency. \- Support **heterogeneous GPUs** if possible. For example, if one GPU is much weaker, the scheduler could assign fewer layers to it. Testing on mixed setups (like one consumer GPU \+ one data-center GPU) would be valuable. \- Refine the **memory management**: When splitting models, we free up memory on each GPU – perhaps use that to increase batch size or do more diffusion steps for quality if desired. Conversely, ensure no memory leaks when sending tensors between devices (use pinned memory, etc., for efficiency). \- **Integration with Krea UI/UX**: Expose the multi-GPU utilization stats (maybe display how GPUs are utilized) for transparency. Handle errors – e.g., if one GPU fails mid-run, can the pipeline recover on remaining GPUs? (Probably out of scope initially, but think about it.)

Throughout, maintain close alignment with upstream developments – e.g., if StreamDiffusionV2 releases new optimizations or if PyTorch adds features that simplify pipeline parallelism, we should leverage them.

## Code Areas Likely to Change

Enabling multi-GPU will touch a few modules in our codebase:

* **Pipeline Initialization:** In KreaRealtimeVideoPipeline (and possibly the PipelineRegistry), we’ll need to allow multiple devices. This might involve passing a list of devices, or creating sub-pipelines. The snippet above shows where everything is sent to one device; we’d refactor that logic to conditionally split (e.g., if len(devices)\>1: ...). Also, any references to self.device throughout the pipeline code will need review – e.g., ensuring schedulers or torch autocast are aware of multiple devices.

* **Model Definition (Generator/UNet):** We might need to subclass or wrap the generator so that it can be partitioned. For example, define two nn.Module subclasses: one for the front half of the model, one for the back half. If the model architecture is accessible, we could programmatically split it by layers (e.g., take model.transformer.blocks\[:N\] vs \[N:\]). The forward call then needs to do: part1 on GPU0 \-\> send output \-\> part2 on GPU1. This could be orchestrated in the pipeline’s \_\_call\_\_ or generate method.

* **StreamDiffusionV2 reference pipeline:** We have a src/scope/core/pipelines/streamdiffusionv2/ in our code (likely integrating their model for comparison). It’s mentioned that this reference pipeline currently has **no multi-GPU yet** (presumably it’s running StreamDiffusionV2 on one GPU for now). If we implement multi-GPU for Krea, we might also later port it to StreamDiffusionV2 pipeline in our system, or at least ensure our design is general.

* **Distributed Communication Utilities:** We may introduce new code under scope/core/utils or similar for distributed operations. If using PyTorch’s high-level APIs (like torch.multiprocessing or pipeline.SyncPipeline), changes might be minimal. But if we use lower-level calls (like dist.send/recv or CUDA streams), we might implement those in a util. The vendored cutlass-cute kernels are unlikely to be needed immediately, but if we ever attempt model-parallel GEMMs (unlikely for now), those could come into play.

* **Testing and Session Management:** In notes/FA4/b300/session-state.md (our performance logs) and similar, we’ll add new scenarios for multi-GPU. We should be careful to test with different number of diffusion steps (the parallelism benefit might diminish at 1-step extreme low-latency mode, or conversely at 4-step high-quality mode, we want to see scaling). Also test that determinism remains (i.e., with fixed seed, single vs multi-GPU runs produce same output – they should, if we carefully order operations and sync RNG).

## Practical Guidance & Next Steps

**Near Term:** Start with Approach A (VAE offload) to get an immediate speed boost and flush out multi-GPU issues on a small scale. In parallel, study StreamDiffusionV2’s code to design our pipeline parallel approach (Approach B). If VAE offload shows, say, \>10% improvement, that can be a win we deliver sooner (especially for users with exactly 2 GPUs). Meanwhile, plan the architecture for Approach B – possibly using the torchrun multi-process model since it’s a proven path.

**Measuring Success:** We should set concrete targets: e.g., on dual RTX 5090 (32GB each), can we hit \~20 FPS at 320×576 (up from \~11 FPS on one 5090)? On four GPUs, maybe 30–40 FPS? And with minimal latency increase (\<0.2s extra). Also monitor memory – each GPU should ideally see \<50% of what one GPU needed, allowing headroom for more frames or higher resolution.

**Long Term:** If multi-GPU works well on one machine, consider multi-node (though that’s explicitly a non-goal for now). Also, keep an eye on upcoming tech: e.g., NVLink bridges or PCIe5 might reduce transfer times; software like FasterTransformer or TensorRT might introduce their own multi-GPU scheduling for diffusion that we could adopt rather than maintain ourselves.

## Open Questions (Revisited)

* **KV Cache Sync:** Does each pipeline stage simply keep its own KV cache state and that suffices? (Likely yes – we just need to ensure the first stage’s output token embeddings, which move to the second stage, align with the second stage’s cached keys for past frames.)

* **Minimum VRAM per GPU:** If one tries to use two smaller GPUs (e.g., two 24 GB cards instead of one 48 GB card), can it even fit? Splitting the model helps memory, but there is overhead to consider (each GPU might still need a copy of the text encoder and other common parts). Testing will reveal the true memory savings.

* **NVLink vs PCIe:** If our target user base is cloud or high-end workstations, they might have NVLink (e.g., dual Ada cards or HGX servers). NVLink can mitigate communication costs. If not, we should design assuming PCIe and see if the FPS scaling is still worth it. It might turn out that 2 GPUs over PCIe ≈ 1.8× speedup, whereas with NVLink it’s 1.95×, for example.

* **torch.compile and other optimizations:** If we’re using PyTorch 2.x with compiled mode (TorchDynamo), we should test that it doesn’t break across devices. It may be safest to disable compilation for the multi-GPU prototype initially. Later, we can see if each partition can be separately compiled for speed. Also, ensure that mixed precision, FlashAttention kernels, etc., all still function in a distributed context (they should, but best to verify).

## Non-Goals (Out of Scope for Now)

* We are focusing on **inference** only. Multi-GPU *training* of Krea models is a separate concern and not planned in this context. The code changes here will be geared toward serving and inference pipeline.

* **Multi-node inference** (using GPUs across different machines) is not targeted. All GPUs are assumed to be in one server with high-speed interconnect. Multi-node would introduce network communication which is significantly slower and would require a more complex scheduling (not to mention changes in how video frames are collected/merged).

* **Automatic partitioning or auto-sharding:** Initially, we will manually decide how to split the model. While there are research works on automated pipeline parallelism, our model is of a fixed architecture, so a one-time manual partition (perhaps with a few options for different GPU counts) is fine. We don’t need to engineer a generic partitioner – that would be overkill.

## Conclusion

Multi-GPU support for Krea Realtime appears both feasible and potentially very rewarding, based on analogues like StreamDiffusionV2. The biggest wins will come from pipeline parallelism (splitting the transformer across devices) and secondarily from offloading auxiliary tasks (VAE). While the implementation will introduce complexity (distributed synchronization, managing multiple processes, etc.), the payoff in performance could enable higher resolutions, more diffusion steps, or simply more concurrent streams in the future. By studying existing systems and incrementally prototyping, we can add this capability in a controlled manner.

The code will need careful refactoring around device placement and model definition, but no fundamental algorithm changes. It’s primarily an engineering challenge – one that aligns well with the trend in state-of-the-art diffusion serving. With thorough testing, we can integrate multi-GPU acceleration into Krea’s pipeline, keeping it competitive and efficient for real-time AI video generation.

## Sources

* StreamDiffusionV2 paper (Feng et al., 2025\) – Introduces the multi-GPU streaming pipeline and reports 58 FPS on 4×H100.

* StreamDiffusionV2 code – Provided insights into multi-GPU launch (using torchrun) and optional block scheduling for load balancing.

* Krea/Scope documentation – Confirms single-GPU requirements (40GB+ VRAM) for current pipeline[\[2\]](https://github.com/daydreamlive/scope/blob/d42c620648aa9b340e07ab42614b3d9de28eb9ed/src/scope/core/pipelines/krea_realtime_video/docs/usage.md#L9-L17).

* DistriFusion (Li et al., CVPR 2024\) – Demonstrates patch-based multi-GPU inference for diffusion, with asynchronous communication to maintain quality.

* Internal Scope code (feature/stream-recording branch) – Current pipeline implementation where all components are moved to one device, indicating areas to modify for multi-GPU. (See KreaRealtimeVideoPipeline in pipeline.py for device usage.)

---

[\[1\]](https://www.reddit.com/r/StableDiffusion/comments/1ocr3re/krea_realtime_14b_an_opensource_realtime_ai_video/#:~:text=,40GB%2B%20VRAM%20recommended) Krea Realtime 14B. An open-source realtime AI video model. : r/StableDiffusion

[https://www.reddit.com/r/StableDiffusion/comments/1ocr3re/krea\_realtime\_14b\_an\_opensource\_realtime\_ai\_video/](https://www.reddit.com/r/StableDiffusion/comments/1ocr3re/krea_realtime_14b_an_opensource_realtime_ai_video/)

[\[2\]](https://github.com/daydreamlive/scope/blob/d42c620648aa9b340e07ab42614b3d9de28eb9ed/src/scope/core/pipelines/krea_realtime_video/docs/usage.md#L9-L17) GitHub

[https://github.com/daydreamlive/scope/blob/d42c620648aa9b340e07ab42614b3d9de28eb9ed/src/scope/core/pipelines/krea\_realtime\_video/docs/usage.md](https://github.com/daydreamlive/scope/blob/d42c620648aa9b340e07ab42614b3d9de28eb9ed/src/scope/core/pipelines/krea_realtime_video/docs/usage.md)

[\[3\]](https://docs.vllm.ai/en/v0.8.0/serving/distributed_serving.html#:~:text=vLLM%20supports%20distributed%20tensor,LM%27s%20tensor%20parallel%20algorithm) Distributed Inference and Serving \- vLLM

[https://docs.vllm.ai/en/v0.8.0/serving/distributed\_serving.html](https://docs.vllm.ai/en/v0.8.0/serving/distributed_serving.html)

[\[4\]](https://github.com/vllm-project/vllm/issues/27239#:~:text=Heterogeneous%20TP%20per%20Pipeline%20Stage,would%20let%20users%20combine) Heterogeneous TP per Pipeline Stage (uneven TP across PP ...

[https://github.com/vllm-project/vllm/issues/27239](https://github.com/vllm-project/vllm/issues/27239)

[\[5\]](https://huggingface.co/docs/diffusers/v0.21.0/en/training/distributed_inference#:~:text=Distributed%20inference%20with%20multiple%20GPUs,a%20GPU%20to%20each) Distributed inference with multiple GPUs \- Hugging Face

[https://huggingface.co/docs/diffusers/v0.21.0/en/training/distributed\_inference](https://huggingface.co/docs/diffusers/v0.21.0/en/training/distributed_inference)