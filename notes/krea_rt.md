---
license: apache-2.0
base_model:
- Wan-AI/Wan2.1-T2V-14B
pipeline_tag: text-to-video
tags:
- diffusion-single-file
- text-to-video
- video-to-video
- realtime
library_name: diffusers
---
Krea Realtime 14B is distilled from the [Wan 2.1 14B text-to-video model](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) using Self-Forcing, a technique for converting regular video diffusion models into autoregressive models. It achieves a text-to-video inference speed of **11fps** using 4 inference steps on a single NVIDIA B200 GPU. For more details on our training methodology and sampling innovations, refer to our [technical blog post](https://www.krea.ai/blog/krea-realtime-14b).

Inference code can be found [here](https://github.com/krea-ai/realtime-video).


<video width="100%" controls>
    <source src="https://cdn-uploads.huggingface.co/production/uploads/62a2712903bf94c3ac3ae004/NI1qn109PHVeO_LvBQIr8.mp4" type="video/mp4">
    Your browser does not support the video tag.
</video>


- Our model is over **10x larger than existing realtime video models**
- We introduce **novel techniques for mitigating error accumulation,** including **KV Cache Recomputation** and **KV Cache Attention Bias**
- We develop **memory optimizations specific to autoregressive video diffusion models** that facilitate training large autoregressive models
- **Our model enables realtime interactive capabilities**: Users can modify prompts mid-generation, restyle videos on-the-fly, and see first frames within 1 second

# Video To Video
Krea realtime allows users to stream real videos, webcam inputs, or canvas primitives into the model, unlocking controllable video synthesis and editing

<div align="center">
    <table>
        <tr>
            <td width="50%">
                <video width="100%" controls>
                    <source src="https://cdn-uploads.huggingface.co/production/uploads/62a2712903bf94c3ac3ae004/iW8bdR6Q4WlZS3PW87Q8c.mp4" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
            </td>
            <td width="50%">
                <video width="100%" controls>
                    <source src="https://cdn-uploads.huggingface.co/production/uploads/62a2712903bf94c3ac3ae004/x2JUP1bvzBr1_nuUpHuM-.mp4" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
            </td>
        </tr>
        <tr>
            <td width="50%">
                <video width="100%" controls>
                    <source src="https://cdn-uploads.huggingface.co/production/uploads/62a2712903bf94c3ac3ae004/lYEE3n5_Ms8B9jCTkfJuq.mp4" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
            </td>
            <td width="50%">
                <video width="100%" controls>
                    <source src="https://cdn-uploads.huggingface.co/production/uploads/62a2712903bf94c3ac3ae004/e_rSMargujaaVqHSS-qm5.mp4" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
            </td>
        </tr>
    </table>
</div>


# Text To Video
Krea realtime allows users to generate videos in a streaming fashion with ~1s time to first frame.
<div align="center">
    <table>
        <tr>
            <td width="50%">
                <video width="100%" controls>
                    <source src="https://cdn-uploads.huggingface.co/production/uploads/62a2712903bf94c3ac3ae004/4Lz7mRSAHXBPi6Q56Vxmi.mp4" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
            </td>
            <td width="50%">
                <video width="100%" controls>
                    <source src="https://cdn-uploads.huggingface.co/production/uploads/62a2712903bf94c3ac3ae004/-rLGuV0eaXRPCDYMcT0Xr.mp4" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
            </td>
        </tr>
        <tr>
            <td width="50%">
                <video width="100%" controls>
                    <source src="https://cdn-uploads.huggingface.co/production/uploads/62a2712903bf94c3ac3ae004/dg5Tf7lIme_bc-JHrNAnD.mp4" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
            </td>
            <td width="50%">
                <video width="100%" controls>
                    <source src="https://cdn-uploads.huggingface.co/production/uploads/62a2712903bf94c3ac3ae004/nRfYFFiMN3KKfshZVYeTz.mp4" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
            </td>
        </tr>
    </table>
</div>

# Use it with our inference code

Set up
```bash
sudo apt install ffmpeg # install if you haven't already
git clone https://github.com/krea-ai/realtime-video
cd realtime-video
uv sync
uv pip install flash_attn --no-build-isolation
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir-use-symlinks False --local-dir wan_models/Wan2.1-T2V-1.3B
huggingface-cli download krea/krea-realtime-video krea-realtime-video-14b.safetensors --local-dir-use-symlinks False --local-dir checkpoints/krea-realtime-video-14b.safetensors
```

Run
```bash
export MODEL_FOLDER=Wan-AI
export CUDA_VISIBLE_DEVICES=0 # pick the GPU you want to serve on
export DO_COMPILE=true

uvicorn release_server:app --host 0.0.0.0 --port 8000
```

And use the web app at http://localhost:8000/ in your browser
(for more advanced use-cases and custom pipeline check out our GitHub repository: https://github.com/krea-ai/realtime-video)

# Use it with 🧨 diffusers

Krea Realtime 14B can be used with the `diffusers` library utilizing the new Modular Diffusers structure

```bash
# Install diffusers from main
pip install git+github.com/huggingface/diffusers.git
```

<details>
<summary>Text to Video</summary>

```py
import torch
from tqdm import tqdm
from diffusers.utils import export_to_video
from diffusers import ModularPipeline
from diffusers.modular_pipelines import PipelineState

repo_id = "krea/krea-realtime-video"
pipe = ModularPipeline.from_pretrained(repo_id, trust_remote_code=True)
pipe.load_components(
    trust_remote_code=True,
    device_map="cuda",
    torch_dtype={"default": torch.bfloat16, "vae": torch.float16},
)
for block in pipe.transformer.blocks:
    block.self_attn.fuse_projections()

num_blocks = 9

frames = []
state = PipelineState()
prompt = ["a cat sitting on a boat"]

generator = torch.Generator(device=pipe.device).manual_seed(42)
for block_idx in tqdm(range(num_blocks)):
    state = pipe(
        state,
        prompt=prompt,
        num_inference_steps=6,
        num_blocks=num_blocks,
        block_idx=block_idx,
        generator=generator,
    )
    frames.extend(state.values["videos"][0])

export_to_video(frames, "output.mp4", fps=24)
```
</details>

<details>
<summary>Video to Video</summary>

```py
import torch
from tqdm import tqdm
from diffusers.utils import load_video, export_to_video
from diffusers import ModularPipeline
from diffusers.modular_pipelines import PipelineState

repo_id = "krea/krea-realtime-video"
pipe = ModularPipeline.from_pretrained(repo_id, trust_remote_code=True)
pipe.load_components(
    trust_remote_code=True,
    device_map="cuda",
    torch_dtype={"default": torch.bfloat16, "vae": torch.float16},
)
for block in pipe.transformer.blocks:
    block.self_attn.fuse_projections()

num_blocks = 9
video = load_video("https://app-uploads.krea.ai/public/a8218957-1a80-43dc-81b2-da970b5f2221-video.mp4")

frames = []
prompt = ["A car racing down a snowy mountain"]

state = PipelineState()
generator = torch.Generator("cuda").manual_seed(42)
for block_idx in tqdm(range(num_blocks)):
    state = pipe(
        state,
        video=video,
        prompt=prompt,
        num_inference_steps=6,
        strength=0.3,
        block_idx=block_idx,
        generator=generator,
    )
    frames.extend(state.values["videos"][0])

export_to_video(frames, "output-v2v.mp4", fps=24)
```
</details>

<details>
<summary>Streaming Video to Video</summary>

Using the `video_stream` input will process video frames in as they arrive, while maintaining temporal consistency across chunks.

```py
import torch
from collections import deque
from tqdm import tqdm
from diffusers.utils import load_video, export_to_video
from diffusers import ModularPipeline
from diffusers.modular_pipelines import PipelineState

repo_id = "krea/krea-realtime-video"
pipe = ModularPipeline.from_pretrained(repo_id, trust_remote_code=True)
pipe.load_components(
    trust_remote_code=True,
    device_map="cuda",
    torch_dtype={"default": torch.bfloat16, "vae": torch.float16},
)
for block in pipe.transformer.blocks:
    block.self_attn.fuse_projections()

n_samples = 9
frame_sample_len = 12
video = load_video(
    "https://app-uploads.krea.ai/public/a8218957-1a80-43dc-81b2-da970b5f2221-video.mp4"
)

# Simulate streaming video input
frame_samples = [
    video[sample_start : sample_start + frame_sample_len]
    for sample_start in range(0, n_samples * frame_sample_len, frame_sample_len)
]

frames = []
state = PipelineState()
prompt = ["A car racing down a snowny mountain road"]

block_idx = 0
generator = torch.Generator("cpu").manual_seed(42)
for frame_sample in tqdm(frame_samples):
    state = pipe(
        state,
        video_stream=frame_sample,
        prompt=prompt,
        num_inference_steps=6,
        strength=0.3,
        block_idx=block_idx,
        generator=generator,
    )
    frames.extend(state.values["videos"][0])

    block_idx += 1

export_to_video(frames, "output-v2v-streaming.mp4", fps=24)
```
</details>

<details>
<summary>Using LoRAs</summary>

```py
import torch
from collections import deque
from tqdm import tqdm
from diffusers.utils import export_to_video
from diffusers import ModularPipeline
from diffusers.modular_pipelines import PipelineState

repo_id = "krea/krea-realtime-video"
pipe = ModularPipeline.from_pretrained(repo_id, trust_remote_code=True)
pipe.load_components(
    trust_remote_code=True,
    device_map="cuda",
    torch_dtype={"default": torch.bfloat16, "vae": torch.float16},
)
pipe.transformer.load_lora_adapter(
    "shauray/Origami_WanLora",
    prefix="diffusion_model",
    weight_name="origami_000000500.safetensors",
    adapter_name="origami",
)
for block in pipe.transformer.blocks:
    block.self_attn.fuse_projections()

num_blocks = 9

frames = []
state = PipelineState()
prompt = ["[origami] a cat sitting on a boat"]

generator = torch.Generator("cuda").manual_seed(42)
for block_idx in tqdm(range(num_blocks)):
    state = pipe(
        state,
        prompt=prompt,
        num_inference_steps=6,
        num_blocks=num_blocks,
        block_idx=block_idx,
        generator=generator,
    )
    frames.extend(state.values["videos"][0])

export_to_video(frames, "output.mp4", fps=24)
```
</details>

<details>
<summary>Optimized Inference</summary>

To optimize inference speed and memory usage on Hopper level GPUs (H100s), we recommend using `torch.compile`, Sageattention and FP8 quantization with [torchao](https://github.com/pytorch/ao).

First let's set up our depedencies by enabling Sageattention via Hub [kernels](https://huggingface.co/docs/kernels/en/index) and installing the `torchao` and `kernels` packages.

```shell
export DIFFUSERS_ENABLE_HUB_KERNELS=true
pip install -U kernels torchao
```

Alternatively, you can use Flash Attention 3 via kernels by disabling Sageattention:

```shell
export DISABLE_SAGEATTENTION=1
```

Then we will iterate over the blocks of the transformer and apply quantization and `torch.compile`.

```py
import torch
from collections import deque
from tqdm import tqdm
from diffusers.utils import export_to_video
from diffusers import ModularPipeline
from diffusers.modular_pipelines import PipelineState
from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, quantize_

repo_id = "krea/krea-realtime-video"
pipe = ModularPipeline.from_pretrained(repo_id, trust_remote_code=True)
pipe.load_components(
    trust_remote_code=True,
    device_map="cuda",
    torch_dtype={"default": torch.bfloat16, "vae": torch.float16},
)

for block in pipe.transformer.blocks:
    block.self_attn.fuse_projections()

# Quantize just the transformer blocks
for block in pipe.transformer.blocks:
    quantize_(block, Float8DynamicActivationFloat8WeightConfig())

# Compile just the attention modules
for submod in pipe.transformer.modules():
    if submod.__class__.__name__ in ["CausalWanAttentionBlock"]:
        submod.compile(fullgraph=False)

num_blocks = 9

state = PipelineState()
prompt = ["a cat sitting on a boat"]

# Compile warmup
for block_idx in range(num_blocks):
    state = pipe(
        state,
        prompt=prompt,
        num_inference_steps=2,
        num_blocks=num_blocks,
        num_frames_per_block=num_frames_per_block,
        block_idx=block_idx,
        generator=torch.Generator("cuda").manual_seed(42),
    )

# Reset state
state = PipelineState()
generator = torch.Generator("cuda").manual_seed(42)
for block_idx in tqdm(range(num_blocks)):
    state = pipe(
        state,
        prompt=prompt,
        num_inference_steps=6,
        num_blocks=num_blocks,
        block_idx=block_idx,
        generator=generator,
    )
    frames.extend(state.values["videos"][0])

export_to_video(frames, "output.mp4", fps=24)
```
</details>