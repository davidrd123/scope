# Context Editing - Code Reference

Extracted from:
- `notes/research/2025-12-24/incoming/project_knowledge.md`
- `notes/research/2025-12-24/incoming/context_editing_and_console_spec.md`

---

## Gemini Frame Editing

```python
# scope/integrations/gemini_edit.py
"""Semantic frame editing via Gemini 2.5 Flash."""

import base64
import io
from PIL import Image
import torch
from google import genai

client = genai.Client()  # Uses GOOGLE_API_KEY env var

def tensor_to_base64(tensor: torch.Tensor) -> str:
    """Convert [1, 1, 3, H, W] tensor to base64 PNG. Values in [-1, 1]."""
    t = tensor.squeeze(0).squeeze(0)  # [3, H, W]
    t = ((t + 1) / 2 * 255).clamp(0, 255).byte()
    t = t.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
    img = Image.fromarray(t)
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()

def base64_to_tensor(b64: str, device, dtype) -> torch.Tensor:
    """Convert base64 PNG to [1, 1, 3, H, W] tensor. Returns [-1, 1]."""
    img_bytes = base64.b64decode(b64)
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    import numpy as np
    t = torch.from_numpy(np.array(img)).to(device=device, dtype=dtype)
    t = t.permute(2, 0, 1)  # [3, H, W]
    t = (t / 255.0) * 2 - 1
    return t.unsqueeze(0).unsqueeze(0)  # [1, 1, 3, H, W]

async def edit_frame(
    frame: torch.Tensor,
    instruction: str,
    model: str = "gemini-2.5-flash-preview-native-image"
) -> torch.Tensor:
    """Apply semantic edit to frame using Gemini."""
    input_b64 = tensor_to_base64(frame)
    response = await client.aio.models.generate_content(
        model=model,
        contents=[{
            "parts": [
                {"inline_data": {"mime_type": "image/png", "data": input_b64}},
                {"text": instruction}
            ]
        }]
    )
    for part in response.candidates[0].content.parts:
        if hasattr(part, 'inline_data') and part.inline_data:
            output_b64 = part.inline_data.data
            return base64_to_tensor(output_b64, frame.device, frame.dtype)
    raise RuntimeError("Gemini did not return an image")

async def describe_frame(
    frame: torch.Tensor,
    model: str = "gemini-2.5-flash-preview"
) -> str:
    """Get VLM description of frame for agent feedback loop."""
    input_b64 = tensor_to_base64(frame)
    response = await client.aio.models.generate_content(
        model=model,
        contents=[{
            "parts": [
                {"text": "Describe this video frame concisely. Focus on: subjects, actions, environment, lighting."},
                {"inline_data": {"mime_type": "image/png", "data": input_b64}}
            ]
        }],
        config={"thinking_config": {"thinking_budget": 0}}
    )
    return response.text
```

---

## Edit API Endpoint

```python
# scope/api/routes/edit.py

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

router = APIRouter(prefix="/api", tags=["edit"])

class EditRequest(BaseModel):
    instruction: str
    strength: float = 0.8

class EditResponse(BaseModel):
    status: str
    anchor_modified: bool
    edit_latency_ms: float

@router.post("/edit", response_model=EditResponse)
async def edit_anchor(req: EditRequest, session: Session = Depends(get_session)):
    """Edit anchor frame in decoded_frame_buffer."""
    driver = session.driver

    if driver.state != DriverState.PAUSED:
        raise HTTPException(400, "Must be paused to edit")

    decoded_buffer = driver.pipeline.state.get('decoded_frame_buffer')
    if decoded_buffer is None or decoded_buffer.shape[1] < 1:
        raise HTTPException(400, "No frames in buffer yet")

    anchor_frame = decoded_buffer[:, :1].clone()

    import time
    start = time.perf_counter()
    edited_frame = await edit_frame(anchor_frame, req.instruction)
    edit_ms = (time.perf_counter() - start) * 1000

    decoded_buffer[:, :1] = edited_frame
    driver.pipeline.state.set('decoded_frame_buffer', decoded_buffer)

    return EditResponse(
        status="edit_applied",
        anchor_modified=True,
        edit_latency_ms=edit_ms
    )
```

---

## Validation Experiment

Run this FIRST to prove the mechanism works:

```python
# test_context_edit.py
import torch
from scope.core.pipelines.krea_realtime_video import KreaRealtimeVideoPipeline
from diffusers.utils import export_to_video

def test_edit_propagation():
    """Mutate anchor frame. If subsequent frames stay tinted, mechanism works."""
    pipeline = KreaRealtimeVideoPipeline(config, ...)

    # Phase 1: Generate enough to fill buffer (~12 chunks)
    frames_pre = []
    for i in range(12):
        output = pipeline(prompts=[{"text": "a red ball bouncing on grass", "weight": 1.0}])
        frames_pre.append(output.cpu())

    # Phase 2: Apply obvious mutation (blue tint)
    decoded_buffer = pipeline.state.get('decoded_frame_buffer')
    anchor = decoded_buffer[:, :1].clone()
    anchor[:, :, 2, :, :] = 1.0   # Max blue
    anchor[:, :, 0, :, :] = 0.0   # Zero red
    decoded_buffer[:, :1] = anchor
    pipeline.state.set('decoded_frame_buffer', decoded_buffer)

    # Phase 3: Continue generating
    frames_post = []
    for i in range(12):
        output = pipeline(prompts=[{"text": "a red ball bouncing on grass", "weight": 1.0}])
        frames_post.append(output.cpu())

    # Phase 4: Export
    all_frames = torch.cat(frames_pre + frames_post)
    export_to_video(all_frames.numpy(), "edit_propagation_test.mp4", fps=16)
    print("SUCCESS CRITERIA: Frames after edit should have persistent blue tint")
```

**Expected outcomes:**

| Result | Interpretation | Action |
|--------|---------------|--------|
| Blue tint persists | Mechanism works | Proceed to Gemini integration |
| Flickers then reverts | Partial propagation | Investigate recompute timing |
| No change | Edit not reaching path | Check buffer indices |
| Artifacts/corruption | Edit too aggressive | Try subtler mutation |

---

## GeneratorDriver Additions

```python
# Additions to GeneratorDriver

def get_anchor_frame(self) -> torch.Tensor | None:
    """Get current anchor frame from decoded buffer."""
    decoded_buffer = self.pipeline.state.get('decoded_frame_buffer')
    if decoded_buffer is None or decoded_buffer.shape[1] < 1:
        return None
    return decoded_buffer[:, :1].clone()

def set_anchor_frame(self, frame: torch.Tensor) -> bool:
    """Replace anchor frame in decoded buffer."""
    decoded_buffer = self.pipeline.state.get('decoded_frame_buffer')
    if decoded_buffer is None or decoded_buffer.shape[1] < 1:
        return False
    decoded_buffer[:, :1] = frame
    self.pipeline.state.set('decoded_frame_buffer', decoded_buffer)
    return True

def get_buffer_info(self) -> dict:
    """Get information about current buffers."""
    decoded = self.pipeline.state.get('decoded_frame_buffer')
    context = self.pipeline.state.get('context_frame_buffer')
    return {
        "decoded_buffer_frames": decoded.shape[1] if decoded is not None else 0,
        "context_buffer_frames": context.shape[1] if context is not None else 0,
    }
```

---

## Frame Tensor Conventions

KREA pipeline uses:
- Shape: `[B, T, C, H, W]` for video tensors
- Values: `[-1, 1]` after VAE decode
- dtype: `torch.bfloat16` typically

```python
def tensor_to_pil(t: torch.Tensor) -> Image:
    """Convert [1, 1, 3, H, W] tensor to PIL Image."""
    t = t.squeeze(0).squeeze(0)  # [3, H, W]
    t = (t + 1) / 2  # [-1,1] -> [0,1]
    t = t.clamp(0, 1)
    t = (t * 255).byte()
    t = t.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
    return Image.fromarray(t)

def pil_to_tensor(img: Image, device, dtype) -> torch.Tensor:
    """Convert PIL Image to [1, 1, 3, H, W] tensor."""
    t = torch.from_numpy(np.array(img)).to(device=device, dtype=dtype)
    t = t.permute(2, 0, 1)  # [3, H, W]
    t = t / 255.0  # [0,255] -> [0,1]
    t = t * 2 - 1  # [0,1] -> [-1,1]
    return t.unsqueeze(0).unsqueeze(0)  # [1, 1, 3, H, W]
```
