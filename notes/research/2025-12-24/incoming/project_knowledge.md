# Scope: Context Editing & Dev Console — Project Knowledge Base

> **Purpose**: Organized knowledge for human review and agent execution.  
> **Last Updated**: December 2024  
> **Timeline**: 16 days to hackathon demo

---

## 1. Current State

### What's Working
- **KREA pipeline running on B300**: 14 fps real-time generation
- **CUDA 13 environment**: Isolated venv, dependencies resolved (Codex 3hr autonomous run)
- **Core generation loop**: Prompt → frames working end-to-end

### What's Validated
- Pipeline source analysis confirms edit mechanism is viable (see §2)
- UI prototype exists (React/Gemini) — proves UX patterns

### What's Not Yet Built
- Edit API endpoints
- Dev console (web or CLI)
- Gemini integration for semantic edits
- Snapshot/branching system

### Key Metrics
| Metric | Value | Notes |
|--------|-------|-------|
| Generation FPS | 14 | B300, current config |
| Decoded buffer size | 9 frames | RGB, 4x temporal downsample |
| Context buffer size | 2 frames | Latents |
| KV cache blocks | ~14 | Approximate |

---

## 2. Core Technical Insight: Why Anchor Edits Propagate

### The Mechanism (from KREA source analysis)

The pipeline maintains two buffers:
```python
context_frame_buffer  # Latents [B, T, 16, H/8, W/8]
decoded_frame_buffer  # RGB pixels [B, T, 3, H, W]
```

During KV cache recomputation (`recompute_kv_cache.py`), the **first frame is re-encoded from RGB**:

```python
# This is the key insight — re-encoding happens
if (current_start_frame - num_frame_per_block) >= kv_cache_num_frames:
    decoded_first_frame = state.decoded_frame_buffer[:, :1]
    reencoded_latent = vae.encode_to_latent(decoded_first_frame)  # <-- HERE
    return torch.cat([reencoded_latent, context_frame_buffer], dim=1)
```

### Why This Matters

If we edit `decoded_frame_buffer[:, :1]` (the anchor frame), the next recomputation will:
1. Re-encode our edited RGB into latent space
2. Rebuild KV cache from edited content
3. All future frames "remember" the edit

This is **retroactive context injection** — we change the model's memory of what happened.

### Edit Surface

```python
# Intervention point
anchor_frame = pipeline.state.get('decoded_frame_buffer')[:, :1]
edited_frame = apply_semantic_edit(anchor_frame, "add tree in background")
pipeline.state.get('decoded_frame_buffer')[:, :1] = edited_frame
# Next generation will encode edited frame into KV cache
```

---

## 3. Validation Experiment

**Goal**: Prove edits propagate before building anything else.

### Runbook: Color Tint Test

```python
# test_context_edit.py
# Run this FIRST before any feature work

import torch
from scope.core.pipelines.krea_realtime_video import KreaRealtimeVideoPipeline
from diffusers.utils import export_to_video

def test_edit_propagation():
    """
    Mutate anchor frame with obvious color change.
    If subsequent frames stay tinted, mechanism works.
    """
    # 1. Setup (use existing config)
    pipeline = KreaRealtimeVideoPipeline(config, ...)
    
    # 2. Generate enough to fill buffer (~12 chunks)
    frames_pre = []
    for i in range(12):
        output = pipeline(prompts=[{"text": "a red ball bouncing on grass", "weight": 1.0}])
        frames_pre.append(output.cpu())
        print(f"Pre-edit chunk {i}, buffer shape: {pipeline.state.get('decoded_frame_buffer').shape}")
    
    # 3. Apply obvious mutation (blue tint)
    decoded_buffer = pipeline.state.get('decoded_frame_buffer')
    anchor = decoded_buffer[:, :1].clone()
    anchor[:, :, 2, :, :] = 1.0   # Max blue
    anchor[:, :, 0, :, :] = 0.0   # Zero red
    decoded_buffer[:, :1] = anchor
    pipeline.state.set('decoded_frame_buffer', decoded_buffer)
    print("EDIT APPLIED: Blue tint to anchor frame")
    
    # 4. Continue generating
    frames_post = []
    for i in range(12):
        output = pipeline(prompts=[{"text": "a red ball bouncing on grass", "weight": 1.0}])
        frames_post.append(output.cpu())
        print(f"Post-edit chunk {i}")
    
    # 5. Export for visual inspection
    all_frames = torch.cat(frames_pre + frames_post)
    export_to_video(all_frames.numpy(), "edit_propagation_test.mp4", fps=16)
    print("Exported: edit_propagation_test.mp4")
    print("SUCCESS CRITERIA: Frames after edit should have persistent blue tint")

if __name__ == "__main__":
    test_edit_propagation()
```

### Expected Outcomes

| Result | Interpretation | Action |
|--------|---------------|--------|
| Blue tint persists | ✅ Mechanism works | Proceed to Gemini integration |
| Flickers then reverts | Partial propagation | Investigate recompute timing |
| No change | Edit not reaching path | Check buffer indices, timing |
| Artifacts/corruption | Edit too aggressive | Try subtler mutation |

---

## 4. Gemini Integration — Reference Implementation

### Python Port (from working React prototype)

```python
# scope/integrations/gemini_edit.py
"""
Semantic frame editing via Gemini 2.5 Flash.
Ported from working React/TypeScript implementation.
"""

import base64
import io
from PIL import Image
import torch
from google import genai

# Initialize client (uses GOOGLE_API_KEY env var)
client = genai.Client()

def tensor_to_base64(tensor: torch.Tensor) -> str:
    """
    Convert [1, 1, 3, H, W] tensor to base64 PNG.
    Assumes tensor values in [-1, 1] range.
    """
    # Remove batch dims
    t = tensor.squeeze(0).squeeze(0)  # [3, H, W]
    
    # Normalize to [0, 255]
    t = ((t + 1) / 2 * 255).clamp(0, 255).byte()
    
    # To PIL
    t = t.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
    img = Image.fromarray(t)
    
    # To base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()

def base64_to_tensor(b64: str, device, dtype) -> torch.Tensor:
    """
    Convert base64 PNG to [1, 1, 3, H, W] tensor.
    Returns tensor in [-1, 1] range.
    """
    # Decode
    img_bytes = base64.b64decode(b64)
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    
    # To tensor
    import numpy as np
    t = torch.from_numpy(np.array(img)).to(device=device, dtype=dtype)
    t = t.permute(2, 0, 1)  # [3, H, W]
    
    # Normalize to [-1, 1]
    t = (t / 255.0) * 2 - 1
    
    return t.unsqueeze(0).unsqueeze(0)  # [1, 1, 3, H, W]

async def edit_frame(
    frame: torch.Tensor,
    instruction: str,
    model: str = "gemini-2.5-flash-preview-native-image"
) -> torch.Tensor:
    """
    Apply semantic edit to frame using Gemini.
    
    Args:
        frame: [1, 1, 3, H, W] tensor in [-1, 1] range
        instruction: Natural language edit instruction
        model: Gemini model to use
        
    Returns:
        Edited frame as [1, 1, 3, H, W] tensor
    """
    # Convert input to base64
    input_b64 = tensor_to_base64(frame)
    
    # Call Gemini
    response = await client.aio.models.generate_content(
        model=model,
        contents=[
            {
                "parts": [
                    {"inline_data": {"mime_type": "image/png", "data": input_b64}},
                    {"text": instruction}
                ]
            }
        ]
    )
    
    # Extract output image from response
    for part in response.candidates[0].content.parts:
        if hasattr(part, 'inline_data') and part.inline_data:
            output_b64 = part.inline_data.data
            return base64_to_tensor(output_b64, frame.device, frame.dtype)
    
    raise RuntimeError("Gemini did not return an image")

async def describe_frame(
    frame: torch.Tensor,
    model: str = "gemini-2.5-flash-preview"
) -> str:
    """
    Get VLM description of frame for agent feedback loop.
    
    Args:
        frame: [1, 1, 3, H, W] tensor
        
    Returns:
        Natural language description
    """
    input_b64 = tensor_to_base64(frame)
    
    response = await client.aio.models.generate_content(
        model=model,
        contents=[
            {
                "parts": [
                    {"text": "Describe this video frame concisely. Focus on: subjects, actions, environment, lighting."},
                    {"inline_data": {"mime_type": "image/png", "data": input_b64}}
                ]
            }
        ],
        config={"thinking_config": {"thinking_budget": 0}}
    )
    
    return response.text
```

### Usage in Edit Endpoint

```python
# In API route
from scope.integrations.gemini_edit import edit_frame

@router.post("/api/edit")
async def apply_edit(instruction: str, session: Session = Depends(get_session)):
    # Get anchor frame
    buffer = session.driver.pipeline.state.get('decoded_frame_buffer')
    anchor = buffer[:, :1].clone()
    
    # Apply semantic edit
    edited = await edit_frame(anchor, instruction)
    
    # Write back
    buffer[:, :1] = edited
    session.driver.pipeline.state.set('decoded_frame_buffer', buffer)
    
    return {"status": "applied"}
```

---

## 5. API Surface

### MVP Endpoints (Hackathon)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/state` | Current driver state, chunk count, prompt |
| POST | `/api/step` | Generate one chunk |
| POST | `/api/run` | Start continuous generation |
| POST | `/api/pause` | Pause generation |
| PUT | `/api/prompt` | Set prompt text |
| POST | `/api/edit` | Edit anchor frame (semantic) |
| GET | `/api/frame/latest` | Get current frame as image |
| GET | `/api/frame/describe` | VLM description of current frame |

### Extended Endpoints (Post-MVP)

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/edit/preview` | Preview edit without applying |
| POST | `/api/snapshot` | Create state snapshot |
| GET | `/api/snapshots` | List snapshots |
| POST | `/api/snapshot/{id}/restore` | Restore snapshot |
| POST | `/api/fork` | Fork current state |
| WS | `/ws/frames` | Real-time frame streaming |

### Request/Response Schemas

```python
# Pydantic models

class DriverState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"

class StateResponse(BaseModel):
    state: DriverState
    chunk: int
    prompt: str | None
    buffer_frames: int

class StepResponse(BaseModel):
    chunk: int
    frames_generated: int
    latency_ms: float

class EditRequest(BaseModel):
    instruction: str
    strength: float = 0.8

class EditResponse(BaseModel):
    status: str  # "applied" | "failed"
    edit_latency_ms: float

class DescribeResponse(BaseModel):
    description: str
```

---

## 6. UI Specification

### Design Reference

The React prototype (`context-editing-_-dev-console.zip`) validates these UX patterns:

**Layout (3-column)**:
- Left sidebar: Navigation, state indicator
- Center: Frame preview (large), timeline strip below
- Right: Edit panel, metrics

**Edit Panel Components**:
1. Anchor frame thumbnail (small preview of what you're editing)
2. Instruction textarea (multi-line)
3. "Apply Semantic Edit" button
4. Explanation text: "Changes trigger KV cache recomputation..."

**Timeline Strip**:
- Horizontal scroll of frame thumbnails
- Anchor frame marked with colored dot
- Selected frame highlighted
- Frame count indicator

**Console/Log**:
- Bottom drawer
- Format: `[HH:MM:SS] TYPE message`
- Color-coded: info (blue), success (green), error (red), command (purple)
- Auto-scroll to bottom

### FastHTML Implementation Notes

Port the layout, not the React code. Key elements:

```python
# Minimal structure
def console_page():
    return Div(
        sidebar(),
        Div(
            header_controls(),      # play/pause, prompt input, snapshot btn
            frame_preview(),        # large current frame
            timeline_strip(),       # horizontal frame thumbnails
            cls="main"
        ),
        Div(
            edit_panel(),           # instruction input, apply button
            metrics_panel(),        # latency, vram, buffer size
            cls="right-panel"
        ),
        console_log(),              # bottom drawer
    )
```

For HTMX patterns, use polling initially (simpler), upgrade to WebSocket if needed.

---

## 7. Capability Observations

Document what we've learned about our tools:

| Tool | Capability | Evidence |
|------|-----------|----------|
| **Gemini 3 Flash** | Builds complete React UIs from specs | Built prototype from artifact description in ~2 min |
| **Codex** | Follows multi-hour runbooks autonomously | CUDA 13 install, venv setup, dependency resolution, profiling — 3hr run |
| **Claude Code** | Parallel execution with Codex | Running separate workstreams simultaneously |
| **Gemini 2.5 Flash (image)** | Semantic image editing | Used in prototype, works for frame edits |
| **Deep Research** | Architectural analysis | Found the re-encode path in KREA source |

### Implications for Workflow

1. **Detailed specs are high-value** — Agents execute them well
2. **Runbook format works** — Step-by-step procedures for autonomous execution
3. **UI can be generated** — Describe layout, let Gemini build it
4. **Parallel workstreams** — Codex on infra, Claude Code on features

---

## 8. Runbooks

### Runbook: Add New API Endpoint

```
1. Define Pydantic request/response models in scope/api/schemas.py
2. Create route function in scope/api/routes/{feature}.py
3. Wire to driver method in GeneratorDriver
4. Add to router in scope/api/main.py
5. Test with curl or httpie
6. Add CLI wrapper command in scope/cli/video_cli.py
```

### Runbook: Integrate New Gemini Model

```
1. Check model string at ai.google.dev/models
2. Update model parameter in scope/integrations/gemini_edit.py
3. Test with simple edit instruction
4. Adjust any config parameters (thinking_budget, etc.)
5. Update capability observations if behavior differs
```

### Runbook: Debug Edit Not Propagating

```
1. Confirm buffer is populated: print(pipeline.state.get('decoded_frame_buffer').shape)
2. Confirm edit applied: save anchor before/after as images
3. Check recompute timing: add logging to get_context_frames()
4. Verify re-encode path hit: log when vae.encode_to_latent called
5. If still failing: check if model "corrects" edit (try aligned prompt)
```

---

## 9. Open Questions / Decisions Needed

### Technical
- [ ] Exact tensor value range from KREA VAE decode ([-1,1] or [0,1]?)
- [ ] Optimal edit strength for Gemini (default 0.8, needs testing)
- [ ] Frame export format for web (JPEG for speed vs PNG for quality)

### Product
- [ ] Should snapshots persist to disk or memory-only?
- [ ] How many parallel sessions to support?
- [ ] Export format for final videos?

### Process
- [ ] Which workstream gets Codex vs Claude Code?
- [ ] Checkpoint/sync cadence during hackathon?

---

## 10. File References

| Artifact | Location | Notes |
|----------|----------|-------|
| Full spec (verbose) | `context_editing_and_console_spec.md` | Complete technical detail |
| React prototype | `context-editing-_-dev-console.zip` | UX reference, Gemini integration |
| Architecture v1.1 | `realtime_video_architecture.md` | Core driver/session design |
| This document | Project files | Active knowledge base |

---

## 11. Next Actions (Priority Order)

1. **Run validation experiment** — Prove mechanism before anything else
2. **Port Gemini integration to Python** — Reference code above
3. **Build MVP API endpoints** — 4 core endpoints
4. **Basic frame visibility** — Web preview, even ugly
5. **Demo script** — "Watch edit propagate" narrative
6. **CLI wrapper** — For agent automation
7. **Rich UI** — Time permitting, or let Gemini generate it

---

*This document is the single source of truth for the context editing feature. Update it as work progresses.*
