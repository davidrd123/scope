# Context Editing & Dev Console Specification

**Status**: Design complete, ready for implementation  
**Depends on**: `realtime_video_architecture.md` (v1.1)  
**Date**: December 2024

---

## Executive Summary

This document specifies two interconnected features:

1. **Retroactive Context Editing**: Edit frames in the pipeline's decoded buffer, triggering KV cache recomputation so changes propagate to future frames.

2. **Dev Console**: CLI + Web interface for interactive prompt iteration, editing, and branching.

These features transform the real-time video pipeline from a "run and hope" tool into an **interactive directing instrument**.

---

## Part 1: Context Editing Mechanism

### Discovery from KREA Source

Analysis of `recompute_kv_cache.py` reveals the pipeline stores two buffers:

```python
# From prepare_context_frames.py

# LATENTS - output of diffusion model, input to VAE decoder
context_frame_buffer = torch.cat([
    context_frame_buffer,
    latents  # Shape: [B, T, 16, H/8, W/8]
], dim=1)[:, -context_frame_buffer_max_size:]

# RGB PIXELS - output of VAE decoder  
decoded_frame_buffer = torch.cat([
    decoded_frame_buffer,
    output_video  # Shape: [B, T, 3, H, W]
], dim=1)[:, -decoded_frame_buffer_max_size:]
```

Critically, during KV cache recomputation, the **first frame is re-encoded from RGB**:

```python
# From recompute_kv_cache.py, get_context_frames()

if (current_start_frame - num_frame_per_block) >= kv_cache_num_frames:
    # RE-ENCODE from decoded RGB buffer
    decoded_first_frame = state.decoded_frame_buffer[:, :1]
    reencoded_latent = vae.encode_to_latent(decoded_first_frame)
    return torch.cat([reencoded_latent, context_frame_buffer], dim=1)
```

### The Edit Surface

This re-encoding path creates an edit surface:

```
decoded_frame_buffer[:, :1] (RGB anchor frame)
        │
        ▼ Edit with image model (e.g., nano-banana)
        │
        ▼ Next recompute: vae.encode_to_latent(edited_frame)
        │
        ▼ KV cache rebuilt from edited content
        │
        ▼ Future frames "remember" the edit
```

### Buffer Sizes

From `recompute_kv_cache.py`:

```python
context_frame_buffer_max_size = kv_cache_num_frames - 1  # Default: 2 latent frames
decoded_frame_buffer_max_size = 1 + (kv_cache_num_frames - 1) * 4  # Default: 9 RGB frames
```

The decoded buffer is 4x larger due to VAE temporal downsampling.

### What This Enables

| Operation | Description |
|-----------|-------------|
| **Error correction** | Remove hallucinated limb, fix identity drift |
| **Retroactive insertion** | "There should have been a knife on the table" |
| **Character modification** | Costume change, add injury, fix expression |
| **Environment tweaks** | Add rain, change lighting, time of day |
| **Continuity rewrite** | Add prop so character can naturally interact with it |

### Prompt Change vs Edit

**Prompt-only change** (current capability):
- New direction, same world state
- Model must reconcile with what's in KV cache
- Constrained by existing context

**Rewind + Edit + New prompt** (new capability):
- Change the world state itself
- Model generates from edited context
- Clean slate for new direction

Example:
```
Generate 100 frames: character walking right
Prompt "walk left": awkward transition, model fights context

vs.

Generate 100 frames: character walking right  
Rewind to frame 50
Edit anchor: "add oasis on left side"
New prompt: "character notices oasis and walks toward it"
→ Natural motivated action
```

---

## Part 2: Validation Experiment

### Hour 1 Test: Prove the Mechanism

Simple test without any image edit model:

```python
# test_context_edit.py
import torch
from scope.core.pipelines.krea_realtime_video import KreaRealtimeVideoPipeline
from diffusers.utils import export_to_video

# Setup pipeline (use existing config)
pipeline = KreaRealtimeVideoPipeline(config, ...)

# Phase 1: Generate enough frames to fill buffer
# Need ~10 chunks so recompute starts using decoded_frame_buffer
frames_out = []
for i in range(12):
    output = pipeline(prompts=[{"text": "a red ball bouncing on grass", "weight": 1.0}])
    frames_out.append(output.cpu())
    print(f"Chunk {i}, decoded_buffer shape: {pipeline.state.get('decoded_frame_buffer').shape}")

# Phase 2: Mutate the anchor frame
decoded_buffer = pipeline.state.get('decoded_frame_buffer')
print(f"Buffer shape before edit: {decoded_buffer.shape}")

# Obvious mutation: tint blue
anchor_frame = decoded_buffer[:, :1].clone()
anchor_frame[:, :, 2, :, :] = 1.0  # Max blue channel
anchor_frame[:, :, 0, :, :] = 0.0  # Zero red channel
decoded_buffer[:, :1] = anchor_frame

# Write back
pipeline.state.set('decoded_frame_buffer', decoded_buffer)

# Phase 3: Continue generating
for i in range(12):
    output = pipeline(prompts=[{"text": "a red ball bouncing on grass", "weight": 1.0}])
    frames_out.append(output.cpu())
    print(f"Post-edit chunk {i}")

# Phase 4: Export and watch
all_frames = torch.cat(frames_out)
export_to_video(all_frames.numpy(), "context_edit_test.mp4", fps=16)
print("Exported to context_edit_test.mp4")
```

### Expected Results

| Outcome | Interpretation | Next Step |
|---------|---------------|-----------|
| Scene goes blue, stays blue | ✅ Edit propagates through KV cache | Integrate real edit model |
| Flickers blue then reverts | Model "corrects" back to prior | Try aligning prompt with edit |
| No visible change | Edit not reaching recompute path | Check timing, buffer indices |
| Generation breaks | Edit too aggressive | Try subtler mutation |

---

## Part 3: Dev Console Architecture

### Design Principles

1. **CLI-first**: Primary interface for both humans and agents
2. **Web for display**: Frames must be seen somewhere visual
3. **JSON everywhere**: Structured I/O for automation
4. **TUI optional**: Nice for humans, not in critical path

### Stack Decision

```
┌─────────────────────────────────────────────────────────────┐
│                         Server                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              GeneratorDriver                        │    │
│  │              (running on GPU)                       │    │
│  └──────────────────────┬──────────────────────────────┘    │
│                         │                                   │
│         ┌───────────────┼───────────────┐                   │
│         │               │               │                   │
│         ▼               ▼               ▼                   │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │  HTTP API  │  │ WebSocket  │  │   CLI      │            │
│  │  (JSON)    │  │ (frames)   │  │  (wrapper) │            │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘            │
└────────┼───────────────┼───────────────┼────────────────────┘
         │               │               │
         ▼               ▼               ▼
   ┌──────────┐   ┌──────────┐   ┌──────────┐
   │  Agent   │   │   Web    │   │  Human   │
   │ (Claude) │   │ Preview  │   │ Terminal │
   │          │   │(FastHTML)│   │          │
   └──────────┘   └──────────┘   └──────────┘
```

### CLI Interface

```bash
# Session management
video-cli state                    # Get current state (JSON)
video-cli sessions                 # List active sessions

# Generation control
video-cli step                     # Generate one chunk
video-cli run [--chunks N]         # Run N chunks (default: until paused)
video-cli pause                    # Pause generation

# Content control  
video-cli prompt "text"            # Set prompt
video-cli edit "instruction"       # Edit anchor frame
video-cli edit-preview "instr"     # Preview edit without applying

# Branching
video-cli snapshot                 # Create snapshot
video-cli snapshots                # List snapshots  
video-cli restore <id>             # Restore snapshot
video-cli fork                     # Fork from current state

# Inspection
video-cli describe-frame           # VLM description of current frame
video-cli frame [--out path.jpg]   # Save current frame
video-cli history                  # Show recent events
```

Example session:

```bash
$ video-cli state
{
  "mode": "paused",
  "chunk": 0,
  "prompt": null,
  "snapshots": []
}

$ video-cli prompt "a red ball bouncing on grass, sunny day"
{"status": "ok", "prompt": "a red ball bouncing on grass, sunny day"}

$ video-cli step
{
  "chunk": 1,
  "frames_generated": 3,
  "latency_ms": 287,
  "compiled_prompt": "stop motion puppet animation, a red ball..."
}

$ video-cli run --chunks 10
{"chunks_generated": 10, "final_chunk": 11, "total_ms": 2834}

$ video-cli edit "add a wooden fence in the background"
{
  "status": "edit_applied",
  "anchor_modified": true,
  "edit_latency_ms": 1200
}

$ video-cli step
{"chunk": 12, "frames_generated": 3, "latency_ms": 312}

$ video-cli describe-frame
{
  "description": "Red ball mid-bounce on grass with rustic wooden fence visible in background"
}
```

### CLI Implementation

```python
# scope/cli/video_cli.py
import click
import httpx
import json

BASE_URL = "http://localhost:8000"

def api(method: str, path: str, **kwargs) -> dict:
    """Make API call and return JSON."""
    r = getattr(httpx, method)(f"{BASE_URL}{path}", **kwargs)
    r.raise_for_status()
    return r.json()

def output(data: dict):
    """Pretty print JSON response."""
    click.echo(json.dumps(data, indent=2))

@click.group()
@click.option("--url", default=BASE_URL, envvar="VIDEO_API_URL")
def cli(url):
    global BASE_URL
    BASE_URL = url

@cli.command()
def state():
    """Get current session state."""
    output(api("get", "/api/state"))

@cli.command()
def step():
    """Generate one chunk."""
    output(api("post", "/api/step"))

@cli.command()
@click.option("--chunks", default=None, type=int)
def run(chunks):
    """Run generation."""
    params = {"chunks": chunks} if chunks else {}
    output(api("post", "/api/run", json=params))

@cli.command()
def pause():
    """Pause generation."""
    output(api("post", "/api/pause"))

@cli.command()
@click.argument("text")
def prompt(text):
    """Set prompt."""
    output(api("put", "/api/prompt", json={"prompt": text}))

@cli.command()
@click.argument("instruction")
def edit(instruction):
    """Edit anchor frame with instruction."""
    output(api("post", "/api/edit", json={"instruction": instruction}))

@cli.command("edit-preview")
@click.argument("instruction")
def edit_preview(instruction):
    """Preview edit without applying."""
    output(api("post", "/api/edit/preview", json={"instruction": instruction}))

@cli.command()
def snapshot():
    """Create snapshot of current state."""
    output(api("post", "/api/snapshot"))

@cli.command()
def snapshots():
    """List available snapshots."""
    output(api("get", "/api/snapshots"))

@cli.command()
@click.argument("snapshot_id")
def restore(snapshot_id):
    """Restore from snapshot."""
    output(api("post", f"/api/snapshot/{snapshot_id}/restore"))

@cli.command()
def fork():
    """Fork from current state."""
    output(api("post", "/api/fork"))

@cli.command("describe-frame")
def describe_frame():
    """Get VLM description of current frame."""
    output(api("get", "/api/frame/describe"))

@cli.command()
@click.option("--out", default=None, help="Output path for frame image")
def frame(out):
    """Get current frame."""
    if out:
        r = httpx.get(f"{BASE_URL}/api/frame/latest")
        with open(out, "wb") as f:
            f.write(r.content)
        click.echo(f"Saved to {out}")
    else:
        output(api("get", "/api/frame/latest/meta"))

@cli.command()
def history():
    """Show recent events."""
    output(api("get", "/api/history"))

if __name__ == "__main__":
    cli()
```

### Web Preview (FastHTML)

Minimal web interface for frame display:

```python
# scope/ui/web/app.py
from fasthtml.common import *

app, rt = fast_app(ws_hdr=True)

@rt("/")
def get():
    return Titled("Video Console",
        Div(
            # Header
            Div(
                Span("Session: ", Strong(id="session-id")),
                Span(" | State: ", Strong(id="state")),
                Span(" | Chunk: ", Strong(id="chunk")),
                cls="header",
                hx_get="/api/state",
                hx_trigger="every 1s",
                hx_swap="none",
                hx_on="htmx:afterRequest: updateHeader(event)"
            ),
            
            # Frame preview
            Div(
                Img(id="frame-preview", src="/api/frame/latest.jpg"),
                cls="preview",
                hx_get="/api/frame/latest.jpg",
                hx_trigger="every 500ms",
                hx_swap="outerHTML",
                hx_target="#frame-preview"
            ),
            
            # Simple controls (CLI is primary, these are convenience)
            Div(
                Button("Step", hx_post="/api/step", hx_swap="none"),
                Button("Run", hx_post="/api/run", hx_swap="none"),
                Button("Pause", hx_post="/api/pause", hx_swap="none"),
                cls="controls"
            ),
            
            # State dump
            Div(
                Pre(id="state-dump"),
                hx_get="/api/state",
                hx_trigger="every 2s",
                hx_target="#state-dump"
            ),
            
            cls="console"
        ),
        Style("""
            .console { font-family: monospace; padding: 20px; }
            .header { margin-bottom: 20px; }
            .preview { margin: 20px 0; }
            .preview img { max-width: 100%; border: 1px solid #333; }
            .controls { margin: 20px 0; }
            .controls button { margin-right: 10px; padding: 8px 16px; }
            #state-dump { background: #1a1a1a; color: #0f0; padding: 10px; }
        """)
    )

# WebSocket for real-time frame streaming (optional upgrade from polling)
@app.ws("/ws/frames")
async def ws_frames(ws):
    """Push frames to client as they generate."""
    # Subscribe to frame bus
    async for frame_data in frame_bus.subscribe():
        await ws.send(frame_data)
```

### API Endpoints (Addition to v1.1)

```python
# scope/api/routes/edit.py

from fastapi import APIRouter, HTTPException
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
    """
    Edit the anchor frame in decoded_frame_buffer.
    
    Uses configured image edit model (e.g., nano-banana) to apply
    semantic edit to anchor frame. Next generation will re-encode
    edited frame into KV cache.
    """
    driver = session.driver
    
    if driver.state != DriverState.PAUSED:
        raise HTTPException(400, "Must be paused to edit")
    
    # Get current anchor frame
    decoded_buffer = driver.pipeline.state.get('decoded_frame_buffer')
    if decoded_buffer is None or decoded_buffer.shape[1] < 1:
        raise HTTPException(400, "No frames in buffer yet")
    
    anchor_frame = decoded_buffer[:, :1].clone()  # [1, 1, 3, H, W]
    
    # Apply edit via image model
    import time
    start = time.perf_counter()
    
    edited_frame = await apply_edit(
        frame=anchor_frame,
        instruction=req.instruction,
        strength=req.strength
    )
    
    edit_ms = (time.perf_counter() - start) * 1000
    
    # Write back
    decoded_buffer[:, :1] = edited_frame
    driver.pipeline.state.set('decoded_frame_buffer', decoded_buffer)
    
    return EditResponse(
        status="edit_applied",
        anchor_modified=True,
        edit_latency_ms=edit_ms
    )

@router.post("/edit/preview")
async def preview_edit(req: EditRequest, session: Session = Depends(get_session)):
    """
    Preview edit without applying.
    
    Returns before/after images for comparison.
    """
    driver = session.driver
    
    decoded_buffer = driver.pipeline.state.get('decoded_frame_buffer')
    anchor_frame = decoded_buffer[:, :1].clone()
    
    edited_frame = await apply_edit(
        frame=anchor_frame,
        instruction=req.instruction,
        strength=req.strength
    )
    
    return {
        "before": frame_to_base64(anchor_frame),
        "after": frame_to_base64(edited_frame),
    }


async def apply_edit(frame: torch.Tensor, instruction: str, strength: float) -> torch.Tensor:
    """
    Apply semantic edit to frame.
    
    TODO: Integrate with nano-banana or other image edit model.
    
    Input: [1, 1, 3, H, W] tensor, values in [-1, 1] or [0, 1]
    Output: Same shape tensor with edit applied
    """
    # Placeholder - implement based on nano-banana interface
    # 
    # Likely something like:
    #   pil_image = tensor_to_pil(frame)
    #   edited_pil = nano_banana.edit(pil_image, instruction, strength)
    #   edited_tensor = pil_to_tensor(edited_pil)
    #   return edited_tensor
    
    raise NotImplementedError("Integrate image edit model here")
```

### GeneratorDriver Additions

```python
# Additions to GeneratorDriver class in scope/engine/generator_driver.py

class GeneratorDriver:
    # ... existing code from v1.1 ...
    
    def get_anchor_frame(self) -> torch.Tensor | None:
        """
        Get the current anchor frame from decoded buffer.
        
        Returns:
            Tensor of shape [1, 1, 3, H, W] or None if buffer empty
        """
        decoded_buffer = self.pipeline.state.get('decoded_frame_buffer')
        if decoded_buffer is None or decoded_buffer.shape[1] < 1:
            return None
        return decoded_buffer[:, :1].clone()
    
    def set_anchor_frame(self, frame: torch.Tensor) -> bool:
        """
        Replace the anchor frame in decoded buffer.
        
        Args:
            frame: Tensor of shape [1, 1, 3, H, W]
            
        Returns:
            True if successful, False if buffer not ready
        """
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
            "decoded_buffer_shape": list(decoded.shape) if decoded is not None else None,
            "context_buffer_shape": list(context.shape) if context is not None else None,
        }
```

---

## Part 4: Agent Integration

### Claude Code Workflow

Claude Code interacts via CLI commands with JSON responses:

```bash
# Example agent session (what Claude Code would execute)

$ video-cli prompt "a character walking through a forest"
{"status": "ok"}

$ video-cli run --chunks 10
{"chunks_generated": 10, "final_chunk": 10}

$ video-cli describe-frame
{"description": "A figure walking on a forest path, dappled sunlight, trees on both sides"}

$ video-cli snapshot
{"snapshot_id": "snap-10", "chunk": 10}

$ video-cli edit "make the character look worried"
{"status": "edit_applied", "anchor_modified": true, "edit_latency_ms": 1150}

$ video-cli step
{"chunk": 11, "frames_generated": 3}

$ video-cli describe-frame
{"description": "A worried-looking figure on a forest path, glancing around nervously"}
```

### Agent Loop Pattern

```python
# Conceptual agent loop (what Claude Code would effectively do)

def direct_scene(goal: str):
    """Agent directs video generation toward a goal."""
    
    # Initial setup
    run(f'video-cli prompt "{goal}"')
    
    for iteration in range(max_iterations):
        # Generate some content
        run("video-cli step")
        
        # Evaluate current state
        result = json.loads(run("video-cli describe-frame"))
        description = result["description"]
        
        # Decide next action based on description
        if "character not visible" in description.lower():
            run('video-cli edit "bring character to center of frame"')
            
        elif "too dark" in description.lower():
            run('video-cli edit "brighten lighting, make it daytime"')
            
        elif iteration == 10:  # Narrative beat
            run('video-cli snapshot')
            run('video-cli prompt "character stops and looks up at something in the sky"')
            
        elif iteration == 15:  # Introduce element
            run('video-cli edit "add a dragon silhouette in the distant sky"')
            run('video-cli prompt "character reacts with fear to the dragon"')
    
    # Export result
    run("video-cli export output.mp4")
```

### Frame Analysis Endpoint

For agent evaluation loop:

```python
# scope/api/routes/frame.py

@router.get("/frame/describe")
async def describe_frame(session: Session = Depends(get_session)):
    """
    Get VLM description of current frame.
    
    Used by agents to evaluate generation and decide next action.
    """
    driver = session.driver
    
    # Get latest frame
    frame = driver.get_latest_frame()
    if frame is None:
        raise HTTPException(400, "No frames generated yet")
    
    # Call VLM for description
    description = await vlm_describe(frame)
    
    return {"description": description}


async def vlm_describe(frame: torch.Tensor) -> str:
    """
    Get VLM description of frame.
    
    TODO: Integrate with available VLM (Gemini, etc.)
    """
    # Placeholder
    raise NotImplementedError("Integrate VLM here")
```

---

## Part 5: File Structure

```
scope/
├── cli/
│   ├── __init__.py
│   └── video_cli.py        # CLI tool
│
├── ui/
│   ├── __init__.py
│   └── web/
│       ├── __init__.py
│       ├── app.py          # FastHTML app
│       └── static/
│           └── style.css
│
├── api/
│   └── routes/
│       ├── sessions.py     # From v1.1
│       ├── control.py      # step, run, pause
│       ├── prompt.py       # prompt management
│       ├── edit.py         # anchor editing (NEW)
│       ├── frame.py        # frame access, VLM describe (NEW)
│       ├── snapshots.py    # branching
│       └── timeline.py     # history
│
├── engine/
│   ├── generator_driver.py # Updated with anchor access
│   ├── control_bus.py
│   ├── frame_bus.py
│   └── branch_graph.py
│
└── integrations/
    ├── __init__.py
    ├── image_edit.py       # nano-banana integration (NEW)
    └── vlm.py              # VLM integration (NEW)
```

---

## Part 6: Open Questions

### Image Edit Integration (nano-banana)

- [ ] What's the Python API? HTTP endpoint? Function call?
- [ ] Input format: PIL Image? Tensor? Base64?
- [ ] Output format: Same as input?
- [ ] Latency expectation: <1s? 1-3s?
- [ ] Strength/guidance controls available?

### File Structure in Scope

- [ ] Where should UI code live? `scope/ui/`? Separate package?
- [ ] How to run alongside existing Scope server?
- [ ] Entry point: `uv run video-cli`? `uv run video-console`?

### Web Preview

- [ ] Frame format: JPEG (smaller) or PNG (quality)?
- [ ] Polling vs WebSocket for frame updates?
- [ ] Any FastHTML patterns from nano-banana to reuse?

### Persistence

- [ ] Snapshots: Memory only or disk?
- [ ] Survive server restart?
- [ ] Export format for sharing?

### Day 5 Reality Check

- [ ] Is v1.1 architecture actually running?
- [ ] What generates frames right now?
- [ ] How much needs stubbing vs real implementation?

---

## Part 7: Build Order

### Phase 1: Validate Edit Mechanism (Day 1)

1. Run color tint experiment (`test_context_edit.py`)
2. Confirm propagation through KV cache
3. Document any timing/buffer constraints discovered

### Phase 2: Basic CLI + API (Days 2-3)

1. Implement core API endpoints:
   - `GET /api/state`
   - `POST /api/step`
   - `POST /api/run`
   - `POST /api/pause`
   - `PUT /api/prompt`
2. Implement `video-cli` wrapper
3. Test: can drive generation entirely from CLI

### Phase 3: Web Preview (Day 4)

1. FastHTML app with frame display
2. Polling-based frame updates
3. Basic state display
4. Test: can see frames while CLI controls

### Phase 4: Edit Integration (Days 5-6)

1. Integrate nano-banana (or placeholder)
2. Implement `/api/edit` endpoint
3. Implement `/api/edit/preview`
4. Add `get_anchor_frame` / `set_anchor_frame` to driver
5. Test: edit instruction → visible change in subsequent frames

### Phase 5: Frame Analysis (Day 7)

1. Integrate VLM for frame description
2. Implement `/api/frame/describe`
3. Test: agent can evaluate frames and make decisions

### Phase 6: Polish & Agent Patterns (Days 8+)

1. WebSocket frame streaming (upgrade from polling)
2. Better branching UI
3. Document agent patterns
4. Example agent scripts

---

## Appendix A: Frame Tensor Conventions

KREA pipeline uses:
- Shape: `[B, T, C, H, W]` for video tensors
- Values: Likely `[-1, 1]` after VAE decode (check this)
- dtype: `torch.bfloat16` typically

When interfacing with image edit models, may need conversion:
```python
def tensor_to_pil(t: torch.Tensor) -> Image:
    """Convert [1, 1, 3, H, W] tensor to PIL Image."""
    t = t.squeeze(0).squeeze(0)  # [3, H, W]
    t = (t + 1) / 2  # [-1,1] -> [0,1] (if needed)
    t = t.clamp(0, 1)
    t = (t * 255).byte()
    t = t.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
    return Image.fromarray(t)

def pil_to_tensor(img: Image, device, dtype) -> torch.Tensor:
    """Convert PIL Image to [1, 1, 3, H, W] tensor."""
    t = torch.from_numpy(np.array(img)).to(device=device, dtype=dtype)
    t = t.permute(2, 0, 1)  # [3, H, W]
    t = t / 255.0  # [0,255] -> [0,1]
    t = t * 2 - 1  # [0,1] -> [-1,1] (if needed)
    return t.unsqueeze(0).unsqueeze(0)  # [1, 1, 3, H, W]
```

## Appendix B: Related Documents

- `realtime_video_architecture.md` (v1.1) - Core architecture, primitives, API surface
- KREA pipeline source: `scope/core/pipelines/krea_realtime_video/`
- Scope documentation: existing usage guides

---

*Last updated: December 2024*
