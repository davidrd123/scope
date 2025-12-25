# Scope: Interface Specifications — CLI, Web, TUI

> **Purpose**: Detailed specs for all user/agent interfaces  
> **Complements**: `project_knowledge.md` (core feature knowledge)  
> **Primary Consumer**: Implementation agents, Claude Code

---

## 1. Interface Architecture

```
                    ┌─────────────────────────────────────┐
                    │         GeneratorDriver             │
                    │         (GPU, pipeline)             │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │         HTTP API            │
                    │         (FastAPI)           │
                    └──────────────┬──────────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         │                         │                         │
         ▼                         ▼                         ▼
   ┌───────────┐            ┌───────────┐            ┌───────────┐
   │    CLI    │            │    Web    │            │    TUI    │
   │           │            │           │            │           │
   │  • Agent  │            │  • Visual │            │  • Watch  │
   │  • Script │            │  • Edit   │            │  • Debug  │
   │  • Batch  │            │  • Browse │            │  • Live   │
   └───────────┘            └───────────┘            └───────────┘
        ▲                        ▲                        ▲
        │                        │                        │
   Claude Code              Human eyes              Human terminal
   Automation               Edit preview            Persistent view
```

### Design Principles

1. **Same API backend** — All interfaces hit identical endpoints
2. **JSON everywhere** — Structured I/O for automation
3. **CLI is canonical** — If it works in CLI, it works for agents
4. **Web for visuals** — Frames, previews, timeline need pixels
5. **TUI optional** — Nice for monitoring, not critical path

---

## 2. CLI Specification

### Installation & Invocation

```bash
# Install (assuming uv)
uv pip install -e ./scope

# Invoke
video-cli [command] [options]

# Or via uv run
uv run video-cli [command] [options]

# Environment
export VIDEO_API_URL=http://localhost:8000  # Default
```

### Command Reference

#### Session & State

```bash
# Get current state (JSON)
video-cli state
# Response: {"state": "paused", "chunk": 47, "prompt": "...", "buffer_frames": 9}

# List active sessions (if multi-session)
video-cli sessions
# Response: {"sessions": [{"id": "abc123", "state": "running", "chunk": 100}]}
```

#### Generation Control

```bash
# Generate single chunk
video-cli step
# Response: {"chunk": 48, "frames_generated": 3, "latency_ms": 287}

# Start continuous generation
video-cli run
# Response: {"status": "running"}

# Run fixed number of chunks
video-cli run --chunks 50
# Response: {"chunks_generated": 50, "final_chunk": 97, "total_ms": 14250}

# Pause generation
video-cli pause
# Response: {"status": "paused", "chunk": 97}
```

#### Prompt Control

```bash
# Set prompt
video-cli prompt "a red ball bouncing on grass, sunny day"
# Response: {"status": "ok", "prompt": "a red ball bouncing on grass, sunny day"}

# Get current prompt
video-cli prompt --get
# Response: {"prompt": "a red ball bouncing on grass, sunny day"}
```

#### Editing

```bash
# Edit anchor frame with semantic instruction
video-cli edit "add a wooden fence in the background"
# Response: {"status": "applied", "edit_latency_ms": 1150}

# Preview edit (returns before/after as base64 or saves to files)
video-cli edit-preview "make the sky purple" --out-before /tmp/before.png --out-after /tmp/after.png
# Response: {"status": "preview_ready", "before": "/tmp/before.png", "after": "/tmp/after.png"}

# Edit with strength parameter
video-cli edit "add dramatic lighting" --strength 0.9
# Response: {"status": "applied", "edit_latency_ms": 1320}
```

#### Frame Access

```bash
# Get current frame as image file
video-cli frame --out /tmp/current.png
# Saves image, Response: {"saved": "/tmp/current.png"}

# Get frame metadata only
video-cli frame --meta
# Response: {"chunk": 47, "timestamp": 1703421234567, "is_anchor": false}

# Get VLM description of current frame
video-cli describe-frame
# Response: {"description": "A red ball mid-bounce against green grass, wooden fence visible in background, warm sunlight"}

# Get anchor frame specifically
video-cli frame --anchor --out /tmp/anchor.png
# Response: {"saved": "/tmp/anchor.png", "is_anchor": true}
```

#### Snapshots & Branching

```bash
# Create snapshot
video-cli snapshot
# Response: {"snapshot_id": "snap-1703421234", "chunk": 47, "prompt": "..."}

# Create named snapshot
video-cli snapshot --name "before-edit-experiment"
# Response: {"snapshot_id": "snap-1703421234", "name": "before-edit-experiment"}

# List snapshots
video-cli snapshots
# Response: {"snapshots": [{"id": "snap-...", "name": "...", "chunk": 47, "timestamp": "..."}]}

# Restore snapshot
video-cli restore snap-1703421234
# Response: {"status": "restored", "chunk": 47}

# Fork (snapshot + continue from copy)
video-cli fork
# Response: {"fork_id": "fork-...", "parent_snapshot": "snap-...", "chunk": 47}

# Delete snapshot
video-cli snapshot-delete snap-1703421234
# Response: {"status": "deleted"}
```

#### Export

```bash
# Export video from current session
video-cli export output.mp4
# Response: {"status": "exported", "path": "output.mp4", "frames": 240, "duration_sec": 15.0}

# Export specific chunk range
video-cli export output.mp4 --start 10 --end 50
# Response: {"status": "exported", "path": "output.mp4", "frames": 120}

# Export with specific fps
video-cli export output.mp4 --fps 24
```

#### History & Debug

```bash
# Show recent events
video-cli history
# Response: {"events": [{"timestamp": "...", "type": "step", "chunk": 47}, ...]}

# Show buffer info
video-cli buffers
# Response: {"decoded_frames": 9, "context_frames": 2, "kv_blocks": 14}

# Show pipeline metrics
video-cli metrics
# Response: {"latency_ms": 287, "vram_gb": 18.4, "fps": 14.2}
```

### Full CLI Implementation

```python
# scope/cli/video_cli.py
"""
CLI interface for Scope video generation.
Designed for both human use and agent automation (Claude Code).
"""

import click
import httpx
import json
import sys
from pathlib import Path

DEFAULT_URL = "http://localhost:8000"

def get_client(ctx) -> httpx.Client:
    """Get HTTP client with base URL from context."""
    return httpx.Client(base_url=ctx.obj["url"], timeout=60.0)

def output(data: dict, ctx):
    """Output JSON response."""
    if ctx.obj.get("pretty"):
        click.echo(json.dumps(data, indent=2))
    else:
        click.echo(json.dumps(data))

def handle_error(response: httpx.Response):
    """Handle HTTP errors."""
    if response.status_code >= 400:
        try:
            error = response.json()
        except:
            error = {"error": response.text}
        click.echo(json.dumps(error), err=True)
        sys.exit(1)

@click.group()
@click.option("--url", envvar="VIDEO_API_URL", default=DEFAULT_URL, help="API base URL")
@click.option("--pretty/--no-pretty", default=True, help="Pretty print JSON")
@click.pass_context
def cli(ctx, url, pretty):
    """Scope video generation CLI."""
    ctx.ensure_object(dict)
    ctx.obj["url"] = url
    ctx.obj["pretty"] = pretty

# --- State Commands ---

@cli.command()
@click.pass_context
def state(ctx):
    """Get current session state."""
    with get_client(ctx) as client:
        r = client.get("/api/state")
        handle_error(r)
        output(r.json(), ctx)

@cli.command()
@click.pass_context
def sessions(ctx):
    """List active sessions."""
    with get_client(ctx) as client:
        r = client.get("/api/sessions")
        handle_error(r)
        output(r.json(), ctx)

# --- Generation Commands ---

@cli.command()
@click.pass_context
def step(ctx):
    """Generate one chunk."""
    with get_client(ctx) as client:
        r = client.post("/api/step")
        handle_error(r)
        output(r.json(), ctx)

@cli.command()
@click.option("--chunks", type=int, default=None, help="Number of chunks to generate")
@click.pass_context
def run(ctx, chunks):
    """Start or run generation."""
    with get_client(ctx) as client:
        payload = {"chunks": chunks} if chunks else {}
        r = client.post("/api/run", json=payload)
        handle_error(r)
        output(r.json(), ctx)

@cli.command()
@click.pass_context
def pause(ctx):
    """Pause generation."""
    with get_client(ctx) as client:
        r = client.post("/api/pause")
        handle_error(r)
        output(r.json(), ctx)

# --- Prompt Commands ---

@cli.command()
@click.argument("text", required=False)
@click.option("--get", "get_only", is_flag=True, help="Get current prompt")
@click.pass_context
def prompt(ctx, text, get_only):
    """Set or get prompt."""
    with get_client(ctx) as client:
        if get_only or text is None:
            r = client.get("/api/prompt")
        else:
            r = client.put("/api/prompt", json={"prompt": text})
        handle_error(r)
        output(r.json(), ctx)

# --- Edit Commands ---

@cli.command()
@click.argument("instruction")
@click.option("--strength", type=float, default=0.8, help="Edit strength 0-1")
@click.pass_context
def edit(ctx, instruction, strength):
    """Edit anchor frame with semantic instruction."""
    with get_client(ctx) as client:
        r = client.post("/api/edit", json={
            "instruction": instruction,
            "strength": strength
        })
        handle_error(r)
        output(r.json(), ctx)

@cli.command("edit-preview")
@click.argument("instruction")
@click.option("--out-before", type=click.Path(), help="Save before image")
@click.option("--out-after", type=click.Path(), help="Save after image")
@click.option("--strength", type=float, default=0.8)
@click.pass_context
def edit_preview(ctx, instruction, out_before, out_after, strength):
    """Preview edit without applying."""
    with get_client(ctx) as client:
        r = client.post("/api/edit/preview", json={
            "instruction": instruction,
            "strength": strength
        })
        handle_error(r)
        data = r.json()
        
        # Save images if paths provided
        if out_before and "before" in data:
            import base64
            img_data = base64.b64decode(data["before"])
            Path(out_before).write_bytes(img_data)
            data["before"] = out_before
            
        if out_after and "after" in data:
            import base64
            img_data = base64.b64decode(data["after"])
            Path(out_after).write_bytes(img_data)
            data["after"] = out_after
        
        output(data, ctx)

# --- Frame Commands ---

@cli.command()
@click.option("--out", type=click.Path(), help="Save frame to file")
@click.option("--meta", is_flag=True, help="Get metadata only")
@click.option("--anchor", is_flag=True, help="Get anchor frame")
@click.pass_context
def frame(ctx, out, meta, anchor):
    """Get current or anchor frame."""
    with get_client(ctx) as client:
        endpoint = "/api/frame/anchor" if anchor else "/api/frame/latest"
        
        if meta:
            r = client.get(f"{endpoint}/meta")
            handle_error(r)
            output(r.json(), ctx)
        elif out:
            r = client.get(endpoint)
            handle_error(r)
            Path(out).write_bytes(r.content)
            output({"saved": out}, ctx)
        else:
            r = client.get(f"{endpoint}/meta")
            handle_error(r)
            output(r.json(), ctx)

@cli.command("describe-frame")
@click.pass_context
def describe_frame(ctx):
    """Get VLM description of current frame."""
    with get_client(ctx) as client:
        r = client.get("/api/frame/describe")
        handle_error(r)
        output(r.json(), ctx)

# --- Snapshot Commands ---

@cli.command()
@click.option("--name", help="Snapshot name")
@click.pass_context
def snapshot(ctx, name):
    """Create snapshot of current state."""
    with get_client(ctx) as client:
        payload = {"name": name} if name else {}
        r = client.post("/api/snapshot", json=payload)
        handle_error(r)
        output(r.json(), ctx)

@cli.command()
@click.pass_context
def snapshots(ctx):
    """List available snapshots."""
    with get_client(ctx) as client:
        r = client.get("/api/snapshots")
        handle_error(r)
        output(r.json(), ctx)

@cli.command()
@click.argument("snapshot_id")
@click.pass_context
def restore(ctx, snapshot_id):
    """Restore from snapshot."""
    with get_client(ctx) as client:
        r = client.post(f"/api/snapshot/{snapshot_id}/restore")
        handle_error(r)
        output(r.json(), ctx)

@cli.command()
@click.pass_context
def fork(ctx):
    """Fork from current state."""
    with get_client(ctx) as client:
        r = client.post("/api/fork")
        handle_error(r)
        output(r.json(), ctx)

@cli.command("snapshot-delete")
@click.argument("snapshot_id")
@click.pass_context
def snapshot_delete(ctx, snapshot_id):
    """Delete a snapshot."""
    with get_client(ctx) as client:
        r = client.delete(f"/api/snapshot/{snapshot_id}")
        handle_error(r)
        output(r.json(), ctx)

# --- Export Commands ---

@cli.command()
@click.argument("output_path", type=click.Path())
@click.option("--start", type=int, help="Start chunk")
@click.option("--end", type=int, help="End chunk")
@click.option("--fps", type=int, default=16, help="Output FPS")
@click.pass_context
def export(ctx, output_path, start, end, fps):
    """Export video."""
    with get_client(ctx) as client:
        payload = {"fps": fps}
        if start is not None:
            payload["start"] = start
        if end is not None:
            payload["end"] = end
            
        r = client.post("/api/export", json=payload)
        handle_error(r)
        
        # Save video content
        Path(output_path).write_bytes(r.content)
        output({"status": "exported", "path": output_path}, ctx)

# --- Debug Commands ---

@cli.command()
@click.option("--limit", type=int, default=20, help="Number of events")
@click.pass_context
def history(ctx, limit):
    """Show recent events."""
    with get_client(ctx) as client:
        r = client.get("/api/history", params={"limit": limit})
        handle_error(r)
        output(r.json(), ctx)

@cli.command()
@click.pass_context
def buffers(ctx):
    """Show buffer info."""
    with get_client(ctx) as client:
        r = client.get("/api/buffers")
        handle_error(r)
        output(r.json(), ctx)

@cli.command()
@click.pass_context
def metrics(ctx):
    """Show pipeline metrics."""
    with get_client(ctx) as client:
        r = client.get("/api/metrics")
        handle_error(r)
        output(r.json(), ctx)

if __name__ == "__main__":
    cli()
```

### pyproject.toml Entry

```toml
[project.scripts]
video-cli = "scope.cli.video_cli:cli"
```

---

## 3. Agent Integration Patterns

### Claude Code Workflow

Claude Code executes CLI commands and parses JSON responses:

```bash
# Typical agent session

$ video-cli prompt "a character walking through a forest"
{"status": "ok", "prompt": "a character walking through a forest"}

$ video-cli run --chunks 10
{"chunks_generated": 10, "final_chunk": 10, "total_ms": 2870}

$ video-cli describe-frame
{"description": "A figure on a forest path, dappled sunlight filtering through trees"}

$ video-cli snapshot --name "forest-start"
{"snapshot_id": "snap-1703421234", "name": "forest-start", "chunk": 10}

$ video-cli edit "make the character look worried, add fog between trees"
{"status": "applied", "edit_latency_ms": 1450}

$ video-cli step
{"chunk": 11, "frames_generated": 3, "latency_ms": 295}

$ video-cli describe-frame
{"description": "A worried figure peers through misty forest, fog weaving between dark tree trunks"}
```

### Agent Decision Loop (Pseudocode)

```python
# What an agent effectively does

def direct_video(goal: str, max_iterations: int = 50):
    """Agent directs video generation toward a narrative goal."""
    
    # Initialize
    run_cli(f'video-cli prompt "{goal}"')
    
    for i in range(max_iterations):
        # Generate
        result = run_cli("video-cli step")
        chunk = json.loads(result)["chunk"]
        
        # Evaluate
        desc = json.loads(run_cli("video-cli describe-frame"))["description"]
        
        # Decide action based on description
        if "character not visible" in desc.lower():
            run_cli('video-cli edit "bring character to center of frame"')
            
        elif "too dark" in desc.lower():
            run_cli('video-cli edit "brighten the scene, add sunlight"')
            
        elif chunk == 20:  # Narrative beat
            run_cli('video-cli snapshot --name "act-1-end"')
            run_cli('video-cli prompt "character discovers a mysterious door"')
            
        elif chunk == 30:  # Introduce element
            run_cli('video-cli edit "add an ornate wooden door against a hillside"')
            
        elif "door" in desc.lower() and chunk > 30:
            run_cli('video-cli prompt "character approaches and opens the door"')
    
    # Export
    run_cli("video-cli export final.mp4")
```

### Batch Scripting

```bash
#!/bin/bash
# batch_generate.sh - Generate multiple variations

PROMPTS=(
    "a red ball bouncing on grass"
    "a blue cube sliding on ice"  
    "a green sphere rolling down a hill"
)

for i in "${!PROMPTS[@]}"; do
    echo "=== Variation $i ==="
    video-cli prompt "${PROMPTS[$i]}"
    video-cli run --chunks 30
    video-cli export "output_$i.mp4"
    video-cli snapshot --name "variation-$i"
done
```

---

## 4. Web UI Specification

### Purpose

- **Visual feedback** for edits (before/after preview)
- **Frame display** during generation
- **Timeline browsing** of buffer contents
- **Monitoring** while CLI/agent controls

### FastHTML Implementation

```python
# scope/ui/web/app.py
"""
Web console for Scope video generation.
Visual complement to CLI - same API, different interface.
"""

from fasthtml.common import *
import httpx

app, rt = fast_app(
    hdrs=[
        Style("""
            :root { --bg: #09090b; --surface: #18181b; --border: #27272a; --text: #fafafa; --muted: #71717a; }
            body { background: var(--bg); color: var(--text); font-family: system-ui, sans-serif; }
            .console { display: grid; grid-template-columns: 250px 1fr 350px; height: 100vh; }
            .sidebar { background: var(--surface); border-right: 1px solid var(--border); padding: 1rem; }
            .main { padding: 1.5rem; display: flex; flex-direction: column; gap: 1rem; }
            .panel { background: var(--surface); border: 1px solid var(--border); border-radius: 0.5rem; padding: 1rem; }
            .preview { flex: 1; display: flex; align-items: center; justify-content: center; background: #000; border-radius: 0.5rem; }
            .preview img { max-width: 100%; max-height: 100%; object-fit: contain; }
            .timeline { height: 100px; overflow-x: auto; display: flex; gap: 0.5rem; padding: 0.5rem; }
            .timeline-frame { width: 120px; flex-shrink: 0; border-radius: 0.25rem; overflow: hidden; border: 2px solid var(--border); cursor: pointer; }
            .timeline-frame.anchor { border-color: #6366f1; }
            .timeline-frame.selected { border-color: #22c55e; }
            .timeline-frame img { width: 100%; height: 100%; object-fit: cover; }
            .controls { display: flex; gap: 0.5rem; align-items: center; }
            .btn { padding: 0.5rem 1rem; border-radius: 0.375rem; border: none; cursor: pointer; font-weight: 500; }
            .btn-primary { background: #6366f1; color: white; }
            .btn-secondary { background: var(--border); color: var(--text); }
            .input { background: var(--bg); border: 1px solid var(--border); border-radius: 0.375rem; padding: 0.5rem; color: var(--text); width: 100%; }
            .textarea { min-height: 80px; resize: vertical; }
            .label { font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.25rem; }
            .status { display: flex; align-items: center; gap: 0.5rem; }
            .status-dot { width: 8px; height: 8px; border-radius: 50%; }
            .status-dot.running { background: #22c55e; animation: pulse 1s infinite; }
            .status-dot.paused { background: #71717a; }
            @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
            .log { font-family: monospace; font-size: 0.75rem; max-height: 150px; overflow-y: auto; }
            .log-entry { padding: 0.25rem 0; border-bottom: 1px solid var(--border); }
            .metrics { display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; }
            .metric { background: var(--bg); padding: 0.5rem; border-radius: 0.25rem; }
            .metric-value { font-size: 1.25rem; font-family: monospace; }
            .metric-label { font-size: 0.625rem; color: var(--muted); text-transform: uppercase; }
        """)
    ]
)

API_URL = "http://localhost:8000"

def api_get(path: str):
    """GET from API."""
    try:
        return httpx.get(f"{API_URL}{path}", timeout=10).json()
    except:
        return {"error": "API unavailable"}

def api_post(path: str, data: dict = None):
    """POST to API."""
    try:
        return httpx.post(f"{API_URL}{path}", json=data or {}, timeout=60).json()
    except Exception as e:
        return {"error": str(e)}

# --- Components ---

def sidebar():
    """Left sidebar with status and navigation."""
    return Div(
        H2("Scope Dev", style="margin-bottom: 1.5rem;"),
        Div(
            Div(cls="label")("Status"),
            Div(
                Div(cls="status-dot paused", id="status-dot"),
                Span("PAUSED", id="status-text"),
                cls="status"
            ),
            style="margin-bottom: 1rem;"
        ),
        Div(
            Div(cls="label")("Chunk"),
            Div("0", id="chunk-count", style="font-family: monospace; font-size: 1.5rem;"),
            style="margin-bottom: 1rem;"
        ),
        Div(
            Div(cls="label")("Buffer"),
            Div("0 frames", id="buffer-count", style="font-family: monospace;"),
        ),
        # Auto-refresh status
        hx_get="/api/status-fragment",
        hx_trigger="every 1s",
        hx_swap="innerHTML",
        cls="sidebar"
    )

def main_area():
    """Central preview and timeline."""
    return Div(
        # Controls bar
        Div(
            Div(
                Button("▶", cls="btn btn-primary", hx_post="/ui/run", hx_swap="none"),
                Button("⏸", cls="btn btn-secondary", hx_post="/ui/pause", hx_swap="none"),
                Button("⏭", cls="btn btn-secondary", hx_post="/ui/step", hx_swap="none"),
                cls="controls"
            ),
            Input(
                type="text",
                name="prompt",
                placeholder="Enter prompt...",
                cls="input",
                style="flex: 1; margin-left: 1rem;",
                hx_post="/ui/prompt",
                hx_trigger="change",
                hx_swap="none"
            ),
            Button("📸 Snapshot", cls="btn btn-secondary", hx_post="/ui/snapshot", hx_swap="none"),
            style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;"
        ),
        
        # Frame preview
        Div(
            Img(src="/api/frame/latest.jpg", id="preview-img", alt="Current frame"),
            cls="preview",
            hx_get="/ui/frame-img",
            hx_trigger="every 500ms",
            hx_swap="innerHTML"
        ),
        
        # Timeline
        Div(
            Div(cls="label")("Context Buffer"),
            Div(
                id="timeline-frames",
                cls="timeline",
                hx_get="/ui/timeline",
                hx_trigger="every 1s",
                hx_swap="innerHTML"
            ),
            cls="panel"
        ),
        cls="main"
    )

def right_panel():
    """Right panel with edit controls and metrics."""
    return Div(
        # Edit panel
        Div(
            Div(cls="label")("Retroactive Edit"),
            Div(
                Img(src="/api/frame/anchor.jpg", style="width: 80px; border-radius: 0.25rem;", id="anchor-thumb"),
                Div("Anchor Frame", style="font-size: 0.75rem; color: var(--muted);"),
                style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem;"
            ),
            Textarea(
                name="instruction",
                placeholder="e.g. 'add tree in background'",
                cls="input textarea",
                id="edit-instruction"
            ),
            Button(
                "Apply Edit",
                cls="btn btn-primary",
                style="width: 100%; margin-top: 0.5rem;",
                hx_post="/ui/edit",
                hx_include="#edit-instruction",
                hx_swap="none"
            ),
            P("Edits modify the anchor frame. KV cache recomputes on next generation.", 
              style="font-size: 0.625rem; color: var(--muted); margin-top: 0.5rem;"),
            cls="panel",
            style="margin-bottom: 1rem;"
        ),
        
        # Metrics panel
        Div(
            Div(cls="label")("Pipeline Metrics"),
            Div(
                Div(
                    Div("287", cls="metric-value", id="metric-latency"),
                    Div("Latency (ms)", cls="metric-label"),
                    cls="metric"
                ),
                Div(
                    Div("18.4", cls="metric-value", id="metric-vram"),
                    Div("VRAM (GB)", cls="metric-label"),
                    cls="metric"
                ),
                Div(
                    Div("9", cls="metric-value", id="metric-buffer"),
                    Div("Buffer Frames", cls="metric-label"),
                    cls="metric"
                ),
                Div(
                    Div("14", cls="metric-value", id="metric-kv"),
                    Div("KV Blocks", cls="metric-label"),
                    cls="metric"
                ),
                cls="metrics"
            ),
            cls="panel",
            style="margin-bottom: 1rem;"
        ),
        
        # Log panel
        Div(
            Div(cls="label")("Console Log"),
            Div(id="log-entries", cls="log"),
            cls="panel"
        ),
        
        style="width: 350px; padding: 1.5rem; border-left: 1px solid var(--border); overflow-y: auto;"
    )

# --- Routes ---

@rt("/")
def get():
    """Main console page."""
    return Titled(
        "Scope Console",
        Div(
            sidebar(),
            main_area(),
            right_panel(),
            cls="console"
        )
    )

@rt("/ui/step")
def post():
    """Step generation."""
    result = api_post("/api/step")
    return ""

@rt("/ui/run")
def post():
    """Start running."""
    result = api_post("/api/run")
    return ""

@rt("/ui/pause")
def post():
    """Pause generation."""
    result = api_post("/api/pause")
    return ""

@rt("/ui/prompt")
def post(prompt: str):
    """Set prompt."""
    result = api_post("/api/prompt", {"prompt": prompt})
    return ""

@rt("/ui/edit")
def post(instruction: str):
    """Apply edit."""
    result = api_post("/api/edit", {"instruction": instruction})
    return ""

@rt("/ui/snapshot")
def post():
    """Create snapshot."""
    result = api_post("/api/snapshot")
    return ""

@rt("/ui/frame-img")
def get():
    """Get current frame image tag."""
    return Img(src=f"/api/frame/latest.jpg?t={time.time()}", alt="Current frame")

@rt("/ui/timeline")
def get():
    """Get timeline frames."""
    # This would fetch from API and render frame thumbnails
    state = api_get("/api/state")
    buffer_frames = state.get("buffer_frames", 0)
    
    frames = []
    for i in range(buffer_frames):
        is_anchor = i == 0
        frames.append(
            Div(
                Img(src=f"/api/frame/{i}.jpg"),
                cls=f"timeline-frame {'anchor' if is_anchor else ''}"
            )
        )
    return frames

# --- Run ---

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
```

### Running Web UI

```bash
# Start API server (port 8000)
uv run python -m scope.api.main

# Start web UI (port 8001)
uv run python -m scope.ui.web.app

# Access at http://localhost:8001
```

---

## 5. TUI Specification (Optional)

### Purpose

Persistent terminal view for monitoring without switching windows. Useful for:
- Watching generation while doing other terminal work
- SSH sessions where browser isn't available
- Debugging pipeline state

### Implementation Options

**Option A: Textual (Python)**
```python
# scope/ui/tui/console.py
from textual.app import App
from textual.widgets import Header, Footer, Static, Log
from textual.containers import Container, Horizontal

class ScopeConsole(App):
    CSS = """
    #status { dock: left; width: 20; }
    #preview { dock: top; height: 50%; }
    #log { dock: bottom; height: 50%; }
    """
    
    def compose(self):
        yield Header()
        yield Container(
            Static(id="status"),
            Static(id="preview"),  # ASCII art frame preview
            Log(id="log"),
        )
        yield Footer()
    
    def on_mount(self):
        self.set_interval(1.0, self.refresh_state)
    
    async def refresh_state(self):
        # Fetch and update display
        pass
```

**Option B: Rich Live Display**
```python
# Simpler, less interactive
from rich.live import Live
from rich.table import Table
from rich.panel import Panel

def watch_state():
    with Live(refresh_per_second=2) as live:
        while True:
            state = fetch_state()
            table = Table()
            table.add_column("Metric")
            table.add_column("Value")
            table.add_row("State", state["state"])
            table.add_row("Chunk", str(state["chunk"]))
            table.add_row("Buffer", f"{state['buffer_frames']} frames")
            live.update(Panel(table, title="Scope"))
            time.sleep(0.5)
```

### Priority

TUI is **nice to have**, not critical path. Build if:
- Web preview has issues on remote GPU setup
- Agent monitoring is needed in pure-terminal context
- Time permits after core features

---

## 6. Interface Testing

### CLI Test Script

```bash
#!/bin/bash
# test_cli.sh - Verify CLI works end-to-end

set -e

echo "=== Testing CLI ==="

echo "1. State check"
video-cli state

echo "2. Set prompt"
video-cli prompt "test scene"

echo "3. Step"
video-cli step

echo "4. Describe frame"
video-cli describe-frame

echo "5. Create snapshot"
video-cli snapshot --name "test"

echo "6. List snapshots"
video-cli snapshots

echo "=== All tests passed ==="
```

### Web UI Test Checklist

- [ ] Page loads without error
- [ ] Status updates automatically
- [ ] Play/pause buttons work
- [ ] Step button generates frame
- [ ] Frame preview updates
- [ ] Edit instruction submits
- [ ] Timeline shows frames

---

## 7. File Structure

```
scope/
├── cli/
│   ├── __init__.py
│   └── video_cli.py      # Full CLI implementation
│
├── ui/
│   ├── __init__.py
│   ├── web/
│   │   ├── __init__.py
│   │   ├── app.py        # FastHTML app
│   │   └── static/       # Optional static assets
│   │
│   └── tui/              # Optional
│       ├── __init__.py
│       └── console.py    # Textual or Rich app
│
├── api/
│   ├── __init__.py
│   ├── main.py           # FastAPI app
│   ├── schemas.py        # Pydantic models
│   └── routes/
│       ├── state.py
│       ├── control.py
│       ├── prompt.py
│       ├── edit.py
│       ├── frame.py
│       └── snapshot.py
```

---

*This document specifies all user/agent interfaces. Implementation order: CLI first (agent automation), Web second (visual feedback), TUI if time permits.*
