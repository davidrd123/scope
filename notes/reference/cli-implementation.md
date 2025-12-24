# CLI Implementation - Code Reference

Extracted from `incoming/interface_specifications.md`.

---

## Full CLI Implementation

```python
# scope/cli/video_cli.py
"""CLI interface for Scope video generation. Designed for agent automation."""

import click
import httpx
import json
import sys
from pathlib import Path

DEFAULT_URL = "http://localhost:8000"

def get_client(ctx) -> httpx.Client:
    return httpx.Client(base_url=ctx.obj["url"], timeout=60.0)

def output(data: dict, ctx):
    if ctx.obj.get("pretty"):
        click.echo(json.dumps(data, indent=2))
    else:
        click.echo(json.dumps(data))

def handle_error(response: httpx.Response):
    if response.status_code >= 400:
        try:
            error = response.json()
        except:
            error = {"error": response.text}
        click.echo(json.dumps(error), err=True)
        sys.exit(1)

@click.group()
@click.option("--url", envvar="VIDEO_API_URL", default=DEFAULT_URL)
@click.option("--pretty/--no-pretty", default=True)
@click.pass_context
def cli(ctx, url, pretty):
    ctx.ensure_object(dict)
    ctx.obj["url"] = url
    ctx.obj["pretty"] = pretty

# --- State ---

@cli.command()
@click.pass_context
def state(ctx):
    """Get current session state."""
    with get_client(ctx) as client:
        r = client.get("/api/state")
        handle_error(r)
        output(r.json(), ctx)

# --- Generation ---

@cli.command()
@click.pass_context
def step(ctx):
    """Generate one chunk."""
    with get_client(ctx) as client:
        r = client.post("/api/step")
        handle_error(r)
        output(r.json(), ctx)

@cli.command()
@click.option("--chunks", type=int, default=None)
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

# --- Prompt ---

@cli.command()
@click.argument("text", required=False)
@click.option("--get", "get_only", is_flag=True)
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

# --- Edit ---

@cli.command()
@click.argument("instruction")
@click.option("--strength", type=float, default=0.8)
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

# --- Frame ---

@cli.command()
@click.option("--out", type=click.Path())
@click.option("--anchor", is_flag=True)
@click.pass_context
def frame(ctx, out, anchor):
    """Get current or anchor frame."""
    with get_client(ctx) as client:
        endpoint = "/api/frame/anchor" if anchor else "/api/frame/latest"
        if out:
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

# --- Snapshots ---

@cli.command()
@click.option("--name")
@click.pass_context
def snapshot(ctx, name):
    """Create snapshot."""
    with get_client(ctx) as client:
        payload = {"name": name} if name else {}
        r = client.post("/api/snapshot", json=payload)
        handle_error(r)
        output(r.json(), ctx)

@cli.command()
@click.pass_context
def snapshots(ctx):
    """List snapshots."""
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

# --- Export ---

@cli.command()
@click.argument("output_path", type=click.Path())
@click.option("--start", type=int)
@click.option("--end", type=int)
@click.option("--fps", type=int, default=16)
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
        Path(output_path).write_bytes(r.content)
        output({"status": "exported", "path": output_path}, ctx)

# --- Debug ---

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

---

## pyproject.toml Entry

```toml
[project.scripts]
video-cli = "scope.cli.video_cli:cli"
```

---

## Example Session

```bash
$ video-cli state
{"state": "paused", "chunk": 0, "prompt": null}

$ video-cli prompt "a red ball bouncing on grass"
{"status": "ok", "prompt": "a red ball bouncing on grass"}

$ video-cli step
{"chunk": 1, "frames_generated": 3, "latency_ms": 287}

$ video-cli run --chunks 10
{"chunks_generated": 10, "final_chunk": 11, "total_ms": 2834}

$ video-cli edit "add a wooden fence in background"
{"status": "applied", "edit_latency_ms": 1200}

$ video-cli describe-frame
{"description": "Red ball mid-bounce on grass with wooden fence visible"}

$ video-cli snapshot --name "with-fence"
{"snapshot_id": "snap-123", "name": "with-fence", "chunk": 11}

$ video-cli export output.mp4
{"status": "exported", "path": "output.mp4"}
```

---

## Agent Loop Pattern

```python
def direct_video(goal: str, max_iterations: int = 50):
    """Agent directs video generation toward a goal."""
    run_cli(f'video-cli prompt "{goal}"')

    for i in range(max_iterations):
        run_cli("video-cli step")
        desc = json.loads(run_cli("video-cli describe-frame"))["description"]

        if "character not visible" in desc.lower():
            run_cli('video-cli edit "bring character to center"')
        elif "too dark" in desc.lower():
            run_cli('video-cli edit "brighten lighting"')
        elif i == 20:
            run_cli('video-cli snapshot --name "act-1"')
            run_cli('video-cli prompt "character discovers door"')

    run_cli("video-cli export final.mp4")
```
