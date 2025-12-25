"""CLI interface for Scope realtime video generation.

Designed for agent automation. All commands return JSON.
"""

import json
import sys
from pathlib import Path

import click
import httpx

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
        except Exception:
            error = {"error": response.text}
        click.echo(json.dumps(error), err=True)
        sys.exit(1)


@click.group()
@click.option("--url", envvar="VIDEO_API_URL", default=DEFAULT_URL, help="API base URL")
@click.option("--pretty/--no-pretty", default=True, help="Pretty print JSON output")
@click.pass_context
def cli(ctx, url, pretty):
    """Scope realtime video CLI - control video generation via REST API."""
    ctx.ensure_object(dict)
    ctx.obj["url"] = url
    ctx.obj["pretty"] = pretty


# --- State ---


@cli.command()
@click.pass_context
def state(ctx):
    """Get current session state."""
    with get_client(ctx) as client:
        r = client.get("/api/v1/realtime/state")
        handle_error(r)
        output(r.json(), ctx)


# --- Generation ---


@cli.command()
@click.pass_context
def step(ctx):
    """Generate one chunk."""
    with get_client(ctx) as client:
        r = client.post("/api/v1/realtime/step")
        handle_error(r)
        output(r.json(), ctx)


@cli.command()
@click.option("--chunks", type=int, default=None, help="Number of chunks to generate")
@click.pass_context
def run(ctx, chunks):
    """Start or run generation."""
    with get_client(ctx) as client:
        params = {"chunks": chunks} if chunks else {}
        r = client.post("/api/v1/realtime/run", params=params)
        handle_error(r)
        output(r.json(), ctx)


@cli.command()
@click.pass_context
def pause(ctx):
    """Pause generation."""
    with get_client(ctx) as client:
        r = client.post("/api/v1/realtime/pause")
        handle_error(r)
        output(r.json(), ctx)


# --- Prompt ---


@cli.command()
@click.argument("text", required=False)
@click.option("--get", "get_only", is_flag=True, help="Only get current prompt")
@click.pass_context
def prompt(ctx, text, get_only):
    """Set or get prompt."""
    with get_client(ctx) as client:
        if get_only or text is None:
            # Get current state which includes prompt
            r = client.get("/api/v1/realtime/state")
            handle_error(r)
            data = r.json()
            output({"prompt": data.get("prompt")}, ctx)
        else:
            r = client.put("/api/v1/realtime/prompt", json={"prompt": text})
            handle_error(r)
            output(r.json(), ctx)


# --- Frame ---


@cli.command()
@click.option("--out", type=click.Path(), help="Output file path")
@click.pass_context
def frame(ctx, out):
    """Get current frame."""
    with get_client(ctx) as client:
        if out:
            r = client.get("/api/v1/realtime/frame/latest")
            handle_error(r)
            Path(out).write_bytes(r.content)
            output({"saved": out, "size_bytes": len(r.content)}, ctx)
        else:
            # Just report that frame exists
            r = client.get("/api/v1/realtime/state")
            handle_error(r)
            output({"chunk_index": r.json().get("chunk_index")}, ctx)


# --- Snapshots (placeholder - uses existing WebRTC API indirectly) ---


@cli.command()
@click.pass_context
def snapshot(ctx):
    """Create snapshot (requires WebRTC session to handle response)."""
    # Note: Snapshot creation goes through update_parameters which
    # sends response via WebRTC data channel. For full REST support,
    # we'd need to add dedicated snapshot endpoints.
    output(
        {
            "status": "not_implemented",
            "message": "Snapshot creation requires dedicated REST endpoint",
        },
        ctx,
    )


@cli.command()
@click.argument("snapshot_id")
@click.pass_context
def restore(ctx, snapshot_id):
    """Restore from snapshot (requires WebRTC session to handle response)."""
    output(
        {
            "status": "not_implemented",
            "message": "Snapshot restore requires dedicated REST endpoint",
        },
        ctx,
    )


def main():
    cli()


if __name__ == "__main__":
    main()
