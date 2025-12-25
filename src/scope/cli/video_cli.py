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


# --- World State ---


@cli.command()
@click.argument("json_data", required=False)
@click.option("--get", "get_only", is_flag=True, help="Only get current world state")
@click.pass_context
def world(ctx, json_data, get_only):
    """Set or get WorldState.

    Examples:
        video-cli world                              # Get current world state
        video-cli world '{"action":"run"}'           # Set world state
    """
    with get_client(ctx) as client:
        if get_only or json_data is None:
            r = client.get("/api/v1/realtime/state")
            handle_error(r)
            data = r.json()
            output({"world_state": data.get("world_state")}, ctx)
        else:
            try:
                world_state = json.loads(json_data)
            except json.JSONDecodeError as e:
                click.echo(json.dumps({"error": f"Invalid JSON: {e}"}), err=True)
                sys.exit(1)
            r = client.put("/api/v1/realtime/world", json={"world_state": world_state})
            handle_error(r)
            output(r.json(), ctx)


# --- Style ---


@cli.group()
@click.pass_context
def style(ctx):
    """Manage active style."""
    pass


@style.command("list")
@click.pass_context
def style_list(ctx):
    """List available styles."""
    with get_client(ctx) as client:
        r = client.get("/api/v1/realtime/style/list")
        handle_error(r)
        output(r.json(), ctx)


@style.command("set")
@click.argument("name")
@click.pass_context
def style_set(ctx, name):
    """Set active style by name."""
    with get_client(ctx) as client:
        r = client.put("/api/v1/realtime/style", json={"name": name})
        handle_error(r)
        output(r.json(), ctx)


@style.command("get")
@click.pass_context
def style_get(ctx):
    """Get currently active style."""
    with get_client(ctx) as client:
        r = client.get("/api/v1/realtime/state")
        handle_error(r)
        data = r.json()
        output(
            {
                "active_style": data.get("active_style"),
                "compiled_prompt": data.get("compiled_prompt"),
            },
            ctx,
        )


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


# --- Playlist ---


@cli.group()
@click.pass_context
def playlist(ctx):
    """Manage prompt playlist from caption files."""
    pass


@playlist.command("load")
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--swap", nargs=2, help="Trigger swap: OLD NEW")
@click.pass_context
def playlist_load(ctx, file_path, swap):
    """Load prompts from a caption file.

    Examples:
        video-cli playlist load captions.txt
        video-cli playlist load captions.txt --swap "1988 Cel Animation" "Rankin/Bass Animagic Stop-Motion"
    """
    with get_client(ctx) as client:
        payload = {"file_path": str(Path(file_path).absolute())}
        if swap:
            payload["old_trigger"] = swap[0]
            payload["new_trigger"] = swap[1]
        r = client.post("/api/v1/realtime/playlist/load", json=payload)
        handle_error(r)
        output(r.json(), ctx)


@playlist.command("status")
@click.pass_context
def playlist_status(ctx):
    """Get current playlist state."""
    with get_client(ctx) as client:
        r = client.get("/api/v1/realtime/playlist")
        handle_error(r)
        output(r.json(), ctx)


@playlist.command("preview")
@click.option("--context", "-c", type=int, default=2, help="Lines of context around current")
@click.pass_context
def playlist_preview(ctx, context):
    """Preview prompts around current position."""
    with get_client(ctx) as client:
        r = client.get("/api/v1/realtime/playlist/preview", params={"context": context})
        handle_error(r)
        output(r.json(), ctx)


@playlist.command("next")
@click.option("--apply/--no-apply", default=True, help="Apply prompt after navigating")
@click.pass_context
def playlist_next(ctx, apply):
    """Move to next prompt."""
    with get_client(ctx) as client:
        r = client.post("/api/v1/realtime/playlist/next", params={"apply": apply})
        handle_error(r)
        output(r.json(), ctx)


@playlist.command("prev")
@click.option("--apply/--no-apply", default=True, help="Apply prompt after navigating")
@click.pass_context
def playlist_prev(ctx, apply):
    """Move to previous prompt."""
    with get_client(ctx) as client:
        r = client.post("/api/v1/realtime/playlist/prev", params={"apply": apply})
        handle_error(r)
        output(r.json(), ctx)


@playlist.command("goto")
@click.argument("index", type=int)
@click.option("--apply/--no-apply", default=True, help="Apply prompt after navigating")
@click.pass_context
def playlist_goto(ctx, index, apply):
    """Go to a specific prompt index."""
    with get_client(ctx) as client:
        r = client.post("/api/v1/realtime/playlist/goto", json={"index": index}, params={"apply": apply})
        handle_error(r)
        output(r.json(), ctx)


@playlist.command("apply")
@click.pass_context
def playlist_apply(ctx):
    """Apply current prompt to generation."""
    with get_client(ctx) as client:
        r = client.post("/api/v1/realtime/playlist/apply")
        handle_error(r)
        output(r.json(), ctx)


@playlist.command("nav")
@click.pass_context
def playlist_nav(ctx):
    """Interactive navigation mode with autoplay.

    Controls:
        →, n, l, SPACE  Next prompt
        ←, p            Previous prompt (stops autoplay)
        o               Toggle autoplay (default 5s interval)
        H               Toggle hard cut mode (reset cache on each transition)
        +/-             Adjust autoplay speed (1-30s)
        g               Go to index (prompts for number)
        a               Apply current prompt
        r               Refresh display
        q, ESC          Quit

    Changes are auto-applied by default.
    Hard cut mode resets the KV cache on each prompt change for clean scene transitions.
    """
    import os
    import select
    import termios
    import time
    import tty

    def get_char_nonblocking(timeout=0.2):
        """Read a char with timeout. Returns None if no input."""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            if select.select([fd], [], [], timeout)[0]:
                ch = os.read(fd, 1).decode("utf-8", errors="ignore")
                if ch == "\x1b":
                    extra = ""
                    for _ in range(5):
                        if select.select([fd], [], [], 0.05)[0]:
                            byte = os.read(fd, 1).decode("utf-8", errors="ignore")
                            extra += byte
                            if len(extra) >= 2 and extra[0] == "[" and extra[-1] in "ABCD":
                                break
                        else:
                            break
                    ch = ch + extra
                return ch
            return None
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def display_preview(client, autoplay=False, interval=5.0, hard_cut=False):
        """Fetch and display preview."""
        import shutil

        term_width = shutil.get_terminal_size().columns

        r = client.get("/api/v1/realtime/playlist/preview", params={"context": 3})
        if r.status_code != 200:
            return None
        data = r.json()
        if data.get("status") == "no_playlist":
            click.echo("\n  No playlist loaded. Use: video-cli playlist load <file>\n")
            return None

        click.echo("\n" + "=" * term_width)
        status = f"  Playlist: {data.get('total', 0)} prompts"
        if autoplay:
            status += f"  [▶ AUTO {interval}s]"
        if hard_cut:
            status += "  [✂ HARD CUT]"
        click.echo(status)
        click.echo("=" * term_width)

        # Calculate prompt display width
        prompt_width = term_width - 10

        for item in data.get("prompts", []):
            marker = "▶ " if item.get("current") else "  "
            idx = item.get("index", 0)
            prompt = item.get("prompt", "")[:prompt_width]
            if item.get("current"):
                click.echo(
                    click.style(f"{marker}[{idx:3d}] {prompt}", fg="green", bold=True)
                )
            else:
                click.echo(f"{marker}[{idx:3d}] {prompt}")

        click.echo("=" * term_width)
        click.echo("  ←/→ nav | o auto | H hard cut | +/- speed | g goto | q quit")
        click.echo("=" * term_width + "\n")
        return data

    click.echo("\nPlaylist Navigation Mode")
    click.echo("Press q or ESC to quit\n")

    # Autoplay state
    autoplay = False
    autoplay_interval = 5.0
    last_advance = time.time()

    # Hard cut state - when enabled, all transitions reset the KV cache
    hard_cut = False

    with get_client(ctx) as client:
        if display_preview(client, autoplay, autoplay_interval, hard_cut) is None:
            return

        while True:
            try:
                ch = get_char_nonblocking(timeout=0.2)

                if ch is not None:
                    # Quit
                    if ch in ("q", "Q", "\x03"):
                        click.echo("\nExiting navigation mode.\n")
                        break
                    elif ch == "\x1b" and len(ch) == 1:
                        click.echo("\nExiting navigation mode.\n")
                        break

                    # Toggle autoplay
                    elif ch == "o":
                        autoplay = not autoplay
                        last_advance = time.time()
                        display_preview(client, autoplay, autoplay_interval, hard_cut)

                    # Toggle hard cut mode
                    elif ch == "H":
                        hard_cut = not hard_cut
                        status = "ON - transitions will reset cache" if hard_cut else "OFF"
                        click.echo(f"  ✂ Hard cut: {status}")
                        display_preview(client, autoplay, autoplay_interval, hard_cut)

                    # Adjust speed
                    elif ch in ("+", "=", "]"):
                        autoplay_interval = max(1.0, autoplay_interval - 1.0)
                        click.echo(f"  Interval: {autoplay_interval}s")
                    elif ch in ("-", "_", "["):
                        autoplay_interval = min(30.0, autoplay_interval + 1.0)
                        click.echo(f"  Interval: {autoplay_interval}s")

                    # Next
                    elif ch in ("\x1b[C", "n", "l", " "):
                        r = client.post(
                            "/api/v1/realtime/playlist/next",
                            params={"apply": True, "hard_cut": hard_cut},
                        )
                        if r.status_code == 200:
                            display_preview(client, autoplay, autoplay_interval, hard_cut)
                        last_advance = time.time()

                    # Previous (stops autoplay)
                    elif ch in ("\x1b[D", "p"):
                        r = client.post(
                            "/api/v1/realtime/playlist/prev",
                            params={"apply": True, "hard_cut": hard_cut},
                        )
                        if r.status_code == 200:
                            display_preview(client, autoplay, autoplay_interval, hard_cut)
                        last_advance = time.time()
                        if autoplay:
                            autoplay = False
                            click.echo("  ⏸ Autoplay stopped")

                    # Goto
                    elif ch == "g":
                        click.echo("\nGoto index: ", nl=False)
                        fd = sys.stdin.fileno()
                        old_settings = termios.tcgetattr(fd)
                        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                        try:
                            idx_str = input()
                            idx = int(idx_str)
                            r = client.post(
                                "/api/v1/realtime/playlist/goto",
                                json={"index": idx},
                                params={"apply": True, "hard_cut": hard_cut},
                            )
                            if r.status_code == 200:
                                display_preview(client, autoplay, autoplay_interval, hard_cut)
                            last_advance = time.time()
                        except ValueError:
                            click.echo("Invalid index")
                        except EOFError:
                            pass

                    # Apply (with hard cut if enabled)
                    elif ch == "a":
                        r = client.post(
                            "/api/v1/realtime/playlist/apply",
                            params={"hard_cut": hard_cut},
                        )
                        if r.status_code == 200:
                            msg = "✓ Prompt applied"
                            if hard_cut:
                                msg += " (hard cut)"
                            click.echo(f"  {msg}")

                    # Refresh
                    elif ch == "r":
                        display_preview(client, autoplay, autoplay_interval, hard_cut)

                # Autoplay advance
                if autoplay and (time.time() - last_advance) >= autoplay_interval:
                    r = client.post(
                        "/api/v1/realtime/playlist/next",
                        params={"apply": True, "hard_cut": hard_cut},
                    )
                    if r.status_code == 200:
                        data = r.json()
                        if not data.get("has_next", False):
                            autoplay = False
                            click.echo("  ⏹ End of playlist")
                        display_preview(client, autoplay, autoplay_interval, hard_cut)
                    last_advance = time.time()

            except KeyboardInterrupt:
                click.echo("\nExiting navigation mode.\n")
                break


def main():
    cli()


if __name__ == "__main__":
    main()
