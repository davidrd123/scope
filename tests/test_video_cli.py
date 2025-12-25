import json
from pathlib import Path

from click.testing import CliRunner

from scope.cli import video_cli


class _FakeResponse:
    def __init__(self, *, status_code: int = 200, json_data=None, content: bytes = b""):
        self.status_code = status_code
        self._json_data = json_data if json_data is not None else {}
        self.content = content
        self.text = json.dumps(self._json_data)

    def json(self):
        return self._json_data


class _FakeClient:
    def __init__(self, responses: dict[tuple[str, str], _FakeResponse], **_kwargs):
        self._responses = responses
        self.calls: list[tuple[str, str]] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, path: str, **_kwargs):
        self.calls.append(("GET", path))
        return self._responses[("GET", path)]

    def post(self, path: str, **_kwargs):
        self.calls.append(("POST", path))
        return self._responses[("POST", path)]

    def put(self, path: str, **_kwargs):
        self.calls.append(("PUT", path))
        return self._responses[("PUT", path)]


def test_video_cli_state(monkeypatch):
    responses = {
        ("GET", "/api/v1/realtime/state"): _FakeResponse(
            json_data={"paused": False, "chunk_index": 12, "prompt": "hi", "session_id": "s"}
        )
    }
    monkeypatch.setattr(video_cli.httpx, "Client", lambda **kwargs: _FakeClient(responses, **kwargs))

    runner = CliRunner()
    result = runner.invoke(video_cli.cli, ["--url", "http://example", "state"])
    assert result.exit_code == 0
    assert '"chunk_index": 12' in result.output


def test_video_cli_prompt_set(monkeypatch):
    responses = {
        ("PUT", "/api/v1/realtime/prompt"): _FakeResponse(json_data={"status": "prompt_set", "chunk_index": 1})
    }
    monkeypatch.setattr(video_cli.httpx, "Client", lambda **kwargs: _FakeClient(responses, **kwargs))

    runner = CliRunner()
    result = runner.invoke(video_cli.cli, ["prompt", "a red ball"])
    assert result.exit_code == 0
    assert '"status": "prompt_set"' in result.output


def test_video_cli_frame_writes_file(tmp_path, monkeypatch):
    out_path = tmp_path / "frame.png"

    responses = {
        ("GET", "/api/v1/realtime/frame/latest"): _FakeResponse(content=b"PNGDATA"),
    }
    monkeypatch.setattr(video_cli.httpx, "Client", lambda **kwargs: _FakeClient(responses, **kwargs))

    runner = CliRunner()
    result = runner.invoke(video_cli.cli, ["frame", "--out", str(out_path)])
    assert result.exit_code == 0
    assert out_path.read_bytes() == b"PNGDATA"
    assert '"saved":' in result.output

