import asyncio
import importlib

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")


class FakePC:
    def __init__(self, *, connection_state: str):
        self.connectionState = connection_state


class FakeFrameProcessor:
    def __init__(self):
        self.paused = False
        self.chunk_index = 0
        self.parameters: dict = {"prompts": [{"text": "test prompt", "weight": 1.0}]}
        self._latest_frame = torch.zeros((1, 1, 3), dtype=torch.uint8)
        self.updates: list[dict] = []
        # Style layer fields (Phase 6a)
        self.world_state = None
        self.style_manifest = None
        self._compiled_prompt = None

    def update_parameters(self, parameters: dict) -> bool:
        self.updates.append(dict(parameters))
        return True

    def get_latest_frame(self):
        return self._latest_frame.clone()


class FakeVideoTrack:
    def __init__(self, fp: FakeFrameProcessor):
        self.frame_processor = fp
        self.pause_calls: list[bool] = []
        self.initialize_calls = 0

    def initialize_output_processing(self):
        self.initialize_calls += 1

    def pause(self, paused: bool):
        self.pause_calls.append(paused)


class FakeSession:
    def __init__(self, *, session_id: str, video_track: FakeVideoTrack):
        self.id = session_id
        self.pc = FakePC(connection_state="connected")
        self.video_track = video_track


class FakeWebRTCManager:
    def __init__(self, session: FakeSession):
        self.sessions = {session.id: session}


def _import_app_module(tmp_path, monkeypatch):
    # Prevent app import-time logging setup from touching real ~/.daydream-scope/logs.
    monkeypatch.setenv("DAYDREAM_SCOPE_LOGS_DIR", str(tmp_path / "logs"))
    import scope.server.app as app_mod

    return importlib.reload(app_mod)


def test_rest_state_pause_step_prompt_and_frame_latest(tmp_path, monkeypatch):
    app_mod = _import_app_module(tmp_path, monkeypatch)

    fp = FakeFrameProcessor()
    vt = FakeVideoTrack(fp)
    manager = FakeWebRTCManager(FakeSession(session_id="s", video_track=vt))

    state = asyncio.run(app_mod.get_realtime_state(webrtc_manager=manager))
    assert state.session_id == "s"
    assert state.prompt == "test prompt"
    assert state.chunk_index == 0

    pause_resp = asyncio.run(app_mod.pause_realtime(webrtc_manager=manager))
    assert pause_resp.status == "paused"
    assert vt.pause_calls[-1] is True
    assert fp.updates[-1] == {"paused": True}

    step_resp = asyncio.run(app_mod.step_realtime(webrtc_manager=manager))
    assert step_resp.status == "step_queued"
    assert fp.updates[-1] == {"paused": True, "_rcp_step": 1}

    prompt_req = app_mod.PromptRequest(prompt="new prompt")
    prompt_resp = asyncio.run(app_mod.set_realtime_prompt(prompt_req, webrtc_manager=manager))
    assert prompt_resp.status == "prompt_set"
    assert fp.updates[-1] == {"prompts": [{"text": "new prompt", "weight": 1.0}]}

    resp = asyncio.run(app_mod.get_latest_frame(webrtc_manager=manager))
    assert resp.media_type == "image/png"
    assert resp.body.startswith(b"\x89PNG\r\n\x1a\n")


def test_rest_run_rejects_negative_chunks(tmp_path, monkeypatch):
    app_mod = _import_app_module(tmp_path, monkeypatch)

    fp = FakeFrameProcessor()
    vt = FakeVideoTrack(fp)
    manager = FakeWebRTCManager(FakeSession(session_id="s", video_track=vt))

    with pytest.raises(app_mod.HTTPException) as excinfo:
        asyncio.run(app_mod.run_realtime(chunks=-1, webrtc_manager=manager))
    assert excinfo.value.status_code == 400
