import queue

import pytest

torch = pytest.importorskip("torch")

try:
    from scope.server.frame_processor import FrameProcessor
    from scope.server.webrtc import apply_control_message, get_active_session
except Exception as e:  # pragma: no cover
    pytest.skip(f"Required server modules not available: {e}", allow_module_level=True)


class FakePC:
    def __init__(self, *, connection_state: str):
        self.connectionState = connection_state


class FakeFrameProcessor:
    def __init__(self, *, accept_updates: bool = True):
        self.accept_updates = accept_updates
        self.updates: list[dict] = []

    def update_parameters(self, parameters: dict) -> bool:
        self.updates.append(dict(parameters))
        return self.accept_updates


class FakeVideoTrack:
    def __init__(self, frame_processor: FakeFrameProcessor | None = None):
        self.frame_processor = frame_processor
        self.initialize_calls = 0
        self.pause_calls: list[bool] = []

    def initialize_output_processing(self):
        self.initialize_calls += 1
        if self.frame_processor is None:
            self.frame_processor = FakeFrameProcessor()

    def pause(self, paused: bool):
        self.pause_calls.append(paused)


class FakeSession:
    def __init__(self, *, session_id: str, pc: FakePC, video_track: FakeVideoTrack | None):
        self.id = session_id
        self.pc = pc
        self.video_track = video_track


class FakeWebRTCManager:
    def __init__(self, sessions: dict[str, FakeSession]):
        self.sessions = sessions


def test_get_active_session_requires_exactly_one_connected_session():
    manager = FakeWebRTCManager(
        sessions={
            "a": FakeSession(
                session_id="a",
                pc=FakePC(connection_state="connecting"),
                video_track=FakeVideoTrack(),
            )
        }
    )
    with pytest.raises(ValueError, match="No active WebRTC session"):
        get_active_session(manager)  # type: ignore[arg-type]

    manager = FakeWebRTCManager(
        sessions={
            "a": FakeSession(
                session_id="a",
                pc=FakePC(connection_state="connected"),
                video_track=FakeVideoTrack(),
            ),
            "b": FakeSession(
                session_id="b",
                pc=FakePC(connection_state="connected"),
                video_track=FakeVideoTrack(),
            ),
        }
    )
    with pytest.raises(ValueError, match="Multiple active sessions"):
        get_active_session(manager)  # type: ignore[arg-type]

    manager = FakeWebRTCManager(
        sessions={
            "a": FakeSession(
                session_id="a",
                pc=FakePC(connection_state="connected"),
                video_track=FakeVideoTrack(),
            )
        }
    )
    session = get_active_session(manager)  # type: ignore[arg-type]
    assert session.id == "a"


def test_apply_control_message_initializes_processor_pauses_and_forwards_params():
    fp = FakeFrameProcessor()
    vt = FakeVideoTrack(frame_processor=fp)
    session = FakeSession(session_id="s", pc=FakePC(connection_state="connected"), video_track=vt)

    ok = apply_control_message(
        session,  # type: ignore[arg-type]
        {"paused": True, "prompts": [{"text": "hello", "weight": 1.0}]},
    )
    assert ok is True
    assert vt.initialize_calls == 1
    assert vt.pause_calls == [True]
    assert fp.updates == [
        {"paused": True, "prompts": [{"text": "hello", "weight": 1.0}]}
    ]


def test_apply_control_message_translates_step_type_to_reserved_key():
    fp = FakeFrameProcessor()
    vt = FakeVideoTrack(frame_processor=fp)
    session = FakeSession(session_id="s", pc=FakePC(connection_state="connected"), video_track=vt)

    ok = apply_control_message(session, {"type": "step"})  # type: ignore[arg-type]
    assert ok is True
    assert fp.updates == [{"_rcp_step": True}]


def test_apply_control_message_restore_requires_snapshot_id():
    fp = FakeFrameProcessor()
    vt = FakeVideoTrack(frame_processor=fp)
    session = FakeSession(session_id="s", pc=FakePC(connection_state="connected"), video_track=vt)

    ok = apply_control_message(session, {"type": "restore_snapshot"})  # type: ignore[arg-type]
    assert ok is False
    assert fp.updates == []


def test_apply_control_message_propagates_queue_saturation_failure():
    fp = FakeFrameProcessor(accept_updates=False)
    vt = FakeVideoTrack(frame_processor=fp)
    session = FakeSession(session_id="s", pc=FakePC(connection_state="connected"), video_track=vt)

    ok = apply_control_message(session, {"paused": True})  # type: ignore[arg-type]
    assert ok is False


class _FakePipelineManager:
    def get_status_info(self):
        return {"pipeline_id": "test", "load_params": {"width": 512, "height": 512}}


def test_frame_processor_update_parameters_drops_oldest_on_full_queue():
    fp = FrameProcessor(pipeline_manager=_FakePipelineManager(), max_parameter_queue_size=1)

    assert fp.update_parameters({"a": 1}) is True
    assert fp.update_parameters({"b": 2}) is True

    latest = fp.parameters_queue.get_nowait()
    assert latest == {"b": 2}
    with pytest.raises(queue.Empty):
        fp.parameters_queue.get_nowait()


def test_frame_processor_flush_output_queue_and_latest_frame_clone():
    fp = FrameProcessor(pipeline_manager=_FakePipelineManager())

    fp.output_queue.put_nowait(torch.zeros((1, 1, 3), dtype=torch.uint8))
    fp.output_queue.put_nowait(torch.ones((1, 1, 3), dtype=torch.uint8))
    assert fp.flush_output_queue() == 2
    assert fp.output_queue.empty()

    with fp.latest_frame_lock:
        fp.latest_frame_cpu = torch.zeros((1, 1, 3), dtype=torch.uint8)

    frame1 = fp.get_latest_frame()
    assert frame1 is not None
    frame1[:] = 255  # mutate clone

    frame2 = fp.get_latest_frame()
    assert frame2 is not None
    assert int(frame2.max().item()) == 0
