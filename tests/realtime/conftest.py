"""Test fixtures for realtime control plane tests."""

import pytest
from typing import Any


class FakePipelineState:
    """Fake implementation of pipeline.state for testing.

    Mimics the Scope/KREA PipelineState key-value store interface.
    """

    def __init__(self):
        self._data: dict[str, Any] = {}

    def get(self, key: str) -> Any:
        """Get a value from state."""
        return self._data.get(key)

    def set(self, key: str, value: Any) -> None:
        """Set a value in state."""
        self._data[key] = value

    def clear(self):
        """Clear all state (for test reset)."""
        self._data.clear()


class FakePipeline:
    """Fake pipeline for testing the control plane without GPU.

    Mimics the KreaRealtimeVideoPipeline interface:
    - __call__(**kwargs) -> frames
    - state: PipelineState-like object
    """

    def __init__(self):
        self.state = FakePipelineState()
        self.call_count = 0
        self.last_kwargs: dict = {}
        self.call_history: list[dict] = []

        # Simulate continuity state keys
        self.state.set("current_start_frame", 0)
        self.state.set("context_frame_buffer_max_size", 4)
        self.state.set("decoded_frame_buffer_max_size", 4)

    def __call__(self, **kwargs) -> list:
        """Simulate pipeline call.

        Returns a list of 'frames' (just integers for testing).
        Records kwargs for test assertions.
        """
        self.call_count += 1
        self.last_kwargs = kwargs.copy()
        self.call_history.append(kwargs.copy())

        # Simulate frame generation
        num_frames = kwargs.get("num_frame_per_block", 3)
        start_frame = self.state.get("current_start_frame") or 0

        # Advance state like the real pipeline would
        self.state.set("current_start_frame", start_frame + num_frames)

        # Return fake frames
        return list(range(start_frame, start_frame + num_frames))

    def reset(self):
        """Reset for new test."""
        self.state.clear()
        self.state.set("current_start_frame", 0)
        self.state.set("context_frame_buffer_max_size", 4)
        self.state.set("decoded_frame_buffer_max_size", 4)
        self.call_count = 0
        self.last_kwargs = {}
        self.call_history = []


@pytest.fixture
def fake_pipeline():
    """Provide a fresh FakePipeline for each test."""
    return FakePipeline()


@pytest.fixture
def fake_pipeline_with_continuity(fake_pipeline):
    """Provide a FakePipeline with simulated continuity buffers."""
    # Simulate what the pipeline would have after generating some chunks
    fake_pipeline.state.set("current_start_frame", 9)  # 3 chunks of 3 frames
    fake_pipeline.state.set("first_context_frame", "tensor_placeholder")
    fake_pipeline.state.set("context_frame_buffer", ["frame1", "frame2", "frame3"])
    fake_pipeline.state.set("decoded_frame_buffer", ["decoded1", "decoded2"])
    return fake_pipeline
