"""Tests for PipelineAdapter - kwargs mapping, edge-triggering, and continuity."""

import pytest

from scope.realtime.control_state import ControlState
from scope.realtime.pipeline_adapter import PipelineAdapter


class TestKwargsForCall:
    """Tests for kwargs_for_call() method."""

    def test_includes_init_cache(self, fake_pipeline):
        """kwargs_for_call always includes init_cache."""
        adapter = PipelineAdapter(fake_pipeline)
        state = ControlState()

        kwargs = adapter.kwargs_for_call(state, init_cache=True)
        assert kwargs["init_cache"] is True

        kwargs = adapter.kwargs_for_call(state, init_cache=False)
        assert kwargs["init_cache"] is False

    def test_includes_base_pipeline_kwargs(self, fake_pipeline):
        """kwargs_for_call includes standard pipeline kwargs."""
        adapter = PipelineAdapter(fake_pipeline)
        state = ControlState(
            prompts=[{"text": "test", "weight": 1.0}],
            base_seed=123,
            kv_cache_attention_bias=0.5,
        )

        kwargs = adapter.kwargs_for_call(state, init_cache=False)

        assert kwargs["prompts"] == [{"text": "test", "weight": 1.0}]
        assert kwargs["base_seed"] == 123
        assert kwargs["kv_cache_attention_bias"] == 0.5

    def test_does_not_include_negative_prompt(self, fake_pipeline):
        """negative_prompt is NOT forwarded to pipeline kwargs.

        The Scope/KREA pipeline does not consume negative prompts.
        """
        adapter = PipelineAdapter(fake_pipeline)
        state = ControlState(negative_prompt="ugly, blurry")

        kwargs = adapter.kwargs_for_call(state, init_cache=False)

        assert "negative_prompt" not in kwargs

    def test_includes_transition_when_set(self, fake_pipeline):
        """kwargs_for_call includes transition when present in state."""
        adapter = PipelineAdapter(fake_pipeline)
        state = ControlState(
            transition={
                "target_prompts": [{"text": "new", "weight": 1.0}],
                "num_steps": 4,
                "temporal_interpolation_method": "linear",
            }
        )

        kwargs = adapter.kwargs_for_call(state, init_cache=False)

        assert "transition" in kwargs
        assert kwargs["transition"]["num_steps"] == 4


class TestLoraScalesEdgeTrigger:
    """Tests for lora_scales edge-triggering behavior."""

    def test_lora_scales_included_on_first_call(self, fake_pipeline):
        """lora_scales is included on first call when set."""
        adapter = PipelineAdapter(fake_pipeline)
        state = ControlState(
            lora_scales=[{"path": "/loras/test.safetensors", "scale": 0.8}]
        )

        kwargs = adapter.kwargs_for_call(state, init_cache=False)

        assert "lora_scales" in kwargs
        assert kwargs["lora_scales"][0]["path"] == "/loras/test.safetensors"

    def test_lora_scales_not_included_when_unchanged(self, fake_pipeline):
        """lora_scales is NOT included when unchanged from previous call."""
        adapter = PipelineAdapter(fake_pipeline)
        state = ControlState(
            lora_scales=[{"path": "/loras/test.safetensors", "scale": 0.8}]
        )

        # First call - included
        kwargs1 = adapter.kwargs_for_call(state, init_cache=False)
        assert "lora_scales" in kwargs1

        # Second call with same scales - NOT included
        kwargs2 = adapter.kwargs_for_call(state, init_cache=False)
        assert "lora_scales" not in kwargs2

    def test_lora_scales_included_when_changed(self, fake_pipeline):
        """lora_scales IS included when changed from previous call."""
        adapter = PipelineAdapter(fake_pipeline)

        state1 = ControlState(
            lora_scales=[{"path": "/loras/test.safetensors", "scale": 0.8}]
        )
        state2 = ControlState(
            lora_scales=[{"path": "/loras/test.safetensors", "scale": 0.5}]  # Changed!
        )

        # First call
        adapter.kwargs_for_call(state1, init_cache=False)

        # Second call with changed scale - included
        kwargs = adapter.kwargs_for_call(state2, init_cache=False)
        assert "lora_scales" in kwargs
        assert kwargs["lora_scales"][0]["scale"] == 0.5

    def test_lora_scales_included_when_path_added(self, fake_pipeline):
        """lora_scales IS included when a new LoRA is added."""
        adapter = PipelineAdapter(fake_pipeline)

        state1 = ControlState(
            lora_scales=[{"path": "/loras/a.safetensors", "scale": 0.8}]
        )
        state2 = ControlState(
            lora_scales=[
                {"path": "/loras/a.safetensors", "scale": 0.8},
                {"path": "/loras/b.safetensors", "scale": 0.5},  # Added!
            ]
        )

        adapter.kwargs_for_call(state1, init_cache=False)
        kwargs = adapter.kwargs_for_call(state2, init_cache=False)

        assert "lora_scales" in kwargs
        assert len(kwargs["lora_scales"]) == 2

    def test_lora_scales_empty_to_empty_not_included(self, fake_pipeline):
        """lora_scales NOT included when going from empty to empty."""
        adapter = PipelineAdapter(fake_pipeline)
        state = ControlState(lora_scales=[])

        # First call with empty
        kwargs1 = adapter.kwargs_for_call(state, init_cache=False)
        # Empty is different from None (initial state), so might be included
        # But second call should definitely not include
        kwargs2 = adapter.kwargs_for_call(state, init_cache=False)
        assert "lora_scales" not in kwargs2

    def test_reset_lora_tracking(self, fake_pipeline):
        """reset_lora_tracking allows lora_scales to be re-sent."""
        adapter = PipelineAdapter(fake_pipeline)
        state = ControlState(
            lora_scales=[{"path": "/loras/test.safetensors", "scale": 0.8}]
        )

        # First call - included
        adapter.kwargs_for_call(state, init_cache=False)

        # Second call - not included (edge-triggered)
        kwargs = adapter.kwargs_for_call(state, init_cache=False)
        assert "lora_scales" not in kwargs

        # Reset tracking
        adapter.reset_lora_tracking()

        # Third call - included again
        kwargs = adapter.kwargs_for_call(state, init_cache=False)
        assert "lora_scales" in kwargs


class TestContinuityCapture:
    """Tests for capture_continuity() method."""

    def test_capture_returns_continuity_keys(self, fake_pipeline_with_continuity):
        """capture_continuity returns known continuity keys from pipeline.state."""
        adapter = PipelineAdapter(fake_pipeline_with_continuity)

        continuity = adapter.capture_continuity()

        assert "current_start_frame" in continuity
        assert "first_context_frame" in continuity
        assert "context_frame_buffer" in continuity
        assert "decoded_frame_buffer" in continuity

    def test_capture_excludes_none_values(self, fake_pipeline):
        """capture_continuity excludes keys with None values."""
        adapter = PipelineAdapter(fake_pipeline)

        # fake_pipeline doesn't set first_context_frame
        continuity = adapter.capture_continuity()

        assert "first_context_frame" not in continuity

    def test_capture_with_no_pipeline(self):
        """capture_continuity returns empty dict when pipeline is None."""
        adapter = PipelineAdapter(None)

        continuity = adapter.capture_continuity()

        assert continuity == {}

    def test_capture_includes_buffer_sizes(self, fake_pipeline):
        """capture_continuity includes buffer size config."""
        adapter = PipelineAdapter(fake_pipeline)

        continuity = adapter.capture_continuity()

        assert "context_frame_buffer_max_size" in continuity
        assert "decoded_frame_buffer_max_size" in continuity


class TestContinuityRestore:
    """Tests for restore_continuity() method."""

    def test_restore_sets_state_keys(self, fake_pipeline):
        """restore_continuity sets keys in pipeline.state."""
        adapter = PipelineAdapter(fake_pipeline)

        continuity = {
            "current_start_frame": 99,
            "first_context_frame": "tensor_data",
            "context_frame_buffer": ["a", "b", "c"],
        }

        adapter.restore_continuity(continuity)

        assert fake_pipeline.state.get("current_start_frame") == 99
        assert fake_pipeline.state.get("first_context_frame") == "tensor_data"
        assert fake_pipeline.state.get("context_frame_buffer") == ["a", "b", "c"]

    def test_restore_with_no_pipeline(self):
        """restore_continuity is a no-op when pipeline is None."""
        adapter = PipelineAdapter(None)

        # Should not raise
        adapter.restore_continuity({"current_start_frame": 99})

    def test_capture_restore_roundtrip(self, fake_pipeline_with_continuity):
        """capture then restore produces equivalent state."""
        adapter = PipelineAdapter(fake_pipeline_with_continuity)

        # Capture current state
        original = adapter.capture_continuity()

        # Clear and restore
        fake_pipeline_with_continuity.state.clear()
        adapter.restore_continuity(original)

        # Capture again
        restored = adapter.capture_continuity()

        assert restored == original
