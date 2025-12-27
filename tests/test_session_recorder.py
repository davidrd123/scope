"""Tests for server-side session recorder.

Based on Definition of Done from notes/proposals/server-side-session-recorder.md:
1. Hard cut replay fidelity
2. Soft cut replay fidelity
3. Transition replay fidelity
4. Stop is async, but path is observable
5. No prompt changes still yields a valid timeline
"""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# --- SessionRecorder unit tests ---


class TestSessionRecorderBasics:
    """Test basic SessionRecorder functionality."""

    def test_recorder_starts_inactive(self):
        """Recorder should not be recording initially."""
        from scope.server.session_recorder import SessionRecorder

        recorder = SessionRecorder()
        assert not recorder.is_recording

    def test_start_creates_active_recording(self):
        """Starting a recording should make is_recording True."""
        from scope.server.session_recorder import SessionRecorder

        recorder = SessionRecorder()
        recorder.start(
            chunk_index=0,
            pipeline_id="krea-realtime-video",
            load_params={"height": 480, "width": 832},
            baseline_prompt="A serene forest",
            baseline_weight=1.0,
        )
        assert recorder.is_recording

    def test_start_requires_pipeline_id(self):
        """Starting without pipeline_id should raise ValueError."""
        from scope.server.session_recorder import SessionRecorder

        recorder = SessionRecorder()
        with pytest.raises(ValueError, match="pipeline_id is required"):
            recorder.start(
                chunk_index=0,
                pipeline_id=None,  # type: ignore
                load_params={},
            )

    def test_stop_returns_recording_and_deactivates(self):
        """Stopping should return the recording and deactivate."""
        from scope.server.session_recorder import SessionRecorder

        recorder = SessionRecorder()
        recorder.start(
            chunk_index=0,
            pipeline_id="test-pipeline",
            load_params={},
            baseline_prompt="test",
        )
        recording = recorder.stop(chunk_index=10)

        assert recording is not None
        assert not recorder.is_recording
        assert recording.start_chunk == 0
        assert recording.end_chunk == 10

    def test_stop_when_not_recording_returns_none(self):
        """Stopping when not recording should return None."""
        from scope.server.session_recorder import SessionRecorder

        recorder = SessionRecorder()
        assert recorder.stop(chunk_index=0) is None


class TestSessionRecorderBaselinePrompt:
    """DoD item 5: No prompt changes still yields a valid timeline."""

    def test_baseline_prompt_recorded_at_start(self):
        """Baseline prompt should be captured at t=0."""
        from scope.server.session_recorder import SessionRecorder

        recorder = SessionRecorder()
        recorder.start(
            chunk_index=100,
            pipeline_id="test-pipeline",
            load_params={},
            baseline_prompt="Initial prompt",
            baseline_weight=1.0,
        )
        recording = recorder.stop(chunk_index=110)

        assert recording is not None
        assert len(recording.events) == 1
        assert recording.events[0].prompt == "Initial prompt"
        assert recording.events[0].wall_time == 0.0  # Relative to start

    def test_no_baseline_prompt_still_valid(self):
        """Recording without baseline prompt should still be valid."""
        from scope.server.session_recorder import SessionRecorder

        recorder = SessionRecorder()
        recorder.start(
            chunk_index=0,
            pipeline_id="test-pipeline",
            load_params={},
            baseline_prompt=None,
        )
        recording = recorder.stop(chunk_index=10)

        assert recording is not None
        # No events, but recording is valid
        timeline = recorder.export_timeline(recording)
        assert "prompts" in timeline
        assert timeline["version"] == "1.1"


class TestSessionRecorderHardCuts:
    """DoD item 1: Hard cut replay fidelity."""

    def test_hard_cut_recorded_as_init_cache(self):
        """Hard cuts should be exported as initCache: true."""
        from scope.server.session_recorder import SessionRecorder

        recorder = SessionRecorder()
        recorder.start(
            chunk_index=0,
            pipeline_id="test-pipeline",
            load_params={},
            baseline_prompt="start",
        )

        # Record a hard cut
        recorder.record_event(
            chunk_index=5,
            wall_time=time.monotonic(),
            prompt="After hard cut",
            hard_cut=True,
        )

        recording = recorder.stop(chunk_index=10)
        timeline = recorder.export_timeline(recording)

        # Find the segment with the hard cut
        hard_cut_segments = [s for s in timeline["prompts"] if s.get("initCache")]
        assert len(hard_cut_segments) == 1
        assert hard_cut_segments[0]["initCache"] is True

    def test_hard_cut_without_prompt_change(self):
        """Hard cut can occur without a prompt change."""
        from scope.server.session_recorder import SessionRecorder

        recorder = SessionRecorder()
        recorder.start(
            chunk_index=0,
            pipeline_id="test-pipeline",
            load_params={},
            baseline_prompt="constant prompt",
        )

        # Record a hard cut without new prompt (uses _last_prompt)
        recorder.record_event(
            chunk_index=5,
            wall_time=time.monotonic(),
            prompt=None,  # Will use last known prompt
            hard_cut=True,
        )

        recording = recorder.stop(chunk_index=10)
        timeline = recorder.export_timeline(recording)

        # Should have segment with initCache
        hard_cut_segments = [s for s in timeline["prompts"] if s.get("initCache")]
        assert len(hard_cut_segments) == 1
        # Prompt should be carried forward
        assert hard_cut_segments[0]["prompts"][0]["text"] == "constant prompt"


class TestSessionRecorderSoftCuts:
    """DoD item 2: Soft cut replay fidelity."""

    def test_soft_cut_recorded_with_restore_semantics(self):
        """Soft cuts should include restoreBias and restoreWasSet."""
        from scope.server.session_recorder import SessionRecorder

        recorder = SessionRecorder()
        recorder.start(
            chunk_index=0,
            pipeline_id="test-pipeline",
            load_params={"kv_cache_attention_bias": 0.3},
            baseline_prompt="start",
        )

        # Record a soft cut
        recorder.record_event(
            chunk_index=5,
            wall_time=time.monotonic(),
            prompt="During soft cut",
            soft_cut_bias=0.1,
            soft_cut_chunks=3,
            soft_cut_restore_bias=0.3,
            soft_cut_restore_was_set=True,
        )

        recording = recorder.stop(chunk_index=10)
        timeline = recorder.export_timeline(recording)

        # Find the segment with soft cut
        soft_cut_segments = [s for s in timeline["prompts"] if s.get("softCut")]
        assert len(soft_cut_segments) == 1

        soft_cut = soft_cut_segments[0]["softCut"]
        assert soft_cut["bias"] == 0.1
        assert soft_cut["chunks"] == 3
        assert soft_cut["restoreBias"] == 0.3
        assert soft_cut["restoreWasSet"] is True

    def test_soft_cut_restore_to_unset(self):
        """Soft cut with restoreWasSet=False means restore to unset."""
        from scope.server.session_recorder import SessionRecorder

        recorder = SessionRecorder()
        recorder.start(
            chunk_index=0,
            pipeline_id="test-pipeline",
            load_params={},  # No kv_cache_attention_bias set
            baseline_prompt="start",
        )

        recorder.record_event(
            chunk_index=5,
            wall_time=time.monotonic(),
            prompt="soft cut",
            soft_cut_bias=0.5,
            soft_cut_chunks=2,
            soft_cut_restore_bias=None,  # Was unset
            soft_cut_restore_was_set=False,
        )

        recording = recorder.stop(chunk_index=10)
        timeline = recorder.export_timeline(recording)

        soft_cut_segments = [s for s in timeline["prompts"] if s.get("softCut")]
        soft_cut = soft_cut_segments[0]["softCut"]
        assert soft_cut["restoreBias"] is None
        assert soft_cut["restoreWasSet"] is False


class TestSessionRecorderTransitions:
    """DoD item 3: Transition replay fidelity."""

    def test_transition_metadata_recorded(self):
        """Transitions should include steps and method."""
        from scope.server.session_recorder import SessionRecorder

        recorder = SessionRecorder()
        recorder.start(
            chunk_index=0,
            pipeline_id="test-pipeline",
            load_params={},
            baseline_prompt="start",
        )

        recorder.record_event(
            chunk_index=5,
            wall_time=time.monotonic(),
            prompt="New prompt",
            transition_steps=4,
            transition_method="slerp",
        )

        recording = recorder.stop(chunk_index=10)
        timeline = recorder.export_timeline(recording)

        # Find segment with transition
        transition_segments = [
            s for s in timeline["prompts"] if s.get("transitionSteps")
        ]
        assert len(transition_segments) == 1
        assert transition_segments[0]["transitionSteps"] == 4
        assert transition_segments[0]["temporalInterpolationMethod"] == "slerp"


class TestSessionRecorderExportFormat:
    """Test the exported timeline format matches spec."""

    def test_export_includes_version(self):
        """Exported timeline should have version 1.1."""
        from scope.server.session_recorder import SessionRecorder

        recorder = SessionRecorder()
        recorder.start(
            chunk_index=0,
            pipeline_id="test-pipeline",
            load_params={},
            baseline_prompt="test",
        )
        recording = recorder.stop(chunk_index=10)
        timeline = recorder.export_timeline(recording)

        assert timeline["version"] == "1.1"

    def test_export_includes_chunk_timebase(self):
        """Segments should include startChunk/endChunk."""
        from scope.server.session_recorder import SessionRecorder

        recorder = SessionRecorder()
        recorder.start(
            chunk_index=100,
            pipeline_id="test-pipeline",
            load_params={},
            baseline_prompt="test",
        )
        recording = recorder.stop(chunk_index=110)
        timeline = recorder.export_timeline(recording)

        assert len(timeline["prompts"]) > 0
        segment = timeline["prompts"][0]
        assert "startChunk" in segment
        assert "endChunk" in segment
        # Chunks should be relative to recording start
        assert segment["startChunk"] == 0

    def test_export_includes_settings(self):
        """Exported timeline should include pipeline settings."""
        from scope.server.session_recorder import SessionRecorder

        recorder = SessionRecorder()
        recorder.start(
            chunk_index=0,
            pipeline_id="krea-realtime-video",
            load_params={
                "height": 480,
                "width": 832,
                "seed": 42,
                "kv_cache_attention_bias": 0.3,
            },
            baseline_prompt="test",
        )
        recording = recorder.stop(chunk_index=10)
        timeline = recorder.export_timeline(recording)

        settings = timeline["settings"]
        assert settings["pipelineId"] == "krea-realtime-video"
        assert settings["resolution"]["height"] == 480
        assert settings["resolution"]["width"] == 832
        assert settings["seed"] == 42
        assert settings["kvCacheAttentionBias"] == 0.3

    def test_export_uses_weight_1_0(self):
        """Prompt weights should default to 1.0 (not 100.0)."""
        from scope.server.session_recorder import SessionRecorder

        recorder = SessionRecorder()
        recorder.start(
            chunk_index=0,
            pipeline_id="test-pipeline",
            load_params={},
            baseline_prompt="test",
            baseline_weight=1.0,
        )
        recording = recorder.stop(chunk_index=10)
        timeline = recorder.export_timeline(recording)

        assert timeline["prompts"][0]["prompts"][0]["weight"] == 1.0


class TestSessionRecorderStatusSnapshot:
    """DoD item 4: Stop is async, but path is observable."""

    def test_status_snapshot_thread_safe(self):
        """get_status_snapshot should return atomic dict."""
        from scope.server.session_recorder import SessionRecorder

        recorder = SessionRecorder()

        # Before recording
        status = recorder.get_status_snapshot()
        assert status["is_recording"] is False

        # During recording
        recorder.start(
            chunk_index=0,
            pipeline_id="test",
            load_params={},
            baseline_prompt="test",
        )
        status = recorder.get_status_snapshot()
        assert status["is_recording"] is True
        assert "start_chunk" in status
        assert "events_count" in status

        # After recording
        recorder.stop(chunk_index=10)
        status = recorder.get_status_snapshot()
        assert status["is_recording"] is False


# --- Placeholder for integration tests ---
# These would require actual FrameProcessor/PipelineManager integration


class TestSessionRecorderIntegration:
    """Integration tests with FrameProcessor (placeholder)."""

    @pytest.mark.skip(reason="Requires SessionRecorder implementation in FrameProcessor")
    def test_reserved_key_starts_recording(self):
        """_rcp_session_recording_start should start recording."""
        pass

    @pytest.mark.skip(reason="Requires SessionRecorder implementation in FrameProcessor")
    def test_reserved_key_stops_recording(self):
        """_rcp_session_recording_stop should stop and save."""
        pass

    @pytest.mark.skip(reason="Requires SessionRecorder implementation in FrameProcessor")
    def test_hard_cut_recorded_on_pipeline_call(self):
        """Hard cut should be recorded when init_cache=True passed to pipeline."""
        pass
