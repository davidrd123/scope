import unittest
from dataclasses import dataclass
from typing import Any

try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:
    from scope.server.frame_processor import FrameProcessor  # type: ignore
except ImportError:  # pragma: no cover
    FrameProcessor = None  # type: ignore[assignment]


class FakePipelineState:
    def __init__(self, initial: dict[str, Any] | None = None):
        self._data = dict(initial or {})

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value


@dataclass(frozen=True)
class FakeRequirements:
    input_size: int


class FakePipeline:
    """CPU-only fake pipeline that mimics the FrameProcessor call contract."""

    def __init__(self):
        self.state = FakePipelineState()
        self.call_history: list[dict[str, Any]] = []
        self.prepare_history: list[dict[str, Any]] = []
        self.vace_enabled = False

    def prepare(self, **kwargs: Any):
        self.prepare_history.append(dict(kwargs))
        return None

    def __call__(self, **kwargs: Any):
        if torch is None:  # pragma: no cover
            raise RuntimeError("torch is required for FakePipeline")
        self.call_history.append(dict(kwargs))
        num_frames = int(kwargs.get("num_frame_per_block", 3))
        return torch.zeros((num_frames, 1, 1, 3), dtype=torch.float32)


class FakeVideoPipeline(FakePipeline):
    def __init__(self, *, input_size: int):
        super().__init__()
        self._input_size = input_size

    def prepare(self, **kwargs: Any):
        self.prepare_history.append(dict(kwargs))
        return FakeRequirements(input_size=self._input_size)


class FakePipelineManager:
    def __init__(self, pipeline: Any, pipeline_id: str = "test-pipeline"):
        self._pipeline = pipeline
        self._pipeline_id = pipeline_id

    def get_pipeline(self):
        return self._pipeline

    def get_status_info(self) -> dict[str, Any]:
        return {
            "pipeline_id": self._pipeline_id,
            "load_params": {"width": 512, "height": 512},
        }


@unittest.skipIf(
    torch is None or FrameProcessor is None,
    "torch and scope.server.frame_processor are required for these tests",
)
class FrameProcessorCharacterizationTests(unittest.TestCase):
    def test_drains_all_parameter_updates_at_chunk_boundary(self):
        pipeline = FakePipeline()
        pm = FakePipelineManager(pipeline)
        fp = FrameProcessor(
            pipeline_manager=pm,
            initial_parameters={"prompts": [{"text": "init", "weight": 1.0}]},
        )

        fp.update_parameters({"prompts": [{"text": "b", "weight": 1.0}]})
        fp.update_parameters({"prompts": [{"text": "c", "weight": 1.0}]})
        fp.update_parameters({"kv_cache_attention_bias": 0.1})
        fp.process_chunk()

        self.assertEqual(
            pipeline.call_history[0]["prompts"],
            [{"text": "c", "weight": 1.0}],
        )
        self.assertEqual(pipeline.call_history[0]["kv_cache_attention_bias"], 0.1)

    def test_pause_update_can_apply_other_state_without_generating(self):
        pipeline = FakePipeline()
        pm = FakePipelineManager(pipeline)
        fp = FrameProcessor(
            pipeline_manager=pm,
            initial_parameters={"prompts": [{"text": "init", "weight": 1.0}]},
        )

        fp.update_parameters(
            {"paused": True, "prompts": [{"text": "paused_prompt", "weight": 1.0}]}
        )
        fp.process_chunk()

        self.assertTrue(fp.paused)
        self.assertEqual(pipeline.call_history, [])
        self.assertEqual(fp.parameters["prompts"], [{"text": "paused_prompt", "weight": 1.0}])

    def test_init_cache_true_first_call_then_false(self):
        pipeline = FakePipeline()
        pm = FakePipelineManager(pipeline)
        fp = FrameProcessor(
            pipeline_manager=pm,
            initial_parameters={"prompts": [{"text": "a", "weight": 1.0}]},
        )

        fp.process_chunk()
        self.assertTrue(pipeline.call_history[0]["init_cache"])
        self.assertTrue(fp.is_prepared)

        fp.process_chunk()
        self.assertFalse(pipeline.call_history[1]["init_cache"])

    def test_reset_cache_clears_output_queue_and_forces_init_cache(self):
        pipeline = FakePipeline()
        pm = FakePipelineManager(pipeline)
        fp = FrameProcessor(
            pipeline_manager=pm,
            initial_parameters={"prompts": [{"text": "a", "weight": 1.0}]},
        )
        fp.is_prepared = True

        fp.output_queue.put_nowait(torch.tensor(123, dtype=torch.uint8))
        self.assertEqual(fp.output_queue.qsize(), 1)

        fp.update_parameters({"reset_cache": True})
        fp.process_chunk()

        self.assertTrue(pipeline.call_history[0]["init_cache"])
        self.assertEqual(fp.output_queue.qsize(), 3)

    def test_lora_scales_is_one_shot(self):
        pipeline = FakePipeline()
        pm = FakePipelineManager(pipeline)
        fp = FrameProcessor(
            pipeline_manager=pm,
            initial_parameters={"prompts": [{"text": "a", "weight": 1.0}]},
        )

        fp.update_parameters({"lora_scales": [{"path": "foo.safetensors", "scale": 1.0}]})
        fp.process_chunk()
        self.assertIn("lora_scales", pipeline.call_history[0])

        fp.process_chunk()
        self.assertNotIn("lora_scales", pipeline.call_history[1])

    def test_transition_completion_clears_transition_and_updates_prompts(self):
        target_prompts = [{"text": "b", "weight": 1.0}]
        pipeline = FakePipeline()
        pipeline.state.set("_transition_active", False)
        pm = FakePipelineManager(pipeline)
        fp = FrameProcessor(
            pipeline_manager=pm,
            initial_parameters={
                "prompts": [{"text": "a", "weight": 1.0}],
                "transition": {
                    "target_prompts": target_prompts,
                    "num_steps": 4,
                    "temporal_interpolation_method": "linear",
                },
            },
        )

        fp.process_chunk()
        self.assertNotIn("transition", fp.parameters)
        self.assertEqual(fp.parameters.get("prompts"), target_prompts)

    def test_new_prompts_without_transition_clears_stale_transition(self):
        pipeline = FakePipeline()
        pipeline.state.set("_transition_active", True)
        pm = FakePipelineManager(pipeline)
        fp = FrameProcessor(
            pipeline_manager=pm,
            initial_parameters={
                "prompts": [{"text": "a", "weight": 1.0}],
                "transition": {
                    "target_prompts": [{"text": "b", "weight": 1.0}],
                    "num_steps": 4,
                    "temporal_interpolation_method": "linear",
                },
            },
        )

        fp.update_parameters({"prompts": [{"text": "new", "weight": 1.0}]})
        fp.process_chunk()
        self.assertNotIn("transition", pipeline.call_history[0])
        self.assertNotIn("transition", fp.parameters)

    def test_video_mode_requires_input_frames_before_call(self):
        pipeline = FakeVideoPipeline(input_size=2)
        pm = FakePipelineManager(pipeline)
        fp = FrameProcessor(
            pipeline_manager=pm,
            initial_parameters={"input_mode": "video"},
        )

        fp.process_chunk()

        self.assertEqual(pipeline.call_history, [])
        self.assertFalse(fp.is_prepared)

    # =========================================================================
    # Snapshot/Restore Tests (Phase 2)
    # =========================================================================

    def test_snapshot_request_creates_snapshot(self):
        """Snapshot request via reserved key creates a snapshot with metadata."""
        pipeline = FakePipeline()
        pipeline.state.set("current_start_frame", 42)
        pipeline.state.set("first_context_frame", torch.zeros(1, 1, 16, 64, 64))
        pm = FakePipelineManager(pipeline)
        fp = FrameProcessor(
            pipeline_manager=pm,
            initial_parameters={"prompts": [{"text": "a", "weight": 1.0}]},
        )
        fp.chunk_index = 5

        # Track response via callback
        responses = []
        fp.snapshot_response_callback = responses.append

        fp.update_parameters({"_rcp_snapshot_request": True})
        fp.process_chunk()

        # Verify snapshot was created
        self.assertEqual(len(fp.snapshots), 1)
        snapshot_id = list(fp.snapshots.keys())[0]
        snapshot = fp.snapshots[snapshot_id]

        self.assertEqual(snapshot.chunk_index, 5)
        self.assertEqual(snapshot.current_start_frame, 42)
        self.assertIsNotNone(snapshot.first_context_frame)
        self.assertEqual(snapshot.parameters["prompts"], [{"text": "a", "weight": 1.0}])

        # Verify response callback was called
        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0]["type"], "snapshot_response")
        self.assertEqual(responses[0]["snapshot_id"], snapshot_id)
        self.assertEqual(responses[0]["chunk_index"], 5)

    def test_restore_clears_output_queue(self):
        """Restore clears output queue to prevent stale pre-restore frames."""
        pipeline = FakePipeline()
        pm = FakePipelineManager(pipeline)
        fp = FrameProcessor(
            pipeline_manager=pm,
            initial_parameters={"prompts": [{"text": "a", "weight": 1.0}]},
        )

        # Create a snapshot
        fp.update_parameters({"_rcp_snapshot_request": True})
        fp.process_chunk()
        snapshot_id = list(fp.snapshots.keys())[0]

        # Add frames to output queue
        fp.output_queue.put_nowait(torch.tensor(1, dtype=torch.uint8))
        fp.output_queue.put_nowait(torch.tensor(2, dtype=torch.uint8))
        self.assertEqual(fp.output_queue.qsize(), 5)  # 3 from chunk + 2 added

        # Restore should clear the queue
        fp.update_parameters({"_rcp_restore_snapshot": {"snapshot_id": snapshot_id}})
        fp.process_chunk()

        # Queue should be empty after restore, then filled with new chunk
        # (restore clears, then process_chunk adds 3 frames)
        self.assertEqual(fp.output_queue.qsize(), 3)

    def test_restore_sets_is_prepared_true(self):
        """Restore sets is_prepared=True to avoid accidental cache reset."""
        pipeline = FakePipeline()
        pm = FakePipelineManager(pipeline)
        fp = FrameProcessor(
            pipeline_manager=pm,
            initial_parameters={"prompts": [{"text": "a", "weight": 1.0}]},
        )

        # Create a snapshot (is_prepared becomes True after process_chunk)
        fp.update_parameters({"_rcp_snapshot_request": True})
        fp.process_chunk()
        snapshot_id = list(fp.snapshots.keys())[0]

        # Reset is_prepared to False
        fp.is_prepared = False

        # Restore should set is_prepared=True
        fp.update_parameters({"_rcp_restore_snapshot": {"snapshot_id": snapshot_id}})
        fp.process_chunk()

        self.assertTrue(fp.is_prepared)

    def test_restore_invalid_snapshot_id_fails_gracefully(self):
        """Restore with invalid snapshot_id returns error response, doesn't crash."""
        pipeline = FakePipeline()
        pm = FakePipelineManager(pipeline)
        fp = FrameProcessor(
            pipeline_manager=pm,
            initial_parameters={"prompts": [{"text": "a", "weight": 1.0}]},
        )

        # Track response via callback
        responses = []
        fp.snapshot_response_callback = responses.append

        fp.update_parameters({"_rcp_restore_snapshot": {"snapshot_id": "nonexistent-id"}})
        fp.process_chunk()

        # Verify error response
        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0]["type"], "restore_response")
        self.assertFalse(responses[0]["success"])

    def test_snapshot_lru_eviction(self):
        """Snapshots beyond MAX_SNAPSHOTS are evicted (oldest first)."""
        from scope.server.frame_processor import MAX_SNAPSHOTS

        pipeline = FakePipeline()
        pm = FakePipelineManager(pipeline)
        fp = FrameProcessor(
            pipeline_manager=pm,
            initial_parameters={"prompts": [{"text": "a", "weight": 1.0}]},
        )

        # Create MAX_SNAPSHOTS + 2 snapshots
        snapshot_ids = []
        for i in range(MAX_SNAPSHOTS + 2):
            fp.chunk_index = i
            fp.update_parameters({"_rcp_snapshot_request": True})
            fp.process_chunk()
            snapshot_ids.append(list(fp.snapshots.keys())[-1])

        # Should have exactly MAX_SNAPSHOTS snapshots
        self.assertEqual(len(fp.snapshots), MAX_SNAPSHOTS)

        # First two snapshots should be evicted
        self.assertNotIn(snapshot_ids[0], fp.snapshots)
        self.assertNotIn(snapshot_ids[1], fp.snapshots)

        # Last MAX_SNAPSHOTS should still exist
        for sid in snapshot_ids[2:]:
            self.assertIn(sid, fp.snapshots)

    def test_restore_restores_parameters_and_paused_state(self):
        """Restore restores both parameters and paused state."""
        pipeline = FakePipeline()
        pm = FakePipelineManager(pipeline)
        fp = FrameProcessor(
            pipeline_manager=pm,
            initial_parameters={"prompts": [{"text": "original", "weight": 1.0}]},
        )

        # Create a snapshot with specific state
        fp.paused = True
        fp.update_parameters({"_rcp_snapshot_request": True})
        fp.process_chunk()
        snapshot_id = list(fp.snapshots.keys())[0]

        # Change state
        fp.paused = False
        fp.parameters["prompts"] = [{"text": "changed", "weight": 1.0}]

        # Restore should bring back original state
        fp.update_parameters({"_rcp_restore_snapshot": {"snapshot_id": snapshot_id}})
        fp.process_chunk()

        self.assertTrue(fp.paused)
        self.assertEqual(fp.parameters["prompts"], [{"text": "original", "weight": 1.0}])

    def test_restore_clears_missing_continuity_keys(self):
        """Restore clears continuity keys that were absent in the snapshot."""
        pipeline = FakePipeline()
        pm = FakePipelineManager(pipeline)
        fp = FrameProcessor(
            pipeline_manager=pm,
            initial_parameters={"prompts": [{"text": "a", "weight": 1.0}]},
        )

        # Snapshot at a point where first_context_frame is unset (None)
        fp.update_parameters({"_rcp_snapshot_request": True})
        fp.process_chunk()
        snapshot_id = list(fp.snapshots.keys())[0]

        # Mutate pipeline state to simulate later continuity being present
        pipeline.state.set("first_context_frame", torch.ones(1, 1, 16, 64, 64))

        # Restore should clear it back to None
        fp.update_parameters({"_rcp_restore_snapshot": {"snapshot_id": snapshot_id}})
        fp.process_chunk()
        self.assertIsNone(pipeline.state.get("first_context_frame"))

    def test_restore_restores_video_mode_and_clears_frame_buffer(self):
        """Restore restores video mode and clears frame buffer when in V2V."""
        pipeline = FakePipeline()
        pm = FakePipelineManager(pipeline)
        fp = FrameProcessor(
            pipeline_manager=pm,
            initial_parameters={"input_mode": "video"},
        )
        fp.paused = True

        fp.update_parameters({"_rcp_snapshot_request": True})
        fp.process_chunk()
        snapshot_id = list(fp.snapshots.keys())[0]

        # Simulate state drift
        fp._video_mode = False
        with fp.frame_buffer_lock:
            fp.frame_buffer.append(object())

        fp.update_parameters({"_rcp_restore_snapshot": {"snapshot_id": snapshot_id}})
        fp.process_chunk()

        self.assertTrue(fp._video_mode)
        with fp.frame_buffer_lock:
            self.assertEqual(len(fp.frame_buffer), 0)

    # =========================================================================
    # Step Tests (Phase 3)
    # =========================================================================

    def test_step_generates_chunk_while_paused(self):
        """Step generates exactly one chunk even when paused."""
        pipeline = FakePipeline()
        pm = FakePipelineManager(pipeline)
        fp = FrameProcessor(
            pipeline_manager=pm,
            initial_parameters={"prompts": [{"text": "a", "weight": 1.0}]},
        )
        fp.paused = True

        # Without step, paused means no generation
        fp.process_chunk()
        self.assertEqual(len(pipeline.call_history), 0)

        # With step, generate one chunk
        fp.update_parameters({"_rcp_step": True})
        fp.process_chunk()

        self.assertEqual(len(pipeline.call_history), 1)
        self.assertTrue(fp.paused)  # Still paused after step

    def test_step_sends_response_with_chunk_index(self):
        """Step sends response with chunk_index after successful generation."""
        pipeline = FakePipeline()
        pm = FakePipelineManager(pipeline)
        fp = FrameProcessor(
            pipeline_manager=pm,
            initial_parameters={"prompts": [{"text": "a", "weight": 1.0}]},
        )
        fp.paused = True
        fp.chunk_index = 10

        responses = []
        fp.snapshot_response_callback = responses.append

        fp.update_parameters({"_rcp_step": True})
        fp.process_chunk()

        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0]["type"], "step_response")
        self.assertEqual(responses[0]["chunk_index"], 11)  # Incremented
        self.assertTrue(responses[0]["success"])

    def test_step_works_when_not_paused(self):
        """Step also works when not paused (just generates normally)."""
        pipeline = FakePipeline()
        pm = FakePipelineManager(pipeline)
        fp = FrameProcessor(
            pipeline_manager=pm,
            initial_parameters={"prompts": [{"text": "a", "weight": 1.0}]},
        )
        fp.paused = False

        responses = []
        fp.snapshot_response_callback = responses.append

        fp.update_parameters({"_rcp_step": True})
        fp.process_chunk()

        self.assertEqual(len(pipeline.call_history), 1)
        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0]["type"], "step_response")

    def test_snapshot_lru_updates_on_restore(self):
        """Restored snapshot moves to end of LRU order (most recently used)."""
        pipeline = FakePipeline()
        pm = FakePipelineManager(pipeline)
        fp = FrameProcessor(
            pipeline_manager=pm,
            initial_parameters={"prompts": [{"text": "a", "weight": 1.0}]},
        )

        # Create 3 snapshots
        snapshot_ids = []
        for i in range(3):
            fp.chunk_index = i
            fp.update_parameters({"_rcp_snapshot_request": True})
            fp.process_chunk()
            snapshot_ids.append(list(fp.snapshots.keys())[-1])

        # Order should be [0, 1, 2]
        self.assertEqual(fp.snapshot_order, snapshot_ids)

        # Restore snapshot 0 - should move to end
        fp.update_parameters({"_rcp_restore_snapshot": {"snapshot_id": snapshot_ids[0]}})
        fp.process_chunk()

        # Order should now be [1, 2, 0]
        self.assertEqual(fp.snapshot_order, [snapshot_ids[1], snapshot_ids[2], snapshot_ids[0]])

    def test_step_not_dropped_when_video_input_not_ready(self):
        """Step persists until V2V input requirements are met (no lost step)."""
        pipeline = FakeVideoPipeline(input_size=2)
        pm = FakePipelineManager(pipeline)
        fp = FrameProcessor(
            pipeline_manager=pm,
            initial_parameters={"input_mode": "video"},
        )
        fp.paused = True

        # Avoid numpy dependency in prepare_chunk by stubbing it.
        def fake_prepare_chunk(chunk_size: int):
            for _ in range(chunk_size):
                fp.frame_buffer.popleft()
            return [torch.zeros((1, 1, 1, 3), dtype=torch.float32) for _ in range(chunk_size)]

        fp.prepare_chunk = fake_prepare_chunk  # type: ignore[method-assign]

        responses = []
        fp.snapshot_response_callback = responses.append

        fp.update_parameters({"_rcp_step": True})
        fp.process_chunk()

        self.assertEqual(len(pipeline.call_history), 0)
        self.assertEqual(responses, [])

        with fp.frame_buffer_lock:
            fp.frame_buffer.append(object())
            fp.frame_buffer.append(object())

        fp.process_chunk()

        self.assertEqual(len(pipeline.call_history), 1)
        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0]["type"], "step_response")
        self.assertTrue(responses[0]["success"])


if __name__ == "__main__":
    unittest.main()
