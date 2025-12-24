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
    def __init__(self, pipeline: Any):
        self._pipeline = pipeline

    def get_pipeline(self):
        return self._pipeline


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


if __name__ == "__main__":
    unittest.main()
