"""Tests for ControlState and related dataclasses."""

import pytest

from scope.realtime.control_state import (
    CompiledPrompt,
    ControlState,
    GenerationMode,
)


class TestControlState:
    """Tests for ControlState."""

    def test_default_values(self):
        """ControlState has sensible defaults."""
        state = ControlState()

        assert state.prompts == []
        assert state.negative_prompt == ""
        assert state.lora_scales == []
        assert state.mode == GenerationMode.T2V
        assert state.num_frame_per_block == 3
        assert state.denoising_step_list == [1000, 750, 500, 250]
        assert state.base_seed == 42
        assert state.branch_seed_offset == 0
        assert state.kv_cache_attention_bias == 0.3  # KREA default
        assert state.transition is None
        assert state.current_start_frame == 0

    def test_effective_seed(self):
        """effective_seed combines base_seed and branch_seed_offset."""
        state = ControlState(base_seed=100, branch_seed_offset=5)
        assert state.effective_seed() == 105

    def test_effective_seed_negative_offset(self):
        """branch_seed_offset can be negative."""
        state = ControlState(base_seed=100, branch_seed_offset=-10)
        assert state.effective_seed() == 90

    def test_to_pipeline_kwargs_basic(self):
        """to_pipeline_kwargs produces correct base kwargs."""
        state = ControlState(
            prompts=[{"text": "test prompt", "weight": 1.0}],
            base_seed=123,
            kv_cache_attention_bias=0.5,
        )

        kwargs = state.to_pipeline_kwargs()

        assert kwargs["prompts"] == [{"text": "test prompt", "weight": 1.0}]
        assert kwargs["num_frame_per_block"] == 3
        assert kwargs["denoising_step_list"] == [1000, 750, 500, 250]
        assert kwargs["base_seed"] == 123  # Uses effective_seed()
        assert kwargs["kv_cache_attention_bias"] == 0.5

    def test_to_pipeline_kwargs_with_seed_offset(self):
        """to_pipeline_kwargs uses effective_seed with offset."""
        state = ControlState(base_seed=100, branch_seed_offset=50)

        kwargs = state.to_pipeline_kwargs()

        assert kwargs["base_seed"] == 150  # 100 + 50

    def test_to_pipeline_kwargs_includes_transition_when_set(self):
        """to_pipeline_kwargs includes transition if present."""
        state = ControlState(
            transition={
                "target_prompts": [{"text": "new prompt", "weight": 1.0}],
                "num_steps": 4,
                "temporal_interpolation_method": "linear",
            }
        )

        kwargs = state.to_pipeline_kwargs()

        assert "transition" in kwargs
        assert kwargs["transition"]["num_steps"] == 4

    def test_to_pipeline_kwargs_excludes_transition_when_none(self):
        """to_pipeline_kwargs does NOT include transition when None."""
        state = ControlState(transition=None)

        kwargs = state.to_pipeline_kwargs()

        assert "transition" not in kwargs

    def test_to_pipeline_kwargs_does_not_include_negative_prompt(self):
        """negative_prompt is NOT included in pipeline kwargs.

        The Scope/KREA pipeline does not consume negative prompts.
        """
        state = ControlState(
            negative_prompt="ugly, blurry",
        )

        kwargs = state.to_pipeline_kwargs()

        assert "negative_prompt" not in kwargs

    def test_to_pipeline_kwargs_does_not_include_lora_scales(self):
        """lora_scales is NOT included in base kwargs.

        PipelineAdapter is responsible for edge-triggering lora_scales.
        """
        state = ControlState(
            lora_scales=[{"path": "/loras/test.safetensors", "scale": 0.8}],
        )

        kwargs = state.to_pipeline_kwargs()

        assert "lora_scales" not in kwargs


class TestCompiledPrompt:
    """Tests for CompiledPrompt dataclass."""

    def test_default_values(self):
        """CompiledPrompt has sensible defaults."""
        prompt = CompiledPrompt()

        assert prompt.positive == []
        assert prompt.negative == ""
        assert prompt.lora_scales == []

    def test_with_values(self):
        """CompiledPrompt stores provided values."""
        prompt = CompiledPrompt(
            positive=[{"text": "test", "weight": 1.0}],
            negative="ugly",
            lora_scales=[{"path": "/test.safetensors", "scale": 0.5}],
        )

        assert prompt.positive == [{"text": "test", "weight": 1.0}]
        assert prompt.negative == "ugly"
        assert prompt.lora_scales == [{"path": "/test.safetensors", "scale": 0.5}]


class TestGenerationMode:
    """Tests for GenerationMode enum."""

    def test_mode_values(self):
        """GenerationMode has expected values."""
        assert GenerationMode.T2V.value == "text_to_video"
        assert GenerationMode.V2V.value == "video_to_video"
