"""Tests for GeneratorDriver - chunk boundaries, events, and snapshots."""

import asyncio

import pytest

from scope.realtime.control_bus import ApplyMode, EventType
from scope.realtime.control_state import ControlState
from scope.realtime.generator_driver import (
    DriverState,
    GenerationResult,
    GeneratorDriver,
)


class TestDriverState:
    """Tests for driver state management."""

    def test_initial_state_is_stopped(self, fake_pipeline):
        """Driver starts in STOPPED state."""
        driver = GeneratorDriver(fake_pipeline)
        assert driver.state == DriverState.STOPPED

    def test_pause_sets_paused_state(self, fake_pipeline):
        """pause() sets driver to PAUSED."""
        driver = GeneratorDriver(fake_pipeline)
        driver.pause()
        assert driver.state == DriverState.PAUSED

    def test_stop_sets_stopped_state(self, fake_pipeline):
        """stop() sets driver to STOPPED."""
        driver = GeneratorDriver(fake_pipeline)
        driver.state = DriverState.RUNNING
        driver.stop()
        assert driver.state == DriverState.STOPPED

    def test_on_state_change_callback(self, fake_pipeline):
        """State changes trigger on_state_change callback."""
        states = []
        driver = GeneratorDriver(
            fake_pipeline,
            on_state_change=lambda s: states.append(s),
        )

        driver.pause()
        driver.stop()

        assert states == [DriverState.PAUSED, DriverState.STOPPED]


class TestChunkGeneration:
    """Tests for chunk generation via step()."""

    @pytest.mark.asyncio
    async def test_step_generates_one_chunk(self, fake_pipeline):
        """step() generates exactly one chunk."""
        driver = GeneratorDriver(fake_pipeline)

        result = await driver.step()

        assert result is not None
        assert fake_pipeline.call_count == 1

    @pytest.mark.asyncio
    async def test_step_returns_generation_result(self, fake_pipeline):
        """step() returns GenerationResult with expected fields."""
        driver = GeneratorDriver(fake_pipeline)

        result = await driver.step()

        assert isinstance(result, GenerationResult)
        assert result.frames is not None
        assert result.chunk_index == 1
        assert result.timing_ms >= 0
        assert "prompts" in result.control_state_snapshot

    @pytest.mark.asyncio
    async def test_step_increments_chunk_index(self, fake_pipeline):
        """step() increments chunk_index on each call."""
        driver = GeneratorDriver(fake_pipeline)

        r1 = await driver.step()
        r2 = await driver.step()

        assert r1.chunk_index == 1
        assert r2.chunk_index == 2

    @pytest.mark.asyncio
    async def test_step_sets_state_to_paused_after(self, fake_pipeline):
        """step() leaves driver in PAUSED state."""
        driver = GeneratorDriver(fake_pipeline)

        await driver.step()

        assert driver.state == DriverState.PAUSED

    @pytest.mark.asyncio
    async def test_step_passes_init_cache_false_after_first(self, fake_pipeline):
        """After first step, init_cache should be False."""
        driver = GeneratorDriver(fake_pipeline)

        await driver.step()
        await driver.step()

        # First call: init_cache=True (not prepared yet)
        # Second call: init_cache=False (prepared)
        assert fake_pipeline.call_history[0]["init_cache"] is True
        assert fake_pipeline.call_history[1]["init_cache"] is False


class TestEventApplication:
    """Tests for applying control events at chunk boundaries."""

    @pytest.mark.asyncio
    async def test_events_applied_at_chunk_boundary(self, fake_pipeline):
        """Events are applied at chunk boundary (before generation)."""
        driver = GeneratorDriver(fake_pipeline)

        # Enqueue prompt update
        driver.control_bus.enqueue(
            EventType.SET_PROMPT,
            payload={"prompts": [{"text": "new prompt", "weight": 1.0}]},
        )

        await driver.step()

        # Prompt should be applied before the pipeline call
        assert driver.control_state.prompts == [{"text": "new prompt", "weight": 1.0}]
        assert fake_pipeline.last_kwargs["prompts"] == [
            {"text": "new prompt", "weight": 1.0}
        ]

    @pytest.mark.asyncio
    async def test_pause_event_stops_generation(self, fake_pipeline):
        """PAUSE event stops the generation loop."""
        driver = GeneratorDriver(fake_pipeline)
        driver.state = DriverState.RUNNING

        driver.control_bus.enqueue(EventType.PAUSE)
        await driver._generate_chunk()

        assert driver.state == DriverState.PAUSED

    @pytest.mark.asyncio
    async def test_stop_event_stops_driver(self, fake_pipeline):
        """STOP event stops the driver completely."""
        driver = GeneratorDriver(fake_pipeline)
        driver.state = DriverState.RUNNING

        driver.control_bus.enqueue(EventType.STOP)
        result = await driver._generate_chunk()

        assert driver.state == DriverState.STOPPED
        assert result is None

    @pytest.mark.asyncio
    async def test_set_seed_event(self, fake_pipeline):
        """SET_SEED event updates control state."""
        driver = GeneratorDriver(fake_pipeline)

        driver.control_bus.enqueue(
            EventType.SET_SEED,
            payload={"base_seed": 999, "branch_seed_offset": 10},
        )

        await driver.step()

        assert driver.control_state.base_seed == 999
        assert driver.control_state.branch_seed_offset == 10

    @pytest.mark.asyncio
    async def test_set_denoise_steps_event(self, fake_pipeline):
        """SET_DENOISE_STEPS event updates control state."""
        driver = GeneratorDriver(fake_pipeline)

        driver.control_bus.enqueue(
            EventType.SET_DENOISE_STEPS,
            payload={"denoising_step_list": [1000, 500]},
        )

        await driver.step()

        assert driver.control_state.denoising_step_list == [1000, 500]

    @pytest.mark.asyncio
    async def test_set_lora_scales_event(self, fake_pipeline):
        """SET_LORA_SCALES event updates control state."""
        driver = GeneratorDriver(fake_pipeline)

        driver.control_bus.enqueue(
            EventType.SET_LORA_SCALES,
            payload={"lora_scales": [{"path": "/loras/test.safetensors", "scale": 0.5}]},
        )

        await driver.step()

        assert driver.control_state.lora_scales == [
            {"path": "/loras/test.safetensors", "scale": 0.5}
        ]


class TestOnChunkCallback:
    """Tests for on_chunk callback."""

    @pytest.mark.asyncio
    async def test_on_chunk_called_after_generation(self, fake_pipeline):
        """on_chunk is called after each chunk is generated."""
        results = []
        driver = GeneratorDriver(
            fake_pipeline,
            on_chunk=lambda r: results.append(r),
        )

        await driver.step()
        await driver.step()

        assert len(results) == 2
        assert results[0].chunk_index == 1
        assert results[1].chunk_index == 2


class TestSnapshot:
    """Tests for snapshot/restore functionality."""

    @pytest.mark.asyncio
    async def test_snapshot_captures_control_state(self, fake_pipeline):
        """snapshot() includes control state."""
        driver = GeneratorDriver(fake_pipeline)
        driver.control_state.base_seed = 123
        driver.control_state.prompts = [{"text": "test", "weight": 1.0}]

        snapshot = driver.snapshot()

        assert snapshot["control_state"]["base_seed"] == 123
        assert snapshot["control_state"]["prompts"] == [{"text": "test", "weight": 1.0}]

    @pytest.mark.asyncio
    async def test_snapshot_captures_chunk_index(self, fake_pipeline):
        """snapshot() includes chunk_index."""
        driver = GeneratorDriver(fake_pipeline)
        driver.chunk_index = 42

        snapshot = driver.snapshot()

        assert snapshot["chunk_index"] == 42

    @pytest.mark.asyncio
    async def test_snapshot_captures_continuity(self, fake_pipeline_with_continuity):
        """snapshot() includes generator_continuity from pipeline.state."""
        driver = GeneratorDriver(fake_pipeline_with_continuity)

        snapshot = driver.snapshot()

        assert "generator_continuity" in snapshot
        assert snapshot["generator_continuity"]["current_start_frame"] == 9

    @pytest.mark.asyncio
    async def test_restore_sets_control_state(self, fake_pipeline):
        """restore() sets control state from snapshot."""
        driver = GeneratorDriver(fake_pipeline)

        snapshot = {
            "control_state": {
                "base_seed": 999,
                "prompts": [{"text": "restored", "weight": 1.0}],
            },
            "chunk_index": 10,
        }

        driver.restore(snapshot)

        assert driver.control_state.base_seed == 999
        assert driver.control_state.prompts == [{"text": "restored", "weight": 1.0}]
        assert driver.chunk_index == 10

    @pytest.mark.asyncio
    async def test_restore_sets_is_prepared(self, fake_pipeline):
        """restore() sets _is_prepared to True to avoid cache reset."""
        driver = GeneratorDriver(fake_pipeline)
        driver._is_prepared = False

        driver.restore({"control_state": {}, "chunk_index": 5})

        assert driver._is_prepared is True

    @pytest.mark.asyncio
    async def test_restore_continuity_to_pipeline(self, fake_pipeline):
        """restore() restores continuity to pipeline.state."""
        driver = GeneratorDriver(fake_pipeline)

        snapshot = {
            "control_state": {},
            "chunk_index": 10,
            "generator_continuity": {
                "current_start_frame": 99,
                "context_frame_buffer": ["a", "b"],
            },
        }

        driver.restore(snapshot)

        assert fake_pipeline.state.get("current_start_frame") == 99
        assert fake_pipeline.state.get("context_frame_buffer") == ["a", "b"]

    @pytest.mark.asyncio
    async def test_snapshot_restore_roundtrip(self, fake_pipeline_with_continuity):
        """snapshot then restore produces equivalent state."""
        driver = GeneratorDriver(fake_pipeline_with_continuity)
        driver.control_state.base_seed = 777
        driver.control_state.prompts = [{"text": "original", "weight": 1.0}]
        driver.chunk_index = 42

        snapshot = driver.snapshot()

        # Modify state
        driver.control_state.base_seed = 999
        driver.chunk_index = 100

        # Restore
        driver.restore(snapshot)

        assert driver.control_state.base_seed == 777
        assert driver.chunk_index == 42

    @pytest.mark.asyncio
    async def test_restore_via_event(self, fake_pipeline):
        """RESTORE_SNAPSHOT event restores from payload."""
        driver = GeneratorDriver(fake_pipeline)

        snapshot = {
            "control_state": {"base_seed": 555},
            "chunk_index": 25,
        }

        driver.control_bus.enqueue(
            EventType.RESTORE_SNAPSHOT,
            payload={"snapshot": snapshot},
        )

        await driver.step()

        # Restore happens before generation, so seed should be 555
        assert driver.control_state.base_seed == 555


class TestNoGpuDriver:
    """Tests for driver without a real pipeline (pure Python)."""

    @pytest.mark.asyncio
    async def test_step_without_pipeline(self):
        """step() works without a real pipeline."""
        driver = GeneratorDriver(None)

        result = await driver.step()

        assert result is not None
        assert result.frames is None  # No pipeline, no frames
        assert result.chunk_index == 1

    @pytest.mark.asyncio
    async def test_snapshot_without_pipeline(self):
        """snapshot() works without a pipeline (returns empty continuity)."""
        driver = GeneratorDriver(None)
        driver.control_state.base_seed = 123

        snapshot = driver.snapshot()

        assert snapshot["control_state"]["base_seed"] == 123
        assert snapshot["generator_continuity"] == {}

    @pytest.mark.asyncio
    async def test_restore_without_pipeline(self):
        """restore() works without a pipeline."""
        driver = GeneratorDriver(None)

        snapshot = {
            "control_state": {"base_seed": 999},
            "chunk_index": 10,
            "generator_continuity": {"current_start_frame": 99},
        }

        # Should not raise
        driver.restore(snapshot)

        assert driver.control_state.base_seed == 999
        assert driver.chunk_index == 10
