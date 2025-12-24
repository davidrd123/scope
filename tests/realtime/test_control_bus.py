"""Tests for ControlBus, ControlEvent, and event semantics."""

import time

import pytest

from scope.realtime.control_bus import (
    ApplyMode,
    ControlBus,
    ControlEvent,
    EventType,
    EVENT_TYPE_ORDER,
    pause_event,
    prompt_event,
    world_state_event,
)


class TestControlEvent:
    """Tests for ControlEvent dataclass."""

    def test_default_values(self):
        """ControlEvent has sensible defaults."""
        event = ControlEvent(type=EventType.SET_PROMPT)

        assert event.type == EventType.SET_PROMPT
        assert event.payload == {}
        assert event.apply_mode == ApplyMode.NEXT_BOUNDARY
        assert event.source == "api"
        assert event.applied_chunk_index is None
        assert event.timestamp > 0
        assert event.event_id  # Non-empty

    def test_with_payload(self):
        """ControlEvent stores payload."""
        event = ControlEvent(
            type=EventType.SET_PROMPT,
            payload={"prompts": [{"text": "test", "weight": 1.0}]},
        )

        assert event.payload["prompts"] == [{"text": "test", "weight": 1.0}]

    def test_with_apply_mode(self):
        """ControlEvent respects apply_mode."""
        event = ControlEvent(
            type=EventType.PAUSE,
            apply_mode=ApplyMode.IMMEDIATE_IF_PAUSED,
        )

        assert event.apply_mode == ApplyMode.IMMEDIATE_IF_PAUSED


class TestControlBus:
    """Tests for ControlBus event queue."""

    def test_enqueue_creates_event(self):
        """enqueue creates and stores an event."""
        bus = ControlBus()

        event = bus.enqueue(
            EventType.SET_PROMPT,
            payload={"prompts": []},
            source="test",
        )

        assert event.type == EventType.SET_PROMPT
        assert len(bus.pending) == 1
        assert bus.pending[0] is event

    def test_drain_pending_returns_events(self):
        """drain_pending returns all pending events."""
        bus = ControlBus()
        bus.enqueue(EventType.SET_PROMPT)
        bus.enqueue(EventType.SET_SEED)

        events = bus.drain_pending()

        assert len(events) == 2
        assert len(bus.pending) == 0  # Queue is drained

    def test_drain_pending_records_chunk_index(self):
        """drain_pending sets applied_chunk_index on events."""
        bus = ControlBus()
        bus.enqueue(EventType.SET_PROMPT)

        events = bus.drain_pending(chunk_index=42)

        assert events[0].applied_chunk_index == 42

    def test_drain_pending_adds_to_history(self):
        """drain_pending moves events to history."""
        bus = ControlBus()
        bus.enqueue(EventType.SET_PROMPT)

        bus.drain_pending()

        assert len(bus.history) == 1
        assert bus.history[0].type == EventType.SET_PROMPT


class TestEventOrdering:
    """Tests for deterministic event ordering at chunk boundaries."""

    def test_lifecycle_before_style(self):
        """Lifecycle events are applied before style events."""
        bus = ControlBus()
        bus.enqueue(EventType.SET_STYLE_MANIFEST)
        bus.enqueue(EventType.PAUSE)

        events = bus.drain_pending()

        assert events[0].type == EventType.PAUSE
        assert events[1].type == EventType.SET_STYLE_MANIFEST

    def test_style_before_world(self):
        """Style events are applied before world events."""
        bus = ControlBus()
        bus.enqueue(EventType.SET_WORLD_STATE)
        bus.enqueue(EventType.SET_STYLE_MANIFEST)

        events = bus.drain_pending()

        assert events[0].type == EventType.SET_STYLE_MANIFEST
        assert events[1].type == EventType.SET_WORLD_STATE

    def test_world_before_prompt(self):
        """World events are applied before prompt events."""
        bus = ControlBus()
        bus.enqueue(EventType.SET_PROMPT)
        bus.enqueue(EventType.SET_WORLD_STATE)

        events = bus.drain_pending()

        assert events[0].type == EventType.SET_WORLD_STATE
        assert events[1].type == EventType.SET_PROMPT

    def test_prompt_before_params(self):
        """Prompt events are applied before param events."""
        bus = ControlBus()
        bus.enqueue(EventType.SET_LORA_SCALES)
        bus.enqueue(EventType.SET_PROMPT)

        events = bus.drain_pending()

        assert events[0].type == EventType.SET_PROMPT
        assert events[1].type == EventType.SET_LORA_SCALES

    def test_full_ordering(self):
        """Full event ordering matches spec.

        Order: lifecycle → snapshot/restore → style → world → prompt → params
        """
        bus = ControlBus()

        # Add in reverse order to test sorting
        bus.enqueue(EventType.SET_LORA_SCALES)  # params
        bus.enqueue(EventType.SET_SEED)  # params
        bus.enqueue(EventType.SET_PROMPT)  # prompt
        bus.enqueue(EventType.SET_WORLD_STATE)  # world
        bus.enqueue(EventType.SET_STYLE_MANIFEST)  # style
        bus.enqueue(EventType.SNAPSHOT_REQUEST)  # snapshot
        bus.enqueue(EventType.RESTORE_SNAPSHOT)  # restore
        bus.enqueue(EventType.PAUSE)  # lifecycle

        events = bus.drain_pending()

        types = [e.type for e in events]

        # Verify order
        assert types.index(EventType.PAUSE) < types.index(EventType.RESTORE_SNAPSHOT)
        assert types.index(EventType.RESTORE_SNAPSHOT) < types.index(
            EventType.SNAPSHOT_REQUEST
        )
        assert types.index(EventType.SNAPSHOT_REQUEST) < types.index(
            EventType.SET_STYLE_MANIFEST
        )
        assert types.index(EventType.SET_STYLE_MANIFEST) < types.index(
            EventType.SET_WORLD_STATE
        )
        assert types.index(EventType.SET_WORLD_STATE) < types.index(EventType.SET_PROMPT)
        assert types.index(EventType.SET_PROMPT) < types.index(EventType.SET_SEED)

    def test_same_type_ordered_by_timestamp(self):
        """Events of the same type are ordered by timestamp."""
        bus = ControlBus()

        # Create events with explicit timestamps
        e1 = ControlEvent(type=EventType.SET_PROMPT, timestamp=100.0)
        e2 = ControlEvent(type=EventType.SET_PROMPT, timestamp=50.0)
        e3 = ControlEvent(type=EventType.SET_PROMPT, timestamp=75.0)

        bus.pending.append(e1)
        bus.pending.append(e2)
        bus.pending.append(e3)

        events = bus.drain_pending()

        assert events[0].timestamp == 50.0
        assert events[1].timestamp == 75.0
        assert events[2].timestamp == 100.0


class TestApplyModes:
    """Tests for NEXT_BOUNDARY vs IMMEDIATE_IF_PAUSED filtering."""

    def test_next_boundary_always_applies(self):
        """NEXT_BOUNDARY events always apply at drain."""
        bus = ControlBus()
        bus.enqueue(EventType.SET_PROMPT, apply_mode=ApplyMode.NEXT_BOUNDARY)

        # Not paused
        events = bus.drain_pending(is_paused=False)
        assert len(events) == 1

    def test_immediate_if_paused_applies_when_paused(self):
        """IMMEDIATE_IF_PAUSED events apply when paused."""
        bus = ControlBus()
        bus.enqueue(EventType.SET_PROMPT, apply_mode=ApplyMode.IMMEDIATE_IF_PAUSED)

        events = bus.drain_pending(is_paused=True)
        assert len(events) == 1

    def test_immediate_if_paused_waits_when_running(self):
        """IMMEDIATE_IF_PAUSED events wait when not paused."""
        bus = ControlBus()
        bus.enqueue(EventType.SET_PROMPT, apply_mode=ApplyMode.IMMEDIATE_IF_PAUSED)

        events = bus.drain_pending(is_paused=False)

        assert len(events) == 0
        assert len(bus.pending) == 1  # Still pending

    def test_mixed_apply_modes(self):
        """Mix of apply modes filters correctly."""
        bus = ControlBus()
        bus.enqueue(EventType.SET_PROMPT, apply_mode=ApplyMode.NEXT_BOUNDARY)
        bus.enqueue(EventType.SET_SEED, apply_mode=ApplyMode.IMMEDIATE_IF_PAUSED)

        # Not paused - only NEXT_BOUNDARY applies
        events = bus.drain_pending(is_paused=False)

        assert len(events) == 1
        assert events[0].type == EventType.SET_PROMPT
        assert len(bus.pending) == 1
        assert bus.pending[0].type == EventType.SET_SEED


class TestHistory:
    """Tests for event history tracking."""

    def test_history_stores_applied_events(self):
        """Applied events are stored in history."""
        bus = ControlBus()
        bus.enqueue(EventType.SET_PROMPT)
        bus.enqueue(EventType.SET_SEED)

        bus.drain_pending()

        assert len(bus.history) == 2

    def test_history_max_limit(self):
        """History is limited to max_history entries."""
        bus = ControlBus(max_history=5)

        for i in range(10):
            bus.enqueue(EventType.SET_PROMPT)
            bus.drain_pending()

        assert len(bus.history) == 5

    def test_get_history_filters_by_timestamp(self):
        """get_history can filter by timestamp."""
        bus = ControlBus()

        e1 = ControlEvent(type=EventType.SET_PROMPT, timestamp=100.0)
        e2 = ControlEvent(type=EventType.SET_PROMPT, timestamp=200.0)
        bus.pending.append(e1)
        bus.pending.append(e2)
        bus.drain_pending()

        filtered = bus.get_history(since_timestamp=150.0)

        assert len(filtered) == 1
        assert filtered[0].timestamp == 200.0

    def test_get_history_filters_by_event_type(self):
        """get_history can filter by event type."""
        bus = ControlBus()
        bus.enqueue(EventType.SET_PROMPT)
        bus.enqueue(EventType.SET_SEED)
        bus.drain_pending()

        filtered = bus.get_history(event_types=[EventType.SET_PROMPT])

        assert len(filtered) == 1
        assert filtered[0].type == EventType.SET_PROMPT

    def test_clear_pending(self):
        """clear_pending removes all pending events."""
        bus = ControlBus()
        bus.enqueue(EventType.SET_PROMPT)
        bus.enqueue(EventType.SET_SEED)

        bus.clear_pending()

        assert len(bus.pending) == 0


class TestConvenienceFunctions:
    """Tests for event creation convenience functions."""

    def test_prompt_event(self):
        """prompt_event creates correct event."""
        event = prompt_event(
            prompts=[{"text": "test", "weight": 1.0}],
            source="test",
        )

        assert event.type == EventType.SET_PROMPT
        assert event.payload["prompts"] == [{"text": "test", "weight": 1.0}]
        assert event.source == "test"

    def test_prompt_event_with_transition(self):
        """prompt_event includes transition when provided."""
        event = prompt_event(
            prompts=[],
            transition={"num_steps": 4},
        )

        assert event.payload["transition"] == {"num_steps": 4}

    def test_world_state_event(self):
        """world_state_event creates correct event."""
        event = world_state_event(
            updates={"tension_level": 0.8},
            source="vlm",
        )

        assert event.type == EventType.SET_WORLD_STATE
        assert event.payload["tension_level"] == 0.8
        assert event.source == "vlm"

    def test_pause_event(self):
        """pause_event creates correct event with NEXT_BOUNDARY mode."""
        event = pause_event(source="api")

        assert event.type == EventType.PAUSE
        assert event.apply_mode == ApplyMode.NEXT_BOUNDARY  # NOT IMMEDIATE_IF_PAUSED
        assert event.source == "api"
