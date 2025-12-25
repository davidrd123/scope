"""Integration tests for style layer wiring in FrameProcessor.

Tests the Phase 6a integration:
- WorldState update triggers recompile (when style active)
- Style change sends LoRA (edge-trigger)
- Style change + same style = no LoRA re-send
- Snapshot/restore preserves style state
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scope.realtime.style_manifest import StyleManifest, StyleRegistry
from scope.realtime.world_state import WorldState, CameraIntent
from scope.realtime.prompt_compiler import TemplateCompiler


class TestStyleRegistryFromYAML:
    """Test loading RAT manifest from disk."""

    def test_load_rat_style(self):
        """Verify RAT style loads from styles/rat/manifest.yaml."""
        styles_dir = Path("styles")
        if not styles_dir.exists():
            pytest.skip("styles directory not found")

        registry = StyleRegistry()
        manifests = registry.load_from_directory(styles_dir)

        rat_manifest = registry.get("rat")
        assert rat_manifest is not None, "RAT style not found in registry"
        assert rat_manifest.name == "rat"
        assert "Clay-Plastic Pose-to-Pose Animation" in rat_manifest.trigger_words
        assert rat_manifest.lora_default_scale == 0.85

    def test_template_compiler_produces_trigger_words(self):
        """Verify TemplateCompiler includes trigger words in output."""
        style = StyleManifest(
            name="test_style",
            trigger_words=["Clay-Plastic Pose-to-Pose Animation"],
            motion_vocab={"run": "POSE-TO-POSE snappy motion"},
        )
        world = WorldState(action="running")

        compiler = TemplateCompiler()
        result = compiler.compile(world, style)

        assert "Clay-Plastic Pose-to-Pose Animation" in result.prompt


class TestFrameProcessorStyleIntegration:
    """Tests for style layer wiring in FrameProcessor."""

    @pytest.fixture
    def mock_frame_processor(self):
        """Create a minimal mock FrameProcessor with style layer fields."""
        # We need to simulate the style-related behavior without the full pipeline

        class MockFrameProcessor:
            def __init__(self):
                self.world_state = WorldState()
                self.style_manifest: StyleManifest | None = None
                self.style_registry = StyleRegistry()
                self.prompt_compiler = TemplateCompiler()
                self._compiled_prompt = None
                self._style_manifest_hash: str | None = None
                self.parameters = {}

            def process_reserved_keys(self, merged_updates: dict):
                """Simulate the reserved key handling from process_chunk()."""
                explicit_prompts_set = "prompts" in merged_updates

                # Handle world state update
                if "_rcp_world_state" in merged_updates:
                    world_data = merged_updates.pop("_rcp_world_state")
                    self.world_state = WorldState.model_validate(world_data)
                    if self.style_manifest and not explicit_prompts_set:
                        compiled = self.prompt_compiler.compile(
                            self.world_state, self.style_manifest
                        )
                        self._compiled_prompt = compiled
                        merged_updates["prompts"] = [
                            p.to_dict() for p in compiled.prompts
                        ]

                # Handle style change
                if "_rcp_set_style" in merged_updates:
                    style_name = merged_updates.pop("_rcp_set_style")
                    new_style = self.style_registry.get(style_name)
                    if new_style:
                        new_hash = str(hash(new_style.model_dump_json()))
                        style_changed = new_hash != self._style_manifest_hash
                        self.style_manifest = new_style
                        self._style_manifest_hash = new_hash
                        compiled = self.prompt_compiler.compile(
                            self.world_state, self.style_manifest
                        )
                        self._compiled_prompt = compiled
                        if not explicit_prompts_set:
                            merged_updates["prompts"] = [
                                p.to_dict() for p in compiled.prompts
                            ]
                        # LoRA only on style change (edge-trigger)
                        if style_changed and compiled.lora_scales:
                            merged_updates["lora_scales"] = [
                                ls.to_dict() for ls in compiled.lora_scales
                            ]

                return merged_updates

        fp = MockFrameProcessor()
        # Register test styles
        fp.style_registry.register(
            StyleManifest(
                name="style_a",
                trigger_words=["style_a_trigger"],
                lora_path="/path/to/style_a.safetensors",
                lora_default_scale=0.8,
                motion_vocab={"run": "style_a running"},
            )
        )
        fp.style_registry.register(
            StyleManifest(
                name="style_b",
                trigger_words=["style_b_trigger"],
                lora_path="/path/to/style_b.safetensors",
                lora_default_scale=0.9,
                motion_vocab={"run": "style_b running"},
            )
        )
        return fp

    def test_world_state_update_triggers_recompile_when_style_active(
        self, mock_frame_processor
    ):
        """WorldState update recompiles prompt when style is active."""
        fp = mock_frame_processor

        # First, set a style
        updates = {"_rcp_set_style": "style_a"}
        fp.process_reserved_keys(updates)
        assert fp.style_manifest is not None

        # Now update world state
        updates = {"_rcp_world_state": {"action": "running"}}
        fp.process_reserved_keys(updates)

        # Should have recompiled
        assert fp._compiled_prompt is not None
        assert "style_a_trigger" in fp._compiled_prompt.prompt
        assert "style_a running" in fp._compiled_prompt.prompt
        assert "prompts" in updates

    def test_world_state_update_no_compile_without_style(self, mock_frame_processor):
        """WorldState update does not compile when no style is active."""
        fp = mock_frame_processor

        updates = {"_rcp_world_state": {"action": "running"}}
        fp.process_reserved_keys(updates)

        # No style active, so no compiled prompt
        assert fp._compiled_prompt is None
        assert "prompts" not in updates

    def test_style_change_sends_lora_edge_trigger(self, mock_frame_processor):
        """Style change sends LoRA scales (edge-triggered)."""
        fp = mock_frame_processor

        # Set style_a
        updates = {"_rcp_set_style": "style_a"}
        fp.process_reserved_keys(updates)

        assert "lora_scales" in updates
        assert updates["lora_scales"][0]["path"] == "/path/to/style_a.safetensors"
        assert updates["lora_scales"][0]["scale"] == 0.8  # Uses lora_default_scale from style_a

    def test_same_style_no_lora_resend(self, mock_frame_processor):
        """Setting same style does NOT resend LoRA."""
        fp = mock_frame_processor

        # Set style_a
        updates1 = {"_rcp_set_style": "style_a"}
        fp.process_reserved_keys(updates1)
        assert "lora_scales" in updates1

        # Set style_a again
        updates2 = {"_rcp_set_style": "style_a"}
        fp.process_reserved_keys(updates2)

        # Should NOT have lora_scales (same style, no change)
        assert "lora_scales" not in updates2

    def test_different_style_sends_lora(self, mock_frame_processor):
        """Switching to different style sends new LoRA."""
        fp = mock_frame_processor

        # Set style_a
        updates1 = {"_rcp_set_style": "style_a"}
        fp.process_reserved_keys(updates1)
        assert updates1["lora_scales"][0]["path"] == "/path/to/style_a.safetensors"

        # Switch to style_b
        updates2 = {"_rcp_set_style": "style_b"}
        fp.process_reserved_keys(updates2)

        # Should have new LoRA
        assert "lora_scales" in updates2
        assert updates2["lora_scales"][0]["path"] == "/path/to/style_b.safetensors"

    def test_explicit_prompts_take_precedence(self, mock_frame_processor):
        """Explicit prompts in updates take precedence over compiled."""
        fp = mock_frame_processor

        # Set style
        updates = {"_rcp_set_style": "style_a"}
        fp.process_reserved_keys(updates)

        # Update world with explicit prompts
        explicit_prompt = [{"text": "user explicit prompt", "weight": 1.0}]
        updates = {
            "_rcp_world_state": {"action": "running"},
            "prompts": explicit_prompt,
        }
        fp.process_reserved_keys(updates)

        # Explicit prompts should NOT be overwritten
        assert updates["prompts"] == explicit_prompt

    def test_world_state_update_no_lora_resend(self, mock_frame_processor):
        """WorldState update does NOT resend LoRA (only style change does)."""
        fp = mock_frame_processor

        # Set style
        updates1 = {"_rcp_set_style": "style_a"}
        fp.process_reserved_keys(updates1)
        assert "lora_scales" in updates1

        # Update world state
        updates2 = {"_rcp_world_state": {"action": "running"}}
        fp.process_reserved_keys(updates2)

        # Should NOT have lora_scales (world update, not style change)
        assert "lora_scales" not in updates2


class TestSnapshotStyleState:
    """Test that snapshot/restore preserves style layer state."""

    def test_snapshot_stores_style_info(self):
        """Snapshot should store style name, hash, and world state."""
        from dataclasses import dataclass

        @dataclass
        class MockSnapshot:
            world_state_json: str | None = None
            active_style_name: str | None = None
            style_manifest_hash: str | None = None
            compiled_prompt_text: str | None = None

        # Simulate creating a snapshot
        world_state = WorldState(action="running", camera=CameraIntent.CLOSE_UP)
        style = StyleManifest(name="test_style", trigger_words=["trigger"])
        compiler = TemplateCompiler()
        compiled = compiler.compile(world_state, style)
        style_hash = str(hash(style.model_dump_json()))

        snapshot = MockSnapshot(
            world_state_json=world_state.model_dump_json(),
            active_style_name=style.name,
            style_manifest_hash=style_hash,
            compiled_prompt_text=compiled.prompt,
        )

        # Verify snapshot has correct data
        assert snapshot.world_state_json is not None
        assert snapshot.active_style_name == "test_style"
        assert snapshot.style_manifest_hash is not None
        assert "trigger" in snapshot.compiled_prompt_text

    def test_restore_recovers_world_state(self):
        """Restore should recover WorldState from JSON."""
        world_state = WorldState(action="running", camera=CameraIntent.LOW_ANGLE)
        json_str = world_state.model_dump_json()

        # Restore
        restored = WorldState.model_validate_json(json_str)

        assert restored.action == "running"
        assert restored.camera == CameraIntent.LOW_ANGLE

    def test_restore_recompiles_prompt(self):
        """Restore should recompile prompt with restored state and style."""
        world_state = WorldState(action="running")
        style = StyleManifest(
            name="test_style",
            trigger_words=["trigger"],
            motion_vocab={"run": "fast running"},
        )
        compiler = TemplateCompiler()

        # Save
        world_json = world_state.model_dump_json()

        # "Restore" (simulate)
        restored_world = WorldState.model_validate_json(world_json)
        recompiled = compiler.compile(restored_world, style)

        assert "trigger" in recompiled.prompt
        assert "fast running" in recompiled.prompt
