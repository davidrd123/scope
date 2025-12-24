"""Tests for the style layer (StyleManifest, WorldState, PromptCompiler)."""

import tempfile
from pathlib import Path

import pytest

from scope.realtime.style_manifest import StyleManifest, StyleRegistry
from scope.realtime.world_state import (
    BeatType,
    CameraIntent,
    CharacterState,
    PropState,
    WorldState,
    create_simple_world,
    create_character_scene,
)
from scope.realtime.prompt_compiler import (
    CompiledPrompt,
    LoRAScaleUpdate,
    PromptEntry,
    TemplateCompiler,
    CachedCompiler,
    InstructionSheet,
)


class TestStyleManifest:
    """Tests for StyleManifest."""

    def test_default_values(self):
        manifest = StyleManifest(name="test")
        assert manifest.name == "test"
        assert manifest.lora_default_scale == 0.85
        assert manifest.trigger_words == []
        assert manifest.max_prompt_tokens == 77

    def test_with_vocab(self):
        manifest = StyleManifest(
            name="rudolph",
            trigger_words=["rudolph1964", "rankinbass"],
            material_vocab={
                "skin": "felt texture, visible stitching",
                "default": "stop-motion puppet",
            },
            emotion_vocab={
                "frustrated": "furrowed brow, clenched pose",
                "happy": "wide eyes, bouncy movement",
            },
        )
        assert manifest.get_vocab("material", "skin") == "felt texture, visible stitching"
        assert manifest.get_vocab("material", "unknown") == "stop-motion puppet"  # default
        assert manifest.get_vocab("emotion", "frustrated") == "furrowed brow, clenched pose"

    def test_get_vocab_fallback(self):
        manifest = StyleManifest(name="test")
        # No vocab defined, should return the key itself
        assert manifest.get_vocab("material", "wood") == "wood"
        # With explicit default
        assert manifest.get_vocab("material", "wood", "fallback") == "fallback"

    def test_get_all_vocab(self):
        manifest = StyleManifest(
            name="test",
            material_vocab={"skin": "felt"},
            emotion_vocab={"happy": "joyful"},
            custom_vocab={"special": {"magic": "sparkles"}},
        )
        all_vocab = manifest.get_all_vocab()
        assert "material" in all_vocab
        assert "emotion" in all_vocab
        assert "special" in all_vocab
        assert all_vocab["material"]["skin"] == "felt"

    def test_yaml_roundtrip(self):
        manifest = StyleManifest(
            name="test_style",
            lora_path="/path/to/lora.safetensors",
            trigger_words=["trigger1", "trigger2"],
            material_vocab={"wood": "carved balsa"},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.yaml"
            manifest.to_yaml(path)

            loaded = StyleManifest.from_yaml(path)
            assert loaded.name == "test_style"
            assert loaded.lora_path == "/path/to/lora.safetensors"
            assert loaded.trigger_words == ["trigger1", "trigger2"]
            assert loaded.material_vocab["wood"] == "carved balsa"


class TestStyleRegistry:
    """Tests for StyleRegistry."""

    def test_register_and_get(self):
        registry = StyleRegistry()
        manifest = StyleManifest(name="style1")
        registry.register(manifest)

        assert registry.get("style1") == manifest
        assert "style1" in registry
        assert len(registry) == 1

    def test_default_style(self):
        registry = StyleRegistry()
        m1 = StyleManifest(name="first")
        m2 = StyleManifest(name="second")

        registry.register(m1)
        registry.register(m2)

        # First registered is default
        assert registry.get_default() == m1

        registry.set_default("second")
        assert registry.get_default() == m2

    def test_list_styles(self):
        registry = StyleRegistry()
        registry.register(StyleManifest(name="a"))
        registry.register(StyleManifest(name="b"))
        registry.register(StyleManifest(name="c"))

        styles = registry.list_styles()
        assert set(styles) == {"a", "b", "c"}


class TestWorldState:
    """Tests for WorldState."""

    def test_default_values(self):
        world = WorldState()
        assert world.beat == BeatType.SETUP
        assert world.tension == 0.5
        assert world.camera == CameraIntent.MEDIUM
        assert world.characters == []

    def test_with_characters(self):
        world = WorldState(
            action="walking",
            characters=[
                CharacterState(name="rooster", emotion="frustrated", action="pacing"),
                CharacterState(name="terry", emotion="nervous", action="hiding"),
            ],
        )
        assert len(world.characters) == 2
        assert world.get_character("rooster").emotion == "frustrated"
        assert world.get_character("terry").action == "hiding"
        assert world.get_character("unknown") is None

    def test_mood_operations(self):
        world = WorldState()
        world.set_mood("tension", 0.8)
        world.set_mood("comedy", 0.3)

        assert world.get_mood("tension") == 0.8
        assert world.get_mood("comedy") == 0.3
        assert world.get_mood("unknown") == 0.5  # default

        # Clamping
        world.set_mood("extreme", 1.5)
        assert world.get_mood("extreme") == 1.0

    def test_to_context_dict(self):
        world = WorldState(
            scene_description="A snowy forest",
            location="forest",
            action="running",
            beat=BeatType.ESCALATION,
            characters=[
                CharacterState(name="hero", emotion="determined", action="running"),
            ],
        )
        context = world.to_context_dict()

        assert context["scene"] == "A snowy forest"
        assert context["location"] == "forest"
        assert context["action"] == "running"
        assert context["beat"] == "escalation"
        assert context["char_0_name"] == "hero"
        assert context["char_0_emotion"] == "determined"

    def test_to_context_dict_includes_props(self):
        world = WorldState(
            props=[
                PropState(name="banana_peel", location="floor", visible=True, material="rubber"),
                PropState(name="key", location="hidden", visible=False, state="locked"),
            ],
        )
        context = world.to_context_dict()

        assert context["prop_0_name"] == "banana_peel"
        assert context["prop_0_location"] == "floor"
        assert context["prop_0_visible"] is True
        assert context["prop_0_material"] == "rubber"
        assert context["prop_1_name"] == "key"
        assert context["prop_1_visible"] is False
        assert context["prop_1_state"] == "locked"

    def test_create_simple_world(self):
        world = create_simple_world(
            action="jumping",
            emotion="excited",
            camera=CameraIntent.LOW_ANGLE,
            tension=0.7,
        )
        assert world.action == "jumping"
        assert world.camera == CameraIntent.LOW_ANGLE
        assert world.tension == 0.7
        assert len(world.characters) == 1
        assert world.characters[0].emotion == "excited"

    def test_create_character_scene(self):
        world = create_character_scene(
            character_name="villain",
            emotion="menacing",
            action="approaching",
            location="dark alley",
            beat=BeatType.CLIMAX,
        )
        assert world.location == "dark alley"
        assert world.beat == BeatType.CLIMAX
        assert world.focus_target == "villain"
        assert world.characters[0].name == "villain"


class TestTemplateCompiler:
    """Tests for TemplateCompiler."""

    def test_basic_compilation(self):
        style = StyleManifest(
            name="test",
            trigger_words=["trigger1"],
            motion_vocab={"walk": "deliberate walk cycle"},
            emotion_vocab={"happy": "joyful expression"},
            camera_vocab={"medium": "standard framing"},
        )
        world = WorldState(
            action="walking",  # Uses "walking" which normalizes to "walk"
            camera=CameraIntent.MEDIUM,
            characters=[CharacterState(name="hero", emotion="happy")],
        )

        compiler = TemplateCompiler()
        result = compiler.compile(world, style)

        assert isinstance(result, CompiledPrompt)
        assert len(result.prompts) == 1
        assert "trigger1" in result.prompt
        assert "deliberate walk cycle" in result.prompt  # Normalized "walking" -> "walk"
        assert result.style_name == "test"
        assert result.compiler_type == "template"

    def test_action_normalization(self):
        """Test that 'walking' normalizes to 'walk' for vocab lookup."""
        style = StyleManifest(
            name="test",
            motion_vocab={"walk": "puppet walk cycle"},
        )
        world = WorldState(action="walking")

        compiler = TemplateCompiler()
        result = compiler.compile(world, style)

        assert "puppet walk cycle" in result.prompt

    def test_action_normalization_custom_aliases(self):
        """Test that custom aliases in style take precedence."""
        style = StyleManifest(
            name="test",
            motion_vocab={"stroll": "leisurely stroll motion"},
            custom_vocab={"action_aliases": {"walking": "stroll"}},
        )
        world = WorldState(action="walking")

        compiler = TemplateCompiler()
        result = compiler.compile(world, style)

        assert "leisurely stroll motion" in result.prompt

    def test_lora_scales_in_output(self):
        """Test that lora_scales is populated from style."""
        style = StyleManifest(
            name="test",
            lora_path="/path/to/lora.safetensors",
            lora_default_scale=0.75,
        )
        world = WorldState()

        compiler = TemplateCompiler()
        result = compiler.compile(world, style)

        assert result.lora_scales == [
            LoRAScaleUpdate(path="/path/to/lora.safetensors", scale=0.75)
        ]

    def test_to_pipeline_kwargs(self):
        """Test conversion to pipeline-ready kwargs."""
        style = StyleManifest(
            name="test",
            trigger_words=["trigger"],
            lora_path="/path/to/lora.safetensors",
            default_negative="bad quality",
        )
        world = WorldState(action="test")

        compiler = TemplateCompiler()
        result = compiler.compile(world, style)
        kwargs = result.to_pipeline_kwargs()

        assert "prompts" in kwargs
        assert len(kwargs["prompts"]) == 1
        assert kwargs["prompts"][0]["text"] is not None
        assert kwargs["prompts"][0]["weight"] == 1.0
        assert kwargs["negative_prompt"] == "bad quality"
        assert kwargs["lora_scales"] == [{"path": "/path/to/lora.safetensors", "scale": 0.85}]

    def test_negative_prompt(self):
        style = StyleManifest(
            name="test",
            default_negative="realistic, photographic",
        )
        world = WorldState()

        compiler = TemplateCompiler()
        result = compiler.compile(world, style)

        assert result.negative_prompt == "realistic, photographic"

    def test_empty_world_state(self):
        style = StyleManifest(name="minimal")
        world = WorldState()

        compiler = TemplateCompiler()
        result = compiler.compile(world, style)

        # Should not crash, produces some output
        assert result.prompt is not None

    def test_manifest_hash_included(self):
        """Test that manifest hash is included for cache invalidation."""
        style = StyleManifest(name="test", trigger_words=["trigger"])
        world = WorldState()

        compiler = TemplateCompiler()
        result = compiler.compile(world, style)

        assert result.manifest_hash != ""
        assert len(result.manifest_hash) == 12  # MD5 truncated to 12 chars


class TestCachedCompiler:
    """Tests for CachedCompiler."""

    def test_caching(self):
        inner = TemplateCompiler()
        cached = CachedCompiler(inner, max_cache_size=10)

        style = StyleManifest(name="test", trigger_words=["trigger"])
        world = WorldState(action="test")

        # First call
        result1 = cached.compile(world, style)

        # Second call (should hit cache)
        result2 = cached.compile(world, style)

        assert result1.prompt == result2.prompt
        assert result1.world_state_hash == result2.world_state_hash

    def test_cache_invalidation_on_manifest_change(self):
        """Test that changing manifest vocab invalidates cache."""
        inner = TemplateCompiler()
        cached = CachedCompiler(inner, max_cache_size=10)

        style1 = StyleManifest(name="test", motion_vocab={"walk": "old walk"})
        style2 = StyleManifest(name="test", motion_vocab={"walk": "new walk"})
        world = WorldState(action="walking")

        # Compile with style1
        result1 = cached.compile(world, style1)
        assert "old walk" in result1.prompt

        # Compile with style2 (same name, different vocab)
        # Should NOT return cached result because manifest hash differs
        result2 = cached.compile(world, style2)
        assert "new walk" in result2.prompt
        assert result1.manifest_hash != result2.manifest_hash

    def test_lru_eviction(self):
        inner = TemplateCompiler()
        cached = CachedCompiler(inner, max_cache_size=2)

        style = StyleManifest(name="test")

        # Fill cache
        cached.compile(WorldState(action="a"), style)
        cached.compile(WorldState(action="b"), style)

        # This should evict "a"
        cached.compile(WorldState(action="c"), style)

        assert len(cached._cache) == 2

    def test_clear_cache(self):
        inner = TemplateCompiler()
        cached = CachedCompiler(inner, max_cache_size=10)

        style = StyleManifest(name="test")
        cached.compile(WorldState(action="test"), style)

        assert len(cached._cache) == 1

        cached.clear_cache()
        assert len(cached._cache) == 0


class TestInstructionSheet:
    """Tests for InstructionSheet."""

    def test_from_markdown(self):
        content = """# Test Sheet

## System Prompt
You are a prompt compiler. Follow these rules:
- Rule 1
- Rule 2

## Examples

### Example 1
**Input:**
action: walking
emotion: happy

**Output:**
happy character walking, joyful

### Example 2
**Input:**
action: running

**Output:**
fast running motion
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "instructions.md"
            path.write_text(content)

            sheet = InstructionSheet.from_markdown(path)

            assert sheet.name == "Test Sheet"
            assert "Rule 1" in sheet.system_prompt
            assert len(sheet.examples) == 2
            assert "walking" in sheet.examples[0]["world_state"]
            assert "joyful" in sheet.examples[0]["output"]
