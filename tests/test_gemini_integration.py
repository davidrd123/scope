"""Tests for Gemini Flash integration.

Tests the GeminiCompiler, GeminiWorldChanger, and create_compiler factory.
Uses mocked Gemini client to avoid API calls.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from scope.realtime import (
    BeatType,
    CameraIntent,
    CharacterState,
    InstructionSheet,
    LLMCompiler,
    TemplateCompiler,
    WorldState,
)
from scope.realtime.prompt_compiler import _load_instruction_sheet, create_compiler
from scope.realtime.style_manifest import StyleManifest


@pytest.fixture
def sample_style():
    """Create a sample style manifest for testing."""
    return StyleManifest(
        name="test_style",
        description="Test style",
        trigger_words=["Test Trigger"],
        motion_vocab={"idle": "stands still", "walk": "walks forward"},
        emotion_vocab={"happy": "smiling", "angry": "scowling"},
        camera_vocab={"close_up": "extreme close-up"},
        beat_vocab={"climax": "peak moment"},
        lora_path="test_lora.safetensors",
        lora_default_scale=0.8,
    )


@pytest.fixture
def sample_world():
    """Create a sample WorldState for testing."""
    return WorldState(
        scene_description="A test scene",
        action="walk",
        camera=CameraIntent.MEDIUM,
        beat=BeatType.SETUP,
        characters=[
            CharacterState(name="TestChar", emotion="happy", action="walking"),
        ],
    )


class TestGeminiCompiler:
    """Tests for GeminiCompiler."""

    def test_compiler_without_api_key(self):
        """Test that GeminiCompiler raises when no API key set."""
        from scope.realtime.gemini_client import GeminiCompiler

        # Ensure no API key
        with patch.dict(os.environ, {}, clear=True):
            if "GEMINI_API_KEY" in os.environ:
                del os.environ["GEMINI_API_KEY"]

            compiler = GeminiCompiler()
            with pytest.raises(RuntimeError, match="client not available"):
                compiler("system", "user")

    def test_compiler_with_mocked_client(self):
        """Test GeminiCompiler with mocked Gemini client."""
        from scope.realtime.gemini_client import GeminiCompiler

        # Mock the client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Generated prompt text"
        mock_client.models.generate_content.return_value = mock_response

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            compiler = GeminiCompiler()
            compiler._client = mock_client

            result = compiler("system prompt", "user message")

            assert result == "Generated prompt text"
            mock_client.models.generate_content.assert_called_once()


class TestGeminiWorldChanger:
    """Tests for GeminiWorldChanger."""

    def test_changer_without_api_key(self, sample_world):
        """Test that GeminiWorldChanger raises when no API key set."""
        from scope.realtime.gemini_client import GeminiWorldChanger

        with patch.dict(os.environ, {}, clear=True):
            if "GEMINI_API_KEY" in os.environ:
                del os.environ["GEMINI_API_KEY"]

            changer = GeminiWorldChanger()
            with pytest.raises(RuntimeError, match="client not available"):
                changer.change(sample_world, "make character angry")

    def test_changer_with_mocked_client(self, sample_world):
        """Test GeminiWorldChanger with mocked Gemini client."""
        from scope.realtime.gemini_client import GeminiWorldChanger

        # Mock response with updated WorldState JSON
        updated_world = sample_world.model_copy()
        updated_world.characters[0].emotion = "angry"
        updated_world.action = "storming_out"

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = updated_world.model_dump_json()
        mock_client.models.generate_content.return_value = mock_response

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            changer = GeminiWorldChanger()
            changer._client = mock_client

            result = changer.change(sample_world, "make character angry and storm out")

            assert result.characters[0].emotion == "angry"
            assert result.action == "storming_out"


class TestGeminiPromptJiggler:
    """Tests for GeminiPromptJiggler."""

    def test_jiggler_without_api_key(self):
        """Test that GeminiPromptJiggler returns original when no API key."""
        from scope.realtime.gemini_client import GeminiPromptJiggler

        with patch.dict(os.environ, {}, clear=True):
            if "GEMINI_API_KEY" in os.environ:
                del os.environ["GEMINI_API_KEY"]

            jiggler = GeminiPromptJiggler()
            # Should return original (graceful fallback)
            result = jiggler.jiggle("Original prompt text")
            assert result == "Original prompt text"

    def test_jiggler_with_mocked_client(self):
        """Test GeminiPromptJiggler with mocked Gemini client."""
        from scope.realtime.gemini_client import GeminiPromptJiggler

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Jiggled prompt variation"
        mock_client.models.generate_content.return_value = mock_response

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            jiggler = GeminiPromptJiggler()
            jiggler._client = mock_client

            result = jiggler.jiggle("Original prompt", intensity=0.5)

            assert result == "Jiggled prompt variation"
            mock_client.models.generate_content.assert_called_once()


class TestCreateCompiler:
    """Tests for create_compiler factory function."""

    def test_template_mode_explicit(self, sample_style):
        """Test that mode='template' returns TemplateCompiler."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove env override if present
            if "SCOPE_LLM_COMPILER" in os.environ:
                del os.environ["SCOPE_LLM_COMPILER"]

            compiler = create_compiler(sample_style, mode="template")
            assert isinstance(compiler, TemplateCompiler)

    def test_auto_mode_without_api_key(self, sample_style):
        """Test that auto mode falls back to template when no API key."""
        with patch.dict(os.environ, {}, clear=True):
            if "GEMINI_API_KEY" in os.environ:
                del os.environ["GEMINI_API_KEY"]
            if "SCOPE_LLM_COMPILER" in os.environ:
                del os.environ["SCOPE_LLM_COMPILER"]

            compiler = create_compiler(sample_style, mode="auto")
            assert isinstance(compiler, TemplateCompiler)

    def test_env_override_to_template(self, sample_style):
        """Test that SCOPE_LLM_COMPILER=template overrides mode."""
        with patch.dict(
            os.environ, {"SCOPE_LLM_COMPILER": "template", "GEMINI_API_KEY": "test-key"}
        ):
            compiler = create_compiler(sample_style, mode="gemini")
            # Should be template due to env override
            assert isinstance(compiler, TemplateCompiler)

    def test_gemini_mode_with_api_key(self, sample_style):
        """Test that gemini mode creates CachedCompiler wrapping LLMCompiler."""
        from scope.realtime.prompt_compiler import CachedCompiler

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            if "SCOPE_LLM_COMPILER" in os.environ:
                del os.environ["SCOPE_LLM_COMPILER"]

            compiler = create_compiler(sample_style, mode="gemini")
            assert isinstance(compiler, CachedCompiler)
            assert isinstance(compiler.inner, LLMCompiler)


class TestInstructionSheet:
    """Tests for InstructionSheet parsing."""

    def test_parse_markdown(self, tmp_path):
        """Test parsing instruction sheet from markdown."""
        md_content = """# Test Style Compiler

## System Prompt

You are a test prompt compiler.
Follow these rules:
1. Always start with trigger
2. Maximum 50 tokens

## Examples

### Example 1
**Input:**
scene: Kitchen
action: walking

**Output:**
Test Trigger, character walking in kitchen

### Example 2
**Input:**
scene: Garden
emotion: happy

**Output:**
Test Trigger, happy character in garden
"""
        md_file = tmp_path / "instructions.md"
        md_file.write_text(md_content)

        sheet = InstructionSheet.from_markdown(md_file)

        assert sheet.name == "Test Style Compiler"
        assert "test prompt compiler" in sheet.system_prompt
        assert len(sheet.examples) == 2
        assert "Kitchen" in sheet.examples[0]["world_state"]
        assert "Test Trigger" in sheet.examples[0]["output"]

    def test_load_instruction_sheet_not_found(self, sample_style):
        """Test that _load_instruction_sheet returns None for non-existent style."""
        # Use a style name that won't have an instructions.md
        sample_style.name = "nonexistent_style_xyz"
        result = _load_instruction_sheet(sample_style)
        assert result is None


class TestLLMCompilerWithGemini:
    """Tests for LLMCompiler using GeminiCompiler."""

    def test_llm_compiler_fallback_on_error(self, sample_style, sample_world):
        """Test that LLMCompiler falls back to template on error."""

        def failing_llm(system, user):
            raise Exception("Simulated API error")

        compiler = LLMCompiler(llm_callable=failing_llm)
        result = compiler.compile(sample_world, sample_style)

        # Should have fallen back to template
        assert result.compiler_type == "template_fallback"
        assert "Test Trigger" in result.prompt

    def test_llm_compiler_success(self, sample_style, sample_world):
        """Test LLMCompiler with successful LLM call."""

        def mock_llm(system, user):
            return "LLM generated prompt text"

        compiler = LLMCompiler(llm_callable=mock_llm)
        result = compiler.compile(sample_world, sample_style)

        assert result.compiler_type == "llm"
        assert result.prompt == "LLM generated prompt text"
        assert result.raw_llm_response == "LLM generated prompt text"


class TestIsGeminiAvailable:
    """Tests for is_gemini_available helper."""

    def test_available_when_key_set(self):
        """Test that is_gemini_available returns True when key is set."""
        from scope.realtime.gemini_client import is_gemini_available

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            assert is_gemini_available() is True

    def test_not_available_when_no_key(self):
        """Test that is_gemini_available returns False when no key."""
        from scope.realtime.gemini_client import is_gemini_available

        with patch.dict(os.environ, {}, clear=True):
            if "GEMINI_API_KEY" in os.environ:
                del os.environ["GEMINI_API_KEY"]
            assert is_gemini_available() is False
