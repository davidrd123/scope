"""
Gemini integration for LLMCompiler.

Provides:
- GeminiCompiler: Implements llm_callable signature for prompt compilation
- GeminiWorldChanger: Natural language WorldState updates
- GeminiPromptJiggler: Prompt variation generator
"""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scope.realtime.world_state import WorldState

logger = logging.getLogger(__name__)

# Default model - Gemini 3 Flash Preview
DEFAULT_MODEL = "gemini-3-flash-preview"

# Rate limiting
_last_call_time = 0.0
_min_call_interval = 0.1  # 10 calls/sec max


def _rate_limit():
    """Simple rate limiter to avoid hitting API limits."""
    global _last_call_time
    now = time.time()
    elapsed = now - _last_call_time
    if elapsed < _min_call_interval:
        time.sleep(_min_call_interval - elapsed)
    _last_call_time = time.time()


def init_client():
    """
    Initialize the Gemini client.

    Reads GEMINI_API_KEY from environment.

    Returns:
        google.genai.Client or None if no API key available
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not set - Gemini features will be unavailable")
        return None

    try:
        from google import genai

        return genai.Client(api_key=api_key)
    except ImportError:
        logger.error("google-genai package not installed")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize Gemini client: {e}")
        return None


class GeminiCompiler:
    """
    Gemini-based LLM compiler.

    Implements the llm_callable signature: (system_prompt, user_message) -> str
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.4,
        max_output_tokens: int = 256,
    ):
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self._client = None

    @property
    def client(self):
        """Lazy-initialize client on first use."""
        if self._client is None:
            self._client = init_client()
        return self._client

    def __call__(self, system_prompt: str, user_message: str) -> str:
        """
        Call Gemini to generate a prompt.

        Matches LLMCompiler's llm_callable signature.

        Args:
            system_prompt: System instructions with vocab, examples, etc.
            user_message: The WorldState context and request

        Returns:
            Generated prompt text

        Raises:
            RuntimeError: If no API key or client unavailable
        """
        if self.client is None:
            raise RuntimeError("Gemini client not available - check GEMINI_API_KEY")

        _rate_limit()

        try:
            from google.genai import types

            response = self.client.models.generate_content(
                model=self.model,
                contents=[user_message],
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=self.temperature,
                    max_output_tokens=self.max_output_tokens,
                    response_mime_type="text/plain",
                ),
            )
            return response.text.strip()

        except Exception as e:
            logger.error(f"Gemini compilation failed: {e}")
            raise


class GeminiWorldChanger:
    """
    Natural language WorldState editor.

    Takes an instruction like "make Rooster angry" and returns updated WorldState.
    """

    SYSTEM_PROMPT = """You are a WorldState editor for an animation system.

Given the current WorldState as JSON and a natural language instruction,
output ONLY valid JSON representing the updated WorldState.

Rules:
- Make minimal changes - only modify what the instruction specifies
- Preserve all fields not mentioned in the instruction
- Valid emotions: happy, sad, angry, frustrated, shocked, neutral, determined, confused, surprised
- Valid beat types: setup, escalation, climax, payoff, reset, transition
- Valid camera intents: establishing, close_up, medium, wide, low_angle, high_angle, tracking, static
- Character actions should be short action verbs or phrases

Output ONLY the JSON - no markdown, no explanation, no comments."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.3,
    ):
        self.model = model
        self.temperature = temperature
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = init_client()
        return self._client

    def change(self, world_state: WorldState, instruction: str) -> WorldState:
        """
        Apply a natural language instruction to update WorldState.

        Args:
            world_state: Current world state
            instruction: Natural language instruction (e.g., "make Rooster angry")

        Returns:
            Updated WorldState

        Raises:
            ValueError: If LLM returns invalid JSON
            RuntimeError: If Gemini unavailable
        """
        from .world_state import WorldState

        if self.client is None:
            raise RuntimeError("Gemini client not available - check GEMINI_API_KEY")

        _rate_limit()

        current_json = world_state.model_dump_json(indent=2)
        user_message = f"""Current WorldState:
{current_json}

Instruction: {instruction}

Output the updated WorldState JSON:"""

        try:
            from google.genai import types

            response = self.client.models.generate_content(
                model=self.model,
                contents=[user_message],
                config=types.GenerateContentConfig(
                    system_instruction=self.SYSTEM_PROMPT,
                    temperature=self.temperature,
                    max_output_tokens=2048,
                    response_mime_type="application/json",
                ),
            )

            response_text = response.text.strip()

            # Parse the JSON response
            return WorldState.model_validate_json(response_text)

        except Exception as e:
            logger.error(f"WorldState change failed: {e}")
            raise ValueError(f"Failed to parse LLM response: {e}") from e


class GeminiPromptJiggler:
    """
    Generates variations of a prompt while preserving meaning.

    Useful for adding visual variety without changing the scene.
    """

    SYSTEM_PROMPT = """You are a prompt variation generator for video generation.

Given a prompt, create a subtle variation that:
- Preserves all core elements (characters, action, camera, style triggers)
- Adjusts word order, synonyms, or emphasis
- Stays within the same token budget
- Maintains the same visual intent

Output ONLY the varied prompt - no explanation, no quotes."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.7,
    ):
        self.model = model
        self.temperature = temperature
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = init_client()
        return self._client

    def jiggle(self, prompt: str, intensity: float = 0.3) -> str:
        """
        Generate a variation of the prompt.

        Args:
            prompt: Original prompt text
            intensity: How different the variation should be (0-1)

        Returns:
            Varied prompt text
        """
        if self.client is None:
            # Graceful fallback - return original
            logger.warning("Gemini unavailable, returning original prompt")
            return prompt

        _rate_limit()

        # Scale temperature with intensity
        adjusted_temp = 0.3 + (intensity * 0.7)  # Range: 0.3 to 1.0

        user_message = f"""Original prompt:
{prompt}

Generate a subtle variation (intensity: {intensity:.1f}):"""

        try:
            from google.genai import types

            response = self.client.models.generate_content(
                model=self.model,
                contents=[user_message],
                config=types.GenerateContentConfig(
                    system_instruction=self.SYSTEM_PROMPT,
                    temperature=adjusted_temp,
                    max_output_tokens=256,
                    response_mime_type="text/plain",
                ),
            )
            return response.text.strip()

        except Exception as e:
            logger.warning(f"Prompt jiggle failed, returning original: {e}")
            return prompt


def is_gemini_available() -> bool:
    """Check if Gemini is configured and available."""
    return os.environ.get("GEMINI_API_KEY") is not None
