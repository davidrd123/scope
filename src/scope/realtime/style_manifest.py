"""
StyleManifest - LoRA-specific vocabulary and metadata.

A StyleManifest captures everything needed to translate abstract world concepts
into effective prompt tokens for a specific LoRA/style.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class StyleManifest(BaseModel):
    """
    Metadata and vocabulary for a specific LoRA/style.

    The vocab dictionaries map abstract concepts to effective prompt tokens
    discovered through experimentation with this specific LoRA.
    """

    # Identity
    name: str
    description: str = ""

    # LoRA configuration
    lora_path: str | None = None
    lora_default_scale: float = 0.85

    # Trigger words (always included in prompt)
    trigger_words: list[str] = Field(default_factory=list)

    # Vocabulary mappings: abstract concept → effective tokens
    # These are populated from your prompt experiments
    material_vocab: dict[str, str] = Field(default_factory=dict)
    motion_vocab: dict[str, str] = Field(default_factory=dict)
    camera_vocab: dict[str, str] = Field(default_factory=dict)
    lighting_vocab: dict[str, str] = Field(default_factory=dict)
    emotion_vocab: dict[str, str] = Field(default_factory=dict)
    beat_vocab: dict[str, str] = Field(default_factory=dict)

    # Custom vocab categories (extensible)
    custom_vocab: dict[str, dict[str, str]] = Field(default_factory=dict)

    # Prompt constraints
    default_negative: str = ""
    max_prompt_tokens: int = 77

    # Priority order for token budget allocation
    # Earlier items are kept when truncating
    priority_order: list[str] = Field(
        default_factory=lambda: [
            "trigger",
            "action",
            "material",
            "camera",
            "mood",
        ]
    )

    # Path to instruction sheet (markdown/text with LLM instructions)
    instruction_sheet_path: str | None = None

    # Additional metadata
    metadata: dict[str, Any] = Field(default_factory=dict)

    def get_vocab(self, category: str, key: str, default: str | None = None) -> str:
        """
        Look up a vocab term by category and key.

        Args:
            category: Vocab category (material, motion, camera, etc.)
            key: The abstract term to look up
            default: Fallback if not found

        Returns:
            The effective prompt tokens, or default/key if not found
        """
        vocab_dict = getattr(self, f"{category}_vocab", None)
        if vocab_dict is None:
            vocab_dict = self.custom_vocab.get(category, {})

        result = vocab_dict.get(key)
        if result is not None:
            return result

        # Check for "default" key in vocab
        if "default" in vocab_dict:
            return vocab_dict["default"]

        return default if default is not None else key

    def get_all_vocab(self) -> dict[str, dict[str, str]]:
        """Return all vocab dictionaries merged."""
        all_vocab = {
            "material": self.material_vocab,
            "motion": self.motion_vocab,
            "camera": self.camera_vocab,
            "lighting": self.lighting_vocab,
            "emotion": self.emotion_vocab,
            "beat": self.beat_vocab,
        }
        all_vocab.update(self.custom_vocab)
        return {k: v for k, v in all_vocab.items() if v}

    @classmethod
    def from_yaml(cls, path: str | Path) -> "StyleManifest":
        """Load a StyleManifest from a YAML file."""
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Save this StyleManifest to a YAML file."""
        path = Path(path)
        with open(path, "w") as f:
            yaml.dump(self.model_dump(exclude_none=True), f, default_flow_style=False)


class StyleRegistry:
    """
    Registry for loading and caching StyleManifests.

    Manifests can be loaded from:
    - Individual YAML files
    - A directory of manifests
    - Programmatic registration
    """

    def __init__(self):
        self._manifests: dict[str, StyleManifest] = {}
        self._default_style: str | None = None

    def register(self, manifest: StyleManifest) -> None:
        """Register a manifest by name."""
        self._manifests[manifest.name] = manifest
        if self._default_style is None:
            self._default_style = manifest.name

    def load_from_file(self, path: str | Path) -> StyleManifest:
        """Load and register a manifest from a YAML file."""
        manifest = StyleManifest.from_yaml(path)
        self.register(manifest)
        return manifest

    def load_from_directory(self, directory: str | Path) -> list[StyleManifest]:
        """Load all manifest.yaml files from a directory tree."""
        directory = Path(directory)
        manifests = []
        for manifest_path in directory.rglob("manifest.yaml"):
            try:
                manifest = self.load_from_file(manifest_path)
                manifests.append(manifest)
            except Exception as e:
                # Log but don't fail on individual manifest errors
                print(f"Warning: Failed to load {manifest_path}: {e}")
        return manifests

    def get(self, name: str) -> StyleManifest | None:
        """Get a manifest by name."""
        return self._manifests.get(name)

    def get_default(self) -> StyleManifest | None:
        """Get the default style manifest."""
        if self._default_style:
            return self._manifests.get(self._default_style)
        return None

    def set_default(self, name: str) -> None:
        """Set the default style by name."""
        if name not in self._manifests:
            raise ValueError(f"Style '{name}' not found in registry")
        self._default_style = name

    def list_styles(self) -> list[str]:
        """List all registered style names."""
        return list(self._manifests.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._manifests

    def __len__(self) -> int:
        return len(self._manifests)
