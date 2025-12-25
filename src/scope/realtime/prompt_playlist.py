"""
PromptPlaylist - Load and navigate through a list of prompts from caption files.

Features:
- Load prompts from text files (one per line)
- Trigger phrase swapping (e.g., "1988 Cel Animation" -> "Rankin/Bass Animagic Stop-Motion")
- Navigation: next, prev, goto, current
- Optional shuffle and loop modes
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PromptPlaylist:
    """A navigable playlist of prompts loaded from a caption file."""

    source_file: str = ""
    prompts: list[str] = field(default_factory=list)
    current_index: int = 0

    # Trigger swapping: (old_trigger, new_trigger)
    trigger_swap: tuple[str, str] | None = None

    # Metadata
    original_count: int = 0

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        trigger_swap: tuple[str, str] | None = None,
        skip_empty: bool = True,
    ) -> "PromptPlaylist":
        """
        Load prompts from a text file (one prompt per line).

        Args:
            path: Path to the caption file
            trigger_swap: Optional (old, new) trigger phrase to swap
            skip_empty: Whether to skip empty lines

        Returns:
            PromptPlaylist instance
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Caption file not found: {path}")

        lines = path.read_text().strip().split("\n")
        original_count = len(lines)

        prompts = []
        for line in lines:
            line = line.strip()
            if skip_empty and not line:
                continue

            # Apply trigger swap if configured
            if trigger_swap:
                old_trigger, new_trigger = trigger_swap
                # Case-insensitive replacement at start of line
                if line.lower().startswith(old_trigger.lower()):
                    line = new_trigger + line[len(old_trigger) :]
                else:
                    # Also try replacing anywhere in the line
                    line = re.sub(
                        re.escape(old_trigger),
                        new_trigger,
                        line,
                        flags=re.IGNORECASE,
                    )

            prompts.append(line)

        logger.info(
            f"Loaded {len(prompts)} prompts from {path.name}"
            + (f" (swapped '{trigger_swap[0]}' -> '{trigger_swap[1]}')" if trigger_swap else "")
        )

        return cls(
            source_file=str(path),
            prompts=prompts,
            current_index=0,
            trigger_swap=trigger_swap,
            original_count=original_count,
        )

    @property
    def current(self) -> str:
        """Get the current prompt."""
        if not self.prompts:
            return ""
        return self.prompts[self.current_index]

    @property
    def total(self) -> int:
        """Total number of prompts."""
        return len(self.prompts)

    @property
    def has_next(self) -> bool:
        """Check if there's a next prompt."""
        return self.current_index < len(self.prompts) - 1

    @property
    def has_prev(self) -> bool:
        """Check if there's a previous prompt."""
        return self.current_index > 0

    def next(self) -> str:
        """Move to next prompt and return it."""
        if self.has_next:
            self.current_index += 1
        return self.current

    def prev(self) -> str:
        """Move to previous prompt and return it."""
        if self.has_prev:
            self.current_index -= 1
        return self.current

    def goto(self, index: int) -> str:
        """Go to a specific prompt index."""
        if self.prompts:
            self.current_index = max(0, min(index, len(self.prompts) - 1))
        return self.current

    def first(self) -> str:
        """Go to first prompt."""
        return self.goto(0)

    def last(self) -> str:
        """Go to last prompt."""
        return self.goto(len(self.prompts) - 1)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for API responses."""
        return {
            "source_file": self.source_file,
            "current_index": self.current_index,
            "total": self.total,
            "current_prompt": self.current,
            "has_next": self.has_next,
            "has_prev": self.has_prev,
            "trigger_swap": list(self.trigger_swap) if self.trigger_swap else None,
        }

    def preview(self, context: int = 2) -> dict[str, Any]:
        """Get a preview window around current position."""
        if not self.prompts:
            return {"prompts": [], "current_index": 0}

        start = max(0, self.current_index - context)
        end = min(len(self.prompts), self.current_index + context + 1)

        items = []
        for i in range(start, end):
            # Truncate long prompts for preview
            prompt = self.prompts[i]
            if len(prompt) > 80:
                prompt = prompt[:77] + "..."
            items.append({
                "index": i,
                "prompt": prompt,
                "current": i == self.current_index,
            })

        return {
            "prompts": items,
            "current_index": self.current_index,
            "total": self.total,
        }
