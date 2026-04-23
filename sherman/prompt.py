"""System prompt resolution for composable prompts.

This module handles system_prompt config values which can be:
- A literal string (returned as-is)
- A list of file paths (concatenated with double newlines)

Logging:
    - DEBUG: system prompt resolved (string vs file list, length)
    - WARNING: empty list resolved to empty string
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def resolve_system_prompt(value: str | list[str], base_dir: Path) -> str:
    """Resolve a system_prompt config value to a string.

    If value is a string, return it directly.
    If value is a list of paths, read each file and concatenate
    with double newlines. Relative paths are resolved against
    base_dir. Absolute paths are used as-is.

    Args:
        value: Either a literal prompt string or a list of file paths
        base_dir: Base directory for resolving relative paths

    Returns:
        The resolved system prompt as a single string

    Raises:
        FileNotFoundError: If any path in the list does not exist.
        TypeError: If value is neither str nor list.

    Logs:
        DEBUG: Resolution method (string/file list) and result length
        WARNING: Empty list resolves to empty string
    """
    if isinstance(value, str):
        logger.debug(
            "system prompt resolved from literal string",
            extra={"length": len(value)},
        )
        return value
    if isinstance(value, list):
        if not value:
            logger.warning("empty system prompt list resolved to empty string")
            return ""
        parts = []
        for entry in value:
            path = Path(entry)
            if not path.is_absolute():
                path = base_dir / path
            parts.append(path.read_text().strip())
        result = "\n\n".join(parts)
        logger.debug(
            "system prompt resolved from file list",
            extra={"file_count": len(value), "length": len(result)},
        )
        return result
    raise TypeError(
        f"system_prompt must be str or list[str], got {type(value).__name__!r}"
    )
