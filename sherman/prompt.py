"""System prompt resolution for composable prompts."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def resolve_system_prompt(value: str | list[str], base_dir: Path) -> str:
    """Resolve a system_prompt config value to a string.

    If value is a string, return it directly.
    If value is a list of paths, read each file and concatenate
    with double newlines. Relative paths are resolved against
    base_dir. Absolute paths are used as-is.

    Raises:
        FileNotFoundError: If any path in the list does not exist.
        TypeError: If value is neither str nor list.
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
