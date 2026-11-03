"""Kernel prompt resolution — one helper for every prompt key.

PRD principle 15: every prompt the framework sends to a model on the
agent's behalf is changeable by the operator, and by the agent within
the operator's locks. This module provides the single resolution path
that every prompt key goes through.

Resolution order for a dotted ``key`` in ``config``:

1. **The key names files.** The value is a dict with a ``"file"`` key
   (single path) or a list of path strings. Paths are resolved relative
   to ``config["_base_dir"]`` (the directory containing ``agent.yaml``).
   Files are read, stripped, and concatenated with double newlines.

2. **The key holds a string.** The string is the prompt text, verbatim.
   A bare string that happens to name an existing file is *not* treated
   as a path (same rule as ``system_prompt``).

3. **Otherwise** (key absent, value is ``None``). The caller's
   ``default`` is returned.

Failures:
- A named file that is missing or unreadable raises ``FileNotFoundError``
  with the key and path named. At startup this is fatal (the daemon
  refuses to boot with a broken prompt). On hot config reload the
  failure changes nothing and logs the key (the previous prompt stays
  in effect).

Static vs runtime-tunable:
- ``resolve_prompt`` reads static config (like ``config_value``). The
  agent may change a prompt by editing the file it points at; the
  operator locks this by making the file read-only. Whether a given key
  is also exposed through ``set_settings`` is the operator's decision
  (principle 15) and is handled by the individual plugin, not here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def read_prompt_files(
    paths: list[str],
    base_dir: Path | None,
    key: str | None = None,
) -> str:
    """Read prompt files and concatenate them.

    Shared by :func:`resolve_prompt` and ``channel.resolve_system_prompt``
    so the file-reading logic lives in one place (#411).

    Args:
        paths: File paths to read. Relative paths resolve against base_dir.
        base_dir: Base directory for relative paths. None means CWD.
        key: Optional config key name for error messages.

    Returns:
        The concatenated text, files joined by double newlines.

    Raises:
        FileNotFoundError: If any file does not exist. The message names
            the key (if provided) and the absolute path.
    """
    parts: list[str] = []
    for entry in paths:
        path = Path(str(entry))
        if not path.is_absolute() and base_dir is not None:
            path = base_dir / path
        try:
            text = path.read_text()
        except OSError as exc:
            prefix = f"prompt key {key!r}: " if key else ""
            raise FileNotFoundError(
                f"{prefix}cannot read {path}: {exc}"
            ) from exc
        parts.append(text.strip())
    return "\n\n".join(parts)


def resolve_prompt(
    config: dict,
    key: str,
    default: str,
) -> str:
    """Resolve a prompt from config.

    Args:
        config: The agent config dict (must contain ``_base_dir`` for
            file resolution; falls back to ``Path(".")`` if absent).
        key: Dotted config key, e.g. ``"heartbeat.prompt"`` or
            ``"agent.compaction_fork_prompt"``.
        default: Built-in prompt text, returned when the key is absent
            or its value is ``None``.

    Returns:
        The resolved prompt string.

    Raises:
        FileNotFoundError: A named file does not exist or is unreadable.
            The message names the config key and the absolute path.
        TypeError: The value is neither ``str``, ``list``, ``dict``,
            nor ``None``.
    """
    value = _dotted_get(config, key)
    if value is None:
        return default
    return _resolve_value(value, key, _base_dir(config))


def _base_dir(config: dict) -> Path:
    """The directory containing agent.yaml, from config."""
    bd = config.get("_base_dir")
    if bd is not None:
        return Path(bd)
    return Path(".")


def _dotted_get(config: dict, key: str) -> Any:
    """Walk a dotted key through nested dicts. Returns None if absent."""
    node: Any = config
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _resolve_value(value: Any, key: str, base: Path) -> str:
    """Dispatch on the value's type and resolve to a string."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        if not value:
            logger.warning("prompt key %r: empty file list resolves to empty string", key)
            return ""
        return read_prompt_files(value, base, key=key)
    if isinstance(value, dict):
        path = value.get("file")
        if path is None:
            raise ValueError(
                f"prompt key {key!r}: dict value must have a 'file' key, "
                f"got {value!r}"
            )
        return read_prompt_files([path], base, key=key)
    raise ValueError(
        f"prompt key {key!r}: expected str, list[str], or {{'file': str}}, "
        f"got {value!r}"
    )



