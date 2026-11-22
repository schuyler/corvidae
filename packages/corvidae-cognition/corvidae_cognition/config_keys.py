"""Readers for a setting that moved from a ``heartbeat.*`` key to its owner (#1079),
and the warning for a key that was removed outright (#1185).

Memory and goals each read their beat settings through these, so the
validation and the old-key rules are one implementation, not three that could
drift apart. Neither plugin imports the other's config code. Appraisal and
gates report the stage-1 keys #1185 removed through ``warn_removed_keys``.

Rules, for a setting with a ``new_key`` and a retired ``old_key``:
    - a key set to ``null``, or whose parent block is not a mapping, is absent;
    - both present: ValueError naming both keys and the remedy;
    - only the old key present: it still takes effect, with a WARNING naming
      the new key (the old keys are dropped after one release);
    - neither: the default.
"""

from __future__ import annotations

import logging
from typing import Any

from corvidae.prompts import _dotted_get, resolve_prompt

logger = logging.getLogger("corvidae.config_keys")


def _pick_key(config: dict, new_key: str, old_key: str) -> str | None:
    """Return the key that is set (new wins only when alone), or None.

    Raises ValueError, naming both keys and the remedy, when both are set.
    """
    new_set = _dotted_get(config, new_key) is not None
    old_set = _dotted_get(config, old_key) is not None
    # The same setting under both names is ambiguous: refuse, don't guess.
    if new_set and old_set:
        raise ValueError(
            f"{old_key} and {new_key} are both set; they are the same setting — "
            f"remove {old_key} from agent.yaml and keep {new_key}"
        )
    if old_set:
        # Old key still works for one release; point at its replacement.
        logger.warning(
            "%s is deprecated and will stop working in a later release; "
            "move its value to %s in agent.yaml",
            old_key,
            new_key,
        )
        return old_key
    return new_key if new_set else None


def read_count(
    config: dict, new_key: str, old_key: str, default: int, noun: str
) -> int:
    """Return a whole-number count setting, ``default`` when absent.

    ``noun`` names what is counted in the error text. Raises ValueError
    naming the key, the value and what is accepted for a boolean, a
    non-integer or a negative value, and for both keys set.
    """
    key = _pick_key(config, new_key, old_key)
    if key is None:
        return default
    value: Any = _dotted_get(config, key)
    # bool is an int subclass; True is not a count. 2.5 and "3" are not either.
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"{key}: got {value!r}; expected a whole number of {noun}, "
            "0 or more (0 means none) — check agent.yaml, "
            f"or remove the key for the default of {default}"
        )
    return value


def warn_removed_keys(config: dict, removed: dict[str, str]) -> list[str]:
    """Log one WARNING per removed key that ``config`` still sets; return them.

    ``removed`` maps a dotted path to why it configures nothing now. A path
    whose value is a mapping (a whole removed block, such as
    ``appraisal.probe``) is reported key by key; a key set to ``null`` is
    absent. The daemon keeps running: a key that configures nothing is
    ignored, and the warning says to delete it.
    """
    found: list[str] = []
    for path, reason in removed.items():
        value = _dotted_get(config, path)
        # Absent, or set to null: nothing to report.
        if value is None:
            continue
        # A removed block names each key the operator set inside it.
        keys = [f"{path}.{key}" for key in value] if isinstance(value, dict) else [path]
        for key in keys:
            logger.warning(
                "agent.yaml: %s was removed (#1185): %s, so it does nothing; "
                "delete it from agent.yaml",
                key,
                reason,
                extra={"key": key},
            )
            found.append(key)
    return found


def read_prompt(config: dict, new_key: str, old_key: str, default: str) -> str:
    """Return a prompt setting (inline, ``{file:}`` or list), ``default`` when absent.

    Raises ValueError for both keys set; ``resolve_prompt`` raises for a bad
    value or an unreadable file.
    """
    key = _pick_key(config, new_key, old_key)
    return default if key is None else resolve_prompt(config, key, default)
