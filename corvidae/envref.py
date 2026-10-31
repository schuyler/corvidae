"""Expand ``${VAR}`` references in config values from the environment.

Shared by MCP ``headers`` and ``llm.<role>.api_key`` so one convention covers
every config value that may carry a secret. A value with no ``${NAME}`` is a
literal and is returned unchanged.
"""

import os
import re

# ${VAR} references, substituted from the environment.
_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class EnvRefError(ValueError):
    """A ${VAR} reference names a variable that is unset, empty or unusable.

    Carries the variable name and the reason; never the variable's value, so
    a secret cannot reach a log through it.
    """

    def __init__(self, var: str, reason: str) -> None:
        super().__init__(f"${{{var}}}: {reason}")
        # Variable name and failure kind ("unset" or "multiline") for callers
        # that rebuild the message in their own words.
        self.var = var
        self.reason = reason


def expand_env_refs(value: str) -> str:
    """Return ``value`` with every ${VAR} replaced from the environment.

    Raises EnvRefError (reason ``"unset"`` for unset or empty, ``"multiline"``
    for a value containing CR, LF or NUL) naming the variable, never its value.
    """
    # Substitute each ${VAR}, refusing values that are unusable as a header.
    def _lookup(match: re.Match) -> str:
        var = match.group(1)
        env_value = os.environ.get(var)
        if not env_value:
            raise EnvRefError(var, "unset")
        # CR, LF or NUL is illegal in a header, and httpx/aiohttp errors for it
        # quote the value into a logged traceback; refuse it here.
        if any(ch in env_value for ch in "\r\n\0"):
            raise EnvRefError(var, "multiline")
        return env_value

    return _ENV_REF.sub(_lookup, value)
