"""System locale support: date/time names and the web-search region (#1186).

Stdlib ``locale`` only. ``adopt_system_locale`` is called once, from
``run_daemon``, before any thread exists (``setlocale`` is not thread-safe);
the other functions only query the adopted ``LC_TIME``.
"""

from __future__ import annotations

import locale
import logging
import os
from typing import Final

logger = logging.getLogger(__name__)

# One localized date/time form for every site under a real locale
# (de_DE: "Montag 23.11.2026 14:20 CET").
LOCALIZED_DATETIME_FORMAT: Final[str] = "%A %x %H:%M %Z"

# ddgs region meaning "no region" (also what the C locale gets).
NO_REGION: Final[str] = "wt-wt"


def adopt_system_locale() -> str | None:
    """Adopt the environment's ``LC_TIME`` (LC_ALL, then LC_TIME, then LANG).

    Returns the adopted name, or None for C/POSIX or when the named locale is
    not installed (a warning is logged and the process behaves as C).
    """
    try:
        adopted = locale.setlocale(locale.LC_TIME, "")
    except locale.Error as exc:
        # Name the variable that supplied the bad value, as the C library
        # would have chosen it.
        source, value = next(
            ((var, os.environ[var]) for var in ("LC_ALL", "LC_TIME", "LANG")
             if os.environ.get(var)),
            ("LANG", ""),
        )
        logger.warning(
            "System locale %r (from %s) is not installed (%s); dates keep the "
            "C locale's English names and web search uses no region — install "
            "it (locale-gen %s; in the container image, rebuild with "
            "--build-arg LOCALE=%s) or set LANG to one listed by locale -a",
            value, source, type(exc).__name__, value, value,
        )
        return None
    return adopted if time_locale() is not None else None


def time_locale() -> str | None:
    """The current LC_TIME name without encoding or modifier; None for C/POSIX.

    Queries only, never sets.
    """
    name = locale.setlocale(locale.LC_TIME)
    # 'de_DE.UTF-8@euro' -> 'de_DE'.
    name = name.split("@", 1)[0].split(".", 1)[0]
    return None if name in ("", "C", "POSIX") else name


def datetime_format(c_format: str) -> str:
    """The localized format under a real locale, else the site's own ``c_format``."""
    if time_locale() is None:
        return c_format
    return LOCALIZED_DATETIME_FORMAT


def web_region() -> str:
    """ddgs region ``<country>-<language>`` from the locale; ``NO_REGION`` if none."""
    name = time_locale()
    if name is None or "_" not in name:
        return NO_REGION
    language, country = name.split("_", 1)
    return f"{country}-{language}".lower()
