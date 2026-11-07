"""Files received from a transport, written under a state directory (#521).

Signal and Discord both take bytes from the network and put them on disk. Three
rules are the same in both places, so they live here: the extension comes from
the remote filename only when it is a plain one, the write is capped, and the
resolved path has to stay inside the directory.

Nothing here raises for a refused file: the callers turn ``None`` into a note in
the message text.
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path

logger = logging.getLogger("corvidae.attachments")

# Longest suffix worth appending: a dot and 1-10 ASCII alphanumerics. Anything
# else (a second dot, a slash, a name with no extension) is dropped, so a remote
# filename can never contribute more than an extension to the path.
_SAFE_SUFFIX = re.compile(r"\.[A-Za-z0-9]{1,10}")

# Default cap for one file, in bytes. Callers pass their configured value.
DEFAULT_MAX_ATTACHMENT_SIZE = 10 * 1024 * 1024


def safe_suffix(name: str | None) -> str:
    """The lowercased extension of a remote filename, or ``""`` if unsafe."""
    if not name or "." not in name:
        return ""
    candidate = name[name.rindex(".") :].lower()
    return candidate if _SAFE_SUFFIX.fullmatch(candidate) else ""


async def save_attachment(
    data: bytes,
    directory: Path,
    stem: str,
    *,
    filename: str | None = None,
    suffix: str = "",
    max_bytes: int = DEFAULT_MAX_ATTACHMENT_SIZE,
    transport: str = "attachment",
) -> Path | None:
    """Write ``data`` as ``directory/<stem><suffix>``, or return ``None``.

    ``suffix`` wins over ``filename`` when given: a transport that knows the
    content type should choose the extension itself rather than trust a remote
    name. ``transport`` only labels the log lines, so ``journalctl | grep
    signal`` keeps working.

    ``None`` means the payload was over ``max_bytes``, the path would have
    resolved outside ``directory``, or the write failed. All three are the
    caller's cue to say "download failed" and carry on.
    """
    if len(data) > max_bytes:
        logger.warning(
            "%s: attachment %s is %d bytes, over the %d byte cap",
            transport, stem, len(data), max_bytes,
        )
        return None

    effective = suffix or safe_suffix(filename)
    # A stem that already carries the extension (e.g. an id that ends in
    # .png) must not get it appended a second time.
    if effective and stem.endswith(effective):
        effective = ""
    path = directory / f"{stem}{effective}"
    try:
        await asyncio.to_thread(directory.mkdir, parents=True, exist_ok=True)
        # The stem is built from ids the transport chose, so a separator in it
        # is a bug or an attack; refuse rather than trust it.
        if not path.resolve().is_relative_to(directory.resolve()):
            logger.warning("%s: resolved path %s escapes %s, refusing", transport, path, directory)
            return None
        await asyncio.to_thread(path.write_bytes, data)
    except OSError:
        logger.exception("%s: cannot write attachment to %s", transport, path)
        return None

    resolved = path.resolve()
    logger.info("%s: wrote attachment to %s (%d bytes)", transport, resolved, len(data))
    # Absolute: the path goes into a message text the agent may read from any
    # working directory, so a relative one would be a trap.
    return resolved
