"""File read/write tools."""

import asyncio
import os
import pathlib

from corvidae.tool import MAX_TOOL_RESULT_CHARS

# Characters read per call when skipping to an offset or counting the rest.
_CHUNK = 64 * 1024


async def read_file(
    path: str,
    max_size: int = 1024 * 1024,
    offset: int | None = None,
    length: int | None = None,
    max_chars: int = MAX_TOOL_RESULT_CHARS,
) -> str:
    """Read a file, or a character range of it when offset or length is given."""
    return await asyncio.to_thread(_read_file_sync, path, max_size, offset, length, max_chars)


def _read_file_sync(
    path: str,
    max_size: int = 1024 * 1024,
    offset: int | None = None,
    length: int | None = None,
    max_chars: int = MAX_TOOL_RESULT_CHARS,
) -> str:
    # Either parameter makes the read ranged; reject out-of-range values first.
    ranged = offset is not None or length is not None
    if offset is not None and offset < 0:
        return f"Error: offset is {offset} for {path}; use 0 or greater"
    if length is not None and length < 1:
        return (
            f"Error: length is {length} for {path}; "
            "use a positive integer, or omit it to read to the end"
        )

    # The path must name an existing regular file.
    p = pathlib.Path(path)
    if not p.exists():
        return f"Error: file not found: {path}"
    if os.path.isdir(path):
        return f"Error: not a file (is a directory): {path}"
    if ranged:
        return _read_range(p, path, offset or 0, length, max_chars)

    # An unranged read loads the whole file, so it is bounded by max_size.
    try:
        size = p.stat().st_size
    except OSError as exc:
        return f"Error: could not stat file: {exc}"
    if size > max_size:
        max_mb = max_size / (1024 * 1024)
        return (
            f"Error: file too large (>{max_mb:.0f}MB): {path}; "
            "pass offset and length to read it in ranges"
        )
    try:
        return p.read_text(errors="replace")
    except OSError as exc:
        return f"Error: could not read file: {exc}"


def _read_range(
    p: pathlib.Path, path: str, offset: int, length: int | None, max_chars: int
) -> str:
    """Return a header naming the range and total, then that range's characters.

    The header plus content never exceeds max_chars, so the result cap never
    cuts a range the header has already reported.
    """
    # One pass: skip to the offset, keep at most max_chars, count the rest.
    try:
        with p.open(errors="replace") as f:
            skipped = 0
            while skipped < offset:
                chunk = f.read(min(_CHUNK, offset - skipped))
                if not chunk:
                    break
                skipped += len(chunk)
            kept = f.read(max_chars if length is None else min(length, max_chars))
            rest = 0
            while chunk := f.read(_CHUNK):
                rest += len(chunk)
    except OSError as exc:
        return f"Error: could not read file: {exc}"

    # Reject a range that starts at or past the end.
    total = skipped + len(kept) + rest
    if total == 0:
        return f"Error: {path} is empty; there is nothing to read"
    if offset >= total:
        return (
            f"Error: offset {offset} is at or past the end of {path}, "
            f"which has {total} characters; use an offset below {total}"
        )

    # Size the content from the longest header this range could produce,
    # leaving room for the newline after it.
    header_bound = len(f"[characters {offset}-{total} of {total} in {path}; next offset {total}]")
    content = kept[: max(0, max_chars - header_bound - 1)]
    end = offset + len(content)
    tail = "end of file" if end == total else f"next offset {end}"
    return f"[characters {offset}-{end} of {total} in {path}; {tail}]\n{content}"


async def write_file(path: str, content: str) -> str:
    """Write content to a file, creating parent directories if needed."""
    return await asyncio.to_thread(_write_file_sync, path, content)


def _write_file_sync(path: str, content: str) -> str:
    p = pathlib.Path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Wrote {len(content)} chars to {path}"
    except PermissionError as exc:
        return f"Error: permission denied: {exc}"
    except OSError as exc:
        return f"Error: could not write file: {exc}"
