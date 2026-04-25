"""File read/write tools."""

import asyncio
import os
import pathlib


async def read_file(path: str) -> str:
    """Read the contents of a file."""
    return await asyncio.to_thread(_read_file_sync, path)


def _read_file_sync(path: str) -> str:
    p = pathlib.Path(path)
    if not p.exists():
        return f"Error: file not found: {path}"
    if os.path.isdir(path):
        return f"Error: not a file (is a directory): {path}"
    try:
        size = p.stat().st_size
    except OSError as exc:
        return f"Error: could not stat file: {exc}"
    if size > 1024 * 1024:
        return f"Error: file too large (>1MB): {path}"
    try:
        return p.read_text(errors="replace")
    except OSError as exc:
        return f"Error: could not read file: {exc}"


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
