"""File read/write tools."""

import os
import pathlib


async def read_file(path: str) -> str:
    """Read the contents of a file."""
    p = pathlib.Path(path)
    if not p.exists():
        return f"Error: file not found: {path}"
    if os.path.isdir(path):
        return f"Error: not a file (is a directory): {path}"
    try:
        size = p.stat().st_size
    except Exception as exc:
        return f"Error: could not stat file: {exc}"
    if size > 1024 * 1024:
        return f"Error: file too large (>1MB): {path}"
    try:
        return p.read_text(errors="replace")
    except Exception as exc:
        return f"Error: could not read file: {exc}"


async def write_file(path: str, content: str) -> str:
    """Write content to a file, creating parent directories if needed."""
    p = pathlib.Path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Wrote {len(content)} chars to {path}"
    except PermissionError as exc:
        return f"Error: permission denied: {exc}"
    except Exception as exc:
        return f"Error: could not write file: {exc}"
