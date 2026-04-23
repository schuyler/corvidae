"""Core tool functions for the sherman agent daemon."""

import asyncio
import os
import pathlib

import aiohttp

from sherman.hooks import hookimpl


async def shell(command: str) -> str:
    """Execute a shell command and return the output."""
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=30
        )
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        return "Error: command timed out after 30 seconds"

    stdout = stdout_bytes.decode(errors="replace").strip()
    stderr = stderr_bytes.decode(errors="replace").strip()

    parts = []
    if stdout:
        parts.append(stdout)
    if stderr:
        parts.append(f"STDERR:\n{stderr}")
    if proc.returncode != 0:
        parts.append(f"Exit code: {proc.returncode}")

    if not parts:
        return "(no output)"

    return "\n".join(parts)


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


async def web_fetch(url: str) -> str:
    """Fetch a URL and return its text content."""
    timeout = aiohttp.ClientTimeout(total=15)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                if response.status != 200:
                    return f"HTTP {response.status}"
                text = await response.text()
                if len(text) > 50000:
                    return text[:50000] + "[truncated]"
                return text
    except asyncio.TimeoutError:
        return f"Error: request timed out after 15 seconds"
    except aiohttp.ClientError as exc:
        return f"Error: {exc}"


class CoreToolsPlugin:
    @hookimpl
    def register_tools(self, tool_registry: list) -> None:
        tool_registry.extend([shell, read_file, write_file, web_fetch])
