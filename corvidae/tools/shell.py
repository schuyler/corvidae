"""Shell execution tool."""

import asyncio
import logging

logger = logging.getLogger(__name__)


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
        except OSError:
            logger.debug("proc.kill() failed (process may have already exited)", exc_info=True)
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
