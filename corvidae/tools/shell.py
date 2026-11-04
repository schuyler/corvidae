"""Shell execution tool."""

import asyncio
import logging

logger = logging.getLogger(__name__)

# Error string returned when the subprocess exceeds the configured timeout.
# Format with timeout=<seconds>. Used by CoreToolsPlugin.shell.
TIMEOUT_ERROR_TEMPLATE = "Error: command timed out after {timeout} seconds"


async def shell(command: str, timeout: int = 30) -> str:
    """Execute a shell command and return the output."""
    output, _ = await run_shell(command, timeout=timeout)
    return output


async def run_shell(command: str, timeout: int = 30) -> tuple[str, int | None]:
    """Execute a shell command; return its formatted output and exit code.

    The exit code is ``None`` when the command produced no exit status of its
    own -- a timeout, which returns the timeout message. Callers that need to
    know whether the command succeeded use this rather than parsing
    ``"Exit code: N"`` out of the text (#105).
    """
    # Set once spawned, so the cancel handler knows whether there is a process.
    proc = None
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except OSError:
            logger.debug("proc.kill() failed (process may have already exited)", exc_info=True)
        return TIMEOUT_ERROR_TEMPLATE.format(timeout=timeout), None
    except asyncio.CancelledError:
        # Cancelled from outside (call timeout, shutdown): don't leave the
        # command running. Kill and reap it, then propagate the cancel.
        if proc is not None and proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                logger.debug("proc.kill() after cancel: process already exited", exc_info=True)
            await proc.wait()
        raise

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
        return "(no output)", proc.returncode

    return "\n".join(parts), proc.returncode


# Expose the module-level constant as a function attribute so that
# `import corvidae.tools.shell as m; m.TIMEOUT_ERROR_TEMPLATE` works
# even when the package re-exports the function under the submodule name.
shell.TIMEOUT_ERROR_TEMPLATE = TIMEOUT_ERROR_TEMPLATE  # type: ignore[attr-defined]
