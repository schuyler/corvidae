# CLIPlugin Design

## Overview

`CLIPlugin` is a transport plugin that bridges stdin/stdout to the sherman agent
loop. It reads lines from stdin as inbound messages and prints agent responses to
stdout. Conditionally active when `cli` transport channels are configured in
`agent.yaml`. Registered in `main.py` before `AgentLoopPlugin`.

## Key Decisions

### Registry access
Access via `self.pm.registry` — matches the `AgentLoopPlugin` pattern (Option B).
The `create_plugin(pm, config)` factory in `channels.md` that uses
`config["_registry"]` predates this decision and should not be followed.

### Stdin reading approach
Use `loop.run_in_executor(None, sys.stdin.readline)` rather than `connect_read_pipe`.
`connect_read_pipe` fails on a real tty (`ValueError: File descriptor 0 is used by
transport`). `run_in_executor` works for both tty and pipe, handles EOF cleanly
(returns `""`), and doesn't require fd manipulation.

Tradeoff: `run_in_executor` cannot be cancelled mid-blocking-call. Cancellation is
deferred until the next `readline` returns. For CLI this is acceptable — the user
pressing Ctrl-D or a signal will unblock it.

### on_stop cancellation
`on_stop` calls `self._task.cancel()` then `await self._task` inside a split-except:
`asyncio.CancelledError` is suppressed silently; `Exception` is logged via
`logger.exception(...)`. Do not fire-and-forget — the task must be awaited so it
reaches the cancelled state before the event loop shuts down (avoids "unhandled
exception in task" warnings).

`_read_loop` does not catch `CancelledError` — it lets it propagate naturally so
the task enters the cancelled state. The `except asyncio.CancelledError: raise`
block inside the executor await is what propagates cancellation out of the while
loop; without it, `CancelledError` would be silently swallowed by `run_in_executor`
and the loop would continue instead of terminating.

### Conditional activation
Check `self.pm.registry.by_transport("cli")` in `on_start`. If empty, skip creating
the task. Always register the plugin unconditionally in `main.py` — keeps `main.py`
clean and the plugin self-contained.

`by_transport("cli")` is used only to check whether CLI is configured at all. The
read loop always connects to `cli:local` via `get_or_create("cli", "local")`. This
is intentional: CLI is single-user/single-scope by design, so `local` is the
canonical scope for the CLI transport. No multi-scope CLI support is planned.

### Registration order
Register `CLIPlugin` before `AgentLoopPlugin`. Transport plugins conventionally go
before the loop plugin. apluggy calls hooks in LIFO order, but correctness doesn't
depend on it here since each plugin guards with `channel.matches_transport("cli")`.

### Sender value
`"user"` — matches the `role: user` semantics that `AgentLoopPlugin` writes to the
conversation log. Using `"cli"` or `"local"` would conflate transport/scope identity
with person identity.

## Class Structure

```python
# sherman/cli_plugin.py

import asyncio
import logging
import sys

from sherman.hooks import hookimpl

logger = logging.getLogger(__name__)


class CLIPlugin:
    """Transport plugin for stdin/stdout interaction."""

    def __init__(self, pm) -> None:
        self.pm = pm
        self._task: asyncio.Task | None = None

    @hookimpl
    async def on_start(self, config: dict) -> None:
        """Start the stdin read loop if any cli channels are configured."""
        if not self.pm.registry.by_transport("cli"):
            logger.debug("CLIPlugin: no cli channels configured, skipping read loop")
            return
        self._task = asyncio.create_task(self._read_loop(), name="cli-read-loop")

    async def _read_loop(self) -> None:
        """Read lines from stdin and dispatch as on_message events."""
        channel = self.pm.registry.get_or_create("cli", "local")
        print("Agent ready. Type a message, or Ctrl-D to quit.\n")

        loop = asyncio.get_running_loop()
        while True:
            sys.stdout.write("> ")
            sys.stdout.flush()
            try:
                line = await loop.run_in_executor(None, sys.stdin.readline)
            except asyncio.CancelledError:
                raise
            if not line:
                # EOF (Ctrl-D or closed pipe)
                break
            text = line.strip()
            if not text:
                continue
            await self.pm.ahook.on_message(
                channel=channel,
                sender="user",
                text=text,
            )

    @hookimpl
    async def send_message(self, channel, text: str) -> None:
        """Print agent response to stdout if this is a cli channel."""
        if not channel.matches_transport("cli"):
            return
        print(f"\n{text}\n")

    @hookimpl
    async def on_stop(self) -> None:
        """Cancel the read loop and wait for it to finish."""
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("CLIPlugin read loop raised unexpected exception")
        self._task = None
```

## Registration in main.py

```python
from sherman.agent_loop_plugin import AgentLoopPlugin
from sherman.channel import ChannelRegistry, load_channel_config
from sherman.cli_plugin import CLIPlugin          # add this import
from sherman.plugin_manager import create_plugin_manager

    # Pre-register channels from YAML config (must happen before on_start)
    load_channel_config(config, registry)

    # Register CLIPlugin before AgentLoopPlugin (transport plugins first)
    cli_plugin = CLIPlugin(pm)
    pm.register(cli_plugin, name="cli")

    # Register AgentLoopPlugin after any tool-providing plugins (none yet in Phase 2)
    agent_loop = AgentLoopPlugin(pm)
    pm.register(agent_loop, name="agent_loop")

    await pm.ahook.on_start(config=config)
```

## Test Strategy

File: `tests/test_cli_plugin.py`

1. `test_on_start_no_cli_channels_skips_task` — no cli channels → `_task` is None
2. `test_on_start_with_cli_channel_creates_task` — cli:local registered → task created
3. `test_read_loop_dispatches_on_message` — mock readline returning a line then `""`;
   verify `pm.ahook.on_message` called with correct channel/sender/text
4. `test_read_loop_skips_blank_lines` — readline returns `"\n"` then `""`;
   verify `on_message` not called
5. `test_send_message_cli_channel_prints` — capture stdout; verify output
6. `test_send_message_non_cli_channel_ignored` — irc channel; verify no stdout output
7. `test_on_stop_cancels_task` — after on_start, on_stop cancels task and sets
   `self._task = None`
8. `test_on_stop_no_task_is_noop` — on_stop with no task doesn't raise

Mocking pattern: patch `sys.stdin` with a mock whose `readline` side_effect returns
a sequence of strings. Patch `loop.run_in_executor` to call the function synchronously.
The mock for `run_in_executor` must return an awaitable — not a raw value — so that
`await loop.run_in_executor(...)` works in the async context. Use a coroutine shim:

```python
async def fake_executor(executor, fn, *args):
    return fn(*args)
monkeypatch.setattr(loop, "run_in_executor", fake_executor)
```

This calls the blocking function directly (on the test thread) and wraps the result
in a coroutine that the event loop can await normally.

### Concurrent on_message (deferred)

The user could type a second message before the first agent response completes,
causing `on_message` to be called while a prior call is still in flight. This creates
a potential race in the conversation log. This is acceptable for Phase 2 — the CLI is
a single-user, low-concurrency interface and the window for the race is narrow. It
will be addressed when needed in a future phase.

## Note on run_in_executor Cancellation

When the task is cancelled while blocked in `run_in_executor`, the thread running
`readline` continues until it returns. `on_stop` may block briefly — on a tty, until
the user hits Enter or sends EOF. This is acceptable for a CLI tool. If unattended
shutdown is required in the future, a threading `Event` could signal the thread to
unblock, but that is out of scope.
