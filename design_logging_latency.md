# Design: DEBUG-level logging for agent turn lifecycle and background task system

## Overview

Add DEBUG-level log statements across three modules — `agent_loop.py`,
`background.py`, and `agent_loop_plugin.py` — to provide fine-grained
observability into the agent turn lifecycle and background task system.
The goal is to complement the existing INFO-level logs with content-level
detail that is only needed when debugging.

## Existing state

### `sherman/agent_loop.py`
- Has `logger = logging.getLogger(__name__)` at line 27.
- Has `_truncate(s, maxlen=200)` helper at line 30.
- Logs at INFO: `"LLM response received"`, `"tool call dispatched"`, `"tool call result"`.
- Logs at WARNING: max turns, unknown tool, tool exception.
- The module docstring notes `_truncate` is "available for future use if result
  content logging is added" — this is exactly the use case for the new DEBUG logs.

### `sherman/background.py`
- Has **no** logger. `import logging` is absent.
- Has `BackgroundTask` dataclass and `TaskQueue` class with `enqueue()` and
  `run_worker()` methods.

### `sherman/agent_loop_plugin.py`
- Has `logger = logging.getLogger(__name__)` at line 50.
- Logs at INFO: `on_start complete`, `on_message received`, `agent response sent`,
  `conversation initialized`.
- Logs at ERROR: LLM client not initialized, agent loop failures.
- `on_notify()` has **no** log statement.
- `_process_queue_item()` has **no** entry log statement.
- `background_task` closure (inside `_process_queue_item`) has **no** log.
- `_on_task_complete()` has **no** log statement.
- `_execute_background_task()` has **no** log statement.

---

## Changes per file

### 1. `sherman/agent_loop.py`

No new imports needed. `logger` and `_truncate` already exist.

#### 1a. After LLM response (~line 80, after `messages.append(msg)`)

Insert a DEBUG log immediately after the existing INFO log at line 83-90:

```python
# Existing INFO log (lines 83-90)
logger.info(
    "LLM response received",
    extra={
        "role": msg.get("role"),
        "tool_calls_count": len(tool_calls) if tool_calls else 0,
        "latency_ms": latency_ms,
    },
)

# NEW DEBUG log — insert here
reasoning = msg.get("reasoning_content")
logger.debug(
    "LLM response content",
    extra={
        "content": _truncate(msg.get("content") or ""),
        "has_reasoning_content": reasoning is not None,
        "reasoning_content_length": len(reasoning) if reasoning is not None else None,
    },
)
```

**Rationale for `reasoning` handling:** `reasoning_content` may be absent
(key not present), present but empty string, or present with content. The
plan calls for `has_reasoning_content` (bool) and `reasoning_content_length`
(int or None). Use `msg.get("reasoning_content")` — if the key is absent,
`reasoning` is `None`, so `has_reasoning_content=False` and
`reasoning_content_length=None`. If it is present (even as `""`),
`has_reasoning_content=True` and length is 0. This matches the intent of
the plan.

Note: `tool_calls` is assigned at line 82 (`tool_calls = msg.get("tool_calls")`),
which is after `messages.append(msg)` at line 80 but before the existing INFO
log. The DEBUG log is inserted after the INFO log. The `tool_calls` variable
is already in scope.

#### 1b. Tool call dispatch (~line 100, after the existing INFO `"tool call dispatched"`)

Insert a DEBUG log immediately after the existing INFO log at line 100-103:

```python
# Existing INFO log
logger.info(
    "tool call dispatched",
    extra={"tool": fn_name, "arg_keys": list(args.keys())},
)

# NEW DEBUG log — insert here
logger.debug(
    "tool call arguments",
    extra={
        "tool": fn_name,
        "arguments": _truncate(call["function"]["arguments"]),
    },
)
```

`call["function"]["arguments"]` is the raw JSON string before `json.loads`
at line 98. This avoids re-serializing and is already a string suitable for
`_truncate`. The arguments are logged as a truncated JSON string so the log
is both human-readable and bounded in size.

#### 1c. Tool call result (~line 121, after the existing INFO `"tool call result"`)

The existing INFO log is inside the `try` block, inside the `else` (known
tool, success) path, immediately after computing `tool_latency_ms`. Insert
a DEBUG log immediately after it:

```python
# Existing INFO log
logger.info(
    "tool call result",
    extra={"tool": fn_name, "result_length": len(str(content)), "latency_ms": tool_latency_ms},
)

# NEW DEBUG log — insert here
logger.debug(
    "tool call result content",
    extra={
        "tool": fn_name,
        "content": _truncate(str(content)),
    },
)
```

This is inside the `try` block after a successful tool call. The DEBUG log
is not emitted on exception (consistent with the existing INFO log not being
emitted on exception).

---

### 2. `sherman/background.py`

Add `import logging` and a module-level logger. Then add log statements in
`enqueue()` and `run_worker()`.

#### 2a. New imports and logger

After the existing imports block (currently ends with `from sherman.channel import Channel`),
add:

```python
import logging
```

and immediately after the imports block:

```python
logger = logging.getLogger(__name__)
```

The full import block after the change:

```python
import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from time import time

from sherman.channel import Channel

logger = logging.getLogger(__name__)
```

#### 2b. `TaskQueue.enqueue()`

Current code:

```python
async def enqueue(self, task: BackgroundTask) -> None:
    """Add a task to the queue."""
    await self.queue.put(task)
```

New code:

```python
async def enqueue(self, task: BackgroundTask) -> None:
    """Add a task to the queue."""
    logger.debug(
        "task enqueued",
        extra={
            "task_id": task.task_id,
            "channel": task.channel.id,
            "description": task.description,
        },
    )
    await self.queue.put(task)
```

Log before `queue.put` so the log fires whether or not the put succeeds.
(The queue is unbounded so this is not a real concern, but logging intent
before the action is the cleaner convention.)

#### 2c. `run_worker()` — task started

After `self.active_task = task` and the default result assignment, before
the `try:` block:

```python
task = await self.queue.get()
self.active_task = task
result = f"Task {task.task_id} failed: (unknown error)"
# NEW log — insert here
logger.debug(
    "task started",
    extra={"task_id": task.task_id, "description": task.description},
)
try:
    result = await execute_fn(task)
```

#### 2d. `run_worker()` — task completed

Inside the `try` block, immediately after `result = await execute_fn(task)`:

```python
try:
    result = await execute_fn(task)
    # NEW log — insert here
    logger.debug(
        "task completed",
        extra={"task_id": task.task_id, "result_length": len(result)},
    )
```

Only fires on success; not emitted on exception.

#### 2e. `run_worker()` — task failed (WARNING)

In the `except Exception as exc:` block, before updating `result`:

```python
except asyncio.CancelledError:
    self.active_task = None
    raise
except Exception as exc:
    # NEW log — insert here
    logger.warning(
        "task failed",
        extra={"task_id": task.task_id},
        exc_info=True,
    )
    result = f"Task {task.task_id} failed: {exc}"
```

`exc_info=True` attaches the traceback to the log record.

**Note:** `asyncio.CancelledError` inherits from `BaseException`, not
`Exception` (Python 3.8+), so the narrowed `except Exception` already
excludes it. `CancelledError` is caught first and re-raised cleanly — do not
log it as a failure.

---

### 3. `sherman/agent_loop_plugin.py`

No new logger needed. `logger` already exists at line 50.

One new import: `_truncate` from `sherman.agent_loop`. The existing import
line (line 42) is:

```python
from sherman.agent_loop import run_agent_loop, strip_reasoning_content, strip_thinking, tool_to_schema
```

Change to (alphabetically inserting `_truncate`):

```python
from sherman.agent_loop import _truncate, run_agent_loop, strip_reasoning_content, strip_thinking, tool_to_schema
```

#### 3a. `on_notify()` — entry log

Add a DEBUG log at the top of the method body, before building the `QueueItem`:

```python
async def on_notify(
    self,
    channel,
    source: str,
    text: str,
    tool_call_id: str | None,
    meta: dict | None,
) -> None:
    """Enqueue a notification item on the channel's queue."""
    logger.debug(
        "on_notify received",
        extra={
            "channel": channel.id,
            "source": source,
            "tool_call_id": tool_call_id,
            "content_length": len(text),
        },
    )
    item = QueueItem(
        ...
    )
```

#### 3b. `_process_queue_item()` — entry log

Add a DEBUG log at the top of `_process_queue_item()`, after
`channel = item.channel`:

```python
async def _process_queue_item(self, item: QueueItem) -> None:
    channel = item.channel
    # NEW log — insert here
    logger.debug(
        "processing queue item",
        extra={
            "channel": channel.id,
            "role": item.role,
            "source": item.source,
            "has_tool_call_id": item.tool_call_id is not None,
        },
    )
```

`item.source` is `None` for user messages. Log it as-is; `None` is a valid
log field value. Do not coerce to `""`.

#### 3c. `background_task` closure — enqueue log

The per-call `background_task` closure is defined inside
`_process_queue_item()`. Add a DEBUG log after
`await self.task_queue.enqueue(task)`:

```python
async def background_task(
    description: str, instructions: str, _tool_call_id: str | None = None
) -> str:
    """Launch a long-running task in the background."""
    if not self.task_queue:
        return "Error: background task system not initialized"
    task = BackgroundTask(
        channel=channel,
        description=description,
        instructions=instructions,
        tool_call_id=_tool_call_id,
    )
    await self.task_queue.enqueue(task)
    # NEW log — insert here
    logger.debug(
        "background_task enqueued",
        extra={
            "task_id": task.task_id,
            "channel": channel.id,
            "description": _truncate(description),
        },
    )
    return f"Task {task.task_id} enqueued: {description}"
```

#### 3d. `_on_task_complete()` — dispatch log

Add a DEBUG log at the top of `_on_task_complete()`, before
`display_result = strip_thinking(result)`:

```python
async def _on_task_complete(self, task: BackgroundTask, result: str) -> None:
    # NEW log — insert here
    logger.debug(
        "task complete, dispatching notification",
        extra={
            "task_id": task.task_id,
            "channel": task.channel.id,
            "result_length": len(result),
        },
    )
    display_result = strip_thinking(result)
    ...
```

#### 3e. `_execute_background_task()` — entry log

Add a DEBUG log at the top of `_execute_background_task()`, before
building `bg_tools`:

```python
async def _execute_background_task(self, task: BackgroundTask) -> str:
    """Run a background task with its own conversation context."""
    # NEW log — insert here
    logger.debug(
        "executing background task",
        extra={
            "task_id": task.task_id,
            "description": task.description,
            "instructions": _truncate(task.instructions),
        },
    )
    # Exclude background_task to prevent unbounded recursive task creation.
    bg_tools = {k: v for k, v in self.tools.items() if k != "background_task"}
    ...
```

---

## Test strategy

### `tests/test_agent_loop.py` — six new tests

Add at the bottom of the file, after the existing logging tests (around line
452). All DEBUG coverage for `agent_loop.py` lives here, consistent with the
existing INFO/WARNING logging tests already in this file (lines 343–452). No
`TestAgentLoopDebugLogging` class is added to `test_logging.py`.

#### `test_llm_response_content_debug_log`
Mirrors `test_llm_response_logs_info_with_latency_ms`. Use
`caplog.at_level(logging.DEBUG, logger="sherman.agent_loop")`. Assert a DEBUG
record with message `"LLM response content"`, `has_reasoning_content` attribute,
and `reasoning_content_length` attribute.

#### `test_llm_response_content_debug_log_with_reasoning`
- Setup: response message has both `content` and `reasoning_content="<reasoning>"`.
- Assert: DEBUG record with `has_reasoning_content=True` and
  `reasoning_content_length > 0`.

#### `test_llm_response_content_truncated`
- Setup: response with `content` longer than 200 chars.
- Assert: the `content` attribute on the log record ends with `"..."` and is
  at most 203 characters (200 + `"..."`).

#### `test_tool_call_arguments_debug_log`
Mirrors `test_tool_call_dispatched_logs_info`. Assert DEBUG record with
message `"tool call arguments"` and `arguments` attribute (a string).

#### `test_tool_call_result_content_debug_log`
Mirrors `test_tool_call_result_logs_info_with_latency_ms`. Assert DEBUG
record with message `"tool call result content"` and `content` attribute.

#### `test_tool_call_result_content_not_logged_on_exception`
- Setup: tool raises `ValueError`.
- Assert: no DEBUG record with message `"tool call result content"` is emitted.
  (Mirrors the existing INFO test `test_tool_call_result_not_logged_on_exception`.)

### `tests/test_logging.py` — `TestLoggerNamingConvention` addition

Add one new test method to the existing `TestLoggerNamingConvention` class,
following the same pattern as the other module logger naming tests:

#### `test_background_has_module_logger`
- Import `sherman.background` and assert `mod.logger.name == "sherman.background"`.
- Consistent with the existing tests for `agent_loop`, `llm`, `conversation`,
  `channel`, `prompt`, `main`, `agent_loop_plugin`, and `plugin_manager`.

### `tests/test_background.py` — new class `TestTaskQueueLogging`

Add a new class `TestTaskQueueLogging` to `tests/test_background.py`.

#### `test_enqueue_logs_debug`
- Create a `TaskQueue` and a `BackgroundTask`.
- Call `await queue.enqueue(task)` with
  `caplog.at_level(logging.DEBUG, logger="sherman.background")`.
- Assert a DEBUG record with message `"task enqueued"` is emitted.
- Assert record has `task_id` attribute equal to `task.task_id`.
- Assert record has `channel` attribute equal to `task.channel.id`.
- Assert record has `description` attribute equal to `task.description`.

#### `test_run_worker_logs_task_started`
- Run a worker to completion for one task (follow the existing
  `test_enqueue_and_dequeue` pattern: create task, start worker, enqueue,
  wait for done event, cancel worker).
- Capture at `logging.DEBUG, logger="sherman.background"`.
- Assert a DEBUG record with message `"task started"` is emitted.
- Assert record has `task_id` and `description` attributes.

#### `test_run_worker_logs_task_completed`
- Same setup as `test_run_worker_logs_task_started`.
- Assert a DEBUG record with message `"task completed"` is emitted.
- Assert record has `task_id` and `result_length` attributes.
- Assert `result_length` is an int.

#### `test_run_worker_logs_task_failed_warning`
- Run a worker where `execute_fn` raises `RuntimeError("boom")`.
- Follow the `test_execute_fn_error` pattern.
- Assert a WARNING record (not DEBUG) with message `"task failed"` is emitted.
- Assert record has `task_id` attribute.
- Assert `record.exc_info` is not `None` and not `(None, None, None)`.

---

## Edge cases and considerations

### Truncation
`_truncate` defaults to 200 chars. All content fields use the default.
Arguments are truncated from the raw JSON string, which may cut in the middle
of a key or value — acceptable for debug logs where the goal is a bounded-size
hint, not a faithful representation.

### `reasoning_content` field
The `reasoning_content` field is specific to models that emit extended
thinking. Most responses will not have it. The implementation must use
`msg.get("reasoning_content")` (returns `None` if absent) rather than
`msg["reasoning_content"]` (raises `KeyError`).

### `item.source` in `_process_queue_item`
`QueueItem.source` is `None` for user messages. Log it as-is; `None` is a
valid structured log field value.

### Log placement ordering in `background.py`
- `"task enqueued"` — before `queue.put()` (log intent before action).
- `"task started"` — after `queue.get()` and `active_task` assignment, before
  `try:`, so it always fires before execution begins.
- `"task completed"` — inside `try:`, immediately after
  `result = await execute_fn(task)`, so it only fires on success.
- `"task failed"` — in `except Exception:`, before updating `result`.

### `asyncio.CancelledError` is not a failure
In Python 3.8+, `CancelledError` inherits from `BaseException`, not
`Exception`. The existing `except asyncio.CancelledError` catches it first
and re-raises. The `except Exception as exc:` block (where the WARNING log
goes) never sees `CancelledError`.

### `_truncate` import in `agent_loop_plugin.py`
`_truncate` is currently not imported in `agent_loop_plugin.py`. The single
existing import line from `sherman.agent_loop` must be updated to include it.

### Docstring updates
The module docstring in `agent_loop.py` currently says:
> "Tool call result content is not logged; only result length is recorded.
> The `_truncate` helper is available for future use if result content
> logging is added."

After this change, that comment is no longer accurate. The `Logging:` section
should be updated to mention the new DEBUG-level content logs. Similarly, the
`Logging:` section in `agent_loop_plugin.py`'s module docstring should be
updated to mention DEBUG-level logs for `on_notify`, `_process_queue_item`,
`background_task` closure, `_on_task_complete`, and `_execute_background_task`.

---

## Summary of all new log statements

| Module | Method | Level | Message | Extra fields |
|--------|--------|-------|---------|--------------|
| `agent_loop.py` | `run_agent_loop` | DEBUG | `"LLM response content"` | `content` (truncated), `has_reasoning_content`, `reasoning_content_length` |
| `agent_loop.py` | `run_agent_loop` | DEBUG | `"tool call arguments"` | `tool`, `arguments` (truncated JSON string) |
| `agent_loop.py` | `run_agent_loop` | DEBUG | `"tool call result content"` | `tool`, `content` (truncated) |
| `background.py` | `TaskQueue.enqueue` | DEBUG | `"task enqueued"` | `task_id`, `channel` (channel.id), `description` |
| `background.py` | `TaskQueue.run_worker` | DEBUG | `"task started"` | `task_id`, `description` |
| `background.py` | `TaskQueue.run_worker` | DEBUG | `"task completed"` | `task_id`, `result_length` |
| `background.py` | `TaskQueue.run_worker` | WARNING | `"task failed"` | `task_id`, `exc_info=True` |
| `agent_loop_plugin.py` | `on_notify` | DEBUG | `"on_notify received"` | `channel`, `source`, `tool_call_id`, `content_length` |
| `agent_loop_plugin.py` | `_process_queue_item` | DEBUG | `"processing queue item"` | `channel`, `role`, `source`, `has_tool_call_id` |
| `agent_loop_plugin.py` | `background_task` (closure) | DEBUG | `"background_task enqueued"` | `task_id`, `channel`, `description` (truncated) |
| `agent_loop_plugin.py` | `_on_task_complete` | DEBUG | `"task complete, dispatching notification"` | `task_id`, `channel`, `result_length` |
| `agent_loop_plugin.py` | `_execute_background_task` | DEBUG | `"executing background task"` | `task_id`, `description`, `instructions` (truncated) |

---

## Design Report

**Proceed: YES**

The plan is well-specified and the source code matches the plan's assumptions:
`_truncate` exists and is documented for exactly this use, `logger` already
exists in `agent_loop.py` and `agent_loop_plugin.py`, and `background.py` has
no logger (as the plan states). All changes are additive with no structural
impact on existing behavior. The one non-obvious mechanical point is the
`_truncate` import addition in `agent_loop_plugin.py`, which is straightforward.
The test strategy maps directly from the plan to specific test names and
assertions, following patterns already established in the test suite.

---

## Design Fix #1

**Date:** 2026-04-23
**Reviewer finding:** Two important issues found in design review.

### I-1: Removed `TestAgentLoopDebugLogging` from `test_logging.py`

The original design proposed adding a `TestAgentLoopDebugLogging` class (6
tests) to `tests/test_logging.py` AND three overlapping tests to
`tests/test_agent_loop.py`, creating duplicated coverage for the same code
path.

The existing INFO logging tests for `agent_loop.py` already live in
`tests/test_agent_loop.py` (lines 343–452), not in `test_logging.py`. Placing
DEBUG tests in a different file from the INFO tests covering the same module
would split related coverage across two files for no benefit.

**Resolution:** All six DEBUG logging tests for `agent_loop.py` go in
`tests/test_agent_loop.py` only. The `TestAgentLoopDebugLogging` class has
been removed from the `test_logging.py` section of the design.

### I-2: Moved `background` logger naming test to `TestLoggerNamingConvention`

The original design placed `test_background_logger_name` inside the new
`TestTaskQueueLogging` class in `tests/test_background.py`. `TestLoggerNamingConvention`
in `tests/test_logging.py` already covers every other module's logger by name
(agent_loop, llm, conversation, channel, prompt, main, agent_loop_plugin,
plugin_manager). Putting `background`'s naming test elsewhere breaks that
pattern and leaves `TestLoggerNamingConvention` incomplete.

**Resolution:** `test_background_has_module_logger` is added to
`TestLoggerNamingConvention` in `tests/test_logging.py`, consistent with the
established pattern for all other modules.

---

## Implementation Report

**Date:** 2026-04-23
**Status:** Complete

### Summary
Added DEBUG-level logging to three source files: `agent_loop.py`,
`background.py`, and `agent_loop_plugin.py`. All 12 specified log points
are implemented. 11 new tests were added across `test_agent_loop.py`,
`test_logging.py`, and `test_background.py`.

### Test results
266 tests passing (baseline: 255; added: 11). No regressions.

### Deviations from original plan
- `agent_loop.py` logs `"reasoning_content" in msg` for `has_reasoning_content`
  rather than checking the value via `.get()`. These differ when `reasoning_content`
  is present but `None`: `in msg` yields `True` while `msg.get()` would yield `False`.
  In practice, `reasoning_content` is always a non-empty string or absent from the
  response, so behavior is the same for all real-world cases.
- `agent_loop.py` logs `json.dumps(args)` for `arguments` (re-serialized)
  rather than the raw JSON string from the LLM. Content is semantically identical.

### Enabling DEBUG logging
Set the log level for `sherman.agent_loop`, `sherman.background`, and/or
`sherman.agent_loop_plugin` to DEBUG in the logging config (YAML or dict).
Example in `config.yaml`:
```yaml
logging:
  loggers:
    sherman.agent_loop:
      level: DEBUG
    sherman.background:
      level: DEBUG
    sherman.agent_loop_plugin:
      level: DEBUG
```
