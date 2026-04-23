# Phase 1 Implementation Plan

## Package Structure

```
sherman/
├── __init__.py
├── hooks.py
├── plugin_manager.py
├── llm.py
├── agent_loop.py
├── conversation.py
├── main.py
tests/
├── __init__.py
├── conftest.py
├── test_hooks.py
├── test_llm.py
├── test_agent_loop.py
├── test_conversation.py
├── test_main.py
```

## Dependencies to add to `pyproject.toml`

Current dependencies: `aiohttp>=3.9`. Add:

```
apluggy>=1.0
pydantic>=2.0
aiosqlite>=0.20
pyyaml>=6.0
```

Dev dependencies to add: none beyond existing `pytest>=8, pytest-asyncio>=0.23`.

### Console script entry point

Add to `pyproject.toml`:

```toml
[project.scripts]
sherman = "sherman.main:cli"
```

### pytest configuration

Add to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
```

Async fixtures must use `@pytest_asyncio.fixture`, not `@pytest.fixture`.

## Deviation from design.md: directory layout

The design doc places modules at the top level (`hooks.py`, `agent_loop.py`, etc.). This plan uses `sherman/` as a Python package directory per the task instructions. All imports within the package use relative or `sherman.` prefixed imports. The design doc's `main.py` becomes `sherman/main.py` with a console script entry point.

## Deviation from design.md: Phase 1 scope boundary

The design doc's Phase 1 list (lines 1192-1199) says "Agent loop (`agent_loop.py`) -- tool dispatch, schema generation" and "Daemon entry point (`main.py`) -- starts, waits for signal, stops." However the `AgentLoopPlugin` class (which wires hooks, manages conversations, and registers tools) is explicitly listed under Phase 2. Phase 1 implements `run_agent_loop()`, `tool_to_schema()`, and `strip_thinking()` as standalone functions. The `AgentLoopPlugin` class is Phase 2.

Similarly, `main.py` in Phase 1 creates the plugin manager, loads config, calls `on_start`/`on_stop`, and waits for a signal. It does NOT load component plugins (hot-loading is Phase 5) or register the `AgentLoopPlugin` (Phase 2). It registers no plugins in Phase 1 -- it demonstrates the lifecycle skeleton.

## Deviation from design.md: thinking token stripping in agent loop

The design doc's `on_message` handler (line 533) appends `raw_response` from `run_agent_loop()` as the assistant message. But `run_agent_loop()` already appends assistant messages to the `messages` list (line 230). This means the agent loop plugin would double-append. The plan treats `run_agent_loop()` as mutating `messages` in place (as documented in its docstring), and the caller reads the final response from the return value without re-appending it. This is a Phase 2 concern but noted here to avoid confusion during agent_loop.py implementation.

## Deviation from design.md: project name

The design doc uses project name `"agent"` for the pluggy marker. This plan uses `"sherman"` to match the package name. Either works; consistency with the package name is cleaner.

---

## Module 1: `sherman/__init__.py`

**File:** `/Users/sderle/code/sherman/sherman/__init__.py`

Empty file. Marks the directory as a package.

---

## Module 2: `sherman/hooks.py`

**File:** `/Users/sderle/code/sherman/sherman/hooks.py`

### Public API

```python
import apluggy as pluggy

hookspec = pluggy.HookspecMarker("sherman")
hookimpl = pluggy.HookimplMarker("sherman")

class AgentSpec:
    """Hook specifications for the agent daemon."""

    @hookspec
    async def on_start(self, config: dict) -> None: ...

    @hookspec
    async def on_stop(self) -> None: ...

    @hookspec
    async def on_message(self, channel_id: str, sender: str, text: str) -> None: ...

    @hookspec
    async def send_message(self, channel_id: str, text: str) -> None: ...

    @hookspec
    def register_tools(self, tool_registry: list) -> None: ...

    @hookspec
    async def on_agent_response(
        self, channel_id: str, request_text: str, response_text: str
    ) -> None: ...

    @hookspec
    async def on_task_complete(
        self, channel_id: str, task_id: str, result: str
    ) -> None: ...
```

### Key details
- `hookspec` and `hookimpl` are module-level constants, imported by other modules.
- `register_tools` is sync (not async). The design doc is explicit about this.
- All other hooks are async.
- The marker name is `"sherman"`. design.md examples use `"agent"`. Update any code copied from design.md to use `"sherman"`.

### Connections
Used by `plugin_manager.py` (to add hookspecs) and by any plugin class (to decorate implementations with `hookimpl`).

---

## Module 3: `sherman/plugin_manager.py`

**File:** `/Users/sderle/code/sherman/sherman/plugin_manager.py`

### Public API

```python
import apluggy as pluggy
from sherman.hooks import AgentSpec

def create_plugin_manager() -> pluggy.PluginManager:
    """Create and configure the plugin manager with AgentSpec hooks."""
    pm = pluggy.PluginManager("sherman")
    pm.add_hookspecs(AgentSpec)
    return pm
```

Single function. Returns a configured `PluginManager`.

### Connections
Used by `main.py` and tests. Depends on `hooks.py`.

---

## Module 4: `sherman/llm.py`

**File:** `/Users/sderle/code/sherman/sherman/llm.py`

### Public API

```python
import aiohttp

class LLMClient:
    """Async client for OpenAI-compatible Chat Completions API."""

    def __init__(self, base_url: str, model: str) -> None:
        """
        Args:
            base_url: e.g. "http://192.168.1.88:8080"
            model: e.g. "qwen3.6-35b-a3b"
        """

    async def start(self) -> None:
        """Create the aiohttp session."""

    async def stop(self) -> None:
        """Close the aiohttp session."""

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> dict:
        """Send a chat completion request.

        Args:
            messages: Conversation messages in OpenAI format.
            tools: Tool schemas in OpenAI format. Omitted if None/empty.

        Returns:
            The full response dict from the API.

        Raises:
            aiohttp.ClientResponseError: On non-2xx status.
        """
```

### Key implementation details
- `self.session: aiohttp.ClientSession | None = None` -- created in `start()`, closed in `stop()`.
- `chat()` POSTs to `{base_url}/v1/chat/completions`.
- Payload always includes `model` and `messages`. Includes `tools` only when provided and non-empty.
- No timeout is set at this layer -- the caller or the session can configure it.
- `base_url` is stripped of trailing slash in `__init__`.

### Connections
Used by `agent_loop.py` (`run_agent_loop`) and `conversation.py` (compaction summarization).

---

## Module 5: `sherman/agent_loop.py`

**File:** `/Users/sderle/code/sherman/sherman/agent_loop.py`

### Public API

```python
import inspect
import json
import re
from typing import Callable

from pydantic import create_model

from sherman.llm import LLMClient


async def run_agent_loop(
    client: LLMClient,
    messages: list[dict],
    tools: dict[str, Callable],
    tool_schemas: list[dict],
    max_turns: int = 10,
) -> str:
    """Run the agent loop to completion.

    Mutates messages in place -- appends assistant and tool messages.

    Args:
        client: LLM client.
        messages: Conversation history.
        tools: Map of tool name -> async callable.
        tool_schemas: JSON schemas for tool definitions.
        max_turns: Safety limit on tool-calling rounds.

    Returns:
        The final text response from the model.
    """


def tool_to_schema(fn: Callable) -> dict:
    """Generate a Chat Completions tool schema from a typed function.

    Uses inspect.signature for parameter types and fn.__doc__ for
    the description (first line only). Pydantic generates the JSON
    schema from a dynamically created model.

    Args:
        fn: An async or sync callable with type annotations.

    Returns:
        A dict in OpenAI tool schema format:
        {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
    """


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from text.

    Defensive fallback -- should be a no-op with llama-server + Qwen3.6
    since thinking goes into reasoning_content, not content.
    """
```

### Key implementation details for `run_agent_loop`
- Loop up to `max_turns` times.
- Each iteration: call `client.chat(messages, tools=tool_schemas or None)`.
- Extract `response["choices"][0]["message"]`.
- Append the assistant message to `messages`.
- If no `tool_calls` in the message, return `msg.get("content", "")`.
- If tool_calls present, iterate them: parse `call["function"]["arguments"]` as JSON, look up the function in `tools` dict, call it with `await fn(**fn_args)`, catch exceptions and return error strings, append tool result messages.
- After the loop exhausts `max_turns`, return `"(max tool-calling rounds reached)"`.

### Key implementation details for `tool_to_schema`
- Use `inspect.signature(fn)` to get parameters.
- For each parameter, use its annotation (default to `str` if missing).
- Build a Pydantic model with `create_model()`.
- Call `model_json_schema()` on it.
- Return the OpenAI tool schema dict wrapping the JSON schema.
- Description is `(fn.__doc__ or "").strip().split("\n")[0]`.
- Strip `title` keys from the Pydantic-generated schema (both top-level and per-property) before returning. Pydantic v2 adds these automatically but they are not part of the OpenAI tool schema spec.

### Key implementation details for `strip_thinking`
- `re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()`

### Connections
`run_agent_loop` depends on `LLMClient` from `llm.py`. `tool_to_schema` is standalone (uses pydantic). Both are used by the `AgentLoopPlugin` in Phase 2.

---

## Module 6: `sherman/conversation.py`

**File:** `/Users/sderle/code/sherman/sherman/conversation.py`

### Public API

```python
import json
import time

import aiosqlite

from sherman.llm import LLMClient


class ConversationLog:
    """Append-only persistent log + active prompt management."""

    def __init__(self, db: aiosqlite.Connection, channel_id: str) -> None:
        """
        Args:
            db: An open aiosqlite connection.
            channel_id: Channel identifier (e.g. "irc:#lex").
        """

    messages: list[dict]       # active prompt messages (not including system)
    system_prompt: str         # set by caller before use

    async def load(self) -> None:
        """Load conversation history from the database for this channel.

        Populates self.messages from the persistent log, ordered by timestamp.
        """

    async def append(self, message: dict) -> None:
        """Append a message to both the active list and the persistent log."""

    def token_estimate(self) -> int:
        """Rough token count: (chars in system_prompt + all message content) / 3.5."""

    async def compact_if_needed(self, client: LLMClient, max_tokens: int) -> None:
        """If token_estimate >= 80% of max_tokens, summarize older messages.

        Keeps the last 20 messages, summarizes the rest into a single
        assistant message prefixed with "[Summary of earlier conversation]".
        Compaction only affects self.messages, not the persistent log.
        """

    def build_prompt(self) -> list[dict]:
        """Return [system_message, *self.messages] for LLM input."""


async def init_db(db: aiosqlite.Connection) -> None:
    """Create the message_log table and index if they don't exist."""
```

### Private methods
- `_persist(message: dict) -> None` -- inserts a single message into `message_log` with `channel_id`, `json.dumps(message)`, and `time.time()`.
- `_summarize(client: LLMClient, messages: list[dict]) -> str` -- sends a summarization prompt to the LLM using `client.chat()` and returns the resulting text string.

### Key implementation details
- `_persist(message)` inserts into `message_log` with `channel_id`, `json.dumps(message)`, and `time.time()`.
- `load()` queries `SELECT message FROM message_log WHERE channel_id = ? ORDER BY timestamp`, deserializes each row, populates `self.messages`.
- `compact_if_needed()` early-returns if `token_estimate() < max_tokens * 0.8` or `len(self.messages) <= 20`. Otherwise, calls `_summarize()` on the older messages and replaces `self.messages` with `[summary_msg, *recent_20]`.
- `_summarize(client, messages)` sends a summarization prompt to the LLM and returns the text.
- `build_prompt()` prepends the system message.

### Deviation
`init_db` is extracted as a standalone async function rather than being a method on `AgentLoopPlugin`. This keeps conversation.py self-contained and testable without the plugin class. The `AgentLoopPlugin` (Phase 2) calls `init_db` during its `on_start`.

### Schema
```sql
CREATE TABLE IF NOT EXISTS message_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id TEXT NOT NULL,
    message TEXT NOT NULL,
    timestamp REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_log_channel ON message_log(channel_id, timestamp);
```

### Connections
Uses `LLMClient` for compaction summarization. Used by `AgentLoopPlugin` (Phase 2).

---

## Module 7: `sherman/main.py`

**File:** `/Users/sderle/code/sherman/sherman/main.py`

### Public API

```python
import asyncio
import signal
from pathlib import Path

import yaml

from sherman.plugin_manager import create_plugin_manager


async def main(config_path: str = "agent.yaml") -> None:
    """Daemon entry point.

    1. Load YAML config.
    2. Create plugin manager.
    3. Call on_start on all registered plugins.
    4. Wait for SIGINT or SIGTERM.
    5. Call on_stop on all registered plugins.
    """


def cli() -> None:
    """Console script entry point. Calls asyncio.run(main())."""
```

### Key implementation details
- Loads config from YAML file. Raises `FileNotFoundError` if file doesn't exist.
- Creates plugin manager via `create_plugin_manager()`.
- In Phase 1, no plugins are registered by `main.py` itself. The function is a skeleton that proves the lifecycle works.
- Signal handling: `loop.add_signal_handler(sig, stop_event.set)` for SIGINT and SIGTERM.
- `stop_event = asyncio.Event()` -- `await stop_event.wait()` blocks until signal.
- After signal, calls `await pm.ahook.on_stop()`.
- Config path can be overridden (useful for tests).

### Connections
Depends on `plugin_manager.py`. In Phase 2+, will register `AgentLoopPlugin` and core tools.

### `pyproject.toml` entry point
```toml
[project.scripts]
sherman = "sherman.main:cli"
```

---

## Tests

All tests in `/Users/sderle/code/sherman/tests/`.

### `tests/__init__.py`

Empty.

### `tests/conftest.py`

Shared fixtures:

`db` -- async fixture:

```python
@pytest_asyncio.fixture
async def db():
    async with aiosqlite.connect(":memory:") as conn:
        await init_db(conn)
        yield conn
```

`plugin_manager` -- fixture that returns `create_plugin_manager()`.

### `tests/test_hooks.py`

Tests:
- `test_hookspec_marker_exists` -- `hookspec` is a `HookspecMarker` instance.
- `test_hookimpl_marker_exists` -- `hookimpl` is a `HookimplMarker` instance.
- `test_register_plugin_with_hookimpl` -- Create a class with `@hookimpl` on `on_start`, register it with a plugin manager, call `pm.ahook.on_start(config={})`, verify it was called.
- `test_sync_hook_register_tools` -- Create a class implementing `register_tools` with `@hookimpl`, register it, call `pm.hook.register_tools(tool_registry=some_list)`, verify the list was mutated.
- `test_multiple_plugins_receive_hook` -- Register two plugins implementing `on_message`, fire `on_message`, verify both received the call.

### `tests/test_llm.py`

Uses `unittest.mock.AsyncMock` to mock HTTP.

Tests:
- `test_chat_simple_response` -- Mock the HTTP response to return a valid chat completion. Call `client.chat(messages)`. Verify the return value matches the mock. Verify the POST was made to `{base_url}/v1/chat/completions` with correct payload.
- `test_chat_with_tools` -- Same but with `tools` parameter. Verify `tools` appears in the payload.
- `test_chat_no_tools_omits_key` -- Call `chat(messages, tools=None)`. Verify `tools` is NOT in the payload.
- `test_start_creates_session` -- After `start()`, `self.session` is not None.
- `test_stop_closes_session` -- After `stop()`, session is closed.
- `test_chat_raises_on_http_error` -- Mock a 500 response. Verify `ClientResponseError` is raised.

### `tests/test_agent_loop.py`

Mock `LLMClient.chat` using `AsyncMock`.

Tests:
- `test_simple_response_no_tools` -- Mock returns a message with `content` and no `tool_calls`. Verify `run_agent_loop` returns the content string. Verify `messages` has the assistant message appended.
- `test_single_tool_call` -- Mock returns a tool call on first call, then a text response on second call. Provide a mock async tool function. Verify the tool was called with correct args. Verify the final return is the text response. Verify `messages` contains: assistant (with tool_call), tool result, assistant (final).
- `test_unknown_tool_returns_error` -- Mock returns a tool call for a name not in `tools` dict. Verify the tool result message contains `"Error: unknown tool"`.
- `test_tool_exception_returns_error` -- Mock tool raises an exception. Verify the tool result message contains `"Error:"`.
- `test_max_turns_exceeded` -- Mock always returns tool calls. Verify return is `"(max tool-calling rounds reached)"` after `max_turns` iterations.
- `test_multiple_tool_calls_in_one_turn` -- Mock returns two tool calls in a single message. Verify both tools are called and both results are appended.
- `test_tool_to_schema_basic` -- Define a function with type hints and docstring. Call `tool_to_schema`. Verify the returned dict has correct structure: `type`, `function.name`, `function.description`, `function.parameters`.
- `test_tool_to_schema_multiple_params` -- Function with multiple typed params. Verify all appear in the schema.
- `test_tool_to_schema_no_annotations` -- Function with no type hints. Verify defaults to string.
- `test_strip_thinking_removes_tags` -- Verify `<think>...</think>` is removed.
- `test_strip_thinking_no_tags` -- Verify text without tags is returned unchanged.
- `test_strip_thinking_multiline` -- Verify multiline thinking blocks are removed.

### `tests/test_conversation.py`

Uses the `db` fixture from conftest.

Tests:
- `test_append_persists_message` -- Append a message, query the database directly, verify the row exists with correct channel_id and JSON content.
- `test_load_restores_messages` -- Insert rows into the database, call `load()`, verify `self.messages` matches.
- `test_load_orders_by_timestamp` -- Insert rows with out-of-order timestamps, verify `load()` returns them in timestamp order.
- `test_load_filters_by_channel` -- Insert rows for two channels, verify `load()` only returns messages for the requested channel.
- `test_token_estimate` -- Set system_prompt and messages with known content lengths. Verify the estimate is `total_chars / 3.5` (integer).
- `test_build_prompt` -- Set system_prompt and messages. Verify `build_prompt()` returns `[{"role": "system", "content": system_prompt}, *messages]`.
- `test_compact_if_needed_below_threshold` -- Set token estimate below 80% of max. Verify messages are unchanged after `compact_if_needed`.
- `test_compact_if_needed_triggers` -- Set up >20 messages that exceed the threshold. Create a mock LLMClient with `chat = AsyncMock(return_value={"choices": [{"message": {"content": "mock summary"}}]})` and pass it as the `client` argument. Verify messages are replaced: `[summary_msg, *last_20]`.
- `test_compact_if_needed_few_messages` -- Set up 15 messages (<=20) that exceed the threshold. Verify no compaction occurs (can't meaningfully compact).
- `test_init_db_creates_table` -- Call `init_db`, query `sqlite_master`, verify `message_log` table and `idx_log_channel` index exist.

### `tests/test_main.py`

Tests:
- `test_main_loads_config_and_creates_pm` -- Create a temp YAML config file. Patch `create_plugin_manager` to track calls. Send SIGINT shortly after starting `main()`. Verify `create_plugin_manager` was called.
- `test_main_calls_on_start_and_on_stop` -- Patch `sherman.main.create_plugin_manager` to return a PM with a pre-registered mock plugin (a class with `@hookimpl` on `on_start` and `on_stop`, with both methods as `AsyncMock`). Run `main()` with a quick SIGINT. Verify `on_start` and `on_stop` were both called on the mock plugin.
- `test_main_missing_config_raises` -- This is an async test. Use `await main("nonexistent.yaml")` inside `pytest.raises(FileNotFoundError)`.

---

## Implementation order and dependencies

The modules have these dependency edges:

```
hooks.py          (no deps)
plugin_manager.py -> hooks.py
llm.py            (no deps within sherman; uses aiohttp)
agent_loop.py     -> llm.py (LLMClient type hint), pydantic
conversation.py   -> llm.py, aiosqlite
main.py           -> plugin_manager.py, yaml
```

Parallelizable groups for Red TDD (tests can be written independently):
- hooks.py + plugin_manager.py tests (tasks 5)
- llm.py tests (task 6)
- agent_loop.py tests (task 7)
- conversation.py tests (task 8)
- main.py tests (task 9)

Parallelizable groups for Green TDD:
- hooks.py + plugin_manager.py (task 12) — no deps
- llm.py (task 13) — no deps
- conversation.py (task 15) — depends on llm.py type only
- agent_loop.py (task 14) — depends on llm.py
- main.py (task 16) — depends on all above
