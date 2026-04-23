# Phase 2 Design: Agent Loop Plugin

## 1. Summary

Phase 2 introduces `AgentLoopPlugin` — a plugin that implements `on_start`,
`on_message`, `on_stop`, and `register_tools` hooks to wire the existing
`LLMClient`, `run_agent_loop()`, and `ConversationLog` components into the
plugin system.

**New file:** `sherman/agent_loop_plugin.py`
**New test file:** `tests/test_agent_loop_plugin.py`
**Modified:** `sherman/main.py` (register plugin)
**Modified:** `sherman/llm.py` (add optional api_key)
**Modified:** `sherman/agent_loop.py` (add `strip_reasoning_content` helper)

**Modified:** `sherman/channel.py` (fix `keep_thinking_in_history` default)

Files that do NOT change: `hooks.py`, `plugin_manager.py`, `conversation.py`,
existing test files (except `test_channel.py` — one assertion update).

## 2. Discrepancies Between design.md and Existing Code

1. **`self.conversations` dict (design.md line 493):** design.md has the plugin
   maintain its own conversations dict. Phase 1.5 put `conversation` on
   `Channel` itself (`channel.conversation`). Phase 2 description (line
   1232-1243) explicitly says to use `channel.conversation`. **Decision:** Use
   `channel.conversation` (no separate dict).

2. **LLMClient has no `api_key` parameter.** design.md YAML shows
   `api_key: "not-needed"` but the class only takes `base_url` and `model`.
   **Decision:** Add optional `api_key` to `LLMClient` for forward
   compatibility.

3. **Double-append bug (design.md line 541):** design.md sketch appends the
   raw response as a separate message after `run_agent_loop()` already
   appended it to `messages`. `run_agent_loop()` mutates `messages` in place.
   **Decision:** Do NOT double-append. Persist only the new messages that
   `run_agent_loop()` added.

## 3. LLM Configuration from YAML

Add optional `api_key` to `LLMClient`:

```python
# sherman/llm.py — modified constructor
class LLMClient:
    def __init__(self, base_url: str, model: str, api_key: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.session: aiohttp.ClientSession | None = None

    async def start(self) -> None:
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        self.session = aiohttp.ClientSession(headers=headers)
```

The plugin reads config from the `llm` namespace:

```python
llm_config = config.get("llm", {})
self.client = LLMClient(
    base_url=llm_config["base_url"],
    model=llm_config["model"],
    api_key=llm_config.get("api_key"),
)
```

`base_url` and `model` are required (KeyError on missing). `api_key` is optional.

## 4. AgentLoopPlugin Class

```python
# sherman/agent_loop_plugin.py

import logging
from collections.abc import Callable

import aiosqlite

from sherman.agent_loop import run_agent_loop, strip_reasoning_content, strip_thinking, tool_to_schema
from sherman.conversation import ConversationLog, init_db
from sherman.hooks import hookimpl
from sherman.llm import LLMClient

logger = logging.getLogger(__name__)


class AgentLoopPlugin:
    """Plugin that wires the agent loop into the hook system."""

    def __init__(self, pm):
        self.pm = pm
        self.client: LLMClient | None = None
        self.db: aiosqlite.Connection | None = None
        self.tools: dict[str, Callable] = {}
        self.tool_schemas: list[dict] = []

    @hookimpl
    async def on_start(self, config: dict) -> None:
        # 1. Create and start LLM client
        llm_config = config.get("llm", {})
        self.client = LLMClient(
            base_url=llm_config["base_url"],
            model=llm_config["model"],
            api_key=llm_config.get("api_key"),
        )
        await self.client.start()

        # 2. Open SQLite database
        db_path = config.get("daemon", {}).get("session_db", "sessions.db")
        self.db = await aiosqlite.connect(db_path)
        await init_db(self.db)

        # 3. Collect tools from all plugins via register_tools hook (sync).
        # AgentLoopPlugin must be registered AFTER any tool-providing plugins
        # so that register_tools collects all available tools here.
        tool_fns: list = []
        self.pm.hook.register_tools(tool_registry=tool_fns)
        for fn in tool_fns:
            self.tools[fn.__name__] = fn
            self.tool_schemas.append(tool_to_schema(fn))

    @hookimpl
    async def on_message(self, channel, sender: str, text: str) -> None:
        # 1. Lazy-initialize conversation on the channel
        await self._ensure_conversation(channel)
        conv = channel.conversation
        resolved = self.pm.registry.resolve_config(channel)

        # 2. Append user message to conversation log (persisted)
        await conv.append({"role": "user", "content": text})

        # 3. Compact if approaching context limit
        await conv.compact_if_needed(self.client, resolved["max_context_tokens"])

        # 4. Build prompt and run agent loop
        messages = conv.build_prompt()
        # NOTE: messages_before must use len(messages), NOT len(conv.messages).
        # build_prompt() returns [system_msg, *conv.messages], so len(messages)
        # is always 1 + len(conv.messages). Using len(conv.messages) here would
        # be off by one and would incorrectly include the last user message in
        # new_messages (step 5).
        messages_before = len(messages)

        try:
            raw_response = await run_agent_loop(
                self.client, messages, self.tools, self.tool_schemas
            )
        except Exception:
            logger.exception("run_agent_loop failed for channel %s", channel.id)
            await self.pm.ahook.send_message(
                channel=channel,
                text="Sorry, I encountered an error and could not process your message.",
            )
            return

        # 5. Persist new messages appended by run_agent_loop
        new_messages = messages[messages_before:]
        for msg in new_messages:
            await conv.append(msg)

        # 6. Thinking token handling for active history.
        # Only strip the newly appended messages — earlier messages were
        # already stripped on prior turns.
        if not resolved["keep_thinking_in_history"]:
            strip_reasoning_content(new_messages)

        # 7. Strip thinking for display and send response
        display_response = strip_thinking(raw_response)

        await self.pm.ahook.on_agent_response(
            channel=channel,
            request_text=text,
            response_text=display_response,
        )
        await self.pm.ahook.send_message(
            channel=channel,
            text=display_response,
        )

    @hookimpl
    async def on_stop(self) -> None:
        if self.client:
            await self.client.stop()
        if self.db:
            await self.db.close()

    async def _ensure_conversation(self, channel) -> None:
        """Lazy-initialize ConversationLog on a channel if not present."""
        if channel.conversation is not None:
            return
        conv = ConversationLog(self.db, channel.id)
        resolved = self.pm.registry.resolve_config(channel)
        conv.system_prompt = resolved["system_prompt"]
        await conv.load()
        channel.conversation = conv
```

## 5. Thinking Token Handling — Three-Layer Strategy

Smoke test confirmed llama-server puts thinking tokens in `reasoning_content`,
not `content`.

**Layer 1 — Display:** Use `content` directly. Apply `strip_thinking()` as
defensive fallback in case `<think>` tags leak into `content`. Display text
goes to `send_message` and `on_agent_response`.

**Layer 2 — Persistent log:** `ConversationLog.append()` persists the full
message dict as JSON, including `reasoning_content`. No stripping. The
persistent log is immutable ground truth.

**Layer 3 — Active history:** After `run_agent_loop()` returns, if
`keep_thinking_in_history` is false (default), strip `reasoning_content` from
the **newly appended** assistant messages only. In-memory mutation only —
persistent log already has the full message. Older messages were already
stripped on prior turns, so re-iterating the full history is unnecessary.

**Default value for `keep_thinking_in_history`:** The recommended default is
`False`. The hardcoded fallback in `ChannelConfig.resolve()` (`channel.py`)
must be changed from `True` to `False`:

```python
# sherman/channel.py — in ChannelConfig.resolve()
"keep_thinking_in_history": (
    self.keep_thinking_in_history
    if self.keep_thinking_in_history is not None
    else agent_defaults.get("keep_thinking_in_history", False)  # was True
),
```

Note: `tests/test_channel.py::test_resolve_missing_agent_defaults` asserts the
current `True` default and must be updated to expect `False`.

Add utility to `agent_loop.py`:

```python
def strip_reasoning_content(messages: list[dict]) -> None:
    """Remove reasoning_content from assistant messages in place."""
    for msg in messages:
        if msg.get("role") == "assistant":
            msg.pop("reasoning_content", None)
```

## 6. ConversationLog Integration via channel.conversation

Lazy initialization on first message:

```python
if channel.conversation is None:
    channel.conversation = ConversationLog(self.db, channel.id)
    resolved = self.pm.registry.resolve_config(channel)
    channel.conversation.system_prompt = resolved["system_prompt"]
    await channel.conversation.load()
```

- No upfront DB queries for channels that never receive messages
- Config resolved at initialization time
- `load()` restores prior history from SQLite
- ConversationLog attached to `channel.conversation`, NOT stored in a separate
  dict on the plugin

## 7. register_tools Hook Wiring

`register_tools` is **synchronous** (not async). Called via
`pm.hook.register_tools()`, not `pm.ahook`. Confirmed in
`tests/test_hooks.py:41`.

The AgentLoopPlugin does NOT implement `register_tools` — it *calls* it to
collect tools from other plugins during `on_start`.

**Registration order requirement:** `AgentLoopPlugin` must be registered AFTER
any plugin that implements `register_tools`. The `register_tools` call in
`on_start` collects tools from plugins already registered at that point. Any
tool-providing plugin registered after `AgentLoopPlugin` will be missed. Phase
3 will introduce `CoreToolsPlugin`, which must be registered before
`AgentLoopPlugin`.

## 8. Prompt Construction and Message Flow

1. `conv.append(user_msg)` — persists user message, adds to `conv.messages`
2. `conv.compact_if_needed(...)` — may summarize older messages
3. `messages = conv.build_prompt()` — returns `[system, *conv.messages]` as a
   **new list** (but dicts are shared references)
4. `run_agent_loop(client, messages, ...)` — **mutates** `messages` by
   appending assistant and tool messages
5. After return: `messages[messages_before:]` are the new messages needing
   persistence

**Warning — `messages_before` must use the `build_prompt()` list:**
`messages_before = len(messages)` must be computed from the list returned by
`build_prompt()`, NOT from `len(conv.messages)`. `build_prompt()` prepends the
system message at index 0, so `len(messages) == len(conv.messages) + 1`.
Using `len(conv.messages)` would set `messages_before` one too low, causing
`messages[messages_before:]` to include the last user message as a "new"
message and double-persist it.

**Important:** `build_prompt()` returns a new list (shallow copy via `+`
operator), but individual message dicts are the *same objects* as in
`conv.messages`. New messages appended by `run_agent_loop()` are new dicts
NOT yet in `conv.messages` — the plugin must append them via `conv.append()`.

## 9. Changes to main.py

```python
from sherman.agent_loop_plugin import AgentLoopPlugin

# In main(), after load_channel_config() and before pm.ahook.on_start():
# Register any tool-providing plugins (e.g., CoreToolsPlugin in Phase 3) FIRST,
# then AgentLoopPlugin last, so register_tools in on_start collects all tools.
agent_loop = AgentLoopPlugin(pm)
pm.register(agent_loop, name="agent_loop")
```

Registered as a core plugin (not hot-loadable). Must be placed after
`load_channel_config()` (so channels are pre-registered) and before
`pm.ahook.on_start()` (so the plugin receives `on_start`).

## 10. Test Strategy

**Test file:** `tests/test_agent_loop_plugin.py`

**What needs mocking:**
- `LLMClient` — mock `start()`, `stop()`, `chat()` (same pattern as existing
  `test_agent_loop.py` and `test_llm.py`)
- `aiosqlite.connect` — use `:memory:` databases (same pattern as `conftest.py`)
- `pm.ahook.send_message` and `pm.ahook.on_agent_response` — `AsyncMock` to
  verify they are called with correct args
- `pm.hook.register_tools` — verify it is called; optionally have a test
  plugin that appends tools

**Required test fixture — `pm.registry`:**
Tests that call `on_message` use `self.pm.registry.resolve_config(channel)`
internally. The plugin manager must have a real `ChannelRegistry` instance
attached as `pm.registry`, constructed with `agent_defaults`:

```python
from sherman.channel import ChannelRegistry

registry = ChannelRegistry(agent_defaults={
    "system_prompt": "You are a test assistant.",
    "max_context_tokens": 8000,
    "keep_thinking_in_history": False,
})
pm.registry = registry
```

**Test scenarios:**

1. `test_on_start_creates_client_and_db` — on_start with valid config creates
   LLMClient and opens DB. Verify client.start() was called.
2. `test_on_start_collects_tools` — Register a test plugin implementing
   register_tools. Verify tools are collected and schemas generated.
3. `test_on_start_missing_llm_config_raises` — Config without `llm.base_url`
   or `llm.model` raises KeyError.
4. `test_on_message_initializes_conversation` — First message on a channel
   creates ConversationLog, sets system_prompt from resolved config, calls load().
5. `test_on_message_reuses_existing_conversation` — Second message on same
   channel reuses the ConversationLog.
6. `test_on_message_appends_user_message` — User message is persisted in the
   conversation log.
7. `test_on_message_calls_run_agent_loop` — Verify run_agent_loop is called
   with correct arguments.
8. `test_on_message_persists_agent_response` — New messages from run_agent_loop
   are persisted to the conversation log.
9. `test_on_message_sends_response` — Verify send_message hook is called with
   display text. Verify on_agent_response hook is called.
10. `test_on_message_strips_thinking_for_display` — If raw response contains
    `<think>` tags, display text has them stripped.
11. `test_on_message_strips_reasoning_content_from_history` — When
    keep_thinking_in_history is false, reasoning_content removed from newly
    appended in-memory messages. Pre-existing assistant messages are not touched.
12. `test_on_message_preserves_reasoning_content_in_history` — When
    keep_thinking_in_history is true, reasoning_content remains.
13. `test_on_message_preserves_reasoning_content_in_persistent_log` —
    Regardless of keep_thinking_in_history setting, the persisted record must
    contain reasoning_content. **Must query `message_log` table in SQLite
    directly** — do NOT rely on `conv.messages`, which may have been stripped
    in memory. Same pattern as `test_conversation.py::test_append_persists_message`.
14. `test_on_message_compacts_before_agent_loop` — When token estimate is high,
    compact_if_needed runs before agent loop.
15. `test_on_message_per_channel_config` — Two channels with different configs
    behave according to their resolved config.
16. `test_on_stop_cleans_up` — on_stop closes LLM client and DB.
17. `test_on_message_tool_call_round_trip` — Mock LLM returns tool call, tool
    executes, LLM returns final text. Full round trip through plugin.
18. `test_on_message_run_agent_loop_error` — When `run_agent_loop` raises an
    exception, `send_message` is called with an error message, `on_agent_response`
    is NOT called, and `on_message` returns without re-raising. The user message
    persisted before the failure remains in the log (dangling user message is
    acceptable ground truth).
