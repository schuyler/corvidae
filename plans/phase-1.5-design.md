# Phase 1.5 Design: Channel Abstraction

## 1. Overview

Phase 1.5 introduces a `Channel` abstraction to replace raw `channel_id: str`
strings throughout the hook system. Today, every consumer parses channel ID
strings independently. The Channel object parses the transport:scope pair once,
carries per-channel configuration, and owns a reference to the channel's
conversation log.

This is a breaking change to four hook signatures (`on_message`,
`send_message`, `on_agent_response`, `on_task_complete`). It must land before
Phase 2 (agent loop plugin) so the agent loop is built against `Channel` from
the start.

### What changes

- New file: `sherman/channel.py` (Channel, ChannelConfig, ChannelRegistry, load_channel_config)
- Modified: `sherman/hooks.py` (four hookspec signatures)
- Modified: `sherman/main.py` (registry creation, config loading, registry injection)
- New file: `tests/test_channel.py`
- Modified: `tests/test_hooks.py` (updated signatures and call sites)

### What does NOT change

- `sherman/conversation.py` — ConversationLog still takes `channel_id: str` in
  its constructor. The Channel object passes `channel.id` when creating one.
- `sherman/agent_loop.py` — no changes (it has no hook impls or channel awareness)
- `sherman/llm.py` — no changes
- `tests/test_conversation.py` — no changes (tests ConversationLog with raw
  strings, which remains valid)
- `tests/test_agent_loop.py` — no changes
- `tests/test_llm.py` — no changes
- `tests/test_main.py` — no changes (mocks PM entirely; registry is internal
  to main)
- `tests/conftest.py` — no changes needed for Phase 1.5 (channel fixtures go
  in test_channel.py)

## 2. New File: `sherman/channel.py`

```python
"""Channel abstraction for conversation scoping."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sherman.conversation import ConversationLog


@dataclass
class ChannelConfig:
    """Per-channel configuration. Falls back to agent-level defaults
    for any field set to None."""

    system_prompt: str | None = None
    max_context_tokens: int | None = None
    keep_thinking_in_history: bool | None = None
    # Future: tool allowlist/denylist, response length, etc.

    def resolve(self, agent_defaults: dict) -> dict:
        """Merge channel overrides with agent-level defaults.

        Channel config wins where set; agent defaults fill the gaps.
        Uses `is not None` for all fields so that falsy values (empty
        string, 0) are treated as intentional overrides.
        """
        return {
            "system_prompt": (
                self.system_prompt if self.system_prompt is not None
                else agent_defaults.get("system_prompt", "You are a helpful assistant.")
            ),
            "max_context_tokens": (
                self.max_context_tokens if self.max_context_tokens is not None
                else agent_defaults.get("max_context_tokens", 24000)
            ),
            "keep_thinking_in_history": (
                self.keep_thinking_in_history
                if self.keep_thinking_in_history is not None
                else agent_defaults.get("keep_thinking_in_history", True)
            ),
        }


@dataclass
class Channel:
    """A conversation scope tied to a transport."""

    transport: str          # "irc", "signal", "cli"
    scope: str              # "#lex", "+15551234567", "local"
    config: ChannelConfig = field(default_factory=ChannelConfig)
    conversation: ConversationLog | None = None
    created_at: float = field(default_factory=time)
    last_active: float = field(default_factory=time)

    @property
    def id(self) -> str:
        """The string key used for storage and routing."""
        return f"{self.transport}:{self.scope}"

    def touch(self) -> None:
        """Update last_active to now."""
        self.last_active = time()

    def matches_transport(self, transport_name: str) -> bool:
        """Check if this channel belongs to the given transport."""
        return self.transport == transport_name


class ChannelRegistry:
    """Manages the lifecycle of channels."""

    def __init__(self, agent_defaults: dict) -> None:
        self.agent_defaults = agent_defaults
        self.channels: dict[str, Channel] = {}

    def get_or_create(
        self,
        transport: str,
        scope: str,
        config: ChannelConfig | None = None,
    ) -> Channel:
        """Look up an existing channel or create a new one.

        Transports call this when a message arrives. The first message
        on a new scope creates the channel.
        """
        channel_id = f"{transport}:{scope}"
        if channel_id not in self.channels:
            self.channels[channel_id] = Channel(
                transport=transport,
                scope=scope,
                config=config or ChannelConfig(),
            )
        channel = self.channels[channel_id]
        channel.touch()
        return channel

    def get(self, channel_id: str) -> Channel | None:
        """Look up a channel by its string ID. Returns None if not found."""
        return self.channels.get(channel_id)

    def resolve_config(self, channel: Channel) -> dict:
        """Resolve channel config with agent-level fallbacks."""
        return channel.config.resolve(self.agent_defaults)

    def all(self) -> list[Channel]:
        """Return all registered channels."""
        return list(self.channels.values())

    def by_transport(self, transport: str) -> list[Channel]:
        """Return all channels for a given transport."""
        return [c for c in self.channels.values() if c.transport == transport]


def load_channel_config(config: dict, registry: ChannelRegistry) -> None:
    """Pre-register channels from the config file.

    Precondition: This must run before on_start is called. The ordering
    in main.py enforces this — load_channel_config is called between
    plugin manager creation and the on_start hook.

    Reads the top-level "channels" dict from config. Each key must be
    in "transport:scope" format.

    Raises:
        ValueError: If a channel key does not contain a colon separator.
    """
    channels_config = config.get("channels", {})
    for channel_id, overrides in channels_config.items():
        parts = channel_id.split(":", 1)
        if len(parts) != 2:
            raise ValueError(
                f"Invalid channel key {channel_id!r} — expected 'transport:scope'"
            )
        transport, scope = parts
        channel_config = ChannelConfig(
            system_prompt=overrides.get("system_prompt"),
            max_context_tokens=overrides.get("max_context_tokens"),
            keep_thinking_in_history=overrides.get("keep_thinking_in_history"),
        )
        registry.get_or_create(transport, scope, config=channel_config)
```

## 3. Changes to `sherman/hooks.py`

Four hookspecs change their signature from `channel_id: str` to
`channel: Channel`. The `on_start`, `on_stop`, and `register_tools` hooks
are unchanged.

### Before

```python
import apluggy as pluggy

hookspec = pluggy.HookspecMarker("sherman")
hookimpl = pluggy.HookimplMarker("sherman")

class AgentSpec:
    @hookspec
    async def on_message(self, channel_id: str, sender: str, text: str) -> None: ...

    @hookspec
    async def send_message(self, channel_id: str, text: str) -> None: ...

    @hookspec
    async def on_agent_response(
        self, channel_id: str, request_text: str, response_text: str
    ) -> None: ...

    @hookspec
    async def on_task_complete(
        self, channel_id: str, task_id: str, result: str
    ) -> None: ...
```

### After

```python
from __future__ import annotations

from typing import TYPE_CHECKING

import apluggy as pluggy

if TYPE_CHECKING:
    from sherman.channel import Channel

hookspec = pluggy.HookspecMarker("sherman")
hookimpl = pluggy.HookimplMarker("sherman")


class AgentSpec:
    @hookspec
    async def on_start(self, config: dict) -> None:
        """Called once when the daemon starts, after config is loaded."""

    @hookspec
    async def on_stop(self) -> None:
        """Called once when the daemon receives SIGINT or SIGTERM."""

    @hookspec
    async def on_message(self, channel: Channel, sender: str, text: str) -> None:
        """Called when an inbound message arrives on a channel."""

    @hookspec
    async def send_message(self, channel: Channel, text: str) -> None:
        """Called to deliver an outbound message to a channel."""

    @hookspec
    def register_tools(self, tool_registry: list) -> None:
        """Called during startup so plugins can add tools to the agent loop."""

    @hookspec
    async def on_agent_response(
        self, channel: Channel, request_text: str, response_text: str
    ) -> None:
        """Called after the agent loop produces a response."""

    @hookspec
    async def on_task_complete(
        self, channel: Channel, task_id: str, result: str
    ) -> None:
        """Called when a background task finishes."""
```

### Summary of changes

| Hook               | Parameter change                          |
|---------------------|------------------------------------------|
| `on_message`        | `channel_id: str` → `channel: Channel`   |
| `send_message`      | `channel_id: str` → `channel: Channel`   |
| `on_agent_response` | `channel_id: str` → `channel: Channel`   |
| `on_task_complete`  | `channel_id: str` → `channel: Channel`   |
| `on_start`          | unchanged                                |
| `on_stop`           | unchanged                                |
| `register_tools`    | unchanged                                |

## 4. Changes to `sherman/main.py`

### Before

```python
async def main(config_path: str = "agent.yaml") -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    pm = create_plugin_manager()

    await pm.ahook.on_start(config=config)
    # ... signal handling, shutdown ...
```

### After

```python
from sherman.channel import ChannelRegistry, load_channel_config

async def main(config_path: str = "agent.yaml") -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    pm = create_plugin_manager()

    # Extract agent-level defaults for channel config resolution
    agent_defaults = config.get("agent", {})
    registry = ChannelRegistry(agent_defaults)

    # Option B injection: attach registry to PM so plugins access it
    # via self.pm.registry
    pm.registry = registry

    # Pre-register channels from YAML config (must happen before on_start)
    load_channel_config(config, registry)

    await pm.ahook.on_start(config=config)
    # ... signal handling, shutdown ...
```

Key changes:
1. Import `ChannelRegistry` and `load_channel_config` from `sherman.channel`
2. Create `ChannelRegistry(agent_defaults)` after PM creation
3. Attach registry to PM: `pm.registry = registry`
4. Call `load_channel_config(config, registry)` before `on_start`

## 5. Changes to Existing Tests

### `tests/test_hooks.py`

The `test_multiple_plugins_receive_hook` test uses `channel_id` in hook
implementations and calls. It must change.

**Before:**
```python
async def test_multiple_plugins_receive_hook():
    received = []

    class PluginA:
        @hookimpl
        async def on_message(self, channel_id, sender, text):
            received.append("A")

    class PluginB:
        @hookimpl
        async def on_message(self, channel_id, sender, text):
            received.append("B")

    pm = create_plugin_manager()
    pm.register(PluginA())
    pm.register(PluginB())
    await pm.ahook.on_message(channel_id="test", sender="user", text="hi")
    assert sorted(received) == ["A", "B"]
```

**After:**
```python
async def test_multiple_plugins_receive_hook():
    from sherman.channel import Channel

    received = []

    class PluginA:
        @hookimpl
        async def on_message(self, channel, sender, text):
            received.append("A")

    class PluginB:
        @hookimpl
        async def on_message(self, channel, sender, text):
            received.append("B")

    pm = create_plugin_manager()
    pm.register(PluginA())
    pm.register(PluginB())
    ch = Channel(transport="test", scope="scope")
    await pm.ahook.on_message(channel=ch, sender="user", text="hi")
    assert sorted(received) == ["A", "B"]
```

### No other test files require changes

- `tests/test_main.py` — mocks PM entirely; `MagicMock` absorbs `.registry`
  assignment silently. `load_channel_config` is a no-op with no `channels`
  key in test config.
- `tests/test_conversation.py` — uses raw string `channel_id` with
  `ConversationLog`, which is unchanged.
- `tests/test_agent_loop.py` — tests `run_agent_loop`, no channel awareness.
- `tests/test_llm.py` — tests `LLMClient`, no channel awareness.

## 6. Test Plan: `tests/test_channel.py`

All tests use pytest-asyncio auto mode. No database fixtures needed.

### 6.1 ChannelConfig.resolve tests

1. **`test_resolve_all_defaults`** — `ChannelConfig()` with no overrides
   resolves all fields from agent_defaults dict.

2. **`test_resolve_channel_overrides_win`** — Channel-level values override
   agent defaults for all three fields.

3. **`test_resolve_partial_override`** — Only some fields overridden; rest
   from defaults.

4. **`test_resolve_empty_string_is_not_none`** — Empty string system_prompt
   is an intentional override, not a fallback. (C1 fix validation.)

5. **`test_resolve_zero_max_context_tokens_is_not_none`** — Zero
   max_context_tokens is an intentional override. (C1 fix validation.)

6. **`test_resolve_false_keep_thinking_is_not_none`** — `False`
   keep_thinking_in_history is an intentional override.

7. **`test_resolve_missing_agent_defaults`** — Empty agent_defaults dict
   falls back to hardcoded defaults.

### 6.2 Channel tests

8. **`test_channel_id_property`** — `channel.id` returns `"transport:scope"`.

9. **`test_channel_matches_transport`** — Returns True for matching, False
   otherwise.

10. **`test_channel_touch_updates_last_active`** — `touch()` updates
    `last_active` to a more recent time.

11. **`test_channel_default_conversation_is_none`** — Default `conversation`
    is None.

12. **`test_channel_default_config`** — Default config has all None fields.

### 6.3 ChannelRegistry tests

13. **`test_get_or_create_new_channel`** — Creates a new channel on first call.

14. **`test_get_or_create_returns_existing`** — Second call with same
    transport+scope returns same object.

15. **`test_get_or_create_touches_channel`** — `get_or_create` calls `touch()`
    on both new and existing channels (updates `last_active`).

16. **`test_get_or_create_with_config`** — Config is applied to new channel.

17. **`test_get_existing`** — `get()` returns channel by string ID.

18. **`test_get_nonexistent`** — `get()` returns None for unknown ID.

19. **`test_resolve_config`** — Delegates to `channel.config.resolve()` with
    registry defaults.

20. **`test_all`** — Returns all registered channels.

21. **`test_by_transport`** — Filters channels by transport name.

### 6.4 load_channel_config tests

22. **`test_load_channel_config_basic`** — Pre-registers channels from config.

23. **`test_load_channel_config_no_channels_section`** — Missing "channels"
    key is a no-op.

24. **`test_load_channel_config_empty_channels`** — Empty channels dict is
    a no-op.

25. **`test_load_channel_config_invalid_key_no_colon`** — Key without colon
    raises ValueError. (C2 fix validation.)

26. **`test_load_channel_config_colon_in_scope`** — Key like `"signal:+1:555"`
    splits correctly on first colon only.

## 7. Implementation Sequence

1. **Create `sherman/channel.py`** — all classes and `load_channel_config`.
   Run `pytest tests/` — all existing tests pass (new file is inert).

2. **Create `tests/test_channel.py`** — all test cases from Section 6.
   Run `pytest tests/test_channel.py` — all new tests pass.

3. **Update `sherman/hooks.py`** — change four hookspec signatures.
   Add `TYPE_CHECKING` import.

4. **Update `tests/test_hooks.py`** — change `test_multiple_plugins_receive_hook`.
   Run `pytest tests/test_hooks.py` — all pass.

5. **Update `sherman/main.py`** — add registry creation, Option B injection,
   and `load_channel_config` call.
   Run `pytest tests/test_main.py` — all pass.

6. **Full test suite**: `pytest tests/` — all tests pass.

## 8. Migration Notes

### Ordering constraint

`load_channel_config` must run before `pm.ahook.on_start`. The code in
`main.py` enforces this structurally. Plugins that access `pm.registry`
during `on_start` will find pre-configured channels already registered.

### Registry injection (Option B)

Plugins access the registry via `self.pm.registry`. This is set in `main.py`
as a plain attribute on the PluginManager instance. Example transport plugin:

```python
class CLIPlugin:
    def __init__(self, pm):
        self.pm = pm

    @hookimpl
    async def on_start(self, config):
        registry = self.pm.registry
        channel = registry.get_or_create("cli", "local")
        self._task = asyncio.create_task(self._read_loop(channel))

    async def _read_loop(self, channel):
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await asyncio.get_running_loop().connect_read_pipe(
            lambda: protocol, sys.stdin
        )
        while True:
            line = await reader.readline()
            if not line:
                break
            text = line.decode().strip()
            if not text:
                continue
            channel.touch()
            await self.pm.ahook.on_message(
                channel=channel, sender="user", text=text,
            )
```

Note: the transport calls `channel.touch()` on each inbound message. The
agent loop's `on_message` handler should also call `channel.touch()` so
that channels receiving responses update their `last_active` timestamp.

### Backward compatibility

This is a breaking change to hook signatures. All existing hookimpls that
use `channel_id` must be updated to use `channel`. Since Phase 2 (agent
loop plugin) has not been built yet, the only existing hookimpls are in
tests. No external plugin compatibility to maintain.

### ConversationLog integration

`ConversationLog.__init__` still takes `channel_id: str`. When the Phase 2
agent loop plugin attaches a ConversationLog to a Channel, it passes
`channel.id`:

```python
if channel.conversation is None:
    channel.conversation = ConversationLog(self.db, channel.id)
    resolved = self.pm.registry.resolve_config(channel)
    channel.conversation.system_prompt = resolved["system_prompt"]
    await channel.conversation.load()
```

This is Phase 2 work. Phase 1.5 only introduces the data model and registry;
actual ConversationLog attachment happens when the agent loop plugin is built.

### Edge cases

- **Empty transport or scope**: Not validated. `Channel(transport="", scope="")`
  produces `id == ":"`. Stricter validation can be added later if needed.
- **Config overrides for existing channels**: If `get_or_create` is called
  with a config for a channel that already exists, the config is ignored
  (the existing channel is returned as-is). This is intentional — YAML
  config sets initial state, runtime creation does not override it.
- **Thread safety**: Not a concern. The daemon is single-threaded asyncio.
