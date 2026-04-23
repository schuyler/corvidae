# Channel Abstraction Design

## Problem

`channel_id` is a string convention with no structure. Every consumer
parses it independently: transports check prefixes, the agent loop
uses it as a dict key, the conversation log uses it as a database key.
There's no place to hang channel-specific behavior, config, or state.

## What a Channel Is

A Channel is a conversation scope. It represents one ongoing
interaction between the agent and one or more participants, over a
specific transport. Examples:

- `irc:#lex` — the #lex IRC channel
- `irc:schuyler` — a DM on IRC
- `signal:+15551234567` — a Signal conversation
- `cli:local` — the CLI stdin/stdout session

A Channel owns:

- **Identity** — transport name + scope, parsed once
- **Conversation log** — the append-only message history and active
  prompt for this channel
- **Configuration** — channel-specific overrides (system prompt,
  response style, tool availability, context limit)
- **Metadata** — created_at, last_active, participant info

A Channel does NOT own:

- The transport connection (that's the transport plugin's job)
- The LLM client (that's the agent loop's job)
- Tool implementations (those are registered globally via hooks)

## Data Model

```python
from dataclasses import dataclass, field
from time import time


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
        """
        return {
            "system_prompt": (
                self.system_prompt
                or agent_defaults.get("system_prompt", "You are a helpful assistant.")
            ),
            "max_context_tokens": (
                self.max_context_tokens
                or agent_defaults.get("max_context_tokens", 24000)
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
    conversation: object = None   # ConversationLog, assigned by agent loop
    created_at: float = field(default_factory=time)
    last_active: float = field(default_factory=time)

    @property
    def id(self) -> str:
        """The string key used for storage and routing."""
        return f"{self.transport}:{self.scope}"

    def touch(self):
        self.last_active = time()

    def matches_transport(self, transport_name: str) -> bool:
        return self.transport == transport_name
```

## Channel Registry

The registry is the single source of truth for all active channels.
The agent loop, transports, and any plugin that needs to look up a
channel goes through the registry.

```python
class ChannelRegistry:
    """Manages the lifecycle of channels."""

    def __init__(self, agent_defaults: dict):
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
        return self.channels.get(channel_id)

    def resolve_config(self, channel: Channel) -> dict:
        """Resolve channel config with agent-level fallbacks."""
        return channel.config.resolve(self.agent_defaults)

    def all(self) -> list[Channel]:
        return list(self.channels.values())

    def by_transport(self, transport: str) -> list[Channel]:
        return [c for c in self.channels.values() if c.transport == transport]
```

## How It Changes the Existing Design

### Transports

Before:
```python
await self.pm.ahook.on_message(
    channel_id="irc:#lex",
    sender=source,
    text=message,
)
```

After:
```python
channel = self.registry.get_or_create("irc", target)
await self.pm.ahook.on_message(
    channel=channel,
    sender=source,
    text=message,
)
```

The transport no longer constructs a string. It asks the registry for
a Channel object (creating one if this is a new conversation) and
passes the object through the hook.

### send_message

Before, transports checked the `channel_id` prefix:
```python
if not channel_id.startswith("irc:"):
    return
```

After, they check the channel's transport field:
```python
if not channel.matches_transport("irc"):
    return
await self.client.message(channel.scope, text)
```

No string parsing. The transport name and scope are already separated.

### Agent Loop

Before:
```python
conv = await self._get_conversation(channel_id)
```

After:
```python
if channel.conversation is None:
    channel.conversation = ConversationLog(self.db, channel.id)
    resolved = self.registry.resolve_config(channel)
    channel.conversation.system_prompt = resolved["system_prompt"]
    await channel.conversation.load()

conv = channel.conversation
```

The ConversationLog is attached to the Channel, not looked up from a
dict. The system prompt comes from the resolved channel config, which
means different channels can have different prompts without any
special-casing in the agent loop.

### Hook Signature Changes

The `on_message` and `send_message` hookspecs change:

```python
class AgentSpec:
    @hookspec
    async def on_message(self, channel: Channel, sender: str, text: str) -> None:
        """A message arrived from a transport."""

    @hookspec
    async def send_message(self, channel: Channel, text: str) -> None:
        """Send a message to a channel."""

    @hookspec
    async def on_agent_response(
        self, channel: Channel, request_text: str, response_text: str
    ) -> None:
        """Called after the agent produces a response."""

    @hookspec
    async def on_task_complete(
        self, channel: Channel, task_id: str, result: str
    ) -> None:
        """A background task finished."""
```

`channel_id: str` becomes `channel: Channel` everywhere. This is a
breaking change to the hook signatures, so it should be done before
Phase 2 (agent loop plugin) is complete.

## Channel Configuration from YAML

Channels can be pre-configured in the YAML config. Channels that
aren't pre-configured get default config when created on first
message.

```yaml
agent:
  max_context_tokens: 24000
  keep_thinking_in_history: true
  system_prompt: |
    You are a personal assistant. Be concise and direct.

channels:
  irc:#lex:
    system_prompt: |
      You are a personal assistant on IRC. Keep responses short.
      Use plain text, no markdown.

  cli:local:
    system_prompt: |
      You are a personal assistant in a terminal session.
      You can use markdown formatting.
    max_context_tokens: 32000

  # A future Signal channel with different behavior
  # signal:+15551234567:
  #   system_prompt: |
  #     You are a personal assistant on Signal.
```

Loading pre-configured channels at startup:

```python
def load_channel_config(config: dict, registry: ChannelRegistry):
    """Pre-register channels from the config file."""
    channels_config = config.get("channels", {})
    for channel_id, overrides in channels_config.items():
        transport, scope = channel_id.split(":", 1)
        channel_config = ChannelConfig(
            system_prompt=overrides.get("system_prompt"),
            max_context_tokens=overrides.get("max_context_tokens"),
            keep_thinking_in_history=overrides.get("keep_thinking_in_history"),
        )
        registry.get_or_create(transport, scope, config=channel_config)
```

## CLI Transport Example

The CLI transport with the Channel abstraction:

```python
import sys
import asyncio
from hooks import hookimpl


class CLIPlugin:
    def __init__(self, pm, registry):
        self.pm = pm
        self.registry = registry
        self._task = None

    @hookimpl
    async def on_start(self, config):
        self._task = asyncio.create_task(self._read_loop())

    async def _read_loop(self):
        channel = self.registry.get_or_create("cli", "local")

        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await asyncio.get_event_loop().connect_read_pipe(
            lambda: protocol, sys.stdin
        )
        print("Agent ready. Type a message, or Ctrl-D to quit.\n")
        while True:
            sys.stdout.write("> ")
            sys.stdout.flush()
            line = await reader.readline()
            if not line:
                break
            text = line.decode().strip()
            if not text:
                continue
            await self.pm.ahook.on_message(
                channel=channel,
                sender="user",
                text=text,
            )

    @hookimpl
    async def send_message(self, channel, text):
        if not channel.matches_transport("cli"):
            return
        print(f"\n{text}\n")

    @hookimpl
    async def on_stop(self):
        if self._task:
            self._task.cancel()


def create_plugin(pm, config):
    # The registry is passed via config or injected separately —
    # see implementation note below.
    registry = config["_registry"]
    return CLIPlugin(pm, registry)
```

## Implementation Notes

### Registry Injection

The `ChannelRegistry` needs to be accessible to both transports (which
create channels) and the agent loop (which attaches conversations to
them). Two options:

**Option A: Pass through config.** Stash the registry in the config
dict as `config["_registry"]`. Simple, slightly ugly.

**Option B: Expose on the plugin manager.** Attach it as an attribute
of the PluginManager instance: `pm.registry = ChannelRegistry(...)`.
Plugins access it via `self.pm.registry`. Cleaner, but couples plugins
to the PM's shape.

**Option C: A hookspec.** Add a `get_registry` hook that the agent
loop plugin implements, and other plugins call to get a reference.
Most decoupled, but indirection for the sake of indirection.

Recommend Option B for now. It's direct and the PM is already the
shared object every plugin holds.

### When to Introduce This

The Channel abstraction should go in before the agent loop plugin is
finalized (Phase 2). The hook signatures change, and it's cheaper to
build the agent loop against `Channel` from the start than to retrofit
it. The CLI transport is the first consumer and validates the pattern
before IRC is built.

Sequence:
1. Phase 1 completes (hooks, LLM client, conversation log, etc.)
2. Introduce Channel, ChannelRegistry, update hookspecs
3. Build CLI transport against the Channel abstraction
4. Validate end-to-end: CLI → agent loop → LLM → CLI
5. Build IRC transport following the same pattern