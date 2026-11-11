# Corvidae Design Document

An asyncio agent daemon that connects to IRC (and later other
transports), routes messages through an LLM via the OpenAI-compatible
Chat Completions API, and supports plugin components via pluggy.

## Architecture

The daemon is a single Python asyncio process. It connects to a local
LLM served by llama-server via OpenAI-compatible API.

Three layers:

1. **Plugin system (apluggy)** — defines lifecycle hooks and extension
   points.

2. **Agent loop** — manages prompt construction, tool calling, LLM
   interaction via aiohttp. Owns conversation state. Uses single-turn
   dispatch: one LLM call per queue item, tool calls dispatched as
   background tasks, results arrive as notifications.

3. **Transport plugins** — CLI, IRC, Signal, Discord, Zulip, web, ACP (stdio). Each
   transport converts platform-specific messages to/from a common Channel
   abstraction.

This document covers the kernel: the `corvidae` package. The cognition
plugins (memory, funnel, appraisal, critique, outcome log) are a second
distribution in the same repository, `corvidae-cognition`, installed with
the `cognition` extra and documented in [cognition.md](cognition.md). The
kernel never imports them; they use the plugin API listed in
[plugin-guide.md](plugin-guide.md#what-the-kernel-exports-to-plugins).

```
┌──────────────────────────────────────────────────────┐
│                   Plugin Manager                     │
│               (apluggy.PluginManager)                │
│                                                      │
│  ┌─────────────┐  ┌─────────┐  ┌──────────────────┐ │
│  │ Persistence │  │   IRC   │  │  Agent Plugin    │ │
│  │   Plugin    │  │Transport│  │  (agent loop)    │ │
│  │  (DB/conv)  │  │ Plugin  │  │                  │ │
│  └──────┬──────┘  └─────────┘  └────────┬─────────┘ │
│         │                               │           │
│  ┌──────▼──────┐              ┌──────────▼─────────┐ │
│  │   SQLite    │              │    llama-server     │ │
│  │  (sessions) │              │   (OpenAI-compat)   │ │
│  └─────────────┘              └────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

## Dependencies

```
apluggy          # async-aware pluggy wrapper
aiohttp          # async HTTP client (LLM API + web_fetch tool)
aiosqlite        # async SQLite for conversation persistence
pydantic         # tool schema generation from type hints
pydle            # asyncio IRC client
pyyaml           # YAML config parsing
```

## Hook Specifications

Defined in `hooks.py`. All hooks are async (called via `pm.ahook`)
except `register_tools` and `register_llm_stages`, which are sync. This is a representative subset of
`AgentSpec` covering the turn-loop and message-lifecycle hooks; the
[Hook reference](#hook-reference) table below is the complete, authoritative
list of every hook (including lifecycle hooks like `on_init` not shown
here).

```python
class AgentSpec:
    async def on_start(self, config: dict) -> None
    async def on_stop(self) -> None
    async def on_message(self, channel: Channel, sender: str, text: str, trace_id: str) -> None
    async def send(self, channel: Channel, event: Outbound) -> None  # corvidae/outbound.py
    def register_tools(self, tool_registry: list) -> None  # sync; plugins append Tool instances or bare callables
    def register_llm_stages(self, stages: dict[str, str]) -> None  # sync; plugins add stages[name] = "chat" | "embedding"
    async def on_agent_response(self, channel: Channel, request_text: str, response_text: str, trace_id: str | None, annotations: dict, logprobs: dict | None) -> None
    async def on_notify(self, channel: Channel, source: str, text: str, tool_call_id: str | None, annotations: dict | None) -> None
    async def should_process_message(self, channel: Channel, sender: str, text: str, trace_id: str | None) -> bool | None
    async def should_send_response(self, channel: Channel, text: str, emission: str, trace_id: str | None, annotations: dict) -> bool | None
    async def on_message_admitted(self, channel: Channel, trace_id: str, sender: str, text: str) -> None
    async def on_message_rejected(self, channel: Channel, trace_id: str, sender: str, text: str) -> None
    async def on_message_persisted(self, channel: Channel, trace_id: str, rowid: int | None, text: str, annotations: dict) -> None
    async def resolve_llm_error(self, channel: Channel, error: Exception) -> str | None
    async def resolve_compaction(self, channel: Channel, conversation: ContextWindow, max_tokens: int) -> None
    async def transform_tool_result(self, tool_name: str, result: str, channel: Channel | None) -> str | None
    async def before_agent_turn(self, channel: Channel, trace_id: str | None, annotations: dict) -> None
    async def after_persist_assistant(self, channel: Channel, message: dict) -> None
    async def transform_display_text(self, channel: Channel, text: str, result_message: dict) -> str | None
    async def transform_system_prompt(self, channel: Channel, text: str) -> str | None
    async def resolve_conversation(self, channel: Channel) -> list[dict] | None
    async def resolve_channel(self, channel_id: str) -> Channel | None
    async def resolve_register_prompt(self, channel: Channel) -> str | None
    async def persist_message(self, channel: Channel, message: dict, message_type: MessageType) -> int | None
    async def before_compaction(self, channel: Channel, compacted_ids: list[int], cut: int) -> None
    async def on_compaction(self, channel: Channel, summary_msg: dict, retain_count: int, compacted_ids: list[int]) -> None
```

`create_plugin_manager()` in `hooks.py` creates the manager and adds
the `AgentSpec` hookspecs.

### Plugin-declared hookspecs

A plugin can own hooks that only it calls by setting `hookspecs` on its
class to a class of `@hookspec` methods (#229). The kernel doesn't change
per hook: `install_plugin_hookspecs(pm)` installs every declaration at
startup, after entry points load and before the arg-binding guard and
`on_init`. pluggy verifies hookimpls registered before their spec when
`add_hookspecs` runs, so registration order between owner and
implementers doesn't matter.

`install_plugin_hookspecs` makes two passes, so no outcome depends on
entry-point order. Pass one collects every declaration and fails on any
name with more than one owner, counting names that already have a spec
(`AgentSpec`). Pass two, per plugin, checks the name against the hook
grammar (`HOOK_GRAMMAR` in `hooks.py`, the same table
`tests/test_hook_grammar.py` uses for `AgentSpec`), with the owner's
registered plugin name right after the prefix (`on_web_<name>`). It also
checks the pluggy options match the prefix, installs the specs, and
requires the owner's own `trylast` seed for each `transform_` hook. Bare
imperatives are kernel-only. Every failure raises `PluginValidationError`
naming the hook, the plugin and the fix.

Implementers declare a hard (`depends_on`) or optional
(`optional_depends_on`) dependency on the owner. If an optional owner is
absent, its implementers' impls stay registered with no spec and are
never called. `report_orphan_hookimpls(pm)`, run after
`validate_dependencies`, logs each hookimpl with no spec at WARNING,
naming the plugin and the hook. The exception is a hook whose owner
segment names one of the implementer's `optional_depends_on` plugins
that isn't loaded (blocked or not installed), which logs at DEBUG.

Known limit: a plugin name containing an underscore makes the owner
segment ambiguous against a shorter name (`memory` could declare
`on_memory_tools_x`). That can't cause a silent clash, because a second
declaration of the same name still fails. Its only effect is that such an
orphan can be logged at DEBUG when the real owner is a different absent
plugin.

The `webhook` plugin owns `on_webhook_subscribe(subscriptions)`, through which
plugins register handlers for webhook endpoints; see "Webhook ingress" under
Transports. The `web` plugin owns `on_web_console_init(panels)`, through which
plugins add read-only console panels; see "Web" under Transports. The `funnel` plugin (cognition extra) owns `on_funnel_ingest(channel, trace_id, annotations, submit)`, the pull path described under [Funnel pull path](#funnel-pull-path).

### Funnel pull path

Some sources cannot give their entries when asked, so they hand the funnel promises instead. An implementation of `on_funnel_ingest` calls `submit(source, future)` and returns; the future resolves to an `Ingest` or to `None`. Implementers submit rather than return because apluggy's `ahook` gathers implementations without `return_exceptions`: one raising implementation would lose every other one's return value. Each channel's `before_agent_turn` runs four steps in order, none of which returns early: expire (only when `funnel.ingest_ttl` is set), sweep finished promises into the channel's next-turn queue, drain the queues through the normal admission path, then collect by calling the hook. Collecting last means a promise submitted in a turn is first swept in the next. A submit for a `(channel, source)` that already has a pending promise cancels that promise, and `on_stop` cancels all of them. Expiry runs only at the channel's next turn, so an idle channel's pending promise runs until then or until shutdown. The contract and its limits are in [cognition.md](cognition.md#funnelplugin-packagescorvidae-cognitioncorvidae_cognitionfunnelpy). A source that searches per user turn does not implement the hook by hand: `RecallSourceMixin` (`corvidae_cognition/recall.py`) does, adding the user-turn gate, a per-channel cooldown and single search in flight, a timeout and fail-soft logging, so the source supplies `search` and `render` only ([Recall sources](cognition.md#recall-sources-packagescorvidae-cognitioncorvidae_cognitionrecallpy)).

### Funnel `ponder` tool

The funnel registers `ponder()`, a bounded wait (5 s) for recall already in flight. It adds no kernel code. A tool call is a Task whose result comes back through `on_notify` with a `tool_call_id`, and the agent loop already holds the next LLM call until the channel's `pending_tool_call_ids` empties, so a tool that simply waits delays the next assistant turn and keeps the channel busy (idle work stays off). What it waits on is the funnel's own pull promises: every `on_funnel_ingest` submission for a turn is made in that turn's `before_agent_turn`, before the model call, so the promises the channel holds when the agent calls a tool are the recall in flight for it (a message that arrives during the wait gets its own turn, which can submit or supersede promises; the tool does not follow those). It returns at once when none is running, and otherwise waits with `asyncio.wait` (which does not cancel the recalls) for the first promise with entries or for all to settle. Push arrivals (`ingest`, `ingest_and_wake`) are not waited on. See [cognition.md](cognition.md#funnelplugin-packagescorvidae-cognitioncorvidae_cognitionfunnelpy).

### Broadcast dispatch and result resolution

All hooks are called via `pm.ahook.<hook_name>(...)`. A broadcast hook
runs every registered implementation and returns a list of results; a
`firstresult=True` hook (below) stops at the first non-None result and
returns that single value. Gate hooks
that need a single resolved value pass the result list to
`resolve_reject_wins(results)` from `hooks.py`: any `False` in the results
returns `False`; otherwise any `True` returns `True`; otherwise `None`.
`should_process_message` and `should_send_response` are the hooks resolved
this way.

Hooks marked `firstresult=True` in the hookspec (`resolve_llm_error`,
`resolve_compaction`, `transform_tool_result`, `transform_display_text`,
`transform_system_prompt`, `resolve_conversation`, `resolve_channel`) use pluggy's own first-non-None-wins dispatch — the
first handler to return a non-None value stops the chain; no separate
resolver function is involved. `transform_tool_result` and
`transform_display_text` and `transform_system_prompt` are wrapper chains: `@hookimpl(wrapper=True)`
implementations compose transforms over an identity seed.

Pure broadcast hooks (e.g. `on_start`, `on_stop`) resolve neither way;
their return values are ignored. `persist_message` is the one broadcast
hook resolved with `resolve_single_result`: every implementation runs and
at most one (persistence) returns a value.

Hook names follow a grammar: the prefix tells you the resolution
(`on_`/`before_`/`after_` observe, `should_` vetoes, `resolve_` takes the
first non-None result, `transform_` runs a wrapper chain), and a short
allowlist of bare imperatives covers the rest. The table and the rule for
adding a hook are in [plugin-guide.md](plugin-guide.md#hook-grammar);
`tests/test_hook_grammar.py` enforces the name/option half for
`AgentSpec`, and `install_plugin_hookspecs` enforces it at startup for
[plugin-declared hooks](#plugin-declared-hookspecs), which also carry the
owner's plugin name after the prefix.

### Hook reference

| Hook | Semantics | Call site |
|------|-----------|-----------|
| `on_init` | broadcast | `Runtime.start()`, after all plugins are registered, before `on_start`; store `pm`/`config`, resolve other plugins — no runtime resources |
| `on_start` | broadcast | daemon startup, after config load |
| `on_stop` | broadcast | SIGINT/SIGTERM |
| `on_message` | broadcast | inbound message from any transport; the transport mints and passes `trace_id` (required, no default) |
| `send` | broadcast (imperative) | outbound delivery. `Agent` sends `OutboundMessage` (replies, error text), `OutboundThinking` (when the LLM response carries `reasoning_content`), `OutboundProgress` (intermediate text alongside tool calls) and `OutboundToolStatus` (tool call dispatched); `TaskPlugin` sends `OutboundToolStatus` on completion. Transports receive it through `TransportPlugin.send`, which filters by `owns(channel)` and dispatches by event type |
| `register_tools` | broadcast (sync) | startup tool collection |
| `register_llm_stages` | broadcast (sync, imperative) | `LLMPlugin.on_start`, once, before it creates clients: plugins add `stages[name] = kind` (`chat` or `embedding`) for each model-calling stage they own |
| `on_agent_response` | broadcast | after agent loop produces a text response, including a final the gate vetoed (or an empty one): the assistant row is already in the window, tagged `UNSENT` when not sent (`final_was_unsent`), and the transport send comes after. Discord, Signal, Zulip, ACP and CLI implement it to release per-turn state for an unsent final; `SilencePlugin` records declines; `MemoryPlugin` runs its turn consolidation trigger from it, awaiting nothing, since the send waits on the hook |
| `on_notify` | broadcast | inject a notification into a channel queue |
| `should_process_message` | broadcast / resolve_reject_wins | `on_message`, before enqueue, carrying the transport-minted `trace_id`; no default implementation; None from empty broadcast allows all messages |
| `should_send_response` | broadcast / resolve_reject_wins | the [output gate](#the-output-gate): before each outbound emission, with `emission` in `final`, `progress`, `thinking`, `error` and the item's `trace_id` and `annotations`; a veto of `final` stores the row as `UNSENT`; `CritiquePlugin` vetoes a critique-origin `final` that is exactly `NO_REPLY`, and `SilencePlugin` vetoes an exact `NO_REPLY` final of any origin (cognition extra); a raising implementation fails open (logged, sent); no kernel implementation, so with none registered everything is sent |
| `on_message_admitted` | broadcast | `on_message`, after a USER message is admitted; fires exactly once per admitted message |
| `on_message_rejected` | broadcast | `on_message`, after a USER message is vetoed; fires exactly once per rejected message |
| `on_message_persisted` | broadcast | dequeue, after the trace-originating row is persisted; fires once per originating item regardless of outcome (`rowid` may be None). Each message of a coalesced burst fires it under its own trace, in arrival order, before the turn's model call |
| `resolve_llm_error` | firstresult=True | `_run_turn`, after LLM exception; return error string or None for default |
| `resolve_compaction` | firstresult=True | `_process_queue_item`, step 6; the first non-None return stops the chain (`CompactionPlugin` is `trylast`, so a default-priority replacement runs first); the caller ignores the value |
| `transform_tool_result` | firstresult=True wrapper chain | `dispatch_tool_call` (both subagent and main agent paths), after tool execution; return replacement string or None for default |
| `before_agent_turn` | broadcast | `_process_queue_item`, before LLM call; plugins inject context into conversation log |
| `after_persist_assistant` | broadcast | `_process_queue_item`, after assistant message is persisted; plugins may mutate the in-memory dict |
| `transform_display_text` | firstresult=True wrapper chain | `_resolve_display_text`, before the reply is sent; return transformed text or None to leave unchanged |
| `transform_system_prompt` | firstresult=True wrapper chain | `Agent.channel_system_prompt`, when the identity and register are composed: a new conversation, a restart, a compaction boundary, and the background views that share it. Return the transformed prompt or None to leave unchanged. A raising wrapper is logged at WARNING with the channel and the untransformed prompt is used. A transform that modifies the identity+register prefix is rejected with a WARNING; the untransformed prompt is used. Appending is the expected use. `SilencePlugin` appends its instruction on conversational channels |
| `resolve_conversation` | firstresult=True | `_process_queue_item`, when `channel.conversation is None`; return list of tagged message dicts or None to defer |
| `resolve_register_prompt` | firstresult=True | `Agent`, when a conversation starts on a channel with no `register_prompt` configured at any layer; the owning transport returns its default register text, others return None. `SignalPlugin`, `DiscordPlugin`, `ZulipPlugin` and `IRCPlugin` implement it, from fragments shipped in `corvidae/channels/registers/`. ACP, web and CLI return nothing. No answer means no register |
| `resolve_channel` | firstresult=True | `SchedulerPlugin`, for a persisted channel id the registry does not hold (e.g. a reminder due after a restart); a transport returns the live Channel for ids it owns, created with transport-specific config, or None to defer. `SignalPlugin`, `DiscordPlugin` and `ZulipPlugin` implement it. A transport may also return None for an id it owns but cannot build yet (Signal before its first alias resolution). For an id whose prefix names a transport that implements the hook, the scheduler retries; for other ids, callers fall back to `registry.get_or_create` from the split id |
| `persist_message` | broadcast (imperative) | `_process_queue_item` after every `conv.append()`, and `FunnelPlugin` for the CONTEXT it persists; every implementation runs (persistence, JSONL logging). `PersistencePlugin` returns the `message_log` rowid and the caller resolves the list with `resolve_single_result` into `_db_id` |
| `before_compaction` | broadcast, awaited | `CompactionPlugin`, after the summary succeeded and before `replace_with_summary()`; the window still holds `messages[:cut]`, so a plugin can fork over the dropped prefix (see [Forks](#forks)); failures logged, never block compaction |
| `on_compaction` | broadcast | `CompactionPlugin`, after `replace_with_summary()`; side effects only (persistence) |
| `on_llm_request` | broadcast | `LLMClient.chat()` via LLMPlugin's injected observer, before the retry loop; side effects only (observability) |
| `on_llm_response` | broadcast | `LLMClient.chat()` via the observer, exactly once per call (success or terminal failure); names the model the request was sent with (an `extra_body` override wins, #297) and carries llama-server's `timings` verbatim, plus the call's start time, total duration across retried attempts, attempts made, and the request's message and tool counts; side effects only (metrics, usage log) |
| `on_metrics` | broadcast | any plugin via `pm.ahook.on_metrics(...)`; built-in emission from `MetricsPlugin.on_llm_response`; never emit from inside an `on_metrics` implementation (recursion) |
| `on_config_reload` | broadcast | `ConfigWatcherPlugin`, when `agent.yaml`'s mtime changes; per-plugin error isolation |

**Resolution at compaction.** `on_compaction` re-resolves the system prompt
from config at every compaction boundary, not only when a runtime override
changed it. A config-file edit to `system_prompt` or `register_prompt`
that was previously ignored until restart now applies at the next
compaction. This fits principle 15 (the override path is unaffected: the
guard compares against the freshly composed prompt, not a stored one) and
the plugin-sections clause in PRD §3.3.8. Between boundaries the prompt
is byte-identical (principle 2).

## Channel System

`channel.py` provides multi-transport, multi-scope channel management.

**ChannelConfig** — per-channel config with agent-level fallback:

```python
@dataclass
class ChannelConfig:
    register_prompt: str | list[str] | None = None   # appended after agent.system_prompt
    max_context_tokens: int | None = None
    reasoning_history: str | None = None   # drop | exchange | keep
    max_turns: int | None = None
```

`resolve(agent_defaults, runtime_overrides=None)` returns a dict with all
values resolved. Resolution order: built-in defaults → agent-level YAML →
matching transport-wide entries, least specific first → per-channel YAML →
runtime overrides (set by `set_settings` tool).
Defaults: `max_context_tokens=24000`, `reasoning_history="drop"`,
`max_turns=10`. `resolve` normalizes `reasoning_history` to the
`ReasoningHistory` enum and raises `ValueError` on any other value.
`validate_agent_config` (startup and config reload) and `load_channel_config`
reject an invalid value or the removed `keep_thinking_in_history` key up front.

**Channel** — identifies a transport + scope combination:

```python
@dataclass
class Channel:
    transport: str
    scope: str
    config: ChannelConfig
    conversation: ContextWindow | None = None
    created_at: float
    last_active: float
    turn_counter: int = 0                   # consecutive LLM turns without user message
    pending_tool_call_ids: set = set()      # tool call IDs awaiting results; cleared when all results collected
    runtime_overrides: dict = {}            # per-channel runtime overrides set by set_settings tool
    wide_entries: tuple[WideEntry, ...] = ()  # matching transport-wide entries, least specific first
```

`channel.id` returns `"{transport}:{scope}"`. `channel.disabled_tools` is
the union of `config.disabled_tools` and every `wide_entries` list.

**Transport-wide entries** (#203). A `channels:` key ending in `*` is a
`WideEntry(key, prefix, config)`: it covers every channel whose id starts
with `prefix`, the key without its `*` (`signal:*`, `discord:dm:*`,
`acp:*`). They exist for channels config can't name ahead of time: Signal
DMs keyed on ACIs, ACP sessions with minted ids, Discord channels the bot is
mentioned in. `parse_wide_key` accepts only `transport:*` and
`transport:<scope-prefix>*`; only a trailing `*` is special, so exact keys
that contain `*` (IRC allows it) load as before. Two prefixes that match the
same id are always prefixes of each other, so ordering by length is total:
more specific wins whatever order the keys appear in.

**ChannelRegistry** — lifecycle management. `get_or_create(transport,
scope, config)` returns an existing channel or creates one, assigning its
matching wide entries. `resolve_config(channel)` overlays the channel's
wide entries' scalar fields on `agent_defaults`, then calls
`channel.config.resolve(layered, runtime_overrides=channel.runtime_overrides)`,
so the existing per-field fallback gives exact > wide > agent and runtime
overrides are factored into every resolution. `set_wide_entries(entries)`
replaces the wide set and re-matches every registered channel, logging at
INFO each channel whose matches change.

`load_channel_config(config, registry)` pre-registers channels from
YAML before `on_start`. It is `parse_channel_config(config)` (pure: validates
every entry, exact and wide, and rejects keys outside `CHANNEL_ENTRY_KEYS`, touching no registry) followed by
`apply_channel_config(parsed, registry)`. Reload calls the two itself:
`ConfigWatcherPlugin` parses before it updates `agent_defaults`, so a rejected
`channels:` entry skips the whole reload. Applying sets the wide entries first, so
exact channels it then registers are matched at creation, and it never
registers a wide key as a channel. Wide entries therefore reload like
`agent:`: scalar keys apply to running channels from the next turn. A
conversation's system prompt is set once, when its `ContextWindow` is
created (compaction and forks keep it), so a prompt edit at any layer
reaches only conversations that start after the reload.

**Live versus restart-only (#408, #473).** Each plugin declares the config-key
prefixes its `on_config_reload` re-reads, as `reloadable_keys` on
`CorvidaePlugin`. Before dispatching the reload, `ConfigWatcherPlugin`
compares the new config against the one the running daemon is using — not
against the previous reload — and logs one WARNING per changed key that no
plugin declares, naming the key and saying a restart applies it. The running
config is the baseline because the warning claims what a restart would change:
a key edited back to its running value changes nothing on restart and is not
reported. A key is reported when its pending value first appears or changes,
not on every poll. Anything not declared is restart-only by default, so a
plugin that forgets to declare a key produces a spurious warning rather than a
silent pending change. A test in `tests/test_config_watcher.py` reads each
reload hook's source and fails if it reads a key its class does not declare, so
the registry cannot drift from the code.

**System prompt layers (#316).** A conversation's system prompt is the
identity (`agent.system_prompt`, never set per channel; a channel-level
`system_prompt` key is skipped at load with a WARNING) followed by a blank line and the
channel's register, then passed through `transform_system_prompt`
(below). `ChannelConfig.resolve` returns the register as
`register_prompt`, or None when no layer sets it; `Agent` then asks the
owning transport through `resolve_register_prompt`. An explicit `""` or
`[]` is no register. A register file that can't be read (`OSError`,
`UnicodeDecodeError`) is logged at WARNING and the conversation starts on
the identity alone.

**`transform_system_prompt`.** `Agent.channel_system_prompt` passes the
composed identity and register through this wrapper chain, so a plugin can
add a section after the register whatever the register's source (a shipped
default, a custom `register_prompt`, or none). The seed on `_SeedHooksPlugin`
returns the text unchanged; a wrapper that raises is logged at WARNING
naming the channel and the untransformed prompt is used. The result is fixed
when the conversation starts or restarts and is not rebuilt at compaction.
`TransportPlugin.conversational` (default False; IRC, Discord, Signal and
Zulip set True) is the neutral fact a plugin reads to tell a chat transport from
one that always answers. `SilencePlugin` is the one implementation: it
appends the `NO_REPLY` instruction on conversational channels
([cognition.md](cognition.md#silenceplugin-packagescorvidae-cognitioncorvidae_cognitionsilencepy)).

**resolve_system_prompt(value, base_dir)** — if `value` is a string,
returns it directly. If a list of paths, reads each file and
concatenates with `\n\n`. Relative paths resolve against `base_dir`
(the directory containing `agent.yaml`). Called at conversation init
time, so editing prompt files takes effect on the next new conversation.
Used for both the identity and a configured register.

## LLM Client

`llm.py` — thin aiohttp wrapper for the Chat Completions API.

```python
class LLMClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        extra_body: dict | None = None,
        max_retries: int = 3,
        retry_base_delay: float = 2.0,
        retry_max_delay: float = 60.0,
        timeout: float | None = None,
        model_not_loaded_wait: float = 60.0,
    )
    async def start(self)
    async def stop(self)
    async def chat(self, messages: list[dict], tools: list[dict] | None = None, extra_body: dict | None = None) -> dict
    def request_model(self, extra_body: dict | None = None) -> str
```

`request_model` returns the `model` a `chat` call with that `extra_body` is
sent with: the configured model, overridden by the instance `extra_body`
and then the call's (`llm.<profile>.extra_body.model` can change the model).
`chat` builds its payload's `model` from it. Memory consolidation stamps it
on the records it writes ([cognition.md](cognition.md#record-stamps)).
`embed` is deliberately not routed through it: it sends the configured
`model` and ignores `extra_body`, so the encoder `embedding_meta` records
can't be switched by an override.

**Merge order.** `chat` builds the payload in three layers, later ones
winning: `model`, `messages` and `tools`; the client's own `extra_body`
(the instance body, which for a profile's client is that profile's
`extra_body`); the `extra_body` passed to the call. So a call-level key
beats the profile's. The main turn relies on that: the channel's runtime
overrides (`set_settings`) are call-level, so on a main turn an override
beats the profile. A fork inverts it for the keys its stage profile sets,
by not sending the shadowed override at all (see Forks, "Prefix parity").
`LLMClient` itself never knows which kind of call it is serving.

`chat` sends `defang_control_tokens(messages)`, not `messages` (#356).
llama-server tokenizes request text with special-token parsing, so a
literal `<|endoftext|>` in any message becomes the real control token. The
model copies it while quoting the text, and generation stops with empty
content. A fetched `<|im_start|>system` would inject template structure.
`defang_control_tokens` walks the messages and rewrites every string match
of `<\|[^|<>\s]{1,40}\|>`, or of a bare `<think>`, `</think>`,
`<tool_call>`, `</tool_call>`, `<tool_response>` or `</tool_response>` tag (#355, #553), to `<` + U+200B + the rest. It returns new
containers, so the window and the stored history keep the original text. It
is a pure function of the messages, so the prompt prefix stays cache-stable.
Every chat profile and caller (main loop, forks, compaction, memory) goes
through `chat`. `embed` and `tools` schemas are not rewritten. The model
can still emit a control token from its own knowledge; this only keeps
input text from supplying one. Other added tokens without the `<|…|>`
shape are not covered.

**Raw reasoning (#355).** llama-server's parsed mode cannot tell a reply
cut off mid-reasoning from a decline: both arrive with empty `content`. When
the client has `raw_reasoning` (`llm.<profile>.raw_reasoning`; the operator
also configures the server to return reasoning inline, e.g. llama.cpp's
`extra_body.reasoning_format: none`), `chat` splits the reply itself (`split_raw_reasoning`): an optional leading
`<think>` is dropped, the text up to the first `</think>` becomes
`reasoning_content`, and the rest becomes `content`. A reply with no
`</think>` is all reasoning with `content: ""`, and the response dict gets a
top-level `reasoning_unfinished: True` (outside the message, so it is never
persisted or sent back). `AgentTurnResult.reasoning_unfinished` carries it.
shortcut: this
assumes the chat template forces thinking open, so set the option only on
such endpoints; elsewhere every non-reasoning reply reads as cut off.

**Reopened reasoning (#348).** After the split, `cut_reopened_reasoning`
looks for a `<think>` or `</think>` in the remaining `content`: the model
finished its answer and started reasoning again instead of ending the turn,
sometimes inventing a user question and answering it. `content` is cut at
the tag (trailing whitespace stripped), the response gets a top-level
`reasoning_reopened: True`, and `chat` logs one warning with the channel
(from attribution, or `(none)`), the model and the full raw completion. The
tail is in neither `content` nor `reasoning_content`, so it reaches no
transport, no row and no memory; the log is its only copy (PRD §3.3.9,
§3.10). This runs on every profile's client. `AgentTurnResult.reasoning_reopened`
carries the flag. `Agent._run_turn` checks it before the cut-off check:
with `reasoning_reopened: rerun` on the main turn's profile (`llm.main` by
default), a final (no tool calls) that
reopened is popped from the messages and regenerated once, and that result
is used whatever it holds (a second reopen is already cut); then the
cut-off check runs on it as on any reply. With `truncate` (the default)
the cut reply is used as is. A stray `</think>` has no start marker, so the
cut lands at the tag and the second block's text before it survives.

Raw mode is decided per call (#360). llama-server can't apply a JSON-schema
grammar with `reasoning_format: none` (HTTP 400), so a call whose merged
payload carries `response_format` (e.g. the forked and standalone appraisal,
critique and memory stages) drops `reasoning_format` from that
request, runs in the server's parsed mode, and is not split: its JSON
`content` is returned as-is, with no cut-off detection. Calls without
`response_format` (main turns, the compaction summary fork) are unchanged.
shortcut: this relies on the server's own default being a parsed mode; a
server started with `--reasoning-format none` still rejects schema calls.
Upgrade to an explicit per-call override if such a deployment appears.
Forks with a schema now send a different `reasoning_format` from main turns;
whether that changes the cached prefix on a given template is unverified.

Retry parameters: `max_retries` (default 3, set to 0 to disable), `retry_base_delay`
(base exponential backoff delay in seconds), `retry_max_delay` (cap on backoff delay).
Retries apply to transient HTTP status codes (429, 500, 502, 503, 504) and connection
errors. Honors `Retry-After` response headers. `timeout` is the total HTTP timeout per
request in seconds; `None` sets no timeout at all.

A llama-server router started with `--no-models-autoload` answers a request
for an unloaded model with HTTP 400 `"model is not loaded"` (#55). That
response is retried with the same backoff until a total of
`model_not_loaded_wait` seconds has been slept (the last sleep is clamped to
fit), independently of `max_retries`; `0` disables it. Any terminal non-2xx
response raises `LLMHTTPError`, a `ClientResponseError` subclass carrying the
response `body` (truncated to 2000 characters) and the client's `model`;
`is_model_not_loaded(error)` recognizes the unloaded-model case.
`is_backend_unreachable(error)` is true for a connection-class failure (any
`aiohttp.ClientConnectionError` that is not also a `TimeoutError`: connect
refused, reset, server disconnected) and for the not-loaded case; it excludes
every timeout class (`ServerTimeoutError`, `SocketTimeoutError`,
`ConnectionTimeoutError`, `TimeoutError`), server HTTP errors and other
exceptions. The Agent retries a turn on it (see Message processing). A malformed
response envelope (missing `choices`, an empty choice list, or a choice with
no `message`) raises `ValueError` naming the provider model and endpoint, the
missing key, and the top-level keys actually received.

### Observability

Every LLM call is metered at the `LLMClient` chokepoint — not the turn
loop — so compaction, subagent, consolidation, appraisal, and critique
background calls are all covered.

**Observer seam.** `LLMClient.observer` (default `None`) is any object
with async `request(**kwargs)`/`response(**kwargs)` methods. `chat()`
mints a `request_id` per call, fires `request` before the retry loop,
and fires `response` exactly once — after success (with the envelope's
`usage` verbatim and measured latency) or when a terminal exception is
about to propagate (with `error` set, `usage=None`). Retried transient
attempts do not fire `response`. Observer failures are individually
caught and logged; they never break the call. `llm.py` stays free of
pluggy imports — `LLMPlugin` injects a `_HookObserver` (defined in
`llm_plugin.py`) that bridges to the `on_llm_request`/`on_llm_response`
hooks with the client's profile (a key under `llm:`, such as `main`) and the
current attribution snapshot.

**Attribution.** `attribution.py` holds a single ContextVar dict
(`stage`, `channel_id`, `trace_id`, `annotations`, plus stage-specific
extras such as `trigger` on consolidation calls). Callers set it at the
top of a logical operation and reset in a `finally`: the Agent turn loop
(`stage="turn"`), CompactionPlugin around its summary call
(`stage="compaction"`, shadowing the turn), and the subagent work loop
(`stage="subagent"`). Stages that can run as forks (compaction,
consolidation, appraisal, critique) add `mode="fork"` (set by
`Agent.fork`) or `mode="standalone"` around their own LLM call; it lands
in `usage_log.mode`. Contextvars snapshot at `asyncio.create_task`
time and TaskQueue workers are created once at startup, so `Task`
captures `contextvars.copy_context()` at creation and the worker runs
`task.work()` inside that context via
`asyncio.create_task(task.work(), context=task.ctx)`.

**Sinks.** `metrics.py` provides `MetricsPlugin` (translates
`on_llm_response` into `on_metrics` events: `llm.tokens.*`,
`llm.latency_ms`, `llm.errors`), `UsageLogPlugin` (one `usage_log` row
per call), and `MetricsJsonlPlugin` (JSONL event sink, config
`daemon.metrics_jsonl`). `outcome_log.py` owns the `exchange_log` table:
one row per exchange, written by `OutcomeLogPlugin`'s hookimpls at
message admission/rejection/persistence and updated in place as memory
retrieval, appraisal, critique, the gates and the silence plugin record
their outcomes. All sinks are
fail-soft.

**Call timing and queue wait (PRD §3.10.6).** Each `usage_log` row records
when its call started and ended, so overlapping calls are found from the
rows alone: in a self-join on `a.id < b.id`, two calls overlap when
`a.started_at < b.ended_at AND b.started_at < a.ended_at` (calls that only
touch at an endpoint do not). `ended_at` is `started_at + total_ms / 1000`,
so it comes from the monotonic duration and cannot disagree with
`total_ms`. `latency_ms` is the final
attempt only. When calls share a llama-server slot (a pinned `id_slot`, or
a single-slot server), a call that arrives while another holds the slot
waits inside the server, and that wait is part of its `latency_ms`. The
server's own work is `prompt_ms + predicted_ms`; what remains is, approximately, the wait:

```sql
SELECT id, stage, latency_ms - (prompt_ms + predicted_ms) AS queue_wait_ms
FROM usage_log;  -- NULL where the backend gave no timings
```

It is not stored: it is a function of stored columns. `prompt_ms` and
`predicted_ms` are NULL, never 0, when the backend reports no `timings`
or the call failed; a reported 0 stays 0. `max_context_tokens` is looked
up at write time through the channel registry (`optional_depends_on`;
`resolve_config(channel)`), so a runtime override that lands between the
request and the write is what the row records; a failing lookup logs a
warning naming the channel and writes NULL. Existing tables gain the
columns in place; older rows read NULL.

```sql
CREATE TABLE usage_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    request_id TEXT NOT NULL,
    profile TEXT NOT NULL,      -- the llm profile of the client that made the call (renamed from `role` in place on startup)
    model TEXT NOT NULL,        -- the model the request was sent with, after any extra_body override (#297)
    stage TEXT,
    channel_id TEXT,
    trace_id TEXT,              -- NULL when the call ran outside any trace
    trigger TEXT,               -- 'compaction', 'idle', 'turn' or 'eval' for consolidation calls, else NULL
    mode TEXT,                  -- 'fork' or 'standalone' for stages that can fork (#51), else NULL
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    cached_tokens INTEGER,      -- usage.prompt_tokens_details.cached_tokens, where the provider reports it
    cache_n INTEGER,            -- llama-server timings.cache_n: prompt tokens reused from cache (PRD §3.13.10)
    prompt_n INTEGER,           -- llama-server timings.prompt_n: prompt tokens prefilled; NULL from other providers
    latency_ms REAL,            -- the final attempt's duration (retried attempts and their sleeps are not in it)
    error TEXT,
    started_at REAL,            -- wall-clock epoch seconds when the call began (#350)
    ended_at REAL,              -- started_at + total_ms / 1000
    total_ms REAL,              -- monotonic, first attempt to end, including retries and the sleeps between
    attempts INTEGER,           -- attempts made; 1 = no retry (transient retries and model-not-loaded waits count)
    message_count INTEGER,      -- messages in the request (embeddings: number of texts)
    tool_count INTEGER,         -- tool schemas in the request (embeddings: 0)
    max_context_tokens INTEGER, -- the channel's budget in force when the row is written; NULL with no channel, no registry or an unknown channel
    prompt_ms REAL,             -- llama-server timings.prompt_ms: the final attempt's prefill time; NULL from other providers and on failure
    predicted_ms REAL           -- llama-server timings.predicted_ms: the final attempt's decode time; NULL likewise
);

CREATE TABLE exchange_log (
    trace_id TEXT PRIMARY KEY,      -- one exchange = what one trace covers (see plugin-guide.md's Traces section)
    channel_id TEXT NOT NULL,
    origin TEXT,                    -- convention: 'user'|'reminder'|'critique'|'heartbeat'|'task'|'webhook:<endpoint>'
    message_rowid INTEGER,          -- message_log.id of the originating message (a coalesced burst's absorbed traces keep theirs; no response completes those rows)
    created_at REAL NOT NULL,
    retrieval_top_score REAL,       -- written by MemoryPlugin retrieval (decayed score)
    retrieval_top_similarity REAL,  -- written by MemoryPlugin retrieval (raw similarity, same candidate)
    retrieval_hit_count INTEGER,    -- written by MemoryPlugin retrieval
    probe_score REAL,               -- written by AppraisalPlugin stage 1
    appraisal TEXT,                 -- JSON; written by AppraisalPlugin stage 2
    provenance_snapshot TEXT,       -- JSON; written by CritiquePlugin
    outcomes TEXT                   -- JSON; merged by multiple writers
);
```

**Eval harness.** Deterministic metric functions (`recall_at_k`, `mrr`,
`tokens_of`) live in `tests/evals/metrics.py` and run in CI with no
network; the fixture format is documented there
(`tests/fixtures/memory_retrieval_basic.json` is the operator-authored
seed). The live runner `scripts/eval_memory.py` is out-of-band only;
live tests carry the `eval` marker (run with `--run-eval`). Its
`--snapshot` mode (#658) scores labelled queries against a copy, taken through a read-only connection,
of a real `sessions.db`; see [cognition.md](cognition.md), "Measuring
retrieval on a real database".

## Agent Loop

The single-turn LLM primitive lives in `turn.py`. The multi-turn loop
that drives subagents lives in `tools/subagent.py`. Text post-processing
utilities for LLM output live in `thinking.py`.

**`turn.run_agent_turn`** — single LLM invocation. Used by Agent for
interactive message processing. Returns an `AgentTurnResult` with the
assistant message, any tool calls, text content, and latency.

```python
@dataclass
class AgentTurnResult:
    message: dict       # raw assistant message from LLM
    tool_calls: list[dict]
    text: str
    latency_ms: float

async def run_agent_turn(
    client: LLMClient,
    messages: list[dict],
    tool_schemas: list[dict],
    extra_body: dict | None = None,
) -> AgentTurnResult
```

`run_agent_turn` appends the assistant message to `messages` in place
before returning. Callers should not append it again.

**`tools.subagent.run_agent_loop`** — multi-turn blocking loop. Used
by subagent execution (its only caller is `SubagentPlugin._launch`).
Calls the LLM via `run_agent_turn`, executes tool calls inline, repeats
until a text response or max_turns. Injects `ToolContext` into tools
that declare a `_ctx` parameter. Accepts an optional `pm` keyword
argument; when provided, passes it to `dispatch_tool_call()`, which
calls the `transform_tool_result` hook after each tool execution. When
`pm` is `None`, the hook is skipped.

Also in `turn.py`:
- `_truncate(text, max_len)` — truncation utility for logging

In `thinking.py` (alongside `ThinkingPlugin`):
- `strip_thinking(text)` — removes `<think>...</think>` blocks
- `strip_reasoning_content(messages)` — removes `reasoning_content`
  from message dicts in place

## Tool System

`tool.py` provides tool registration and schema generation.

**Tool** — wraps a function with its schema:
```python
@dataclass
class Tool:
    name: str
    fn: Callable
    schema: dict
```

`Tool.from_function(fn)` generates the schema via `tool_to_schema`.

**ToolRegistry** — collection with `add()`, `as_dict()`, `schemas()`,
`exclude(*names)`.

**tool_to_schema(fn)** — generates a Chat Completions tool schema from
a function's type hints and docstring using pydantic. Parameters
starting with `_` are excluded from the schema (they are
system-injected, not LLM-supplied).

**ToolContext** — injected into tools that declare a `_ctx` parameter:

```python
@dataclass
class ToolContext:
    channel: Channel | None
    tool_call_id: str
    task_queue: TaskQueue | None
```

Tools that don't declare `_ctx` work without modification.

**ToolTimeouts** — frozen value type (`default`, `per_tool`, `for_tool(name)`)
holding `tools.call_timeout` (default 600 s) and `tools.timeouts`; `None`
means no bound. `dispatch_tool_call(..., call_timeouts=...)` bounds each call
with it (see [Tool call dispatch](#tool-call-dispatch)). A tool may set an
`own_timeout` attribute (number or zero-argument callable, read per dispatch;
`resolve_own_timeout`) and the outer bound is skipped when it is not longer.

### Per-channel tool sets (`tool_policy.py`)

`channels.<id>.tools.disabled` (#149) narrows one channel on top of the
global `tools.disabled`. It is parsed into `ChannelConfig.disabled_tools`
by `load_channel_config`. The global list is already gone from the
registry, so the channel's effective set is the registry minus
`Channel.disabled_tools`: its own list unioned with every matching
transport-wide entry's (#203), so no layer can re-enable a tool another
disabled. `expand_disabled` adds coupled tools: `shell` and `task_pipeline`
go together (PRD §3.5.3), for the global list too.

- **Schemas are fixed per conversation.** When `Agent` initializes a
  channel's `ContextWindow` it resolves a `ToolView` (disabled set, tools
  dict, schema list) beside the system prompt and stores it as
  `conv.tool_view`. Every turn and every `Agent.fork` of the channel sends
  that same list object, so the prompt head is byte-identical across turns
  (principle 2) and forks keep prefix parity (see [Forks](#forks)).
  `Agent` also implements `on_compaction` and re-resolves the view there.
  Compaction already rebuilds the KV cache, so this is the only point where
  a running conversation's schemas can change, with one exception: a
  registry change.
- **Registry changes reach the next turn.** `ToolCollectionPlugin.generation`
  goes up whenever `rebuild_registry` yields different tool names or schemas
  (an MCP server reconnecting, #107/#109). `Agent` records the generation
  beside its `get_tools()` snapshot and, at the top of each queue item,
  re-reads the snapshot when the generation has moved, logging one INFO line
  naming the tools added and removed. Each conversation's view is then
  rebuilt lazily from the new set and the view's own `disabled` set (a
  reloaded `channels.<id>.tools.disabled` still waits for compaction). With
  no change nothing is rebuilt and the schema list stays the same object.
  The change costs one prefix-cache miss on that channel. Dispatch reads the
  refreshed dict, so a removed tool is refused as unknown.
- **Dispatch enforces.** Each tool task checks the name against
  `disabled_on(channel)` when it runs. That is the conversation view's set
  plus the live `Channel.disabled_tools`. A disabled call returns
  `refusal_message` as the tool result (naming the tool, channel and every
  config key that disables it, via `disabling_keys`, so a tool a wide
  entry disabled is reported under `channels.<wide key>.tools.disabled`)
  and is logged at WARNING with the same keys. Hiding is not access control: the
  model can call a tool it saw earlier or one named in injected text. This
  is a config list, not the `should_execute_tool` hook (#148).
- **Subagents inherit.** `SubagentPlugin` excludes `disabled_on(channel)`
  as well as itself from the registry it hands the subagent.
- **Hot reload.** `parse_channel_config` validates every entry before
  `apply_channel_config` applies any (and before `agent:` defaults change). For channels already registered it updates only
  `disabled_tools`, in place, so a Signal ACI channel that shares its
  alias's `ChannelConfig` follows. Transport-wide entries are replaced
  and re-matched on every live channel (`set_wide_entries`). Because
  dispatch reads the union, narrowing is enforced on the next call. The schemas, and any widening,
  wait for the next compaction or conversation start.

## Conversation Management

`context.py` provides in-memory context management. `persistence.py` handles all SQLite I/O.

**ContextWindow** — per-channel in-memory conversation context (`corvidae/context.py`).
All operations are synchronous; no database access:

- `append(message, message_type=MessageType.MESSAGE)` — appends to in-memory list with `_message_type` tag
- `build_prompt()` — returns `[system_msg, *messages]` with `_message_type` stripped.
  Every assistant `tool_calls` message is emitted contiguously with its results,
  at the arrival position of its *last* result (in place, with placeholders, if
  any call is still unanswered) — a corvidae convention, not compliance with a
  documented rule, since the OpenAI and Anthropic formats make tool-result
  adjacency normative but say nothing about a user turn arriving inside a tool
  cycle. The consequence this ordering secures: the emitted prompt never ends on
  an assistant message, which llama.cpp reads as a prefill and rejects under
  `enable_thinking`.
  A `tool` message whose call is not in the window (compaction dropped the
  call, #79) renders in place as a framed notification:
  `role: system`, content `[result of an earlier tool call that is no longer
  shown]` plus the result text, no `tool_call_id`. Strict chat templates
  reject a tool result with no preceding call. Because the frame is a
  `system` message it is then re-roled like any other (below). The window
  and `message_log` keep the stored `tool` message.
  `build_prompt(pending_tool_call_ids, mid_conversation_system_role="system")`
  also renders every `system` message after the leading system prompt
  (CONTEXT, untargeted notifications) under `mid_conversation_system_role`.
  `Agent` passes `agent.mid_conversation_system_role`. Strict chat templates
  (Qwen 3.8, Swift-1.5) reject a non-leading system message, and `user`
  keeps untrusted text out of the `system` role. Only the returned copy is
  re-roled: the window and `message_log` keep `role: system`. That matters
  because memory retrieval keys on a plain `user` tail, and because the
  rewrite must stay a pure function of each message for the prompt prefix
  to stay cache-stable.

  Every row the window can hold is renderable (PRD §3.1.18.1, #392), because
  these steps run unwrapped on every turn and an exception here is a
  deterministic per-turn failure that wedges the channel. A row that is not a
  message object is left out; a malformed `tool_calls` value (no `id`, a
  non-list value including a falsy one such as `""`, a non-dict entry) is
  filtered out by `well_formed_calls()`, and the row renders without it, or is
  left out entirely when that was its whole payload. A `tool_call_id` that is
  not a non-empty string — `message_log` never validates the field — is read
  through `result_call_id()`, so an unhashable value renders as an orphan
  result instead of raising. Each omission logs one WARNING naming the channel
  and the row (`message_log` id, or the window position when the row was never
  persisted). `snap_cut_forward`, `snap_cut_to_tool_boundary`,
  `_first_result_index` and `build_fork_prompt` read `tool_calls` through the
  same helper, so the cut snaps, the render and a fork's prefix agree on which
  calls exist.
- `corvidae.context.unanswered_user_messages(messages)` (a module function,
  not a method) — the user-role MESSAGE rows after the window's last
  assistant row, in order (all of them when there is no assistant row). A
  coalesced burst (#398) is exactly these rows when its turn reaches
  `before_agent_turn`; CONTEXT rows and notification rows are skipped by
  type and role, so hook order doesn't matter. Cognition's memory and
  critique plugins read the burst through it.
- `token_estimate()` — sums the token count of every message via tiktoken
  (`cl100k_base`), with a character-based fallback when tiktoken is
  unavailable. Special-token strings such as `<|endoftext|>` count as plain
  text, and any tiktoken failure logs a warning and falls back to the
  character estimate, so `count_tokens` never raises. Each message is sized
  by its JSON-serialized visible fields (role, content, `tool_calls`,
  `reasoning_content`, etc.), not just the `content` string, so a message
  whose size is mostly in tool-call arguments or reasoning content is still
  counted accurately. It never raises on a row's content (§3.1.18.1, #392):
  a value JSON cannot encode is counted by its `str()`, and a row that is not
  a mapping is counted by its `repr`, so both still contribute to the budget.
- `replace_with_summary(summary_msg, retain_count)` — replaces older messages
  with a summary in-memory, retaining the `retain_count` most-recent entries.
  Raises `ValueError` if `retain_count` exceeds `len(messages)`.
- `fit_to_budget(max_tokens)` — the over-budget fallback (#308, PRD §3.3.2).
  Pure and synchronous, no LLM call. Drops the oldest rows from the head of
  the in-memory window (in place; `message_log` is untouched), moving the
  cut forward past a straddling tool cycle (`snap_cut_forward`) unless that
  would drop the newest row, in which case the orphaned result renders as a
  framed notification. The newest row is always kept; if it alone exceeds
  the budget its content is truncated in memory only, with a visible
  marker. A newest row that is not a message object has no content to
  truncate, so it stands as it is and the caller reports the overage.
  Returns `(dropped, truncated)`.
- The window only grows at the tail (`append`) and truncates at the head
  (`replace_with_summary`, `fit_to_budget`). Nothing is inserted, removed or mutated
  mid-window, so there is no method for removing entries by type; stale
  `CONTEXT` entries retire by aging past the compaction boundary.

After each `conv.append()`, `Agent` fires `persist_message` (broadcast)
so persistence plugins can write to their storage. After `replace_with_summary()`,
`CompactionPlugin` fires `on_compaction` (broadcast) so persistence plugins can
update the DB boundary.

### Append-only log

Conversation history is irreplaceable and must never be deleted. The
`message_log` table in `PersistencePlugin` is append-only — no rows are
ever deleted, and their order is never rewritten. The one sanctioned
mutation is the operator-only `corvidae redact` tombstone (see
[Operator redaction](#operator-redaction-corvidae-redact)), which rewrites a
row's `message` in place but keeps the row. The DB is a complete audit log
including `reasoning_content`. This invariant applies to all persistence
paths: SQLite, JSONL, and any future storage backends.

`on_compaction` inserts a summary row whose timestamp encodes the boundary:
`oldest_retained_message.timestamp - 1e-6`. `resolve_conversation` filters with
`WHERE timestamp > summary_ts`, returning only the summary plus retained and new
messages. Old rows remain in the DB but are invisible to the working set. If the
newest summary row fails to decode, `resolve_conversation` falls back to an older
summary or the full history (see [resolve_conversation](#resolve_conversation)).

Old CONTEXT rows in the DB become invisible after the next compaction (their
timestamps fall below the boundary), exactly as they leave the in-memory window.

**Thinking token handling** — three layers:
- Display: `ThinkingPlugin.transform_display_text` calls `strip_thinking()`
  to remove `<think>` blocks before text is sent to the channel
- Persistent log: full message dict preserved (including `reasoning_content`)
- Active prompt: `ThinkingPlugin.before_agent_turn` applies the channel's
  `reasoning_history` policy to the whole in-memory window before each LLM
  call (`drop`: strip all; `exchange`: strip assistant messages before the
  last `role: user` message; `keep`: nothing). Under `drop`,
  `after_persist_assistant` also strips the just-appended message.

If `ThinkingPlugin` is not registered, `<think>` blocks pass through to the channel
and `reasoning_content` remains in the in-memory history.

### Message types

```python
class MessageType(str, enum.Enum):
    MESSAGE = "message"    # ordinary conversation turn (user or assistant)
    SUMMARY = "summary"    # compaction summary replacing older messages
    CONTEXT = "context"    # plugin-injected context (memory, notes, etc.)
    UNSENT = "unsent"      # assistant message a should_send_response gate vetoed
```

`message_type` is a persistence category, orthogonal to conversational role
(`user`, `assistant`, `tool`, `system`). `CompactionPlugin` only summarizes `MESSAGE`
entries; `SUMMARY` and `CONTEXT` rows are never summarized, but a CONTEXT row
in the range a compaction drops leaves the window with it (the retained tail
keeps its CONTEXT). `UNSENT` rows are
in the window and reload into it, but no transport sees them, including
history replay. Memory consolidation's dialog (what a record's range sentence and degraded
transcript are built from) reads `MESSAGE` rows only, so it never counts an
`UNSENT` row as said; compaction's summary and a catch-up view render the
window or its rebuilt rows, where the row's not-sent marker sits beside it
(see [The output gate](#the-output-gate)).

In-memory message dicts carry a `_message_type` metadata key (set by `append()`).
`build_prompt()` strips it before returning messages to the LLM.

The `before_agent_turn` hook gives plugins a chance to inject entries (e.g., `CONTEXT`
rows) into the conversation before each LLM call. `Agent` fires
`persist_message` for each injected message after the hook returns.

### Database schema

```sql
CREATE TABLE message_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id TEXT NOT NULL,
    message TEXT NOT NULL,      -- JSON
    timestamp REAL NOT NULL,
    message_type TEXT NOT NULL DEFAULT 'message'
);
CREATE INDEX idx_log_channel ON message_log(channel_id, timestamp);
```

Plugins keep their own tables on the same connection, created lazily on
first use because the `on_start` broadcast is concurrent: `memory` and
friends ([MemoryPlugin](cognition.md#memoryplugin)), `exchange_log` (OutcomeLogPlugin),
`usage_log` (UsageLogPlugin), and `schedule`
([SchedulerPlugin](#schedulerplugin)).

**Connections to the database file.** Most plugins write on the one
persistence connection (`PersistencePlugin.db`). The others:

| Connection | Opened by | Writes |
|---|---|---|
| persistence (`PersistencePlugin.db`) | persistence `on_start` | `message_log`, and every other plugin's tables (scheduler, usage_log, retrieval_log, retrieval_candidate, goals, ...) |
| memory write (`MemoryPlugin._write_db`) | memory `on_start`, after its schema exists | memory records only: consolidation (`memory`, `memory_vec`, `memory_fts` via trigger, `memory_prompt`, `consolidation_watermark`) and `write_record`, each one `BEGIN IMMEDIATE` transaction |
| appraisal probe, critique probe | each plugin's `on_start` | read only (`message_fts` probes) |

Only the memory write connection is isolated for atomicity: a commit on
the persistence connection can't land inside a memory transaction, and a
rollback there can't discard another plugin's pending row. WAL lets
readers on the persistence connection see a committed record at once.

## Agent

`agent.py` — the central plugin. Implements `on_start`, `on_message`,
`on_notify`, `on_stop`. Declares `depends_on = {"registry", "task", "llm", "tools"}`
so `validate_dependencies(pm)` verifies those dependencies are registered at startup.
Retrieves `ChannelRegistry` during `on_start` via
`get_dependency(pm, "registry", ChannelRegistry)` from `hooks.py`.

DB lifecycle (open/close) is delegated to `PersistencePlugin`. Conversation
initialization is handled directly in `Agent._process_queue_item`: a
`ContextWindow` is created, then `resolve_conversation` (firstresult=True) is
called so persistence plugins can populate history. `Agent` reads `_chars_per_token`
and `_base_dir` from config in `_start_plugin`.

The LLM client is borrowed from `LLMPlugin` (`get_dependency(pm, "llm", LLMPlugin)`)
in `_start_plugin`. The tool registry and `max_result_chars` are borrowed from
`ToolCollectionPlugin` (`get_dependency(pm, "tools", ToolCollectionPlugin)`).
Agent does not own either lifecycle.

`Agent.started` is True from the end of `on_start` until `on_stop`.
`Runtime.start()` broadcasts `on_start` to every plugin concurrently and
calls `Agent.on_start` only afterwards, so a producer whose background
task begins in its own `on_start` (`SchedulerPlugin`) waits for this flag
before calling `on_notify`; otherwise the queue consumer could run before
`Agent` has a registry and client.

Inbound messages are held, not dropped (#71). A transport can deliver from
its own `on_start`, before step 12 starts the Agent; `Agent.on_message`
checks the trace id first (an empty one raises at once), then awaits an
internal started event until the Agent has started (the gate is `_started`, not
the client: a config reload can set the client earlier). Once the Agent starts,
the held call proceeds normally: the gate, `on_message_admitted` and the
enqueue all happen before `on_message` returns, so a transport that frees
resources in a `finally` around `on_message` (the Signal and Discord typing
slots) is unaffected. Held calls from one channel resume in arrival order.
If the Agent stops before it starts (`Agent.on_stop`, or
`Runtime._cleanup_failed_start` through `Agent.release_held_messages`),
every held call returns without admission and logs one WARNING naming the
channel and trace id. There is no cap on held messages: the window is
bounded by startup.

`Agent.queues` is a public `dict[str, SerialQueue]` (keyed by channel ID).
Queues added after startup are visible to other plugins that hold a reference
to this dict because the reference is live.

### Message processing

A transport mints the trace id for each inbound user message
(`corvidae.trace.mint_trace_id()`, a stdlib-only leaf module;
`corvidae.agent` re-exports it) and passes it to `on_message`. `Agent.on_message`
uses it verbatim for the gate, the admitted/rejected hooks and the
`QueueItem`, so `on_message_persisted` carries the same id. It never mints
for a USER item: a falsy id raises `ValueError`, checked before the
readiness check so a mis-wired transport fails even while the agent starts.
Omitting the argument entirely fails at the hook call (`HookCallError`).
Notification-born items with no inherited `annotations["trace_id"]` are the
exception: the Agent mints their id at dequeue.

`on_message` broadcasts `should_process_message` and passes the result
list to `resolve_reject_wins`. If the resolved result is `False` (identity
check, not falsiness), the message is dropped. Any other result (`True`
or `None`) proceeds to enqueue.

`on_message` and `on_notify` both enqueue a `QueueItem` onto a
per-channel `SerialQueue`. The queue's consumer calls
`_process_queue_item`, which delegates to four helper methods. Before them
comes the coalescing dequeue described next.

**Coalescing (#398, PRD §3.1.19).** Messages that queue up behind a turn
coalesce into one turn when the queue reaches them. At dequeue, a USER item
that is not a retry calls `SerialQueue.take_while` with the predicate
"USER role and not `retry`". That removes the items directly behind it for
as long as they match and returns them in queue order; the first item that
fails the predicate (a tool result, a notification, a retry copy) stays
queued, unmoved, so those keep their place. When the call returns
anything, the last item taken owns the turn, and the item that was
dequeued plus the other taken items are *absorbed*: `absorbed = [item,
*following[:-1]]`, `item = following[-1]`. One INFO log names the channel,
the burst size and the owning trace id. Trace
resolution, `before_agent_turn`, the model call, the gates,
`on_agent_response` and every usage row use the owner's trace; absorbed
traces see admission and persistence only. The queue is never touched while
a turn runs, so a reply in flight is not interrupted; coalescing happens
only when the consumer asks for its next item.

`take_while` is synchronous (no awaits), so nothing can enqueue between its
peek and its take. It removes each item with `get_nowait()` and calls
`task_done()` for it at once. The consumer's own item is not marked done
until the turn ends, so `SerialQueue.drain()` (`Queue.join`) still waits for
the whole burst's turn, and `_note_settled` runs once, at the end of the
turn, after which the queue holds none of the burst's items; no idle signal
fires between the dequeue and the end of the turn. Peeking reads
`asyncio.Queue._queue`, the deque that asyncio's own `LifoQueue` and
`PriorityQueue` override. `shortcut: relies on a CPython-private
attribute; replace the asyncio.Queue with an owned deque plus an Event if
it ever moves.`

Inside `_process_queue_item_attributed`, steps 1 to 4 below run once. Step 5
(append, `persist_message`, then `on_message_persisted`, one helper,
`_append_and_persist_message`, after `_build_conversation_message` builds
the item's message) runs once per absorbed item, in arrival
order and each under its own item's trace, then for the owner as usual.
Steps 5b onward run once, for the owner. The result: every message is its
own `message_log` row with its own persisted event, all before the model
call; `turn_counter` resets once, on the owner, so a burst is one exchange
for `max_turns`; a burst that arrives mid-tool-cycle is charged to that
cycle and leaves the pending set alone. A retry (below) copies the owner;
the absorbed rows are already in the window and the log, and a retry skips
the persist step, so nothing is persisted twice. A coalesced turn that
fails before the model call (conversation init raising, say) drops every
absorbed item, not one.

`on_agent_response`'s `request_text` is the text of the message that owns
the turn: for a coalesced burst, the last message. An absorbed trace's
`exchange_log` row (when the outcome log is loaded) is written at
persistence and no response ever completes it, because no reply carries
that trace.

1. **`_build_conversation_message`** — converts a `QueueItem` into a
   conversation message dict and `request_text`. Returns `None` on
   unrecognized role (logged, item dropped).
2. When `channel.conversation is None`, creates a `ContextWindow`, calls
   `resolve_conversation` (firstresult=True) to populate history, and assigns
   `channel.conversation`
3. Resets `turn_counter` to 0 when the item originates a new trace
   (`item.originating`, set during trace resolution: every USER
   item, and a notification with no inherited `annotations["trace_id"]`)
   **and** no tool cycle is in flight (`channel.pending_tool_call_ids` is
   empty). An item arriving mid-cycle is charged to the running cycle
   instead (R4), so a chatty channel cannot extend `max_turns`. Keying on
   `originating` rather than the USER role makes `max_turns` a
   per-exchange budget for clock-driven exchanges too: a channel fed only
   by scheduler notifications would otherwise exhaust it after ten
   responses and have every later tool call suppressed.
4. Resolves channel config (including `max_turns`)
5. Appends the inbound message to conversation; fires `persist_message`
   (broadcast: every implementation runs; `PersistencePlugin`'s rowid is
   resolved with `resolve_single_result` and stored as `_db_id`)
5b. **Tool result batching** — if the item is a tool-result notification
    (`role=NOTIFICATION` with `tool_call_id` set), removes the ID from
    `channel.pending_tool_call_ids`. If the set is non-empty after the
    removal, returns immediately (more results still pending). The LLM call
    is deferred until the last result clears the set. User messages can
    interleave during this window — they don't clear the pending set.
6. Calls `resolve_compaction` (`firstresult=True`): the first non-None
   return stops the chain, so a default-priority replacement runs before
   the `trylast` `CompactionPlugin`; the return value is not used. A
   `tryfirst` handler may return None, which doesn't stop the chain.
   Right after the hook, whatever it did or raised, the kernel calls
   `conv.fit_to_budget(max_context_tokens)` (also when no compaction
   plugin is loaded) and logs one WARNING if it dropped rows, truncated
   the newest row, or the window is still over budget
7. Calls `before_agent_turn` hook (broadcast); fires `persist_message`
   for any messages injected by the hook
8. **`_run_turn`** — calls `run_agent_turn` (single LLM invocation).
   On exception, calls `resolve_llm_error` (`firstresult=True`; if the hook
   itself raises, the failure is logged at WARNING and the default message
   is used), sends the error message to the channel, and returns `None` to abort
   processing. `LLMPlugin` implements it `trylast`: for an unloaded
   backend model it returns a message naming the model; for any non-2xx
   `LLMHTTPError` it names the status, an actionable hint (API key,
   quota, endpoint/model) and a short provider body; for connection and
   timeout failures it points at `llm.main.base_url`. It defers (`None`)
   on every other error, so the generic message still applies.
   **Retry when the backend is unreachable (#281).** When the failure
   satisfies `is_backend_unreachable` (the client has already spent its own
   `max_retries` and `model_not_loaded_wait`), the Agent also schedules a
   retry on the kernel clock, before the error gate so a vetoed fallback
   cannot cancel it. The fallback goes out once, on the first failure. The
   retry is a copy of the failed `QueueItem` (`retry=True`,
   `originating=False`, same trace id) enqueued on the channel's serial queue
   after `min(5 * 2**n, 60)` seconds (`TURN_RETRY_BASE_DELAY`,
   `TURN_RETRY_MAX_DELAY`: 5, 10, 20, 40, 60, 60, ... forever). Each
   scheduling logs one WARNING naming the channel and the delay. A retry that
   fails again for the same reason reschedules without a reply; any other failure
   ends the retries with a WARNING and no reply (the user can resend). One
   retry is pending per channel: any successful turn on the channel drops it
   (a newer message's turn already contains the earlier one), a newer failed
   turn replaces it, and a retry item that is no longer the pending one is
   skipped when dequeued. `Agent.on_stop` cancels pending retries. Without a
   registered `clock` the retry is skipped with a WARNING. Out of scope:
   background stages (they retry on their own triggers) and retries across a
   restart.
   **Cut-off replies (#355).** A result flagged `reasoning_unfinished` with
   no tool calls logs a WARNING (channel, completion tokens, last 200
   characters of reasoning) and is re-run once with the same messages (the
   failed message is dropped from the list first). A second cut-off raises
   `GenerationCutOff` inside the same `try`, so it takes the LLM-error path
   above (error hook, error gate, error send; no turn retry). The user-facing
   cause names the channel and says the reply was cut off mid-reasoning twice
   and to resend or rephrase.
8b. **The final send gate.** When the result's tool calls will not
    dispatch (no tool calls, or `turn_counter >= max_turns`), resolves the
    display text once (the max-turns fallback on that branch) and fires
    `should_send_response(emission="final")` over it. On the max-turns
    branch, `tool_calls` is stripped from the assistant message before it
    is appended, so the row and the window copy both lose it, whatever the
    verdict. A result with no tool calls whose display text is empty or
    whitespace is left unsent as a decline (`UNSENT`, final marker) without
    firing the gate (#355).
9. Appends the assistant message to conversation (as `UNSENT` after a
   final veto, else `MESSAGE`); fires `persist_message`. Then sends
   `OutboundThinking` for `reasoning_content`, unless the final was vetoed
   (the reasoning goes with it) or `should_send_response(emission="thinking")`
   vetoes it
10. Calls `after_persist_assistant` hook (broadcast); `ThinkingPlugin`
    uses this to strip `reasoning_content` from the in-memory copy. Then
    appends a not-sent marker for each vetoed emission
11. **`_handle_response`** — decision point:
    - **Tool calls, under limit** (`turn_counter < max_turns`):
      increment counter, send the intermediate text as `OutboundProgress`
      unless `should_send_response(emission="progress")` vetoes it (then
      a marker is appended instead), dispatch tool calls as Tasks, return
      without sending a response
    - **Tool calls, at limit**: suppresses tool dispatch. Uses the display
      text step 8b resolved (`MAX_TURNS_FALLBACK_MESSAGE` as fallback),
      fires `on_agent_response`, sends text response unless the final was
      vetoed.
    - **No tool calls**: increments counter, uses the display text step
      8b resolved, fires `on_agent_response`, sends text response unless
      the final was vetoed

Step 7's `conv.build_prompt(...)` never returns a prompt ending on an
assistant message. Three facts compose to guarantee it: step 4 only ever
appends a message produced by `_build_conversation_message`, which emits
`role: "user"`, `"tool"`, or `"system"` and never `"assistant"`; compaction's
`replace_with_summary` always retains at least one message
(`retain_count >= 1`), so it cannot strand step 4's append; and
`build_prompt`'s deferral never leaves an assistant `tool_calls` message
last — its call+results block always ends on a `tool` message, real or
placeholder. This matters because llama.cpp reads a trailing assistant
message as a prefill and rejects it under `enable_thinking`.

The Agent also feeds the kernel idle tracker (see [Clock and idle
tracker](#clock-and-idle-tracker-clockpy-idlepy)). It calls
`IdleTracker.note_activity(channel_id)` when an item is enqueued, and
`note_settled(channel.id)` when an item finishes and nothing remains on the
channel: its queue is empty and it has no pending tool calls (subagent
tasks included; their results arrive as notifications that settle the
channel again). The Agent only pushes; the tracker never reads the
queues. Without an `"idle"` plugin the Agent logs a warning and idle
signals are off.

**`_resolve_display_text`** calls `transform_display_text`
(`firstresult=True` wrapper chain). If the hook returns a value,
that value is used; otherwise falls back to `result.text`. When a
`fallback` is provided and the resolved text is falsy, the fallback is
returned instead. Hook exceptions are caught, logged, and the input text
is returned.

### The output gate

`should_send_response(channel, text, emission, trace_id, annotations)` is the
outbound mirror of `should_process_message`: reject-wins over every
implementation, any `False` vetoes. It fires at four sites, in two modes.

**The persistence-controlling firing** (`emission="final"`, step 8b) runs
between generation and persistence, and only for results whose tool calls
will not dispatch: a text result, or the max-turns branch. Never for a
result whose calls dispatch: hiding that assistant message would orphan its
`role: "tool"` rows on reload, which OpenAI-compatible servers reject. A
veto:

- persists the assistant message as `MessageType.UNSENT`;
- skips the `OutboundMessage` send;
- still fires `on_agent_response` (the agent thought it; critique can
  still object);
- appends a not-sent marker to the window.

An empty or whitespace-only final with no tool calls takes the same path
without the gate firing: the model declined to speak, and there is nothing to
ask a gate about.

**An unsent final releases transport state.** An unsent final never
reaches `send_message`, where transports end the turn (Discord and Signal
give back every outstanding typing slot, ACP completes the prompt, the CLI
reprints its prompt). `on_agent_response` runs after the row is appended and tagged, so
those four transports implement it, guard it with `owns()`, and release the
same state when `final_was_unsent(channel)` (newest assistant row is
`UNSENT`) is true. A sent final releases only in `send_message`, so each
final releases once. This runs whether or not any plugin vetoed, so an empty
final ends the turn too. Web needs nothing: it clears "working" on the idle
tracker's settle signal. The rule for plugin authors is in
[plugin-guide.md](plugin-guide.md#transports-with-per-turn-state).

The max-turns branch strips `tool_calls` from both the persisted row and
the window copy regardless of the verdict: those calls never dispatch, so a
row that kept them would dangle on reload, and stripping only the row would
leave the live and reloaded windows disagreeing.

**Per-emission firings** at the progress, thinking and error-fallback
sends suppress that emission only. The assistant row stays `MESSAGE`, keeps
its `tool_calls`, and tool dispatch proceeds. A progress or thinking veto
appends a marker; an error veto (in `_run_turn`, where no assistant message
exists) suppresses the send and nothing else. A vetoed final takes its
reasoning with it: the thinking send is skipped without consulting the gate.

**The not-sent marker** is a `CONTEXT` system row the Agent appends and
persists itself (`corvidae.context.unsent_marker_text`), right after the
assistant row, so it reloads beside it. It opens `[not sent: <emission>]`
and tells the model the channel didn't see that text and not to cite it as
said. Only the progress marker quotes its text (40 characters): `build_prompt`
moves a tool-calls block to its last result, so a progress marker renders
before the block it describes and needs the excerpt to say which text it
means. The final marker sits right below its row; the thinking marker quotes
nothing, so reasoning never outlives `reasoning_history: drop` inside one.

**Transports never see a vetoed emission, including on replay.** The web
transport's history skips `UNSENT` rows and drops the progress or thinking
frame of an assistant row followed by a progress or thinking marker. ACP's
`load_session` replay skips `UNSENT` rows and an assistant row followed by a
progress marker (it doesn't replay reasoning). `unsent_marker_emission(content)`
is the shared parser.

**Failure modes.** `_gate_send` catches any exception from the hook, logs it
at WARNING with the emission and channel, and sends: a broken gate never
silences the agent. With no implementation registered (the kernel ships
none), every emission is sent; the only behavioral change from before the
hook is the max-turns `tool_calls` strip. `annotations` is the item's
annotations dict; origin is not a kernel concept, so a gate that needs it reads
`annotations["origin"]` (stamped by producers such as the scheduler) the way
cognition's `GatePlugin` does ([cognition.md](cognition.md#gateplugin)).

### Tool call dispatch

`_dispatch_tool_calls` records all tool call IDs in
`channel.pending_tool_call_ids`, then creates a `Task` for each tool call
and enqueues it on the `TaskQueue` (accessed via
`getattr(pm.get_plugin("task"), "task_queue", None)`). Each task's `work`
closure calls `dispatch_tool_call()` from `tool.py`, which handles JSON
parsing, unknown-tool detection, invocation, error wrapping, logging, and
the `transform_tool_result` hook. The closure returns `result.content`; the
TaskPlugin delivers it via `on_notify`.

**Call bound.** `dispatch_tool_call` wraps the invocation (`backend.run` or
`execute_tool_call`, not the hook) in `asyncio.timeout`, using a
`ToolTimeouts` (`tools.call_timeout`, default 600 s, plus `tools.timeouts`
per-tool overrides; `ToolCollectionPlugin.call_timeouts`, copied to the Agent
and passed to the subagent loop). The bound is skipped when the tool declares
an `own_timeout` (a number or zero-argument callable on the registered
callable, read at each dispatch) that is not longer: `shell`, `web_fetch` and
MCP tools declare theirs. On expiry the tool is cancelled and the result is
`Error: tool '<name>' timed out after <n>s` (flagged as an error), so the
turn continues and the task worker is freed; `transform_tool_result` still
fires. One `WARNING` names the tool, channel, call id and bound. A tool that
swallows the cancellation has its late result discarded and a second kind of
`WARNING` says so. A tool's own `TimeoutError` before the deadline is an
ordinary failure. An unreadable `own_timeout` falls back to the outer bound
with one `WARNING` per tool per process.

`TaskPlugin` is declared in `Agent.depends_on`. Without a task queue,
tool dispatch logs an error naming the channel and the requested tools (with
the remedy: the `task` plugin is registered, since startup requires it, but
has no queue, so check that `TaskPlugin.on_start` ran and that the plugin
registered as `task` is `TaskPlugin`) and sends an error result for each call
through `on_notify`. If that broadcast raises for a call, dispatch logs a
WARNING with the channel, tool call id and tool name, and falls back to
`Agent.deliver_tool_error` (#110). One call's failure does not stop the rest
of the batch.

Closure capture uses default argument binding to avoid the
loop-variable capture bug.

### Notification-driven continuation

When a tool task completes, `TaskPlugin` fires `on_notify` with the
result and `tool_call_id`. `Agent.on_notify` enqueues a
notification `QueueItem`. The next `_process_queue_item` call uses
`_build_conversation_message` to format it as a `role: "tool"` message
(if `tool_call_id` is set) or a system message, appends it to
conversation, and calls `_run_turn` again. The system message is stored
as such and sent as `agent.mid_conversation_system_role` (see
`build_prompt` above).

If the `on_notify` broadcast raises, `TaskPlugin` logs a WARNING with the
channel, task id and tool call id and, for a tool task, calls
`Agent.deliver_tool_error`, which queues an error `role: "tool"` result
straight onto the channel's serial queue. Without it the id would stay in
`pending_tool_call_ids` and the tool cycle would never finish (#173). The
broadcast runs every hookimpl via `asyncio.gather`, so a third-party
hookimpl can raise after `Agent.on_notify` already queued the real result.
The Agent therefore records each queued tool result as
`(channel_id, tool_call_id)` in `_queued_tool_results` until the item is
processed (dropped in the same step that discards the id from
`pending_tool_call_ids`), and `deliver_tool_error` does nothing if the
result is already queued or no longer pending. The model never sees two
results for one call. A task lost before it reaches `_on_task_complete`
is not covered. The no-task-queue path in `_dispatch_tool_calls` uses the
same fallback and guards.

Multi-turn reasoning emerges from this cycle without blocking the
channel queue.

### QueueItem

```python
class QueueItemRole(Enum):
    USER = "user"
    NOTIFICATION = "notification"

@dataclass
class QueueItem:
    role: QueueItemRole
    content: str
    channel: Channel
    sender: str | None = None
    source: str | None = None
    tool_call_id: str | None = None
    annotations: dict = field(default_factory=dict)
    trace_id: str | None = None
    originating: bool = False
    retry: bool = False
    metadata: ItemMetadata | None = None
```

`metadata` is what the kernel itself observed about the item
(`corvidae/provenance.py`); see [Provenance](#provenance-header-and-fence).
`source` is a log label only.

`retry` marks the re-run of a turn whose LLM call could not reach the backend
(#281). The item's message is already in the window and the log, so the
retry skips the `turn_counter` reset, the append/persist,
`on_message_persisted`, tool-result batching, compaction and
`before_agent_turn` and goes straight to the LLM call; the
`max_context_tokens` ceiling (`fit_to_budget`, #308) still runs first.

### Provenance header and fence

`ItemMetadata` is a frozen dataclass of what the kernel observed, never
what a plugin or transport asserted. Which fields apply follows the row's
role:

- user: `sender`, `sender_name` (from `channel.sender_names` at admission)
  `channel_name` and `addressed` (from `channel.addressed` at admission), set
  in `on_message`;
- notification: `origin_channel` and `origin_trace`, read from the
  attribution of the code that called `on_notify`; and, for a notification
  the spill service cut, `cut_shown`, `cut_total`, `spill_path`,
  `spill_chars` and `spill_failed` (see [Notification
  spill](#notification-spill-spillpy));
- tool result: `tool` and `target`, the dispatched call's name and the first
  argument naming another registered channel, set by `on_notify` with a
  `tool_call_id` and by `deliver_tool_error` (the name from the dispatch
  record when there is one, else the call's row; the target from the call's
  row in the window); and `untrusted`, True when the tool declared its
  output untrusted as of dispatch (see [Untrusted tool
  results](#untrusted-tool-results)), otherwise absent.

`metadata` is never passed as a hook argument; the kernel fills it from what
it observes (the transport's sender and channel, the tool-call row in the window, the
attribution context). `_build_conversation_message` stores
it on the row under `corvidae_provenance` (`encode`: set fields only, plus
an `annotations` entry holding only the keys a plugin declared in
`model_visible_annotations`; the Agent reads the declarations once in
`_start_plugin`). Other annotations are never persisted into the row or
shown to the model. `decode` checks each field's type (strings, booleans for
`addressed`, `spill_failed` and `untrusted`, integers for the three counts; a boolean is
not an integer) and rejects the whole record on a mismatch. A stored record that does not decode renders as a row
without provenance and logs a warning naming the row.

`build_prompt` renders it through `provenance.frame` on every build; the
stored row is never changed. The header is a `[metadata: ...]` line built
in code:

- A person's message with a sender or channel name, or a known `addressed`,
  gets a separate system message before it: `[metadata: sender: <name> (id
  <id>); channel: <name> (id <channel id>); addressed: yes|no]`, with
  `direct message` for a DM scope. The `addressed` clause is absent when the
  transport did not say. A burst from
  one sender carries one header: a row's header is dropped when it equals
  the one computed for the row before it. The message text is unfenced.
- A notification gets `[metadata: notification from your own activity on
  <channel> (trace <id>); not from this channel's human]`.
- A cut notification gets a second line, the cut line, in one of four
  wordings: `[metadata: cut to the first N of M characters; the full text is
  in <path>]`; `[metadata: cut to the first N of M characters; the first S
  characters are in <path>]` when the file holds less than the whole text
  (the spill cap); `[metadata: cut to the first N of M characters; the full
  text was not saved]` when the write failed; and `[metadata: cut to the
  first N of M characters]` when the channel has no tool that can read the
  file. The kernel builds it from the stored fields, so no plugin writes or
  forges it; the path goes through `clean_value`.
- A cross-channel tool result gets `[metadata: result of your own <tool>
  action on <target>; ...]`, in the tool message itself so call and result
  stay adjacent.
- A tool result with `untrusted` set gets `[metadata: result of your own
  <tool> call <call id>; untrusted outside content, not from this channel's
  human]`, and when it also has a `target`, `[metadata: result of your own
  <tool> call <call id> on <target>; untrusted outside content, not from
  this channel's human]`. The call id is read from the stored row, so a
  reload renders the same bytes; a row with no string id renders `?`. Tool,
  call id and target go through `clean_value`, which matters because call ids
  come from model output. A result with a `target` and no `untrusted` keeps
  the cross-channel wording above.
- Declared annotations follow as `[annotations: k=v, ...]`.

Notification payloads, cross-channel results (tool results with a
`target`) and untrusted tool results (`untrusted` set) are fenced; other
tool results are not:
`[payload <nonce> begins]` ... `[payload <nonce> ends]`. The nonce is the
first 16 hex digits of an HMAC-SHA256 of `message_log:<row id>` keyed by
the fence key, so it is stable across renders and restarts and differs per
row; a row with no id yet uses its text. The key is 32 random bytes in
`provenance.key` in the base directory, created mode
0600; an unreadable or short file logs a warning and degrades to a key
that lasts until restart. `ContextWindow.fence_key` carries it.

Sanitizing happens at render time only (`clean_value` for header values:
NFKC, no control, format or bracket characters, one line, names cut at 64
characters and ids never cut; `clean_payload` for fenced text: control and
format characters removed, newline and tab kept). The log keeps the
original text. A cross-channel result with no content fences an empty payload. The
fenced text is the stored `content` after `clean_payload` (the stored row
is unchanged); an orphaned result
(`ContextWindow.build_prompt` rendering it as a `system` message, #79) gets
the same header and fence inside the orphan label.

#### Untrusted tool results

A tool plugin declares that a tool returns outside content by setting
`returns_untrusted = True` on the registered callable, the form
`reads_spill_files` and `own_timeout` take; `tool.declares_untrusted_output`
reads it and is strict (`is True`, so a truthy non-bool does not count). It
is per callable and not a plugin class attribute because MCP tools are
created per server at connect time and no class-level list could name them.
Declared today: `read_file` (so a spill file read back is fenced, and so is
any other file), `web_fetch`, `web_search`, and every tool `mcp_client`
builds. `shell` and `task_pipeline` are not declared (#580).

The declaration is evaluated when the call is dispatched, not when the
result arrives. `_dispatch_tool_calls` writes `Agent._dispatched_calls`,
`(channel id, call id)` -> `(tool name, untrusted)`, for every call before
the task is enqueued (and before the no-task-queue path's `on_notify`).
Two things make the record necessary. Compaction can drop the call row
while the tool runs (a user message can interleave and trigger it), so
the window no longer names the call when the result arrives; and a registry
change between dispatch and result (an MCP server dropping, a rebuild) must
not unfence a call that was declared when dispatched. `_tool_result_metadata`
takes the name and `untrusted` from the record, then the window's call row
for `target`; with the call row gone it returns the record's name, `untrusted`
and no target. The record is removed in `_process_queue_item` where the
`_queued_tool_results` entry is discarded, after `on_notify` computed the
metadata. A second delivery for the same id (a subagent's confirmation, then
its result) processed after the first finds no record: its name comes from the window and it is not
stamped, so the rule is one: declared at dispatch, or not fenced. Kernel error
text for a declared tool (refusal, timeout, task unavailable) is stamped and
fenced too.

The flag is stamped into the row's provenance record when the result is
built, so a later change to a declaration does not re-render history. The
header names no spill source: it says the content is untrusted, not that it
came from a notification's spill file (#576). The estimator counts the stored
row, not the added framing, so a fenced row costs more tokens than it is
estimated at (#577).

Items the kernel builds carry no attribution of their own: the queue
consumer runs in an empty `contextvars.Context`, so a notification
enqueued first from another channel's turn does not inherit that turn's
attribution.

## Notification spill (`spill.py`)

A notification too long to show inline is cut, and its full text is written
once to a file the agent reads on demand. `NotificationSpill` is a kernel
service, not an entry-point plugin: `Runtime.start()` registers it as
`"spill"` beside `"registry"`, `"clock"` and `"idle"`, and listing it in
`plugins.disabled` fails dependency validation. It depends on `"clock"` and
`"tools"`.

The plugin that produces the text chooses the limit; the service applies it.
A plugin that cuts calls `spill.notify(channels, source=..., text=...,
annotations=..., limit=...)` instead of firing `on_notify` itself
(`webhook_notify` passes `max_payload_chars`). `notify` returns the file path,
or `None` when nothing was written.

- **Text that fits** (`len(text) <= limit`) goes to each channel through
  `on_notify` unchanged, with no cut recorded.
- **Longer text** is cut to `limit` characters. The service decides per
  channel whether the pointer is offered: only where an enabled tool declares
  `reads_spill_files = True` (`read_file` and `shell` do) on that channel,
  after `tool_policy`'s per-channel disabled set. If any channel qualifies,
  one file holds the text, cut to `notification_spill_max_chars`, and every
  channel's row records its own cut: a pointer where it can read the file,
  "not saved" where it could have but the write failed, neither where it
  cannot read it. With no tool registry yet (a delivery before the tool
  collection is built), no channel counts as a reader.
- **The file** is `<uuid4 hex>.txt` in the spill directory, created
  exclusively with mode 0600 and written as UTF-8 with replacement, so a lone
  surrogate costs a character's worth of `?` instead of the delivery and the
  header's counts hold. The name carries nothing the sender controls. The
  directory is created on demand, so removing it while the daemon runs costs
  nothing.
- **A failed write** (`OSError`) is logged at WARNING naming the directory
  and the remedy; the notification is delivered cut, with the "not saved"
  wording. Delivery is never lost to the spill.

The service tells `Agent.on_notify` what it did through a context variable,
`current_cut()`, set around each channel's `on_notify` and reset after it.
`Agent._notification_metadata` copies it into `ItemMetadata`, so the header
comes from the stored row and a plugin cannot assert it. An `on_notify`
hookimpl must not spawn a task or fire `on_notify` again, or it carries the
cut into that context; `Agent.on_notify` is the only one, and does neither.

**Cleanup.** `on_start` sweeps the directory, then arms an hourly
`Clock.call_later` timer (`notification-spill-cleanup`) that re-arms itself
before it sweeps, so a failed sweep never stops it. The sweep deletes only
regular files named `<32 hex>.txt` whose mtime is older than
`notification_spill_max_age_days`; a missing directory is not an error, and an
undeletable file logs a WARNING and the sweep goes on; an unlistable directory
logs a WARNING and that sweep stops, but the timer goes on.
The age is the only policy: nothing bounds the directory's total size, so its
worst case is the cap times the notification rate times the age.

**Reading it back.** `read_file` takes any path, so the agent pages through a
spill file with `offset` and `length` (or `start_line` and `end_line`). `read_file` asks the service first: a
missing path inside the spill directory (a file the sweep deleted, or a name
that never existed) returns an error saying it was a spill file, that spill
files are deleted after the configured days, and which key sets that. The
check is lexical (`os.path.abspath`), so a path with a NUL byte cannot raise.

**Configuration** (`daemon`, applied at restart):

- `notification_spill_dir`, default `state/notification-spill`, resolved
  against the config directory. It is refused at startup when `clean_value`
  would alter the resolved path, because the header renders the path through
  `clean_value` and a pointer `read_file` cannot open is worse than none.
- `notification_spill_max_chars`, default 1000000: a positive integer.
- `notification_spill_max_age_days`, default 7: a positive number (booleans
  and NaN are rejected).

A bad value raises `ValueError` naming the key, the value and what is
accepted. See [configuration.md](configuration.md#daemon--daemon-level-settings).

Logging: INFO when a cut notification is spilled (characters, path, channel
counts) and when a sweep deletes files; WARNING for a failed write, listing or
delete.

## PersistencePlugin

`persistence.py` — manages the SQLite database lifecycle and conversation
persistence. Registered as the `"persistence"` entry point. Declares
`depends_on = set()`.

`PersistencePlugin.on_start` reads `daemon.session_db` from config (default
`"sessions.db"`), opens the database with `aiosqlite.connect`, and runs schema
creation. `on_stop` closes the connection.

`PersistencePlugin.db` is public for test injection: setting `persistence.db`
before `on_start` causes the `if self.db is None:` guard to skip the open.

### resolve_conversation

Implements `resolve_conversation` (firstresult=True). Called by `Agent` when
`channel.conversation is None`. Queries `message_log` for the channel, applying
the timestamp filter so only the compaction boundary and newer rows are returned.
Returns a list of tagged message dicts (with `_message_type` set). Logs an INFO
event when history is loaded.

A corrupt summary row degrades the load instead of aborting it. A summary row
that fails to decode (invalid JSON, invalid UTF-8, or JSON that is not an
object) is skipped with a WARNING naming its row id and channel, and the
next-newest summary is tried with its own timestamp boundary. Because
`message_log` is append-only, an older summary plus the rows after it still
covers the whole conversation, only less compressed. If no summary decodes, the
query widens to the full non-summary history. Either fallback can return more
history than the last compaction kept; the `resolve_compaction` pass before each
LLM call can compact it again once it crosses the compaction threshold.

A corrupt ordinary row is skipped the same way, on both the summary-boundary
branch and the no-summary branch: a non-summary row whose message is invalid
JSON or invalid UTF-8, decodes to something other than a JSON object, or carries
a `message_type` outside `MessageType` is dropped with a WARNING naming its row
id and channel, and the rows on either side of it still load in order. Every
query selects `CAST(message AS BLOB)`, so invalid UTF-8 is caught while
decoding in Python instead of raising inside the sqlite3 fetch, where the cursor
cannot step past the row.

Every other reader of `message_log.message` shares this decoder
(`decode_message_row` and `parse_message_type` in `corvidae/persistence.py`)
and the same `CAST(message AS BLOB)` select, and skips a corrupt row with a
WARNING naming its row id and channel (never its bytes) and saying to inspect
or redact the row (#232):

- **Web history replay** (`_fetch_history_rows` / `_rows_to_frames`): a corrupt
  row, or one with an unknown `message_type`, among the newest rows is skipped
  and the other rows still replay; it no longer blanks the channel's history.
- **Memory** (consolidation, `recall_raw`, the `message_fts` backfill): see
  [cognition.md](cognition.md#corrupt-message_log-rows).
- **Redact** (below): the tool that fixes a corrupt row works on it.

### persist_message

Implements `persist_message` (broadcast). Called after every `conv.append()`.
Strips `_message_type` from the message dict before inserting into `message_log`.

### on_compaction

Implements `on_compaction` (broadcast). Called after `replace_with_summary()`.
Determines the timestamp boundary from the most-recent non-retained row, inserts
the summary row with `timestamp = boundary - 1e-6`, and commits.

**Graceful degradation:** without `PersistencePlugin`, `resolve_conversation` returns
no history and conversation events are not persisted.

## Operator redaction (`corvidae redact`)

`corvidae/commands/redact.py` — operator-only CLI for permanent content
removal. Never a registered agent tool; never reachable from any hook the
model can influence.

**Forms:**

```
corvidae redact --db sessions.db message <id> [<id2>...]   # specific message_log rows
corvidae redact --db sessions.db memory <memory_id>        # a record + its raw range
corvidae redact --db sessions.db range <start_id> <end_id> # message_log id range (inclusive)
```

A corrupt row (invalid JSON or UTF-8, JSON that is not an object, an
oversized integer) is tombstoned like any other. Every read casts the column
to BLOB and decodes through `decode_message_row`, so the row becomes
`{"role": "unknown", "content": "[redacted by operator <date>]"}` (valid
UTF-8 JSON) with a WARNING naming the row and channel, and a row that did
decode keeps its `role`. The sample token for the verification step comes
back `None` for a corrupt row, so the CLI reports the FTS check as skipped
for it instead of crashing.

`--dry-run` prints affected row counts without writing. Both dry-run and
real runs count only rows that exist and were actually touched, and the
`message` and `range` forms print `<found> of <requested>` so a mistyped id
is visible. A `--db` path that is not an existing file aborts with
`Database not found: <path>` before opening a connection (opening would
silently create an empty database).

**WAL requirement.** The CLI opens its own connection (`PRAGMA busy_timeout
= 5000`, short transactions) and asserts WAL journal mode at startup. If
the mode is not WAL, it aborts with a message naming the current mode and
how to convert. The `corvidae` daemon configures WAL on startup; a
daemon-created DB is always WAL.

**sqlite-vec.** `memory_vec` is a `vec0` virtual table, so the CLI's
connection must load sqlite-vec before the cascade touches it. Each form
loads it right after connecting when `memory_vec` exists (lazily
imported, so kernel-only installs never need it). If it can't be loaded,
the form refuses before writing anything: one line naming `memory_vec`
and the remedy (install the `cognition` extra, or run redact in the
daemon's environment). Before #198 the cascade crashed on `vec0` after
`message_log` was already committed.

**Schema-presence probe (pre-memory-schema and kernel-only DBs).** Before
touching any surface, the CLI probes `sqlite_master` for the tables it
needs. The `message` and `range` forms always write the message
tombstone and skip each absent surface without printing anything: no
`message_fts` means there is nothing to clear (if `MemoryPlugin` creates
it later, its backfill indexes the tombstone, not the original); no
`memory` means no memory cascade; no `memory_vec` skips only the vec-row
delete. A scrubber whose table is absent returns None and counts 0 (the
CLI prints `0 <name> rows`; a library caller of `redact_messages` also
gets a `SKIP` line in its `notices` list). The `memory` form needs the `memory` table and exits with an error
(`memory table not present in this database`) without it. The CLI never
creates schema; schema is daemon-owned.

**Cascade (per-form behavior).**

- **`message` form:** for each supplied `message_log` id, rewrites the
  `message` JSON — keeping `role` and structural keys, replacing `content`
  with `[redacted by operator YYYY-MM-DD]`, dropping `tool_calls` and
  payload fields. Row id and timestamps are unchanged. The `message_log_au`
  trigger propagates the change to `message_fts` automatically (when
  present). Then cascades to memory: for every `memory` row whose
  `[msg_id_start, msg_id_end]` contains the redacted message's id **and**
  whose `channel_id` matches — the `message_log` id sequence is global
  across all channels, so a numeric-range match alone would over-redact
  unrelated channels — tombstones the memory summary to `[redacted by
  operator]`, sets `redacted=1`, `indexed=0`, `embedded=0`, and deletes its
  vec row. The `memory_au` trigger propagates to `memory_fts`.
- **`memory` form:** resolves the record's `[msg_id_start, msg_id_end]`
  range with `WHERE channel_id = record.channel_id AND id BETWEEN ...` —
  the per-channel predicate is load-bearing because the global-id range
  otherwise includes interleaved foreign-channel rows — then tombstones
  those messages. Tombstones the target memory record itself and cascades
  to any other same-channel memory records whose ranges intersect the
  tombstoned message ids.
- **`range` form:** collects all `message_log` ids in the range and
  delegates to the `message` form cascade.
- **Registered scrubbers (all forms, #160):** each form then runs every
  scrubber registered under the `corvidae.redact` entry-point group, once
  per channel, over the message ids it tombstoned (contract in
  [plugin-guide.md](plugin-guide.md#registering-a-redact-scrubber)). The
  kernel doesn't know the tables. The cognition extra registers
  `exchange_log` (`corvidae_cognition.outcome_log:redact_exchange_log`),
  which scrubs the conversation text that table holds. The
  exchange that owns a redacted id X (in that channel, the row with the
  largest `message_rowid <= X`) has every `appraisal.stage2.past_claims`
  claim and query replaced with the tombstone text. The number of claims
  and queries is kept for calibration, and the other appraisal fields are
  untouched. Every exchange in the channel with `message_rowid >=` the
  smallest redacted id has its `provenance_snapshot` set to NULL. That
  deliberately over-scrubs: the snapshot holds the CONTEXT in the window
  at the turn, nothing maps it back to rows, and any later window may
  have carried X's text. Exchanges with a NULL `message_rowid` aren't
  reached. The CLI prints `<N> <name> rows` per scrubber, here
  `<N> exchange_log rows` (distinct exchanges changed; a re-run that
  changes nothing counts zero), and the result dict carries the counts
  as `scrubbed: {name: N}`. The cognition extra also registers
  `retrieval_candidate` (`corvidae_cognition.memory:redact_retrieval_candidates`,
  #407): the per-candidate retrieval log holds ids and numbers only, so it
  has no text to tombstone, and the scrubber deletes every row whose
  `memory_id` is a redacted memory (already flagged, or whose range
  contains a redacted message id), whichever channel's retrieval wrote it.
  The CLI prints `<N> retrieval_candidate rows` (counted, not deleted, under
  `--dry-run`). Schema and outcomes: "Retrieval (read path)" in
  [cognition.md](cognition.md).

**Verification pass.** After each cascade, the CLI extracts a sample token
from the original content (before tombstoning) and runs FTS `MATCH` queries
against `message_fts` and `memory_fts`. It prints `"verified: 0 hits for
'<token>' in <tables>"` when clean or `"WARNING: FTS still contains hits for
'<token>': <table>: <N> hits"` when the cascade left searchable content — every invocation proves its own
completeness. Skipped when `--dry-run` is in effect or when no suitable
sample token can be extracted.

The memory cascade above needs the cognition extra's tables. On a kernel-only
database (`memory`, `memory_vec`, `memory_fts` and `message_fts` absent) the
`message` and `range` forms tombstone the `message_log` rows, and the
verification line reports `verified: no FTS surfaces present`; the
`memory` form exits with the error above. If the extra was installed once
and later removed, `message_fts` and its `message_log_ai`/`message_log_au`
triggers stay in `sessions.db` and keep indexing into a table nothing reads;
`redact` still clears them because it probes for them
([cognition.md](cognition.md)).

## JsonlLogPlugin

`jsonl_log.py` — writes an append-only JSONL log alongside the SQLite
conversation store. Registered as the `"jsonl_log"` entry point.

Configured via `daemon.jsonl_log_dir` in `agent.yaml`. If the key is
absent, the plugin is a complete no-op — all hookimpls return early.
When configured, `on_start` resolves the directory relative to
`_base_dir` and creates it if necessary.

### persist_message

Implements `persist_message` (broadcast). Strips `_message_type`
from the message dict and writes a JSON line with `ts`, `channel`,
`type`, and `message` fields to a per-channel `.jsonl` file.

### on_compaction

Implements `on_compaction` (broadcast). Writes the compaction summary as
a JSON line with `type: "summary"`.

### File handles

Per-channel files are opened lazily in `_get_handle` as `RotatingAppender`s
(`jsonl_rotate.py`) and held open for the plugin lifetime. Channel IDs are
sanitized (`/` and `:` → `_`) for filenames. All are flushed and closed in
`on_stop`.

### Rotation (#404)

`MetricsJsonlPlugin` writes through the same class. Before a write that would
pass `logging.jsonl.max_bytes` (default 10 MiB; `0` disables), the appender
closes the file, shifts `x.jsonl` → `.1` → … → `.N` (`backup_count`, default
5, the oldest dropped) and reopens in append mode, logging the rotation at INFO
with the file's path. The size is read from the open file, so a file already
over the cap rotates on its first write and an external `copytruncate` still
works. A lock serialises writers, since one appender can be written from
several worker threads at once. A rotation that fails (a read-only directory)
is logged once per streak at WARNING, naming the file, and the file keeps
growing past the cap until rotation works again. `parse_rotation` validates
the two keys in `on_init`; a bad value is a config error. Nothing reads a
rotated segment; `message_log` is the durable record (principle 4).

## Clock and idle tracker (`clock.py`, `idle.py`)

Time passing and the system going quiet are kernel inputs. `Runtime.start()`
builds one `Clock` and one `IdleTracker` and registers them with the plugin
manager as `"clock"` and `"idle"` before the entry points load. They are
not entry-point plugins. Listing either in `plugins.disabled` unregisters
it, and startup then fails dependency validation. Plugins declare
`depends_on = frozenset({"clock"})` or `{"clock", "idle"}` and resolve them
with `get_dependency`.

**Clock.** `now()` is wall time, `monotonic()` is monotonic time, and
`sleep(delay, wake=None)` sleeps on the monotonic clock until `delay`
passes or the optional `asyncio.Event` is set. `call_later(delay, cb,
name=...)` and `call_at(when, cb, name=...)` run an async callback once and
return a `Timer` whose `cancel()` stops it while it is still waiting
(a no-op once the callback has started). `call_soon(cb, name=...)` runs it
once on the next loop step, without waiting on time, and tracks it like any
timer, so `live_timer_names()` lists it and `close()` cancels it. `call_at`
waits on wall time in steps of at most an hour, so a suspend or an NTP step is noticed within
the hour. Timers are one-shot: a periodic job re-arms itself from its
callback. A callback that raises is logged with the timer's name. A callback
must let `asyncio.CancelledError` through, because `Clock.close()` stops
running callbacks that way.

The Agent uses the clock for turn retries: it schedules a
`turn-retry:<channel id>` timer after a backend-unreachable failure (see
Message processing). The notification spill service uses it for its hourly
cleanup of aged spill files, and for the wall time those files age against
([Notification spill](#notification-spill-spillpy)).

**IdleTracker.** It answers "has this channel been quiet for N seconds?" and
"has the whole system?". A channel is busy from `note_activity` until
`note_settled`, and quiet from then on. The system is quiet when no
channel is busy, counted from the last settle (or from `start()`).

A plugin registers a watch:

```python
self._watch = idle.watch_channels(1800, self._on_quiet, name="my.plugin")
self._sys = idle.watch_system(60, self._on_system_quiet, name="my.plugin.sys")
```

`watch_channels(period, cb)` calls `cb(channel_id)` once a channel has been
quiet `period` seconds. `watch_system(period, cb)` calls `cb()` once the
whole system has been. Both return an `IdleWatch` with `cancel()`. A watch
repeats every `repeat` seconds (default `period`) while the quiet lasts, and
only if nothing became active while the callback ran. `repeat` must be a
positive finite number, and a negative `period` is rejected: both raise
`ValueError` naming the watch. A zero period therefore needs an explicit
`repeat`. A due zero-period watch runs through `Clock.call_soon`, so
`clock.close()` cancels it like any other timer. Activity cancels the
waiting timers. An upkeep callback (consolidation, retention) never starts
a turn or enqueues anything; it does plugin-owned work. The one agency
watcher, cognition's `HeartbeatPlugin`, starts a turn from its system
watch through `on_notify` (principle 13; [cognition.md](cognition.md#heartbeatplugin)).
A callback that awaits before acting can call `system_quiet()` to re-check
that no channel went busy in the meantime.

This is a registration API, not a hook. A hook cannot carry per-plugin
periods, and a broadcast with a global cooldown starves any plugin whose
period is longer than the cooldown.

**Restart seed.** A freshly booted daemon has seen no traffic, but its
channels have history. `PersistencePlugin.on_start` calls
`note_restored(channel_id, last_activity)` for every channel in
`message_log`, so each is quiet since its last message (wall-time age,
converted to monotonic). A channel with history that has not spoken since
before the restart therefore reaches its idle period on schedule instead
of never. `note_restored` ignores channels the tracker already knows.

**Lifecycle.** The tracker does not fire until `Runtime.start()` calls
`idle.start()`, after the Agent can take work. Shutdown runs
`agent.on_stop`, the `on_stop` broadcast, `idle.stop()` and then
`clock.close()`, which cancels any timers plugins left behind. Plugins
should still cancel their own watches and timers in `on_stop`.

**Shutdown bounds and the exit watchdog (#746).** Two bounds, both
`daemon:` keys parsed by `corvidae/shutdown.py` as the first step of
startup validation (a bad value is a `Config error` at startup; neither
key is validated on hot reload, so a bad edit is caught by the next
restart or the restart tool's pre-flight):

1. `daemon.shutdown_timeout` (default 3 s) bounds the graceful phase
   above. On overrun `stop()` logs a WARNING naming the key and calls
   `os._exit(1)`.
2. `daemon.exit_timeout` (default 5 s; `0`/`null` disables) bounds what
   comes after. When `stop()` returns, CPython still has to finish
   `asyncio.run` (which joins executor threads for up to 300 s) and
   then join every non-daemon thread in `threading._shutdown`, with no
   limit. A thread a plugin leaked (an aiosqlite connection never closed
   keeps one alive) hangs the process there forever, and `atexit` runs
   only after that join, so it cannot help. `Runtime.run(exit_watchdog=True)`
   arms a daemon thread in a `finally` around `await self.stop()`, so it
   is armed even when an `on_stop` raises (apluggy's broadcast propagates
   the first exception and skips the rest of shutdown, the likeliest way
   to leak). If the process is still alive `exit_timeout` seconds later,
   the thread logs a WARNING naming the non-daemon threads still running
   and the key, flushes the root logger's handlers (not
   `logging.shutdown()`, which runs from `atexit` and would close
   handlers the main thread may still use) and calls `os._exit(1)`.
   The non-zero status is deliberate: the supervisor records an
   abnormal run, and the restart plugin's next boot reports it.

The watchdog is armed only through `corvidae.runtime.run_daemon(runtime)`,
which `serve`, `cli` and `acp` call. `run()` defaults the flag off because
tests drive `run()` and `stop()` in-process, where a watchdog would end
the pytest process. Worst case after SIGTERM:
`shutdown_timeout + exit_timeout`. The restart pre-flight and a failed
`start()` are not covered.

**Failed start.** If `Runtime.start()` raises anything, `BaseException`
included (a `CancelledError` leaked from a plugin's `on_start`, #282; a
config error's `SystemExit` before any plugin loads finds nothing to close),
`Runtime.run` runs `_cleanup_failed_start()` before re-raising: the
`on_stop` broadcast, `idle.stop()` and `clock.close()`, under the same
bound as `stop()` (`daemon.shutdown_timeout`, 3 s by default) (as there, an `on_stop` that raises skips the last two;
`asyncio.run` cancels leftover timers on exit). `agent.on_stop` is skipped, since the Agent may not
have started. Without this, persistence's open aiosqlite connection (a
non-daemon worker thread) kept the interpreter from exiting, so a daemon
whose plugin failed to start stayed alive and silent (#177). A cleanup
failure or timeout is logged at WARNING and the start exception is the one
that propagates, so the process exits non-zero. A plugin's `on_stop`
therefore has to tolerate running without its `on_start` having finished.

**Config.** None. Each plugin sets its own period (for instance
`memory.idle_consolidate_after`). The former `daemon.idle_cooldown_seconds`
and `daemon.idle_poll_interval` keys are removed, and a config that
carries either fails at startup (a reload carrying one is rejected).

**Tests.** `tests/fake_clock.py` provides a fake clock and
`install_kernel_clock(pm)`, which registers a fake clock and a real
tracker on a plugin manager; tests advance time without sleeping.

## ThinkingPlugin

`thinking.py` — strips `<think>` blocks and `reasoning_content` for
display. Registered as the `"thinking"` entry point.

Implements three hooks, driven by the resolved `reasoning_history`:

- `before_agent_turn`: applies the policy to the whole in-memory window.
  `drop` strips every assistant message; `exchange` strips those before
  `completed_exchange_end()` (the last `role: user` message; tool results
  and system notifications do not end an exchange); `keep` is a no-op.
  Covering the whole window is what makes the policy hold for history
  reloaded from the DB, which always carries `reasoning_content`.
- `after_persist_assistant`: under `drop`, strips the just-appended
  message. The DB copy is already written; this only affects subsequent
  prompt builds.
- `transform_display_text`: calls `strip_thinking` on the response text.
  Returns the stripped string if it differs from the input, or `None`
  if no `<think>` tags were present.

**Graceful degradation:** without `ThinkingPlugin`, `<think>` blocks
pass through to the channel and `reasoning_content` remains in the
in-memory history regardless of `reasoning_history`.

## CompactionPlugin

`compaction.py` — implements the `resolve_compaction` hook to keep
conversation history within the configured token budget. Registered as
the `"compaction"` entry point. Declares `depends_on = {"llm"}`.
Logger name: `corvidae.compaction`.

**ContextCompactPlugin** (`context_compact.py`) has been removed: superseded
by `MemoryPlugin` (memory consolidation plus retrieval), and
per-turn token stats now live in the `usage_log` table.

### resolve_compaction

**Over-budget fallback.** Compaction can decline (too few rows, failure
cooldown, too soon, an empty summary) or fail, so after the hook the
kernel enforces `max_context_tokens` itself with
`ContextWindow.fit_to_budget` (see the ContextWindow section): oldest rows
leave the in-memory window, `message_log` keeps them, no compaction hook
fires, and one WARNING names the channel, the rows dropped (with their
`message_log` id range), any truncation, and the remedy (raise
`max_context_tokens` or check compaction). When `min_messages_to_compact + 1`
rows don't fit the budget, compaction never runs on that channel and the
fallback alone keeps it under budget.

Implements the `resolve_compaction` hook (`firstresult=True`) at
`trylast`, so it runs only if no earlier implementation (tryfirst or
default priority) returned non-None first. The caller does not use the return value.

**Algorithm:**

1. If `conversation.token_estimate()` — the same JSON-serialized
   per-message measure described above, applied consistently across the
   trigger check and the retain-walk below —
   is less than 80% of `max_tokens`, return `None` (skip).
2. If `len(conversation.messages) <= 5`, return `None` (skip — too few messages to compact).
3. Backward walk: starting from the most recent message, accumulate messages
   until their token estimate would exceed 50% of `max_tokens`. The count
   of accumulated messages is `retain_count`. The cut
   (`len(messages) - retain_count`) is then moved so no tool cycle
   straddles it (`_clean_cut`), because a straddle drops a call while
   keeping its results, and makes the dropped prefix render differently
   from the start of the full window, which a compaction fork needs.
   First choice: back up with `snap_cut_to_tool_boundary` to keep the
   cycle, if that drops some dialog (not just the previous summary) and
   the window after compaction (system prompt, a 2000-token summary
   allowance, and the kept messages) stays at or below the midpoint
   between `compaction_retention` and `compaction_threshold`. Otherwise (a
   long-running call near the head with chat interleaved behind it) move
   forward with `snap_cut_forward` to drop the cycle whole, if at least
   half the retention budget survives. Otherwise keep the walk's cut, as
   before #51. Backing up without that bound could leave compaction
   dropping almost nothing and re-running every few messages. In-flight
   calls have no result yet and never straddle; one dropped by compaction
   leaves its later result orphaned, as does the walk's own cut when it
   stands. `build_prompt` renders such orphans as framed notifications
   (see `ContextWindow` above), so the cut logic need not prevent them.
4. If `retain_count >= len(conversation.messages)`, return `None` (all messages fit — no-op).
5. The summary is written from the window prefix at the cut (`messages[:cut]`),
   not from a filtered copy: the prefix already holds any prior summary at its
   head, and CONTEXT rows render as they do in a main turn.
6. With forks on for `compaction` (`agent.forks`, see [Forks](#forks)),
   fork the window at the cut with `agent.compaction_fork_prompt` (default
   `CompactionPlugin.DEFAULT_FORK_PROMPT`, `prompts/compaction_fork.md`)
   and use the reply as the summary. If forks are off for the stage, or the
   fork raises or comes back empty (logged at WARNING), `_view_summary`
   sends the same render (`build_fork_prompt` over the window at the cut,
   the channel's pending tool-call ids, the Agent's
   `mid_conversation_system_role`) as a plain call on the `compaction` stage's profile,
   with no tools, through `view_call` (PRD §3.8). The render is the
   uncapped prefix: the window is over budget when compaction runs. If the fork
   raises a connection-class error (`is_backend_unreachable`), the view call is not
   attempted: it would spend the same wait on a backend that is already
   down. `_fork_summary` raises `BackendUnreachable`, `resolve_compaction`
   records the failure cooldown and returns, and the turn's own retry loop
   waits for the backend (#400, PRD §3.1). If the view call raises or returns
   nothing (logged at WARNING; `RuntimeError` when no Agent can render a
   view), compaction declines and the over-budget fallback (#308) brings the
   window back within budget. The client is resolved when
   compaction runs (`get_dependency(pm, "llm", LLMPlugin).get_client("compaction")`,
   a `chat` stage, so `llm.stages.compaction`, else the default chat profile), not in `on_start`. `agent.compaction_prompt` and the standalone summarizer
   (`_summarize`, with its prior-summary cap and head/tail truncation) are
   gone; setting the key logs a startup warning and is otherwise ignored.
7. Build the summary message (`[Summary of earlier conversation]\n…`).
   Its role is `assistant`, unless the retained tail holds no `user`
   message (a long tool loop); then it is `user`, because strict chat
   templates (Swift-1.5 / Qwen 3.8) reject a prompt with no user query
   (#56). The role is persisted with the summary row, so a reload renders
   the same prompt.
8. Fire `before_compaction(channel, compacted_ids, cut)` (broadcast,
   awaited) while the window is still whole. `MemoryPlugin` copies the
   window here and starts the background fork of the compaction-time
   memory record (#628), without waiting for it. Failures are logged and
   ignored.
9. Call `conversation.replace_with_summary(summary_msg, retain_count)`.
10. Fire `on_compaction(channel, summary_msg, retain_count, compacted_ids)`
   (broadcast) so persistence plugins can update the DB.
11. Return `True`.

The summary LLM call is attributed `stage="compaction"` plus
`mode="fork"` or, for the view call, `mode="standalone"`.

**Graceful degradation:** without `CompactionPlugin`, the
`resolve_compaction` hook returns `None` on every turn and compaction
does not run. The conversation history will grow without bound until the
LLM context window is exceeded.

## Forks

`fork.py` plus `Agent.fork` — background processes run as branches of the
main window (GitHub issue #51; plan of record `plans/forks.md`). On by
default (#320): `agent.forks.enabled`.

A **fork** renders the channel's window exactly as the main turn does, up
to a cut, and appends one `user` instruction. The prompt is then a prefix of
what the server already holds, so it reuses the cached KV, and it carries
the agent's own system prompt, so first-person records and summaries are
written by the agent they describe.

```
fork_prompt = build_prompt(window[:cut]) + [{"role": "user", "content": instruction}]
```

That render is the **rendered view** (PRD §3.8), `build_fork_prompt` in
`fork.py`: the system prompt, a window, and one `user` instruction. A fork
sends the live window to its stage's profile with the channel's tools. The **view
call** (`view_call`) sends the same kind of render to its stage's profile with
no tools and no `tool_choice`, attributed `mode="standalone"`. It has two
producers: compaction on a fork miss (the live window at the cut, sent as
the `compaction` stage) and memory catch-up (a window rebuilt from
`message_log`, below, sent as the `consolidation` stage). It carries no cache
benefit: it is a cold prefill, on the slot of the stage's profile (the default chat
profile's when the stage is unmapped).

**View API (kernel, `corvidae.fork` and the Agent).** Cognition gets
everything channel-specific from one value, `ViewInputs` (`system_prompt`,
`max_tokens`, `mid_role`, and `apply_reasoning_history`, the channel's
policy bound to a callable), returned by `view_agent(pm).view_inputs(channel_id)`.
`view_agent` is duck-typed like `fork_agent` but ignores `agent.forks`, so
forks off still gets a view. `Agent.view_inputs` resolves a persisted channel
id with `resolve_persisted_channel` (the registry, then the owning
transport's `resolve_channel`, then a bare channel; it raises `ChannelNotReady`
when a transport cannot build the channel yet, so a Signal alias unresolved
after a restart retries on the next trigger), then fills the system prompt
(the live conversation's own, else `Agent.channel_system_prompt`: identity,
blank line, register), the resolved `max_context_tokens`, the
`mid_conversation_system_role` property, and `apply_reasoning_history`
(`thinking.py`) bound to the channel's `reasoning_history`. The conversation
init block calls `channel_system_prompt` too, so the two can't drift.
`persistence.view_rows` (the rows reload loads under a summary, or the
whole history) and `decodable_summaries` (every summary row that decodes,
newest first) are shared by `resolve_conversation` and the catch-up builder.
`view_call` raises `ForkRefused` on a reply with tool calls (structured or
written out); transport errors propagate.

**Prefix parity.** `Agent.fork(channel, instruction, *, stage, cut=None,
extra_body=None, snapshot=None)` renders with exactly what step 7 of the agent loop uses:
the channel's
conversation tool schemas (`conv.tool_view.schemas`, per-channel since #149),
`build_prompt(pending_ids, mid_conversation_system_role)`, and the same
runtime-override filter (`_llm_overrides`). Any difference in what reaches
the template breaks the prefix and the cache with it. Only where the request
goes differs: `stage` is a required keyword, and the call goes to
`LLMPlugin.get_client(stage)`, the client of the profile `llm.stages` maps
that stage to (else the default chat profile). Its endpoint, model and
`extra_body` (e.g. `id_slot`) are the stage profile's, not the main turn's.
With no client for the stage, `Agent.fork` raises `ForkUnavailable`.
`snapshot` (a `WindowSnapshot` from `corvidae.fork.snapshot_window`, #628)
renders that copy of the window and its pending ids instead of the live
window; it is how compaction's consolidation forks after the turn, from the
window as it stood before `replace_with_summary`. The copy holds the whole
message list, since the pending-at-cut rule looks past the cut.

Whether a fork hits the cache is therefore the operator's configuration,
not a guarantee. A fork whose stage maps to a profile on another endpoint or
model misses the main window's prefix, and so does one whose profile sets a
template-visible key (such as `reasoning_effort`) the main profile does not.
Corvidae passes profile parameters through without interpreting them. To keep
the prefix, derive the stage's profile from the main one with `extends` and
change only sampler-side keys; #440 reports misses. With a `cut`, calls before the cut whose first result falls at or after it render as pending, not abandoned, because they were pending as of the cut (#185). `reasoning_effort`
is left alone, because Qwen 3.8/Swift's template renders the effort
sentence first in the system block: a fork at a different effort would
miss the cache from its first token.
On top of that a fork sends `tool_choice: "none"` (tools stay in the
render) and the stage's own `extra_body` (`response_format`; consolidation
sends the memory-record schema). A reasoning budget is no longer a fork
setting: put `reasoning_budget_tokens` in the `extra_body` of the profile the
stage maps to, and a fork sends none when its profile has none.

**Body precedence.** `LLMClient.chat` merges the profile's `extra_body` (the
instance body) first and the call's `extra_body` over it, so a fork's call
body is built to give this order, later entries winning:

1. `model`, `messages` and `tools`.
2. The stage profile's resolved `extra_body` (inherited keys included).
3. The channel's runtime overrides, minus every key the stage profile's
   `extra_body` also sets.
4. The stage's `extra_body` argument (`response_format`).
5. `tool_choice: "none"`.

So the profile beats `set_settings`: a channel's runtime `max_tokens: 200`
against a stage profile with `extra_body.max_tokens: 4096` sends 4096. The
stage's schema beats the profile, and `tool_choice` beats everything, so
`set_settings` can't re-enable tool calls on a fork. `fork_body(overrides,
profile_body, stage_body)` in `fork.py` is the pure function that builds the
call body. `model` is not an inference override, so the profile's model
always applies.

The cost is a cache miss when a channel override changes a key the stage's
profile also sets. Every key `_llm_overrides` can carry is sampler-side
(temperature, top_p, top_k, the penalties, `max_tokens`, `seed`), so on
llama-server none of them changes the rendered prompt, and a dropped
override changes only the fork's sampling. A fork whose stage is unmapped
resolves to the default chat profile, so an override of a key in that
profile's `extra_body` is dropped for its forks while still applying to the
channel's main turns.

**Invariants.** A fork mutates nothing, persists nothing, fires no hooks,
and never dispatches tools: a reply carrying `tool_calls`, or a tool call
written out as text (`<tool_call>`, a `<function…>` tag, a bare
`{"name", "arguments"}` object), raises `ForkRefused`. No live window or
no client for its stage raises `ForkUnavailable`.
Every caller treats any fork exception, or an empty or unparseable reply,
as a miss: it logs at WARNING and the live-window stage skips, with no
standalone call (PRD §3.8.5, #320). The exceptions are compaction, whose
summary on a miss is the standalone view call and then #308, and
consolidation of a range that has left the window (the compaction trigger's
task after the drop, an idle range not wholly in the window), which is
written from a rebuilt view sent as the `consolidation` stage (catch-up, below). Idle
and turn consolidation of a range wholly in the live window still skip on a
miss, and are retried on the next trigger of the same kind (the warning
names it); once the range has left the window an idle range reaches
catch-up, and a turn range is skipped at debug and left to idle or
compaction. A JSON parse failure names the failing phase
(consolidation, appraisal, or critique) and includes up to 500 characters of
the provider text in the `ValueError`.

**Which process forks from where.** Plugins reach the Agent through
`fork_agent(pm, stage)`, which returns it only when that stage is enabled.
Stage names are opaque to core (it never names cognition plugins).

| Stage | Cut | Runs | On a miss |
|---|---|---|---|
| `compaction` | dropped prefix | inline in `resolve_compaction` | standalone view call on the same prefix (the `compaction` stage's profile, no tools), then the over-budget fallback (#308); on a connection-class failure the view is skipped and compaction defers to the turn's retry loop (#400) |
| `consolidation` (compaction trigger) | dropped prefix of a copy of the window taken in `before_compaction` (#628) | in a background task `before_compaction` spawns (it returns at once), after waiting for the channel's in-flight turn or idle consolidation task (#621); the task is stashed and the `on_compaction` task, which waits the same way, awaits its record | catch-up in the `on_compaction` task: no stash (range not in `messages[:cut]`, fork failed) or the watermark moved since the fork; the rebuilt view, sent as the `consolidation` stage |
| `consolidation` (idle trigger) | window up to `range_end` (cut just past it, #622) | in the idle task | range wholly in the live window: skip with a warning, watermark unchanged, retried on the next idle trigger; no registered channel or window, or a range row not in the window: catch-up (rebuilt view, sent as the `consolidation` stage) |
| `consolidation` (turn trigger) | window up to `range_end` (cut just past it, #622) | in the task `on_agent_response` spawns, once the new window tokens in `(watermark, range_end]` reach `memory.consolidate_after_tokens`; never attempted with forks off for `consolidation` (silent) | range wholly in the live window: skip with a warning, watermark unchanged, retried on the next turn trigger, until three consecutive failures back the trigger off for the channel (one warning) until the watermark moves or `memory.idle_consolidate_after` passes (#620); no registered channel or window, or a range row not in the window: skip at DEBUG, never catch-up (idle or compaction consolidates it) |
| `appraisal` (stage 2) | full window at run time | silent TaskQueue task | skip with a warning (forks off, originating rowid left the window or was never persisted, fork failed or unusable) |
| `critique` (each stylistic lens) | full window at run time | silent TaskQueue task | same as appraisal; a lens whose fork misses is skipped alone; if no lens ran, no outcome is recorded |
| `critique` (provenance) | prefix ending at the exchange's final reply | silent TaskQueue task | skip with a warning (forks off, no live window, reply not recorded or not in the window, a compaction since the reply, or a failed fork) |

Seeding, imports and the `eval_*` scripts have no live window and always
run standalone. `consolidate_replay.py` sends the daemon's own catch-up
request. Subagents are unchanged: workers with their own persona-neutral
prompts.

**Catch-up consolidation.** A range that has left the window has no live
prefix to fork, so `MemoryPlugin` rebuilds the agent's view of it from
`message_log` (`build_catchup_segments`, which the replay harness calls too).
For each non-summary range row the *summary in force* is the newest
decodable summary row dated before it; a *run* is a maximal sequence of rows
under the same summary, which is one window. A run's view is that summary
plus the rows reload would load under it (in id order, through the run's
last row). A run is then cut into *segments* that each fit the channel's
`max_context_tokens` (system prompt, the instruction and the segment's own
rows; a tool cycle straddling a cut goes whole into the next segment, and a
unit that alone overflows is its own segment), so every range row lands in
some view. Each segment's view is trimmed oldest-first with
`ContextWindow.fit_to_budget` to the budget less the instruction's size, then
put under the channel's `reasoning_history` policy (the live loop's order),
and ends with the consolidation instruction (`memory.consolidation_fork_prompt`
plus the range sentence). Row dicts are copied per segment so the policy's
stripping can't alter rows a later segment reuses as context. Warnings, each
naming the channel and message ids: the budget trim dropped a segment's own
rows (they commit without reaching a view); a row was truncated; the
finished request still measures over budget (it is sent anyway). Each
segment is one `view_call` on the `consolidation` stage's profile, one consolidation (one or more
entries, each its own record), and one commit
that is a compare-and-set on the previous segment's end; a lost
compare-and-set discards the rest, and a failure leaves earlier segments
committed and the failure log says through which id. An unusable reply is
retried once, then degrades to the raw transcript, split into records that fit the memory admission budget and stamped `mode = raw` (#626). A segment
with no dialog that is followed by another segment folds its id span into
the next segment's record (no record or commit of its own); a no-dialog
last segment advances the watermark without a record. Records are
stamped `(memory.consolidation_fork_prompt, "standalone", the
request model)`. A view that the stage's profile rejects for length is a
transport error: the watermark stays and the next trigger retries.

The compaction and consolidation forks share one prefix: the second
fork's prefill is just its instruction. Containment for consolidation is
exact: every `message`-type row in the range must be a `_db_id` in the
fork's window (summary rows are skipped; a live window's summary has no
rowid and summaries are never consolidated). Appraisal and the stylistic lenses check
the originating rowid recorded by their own `on_message_persisted`
(critique per lens, right before each fork, since an earlier lens's fork
is an await during which compaction can drop the exchange), and read the
window at run time, so a fork after the user has said more still sees the
exchange. The instruction names it by quoting its opening message and the
response the user was sent, which can differ from the window's last
assistant message (the max-turns fallback). The provenance check instead
records the reply's rowid and the channel's compaction count in
`on_agent_response` (the count is bumped in `before_compaction`, before
the window changes), and checks both before its search and again right
before the fork.

**Instructions.** Each stage has a fork instruction, written to the agent in the second person.
The voice is a choice, not a measured result; #186's prompt-voice A/B will
say whether it matters:
`agent.compaction_fork_prompt`, `memory.consolidation_fork_prompt`,
`appraisal.stage2_fork_prompt`, `critique.fork_prompt.<lens>` (documented
copies in `prompts/*_fork.md`). They are read from static config with
`tuning.config_value`, never `resolve_tunable`, so `set_settings` can't
write them. Consolidation appends a sentence naming where the range
starts ("the whole conversation above", "what follows the summary at its
top", or a quote of the first new message when rows below the watermark
are still in view). Appraisal appends the exchange quotes and the
retrieval profile. Critique appends the exchange quotes, and the
provenance check also passes the CONTEXT snapshot inline, along with each
past claim and its closest earlier messages.

`ForkResult.model` is the model the fork request was sent with
(`LLMClient.request_model` of the fork's body, so an `extra_body` override
shows). `Agent.fork` always sets it, so it is None only when a
`ForkResult` is built without it. Consolidation stamps it on a fork record, together with
`memory.consolidation_fork_prompt` but not the range sentence
([cognition.md](cognition.md#record-stamps)).

**Configuration** (`agent.forks`, validated at startup and on reload by
`validate_agent_config`):

```yaml
agent:
  forks:
    enabled: true                 # default; false: live-window stages skip, compaction uses a view call
    stages: null                  # null = every stage; or a list of names
```

`reasoning_budget_tokens` is no longer a key here: a config that still sets
it fails startup with an error that names it and points at `llm.stages`.

A fork's profile, and so its endpoint, model, reasoning budget and token cap,
come from `llm.stages` (see LLMPlugin, "Configuration"), not from `agent.forks`.

### Forks on the deployed backend

The PRD (§3.8) states what forks must achieve. What follows is how the
deployment meets it, and holds for llama.cpp's llama-server with one KV
slot and Swift-1.5 (a Qwen 3.8 fine-tune); another backend may answer
differently.

**Fork by default (#180, #320).** Every background call that has a live
window forks it: compaction from the dropped prefix; compaction-time
consolidation from the same prefix of a copy taken in `before_compaction`,
while the window is whole, forked in the background (#628); idle and turn consolidation from the window up to `range_end` (#622); stage-2
appraisal and each critique lens from the full window at run time;
provenance from the prefix ending at the exchange's final reply. Each forks
only while the rows it needs are still in the window. A miss skips the
stage with a warning (the rescue is a retry or a skip, never a standalone
call), except compaction, which on a miss sends the same view to
the `compaction` stage's profile and, if that fails too, leaves the window to the
over-budget fallback (#308), and consolidation of a range that has left the
window by the compaction or idle trigger, which is written from its rebuilt view sent as the `consolidation` stage (the turn trigger skips such a range at debug). Skip-on-miss
and the default are implemented (#320).
Still targets, not enforced today: stage 2 and the lenses forking only the
prefix ending at the reply, a reply cap that fits the slot's context (an
operator can set `max_tokens` per stage in the stage profile's `extra_body`;
nothing checks that it fits), the headroom guarantee in the cost model below,
and treating a fork that would not fit the slot's context as a miss.

**Cost model.** A fork reuses the cached prefix and prefills only its
instruction and reply; the window is append-only, so a fork's prefix is
still cached as long as no compaction has run since. A view call (compaction's fallback, catch-up consolidation)
costs its own cold prefill on the slot of the stage's profile (the default chat profile's when the stage is unmapped): at most the channel's
budget per catch-up segment, and for compaction's fallback the whole
uncapped prefix at the cut. The consolidation and compaction prompts that
described the conversation to a model are gone; what still makes a
standalone call of a rebuilt prompt (the importance prior's rubric call
and evals) pays the same cold prefill. Neither kind
evicts the main window when the server keeps `cache-ram`: on Lex (one slot,
`cache-ram = 32768`) #73's K1 measured the main turn after background
calls fully cached, restored from host RAM. Without `cache-ram`, a
view or standalone call on a single slot evicts the main window and the next main
turn re-prefills it. Either way only one call runs on the slot at a time,
so a background call delays a waiting main turn by its own slot time;
today only what the operator puts in a stage's profile's `extra_body` (a
`reasoning_budget_tokens`, a `max_tokens`) bounds a fork's share of that; an
unmapped fork is bounded by the default chat profile's. #181's target adds the
rest: exchange-scoped forks scheduled before the slot's next main turn (PRD
principle 3), and headroom (the window at the compaction threshold plus a
fork's instruction and its profile's caps fits the slot's
context, so a fork just short of compaction still runs). #182 measures
how long main turns wait today.

**Provenance is fork-only (#164).** The standalone provenance critic
judged a rebuilt context: a search of earlier messages plus a CONTEXT
snapshot. It missed support in the system prompt and the exchange's own
tool results, and objected to claims those supported (a `USER.md` fact, a
`grep` result from two turns earlier). The forked window up to the reply
is what the agent had, so provenance runs only as that fork and skips on
a miss.

**Cache probes (#73).** #73's probes measured what forking costs
on a backend: the fork row and the main turn after it reuse the cached prefix
(K1), the fork prompt begins with the main prompt (K2), and
`response_format` holds under `tool_choice: "none"` with tools present
(K3). On Lex at 913f7ab (appraisal and critique forking) all three
passed; K3 ran offline on a Lex `sessions.db` snapshot. K3 also found
the 2,048 reasoning budget binding on 5/10 appraisal and 9/10 provenance
forks; #186 tracks it. The flip of `enabled` was made by #320. #73's comparisons of fork
against standalone (planted-defect recall per lens, consolidation and
compaction fidelity) are benchmarks, not gates.

## Task System

`task.py` — general-purpose async work dispatch.

**Task** — an async callable with delivery context:

```python
@dataclass
class Task:
    work: Callable[[], Awaitable[str]]
    channel: Channel
    task_id: str           # auto-generated 12-char hex
    created_at: float
    tool_call_id: str | None = None
    description: str = ""
```

The queue doesn't know what `work` does — it calls `await task.work()`,
catches exceptions, and delivers the result via `on_notify`.

**TaskQueue** — FIFO async worker with configurable concurrency. Accepts
`max_workers` (default 1) to run up to that many tasks simultaneously.
Tracks active tasks and completed results (bounded deque, last 100).
`is_idle` property returns `True` when the queue has no pending items
and no tasks are currently executing.

**TaskPlugin** — hookimpl that owns the TaskQueue. Reads
`daemon.max_task_workers` from config (default 4) to set concurrency,
`daemon.completed_task_buffer` (default 100) to set the completed-result
history bound, and `daemon.task_results_path` (unset by default) to enable
JSONL result persistence. Starts/stops the worker on lifecycle hooks.
Registers the `task_status` and `task_result` tools. On task completion,
persists a JSONL record (when enabled) and fires `on_notify`.

Results are kept in two places: an in-memory bounded deque (last
`completed_task_buffer` entries, serving `task_status` and the fast path of
`task_result`) and, when `daemon.task_results_path` is set, a JSONL audit
file — one record per completion, `{"ts", "task_id", "channel",
"description", "result"}` — that outlives the deque and a daemon restart.
`TaskQueue.get_result(task_id)` serves the fast path from the deque and
falls back to a file scan; the `task_result` tool runs the deque leg on the
event-loop thread and the file leg off it, because the worker appends to (and
evicts from) the deque on the loop thread and CPython's deque iterator raises
if it is mutated during iteration. The file has no rotation or cap; operators
own rotating it (an index by `task_id`, e.g. SQLite, is the longer-term
option for lookups that outgrow a full re-scan).

## LLMPlugin

`llm_plugin.py` — owns `LLMClient` instance lifecycle. Registered as the
`"llm"` entry point. Declares `depends_on = set()`.

### Lifecycle

- `on_start` — collects the model-calling stages (below), parses the `llm:`
  section with `parse_llm_config`, checks `llm.stages` against the
  registered stages (`check_stage_map`), then creates an `LLMClient` for
  each resolved profile and calls `start()` on it. It logs a WARNING for
  each kind that has stages falling back to a default that doesn't exist,
  then logs where each stage resolves: `llm stages resolved: ...`
  (`→(none)` for a stage with no profile).
- `on_stop` — calls `stop()` on all clients.
- `on_config_reload` — swaps no client. If the reloaded `llm:` section
  parses to something other than what the daemon started with, it logs one
  summary WARNING (ConfigWatcher's per-key restart-only warnings fire too,
  since `LLMPlugin` declares no `reloadable_keys`): `llm config changed; LLM profiles, llm.stages and llm.defaults
  apply on restart`. Everything under `llm:` is restart-only, `llm.main`
  included. Live reload would need a rollback path, an ordering against the
  Agent re-borrowing its client, and an encoder re-check for embeddings, and
  nothing asked for it. PRD §3.6.1 says a change to a base profile reaches
  its derived ones on reload too; that gap is #465.

`Runtime.start` (step 7) and `ConfigWatcher` (step 3) both run
`parse_llm_config`, so a malformed `llm:` section exits at startup with the
usual `Config error: ... — check <path>` line before any plugin loads, and
a reload that carries one is refused as a whole (logged at `ERROR`, nothing
applied). The stage-map check that needs the registered stages runs in
`on_start` only.

`_create_client` expands `${VAR}` in a profile's `api_key` from the daemon's
environment (`corvidae/envref.py::expand_env_refs`, shared with MCP
`headers`). An unset, empty or multi-line variable raises a `ValueError`
naming `llm.<profile>.api_key` and the variable, never the value; `on_start`
goes through it. The resolved key lives only on the client, so
`LLMConfig.profiles` keeps the raw `${VAR}` text. `extends`, `kind` and
`dimensions` are consumed by `LLMPlugin` and never reach `LLMClient`.

### Client access

Every key under `llm:` except `defaults` and `stages` is a **profile**,
built into one client. A plugin that calls a model
registers each call site as a **stage** through the `register_llm_stages`
hook (`stages[name] = kind`, kind `"chat"` or `"embedding"`), which
`LLMPlugin.on_start` broadcasts once. Core registers `compaction` and
`subagent`; the cognition plugins register `consolidation`, `rubric`,
`embedding`, `appraisal` and `critique`. A plugin names stages, never
profiles. A stage resolves to the profile `llm.stages` maps it to; a stage
the map doesn't name uses its kind's default profile (`LLMConfig.profile_for`).
There are no fallback chains. Main turns are not a stage: they use the
default chat profile.

`LLMPlugin.get_client()` returns the default chat profile's client (main
turns). `get_client(stage)` returns the client for a registered stage, or
`None` when the stage has no profile (the kind has no default and the map
doesn't name the stage: with no embedding profile the `embedding` stage gets
no client and memory degrades to keyword search). An unregistered name
raises `ValueError`, profile names such as `"main"` included.
`resolve_profile(stage)` returns the profile name and `profile_config(stage)`
the resolved settings (read-only; `Agent.fork` reads its `extra_body`).
Other plugins retrieve the plugin via `get_dependency(pm, "llm", LLMPlugin)`.
`MemoryPlugin` reads the embedding stage's encoder identity, `dimensions` and
prefixes the same way, with `parse_llm_config(config.get("llm"))` in its
`on_init`, so it follows the map and `extends` and can't diverge from the
client `LLMPlugin` builds.

Every call a stage makes goes to its profile: its fork, its view call, its
windowless call, a subagent's turns, the rubric prior and embeddings.
`usage_log` and the metrics name the stage's profile on every call, forks
included, because the profile's client carries its name.

### Configuration

```yaml
llm:
  main:                       # a profile; kind omitted means chat
    base_url: "http://host:8080"
    model: "model-name"
    api_key: "optional"
    extra_body: {}
    max_retries: 3
    retry_base_delay: 2.0
    retry_max_delay: 60.0
    model_not_loaded_wait: 60.0
    timeout: null             # optional; no timeout if absent
  forks:
    extends: main             # deep overlay of main; null deletes an inherited key
    extra_body:
      reasoning_budget_tokens: 2048
  embedding:                  # optional; absent means no embedding client
    kind: embedding           # required for an embedding profile
    base_url: "..."
    model: "..."
    dimensions: 768           # required
    # same keys as main
  defaults:                   # optional: kind -> default profile
    chat: main                # these two are the fallbacks when omitted
    embedding: embedding
  stages:                     # optional: stage -> profile
    compaction: forks
    consolidation: forks
```

`defaults` and `stages` are reserved names; a profile can't use them. A
profile block that is `null` is treated as absent.

**`kind` and `extends` (`merge_profile`).** `kind` is `chat` or `embedding`.
An omitted `kind` means `chat`, applied after `extends` is merged, so a child
of an embedding profile stays an embedding profile. `extends: <profile>` lays
the block over the named profile: mappings merge recursively, lists and
scalars replace, and a child's `null` deletes the inherited key (unlike
`runtime.deep_merge`, where `None` is a no-op). Chains resolve root first. A
resolved profile must have `base_url` and `model` (a partial block that exists
only to be extended is not supported) and, for an embedding profile,
`dimensions`.

**Defaults.** `llm.defaults.<kind>` names each kind's default profile. When
it is absent, chat defaults to `main` and embedding to `embedding`, if those
exist. Only a default chat profile is required: `defaults.chat: foo` with no
`main` boots and sends main turns to `foo`. A default, configured or by
name, that names a profile of another kind is an error, which is how an
existing `llm.embedding` block without `kind: embedding` fails startup
instead of becoming a chat profile and silently dropping memory to keyword
search.

**Validation.** A malformed set of profiles aborts startup and changes nothing
on reload. Each error names the key:

| Case | Error |
|---|---|
| unknown base | `llm.<name>.extends names unknown profile '<base>'; defined profiles: ...` |
| cycle | `llm.<name>.extends forms a cycle: a → b → a; remove one extends` |
| unknown kind | `llm.<name>.kind must be one of chat, embedding; got '<v>'` |
| embedding, no dimensions | `llm.<name>.dimensions is required for an embedding profile ...` |
| missing endpoint | `llm.<name>.base_url is required (after extends)` (also `model`) |
| not a mapping | `llm.<name> must be a mapping; got <type>` (also `llm.defaults`, `llm.stages`) |
| bad default | `llm.defaults.<kind> names unknown profile '<p>'`; `... names '<p>', a <k> profile`; `llm.defaults: unknown kind '<k>'` |
| fallback of the wrong kind | `llm.<fallback> is a <k> profile but is the default <kind> profile by name; add kind: <kind> to it, or set llm.defaults.<kind>` |
| no chat default | `no default chat profile: define llm.main or set llm.defaults.chat` |
| stage maps to a missing profile | `llm.stages.<stage> names unknown profile '<p>'; defined profiles: ...` |
| default or stage value not a string | `llm.defaults.<kind> must be a profile name; got [...]; defined profiles: ...` (likewise `llm.stages.<stage>`) |
| stage maps to the wrong kind (`on_start`) | `llm.stages.<stage> maps a <k> stage to '<p>', a <k2> profile; map it to a <k> profile` |

An `llm.stages` entry that names no registered stage logs a WARNING and is
ignored, because a disabled plugin's stage is expected there. A kind with no
default but with stages that would use it logs a WARNING at startup, naming
those stages: they have no client and degrade.

**Fork budget.** `agent.forks.reasoning_budget_tokens` is gone. A fork
carries a budget only if the profile its stage maps to has
`reasoning_budget_tokens` in `extra_body`. The body a fork sends is ordered
in Forks, "Body precedence".

**Graceful degradation:** with no default chat profile `parse_llm_config`
raises at startup. A missing embedding profile is not an error: memory falls
back to keyword search.

## ToolCollectionPlugin

`tool_collection.py` — collects tools from all plugins at startup. Registered as
the `"tools"` entry point. Declares `depends_on = set()`. Its `on_start` is
decorated with `@hookimpl(trylast=True)` so it fires after all other `on_start`
hooks have run, ensuring every tool-providing plugin has had a chance to finish
setup.

### Lifecycle

- `on_start` — reads `tools.max_result_chars` from config (default 100,000).
  Falls back to `agent.max_tool_result_chars` with a deprecation warning.
  Calls `pm.hook.register_tools(tool_registry=collected)` (sync broadcast) to
  collect tools from all plugins, then builds a `ToolRegistry`. Tools whose
  name is in `tools.disabled` are dropped from the registry here — the
  single chokepoint every registered tool passes through, so there are no
  per-plugin opt-outs. The list is validated in `on_init` (a non-list
  raises naming `tools.disabled` and the value) and widened to coupled
  tools, so `shell` and `task_pipeline` go together. Per-channel narrowing
  happens downstream in `Agent`; see "Per-channel tool sets". After the
  first registry build it logs one `WARNING` for each `tools.timeouts` entry
  naming no registered tool (the entry is kept: an MCP tool registers when its
  server connects).
- `on_init` — also parses `tools.call_timeout` and `tools.timeouts` with
  `parse_tool_timeouts` into `call_timeouts`. `Runtime.start` and the config
  watcher validate the same keys first, so a bad value is a `Config error` at
  startup and a refused reload.

### Tool access

`ToolCollectionPlugin.get_tools()` returns `(tools_dict, tool_schemas)` for the
agent loop. `ToolCollectionPlugin.get_registry()` returns the full `ToolRegistry`
for inspection or filtering (e.g., `exclude("subagent")`).

`ToolCollectionPlugin.max_result_chars` is a public attribute read by `Agent` and
`SubagentPlugin`. `ToolCollectionPlugin.call_timeouts` (a `ToolTimeouts`) is
read by both too and passed to `dispatch_tool_call`.

### Configuration

```yaml
tools:
  max_result_chars: 100000   # truncation limit for tool result strings (default 100_000)
  disabled: [write_file]     # dropped for every channel; shell <-> task_pipeline coupled
  call_timeout: 600          # seconds bound on every tool call; 0 or null disables
  timeouts: {task_pipeline: 3600}   # per-tool override of call_timeout

# Legacy (deprecated):
agent:
  max_tool_result_chars: 100000   # use tools.max_result_chars instead
```

**Graceful degradation:** without `ToolCollectionPlugin`, `Agent.depends_on`
validation fails at startup.

## SchedulerPlugin

`scheduler.py` — the clock-driven producer: reminders and operator-declared
periodic jobs (#59; design rationale in
`plans/scheduler.md`). Entry point name: `"scheduler"`;
`depends_on = {"registry", "persistence", "clock"}`.

**Rows.** One `schedule` table on the persistence connection. A row is a
due time, an optional interval, a target channel id, the text, and an
origin (`reminder`; `heartbeat` is still accepted so old configs load, but
the heartbeat is now cognition's `HeartbeatPlugin`, started on system idle).
Reminders come from the `remind_me`
tool and are pinned to the calling channel. Jobs come from
`scheduler.jobs` in `agent.yaml` and are synced by `name`.

```sql
CREATE TABLE schedule (
  id INTEGER PRIMARY KEY,
  name TEXT UNIQUE,            -- config jobs only; NULL for tool-created rows
  channel_id TEXT NOT NULL,    -- "transport:scope", the persistence key
  origin TEXT NOT NULL,        -- 'reminder' ('heartbeat' accepted for old configs)
  text TEXT NOT NULL,          -- the objective, as the agent or operator wrote it
  next_fire_at REAL NOT NULL,  -- unix seconds, UTC
  interval_seconds REAL,       -- NULL means one-shot
  created_at REAL NOT NULL,
  created_by TEXT NOT NULL,    -- 'config' | 'tool'
  last_fired_at REAL,
  enabled INTEGER NOT NULL DEFAULT 1,
  disabled_reason TEXT,        -- 'fired' | 'cancelled' | 'missed' | 'removed_from_config'
  schedule_spec TEXT           -- config jobs: canonical every/at/timezone, to detect changes
);
CREATE INDEX schedule_due ON schedule (enabled, next_fire_at);
```

There is no firing log: a firing is a persisted conversation row, and its
`annotations["schedule_id"]` travels with the trace.

**Timer.** One asyncio task sleeps until the earliest enabled row is due
(at most an hour, so a wall-clock jump is noticed) with `Clock.sleep`, and
reads time from `Clock.now`. An `asyncio.Event` wakes it early whenever a
row is added, cancelled, or re-synced. Before firing anything it waits
until the persistence DB is open and `Agent.started` is True. Tests
register a fake clock and never sleep.

**Firing.** The channel is resolved first: `registry.get`, then the
`resolve_channel` hook, then `registry.get_or_create` from the split id.
The last step is skipped for a transport that implements `resolve_channel`,
meaning a plugin whose `transport` attribute is the id's prefix. For that
transport, None means "not yet": the transport cannot apply the channel's
config, for example Signal before its first alias resolution (PRD §3.2.7,
#76). The row is then left untouched and due, its overdue note is kept, and
the pass re-arms after `CHANNEL_RETRY_SECONDS` (5 s). A WARNING is logged
once per row, not once per retry. Once the channel resolves, the row is
advanced (recurring) or disabled as `fired` (one-shot), at most once.
`channel.touch()` then runs, which only updates `last_active` and with it the order of the web and ACP channel lists. The plugin then calls:

```python
await pm.ahook.on_notify(
    channel=channel, source=row.origin, text=framed_text,
    tool_call_id=None, annotations={"origin": row.origin, "schedule_id": row.id},
)
```

From there the normal pipeline runs: serial queue, persistence,
origin-keyed consumers, attribution, and `send`, whose reply on the
row's channel is the unprompted message. A job on a channel no
transport owns (such as `internal:notes`) reaches the log and persistence
only, because `TransportPlugin.send` delivers only to the transport that
owns a channel. The originating notification resets `turn_counter`
(step 3 of [Message processing](#message-processing)).

**Framing and role.** Reminder text is framed as data:

> A reminder you set on 2026-10-23 14:02 (PDT) for this conversation is
> due. Reminder text: «...». Decide whether and how to act on it.

`«` and `»` inside the text are replaced, so the text cannot close the
quote. Config jobs read `Scheduled job "<name>" (set by the operator,
every <interval>) is due. Job prompt: «...». ...`. A late catch-up firing
is prefixed `[overdue by 42m, daemon was down]`. The row is stored as
`role: system` with the `[reminder]` prefix like any
notification, and rendered under `agent.mid_conversation_system_role`
(`user` on templates that reject a non-leading system message). No role
parameter was added to `on_notify`.

**Recurrence.** A fixed interval with an optional anchor (`at`: `HH:MM`,
or an ISO datetime for a weekday anchor). The next slot is computed from
the scheduled time, not the fire time, so there is no drift; a row more
than one interval behind skips to the next future slot. Steps are taken in
local wall-clock time in `scheduler.timezone`, so `every: 1d` at 08:30
stays at 08:30 across DST.

**Startup.** `on_start` (or the timer's first pass, if the DB was not
open yet) syncs config jobs and then applies the catch-up policy. A
one-shot overdue by less than `catch_up_window` fires once with the stale
prefix; an older one is disabled as `missed`. A recurring row less than
one interval behind fires once with the prefix; one further behind skips
ahead without firing. Startup never fires directly; the timer does, once
it is ready.

**Job sync.** On `on_start` and `on_config_reload`, by `name`. A new name
is inserted. An existing name updates text, channel, and origin, and
recomputes `next_fire_at` only when `schedule_spec` changed or the row
was disabled. A name no longer configured is disabled as
`removed_from_config`, a durable stop. An invalid entry is logged at
ERROR and its existing row is left alone, so a typo does not stop a
running job. A reload wakes the timer.

**Tools.** `remind_me(when, text)`, `list_reminders()`, and
`cancel_reminder(id)` act on `_ctx.channel` only and take no channel
argument. The tool, not the model, parses `when` against
`scheduler.timezone`: a relative duration (`20m`, `1h30m`), ISO 8601 with
or without an offset, or bare `HH:MM` for the next occurrence; a bare
number is refused as ambiguous. It returns the resolved absolute time.
`scheduler.max_reminders_per_channel` caps pending tool rows per channel,
and text is capped at 2000 characters. `cancel_reminder` refuses config
jobs, and treats another channel's row as not found.

**Shutdown.** `on_stop` cancels the timer task; firings already enqueued
drain with the channel queues.

**Not in phase one:** a `send_to_channel` tool, recurring reminders by
tool, cron syntax, hook-kind jobs (see `plans/scheduler.md`).

## Subagent Tool

`tools/subagent.py` — `SubagentPlugin` registers the `subagent` tool, which
launches a background agent using the shared LLM client from `LLMPlugin` and
the full tool set minus `subagent` itself and minus whatever the calling
channel's `channels.<id>.tools.disabled` removes (#149).

### Tool signature

```python
async def subagent(instructions: str, description: str, _ctx: ToolContext) -> str
```

`_ctx` is system-injected (excluded from the LLM-visible schema). Returns a
confirmation string with the enqueued task ID on success, or an error string
if the task queue or channel context is unavailable.

### How it works

Declares `depends_on = {"llm", "tools"}`.

1. At tool-call time (not `on_start`), retrieves `ToolCollectionPlugin` via
   `get_dependency(self.pm, "tools", ToolCollectionPlugin)` and reads
   `max_result_chars` and the tool registry from it.
2. Calls `registry.exclude("subagent")` to remove itself from the subagent's tool set.
3. Retrieves the shared `LLMClient` for the `subagent` stage from `LLMPlugin` via
   `get_dependency(self.pm, "llm", LLMPlugin).get_client("subagent")`. The
   subagent does NOT start or stop this client — lifecycle is owned by `LLMPlugin`.
4. Builds a `messages` list (system prompt + user instructions) and a
   `work` coroutine that calls `run_agent_loop` with `channel` and `task_queue`
   from `_ctx` (enabling the subagent to enqueue nested tasks), then strips
   thinking from the result.
5. Wraps `work` in a `Task` with `tool_call_id` from `_ctx`, enqueues it on
   `ctx.task_queue`.
6. Returns immediately; result is delivered via `TaskPlugin → on_notify →
   Agent` (same path as all other tasks).

### LLM configuration

Uses the `LLMClient` for the `subagent` stage from `LLMPlugin`: the profile
`llm.stages.subagent` names, else the default chat profile (`llm.main` unless
`llm.defaults.chat` says otherwise). The client is retrieved at tool-call
time, not captured at `on_start`. Runtime overrides don't reach a subagent,
so the profile's `extra_body` applies as written.

### System prompt

The `SUBAGENT_SYSTEM_PROMPT` constant (defined in `tools/subagent.py`) is
used as the subagent's system message. It instructs the subagent to work
step-by-step and summarize results on completion.

## McpClientPlugin

`mcp_client.py` — connects to external MCP servers during `on_start`, caches
their tool lists, and exposes them to the agent loop via `register_tools`.

### Lifecycle

- `on_start` — reads the `mcp.servers` config block, iterates each server, and
  calls `_connect_server` for each. Each server's transport and session are
  owned by a long-lived task (see "Owner tasks") so they stay alive for the
  daemon's lifetime. After all connections, calls `_build_tool_list()` and caches the
  result. Connection failures per server are caught, logged as warnings, and
  skipped; a failed server does not abort startup. When at least one server
  connected, it then calls `ToolCollectionPlugin.rebuild_registry()` (#405).
  The `on_start` broadcast runs implementations concurrently, and `trylast`
  on `ToolCollectionPlugin.on_start` orders when it starts, not when it
  finishes, so the first registry build usually runs while the connects are
  still in flight and collects no MCP tools. The rebuild adds them before the
  broadcast returns, so `Agent.on_start`, which `Runtime` runs after the
  broadcast, sees them. A failed rebuild logs a WARNING naming the servers
  and does not abort startup.
- `register_tools` — extends `tool_registry` with the cached `Tool` instances
  (sync). It runs on every `rebuild_registry`: the tools plugin's own build
  in `on_start` (possibly before any server has connected, which gives an
  empty list), the rebuild at the end of this plugin's `on_start`, and the
  rebuild after a retry reconnects a server.
- `on_stop` — signals every server's owner task to close its session and
  transport, and awaits them all together (see "Owner tasks").

### Transports and failure handling

A server connects over stdio, SSE or streamable HTTP, chosen by its
`transport` key. Each server has its own exit stack and a bounded handshake
(see "Connect timeout"), so one server's failure never touches another.
Servers that did not connect at startup are queued in `_pending_servers` and
retried by a background `_retry_loop` task (started in `on_start` only when
something is pending) that sleeps on the kernel clock (`depends_on` "clock")
through the transports' ladder: 10 s, doubling, capped at 300 s, reset after a
round that reconnects something, exit when nothing is pending. Every MCP tool
call also runs a retry round as a fast path. One `asyncio.Lock` serialises
rounds, and a tool call that finds it held skips rather than wait out a
handshake, so the loop and a tool call never connect a server twice. A
successful retry rebuilds the cached tool list and the shared registry, and the
running Agent re-reads it on its next turn (#109). Config errors (any exception
from `_parse_server_config`, which runs before any transport opens:
`McpHeaderEnvError`, `ValueError` for a wrong type or non-finite timeout,
`KeyError` for a missing key) are reported once, naming
`mcp.servers.<name>.<key>` and the value, and never queued. A malformed
top-level block (`mcp` not a mapping, `mcp.servers` neither null nor a mapping)
is caught by `validate_mcp_config`, which `Runtime.start()` runs with the other
config checks (skipped when `mcp` is in `plugins.disabled`) and `on_init` runs
again, so it exits with a `Config error` instead of an AttributeError.
`on_stop` sets a stopping flag, cancels the loop, then
stops the owners; a connect still in flight (loop or tool call) that finishes
after the flag is set tears its owner down instead of registering it.

### Owner tasks

anyio requires a cancel scope to exit in the task that entered it, and
apluggy's `gather` runs `on_start` and `on_stop` in different tasks (a shared
`AsyncExitStack` closed from `on_stop` raised "Attempted to exit cancel scope
in a different task", #213). So `_connect_server` starts one `_ServerOwner`
task per server (`_run_owner`), which enters the transport, HTTP client and
`ClientSession` on its own `AsyncExitStack`, runs the handshake under
`asyncio.timeout`, and resolves a `ready` future with the server's state (or
the exception). It then waits on its `stop` event and closes the stack in the
same task. Owners live in `self._owners` (name to owner).

`on_stop` sets every owner's `stop` event at once and awaits all the tasks
together with `asyncio.wait`, bounded by `_STOP_BOUND_SECONDS` (2.5 s, inside
Runtime's default 3 s stop budget, `daemon.shutdown_timeout`). The SDK's `stdio_client` can take about 2 s to
terminate a process, so closing serially would blow the budget. An owner
still running at the bound is cancelled and logged at WARNING with the server
name and a "may need to be killed by hand" remedy. An exception while closing
is logged by the owner at WARNING with the server, the exception type (the
leaves of an exception group) and the traceback, and does not stop the other
servers. A `CancelledError` delivered to `on_start` stops and awaits the
owners already started before it propagates, and a cancelled or failed
mid-connect owner is awaited before `_connect_server` returns.

A transport can cancel its own owner. When a streamable-HTTP POST fails
(say `ConnectError` from an unreachable server), the SDK's anyio task group
cancels its host scope, so the owner sees a `CancelledError` while its stack
is still open. Once the stack is closed, `asyncio.current_task().cancelling()`
tells the two apart: anyio uncancels its own cancel when its scope exits,
leaving 0, while a cancel from outside leaves it at 1 or more. With 0 and an
exception from the teardown, which is the real failure
(`ExceptionGroup[ConnectError]`), the owner resolves `ready` with that
exception, and the server is skipped and queued like any failed connect
(#385). As a backstop, `on_start` and `_retry_round` treat a `CancelledError`
that arrives while their own task isn't being cancelled as a connect failure,
and re-raise a real cancel. The same test covers a `BaseExceptionGroup`
(`_is_connect_failure`, #596): a group is not an `Exception` when a member is
a `CancelledError`, so it is judged by its leaves. A group of failures and
uncancelled cancels is a failed connect, skipped and queued like any other; a
group around a real cancel propagates, and `on_start` still stops the owners
already started; a group holding `KeyboardInterrupt` or `SystemExit` always
propagates. The connect-failure warnings name the server, its endpoint (URL or
command) and the unwrapped exception types, and say the background retry will
try again. A line-boundary character in that endpoint (a URL, or a stdio command) is
percent-encoded, so one warning is one log line (#588).

### Tool naming

Each tool is named `{tool_prefix}__{mcp_tool_name}`. `tool_prefix` defaults to
the server name if not set. Example: server `files`, tool `read` →
`files__read`.

### Tool name collision

`_build_tool_list` tracks seen prefixed names. If two servers expose a tool
with the same prefixed name, the second is skipped and a warning is logged
(first-wins).

### Schema sanitization

`_mcp_tool_to_schema` strips the keys `$schema`, `$id`, `$comment`, `$defs`,
and `definitions` from the top level of `mcp_tool.input_schema` before wrapping
it in the OpenAI function-call envelope. Sanitization is shallow (top-level
only); nested occurrences in `properties` are not removed.

### Configuration

```yaml
mcp:
  servers:
    <name>:
      transport: stdio | sse | streamable_http
      command: <executable>   # stdio only
      args: [...]             # stdio only
      env: {...}              # stdio only, optional
      url: <endpoint>         # sse and streamable_http
      headers: {...}          # sse and streamable_http, optional; ${VAR} from the environment
      tool_prefix: <prefix>   # optional; default: server name
      timeout_seconds: 30     # optional; per tool call; default: 30; above tools.call_timeout the call is cut at call_timeout
      connect_timeout_seconds: 30  # optional; handshake bound; default: 30
```

### Connect timeout

The owner task opens each server's transport and session on its own
`AsyncExitStack`, inside `asyncio.timeout(connect_timeout_seconds)` (the
timeout wraps the handshake inside the owner, not the `ready` wait, so
expiry cancels the handshake itself and the subprocess dies with it). A
server that never answers `initialize` therefore cannot hold up the
`on_start` broadcast (#72). On success the owner keeps the stack open until
`on_stop`. On a timeout or any other failure the owner closes it
immediately, which terminates a stdio subprocess, and `_connect_server`
awaits that teardown before returning. On a timeout the owner logs it once
and `_connect_server` returns without registering the server; any other
failure propagates to `on_start`, which logs and skips it. The close runs
after the timeout has turned its cancellation into `TimeoutError`, so
teardown is not itself cancelled.

### HTTP transports and headers

`streamable_http` uses the SDK's `streamable_http_client` with an HTTP client
built by `create_mcp_http_client(headers=...)`. The SDK closes only a client
it created, so the plugin enters its client into the owner's stack, and it
closes with the server. `sse` passes the same headers to `sse_client`.
`_resolve_headers` replaces each `${VAR}` in a header value from
`os.environ` before anything opens. An unset or empty variable, or one holding CR, LF or NUL (which httpx would reject with an error quoting the value), raises
`McpHeaderEnvError` (a `ValueError`), whose message names the server, header
and variable but never a value. `on_start` logs it once and skips the server
without adding it to `_pending_servers`: the daemon's environment can't
change, so a retry would fail the same way.

The client targets the mcp 2.x API (`mcp>=2.0`): tool schemas are read from
`Tool.input_schema` and error results from `CallToolResult.is_error`.

**Graceful degradation:** without `McpClientPlugin`, no MCP tools are
registered. The plugin declares no `depends_on`.

## RuntimeSettingsPlugin

`tools/settings.py` — registers the `set_settings` tool, which allows the
agent to update per-channel LLM inference parameters and framework parameters
at runtime via tool call.

### Tool signature

```python
async def set_settings(settings: dict, _ctx: ToolContext) -> str
```

`_ctx` is system-injected (excluded from the LLM-visible schema).

### Parameters

The `settings` dict is intended for:

- LLM inference keys (`INFERENCE_KEYS`, `channel.py`): `temperature`,
  `top_p`, `top_k`, `frequency_penalty`, `presence_penalty`, `max_tokens`,
  `seed`
- Framework keys (`TUNABLE_FRAMEWORK_KEYS`, `tools/settings.py`):
  `max_turns`, `max_context_tokens`, `reasoning_history`
- Dotted plugin tunables (`critique.*`, `appraisal.*`; see configuration.md)

Undotted keys are allowlisted (GitHub issue #206): any undotted key outside
the first two lists rejects the whole call, with no overrides applied, a
`null` included. The request body accepts keys that replace the
conversation (`messages`), repoint the server (`model`) or rewrite the
tool list (`tools`, `tool_choice`), and a blocklist can't name every key a
server honors; see [threat-model.md](threat-model.md), finding 1. As a
second line, `Agent._llm_overrides` forwards only `INFERENCE_KEYS` into
the request body. Values are not validated, except `reasoning_history`,
which is rejected unless it is `drop`, `exchange`, or `keep` — a stored bad
value would make every later config resolve raise.

Pass `null` for a key to clear that override and revert to the static config value.

### Return value

On success: a confirmation string listing current overrides (or noting that all
overrides have been cleared). On blocked keys: an error string naming the
blocked keys. On keys outside the allowlist: an error string naming them
and listing the accepted keys.

### Blocklist

`system_prompt` is always blocked regardless of operator config. Additional
keys are blocked via `agent.immutable_settings` in `agent.yaml`:

```yaml
agent:
  immutable_settings:   # keys the agent must not change (system_prompt always added)
    - max_turns
```

The blocklist is checked first, then the allowlist above, so it only
narrows what the allowlist accepts. Its main use is the dotted gate keys
(configuration.md, "Runtime-tunable gate parameters").

### Persistence

Changes apply to `channel.runtime_overrides` in memory only. They are lost on
daemon restart; no DB persistence.

**Graceful degradation:** without `RuntimeSettingsPlugin`, the `set_settings`
tool is not registered. The plugin declares no `depends_on`.

## Tools

Registered by three plugins in `tools/__init__.py`, each with its own entry
point and its own `tools.*` keys, so each can be dropped by name in
`plugins.disabled` (#16):

- `ShellToolPlugin` (`shell_tools`): `shell` and `task_pipeline`. The two go
  together because `task_pipeline` runs each step through the shell (PRD
  §3.5.3). Without it the agent cannot run commands (an MCP server can
  still offer its own).
- `FileToolsPlugin` (`file_tools`): `read_file` and `write_file`. Without it
  the agent has no dedicated file tools; `shell`, if enabled, can still
  read and write any file the daemon user can. A notification too long for the
  window still gets a spill file and a pointer while `shell` is enabled
  (another spill reader); with neither, it is cut and no file is written.
- `WebToolsPlugin` (`web_tools`): `web_fetch` and `web_search`; owns the
  `aiohttp.ClientSession` that `web_fetch` reuses. Without it the agent
  has no dedicated web tools; `shell` (`curl`) or an MCP server can still
  reach the network. To take a capability away entirely, disable every
  plugin that provides it (see [threat-model.md](threat-model.md)).

They replaced the combined `CoreToolsPlugin` (`core_tools`), with no alias.

| Tool | File | Purpose |
|------|------|---------|
| `shell(command)` | `tools/shell.py` | Execute shell command (declares `reads_spill_files`), 30s timeout (configurable). On cancellation (the call bound, shutdown) it kills and reaps its `/bin/sh` process, then re-raises; `task_pipeline` inherits this. Grandchildren of the shell are not killed. Its companion `run_shell` returns the exit code beside the output, for callers that must know whether the command succeeded (#105). |
| `read_file(path, offset, length, start_line, end_line)` | `tools/files.py` | Read file (declares `reads_spill_files`; a missing path in the spill directory gets the stale-spill error; 1MB limit, configurable), or a character range (`offset`, `length`) or a line range (`start_line`, `end_line`, 1-based, inclusive) of any size behind a header naming the range, total and next position; the two kinds cannot be combined; a range is clamped to `tools.max_result_chars`, a line range backing up to the last whole line (a first line longer than a page is cut, and the header gives the character offset to continue from and the start line where line paging resumes) |
| `write_file(path, content)` | `tools/files.py` | Write file, creates parent dirs |
| `web_fetch(url)` | `tools/web.py` | Fetch URL, 15s timeout, 50KB truncation (configurable) |
| `web_search(query, max_results)` | `tools/web.py` | Search via DuckDuckGo, default 8 results (configurable) |
| `task_pipeline(definition)` | `tools/task_pipeline.py` | Execute YAML/JSON task DAG with dependency resolution. A step whose command exits non-zero is marked failed and its dependents do not run; a timeout leaves the step done, since the model decides whether to retry (#105). |

Additionally, `SubagentPlugin` registers `subagent` and `TaskPlugin`
registers `task_status` as tool closures during `on_start`.

| Tool | File | Purpose |
|------|------|---------|
| `subagent(instructions, description)` | `tools/subagent.py` | Launch background subagent using shared LLM client |
| `task_status()` | `task.py` | Report task queue status |
| `task_result(task_id)` | `task.py` | Return a completed task's stored result by id (in-memory deque, then the JSONL file when persistence is enabled) |

`SchedulerPlugin` registers three tools scoped to the calling channel
(see [SchedulerPlugin](#schedulerplugin)):

| Tool | File | Purpose |
|------|------|---------|
| `remind_me(when, text)` | `scheduler.py` | Create a one-shot reminder on this channel; returns the resolved time |
| `list_reminders()` | `scheduler.py` | Pending rows on this channel, config jobs flagged read-only |
| `cancel_reminder(id)` | `scheduler.py` | Cancel a tool-created reminder on this channel |

## Transports

Every transport (CLI, IRC, Signal, Discord, Zulip, web, ACP) subclasses
`TransportPlugin` (`hooks.py`) and sets `transport` to the name its
channel ids start with. A chat transport also sets
`conversational = True` (IRC, Discord, Signal and Zulip do; the default is False),
which a `transform_system_prompt` wrapper can read to tailor the prompt to chats. The base class implements the outbound `send`
hook once: it drops events for channels the transport does not own
(`owns(channel)`, by default `channel.matches_transport(self.transport)`;
`WebPlugin` narrows it to also require that it is enabled), then
dispatches by event type to the plain methods `send_message`,
`send_thinking`, `send_tool_status` and `send_progress`. Those take the
channel and an event from `outbound.py` and default to no-ops, so a
transport overrides only what it can display; the per-transport sections
below name them as that transport's methods. The base class also declares
`depends_on = frozenset({"registry"})`. Inbound hooks
(`should_process_message`, `on_message_admitted`/`rejected`/`persisted`)
are not filtered by the base class; each transport's implementations open
with their own ownership check.

### CLI (`channels/cli.py`)

`CLIPlugin` — stdin/stdout transport. Routes to `cli:local` channel.
Reads stdin line-by-line in a background task. Prints responses to
stdout with latency timing.

EOF ends the session at an interactive terminal (Ctrl-D), and always
under `corvidae cli`, which is a session by definition (`corvidae cli <
script.txt` exits when the script runs out): SIGINT to the process for a
graceful shutdown. Under `corvidae serve`, EOF on a non-TTY stdin
(systemd's `/dev/null`, a closed pipe) stops just the CLI transport with a
warning and the daemon keeps running; with no stdin at all the read loop
never starts. So a service-managed config may keep a `cli:` channel. The
banner and `> ` prompts print only when stdin is a terminal.

`Runtime.run` installs its SIGINT handler before startup, so a SIGINT
that arrives while plugins are still starting (this EOF path, or a Ctrl-C
during startup) is held until startup finishes, then shuts down normally;
it logs that it is waiting, and a second interrupt force-exits. SIGTERM
keeps its default action until startup finishes, so `systemctl stop`
during a hung startup ends the process at once.

### IRC (`channels/irc.py`)

`IRCPlugin` — IRC transport via pydle. Connects with exponential
backoff retry (10s→300s cap); each retry WARNING names the host, port and
nick it failed to reach. Joins configured channels on connect.
Forwards both channel and private messages. `split_message` splits
outgoing text into 400-byte chunks preserving paragraph/sentence
boundaries.

Sender authorization (#77) is default-deny, as for Signal and Discord.
`irc.allow` is validated in `on_init` (not a list of strings raises, naming
the field) and the plugin's `should_process_message` refuses every sender
not on it, in channels and PMs, with one INFO line and no reply; a missing
`allow` logs a startup WARNING. Where pydle has account data (IRCv3
`account-notify`/`account-tag`/`extended-join` negotiated, or an account
already known for the sender), entries match the services account and an
unidentified or unknown sender is refused; with no account support they
match the nick. The allowlist is checked before the channel is resolved
(as in Signal, #103): a refused sender gets an unregistered `Channel` (or
an existing registry entry), so nothing is added to the registry and no
`message_log` rows or memory follow. With `account-tag` negotiated,
`IRCClient.on_raw` captures the PRIVMSG's own `account` tag (none means not
logged in) into a task-local `ContextVar` before any `await`; each `on_raw`
runs in its own task, so the value survives pydle's WHOIS suspensions.
`IRCPlugin._decide` reads only that value and never `users[nick]`, which is
shared across lines and which WHOIS replies rewrite for whoever holds the
nick (on a network without WHOX, an unknown nick makes pydle WHOIS it
mid-message). There is no nick fallback in that mode. `on_raw` also
overwrites the pydle record from the tag, only to keep pydle consistent. The
verdict is decided once per message, synchronously, in
`IRCPlugin._dispatch` (before any `await`) with a single account lookup, and
recorded by the message's transport-minted `trace_id`.
`should_process_message` only reports that recorded verdict (no verdict
means deny) and the entry is dropped when `on_message` returns, because
pydle runs a task per line and a later line's `on_raw` can rewrite
`users[nick]` while an earlier message is still in the hook pipeline.
Without `account-tag`, the account is whatever `users[nick]` holds when the
message arrives (pydle's tracked state; a residual risk for senders who
share a channel with the bot). `_dispatch` then applies one more synchronous
step to every message, PM or channel: if the sender would be admitted, an
account is tracked, `account-tag` is not negotiated and
`_shares_channel(nick)` (a casemapping-folded scan of pydle's
`channels[*]["users"]`) is false, the verdict becomes a refusal carrying the
`_UNVERIFIABLE_ACCOUNT` reason (#264). `_decide` returns
`(allowed, account, reason)` with `reason` None; the recorded verdict is the
same 3-tuple and `should_process_message` appends the reason to its INFO
line. `IRCClient.on_connect` calls `_warn_unverifiable_accounts()` once per
connection: one WARNING when `irc.allow` is set, `account-tag` is not
negotiated and `account-notify` or `extended-join` is. Entries and
senders are compared under the server's CASEMAPPING via the client's
`normalize`.

While the client is disconnected (including the reconnect backoff, up to
300s), outbound text is not dropped: it goes into a bounded in-memory queue
(`_OUTBOUND_QUEUE_MAX`, 50, a module constant, not configurable). Each queued
message logs a WARNING with the channel id, length and queue depth. When the
queue is full the oldest entry is dropped, with a WARNING naming its channel
and the number dropped so far in the outage. After a successful connect,
`on_connect` joins the configured channels and then flushes the queue in
order, splitting each message exactly as a connected send does. A send made
while the queue is non-empty queues behind it, so nothing overtakes older
output. Sends also queue from TCP connect until `on_connect` has finished its
joins: `IRCPlugin._ready` is set there and cleared in `on_disconnect`, so a
reply in that window is not sent before the bot is in the channel (#251). Limits: a connection that drops mid-flush leaves the unsent entries
queued, and a partly sent multi-line message is resent whole (duplicates
rather than losses); pydle's `join` does not await the server's JOIN, so a
channel that refuses the join drops the flushed text server-side; and the
queue is lost at shutdown (logged as a WARNING with the count).

`split_message` itself lives in `channels/split.py`, not in the IRC
plugin, so a transport can size its own chunks without importing IRC.

Its contract: `max_len` is in UTF-8 bytes and must be at least 6, or it
raises `ValueError` regardless of the text (so a bad limit is not
input-dependent; a 6-byte floor leaves room for one 4-byte codepoint
after the 2-byte paragraph separator is re-attached). For any
`max_len >= 6`, every chunk is at most `max_len` bytes and
`"".join(chunks) == text`. The word tier carries each word's leading space
with it and cuts a word that cannot fit on its own on codepoint
boundaries, so nothing is dropped and nothing overflows. Earlier versions
could lose characters or exceed the limit on long unbroken runs and
multi-byte text. That affected IRC and Signal as well, and the same fix
applies to both. `irc.message_chunk_size` and `signal.message_chunk_size`
below 6 now raise `ValueError` when a reply is first split, not at startup.

### Signal (`channels/signal.py`)

`SignalPlugin` — Signal DM transport, speaking newline-delimited JSON-RPC
2.0 to an already-running `signal-cli daemon` over a unix socket. corvidae
connects to that daemon; it never spawns or supervises it. Registering the
account and running the daemon are covered in [signal-ops.md](signal-ops.md).

With no `signal:` block in config the plugin loads, resolves its registry
dependency, and does nothing else — no socket, no channels, nothing logged
above DEBUG. A block that is present but missing `socket` or `account`, or
whose `allow` is not a list, raises in `on_init` and aborts startup.
Silence is for absence; errors are loud.

**Connection.** `on_start` launches two long-lived tasks: a connection loop
and a notification processor. The connection loop opens the socket and
reads frames until EOF, retrying on the same ladder IRC uses — 10s,
doubling, capped at 300s, reset to the initial delay after any successful
connect. A disconnect fails every outstanding request with
`ConnectionError` rather than leaving it hung, and a malformed frame is
logged and skipped rather than killing the connection.

Frames are routed by shape: one carrying a `method` is an unsolicited
notification and goes onto a queue; one carrying an `id` and no `method` is
a response and resolves the future its request is awaiting. Notifications
are never handled inline in the read loop. Handling one sends further
JSON-RPC requests (typing, receipts) and awaits their responses, and those
responses can only arrive by the read loop reading the next line —
inline handling would deadlock the transport against its own request. The
single-consumer processor also keeps rapid-succession messages in arrival
order.

**Identity.** A channel's scope is the sender's ACI — `signal:<aci>` —
never their phone number. `sourceNumber` is absent whenever a sender has
phone-number privacy on, and channel ids are the persistence key for
`sessions.db`, the jsonl logs, and memory; keying on a field that can
vanish would fork one sender's history into two.

Config may still name senders by E.164 number, both in `allow:` and as
`channels:` keys. Those are resolved forward to ACIs by a single batched
`getUserStatus` call — batched because it is a CDS lookup with a
server-side rate limit — issued after each successful connect, never at
`on_init`/`on_start`, so an unreachable daemon at boot leaves the plugin
inert instead of blocking startup. A resolved alias is cached for the
process lifetime; one that failed to resolve is retried after the next
connect. Reverse ACI→number resolution is not used anywhere: signal-cli's
contact store answers an unknown ACI with a phantom entry carrying a null
number, which is indistinguishable from "not found".

**`resolve_channel`.** For a `signal:<aci>` id the registry does not hold
(a reminder due after a restart, before the sender has written again),
`SignalPlugin.resolve_channel` goes through the same alias path as an
inbound message under phone-number privacy: ACI as scope, no
`sourceNumber`, so an E.164-keyed `channels:` override is applied through
forward resolution. It returns None for ids it does not own. It also
returns None for its own ids while the override cannot be applied yet: before
`on_start` has resolved the registry, and while any E.164 `channels:` key
that is also in `allow:` is still unresolved and no resolution attempt has
finished. Forward resolution starts only after a connect, so before the
first connect the alias's ACI is unknown. A call that lands while a
resolution is in flight waits for it, bounded at
`ALIAS_RESOLUTION_WAIT_SECONDS` (5 s). The caller retries (the scheduler
holds the row, see SchedulerPlugin) rather than caching a default-configured
channel (#76). Once an attempt has finished, an alias that failed to resolve
stops blocking. A batch failure is retried only after the next connect, so
deferring until then could hold a reminder for the life of the connection.
The channel is created without the alias's config, and a WARNING names the
channel and the unresolved aliases. An E.164 `channels:` key missing from
`allow:` is never resolved, so it never blocks.

**Inbound filtering.** An envelope is dropped unless it carries a
`dataMessage` with non-empty text from someone other than the bot's own
account, outside a group. Dropping non-`dataMessage` envelopes is what
keeps a `syncMessage` — the account's own traffic echoed from another
device signed into it — from being read as input and driving the agent in
a loop. Attachments are downloaded under a size cap once the sender passes the
allowlist, written under `signal.attachment_dir` and named in the message text
(`[image: /abs/path]`); a refused sender's file is named, never fetched. Reactions, quotes, edits, and remote deletes all fall out
of the empty-text rule, since none of them carry a `message` field. An
envelope whose send timestamp is more than five minutes old gets a
`[sent YYYY-MM-DD HH:MM UTC]` prefix prepended, so backlog delivered after
a reconnect reads as backlog rather than as current.

**Outbound files.** `register_tools` offers `send_signal_attachment` when a
`signal:` block is present. It takes a `signal:<aci>` channel the registry
knows, a path and an optional caption, and sends one `send` RPC with the
file as a data URI (`data:<type>;filename=<name>;base64,<payload>`; the type is
guessed from the file name, falling back to `application/octet-stream`, and
`filename=` is the file's name; read and encoded in one thread call) under
`attachments` and the caption as `message`
(signal-cli's JSON-RPC takes a multi-value option as an array: v0.14.7
`signal-cli-jsonrpc(5)`). The path is resolved and refused if it lands
outside `signal.send_attachment_root` (default `state`), and the file is refused
above `signal.max_attachment_size`; the jail runs before any existence
check, so a missing path outside the root gets the root error. The
caption splits like any reply, the first piece riding with the file; the rest go
out through `_send_rpc`, not `_deliver`, because `_deliver` logs and swallows a
failure and the tool must not report success for a lost caption. A failure
returns a "Could not send:" string that says how far the send got: the file
unreadable ("cannot read ... check the path and its permissions"), nothing
delivered (retry), signal-cli not
connected, or the file delivered with caption part *i* of *n* failed (do not
resend the file). The name differs from Discord's `send_attachment` because the
tool registry keeps the last of two tools sharing a name while still advertising
both schemas. Known limits: a `send` result that reports a per-recipient
failure without a JSON-RPC error is not inspected, for files or text alike;
a file send is not recorded in the contact's history; and the file send has
been run against a live signal-cli 0.14.7 daemon only with a tiny plain-named
file, so large files and unusual file names (`;`, `,` are not escaped) are
unverified; `tests/test_signal_send_attachment.py` pins the wire shape. The target gate is
registry membership, as for Discord's tool and PRD §3.5.9 (nothing gates which
channel a turn may speak on): a channel is in the registry, for example, when an
allowlisted sender has written since the last start, when it is listed under
`channels:` (including a `signal:+E164` alias, which then receives the
file at the E.164 address, bypassing `resolve_channel`), or when a reminder or
scheduler job (or any other caller that resolves a persisted id) has resolved it since the start.

**Authorization.** `should_process_message` is a default-deny gate: the
sender's ACI must appear in `allow:` directly, or be what a configured
E.164 `allow` entry resolved to. A refusal is logged at INFO naming the
sender's ACI, and nothing is sent back. The transport fires `on_message`
for every decoded envelope regardless: the gate is the one place that
vetoes and the one place that logs the attempt. The transport checks its
allowlist before it resolves the channel, so a refused sender never
creates a registry entry (#103). Their message is gated on a transient
`Channel("signal", <aci>)` that is never registered, or on the live
entry if one already exists (a sender since dropped from `allow:`).
Otherwise anyone who knows the number could grow the registry by one
channel per distinct ACI.

**Liveness.** Two indicators, both gated on the allowlist — an
unauthorized sender gets no evidence that the account is live and
monitored.

- *Typing.* On the first outstanding message for a channel the transport
  sends `sendTyping` immediately, covering queueing delay, and starts a
  refresh loop that re-sends every 10s while the channel's outstanding
  count stays positive (Signal clients expire an unrefreshed indicator
  after several seconds; the exact figure is not verified).
  A final gives back *every* outstanding slot, whether it is sent
  (`send_message`) or withheld (`on_agent_response` when the final was
  unsent, #417): one final answers a coalesced burst (#398), so the count
  goes to zero, and when it was above zero the transport sends one explicit
  stop (below). A message still queued behind that final shows no
  indicator, through its own turn, unless another message arrives. The
  inbound handler gives back one slot for any message `on_message`
  returns without admitting: the transport mints the message's trace id, records its envelope
  timestamp under that id and takes the slot before calling `on_message`,
  and a `finally` releases both unless `on_message_admitted` fired for the
  id. Admission and rejection fire before `on_message` returns, so a gate
  veto, an agent that is not ready yet, or a raise or cancellation all land
  there. Only a message that took a slot (an allowlisted sender with an
  envelope timestamp) has an entry to release, so a stranger's refusal
  gives nothing back and no `on_message_rejected` implementation is needed.
  A refresh loop still asleep from an earlier
  turn is reused rather than doubled, so a channel has at most one, and
  `on_stop` cancels it. The release that takes the count from 1 to 0 —
  the last owed reply (a reply to a burst zeroes the count in one step), or
  a lone message that was vetoed or never admitted — sends `sendTyping` with `stop: true`, so the indicator clears at once
  instead of lingering until the client expires it (#390). A release at a
  count already at zero sends nothing. The count is decremented before the
  stop is awaited, and the refresh loop's count check and its frame write
  have no await between them, so a refresh can't land after the stop and a
  message arriving during the stop sends its start after it. A failed stop
  is logged at WARNING with the channel id and the indicator is left to
  expire; it never prevents a reply from being delivered. A liveness RPC
  waits at most `LIVENESS_RPC_TIMEOUT` (10 s) for signal-cli's response, so
  a daemon that accepts the frame and never answers costs one delay and a
  WARNING rather than holding the inbound processor or the reply path
  (#413). The bound covers typing and receipts only: `send` is not bounded
  this way, because a send that times out may have been delivered.
- *Read receipt.* Sent from `on_message_persisted` — processing start —
  not on arrival. A message can sit queued behind contended local
  inference, and a "read" marker followed by a long silence is worse than
  no marker. Envelope timestamps are held per channel keyed by trace
  id and popped by that id when the matching turn starts, so a vetoed or
  dropped message cannot shift a later receipt onto the wrong timestamp.
  `on_message_persisted` also fires for notification-originated turns,
  whose ids were never recorded, so those are a no-op. An admitted message
  whose turn dies before persistence (conversation init raised, the item
  was dropped) would otherwise keep its entry and its typing slot until some
  later reply cleared the count, and the slot would keep the indicator
  refreshing meanwhile (#104). The cleanup
  relies on ordering. Entries are inserted in enqueue order, and the
  channel's serial queue processes items in that order. So when an id is
  persisted, every entry recorded before it belongs to a message whose
  turn has already ended without persisting. Those entries are dropped
  without a receipt, their slots are given back, and each is logged at
  WARNING. If every turn on a channel fails before persistence, entries
  accumulate until the fault clears, and the next message that persists
  then sweeps them all. That
  `sendReceipt`'s `targetTimestamp` takes the inbound envelope's
  `timestamp` is convention, not documented by signal-cli — it matches
  `sendReaction`, `remoteDelete`, and `quoteTimestamp`, but it has not been
  confirmed against the upstream spec.

**Outbound.** `send_message` and `send_progress` both split through
`split_message` at `message_chunk_size` bytes and deliver every chunk —
text is never truncated, and chunks go out verbatim, without the newline
stripping IRC needs. `send_progress` uses the same delivery path but skips
the outstanding-count bookkeeping: it is mid-turn liveness, not the reply.
Replies and progress are text; a file goes out through the separate
`send_signal_attachment` tool (**Outbound files**, above).

Signal delivers a reply to a turn of any origin on a channel it owns. A
reminder firing on a `signal:` channel injects through `on_notify`, and
the reply leaves through the same `send` path as a reply to a person; it
needs no inbound message to answer. The outstanding-reply count is guarded
at zero for such a reply (`SignalPlugin._release_all` stops only a count
above 0), so a reply that answers no message neither errors nor drives the
count negative. The count can't tell which message a final answers, though:
any final, sent or withheld, a reminder's or a declining critique or
heartbeat turn's included, gives back every outstanding slot, so one that
lands while a person's message is outstanding clears the typing indicator
early (#101), and the release also sends the explicit
stop (#390). Since a reply to a coalesced burst already zeroes the count
(#398), the early clear is total rather than one slot. This is
[PRD §3.2.10](prd.md): a transport delivers every emission on a channel it
owns, whatever originated the turn, and liveness and reply bookkeeping
tolerate a reply that answers no outstanding message. Sending to a channel
*other than* the turn's own is a separate capability (PRD §3.5.9: operator
allowlist that starts empty, plus a budget); see the scheduler's "Not in
phase one: a `send_to_channel` tool" note under
[SchedulerPlugin](#schedulerplugin) and the deferred entry in
`docs/threat-model.md`.

Group conversations are out of scope, as is any privacy handling specific
to disappearing messages.

### Discord (`channels/discord.py`)

`DiscordPlugin` — Discord transport, built on discord.py's gateway client.
It logs in as a bot account and talks in DMs, server channels, and threads.
Creating the application and inviting the bot are covered in
[discord-ops.md](discord-ops.md).

With no `discord:` block in config the plugin loads, resolves its registry
dependency, and does nothing else — no client, no tasks, nothing logged
above DEBUG. A block that is present but has no token (neither `token` nor
the `DISCORD_TOKEN` environment variable), an `allow` that is not a list,
or a `message_chunk_size` outside 100–2000 raises in `on_init` and aborts
startup. Silence is for absence; errors are loud.

**Connection.** discord.py owns the gateway: heartbeats, resumes, and
reconnects within a session are its job, not ours. `on_start` launches two
long-lived tasks, a connection loop and an inbound processor. The loop
builds a new client for every attempt and closes it in a `finally`, so a
failed login never leaks its HTTP session. A `LoginFailure` (bad token) is
logged once at ERROR and not retried, since retrying a rejected token does
not improve it. Any other failure, or the client stopping on its own, is
retried on the ladder IRC and Signal use: 10s, doubling, capped at 300s,
reset to 10s when the client reports ready. The plugin stays loaded after
a login failure, but is deaf and mute until the daemon restarts.

**No privileged intent.** The client uses discord.py's default intents and
does not request the privileged Message Content intent, so operators need
not enable it in the Developer Portal. Discord still delivers message
content for DMs and for messages that mention the bot, and those are the
only messages this transport processes. An un-mentioned guild message in a
channel that requires a mention is dropped before the kernel sees it and
logged at DEBUG (PRD §3.2 item 2 allows the drop; `discord-ops.md` says what
the line carries). If a guild message mentions the bot and arrives with empty
content anyway, the transport logs a WARNING and drops it. That should not
happen, and the warning is how a change on Discord's side would show up. The client requests Message Content when any
channel answers unaddressed — the transport default (`discord.require_mention:
false`) or a `discord.channels` entry with `require_mention: false` — because
the intent is per client, not per channel. The mention check itself is per
channel: a channel that answers unaddressed skips it and the allowlist is the
only gate there, while every other channel stays mention-only. The bot's
mention token is stripped when present either way.

**Inbound filtering.** A message is dropped if its author is a bot (this
covers the bot's own messages and other bots', so two bots cannot talk each
other into a loop). In DMs every message from a human is processed. In
guild channels and threads the bot must itself appear in
`message.mentions`. `@everyone`, `@here`, and role pings do not count, even
though discord.py's `mentioned_in` would accept some of them. The bot's
mention token (`<@id>` or `<@!id>`) is stripped and the text is trimmed.

**Attachments.** Images attached to a message from an allowlisted sender are
downloaded after the allowlist check (never before it), streamed under
`discord.max_attachment_size`, and written to `discord.attachment_dir` with an
extension taken from the content type rather than the remote filename. The
message text carries `[image: /abs/path]` for each, `[image: download failed]`
for one that was over the cap or failed, and `[video attached]` /
`[document attached]` / `[sticker attached]` for kinds this transport does not
fetch. A message whose only content is an attachment is no longer dropped; a
refused sender's file is named and never fetched.

**Outbound files.** The `send_attachment` tool posts a file from
`discord.send_attachment_root` (default `state/`) to a channel the model names,
with the caption split like any other reply. The path is resolved and refused
if it lands outside that root, and the file is refused above
`discord.max_attachment_size`.

**Starting a thread.** The `start_thread` tool creates one and returns its
`discord:<thread_id>` scope (#593). Given `from_message_id` it hangs a thread
off that message; without one it starts a standalone thread on the channel, so
a DM — which has no threads — is refused, a forum is refused because it takes
posts rather than plain threads, and so is any other channel with no threads of
its own. Discord's 1–100 character name rule is enforced in the tool, and
an optional first message is posted into the thread rather than into the parent,
split like any reply and sent with `allowed_mentions` none. The new thread is
registered in the channel registry, because a thread is a channel in its own
right: a later send to that scope has to find it, and an inbound message would
have registered it had someone typed in the thread first. A permission refusal
is reported as the permission Discord wanted (Create Public Threads), not as an
SDK exception.

**Ordering.** discord.py runs every gateway event as its own task, so
handling a message inside `on_message` would let two quick messages
overtake each other. The client's `on_message` therefore only puts the
message on a queue (synchronously, which preserves arrival order), and a
single processor task consumes it. A failure handling one message is
logged and does not stop the processor.

**Identity.** A channel's scope is a numeric Discord id, never a name:
`discord:<channel_id>` for a server channel, and `discord:<thread_id>` for
a thread, which has its own id and therefore its own history, separate from
its parent channel. DMs are `discord:dm:<user_id>`, keyed on the user, not
the DM channel's id. Names (nicks, channel names, usernames) change, and
channel ids are the persistence key for `sessions.db`, the jsonl logs, and
memory, so keying on a name would fork a history on a rename. The sender
passed to `on_message` is the author's user id.

**`resolve_channel`.** For a `discord:<scope>` id the registry does not
hold (a reminder due after a restart), `DiscordPlugin.resolve_channel`
returns the registry's channel for it. It returns None for other
transports, for scopes that are not all ASCII digits (optionally prefixed
`dm:`), and before `on_start` has resolved the registry. Sending to such a
channel looks the destination up lazily (cache first, then a fetch), so it
works as long as the bot can still see that channel or user.

**Authorization.** `should_process_message` is a default-deny gate: the
sender's user id must appear in `allow:` (entries are compared as strings,
so YAML ints work). A refusal is logged at INFO naming the sender's id, and
nothing is sent back. The transport fires `on_message` for every message
that passes inbound filtering regardless: the gate is the one place that
vetoes and the one place that logs the attempt. The transport checks its
allowlist before it resolves the channel, so a refused sender never creates
a registry entry (#274, as #103 for Signal): their message is gated on a
transient `Channel("discord", <scope>)` that is never registered, or on the
live entry if one exists (a guild channel an allowed user talks in, or a
sender since dropped from `allow:`). Note what that means in a
server: anyone who can see the channel can @-mention the bot, and gets
silence. The bot has no way to tell them it is configured to ignore them.

**Liveness.** Typing only; Discord has no bot read receipts. On the first
outstanding message for a channel the transport sends the typing indicator
immediately and starts a refresh loop that re-sends every 8s (Discord's
indicator lasts about 10s) while the channel's outstanding count stays
positive. The transport mints the message's trace id, takes the
slot for an allowlisted sender, and passes the id to `on_message`; a
`finally` around that call gives the slot back unless `on_message_admitted`
fired for the id. Admission fires before `on_message` returns, so a gate
veto, an agent that is not ready yet, or a raise or cancellation all land
there. A final, sent (`send_message`) or withheld (`on_agent_response`
for an unsent final, #417), gives back every outstanding slot (one final
answers a coalesced burst, #398): the count goes to zero and the refresh
loop ends when it next sees no slot owed. Discord has no stop frame, so
nothing else is sent. Only an
allowlisted sender ever took a slot, so a refused stranger @-mentioning the
bot gives nothing back and cannot stop typing owed to an allowed user in
the same channel; no `on_message_rejected` implementation is needed.
A refresh loop still asleep from an earlier turn is reused rather than
doubled, so a channel has at most one, and `on_stop` cancels it. Typing is
only ever sent for an allowlisted sender, so an unauthorized user gets no
sign the bot is live.
A failed typing call is logged and never blocks the reply.

**Outbound.** `send_message` and `send_progress` both split through
`split_discord_message` at `message_chunk_size` and send every non-blank
chunk; `send_progress` skips the outstanding-count bookkeeping, as on
Signal. Two properties apply to every send:

- *Mentions are disabled.* Each `send` passes
  `AllowedMentions.none()` explicitly, whatever the client's default is.
  The reply is built from model output, which can contain text from any
  untrusted source, and a literal `@everyone`, role mention, or user
  mention in it must render as text and ping no one.
- *Splitting is fence-aware.* `split_discord_message` builds on the shared
  `split_message` for oversized lines, but keeps ``` code fences intact:
  a fence left open at a chunk boundary is closed at the end of that chunk
  and reopened, with the same language tag, at the start of the next, so
  every chunk renders its code as code. Chunks are at most
  `message_chunk_size` UTF-8 bytes (the limit is in bytes, though
  Discord's 2000 is in characters, so multi-byte text splits a little
  early), and the chunks' bodies concatenate to the original text, so
  nothing is truncated. A chunk that is only whitespace is not sent,
  because Discord rejects empty messages.

Known limits of the splitter, none of which loses text:

- Only ``` fences are tracked. `~~~` fences and runs of four or more
  backticks are not recognized as fences, and a longer run is treated as
  ``` plus extra backticks.
- A ``` run that happens to be cut inside a single space-free word of
  about `message_chunk_size` bytes inverts the tracked fence state for the rest of the
  message. Chunks stay balanced, but later code may render as prose and
  vice versa.
- A language tag longer than 16 bytes, or one containing a backtick, is
  not carried across a boundary; the reopened fence has no tag.
- A cut can fall inside a ZWJ emoji sequence or between a base character
  and its combining marks, because splitting is by codepoint.

Outbound is text only. `send_thinking` and `send_tool_status` are not
overridden (the `TransportPlugin` no-ops apply), so reasoning and tool
traffic never reach Discord.

There is no local Discord server, so there is no harness end-to-end test
for this transport; behavior is pinned by `tests/test_discord_plugin.py`
against fake clients. The first real connection is the real test.

Replies to agent-started turns on a Discord channel (a reminder, say) are
delivered through the normal send path: `_deliver` resolves the destination
from the channel id, so no inbound message is needed. The outstanding count
is guarded at zero in `DiscordPlugin._release_all`, as on Signal,
with the same early typing release (#101): any final, sent or withheld,
zeroes the count.
Sending to a channel other than the turn's own is a separate capability,
not provided by the transport: PRD §3.5.9 (operator allowlist that starts
empty, plus a budget); see the scheduler's "Not in phase one: a
`send_to_channel` tool" note and `docs/threat-model.md`.

Out of scope: voice, slash commands, embeds, and reactions. Attachments are in
scope in both directions — see **Attachments** above for the inbound rules and
`send_attachment` for the outbound one.

### Zulip (`channels/zulip.py`)

`ZulipPlugin` — Zulip transport over Zulip's REST event-queue API. It logs in as a bot user and talks in 1:1 DMs, group DMs and channels. There is no SDK and no extra: five endpoints over aiohttp (`GET /users/me`, `POST /register`, `GET /events`, `POST /messages`, `POST /typing`). Creating the bot and finding ids are in [zulip-ops.md](zulip-ops.md); the keys are in [configuration.md](configuration.md#zulip--zulip-transport). It lives in the kernel next to the other transports, by the owner's decision on #640, not as a package.

The requirements live in [`corvidae/channels/zulip-requirements.md`](../corvidae/channels/zulip-requirements.md), not in the PRD: Zulip is exempt from PRD updates by the owner's decision on #640.

With no `zulip:` block the plugin loads, resolves its registry dependency and does nothing else: no session, no tasks, nothing logged above DEBUG. A malformed block raises in `on_init` naming the field (`zulip.<key>`) and what is accepted. All validation runs before anything is stored, so a failed init leaves the plugin inert and not half-configured.

**Connection.** `_connection_loop` registers a queue and long-polls it. `/register` is sent with `event_types=["message"]` (edits and deletes are other event types and are never delivered, which is how the append-only log is kept), `apply_markdown=false` (raw Markdown for the model), `client_capabilities={"empty_topic_name": true}` and `fetch_event_types=["realm"]`. The last one matters: `max_message_length`, `max_topic_length` and the empty-topic label are returned only if `realm` is in `fetch_event_types`, which otherwise defaults to `event_types`. The response's `zulip_feature_level` must be at least 248 (Zulip 9.0, the first release with `channel` as a message and typing type), or the loop ends with one ERROR naming the site, the level and the remedy. Every call sorts its outcome into one of five signals: `_Fatal` (401 or a deactivated user or organisation: one ERROR, loop ends, not retried), `_QueueLost` (`BAD_EVENT_QUEUE_ID`: WARNING, re-register at once), `_RateLimited` (429: wait the server's `retry-after`, capped at 60 s), `_Transient` (network error, timeout, 5xx, a reply that is not Zulip's JSON: retry the same queue on the shared ladder, 10 s doubling to 300 s, reset on success) and `_Rejected` (a 4xx the server meant). Events are processed in id order and acknowledged by the next poll's `last_event_id`, after handling. A malformed message event is skipped with a WARNING naming the event and message ids, and does not stop the loop.

**A lost queue is not replayed (#640 D4).** The server drops an idle queue, and a restart starts a new one. Messages sent in the gap are not fetched; the WARNING says so, naming the site and the bot. Catching up (persist the last admitted id per channel, `GET /messages` from it on re-register, deduplicate by id) is a follow-up.

**Identity.** A channel's scope is built from ids only (PRD principle 10): `zulip:<stream_id>`; `zulip:dm:<user_id>` for a 1:1 DM, keyed on the other person; `zulip:dm:<id>,<id>` for a group DM, ids ascending, the bot's own left out. A Zulip *topic* has no stable id (the API pages for register, events, messages, topics and update-message expose none) and users can rename or move topics, so a topic-keyed scope would orphan its history on a rename. It is therefore a label and reply routing, never a key: `channel.name` is `#general > plans` (the empty topic shows as the realm's `realm_empty_topic_display_name`), and the topic of the newest *allowed* message is kept in `_topics[channel.id]`, in memory only. This is the same rule Discord applies to threads, which have their own immutable id; Zulip topics don't. The sender passed to `on_message` is the sender's user id, with the display name in `channel.sender_names`.

**The empty topic.** From Zulip 10 a channel message may have `""` as its topic. Without the `empty_topic_name` capability the server rewrites it to its display label in events; with it, the transport receives `""`, replies with the same `""`, and shows the label only to the model. The reply therefore lands where the question was asked on every server version.

**`resolve_channel`.** For a `zulip:<scope>` id the registry does not hold, the plugin returns the registry's channel. It returns None for other transports, for scopes that are not a stream id or `dm:` and ascending user ids, and before `on_start`. A reply needs no connection to be addressed: the destination is built from the scope.

**Authorization.** `should_process_message` is default-deny: the sender's user id must be in `allow`. A refusal is one INFO line naming the id, and nothing is sent. The allowlist is checked before the channel is resolved, so a refused sender never creates a registry entry (a transient `Channel("zulip", <scope>)` is gated instead), and only an allowed sender's message sets the channel's label, the sender-name cache or the reply topic: a stranger's topic must not steer where the agent answers.

**Addressing.** A message in a channel or group DM is admitted only when `flags` contains `mentioned` (Zulip sets it for a direct mention and for a mention through a user group the bot belongs to; the group's token is not the bot's own, so it stays in the text); `stream_wildcard_mentioned` and `topic_wildcard_mentioned` do not count, as `@everyone` does not for Discord. A 1:1 DM needs none. `channel.addressed` is set before every `on_message`, `None` for a 1:1 DM. The bot's own mention token (`@**Name**`, `@**Name|id**`, silent `@_**Name**`, or any name with the bot's id) is stripped from the text; other people's mentions stay. The bot's own messages come back through its queue and are dropped by sender id, which is the only thing between the agent and answering itself if its id is ever put on the allowlist with `require_mention: false`.

**Liveness.** Typing only; Zulip bots have no read receipts. The first outstanding message on a channel sends `POST /typing` `op=start` (for a channel, `type=channel` with `stream_id` and `topic`; for a DM, `type=direct` with the user ids) and starts a refresh loop re-sending every 8 s while the count stays positive. Unlike Discord, Zulip has a stop frame, so a final (`send_message`), a withheld final (`on_agent_response` with `final_was_unsent`) or a message the gate never admitted sends `op=stop` once the last slot is released. On the withheld-final path (`on_agent_response`) the ownership check, `final_was_unsent` and the release happen before the first `await`, as for Discord and Signal; a sent final releases after it has delivered, so a message that arrives mid-delivery has its slot released with the rest. A failed typing call is a WARNING, never a lost reply. Only an allowed sender ever takes a slot.

**Outbound.** `_deliver` defuses mentions, splits with `split_message` at the smaller of `message_chunk_size` and the server's `max_message_length` (bytes, conservatively, since the server counts code points), skips whitespace-only chunks, and posts each in order through a bounded queue (50 chunks): a transient failure leaves it queued with a WARNING and the connection loop flushes it after the next successful poll (PRD §3.2 item 8); past the bound the oldest is dropped and the count logged; a refusal drops the rest of that reply and logs the channel, the topic and the server's own explanation. A 429 on a send waits `retry-after` up to twice. Two properties apply to every send:

- *Mentions are defused.* Zulip has no allowed-mentions switch, so `@` followed by `*` (`@**all**`, `@**Name**`, `@*group*`) gets a zero-width space between them before splitting, and the chunks concatenate to the defused text. Silent mentions (`@_**Name**`) notify no one and pass through.
- *Replies go to the topic they answer.* The topic of the newest allowed message, else `default_topic`; a DM goes to its user ids.

`send_thinking` and `send_tool_status` are not overridden, so reasoning and tool traffic never reach Zulip.

**Testing.** There is no local Zulip server to run in `harness/`, so this transport has no end-to-end harness test (PRD §3.2 item 9). `tests/test_zulip_plugin.py` runs the real client against an aiohttp server on loopback that implements the five endpoints, with scripted failures, so request encoding, Basic auth, 429 and `BAD_EVENT_QUEUE_ID` are exercised for real. What it cannot show is a real server's behaviour. Everything taken from Zulip's API documentation and not observed is listed in [`zulip-requirements.md`](../corvidae/channels/zulip-requirements.md#taken-from-the-documentation-not-observed).

**Known limits.** No attachments (an upload is the link Zulip put in the text), no edits or deletes, one organisation per block, no catch-up after a lost queue, and a plain split that can cut a Markdown code fence across two messages (`shortcut:` upgrade to a fence-aware split, as Discord has, if seen).

### Web (`channels/web.py`)

`WebPlugin` serves a browser chat UI over `aiohttp.web`: a chat page, one
WebSocket per tab, JSON frames both ways, plus read-only panels that other
plugins contribute. Configuration and remote
access are in [configuration.md](configuration.md#web); what an attacker
can do with the listener is in [threat-model.md](threat-model.md).

With no `web:` block the plugin is inert: no listener, nothing logged above
DEBUG. A malformed block raises in `on_init`, naming the field. A bind
failure raises out of `on_start`.

**Identity.** One channel per named conversation: `web:<slug>`, where the
slug matches `[a-z0-9][a-z0-9_-]{0,63}`. Every message on a `web:*`
channel originates here, with sender `web`. Frames carry the bare slug
(`main`); internal state is keyed by channel id (`web:main`).

**Auth.** The token comes from `web.token`, then `CORVIDAE_WEB_TOKEN`, then
`secrets.token_urlsafe(32)`, which is logged once at WARNING as a login URL
(pre-formatted into the message, so it's never in `args` or an `extra`
field). Opening `/?token=…` returns a 200 page that refreshes to `/` and
sets `corvidae_web` (`HttpOnly; SameSite=Strict; Path=/; Max-Age` 400
days, plus `Secure` when the request's `Host` belongs to an `https://`
allowed origin). A 200 rather than a redirect, because a redirect after a
cross-site navigation wouldn't carry the Strict cookie. The cookie value is
`HMAC-SHA256(token, "corvidae-web-session")`, so it survives restarts with
the same token and dies when the token changes. Every route needs it.
Comparisons are constant-time and treat non-ASCII input as unequal.

Middlewares, outermost first:
1. `error_guard`: an unexpected exception becomes a 500 logged without the
   URL (it may carry `?token=`);
2. a `Host` check against values derived from the allowed origins (a
   DNS-rebinding guard);
3. auth, as above;
4. an `Origin` check on any request whose `Upgrade` header is `websocket`
   (compared case-insensitively, as aiohttp accepts it): the `Origin` must
   be in the allowed set, else 403. It covers the chat socket and any
   WebSocket a panel handler opens. It runs after auth, so a request
   without the cookie gets 401 first.

Security headers go on every response through an `on_response_prepare`
signal handler, which runs just before a response's headers are sent:
normal, streamed (a handler that prepares its own `StreamResponse`),
`HTTPException`, 404 and the guard's 500 alike. Because it runs as the
headers go out, a security header a handler sets itself is overwritten. The
headers: a strict CSP (`default-src 'self'`, scripts and styles from
`'self'` only, `img-src 'self' data:`, `connect-src 'self'` plus an
explicit `ws://`/`wss://` source per allowed origin, IPv6 literals excepted
since CSP can't express them), `nosniff`, `no-referrer`, `no-store`.

Allowed origins are the bind origin (for a loopback bind, all of
`127.0.0.1`, `localhost` and `[::1]` on the bound port) plus
`web.allowed_origins`, all canonicalised the same way the incoming headers
are. The `AppRunner` runs with `access_log=None`: aiohttp's access log
would write `?token=` to disk.

**Protocol.** Client → server: `list`, `subscribe {conversation}`, `send
{conversation, text, seq}`. Server → client: `conversations`, `history
{conversation, items, working}`, `user {text, id}`, `message`, `thinking`,
`progress`, `tool_status {tool, status, summary}`, `working {value}`,
`rejected {text, id}`, `divider` (replay only), `sent {seq}`, and `error
{request, seq, code, message}` with code `starting` (the agent hasn't
started; the client retries) or `bad_request`. Each socket handles its
frames one at a time, in order. Malformed input (invalid JSON, deep
nesting, a non-object, a bad slug, non-string text, lone surrogates, an
unknown type) gets `bad_request`, and the socket stays open; an exception
out of `on_message` (a third-party gate raising) is logged with its
traceback and answered the same way. Text that is empty after `str.strip()`
is acknowledged and otherwise ignored.

Every `send` is answered once, on its own socket only: `sent` when the
handler finishes, or `error` carrying the same `seq` when it fails, which
includes an exception out of `on_message`. `seq` belongs to the tab: the
page counts its sends from 1, and the server echoes the value without
interpreting it, as null where the frame had none. `sent` carries no
`conversation`, so the page's current-conversation filter can't drop it after
a switch. A handler cancelled at shutdown answers nothing, and its socket
closes.

**Unsent messages.** A user message is shown from the moment it's sent,
but its row reaches `message_log` only at dequeue. Between the two it's an
`_Unsent` entry with a server-minted display id. The entry also carries
the message's trace id, minted by the send handler when it creates
the entry and passed to `on_message`, plus an `admitted` flag. The admitted
and rejected hooks find their entry by that id, with no ambient state:
admission sets `admitted`, rejection removes the entry (and the `rejected`
frame carries its display id, so the right item is marked even when two
tabs send the same text), and `on_message_persisted` removes the entry.
A hook carrying an id no entry holds changes nothing (a rejection reports a
null id). Admission and rejection fire before `on_message` returns, so the
send handler's `finally` removes any entry still un-admitted: dropped,
raised or cancelled. Removal is idempotent.

**Working indicator.** On at admission. Off when the idle tracker reports
the channel settled (a `watch_channels(0, ...)` watch, filtered to web's
channels), unless a reply is still owed: an admitted-but-unpersisted entry
or pending tool calls; the next settle re-checks. Web never reads the
Agent's queues. A zero-period watch fires at settle, on the next loop step.
Without an idle tracker (hand-built plugin managers) web logs a warning and
clears on the final `send_message` instead.

**Replay.** `subscribe` sets the subscription, starts buffering live
frames and snapshots the unsent entries, all before its first `await`;
then it reads the newest 200 `message_log` rows (`ORDER BY timestamp
DESC, id DESC`, reversed) and queues `history` (rows mapped to frames,
then one `user` frame per snapshot entry), then the buffered frames. A
summary row becomes a `divider` at its timestamp; `context` rows,
`unsent` rows and `system` messages are skipped, and a not-sent marker for
progress or thinking drops that frame of the assistant row it follows
([The output gate](#the-output-gate)); assistant rows give `thinking` (from
`reasoning_content`), then `progress` or `message` (text through
`strip_thinking`), then a `dispatched` chip per tool call; tool rows give a
`completed` chip, named via the call id, with task.py's `[Task <id>] `
prefix removed. History favors duplicates over losses: a reply persisted
just before the read and broadcast while buffering can show twice until
the next reload. The 200-row window counts rows that render as nothing
(`context`), so visible history can be shorter. Without the persistence
plugin, history is empty (WARNING once, then DEBUG).

Replay and live differ in these ways:
- the LLM error text is sent live but never persisted, so it doesn't replay;
- a vetoed message has no row, so on reload it's gone entirely;
- an assistant message whose tool calls `max_turns` suppressed replays as
  progress plus `dispatched` chips for calls that never ran;
- with the `thinking` plugin disabled, live `message` text keeps `<think>`
  blocks and replay strips them;
- a `completed` chip is sent when the tool finishes, but its row is written
  when the result is dequeued, so a reload in between shows the call
  without its completion.

The conversation list merges `message_log`'s `web:*` channels with the
registry's live ones (the later `last_active` wins), drops names that fail
the slug rule, and sorts newest first.

**Panels.** `WebPlugin` owns `on_web_console_init(panels: list[Panel])`
(spec class `WebConsoleSpec`). Each implementer appends one frozen `Panel`:
`plugin` (the registering plugin), `title`, `static_dir` (holding
`panel.js` and `panel.css`) and `routes`, a sequence of aiohttp
`RouteDef`s. The hook runs at the end of `on_init`, only when the `web`
block is present, and the routes are validated right after it returns:
each must be a `RouteDef` with method `GET`, else `PluginValidationError`
naming the plugin. `ValueError` would be reported as a config error, and
the plugin, not agent.yaml, is at fault. Running in `on_init` means the
restart pre-flight, which skips `on_start`, refuses a bad panel. The panel
is named by its plugin's registered name, which is unique. The stored
list is `(name, panel)` pairs in registration order.

`_build_app` mounts each panel on the console's own app, after the
console's routes, so every middleware and the header signal apply with no
per-panel code:
- `GET /panel/<name>/`: the shell page;
- `GET /panel/<name>/static/<file>`: `add_static` on `static_dir`, with
  the same options as `/static` (no symlinks, no index);
- `GET /panel/<name><path>` per route, through `add_get` (which also
  answers HEAD) with the route's `name` kwarg dropped, since route names
  are app-wide and two panels could collide.

A plugin registers one panel; a second panel from the same plugin is not
supported. The app is frozen once running, so a handler can't add routes,
middlewares or signals through `request.app`.

`on_start` renders the navigation once: `<nav class="panels">` with one
`<a href="/panel/<name>/">` per panel, `name` and `title` HTML-escaped, or
the empty string with no panels. It replaces the `<!--panels-->` marker in
`index.html` (at the bottom of the sidebar, so on a phone it collapses
with the conversation list) and in `panel.html`, and both pages are served
from those strings. With no panels the chat page is unchanged but for the
marker. `panel.html` is one static shell shared by every panel: the
console's `/static/app.css`, then relative `static/panel.css` and
`static/panel.js`, a header with a link back to `/` and the navigation,
and an empty `<main id="panel">` that `panel.js` fills. It has no inline
script or style, so the CSP needs nothing extra. Because the asset URLs
are relative, the page URL needs its trailing slash; `/panel/<name>` is a
404.

With no `web` block the hook never fires. With `web` blocked or not
installed the spec is never installed, so a panel plugin's impl is an
orphan, logged at DEBUG when the plugin lists `web` in
`optional_depends_on`.

**Fan-out and backpressure.** Every server frame goes through the client's
queue (1024) and a writer task. Enqueueing never awaits. A client whose
queue fills, or whose subscribe buffer passes 512 frames, is removed from
fan-out and closed with 1008 in a tracked task.

**Connection ownership and shutdown.** Each connection is owned by its
handler task. `_handlers` holds every live handler, and only the handler
removes itself; `_clients` is just the fan-out set, which the drop path
and token rotation shrink. The handler's `finally` releases the
connection: out of fan-out, writer cancelled, and the transport aborted if
it still holds unsent bytes. `on_stop`:
1. closes every fan-out socket (each close bounded at 1 s, then
   `abort()`) and awaits pending drop-path closes;
2. cancels every handler still in `_handlers` and waits up to 0.5 s. This
   matters: once a connection is lost, aiohttp drops its own reference to
   the handler task, so `runner.cleanup()` can't cancel a handler stuck in
   a gate. The cancellation unwinds through `on_message`, and the send
   handler's `finally` removes its entry;
3. runs `runner.cleanup()` (`shutdown_timeout=0.5`).

The worst case is about 2.5 s, inside Runtime's default 3 s budget
(`daemon.shutdown_timeout`).
New WebSocket handshakes during shutdown get 503. Aborting rather than
closing matters with a peer that stopped reading (a phone asleep):
`transport.close()` waits to flush a full buffer forever, and aiohttp's
own close paths (a timed-out close, a write error, a heartbeat timeout)
only call `close()`.

**Config reload.** A token change swaps the session value and closes
every socket with 4001 ("token changed"); the page then shows "session
expired" and stops reconnecting. A `web.token` removed from config falls
back to `CORVIDAE_WEB_TOKEN` (as the daemon started with it), else a fresh
generated one (logged); a generated token is kept while none is provided.
Only `web.token` rotates on reload: the daemon's environment can't change
under it, so rotating `CORVIDAE_WEB_TOKEN` takes a restart. `host`, `port` and `allowed_origins` changes log "restart to
apply". A removed block logs the same and changes nothing. A malformed
block raises (the watcher logs it) and changes nothing.

**Frontend.** `web_static/` ships as package data: `index.html`,
`panel.html`, `app.css`, `app.js` (plain JS, no build) and vendored marked 15.0.12 and
DOMPurify 3.2.6 (`vendor/README.md` records versions, hashes and
licences). Model text reaches the DOM through one sink:
`DOMPurify.sanitize(marked.parse(text))` with `style`, `class`, `id` and
`data-*` attributes and `style` tags forbidden (so a reply can't borrow the
app's classes or ids); links get `target="_blank" rel="noopener noreferrer"`.
Everything else uses `textContent`. Thinking and dividers render in closed
`<details>`. The page reconnects with backoff (1 s → 30 s) and, after
three failed connects, checks `/` for a 401 to detect an expired session.
`tests/test_web_render_smoke.py` checks the sanitising and the
unconfirmed-send restore in headless Chrome; it skips without Playwright,
so the `web-smoke` CI job installs a pinned Playwright and Chromium and
fails if the test is skipped. Run it locally with
`uv run --with playwright pytest tests/test_web_render_smoke.py`.

The page keeps a map of sends the server hasn't answered, keyed by `seq`,
and clears the composer on send. `sent` removes the entry. An `error` for a
pending send, or any socket close, moves the affected sends to an
"unconfirmed" banner, each labelled with its conversation and shown with
`textContent` only. The banner has a Dismiss control, the composer draft is
never touched, and nothing is resent automatically.

Known limitations:
- a turn that ends in an exception without a `send_message` leaves the
  indicator on until the channel's next `send_message`;
- an exception in the agent after dequeue but before step 4c
  (`on_message_persisted`) leaves that message's entry until restart: the
  indicator stays on and the message keeps appearing at the end of
  history. That path is logged at ERROR, and such a channel gets no
  replies at all;
- a gate that's slow to decide holds up the socket's later frames, since
  frames are handled in order;
- messages admitted or queued at shutdown are lost after being
  acknowledged, as with every transport, and in the pane they simply
  disappear on reload; one still in a gate when its socket closes gets no
  answer, so the page restores it;
- a dead socket is only noticed when the browser closes it, and the page
  restores unanswered sends then, not at send time;
- a send whose socket closes after the agent took it but before `sent`
  arrived is restored anyway, and may be sent twice.

Out of scope for v1: paging back past 200 rows, token streaming, and views
of other transports' channels.

Behavior is pinned by `tests/test_web_plugin.py`, which drives a real
listener with aiohttp's client and raw sockets.

### Context panel (`context-panel` package)

`ContextPanelPlugin` registers one panel when `context_panel` is in the
config (a bare `context_panel:` counts) and otherwise nothing, deciding in
`on_init`. It depends on `registry` and `persistence`. It builds each
response from `channel.conversation` and the registry without an `await`,
so every response is one consistent view; only then does it await the
timestamp lookup below, which cannot change the response because a row's
timestamp never changes. Three GET routes; `/window` and
`/poll` are keyed by `?channel=`, and a channel that is unregistered or has
no loaded window answers `{"loaded": false}`:

- `/channels`: loaded channels with message count, token estimate and budget,
  and `name` (the transport's readable name, `Channel.name`) when one is set.
- `/window`: system prompt, tool schemas and the newest 200 rows (all with
  `all=1`), each row with its `_message_type` tag and its visible fields
  (`_`-prefixed keys dropped), plus the cursor. A row whose `_db_id` has a
  `message_log` row on that channel also carries `ts`, that row's epoch
  timestamp; one query per response, and `ts` is absent otherwise. `/poll`
  rows carry `ts` the same way, but a row is sent once, so one persisted
  after it was sent gets no `ts` until the window is reopened.
- `/poll?count=&digest=`: the rows after the client's cursor, the new
  cursor and the live events.

**Cursor.** `{count, digest}`: the row count and a sha256 over the head
(system prompt and tool schemas) followed by the first `count` rows. A poll
recomputes the digest over the same prefix of the current window. A match
means the client's rows are still the window's prefix, so the reply is the
rows after `count`. A mismatch, or a `count` past the window, replies
`resync: true` and the client reopens the window. Content is compared, not
identity, because trims and reasoning stripping fire no hook:

| Window change | Poll answer |
|---|---|
| rows appended | the new rows |
| head trim, compaction (`replace_with_summary`), newest-row truncation | resync |
| interior row rewritten (reasoning stripped under `drop` or `exchange`) | resync |
| system prompt or tool view changed (late prompt refresh) | resync |

**Live events.** `send` runs for every channel; while the panel is serving
it buffers thinking, progress and tool-status events per channel, each with
a global sequence number. The buffer is stamped with the window's mark, the
row count and a digest of row 0, when its first event arrives. `/poll`
returns the buffer only while the window's mark equals the stamp, so a
landed row or a changed first row hides it; the channel's next event
replaces it with a new buffer. The final reply is not buffered: it becomes
a row. A tool that finishes between two polls hides its events before the
client sees them.

The client polls one second after each poll completes, and polls again
immediately after a window opens, so two polls can be in flight. A response
is dropped if the selection changed, the window was reopened, or the page's
cursor is no longer the one the request carried.

### Webhook ingress (`channels/webhook.py`)

`WebhookPlugin` is a `CorvidaePlugin`, not a transport: it owns no channels and
never writes to a conversation. It accepts `POST /hook/<endpoint>` over
`aiohttp.web` and routes each accepted JSON object to the plugins subscribed to
that endpoint. Configuration is in [configuration.md](configuration.md#webhook--webhook-ingress).

**Subscribing.** The plugin owns `on_webhook_subscribe(subscriptions)`. At
`on_start` it calls the hook with an empty `dict[str, list[handler]]`; each
implementer appends async handlers under endpoint names. Each (endpoint,
handler) pair becomes a subscriber with its own bounded queue (100) and worker
task, so a slow or raising handler affects only itself. A raising handler is
logged at `ERROR` with its traceback, naming endpoint and handler.

**Request path.** Unknown or subscriber-less endpoint: 404 (before auth, so a
typo is visible). Bearer is checked before the body is read; `hmac-sha256` is
checked after, over the raw bytes. aiohttp's `client_max_size` rejects an
oversized stream with 413. The body must parse as a JSON object (400). Then the
delivery id (`X-GitHub-Delivery`, else `Idempotency-Key`) is looked up in the
endpoint's 32-entry window (200 duplicate) and recorded; the delivery is queued
for every subscriber, dropping a full queue's oldest item with a `WARNING`, and
the reply is 202. The delivery is a frozen `WebhookDelivery(endpoint, payload,
delivery_id)` whose `origin` is `webhook:<endpoint>`, minted by the ingress.
Only the delivery id, taken from `X-GitHub-Delivery` or `Idempotency-Key`, reaches a subscriber from the headers.

**Failure handling.** A bind failure is not a fatal startup path (PRD
3.1.17): it logs `ERROR` naming `host:port` and the plugin stays inert. Config
errors raise `ValueError` from `on_init`. `on_stop` closes the listener first,
then cancels the workers, discarding queued deliveries. At start a `WARNING`
lists configured endpoints with no subscriber and subscribed endpoints with no
config.

### Webhook notifications (`webhook-notify` package)

`WebhookNotifyPlugin` subscribes one handler to each endpoint in
`webhook_notify.endpoints`. For a delivery it serialises the payload as compact
JSON (`ensure_ascii=False`), resolves each target channel at delivery with
`resolve_persisted_channel` (a `ChannelNotReady` target is skipped with a
`WARNING`) and makes one `spill.notify` call with the resolved channels and
`limit=max_payload_chars`. The service cuts a longer payload, writes the full
text to a spill file where a channel can read it, and notifies each channel
([Notification spill](#notification-spill-spillpy)); the plugin writes no cut
marker. The notification carries `source` and `annotations["origin"]` set to
`webhook:<endpoint>` (minted by the ingress); `origin` is declared
model-visible, so the kernel renders it in the header outside the payload
fence, and a cut payload gets its cut line there too. The plugin depends on
`"spill"`. Configuration is in
[configuration.md](configuration.md#webhook_notify--webhook-deliveries-as-notifications).

### ACP (`channels/acp.py`)

`AcpPlugin` + `corvidae acp` — Agent Client Protocol **v1** over stdio for
editors (bb `customAcpAgents`, Zed, …). Requires the optional
`agent-client-protocol` extra (`--extra acp` on the `uv sync` line).

**Process model.** ACP is a dedicated subprocess (`corvidae acp`), not
attached to long-lived `corvidae serve`. The command sets
`config["_acp_mode"]`; without that flag the plugin is inert (no stdin/
stdout ownership under `serve` / `cli`). Stdout is ACP JSON-RPC only;
logs go to stderr or a log file.

**Channels.** Sessions are `acp:<sessionId>`. `session/new` stores the
client `cwd` on `channel.runtime_overrides["cwd"]` (transport metadata,
not LLM config). Tool calls on ACP channels go through a swappable
`ToolBackend` (`LocalToolBackend` by default; see
`corvidae/tools/backends.py`). The call bound sits above the backend: when it
expires dispatch gives up and cancels `backend.run`, but a client-side backend
may keep running the tool on the client.

**Session durability (Phase 2).** `initialize` advertises
`loadSession: true` plus `sessionCapabilities` for `list`, `resume`, and
`close` (`delete` is not advertised). `session/load` and `session/resume`
reattach the **same** `acp:<sessionId>` from `sessions.db` (no forked
history). `session/load` replays user/assistant text (and compaction
summaries) as `session/update` chunks before responding, skipping `UNSENT`
rows and assistant text whose progress send a gate vetoed;
`session/resume` does not. `session/close` cancels in-flight work and
drops the in-memory channel while leaving `message_log` intact.
`session/list` reads an `acp_session_index` sidecar table in the same DB
(cwd + `updated_at`).

**Hooks.** `AcpPlugin` is a `TransportPlugin` with `transport = "acp"`,
so outbound events reach it only for `acp:` channels. Tool status is
keyed on `(channel.id, tool_name)`; the event's `tool_call_id` is not used
yet. `session/prompt` mints the message's
trace id and passes it to `on_message`, and completes with `end_turn`
after tool drain (from `send_message`, or from `on_agent_response` when
the final was unsent), including when the final `session/update` cannot be
delivered (logged at WARNING; #99); `session/cancel` yields `cancelled`.
Completion is keyed on that trace (#419): the prompt's trace is stored
beside its future, and a final carrying a different one leaves the future
pending at DEBUG. That covers a cancelled turn whose late final lands after
the client has sent a new prompt, and a notification-originated turn (a
reminder firing on an `acp:` channel), neither of which answers the
client's prompt. A final carrying no trace id still completes, because a
prompt that never resolves hangs the client's `session/prompt`.
A second `session/prompt` on a session with one in flight is rejected
with `invalidRequest` and the first is left untouched (#93). The stdio loop
starts reading only once `Agent.started` is true: Runtime starts Agent
after the `on_start` broadcast, and a prompt read before then would be
dropped with its response never sent.

Design of record: `plans/acp-transport-design.md`. Gate B harness:
`tests/test_acp_conformance.py`. Subprocess stdio smoke (mocked LLM, no
live API): `tests/test_acp_stdio_smoke.py`.

## Configuration

```yaml
daemon:
  session_db: sessions.db

llm:
  main:
    base_url: "http://host:8080"
    model: "model-name"
    api_key: "optional"
    extra_body: {}              # optional, passed through to LLM
    max_retries: 3              # retry attempts on transient errors (default 3)
    retry_base_delay: 2.0       # base exponential backoff delay in seconds
    retry_max_delay: 60.0       # maximum retry delay cap in seconds
    model_not_loaded_wait: 60.0 # seconds to wait out an unloaded router model
    timeout: null               # HTTP timeout per request; null = no timeout
  embedding:                    # optional embedding model for memory retrieval
    kind: embedding             # required; an omitted kind means chat
    base_url: "..."
    model: "..."
    dimensions: 768             # required
    # same keys as main
  # forks:                      # any other name is a profile too
  #   extends: main             # deep overlay; a child's null deletes an inherited key
  #   extra_body: {reasoning_budget_tokens: 2048}
  # defaults: {chat: main, embedding: embedding}   # optional; these are the fallbacks
  # stages: {compaction: forks}                    # optional: stage -> profile
  # All of llm: is read at startup; a reload that changes it only logs a warning.

agent:
  system_prompt: "..."          # string or list of file paths
  max_context_tokens: 24000
  reasoning_history: drop       # drop | exchange | keep
  max_turns: 10
  mid_conversation_system_role: system  # system | user — role for non-leading system messages
  immutable_settings:           # keys the agent must not change (system_prompt always added)
    - max_turns

tools:
  max_result_chars: 100000      # truncation limit for tool result strings (default 100_000)
  # call_timeout: 600           # seconds bound on every tool call; 0 or null disables
  # timeouts: {task_pipeline: 3600}  # per-tool override of call_timeout
  shell_timeout: 30             # shell tool timeout in seconds (default 30)
  web_fetch_timeout: 15         # web_fetch timeout in seconds (default 15)
  web_max_response_bytes: 50000 # web_fetch response truncation (default 50_000)
  max_file_read_bytes: 1048576  # unranged read_file size limit in bytes (default 1MB)
  web_search_max_results: 8     # web_search result count (default 8)

memory:
  idle_consolidate_after: 1800  # seconds a channel must be quiet before idle consolidation
  consolidate_after_tokens: 4000  # new window tokens that start a turn consolidation (forks only)
  half_life_days: 30            # retrieval recency decay (e-folding time, days)
  retrieval:
    k: 8                        # KNN candidate count
    band_on: relative           # relative | score | similarity — what the bands apply to (#409)
    recency_from: created       # created | last_used — where ranking age runs from
    query_context: previous_reply  # previous_reply | none — the agent's last reply joins the query
    candidates: hybrid          # hybrid | vector — fuse KNN with FTS5 by reciprocal rank
    bands:
      strong: 4.5               # shown bare at or above this value (z under relative; 0.75 under score/similarity)
      moderate: 3.0             # shown [marginal] at or above this value, else dropped (0.60 under score/similarity)

scheduler:
  timezone: UTC                 # IANA name; resolves offset-less times (default UTC)
  catch_up_window: 3600         # seconds a missed one-shot may be late and still fire
  max_reminders_per_channel: 50 # pending tool-created reminders per channel
  jobs:                         # operator-declared recurring jobs, synced by name
    - name: morning-checkin
      channel: irc:#lex
      every: 1d                 # required
      at: "08:00"               # optional anchor: HH:MM or ISO datetime
      prompt: "..."             # or prompt_file: a path relative to agent.yaml

heartbeat:
  after: 3600                   # seconds of system quiet before a beat; absent = asleep

funnel:
  default_budget: 512           # tokens per admission-funnel call
  budgets:
    memory: 512                 # optional per-source overrides
  # window_budgets:             # optional cap on one source's CONTEXT per window
  #   memory: 4096
  # ingest_ttl: 600             # optional seconds before a queued or pending ingest is dropped

channels:
  "irc:#channel":
    register_prompt:            # appended after agent.system_prompt
      - prompts/lex.md
    max_context_tokens: 8000
    max_turns: 5

irc:
  host: irc.lan
  port: 6667
  nick: agent
  allow: [alice]                # default-deny; nick, or services account where the network has them
  channels:
    - "#channel"

discord:                        # optional; omit to leave the transport inert
  token: "..."                  # or set DISCORD_TOKEN in the environment
  allow:                        # Discord user ids; default-deny
    - 123456789012345678
  message_chunk_size: 2000      # 100-2000

web:                            # optional; omit to leave the transport inert
  host: 127.0.0.1
  port: 8765
  token: null                   # or CORVIDAE_WEB_TOKEN; else generated and logged
  allowed_origins: null         # extra origins, e.g. https://box.tailnet.ts.net

logging:                        # optional, falls back to defaults
  level: INFO                   # DEBUG | INFO | WARNING | ERROR | CRITICAL
  file: corvidae.log            # omit for stderr; CLI mode defaults to corvidae.log
```

When `system_prompt` or `register_prompt` is a list, each path is read and concatenated.
Relative paths resolve against the directory containing `agent.yaml`.

## Plugin Registration Order

`Runtime.start()` (`runtime.py`) registers plugins in two stages, not a
fixed hand-written order:

1. Four kernel objects are registered by hand: `ChannelRegistry` as
   `"registry"`, the `Clock` as `"clock"`, the `IdleTracker` as `"idle"` and
   the `NotificationSpill` as `"spill"`
   (`self.pm.register(..., name=...)` in `Runtime.start()`).
2. Every other plugin loads from the `corvidae` setuptools entry-point
   group: `self.pm.load_setuptools_entrypoints("corvidae")`
   (`runtime.py`), reading `[project.entry-points.corvidae]` in
   `pyproject.toml`. Names listed in `plugins.disabled` are blocked
   via `pm.set_blocked(name)` before this call, so disabled plugins are
   never instantiated.

The kernel's entry points (`[project.entry-points.corvidae]` in the root
`pyproject.toml`): `persistence`, `jsonl_log`, `shell_tools`, `file_tools`,
`web_tools`, `cli`, `irc`,
`signal`, `discord`, `web`, `webhook`, `acp`, `task`, `subagent`, `mcp`, `llm`,
`compaction`, `thinking`, `runtime_settings`, `tools`, `metrics`,
`usage_log`, `metrics_jsonl`, `scheduler`, `agent`,
`config_watcher`. The cognition extra adds `memory`, `memory_tools`,
`funnel`, `outcome_log`, `appraisal`, `critique`, `goals` and `silence` from
`packages/corvidae-cognition/pyproject.toml`.

`Runtime.start()` loads them with pluggy's `load_setuptools_entrypoints`,
which does not catch errors: an entry point that fails to import stops
startup. That is why a transport with an optional SDK (ACP, Discord)
imports it lazily rather than at module load.

Entry-point loading does not guarantee a meaningful cross-plugin order;
plugins that need another plugin's state resolve it lazily via
`get_dependency`/`on_start` ordering (`trylast=True`), not registration
sequence.

Immediately after entry points load, `Runtime.start()` logs `"plugins
loaded"` with `loaded` (the sorted names `pm.list_name_plugin()` returns
with a plugin object; pluggy keeps each blocked name as `(name, None)`, and
those are left out) and `blocked` (the configured `plugins.disabled` list, verbatim
— not narrowed to names that matched a real plugin). A disable entry that
names no plugin (a typo, a retired name, a plugin whose distribution is
not installed) also gets one `WARNING` per distinct name, naming
`plugins.disabled`, the value, and the installed entry-point names
(`_warn_unknown_disabled`). A name counts as known when it is a
`corvidae` entry point of any installed distribution or was registered
directly before step 7b; that snapshot is taken before blocking, because
`set_blocked` unregisters a plugin already registered under the name. Only
entry-point names are offered as choices. It is not an error and does not
stop the daemon.

Next, `install_plugin_hookspecs(pm)` installs the hookspecs plugins
declare ([Plugin-declared hookspecs](#plugin-declared-hookspecs)), and
the hook arg-binding guard checks every impl against the full set of
specs. Then `validate_dependencies(pm)` runs
to verify every plugin's `depends_on` set names a registered plugin.
Startup aborts with `RuntimeError` if any dependency is missing.
`report_orphan_hookimpls(pm)` then logs any hookimpl left with no spec,
before the `on_init` broadcast.

## Plugin Dependencies

Plugins declare dependencies using a class-level `depends_on` attribute (a
set of plugin name strings). `validate_dependencies(pm)` in `hooks.py`
iterates all registered plugins and raises `RuntimeError` if any declared
dependency is not found in the PM. It runs in `Runtime.start()`
(`runtime.py`) after all plugins are registered, before `on_start`.

**`get_dependency(pm, name, expected_type)`** — typed lookup via
`pm.get_plugin(name)`. Raises `RuntimeError` if the plugin is not
registered; raises `TypeError` if the plugin is not an instance of
`expected_type`. Returns the plugin cast to `expected_type`. Defined in
`hooks.py`.

**`validate_dependencies(pm)`** — iterates every registered plugin, checks
for a `depends_on` attribute, and raises `RuntimeError` naming the missing
dependency and the plugin that declared it. Runs once at startup.

**`optional_depends_on`** — a class-level set of plugin names the plugin
uses when they are loaded. `validate_dependencies` ignores it, so a
missing name is fine and it takes no part in cycle detection. Its only
reader is `report_orphan_hookimpls`, which logs at DEBUG rather than
WARNING for an impl of a hook owned by an absent optional dependency.

### Current dependency graph

```
Agent               → "registry", "task", "llm", "tools"
NotificationSpill   → "clock", "tools"  (kernel object, registered as "spill")
ShellToolPlugin     → (none)
FileToolsPlugin     → "tools", "spill"
WebToolsPlugin      → (none)
WebhookNotifyPlugin → "registry", "spill"; soft-uses "webhook"
CLIPlugin           → "registry"    (ChannelRegistry)
IRCPlugin           → "registry"    (ChannelRegistry)
WebPlugin           → "registry"    (ChannelRegistry); looks up "agent" and "persistence" lazily
AcpPlugin           → "registry"    (ChannelRegistry)
SubagentPlugin      → "llm", "tools"
McpClientPlugin     → "clock"       (background reconnect loop)
CompactionPlugin    → "llm"         (LLMPlugin); soft-uses "agent" for forks
ThinkingPlugin      → "registry"    (ChannelRegistry)
MemoryPlugin        → "persistence", "llm", "clock", "idle"; soft-uses "agent", "registry" for forks  (cognition extra)
MemoryToolsPlugin   → "memory"      (MemoryPlugin)                                (cognition extra)
CritiquePlugin      → "task", "llm"; soft-uses "agent" for forks                  (cognition extra)
OutcomeLogPlugin    → "persistence"                                               (cognition extra)
AppraisalPlugin     → "persistence"; soft-uses "agent" for forks                  (cognition extra)
SilencePlugin       → none; soft-uses "outcome_log" to record declines          (cognition extra)
HeartbeatPlugin     → "idle", "registry"; needs "memory" only when awake          (cognition extra)
UsageLogPlugin      → "persistence"
SchedulerPlugin     → "registry", "persistence", "clock"
```

Plugins with `depends_on = set()` (declared but empty):
`LLMPlugin`, `ToolCollectionPlugin`, `PersistencePlugin`,
`McpClientPlugin`, `RuntimeSettingsPlugin`, `JsonlLogPlugin`,
`TaskPlugin`, `MetricsPlugin`, `MetricsJsonlPlugin`, `ConfigWatcherPlugin`,
`FunnelPlugin` (cognition extra).

"Soft-uses" means `fork_agent(pm, stage)` (`pm.get_plugin("agent")`,
duck-typed): with the agent plugin absent or forks off, the live-window stages skip
(compaction's view call needs the agent plugin too; without it the call
fails and #308 applies).

Transport plugins use `get_dependency(pm, "registry", ChannelRegistry)` in
`on_start` to retrieve the shared `ChannelRegistry`. `SubagentPlugin` calls
`get_dependency(pm, "tools", ToolCollectionPlugin)` and
`get_dependency(pm, "llm", LLMPlugin)` at tool-call time inside `_launch`.

## Directory Layout

```
corvidae/
├── hooks.py              # AgentSpec, hookimpl, create_plugin_manager
├── tool.py               # Tool, ToolRegistry, tool_to_schema, ToolContext
├── tool_collection.py    # ToolCollectionPlugin (collects tools at startup)
├── channel.py            # Channel, ChannelConfig, ChannelRegistry, resolve_system_prompt
├── queue.py              # SerialQueue (is_empty property, take_while for coalescing)
├── llm.py                # LLMClient
├── llm_plugin.py         # LLMPlugin (LLM client lifecycle)
├── turn.py               # run_agent_turn(), AgentTurnResult
├── context.py            # ContextWindow, MessageType, DEFAULT_CHARS_PER_TOKEN
├── jsonl_log.py          # JsonlLogPlugin (persist_message, on_compaction)
├── logging.py            # StructuredFormatter, configure_logging()
├── agent.py              # Agent (single-turn dispatch)
├── persistence.py        # PersistencePlugin (DB lifecycle, conversation init)
├── clock.py              # Clock, Timer (kernel clock: now, sleep, call_later, call_at, call_soon)
├── idle.py               # IdleTracker, IdleWatch (per-channel and system quiet)
├── spill.py              # NotificationSpill (cut notifications spilled to files), Cut, current_cut
├── thinking.py           # ThinkingPlugin, strip_thinking(), strip_reasoning_content()
├── compaction.py         # CompactionPlugin
├── fork.py               # fork prompt builder, cut snap, agent.forks parser (#51)
├── task.py               # Task, TaskQueue, TaskPlugin
├── mcp_client.py         # McpClientPlugin (MCP server bridge)
├── main.py               # daemon entry point
├── channels/
│   ├── cli.py            # CLIPlugin
│   ├── irc.py            # IRCPlugin
│   ├── signal.py         # SignalPlugin (signal-cli JSON-RPC transport)
│   ├── discord.py        # DiscordPlugin (discord.py gateway transport)
│   ├── web.py            # WebPlugin (browser chat UI over aiohttp)
│   ├── webhook.py        # WebhookPlugin (POST /hook/<endpoint> ingress)
│   ├── web_static/       # its page, script, styles, vendored marked + DOMPurify
│   ├── acp.py            # AcpPlugin + corvidae acp command
│   └── split.py          # split_message() — shared outbound chunking
├── commands/
│   └── redact.py         # corvidae redact CLI (operator-only)
└── tools/
    ├── __init__.py       # ShellToolPlugin, FileToolsPlugin, WebToolsPlugin
    ├── shell.py
    ├── files.py
    ├── web.py            # web_fetch, web_search
    ├── subagent.py       # SubagentPlugin, subagent tool, run_agent_loop()
    ├── settings.py       # RuntimeSettingsPlugin, set_settings tool
    └── task_pipeline.py  # TaskPipelinePlugin, task_pipeline tool

packages/corvidae-cognition/   # the cognition extra (docs/cognition.md)
├── pyproject.toml        # distribution corvidae-cognition; its entry points
└── corvidae_cognition/
    ├── memory.py         # MemoryPlugin (consolidation + retrieval + message_fts schema)
    ├── retention.py      # run_retention_job(), retention_score()
    ├── memory_tools.py   # MemoryToolsPlugin, search_memory, recall_raw
    ├── funnel.py         # FunnelPlugin (context-admission funnel)
    ├── appraisal.py      # AppraisalPlugin
    ├── critique.py       # CritiquePlugin
    ├── goals.py          # GoalsPlugin (goals table, add/update/retire_goal)
    ├── silence.py        # SilencePlugin (NO_REPLY veto, decline record, system-prompt instruction)
    └── outcome_log.py    # OutcomeLogPlugin (exchange_log)
```

## Known Risks

**Tool result ordering.** When the LLM requests multiple tool calls,
results arrive independently as task completions. Tool results are
batched via `channel.pending_tool_call_ids`: each result is appended
to the conversation as it arrives, but the LLM call is deferred until
the last pending result clears the set. This means the LLM sees all
results from a batch in a single turn. Partial failure (one tool errors)
still delivers all results; the failed tool returns an error string.
User messages can interleave mid-batch without disrupting the pending set.

**User message interleaving.** If a user sends a new message while tool
results are still pending, the conversation may interleave user
messages with tool results. The SerialQueue serializes processing (no
races), but ordering depends on arrival time.

**Tool sandboxing.** The `shell` tool runs commands without any
sandbox or privilege restriction, and `task_pipeline` (which calls the
same shell implementation), `read_file`, `write_file`, and `web_fetch`
are likewise unconfined. This was a scoping decision for the MVP:
Corvidae is a personal agent daemon on the user's own machine, and
sandboxing was not a priority for getting the loop working. It still
matters. The agent reads text it did not write (IRC, web pages, MCP
results), and an injected instruction runs with the daemon's full
privilege — the attacker is whoever can post in a channel the bot sits
in, not another tenant. [threat-model.md](threat-model.md) lays out the
exposure and an ordered fix list; `tools.disabled` and
`plugins.disabled` are the current per-deployment valves.
`harness/buster.yaml.in` is a partially narrowed profile (no subagent,
no MCP, no `shell`) but still carries `task_pipeline` and `write_file`.

## Unimplemented

The following items appear in earlier design documents but are not yet
implemented. Each needs discussion before proceeding.

### Hot-loading

The original design included a `ComponentLoader` with watchdog
filesystem watcher for hot-reloading plugin modules from a
`components/` directory. Not implemented. Transport plugins load via
setuptools entry points, like every other built-in plugin.

### Double-buffer compaction

An optimization where a new compacted prompt is built from the full log
in a second llama-server KV cache slot, warmed, and swapped in
seamlessly. Not implemented; current compaction is stop-the-world.

### Additional transports

BlueSky and other transports are future work. The hook system supports
them; the implemented transports are CLI, IRC, Signal, Discord, Zulip, and Web.
