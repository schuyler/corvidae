# Corvidae Plugin Guide

Corvidae plugins extend the agent daemon using [apluggy](https://pypi.org/project/apluggy/) (async pluggy). A plugin subclasses `CorvidaePlugin` and decorates methods with `@hookimpl` for each hook it handles.

```python
from corvidae.hooks import CorvidaePlugin, hookimpl
from corvidae.outbound import OutboundMessage

class GreetPlugin(CorvidaePlugin):
    @hookimpl
    async def on_message(self, channel, sender: str, text: str) -> None:
        # An impl may take any subset of the spec's args; trace_id is omitted here.
        if text.strip().lower() == "hello":
            await self.pm.ahook.send(
                channel=channel, event=OutboundMessage(text=f"Hello, {sender}!")
            )
```

`CorvidaePlugin.on_init` stores `pm` and `config` as instance attributes. Subclasses that need extra initialization override `on_init` and call `super().on_init(pm, config)` first.

## Plugin anatomy

A plugin subclasses `CorvidaePlugin` from `corvidae.hooks`. It only needs to implement the hooks it cares about — all hooks are optional. The `@hookimpl` decorator marks each implementation. Plugins use no-argument constructors; `pm` and `config` are received via the `on_init` hook.

```python
from corvidae.hooks import CorvidaePlugin, hookimpl

class MyPlugin(CorvidaePlugin):
    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        await super().on_init(pm, config)
        self.setting = config.get("my_plugin", {}).get("setting", "default")

    @hookimpl
    async def on_start(self, config: dict) -> None:
        # open runtime resources (connections, file handles, asyncio tasks)
        pass

    @hookimpl
    async def on_stop(self) -> None:
        # clean up resources
        pass
```

## Registering a plugin

Plugins are loaded via setuptools entry points. Declare them in `pyproject.toml`:

```toml
[project.entry-points.corvidae]
my_plugin = "my_package:MyPlugin"
```

`Runtime.start()` calls `pm.load_setuptools_entrypoints("corvidae")`, which instantiates each entry point class with no arguments and registers it. Then `install_plugin_hookspecs(pm)` installs [plugin-declared hooks](#declaring-hooks-in-a-plugin), the arg-binding guard checks every hookimpl, `validate_dependencies(pm)` runs, hookimpls with no spec are reported, and `pm.ahook.on_init(pm=pm, config=config)` broadcasts to all registered plugins.

All plugin classes must be instantiable with no arguments. `pm` and `config` are delivered via `on_init`.

**ChannelRegistry** is the exception: it is not an entry-point plugin. It is a plain class with no `@hookimpl` decorators, constructed explicitly in `Runtime.start()` and populated from config before the entry point plugins are loaded.

## Writing replacement plugins

To replace a built-in hook handler, implement the hook at default priority. Built-in handlers use `trylast=True` and serve as fallbacks.

```python
class FancyCompactionPlugin(CorvidaePlugin):
    @hookimpl
    async def resolve_compaction(self, channel, conversation, max_tokens):
        # Custom strategy
        await self.my_compaction(conversation, max_tokens)
        return True  # non-None stops the chain; built-in never runs
```

To observe without replacing, use `tryfirst=True` and return None:

```python
class CompactionObserver(CorvidaePlugin):
    @hookimpl(tryfirst=True)
    async def resolve_compaction(self, channel, conversation, max_tokens):
        self.record_metrics(conversation)
        return None  # chain continues to the actual compaction handler
```

This pattern applies to any `firstresult=True` hook. For broadcast hooks (where all handlers run regardless), `tryfirst` and `trylast` affect only execution order, not whether subsequent handlers run.

## Registering CLI subcommands

Plugins can add subcommands to the `corvidae` CLI by registering entries in the `corvidae.commands` entry point group. Each entry point maps a subcommand name to a `click.Command` (or `click.Group` for nested subcommands).

```toml
[project.entry-points."corvidae.commands"]
scaffold = "corvidae_scaffold:scaffold_command"
```

The entry point value must be a `click.Command` or `click.Group` object. `corvidae.main` discovers all registered entries at import time and adds them to the top-level `corvidae` group. If an entry point fails to load, a warning is logged and the subcommand is skipped — other subcommands are unaffected.

Subcommands are standard click commands managed by the `corvidae` click Group. Subcommands that need the agent runtime construct a `Runtime` instance and call `asyncio.run(runtime.run())`:

```python
import asyncio
import click
from corvidae.runtime import Runtime

@click.command("myplugin")
@click.option("--config", default="agent.yaml", help="Path to config file")
def myplugin_command(config):
    """Start corvidae with myplugin active."""
    runtime = Runtime(
        config_path=config,
        overrides={"channels": {"cli:local": {}}},
    )
    asyncio.run(runtime.run())
```

Utility subcommands that do not need the agent runtime omit the `Runtime` import entirely. They can perform their work synchronously without booting the plugin system.

Entry point modules in `corvidae.commands` are imported at dispatcher load time, before `Runtime.start()` configures logging or creates an event loop. These modules must not have module-level side effects that depend on logging configuration or a running event loop. Standard library imports (`asyncio`, `click`) and `corvidae.runtime` are safe at module level. Avoid importing modules with side effects (network connections, file I/O, logging configuration) at the top level — use lazy imports inside the command function for those.

The existing `corvidae` entry point group (for plugins), the `corvidae.commands` group (for subcommands) and the `corvidae.redact` group (for redact scrubbers, below) are distinct setuptools groups. Registering in one does not affect the others.

The built-in subcommand `scaffold` is registered by corvidae itself under `corvidae.commands`. Do not register an entry point named `scaffold` in the `corvidae.commands` group — a collision will silently overwrite the built-in command, causing unpredictable behavior.

Built-in `corvidae acp` (from `corvidae.channels.acp:acp_command`) starts an ACP v1 agent on stdio. It boots `Runtime` with `overrides={"_acp_mode": True}` so `AcpPlugin` owns the connection; without that flag the plugin stays inert. Install the optional extra first: add `--extra acp` to your `uv sync` line.

## Registering a redact scrubber

`corvidae redact` (see "Operator redaction" in [design.md](design.md)) tombstones `message_log` rows and cascades to memory. A plugin package that copies conversation text into a table of its own registers a scrubber under the `corvidae.redact` entry-point group, so a redaction reaches that copy too. The kernel never names the table.

```toml
[project.entry-points."corvidae.redact"]
exchange_log = "corvidae_cognition.outcome_log:redact_exchange_log"
```

The entry-point name labels the surface in the CLI output (`2 exchange_log rows`) and in skip notices. The object is an async function:

```python
async def redact_my_table(
    db: aiosqlite.Connection,  # the CLI's own connection (WAL, busy_timeout 5000)
    channel_id: str,           # one channel at a time
    msg_ids: list[int],        # the message_log ids just tombstoned in it
    dry_run: bool,             # count what would change, write nothing
    tombstone: str,            # the operator tombstone text to write
) -> int | None: ...
```

Every redact form (`message`, `range`, `memory`) calls each scrubber once per channel, after the message tombstones and the memory cascade have committed. Return the number of rows changed (or that would be, under `--dry-run`). Return `None` if your table isn't in this database (the daemon may never have loaded your plugin). The count is then 0, and one `SKIP` line naming your scrubber goes into the `notices` list a caller of `redact_messages` passes. The CLI doesn't print notices, so an operator sees only `0 <name> rows`. The scrubber commits its own writes. Probe for your table rather than creating it, because schema is daemon-owned. An exception propagates and aborts the command after the earlier steps have committed, so make the scrub idempotent: a re-run must finish the job and count nothing already scrubbed. The module is imported each time a redact form runs, inside its event loop, so keep it free of import-time side effects.

## Hook grammar

A hook's name tells you how its results are resolved. Every hookspec name
starts with a prefix that fixes its pluggy options, or is a bare imperative
verb on a short allowlist. `tests/test_hook_grammar.py` checks names against
pluggy options. Hooks a plugin declares itself also carry the owner's
plugin name after the prefix, can't be bare imperatives, and are checked
at startup; see [Declaring hooks in a plugin](#declaring-hooks-in-a-plugin).

| Prefix | pluggy spec | Resolution | Core uses result | Implementations |
|---|---|---|---|---|
| `on_`, `before_`, `after_` | broadcast | none | no | any number; side effects allowed, including on objects passed in |
| `should_` | broadcast | `resolve_reject_wins` | yes, `bool` | any number; `False` from any one rejects; `None` abstains |
| `resolve_` | `firstresult=True` | first non-`None` | yes | any number; first to answer wins, so order matters and the docstring says who answers |
| `transform_` | `firstresult=True` wrapper chain with a seed in `_SeedHooksPlugin` | chain | yes | any number, LIFO |
| bare imperative | broadcast | caller-defined over the result list | sometimes | any number; the docstring states what the caller does with results |

The observe class has three prefixes because `before_agent_turn`,
`before_compaction` and `after_persist_assistant` are observations placed
relative to a stage. All three mean the same thing to pluggy.

An observe hook may mutate an object it was handed (`after_persist_assistant`
strips `reasoning_content` in place; `before_agent_turn` appends to
`channel.conversation`), but its return value is never read. A transform
returns the replacement and the core uses it.

`resolve_channel` shows the other safe shape for `resolve_`: each
implementer answers only for ids under its own `transport:` prefix and
returns `None` for the rest, so the order pluggy calls them in cannot change
the answer.

The bare imperative allowlist is `send` (transports receive it through
`TransportPlugin`; see [Writing a transport](#writing-a-transport)),
`register_tools` (implementers append to the list passed in) and
`persist_message` (every implementation runs; the caller resolves the list
with `resolve_single_result`).

### Adding a hook

A hookspec earns a name when it is a decision point (veto, resolve,
transform), when it is an observation with at least two plausible
implementers, or when it is an observability hook that pairs with an
existing one (`on_llm_request` with `on_llm_response`). A single-consumer
observation does not get a hookspec; it gets an entry in `meta`, a return
value on an existing hook, or waits for its second consumer. Pick the name
from the table above. The test checks the prefix against the pluggy
options; whether the core reads a hook's return value is not something
pluggy exposes, so that half is checked in review.

### Declaring hooks in a plugin

A hook that only one plugin calls doesn't belong in the kernel's
`AgentSpec`. The plugin that calls it owns it instead. Set `hookspecs` on
the plugin class to a class of `@hookspec` methods:

```python
from corvidae.hooks import TransportPlugin, hookspec

# PanelRegistry stands for whatever object the owner hands implementers.
class WebConsoleSpec:
    @hookspec
    async def on_web_console_init(self, panels: PanelRegistry) -> None:
        """Panel plugins add their panels to ``panels``."""

class WebPlugin(TransportPlugin):
    hookspecs = WebConsoleSpec

    async def on_start(self, config):
        await self.pm.ahook.on_web_console_init(panels=self._panels)
```

`Runtime.start()` installs every declared spec right after entry points
load (`install_plugin_hookspecs`), before the arg-binding guard and
before `on_init`, so any plugin's `on_init` can already call the hook.
Startup fails, naming the hook, the plugin and the fix, when a
declaration breaks one of these rules:

- **Grammar, with the owner named.** The name starts with a grammar prefix
  from the table above, then the owning plugin's registered name and an
  underscore: `on_web_…`, `should_web_…`, `resolve_web_…`,
  `transform_web_…`. Bare imperatives stay kernel-only. The prefix still
  fixes the pluggy options (`resolve_`/`transform_` need
  `firstresult=True`; the others must not have it).
- **One owner per hook.** A name declared by two plugins, or by a plugin
  and `AgentSpec`, is an error in either registration order. To share a
  hook, depend on its owner and call it; any plugin can call any hook.
- **Transform seeds come from the owner.** A `transform_` hook needs an
  `@hookimpl(trylast=True)` on the owner itself that returns its input
  unchanged, which is what `_SeedHooksPlugin` provides for kernel transforms.

The owner segment is the plugin's registered name: its entry-point name,
or the `name=` passed to `pm.register`. Renaming an entry point renames
its hooks. A plugin registered without a name, or under a name that can't
appear in a Python identifier (`web-console`), can't own hooks. The
declaration is read from the class, so a base class can declare for a
subclass, but only one plugin registered from that class hierarchy can
own the hooks. A plugin that is a module rather than a class instance
can't declare hooks.

A plugin hook still has to earn its name under the admission rule in
[Adding a hook](#adding-a-hook). Owning it yourself changes where the spec
lives, not whether it should exist.

Implementing a plugin's hook is the same as implementing a kernel hook:
write a `@hookimpl` with the same name, and declare the dependency (see
[Plugin dependencies](#plugin-dependencies)). The hook's contract is
documented by its owner, not in [Available hooks](#available-hooks).

At startup, a hookimpl whose name matches no spec is logged at WARNING
(`plugin 'x' implements hook 'on_web_consol_init', which no loaded plugin
declares`), since it will never be called. Usually that's a typo. The
exception is an impl for a hook owned by one of your `optional_depends_on`
plugins that isn't loaded: that's expected, and is logged at DEBUG.

## Available hooks

### Lifecycle

| Hook | Type | When |
|------|------|------|
| `on_init(pm, config: dict)` | async broadcast | After all plugins are registered, before `on_start`. Use to store `pm`, read config values, and resolve references to other plugins. Do not create runtime resources here. |
| `on_start(config: dict)` | async broadcast | Once at startup, after `on_init`. Use to open runtime resources: DB connections, network clients, file handles, asyncio tasks. |
| `on_stop()` | async broadcast | On SIGINT/SIGTERM, before process exits (a SIGINT during startup is held until `on_start` finishes; a SIGTERM then ends the process without `on_stop`). Also after a failed startup, before the error propagates, so it may run when this plugin's `on_start` (or `on_init`) never ran or didn't finish: guard each resource with a None check. |
| `on_config_reload(config: dict)` | async broadcast | When `agent.yaml` is modified while the daemon is running. The config dict has been re-parsed, merged with CLI overrides, and validated before dispatch. |

`config` is the full parsed `agent.yaml` dict. The key `_base_dir` (a `Path`) is set by `Runtime.start()` pointing to the config file's directory.

`CorvidaePlugin.on_init` stores `pm` and `config` as instance attributes. Subclasses that override `on_init` must call `await super().on_init(pm, config)` to preserve this behavior.

#### Supporting hot-reload in plugins

Implement `on_config_reload` to re-read configuration values when `agent.yaml` changes at runtime. The hook fires on the event loop thread; implementations must not block.

```python
from corvidae.hooks import CorvidaePlugin, hookimpl

class MyPlugin(CorvidaePlugin):
    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        await super().on_init(pm, config)
        self.threshold = config.get("my_plugin", {}).get("threshold", 0.8)

    @hookimpl
    async def on_config_reload(self, config: dict) -> None:
        self.threshold = config.get("my_plugin", {}).get("threshold", 0.8)
```

`on_config_reload` receives the full re-parsed config dict, already merged with CLI overrides. Per-plugin errors are caught and logged by `ConfigWatcherPlugin`; a failing `on_config_reload` in one plugin does not prevent other plugins from receiving the reload.

The `config` parameter is optional in `on_config_reload` hookimpls. Pluggy forwards only the arguments declared in the hookimpl signature, so a hookimpl that omits `config` is called with no arguments.

### Messaging

| Hook | Type | When |
|------|------|------|
| `on_message(channel, sender: str, text: str, trace_id: str)` | async broadcast | Inbound message arrives; the transport mints and passes the id (see [Traces](#traces)) |
| `should_process_message(channel, sender: str, text: str, trace_id: str \| None)` | async broadcast, reject-wins | Gate: decide whether to process a message. Return False to reject, True to accept, None for no opinion. |
| `on_message_admitted(channel, trace_id: str, sender: str, text: str)` | async broadcast | After should_process_message admits a USER message |
| `on_message_rejected(channel, trace_id: str, sender: str, text: str)` | async broadcast | After should_process_message vetoes a USER message |
| `should_send_response(channel, text: str, emission: str, trace_id: str \| None, meta: dict)` | async broadcast, reject-wins | Outbound gate, before each emission (`final`, `progress`, `thinking`, `error`). Return False to withhold, None for no opinion. See [Output gating](#output-gating). |
| `on_message_persisted(channel, trace_id: str, rowid: int \| None, text: str, meta: dict)` | async broadcast | After the trace-originating message is persisted to the database. Fires once per originating item regardless of persistence outcome (`rowid` may be None). |
| `send(channel, event: Outbound)` | async broadcast | Outbound delivery: `event` is an `OutboundMessage`, `OutboundThinking`, `OutboundToolStatus` or `OutboundProgress` (`corvidae/outbound.py`). Transports receive it through `TransportPlugin` (see [Writing a transport](#writing-a-transport)) |
| `on_notify(channel, source: str, text: str, tool_call_id: str \| None, meta: dict \| None)` | async broadcast | Notification injected into channel |

Any plugin can send: `await self.pm.ahook.send(channel=channel, event=OutboundMessage(text="..."))`. `send` is broadcast to every plugin, and `TransportPlugin` drops the events for channels a transport does not own, so a sender never addresses a transport by name.

### Extension points

| Hook | Type | When |
|------|------|------|
| `register_tools(tool_registry: list)` | sync broadcast | During `on_start`, to collect tools |
| `on_agent_response(channel, request_text: str, response_text: str, trace_id: str \| None, meta: dict, logprobs: dict \| None)` | async broadcast | After agent produces a response |
| `before_agent_turn(channel, trace_id: str \| None, meta: dict)` | async broadcast | Before each LLM invocation |
| `after_persist_assistant(channel, message: dict)` | async broadcast | After assistant message is written to DB; plugins may mutate the in-memory dict |
| `persist_message(channel, message: dict, message_type: MessageType) -> int \| None` | async broadcast | After every `conv.append()` call; message is the untagged dict (no `_message_type` key). Persistence returns the inserted `message_log` rowid; **exactly one implementation may return non-None** (all others return None). Callers resolve with `resolve_single_result` and attach the rowid to the in-window message as `_db_id`. |
| `before_compaction(channel, compacted_ids: list[int], cut: int)` | async broadcast, awaited | After the compaction summary succeeded and before `replace_with_summary`. The window still holds the messages about to be dropped (`channel.conversation.messages[:cut]`; `cut` is moved off tool cycles where compaction can afford it, see design.md), so this is where to fork over that prefix (see "Forking the window"). Runs on the response path; failures are logged and never block compaction. |
| `on_compaction(channel, summary_msg: dict, retain_count: int, compacted_ids: list[int])` | async broadcast | After compaction replaces older messages with a summary; `summary_msg` is the untagged summary dict. `compacted_ids` carries the `message_log` rowids of exactly the removed messages (empty when unknown) — the consolidation range consumed by `MemoryPlugin`. |
| `resolve_channel(channel_id: str) -> Channel \| None` | async firstresult | A caller holding only a persisted `transport:scope` id (`SchedulerPlugin` firing a row after a restart) asks transports for the live channel. Return it for ids your transport owns, else None. Return None for your own id only when it cannot be built yet; the scheduler then retries. See [Channels](#channels). |

`after_persist_assistant` — the DB row is already written when this
hook fires. Mutations to `message` affect in-memory prompt construction
only; they do not update the persisted record.

`before_agent_turn` — messages injected via `channel.conversation.append()` inside this hook are passed through `persist_message` and persisted to the DB, unless they already carry a `_db_id` (i.e. their producer persisted them itself).

Window messages carry internal `_`-prefixed tags (`_message_type`, `_db_id`). Every serialization boundary (prompt build, compaction summarizer prep, persistence, JSONL log) strips all `_`-prefixed keys — internal tags never reach the LLM or the DB.

### Traces

A trace is everything one inbound item causes: its turns, its tool tasks, follow-up turns from their completions, and every LLM call in them. Each trace has an opaque `trace_id`. One exists for every inbound user message and for every notification-born queue item that isn't itself a continuation of an existing trace. The transport mints a user message's id with `corvidae.trace.mint_trace_id()` and passes it to `on_message`; the Agent mints at dequeue for a notification-born item with no inherited id. It travels alongside a `meta` dict — an extensible, framework-opaque metadata bag — through the messaging and extension-point hooks above, into `Task.trace_id`/`Task.meta`, and back out through the completion notification's `meta` (`TaskPlugin._on_task_complete` merges `{"task_id", "trace_id", **task.meta}`). This is how a tool cycle several hops deep still carries the trace id and metadata its originating message started with.

**Transport contract.** A transport mints a fresh, unique id per inbound message, passes it as `trace_id` to `on_message`, and matches `should_process_message`, `on_message_admitted`, `on_message_rejected` and `on_message_persisted` back to its message by that id, never by ordering or context inheritance. Import `mint_trace_id` from `corvidae.trace` (stdlib only), not from `corvidae.agent`. The Agent uses the id verbatim. Omitting the argument raises `pluggy.HookCallError`; a falsy id raises `ValueError`. The lifecycle hooks broadcast to every plugin, and unlike `send` they get no filtering from `TransportPlugin`, so each of a transport's lifecycle hookimpls opens with an ownership check (`if not self.owns(channel): return`; Discord and Signal spell it `channel.matches_transport(...)`, which is the same test for them) and ignores ids it never minted.

`on_message_admitted` or `on_message_rejected` fires before `on_message` returns, unless the agent drops the message without either (for example, it is not ready yet). State a transport took before the call (a typing slot, a pending read receipt, an unsent-message entry) can therefore be released in a `finally` when the id was not admitted:

```python
tid = mint_trace_id()
self._pending[tid] = state          # taken before the call
try:
    await self.pm.ahook.on_message(
        channel=channel, sender=sender, text=text, trace_id=tid,
    )
finally:
    if tid not in self._admitted:   # set by on_message_admitted
        self._pending.pop(tid, None)
    self._admitted.discard(tid)

@hookimpl
async def on_message_admitted(self, channel, trace_id, sender, text):
    if not self.owns(channel):
        return
    self._admitted.add(trace_id)
```

The framework itself never stamps meaning into `meta` — it is a pass-through. Cognition plugins (see below) use the convention `meta["origin"]` to record where a queue item came from ("user" recognized by subscribing to `on_message_admitted`, or an explicit string like `"reminder"` stamped by a producer); that convention is plugin-side, not a core concept.

`SchedulerPlugin` is the producer that stamps it for clock-driven items. A due reminder fires as

```python
await self.pm.ahook.on_notify(
    channel=channel, source="reminder", text=framed_text, tool_call_id=None,
    meta={"origin": "reminder", "schedule_id": row_id},
)
```

and core carries `meta` untouched to `on_message_persisted`, `before_agent_turn`, `on_agent_response`, every tool Task in the cycle, and the usage-log attribution. A notification like this, with no inherited `meta["trace_id"]`, originates its own trace, so it also gets a fresh `max_turns` budget (reset only when no tool cycle is in flight).

Producers that start a background task in `on_start` and inject notifications should wait for `pm.get_plugin("agent").started` before calling `on_notify`: `on_start` is broadcast to every plugin concurrently, and `Agent` starts after the broadcast.

### Observability

| Hook | Type | When |
|------|------|------|
| `on_llm_request(role, model, request_id, message_count, tool_count, attribution)` | async broadcast | Immediately before an LLM chat-completion call |
| `on_llm_response(role, model, request_id, usage, latency_ms, attribution, error)` | async broadcast | Exactly once per LLM call, after success or terminal failure |
| `on_metrics(name, value, tags)` | async broadcast | Whenever any plugin emits a metric event |

`on_llm_request` / `on_llm_response` fire from the `LLMClient` chokepoint
(via an observer injected by `LLMPlugin`), so **every** LLM call in the
system is covered — turn-loop, compaction, and subagent calls alike.
`request_id` pairs a request with its response. Retried transient attempts
do not fire `on_llm_response`; one request pairs with exactly one response.
`usage` is the response envelope's `usage` field verbatim (or `None`);
`error` is `None` on success and the exception string on terminal failure.
The request payload is summarized (message/tool counts), never shipped
wholesale. Observer failures are logged and never break the LLM call.

`attribution` is a snapshot of `corvidae.attribution.get_attribution()` —
a contextvar dict describing what the current code path is doing on whose
behalf. Built-in callers set `stage` (`"turn"`, `"compaction"`,
`"subagent"`, `"consolidation"`) and `channel_id`; the dict is open for
plugin-defined fields. `MemoryPlugin`'s consolidation stage adds `trigger`
(`"compaction"` or `"idle"`), naming which of the two consolidation
triggers is driving the current `on_llm_request`/`on_llm_response` pair —
a plugin consuming the attribution snapshot on a `stage="consolidation"`
call can rely on `trigger` being present. Set it around a logical
operation and always reset in a `finally`:

```python
from corvidae.attribution import reset_attribution, set_attribution

token = set_attribution(
    stage="consolidation", channel_id=channel.id, trigger="idle"
)
try:
    await client.chat(messages)
finally:
    reset_attribution(token)
```

Contextvars snapshot at `asyncio.create_task` time, so attribution does
not automatically reach long-lived workers. `Task` (corvidae/task.py)
captures `contextvars.copy_context()` at creation and the TaskQueue worker
runs the work inside that context — attribution set by the enqueuer is
visible in the task body.

`on_metrics` carries a named numeric value with dimensional tags
(StatsD/OpenTelemetry common denominator; dotted names like
`llm.tokens.prompt`, `llm.latency_ms`). Emit metrics from any hook or tool
with `await self.pm.ahook.on_metrics(name=..., value=..., tags=...)`.
**Reentrancy constraint:** never emit from inside an `on_metrics`
implementation — pluggy dispatches back into the same implementation and
recurses forever.

### Hook result resolution

These hooks return a value. Hooks marked `firstresult=True` (sequential) stop at the first non-None return. Wrapper chain hooks use `firstresult=True` with an identity seed; implementations use `@hookimpl(wrapper=True)` to compose transforms. Broadcast gate hooks use `resolve_reject_wins` for result resolution.

| Hook | Strategy | Returns | Behavior |
|------|----------|---------|----------|
| `should_process_message(channel, sender, text, trace_id)` | reject-wins (broadcast) | `bool \| None` | Any `False` vetoes the message; any `True` (with no `False`) accepts; `None` if all defer |
| `should_send_response(channel, text, emission, trace_id, meta)` | reject-wins (broadcast) | `bool \| None` | Any `False` withholds the emission; anything else sends it |
| `resolve_llm_error(channel, error)` | `firstresult=True` | `str \| None` | First non-None string wins; chain stops. If all return None, the default message names the failure class (timeout, unreachable server, or an HTTP status) and asks the user to try again. `LLMPlugin` registers a `trylast` handler that explains an unloaded router model, any non-2xx `LLMHTTPError` (naming the status, an actionable hint, and a short provider body), and connection/timeout failures that point at `llm.main.base_url`; it defers (`None`) for anything else. To inspect HTTP failures yourself, check `isinstance(error, corvidae.llm.LLMHTTPError)` and read `error.status`, `error.body`, and `error.model`. |
| `resolve_compaction(channel, conversation, max_tokens)` | `firstresult=True` | `bool \| None` | First non-None return stops the chain. `CompactionPlugin` uses `trylast` (returns True). Third-party plugins run at default priority. |
| `transform_tool_result(tool_name, result, channel)` | `firstresult=True` wrapper chain | `str \| None` | Wrappers compose transforms in LIFO order. The seed returns the input unchanged. Non-wrapper hookimpls short-circuit the seed via firstresult. |
| `transform_display_text(channel, text, result_message)` | `firstresult=True` wrapper chain | `str \| None` | Wrappers compose transforms in LIFO order. The seed returns the input text unchanged. Non-wrapper hookimpls short-circuit the seed via firstresult. |
| `resolve_conversation(channel)` | `firstresult=True` | `list[dict] \| None` | First non-None result wins. `PersistencePlugin` uses `trylast` as fallback. Called once when a channel's conversation is first initialized, and by `AcpPlugin` for `session/load`. |
| `persist_message(channel, message, message_type)` | `resolve_single_result` (broadcast) | `int \| None` | Every implementation runs. `PersistencePlugin` returns the `message_log` rowid; others return None. More than one non-None is a configuration error: logged, and the first is used. |

`resolve_compaction` — sequential hook (`firstresult=True`). The first handler returning a non-None value stops the chain. `CompactionPlugin` runs last (`trylast=True`) as the default strategy and returns True. A third-party plugin at default priority runs before it — if it returns non-None, `CompactionPlugin` is never called.

`transform_tool_result` fires for all tool calls — it is invoked from `dispatch_tool_call` in `corvidae/tool.py`, which is used by both the main agent loop and background subagent loops. It does not fire for pre-dispatch errors (JSON parse failure or unknown tool name).

`transform_display_text` — `result_message` is the raw assistant message dict from the LLM response. It may contain `reasoning_content` if the model produces thinking tokens. `text` is the string content extracted from that message. Wrapper implementations receive the chain result (the output of inner wrappers or the seed) via `yield`. Non-wrapper implementations that return `None` defer to the seed, which returns `text` unchanged.

Example hook returning a value:

```python
@hookimpl
async def should_process_message(self, channel, sender: str, text: str, trace_id: str | None) -> bool | None:
    if sender in self.blocklist:
        return False
    return None  # no opinion
```

## Tool registration

Tools **must** be coroutine functions (`async def`) with type-annotated parameters. `Tool.from_function()` enforces this at registration time and raises `TypeError` if given a sync callable. The docstring's first line becomes the tool description for the LLM. See [Avoid blocking the event loop](#avoid-blocking-the-event-loop) for what `async def` means in practice — the check rejects sync functions, but it cannot detect blocking I/O called from inside an async one. Register tools in `register_tools`:

```python
from corvidae.hooks import hookimpl
from corvidae.tool import Tool

class WeatherPlugin:
    @hookimpl
    def register_tools(self, tool_registry: list) -> None:
        async def get_weather(city: str, units: str = "metric") -> str:
            """Get current weather for a city."""
            return await fetch_weather(city, units)

        tool_registry.append(Tool.from_function(get_weather))
```

`Tool.from_function()` infers the tool name from `fn.__name__` and generates a JSON schema from the type annotations. Parameters prefixed with `_` are excluded from the schema — they are injected at call time, not supplied by the LLM.

## Context injection (ToolContext)

Tools that need channel context, the task queue, or the tool call ID declare a `_ctx: ToolContext` parameter. Corvidae injects it automatically; the LLM never sees it.

```python
from corvidae.tool import Tool, ToolContext

class MyPlugin:
    @hookimpl
    def register_tools(self, tool_registry: list) -> None:
        async def enqueue_work(instructions: str, _ctx: ToolContext) -> str:
            """Enqueue background work."""
            from corvidae.task import Task

            async def work():
                return await do_something(instructions)

            task = Task(
                work=work,
                channel=_ctx.channel,
                tool_call_id=_ctx.tool_call_id,
                description=instructions[:80],
            )
            await _ctx.task_queue.enqueue(task)
            return f"Enqueued task {task.task_id}"

        tool_registry.append(Tool.from_function(enqueue_work))
```

`ToolContext` attributes:
- `channel: Channel | None` — the channel this tool call is executing on
- `tool_call_id: str` — the LLM-assigned call ID for this invocation
- `task_queue: TaskQueue | None` — the task queue (None if TaskPlugin is not registered)

## Avoid blocking the event loop

Every tool runs on the single main asyncio event loop. A synchronous blocking call inside a tool — `requests.get`, `time.sleep`, a slow `open(...).read()`, a sync database driver — stalls the entire process: the main agent, every task-queue worker, and every channel queue all wait until the call returns.

The `Tool.from_function()` async check rejects functions defined with `def` instead of `async def`, but it cannot see what your `async def` does internally. If your code calls a blocking library, you are responsible for moving that call off the event loop with `asyncio.to_thread()`. The pattern used by the built-in file tools (`corvidae/tools/files.py`) is the model: a small sync helper does the I/O, the async tool awaits it via `to_thread`.

```python
import asyncio
from corvidae.hooks import hookimpl
from corvidae.tool import Tool

def _lookup_sync(symbol: str) -> str:
    # Blocking call — sync HTTP client, sync DB driver, slow file read, etc.
    import requests
    return requests.get(f"https://api.example.com/{symbol}", timeout=5).text

class QuotePlugin:
    @hookimpl
    def register_tools(self, tool_registry: list) -> None:
        async def lookup(symbol: str) -> str:
            """Look up a stock quote."""
            return await asyncio.to_thread(_lookup_sync, symbol)

        tool_registry.append(Tool.from_function(lookup))
```

For common cases the standard library and ecosystem offer native async alternatives — prefer them when available:

- HTTP — use `aiohttp` (see `corvidae/tools/web.py`).
- Subprocesses — use `asyncio.create_subprocess_shell` / `create_subprocess_exec` (see `corvidae/tools/shell.py`).

As a last-resort safety net, `execute_tool_call` detects a sync callable registered as a bare `Tool(name=..., fn=sync_fn, schema=...)` (bypassing `Tool.from_function()`) and wraps the call in `asyncio.to_thread()`, logging a warning. Treat that warning as a bug to fix in the plugin, not a supported registration path.

## What the kernel exports to plugins

A plugin outside the `corvidae` package (the `corvidae-cognition` plugins,
or your own) should import only these kernel modules. This is the API the
cognition plugins use, and `tests/test_kernel_boundary.py` fails if
`corvidae_cognition` imports any other kernel module. Changing a signature
here is a plugin-API change and needs a line in this guide.

| Module | Names |
|---|---|
| `corvidae.hooks` | `CorvidaePlugin`, `TransportPlugin`, `hookimpl`, `get_dependency`, `resolve_single_result` |
| `corvidae.fork` | `fork_agent`, `quote`, `window_db_ids` (see [Forking the window](#forking-the-window)) |
| `corvidae.attribution` | `set_attribution`, `reset_attribution` |
| `corvidae.tuning` | `config_value`, `resolve_tunable` |
| `corvidae.context` | `MessageType`, `ContextWindow`, `count_tokens` |
| `corvidae.task` | `Task` |
| `corvidae.tool` | `Tool`, `tool_to_schema` |
| `corvidae.persistence` | `PersistencePlugin`, `add_column_if_missing` |
| `corvidae.llm_plugin` | `LLMPlugin` |
| `corvidae.clock` | `Clock`, `Timer` (see [Acting on time and quiet](#acting-on-time-and-quiet)) |
| `corvidae.idle` | `IdleTracker`, `IdleWatch` |
| `corvidae.outbound` | `OutboundMessage`, `OutboundThinking`, `OutboundToolStatus`, `OutboundProgress` (for senders and transports) |

The kernel never imports a plugin outside it. A plugin that needs another
plugin resolves it at runtime with `get_dependency(self.pm, "<name>", Type)`.

## Acting on time and quiet

To run something later, or when a channel or the system has been quiet,
use the kernel clock and idle tracker. They are not hooks. Declare both
dependencies and resolve them in `on_init`:

```python
from corvidae.clock import Clock
from corvidae.hooks import CorvidaePlugin, get_dependency, hookimpl
from corvidae.idle import IdleTracker


class Tidy(CorvidaePlugin):
    depends_on = frozenset({"clock", "idle"})

    @hookimpl
    async def on_init(self, pm, config):
        self.pm = pm
        self.clock = get_dependency(pm, "clock", Clock)
        self.idle = get_dependency(pm, "idle", IdleTracker)

    @hookimpl
    async def on_start(self, config):
        self.watch = self.idle.watch_channels(600, self.on_quiet, name="tidy")
        self.timer = self.clock.call_later(3600, self.hourly, name="tidy.hourly")

    @hookimpl
    async def on_stop(self):
        self.watch.cancel()
        self.timer.cancel()

    async def on_quiet(self, channel_id: str) -> None: ...
    async def hourly(self) -> None: ...
```

`call_later` and `call_at` are one-shot; re-arm from the callback for a
periodic job. `call_soon(cb, name=...)` runs once on the next loop step.
`watch_channels` repeats every `repeat` seconds (default: the period) while
the channel stays quiet, and `watch_system` does the same for the whole
system. `repeat` must be a positive finite number, so `watch_channels(0, cb,
name=...)` without `repeat=` (or a negative period) raises `ValueError`. To
learn that a channel just drained, as a transport does for a working
indicator, watch with `watch_channels(0, cb, name=..., repeat=3600)`: a zero
period fires at settle, and the callback sees every channel, so filter on
the id. A zero-period callback that awaits something outside the clock
blocks `FakeClock.settle()` in tests until it finishes. Cancel watches and
timers in `on_stop`; the Runtime cancels stragglers afterwards. Callbacks
must let `asyncio.CancelledError` through. An idle or timer callback is
upkeep: it does plugin-owned work and must never start a turn or enqueue work on a channel (principle 13), so it
never calls `on_notify`. Starting a turn on time is agency, which is the
scheduler's job: its firings go through `on_notify` and run a full turn. In
tests, `tests/fake_clock.py`'s `install_kernel_clock(pm)` registers
a fake clock and a real tracker, so time moves only when the test says so.
Full semantics: [design.md](design.md#clock-and-idle-tracker-clockpy-idlepy).

## Plugin dependencies

Declare `depends_on` as a class attribute (a set of plugin names). `validate_dependencies()` raises `RuntimeError` at startup if any declared dependency is not registered, or if the dependency graph contains a cycle. The error message includes the full cycle path, e.g. `Dependency cycle detected: a -> b -> a`.

```python
class MyPlugin(CorvidaePlugin):
    depends_on = {"agent", "task"}
```

To get a typed reference to a dependency:

```python
from corvidae.hooks import get_dependency
from corvidae.tool_collection import ToolCollectionPlugin

tools_plugin = get_dependency(self.pm, "tools", ToolCollectionPlugin)
registry = tools_plugin.get_registry()
```

`get_dependency` raises `RuntimeError` if the plugin is not found and `TypeError` if it is the wrong type.

A plugin that uses another plugin when it's there and works without it
declares `optional_depends_on` instead. A missing optional dependency is
not an error, and optional dependencies aren't part of cycle detection.
Look the plugin up with `pm.get_plugin(name)` and treat `None` as absent
(it's `None` whether the plugin is disabled or not installed):

```python
class ProfilerPanel(CorvidaePlugin):
    optional_depends_on = frozenset({"web"})

    @hookimpl
    async def on_web_console_init(self, panels):
        panels.add(...)  # only called when web is loaded
```

If the owner is absent, the impl stays registered with no spec and is
never called. See [Declaring hooks in a plugin](#declaring-hooks-in-a-plugin).

## Injecting context before agent turns

`before_agent_turn` fires before every LLM call. Use it to inject contextual information (memory retrieval, current state, etc.) into the conversation. The hook receives the current turn's trace id and metadata dict:

```python
from corvidae.context import MessageType
from corvidae.hooks import CorvidaePlugin, hookimpl

class MemoryPlugin(CorvidaePlugin):
    @hookimpl
    async def before_agent_turn(self, channel, trace_id: str | None, meta: dict) -> None:
        notes = await self.fetch_relevant_notes(channel.id, trace_id)
        if notes:
            channel.conversation.append(
                {"role": "user", "content": f"[Context]\n{notes}"},
                message_type=MessageType.CONTEXT,
            )
```

`MessageType.CONTEXT` entries survive compaction — compaction only summarizes `MESSAGE` entries.

Prefer routing CONTEXT through the admission funnel instead of appending directly: `pm.get_plugin("funnel").admit(channel, conv, "<source>", entries)` gives you window dedupe, per-source token budgets, and the mandatory data-not-instructions framing for free (see [FunnelPlugin](cognition.md#funnelplugin-packagescorvidae-cognitioncorvidae_cognitionfunnelpy) in cognition.md; the funnel is part of the cognition extra, so check that `pm.get_plugin("funnel")` is not None). Direct appends bypass all three.

## Forking the window

A background process that wants the agent's own context and voice, and on
a single-slot llama-server wants to reuse the cached prompt instead of
evicting it, can run as a **fork** of the channel's window (GitHub issue
#51, [design.md](design.md#forks)). A fork is one LLM call: the main
turn's render of the window, up to a cut, plus one `user` instruction.

```python
from corvidae.fork import fork_agent, window_db_ids

class MyPlugin(CorvidaePlugin):
    async def review(self, channel, rowid: int) -> str | None:
        agent = fork_agent(self.pm, "review")   # None unless agent.forks enables "review"
        conv = channel.conversation
        if agent is None or conv is None or rowid not in window_db_ids(conv.messages):
            return None                         # caller makes its standalone call
        try:
            result = await agent.fork(
                channel, "Pause here and review your last answer. Reply in JSON.",
                extra_body={"response_format": {...}},
            )
        except Exception:
            logger.warning("review fork failed; using the standalone call", exc_info=True)
            return None
        return result.text
```

- `Agent.fork(channel, instruction, *, cut=None, extra_body=None)` returns
  `ForkResult(text, message, usage, model)`. `model` is the model the fork
  request was sent with (`LLMClient.request_model(body)`, so an
  `extra_body["model"]` override shows), for plugins that record how their
  output was made; `Agent.fork` always sets it, so it is None only when a
  `ForkResult` is built without it. `cut=None` forks
  the whole window;
  `cut=k` forks `messages[:k]` (use `snap_cut_to_tool_boundary` if `k`
  might split a tool cycle, or the prefix won't match the cache).
- It uses `llm.main`, the agent's tool schemas with `tool_choice: "none"`,
  and the channel's runtime overrides, so the prefix matches the main
  turn. Don't pass `reasoning_effort` or anything else the chat template
  renders into the prompt: that breaks the cached prefix. Bound cost with
  `agent.forks.reasoning_budget_tokens` instead.
- It mutates nothing, persists nothing, fires no hooks and never runs
  tools: a reply with `tool_calls` (or a call written out as text) raises `ForkRefused`; no window or no
  client raises `ForkUnavailable`. Always keep a standalone fallback.
- Set your own `stage` attribution around the call; `Agent.fork` adds
  `mode="fork"`. Tag your standalone call `mode="standalone"`.
- Check containment yourself: a fork only makes sense while the rows your
  process is about are still in the window (`window_db_ids`).
- Read the instruction from static config (`corvidae.tuning.config_value`),
  not `resolve_tunable`, unless you want the agent to be able to rewrite it.
- Stage names are opaque strings. List yours in `agent.forks.stages` (or
  leave it null for all stages).

## Message gating and lifecycle

The `should_process_message` gate fires before a message enters the conversation. Implement it to filter messages by sender, rate limit, or apply channel muting:

```python
class FilterPlugin(CorvidaePlugin):
    @hookimpl
    async def should_process_message(self, channel, sender: str, text: str, trace_id: str | None) -> bool | None:
        if sender in self.muted_users:
            return False  # Reject
        return None  # No opinion; other plugins decide
```

When a message is admitted or rejected, plugins are notified via `on_message_admitted` or `on_message_rejected`. These fire exactly once per message. The `on_message_persisted` hook fires after the trace-originating message (the initial USER message or a standalone notification) has been written to the database, regardless of persistence outcome — it receives the database rowid (nullable), the originating text, and the trace's metadata dict:

```python
class LogPlugin(CorvidaePlugin):
    @hookimpl
    async def on_message_admitted(self, channel, trace_id: str, sender: str, text: str) -> None:
        await self.log_event("message_admitted", trace_id, sender)

    @hookimpl
    async def on_message_persisted(self, channel, trace_id: str, rowid: int | None, text: str, meta: dict) -> None:
        await self.log_event("message_persisted", trace_id, rowid, meta)
```

## Channels

A channel is a `transport:scope` pair like `irc:#general` or `cli:local`. Each channel has its own `ContextWindow` initialized on first message.

```python
channel.id              # "irc:#general"
channel.transport       # "irc"
channel.scope           # "#general"
channel.conversation    # ContextWindow | None
channel.matches_transport("irc")  # True
```

Channels are created on-demand when messages arrive, or pre-registered in `agent.yaml`:

```yaml
channels:
  irc:#general:
    system_prompt: "You are the channel bot."
    max_context_tokens: 16000
```

The `irc` transport is provided by `IRCPlugin` (registered via the `corvidae` entry point group). Its config block:

```yaml
irc:
  host: irc.libera.chat       # IRC server (default irc.libera.chat)
  port: 6667                   # IRC port (default 6667)
  nick: corvidae               # Bot nickname (default corvidae)
  tls: false                   # Use TLS (default false)
  channels: ["#general"]       # Channels to join
  message_chunk_size: 400      # Max UTF-8 bytes per IRC message (default 400)
```

The `signal` transport is provided by `SignalPlugin` (same entry point group). It speaks JSON-RPC to a `signal-cli daemon` that is already running — corvidae never starts one. Its config block:

```yaml
signal:
  socket: /run/corvidae/signal.sock   # signal-cli daemon socket (required)
  account: "+15550001111"             # the bot's own number (required)
  allow:                              # default-deny; E.164 or ACI
    - "+15551234567"
  message_chunk_size: 2000            # Max UTF-8 bytes per message (default 2000)
```

Omitting the block leaves the plugin loaded but inert. Signal channel scopes are always the sender's ACI (`signal:8f2c1c9a-…`), never a phone number — `sourceNumber` is absent under phone-number privacy, and a scope that can vanish would fork the channel's persisted history. Numbers in config are resolved forward to ACIs after the transport connects. See [signal-ops.md](signal-ops.md) for provisioning and [configuration.md](configuration.md) for the full key reference.

### `resolve_channel` for transport authors

Something other than an inbound message sometimes needs a channel: `SchedulerPlugin` fires a reminder after a restart, holding only the persisted id. It calls `registry.get(channel_id)`, then the `resolve_channel` hook, then `registry.get_or_create(transport, scope)` from the split id. If your transport creates channels with config a bare `get_or_create` would miss, implement the hook, and keep it default-free (the startup arg-binding guard, `tests/test_hook_arg_binding.py`, rejects a defaulted `channel_id`: pluggy would silently pass your default instead of the caller's id):

```python
@hookimpl
async def resolve_channel(self, channel_id: str) -> Channel | None:
    transport, sep, scope = channel_id.partition(":")
    if transport != "mytransport" or not sep or not scope:
        return None          # not ours: defer to the next implementation
    return self._channel_for(scope)   # the same path inbound messages take
```

`SignalPlugin` implements it so an ACI it has not seen since startup still gets its E.164 alias's `ChannelConfig`. When your transport owns the id but cannot apply its config yet (Signal before its first alias resolution), return None. The scheduler sees that a plugin with `transport` equal to the id's prefix implements the hook, so it holds the row and retries every few seconds rather than falling back to a default-configured channel. Don't return None for an id you own and will never be able to build: the row would retry forever. IRC and CLI don't need it; the registry fallback is already correct for them. Channels no transport owns (`internal:heartbeat`) also take the fallback, and since `TransportPlugin.send` delivers only to the transport that owns a channel, replies there go only to the log and `sessions.db`.

### Writing a transport

Subclass `TransportPlugin` (in `corvidae.hooks`), set `transport` to the name your channel ids start with, and override the outbound methods your transport can display. They are plain methods, not hookimpls, and each takes the channel and an event from `corvidae.outbound`:

| Method | Event | Fields |
|---|---|---|
| `send_message` | `OutboundMessage` | `text`, `latency_ms` (None if unknown) |
| `send_thinking` | `OutboundThinking` | `text` |
| `send_tool_status` | `OutboundToolStatus` | `tool_name`, `status` (`"dispatched"` or `"completed"`), `args_summary`, `result_summary`, `tool_call_id` (None if unknown) |
| `send_progress` | `OutboundProgress` | `text` |

The base class implements the `send` hook once: it drops events for channels the transport does not own, then dispatches by event type. Methods you don't override are no-ops, which is how a chat transport skips reasoning and tool status. `TransportPlugin` also declares `depends_on = frozenset({"registry"})`.

```python
from corvidae.hooks import TransportPlugin
from corvidae.outbound import OutboundMessage

class MyTransport(TransportPlugin):
    transport = "mytransport"

    async def send_message(self, channel, event: OutboundMessage) -> None:
        await self.client.post(channel.scope, event.text)
```

The filter is `owns(channel)`, which by default is `channel.matches_transport(self.transport)`. Override it only to narrow it: `WebPlugin` also requires that it is enabled. Inbound hooks are not filtered for you; open each inbound hookimpl with `if not self.owns(channel): return`.

### ACP transport

`AcpPlugin` (`corvidae` entry point `acp`) speaks Agent Client Protocol v1
over stdio when launched via `corvidae acp`. It does **not** connect under
`serve` / `cli` unless `config["_acp_mode"]` is set (only the `acp`
command does that). Startup misconfiguration — a missing config file, an
invalid YAML document, or an invalid config value — exits non-zero with a
single line naming the config path (`Config error:` from Runtime for an
unreadable or malformed YAML file, `Error:` otherwise), so an editor shows
the diagnosis instead of a traceback.

Optional YAML block (all keys optional today; presence is enough to
document intent):

```yaml
# acp:
#   # Reserved for future agentCapabilities / logging hints.
#   # Mode is not enabled by YAML alone — use `corvidae acp`.
```

**bb smoke (`customAcpAgents`):**

1. `uv sync --extra acp --extra dev --extra cognition` (add `--extra discord` if you use it)
2. Point bb at Corvidae, e.g. command `uv` with args
   `run --directory /path/to/corvidae corvidae acp` (or a wrapper script).
3. `bb thread spawn --provider acp-corvidae --prompt "…" ` (provider name
   per your bb config).
4. Confirm a reply and any tool rows in the timeline.

Channel ids are `acp:<sessionId>` and stay stable across `session/load`
and `session/resume` (history is keyed on that id in `sessions.db`).
Advertised capabilities: `loadSession`, plus `sessionCapabilities.list` /
`resume` / `close`. `session/delete` is not advertised. Disable the
plugin with `plugins.disabled: [acp]` if needed.

**Client input this transport rejects or ignores.** `session/new`,
`session/load`, and `session/resume` require `cwd` to be an existing
directory; anything else returns JSON-RPC `invalidParams`. The `cwd` is
recorded for `session/list`, but tools still run in the process working
directory — a differing `cwd` is logged at WARNING rather than acting as a
chdir. `mcpServers` and `additionalDirectories` are accepted and ignored
with a WARNING; configure `mcp:` in `agent.yaml` for MCP servers. Prompts
must carry at least one text block: image, audio, and resource blocks are
logged and dropped, and a prompt with no text at all returns
`invalidParams` instead of an empty turn. An unknown session id is
`resourceNotFound` for `prompt`, `load`, `resume`, and `close` alike.

A prompt always ends. If the client connection dies while the final
message is being delivered, the turn still completes with `end_turn` and
the failed delivery is logged at WARNING: the reply is in `message_log`
and a later `session/load` replays it (#99). If the turn never starts
(the enqueue fails), its future is cleared so a later prompt cannot
inherit it. A second `session/prompt` on a session that already has one
in flight is `invalidRequest`; the first prompt is left untouched, and a
client that wants to steer sends `session/cancel` first (#93; the policy
and its revisit trigger are in `docs/prd.md`).

The ACP loop owns stdin/stdout, so the process stops when the loop ends:
a client disconnect initiates a normal shutdown, and a loop crash is
logged and then exits non-zero, so a client sees a failed provider
instead of a live process with a dead transport.

## Registration order

Plugins are loaded via `pm.load_setuptools_entrypoints("corvidae")`. The entry point loading order is non-deterministic — it is not guaranteed to match any specific sequence. The following plugins are registered:

```
registry          (ChannelRegistry)     — explicit, before entry points load
persistence       (PersistencePlugin)   — entry point
jsonl_log         (JsonlLogPlugin)      — entry point
core_tools        (CoreToolsPlugin)     — entry point
cli               (CLIPlugin)           — entry point
irc               (IRCPlugin)           — entry point
signal            (SignalPlugin)        — entry point
acp               (AcpPlugin)           — entry point (inert unless `_acp_mode`)
task              (TaskPlugin)          — entry point
subagent          (SubagentPlugin)      — entry point
mcp               (McpClientPlugin)     — entry point
llm               (LLMPlugin)           — entry point
compaction        (CompactionPlugin)    — entry point
thinking          (ThinkingPlugin)      — entry point
runtime_settings  (RuntimeSettingsPlugin) — entry point
tools             (ToolCollectionPlugin) — entry point
memory            (MemoryPlugin)        — entry point (cognition extra)
memory_tools      (MemoryToolsPlugin)   — entry point (cognition extra)
funnel            (FunnelPlugin)        — entry point (cognition extra)
goals             (GoalsPlugin)         — entry point (cognition extra)
agent             (Agent)               — entry point
config_watcher    (ConfigWatcherPlugin) — entry point
```

`ChannelRegistry` is constructed and registered explicitly before entry points load. Its `agent_defaults` attribute and channel config are populated from config before the `on_init` broadcast runs.

**Tool collection:** `ToolCollectionPlugin.on_start` uses `@hookimpl(trylast=True)` so it fires after all other `on_start` hooks, then calls `register_tools` to collect tools from every registered plugin. The non-deterministic loading order does not affect tool collection because `trylast=True` ensures `ToolCollectionPlugin.on_start` always runs last in the broadcast.

**Startup order:** `Runtime.start()` broadcasts `on_init`, then `on_start`. `ToolCollectionPlugin.on_start` runs last (trylast) to collect tools. `Runtime.start()` then calls `agent.on_start(config=config)` explicitly after the broadcast completes. `Agent.on_start` does not have `@hookimpl` — it is called only by `Runtime.start()`. A transport may deliver inbound messages from its own `on_start`: `Agent.on_message` holds them until the Agent has started (the awaiting `on_message` call simply takes longer to return) rather than dropping them. If the Agent stops before starting, the held call returns without admission and a WARNING names the channel and trace id. Adding `@hookimpl` to `Agent.on_start` would cause double initialization.

**Shutdown order:** `Runtime.stop()` calls `agent.on_stop()` first (drains queues), then `pm.ahook.on_stop()` to tear down all other plugins, then stops the idle tracker and closes the kernel clock, which cancels any timers left running.

**Failed startup:** if `Runtime.start()` raises an exception (a plugin's `on_init` or `on_start`, say; not a config error, which exits before any plugin loads), `Runtime.run` runs the `on_stop` broadcast, stops the idle tracker and closes the clock, under the same 3 s bound, then re-raises so the process exits non-zero. `agent.on_stop()` is skipped. A plugin's `on_stop` must therefore cope with resources its `on_start` never opened.

`ChannelRegistry`, the `Clock` (`"clock"`) and the `IdleTracker` (`"idle"`) are registered by `Runtime.start()` before the entry points load, so any plugin can depend on them. See [Acting on time and quiet](#acting-on-time-and-quiet).

## Plugin disable

Disable entry-point plugins via the `plugins.disabled` config key:

```yaml
plugins:
  disabled:
    - compaction          # disable the built-in compaction plugin
```

Names must match the entry-point name from `[project.entry-points.corvidae]` in `pyproject.toml`. `Runtime.start()` calls `pm.set_blocked(name)` for each entry before loading entry points. Blocked plugins are not instantiated.

Only entry-point plugins can be disabled this way. Manually registered plugins (e.g., `ChannelRegistry`) cannot be blocked via config.

This is a startup-time mechanism. It does not interact with hot-reload.

Right after entry points load, `Runtime.start()` logs a `"plugins loaded"`
line with `loaded` (the sorted names that actually registered, not the
blocked ones) and
`blocked` (the `plugins.disabled` list from config, verbatim). Compare the
two to confirm a disable took effect: a misspelled name lands in
`blocked` while the plugin it was meant to block still appears in
`loaded` — `pm.set_blocked` on a name that matches nothing is silent
otherwise.

## Hook exception safety

Broadcast hook calls from `Agent` are wrapped in `try/except`. If a plugin raises an exception from `before_agent_turn`, `after_persist_assistant`, `on_agent_response`, or `send`, the exception is logged at WARNING or ERROR level and processing continues. Plugins do not need to catch their own exceptions to protect the queue consumer.

For `firstresult=True` hooks (`resolve_compaction`, `resolve_conversation`, `resolve_llm_error`), pluggy propagates the first handler's exception immediately — the chain stops and no further handlers run. The call sites in `Agent` handle these as follows:

- `resolve_compaction` — wrapped in try/except; failure is logged at WARNING and the turn continues without compaction.
- `resolve_conversation` — not wrapped; an exception propagates to `_process_queue_item`, which will fail the queue item.
- `resolve_llm_error` — wrapped in its own try/except inside `_run_turn`'s error handler; failure is logged at WARNING and the default error message is sent instead.

For `should_process_message` (broadcast, resolved with `resolve_reject_wins`), exceptions propagate to the call site.

For `should_send_response`, the Agent catches any exception, logs it at WARNING and sends the emission anyway (fail open). Because apluggy gathers a broadcast without `return_exceptions`, one raising implementation discards every sibling's verdict for that emission, so a gate that can fail should catch and log its own errors and return None.

For `transform_display_text` and `transform_tool_result` (wrapper chain hooks), pluggy propagates exceptions from wrapper implementations to the call site. `transform_display_text` is wrapped in try/except in `_resolve_display_text`.

## Output gating

`should_send_response(channel, text, emission, trace_id, meta)` is the outbound mirror of `should_process_message`: return `False` to withhold, `None` for no opinion. `emission` is `final` (the reply), `progress` (text alongside dispatching tool calls), `thinking` (reasoning) or `error` (the LLM-error fallback); treat anything else as no opinion. `text` is exactly what would be sent. Where each fires and what a veto does to the stored row and the window is in [design.md](design.md#the-output-gate); in short, a vetoed final is kept in the window as `MessageType.UNSENT` and never sent, and any other veto suppresses only that send.

Origin is not a kernel parameter: read `meta["origin"]` if your policy depends on it, and record the traces you see in `on_message_admitted` if you need to recognise user exchanges.

```python
class QuietHoursPlugin(CorvidaePlugin):
    @hookimpl
    async def should_send_response(self, channel, text, emission, trace_id, meta):
        # Hold back self-initiated finals overnight; never a reply to a person.
        if emission == "final" and meta.get("origin") == "heartbeat" and self.is_night():
            return False
        return None
```

If you write a transport with history replay, skip `message_type == "unsent"` rows, and use `corvidae.context.unsent_marker_emission(content)` to hold back progress or thinking that a marker says was suppressed. Cognition's `GatePlugin` is the reference implementation ([cognition.md](cognition.md#gateplugin)).

## Async considerations

All broadcast hooks are `async`. Corvidae uses apluggy's `pm.ahook.*` for async dispatch.

For hooks with `firstresult=True` (`resolve_compaction`, `resolve_conversation`, `resolve_llm_error`), `pm.ahook.<hook>(...)` returns a single value (or None) directly:

```python
result = await pm.ahook.resolve_conversation(channel=channel)
```

For `transform_display_text` and `transform_tool_result`, use `@hookimpl(wrapper=True)` to participate in the wrapper chain. The seed plugin returns the input value unchanged as the innermost result; each wrapper receives that result, optionally transforms it, and returns the modified value:

```python
from corvidae.hooks import hookimpl

@hookimpl(wrapper=True)
def transform_display_text(self, **kwargs):
    result = yield
    if result is not None:
        return my_transform(result)
    return result
```

For the `should_process_message` broadcast hook, call `pm.ahook.<hook>(...)` and pass the result list to `resolve_reject_wins`:

```python
from corvidae.hooks import resolve_reject_wins

results = await pm.ahook.should_process_message(channel=channel, ...)
result = resolve_reject_wins(results)
```

`@hookimpl(tryfirst=True)` and `@hookimpl(trylast=True)` markers are respected by apluggy's dispatch and affect execution order. For `firstresult=True` hooks, `tryfirst` handlers run before default-priority handlers, which run before `trylast` handlers. The chain stops at the first non-None return.

The async-only rule extends to **tool functions** as well as hooks: `Tool.from_function()` enforces it at registration time, and any blocking I/O inside a tool must be wrapped with `asyncio.to_thread()`. See [Avoid blocking the event loop](#avoid-blocking-the-event-loop).

## Stock tools

These tools are registered by built-in plugins. They are available to the LLM in every standard Corvidae deployment.

All tool results are truncated at `MAX_TOOL_RESULT_CHARS` (default 100,000 characters) by `execute_tool_call` in `corvidae/tool.py`. The truncation appends `[truncated — N chars total]` so the LLM knows output was cut. Override via config:

```yaml
tools:
  max_result_chars: 100000  # read by ToolCollectionPlugin during on_start
```

The legacy key `agent.max_tool_result_chars` is still accepted but deprecated; a warning is logged at startup when it is present.

### CoreToolsPlugin tools

Registered by `CoreToolsPlugin` (entry point name: `core_tools`).

| Tool | Parameters | What it does |
|------|------------|--------------|
| `shell` | `command: str` | Runs a shell command and returns combined stdout/stderr. Times out after `tools.shell_timeout` seconds (default 30). Returns `"(no output)"` if the command produces none. Non-zero exit codes are appended to the output. |
| `read_file` | `path: str` | Reads a file and returns its text content. Returns an error string for missing files, directories, unreadable files, or files larger than `tools.max_file_read_bytes` bytes (default 1 MB). |
| `write_file` | `path: str`, `content: str` | Writes `content` to `path`, creating parent directories as needed. Returns a confirmation with the byte count, or an error string on failure. |
| `web_fetch` | `url: str` | Fetches a URL via HTTP GET and returns the response body as text. Times out after `tools.web_fetch_timeout` seconds (default 15). Truncates responses at `tools.web_max_response_bytes` characters (default 50,000, independent of `MAX_TOOL_RESULT_CHARS`). A network failure names the URL and the failure class. |
| `web_search` | `query: str`, `max_results: int` (optional) | Searches the web via DuckDuckGo and returns formatted results with titles, URLs, and snippets. Defaults to 8 results per page. |
| `task_pipeline` | `definition: str` | Executes a task graph defined in YAML or JSON. The definition must contain a `tasks` key with a list of objects, each having `name`, `command`, and optionally `depends_on`. Tasks run in topological order; failed tasks block their dependents. Returns a status summary; a task failure names the task and the command. |

**CoreToolsPlugin config:**

```yaml
tools:
  shell_timeout: 30                  # seconds before shell command is killed
  web_fetch_timeout: 15              # seconds before web request is aborted
  web_max_response_bytes: 50000      # response body truncation limit
  web_search_max_results: 8          # max results returned per search query
  max_file_read_bytes: 1048576   # file size limit (1 MB)
```

### SubagentPlugin tools

Registered by `SubagentPlugin` (entry point name: `subagent`).

| Tool | Parameters | What it does |
|------|------------|--------------|
| `subagent` | `instructions: str`, `description: str`, `_ctx: ToolContext` | Enqueues a background task that runs a full agent loop with `instructions` as its prompt. Returns immediately with the enqueued task ID. Results are delivered via `on_notify` when the task completes. |

`subagent` requires `ToolContext` (injected automatically — not an LLM parameter). It excludes itself from the subagent's tool registry to prevent recursion. The subagent uses the `llm.background` config block if present, falling back to `llm.main`.

### RuntimeSettingsPlugin tools

Registered by `RuntimeSettingsPlugin` (entry point name: `runtime_settings`).

| Tool | Parameters | What it does |
|------|------------|--------------|
| `set_settings` | `settings: dict` | Updates per-channel runtime settings. Accepts LLM inference parameters, framework parameters and dotted plugin tunables; any other key rejects the call. Pass `null` for a key to revert it to the static config value. Returns the current overrides after the update. |

### SchedulerPlugin tools

Registered by `SchedulerPlugin` (entry point name: `scheduler`). All three act on `_ctx.channel` only and take no channel argument, so a speaker on one channel cannot schedule a message to another ([threat-model.md](threat-model.md)).

| Tool | Parameters | What it does |
|------|------------|--------------|
| `remind_me` | `when: str`, `text: str` | Creates a one-shot reminder on the calling channel. `when` is a relative duration (`20m`, `2h`, `3d`, `1h30m`), an ISO 8601 time (no offset means `scheduler.timezone`), or `HH:MM` for the next occurrence; a bare number is refused as ambiguous. The tool does the date arithmetic and returns the resolved absolute time, e.g. `Reminder #12 set for Fri 2026-10-23 14:22 PDT (in 20m).` Refuses past times, empty text, text over 2000 characters, and more than `scheduler.max_reminders_per_channel` pending reminders. |
| `list_reminders` | — | Pending rows on the calling channel, earliest first, with ids and due times. Operator jobs targeting the channel are listed as read-only. |
| `cancel_reminder` | `id: int` | Cancels a tool-created reminder on the calling channel. Refuses operator jobs; another channel's row reads as not found. |

## Bundled plugins

The kernel's plugins follow. The cognition plugins (`MemoryPlugin`,
`MemoryToolsPlugin`, `FunnelPlugin`, `AppraisalPlugin`, `CritiquePlugin`,
`OutcomeLogPlugin`, `GatePlugin`, `HeartbeatPlugin`) ship in the `cognition` extra and are described in
[cognition.md](cognition.md), which has reference sections for the memory,
funnel and outcome-log plugins.

These plugins ship with Corvidae and are registered via entry points as part of the default daemon. They can be omitted if the application does not need their functionality; each degrades gracefully when absent.

### McpClientPlugin (`corvidae/mcp_client.py`)

Connects to external [MCP](https://modelcontextprotocol.io/) servers and
exposes their tools to the agent loop. Entry point name: `"mcp"`.

Implements three hooks:

- `on_start` — connects to all servers listed under `mcp.servers`
  in `agent.yaml`, fetches their tool lists, and builds cached `Tool` instances.
  Runs in the broadcast; completes before `ToolCollectionPlugin.on_start`
  (trylast) calls `register_tools`.
- `register_tools` — appends the cached tools to `tool_registry`.
- `on_stop` — signals each server's owner task to close its session and
  transport (enter and exit happen in the same task), all servers at once.

**Config:**
```yaml
mcp:
  servers:
    filesystem:
      transport: stdio
      command: npx
      args: ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/projects"]
      env:                     # optional: extra env vars for the subprocess
        NODE_ENV: production
      tool_prefix: "fs"        # optional: prefix for tool names; defaults to server name
      timeout_seconds: 30      # optional: per-call timeout; default 30
      connect_timeout_seconds: 30  # optional: startup handshake bound; default 30

    remote_api:
      transport: sse
      url: "http://localhost:8001/sse"
      timeout_seconds: 60

    wiki:
      transport: streamable_http
      url: "http://localhost:8090/mcp"
      headers:
        Authorization: "Bearer ${OTTERWIKI_MCP_TOKEN}"
```

`streamable_http` is MCP's current HTTP transport; `sse` is the older one.
Both take an optional `headers` map, sent on every request. `${VAR}` in a
header value is replaced from the daemon's environment at connect time, so
a token never has to sit in `agent.yaml`. A variable that is unset, empty, or holds a newline or NUL
skips that server with one WARNING naming the server, header and variable,
and the server is not queued for the retry below (the environment
can't change without a restart). The same holds for every config error:
unknown `transport`, missing `command`/`url`, non-positive
`connect_timeout_seconds`. Header values never reach the log. The
plugin builds the HTTP client itself (`create_mcp_http_client`) and closes
it with the server.

Tool names are prefixed to avoid collisions: a server named `filesystem`
produces tools like `filesystem__read_file`. Set `tool_prefix: ""` to
disable prefixing. If two servers produce the same tool name, the first
wins and the duplicate is skipped with a WARNING. A server that does not
complete its handshake within `connect_timeout_seconds` is skipped with a
WARNING and its subprocess terminated; startup continues without it. A
server that fails to connect at startup is kept and retried by a background
loop on the kernel clock (the plugin declares `depends_on = {"clock"}`): 10 s,
doubling to 300 s, reset on success, stopped in `on_stop`. An MCP tool call
also triggers a retry unless one is already running. A successful retry adds
its tools to the cached list and the shared registry; a failed retry logs a
WARNING naming the server, the exception type and the next delay, and leaves
the current tool call untouched. A reconnect rebuilds the registry
and the running Agent picks the tools up on its next turn (#109); a plugin
that calls `ToolCollectionPlugin.rebuild_registry` at runtime reaches the
Agent the same way.

**Without this plugin:** MCP servers cannot be used. No tools are affected
if the `mcp:` config key is absent — the plugin no-ops.

### ThinkingPlugin (`corvidae/thinking.py`)

Strips `<think>...</think>` blocks and `reasoning_content` from LLM
output. Entry point name: `"thinking"`.

Implements three hooks, driven by the resolved `reasoning_history`
(`drop` | `exchange` | `keep`):

- `before_agent_turn` — applies the policy to the whole in-memory window
  before each LLM call: `drop` strips every assistant message's
  `reasoning_content`, `exchange` only those before the last `role: user`
  message, `keep` none.
- `after_persist_assistant` — under `drop`, strips the just-appended
  message. The DB copy is already written; this only affects subsequent
  prompt builds.
- `transform_display_text` (`@hookimpl(wrapper=True)`) — receives the
  chain result and calls `strip_thinking` on it unconditionally. Returns
  the stripped string. Operates as a sync wrapper; does not take
  `channel`, `text`, or `result_message` parameters directly.

**Without this plugin:** `<think>` blocks pass through to the channel
verbatim, and `reasoning_content` remains in in-memory history
regardless of the `reasoning_history` config value.

### CompactionPlugin (`corvidae/compaction.py`)

Compacts conversation history when it approaches the channel's
`max_context_tokens` limit. Entry point name: `"compaction"`.

Implements one hook:

- `resolve_compaction` (`trylast=True`) — fires before each LLM call. Checks whether the token estimate exceeds `compaction_threshold * max_tokens`. If so, and if the conversation has more than `min_messages_to_compact` messages, summarizes older messages to fit within `compaction_retention * max_tokens`, moving the cut off any tool cycle that straddles it where that still leaves a useful compaction (design.md, CompactionPlugin step 3). With `agent.forks` on for `compaction`, the summary comes from a fork of the dropped prefix (`agent.compaction_fork_prompt`); otherwise, or if the fork fails, from the standalone summarizer. Fires `before_compaction` before truncating and `on_compaction` after. Returns True to stop the `firstresult` chain. Runs after any third-party handlers at default priority.

Token counting uses tiktoken (cl100k_base encoding) via `count_tokens()` in `corvidae/context.py`. When tiktoken is unavailable, a character-based fallback is used (3.5 chars per token). `agent.chars_per_token` is deprecated and only takes effect in the fallback path.

**Config:**
```yaml
agent:
  compaction_threshold: 0.8      # compact when token estimate exceeds this fraction of max_context_tokens
  compaction_retention: 0.5      # retain this fraction of max_context_tokens after compaction
  min_messages_to_compact: 5     # skip compaction if conversation has this many messages or fewer
  # chars_per_token: 3.5         # deprecated; only used when tiktoken is unavailable
```

**Without this plugin:** conversations grow without bound. The LLM will
receive an error from the API when the context limit is exceeded.

### PersistencePlugin (`corvidae/persistence.py`)

Opens the SQLite database, runs schema migrations, and sets the journal mode.
Entry point name: `"persistence"`. Implements `on_start`,
`on_stop`, `resolve_conversation`, `persist_message`, and `on_compaction`.

**Config:**
```yaml
daemon:
  session_db: sessions.db       # path to SQLite database file (default "sessions.db")
  sqlite_journal_mode: wal      # SQLite journal mode (default "wal");
                                # allowed values: delete, truncate, persist, memory, wal, off
```

**Without this plugin:** `resolve_conversation` returns no history; conversation
history is not persisted across restarts.

### JsonlLogPlugin (`corvidae/jsonl_log.py`)

Writes an append-only JSONL log of conversation events alongside the SQLite
store. Each `persist_message` and `on_compaction` call produces one
JSON line in a per-channel file. Entry point name: `"jsonl_log"`.

Implements three hooks:

- `on_start` — reads `daemon.jsonl_log_dir` and creates the directory.
  If the key is absent the plugin is a no-op for the rest of its lifetime.
- `persist_message` — writes a record with fields `ts`, `channel`,
  `type` (message type string), and `message` (the untagged message dict).
- `on_compaction` — writes a record with `type: "summary"` and the untagged
  summary dict.

File names are derived from the channel ID with `/` and `:` replaced by `_`.
File handles are opened in append mode and kept open until `on_stop`.

**Config:**
```yaml
daemon:
  jsonl_log_dir: logs/   # path relative to the config file; omit to disable
```

**Without this plugin:** no JSONL log is written. Conversation history is
still persisted in SQLite by `PersistencePlugin`.

### MetricsPlugin, UsageLogPlugin, MetricsJsonlPlugin (`corvidae/metrics.py`)

The observability sinks. All three are fail-soft: a metering failure is
logged with a traceback and never breaks an LLM call.

- **`MetricsPlugin`** (entry point `"metrics"`) — consumes `on_llm_response`
  and emits `on_metrics` events: `llm.tokens.prompt`, `llm.tokens.completion`,
  `llm.tokens.total` (each skipped if the usage dict lacks it),
  `llm.latency_ms` (always), and `llm.errors` (value 1.0, on terminal
  failure). Tags: `{"role", "model", "stage", "channel"}` — stage and
  channel come from the call's attribution snapshot.
- **`UsageLogPlugin`** (entry point `"usage_log"`, depends on
  `"persistence"`) — consumes `on_llm_response` and writes one row per LLM
  call into the `usage_log` SQLite table (in the persistence plugin's
  database): timestamp, request id, role, model, stage, channel id,
  trace id (NULL for calls outside any trace), trigger,
  mode (`fork`/`standalone` for stages that can fork, else NULL), token
  counts, cached tokens, latency, and error.
  The table is created lazily on first use (`CREATE TABLE IF NOT EXISTS`).
- **`MetricsJsonlPlugin`** (entry point `"metrics_jsonl"`) — consumes
  `on_metrics` and appends one JSON line per event
  (`{"ts", "name", "value", "tags"}`) to the file named by
  `daemon.metrics_jsonl`. A complete no-op when the key is unset.

**Config:**
```yaml
daemon:
  metrics_jsonl: metrics.jsonl   # path relative to the config file; omit to disable
```

**Without these plugins:** the `on_llm_request`/`on_llm_response` hooks
still fire from LLMPlugin's observer; nothing records them.

### LLMPlugin (`corvidae/llm_plugin.py`)

Owns the `LLMClient` instance lifecycle. Entry point name: `"llm"`. Other plugins retrieve clients via
`get_dependency(pm, "llm", LLMPlugin)`.

Implements two hooks:

- `on_start` — reads `llm.main` (required) and `llm.background` (optional),
  creates `LLMClient` instances, and starts their aiohttp sessions.
- `on_stop` — closes all aiohttp sessions.

Each created client gets a hook observer injected that fires
`on_llm_request`/`on_llm_response` (with the client's role and the current
attribution snapshot) around every chat call. `corvidae/llm.py` itself
stays free of plugin-system imports — the observer seam is a plain object
with async `request`/`response` methods, defined in `llm_plugin.py`.

`get_client(role)` returns the client for `"main"` or `"background"`. If no
background client is configured, `get_client("background")` falls back to
the main client.

**Config:**
```yaml
llm:
  main:                        # required
    base_url: https://api.openai.com/v1
    model: gpt-4o
    api_key: sk-...            # optional; can also use environment variable
    extra_body: {}             # optional: extra fields merged into request body
    max_retries: 3             # optional (default 3)
    retry_base_delay: 2.0      # optional (default 2.0)
    retry_max_delay: 60.0      # optional (default 60.0)
    timeout: 120               # optional: request timeout in seconds
  background:                  # optional — absent means use llm.main
    base_url: https://api.openai.com/v1
    model: gpt-4o-mini
```

**Without this plugin:** `Agent` and `CompactionPlugin` cannot create LLM
clients and will raise `RuntimeError` during startup.

### TaskPlugin (`corvidae/task.py`)

Owns the `TaskQueue` and delivers task results via the `on_notify` hook.
Entry point name: `"task"`. If that broadcast raises for a tool task, it
logs the failure and asks the Agent (`deliver_tool_error`) to queue an
error tool result instead, unless the real result was already queued, so
the tool cycle still finishes. The Agent's own no-task-queue error path
falls back to `deliver_tool_error` the same way.

**Config:**
```yaml
daemon:
  max_task_workers: 4       # concurrent task workers (default 4)
  completed_task_buffer: 100  # number of completed task records to retain in memory
```

**Without this plugin:** tool calls that return asynchronously (e.g., `subagent`)
cannot complete. `Agent` logs an error when tool dispatch is attempted
without a `TaskQueue`.

### RuntimeSettingsPlugin (`corvidae/tools/settings.py`)

Registers the `set_settings` tool, which lets the agent update per-channel
LLM inference parameters and framework settings at runtime. Entry point name: `"runtime_settings"`.

Implements one hook:

- `register_tools` — appends the `set_settings` tool to `tool_registry`.

The `set_settings` tool accepts a `settings` dict. Supported keys include
LLM inference parameters (`temperature`, `top_p`, `top_k`,
`frequency_penalty`, `presence_penalty`, `max_tokens`, `seed`), framework
parameters (`max_turns`, `max_context_tokens`, `reasoning_history`),
and dotted plugin tunables (`critique.*`, `appraisal.*`; see
configuration.md). Pass `null` for a key to clear the override and revert to
the static config value. `system_prompt` is always blocked. Additional keys
can be blocked via `agent.immutable_settings` in `agent.yaml`. Any other
undotted key (`messages`, `model`, `tools`, …) rejects the whole call; only
the inference parameters reach the LLM request body. See
[threat-model.md](threat-model.md), finding 1.

**Config:**
```yaml
agent:
  immutable_settings: [temperature]   # keys the agent cannot change at runtime
```

**Without this plugin:** the `set_settings` tool is not available. Channel
settings can only be configured statically in `agent.yaml`.

### ToolCollectionPlugin (`corvidae/tool_collection.py`)

Collects tools from all registered plugins and owns the `ToolRegistry`.
Entry point name: `"tools"`.

Implements one hook:

- `on_start` (trylast=True) — calls the sync `register_tools` broadcast after
  all other `on_start` hooks have completed. Builds a `ToolRegistry` from the
  collected items, then drops any tool named in `tools.disabled` — the
  single chokepoint every registered tool passes through, so there are no
  per-plugin opt-outs. Reads `tools.max_result_chars` (or the deprecated
  `agent.max_tool_result_chars`) to configure the per-call result truncation
  limit.

`Agent` retrieves the registry via
`get_dependency(pm, "tools", ToolCollectionPlugin)`. The `trylast=True`
marker guarantees all tool providers have fully initialized before collection
runs.

Other plugins can access the registry:
```python
from corvidae.hooks import get_dependency
from corvidae.tool_collection import ToolCollectionPlugin

tools_plugin = get_dependency(self.pm, "tools", ToolCollectionPlugin)
registry = tools_plugin.get_registry()
tools_dict, schemas = tools_plugin.get_tools()
```

**Without this plugin:** `Agent` cannot retrieve tools and will raise
`RuntimeError` during startup.

### SchedulerPlugin (`corvidae/scheduler.py`)

Clock-driven producer of reminders and operator jobs (the heartbeat is cognition's `HeartbeatPlugin` now; see [cognition.md](cognition.md#heartbeatplugin)). Entry point name: `"scheduler"`; `depends_on = {"registry", "persistence", "clock"}`. Owns the `schedule` table in `sessions.db` (created on first use) and one timer task, which sleeps until the earliest due row and is woken early whenever a row changes. A due row fires through `on_notify` with `source` and `meta["origin"]` set to `reminder`, so persistence, critique eligibility, attribution, and the outbound send work unchanged. Jobs under `scheduler.jobs` sync by name on start and on every reload. Full behaviour: [design.md](design.md#schedulerplugin); keys: [configuration.md](configuration.md#scheduler--reminders-and-periodic-jobs).

It follows the two patterns a plugin owning persistent state and a background task needs. Schema creation is lazy on first use (`PersistencePlugin.db` may not be open during its `on_start`, as with `MemoryPlugin`). The timer is an `asyncio.Task` cancelled in `on_stop` (as with `ConfigWatcherPlugin`). It sleeps on the kernel clock, so its tests use a fake clock and never sleep.

**Without this plugin:** no reminders and no scheduled jobs; the three tools are absent.

### ConfigWatcherPlugin (`corvidae/config_watcher.py`)

Polls `agent.yaml` for mtime changes and dispatches `on_config_reload` to all registered plugins when the file changes. Entry point name: `"config_watcher"`.

Implements three hooks:

- `on_init` — reads `_config_path`, `_cli_overrides`, and `daemon.config_poll_interval` from the config dict.
- `on_start` — records the file's current mtime (to suppress a reload on the first poll) and starts the background polling task.
- `on_stop` — cancels the polling task and waits for it to finish.

On each poll cycle, when the mtime has changed:
1. The file is parsed as YAML.
2. CLI overrides are deep-merged on top of the new YAML.
3. The result is validated (must be a dict with `llm.main`).
4. `ChannelRegistry.agent_defaults` is updated from the new `agent` section.
5. `load_channel_config` is called: it registers new exact entries, updates `tools.disabled` on channels already registered, and replaces transport-wide entries (`signal:*`), re-matching them on every live channel. A malformed section changes no channel.
6. `on_config_reload` is dispatched to each plugin individually. A plugin that raises logs an error but does not block the other plugins.

**Config:**
```yaml
daemon:
  config_poll_interval: 2.0   # seconds between mtime checks (default 2.0)
```

See [Hot-reload](configuration.md#hot-reload) in the configuration reference for which settings update on reload and which require restart.

**Without this plugin:** `agent.yaml` changes are not detected at runtime. Reload requires a daemon restart.
