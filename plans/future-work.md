# Future Work

Ideas registered for potential future implementation. Not prioritized.

## Systemd / journald logging

Running corvidae as a headless systemd service. Current logging goes to
stderr via `StructuredFormatter`. Questions to answer:

- Should `StructuredFormatter` emit structured JSON when stdout is not a
  tty (or when a config flag is set)? journald can ingest structured
  fields via `systemd.journal` or JSON on stdout.
- Do we need a `corvidae.service` unit file in the repo?
- `_DEFAULT_LOGGING` in `corvidae/logging.py` currently targets stderr
  with a human-readable format. A journald mode would want either no
  formatter (journald adds timestamps) or JSON output.
- Signal handling (`main.py`) already catches SIGTERM — verify clean
  shutdown under systemd `TimeoutStopSec`.

## SQLite schema migrations

`init_db` in `persistence.py` runs `CREATE TABLE IF NOT EXISTS` on
startup. There is no versioning or migration path. If the schema
changes, existing databases will silently diverge.

- Consider a `schema_version` table and a migration runner.
- Scope: corvidae is a personal daemon, not a multi-tenant service.
  Migrations could be as simple as sequential SQL scripts applied on
  startup when `schema_version < current`.

## JSONL log rotation

`JsonlLogPlugin` opens per-channel `.jsonl` files and appends
indefinitely. No rotation, size limits, or cleanup.

- Options: built-in rotation (by size or date), or rely on external
  `logrotate` with SIGHUP to reopen handles.
- If built-in: `on_conversation_event` could check file size before
  writing and rotate when a threshold is crossed. Would need to close
  and reopen the handle.
- If external: add a SIGHUP handler (or a hook) that calls
  `_close_all_handles()` so logrotate's `postrotate` can trigger
  reopening.

## Split CoreToolsPlugin into discrete components

`CoreToolsPlugin` (`corvidae/tools/__init__.py`) bundles shell, file,
and web tools into a single plugin. Each has independent config, an
independent implementation module, and no shared state except the plugin
instance holding config values.

Proposed split:

- `ShellToolPlugin` — registers `shell` tool. Owns `_shell_timeout`.
- `FileToolsPlugin` — registers `read_file`, `write_file`. Owns
  `_max_file_read_bytes`.
- `WebToolsPlugin` — registers `web_fetch`. Owns `_session`,
  `_web_fetch_timeout`, `_web_max_response_bytes`. Manages the
  `aiohttp.ClientSession` lifecycle.

Benefits: each can be independently omitted from a config, tested in
isolation, and doesn't carry config for unrelated tools. The
implementation modules (`shell.py`, `files.py`, `web.py`) are already
separate — only the plugin wiring is bundled.

Cost: three plugin registrations in `main.py` instead of one. Minor.

## Remove dead ChannelRegistry reference from PersistencePlugin

`PersistencePlugin` fetches `ChannelRegistry` via `get_dependency` in
`on_start` and stores it as `self._registry`, but never reads it. This
was needed when `PersistencePlugin` owned `ensure_conversation` (which
resolved channel config); after the context window refactor, that
responsibility moved to `AgentPlugin`. Remove `_registry`, the
`get_dependency` call, and the `ChannelRegistry` import.

## Standardize plugin constructor signature

Plugins inconsistently accept `pm` in `__init__`. Some take it (e.g.,
`AgentPlugin`, `CompactionPlugin`, `SubagentPlugin`), some don't (e.g.,
`CoreToolsPlugin`, `McpClientPlugin`). Plugins that don't take `pm`
can't use `get_dependency` outside of hook calls, which leads to
workarounds like inline imports or deferred lookups.

Proposed: define a `Plugin` protocol (or ABC) that requires `pm` as a
constructor argument and exposes it as a typed attribute:

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Plugin(Protocol):
    pm: PluginManager
    depends_on: set[str]

    def __init__(self, pm: PluginManager) -> None: ...
```

Benefits:
- Type checkers catch plugins that forget `pm` or `depends_on`.
- `get_dependency` can return a `Plugin`-typed result with `pm`
  guaranteed present.
- `main.py` registration becomes uniform: every plugin gets `pm`.

Cost: updating ~5 plugins that currently take no args or different args.
Mechanical change, no behavioral impact. Could be folded into Part 5
of the agent decomposition or done independently.

## Plugin coupling concerns

Analysis of how plugins depend on each other and where the coupling may
cause problems as the codebase grows.

### ChannelRegistry serves two unrelated roles

It's a channel factory (`get_or_create`, used by transports) and a
config resolver (`resolve_config`, used by AgentPlugin). Transports
never call `resolve_config`; AgentPlugin never calls `get_or_create`.
They're bundled because both involve channels, but the interfaces are
disjoint. If ChannelRegistry grows, splitting factory from config
resolution would give each dependent a narrower interface.

### Transports own channel creation but not channel identity

A transport decides the scope string (`"#lex"`, `"local"`) and calls
`get_or_create`, but the `Channel` dataclass and its `transport:scope`
ID format are defined in `channel.py`. If a future transport needs
different scoping (e.g., Signal group chats identified by UUID), it's
constrained by the `Channel` dataclass shape. Fine for now — the
dataclass is minimal — but transport-specific concerns could leak into
the shared type.

### Broadcast-filter pattern in send_message

Every transport must check `channel.matches_transport()` and return
early in `send_message`. pluggy has no way to route a hook call to a
specific implementor — broadcast is all it offers. If a new transport
forgets the filter, it'll try to send every message. This is the pattern
most likely to produce a bug when adding a third transport.

A transport registry that routes `send_message` to the right transport
by `channel.transport` would eliminate the pattern. Could be a thin
wrapper around the hook call in AgentPlugin, or a new hookspec that
takes `transport` as a discriminator.

### Config resolution is scattered

`_base_dir`, `chars_per_token`, `max_tool_result_chars` are read from
the config dict in `AgentPlugin._start_plugin`. LLM config is read
there too. Tool config in `CoreToolsPlugin.on_start`. Channel config in
`ChannelRegistry.resolve_config`. No single place parses or validates
the full config — each plugin grabs what it needs from the raw dict.
Config errors surface at different times during startup depending on
which plugin reads them first. A config validation pass before
`on_start` would catch errors earlier and in one place.

### Priority

1. ~~**Duplicated tool dispatch**~~ — completed.
2. **AgentPlugin decomposition** — Parts 1–4 done, Part 5 (rename)
   remaining. See `plans/agent-decomposition.md`.
3. ~~**TaskPlugin undeclared dependency**~~ — completed.
4. ~~**agent_loop.py re-exports**~~ — completed.
5. ~~**SubagentPlugin coupling**~~ — completed (Parts 3–4).
   SubagentPlugin now depends on `"llm"` and `"tools"` directly.
6. ~~**Idle detection inversion**~~ — completed (Part 2). Agent fires
   `on_idle` push-based; IdleMonitorPlugin is a pure consumer.
7. **Config validation** — address when config grows or errors become
   a problem.
8. **Broadcast-filter** — only matters when adding a third transport.
9. **ChannelRegistry split** — address if the class starts growing.
10. **Channel identity** — revisit when adding a non-IRC/CLI transport.

## Tool plugin scaffolding script

A CLI script (or corvidae subcommand) that generates a new Python
package for an external tool plugin. Should produce:

- `pyproject.toml` with corvidae as a dependency
- Plugin module with a class implementing `register_tools` hookimpl
- At least one tool function skeleton
- A test file with fixture wiring
- Entry point registration (if we adopt setuptools entry points for
  plugin discovery) or instructions for manual registration

Goal: lower the barrier to writing a tool plugin to "run one command,
fill in the function body." See `docs/plugin-guide.md` for the patterns
the scaffold should follow.

## Defend against blocking tool functions

All tools run on the main asyncio event loop in a single thread. A tool
that performs synchronous blocking I/O (e.g., `requests.get`,
`time.sleep`, `open().read()` on a slow path) stalls the entire process
— main agent, all task queue workers, all channel queues.

Existing built-in tools are async-safe (file tools use
`asyncio.to_thread`, shell uses `create_subprocess_shell`, web uses
`aiohttp`). The risk is third-party tools registered via
`register_tools` — there's no enforcement or defense.

Three changes:

1. **Hard-fail at registration**: `Tool.from_function()` should raise if
   the function is not a coroutine function
   (`asyncio.iscoroutinefunction`). This is an enforced invariant —
   tools must be async. This won't catch every case (an async function
   can still call blocking code internally), but it puts up the
   appropriate guardrail at the API boundary.

2. **Defensive wrap as fallback**: For tools registered directly as
   `Tool` instances (bypassing `from_function`), `execute_tool_call`
   should detect sync callables and wrap them in
   `asyncio.to_thread()`. This is the safety net, not the normal path.
   Log a warning when it fires.

3. **Documentation**: `docs/plugin-guide.md` should call out the async
   requirement explicitly, with examples of `asyncio.to_thread()` for
   wrapping blocking library calls (the file tools are the model).

Minor related finding: `JsonlLogPlugin` does synchronous `fh.write()`
and `fh.flush()` inside async hook handlers. Impact is low (~1-5ms per
write) but should be moved to `asyncio.to_thread()` for consistency.

## Move run_agent_loop into SubagentPlugin

`run_agent_loop` in `agent_loop.py` is only called by
`SubagentPlugin`. `run_agent_turn` (single LLM call, no tool execution)
is the shared primitive that `AgentPlugin` uses. The two functions are
in the same module for historical reasons — `run_agent_turn` was
extracted from `run_agent_loop` — but the dependency now points the
wrong way.

Proposed:
- Move `run_agent_loop` into `corvidae/tools/subagent.py`.
- `agent_loop.py` retains only `run_agent_turn`, `AgentTurnResult`,
  `strip_thinking`, and `strip_reasoning_content`. Could be collapsed
  into `agent.py` at that point, or renamed to `llm_turn.py` to reflect
  what it actually contains.

This also opens the door to wrapping the subagent's `run_agent_loop`
call in `asyncio.to_thread()` for thread-level isolation from the main
event loop, without affecting the shared `run_agent_turn` code path.

Full process isolation (subprocess) would give stronger guarantees —
own event loop, resource limits, kill-on-timeout — but requires solving
tool serialization across the process boundary. Tools are in-memory
callables that can't be pickled; the subprocess would need to either
re-register tools (duplicating startup) or proxy tool calls back to the
parent via IPC. Worth revisiting if subagents become untrusted or
resource-constrained, but `to_thread` is the 80% solution for now.
