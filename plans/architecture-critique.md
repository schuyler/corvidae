# Architecture Critique

## The big ones

### 1. The agent loop is a distributed state machine

A single user→response cycle touches `AgentPlugin._process_queue_item` →
`_dispatch_tool_calls` → `TaskQueue.run_worker` →
`TaskPlugin._on_task_complete` → `on_notify` → back to `AgentPlugin`. No
single place in the code shows the whole flow. This is the hardest thing
to reason about in the codebase, and it's the most important thing to
reason about.

### 2. Tool dispatch is implemented twice

`_ctx` injection and tool execution exist in both
`AgentPlugin._dispatch_tool_calls` (main agent, task-based) and
`run_agent_loop` (subagent, inline). Any change to how tools receive
context must be made in two places. The two paths also have different
concurrency characteristics: the main agent's tools run as separate
tasks, while a subagent's tools block the task worker for the entire
loop duration.

### 3. `run_agent_turn` is duplicated inside `run_agent_loop`

The inner loop of `run_agent_loop` repeats the same LLM-call → parse →
log → append code that `run_agent_turn` already does. `run_agent_loop`
should call `run_agent_turn` per iteration and add the tool dispatch
layer on top.

### 4. The plugin manager is a service locator

~~`pm.registry`, `pm.task_plugin`, `pm.agent_plugin` are monkey-patched
onto the PM at runtime. This makes initialization order load-bearing
(enforced only by code comments in `main.py`) and isn't type-safe. It's
the pattern pluggy is supposed to prevent.~~

**Addressed.** Plugins now declare `depends_on` class attributes (sets of
plugin name strings) and retrieve dependencies via `get_dependency(pm,
name, expected_type)` from `hooks.py`, which raises typed errors on
missing or mistyped plugins. `validate_dependencies(pm)` runs at startup
in `main.py` after all registrations, verifying the full dependency graph
before `on_start` fires. `ChannelRegistry` is registered as a named
plugin (`"registry"`) on the PM rather than monkey-patched.

## Medium concerns

### 5. Compaction isn't durable

`compact_if_needed` updates the in-memory message list but never writes
back to the DB. After restart, `load()` replays the full uncompacted
history. Compaction only works within a single session.

### 6. `send_message` is an unusual hook

Pluggy hooks broadcast to all listeners. `send_message` is meant to
dispatch to a single transport, but if both CLI and IRC are registered,
both get called. Routing is implicit — each transport filters
internally. This works until it doesn't.

### 7. `on_task_complete` is dead surface area

Nothing implements it. `on_notify` already handles the same event path.
Two hooks fire after every task completion for the price of one.

### 8. `resolve_system_prompt` is misplaced

It lives in `conversation.py` with no dependency on `ConversationLog` or
the DB. It belongs in `channel.py` or a standalone module.

## Minor but worth noting

- **`QueueItem.role` is stringly typed** — `"user"` / `"notification"`
  as a string discriminator, with branching in `_process_queue_item`. An
  enum would be cleaner.
- **`completed` dict in `TaskQueue` grows unboundedly** — no eviction.
- **`split_message` in `irc.py`** is the densest function in the codebase
  (~108 lines, three-tier recursive splitting) with no visible tests.
- **`CLIPlugin.send_message` accepts `latency_ms`;
  `IRCPlugin.send_message` does not** — silent interface divergence.
- **`token_estimate` may crash on tool messages** —
  `len(msg.get("content", ""))` on `None` content raises `TypeError`.
- **Backward-compat path for bare callables in `register_tools`** has no
  callers — dead code.
- **`Channel.turn_counter`** is state logically owned by the agent loop,
  not the channel abstraction.
- **`run_agent_loop` exception error message is generic** —
  `agent_loop.py:199` uses `"Error: unknown error"` on tool exceptions,
  omitting the tool name. The adjacent unknown-tool path includes
  `fn_name`. Also a redundant f-string with no interpolation.

## What's good

The tool implementations (`shell.py`, `files.py`, `web.py`) are
appropriately simple. `LLMClient` is genuinely thin. `SerialQueue` does
one thing well. `ChannelConfig.resolve()` is correct and readable. The
overall layering (transport → agent → LLM) is sound. The codebase is
~2000 lines for a meaningful amount of functionality.

## Refactoring opportunities, ranked by impact

1. ~~**Unify tool dispatch** — extract the `_ctx` injection + tool
   execution into a single function in `tool.py`, called by both
   `_dispatch_tool_calls` and `run_agent_loop`.~~ Done (`execute_tool_call`
   in `tool.py`).
2. ~~**Make `run_agent_loop` call `run_agent_turn`** — eliminate the
   duplicated LLM-call code.~~ Done (same commit as #1).
3. ~~**Replace PM monkey-patching with explicit dependency injection** —
   pass `TaskQueue` and `ToolRegistry` directly rather than fishing them
   off the PM at runtime.~~ Addressed via `depends_on` / `get_dependency()`
   / `validate_dependencies()`. Uses typed PM lookup rather than direct
   injection, but eliminates the monkey-patching.
4. **Make compaction durable** — either write a compaction marker to the
   DB, or accept that compaction is session-scoped and document it.
5. **Move `resolve_system_prompt` out of `conversation.py`**.
