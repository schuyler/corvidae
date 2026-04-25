# Architecture Critique

## The big ones

### 1. The agent loop is a distributed state machine

A single user→response cycle touches `AgentPlugin._process_queue_item` →
`_dispatch_tool_calls` → `TaskQueue.run_worker` →
`TaskPlugin._on_task_complete` → `on_notify` → back to `AgentPlugin`. No
single place in the code shows the whole flow. This is the hardest thing
to reason about in the codebase, and it's the most important thing to
reason about.

**Assessed and documented.** `_process_queue_item` is the center — it owns
all decisions (LLM calls, tool dispatch vs. response, turn counting,
compaction). The "distributed" quality comes from tool execution: results
re-enter via `on_notify` → serial queue → `_process_queue_item` rather
than returning inline. This is intentional — it keeps the channel's serial
queue unblocked during tool execution, so user messages can interleave
mid-cycle. Collapsing it into a literal loop (as `run_agent_loop` does for
subagents) would mean either blocking the queue or giving up interleaving.
The tradeoff is documented in `_process_queue_item`'s docstring.

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

~~`compact_if_needed` updates the in-memory message list but never writes
back to the DB. After restart, `load()` replays the full uncompacted
history. Compaction only works within a single session.~~

**Addressed.** `compact_if_needed` now persists a summary row with
`is_summary = 1` to the DB via `_persist_summary()`. On `load()`, the
newest summary is found and only non-summary rows after it are loaded.
Old messages remain untouched in the DB (additive-only writes). Schema
migration handled via `ALTER TABLE` with `OperationalError` catch in
`init_db()`.

### 6. `send_message` is an unusual hook

~~Pluggy hooks broadcast to all listeners. `send_message` is meant to
dispatch to a single transport, but if both CLI and IRC are registered,
both get called. Routing is implicit — each transport filters
internally. This works until it doesn't.~~

**Addressed (documented).** The broadcast-filter pattern is documented in
the `send_message` hookspec docstring in `hooks.py`, and inline comments
in `channels/cli.py` and `channels/irc.py` explain the transport check.
The pattern is retained as-is — restructuring overhead exceeds benefit
given the transport count will remain small.

### 9. `compact_if_needed` hardcodes 20-message retention

~~The boundary between "summarize" and "keep" is a magic number.
Parameterizing it in config would be slightly better, but the right
approach is to choose the compaction boundary by token count or at least
by text size (KB), since message sizes vary wildly and a fixed count is
meaningless as a proxy for context consumption.~~

**Addressed:** replaced hardcoded 20-message split with token-budget backward walk. Retain budget = 50% of max_context_tokens. Guard changed from >20 to >5. Also fixed token_estimate() to handle None/non-string content.

### 7. `on_task_complete` is dead surface area

~~Nothing implements it. `on_notify` already handles the same event path.
Two hooks fire after every task completion for the price of one.~~

**Addressed.** `on_task_complete` hookspec removed from `AgentSpec` in
`hooks.py`. The `pm.ahook.on_task_complete(...)` broadcast call removed
from `TaskPlugin._on_task_complete` in `task.py`. The private method
`_on_task_complete` is retained — it delivers results via `on_notify`.
Associated hookspec test and mock stubs cleaned up.

### 8. `resolve_system_prompt` is misplaced

~~It lives in `conversation.py` with no dependency on `ConversationLog` or
the DB. It belongs in `channel.py` or a standalone module.~~

**Addressed.** `resolve_system_prompt` moved to `channel.py`. Imports
updated in `agent.py`, `tests/test_prompt.py`, and `tests/test_logging.py`.

## Minor but worth noting

- ~~**`QueueItem.role` is stringly typed** — `"user"` / `"notification"`
  as a string discriminator, with branching in `_process_queue_item`. An
  enum would be cleaner.~~ **Fixed:** `QueueItemRole` enum added to `agent.py`.
- ~~**`completed` dict in `TaskQueue` grows unboundedly** — no eviction.~~
  **Fixed:** replaced with `collections.deque(maxlen=100)` of `(task_id, result)` tuples.
- **`split_message` in `irc.py`** is the densest function in the codebase
  (~108 lines, three-tier recursive splitting) with no visible tests.
  **Not a concern:** 27 tests exist in `test_irc_plugin.py`.
- ~~**`CLIPlugin.send_message` accepts `latency_ms`;
  `IRCPlugin.send_message` does not** — silent interface divergence.~~
  **Fixed:** `IRCPlugin.send_message` now accepts `latency_ms: float | None = None`.
- ~~**`token_estimate` may crash on tool messages** —
  `len(msg.get("content", ""))` on `None` content raises `TypeError`.~~
  **Fixed:** addressed in critique #9 (token_estimate handles None/non-string content).
- **Backward-compat path for bare callables in `register_tools`** — not
  dead code. `test_agent_loop_plugin::test_on_start_collects_tools` appends
  a bare function, and third-party plugins could do the same. Retained.
- ~~**`Channel.turn_counter`** is state logically owned by the agent loop,
  not the channel abstraction.~~ **Documented:** extended comment on `turn_counter`
  in `channel.py` explains why it lives on `Channel` (re-entrant loop design).
- ~~**`run_agent_loop` exception error message is generic** —
  `agent_loop.py:199` uses `"Error: unknown error"` on tool exceptions,
  omitting the tool name. The adjacent unknown-tool path includes
  `fn_name`. Also a redundant f-string with no interpolation.~~
  **Fixed:** error message now reads `f"Error: tool '{fn_name}' raised an exception"`.

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
4. ~~**Make compaction durable** — either write a compaction marker to the
   DB, or accept that compaction is session-scoped and document it.~~ Done
   (summary rows with `is_summary = 1`, summary-aware `load()`).
5. ~~**Move `resolve_system_prompt` out of `conversation.py`**.~~ Done (moved to `channel.py`).
