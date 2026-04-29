# Plan: AgentPlugin Decomposition

**Date:** 2026-04-28
**Status:** Design (pending approval)

## Overview

AgentPlugin is being decomposed from a god object into focused components
with documented public interfaces. The orchestration loop stays as one piece
(renamed to `Agent`); lifecycle concerns are extracted into independent plugins.

Five parts, executed sequentially:

1. Clean up external coupling interfaces (precursor)
2. Invert idle detection (decouple IdleMonitorPlugin from Agent internals)
3. Extract LLM client lifecycle into LLMPlugin
4. Extract tool collection into ToolCollectionPlugin
5. Rename AgentPlugin → Agent, registered name "agent_loop" → "agent"

## Session Breakdown

Estimated 2–3 sessions:
- **Session 1:** Parts 1–2 (small scope, low risk)
- **Session 2:** Parts 3–4 (new plugins, hookspec changes, heaviest work)
- **Session 3:** Part 5 (mechanical rename, ~20 files touched)

Parts 3 and 4 may fit in one session if things go smoothly, or split
across two if the `compact_conversation` hookspec change or
`register_tools` ordering requires iteration.

## Execution Order

```
Part 1 → Part 2 → Part 3 → Part 4 → Part 5
```

Sequential. Each part builds on the previous:
- Part 1 establishes the interface pattern that Parts 3–4 use
- Part 2 removes IdleMonitor's coupling to `.queues` (which simplifies Part 5's rename)
- Parts 3 and 4 extract the pieces (order matters: LLM first because
  ToolCollectionPlugin has no LLM dependency, but changing the
  compact_conversation hookspec in Part 3 should happen before Part 4
  touches the same startup code)
- Part 5 renames — must be last because it touches every `depends_on` and
  `get_dependency` reference

Each part is a separate Standard-tier task with its own red/green/review cycle.

---

## Part 1: Clean Up External Coupling Interfaces

### Summary

SubagentPlugin reaches into AgentPlugin internals (`._max_tool_result_chars`
and `.tool_registry`). Replace with a public method. This establishes the
pattern for how extracted plugins will expose their data.

### Changes

**`corvidae/agent.py`:**
- Initialize `self.tool_registry = None` in `__init__` (currently only
  assigned dynamically in `_start_plugin` at line 562 — not declared in
  `__init__`, which confuses type checkers and risks AttributeError if
  called before startup)
- Add public method:
  ```python
  def get_tool_config(self) -> tuple[ToolRegistry, int]:
      """Return (tool_registry, max_tool_result_chars) for subagent use.

      Only valid after on_start has completed.
      """
      return self.tool_registry, self._max_tool_result_chars
  ```
- Document `.tool_registry` as public in the class docstring
- Note: `get_tool_config()` is a short-lived interface — Part 4 supersedes
  it with `ToolCollectionPlugin`. It exists to establish the pattern and
  decouple SubagentPlugin from private attributes in the interim.

**`corvidae/tools/subagent.py`:**
- Change from:
  ```python
  agent = get_dependency(self.pm, "agent_loop", AgentPlugin)
  max_result_chars = agent._max_tool_result_chars
  registry = agent.tool_registry.exclude("subagent")
  ```
- To:
  ```python
  agent = get_dependency(self.pm, "agent_loop", AgentPlugin)
  tool_registry, max_result_chars = agent.get_tool_config()
  registry = tool_registry.exclude("subagent")
  ```

### Interface Changes

| Type | Name | Change |
|------|------|--------|
| Add | `AgentPlugin.get_tool_config()` | New public method |
| Remove | Direct access to `._max_tool_result_chars` | SubagentPlugin stops reaching in |

### Migration Path

- SubagentPlugin: update access pattern (shown above)
- Tests: any test that sets `agent._max_tool_result_chars` directly can continue — the method just wraps the attribute

### Risk Assessment

Very low. Single method addition, one caller update. No behavioral change.

### Dependencies

None — this is the first step.

---

## Part 2: Invert Idle Detection

### Summary

IdleMonitorPlugin currently polls `AgentPlugin.queues` every 2 seconds to detect
idle state. This is backwards. Agent owns the queues, knows when items complete,
and should push the `on_idle` signal. IdleMonitorPlugin becomes a pure consumer.

### Changes

**`corvidae/agent.py`:**
- Add idle throttle state:
  ```python
  self._idle_cooldown: float = 30.0  # from config
  self._last_idle_fire: float = 0.0
  ```
- After `_process_queue_item` completes (in the queue consumer callback), check:
  ```python
  async def _maybe_fire_idle(self):
      """Fire on_idle if all queues are empty and cooldown has elapsed."""
      for q in self.queues.values():
          if not q.is_empty:
              return
      # Check task queue
      task_plugin = self.pm.get_plugin("task")
      if task_plugin is not None:
          tq = getattr(task_plugin, "task_queue", None)
          if tq is not None and not tq.is_idle:
              return
      if time.monotonic() - self._last_idle_fire < self._idle_cooldown:
          return
      self._last_idle_fire = time.monotonic()
      try:
          await self.pm.ahook.on_idle()
      except Exception:
          logger.warning("on_idle hook raised exception", exc_info=True)
  ```
- Read `daemon.idle_cooldown_seconds` from config in `_start_plugin`
- Call `await self._maybe_fire_idle()` at the end of `_process_queue_item`

**`corvidae/idle.py`:**
- Remove `IdleMonitor` class entirely (the polling loop)
- `IdleMonitorPlugin` simplifies to:
  ```python
  class IdleMonitorPlugin:
      """Pure consumer of the on_idle hook. Implements idle behaviors."""
      depends_on = set()  # No longer depends on agent_loop or task

      def __init__(self, pm):
          self.pm = pm

      @hookimpl
      async def on_idle(self):
          # Whatever idle behaviors this plugin performs
          pass
  ```
- If IdleMonitorPlugin has no actual idle *behavior* (it was just the detector),
  it can be removed entirely and the `on_idle` hook becomes a pure extension point.

**`corvidae/main.py`:**
- Remove IdleMonitorPlugin registration if it becomes empty
- Or keep it if it has actual idle behaviors to perform

### Interface Changes

| Type | Name | Change |
|------|------|--------|
| Remove | `AgentPlugin.queues` external access | IdleMonitor no longer reads it |
| Remove | `IdleMonitor` class | Polling loop eliminated |
| Add | `Agent._maybe_fire_idle()` | Private method, push-based |
| Change | `IdleMonitorPlugin.depends_on` | `{"agent_loop", "task"}` → `set()` |
| Add | Config `daemon.idle_cooldown_seconds` | Replaces IdleMonitor's constructor params |

### Migration Path

- IdleMonitorPlugin: rewrite (much simpler)
- Tests: idle detection tests rewrite to verify push behavior
- `.queues` remains public (tests still drain it) but no production code reads it externally

### Risk Assessment

Medium. Behavioral change — idle detection timing will differ slightly from
polling. The throttle ensures on_idle doesn't fire too often, but the trigger
point changes from "poll detected empty" to "last item completed." Test
carefully that idle fires reliably after all work drains.

### Dependencies

Part 1 (not strictly required, but keeps the work sequential and clean).

---

## Part 3: Extract LLM Client Lifecycle into LLMPlugin

### Summary

LLM client creation, config parsing, and session lifecycle moves from
AgentPlugin into a new `LLMPlugin`. Other plugins access clients via
`get_dependency("llm")`. This also consolidates the duplicate config
parsing in SubagentPlugin.

### Changes

**New file `corvidae/llm_plugin.py`:**
```python
"""LLMPlugin — owns LLM client lifecycle and configuration.

Parses llm.main and llm.background config, creates LLMClient instances,
manages their aiohttp session lifecycle. Other plugins access clients
via get_dependency("llm").

Config:
    llm:
      main:           # required
        base_url: ...
        model: ...
        api_key: ...        # optional
        extra_body: ...     # optional
        max_retries: 3      # optional
        retry_base_delay: 2.0
        retry_max_delay: 60.0
        timeout: ...        # optional
      background:     # optional — absent means use llm.main
        (same keys)
"""
import logging
from corvidae.hooks import hookimpl
from corvidae.llm import LLMClient

logger = logging.getLogger(__name__)


class LLMPlugin:
    """Plugin that owns LLM client instances and their lifecycle."""

    depends_on = set()

    def __init__(self, pm):
        self.pm = pm
        self.main_client: LLMClient | None = None
        self.background_client: LLMClient | None = None

    @hookimpl
    async def on_start(self, config: dict) -> None:
        llm_config = config.get("llm", {})
        main_config = llm_config["main"]  # required
        self.main_client = self._create_client(main_config)
        await self.main_client.start()

        bg_config = llm_config.get("background")
        if bg_config:
            self.background_client = self._create_client(bg_config)
            await self.background_client.start()

    @hookimpl
    async def on_stop(self) -> None:
        if self.main_client:
            await self.main_client.stop()
        if self.background_client:
            await self.background_client.stop()

    def get_client(self, role: str = "main") -> LLMClient:
        """Return the client for the given role.

        Args:
            role: "main" or "background". Background falls back to main
                  if no background client is configured.
        """
        if role == "background":
            return self.background_client or self.main_client
        return self.main_client

    @staticmethod
    def _create_client(cfg: dict) -> LLMClient:
        return LLMClient(
            base_url=cfg["base_url"],
            model=cfg["model"],
            api_key=cfg.get("api_key"),
            extra_body=cfg.get("extra_body"),
            max_retries=cfg.get("max_retries", 3),
            retry_base_delay=cfg.get("retry_base_delay", 2.0),
            retry_max_delay=cfg.get("retry_max_delay", 60.0),
            timeout=cfg.get("timeout"),
        )
```

**`corvidae/agent.py`:**
- Remove LLM config parsing from `_start_plugin`
- Remove `self.client` attribute
- Add `depends_on = {"registry", "task", "llm"}`
- In methods that need the client:
  ```python
  llm = get_dependency(self.pm, "llm", LLMPlugin)
  client = llm.main_client
  ```
  Or cache in `_start_plugin`: `self._client = get_dependency(...).main_client`

**`corvidae/tools/subagent.py`:**
- Remove LLM client creation from `on_start` (currently creates a fresh
  `LLMClient` per subagent invocation inside the work closure)
- Use `get_dependency(self.pm, "llm", LLMPlugin).get_client("background")`
- Change `depends_on` to include `"llm"`
- **Behavioral change:** SubagentPlugin currently creates an ephemeral
  LLMClient per task. After this change, all subagents share
  `LLMPlugin.background_client`. This is safe: `LLMClient` holds an
  aiohttp `ClientSession` which is concurrent-safe and provides HTTP
  connection pooling. Sharing is strictly better — avoids per-task TCP
  setup/teardown overhead. The per-task pattern was accidental (SubagentPlugin
  had no access to a shared client, so it made its own).

**`corvidae/hooks.py`:**
- Change `compact_conversation` hookspec:
  ```python
  # Before:
  async def compact_conversation(self, channel, conversation, client, max_tokens):
  # After:
  async def compact_conversation(self, channel, conversation, max_tokens):
  ```

**`corvidae/compaction.py`** (or wherever CompactionPlugin lives):
- Add `depends_on = {"llm"}`
- Get client from LLMPlugin instead of hook parameter

**`corvidae/main.py`:**
- Register LLMPlugin before AgentPlugin
- Remove LLM-related config validation from AgentPlugin registration

### Interface Changes

| Type | Name | Change |
|------|------|--------|
| Add | `LLMPlugin` class | New plugin |
| Add | `LLMPlugin.get_client(role)` | Public method |
| Remove | `AgentPlugin.client` | No longer owned here |
| Change | `compact_conversation` hookspec | Drop `client` parameter |
| Change | `AgentPlugin.depends_on` | Add `"llm"` |
| Change | `SubagentPlugin.depends_on` | Add `"llm"` |
| Change | `CompactionPlugin.depends_on` | Add `"llm"` |

### Migration Path

- CompactionPlugin: update hookimpl signature, get client from LLMPlugin
- Tests that mock `agent.client`: mock via LLMPlugin instead, or inject directly
- The `compact_conversation` hookspec change is breaking — all implementations
  must update simultaneously

### Risk Assessment

Medium. The `compact_conversation` hookspec change is the riskiest part — it's
a breaking interface change. All implementations must be updated atomically.
Grep for all `compact_conversation` implementations and update them in the same
commit.

The LLM client behavior is unchanged — same config, same class, same session
management. Only the ownership moves.

### Dependencies

Parts 1 and 2 (sequential execution).

---

## Part 4: Extract Tool Collection into ToolCollectionPlugin

### Summary

Tool registration (calling the `register_tools` hook, building ToolRegistry)
moves from AgentPlugin into a new `ToolCollectionPlugin`. The result is
immutable after startup. Any plugin needing tools declares a dependency
on `"tools"`.

### Changes

**New file `corvidae/tool_collection.py`:**
```python
"""ToolCollectionPlugin — collects tools from all plugins at startup.

Calls the register_tools hook during on_start (broadcast phase), builds
a ToolRegistry, and exposes it as an immutable collection. Plugins that
need tool access declare depends_on = {"tools"}.

Config:
    tools:
      max_result_chars: 100000  # optional, default 100_000
"""
import logging
from corvidae.hooks import hookimpl
from corvidae.tool import Tool, ToolRegistry

logger = logging.getLogger(__name__)


class ToolCollectionPlugin:
    """Plugin that collects and owns the tool registry."""

    depends_on = set()

    def __init__(self, pm):
        self.pm = pm
        self.registry: ToolRegistry | None = None
        self.max_result_chars: int = 100_000

    @hookimpl(trylast=True)
    async def on_start(self, config: dict) -> None:
        tools_config = config.get("tools", config.get("agent", {}))
        self.max_result_chars = tools_config.get("max_tool_result_chars", 100_000)

        # Collect tools from all plugins
        collected: list = []
        self.pm.hook.register_tools(tool_registry=collected)

        # Build registry
        tool_registry = ToolRegistry()
        for item in collected:
            if isinstance(item, Tool):
                tool_registry.add(item)
            else:
                tool_registry.add(Tool.from_function(item))

        self.registry = tool_registry
        logger.info("Tools collected: %d", len(tool_registry))

    def get_tools(self) -> tuple[dict, list[dict]]:
        """Return (tools_dict, tool_schemas) for the agent loop."""
        return self.registry.as_dict(), self.registry.schemas()

    def get_registry(self) -> ToolRegistry:
        """Return the full ToolRegistry for inspection/filtering."""
        return self.registry
```

**`corvidae/agent.py`:**
- Remove `register_tools` call from `_start_plugin`
- Remove `self.tools`, `self.tool_schemas`, `self.tool_registry` attributes
- Add `"tools"` to `depends_on`
- In `_start_plugin` or lazily:
  ```python
  tools_plugin = get_dependency(self.pm, "tools", ToolCollectionPlugin)
  self._tools, self._tool_schemas = tools_plugin.get_tools()
  self._max_tool_result_chars = tools_plugin.max_result_chars
  ```
- Remove `get_tool_config()` method (added in Part 1) — SubagentPlugin now
  goes directly to ToolCollectionPlugin

**`corvidae/tools/subagent.py`:**
- Change dependency from `"agent_loop"` to `"tools"` for tool access:
  ```python
  tools_plugin = get_dependency(self.pm, "tools", ToolCollectionPlugin)
  registry = tools_plugin.get_registry().exclude("subagent")
  max_result_chars = tools_plugin.max_result_chars
  ```
- Keep `"llm"` dependency for LLM client access
- May still need `"agent_loop"` if it accesses other Agent state (verify)

**`corvidae/main.py`:**
- Register ToolCollectionPlugin before AgentPlugin (in broadcast phase)
- Ensure tool-providing plugins (CoreToolsPlugin, McpClientPlugin,
  SubagentPlugin, RuntimeSettingsPlugin, TaskPlugin) are registered before
  ToolCollectionPlugin — their `register_tools` hookimpls must be available
  when ToolCollectionPlugin's `on_start` fires

### Interface Changes

| Type | Name | Change |
|------|------|--------|
| Add | `ToolCollectionPlugin` class | New plugin |
| Add | `ToolCollectionPlugin.get_tools()` | Public method |
| Add | `ToolCollectionPlugin.get_registry()` | Public method |
| Add | `ToolCollectionPlugin.max_result_chars` | Public attribute |
| Remove | `AgentPlugin.tools` | Moved to ToolCollectionPlugin |
| Remove | `AgentPlugin.tool_schemas` | Moved to ToolCollectionPlugin |
| Remove | `AgentPlugin.tool_registry` | Moved to ToolCollectionPlugin |
| Remove | `AgentPlugin.get_tool_config()` | Superseded by ToolCollectionPlugin |
| Change | `AgentPlugin.depends_on` | Add `"tools"` |
| Change | `SubagentPlugin.depends_on` | `"agent_loop"` → `"tools"`, `"llm"` |

### Migration Path

- SubagentPlugin: depends on `"tools"` and `"llm"` instead of `"agent_loop"`
- Agent: reads tools from ToolCollectionPlugin at startup, stores locally for fast access
- Config: `agent.max_tool_result_chars` moves to `tools.max_result_chars`
  (support both during transition, prefer new key)
- Tests: update tool injection to go through ToolCollectionPlugin

### Risk Assessment

Medium. The `register_tools` hook is synchronous and fires during broadcast
`on_start`. Registration order in main.py must ensure tool-providing plugins
are registered before ToolCollectionPlugin. Currently all tool providers are
registered before AgentPlugin; ToolCollectionPlugin goes in the same slot.

The key risk: `register_tools` is currently called by AgentPlugin in its
explicit `on_start` (after broadcast). Moving it to ToolCollectionPlugin's
broadcast `on_start` means it fires earlier. All tool providers must have
their tools ready when their `register_tools` hookimpl is called — verify
that none depend on AgentPlugin being initialized first.

### Dependencies

Parts 1, 2, 3.

---

## Part 5: Rename AgentPlugin → Agent

### Summary

AgentPlugin isn't a plugin — its lifecycle is managed explicitly by main.py.
It's the framework core. Rename to `Agent`, change registered name from
`"agent_loop"` to `"agent"`.

### Changes

**`corvidae/agent.py`:**
- Rename class `AgentPlugin` → `Agent`
- Update module docstring

**`corvidae/main.py`:**
- `agent = Agent(pm)`
- `pm.register(agent, name="agent")`
- Update explicit `on_start`/`on_stop` calls

**All files with `depends_on` referencing `"agent_loop"`:**
- After Parts 2–4, IdleMonitorPlugin and SubagentPlugin may no longer depend
  on Agent at all. Verify what remains.
- Any remaining `depends_on = {"agent_loop"}` → `depends_on = {"agent"}`

**All files with `get_dependency(..., "agent_loop", ...)`:**
- Update to `get_dependency(..., "agent", Agent)`

**All test files:**
- `from corvidae.agent import AgentPlugin` → `from corvidae.agent import Agent`
- `name="agent_loop"` → `name="agent"`
- Mock references

**`corvidae/hooks.py`:**
- If `validate_dependencies` or any utility references `"agent_loop"` by name

### Interface Changes

| Type | Name | Change |
|------|------|--------|
| Rename | `AgentPlugin` → `Agent` | Class name |
| Rename | `"agent_loop"` → `"agent"` | Registered plugin name |

### Migration Path

This is a sweeping rename. Use `replace_all` in the editor or a sed pass.
Grep for both the class name and the registered name string. Expect ~18 test
files to need updates.

### Risk Assessment

Low technical risk (pure rename), high churn. Every file that references
AgentPlugin or "agent_loop" needs updating. The change is mechanical but
the blast radius is large. Do it last when all other parts have stabilized
the dependency graph.

No backward compatibility shim — this is an internal project, not a public API.

### Dependencies

Parts 1–4 (must be last).

---

## Cross-Cutting Concerns

### The `compact_conversation` Hookspec Change (Part 3)

This is the only breaking hookspec change in the plan. Current signature:

```python
@hookspec
async def compact_conversation(self, channel, conversation, client, max_tokens):
    """Compact conversation history when approaching token limits."""
```

New signature (Part 3):

```python
@hookspec
async def compact_conversation(self, channel, conversation, max_tokens):
    """Compact conversation history when approaching token limits."""
```

**All implementations must update atomically.** Grep results to update:
- `corvidae/compaction.py` — CompactionPlugin (production)
- Test mocks/fixtures

CompactionPlugin will `get_dependency("llm")` in its `on_start` and cache
the client reference.

### Future: `compact_conversation` as a Strategy Hook

The Part 3 change to `compact_conversation` (dropping `client`) is a good
time to consider the hook's dispatch model. Currently it's a broadcast hook
— all implementations fire. This works with one CompactionPlugin but breaks
down when multiple context window strategies exist (e.g., summarization,
sliding window, RAG-backed retrieval, hybrid).

A strategy pattern wants exactly one implementation active per channel,
selected by configuration. Two options:

**Option A: Convention-based.** Keep the broadcast hookspec. Each
implementation checks `channel.config["compaction_strategy"]` and returns
early if it's not the selected one. Simple, no framework changes, but
every implementation must remember the guard.

**Option B: Routed dispatch.** Add a strategy selection layer — either a
`compaction_strategy` config key that the caller uses to invoke a specific
named plugin, or a new hookspec pattern that filters implementations by
name. More framework work, but the intent is explicit.

Not in scope for Part 3 — the current change (drop `client`, one
CompactionPlugin) is sufficient. But the hookspec signature chosen in
Part 3 should not preclude either option. Flagged here so the Part 3
implementer doesn't paint us into a corner.

### The `register_tools` Ordering Concern (Part 4)

Currently:
1. All plugins registered (including tool providers)
2. Broadcast `on_start` fires (tool providers initialize)
3. AgentPlugin explicit `on_start` fires (calls `register_tools`)

After Part 4:
1. All plugins registered (including tool providers and ToolCollectionPlugin)
2. Broadcast `on_start` fires — ToolCollectionPlugin calls `register_tools`
3. Agent explicit `on_start` fires (reads from ToolCollectionPlugin)

The concern: do any tool providers' `register_tools` hookimpls depend on
state set in their own `on_start`? If so, the broadcast ordering matters.

Check each `register_tools` implementation:
- CoreToolsPlugin: registers functions defined at module level — no startup dependency
- McpClientPlugin: may need MCP connections established in `on_start` first
- SubagentPlugin: registers a closure — no startup dependency
- RuntimeSettingsPlugin: registers a closure — no startup dependency
- TaskPlugin: registers a closure — no startup dependency

**McpClientPlugin is the risk.** If it connects to MCP servers in `on_start`
and uses those connections in `register_tools`, then ToolCollectionPlugin's
`on_start` must run *after* McpClientPlugin's `on_start`.

**Solution:** Use `@hookimpl(trylast=True)` on `ToolCollectionPlugin.on_start`.
pluggy guarantees `trylast` implementations run after all normal ones,
regardless of registration order. This is self-documenting and doesn't
break if `main.py` registration order changes. (Same pattern IdleMonitorPlugin
already uses for the same reason.)

Verify McpClientPlugin's implementation before executing Part 4.

### Registered Name Change (Part 5)

`"agent_loop"` → `"agent"` affects:
- `depends_on` sets in other plugins
- `get_dependency()` calls
- `pm.get_plugin("agent_loop")` calls
- Test registration code

No backward compat shim needed — this is an internal project.

### Config Key Migration (Part 4)

`agent.max_tool_result_chars` → `tools.max_result_chars`

Support both during transition:
```python
tools_config = config.get("tools", {})
fallback = config.get("agent", {}).get("max_tool_result_chars")
self.max_result_chars = tools_config.get(
    "max_result_chars",
    fallback if fallback is not None else 100_000,
)
```

Deprecation warning if the old key is used. Remove old key support in a
future release.

### `_chars_per_token` Stays in Agent

`AgentPlugin._chars_per_token` (from `agent.chars_per_token` config) is used
only for `ContextWindow` construction, which remains in the orchestration loop.
It stays as an Agent-owned config value. The `agent:` config section partially
survives — it retains `chars_per_token` while `max_tool_result_chars` moves to
`tools:`.

---

## Summary Table

| Part | Scope | Risk | Files Modified | New Files |
|------|-------|------|---------------|-----------|
| 1 | Small | Very low | 2 | 0 |
| 2 | Medium | Medium | 3–4 | 0 |
| 3 | Large | Medium | 5–6 | 1 |
| 4 | Large | Medium | 4–5 | 1 |
| 5 | Sweeping | Low (mechanical) | ~20 | 0 |

Estimated total: 5 Standard-tier tasks, each with its own red/green/review cycle.

---

## Part 1 Red Phase Report

**Status:** PASS
**Tests added:**
- `TestGetToolConfig.test_tool_registry_attribute_exists_after_init`
- `TestGetToolConfig.test_get_tool_config_method_exists`
- `TestGetToolConfig.test_get_tool_config_returns_tuple`
- `TestGetToolConfig.test_get_tool_config_returns_tool_registry`
- `TestGetToolConfig.test_get_tool_config_returns_max_tool_result_chars`
- `TestGetToolConfig.test_get_tool_config_reflects_updated_max_chars`

**Failure mode:** 1 test fails with `AssertionError` (`tool_registry` not initialised in `__init__`); 5 tests fail with `AttributeError` (`'AgentPlugin' object has no attribute 'get_tool_config'`). All 6 fail as expected.
**Proceed:** yes

---

## Part 3 Completion Report

**Status:** COMPLETE — all tests pass
**Baseline test count:** 746

**What was done:**
- Created `corvidae/llm_plugin.py` with `LLMPlugin` class
  - Owns `main_client` and `background_client` (`LLMClient` instances)
  - Parses `llm.main` (required) and `llm.background` (optional) config
  - Manages aiohttp session lifecycle (`on_start` / `on_stop`)
  - `get_client(role)` returns background client with fallback to main
- Removed LLM config parsing and client creation from `AgentPlugin._start_plugin`
- `AgentPlugin` now borrows `main_client` from `LLMPlugin` at startup via `get_dependency`
- `AgentPlugin.depends_on` updated: added `"llm"`
- `compact_conversation` hookspec updated: removed `client` parameter
  - `CompactionPlugin` updated to get client from `LLMPlugin` in its `on_start`
  - `CompactionPlugin.depends_on` updated: added `"llm"`
- `SubagentPlugin` updated: removed per-task ephemeral `LLMClient` creation
  - Uses `LLMPlugin.get_client("background")` for the shared background client
  - `SubagentPlugin.depends_on` updated: added `"llm"`

**Tests added:** tests covering `LLMPlugin` startup, client access, `on_stop` cleanup,
`compact_conversation` hookspec without `client`, and `SubagentPlugin` client sharing.

**Proceed:** yes

---

## Part 4 Completion Report

**Status:** COMPLETE — all tests pass
**Final test count:** 775 (baseline was 746, net +29)

**What was done:**
- Created `corvidae/tool_collection.py` with `ToolCollectionPlugin` class
  - Uses `@hookimpl(trylast=True)` on `on_start` to run after all tool providers
  - Calls `register_tools` broadcast, builds `ToolRegistry`
  - Exposes `get_tools()` → `(tools_dict, tool_schemas)` and `get_registry()` → `ToolRegistry`
  - Reads `tools.max_result_chars` config (with fallback to `agent.max_tool_result_chars`)
- `AgentPlugin`:
  - Removed `register_tools` call and direct tool attributes (`self.tools`, `self.tool_schemas`, `self.tool_registry`)
  - Now reads `_tools`, `_tool_schemas`, `_max_tool_result_chars` from `ToolCollectionPlugin` at startup
  - `get_tool_config()` method removed (superseded by `ToolCollectionPlugin`)
  - `depends_on` updated: added `"tools"`
- `SubagentPlugin`:
  - Removed dependency on `"agent_loop"` for tool access
  - Uses `ToolCollectionPlugin.get_registry().exclude("subagent")` and `tools_plugin.max_result_chars`
  - `depends_on` is now `{"llm", "tools"}` (no longer includes `"agent_loop"`)

**Config key migration:** `agent.max_tool_result_chars` is superseded by `tools.max_result_chars`.
Both keys are read during transition; `tools.max_result_chars` takes precedence.
