# Plugin Extraction: Phase B (ThinkingPlugin) and Phase C (IdleMonitorPlugin)

## Overview

Phases B and C extract two bundled behaviors from `AgentPlugin` into standalone plugins that ship with Sherman and are registered by default. Each can be unregistered without crashing the system.

---

## Phase B: ThinkingPlugin

### Goal

Extract the `<think>` block stripping logic from `agent.py` into a standalone `ThinkingPlugin` in `sherman/thinking.py`.

### New hookspecs (`sherman/hooks.py`)

**`after_persist_assistant(channel, message)`** — broadcast

Called after the assistant message has been persisted to the conversation log. Plugins may mutate the in-memory `message` dict (e.g., to strip `reasoning_content`). The DB copy is already written at call time; mutations affect only subsequent prompt builds. No return value is used.

**`transform_display_text(channel, text, result_message) -> str | None`** — firstresult

Called before sending the final text response to the channel. Return a transformed string to replace `text`, or `None` to leave it unchanged. First non-None result wins.

### New file: `sherman/thinking.py`

`ThinkingPlugin` implements two hookimpls:

**`after_persist_assistant(channel, message)`**

Reads `keep_thinking_in_history` from the resolved channel config. If false, calls `strip_reasoning_content([message])` on the in-memory message object.

Does not use `depends_on = {"registry"}`. Uses `self.pm.get_plugin("registry")` with an early return if None. This is the graceful degradation pattern: the plugin silently skips stripping if the registry is not registered, rather than failing startup.

**`transform_display_text(channel, text, result_message) -> str | None`**

Calls `strip_thinking(text)`. Returns the result if it differs from `text`, else returns `None`.

### Changes to `sherman/agent.py`

Replace lines 374–376 (strip_reasoning_content inline call):

```python
# Before
if not resolved["keep_thinking_in_history"]:
    strip_reasoning_content([conv.messages[-1]])

# After
await self.pm.ahook.after_persist_assistant(
    channel=channel, message=conv.messages[-1]
)
```

Replace lines 391 and 395 (strip_thinking inline calls):

```python
# Before — max turns branch (line 391):
display_response = strip_thinking(result.text) or "(max tool-calling rounds reached)"
# Before — normal branch (line 395):
display_response = strip_thinking(result.text)

# After — max turns branch:
transformed = await call_firstresult_hook(
    self.pm, "transform_display_text",
    channel=channel, text=result.text, result_message=result.message,
)
display_response = (transformed if transformed is not None else result.text) or "(max tool-calling rounds reached)"
# After — normal branch:
transformed = await call_firstresult_hook(
    self.pm, "transform_display_text",
    channel=channel, text=result.text, result_message=result.message,
)
display_response = transformed if transformed is not None else result.text
```

Note: Use explicit `is not None` checks, not `or` fallbacks. The `or` operator would treat an empty string return from `transform_display_text` as falsy and fall back to `result.text`, which would be wrong if a plugin intentionally returns an empty string.

Remove `strip_reasoning_content` import from `agent.py`. The `strip_thinking` import may be removed as well if no other callers remain in that file after the changes. Both functions stay in `agent_loop.py` — they are pure functions; `ThinkingPlugin` and `SubagentPlugin` import them from there. `SubagentPlugin` continues importing `strip_thinking` directly because it handles a different code path (subagent loop result, not interactive agent turn).

### Changes to `sherman/main.py`

Register `ThinkingPlugin` before `AgentPlugin`:

```python
from sherman.thinking import ThinkingPlugin
thinking_plugin = ThinkingPlugin(pm)
pm.register(thinking_plugin, name="thinking")
```

### New file: `tests/test_thinking.py`

Tests to cover:

- `transform_display_text` strips `<think>` block and returns modified text
- `transform_display_text` returns `None` when no `<think>` block is present (no-op)
- `after_persist_assistant` strips `reasoning_content` from in-memory message when `keep_thinking_in_history=False`
- `after_persist_assistant` leaves message unchanged when `keep_thinking_in_history=True`
- Graceful degradation: no ThinkingPlugin registered → `<think>` blocks appear in output, no crash
- Empty string from `transform_display_text` (edge case): treated as non-None; `display_response` receives `""` on normal branch, and `"(max tool-calling rounds reached)"` on max-turns branch (because `"" or fallback` applies only to the final `or` on the max-turns expression)

### Changes to `tests/test_agent_single_turn.py`

Update assertions that expect stripped output to work through the hook. The `after_persist_assistant` and `transform_display_text` hooks will be exercised by the registered `ThinkingPlugin` in any test that builds a full plugin graph. Tests that use a minimal plugin graph without `ThinkingPlugin` will see unstripped output.

### Graceful degradation

No `ThinkingPlugin` registered → `transform_display_text` hook returns `None` for all impls → `display_response = result.text` (unstripped). `after_persist_assistant` fires with no impls → `reasoning_content` stays in history. Neither produces a crash.

---

## Phase C: IdleMonitorPlugin

### Goal

Move the `IdleMonitor` class and its lifecycle wiring out of `AgentPlugin` into a standalone `IdleMonitorPlugin` in `sherman/idle.py`. `AgentPlugin.queues` becomes a public attribute that sibling plugins can reference.

### New file: `sherman/idle.py`

Contains two items:

**`IdleMonitor` class** — moved unchanged from `agent.py`. The class itself has no Sherman-specific coupling beyond accepting a `pm` and `queues` reference.

**`IdleMonitorPlugin`** — new plugin class:

```python
class IdleMonitorPlugin:
    depends_on = {"agent_loop"}

    def __init__(self, pm) -> None:
        self.pm = pm
        self._monitor: IdleMonitor | None = None

    @hookimpl(trylast=True)
    async def on_start(self, config: dict) -> None:
        agent = get_dependency(self.pm, "agent_loop", AgentPlugin)
        daemon_config = config.get("daemon", {})
        self._monitor = IdleMonitor(
            pm=self.pm,
            queues=agent.queues,  # public reference; dict is pre-initialized in AgentPlugin.__init__
            cooldown_seconds=daemon_config.get("idle_cooldown_seconds", 30),
            poll_interval=daemon_config.get("idle_poll_interval", 2),
        )
        self._monitor.start()

    @hookimpl
    async def on_stop(self) -> None:
        if self._monitor:
            await self._monitor.stop()
```

**Hook ordering notes:**

`on_start` uses `@hookimpl(trylast=True)`. Pluggy fires broadcast hooks in LIFO (last-in, first-out) order by default, so a plugin registered later fires first. `trylast=True` pushes `IdleMonitorPlugin.on_start` to the end of the call sequence regardless of registration order. This guarantees it fires after `AgentPlugin.on_start` has initialized `agent.queues` contents.

The `agent.queues` dict is initialized as `{}` in `AgentPlugin.__init__`, so the reference is valid before `AgentPlugin.on_start` runs. However, queues for specific channels are added lazily (on first message). `IdleMonitorPlugin` passes the dict reference to `IdleMonitor`, which iterates its `.values()` on each poll — it picks up new channels automatically.

`on_stop` does not use `trylast=True`. In LIFO order, `IdleMonitorPlugin.on_stop` fires before `AgentPlugin.on_stop`, which is correct: the monitor must be stopped before the queues it references are torn down.

### Changes to `sherman/agent.py`

1. Rename `_queues` to `queues` (public attribute, now part of the contract for sibling plugins). Update all internal references (`_get_or_create_queue`, `on_stop`, the docstring).

2. Remove `_idle_monitor` attribute from `__init__`.

3. Remove idle monitor creation and start from `_start_plugin`:
   ```python
   # Remove:
   self._idle_monitor = IdleMonitor(...)
   self._idle_monitor.start()
   ```

4. Remove idle monitor stop from `on_stop`:
   ```python
   # Remove:
   if self._idle_monitor:
       await self._idle_monitor.stop()
   ```

5. Remove `IdleMonitor` class from `agent.py` (moved to `idle.py`).

6. Update docstring on `AgentPlugin` to reflect that `queues` is now public and that idle monitoring is handled by `IdleMonitorPlugin`.

### Changes to `sherman/main.py`

Register `IdleMonitorPlugin` after `AgentPlugin`:

```python
from sherman.idle import IdleMonitorPlugin
idle_monitor_plugin = IdleMonitorPlugin(pm)
pm.register(idle_monitor_plugin, name="idle_monitor")
```

Registration order (post-Phases B and C):

```python
pm.register(registry, name="registry")
pm.register(core_tools, name="core_tools")
pm.register(cli_plugin, name="cli")
pm.register(irc_plugin, name="irc")
pm.register(task_plugin, name="task")
pm.register(subagent_plugin, name="subagent")
pm.register(compaction_plugin, name="compaction")
pm.register(thinking_plugin, name="thinking")
pm.register(agent_loop, name="agent_loop")
pm.register(idle_monitor_plugin, name="idle_monitor")
```

### Changes to `tests/test_idle_monitor.py`

Update imports: `IdleMonitor` moves from `sherman.agent` to `sherman.idle`. No behavioral changes to the tests themselves — `IdleMonitor` is unchanged.

### Tests affected by `_queues` rename

The following test files reference `agent._queues` and must be updated to `agent.queues`:

- `tests/test_agent_loop_plugin.py`
- `tests/test_agent_single_turn.py`
- `tests/test_logging.py`
- `tests/test_queue.py`

### New file: `tests/test_idle_monitor_plugin.py`

Tests to cover:

- `IdleMonitorPlugin.on_start` creates and starts an `IdleMonitor` pointing at `agent.queues`
- `IdleMonitorPlugin.on_stop` stops the monitor
- Graceful degradation: no `IdleMonitorPlugin` registered → `on_idle` hook never fires, no crash
- `on_start` LIFO ordering: monitor starts after `AgentPlugin.on_start` completes (verified by checking `agent.queues` reference is the same dict)

### Graceful degradation

No `IdleMonitorPlugin` registered → `on_idle` hook fires only if another plugin fires it directly. The `AgentPlugin` itself no longer creates or manages `IdleMonitor`. No crash.

---

## Files Created and Modified (Phases B and C)

| File | Action | Phase |
|------|--------|-------|
| `sherman/thinking.py` | NEW — `ThinkingPlugin` | B |
| `sherman/idle.py` | NEW — `IdleMonitor` class + `IdleMonitorPlugin` | C |
| `sherman/hooks.py` | Add `after_persist_assistant`, `transform_display_text` hookspecs | B |
| `sherman/agent.py` | Replace inline strip calls with hook calls; rename `_queues` → `queues`; remove `IdleMonitor` class; remove idle monitor lifecycle | B, C |
| `sherman/main.py` | Register `ThinkingPlugin`, `IdleMonitorPlugin` | B, C |
| `tests/test_thinking.py` | NEW | B |
| `tests/test_idle_monitor_plugin.py` | NEW | C |
| `tests/test_idle_monitor.py` | Update import path (`sherman.agent` → `sherman.idle`) | C |
| `tests/test_agent_loop_plugin.py` | Update `_queues` → `queues` | C |
| `tests/test_agent_single_turn.py` | Update `_queues` → `queues`; update ThinkingPlugin fixture assumptions | B, C |
| `tests/test_logging.py` | Update `_queues` → `queues` | C |
| `tests/test_queue.py` | Update `_queues` → `queues` | C |

---

## Cross-phase note: `plans/integration-tests.md`

The integration test design at `plans/integration-tests.md` references `agent._queues` in the `drain_all()` method of `IntegrationHarness`. After Phase C renames `_queues` to `queues`, that file must be updated. The fixture assembly section also does not include `ThinkingPlugin` or `IdleMonitorPlugin` — they should be added to match the post-B/C `main.py` registration order.

---

## Verification (each phase)

1. `uv run pytest` — all tests pass
2. `python -c "import sherman.thinking"` / `python -c "import sherman.idle"` — no circular imports
3. Unregistering the plugin produces graceful degradation (no crash, verified by test)

---

## Appendix: Design Review Report and Corrections

### Issues found in original design output

#### Critical issue 1: LIFO hook ordering for `IdleMonitorPlugin.on_start`

**Problem:** The original design stated "Register `IdleMonitorPlugin` after `AgentPlugin`; `on_start` fires in registration order, so `IdleMonitorPlugin.on_start` runs after `AgentPlugin`'s." This is wrong. Pluggy fires broadcast hooks in LIFO order (last-in, first-out). A plugin registered later fires first, not last. Without a `trylast` annotation, `IdleMonitorPlugin.on_start` would fire before `AgentPlugin.on_start`, meaning `agent.queues` would exist as an empty dict but `AgentPlugin` would not yet be fully initialized.

**Correction:** Add `@hookimpl(trylast=True)` to `IdleMonitorPlugin.on_start`. This pushes it to the end of the LIFO sequence regardless of registration order. Document that `agent.queues` is initialized as `{}` in `AgentPlugin.__init__` (not in `on_start`), so the reference is always valid even before `on_start` runs.

#### Critical issue 2: Empty string handling in `transform_display_text`

**Problem:** The original design used `or` fallback:
```python
display_response = await call_firstresult_hook(...) or result.text
```
This treats an empty string return as falsy, falling back to `result.text` even when the plugin intentionally returned `""`. It also conflates "no plugin returned a result" (None) with "plugin returned empty string" ("").

**Correction:** Use explicit `is not None` checks:
```python
transformed = await call_firstresult_hook(
    self.pm, "transform_display_text",
    channel=channel, text=result.text, result_message=result.message,
)
# Max turns branch:
display_response = (transformed if transformed is not None else result.text) or "(max tool-calling rounds reached)"
# Normal branch:
display_response = transformed if transformed is not None else result.text
```

The max-turns branch applies the `or "(max tool-calling rounds reached)"` after the None check, so an empty string from a plugin would still produce the fallback text (which is correct behavior: an empty response is not a valid reply when max turns are reached).

### Important issues incorporated

**`on_stop` LIFO ordering is correct as-is.** `IdleMonitorPlugin.on_stop` fires before `AgentPlugin.on_stop` in LIFO order (last registered fires first on stop). This is the desired behavior: stop the monitor before tearing down the queues it references. No `tryfirst` annotation needed on `on_stop`.

**`ThinkingPlugin` does not use `depends_on = {"registry"}`** — it uses `pm.get_plugin("registry")` with an early return if None. `depends_on` would cause `validate_dependencies()` to raise at startup if the registry were absent. Using `get_plugin` with a None check is the graceful degradation pattern for optional dependencies.

**`_queues` rename affects five test files**: `test_agent_loop_plugin.py`, `test_agent_single_turn.py`, `test_logging.py`, `test_queue.py`, and `test_idle_monitor.py` (import path change). All five must be updated as part of Phase C.

**`plans/integration-tests.md` references `agent._queues`** in the `drain_all()` method. This file must be updated after Phase C.
