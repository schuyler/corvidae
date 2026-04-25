# Replace PM monkey-patching with typed plugin dependencies

## Context

Three attributes are monkey-patched onto the `PluginManager` at runtime:
`pm.registry` (ChannelRegistry), `pm.task_plugin` (TaskPlugin), and
`pm.agent_plugin` (AgentPlugin). This makes the dependency graph invisible
to static analysis, produces `AttributeError` instead of clear startup
failures, and forces every test to repeat the same monkey-patch boilerplate.

The fix: plugins declare dependencies by name, pluggy's existing
`pm.get_plugin(name)` provides typed lookup, and a validation step at
startup fails fast if anything is missing.

## Approach

### 1. Add dependency infrastructure to `sherman/hooks.py`

- Add `get_dependency(pm, name, expected_type) -> T` — typed wrapper around
  `pm.get_plugin(name)`. Raises `RuntimeError` if not found, `TypeError` if
  wrong type.
- Add `validate_dependencies(pm)` — iterates registered plugins, checks each
  for a `depends_on: ClassVar[set[str]]` attribute, verifies all named
  dependencies are registered. Called once after all registrations, before
  `on_start`.

### 2. Register ChannelRegistry on the PM (`sherman/main.py`)

Replace `pm.registry = registry` with `pm.register(registry, name="registry")`.
ChannelRegistry has no methods matching hook specs, so pluggy will scan it
harmlessly and find no hooks.

Remove `pm.agent_plugin = agent_loop` — AgentPlugin is already registered
as `"agent_loop"`, accessible via `pm.get_plugin("agent_loop")`.

Add `validate_dependencies(pm)` call after all registrations, before
`await pm.ahook.on_start(config=config)`.

### 3. Update each plugin

**AgentPlugin** (`sherman/agent.py`):
- Add `depends_on = {"registry"}` (task_plugin is optional, not declared)
- In `on_start`: `self._registry = get_dependency(self.pm, "registry", ChannelRegistry)`
- Replace `self.pm.registry` → `self._registry` (3 call sites: lines 201, 323, 373)
- Replace `getattr(self.pm, "task_plugin", None)` → `self.pm.get_plugin("task")` at line 269

**CLIPlugin** (`sherman/channels/cli.py`):
- Add `depends_on = {"registry"}`
- In `on_start`: `self._registry = get_dependency(self.pm, "registry", ChannelRegistry)`
- Replace `self.pm.registry` → `self._registry` (lines 26, 37)

**IRCPlugin** (`sherman/channels/irc.py`):
- Add `depends_on = {"registry"}`
- In `on_start`: `self._registry = get_dependency(self.pm, "registry", ChannelRegistry)`
- Replace `self.pm.registry` → `self._registry` (lines 114, 125)

**TaskPlugin** (`sherman/task.py`):
- Remove `self.pm.task_plugin = self` from `on_start` (line 164)
- No `depends_on` needed — it doesn't depend on other plugins

**SubagentPlugin** (`sherman/tools/subagent.py`):
- Add `depends_on = {"agent_loop"}`
- In `_launch`: replace `self.pm.agent_plugin` → `get_dependency(self.pm, "agent_loop", AgentPlugin)` (line 57)
  (Resolved at call time, not `on_start`, because it's only needed during tool execution)

### 4. Update tests

Replace all `pm.registry = registry` with `pm.register(registry, name="registry")`.
Remove all `pm.task_plugin = ...` and `pm.agent_plugin = ...` assignments.
Tests that need to look up plugins use `pm.get_plugin("name")`.

Affected test files (grep confirms these):
- `tests/test_agent_loop_plugin.py` (~10 sites)
- `tests/test_agent_single_turn.py` (~7 sites)
- `tests/test_cli_plugin.py` (1 site)
- `tests/test_irc_plugin.py` (1 site)
- `tests/test_subagent.py` (1 site)
- `tests/test_task.py` (1 site)
- `tests/test_logging.py` (3 sites)
- `tests/test_main.py` (1 site, also update assertion)

### 5. Update `plans/design.md`

Update references to `pm.registry`, `pm.task_plugin`, `pm.agent_plugin` to
reflect the new `pm.get_plugin()` / `get_dependency()` pattern.

## Files modified

| File | Change |
|------|--------|
| `sherman/hooks.py` | Add `get_dependency()`, `validate_dependencies()` |
| `sherman/main.py` | Register registry as plugin, remove monkey-patches, add validation |
| `sherman/agent.py` | `depends_on`, resolve registry in `on_start`, update access sites |
| `sherman/channels/cli.py` | `depends_on`, resolve registry in `on_start`, update access sites |
| `sherman/channels/irc.py` | `depends_on`, resolve registry in `on_start`, update access sites |
| `sherman/task.py` | Remove `self.pm.task_plugin = self` |
| `sherman/tools/subagent.py` | `depends_on`, replace `pm.agent_plugin` access |
| `tests/test_*.py` | Replace monkey-patches with `pm.register()` |
| `plans/design.md` | Update plugin wiring documentation |

## Verification

1. `uv run pytest` — all tests pass
2. Grep for `pm\.registry\b`, `pm\.task_plugin`, `pm\.agent_plugin` — zero hits
   outside `plans/` and this plan
3. `mypy` or `pyright` (if configured) — no new type errors

---

Design complete — proceed to review.
