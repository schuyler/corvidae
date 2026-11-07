---
name: write-corvidae-plugin
description: Use when adding or changing a corvidae plugin (a tool, a hook implementation, a producer of channel work, a transport helper). Gives a short checklist of placement rules, conventions and traps that fail silently. Written for small, fast agents: follow it in order, and stop and ask the owner where it says to.
---

# Plugin checklist

A short, ordered checklist for adding or changing a corvidae plugin, written
for small, fast agents. Follow the steps in order, and stop and ask the owner
where it says to. The traps in step 4 fail **silently** (no error, the plugin
just does nothing or reads the wrong value), so read them before you write code.

Depth lives in `docs/plugin-guide.md`. Requirements live in `docs/prd.md`. This
file is the short path; it does not replace either.

## 1. Before you write anything

1. Read `docs/prd.md` principles 1, 5, 9 and 12 (the numbered list near the top, roughly lines 50–190).
2. If the work comes from an issue or a plan, compare it with the PRD. List every
   place they disagree, **post the list and your plan as a comment on the issue,
   and wait for the owner.** Do not pick a side.
3. Decide where the code goes. Default: **a plugin, never the kernel.**
   - Principle 1: every capability is an optional plugin. The kernel is only the
     agent loop, queues, context window, channels, hooks, tools, LLM client,
     task queue, clock and persistence.
   - If you think the feature needs a change under `corvidae/`, **stop and ask
     the owner.** Say what you tried first and what you could not express with
     existing hooks. A plugin that adds a kernel queue role, API method or
     renderer for its own use is almost always the wrong design.
   - To put work on a channel's queue, call the `on_notify` hook (the scheduler
     and the task queue do). Do not add a method to `Agent`.

## 2. Layout (follow the cognition precedent)

A new optional capability is its own distribution, laid out like
`packages/corvidae-cognition`:

```
packages/<name>/pyproject.toml          # entry point + [tool.uv.sources] corvidae = { workspace = true }
packages/<name>/<package>/__init__.py   # the plugin class
tests/<package>/__init__.py             # tests live here, like tests/cognition/
tests/<package>/test_<thing>.py
```

Wiring, all of it, or CI breaks or the plugin never loads:

- `pyproject.toml` of the package: `[project.entry-points.corvidae]
  <name> = "<package>:<Class>"`. **Without this the plugin is never loaded and
  nothing is logged.** The entry-point name is what `plugins.disabled` matches.
- Root `pyproject.toml`: an optional extra that names the package, and
  `<name> = { workspace = true }` under `[tool.uv.sources]`. Then run `uv lock`
  and commit `uv.lock` (CI syncs with `--locked`).
- `.github/workflows/tests.yml`: the `kernel` job runs
  `pytest --ignore=tests/cognition`; add `--ignore=tests/<package>`, and add
  the extra to the `full` job's `uv sync`. The kernel must install, boot and
  pass its tests **without** your plugin.
- `corvidae scaffold <name>` generates a standalone project. Use it only as a
  starting point; adapt it to the layout above.

## 3. The plugin skeleton (this runs; tested against the real plugin manager)

```python
"""Corvidae plugin: echo (example)."""

from __future__ import annotations

import logging

from corvidae.hooks import CorvidaePlugin, hookimpl
from corvidae.tuning import config_value

logger = logging.getLogger(__name__)

DEFAULT_MAX_CHARS = 200


class EchoPlugin(CorvidaePlugin):
    """Gives the agent an `echo` tool."""

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        await super().on_init(pm, config)  # stores self.pm and self.config
        # config is the WHOLE agent.yaml dict: read your own section.
        raw = config_value(config, "echo.max_chars", DEFAULT_MAX_CHARS)
        if type(raw) is not int or raw < 1:  # malformed: loud, names key + value
            raise ValueError(
                f"Config error: echo.max_chars must be a positive integer, "
                f"got {raw!r} — check agent.yaml"
            )
        self._max_chars = raw

    @hookimpl
    def register_tools(self, tool_registry: list) -> None:
        tool_registry.append(self.echo)  # bare async callables are auto-wrapped

    async def echo(self, text: str) -> str:
        """Return the text unchanged.

        Args:
            text: What to echo.
        """
        if len(text) > self._max_chars:
            # Says what failed, which limit, and the next step.
            return (
                f"Could not echo: text is {len(text)} chars, the limit is "
                f"{self._max_chars}; shorten it."
            )
        return text
```

A test that goes through the real plugin manager (use this shape; a test that
only calls your methods will not catch hook-binding or registration bugs):

```python
async def test_registers_tool_through_the_real_pm():
    pm = create_plugin_manager()          # corvidae.hooks
    plugin = EchoPlugin()
    pm.register(plugin, name="echo")
    await pm.ahook.on_init(pm=pm, config={})
    registry: list = []
    pm.hook.register_tools(tool_registry=registry)  # register_tools is sync
    assert [Tool.from_function(f).name for f in registry] == ["echo"]
```

## 4. Traps that fail silently

0. **`annotations` are invisible to the model.** A key reaches the prompt only
   if the plugin class lists it in `model_visible_annotations` (read once at
   startup; only `origin` is declared today), and then only for short
   plugin-chosen text, never payload.

1. **Not registered** in the package's entry points: never loaded. Check by
   starting the daemon and reading the `"plugins loaded"` log line (`loaded` and
   `blocked` lists).
2. **Config is the whole dict.** `config.get("max_hops")` reads a *top-level*
   key. Your keys live under your own section (`config_value(config,
   "<name>.<key>", default)`). Test that a top-level key with the same name is
   ignored. No block at all: stay silent and use defaults. Malformed block:
   raise in `on_init`, naming the key and the rejected value.
3. **Hook params with defaults are silently dropped by pluggy.** In your
   `@hookimpl` signatures, never give a default to a param the hookspec requires.
   When you *call* a hook (`pm.ahook.on_notify(...)`), pass **every** parameter
   by keyword. `tests/test_hook_arg_binding.py` is the guard.
4. **No-argument constructors.** `pm` and config arrive in `on_init`. Never
   resolve another plugin in `__init__`; entry-point load order is not
   deterministic.
5. **Dependencies.** `depends_on = frozenset({"name"})` for a hard dependency;
   resolve it at runtime with `get_dependency(self.pm, "name", Type)` in
   `on_start`. For one you can work without, `optional_depends_on` and
   `self.pm.get_plugin("name")`, treating `None` as absent.
6. **Tools** are `async def` with type-annotated parameters; the first docstring
   line is the description the LLM sees. Parameters named `_something` are
   injected, not supplied by the LLM (`_ctx: ToolContext` gives the channel and
   tool call id). Never call blocking code (`requests`, `time.sleep`, a sync DB
   driver) from a tool: wrap it in `asyncio.to_thread`. Do not swallow
   `CancelledError`; clean up and re-raise.
7. **Inject channel work with `on_notify`**, passing `tool_call_id=None`. Origin
   is a convention in `annotations["origin"]`, and cognition plugins key on it (critique,
   gates, funnel, appraisal, outcome log): when you introduce a new origin,
   read those consumers and say in your plugin docstring what each does with it.
   **Do not put `trace_id` in `annotations` unless you mean to inherit the sender's
   trace**: the kernel then keeps `originating=False`, skips the per-exchange
   `turn_counter` reset and does not mint a new trace. Use another key (for
   example `source_trace_id`) to record where it came from.
8. **Provenance comes from code, not from the model.** Derive the sending
   channel from `corvidae.attribution.get_attribution()`, never from a tool
   argument. (`get_attribution` is not yet in the guide's export table, which
   lists only `set_attribution` and `reset_attribution`; if you use it, add it to
   the table in the same change, per trap 9.)
9. **Import only the documented kernel API**: the table in
   `docs/plugin-guide.md`, "What the kernel exports to plugins". Do not import
   `corvidae.agent`. The kernel never imports a plugin.
   `tests/test_kernel_boundary.py` enforces the allowlist for the cognition
   package; extend its scan to your package, and add a module to the allowlist
   and the guide's table only deliberately.
10. **Channel ids are persistence keys** (`transport:scope`). Never derive them
    from names or reformat them.
11. **Transports** (anything that owns channels) subclass `TransportPlugin` and
    every inbound hookimpl opens with `if not self.owns(channel): return`. See
    "Writing a transport" in the guide before starting one.

## 5. Conventions (AGENTS.md; reviewers check these)

- Red/green TDD: write the failing test first and watch it fail.
- Every pytest run is under a timeout (the repo sets `timeout = 15`; do not
  raise or disable it; mark the one slow test).
- Never `except Exception: pass`. Log with the traceback, pass through, or
  re-raise.
- Type hints on everything; a comment on every block saying why it is there.
- **Error text** has two audiences. The human sees what failed and what to do
  next, and **names the identifier** (the path, channel, tool, config key).
  The log line carries the exception type and traceback. Try the cheap rescue
  first (retry, fall back, skip the bad record) and log it at WARNING.
- **Test the message, not just the raise**: assert the identifier and the
  remedy appear in the text.
- On rejected input, name the key, the rejected value and the accepted values.

## 6. Docs (same change, not a follow-up)

- `docs/configuration.md` and `agent.yaml.example` (commented out): every new
  or changed key and its default.
- `docs/plugin-guide.md`: anything another plugin author needs to know.
- `docs/design.md`: architecture, hook specs, schemas.
- `docs/threat-model.md`: required when untrusted text can now reach somewhere
  new, or a tool can do something new (PRD principle 9).
- `CLAUDE.md`: if you added an extra or a CI job, update its install line.
- Do not edit `docs/prd.md` unless the owner asked; requirements changes go
  through the owner.

## 7. Verify before you say it is done

```bash
uv sync --extra dev --extra acp --extra discord --extra cognition --extra <your-extra>
uv run pytest tests/<package>                               # your tests, red then green
uv run pytest --ignore=tests/cognition --ignore=tests/<package>  # the kernel suite, without you
uv run pytest                                               # everything, with every extra
uv run ruff check . ; uv run ruff format --check .          # advisory today; do not add new findings
```

Done means:

- [ ] `git diff origin/main --stat -- corvidae/` shows nothing (or the owner approved each kernel change).
- [ ] The kernel suite passes with your plugin **absent** (only `--extra dev --extra acp`).
- [ ] `plugins.disabled: [<name>]` removes your tools and changes nothing else.
- [ ] The `"plugins loaded"` log line lists your plugin when it is enabled.
- [ ] `uv.lock` is committed and `uv sync --locked` succeeds.
- [ ] Every error path has a test that asserts the identifier and the remedy.
- [ ] The docs in step 6 are updated.
- [ ] You compared the work with the PRD and flagged every difference to the owner.
