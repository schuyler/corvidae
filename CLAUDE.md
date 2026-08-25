@AGENTS.md

## Environment

Dependencies are installed by `.claude/hooks/session-start.sh` in Claude Code on
the web. If an import fails, run `uv sync --extra dev` — that installs both the
`dev` extra (pytest, pytest-asyncio) and the `dev` dependency group
(pytest-timeout), which the pytest config needs.

```sh
uv sync --extra dev              # install everything
uv run pytest                    # full suite
uv run pytest tests/test_foo.py  # one file
uv run pytest --run-eval         # include live-LLM eval tests (deselected by default)
uv run corvidae                  # run the daemon; reads ./agent.yaml
```

`agent.yaml` is gitignored. Copy `agent.yaml.example` to create one.

`tool.pytest.ini_options` sets a global `timeout = 15`. Every async test inherits
it — this is the hang safety net AGENTS.md calls for, so don't disable it. If a
test legitimately needs longer, mark that test rather than raising the global.

## Architecture pointers

Three layers: an apluggy plugin system, the agent loop, and transport plugins.
`docs/design.md` is the authority; `docs/plugin-guide.md` is the how-to.

Things that are easy to get wrong and won't fail loudly:

- **A new plugin must be registered** in `[project.entry-points.corvidae]` in
  `pyproject.toml`. Without it the plugin is simply never loaded — no error.
- **Every `send_*` hook must broadcast-filter.** pluggy calls all transports for
  every send, so each transport's `send_message`, `send_thinking`,
  `send_tool_status`, and `send_progress` must open with
  `if not channel.matches_transport("<name>"): return`. Omitting it makes one
  transport emit another's output.
- **Declare dependencies** with `depends_on = frozenset({"registry"})` and
  resolve them via `get_dependency(self.pm, "registry", ChannelRegistry)`.
  Entry-point load order is not deterministic.
- **Channel IDs are persistence keys.** `transport:scope` is what `sessions.db`,
  the jsonl logs, and memory are keyed on. Changing how a scope is derived
  orphans existing history.
- Hook params with defaults can be silently dropped by pluggy — see
  `tests/test_hook_arg_binding.py` for the guard.

## Documentation

AGENTS.md requires docs to be current before work is complete. For most changes
that means one or more of:

- `docs/design.md` — architecture, hook specs, transports, schemas
- `docs/configuration.md` — any new or changed `agent.yaml` key
- `docs/plugin-guide.md` — anything a plugin author would need
- `agent.yaml.example` — new config keys, commented out

## Claude Code on the web

The container is ephemeral and the repo is cloned fresh, so nothing survives
unless it is committed and pushed. There is no `gh` CLI — use the GitHub MCP
tools. Don't open a pull request unless asked.
