@AGENTS.md

## Environment

Dependencies are installed by `.claude/hooks/session-start.sh` in Claude Code on
the web. If an import fails, run
`uv sync --extra dev --extra acp --extra discord --extra cognition` — that
installs the `dev` extra (pytest, pytest-asyncio), the `dev` dependency group
(pytest-timeout), which the pytest config needs, the `acp` and `discord`
extras, whose transport tests import the SDKs, and the `cognition` extra
(`packages/corvidae-cognition`: memory, appraisal, critique, funnel,
outcome_log), which `tests/cognition/` imports. That is CI's `full` job. The
kernel alone is `--extra dev --extra acp`; CI's `kernel` job runs everything
outside `tests/cognition/` that way, so a kernel module must never import
`corvidae_cognition` (`tests/test_kernel_boundary.py`).

```sh
uv sync --extra dev --extra acp --extra discord --extra cognition  # everything
uv run pytest                    # full suite
uv run pytest --ignore=tests/cognition  # the kernel suite
uv run pytest tests/test_foo.py  # one file
uv run pytest --run-eval         # include live-LLM eval tests (deselected by default)
uv run corvidae                  # run the daemon; reads ./agent.yaml
```

`agent.yaml` is gitignored. Copy `agent.yaml.example` to create one.

`uv.lock` is committed and pins every dependency, including the apluggy git
commit (`pyproject.toml` tracks its `main` branch). Upgrade deliberately and
commit the lock change: `uv lock --upgrade-package apluggy` for one package,
`uv lock --upgrade` for all. After editing dependencies in `pyproject.toml`,
run `uv lock`; CI syncs with `--locked` and fails on a stale lock.

CI (`.github/workflows/tests.yml`) runs two jobs on every push to `main` and
every PR: `kernel` (`--extra dev --extra acp`; checks that sqlite-vec, the
Discord SDK and `corvidae_cognition` are absent, then runs everything outside
`tests/cognition/`) and `full` (every extra; checks sqlite-vec and the Discord
SDK load, then runs the whole suite). It doesn't run the harness or
`--run-eval`.

`tool.pytest.ini_options` sets a global `timeout = 15`. Every async test inherits
it — this is the hang safety net AGENTS.md calls for, so don't disable it. If a
test legitimately needs longer, mark that test rather than raising the global.

The timeout covers tests and their fixtures, not interpreter shutdown. An aiosqlite
connection left open by the last test to fail would hang pytest at exit
(its worker thread is non-daemon), so `tests/aiosqlite_guard.py`, imported by
`tests/conftest.py`, closes every connection a test opened through
`aiosqlite.connect` at teardown (#64). Prefer `async with` or the `db`
fixture anyway; the guard is a backstop.

## Architecture pointers

Three layers: an apluggy plugin system, the agent loop, and transport plugins.
`docs/design.md` is the authority; `docs/plugin-guide.md` is the how-to.

Things that are easy to get wrong and won't fail loudly:

- **A new plugin must be registered** in `[project.entry-points.corvidae]` in
  `pyproject.toml` (the root one for kernel plugins,
  `packages/corvidae-cognition/pyproject.toml` for cognition plugins).
  Without it the plugin is simply never loaded — no error.
- **Transports subclass `TransportPlugin`** (`corvidae/hooks.py`), which
  filters outbound `send` events for them. Inbound hooks
  (`should_process_message`, `on_message_admitted`/`rejected`/`persisted`)
  still reach every transport, so a transport's inbound hookimpls must open
  with `if not self.owns(channel): return`. Omitting it makes one transport
  act on another's messages.
- **Declare dependencies** with `depends_on = frozenset({"registry"})` and
  resolve them via `get_dependency(self.pm, "registry", ChannelRegistry)`.
  Entry-point load order is not deterministic.
- **Channel IDs are persistence keys.** `transport:scope` is what `sessions.db`,
  the jsonl logs, and memory are keyed on. Changing how a scope is derived
  orphans existing history.
- Hook params with defaults can be silently dropped by pluggy — see
  `tests/test_hook_arg_binding.py` for the guard.
- **Plugins can be switched off** via `plugins.disabled` in `agent.yaml`.
  `agent.minimal.yaml.example` uses it to run the core loop without the
  cognition set (memory, appraisal, critique, funnel, outcome_log), which is
  the fastest config to reproduce a core-loop bug against.

`harness/` drives the agent end-to-end over local IRC against a real
llama-server and greps mechanical pass criteria out of the replies and
`sessions.db` (`harness/README.md`). Behavior that only shows up across a
restart, a compaction, or two interleaved messages is proven there, not in
pytest.

## Documentation

AGENTS.md requires docs to be current before work is complete. For most changes
that means one or more of:

- `docs/design.md` — architecture, hook specs, transports, schemas
- `docs/configuration.md` — any new or changed `agent.yaml` key
- `docs/plugin-guide.md` — anything a plugin author would need
- `docs/cognition.md` — the cognition plugins (`packages/corvidae-cognition`)
- `docs/threat-model.md` — anything that changes what untrusted text can
  reach or what a tool can do (new tools, transports, tunables, roles)
- `agent.yaml.example` — new config keys, commented out

## Claude Code on the web

The container is ephemeral and the repo is cloned fresh, so nothing survives
unless it is committed and pushed. There is no `gh` CLI — use the GitHub MCP
tools. Don't open a pull request unless asked.
