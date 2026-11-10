@AGENTS.md

## Environment

Dependencies are installed by `.claude/hooks/session-start.sh` in Claude Code on
the web. If an import fails, run
`uv sync --extra dev --extra acp --extra discord --extra cognition --extra cross-channel-send --extra webhook-notify --extra context-panel --extra restart` —
that installs the `dev` extra (pytest, pytest-asyncio), the `dev` dependency
group (pytest-timeout), which the pytest config needs, the `acp` and `discord`
extras, whose transport tests import the SDKs, the `cognition` extra
(`packages/corvidae-cognition`: memory, appraisal, critique, funnel,
outcome_log, goals), which `tests/cognition/` imports, and the
`cross-channel-send` extra (`packages/cross-channel-send`: the send_message
and list_channels tools), which `tests/test_cross_channel_send/` imports, and the `webhook-notify` extra
(`packages/webhook-notify`: webhook deliveries as notifications), which
`tests/test_webhook_notify/` imports, and the `context-panel` extra
(`packages/context-panel`: the web console's context panel), which
`tests/test_context_panel/` imports. The
`cognition` extra includes `sqlite-vec` through the distribution's `vec`
sub-extra; leave it out to exercise the FTS5-only path (`docs/cognition.md`),
which is what CI's `cognition-fts` job does. That is CI's `full`
job. The kernel alone is `--extra dev --extra acp`; CI's `kernel` job runs
everything outside `tests/cognition/`, `tests/test_cross_channel_send/`,
`tests/test_webhook_notify/`, `tests/test_context_panel/` and `tests/corvidae_restart/` (which needs the
`restart` extra) that way, so a kernel module must never import `corvidae_cognition`
(`tests/test_kernel_boundary.py`).

```sh
uv sync --extra dev --extra acp --extra discord --extra cognition --extra cross-channel-send --extra webhook-notify --extra context-panel --extra restart  # everything
uv run pytest                    # full suite
uv run pytest --ignore=tests/cognition --ignore=tests/test_cross_channel_send --ignore=tests/test_webhook_notify --ignore=tests/test_context_panel --ignore=tests/corvidae_restart  # the kernel suite
uv run pytest tests/test_foo.py  # one file
uv run pytest --run-eval         # include live-LLM eval tests (deselected by default)
uv run ruff check .              # lint
uv run ruff format --check .     # formatting
uv run basedpyright              # type check
uv run corvidae                  # run the daemon; reads ./agent.yaml
```

Lint, formatting and type checks are **advisory**: CI runs them with
`continue-on-error`, so they report but do not fail a run. The tree is not
clean yet (445 ruff findings, 218 files that would reformat, 113 type
errors). Cleaning it and making the checks fail is #455; do not flip them
before then.

`agent.yaml` is gitignored. Copy `agent.yaml.example` to create one.

`uv.lock` is committed and pins every dependency, including the apluggy git
commit (`pyproject.toml` tracks its `main` branch). Upgrade deliberately and
commit the lock change: `uv lock --upgrade-package apluggy` for one package,
`uv lock --upgrade` for all. After editing dependencies in `pyproject.toml`,
run `uv lock`; CI syncs with `--locked` and fails on a stale lock.

CI (`.github/workflows/tests.yml`) runs four jobs on every push to `main` and
every PR: `kernel` (`--extra dev --extra acp`; checks that sqlite-vec, the
Discord SDK and `corvidae_cognition` are absent, then runs the kernel suite
above), `full` (every extra; checks sqlite-vec and the Discord
SDK load, then runs the whole suite), `cognition-fts` (the kernel install plus
`packages/corvidae-cognition` without sqlite-vec; checks it is absent, then runs `tests/cognition/` on the
FTS5 path), and `web-smoke` (kernel install plus the context panel and a
pinned Playwright and Chromium; runs `tests/test_web_render_smoke.py` and
`tests/test_context_panel/test_render_smoke.py` and fails
if it skips). It doesn't run the harness or `--run-eval`.

`tool.pytest.ini_options` sets a global `timeout = 15`. Every async test inherits
it — this is the hang safety net AGENTS.md calls for, so don't disable it. If a
test legitimately needs longer, mark that test rather than raising the global.

The timeout covers tests and their fixtures, not interpreter shutdown. An aiosqlite
connection left open by the last test to fail would hang pytest at exit
(its worker thread is non-daemon), so `tests/aiosqlite_guard.py`, imported by
`tests/conftest.py`, closes every connection a test opened through
`aiosqlite.connect` at teardown (#64). Prefer `async with` or the `db`
fixture anyway; the guard is a backstop.

Plugins default their output to relative paths under `state/` (attachments,
spill files), so a test that doesn't point that config at `tmp_path` writes
into the checkout. `tests/state_dir_guard.py`, also imported by
`tests/conftest.py`, fails any test that adds files under the working
directory's `state/`, naming the files. Run the daemon from another directory
while the suite runs, or its own writes there will trip it.

## Architecture pointers

Three layers: an apluggy plugin system, the agent loop, and transport plugins.
`docs/prd.md` is the reference for what must be true; `docs/design.md` describes
how the tree does it; `docs/plugin-guide.md` is the how-to.

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
  cognition set (memory, appraisal, critique, funnel, outcome_log, goals),
  which is the fastest config to reproduce a core-loop bug against.

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
tools. Always open a pull request when work is complete, so it can be
reviewed and merged even if the session goes idle before that happens.
