@AGENTS.md

## Environment

Dependencies are installed by `.claude/hooks/session-start.sh` in Claude Code on
the web. If an import fails, run
`uv sync --extra dev --extra acp --extra discord --extra cognition --extra cross-channel-send --extra webhook-notify --extra github-webhook --extra context-panel --extra restart --extra timepiece --extra inotify --extra rss --extra signal --extra outbound-cap --extra goals-notify --extra vision --extra core-periphery --extra planning --extra alignment --extra whatsapp` —
that installs the `dev` extra (pytest, pytest-asyncio), the `dev` dependency
group (pytest-timeout), which the pytest config needs, the `acp` and `discord`
extras, whose transport tests import the SDKs, the `cognition` extra
(`packages/corvidae-cognition`: memory, appraisal, critique, funnel,
outcome_log, goals), which `tests/cognition/` imports, and the
`cross-channel-send` extra (`packages/cross-channel-send`: the notify_other_channel
and list_channels tools), which `tests/test_cross_channel_send/` imports, and the `webhook-notify` extra
(`packages/webhook-notify`: webhook deliveries as notifications), which
`tests/test_webhook_notify/` imports, and the `github-webhook` extra
(`packages/github-webhook`: GitHub events as terminal outcomes), which
`tests/test_github_webhook/` imports, and the `context-panel` extra
(`packages/context-panel`: the web console's context panel), which
`tests/test_context_panel/` imports, and the `timepiece` extra
(`packages/corvidae-timepiece`: ambient wall-clock stamps), which
`tests/test_timepiece/` imports, and the `inotify` extra
(`packages/corvidae-inotify`: completion pings for background jobs), which
`tests/test_inotify/` imports, and the `rss` extra
(`packages/rss-reader`: RSS/Atom feed polling), which
`packages/rss-reader/tests/` imports, and the `goals-notify` extra
(`packages/goals-notify`: goal state changes to the goal's origin channel),
which `tests/test_goals_notify/` imports, and the `vision` extra
(`packages/corvidae-vision`: image analysis via the local multimodal LLM), which
`packages/corvidae-vision/tests/` imports, and the `core-periphery` extra
(`packages/corvidae-core-periphery`: the `limits` plugin), which
`tests/test_core_periphery/` imports, and the `planning` extra
(`packages/corvidae-planning`: the planning critique lens), which
`tests/test_planning/` imports, and the `alignment` extra
(`packages/corvidae-alignment`: the Grice alignment critique lens), which
`tests/test_alignment/` imports; without `vision` the vision tests skip, so the
kernel suite collects that directory rather than ignoring it, which keeps the skip tested. The
`cognition` extra includes `sqlite-vec` through the distribution's `vec`
sub-extra; leave it out to exercise the FTS5-only path (`docs/cognition.md`),
which is what CI's `cognition-fts` job does. That is CI's `full`
job. The kernel alone is `--extra dev --extra acp`; CI's `kernel` job runs
everything outside `tests/cognition/`, `tests/test_cross_channel_send/`,
`tests/test_webhook_notify/`, `tests/test_github_webhook/`,
`tests/test_context_panel/`, `tests/test_timepiece/`, `tests/test_inotify/`,
`tests/test_outbound_cap/`, `tests/test_goals_notify/`, `tests/test_core_periphery/`,
`tests/test_planning/`, `tests/test_alignment/`,
`packages/rss-reader/tests/` and `tests/corvidae_restart/` (which needs the `restart` extra) that way, so a kernel module must never import `corvidae_cognition`
(`tests/test_kernel_boundary.py`).

```sh
uv sync --extra dev --extra acp --extra discord --extra cognition --extra cross-channel-send --extra webhook-notify --extra github-webhook --extra context-panel --extra restart --extra timepiece --extra inotify --extra rss --extra signal --extra outbound-cap --extra goals-notify --extra vision --extra core-periphery --extra planning --extra alignment --extra whatsapp  # everything
uv run pytest                    # full suite
uv run pytest --ignore=tests/cognition --ignore=tests/test_cross_channel_send --ignore=tests/test_webhook_notify --ignore=tests/test_github_webhook --ignore=tests/test_context_panel --ignore=tests/corvidae_restart --ignore=tests/test_timepiece --ignore=tests/test_inotify --ignore=tests/test_outbound_cap --ignore=packages/rss-reader/tests --ignore=tests/test_goals_notify --ignore=tests/test_core_periphery --ignore=tests/test_planning --ignore=tests/test_alignment  # the kernel suite
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
Discord SDK, `corvidae_cognition` and `corvidae_core_periphery` are absent, then runs the kernel suite
above), `full` (every extra; checks sqlite-vec, the Discord
SDK, `emoji` and `corvidae_vision` load, then runs the whole suite), `cognition-fts` (the kernel install plus
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

## Responding to the owner

The owner often reads on a phone and needs replies short enough to take
in at a glance.

- Keep every reply under 150 words.
- Handle one point per reply. If there is more to say, end with a line
  saying so, and wait to be asked.
- Don't repeat a question you have already asked. Hold open questions
  until the owner asks whether you have any.

## Review and merge agents

These rules apply only when you (Claude) are tasked with reviewing other
agents' work. They do not change how you work on your own PRs. When reviewing,
follow AGENTS.md ("Workflow labels", "Review loops", "Merging") to the letter.
An instruction such as "squash merge the PRs that pass review" adds to those
rules and replaces none of them.

- Review only PRs labeled `needs-review`. Ask before touching any other PR.
- Merge only a PR that carries `merge-ready`. Before every merge, quote the PR's
  current labels and the sha of the verdict that covers its head.
- Labels follow AGENTS.md "Workflow labels": `ready` and `claimed` go on the
  issue, the review stages on the PR (line 53), and the author normally sets them. A reviewer sets `needs-revision`
  or `merge-ready` only when the owner has told it to; otherwise it posts the
  verdict and stops, and the author moves the label.
- When the owner has told you to set labels: set `merge-ready` only on a clean
  verdict (`SOUND`, `APPROVE WITH FOLLOW-UPS`) covering the current head, and a
  PR that fails review gets `needs-revision` on the PR with a comment saying
  what to fix. Remove the old stage label in the same step.
- A loop stops after the fourth round (a fourth round ending `NEEDS-CHANGES`).
  It stays stopped until the owner says to continue, in a live session, and a
  stopped loop needs a meta-review posted on the PR first. A new review does
  not restart it.
- Reviews are posted through the owner's token, so the author field proves
  nothing (AGENTS.md "Merging"). A review counts as the blind round when it was
  written by an agent other than the author, from the diff, the requirement and
  the severity rules only. It never counts as the owner's permission to merge;
  that is the owner's words in a live session. Say so in the verdict.
- Record every review and label change on GitHub as soon as it is decided:
  post the verdict comment and, where you were told to set labels, set them in
  the same step. Do not wait to be asked, and do not leave a verdict only in
  the session. A review that is not on the PR did not happen.
- Never self-certify: the agent that reviewed a change does not also mark it
  `merge-ready` and merge it in one pass without the owner's permission for
  that change.
