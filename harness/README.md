# Buster harness

Runs the minimal-profile corvidae agent ("Buster") unattended against the
standing llama-server on sagan, drives it over local IRC through a scripted
probe corpus, and writes a pass/fail report. One command from the Mac:

```
harness/run-remote.sh [rev]
```

defaults to `HEAD`. It pushes that rev to sagan, checks it out, and runs
the whole thing over one SSH session — output streams back live, exit code
is the run's pass/fail.

## What it checks

Six IRC channels, one probe each, every pass criterion mechanical (planted
tokens grepped from replies, or facts read straight out of `sessions.db`):

| Probe | Channel | Checks |
|---|---|---|
| `smoke_echo` | `#p-smoke` | Basic round trip |
| `tool_loop` | `#p-toolloop` | Multi-hop tool use through the task queue |
| `interleave` | `#p-interleave` | A second message gets answered while a slow tool call is in flight |
| `compaction` | `#p-compact` | A capacity-limited channel compacts under paced filler load, confirmed by a real summary row |
| `restart_recovery` | `#p-restart` | Conversation history survives a daemon restart |
| `kv_slot` | `#p-slots` | A follow-up turn reuses the cached prompt prefix instead of reprocessing it |

Full pass criteria: `.claude/plans/disentangle-buster.md` §5.5.

The harness introduces concurrency only where concurrency is the thing under
test. Every probe but `interleave` sends one line and waits for the bot's
reply before sending the next; an unawaited send turns queue depth into an
uncontrolled variable, which would make a probe like `compaction` measure
queue depth rather than compaction. `interleave` is the single deliberate
exception: sending the second message while the first's tool call is still
in flight is its subject.

## llama-server and ngircd

Both are standing, Schuyler-managed services on sagan. The harness never
starts, stops, or configures either one — it only checks they're up and
fails the run immediately and legibly if not:

- **llama-server** (`sagan:8080`) — `run.sh` checks `GET /health`, and
  snapshots `GET /props` / `GET /v1/models` verbatim into the run report
  so the model and serving flags are a recorded dimension, not an
  assumption. The `kv_slot` probe confirms that a follow-up turn reuses the
  cached prompt prefix instead of reprocessing it, reading
  `usage_log.cached_tokens` and `usage_log.prompt_tokens` out of
  `sessions.db` — no server introspection endpoint involved.
- **ngircd** (`sagan:6667`) — runs as the system `ngircd.service`
  (`systemctl status ngircd`), default config. `run.sh` checks that
  `127.0.0.1:6667` accepts a connection before doing anything else.

## One-time setup

On sagan:

```
git init --bare ~/code/llm/buster/repo.git
git clone ~/code/llm/buster/repo.git ~/code/llm/buster/repo
```

On the Mac, in this repo:

```
git remote add buster ssh://sagan/~/code/llm/buster/repo.git
```

That's it — `run-remote.sh` handles pushing the rev and checking it out
on sagan for every run after that.

## Prerequisites on sagan

Checked by an environment survey, current as of 2026-08-24:

- **screen** — present (4.09.00).
- **ngircd** — present and running as a system service (`systemctl
  is-active ngircd` → `active`, listening on `0.0.0.0:6667`), Schuyler's
  prerequisite. The harness only checks reachability; see "llama-server
  and ngircd" above.
- **uv** — present at `~/.local/bin/uv`, not on PATH for non-interactive
  SSH sessions (no login shell). `run.sh` and `restart-daemon.sh` both
  export `PATH="$HOME/.local/bin:$PATH"` explicitly to compensate.
- **python3** — system Python is 3.10 (corvidae needs >=3.13). This only
  matters for running corvidae itself, which goes through `uv run` — uv
  fetches a matching interpreter on first use. `driver.py` is stdlib-only
  and runs fine under the system 3.10, so it's invoked directly.
- **git** — present (2.34.1).
- First run bootstraps corvidae's `uv`-managed venv (interpreter + deps),
  which can take a few minutes — that's why timeouts throughout are
  generous, not just for LLM decode.

## Layout

```
~/code/llm/buster/
  repo/          deployed corvidae checkout (bare repo.git alongside)
  state/         Buster's instance state — sessions.db, corvidae.log,
                 metrics.jsonl, rendered agent.yaml, probe_data/
  runs/<ts>/     one self-contained report dir per run: report.json,
                 report.txt, daemon.log
  snapshots/     <ts>.tgz state snapshots (harness/snapshot.sh,
                 harness/restore.sh)
```

`buster-daemon` is stopped and restarted fresh every run. `ngircd` is the
system service, not something this harness's runs manage.

## Joining as a human

`ngircd` listens on all interfaces (`0.0.0.0:6667`), so connect directly
over the LAN — no SSH tunnel needed:

```
/server sagan 6667
```

(or point any IRC client at `sagan:6667`), no password, and join whichever
`#p-*` channel you want to watch or interject in.

## Chatting with Buster

```
harness/chat.sh
```

starts `buster-daemon` if it isn't already running (rendering
`state/agent.yaml` first if needed) — no probes, no snapshot — and prints
join instructions: any IRC client to `sagan:6667`, no password, then
`/join #chat`. `#chat` is a channel no probe touches, so it's safe to talk
in even while a probe run is in progress on `run.sh`'s daemon — but stay
out of the `#p-*` channels while probes are running, since a human
message there will confuse a probe's reply-matching.

Chatting mutates Buster's persistent state (`sessions.db` and friends)
just like a probe run does. Run `harness/snapshot.sh` first if you want a
rollback point.

## Manual state management

```
harness/snapshot.sh              # tar state/ -> snapshots/<ts>.tgz
harness/restore.sh <ts|path>     # inverse; stop buster-daemon first
harness/restart-daemon.sh        # bounce the daemon (used by the restart_recovery probe)
```

All three assume they're run on sagan with `~/code/llm/buster/repo` as
the checkout.
