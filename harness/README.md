# Buster harness

Runs the minimal-profile corvidae agent ("Buster") unattended against the
standing llama-server on the Buster host, drives it over local IRC through a scripted
probe corpus, and writes a pass/fail report. One command from the Mac:

```
harness/run-remote.sh [rev]
```

defaults to `HEAD`. It pushes that rev to the Buster host, checks it out, and runs
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

Full pass criteria: the probe functions' verdict logic in `harness/driver.py`.

The harness introduces concurrency only where concurrency is the thing under
test. Every probe but `interleave` sends one line and waits for the bot's
reply before sending the next; an unawaited send turns queue depth into an
uncontrolled variable, which would make a probe like `compaction` measure
queue depth rather than compaction. `interleave` is the single deliberate
exception: sending the second message while the first's tool call is still
in flight is its subject.

## llama-server and ngircd

Both are standing services on the Buster host, managed outside the harness. It never
starts, stops, or configures either one — it only checks they're up and
fails the run immediately and legibly if not:

- **llama-server** (`$BUSTER_HOST:8080`) — `run.sh` checks `GET /health`, and
  snapshots `GET /props` / `GET /v1/models` verbatim into the run report
  so the model and serving flags are a recorded dimension, not an
  assumption. The `kv_slot` probe confirms that a follow-up turn reuses the
  cached prompt prefix instead of reprocessing it, reading
  `usage_log.cached_tokens` and `usage_log.prompt_tokens` out of
  `sessions.db` — no server introspection endpoint involved.
- **ngircd** (`$BUSTER_HOST:6667`) — runs as the system `ngircd.service`
  (`systemctl status ngircd`), default config. `run.sh` checks that
  `127.0.0.1:6667` accepts a connection before doing anything else.

## One-time setup

First, tell the harness where the instance lives:

```
cp harness/config.local.sh.example harness/config.local.sh
$EDITOR harness/config.local.sh      # set BUSTER_HOST and BUSTER_ROOT
```

`config.local.sh` is gitignored — it names a machine on your network.

**Write `BUSTER_ROOT` absolute, not with `$HOME`.** It is a path on the Buster
host, but the file is read on your workstation, so `$HOME` would expand to the
wrong home — `/Users/you` reaching a box whose home is `/home/you`. The
scripts that ssh out (`run-remote.sh`, `session.sh`) require both variables
and refuse to start without them.

Scripts that run on the host itself read `BUSTER_ROOT` when present and
otherwise fall back to `$HOME/code/llm/buster` — correct there, since that
`$HOME` is the host's.

That fallback is why a remote checkout usually needs no copy of the config.
It stops being true the moment `BUSTER_ROOT` names anything else: `run-remote.sh`
would `cd` into the checkout you configured while `run.sh`, falling back, writes
state and run reports under `$HOME/code/llm/buster` instead. Nothing errors —
the reports just land somewhere nobody looks. If the instance lives anywhere
other than `$HOME/code/llm/buster` on the host, put a `config.local.sh` there
too.

On the Buster host:

```
git init --bare $BUSTER_ROOT/repo.git
git clone $BUSTER_ROOT/repo.git $BUSTER_ROOT/repo
```

On the Mac, in this repo:

```
git remote add buster ssh://$BUSTER_HOST/$BUSTER_ROOT/repo.git
```

That's it — `run-remote.sh` handles pushing the rev and checking it out
on the Buster host for every run after that.

## Prerequisites on the Buster host

The host has to provide:

- **screen**, **git**, and **uv**. If `uv` lives somewhere like
  `~/.local/bin`, note that non-interactive SSH gets no login shell and so
  no profile PATH — `run.sh` and `restart-daemon.sh` export
  `PATH="$HOME/.local/bin:$PATH"` themselves to compensate.
- **ngircd**, running and reachable. Yours to provide and supervise; the
  harness only checks that it answers. See "llama-server and ngircd" above.
- **python3**, any 3.10 or later. corvidae itself needs >=3.13, but that
  goes through `uv run`, which fetches a matching interpreter on first use.
  `driver.py` and `session_driver.py` are stdlib-only and 3.10-compatible
  precisely so they can be invoked directly under whatever the host has.

First run bootstraps corvidae's `uv`-managed venv — interpreter plus deps —
which can take a few minutes. That, not LLM decode, is why the timeouts
throughout are so generous.

## Layout

```
$BUSTER_ROOT/
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
/server $BUSTER_HOST 6667
```

(or point any IRC client at `$BUSTER_HOST:6667`), no password, and join whichever
`#p-*` channel you want to watch or interject in.

## Chatting with Buster

```
harness/chat.sh
```

starts `buster-daemon` if it isn't already running (rendering
`state/agent.yaml` first if needed) — no probes, no snapshot — and prints
join instructions: any IRC client to `$BUSTER_HOST:6667`, no password, then
`/join #chat`. `#chat` is a channel no probe touches, so it's safe to talk
in even while a probe run is in progress on `run.sh`'s daemon — but stay
out of the `#p-*` channels while probes are running, since a human
message there will confuse a probe's reply-matching.

Chatting mutates Buster's persistent state (`sessions.db` and friends)
just like a probe run does. Run `harness/snapshot.sh` first if you want a
rollback point.

## Shakedown sessions

A session is a recorded, turn-by-turn conversation with Buster — for
exercising things a scripted probe can't, like conversational drift or
compaction under human-paced exchanges — driven from the Mac one turn at a
time via `harness/session.sh`. See `docs/shakedown-findings.md` for the
campaign methodology and the historical record of what's been run and
found.

```
harness/session.sh start ['#channel']       # default #chat; prints a session id
harness/session.sh send  <id> <text...>     # one turn; prints the reply + latency
harness/session.sh stop  <id>                # stop the driver, collect artifacts
```

`start` requires `buster-daemon` already running (`harness/chat.sh` first);
it never starts the daemon itself. It launches
`harness/session_driver.py serve` on the Buster host under a `buster-session-<id>`
screen session, which holds one continuous IRC connection to the channel
for the whole session — no join/part churn between turns. Each `send` is
one blocking ssh call to the host; the driver replies with the turn's outbox
record (reply text, per-turn latency, timeout status) as JSON.

Two channels are available:

- `#chat` — the default 24000-token budget, for free-form conversation.
- `#s-compact` — a reduced 3000-token budget, for exercising compaction
  within a human-scale sitting (roughly 20-35 exchanges) instead of the
  probe corpus's paced filler.

Two sessions can run concurrently by starting them on different channels —
each gets its own screen session, session directory, and IRC nick (derived
from the channel name); nothing is shared between them but the daemon.

`stop` writes the stop sentinel, waits for the driver to exit cleanly, and
copies `sessions.db`, `metrics.jsonl`, and `corvidae.log*` from `state/`
into the session's own directory — a snapshot of what the session actually
saw, independent of whatever `run.sh` does to `state/` afterward. **Stop and
collect before running probes**: `run.sh` wipes `sessions.db*` and
`corvidae.log*` at the start of every run, and anything not collected first
is gone.

Sessions land under `$BUSTER_ROOT/sessions/<UTC-ts>/`, a sibling of
`runs/`:

```
sessions/<UTC-ts>/
  session.json        # channel, nick, bot_nick, started_at, stopped_at
  transcript.log      # timestamped >>/<</-- lines, appended live
  inbox/, outbox/      # the send/reply protocol's spool — inspectable mid-session
  sessions.db, metrics.jsonl, corvidae.log*   # copied in at stop time
```

The checkout at `$BUSTER_ROOT/repo` needs the session tooling
deployed to it — same path as probes (`harness/run-remote.sh` pushes a
rev; a plain `git pull` there works too since sessions don't need
`run.sh`'s setup steps).

Adding or changing a channel's `max_context_tokens` override in
`harness/buster.yaml.in` takes effect only after `state/agent.yaml` is
re-rendered and the daemon is bounced — `chat.sh` only renders
`agent.yaml` when it's missing, so after an override change, re-render it
by hand (the same `sed` `harness/chat.sh` uses) and run
`harness/restart-daemon.sh`.

## Replaying consolidation prompts

`harness/consolidate_replay.py` re-runs the daemon's consolidation
summarization over a message range you already recorded, so you can compare
`memory.consolidation_prompt` variants against identical input. A live session
can't do that — running the second variant changes the history it would have
been scored against.

```
uv run python harness/consolidate_replay.py \
    --db harness/fixtures/s-compact-20260909.db \
    --channel 'irc:#s-compact' --range 76:163 \
    --prompt-file prompts/memory_consolidation.md \
    --trials 5 --config /path/to/buster/agent.yaml
```

`--range` is inclusive at both ends, matching how a `memory` row's
`msg_id_start`/`msg_id_end` read. Rows are selected and filtered by the same
functions the daemon uses — `fetch_range_rows` and `dialog_from_rows` in
`corvidae/memory.py` — and sent as the same two-message call
`_summarize_range` builds, so a result here is a result about production
rather than about a lookalike that drifted.

`--config` wants the *rendered* Buster config, not a fresh one: model,
timeout, and `extra_body` all have to match what the daemon was running, or
you are measuring a different setup. It defaults to `agent.yaml` in the
current directory, which in the repo root is the local dev config — pass the
flag explicitly rather than measuring whichever model that happens to point
at. `--prompt-file` defaults to `DEFAULT_CONSOLIDATION_PROMPT` (the same text
`prompts/memory_consolidation.md` holds); `--trials` defaults to 1.

### The tunnel

llama-server listens on the Buster host's loopback, and your workstation may
have no route to it, so
replay needs a forward first:

```
ssh -N -L 8080:127.0.0.1:8080 "$BUSTER_HOST"
```

That makes `base_url: http://127.0.0.1:8080/v1` — what the config already
says — correct on the Mac too. Tear it down with
`ssh -O cancel -L 8080:127.0.0.1:8080 "$BUSTER_HOST"`, not by killing a process: with a
ControlMaster socket open, the forward belongs to the shared mux master, and
killing by port drops every session to that host. Without the tunnel the run
exits naming the base_url.

### Snapshotting a fixture

Point replay at a snapshot, not the live database, and take the snapshot with
`VACUUM INTO` while the daemon is still running:

```
. harness/config.local.sh
ssh "$BUSTER_HOST" \
    "sqlite3 'file:$BUSTER_ROOT/state/sessions.db?mode=ro' \
     'VACUUM INTO \"/tmp/fixture.db\";'"
scp "$BUSTER_HOST:/tmp/fixture.db" harness/fixtures/
```

`VACUUM INTO` will not overwrite, so delete the target before re-snapshotting.

Both halves matter. `cp sessions.db` gives you a file that opens cleanly and
is missing most of the history — the daemon runs in WAL mode and the main file
can lag the `-wal` by weeks. And a `mode=ro` connection needs the `-shm`
index, which it cannot create itself, so snapshotting *after* stopping the
daemon can fail outright.

### Output

Each run writes a timestamped directory under `--out` (default
`harness/replay_out`, resolved against the current directory — pass it
explicitly if you are not in the repo root):

- `input.txt` — the transcript exactly as sent
- `meta.json` — channel, the inclusive range and the resolved half-open
  bounds, raw row and dialog counts, prompt path and sha256, model, base_url
- `trials.jsonl` — one record per trial: raw output, parsed summary, latency

A malformed completion is recorded as that trial's `error` and the run
continues, since a model emitting unparseable JSON is a result. An unreachable
server aborts the whole run, since it is not.

Nothing here scores anything. The point is to put N summaries of identical
input side by side so a reader can judge whether a variant covers every topic
in a heterogeneous range and attributes facts to the right one.

## Manual state management

```
harness/snapshot.sh              # tar state/ -> snapshots/<ts>.tgz
harness/restore.sh <ts|path>     # inverse; stop buster-daemon first
harness/restart-daemon.sh        # bounce the daemon (used by the restart_recovery probe)
```

All three assume they're run on the Buster host with `$BUSTER_ROOT/repo` as
the checkout.
