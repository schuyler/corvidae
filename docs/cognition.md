# Cognition plugins

The cognition plugins give the agent memory and judgement on top of the
kernel's agent loop. They live in a second distribution in this repository,
`corvidae-cognition` (`packages/corvidae-cognition/`), installed with the
`cognition` extra:

```sh
uv sync --extra cognition
```

`uv sync` installs exactly the extras you name and removes the rest, so add
`--extra discord` or `--extra acp` to the same line if you use those
transports.

| Plugin | Entry point | What it does |
|---|---|---|
| `MemoryPlugin` | `memory` | Consolidates compacted conversation into durable memories and retrieves them into context; owns `sqlite-vec` |
| `MemoryToolsPlugin` | `memory_tools` | `search_memory` and `recall_raw` tools for the model |
| `FunnelPlugin` | `funnel` | Admits retrieved context into the window under a budget |
| `AppraisalPlugin` | `appraisal` | Scores every inbound message at the gate (novelty, commitment density, disagreement, question, salience), then appraises the exchange once the agent has answered |
| `CritiquePlugin` | `critique` | Reviews the agent's replies after the fact through the stylistic lenses (predictive, constrained, adversarial) and the provenance check |
| `GoalsPlugin` | `goals` | The agent's open goals: three tools to add, update and retire them, and a CONTEXT entry listing the open ones each turn |
| `OutcomeLogPlugin` | `outcome_log` | Records each exchange and its outcome in `exchange_log` |
| `GatePlugin` | `gates` | The engagement gate (may the agent ignore a message?) and the decide gate (may it stay silent instead of answering?), both over the appraisal vector |
| `HeartbeatPlugin` | `heartbeat` | Starts a turn on `internal:heartbeat` when the whole system has been quiet for `heartbeat.after` seconds; the turn reviews recent memory (and goals, when loaded) and acts only if something warrants it. Asleep unless configured |

Entry-point names are the same as when these plugins lived in the kernel, so
`plugins.disabled` lists in existing `agent.yaml` files keep working. Without
the extra, those names match nothing and disable nothing, which is harmless.

## What the plugins rely on

They are ordinary plugins: they use only the kernel API documented in
[plugin-guide.md](plugin-guide.md#what-the-kernel-exports-to-plugins), which
`tests/test_kernel_boundary.py` enforces, and the kernel never imports them.
Their loggers keep their `corvidae.*` names (`corvidae.memory`,
`corvidae.appraisal`, ...), so log configuration and log lines are the same
as before the split.

They own these tables in `sessions.db`: `memory`, `memory_fts`,
`memory_vec`, `embedding_meta`, `consolidation_watermark`,
`retention_meta`, `retrieval_log`, `exchange_log` and `goals`.

**One object crosses the line.** `MemoryPlugin` creates `message_fts` and
two triggers, `message_log_ai` and `message_log_au`, on the kernel's
`message_log` table. The kernel never reads them. They live in the database,
not in code, so an operator who removes the extra keeps them, and the
triggers keep indexing into a table nothing reads. That is harmless, but
`memory_vec` stays too, and `corvidae redact` needs sqlite-vec to touch it:
reinstall the extra before redacting, or every form refuses
([design.md](design.md#operator-redaction-corvidae-redact)).

## Design

### MemoryPlugin

`memory.py` — autobiographical memory. Entry point
name: `"memory"`; `depends_on = {"persistence", "llm", "clock", "idle"}`. Absorbs the former
`DreamPlugin` embryo (removed): compaction becomes memory formation instead
of file-based fact scraping.

#### Consolidation (write path)

Two triggers, one watermarked code path:

- **`on_compaction(channel, summary_msg, retain_count, compacted_ids)`** —
  the moment dialog leaves the active window it enters long-term memory.
  The compacted `message_log` id range rides the hook payload (rowid
  threading, below); consolidation never depends on the summary row being
  in the DB.
- **Channel idle** — a channel that has been quiet (per the kernel idle
  tracker, [design.md](design.md#clock-and-idle-tracker-clockpy-idlepy))
  for `memory.idle_consolidate_after` seconds gets its un-consolidated
  tail consolidated, and again each period while the quiet lasts. Channels
  with history but no traffic since boot count as quiet since their last
  message, so a restart does not lose them. A channel that never goes
  quiet never reaches this trigger; it is consolidated through compaction,
  as dialog leaves the window.

Both spawn a tracked plugin-owned `asyncio.create_task` (never the
TaskQueue — every TaskQueue completion wakes the main model). The task
reads the per-channel **consolidation watermark**, fetches the working
range, produces the record (first-person summary with epistemic framing,
JSON out) — from a fork of the live window when one is available (see
[Forks](design.md#forks)), else one standalone `llm.background` call
(`prompts/memory_consolidation.md`); both are constrained by a
`response_format` JSON schema (`summary` required, `topic_tags` and
`participants` optional), like appraisal and critique, so a model can't
answer the transcript instead of writing a record —
scores an importance prior through the pluggable
`MemoryPlugin.importance_prior` (default: cheap-model `RubricPrior`,
schema-constrained to `{"importance": number}`, 0.5 on failure; `AppraisalPlugin` wraps it in `AppraisalPrior` when
appraisal is enabled; the prior and the record's mean stage-2 valence read only
the record's own channel's exchanges, and a failed `exchange_log` lookup is logged
and falls back to the rubric prior and NULL valence), embeds the summary via
`llm.embedding`, and commits the memory row (with its vec row and any near-duplicate
merge) together with the watermark advance in one compare-and-set
section.

### Corrupt `message_log` rows

A corrupt row (invalid JSON, invalid UTF-8, JSON that is not an object, an
oversized integer, or a `message_type` outside `MessageType`) never wedges
memory (#232). Each reader selects the column as `CAST(message AS BLOB)` and
decodes it with the kernel's `decode_message_row`, then skips the row with a
WARNING naming the row id, the channel and the next step (inspect or redact
the row):

- **Consolidation** counts the row as non-dialog. The record is built from
  the valid dialog, and the watermark advances past the whole range, corrupt
  row included. A range whose only rows are corrupt advances the watermark
  with no record and no LLM call.
- **`recall_raw`** returns the valid rows of the range in order and omits the
  corrupt one.
- **The `message_fts` backfill** indexes every valid row and indexes a corrupt
  row as empty text, so it is not retried on every start.

**Write connection and atomicity (#221).** That section is one SQLite
transaction (`BEGIN IMMEDIATE` ... `COMMIT`) on a dedicated connection
only `MemoryPlugin` writes to, opened in `on_start` once the schema
exists (lazily on first write when the persistence DB wasn't open yet)
and closed in `on_stop`. It opens the same file as persistence, in the
journal mode persistence runs in (WAL by default), with sqlite-vec loaded
when memory uses vectors and `busy_timeout` set to
`WRITE_BUSY_TIMEOUT_MS` (5000, a module constant). So no other plugin's
commit can land mid-section and a rollback cannot discard another
plugin's pending row; a consolidation's effects are all visible or none
are. Any failure rolls back, then logs a WARNING naming the channel, the
trigger and the message-id range, saying the range will be retried;
the watermark hasn't moved, so the next trigger stores exactly one
record. A failed rollback and a write blocked past the busy timeout (the
WARNING names the database path and the timeout) are logged too.
`write_record` (goal retirement) uses the same transaction, as do retention's
three passes (demotion, re-promotion, vec backfill): each batch commits whole
or rolls back whole, and the backfill's embeddings are computed before the
transaction opens. `_db_lock`
stays, serializing memory's own writers (consolidation versus goals);
it no longer carries cross-plugin atomicity. A persistence DB with no
file (in-memory, tests only) can't be opened twice, so memory's writes
share the persistence connection there, with a one-time WARNING. The row's `created_at` is the
`message_log.timestamp` of the last row in the source range, i.e. when
the conversation happened, not when consolidation ran, so history
consolidated long after the fact is aged correctly by retrieval and
retention.

**Near-duplicate merge.** Before inserting the new vector, consolidation
looks for the nearest indexed record with cosine similarity
≥ `memory.dup_threshold` among the records in the channel's retrieval
scope (#58). A record on another channel qualifies only if that channel's
scope has the same reach (`ChannelScope.same_reach`), because groups are
not transitive: with groups {A, B} and {A, C}, folding C's record into one
owned by A, and later A's into one owned by B, would hide the fact from C.
Equal reach keeps the survivor visible to exactly the channels that could
see the superseded record. Under a single `*` group every channel
qualifies. The old record is superseded (`superseded_by`, `indexed=0`,
vector deleted); the survivor takes the summed retrieval count, the
higher importance, and the later of the two records' dates. Message
ranges fold only within one channel: `message_log` ids are global, and
`recall_raw` and `/redact` read a record's range on its own channel, so on
a cross-channel merge each record keeps its own range and the superseded
record still leads to its channel's source dialog. Overlapping triggers on the same
range discard the loser's record — no duplicates. Embedding failure
stores the record with `embedded=0` for later backfill; the summary text
is canonical and vectors are a rebuildable cache. All consolidation LLM
calls carry `stage="consolidation"` attribution, along with the `trigger`
that started the run (`"compaction"` or `"idle"`) — set once at
`_consolidate_range`'s entry, so it rides into the `on_llm_response`
attribution snapshot for that call, into `MetricsPlugin`'s emitted tags,
the `usage_log.trigger` column, and the `memory consolidated` completion
log line.

#### Retrieval (read path)

`before_agent_turn` runs when the window tail is a plain user MESSAGE
(notification-triggered turns skip retrieval entirely). The inbound text is embedded and matched against `memory_vec`
(sqlite-vec exact KNN, cosine distance); candidates are filtered to
`indexed=1 AND redacted=0` and channel-compartmentalized (`channel_id`
match, or shared scope via `memory.channel_groups`, whose members may be
`*` / `?` globs matched with SQLite `GLOB`). Scores are
`cosine_similarity × exp(-age_days / memory.half_life_days)`, where
`age_days` runs from `created_at`, or from `max(created_at,
last_retrieved_at)` under `memory.retrieval.recency_from: last_used`.
Candidates rank by score and are annotated into relevance bands (strong
≥ 0.75, moderate ≥ 0.60, else weak; weak drops unless nothing else
matched). The bands apply to the score by default, or to raw similarity
under `memory.retrieval.band_on: similarity`, so an old but apt memory is
not presented as a guess. The `(age)` shown always runs from
`created_at`. Winners enter the window as
`[band] (age) summary` lines through the admission funnel — never by
direct CONTEXT append. Admitted memories get their access statistics
bumped (`retrieval_count`, `last_retrieved_at` — the write half of
reconsolidation), and every retrieval writes a `retrieval_log` row
(the top-ranked candidate's decayed score and raw similarity,
hit/admitted counts, degradation flag).

**Graceful degradation:** encoder down → FTS5 keyword retrieval over
`memory_fts` (flagged `degraded_to_fts=1`); sqlite-vec unloadable → FTS5
with one clear startup warning; no `llm.embedding` role → records stored
with `embedded=0`, retrieval FTS5-only; a stored encoder differing from
config → ERROR and embeddings disabled rather than silently mixing
encoders. Nothing in the memory path raises into the turn loop.

#### Schema

`memory` (records, with `indexed`, `superseded_by`, `redacted`, and
`embedded` columns), `consolidation_watermark`, `embedding_meta`,
`memory_fts` (external-content FTS5 with insert/update sync triggers; no
delete trigger — memory rows are never deleted), `memory_vec` (vec0,
created only when the extension loads), `retrieval_log`, `retention_meta`
(one-row table persisting `last_run` for the retention job interval), and
`message_fts` (regular FTS5 table over `message_log` content, maintained by
`message_log_ai`/`message_log_au` triggers; queried and cleared by `corvidae
redact` — not by `search_memory`, which queries `memory_fts`). All on the persistence connection in
`sessions.db`; `message_log` stays append-only except for the
operator-only redaction tombstone.

The epistemic-calibration prompt fragment (`prompts/memory_calibration.md`)
is documented but NOT auto-injected — hedging vocabulary is persona, not
architecture.

### FunnelPlugin

`funnel.py` — the context-admission funnel. Entry
point name: `"funnel"`. The single chokepoint for tail CONTEXT admission:

1. **Dedupe** against CONTEXT already in the window.
2. **Budget** per source (`funnel.default_budget`, default 512 tokens;
   `funnel.budgets.<source>` overrides), greedy in given order, drops
   logged rather than silent.
3. **Frame** every admitted batch in explicit data-not-instructions
   wrapping with a source label (injection defense, written once here).
4. **Append + persist** one CONTEXT message at the tail through
   `persist_message`, with the rowid attached, so the window matches
   its reload. Stale CONTEXT retires by aging past the compaction boundary.

Callers: memory retrieval (and optionally date/time grounding). The
funnel also owns deferred registration: non-tool-call notification
payloads queue per `(channel, origin)` via `register_and_wake()`, one
stub notification wakes the channel per pending pair, and the drain in
`before_agent_turn` admits everything queued for the triggering
exchange's origin. `register()` queues without a stub: those payloads wake
nothing and are admitted at the channel's next turn, whatever its origin
(user turns included), before the origin-matched drain.

## Plugin reference

### MemoryPlugin (`packages/corvidae-cognition/corvidae_cognition/memory.py`)

Autobiographical memory: consolidates compacted dialog into first-person
memory records (SQLite + sqlite-vec + FTS5) and retrieves them into the
prompt through the admission funnel. Entry point name: `"memory"`;
`depends_on = {"persistence", "llm", "clock", "idle"}`. Absorbs the removed `DreamPlugin`.

Implements five hooks and two kernel timers:

- `on_start` — creates the memory schema on the persistence connection
  (retried lazily on first use; on_start ordering is LIFO across plugins),
  runs the retention check, then registers the two clock-driven pieces
  below. Without the kernel clock and idle tracker it logs a warning and
  skips them.
- `before_compaction` — with `agent.forks` on for `consolidation`, forks
  the memory record from the prefix about to be dropped (when every
  dialog row of the range is in it) and stashes it for the task below.
- `on_compaction` — spawns a tracked background task consolidating the
  `compacted_ids` range above the per-channel watermark, using the stashed
  fork record when there is one for that range and watermark, else one
  standalone `llm.background` call.
- `before_agent_turn` — embeds the inbound user text, scores candidates
  (`similarity × exp(-age_days / half_life)`, with age from `created_at` or,
  under `memory.retrieval.recency_from: last_used`, from the last
  retrieval), bands them (strong/moderate/weak, on the score or, under
  `memory.retrieval.band_on: similarity`, on raw similarity), admits winners through the funnel, bumps access
  stats for admitted records, and writes a `retrieval_log` row.
- `on_stop` — cancels the idle watch and retention timers, then in-flight
  background tasks.

Two kernel registrations replace the old `on_idle` hook:

- **Idle watch** (`watch_channels`, period `memory.idle_consolidate_after`)
  — consolidates the un-consolidated tail of the quiet channel, forking
  the live window when forks are on and the range is in it. One run per
  channel at a time.
- **Retention timer** — fires at `last_run + memory.retention.interval`
  on the clock. It then prefers system quiet: it registers a system idle
  watch for 60 s of quiet and a deadline `memory.retention.max_delay`
  seconds out, and runs retention when the first of them fires. Traffic
  or a frequent heartbeat can delay a run by at most `max_delay`. The
  `last_run` timestamp is persisted in `retention_meta`, so restarts do
  not reset the schedule, and a daemon down longer than the interval runs
  the job at startup. At boot memory starts before persistence opens the
  DB, so arming the timer is deferred: one plain WARNING (`retention:
  persistence DB not open yet; arming the retention timer deferred,
  re-arming in 5.0s`), then it re-arms 5 s later. The line repeats every
  5 s for as long as the DB stays closed (persistence disabled or failed). A real failure to read
  `retention_meta` with the DB open logs a WARNING with its traceback and
  re-arms the same way.

Changes to `memory.idle_consolidate_after` and `memory.retention.*` take
effect at restart.

The importance prior is pluggable (`MemoryPlugin.importance_prior`,
default `RubricPrior`); `AppraisalPlugin` wraps it in `AppraisalPrior`
when appraisal is enabled; it reads only the consolidating channel's own
exchanges (message ids are global), logging a failed lookup and falling back to
the rubric prior and NULL valence. Configuration is documented in
[configuration.md](configuration.md) (`memory.*`, `llm.embedding`).

`recent(channel_id, limit)` is the one read API for other plugins: the
newest `limit` records (indexed, not redacted) owned by the *other*
channels in `channel_id`'s scope, as `{summary, channel_id, age_seconds}`
dicts, newest first, with ages from the kernel clock. The caller's own
records are left out. `HeartbeatPlugin` uses it. It raises `RuntimeError`
while the memory DB is not open.

**Without this plugin:** no memory records form and no retrieval happens;
compaction remains plain lossy summarization.

A malformed consolidation reply is retried once; if the retry is also
unusable, the range is stored with its raw transcript as the summary rather
than being dropped (the LLM call has already been paid for).

### MemoryToolsPlugin (`packages/corvidae-cognition/corvidae_cognition/memory_tools.py`)

Registers the `search_memory` and `recall_raw` agent tools — the active
memory surface ("remember harder"). Entry point
name: `"memory_tools"`; `depends_on = frozenset({"memory"})`. Reaches the
DB and channel-scope logic through `get_dependency(pm, "memory",
MemoryPlugin)`.

Implements one hook:

- `register_tools` — appends `search_memory` and `recall_raw` to
  `tool_registry`.

**Compartment scoping:** both tools enforce the calling channel's
`_channel_scope` (the channel itself plus any group siblings configured in
`memory.channel_groups` — identical to the scope `MemoryPlugin`'s passive
retrieval uses). `_channel_scope` returns a `ChannelScope`, not a list:
group members may be `*` / `?` globs, so test membership with
`scope.contains(channel_id)` and build SQL with
`scope.sql("m.channel_id")`, which emits `IN (...)` for literal ids and
`GLOB ?` for patterns (an empty `ChannelScope` emits `(0)` and matches
nothing). `search_memory`'s `channel=` argument narrows by equality and is
never treated as a pattern. The calling channel comes from `_ctx.channel`; when
`_ctx.channel is None` both tools return an error string and never fall
back to unscoped access. The tools are broader than passive retrieval in
*status* (they reach demoted, superseded, and redacted-tombstone records)
but never broader in *compartment* — channel B cannot enumerate or raw-recall
channel A's memories.

Both tools retry briefly when the memory schema is still being created by
another task. If `search_memory`'s FTS5 query fails, it falls back to a
`LIKE` scan over record summaries instead of returning an internal error.

**Tools:**

| Tool | Parameters | What it does |
|------|------------|--------------|
| `search_memory` | `query: str`, `channel: str\|None` (default `None`), `tags: list[str]\|None` (default `None`), `after: str\|None` (default `None`), `before: str\|None` (default `None`), `include_demoted: bool` (default `True`), `max_tokens: int` (default `2000`) | FTS5 keyword search over `memory_fts`. Returns a numbered list `id. score (age)[flags] summary` sorted by FTS relevance, capped at approximately `max_tokens`. `channel` narrows within the calling channel's scope; a value outside scope returns an error string naming visible channels. `after`/`before` are ISO-8601 (`YYYY-MM-DD` or full timestamp; naive values treated as UTC) compared against `created_at` (`after` ⇒ `≥`, `before` ⇒ `<`); unparseable values return an error string. `include_demoted=False` restricts to `indexed=1` records, hiding demoted, superseded, and redacted-tombstone records (all are `indexed=0`). `tags` filters by `topic_tags` JSON array intersection. **Does not update access stats.** |
| `recall_raw` | `memory_id: int`, `max_tokens: int` (default `1000`) | Replays the raw `message_log` dialog for the record's `[msg_id_start, msg_id_end]` range using `WHERE channel_id = record.channel_id AND id BETWEEN ...` — the channel predicate is load-bearing because the `message_log` id sequence is global across all channels. Returns a participants header (`[memory N: messages S–E, participants: {...}]`) followed by verbatim `role: content` lines, truncated at `max_tokens` with an explicit marker. Works on demoted, superseded, and tombstoned records. Rejects out-of-scope record IDs without bumping stats. **Bumps `retrieval_count` and `last_retrieved_at`** on the record — the bump is what makes the record eligible for re-promotion by the next retention job run. |

**Recall-stat semantics:** `recall_raw` bumps the record's access stats;
`search_memory` does not. `search_memory` is a catalog for choosing what
to recall; bumping every listed record would inflate scores for marginal
matches and distort retention scoring. The re-promotion path is:
`search_memory` surfaces a demoted record (flagged `[demoted]`) → agent
calls `recall_raw` → next retention job run finds the score above the
demotion threshold and re-promotes.

**Without this plugin:** the `search_memory` and `recall_raw` tools are
not available. Memory records are still formed and passively retrieved
by `MemoryPlugin`; only the agent's active access surface is absent.

### FunnelPlugin (`packages/corvidae-cognition/corvidae_cognition/funnel.py`)

The context-admission funnel — the single chokepoint for tail CONTEXT
admission. Entry point name: `"funnel"`. Exposes
`admit(channel, conv, source, entries, budget_tokens=None) -> int`:
dedupes against CONTEXT already in the window, budgets tokens per source
(drops are logged, never silent), wraps everything in explicit
data-not-instructions framing with a source label, appends one CONTEXT
message at the tail, and persists it with the rowid attached.

**Without this plugin:** sources that route through the funnel (memory
retrieval) log a warning and admit nothing.

`register_and_wake(channel, origin, source, entries)` is the deferred
admission path for notification producers without a `tool_call_id`:
`origin` is stamped into the wake stub's `meta["origin"]` (see
[Traces](plugin-guide.md#traces)) so `before_agent_turn`'s drain, and any
plugin gating eligibility off that origin, never has to infer it from
stub text. The `'user'|'reminder'|'critique'|'heartbeat'|'task'`
taxonomy is a convention cognition plugins (AppraisalPlugin,
CritiquePlugin, OutcomeLogPlugin) share over this key — not a core
concept. `'user'` is recognized by subscribing to `on_message_admitted`
rather than by a value in `meta`; `'task'` is the default OutcomeLogPlugin
applies when `meta["origin"]` is absent on a notification-born
trace. `stub_text=` replaces the default "N pending … item(s)" stub text
(the provenance check's use). `register(channel, source, entries)` is the
no-wake variant: no stub, no origin.

### GoalsPlugin (`packages/corvidae-cognition/corvidae_cognition/goals.py`)

Keeps the agent's open goals in SQLite and shows them in the window. Entry
point name: `"goals"`; `depends_on = frozenset({"persistence", "memory",
"funnel", "clock"})`. Owns the `goals` table in `sessions.db` (`id`, `text`,
`created_at`, `updated_at`, `retired_at`, `memory_id`), created in
`on_start` when the DB is open and on first use otherwise.

**Tools:** one per verb, so a channel can disable creation alone with
`channels.<id>.tools.disabled`.

| Tool | Parameters | What it does |
|------|------------|--------------|
| `add_goal` | `text: str` | Inserts an open goal and returns `Goal #N added.` Whitespace runs in `text`, newlines included, collapse to one space; empty text returns an error string. |
| `update_goal` | `id: int`, `text: str` | Replaces the text of an open goal and bumps its `updated_at`, returning `Goal #N updated.` An unknown or retired id, or empty text, returns an error string. |
| `retire_goal` | `id: int`, `conclusion: str` | Writes a memory record of the conclusion, then marks the goal retired, returning `Goal #N retired; conclusion recorded as memory #M.` An unknown or retired id, or an empty conclusion, returns an error string, as does a missing channel context. |

**Goals entry:** `before_agent_turn` admits one CONTEXT entry through the
funnel under the source `goals` (budget `funnel.budgets.goals`, falling
back to `funnel.default_budget`; see [configuration.md](configuration.md)).
With open goals it reads:

```
Open goals as of 2026-10-30 14:02:11 UTC (this list replaces any earlier goals list):
#3 Finish the migration notes
#5 Learn what Schuyler wants from the wiki
```

With none, including when every goal is retired, it reads `No open goals as
of … (this replaces any earlier goals list).` A database with no goal rows
at all admits nothing. The timestamp is the latest `updated_at` over all
rows, retired ones included, not the current time, so the entry changes only
when a goal does and the funnel's dedupe drops an unchanged one. The
supersession clause is what lets an older entry still in the window be read
as stale: the latest entry is current.

**Visibility:** goals are agent-wide. Every channel sees every goal,
whatever `memory.channel_groups` says. A goal added or retired on one channel
shows up on the others at their next turn. This is a recorded variance from
the compartmentalization of memory; see finding 10 in
[threat-model.md](threat-model.md).

**Retirement record:** `retire_goal` calls `MemoryPlugin.write_record` first
and updates the goal only once that succeeds, so no goal is retired without
its record. If the record can't be written, the tool returns an error string
and the goal stays open. The record belongs to the retiring channel, has
importance `1.0` (above the retention floor, so it stays retrievable), and
links to the newest real message on that channel as both ends of its message
range. Its summary is:

```
I retired my goal "<goal text>" (set YYYY-MM-DD). <conclusion>
```

The goal row keeps the new record's id in `memory_id`. Because the record is
an ordinary memory, passive retrieval and `search_memory` find it under the
usual channel-scope rules. `write_record` embeds the summary (falling back
to `embedded=0` for backfill) and inserts under the memory plugin's DB lock;
`write_record` shares `_insert_record` (insert, vec row, near-dup merge)
with consolidation, so the record can absorb a near-duplicate memory; the
returned id is the surviving record's.

**Operator CLI:** `corvidae goals --db PATH seed TEXT` inserts an open goal
from a second process, registered under the `corvidae.commands` entry-point
group. `PATH` must be an existing `sessions.db` in WAL mode (the daemon sets
it on startup); the command refuses a missing file or another journal mode,
and an empty `TEXT`. The daemon reads the table each turn, so the goal
appears on every channel at its next turn.

**Heartbeat review.** A beat's turn carries the goals entry like any other
turn. When this plugin is loaded, the heartbeat also asks
`retired_goals(20)` for the newest retirement records (agent-wide, newest
first, each with its age) and appends them to the beat with an instruction
to take a next step on an open goal or retire a finished one, and not to
take up a retired goal again without something new (see HeartbeatPlugin).

**Without this plugin:** the goal tools, the goals entry and the `goals`
table are absent. Goals already in an existing database stay there unread.

### OutcomeLogPlugin (`packages/corvidae-cognition/corvidae_cognition/outcome_log.py`)

Owns the `exchange_log` table — one row per exchange (everything one
trace covers, keyed by its `trace_id`), accumulating the
retrieval/appraisal/outcome profile used to correlate memory retrieval,
appraisal, and critique for calibration. Entry point name:
`"outcome_log"`; depends on `"persistence"`.

Implements the message-lifecycle hooks (`on_message_admitted`,
`on_message_rejected`, `on_message_persisted`) to record the identity row
for every exchange (origin, channel, originating message rowid). The
remaining columns — retrieval score/raw similarity/hit count, appraisal probe score and
stage-2 JSON, critique's provenance snapshot, and merged outcomes — are
filled in by `MemoryPlugin`, `AppraisalPlugin`, and `CritiquePlugin`
through the writer API below, not by `OutcomeLogPlugin` itself.

A message refused by a transport gate (an allowlist refusing a stranger)
leaves no row: both insert paths (`record_exchange` and `upsert_exchange`)
skip any channel the `ChannelRegistry` does not hold, so no writer
(`on_message_rejected`, appraisal's stage-1 persist, the gates) can grow
`exchange_log` or leave a refused sender's channel id behind. With no
registry plugin the guard fails open and records. Rejections on known
channels (the engagement gate) keep their `gate: rejected` row, with a null
`message_rowid`.

It also registers the `exchange_log` scrubber (`redact_exchange_log`)
under the `corvidae.redact` entry-point group. When `corvidae redact`
tombstones a message, the owning exchange's stage-2 `past_claims` text is
tombstoned (claim and query counts kept), and every snapshot in that
channel from the message on is dropped (see "Operator redaction" in
[design.md](design.md)).

The table is created in `on_start` when the DB is already open. At boot it
usually isn't (persistence starts after outcome_log, since `on_start` is
LIFO), so the plugin logs one plain WARNING (`exchange_log table creation
deferred until the first write: persistence DB not open yet`) and creates
the table on the first write. A DDL or migration failure with the DB open
is still logged at WARNING with its traceback.

Writer API (used by other plugins, not hooks):

- `record_exchange(trace_id, channel_id, origin=None, message_rowid=None)`
  — idempotent insert (`INSERT OR IGNORE`).
- `update_exchange(trace_id, **columns)` — guarded update of the
  nullable profile columns only; unknown or immutable column names raise
  `ValueError` (SQL is never built from arbitrary kwargs).
- `upsert_exchange(trace_id, channel_id, origin=None, **columns)` —
  create-or-update in one call: `INSERT OR IGNORE` the identity row, then
  a guarded update applying `**columns`. For gate-time writers (e.g.
  `AppraisalPlugin._persist_stage1`) that run before or race
  `on_message_admitted`/`on_message_persisted`'s own inserts — without it,
  those hooks' `INSERT OR IGNORE` would no-op against a row that doesn't
  exist yet, silently dropping the gate-path columns. Idempotent and
  write-order independent.

### AppraisalPlugin: the outbound vector (`packages/corvidae-cognition/corvidae_cognition/appraisal.py`)

Alongside the inbound stage-1 vector it computes at the gate,
`AppraisalPlugin` computes an outbound vector over the final response text:
`get_or_compute_out(channel, trace_id, text)` is the twin of
`get_or_compute`, with the same surface heuristics and FTS5 probe blend and
the same fail-open probe budget, and no model call. It has its own cache
slot and in-flight future, keyed `(trace_id, "out")`. A user exchange
computes both directions under one trace id, and a shared slot would hand
the outbound caller the inbound vector. The first compute persists the
vector under the `appraisal` envelope's `stage1_out` key, with no origin, so
the upsert can't relabel the exchange's row.

A thin `should_send_response` implementation calls it for every `final`
emission and always returns None, so the outbound vector exists even with
no gate registered. It catches and logs every exception: one raising
implementation in that broadcast would discard every sibling's verdict.
`get_appraisal_out(trace_id)` reads the vector back.

**Stage 2** (`on_agent_response`) is a silent model call over the
finished exchange: valence, stakes, ambiguity, commitment density, novelty,
correction, and `past_claims`. `past_claims` is a list of `{claim,
queries}`: each statement in the reply that presents something as
established before the exchange, with up to 3 short search phrases. A
missing or malformed field reads as no claims; malformed entries are
dropped with a warning; it's capped at 5 claims. The vector is merged into
the `appraisal` envelope's `stage2` key. Then every listener registered with
`add_stage2_listener` is awaited with `channel, trace_id, response_text,
meta, stage2`, outside the appraisal attribution. A failing listener is
logged and doesn't stop the others; a failed stage 2 notifies no one. It's
a push rather than a wait because a consumer that awaited stage 2 from a
queued task would hold a TaskQueue worker the stage-2 task may need.

### CritiquePlugin: the provenance check (`packages/corvidae-cognition/corvidae_cognition/critique.py`)

The provenance check looks for confabulation: a reply that presents
something as already said, agreed or done when no record of it exists. A
model makes both language judgments; the code only fetches evidence:

1. **Detection.** Stage-2 appraisal lists the reply's claims about the past
   in `past_claims`, with a few search phrases for each (see
   [AppraisalPlugin](#appraisalplugin-the-outbound-vector-packagescorvidae-cognitioncorvidae_cognitionappraisalpy)).
   That works in any language and any phrasing, and it costs no extra model
   call because stage 2 already runs on every exchange.
2. **Evidence.** When stage 2 lands, AppraisalPlugin notifies its listeners.
   CritiquePlugin registers one in `on_start`. For an eligible exchange
   (same rule as the stylistic lenses: `user` or `reminder`) with at least
   one claim, it enqueues a silent task. That task searches `message_fts`
   with each claim's phrases (or the claim's own words when it has none):
   every whitespace-separated chunk quoted, then OR-joined. FTS5
   tokenizes the quotes with the index's own tokenizer, so no
   word-splitting or stopword list of ours decides what matches, and query
   syntax in the model's phrases is never parsed. Results are ranked
   by bm25, and up to 5 per claim are kept. Only these rows count:
   - rows from before the exchange's originating row. The exchange's own
     rows, the reply included, are already indexed when the check runs;
     the judge sees them in the window instead, and the reply is no record
     of itself.
   - `message` and `summary` rows. CONTEXT (injected data, including
     earlier critique) and `unsent` rows are never a record of something
     said.
   - rows from channels in the calling channel's memory scope
     (`memory.channel_groups`), or the channel alone without MemoryPlugin.
     The excerpts reach the critic, and its objections re-enter the window,
     so an unscoped search would carry text across channels.
3. **Judgment.** The check runs as a fork of the window, cut just after
   the exchange's final reply, so the judge sees what the agent saw
   when it wrote it: the system prompt, the exchange's own rows (tool
   results included), and the CONTEXT admitted for it. Rows written after
   the reply aren't in it. The instruction appended to that prefix lists
   each claim with its search hits and the CONTEXT snapshot recorded for
   the exchange. A hit longer than 500 characters is shown around its
   matched terms, not from its start, so support deep in a long tool result
   still reaches the judge. The judge is told the search shows near
   matches, not proof of absence, and marks a claim unsupported only when
   nothing supports it, judged by meaning rather than wording. The system prompt and
   every tool result, this exchange's included, are records. The agent's
   own earlier messages show only that something was said: they support a
   claim that it was said, not a claim about a fact. Summary rows are shown
   as `summary`, whatever role compaction stored them under. The reply
   under audit is in the prefix but is no record of itself. The judge is also
   told it can't call tools and isn't replying to the user.
4. **Verdicts.** The fork answers with one verdict per claim, in order:
   `{"verdicts": [{"claim": 1, "reason": "...", "supported": true}, ...]}`.
   The response schema pins position *i* to claim *i*, and the parser
   checks it again (a server may ignore parts of a schema). A reply that
   isn't exactly one well-formed verdict per claim, in order, fails the
   fork: an empty or partial answer can't pass as "all supported". Each
   unsupported verdict becomes an objection quoting the claim from our
   list, with the judge's reason, and reaches the funnel tagged
   `[provenance]`. Stylistic objections queue through `register` and wait for
the channel's next turn. Provenance objections wake the channel with a
stub whose text tells the agent the default is to send nothing: reply
exactly `NO_REPLY`, and reply in the channel only if its earlier reply
could lead the other party to take an incorrect action. The plugin's
`should_send_response` vetoes a `final` on a critique-origin turn whose
stripped text is `NO_REPLY`; the kernel stores it as `unsent` with the
not-sent marker. No other emission or origin is affected. The outcome is
recorded as
   `outcomes.provenance = {claims, candidates, supported, objections}`;
   `objections: 0` means every claim was judged supported.

The check never runs as a standalone call: a judge working from a rebuilt
copy of the conversation can't see the system prompt or the exchange's
tool results, and flags well-grounded facts as made up (#164). When the
fork isn't possible, the check skips and logs `critique provenance check
skipped for trace … on channel …: <reason>`. The reasons:

- forks are not enabled for critique (`agent.forks`);
- no live window;
- the reply was not recorded or not persisted;
- the window was compacted after the reply (the prefix is no longer what
  the agent saw; CritiquePlugin counts compactions in `before_compaction`);
- the reply is not in the window;
- the fork failed, or its reply wasn't exactly one well-formed verdict per
  claim (logged with the traceback).

The fork's availability is checked before the search and again just before
the fork, after the last await. A skip records no outcome. The check also
doesn't run, and logs a warning naming what's missing, when there is no
AppraisalPlugin, no read-only probe on `message_fts`, no originating row
for the exchange, or a search fails. An absence it couldn't look for must
never reach the judge as "nothing matched". A failed stage 2 (malformed
model output) leaves that exchange unchecked; appraisal logs the failure.
The candidate count and excerpt length are constants, not tunables: a
lever that hid evidence would make the judge object to everything.
The probe connection opens lazily, on the first provenance check, not at
startup, and a missing `message_fts` (or an unopenable DB) is retried at
most once every `PROBE_RETRY_SECONDS` (30 s, `corvidae_cognition/probe.py`),
so a table that appears after startup is picked up without a restart. Each
failed attempt logs a warning naming the path and the reason; recovery logs
one INFO line; `on_stop` closes the connection. Appraisal's `memory_fts`
probe uses the same helper (`ReadOnlyProbe`).
`critique.provenance.enabled` is the one switch. A
`critique.fork_prompt.provenance` override must ask for the verdicts format
above; one written for an objections list fails parsing, and every run
skips.

### GatePlugin (`packages/corvidae-cognition/corvidae_cognition/gates.py`)

The agent's right to silence (PRD §3.7). Entry point name: `"gates"`;
`depends_on` is empty, and it reads `appraisal` and `outcome_log`
fail-soft. Both gates *pull* their vector through AppraisalPlugin's compute
API rather than relying on hook order: apluggy runs a broadcast's
implementations concurrently, and the pull API's per-key dedup makes the
compute run once whoever asks first.

**Origin.** Read from propagated state, never inferred: `meta["origin"]`
when a producer stamped one (the scheduler stamps `reminder`,
`HeartbeatPlugin` stamps `heartbeat`, the funnel stamps `critique`), else `user` for a trace the
plugin saw in `on_message_admitted` (a bounded LRU of 1024), else the origin
`exchange_log` already holds for the trace (so a user exchange whose trace
fell out of the LRU during a long tool cycle is still `user`), else `task`.
Tool cycles carry the item's meta forward, so an exchange keeps its origin
through its tool calls. The decide gate never passes its derived origin to
`upsert_exchange`, so it can't relabel a row; only the engagement gate,
which fires only for user messages, writes `origin="user"`.

**The engagement gate** (`should_process_message`) pulls the inbound
stage-1 vector and compares its salience with `gate.engagement.threshold`
(default 0.2). It ships in shadow mode: until `gate.engagement.enforce` is
set it rejects nothing, and records `outcomes.engagement = {salience,
threshold, would_reject, enforced}` so the offline calibration corpus
accumulates from day one. It never samples below threshold live: a
rejected message is something a user notices, so the threshold is fitted
from the recorded verdicts instead. The record is written with
`upsert_exchange` from a fire-and-forget task, because the gate fires before
the admission row exists and must not wait on the shared write connection.
No AppraisalPlugin, or a compute that fails, returns None and records
nothing.

**The decide gate** (`should_send_response`):

- `final`: pulls the outbound vector. A user-origin final is never
  withheld: a direct reply to a person is not swallowed by a guessed
  threshold. A final of any other origin is withheld when its salience is
  below `gate.send.min_salience` (default 0.15) and `gate.send.enforce`
  (default true) is set: self-initiated traffic that adds nothing stays
  silent. A missing vector (no AppraisalPlugin, or the compute failed) fails
  open. Every decision is recorded as `outcomes.send = {emission, origin,
  salience, min_salience, enforced, vetoed, would_veto, reason, advisory}`.
  `reason` is one of `user_origin`, `below_threshold`,
  `at_or_above_threshold` or `no_vector`; on `no_vector` the `salience` key
  is absent (a null in a `json_patch` fragment deletes the key). `advisory`
  carries the input-side salience and the previous exchange's stage-2 vector
  (the current one is unfinished when the gate fires), as context for
  calibration; neither affects the decision, and user-origin finals skip it.
  The record is per trace: if a later item under the same trace reaches a
  final again, its decision overwrites the earlier one key by key.
- `progress` / `thinking`: vetoed when `gate.send.allow_progress` /
  `gate.send.allow_thinking` (both default true) are off, recorded as
  `outcomes.suppressed = {"progress": true}` or `{"thinking": true}`. Core
  never writes the outcome log; the vetoing plugin does.
- `error` and anything else: no opinion.

What the kernel does with a withheld final is in
[design.md](design.md#the-output-gate). On the cognition side, critique still
judges a withheld `user` or `reminder` final, its provenance probe doesn't
count a withheld row as a record of anything said, and `recall_raw` labels
withheld rows `(withheld, not sent)`.

Every tunable resolves through `resolve_tunable` at decision time, so a
`set_settings` override or a hot config reload applies on the next
exchange. Values are type-checked on read: numbers and numeric strings for
thresholds, booleans and the words true/false/yes/no/on/off/1/0 for switches.
Anything else logs a warning naming the key and value and uses the default.
Both gates catch any other error, log it and fail open, because an exception
in either broadcast would drop the message or the reply. The keys are in
[configuration.md](configuration.md#gate--engagement-and-decide-gates), including the ones an
operator should consider adding to `agent.immutable_settings` on channels
with untrusted participants.

### HeartbeatPlugin (`packages/corvidae-cognition/corvidae_cognition/heartbeat.py`)

A turn the agent starts on its own when nothing else is happening (PRD
§3.7 Heartbeat, principle 13). Entry point name: `"heartbeat"`;
`depends_on = {"idle", "registry"}`. It needs `memory` only when awake, so
a config that disables memory still starts while the heartbeat sleeps.

**When it beats.** At `on_start`, if `heartbeat.after` is set, it registers
a system idle watch (`IdleTracker.watch_system`) with that period. The
tracker calls it once no channel has had inbound or in-flight work for
`after` seconds, never while one does. The beat is itself inbound work on
`internal:heartbeat`, so the system is busy until the beat's turn
settles, and the next beat is due `after` seconds after that. Traffic on
any channel cancels the pending beat. The result is at most one beat per
`after` seconds, never two at once, and none during a conversation. Under
traffic that never stops it never beats; that is accepted until #214.

**What a beat is.** The plugin reads `MemoryPlugin.recent("internal:heartbeat", 20)`,
re-checks that the system is still quiet (a message may have landed
during the read), and calls `on_notify` on `internal:heartbeat` with
`source="heartbeat"` and `meta={"origin": "heartbeat"}`. The text is the
heartbeat prompt (`DEFAULT_HEARTBEAT_PROMPT`; documented copy in
`prompts/heartbeat.md`) followed by the memories, framed as the agent's own
summaries and as data, or `(none)`. When the goals plugin is loaded, the
text then carries `GOALS_INSTRUCTION` and the conclusions of up to 20
retired goals (`GoalsPlugin.retired_goals`), framed as data, or `(none)`;
without it, no goals text appears. A failed memory or retired-goals read
skips the beat, which is retried after one period. The turn then runs the
ordinary notification pipeline with the agent's normal tools. The `heartbeat`
origin keeps the existing conventions: critique doesn't review it, the
decide gate treats it as non-user, and `exchange_log` records it.

**What it can reach.** No transport owns `internal:heartbeat`, so a beat's
reply goes to `sessions.db` and the jsonl log, never to a person. Reaching
a person needs the send or note tools (PRD §3.5.9–10, #234, #235), which
don't exist yet; when they land, the beat uses them like any other turn,
with no allowlist. The prompt tells the model that most beats warrant
nothing, and to say so in one line and stop.

**What it reviews.** Memory is compartmented, so the beat sees only the
channels that share a `memory.channel_groups` group with
`internal:heartbeat`. With no group, every beat's list is `(none)`.
Records consolidated from the beat channel itself are left out of the
review, so beats don't review earlier beats. One edge: when near-duplicate
merging leaves a beat-owned record as the survivor of a conversation
memory, that memory drops out of later reviews.

**Sleep.** The heartbeat is asleep whenever `heartbeat.after` is absent
(the block deleted, `null` or empty). It reads the key once at `on_init`,
so sleep is set in `agent.yaml` and survives every restart;
`plugins.disabled: [heartbeat]` also works. A config reload doesn't change
it.

**Failures.** A memory or retired-goals read that raises is logged as a WARNING naming the
channel and the exception, and that beat is skipped; the tracker retries
`after` seconds later. An invalid `heartbeat.after` (zero, negative, non-finite,
boolean, non-numeric) fails startup with a `ValueError`, and an awake
heartbeat without the memory plugin fails startup with a `RuntimeError`
saying to enable memory or remove `heartbeat.after`.

There is no goal review yet: goals (#133) are a separate plugin, and
whichever of the two lands second wires the beat to the goals store.
Distillation and rehearsal are not part of the heartbeat.
