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

**Vector memory is optional.** `sqlite-vec` is a compiled extension and only
`MemoryPlugin` and retention use it, so it lives in the distribution's `vec`
sub-extra. The `cognition` extra pulls it in, so the line above keeps vector
memory. To run without it — funnel, appraisal, critique and goals, with memory
falling back to FTS5 keyword retrieval and one startup warning (PRD §3.7,
Memory 5):

```sh
uv pip install --no-deps -e packages/corvidae-cognition   # no [vec]
```

CI proves that path: the `cognition-fts` job installs cognition without
`sqlite-vec` and runs `tests/cognition` against it (#130).

| Plugin | Entry point | What it does |
|---|---|---|
| `MemoryPlugin` | `memory` | Consolidates conversation (at turn ends, on idle, and at compaction) into durable memories and retrieves them into context; owns `sqlite-vec` |
| `MemoryToolsPlugin` | `memory_tools` | `search_memory`, `search_transcripts` and `recall_raw` tools for the model |
| `FunnelPlugin` | `funnel` | The admission policy for held text: dedupes it against the window, budgets it per source and drops it past a TTL (the kernel's `held` store frames and appends it); gives the agent `ponder`, a wait of at most 5 s for in-flight recall |
| `AppraisalPlugin` | `appraisal` | Appraises each exchange once the agent has answered, in a fork of the window: the reply's claims about the past and the critique vector |
| `CritiquePlugin` | `critique` | Reviews the agent's replies after the fact through the stylistic lenses (predictive, constrained, adversarial; every lens chosen for an exchange runs in one fork of the window, which is skipped with a warning on a miss), the retention lens when a knowledge base root is known (`critique.retention.root`, else markdown_recall's root), the proactivity lens (permission asked for something the agent owns, #1126), the provenance check, and the lenses other plugins contribute through `on_critique_init` |
| `GoalsPlugin` | `goals` | The agent's open goals: three tools to add, update and retire them, and a list of the open ones in every heartbeat beat |
| `TracePlugin` | `trace` | The protocol plugin that owns `on_trace_record` and `resolve_trace_origin`, the hooks the reporting plugins fire; warns at startup when nothing records what they report |
| `OutcomeLogPlugin` | `outcome_log` | Appends every trace record to `<session db stem>.outcomes.jsonl` beside the session database |
| `GatePlugin` | `gates` | The decide gate: withholds every emission of an origin the operator made quiet (with the agent's `speak_up` hatch), and switches progress and thinking emissions off |
| `SilencePlugin` | `silence` | Lets the agent end any turn without sending anything: a final that contains `NO_REPLY` goes unsent, and each decline is recorded in the outcome log. It adds the instruction to the system prompt of chat channels, and with `silence.mode: tool` or `both` an `end_turn` tool |
| `HeartbeatPlugin` | `heartbeat` | Starts a turn on `internal:heartbeat` when the whole system has been quiet for `heartbeat.after` seconds; the turn is `heartbeat.prompt` plus the sections plugins add ahead of it through `on_heartbeat_collect_prompts` (memory's recent records, a capped goals list and pick rule), and acts only if something warrants it. Asleep unless configured |
| `SkillsPlugin` | `skills` | Scans a skills directory for `SKILL.md` files; contributes an index (name + description) to the system prompt via `transform_system_prompt` at conversation start and compaction; provides a `use_skill` tool that loads a skill's full text on demand |

Entry-point names are the same as when these plugins lived in the kernel, so
`plugins.disabled` lists in existing `agent.yaml` files keep working. Without
the extra, those names match nothing and disable nothing, which is harmless.

## What the plugins rely on

They are ordinary plugins: they import the kernel modules mapped in
[plugin-guide.md](plugin-guide.md#what-the-kernel-exports-to-plugins), and
the kernel never imports them (`tests/test_kernel_boundary.py`).
Their loggers keep their `corvidae.*` names (`corvidae.memory`,
`corvidae.appraisal`, ...), so log configuration and log lines are the same
as before the split.

They own these tables in `sessions.db`: `memory`, `memory_fts`,
`memory_vec`, `memory_prompt`, `embedding_meta`,
`consolidation_watermark`, `retention_meta`, `retrieval_log`,
`retrieval_candidate` and `goals`. `exchange_log` is no longer created or
written (see "OutcomeLogPlugin"); rows already in an old database stay as history.

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
name: `"memory"`; `depends_on = {"persistence", "llm", "clock", "idle", "trace", "registry"}`. Absorbs the former
`DreamPlugin` embryo (removed): compaction becomes memory formation instead
of file-based fact scraping.

#### Consolidation (write path)

Three triggers, one watermarked code path:

- **End of a turn** (`on_agent_response`, #636) — when a final reply
  leaves enough new conversation in the window, consolidation forks it
  then, instead of waiting for compaction or quiet. "Enough" is
  `memory.consolidate_after_tokens` (default 4000) window tokens, counted
  with the window's own estimator (`message_tokens`) over the window
  messages above the watermark and at or below `range_end`, skipping
  `CONTEXT` (admitted memories aren't new conversation), `SUMMARY` and
  `UNSENT` rows. `range_end` is the newest `_db_id` in the window when the
  hook runs and is fixed there, so a message persisted after the reply
  isn't in the range. The fork renders the window with a cut just past
  `range_end` (#622; idle forks the same way), so a message that reaches
  the window between the hook and the fork's render isn't narrated by this
  consolidation and then again by the next. The hook runs before the reply is sent, so it awaits
  nothing: it checks, fixes `range_end` and spawns the task; the task
  reads the watermark and does the counting. This trigger only ever forks.
  With forks off for `consolidation` it does nothing and logs nothing, and
  a reply produced while tool calls are still outstanding isn't a turn
  boundary, so it doesn't fire then either. A turn range that isn't wholly
  in the live window (the watermark lags behind the window, say after a
  failed compaction consolidation) is skipped with one DEBUG line and
  never goes to catch-up: idle or compaction consolidates it, and a
  warning would repeat every turn until one of them does. A fork that
  keeps failing backs the trigger off (#620): after three consecutive
  failed turn forks on a channel (the fork raised, or its reply was
  unusable; each is its own skip warning), one more warning names the
  channel, the message-id range and the remedy, and the turn trigger stops
  forking for that channel. It resumes once any consolidation moves the
  channel's watermark (a later success, or an idle or compaction run) or
  after `memory.idle_consolidate_after` seconds; a retry that fails then
  backs off again with one more warning. Idle and compaction are never
  backed off. The count and the limit live in memory only (the limit is the
  module constant `TURN_FORK_FAILURE_LIMIT`), so a restart starts over.
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
  quiet never reaches this trigger; it is consolidated at the end of a
  turn (above) when forks are on for `consolidation`, and through
  compaction as dialog leaves the window.

All three spawn a tracked plugin-owned `asyncio.create_task` (never the
TaskQueue — every TaskQueue completion wakes the main model). The turn and
idle triggers share one in-flight set: while a channel has a turn or idle
task running (or an idle firing is still setting one up), neither trigger
starts another for it, so at most one turn-or-idle consolidation runs per
channel. Compaction is never skipped, so it doesn't consult the set;
instead it waits (#621), so neither reply is thrown away, and it waits off
the response path (#628). `before_compaction` copies the window as it
stands (`snapshot_window`, before `replace_with_summary` drops the prefix)
and spawns a task that forks that copy, so the turn doesn't wait for the
fork. That task first waits for an in-flight turn or idle task, then
forks only what lies past the watermark it left; the `on_compaction`
task waits the same way, then awaits the fork task's record (or makes the
view call on a miss). A failed or cancelled in-flight task just means
compaction consolidates the whole range, and cancelling compaction's fork
task doesn't cancel the task it waits on. The copy holds the whole window
(each message copied) until the fork is done. An idle firing that has claimed the channel but not yet spawned its
task has nothing to wait for; that overlap, like any other, is settled by
the watermark compare-and-set below. The task
reads the per-channel **consolidation watermark**, fetches the working
range, produces the entries (first-person narrative with epistemic framing,
JSON out) — from a fork of the live window when the range is in it (see
[Forks](design.md#forks)), else from the agent's rebuilt view of the range
sent as the `consolidation` stage (catch-up, below). An idle or turn range
that is in the window but can't be forked (forks off, fork failed, unusable
reply) is skipped instead, with a warning naming what retries it: a turn
range "the next turn trigger retries", an idle range "a new message, the turn
trigger or compaction retries" (the next idle firing does not, #342), and a
range skipped because forks were reloaded off names only the triggers that can
still run — "a new message or compaction" for an idle range, "idle or
compaction" for a turn range. The watermark stays put. An idle range reaches
catch-up once it has left the window; a turn range is left to idle or
compaction, as above. Both
calls are constrained by a
`response_format` JSON schema, like appraisal and critique, so a model can't
answer the transcript instead of writing a reply. The reply is one object:

```json
{"entries": [{"narrative": "...", "topic_tags": ["..."]}],
 "participants": ["..."], "importance": 0.7}
```

`entries` and `importance` are required by the schema; each entry needs a
`narrative`, and `topic_tags` and `participants` are optional. The model
writes one entry per distinct stretch or topic, in order, each readable on
its own, so a record is about one thing rather than the centroid of
several. The parser is more forgiving than the schema, because providers
that ignore `response_format` still get the prompt's JSON instruction:

- An entry whose `narrative` is missing, not a string or blank is dropped,
  as is one that isn't an object or whose `topic_tags` can't be read as a
  list. If none are left, or `entries` isn't a list, the reply is unusable
  (below).
- A reply with no `entries` key but a `summary` (the old shape) is read as
  one entry, with its top-level `topic_tags`.
- A lone string where a list belongs (`"participants": "kestrel"`) reads as
  a one-item list. Any other non-list `participants` reads as no
  participants, with a warning naming the key, and the entries are kept
  (#655); a non-list legacy `topic_tags` makes the reply unusable, and the
  error names the key.
- `importance` is a number, or a string `float()` accepts (`"0.7"`), if it
  is finite and within [0, 1]. A bool, a non-number, `nan`, or a value
  outside [0, 1] (a rating on another scale, such as `7` or `"1e3"`) counts
  as absent, so the rubric rates the range instead (#655). A key that is
  present but unusable logs one WARNING naming the channel, the message-id
  range, the rejected value (truncated) and the accepted `[0, 1]`, because an
  operator otherwise sees the extra model call with no reason for it (#929).
  An absent key is the normal case and logs nothing.

The task then scores an importance prior through the pluggable
`MemoryPlugin.importance_prior`, once for the whole range, passing the
reply's `importance` as `hint` (default: cheap-model `RubricPrior`,
which returns the hint clamped to [0, 1] with no model call, and
otherwise asks the model, schema-constrained to `{"importance": number}`,
0.5 on failure, and that call carries `mode="rubric"` in its `usage_log` row so
it is distinguishable from a pre-stamp row (#929); appraisal no longer
supplies a prior, #1189), embeds every entry's
narrative in one call on the `embedding` stage's profile (`llm.profiles.embedding` by default), and
commits one memory row per entry (each with its vec row and any near-duplicate
merge) together with the watermark advance in one compare-and-set
section. At insert the rows share the message range, `created_at`, importance,
participants and [record stamp](#record-stamps); each has its own
narrative (stored as `summary`), `topic_tags` and embedding. A lost
compare-and-set stores none of them. (A row that then merges with an older
near-duplicate takes the wider range, the later date and the higher
importance, as any merge does.)

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
  corrupt one; **`search_transcripts`** drops it from its hits the same way.
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
passes (demotion, re-promotion, retrieval-log pruning when
`memory.retention.retrieval_log_max_age_days` is set (#654), vec backfill;
a pruning failure is logged and leaves backfill to run): each batch commits whole
or rolls back whole, and the backfill's embeddings are computed before the
transaction opens. The prune takes `RETENTION_PRUNE_BATCH` (1000) rows per
transaction and releases the lock between them (#682), so enabling it against
a large existing log cannot push another memory write past its busy timeout;
`retrieval_log(ts)` is indexed so each batch's cutoff lookup does not scan. Each pass selects its candidates before its transaction,
so every write re-checks the selection: demotion skips a row redacted,
already demoted or retrieved since the SELECT; re-promotion never
re-indexes a row superseded or redacted meanwhile; the backfill writes no
vec row for one demoted, redacted or superseded while its embedding was
computed (#272, #309). A demotion whose `memory_vec` delete fails still
demotes the row and logs a WARNING naming the record and that its vec row was
left behind. No later pass clears that row — the backfill selects `indexed = 1`
and the row is now `indexed = 0`, and the demotion pass only flips rows it
finds at `indexed = 1` — but retrieval never returns it either: both KNN
queries join `m.indexed = 1`. What the stale row costs is the vec-rows ≤
indexed invariant, and one dead slot in each over-fetch, until it is deleted
by hand with `DELETE FROM memory_vec WHERE memory_id = ?` in a sqlite3 that has
sqlite-vec loaded (`memory_vec` is a vec0 virtual table). The log
lines count the rows actually written. `_db_lock`
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
≥ `memory.dup_threshold` anywhere in the agent's memory, on whatever channel
it lives (PRD §3.7 Memory 3, #168). Retrieval spans the whole memory, so
every channel sees the survivor. Entries of the same consolidation never
merge into each other (they are excluded by id); each can merge with older
records. The old record is superseded (`superseded_by`, `indexed=0`,
vector deleted); the survivor takes the summed retrieval count, the
higher importance, and the later of the two records' dates. Message
ranges fold only within one channel: `message_log` ids are global, and
`recall_raw` and `/redact` read a record's range on its own channel, so on
a cross-channel merge each record keeps its own range and the superseded
record still leads to its channel's source dialog. The survivor of a
cross-channel merge, or of a merge with a record that already was one, is
marked `merged_foreign = 1` so window exclusion never hides it (#627). Overlapping triggers on the same
range discard the loser's record — no duplicates. Embedding failure —
including a wrong-width vector, which is refused before the write
transaction (vec0 rejects it on insert, and the resulting
`OperationalError` would roll back the whole consolidation: record,
vec row and watermark, stalling the channel on every later trigger) —
stores the entry with `embedded=0` for later backfill. The embedding call covers all
of a consolidation's entries at once. If that batch fails, it is retried
as two halves, recursively, with a warning, so a server that caps the
batch size still embeds every entry (#655); if a single-entry request
fails or returns the wrong number of vectors (the encoder is failing, not
capping), every entry is stored `embedded=0`. A
wrong-width vector costs only its own entry. The narrative text
is canonical and vectors are a rebuildable cache. All consolidation LLM
calls carry `stage="consolidation"` attribution, along with the `trigger`
that started the run (`"compaction"`, `"idle"` or `"turn"`; `"eval"` from
`scripts/eval_memory.py`) — set once at
`_consolidate_range`'s entry, so it rides into the `on_llm_response`
attribution snapshot for that call, into `MetricsPlugin`'s emitted tags,
the `usage_log.trigger` column, and the `memory consolidated` completion
log line, which carries `memory_ids`, `entries` (their count), `embedded`
(how many were stored with a vector), the message range and `trigger`. A
`memory consolidation start` record opens the stage once the watermark check
says there is work, and `memory consolidation timing` closes it in a
`finally`, carrying the same channel, `trigger`, `range_end` and `trace_id`
plus `duration_ms`, how many segments were committed (`records`; the
`memory consolidated` lines carry `entries`, the stored-entry count), and an `outcome`
(`ok`, `superseded`, `failed` or `cancelled`) — so one start always has exactly one close
and the duration is the gap between them. The close is gated on the start, not
on a list of outcomes: a failure in `_wait_for_inflight`, `_ensure_schema` or
`_get_watermark` happens before the stage opens, so it logs no close, and on
the compaction trigger `duration_ms` is the work after the in-flight wait, not
the wait (#469 review). The `memory consolidated` lines between them are per
stored record, so a multi-segment catch-up logs several, each carrying its
segment's `range_end` — the end of that segment, which differs from the call's
on the start and timing records, so `trace_id` is what ties a line to the pair
(#469 review). A range that is already consolidated logs neither: routine skip,
no stage.
`_consolidate_range` also opens a lease scope (#363, #985): its catch-up views
and every rubric call (after a fork record too) run under a per-run key,
`memory:consolidation:<channel id>:<id>` (the run's trace id when attribution holds one,
otherwise an id minted once per run), so concurrent runs are distinct queues
in the lease table, snapshot and `lease.*` metrics, while the consolidation
forks run under the channel id. The key lands in `usage_log.lease_key`.

### Corrupt `exchange_log` rows

These are history rows from before the outcome log moved to a file. A corrupt `appraisal` value (unparseable JSON, invalid UTF-8, or JSON that
is not an envelope object) can only come from an out-of-band write.
`corvidae redact`'s scrubber skips such a row with one WARNING naming the
trace id and channel (#311) and finishes over the others. A row the scrubber skips keeps its
`past_claims`, and redact still exits 0, so check the log for the warning
after a redact run on a database you suspect is damaged.

#### Retrieval (read path)

Why retrieved memories come back vague, and what we propose to do about
it, is written up in [memory-retrieval-fidelity.md](memory-retrieval-fidelity.md).

Retrieval runs in `before_compose` (#1176), under the channel's check allowance,
when the window's last non-CONTEXT row is a plain user MESSAGE (CONTEXT rows
below it, such as a timestamp stamped by `before_agent_turn`, are looked
through; notification-triggered turns skip retrieval entirely). Memory returns
its lines as a `Contribution`: a retrieval that finishes within the allowance
is in the same turn's prompt, and a slower one is held and admitted at the
channel's next turn. The query is every
unanswered user message in the window (`unanswered_user_messages`), joined
by blank lines: a coalesced burst (#398) is retrieved for as a whole, and
one message is just itself. Ahead of them, as its own paragraph, goes the
agent's previous reply (#665): the window's last assistant MESSAGE row with
text. Tool-call-only rows have no text to refer back to, and an UNSENT reply
never reached the person, so neither counts. A follow-up such as "thanks,
that worked" or "what about the other one?" has no referent of its own, so
it is retrieved for together with the reply it answers. The reply is used
whole. If labelled self-contained probes show a long reply diluting a short
question, trimming or capping it is the first adjustment.
`memory.retrieval.query_context: none` restores the user-message-only query.
`retrieval_query(messages, query_context)` builds the text, and
`scripts/eval_memory.py` builds its queries with the same function. The query is embedded and matched against `memory_vec`
(sqlite-vec exact KNN, cosine distance); candidates are filtered to
`indexed=1 AND redacted=0`. There is no per-channel scope: records from
every channel are candidates (PRD principle 14, #168), and
[threat-model.md](threat-model.md) finding 15 records what that costs.

**Hybrid candidates (#666).** Dense embeddings handle rare tokens poorly
(people's and project names, hostnames, ticket numbers), so on the vector
path the KNN is joined by an FTS5 search over `memory_fts` with the same
query and filters, `k × 4` rows in bm25 order. Its query is built as in
[Keyword search](#keyword-search-and-the-fts5-indexes-1186): 3-character
pieces ranked by bm25, no word list, so a common word buys little rank but
every list is full. The two are ranked together
by reciprocal rank fusion. Each candidate earns `1 / (60 + rank)` from each
of two lists:

- every candidate with a measured cosine, ordered by decayed cosine, so
  recency still counts;
- every FTS match, in bm25 order.

The sum orders the pool, with the decayed score breaking ties. An FTS hit
the KNN didn't return gets its real cosine from its stored vector (one
`vec_distance_cosine` point lookup), so it is ranked in the first list as
well. A hit with no vector (a record not yet embedded) carries the 0.6 proxy,
earns its key from the FTS list alone, and ranks among the unmeasured only —
behind every measured candidate — with the weak band. A query with no lexical match ranks exactly as
the KNN alone would. If the FTS side fails, a warning is logged and the
KNN rows are ranked alone. `memory.retrieval.candidates: vector` turns
fusion off. The FTS-degraded path below is unchanged: it has no cosine to
fuse with.

**Window exclusion (#634):** records that only echo the live window are
left out. A record is excluded when it belongs to the current channel
*and* its `msg_id_start` is at or past the oldest `message_log` id in the
window (`min(window_db_ids(conv.messages))`), that is, its whole range is
still in view. Both conditions matter: `message_log` ids are global, so an
id test alone would hide other channels' recent records too. A record that
straddles the window's oldest id is still admitted, so the part of it that
compaction dropped stays recallable; a window with no ids excludes nothing.
A record marked `merged_foreign` (it superseded another channel's record
through a near-duplicate merge, directly or through a chain of merges) is
never excluded, since the window doesn't hold that channel's content
(#627). A same-channel merge needs no mark: it folds the old record's range
in, so an older record pulls the survivor's start out of the window. Rows
from before the column read 0 (no backfill; exclusion only applies while a
range is still in the window).
The exclusion is a `retrieve(..., exclude=(channel_id, min_msg_id))`
argument applied before the cut to k: the degraded FTS5 path filters it in
`WHERE` ahead of its `LIMIT`, and the vector path selects it as a flag (#407)
over a KNN that fetches `k × 4` plus a `COUNT(*)` of the excluded records,
then sets the flagged rows aside, so in-window records can't take every
slot. That `k` is clamped to sqlite-vec's ceiling of 4096 rows
(`SQLITE_VEC_KNN_MAX`), which the extension enforces by refusing the query
outright: a large `k` or a long window used to come back as an embedding
failure and a fall back to FTS5 on every turn. Now the clamp keeps the
vector path, logs one WARNING per process naming the limit, the `k`, the
over-fetch and the channel's in-window count, and reads only the nearest
4096 rows (#637). The hybrid FTS side does the same over its own `k × 4`
rows. A KNN failure for any other reason logs `retrieval KNN query failed;
degrading to FTS5`, separate from the embedding warning, so the operator is
not sent to the embed server for a fault in the query — and a degraded query
ranks the FTS rows by the decayed score, never through the hybrid fusion. Only `before_compose` passes it; `scripts/eval_memory.py`
and `search_memory` see everything. It is not a scope: other channels'
records, and this channel's once their rows leave the window, are
retrieved normally. Known limit: the bound assumes the window is a
contiguous id suffix of the channel's log. Reload loads rows by
timestamp, so a row with a newer id but an older timestamp breaks it:
rows a database already holds from the nanobot importers (removed, #638),
a corrupt row skipped on reload, or a backward wall-clock step. Such a
record is hidden on its own channel until the window moves past it.

Scores are `cosine_similarity × exp(-age_days / memory.half_life_days)`, where
`age_days` runs from `created_at`, or from `max(created_at,
last_retrieved_at)` under `memory.retrieval.recency_from: last_used`.
On the hybrid path candidates rank measured-first: every measured candidate ahead
of every unmeasured one, each group in its own fused order (#856, #892). Off it,
by score. They are
banded strong, moderate or weak. Weak records are dropped, with no fallback:
a query with nothing above weak admits nothing (#409). What the bands compare
is `memory.retrieval.band_on`:

- `relative` (the default) bands on how far a record's raw cosine stands out
  from the query's cosine distribution over the whole memory: `z = (cosine −
  mean) / stdev`, where the mean and stdev come from one aggregate
  `vec_distance_cosine` pass over every indexed, unredacted record with a
  stored vector (about the cost of the KNN scan, with no `k` limit).
  Thresholds are z values: strong ≥ 4.5, moderate ≥ 3.0. Cosine has no stable
  scale across queries (#409 measured per-query medians from 0.28 to 0.45 on
  one encoder), so a fixed cosine threshold can't tell "this answers the
  question" from "this query's register runs high"; a z can. Recency plays
  no part in the band, so an old, apt memory isn't presented as a guess. With
  fewer than 50 embedded records, or no spread, the band falls back to raw
  similarity on the cosine defaults (0.75 / 0.60), logged once per process;
  a failed spread scan falls back the same way, with a warning per query.
  The minimum is set by arithmetic: a z over N records can't exceed √(N − 1),
  so a smaller memory could never reach the strong threshold.
- `similarity` bands raw cosine (strong ≥ 0.75, moderate ≥ 0.60).
- `score` bands the decayed score on the same thresholds, so with the default
  half-life nothing older than about 9 days can be strong.

A candidate only the hybrid FTS side found bands on its measured cosine like
any other; one with no stored vector (not yet embedded) bands weak, so it
isn't admitted passively, though `search_memory` still finds it. The
keyword-only floor #666 introduced is gone: one shared word no longer carries
a record into the window. On the FTS5-degraded path there is no similarity to
gate on, so the top `k` keyword hits are admitted, every one marked
`[marginal]`, and `retrieval_log.degraded_to_fts` flags it. Each
`retrieval_candidate` row carries the record's `z` (NULL when none was
computed), which is what the z thresholds are refitted from; the starting
values rest on six probes. The `(age)` shown always runs from
`created_at`. Winners enter the window as `(age, memory <id>) summary`
lines, or `[marginal] (age, memory <id>) summary` for a moderate record,
through the kernel's held-text store — never by direct CONTEXT append: memory's
`before_compose` returns them as a `Contribution` (source `memory`) and the
store frames and appends them. The id is
the record's `memory.id`, the one `recall_raw` takes; its tool description
says where ids appear, so a model shown a memory can replay its dialog
(#632). Each line carries the key `memory <id>)`, so with the funnel loaded a
record already in the window is not admitted again when its age or band reads
differently now (#624); the closing paren keeps `memory 12` from matching
`memory 123`. Memory writes its records from `on_held_admitted`, once the
store has said what the turn admitted (a retrieval whose lines were held
because a tool call was outstanding writes them at the later turn). Admitted
memories get their access statistics
bumped (`retrieval_count`, `last_retrieved_at` — the write half of
reconsolidation), and every retrieval writes a `retrieval_log` row
(the top-ranked candidate's decayed score and raw similarity,
hit/admitted counts, degradation flag, banding path) and one
`retrieval_candidate` row per logged candidate (#407, below).

`retrieval_log.banding_path` says which banding path that retrieval actually
took (#931): one of the configured `band_on` values (`score`, `similarity`,
`relative`), `relative_fallback:<reason>` when `relative` could not measure the
spread (`few_records`, `no_spread` or `scan_failed`; banding then used raw
similarity), or `fts5` when the retrieval degraded to keyword search. Without
it a z-gated retrieval and a fallback retrieval look identical once the
fallback's once-per-process warning has gone quiet. Rows written before the
column read `NULL`. `eval_memory --snapshot` records the same value per query
and counts them in `banding_paths`, so an eval run shows how many queries the
fallback banded. A retrieval that ran before the schema existed is bucketed as
`no_db` in `banding_paths` and writes no log row, because the insert needs the
database.

**The candidate log (#407).** `retrieval_log` says what a retrieval
returned; `retrieval_candidate` says what it considered and what happened
to each record. Every passive retrieval writes one row per logged
candidate, in the same commit as its `retrieval_log` row. The rows hold ids
and numbers, never text:

| column | meaning |
|---|---|
| `retrieval_id` | `retrieval_log.id` |
| `memory_id` | `memory.id` |
| `similarity` | cosine similarity, or the fixed FTS proxy when it can't be measured (a degraded retrieval, or an FTS hit with no vector) |
| `score` | the recency-decayed score (`similarity × decay`); the rank key on the vector-only and degraded paths, the tie-break under fusion |
| `band` | `strong`, `moderate` or `weak` (above; a hit with no measured cosine is `weak`, and every degraded hit `moderate`) |
| `rank` | 1-based position among the not-excluded candidates: measured-first on the hybrid path, fused order within each group (#856, #892), score otherwise (excluded rows: see below) |
| `outcome` | one of the values below |
| `source` | `vec` (KNN only), `fts` (FTS only, or any degraded candidate) or `both` (#666); NULL on rows from before it |
| `z` | under `band_on: relative`, the z-score the band came from (#409); NULL when none was computed (other modes, a small memory, no spread, a failed spread scan, a hit with no vector, degraded retrieval, rows from before it) |

`(retrieval_id, memory_id)` is the key, and an index on `memory_id` makes
per-record lookups cheap. A candidate's `outcome` is one of:

- `admitted`: the held-text store appended its line to the window.
- `dropped_weak`: banded weak, so it was never offered to the store. Since
  #409 this holds even when every candidate is weak: that retrieval admits
  nothing.
- `dropped_budget`: offered, but neither admitted nor marked a duplicate. The
  usual cause is the `memory` source's token budget, or what was left under
  its window cap (`funnel.window_budgets.memory`), having no room for it; the
  funnel's log line says which limit bound. The same outcome records a line
  the store dropped for another reason: a newer memory contribution replaced
  it while it was held, or `funnel.ingest_ttl` expired it. With no funnel
  loaded, nothing is budgeted and every offered line is admitted.
- `dropped_dup`: offered, but the window's memory CONTEXT already held that
  record (its `memory <id>)` key, #624), whatever its age or band read then.
  A record whose own range is still in this channel's window is
  `excluded_in_window` first and never reaches this test.
- `cut`: scored but ranked below k, so never returned.
- `excluded_in_window`: left out by the window exclusion, but its score
  would have placed it in the top k.

The log keeps the whole scored pool, not just the k returned. On the
vector path that is every not-excluded row of the KNN (which fetches
`k × 4` plus the count of in-window records) and, under fusion, of the FTS
side, so a right record pushed past k, by lower similarity or by recency
decay, shows up as `cut` with its score. The FTS pool is the k rows the query's `LIMIT`
returns, so FTS retrievals never log `cut` rows. The window exclusion no
longer filters in the vector query's `WHERE`; the query selects it as a
flag, so in-window rows come back from the same KNN with their real
similarity and the rows returned are unchanged.

`excluded_in_window` rows carry a would-be rank: one plus the number of
not-excluded candidates ranking at least as high (under fusion, its
fused value is computed from the ranks it would have had in each list).
"At least as high" follows the pool's measured-first order (#705): a record
with a measured cosine counts only measured candidates, and one without a
stored vector counts every measured candidate plus the unmeasured ones at
least as high. It can therefore equal a
returned row's rank. Each excluded record is judged alone against the
ranked pool, not against the other excluded records, so two excluded rows
may share a would-be rank, and an excluded record that would not have made
the top k gets no row.

`retrieval_log.excluded_in_window` counts the in-window records the search
matched (a `merged_foreign` record is not in-window by this rule); it is NULL when no exclusion applied (a window with no ids) and on
rows from before #407. The two paths count differently. On the vector path
it is the in-window rows inside the KNN (and the hybrid FTS side). On the FTS path it is every
in-window record the keyword query matches, found with one extra
`COUNT(*)`: FTS has no scores for rows it dropped, so that path logs the
count but no `excluded_in_window` rows, and the count is not limited to
records that would have made the top k.

Only passive retrieval (`before_compose`) logs; the records are written from `on_held_admitted`. A direct `retrieve()`
or `_retrieve_pool()` call (`scripts/eval_memory.py`) and `search_memory`
write no `retrieval_log` rows. There is
no `filtered` outcome: redacted, superseded and demoted records have no
`memory_vec` row, so the KNN never reaches them. Logging is fail-soft: if
the candidate insert fails, the turn still runs, the `retrieval_log` row
and exchange profile are still written, and a WARNING names the channel and
the `retrieval_log` id. Rows are never pruned. Old retrievals have no
candidate rows.

Why didn't it remember X (here record 123)?

```sql
SELECT l.ts, l.channel_id, c.outcome, c.rank, c.score, c.similarity, c.band
FROM retrieval_candidate c
JOIN retrieval_log l ON l.id = c.retrieval_id
WHERE c.memory_id = 123
ORDER BY l.ts DESC;
```

No row means the record was not logged for that retrieval: outside the
KNN's reach, not indexed, or redacted; in the window and not good enough to
make the top k; or, on an FTS-degraded retrieval, not matched by the
keyword query or past its `LIMIT`. `cut` and `dropped_weak` mean it ranked
but lost on rank or banding (by default the z of its cosine, `retrieval_candidate.z`). `corvidae redact` deletes
rows that reference a redacted memory (below).

**Graceful degradation:** encoder down → FTS5 keyword retrieval over
`memory_fts` (flagged `degraded_to_fts=1`; the FTS query caps each
blank-line-separated paragraph at 20 terms, so every message of a burst
keeps its own 20 and a single message is capped at 20 as before); sqlite-vec unloadable → FTS5
with one clear startup warning; no embedding profile (the `embedding` stage isn't mapped under `llm.stages` and there is no default embedding profile: no `llm.profiles.embedding` and no `llm.defaults.embedding`) → records stored
with `embedded=0`, retrieval FTS5-only; a stored encoder differing from
config → ERROR and embeddings disabled rather than silently mixing
encoders; fewer than 50 embedded records (or no spread) under `band_on:
relative` → bands on raw similarity, logged once. Nothing in the memory path
raises into the turn loop.

#### Measuring retrieval on a real database (`scripts/eval_memory.py --snapshot`)

`scripts/eval_memory.py` has three modes. With no mode flag it scores a
`--retriever` callable against a fixture. `--live` runs the real
`MemoryPlugin` against a scratch database it fills from that fixture.
`--snapshot` (#658) scores a deployment's real records:

```sh
uv run python scripts/eval_memory.py --snapshot /path/to/sessions.db \
    --labels labels.json --config /path/to/agent.yaml \
    --pool-out pool.json [--k N]
```

What it does:

- **Copies the database, never writes it.** The source is opened only
  through a read-only (`mode=ro`) connection and copied with the sqlite
  backup API in one step. The copy is consistent even while the daemon has
  the source open in WAL mode, and it includes committed rows still in the
  `-wal`. Every plugin connection is opened on the copy, because memory's
  schema setup writes on open.
- **Runs the real scorer under the deployment's config.** `--config` is
  loaded the way the daemon loads `agent.yaml`, so the embedding profile
  and the `memory.*` retrieval settings (`half_life_days`, `retrieval.k`,
  `band_on`, `recency_from`, the band thresholds) are the deployment's.
  Each labelled query goes through `_retrieve_pool`, the call `retrieve()`
  and passive retrieval are built on, with no window exclusion. Memory's
  `on_start` is not run, so no retention, backfill or consolidation
  happens, and no chat request is sent. The only model calls are one
  `kind="query"` embedding per labelled query, on the vector path.
- **Reports Recall@k and MRR** for each query and as means over the scored
  queries. Both are computed on the top k, so MRR is MRR@k. `--k` defaults
  to `memory.retrieval.k` from `--config` and must be at least 1; the vector
  path fetches `k × 4` candidates, so a different k changes the pool too.
  `--candidates hybrid|vector` overrides `memory.retrieval.candidates`
  (#666), so the fused pool can be scored against the KNN alone; each pool
  candidate records its `source`. `--now EPOCH` pins the clock decay is
  measured against (#661), so the same snapshot scores identically on two
  run dates; each pool records the pin it used, or `null` for the wall clock.
- **Names labelled ids it can't reach.** A labelled id is reachable when its
  row exists with `indexed = 1` and `redacted = 0`, the filter both searches
  apply. Anything else is listed by id with its status, checked in this
  order: `absent`, `redacted`, `superseded`, `not_indexed`. Unreachable ids
  are left out of scoring. A query left with no reachable id is not
  scored: its metrics are `null`, stdout says
  `not scored: no reachable relevant id`, and the means say
  `over M of N queries`.
- **Scores negative probes on false admits** (#802). A query labelled
  `"relevant": []` asks for the opposite of recall: it should admit
  nothing, so what the gate does admit is counted as false admits, split by
  the band that showed it — `strong` is shown bare, `moderate` as
  `[marginal]`. Recall@k and MRR stay `null` for it. The gate is the
  daemon's own, `admitted_candidates`, shared with `before_compose`, so
  a threshold fitted from these counts is the one the daemon applies: on
  the degraded path the top k is admitted ungated, otherwise every
  candidate above the weak band. A negative probe whose pool degraded is
  reported apart and left out of the rate, since no gate ran. An empty
  reachable set is *not* a negative probe: a right answer existed and was
  lost, and counting admits there would charge the gate for a labelling
  gap. The pool carries `negative`, per-probe `false_admits`
  (`{"strong": n, "marginal": n}`), and `admitted` on every candidate, so
  admission on positive probes is replayable offline; the summary and each
  subset carry `negative_probes`, `negative_scored`, `false_admit_rate`
  (the share of scored negative probes that admitted anything),
  `mean_false_admits` and `strong_false_admits` (a total, not a mean, since
  it is the count a threshold change moves first).
- **Counts FTS5 fallback.** Each query reports `degraded`, and the summary
  prints `FTS5 fallback on D of N queries` whenever any query fell back.
- **Writes every query's whole scored pool to `--pool-out`**, for #409's
  calibration: every candidate the scorer ranked, not only the top k.

The labels file lists queries and the snapshot's integer `memory.id`s that
should come back. `text` is a non-empty string; `relevant` is a list of
integers (it may be empty). `description` and `note` are free text the
script ignores, and so are unknown keys. A malformed file is rejected
before the copy, with the key, the value and what was expected.

Two optional keys support comparing query variants (#665):

- `context` is the agent reply the probe followed. The query is built the
  way the daemon builds it, with `retrieval_query`: under
  `memory.retrieval.query_context: previous_reply` the context goes first,
  as its own paragraph, and under `none` it is left out.
  `--query-context previous_reply|none` overrides the config, so variant A
  and the user-message-only baseline can be scored against one snapshot.
  The query actually used is written to the pool as `query`.
- `subset` is a non-empty tag such as `deictic` or `self-contained`. The
  summary reports `queries`, `scored` and the two means for each tag, with
  untagged probes grouped as `(untagged)`, and prints them when any probe
  is tagged.

```json
{
  "description": "optional",
  "queries": [
    {"text": "what did we decide about the battery?", "relevant": [412, 415],
     "note": "optional"},
    {"text": "and the other one?", "context": "The porch ESP32 was the one dropping off wifi.",
     "subset": "deictic", "relevant": [398]}
  ]
}
```

The pool file is written once, after the last query:

```json
{
  "run_at": 1762700000.0,
  "now": 1762700000.0,
  "k": 8,
  "settings": {"half_life_days": 7, "recency_from": "created",
               "band_on": "relative", "band_strong": 4.5,
               "band_moderate": 3.0, "k": 8,
               "encoder": "nomic-embed-text", "dimensions": 768},
  "queries": [
    {
      "index": 0,
      "text": "what did we decide about the battery?",
      "context": null,
      "subset": null,
      "query": "what did we decide about the battery?",
      "relevant": [412, 415],
      "unscorable": [{"id": 415, "status": "superseded"}],
      "degraded": false,
      "negative": false,
      "false_admits": null,
      "recall_at_k": 1.0,
      "mrr": 0.5,
      "candidates": [
        {"id": 398, "rank": 1, "similarity": 0.81, "score": 0.77, "band": "strong",
         "z": 2.4, "source": "both", "admitted": true},
        {"id": 412, "rank": 2, "similarity": 0.79, "score": 0.74, "band": "moderate",
         "z": 1.1, "source": "vec", "admitted": true}
      ]
    }
  ],
  "summary": {"queries": 1, "scored": 1, "degraded": 0, "all_degraded": false,
              "mean_recall_at_k": 1.0, "mean_mrr": 0.5,
              "query_context": "previous_reply", "candidates": "hybrid",
              "negative_probes": 0, "negative_scored": 0,
              "false_admit_rate": null, "mean_false_admits": null,
              "strong_false_admits": 0,
              "subsets": {"(untagged)": {"queries": 1, "scored": 1,
                                         "mean_recall_at_k": 1.0, "mean_mrr": 0.5,
                                         "negative_probes": 0, "negative_scored": 0,
                                         "false_admit_rate": null,
                                         "mean_false_admits": null,
                                         "strong_false_admits": 0}}}
}
```

`candidates` is the ranked pool, best score first; the entries with
`rank <= k` are `retrieve()`'s top k, and `admitted` is the daemon's own
gate applied to them (`admitted_candidates`, #802). A candidate outside the
top k was never offered to the gate, so it is not admitted. The gate drops
weak-band entries (all of them, so an all-weak pool admits nothing, #409) and
keeps the top k on a degraded pool; the held-text store then admits the survivors,
with no window exclusion. A negative probe carries its `false_admits` counts
and `null` metrics. It holds ids and scores but
no memory text beyond the labels' own `text`, `context` and `query`; look the
ids up in the source. `z` is the statistic a `band_on: relative` band came
from — how far this record's similarity stands out from the query's spread
over the whole memory — so the z thresholds can be fitted offline instead of
reading `retrieval_candidate.z` from live traffic (#704). It is `null` when no
spread was computed: a memory under `RELATIVE_MIN_RECORDS` (50) embedded
records, or a run banded on `similarity`/`score`. `run_at` is the wall-clock
time the run started. The means are `null` when no query was scored.

Operator notes:

- **The chat profile's key must be set.** The config still needs a chat
  profile (the daemon refuses one without), and the daemon's client setup
  expands every profile's `api_key`, so its `${VAR}` must be set in the
  environment even though no chat request is made. If it isn't, the run
  stops with `Config error: llm.providers.<name>.api_key references ${VAR}, which
  is not set … — check <config>`.
- **Scores drift with the run date unless `--now` pins it.** Decay runs
  against the clock at query time, not a time taken from the snapshot, so
  scoring the same snapshot a month later multiplies every score by the
  same factor: the order, Recall@k and MRR are unchanged, but absolute
  scores are lower, and so are bands under `band_on: score` (bands under
  `relative` and `similarity` don't use decay). Pass `--now <epoch>` to
  pin it; two runs with the same pin produce identical scores and bands
  whatever day they ran, which is what makes pool files comparable. The
  file records both the pin (`now`) and the settings that scored it
  (`settings`: half-life, `recency_from`, `band_on` and its thresholds, k,
  and the encoder identity and dimensions), so a comparison never has to
  assume the config behind a pool.
- **A run where every query degraded exits non-zero.** An FTS-only pool is
  not a vector measurement — every similarity is FTS5's constant 0.6 — so
  after writing the file (`summary.all_degraded` marks it, and the scores are
  still worth reading) the run prints the cause and exits 1: a missing
  `llm.profiles.embedding` profile, a snapshot whose `embedding_meta` names a different
  encoder, `sqlite-vec` not loading, or an embedding server that failed every
  call. A run where only some queries degraded still exits 0, and says how
  many on stdout (#662).
- **An unreachable embedding server turns every query into FTS5.** Each
  query's embedding call fails after the client's retries (slow), logs a
  `retrieval embedding failed; degrading to FTS5` warning with a traceback
  (noisy), and falls back to keyword search, where every candidate's
  similarity is the constant 0.6. Such a pool says nothing about cosine
  similarity, so check the summary's `FTS5 fallback` line before using a
  pool file for calibration. An encoder in `--config` that differs from the
  one the snapshot was built with does the same, after memory's encoder
  mismatch ERROR.
- **FTS-only installs log a sqlite-vec warning.** With an embedding
  profile configured but without sqlite-vec, memory logs `sqlite-vec extension failed to load` with a traceback, and
  every query reports `degraded`. That is expected.
- **The copy lives in the system temp directory** (in a 0700 directory)
  and is deleted when the run ends, so a large `sessions.db` needs that
  much free space there. A SIGKILL mid-run leaves the copy behind; it is a
  full `sessions.db`, so delete it.
- Opening a WAL-mode source read-only can leave an empty `-wal` and a
  32 KiB `-shm` index beside it when the daemon isn't running; the main
  file's bytes are unchanged. If those files don't already exist, the
  source's directory must be writable so SQLite can create the `-shm`.
- `--pool-out` may not name the source, its `-wal` or `-shm`, the labels
  file or the config. The pool file is overwritten if it exists, and is
  not written at all if a query raises partway through.

#### Record stamps

**Catch-up.** A range that has left the live window is written from the
agent's rebuilt view (PRD §3.8), not from a prompt that describes the
conversation. `build_catchup_segments` (shared with
`harness/consolidate_replay.py`) takes each non-summary row's *summary in
force* (the newest decodable summary dated before the row), groups rows
under the same summary into runs, and rebuilds each run as that summary
plus the rows reload would load under it. A run is cut into segments that
each fit the channel's `max_context_tokens` (a tool cycle goes whole into one segment; a call whose result arrives after a later user or assistant row is no cycle, so a segment may end between them and show the call as still running, #1046), so a watermark that lagged
across several compactions or a long unsummarized history yields several consolidations and every range row reaches some view. Each
view is trimmed oldest-first to the budget, put under the channel's
`reasoning_history`, headed by the channel's system prompt (identity and
register, from `Agent.view_inputs`, which also resolves the channel after a
restart) and ends with `memory.consolidation_fork_prompt` plus the entry size
sentence and the range sentence. It goes on the `consolidation` stage's profile with no tools and the reply
schema. Each segment is its own consolidation (one or more entries) and its own watermark advance; a
failure leaves earlier segments stored and retries the rest. A segment
with no dialog that is followed by another folds its id span into the next
segment's records (no record or commit of its own); a no-dialog last
segment advances the watermark without a record. Because the
instruction runs under the agent's own prompt, the default tells the agent
the record is its private memory, not a message to anyone on the channel.
`memory.consolidation_prompt` is removed: a config that still sets it logs a
startup warning naming `memory.consolidation_fork_prompt`.

**Entry size target** (plugin-defined, #686). `memory.entry_target_tokens`
(default 256) feeds a sentence appended to the consolidation instruction,
default or override, asking for about `target` tokens per entry (256 by
default, stated as tokens, not words, so it reads the same in any language)
and consecutive entries for a longer topic. It sits
after the instruction and before the range sentence. An entry whose
narrative exceeds the target by the kernel's tiktoken `count_tokens` (not
the serving model's tokenizer) is stored in full, with no truncation and no
schema limit; after the write commits a warning names the channel, memory
id, tokens, target, mode and prompt hash (fork and standalone entries).
Whether the target holds in practice is unmeasured (#885).

Every record says how it was made (#299, #633, PRD §3.7 Memory 2), in four
nullable `memory` columns:

- `prompt_hash`: sha256 of the UTF-8 instruction text the call used. The
  text itself is stored once per hash in `memory_prompt`, however many
  records share it. An empty `memory.consolidation_fork_prompt` is still a
  prompt: it is stored as a blank line plus the entry size sentence.
- `mode`: `fork`, `standalone` (the view call on the `consolidation` stage's profile),
  `raw` (a catch-up range stored as its transcript after both view-call
  replies were unusable, #626; the prompt hash, model and trigger are the
  failed call's) or `direct`.
- `model`: the model the call was sent with, after any
  `llm.profiles.<name>.extra_body.model` override, so a record says what actually
  ran rather than what `llm.profiles.<name>.model` names. `Agent.fork` always sets
  `ForkResult.model` from `LLMClient.request_model`, so it is NULL only for a
  row written without a model (a pre-stamp row, or a `ForkResult` built
  without one).
- `trigger`: what started the consolidation, the value `_consolidate_range`
  was called with: `compaction`, `idle`, `turn`, or `eval`
  (`scripts/eval_memory.py`); or `reconsolidate`, which `corvidae
  reconsolidate` stamps on the records it writes (with `mode = standalone`,
  so PRD §3.7 Memory 2's `fork | standalone | raw` holds; see
  [Reconsolidation](#reconsolidation-corvidae-reconsolidate)).
  Catch-up is a path, not a trigger: a compaction or idle trigger that falls
  through to the view call stores its own trigger, and its
  `mode = standalone` already marks the view. `_commit` sets it, so every
  consolidation path gets it. NULL for `direct` records and for rows
  written before the column existed (#633).

What each path stamps:

| Path | `mode` | Instruction text | `model` |
|---|---|---|---|
| Fork of the live window (compaction, idle or turn) | `fork` | `memory.consolidation_fork_prompt` + size sentence | `ForkResult.model` |
| Catch-up: the rebuilt view on the `consolidation` stage's profile | `standalone` | `memory.consolidation_fork_prompt` + size sentence | `LLMClient.request_model` of that profile's client |
| `corvidae reconsolidate`: the rebuilt view, offline, on a copy | `standalone` | the instruction the operator chose (default `memory.consolidation_fork_prompt` + size sentence); `trigger = reconsolidate` | `LLMClient.request_model` of the consolidation profile's client |
| Raw-transcript fallback after two unusable replies | `raw` | `memory.consolidation_fork_prompt` + size sentence | as above (the call was sent; only its reply was unusable) |
| `write_record` (goal retirement) | `direct` | none: `prompt_hash` NULL, no `memory_prompt` row | NULL (`trigger` NULL too) |

Catch-up records written before the view was introduced carry the removed
`memory.consolidation_prompt` as their stamp; they are not migrated, so the
records that prompt wrote stay findable by its hash. The stamped instruction is `memory.consolidation_fork_prompt` plus the
entry size sentence (`memory.entry_target_tokens`, a static value), so the
hash covers both: changing the target changes the hash, and an override's
hash is not sha256 of its config text. A fork's or view's instruction also
carries a code-generated sentence naming where
the range starts, which can quote the first new message. That sentence is
left out of the stamp: `memory_prompt` is outside `corvidae redact`'s
cascade, so storing conversation text there would put it where redaction
can't reach ([redaction.md](redaction.md)). `memory_prompt` holds configured
instructions plus the static size sentence only. The stamp is the consolidation instruction, not the
agent's system prompt and window that frame a fork.

A `direct` record was written by no consolidation call, so it has nothing
to stamp but its mode; it lists apart from the unknown rows.

Rows written before stamping have all the stamp columns NULL, meaning
"unknown"; rows written between #299 and #633 read `trigger` NULL.
Opening an older database adds the columns in place
(`add_column_if_missing`) without rewriting any row; nothing backfills
them.

**Finding records by stamp.** There is no command for this; query
`sessions.db` directly (`sqlite3 sessions.db`). One row per stamp, with the
record count, models, channels and date range (`created_at` is when the
source conversation happened):

```sql
SELECT
  COALESCE(mode, 'unknown')               AS mode,
  substr(prompt_hash, 1, 12)              AS stamp,
  COUNT(*)                                AS records,
  GROUP_CONCAT(DISTINCT model)            AS models,
  GROUP_CONCAT(DISTINCT channel_id)       AS channels,
  datetime(MIN(created_at), 'unixepoch')  AS first,
  datetime(MAX(created_at), 'unixepoch')  AS last
FROM memory
GROUP BY prompt_hash, mode
ORDER BY mode IS NULL, mode, records DESC;
```

`mode` reads `unknown` for pre-stamp rows; `stamp` is empty for those and
for `direct` rows. Adding `WHERE superseded_by IS NULL AND redacted = 0`
excludes superseded and redacted records; adding `AND indexed = 1` counts
only the retrievable ones. To read a stamp's instruction text, and the
records it wrote:

```sql
SELECT hash, text FROM memory_prompt WHERE hash LIKE 'abcdef123456%';

SELECT id, channel_id, msg_id_start, msg_id_end
FROM memory WHERE prompt_hash LIKE 'abcdef123456%';
```

#### Schema

`memory` (records, with `indexed`, `superseded_by`, `redacted`, and
`embedded` columns, the stamp columns `prompt_hash`, `mode`,
`model` and `trigger`, and `merged_foreign`, set on a near-dup survivor that
holds another channel's record, #627), `memory_prompt` (`hash`, `text`: each distinct instruction text
once), `consolidation_watermark`, `embedding_meta`,
`memory_fts` (external-content FTS5, trigram tokenizer, with insert/update sync triggers; no
delete trigger — memory rows are never deleted), `memory_vec` (vec0,
created only when the extension loads), `retrieval_log`,
`retrieval_candidate` (one row per logged retrieval candidate; see
"Retrieval (read path)"), `retention_meta`
(one-row table persisting `last_run` for the retention job interval), and
`message_fts` (regular FTS5 table, trigram tokenizer, over `message_log` content, maintained by
`message_log_ai`/`message_log_au` triggers; queried and cleared by `corvidae
redact` — not by `search_memory`, which queries `memory_fts` and `memory_vec`). All on the persistence connection in
`sessions.db`; `message_log` stays append-only except for the
operator-only redaction tombstone.

The epistemic-calibration prompt fragment (`prompts/memory_calibration.md`)
is documented but NOT auto-injected — hedging vocabulary is persona, not
architecture.

### FunnelPlugin

`funnel.py` — the admission policy for held text. Entry point name:
`"funnel"`. The kernel's `held` store (`corvidae/held.py`) holds text for a
channel's next turn, frames it as data and appends it, and the funnel decides
*what* of the held text a turn admits. It implements
`transform_held_admission`, which the store calls once per admitting turn with
every due batch. For each batch, in the order offered:

1. **TTL.** With `funnel.ingest_ttl` set, a batch with no origin (not a
   stub batch from `hold_and_wake`) older than the TTL on the kernel clock is
   dropped whole.
2. **Dedupe** against CONTEXT already in the window. By default an entry
   is a duplicate when its exact text appears in any CONTEXT row. A batch
   may carry `keys`, one per entry (memory's are `memory <id>)`); then an
   entry is a duplicate when its key appears in a CONTEXT row framed for
   the same source, so a stable identity dedupes even when the rendered
   line changes, and another source quoting the line suppresses nothing
   (#624).
3. **Budget** per source (`funnel.default_budget`, default 512 tokens;
   `funnel.budgets.<source>` overrides), in given order, skipping any
   entry that would overrun what is left of the budget, so one large
   entry never blocks smaller ones after it; drops logged rather than
   silent. An optional window cap (`funnel.window_budgets.<source>`,
   unset by default) further limits the budget to what is left after that
   source's CONTEXT already in the window, frames included, so admission
   can't crowd the window between compactions. Compaction frees the rows
   it drops; rows in the retained tail still count. A drop the cap caused
   is logged at info with the channel, the tokens held, the cap and the
   key. shortcut: new entries are charged without their frame, so the
   window can pass the cap by one frame; charge it if caps under ~200
   tokens matter.

Entries admitted from an earlier batch of the same source in the same call
count as already in the window, for dedupe and the cap. An entry too large to
fit the source's whole budget or cap is dropped with a warning naming the key
to raise. A budget skip on a contribution (a batch offered at one turn only) is
a drop; on a pushed batch (`hold`, `hold_and_wake`, a late check) the entry is
left unlisted, so the store keeps it for a later turn.

The kernel, not the funnel, does the rest: it frames every admitted batch in
the data-not-instructions wrapper with a source label (so disabling the funnel
removes none of that), appends one CONTEXT message at the tail through
`persist_message` with the rowid attached, cuts an entry over
`agent.held_entry_max_chars`, and never places text between a tool call and its
result. Stale CONTEXT retires by aging past the compaction boundary.

Callers are every source that contributes or holds text: memory retrieval and
the recall sources (through `before_compose`), critique's findings and any plugin that calls
`held.hold` or `held.hold_and_wake`. The funnel also gives the agent the
`ponder` tool. The held store is in memory by design: text still held at
shutdown is dropped, with a WARNING. The reference below has the details.

## Plugin reference

### MemoryPlugin (`packages/corvidae-cognition/corvidae_cognition/memory.py`)

Autobiographical memory: consolidates dialog (at turn ends, on idle, and at compaction) into first-person
memory records (SQLite + sqlite-vec + FTS5) and retrieves them into the
prompt as a `before_compose` contribution that the held-text store frames and appends. Entry point name: `"memory"`;
`depends_on = {"persistence", "llm", "clock", "idle", "trace", "registry"}`. Absorbs the removed `DreamPlugin`.

Implements seven hooks and two kernel timers:

- `on_start` — creates the memory schema on the persistence connection
  (retried lazily on first use; on_start ordering is LIFO across plugins),
  runs the retention check, then registers the two clock-driven pieces
  below.
- `before_compaction` — with `agent.forks` on for `consolidation`, copies
  the window and spawns a background task that forks the memory record
  from the prefix about to be dropped (when every dialog row of the range
  is in it), and stashes the task for the one below to await (#628); the
  hook itself awaits nothing.
- `on_compaction` — spawns a tracked background task consolidating the
  `compacted_ids` range above the per-channel watermark, using the stashed
  fork record when there is one for that range and watermark, else catch-up
  (after the drop the range has left the window).
- `before_compose` — embeds the inbound user text, preceded by the
  agent's previous reply (`memory.retrieval.query_context`), fuses the KNN
  with an FTS5 search (`memory.retrieval.candidates`), leaves out this
  channel's records still wholly in its window, scores candidates
  (`similarity × exp(-age_days / half_life)`, with age from `created_at` or,
  under `memory.retrieval.recency_from: last_used`, from the last
  retrieval), bands them (strong/moderate/weak; by default on the z of the
  record's cosine against the whole memory, `memory.retrieval.band_on`),
  drops weak ones (on the FTS5-degraded path, where every hit is moderate,
  it admits the top k), and returns the rest as a `Contribution` keyed on
  `memory <id>)`. An empty `Contribution` goes back when every candidate is
  weak, so the store still announces the outcome. It stashes the pool and
  selection under `(channel, trace_id)`.
- `on_held_admitted` — for `source == "memory"` and a trace it stashed a
  selection for, bumps access stats for the admitted records and writes the
  `retrieval_log` row, one `retrieval_candidate` row per logged candidate and
  the retrieval profile on the trace (`on_trace_record`). Any other source's
  or trace's event is ignored.
- `on_agent_response` — the turn trigger: with `agent.forks` on for
  `consolidation`, no tool calls pending and no turn or idle consolidation
  already in flight for the channel, fixes `range_end` at the window's
  newest rowid and spawns a task that forks `(watermark, range_end]` once
  its new window tokens reach `memory.consolidate_after_tokens`. Awaits
  nothing, since it runs before the reply is sent.
- `on_stop` — cancels the idle watch and retention timers, then in-flight
  background tasks.

Two kernel registrations replace the old `on_idle` hook:

- **Idle watch** (`watch_channels`, period `memory.idle_consolidate_after`, which must be a positive finite number)
  — consolidates the un-consolidated tail of the quiet channel, forking
  the live window when the range is wholly in it; if that fork misses
  (forks off, failed, unusable reply) the run is skipped with a warning, the
  watermark stays and the next period retries. A range not wholly in a live
  window (no registered channel, trimmed rows) goes to catch-up.
  One run per channel at a time, counting a turn-trigger task: a firing
  while one is in flight is skipped.
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

Changes to `memory.idle_consolidate_after`,
`memory.consolidate_after_tokens`, `memory.entry_target_tokens` and
`memory.retention.*` take effect at restart.

The importance prior is pluggable (`MemoryPlugin.importance_prior`,
default `RubricPrior`). `score(messages, msg_id_range=None, channel=None, *,
hint=None)` takes the consolidation reply's own rating as the keyword-only
`hint`; a replacement prior must accept it. Memory no longer writes the
`valence` column (#1189): new rows hold NULL, and older rows keep the mean
stage-2 valence they were written with. Configuration is documented in
[configuration.md](configuration.md) (`memory.*`, `llm.profiles.embedding`).

The cognition distribution registers the `retrieval_candidate` scrubber
(`corvidae_cognition.memory:redact_retrieval_candidates`) under the
`corvidae.redact` entry-point group. The rows hold no
text, but a row for a redacted memory still records that it was retrieved,
so `corvidae redact` deletes every `retrieval_candidate` row whose memory
is already flagged redacted or whose range contains a redacted message id,
including rows written by retrievals on other channels. `--dry-run` reports
the count without deleting. The CLI prints it as `<N> retrieval_candidate
rows`.

`recent(limit)` is the one read API for other plugins: the newest `limit`
records (indexed, not redacted) of the whole memory, every channel's, as
`{summary, channel_id, age_seconds}` dicts, newest first, with ages from the
kernel clock. Nothing is left out by owner, so a caller sees its own records
too. Memory's own beat section uses it (below). It raises `RuntimeError` while the memory DB
is not open.

**Heartbeat contribution (#1079).** This plugin implements
`on_heartbeat_collect_prompts` (`tryfirst`, so its section precedes goals'
`trylast` one) and returns the newest `memory.heartbeat_limit` (default 5; `0`
adds none) memories through `recent`, framed as the agent's own summaries and
as data, oldest first. It keeps its own only-what's-new state (#716): all of
them on the first beat, then only those not yet sent, and no section when
none are new. Unlike goals' state it is not reset by compaction of the beat
channel, since memories are narrative a summary carries. A failed read raises
and the beat goes out without the section. `heartbeat.memory_limit` still
works for one release, logs a WARNING naming `memory.heartbeat_limit`, and
fails startup when set beside it. Without a heartbeat the hook is never
called.

**Without this plugin:** no memory records form and no retrieval happens;
compaction remains plain lossy summarization. A heartbeat still beats, with
no memory section.

A malformed consolidation reply is retried once; if the retry is also
unusable, the range is stored as its raw transcript rather than being
dropped (the LLM call has already been paid for). The transcript is split
on message lines (a message longer than a piece is cut by characters) into
records that each fit `memory.entry_target_tokens` as a retrieved memory
line (default 256, read at startup; memory reads no `funnel` key since #1176,
so these pieces are smaller than the old default of 512 tokens; each piece leaves room for the longest
`[marginal] (age, memory <id>) ` prefix, a five-digit day count and a
twelve-digit id), so a raw record can actually be admitted (#626). The pieces are
records of one consolidation (shared range, importance and stamp) stamped
`mode = raw`. A target too small for any memory line keeps the transcript
as one record, with a warning naming `memory.entry_target_tokens`. The target
alone sizes a raw piece, so if `funnel.budgets.memory` is set below it the funnel
drops a raw piece, as too large to ever fit, when it is offered.

### MemoryToolsPlugin (`packages/corvidae-cognition/corvidae_cognition/memory_tools.py`)

Registers the `search_memory`, `search_transcripts` and `recall_raw` agent
tools — the active memory surface ("remember harder"). Entry point
name: `"memory_tools"`; `depends_on = frozenset({"memory"})`. Reaches the
DB through `get_dependency(pm, "memory", MemoryPlugin)`.

Implements one hook:

- `register_tools` — appends `search_memory`, `search_transcripts` and
  `recall_raw` to `tool_registry`.

**Whole memory:** all three tools span every channel, as passive
retrieval does (PRD principle 14, #168); none takes the calling channel
into account, so none declares `_ctx`. They reach further than passive
retrieval in *status* (demoted, superseded, and redacted-tombstone records),
and they don't apply passive retrieval's window exclusion (#634).
`search_memory`'s optional `channel=` argument narrows to one channel's
records by equality and is never treated as a pattern; an id nothing was
written on simply matches nothing. `recall_raw` replays a record's raw range
from the record's own channel: `message_log` ids are global, so the range
holds other channels' interleaved rows and the `channel_id` predicate keeps
them out.

**Raw transcripts without a record (#648).** Consolidation is lossy, so the
agent can also reach `message_log` directly. `search_transcripts` is a
keyword search over `message_fts` (FTS5 over each row's `$.content`,
trigger-synced) across every channel, in bm25 order; each hit names its
`message_log` id, channel, UTC timestamp and role, with a snippet around the
match. A hit is widened with `recall_raw`'s range form, `channel` +
`start_id` + `end_id`, which replays that channel's rows in the inclusive
id range exactly as the record form does. It covers the rows `recall_raw`
replays: dialogue, compaction summaries, CONTEXT rows (so a hit can be an
injected memory or note line rather than something said) and withheld rows,
labelled. Corrupt rows and unknown types are skipped with the standard
warning. Neither touches access statistics. Redaction needs nothing of its
own: `corvidae redact` rewrites the row, and the `message_log_au` trigger
re-indexes the tombstone, so the original text is neither found nor
replayed. The index holds only `content`, so tool-call arguments are not
searchable, and the trigram search's limits apply (see
[Keyword search](#keyword-search-and-the-fts5-indexes-1186): a term must share
a 3-character run with the text).

**Fencing.** `search_transcripts` and `recall_raw` declare
`returns_untrusted`: their output reaches the model fenced as untrusted
outside content, because replayed rows carry whatever the original senders
wrote. Both `recall_raw` forms are fenced. `search_memory` is not: it
returns the agent's own record summaries.

**Read-only (#1190).** `search_memory` and `search_transcripts` declare
`reads_only = True`, so a fork whose caller allows tool calls may run them
(design.md, "Read-only tool calls"). Inside a fork every result is fenced,
`search_memory`'s included. `recall_raw` does not declare it: its
by-record form bumps access statistics. One known limit: both tools wait
for, and if need be create, memory's schema before their first query, so
before memory's schema is ready they are not strictly read-only. In
practice that window is startup.

**Search by meaning (#667).** With the vector path up, `search_memory`
embeds the query (`kind="query"`) and ranks two lists over the same filters
by reciprocal rank fusion, as passive retrieval does (#666):

- an exact cosine scan of the filtered records that have a vector
  (`memory JOIN memory_vec`, filtered in `WHERE`, `LIMIT 50`). A KNN would
  cut before the filters and could leave a `channel=` or date-filtered
  search with no vector hits;
- the FTS5 keyword search (trigram pieces, see
  [Keyword search](#keyword-search-and-the-fts5-indexes-1186)).

So a paraphrase ("what did we decide about the deploy schedule?") finds a
record that shares no words with it ("agreed to ship on Thursdays"), and a
name still finds the record that names it. Unlike passive retrieval, it
ranks by raw cosine with no recency decay: a deliberate search wants the best
answer, and the age is on every line. A keyword hit the scan didn't reach
gets its cosine from its stored vector. Demoted and superseded records have
no vector (retention and the near-duplicate merge delete it), so with
`include_demoted=True` they are found by keyword only and show `kw` as their
score. If the FTS query fails, a warning is logged and the cosine results
stand alone; if the cosine scan itself fails, the tool falls back to the
keyword search below.

**Keyword fallback.** With no embedding profile, sqlite-vec not loaded, a
stored encoder that differs from the configured one, or the query encoder
failing, `search_memory` logs a warning naming itself and the cause and runs
the keyword search alone. Its first output line is then
`(keyword search only: semantic search unavailable)`, so the agent knows a
paraphrase may have been missed. If that FTS5 query fails as well, it falls
back to a `LIKE` scan over record summaries instead of returning an internal
error. Tool searches write no `retrieval_log` rows.

All three tools retry briefly when the memory schema is still being created by
another task.

**Tools:**

| Tool | Parameters | What it does |
|------|------------|--------------|
| `search_memory` | `query: str`, `channel: str\|None` (default `None`), `tags: list[str]\|None` (default `None`), `after: str\|None` (default `None`), `before: str\|None` (default `None`), `include_demoted: bool` (default `True`), `max_tokens: int` (default `2000`) | Search by meaning and keywords (above). Returns a list `id. score (age)[flags] summary`, led by each memory id (for `recall_raw`), best match first, capped at approximately `max_tokens`. The score is the cosine similarity to the query (two decimals), or `kw` for a keyword hit with no vector. In keyword-only fallback the first line says so, results are in FTS order, and the score is the negated bm25 rank; a query with no term of 3+ characters (`LIKE` path) comes back newest first with score `-0.0000`. `channel` narrows to one channel's records by exact id; a channel with no records matches nothing. `after`/`before` are ISO-8601 (`YYYY-MM-DD` or full timestamp; naive values treated as UTC) compared against `created_at` (`after` ⇒ `≥`, `before` ⇒ `<`); unparseable values return an error string. `include_demoted=False` restricts to `indexed=1` records, hiding demoted, superseded, and redacted-tombstone records (all are `indexed=0`). `tags` filters by `topic_tags` JSON array intersection. **Does not update access stats.** |
| `search_transcripts` | `query: str`, `channel: str\|None` (default `None`), `after: str\|None` (default `None`), `before: str\|None` (default `None`), `max_tokens: int` (default `2000`) | Keyword search over every channel's `message_log` content (above). Returns up to 50 lines `id. [timestamp] channel role: snippet`, best bm25 match first, the match wrapped in `**`, capped at approximately `max_tokens`. `channel` narrows by exact id; `after`/`before` parse as for `search_memory` and compare against the row's `timestamp`. A query with no word tokens, no hits (naming the channel filter when set) and a failing FTS query (logged at WARNING; retry with a simpler query) each return a plain message. **Does not update access stats.** Fenced. |
| `recall_raw` | `memory_id: int\|None`, `channel: str\|None`, `start_id: int\|None`, `end_id: int\|None` (all default `None`), `max_tokens: int` (default `1000`) | Two forms; exactly one per call, or an error naming both forms and what was passed. **By range** (`channel`, `start_id`, `end_id`): replays that channel's rows with `start_id ≤ id ≤ end_id` under a `[channel C: messages S–E]` header; says so when the range holds none; bumps nothing. **By record** (`memory_id`): replays the raw `message_log` dialog for the record's `[msg_id_start, msg_id_end]` range using `WHERE channel_id = record.channel_id AND id BETWEEN ...` — the channel predicate is load-bearing because the `message_log` id sequence is global across all channels. Returns a participants header (`[memory N: messages S–E, participants: {...}]`) followed by verbatim `role: content` lines, truncated at `max_tokens` with an explicit marker. Works on demoted, superseded, and tombstoned records. Reaches any channel's record. **Bumps `retrieval_count` and `last_retrieved_at`** on the record — the bump is what makes the record eligible for re-promotion by the next retention job run. |

**Recall-stat semantics:** `recall_raw` by record bumps the record's access
stats; `search_memory`, `search_transcripts` and `recall_raw` by range do
not (no record is involved in the last two). `search_memory` is a catalog for choosing what
to recall; bumping every listed record would inflate scores for marginal
matches and distort retention scoring. The re-promotion path is:
`search_memory` surfaces a demoted record (flagged `[demoted]`) → agent
calls `recall_raw` → next retention job run finds the score above the
demotion threshold and re-promotes.

**Without this plugin:** the `search_memory`, `search_transcripts` and
`recall_raw` tools are
not available. Memory records are still formed and passively retrieved
by `MemoryPlugin`; only the agent's active access surface is absent.

### Keyword search and the FTS5 indexes (#1186)

Memory, the transcript search, critique's provenance check and markdown recall
all search an FTS5 table, and all four build their query the same way
(`corvidae_cognition/fts.py`). Nothing in the build decides by one language's
words: there is no stopword list, no prefix match and no assumption that words
are separated by spaces.

**The tables.** `memory_fts`, `message_fts` (both in `sessions.db`) and
`chunks_fts` (the markdown recall index file) use the `trigram` tokenizer with
`remove_diacritics 1` (so `cafe` still finds `café`; matching is
case-insensitive). The index holds every 3-character window of the text, so a
query finds a record by any substring of three or more characters, in any
script, spaced or not.

**How a query is built.**

1. The text is split into terms: runs of letters, combining marks and numbers
   (Unicode categories L, M, N), plus the zero-width joiners U+200C/U+200D
   that sit inside Persian, Indic and emoji words. Punctuation, symbols and whitespace separate
   terms, so a term never carries FTS syntax. The per-paragraph cap (the first
   20 terms of each blank-line-separated paragraph) is kept where it was.
2. A term of 3 or more characters is searched as its overlapping 3-character
   pieces (`喜欢吃苹果` gives `喜欢吃`, `欢吃苹`, `吃苹果`), and a term of 4 or more
   is also searched whole. The pieces are quoted phrases joined by `OR`,
   deduplicated case-insensitively, and ranked by bm25, so a record sharing
   rare pieces with the query ranks above one sharing `ing` or `the`. A query
   that *contains* a word finds a record containing it even when neither is
   spaced: `你还记得我喜欢吃什么吗` finds `我喜欢吃苹果`.
3. A `LIKE '%term%'` scan runs only when the query has no term of 3 or more
   characters (a bare `苹果`), because trigram matches nothing shorter. Its
   results come newest first, since `LIKE` has no rank (markdown recall:
   index order).

**What this costs, and its limits.**

- Every FTS list is full. A chatty English query shares pieces such as `ing`
  or `the` with most of the table, so the keyword side returns its whole limit
  and relies on bm25 to put real matches first. They do, but a record sharing
  several mid-frequency pieces without sharing a word can run a close second.
  A query of only common words still runs and returns rows.
- Where that noise lands. Passive retrieval fuses the FTS list with the vector
  list, and a record in both outranks one in only one, so FTS noise can
  displace a record from the top k but cannot be admitted without a cosine
  that bands above weak. On the degraded path (no embedding) the top k by bm25
  are admitted with no gate, marked `[marginal]`. `search_memory`,
  `search_transcripts` and markdown recall return their ranked tail with no
  threshold. Critique's provenance search keeps the top 5 per claim, and
  "no earlier message matched the search" now means no 3-character piece of
  the claim occurs in any earlier message, which is rare for English.
- A query whose words share no 3-character run with the record finds nothing
  by keyword (`苹果好吃吗` against `我喜欢吃苹果`): the vector path covers that.
  The `LIKE` fallback folds ASCII case only and keeps diacritics.
- The index is about three times the size of the old one, and a query costs
  roughly twice a whole-word one (more for a burst of many messages: no cap
  on pieces exists).
- `search_transcripts` shows a snippet of about 64 trigram tokens around the
  match (roughly 64 characters).

**Markdown recall chunking.** Chunk sizes are in UTF-8 bytes, not words, so an
unspaced page is split like any other: a target of 900 bytes, 200 bytes of
overlap, one chunk for a page under 1800 bytes, and sections under 300 bytes
merged into a neighbour. A chunk is a span of the source, so nothing is
inserted (no space, no blank line); the overlap starts on a word start where
the text has spaces and on a character otherwise. A paragraph is split at
sentence ends: `.`, `!` or `?` followed by whitespace, a full-width `。`, `！`
or `？` (no whitespace needed), or a newline, with no condition on the next
letter's case. A sentence longer than twice the target is cut at spaces, or on
characters when there are none. The chunker version changed (`otterwiki-3`),
so the first start after upgrading re-chunks and re-embeds the whole folder
once.

**SQLite 3.45 or newer.** `remove_diacritics` on the trigram tokenizer needs
SQLite 3.45 (January 2024). The container image and uv's Python meet it; a
distribution Python linked against an older system SQLite (Debian bookworm
ships 3.40) does not. Below the floor:

- A new database cannot create the FTS tables. `create_fts_table` logs an
  ERROR naming the table, the SQLite version and the remedy (run on a Python
  linked against SQLite 3.45+) and re-raises. Memory stays without a schema
  (no records, no retrieval, no memory tools) and logs the ERROR at each
  retry; markdown recall's reconcile and search raise to their callers. The
  daemon and the kernel keep running.
- An existing install keeps its old `unicode61` tables, since
  `CREATE ... IF NOT EXISTS` returns before it reads the tokenizer. Searches
  keep working on whole words (a term of 4 or more is searched whole beside
  its pieces for this reason) but unspaced text is not found by a contained
  word, and markdown recall loses its old prefix matching (`runn` no longer
  finds `running`).
- A database whose tables are trigram cannot be written by a Python linked to
  an older SQLite: the `message_log_ai` trigger fails every kernel insert.
  Run every tool against the database with a
  SQLite at or above the floor.

**Switching an existing install by hand.** There is no automatic migration:
an install made before #1186 keeps its `unicode61` tables until the operator
rebuilds them. Nothing is deleted from a content table; only the three indexes
are rebuilt. Every step runs with the daemon **stopped**, and it starts only
after the last one: an external-content table left empty makes the
`memory_au` and `chunks_ad` triggers fail every update or delete of a content
row (`database disk image is malformed`) until it is rebuilt.

Check first that `sqlite3 --version` is 3.45 or newer, and run each file's
steps with `sqlite3 -bail` inside one `BEGIN; ... COMMIT;`, so an old `sqlite3`
that cannot create the table stops before the drop commits and leaves the old
table in place. For each table, in its file (`memory_fts` and `message_fts` in
`sessions.db`, `chunks_fts` in the file at `markdown_recall.db_path`), drop
it, create it with the trigram statement, and repopulate it:

```sql
DROP TABLE memory_fts;
CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    summary, content='memory', content_rowid='id',
    tokenize='trigram remove_diacritics 1'
);
INSERT INTO memory_fts(memory_fts) VALUES('rebuild');

DROP TABLE message_fts;
CREATE VIRTUAL TABLE IF NOT EXISTS message_fts USING fts5(
    content_text, tokenize='trigram remove_diacritics 1'
);
INSERT INTO message_fts(rowid, content_text)
    SELECT id, CASE WHEN json_valid(message)
                    THEN coalesce(json_extract(message, '$.content'), '')
                    ELSE '' END
    FROM message_log;

-- in the markdown recall index file, not sessions.db
DROP TABLE chunks_fts;
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, content='chunks', content_rowid='rowid',
    tokenize='trigram remove_diacritics 1'
);
INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild');
```

The order inside a file is the point: drop, create, then repopulate, all
before the daemon opens the database. The content tables' triggers belong to
the content tables, so dropping an FTS table leaves them in place. The
markdown index is a cache, so deleting the file and letting the next
reconcile rebuild it is an alternative to the `chunks_fts` steps.

### FunnelPlugin (`packages/corvidae-cognition/corvidae_cognition/funnel.py`)

The admission policy for held text. Entry point name: `"funnel"`;
`depends_on = frozenset({"held", "clock"})`, both resolved by type when first
needed (not in `on_init`). The funnel holds nothing and appends nothing: the
kernel's `held` store (`corvidae/held.py`, [design.md](design.md#held-text-and-before_compose))
does, and asks the funnel once per admitting turn, through the hook
`transform_held_admission(channel, conversation, batches)`, what to admit. The
funnel answers with one `BatchPlan(batch, admit, duplicate, drop)` per batch,
in the order offered, by the three steps in the overview above (TTL, dedupe,
budget with window cap). Its config keys are `funnel.default_budget`,
`funnel.budgets.<source>`, `funnel.window_budgets.<source>` and
`funnel.ingest_ttl` ([configuration.md](configuration.md#funnel--admission-policy-for-held-text));
all four are re-read on config reload (`reloadable_keys`).

What it removed with #1176: `admit()`, `admit_detailed()` and `Admission`,
`ingest()` and `ingest_and_wake()`, the `on_funnel_ingest` hook with its
`FunnelSpec`, `Ingest` and `submit` promises, and the per-turn expire, sweep,
drain and collect steps. A plugin that held or admitted text through them now
returns a `Contribution` from `before_compose` or calls `held.hold` and
`held.hold_and_wake` ([plugin-guide.md](plugin-guide.md#contributing-text-for-the-next-turn)).

**Without this plugin** (`plugins.disabled: [funnel]`): the agent has no
`ponder` tool, and the store's seed policy admits every held entry as offered:
no dedupe, no budget, no window cap, no TTL. The frame, the persistence and the
`agent.held_entry_max_chars` cut stay, because the kernel owns them. Nothing
logs a warning for the missing policy, and goals, critique and memory start
without it.

**Origin and the wake stub.** `held.hold_and_wake(channel, origin, source,
entries, stub_text=None, *, stub_text_wins=False)` stamps `origin` into the wake stub's
`annotations["origin"]` (see [Traces](plugin-guide.md#traces)), together with
the kernel's `held_stub` key, so the turn the stub triggers, and any plugin
gating eligibility off that origin, never has to infer it from stub text. The
`'user'|'reminder'|'critique'|'heartbeat'|'task'` taxonomy is a convention
cognition plugins (AppraisalPlugin, CritiquePlugin, OutcomeLogPlugin) share
over this key — not a core concept. Critique judges every main-loop turn
whatever its origin (#1031) and reads only `'critique'`, to keep a correction's
findings from waking another turn. GatePlugin recognizes `'user'` by
subscribing to `on_message_admitted` rather than by a value in `annotations`;
`'task'` is the default OutcomeLogPlugin applies when `annotations["origin"]` is
absent on a notification-born trace. `stub_text=` replaces the default "N
pending … item(s)" stub text (the provenance check's use). `held.hold(channel,
source, entries)` is the no-wake variant: no stub, no origin, admitted at the
channel's next turn whatever its origin.

**`ponder` tool.** The funnel registers one tool, `ponder()`, which takes no
arguments: a bounded wait for recall already in flight. The store keeps the
stragglers of each turn's `before_compose` gather (the implementations that
had not answered within the check allowance, as futures), and the gather runs
before the model call, so when the agent can call a tool the channel's
stragglers are the recall still coming in for it. `ponder` waits on
`held.in_flight(channel_id)`, as it stood when it was called:

- **Nothing in flight: it returns at once.** "Nothing is being recalled for
  this channel", since waiting cannot bring more. If a straggler already
  finished with entries and the next turn has not admitted it yet, it returns
  "Recall already landed".
- **Otherwise it waits** for the first straggler that brought entries ("Recall
  landed after Xs") or until every one has settled without any ("Recall
  finished after Xs with nothing to add"). A straggler's future is `True` only
  when its late answer was a non-empty `Contribution` and the store has
  already held it, so "landed" never precedes the hold; an empty answer, a
  raise and a cancellation all count as a finished recall with nothing. It
  cannot tell "no hits" from a failure, which the recall mixin both resolves
  to `None` (it logs failures).
- **A timeout of 5 s** (`PONDER_TIMEOUT_SECONDS`, a constant, not config) keeps a
  slow source from stalling the agent: "Still waiting on recall". The recalls
  keep running, because the wait uses `asyncio.wait`, which does not cancel what
  it waits on, and are admitted at the first turn after they resolve.
- **Only before-compose stragglers count.** `hold` and `hold_and_wake` arrivals
  neither end nor lengthen the wait: `hold_and_wake` starts a turn of its own
  through its stub, and `hold` text waits for the next turn.
- **Held turn, busy channel.** Like any tool call, a ponder keeps its call id
  pending: the kernel holds the next assistant turn until the result lands,
  and the channel counts as busy, so idle work (heartbeat, consolidation)
  stays off while it runs. That is intended. A message that arrives during a
  ponder still gets a turn and its own gather, as it does during any tool call,
  which the waiting ponder does not follow. A waiting ponder
  also occupies a worker of the shared task queue (see the threat model).
- **The tool admits nothing.** Its result says what happened; the next turn
  admits whatever landed, within the source's budget and the funnel's dedupe
  and TTL.
- At shutdown the kernel clock cancels the implementations still running; their
  futures settle empty, so a waiting ponder returns.

The model sees only the first docstring line (`tool_to_schema` keeps one line),
so that line carries the guidance: pause for a moment (up to 5 seconds) to let
your recollections catch up, and it returns at once if none are still coming in.
Memory's retrieval and the recall sources both run in `before_compose`, so a
retrieval that is slower than the channel's check allowance is the recall `ponder`
waits for; one that answered in time is already in the prompt and `ponder`
finds nothing in flight. A recall source is the plugin-guide's [Writing a recall
source](plugin-guide.md#writing-a-recall-source).

Remove the tool with `tools.disabled: [ponder]`, or per channel with
`channels.<id>.tools.disabled: [ponder]`. `plugins.disabled: [funnel]` removes it
too, and starts cleanly: goals no longer depends on the funnel (#1176).

### Recall sources (`packages/corvidae-cognition/corvidae_cognition/recall.py`)

Not a plugin: a helper for plugins that recall something on each user turn
and contribute it to the turn's prompt (#718, part of #711). It has
no entry point and no config of its own. `RecallSourceMixin` implements
`before_compose` so a source supplies only `search(query, ctx)` and
`render(hit)`; the how-to is in
[plugin-guide.md](plugin-guide.md#writing-a-recall-source).

- **Gate.** A turn is searched for only when the window ends in a user
  MESSAGE with text, looking through trailing CONTEXT rows (a plugin's
  `before_agent_turn` injection, such as a timestamp, sits there before this hook
  runs). The query is `retrieval_query(messages, "previous_reply")` from
  `memory.py`, so a source sees what memory sees. Memory uses the same tail test,
  `tail_is_plain_user_message`.
- **`SearchGate`.** One cooldown and one in-flight slot per channel, on an
  injectable clock. `try_start` records a start and `finish` releases the slot;
  the cooldown runs from the start, so a failed or timed-out search counts.
- **Settings.** `parse_recall_settings(config, key)` reads `cooldown`
  (default 5 s, 0 allowed) and `timeout` (default 10 s) from the source's own
  block and raises `ValueError` for anything else, naming the key, the rejected
  value and the accepted range.
- **Failure.** A search that raises or times out contributes nothing (`None`) and logs at
  WARNING (logger `corvidae.recall`) with the source, the channel and the
  traceback. A gather cancelled at shutdown is not logged.
- **Timing.** The mixin awaits the search inside `before_compose`, so a hit
  that comes back within the channel's check allowance is in the same turn's
  prompt, and a slower one is held and admitted at the channel's next turn
  (#1176). It passes no dedupe keys, so the funnel dedupes it on whole-line
  text.

Nothing here decides relevance (#674) or sizes the result: the funnel's
per-source budget does, when the funnel is loaded. Threat model: the mixin adds
no route for untrusted text, because everything it hands over is framed by the
kernel's held-text store; a source that
reads from outside (an MCP tool, a folder) records its own entry in
[threat-model.md](threat-model.md).

### Reconsolidation (`corvidae reconsolidate`)

`packages/corvidae-cognition/corvidae_cognition/reconsolidate.py`, a
`corvidae.commands` entry point beside `goals`. An operator-only, offline
command that re-consolidates a channel's `message_log` into a **copy** of
`sessions.db`, to measure a prompt or segment-budget variant on real history
(#583, #932). Never a tool; no hook calls it; it needs no daemon (it builds
its model clients from `agent.yaml`).

```
corvidae reconsolidate --db copy.db --snapshot-from sessions.db \
    --system-prompt identity.txt --segment-budget 16000 \
    [--prompt instruction.txt] [--channel irc:#lex ...] \
    [--since 2026-11-01 --until 2026-11-10] [--min-source-tokens 8000] \
    [--config agent.yaml] [--dry-run]
```

- **A copy only.** `--snapshot-from` copies the file with sqlite's backup API
  and adds a `reconsolidation_copy` marker recording the source's path. A
  `--db` without the marker, the file `daemon.session_db` names, or a copy
  sitting at the path it was snapshotted from (an adopted copy, see below) is
  refused. One copy per variant.
- **Segments.** Each channel is re-segmented from id 0 (or from the selected
  records) with `build_catchup_segments` at `--segment-budget` tokens, which
  replaces `max_context_tokens`. The channel's `reasoning_history` is resolved
  from `agent.yaml` as the daemon layers it (exact `channels:` entry,
  transport-wide entries, `agent:`). `--system-prompt` is the final prompt:
  no `transform_system_prompt` hooks and no runtime overrides run offline.
- **Records.** Each segment is one consolidation call; its entries become
  records stamped `mode = standalone`, `trigger = reconsolidate`, with the
  instruction hash and `memory.segment_budget`, sharing the segment's range
  (PRD Memory 2). Valence is left NULL.
- **Supersede, don't delete.** Active `fork` / `standalone` records of the
  channel whose range meets a segment get `superseded_by` and `indexed = 0`,
  and lose their vector rows. `retrieval_count` is summed into the new
  records, `last_retrieved_at` is the latest, `importance` the larger (the
  near-dup merge's rules). A record straddling two segments is credited to the
  first only. Goal (`direct`) records, `raw` records, redacted records and
  records holding another channel's text are left alone.
- **Vectors.** With a `memory_vec` table, sqlite-vec must load (else the run
  refuses before writing). If `agent.yaml` has an embedding profile, new
  entries are embedded and get their vector rows, so retrieval on the copy
  ranks old and new records alike; `embedding_meta` must match the profile
  (else the run refuses before writing). A failed or wrong-width embedding
  leaves those entries `embedded = 0`, with a warning, for the retention
  job's backfill. Without an embedding profile every new entry is `embedded = 0`.
- **The reply is kept.** `memory_reply` holds the whole reply per segment
  (so a later `claims` array can be indexed without another pass), the model,
  the budget, and the instruction and system-prompt hashes; their texts are in
  `memory_prompt` (PRD Memory 7). `corvidae redact` does not reach
  `memory_reply`; redact a copy's source messages in the copy, or delete it.
- **Scope.** `--since` / `--until` (ISO-8601 UTC or epoch seconds, on the
  record's `created_at`) and `--min-source-tokens` (dialog tokens in the
  record's range) select the active records to redo; overlapping ranges join
  into stretches. Records the selection did not name but whose range meets a
  stretch are superseded too, and `--dry-run` lists them first. Selecting by
  `trigger` would miss oversized records written by compaction or idle while
  `consolidate_after_tokens` was high, hence the window and the size.
- **Resumable.** The stretches are fixed at the first start
  (`reconsolidation_plan`). Each segment commits in one transaction, so a
  stopped run keeps what finished, and rerunning the same command skips the
  segments that have a `memory_reply` row. A reply unusable twice leaves its
  segment (old records untouched) and the run exits non-zero listing it.
- **`--dry-run`** writes nothing to the copy and calls no model: per channel,
  the records selected and met, and each segment's rows and estimated tokens.
  With `--snapshot-from` it still makes the copy.

shortcut: no near-dup merge, no halving of a rejected embedding
batch, and no widening of a scoped stretch over a record that reaches past it. A record whose range runs
past a stretch is superseded even though part of its range is not redone;
`--dry-run` shows which. Upgrade if a measurement needs it.

To adopt a variant the operator replaces `sessions.db` with the copy while the
daemon is stopped; the command never does. The marker comes along, and the
adopted file now sits at the path its snapshot came from, so the command
refuses it as `--db` whichever `--config` is passed; to try another variant,
snapshot it again with `--snapshot-from`.

### GoalsPlugin (`packages/corvidae-cognition/corvidae_cognition/goals.py`)

Keeps the agent's open goals in SQLite and shows them to the heartbeat. Entry
point name: `"goals"`; `depends_on = frozenset({"persistence", "memory",
"clock"})`. Owns the `goals` table in `sessions.db` (`id`, `text`,
`created_at`, `updated_at`, `retired_at`, `memory_id`, `status`,
`position`, `origin_channel`), created in `on_start` when the DB is open
and on first use otherwise. An existing table gains the three newer
columns on first use, with `position = id`, so the queue keeps its old
order.

**Tools:** one per verb, so a channel can disable creation alone with
`channels.<id>.tools.disabled`.

| Tool | Parameters | What it does |
|------|------------|--------------|
| `add_goal` | `text: str`, `position: str = "append"` | Inserts an open goal and returns `Goal #N added.` `position` is `prepend` (front of the queue) or `append` (back); anything else returns an error string. The goal records the channel whose turn created it as its origin. Whitespace runs in `text`, newlines included, collapse to one space; empty text returns an error string. |
| `update_goal` | `id: int`, `text: str` | Replaces the text of an open goal and bumps its `updated_at`, returning `Goal #N updated.` An unknown or retired id, or empty text, returns an error string. |
| `retire_goal` | `id: int`, `conclusion: str` | Writes a memory record of the conclusion, then marks the goal retired (`retired_at` and `status='retired'`, one statement), returning `Goal #N retired; conclusion recorded as memory #M.` An unknown or retired id, or an empty conclusion, returns an error string, as does a missing channel context. |
| `set_goal_status` | `id: int`, `status: str` | Sets an open goal's status to one of `open`, `in-progress`, `blocked`, `waiting`, returning `Goal #N status set to '<status>'`. An unknown or retired id, or any other status, returns an error string. Setting the status a goal already has changes nothing and fires no hook. |
| `list_goals` | `include_retired: bool = False` | Lists goals in queue order with their ids and statuses; retired ones only when asked. |

**Queue, states and the state-change hook:** goals form a priority queue
ordered by `position` (then `id`). `add_goal(position="prepend")` puts a goal
at the front; `append`, the default, puts it at the back, and so does
`corvidae goals seed`. (Seeding into a table the daemon has not migrated yet
still works: the row is inserted without a position, with a warning, and
the daemon's migration then orders it by id.) A goal's status is
`open`, `in-progress` (being worked, so a restart resumes it rather than
re-picking), `blocked` (waiting on something external) or `waiting` (done,
awaiting review); retiring sets `retired_at` and `status='retired'`. The beat's goals list lists open goals
front of the queue first and marks every goal that is not plain `open` with its
status, and the goals instruction the beat carries tells the model to resume an
in-progress goal, otherwise take the first goal that is neither blocked nor
waiting. That pick rule is a prompt, not a framework rule, and the heartbeat
still decides whether a beat warrants action at all.

The plugin owns a hook spec (`GoalsSpec`):
`on_goals_state_changed(goal_id, kind, old_status, new_status,
origin_channel, actor_channel, conclusion)`. It fires after a status change
or a retirement commits, outside the database lock, and not for a no-op
status change or an already-retired goal. `kind` is `"status"` or
`"retired"`; `conclusion` is `None` for a status change; `actor_channel` is
`""` when the turn has no channel (a background turn). A subscriber's
exception is logged at WARNING and never undoes the change. Without a
subscriber, state changes are silent. The `goals-notify` package (its own
extra; the container image installs it by default) (#841,
[design.md](design.md#goals-notifications-goals-notify-package)) subscribes:
it sends each change to the goal's `origin_channel` through
`notify_other_channel`, unless the goal has no origin (seeded goals) or the
origin made the change itself (so a goal the heartbeat created stays silent
while the heartbeat is the one changing it).

Goals also answers the planning lens's `resolve_corvidae_planning_goals`
(#1181), a hook planning owns, by returning `open_goals()`: every goal not
retired, in queue order, as `{"id", "age_seconds", "summary", "status"}`.
Goals imports nothing from planning; it lists `corvidae_planning` under
`optional_depends_on`, so with planning absent the implementation is
reported at `DEBUG` and never called
([design.md](design.md#planning-lens-corvidae-planning-package)).

**Goals list (#1091):** open goals reach the agent in the heartbeat's beat
and nowhere else; no turn admits a goals entry to a window. Goals'
`on_heartbeat_collect_prompts` section opens with the list, in full on every
beat: the frontmost open or in-progress goals up to
`goals.heartbeat_limit` (default 5), in queue order. Blocked and waiting
goals stay on the board (`list_goals`) but off the beat; a final line names
any pursuable goals the cap left out. With pursuable goals it reads:

```
Open or in-progress goals as of 2026-10-30 14:02:11 UTC (this list replaces any earlier goals list):
#5 [in-progress] Learn what Schuyler wants from the wiki
#3 Finish the migration notes
(+2 more pursuable - call list_goals)
```

Goals are listed in queue order (`position`, then `id`), the order the
beat's list and its pick rule use. An `in-progress` goal carries its status
in brackets; `blocked` and `waiting` goals are not listed.

With none, including when every goal is retired or only blocked and waiting
remain, it reads `No open goals as of … (this replaces any earlier goals
list).` Blocked, waiting and retired goals are not named in that line (owner
ruling, 2026-11-19): the beat has nothing to do with them, so it does not
even count them. A database with no goal rows
at all adds no list. The timestamp is the latest `updated_at` over all
rows, retired ones included, not the current time, so the text changes only
when a goal does. The supersession clause is what lets an older beat still
in the beat channel's window be read as stale: the latest list is current.
The list is not held text and does not go through the held-text store, so no
funnel budget can drop it and leave a stale copy standing.

**Visibility:** goals are agent-wide. Every channel sees every goal, as it
sees every memory. A goal added or retired on one channel reaches the
heartbeat at its next beat; see finding 10 in
[threat-model.md](threat-model.md).

**Concurrency:** `retire_goal` and `set_goal_status` take one plugin-level
lock from their check through their update, so two racing retirements write
one record (the loser gets the `already retired` error) and racing status
changes report a consistent chain of `old_status` values. The hooks fire
after the lock is released, so a slow subscriber can receive two racing
events out of commit order; a subscriber that tracks a goal's latest status
should compare each event's `old_status` against its own state. The
`goals-notify` subscriber accepts this: its messages can reach the origin
channel out of commit order, and each names `old → new` so a reordering is
visible to the receiving turn. Ordered delivery is the pubsub facility's
(#937). `update_goal` takes
the same lock around its check and update, so a retirement record always
quotes the text the goal had when it was retired (#846). `add_goal` computes
the new goal's `position` inside its INSERT, as `corvidae goals seed` does, so
the seed CLI running beside the daemon cannot tie two positions. The lock is
shared by every goal, and a retirement holds it across the record's embedding
call; retirements are rare, so a per-goal lock is the upgrade if they stop
being. `set_goal_status` writes without the memory plugin's `_db_lock`, unlike
the other goal tools; that is harmless while the memory plugin has its own
write connection, and matters only if it falls back to the shared one.

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
returned id is the surviving record's. The record is stamped mode `direct`
with no prompt hash and no model, since no consolidation call produced it
(see [Record stamps](#record-stamps)).

**Operator CLI:** the `corvidae goals` group runs from a second process,
registered under the `corvidae.commands` entry-point group, and every command
needs `--db`: `corvidae goals --db PATH seed TEXT` inserts an open goal, and
`corvidae goals --db PATH repair-statuses [--apply]` reconciles a row's `status`
with its `retired_at` (#205; reports and changes nothing without `--apply`).
`PATH` must be an existing `sessions.db` in WAL mode (the daemon sets it on
startup); both commands refuse a missing file or another journal mode, and
`seed` refuses an empty `TEXT`. The daemon reads the table each beat, so a
seeded goal reaches the heartbeat at its next beat.

**Heartbeat contribution (#1084).** This plugin implements
`on_heartbeat_collect_prompts` (`trylast`, so its sections end the beat, just before the prompt) and
returns two things: the goals list above (every beat, in full within its
cap), and `GOALS_INSTRUCTION`, the pick rule that tells the model to resume
an in-progress goal or advance the first listed one, to retire a finished
goal, not to take up a retired goal again without something new. Nothing off
the beat is mentioned there at all (owner ruling, 2026-11-19): retired, blocked and
waiting goals do not reach the beat (owner spec, 2026-11-19): a beat acts on
what it can advance, and the board keeps the rest. The instruction goes on
every beat.
`goals.heartbeat_limit` and `goals.heartbeat_instruction` (the
latter resolved like `heartbeat.prompt`) are read here; a bad limit fails this
plugin's startup. `heartbeat.goals_instruction` still works for one
release, logs a WARNING naming the new key, and fails startup when set beside
it. `goals.heartbeat_limit` never had a `heartbeat.*` spelling.

A goal's `retired_at`, not its `status`, is what marks it off the board:
every query and this plugin's list filter on `retired_at`. Retirement writes
both, in one statement (#1119), so the column agrees with the timestamp. Rows
retired before that fix kept the status they had while open; they are harmless
to every reader above, which keys on `retired_at`, and are reconciled by
`corvidae goals --db PATH repair-statuses --apply` (see the Operator CLI note
above; dry-run without the flag).

**Without this plugin:** the goal tools, the beat's goals list and the `goals`
table are absent. Goals already in an existing database stay there unread.

### TracePlugin (`packages/corvidae-cognition/corvidae_cognition/trace.py`)

The protocol plugin for trace records. Entry point name: `"trace"`. It owns
two hooks and implements neither:

- `on_trace_record(trace_id, channel_id, source, origin, record)`, an
  observe hook. A reporter fires it to say one fact about a trace. `source`
  is the reporter's plugin name and `origin` is the trace's origin when the
  reporter holds one, else `None`. `record` is a JSON-shaped dict whose shape
  belongs to the source.
- `resolve_trace_origin(trace_id) -> str | None`, a firstresult hook that
  asks for a trace's origin when the asker no longer holds it.

`source` is a contract: listeners select records by it, so renaming a
reporter breaks them. `appraisal`, `critique`, `gates`, `memory` and
`silence` name `trace` in `depends_on` and fire the hook with their own
plugin name as `source` and `origin=None`. Disabling `trace` makes startup
fail naming a dependent.

It has one behaviour. At start, if `outcome_log` is not loaded, it logs one
WARNING naming the loaded plugins whose `depends_on` names `trace` (their
records are not written) and saying to remove `outcome_log` from
`plugins.disabled`. With no such plugin loaded, as in
`agent.minimal.yaml.example`, it says nothing. It is the first protocol
plugin in the tree (see [plugin-guide.md](plugin-guide.md)).

### OutcomeLogPlugin (`packages/corvidae-cognition/corvidae_cognition/outcome_log.py`)

Records what the reporting plugins say about each trace, in a file. Entry
point name: `"outcome_log"`; depends on `"persistence"`, to find the session
database's path, and on `"trace"`, which owns the hooks it implements. It implements `on_trace_record` and `resolve_trace_origin`
and has no config key.

**The file.** Each record is one JSON line appended to
`<session db stem>.outcomes.jsonl` beside the session database file
(`sessions.db` gives `sessions.outcomes.jsonl`). A line has the keys `ts`,
`trace_id`, `channel_id`, `source`, `origin` and `record`. A trace has many
lines, one per fact, so a reader groups by `trace_id` and takes the last
line per top-level key of a source's records. The old `json_patch()`
merge was recursive, so a reporter that split one key's nested fields
across two records would read differently; none does today. There is no rotation. The record shapes are the
ones the reporters used for `exchange_log.outcomes` (gates: `send`,
`suppressed`, `speak_up`; silence: `silence`, `silence_reason`; critique:
`lens_selection`, `critique`, `provenance`; memory: `retrieval_top_score`,
`retrieval_top_similarity`, `retrieval_hit_count`; appraisal: `stage2` and,
when computed, `entropy`).

Elsewhere on this page, `outcomes.<key>` is shorthand for a top-level key of
a source's records.

**Its own records.** From the message-lifecycle hooks it writes lines with
`source: "outcome_log"`: `{"event": "admitted"}`, `{"event": "rejected"}`
and `{"event": "persisted", "message_rowid": n}`. The origin is `user` for
admitted and rejected, and `annotations.origin or "task"` for persisted. The
first origin seen for a trace wins and its later lines carry it; a
reporter's line written before any origin is known carries `null`.

**Origin index.** `resolve_trace_origin` answers from an in-memory LRU of
65,536 traces (`ORIGIN_INDEX_MAX`), lost on restart. After 65,536 newer
traces an evicted trace resolves to `None`; the gate treats that as `task`.

A message refused by a transport gate (an allowlist refusing a stranger) is
not recorded: a channel the `ChannelRegistry` does not hold is skipped, so
no refused sender's channel id lands in the file. The registry is a declared
dependency, so it is always there. Rejections on known channels (the
`limits` plugin's, say) are recorded, with no `message_rowid`.

**When it cannot write.** An in-memory session database has no file to sit
beside, so nothing is written and one WARNING names `daemon.session_db`.
While the session database is not open (before start, or after shutdown
closed it) each record is lost with a WARNING saying the next one retries.
A failed append logs a WARNING naming the file, trace and source, and that
record is lost.

**`exchange_log` is retired.** The plugin no longer creates or writes the
table, and its writer API (`record_exchange`, `update_exchange`,
`upsert_exchange`) is gone. Rows in an old database stay as history, and
`redact_exchange_log` still scrubs them.

**Redaction.** The plugin registers two scrubbers under the
`corvidae.redact` entry-point group: `exchange_log` for the history rows and
`outcome_log_jsonl` for the file. The file scrubber tombstones appraisal
`stage2.past_claims`, and critique `provenance` and `critique`
`detail[*].claim` and `objection` text, in the records of the trace that
owns each redacted message (the trace with the largest persisted
`message_rowid` at or below the id, on that channel). See "Operator
redaction" in [design.md](design.md) and [redaction.md](redaction.md). The
detail of lenses other plugins add, and the free text of `silence_reason`
and `speak_up` reasons, are not reached. Stop the daemon before redacting:
the scrubber rewrites the file through a temp file and `os.replace`, so a
line appended between its read and the replace is lost.

### AppraisalPlugin (`packages/corvidae-cognition/corvidae_cognition/appraisal.py`)

Stage-1 appraisal is gone (#1185). It scored every inbound message and
every final at the gates from English word lists and a word-match probe of
memory's `memory_fts`, with no model call, and fed the engagement gate and
the decide gate's salience floor, both removed with it. `AppraisalPlugin`
now implements no `should_process_message`; its only gate is the
before-send scoring below, which refuses a draft on its peers' objections. A config that still
sets `appraisal.probe.*`, `appraisal.weights.*` or `appraisal.novelty.*`,
or one of the importance-prior weights removed with the appraisal vector
(`appraisal.prior.w_stakes`, `appraisal.prior.w_valence`,
`appraisal.prior.w_novelty`, #1189), gets one WARNING per key at startup and at reload, naming it and saying to
delete it; the key is otherwise ignored.

**Stage 2** (`on_agent_response`) is a silent fork of the live window over
the finished exchange. A decline gets none (#420; PRD §3.7 Critique 3): when
the final went unsent and its text carries `NO_REPLY` or is empty, the
agent stayed silent and there is no response to appraise, so no task is
enqueued. The test is
`silence.is_decline`, the rule `SilencePlugin` records declines by; it
doesn't need that plugin loaded. A final another gate withheld has content
and still gets stage 2; a draft held for revision gets none (its revision does). If the test raises, stage 2 runs anyway and a
warning names the channel and trace. On a miss (forks off, the exchange left the window,
the fork failed or its reply was unusable) it is skipped with a warning
naming the stage, channel, trace and reason: the exchange gets no stage 2
and no standalone call is made. It produces `past_claims` and, when critique
offers lenses, `lenses`. The five-dimension appraisal vector (valence,
stakes, ambiguity, commitment density, novelty), the correction flag and the
memory-retrieval note the fork prompt carried were removed as unused
(#1189); `stage2` envelopes written before then still hold them, and nothing
reads them. `past_claims` is a list of `{claim,
queries}`: each statement in the reply that presents something as
established before the exchange, with up to 3 short search phrases. A
missing or malformed field reads as no claims; malformed entries are
dropped with a warning; it's capped at 5 claims. When critique has lenses on
offer, stage 2 also returns `lenses`, the critique vector (see "How a lens is
chosen" under CritiquePlugin below). The result is reported to
the outcome log as the `stage2` key of an appraisal record; the lens scores kept there are
the labels a trained readout head for lens selection would learn from (#141). Then `on_appraisal_idle` fires with `channel, trace_id, response_text,
annotations, lenses, past_claims`, outside the appraisal attribution. Every
enqueued stage 2 fires it exactly once: `lenses` and `past_claims` are the
result, or both `None` when none was produced (a miss above, or an
unexpected exception, which is logged at warning with the trace). That is
how an implementer tells "not yet" from "never" without a timer.
Cancellation (shutdown) fires nothing. A failing implementer is logged and
doesn't stop the others. It's a push rather than a wait because a consumer
that awaited stage 2 from a queued task would hold a TaskQueue worker the
stage-2 task may need.

**The hooks appraisal owns** (#1169). `AppraisalPlugin.hookspecs` declares
three hooks, so peers (critique is the one in the tree) reach appraisal
through hooks and never call methods on it:

- `on_appraisal_prompts(channel, trigger)` collects the lenses a peer wants
  scored. `trigger` is `"idle"` (stage 2) or `"before_send"`. An
  implementer returns a list of `(lens name, one-line question)` pairs, or
  `None`. Offers merge in hook-result order; a later duplicate name, a
  non-list result and a malformed pair (not two non-blank strings) are
  dropped with a warning. With no offers, the fork asks for the claims only
  (idle) or doesn't run (before send).
- `on_appraisal_before_send(channel, trace_id, text, annotations, lenses)`
  tells peers the lens shares of a final draft before it is sent. `lenses`
  is `None` after a miss. An implementer returns a list of objection
  strings, or `None`. Appraisal concatenates the lists in hook-result order
  and refuses the draft with them. A non-list result and a non-string item
  are dropped with a warning naming the trace, the channel and the type;
  a blank string is skipped.
- `on_appraisal_idle(channel, trace_id, response_text, annotations, lenses,
  past_claims)` is the stage-2 push above.

All three are `on_` hooks, so a failing implementer is logged and the others
still run.

**Before send.** `AppraisalPlugin` implements `should_send_response`. For a
`final` that is not a decline (`silence.decline_kind`, since the draft isn't
in the window yet) and has a trace, it collects the `before_send` offers,
forks the live window once with `DEFAULT_BEFORE_SEND_FORK_PROMPT`, and
fires `on_appraisal_before_send` with the shares. When peers return
objections, `should_send_response` returns a `Refusal` carrying them (see
"Held for a revision" below); otherwise it returns `None`, and any exception
is logged and the send goes ahead. A final is not checked while a hold is
outstanding on its channel. The instruction quotes the whole draft on one line, not
truncated, because the draft is what is being scored and the window doesn't
hold it yet. The fork asks only for the `lenses` object, with no claims
(the provenance check runs after the exchange only). No offers fires
nothing (debug log). A miss (forks off, the exchange out of the window, the
originating text unknown, the fork failing or unusable) logs a warning
naming the trace and fires the hook with `lenses=None`. The fork runs under
the `appraisal` stage and profile. The instruction is replaceable from
static config with `appraisal.before_send_fork_prompt` (not a runtime
tunable); the documented copy is `prompts/appraisal_before_send_fork.md`.
At startup, when forks are not enabled for the appraisal stage, appraisal
warns once naming `agent.forks.enabled` and `agent.forks.stages`: with no
fork it scores nothing, at either point. Because of that one warning, a
before-send miss for forks being off logs at debug per final; every other
miss warns. The before-send scores are also reported on `on_trace_record`
(`source: appraisal`) as `{"phase": "before_send", "lenses": {...}}`, with
`lenses` null on a miss so a miss is countable; stage 2's record gains
`"phase": "idle"`, so one exchange has a record for each phase.

**Held for a revision** (#1149, `corvidae_cognition/presend.py`). A draft
that draws objections before send ends one of two ways: sent as written, or
held back for one revision. Nothing is withheld on a lens's say-so.

- **Held.** The kernel decides whether the refusal came in time. If it did,
  the draft is stored unsent with the window's not-sent marker, so
  transports never see it. The hold's owner (appraisal, or critique's fallback) records `{"held": {"objections":
  [...]}}` on `on_trace_record` and hands the objections to the kernel's
  held-text store with `held.hold_and_wake` (source `pre-send check`,
  origin of the held exchange: the `origin` annotation, else the trace's
  recorded origin, else `user`). The stub that wakes the revising turn says
  the reply was held for the reasons in the CONTEXT above and asks for a
  revised one. It names no `NO_REPLY`, so Critique 4's silence default does
  not apply, though the agent may still decline.
- **The revising turn.** It is the turn of that origin at which the store
  offers the batch, and it sees the objections framed as CONTEXT from
  `pre-send check`. A held trace gets no stage 2, so critique runs no lens
  and no provenance check on it; the revision is judged as usual.
- **No second hold.** While a hold is outstanding, no final on the channel
  is checked, the revision's own included. The hold ends at the revision's
  final, or at the channel's first final after the offer with no tool call
  outstanding (a revision whose tools are still running, or whose turn
  failed, ends at the next such final). The span is bounded by the timeouts of
  the channel's outstanding calls (`tools.call_timeout`, `tools.timeouts`, and
  a tool's own); a call the operator unbounds can hold it open until restart
  (#1225). Finals from other turns in that span
  go out unchecked.
- **A late refusal holds nothing.** When the check misses the channel's
  allowance, the reply has already gone: nothing is held and no turn is
  woken, and the objections reach the channel's next turn through the
  kernel's late path (`late check` CONTEXT).
- **Objections not admitted.** If the revising turn's admission took none of
  the held objections (duplicates, or no room under the funnel's budget),
  one warning names the channel and says the revision may have gone out
  without them.
- **Failures.** If storing the held objections raises, one warning names the
  channel and trace, the objections are lost, no held record is written and
  no turn is woken; the next draft is checked as usual.

Known behaviours:

- A stub turn the kernel defers while a tool call is outstanding still makes
  its LLM call, with no objections in its window, before the turn that pops
  the batch.
- Held objections and the stub are in memory only; a restart between the hold
  and the revision loses both.
- A hold whose origin already has a pending stub for the same (channel,
  origin) queues behind it, and its revision text replaces that stub's
  (#1222): the one turn reads the revision stub and admits both batches. If
  that stub's turn was already deferred, its row with the earlier text stays
  in the window ahead of the re-fired revision row; and a row whose text was
  replaced keeps the earlier stub's source in its provenance.
- Over ACP, `cancel_prompt` empties the channel's outstanding-call set, so
  an interleaved final can end the unchecked span early and the revision's
  final can then be checked.
- If another gate vetoes in time while appraisal's refusal arrives late but
  before `on_agent_response`, appraisal holds and wakes a reply the other
  gate withheld: one extra turn.

### CritiquePlugin: the provenance check (`packages/corvidae-cognition/corvidae_cognition/critique.py`)

The provenance check looks for confabulation: a reply that presents
something as already said, agreed or done when no record of it exists. A
model makes both language judgments; the code only fetches evidence:

1. **Detection.** Stage-2 appraisal lists the reply's claims about the past
   in `past_claims`, with a few search phrases for each (see
   [AppraisalPlugin](#appraisalplugin-packagescorvidae-cognitioncorvidae_cognitionappraisalpy)).
   That works in any language and any phrasing, and it costs no extra model
   call because stage 2 already runs on every answered exchange. A decline
   gets no stage 2, so it gets no provenance check either.
2. **Evidence.** When stage 2 lands, AppraisalPlugin fires
   `on_appraisal_idle`, which CritiquePlugin implements. For an exchange with at
   least one claim it enqueues a silent task. Every main-loop turn with a
   trace is eligible, whatever started it (#1031; the same rule as the
   stylistic lenses), corrections included, so stage 2 runs on them too. That task searches `message_fts`
   with each claim's phrases (or the claim's own words when it has none):
   the phrases are split into terms (letters, marks, numbers), each term
   becomes its 3-character pieces (and whole when 4+ characters), quoted and
   OR-joined, with a `LIKE` scan only for a query of nothing longer than two
   characters. No word list of ours decides what matches, and query syntax in
   the model's phrases is never parsed. A non-string entry in the model's
   `queries` is skipped. Results are ranked
   by bm25, and up to 5 per claim are kept. Only these rows count:
   - rows from before the exchange's originating row. The exchange's own
     rows, the reply included, are already indexed when the check runs;
     the judge sees them in the window instead, and the reply is no record
     of itself. For a coalesced burst (#398) the boundary is the burst's
     first row, so the burst's own earlier messages never count as prior
     evidence. That applies only when the turn's own row is among the
     unanswered messages; a notification-born turn keeps its recorded row.
     The stylistic lenses' instruction still quotes the last burst message
     as where the exchange began, as appraisal's does (accepted: stage 2
     forks the whole window, so it sees every burst message).
   - `message` and `summary` rows. CONTEXT (injected data, including
     earlier critique) and `unsent` rows are never a record of something
     said.
   - rows from every channel (PRD §3.7 Critique 2, #168): the search
     reaches what has left the window, wherever it was said. The excerpts
     reach the critic, and its objections re-enter the window, so text can
     cross channels here; [threat-model.md](threat-model.md) finding 15
     records it.
3. **Judgment.** The check runs as a fork of the window, cut just after
   the exchange's final reply, so the judge sees what the agent saw
   when it wrote it: the system prompt, the exchange's own rows (tool
   results included), and the CONTEXT admitted for it. Rows written after
   the reply aren't in it. The instruction appended to that prefix lists
   each claim with its search hits. A hit longer than 500 characters is
   shown around its matched terms, not from its start, so support deep in
   a long tool result still reaches the judge. The judge is told the search shows near
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
   list, with the judge's reason, and reaches the held-text store tagged
   `[provenance]`. Stylistic objections queue through `held.hold` and wait for
the channel's next turn. Provenance objections wake the channel with a
stub whose text tells the agent the default is to send nothing: reply
exactly `NO_REPLY`, and reply in the channel only if its earlier reply
could lead the other party to take an incorrect action. With a
retention root (`critique.retention.root`, else markdown_recall's root), retention findings wake the channel too, and
every critique wake (provenance included) uses one stub that adds how to
file them: read the named file under the root and append only what is
missing. The held-text store keeps one pending stub per channel and origin, so the
first wake's text governs everything queued behind it. That is why the stub
depends on configuration only, never on which lens woke: it is the retention
note when a root is set and the provenance note otherwise. While the
proactivity lens is offered (`critique.lens.proactivity.enabled`, #1126), a
sentence follows it telling the agent to take the action a proactivity
finding names and then send nothing unless the other party would otherwise
act on a wrong belief. Once a loaded
plugin lens declares `wakes=True` one more sentence comes last ("Findings from
other self-check lenses may also be in the critique context: reply in the
channel about one only if your earlier reply could lead the other party to
take an incorrect action; otherwise send nothing, as above."), so the
silence default holds for every waking lens however the stubs coalesce. A
note's opening can therefore name a cause that didn't produce this wake
(shortcut: the provenance note's first sentence misstates the cause when
only proactivity or a plugin lens woke, which under the default config is
the common case for proactivity; rewrite both notes neutrally when that
confusion shows up in the harness).
On a critique-origin turn (a correction, or a turn a note from one woke)
nothing wakes: its provenance, proactivity, retention and other waking findings all wait
for the channel's next turn, so a correction never starts another (PRD
§3.7 Critique 4). Retention findings then wait without the filing stub. The plugin's
`should_send_response` vetoes a `final` on a critique-origin turn whose
stripped text is `NO_REPLY`; the kernel stores it as `unsent` with the
not-sent marker. That veto is critique's own and always on with this plugin
loaded; [SilencePlugin](#silenceplugin-packagescorvidae-cognitioncorvidae_cognitionsilencepy)
vetoes any final containing the token on every origin. The outcome is
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
one INFO line; `on_stop` closes the connection.
`critique.provenance.enabled` is the one switch. A
`critique.provenance.fork_prompt` override must ask for the verdicts format
above; one written for an objections list fails parsing, and every run
skips.

### CritiquePlugin: lenses other plugins add (`on_critique_init`)

A plugin adds a critique lens without editing `critique.py` (#1033). The
lens is data: critique decides when it runs, runs it through the same fork
machinery as the stylistic lenses, records it, and queues its objections.

**The hook.** CritiquePlugin owns `on_critique_init(config, add_lens)`
(declared in its `hookspecs`, so it follows the owner-segment rule in
[plugin-guide.md](plugin-guide.md#declaring-hooks-in-a-plugin)). Critique
calls it once, from its own `on_init`, and the lens set is fixed until
restart. An implementer calls `add_lens(lens)` once per lens:

```python
from corvidae.hooks import CorvidaePlugin, hookimpl
from corvidae_cognition.critique import Lens

class AlignmentLenses(CorvidaePlugin):
    optional_depends_on = frozenset({"critique"})

    @hookimpl
    async def on_critique_init(self, config, add_lens):
        add_lens(Lens(
            name="alignment",
            fork_prompt="[Self-critique: alignment] Pause the conversation "
                        "here. ...",
            appraisal_question="Does the reply break a Gricean maxim: too "
                               "much or too little, unfounded, off-topic, "
                               "or unclear?",
        ))
```

`config` is passed because critique's `on_init` runs concurrently with the
implementer's own, so the implementer can't count on having stored config
yet. A static precondition ("only when a knowledge base root is set") is
the implementer's to check: it doesn't add the lens. Declare
`optional_depends_on = frozenset({"critique"})`, so that with critique
disabled the orphaned hookimpl is reported at debug rather than warning.

Critique calls each implementer separately, with only the arguments its
hookimpl declares and its own `add_lens`, so every warning can name the
plugin. An implementer that raises is logged by name with the traceback,
and the lenses it added before the raise are kept.

**The lens.** Frozen dataclasses, holding no callables:

| Field | Meaning |
|---|---|
| `name` | Matches `^[a-z][a-z0-9_]*$`; it names the lens's config keys and its outcome entry |
| `fork_prompt` | The brief only. Critique appends the reply format (one JSON key per lens in the fork), so it is critique's, not the author's. `critique.lens.<name>.fork_prompt` replaces the brief (static config, reloaded like the built-ins' prompts) |
| `appraisal_question` | One non-empty line the appraisal fork answers to score this lens against the others and "none" (see below). `critique.lens.<name>.appraisal_question` replaces it (static config) |
| `before_send` | Default `False`. `True` also offers the lens on a draft before it is sent (see below); `False` keeps it to the after-the-exchange point. Must be a bool |
| `wakes` | Default `False`. `True` makes the lens's non-empty objections wake the channel (PRD §3.7 Critique 4); `False` queues them as CONTEXT for the next turn |

**Rejections never stop startup** (PRD principle 5, §3.1.17). A lens that
fails validation is not loaded and a WARNING names the plugin, the lens,
the rejected value, what was expected and the remedy; the other lenses
load. Rejected: anything that isn't a `Lens`; a bad name; an empty or
non-string prompt; a missing, blank, non-string or multi-line
`appraisal_question` (the warning says to declare a one-line question the
appraisal fork answers to decide whether the lens runs); a `wakes` or `before_send` that
isn't a bool; and a reserved name. A lens built with a removed selector
(`thresholds`, `reply_patterns`, `sample_rate`) raises a `TypeError` in the
plugin's `on_critique_init`, and the warning for that names
`appraisal_question` as the replacement. Reserved are the built-in lenses
(`predictive`, `constrained`, `adversarial`, `retention`, `proactivity`,
`provenance`), `none` (the vector's no-lens option), and the stylistic
run's outcome keys (`lenses`, `objections`, `sampled_below_threshold`,
`detail`, which a lens's outcome entry would overwrite). A name accepted from more than one plugin loads
from none of them, with one warning naming each contributor: entry-point
order isn't deterministic, so "first wins" would load a different lens on
different starts.

**Selection.** See the next section: the lens is chosen from the critique
vector, not by a rule the lens carries.

### How a lens is chosen: the critique vector (#1078)

Every built-in or plugin lens that beats "none" runs at each of two points per
exchange: before the reply is sent, on the draft, and after the exchange.
The lenses selected at a point run together in one fork (below).
The built-ins (`predictive`, `constrained`, `adversarial`, `retention` while a
knowledge base root is known, and `proactivity` while
`critique.lens.proactivity.enabled`) and every added lens are *offered* to
the appraisal fork, which scores them. Retention and proactivity
wake the channel; the other built-ins wait.

- **The offer.** `CritiquePlugin` implements `on_appraisal_prompts`, and
  appraisal collects the offer for each trigger at call time. It returns
  `(name, question)` pairs in a fixed order: built-ins in table order, then
  added lenses by name. At `"idle"` that is every lens; at `"before_send"`
  it is the built-ins except retention (a fact filed from a draft may never
  be sent) and the added lenses that declare `before_send=True`. A lens has
  one question, used at both points: the declared one unless
  `critique.lens.<name>.appraisal_question` overrides it; it is read from
  static config, so `set_settings` can't reword the critic it is scored
  for. An override that is not one non-blank line of text fails startup
  naming the key; at reload it logs a warning and the lens offers its own
  question until the config is fixed (#1127). Each call asks afresh, so a
  reload takes effect on the next exchange.
- **The ask.** The stage-2 fork instruction (the configured
  `appraisal.stage2_fork_prompt` as of the last reload, then the exchange
  quote) gets a `[Critique selection]` block appended (the before-send fork
  gets the same block after the draft),
  listing each offered lens as `- <name>: <question>`, and asks for a
  percentage per lens plus `none`, summing to 100, as
  `lenses: {name: percent, ..., none: percent}`. The call's response schema
  is a per-call copy with `lenses` required (the shared `STAGE2_SCHEMA` is
  untouched). It is the same single fork call: the call count doesn't
  change. With no lenses on offer the block and the schema addition are
  omitted and stage 2 is as before. The block is documented in
  `prompts/appraisal_fork.md`.
- **Normalization.** `_lens_vector` turns the reply into shares of 1 over
  the offered names and `none`. A name that wasn't offered, and a value
  that isn't a finite, non-negative number (a bool isn't one), is dropped
  with a warning naming the entry and the trace. The rest is renormalized;
  an offered name the reply left out counts 0. A missing or non-object
  field, or one with nothing usable left, gives no vector and a warning
  naming the trace and the reason. A bad `lenses` field never costs the
  claims: the rest of stage 2 still lands. A usable vector is stored
  in the exchange's `appraisal.stage2.lenses` and reaches critique through
  `on_appraisal_idle` (the before-send vector reaches it through
  `on_appraisal_before_send` and is not stored by appraisal).
- **The choice.** In `on_appraisal_idle`, after the provenance check, and
  in `on_appraisal_before_send` on the draft's scores, critique selects, from
  *this* exchange's vector (never another exchange's), every offered lens
  whose share is higher than `none`'s, in offer order. `none` wins any tie it is part of. A lens that
  beat `none` but is no longer offered (a reload changed the
  offer in between) is dropped, with a warning. No usable vector also
  counts as `none`: when the scoring produced nothing, a warning names the
  trace (before send with appraisal loaded, that warning is appraisal's own
  skip warning and critique does not warn again; with forks off on the
  before-send path it is DEBUG); when stage 2 is present but its `lenses` field is unusable,
  appraisal has already warned, so critique logs at debug.
- **The sampled path.** When `none` wins, or there is no vector, with
  probability `critique.sample_below_rate` (runtime-tunable per channel) one
  random offered lens runs instead, marked `sampled`: the standing
  false-negative bound (operator directive 3). The pool is every offered
  lens, built-ins and plugin lenses alike. The pool includes retention and
  proactivity, which wake, so each can wake a turn through this path at
  frequency `sample_below_rate` × 1/(offered lenses), the vector aside. The
  proactivity brief says to object only to a permission question, so on a
  reply that asks none it should return nothing.
- **Before send.** The before-send lens runs inline, inside the
  `should_send_response` check allowance, so the reply waits on it (if the
  allowance runs out the run still finishes in the background). Its
  objections, as text, are the return value of `on_appraisal_before_send`:
  appraisal refuses the draft on them and holds it for one revision
  ("Held for a revision", above). They wake nothing themselves, and a held
  trace is not judged afterwards. The provenance check never runs here; it
  runs after the exchange only. Critique's own `should_send_response` keeps
  the `NO_REPLY` veto and nothing else when appraisal is registered. Its
  `before_send` record (`critique_before_send`) carries `lenses`, a list, where it carried `lens`; one fork covers all the selected lenses, with no tools. Its objections open with `on your draft «…»: `.
- **No appraisal plugin.** Critique asks `pm.has_plugin("appraisal")` once,
  at `on_start`. Without it there is no vector to read, so one INFO line at
  startup names the fallback, and at each point (before send, on a final
  that is not a decline with a trace; and after the exchange) one random
  lens offered for that point runs on a share of exchanges,
  `critique.fallback_rate` (runtime-tunable per channel, default 0.25),
  built-ins and plugin lenses alike. This replaces the older run-every-exchange
  fallback (PRD §3.2 degrade-to-spec); the rate keeps the cost bounded. The
  provenance check is off, since it needs appraisal's claims. The fallback
  also holds: a sampled draft whose lens objects is refused by critique's own
  `should_send_response`, and critique runs the same hold-and-revise state
  (one revising turn, unchecked; no idle lens on the held trace) through the
  shared `PreSendHold`. A hold is never started while one is outstanding.
- **Record.** The choice is reported as the `lens_selection` key of a critique record
  (`on_trace_record`, `source: critique`; `lens_selection_before_send` for the
  before-send point, and `critique_before_send` for the run's outcome)
  whether or not a lens runs, so a `none` is countable:
  `{"lenses": ["<name>", ...], "sampled": false, "basis": "vector"}`. `lenses` is
  omitted when nothing was selected (it was `lens`, a single name, before #1190). `basis` is `"vector"` when a lens won the vector,
  `"none"` when the vector was usable and `none` won (also after a sampled
  draw), and `"no_vector"` when there was no usable vector.

Two consequences. With appraisal loaded, the stylistic lenses start after
stage 2 lands (or on the draft's scores), not at `on_agent_response`, so a stage-2 miss (forks off, the exchange left the
window, an unusable reply) also costs the exchange its lens: fork misses
are more likely to skip a lens than before. And planning and alignment no
longer have a 10% rate of their own: they run when their share beats "none"'s, plus
their share of the `sample_below_rate` draw.

**Writing a question.** One line, phrased about the exchange the fork just
read, answerable yes/no from the window, and about what the lens would
find rather than what it would say. Specific beats general: "Does the reply
make concrete commitments, figures or promises that could be wrong or go
unkept?" separates exchanges; "Is the reply good?" scores everything the
same. Questions compete, so a lens whose question is broader than its brief
steals exchanges from the others. Reword the question, not a threshold, to
tune how often a lens runs.

**Running and delivery.** After the exchange, critique enqueues one silent
`critique` task for all the lenses chosen, built-in and added. It makes one
fork of the live window with the containment check: forks off, or the
exchange gone from the window, skips every lens with a warning naming it
and records nothing.

**The shared fork (#1190).** The instruction is one `## Lens: <name>` section
per selected lens, the quote of the exchange, and one reply format that asks for
a single JSON object with one key per lens (`{"<lens>": {"objections": [...]}}`).
A lens whose key is missing from the reply, or is not an object, is skipped
alone, with a warning; an unparseable reply skips them all. Each lens keeps
its own wake setting; their outcomes share one `critique` record (below). Before send,
the fork makes no tool calls and does not wait. After the exchange:

- It waits before it renders, and again before each request that follows a tool
  round, while a user message waits on the channel (polled once a second), so it
  never takes the model from a turn in flight.
- It sees the window cut at the reply, not later turns that landed while it
  waited (PRD §3.8). A reply that was not recorded or not persisted is a miss,
  as for provenance, which is newly true of the lenses: they used to run. A
  compaction since the reply, or during the fork, is a miss too.
- It may call read-only tools, up to `critique.max_tool_calls` (default 3), then
  must answer. With tools allowed the fork sends no `response_format`, so the
  reply is parsed from text; with none it does.
- `critique.fork_timeout` (default 120 s) bounds the whole fork, wall clock,
  pauses included. Past it, or a channel busy past it before the fork starts,
  every lens is skipped with a warning naming lens, channel and trace; nothing is
  recorded. The provenance fork has no deadline.

A lens that ran reports, in a critique record,

```json
{"critique": {"<name>": {"objections": 1, "sampled": false, "detail": [...]}}}
```

(`detail` only when there are objections), in the same `critique`
record as the built-in lenses' keys (`"lenses": [...], "objections": n, ...`)
when both ran in one fork: one record per run, so OutcomeLogPlugin's last-line-per-key rule
loses nothing.
Its objections, tagged `[<name>]`, re-enter through
`held.hold` as framed CONTEXT at the channel's next turn and wake
nothing, unless the lens declares `wakes=True` (PRD §3.7 Critique 4): a
waking lens's objections re-enter at the turn its stub triggers, under the
same silence default as provenance's, and still wake nothing on a
critique-origin turn. Each finding names the reply it judges: it opens
`on your reply «<start of the reply>»: ` (`on your draft «<start of the draft>»: ` for a
before-send finding, since the reply may be sent by the time it lands), so
the agent can tell which of its replies a late finding is about (PRD §3.7
Critique 4). Redaction doesn't reach
`critique.<name>.detail` yet; redaction is on hold (#1005).

### CritiquePlugin: lens context from other plugins (`transform_critique_prompt`)

A lens's brief is static, so a lens that needs live context (the planning
lens needs the open goal list) adds it per fork through
`transform_critique_prompt(channel, trace_id, lens_name, instruction) -> str`,
which CritiquePlugin owns and declares in its `hookspecs` (#1067).

It fires once per lens in `_fork_lenses`, on that lens's section of the fork
instruction: for every built-in lens (proactivity included, #1126) and every
added lens, but not provenance, which builds its own fork. The lenses at a
point share one fork (#1190), so the instruction is one `## Lens: <name>`
section per lens, then the exchange quote, then one reply format. The hook
receives the section (its heading, the brief, and retention's knowledge base
root) and not the quote or the reply format, so an implementer returns a
section, not a whole prompt. It runs before the containment check, so a slow
implementer can't let the exchange leave the window unnoticed.

Resolution is `firstresult` over a trylast seed, CritiquePlugin's own
implementation, which returns the instruction unchanged. A plain hookimpl
that returns a string ends the chain, so an implementer answers only for
its own lens and returns `None` for every other, the way `resolve_channel`
implementers answer only for their own prefix; the order pluggy calls them
in then can't change the result. A plain implementer may await.
`@hookimpl(wrapper=True)` implementers compose around the result, LIFO, but
apluggy wrappers are sync generators and can't await. Return the
section with your text appended rather than rewritten. A result that is
not a string is ignored, with a warning naming the lens, trace and channel,
and the fork uses the untransformed section.

An implementer that raises is logged at WARNING naming the lens, the trace
and the channel, with the traceback, and the lens's section stays
untransformed.

### GatePlugin (`packages/corvidae-cognition/corvidae_cognition/gates.py`)

The decide gate (PRD §3.7 Output gate). Entry point name: `"gates"`;
`depends_on` is `trace`; it fires `on_trace_record` and asks
`resolve_trace_origin` for an origin it no longer holds. It withholds an emission on its origin and the operator's
switches. The engagement gate and the decide gate's salience floor were
removed with stage-1 appraisal (#1185), so this plugin has no inbound hook
and withholds no final from an origin that is not quiet.

**Origin.** Read from propagated state, never inferred: `annotations["origin"]`
when a producer stamped one (the scheduler stamps `reminder`,
`HeartbeatPlugin` stamps `heartbeat`, critique's `hold_and_wake` stamps `critique`), else `user` for a trace the
plugin saw in `on_message_admitted` (a bounded LRU of 1024), else the origin `resolve_trace_origin` returns for the trace (so a user exchange whose trace
fell out of the LRU during a long tool cycle is still `user`), else `task`.
Tool cycles carry the item's annotations forward, so an exchange keeps its origin
through its tool calls. The gate reports with `origin=None`, so it can't relabel a trace. The origin is looked up only
when a quiet origin is configured.

**The decide gate** (`should_send_response`):

- `final`: withheld only from a quiet origin (below). Any other final is
  sent and recorded nowhere by this gate.
- `progress` / `thinking`: vetoed when `gate.send.allow_progress` /
  `gate.send.allow_thinking` (both default true) are off, recorded as
  `outcomes.suppressed = {"progress": true}` or `{"thinking": true}`. Core
  never writes the outcome log; the vetoing plugin reports it.
- `error`: withheld only from a quiet origin (below, #936); otherwise no
  opinion.
- Anything else: no opinion.

Rows written before #1185 carry the removed gates' records
(`outcomes.engagement`, and `outcomes.send` with `salience`, `min_salience`,
`would_veto` and `advisory` for every final); nothing writes those keys now.

**Quiet origins** (#790). `gate.send.quiet_origins` (default empty; a
per-channel runtime tunable, #928) lists origins whose turns send nothing to the human, such as
`webhook:github`. For such a turn the gate withholds the `final`, recorded as
`outcomes.send = {emission, origin, vetoed: true, reason: "quiet_origin",
surfaced: false}`, and withholds `progress`, `thinking` and `error` ahead of the
allow switches, recorded as `outcomes.suppressed = {<emission>: true}` plus
`quiet_origin: <origin>`. Withholding `error` (#936) is what keeps a model
outage during a burst of notifications from sending one failure message per
turn; the kernel logs the exception at ERROR, with its traceback, before this
hook runs, and records `error fallback not sent: vetoed by should_send_response`
at INFO. Neither kernel line names the origin, so what ties the missing message
to it is this row: `outcomes.suppressed = {"error": true}` together with
`quiet_origin`. `user` and
`reminder` are never quiet; listing either is ignored with a warning, and a
value that is not a list of strings fails open with a warning naming the
channel. The list resolves like any tunable: a channel's `set_settings`
override replaces it there, otherwise the config value applies; a hot
reload takes effect on the next decision except where an override holds.

The agent always has a `speak_up(reason)` tool while `gates` is loaded,
whatever the list holds, so an origin added by hot reload has its hatch
without a restart (tools are collected only at startup, #925; with nothing
quiet, the call lifts nothing). Calling it marks the turn's trace as
surfaced (a bounded LRU of 1024, never cleared explicitly, because a final's
thinking is gated after the final) and records `outcomes.speak_up = <reason>` (cut to 200 characters; reported
with the turn's trace id). A surfaced quiet final
is then sent, recorded as `outcomes.send = {emission, origin, vetoed: false,
reason: "surfaced", surfaced: true}`. Progress and thinking sent in
the same response as the `speak_up` call are gated before the tool runs and
stay withheld. If `speak_up` is in a batch cut off at `max_turns`, it never
runs and the fallback final stays withheld.

What the kernel does with a withheld final is in
[design.md](design.md#the-output-gate). On the cognition side, critique still
judges a withheld final, whatever its origin, though not a decline (an
unsent `NO_REPLY` or empty final, #420), its provenance probe doesn't
count a withheld row as a record of anything said, and `recall_raw` and
`search_transcripts` label withheld rows `(withheld, not sent)`.

Every tunable resolves through `resolve_tunable` at decision time, so a
`set_settings` override or a hot config reload applies on the next
exchange. Switches are type-checked on read: booleans and the words
true/false/yes/no/on/off/1/0. Anything else logs a warning naming the key
and value and uses the default. The gate catches any other error, logs it
and fails open, because an exception in that broadcast would drop the
reply. A config that still sets a key #1185 removed (`gate.engagement.*`,
`gate.send.min_salience`, `gate.send.enforce`) gets one WARNING per key at
startup and at reload, naming it and saying to delete it; the key is
otherwise ignored. The keys are in
[configuration.md](configuration.md#gate--the-decide-gate), including the ones an
operator should consider adding to `agent.immutable_settings` on channels
with untrusted participants.

### SilencePlugin (`packages/corvidae-cognition/corvidae_cognition/silence.py`)

SilencePlugin lets the agent end a turn without saying anything, on any channel, including a turn that answers a user directly. Entry point name: `"silence"`; `depends_on` is `trace`.

The test for staying quiet is Grice's: does the agent have anything true, relevant and worth saying to add? "thanks, night!" and "lol ok" fail it; a question passes. The decision is the model's, made in the turn, and it is driven by the system prompt rather than by a score: a channel only gets the option if its prompt says so.

**The instruction.** The plugin wraps the kernel's `transform_system_prompt` hook. Once a channel's identity and register are composed, it adds a blank line and the paragraph below to the end, but only when the transport that owns the channel is conversational (`TransportPlugin.conversational`, true for IRC, Discord, Signal and Zulip). ACP, web and the CLI are not, so they never hear about `NO_REPLY` and always answer; that is deliberate, because a request addressed to the agent in a task session should never go unanswered. (A model can still end a turn with an empty reply anywhere. The kernel leaves that unsent too, and it is recorded the same way.) The built-in text is `REGISTER_FRAGMENT` in `silence.py` (`TOOL_FRAGMENT` or `BOTH_FRAGMENT` when `silence.mode` is `tool` or `both`, see below); `silence.instruction` replaces it (inline, a file, or a list of files; read at startup, see [configuration.md](configuration.md#silence--the-right-to-silence)):

```
You don't have to answer every message. Before replying, ask whether you have anything true, relevant and worth saying to add. When a conversation has wound down (thanks, goodnight, "lol ok", an acknowledgement that needs nothing back), stay silent: reply with NO_REPLY (and nothing else), and nothing is sent. If they wanted more, they'll say so. A question or request addressed to you still gets a real answer.
```

The hook runs on the composed prompt whatever the register's source, so a channel with its own `register_prompt`, or with `register_prompt: ""`, still gets the paragraph on a conversational transport. There is no per-channel switch; to turn it off, disable the plugin. The paragraph is part of the system prompt, which is fixed when a conversation starts or restarts and is not rebuilt at compaction.

**The veto.** `should_send_response` returns `False` for a `final` emission whose text contains `NO_REPLY` (a containment check, not an exact match), on every turn origin. The kernel stores the row as `unsent` with the not-sent marker, so on later turns the model can see it stayed silent instead of believing it spoke. Nothing reaches the transport. A final that mentions the token in passing (e.g. "reply with NO_REPLY to stop") is also dropped — a known tradeoff of the containment design. Only `final` is covered: a `NO_REPLY` that shows up as progress text beside tool calls goes out. A veto that raises fails open, with a warning, and the text is sent.

**The record.** `on_agent_response` fires for unsent finals too, after the row is in the window. When the final was unsent (`final_was_unsent`) and its text contains `NO_REPLY` (the same test as the veto) or is empty, the plugin reports `silence` in a `silence` record for that trace, from a background task so the reply path never waits on the write:

| `silence` | Meaning |
|---|---|
| `no_reply` | The model's final was `NO_REPLY`, or contained it, and the veto kept it unsent |
| `empty` | The final had no text and no tool calls; the kernel left it unsent without asking any gate |
| `tool` | The model called `end_turn` (`silence.mode` `tool` or `both`). It records the call, not what followed: it is written when the tool runs, before the follow-up exists, so a model that then answers anyway still has this row. `silence_reason` carries the model's own words, cut at 200 characters |

A `NO_REPLY` that was actually sent (say another gate's veto raised and the send failed open) is not recorded, and neither is a final some other gate vetoed; the `tool` row above is the exception, because it is written before the final exists. A turn with two finals, a reply and then a declined follow-up, gets the flag. Without `outcome_log` the records go nowhere (`trace` warns at startup) and declines go unrecorded; the veto still works. A listener that fails is logged by hook dispatch, naming the hook and the plugin; outcome_log's own failed append names the file, trace and source. A decline with no trace id logs a warning and records nothing.

**The `end_turn` tool (`silence.mode`, #797).** Writing a literal `NO_REPLY` is an unusual thing for a chat-trained model to do, since such models are trained to alternate turns and answer. The hypothesis behind the tool is that a tool call is a shape they are trained to emit more readily; whether that holds is what the experiment measures (see The tool's record). With `silence.mode: tool` (or `both`) the plugin registers `end_turn(reason)` through `register_tools`, and the instruction says to call it on its own when nothing needs saying (`TOOL_FRAGMENT`; `both` adds a fallback to the token, `BOTH_FRAGMENT`). `token`, the default, registers no tool and behaves as before. A `silence.instruction` replaces the built-in text in every mode, so in `tool` or `both` mode it has to mention `end_turn`.

**What the tool does, and what it cannot.** The tool has no kernel support, so it does not end the turn: the loop calls the model again after any tool result. The result therefore says `Nothing was sent. Your next message must be exactly NO_REPLY and nothing else.`, and the veto withholds that echo *when the model complies*. If it writes anything else (a near miss, a sign-off, a real answer), that text is sent. The call is the decision and the echo is mechanical; it costs one more model call, which a backend that reuses the cached prompt prefix makes cheap. The tool refuses on a channel whose transport isn't conversational, including a channel no transport owns (an `Error:` result naming the channel and saying to answer normally), and with no channel context.

**The tool's record.** It reports `silence = "tool"` and `silence_reason` when the tool is called, from a tracked task like the other records, when the turn has a trace id. The turn's own final is then never recorded over it, whether it was withheld or sent. A `tool` row therefore means "called", not "stayed silent"; count rows against the wind-down messages to measure whether the model uses the tool, and read the sent rows to see whether the follow-up complied.

**Known limits.**
- Text the model writes alongside the call goes out as progress, because the progress gate runs before the tool executes and receives no tool calls. The tool's description says to call it alone.
- At `max_turns` the kernel suppresses the tool call and sends the model's own text, or its fallback text if it wrote none, so that turn is not silent and nothing is recorded.
- The tool is in the tool list of every channel and refuses where it can't work. To remove it, add `end_turn` to `tools.disabled` (globally or per channel); the instruction would still tell the model to call it, so also set `silence.mode: token` or a `silence.instruction` that doesn't mention it.
- A kernel-level turn-ending tool, which would skip the follow-up call, is a separate proposal.

**Divergences from the PRD.** The code is what runs; where it differs from [prd.md](prd.md) §3.7 (Output gate and the right to silence), this section says how, and the PRD is unchanged until the owner decides which side moves.

- *A signal followed by sent text is not a decline (item 5).* In `tool` and `both` mode `end_turn` does not end the turn (the loop calls the model again), so nothing is sent only when the model then echoes `NO_REPLY`; any other follow-up is sent, and text written beside the call goes out as progress. The `tool` row is written when the tool is called, so it also exists for a turn that went on to reply: it records a call, not a decline, where the PRD says such a turn is not recorded as one. Closing this means writing the row when the final is known (#901).
- *The veto acts on the agent's own signal (item 5).* The veto is a containment check, so it also withholds a final that only quotes the token, such as a reply repeating an incoming message that contains `NO_REPLY`: untrusted text can get one reply suppressed without the agent choosing silence. The row is stored as `unsent` and recorded as `no_reply`, so it is visible, and the cost is one re-prompt. Matching only an exact token would instead send the token to the user whenever the model writes anything beside it, which is why containment was chosen. Narrowing the match is #902; the exposure is in [threat-model.md](threat-model.md).

**Ending the turn.** An unsent final never reaches `send_message`, which is where Discord and Signal give back every outstanding typing slot, ACP completes the prompt and the CLI prints its next `> `. Those four transports now do the same in an `on_agent_response` of their own when the final was unsent (see [the plugin guide](plugin-guide.md#transports-with-per-turn-state)). That runs whether or not this plugin is loaded, so an empty final no longer keeps the typing indicator refreshing or an ACP prompt hanging. On Signal, a release that takes the count to zero sends an explicit typing stop, so the indicator clears at once; Discord has no stop, so its indicator lingers until the client times it out.

**Disabling it.** `plugins.disabled: [silence]` removes the instruction, the veto and the `end_turn` tool; `agent.minimal.yaml.example` does this. A restart rebuilds every conversation's system prompt, so disabling the plugin drops the instruction everywhere at once. The windows still hold earlier `NO_REPLY` rows, though, and with no veto the model may imitate them and send the literal text. If that matters, start those conversations fresh.

**Appraisal and critique skip a decline** (#420). The same rule, exported as `is_decline(channel, response_text)` (with `decline_kind(text)` for the shape), gates `AppraisalPlugin`'s stage 2 and `CritiquePlugin`'s lenses, so a `NO_REPLY` is never scored or objected to. Both call it before their first await and fail toward running if it raises. A `NO_REPLY` that was sent is a response and is judged.

**Known gaps.** The instruction is the same in DMs and group channels; a group-specific variant that names addressing isn't built.

### HeartbeatPlugin (`packages/corvidae-cognition/corvidae_cognition/heartbeat.py`)

A turn the agent starts on its own when nothing else is happening (PRD
§3.7 Heartbeat, principle 13). Entry point name: `"heartbeat"`;
`depends_on = {"idle", "registry"}`. It does not need `memory`: with memory
disabled it starts and beats without a memory section.

**When it beats.** At `on_start`, if `heartbeat.after` is set, it registers
a system idle watch (`IdleTracker.watch_system`) with that period. The
tracker calls it once no channel has had inbound or in-flight work for
`after` seconds, never while one does. The beat is itself inbound work on
`internal:heartbeat`, so the system is busy until the beat's turn
settles, and the next beat is due `after` seconds after that. Traffic on
any channel cancels the pending beat. The result is at most one beat per
`after` seconds, never two at once, and none during a conversation. Under
traffic that never stops it never beats; that is accepted until #214.

**What a beat is.** `heartbeat.prompt` (`DEFAULT_HEARTBEAT_PROMPT`; documented
copy in `prompts/heartbeat.md`, and the key's resolution in
[configuration.md](configuration.md#heartbeat--a-turn-on-system-idle)) is the
whole beat unless plugins add to it. The plugin owns `on_heartbeat_collect_prompts(channel)`
and calls it through `pm.ahook`, which fails soft (PRD §3.1.10); every implementation
returns one section or `None`, and the sections come first, in hook order, then
the prompt last, separated by blank lines. A raising implementation is logged
at WARNING naming the hook and plugin and skipped, and the beat goes out without
its section. Before notifying, the beat re-checks that
the system is still quiet (a message may have landed during the gather), then
calls `on_notify` on `internal:heartbeat` with `source="heartbeat"` and
`annotations={"origin": "heartbeat"}`. The heartbeat adds nothing of its own; two implementations ship in other plugins. Memory's
(see [MemoryPlugin](#memoryplugin-packagescorvidae-cognitioncorvidae_cognitionmemorypy)), `tryfirst`, returns the recent
memories under `memory.heartbeat_limit` (default 5), framed as the agent's own summaries and as data, oldest first: all of
them on the first beat, then only those not yet sent (none new means no
section). Goals' (see [GoalsPlugin](#goalsplugin-packagescorvidae-cognitioncorvidae_cognitiongoalspy)), `trylast`, is the last
section: the capped goals list (open and in-progress only), then its pick rule, directly before the prompt, which ends the beat. With neither (memory or goals disabled, say), a beat is the bare prompt; the default prompt is plugin-neutral ("Review the sections above, if there are any, ...") so it reads true either way. Nothing
adds a "nothing new" line. The memory and goals state is committed when the hook
runs, so a beat that the quiet re-check then skips doesn't resend its items.
`prompts/heartbeat.md` shows a full beat. The turn then runs the
ordinary notification pipeline with the agent's normal tools. The `heartbeat`
origin follows the general conventions: critique judges it like any other
turn (#1031), `gate.send.quiet_origins` can make it quiet, and the outcome log
records it.

**Its window.** The window grows beat by beat and is bounded by ordinary
compaction at the channel's budget, `channels.internal:heartbeat.max_context_tokens`
(else the agent default). The summary and the rows stay in `message_log`, and
the channel is consolidated like any other: at turn ends (forks on),
on idle, and at compaction. After a
compaction the window holds the summary, then the most recent rows that
`agent.compaction_retention` keeps, then the beats since.

**What it can reach.** No transport owns `internal:heartbeat`, so a beat's
reply goes to `sessions.db` and the jsonl log, never to a person. Reaching
a person needs the send or note tools (PRD §3.5.9–10, #234, #235), which
don't exist yet; when they land, the beat uses them like any other turn,
with no allowlist. The prompt tells the model that most beats warrant
nothing, and to say so in one line and stop.

**What it reviews.** With memory loaded, the newest `memory.heartbeat_limit` (default 5)
indexed, unredacted records of the whole memory, every channel's, listed
oldest first. The beat reaches every channel's records (#168). Records consolidated from the
beat channel itself are in the list too, so a beat can see an earlier
beat's conclusions once they are consolidated.

**Known limitation: a beat can lose sight of its own outreach.** PRD §3.5.9
and §3.7 Heartbeat let a beat speak on another channel or leave a note
there. A beat sees its own earlier actions only in the window, which, after a compaction, holds
the summary, then the most recent rows `agent.compaction_retention` keeps,
then the beats since. Older beats leave the window, and memory brings them
back only once they are consolidated; the review shows only the newest
`memory.heartbeat_limit` records. So once the send and note tools exist, a beat could repeat an
outreach that compaction has summarized out of the window. Today it
can't happen: there is no send, note or read tool yet (#234, #235), so a
beat can't reach anyone. The bounds once they land are the
window, the beat's goals list (a beat that records an outreach as a goal step sees
it in later beats), and the target channel's memories, once its rows have been consolidated and
are among the newest `memory.heartbeat_limit`; the
PRD's per-destination limit on messages the agent starts caps repeats too,
but only when the operator sets it. The
cross-channel tools (§3.5.9–11) should record each outreach where a later
beat can see it, for example as a row on the target channel that the review
picks up. The same holds for a beat's other side effects (`write_file`,
`shell`): nothing records them outside the window either.

**Sleep.** The heartbeat is asleep whenever `heartbeat.after` is absent
(the block deleted, `null` or empty). It reads the key once at `on_init`,
so sleep is set in `agent.yaml` and survives every restart;
`plugins.disabled: [heartbeat]` also works. A config reload doesn't change
it.

**Failures.** A contributor whose read raises (memory's or goals') is logged as a
WARNING naming the hook and the exception; its section is left out and the
beat still goes out. An invalid `heartbeat.after` (zero, negative, non-finite,
boolean, non-numeric) fails startup with a `ValueError`, naming the
key, the value and what is accepted, while an awake
heartbeat without the memory plugin starts and beats with no memory section.

Distillation and rehearsal are not part of the heartbeat.
