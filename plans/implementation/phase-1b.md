# Phase 1b — Reconsolidation, demotion, redaction, memory tools, evals

**Effort:** M. **Dependencies:** Phase 1a complete (schema, consolidation,
retrieval, funnel all live — 1a's DDL already carries the columns this phase
uses: `indexed`, `superseded_by`, `redacted`, `retrieval_count`,
`last_retrieved_at`).
**Normative references:** `bootstrap-mapping.md` §3.1 (retention,
reconsolidation, redaction), §4.11–4.12 (FTS surfaces, redact cascade,
second-process discipline), §5 divergence 6, §6 (eval discipline), §7 row 1b.

**Goal:** the memory store becomes self-maintaining and bounded — records
strengthen through recall and demote when dead weight, near-duplicates merge
instead of crowding retrieval, the operator gets a real deletion path
(`redact`), the agent gets its "remember harder" tools, and the retrieval
quality claims become benchmarks.

## Read first

- `plans/bootstrap-mapping.md` §3.1 "Retention — reconsolidation, not
  culling" and the three disciplines + redact exception; §4.11–4.12; §6.
- `corvidae/memory.py` as built in Phase 1a (schema, consolidation task,
  retrieval scoring).
- `corvidae/tool.py` / `RuntimeSettingsPlugin` (`tools/settings.py:63`) for
  the `register_tools` pattern. Do NOT copy `tools/goal_tracker.py` — it is
  vestigial (obsolete hook signatures, no `register_tools`, not in the
  entry points).
- `corvidae/commands/serve.py` for the `corvidae.commands` entry-point
  pattern (the redact CLI follows it).
- `tests/evals/metrics.py` and `scripts/eval_memory.py` from Phase 0.

## Amendments after embedding-prefixes (2026-07-06)

The embedding-prefixes change (merged to main after 1a) postdates this plan.
Three consequences, folded into the WP text below and repeated here for the
design phase:

1. Every 1b call site that embeds memory text (the retention job's
   backfill step — re-promotion defers to it, see WP1b.1) must call
   `embed(texts, kind="document")` — unprefixed vectors are inconsistent
   with the prefixed index.
2. The `embedding_meta` guard now covers `document_prefix`/`query_prefix`
   as well as encoder/dimensions. On mismatch, embedding is DISABLED and
   config revert is the only remediation. The retention job's backfill step
   must check for that disabled state and skip (log once, DEBUG), not retry
   into an error loop.
3. Consolidation-time dup detection (WP1b.2) and the stub-embedder
   benchmarks (WP1b.5) are unaffected: the former reuses the consolidation
   document embedding, the latter is prefix-agnostic by construction.

## Design constraints and traps

1. **Demote, never delete** (§3.1, §5 divergence 6). Demotion = remove the
   vec row + `indexed=0` + `embedded=0` (once the vec row is deleted no
   stored vector exists — `memory_vec` is the only vector storage, so the
   flag must say so; see WP1b.1). The `memory` row, its FTS entry, and the raw
   `message_log` range all remain. Retrieval (`before_agent_turn`) excludes
   demoted records; the `search_memory`/`recall_raw` tools include them,
   flagged — "remember harder" must reach what passive retrieval no longer
   surfaces. The tools reach further in *status*, never in *compartment*:
   they enforce the calling channel's `_channel_scope` (WP1b.3).
2. **The one exception is `redact`, and it is operator-only.** It is a CLI
   command, never a registered agent tool, never reachable from any hook the
   model can influence. Its cascade is complete or it is a lie: message
   tombstone + memory tombstone + vec delete + **FTS delete/reinsert**
   (external-content FTS5 does not notice content changes by itself —
   §4.11; the 1a `memory_au` trigger covers the memory side ONLY if the
   tombstone is written via `UPDATE memory SET summary=...`; verify, don't
   assume).
3. **Grace period and floor before any demotion math** (§3.1): a record
   younger than the grace period is exempt (a never-retrieved record is the
   one nobody thought to ask about yet); a record with importance ≥ the
   floor never demotes for lack of traffic.
4. **Merge at consolidation time, not retrieval time** (§3.1): usage
   weighting inherits similarity bias — near-duplicates reinforce each other
   and crowd the budget. The dup check runs when a NEW record is created.
5. **Retention/merge jobs are silent background work** — plugin-owned
   `asyncio.create_task` with attribution (`stage="retention"`), tracked
   handles, never the TaskQueue (same rule as 1a trap #5). `on_idle` is
   push-based and never fires on a zero-traffic daemon (§3.1), so the
   retention job ALSO runs once at startup. The rate limit is persisted
   (`retention_meta.last_run`, WP1b.1) so a crash-looping daemon does not
   re-run the job on every start; the startup trigger honors it — a daemon
   idle longer than the interval still runs at next start, preserving the
   zero-traffic guarantee.
6. **The redact CLI is a second process on the same DB** (§4.12): its own
   connection, `PRAGMA busy_timeout = 5000`, short transactions, WAL
   assumed (it is the configured default — assert and abort with a clear
   message if the journal mode is not WAL).
7. **CI benchmarks are deterministic** (stub embedder, fixed fixtures,
   asserted floors). LLM-judged fidelity checks live behind the existing
   `eval` marker / `scripts/eval_memory.py`, out of CI (§6).
8. **`message_log` rows are never deleted, reordered, or renumbered** —
   the redact tombstone rewrites the `message` JSON content in place,
   preserving row id, role structure, and timestamps.

## Work packages (in order)

### WP1b.1 — Retention scoring, demotion, re-promotion

**Files:** `corvidae/retention.py` (new), `corvidae/memory.py` (schema DDL
and silent-task wiring only)

Module split: `memory.py` is already ~1,000 lines — the "~500 lines"
threshold this plan originally set has long since triggered. The score
function, constants, and the retention-pass logic live in
`corvidae/retention.py` (plain module, no plugin — it needs MemoryPlugin's
db/embedding internals, so a separate plugin would only add indirection);
`MemoryPlugin` keeps the hook wiring (`on_start` startup trigger, `on_idle`)
and spawns the silent task that calls into it. Tests target the module.

1. **Retention score** (module-level function in `retention.py`, constants
   commented as §6-tunable):

```python
def retention_score(importance: float, retrieval_count: int,
                    last_activity: float, now: float,
                    half_life_days: float) -> float:
    """Usage-weighted retention (bootstrap-mapping §3.1).

    last_activity = max(created_at, last_retrieved_at or 0).
    """
    recency = math.exp(-((now - last_activity) / 86400.0) / half_life_days)
    return importance * (1.0 + RETRIEVAL_BOOST * math.log1p(retrieval_count)) * recency
```

   Constants: `RETRIEVAL_BOOST = 0.5`, demotion threshold
   `memory.retention.demote_below` (default 0.15), grace
   `memory.retention.grace_days` (default 14), floor
   `memory.retention.importance_floor` (default 0.8), retention half-life
   `memory.retention.half_life_days` (default 90 — distinct from the
   retrieval half-life).
2. **The retention job** (silent task — trap #5; triggers: startup + each
   `on_idle`, rate-limited to once per `memory.retention.interval` seconds,
   default 6h; the last-run timestamp is PERSISTED in a one-row
   `retention_meta (last_run REAL)` table — DDL alongside the other memory
   schema in `memory.py` — so restarts don't reset the limit; both triggers
   consult it). Three passes, in this order (so a record re-promoted in
   this run is eligible for the same run's backfill batch):
   - **Demote.** For every `indexed=1 AND redacted=0` record past grace and
     below floor: if score < threshold → `indexed=0`, `embedded=0`,
     `DELETE FROM memory_vec WHERE memory_id=?`. Log count. Setting
     `embedded=0` is what keeps the flag truthful: `memory_vec` is the only
     vector storage, so after the delete no vector exists for the record,
     and `embedded` is documented as "0 = embedding pending/failed
     (backfillable)" — leaving it 1 would strand re-promotion with no
     vector source.
   - **Re-promote.** For every `indexed=0 AND redacted=0 AND superseded_by
     IS NULL` record: if score ≥ threshold (it was recalled via
     `recall_raw` since demotion — see step 3) → `indexed=1`, nothing else.
     Re-promotion NEVER embeds inline: the record has `embedded=0` (set at
     demotion), so the backfill pass below restores its vec row through the
     one code path that already handles prefixes, batching, and the
     disabled-embedding guard. When embedding is disabled (the
     `embedding_meta` mismatch guard) or the encoder is down, re-promotion
     still proceeds — an `indexed=1` record without a vec row is
     FTS-reachable, which is exactly the state 1a already produces when
     consolidation-time embedding fails (`memory.py` inserts `embedded=0`
     with `indexed` defaulting to 1); vector coverage returns when backfill
     next runs. Demotion is reversible or it isn't demotion (§3.1).
   - **Backfill.** Records with `indexed=1 AND redacted=0 AND
     superseded_by IS NULL AND embedded=0` and an available encoder:
     embed with `kind="document"` (bounded batch per run,
     `memory.retention.backfill_batch` default 32), insert the vec row, set
     `embedded=1`. The `indexed=1` filter is load-bearing: demotion (pass 1
     of this same run) sets `embedded=0`, and an unfiltered backfill would
     immediately re-embed every just-demoted record — undoing the vec
     delete and violating trap #1 and the DoD's vec-rows ≤ indexed-records
     bound. Only indexed records get vectors; demoted/superseded/redacted
     records regain one solely by first becoming `indexed=1` again
     (re-promotion). Skip the pass entirely when embedding is disabled by
     the `embedding_meta` prefix/encoder mismatch guard (see Amendments) —
     log once, DEBUG.
3. **What counts as recall** (feeds re-promotion): `recall_raw` (WP1b.3)
   updates its one record's `retrieval_count`/`last_retrieved_at` —
   deliberately fetching a record's raw range is the strongest usage
   signal. `search_memory` updates NOTHING: its output is a catalog (id,
   score, flags, summary line) for choosing what to recall, and bumping
   every listed record would let a single broad query re-promote swaths of
   demoted records and inflate counts for marginal matches — distorting the
   scores this phase exists to make meaningful. This mirrors 1a's passive
   path, which bumps only memories admitted into the window
   (`memory.py:659-666`), not everything matched. Undercounting is
   fail-safe (passive retrieval keeps bumping indexed records);
   overcounting breaks demotion. Re-promotion is therefore reached by:
   `search_memory` surfaces the demoted record (flagged) → the agent
   `recall_raw`s it → the next job run re-promotes.

**Red tests** (`tests/test_memory_retention.py`, frozen clock):
- Young unretrieved record: exempt (grace). High-importance stale record:
  exempt (floor). Old, low-importance, never-retrieved: demoted, vec row
  gone, `embedded=0`, `memory` row and FTS intact — and the vec row is
  STILL gone after the same run's backfill pass with a live encoder
  (regression for the backfill `indexed=1` scope: demotion must not be
  undone by pass 3).
- Demoted record excluded from `before_agent_turn` retrieval; still found by
  `search_memory` (flagged).
- Demoted record whose stats are bumped via `recall_raw` re-promotes on the
  next job run (`indexed=1`, no inline embed); the same run's backfill pass
  restores its vec row when the encoder is up.
- Re-promotion with embedding disabled (`embedding_meta` mismatch):
  `indexed=1`, no vec row, no embed call, no error loop; the record
  surfaces via FTS retrieval.
- `search_memory` leaves access stats unchanged (regression against
  recall-count inflation).
- Job runs at startup with no traffic (regression for the push-based
  `on_idle` starvation — §3.1); a second startup within
  `memory.retention.interval` does NOT re-run it (`retention_meta.last_run`
  persists across restarts).
- Score function unit-tested at the boundaries (grace edge, floor edge,
  zero retrievals).

### WP1b.2 — Near-duplicate merge at consolidation

**Files:** `corvidae/memory.py` (extends the 1a consolidation task, step
between embed and insert)

1. After embedding the new candidate record: top-1 vec query among
   `indexed=1` records with the SAME `channel_id` — NOT the group scope
   that `_channel_scope` gives retrieval (`memory.py:755-765`). Rationale:
   channel groups grant shared *retrieval* (bootstrap-mapping §3.1's
   "optional YAML channel-group map for shared memory"), not shared
   ownership — merging across siblings would leave one channel's content
   reachable only through another channel's record (a compartment leak the
   moment group config changes), and msg-id ranges from different channels
   cannot be folded (`recall_raw` over a cross-channel range would
   interleave foreign rows). A near-dup living on a sibling channel is left
   alone — both stay indexed; cross-sibling crowding is bounded and a
   summarizing merge is a §6-gated refinement. If cosine similarity ≥
   `memory.dup_threshold` (default 0.95):
   - insert the new record as usual, then fold the old into it:
     `new.retrieval_count += old.retrieval_count`,
     `new.importance = max(new, old)`,
     `new.last_retrieved_at = max(new, old)` (NULL-aware — preserves the
     recency signal, though new's `created_at` dominates the retention
     score anyway), extend the range to
     `msg_id_start = min(old, new)`, `msg_id_end = max(old, new)` (the
     ranges share a channel by construction of the same-`channel_id` query
     above). Note the merged range spans intermediate same-channel messages
     belonging to other records, so `recall_raw` on a merged record replays
     more than the two merged ranges — acceptable: raw recall is verbatim
     context, not an exact record boundary;
   - mark the old: `superseded_by = new.id`, `indexed = 0`, `embedded = 0`,
     delete its vec row (`embedded=0` keeps the flag truthful — same
     rationale as demotion in WP1b.1; the backfill pass cannot re-embed it,
     being scoped to `indexed=1 AND superseded_by IS NULL`). Superseded
     records are terminal — the retention job skips them (already excluded
     by `superseded_by IS NULL` in WP1b.1).
2. No LLM call in the merge path — this is the mechanical first pass; a
   summarizing merge is a §6-gated refinement, not this phase.
3. When the vec extension is unavailable, skip dup detection entirely (FTS
   similarity is not a substitute; log at DEBUG). Likewise per record: when
   the NEW record's embedding is unavailable (embed failed, or embedding
   disabled by the meta guard), skip dup detection for that record and
   insert it with `embedded=0` exactly as 1a does — there is no vector to
   compare (log at DEBUG).

**Red tests** (`tests/test_memory_dedup.py`, stub embedder with controllable
similarity): near-identical consolidations → one indexed record carrying
summed stats and a superseded chain; sub-threshold pairs → both indexed;
a near-identical record on a sibling group channel is NOT merged (both stay
indexed — compartment boundary); a new record whose embedding failed skips
dup detection and inserts with `embedded=0`; `recall_raw` on a superseded
record still works (raw range intact).

### WP1b.3 — `message_log` FTS + memory tools

**Files:** `corvidae/tools/memory_tools.py` (new — `memory.py` is already
~1,000 lines; the "~500 lines" conditional has triggered), `corvidae/memory.py`
(the `message_fts` DDL + startup backfill task ONLY — schema stays
single-owner in `MemoryPlugin._ensure_schema` so the redact CLI can rely on
daemon-created schema, see WP1b.4), `pyproject.toml` (plugin entry point
under `[project.entry-points.corvidae]`:
`memory_tools = "corvidae.tools.memory_tools:MemoryToolsPlugin"`).

`MemoryToolsPlugin`: `depends_on = frozenset({"memory"})`, reaches the DB
and access-stat updates through `pm.get_plugin("memory")` (the funnel
precedent, `memory.py:643`); `register_tools` per `RuntimeSettingsPlugin`
(`tools/settings.py:63`).

1. **`message_fts`** (§4.11 — the search-tool surface, distinct from
   `memory_fts`): a standalone FTS5 table (not external-content — the
   indexed text is a JSON extraction, so content-sync triggers do the
   extraction):

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS message_fts USING fts5(content_text);
CREATE TRIGGER IF NOT EXISTS message_log_ai AFTER INSERT ON message_log BEGIN
    INSERT INTO message_fts(rowid, content_text)
    VALUES (new.id, coalesce(json_extract(new.message, '$.content'), ''));
END;
CREATE TRIGGER IF NOT EXISTS message_log_au AFTER UPDATE OF message ON message_log BEGIN
    DELETE FROM message_fts WHERE rowid = old.id;
    INSERT INTO message_fts(rowid, content_text)
    VALUES (new.id, coalesce(json_extract(new.message, '$.content'), ''));
END;
```

   The update trigger uses plain `DELETE`, NOT the FTS5 `'delete'` command:
   `message_fts` is a regular (non-external-content) FTS5 table, and the
   `'delete'` command is only accepted by contentless/external-content
   tables — with a regular table it makes every `UPDATE message_log SET
   message=...` (including the redact tombstone) abort with "SQL logic
   error" (verified empirically, SQLite 3.47.1). Plain `DELETE` works on
   regular FTS5 tables. (The 1a `memory_au` trigger correctly keeps the
   `'delete'` command — `memory_fts` IS external-content.)

   One-time backfill of pre-existing rows on start, batched, in a silent
   startup task, selecting `message_log` rows absent from the index
   (`WHERE id NOT IN (SELECT rowid FROM message_fts)`) — self-guarding and
   crash-safe: a rerun after a mid-backfill crash picks up only the missing
   rows instead of hitting FTS5 rowid uniqueness on re-insert (a row-count
   guard would do neither; verified empirically). Latent note:
   `json_extract(..., '$.content')` returns raw JSON if `content` is ever
   non-string; today `agent.py` persists string content for all roles, so
   no handling is needed yet. A second explicit note: `message_log` also
   holds `message_type='summary'` rows (compaction summaries,
   `persistence.py`) in the same id sequence; the triggers/backfill index
   them and `recall_raw`'s range fetch replays them like any other row —
   deliberate, not ad hoc: raw recall is verbatim context, not an exact
   record boundary, and summaries must be as searchable (and, under
   redaction, as tombstonable) as the dialog they condense. The update
   trigger is what makes the redact
   cascade automatic on the message side — note it in WP1b.4.
2. **Tools** (registered via the standard `register_tools` hook; access-stat
   semantics per WP1b.1.3 — `recall_raw` bumps, `search_memory` does not).

   **Compartment scoping — both tools enforce the calling channel's
   `_channel_scope`** (channel + configured group siblings,
   `memory.py:755-765` — the same scope Phase 1a's passive retrieval
   uses). Rationale: the tools are the *active* analogue of passive
   retrieval and share its compartment (bootstrap-mapping §3.1); they are
   deliberately broader in *status* (they see demoted, superseded, and
   redacted-tombstone records — "remember harder") but never broader in
   *compartment* — otherwise an agent on channel B could enumerate and
   raw-recall channel A's memories, the leak class this plan's own
   cross-channel-merge rejection (WP1b.2) exists to prevent. The calling
   channel comes from `_ctx.channel` (ToolContext injection — the
   `RuntimeSettingsPlugin` precedent, `tools/settings.py:85`); when
   `_ctx.channel is None` the tools return an error string — never fall
   back to unscoped.
   - `search_memory(query: str, channel: str | None = None,
     tags: list[str] | None = None, after: str | None = None,
     before: str | None = None, include_demoted: bool = True) -> str` —
     `memory_fts MATCH` plus column filters, always restricted to
     `channel_id IN _channel_scope(calling channel)`; the `channel`
     argument narrows WITHIN that scope (a value outside scope returns an
     error string naming the visible channels, not silence — silence would
     read as "no memories exist"). Returns a compact numbered list
     (id, band-less score, age, flags `[demoted]`/`[superseded]`, summary),
     token-capped. `include_demoted=False` restricts results to
     `indexed=1` records — hiding `[demoted]`, `[superseded]`, and
     redacted-tombstone records alike (all are `indexed=0`); the default
     `True` is the "remember harder" behavior. Does NOT update access
     stats (WP1b.1.3). `after`/`before`
     are ISO-8601 (`YYYY-MM-DD` or a full timestamp; values without a
     timezone are UTC), parsed with `datetime.fromisoformat` and compared
     against `created_at` as `after ⇒ created_at >= t`,
     `before ⇒ created_at < t`; unparseable values return an error string,
     not an exception. This is Persyn §4.6's management surface (§3.1).
   - `recall_raw(memory_id: int, max_tokens: int = 1000) -> str` — the
     "remember harder" tool (§3.1): rejects (error string, no stat bump)
     any record whose `channel_id` is outside the calling channel's
     `_channel_scope`; otherwise fetch the record's raw range from
     `message_log` with `WHERE channel_id = record.channel_id AND id
     BETWEEN msg_id_start AND msg_id_end` — the `channel_id` predicate is
     load-bearing: `message_log` ids are one global AUTOINCREMENT sequence
     across all channels, so with concurrent channels the numeric range
     contains interleaved foreign-channel rows, and an unfiltered
     `BETWEEN` would replay other channels' dialog. Rows are formatted
     `role: content` — sender is NOT in the persisted JSON (`agent.py:249`
     keeps only `role`/`content`), so per-line sender attribution is
     unimplementable; the record-level `participants` field supplies it in
     a header line (`[memory {id}: messages {start}–{end}, participants:
     {...}]`) — truncated to the token cap with an explicit truncation
     marker. Bumps the record's access stats (WP1b.1.3). Works on demoted,
     superseded, and (tombstoned) redacted ranges alike.

**Red tests** (`tests/test_memory_tools.py`): keyword search finds seeded
content with filters honored (including `after`/`before` date bounds);
demoted flagged; `recall_raw` returns verbatim `role: content` dialog for
the range with the participants header and respects the cap; with two
channels' rows interleaved in `message_log`, `recall_raw` returns ONLY the
record's own channel's rows (global-id regression); `search_memory` from
channel B never lists channel A's memories and `recall_raw` of an
out-of-scope id returns an error without bumping stats, while a
group-sibling's record IS visible and recallable (scope tests);
`search_memory(channel=<out-of-scope>)` returns the error string;
`recall_raw` bumps access stats while `search_memory` leaves them
unchanged; backfill indexes pre-existing rows exactly once and a rerun
after a simulated partial backfill completes the remainder without a
uniqueness error.

### WP1b.4 — `redact` CLI

**New file:** `corvidae/commands/redact.py`; entry point
`[project.entry-points."corvidae.commands"] redact = "corvidae.commands.redact:redact_command"`

```
corvidae redact --db sessions.db message <id> [<id2>...]   # message_log rows
corvidae redact --db sessions.db memory <memory_id>        # a record + its raw range
corvidae redact --db sessions.db range <start_id> <end_id> # message_log id range
```

Behavior (own connection; trap #6 discipline; every step in one short
transaction per row batch):
0. **Schema-presence probe** — the DB may predate 1b (no `message_fts`/
   triggers) or even 1a (no `memory` tables); the daemon is what creates
   them. Check `sqlite_master` for each surface before touching it, and
   SKIP what is absent with a printed notice — never abort, never create
   schema. Missing `message_fts` is safe to skip because the WP1b.3
   triggers/backfill read *current* row content: when the daemon later
   builds the index it indexes the tombstone, never the secret. Missing
   `memory` table → skip the memory cascade (notice); missing `memory_vec`
   → skip vec deletion. The verification pass (step 3) runs only against
   the FTS surfaces that exist. The CLI must not create schema itself:
   schema is daemon-owned (single owner — WP1b.3), and a partial
   CLI-created schema would diverge from the daemon's (`embedding_meta`
   and vec setup need config the CLI does not have).
1. **Message tombstone:** rewrite the `message` JSON — keep `role` and
   structural keys, replace `content` with
   `"[redacted by operator {ISO-date}]"`, and drop `tool_calls`/tool
   payload fields. Row id and timestamp unchanged (trap #8). The `memory`
   form resolves its record's raw range to rows with `WHERE channel_id =
   record.channel_id AND id BETWEEN msg_id_start AND msg_id_end` — the
   same load-bearing channel predicate as `recall_raw` (WP1b.3):
   `message_log` ids are one global sequence, so an unfiltered `BETWEEN`
   would tombstone interleaved foreign-channel rows (destructive
   over-redaction of unrelated channels' logs), and step 2's per-channel
   cascade would then tombstone *those* channels' memories too. The WP1b.3
   update trigger propagates to `message_fts` automatically — but assert it
   in tests, not in faith (trap #2).
2. **Memory cascade:** for every `memory` row whose
   `[msg_id_start, msg_id_end]` contains a redacted message's id AND whose
   `channel_id` equals that message's `channel_id` (i.e. intersect
   per-channel, not on the bare numeric range — `message_log` ids are
   global across channels, so a foreign-channel memory's range can
   numerically contain a redacted id it has no relation to; tombstoning it
   would be over-redaction): `summary = '[redacted by operator]'` via
   UPDATE (the 1a `memory_au` trigger syncs `memory_fts`), `redacted=1`,
   `indexed=0`, `embedded=0`, delete its `memory_vec` row (`embedded=0` for
   flag truthfulness, per WP1b.1; the `indexed=1`-scoped backfill cannot
   re-embed it). The `range` form applies the same per-channel rule to
   each row in the id range.
3. **Verification pass** (part of the command, printed): after the cascade,
   run `memory_fts` and `message_fts` MATCH queries for a sample token of
   the redacted text and report zero hits — the cascade proves itself on
   every invocation (§3.1: "the cascade must name FTS or the redacted text
   remains keyword-searchable").
4. `--dry-run` prints the affected row/record counts without writing.

**Red tests** (`tests/test_redact.py`, direct function invocation against a
temp DB): tombstone preserves row count/ids/roles; both FTS surfaces return
zero hits for redacted text afterward; vec rows gone and `embedded=0`;
same-channel intersecting memory records tombstoned; non-intersecting
untouched; a foreign-channel memory whose range merely numerically
contains a redacted id is untouched (global-id regression); `redact
memory` on an interleaved two-channel log tombstones only the record's
own channel's rows within its range, leaving the foreign channel's
messages AND memories untouched (global-id regression for the memory
form); dry-run writes nothing;
non-WAL journal mode aborts with a clear error; against a pre-1b DB
(`message_log` only) the message tombstones are written, skip notices
printed, exit code 0 — and running the WP1b.3 backfill afterwards indexes
the tombstone text, not the redacted secret.

### WP1b.5 — Retrieval benchmarks and eval wiring

**Files:** `tests/fixtures/`, `tests/evals/`, `scripts/eval_memory.py`

1. **Fixtures:** grow the Phase 0 seed into at least two operator-authored
   fixtures (§6 — labels are hand-written, not generated): one
   general-recall set (≥15 memories, ≥8 queries incl. 2 with NO relevant
   memory — the "no memory of that" cases), one contradiction-bearing set
   (two conflicting records for the same claim) reserved for the §3.1
   contradiction-annotation behavior — mark its assertions `xfail` with a
   pointer to the mapping section until that feature lands.
2. **CI benchmark** (`tests/evals/test_retrieval_benchmark.py`,
   deterministic stub embedder): recall@5 and MRR floors over the
   general-recall fixture, plus the negative cases asserting empty
   admission. Floors start at the measured value minus a small margin —
   they are regression trips, not aspirations; raising them is a §6
   activity.
3. **Live wiring:** `scripts/eval_memory.py` gains `--live` — spins up
   MemoryPlugin against a scratch DB and a real llama-server
   (base-url/model/api-key args, mirroring `eval_compaction.py`), ingests
   the fixture conversation through the real consolidation path, runs the
   queries through the real retrieval path, reports the deterministic
   metrics and per-stage token cost from `usage_log` ("recall at a fixed
   token budget" — §6's currency). The reserved `--judge` mode scores
   consolidation summaries for epistemic-framing preservation
   (LLM-judged, out-of-band only).

**Red tests:** the CI benchmark itself (write it against 1a's retrieval
before implementing anything else in this phase — it should pass already;
its value is the floor).

## Non-goals

- Contradiction annotation at retrieval time (§3.1) — fixture reserved,
  feature is Phase 2+ territory once the appraisal/critique machinery
  exists to consume it.
- Semantic facts, trust, sensitivity filters — Phases 4–5.
- Encoder-migration tooling (bulk re-embed command) — the `embedded=0`
  backfill path and `embedding_meta` guard (which, post
  embedding-prefixes, also covers `document_prefix`/`query_prefix`, with
  config revert as the only mismatch remediation) are sufficient until an
  encoder change actually happens; write the runbook note in
  `docs/design.md` instead ("text is canonical; embeddings are a
  rebuildable cache").
- No changes to retrieval scoring or budgets — tuning belongs to the §6
  benchmarks, with Phase 0's cost data in hand.

## Definition of done

- All red tests green; full suite passes; CI benchmark enforcing floors.
- Live check: seed a conversation, let retention + dedup jobs run
  (startup-trigger them), verify: a stale trivial memory demotes and stops
  surfacing passively but `search_memory` still finds it; `recall_raw`
  returns the verbatim dialog; `corvidae redact` removes a planted secret
  from both FTS surfaces and passive retrieval, with the verification pass
  printing zero hits.
- §7 row 1b acceptance: hedging + remember-harder (prompt fragment + tools
  demonstrated live), bounded store (vec index row count ≤ indexed
  records; demotion demonstrably bounds it).
- `docs/configuration.md` documents all `memory.retention.*` and
  `memory.dup_threshold` keys; `docs/design.md` gains the redaction runbook
  paragraph; `docs/plugin-guide.md` notes the tool surface including its
  channel scoping.

## Design fix report (2026-07-07)

Amendment pass resolving the pre-implementation audit
(`phase-1b-assessment.md`, verdict MEDIUM): all 5 important findings, all
3 drift items, and the cosmetic items. Evidence cited by the audit was
re-verified against the working tree before each fix.

**Important 1 — re-promotion vector-source contradiction.** Resolved by
setting `embedded=0` at demotion and making re-promotion flip `indexed=1`
only, deferring the vector to the backfill pass (WP1b.1 step 2). Rationale:
`memory_vec` is the only vector storage, so post-demotion the flag's
documented meaning ("0 = pending/failed, backfillable", `memory.py:205`)
demands 0; re-promotion then composes with the one embedding path that
already handles prefixes, batching, and the disabled guard, instead of
duplicating it. Chosen over unconditional re-embed on re-promotion because
"text is canonical; embeddings are a rebuildable cache" (§3.1) — rebuild
belongs to backfill.

**Important 2 — re-promotion under disabled embedding.** Dissolved by the
finding-1 design: re-promotion never embeds, so there is no re-embed call
to guard. FTS-only re-promotion (`indexed=1`, no vec row) is explicitly
sanctioned — it is the same state 1a produces when consolidation-time
embedding fails (`memory.py:541`) — with a red test covering the
disabled-guard case (no embed call, no error loop, FTS-reachable).

**Important 3 — cross-channel dup merge.** Resolved by scoping the dup
query to the same `channel_id`, not `_channel_scope` (WP1b.2). Sibling
near-dups are skipped, both stay indexed. Rationale: groups grant shared
retrieval, not shared ownership (§3.1's channel-group map); a cross-channel
merge would be a compartment leak under group-config change, and
cross-channel msg-id ranges cannot be folded without `recall_raw`
interleaving foreign rows. Red test added for the sibling-skip.

**Important 4 — recall-stat inflation.** Resolved in WP1b.1 step 3:
`recall_raw` bumps its record; `search_memory` bumps nothing. Rationale:
mirrors 1a's admitted-only bump (`memory.py:659-666`) — the search list is
a catalog, not consumed memory; bumping listed results would let one broad
query re-promote swaths of demoted records. Undercounting is fail-safe,
overcounting breaks demotion. Re-promotion path is documented as
search-then-recall_raw; red tests updated on both WPs.

**Important 5 — redact CLI pre-1b schema.** Resolved by a schema-presence
probe (WP1b.4 step 0): probe `sqlite_master`, skip absent surfaces with
printed notices, never abort, never create schema (schema is daemon-owned;
a partial CLI copy would diverge). Skipping `message_fts` is proven safe
because triggers/backfill index current content — post-redaction backfill
indexes the tombstone. Red test added for the pre-1b DB case including the
backfill-after-redact assertion.

**Drift 1 — module size.** WP1b.1 moves scoring + the retention pass to a
new `corvidae/retention.py` (plain module; MemoryPlugin keeps hook wiring);
WP1b.3's conditional is made unconditional: tools live in
`corvidae/tools/memory_tools.py` as `MemoryToolsPlugin` (entry point
specified), with only the `message_fts` DDL/backfill staying in `memory.py`
so schema keeps a single owner (which finding 5's fix depends on). WP1b.2
stays in `memory.py` — it is a bounded step inside the existing
consolidation task.

**Drift 2 — goal_tracker.py vestigial.** "Read first" now points at
`RuntimeSettingsPlugin` (`tools/settings.py:63`) and warns off
`goal_tracker.py` explicitly.

**Drift 3 — sender not persisted.** `recall_raw` now specifies
`role: content` formatting (verified: `agent.py:249` persists only
`role`/`content`), with record-level `participants` surfaced in a header
line to preserve attribution.

**Cosmetic.** "before implementing nothing" fixed; entry-point value
quoted; `after`/`before` defined as ISO-8601, naive-as-UTC, `>=`/`<`
semantics with error-string handling; retention rate limit persisted in a
one-row `retention_meta` table honored by the startup trigger (crash-loop
safe, zero-traffic guarantee preserved — trap #5 updated); per-record dup
skip on new-record embedding failure stated explicitly with a red test.
The 2026-07-06 amendments note and trap #1 were updated for consistency
with the finding-1/2 design (backfill is the sole embed call site;
demotion includes `embedded=0`).

## Design review report (2026-07-07)

Fresh-context review of the amended plan against
`phase-1b-assessment.md` and the working tree (memory.py, llm.py,
persistence.py, agent.py, tools/settings.py, pyproject.toml,
plans/bootstrap-mapping.md). Scope: (a) is each assessment finding
resolved in the plan body, (b) is the amended plan internally
consistent, (c) do the cited code behaviors hold and are the designs
implementable.

### Assessment-resolution verification — all resolved in the body

- **Important 1 (vector source):** trap #1 and WP1b.1 step 2 now demote
  with `indexed=0`, `embedded=0`, vec delete; re-promotion flips
  `indexed=1` only and defers vectors to backfill. `embedded` column
  comment verified (`memory.py:205`); `memory_vec` as sole vector
  storage verified (`memory.py:545-549`). Red tests updated to match.
- **Important 2 (disabled embedding):** re-promotion never embeds;
  FTS-only `indexed=1`-without-vec sanctioned and correctly equated with
  1a's embed-failure state (`memory.py:541`: `embedded=0`, `indexed`
  defaults 1). Red test present.
- **Important 3 (cross-channel merge):** dup query scoped to same
  `channel_id`, sibling near-dups skipped, sibling red test added.
  `_channel_scope` group behavior verified (`memory.py:755-765`).
- **Important 4 (stat inflation):** WP1b.1 step 3 — `recall_raw` bumps
  its one record, `search_memory` bumps nothing; mirrors the 1a
  admitted-only bump (verified, `memory.py:659-666`). Regression red
  tests present in WP1b.1 and WP1b.3.
- **Important 5 (pre-1b redact):** WP1b.4 step 0 probes `sqlite_master`,
  skips absent surfaces with notices, never creates schema; pre-1b red
  test incl. backfill-after-redact present.
- **Drift 1–3 and all six cosmetic items:** verified fixed in the body
  (retention.py / memory_tools.py split with DDL single-owner in
  memory.py; goal_tracker warned off, `RuntimeSettingsPlugin`
  `register_tools` at `tools/settings.py:63` verified; `role: content`
  + participants header, `agent.py` persists role/content only —
  verified; garble fixed; entry-point values quoted and both entry-point
  groups match `pyproject.toml`; ISO-8601 semantics defined;
  `retention_meta.last_run` persisted and honored by the startup
  trigger; per-record dup skip stated with red test).

Other citations spot-checked and holding: retrieval filters
`indexed=1 AND redacted=0` in both paths (`memory.py:784`, `811`);
`embedding_meta` prefix guard (`memory.py:960-995`); `embed(texts,
kind)` required-arg prefixing (`llm.py:230-256`); `memory_au` is `AFTER
UPDATE OF summary` on an external-content table where the FTS `'delete'`
command is valid (`memory.py:243-248`); WAL default and `message_log`
DDL (`persistence.py`); §3.1 retention/redact language and §7 row 1b in
`plans/bootstrap-mapping.md` (lines 330ff, 1214).

### New findings

**Critical:** none.

**Important:**

1. **WP1b.3 `message_log_au` trigger DDL is invalid SQL.** The FTS5
   `'delete'` command (`INSERT INTO message_fts(message_fts, rowid,
   content_text) VALUES ('delete', ...)`) is only accepted by
   contentless/external-content FTS5 tables; `message_fts` is a regular
   table (the plan says so explicitly). Verified empirically (SQLite
   3.47.1): the trigger creates fine but any `UPDATE message_log SET
   message=...` then aborts with `SQL logic error` — i.e., the redact
   tombstone itself fails. Fix is `DELETE FROM message_fts WHERE rowid =
   old.id;` (verified working). The WP1b.4 red tests would catch this,
   but the plan ships broken literal DDL for the phase's safety-critical
   path.
2. **Backfill pass scope contradicts demotion.** WP1b.1's backfill is
   specified as "records with `embedded=0`" with no `indexed` filter.
   Demotion (pass 1 of the same run) sets `embedded=0` — so pass 3 as
   written immediately re-embeds every just-demoted record, undoing the
   vec delete, contradicting trap #1 ("remove the vec row") and the DoD
   bound ("vec index row count ≤ indexed records"). The demotion red
   test ("vec row gone" after a job run) would trip the naive reading,
   but the body text is self-contradictory. Backfill must be scoped
   `indexed=1 AND redacted=0 AND superseded_by IS NULL AND embedded=0`.
3. **`message_log` ids are global; range consumers need a channel
   predicate.** `message_log` is one table with a global AUTOINCREMENT
   id shared by all channels; consolidation stores per-channel endpoint
   ids over that global sequence, so with concurrent channels a record's
   `[msg_id_start, msg_id_end]` numerically contains interleaved
   foreign-channel rows. WP1b.3's `recall_raw` ("fetch the record's
   msg_id_start..msg_id_end rows from message_log verbatim") specifies
   no `channel_id` filter — it would replay other channels' dialog, the
   exact compartment-leak class the Important-3 fix exists to prevent.
   WP1b.4's memory cascade ("every memory row whose range intersects the
   redacted ids") has the dual problem: it tombstones foreign-channel
   memories whose ranges merely numerically intersect (over-redaction,
   fail-safe but destructive). Both need `channel_id` predicates
   (recall_raw: `WHERE channel_id = record.channel_id AND id BETWEEN
   ...`; redact cascade: intersect per redacted row's channel).
4. **Tool-surface compartment scoping unspecified.** `search_memory`
   takes `channel: str | None = None` as a free filter and `recall_raw`
   takes any `memory_id`, with no statement of scope enforcement. As
   specified, an agent on channel B can enumerate and raw-recall channel
   A's memories — inconsistent with §3.1's retrieval
   compartmentalization and with the compartment-boundary rationale this
   amendment itself uses to reject cross-channel merges. The plan must
   state whether the tools enforce `_channel_scope` of the calling
   channel (recommended: yes — `channel` filters within scope, and
   `recall_raw` rejects out-of-scope ids) or deliberately don't, and
   why.

**Cosmetic:**

- WP1b.2 fold: `msg_id_end` disposition unstated (implicitly new's —
  state it, or `max(old, new)`); `last_retrieved_at` not folded
  (`max()` would preserve the recency signal, though new's `created_at`
  dominates the score anyway). Also worth one sentence: the merged
  range `[min(start), new.end]` spans intermediate same-channel
  messages belonging to other records, so `recall_raw` on a merged
  record replays more than the two merged ranges.
- WP1b.2 supersede path and WP1b.4 memory cascade delete the vec row
  without setting `embedded=0` — the flag-truthfulness rationale of
  fix 1 applies; harmless once backfill is scoped per new finding 2,
  but inconsistent as written.
- WP1b.3 backfill guard "row-count comparison" is fragile: a crash
  mid-backfill leaves a count mismatch, and a rerun that re-inserts
  from the start hits FTS5 rowid uniqueness (`constraint failed`,
  verified). Backfill should select `message_log` rows absent from
  `message_fts` (e.g., `WHERE id NOT IN (SELECT rowid FROM
  message_fts)`), which also makes the count guard unnecessary.
- `json_extract(new.message, '$.content')` indexes raw JSON if
  `content` is ever non-string; today `agent.py` persists string
  content for all roles, so this is a latent note only.

### Gate

**FAIL** — four important findings (no criticals). Findings 1–3 are
mechanical to fix; finding 4 is a small design decision that should be
made in the plan, not ad hoc mid-implementation. The architecture and
the assessment-driven amendments are otherwise sound and fully
reflected in the body.

## Design fix report #1 (2026-07-07)

Tranche #1 pass resolving the 2026-07-07 design review: all 4 important
and all 4 cosmetic findings. Fixes verified against the working tree
(`persistence.py` `message_log` DDL — global AUTOINCREMENT id +
`channel_id`; `memory.py` `_channel_scope` at 755-765, `memory` DDL with
`channel_id`/`embedded`; `tool.py` ToolContext `_ctx.channel` injection;
`tools/settings.py:85` precedent) and, where SQL behavior was at issue,
empirically against SQLite 3.47.1 via Python `sqlite3`.

**Important 1 — invalid `message_log_au` DDL.** WP1b.3's update trigger
now uses `DELETE FROM message_fts WHERE rowid = old.id` and explains why
the FTS5 `'delete'` command is wrong for a regular (non-external-content)
table. Re-verified empirically: the original DDL aborts every
`message_log` UPDATE with "SQL logic error"; the plain-DELETE trigger
tombstones correctly (secret unsearchable, tombstone indexed). The 1a
`memory_au` trigger keeps the `'delete'` command — `memory_fts` is
external-content, where it is required.

**Important 2 — backfill undoing demotion.** WP1b.1's backfill pass is
now scoped `indexed=1 AND redacted=0 AND superseded_by IS NULL AND
embedded=0`, with the rationale spelled out (demotion in pass 1 of the
same run sets `embedded=0`; unfiltered backfill would re-embed it,
contradicting trap #1 and the DoD vec-rows ≤ indexed-records bound).
Demotion red test extended to assert the vec row stays gone after the
same run's backfill with a live encoder.

**Important 3 — global msg-ids need channel predicates.** `recall_raw`
now fetches `WHERE channel_id = record.channel_id AND id BETWEEN
msg_id_start AND msg_id_end` (rationale in WP1b.3: the global id sequence
interleaves foreign-channel rows). The WP1b.4 memory cascade now
intersects per-channel: a memory is tombstoned only when a redacted
message's id falls in its range AND the channels match; the `range` form
applies the same rule per row. Red tests added on both sides (interleaved
two-channel recall returns only own-channel rows; foreign-channel
numerically-containing memory untouched by redact).

**Important 4 — tool compartment scoping.** WP1b.3 now specifies that
both tools enforce the calling channel's `_channel_scope` (channel +
group siblings — identical to 1a passive retrieval), sourced from
`_ctx.channel` (ToolContext injection, `tools/settings.py:85` precedent);
`_ctx.channel is None` is an error, never unscoped. `search_memory`'s
`channel` arg narrows within scope and errors (naming visible channels)
outside it; `recall_raw` rejects out-of-scope ids without a stat bump.
Rationale recorded: tools are broader than passive retrieval in *status*
(demoted/superseded/redacted visible) but never in *compartment* —
consistent with §3.1 and the plan's own cross-channel-merge rejection.
Trap #1 and the DoD docs bullet updated to match; scope red tests added
(sibling visible, foreign channel not).

**Cosmetic.** (1) WP1b.2 fold now states `msg_id_end = max(old, new)`,
folds `last_retrieved_at = max(old, new)` (NULL-aware), and notes that
the merged range replays intermediate same-channel messages. (2) The
WP1b.2 supersede path and WP1b.4 cascade now set `embedded=0` alongside
the vec delete (flag truthfulness; the `indexed=1`-scoped backfill cannot
re-embed either). (3) The `message_fts` backfill selects rows absent from
the index (`id NOT IN (SELECT rowid FROM message_fts)`) instead of a
row-count guard — crash-safe rerun verified empirically (no rowid
uniqueness error); WP1b.3 red test extended with the partial-backfill
rerun. (4) The `json_extract` non-string-content caveat is recorded as a
latent note in WP1b.3 (today `agent.py` persists string content only).

## Design re-review report #1 (2026-07-07)

Fresh-context full re-review after Design fix report #1 (not a
spot-check of the four fixes). Scope: (a) each 2026-07-07 review finding
genuinely resolved in the plan BODY, (b) new inconsistencies introduced
by the fixes, (c) load-bearing code citations re-verified against the
working tree (`memory.py`, `persistence.py`, `tool.py`,
`tools/settings.py`, `agent.py`, `pyproject.toml`), (d) SQL behavior
re-verified empirically (SQLite 3.47.1 via Python `sqlite3`).

### Resolution verification — all four important + four cosmetic fixes present in the body

- **Important 1 (trigger DDL):** WP1b.3's `message_log_au` now uses
  `DELETE FROM message_fts WHERE rowid = old.id`. Re-verified
  empirically: this trigger tombstones correctly (secret 0 FTS hits,
  tombstone indexed); the FTS5 `'delete'`-command variant on a regular
  FTS5 table aborts every `message_log` UPDATE with "SQL logic error".
  The contrast note (1a `memory_au` keeps `'delete'` — external-content)
  matches `memory.py` (`MEMORY_FTS_UPDATE_TRIGGER`, external-content
  `content='memory'`).
- **Important 2 (backfill scope):** WP1b.1 pass 3 is scoped
  `indexed=1 AND redacted=0 AND superseded_by IS NULL AND embedded=0`,
  with the load-bearing rationale inline and a regression red test
  (vec row still gone after same-run backfill with a live encoder).
  Pass ordering (demote → re-promote → backfill) is self-consistent:
  just-demoted records (`indexed=0`) are excluded; just-re-promoted
  records (`indexed=1, embedded=0`) are eligible, as the plan claims.
- **Important 3 (channel predicates):** `recall_raw` specifies
  `WHERE channel_id = record.channel_id AND id BETWEEN ...`; the WP1b.4
  cascade intersects per redacted row's channel, and the `range` form
  applies the same rule. Global-id premise re-verified
  (`persistence.py` `message_log` DDL: one AUTOINCREMENT id +
  `channel_id`). Red tests present on both sides. (But see new
  finding 1 — a third range-dereference site was missed.)
- **Important 4 (tool scoping):** WP1b.3 specifies both tools enforce
  the calling channel's `_channel_scope`, sourced from `_ctx.channel`;
  `None` → error string, never unscoped; `channel` arg narrows within
  scope and errors (naming visible channels) outside it; `recall_raw`
  rejects out-of-scope ids with no stat bump. Citations verified:
  `ToolContext.channel` injection exists (`tool.py` — `channel:
  Channel | None`, injected at dispatch), the `settings.py:85`
  precedent (`channel = _ctx.channel`; None → error string) holds, and
  `_channel_scope` (`memory.py:755-765`) behaves as described. Trap #1
  and the DoD docs bullet are consistent; scope red tests present.
- **Cosmetic 1–4:** all present — `msg_id_end = max(old,new)` and
  NULL-aware `last_retrieved_at` fold with the merged-range replay
  note; `embedded=0` on the WP1b.2 supersede path and WP1b.4 cascade;
  `NOT IN`-based `message_fts` backfill (crash-safe rerun re-verified
  empirically — no rowid uniqueness error, no double-index) with the
  partial-backfill red test; `json_extract` latent note recorded.

Cross-consistency of the fixed body checked: retention-pass filters
compose (superseded records excluded from re-promotion and backfill;
redacted records excluded from re-promotion, so a `recall_raw` bump on
a redacted record can never resurrect it); traps #1/#2/#5/#6/#8, the
2026-07-06 amendments note, red-test lists, and the DoD all match the
fixed designs; entry-point groups match `pyproject.toml`
(`corvidae.commands`, `corvidae`); retrieval filters
`indexed=1 AND redacted=0` and the prefix-aware `embedding_meta` guard
re-verified in `memory.py`.

### New findings

**Critical:** none.

**Important:**

1. **Redact `memory <memory_id>` form dereferences a msg-id range with
   no channel predicate.** The Important-3 fix covered `recall_raw` and
   the message→memory cascade direction, but the CLI's `memory` form
   ("a record + its raw range") must resolve the record's
   `[msg_id_start, msg_id_end]` to `message_log` rows to tombstone —
   the third range-dereference site, and WP1b.4 never states its scope.
   As written, an unfiltered `BETWEEN` tombstones interleaved
   foreign-channel rows (destructive over-redaction of unrelated
   channels' logs — the exact class the plan's own rationale names),
   and step 2's per-channel cascade then tombstones *those* channels'
   memories too, propagating the damage. Fix is mechanical: the memory
   form tombstones `WHERE channel_id = record.channel_id AND id BETWEEN
   msg_id_start AND msg_id_end`; add a red test (interleaved
   two-channel log, `redact memory` leaves the foreign channel's rows
   and memories untouched).

**Cosmetic:**

- `search_memory`'s `include_demoted: bool = True` parameter is in the
  signature but its behavior is never specified — presumably
  `False` filters `[demoted]` records out of results; whether it also
  hides `[superseded]` (and tombstoned-redacted) records is unstated.
  One sentence would settle it.
- `message_log` also holds `message_type='summary'` rows (compaction
  summaries, `persistence.py`) in the same id sequence. `recall_raw`'s
  range fetch and the `message_fts` triggers/backfill index and replay
  them as if dialog. Likely acceptable under the plan's "raw recall is
  verbatim context, not an exact record boundary" stance, but worth an
  explicit sentence (or a `message_type != 'summary'` filter decision)
  so implementation doesn't decide ad hoc.

### Gate

**FAIL** — one important finding (no criticals). The finding is
mechanical and tightly localized to WP1b.4's `memory` form; everything
else, including all eight prior fixes and their knock-on consistency,
holds. One more fix → re-review cycle should close it.

## Design fix report #2 (2026-07-07)

Tranche #2 pass resolving Design re-review report #1: the 1 important and
2 cosmetic findings.

**Important 1 — `redact memory` range dereference unscoped.** WP1b.4
step 1 now states that the `memory` form resolves its record's raw range
with `WHERE channel_id = record.channel_id AND id BETWEEN msg_id_start
AND msg_id_end`, with the same load-bearing global-id rationale as the
two sibling fixes (`recall_raw`, WP1b.3; the message→memory cascade,
WP1b.4 step 2), including the damage-propagation note (unfiltered
`BETWEEN` tombstones interleaved foreign-channel rows, and step 2's
per-channel cascade then tombstones those channels' memories). Red test
added to WP1b.4: interleaved two-channel log, `redact memory` tombstones
only the record's own channel's rows, foreign channel's messages and
memories untouched. DoD/acceptance text reviewed: no DoD sentence
references the memory form's range dereference, so no DoD change was
needed.

**Cosmetic 1 — `include_demoted` unspecified.** WP1b.3's `search_memory`
now states: `include_demoted=False` restricts results to `indexed=1`
records, hiding `[demoted]`, `[superseded]`, and redacted-tombstone
records alike (all `indexed=0`); the default `True` is the "remember
harder" behavior.

**Cosmetic 2 — `message_type='summary'` rows.** WP1b.3 step 1 now
records the decision explicitly: the `message_fts` triggers/backfill
index summary rows and `recall_raw`'s range fetch replays them like any
other row — deliberate under the plan's "raw recall is verbatim context,
not an exact record boundary" stance, and summaries must be as
searchable (and tombstonable under redaction) as the dialog they
condense.

Nothing unresolved.

## Design re-review report #2 (2026-07-07)

Fresh-context re-review after Design fix report #2. Scope: (a) the
re-review-#1 findings genuinely resolved in the plan body, (b) no new
inconsistencies introduced by fix #2, (c) a final sweep for any other
msg-id range-dereference site lacking a channel predicate, plus a
consistency pass over the changed sections. New code citation
introduced by fix #2 re-verified against the working tree.

### Resolution verification — all three findings resolved in the body

- **Important 1 (`redact memory` range dereference):** WP1b.4 step 1
  now states the `memory` form resolves its record's raw range with
  `WHERE channel_id = record.channel_id AND id BETWEEN msg_id_start AND
  msg_id_end`, carrying the same global-id rationale as the two sibling
  sites and the damage-propagation note (unfiltered `BETWEEN` would
  tombstone interleaved foreign-channel rows, and step 2's per-channel
  cascade would then tombstone those channels' memories). The matching
  red test is present in WP1b.4: interleaved two-channel log, `redact
  memory` tombstones only the record's own channel's rows within its
  range, foreign channel's messages AND memories untouched. The fix
  report's claim that no DoD sentence needed changing is correct — the
  DoD references `corvidae redact` only generically.
- **Cosmetic 1 (`include_demoted`):** WP1b.3 now specifies
  `include_demoted=False` restricts results to `indexed=1` records,
  hiding `[demoted]`, `[superseded]`, and redacted-tombstone records
  alike (all `indexed=0`), with `True` as the "remember harder"
  default. Consistent with WP1b.1 (demotion → `indexed=0`), WP1b.2
  (supersede → `indexed=0`), and WP1b.4 (redact → `indexed=0`).
- **Cosmetic 2 (summary rows):** WP1b.3 step 1 records the decision
  explicitly (index and replay summary rows; deliberate under the
  verbatim-context stance; tombstonable under redaction). The factual
  premise re-verified against `persistence.py`: compaction summaries
  are inserted into the same `message_log` table (`message_type =
  'summary'`), sharing the global id sequence and carrying
  `channel_id` — so the triggers, backfill, `recall_raw` range fetch,
  and redact tombstone all handle them through the existing per-channel
  machinery with no special casing.

### Range-dereference sweep — no unscoped sites remain

Every place the plan turns a `[msg_id_start, msg_id_end]` range into
`message_log` rows now carries a channel predicate or is same-channel
by construction:

- WP1b.3 `recall_raw`: `WHERE channel_id = record.channel_id AND id
  BETWEEN ...` (with red test).
- WP1b.4 step 1 `memory` form: same predicate (with red test — fix #2).
- WP1b.4 step 2 message→memory cascade: per-channel intersection (with
  red test); the `range` form applies the same rule per row.
- WP1b.4 `range`/`message` forms' message tombstones: operator-supplied
  literal `message_log` ids — exact addressing, not a range dereference
  from a record; no predicate needed.
- WP1b.2 merged ranges: same-channel by construction of the
  same-`channel_id` dup query, stated inline; the merged-range replay
  note covers the intermediate-rows consequence.

### New findings

**Critical:** none.

**Important:** none.

**Cosmetic:** none. (One observation, no action needed: `search_memory`
lists flags `[demoted]`/`[superseded]` but no `[redacted]` flag —
harmless, since a redacted record's summary IS the tombstone text
`'[redacted by operator]'`, which is self-labeling in the result list.)

### Gate

**PASS** — zero critical, zero important. All re-review-#1 findings are
resolved in the body with matching red tests, fix #2 introduced no
stale or contradicting text, and the full-plan sweep found no remaining
unscoped range-dereference sites. Proceed to the next pipeline phase.

## Red test report (2026-07-07)

### Test count per WP

| WP | File | Tests | Failure mode |
|----|------|-------|--------------|
| WP1b.1 | `tests/test_memory_retention.py` | 22 | CollectionError — `ModuleNotFoundError: No module named 'corvidae.retention'` |
| WP1b.2 | `tests/test_memory_dedup.py` | 7 | 3 fail at assertion (AssertionError: expected 1 indexed, got 2); 4 pass (regression guards for "no merge" scenarios that are already correct) |
| WP1b.3 | `tests/test_memory_tools.py` | 23 | CollectionError — `ModuleNotFoundError: No module named 'corvidae.tools.memory_tools'` |
| WP1b.4 | `tests/test_redact.py` | 12 | CollectionError — `ModuleNotFoundError: No module named 'corvidae.commands.redact'` |
| WP1b.5 | `tests/evals/test_retrieval_benchmark.py` | 6 | 4 pass (1a retrieval already meets floors); 2 xfail (contradiction annotation, Phase 2+) |
| **Total** | | **70** | |

### Fixtures created

- `tests/fixtures/memory_retrieval_general.json`: 15 memories, 10 queries (8 with relevant, 2 no-hit negative cases). Channels: irc:#electronics, irc:#garden, irc:#home.
- `tests/fixtures/memory_retrieval_contradictions.json`: 3 records (c1, c2, c3) on the PlatformIO-vs-Arduino-IDE contradiction. 2 queries, both xfail.

### Measured WP1b.5 floors (stub embedder, 2026-07-07)

- recall@5: measured 0.625 → floor set at 0.60
- MRR: measured 0.448 → floor set at 0.40

### Full-suite result (--continue-on-collection-errors)

`3 failed, 1267 passed, 2 skipped, 2 xfailed, 3 errors`

- Failures: 3 WP1b.2 dedup tests (designed reason: missing dedup logic in `_consolidate_range`).
- Errors: 3 collection errors (designed reason: missing modules `corvidae.retention`, `corvidae.tools.memory_tools`, `corvidae.commands.redact`).
- All 1259 pre-existing tests still pass; 2 pre-existing skips unchanged.
- No pre-existing test broken.

**Ready for red review: YES**

## Red test review report (2026-07-07)

Fresh-context review of the 70 red tests written against the final plan
body (after Design re-review #2 gate PASS). Scope: coverage, correctness,
failure-mode confirmation, suite conventions, full-suite run.

### Test counts

| WP | File | Tests | Failure mode confirmed |
|----|------|-------|----------------------|
| WP1b.1 | `tests/test_memory_retention.py` | 22 | CollectionError — `ModuleNotFoundError: No module named 'corvidae.retention'` ✓ |
| WP1b.2 | `tests/test_memory_dedup.py` | 7 | 3 AssertionError (got 2 indexed, expected 1); 4 pass ✓ |
| WP1b.3 | `tests/test_memory_tools.py` | 23 | CollectionError — `ModuleNotFoundError: No module named 'corvidae.tools.memory_tools'` ✓ |
| WP1b.4 | `tests/test_redact.py` | 12 | CollectionError — `ModuleNotFoundError: No module named 'corvidae.commands.redact'` ✓ |
| WP1b.5 | `tests/evals/test_retrieval_benchmark.py` | 6 | 4 pass (floors met); 2 xfail ✓ |

### Coverage

All plan-specified behaviors have tests. The nine behaviors called out in the
review prompt were verified individually:

- Demotion sets `embedded=0`: `test_old_low_importance_unaccessed_record_demotes` asserts `m["embedded"] == 0` and `not vec_row_exists`.
- Backfill scope regression (`indexed=1 AND redacted=0 AND superseded_by IS NULL AND embedded=0`): `test_demotion_not_undone_by_same_run_backfill` asserts the vec row is still gone after the full run including backfill, with a live encoder.
- Channel predicate on all three msg-id range dereferences: `test_recall_raw_channel_id_predicate_excludes_foreign_rows` (WP1b.3), `test_foreign_channel_numerically_containing_memory_untouched` (WP1b.4 message→memory cascade), `test_redact_memory_tombstones_own_channel_rows_only` (WP1b.4 memory form).
- Tool compartment scoping + error semantics + no stat bump on rejection: `TestToolCompartmentScoping` — five tests covering foreign rejection, sibling visibility, out-of-scope channel arg, and stat non-bump.
- `recall_raw` bumps / `search_memory` does not: `TestStatSemantics` plus inline assertions in both WPs.
- Pre-1b schema probe: `TestPrePhase1bDb` — notices collected, message tombstoned, no crash, backfill-after-redact indexes tombstone not secret.
- `message_log_au` plain-DELETE trigger: `test_fts_trigger_on_update_message` — UPDATE does not abort; original unsearchable, tombstone indexed.
- Grace period and floor exemptions: `test_young_record_exempt_from_demotion` and `test_high_importance_record_exempt_from_demotion`.
- FTS-only re-promotion (embedding disabled): `test_repromotion_with_embedding_disabled_fts_reachable` — `indexed=1`, no vec, no embed call, FTS MATCH returns the row.

### Failure-mode verification

Full run (`--continue-on-collection-errors`) reproduced: `3 failed, 8 passed, 2 xfailed, 3 errors` from the new files; pre-existing suite: `1259 passed, 2 skipped`. Combined total matches the plan's reported `3 failed, 1267 passed, 2 skipped, 2 xfailed, 3 errors`. All designed failure modes are as documented. The 4 dedup regression guards and 4 benchmark floor tests pass pre-implementation as required (they test existing 1a behavior or already-correct absence of merge).

### Correctness

Tests assert the designed behavior, not plausible variants. Assertions are
generally specific enough to reject wrong implementations. Two issues found:

### New findings

**Critical:** none.

**Important:**

1. **`redact_messages` import comment omits `notices` parameter.** The import
   block (line 40) documents `redact_messages(db, message_ids, dry_run=False)
   -> dict` but `test_pre_1b_db_tombstones_messages_skips_absent_tables`
   (line 547) calls it as `await redact_messages(db, [msg_id],
   notices=notices)`. An implementer trusting the comment will implement
   without `notices`, causing `TypeError: unexpected keyword argument
   'notices'` — a misleading failure unrelated to the behavior under test.
   Fix: update the import comment to `redact_messages(db, message_ids,
   dry_run=False, notices: list | None = None) -> dict`.

2. **Logically vacuous assertion in `test_group_sibling_recall_raw_works`.**
   Line 622: `assert "error" not in result.lower() or "scope" not in
   result.lower()` — this passes for any error message that does not contain
   the word "scope" (e.g., "error: access denied"), which is the common case.
   The assertion only rejects responses that contain BOTH "error" and "scope".
   The real guard is the subsequent `assert "sibling channel content" in
   result`, which correctly catches any non-dialog return. The first assertion
   is misleading code that could cause an implementer to believe the partial
   check is enforcing scope semantics. Fix: replace with
   `assert "error" not in result.lower()` or remove it and rely solely on the
   content assertion.

**Cosmetic:**

1. `test_retrieval_benchmark.py` comment (line 133) says "recall@5 measured
   ≈ 0.875" but the Red test report documents "measured 0.625". The floor
   (0.60) is correct against the actual measurement, but the comment
   contradicts the plan. Fix: correct to "recall@5 measured ≈ 0.625".

2. `test_memory_retention.py` imports `MemoryToolsPlugin` (line 37), an
   undocumented cross-WP dependency. Implementing WP1b.1 alone does not
   unlock the retention tests — the collection error shifts from
   `corvidae.retention` to `corvidae.tools.memory_tools`. The file's
   documented failure mode mentions only the retention module. Fix: note the
   dependency in the file docstring or implement WP1b.1 and WP1b.3 together.

### Suite conventions

All test files follow existing patterns: stub embed clients with `kind=None`
parameter, in-memory aiosqlite DBs, no `@pytest.mark.eval` leaks into the
default run (benchmark tests live in `tests/evals/` and run in CI by design,
consistent with existing `test_metrics.py`). The `xfail` contradiction tests
use `strict=False`, correct for a Phase 2+ feature.

### Gate

**FAIL** — two important findings. Both are localized and mechanically fixable:
update the import comment to include `notices`, and replace the vacuous
`or`-assertion with `assert "error" not in result.lower()`. No structural
redesign or coverage gaps. Fix-review-gate loop should close in one tranche.

---

## Red fix report #1 (2026-07-07)

### Fixes applied

**Important #1 — `redact_messages` import comment missing `notices`**
`tests/test_redact.py` line 40: updated comment from
`# async: (db, message_ids, dry_run=False) -> dict` to
`# async: (db, message_ids, dry_run=False, notices: list | None = None) -> dict`.

**Important #2 — vacuous assertion in `test_group_sibling_recall_raw_works`**
`tests/test_memory_tools.py` line 622: replaced
`assert "error" not in result.lower() or "scope" not in result.lower()`
with `assert "error" not in result.lower()`.

**Cosmetic #1 — recall@5 comment contradicted plan**
`tests/evals/test_retrieval_benchmark.py` line 131: corrected comment from
`recall@5 measured ≈ 0.875` to `recall@5 measured ≈ 0.625`.

**Cosmetic #2 — undocumented cross-WP dependency in test_memory_retention.py**
`tests/test_memory_retention.py`: added a 4-line comment block before the
`MemoryToolsPlugin` import documenting that WP1b.1 tests require WP1b.3's
module to exist and that both WPs should be implemented together.

### Suite status

- **WP1b.1** (`test_memory_retention.py`): CollectionError unchanged —
  `ModuleNotFoundError: No module named 'corvidae.retention'`.
- **WP1b.2** (`test_memory_dedup.py`): 3 designed failures unchanged (AssertionError),
  4 passing tests still pass.
- **WP1b.3** (`test_memory_tools.py`): CollectionError unchanged —
  `ModuleNotFoundError: No module named 'corvidae.tools.memory_tools'`.
- **WP1b.4** (`test_redact.py`): CollectionError unchanged —
  `ModuleNotFoundError: No module named 'corvidae.commands.redact'`.
- **WP1b.5** (`test_retrieval_benchmark.py`): 4 passed, 2 xfailed — unchanged.
- **Pre-existing suite** (excluding new red-phase files): 1263 passed, 2 skipped,
  3 failed (the 3 designed WP1b.2 dedup failures). The +4 vs. the original 1259
  baseline are the 4 passing dedup guard tests that exercise existing 1a behavior.
  No regressions.

### Gate

**PASS** — all four findings fixed, no failure modes altered, pre-existing tests
unaffected. Ready for re-review.

## Red re-review report #1 (2026-07-07)

Fresh-context re-review after Red fix report #1. Scope: (a) each finding
from Red test review report genuinely fixed in the test files, (b) the
de-vacuousized assertion combined with its following positive assertion now
rejects both error responses and scope leaks, (c) no collateral changes
introduced, (d) full-suite run matches expected baseline.

### Fix verification

**Important #1 — `redact_messages` import comment now includes `notices`.**
`tests/test_redact.py` line 40:
`# async: (db, message_ids, dry_run=False, notices: list | None = None) -> dict` —
present. Matches the call site at line 547 (`await redact_messages(db,
[msg_id], notices=notices)`). An implementer reading the comment will now
implement the `notices` parameter correctly.

**Important #2 — vacuous assertion replaced.**
`tests/test_memory_tools.py` line 622: `assert "error" not in result.lower()`
— present. The subsequent assertion at line 625 (`assert "sibling channel
content" in result`) is also confirmed present. Together they form a correct
two-part guard: the first rejects any error response; the second rejects
any return value that does not contain the expected sibling-channel dialog
(including silent access denials and scope leaks). The combination is not
vacuous.

**Cosmetic #1 — benchmark comment corrected.**
`tests/evals/test_retrieval_benchmark.py` line 131:
`#   recall@5 measured ≈ 0.625 → floor 0.60 (conservative margin)` —
present. No stale 0.875 reference found anywhere in the file.

**Cosmetic #2 — cross-WP dependency comment added.**
`tests/test_memory_retention.py` lines 37–40: a 4-line comment block
documenting that WP1b.1 tests require WP1b.3's module and that both WPs
should be implemented together — present.

### Collateral-change check

`git diff` of all four test files against HEAD produced no output (all four
are untracked new files, no committed baseline). Manual scan of each fix
site confirmed no surrounding context was altered beyond the stated change.

### Full-suite run

`python -m pytest -q --continue-on-collection-errors`
result: `3 failed, 1267 passed, 2 skipped, 2 xfailed, 3 errors`

Matches the Red test report's documented baseline exactly:
- 3 designed failures: WP1b.2 dedup AssertionError ✓
- 1267 passed: 1263 pre-existing + 4 dedup regression guards ✓
- 2 skipped: pre-existing ✓
- 2 xfailed: contradiction benchmark tests ✓
- 3 errors: collection errors for missing modules (corvidae.retention,
  corvidae.tools.memory_tools, corvidae.commands.redact) ✓

No regressions introduced by the fixes.

### Findings

**Critical:** 0
**Important:** 0
**Cosmetic:** 0

### Gate

**PASS** — all four findings resolved correctly, no collateral changes,
full suite matches expected baseline. Proceed to the next pipeline phase.

## Green implementation report (2026-07-07)

### Files created
- `corvidae/retention.py` — retention scoring, demotion pass, retention job
- `corvidae/tools/memory_tools.py` — `MemoryToolsPlugin`, recall_raw/search_memory tools
- `corvidae/commands/redact.py` — redact CLI: message, memory, range forms; WAL check; schema-presence probe; dry-run; FTS verification

### Files modified
- `corvidae/memory.py` — WP1b.2 dedup (supersede path), message_fts DDL/backfill/trigger, memory_au trigger, embedded=0 on supersede
- `pyproject.toml` — entry points for corvidae.tools.memory_tools and corvidae.commands.redact

### Suite numbers
1326 passed, 1 failed, 2 skipped, 2 xfailed (1331 total collected)

### One unresolved failure: test_non_wal_aborts_with_clear_error

`tests/test_redact.py::TestNonWalJournalMode::test_non_wal_aborts_with_clear_error`

This test is **wrong/unimplementable** per the following analysis:

The test creates `aiosqlite.connect(":memory:")` without issuing `PRAGMA journal_mode = WAL`, then expects `redact_messages` to raise `RuntimeError`/`SystemExit`/`ValueError` mentioning "wal" or "journal". The test comment says "Leave journal_mode as the default (DELETE)".

The comment is factually incorrect. SQLite in-memory databases (`":memory:"`) always report `journal_mode = "memory"` regardless of whether `PRAGMA journal_mode = WAL` was issued; the `WAL` pragma is silently ignored for in-memory connections. The default for file-based databases is "delete", but not for in-memory ones.

The other two tests in `TestPrePhase1bDb` (lines 538 and 571) also use `aiosqlite.connect(":memory:")`, issue `PRAGMA journal_mode = WAL` (which silently fails), and call the same `redact_messages` — yet they are expected to succeed. From the database's perspective, both scenarios produce an identical state (journal_mode = "memory"). Every SQLite pragma, `wal_checkpoint`, and `database_list` filename returns identical values for both cases. There is no detectable difference.

**The `_check_wal` implementation currently allows "wal" and "memory"** (so in-memory tests pass) which is correct for all passing tests. Removing "memory" from the allowlist would fix this test but break all 11 other redact tests that use in-memory DBs.

**Judgment call needed from Schuyler:** the test should use a temp-file database (`tempfile.NamedTemporaryFile`) rather than `:memory:` so that WAL mode is actually meaningful. That change would make `_check_wal` work correctly and the test pass without breaking any other tests. Since the instructions say "do not modify tests," this needs explicit approval.

---

## Green fix report #1 (2026-07-07)

### Diagnosis confirmed

`aiosqlite.connect(":memory:")` always yields `journal_mode = 'memory'` regardless of any `PRAGMA journal_mode = WAL` attempt (which silently no-ops). The comment in the original test claiming the default would be DELETE was factually wrong for in-memory connections — DELETE is only the default for file-based databases. The `_check_wal` implementation correctly allows `"memory"` so that all other in-memory tests pass; the non-WAL condition was therefore untestable in-memory without breaking every other test in the file.

### Fix applied

`tests/test_redact.py::TestNonWalJournalMode::test_non_wal_aborts_with_clear_error` was changed to:

- Accept a `tmp_path` pytest fixture parameter (consistent with async test conventions elsewhere in the suite, e.g. `test_jsonl_log.py`, `test_hot_reload.py`).
- Connect to `tmp_path / "non_wal.db"` instead of `":memory:"`.
- Assert up front that the journal mode is genuinely non-WAL (asserts `!= "wal"` — for a fresh file DB this will be `"delete"`), so a future regression in test setup would be caught immediately.
- Updated the stale comment explaining why a file DB is required.
- Assertions on the raised exception and error message text are unchanged.

### Final suite numbers

```
1327 passed, 2 skipped, 2 xfailed
```

No regressions. All expected counts match the target stated in the agent brief.

### Flag for green reviewer

The fix was authorized under Schuyler's standing authorization for this test repair. The change strengthens rather than weakens the test: the non-WAL condition is now genuinely exercised on a real file-based DB.

## Green review report (2026-07-07)

Fresh-context review of the complete Phase 1b implementation against the
final plan body (Design re-review #2 PASS → Red re-review #1 PASS).
Scope: plan conformance, code quality, security, test-change legitimacy,
suite reproduction, eval benchmark floors.

### Suite and benchmarks

Full suite: `1327 passed, 2 skipped, 2 xfailed` — matches the implementation
report exactly. Eval benchmarks (`tests/evals/test_retrieval_benchmark.py`):
recall@5 PASS (floor 0.60), MRR PASS (floor 0.40), negative-query admission
PASS, 2 xfailed (contradiction, Phase 2+).

### Authorized test change verification

`tests/test_redact.py::TestNonWalJournalMode::test_non_wal_aborts_with_clear_error`
uses `tmp_path` fixture and connects to `tmp_path / "non_wal.db"`, pre-asserts
`row[0].lower() != "wal"` (verifying genuine non-WAL state), and keeps the
original assertions on the raised exception's text unchanged. All other test
files are new untracked files with no committed baseline to diff against.
The change matches the green fix report's description exactly and is legitimate.

### Plan-conformance verification (spot-checked items)

- Demotion: `indexed=0, embedded=0`, `DELETE FROM memory_vec` — ✓ (`retention.py:85-96`)
- Re-promotion: `indexed=1` only, no inline embed — ✓ (`retention.py:123-124`)
- Backfill scope: `indexed=1 AND redacted=0 AND superseded_by IS NULL AND embedded=0` — ✓ (`retention.py:146-150`)
- `embed(texts, kind="document")` in backfill — ✓ (`retention.py:160`)
- Retention job via `_spawn` / `on_idle` / `_retention_startup` / `retention_meta` — ✓
- Grace period and importance-floor exemptions — ✓ (`retention.py:73-75`)
- Dedup: same-channel only (`channel_id = ?`, not `_channel_scope`) — ✓ (`memory.py:693`)
- Superseded: `indexed=0, embedded=0`, vec delete — ✓ (`memory.py:754-763`)
- Backfill `indexed=1` filter: demoted records excluded even after same-run backfill — ✓ (same filter)
- `recall_raw` channel predicate: `WHERE channel_id = ? AND id BETWEEN ? AND ?` — ✓ (`memory_tools.py:214-218`)
- Redact memory form channel predicate: same pattern — ✓ (`redact.py:244-248`)
- Memory→message cascade per-channel intersection — ✓ (`redact.py:157-169`)
- `_ctx.channel is None` → error, no fallback — ✓ (`memory_tools.py:69-71`)
- Out-of-scope rejection without stat bump — ✓ (`memory_tools.py:204-208`; bump at 263)
- `recall_raw` bumps; `search_memory` does not — ✓
- `sqlite_master` probe; skip-notice on absent surfaces — ✓ (`redact.py:71-110`)
- WAL assertion — ✓ (`redact.py:58-68`)
- `message_log_au` plain-DELETE trigger — ✓ (`memory.py:294-298`)
- SQL injection: all user input goes through Click integer typing or `?` params — no risk

### Findings

**Critical:** 0

**Important:**

1. **Retention job missing `set_attribution(stage="retention")`.**
   `run_retention_job` in `retention.py` makes real embedding calls in the
   backfill pass (`client.embed(texts, kind="document")`) without calling
   `set_attribution(stage="retention", ...)`. The consolidation task correctly
   sets `set_attribution(stage="consolidation", ...)` at `memory.py:584`.
   The plan says the retention job runs "with attribution (`stage='retention'`)".
   Backfill embedding costs will appear unattributed in `usage_log`, making
   them invisible to the Phase 0 cost metering that §6 eval discipline depends
   on. Fix: add `set_attribution` / `reset_attribution` around the backfill pass
   in `run_retention_job` (same pattern as `memory.py:584,790`).

2. **Verification pass (WP1b.4 step 3) not implemented.**
   The plan requires that after the cascade, the CLI runs `memory_fts` and
   `message_fts` MATCH queries for a sample token of the redacted text and
   prints "zero hits." None of the three CLI subcommands implement this;
   neither does `redact_messages` nor `redact_memory_id`. The DoD explicitly
   says "verification pass printing zero hits." The underlying FTS state is
   correct (covered by tests), but the operator receives no printed confirmation
   that the cascade was complete. Fix: after each cascade, extract a sample
   token from the original text and run MATCH against both FTS tables; print
   the results (or "verified: 0 hits" if clean).

3. **`tags` filter silently ignored in `search_memory`.**
   The plan specifies `tags: list[str] | None = None` as a column filter
   alongside `after`/`before`. The parameter is in the signature but never
   applied in the SQL query (`memory_tools.py:54-168`). The `topic_tags`
   column exists in `memory` (`memory.py:196`). An agent calling
   `search_memory("query", tags=["electronics"])` receives unfiltered results
   with no indication that tags were not applied. Fix: add a JSON array
   intersection condition on `topic_tags` when `tags` is non-empty.

4. **`search_memory` not token-capped.**
   The plan says "token-capped" (`memory_tools.py` registers `recall_raw`
   with a `max_tokens` parameter and correctly truncates; `search_memory` has
   no equivalent). The `LIMIT 50` result count cap is the only constraint.
   At typical summary lengths (~200 chars each), 50 results exceed 10,000
   characters / ~2,500 tokens, inflating the agent's context window on broad
   queries. Fix: apply the same `_CHARS_PER_TOKEN` cap pattern used in
   `recall_raw` to the result list in `search_memory`.

**Cosmetic:**

1. **`search_memory` output missing "band-less score"**: plan specifies
   "(id, band-less score, age, flags, summary)"; implementation outputs
   "(id, age, summary, flags)". Results are ranked by FTS relevance via
   `ORDER BY rank` — relevance ordering is correct, score is just not exposed.

2. **Green implementation report typo**: lists `summarize_memory` as a
   registered tool; actual code has only `search_memory` and `recall_raw`
   per plan. Code is correct; report is wrong.

3. **`new_rc` variable in dedup** (`memory.py:733`): represents what to add
   to the new record's `retrieval_count` (i.e., the old record's count), not
   the new record's own count as the name implies. Functionally correct.

### Gate

**FAIL** — 4 important findings. Findings 1 and 4 are mechanical fixes (add
attribution call; add token-cap loop). Findings 2 and 3 require a small
amount of new logic (FTS verification query + print; JSON array filter for
tags). No critical issues; architecture, channel-predicate safety, and test
integrity are sound. One fix-review-gate loop should close it.

---

## Green fix report #2 (2026-07-07)

### Important finding fixes

**Important 1 — Retention backfill missing attribution**

Fix: added `from corvidae.attribution import reset_attribution, set_attribution`
to `corvidae/retention.py`. Wrapped the backfill embed call in
`attribution_token = set_attribution(stage="retention")` / `finally:
reset_attribution(attribution_token)` (same pattern as `memory.py:584,790`).

New test: `TestBackfillAttribution::test_backfill_embed_carries_retention_attribution`
(`tests/test_memory_retention.py`). Uses an `AttributionCapturingEmbedClient`
that records `get_attribution()` at embed time; asserts `stage == "retention"`.

**Important 2 — WP1b.4 verification pass not implemented**

Fix: added two helper functions to `corvidae/commands/redact.py`:
- `_extract_sample_token(text)` — extracts the first 4+-char word from JSON
  message content or plain text (memory summaries), skipping tombstone words.
- `_verify_fts_clean(db, sample_token)` — runs `COUNT(*)` MATCH queries against
  `message_fts` and `memory_fts` (if present) and returns a "verified: 0 hits"
  or "WARNING: …hits…" string.

Each CLI subcommand (`_message_cmd`, `_memory_cmd`, `_range_cmd`) now captures
a sample token from the original content before tombstoning, then calls
`_verify_fts_clean` after the cascade and prints the result. Dry-run mode skips
verification (no write, no need).

New tests: `TestVerificationPass` in `tests/test_redact.py`:
- `test_extract_sample_token_from_json_message`
- `test_extract_sample_token_from_plain_text`
- `test_extract_sample_token_returns_none_for_empty`
- `test_verify_fts_clean_returns_zero_hits_after_redact`
- `test_verify_fts_clean_returns_warning_when_hits_remain`
- `test_verify_fts_clean_handles_absent_fts_tables`

**Important 3 — `tags` parameter silently ignored in `search_memory`**

Fix: added a JSON array intersection condition in `corvidae/tools/memory_tools.py`
when `tags` is non-empty:
```sql
m.topic_tags IS NOT NULL AND EXISTS (
    SELECT 1 FROM json_each(m.topic_tags)
    WHERE json_each.value IN (?, ...)
)
```
Parameters are appended to `params` list — no string interpolation.

New tests: `TestTagsFilter` in `tests/test_memory_tools.py`:
- `test_tags_filter_returns_only_matching_records`
- `test_tags_filter_excludes_untagged_and_wrong_tagged_records`
- `test_tags_filter_none_returns_all_records`

**Important 4 — `search_memory` not token-capped**

Fix: added `max_tokens: int = 2000` parameter to `search_memory` in
`corvidae/tools/memory_tools.py`. Result rows are accumulated with a
`max_chars = max_tokens * _CHARS_PER_TOKEN` guard identical to `recall_raw`;
truncation appends `[truncated — token limit reached]`. Docstring updated to
document the new parameter.

New tests: `TestTokenCap` in `tests/test_memory_tools.py`:
- `test_search_memory_token_cap_truncates_oversized_result`
- `test_search_memory_large_token_cap_returns_all`

### Cosmetic finding fixes

**Cosmetic 1 — `search_memory` output missing band-less score**

Fix: SQL query now selects `memory_fts.rank`. Output format changed from
`"{mid}. ({age_str}) {summary}{flag_str}"` to
`"{mid}. {score:.4f} ({age_str}){flag_str} {summary}"` where
`score = -(rank or 0.0)` (positive, higher = more relevant).

**Cosmetic 2 — Green implementation report typo**

The green implementation report (appended earlier in this file) lists
`summarize_memory` as a registered tool; actual code registers `search_memory`
and `recall_raw`. Code is correct; the report note stands as a documentation
artifact. No code change needed.

**Cosmetic 3 — `new_rc` misleading variable name in `memory.py:733`**

Fix: renamed `new_rc` to `inherited_rc` in the dedup fold-stats block
(`corvidae/memory.py`) to clarify it is the old record's retrieval count
being folded into the new record, not the new record's own count.

### Final suite numbers

1339 passed, 0 failed, 0 errors, 2 skipped, 2 xfailed.
Baseline was 1327 passed; 12 new tests added.

### New test names (12 total)

`tests/test_memory_retention.py`:
- `TestBackfillAttribution::test_backfill_embed_carries_retention_attribution`

`tests/test_redact.py`:
- `TestVerificationPass::test_extract_sample_token_from_json_message`
- `TestVerificationPass::test_extract_sample_token_from_plain_text`
- `TestVerificationPass::test_extract_sample_token_returns_none_for_empty`
- `TestVerificationPass::test_verify_fts_clean_returns_zero_hits_after_redact`
- `TestVerificationPass::test_verify_fts_clean_returns_warning_when_hits_remain`
- `TestVerificationPass::test_verify_fts_clean_handles_absent_fts_tables`

`tests/test_memory_tools.py`:
- `TestTagsFilter::test_tags_filter_returns_only_matching_records`
- `TestTagsFilter::test_tags_filter_excludes_untagged_and_wrong_tagged_records`
- `TestTagsFilter::test_tags_filter_none_returns_all_records`
- `TestTokenCap::test_search_memory_token_cap_truncates_oversized_result`
- `TestTokenCap::test_search_memory_large_token_cap_returns_all`

---

## Green re-review report #2 (2026-07-07)

### Scope

Re-review of the 4 important fixes and 12 new tests from "Green fix report #2
(2026-07-07)". Suite baseline: 1327 → 1339 (12 new tests). Expected: 1339
passed, 0 failed.

### Fix verification

**Important 1 — Retention backfill attribution**

`retention.py:158` calls `set_attribution(stage="retention")` before
`client.embed(texts, kind="document")` and `reset_attribution(attribution_token)`
in a `finally` block — matching the `try/finally` pattern at `memory.py:584`.
Correct.

**Important 2 — WP1b.4 verification pass**

`_extract_sample_token` and `_verify_fts_clean` added to `redact.py:59–109`.
All three subcommands (`_message_cmd`, `_memory_cmd`, `_range_cmd`) capture a
sample token from the original content before tombstoning and call
`click.echo(await _verify_fts_clean(conn, sample_token))` after the cascade.
Dry-run skips verification. Correct.

**Important 3 — `search_memory` tags filter**

`memory_tools.py:142–149` adds a JSON array intersection condition when `tags`
is non-empty:
`m.topic_tags IS NOT NULL AND EXISTS (SELECT 1 FROM json_each(m.topic_tags) WHERE json_each.value IN (?, ...))`
Tags are appended to `params` via `params.extend(tags)` — no string
interpolation. Correct.

**Important 4 — `search_memory` token cap**

`max_tokens: int = 2000` parameter added. `max_chars = max_tokens *
_CHARS_PER_TOKEN` guard at `memory_tools.py:171` identical to `recall_raw`'s
pattern at line 266. Truncation appends `[truncated — token limit reached]`.
Correct.

### Test non-vacuousness (spot-check)

- `TestBackfillAttribution`: `AttributionCapturingEmbedClient` records
  `get_attribution()` at call time; asserts `stage == "retention"`. Removing
  `set_attribution` from retention.py would produce `attr.get("stage") == None`
  → assertion fails. Non-vacuous.
- `TestVerificationPass::test_verify_fts_clean_returns_zero_hits_after_redact`:
  seeds a unique token, confirms it is findable pre-redact, tombstones the row,
  then checks `_verify_fts_clean` returns "verified" and "0 hits". Removing
  the function would break the import. Removing the FTS cascade from
  `redact_messages` would leave the token findable → "WARNING" not "verified". Non-vacuous.
- `TestTagsFilter::test_tags_filter_returns_only_matching_records`: two records
  share a query keyword but differ by tag; asserts the wrong-tag record is
  absent. Without the `if tags:` block both records are returned → assertion
  fails. Non-vacuous.
- `TestTokenCap::test_search_memory_token_cap_truncates_oversized_result`:
  seeds 15 records with long summaries, calls with `max_tokens=1`; asserts
  `[truncated` in result. Without the cap the full 15 records return untruncated
  → assertion fails. Non-vacuous.

Note: `TestVerificationPass` tests the `_verify_fts_clean` helper directly; no
test exercises the `click.echo(await _verify_fts_clean(...))` wiring in the CLI
commands. This is consistent with the plan's "direct function invocation"
testing strategy and does not constitute a blocking gap — the CLI integration
is verifiable by code inspection and the helper behavior is locked by tests.

### Cosmetic findings check

**Cosmetic 2 — Plan-file typo** (`summarize_memory` at line 1267 of the green
implementation report section): was not corrected by the fix agent. Corrected
inline in this append: changed to `recall_raw/search_memory tools`.

**Cosmetic 1 and 3**: score display (`memory_tools.py:187`) and `inherited_rc`
rename (`memory.py:733`) both confirmed present.

### New issues introduced

None. Tags filter uses parameterized SQL; token cap follows the established
`recall_raw` pattern; attribution wrapping is tightly scoped; no scope creep or
convention breaks observed.

### Suite

1339 passed, 2 skipped, 2 xfailed, 0 failed, 0 errors — matches claimed
baseline.

### Findings

- Critical: 0
- Important: 0
- Cosmetic: 0 (plan-file typo corrected inline above)

### Gate

**PASS**

---

## Requirements gate report (2026-07-07)

Reviewed by: requirements-gate reviewer (fresh context).
Sources consulted: plan body (WPs, DoD, Non-goals), `git status --short`,
`git diff --stat HEAD`, `plans/implementation/README.md` Phase 1b row,
`plans/bootstrap-mapping.md` §3.1/§7 row 1b, `docs/configuration.md`,
`docs/design.md`, `docs/plugin-guide.md`, `pyproject.toml`, test suite.

### Working-tree surface

Modified: `corvidae/memory.py`, `pyproject.toml`, `plans/implementation/phase-1b.md`.
Untracked (new): `corvidae/retention.py`, `corvidae/tools/memory_tools.py`,
`corvidae/commands/redact.py`, `tests/test_memory_retention.py`,
`tests/test_memory_dedup.py`, `tests/test_memory_tools.py`,
`tests/test_redact.py`, `tests/evals/test_retrieval_benchmark.py`,
`tests/fixtures/memory_retrieval_general.json`,
`tests/fixtures/memory_retrieval_contradictions.json`.
Documentation files (`docs/configuration.md`, `docs/design.md`,
`docs/plugin-guide.md`): no diff and no untracked changes — untouched.

### DoD checklist

| # | Item | Status |
|---|------|--------|
| 1 | All red tests green; full suite passes | SATISFIED — 1339 passed, 0 failed, 2 skipped, 2 xfailed (confirmed via `pytest -q`). |
| 2 | CI benchmark enforcing floors | SATISFIED — `tests/evals/test_retrieval_benchmark.py` in place; recall@5 floor 0.60, MRR floor 0.40, negative-query assertion. |
| 3 | Live check (demotion/search_memory/recall_raw/redact verification pass) | NOT VERIFIABLE STATICALLY — code exists and is correct per green review; runtime demonstration is required. |
| 4 | §7 row 1b: hedging prompt fragment + tools | SATISFIED — `prompts/memory_calibration.md` present (Phase 1a); `search_memory` and `recall_raw` registered via `MemoryToolsPlugin` entry point. "Demonstrated live" cannot be verified statically. |
| 5 | §7 row 1b: bounded store (vec rows ≤ indexed records) | SATISFIED — demotion deletes the vec row and sets `embedded=0`; backfill is scoped to `indexed=1`; green review confirmed at `retention.py:85-96` and `retention.py:146-150`. |
| 6 | `docs/configuration.md` documents `memory.retention.*` and `memory.dup_threshold` | **NOT SATISFIED** — none of these keys appear in `docs/configuration.md`. The file was not modified; the `memory` table ends at `memory.channel_groups` / `memory.consolidation_prompt` (Phase 1a keys). |
| 7 | `docs/design.md` gains the redaction runbook paragraph | **NOT SATISFIED** — `docs/design.md` was not modified. The only redact references are the pre-existing Phase 1a schema listing (`redacted` column) and the placeholder "remaining memory work … is Phase 1b" at line 1360, which was not updated. |
| 8 | `docs/plugin-guide.md` notes the tool surface including channel scoping | **NOT SATISFIED** — `docs/plugin-guide.md` was not modified. No mention of `search_memory`, `recall_raw`, `MemoryToolsPlugin`, or `_channel_scope` enforcement exists in that file. |

### WP acceptance criteria

| WP | Status |
|----|--------|
| WP1b.1 (retention scoring, demotion, re-promotion, retention job, `retention_meta`) | SATISFIED — `corvidae/retention.py` present; all plan-specified behaviors verified in green re-review. |
| WP1b.2 (near-dup merge at consolidation, same-channel only, supersede path) | SATISFIED — dedup logic in `corvidae/memory.py`; dedup tests all green. |
| WP1b.3 (`message_fts` DDL + triggers + backfill, `search_memory`, `recall_raw`, `MemoryToolsPlugin` entry point, channel scoping) | SATISFIED — `corvidae/tools/memory_tools.py` present; entry point in `pyproject.toml`; all tools tests green. |
| WP1b.4 (`corvidae redact` CLI, 3 forms, WAL check, schema probe, dry-run, FTS verification pass) | SATISFIED — `corvidae/commands/redact.py` present; entry point in `pyproject.toml`; all redact tests green. |
| WP1b.5 (general-recall fixture ≥15 memories ≥8 queries 2 negative; contradiction fixture; CI benchmark) | SATISFIED — fixtures confirmed (15 memories, 10 queries, 2 negative); benchmark tests in place. |

### Scope observations

No out-of-scope changes detected. The three documentation files required by the DoD were not modified — not replaced by alternatives, simply absent. `docs/design.md` line 1360 ("The remaining memory work … is Phase 1b") was not updated to reflect that Phase 1b work is now complete.

### Upstream cross-check

- `plans/implementation/README.md` Phase 1b row: "Reconsolidation/demotion (grace period, near-dup merge); `redact` with full cascade; memory tools (`search_memory`, `recall_raw`); fixture evals (§3.1, §6)" — all code deliverables present.
- `bootstrap-mapping.md` §7 row 1b acceptance ("hedging + remember-harder; bounded store") — code satisfies the structural requirements; live demonstration is unverifiable statically.

### Verdict

**FAIL**

Three DoD items are not satisfied, all documentation:

1. `docs/configuration.md` missing `memory.retention.*` keys (`demote_below`, `grace_days`, `importance_floor`, `half_life_days` [retention], `interval`, `backfill_batch`) and `memory.dup_threshold`.
2. `docs/design.md` missing the redaction runbook paragraph.
3. `docs/plugin-guide.md` missing the `MemoryToolsPlugin` / `search_memory` / `recall_raw` tool surface entry including channel-scoping semantics.

All implementation code is complete and all 1339 tests pass. The gap is entirely in the documentation deliverables.

## Documentation report (2026-07-07)

### Files changed

- `docs/configuration.md` — added `memory.dup_threshold` row to the `memory`
  table; added `### memory.retention — Retention job` subsection with a table
  covering `grace_days`, `importance_floor`, `demote_below`, `half_life_days`,
  `interval`, and `backfill_batch` (all defaults verified against
  `corvidae/retention.py` and `corvidae/memory.py:541`).

- `docs/design.md` — updated `### Schema` paragraph to add `retention_meta`
  and `message_fts` to the MemoryPlugin schema listing; added
  `### Operator redaction (corvidae redact)` subsection under `## MemoryPlugin`
  covering forms, WAL requirement, sqlite_master probe behavior on pre-1b DBs,
  cascade steps (message tombstone + memory cascade + FTS propagation), and the
  verification pass; updated the stale `### Memory retrieval` placeholder in
  `## Unimplemented` to reflect both Phase 1a and 1b are complete; added
  `MemoryToolsPlugin → "memory"` to the dependency graph; added
  `MemoryToolsPlugin` (entry 16) to the Plugin Registration Order list;
  updated the Directory Layout to include `retention.py`,
  `commands/redact.py`, and `tools/memory_tools.py`.

- `docs/plugin-guide.md` — added `### MemoryToolsPlugin` section in the
  Bundled plugins section (after `MemoryPlugin`) documenting `search_memory`
  and `recall_raw` tools with parameter tables, channel-scope enforcement,
  recall-stat semantics, and `--Without this plugin--` note; added
  `memory_tools (MemoryToolsPlugin)` to the Registration order list.

### Sections added

- `docs/configuration.md`: `memory.dup_threshold` row; `memory.retention`
  subsection (6 keys + explanatory paragraph)
- `docs/design.md`: `### Operator redaction` under `## MemoryPlugin`;
  updates to Schema, Unimplemented, dependency graph, registration order,
  directory layout
- `docs/plugin-guide.md`: `### MemoryToolsPlugin` section; registration
  order update

### Ready for documentation review: YES

## Documentation review report (2026-07-07)

Fresh-context review against the working tree (`corvidae/retention.py`,
`corvidae/memory.py`, `corvidae/tools/memory_tools.py`,
`corvidae/commands/redact.py`) and `git diff docs/`. Scope: accuracy,
completeness, cross-file consistency, voice, and plan-promised items not yet
written.

### Accuracy

**configuration.md — `memory.dup_threshold` and `memory.retention` subsection**

All six `memory.retention.*` defaults and the `memory.dup_threshold` default
verified against code:

- `grace_days` 14, `importance_floor` 0.8, `demote_below` 0.15,
  `half_life_days` 90, `backfill_batch` 32 — `retention.py:55-59` ✓
- `interval` 21600 — `memory.py:541` ✓
- `memory.dup_threshold` 0.95 — `memory.py:336,376` ✓
- Retention score formula matches `retention.py:36-37` ✓
- Backfill skip on `embedding_meta` mismatch matches `retention.py:139` ✓

**design.md — Schema, Operator redaction, Registration order, Dependency
graph, Directory layout, Unimplemented**

- `retention_meta` (one-row, `last_run REAL`) — `memory.py:268-270` ✓
- `message_fts` as a regular (non-external-content) FTS5 table — `memory.py`
  DDL ✓
- WAL error message naming mode and conversion steps — `redact.py:118-121` ✓
- Cascade per-form behavior (channel predicate, trigger propagation, embedded=0)
  — `redact.py:134-165` ✓
- Verification pass "verified: 0 hits for '`<token>`' in `<tables>`" — exact
  match at `redact.py:107` ✓
- MemoryToolsPlugin at position 16, dependency graph, directory layout — all
  correct ✓
- Unimplemented section updated from Phase 1a to Phases 1a and 1b ✓

**One important accuracy error in design.md schema description (line 913-914):**

The schema paragraph describes `message_fts` as "searched by `search_memory`".
This is wrong. `search_memory` queries `memory_fts` (the memory-summary FTS
table), not `message_fts` (the raw `message_log` FTS table). Confirmed at
`memory_tools.py:154-156` (`FROM memory_fts JOIN memory m …`). The
plugin-guide.md correctly states "FTS5 keyword search over `memory_fts`".
`message_fts` is searched only by the `corvidae redact` verification pass
(`redact.py:92-95`) and, in the degradation path, indirectly by the retrieval
path — not by `search_memory`. Fix: change "searched by `search_memory`" to
"searched by the `corvidae redact` verification pass" in the schema paragraph.

**plugin-guide.md — MemoryToolsPlugin section**

- Entry point name `"memory_tools"` — `pyproject.toml:46` ✓
- `depends_on = frozenset({"memory"})` — `memory_tools.py:36` ✓
- `search_memory` parameters/defaults (channel/tags/after/before/include_demoted/
  max_tokens with correct defaults) — `memory_tools.py:54-62` ✓
- `recall_raw` parameters (`memory_id: int`, `max_tokens: int = 1000`) —
  `memory_tools.py:199-201` ✓
- Output format `id. score (age)[flags] summary` — `memory_tools.py:187` ✓
- Channel scope enforcement, stat semantics — match code ✓
- States searches `memory_fts` — correct, unlike design.md ✓

### Completeness against DoD

| DoD item | Status |
|----------|--------|
| `memory.retention.*` and `memory.dup_threshold` in `configuration.md` | SATISFIED — all 7 keys present |
| Redaction runbook in `design.md` | SATISFIED |
| Tool surface with channel scoping in `plugin-guide.md` | SATISFIED |

No plan-promised documentation beyond these three files is outstanding: the WP
sections reference these three files only; `eval_memory.py --live` docs are
not a DoD requirement.

### Consistency

- Registration order lists: design.md (numbered, 19 entries) and plugin-guide.md
  (unnumbered block, 20 entries) disagree on whether `ConfigWatcherPlugin` is
  present. The design.md list was already missing it before Phase 1b docs
  (pre-existing); the Phase 1b changes correctly inserted `MemoryToolsPlugin`
  at position 16 in both lists. Not introduced by this phase.
- Voice and formatting are consistent with each file's existing style ✓
- No other cross-file contradictions; plugin-guide.md and design.md agree on
  all Phase 1b-added content except the `message_fts`/`memory_fts`
  attribution noted above.

### Findings

**Critical:** 0

**Important:**

1. **design.md schema description misattributes `message_fts` to
   `search_memory`.** Line 913-914: "searched by `search_memory`" is wrong;
   `search_memory` queries `memory_fts`. `message_fts` is searched only by
   the redact CLI's verification pass. The plugin-guide.md is correct.
   Fix: change "searched by `search_memory`" to "searched by the `corvidae
   redact` verification pass" in the `message_fts` schema parenthetical.

**Cosmetic:**

1. **design.md WARNING format string imprecise.** Runbook says the CLI prints
   `"WARNING: <N> hits remain"`. Code prints `"WARNING: FTS still contains
   hits for '<token>': <table>: N hits"` (`redact.py:104`). The meaning is
   preserved; the exact text differs. An operator scripting against this output
   would need to check the code.

2. **Pre-existing cross-file inconsistency in registration order.** design.md's
   numbered list has 19 entries; plugin-guide.md's block has 20 (includes
   `config_watcher`). Predates Phase 1b docs; not introduced here. No action
   required for this phase.

### Gate

**FAIL** — one important finding. The fix is a single sentence in design.md's
schema paragraph (change `searched_by` attribution from `search_memory` to the
`corvidae redact` verification pass). Everything else — all defaults, all
parameter tables, channel-scoping prose, DoD completeness — is accurate.

---

## Documentation fix report #1 (2026-07-07)

### Fixes applied

1. **Important — docs/design.md schema paragraph (~line 913-914).** Removed
   the false claim that `message_fts` is "searched by `search_memory`".
   `search_memory` (memory_tools.py:154-156) queries `memory_fts` only.
   `message_fts` is queried by the `corvidae redact` verification pass
   (redact.py:87-95). New text: "queried and cleared by `corvidae redact` —
   not by `search_memory`, which queries `memory_fts`".

2. **Cosmetic — docs/design.md runbook WARNING format (~line 982).** Doc said
   `"WARNING: <N> hits remain"`; actual code (redact.py:104-106) outputs
   `"WARNING: FTS still contains hits for '<token>': <table>: <N> hits"`.
   Updated doc to match.

### Deferred

3. **Cosmetic 2 — registration-order list discrepancy (ConfigWatcherPlugin).**
   Pre-existing cross-file inconsistency between design.md (19 entries) and
   plugin-guide.md (20 entries, includes `config_watcher`). Out of Phase 1b
   scope; deferred.

### Gate

**Ready for re-review: yes.**

---

## Documentation re-review report #1 (2026-07-07)

### Scope

Verified (a) the important fix in docs/design.md (~line 913), (b) WARNING-format
wording, (c) absence of unrelated changes, (d) no new inaccuracies introduced by
the fix report's broader additions (docs/configuration.md, docs/plugin-guide.md).

### Findings

**Critical: 0. Important: 0. Cosmetic: 1.**

- **Cosmetic** — docs/design.md line 982-983: WARNING format is documented as
  `"WARNING: FTS still contains hits for '<token>': <table>: <N> hits"`, which
  represents the single-table case correctly. The actual code (redact.py:104-106)
  joins multiple tables with `"; "` when both message_fts and memory_fts have
  hits; the documented pattern does not reflect that multi-table variant. Minor
  incompleteness only — the single-table template is accurate and the multi-table
  case is unambiguously derivable.

### Verification details

1. **Important fix (message_fts / memory_fts):** docs/design.md line 913-915 now
   reads "queried and cleared by `corvidae redact` — not by `search_memory`, which
   queries `memory_fts`". Code confirms: `memory_tools.py:154-156` queries
   `memory_fts` only (`FROM memory_fts JOIN memory … WHERE memory_fts MATCH ?`);
   `redact.py:87-95` queries both `message_fts` and `memory_fts` in its
   verification pass. Fix is correct; no contradiction with plugin-guide.md.

2. **WARNING-format wording:** previously undocumented (new addition in fix
   report). Current text matches the single-table code path. See cosmetic finding
   above for multi-table gap.

3. **Scope of changes:** `git diff docs/` shows three files changed —
   `docs/design.md`, `docs/plugin-guide.md`, `docs/configuration.md`. All changes
   are Phase 1b documentation additions (retention config table, MemoryToolsPlugin
   section, operator-redaction runbook, schema paragraph updates). Spot-checked:
   `retention_meta` table and `message_log_ai`/`message_log_au` triggers confirmed
   in memory.py; `MemoryToolsPlugin` entry point confirmed in pyproject.toml
   (`memory_tools = "corvidae.tools.memory_tools:MemoryToolsPlugin"`);
   `depends_on = frozenset({"memory"})` matches code. No unrelated changes found.

4. **No new inaccuracies found** in the broader additions beyond the cosmetic
   multi-table WARNING gap noted above.

### Gate

**PASS.** Zero critical, zero important findings. Cosmetic finding does not block.

## Requirements gate addendum (2026-07-07)

Re-check of the three unmet documentation DoD items from the original Requirements gate report. Verified against current working tree (`docs/configuration.md`, `docs/design.md`, `docs/plugin-guide.md`).

| # | Item | Status |
|---|------|--------|
| 1 | `docs/configuration.md` documents `memory.retention.*` keys (`demote_below`, `grace_days`, `importance_floor`, `half_life_days`, `interval`, `backfill_batch`) and `memory.dup_threshold` | **SATISFIED** — all seven keys present in the `memory` table and `### memory.retention — Retention job` subsection (lines 140–155). |
| 2 | `docs/design.md` contains the redaction runbook per WP1b.4 DoD | **SATISFIED** — `### Operator redaction (corvidae redact)` subsection present covering all three forms, WAL requirement, schema-presence probe, per-form cascade behavior, and verification pass. |
| 3 | `docs/plugin-guide.md` documents the memory tool surface (`MemoryToolsPlugin`, `search_memory`, `recall_raw`, channel-scope enforcement) | **SATISFIED** — `### MemoryToolsPlugin` section present with full parameter tables for both tools, compartment-scoping prose, recall-stat semantics, and "Without this plugin" note. |

**Verdict: PASS**
