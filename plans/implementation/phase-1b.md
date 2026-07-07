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
- `corvidae/tool.py` / an existing `register_tools` implementation (e.g.
  `tools/goal_tracker.py` or `RuntimeSettingsPlugin`) for the tool
  registration pattern.
- `corvidae/commands/serve.py` for the `corvidae.commands` entry-point
  pattern (the redact CLI follows it).
- `tests/evals/metrics.py` and `scripts/eval_memory.py` from Phase 0.

## Amendments after embedding-prefixes (2026-07-06)

The embedding-prefixes change (merged to main after 1a) postdates this plan.
Three consequences, folded into the WP text below and repeated here for the
design phase:

1. Every 1b call site that embeds memory text (backfill, re-promotion
   re-embed) must call `embed(texts, kind="document")` — unprefixed vectors
   are inconsistent with the prefixed index.
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
   vec row + `indexed=0`. The `memory` row, its FTS entry, and the raw
   `message_log` range all remain. Retrieval (`before_agent_turn`) excludes
   demoted records; the `search_memory`/`recall_raw` tools include them,
   flagged — "remember harder" must reach what passive retrieval no longer
   surfaces.
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
   retention job ALSO runs once at startup.
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

**Files:** `corvidae/memory.py`

1. **Retention score** (module-level function, constants commented as
   §6-tunable):

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
   default 6h):
   - For every `indexed=1 AND redacted=0` record past grace and below
     floor: if score < threshold → demote (`indexed=0`, `DELETE FROM
     memory_vec WHERE memory_id=?`). Log count.
   - For every `indexed=0 AND redacted=0 AND superseded_by IS NULL` record:
     if score ≥ threshold (it was recalled via the tools since demotion) →
     re-promote (`indexed=1`, re-insert vec row — re-embed if `embedded=0`,
     with `kind="document"`; see Amendments).
     Demotion is reversible or it isn't demotion (§3.1).
   - Records with `embedded=0` and an available encoder: backfill embed
     with `kind="document"` (bounded batch per run,
     `memory.retention.backfill_batch` default 32). Skip entirely when
     embedding is disabled by the `embedding_meta` prefix/encoder mismatch
     guard (see Amendments).
3. **Tools count as recall:** `search_memory`/`recall_raw` (WP1b.3) update
   `retrieval_count`/`last_retrieved_at` exactly as passive retrieval does —
   that is what makes re-promotion reachable.

**Red tests** (`tests/test_memory_retention.py`, frozen clock):
- Young unretrieved record: exempt (grace). High-importance stale record:
  exempt (floor). Old, low-importance, never-retrieved: demoted, vec row
  gone, `memory` row and FTS intact.
- Demoted record excluded from `before_agent_turn` retrieval; still found by
  `search_memory` (flagged).
- Demoted record whose count is bumped via tool recall re-promotes on the
  next job run.
- Job runs at startup with no traffic (regression for the push-based
  `on_idle` starvation — §3.1).
- Score function unit-tested at the boundaries (grace edge, floor edge,
  zero retrievals).

### WP1b.2 — Near-duplicate merge at consolidation

**Files:** `corvidae/memory.py` (extends the 1a consolidation task, step
between embed and insert)

1. After embedding the new candidate record: top-1 vec query among
   `indexed=1` records in the same channel scope. If cosine similarity ≥
   `memory.dup_threshold` (default 0.95):
   - insert the new record as usual, then fold the old into it:
     `new.retrieval_count += old.retrieval_count`,
     `new.importance = max(new, old)`, extend `msg_id_start` to
     `min(old.msg_id_start, new.msg_id_start)` **only if the ranges are on
     the same channel** (they are, by scope);
   - mark the old: `superseded_by = new.id`, `indexed = 0`, delete its vec
     row. Superseded records are terminal — the retention job skips them
     (already excluded by `superseded_by IS NULL` in WP1b.1).
2. No LLM call in the merge path — this is the mechanical first pass; a
   summarizing merge is a §6-gated refinement, not this phase.
3. When the vec extension is unavailable, skip dup detection entirely (FTS
   similarity is not a substitute; log at DEBUG).

**Red tests** (`tests/test_memory_dedup.py`, stub embedder with controllable
similarity): near-identical consolidations → one indexed record carrying
summed stats and a superseded chain; sub-threshold pairs → both indexed;
`recall_raw` on a superseded record still works (raw range intact).

### WP1b.3 — `message_log` FTS + memory tools

**Files:** `corvidae/memory.py` (or `corvidae/tools/memory_tools.py` if
`memory.py` is past ~500 lines), `pyproject.toml` if a new module

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
    INSERT INTO message_fts(message_fts, rowid, content_text) VALUES ('delete', old.id,
        coalesce(json_extract(old.message, '$.content'), ''));
    INSERT INTO message_fts(rowid, content_text)
    VALUES (new.id, coalesce(json_extract(new.message, '$.content'), ''));
END;
```

   One-time backfill of pre-existing rows on first start (guard: row-count
   comparison), batched, in a silent startup task. The update trigger is
   what makes the redact cascade automatic on the message side — note it in
   WP1b.4.
2. **Tools** (registered via the standard `register_tools` hook; both update
   access stats per WP1b.1.3):
   - `search_memory(query: str, channel: str | None = None,
     tags: list[str] | None = None, after: str | None = None,
     before: str | None = None, include_demoted: bool = True) -> str` —
     `memory_fts MATCH` plus column filters; returns a compact numbered list
     (id, band-less score, age, flags `[demoted]`/`[superseded]`, summary),
     token-capped. This is Persyn §4.6's management surface (§3.1).
   - `recall_raw(memory_id: int, max_tokens: int = 1000) -> str` — the
     "remember harder" tool (§3.1): fetch the record's
     `msg_id_start..msg_id_end` rows from `message_log` verbatim, formatted
     `sender: content`, truncated to the token cap with an explicit
     truncation marker. Works on demoted, superseded, and (tombstoned)
     redacted ranges alike.

**Red tests** (`tests/test_memory_tools.py`): keyword search finds seeded
content with filters honored; demoted flagged; `recall_raw` returns verbatim
dialog for the range and respects the cap; both tools bump access stats;
backfill indexes pre-existing rows exactly once.

### WP1b.4 — `redact` CLI

**New file:** `corvidae/commands/redact.py`; entry point
`[project.entry-points."corvidae.commands"] redact = corvidae.commands.redact:redact_command`

```
corvidae redact --db sessions.db message <id> [<id2>...]   # message_log rows
corvidae redact --db sessions.db memory <memory_id>        # a record + its raw range
corvidae redact --db sessions.db range <start_id> <end_id> # message_log id range
```

Behavior (own connection; trap #6 discipline; every step in one short
transaction per row batch):
1. **Message tombstone:** rewrite the `message` JSON — keep `role` and
   structural keys, replace `content` with
   `"[redacted by operator {ISO-date}]"`, and drop `tool_calls`/tool
   payload fields. Row id and timestamp unchanged (trap #8). The WP1b.3
   update trigger propagates to `message_fts` automatically — but assert it
   in tests, not in faith (trap #2).
2. **Memory cascade:** for every `memory` row whose
   `[msg_id_start, msg_id_end]` intersects the redacted ids:
   `summary = '[redacted by operator]'` via UPDATE (the 1a `memory_au`
   trigger syncs `memory_fts`), `redacted=1`, `indexed=0`, delete its
   `memory_vec` row.
3. **Verification pass** (part of the command, printed): after the cascade,
   run `memory_fts` and `message_fts` MATCH queries for a sample token of
   the redacted text and report zero hits — the cascade proves itself on
   every invocation (§3.1: "the cascade must name FTS or the redacted text
   remains keyword-searchable").
4. `--dry-run` prints the affected row/record counts without writing.

**Red tests** (`tests/test_redact.py`, direct function invocation against a
temp DB): tombstone preserves row count/ids/roles; both FTS surfaces return
zero hits for redacted text afterward; vec rows gone; intersecting memory
records tombstoned; non-intersecting untouched; dry-run writes nothing;
non-WAL journal mode aborts with a clear error.

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
before implementing nothing — it should pass already; its value is the
floor).

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
  paragraph; `docs/plugin-guide.md` notes the tool surface.
