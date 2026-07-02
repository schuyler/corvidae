# Phase 0 — Observability, attribution, and eval-harness foundations

**Effort:** M–L. **Dependencies:** none (this phase is independent of all
later phases and unblocks their measurement).
**Normative references:** `bootstrap-mapping.md` §3.7, §6, §7 row 0;
`plans/new-hooks.md` (the `on_llm_request`/`on_llm_response`/`on_metrics`
designs — this phase resolves its open question #1 in favor of `LLMClient`).

**Goal:** every LLM call in the system is metered at the single chokepoint
all calls pass through, with attribution (stage, channel) that survives
background-task boundaries; token/latency records land in SQLite and JSONL;
the outcome-log table exists (schema + writer API, populated by later
phases); and the eval harness has its skeleton (fixture format, deterministic
metric functions, out-of-band LLM-judge runner).

## Read first

- `plans/bootstrap-mapping.md` §3.7 (metering site, contextvars correction,
  eval-readiness acceptance criterion) and §6 (ground truth and CI
  discipline).
- `corvidae/llm.py` — `LLMClient.chat()` is the chokepoint. Note it has no
  plugin-system dependency; keep it that way (see traps).
- `corvidae/llm_plugin.py` — `LLMPlugin` creates clients and holds `self.pm`;
  this is where observers get injected.
- `corvidae/task.py` — `Task` dataclass and `TaskQueue._run_one_worker`;
  attribution must cross this boundary.
- `corvidae/persistence.py` — `init_db()` pattern for DDL; `PersistencePlugin.db`
  is the shared connection, accessed by other plugins via
  `get_dependency(self.pm, "persistence")`.
- `corvidae/jsonl_log.py` — the pattern for a fail-soft JSONL sink plugin.
- `scripts/eval_compaction.py` + `tests/fixtures/death_spiral_compaction.json`
  — the existing eval pattern to generalize.
- `pyproject.toml` — entry points under `[project.entry-points.corvidae]`;
  the `eval` pytest marker already exists.

## Design constraints and traps (violating any of these is a bug)

1. **Meter at `LLMClient`, not `run_agent_turn`.** Compaction, consolidation,
   critique, and subagent calls all bypass the turn loop; metering the turn
   loop misses the majority of future spend. Supporting evidence that the
   turn loop is the wrong site: `agent.py` (step-11 timing block, currently
   ~line 582) reads `result.message.get("usage")` — but `run_agent_turn`
   keeps only `response["choices"][0]["message"]` (`turn.py` ~line 75) and
   discards the envelope where `usage` actually lives, so that field is
   always `None` today. Do not "fix" that read; it becomes obsolete — usage
   flows through `on_llm_response`.
2. **Keep `corvidae/llm.py` free of pluggy imports.** `LLMClient` must not
   know about the plugin system (layering). It gains an optional *observer*
   attribute (plain async callables, default `None`); `LLMPlugin` injects an
   observer that fires the hooks. If the observer is `None`, behavior is
   byte-for-byte today's.
3. **Observer failures never break an LLM call.** Every observer invocation
   is wrapped: `except Exception: logger.warning(..., exc_info=True)`. A
   metering bug must not take down the agent.
4. **`contextvars` do NOT propagate into TaskQueue workers.** Contextvars
   snapshot at `asyncio.create_task` time; the worker coroutines are created
   once at startup (`task.py`, `TaskPlugin.on_start`), so a var set at
   enqueue time is invisible when `task.work()` runs. The fix is mandatory,
   not optional: `Task` captures `contextvars.copy_context()` at creation
   and the worker runs the work inside that context (see WP0.2).
5. **`on_metrics` reentrancy** (from `plans/new-hooks.md`): never call
   `pm.ahook.on_metrics(...)` from inside an `on_metrics` implementation —
   pluggy dispatches back into the same implementation and recurses forever.
6. **The outcome log is schema-only this phase.** Its retrieval-profile
   columns are populated from Phase 1a (there is no retrieval to profile
   before then) and its appraisal/origin columns from Phase 2. Do not stub
   fake writers; ship the table, the writer API, and its tests.
7. **CI metrics are deterministic; LLM-judge is out-of-band** (§6). Anything
   under `tests/` that runs in CI must be red/green with no network. The
   LLM-judge runner lives in `scripts/` and/or behind the existing `eval`
   pytest marker (deselected by default, run with `--run-eval`).

## Work packages (in order)

### WP0.1 — Attribution contextvar module

**New file:** `corvidae/attribution.py`

```python
"""Call attribution for observability.

A single ContextVar holds a small dict describing what the current code
path is doing on whose behalf. LLMPlugin's observer reads it when a call
fires; callers set it at the top of a logical operation.
"""
import contextvars

_attribution: contextvars.ContextVar[dict] = contextvars.ContextVar(
    "corvidae_attribution", default={}
)

def set_attribution(**fields) -> contextvars.Token:
    """Merge fields into the current attribution; returns a reset token."""
    merged = {**_attribution.get(), **fields}
    return _attribution.set(merged)

def get_attribution() -> dict:
    """Return the current attribution dict (possibly empty). Never None."""
    return _attribution.get()

def reset_attribution(token: contextvars.Token) -> None:
    _attribution.reset(token)
```

Recognized fields (document in the module docstring; the dict is open):
`stage` (str: `"turn"`, `"compaction"`, `"subagent"`, later `"consolidation"`,
`"appraisal"`, `"critique"`), `channel_id` (str), `exchange_key` (str,
Phase 2).

**Call sites to set attribution** (each sets at operation start, resets in a
`finally`):
- `Agent._process_queue_item` — `stage="turn", channel_id=channel.id`, set
  just before step 7's `_run_turn` call (or at the top of the method; either
  is fine as long as compaction inside the turn gets re-labeled, next bullet).
- `CompactionPlugin` (`compaction.py`, in the method that makes the summary
  LLM call) — `stage="compaction", channel_id=channel.id`. Set/reset around
  the LLM call only, so it correctly shadows the turn attribution and
  restores it after.
- `tools/subagent.py` — `stage="subagent", channel_id=channel.id` around the
  subagent's LLM calls.

**Red tests** (`tests/test_attribution.py`):
- `set_attribution` merges rather than replaces; `reset_attribution` restores.
- Attribution set in one asyncio task is invisible in a sibling task created
  before the set (documents the propagation rule).
- `get_attribution()` returns `{}` (not None) when nothing was set.

### WP0.2 — Attribution across the TaskQueue boundary

**Files:** `corvidae/task.py`

1. `Task` gains a field:
   `ctx: contextvars.Context = field(default_factory=contextvars.copy_context)`
   — captured automatically at Task creation time, i.e. in the enqueuing
   caller's context.
2. In `TaskQueue._run_one_worker`, run the work inside the captured context.
   With `requires-python >= 3.13`, the supported form is:
   `result = await asyncio.create_task(task.work(), context=task.ctx)`
   (wrap in the existing exception handling; the extra task indirection is
   the accepted cost of context entry for coroutines).
3. No behavior change for tasks whose creator set no attribution.

**Red tests** (`tests/test_task.py`, add to existing):
- Enqueue a task from inside a context where `set_attribution(stage="x")`
  was called; the task body reads `get_attribution()["stage"] == "x"` even
  though the worker was started before the attribution was set.
- Two tasks enqueued from different attribution contexts each see their own.

### WP0.3 — `on_llm_request` / `on_llm_response` hookspecs and the client observer

**Files:** `corvidae/hooks.py`, `corvidae/llm.py`, `corvidae/llm_plugin.py`

1. **Hookspecs** (add to `AgentSpec` in `hooks.py`; broadcast, side-effect
   only, following the existing docstring style):

```python
@hookspec
async def on_llm_request(
    self,
    role: str,            # LLMPlugin role that made the call ("main", "background", ...)
    model: str,
    request_id: str,      # uuid hex minted per call; pairs request with response
    message_count: int,
    tool_count: int,
    attribution: dict,    # snapshot of corvidae.attribution.get_attribution()
) -> None: ...

@hookspec
async def on_llm_response(
    self,
    role: str,
    model: str,
    request_id: str,
    usage: dict | None,   # the response's "usage" field verbatim, or None
    latency_ms: float,
    attribution: dict,
    error: str | None,    # None on success; exception string on terminal failure
) -> None: ...
```

   Note the request payload is summarized (counts), not shipped wholesale —
   full messages in a broadcast hook would copy the entire prompt per call.
   If a debugging consumer needs full payloads later, that is a new hook, not
   a widening of this one.

2. **`LLMClient` observer seam** (`llm.py`): add constructor/attribute
   `observer: object | None = None` where an observer is any object with
   async methods `request(**kwargs)` and `response(**kwargs)` matching the
   hookspec fields. In `chat()`:
   - mint `request_id = uuid.uuid4().hex[:12]` per call;
   - fire `observer.request(...)` immediately before the retry loop;
   - fire `observer.response(...)` after success (with `usage`, `latency_ms`,
     `error=None`) — reuse the already-computed `elapsed`;
   - fire `observer.response(...)` with `usage=None, error=str(exc)` when a
     terminal (non-retried) exception is about to propagate;
   - every observer call is individually try/excepted (trap #3). Retried
     transient attempts do NOT fire `response`; one request → exactly one
     `response`.
3. **Observer injection** (`llm_plugin.py`): `LLMPlugin._create_client`
   gains a `role: str` argument; after constructing each client, set
   `client.observer = _HookObserver(self.pm, role, cfg["model"])` where
   `_HookObserver.request/response` call `self.pm.ahook.on_llm_request/
   on_llm_response` with `attribution=get_attribution()`. (The observer is
   defined in `llm_plugin.py`, keeping `llm.py` pluggy-free — trap #2.)

**Red tests** (`tests/test_llm_hooks.py`):
- With an observer installed and a stubbed HTTP session, `chat()` fires
  request then response exactly once, with matching `request_id`, correct
  `usage` passthrough, and `latency_ms > 0`.
- A raising observer does not prevent `chat()` from returning the response
  (assert a warning was logged; do not assert on its text).
- Terminal HTTP error → one `response` with `error` set and `usage=None`.
- Transient-then-success (429 then 200) → exactly one `response`.
- `LLMPlugin` wires observers on both main and background clients; hook
  receives the right `role`.

### WP0.4 — `on_metrics` hookspec + built-in emission + sinks

**Files:** `corvidae/hooks.py`, new `corvidae/metrics.py`, `pyproject.toml`

1. **Hookspec** — verbatim from `plans/new-hooks.md`:
   `on_metrics(name: str, value: float, tags: dict[str, str]) -> None`,
   broadcast, side-effect only. Document the reentrancy constraint (trap #5)
   in the docstring.
2. **`MetricsPlugin`** (`corvidae/metrics.py`, entry point `metrics`):
   implements `on_llm_response` and emits (never from inside `on_metrics`):
   - `llm.tokens.prompt`, `llm.tokens.completion`, `llm.tokens.total`
     (skip any the usage dict lacks),
   - `llm.latency_ms`,
   - `llm.errors` (value 1.0) when `error` is not None;
   with tags `{"role": role, "model": model, "stage": attribution.get("stage", ""),
   "channel": attribution.get("channel_id", "")}`. All emission fail-soft.
3. **Sinks:**
   - `UsageLogPlugin` (also in `metrics.py`, entry point `usage_log`;
     `depends_on = frozenset({"persistence"})`): implements
     `on_llm_response`, writes one row per call into `usage_log` via the
     persistence plugin's connection
     (`get_dependency(self.pm, "persistence").db`). DDL (create in its
     `on_start`, same `CREATE TABLE IF NOT EXISTS` pattern as `init_db`):

     ```sql
     CREATE TABLE IF NOT EXISTS usage_log (
         id INTEGER PRIMARY KEY AUTOINCREMENT,
         ts REAL NOT NULL,
         request_id TEXT NOT NULL,
         role TEXT NOT NULL,
         model TEXT NOT NULL,
         stage TEXT,
         channel_id TEXT,
         exchange_key TEXT,          -- NULL until Phase 2
         prompt_tokens INTEGER,
         completion_tokens INTEGER,
         total_tokens INTEGER,
         latency_ms REAL,
         error TEXT
     );
     CREATE INDEX IF NOT EXISTS idx_usage_ts ON usage_log (ts);
     ```
   - `MetricsJsonlPlugin` (entry point `metrics_jsonl`): `on_metrics`
     consumer appending `{"ts": ..., "name": ..., "value": ..., "tags": ...}`
     lines to a configurable path (config `daemon.metrics_jsonl`; disabled
     when unset). Mirror `JsonlLogPlugin`'s open/rotate/fail-soft behavior.
4. All three plugins registered in `pyproject.toml` entry points.

**Red tests** (`tests/test_metrics.py`):
- `on_llm_response` with a usage dict → the three token metrics + latency
  emitted with correct tags; missing usage → only latency (+ error metric on
  error).
- `UsageLogPlugin` writes a row readable back with the same request_id.
- JSONL sink writes one valid-JSON line per event; disabled when unconfigured.
- Reentrancy guard: a consumer that emits from `on_metrics` is a
  documentation constraint, not enforced — no test; instead test that two
  registered consumers both receive the same event (broadcast).

### WP0.5 — Outcome-log (exchange) table: schema + writer API only

**Files:** `corvidae/metrics.py` (or a new `corvidae/outcome_log.py` if
`metrics.py` grows past ~300 lines), `pyproject.toml`

DDL (created via the same `on_start` pattern, persistence connection):

```sql
CREATE TABLE IF NOT EXISTS exchange_log (
    exchange_key TEXT PRIMARY KEY,
    channel_id TEXT NOT NULL,
    origin TEXT,                    -- 'user'|'reminder'|'critique'|'heartbeat'|'task'; NULL until Phase 2
    message_rowid INTEGER,          -- message_log.id of the originating message; NULL for gate-rejected
    created_at REAL NOT NULL,
    retrieval_top_score REAL,       -- Phase 1a
    retrieval_hit_count INTEGER,    -- Phase 1a
    probe_score REAL,               -- Phase 2
    appraisal TEXT,                 -- JSON vector; Phase 2
    provenance_snapshot TEXT,       -- JSON; Phase 2
    outcomes TEXT                   -- JSON (critique verdicts, engagement outcomes); Phase 2
);
CREATE INDEX IF NOT EXISTS idx_exchange_channel ON exchange_log (channel_id, created_at);
```

Writer API on the owning plugin (public methods, used by Phases 1–2):
`async def record_exchange(exchange_key, channel_id, origin=None, message_rowid=None)`
(INSERT OR IGNORE) and
`async def update_exchange(exchange_key, **columns)` (guarded UPDATE of the
named nullable columns only — reject unknown column names, do not build SQL
from arbitrary kwargs).

**Red tests:** insert/update round-trip; unknown column rejected; duplicate
`record_exchange` is idempotent.

### WP0.6 — Eval-harness foundations

**Files:** new `tests/evals/__init__.py`, `tests/evals/metrics.py`,
`tests/evals/test_metrics.py`, `tests/fixtures/memory_retrieval_basic.json`,
new `scripts/eval_memory.py`

1. **Fixture format** (documented in a module docstring in
   `tests/evals/metrics.py`; JSON):

```json
{
  "description": "...",
  "conversation": [{"role": "user", "content": "...", "sender": "...", "ts": 0}],
  "memories": [{"id": "m1", "summary": "...", "channel_id": "..."}],
  "queries": [{"text": "...", "relevant": ["m1"], "note": "why m1 is the answer"}]
}
```

   Labels are **operator-authored** (§6) — the fixture shipped in this phase
   is a small hand-written seed (5–10 memories, 3–5 queries) proving the
   plumbing; real fixtures accumulate during Phase 1.
2. **Deterministic metric functions** (`tests/evals/metrics.py`, pure
   functions, no I/O): `recall_at_k(ranked_ids, relevant_ids, k) -> float`,
   `mrr(ranked_ids, relevant_ids) -> float`,
   `tokens_of(entries, encoder) -> int` (reuse the tiktoken path from
   `corvidae/context.py`). Unit-tested exhaustively in
   `tests/evals/test_metrics.py` — these run in CI forever, get them right.
3. **LLM-judge runner skeleton** (`scripts/eval_memory.py`): mirror
   `scripts/eval_compaction.py`'s argparse/fixture/report structure; takes a
   fixture and a live retrieval callable (wired in Phase 1b), scores with the
   deterministic metrics, and reserves a `--judge` mode stub for
   summary-fidelity judging. Runs out-of-band only (never imported by CI
   tests). Tests that exercise it live are marked `@pytest.mark.eval`
   (marker already configured in `pyproject.toml`).

## Non-goals (do not build in this phase)

- No retrieval, no memory tables, no embeddings (Phase 1a).
- No exchange-key minting, no gate hooks, no appraisal (Phase 2) — the
  `exchange_key` columns stay NULL.
- No Prometheus/OTel exporters — the `on_metrics` hookspec is the seam;
  external exporters are third-party plugin territory.
- Do not implement the other hooks from `plans/new-hooks.md`
  (`should_compact_conversation`, `filter_tools`, etc.) — they are unrelated
  to this phase.

## Definition of done

- All red tests above green; full suite passes (`uv run pytest`).
- Running the daemon with a local llama-server shows: usage rows in
  `usage_log` for turn AND compaction calls, with correct `stage` tags
  (this is the acceptance check that the metering site is right — force a
  compaction and verify its row).
- A task enqueued during a turn produces LLM rows attributed to the turn's
  channel (the contextvars fix demonstrably works across the queue).
- `tests/evals/test_metrics.py` green in CI; `scripts/eval_memory.py --help`
  runs.
- `docs/plugin-guide.md` documents the three new hookspecs;
  `docs/configuration.md` documents `daemon.metrics_jsonl`.
