#!/usr/bin/env python3
"""Evaluation runner for memory retrieval quality.

Scores a retriever against an operator-labeled fixture (format documented
in tests/evals/metrics.py) using the deterministic metrics recall@k and
MRR, plus token cost of the retrieved entries. Runs out-of-band only —
never imported by CI tests; live-LLM paths belong behind the `eval`
pytest marker.

Usage:
    python scripts/eval_memory.py [--fixture tests/fixtures/memory_retrieval_basic.json]
                                  [--retriever pkg.module:callable] [--k 5]
                                  [--judge] [--base-url URL] [--model MODEL] [--api-key KEY]
    python scripts/eval_memory.py --snapshot SESSIONS_DB --labels LABELS_JSON
                                  --config AGENT_YAML --pool-out POOL_JSON [--k N]

The retriever callable receives (query_text: str, memories: list[dict])
and returns a ranked list of memory ids (best first). It may be sync or
async. Phase 1b wires the real MemoryPlugin retrieval path here; until
then, pass any callable via --retriever to exercise the harness.

--judge reserves the summary-fidelity judging mode (LLM scores whether
retrieved summaries faithfully answer the query). It is a stub in Phase 0.

--snapshot (#658) scores a deployment's real memory. It copies SESSIONS_DB
with the sqlite backup API from a read-only connection (the original is never
opened for write, even while the daemon has it open in WAL mode), runs the real
MemoryPlugin retrieval path over the copy with the embedding profile and
memory.* settings of --config (loaded as the daemon loads agent.yaml), and
reports Recall@k and MRR per labelled query and as means. Retrieval only: no
chat request is made and no document is embedded. The copy lives in a 0700
temporary directory and is deleted at exit; it needs that much free space in
the system temp directory, and a SIGKILL mid-run leaves it there.

The config must still define a chat profile (the daemon refuses one without),
and that profile's api_key environment variable must be set, because the
daemon's client setup expands every profile's key; no request is sent to it.
--k defaults to memory.retrieval.k from --config in this mode (5 otherwise).
Scores decay against the wall clock of the run, which is recorded as run_at.
Without sqlite-vec, or when the config's encoder differs from the snapshot's,
retrieval degrades to FTS5; the run says so per query and in the summary (on an
install without sqlite-vec memory logs a load warning with a traceback).

LABELS_JSON (queries, and the snapshot's integer memory ids they should find;
an empty "relevant" list is a negative probe: its pool is written but it is
not scored; "description" and "note" are free text and unused):
    {"queries": [{"text": "what did we decide about the battery?",
                  "relevant": [412, 415]}]}
A query may also carry "context", the agent reply it followed (#665), and
"subset", a tag such as "deictic" or "self-contained". The query is built as
the daemon builds it (memory.retrieval.query_context; previous_reply puts the
context before the text), and the summary reports means per subset.
--query-context previous_reply|none overrides the config's value, so variant
A and the user-message-only baseline score against one snapshot, and
--candidates hybrid|vector overrides memory.retrieval.candidates (#666) to score
the fused pool against vec KNN alone.

POOL_JSON is written once after all queries: run_at, k, one entry per query
(index, text, context, subset, query (the text actually retrieved with),
relevant, unscorable ids with their status absent / redacted /
superseded / not_indexed, degraded, recall_at_k, mrr, and every scored
candidate as id, rank, similarity, score, band, source) and a summary (with
query_context, candidates and per-subset means). It holds no memory text beyond the
labels' own; look the ids up in the source.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import logging
import sqlite3
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

if TYPE_CHECKING:
    import aiosqlite
    from corvidae_cognition.memory import RetrievalPool

logger = logging.getLogger("eval_memory")

# The deterministic metric functions live in tests/evals; make them
# importable when running from the repo root without installing tests.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tests"))

from evals.metrics import false_admit, mrr, recall_at_k, tokens_of  # noqa: E402

FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures"
DEFAULT_FIXTURE = FIXTURE_DIR / "memory_retrieval_basic.json"


def load_fixture(fixture_path: str | Path) -> dict:
    """Load and minimally validate a memory-retrieval fixture."""
    with open(fixture_path) as f:
        fixture = json.load(f)
    missing = {"description", "memories", "queries"} - set(fixture)
    if missing:
        raise ValueError(f"fixture missing required keys: {sorted(missing)}")
    return fixture


def load_retriever(spec: str):
    """Import a retriever callable from a 'pkg.module:callable' spec."""
    module_name, _, attr = spec.partition(":")
    if not attr:
        raise ValueError(f"--retriever must be 'pkg.module:callable', got {spec!r}")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


async def rank(retriever, query_text: str, memories: list[dict]) -> list[str]:
    """Invoke a sync-or-async retriever and return its ranked id list."""
    result = retriever(query_text, memories)
    if asyncio.iscoroutine(result):
        result = await result
    return list(result)


async def run_live_evaluation(args) -> None:
    """WP1b.5 --live mode: real MemoryPlugin against a scratch DB.

    Spins up the memory stack with a real llama-server (base-url/model/api-key,
    mirroring eval_compaction.py), ingests the fixture conversation through the
    real consolidation path, runs the queries through the real retrieval path,
    and reports the deterministic metrics plus per-stage token cost from
    usage_log ("recall at a fixed token budget" — bootstrap-mapping §6).
    """

    import aiosqlite
    from corvidae_cognition.funnel import FunnelPlugin
    from corvidae_cognition.memory import MemoryPlugin

    from corvidae.hooks import create_plugin_manager
    from corvidae.llm_plugin import LLMPlugin
    from corvidae.metrics import UsageLogPlugin
    from corvidae.persistence import PersistencePlugin, init_db

    if not args.base_url or not args.model:
        print("--live requires --base-url and --model", file=sys.stderr)
        sys.exit(1)

    fixture = load_fixture(args.fixture)
    memories = fixture["memories"]
    queries = fixture["queries"]
    conversation = fixture.get("conversation") or []

    scratch = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    scratch.close()
    print(f"📖 Fixture: {args.fixture}")
    print(f"   scratch DB: {scratch.name}")

    config = {
        "llm": {
            "main": {
                "base_url": args.base_url, "model": args.model,
                "api_key": args.api_key,
            },
            "embedding": {
                "kind": "embedding",
                "base_url": args.embedding_base_url or args.base_url,
                "model": args.embedding_model or args.model,
                "dimensions": args.embedding_dimensions,
                "api_key": args.api_key,
            },
        },
        "memory": {},
    }

    db = await aiosqlite.connect(scratch.name)
    await init_db(db)

    pm = create_plugin_manager()
    persistence = PersistencePlugin()
    persistence.db = db
    pm.register(persistence, name="persistence")

    llm = LLMPlugin()
    pm.register(llm, name="llm")
    await llm.on_init(pm=pm, config=config)

    # MemoryPlugin registers its LLM stages when LLMPlugin starts, so it
    # must be registered first or get_client() can't resolve them.
    memory_plugin = MemoryPlugin()
    pm.register(memory_plugin, name="memory")
    await memory_plugin.on_init(pm=pm, config=config)

    await llm.on_start(config=config)

    usage = UsageLogPlugin()
    pm.register(usage, name="usage_log")
    await usage.on_init(pm=pm, config=config)

    funnel = FunnelPlugin()
    pm.register(funnel, name="funnel")
    await funnel.on_init(pm=pm, config=config)

    await memory_plugin.on_start(config=config)

    # Ingest the fixture conversation through the real consolidation path.
    # Conversation entries: {"channel_id": str, "messages": [{"role","content"},...]}
    # or a flat list of {"channel_id","role","content"} messages.
    segments: dict[str, list[int]] = {}
    if conversation:
        import time as _time
        for entry in conversation:
            if "messages" in entry:
                channel_id = entry["channel_id"]
                msgs = entry["messages"]
            else:
                channel_id = entry["channel_id"]
                msgs = [entry]
            for m in msgs:
                cursor = await db.execute(
                    "INSERT INTO message_log (channel_id, message, timestamp, "
                    "message_type) VALUES (?, ?, ?, ?)",
                    (channel_id,
                     json.dumps({"role": m["role"], "content": m["content"]}),
                     _time.time(), "message"),
                )
                segments.setdefault(channel_id, []).append(cursor.lastrowid)
        await db.commit()
        for channel_id, ids in segments.items():
            await memory_plugin._consolidate_range(channel_id, max(ids), "eval")
        await memory_plugin.wait_for_background_tasks()
        print(f"   ingested {sum(len(v) for v in segments.values())} messages "
              f"across {len(segments)} channels via consolidation")
    else:
        print("   ⚠️  fixture has no conversation; seeding memory summaries "
              "directly (retrieval-path-only evaluation)")
        import time as _time
        for record in memories:
            await db.execute(
                "INSERT INTO memory (channel_id, created_at, summary, "
                "importance, msg_id_start, msg_id_end, embedded) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (record["channel_id"], _time.time(), record["summary"],
                 0.5, 1, 2, 0),
            )
        await db.commit()
        # Backfill embeds the seeded records through the real embed path.
        from corvidae_cognition.retention import run_retention_job
        await run_retention_job(memory_plugin)

    # Run the queries through the real retrieval path.
    # Retrieval spans the whole memory, whatever channel wrote each record.
    recalls, mrrs = [], []
    positive = [q for q in queries if q["relevant"]]
    for query in positive:
        candidates, degraded = await memory_plugin.retrieve(
            query["text"], k=args.k
        )
        # Map retrieved summaries back to fixture ids by exact-summary match
        # (live consolidation rewrites summaries, so fall back to reporting raw).
        summary_to_fid = {m["summary"]: m["id"] for m in memories}
        ranked = [
            summary_to_fid.get(c["summary"], f"live:{c['id']}")
            for c in candidates
        ]
        query_recall = recall_at_k(ranked, query["relevant"], args.k)
        query_mrr = mrr(ranked, query["relevant"])
        recalls.append(query_recall)
        mrrs.append(query_mrr)
        print(f"Q: {query['text']}")
        print(f"   recall@{args.k}={query_recall:.2f}  mrr={query_mrr:.2f}  "
              f"degraded={degraded}  ranked={ranked[: args.k]}")

    if recalls:
        print(f"\n📊 Mean recall@{args.k}: {sum(recalls) / len(recalls):.3f}")
        print(f"📊 Mean MRR:        {sum(mrrs) / len(mrrs):.3f}")

    # Per-stage token cost from usage_log (§6's currency).
    try:
        async with db.execute(
            "SELECT stage, COUNT(*), SUM(prompt_tokens), SUM(completion_tokens) "
            "FROM usage_log GROUP BY stage"
        ) as cursor:
            rows = await cursor.fetchall()
        if rows:
            print("\n📊 Per-stage token cost (usage_log):")
            for stage, n, prompt_toks, completion_toks in rows:
                print(f"   {stage or '(none)'}: {n} calls, "
                      f"{prompt_toks or 0} prompt + {completion_toks or 0} completion tokens")
        else:
            print("\n📊 usage_log: no rows recorded")
    except Exception as exc:
        print(f"\n⚠️  usage_log unavailable: {exc}")

    if args.judge:
        print("\n⚠️  --judge (consolidation epistemic-framing fidelity) is "
              "reserved; LLM-judged scoring is out-of-band only (§6).")

    await llm.on_stop()
    await db.close()
    print(f"\nScratch DB retained at {scratch.name}")


async def run_evaluation(args) -> None:
    """Score every fixture query with the deterministic metrics and report."""
    fixture = load_fixture(args.fixture)
    memories = fixture["memories"]
    memories_by_id = {m["id"]: m for m in memories}
    queries = fixture["queries"]

    print(f"📖 Fixture: {args.fixture}")
    print(f"   {fixture['description']}")
    print(f"   {len(memories)} memories, {len(queries)} queries, k={args.k}")

    if args.retriever is None:
        print(
            "\n⚠️  No --retriever given. The live MemoryPlugin retrieval path "
            "is wired in Phase 1b; until then pass any 'pkg.module:callable' "
            "returning ranked memory ids to exercise the harness."
        )
        return

    retriever = load_retriever(args.retriever)

    # Token counting uses the same tiktoken path as the production context
    # window; fall back to a cheap word encoder if tiktoken is unavailable.
    from corvidae.context import _encoder

    class _WordEncoder:
        def encode(self, text: str) -> list[int]:
            return [0] * len(text.split())

    encoder = _encoder or _WordEncoder()

    recalls, mrrs = [], []
    print()
    for query in queries:
        ranked_ids = await rank(retriever, query["text"], memories)
        query_recall = recall_at_k(ranked_ids, query["relevant"], args.k)
        query_mrr = mrr(ranked_ids, query["relevant"])
        retrieved = [
            memories_by_id[mid] for mid in ranked_ids[: args.k] if mid in memories_by_id
        ]
        cost = tokens_of(retrieved, encoder)
        recalls.append(query_recall)
        mrrs.append(query_mrr)
        print(f"Q: {query['text']}")
        print(
            f"   recall@{args.k}={query_recall:.2f}  mrr={query_mrr:.2f}  "
            f"tokens@{args.k}={cost}  ranked={ranked_ids[: args.k]}"
        )

    print(f"\n📊 Mean recall@{args.k}: {sum(recalls) / len(recalls):.3f}")
    print(f"📊 Mean MRR:        {sum(mrrs) / len(mrrs):.3f}")

    if args.judge:
        # Reserved: LLM-judge summary-fidelity scoring (Phase 1b+). The
        # judge asks whether the retrieved summaries actually answer the
        # query, mirroring scripts/eval_compaction.py's scoring loop.
        print("\n⚠️  --judge is a stub in Phase 0; summary-fidelity judging lands with Phase 1b.")


# Snapshot-mode flags that are only meaningful together, with the remedy text
# naming what each one takes.
SNAPSHOT_REQUIRED_FLAGS: tuple[tuple[str, str, str], ...] = (
    ("--labels", "labels",
     "PATH (a JSON file of queries and relevant memory ids; see the module docstring)"),
    ("--config", "config", "PATH (the deployment's agent.yaml)"),
    ("--pool-out", "pool_out", "PATH (where to write the candidate pool JSON)"),
)


# Snapshot-only flags that pick a retrieval variant over the config's value:
# (flag, args attribute).
SNAPSHOT_VARIANT_FLAGS: tuple[tuple[str, str], ...] = (
    ("--query-context", "query_context"),
    ("--candidates", "candidates"),
)


@dataclass(frozen=True)
class LabelQuery:
    """One labelled query: its text and the snapshot memory ids it should find."""

    index: int
    text: str
    relevant: list[int]
    # The agent reply the probe followed (#665), or None for a probe that
    # opens a conversation; becomes the query's previous_reply context.
    context: str | None = None
    # Free tag grouping probes in the summary (e.g. deictic, self-contained).
    subset: str | None = None


def load_labels(path: Path) -> list[LabelQuery]:
    """Read and validate a labels file; raise ValueError naming the problem."""
    resolved = path.resolve()

    def bad(detail: str) -> ValueError:
        """Build the one-line labels error: file, detail, remedy."""
        return ValueError(f"labels {resolved}: {detail}; fix the labels file.")

    # Read and decode the JSON, naming an unreadable file or a syntax error.
    try:
        data = json.loads(resolved.read_text(encoding="utf-8"))
    except OSError as exc:
        raise bad(f"cannot read file ({exc.strerror or exc})") from exc
    except json.JSONDecodeError as exc:
        raise bad(f"invalid JSON ({exc.msg} at line {exc.lineno} column {exc.colno})") from exc

    # The top level must be an object holding a list under "queries".
    if not isinstance(data, dict) or "queries" not in data:
        raise bad("missing key 'queries'; expected {\"queries\": [...]}")
    raw_queries = data["queries"]
    if not isinstance(raw_queries, list):
        raise bad(f"queries: got {json.dumps(raw_queries)}; expected a list of queries")

    # Each query needs a non-empty text and a list of integer ids (bool is not one).
    queries: list[LabelQuery] = []
    for index, raw in enumerate(raw_queries):
        if not isinstance(raw, dict):
            raise bad(f"queries[{index}]: got {json.dumps(raw)}; expected an object")
        text = raw.get("text")
        if not isinstance(text, str) or not text:
            raise bad(f"queries[{index}].text: got {text!r}; expected a non-empty string")
        relevant = raw.get("relevant")
        if not isinstance(relevant, list) or not all(
            isinstance(i, int) and not isinstance(i, bool) for i in relevant
        ):
            raise bad(
                f"queries[{index}].relevant: got {json.dumps(relevant)}; "
                "expected a list of integer memory ids"
            )
        # Optional: the previous reply the probe followed, and its subset tag.
        context = raw.get("context")
        if context is not None and not isinstance(context, str):
            raise bad(f"queries[{index}].context: got {json.dumps(context)}; expected a string")
        subset = raw.get("subset")
        if subset is not None and (not isinstance(subset, str) or not subset):
            raise bad(
                f"queries[{index}].subset: got {json.dumps(subset)}; "
                "expected a non-empty string"
            )
        queries.append(LabelQuery(
            index=index, text=text, relevant=relevant, context=context, subset=subset,
        ))
    return queries


def copy_snapshot(source: Path, dest: Path) -> None:
    """Copy a database with the backup API from a read-only connection.

    The source is opened only through a ``mode=ro`` URI, so it is never
    written or checkpointed; the backup is one consistent snapshot even
    while a daemon writes to the source in WAL mode. Raises ValueError with
    the operator-facing text when the open or the backup fails.
    """
    resolved = source.resolve()
    uri = f"{resolved.as_uri()}?mode=ro"
    try:
        src = sqlite3.connect(uri, uri=True)
        try:
            dst = sqlite3.connect(dest)
            try:
                src.backup(dst)
            finally:
                dst.close()
        finally:
            src.close()
    except sqlite3.Error as exc:
        raise ValueError(
            f"cannot open snapshot {resolved} read-only ({exc}); check the "
            "--snapshot path and that the file is readable (for a database "
            "in WAL mode its directory must also be writable, for the -shm file)."
        ) from exc


async def label_status(db: aiosqlite.Connection, ids: set[int]) -> dict[int, str]:
    """Map each unreachable memory id to why; reachable ids are omitted.

    Reachable means the row exists with indexed=1 and redacted=0, the filter
    both candidate searches apply.
    """
    if not ids:
        return {}
    ordered = sorted(ids)
    marks = ",".join("?" for _ in ordered)
    async with db.execute(
        "SELECT id, indexed, superseded_by, redacted FROM memory "
        f"WHERE id IN ({marks})",
        ordered,
    ) as cursor:
        rows = {row[0]: row for row in await cursor.fetchall()}

    # Checked in order: a superseded row also has indexed=0.
    status: dict[int, str] = {}
    for ident in ordered:
        row = rows.get(ident)
        if row is None:
            status[ident] = "absent"
        elif row[3]:
            status[ident] = "redacted"
        elif row[2] is not None:
            status[ident] = "superseded"
        elif not row[1]:
            status[ident] = "not_indexed"
    return status


def candidate_entries(pool: RetrievalPool) -> list[dict]:
    """The pool's whole ranked list as the pool JSON's ``candidates``."""
    return [
        {
            "id": cand["id"],
            "rank": rank,
            "similarity": cand["similarity"],
            "score": cand["score"],
            "band": cand["band"],
            "source": cand["source"],
        }
        for rank, cand in enumerate(pool.ranked, start=1)
    ]


def probe_query(query: LabelQuery, query_context: str) -> str:
    """The retrieval query for one probe, built the way the daemon builds it.

    The probe becomes a two-row window (its context as the previous reply,
    then its text as the unanswered user message) and goes through
    ``retrieval_query``, so the eval cannot drift from ``before_agent_turn``.
    """
    from corvidae_cognition.memory import retrieval_query

    messages: list[dict] = []
    if query.context:
        messages.append({"role": "assistant", "content": query.context})
    messages.append({"role": "user", "content": query.text})
    return retrieval_query(messages, query_context)


# The summary group for probes with no subset tag.
UNTAGGED = "(untagged)"


def subset_summary(results: list[dict]) -> dict[str, dict]:
    """Scored count and mean Recall@k / MRR per subset tag, untagged grouped."""
    groups: dict[str, list[dict]] = {}
    for result in results:
        groups.setdefault(result["subset"] or UNTAGGED, []).append(result)
    summary: dict[str, dict] = {}
    for name, members in groups.items():
        scored_recalls = [r["recall_at_k"] for r in members if r["recall_at_k"] is not None]
        scored_mrrs = [r["mrr"] for r in members if r["mrr"] is not None]
        summary[name] = {
            "queries": len(members),
            "scored": len(scored_recalls),
            "mean_recall_at_k": mean(scored_recalls),
            "mean_mrr": mean(scored_mrrs),
        }
    return summary


def fail(message: str) -> NoReturn:
    """Print a one-line operator error to stderr and exit 1."""
    print(message, file=sys.stderr)
    raise SystemExit(1)


def fmt_metric(value: float | None) -> str:
    """Format a metric for stdout; None (not scored) prints as n/a."""
    return "n/a" if value is None else f"{value:.2f}"


def mean(values: list[float]) -> float | None:
    """Mean of the values, or None when there are none."""
    return sum(values) / len(values) if values else None


async def run_snapshot_evaluation(
    source: Path, labels_path: Path, config_path: Path,
    pool_out: Path, k: int | None, work_dir: Path,
    query_context: str | None = None,
    candidates: str | None = None,
) -> None:
    """Copy the snapshot, score the labelled queries, print and write the pool."""
    import aiosqlite
    from corvidae_cognition.memory import MemoryPlugin

    from corvidae.hooks import create_plugin_manager
    from corvidae.llm_plugin import LLMPlugin
    from corvidae.persistence import PersistencePlugin
    from corvidae.runtime import Runtime

    run_at = time.time()

    # Labels are validated before the copy so a typo costs no disk or time.
    try:
        queries = load_labels(labels_path)
    except ValueError as exc:
        fail(f"eval_memory: {exc}")

    # The copy: the only thing ever opened on the source path.
    snapshot_db = work_dir / "snapshot.db"
    try:
        copy_snapshot(source, snapshot_db)
    except ValueError as exc:
        fail(f"eval_memory: {exc}")

    db: aiosqlite.Connection | None = None
    llm: LLMPlugin | None = None
    llm_start_reached = False
    try:
        # The daemon's own YAML load (exits 1 with its Config error line).
        config = Runtime(config_path=str(config_path)).load_config()

        # Plugin setup on the copy; a ValueError here is a config problem.
        try:
            db = await aiosqlite.connect(str(snapshot_db))
            pm = create_plugin_manager()
            persistence = PersistencePlugin()
            persistence.db = db
            pm.register(persistence, name="persistence")

            # Memory goes before the LLM plugin so register_llm_stages reaches
            # it; its on_start is deliberately not run (no retention/backfill).
            memory = MemoryPlugin()
            pm.register(memory, name="memory")
            await memory.on_init(pm=pm, config=config)

            llm = LLMPlugin()
            pm.register(llm, name="llm")
            await llm.on_init(pm=pm, config=config)
            llm_start_reached = True
            await llm.on_start(config=config)
        except ValueError as exc:
            fail(f"Config error: {exc} — check {config_path}")

        # Unreachable labelled ids, over every query, from one lookup.
        all_ids = {ident for q in queries for ident in q.relevant}
        statuses = await label_status(db, all_ids)

        # A --query-context flag overrides the config's value (#665), so one
        # snapshot scores variant A and the baseline from the same config.
        if query_context is not None:
            memory._query_context = query_context
        # Likewise --candidates (#666): fused vs vec-only on one snapshot.
        if candidates is not None:
            memory._candidates = candidates
        effective_k = k if k is not None else memory._retrieval_k
        results: list[dict] = []
        print(f"📖 Snapshot: {source}")
        print(f"   {len(queries)} queries, k={effective_k}, "
              f"query_context={memory._query_context}, "
              f"candidates={memory._candidates}")
        for query in queries:
            reachable = [i for i in query.relevant if i not in statuses]
            unscorable = [
                {"id": i, "status": statuses[i]} for i in query.relevant if i in statuses
            ]
            text = probe_query(query, memory._query_context)
            pool = await memory._retrieve_pool(text, k=k, exclude=None)
            # Top k is the expression retrieve() returns.
            top_ids = [c["id"] for c in pool.ranked[: pool.k]]
            scored = bool(reachable)
            recall = recall_at_k(top_ids, reachable, pool.k) if scored else None
            reciprocal = mrr(top_ids, reachable) if scored else None
            fa = false_admit(candidate_entries(pool), effective_k) if not query.relevant else None
            results.append({
                "index": query.index,
                "text": query.text,
                "context": query.context,
                "subset": query.subset,
                "query": text,
                "relevant": query.relevant,
                "unscorable": unscorable,
                "degraded": pool.degraded,
                "recall_at_k": recall,
                "mrr": reciprocal,
                "false_admit": fa,
                "candidates": candidate_entries(pool),
            })
            print(f"\nQ: {query.text}")
            print(
                f"   recall@{pool.k}={fmt_metric(recall)}  mrr={fmt_metric(reciprocal)}  "
                f"degraded={pool.degraded}  ranked={top_ids}"
            )
            if unscorable:
                named = ", ".join(f"{u['id']} ({u['status']})" for u in unscorable)
                print(f"   unscorable: {named}")
            if not scored:
                print("   not scored: no reachable relevant id")

        # Summary over the scored queries; FTS5 fallback counted separately.
        recalls = [r["recall_at_k"] for r in results if r["recall_at_k"] is not None]
        mrrs = [r["mrr"] for r in results if r["mrr"] is not None]
        degraded = sum(1 for r in results if r["degraded"])
        fas = [r["false_admit"] for r in results if r["false_admit"] is not None]
        summary = {
            "queries": len(results),
            "scored": len(recalls),
            "degraded": degraded,
            "mean_recall_at_k": mean(recalls),
            "mean_mrr": mean(mrrs),
            "false_admit_rate": mean(fas) if fas else None,
            "query_context": memory._query_context,
            "candidates": memory._candidates,
            "subsets": subset_summary(results),
        }
        print(f"\n📊 Mean recall@{effective_k}: {fmt_metric(summary['mean_recall_at_k'])} "
              f"(over {len(recalls)} of {len(results)} queries)")
        print(f"📊 Mean MRR:        {fmt_metric(summary['mean_mrr'])} "
              f"(over {len(recalls)} of {len(results)} queries)")
        if fas:
            print(f"📊 False admits@{effective_k}:   {summary['false_admit_rate']:.2f} "
                  f"({sum(fas)} of {len(fas)} negative probes)")
        # Per-subset means, only worth printing when the probes are tagged.
        if any(r["subset"] for r in results):
            for name, group in summary["subsets"].items():
                print(f"   {name}: recall@{effective_k}="
                      f"{fmt_metric(group['mean_recall_at_k'])}  "
                      f"mrr={fmt_metric(group['mean_mrr'])}  "
                      f"(over {group['scored']} of {group['queries']} queries)")
        if degraded:
            print(f"⚠️  FTS5 fallback on {degraded} of {len(results)} queries")

        pool_out.write_text(json.dumps(
            {"run_at": run_at, "k": effective_k, "queries": results, "summary": summary},
            indent=2,
        ), encoding="utf-8")
        print(f"\nPool written to {pool_out}")
    finally:
        # Release everything before main's temp dir deletes the copy; a
        # failure here is logged and must not mask the original exception.
        if llm is not None and llm_start_reached:
            try:
                await llm.on_stop()
            except Exception:
                logger.warning("eval_memory: stopping LLM clients failed", exc_info=True)
        if db is not None:
            try:
                await db.close()
            except Exception:
                logger.warning(
                    "eval_memory: closing the snapshot copy %s failed", snapshot_db,
                    exc_info=True,
                )


def snapshot_arg_error(args: argparse.Namespace) -> str | None:
    """The flag-combination rules; None when valid, else the error text."""
    # Without --snapshot, the snapshot-only flags have nothing to attach to.
    if args.snapshot is None:
        for flag, attr in (
            *((f, a) for f, a, _hint in SNAPSHOT_REQUIRED_FLAGS),
            *SNAPSHOT_VARIANT_FLAGS,
        ):
            if getattr(args, attr) is not None:
                return f"{flag} is only used with --snapshot; add --snapshot PATH or drop {flag}."
        return None

    # With --snapshot, each of the three paths is required.
    for flag, attr, hint in SNAPSHOT_REQUIRED_FLAGS:
        if getattr(args, attr) is None:
            return f"--snapshot needs {flag} {hint}."

    # A rank cutoff below 1 would silently fall back to the config's k (0)
    # or drop candidates (negative).
    if args.k is not None and args.k <= 0:
        return (
            f"--k must be a positive integer, got {args.k}; pass --k 1 or "
            "more, or omit it to use memory.retrieval.k."
        )

    # The pool file must not be the source (or its WAL/shm files), the
    # labels or the config it reads (S1).
    target = args.pool_out.resolve()
    source = args.snapshot.resolve()
    protected = (
        ("--snapshot", source),
        ("--snapshot's WAL file", source.with_name(source.name + "-wal")),
        ("--snapshot's shared-memory file", source.with_name(source.name + "-shm")),
        ("--labels", args.labels.resolve()),
        ("--config", args.config.resolve()),
    )
    for flag, other in protected:
        if target == other:
            return (
                f"--pool-out {target} is the same file as {flag}; choose a "
                "different --pool-out path so that file is not overwritten."
            )
    return None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate memory retrieval quality")
    parser.add_argument(
        "--fixture", default=str(DEFAULT_FIXTURE),
        help=f"Path to memory-retrieval fixture JSON (default: {DEFAULT_FIXTURE})",
    )
    parser.add_argument(
        "--retriever", default=None,
        help="Retriever callable as 'pkg.module:callable' (wired to MemoryPlugin in Phase 1b)",
    )
    parser.add_argument(
        "--k", type=int, default=None,
        help="Rank cutoff for recall@k (default: 5; snapshot mode: memory.retrieval.k)",
    )
    parser.add_argument(
        "--judge", action="store_true",
        help="Also run LLM-judge summary-fidelity scoring (stub in Phase 0)",
    )
    # --live and --snapshot select different modes, so argparse enforces
    # that at most one is given.
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--live", action="store_true",
        help="Run against the real MemoryPlugin stack with a scratch DB and "
             "a real llama-server (requires --base-url and --model)",
    )
    mode.add_argument(
        "--snapshot", type=Path, default=None, metavar="SESSIONS_DB",
        help="Score a copy of this sessions.db with the real retrieval path "
             "(requires --labels, --config and --pool-out)",
    )
    parser.add_argument("--labels", type=Path, default=None,
                        help="Snapshot mode: labels JSON of queries and relevant memory ids")
    parser.add_argument("--config", type=Path, default=None,
                        help="Snapshot mode: the deployment's agent.yaml")
    parser.add_argument("--pool-out", type=Path, default=None,
                        help="Snapshot mode: where to write the candidate pool JSON")
    parser.add_argument("--query-context", default=None,
                        choices=("previous_reply", "none"),
                        help="Snapshot mode: override memory.retrieval.query_context "
                             "(previous_reply puts each probe's context before its text)")
    parser.add_argument("--candidates", default=None, choices=("hybrid", "vector"),
                        help="Snapshot mode: override memory.retrieval.candidates "
                             "(hybrid fuses vec KNN with FTS5; vector is KNN alone)")
    # LLM connection flags; mirrors scripts/eval_compaction.py.
    parser.add_argument("--base-url", default=None, help="LLM API base URL")
    parser.add_argument("--model", default=None, help="LLM model name")
    parser.add_argument("--api-key", default=None, help="LLM API key")
    parser.add_argument("--embedding-base-url", default=None,
                        help="Embedding API base URL (--live; defaults to --base-url)")
    parser.add_argument("--embedding-model", default=None,
                        help="Embedding model name (--live; defaults to --model)")
    parser.add_argument("--embedding-dimensions", type=int, default=768,
                        help="Embedding dimensions (--live; default 768)")
    args = parser.parse_args(argv)

    # Combination rules are checked before any I/O.
    error = snapshot_arg_error(args)
    if error is not None:
        parser.error(error)

    if args.snapshot is not None:
        # The copy lives in a 0700 temp dir that is deleted when the run ends.
        with tempfile.TemporaryDirectory() as work:
            asyncio.run(run_snapshot_evaluation(
                args.snapshot, args.labels, args.config, args.pool_out,
                args.k, Path(work), query_context=args.query_context,
                candidates=args.candidates,
            ))
        return

    # Fixture and --live modes keep their historical default cutoff.
    if args.k is None:
        args.k = 5

    if args.live:
        asyncio.run(run_live_evaluation(args))
    else:
        asyncio.run(run_evaluation(args))


if __name__ == "__main__":
    main()
