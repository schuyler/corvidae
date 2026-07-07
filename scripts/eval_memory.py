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

The retriever callable receives (query_text: str, memories: list[dict])
and returns a ranked list of memory ids (best first). It may be sync or
async. Phase 1b wires the real MemoryPlugin retrieval path here; until
then, pass any callable via --retriever to exercise the harness.

--judge reserves the summary-fidelity judging mode (LLM scores whether
retrieved summaries faithfully answer the query). It is a stub in Phase 0.
"""

import argparse
import asyncio
import importlib
import json
import sys
from pathlib import Path

# The deterministic metric functions live in tests/evals; make them
# importable when running from the repo root without installing tests.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tests"))

from evals.metrics import mrr, recall_at_k, tokens_of  # noqa: E402

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
    import tempfile

    import aiosqlite

    from corvidae.funnel import FunnelPlugin
    from corvidae.hooks import create_plugin_manager
    from corvidae.llm_plugin import LLMPlugin
    from corvidae.memory import MemoryPlugin
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
            "background": {
                "base_url": args.base_url, "model": args.model,
                "api_key": args.api_key,
            },
            "embedding": {
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
    await llm.on_start(config=config)

    usage = UsageLogPlugin()
    pm.register(usage, name="usage_log")
    await usage.on_init(pm=pm, config=config)

    funnel = FunnelPlugin()
    pm.register(funnel, name="funnel")
    await funnel.on_init(pm=pm, config=config)

    memory_plugin = MemoryPlugin()
    pm.register(memory_plugin, name="memory")
    await memory_plugin.on_init(pm=pm, config=config)
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
            await memory_plugin._consolidate_range(channel_id, max(ids))
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
        from corvidae.retention import run_retention_job
        await run_retention_job(memory_plugin)

    # Run the queries through the real retrieval path.
    channels = sorted({m["channel_id"] for m in memories})
    # Group all fixture channels so retrieval sees the whole corpus.
    memory_plugin._channel_groups = {"fixture": channels}

    recalls, mrrs = [], []
    positive = [q for q in queries if q["relevant"]]
    for query in positive:
        candidates, degraded = await memory_plugin.retrieve(
            channels[0], query["text"], k=args.k
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate memory retrieval quality")
    parser.add_argument(
        "--fixture", default=str(DEFAULT_FIXTURE),
        help=f"Path to memory-retrieval fixture JSON (default: {DEFAULT_FIXTURE})",
    )
    parser.add_argument(
        "--retriever", default=None,
        help="Retriever callable as 'pkg.module:callable' (wired to MemoryPlugin in Phase 1b)",
    )
    parser.add_argument("--k", type=int, default=5, help="Rank cutoff for recall@k (default: 5)")
    parser.add_argument(
        "--judge", action="store_true",
        help="Also run LLM-judge summary-fidelity scoring (stub in Phase 0)",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Run against the real MemoryPlugin stack with a scratch DB and "
             "a real llama-server (requires --base-url and --model)",
    )
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
    args = parser.parse_args()

    if args.live:
        asyncio.run(run_live_evaluation(args))
    else:
        asyncio.run(run_evaluation(args))


if __name__ == "__main__":
    main()
