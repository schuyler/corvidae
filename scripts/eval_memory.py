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
    # LLM connection flags for --judge mode; mirrors scripts/eval_compaction.py.
    parser.add_argument("--base-url", default=None, help="LLM API base URL (judge mode)")
    parser.add_argument("--model", default=None, help="LLM model name (judge mode)")
    parser.add_argument("--api-key", default=None, help="LLM API key (judge mode)")
    args = parser.parse_args()

    asyncio.run(run_evaluation(args))


if __name__ == "__main__":
    main()
