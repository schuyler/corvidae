"""Deterministic metric functions for memory-retrieval evals.

Pure functions, no I/O — safe to run in CI forever. The LLM-judge runner
(scripts/eval_memory.py) composes these; anything requiring a live model
stays out-of-band behind the `eval` pytest marker.

Fixture format (JSON, operator-authored labels — see bootstrap-mapping §6):

    {
      "description": "what this fixture exercises",
      "conversation": [
        {"role": "user", "content": "...", "sender": "...", "ts": 0}
      ],
      "memories": [
        {"id": "m1", "summary": "...", "channel_id": "..."}
      ],
      "queries": [
        {"text": "...", "relevant": ["m1"], "note": "why m1 is the answer"}
      ]
    }

- ``conversation`` is the raw material the memories were consolidated from
  (may be empty for synthetic fixtures).
- ``memories`` is the corpus a retriever is expected to index.
- ``queries`` are labeled probes: ``relevant`` lists the memory ids a good
  retriever should surface for ``text``; ``note`` records the operator's
  reasoning so labels stay auditable.

The seed fixture shipped in Phase 0 is a small hand-written set proving
the plumbing; real fixtures accumulate during Phase 1.
"""

from collections.abc import Sequence


def recall_at_k(
    ranked_ids: Sequence[str], relevant_ids: Sequence[str], k: int
) -> float:
    """Fraction of relevant ids that appear in the top-k of the ranking.

    Edge cases: an empty ``relevant_ids`` is vacuously perfect (1.0 —
    nothing was missed); ``k <= 0`` considers no results, scoring 0.0
    against a non-empty relevant set.
    """
    relevant = set(relevant_ids)
    if not relevant:
        return 1.0
    if k <= 0:
        return 0.0
    top_k = set(ranked_ids[:k])
    return len(relevant & top_k) / len(relevant)


def mrr(ranked_ids: Sequence[str], relevant_ids: Sequence[str]) -> float:
    """Reciprocal rank of the first relevant id in the ranking.

    Returns 0.0 when no relevant id is ranked (or relevant is empty).
    """
    relevant = set(relevant_ids)
    for position, ranked_id in enumerate(ranked_ids, start=1):
        if ranked_id in relevant:
            return 1.0 / position
    return 0.0


def tokens_of(entries: Sequence, encoder) -> int:
    """Total token count of a list of entries under the given encoder.

    Each entry is either a plain string or a dict whose text lives in
    ``content`` (message dicts) or ``summary`` (memory dicts). Non-string
    and missing text counts as 0. The production encoder is the tiktoken
    cl100k_base instance from ``corvidae.context`` — pass it in; this
    function does no I/O of its own.
    """
    total = 0
    for entry in entries:
        if isinstance(entry, str):
            text = entry
        else:
            text = entry.get("content") or entry.get("summary") or ""
        if isinstance(text, str) and text:
            total += len(encoder.encode(text))
    return total
