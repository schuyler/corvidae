"""Exhaustive unit tests for the deterministic eval metric functions.

These run in CI forever — they define the scoring semantics for every
memory-retrieval eval in later phases.
"""

import json
from pathlib import Path

import pytest

from evals.metrics import mrr, recall_at_k, tokens_of

FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent / "fixtures" / "memory_retrieval_basic.json"
)


class TestRecallAtK:
    def test_perfect_recall(self):
        assert recall_at_k(["m1", "m2", "m3"], ["m1", "m2"], k=3) == 1.0

    def test_partial_recall(self):
        assert recall_at_k(["m1", "m9", "m2"], ["m1", "m2", "m3"], k=3) == pytest.approx(2 / 3)

    def test_zero_recall(self):
        assert recall_at_k(["m8", "m9"], ["m1"], k=2) == 0.0

    def test_k_cuts_off_ranked_list(self):
        # m2 is ranked, but below the k cutoff.
        assert recall_at_k(["m9", "m8", "m2"], ["m2"], k=2) == 0.0
        assert recall_at_k(["m9", "m8", "m2"], ["m2"], k=3) == 1.0

    def test_k_larger_than_ranked_list(self):
        assert recall_at_k(["m1"], ["m1", "m2"], k=10) == 0.5

    def test_empty_ranked_list(self):
        assert recall_at_k([], ["m1"], k=5) == 0.0

    def test_empty_relevant_is_vacuously_perfect(self):
        # No relevant items to find → nothing was missed.
        assert recall_at_k(["m1"], [], k=5) == 1.0

    def test_k_zero_scores_zero_against_nonempty_relevant(self):
        assert recall_at_k(["m1"], ["m1"], k=0) == 0.0

    def test_duplicate_ranked_ids_do_not_double_count(self):
        assert recall_at_k(["m1", "m1", "m1"], ["m1", "m2"], k=3) == 0.5

    def test_returns_float(self):
        assert isinstance(recall_at_k(["m1"], ["m1"], k=1), float)


class TestMRR:
    def test_first_position(self):
        assert mrr(["m1", "m2"], ["m1"]) == 1.0

    def test_second_position(self):
        assert mrr(["m9", "m1"], ["m1"]) == 0.5

    def test_third_position(self):
        assert mrr(["m9", "m8", "m1"], ["m1"]) == pytest.approx(1 / 3)

    def test_first_relevant_wins_among_several(self):
        # Reciprocal rank of the FIRST relevant hit.
        assert mrr(["m9", "m2", "m1"], ["m1", "m2"]) == 0.5

    def test_no_relevant_in_ranking(self):
        assert mrr(["m8", "m9"], ["m1"]) == 0.0

    def test_empty_ranked_list(self):
        assert mrr([], ["m1"]) == 0.0

    def test_empty_relevant(self):
        assert mrr(["m1"], []) == 0.0

    def test_returns_float(self):
        assert isinstance(mrr(["m1"], ["m1"]), float)


class _StubEncoder:
    """Deterministic encoder stub: one token per whitespace-separated word."""

    def encode(self, text: str) -> list[int]:
        return [0] * len(text.split())


class TestTokensOf:
    def test_counts_content_of_message_dicts(self):
        entries = [
            {"role": "user", "content": "one two three"},
            {"role": "assistant", "content": "four five"},
        ]
        assert tokens_of(entries, _StubEncoder()) == 5

    def test_counts_summary_field_when_content_absent(self):
        entries = [{"id": "m1", "summary": "a b c d"}]
        assert tokens_of(entries, _StubEncoder()) == 4

    def test_plain_strings_accepted(self):
        assert tokens_of(["one two", "three"], _StubEncoder()) == 3

    def test_empty_and_none_content_count_zero(self):
        entries = [{"content": ""}, {"content": None}, {}]
        assert tokens_of(entries, _StubEncoder()) == 0

    def test_empty_entries(self):
        assert tokens_of([], _StubEncoder()) == 0

    def test_real_tiktoken_encoder_path(self):
        # The production path: the cl100k_base encoder from corvidae.context.
        from corvidae.context import _encoder

        if _encoder is None:
            pytest.skip("tiktoken unavailable")
        entries = [{"content": "hello world"}]
        assert tokens_of(entries, _encoder) == len(_encoder.encode("hello world"))


class TestFixtureSeed:
    """The shipped seed fixture must conform to the documented format."""

    def test_fixture_loads_and_has_required_keys(self):
        fixture = json.loads(FIXTURE_PATH.read_text())
        assert set(fixture) >= {"description", "conversation", "memories", "queries"}

    def test_memories_and_queries_are_consistent(self):
        fixture = json.loads(FIXTURE_PATH.read_text())
        memory_ids = {m["id"] for m in fixture["memories"]}
        assert 5 <= len(memory_ids) <= 10
        assert 3 <= len(fixture["queries"]) <= 5
        for query in fixture["queries"]:
            assert query["text"]
            assert query["note"]
            # Every labeled-relevant id refers to a real memory.
            assert set(query["relevant"]) <= memory_ids

    def test_memories_have_required_fields(self):
        fixture = json.loads(FIXTURE_PATH.read_text())
        for memory in fixture["memories"]:
            assert memory["id"]
            assert memory["summary"]
            assert memory["channel_id"]
