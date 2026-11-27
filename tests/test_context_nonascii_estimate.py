r"""A row of non-ASCII text is estimated at what the model sees, not at its escapes (#754).

PRD §3.3.1 estimates with a tokenizer over "the JSON-serialized visible fields"
of a message, and the same item says a row that renders is "estimated as it
renders, not by the record": the estimate charges for what arrives at the model.
`json.dumps` escapes every non-ASCII character to a ``\uXXXX`` sequence by
default, so the estimator was charging roughly 6 characters' worth of tokens for
each one — a window full of CJK or emoji compacted, dropped rows or truncated
its newest row at a fraction of the provider's window.

These tests run the real counter, not a stub: the defect is a ratio, and a
mocked ``count_tokens`` cannot see one.
"""

from __future__ import annotations

import json

from corvidae.context import count_tokens, message_tokens

# A row's JSON envelope costs a little over the bare text: {"role": "user",
# "content": "..."}. Any envelope is fine; several times the text is not.
ENVELOPE_SLACK = 30


def _row(text: str) -> dict:
    return {"role": "user", "content": text}


class TestNonAsciiEstimateIsNotInflatedByEscapes:
    def test_cjk_row_estimates_near_its_own_token_cost(self):
        text = "中文" * 2880  # 5760 characters
        direct = count_tokens(text)
        est = message_tokens(_row(text))
        assert est <= direct + ENVELOPE_SLACK, (
            f"a CJK row estimated at {est} tokens against {direct} for the same "
            f"text: the estimate is charging for \\uXXXX escapes, not characters (#754)"
        )

    def test_emoji_row_estimates_near_its_own_token_cost(self):
        text = "\U0001f600" * 1440  # 1440 characters
        direct = count_tokens(text)
        est = message_tokens(_row(text))
        assert est <= direct + ENVELOPE_SLACK, (
            f"an emoji row estimated at {est} tokens against {direct} for the same "
            f"text: the estimate is charging for \\uXXXX escapes, not characters (#754)"
        )

    def test_estimate_is_below_the_escaped_serialization_it_replaced(self):
        """The guard on the mechanism itself, for any non-ASCII row.

        ``ensure_ascii=True`` is the defect, so the estimate must be cheaper
        than the escaped form it used to count. Written as a relation between
        two serializations rather than a hard-coded number, so it survives a
        tokenizer change.
        """
        text = "naïve café 中文 日本語 🎉"
        row = _row(text)
        escaped = json.dumps(row)  # the form the estimator used
        assert message_tokens(row) < count_tokens(escaped), (
            "the estimate still counts the escaped serialization (#754)"
        )


class TestTheEstimateStaysItself:
    def test_ascii_row_is_unchanged(self):
        """ASCII has no escaped form, so its count cannot move with this fix."""
        row = _row("a" * 5760)
        assert message_tokens(row) == count_tokens(json.dumps(row))

    def test_estimate_survives_a_json_round_trip(self):
        """The estimate is a pure function of the stored row, so a reload agrees.

        Compaction decisions compare a row's cost across restarts; if the
        estimate depended on how the row was decoded, the cut would move.
        """
        row = _row("café 中文 🎉")
        reloaded = json.loads(json.dumps(row))
        assert message_tokens(row) == message_tokens(reloaded)

    def test_a_row_that_is_not_a_mapping_is_still_counted_by_its_repr(self):
        """§3.1.18.1: estimating never raises, whatever the row holds."""
        for oddity in (42, ["not", "a", "row"], None, "bare string"):
            assert message_tokens(oddity) == count_tokens(repr(oddity))

    def test_tagged_and_clean_rows_still_agree(self):
        """`_`-prefixed tags stay out of the count, as before the fix."""
        text = "día 中文"
        assert message_tokens({**_row(text), "_message_type": "message"}) == message_tokens(
            _row(text)
        )
