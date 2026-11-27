r"""A row of non-ASCII text is estimated at what the model sees, not at its escapes (#754).

PRD §3.3.1 estimates with a tokenizer over "the JSON-serialized visible fields"
of a message, and the same item says a row that renders is "estimated as it
renders, not by the record": the estimate charges for what arrives at the model.
`json.dumps` escapes every non-ASCII character to a ``\uXXXX`` sequence by
default, so the estimator was charging roughly six characters' worth of tokens
for each one — a window of CJK or emoji compacted, dropped rows or truncated
its newest row at a fraction of the provider's window.

The counter in these tests is the real one wherever the assertion is about a
ratio: a stubbed ``count_tokens`` cannot see one. Where a test needs the
character fallback it swaps the encoder out explicitly and says so.
"""

from __future__ import annotations

import json

import pytest

import corvidae.context as context
from corvidae.context import count_tokens, message_tokens

CJK = "中文" * 2880  # 5760 characters
EMOJI = "\U0001f600" * 1440  # 1440 characters
ASCII = "a" * 5760

# What the JSON around an empty row costs. A row's estimate is the text plus
# this envelope; tokenization is not perfectly additive across the seam, so the
# lower edge of the band allows a little merge at the boundary.
ENVELOPE = count_tokens(json.dumps({"role": "user", "content": ""}, ensure_ascii=False))
SEAM_SLACK = 8


def _row(text: str) -> dict:
    return {"role": "user", "content": text}


class TestEscapesAreNotCharged:
    """Two-sided bands. An upper bound alone would pass on an estimate of 0."""

    @pytest.mark.parametrize("name,text", [("cjk", CJK), ("emoji", EMOJI)])
    def test_row_estimates_within_its_envelope_of_the_text(self, name, text):
        direct = count_tokens(text)
        est = message_tokens(_row(text))
        assert est >= direct - SEAM_SLACK, (
            f"{name}: estimated {est}, below the {direct} tokens the same text costs on its "
            f"own — the estimate dropped content rather than counting it (#754)"
        )
        assert est <= direct + ENVELOPE, (
            f"{name}: estimated {est} against {direct} for the text plus {ENVELOPE} for the "
            f"JSON around it — the surplus is escaped characters, not content (#754)"
        )

    def test_estimate_is_the_count_of_the_row_with_characters_intact(self):
        """Deliberately coupled to the serialization: this is the semantics.

        The estimate is `count_tokens` of the row serialized with non-ASCII left
        as themselves. If a future change moves to a different serialization,
        this test should fail loudly and be rewritten on purpose, not drift.
        """
        row = _row("naïve café 中文 日本語 🎉")
        assert message_tokens(row) == count_tokens(json.dumps(row, ensure_ascii=False))

    def test_escaped_form_is_explicitly_the_thing_being_refused(self):
        """The relation the fix is about, kept as its own regression guard."""
        row = _row("día 中文 🎉")
        assert message_tokens(row) < count_tokens(json.dumps(row)), (
            "the estimate still counts the escaped serialization (#754)"
        )


class TestFallbackCountsNonAsciiToo:
    """The PRD allows a character-based fallback; it must not be wrong by 3x.

    A flat chars/token is right for neither CJK nor emoji: measured against the
    tokenizer at 14:42:50Z it came out 3.4x low on CJK and 6.9x low on emoji,
    which compacts a non-ASCII window late — the provider's limit, not the
    operator's. The fallback weights by codepoint range now.
    """

    @pytest.mark.parametrize("name,text", [("cjk", CJK), ("emoji", EMOJI)])
    def test_fallback_lands_near_the_tokenizer_within_its_envelope(self, name, text, monkeypatch):
        if context._encoder is None:
            pytest.skip("no tokenizer: `count_tokens` here would be the fallback being tested")
        truth = count_tokens(text)  # tokenizer path, before the fallback is forced
        monkeypatch.setattr(context, "_encoder", None)
        est = message_tokens(_row(text))
        assert est >= truth * 0.7, (
            f"{name}: the fallback estimated {est} against {truth} tokens the tokenizer counts "
            f"— a non-ASCII window would compact late and overflow at the provider (#754)"
        )
        assert est <= truth * 1.5, f"{name}: the fallback estimated {est} against {truth}"

    def test_ascii_fallback_is_unchanged_by_the_weighting(self, monkeypatch):
        """The ASCII heuristic stays the one every other test pins (3.5 chars)."""
        row = _row(ASCII)
        monkeypatch.setattr(context, "_encoder", None)
        assert message_tokens(row) == int(len(json.dumps(row)) / context._FALLBACK_CHARS_PER_TOKEN)


class TestTheWeightingBeatsTheFlatHeuristic:
    """The fallback is still a guess; the claim is only that it is a better one.

    A sweep of 17 scripts at 15:14:09Z put the weighted estimate between 0.36x
    of the tokenizer's count and 1.88x, where the same fallback run flat at 3.5
    characters per token ranged 0.11x to 1.88x. This pins the direction of the
    improvement on a corpus rather than the constants: if a future change breaks
    the weighting, the flat heuristic wins and this fails.

    It deliberately does not compare against the pre-#754 form (flat on escaped
    text, 0.59x to 2.62x): over-count and under-count can be equally far from
    the truth, and two errors cancelling was never the thing to preserve.
    """

    CORPUS = [
        "中文汉字测试窗口令牌估算",  # Han
        "привет мир оценка",  # Cyrillic
        "ሰላም ዓለም ግምት",  # Ge'ez: the worst under-count in the sweep
        "สวัสดีชาวโลก",  # Thai
        "नमस्ते दुनिया अनुमान",  # Devanagari
        "e\u0301k\u0307m\u0307 n\u0303a\u0308o\u030c",  # combining marks
        "\U0001f600\U0001f680",  # emoji
    ]

    def test_weighted_fallback_is_closer_than_three_and_a_half_chars(self, monkeypatch):
        if context._encoder is None:
            pytest.skip("no tokenizer to measure the two guesses against")
        row = _row(" ".join(phrase * 40 for phrase in self.CORPUS))
        truth = count_tokens(json.dumps(row, ensure_ascii=False))
        monkeypatch.setattr(context, "_encoder", None)
        weighted = message_tokens(row)
        flat = int(len(json.dumps(row, ensure_ascii=False)) / context._FALLBACK_CHARS_PER_TOKEN)
        assert abs(weighted - truth) < abs(flat - truth), (
            f"weighted fallback {weighted} is no closer to {truth} than the flat "
            f"heuristic's {flat} — the weighting is doing nothing (#754)"
        )


class TestTheEstimateStaysItself:
    """Regression guards, not evidence of the fix.

    Each of these passes with `ensure_ascii=True` as well; they are here so a
    later change cannot quietly move ASCII cost, reload behaviour or tag
    handling while "fixing" the estimate. The cases that exercise #754 are in
    the two classes above.
    """

    def test_ascii_row_is_unchanged(self):
        """ASCII has no escaped form, so its count cannot move with this fix."""
        assert message_tokens(_row(ASCII)) == count_tokens(json.dumps(_row(ASCII)))

    def test_estimate_survives_a_json_round_trip(self):
        """The estimate is a pure function of the stored row, so a reload agrees.

        Compaction decisions compare a row's cost across restarts; if the
        estimate depended on how the row was decoded, the cut would move.
        """
        row = _row("café 中文 🎉")
        assert message_tokens(row) == message_tokens(json.loads(json.dumps(row)))

    def test_a_row_that_is_not_a_mapping_is_counted_by_its_repr(self):
        """A row with no visible dict is counted by `repr`, not skipped.

        This pins the non-mapping branch only. The wider "estimating never
        raises" of §3.1.18.1 does not hold yet for rows that fail before any
        count is attempted — a non-scalar key raises in `_visible`, a value
        whose `str()` raises escapes `json.dumps`' `default=str`. Measured
        identical on `8c8b2794` and filed as #1333; this test must not be read
        as certifying more than the branch it exercises.
        """
        for oddity in (42, ["not", "a", "row"], None, "bare string"):
            assert message_tokens(oddity) == count_tokens(repr(oddity))

    def test_an_unpaired_surrogate_is_estimated_without_raising(self, monkeypatch):
        """A lone surrogate decodes from `json.loads` and must still be countable.

        With escapes it never reached the tokenizer; unescaped it does, and
        tiktoken refuses it — the rescue path in `count_tokens` is what keeps
        §3.1.18.1 true, so this pins the rescue rather than assuming it.
        """

        # No skip guard: this case installs its own encoder, so it exercises the
        # rescue on a tokenizer-less machine too. `count_tokens` only bypasses
        # `encode` when `_encoder` is None, and here it is not.
        class RefusingEncoder:
            """Any encoder failure routes to the fallback; tiktoken refuses a lone surrogate."""

            def encode(self, text, **kwargs):
                raise UnicodeEncodeError("utf-8", text, 0, len(text), "surrogates not allowed")

        monkeypatch.setattr(context, "_encoder", RefusingEncoder())
        lone = json.loads('"\\ud800"')
        est = message_tokens(_row(f"before{lone}after"))
        assert isinstance(est, int) and est > 0, f"surrogate row estimated as {est!r}"

    def test_tagged_and_clean_rows_still_agree(self):
        """`_`-prefixed tags stay out of the count, as before the fix."""
        text = "día 中文"
        assert message_tokens({**_row(text), "_message_type": "message"}) == message_tokens(
            _row(text)
        )
