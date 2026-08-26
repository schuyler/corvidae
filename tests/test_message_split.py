"""Tests for corvidae.channels.split.split_message.

split_message fits a long reply under a transport's per-message character
limit by splitting at the highest-priority boundary that fits: paragraph,
then sentence, then word. These tests pin the algorithm itself, independent
of any one transport — split_message moves out of corvidae/channels/irc.py
into corvidae/channels/split.py, and IRC keeps working via an import shim
left behind in irc.py. Transport-specific integration (chunk delivery,
whitespace-only-chunk filtering) stays in tests/test_irc_plugin.py.
"""

from corvidae.channels.split import split_message


# ---------------------------------------------------------------------------
# Basic splitting behavior — one assertion per tier of the ladder, plus the
# invariants (byte-length limit, UTF-8 counting, full content preserved)
# that must hold regardless of which tier produced a given chunk.
# ---------------------------------------------------------------------------


class TestBasicSplitting:
    def test_short_message_returned_unchanged(self):
        """Text already under max_len is returned as a single-element list."""
        result = split_message("short message", max_len=400)
        assert result == ["short message"]

    def test_splits_on_paragraph_boundaries(self):
        """Splits on \\n\\n when the whole text doesn't fit but paragraphs do."""
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        result = split_message(text, max_len=30)
        assert len(result) > 1
        assert all(len(part.encode("utf-8")) <= 30 for part in result)
        assert "".join(result) == text

    def test_splits_on_sentence_boundaries(self):
        """Falls to .!? + whitespace when no paragraph boundary fits."""
        text = "This is sentence one. This is sentence two! This is sentence three?"
        result = split_message(text, max_len=25)
        assert len(result) > 1
        assert all(len(part.encode("utf-8")) <= 25 for part in result)
        assert "".join(result) == text

    def test_splits_on_word_boundaries(self):
        """Falls to word (space) boundaries when no sentence boundary fits."""
        text = "This is a verylongwordthatwontfit and another word"
        result = split_message(text, max_len=20)
        assert len(result) > 1
        assert all(len(part.encode("utf-8")) <= 20 for part in result)
        assert "".join(result) == text

    def test_oversized_word_is_split_across_chunks(self):
        """A single word longer than max_len is hard-split, never dropped."""
        text = "a" * 500
        result = split_message(text, max_len=400)
        assert len(result) > 1
        assert all(len(part.encode("utf-8")) <= 400 for part in result)
        assert "".join(result) == text
        # The first chunk should be close to max_len, not truncated short.
        assert len(result[0].encode("utf-8")) >= 390

    def test_reassembly_preserves_original_content(self):
        """Joining every chunk reproduces the input exactly — no truncation."""
        text = "Para one. Para two!\n\nNew paragraph. Another sentence.\n\nFinal para."
        result = split_message(text, max_len=30)
        assert "".join(result) == text

    def test_uses_utf8_byte_length_not_character_count(self):
        """Multi-byte characters are counted by UTF-8 byte length."""
        text = "Hello 世界 " * 100  # each Chinese char is 3 bytes
        result = split_message(text, max_len=400)
        assert all(len(part.encode("utf-8")) <= 400 for part in result)
        assert "".join(result) == text


# ---------------------------------------------------------------------------
# Boundary-preference ladder — a chunk boundary falls at the highest tier
# that fits: paragraph before sentence, sentence before word. Each test
# below constructs input where a *lower* tier would technically produce
# valid-sized chunks too, and asserts the splitter still prefers the
# higher tier rather than fragmenting further than necessary.
# ---------------------------------------------------------------------------


class TestBoundaryPreferenceLadder:
    def test_prefers_paragraph_boundary_when_paragraphs_individually_fit(self):
        """Whole paragraphs that fit max_len are kept intact, not further
        split at their internal sentence boundaries."""
        para = "Sentence one. Sentence two."
        text = "\n\n".join([para] * 4)
        result = split_message(text, max_len=40)
        assert len(result) > 1
        assert "".join(result) == text
        # Every chunk is a whole paragraph (optionally prefixed by the \n\n
        # separator) — never a sentence-level fragment of one.
        for chunk in result:
            assert chunk.lstrip("\n") == para

    def test_falls_to_sentence_boundary_when_paragraph_exceeds_limit(self):
        """A paragraph too large for max_len splits at sentence ends, not
        mid-sentence, when the sentences themselves fit."""
        text = "First sentence here. Second sentence here. Third sentence here."
        result = split_message(text, max_len=25)
        assert len(result) > 1
        assert "".join(result) == text
        # Every non-final chunk ends at a sentence terminator.
        for chunk in result[:-1]:
            assert chunk.rstrip()[-1] in ".!?"

    def test_falls_to_word_boundary_when_sentence_exceeds_limit(self):
        """A sentence too large for max_len, with no punctuation boundary,
        splits on spaces rather than mid-word."""
        text = "one two three four five six seven eight nine ten"
        result = split_message(text, max_len=12)
        assert len(result) > 1
        assert "".join(result) == text
        # No chunk boundary falls inside a word: every chunk, stripped of a
        # single leading separator space, is composed of whole words.
        for chunk in result:
            words = chunk.strip().split(" ")
            for word in words:
                assert word in text.split(" ")


# ---------------------------------------------------------------------------
# A long reply over the per-message limit is delivered in full, never
# truncated. A 5000-character reply under a 2000-character limit is the
# canonical case exercised below.
# ---------------------------------------------------------------------------


class TestLongReplyNeverTruncated:
    def test_5000_char_reply_under_2000_char_limit_delivered_in_full(self):
        """A ~5000-character multi-paragraph reply, split at a 2000-char
        limit, reassembles to the original with every chunk within bounds."""
        paragraph_a = (
            "This is the first sentence of a long reply. Here is a second "
            "sentence with more detail. And a third one for good measure, "
            "questioning things? Yes indeed."
        )
        paragraph_b = (
            "Now we pivot to another topic entirely. It has its own "
            "sentences too, each reasonably sized. The reply keeps going "
            "for quite a while, covering a lot of ground."
        )
        text = "\n\n".join([paragraph_a, paragraph_b] * 16)
        assert len(text) >= 5000

        result = split_message(text, max_len=2000)
        assert len(result) > 1
        assert all(len(part.encode("utf-8")) <= 2000 for part in result)
        assert "".join(result) == text

    def test_long_reply_with_embedded_oversized_token_delivered_in_full(self):
        """A reply containing an unsplittable long token (e.g. a hash or
        URL) still delivers every byte — the token is hard-split, the
        surrounding words are not."""
        text = (
            "short prefix words here then "
            + ("a" * 3000)
            + " and more words after that"
        )
        result = split_message(text, max_len=2000)
        assert len(result) > 1
        assert all(len(part.encode("utf-8")) <= 2000 for part in result)
        assert "".join(result) == text


# ---------------------------------------------------------------------------
# Known defects in the current algorithm, now in scope because the Signal
# transport's outbound path depends on split_message, which must deliver a
# long reply in full at whatever limit the transport configures. Both were
# found validating the tests above; the fixes land in the green phase, not
# here.
# ---------------------------------------------------------------------------


class TestSplitDefects:
    def test_oversized_token_mid_paragraph_terminates_without_unbounded_recursion(self):
        """A paragraph with sentence punctuation before and after one
        oversized, unsplittable token must terminate and return chunks
        within the caller's limit — not recurse without the input shrinking.

        The sentence-tier recursive call can be fed a "sentence" segment
        that still contains its own [.!?]-delimited pieces, so each
        recursive call re-splits a same-shaped remainder instead of making
        progress toward a base case, exhausting the call stack.
        """
        text = (
            "Finally "
            + ("x" * 2500)
            + " an oversized token appears here, embedded in an otherwise "
            "normal paragraph with sentences around it. Done."
        )
        result = split_message(text, max_len=2000)
        # Never truncated: every byte of input reappears in the output, so a
        # fix that silently drops the oversized token also fails this test.
        assert "".join(result) == text
        # Every chunk fits the caller's limit, including the hard-split token.
        assert all(len(part.encode("utf-8")) <= 2000 for part in result)

    def test_recursive_paragraph_split_never_exceeds_caller_limit(self):
        """A short paragraph followed by one too large to fit, once
        recursively split, must have every resulting chunk — including the
        re-attached '\\n\\n' separator on the recursive split's first
        sub-chunk — still within max_len.

        The separator is currently prepended to the recursive split's first
        sub-chunk after that sub-chunk was already sized to exactly
        max_len, pushing it over the caller's limit by the separator's
        length.
        """
        text = "short line\n\n" + ("x" * 3000)
        result = split_message(text, max_len=2000)
        assert "".join(result) == text
        assert all(len(part.encode("utf-8")) <= 2000 for part in result)
