"""Tests for harness/driver.py's pure helpers.

split_irc_text guards against RFC 1459's 512-byte line limit — probe text
(e.g. build_filler_paragraph's output) that gets sent unsplit is silently
dropped: ngircd kills the connection for "Request too long" instead of
delivering it, which cascades into every probe sent after it failing too.
"""

from harness.driver import split_irc_text


def test_text_under_limit_is_one_chunk():
    text = "short message"
    assert split_irc_text(text, max_bytes=100) == [text]


def test_text_over_limit_splits_into_multiple_chunks():
    text = "a" * 1000
    chunks = split_irc_text(text, max_bytes=100)
    assert len(chunks) > 1
    assert all(len(c.encode("utf-8")) <= 100 for c in chunks)
    assert "".join(chunks) == text


def test_multibyte_character_never_split_across_chunks():
    # "é" is 2 bytes in UTF-8 — a byte-oblivious splitter at an odd boundary
    # would sever it and corrupt the text on decode.
    text = "é" * 300
    chunks = split_irc_text(text, max_bytes=101)
    assert all(len(c.encode("utf-8")) <= 101 for c in chunks)
    assert all(c.encode("utf-8").decode("utf-8") == c for c in chunks)
    assert "".join(chunks) == text
