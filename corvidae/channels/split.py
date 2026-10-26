"""Message splitting for transports with a per-message character limit.

split_message fits a long reply under a transport's outbound size limit by
splitting at the highest-priority boundary that fits: paragraph, then
sentence, then word. Lives here, shared across transports, so no transport
has to import another transport's plugin module to use it.
"""

import re

# Bytes reserved for the '\n\n' paragraph separator re-attached ahead of a
# recursively split paragraph's first sub-chunk (see Tier 1 below).
_PARAGRAPH_SEP = "\n\n"
_PARAGRAPH_SEP_LEN = len(_PARAGRAPH_SEP.encode("utf-8"))


# Smallest usable limit. Tier 1 recurses at max_len - 2 (to leave room for the
# re-attached '\n\n'), and that shrunken budget must still hold one 4-byte
# UTF-8 codepoint so an oversized piece can always be cut on a codepoint
# boundary.
_MIN_MAX_LEN = 6


def split_message(text: str, max_len: int = 400) -> list[str]:
    """Split text into chunks that fit within max_len UTF-8 bytes.

    Three-tier splitting: paragraphs -> sentences -> words.
    Oversized words are split into multiple max_len chunks, cut on codepoint
    boundaries. Invariants (for max_len >= 6): every chunk is at most max_len
    UTF-8 bytes, and "".join(chunks) == text.

    Raises ValueError if max_len < 6, regardless of the text, so a
    misconfigured limit is not input-dependent.
    """
    # Validate once at the public entry point; the recursive helper below
    # legitimately shrinks its budget by 2 and must not re-trip this check.
    if max_len < _MIN_MAX_LEN:
        raise ValueError(
            f"max_len must be at least {_MIN_MAX_LEN} bytes to hold one "
            f"UTF-8 codepoint, got {max_len}"
        )
    return _split(text, max_len)


def _cut_codepoints(piece: str, max_len: int) -> list[str]:
    """Cut piece into parts of at most max_len UTF-8 bytes, only ever on
    codepoint boundaries, so no part is invalid UTF-8 and no data is lost.
    Requires max_len >= 4 (the widest codepoint)."""
    parts: list[str] = []
    current: list[str] = []
    size = 0
    for ch in piece:
        # Width of this codepoint in UTF-8.
        width = len(ch.encode('utf-8'))
        if size + width > max_len:
            parts.append(''.join(current))
            current = []
            size = 0
        current.append(ch)
        size += width
    if current:
        parts.append(''.join(current))
    return parts


def _split(text: str, max_len: int) -> list[str]:
    """Recursive worker for split_message; assumes max_len >= 6."""
    if len(text.encode('utf-8')) <= max_len:
        return [text]

    chunks = []

    # Tier 1: Try splitting on paragraph boundaries (\n\n)
    paragraphs = text.split('\n\n')
    if len(paragraphs) > 1:
        current = ""
        for i, para in enumerate(paragraphs):
            # For the first paragraph, don't add separator
            # For subsequent paragraphs, add separator before the paragraph
            if i == 0:
                candidate = current + para
            else:
                candidate = current + '\n\n' + para
            if len(candidate.encode('utf-8')) <= max_len:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # Recursively split the paragraph that doesn't fit. For a
                # non-first paragraph, the '\n\n' separator gets re-attached
                # to the first sub-chunk below, so that sub-chunk is split
                # against a budget shrunk by the separator's byte length --
                # otherwise the re-attach can push it past max_len.
                if i > 0:
                    sub_chunks = _split(para, max_len - _PARAGRAPH_SEP_LEN)
                else:
                    sub_chunks = _split(para, max_len)
                if i > 0 and sub_chunks:
                    sub_chunks[0] = '\n\n' + sub_chunks[0]
                chunks.extend(sub_chunks)
                current = ""
        if current:
            chunks.append(current)
        return chunks

    # Tier 2: Try splitting on sentence boundaries (.!? + whitespace)
    # Use lookahead to preserve the whitespace after sentence endings
    sentences = re.split(r'(?<=[.!?])(\s+)', text)
    # A trailing whitespace group after the sole sentence in the text makes
    # re.split return more than one element even though there is only one
    # real sentence to split on. Counting only the non-whitespace segments
    # detects that case: recursing on that "sentence" would hand the next
    # call the exact same text back (whitespace re-split the same way),
    # making no progress toward a base case and recursing without bound.
    real_sentences = [s for s in sentences if s.strip()]
    if len(real_sentences) > 1:
        current = ""
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            # Include the whitespace that follows the sentence
            if i + 1 < len(sentences) and re.match(r'^\s+$', sentences[i + 1]):
                sentence += sentences[i + 1]
                i += 2
            else:
                i += 1

            candidate = current + sentence
            if len(candidate.encode('utf-8')) <= max_len:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                chunks.extend(_split(sentence, max_len))
                current = ""
        if current:
            chunks.append(current)
        return chunks

    # Tier 3: Split on word boundaries (spaces). Each piece carries its own
    # leading space (piece_0 = word_0, piece_i = ' ' + word_i), so joining the
    # pieces reproduces the text exactly and the space is counted against the
    # byte budget.
    words = text.split(' ')
    pieces = [words[0]] + [' ' + w for w in words[1:]]
    current = ""
    for piece in pieces:
        # A piece that cannot fit even alone is flushed against and then cut
        # on codepoint boundaries into parts that each fit.
        if len(piece.encode('utf-8')) > max_len:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_cut_codepoints(piece, max_len))
            continue
        # Greedily pack the piece onto the current chunk if it fits;
        # otherwise flush the chunk and start a new one. A flushed chunk is
        # never touched again.
        candidate = current + piece
        if len(candidate.encode('utf-8')) <= max_len:
            current = candidate
        else:
            chunks.append(current)
            current = piece
    if current:
        chunks.append(current)

    return chunks if chunks else [text]
