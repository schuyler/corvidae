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


def split_message(text: str, max_len: int = 400) -> list[str]:
    """Split text into chunks that fit within max_len UTF-8 bytes.

    Three-tier splitting: paragraphs -> sentences -> words.
    Oversized words are split into multiple max_len chunks.
    Preserves all whitespace and separators when reassembling chunks.
    """
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
                    sub_chunks = split_message(para, max_len - _PARAGRAPH_SEP_LEN)
                else:
                    sub_chunks = split_message(para, max_len)
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
                chunks.extend(split_message(sentence, max_len))
                current = ""
        if current:
            chunks.append(current)
        return chunks

    # Tier 3: Split on word boundaries (spaces)
    words = text.split(' ')
    current = ""
    for i, word in enumerate(words):
        # Track if this word should have a leading space
        has_leading_space = i > 0

        # Check if this word itself exceeds max_len
        if len(word.encode('utf-8')) > max_len:
            # Output current chunk first
            if current:
                # If this is not the first word, add trailing space to preserve it
                if has_leading_space and not current.endswith(' '):
                    current = current + ' '
                chunks.append(current)
                current = ""

            # Split the oversized word into max_len chunks
            # Note: we don't include the leading space in the oversized word chunks
            # to avoid exceeding max_len. The space was added to the previous chunk.
            word_bytes = word.encode('utf-8')
            for start in range(0, len(word_bytes), max_len):
                chunk_bytes = word_bytes[start:start + max_len]
                chunk = chunk_bytes.decode('utf-8', errors='ignore')
                chunks.append(chunk)
            # Continue to next word (don't add space handling below)
            continue

        # Add space for non-first words
        if has_leading_space:
            word = ' ' + word
        candidate = current + word
        if len(candidate.encode('utf-8')) <= max_len:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = word
    if current:
        chunks.append(current)

    return chunks if chunks else [text]
