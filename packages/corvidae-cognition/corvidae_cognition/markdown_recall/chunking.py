"""Page content chunking for vector embedding (vendored).

Vendored from schuyler/otterwiki-semantic-search (MIT, same author) per the
#719 design. Public API: chunk_page(pagepath, content) -> [{id, text, metadata}].
"""

import re

from corvidae.channels.split import split_message

from corvidae_cognition.markdown_recall.frontmatter import parse_frontmatter

# Sizes are UTF-8 bytes (#1186 R4): bytes per token is far steadier across
# scripts than characters or whitespace words, and unspaced text has no words.
# Converted from the old word sizes at ~6 bytes per word.
TARGET_BYTES = 900  # was TARGET_WORDS = 150
OVERLAP_BYTES = 200  # was OVERLAP_WORDS = 35
MIN_CHUNK_BYTES = 1800  # was MIN_CHUNK_WORDS = 300; pages below are one chunk
STUB_BYTES = 300  # was STUB_WORDS = 50
MAX_PREFIX_DEPTH = 3

HEADING_RE = re.compile(r'^(#{1,6})\s+(.+)$')

# One sentence piece: lazily up to a boundary. A boundary is a Latin
# terminal plus whitespace, a full-width terminal plus optional whitespace,
# a newline, or the end of the text. No condition on what follows (#1186 R5).
_SENTENCE_RE = re.compile(r'.*?(?:(?<=[.!?])\s+|(?<=[\u3002\uff01\uff1f])\s*|\n|\Z)')


def _split_sentences(text: str) -> list[str]:
    """Split text into sentence pieces that concatenate to ``text`` exactly."""
    return [p for p in _SENTENCE_RE.findall(text) if p]


def _size(text: str) -> int:
    """Size of ``text`` in UTF-8 bytes; the unit of every chunking constant."""
    return len(text.encode("utf-8"))


def _pack(text: str, units: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Greedily pack unit spans into chunk spans of about TARGET_BYTES.

    A chunk runs from its first unit's start to its last unit's end, so
    whatever separated the units in the source is kept exactly.
    """
    spans: list[tuple[int, int]] = []
    cur_start = cur_end = -1
    for start, end in units:
        # Close the open chunk when adding this unit would pass the target.
        if cur_end >= 0 and _size(text[cur_start:end]) > TARGET_BYTES:
            spans.append((cur_start, cur_end))
            cur_start = -1
        if cur_start < 0:
            cur_start = start
        cur_end = end
    if cur_end >= 0:
        spans.append((cur_start, cur_end))
    return spans


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    """Shrink a span past whitespace at both edges."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _sentence_units(text: str, base: int) -> list[tuple[int, int]]:
    """Offsets (into the section text) of a paragraph's sentence units.

    ``base`` is the paragraph's start. A piece over 2 x TARGET_BYTES is cut
    with split_message, which inserts and loses nothing, so offsets simply
    accumulate over the cuts.
    """
    units: list[tuple[int, int]] = []
    pos = base
    for piece in _split_sentences(text):
        # An over-long piece (a long unpunctuated line) is cut further.
        parts = (
            split_message(piece, TARGET_BYTES)
            if _size(piece) > TARGET_BYTES * 2
            else [piece]
        )
        for part in parts:
            if part.strip():
                units.append((pos, pos + len(part)))
            pos += len(part)
    return units


def _chunk_spans(text: str) -> list[tuple[int, int]]:
    """Split section text into chunk spans (start, end) by paragraph and
    sentence boundaries. No overlap; the spans are slices of ``text``."""
    chunks: list[tuple[int, int]] = []
    pending: list[tuple[int, int]] = []  # paragraph units awaiting packing

    for m in re.finditer(r'\S(?:.|\n(?!\n))*', text):
        # Paragraphs are runs between blank lines, edges trimmed.
        start, end = _trim(text, m.start(), m.end())
        if _size(text[start:end]) > TARGET_BYTES * 2:
            # A huge paragraph: flush the pending ones, then split it into
            # sentence units and pack those on their own.
            chunks.extend(_pack(text, pending))
            pending = []
            sentences = _sentence_units(text[start:end], start)
            chunks.extend(_pack(text, sentences))
        else:
            pending.append((start, end))
    chunks.extend(_pack(text, pending))
    return [_trim(text, s, e) for s, e in chunks]


def _overlap_start(text: str, prev_start: int, start: int, max_bytes: int) -> int:
    """Offset where chunk ``start`` is widened back to, for overlap.

    The smallest offset >= prev_start with text[o:start] at most max_bytes.
    A cut inside a run of non-whitespace moves forward to the first word
    start inside the overlap (spaced text overlaps on a word); unspaced text
    keeps the code-point cut.
    """
    o = start
    size = 0
    # Walk back one code point at a time while the overlap still fits.
    while o > prev_start:
        width = _size(text[o - 1])
        if size + width > max_bytes:
            break
        size += width
        o -= 1
    # A cut mid-word moves to the first whitespace-then-word boundary, if any.
    if 0 < o < start and not text[o - 1].isspace() and not text[o].isspace():
        for j in range(o + 1, start):
            if text[j - 1].isspace() and not text[j].isspace():
                return j
    return o


FENCE_RE = re.compile(r'^\s*(`{3,}|~{3,})')


def _split_into_sections(body):
    """Split body into list of dicts: {level, heading, text}.

    Level 0 = preamble (text before any heading).
    Heading lines inside fenced code blocks are ignored.
    """
    lines = body.split('\n')
    sections = []
    current_level = 0
    current_heading = ""
    current_lines = []

    in_fence = False
    fence_char = None
    fence_len = 0

    for line in lines:
        # Track fenced code block state
        fm = FENCE_RE.match(line)
        if fm:
            fence_str = fm.group(1)
            char = fence_str[0]
            length = len(fence_str)
            if not in_fence:
                in_fence = True
                fence_char = char
                fence_len = length
            elif char == fence_char and length >= fence_len:
                in_fence = False
                fence_char = None
                fence_len = 0
            # Either way, add line to current section content
            current_lines.append(line)
            continue

        if not in_fence:
            m = HEADING_RE.match(line)
            if m:
                # Save current section
                text = "\n".join(current_lines).strip()
                sections.append({
                    "level": current_level,
                    "heading": current_heading,
                    "text": text,
                })
                current_level = len(m.group(1))
                current_heading = m.group(2).strip()
                current_lines = []
                continue

        current_lines.append(line)

    # Final section
    text = "\n".join(current_lines).strip()
    sections.append({
        "level": current_level,
        "heading": current_heading,
        "text": text,
    })

    return sections


def _build_header_stack(sections, page_title):
    """Annotate each section dict with section_path and section fields.

    Mutates sections in place, returns them.
    """
    # Stack of (level, title) tuples — does not include page_title entry
    stack = []

    for sec in sections:
        level = sec["level"]
        heading = sec["heading"]

        if level == 0:
            # Preamble
            sec["section"] = ""
            sec["section_path"] = page_title
        else:
            # Pop stack entries at same or deeper level
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading))

            # Build path: page_title + up to MAX_PREFIX_DEPTH levels from stack
            path_parts = [page_title] + [s[1] for s in stack[:MAX_PREFIX_DEPTH]]
            sec["section_path"] = " > ".join(path_parts)
            sec["section"] = heading

    return sections


def _render_stub(section):
    """A stub section as text for merging: its heading line, then its body.

    A merged stub keeps no metadata of its own, so its heading has to survive
    in the text: "[page > Known bugs]" is the only thing a search for the
    heading words can then match. Without it a short section's title vanished
    from the index entirely, which is worst exactly for the short sections that
    carry the point (Known bugs, Status, Decision).
    """
    heading = section["heading"]
    text = section["text"].strip()
    if not heading:
        return text
    title = f"{'#' * max(section['level'], 1)} {heading}"
    return f"{title}\n\n{text}" if text else title


def _merge_stub_sections(sections):
    """Merge stub sections (text < STUB_BYTES) forward or backward.

    Only non-preamble (level > 0) sections are candidates for merging.
    Returns new list of section dicts with stubs merged.
    Uses a single O(n) forward pass — no list.pop() inside loops.
    """
    if not sections:
        return sections

    # Forward pass: stubs merge into the next substantial section.
    # We defer each stub until we see a substantial section to absorb it.
    # Trailing stubs are merged backward into the preceding substantial section.
    merged = []
    pending_stubs = []  # stubs waiting for a substantial section

    for section in sections:
        is_stub = (section["level"] > 0
                   and _size(section["text"]) < STUB_BYTES)

        if is_stub:
            pending_stubs.append(section)
        else:
            # Substantial (or preamble) section — prepend any pending stubs
            if pending_stubs:
                prefix_text = "\n\n".join(
                    _render_stub(s) for s in pending_stubs
                ).strip()
                if prefix_text:
                    section = dict(section)
                    section["text"] = (prefix_text + "\n\n" + section["text"]
                                       if section["text"] else prefix_text)
                pending_stubs = []
            merged.append(section)

    # Any remaining pending stubs merge backward into the last substantial section
    if pending_stubs:
        stub_text = "\n\n".join(_render_stub(s) for s in pending_stubs).strip()
        if stub_text:
            # Find last substantial non-preamble section in merged
            for idx in range(len(merged) - 1, -1, -1):
                if merged[idx]["level"] > 0 and _size(merged[idx]["text"]) >= STUB_BYTES:
                    merged[idx] = dict(merged[idx])
                    merged[idx]["text"] = (merged[idx]["text"] + "\n\n" + stub_text
                                           if merged[idx]["text"] else stub_text)
                    break
            else:
                # No substantial section found — keep stubs as-is
                merged.extend(pending_stubs)
        else:
            # Empty stubs — discard
            pass

    return merged


def chunk_page(pagepath, content):
    """Split a page into overlapping chunks for embedding.

    Returns list of dicts with keys: id, text, metadata.
    """
    frontmatter, body = parse_frontmatter(content)
    body = body.strip()

    if not body:
        return []

    # Build metadata from frontmatter
    meta = {"page_path": pagepath}
    if frontmatter:
        if "category" in frontmatter:
            meta["category"] = str(frontmatter["category"])
        if "tags" in frontmatter:
            tags = frontmatter["tags"]
            if isinstance(tags, list):
                meta["tags"] = ", ".join(str(t) for t in tags)
            else:
                meta["tags"] = str(tags)
        if "title" in frontmatter:
            meta["title"] = str(frontmatter["title"])

    page_title = frontmatter.get("title", pagepath) if frontmatter else pagepath
    page_bytes = _size(body)

    # Split body into sections, annotate with paths, merge stubs
    sections = _split_into_sections(body)
    sections = _build_header_stack(sections, page_title)
    sections = _merge_stub_sections(sections)

    # Filter out sections with no text
    sections = [s for s in sections if s["text"].strip()]

    if not sections:
        return []

    # Build all chunks across sections
    all_chunks = []  # list of (text_with_prefix, section, section_path)

    for sec in sections:
        section_path = sec["section_path"]
        section = sec["section"]
        prefix = f"[{section_path}] "

        sec_text = sec["text"].strip()

        # For short sections / short total body, keep as single chunk
        if page_bytes < MIN_CHUNK_BYTES or _size(sec_text) < MIN_CHUNK_BYTES:
            all_chunks.append((prefix + sec_text, section, section_path))
        else:
            spans = _chunk_spans(sec_text)
            for i, (start, end) in enumerate(spans):
                # Widen later chunks backwards over the source (within the
                # section only); nothing is inserted.
                if i > 0:
                    start = _overlap_start(
                        sec_text, spans[i - 1][0], start, OVERLAP_BYTES
                    )
                all_chunks.append((prefix + sec_text[start:end], section, section_path))

    total_chunks = len(all_chunks)

    return [
        {
            "id": f"{pagepath}::chunk_{i}",
            "text": text,
            "metadata": {
                **meta,
                "chunk_index": i,
                "section": section,
                "section_path": section_path,
                "page_bytes": page_bytes,
                "total_chunks": total_chunks,
            },
        }
        for i, (text, section, section_path) in enumerate(all_chunks)
    ]
