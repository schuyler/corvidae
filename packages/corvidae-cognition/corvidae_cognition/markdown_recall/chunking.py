"""Page content chunking for vector embedding (vendored).

Vendored from schuyler/otterwiki-semantic-search (MIT, same author) per the
#719 design. Public API: chunk_page(pagepath, content) -> [{id, text, metadata}].
"""

import re

from corvidae_cognition.markdown_recall.frontmatter import parse_frontmatter

TARGET_WORDS = 150
OVERLAP_WORDS = 35
MIN_CHUNK_WORDS = 300  # Pages below this are a single chunk
STUB_WORDS = 50
MAX_PREFIX_DEPTH = 3

HEADING_RE = re.compile(r'^(#{1,6})\s+(.+)$')


def _split_sentences(text):
    """Split text on sentence boundaries."""
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])|\n', text)
    return [p for p in parts if p.strip()]


def _word_count(text):
    return len(text.split())


def _chunk_text(text: str) -> list:
    """Split text into chunks by paragraph/sentence boundaries. No overlap."""
    paragraphs = re.split(r'\n\n+', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks = []
    current_text_parts = []

    for para in paragraphs:
        para_wc = _word_count(para)

        # If a single paragraph is too large, split on sentences
        if para_wc > TARGET_WORDS * 2:
            # Flush current accumulation first
            if current_text_parts:
                chunks.append("\n\n".join(current_text_parts))
                current_text_parts = []

            sentences = _split_sentences(para)
            sent_acc = []
            sent_wc = 0
            for sent in sentences:
                sw = _word_count(sent)
                if sent_wc + sw > TARGET_WORDS and sent_acc:
                    chunks.append(" ".join(sent_acc))
                    sent_acc = []
                    sent_wc = 0
                sent_acc.append(sent)
                sent_wc += sw
            if sent_acc:
                chunks.append(" ".join(sent_acc))
            continue

        word_total = sum(_word_count(w) for w in current_text_parts) + para_wc

        if word_total > TARGET_WORDS and current_text_parts:
            chunks.append("\n\n".join(current_text_parts))
            current_text_parts = []

        current_text_parts.append(para)

    if current_text_parts:
        chunks.append("\n\n".join(current_text_parts))

    return chunks


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
    """Merge stub sections (text < STUB_WORDS) forward or backward.

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
                   and _word_count(section["text"]) < STUB_WORDS)

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
                if merged[idx]["level"] > 0 and _word_count(merged[idx]["text"]) >= STUB_WORDS:
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
    page_word_count = _word_count(body)

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
        if page_word_count < MIN_CHUNK_WORDS or _word_count(sec_text) < MIN_CHUNK_WORDS:
            all_chunks.append((prefix + sec_text, section, section_path))
        else:
            raw_chunks = _chunk_text(sec_text)
            if not raw_chunks:
                continue

            # Apply overlap within section only
            if len(raw_chunks) > 1:
                overlapped = [raw_chunks[0]]
                for i in range(1, len(raw_chunks)):
                    prev_words = raw_chunks[i - 1].split()
                    overlap_words = prev_words[-OVERLAP_WORDS:]
                    overlapped.append(" ".join(overlap_words) + "\n\n" + raw_chunks[i])
                raw_chunks = overlapped

            for chunk_text in raw_chunks:
                all_chunks.append((prefix + chunk_text, section, section_path))

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
                "page_word_count": page_word_count,
                "total_chunks": total_chunks,
            },
        }
        for i, (text, section, section_path) in enumerate(all_chunks)
    ]
