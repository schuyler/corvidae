"""File read/write tools."""

import asyncio
import os
import pathlib
import re

from corvidae.context import message_tokens
from corvidae.tool import MAX_TOOL_RESULT_CHARS

# Characters per token assumed when sizing a page from a token budget: the
# measured p10 of ordinary text, so a page is unlikely to exceed its share.
CHARS_PER_TOKEN_ESTIMATE = 3.0

# Smallest page: leaves room for the range header and some content.
MIN_PAGE_CHARS = 2000

# Content a fitted page keeps beyond its header: a page that named where to
# continue without carrying anything to read would page nowhere, so the header
# plus this many characters is the smallest page counting will cut to (#752).
MIN_PAGE_CONTENT_CHARS = 4

# Characters read per call when skipping to an offset or counting the rest.
_CHUNK = 64 * 1024

# Reads of one page allowed when fitting it to a token share: the character
# estimate is the first try, and each retry scales it by the measured overrun.
_FIT_ATTEMPTS = 3


def page_chars(max_result_chars: int, budget: int | None, fraction: float) -> int:
    """Characters in one read_file page for a channel budget in tokens.

    No budget gives max_result_chars; otherwise the budget's share, at least
    MIN_PAGE_CHARS, and never more than max_result_chars.
    """
    # Without a channel budget only the result cap bounds a page.
    if budget is None:
        return max_result_chars
    share = int(fraction * budget * CHARS_PER_TOKEN_ESTIMATE)
    return min(max_result_chars, max(MIN_PAGE_CHARS, share))


def page_message_tokens(page: str) -> int:
    """Tokens one page costs the window, counted as the tool result it becomes."""
    return message_tokens({"role": "tool", "content": page})


# The header a ranged read writes, which is what makes a page a page.
PAGE_HEADER_RE = re.compile(r"\[(?:characters|lines) \d+-\d+ of \d+ in ")


def page_body(page: str) -> str:
    """The page's content: a ranged page's text after its header, else all of it."""
    head, sep, rest = page.partition("\n")
    return rest if sep and PAGE_HEADER_RE.match(head) else page


def page_header_chars(page: str) -> int:
    """Characters the page's header costs, or 0 for a page without one."""
    return len(page) - len(page_body(page))


def page_header_tokens(page: str) -> int:
    """Tokens the page's header costs, or 0 for a page without one."""
    header = page_header_chars(page)
    return page_message_tokens(page[:header]) if header else 0


def fit_page_chars(page: str, max_chars: int, max_tokens: int) -> int | None:
    """The character page to retry with when ``page`` is over ``max_tokens``.

    None when the page already fits, or when it cannot shrink further: the
    header plus ``MIN_PAGE_CONTENT_CHARS`` is the smallest page that still
    carries something to read, and a share below that is one no page meets.
    """
    used = page_message_tokens(page)
    # Fitting costs nothing; over the share the page is re-read smaller.
    if used <= max_tokens:
        return None
    # The header's cost is fixed whatever the content, so only the content is
    # scaled by the overrun, and a margin covers what the ratio mispredicts.
    header = page_header_chars(page)
    header_tokens = page_header_tokens(page)
    # The page's own length, not the budget it was read with: a short file
    # comes back whole and bare, and scaling by a budget it never used would
    # only ask for the page it already has.
    content_chars = len(page) - header
    content_tokens = used - header_tokens
    # The floor keeps content, so the header is never a page on its own.
    smallest = header + MIN_PAGE_CONTENT_CHARS
    if max_chars <= smallest or content_tokens <= 0:
        return None
    share_content = max_tokens - header_tokens
    if share_content <= 0:
        # The header alone is over the share: only the floor is left.
        return smallest if smallest < max_chars else None
    scaled = int(content_chars * share_content / content_tokens * 0.95)
    return max(smallest, min(max_chars - 1, header + scaled))


async def read_file(
    path: str,
    max_size: int = 1024 * 1024,
    offset: int | None = None,
    length: int | None = None,
    max_chars: int = MAX_TOOL_RESULT_CHARS,
    start_line: int | None = None,
    end_line: int | None = None,
    max_tokens: int | None = None,
) -> str:
    """Read a file, a character range of it (offset, length) or a line range
    of it (start_line, end_line).

    ``max_chars`` sizes the page from a character estimate; ``max_tokens``,
    when given, is the share of the window the page may take, and the page is
    re-read smaller until its counted cost fits it (#752).
    """
    page = await asyncio.to_thread(
        _read_file_sync, path, max_size, offset, length, max_chars, start_line, end_line
    )
    for _ in range(_FIT_ATTEMPTS):
        if max_tokens is None:
            break
        smaller = fit_page_chars(page, max_chars, max_tokens)
        if smaller is None:
            break
        # A page that came back without content pages nowhere, and one that
        # came back unchanged was not cut by its size at all: keep the page
        # that read, so a read always makes progress.
        fitted = await asyncio.to_thread(
            _read_file_sync, path, max_size, offset, length, smaller, start_line, end_line
        )
        if not page_body(fitted) or fitted == page:
            break
        max_chars, page = smaller, fitted
    return page


def _read_file_sync(
    path: str,
    max_size: int = 1024 * 1024,
    offset: int | None = None,
    length: int | None = None,
    max_chars: int = MAX_TOOL_RESULT_CHARS,
    start_line: int | None = None,
    end_line: int | None = None,
) -> str:
    # Any range parameter makes the read ranged; reject bad values first.
    lined = start_line is not None or end_line is not None
    charred = offset is not None or length is not None
    ranged = lined or charred
    # The two kinds of range name positions differently, so one call takes one.
    if lined and charred:
        return (
            f"Error: start_line/end_line and offset/length were both given for {path}; "
            "use a line range (start_line, end_line) or a character range "
            "(offset, length), not both"
        )
    if start_line is not None and start_line < 1:
        return f"Error: start_line is {start_line} for {path}; use 1 or greater"
    if end_line is not None and end_line < 1:
        return f"Error: end_line is {end_line} for {path}; use 1 or greater"
    if start_line is not None and end_line is not None and end_line < start_line:
        return (
            f"Error: end_line {end_line} is before start_line {start_line} for {path}; "
            f"use an end_line of {start_line} or greater"
        )
    if offset is not None and offset < 0:
        return f"Error: offset is {offset} for {path}; use 0 or greater"
    if length is not None and length < 1:
        return (
            f"Error: length is {length} for {path}; "
            "use a positive integer, or omit it to read to the end"
        )

    # The path must name an existing regular file.
    p = pathlib.Path(path)
    if not p.exists():
        return f"Error: file not found: {path}"
    if os.path.isdir(path):
        return f"Error: not a file (is a directory): {path}"
    if lined:
        return _read_lines(p, path, start_line or 1, end_line, max_chars)
    if ranged:
        return _read_range(p, path, offset or 0, length, max_chars)

    # An unranged read loads the whole file, so it is bounded by max_size.
    try:
        size = p.stat().st_size
    except OSError as exc:
        return f"Error: could not stat file: {exc}"
    if size > max_size:
        max_mb = max_size / (1024 * 1024)
        return (
            f"Error: file too large (>{max_mb:.0f}MB): {path}; "
            "pass offset and length to read it in ranges"
        )
    try:
        text = p.read_text(errors="replace")
    except OSError as exc:
        return f"Error: could not read file: {exc}"
    # A file longer than a page returns its first page, header included.
    if len(text) > max_chars:
        return _read_range(p, path, 0, None, max_chars)
    return text


def _read_range(
    p: pathlib.Path, path: str, offset: int, length: int | None, max_chars: int
) -> str:
    """Return a header naming the range and total, then that range's characters.

    The header plus content never exceeds max_chars, so the result cap never
    cuts a range the header has already reported.
    """
    # One pass: skip to the offset, keep at most max_chars, count the rest.
    try:
        with p.open(errors="replace") as f:
            skipped = 0
            while skipped < offset:
                chunk = f.read(min(_CHUNK, offset - skipped))
                if not chunk:
                    break
                skipped += len(chunk)
            kept = f.read(max_chars if length is None else min(length, max_chars))
            rest = 0
            while chunk := f.read(_CHUNK):
                rest += len(chunk)
    except OSError as exc:
        return f"Error: could not read file: {exc}"

    # Reject a range that starts at or past the end.
    total = skipped + len(kept) + rest
    if total == 0:
        return f"Error: {path} is empty; there is nothing to read"
    if offset >= total:
        return (
            f"Error: offset {offset} is at or past the end of {path}, "
            f"which has {total} characters; use an offset below {total}"
        )

    # Size the content from the longest header this range could produce,
    # leaving room for the newline after it.
    header_bound = len(f"[characters {offset}-{total} of {total} in {path}; next offset {total}]")
    content = kept[: max(0, max_chars - header_bound - 1)]
    end = offset + len(content)
    tail = "end of file" if end == total else f"next offset {end}"
    return f"[characters {offset}-{end} of {total} in {path}; {tail}]\n{content}"


def _read_lines(
    p: pathlib.Path, path: str, start: int, end: int | None, max_chars: int
) -> str:
    """Return a header naming the lines returned, then those lines.

    One streaming pass in ``_CHUNK``-sized pieces, so a file of any size, even
    one that is a single enormous line, is never held whole: it skips to
    ``start``, keeps at most ``max_chars`` of the selected lines, and counts
    the rest. The header plus content never exceeds ``max_chars``. Characters
    are counted the way ``_read_range`` counts them (same text-mode open), so
    the offset a cut line reports can be passed to a character-range read.
    """
    pos = 0  # characters consumed before the current piece
    line_no = 1  # the line the next character belongs to
    ended_with_newline = False
    region_start: int | None = None  # file offset of the first selected character
    region_len = 0  # characters in the selected lines, kept or not
    kept: list[str] = []
    kept_len = 0
    try:
        with p.open(errors="replace") as f:
            while chunk := f.read(_CHUNK):
                ended_with_newline = chunk.endswith("\n")
                # A piece is the text of one line within this chunk; every piece
                # but the last ends where a newline did.
                pieces = chunk.split("\n")
                for i, piece in enumerate(pieces):
                    text = piece + "\n" if i < len(pieces) - 1 else piece
                    # Selected lines are start..end (an open end runs to EOF).
                    if text and line_no >= start and (end is None or line_no <= end):
                        if region_start is None:
                            region_start = pos
                        region_len += len(text)
                        if kept_len < max_chars:
                            kept.append(text)
                            kept_len += len(text)
                    pos += len(text)
                    if i < len(pieces) - 1:
                        line_no += 1
    except OSError as exc:
        return f"Error: could not read file: {exc}"

    # An unterminated last line still counts; a trailing newline does not open one.
    total_lines = line_no - 1 if ended_with_newline else line_no
    # An empty file has no line to start from.
    if pos == 0:
        return f"Error: {path} is empty; there is nothing to read"
    # A start beyond the last line names the file and its line count.
    if start > total_lines:
        return (
            f"Error: start_line {start} is past the end of {path}, "
            f"which has {total_lines} lines; use a start_line of {total_lines} or less"
        )

    # Size the content from the longest header this range could produce,
    # leaving room for the newline after it.
    header_bound = len(
        f"[lines {start}-{total_lines} of {total_lines} in {path}; "
        f"line {total_lines} cut after {max_chars} characters; "
        f"continue with offset {pos}, then next start_line {total_lines}]"
    )
    body = "".join(kept)[: max(0, max_chars - header_bound - 1)]
    last_wanted = total_lines if end is None else min(end, total_lines)

    if len(body) == region_len:
        # Everything selected fit.
        tail = "end of file" if last_wanted == total_lines else f"next start_line {last_wanted + 1}"
        return f"[lines {start}-{last_wanted} of {total_lines} in {path}; {tail}]\n{body}"

    # The cap cut the selection. Back up to the last whole line when there is
    # one, so paging continues by line number; only a first line that is by
    # itself longer than a page is cut inside.
    complete = body.count("\n")
    if complete:
        body = body[: body.rfind("\n") + 1]
        last = start + complete - 1
        return f"[lines {start}-{last} of {total_lines} in {path}; next start_line {last + 1}]\n{body}"

    # The first line alone overflows the page: say how far it was cut and where
    # to continue, as a character offset into the file, then where line paging
    # resumes once that line is finished (when a later line exists).
    resume = f", then next start_line {start + 1}" if start < total_lines else ""
    return (
        f"[lines {start}-{start} of {total_lines} in {path}; line {start} cut "
        f"after {len(body)} characters; continue with offset {region_start + len(body)}{resume}]\n{body}"
    )


async def write_file(path: str, content: str) -> str:
    """Write content to a file, creating parent directories if needed."""
    return await asyncio.to_thread(_write_file_sync, path, content)


def _write_file_sync(path: str, content: str) -> str:
    p = pathlib.Path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Wrote {len(content)} chars to {path}"
    except PermissionError as exc:
        return f"Error: permission denied: {exc}"
    except OSError as exc:
        return f"Error: could not write file: {exc}"
