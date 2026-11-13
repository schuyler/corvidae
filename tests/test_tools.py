"""Tests for corvidae.tools — the built-in tool plugins (ShellToolPlugin,
FileToolsPlugin, WebToolsPlugin) and their tool functions.
"""

import asyncio
import re
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# The plugins and raw tool functions under test.
from corvidae.tools import (
    FileToolsPlugin,
    ShellToolPlugin,
    WebToolsPlugin,
    read_file,
    shell,
    web_fetch,
    web_search,
    write_file,
)


# ---------------------------------------------------------------------------
# TestShell
# ---------------------------------------------------------------------------


class TestShell:
    async def test_shell_simple_command(self):
        result = await shell("echo hello")
        assert "hello" in result

    async def test_shell_returns_stderr(self):
        # Write to stderr via shell redirection
        result = await shell("echo errtext >&2")
        assert "STDERR" in result
        assert "errtext" in result

    async def test_shell_nonzero_exit_code(self):
        result = await shell("exit 1")
        assert "1" in result

    async def test_shell_timeout(self):
        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
            result = await shell("sleep 9999")
        assert result == "Error: command timed out after 30 seconds"

    async def test_shell_no_output(self):
        result = await shell("true")
        assert result == "(no output)"


# ---------------------------------------------------------------------------
# TestReadFile
# ---------------------------------------------------------------------------


class TestReadFile:
    async def test_read_file_success(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("file contents here")
        result = await read_file(str(f))
        assert result == "file contents here"

    async def test_read_file_not_found(self, tmp_path):
        missing = tmp_path / "does_not_exist.txt"
        result = await read_file(str(missing))
        assert "error" in result.lower() or "not found" in result.lower() or "no such" in result.lower()

    async def test_read_file_too_large(self, tmp_path):
        f = tmp_path / "big.txt"
        f.write_text("x")
        # Mock stat to return size > 1MB
        mock_stat = MagicMock()
        mock_stat.st_size = 1024 * 1024 + 1
        with patch("pathlib.Path.stat", return_value=mock_stat):
            result = await read_file(str(f))
        assert "error" in result.lower() or "too large" in result.lower() or "1mb" in result.lower()

    async def test_read_file_directory(self, tmp_path):
        result = await read_file(str(tmp_path))
        assert "error" in result.lower() or "directory" in result.lower() or "not a file" in result.lower()


# ---------------------------------------------------------------------------
# TestReadFileRange — character offset/length on read_file (#570)
# ---------------------------------------------------------------------------

# Parses a ranged result's header: start, end, total, and next offset (None at
# end of file).
_RANGE_HEADER = re.compile(
    r"^\[characters (\d+)-(\d+) of (\d+) in (.+); (?:next offset (\d+)|end of file)\]\n"
)


async def _ranged_read_file(
    tools_config: dict | None = None,
    daemon: dict | None = None,
    registry=None,
):
    """Return the registered read_file callable and the ToolCollectionPlugin.

    Ranged calls look up the "tools" plugin for max_result_chars, and every
    call looks up the "spill" service for the stale-pointer check, so this
    builds a real plugin manager with the tools, the spill service on a
    FakeClock, and FileToolsPlugin registered, and the tool registry built.
    ``daemon`` is the config's daemon section (the spill keys). ``registry``,
    when given, is registered as the "registry" plugin so read_file can size
    its page to a channel.
    """
    from corvidae.hooks import create_plugin_manager
    from corvidae.spill import NotificationSpill
    from corvidae.tool_collection import ToolCollectionPlugin
    from fake_clock import install_kernel_clock

    config = {"tools": tools_config or {}, "daemon": daemon or {}}
    pm = create_plugin_manager()
    install_kernel_clock(pm, started=False)
    collection = ToolCollectionPlugin(pm)
    pm.register(collection, name="tools")
    spill = NotificationSpill()
    pm.register(spill, name="spill")
    # The channel registry, when the test wants channel-sized pages.
    if registry is not None:
        pm.register(registry, name="registry")
    await collection.on_init(pm=pm, config=config)
    await spill.on_init(pm=pm, config=config)
    files = FileToolsPlugin(pm)
    pm.register(files, name="file_tools")
    await files.on_init(pm=pm, config=config)
    await collection.rebuild_registry()
    return _tool_fns(files)["read_file"], collection


class TestReadFileRange:
    async def test_unranged_output_unchanged(self, tmp_path):
        """R1: with neither parameter, or both null, the content comes back bare."""
        f = tmp_path / "plain.txt"
        f.write_text("file contents here")
        fn, _ = await _ranged_read_file()
        assert await fn(str(f)) == "file contents here"
        assert await fn(str(f), offset=None, length=None) == "file contents here"

    def test_schema_exposes_nullable_offset_and_length(self):
        """R1: the schema lists offset and length as nullable integers."""
        from corvidae.tool import Tool
        registry: list = []
        FileToolsPlugin(None).register_tools(tool_registry=registry)
        tool = next(t for t in registry if isinstance(t, Tool) and t.name == "read_file")
        props = tool.schema["function"]["parameters"]["properties"]
        nullable_int = {"anyOf": [{"type": "integer"}, {"type": "null"}]}
        assert props["offset"] == nullable_int
        assert props["length"] == nullable_int

    async def test_header_states_range_and_total_in_characters(self, tmp_path):
        """R1, R2: the header names the half-open range and the character total."""
        f = tmp_path / "accents.txt"
        f.write_text("ééééé", encoding="utf-8")  # 5 characters, 10 bytes
        result = await read_file(str(f), offset=1, length=2)
        assert result == f"[characters 1-3 of 5 in {f}; next offset 3]\néé"
        result = await read_file(str(f), offset=3)
        assert result == f"[characters 3-5 of 5 in {f}; end of file]\néé"

    async def test_paging_reconstructs_large_one_line_file(self, tmp_path):
        """R2, R6: following each next offset rebuilds the file exactly, and no
        page exceeds max_result_chars, so execute_tool_call never truncates."""
        content = "".join(chr(ord("a") + i % 26) for i in range(300_000))
        f = tmp_path / "big.json"
        f.write_text(content)
        # A configured cap below the default proves the clamp reads the setting.
        cap = 30_000
        fn, _ = await _ranged_read_file({"max_result_chars": cap})
        length = cap * 2

        # Page through the file, starting each read where the last one ended.
        pieces: list[str] = []
        offset = 0
        while True:
            result = await fn(str(f), offset=offset, length=length)
            assert len(result) <= cap
            match = _RANGE_HEADER.match(result)
            assert match, result[:200]
            start, end, total = (int(match.group(i)) for i in (1, 2, 3))
            assert (start, total) == (offset, 300_000)
            body = result[match.end():]
            assert len(body) == end - start
            pieces.append(body)
            if match.group(5) is None:
                break
            offset = int(match.group(5))
            assert offset == end
        assert "".join(pieces) == content

    @pytest.mark.parametrize("offset", [10, 11])
    async def test_offset_at_or_past_end_names_total(self, tmp_path, offset):
        """R3: an offset at or past the end names the file and its true total."""
        f = tmp_path / "ten.txt"
        f.write_text("0123456789")
        result = await read_file(str(f), offset=offset)
        assert result.startswith("Error:")
        assert str(f) in result
        assert "10 characters" in result
        assert "use an offset below 10" in result

    async def test_empty_file_ranged_read(self, tmp_path):
        """R3: a ranged read of an empty file says it is empty."""
        f = tmp_path / "empty.txt"
        f.write_text("")
        result = await read_file(str(f), offset=0)
        assert result == f"Error: {f} is empty; there is nothing to read"

    async def test_negative_offset_rejected(self, tmp_path):
        """R4: names the parameter, the rejected value and the accepted values."""
        f = tmp_path / "ten.txt"
        f.write_text("0123456789")
        result = await read_file(str(f), offset=-5)
        assert result == f"Error: offset is -5 for {f}; use 0 or greater"

    async def test_non_positive_length_rejected(self, tmp_path):
        """R4: names the parameter, the rejected value and the accepted values."""
        f = tmp_path / "ten.txt"
        f.write_text("0123456789")
        result = await read_file(str(f), length=0)
        assert result == (
            f"Error: length is 0 for {f}; use a positive integer, or omit it to read to the end"
        )

    async def test_ranged_read_ignores_max_file_read_bytes(self, tmp_path):
        """R5: a ranged read of an over-limit file succeeds; unranged is refused
        with the remedy."""
        f = tmp_path / "over.txt"
        f.write_text("x" * 1000)
        fn, _ = await _ranged_read_file({"max_file_read_bytes": 100})

        result = await fn(str(f), offset=900, length=50)
        assert result == f"[characters 900-950 of 1000 in {f}; next offset 950]\n" + "x" * 50

        refused = await fn(str(f))
        assert refused.startswith("Error: file too large")
        assert str(f) in refused
        assert "pass offset and length to read it in ranges" in refused


# ---------------------------------------------------------------------------
# TestReadFileLines — line range on read_file (#573)
# ---------------------------------------------------------------------------

# A line-ranged header: first, last, total lines, then the next start line or EOF.
_LINE_HEADER = re.compile(
    r"^\[lines (\d+)-(\d+) of (\d+) in (.+?); (?:next start_line (\d+)|end of file)\]\n"
)


class TestReadFileLines:
    def test_schema_exposes_nullable_start_line_and_end_line(self):
        from corvidae.tool import Tool
        registry: list = []
        FileToolsPlugin(None).register_tools(tool_registry=registry)
        tool = next(t for t in registry if isinstance(t, Tool) and t.name == "read_file")
        props = tool.schema["function"]["parameters"]["properties"]
        nullable_int = {"anyOf": [{"type": "integer"}, {"type": "null"}]}
        assert props["start_line"] == props["end_line"] == nullable_int

    async def test_ranges_are_one_based_inclusive_and_name_lines_total_and_next(self, tmp_path):
        f = tmp_path / "four.txt"
        f.write_text("a\nb\nc\nd\n")
        fn, _ = await _ranged_read_file()
        assert await fn(str(f), start_line=2, end_line=3) == (
            f"[lines 2-3 of 4 in {f}; next start_line 4]\nb\nc\n"
        )
        # An open end reads to the last line; an end past the file is clamped to it.
        assert await fn(str(f), start_line=3) == f"[lines 3-4 of 4 in {f}; end of file]\nc\nd\n"
        assert await fn(str(f), start_line=4, end_line=99) == f"[lines 4-4 of 4 in {f}; end of file]\nd\n"
        # An open start begins at line 1.
        assert await fn(str(f), end_line=1) == f"[lines 1-1 of 4 in {f}; next start_line 2]\na\n"

    async def test_an_unterminated_last_line_counts_as_a_line(self, tmp_path):
        f = tmp_path / "tail.txt"
        f.write_text("a\nb")
        fn, _ = await _ranged_read_file()
        assert await fn(str(f), start_line=2) == f"[lines 2-2 of 2 in {f}; end of file]\nb"

    async def test_a_line_range_cannot_be_combined_with_a_character_range(self, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("a\nb\n")
        fn, _ = await _ranged_read_file()
        result = await fn(str(f), start_line=1, offset=0)
        assert result.startswith("Error:") and str(f) in result and "not both" in result
        assert all(name in result for name in ("start_line", "end_line", "offset", "length"))

    @pytest.mark.parametrize("kwargs, expected", [
        ({"start_line": 0}, "start_line is 0 for {f}; use 1 or greater"),
        ({"end_line": -2}, "end_line is -2 for {f}; use 1 or greater"),
        ({"start_line": 3, "end_line": 2},
         "end_line 2 is before start_line 3 for {f}; use an end_line of 3 or greater"),
        ({"start_line": 4}, "start_line 4 is past the end of {f}, which has 3 lines; "
                            "use a start_line of 3 or less"),
    ])
    async def test_bad_ranges_name_the_parameter_value_and_accepted_values(
        self, tmp_path, kwargs, expected
    ):
        f = tmp_path / "three.txt"
        f.write_text("a\nb\nc\n")
        fn, _ = await _ranged_read_file()
        assert await fn(str(f), **kwargs) == "Error: " + expected.format(f=f)

    async def test_a_line_range_is_not_refused_for_file_size(self, tmp_path):
        f = tmp_path / "over.txt"
        f.write_text("".join(f"line {i}\n" for i in range(1000)))
        fn, _ = await _ranged_read_file({"max_file_read_bytes": 100})
        assert await fn(str(f), start_line=500, end_line=500) == (
            f"[lines 500-500 of 1000 in {f}; next start_line 501]\nline 499\n"
        )

    async def test_paging_by_line_reconstructs_the_file_within_the_cap(self, tmp_path):
        """The clamp backs up to whole lines, so each next start_line resumes cleanly."""
        content = "".join(f"row {i} " + "x" * (i % 37) + "\n" for i in range(3000))
        f = tmp_path / "rows.log"
        f.write_text(content)
        cap = 2_000
        fn, _ = await _ranged_read_file({"max_result_chars": cap})

        pieces, start = [], 1
        while True:
            result = await fn(str(f), start_line=start)
            match = _LINE_HEADER.match(result)
            assert len(result) <= cap and match and int(match.group(1)) == start, result[:200]
            pieces.append(result[match.end():])
            if match.group(5) is None:
                break
            start = int(match.group(5))
        assert "".join(pieces) == content

    async def test_a_line_longer_than_a_page_is_cut_with_a_continuing_offset(self, tmp_path):
        long_line = "".join(chr(ord("a") + i % 26) for i in range(5000))
        f = tmp_path / "wide.txt"
        f.write_text("first\n" + long_line + "\nlast\n")
        fn, _ = await _ranged_read_file({"max_result_chars": 1_000})

        header, _, body = (await fn(str(f), start_line=2)).partition("\n")

        assert "line 2 cut after" in header and "of 3 in" in header
        assert header.endswith(", then next start_line 3]")
        offset = int(re.search(r"continue with offset (\d+)", header).group(1))
        assert offset == len("first\n") + len(body)
        rest = await fn(str(f), offset=offset, length=50)  # the character read resumes the line
        assert rest.partition("\n")[2] == long_line[len(body):len(body) + 50]


# ---------------------------------------------------------------------------
# Notification spill files (#558): read_file and shell can read them, and a
# removed one fails naming the path and the age policy.
# ---------------------------------------------------------------------------


def test_read_file_and_shell_declare_spill_reading():
    """The read_file and shell tools declare, statically, that they read spill files."""
    from corvidae.tool import declares_spill_reader

    fns = _tool_fns(FileToolsPlugin(None)) | _tool_fns(ShellToolPlugin(None))
    assert declares_spill_reader(fns["read_file"])
    assert declares_spill_reader(fns["shell"])


async def test_read_file_pages_a_spill_file(tmp_path):
    """read_file(pointer, offset, length) returns that range of the
    spilled text."""
    from corvidae.channel import ChannelRegistry

    fn, collection = await _ranged_read_file(daemon={"notification_spill_dir": str(tmp_path)})
    spill = collection.pm.get_plugin("spill")
    text = "".join(chr(ord("a") + i % 26) for i in range(5000))
    channel = ChannelRegistry().get_or_create("irc", "#a")
    pointer = await spill.notify([channel], source="test", text=text, annotations=None, limit=100)

    result = await fn(pointer, offset=1000, length=500)
    assert result == f"[characters 1000-1500 of 5000 in {pointer}; next offset 1500]\n" + text[1000:1500]


async def test_read_file_on_a_removed_spill_file_names_path_and_age(tmp_path):
    """The read_file tool on a missing path inside the spill directory names
    the path and the age policy, and its config key."""
    fn, _ = await _ranged_read_file(daemon={"notification_spill_dir": str(tmp_path),
                                            "notification_spill_max_age_days": 3})
    missing = tmp_path.resolve() / ("0" * 32 + ".txt")

    result = await fn(str(missing))
    assert result.startswith(f"Error: file not found: {missing}")
    assert "3 days" in result
    assert "daemon.notification_spill_max_age_days" in result


async def test_unranged_read_of_a_spill_file_pages_from_the_start(tmp_path):
    """An unranged read_file of a spill file returns its first page, clamped
    to tools.max_result_chars with a header naming the next offset, so
    reading a spill file back never produces a result that is cut and
    spilled again (#575)."""
    from corvidae.channel import ChannelRegistry

    fn, collection = await _ranged_read_file(
        tools_config={"max_result_chars": 1000},
        daemon={"notification_spill_dir": str(tmp_path)},
    )
    spill = collection.pm.get_plugin("spill")
    text = "".join(chr(ord("a") + i % 26) for i in range(5000))
    channel = ChannelRegistry().get_or_create("irc", "#a")
    pointer = await spill.notify([channel], source="test", text=text, annotations=None, limit=100)

    result = await fn(pointer)
    # The whole result, header included, fits the cap, so it is never cut.
    assert len(result) <= 1000
    header, body = result.split("\n", 1)
    assert header == f"[characters 0-{len(body)} of 5000 in {pointer}; next offset {len(body)}]"
    assert body == text[: len(body)]


async def test_read_file_elsewhere_keeps_file_not_found(tmp_path):
    """A missing path outside the spill directory keeps read_file's
    own file-not-found message."""
    fn, _ = await _ranged_read_file(daemon={"notification_spill_dir": str(tmp_path / "spill")})
    missing = tmp_path / "other.txt"

    assert await fn(str(missing)) == f"Error: file not found: {missing}"


# ---------------------------------------------------------------------------
# read_file page clamp (#578) — a page is sized to the channel's budget
# ---------------------------------------------------------------------------

# Ordinary prose, so the token count of a page resembles real file reads.
_PROSE = "The quick brown fox jumps over the lazy dog while reading a file.\n"


def _prose_file(tmp_path, chars: int):
    """Write about ``chars`` characters of ordinary ASCII text; return its path."""
    f = tmp_path / "prose.txt"
    f.write_text((_PROSE * (chars // len(_PROSE) + 1))[:chars])
    return f


def _channel_with_budget(budget: int):
    """A real registry whose agent default budget is ``budget``, and a channel on it."""
    from corvidae.channel import ChannelRegistry

    registry = ChannelRegistry(agent_defaults={"max_context_tokens": budget})
    return registry, registry.get_or_create("irc", "#room")


def _ctx_for(channel):
    """The ToolContext a dispatched read_file call receives for ``channel``."""
    from corvidae.tool import ToolContext

    return ToolContext(channel=channel, tool_call_id="t", task_queue=None)


@pytest.mark.parametrize(
    ("cap", "budget", "expected"),
    [
        # Fraction term binds: 0.08 x 24000 x 3.0.
        (100_000, 24_000, 5_760),
        # Cap term binds (R3), also when the cap is below the minimum page.
        (32_000, 220_000, 32_000),
        (1_000, 1_500, 1_000),
        # Minimum-page term binds (R7).
        (100_000, 1_500, 2_000),
        # No channel budget: the cap alone.
        (30_000, None, 30_000),
    ],
)
def test_page_chars_formula(cap, budget, expected):
    """R8: page = min(cap, max(MIN_PAGE_CHARS, fraction x budget x 3.0))."""
    from corvidae.tools.files import page_chars

    assert page_chars(cap, budget, 0.08) == expected


async def _page_chain(fn, path, ctx, **first):
    """Follow the continuation headers from ``first``; return the page bodies."""
    # Each page's header names the next offset or start_line; stop when absent.
    pattern = _LINE_HEADER if "start_line" in first else _RANGE_HEADER
    key = "start_line" if "start_line" in first else "offset"
    bodies, kwargs = [], first
    while kwargs is not None:
        page = await fn(str(path), _ctx=ctx, **kwargs)
        header = pattern.match(page)
        assert header, page[:200]
        bodies.append(page[header.end():])
        kwargs = {**first, key: int(header.group(5))} if header.group(5) else None
    return bodies


async def test_channel_page_is_a_bounded_share_of_the_window(tmp_path):
    """R1/R11: on a 24000-token channel an unranged read of a long file and a
    line-range page are each at most 8% of the window with a continuation
    header; a file within a page comes back bare."""
    from corvidae.context import message_tokens

    # A 24000-token channel and the read_file tool wired to its registry.
    registry, channel = _channel_with_budget(24_000)
    fn, _ = await _ranged_read_file(registry=registry)
    ctx = _ctx_for(channel)

    # Unranged long file: first page with the header naming the next offset.
    first = await fn(str(_prose_file(tmp_path, 40_000)), _ctx=ctx)
    assert message_tokens({"role": "tool", "content": first}) <= 0.08 * 24_000
    header = _RANGE_HEADER.match(first)
    assert header.group(1) == "0" and header.group(3) == "40000"
    assert header.group(5) is not None

    # Line range of short lines: the page limit binds and names next start_line.
    lines = tmp_path / "lines.txt"
    lines.write_text("".join(f"{_PROSE.strip()} {n}\n" for n in range(2_000)))
    line_page = await fn(str(lines), start_line=1, _ctx=ctx)
    assert message_tokens({"role": "tool", "content": line_page}) <= 0.08 * 24_000
    assert _LINE_HEADER.match(line_page).group(5) is not None

    # A file that fits in a page is bare.
    small = tmp_path / "small.txt"
    small.write_text("fits in one page")
    assert await fn(str(small), _ctx=ctx) == "fits in one page"


async def test_page_follows_runtime_budget_override(tmp_path):
    """R2: a runtime override of the budget changes the next page, no restart."""
    # A 24000-token channel and a file longer than any page.
    registry, channel = _channel_with_budget(24_000)
    fn, _ = await _ranged_read_file(registry=registry)
    f = _prose_file(tmp_path, 60_000)
    ctx = _ctx_for(channel)

    # First page at the agent-default budget.
    small_page = await fn(str(f), offset=0, length=10**6, _ctx=ctx)
    # set_settings stores its value here.
    channel.runtime_overrides["max_context_tokens"] = 48_000
    # Second page, same call, after the override.
    large_page = await fn(str(f), offset=0, length=10**6, _ctx=ctx)

    # The page grew with the budget: 5760 characters, then up to 11520.
    assert len(small_page) <= 5_760 < len(large_page) <= 11_520


@pytest.mark.parametrize(
    ("budget", "tools_config", "expected"),
    [
        # R3: the cap binds at 220000 tokens; the default cap at 256000.
        (220_000, {"max_result_chars": 32_000}, 32_000),
        (256_000, {}, 61_440),
        # R7, revised by #752: a 1500-token budget's share (120 tokens) is
        # smaller than the 2000-character minimum page counts, so counting cuts
        # the page to the floor that still leaves room for the header. The cap
        # does not bind below it.
        (1_500, {}, 506),
        (1_500, {"max_result_chars": 1_000}, 506),
        # R9: a configured fraction sets the page (2% x 100000 x 3.0).
        (100_000, {"read_file_window_fraction": 0.02}, 6_000),
    ],
)
async def test_channel_page_size_by_configuration(tmp_path, budget, tools_config, expected):
    """R3/R7/R9: the first page of a long file has the configured size, header intact."""
    # A channel at the budget, the tool configured, and a file longer than any page.
    registry, channel = _channel_with_budget(budget)
    fn, _ = await _ranged_read_file(tools_config, registry=registry)
    f = _prose_file(tmp_path, 150_000)

    # Read the first page.
    page = await fn(str(f), offset=0, length=10**6, _ctx=_ctx_for(channel))

    # The header is intact and the page is within 200 characters of the size.
    assert _RANGE_HEADER.match(page), page[:200]
    assert expected - 200 <= len(page) <= expected


@pytest.mark.parametrize("case", ["no_ctx", "no_channel", "no_registry"])
async def test_no_channel_budget_is_bounded_by_max_result_chars(tmp_path, case):
    """R4: with no _ctx, no channel, or no registry, a long file reads as a
    page of tools.max_result_chars."""
    # A channel exists; the registry is registered unless the case omits it.
    cap = 30_000
    registry, channel = _channel_with_budget(24_000)
    fn, _ = await _ranged_read_file(
        {"max_result_chars": cap}, registry=None if case == "no_registry" else registry
    )
    # The context each case passes: none, one without a channel, or one with.
    ctx = {"no_ctx": None, "no_channel": _ctx_for(None), "no_registry": _ctx_for(channel)}[case]

    # Read a long file with no channel budget available.
    page = await fn(str(_prose_file(tmp_path, 100_000)), _ctx=ctx)

    # The page is the cap alone, with its header.
    assert cap - 200 <= len(page) <= cap
    assert _RANGE_HEADER.match(page)


async def test_dispatched_read_file_is_sized_to_the_calling_channel(tmp_path):
    """R5: dispatch_tool_call with a channel gives the channel-sized page."""
    import json

    from corvidae.tool import dispatch_tool_call

    # A 24000-token channel and the read_file tool wired to its registry.
    registry, channel = _channel_with_budget(24_000)
    fn, _ = await _ranged_read_file(registry=registry)
    # The LLM's tool call for a long file.
    call = {
        "id": "c1",
        "function": {
            "name": "read_file",
            "arguments": json.dumps({"path": str(_prose_file(tmp_path, 40_000))}),
        },
    }

    # Dispatch with the channel in context.
    result = await dispatch_tool_call(call, {"read_file": fn}, channel=channel)

    # 24000 x 8% x 3.0 = 5760 characters at most.
    assert 5_000 <= len(result.content) <= 5_760


def test_read_file_schema_has_no_ctx_and_keeps_its_parameters():
    """R6: the schema does not expose _ctx and the parameter list is unchanged."""
    from corvidae.tool import Tool

    # Register the tools and find read_file.
    registry: list = []
    FileToolsPlugin(None).register_tools(tool_registry=registry)
    tool = next(t for t in registry if isinstance(t, Tool) and t.name == "read_file")
    # The exposed parameters are exactly the original five.
    props = tool.schema["function"]["parameters"]["properties"]
    assert list(props) == ["path", "offset", "length", "start_line", "end_line"]


async def test_character_and_line_paging_under_a_small_budget_reconstruct_the_file(tmp_path):
    """R6: under a small budget, paging by character and by line each takes
    several pages and reproduces the file exactly."""
    # A 1500-token channel and a file of several minimum pages.
    registry, channel = _channel_with_budget(1_500)
    fn, _ = await _ranged_read_file(registry=registry)
    ctx = _ctx_for(channel)
    f = _prose_file(tmp_path, 9_000)
    text = f.read_text()

    # Page through the file by character range.
    by_chars = await _page_chain(fn, f, ctx, offset=0, length=10**6)
    # Page through the file by line range.
    by_lines = await _page_chain(fn, f, ctx, start_line=1, end_line=len(text.splitlines()))

    # Each takes several pages and joins back to the file.
    for bodies in (by_chars, by_lines):
        assert len(bodies) > 1
        assert "".join(bodies) == text


# ---------------------------------------------------------------------------
# TestWriteFile
# ---------------------------------------------------------------------------


class TestWriteFile:
    async def test_write_file_success(self, tmp_path):
        target = tmp_path / "out.txt"
        result = await write_file(str(target), "hello world")
        assert "wrote" in result.lower() or str(target) in result
        assert target.read_text() == "hello world"

    async def test_write_file_creates_parents(self, tmp_path):
        target = tmp_path / "a" / "b" / "c.txt"
        result = await write_file(str(target), "nested")
        assert target.exists()
        assert target.read_text() == "nested"

    async def test_write_file_overwrites(self, tmp_path):
        target = tmp_path / "file.txt"
        await write_file(str(target), "first")
        result = await write_file(str(target), "second")
        assert target.read_text() == "second"

    async def test_write_file_permission_error(self, tmp_path):
        target = tmp_path / "file.txt"
        with patch("pathlib.Path.write_text", side_effect=PermissionError("denied")):
            result = await write_file(str(target), "content")
        assert "error" in result.lower() or "permission" in result.lower() or "denied" in result.lower()


# ---------------------------------------------------------------------------
# TestWebFetch
# ---------------------------------------------------------------------------


def _make_mock_session(status: int, text: str, max_response_bytes: int = 50_000) -> MagicMock:
    """Build a mock aiohttp.ClientSession that returns the given status/text.

    Short responses (those that fit within max_response_bytes) are modelled by
    having readexactly raise IncompleteReadError so the caller receives the
    partial bytes without a truncation indicator.  Large responses are modelled
    by returning exactly max_response_bytes bytes from readexactly so the caller
    appends the truncation indicator — the caller never sees the remainder.

    For simplicity this helper always encodes text as UTF-8 and sets the mock
    encoding to "utf-8".
    """
    text_bytes = text.encode("utf-8")

    mock_content = MagicMock()
    if len(text_bytes) >= max_response_bytes:
        # Simulate a full read: readexactly returns exactly max_response_bytes bytes,
        # which causes the production code to append TRUNCATION_INDICATOR.
        mock_content.readexactly = AsyncMock(return_value=text_bytes[:max_response_bytes])
    else:
        # Simulate a response shorter than the limit: readexactly raises
        # IncompleteReadError carrying all the bytes.
        incomplete_error = asyncio.IncompleteReadError(text_bytes, None)
        mock_content.readexactly = AsyncMock(side_effect=incomplete_error)

    mock_response = MagicMock()
    mock_response.status = status
    mock_response.content = mock_content
    mock_response.get_encoding = MagicMock(return_value="utf-8")

    # Response is used as an async context manager: async with session.get(url) as resp
    mock_response_ctx = AsyncMock()
    mock_response_ctx.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=mock_response_ctx)

    # Session itself is used as an async context manager
    mock_session_ctx = AsyncMock()
    mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

    return mock_session_ctx


class TestWebFetch:
    async def test_web_fetch_success(self):
        mock_session_ctx = _make_mock_session(200, "page content")
        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            result = await web_fetch("http://example.com")
        assert "page content" in result

    async def test_web_fetch_non_200(self):
        mock_session_ctx = _make_mock_session(404, "not found body")
        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            result = await web_fetch("http://example.com/missing")
        assert "404" in result

    async def test_web_fetch_truncates_large_response(self):
        large_body = "x" * 60000
        mock_session_ctx = _make_mock_session(200, large_body)
        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            result = await web_fetch("http://example.com/big")
        assert len(result) < 60000
        assert "[truncated]" in result

    async def test_web_fetch_timeout(self):
        import aiohttp

        mock_session = MagicMock()
        mock_response_ctx = AsyncMock()
        mock_response_ctx.__aenter__ = AsyncMock(side_effect=asyncio.TimeoutError)
        mock_session.get = MagicMock(return_value=mock_response_ctx)

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            result = await web_fetch("http://example.com/slow")
        assert "timed out" in result.lower() or "timeout" in result.lower()

    async def test_web_fetch_connection_error(self):
        import aiohttp

        mock_session = MagicMock()
        mock_response_ctx = AsyncMock()
        mock_response_ctx.__aenter__ = AsyncMock(side_effect=aiohttp.ClientError("connection failed"))
        mock_session.get = MagicMock(return_value=mock_response_ctx)

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            result = await web_fetch("http://example.com/broken")
        assert "error" in result.lower()


# ---------------------------------------------------------------------------
# TestWebFetchStreaming — Fix 2: stream-limit response body via readexactly
# ---------------------------------------------------------------------------


def _make_mock_session_streaming(
    status: int,
    *,
    readexactly_return: bytes | None = None,
    readexactly_error: asyncio.IncompleteReadError | None = None,
    encoding: str = "utf-8",
) -> MagicMock:
    """Build a mock aiohttp.ClientSession using readexactly/get_encoding.

    Pass exactly one of readexactly_return (bytes) or readexactly_error.
    """
    mock_content = MagicMock()
    if readexactly_error is not None:
        mock_content.readexactly = AsyncMock(side_effect=readexactly_error)
    else:
        mock_content.readexactly = AsyncMock(return_value=readexactly_return)

    mock_response = MagicMock()
    mock_response.status = status
    mock_response.content = mock_content
    mock_response.get_encoding = MagicMock(return_value=encoding)

    mock_response_ctx = AsyncMock()
    mock_response_ctx.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=mock_response_ctx)

    return mock_session


class TestWebFetchStreaming:
    """Tests for Fix 2: streaming response body with readexactly."""

    async def test_web_fetch_uses_readexactly(self):
        """web_fetch_with_session must call response.content.readexactly(max_response_bytes)."""
        from corvidae.tools.web import web_fetch_with_session

        partial_data = b"short response"
        error = asyncio.IncompleteReadError(partial_data, None)
        mock_session = _make_mock_session_streaming(
            200, readexactly_error=error
        )
        await web_fetch_with_session(mock_session, "http://example.com", max_response_bytes=50_000)
        mock_session.get.return_value.__aenter__.return_value.content.readexactly.assert_called_once_with(50_000)

    async def test_web_fetch_truncation_on_full_read(self):
        """When readexactly returns exactly max_response_bytes, result ends with TRUNCATION_INDICATOR."""
        from corvidae.tools.web import TRUNCATION_INDICATOR, web_fetch_with_session

        max_bytes = 100
        full_data = b"a" * max_bytes
        mock_session = _make_mock_session_streaming(
            200, readexactly_return=full_data
        )
        result = await web_fetch_with_session(
            mock_session, "http://example.com", max_response_bytes=max_bytes
        )
        assert result.endswith(TRUNCATION_INDICATOR), (
            f"Expected result to end with {TRUNCATION_INDICATOR!r}, got: {result!r}"
        )

    async def test_web_fetch_no_truncation_on_short_response(self):
        """When readexactly raises IncompleteReadError with partial data, no truncation indicator."""
        from corvidae.tools.web import TRUNCATION_INDICATOR, web_fetch_with_session

        partial_data = b"short response"
        error = asyncio.IncompleteReadError(partial_data, None)
        mock_session = _make_mock_session_streaming(
            200, readexactly_error=error
        )
        result = await web_fetch_with_session(
            mock_session, "http://example.com", max_response_bytes=50_000
        )
        assert TRUNCATION_INDICATOR not in result, (
            f"Expected no truncation indicator for short response, got: {result!r}"
        )
        assert "short response" in result

    async def test_web_fetch_decode_error_replaced(self):
        """Incomplete UTF-8 sequence in partial data must not raise; replacement char present."""
        from corvidae.tools.web import web_fetch_with_session

        # b"\xe2\x82" is an incomplete 3-byte UTF-8 sequence (euro sign is \xe2\x82\xac)
        incomplete_utf8 = b"hello " + b"\xe2\x82"
        error = asyncio.IncompleteReadError(incomplete_utf8, None)
        mock_session = _make_mock_session_streaming(
            200, readexactly_error=error
        )
        # Must not raise UnicodeDecodeError
        result = await web_fetch_with_session(
            mock_session, "http://example.com", max_response_bytes=50_000
        )
        assert "\ufffd" in result, (
            f"Expected Unicode replacement character in result, got: {result!r}"
        )

    async def test_web_fetch_no_charset_falls_back_to_utf8(self):
        """get_encoding() raises RuntimeError when the server sends no charset in
        Content-Type and the body was consumed via .content.readexactly (aiohttp
        can only compute the fallback encoding after its own read() API runs).
        This must not propagate out of web_fetch_with_session.
        """
        from corvidae.tools.web import web_fetch_with_session

        partial_data = b"short response, no charset"
        error = asyncio.IncompleteReadError(partial_data, None)
        mock_content = MagicMock()
        mock_content.readexactly = AsyncMock(side_effect=error)

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.content = mock_content
        mock_response.get_encoding = MagicMock(
            side_effect=RuntimeError("Cannot compute fallback encoding of a not yet read body")
        )

        mock_response_ctx = AsyncMock()
        mock_response_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response_ctx)

        result = await web_fetch_with_session(
            mock_session, "http://example.com", max_response_bytes=50_000
        )
        assert result == "short response, no charset"


# ---------------------------------------------------------------------------
# TestWebToolsPlugin (which plugin registers which tool: tests/test_tool_plugins.py)
# ---------------------------------------------------------------------------


class TestWebToolsPlugin:
    def test_web_search_max_results_config(self):
        plugin = WebToolsPlugin(None)
        assert plugin._web_search_max_results == 8

        plugin2 = WebToolsPlugin(None)
        import asyncio
        from unittest.mock import MagicMock
        config = {"tools": {"web_search_max_results": 15}}
        asyncio.run(plugin2.on_init(pm=MagicMock(), config=config))
        asyncio.run(plugin2.on_start(config=config))
        try:
            assert plugin2._web_search_max_results == 15
        finally:
            asyncio.run(plugin2.on_stop())


# ---------------------------------------------------------------------------
# TestWebFetchSessionReuse
# ---------------------------------------------------------------------------


class TestWebFetchSessionReuse:
    def test_web_tools_plugin_has_on_start(self):
        plugin = WebToolsPlugin(None)
        assert hasattr(plugin, "on_start"), "WebToolsPlugin must have an on_start method"
        assert callable(plugin.on_start)

    def test_web_tools_plugin_has_on_stop(self):
        plugin = WebToolsPlugin(None)
        assert hasattr(plugin, "on_stop"), "WebToolsPlugin must have an on_stop method"
        assert callable(plugin.on_stop)

    async def test_session_created_on_start(self):
        import aiohttp
        plugin = WebToolsPlugin(None)
        await plugin.on_start(config={})
        try:
            assert hasattr(plugin, "_session"), "WebToolsPlugin must have a _session attribute after on_start"
            assert isinstance(plugin._session, aiohttp.ClientSession)
        finally:
            if hasattr(plugin, "_session") and plugin._session is not None:
                await plugin._session.close()

    async def test_session_closed_on_stop(self):
        plugin = WebToolsPlugin(None)
        await plugin.on_start(config={})
        session = plugin._session
        await plugin.on_stop()
        assert session.closed, "Session must be closed after on_stop"


# ---------------------------------------------------------------------------
# TestFileToolsAsyncIO
# ---------------------------------------------------------------------------


class TestFileToolsAsyncIO:
    async def test_read_file_uses_thread(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("contents")
        with patch("asyncio.to_thread", wraps=asyncio.to_thread) as mock_to_thread:
            await read_file(str(f))
        mock_to_thread.assert_called()

    async def test_write_file_uses_thread(self, tmp_path):
        target = tmp_path / "out.txt"
        with patch("asyncio.to_thread", wraps=asyncio.to_thread) as mock_to_thread:
            await write_file(str(target), "data")
        mock_to_thread.assert_called()


# ---------------------------------------------------------------------------
# TestWebSearch — Mocked DuckDuckGo integration
# ---------------------------------------------------------------------------


def _make_ddgs_mock(search_results):
    """Return a MagicMock that mimics DDGS().text() yielding the given results.

    search_results should be an iterable of dicts with optional keys:
      title (str), href (str), body (str)
    """
    mock_ddgs = MagicMock()
    mock_ctx = MagicMock()
    mock_ddgs.text.return_value = search_results
    mock_ctx.__enter__ = MagicMock(return_value=mock_ddgs)
    mock_ctx.__exit__ = MagicMock(return_value=False)
    return mock_ctx


class TestWebSearch:
    async def test_web_search_returns_formatted_results(self):
        results = [
            {"title": "First Result", "href": "http://example.com/1", "body": "snippet one"},
            {"title": "Second Result", "href": "http://example.com/2", "body": "snippet two"},
        ]
        mock_ctx = _make_ddgs_mock(results)
        with patch("corvidae.tools.web.DDGS", return_value=mock_ctx):
            output = await web_search("test query")
        assert "test query" in output
        assert "First Result" in output
        assert "http://example.com/1" in output
        assert "snippet one" in output
        assert "Second Result" in output

    async def test_web_search_skips_results_without_url(self):
        results = [
            {"title": "Bad Entry", "href": "", "body": "no url"},
            {"title": "Good Entry", "href": "http://good.com", "body": "has url"},
        ]
        mock_ctx = _make_ddgs_mock(results)
        with patch("corvidae.tools.web.DDGS", return_value=mock_ctx):
            output = await web_search("test")
        assert "Bad Entry" not in output
        assert "Good Entry" in output

    async def test_web_search_no_results(self):
        mock_ctx = _make_ddgs_mock([])
        with patch("corvidae.tools.web.DDGS", return_value=mock_ctx):
            output = await web_search("zxcvbnmqwerty no results ever")
        assert "No results found." in output

    async def test_web_search_custom_max_results(self):
        # The web_search function passes max_results to ddgs.text(). We verify
        # the count by checking the output only contains 3 result blocks when
        # the mock yields more than 3 results total.
        all_results = [
            {"title": f"Result {i}", "href": f"http://example.com/{i}", "body": f"snippet {i}"}
            for i in range(20)
        ]
        mock_ddgs = MagicMock()
        # ddgs.text(query, max_results=3) should only return the first 3 items
        mock_ddgs.text.return_value = all_results[:3]
        mock_ctx = MagicMock()
        mock_ddgs.text.return_value = all_results[:3]
        mock_ctx.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ctx.__exit__ = MagicMock(return_value=False)

        async def _mock_to_thread(fn):
            return fn()

        with patch("corvidae.tools.web.DDGS", return_value=mock_ctx):
            with patch("asyncio.to_thread", new=_mock_to_thread):
                output = await web_search("test", max_results=3)
        # Count how many separator blocks appear (3 results = 2 separators)
        assert output.count("=" * 60) == 2, f"Expected 2 separators for 3 results, got {output[:200]}"

    async def test_web_search_error_handling(self):
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(side_effect=RuntimeError("network down"))
        mock_ctx.__exit__ = MagicMock(return_value=False)
        with patch("corvidae.tools.web.DDGS", return_value=mock_ctx):
            output = await web_search("error test")
        assert "Error searching the web" in output

    async def test_web_search_error_names_the_query_and_failure_class(self):
        """#105 item 2: an unnameable failure cannot be traced to its request.

        Same shape as web_fetch: operation, input, failure class, detail.
        """
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(side_effect=RuntimeError("network down"))
        mock_ctx.__exit__ = MagicMock(return_value=False)
        with patch("corvidae.tools.web.DDGS", return_value=mock_ctx):
            output = await web_search("unusual query terms")
        assert "unusual query terms" in output, (
            f"the error must name the query that failed: {output!r}"
        )
        assert "RuntimeError" in output, (
            f"the error must name the failure class: {output!r}"
        )
        assert "network down" in output, f"the error must keep the detail: {output!r}"

    async def test_web_search_separator(self):
        results = [
            {"title": "R1", "href": "http://a.com", "body": "s1"},
            {"title": "R2", "href": "http://b.com", "body": "s2"},
        ]
        mock_ctx = _make_ddgs_mock(results)
        with patch("corvidae.tools.web.DDGS", return_value=mock_ctx):
            output = await web_search("sep test")
        assert "=" * 60 in output

    async def test_web_search_uses_asyncio_to_thread(self, tmp_path):
        """web_search must run ddgs via asyncio.to_thread to avoid blocking."""
        with patch("asyncio.to_thread", wraps=asyncio.to_thread) as mock_to_thread:
            mock_ctx = _make_ddgs_mock([])
            with patch("corvidae.tools.web.DDGS", return_value=mock_ctx):
                await web_search("to thread test")
        mock_to_thread.assert_called_once()


class TestWebFetchErrorContext:
    async def test_connection_error_names_url_and_failure_class(self):
        """A connection failure names the request URL and the failure class."""
        import aiohttp

        mock_session = MagicMock()
        mock_response_ctx = AsyncMock()
        mock_response_ctx.__aenter__ = AsyncMock(
            side_effect=aiohttp.ClientConnectorError(
                connection_key=MagicMock(), os_error=OSError("refused")
            )
        )
        mock_session.get = MagicMock(return_value=mock_response_ctx)

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            result = await web_fetch("http://example.com/broken")

        assert "http://example.com/broken" in result
        assert "ClientConnectorError" in result


# ---------------------------------------------------------------------------
# Issue #94: own_timeout declarations and shell cancellation cleanup
# ---------------------------------------------------------------------------


def _tool_fns(plugin) -> dict:
    """Map tool name -> callable for everything the plugin registers."""
    reg: list = []
    plugin.register_tools(tool_registry=reg)
    return {t.name: t.fn for t in reg}


async def test_shell_and_web_fetch_declare_live_own_timeout():
    """R4: shell/web_fetch declare own_timeout, read live from the plugin."""
    from corvidae.tool import resolve_own_timeout

    config = {"tools": {"shell_timeout": 7, "web_fetch_timeout": 9}}
    shell_plugin = ShellToolPlugin(None)
    web_plugin = WebToolsPlugin(None)
    await shell_plugin.on_init(pm=MagicMock(), config=config)
    await web_plugin.on_init(pm=MagicMock(), config=config)
    try:
        fns = _tool_fns(shell_plugin) | _tool_fns(web_plugin)
        assert resolve_own_timeout(fns["shell"]) == 7
        assert resolve_own_timeout(fns["web_fetch"]) == 9

        # Read at call time: changing the plugin attribute changes the answer.
        shell_plugin._shell_timeout = 3
        assert resolve_own_timeout(fns["shell"]) == 3
        web_plugin._web_fetch_timeout = 2
        assert resolve_own_timeout(fns["web_fetch"]) == 2
    finally:
        # on_stop closes a session only if on_start opened one.
        await web_plugin.on_stop()


def _pid_alive(pid: int) -> bool:
    """True while ``pid`` exists (signal 0 probes without delivering)."""
    import os

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX sh and signals")
async def test_shell_cancelled_by_call_timeout_kills_process(tmp_path):
    """R5: the outer timer cancelling shell kills and reaps its process."""
    import json
    import os
    import signal
    import time

    from corvidae.tool import ToolTimeouts, dispatch_tool_call

    plugin = ShellToolPlugin(None)
    await plugin.on_init(pm=MagicMock(), config={"tools": {"shell_timeout": 30}})
    shell_fn = _tool_fns(plugin)["shell"]

    pidfile = tmp_path / "pid"
    marker = tmp_path / "marker"
    # Record the shell's pid, linger, then do a visible side effect.
    command = f"echo $$ > {pidfile}; sleep 1; echo hi > {marker}"
    call = {"id": "c1", "function": {"name": "shell", "arguments": json.dumps({"command": command})}}

    start = time.monotonic()
    try:
        # Own timeout 30 > 0.3, so the outer timer applies.
        r = await asyncio.wait_for(
            dispatch_tool_call(call, {"shell": shell_fn}, call_timeouts=ToolTimeouts(600, {"shell": 0.3})),
            5,
        )
        elapsed = time.monotonic() - start
        assert r.content == "Error: tool 'shell' timed out after 0.3s"
        assert elapsed < 2.0

        assert pidfile.exists(), "shell never wrote its pid file"
        pid = int(pidfile.read_text().strip())
        # Killed and reaped: the pid is gone.
        assert not _pid_alive(pid), f"shell process {pid} survived the call timeout"

        # And its follow-up side effect never happens.
        await asyncio.sleep(1.2)
        assert not marker.exists()
    finally:
        # Never leave the test's process behind if the assertion failed.
        if pidfile.exists():
            try:
                os.kill(int(pidfile.read_text().strip()), signal.SIGKILL)
            except (ProcessLookupError, ValueError):
                pass


async def test_shell_cancel_before_spawn_reraises():
    """Regression guard (passes today): cancelling shell during spawn re-raises
    CancelledError, with no UnboundLocalError from a missing ``proc``."""
    started = asyncio.Event()

    async def blocked_spawn(*args, **kwargs):
        # Stand in for create_subprocess_shell: block until cancelled.
        started.set()
        await asyncio.sleep(30)

    with patch("asyncio.create_subprocess_shell", side_effect=blocked_spawn):
        task = asyncio.ensure_future(shell("echo hi"))
        await asyncio.wait_for(started.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)


# ---------------------------------------------------------------------------
# Untrusted tool output (#557, #571): the declaration and who makes it.
# ---------------------------------------------------------------------------


def test_declares_untrusted_output_is_true_only_for_the_literal_true():
    """R1: only ``returns_untrusted is True`` declares; absent or truthy-non-True does not."""
    from corvidae.tool import declares_untrusted_output

    def fn() -> None: ...

    assert declares_untrusted_output(fn) is False
    fn.returns_untrusted = 1  # type: ignore[attr-defined]
    assert declares_untrusted_output(fn) is False
    fn.returns_untrusted = True  # type: ignore[attr-defined]
    assert declares_untrusted_output(fn) is True


def test_read_file_web_fetch_and_web_search_declare_untrusted_output():
    """R5/R6: the built-in tools that return outside text declare it."""
    from corvidae.tool import declares_untrusted_output

    fns = _tool_fns(FileToolsPlugin(None)) | _tool_fns(WebToolsPlugin(None))
    for name in ("read_file", "web_fetch", "web_search"):
        assert declares_untrusted_output(fns[name]), f"{name} must declare returns_untrusted"


# ---------------------------------------------------------------------------
# read_file page clamp measured in tokens (#752)
# ---------------------------------------------------------------------------

# Dense content: base64-style text tokenizes near 1.4 characters per token
# under the window's own counter, so the #578 character estimate over-sizes it.
_DENSE = "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVphYmNkZWZnaGlqa2xtbm9wcXJzdHV2d3h5ejAxMjM0NTY3ODkrLw=="


def _dense_file(tmp_path, chars: int):
    """Write about ``chars`` characters of dense (low chars/token) content."""
    f = tmp_path / "dense.txt"
    f.write_text((_DENSE * (chars // len(_DENSE) + 1))[:chars])
    return f


def _tool_message_tokens(page: str) -> int:
    """The page's cost the way the window counts a tool result."""
    from corvidae.context import message_tokens

    return message_tokens({"role": "tool", "content": page})


async def test_dense_page_fits_the_share_by_measurement(tmp_path):
    """R: whatever a page contains, it takes no more than the configured share
    of the channel's window by message_tokens."""
    # A 24000-token channel: the share is 1920 tokens.
    registry, channel = _channel_with_budget(24_000)
    fn, _ = await _ranged_read_file(registry=registry)
    ctx = _ctx_for(channel)
    share = 0.08 * 24_000

    # An unranged read of a file far longer than any page.
    dense = _dense_file(tmp_path, 40_000)
    first = await fn(str(dense), _ctx=ctx)
    assert _RANGE_HEADER.match(first), first[:200]
    assert _tool_message_tokens(first) <= share, _tool_message_tokens(first)

    # The character estimate alone would have let this page exceed the share,
    # so the measurement is what clamped it.
    from corvidae.tools.files import page_chars

    estimated_chars = page_chars(100_000, 24_000, 0.08)
    estimated_page = str(dense.read_text()[:estimated_chars])
    assert _tool_message_tokens(estimated_page) > share

    # A line range of dense content is clamped too.
    lines_file = tmp_path / "dense_lines.txt"
    lines_file.write_text("".join(f"{_DENSE}\n" for _ in range(200)))
    line_page = await fn(str(lines_file), start_line=1, _ctx=ctx)
    assert _LINE_HEADER.match(line_page), line_page[:200]
    assert _tool_message_tokens(line_page) <= share, _tool_message_tokens(line_page)


async def test_dense_pages_reconstruct_the_file(tmp_path):
    """R: the clamped page still pages the file back exactly."""
    registry, channel = _channel_with_budget(24_000)
    fn, _ = await _ranged_read_file(registry=registry)
    ctx = _ctx_for(channel)
    f = _dense_file(tmp_path, 12_000)
    text = f.read_text()

    by_chars = await _page_chain(fn, f, ctx, offset=0, length=10**6)

    assert len(by_chars) > 1
    assert "".join(by_chars) == text


async def test_prose_page_is_not_shrunk_by_the_measurement(tmp_path):
    """R: ordinary text keeps the #578 page; counting only refines it."""
    registry, channel = _channel_with_budget(24_000)
    fn, _ = await _ranged_read_file(registry=registry)
    ctx = _ctx_for(channel)

    page = await fn(str(_prose_file(tmp_path, 150_000)), offset=0, length=10**6, _ctx=ctx)

    # 0.08 x 24000 x 3.0 = 5760 characters, and prose fits the share there.
    assert 5_500 <= len(page) <= 5_760
