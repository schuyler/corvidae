"""The notification spill service: oversized notification content written to
a file the agent reads on demand (#558).

A real plugin manager with a FakeClock, a real ToolCollectionPlugin, a stub
tool plugin whose tools declare ``reads_spill_files``, and a real Agent that
is never started: its queues only collect what ``on_notify`` enqueues, and
each item's header is rendered from the row the Agent would store.

Kernel only: nothing here imports webhook_notify or ``corvidae_cognition``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from corvidae.agent import Agent, QueueItem
from corvidae.channel import Channel, ChannelRegistry, load_channel_config
from corvidae.hooks import create_plugin_manager, get_dependency, hookimpl
from corvidae.provenance import frame
from corvidae.tool import Tool
from corvidae.tool_collection import ToolCollectionPlugin
from fake_clock import FakeClock, install_kernel_clock
from corvidae.spill import SPILL_NAME, NotificationSpill

KEY = b"k" * 32
DAY = 86400
AGE_KEY = "daemon.notification_spill_max_age_days"
# 50 distinct-looking characters, so a slice is recognisable.
TEXT = "".join(chr(ord("a") + i % 26) for i in range(50))


def declaring(name: str):
    """A stub tool named ``name`` that declares it can read spill files."""
    async def fn(path: str) -> str:
        """Stub tool."""
        return ""
    fn.__name__ = name
    fn.reads_spill_files = True  # type: ignore[attr-defined]
    return fn


def plain(name: str):
    """A stub tool named ``name`` that declares nothing."""
    async def fn(path: str) -> str:
        """Stub tool."""
        return ""
    fn.__name__ = name
    return fn


class Readers:
    """Registers read_file- and shell-named declaring tools, a non-declaring
    write_file, and any extra tools."""

    def __init__(self, *extra) -> None:
        self.extra = extra

    @hookimpl
    def register_tools(self, tool_registry: list) -> None:
        fns = [declaring("read_file"), declaring("shell"), plain("write_file"), *self.extra]
        tool_registry.extend(Tool.from_function(fn) for fn in fns)


class Pager:
    """A notifying plugin that is not webhook_notify: its only spill code is
    the one notify call."""

    depends_on = frozenset({"spill"})

    def __init__(self, pm) -> None:
        self.pm = pm

    async def page(self, channels: list[Channel], text: str) -> str | None:
        spill = get_dependency(self.pm, "spill", NotificationSpill)
        return await spill.notify(
            channels, source="pager", text=text, annotations={"origin": "pager"}, limit=10,
        )


@dataclass
class Harness:
    """One assembled plugin manager and what its Agent was handed."""

    pm: object
    spill: NotificationSpill
    clock: FakeClock
    registry: ChannelRegistry
    agent: Agent
    config: dict
    base: Path
    # Every QueueItem Agent.on_notify enqueued, in order.
    items: list[QueueItem]

    @property
    def dir(self) -> Path:
        """The spill directory the harness configured (base/spill)."""
        return (self.base / "spill").resolve()

    def channels(self, *ids: str) -> list[Channel]:
        """Channels as a transport would create them."""
        return [self.registry.get_or_create(*cid.split(":", 1)) for cid in ids]

    def header(self, item: QueueItem) -> list[str]:
        """The rendered header lines of the row the Agent stores for ``item``."""
        msg, _ = self.agent._build_conversation_message(item)
        _, out = frame(msg, msg, item.channel.id, KEY)
        lines = out["content"].split("\n")
        return lines[: next(i for i, line in enumerate(lines) if line.startswith("[payload "))]

    def cut_lines(self) -> dict[str, str | None]:
        """Per channel id, the header's cut line, or None without one."""
        return {
            item.channel.id: next(
                (line for line in self.header(item) if line.startswith("[metadata: cut ")), None
            )
            for item in self.items
        }

    async def notify(self, *ids: str, text: str = TEXT, limit: int = 10, source: str = "test",
                     annotations: dict | None = None) -> str | None:
        """Hand ``text`` to the service for the named channels."""
        return await self.spill.notify(
            self.channels(*ids), source=source, text=text, annotations=annotations, limit=limit,
        )


def files_in(directory: Path) -> list[Path]:
    """Every entry directly in ``directory``, or [] when it does not exist."""
    return sorted(directory.iterdir()) if directory.is_dir() else []


def pointer(path: str, shown: int = 10, total: int = 50) -> str:
    """The cut line for a channel that got the pointer to a whole spill."""
    return f"[metadata: cut to the first {shown} of {total} characters; the full text is in {path}]"


def plain_cut(shown: int = 10, total: int = 50) -> str:
    """The cut line for a channel that cannot read spill files."""
    return f"[metadata: cut to the first {shown} of {total} characters]"


@pytest.fixture
async def make(tmp_path):
    """Build a Harness; every FakeClock it made is closed at teardown."""
    clocks: list[FakeClock] = []

    async def build(
        *,
        base: Path | None = None,
        daemon: dict | None = None,
        tools: dict | None = None,
        channels: dict | None = None,
        extra_tools: tuple = (),
        collect: bool = True,
        start: bool = True,
    ) -> Harness:
        base = base or tmp_path
        # Unless a test says otherwise, spill into base/spill.
        if daemon is None:
            daemon = {"notification_spill_dir": str(base / "spill")}
        config = {"_base_dir": base, "daemon": daemon, "tools": tools or {},
                  "channels": channels or {}}
        pm = create_plugin_manager()
        clock, _idle = install_kernel_clock(pm, started=False)
        clocks.append(clock)
        registry = ChannelRegistry()
        load_channel_config(config, registry)
        pm.register(registry, name="registry")
        collection = ToolCollectionPlugin(pm)
        pm.register(collection, name="tools")
        pm.register(Readers(*extra_tools), name="readers")

        # A real Agent whose queues only collect what on_notify enqueues.
        items: list[QueueItem] = []

        class Collect:
            async def enqueue(self, item: QueueItem) -> None:
                items.append(item)

        agent = Agent(pm)
        agent._get_or_create_queue = lambda channel: Collect()
        pm.register(agent, name="agent")

        spill = NotificationSpill()
        pm.register(spill, name="spill")
        await collection.on_init(pm=pm, config=config)
        await spill.on_init(pm=pm, config=config)
        # Without collect, the tool registry is not built yet, as before
        # ToolCollectionPlugin's trylast on_start.
        if collect:
            await collection.rebuild_registry()
        if start:
            await spill.on_start(config=config)
        return Harness(pm, spill, clock, registry, agent, config, base, items)

    yield build
    for clock in clocks:
        await clock.close()


def age(path: Path, clock: FakeClock, seconds: float) -> None:
    """Set ``path``'s mtime ``seconds`` before the fake clock's now."""
    when = clock.now() - seconds
    os.utime(path, (when, when))


# ---------------------------------------------------------------------------
# One kernel service, one file per item
# ---------------------------------------------------------------------------


async def test_a_plugin_spills_with_no_spill_code_of_its_own(make):
    """A plugin that is not webhook_notify spills through the service
    with one notify call; one file, and the header carries the pointer."""
    h = await make()
    path = await Pager(h.pm).page(h.channels("irc:#a"), TEXT)

    # One file, holding the full text, in the spill directory.
    assert files_in(h.dir) == [Path(path)]
    assert Path(path).read_text(encoding="utf-8") == TEXT
    # The channel got the cut text, the plugin's annotations, and the pointer.
    [item] = h.items
    assert item.content == TEXT[:10]
    assert item.annotations == {"origin": "pager"}
    assert h.cut_lines() == {"irc:#a": pointer(path)}


async def test_fanout_to_three_channels_writes_one_file_named_in_every_header(make):
    """A delivery to 3 channels leaves exactly 1 new file, and all 3
    headers name it."""
    h = await make()
    path = await h.notify("irc:#a", "irc:#b", "discord:1")

    assert files_in(h.dir) == [Path(path)]
    assert h.cut_lines() == {cid: pointer(path) for cid in ("irc:#a", "irc:#b", "discord:1")}


async def test_sender_controlled_text_does_not_shape_the_file_name(make, tmp_path):
    """``../``, ``/`` and a sender-chosen delivery id in the text,
    source or annotations change neither the file's name nor its directory."""
    h = await make()
    hostile = "../../etc/x"
    path = await h.notify(
        "irc:#a", text="../" + "/" * 5 + hostile + "a" * 40,
        source=hostile, annotations={"delivery_id": hostile},
    )

    # The only file written anywhere under the test directory is service-named.
    written = sorted(p for p in tmp_path.rglob("*") if p.is_file())
    assert written == [Path(path)]
    assert SPILL_NAME.fullmatch(Path(path).name)
    assert Path(path).parent == h.dir


@pytest.mark.parametrize("setting, expected", [
    (None, "base/state/notification-spill"),
    ("spills/x", "base/spills/x"),
    ("ABSOLUTE", "elsewhere/spill"),
], ids=["unset", "relative", "absolute"])
async def test_dir_defaults_under_base_dir_state_and_follows_the_key(make, tmp_path, setting, expected):
    """Unset, files land under <config dir>/state/notification-spill;
    set, under the configured path (relative to the config directory)."""
    if setting is None:
        daemon = {}
    elif setting == "ABSOLUTE":
        daemon = {"notification_spill_dir": str(tmp_path / "elsewhere" / "spill")}
    else:
        daemon = {"notification_spill_dir": setting}
    h = await make(base=tmp_path / "base", daemon=daemon)
    path = await h.notify("irc:#a")

    assert Path(path).parent == (tmp_path / expected).resolve()


async def test_content_over_the_cap_writes_exactly_cap_characters(make, tmp_path):
    """Content longer than the cap writes a file of exactly cap
    characters, and the header says the file holds the first cap of total."""
    h = await make(daemon={"notification_spill_dir": str(tmp_path / "spill"),
                           "notification_spill_max_chars": 100})
    text = "".join(chr(ord("a") + i % 26) for i in range(250))
    path = await h.notify("irc:#a", text=text)

    assert Path(path).read_text(encoding="utf-8") == text[:100]
    assert h.cut_lines() == {"irc:#a": (
        f"[metadata: cut to the first 10 of 250 characters; "
        f"the first 100 characters are in {path}]"
    )}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


_CHARS_FIX = "use a positive integer number of characters (default 1000000)"
_DAYS_FIX = "use a positive number of days (default 7)"
_DIR_RENDER_FIX = "choose a directory whose path has no brackets"


@pytest.mark.parametrize("key, value, shown, remedy", [
    ("notification_spill_max_chars", 0, "0", _CHARS_FIX),
    ("notification_spill_max_chars", -1, "-1", _CHARS_FIX),
    ("notification_spill_max_chars", True, "True", _CHARS_FIX),
    ("notification_spill_max_chars", "7", "'7'", _CHARS_FIX),
    ("notification_spill_max_chars", 1.5, "1.5", _CHARS_FIX),
    ("notification_spill_max_age_days", 0, "0", _DAYS_FIX),
    ("notification_spill_max_age_days", -1, "-1", _DAYS_FIX),
    ("notification_spill_max_age_days", True, "True", _DAYS_FIX),
    ("notification_spill_max_age_days", "7", "'7'", _DAYS_FIX),
    ("notification_spill_max_age_days", float("nan"), "nan", _DAYS_FIX),
    ("notification_spill_dir", 3, "3", "use a directory path such as state/notification-spill"),
    ("notification_spill_dir", "state/a;b", "state/a;b", _DIR_RENDER_FIX),
    ("notification_spill_dir", "state/[x]", "state/[x]", _DIR_RENDER_FIX),
])
async def test_config_rejects_bad_values_naming_key_value_and_fix(make, key, value, shown, remedy):
    """A bad daemon.notification_spill_* value is refused at
    startup with a message naming the key, the rejected value and the fix."""
    with pytest.raises(ValueError) as exc:
        await make(daemon={key: value})
    for part in (f"daemon.{key}", shown, remedy):
        assert part in str(exc.value)


async def test_a_lone_surrogate_payload_still_spills(make):
    """A text carrying a lone surrogate is still written (one
    character per code point) and every reading channel gets the pointer."""
    text = '"\ud83d"' + "x" * 50
    h = await make()
    path = await h.notify("irc:#a", "irc:#b", text=text)

    assert files_in(h.dir) == [Path(path)]
    assert len(Path(path).read_text(encoding="utf-8")) == len(text)
    assert h.cut_lines() == {cid: pointer(path, total=len(text)) for cid in ("irc:#a", "irc:#b")}


# ---------------------------------------------------------------------------
# Pointer only where the file can be read
# ---------------------------------------------------------------------------


_BOTH = {"disabled": ["read_file", "shell"]}


@pytest.mark.parametrize("tools, channels, readers", [
    (_BOTH, {}, set()),
    ({}, {"irc:#a": {"tools": _BOTH}}, {"discord:1"}),
    ({}, {"irc:*": {"tools": _BOTH}}, {"discord:1"}),
], ids=["global", "per-channel", "transport-wide"])
async def test_pointer_only_where_a_declaring_tool_is_enabled(make, tools, channels, readers):
    """A channel with a declaring tool gets the pointer; one with
    every declaring tool disabled (globally, per channel, or by a
    transport-wide entry) gets the plain cut line."""
    h = await make(tools=tools, channels=channels)
    path = await h.notify("irc:#a", "discord:1")

    assert h.cut_lines() == {
        cid: pointer(path) if cid in readers else plain_cut()
        for cid in ("irc:#a", "discord:1")
    }
    assert [item.content for item in h.items] == [TEXT[:10]] * 2


async def test_a_non_builtin_declaring_tool_earns_the_pointer(make):
    """The kernel names no tool; any enabled tool that declares
    reads_spill_files earns the pointer."""
    h = await make(tools=_BOTH, extra_tools=(declaring("peek"),))
    path = await h.notify("irc:#a")

    assert path is not None
    assert h.cut_lines() == {"irc:#a": pointer(path)}


async def test_no_reader_anywhere_writes_no_file_and_some_writes_one(make, tmp_path):
    """No target channel with a declaring tool writes no file; some
    with one writes exactly one."""
    none = await make(base=tmp_path / "none", tools=_BOTH)
    assert await none.notify("irc:#a", "irc:#b") is None
    assert files_in(none.dir) == []

    some = await make(base=tmp_path / "some", channels={"irc:#a": {"tools": _BOTH}})
    path = await some.notify("irc:#a", "irc:#b")
    assert files_in(some.dir) == [Path(path)]


async def test_delivery_before_the_tool_registry_exists_is_cut_without_a_pointer(make):
    """Before the tool registry is built, every channel gets
    the plain cut line, no file is written, and nothing raises."""
    h = await make(collect=False)
    assert await h.notify("irc:#a", "irc:#b") is None

    assert files_in(h.dir) == []
    assert h.cut_lines() == {"irc:#a": plain_cut(), "irc:#b": plain_cut()}


async def test_unwritable_dir_delivers_cut_with_not_saved_and_warns(make, tmp_path, caplog):
    """With an unwritable spill directory the notification still
    arrives cut, a reading channel's header says the full text was not
    saved, and a WARNING names the directory and the remedy."""
    # The spill directory's path is an existing regular file.
    (tmp_path / "spill").write_text("not a directory")
    h = await make(channels={"irc:#b": {"tools": _BOTH}}, start=False)
    with caplog.at_level(logging.WARNING):
        assert await h.notify("irc:#a", "irc:#b") is None

    assert h.cut_lines() == {
        "irc:#a": "[metadata: cut to the first 10 of 50 characters; the full text was not saved]",
        "irc:#b": plain_cut(),
    }
    assert [item.content for item in h.items] == [TEXT[:10]] * 2
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(str(h.dir) in w and "writable" in w for w in warnings), warnings


# ---------------------------------------------------------------------------
# Cleanup by age
# ---------------------------------------------------------------------------


async def test_startup_sweep_deletes_old_spill_files_and_keeps_others(make):
    """At startup a spill file older than the age is deleted, a
    younger one is kept, and an old file the service did not name is kept."""
    h = await make(start=False)
    h.dir.mkdir()
    old, young, operator = (h.dir / n for n in ("a" * 32 + ".txt", "b" * 32 + ".txt", "restart.json"))
    for path in (old, young, operator):
        path.write_text("x")
    age(old, h.clock, 8 * DAY)
    age(young, h.clock, 1 * DAY)
    age(operator, h.clock, 8 * DAY)

    await h.spill.on_start(config=h.config)

    assert files_in(h.dir) == sorted([young, operator])


async def test_hourly_timer_deletes_a_file_that_ages_out(make):
    """The timer deletes a file that ages out while the daemon runs,
    and re-arms, so a later file ages out on the next pass."""
    h = await make(start=False)
    h.dir.mkdir()
    first, second = (h.dir / n for n in ("c" * 32 + ".txt", "d" * 32 + ".txt"))
    for path in (first, second):
        path.write_text("x")
    # first ages out half an hour after start, second ninety minutes after.
    age(first, h.clock, 7 * DAY - 1800)
    age(second, h.clock, 7 * DAY - 5400)

    await h.spill.on_start(config=h.config)
    assert files_in(h.dir) == [first, second]

    await h.clock.advance(3600)
    assert files_in(h.dir) == [second]
    assert "notification-spill-cleanup" in h.clock.live_timer_names()

    await h.clock.advance(3600)
    assert files_in(h.dir) == []
    assert "notification-spill-cleanup" in h.clock.live_timer_names()


# ---------------------------------------------------------------------------
# Stale pointers
# ---------------------------------------------------------------------------


async def test_stale_pointer_error_inside_dir_only(make):
    """A missing path inside the spill directory gets text naming the
    path and the age policy; a missing path elsewhere gets None."""
    h = await make(start=False)
    missing = h.dir / ("0" * 32 + ".txt")

    error = h.spill.stale_pointer_error(str(missing))
    assert error is not None
    assert error.startswith(f"Error: file not found: {missing}")
    assert "7 days" in error
    assert AGE_KEY in error

    assert h.spill.stale_pointer_error(str(h.base / "elsewhere.txt")) is None


# ---------------------------------------------------------------------------
# Tool results (#575): a result over tools.max_result_chars is cut by the
# Agent and spilled through the same service, with the same header
# ---------------------------------------------------------------------------


async def deliver_result(h: Harness, cid: str, text: str = TEXT, *, limit: int = 10,
                         call_id: str = "call_1") -> QueueItem:
    """Deliver ``text`` as the result of tool call ``call_id`` on ``cid``,
    with the Agent's result cap at ``limit``; return the enqueued item."""
    h.agent._max_tool_result_chars = limit
    [channel] = h.channels(cid)
    await h.agent.on_notify(
        channel=channel, source="task", text=text, tool_call_id=call_id, annotations={},
    )
    return h.items[-1]


def rendered(h: Harness, item: QueueItem) -> str:
    """The content of the row the Agent stores for ``item``, as the model sees it."""
    msg, _ = h.agent._build_conversation_message(item)
    _, out = frame(msg, msg, item.channel.id, KEY)
    return out["content"]


async def test_tool_result_over_the_cap_is_cut_spilled_and_headed(make):
    """A trusted tool result over the cap reaches the window as its head,
    preceded by the kernel's cut line naming the spill file; no fence."""
    h = await make()
    item = await deliver_result(h, "irc:#a")

    [path] = files_in(h.dir)
    assert path.read_text(encoding="utf-8") == TEXT
    assert item.content == TEXT[:10]
    assert rendered(h, item) == f"{pointer(str(path))}\n{TEXT[:10]}"


async def test_tool_result_within_the_cap_is_untouched(make):
    """A result at or under the cap is not cut, spilled or headed."""
    h = await make()
    item = await deliver_result(h, "irc:#a", limit=50)

    assert files_in(h.dir) == []
    assert item.content == TEXT
    assert rendered(h, item) == TEXT


async def test_tool_result_without_a_reader_is_cut_without_a_pointer(make):
    """A channel with no spill-reading tool gets the plain cut line and no file."""
    h = await make(tools=_BOTH)
    item = await deliver_result(h, "irc:#a")

    assert files_in(h.dir) == []
    assert rendered(h, item) == f"{plain_cut()}\n{TEXT[:10]}"


async def test_untrusted_tool_result_keeps_its_fence_and_gains_the_cut_line(make):
    """A declared-untrusted result keeps its header and fence, and the cut
    line sits in the header, outside the fence."""
    h = await make()
    h.agent._dispatched_calls[("irc:#a", "call_1")] = ("web_fetch", True)
    item = await deliver_result(h, "irc:#a")

    [path] = files_in(h.dir)
    lines = rendered(h, item).split("\n")
    assert lines[0].startswith("[metadata: result of your own web_fetch call call_1;")
    assert lines[1] == pointer(str(path))
    assert lines[2].startswith("[payload ") and lines[2].endswith(" begins]")
    assert lines[3] == TEXT[:10]


async def test_tool_result_over_the_spill_cap_says_how_much_was_saved(make, tmp_path):
    """Content over daemon.notification_spill_max_chars is spilled up to it,
    and the header says how much the file holds."""
    h = await make(daemon={"notification_spill_dir": str(tmp_path / "spill"),
                           "notification_spill_max_chars": 20})
    item = await deliver_result(h, "irc:#a")

    [path] = files_in(h.dir)
    assert path.read_text(encoding="utf-8") == TEXT[:20]
    assert rendered(h, item).split("\n")[0] == (
        f"[metadata: cut to the first 10 of 50 characters; "
        f"the first 20 characters are in {path}]"
    )


async def test_tool_result_with_an_unwritable_dir_says_not_saved_and_warns(
        make, tmp_path, caplog):
    """A failed write still delivers the head, says the full text was not
    saved, and logs a WARNING naming the directory and the remedy."""
    (tmp_path / "spill").write_text("not a directory")
    h = await make(start=False)
    with caplog.at_level(logging.WARNING):
        item = await deliver_result(h, "irc:#a")

    assert rendered(h, item) == (
        "[metadata: cut to the first 10 of 50 characters; the full text was not saved]\n"
        + TEXT[:10]
    )
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(str(h.dir) in w and "writable" in w for w in warnings), warnings


async def test_notification_without_a_call_id_is_not_cut_by_the_agent(make):
    """Only tool results are cut by the Agent; other notifications are
    their sender's to cut (#744)."""
    h = await make()
    h.agent._max_tool_result_chars = 10
    [channel] = h.channels("irc:#a")
    await h.agent.on_notify(
        channel=channel, source="task", text=TEXT, tool_call_id=None, annotations={},
    )

    assert h.items[-1].content == TEXT
    assert files_in(h.dir) == []


async def test_stale_pointer_error_names_tool_results_too(make):
    """The stale-pointer error describes spill files generally, not only
    notifications."""
    h = await make(start=False)
    error = h.spill.stale_pointer_error(str(h.dir / ("0" * 32 + ".txt")))
    assert error is not None
    assert "tool result" in error


async def test_contains_is_true_only_inside_the_spill_dir(make):
    """contains() says whether a path is inside the spill directory, so
    read_file can clamp an unranged read of a spill file."""
    h = await make(start=False)
    assert h.spill.contains(str(h.dir / ("0" * 32 + ".txt")))
    assert not h.spill.contains(str(h.base / "elsewhere.txt"))
