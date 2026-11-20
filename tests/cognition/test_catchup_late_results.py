"""Catch-up segmentation follows the tool-cycle rule (#1046): a result that
arrives after anyone moved on belongs to no cycle, so it neither swallows the
dialog between it and its call (R1) nor makes the segment that ends before it
call the tool "did not complete" (R2)."""

import json

import aiosqlite
from corvidae_cognition.memory import CatchupSegment, build_catchup_segments, logger

from corvidae.context import TOOL_RESULT_ABANDONED, TOOL_RESULT_PENDING
from corvidae.fork import ViewInputs
from corvidae.thinking import strip_reasoning_content

CHANNEL = "irc:#late"
FORK_PROMPT = "Consolidate."
BODY = "word " * 50  # a row of roughly 70 tokens


def _inputs(max_tokens: int) -> ViewInputs:
    """View inputs with a short system prompt, the given budget and the
    "drop" reasoning policy."""
    return ViewInputs("SYS", max_tokens, "system", strip_reasoning_content, b"k" * 32)


def _call(call_id: str, content: str | None = None, **extra) -> tuple[str, dict]:
    """An assistant row making one tool call; its content starts with the
    tag ``CALL-<id>`` unless given, so it counts as a dialog row."""
    return ("message", {
        "role": "assistant", "content": content or f"CALL-{call_id} {BODY}", **extra,
        "tool_calls": [{"id": call_id, "type": "function", "function": {"name": "shell", "arguments": "{}"}}],
    })


def _result(call_id: str) -> tuple[str, dict]:
    """The tool result for ``call_id``."""
    return ("message", {"role": "tool", "tool_call_id": call_id, "content": f"RESULT-{call_id}"})


def _say(role: str, tag: str) -> tuple[str, dict]:
    """A user or assistant dialog row whose content starts with ``tag``."""
    return ("message", {"role": role, "content": f"{tag} {BODY}"})


def _ctx(tag: str) -> tuple[str, dict]:
    """A CONTEXT row."""
    return ("context", {"role": "system", "content": f"[CONTEXT]\n{tag}"})


async def _segments(rows: list[tuple[str, dict]], budget: int) -> list[CatchupSegment]:
    """Seed an in-memory message_log with ``rows`` (ids from 1, no summary) and
    build the catch-up segments for all of it."""
    db = await aiosqlite.connect(":memory:")
    try:
        await db.execute(
            "CREATE TABLE message_log (id INTEGER PRIMARY KEY, channel_id TEXT, "
            "message TEXT, timestamp REAL, message_type TEXT)"
        )
        await db.executemany(
            "INSERT INTO message_log VALUES (?, ?, ?, ?, ?)",
            [(i, CHANNEL, json.dumps(m), float(i), kind) for i, (kind, m) in enumerate(rows, start=1)],
        )
        await db.commit()
        return await build_catchup_segments(db, CHANNEL, 0, len(rows), _inputs(budget), FORK_PROMPT)
    finally:
        await db.close()


def _interleaved_shape() -> list[tuple[str, dict]]:
    """The 2470-2482 shape: a call with large reasoning, a stretch of dialog,
    context and system rows, a second call left unanswered, then the first
    call's result."""
    return [
        _call("c1", reasoning_content="think " * 400),
        _say("user", "U1"), _ctx("k1"), _say("assistant", "A1"),
        ("message", {"role": "system", "content": "SYSNOTE " + BODY}), _ctx("k2"),
        _say("assistant", "A2"), _say("user", "U2"), _ctx("k3"),
        _say("assistant", "A3"), _say("user", "U3"),
        _call("c2"),
        _result("c1"),
    ]


def _chained_shape() -> list[tuple[str, dict]]:
    """Each assistant row makes one call and its result lands after the next
    call, with system rows interleaved."""
    return [
        _say("user", "U1"),
        _call("a", "A1 " + BODY), _ctx("k1"),
        _call("b", "A2 " + BODY), _result("a"),
        _say("user", "U2"), _ctx("k2"),
        _call("c", "A3 " + BODY), _result("b"),
        ("message", {"role": "system", "content": "SYSNOTE " + BODY}),
        _call("d", "A4 " + BODY), _result("c"),
        _say("user", "U3"), _result("d"),
        _say("assistant", "A5"),
    ]


def _dialog_tags(rows: list[tuple[str, dict]]) -> list[tuple[int, str]]:
    """(row id, leading text) of every user/assistant MESSAGE row."""
    return [
        (i, m["content"].split()[0]) for i, (kind, m) in enumerate(rows, start=1)
        if kind == "message" and m["role"] in ("user", "assistant") and m["content"]
    ]


async def _assert_dialog_reaches_a_view(shape, budget: int, caplog) -> None:
    """Every dialog row of ``shape`` appears in some segment's messages and no
    "dropped its own rows" warning names a dialog row id."""
    rows = shape()
    with caplog.at_level("WARNING", logger=logger.name):
        segments = await _segments(rows, budget)
    seen = "\n".join(json.dumps(m) for seg in segments for m in seg.messages)
    for row_id, tag in _dialog_tags(rows):
        assert tag in seen, f"dialog row {row_id} ({tag}) reached no segment's view"
    dropped = [r.getMessage() for r in caplog.records if "dropped its own rows" in r.getMessage()]
    dialog_ids = {str(i) for i, _ in _dialog_tags(rows)}
    for text in dropped:
        named = set(text.split("(message ids ")[1].split(")")[0].split(", "))
        assert not named & dialog_ids, text
    # The budget is below the stretch, so a single segment means it was swallowed whole.
    assert len(segments) >= 2


async def test_interleaved_stretch_loses_no_dialog_to_the_view(caplog):
    """R1: the 2470-2482 shape at a budget below the stretch."""
    await _assert_dialog_reaches_a_view(_interleaved_shape, 700, caplog)


async def test_chained_calls_lose_no_dialog_to_the_view(caplog):
    """R1: calls whose results each land after a later call."""
    await _assert_dialog_reaches_a_view(_chained_shape, 600, caplog)


def _tool_text(segment: CatchupSegment, call_id: str) -> list[str]:
    """The content of every tool message answering ``call_id`` in the segment."""
    return [m["content"] for m in segment.messages if m.get("tool_call_id") == call_id]


async def test_segment_ending_before_a_late_result_renders_the_call_pending():
    """R2: the call is in one segment and its late result in a later one; the
    earlier segment shows the call as still running, not as not completed."""
    rows = [
        _call("c1"),
        _say("user", "U1"), _say("assistant", "A1"), _say("user", "U2"),
        _say("assistant", "A2"), _say("user", "U3"), _say("assistant", "A3"),
        _result("c1"),
        _say("user", "U4"), _say("assistant", "A4"),
    ]
    segments = await _segments(rows, 350)
    # Engineered split: the call and its result are in different segments.
    holder = [k for k, s in enumerate(segments) if any(m.get("tool_calls") for m in s.messages)]
    result_in = [k for k, s in enumerate(segments) if any(r[0] == 8 for r in s.rows)]
    assert holder and result_in and holder[0] < result_in[0]
    texts = _tool_text(segments[holder[0]], "c1")
    assert texts == [TOOL_RESULT_PENDING]
    assert TOOL_RESULT_ABANDONED not in texts
