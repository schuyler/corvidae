"""Catch-up consolidation from the agent's rendered view (#380, #388).

A consolidation range that has left the live window is written from the
view the agent had: the channel's system prompt, the summary in force plus
the rows after its boundary (trimmed to the channel's budget, under its
reasoning_history policy), and the consolidation instruction, sent on the
consolidation stage's client (main) with no tools. A range spanning more
than one window, or more than one budget, is split into one view and one
record per segment.

Summaries are created the way the daemon creates them: rows are persisted,
then ``PersistencePlugin.on_compaction`` writes the summary row (never an
INSERT of a summary row by hand). Timestamps come from a deterministic
clock so the summary-boundary rule (a summary is dated just before the
oldest row it retained) is exact.
"""

import json
import logging
import re
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

import corvidae.persistence as persistence_module
from corvidae.context import MessageType, count_tokens, message_tokens
from corvidae_cognition.memory import (
    CONSOLIDATION_EXTRA_BODY,
    DEFAULT_CONSOLIDATION_FORK_PROMPT,
    _fork_range_sentence,
)
from helpers import FakeForkAgent, RecordingClient
from test_compaction_user_query import _as_server_renders
from test_mid_conversation_system_role import render_swift

from .test_memory_consolidation import (
    CONSOLIDATION_JSON,
    assert_shared_columns_equal,
    build_consolidation_env,
    fetch_memories,
    fetch_stamped_memories,
    fetch_watermark,
)

SYSTEM = FakeForkAgent.SYSTEM_PROMPT  # "IDENTITY\n\nREGISTER"
PAD = "word " * 90  # about 90 tokens of filler per row
SUMMARY_PREFIX = "[Summary of earlier conversation]\n"


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def tick_clock(monkeypatch):
    """A deterministic clock for persistence: every timestamp is one second
    after the last, so row order and the summary boundary are exact."""
    state = {"now": 1000.0}

    def tick() -> float:
        """The next timestamp."""
        state["now"] += 1.0
        return state["now"]

    monkeypatch.setattr(persistence_module, "time", SimpleNamespace(time=tick))


@dataclass
class Env:
    """The consolidation env plus view-test helpers."""

    memory: object
    persistence: object
    channel: object
    db: object
    agent: FakeForkAgent
    main: RecordingClient
    bg: RecordingClient
    stages: list

    async def rows(self, messages: list[dict]) -> list[int]:
        """Persist ``messages`` as ordinary message rows; returns their ids."""
        return [
            await self.persistence.persist_message(
                channel=self.channel, message=dict(m), message_type=MessageType.MESSAGE
            )
            for m in messages
        ]

    async def compact(self, text: str, retain: int, compacted: list[int]) -> int:
        """Write a summary row through the real ``on_compaction``; returns
        its row id. ``retain`` is the retained tail's length."""
        await self.persistence.on_compaction(
            channel=self.channel,
            summary_msg={"role": "assistant", "content": SUMMARY_PREFIX + text},
            retain_count=retain,
            compacted_ids=compacted,
        )
        async with self.db.execute(
            "SELECT MAX(id) FROM message_log WHERE channel_id = ? AND message_type = 'summary'",
            (self.channel.id,),
        ) as cursor:
            return (await cursor.fetchone())[0]

    async def watermark_at(self, row_id: int) -> None:
        """Move the consolidation watermark to ``row_id``."""
        mem_db = await self.memory._ensure_schema()
        await self.memory._set_watermark(mem_db, self.channel.id, row_id)
        await mem_db.commit()

    async def corrupt(self, row_id: int) -> None:
        """Make a stored row undecodable (as a hand edit or an old version
        could), past the FTS triggers that would reject invalid JSON."""
        await self.db.execute("DROP TRIGGER IF EXISTS message_log_au")
        await self.db.execute(
            "UPDATE message_log SET message = '{not valid json' WHERE id = ?", (row_id,)
        )
        await self.db.commit()

    async def consolidate(self, range_end: int) -> None:
        """Run the idle consolidation of everything up to ``range_end``."""
        await self.memory._consolidate_range(self.channel.id, range_end, "idle")


@pytest.fixture
async def env():
    """The consolidation env with a recording main and background client (the
    consolidation stage resolves to main), the forks-off view-capable agent,
    and a log of requested LLM stages."""
    memory, persistence, _stub, channel, db = await build_consolidation_env()
    llm = memory._llm()
    main = RecordingClient(CONSOLIDATION_JSON, model="main-model")
    bg = RecordingClient("BACKGROUND SHOULD NOT BE CALLED")
    llm._clients["main"] = main
    llm._clients["background"] = bg
    # Record every stage asked for (instance attribute shadows the method).
    stages: list[str | None] = []
    real_get_client = llm.get_client

    def get_client(stage: str | None = None):
        stages.append(stage)
        return real_get_client(stage)

    llm.get_client = get_client
    agent = memory.pm.get_plugin("agent")
    # The stub embeds every record to one vector, so near-dup merge would fold
    # consecutive segment records together; real summaries differ.
    memory._dup_threshold = 1.1
    yield Env(memory, persistence, channel, db, agent, main, bg, stages)
    await db.close()


def user(text: str) -> dict:
    """A user message."""
    return {"role": "user", "content": text}


def assistant(text: str) -> dict:
    """An assistant message."""
    return {"role": "assistant", "content": text}


def alternating(texts: list[str], start: str = "user") -> list[dict]:
    """Messages with these contents, roles alternating from ``start``."""
    roles = ["user", "assistant"] if start == "user" else ["assistant", "user"]
    return [{"role": roles[i % 2], "content": t} for i, t in enumerate(texts)]


def rendered_size(messages: list[dict]) -> int:
    """The size of a request as sent (the measure every budget claim uses):
    the first message's content by ``count_tokens`` plus every later message
    by ``message_tokens``."""
    return count_tokens(messages[0]["content"]) + sum(message_tokens(m) for m in messages[1:])


def dump(messages: list[dict]) -> str:
    """The request as one searchable string."""
    return json.dumps(messages)


def expected_instruction(prefix: list[dict], dialog_ids: set[int]) -> dict:
    """The consolidation instruction message: the fork prompt, a blank line,
    and the range sentence over ``prefix`` (the segment's window)."""
    return {
        "role": "user",
        "content": DEFAULT_CONSOLIDATION_FORK_PROMPT + "\n\n"
        + _fork_range_sentence(prefix, dialog_ids),
    }


def tagged(message: dict, row_id: int, kind=MessageType.MESSAGE) -> dict:
    """A window message tagged the way ``resolve_conversation`` tags it."""
    return {**message, "_db_id": row_id, "_message_type": kind}


def budget_for_rows(rows_per_segment: float) -> int:
    """A ``max_context_tokens`` under which about ``rows_per_segment`` rows
    of ``PAD`` fit beside the system prompt and a full consolidation
    instruction (the long range sentence, which is the larger one)."""
    row = {"role": "user", "content": "ROW00 " + PAD}
    sentence = _fork_range_sentence(
        [tagged(user("x"), 0), tagged(row, 1)], {1}
    )
    instruction = {
        "role": "user", "content": DEFAULT_CONSOLIDATION_FORK_PROMPT + "\n\n" + sentence
    }
    return (
        count_tokens(SYSTEM) + message_tokens(instruction)
        + int(rows_per_segment * message_tokens(row) + 0.4 * message_tokens(row))
    )


def warnings_naming(caplog, *needles: str) -> list[str]:
    """WARNING texts from the memory plugin that contain every needle (a
    needle that is all digits matches as a whole number)."""
    found = []
    for record in caplog.records:
        if record.levelno != logging.WARNING or record.name != "corvidae.memory":
            continue
        text = record.getMessage()
        if all(
            re.search(rf"\b{n}\b", text) if n.isdigit() else n in text for n in needles
        ):
            found.append(text)
    return found


# ---------------------------------------------------------------------------
# R1: the request is the channel's rendered view
# ---------------------------------------------------------------------------


async def test_catchup_request_is_the_channels_rendered_view(env):
    """R1: a range outside the window is written from the view: the channel's
    system prompt, the summary in force, every row from its boundary through
    the range end (rows below the watermark included, rows after it not), and
    the consolidation instruction; never the standalone prompt; no tools."""
    old = await env.rows(alternating(["old one", "old two"]))
    summary_id = await env.compact("what came before", 0, old)
    rows = await env.rows(alternating(["r1 asks", "r2 answers", "r3 asks", "r4 answers", "r5 after"]))
    # r1 and r2 were consolidated earlier: context, not part of the range.
    await env.watermark_at(rows[1])

    await env.consolidate(rows[3])

    [call] = env.main.chat_calls
    messages = call["messages"]
    assert messages[0] == {"role": "system", "content": SYSTEM}
    summary = {"role": "assistant", "content": SUMMARY_PREFIX + "what came before"}
    expected_view = alternating(["r1 asks", "r2 answers", "r3 asks", "r4 answers"])
    assert messages[1:-1] == [summary] + expected_view
    assert "r5 after" not in dump(messages)
    window = [tagged(summary, summary_id, MessageType.SUMMARY)] + [
        tagged(m, i) for m, i in zip(expected_view, rows)
    ]
    assert messages[-1] == expected_instruction(window, {rows[2], rows[3]})
    assert "You are consolidating a conversation segment" not in dump(messages)
    assert call["tools"] is None
    assert rendered_size(messages) <= env.agent.max_tokens
    # One record for the range, the watermark at its end.
    [record] = await fetch_memories(env.db)
    assert (record["msg_id_start"], record["msg_id_end"]) == (rows[2], rows[3])
    assert await fetch_watermark(env.db, env.channel.id) == rows[3]


async def test_summary_in_force_is_newest_decodable_before_range(env):
    """R1: the summary in force is the newest decodable summary dated before
    the range: a corrupt newer one is skipped, one written after the range
    is not in force."""
    old = await env.rows(alternating(["old one", "old two"]))
    await env.compact("OLDEST decodable", 0, old)
    middle = await env.rows(alternating(["middle one", "middle two"]))
    corrupt_id = await env.compact("MIDDLE corrupt", 0, middle)
    rows = await env.rows(alternating(["range one", "range two", "range three"]))
    await env.compact("LATE after the range", 0, rows)
    await env.corrupt(corrupt_id)
    # Everything before the range was consolidated.
    await env.watermark_at(middle[-1])

    await env.consolidate(rows[-1])

    [call] = env.main.chat_calls
    messages = call["messages"]
    assert messages[0] == {"role": "system", "content": SYSTEM}
    assert messages[1] == {"role": "assistant", "content": SUMMARY_PREFIX + "OLDEST decodable"}
    text = dump(messages)
    assert "MIDDLE corrupt" not in text and "LATE after the range" not in text
    for content in ("middle one", "range one", "range two", "range three"):
        assert content in text


# ---------------------------------------------------------------------------
# R1/R2: the consolidation stage
# ---------------------------------------------------------------------------


async def test_catchup_call_asks_for_consolidation_stage(env):
    """R1/R2: the catch-up call asks for the ``consolidation`` stage, which
    resolves to main (nothing resolves through background), as a view (the
    system prompt first); the background client gets no consolidation call."""
    rows = await env.rows(alternating(["hello there", "hi, how can I help"]))

    await env.consolidate(rows[-1])

    assert "consolidation" in env.stages
    [call] = env.main.chat_calls
    assert call["messages"][0]["content"] == "IDENTITY\n\nREGISTER"
    assert env.bg.chat_calls == []


async def test_record_embedding_asks_for_the_embedding_stage(env):
    """R1: embedding the written record requests the ``embedding`` stage."""
    rows = await env.rows(alternating(["hello there", "hi, how can I help"]))

    await env.consolidate(rows[-1])

    assert "embedding" in env.stages


# ---------------------------------------------------------------------------
# R7: the channel's reasoning_history policy
# ---------------------------------------------------------------------------


async def test_catchup_view_applies_reasoning_history_drop(env):
    """R7: with ``drop``, reasoning stored on an assistant row never reaches
    the request, though the row's content does."""
    env.agent.reasoning_history = "drop"
    rows = await env.rows([
        user("what is the capital of France?"),
        {"role": "assistant", "content": "ANSWER: Paris", "reasoning_content": "SECRET THINKING"},
    ])

    await env.consolidate(rows[-1])

    [call] = env.main.chat_calls
    messages = call["messages"]
    # A view (system prompt first) that carries the stored row's content.
    assert messages[0] == {"role": "system", "content": SYSTEM}
    assert any(m.get("content") == "ANSWER: Paris" for m in messages[1:])
    assert "SECRET THINKING" not in dump(messages)
    assert not any("reasoning_content" in m for m in messages)


# ---------------------------------------------------------------------------
# O1, C1: a range spanning two windows is one record per window
# ---------------------------------------------------------------------------


async def test_catchup_spanning_two_windows_writes_one_record_per_segment(env):
    """O1/C1: rows A, the retained tail T, then a summary written by the real
    ``on_compaction`` (its row lands after T, dated just before T's oldest
    row), then rows B. One range from below A gives two views and two
    records: A under no summary; T and B under the summary. The summary row's
    id lies inside the second record's span."""
    a_rows = await env.rows(alternating(["A one", "A two", "A three"]))
    t_rows = await env.rows(alternating(["T one", "T two"], start="assistant"))
    summary_id = await env.compact("S1 of the first window", len(t_rows), a_rows)
    b_rows = await env.rows(alternating(["B one", "B two"], start="assistant"))
    assert t_rows[-1] < summary_id < b_rows[0]

    await env.consolidate(b_rows[-1])

    first, second = env.main.chat_calls
    # First window: no summary, A's rows only (T and B are after it).
    assert first["messages"][0] == {"role": "system", "content": SYSTEM}
    assert first["messages"][1:-1] == alternating(["A one", "A two", "A three"])
    first_text = dump(first["messages"])
    assert "S1 of the first window" not in first_text and "T one" not in first_text
    # Second window: opens with the summary in force, then T and B.
    assert second["messages"][0] == {"role": "system", "content": SYSTEM}
    assert second["messages"][1] == {
        "role": "assistant", "content": SUMMARY_PREFIX + "S1 of the first window"
    }
    assert second["messages"][2:-1] == alternating(
        ["T one", "T two", "B one", "B two"], start="assistant"
    )
    assert "A one" not in dump(second["messages"])
    # Two records tiling the range; the summary row is inside the second.
    memories = await fetch_memories(env.db)
    assert [(m["msg_id_start"], m["msg_id_end"]) for m in memories] == [
        (a_rows[0], a_rows[-1]),
        (t_rows[0], b_rows[-1]),
    ]
    assert memories[1]["msg_id_start"] < summary_id < memories[1]["msg_id_end"]
    assert await fetch_watermark(env.db, env.channel.id) == b_rows[-1]


# ---------------------------------------------------------------------------
# A1: a long unsummarized history is split to the channel's budget
# ---------------------------------------------------------------------------


def row_text(i: int) -> str:
    """Unique, findable content of history row ``i``."""
    return f"ROW{i:02d} {PAD}"


def history(kind: str) -> tuple[list[dict], int | None]:
    """Twelve history messages for the A1 test, and the index of the row the
    watermark sits after (None: before the first row)."""
    messages = alternating([row_text(i) for i in range(12)])
    watermark_after = None
    if kind in ("tool_cycle_at_cut", "watermark_mid_cycle"):
        call = {"id": "c1", "type": "function", "function": {"name": "shell", "arguments": "{}"}}
        # Where the natural overflow point falls (three rows per view), a
        # call and its result; or, mid-cycle, right at the start.
        at = 2 if kind == "tool_cycle_at_cut" else 1
        messages[at] = {
            "role": "assistant", "content": row_text(at), "tool_calls": [call],
        }
        messages[at + 1] = {"role": "tool", "tool_call_id": "c1", "content": row_text(at + 1)}
        if kind == "watermark_mid_cycle":
            watermark_after = at  # between the call and its result
    return messages, watermark_after


@pytest.mark.parametrize(
    "kind", ["plain", "tool_cycle_at_cut", "watermark_mid_cycle", "watermark_mid_cycle_tight"]
)
async def test_long_unsummarized_history_splits_within_budget(env, kind):
    """A1: with no summary and a budget that holds about three rows, the
    range is cut into several views, each within budget as rendered, every
    range row in the request for its own record, older rows trimmed from
    later views, the records tiling the range, and tool cycles kept whole."""
    # Just under four rows of room (a first view's shorter range sentence
    # frees a little more), so a view holds three.
    # watermark_mid_cycle: tool-call rows cost more than plain rows, so give
    # call + result + a dialog row room in the first segment.
    # watermark_mid_cycle_tight: the same history at the plain budget, where
    # the lone tool result becomes a segment with no dialog; its id span must
    # fold into the next record so the records still tile the range.
    tight = kind == "watermark_mid_cycle_tight"
    env.agent.max_tokens = budget = budget_for_rows(
        3.4 if kind == "watermark_mid_cycle" else 2.8
    )
    messages, watermark_after = history("watermark_mid_cycle" if tight else kind)
    ids = await env.rows(messages)
    watermark = 0 if watermark_after is None else ids[watermark_after]
    if watermark_after is not None:
        await env.watermark_at(watermark)

    await env.consolidate(ids[-1])

    calls = env.main.chat_calls
    assert len(calls) >= 3
    for call in calls:
        assert call["tools"] is None
        assert rendered_size(call["messages"]) <= budget
    memories = await fetch_memories(env.db)
    assert len(memories) == len(calls)
    # The records tile (watermark, range_end]: no gap, no overlap.
    assert memories[0]["msg_id_start"] == watermark + 1
    for previous, following in zip(memories, memories[1:]):
        assert following["msg_id_start"] == previous["msg_id_end"] + 1
    assert memories[-1]["msg_id_end"] == ids[-1]
    assert await fetch_watermark(env.db, env.channel.id) == ids[-1]
    # Every range row is in the request for the record that covers it. In the
    # tight case a lone tool result is not dialog (consolidation summarizes
    # dialog only), so only its id span is required to be covered.
    marker = {row_id: f"ROW{i:02d}" for i, row_id in enumerate(ids)}
    is_tool = {row_id: m.get("role") == "tool" for m, row_id in zip(messages, ids)}
    for call, memory in zip(calls, memories):
        text = dump(call["messages"])
        for row_id in range(memory["msg_id_start"], memory["msg_id_end"] + 1):
            if tight and is_tool[row_id]:
                continue
            assert marker[row_id] in text, f"row {row_id} missing from its request"
    # Older context is trimmed from later views, as the live window's would be.
    assert "ROW00" not in dump(calls[1]["messages"])
    if kind == "tool_cycle_at_cut":
        # The call and its result share one request.
        assert any(
            "ROW02" in dump(c["messages"]) and "ROW03" in dump(c["messages"]) for c in calls
        )
    if kind == "watermark_mid_cycle":
        # The result opens the range; its call is context in the same request,
        # and the first record starts just above the watermark and spans it.
        first_text = dump(calls[0]["messages"])
        assert "ROW01" in first_text and "ROW02" in first_text
        assert memories[0]["msg_id_start"] == ids[2] == watermark + 1
        assert memories[0]["msg_id_start"] <= ids[2] <= memories[0]["msg_id_end"]


# ---------------------------------------------------------------------------
# A1 (step 8): rows a view cannot hold are warned about
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["oversized_unit", "dropped_call", "residual_overage"])
async def test_catchup_warns_when_view_loses_rows(env, caplog, kind):
    """A1: a view that cannot hold its own rows still yields a request and a
    record, and a WARNING names the channel, what was lost and the remedy
    (``max_context_tokens``)."""
    env.agent.max_tokens = small = budget_for_rows(0) + 150
    if kind == "oversized_unit":
        # One row alone exceeds the budget.
        ids = await env.rows([user("HUGE " + "word " * 700)])
        needles = (env.channel.id, str(ids[0]), "max_context_tokens")
    elif kind == "dropped_call":
        # One tool cycle alone exceeds the budget; its result alone fits.
        call = {"id": "c1", "type": "function", "function": {"name": "shell", "arguments": "{}"}}
        ids = await env.rows([
            {"role": "assistant", "content": "CALLROW " + "word " * 700, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "c1", "content": "RESULT-OK"},
        ])
        needles = (env.channel.id, str(ids[0]), "max_context_tokens")
    else:
        # The system prompt alone exceeds the budget.
        env.agent.system_prompt = "word " * 600
        env.agent.max_tokens = small = 300
        ids = await env.rows([user("hello")])
        needles = ()

    with caplog.at_level(logging.WARNING, logger="corvidae.memory"):
        await env.consolidate(ids[-1])

    # Still a request and a record.
    [call_made] = env.main.chat_calls
    assert len(await fetch_memories(env.db)) == 1
    if kind == "dropped_call":
        text = dump(call_made["messages"])
        assert "CALLROW" not in text
        # The orphaned result is framed as a system notification.
        assert any(
            m["role"] == "system" and m["content"].endswith("RESULT-OK")
            for m in call_made["messages"][1:]
        )
    if kind == "residual_overage":
        needles = (
            env.channel.id, f"{ids[0]}-{ids[-1]}",
            str(rendered_size(call_made["messages"])), str(small),
        )
    assert warnings_naming(caplog, *needles), (
        f"no WARNING naming {needles}; got "
        f"{[r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]}"
    )


# ---------------------------------------------------------------------------
# O3: a view with tool-call rows and no tools is accepted
# ---------------------------------------------------------------------------


async def test_catchup_request_with_tool_rows_renders_without_tools(env):
    """O3: the view holds an assistant tool_calls row and its result and is
    sent with no tools. The deployed Swift-1.5 chat template (the one the
    suite already renders with) accepts exactly that request."""
    call = {"id": "c1", "type": "function", "function": {"name": "shell", "arguments": "{}"}}
    ids = await env.rows([
        user("please check the file"),
        {"role": "assistant", "content": "checking", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "c1", "content": "file is fine"},
        assistant("the file is fine"),
    ])

    await env.consolidate(ids[-1])

    [made] = env.main.chat_calls
    assert made["tools"] is None
    assert "tool_choice" not in (made["extra_body"] or {})
    assert made["extra_body"] == CONSOLIDATION_EXTRA_BODY
    # The request really carries the tool-call row and its result.
    assert any(m.get("tool_calls") for m in made["messages"])
    assert any(m.get("role") == "tool" for m in made["messages"])
    # The template renders it (raises on a malformed request) and emits the call.
    rendered = render_swift(_as_server_renders(made["messages"]))
    assert "<tool_call>" in rendered


# ---------------------------------------------------------------------------
# Several narrative entries per consolidation (#635)
# ---------------------------------------------------------------------------


async def test_catchup_segment_reply_with_two_entries_stores_two_rows(env):
    """C1: a catch-up segment whose reply has 2 entries stores 2 rows that
    share range, importance, participants and stamp, each with its own text
    and tags."""
    env.main.text = json.dumps({
        "entries": [
            {"narrative": "I diagnosed the wifi dropouts.", "topic_tags": ["wifi"]},
            {"narrative": "I ordered transistors.", "topic_tags": ["parts"]},
        ],
        "participants": ["kestrel"],
        "importance": 0.4,
    })
    env.memory._llm()._clients["embedding"].embed_result = [[1.0, 0, 0, 0], [0, 1.0, 0, 0]]
    rows = await env.rows(alternating(["hello there", "hi, how can I help"]))

    await env.consolidate(rows[-1])

    stored = await fetch_stamped_memories(env.db)
    assert len(stored) == 2
    assert_shared_columns_equal(stored)
    assert (stored[0]["msg_id_start"], stored[0]["msg_id_end"]) == (rows[0], rows[-1])
    assert [r["summary"] for r in stored] == [
        "I diagnosed the wifi dropouts.", "I ordered transistors.",
    ]
    assert [json.loads(r["topic_tags"]) for r in stored] == [["wifi"], ["parts"]]


async def test_catchup_unusable_replies_store_the_raw_transcript(env):
    """C2 (catch-up): the first reply has only blank entries (beside a stray
    summary), the retry has neither entries nor summary; after exactly one
    retry the segment is stored as one row holding the raw transcript."""
    replies = [
        json.dumps({"entries": [{"narrative": " "}], "summary": "A stray summary."}),
        json.dumps({"participants": ["kestrel"]}),
    ]

    async def chat(messages, tools=None, extra_body=None):
        """Answer the next canned reply."""
        env.main.chat_calls.append({"messages": messages})
        return {"choices": [{"message": {"role": "assistant", "content": replies.pop(0)}}]}

    env.main.chat = chat
    rows = await env.rows(alternating(["the weather station keeps dropping", "try WIFI_PS_NONE"]))

    await env.consolidate(rows[-1])

    assert len(env.main.chat_calls) == 2
    [row] = await fetch_memories(env.db)
    assert "the weather station keeps dropping" in row["summary"]
    assert "assistant: try WIFI_PS_NONE" in row["summary"]
    assert "stray summary" not in row["summary"]
