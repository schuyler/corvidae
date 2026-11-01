"""The heartbeat window keeps only its last N beats (#286).

One test per acceptance criterion of #286's approved requirements R1-R6. A
beat is simulated the way HeartbeatPlugin._beat produces it: an untargeted
``on_notify(source="heartbeat")`` on ``internal:heartbeat``, which the real
Agent turns into a ``[heartbeat]`` system row. The Agent, persistence and
task plugins are real; only the LLM client is a stub, so there is no
network. The prompt each call sends is recorded at call time.
"""

from __future__ import annotations

import json
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from corvidae.compaction import CompactionPlugin
from corvidae.context import MessageType, count_tokens, message_tokens
from corvidae_cognition.heartbeat import HEARTBEAT_CHANNEL, HeartbeatPlugin
from helpers import build_plugin_and_channel, drain, drain_task_queue
from llm_response_fixtures import (
    _make_text_response,
    _make_tool_call,
    _make_tool_call_response,
)

SYSTEM_PROMPT = "You are a test assistant."


def beat_rows(i: int, text: str | None = None) -> list[str]:
    """The two contents one simple beat leaves in the window."""
    return [f"[heartbeat]\n\n{text or f'beat {i}'}", f"reply {i}"]


def notification(i: int) -> str:
    """The content of beat i's notification row."""
    return f"[heartbeat]\n\nbeat {i}"


class Rig:
    """A real Agent with a HeartbeatPlugin on an LLM stub that records prompts."""

    def __init__(self, plugin, db, hb) -> None:
        self.plugin = plugin
        self.db = db
        self.hb = hb
        # Full message dicts of every prompt sent, copied at call time.
        self.prompts: list[list[dict]] = []
        self.script: list[dict] | None = None
        self.hb_ch = plugin._registry.get_or_create("internal", "heartbeat")

    def contents(self, k: int) -> list:
        """Contents of prompt k without its system prompt."""
        return [m["content"] for m in self.prompts[k][1:]]

    async def chat(self, messages, *args, **kwargs) -> dict:
        """Record the prompt, then answer from the script or with 'reply N'."""
        self.prompts.append([dict(m) for m in messages])
        if self.script is not None:
            return self.script.pop(0)
        last = messages[-1]["content"]
        return _make_text_response(f"reply {last.rsplit(' ', 1)[-1]}")

    async def beat(self, i: int, text: str | None = None, channel=None) -> None:
        """Run beat i through the Agent and wait for its turn to finish."""
        channel = channel or self.hb_ch
        await self.plugin.on_notify(
            channel=channel, source="heartbeat",
            text=text or f"beat {i}", tool_call_id=None,
            meta={"origin": "heartbeat"},
        )
        await drain(self.plugin, channel)

    async def stop(self) -> None:
        """Tear down the task plugin and the DB."""
        task_plugin = self.plugin.pm.get_plugin("task")
        if task_plugin:
            await task_plugin.on_stop()
        await self.db.close()


async def make_rig(config: dict | None = None, *, agent_defaults=None,
                   db_path: str = ":memory:") -> Rig:
    """Build the Agent graph, register a heartbeat (asleep) and stub the LLM."""
    plugin, _channel, db = await build_plugin_and_channel(
        agent_defaults=agent_defaults, db_path=db_path
    )
    hb = HeartbeatPlugin(plugin.pm)
    plugin.pm.register(hb, name="heartbeat")
    # No heartbeat.after, so the heartbeat is asleep and needs no memory.
    await hb.on_init(pm=plugin.pm, config=config or {})
    rig = Rig(plugin, db, hb)
    client = MagicMock()
    client.chat = AsyncMock(side_effect=rig.chat)
    plugin._client = client
    plugin._mark_started()
    rig.client = client
    return rig


@pytest.fixture
async def rig_factory():
    """Factory for rigs, torn down after the test."""
    rigs: list[Rig] = []

    async def make(*args, **kwargs) -> Rig:
        rig = await make_rig(*args, **kwargs)
        rigs.append(rig)
        return rig

    yield make
    for rig in rigs:
        await rig.stop()


# ---------------------------------------------------------------------------
# R1: a beat's prompt doesn't grow with the number of earlier beats
# ---------------------------------------------------------------------------


async def test_beat_51_prompt_is_system_last_three_beats_and_notification(rig_factory):
    """R1: with the default keep_beats of 3, beat 51's prompt is the system
    prompt, beats 48-50's exchanges and the new notification, nothing older."""
    rig = await rig_factory()
    for i in range(1, 52):
        await rig.beat(i)
    assert rig.prompts[-1][0]["content"] == SYSTEM_PROMPT
    assert rig.contents(-1) == (
        beat_rows(48) + beat_rows(49) + beat_rows(50) + [notification(51)]
    )


# ---------------------------------------------------------------------------
# R2: the agent still sees its most recent beats
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("keep, kept", [(2, [49, 50]), (0, [])])
async def test_keep_beats_bounds_the_window(rig_factory, keep, kept):
    """R2: heartbeat.keep_beats earlier beats stay in the window; 0 starts
    every beat empty."""
    rig = await rig_factory({"heartbeat": {"keep_beats": keep}})
    for i in range(1, 52):
        await rig.beat(i)
    expected = sum((beat_rows(i) for i in kept), []) + [notification(51)]
    assert rig.contents(-1) == expected


# ---------------------------------------------------------------------------
# R3: dropping old beats makes no LLM call
# ---------------------------------------------------------------------------


async def test_trimming_makes_no_llm_call(rig_factory):
    """R3: over 51 beats whose whole history would blow the compaction
    budget, only the 51 beat turns call the LLM: the trim keeps the window
    under budget, so compaction never summarizes."""
    max_tokens = 2000
    budget = 0.8 * max_tokens
    pad = "pad " * 60
    # Precondition: 50 padded beats do not fit, so today compaction would fire.
    assert 50 * count_tokens(pad) > budget
    # Precondition: the kept window, (keep_beats + 1) beats, does fit, so a
    # failure below points at the trim and not at the padding.
    kept_beats = 3 + 1
    kept_tokens = count_tokens(SYSTEM_PROMPT) + kept_beats * sum(
        message_tokens({"role": "system", "content": c})
        for c in beat_rows(0, text=pad)
    )
    assert kept_tokens < budget

    rig = await rig_factory(
        agent_defaults={
            "system_prompt": SYSTEM_PROMPT,
            "max_context_tokens": max_tokens,
            "reasoning_history": "drop",
        },
    )
    compaction = CompactionPlugin(rig.plugin.pm)
    compaction._summarize = AsyncMock(return_value="summary")
    compaction._fork_summary = AsyncMock(return_value="summary")
    rig.plugin.pm.register(compaction, name="compaction")

    for i in range(1, 52):
        await rig.beat(i, text=f"{pad} {i}")

    assert rig.client.chat.await_count == 51
    assert compaction._summarize.await_count == 0
    assert compaction._fork_summary.await_count == 0


async def test_compaction_still_runs_when_the_kept_beats_exceed_the_budget(rig_factory):
    """PRD §3.3.2: the trim returns None, so CompactionPlugin still runs after
    it as the over-budget safety net. Beats 1-4 are short and beats 5-8 long;
    three long notifications fit under the threshold and four don't. So beat
    8 is the first window over budget, and only after the trim has dropped
    beats 1-4: compaction summarizes the kept beats, never the trimmed ones."""
    max_tokens = 2000
    budget = 0.8 * max_tokens
    pad = "pad " * 450

    def beat_tokens(text: str) -> int:
        """Tokens of one beat's two rows, notification and reply."""
        return sum(
            message_tokens({"role": "system", "content": c})
            for c in beat_rows(0, text=text)
        )

    # Precondition: beat 7's trimmed window (short beat 4, long beats 5-6 and
    # beat 7's notification) fits, so compaction can't fire before beat 8.
    assert count_tokens(SYSTEM_PROMPT) + beat_tokens("beat 4") + 3 * beat_tokens(pad) < budget
    # Precondition: beat 8's trimmed window (long beats 5-7 and beat 8's
    # notification) alone passes the threshold.
    assert 4 * message_tokens({"role": "system", "content": f"[heartbeat]\n\n{pad}"}) > budget

    rig = await rig_factory(
        agent_defaults={
            "system_prompt": SYSTEM_PROMPT,
            "max_context_tokens": max_tokens,
            "reasoning_history": "drop",
        },
    )
    compaction = CompactionPlugin(rig.plugin.pm)
    # No fork, so the standalone summarizer receives the rows it replaces.
    compaction._fork_summary = AsyncMock(return_value=None)
    compaction._summarize = AsyncMock(return_value="summary")
    rig.plugin.pm.register(compaction, name="compaction")

    # Four short beats, then four long ones.
    for i in range(1, 5):
        await rig.beat(i)
    for i in range(5, 9):
        await rig.beat(i, text=f"{pad} {i}")

    # Compaction ran once, at beat 8, on the trimmed window.
    assert compaction._summarize.await_count == 1
    summarized = [m["content"] for m in compaction._summarize.await_args.args[0]]
    assert summarized and all(
        c.startswith(f"[heartbeat]\n\n{pad}") or c in {"reply 5", "reply 6", "reply 7"}
        for c in summarized
    )
    assert f"[heartbeat]\n\n{pad} 5" in summarized
    # Beat 8's prompt starts with the summary, not with any short beat.
    assert not any(c in rig.contents(-1) for c in beat_rows(4))
    assert "summary" in rig.contents(-1)[0]


# ---------------------------------------------------------------------------
# R4: nothing is deleted from the log
# ---------------------------------------------------------------------------


async def _count(db, extra: str = "") -> int:
    """Rows in message_log for the heartbeat channel."""
    cur = await db.execute(
        "SELECT COUNT(*) FROM message_log WHERE channel_id=?" + extra,
        (HEARTBEAT_CHANNEL,),
    )
    return (await cur.fetchone())[0]


async def test_message_log_only_grows(rig_factory):
    """R4: with every beat trimmed away (keep_beats 0), message_log still
    gains its two rows per beat, loses none and gains no summary row."""
    rig = await rig_factory({"heartbeat": {"keep_beats": 0}})
    counts = []
    for i in range(1, 21):
        await rig.beat(i)
        counts.append(await _count(rig.db))
    assert all(a < b for a, b in zip(counts, counts[1:]))
    assert counts[-1] == 40
    assert await _count(rig.db, " AND message_type='summary'") == 0
    cur = await rig.db.execute(
        "SELECT message FROM message_log WHERE channel_id=? ORDER BY id LIMIT 1",
        (HEARTBEAT_CHANNEL,),
    )
    assert json.loads((await cur.fetchone())[0])["content"] == notification(1)


# ---------------------------------------------------------------------------
# R5: a beat in progress is never cut
# ---------------------------------------------------------------------------


async def test_beat_with_two_tool_calls_is_never_cut(rig_factory):
    """R5: a beat whose turn makes two tool calls keeps its whole exchange in
    every prompt until it ends, even with keep_beats 0; the old beat goes."""
    rig = await rig_factory({"heartbeat": {"keep_beats": 0}})

    async def probe() -> str:
        """A test tool."""
        return "probed"

    rig.plugin._tools = {"probe": probe}
    rig.script = [
        _make_text_response("reply 1"),
        _make_tool_call_response([_make_tool_call("c1", "probe", {})]),
        _make_tool_call_response([_make_tool_call("c2", "probe", {})]),
        _make_text_response("reply 2"),
    ]
    await rig.beat(1)
    # Beat 2: notification, then a tool cycle twice, each result a new item.
    await rig.beat(2)
    for _ in range(2):
        await drain_task_queue(rig.plugin)
        await drain(rig.plugin, rig.hb_ch)

    assert len(rig.prompts) == 4
    for k in (1, 2, 3):
        assert "[heartbeat]\n\nbeat 1" not in rig.contents(k)
    # Prompt 3 holds the whole of beat 2 so far, in order.
    p3 = rig.prompts[2]
    pos_note = next(i for i, m in enumerate(p3) if m["content"] == notification(2))
    pos_c1 = next(
        i for i, m in enumerate(p3)
        if m.get("tool_calls") and m["tool_calls"][0]["id"] == "c1"
    )
    pos_r1 = next(i for i, m in enumerate(p3) if m.get("tool_call_id") == "c1")
    assert pos_note < pos_c1 < pos_r1
    # Prompt 4 also carries the second call and its result.
    p4 = rig.prompts[3]
    assert any(m.get("tool_calls") and m["tool_calls"][0]["id"] == "c2" for m in p4)
    assert any(m.get("tool_call_id") == "c2" for m in p4)
    # After the beat: exactly beat 2, from its notification to its reply.
    window = rig.hb_ch.conversation.messages
    assert window[0]["content"] == notification(2)
    assert len(window) == 6


async def test_uneven_beats_are_kept_whole(rig_factory):
    """R2/R5: with keep_beats 1, beat 3's prompt holds all of beat 2 (its
    notification, two tool cycles and reply) and nothing of beat 1: beats are
    counted as beats, not as a fixed number of messages."""
    rig = await rig_factory({"heartbeat": {"keep_beats": 1}})

    async def probe() -> str:
        """A test tool."""
        return "probed"

    rig.plugin._tools = {"probe": probe}
    rig.script = [
        _make_text_response("reply 1"),
        _make_tool_call_response([_make_tool_call("c1", "probe", {})]),
        _make_tool_call_response([_make_tool_call("c2", "probe", {})]),
        _make_text_response("reply 2"),
        _make_text_response("reply 3"),
    ]
    await rig.beat(1)
    await rig.beat(2)
    for _ in range(2):
        await drain_task_queue(rig.plugin)
        await drain(rig.plugin, rig.hb_ch)
    await rig.beat(3)

    assert len(rig.prompts) == 5
    # Beat 2's rows: notification, call c1, result c1, call c2, result c2, reply.
    # Tool results carry a random "[Task <id>]" prefix; strip it to compare.
    contents = [
        c.split("] ", 1)[1] if c.startswith("[Task ") else c
        for c in rig.contents(4)
    ]
    assert contents == [
        notification(2), "", "probed", "", "probed", "reply 2", notification(3),
    ]
    ids = [
        m["tool_calls"][0]["id"] if m.get("tool_calls") else m.get("tool_call_id")
        for m in rig.prompts[4][1:]
    ]
    assert ids == [None, "c1", "c1", "c2", "c2", None, None]


def _msg(role: str, content, **extra) -> dict:
    """A window message tagged as the ContextWindow tags it."""
    return {"role": role, "content": content,
            "_message_type": MessageType.MESSAGE, **extra}


def _call(call_id: str) -> dict:
    """An assistant message that calls one tool."""
    return _msg("assistant", "", tool_calls=[
        {"id": call_id, "type": "function",
         "function": {"name": "probe", "arguments": "{}"}}
    ])


SUMMARY_TEXT = "[Summary of earlier conversation]\nthe agent reviewed goals"


def _summary_row() -> dict:
    """A compaction summary as the kernel stores it (role user, SUMMARY tag)."""
    return {"role": "user", "content": SUMMARY_TEXT,
            "_message_type": MessageType.SUMMARY}


@pytest.mark.parametrize("case", ["older_beats", "compacted_mid_beat",
                                  "notification_compacted_away"])
def test_beat_cut_never_trims_on_a_tool_result_tail(case):
    """R5: a window whose tail is a tool result is mid-beat, so the cut is 0.

    The last two cases are windows CompactionPlugin leaves when it fires
    mid-beat (a long beat's tool results pass the threshold). Without the
    tail gate, "compacted_mid_beat" would drop the fresh summary inside the
    tool cycle, and "notification_compacted_away" (no beat notification left
    at all) would raise IndexError, which makes the whole resolve_compaction
    call fail and skips CompactionPlugin for the rest of that beat.
    """
    # beat_cut is new in #286: imported here so the other tests collect.
    from corvidae_cognition.heartbeat import beat_cut

    # Each window ends in the result of a tool call the current beat made.
    window, keep = {
        # Older beats still in the window, keep_beats 0.
        "older_beats": ([
            _msg("system", notification(1)), _msg("assistant", "reply 1"),
            _msg("system", notification(2)), _call("c1"),
            _msg("tool", "probed", tool_call_id="c1"),
        ], 0),
        # A summary ahead of the current beat's notification.
        "compacted_mid_beat": ([
            _summary_row(), _msg("system", notification(9)), _call("c5"),
            _msg("tool", "probed", tool_call_id="c5"),
        ], 3),
        # The summary swallowed the current beat's notification too.
        "notification_compacted_away": ([
            _summary_row(), _call("c5"),
            _msg("tool", "probed", tool_call_id="c5"),
        ], 3),
    }[case]
    assert beat_cut(window, keep) == 0


@pytest.mark.parametrize("leader", ["summary", "context", "unsent", "tail"])
def test_beat_cut_drops_non_beat_rows_ahead_of_the_first_beat(leader):
    """R1/R2: with fewer beats than keep_beats, whatever precedes the first
    beat still goes: a summary, a CONTEXT row (goals entry or not-sent
    marker), an UNSENT reply, or the tail of a beat compaction cut open. Each
    is a row kind that can lead a compacted heartbeat window on reload."""
    from corvidae_cognition.heartbeat import beat_cut

    lead = {
        "summary": _summary_row(),
        "context": {"role": "system", "content": "Open goals as of now",
                    "_message_type": MessageType.CONTEXT},
        "unsent": {"role": "assistant", "content": "held back",
                   "_message_type": MessageType.UNSENT},
        "tail": _msg("assistant", "reply 0"),
    }[leader]
    window = [lead, _msg("system", notification(1)), _msg("system", notification(2))]
    assert beat_cut(window, 3) == 1



def test_beat_cut_skips_a_straddling_cycle_with_warning(caplog):
    """R5: a cut that would separate a tool call from its result is skipped
    (returns 0) with a WARNING naming the channel and why older beats stay."""
    from corvidae_cognition.heartbeat import beat_cut

    # Beat 1 calls c1; the result only arrives after beat 2's notification.
    # With keep_beats=1 the cut falls at beat 2's notification (index 2),
    # between the c1 call (index 1) and its result (index 3).
    window = [
        _msg("system", notification(1)), _call("c1"),
        _msg("system", notification(2)),
        _msg("tool", "probed", tool_call_id="c1"),
        _msg("assistant", "reply 2"),
        _msg("system", notification(3)),
    ]
    with caplog.at_level(logging.WARNING):
        assert beat_cut(window, 1) == 0
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert HEARTBEAT_CHANNEL in message
    assert "older beats kept this turn" in message


# ---------------------------------------------------------------------------
# R6: the bound survives a restart
# ---------------------------------------------------------------------------


async def test_bound_survives_restart(tmp_path):
    """R6: after a restart reloads every row, the next beat's prompt still
    holds only the last keep_beats beats, and the log keeps all 22 rows."""
    db_path = str(tmp_path / "sessions.db")
    config = {"heartbeat": {"keep_beats": 2}}

    first = await make_rig(config, db_path=db_path)
    try:
        for i in range(1, 11):
            await first.beat(i)
    finally:
        await first.stop()

    # A new process: fresh graph on the same DB, window reloaded from rows.
    second = await make_rig(config, db_path=db_path)
    try:
        await second.beat(11)
        assert len(second.prompts) == 1
        assert second.contents(0) == (
            beat_rows(9) + beat_rows(10) + [notification(11)]
        )
        assert await _count(second.db) == 22
    finally:
        await second.stop()


async def test_restart_with_a_compaction_summary_in_the_log(tmp_path):
    """R6: a sessions.db that already holds a compaction summary for the
    heartbeat (the upgrade path: compaction fires there today) reloads it at
    window[0]; the next beat's prompt drops it and holds only the kept beats."""
    db_path = str(tmp_path / "sessions.db")
    config = {"heartbeat": {"keep_beats": 2}}

    first = await make_rig(config, db_path=db_path)
    try:
        for i in range(1, 11):
            await first.beat(i)
        # Seed the summary as on_compaction writes it: stamped just before
        # the oldest retained row, so a reload keeps every row after it.
        cur = await first.db.execute(
            "SELECT MIN(timestamp) FROM message_log WHERE channel_id=?",
            (HEARTBEAT_CHANNEL,),
        )
        oldest = (await cur.fetchone())[0]
        await first.db.execute(
            "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
            "VALUES (?, ?, ?, 'summary')",
            (HEARTBEAT_CHANNEL,
             json.dumps({"role": "user", "content": SUMMARY_TEXT}), oldest - 1),
        )
        await first.db.commit()
    finally:
        await first.stop()

    second = await make_rig(config, db_path=db_path)
    try:
        await second.beat(11)
        assert len(second.prompts) == 1
        assert second.contents(0) == (
            beat_rows(9) + beat_rows(10) + [notification(11)]
        )
        # The summary row is still in the log: 20 beat rows, 1 summary, 2 new.
        assert await _count(second.db) == 23
    finally:
        await second.stop()


# ---------------------------------------------------------------------------
# Scope: only internal:heartbeat is trimmed
# ---------------------------------------------------------------------------


async def test_other_channels_are_never_trimmed(rig_factory):
    """Scope: heartbeat-framed notifications on an ordinary channel keep
    their whole history, even with keep_beats 0."""
    rig = await rig_factory({"heartbeat": {"keep_beats": 0}})
    other = rig.plugin._registry.get_or_create("test", "scope1")
    for i in range(1, 6):
        await rig.beat(i, channel=other)
    expected = sum((beat_rows(i) for i in range(1, 5)), []) + [notification(5)]
    assert rig.contents(-1) == expected


# ---------------------------------------------------------------------------
# Guard: a pending tool call blocks the trim
# ---------------------------------------------------------------------------


async def test_pending_tool_call_skips_the_trim_with_warning(rig_factory, caplog):
    """R5: with a tool call pending on the heartbeat channel, the trim is
    skipped and a WARNING names the channel and the reason."""
    rig = await rig_factory({"heartbeat": {"keep_beats": 0}})
    for i in range(1, 3):
        await rig.beat(i)
    # Simulate an in-flight call, then start another beat.
    rig.hb_ch.pending_tool_call_ids = frozenset({"c9"})
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        await rig.plugin.on_notify(
            channel=rig.hb_ch, source="heartbeat", text="beat 3",
            tool_call_id=None, meta={"origin": "heartbeat"},
        )
        await drain(rig.plugin, rig.hb_ch)
    # Nothing was cut, even with keep_beats 0.
    # (beat 2 would have been dropped by the trim, as beat 1 was before it).
    assert rig.contents(-1) == beat_rows(2) + [notification(3)]
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(
        HEARTBEAT_CHANNEL in m and "tool calls are pending" in m for m in messages
    )


# ---------------------------------------------------------------------------
# #308: the over-budget fallback covers a window the trim skipped
# ---------------------------------------------------------------------------


async def test_skipped_trim_over_budget_is_brought_within_budget(rig_factory):
    """#308 (design test 17): with tool calls pending the trim is skipped, and
    no CompactionPlugin is registered, so only the kernel's fallback can bring
    the grown window within max_context_tokens before each beat's LLM call."""
    budget = 300
    rig = await rig_factory(agent_defaults={
        "system_prompt": SYSTEM_PROMPT,
        "max_context_tokens": budget,
        "reasoning_history": "drop",
    })
    # Record the window's estimate at the moment of every LLM call.
    estimates: list[int] = []
    inner = rig.chat

    async def spy(messages, *args, **kwargs) -> dict:
        """Snapshot conv.token_estimate(), then answer as the rig does."""
        estimates.append(rig.hb_ch.conversation.token_estimate())
        return await inner(messages, *args, **kwargs)

    rig.client.chat.side_effect = spy
    # One beat to initialise, then an in-flight call that blocks the trim.
    await rig.beat(0)
    rig.hb_ch.pending_tool_call_ids = frozenset({"c9"})
    # Beats of about 60 tokens each: eight of them overrun 300 tokens.
    big = " ".join(["word"] * 60)
    for i in range(1, 9):
        await rig.beat(i, text=big)
    assert len(estimates) == 9
    assert all(e <= budget for e in estimates), estimates
