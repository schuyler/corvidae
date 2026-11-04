"""The heartbeat window trims on a watermark, not every beat (#286, #426).

The #286 window keeps the heartbeat's prompt from growing without bound;
#426 changed when the trim fires. The keep_beats trim cut at the start of
every beat, so the prompt changed right after the system prompt every time
and a prompt-caching backend re-prefilled the whole retained window on
every beat. The hysteresis trim fires only when the window has grown to
``trim_high_watermark`` of the channel's compaction budget, and then drops
older beats down to ``trim_low_watermark`` of it. Between trims the window
only grows, so each beat's prompt is a superset of the previous beat's and
the cache serves the whole prefix: a beat prefills only its new tail, and
the cold prefill happens once per trim instead of once per beat.

One test per acceptance criterion of #286's approved requirements R1-R6,
re-expressed for the watermark trim, plus the #308 over-budget fallback.
A beat is simulated the way HeartbeatPlugin._beat produces it: an
untargeted ``on_notify(source="heartbeat")`` on ``internal:heartbeat``,
which the real Agent turns into a ``[heartbeat]`` system row. The Agent,
persistence and task plugins are real; only the LLM client is a stub, so
there is no network. The prompt each call sends is recorded at call time.

Sizing: padded beats use ``filler``, distinct words the tokenizer cannot
compress, so a beat's token count is the nominal one.
"""

from __future__ import annotations

import json
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from corvidae.compaction import CompactionPlugin
from corvidae.context import MessageType, count_tokens, message_tokens
from corvidae_cognition.heartbeat import (
    DEFAULT_TRIM_HIGH_WATERMARK,
    DEFAULT_TRIM_LOW_WATERMARK,
    HEARTBEAT_CHANNEL,
    HeartbeatPlugin,
    hysteresis_cut,
)
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


def filler(tokens: int) -> str:
    """A string that counts to at least ``tokens`` raw-text tokens.

    Distinct words, so the tokenizer can't compress a repetition into a
    fraction of the nominal count (repetitive padding like "pad " * N
    counts for a few dozen tokens no matter how long it is).
    """
    parts: list[str] = []
    n = 0
    while True:
        n += 1
        parts.append(f"filler{n:05d}")
        text = " ".join(parts)
        if count_tokens(text) >= tokens:
            return text


def padded(i: int, pad: str) -> str:
    """The content of beat i's notification with a padded body."""
    return f"[heartbeat]\n\n{pad} {i}"


def beat_tokens(text: str) -> int:
    """Tokens of one beat's two rows, notification and reply."""
    return sum(
        message_tokens({"role": "system", "content": c})
        for c in beat_rows(0, text=text)
    )


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
# R1: the window grows freely under the high watermark, bounded above it
# ---------------------------------------------------------------------------


async def test_beats_below_the_high_watermark_are_never_trimmed(rig_factory):
    """R1/#426: while the window is under trim_high_watermark of the budget,
    no beat drops anything: beat 51's prompt holds all 50 earlier beats.
    This is the property the keep_beats trim broke — every beat's prompt
    started over from the system prompt, cold-prefilling the whole window
    on a caching backend."""
    rig = await rig_factory(agent_defaults={
        "system_prompt": SYSTEM_PROMPT,
        "max_context_tokens": 1_000_000,
    })
    for i in range(1, 52):
        await rig.beat(i)
    assert rig.prompts[-1][0]["content"] == SYSTEM_PROMPT
    assert rig.contents(-1) == (
        sum((beat_rows(i) for i in range(1, 51)), []) + [notification(51)]
    )


async def test_the_high_watermark_bounds_the_window(rig_factory):
    """R1: with a tight budget, padded beats grow the window past the high
    watermark, and the trim keeps it in check: the last prompt is a
    contiguous tail of beats — nothing older, nothing missing — and the
    window sits under the high watermark at beat end."""
    max_tokens = 2000
    pad = filler(320)
    bt = beat_tokens(pad)
    # Precondition: 51 padded beats cannot fit under the high watermark (a
    # trim must have fired), and three beats don't fit the low watermark
    # (the kept tail is at most two beats, so the assertion below is about
    # trimming, not about the window never having grown).
    assert count_tokens(SYSTEM_PROMPT) + 51 * bt > (
        DEFAULT_TRIM_HIGH_WATERMARK * max_tokens
    )
    assert 3 * bt > DEFAULT_TRIM_LOW_WATERMARK * max_tokens

    rig = await rig_factory(agent_defaults={
        "system_prompt": SYSTEM_PROMPT,
        "max_context_tokens": max_tokens,
        "reasoning_history": "drop",
    })
    for i in range(1, 52):
        await rig.beat(i, text=f"{pad} {i}")

    window = rig.hb_ch.conversation
    # The trim landed at the low watermark at beat start; the beat added its
    # own two rows. So the window is bounded by low + one beat.
    assert window.token_estimate() <= (
        DEFAULT_TRIM_LOW_WATERMARK * max_tokens + bt
    )
    # A contiguous tail, in order, with beat 51's notification last.
    contents = rig.contents(-1)
    assert contents[-1] == padded(51, pad)
    n = len(contents)
    assert (n - 1) % 2 == 0
    k = 51 - (n - 1) // 2
    assert contents == (
        sum((beat_rows(i, text=f"{pad} {i}") for i in range(k, 51)), [])
        + [padded(51, pad)]
    )
    assert k >= 2  # at least one earlier beat was dropped


async def test_a_lower_high_watermark_trims_sooner(rig_factory):
    """R2: the watermarks bound the window, and a lower high watermark trims
    sooner: with the same beats, fewer beats are retained."""
    max_tokens = 2000
    pad = filler(320)

    async def retained(high: float, low: float) -> int:
        rig = await rig_factory(
            {"heartbeat": {"trim_high_watermark": high,
                           "trim_low_watermark": low}},
            agent_defaults={
                "system_prompt": SYSTEM_PROMPT,
                "max_context_tokens": max_tokens,
                "reasoning_history": "drop",
            },
        )
        for i in range(1, 26):
            await rig.beat(i, text=f"{pad} {i}")
        contents = rig.contents(-1)
        return (len(contents) - 1) // 2  # beats retained before beat 26

    loose = await retained(0.95, 0.6)
    tight = await retained(0.5, 0.3)
    assert loose > tight


# ---------------------------------------------------------------------------
# R3: trimming makes no LLM call
# ---------------------------------------------------------------------------


async def test_trimming_makes_no_llm_call(rig_factory):
    """R3: over 51 beats whose whole history would blow the compaction
    budget, only the 51 beat turns call the LLM: the trim keeps the window
    under budget, so compaction never summarizes."""
    max_tokens = 2000
    budget = 0.8 * max_tokens
    pad = filler(320)
    bt = beat_tokens(pad)
    # Precondition: 50 padded beats do not fit, so today compaction would
    # fire.
    assert 50 * bt > budget
    # Precondition: the trimmed window (low watermark plus the beat's own
    # rows) fits, so a failure below points at the trim and not the padding.
    assert (DEFAULT_TRIM_LOW_WATERMARK * max_tokens + bt) < budget

    rig = await rig_factory(
        agent_defaults={
            "system_prompt": SYSTEM_PROMPT,
            "max_context_tokens": max_tokens,
            "reasoning_history": "drop",
        },
    )
    compaction = CompactionPlugin(rig.plugin.pm)
    compaction._view_summary = AsyncMock(return_value="summary")
    compaction._fork_summary = AsyncMock(return_value="summary")
    rig.plugin.pm.register(compaction, name="compaction")

    for i in range(1, 52):
        await rig.beat(i, text=f"{pad} {i}")

    assert rig.client.chat.await_count == 51
    assert compaction._view_summary.await_count == 0
    assert compaction._fork_summary.await_count == 0


async def test_a_mid_beat_overrun_is_compactions_job(rig_factory):
    """#426: the trim only fires at a beat start. Short beats accumulate
    under the high watermark with no trim at all; then one beat's tool
    result pushes the window past the compaction threshold mid-beat, and
    CompactionPlugin — not the trim — summarizes it."""
    max_tokens = 2000
    threshold = 0.8 * max_tokens
    big = filler(500)
    big_tokens = message_tokens({"role": "tool", "content": big})
    base = count_tokens(SYSTEM_PROMPT) + 40 * beat_tokens("beat 1")
    # Precondition: forty short beats sit under the threshold (no trim or
    # compaction has fired); beat 41 plus the big result crosses it, while
    # the tool call and result fit the 0.5 retention, so compaction's cut
    # lands before the cycle, not inside it.
    assert base < threshold
    assert base + beat_tokens("beat 41") + big_tokens > threshold
    assert 25 + big_tokens <= 0.5 * max_tokens

    rig = await rig_factory(
        agent_defaults={
            "system_prompt": SYSTEM_PROMPT,
            "max_context_tokens": max_tokens,
            "reasoning_history": "drop",
        },
    )
    compaction = CompactionPlugin(rig.plugin.pm)
    compaction._fork_summary = AsyncMock(return_value=None)
    summarized: list[list[dict]] = []

    async def fake_view_summary(channel, conversation, cut):
        summarized.append(list(conversation.messages[:cut]))
        return "summary"

    compaction._view_summary = fake_view_summary
    rig.plugin.pm.register(compaction, name="compaction")

    for i in range(1, 41):
        await rig.beat(i)
    # No trim has fired: all forty beats are still in the window.
    assert len(rig.hb_ch.conversation.messages) == 80

    async def big_tool() -> str:
        """A test tool with a large result."""
        return big

    rig.plugin._tools = {"big": big_tool}
    rig.script = [
        _make_tool_call_response([_make_tool_call("c41", "big", {})]),
        _make_text_response("reply 41"),
    ]
    await rig.beat(41)
    for _ in range(2):
        await drain_task_queue(rig.plugin)
        await drain(rig.plugin, rig.hb_ch)

    # Compaction ran mid-beat, summarizing the old beats (not a trim: the
    # trim was not due — the window was under the high watermark until the
    # tool result landed).
    assert summarized
    assert any(m.get("content") == notification(1) for m in summarized[0])
    # The tool cycle survived intact: the final window carries the call and
    # its result in order, then the reply.
    final = rig.hb_ch.conversation.messages
    pos_c = next(
        i for i, m in enumerate(final)
        if m.get("tool_calls") and m["tool_calls"][0]["id"] == "c41"
    )
    pos_r = next(i for i, m in enumerate(final) if m.get("tool_call_id") == "c41")
    assert pos_c < pos_r
    assert final[-1]["content"] == "reply 41"
    # The window now starts with the summary, not with beat 1.
    assert rig.contents(-1)[0].startswith("[Summary")
    assert notification(1) not in rig.contents(-1)


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
    """R4: with beats trimmed away on a tight budget, message_log still
    gains its two rows per beat, loses none and gains no summary row."""
    max_tokens = 2000
    pad = filler(320)
    rig = await rig_factory(agent_defaults={
        "system_prompt": SYSTEM_PROMPT,
        "max_context_tokens": max_tokens,
        "reasoning_history": "drop",
    })
    counts = []
    for i in range(1, 21):
        await rig.beat(i, text=f"{pad} {i}")
        counts.append(await _count(rig.db))
    assert all(a < b for a, b in zip(counts, counts[1:]))
    assert counts[-1] == 40
    assert await _count(rig.db, " AND message_type='summary'") == 0
    cur = await rig.db.execute(
        "SELECT message FROM message_log WHERE channel_id=? ORDER BY id LIMIT 1",
        (HEARTBEAT_CHANNEL,),
    )
    assert json.loads((await cur.fetchone())[0])["content"] == padded(1, pad)


# ---------------------------------------------------------------------------
# R5: a beat in progress is never cut
# ---------------------------------------------------------------------------


async def test_beat_with_two_tool_calls_is_never_cut(rig_factory):
    """R5: a beat whose turn makes two tool calls keeps its whole exchange in
    every prompt until it ends, even when the window is over the high
    watermark; the old beat goes."""
    max_tokens = 2000
    big = filler(1500)
    small = filler(120)
    rig = await rig_factory(agent_defaults={
        "system_prompt": SYSTEM_PROMPT,
        "max_context_tokens": max_tokens,
        "reasoning_history": "drop",
    })
    # Precondition: at beat 2's start the window is over the high watermark,
    # so the trim is due (and drops beat 1).
    assert (
        count_tokens(SYSTEM_PROMPT)
        + beat_tokens(big)
        + beat_tokens(small)
        >= DEFAULT_TRIM_HIGH_WATERMARK * max_tokens
    )

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
    await rig.beat(1, text=big)
    # Beat 2: notification, then a tool cycle twice, each result a new item.
    await rig.beat(2, text=small)
    for _ in range(2):
        await drain_task_queue(rig.plugin)
        await drain(rig.plugin, rig.hb_ch)

    note1, note2 = f"[heartbeat]\n\n{big}", f"[heartbeat]\n\n{small}"
    assert len(rig.prompts) == 4
    for k in (1, 2, 3):
        assert note1 not in rig.contents(k)
    # Prompt 3 holds the whole of beat 2 so far, in order.
    p3 = rig.prompts[2]
    pos_note = next(i for i, m in enumerate(p3) if m["content"] == note2)
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
    # After the beat: the window holds beat 2 whole, from notification to
    # reply (beat 1 was trimmed at beat 2's start — the window was over the
    # high watermark — and the tool cycle was never cut mid-beat).
    window = rig.hb_ch.conversation.messages
    assert window[0]["content"] == note2
    assert len(window) == 6


async def test_uneven_beats_are_kept_whole(rig_factory):
    """R2/R5: with a tight budget, beat 3's prompt holds all of beat 2 (its
    notification, two tool cycles and reply) and nothing of beat 1: the cut
    snaps to beat boundaries, never inside a beat."""
    max_tokens = 2000
    big = filler(1150)
    small = filler(120)
    rig = await rig_factory(agent_defaults={
        "system_prompt": SYSTEM_PROMPT,
        "max_context_tokens": max_tokens,
        "reasoning_history": "drop",
    })
    # Precondition: at beat 3's start the window is over the high watermark,
    # so the trim is due (and drops beat 1, keeping beat 2 whole).
    assert (
        count_tokens(SYSTEM_PROMPT)
        + beat_tokens(big)
        + 3 * beat_tokens(small)
        >= DEFAULT_TRIM_HIGH_WATERMARK * max_tokens
    )

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
    await rig.beat(1, text=big)
    await rig.beat(2, text=small)
    for _ in range(2):
        await drain_task_queue(rig.plugin)
        await drain(rig.plugin, rig.hb_ch)
    await rig.beat(3, text=small)

    note2, note3 = f"[heartbeat]\n\n{small}", f"[heartbeat]\n\n{small}"
    assert len(rig.prompts) == 5
    # Beat 2's rows: notification, call c1, result c1, call c2, result c2,
    # reply. Tool results carry a random "[Task <id>]" prefix; strip it.
    contents = [
        c.split("] ", 1)[1] if c.startswith("[Task ") else c
        for c in rig.contents(4)
    ]
    assert contents == [
        note2, "", "probed", "", "probed", "reply 2",
        note3,
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


def _estimate(messages: list[dict], system_tokens: int = 10) -> int:
    """token_estimate for a hand-built window: system plus every message."""
    return system_tokens + sum(message_tokens(m) for m in messages)


def _over_hwm_budget(est: int) -> int:
    """A budget whose high watermark (0.8) sits just at or under ``est``."""
    return int(est / 0.8)


@pytest.mark.parametrize("case", ["older_beats", "compacted_mid_beat",
                                  "notification_compacted_away"])
def test_hysteresis_cut_never_trims_on_a_tool_result_tail(case):
    """R5: a window whose tail is a tool result is mid-beat, so the cut is 0
    even when the window is over the high watermark.

    The last two cases are windows CompactionPlugin leaves when it fires
    mid-beat (a long beat's tool results pass the threshold). Without the
    tail gate, "compacted_mid_beat" would drop the fresh summary inside the
    tool cycle, and "notification_compacted_away" (no beat notification left
    at all) would raise IndexError, which makes the whole resolve_compaction
    call fail and skips CompactionPlugin for the rest of that beat.
    """
    windows = {
        # Older beats still in the window.
        "older_beats": [
            _msg("system", notification(1)), _msg("assistant", "reply 1"),
            _msg("system", notification(2)), _call("c1"),
            _msg("tool", "probed", tool_call_id="c1"),
        ],
        # A summary ahead of the current beat's notification.
        "compacted_mid_beat": [
            _summary_row(), _msg("system", notification(9)), _call("c5"),
            _msg("tool", "probed", tool_call_id="c5"),
        ],
        # The summary swallowed the current beat's notification too.
        "notification_compacted_away": [
            _summary_row(), _call("c5"),
            _msg("tool", "probed", tool_call_id="c5"),
        ],
    }[case]
    est = _estimate(windows)
    budget = _over_hwm_budget(est)
    # Over the high watermark: the only reason the cut is 0 is the tail gate.
    assert est >= 0.8 * budget
    assert hysteresis_cut(windows, est, 10, 0.8, 0.5, budget) == 0


@pytest.mark.parametrize("leader", ["summary", "context", "unsent", "tail"])
def test_hysteresis_cut_drops_non_beat_rows_ahead_of_the_first_beat(leader):
    """R1/R2: with the window over the high watermark, the cut lands on a
    beat boundary at or past the first beat, so whatever precedes the first
    beat goes: a summary, a CONTEXT row (goals entry or not-sent marker), an
    UNSENT reply, or the tail of a beat compaction cut open."""
    beat = filler(60)
    lead = {
        "summary": _summary_row(),
        "context": {"role": "system", "content": "Open goals as of now",
                    "_message_type": MessageType.CONTEXT},
        "unsent": {"role": "assistant", "content": "held back",
                   "_message_type": MessageType.UNSENT},
        "tail": _msg("assistant", "reply 0"),
    }[leader]
    window = [
        lead,
        _msg("system", f"[heartbeat]\n\n{beat} 1"),
        _msg("system", f"[heartbeat]\n\n{beat} 2"),
    ]
    est = _estimate(window)
    budget = _over_hwm_budget(est)
    assert est >= 0.8 * budget
    cut = hysteresis_cut(window, est, 10, 0.8, 0.5, budget)
    # A trim fired (the window is over the high watermark) and it lands on a
    # beat boundary, so the leading non-beat row is dropped.
    assert cut > 0
    assert window[cut]["content"].startswith("[heartbeat]")


def test_hysteresis_cut_no_trim_below_the_high_watermark():
    """#426: the cache-preserving case. Five beats, window under the high
    watermark: nothing is dropped, so the next beat's prompt extends this
    one and the cache serves the whole prefix."""
    window = [
        _msg("system", notification(1)), _msg("assistant", "reply 1"),
        _msg("system", notification(2)), _msg("assistant", "reply 2"),
        _msg("system", notification(3)), _msg("assistant", "reply 3"),
    ]
    est = _estimate(window)
    budget = int(est / 0.8) + 100  # high watermark well above the window
    assert hysteresis_cut(window, est, 10, 0.8, 0.5, budget) == 0


def test_hysteresis_cut_lands_at_the_low_watermark_on_a_beat_boundary():
    """#426: over the high watermark, the cut is the oldest beat boundary
    whose tail (system prompt included) fits the low watermark — and the
    previous boundary keeps too much, so the snap is doing work."""
    beat = filler(60)
    window = []
    for i in range(1, 13):
        window += [_msg("system", f"[heartbeat]\n\n{beat} {i}"),
                   _msg("assistant", f"reply {i}")]
    # The trim runs at beat 13's start: the window ends in its notification.
    window.append(_msg("system", f"[heartbeat]\n\n{beat} 13"))
    est = _estimate(window)
    budget = 1000
    low = 0.5 * budget
    assert est >= 0.8 * budget  # over the high watermark
    starts = [i for i, m in enumerate(window) if m["content"].startswith("[heartbeat]")]
    cut = hysteresis_cut(window, est, 10, 0.8, 0.5, budget)
    # The oldest boundary whose tail fits the target.
    expected = next(s for s in starts
                    if 10 + sum(message_tokens(m) for m in window[s:]) <= low)
    assert cut == expected
    # The previous boundary keeps too much (so the cut is not "keep all").
    prev = starts[starts.index(cut) - 1]
    assert 10 + sum(message_tokens(m) for m in window[prev:]) > low
    assert cut > 0


def test_hysteresis_cut_keeps_only_the_current_beat_when_nothing_fits():
    """#426: when even the current beat's notification exceeds the low
    watermark, the cut is the current beat's start — the beat can't be
    trimmed mid-flight (compaction is the over-budget net)."""
    big = filler(900)
    small = filler(20)
    window = [
        _msg("system", f"[heartbeat]\n\n{small} 1"), _msg("assistant", "reply 1"),
        _msg("system", f"[heartbeat]\n\n{big}"),
    ]
    est = _estimate(window)
    budget = 1000
    # The current notification alone is over the low watermark, and the
    # window is over the high watermark.
    assert message_tokens(window[-1]) > 0.5 * budget
    assert est >= 0.8 * budget
    assert hysteresis_cut(window, est, 10, 0.8, 0.5, budget) == 2


def test_hysteresis_cut_skips_a_straddling_cycle_with_warning(caplog):
    """R5: a cut that would separate a tool call from its result is skipped
    (returns 0) with a WARNING naming the channel and why older beats stay."""
    # Beat 1 calls c1; the result only arrives after beat 2's notification.
    # A cut at beat 2's notification (index 2) would land between the c1 call
    # (index 1) and its result (index 3).
    window = [
        _msg("system", notification(1)), _call("c1"),
        _msg("system", notification(2)),
        _msg("tool", "probed", tool_call_id="c1"),
        _msg("assistant", "reply 2"),
        _msg("system", notification(3)),
    ]
    est = _estimate(window)
    budget = _over_hwm_budget(est)
    assert est >= 0.8 * budget
    with caplog.at_level(logging.WARNING):
        assert hysteresis_cut(window, est, 10, 0.8, 0.5, budget) == 0
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
    respects the watermarks: with a tight budget it holds only a contiguous
    tail of the beats that fit, and the log keeps all 22 rows."""
    db_path = str(tmp_path / "sessions.db")
    max_tokens = 2000
    pad = filler(320)
    bt = beat_tokens(pad)
    # Precondition: ten padded beats cannot fit under the high watermark, so
    # the window was trimmed before the restart and is again after it.
    assert count_tokens(SYSTEM_PROMPT) + 10 * bt > (
        DEFAULT_TRIM_HIGH_WATERMARK * max_tokens
    )

    agent_defaults = {
        "system_prompt": SYSTEM_PROMPT,
        "max_context_tokens": max_tokens,
        "reasoning_history": "drop",
    }

    first = await make_rig(None, agent_defaults=agent_defaults, db_path=db_path)
    try:
        for i in range(1, 11):
            await first.beat(i, text=f"{pad} {i}")
    finally:
        await first.stop()

    # A new process: fresh graph on the same DB, window reloaded from rows.
    second = await make_rig(None, agent_defaults=agent_defaults, db_path=db_path)
    try:
        await second.beat(11, text=f"{pad} 11")
        assert len(second.prompts) == 1
        contents = second.contents(0)
        assert contents[-1] == padded(11, pad)
        n = len(contents)
        assert (n - 1) % 2 == 0
        k = 11 - (n - 1) // 2
        assert contents == (
            sum((beat_rows(i, text=f"{pad} {i}") for i in range(k, 11)), [])
            + [padded(11, pad)]
        )
        # Bounded by the low watermark plus the beat's own rows.
        assert second.hb_ch.conversation.token_estimate() <= (
            DEFAULT_TRIM_LOW_WATERMARK * max_tokens + bt
        )
        assert await _count(second.db) == 22
    finally:
        await second.stop()


async def test_restart_with_a_compaction_summary_in_the_log(tmp_path):
    """R6: a sessions.db that already holds a compaction summary for the
    heartbeat (the upgrade path: compaction fires there today) reloads it at
    window[0]; the next beat's trim drops it, whatever it is."""
    db_path = str(tmp_path / "sessions.db")
    max_tokens = 2000
    pad = filler(320)
    agent_defaults = {
        "system_prompt": SYSTEM_PROMPT,
        "max_context_tokens": max_tokens,
        "reasoning_history": "drop",
    }

    first = await make_rig(None, agent_defaults=agent_defaults, db_path=db_path)
    try:
        for i in range(1, 11):
            await first.beat(i, text=f"{pad} {i}")
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

    second = await make_rig(None, agent_defaults=agent_defaults, db_path=db_path)
    try:
        await second.beat(11, text=f"{pad} 11")
        assert len(second.prompts) == 1
        # The summary row is gone from the prompt (trimmed ahead of the
        # oldest kept beat); the log keeps it.
        assert SUMMARY_TEXT not in second.contents(0)
        assert await _count(second.db) == 23
    finally:
        await second.stop()


# ---------------------------------------------------------------------------
# Scope: only internal:heartbeat is trimmed
# ---------------------------------------------------------------------------


async def test_other_channels_are_never_trimmed(rig_factory):
    """Scope: heartbeat-framed notifications on an ordinary channel keep
    their whole history, whatever the watermarks."""
    rig = await rig_factory(
        {"heartbeat": {"trim_high_watermark": 0.1, "trim_low_watermark": 0.05}},
        agent_defaults={
            "system_prompt": SYSTEM_PROMPT,
            "max_context_tokens": 2000,
            "reasoning_history": "drop",
        },
    )
    other = rig.plugin._registry.get_or_create("test", "scope1")
    pad = filler(100)
    for i in range(1, 6):
        await rig.beat(i, text=f"{pad} {i}", channel=other)
    expected = (
        sum((beat_rows(i, text=f"{pad} {i}") for i in range(1, 5)), [])
        + [padded(5, pad)]
    )
    assert rig.contents(-1) == expected


# ---------------------------------------------------------------------------
# Guard: a pending tool call blocks the trim
# ---------------------------------------------------------------------------


async def test_pending_tool_call_skips_the_trim_with_warning(rig_factory, caplog):
    """R5: with a tool call pending on the heartbeat channel, the trim is
    skipped and a WARNING names the channel and the reason."""
    max_tokens = 2000
    pad = filler(900)
    rig = await rig_factory(agent_defaults={
        "system_prompt": SYSTEM_PROMPT,
        "max_context_tokens": max_tokens,
        "reasoning_history": "drop",
    })
    # Precondition: at beat 2's start (beat 1 plus beat 2's notification) the
    # window is over the high watermark, so the trim is due.
    assert (
        count_tokens(SYSTEM_PROMPT)
        + beat_tokens(pad)
        + message_tokens({"role": "system", "content": padded(2, pad)})
        >= DEFAULT_TRIM_HIGH_WATERMARK * max_tokens
    )
    await rig.beat(1, text=f"{pad} 1")
    rig.hb_ch.pending_tool_call_ids = frozenset({"c9"})
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        await rig.beat(2, text=f"{pad} 2")
    # The pending call blocked the trim: the window is still over the high
    # watermark, and beat 1 is still in it.
    assert rig.hb_ch.conversation.token_estimate() >= (
        DEFAULT_TRIM_HIGH_WATERMARK * max_tokens
    )
    assert padded(1, pad) in rig.contents(-1)
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(
        HEARTBEAT_CHANNEL in m and "tool calls are pending" in m for m in messages
    )


# ---------------------------------------------------------------------------
# #308: the over-budget fallback covers a window the trim skipped
# ---------------------------------------------------------------------------


async def test_skipped_trim_over_budget_is_brought_within_budget(rig_factory):
    """#308 (design test 17): with tool calls pending the trim is skipped,
    and no CompactionPlugin is registered, so only the kernel's fallback can
    bring the grown window within max_context_tokens before each beat's LLM
    call."""
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
