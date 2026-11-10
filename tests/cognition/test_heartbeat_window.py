"""Ordinary compaction alone bounds the heartbeat window (#426).

HeartbeatPlugin no longer edits the window, so each beat extends the previous
one's prompt and the kernel's compaction (with its consolidation fork) is the
only thing that shortens it. A beat is simulated the way HeartbeatPlugin._beat
produces it: an untargeted ``on_notify(source="heartbeat")`` on
``internal:heartbeat``, which the real Agent turns into a
system row. The Agent, persistence and task plugins are real; only the LLM
client is a stub, so there is no network. The prompt each call sends is
recorded at call time.
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from corvidae.channel import Channel
from corvidae.compaction import CompactionPlugin
from corvidae.hooks import hookimpl
from corvidae_cognition.heartbeat import HEARTBEAT_CHANNEL, HeartbeatPlugin
from corvidae_cognition.memory import MemoryPlugin, fetch_range_rows
from helpers import build_plugin_and_channel, drain
from llm_response_fixtures import _make_text_response

SYSTEM_PROMPT = "You are a test assistant."


def beat_rows(i: int, text: str | None = None) -> list[str]:
    """The two contents one simple beat leaves in the window."""
    return [text or f"beat {i}", f"reply {i}"]


def unframe(content: str) -> str:
    """A rendered notification's payload, without its header and fence."""
    return re.sub(
        r"\A\[metadata:[^\n]*\n\[payload \w+ begins\]\n(.*)\n\[payload \w+ ends\]\Z",
        r"\1", content, flags=re.S,
    )


def notification(i: int) -> str:
    """The content of beat i's notification row."""
    return f"beat {i}"


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
        return [unframe(m["content"]) for m in self.prompts[k][1:]]

    async def chat(self, messages, *args, **kwargs) -> dict:
        """Record the prompt, then answer from the script or with 'reply N'."""
        self.prompts.append([dict(m) for m in messages])
        if self.script is not None:
            return self.script.pop(0)
        last = unframe(messages[-1]["content"])
        return _make_text_response(f"reply {last.rsplit(' ', 1)[-1]}")

    async def beat(self, i: int, text: str | None = None, channel=None) -> None:
        """Run beat i through the Agent and wait for its turn to finish."""
        channel = channel or self.hb_ch
        await self.plugin.on_notify(
            channel=channel, source="heartbeat",
            text=text or f"beat {i}", tool_call_id=None,
            annotations={"origin": "heartbeat"},
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
# R1: the heartbeat never edits its own window
# ---------------------------------------------------------------------------


async def test_heartbeat_implements_only_lifecycle_hooks(
    rig_factory: Callable[..., Awaitable[Rig]],
) -> None:
    """R1: HeartbeatPlugin registers no hook that could edit the window; the
    only hooks it implements are its lifecycle ones."""
    rig = await rig_factory()
    names = {caller.name for caller in rig.plugin.pm.get_hookcallers(rig.hb)}
    assert names <= {"on_init", "on_start", "on_stop", "on_compaction"}


async def test_each_beat_extends_the_previous_window(
    rig_factory: Callable[..., Awaitable[Rig]],
) -> None:
    """R1: with no compaction plugin, beat k's prompt starts with beat k-1's
    prompt and reply, the system prompt never changes, and beat 51's prompt
    holds every earlier beat."""
    rig = await rig_factory()
    # Beats 1..51 with no compaction plugin, so nothing may shorten the window.
    for i in range(1, 52):
        await rig.beat(i)
    # Each prompt must start with the previous prompt plus the previous reply.
    for k in range(1, 51):
        previous = rig.contents(k - 1) + [f"reply {k}"]
        assert rig.contents(k)[: len(previous)] == previous
    # The system prompt is the same on every call, so the prefix stays stable.
    assert {m[0]["content"] for m in rig.prompts} == {SYSTEM_PROMPT}
    # The last prompt holds every earlier beat whole, then the new notification.
    assert rig.contents(-1) == (
        sum((beat_rows(i) for i in range(1, 51)), []) + [notification(51)]
    )


# ---------------------------------------------------------------------------
# R1, R2: compaction alone bounds the window and keeps consolidation's range
# ---------------------------------------------------------------------------

MAX_TOKENS = 2000
# About 60 tokens a beat notification, so 51 of them overrun the budget
# several times and compaction has to run repeatedly.
PAD = "pad " * 60
# The text the stubbed fork and view return, distinct from the kernel's
# "[Summary of ..." header so a match proves the stub text reached the window.
STUB_SUMMARY = "stub-fork-summary"


async def padded_rig(
    rig_factory: Callable[..., Awaitable[Rig]],
) -> tuple[Rig, CompactionPlugin]:
    """A rig with a small budget and a real CompactionPlugin whose summary
    comes from a stubbed fork, so no LLM call happens beyond the beats."""
    rig = await rig_factory(
        agent_defaults={
            "system_prompt": SYSTEM_PROMPT,
            "max_context_tokens": MAX_TOKENS,
            "reasoning_history": "drop",
        },
    )
    compaction = CompactionPlugin(rig.plugin.pm)
    compaction._view_summary = AsyncMock(return_value=STUB_SUMMARY)
    compaction._fork_summary = AsyncMock(return_value=STUB_SUMMARY)
    rig.plugin.pm.register(compaction, name="compaction")
    return rig, compaction


async def test_compaction_alone_bounds_the_window(
    rig_factory: Callable[..., Awaitable[Rig]],
) -> None:
    """R1: over 51 padded beats the window stays within max_context_tokens at
    every LLM call, compaction runs, and each compaction keeps the newest
    beat and puts its summary first."""
    rig, compaction = await padded_rig(rig_factory)
    # The window's estimate and the fork count at the moment of every call.
    estimates: list[int] = []
    forks: list[int] = []
    inner = rig.chat

    async def spy(messages: list[dict], *args: Any, **kwargs: Any) -> dict:
        """Snapshot the window estimate and fork count, then answer as the
        rig does."""
        estimates.append(rig.hb_ch.conversation.token_estimate())
        forks.append(compaction._fork_summary.await_count)
        return await inner(messages, *args, **kwargs)

    rig.client.chat.side_effect = spy
    # 51 padded beats, enough to overrun the budget several times.
    for i in range(1, 52):
        await rig.beat(i, text=f"{PAD} {i}")

    # One call per beat, every window within budget, at least one compaction.
    assert len(estimates) == 51
    assert all(e <= MAX_TOKENS for e in estimates), estimates
    assert compaction._fork_summary.await_count >= 1
    for k in range(1, 51):
        # The previous beat's reply survives every compaction.
        assert f"reply {k}" in rig.contents(k)
        # A prompt that follows a compaction starts with its summary.
        if forks[k] > forks[k - 1]:
            assert STUB_SUMMARY in rig.contents(k)[0]


async def test_compaction_range_stays_in_the_window_across_beats(
    rig_factory: Callable[..., Awaitable[Rig]],
) -> None:
    """R2 (#456): at each of at least three compactions, the rows
    consolidation would summarize (those after its watermark, up to the
    compaction's last row) are all in the window before_compaction sees."""
    rig, _compaction = await padded_rig(rig_factory)
    contained: list[Any] = []

    class Recorder:
        """Records whether each compaction's range is in the window, then
        advances the watermark as a consolidation would."""

        def __init__(self) -> None:
            self.watermark = 0

        @hookimpl
        async def before_compaction(
            self, channel: Channel, compacted_ids: list[int], cut: int
        ) -> None:
            """Check the range through the production containment predicate."""
            end = max(compacted_ids)
            rows = await fetch_range_rows(
                rig.db, HEARTBEAT_CHANNEL, self.watermark, end
            )
            contained.append(
                MemoryPlugin._range_in_window(channel.conversation, rows, cut)
            )
            self.watermark = end

    rig.plugin.pm.register(Recorder(), name="recorder")
    # 51 padded beats, enough for several compactions.
    for i in range(1, 52):
        await rig.beat(i, text=f"{PAD} {i}")

    # At least three compactions ran and every range was found in the window.
    assert len(contained) >= 3
    assert all(result is not None for result in contained)


# ---------------------------------------------------------------------------
# Restart: the window reloads every row since the last summary
# ---------------------------------------------------------------------------


async def _count(db, extra: str = "") -> int:
    """Rows in message_log for the heartbeat channel."""
    cur = await db.execute(
        "SELECT COUNT(*) FROM message_log WHERE channel_id=?" + extra,
        (HEARTBEAT_CHANNEL,),
    )
    return (await cur.fetchone())[0]


SUMMARY_TEXT = "[Summary of earlier conversation]\nthe agent reviewed goals"


async def test_restart_reloads_every_row_since_the_summary(tmp_path: Path) -> None:
    """A sessions.db that holds a compaction summary for the heartbeat
    reloads it at window[0] with every row after it; the next beat's prompt
    holds the summary, all ten earlier beats and the new notification, and
    the log keeps every row."""
    db_path = str(tmp_path / "sessions.db")

    first = await make_rig(db_path=db_path)
    try:
        # Ten beats to leave rows in the log.
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

    second = await make_rig(db_path=db_path)
    try:
        await second.beat(11)
        assert len(second.prompts) == 1
        assert second.contents(0) == (
            [SUMMARY_TEXT]
            + sum((beat_rows(i) for i in range(1, 11)), [])
            + [notification(11)]
        )
        # The summary row is still in the log: 20 beat rows, 1 summary, 2 new.
        assert await _count(second.db) == 23
    finally:
        await second.stop()
