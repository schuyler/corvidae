"""Agent-level tests for the over-budget fallback (#308, PRD §3.3.2).

A real Agent, persistence and CompactionPlugin; only the LLM turn is stubbed.
The window is measured inside the stubbed run_agent_turn (the moment of the
main-turn LLM call), via ``conv.token_estimate()``: that is R1's acceptance
criterion, and summing the built prompt would add wrapper tokens the estimate
does not count. Tests 8-16 of the approved design.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from corvidae.compaction import CompactionPlugin
from corvidae.context import MessageType, count_tokens, message_tokens
from corvidae.hooks import resolve_single_result
from corvidae.trace import mint_trace_id
from corvidae.turn import AgentTurnResult
from helpers import build_plugin_and_channel, drain
from llm_response_fixtures import _make_text_response

pytestmark = pytest.mark.timeout(15)

SYSTEM_PROMPT = "You are a test assistant."
MAX_TOKENS = 600


def words(n: int) -> str:
    """Text of about n tokens (one 'word' per token)."""
    return " ".join(["word"] * n)


@dataclass
class Rig:
    """A real Agent on a stubbed turn, with a snapshot of the window taken
    at the moment of the main-turn LLM call."""

    plugin: object
    channel: object
    db: object
    compaction: CompactionPlugin | None = None
    seen: dict = field(default_factory=dict)

    async def seed(self, sizes: list[int]) -> list[int]:
        """Persist and window one row per entry of ``sizes`` (tokens each),
        alternating user/assistant, after the conversation is initialised.
        Returns the message_log ids."""
        conv = self.channel.conversation
        ids: list[int] = []
        for i, n in enumerate(sizes):
            msg = {"role": "user" if i % 2 == 0 else "assistant",
                   "content": words(n)}
            results = await self.plugin.pm.ahook.persist_message(
                channel=self.channel, message=msg,
                message_type=MessageType.MESSAGE,
            )
            rowid = resolve_single_result(results, "persist_message")
            conv.append(msg)
            conv.messages[-1]["_db_id"] = rowid
            ids.append(rowid)
        return ids

    async def send(self, text: str = "next question") -> None:
        """Send one user message through the Agent and drain; the stub
        snapshots the window at the main-turn call."""
        async def fake_turn(client, messages, tool_schemas, **kwargs):
            """Record the window at the LLM call, then answer."""
            conv = self.channel.conversation
            self.seen["estimate"] = conv.token_estimate()
            self.seen["rows"] = len(conv.messages)
            self.seen["system_prompt"] = conv.system_prompt
            self.seen["newest_tokens"] = message_tokens(conv.messages[-1])
            return AgentTurnResult(
                message={"role": "assistant", "content": "ok"},
                tool_calls=[], text="ok", latency_ms=1.0,
            )

        with patch("corvidae.agent.run_agent_turn", fake_turn):
            await self.plugin.on_message(
                channel=self.channel, sender="user", text=text,
                trace_id=mint_trace_id(),
            )
            await drain(self.plugin, self.channel)

    async def log_rows(self) -> list[tuple]:
        """Every message_log row for the channel, in id order."""
        async with self.db.execute(
            "SELECT id, message, message_type FROM message_log "
            "WHERE channel_id = ? ORDER BY id", (self.channel.id,),
        ) as cursor:
            return await cursor.fetchall()


async def make_rig(*, max_tokens: int = MAX_TOKENS, compaction: bool = True,
                   system_prompt: str = SYSTEM_PROMPT) -> Rig:
    """Build the Agent graph, optionally with a real CompactionPlugin, and
    initialise the channel's conversation with one throwaway exchange."""
    plugin, channel, db = await build_plugin_and_channel(agent_defaults={
        "system_prompt": system_prompt,
        "max_context_tokens": max_tokens,
        "reasoning_history": "drop",
    })
    client = MagicMock()
    client.chat = AsyncMock(return_value=_make_text_response("ok"))
    plugin._client = client
    plugin._mark_started()
    rig = Rig(plugin=plugin, channel=channel, db=db)
    if compaction:
        comp = CompactionPlugin(plugin.pm)
        # The summary must never go to a fork or a real client in these tests.
        comp._fork_summary = AsyncMock(return_value=None)
        comp._summarize = AsyncMock(return_value="a summary")
        plugin.pm.register(comp, name="compaction")
        rig.compaction = comp
    # Initialise the conversation, then clear it so each test seeds its own.
    await rig.send("hi")
    channel.conversation.messages.clear()
    return rig


@pytest.fixture
async def rig_factory():
    """Factory for rigs, torn down after the test."""
    rigs: list[Rig] = []

    async def make(**kwargs) -> Rig:
        rig = await make_rig(**kwargs)
        rigs.append(rig)
        return rig

    yield make
    for rig in rigs:
        task_plugin = rig.plugin.pm.get_plugin("task")
        if task_plugin:
            await task_plugin.on_stop()
        await rig.db.close()


# ---------------------------------------------------------------------------
# R1: each skip path ends within budget at the main-turn call (tests 8-13)
# ---------------------------------------------------------------------------


async def test_too_few_rows_with_huge_ones_ends_within_budget(rig_factory):
    """8: min_messages_to_compact or fewer rows, each huge: compaction
    declines, the fallback still brings the window within budget."""
    rig = await rig_factory()
    await rig.seed([350, 350, 350, 350])  # 4 rows + new message = 5 <= min
    await rig.send()
    assert rig.seen["estimate"] <= MAX_TOKENS
    rig.compaction._summarize.assert_not_awaited()


async def test_failure_cooldown_ends_within_budget(rig_factory):
    """9: compaction in failure cooldown: still within budget."""
    rig = await rig_factory()
    rig.compaction._last_failed_compaction[rig.channel.id] = time.monotonic()
    await rig.seed([100] * 10)
    await rig.send()
    assert rig.seen["estimate"] <= MAX_TOKENS
    rig.compaction._summarize.assert_not_awaited()


async def test_too_soon_since_last_compaction_ends_within_budget(rig_factory):
    """10: too soon since the last compaction: still within budget."""
    rig = await rig_factory()
    await rig.seed([100] * 10)
    rig.compaction._last_compaction_msg_count[rig.channel.id] = 8
    await rig.send()
    assert rig.seen["estimate"] <= MAX_TOKENS
    rig.compaction._summarize.assert_not_awaited()


async def test_empty_summary_ends_within_budget(rig_factory):
    """11: the summary comes back empty: window unchanged by compaction,
    the fallback brings it within budget."""
    rig = await rig_factory()
    rig.compaction._summarize = AsyncMock(return_value="")
    await rig.seed([100] * 10)
    await rig.send()
    rig.compaction._summarize.assert_awaited()
    assert rig.seen["estimate"] <= MAX_TOKENS


async def test_summary_call_raising_ends_within_budget(rig_factory):
    """12: the summary call raises (the agent's except path): still within
    budget."""
    rig = await rig_factory()
    rig.compaction._summarize = AsyncMock(side_effect=RuntimeError("boom"))
    await rig.seed([100] * 10)
    await rig.send()
    rig.compaction._summarize.assert_awaited()
    assert rig.seen["estimate"] <= MAX_TOKENS


async def test_no_compaction_plugin_ends_within_budget(rig_factory):
    """13: with no compaction plugin loaded the fallback still applies."""
    rig = await rig_factory(compaction=False)
    await rig.seed([100] * 10)
    await rig.send()
    assert rig.seen["estimate"] <= MAX_TOKENS


# ---------------------------------------------------------------------------
# R2 guard: message_log is never touched (test 14; passes before the change)
# ---------------------------------------------------------------------------


async def test_message_log_is_untouched_by_the_fallback(rig_factory):
    """14 (guard): the seeded rows are all still in message_log, byte for
    byte, after the fallback has dropped them from the window."""
    rig = await rig_factory(compaction=False)
    await rig.seed([100] * 10)
    before = await rig.log_rows()
    await rig.send()
    after = await rig.log_rows()
    # Existing rows are identical; the only addition is the turn's own rows.
    assert after[:len(before)] == before
    assert len(after) >= len(before)


# ---------------------------------------------------------------------------
# R4: the WARNING (test 15) and the single-row cases of test 7
# ---------------------------------------------------------------------------


def fallback_warnings(caplog) -> list[str]:
    """WARNING messages that mention the token budget fallback."""
    return [
        r.getMessage() for r in caplog.records
        if r.levelno == logging.WARNING and "max_context_tokens" in r.getMessage()
    ]


async def test_warning_names_channel_rows_ids_truncation_and_remedy(
    rig_factory, caplog
):
    """15: one WARNING with the channel id, the dropped row count and id
    range, the truncation, max_context_tokens and compaction."""
    rig = await rig_factory(compaction=False)
    ids = await rig.seed([100] * 6)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        # A newest message larger than the whole budget forces every older
        # row out and a truncation of the newest.
        await rig.send(words(MAX_TOKENS + 400))
    messages = fallback_warnings(caplog)
    assert len(messages) == 1, messages
    text = messages[0]
    assert rig.channel.id in text
    assert f"dropped {len(ids)} oldest rows" in text
    assert f"{min(ids)}-{max(ids)}" in text
    assert "truncated" in text
    assert "max_context_tokens" in text
    assert "compaction" in text
    assert rig.seen["estimate"] <= MAX_TOKENS
    # The log keeps the newest user row whole: full content, no marker.
    last_message = json.loads((await rig.log_rows())[-2][1])
    assert "truncated to fit" not in last_message["content"]
    assert len(last_message["content"]) >= len(words(MAX_TOKENS + 400))


async def test_under_budget_logs_no_warning_and_keeps_the_window(
    rig_factory, caplog
):
    """16 (guard): a window within budget is sent unchanged and silently."""
    rig = await rig_factory()
    await rig.seed([20, 20])
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        await rig.send()
    assert fallback_warnings(caplog) == []
    # Two seeded rows + the new user message, nothing dropped.
    assert rig.seen["rows"] == 3


async def test_system_prompt_alone_over_budget_still_warns_with_numbers(
    rig_factory, caplog
):
    """7 (Agent level, system prompt over budget): a single-row window whose
    system prompt alone exceeds the budget still logs the WARNING, stating
    the system prompt's tokens, the newest row's tokens and the budget, and
    says 'system prompt alone'."""
    prompt = words(300)
    rig = await rig_factory(max_tokens=100, system_prompt=prompt,
                            compaction=False)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        await rig.send("hello")
    messages = fallback_warnings(caplog)
    assert len(messages) == 1, messages
    text = messages[0]
    system_tokens = count_tokens(rig.seen["system_prompt"])
    assert system_tokens > 100
    assert f"the system prompt alone ({system_tokens} tokens) exceeds the 100 budget" in text
    assert "shorten the system prompt" in text


async def test_newest_row_not_fitting_warns_with_exact_numbers(
    rig_factory, caplog
):
    """7 (Agent level, system prompt within budget): the newest row cannot
    be truncated to fit; the WARNING reports system prompt, newest row and
    budget accurately and does not claim 'system prompt alone'."""
    rig = await rig_factory(max_tokens=100, system_prompt=words(90),
                            compaction=False)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        await rig.send("hi")
    system_tokens = count_tokens(rig.seen["system_prompt"])
    newest_tokens = rig.seen["newest_tokens"]
    # Sanity of the scenario: the prompt fits alone, the row does not.
    assert system_tokens <= 100 < system_tokens + newest_tokens
    messages = fallback_warnings(caplog)
    assert len(messages) == 1, messages
    text = messages[0]
    assert (
        f"system prompt {system_tokens} tokens, newest row {newest_tokens} "
        "tokens, budget 100"
    ) in text
    assert "system prompt alone" not in text
