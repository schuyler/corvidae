"""Consolidation as a fork of the window (GitHub issue #51, phase 3).

Compaction trigger: MemoryPlugin.before_compaction forks the record from
the dropped prefix and stashes it; the on_compaction task consumes the
stash instead of the standalone background call. Idle trigger: fork from
the full window when the range is in it. Everything after the LLM call is
unchanged, and any miss falls back to the standalone call.
"""

import json

import pytest

from corvidae.attribution import get_attribution
from corvidae.context import ContextWindow, MessageType
from corvidae.fork import ForkRefused, ForkResult
from corvidae.memory import DEFAULT_CONSOLIDATION_FORK_PROMPT, FORK_STASH_MAXSIZE

from test_memory_consolidation import (
    build_consolidation_env,
    fetch_memories,
    fetch_watermark,
    seed_dialog,
)

FORK_JSON = json.dumps({
    "summary": "I told kestrel the wifi drops were the brownout detector.",
    "topic_tags": ["wifi"],
    "participants": ["kestrel"],
})

DIALOG = [
    ("user", "my weather station keeps dropping wifi"),
    ("assistant", "Does it happen when the heater kicks on?"),
    ("user", "yes, every time"),
    ("assistant", "That's the brownout detector resetting the radio."),
]


class FakeAgent:
    """Stands in for Agent: records forks and returns canned text."""

    def __init__(self, text=FORK_JSON, stages=("consolidation",), error=None):
        self.text = text
        self.stages = set(stages)
        self.error = error
        self.calls = []

    def fork_enabled(self, stage):
        return stage in self.stages

    async def fork(self, channel, instruction, *, cut=None, extra_body=None):
        self.calls.append({
            "instruction": instruction, "cut": cut, "extra_body": extra_body,
            "attribution": dict(get_attribution()),
        })
        if self.error is not None:
            raise self.error
        return ForkResult(text=self.text, message={"role": "assistant", "content": self.text})


class FakeRegistry:
    """Just enough ChannelRegistry for the idle trigger's channel lookup."""

    def __init__(self, channel):
        self.channel = channel

    def get(self, channel_id):
        return self.channel if channel_id == self.channel.id else None


def _window(rows: list[tuple[str, str, int]], summary: bool = False) -> ContextWindow:
    """A live window holding (role, content, rowid) messages."""
    conv = ContextWindow("irc:#test")
    conv.system_prompt = "You are Lex."
    if summary:
        conv.messages.append({
            "role": "assistant", "content": "[Summary of earlier conversation]\nold",
            "_message_type": MessageType.SUMMARY,
        })
    for role, content, rowid in rows:
        conv.messages.append({
            "role": role, "content": content, "_db_id": rowid,
            "_message_type": MessageType.MESSAGE,
        })
    return conv


async def _env(agent=None):
    memory, persistence, stub, channel, db = await build_consolidation_env()
    if agent is not None:
        memory.pm.register(agent, name="agent")
    memory.pm.register(FakeRegistry(channel), name="registry")
    return memory, persistence, stub, channel, db


def _standalone_calls(stub):
    """Chat calls the standalone consolidation path made."""
    return [c for c in stub.chat_calls if c["attribution"].get("stage") == "consolidation"]


# ---------------------------------------------------------------------------
# Compaction trigger
# ---------------------------------------------------------------------------


class TestCompactionTrigger:
    async def test_fork_record_is_stashed_and_consumed(self):
        agent = FakeAgent()
        memory, persistence, stub, channel, db = await _env(agent)
        try:
            ids = await seed_dialog(persistence, channel, DIALOG + [("user", "and now?")])
            channel.conversation = _window(
                [(r, c, i) for (r, c), i in zip(DIALOG + [("user", "and now?")], ids)]
            )
            compacted = ids[:4]
            await memory.before_compaction(channel=channel, compacted_ids=compacted, cut=4)

            assert len(agent.calls) == 1
            call = agent.calls[0]
            assert call["cut"] == 4
            assert call["instruction"].startswith(DEFAULT_CONSOLIDATION_FORK_PROMPT)
            assert call["attribution"]["stage"] == "consolidation"
            assert call["attribution"]["trigger"] == "compaction"

            await memory.on_compaction(
                channel=channel, summary_msg={}, retain_count=1, compacted_ids=compacted,
            )
            await memory.wait_for_background_tasks()

            assert _standalone_calls(stub) == []
            records = await fetch_memories(db)
            assert len(records) == 1
            assert records[0]["summary"] == json.loads(FORK_JSON)["summary"]
            assert (records[0]["msg_id_start"], records[0]["msg_id_end"]) == (ids[0], ids[3])
            assert await fetch_watermark(db, channel.id) == ids[3]
            assert memory._fork_stash == {}
        finally:
            await db.close()

    async def test_range_not_in_window_falls_back(self):
        agent = FakeAgent()
        memory, persistence, stub, channel, db = await _env(agent)
        try:
            ids = await seed_dialog(persistence, channel, DIALOG)
            # The window lost the first row (an earlier compaction whose
            # consolidation failed): the fork can't see the whole range.
            channel.conversation = _window(
                [(r, c, i) for (r, c), i in zip(DIALOG[1:], ids[1:])]
            )
            await memory.before_compaction(channel=channel, compacted_ids=ids[1:3], cut=2)
            assert agent.calls == []

            await memory.on_compaction(
                channel=channel, summary_msg={}, retain_count=1, compacted_ids=ids[1:3],
            )
            await memory.wait_for_background_tasks()
            calls = _standalone_calls(stub)
            assert len(calls) == 1
            assert calls[0]["attribution"]["mode"] == "standalone"
        finally:
            await db.close()

    async def test_forks_disabled_is_standalone(self):
        agent = FakeAgent(stages=())
        memory, persistence, stub, channel, db = await _env(agent)
        try:
            ids = await seed_dialog(persistence, channel, DIALOG)
            channel.conversation = _window([(r, c, i) for (r, c), i in zip(DIALOG, ids)])
            await memory.before_compaction(channel=channel, compacted_ids=ids[:3], cut=3)
            assert agent.calls == []
            await memory.on_compaction(
                channel=channel, summary_msg={}, retain_count=1, compacted_ids=ids[:3],
            )
            await memory.wait_for_background_tasks()
            assert len(_standalone_calls(stub)) == 1
        finally:
            await db.close()

    @pytest.mark.parametrize("agent", [
        FakeAgent(error=ForkRefused("tool call")),
        FakeAgent(error=RuntimeError("http 500")),
        FakeAgent(text="not json at all"),
        FakeAgent(text=json.dumps({"summary": "  "})),
    ])
    async def test_fork_failure_leaves_no_stash_and_falls_back(self, agent):
        memory, persistence, stub, channel, db = await _env(agent)
        try:
            ids = await seed_dialog(persistence, channel, DIALOG)
            channel.conversation = _window([(r, c, i) for (r, c), i in zip(DIALOG, ids)])
            await memory.before_compaction(channel=channel, compacted_ids=ids[:3], cut=3)
            assert memory._fork_stash == {}
            await memory.on_compaction(
                channel=channel, summary_msg={}, retain_count=1, compacted_ids=ids[:3],
            )
            await memory.wait_for_background_tasks()
            assert len(_standalone_calls(stub)) == 1
            assert len(await fetch_memories(db)) == 1
        finally:
            await db.close()

    async def test_stash_from_a_different_watermark_is_discarded(self):
        """If the watermark moved between the fork and the task, the stashed
        record covers the wrong range: use the standalone call instead."""
        agent = FakeAgent()
        memory, persistence, stub, channel, db = await _env(agent)
        try:
            ids = await seed_dialog(persistence, channel, DIALOG)
            channel.conversation = _window([(r, c, i) for (r, c), i in zip(DIALOG, ids)])
            await memory.before_compaction(channel=channel, compacted_ids=ids[:3], cut=3)
            assert len(memory._fork_stash) == 1

            mem_db = await memory._ensure_schema()
            await memory._set_watermark(mem_db, channel.id, ids[0])
            await mem_db.commit()

            await memory.on_compaction(
                channel=channel, summary_msg={}, retain_count=1, compacted_ids=ids[:3],
            )
            await memory.wait_for_background_tasks()
            assert len(_standalone_calls(stub)) == 1
            records = await fetch_memories(db)
            assert records[0]["msg_id_start"] == ids[1]
            assert memory._fork_stash == {}
        finally:
            await db.close()

    async def test_summary_rows_in_range_do_not_block_the_fork(self):
        """A live window's summary carries no _db_id, but its message_log
        row can sit inside the range. Summaries aren't consolidated, so only
        dialog rows count for containment."""
        agent = FakeAgent()
        memory, persistence, stub, channel, db = await _env(agent)
        try:
            ids = await seed_dialog(persistence, channel, DIALOG[:2])
            await db.execute(
                "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
                "VALUES (?, ?, ?, 'summary')",
                (channel.id, json.dumps({"role": "assistant", "content": "s"}), 0.0),
            )
            await db.commit()
            ids += await seed_dialog(persistence, channel, DIALOG[2:])
            channel.conversation = _window(
                [(r, c, i) for (r, c), i in zip(DIALOG, ids)], summary=True
            )
            await memory.before_compaction(channel=channel, compacted_ids=ids[:3], cut=4)
            assert len(agent.calls) == 1
        finally:
            await db.close()

    async def test_stash_is_bounded(self):
        memory, persistence, stub, channel, db = await _env(FakeAgent())
        try:
            for i in range(FORK_STASH_MAXSIZE + 5):
                memory._stash_put(("c", i), (0, {"summary": "s"}))
            assert len(memory._fork_stash) == FORK_STASH_MAXSIZE
            assert ("c", 0) not in memory._fork_stash
        finally:
            await db.close()


# ---------------------------------------------------------------------------
# Idle trigger
# ---------------------------------------------------------------------------


class TestIdleTrigger:
    async def test_idle_forks_from_the_full_window(self):
        agent = FakeAgent()
        memory, persistence, stub, channel, db = await _env(agent)
        try:
            ids = await seed_dialog(persistence, channel, DIALOG)
            channel.conversation = _window([(r, c, i) for (r, c), i in zip(DIALOG, ids)])
            await memory._consolidate_range(channel.id, ids[-1], "idle")

            assert len(agent.calls) == 1
            assert agent.calls[0]["cut"] is None
            assert agent.calls[0]["attribution"]["trigger"] == "idle"
            assert _standalone_calls(stub) == []
            records = await fetch_memories(db)
            assert records[0]["summary"] == json.loads(FORK_JSON)["summary"]
        finally:
            await db.close()

    async def test_idle_without_live_window_is_standalone(self):
        agent = FakeAgent()
        memory, persistence, stub, channel, db = await _env(agent)
        try:
            ids = await seed_dialog(persistence, channel, DIALOG)
            channel.conversation = None
            await memory._consolidate_range(channel.id, ids[-1], "idle")
            assert agent.calls == []
            assert len(_standalone_calls(stub)) == 1
        finally:
            await db.close()

    async def test_idle_fork_failure_is_standalone(self):
        agent = FakeAgent(error=RuntimeError("down"))
        memory, persistence, stub, channel, db = await _env(agent)
        try:
            ids = await seed_dialog(persistence, channel, DIALOG)
            channel.conversation = _window([(r, c, i) for (r, c), i in zip(DIALOG, ids)])
            await memory._consolidate_range(channel.id, ids[-1], "idle")
            assert len(_standalone_calls(stub)) == 1
            assert len(await fetch_memories(db)) == 1
        finally:
            await db.close()


# ---------------------------------------------------------------------------
# Where the range starts
# ---------------------------------------------------------------------------


class TestBoundary:
    async def _instruction(self, window_rows, summary, watermark_index):
        agent = FakeAgent()
        memory, persistence, stub, channel, db = await _env(agent)
        try:
            ids = await seed_dialog(persistence, channel, DIALOG)
            rows = [(r, c, i) for (r, c), i in zip(DIALOG, ids)][window_rows]
            channel.conversation = _window(rows, summary=summary)
            if watermark_index is not None:
                mem_db = await memory._ensure_schema()
                await memory._set_watermark(mem_db, channel.id, ids[watermark_index])
                await mem_db.commit()
            await memory._consolidate_range(channel.id, ids[-1], "idle")
            return agent.calls[0]["instruction"]
        finally:
            await db.close()

    async def test_whole_window(self):
        text = await self._instruction(slice(None), False, None)
        assert "the whole conversation above" in text

    async def test_after_the_summary(self):
        text = await self._instruction(slice(None), True, None)
        assert "follows the summary" in text

    async def test_quotes_first_message_after_consolidated_rows(self):
        """Rows below the watermark are still in the window (idle
        consolidated them earlier): name the first new message."""
        text = await self._instruction(slice(None), True, 1)
        assert "«yes, every time»" in text

    async def test_configured_fork_prompt_is_used(self):
        """memory.consolidation_fork_prompt is static config."""
        agent = FakeAgent()
        memory, persistence, stub, channel, db = await _env(agent)
        try:
            await memory.on_init(memory.pm, {"memory": {"consolidation_fork_prompt": "Remember it."}})
            ids = await seed_dialog(persistence, channel, DIALOG)
            channel.conversation = _window([(r, c, i) for (r, c), i in zip(DIALOG, ids)])
            await memory._consolidate_range(channel.id, ids[-1], "idle")
            assert agent.calls[0]["instruction"].startswith("Remember it.")
        finally:
            await db.close()
