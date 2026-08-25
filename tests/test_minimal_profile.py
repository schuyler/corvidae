"""RED test for the minimal profile (disentangle-buster.md Design §3).

A documented configuration with the cognition plugin set (memory,
memory_tools, funnel, appraisal, critique, outcome_log) disabled must boot,
chat, run the multi-turn tool loop, compact, persist, and survive restart —
all with a real Runtime (real entry-point plugin loading, real
plugins.disabled blocking), a tmp_path agent.yaml, and a fake LLM client.

This is deliberately ONE flow (§3's own framing): each assertion depends on
plugin/channel state built by the previous one, and re-deriving that state
per-test would mean re-running the same boot+turns repeatedly for no
independent value.
"""

import asyncio
import json
from pathlib import Path

import pytest
import yaml

from llm_response_fixtures import (
    _make_text_response,
    _make_tool_call,
    _make_tool_call_response,
)

MINIMAL_DISABLED_PLUGINS = [
    "memory", "memory_tools", "funnel", "appraisal", "critique", "outcome_log",
]

REPO_ROOT = Path(__file__).resolve().parent.parent
MINIMAL_EXAMPLE_PATH = REPO_ROOT / "agent.minimal.yaml.example"

pytestmark = pytest.mark.timeout(15)


class TestShippedMinimalExample:
    """R3 acceptance: 'a minimal agent.yaml example ships' (Design §3,
    sibling of agent.yaml.example at repo root)."""

    def test_example_file_exists_with_expected_shape(self):
        assert MINIMAL_EXAMPLE_PATH.exists(), (
            f"{MINIMAL_EXAMPLE_PATH} must exist -- the shipped minimal-profile "
            "example config"
        )
        raw = MINIMAL_EXAMPLE_PATH.read_text()
        config = yaml.safe_load(raw)

        assert "main" in config.get("llm", {}), "llm.main block required"
        assert "session_db" in config.get("daemon", {}), "daemon.session_db required"
        assert len(config.get("channels", {})) == 1, "exactly one channel"
        assert set(config.get("plugins", {}).get("disabled", [])) == set(
            MINIMAL_DISABLED_PLUGINS
        )
        # A comment naming what the profile gives up.
        assert "memory" in raw.lower()
        assert "appraisal" in raw.lower() or "critique" in raw.lower()


def _write_minimal_config(tmp_path, session_db_path, extra_channels=None):
    config = {
        "agent": {
            "system_prompt": "You are a test assistant.",
            "max_context_tokens": 8000,
        },
        "llm": {
            "main": {"base_url": "http://127.0.0.1:1/v1", "model": "test-model"},
        },
        "daemon": {"session_db": str(session_db_path)},
        "plugins": {"disabled": list(MINIMAL_DISABLED_PLUGINS)},
    }
    if extra_channels:
        config["channels"] = extra_channels
    config_path = tmp_path / "agent.yaml"
    config_path.write_text(yaml.dump(config))
    return str(config_path)


def _make_fake_chat(marker_path, tool_used: dict):
    """A stand-in LLM: triggers one read_file tool call for the marker
    message, echoes tool results, and otherwise gives a short ack — good
    enough for both ordinary turns and CompactionPlugin's summarizer call
    (which doesn't care about summary content for this test)."""

    async def fake_chat(messages, tools=None, extra_body=None):
        last = messages[-1]
        role = last.get("role")
        content = last.get("content") or ""
        if role == "tool":
            return _make_text_response(f"the file said: {content}")
        if "read the marker file" in content and not tool_used["done"]:
            tool_used["done"] = True
            return _make_tool_call_response(
                [_make_tool_call("call_1", "read_file", {"path": str(marker_path)})]
            )
        return _make_text_response(f"ack: {content[:30]}")

    return fake_chat


async def _drain_once(agent):
    """One drain cycle: channel queues -> pending tasks -> channel queues."""
    for queue in agent.queues.values():
        await queue.drain()
    task_plugin = agent.pm.get_plugin("task")
    if task_plugin and task_plugin.task_queue:
        await task_plugin.task_queue.queue.join()
        for _ in range(3):
            await asyncio.sleep(0)
    for queue in agent.queues.values():
        await queue.drain()


async def _drain_all(agent, max_iterations: int = 20):
    """Drain repeatedly until the task queue is idle and no channel queue
    produces further work.

    One _drain_once() pass is not always enough: task_done() (which
    queue.join() waits on) is called before on_complete() -> on_notify() ->
    re-enqueue, so a single pass can observe an idle task queue before the
    resulting notification has landed on the channel's SerialQueue. Loop
    until stable (mirrors IntegrationHarness.drain_until_stable in
    test_integration.py).
    """
    prev_counts = None
    stable = 0
    for _ in range(max_iterations):
        await _drain_once(agent)
        for _ in range(3):
            await asyncio.sleep(0)
        task_plugin = agent.pm.get_plugin("task")
        tq = task_plugin.task_queue if task_plugin else None
        task_idle = tq is None or tq.is_idle
        queues_empty = all(q.is_empty for q in agent.queues.values())
        signal = (queues_empty, task_idle)
        if signal == prev_counts:
            stable += 1
            if stable >= 2 and queues_empty and task_idle:
                return
        else:
            stable = 0
        prev_counts = signal
    raise AssertionError("drain did not stabilize")


class TestMinimalProfile:
    async def test_boots_chats_tool_loops_compacts_persists_and_survives_restart(
        self, tmp_path
    ):
        from corvidae.runtime import Runtime
        from unittest.mock import AsyncMock

        session_db = tmp_path / "sessions.db"

        # (a) Runtime.start() succeeds with the disabled list, and none of
        # the six cognition plugins is registered.
        config_path = _write_minimal_config(tmp_path, session_db)
        rt = Runtime(config_path=config_path)
        await rt.start()
        try:
            for name in MINIMAL_DISABLED_PLUGINS:
                assert rt.pm.get_plugin(name) is None, (
                    f"{name!r} must not be registered in the minimal profile"
                )

            agent = rt.pm.get_plugin("agent")
            assert agent is not None

            marker_path = tmp_path / "marker.txt"
            marker_path.write_text("QRT-9083")
            tool_used = {"done": False}
            llm = rt.pm.get_plugin("llm")
            llm.main_client.chat = AsyncMock(
                side_effect=_make_fake_chat(marker_path, tool_used)
            )

            registry = rt.pm.get_plugin("registry")
            main_channel = registry.get_or_create("irc", "main")

            # (b) A user message produces a response.
            await rt.pm.ahook.on_message(
                channel=main_channel, sender="user", text="hello there"
            )
            await _drain_all(agent)
            assert main_channel.conversation is not None
            assert main_channel.conversation.messages[-1]["role"] == "assistant"

            # (c) A scripted tool-call-then-final-answer exchange completes
            # (tool loop through TaskQueue re-entry).
            await rt.pm.ahook.on_message(
                channel=main_channel, sender="user",
                text="please read the marker file and report its contents",
            )
            await _drain_all(agent)
            last_msg = main_channel.conversation.messages[-1]
            assert last_msg["role"] == "assistant"
            assert "QRT-9083" in (last_msg.get("content") or "")

            # (d) With a small max_context_tokens, compaction fires and the
            # window shrinks: a dedicated channel, pre-registered with a
            # tight limit, receives enough filler turns to cross the
            # compaction threshold.
            compact_config_path = _write_minimal_config(
                tmp_path, session_db,
                extra_channels={"irc:compact": {"max_context_tokens": 200}},
            )
            # Re-render with the tight-limit channel pre-registered; same
            # running daemon, so reload the channel config directly rather
            # than restarting (restart is exercised separately, below).
            from corvidae.channel import load_channel_config
            with open(compact_config_path) as f:
                reloaded = yaml.safe_load(f)
            load_channel_config(reloaded, registry)
            compact_channel = registry.get("irc:compact")
            assert compact_channel is not None

            filler = "the quick brown fox jumps over the lazy dog. " * 15
            for i in range(8):
                await rt.pm.ahook.on_message(
                    channel=compact_channel, sender="user",
                    text=f"filler message {i}: {filler}",
                )
                await _drain_all(agent)

            persistence = rt.pm.get_plugin("persistence")
            async with persistence.db.execute(
                "SELECT COUNT(*) FROM message_log "
                "WHERE channel_id = ? AND message_type = 'summary'",
                (compact_channel.id,),
            ) as cursor:
                (summary_count,) = await cursor.fetchone()
            assert summary_count >= 1, (
                "compaction never fired: expected at least one summary-type "
                "message_log row for the tight-limit channel"
            )
            pre_compaction_len = len(compact_channel.conversation.messages)
            assert pre_compaction_len < 8 * 2, (
                "compaction should have replaced older messages with a "
                f"summary, but the window still has {pre_compaction_len} messages"
            )
        finally:
            await rt.stop()

        # (e) After Runtime.stop() and a fresh Runtime.start() on the same
        # session_db, load_conversation restores history and the channel
        # still answers.
        rt2 = Runtime(config_path=config_path)
        await rt2.start()
        try:
            for name in MINIMAL_DISABLED_PLUGINS:
                assert rt2.pm.get_plugin(name) is None

            agent2 = rt2.pm.get_plugin("agent")
            llm2 = rt2.pm.get_plugin("llm")
            tool_used2 = {"done": True}  # tool already exercised pre-restart
            llm2.main_client.chat = AsyncMock(
                side_effect=_make_fake_chat(marker_path, tool_used2)
            )

            registry2 = rt2.pm.get_plugin("registry")
            main_channel2 = registry2.get_or_create("irc", "main")

            await rt2.pm.ahook.on_message(
                channel=main_channel2, sender="user", text="are you still there?"
            )
            await _drain_all(agent2)

            reloaded_contents = [
                m.get("content") for m in main_channel2.conversation.messages
            ]
            assert any(
                c and "hello there" in c for c in reloaded_contents
            ), "restart must reload prior history via load_conversation"
            assert main_channel2.conversation.messages[-1]["role"] == "assistant"
        finally:
            await rt2.stop()
