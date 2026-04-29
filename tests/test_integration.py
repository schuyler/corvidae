"""Integration tests for the Corvidae agent daemon.

Assembles the full plugin graph (with FakeTransportPlugin replacing CLI/IRC)
and exercises message routing, tool dispatch, persistence, multi-channel
isolation, error handling, and tool collection.

Baseline test count: 525 (all passing before this file was added).
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from corvidae.agent import Agent
from corvidae.channel import Channel, ChannelConfig, ChannelRegistry
from corvidae.compaction import CompactionPlugin
from corvidae.hooks import create_plugin_manager, hookimpl, validate_dependencies
from corvidae.idle import IdleMonitorPlugin
from corvidae.llm import LLMClient
from corvidae.llm_plugin import LLMPlugin
from corvidae.mcp_client import McpClientPlugin
from corvidae.persistence import PersistencePlugin
from corvidae.task import TaskPlugin
from corvidae.thinking import ThinkingPlugin
from corvidae.tools import CoreToolsPlugin
from corvidae.tools.subagent import SubagentPlugin

pytestmark = pytest.mark.timeout(30)


# ---------------------------------------------------------------------------
# Response builder helpers (shared with other test files)
# ---------------------------------------------------------------------------


def _make_text_response(text: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


def _make_tool_call_response(calls: list[dict]) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": calls,
                }
            }
        ]
    }


def _make_tool_call(call_id: str, name: str, args: dict) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(args),
        },
    }


# ---------------------------------------------------------------------------
# FakeTransportPlugin
# ---------------------------------------------------------------------------


class FakeTransportPlugin:
    """Replaces CLIPlugin and IRCPlugin. Captures send_message calls."""

    depends_on = {"registry"}

    def __init__(self, pm):
        self.pm = pm
        self.sent: list[tuple[Channel, str, float | None]] = []
        self._registry = None

    @hookimpl
    async def on_start(self, config):
        from corvidae.hooks import get_dependency
        self._registry = get_dependency(self.pm, "registry", ChannelRegistry)

    @hookimpl
    async def send_message(self, channel, text, latency_ms=None):
        self.sent.append((channel, text, latency_ms))

    @hookimpl
    async def on_stop(self):
        pass


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------


def make_config(*, db_path: str = ":memory:", extra: dict | None = None) -> dict:
    config = {
        "agent": {
            "system_prompt": "You are a test agent.",
            "max_context_tokens": 4096,
        },
        "llm": {
            "main": {
                "base_url": "http://fake",
                "model": "test-model",
            },
        },
        "daemon": {
            "session_db": db_path,
        },
        "_base_dir": Path("."),
    }
    if extra:
        config.update(extra)
    return config


# ---------------------------------------------------------------------------
# IntegrationHarness
# ---------------------------------------------------------------------------


@dataclass
class IntegrationHarness:
    pm: object
    agent: Agent
    task_plugin: TaskPlugin
    registry: ChannelRegistry
    transport: FakeTransportPlugin
    mock_client: object

    def set_llm_responses(self, responses: list[dict]) -> None:
        self.mock_client.chat = AsyncMock(side_effect=responses)

    async def inject_message(self, channel_key: str, sender: str, text: str) -> None:
        transport_name, scope = channel_key.split(":", 1)
        channel = self.registry.get_or_create(transport_name, scope)
        await self.pm.ahook.on_message(channel=channel, sender=sender, text=text)

    async def drain_all(self) -> None:
        """Drain one cycle: serial queues → task queue → serial queues.

        This drains the channel serial queues, waits for any pending tasks
        to complete, and drains the serial queues again to pick up the
        resulting notifications. One call may not be enough for multi-turn
        tool cycles — use drain_until_stable() for those.
        """
        for queue in self.agent.queues.values():
            await queue.drain()
        if self.task_plugin.task_queue:
            await self.task_plugin.task_queue.queue.join()
            # Worker needs 3 cycles: resume → await on_complete → on_notify → enqueue
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            await asyncio.sleep(0)
        # Second drain for notification re-entry
        for queue in self.agent.queues.values():
            await queue.drain()

    async def drain_until_stable(self, max_iterations: int = 20) -> None:
        """Drain repeatedly until everything is idle and no new messages are sent.

        Loops drain_all until both sent_messages count and task queue idle
        state have stabilized. Uses a combined check because one drain_all
        cycle may dispatch new tasks that complete before the next cycle's
        idle check.
        """
        prev_sent = -1
        stable_count = 0
        for _i in range(max_iterations):
            await self.drain_all()
            # Yield a few times to let the event loop settle any background work
            for _ in range(5):
                await asyncio.sleep(0)
            tq = self.task_plugin.task_queue
            task_idle = (tq is None) or tq.is_idle
            current_sent = len(self.sent_messages)
            if current_sent == prev_sent and task_idle:
                stable_count += 1
                if stable_count >= 2:
                    # Stable for 2 consecutive checks — done
                    return
            else:
                stable_count = 0
            prev_sent = current_sent
        raise AssertionError(
            f"drain did not stabilize after {max_iterations} iterations"
        )

    @property
    def sent_messages(self) -> list[tuple[Channel, str, float | None]]:
        return self.transport.sent


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


async def _build_harness(config: dict) -> IntegrationHarness:
    """Assemble the full plugin graph mirroring main.py registration order."""
    pm = create_plugin_manager()

    agent_defaults = config.get("agent", {})
    registry = ChannelRegistry(agent_defaults)
    pm.register(registry, name="registry")

    persistence_plugin = PersistencePlugin(pm)
    pm.register(persistence_plugin, name="persistence")

    core_tools = CoreToolsPlugin(pm)
    pm.register(core_tools, name="core_tools")

    fake_transport = FakeTransportPlugin(pm)
    pm.register(fake_transport, name="fake_transport")

    task_plugin = TaskPlugin(pm)
    pm.register(task_plugin, name="task")

    subagent_plugin = SubagentPlugin(pm)
    pm.register(subagent_plugin, name="subagent")

    mcp_plugin = McpClientPlugin(pm)
    pm.register(mcp_plugin, name="mcp")

    mock_client = AsyncMock(spec=LLMClient)
    mock_client.start = AsyncMock()
    mock_client.stop = AsyncMock()

    llm_plugin = LLMPlugin(pm)
    pm.register(llm_plugin, name="llm")

    compaction_plugin = CompactionPlugin(pm=pm)
    pm.register(compaction_plugin, name="compaction")

    thinking_plugin = ThinkingPlugin(pm)
    pm.register(thinking_plugin, name="thinking")

    from corvidae.tool_collection import ToolCollectionPlugin
    tool_collection_plugin = ToolCollectionPlugin(pm)
    pm.register(tool_collection_plugin, name="tools")

    agent_loop = Agent(pm)
    pm.register(agent_loop, name="agent")

    idle_monitor_plugin = IdleMonitorPlugin(pm)
    pm.register(idle_monitor_plugin, name="idle_monitor")

    validate_dependencies(pm)

    # Patch LLMClient so LLMPlugin.on_start() uses the mock instead of creating a real session.
    with patch("corvidae.llm_plugin.LLMClient", return_value=mock_client):
        await pm.ahook.on_start(config=config)
        await agent_loop.on_start(config=config)

    return IntegrationHarness(
        pm=pm,
        agent=agent_loop,
        task_plugin=task_plugin,
        registry=registry,
        transport=fake_transport,
        mock_client=mock_client,
    )


@pytest.fixture
async def harness():
    h = await _build_harness(make_config())
    yield h
    if not getattr(h, "_stopped", False):
        await h.agent.on_stop()
        await h.pm.ahook.on_stop()


# ---------------------------------------------------------------------------
# Group A: Full Lifecycle
# ---------------------------------------------------------------------------


class TestGroupALifecycle:

    async def test_a1_plugin_graph_assembly_and_dependency_validation(self):
        """A1. Assemble plugin graph and validate dependencies — no RuntimeError."""
        pm = create_plugin_manager()
        agent_defaults = {"system_prompt": "test", "max_context_tokens": 4096}
        registry = ChannelRegistry(agent_defaults)
        pm.register(registry, name="registry")

        persistence_plugin = PersistencePlugin(pm)
        pm.register(persistence_plugin, name="persistence")

        core_tools = CoreToolsPlugin(pm)
        pm.register(core_tools, name="core_tools")

        fake_transport = FakeTransportPlugin(pm)
        pm.register(fake_transport, name="fake_transport")

        task_plugin = TaskPlugin(pm)
        pm.register(task_plugin, name="task")

        subagent_plugin = SubagentPlugin(pm)
        pm.register(subagent_plugin, name="subagent")

        mcp_plugin = McpClientPlugin(pm)
        pm.register(mcp_plugin, name="mcp")

        llm_plugin = LLMPlugin(pm)
        llm_plugin.main_client = MagicMock()
        pm.register(llm_plugin, name="llm")

        compaction_plugin = CompactionPlugin(pm)
        pm.register(compaction_plugin, name="compaction")

        thinking_plugin = ThinkingPlugin(pm)
        pm.register(thinking_plugin, name="thinking")

        from corvidae.tool_collection import ToolCollectionPlugin
        tool_collection_plugin = ToolCollectionPlugin(pm)
        pm.register(tool_collection_plugin, name="tools")

        agent_loop = Agent(pm)
        pm.register(agent_loop, name="agent")

        idle_monitor_plugin = IdleMonitorPlugin(pm)
        pm.register(idle_monitor_plugin, name="idle_monitor")

        # Should not raise
        validate_dependencies(pm)

    async def test_a2_on_start_initializes_all_components(self, harness):
        """A2. After on_start, verify agent tools, schemas, client, task queue, worker."""
        agent = harness.agent

        # Tool set matches CoreToolsPlugin + SubagentPlugin + TaskPlugin
        expected_tools = {"shell", "read_file", "write_file", "web_fetch", "web_search", "subagent", "task_status", "task_pipeline"}
        assert set(agent.tools.keys()) == expected_tools

        # 8 schemas, each with type="function"
        assert len(agent.tool_schemas) == 8
        for schema in agent.tool_schemas:
            assert schema["type"] == "function"

        # Client is the mock (borrowed from LLMPlugin)
        assert agent._client is harness.mock_client

        # TaskPlugin initialized
        assert harness.task_plugin.task_queue is not None
        assert harness.task_plugin._worker_task is not None
        assert not harness.task_plugin._worker_task.done()

    async def test_a3_on_stop_tears_down_cleanly(self, harness):
        """A3. After on_start + one message + on_stop, all async resources cleaned up."""
        harness.mock_client.chat = AsyncMock(
            return_value=_make_text_response("Hello!")
        )
        await harness.inject_message("test:scope", "user", "hi")
        await harness.drain_all()

        tasks_before = set(asyncio.all_tasks())
        await harness.agent.on_stop()
        await harness.pm.ahook.on_stop()

        # All SerialQueue consumers done
        for queue in harness.agent.queues.values():
            assert queue._consumer_task is None or queue._consumer_task.done()

        # TaskPlugin worker stopped
        assert harness.task_plugin._worker_task is None or harness.task_plugin._worker_task.done()

        # IdleMonitorPlugin is now a pure on_idle consumer (no polling task)

        # LLM client stop() was called
        harness.mock_client.stop.assert_called_once()

        # No new async tasks leaked: tasks after on_stop should be subset of tasks_before
        tasks_after = set(asyncio.all_tasks())
        new_tasks = tasks_after - tasks_before
        assert not new_tasks, f"Leaked tasks after on_stop: {new_tasks}"

        # Mark fixture teardown as already done
        harness._stopped = True


# ---------------------------------------------------------------------------
# Group B: Message-Response Round-Trip
# ---------------------------------------------------------------------------


class TestGroupBRoundTrip:

    async def test_b1_simple_message_text_response(self, harness):
        """B1. User message → LLM returns text → FakeTransportPlugin captures it."""
        harness.mock_client.chat = AsyncMock(
            return_value=_make_text_response("Hello!")
        )

        await harness.inject_message("test:main", "alice", "hi")
        await harness.drain_all()

        assert len(harness.sent_messages) == 1
        channel, text, latency = harness.sent_messages[0]
        assert text == "Hello!"

        # Conversation has user + assistant
        chan = harness.registry.get("test:main")
        assert chan is not None
        assert chan.conversation is not None
        msgs = chan.conversation.messages
        roles = [m["role"] for m in msgs]
        assert roles == ["user", "assistant"]

    async def test_b2_tool_call_full_cycle(self, harness):
        """B2. Message → tool call → result → response (full cycle)."""
        # Override agent tools with test_echo after on_start
        call_args_captured = []

        async def test_echo(text: str) -> str:
            """Echo the text."""
            call_args_captured.append(text)
            return "hi\n"

        harness.agent._tools["test_echo"] = test_echo
        harness.agent._tool_schemas.append({
            "type": "function",
            "function": {
                "name": "test_echo",
                "description": "Echo the text.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                    },
                    "required": ["text"],
                },
            },
        })

        harness.mock_client.chat = AsyncMock(
            side_effect=[
                _make_tool_call_response([
                    _make_tool_call("call-b2", "test_echo", {"text": "hi"})
                ]),
                _make_text_response("The output was: hi"),
            ]
        )

        await harness.inject_message("test:b2", "user", "echo please")
        await harness.drain_until_stable()

        # Tool was called with correct args
        assert call_args_captured == ["hi"]

        # One final text response sent
        assert len(harness.sent_messages) == 1
        assert harness.sent_messages[0][1] == "The output was: hi"

        # latency_ms present and >= 0
        latency = harness.sent_messages[0][2]
        assert latency is None or latency >= 0

        # Conversation: user, assistant(tool_calls), tool(result), assistant(text)
        chan = harness.registry.get("test:b2")
        assert chan is not None
        msgs = chan.conversation.messages
        roles = [m["role"] for m in msgs]
        assert roles == ["user", "assistant", "tool", "assistant"]

    async def test_b3_multiple_tool_calls_as_separate_tasks(self, harness):
        """B3. LLM returns 3 tool calls in one response → 3 tasks dispatched."""
        call_log = []

        async def tool_a(x: str) -> str:
            """Tool A."""
            call_log.append(("a", x))
            return f"a:{x}"

        async def tool_b(x: str) -> str:
            """Tool B."""
            call_log.append(("b", x))
            return f"b:{x}"

        async def tool_c(x: str) -> str:
            """Tool C."""
            call_log.append(("c", x))
            return f"c:{x}"

        for fn in (tool_a, tool_b, tool_c):
            harness.agent._tools[fn.__name__] = fn
            harness.agent._tool_schemas.append({
                "type": "function",
                "function": {
                    "name": fn.__name__,
                    "description": fn.__doc__,
                    "parameters": {
                        "type": "object",
                        "properties": {"x": {"type": "string"}},
                        "required": ["x"],
                    },
                },
            })

        harness.mock_client.chat = AsyncMock(
            side_effect=[
                # Initial: 3 tool calls
                _make_tool_call_response([
                    _make_tool_call("c1", "tool_a", {"x": "1"}),
                    _make_tool_call("c2", "tool_b", {"x": "2"}),
                    _make_tool_call("c3", "tool_c", {"x": "3"}),
                ]),
                # 3 continuations (one per tool result notification)
                _make_text_response("after a"),
                _make_text_response("after b"),
                _make_text_response("after c"),
            ]
        )

        # Spy on enqueue to count tasks
        enqueued_tasks = []
        original_enqueue = harness.task_plugin.task_queue.enqueue

        async def spy_enqueue(task):
            enqueued_tasks.append(task)
            await original_enqueue(task)

        harness.task_plugin.task_queue.enqueue = spy_enqueue

        await harness.inject_message("test:b3", "user", "do three things")
        await harness.drain_until_stable()

        # 3 tasks were enqueued
        assert len(enqueued_tasks) == 3
        call_ids = {t.tool_call_id for t in enqueued_tasks}
        assert call_ids == {"c1", "c2", "c3"}

        # LLM called 4 times (1 initial + 3 continuations)
        assert harness.mock_client.chat.call_count == 4

        # 3 text responses sent (one per continuation)
        assert len(harness.sent_messages) == 3

    async def test_b4_max_turns_enforcement(self, harness):
        """B4. max_turns=2 → third LLM tool call suppressed, fallback sent."""
        # Set up a channel with max_turns=2
        channel = harness.registry.get_or_create(
            "test", "b4",
            config=ChannelConfig(max_turns=2),
        )

        async def dummy_tool(x: str) -> str:
            """A dummy tool."""
            return f"result:{x}"

        harness.agent._tools["dummy_tool"] = dummy_tool
        harness.agent._tool_schemas.append({
            "type": "function",
            "function": {
                "name": "dummy_tool",
                "description": "A dummy tool.",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                    "required": ["x"],
                },
            },
        })

        harness.mock_client.chat = AsyncMock(
            side_effect=[
                # turn_counter 0 < 2: dispatch (→ 1)
                _make_tool_call_response([_make_tool_call("d1", "dummy_tool", {"x": "a"})]),
                # turn_counter 1 < 2: dispatch (→ 2)
                _make_tool_call_response([_make_tool_call("d2", "dummy_tool", {"x": "b"})]),
                # turn_counter 2 < 2 is False: suppressed → fallback text
                _make_tool_call_response([_make_tool_call("d3", "dummy_tool", {"x": "c"})]),
            ]
        )

        enqueued_tasks = []
        original_enqueue = harness.task_plugin.task_queue.enqueue

        async def spy_enqueue(task):
            enqueued_tasks.append(task)
            await original_enqueue(task)

        harness.task_plugin.task_queue.enqueue = spy_enqueue

        await harness.pm.ahook.on_message(channel=channel, sender="user", text="go")
        await harness.drain_until_stable()

        # Only 2 tasks dispatched
        assert len(enqueued_tasks) == 2

        # LLM called 3 times
        assert harness.mock_client.chat.call_count == 3

        # Fallback text sent
        assert len(harness.sent_messages) == 1
        assert "max tool-calling rounds reached" in harness.sent_messages[0][1]

    async def test_b5_user_message_interleaves_during_tool_execution(self, harness):
        """B5. User msg interleaves while tool is blocked on gate."""
        gate = asyncio.Event()

        async def gated_tool(x: str) -> str:
            """A tool that blocks until gate is set."""
            await gate.wait()
            return f"gated:{x}"

        harness.agent._tools["gated_tool"] = gated_tool
        harness.agent._tool_schemas.append({
            "type": "function",
            "function": {
                "name": "gated_tool",
                "description": "A tool that blocks.",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                    "required": ["x"],
                },
            },
        })

        harness.mock_client.chat = AsyncMock(
            side_effect=[
                # msg1: tool call → task dispatched (queue freed)
                _make_tool_call_response([_make_tool_call("g1", "gated_tool", {"x": "v"})]),
                # msg2: enters queue while tool blocked → text response
                _make_text_response("response to msg2"),
                # gate released → notification → text response
                _make_text_response("response to tool result"),
            ]
        )

        channel = harness.registry.get_or_create("test", "b5")

        # Send msg1 → LLM call → task dispatched, queue slot freed
        await harness.pm.ahook.on_message(channel=channel, sender="user", text="msg1")
        await channel_queue_drain(harness, channel)

        # msg2 arrives while tool is blocked
        await harness.pm.ahook.on_message(channel=channel, sender="user", text="msg2")
        await channel_queue_drain(harness, channel)

        # Release gate → tool completes → notification → third LLM call
        gate.set()
        await harness.drain_all()

        assert harness.mock_client.chat.call_count == 3
        assert len(harness.sent_messages) == 2
        texts = [m[1] for m in harness.sent_messages]
        assert "response to msg2" in texts
        assert "response to tool result" in texts

        # Conversation log must have exact interleaved role sequence:
        # user(msg1), assistant(tool_call), user(msg2), assistant(response to msg2),
        # tool(result), assistant(response to tool result)
        msgs = channel.conversation.messages
        roles = [m["role"] for m in msgs]
        assert roles == ["user", "assistant", "user", "assistant", "tool", "assistant"]


async def channel_queue_drain(harness: IntegrationHarness, channel: Channel) -> None:
    """Drain just the queue for one specific channel."""
    if channel.id in harness.agent.queues:
        await harness.agent.queues[channel.id].drain()


# ---------------------------------------------------------------------------
# Group C: Conversation Persistence
# ---------------------------------------------------------------------------


class TestGroupCPersistence:

    async def test_c1_messages_survive_stop_restart(self, tmp_path):
        """C1. Messages survive stop/restart with file-based DB."""
        db_path = str(tmp_path / "test.db")
        config = make_config(db_path=db_path)

        # First harness: send 3 messages
        h1 = await _build_harness(config)
        h1.mock_client.chat = AsyncMock(return_value=_make_text_response("ok"))

        for i in range(3):
            await h1.inject_message("test:persist", "user", f"message {i}")
            await h1.drain_all()

        await h1.agent.on_stop()
        await h1.pm.ahook.on_stop()

        # Second harness: same DB, new plugin graph
        captured_messages = []

        async def capturing_chat(messages, tools=None, extra_body=None):
            captured_messages.extend(messages)
            return _make_text_response("new response")

        h2 = await _build_harness(config)
        h2.mock_client.chat = capturing_chat

        await h2.inject_message("test:persist", "user", "new message")
        await h2.drain_all()

        await h2.agent.on_stop()
        await h2.pm.ahook.on_stop()

        # build_prompt prepends system message, so:
        # [system] + [6 history] + [new user] = 8 entries
        assert len(captured_messages) == 8

    async def test_c3_tool_call_messages_round_trip_through_db(self, tmp_path):
        """C3. Tool call messages round-trip through DB and reload correctly."""
        db_path = str(tmp_path / "c3.db")
        config = make_config(db_path=db_path)

        h1 = await _build_harness(config)

        async def test_echo(text: str) -> str:
            """Echo."""
            return "echo_result"

        h1.agent._tools["test_echo"] = test_echo
        h1.agent._tool_schemas.append({
            "type": "function",
            "function": {
                "name": "test_echo",
                "description": "Echo.",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            },
        })

        h1.mock_client.chat = AsyncMock(
            side_effect=[
                _make_tool_call_response([_make_tool_call("tc1", "test_echo", {"text": "hi"})]),
                _make_text_response("done"),
            ]
        )

        await h1.inject_message("test:c3", "user", "use echo")
        await h1.drain_until_stable()

        await h1.agent.on_stop()
        await h1.pm.ahook.on_stop()

        # Second harness: verify tool messages loaded
        h2 = await _build_harness(config)
        h2.mock_client.chat = AsyncMock(return_value=_make_text_response("ok"))

        # Force conversation to load by sending a message
        await h2.inject_message("test:c3", "user", "check")
        await h2.drain_all()

        chan = h2.registry.get("test:c3")
        assert chan is not None
        msgs = chan.conversation.messages

        # Find assistant message with tool_calls
        assistant_with_tools = [m for m in msgs if m.get("role") == "assistant" and m.get("tool_calls")]
        assert len(assistant_with_tools) >= 1
        assert assistant_with_tools[0]["tool_calls"][0]["id"] == "tc1"

        # Find tool result message
        tool_msgs = [m for m in msgs if m.get("role") == "tool"]
        assert len(tool_msgs) >= 1

        await h2.agent.on_stop()
        await h2.pm.ahook.on_stop()

    async def test_c2_compaction_survives_restart(self, tmp_path):
        """C2. Compaction summary survives stop/restart."""
        db_path = str(tmp_path / "c2.db")
        # Small token budget to force compaction
        config = make_config(db_path=db_path, extra={
            "agent": {
                "system_prompt": "You are a test agent.",
                "max_context_tokens": 256,
            },
        })

        h1 = await _build_harness(config)

        # Build large messages to exceed 80% of 256 tokens (~205 chars of content)
        long_text = "x" * 300
        response_text = "y" * 300

        # Patch _summarize to return a canned summary without making an LLM call.
        # This keeps the side_effect list deterministic: one response per user message,
        # no ordering dependency on when compaction happens to trigger.
        async def fake_summarize(self, messages):
            return "test summary"

        # We need >5 messages to trigger compaction.
        # side_effect contains only normal conversation LLM responses (one per message).
        h1.mock_client.chat = AsyncMock(
            side_effect=[
                _make_text_response(response_text),  # msg 1
                _make_text_response(response_text),  # msg 2
                _make_text_response(response_text),  # msg 3
                _make_text_response(response_text),  # msg 4
                _make_text_response(response_text),  # msg 5
                _make_text_response(response_text),  # msg 6 (post-compaction)
            ]
        )

        with patch.object(CompactionPlugin, "_summarize", fake_summarize):
            for i in range(6):
                await h1.inject_message("test:c2", "user", long_text)
                await h1.drain_all()

        await h1.agent.on_stop()
        await h1.pm.ahook.on_stop()

        # Second harness: verify the conversation loaded is the compacted version
        h2 = await _build_harness(config)
        h2.mock_client.chat = AsyncMock(return_value=_make_text_response("ok"))

        await h2.inject_message("test:c2", "user", "continue")
        await h2.drain_all()

        chan = h2.registry.get("test:c2")
        assert chan is not None
        msgs = chan.conversation.messages

        # Compacted conversation should contain a summary message
        summary_msgs = [m for m in msgs if m.get("content", "").startswith("[Summary")]
        assert len(summary_msgs) >= 1

        await h2.agent.on_stop()
        await h2.pm.ahook.on_stop()


# ---------------------------------------------------------------------------
# Group D: Multi-Channel Isolation
# ---------------------------------------------------------------------------


class TestGroupDMultiChannel:

    async def test_d1_two_channels_independent_conversations(self, harness):
        """D1. Two channels maintain independent conversations and routing."""
        # Use return_value to avoid non-deterministic ordering issues
        harness.mock_client.chat = AsyncMock(
            return_value=_make_text_response("channel response")
        )

        await harness.inject_message("test:chan_a", "user", "hello from a")
        await harness.inject_message("test:chan_b", "user", "hello from b")
        await harness.drain_all()

        chan_a = harness.registry.get("test:chan_a")
        chan_b = harness.registry.get("test:chan_b")

        assert chan_a is not None
        assert chan_b is not None
        assert chan_a is not chan_b

        # Each has its own ContextWindow
        assert chan_a.conversation is not None
        assert chan_b.conversation is not None
        assert chan_a.conversation is not chan_b.conversation

        # Each has its own SerialQueue
        assert chan_a.id in harness.agent.queues
        assert chan_b.id in harness.agent.queues
        assert harness.agent.queues[chan_a.id] is not harness.agent.queues[chan_b.id]

        # Responses routed to correct channels
        assert len(harness.sent_messages) == 2
        sent_channels = [m[0] for m in harness.sent_messages]
        assert chan_a in sent_channels
        assert chan_b in sent_channels

    async def test_d2_per_channel_config_resolution(self, harness):
        """D2. Limited channel hits max_turns=1; default channel continues."""
        async def dummy_tool(x: str) -> str:
            """Dummy."""
            return "result"

        harness.agent._tools["dummy_tool"] = dummy_tool
        harness.agent._tool_schemas.append({
            "type": "function",
            "function": {
                "name": "dummy_tool",
                "description": "Dummy.",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                    "required": ["x"],
                },
            },
        })

        # Pre-register limited channel with max_turns=1
        limited_channel = harness.registry.get_or_create(
            "test", "d2_limited",
            config=ChannelConfig(max_turns=1),
        )
        # Default channel uses agent default max_turns (10)
        default_channel = harness.registry.get_or_create("test", "d2_default")

        # Use a side_effect function to route responses by inspecting message content.
        # Ordering between channels is non-deterministic, so we identify which channel
        # is calling by checking whether "limited_go" appears in the messages.
        async def chat_side_effect(messages, tools=None, extra_body=None):
            is_limited = any(
                m.get("content") == "limited_go"
                for m in messages
                if isinstance(m.get("content"), str)
            )
            if is_limited:
                # Limited channel: return tool call on first visit, then suppressed second visit
                call_count = chat_side_effect.limited_calls
                chat_side_effect.limited_calls += 1
                if call_count == 0:
                    # turn 0 < 1 → dispatch tool call (→ turn 1)
                    return _make_tool_call_response([_make_tool_call("lim1", "dummy_tool", {"x": "a"})])
                else:
                    # turn 1 < 1 is False → suppressed (fallback sent)
                    return _make_tool_call_response([_make_tool_call("lim2", "dummy_tool", {"x": "b"})])
            else:
                # Default channel: return tool call on first visit, then text after tool result
                call_count = chat_side_effect.default_calls
                chat_side_effect.default_calls += 1
                if call_count == 0:
                    # turn 0 < 10 → dispatch tool call (continues)
                    return _make_tool_call_response([_make_tool_call("def1", "dummy_tool", {"x": "c"})])
                else:
                    # After tool result → text response
                    return _make_text_response("default done")

        chat_side_effect.limited_calls = 0
        chat_side_effect.default_calls = 0

        harness.mock_client.chat = chat_side_effect

        await harness.pm.ahook.on_message(channel=limited_channel, sender="user", text="limited_go")
        await harness.pm.ahook.on_message(channel=default_channel, sender="user", text="default_go")
        await harness.drain_until_stable()

        # Limited channel sends fallback (max_turns suppressed)
        limited_msgs = [m for m in harness.sent_messages if m[0] is limited_channel]
        assert len(limited_msgs) == 1
        assert "max tool-calling rounds reached" in limited_msgs[0][1]

        # Default channel is NOT suppressed: tool dispatched and final text response sent
        default_msgs = [m for m in harness.sent_messages if m[0] is default_channel]
        assert len(default_msgs) == 1
        assert default_msgs[0][1] == "default done"

    async def test_d3_concurrent_compaction_and_tool_call(self):
        """D3. Compaction on channel A while channel B has an in-flight tool call.

        Channel A: accumulates enough messages (with small max_context_tokens) to
        trigger compaction. Channel B: sends a message that dispatches a gated tool
        (blocking). While channel B's tool is blocked, channel A receives another
        message that pushes it over the compaction threshold. After releasing channel
        B's gate, verify:
        - Channel A's compaction completed and a summary is present
        - Channel B's tool call completed normally (conversation not corrupted)
        - Neither channel's ContextWindow references the other's data
        """
        # Build a custom harness with calibrated max_context_tokens to trigger
        # compaction exactly at the 6th user message (phase 3), not before.
        #
        # Token math (token_estimate = int(total_chars / 3.5)):
        #   system_prompt = "You are a test agent." = 21 chars
        #   long_text = "a_msg_" + "x"*50 = 56 chars per user msg
        #   assistant response = "a response" = 10 chars per assistant msg
        #
        #   After 5 round-trips (10 msgs):
        #     total_chars = 21 + 5*56 + 5*10 = 351
        #     tokens = int(351 / 3.5) = 100
        #
        #   After 6th user msg appended (11 msgs, phase 3):
        #     total_chars = 21 + 6*56 + 5*10 = 407
        #     tokens = int(407 / 3.5) = 116
        #
        #   With max_context_tokens=140, threshold = 0.8*140 = 112:
        #     100 < 112 → no compaction after phase 1 ✓
        #     116 > 112 AND len(msgs)=11 > 5 → compaction fires in phase 3 ✓
        config = make_config(extra={
            "agent": {
                "system_prompt": "You are a test agent.",
                "max_context_tokens": 140,
            },
        })
        h = await _build_harness(config)

        gate = asyncio.Event()

        async def gated_tool(x: str) -> str:
            """Gated tool that blocks until gate is set."""
            await gate.wait()
            return "gated_result"

        h.agent._tools["gated_tool"] = gated_tool
        h.agent._tool_schemas.append({
            "type": "function",
            "function": {
                "name": "gated_tool",
                "description": "Gated tool that blocks until gate is set.",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                    "required": ["x"],
                },
            },
        })

        chan_a = h.registry.get_or_create("test", "d3_a")
        chan_b = h.registry.get_or_create("test", "d3_b")

        # Use a function side_effect because two channels are involved and
        # call ordering between them is non-deterministic.
        # Identify channel by presence of unique content markers in messages.
        async def chat_side_effect(messages, tools=None, extra_body=None):
            # Check for channel B marker ("b_msg") in any user message content.
            has_b_marker = any(
                "b_msg" in (m.get("content") or "")
                for m in messages
                if isinstance(m.get("content"), str)
            )
            if has_b_marker:
                call_n = chat_side_effect.b_calls
                chat_side_effect.b_calls += 1
                if call_n == 0:
                    # First call from B: dispatch gated tool
                    return _make_tool_call_response([
                        _make_tool_call("d3_gate", "gated_tool", {"x": "v"})
                    ])
                else:
                    # Second call from B: after tool result → text response
                    return _make_text_response("b after tool")
            else:
                # Channel A: always returns a simple text response.
                # Compaction is handled by fake_summarize, not the LLM mock.
                return _make_text_response("a response")

        chat_side_effect.b_calls = 0
        h.mock_client.chat = chat_side_effect

        long_text = "a_msg_" + ("x" * 50)

        async def fake_summarize(self_inner, messages):
            return "channel A summary"

        with patch.object(CompactionPlugin, "_summarize", fake_summarize):
            # Phase 1: Build up channel A's conversation history (5 round-trips).
            # After 5 user messages + 5 assistant responses, the token estimate
            # is 100 tokens, which is below the 112-token threshold (80% of 140).
            # The 6th message (phase 3) pushes it to 116 tokens and triggers compaction.
            for i in range(5):
                await h.pm.ahook.on_message(
                    channel=chan_a, sender="user", text=long_text
                )
                await channel_queue_drain(h, chan_a)

            # Phase 2: Send channel B's message and drain its queue so the tool
            # task is dispatched (queue slot freed) but the tool is still blocked.
            await h.pm.ahook.on_message(
                channel=chan_b, sender="user", text="b_msg trigger tool"
            )
            await channel_queue_drain(h, chan_b)
            # Yield to the event loop so the task worker starts consuming the
            # gated tool task before we move to phase 3.
            await asyncio.sleep(0)

            # At this point: chan_b's gated_tool is in flight, gate not set.
            # Phase 3: Trigger compaction on channel A by sending a 6th message.
            # This appends the user message, token_estimate exceeds threshold,
            # CompactionPlugin.compact_conversation runs (calls fake_summarize),
            # then the LLM is called for the 6th response.
            await h.pm.ahook.on_message(
                channel=chan_a, sender="user", text=long_text
            )
            await channel_queue_drain(h, chan_a)

            # Phase 4: Release the gate → channel B's tool completes → on_notify
            # enqueues the tool result → LLM called again → final text response.
            gate.set()
            await h.drain_all()

        # Verify channel A: summary is present in conversation
        msgs_a = chan_a.conversation.messages
        summary_msgs_a = [
            m for m in msgs_a
            if isinstance(m.get("content"), str)
            and m["content"].startswith("[Summary")
        ]
        assert len(summary_msgs_a) >= 1, (
            "Channel A should have a compaction summary after 6 large messages"
        )

        # Verify channel B: tool call completed normally
        msgs_b = chan_b.conversation.messages
        roles_b = [m["role"] for m in msgs_b]
        assert "tool" in roles_b, (
            "Channel B should have a tool result message after gate release"
        )
        # Last assistant message should be the post-tool text response
        assistant_msgs_b = [m for m in msgs_b if m["role"] == "assistant" and not m.get("tool_calls")]
        assert any(
            m.get("content") == "b after tool" for m in assistant_msgs_b
        ), "Channel B should have received the final 'b after tool' response"

        # Verify no cross-contamination: channel A has no channel B content, vice versa
        a_contents = [
            m.get("content", "") for m in msgs_a
            if isinstance(m.get("content"), str)
        ]
        b_contents = [
            m.get("content", "") for m in msgs_b
            if isinstance(m.get("content"), str)
        ]
        assert not any("b_msg" in c for c in a_contents), (
            "Channel A's conversation should not contain channel B's messages"
        )
        assert not any("a_msg_" in c for c in b_contents), (
            "Channel B's conversation should not contain channel A's messages"
        )

        await h.agent.on_stop()
        await h.pm.ahook.on_stop()


# ---------------------------------------------------------------------------
# Group E: Error Handling
# ---------------------------------------------------------------------------


class TestGroupEErrorHandling:

    async def test_e1_llm_error_mid_conversation(self, harness):
        """E1. LLM raises ClientResponseError → error sent, subsequent messages work."""
        # Simulate an aiohttp error
        request_info = MagicMock()
        request_info.real_url = "http://fake"

        harness.mock_client.chat = AsyncMock(
            side_effect=[
                aiohttp.ClientResponseError(request_info, (), status=500),
                _make_text_response("I am back!"),
            ]
        )

        # First message: LLM errors
        await harness.inject_message("test:e1", "user", "hello")
        await harness.drain_all()

        # Error message sent to channel
        assert len(harness.sent_messages) == 1
        assert "error" in harness.sent_messages[0][1].lower() or \
               "sorry" in harness.sent_messages[0][1].lower()

        # Second message: conversation not corrupted
        await harness.inject_message("test:e1", "user", "try again")
        await harness.drain_all()

        assert len(harness.sent_messages) == 2
        assert "I am back!" in harness.sent_messages[1][1]

        # Conversation log should not have a corrupt assistant message
        chan = harness.registry.get("test:e1")
        assert chan is not None
        for msg in chan.conversation.messages:
            if msg.get("role") == "assistant":
                # Should not be empty/corrupt
                content = msg.get("content", "")
                assert content is not None

    async def test_e2_tool_exception_in_integrated_context(self, harness):
        """E2. Tool raises RuntimeError → error delivered via on_notify → LLM continues."""
        async def exploding_tool(x: str) -> str:
            """A tool that always explodes."""
            raise RuntimeError("tool explosion")

        harness.agent._tools["exploding_tool"] = exploding_tool
        harness.agent._tool_schemas.append({
            "type": "function",
            "function": {
                "name": "exploding_tool",
                "description": "Explodes.",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                    "required": ["x"],
                },
            },
        })

        harness.mock_client.chat = AsyncMock(
            side_effect=[
                _make_tool_call_response([_make_tool_call("e2c1", "exploding_tool", {"x": "boom"})]),
                _make_text_response("I see the tool failed."),
            ]
        )

        await harness.inject_message("test:e2", "user", "use exploding tool")
        await harness.drain_until_stable()

        # Task completed with error string
        chan = harness.registry.get("test:e2")
        tool_msgs = [m for m in chan.conversation.messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert "error" in tool_msgs[0]["content"].lower()

        # LLM was called again with error in tool result
        assert harness.mock_client.chat.call_count == 2

        # Final response sent
        assert len(harness.sent_messages) == 1
        assert "I see the tool failed." in harness.sent_messages[0][1]

    async def test_e3_missing_task_plugin_degrades_gracefully(self, caplog):
        """E3. No TaskPlugin → error logged, no crash, subsequent text messages work."""
        pm = create_plugin_manager()
        agent_defaults = {"system_prompt": "test", "max_context_tokens": 4096}
        registry = ChannelRegistry(agent_defaults)
        pm.register(registry, name="registry")

        persistence_plugin = PersistencePlugin(pm)
        pm.register(persistence_plugin, name="persistence")

        core_tools = CoreToolsPlugin(pm)
        pm.register(core_tools, name="core_tools")

        fake_transport = FakeTransportPlugin(pm)
        pm.register(fake_transport, name="fake_transport")

        # No TaskPlugin

        mock_client = AsyncMock(spec=LLMClient)
        mock_client.start = AsyncMock()
        mock_client.stop = AsyncMock()

        llm_plugin = LLMPlugin(pm)
        pm.register(llm_plugin, name="llm")

        compaction_plugin = CompactionPlugin(pm=pm)
        pm.register(compaction_plugin, name="compaction")

        thinking_plugin = ThinkingPlugin(pm)
        pm.register(thinking_plugin, name="thinking")

        from corvidae.tool_collection import ToolCollectionPlugin
        tool_collection_plugin = ToolCollectionPlugin(pm)
        pm.register(tool_collection_plugin, name="tools")

        agent_loop = Agent(pm)
        pm.register(agent_loop, name="agent")

        async def tool_raiser(x: str) -> str:
            """Raises."""
            raise RuntimeError("no task plugin")

        config = make_config()
        with patch("corvidae.llm_plugin.LLMClient", return_value=mock_client):
            await pm.ahook.on_start(config=config)
            await agent_loop.on_start(config=config)

        # Inject a tool that would need TaskPlugin
        agent_loop._tools["test_tool"] = tool_raiser

        mock_client.chat = AsyncMock(
            side_effect=[
                _make_tool_call_response([_make_tool_call("e3c1", "test_tool", {"x": "v"})]),
                _make_text_response("text only"),  # fallback for text-only
            ]
        )

        channel = registry.get_or_create("test", "e3")

        with caplog.at_level(logging.ERROR, logger="corvidae.agent"):
            await pm.ahook.on_message(channel=channel, sender="user", text="use tool")
            await agent_loop.queues[channel.id].drain()

        # Error logged about missing task queue
        error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert any(
            "task" in r.message.lower() or "queue" in r.message.lower()
            for r in error_records
        )

        # No crash; subsequent text-only message works
        mock_client.chat = AsyncMock(return_value=_make_text_response("text only"))
        await pm.ahook.on_message(channel=channel, sender="user", text="just text")
        await agent_loop.queues[channel.id].drain()

        assert len(fake_transport.sent) == 1

        await agent_loop.on_stop()
        await pm.ahook.on_stop()


# ---------------------------------------------------------------------------
# Group F: Plugin Tool Collection
# ---------------------------------------------------------------------------


class TestGroupFToolCollection:

    async def test_f1_all_expected_tools_registered(self, harness):
        """F1. After on_start with full plugin graph, expected tool set is exact."""
        expected = {"shell", "read_file", "write_file", "web_fetch", "web_search", "subagent", "task_status", "task_pipeline"}
  assert set(harness.agent.tools.keys()) == expected

    async def test_f2_tool_schemas_structurally_valid(self, harness):
        """F2. Each tool schema has correct structure and no _ctx in parameters."""
        tool_names = set(harness.agent.tools.keys())

        for schema in harness.agent.tool_schemas:
            assert schema["type"] == "function", f"Schema missing type=function: {schema}"
            fn = schema["function"]
            assert "name" in fn, f"Schema missing function.name: {schema}"
            assert fn["name"] in tool_names, \
                f"Schema name {fn['name']!r} not in tools dict"
            assert "parameters" in fn, f"Schema missing function.parameters: {schema}"
            params = fn["parameters"]
            assert params.get("type") == "object", \
                f"Parameters type is not 'object': {schema}"
            properties = params.get("properties", {})
            assert "_ctx" not in properties, \
                f"_ctx leaked into parameters.properties for {fn['name']!r}"
