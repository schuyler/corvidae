"""Nested background agents are bounded by depth (#816).

The bound is kernel attribution: ``agent_depth`` lives in the attribution
dict, which snapshots into every Task created under it, so a subagent that
calls ``fork`` (or a fork that calls ``subagent``) is refused at
``tools.max_agent_depth`` instead of running forever.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from corvidae.agent import Agent
from corvidae.attribution import (
    agent_depth,
    agent_depth_refusal,
    next_agent_depth,
    reset_attribution,
    set_attribution,
)
from corvidae.channel import Channel, ChannelConfig, ChannelRegistry
from corvidae.hooks import create_plugin_manager
from corvidae.llm_plugin import LLMPlugin
from corvidae.task import Task, TaskQueue
from corvidae.tool import Tool, ToolContext, ToolRegistry
from corvidae.tool_collection import (
    DEFAULT_MAX_AGENT_DEPTH,
    ToolCollectionPlugin,
    parse_max_agent_depth,
)
from corvidae.tools import fork_tool, subagent
from corvidae.tools.fork_tool import ForkToolPlugin
from corvidae.tools.subagent import SubagentPlugin

from helpers import KERNEL_LLM_STAGES, stub_llm_config

CONFIG = {"llm": {"main": {"base_url": "http://localhost:8080", "model": "test-model"}}}
MY_PROMPT = "I am the agent."


def _channel() -> Channel:
    return Channel(transport="test", scope="scope1", config=ChannelConfig())


def _pm():
    """A plugin manager with the dependencies a spawning tool needs."""
    pm = create_plugin_manager()
    pm.register(ChannelRegistry({}), name="registry")

    llm = LLMPlugin(pm)
    llm._stages = dict(KERNEL_LLM_STAGES)
    llm._stages["fork"] = "chat"
    llm._clients["main"] = MagicMock(name="main")
    stub_llm_config(llm)
    pm.register(llm, name="llm")

    agent = Agent.__new__(Agent)
    agent.channel_system_prompt = AsyncMock(return_value=MY_PROMPT)
    pm.register(agent, name="agent")

    registry = ToolRegistry()
    for name in ("shell", "subagent", "fork"):
        async def _placeholder() -> str:
            """Placeholder tool."""
            return "ok"

        _placeholder.__name__ = name
        registry.add(Tool(name=name, fn=_placeholder, schema={}))

    # Registry set directly, as the tool tests do: on_start would rebuild it.
    tools = ToolCollectionPlugin(pm)
    tools.registry = registry
    pm.register(tools, name="tools")
    return pm


async def _started(cls, max_depth: int = DEFAULT_MAX_AGENT_DEPTH):
    pm = _pm()
    plugin = cls(pm)
    await plugin.on_start(config=CONFIG)
    # on_init (which parses the key) is not run here; set what it would set.
    pm.get_plugin("tools").max_agent_depth = max_depth
    return plugin


def _ctx(queue) -> ToolContext:
    return ToolContext(channel=_channel(), tool_call_id="call-1", task_queue=queue)


def _queue() -> MagicMock:
    queue = MagicMock(spec=TaskQueue)
    queue.enqueue = AsyncMock()
    return queue


# ---------------------------------------------------------------------------
# The config key
# ---------------------------------------------------------------------------


class TestConfigKey:
    def test_default_is_two_levels(self):
        assert DEFAULT_MAX_AGENT_DEPTH == 2
        assert parse_max_agent_depth({}) == 2
        assert parse_max_agent_depth(None) == 2

    @pytest.mark.parametrize("value", [1, 2, 5])
    def test_accepts_a_positive_integer(self, value):
        assert parse_max_agent_depth({"max_agent_depth": value}) == value

    @pytest.mark.parametrize("value", [0, -1, 1.5, "2", True, None])
    def test_rejects_anything_else_naming_key_value_and_rule(self, value):
        with pytest.raises(ValueError) as excinfo:
            parse_max_agent_depth({"max_agent_depth": value})
        message = str(excinfo.value)
        assert "tools.max_agent_depth" in message
        assert repr(value) in message
        assert "integer of at least 1" in message


# ---------------------------------------------------------------------------
# The kernel helper
# ---------------------------------------------------------------------------


class TestDepthHelper:
    def test_a_main_turn_is_depth_zero_and_may_spawn(self):
        assert agent_depth() == 0
        assert next_agent_depth(2) == 1

    def test_the_limit_refuses(self):
        token = set_attribution(agent_depth=2)
        try:
            assert next_agent_depth(2) is None
        finally:
            reset_attribution(token)

    def test_one_level_allows_exactly_one(self):
        assert next_agent_depth(1) == 1
        token = set_attribution(agent_depth=1)
        try:
            assert next_agent_depth(1) is None
        finally:
            reset_attribution(token)

    def test_a_junk_depth_reads_as_zero(self):
        # Attribution is a plain dict plugins may write; a bad value must not
        # crash a spawn or be read as "already deep".
        token = set_attribution(agent_depth=True)
        try:
            assert agent_depth() == 0
        finally:
            reset_attribution(token)

    def test_refusal_names_the_limit_the_key_and_the_remedy(self):
        assert agent_depth_refusal(2) == (
            "Error: nested agent depth 2 reached (tools.max_agent_depth); "
            "do this work in the current task instead."
        )


# ---------------------------------------------------------------------------
# Both spawning tools honour it
# ---------------------------------------------------------------------------


class TestSpawningToolsRefuse:
    async def test_subagent_refuses_at_the_limit_and_enqueues_nothing(self):
        plugin = await _started(SubagentPlugin, max_depth=1)
        queue = _queue()
        token = set_attribution(agent_depth=1)
        try:
            result = await plugin._launch("do something", "desc", _ctx(queue))
        finally:
            reset_attribution(token)
        assert result == agent_depth_refusal(1)
        queue.enqueue.assert_not_awaited()

    async def test_fork_refuses_at_the_limit_and_enqueues_nothing(self):
        plugin = await _started(ForkToolPlugin, max_depth=2)
        queue = _queue()
        token = set_attribution(agent_depth=2)
        try:
            result = await plugin._launch("do something", _ctx(queue))
        finally:
            reset_attribution(token)
        assert result == agent_depth_refusal(2)
        queue.enqueue.assert_not_awaited()

    @pytest.mark.parametrize(
        "cls,module", [(SubagentPlugin, subagent), (ForkToolPlugin, fork_tool)],
        ids=["subagent", "fork"],
    )
    async def test_a_main_turn_may_spawn_and_the_spawned_loop_is_one_deep(
        self, cls, module, monkeypatch
    ):
        """The spawned work() attributes the depth the helper handed it, so a
        tool called from inside that loop asks for depth 2."""
        plugin = await _started(cls, max_depth=2)
        queue = _queue()

        seen: list[int] = []

        async def fake_loop(*args, **kwargs):
            seen.append(agent_depth())
            return "done"

        monkeypatch.setattr(module, "run_agent_loop", fake_loop)

        if cls is SubagentPlugin:
            result = await plugin._launch("do something", "desc", _ctx(queue))
        else:
            result = await plugin._launch("do something", _ctx(queue))

        assert "enqueued" in result
        task = queue.enqueue.call_args[0][0]
        assert isinstance(task, Task)
        await task.work()

        assert seen == [1], "the spawned loop must run at depth 1"
        assert agent_depth() == 0, "the main turn's depth is untouched"

    async def test_a_task_carries_the_depth_it_was_created_in(self):
        """A Task made inside a spawned loop carries its depth, so a spawn from
        there is the next level — the queue worker runs work in that context."""

        async def spawn() -> int | None:
            return next_agent_depth(2)

        token = set_attribution(agent_depth=2)
        try:
            task = Task(work=spawn, channel=_channel())
        finally:
            reset_attribution(token)

        holder: dict = {}
        task.ctx.run(lambda: holder.setdefault("t", asyncio.create_task(spawn())))
        assert await holder["t"] is None, "depth 2 is the limit; the next spawn is refused"
