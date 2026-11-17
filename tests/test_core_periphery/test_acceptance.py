"""Phase 1 acceptance probe (issue #1050): limits.max_waiting end to end.

Real Agent, TaskPlugin, SubagentPlugin and LimitsPlugin. The subagent call
goes through ``Agent._dispatch_tool_calls``, as a model's tool call does: it
runs as a carrier Task on the TaskQueue, offers its work at the seam, and its
result comes back through TaskPlugin and ``on_notify``. With waiting work at
the ceiling the start is refused with the tool-result text and a warning, and
that task's completion, a tool result at the same depth, still reaches the
channel's queue. Only the model turn itself is replaced (a recorder stands in
for ``_process_queue_item``), because what is under test is admission and
delivery, not generation.
"""

import asyncio
import json
import logging
from unittest.mock import MagicMock

from corvidae.llm_plugin import LLMPlugin
from corvidae.task import Task
from corvidae.tool import ToolRegistry
from corvidae.tool_collection import ToolCollectionPlugin

from helpers import KERNEL_LLM_STAGES, build_plugin_and_channel, stub_llm_config

CONFIG = {
    "limits": {"max_waiting": 2},
    "llm": {
        "providers": {"main": {"base_url": "http://localhost:8080"}},
        "profiles": {"main": {"model": "test-model"}},
    },
}


async def _until(predicate, timeout=5.0):
    """Yield to the loop until ``predicate()`` holds, bounded by ``timeout``."""

    async def _spin():
        while not predicate():
            await asyncio.sleep(0)

    await asyncio.wait_for(_spin(), timeout)


async def test_task_start_refused_but_tool_result_accepted_at_max_waiting(
    monkeypatch, caplog
):
    """At waiting work == max_waiting the subagent tool returns the refusal
    text and logs a warning; an on_notify tool result at that depth is
    enqueued and processed."""
    from corvidae.tools.subagent import SubagentPlugin
    from corvidae_core_periphery.limits import LimitsPlugin
    from fake_clock import install_kernel_clock

    agent, channel, db = await build_plugin_and_channel()
    pm = agent.pm
    task_plugin = pm.get_plugin("task")
    task_queue = task_plugin.task_queue
    clock, idle = install_kernel_clock(pm)
    release = asyncio.Event()
    processed = []

    async def record(item):
        # Stand in for the model turn: record what reached the queue consumer.
        processed.append(item)

    monkeypatch.setattr(agent, "_process_queue_item", record)

    # The kernel pieces the subagent tool needs: a stub LLM and the tool set.
    llm = LLMPlugin(pm)
    llm._stages = dict(KERNEL_LLM_STAGES)
    llm._clients["main"] = MagicMock(name="main")
    stub_llm_config(llm)
    pm.register(llm, name="llm")
    tools = ToolCollectionPlugin(pm)
    tools.registry = ToolRegistry()
    pm.register(tools, name="tools")
    subagent = SubagentPlugin(pm)
    pm.register(subagent, name="subagent")
    await subagent.on_start(config=CONFIG)
    tool_list: list = []
    subagent.register_tools(tool_registry=tool_list)
    # The Agent dispatches tools from its own table, as after refresh.
    agent._tools = {t.name: t.fn for t in tool_list}

    limits = LimitsPlugin()
    pm.register(limits, name="limits")
    await limits.on_init(pm=pm, config=CONFIG)
    await limits.on_start(config=CONFIG)

    async def blocked_work():
        # Hold a worker until released so the task stays waiting work.
        await release.wait()
        return "done"

    try:
        # Waiting work reaches max_waiting: two tasks parked on workers.
        for i in range(2):
            await task_queue.enqueue(
                Task(work=blocked_work, channel=channel, description=f"t{i}")
            )
        await _until(lambda: agent.waiting_work() == {channel.id: 2})

        # The agent's tool path: the model's subagent call runs as a carrier
        # Task and is refused at the ceiling.
        call = {
            "id": "call-sub",
            "function": {
                "name": "subagent",
                "arguments": json.dumps({"instructions": "go", "description": "d"}),
            },
        }
        with caplog.at_level(logging.WARNING):
            await agent._dispatch_tool_calls([call], channel)
            # The carrier's completion is delivered at the same depth: the
            # two parked tasks still hold waiting work at max_waiting.
            await _until(lambda: any(i.tool_call_id == "call-sub" for i in processed))
        delivered = next(i for i in processed if i.tool_call_id == "call-sub")
        assert delivered.content.startswith("Not started: limits.max_waiting reached")
        assert any(
            r.levelno == logging.WARNING
            and getattr(r, "limit", None) == "limits.max_waiting"
            and getattr(r, "kind", None) == "task"
            for r in caplog.records
        )
        # The refused start enqueued nothing: only the two parked tasks wait.
        assert agent.waiting_work() == {channel.id: 2}
    finally:
        # Release the parked tasks, then stop everything that holds a task.
        release.set()
        for q in agent.queues.values():
            await q.stop()
        idle.stop()
        await clock.close()
        await task_plugin.on_stop()
        await db.close()
