"""Phase 1 acceptance probe (issue #1050): limits.max_waiting end to end.

Real Agent, TaskPlugin, SubagentPlugin and LimitsPlugin. With waiting work at
the ceiling, a task start is refused with the tool-result text and a warning,
while a tool result delivered at the same depth is accepted.
"""

import asyncio
import json
import logging
from unittest.mock import MagicMock

from corvidae.llm_plugin import LLMPlugin
from corvidae.task import Task
from corvidae.tool import ToolRegistry, dispatch_tool_call
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
    tool_fns = {t.name: t.fn for t in tool_list}

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

        # The agent's tool path: a subagent start is refused at the ceiling.
        call = {
            "id": "call-sub",
            "function": {
                "name": "subagent",
                "arguments": json.dumps({"instructions": "go", "description": "d"}),
            },
        }
        with caplog.at_level(logging.WARNING):
            result = await dispatch_tool_call(
                call, tool_fns, channel=channel, task_queue=task_queue, pm=pm
            )
        assert result.content.startswith("Not started: limits.max_waiting reached")
        assert any(
            r.levelno == logging.WARNING and "limits.max_waiting" in r.getMessage()
            + str(getattr(r, "limit", ""))
            for r in caplog.records
        )
        # Nothing was enqueued by the refused start: still just the two tasks.
        assert agent.waiting_work() == {channel.id: 2}

        # A tool result at the same depth is never refused: it reaches the queue.
        channel.pending_tool_call_ids.add("call-pending")
        await pm.ahook.on_notify(
            channel=channel,
            source="task",
            text="r",
            tool_call_id="call-pending",
            annotations=None,
        )
        await _until(lambda: any(i.content == "r" for i in processed))
        delivered = next(i for i in processed if i.content == "r")
        assert delivered.tool_call_id == "call-pending"
    finally:
        # Release the parked tasks, then stop everything that holds a task.
        release.set()
        for q in agent.queues.values():
            await q.stop()
        idle.stop()
        await clock.close()
        await task_plugin.on_stop()
        await db.close()
