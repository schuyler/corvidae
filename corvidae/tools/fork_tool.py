"""Fork tool — a background task that is *me*, fresh and focused.

Unlike the subagent (a blank-context task with a generic system prompt) and
the view fork (``corvidae/fork.py``, a single no-tools KV-prefix call), the
fork tool runs the multi-turn agent loop with **the agent's own system prompt**
— my identity, standards, voice — in a **fresh** window (no history from the
invoking channel), with my query as the first user turn and the full tool set.
It is "me, before I started writing this": same judgment, a different vantage.

The freshness is the point: the messages list is built from my prompt plus the
query, with no channel history, so the fork doesn't carry the frame of the
conversation that produced the thing it's reviewing. It reuses the subagent's
``run_agent_loop`` (multi-turn + tools) and Task enqueue (background,
non-blocking); the only difference from the subagent is the system prompt.

Design note: docs/fork-tool.md.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from corvidae.attribution import reset_attribution, set_attribution
from corvidae.hooks import CorvidaePlugin, get_dependency, hookimpl
from corvidae.task import Task
from corvidae.thinking import strip_thinking
from corvidae.tool import (
    DEFAULT_TOOL_TIMEOUTS,
    MAX_TOOL_RESULT_CHARS,
    ToolContext,
)
from corvidae.tools.subagent import (
    parse_subagent_max_turns,
    run_agent_loop,
)

if TYPE_CHECKING:
    from corvidae.channel import Channel
    from corvidae.task import TaskQueue

logger = logging.getLogger(__name__)


class ForkToolPlugin(CorvidaePlugin):
    """Registers the ``fork`` tool: a background task that is me, fresh.

    Mirrors SubagentPlugin, but the system prompt is the agent's own identity
    (via ``Agent.channel_system_prompt``), not a generic subagent prompt. The
    fork runs the subagent's multi-turn loop with the full tool set (minus the
    fork tool itself, to prevent recursion) and is enqueued as a background
    Task, result delivered via TaskPlugin -> on_notify -> Agent.

    Depends on "llm" (shared client), "tools" (registry), and "agent"
    (the system prompt).
    """

    depends_on = frozenset({"llm", "tools", "agent"})

    def __init__(self, pm=None) -> None:
        if pm is not None:
            self.pm = pm

    @hookimpl
    async def on_start(self, config: dict) -> None:
        # Reuse the subagent's tool-round budget: a fork is a subagent variant.
        self._max_turns = parse_subagent_max_turns(config)
        logger.debug("ForkToolPlugin started")

    @hookimpl
    def register_llm_stages(self, stages: dict[str, str]) -> None:
        stages["fork"] = "chat"

    @hookimpl
    def register_tools(self, tool_registry: list) -> None:
        from corvidae.tool import Tool

        plugin = self  # captured by closure

        async def fork(task: str, _ctx: ToolContext) -> str:
            """Fork a fresh instance of me to work a task in the background.

            Unlike the subagent (a blank agent), the fork runs with my own
            system prompt — my identity and standards — in a fresh context, so
            it has my judgment with fresh eyes. Use it for focused second-
            opinion work: reviewing a draft, fact-checking claims, re-deriving
            a decision.
            """
            return await plugin._launch(task, _ctx)

        tool_registry.append(Tool.from_function(fork))

    async def _launch(self, task: str, ctx: ToolContext) -> str:
        """Build and enqueue a Task that runs a fork (me, fresh) loop.

        The messages list is ``[my prompt, the task]`` — a fresh window with no
        channel history. That's the freshness that gives the fork fresh eyes.
        """
        task_queue = ctx.task_queue
        if task_queue is None:
            return (
                "Error: task queue not available — ensure the task plugin is "
                "loaded, then retry."
            )

        channel = ctx.channel
        if channel is None:
            return (
                "Error: no channel context available for fork — call this tool "
                "from an agent turn that has a channel, then retry."
            )

        # My system prompt (the agent's identity), not a generic subagent prompt.
        from corvidae.agent import Agent

        agent = get_dependency(self.pm, "agent", Agent)
        my_prompt = await agent.channel_system_prompt(channel)

        # The full tool set, minus the fork tool itself (no recursion) and
        # minus any the calling channel has disabled.
        from corvidae.tool_policy import disabled_on
        from corvidae.tool_collection import ToolCollectionPlugin
        from corvidae.llm_plugin import LLMPlugin

        tools_plugin = get_dependency(self.pm, "tools", ToolCollectionPlugin)
        registry = tools_plugin.get_registry().exclude("fork", *disabled_on(channel))
        max_result_chars = tools_plugin.max_result_chars
        call_timeouts = tools_plugin.call_timeouts
        tools_dict = registry.as_dict()
        tool_schemas = registry.schemas()

        # The shared fork-stage client from LLMPlugin (do NOT call start/stop).
        llm = get_dependency(self.pm, "llm", LLMPlugin)
        client = llm.get_client("fork")

        # Fresh window: my prompt + the task. No channel history.
        messages = [
            {"role": "system", "content": my_prompt},
            {"role": "user", "content": task},
        ]

        async def work():
            # Attribute the fork's LLM calls to the fork stage.
            attribution_token = set_attribution(stage="fork", channel_id=channel.id)
            try:
                result = await run_agent_loop(
                    client, messages, tools_dict, tool_schemas,
                    channel=channel,
                    task_queue=task_queue,
                    max_turns=self._max_turns,
                    max_result_chars=max_result_chars,
                    call_timeouts=call_timeouts,
                )
            finally:
                reset_attribution(attribution_token)
            return strip_thinking(result)

        fork_task = Task(
            work=work,
            channel=channel,
            tool_call_id=ctx.tool_call_id,
            description=f"fork: {task[:80]}",
        )
        await task_queue.enqueue(fork_task)
        return f"Fork task {fork_task.task_id} enqueued: {task[:80]}"
