"""Subagent tool for the corvidae agent daemon."""

from __future__ import annotations

import logging

from corvidae.agent_loop import run_agent_loop, strip_thinking
from corvidae.hooks import get_dependency, hookimpl
from corvidae.task import Task
from corvidae.tool import ToolContext

logger = logging.getLogger(__name__)

SUBAGENT_SYSTEM_PROMPT = (
    "You are executing a background task. "
    "Work through the instructions step by step. Be thorough. "
    "When finished, provide a clear summary of what you accomplished."
)


class SubagentPlugin:
    """Plugin that registers the subagent tool.

    Launches background agents via run_agent_loop. All subagents share the
    background LLMClient from LLMPlugin (using llm.background if configured,
    otherwise llm.main) and the full tool set minus the subagent tool itself
    (to prevent recursion).

    Depends on "llm" (LLMPlugin) to access the shared background LLM client,
    and on "tools" (ToolCollectionPlugin) to read the tool registry and
    max_result_chars at tool-call time.
    """

    depends_on = {"llm", "tools"}

    def __init__(self, pm) -> None:
        self.pm = pm

    @hookimpl
    async def on_start(self, config: dict) -> None:
        logger.debug("SubagentPlugin started")

    @hookimpl
    def register_tools(self, tool_registry: list) -> None:
        from corvidae.tool import Tool

        plugin = self  # captured by closure

        async def subagent(
            instructions: str, description: str, _ctx: ToolContext
        ) -> str:
            """Launch a subagent to work on a task in the background."""
            return await plugin._launch(instructions, description, _ctx)

        tool_registry.append(Tool.from_function(subagent))

    async def _launch(
        self, instructions: str, description: str, ctx: ToolContext
    ) -> str:
        """Build and enqueue a Task that runs a subagent loop.

        Retrieves ToolCollectionPlugin to read tool_registry and max_result_chars,
        excludes the subagent tool from the set, uses the shared background client
        from LLMPlugin, and enqueues the task. Returns immediately; the result
        is delivered via TaskPlugin -> on_notify -> Agent.

        Args:
            instructions: The user-provided instructions for the subagent.
            description: Human-readable task description (used in Task.description).
            ctx: Injected ToolContext providing channel and task_queue.

        Returns:
            Confirmation string with the enqueued task ID, or an error string
            if the task queue or channel context is unavailable.
        """
        task_queue = ctx.task_queue
        if task_queue is None:
            return "Error: task queue not available"

        channel = ctx.channel
        if channel is None:
            return "Error: no channel context available for subagent"

        # Get tools, excluding subagent itself to prevent recursion
        from corvidae.tool_collection import ToolCollectionPlugin
        from corvidae.llm_plugin import LLMPlugin
        tools_plugin = get_dependency(self.pm, "tools", ToolCollectionPlugin)
        registry = tools_plugin.get_registry().exclude("subagent")
        max_result_chars = tools_plugin.max_result_chars
        tools_dict = registry.as_dict()
        tool_schemas = registry.schemas()

        # Use the shared background client from LLMPlugin (do NOT call start/stop).
        llm = get_dependency(self.pm, "llm", LLMPlugin)
        client = llm.get_client("background")

        messages = [
            {"role": "system", "content": SUBAGENT_SYSTEM_PROMPT},
            {"role": "user", "content": instructions},
        ]

        async def work():
            result = await run_agent_loop(
                client, messages, tools_dict, tool_schemas,
                channel=channel,
                task_queue=task_queue,
                max_result_chars=max_result_chars,
            )
            return strip_thinking(result)

        task = Task(
            work=work,
            channel=channel,
            tool_call_id=ctx.tool_call_id,
            description=description,
        )
        await task_queue.enqueue(task)
        return f"Subagent task {task.task_id} enqueued: {description}"
