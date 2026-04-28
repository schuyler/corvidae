"""Subagent tool for the corvidae agent daemon."""

from __future__ import annotations

import logging

from corvidae.agent_loop import run_agent_loop, strip_thinking
from corvidae.hooks import get_dependency, hookimpl
from corvidae.llm import LLMClient
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

    Launches background agents via run_agent_loop. Each subagent gets its own
    LLMClient (using llm.background if configured, otherwise llm.main) and the
    full tool set minus the subagent tool itself (to prevent recursion).

    Depends on the "agent_loop" plugin (AgentPlugin) to read tool_registry
    and _max_tool_result_chars at tool-call time.
    """

    depends_on = {"agent_loop"}

    def __init__(self, pm) -> None:
        self.pm = pm
        self._llm_config: dict | None = None

    @hookimpl
    async def on_start(self, config: dict) -> None:
        llm_config = config.get("llm", {})
        self._llm_config = llm_config.get("background") or llm_config["main"]
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

        Retrieves AgentPlugin to read tool_registry and _max_tool_result_chars,
        excludes the subagent tool from the set, creates a fresh LLMClient inside
        the work closure, and enqueues the task. Returns immediately; the result
        is delivered via TaskPlugin -> on_notify -> AgentPlugin.

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
        from corvidae.agent import AgentPlugin
        agent = get_dependency(self.pm, "agent_loop", AgentPlugin)
        tool_registry, max_result_chars = agent.get_tool_config()
        registry = tool_registry.exclude("subagent")
        tools_dict = registry.as_dict()
        tool_schemas = registry.schemas()

        llm_cfg = self._llm_config
        plugin = self

        messages = [
            {"role": "system", "content": SUBAGENT_SYSTEM_PROMPT},
            {"role": "user", "content": instructions},
        ]

        async def work():
            client = LLMClient(
                base_url=llm_cfg["base_url"],
                model=llm_cfg["model"],
                api_key=llm_cfg.get("api_key"),
                extra_body=llm_cfg.get("extra_body"),
                max_retries=llm_cfg.get("max_retries", 3),
                retry_base_delay=llm_cfg.get("retry_base_delay", 2.0),
                retry_max_delay=llm_cfg.get("retry_max_delay", 60.0),
                timeout=llm_cfg.get("timeout"),
            )
            await client.start()
            try:
                result = await run_agent_loop(
                    client, messages, tools_dict, tool_schemas,
                    channel=channel,
                    task_queue=task_queue,
                    max_result_chars=max_result_chars,
                )
                return strip_thinking(result)
            finally:
                await client.stop()

        task = Task(
            work=work,
            channel=channel,
            tool_call_id=ctx.tool_call_id,
            description=description,
        )
        await task_queue.enqueue(task)
        return f"Subagent task {task.task_id} enqueued: {description}"
