"""Subagent tool for the sherman agent daemon."""

import logging

from sherman.agent_loop import run_agent_loop, strip_thinking
from sherman.hooks import hookimpl
from sherman.llm import LLMClient
from sherman.task import Task
from sherman.tool import ToolContext

logger = logging.getLogger(__name__)

SUBAGENT_SYSTEM_PROMPT = (
    "You are executing a background task. "
    "Work through the instructions step by step. Be thorough. "
    "When finished, provide a clear summary of what you accomplished."
)


class SubagentPlugin:
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
        from sherman.tool import Tool

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
        task_queue = ctx.task_queue
        if task_queue is None:
            return "Error: task queue not available"

        channel = ctx.channel
        if channel is None:
            return "Error: no channel context available for subagent"

        # Get tools, excluding subagent itself to prevent recursion and
        # background_task which is a placeholder that raises RuntimeError
        agent = self.pm.agent_plugin
        registry = agent.tool_registry.exclude("subagent", "background_task")
        tools_dict = registry.as_dict()
        tool_schemas = registry.schemas()

        llm_cfg = self._llm_config

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
            )
            await client.start()
            try:
                result = await run_agent_loop(
                    client, messages, tools_dict, tool_schemas,
                    channel=channel,
                    task_queue=task_queue,
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
