"""Subagent tool for the corvidae agent daemon.

Owns ``run_agent_loop`` — the LLM-call-plus-tool-dispatch loop used to drive
background subagents to completion. The main agent (``corvidae.agent``) does
not use this loop; it re-enters its serial queue between LLM calls and tool
results so user messages can interleave mid-cycle. Subagents block on the
loop because they execute as a single Task on the task queue.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

from corvidae.turn import run_agent_turn
from corvidae.hooks import CorvidaePlugin, get_dependency, hookimpl
from corvidae.llm import LLMClient
from corvidae.task import Task
from corvidae.thinking import strip_thinking
from corvidae.tool import MAX_TOOL_RESULT_CHARS, ToolContext, dispatch_tool_call

if TYPE_CHECKING:
    from corvidae.channel import Channel
    from corvidae.task import TaskQueue

logger = logging.getLogger(__name__)

# Note: same text as MAX_TURNS_FALLBACK_MESSAGE in agent.py — kept in sync.
MAX_ROUNDS_REACHED_MESSAGE = "(max tool-calling rounds reached)"

SUBAGENT_SYSTEM_PROMPT = (
    "You are executing a background task. "
    "Work through the instructions step by step. Be thorough. "
    "When finished, provide a clear summary of what you accomplished."
)


async def run_agent_loop(
    client: LLMClient,
    messages: list[dict],
    tools: dict[str, Callable],
    tool_schemas: list[dict],
    max_turns: int = 10,
    *,
    channel: "Channel | None" = None,
    task_queue: "TaskQueue | None" = None,
    pm=None,
    max_result_chars: int = MAX_TOOL_RESULT_CHARS,
) -> str:
    """Run the agent loop to completion. Mutates messages in place.

    The agent loop:
    1. Calls LLM with current messages and tool schemas
    2. Appends LLM response to messages
    3. If response contains tool calls, executes them and appends results
    4. Repeats until no tool calls or max_turns reached

    Args:
        client: LLM client for chat completions
        messages: In-memory message list (mutated in place)
        tools: Dict mapping tool names to async callable functions
        tool_schemas: List of tool schemas for LLM function calling
        max_turns: Maximum iterations before giving up
        max_result_chars: Maximum length of tool result strings before truncation.
            Defaults to MAX_TOOL_RESULT_CHARS.

    Returns:
        Final text content from the last LLM response, or a placeholder
        if max_turns was reached

    Logs:
        DEBUG: LLM response content, tool call arguments, tool call result content
        INFO: Per-turn LLM response metadata, tool calls, results
        WARNING: Max turns reached, unknown tools, tool exceptions
    """
    for _ in range(max_turns):
        result = await run_agent_turn(client, messages, tool_schemas)
        tool_calls = result.tool_calls or None  # run_agent_turn returns [] for no tool calls; existing code checks `if not tool_calls`

        if not tool_calls:
            return result.text

        for call in tool_calls:
            result = await dispatch_tool_call(
                call, tools,
                channel=channel,
                task_queue=task_queue,
                max_result_chars=max_result_chars,
                pm=pm,
            )
            messages.append({
                "role": "tool",
                "tool_call_id": result.tool_call_id,
                "content": result.content,
            })

    logger.warning("max tool-calling rounds reached")
    return MAX_ROUNDS_REACHED_MESSAGE


class SubagentPlugin(CorvidaePlugin):
    """Plugin that registers the subagent tool.

    Launches background agents via run_agent_loop. All subagents share the
    background LLMClient from LLMPlugin (using llm.background if configured,
    otherwise llm.main) and the full tool set minus the subagent tool itself
    (to prevent recursion).

    Depends on "llm" (LLMPlugin) to access the shared background LLM client,
    and on "tools" (ToolCollectionPlugin) to read the tool registry and
    max_result_chars at tool-call time.
    """

    depends_on = frozenset({"llm", "tools"})

    def __init__(self, pm=None) -> None:
        if pm is not None:
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
