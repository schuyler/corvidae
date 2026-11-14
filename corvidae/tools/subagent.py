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

from corvidae.prompts import resolve_prompt
from corvidae.turn import run_agent_turn
from corvidae.attribution import (
    reset_attribution,
    set_attribution,
    next_agent_depth,
    parse_max_agent_depth,
)
from corvidae.hooks import CorvidaePlugin, get_dependency, hookimpl
from corvidae.llm import LLMClient
from corvidae.provenance import cut_text
from corvidae.spill import Cut, NotificationSpill
from corvidae.task import Task
from corvidae.thinking import strip_thinking
from corvidae.tool import (
    DEFAULT_TOOL_TIMEOUTS,
    MAX_TOOL_RESULT_CHARS,
    ToolContext,
    ToolTimeouts,
    dispatch_tool_call,
)

if TYPE_CHECKING:
    from corvidae.channel import Channel
    from corvidae.task import TaskQueue

logger = logging.getLogger(__name__)

# Note: same text as MAX_TURNS_FALLBACK_MESSAGE in agent.py — kept in sync.
MAX_ROUNDS_REACHED_MESSAGE = "(max tool-calling rounds reached)"

DEFAULT_SUBAGENT_MAX_TURNS = 10


def parse_subagent_max_turns(config: object) -> int:
    """Parse ``subagent.max_turns`` into the per-subagent tool-round budget.

    Raises ValueError naming the key, the rejected value and the accepted
    values. An absent key gives the 10-round default. Called at startup
    (runtime step 7, clean one-line config error) and again in
    SubagentPlugin.on_start, which re-parses like every other plugin.
    """
    if config is None:
        return DEFAULT_SUBAGENT_MAX_TURNS
    if not isinstance(config, dict):
        raise ValueError(
            f"config must be a mapping; got {config!r} ({type(config).__name__})"
        )
    section = config.get("subagent")
    if section is None:
        return DEFAULT_SUBAGENT_MAX_TURNS
    if not isinstance(section, dict):
        raise ValueError(
            "subagent must be a mapping of subagent settings "
            f"(e.g. {{max_turns: 40}}); got {section!r} ({type(section).__name__})"
        )
    raw = section.get("max_turns", DEFAULT_SUBAGENT_MAX_TURNS)
    # bool is a subclass of int: int(False) == 0 would make every subagent
    # return the bare marker without ever calling the model, so reject it
    # explicitly. A non-positive budget is a no-op, never a valid setting.
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
        raise ValueError(
            "subagent.max_turns must be a positive integer "
            f"(e.g. 40); got {raw!r} ({type(raw).__name__})"
        )
    return raw


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
    call_timeouts: ToolTimeouts = DEFAULT_TOOL_TIMEOUTS,
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
        max_result_chars: Longest tool result kept whole; a longer one is cut
            to this, its full text spilled where a tool in ``tools`` can read
            it, and a plain-text cut line put ahead of it.
        call_timeouts: Per-call tool time bounds (tools.call_timeout).

    Returns:
        Final text content from the last LLM response, or a placeholder
        if max_turns was reached

    Logs:
        DEBUG: LLM response content, tool call arguments, tool call result content
        INFO: Per-turn LLM response metadata, tool calls, results
        WARNING: Max turns reached, unknown tools, tool exceptions
    """
    for _ in range(max_turns):
        turn = await run_agent_turn(client, messages, tool_schemas)

        if not turn.tool_calls:
            return turn.text

        for call in turn.tool_calls:
            tool_result = await dispatch_tool_call(
                call, tools,
                channel=channel,
                task_queue=task_queue,
                pm=pm,
                call_timeouts=call_timeouts,
            )
            content = tool_result.content
            # A long result is cut and spilled here (#575): this loop stores
            # no rows, so the cut line is plain text ahead of the head.
            if len(content) > max_result_chars:
                content = await _cut_result(content, max_result_chars, tools, channel, pm)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_result.tool_call_id,
                "content": content,
            })

    logger.warning("max tool-calling rounds reached")
    # Carry the subagent's last progress note so the parent channel gets a
    # partial result, not a bare four-word marker (#494).
    last_text = next(
        (
            m["content"]
            for m in reversed(messages)
            if m.get("role") == "assistant"
            and isinstance(m.get("content"), str)
            and m["content"].strip()
        ),
        None,
    )
    if last_text is None:
        return MAX_ROUNDS_REACHED_MESSAGE
    return f"{MAX_ROUNDS_REACHED_MESSAGE}\n\nLast progress: {last_text.strip()}"


async def _cut_result(
    content: str, limit: int, tools: dict[str, Callable], channel: "Channel | None", pm
) -> str:
    """Cut ``content`` to ``limit``, spill its full text, and head it with the cut line."""
    spill = pm.get_plugin("spill") if pm is not None else None
    if isinstance(spill, NotificationSpill):
        cut = await spill.cut(
            content, limit=limit, source="subagent tool result",
            can_read=spill.can_read(channel, tools),
        )
    else:
        # No spill service (a bare loop): the head is kept and the line says it was cut.
        logger.warning(
            "subagent tool result of %d characters cut to %d with no spill service "
            "registered, so the full text was not saved",
            len(content), limit,
        )
        cut = Cut(shown=limit, total=len(content))
    words = cut_text(cut.shown, cut.total, cut.path, cut.spilled, cut.failed)
    return f"[{words}]\n{content[:limit]}"


class SubagentPlugin(CorvidaePlugin):
    """Plugin that registers the subagent tool.

    Launches background agents via run_agent_loop. All subagents share the
    LLMClient LLMPlugin resolves for the subagent stage and the full tool set
    minus the subagent tool itself (to prevent recursion).

    Depends on "llm" (LLMPlugin) to access the shared LLM client,
    and on "tools" (ToolCollectionPlugin) to read the tool registry and
    max_result_chars at tool-call time.
    """

    depends_on = frozenset({"llm", "tools"})

    def __init__(self, pm=None) -> None:
        if pm is not None:
            self.pm = pm

    @hookimpl
    async def on_start(self, config: dict) -> None:
        self._system_prompt = resolve_prompt(
            config, "tools.subagent.system_prompt", SUBAGENT_SYSTEM_PROMPT
        )
        # Tool-round budget per subagent (#494): fetch-heavy research tasks
        # need more than the default 10. Non-positive-integer values are
        # rejected at startup with a one-line config error (runtime step 7
        # calls the parser; on_start re-parses like every other plugin).
        self._max_turns = parse_subagent_max_turns(config)
        self._max_agent_depth = parse_max_agent_depth(config)
        logger.debug("SubagentPlugin started")

    @hookimpl
    def register_llm_stages(self, stages: dict[str, str]) -> None:
        stages["subagent"] = "chat"

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
        excludes the subagent tool from the set, uses the shared subagent-stage client
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
        # Depth limit: refuse if we are at the max nesting depth.
        try:
            depth = next_agent_depth(self._max_agent_depth)
        except ValueError as exc:
            return str(exc)

        task_queue = ctx.task_queue
        if task_queue is None:
            # Name the internal dependency and the operator action that fixes it.
            return (
                "Error: task queue not available — ensure the task plugin is "
                "loaded, then retry."
            )

        channel = ctx.channel
        if channel is None:
            # Explain how to give the tool the context it needs.
            return (
                "Error: no channel context available for subagent — call this "
                "tool from an agent turn that has a channel, then retry."
            )

        # Get tools, excluding subagent itself to prevent recursion
        from corvidae.tool_policy import disabled_on
        from corvidae.tool_collection import ToolCollectionPlugin
        from corvidae.llm_plugin import LLMPlugin
        tools_plugin = get_dependency(self.pm, "tools", ToolCollectionPlugin)
        # A subagent never gets a tool its calling channel cannot use
        # (channels.<id>.tools.disabled, issue #149); otherwise a channel
        # without shell would get it back one hop away.
        registry = tools_plugin.get_registry().exclude("subagent", *disabled_on(channel))
        max_result_chars = tools_plugin.max_result_chars
        call_timeouts = tools_plugin.call_timeouts
        tools_dict = registry.as_dict()
        tool_schemas = registry.schemas()

        # Use the shared subagent-stage client from LLMPlugin (do NOT call start/stop).
        llm = get_dependency(self.pm, "llm", LLMPlugin)
        client = llm.get_client("subagent")

        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": instructions},
        ]

        async def work():
            # Attribute the subagent's LLM calls to the subagent stage,
            # shadowing whatever attribution the enqueuing turn captured.
            attribution_token = set_attribution(
                stage="subagent", channel_id=channel.id,
                agent_depth=depth,
            )
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

        task = Task(
            work=work,
            channel=channel,
            tool_call_id=ctx.tool_call_id,
            description=description,
        )
        await task_queue.enqueue(task)
        return f"Subagent task {task.task_id} enqueued: {description}"
