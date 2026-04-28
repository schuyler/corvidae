"""Agent loop: LLM interaction with tool calling.

This module implements the core agent loop that alternates between LLM calls
and tool execution. The loop continues until the LLM responds without tool
calls or max_turns is reached.

Logging:
    - DEBUG: LLM response content (truncated)
    - INFO: LLM response (role, tool_calls count, latency_ms)
    - WARNING: max rounds reached

Tool dispatch logging (dispatched, arguments, result, errors) is handled by
corvidae.tool.dispatch_tool_call.
"""

import logging
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from corvidae.llm import LLMClient
from corvidae.tool import MAX_TOOL_RESULT_CHARS, dispatch_tool_call

if TYPE_CHECKING:
    from corvidae.channel import Channel
    from corvidae.task import TaskQueue

logger = logging.getLogger(__name__)

LOG_TRUNCATION_LENGTH = 200
# Note: same text as MAX_TURNS_FALLBACK_MESSAGE in agent.py — kept in sync.
MAX_ROUNDS_REACHED_MESSAGE = "(max tool-calling rounds reached)"


@dataclass
class AgentTurnResult:
    """Result of a single agent turn (one LLM invocation).

    Attributes:
        message: The raw assistant message dict from the LLM response.
            Always has "role": "assistant". May contain tool_calls,
            content, reasoning_content.
        tool_calls: List of tool call dicts from the response. Empty list
            if the LLM did not request any tool calls.
        text: The text content of the response. Empty string if the LLM
            produced only tool calls with no text.
        latency_ms: Wall-clock time for the LLM call in milliseconds,
            rounded to one decimal place.
    """

    message: dict
    tool_calls: list[dict]
    text: str
    latency_ms: float


async def run_agent_turn(
    client: LLMClient,
    messages: list[dict],
    tool_schemas: list[dict],
    extra_body: dict | None = None,
) -> AgentTurnResult:
    """Single LLM invocation. Returns the response; does not execute tools.

    Calls client.chat() once with the given messages and tool schemas.
    Appends the assistant message to messages (mutates in place, matching
    run_agent_loop convention). Logs at the same levels as run_agent_loop.

    Args:
        client: LLM client for chat completions.
        messages: Conversation history. The assistant response is appended
            in place.
        tool_schemas: Tool schemas for LLM function calling. Pass empty
            list for no tools (converted to None for the API call).
        extra_body: Optional dict of extra fields to merge into the LLM
            request body (e.g. inference params like temperature). Only
            forwarded when not None.

    Returns:
        AgentTurnResult with the parsed response.
    """
    start = time.monotonic()
    kwargs: dict = {"tools": tool_schemas or None}
    if extra_body is not None:
        kwargs["extra_body"] = extra_body
    response = await client.chat(list(messages), **kwargs)  # Shallow copy: prevents mock assertion issues; client.chat() is read-only.
    latency_ms = round((time.monotonic() - start) * 1000, 1)
    msg = response["choices"][0]["message"]
    msg.setdefault("role", "assistant")
    messages.append(msg)
    tool_calls = msg.get("tool_calls") or []
    text = msg.get("content", "") or ""
    logger.info(
        "LLM response received",
        extra={
            "role": msg.get("role"),
            "tool_calls_count": len(tool_calls),
            "latency_ms": latency_ms,
        },
    )
    logger.debug(
        "LLM response content",
        extra={
            "content": _truncate(msg.get("content") or ""),
            "has_reasoning_content": "reasoning_content" in msg,
            "reasoning_content_length": len(msg["reasoning_content"]) if "reasoning_content" in msg else None,
        },
    )
    return AgentTurnResult(message=msg, tool_calls=tool_calls, text=text, latency_ms=latency_ms)


def _truncate(s: str, maxlen: int = LOG_TRUNCATION_LENGTH) -> str:
    """Truncate a string to maxlen characters, appending '...' if truncated.

    Available for use in log calls where result content needs to be included
    without logging sensitive user data in full.
    """
    return s[:maxlen] + "..." if len(s) > maxlen else s


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


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from text."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def strip_reasoning_content(messages: list[dict]) -> None:
    """Remove reasoning_content from assistant messages in place."""
    for msg in messages:
        if msg.get("role") == "assistant":
            msg.pop("reasoning_content", None)
