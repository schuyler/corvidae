"""Agent loop: LLM interaction with tool calling.

This module implements the core agent loop that alternates between LLM calls
and tool execution. The loop continues until the LLM responds without tool
calls or max_turns is reached.

Logging:
    - DEBUG: LLM response content (truncated), tool call arguments (truncated JSON),
      tool call result content (truncated)
    - INFO: LLM response (role, tool_calls count, latency_ms), tool call dispatched,
      tool call result
    - WARNING: max rounds reached, unknown tool called, tool exception
"""

import inspect
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from sherman.llm import LLMClient
from sherman.tool import ToolContext, tool_to_schema  # noqa: F401 — re-exported for backward compat

if TYPE_CHECKING:
    from sherman.channel import Channel
    from sherman.task import TaskQueue

logger = logging.getLogger(__name__)


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

    Returns:
        AgentTurnResult with the parsed response.
    """
    start = time.monotonic()
    response = await client.chat(list(messages), tools=tool_schemas or None)  # Shallow copy: prevents mock assertion issues; client.chat() is read-only.
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


def _truncate(s: str, maxlen: int = 200) -> str:
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

    Returns:
        Final text content from the last LLM response, or a placeholder
        if max_turns was reached

    Logs:
        DEBUG: LLM response content, tool call arguments, tool call result content
        INFO: Per-turn LLM response metadata, tool calls, results
        WARNING: Max turns reached, unknown tools, tool exceptions
    """
    for _ in range(max_turns):
        start = time.monotonic()
        response = await client.chat(messages, tools=tool_schemas or None)
        latency_ms = round((time.monotonic() - start) * 1000, 1)
        msg = response["choices"][0]["message"]
        msg.setdefault("role", "assistant")
        messages.append(msg)

        tool_calls = msg.get("tool_calls")
        logger.info(
            "LLM response received",
            extra={
                "role": msg.get("role"),
                "tool_calls_count": len(tool_calls) if tool_calls else 0,
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

        if not tool_calls:
            return msg.get("content", "")

        for call in tool_calls:
            call_id = call["id"]
            fn_name = call["function"]["name"]
            args = json.loads(call["function"]["arguments"])

            logger.info(
                "tool call dispatched",
                extra={"tool": fn_name, "arg_keys": list(args.keys())},
            )
            logger.debug(
                "tool call arguments",
                extra={
                    "tool": fn_name,
                    "arguments": _truncate(json.dumps(args)),
                },
            )

            if fn_name not in tools:
                logger.warning("unknown tool called: %s", fn_name)
                content = f"Error: unknown tool '{fn_name}'"
            else:
                try:
                    tool_start = time.monotonic()
                    tool_fn = tools[fn_name]
                    # Inject _-prefixed parameters from context, not from
                    # LLM-supplied args.
                    tool_sig = inspect.signature(tool_fn)
                    call_kwargs = dict(args)

                    # Inject ToolContext for tools that declare _ctx
                    if "_ctx" in tool_sig.parameters:
                        call_kwargs["_ctx"] = ToolContext(
                            channel=channel,
                            tool_call_id=call_id,
                            task_queue=task_queue,
                        )

                    content = await tool_fn(**call_kwargs)
                    tool_latency_ms = round((time.monotonic() - tool_start) * 1000, 1)
                    logger.info(
                        "tool call result",
                        extra={"tool": fn_name, "result_length": len(str(content)), "latency_ms": tool_latency_ms},
                    )
                    logger.debug(
                        "tool call result content",
                        extra={
                            "tool": fn_name,
                            "content": _truncate(str(content)),
                        },
                    )
                except Exception:
                    logger.warning(
                        "tool %s raised exception", fn_name, exc_info=True
                    )
                    content = f"Error: unknown error"

            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": str(content),
            })

    logger.warning("max tool-calling rounds reached")
    return "(max tool-calling rounds reached)"


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from text."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def strip_reasoning_content(messages: list[dict]) -> None:
    """Remove reasoning_content from assistant messages in place."""
    for msg in messages:
        if msg.get("role") == "assistant":
            msg.pop("reasoning_content", None)
