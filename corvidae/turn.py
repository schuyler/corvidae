"""Single-turn LLM execution.

This module wraps a single LLM chat call: it invokes the client, parses the
response, appends the assistant message to the in-place message list, and
records timing/logging. It does not execute tools or implement any loop.

Logging:
    - DEBUG: LLM response content (truncated)
    - INFO: LLM response (role, tool_calls count, latency_ms)
"""

import logging
import time
from dataclasses import dataclass

from corvidae.llm import LLMClient

logger = logging.getLogger(__name__)

LOG_TRUNCATION_LENGTH = 200


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
    Appends the assistant message to messages (mutates in place).

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
