"""Agent loop: LLM interaction with tool calling.

This module implements the core agent loop that alternates between LLM calls
and tool execution. The loop continues until the LLM responds without tool
calls or max_turns is reached.

Logging:
    - INFO: LLM response (role, tool_calls count, latency_ms), tool call dispatched,
      tool call result
    - WARNING: max rounds reached, unknown tool called, tool exception

Tool call result content is not logged; only result length is recorded. The
`_truncate` helper is available for future use if result content logging is added.
"""

import inspect
import json
import logging
import re
import time
from typing import Callable

from pydantic import create_model

from sherman.llm import LLMClient

logger = logging.getLogger(__name__)


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
    extra_body: dict | None = None,
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
        extra_body: Optional dict of extra fields to merge into LLM request payload

    Returns:
        Final text content from the last LLM response, or a placeholder
        if max_turns was reached

    Logs:
        INFO: Per-turn LLM response metadata, tool calls, results
        WARNING: Max turns reached, unknown tools, tool exceptions
    """
    for _ in range(max_turns):
        start = time.monotonic()
        if extra_body is not None:
            response = await client.chat(messages, tools=tool_schemas or None, extra_body=extra_body)
        else:
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

            if fn_name not in tools:
                logger.warning("unknown tool called: %s", fn_name)
                content = f"Error: unknown tool '{fn_name}'"
            else:
                try:
                    tool_start = time.monotonic()
                    tool_fn = tools[fn_name]
                    # Inject _-prefixed parameters from context, not from
                    # LLM-supplied args. Currently only _tool_call_id is
                    # injected; future injected params follow the same pattern.
                    tool_sig = inspect.signature(tool_fn)
                    call_kwargs = dict(args)
                    if "_tool_call_id" in tool_sig.parameters:
                        call_kwargs["_tool_call_id"] = call_id
                    content = await tool_fn(**call_kwargs)
                    tool_latency_ms = round((time.monotonic() - tool_start) * 1000, 1)
                    logger.info(
                        "tool call result",
                        extra={"tool": fn_name, "result_length": len(str(content)), "latency_ms": tool_latency_ms},
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


def tool_to_schema(fn: Callable) -> dict:
    """Generate a Chat Completions tool schema from a typed function.

    Extracts parameter types from the function signature and generates
    a JSON schema compatible with OpenAI's function calling format.
    The first line of the docstring becomes the tool description.

    Args:
        fn: An async function with type-annotated parameters

    Returns:
        Dict with 'type: function' and 'function' key containing name,
        description, and parameters schema
    """
    sig = inspect.signature(fn)
    fields = {}
    for param_name, param in sig.parameters.items():
        # CR1: Explicitly skip _-prefixed parameters — they are injected at
        # call time by run_agent_loop, not supplied by the LLM. Do not rely
        # on pydantic's private-attribute convention as the implementation
        # strategy; this check is the authoritative filter.
        if param_name.startswith("_"):
            continue
        annotation = param.annotation if param.annotation is not inspect.Parameter.empty else str
        fields[param_name] = (annotation, ...)

    model = create_model(fn.__name__, **fields)
    schema = model.model_json_schema()

    # Strip top-level title
    schema.pop("title", None)

    # Strip title from each property
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)

    description = ""
    if fn.__doc__:
        description = fn.__doc__.strip().splitlines()[0]

    return {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": description,
            "parameters": schema,
        },
    }


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from text."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def strip_reasoning_content(messages: list[dict]) -> None:
    """Remove reasoning_content from assistant messages in place."""
    for msg in messages:
        if msg.get("role") == "assistant":
            msg.pop("reasoning_content", None)
