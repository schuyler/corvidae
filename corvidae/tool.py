"""Tool abstraction for the agent loop.

This module defines the Tool dataclass, ToolRegistry, tool_to_schema(),
execute_tool_call(), dispatch_tool_call(), and ToolCallResult.
These are part of the public API for external plugins.

Public API:
    - Tool: dataclass wrapping a callable with its name and schema
    - ToolRegistry: collection of Tool instances with dict/schema views
    - tool_to_schema(): generate a Chat Completions schema from a typed function
    - ToolContext: context injected into tools that declare a ``_ctx`` parameter
    - ToolCallResult: result of a single dispatched tool call
    - dispatch_tool_call(): parse, dispatch, and wrap a tool call from an LLM response
"""

from __future__ import annotations

import inspect
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import create_model

# Maximum characters in a tool result before truncation. Override via agent.max_tool_result_chars in agent.yaml.
MAX_TOOL_RESULT_CHARS = 100_000
# Appended to truncated tool results. Format with original_len=<int>. Uses em-dash (U+2014).
TOOL_TRUNCATION_TEMPLATE = "\n[truncated \u2014 {original_len} chars total]"  # em-dash, matching current code

_LOG_TRUNCATION_LENGTH = 200

if TYPE_CHECKING:
    from corvidae.channel import Channel
    from corvidae.task import TaskQueue

logger = logging.getLogger(__name__)


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
        # Explicitly skip _-prefixed parameters — they are injected at
        # call time by run_agent_loop, not supplied by the LLM.
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


@dataclass
class Tool:
    """A tool wrapping a callable with its name and schema.

    Attributes:
        name: The tool's name (used as the key in the tools dict)
        fn: The async callable implementing the tool
        schema: The Chat Completions schema dict for LLM function calling
    """

    name: str
    fn: Callable
    schema: dict

    @classmethod
    def from_function(cls, fn: Callable) -> "Tool":
        """Create a Tool from a callable, auto-generating name and schema.

        Args:
            fn: A callable (typically async) with type-annotated parameters

        Returns:
            Tool instance with name=fn.__name__ and auto-generated schema
        """
        schema = tool_to_schema(fn)
        return cls(name=fn.__name__, fn=fn, schema=schema)


class ToolRegistry:
    """Holds Tool instances. Provides dict and schema list views.

    Plugins append Tool instances (or bare callables) to the registry
    during the register_tools hook. Bare callables are auto-wrapped via
    Tool.from_function() for backward compatibility.
    """

    def __init__(self) -> None:
        self._tools: list[Tool] = []

    def add(self, tool: Tool) -> None:
        """Add a Tool instance to the registry.

        Args:
            tool: The Tool to add
        """
        self._tools.append(tool)

    def as_dict(self) -> dict[str, Callable]:
        """Return a dict mapping tool names to their callables.

        Returns:
            Dict of {name: fn} for all registered tools
        """
        return {t.name: t.fn for t in self._tools}

    def schemas(self) -> list[dict]:
        """Return a list of tool schemas for LLM function calling.

        Returns:
            List of schema dicts in Chat Completions format
        """
        return [t.schema for t in self._tools]

    def exclude(self, *names: str) -> "ToolRegistry":
        """Return a new ToolRegistry without the named tools.

        Args:
            *names: Tool names to exclude

        Returns:
            New ToolRegistry containing all tools except those named
        """
        excluded = set(names)
        new_registry = ToolRegistry()
        for tool in self._tools:
            if tool.name not in excluded:
                new_registry.add(tool)
        return new_registry

    def __len__(self) -> int:
        return len(self._tools)


async def execute_tool_call(
    tool_fn: Callable,
    args: dict,
    *,
    channel: "Channel | None" = None,
    tool_call_id: str,
    task_queue: "TaskQueue | None" = None,
    max_result_chars: int = MAX_TOOL_RESULT_CHARS,
) -> str:
    """Invoke a tool function, injecting ToolContext if the function declares ``_ctx``.

    Inspects the function signature for a ``_ctx`` parameter. If present,
    constructs a ToolContext from the provided channel, tool_call_id, and
    task_queue and injects it. Otherwise, calls with args only.

    Does **not** catch exceptions — callers are responsible for error handling
    since the two call sites (Agent and run_agent_loop) have different
    error-reporting requirements.

    Args:
        tool_fn: The async tool callable to invoke.
        args: Dict of keyword arguments from the LLM (JSON-parsed).
        channel: Channel for ToolContext injection. None when unavailable.
        tool_call_id: The LLM-assigned call ID for this invocation.
        task_queue: TaskQueue for ToolContext injection. None when unavailable.
        max_result_chars: Maximum length of the result string before truncation.
            Defaults to MAX_TOOL_RESULT_CHARS.

    Returns:
        str(result) of the tool function's return value.
    """
    sig = inspect.signature(tool_fn)
    call_kwargs = dict(args)

    if "_ctx" in sig.parameters:
        call_kwargs["_ctx"] = ToolContext(
            channel=channel,
            tool_call_id=tool_call_id,
            task_queue=task_queue,
        )

    result = await tool_fn(**call_kwargs)
    result_str = str(result)
    if len(result_str) > max_result_chars:
        original_len = len(result_str)
        result_str = result_str[:max_result_chars] + TOOL_TRUNCATION_TEMPLATE.format(original_len=original_len)
    return result_str


def _truncate(s: str, maxlen: int = _LOG_TRUNCATION_LENGTH) -> str:
    """Truncate a string to maxlen characters, appending '...' if truncated."""
    return s[:maxlen] + "..." if len(s) > maxlen else s


@dataclass
class ToolCallResult:
    """Result of executing a single tool call.

    Attributes:
        tool_call_id: The LLM-assigned call ID for this invocation.
        tool_name: The name of the tool that was called.
        content: The result string to return to the LLM (may be an error message).
        latency_ms: Wall-clock time for execute_tool_call in milliseconds, or None
            if the tool was never invoked (JSON parse error, unknown tool).
        error: True when the result is an error message, False on success.
    """

    tool_call_id: str
    tool_name: str
    content: str
    latency_ms: float | None
    error: bool


async def dispatch_tool_call(
    call: dict,
    tools: dict[str, Callable],
    *,
    channel: "Channel | None" = None,
    task_queue: "TaskQueue | None" = None,
    max_result_chars: int = MAX_TOOL_RESULT_CHARS,
    pm=None,
) -> ToolCallResult:
    """Parse, dispatch, and wrap a single tool call from an LLM response.

    Handles the full lifecycle of a tool call: JSON parsing, unknown-tool
    detection, invocation via execute_tool_call, error wrapping, logging,
    and firing the process_tool_result hook.

    The process_tool_result hook fires ONLY when execute_tool_call was
    invoked (success or exception). Pre-dispatch errors (JSON parse failure,
    unknown tool) return early and skip the hook.

    Args:
        call: A tool call dict from the LLM response with keys 'id' and 'function'.
        tools: Dict mapping tool names to async callable functions.
        channel: Channel for ToolContext injection. None when unavailable.
        task_queue: TaskQueue for ToolContext injection. None when unavailable.
        max_result_chars: Maximum length of tool result strings before truncation.
        pm: pluggy PluginManager for firing process_tool_result. None to skip.

    Returns:
        ToolCallResult with the content, error flag, and timing information.
    """
    # Deferred to avoid circular import: hooks may import from tool.
    from corvidae.hooks import resolve_hook_results, HookStrategy

    call_id: str = call["id"]
    fn_name: str = call["function"]["name"]
    raw_args: str = call["function"]["arguments"]

    # Step 1: Parse arguments
    try:
        args = json.loads(raw_args)
    except json.JSONDecodeError:
        logger.warning(
            "malformed tool call arguments",
            extra={"tool": fn_name, "raw_args": _truncate(raw_args)},
        )
        content = f"Error: malformed arguments for tool '{fn_name}'"
        return ToolCallResult(
            tool_call_id=call_id,
            tool_name=fn_name,
            content=content,
            latency_ms=None,
            error=True,
        )

    # Step 2: Log dispatch
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

    # Step 3: Check tool exists
    if fn_name not in tools:
        logger.warning("unknown tool called: %s", fn_name)
        content = f"Error: unknown tool '{fn_name}'"
        return ToolCallResult(
            tool_call_id=call_id,
            tool_name=fn_name,
            content=content,
            latency_ms=None,
            error=True,
        )

    # Step 4: Execute the tool
    tool_fn = tools[fn_name]
    latency_ms: float | None = None
    error = False
    tool_start = time.monotonic()
    try:
        content = await execute_tool_call(
            tool_fn,
            args,
            channel=channel,
            tool_call_id=call_id,
            task_queue=task_queue,
            max_result_chars=max_result_chars,
        )
        latency_ms = round((time.monotonic() - tool_start) * 1000, 1)
        logger.info(
            "tool call result",
            extra={"tool": fn_name, "result_length": len(str(content)), "latency_ms": latency_ms},
        )
        logger.debug(
            "tool call result content",
            extra={
                "tool": fn_name,
                "content": _truncate(str(content)),
            },
        )
    except Exception:
        latency_ms = round((time.monotonic() - tool_start) * 1000, 1)
        logger.warning("tool %s raised exception", fn_name, exc_info=True)
        content = f"Error: tool '{fn_name}' failed"
        error = True

    # Step 5: Fire process_tool_result hook (only when execute_tool_call was invoked)
    if pm is not None:
        results = await pm.ahook.process_tool_result(
            tool_name=fn_name, result=content, channel=channel,
        )
        hook_result = resolve_hook_results(
            results, "process_tool_result", HookStrategy.VALUE_FIRST, pm=pm,
        )
        if hook_result is not None:
            content = hook_result

    return ToolCallResult(
        tool_call_id=call_id,
        tool_name=fn_name,
        content=content,
        latency_ms=latency_ms,
        error=error,
    )


@dataclass
class ToolContext:
    """Context injected into tools that declare a ``_ctx`` parameter.

    Constructed per tool call by run_agent_loop. Tools without ``_ctx``
    work exactly as before.

    Attributes:
        channel: The channel this tool call is executing on. None when
            run_agent_loop is called without channel context (e.g.,
            background task sub-agent loops in Phase 1).
        tool_call_id: The LLM-assigned call ID for this invocation.
        task_queue: The TaskQueue for enqueueing background work. None
            when no TaskPlugin is registered.
    """

    channel: Channel | None
    tool_call_id: str
    task_queue: TaskQueue | None
