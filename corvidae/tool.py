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
    - ToolTimeouts: the per-call time bound (default plus per-tool overrides)
    - resolve_own_timeout(): read a tool's declared ``own_timeout`` at dispatch
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic import create_model

# Maximum characters in a tool result before truncation. Override via agent.max_tool_result_chars in agent.yaml.
MAX_TOOL_RESULT_CHARS = 100_000
# Appended to truncated tool results. Format with original_len=<int>. Uses em-dash (U+2014).
TOOL_TRUNCATION_TEMPLATE = "\n[truncated \u2014 {original_len} chars total]"  # em-dash, matching current code

_LOG_TRUNCATION_LENGTH = 200

# Default bound, in seconds, on a single tool call (tools.call_timeout).
DEFAULT_CALL_TIMEOUT: float = 600.0
# Result string for a timed-out call. Format with name=<tool>, bound=<seconds>.
TIMEOUT_RESULT_TEMPLATE = "Error: tool '{name}' timed out after {bound:g}s"
# WARNING when the tool honoured the cancellation.
TIMEOUT_WARNING_HONOURED = (
    "tool %s timed out after %gs on %s (call %s) and was cancelled; "
    "raise tools.timeouts.%s if this tool legitimately runs longer"
)
# WARNING when the tool swallowed the cancellation and its late result was dropped.
TIMEOUT_WARNING_IGNORED = (
    "tool %s timed out after %gs on %s (call %s) but ignored cancellation; "
    "its late result was discarded \u2014 the tool must re-raise CancelledError; "
    "raise tools.timeouts.%s if this tool legitimately runs longer"
)

if TYPE_CHECKING:
    from corvidae.channel import Channel
    from corvidae.task import TaskQueue

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolTimeouts:
    """Per-call time bounds: a default plus per-tool overrides.

    ``None`` means no bound. Config-free: the parser lives with the
    ToolCollectionPlugin, which owns the ``tools.*`` keys.
    """

    default: float | None = DEFAULT_CALL_TIMEOUT
    per_tool: Mapping[str, float | None] = field(default_factory=dict)

    def for_tool(self, name: str) -> float | None:
        """Return the bound for ``name``; an override wins, including None."""
        return self.per_tool[name] if name in self.per_tool else self.default


# Bound applied when a caller passes nothing, so third-party callers of the
# public dispatch_tool_call are bounded too.
DEFAULT_TOOL_TIMEOUTS = ToolTimeouts()

# Tool names whose unreadable own_timeout has already been warned about.
_warned_own_timeout: set[str] = set()


def resolve_own_timeout(tool_fn: Callable, name: str | None = None) -> float | None:
    """Read the ``own_timeout`` a tool declares, or None if it declares none.

    ``own_timeout`` is a number or a zero-argument callable returning
    ``float | None``, set as an attribute on the registered callable. It is
    read per call so it tracks live config. A bound method reads its
    function's attribute. For a partial or callable instance, set the
    attribute on that object itself; a class-level property is not read.

    A callable that raises, or a value that is not a finite real number,
    is rescued as None (the outer bound then applies) with one WARNING per
    tool name per process.
    """
    # Look up via vars() so Mock/AsyncMock tools cannot invent an attribute.
    target = getattr(tool_fn, "__func__", tool_fn)
    try:
        declared = vars(target).get("own_timeout")
    except TypeError:
        # No __dict__ on this callable, so nothing can be declared.
        return None
    if declared is None:
        return None
    label = name or getattr(tool_fn, "__name__", repr(tool_fn))
    try:
        value = declared() if callable(declared) else declared
    except Exception as exc:
        _warn_own_timeout(label, declared, type(exc).__name__, exc_info=True)
        return None
    if value is None:
        return None
    # bool is an int subclass but never a sensible timeout.
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        _warn_own_timeout(label, value, "invalid value", exc_info=False)
        return None
    return float(value)


def _warn_own_timeout(label: str, value: object, why: str, *, exc_info: bool) -> None:
    """Log (once per tool per process) that own_timeout was unreadable."""
    if label in _warned_own_timeout:
        return
    _warned_own_timeout.add(label)
    logger.warning(
        "tool %s declares an unreadable own_timeout %r (%s); applying the "
        "outer tools.call_timeout instead \u2014 fix the tool's own_timeout",
        label, value, why, exc_info=exc_info,
    )


def _outer_bound(
    fn_name: str, tool_fn: Callable, call_timeouts: ToolTimeouts
) -> float | None:
    """Effective outer bound for a call, skipping it when the tool's own is shorter."""
    bound = call_timeouts.for_tool(fn_name)
    if bound is None:
        return None
    own = resolve_own_timeout(tool_fn, fn_name)
    # R4: the tool's own (<=) timeout fires first and reports itself.
    if own is not None and own <= bound:
        logger.debug(
            "outer call timeout skipped: tool's own timeout is not longer",
            extra={"tool": fn_name, "own_timeout": own, "bound": bound},
        )
        return None
    return bound


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
        # call time by dispatch_tool_call, not supplied by the LLM.
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
            fn: An async callable with type-annotated parameters

        Returns:
            Tool instance with name=fn.__name__ and auto-generated schema

        Raises:
            TypeError: If fn is not a coroutine function (i.e. not async).
        """
        if not asyncio.iscoroutinefunction(fn):
            raise TypeError(
                f"Tool function {fn.__qualname__!r} must be async (a coroutine function)"
            )
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
    since the two call sites (Agent and tools.subagent.run_agent_loop) have
    different error-reporting requirements.

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

    if not asyncio.iscoroutinefunction(tool_fn):
        logger.warning("Tool %s is not async; wrapping in to_thread", tool_fn.__qualname__)
        result = await asyncio.to_thread(tool_fn, **call_kwargs)
    else:
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


def _timed_out(
    fn_name: str,
    bound: float | None,
    channel_label: str,
    call_id: str,
    *,
    honoured: bool,
    exc: BaseException | None = None,
) -> str:
    """Log the timeout WARNING and return the error result string."""
    template = TIMEOUT_WARNING_HONOURED if honoured else TIMEOUT_WARNING_IGNORED
    logger.warning(
        template, fn_name, bound, channel_label, call_id, fn_name,
        extra={
            "tool": fn_name,
            "channel": channel_label,
            "tool_call_id": call_id,
            "timeout": bound,
        },
        exc_info=exc,
    )
    return TIMEOUT_RESULT_TEMPLATE.format(name=fn_name, bound=bound)


async def dispatch_tool_call(
    call: dict,
    tools: dict[str, Callable],
    *,
    channel: "Channel | None" = None,
    task_queue: "TaskQueue | None" = None,
    max_result_chars: int = MAX_TOOL_RESULT_CHARS,
    pm=None,
    call_timeouts: ToolTimeouts = DEFAULT_TOOL_TIMEOUTS,
) -> ToolCallResult:
    """Parse, dispatch, and wrap a single tool call from an LLM response.

    Handles the full lifecycle of a tool call: JSON parsing, unknown-tool
    detection, invocation via execute_tool_call, error wrapping, logging,
    and firing the transform_tool_result hook.

    The transform_tool_result hook fires ONLY when execute_tool_call was
    invoked (success or exception). Pre-dispatch errors (JSON parse failure,
    unknown tool) return early and skip the hook.

    Args:
        call: A tool call dict from the LLM response with keys 'id' and 'function'.
        tools: Dict mapping tool names to async callable functions.
        channel: Channel for ToolContext injection. None when unavailable.
        task_queue: TaskQueue for ToolContext injection. None when unavailable.
        max_result_chars: Maximum length of tool result strings before truncation.
        pm: pluggy PluginManager for firing transform_tool_result. None to skip.
        call_timeouts: Bound on the call (default 600 s, per-tool overrides).
            A timeout cancels the tool and returns an error result; the
            outer bound is skipped when the tool declares a not-longer
            ``own_timeout``.

    Returns:
        ToolCallResult with the content, error flag, and timing information.
    """
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

    # Step 4: Execute the tool (ACP / overrides go through ToolBackend; else inline)
    from corvidae.tools.backends import resolve_tool_backend

    backend = resolve_tool_backend(
        channel,
        tools,
        tool_call_id=call_id,
        task_queue=task_queue,
        max_result_chars=max_result_chars,
    )
    latency_ms: float | None = None
    error = False
    tool_start = time.monotonic()
    channel_label = channel.id if channel is not None else "no channel"
    tool_fn = tools[fn_name]
    # Outer bound (None => no deadline). The scope wraps step 4 only, so a
    # slow transform_tool_result hook is never reported as a tool timeout.
    bound = _outer_bound(fn_name, tool_fn, call_timeouts)
    scope = asyncio.timeout(bound)
    try:
        async with scope:
            if backend is not None:
                content = await backend.run(fn_name, args, channel)
            else:
                content = await execute_tool_call(
                    tool_fn,
                    args,
                    channel=channel,
                    tool_call_id=call_id,
                    task_queue=task_queue,
                    max_result_chars=max_result_chars,
                )
        latency_ms = round((time.monotonic() - tool_start) * 1000, 1)
        if scope.expired():
            # The tool swallowed the cancel and returned: discard the late result.
            content = _timed_out(fn_name, bound, channel_label, call_id, honoured=False)
            error = True
        else:
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
    except Exception as exc:
        latency_ms = round((time.monotonic() - tool_start) * 1000, 1)
        if scope.expired():
            # Our deadline fired: a TimeoutError means the cancel was honoured,
            # anything else means the tool raised while handling it.
            content = _timed_out(
                fn_name, bound, channel_label, call_id,
                honoured=isinstance(exc, TimeoutError), exc=exc,
            )
        else:
            # Includes a tool's OWN TimeoutError raised before the deadline.
            logger.warning("tool %s raised exception", fn_name, exc_info=True)
            content = f"Error: tool '{fn_name}' failed"
        error = True

    # Step 5: Fire transform_tool_result hook (only when the tool was invoked)
    if pm is not None:
        hook_result = await pm.ahook.transform_tool_result(
            tool_name=fn_name, result=content, channel=channel,
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

    Constructed per tool call by ``dispatch_tool_call``. Tools without
    ``_ctx`` work exactly as before.

    Attributes:
        channel: The channel this tool call is executing on. None when
            dispatch_tool_call is called without channel context (e.g.,
            background task sub-agent loops in Phase 1).
        tool_call_id: The LLM-assigned call ID for this invocation.
        task_queue: The TaskQueue for enqueueing background work. None
            when no TaskPlugin is registered.
    """

    channel: Channel | None
    tool_call_id: str
    task_queue: TaskQueue | None
