"""Tool abstraction for the agent loop.

This module defines the Tool dataclass, ToolRegistry, and tool_to_schema()
function. These are part of the public API for external plugins.

Public API:
    - Tool: dataclass wrapping a callable with its name and schema
    - ToolRegistry: collection of Tool instances with dict/schema views
    - tool_to_schema(): generate a Chat Completions schema from a typed function
    - ToolContext: context injected into tools that declare a ``_ctx`` parameter
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic import create_model

if TYPE_CHECKING:
    from sherman.channel import Channel
    from sherman.task import TaskQueue


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
) -> str:
    """Invoke a tool function, injecting ToolContext if the function declares ``_ctx``.

    Inspects the function signature for a ``_ctx`` parameter. If present,
    constructs a ToolContext from the provided channel, tool_call_id, and
    task_queue and injects it. Otherwise, calls with args only.

    Does **not** catch exceptions — callers are responsible for error handling
    since the two call sites (AgentPlugin and run_agent_loop) have different
    error-reporting requirements.

    Args:
        tool_fn: The async tool callable to invoke.
        args: Dict of keyword arguments from the LLM (JSON-parsed).
        channel: Channel for ToolContext injection. None when unavailable.
        tool_call_id: The LLM-assigned call ID for this invocation.
        task_queue: TaskQueue for ToolContext injection. None when unavailable.

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
    return str(result)


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
