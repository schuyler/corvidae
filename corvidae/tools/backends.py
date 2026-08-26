"""Swappable tool backends for channel-scoped tool execution.

ACP sessions (and any future client-mediated tool path) go through a
``ToolBackend`` so Phase-1 local tools can later swap to client-side
backends without rewriting session dispatch. Non-ACP channels keep the
direct ``execute_tool_call`` path unless a backend is explicitly set on
``channel.runtime_overrides["tool_backend"]``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from corvidae.tool import MAX_TOOL_RESULT_CHARS, execute_tool_call

if TYPE_CHECKING:
    from corvidae.channel import Channel
    from corvidae.task import TaskQueue


class ToolBackend(Protocol):
    """Execute a named tool for a channel; return the string result for the LLM."""

    async def run(self, name: str, args: dict, channel: Channel) -> str:
        """Invoke ``name`` with ``args`` in the context of ``channel``."""
        ...


@dataclass
class LocalToolBackend:
    """In-process backend: look up callables and run ``execute_tool_call``.

    This is the Phase-1 default for ACP channels (design D3). Callers that
    need ``ToolContext`` injection should pass ``tool_call_id`` / ``task_queue``
    at construction (as ``resolve_tool_backend`` does per dispatch).
    """

    tools: dict[str, Callable]
    tool_call_id: str = "local"
    task_queue: TaskQueue | None = None
    max_result_chars: int = MAX_TOOL_RESULT_CHARS

    async def run(self, name: str, args: dict, channel: Channel) -> str:
        """Resolve ``name`` in ``self.tools`` and execute it in-process."""
        tool_fn = self.tools[name]
        return await execute_tool_call(
            tool_fn,
            args,
            channel=channel,
            tool_call_id=self.tool_call_id,
            task_queue=self.task_queue,
            max_result_chars=self.max_result_chars,
        )


def resolve_tool_backend(
    channel: Channel | None,
    tools: dict[str, Callable],
    *,
    tool_call_id: str,
    task_queue: TaskQueue | None = None,
    max_result_chars: int = MAX_TOOL_RESULT_CHARS,
) -> ToolBackend | None:
    """Pick a backend for this dispatch, or None to use inline execute_tool_call.

    Order:
    1. Explicit ``channel.runtime_overrides["tool_backend"]`` (tests / Phase 3).
    2. ACP transport → fresh ``LocalToolBackend`` for this call.
    3. Otherwise None (CLI/IRC/Signal keep the historical path).
    """
    if channel is None:
        return None

    override = channel.runtime_overrides.get("tool_backend")
    if override is not None:
        return override

    if channel.matches_transport("acp"):
        return LocalToolBackend(
            tools=tools,
            tool_call_id=tool_call_id,
            task_queue=task_queue,
            max_result_chars=max_result_chars,
        )

    return None
