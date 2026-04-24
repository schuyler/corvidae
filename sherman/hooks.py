from __future__ import annotations

from typing import TYPE_CHECKING

import apluggy as pluggy

if TYPE_CHECKING:
    from sherman.channel import Channel

# NOTE: marker name is "sherman", not "agent" as in design.md
hookspec = pluggy.HookspecMarker("sherman")
hookimpl = pluggy.HookimplMarker("sherman")

class AgentSpec:
    """Hook specifications for the agent daemon.

    Plugins implement these hooks to receive lifecycle events, handle
    incoming messages, send outgoing messages, and register tools with
    the agent loop. All hooks are optional; a plugin only needs to
    implement the hooks it cares about.
    """

    @hookspec
    async def on_start(self, config: dict) -> None:
        """Called once when the daemon starts, after config is loaded.

        Args:
            config: The full parsed YAML config dict.
        """

    @hookspec
    async def on_stop(self) -> None:
        """Called once when the daemon receives SIGINT or SIGTERM."""

    @hookspec
    async def on_message(self, channel: Channel, sender: str, text: str) -> None:
        """Called when an inbound message arrives on a channel.

        Args:
            channel: The Channel object for this conversation scope.
            sender: The user or entity that sent the message.
            text: The message content.
        """

    @hookspec
    async def send_message(self, channel: Channel, text: str, latency_ms: float | None = None) -> None:
        """Called to deliver an outbound message to a channel.

        Transport plugins implement this to forward the message over
        their protocol.

        Args:
            channel: The Channel object identifying the target.
            text: The message content to send.
            latency_ms: Optional agent loop latency in milliseconds.
                        Transport plugins may use this to display timing to the user.
        """

    @hookspec
    def register_tools(self, tool_registry: list) -> None:
        """Called during startup so plugins can add tools to the agent loop.

        Args:
            tool_registry: A mutable list. Plugins append callable tool
                functions to this list; the agent loop will make them
                available to the LLM.
        """

    @hookspec
    async def on_agent_response(
        self, channel: Channel, request_text: str, response_text: str
    ) -> None:
        """Called after the agent loop produces a response to a message.

        Args:
            channel: The Channel where the conversation occurred.
            request_text: The original user message that triggered the loop.
            response_text: The final text produced by the agent loop.
        """

    @hookspec
    async def on_task_complete(
        self, channel: Channel, task_id: str, result: str
    ) -> None:
        """Called when a background task finishes.

        Args:
            channel: The Channel associated with the task.
            task_id: Unique identifier for the completed task.
            result: The output produced by the task.
        """
