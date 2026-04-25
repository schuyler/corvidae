from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import apluggy as pluggy

if TYPE_CHECKING:
    from sherman.channel import Channel

# NOTE: marker name is "sherman", not "agent" as in design.md
hookspec = pluggy.HookspecMarker("sherman")
hookimpl = pluggy.HookimplMarker("sherman")

_pm_logger = logging.getLogger("sherman.plugin_manager")


def create_plugin_manager() -> pluggy.PluginManager:
    """Create and configure the plugin manager with AgentSpec hooks.

    Returns:
        A pluggy.PluginManager instance with the Sherman AgentSpec loaded.

    Logs a DEBUG message when the manager is created.
    """
    pm = pluggy.PluginManager("sherman")
    pm.add_hookspecs(AgentSpec)

    _pm_logger.debug("plugin manager created")

    return pm


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

    @hookspec
    async def on_notify(
        self,
        channel: Channel,
        source: str,
        text: str,
        tool_call_id: str | None,
        meta: dict | None,
    ) -> None:
        """Called to inject a notification into a channel's processing queue.

        Plugins implementing this hook can inject messages into a channel so
        the agent loop sees and reacts to them. The AgentPlugin hookimpl
        enqueues a QueueItem(role="notification") on the channel's queue.

        Note: all parameters are required (no defaults) so pluggy forwards
        them correctly to hookimpl implementations. Callers should pass
        tool_call_id=None and meta=None when not applicable.

        Args:
            channel: The Channel to notify.
            source: Origin of the notification (e.g. "task").
            text: The notification content.
            tool_call_id: If set, the notification is a deferred tool result
                          and will be formatted as role="tool" in the conversation.
                          Pass None when not a deferred tool result.
            meta: Optional extensible metadata (task_id, etc.). Pass None if unused.
        """
