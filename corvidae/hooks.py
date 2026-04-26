"""Plugin system for the Corvidae agent daemon.

Defines the AgentSpec hookspecs that plugins implement, the hookimpl
marker for decorating implementations, and utility functions for the
plugin manager lifecycle.

Exports:
    hookspec        — decorator for hookspec methods on AgentSpec
    hookimpl        — decorator for plugin implementations
    AgentSpec       — hook specifications for all lifecycle, messaging,
                      and extension-point hooks
    create_plugin_manager()     — create and configure the PluginManager
    call_firstresult_hook()     — async firstresult helper (apluggy workaround)
    get_dependency()            — typed plugin lookup
    validate_dependencies()     — dependency graph verification at startup
"""
from __future__ import annotations

import asyncio
import inspect
import logging
from typing import TYPE_CHECKING, TypeVar

import apluggy as pluggy

if TYPE_CHECKING:
    from corvidae.channel import Channel
    from corvidae.conversation import ConversationLog
    from corvidae.llm import LLMClient

T = TypeVar("T")

# NOTE: marker name is "corvidae", not "agent" as in design.md
hookspec = pluggy.HookspecMarker("corvidae")
hookimpl = pluggy.HookimplMarker("corvidae")

_pm_logger = logging.getLogger("corvidae.plugin_manager")


def get_dependency(pm: pluggy.PluginManager, name: str, expected_type: type[T]) -> T:
    """Typed wrapper around pm.get_plugin(name).

    Raises RuntimeError if the plugin is not registered.
    Raises TypeError if the plugin is not an instance of expected_type.
    """
    plugin = pm.get_plugin(name)
    if plugin is None:
        raise RuntimeError(f"Required plugin {name!r} is not registered")
    if not isinstance(plugin, expected_type):
        raise TypeError(
            f"Plugin {name!r} is {type(plugin).__name__}, expected {expected_type.__name__}"
        )
    return plugin


def validate_dependencies(pm: pluggy.PluginManager) -> None:
    """Verify all declared plugin dependencies are registered and acyclic.

    Iterates registered plugins, checks each for a `depends_on` attribute,
    raises RuntimeError if any declared dependency is not registered, and
    detects dependency cycles via DFS.
    """
    # Build adjacency graph: plugin_name -> set of dependency names
    graph: dict[str, set[str]] = {}

    for plugin in pm.get_plugins():
        depends_on = getattr(plugin, "depends_on", None)
        if depends_on is None:
            continue
        # Find the registered name for this plugin
        plugin_name = None
        for p_name, p in pm.list_name_plugin():
            if p is plugin:
                plugin_name = p_name
                break
        if plugin_name is None:
            continue
        for dep_name in depends_on:
            if pm.get_plugin(dep_name) is None:
                raise RuntimeError(
                    f"Plugin {type(plugin).__name__} depends on {dep_name!r}, "
                    f"which is not registered"
                )
        graph[plugin_name] = set(depends_on)

    # Detect cycles via DFS with three-color marking
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {name: WHITE for name in graph}
    path: list[str] = []

    def dfs(node: str) -> None:
        if node not in color:
            return  # leaf node with no depends_on
        if color[node] == BLACK:
            return
        if color[node] == GRAY:
            cycle_start = path.index(node)
            cycle = path[cycle_start:] + [node]
            raise RuntimeError(
                f"Dependency cycle detected: {' -> '.join(cycle)}"
            )
        color[node] = GRAY
        path.append(node)
        for dep in graph.get(node, set()):
            dfs(dep)
        path.pop()
        color[node] = BLACK

    for node in list(graph.keys()):
        if color.get(node, WHITE) == WHITE:
            dfs(node)


def create_plugin_manager() -> pluggy.PluginManager:
    """Create and configure the plugin manager with AgentSpec hooks.

    Returns:
        A pluggy.PluginManager instance with the Corvidae AgentSpec loaded.

    Logs a DEBUG message when the manager is created.
    """
    pm = pluggy.PluginManager("corvidae")
    pm.add_hookspecs(AgentSpec)

    _pm_logger.debug("plugin manager created")

    return pm


async def call_firstresult_hook(pm: pluggy.PluginManager, hook_name: str, **kwargs) -> object | None:
    """Call an async hook, returning the first non-None result.

    Workaround for apluggy's inability to handle firstresult=True on async hooks.
    Iterates hook implementations in priority order: tryfirst implementations
    first, then regular implementations in registration order (FIFO), then
    trylast implementations. Returns the first non-None result, or None if all
    impls return None or no impls are registered.

    Args:
        pm: The plugin manager instance.
        hook_name: The hook method name on AgentSpec (e.g. "should_process_message").
        **kwargs: Arguments to pass to each hook implementation.

    Returns:
        The first non-None return value, or None.

    Note: Hook wrappers (@hookimpl(wrapper=True) and
    @hookimpl(hookwrapper=True)) are not supported and will be silently
    skipped.
    """
    hook_caller = getattr(pm.hook, hook_name, None)
    if hook_caller is None:
        return None
    # get_hookimpls() returns implementations in registration order.
    # Sort by priority: tryfirst first (0), regular middle (1), trylast last (2).
    # Within each group, Python's stable sort preserves registration order.
    impls = sorted(
        hook_caller.get_hookimpls(),
        key=lambda i: 0 if i.tryfirst else (2 if i.trylast else 1),
    )
    for impl in impls:
        # Skip wrappers -- they are not direct result producers.
        if impl.wrapper or impl.hookwrapper:
            continue
        # Filter kwargs to only those the impl accepts.
        filtered = {k: v for k, v in kwargs.items() if k in impl.argnames}
        result = impl.function(**filtered)
        if inspect.isawaitable(result):
            result = await result
        if result is not None:
            return result
    return None


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

        Pluggy broadcasts this hook to all registered transport plugins.
        Each transport must filter for its own channels by calling
        ``channel.matches_transport("<transport_name>")`` and returning
        early if the channel does not belong to it.

        ``latency_ms`` is optional — transports that do not display
        timing information (e.g. IRC) may omit it from their hookimpl
        signature entirely. Pluggy tolerates hookimpls that omit
        optional parameters.

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
        enqueues a QueueItem(role=QueueItemRole.NOTIFICATION) on the channel's queue.

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

    @hookspec
    async def should_process_message(
        self, channel: Channel, sender: str, text: str
    ) -> bool | None:
        """Gate hook: decide whether to process an incoming message.

        Called before the message is enqueued. Return False to reject,
        True to explicitly accept (short-circuits), None for no opinion.
        First non-None result wins.
        Note: Hook wrappers are not supported for this hook.
        """

    @hookspec
    async def on_llm_error(
        self, channel: Channel, error: Exception
    ) -> str | None:
        """Called when run_agent_turn raises an exception.

        Return a string to use as the error message sent to the channel.
        Return None to use the default error message.
        First non-None result wins.
        Note: Hook wrappers are not supported for this hook.
        """

    @hookspec
    async def compact_conversation(
        self, conversation: "ConversationLog", client: "LLMClient", max_tokens: int
    ) -> bool | None:
        """Optionally replace the default compaction strategy.

        Return True if compaction was handled (skip default).
        Return None to defer to the next implementation or the default.
        Do NOT return False: call_firstresult_hook stops iteration on any
        non-None result, so False would stop other plugins from running but
        the call site (which checks truthiness) would still run the default —
        a confusing combination. Use None to defer, True to handle.
        Note: Hook wrappers are not supported for this hook.
        """

    @hookspec
    async def process_tool_result(
        self, tool_name: str, result: str, channel: "Channel | None"
    ) -> str | None:
        """Transform a tool result before it enters the conversation.

        Called after execute_tool_call returns. Return a replacement string
        to use instead of the default result. Return None to keep the default.
        First non-None result wins.
        Note: Hook wrappers are not supported for this hook.
        Note: This hook only fires during subagent execution (run_agent_loop),
        not during interactive message processing via AgentPlugin.
        """

    @hookspec
    async def on_idle(self) -> None:
        """Broadcast hook: fired when all queues are empty and cooldown has elapsed.

        Plugins can use this to perform periodic background work such as
        polling RSS feeds, checking email, or running maintenance tasks.
        """

    @hookspec
    async def ensure_conversation(self, channel: "Channel") -> "bool | None":
        """Lazy-initialize a ConversationLog on a channel.

        Called before each agent turn when channel.conversation is None.
        Return True if conversation was initialized (or was already present),
        None to defer to the next implementation.
        First non-None result wins (called via call_firstresult_hook).

        Args:
            channel: The Channel that needs a ConversationLog.
        """

    @hookspec
    async def before_agent_turn(self, channel: "Channel") -> None:
        """Called before each LLM invocation, after compaction.

        Plugins can inject context entries into the conversation log by calling
        ``channel.conversation.append(msg, message_type=MessageType.CONTEXT)``.
        Injected entries will appear in the prompt for this turn. They are
        persisted to the DB and survive compaction (compaction only summarizes
        MESSAGE entries).

        This hook fires on every turn including tool-result turns. Plugins can
        inspect channel state to filter if needed.

        Args:
            channel: The Channel being processed. Use ``channel.conversation``
                to access the ConversationLog for appending context entries.
        """

    @hookspec
    async def after_persist_assistant(self, channel: "Channel", message: dict) -> None:
        """Called after the assistant message has been persisted to the conversation log.

        Plugins may mutate the in-memory message dict (e.g., to strip
        reasoning_content). The DB copy is already written at call time;
        mutations affect only subsequent prompt builds. No return value is used.

        Args:
            channel: The Channel where the conversation is happening.
            message: The in-memory assistant message dict just appended to the
                conversation log. Mutations affect subsequent prompt builds only.
        """

    @hookspec
    async def transform_display_text(
        self, channel: "Channel", text: str, result_message: dict
    ) -> "str | None":
        """Called before sending the final text response to the channel.

        Return a transformed string to replace text, or None to leave it
        unchanged. First non-None result wins.
        Note: Hook wrappers are not supported for this hook.

        Args:
            channel: The Channel the response will be sent to.
            text: The raw response text from the LLM.
            result_message: The full assistant message dict from the LLM response.

        Returns:
            A transformed string, or None to leave text unchanged.
        """
