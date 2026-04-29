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
    HookStrategy                — enum of result-resolution strategies
    resolve_hook_results()      — post-process broadcast hook result lists
    get_dependency()            — typed plugin lookup
    validate_dependencies()     — dependency graph verification at startup
"""
from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, TypeVar

import apluggy as pluggy

if TYPE_CHECKING:
    from corvidae.channel import Channel
    from corvidae.context import ContextWindow, MessageType

T = TypeVar("T")

# NOTE: marker name is "corvidae", not "agent" as in design.md
hookspec = pluggy.HookspecMarker("corvidae")
hookimpl = pluggy.HookimplMarker("corvidae")

_pm_logger = logging.getLogger("corvidae.plugin_manager")
_resolve_logger = logging.getLogger("corvidae.hooks")


class HookStrategy(Enum):
    """Strategy for resolving a list of broadcast hook results into a single value."""

    REJECT_WINS = "reject_wins"
    ACCEPT_WINS = "accept_wins"
    VALUE_FIRST = "value_first"


def resolve_hook_results(
    results: list,
    hook_name: str,
    strategy: HookStrategy,
    *,
    pm: pluggy.PluginManager | None = None,
) -> object | None:
    """Resolve a list of broadcast hook results into a single value.

    Args:
        results: The list returned by ``await pm.ahook.<hook_name>(...)``.
        hook_name: The hook method name (used for logging and tiebreaking).
        strategy: One of HookStrategy.REJECT_WINS, ACCEPT_WINS, or VALUE_FIRST.
        pm: Plugin manager, required for VALUE_FIRST tiebreaking with multiple
            non-None results. If None, falls back to returning the first non-None
            result with a warning logged.

    Returns:
        A single resolved value, or None.
    """
    if strategy is HookStrategy.REJECT_WINS:
        non_none = [r for r in results if r is not None]
        if any(r is False for r in non_none):
            return False
        if any(r is True for r in non_none):
            return True
        return None

    if strategy is HookStrategy.ACCEPT_WINS:
        if any(r is True for r in results):
            return True
        return None

    # VALUE_FIRST
    non_none = [r for r in results if r is not None]
    if len(non_none) == 0:
        return None
    if len(non_none) == 1:
        return non_none[0]

    # Multiple non-None results: tiebreak by alphabetically-first plugin name.
    if pm is None:
        _resolve_logger.warning(
            "hook %s: multiple non-None results but pm is None; returning first non-None",
            hook_name,
        )
        return non_none[0]

    hook_caller = getattr(pm.hook, hook_name)
    # pluggy executes hooks in reversed(get_hookimpls()) order, so results[i]
    # corresponds to reversed(get_hookimpls())[i].
    impls = list(reversed(hook_caller.get_hookimpls()))
    candidates = []
    for impl, result in zip(impls, results):
        if result is not None:
            name = pm.get_name(impl.plugin) or type(impl.plugin).__name__
            candidates.append((name, result))
    candidates.sort(key=lambda pair: pair[0])
    _resolve_logger.warning(
        "hook %s: %d plugins returned non-None results: %s; using result from %s",
        hook_name,
        len(candidates),
        [c[0] for c in candidates],
        candidates[0][0],
    )
    return candidates[0][1]


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

        Note:
            Extension point for observability plugins (logging, metrics,
            analytics). No built-in plugin implements this hook.
            Implementations receive the final response after tool dispatch
            is complete.
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
        the agent loop sees and reacts to them. The Agent hookimpl
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

        Broadcast hook. All implementations run. Any False vetoes (reject-wins).
        Return False to reject, True to accept, None for no opinion.

        Note:
            Extension point for message filtering (rate limiting, blocklists,
            channel muting). No built-in plugin implements this hook. Return
            False to reject, True to force-accept, None for no opinion.
            Resolved with REJECT_WINS.
        """

    @hookspec
    async def on_llm_error(
        self, channel: Channel, error: Exception
    ) -> str | None:
        """Called when run_agent_turn raises an exception.

        Broadcast hook. Return a string to replace the default error message,
        or None to defer. If multiple plugins return non-None, the
        alphabetically-first plugin's result is used and a warning is logged.
        """

    @hookspec
    async def compact_conversation(
        self, channel: "Channel", conversation: "ContextWindow", max_tokens: int
    ) -> None:
        """Optionally replace the default compaction strategy.

        Broadcast hook. All implementations are called for side effects.
        Return value is not used by the caller.
        """

    @hookspec
    async def process_tool_result(
        self, tool_name: str, result: str, channel: "Channel | None"
    ) -> str | None:
        """Transform a tool result before it enters the conversation.

        Broadcast hook. Return a replacement string or None to keep the
        original. If multiple plugins return non-None, the alphabetically-first
        plugin's result is used and a warning is logged.

        Fires when execute_tool_call was invoked (success or exception).
        Pre-dispatch errors (JSON parse failure, unknown tool) skip this hook.
        """

    @hookspec
    async def on_idle(self) -> None:
        """Broadcast hook: fired when all queues are empty and cooldown has elapsed.

        Plugins can use this to perform periodic background work such as
        polling RSS feeds, checking email, or running maintenance tasks.
        """

    @hookspec
    async def load_conversation(self, channel: "Channel") -> "list[dict] | None":
        """Load conversation history for a channel.

        VALUE_FIRST resolution. Return a list of tagged message dicts if this
        plugin has stored history, or None to defer to another plugin.

        Args:
            channel: The Channel to load history for.
        """

    @hookspec
    async def on_conversation_event(
        self, channel: "Channel", message: dict, message_type: "MessageType"
    ) -> None:
        """Fired after a message is appended to the conversation.

        Broadcast hook (side effects only). Called after every conv.append().

        Args:
            channel: The Channel where the event occurred.
            message: The untagged message dict (no _message_type key).
            message_type: The MessageType of the message.
        """

    @hookspec
    async def on_compaction(
        self, channel: "Channel", summary_msg: dict, retain_count: int
    ) -> None:
        """Fired after compaction replaces older messages with a summary.

        Broadcast hook (side effects only). Called after replace_with_summary.

        Args:
            channel: The Channel where compaction occurred.
            summary_msg: The untagged summary message dict.
            retain_count: Number of messages retained alongside the summary.
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
                to access the ContextWindow for appending context entries.
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

        Broadcast hook. Return a transformed string or None to leave unchanged.
        If multiple plugins return non-None, the alphabetically-first plugin's
        result is used and a warning is logged.

        Args:
            channel: The Channel the response will be sent to.
            text: The raw response text from the LLM.
            result_message: The full assistant message dict from the LLM response.

        Returns:
            A transformed string, or None to leave text unchanged.
        """
