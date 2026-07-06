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
    HookStrategy                — enum with REJECT_WINS for gate hooks
    resolve_hook_results()      — resolve broadcast gate-hook result lists
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


def resolve_hook_results(
    results: list,
    hook_name: str,
    strategy: HookStrategy,
) -> object | None:
    """Resolve a list of broadcast hook results into a single value.

    Args:
        results: The list returned by ``await pm.ahook.<hook_name>(...)``.
        hook_name: The hook method name (used for logging).
        strategy: One of HookStrategy.REJECT_WINS.

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


def resolve_single_result(results: list, hook_name: str) -> object | None:
    """Exactly-one-non-None resolution (bootstrap-mapping §4.8).

    For broadcast hooks where exactly one implementation is expected to
    return a value (e.g. on_conversation_event's rowid from persistence).
    More than one non-None result is a configuration error: log it and
    use the first.
    """
    non_none = [r for r in results if r is not None]
    if len(non_none) > 1:
        _resolve_logger.error(
            "%s: %d plugins returned values; configuration error, using first",
            hook_name, len(non_none),
        )
    return non_none[0] if non_none else None


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


class _SeedHooksPlugin:
    """Internal plugin providing seed values for wrapper-chain hooks.

    Registered with trylast=True so it runs innermost in the chain.
    Returns the input value unchanged — wrappers above transform it.
    """

    @hookimpl(trylast=True)
    async def transform_display_text(self, text, **kwargs) -> str:
        return text

    @hookimpl(trylast=True)
    async def process_tool_result(self, result, **kwargs) -> str:
        return result


def create_plugin_manager() -> pluggy.PluginManager:
    """Create and configure the plugin manager with AgentSpec hooks.

    Returns:
        A pluggy.PluginManager instance with the Corvidae AgentSpec loaded.

    Logs a DEBUG message when the manager is created.
    """
    pm = pluggy.PluginManager("corvidae")
    pm.add_hookspecs(AgentSpec)
    pm.register(_SeedHooksPlugin(), name="_seed_hooks")

    _pm_logger.debug("plugin manager created")

    return pm


class CorvidaePlugin:
    """Optional base class for corvidae plugins.

    Stores pm and config as instance attributes in on_init, making them
    available before on_start is called. Plugins that need extra init
    should override on_init and call ``await super().on_init(pm, config)``
    first.

    Class attribute:
        depends_on: frozenset of plugin names this plugin requires to be
            registered. Checked by validate_dependencies at startup.
            Override as a class-level assignment, not by mutation.
    """

    depends_on: frozenset[str] = frozenset()

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        """Store pm and config. Called after all plugins are registered."""
        self.pm = pm
        self.config = config


class AgentSpec:
    """Hook specifications for the agent daemon.

    Plugins implement these hooks to receive lifecycle events, handle
    incoming messages, send outgoing messages, and register tools with
    the agent loop. All hooks are optional; a plugin only needs to
    implement the hooks it cares about.
    """

    @hookspec
    async def on_init(self, pm, config: dict) -> None:
        """Called after all plugins are registered, before on_start.

        Use for storing pm, reading config values, and resolving references
        to other plugins. Do not create runtime resources here — use on_start.

        Args:
            pm: The plugin manager.
            config: The full parsed YAML config dict.
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
    async def send_thinking(self, channel: Channel, text: str) -> None:
        """Called to display reasoning/thinking content from the LLM.

        Transport plugins may render this differently from the main response
        (e.g., in a dim or distinct color). Only fires when the LLM returns
        reasoning_content in its response.

        Args:
            channel: The Channel the thinking is associated with.
            text: The reasoning/thinking content from the LLM.
        """

    @hookspec
    async def send_tool_status(self, channel: Channel, tool_name: str, status: str, args_summary: str | None = None, result_summary: str | None = None) -> None:
        """Called to display tool call lifecycle events.

        Transport plugins may render tool activity as inline status
        indicators. Fires when a tool is dispatched and when it completes.

        Args:
            channel: The Channel the tool call is associated with.
            tool_name: Name of the tool being called.
            status: One of "dispatched" or "completed".
            args_summary: Short summary of arguments (for "dispatched"), or None.
            result_summary: Short summary of result (for "completed"), or None.
        """

    @hookspec
    async def send_progress(self, channel: Channel, text: str) -> None:
        """Called to display intermediate assistant text before tool dispatch.

        Fires when the LLM produces text content alongside tool calls.
        Transport plugins may render this as a status indicator (e.g., grey text)
        since it is not the final response.

        Args:
            channel: The Channel the text is associated with.
            text: Intermediate text content from the LLM response.
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

    @hookspec(firstresult=True)
    async def on_llm_error(
        self, channel: Channel, error: Exception
    ) -> str | None:
        """Called when run_agent_turn raises an exception.

        Sequential hook (firstresult=True). Return a string to replace the
        default error message, or None to defer to the next handler. The first
        non-None return value wins; subsequent handlers are not called.
        """

    @hookspec(firstresult=True)
    async def compact_conversation(
        self, channel: "Channel", conversation: "ContextWindow", max_tokens: int
    ) -> "bool | None":
        """Compact the conversation when approaching the context limit.

        Sequential hook (firstresult=True). Handlers are called in priority
        order. The first handler that returns a non-None value wins and the
        chain stops. Return True to signal that compaction was performed,
        or None to defer to the next handler.

        Ordering:
        - tryfirst handlers run first (side-effect handlers that return
          None and do not stop the chain).
        - Default-priority handlers run next (third-party replacements).
        - trylast handlers run last (CompactionPlugin as fallback).

        Args:
            channel: The Channel being compacted.
            conversation: The ContextWindow to compact.
            max_tokens: The channel's max_context_tokens limit.
        """

    @hookspec(firstresult=True)
    async def process_tool_result(
        self, tool_name: str, result: str, channel: "Channel | None"
    ) -> str | None:
        """Transform a tool result before it enters the conversation.

        Wrapper chain hook (firstresult=True). Use ``@hookimpl(wrapper=True)``
        to wrap the chain result. The seed plugin returns the input unchanged;
        wrappers above it compose transforms in LIFO order.

        Example::

            @hookimpl(wrapper=True)
            def process_tool_result(self, **kwargs):
                result = yield
                if result is not None:
                    return result.upper()
                return result

        Fires when execute_tool_call was invoked (success or exception).
        Pre-dispatch errors (JSON parse failure, unknown tool) skip this hook.
        """

    @hookspec
    async def on_llm_request(
        self,
        role: str,
        model: str,
        request_id: str,
        message_count: int,
        tool_count: int,
        attribution: dict,
    ) -> None:
        """Fired immediately before an LLM chat-completion call is made.

        Broadcast hook (side effects only). Fires from the LLMClient
        chokepoint via an observer injected by LLMPlugin, so every LLM
        call in the system is covered — turn-loop, compaction, subagent,
        and future background calls alike.

        The request payload is summarized (counts), not shipped wholesale —
        full messages in a broadcast hook would copy the entire prompt per
        call. A debugging consumer needing full payloads is a new hook, not
        a widening of this one.

        Args:
            role: The LLMPlugin role that made the call ("main", "background").
            model: The model name configured for the client.
            request_id: uuid hex minted per call; pairs request with response.
            message_count: Number of messages in the request payload.
            tool_count: Number of tool schemas in the request payload.
            attribution: Snapshot of corvidae.attribution.get_attribution()
                at call time (keys like "stage", "channel_id"; may be empty).
        """

    @hookspec
    async def on_llm_response(
        self,
        role: str,
        model: str,
        request_id: str,
        usage: dict | None,
        latency_ms: float,
        attribution: dict,
        error: str | None,
    ) -> None:
        """Fired exactly once per LLM call, after it succeeds or terminally fails.

        Broadcast hook (side effects only). Retried transient attempts do
        NOT fire this hook — one on_llm_request pairs with exactly one
        on_llm_response, matched by request_id.

        Args:
            role: The LLMPlugin role that made the call ("main", "background").
            model: The model name configured for the client.
            request_id: The id minted for the matching on_llm_request.
            usage: The response's "usage" field verbatim, or None (missing
                usage or terminal failure).
            latency_ms: Wall-clock latency of the final attempt in ms.
            attribution: Snapshot of corvidae.attribution.get_attribution().
            error: None on success; exception string on terminal failure.
        """

    @hookspec
    async def on_metrics(
        self, name: str, value: float, tags: dict[str, str]
    ) -> None:
        """Broadcast hook: consume a metric event (side effects only).

        The shape follows the StatsD/OpenTelemetry common denominator — a
        named numeric value with dimensional tags (dotted-hierarchy names,
        e.g. "llm.tokens.prompt", "llm.latency_ms"). Metric types (counter
        vs gauge) are inferred from the name convention, not modeled.

        Any plugin can emit metrics by calling
        ``await self.pm.ahook.on_metrics(name=..., value=..., tags=...)``
        from tool functions, other hook implementations, or background tasks.

        Reentrancy constraint: never call ``pm.ahook.on_metrics(...)`` from
        inside an on_metrics implementation — pluggy dispatches back into
        the same implementation and recurses forever. A plugin that both
        produces and consumes metrics emits from its other hooks, never
        from on_metrics itself.

        Args:
            name: Dotted metric name (e.g. "llm.tokens.total").
            value: The numeric measurement.
            tags: Dimensional tags (e.g. {"role": "main", "stage": "turn"}).
        """

    @hookspec
    async def on_idle(self) -> None:
        """Broadcast hook: fired when all queues are empty and cooldown has elapsed.

        Plugins can use this to perform periodic background work such as
        polling RSS feeds, checking email, or running maintenance tasks.
        """

    @hookspec(firstresult=True)
    async def load_conversation(self, channel: "Channel") -> "list[dict] | None":
        """Load conversation history for a channel.

        Sequential hook (firstresult=True). Return a list of tagged message
        dicts if this plugin has stored history, or None to defer to the next
        handler. The first non-None return value wins.

        Args:
            channel: The Channel to load history for.
        """

    @hookspec
    async def on_conversation_event(
        self, channel: "Channel", message: dict, message_type: "MessageType"
    ) -> "int | None":
        """Fired after a message is appended to the conversation.

        Broadcast hook. Called after every conv.append(). Persistence
        returns the inserted ``message_log`` rowid; exactly one
        implementation may return non-None; all others return None.
        Callers resolve the result list with ``resolve_single_result``
        and attach the rowid to the in-window message copy as ``_db_id``
        (bootstrap-mapping §4.8).

        Args:
            channel: The Channel where the event occurred.
            message: The untagged message dict (no _message_type key).
            message_type: The MessageType of the message.

        Returns:
            The message_log rowid from the persistence implementation,
            None from every other implementation.
        """

    @hookspec
    async def on_compaction(
        self,
        channel: "Channel",
        summary_msg: dict,
        retain_count: int,
        compacted_ids: list[int],
    ) -> None:
        """Fired after compaction replaces older messages with a summary.

        Broadcast hook (side effects only). Called after replace_with_summary.

        Args:
            channel: The Channel where compaction occurred.
            summary_msg: The untagged summary message dict.
            retain_count: Number of messages retained alongside the summary.
            compacted_ids: The ``message_log`` rowids (``_db_id`` tags) of
                the messages that were removed by this compaction, in
                window order. Empty list when unknown (e.g. messages that
                were never persisted). Consumers such as MemoryPlugin use
                this range for consolidation (bootstrap-mapping §3.1, §4.8).
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

    @hookspec(firstresult=True)
    async def transform_display_text(
        self, channel: "Channel", text: str, result_message: dict
    ) -> "str | None":
        """Called before sending the final text response to the channel.

        Wrapper chain hook (firstresult=True). Use ``@hookimpl(wrapper=True)``
        to wrap the chain result. The seed plugin returns the input text
        unchanged; wrappers above it compose transforms in LIFO order.

        Example::

            @hookimpl(wrapper=True)
            def transform_display_text(self, **kwargs):
                result = yield
                if result is not None:
                    return result.upper()
                return result

        Args:
            channel: The Channel the response will be sent to.
            text: The raw response text from the LLM.
            result_message: The full assistant message dict from the LLM response.

        Returns:
            A transformed string, or None to leave text unchanged.
        """

    @hookspec
    async def on_plugin_added(self, name: str, plugin: object) -> None:
        """Broadcast after a plugin is registered and initialized at runtime.

        Args:
            name: The registered name of the plugin.
            plugin: The plugin instance that was added.
        """

    @hookspec
    async def on_plugin_removed(self, name: str) -> None:
        """Broadcast after a plugin is unregistered at runtime.

        Args:
            name: The registered name of the plugin that was removed.
        """

    @hookspec
    async def on_config_reload(self, config: dict) -> None:
        """Called when agent.yaml is reloaded from disk.

        Plugins should re-read their configuration from the new config dict.
        The config dict has already been merged with CLI overrides and validated.

        This hook fires on the event loop thread. Plugins must not block.
        Errors are caught per-plugin and logged; they do not prevent other
        plugins from receiving the update.

        Args:
            config: The full re-parsed and merged config dict.
        """
