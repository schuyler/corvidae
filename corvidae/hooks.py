"""Plugin system for the Corvidae agent daemon.

Defines the AgentSpec hookspecs that plugins implement, the hookimpl
marker for decorating implementations, and utility functions for the
plugin manager lifecycle.

Exports:
    hookspec        — decorator for hookspec methods on AgentSpec
    hookimpl        — decorator for plugin implementations
    AgentSpec       — hook specifications for all lifecycle, messaging,
                      and extension-point hooks
    CorvidaePlugin  — optional plugin base class (stores pm and config)
    TransportPlugin — transport base class: the outbound ``send`` filter
                      and dispatch to send_* methods
    create_plugin_manager()     — create and configure the PluginManager
    resolve_reject_wins()       — resolve broadcast gate-hook result lists
    get_dependency()            — typed plugin lookup
    validate_dependencies()     — dependency graph verification at startup
"""
from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, ClassVar, TypeVar

import apluggy as pluggy

from corvidae.outbound import (
    Outbound,
    OutboundMessage,
    OutboundProgress,
    OutboundThinking,
    OutboundToolStatus,
)

if TYPE_CHECKING:
    from corvidae.channel import Channel
    from corvidae.context import ContextWindow, MessageType

T = TypeVar("T")

# NOTE: marker name is "corvidae", not "agent" as in design.md
hookspec = pluggy.HookspecMarker("corvidae")
hookimpl = pluggy.HookimplMarker("corvidae")

_pm_logger = logging.getLogger("corvidae.plugin_manager")
_resolve_logger = logging.getLogger("corvidae.hooks")
_transport_logger = logging.getLogger("corvidae.hooks")


# Hook grammar (docs/plugin-guide.md, "Hook grammar"): prefix class ->
# (prefixes, the pluggy spec options every member must have). Shared by
# tests/test_hook_grammar.py (AgentSpec) and install_plugin_hookspecs
# (plugin-declared specs).
HOOK_GRAMMAR: dict[str, tuple[tuple[str, ...], dict[str, bool]]] = {
    "observe": (("on_", "before_", "after_"), {"firstresult": False}),
    "veto": (("should_",), {"firstresult": False}),
    "resolve": (("resolve_",), {"firstresult": True}),
    "transform": (("transform_",), {"firstresult": True}),
}

# Bare imperative names: broadcast, caller-defined use of the result list.
# Kernel-only; plugin-declared hooks must use a prefixed class.
IMPERATIVE_HOOKS = frozenset({"send", "register_tools", "persist_message"})


def grammar_class(name: str) -> str | None:
    """Return the grammar class whose prefix ``name`` starts with, or None."""
    for klass, (prefixes, _opts) in HOOK_GRAMMAR.items():
        if name.startswith(prefixes):
            return klass
    return None


def _split_grammar_prefix(name: str) -> tuple[str | None, str]:
    """Split a hook name into (grammar prefix, rest); (None, name) if unprefixed."""
    for prefixes, _opts in HOOK_GRAMMAR.values():
        for prefix in prefixes:
            if name.startswith(prefix):
                return prefix, name[len(prefix):]
    return None, name


def _accepted_prefixes() -> str:
    """Every grammar prefix, comma-separated, for error messages."""
    return ", ".join(p for prefixes, _ in HOOK_GRAMMAR.values() for p in prefixes)


def install_plugin_hookspecs(pm: pluggy.PluginManager) -> None:
    """Validate and install every plugin's declared hookspecs (#229).

    A plugin owns hooks by setting ``hookspecs`` on its class to a class of
    ``@hookspec`` methods. Called once at startup, after every plugin is
    registered and before the arg-binding guard and ``on_init``.

    Two passes, so no outcome depends on registration order (entry-point
    order is not deterministic):

    1. Collect every declaration and fail on any hook name with more than
       one owner, counting hooks that already have a spec (``AgentSpec``).
    2. Per plugin: check the hook grammar, the owner segment
       (``<prefix><plugin>_<name>``) and the pluggy options, install the
       specs, then check each ``transform_`` hook has the owner's own
       ``trylast`` seed.

    Raises:
        PluginValidationError: naming the hook, the plugin and the fix.
    """
    # Pass 1a: owners of every hook that already has a spec, so a plugin
    # redeclaring an AgentSpec name is reported as a duplicate.
    owners: dict[str, list[str]] = {}
    for hook_name, caller in vars(pm.hook).items():
        spec = getattr(caller, "spec", None)
        if spec is None:
            continue
        namespace = getattr(spec, "namespace", None)
        label = "AgentSpec" if namespace is AgentSpec else getattr(namespace, "__name__", repr(namespace))
        owners[hook_name] = [label]

    # Pass 1b: collect each plugin's declaration. Blocked plugins are
    # (name, None) and declare nothing; getattr on the class lets a base
    # class declare for its subclasses.
    declared: list[tuple[str, object, type, list[str]]] = []
    first_owner: dict[str, object] = {}
    for plugin_name, plugin in pm.list_name_plugin():
        if plugin is None:
            continue
        spec_cls = getattr(type(plugin), "hookspecs", None)
        if spec_cls is None:
            continue
        names = sorted(
            attr for attr in dir(spec_cls)
            if pm.parse_hookspec_opts(spec_cls, attr) is not None
        )
        if not names:
            raise pluggy.PluginValidationError(
                plugin,
                f"Plugin {plugin_name!r} sets hookspecs = {spec_cls.__name__}, "
                f"which has no @hookspec methods. Decorate its hook methods "
                f"with corvidae.hooks.hookspec, or remove the hookspecs attribute.",
            )
        declared.append((plugin_name, plugin, spec_cls, names))
        for hook_name in names:
            owners.setdefault(hook_name, []).append(f"plugin {plugin_name!r}")
            first_owner.setdefault(hook_name, plugin)

    # Pass 1c: a hook has exactly one owner.
    for hook_name in sorted(owners):
        if len(owners[hook_name]) > 1:
            raise pluggy.PluginValidationError(
                first_owner[hook_name],
                f"Hook {hook_name!r} is declared by more than one owner: "
                f"{', '.join(owners[hook_name])}. A hook has one owner: "
                f"depend on the existing owner and call its hook, or rename "
                f"the new declaration (plugin hooks are named "
                f"<prefix><plugin>_<name>).",
            )

    # Pass 2: per-plugin grammar, owner segment and options; then install
    # and check transform seeds.
    for plugin_name, plugin, spec_cls, names in declared:
        for hook_name in names:
            _check_plugin_hook_name(pm, plugin_name, plugin, spec_cls, hook_name)
        pm.add_hookspecs(spec_cls)
        for hook_name in names:
            if grammar_class(hook_name) == "transform":
                _check_transform_seed(pm, plugin_name, plugin, hook_name)


def _check_plugin_hook_name(
    pm: pluggy.PluginManager, plugin_name: str, plugin: object, spec_cls: type, hook_name: str
) -> None:
    """Check one declared hook's grammar prefix, owner segment and options."""
    # Grammar prefix: bare imperatives and unprefixed names are kernel-only.
    klass = grammar_class(hook_name)
    if klass is None:
        raise pluggy.PluginValidationError(
            plugin,
            f"Plugin {plugin_name!r} declares hook {hook_name!r}, which has no "
            f"grammar prefix. Plugin hooks start with one of {_accepted_prefixes()} "
            f"followed by the plugin name, e.g. on_{plugin_name}_<name>; bare "
            f"imperatives ({', '.join(sorted(IMPERATIVE_HOOKS))}) are kernel-only.",
        )

    # Owner segment: "<plugin>_" right after the prefix, then a non-empty name.
    prefix, rest = _split_grammar_prefix(hook_name)
    segment = f"{plugin_name}_"
    if not rest.startswith(segment) or len(rest) == len(segment):
        raise pluggy.PluginValidationError(
            plugin,
            f"Plugin {plugin_name!r} declares hook {hook_name!r}, which doesn't "
            f"name its owner. Plugin hooks put the plugin name right after the "
            f"grammar prefix: {prefix}{plugin_name}_<name>.",
        )

    # Options: the prefix fixes the pluggy spec options.
    opts = pm.parse_hookspec_opts(spec_cls, hook_name) or {}
    for option, required in HOOK_GRAMMAR[klass][1].items():
        if bool(opts.get(option)) is not required:
            raise pluggy.PluginValidationError(
                plugin,
                f"Plugin {plugin_name!r} declares hook {hook_name!r} with "
                f"{option}={bool(opts.get(option))}, but {prefix} hooks ({klass}) "
                f"require {option}={required}. Fix the @hookspec options or use "
                f"the prefix that matches them (docs/plugin-guide.md, Hook grammar).",
            )


def _check_transform_seed(
    pm: pluggy.PluginManager, plugin_name: str, plugin: object, hook_name: str
) -> None:
    """Check the owner registers its own trylast seed for a transform hook."""
    seeds = [
        impl for impl in getattr(pm.hook, hook_name).get_hookimpls()
        if impl.plugin is plugin and impl.trylast
    ]
    if not seeds:
        raise pluggy.PluginValidationError(
            plugin,
            f"Plugin {plugin_name!r} declares transform hook {hook_name!r} but "
            f"registers no seed for it. Add an @hookimpl(trylast=True) "
            f"{hook_name} to {type(plugin).__name__} that returns its input "
            f"unchanged, as _SeedHooksPlugin does for kernel transforms.",
        )


def report_orphan_hookimpls(pm: pluggy.PluginManager) -> None:
    """Log hookimpls that have no spec (#229).

    An impl with no spec is never verified and usually never called: a typo
    in the hook name, or a hook whose owning plugin isn't loaded. Logged at
    WARNING naming the plugin and hook, except when the hook's owner segment
    names one of the implementer's ``optional_depends_on`` plugins and that
    plugin isn't loaded (blocked or not installed): that's expected, so DEBUG.
    """
    for hook_name, caller in list(vars(pm.hook).items()):
        # Only real hook callers without a spec; skips anything else on
        # pm.hook (and mocks, whose spec attribute is never None).
        if getattr(caller, "spec", None) is not None or not hasattr(caller, "get_hookimpls"):
            continue
        _prefix, rest = _split_grammar_prefix(hook_name)
        for impl in caller.get_hookimpls():
            plugin_name = pm.get_name(impl.plugin)
            # The absent optional owner whose segment this hook carries, if any.
            absent_owner = next(
                (
                    dep for dep in sorted(getattr(impl.plugin, "optional_depends_on", ()))
                    if rest.startswith(f"{dep}_") and pm.get_plugin(dep) is None
                ),
                None,
            )
            if absent_owner is not None:
                _pm_logger.debug(
                    "plugin %r implements %r, whose owner %r is not loaded "
                    "(optional dependency); the hookimpl is inactive",
                    plugin_name, hook_name, absent_owner,
                )
            else:
                _pm_logger.warning(
                    "plugin %r implements hook %r, which no loaded plugin "
                    "declares; check the hook name for a typo, or add its "
                    "owner to depends_on or optional_depends_on",
                    plugin_name, hook_name,
                )


def resolve_reject_wins(results: list) -> bool | None:
    """Resolve a list of broadcast gate-hook results with reject-wins semantics.

    Any False vetoes; else any True accepts; else None (no opinion).

    Args:
        results: The list returned by ``await pm.ahook.<hook_name>(...)``.

    Returns:
        False if any result is False, True if any result is True (and none
        is False), else None.
    """
    non_none = [r for r in results if r is not None]
    if any(r is False for r in non_none):
        return False
    if any(r is True for r in non_none):
        return True
    return None


def resolve_single_result(results: list, hook_name: str) -> object | None:
    """Exactly-one-non-None resolution (bootstrap-mapping §4.8).

    For broadcast hooks where exactly one implementation is expected to
    return a value (e.g. persist_message's rowid from persistence).
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
    async def transform_tool_result(self, result, **kwargs) -> str:
        return result


def _check_hook_arg_binding(pm: pluggy.PluginManager) -> None:
    """Guard against pluggy's silent-drop hookimpl arg-binding bug class.

    pluggy forwards a caller's value to a hookimpl param ONLY when that
    param has NO default in the hookimpl's own signature. A hookimpl that
    re-declares a spec-REQUIRED param WITH a default (e.g. ``config=None``)
    silently receives its own default instead of the caller's value --
    pluggy raises no error for this, and pluggy's own ``_verify_hook`` does
    not catch it either, because the impl signature is individually valid;
    the problem only exists in the *relationship* between the spec and the
    impl.

    The invariant enforced here: for each hookcaller, the spec's required
    params are ``spec.argnames`` (spec params with NO default). For each
    registered impl, the params the impl itself defaults are
    ``impl.kwargnames``. If a required spec param also appears in an
    impl's kwargnames, that impl is silently dropping a required
    caller-supplied value -- raise PluginValidationError naming the
    plugin, hook, and offending param(s).

    Spec-OPTIONAL params (params the spec itself declares with a default,
    e.g. ``latency_ms``, ``args_summary``) are not required, so an impl
    defaulting those is legitimate and must not trigger this guard.

    This function reads pluggy's internal HookCaller/HookSpec/HookImpl
    attributes (``.spec``, ``.argnames``, ``.kwargnames``,
    ``.get_hookimpls()``), which are not part of pluggy's public API and
    could change in a future pluggy release. Any exception raised while
    walking these internals (other than PluginValidationError raised by
    this function itself upon finding a real violation) is caught and
    logged as a warning; the guard degrades to "skipped" rather than
    crashing app startup.

    Args:
        pm: The plugin manager to check, after all hookspecs and plugins
            have been registered.

    Raises:
        PluginValidationError: if any registered impl defaults a
            spec-required param for some hook.
    """
    try:
        hookcallers = list(pm.hook.__dict__.values())
    except Exception as exc:  # noqa: BLE001 - defensive, see docstring
        warnings.warn(
            f"hook arg-binding guard skipped: pluggy internals unavailable ({exc})",
            stacklevel=2,
        )
        return

    for caller in hookcallers:
        try:
            spec = caller.spec
            if spec is None:
                continue
            required_params = set(spec.argnames)
            if not required_params:
                continue
            hookimpls = caller.get_hookimpls()
            hook_name = caller.name
        except Exception as exc:  # noqa: BLE001 - defensive, see docstring
            warnings.warn(
                f"hook arg-binding guard skipped for a hookcaller: "
                f"pluggy internals unavailable ({exc})",
                stacklevel=2,
            )
            continue

        for impl in hookimpls:
            try:
                dropped = required_params & set(impl.kwargnames)
                plugin = impl.plugin
            except Exception as exc:  # noqa: BLE001 - defensive, see docstring
                warnings.warn(
                    f"hook arg-binding guard skipped for a hookimpl: "
                    f"pluggy internals unavailable ({exc})",
                    stacklevel=2,
                )
                continue

            if dropped:
                try:
                    plugin_name = pm.get_name(plugin) or repr(plugin)
                except Exception:  # noqa: BLE001 - defensive, see docstring
                    plugin_name = repr(plugin)
                raise pluggy.PluginValidationError(
                    plugin,
                    f"Plugin {plugin_name!r} hookimpl {hook_name!r} "
                    f"declares spec-required param(s) "
                    f"{sorted(dropped)!r} with a default value. "
                    f"pluggy will not forward the caller's value for "
                    f"these params -- it silently uses the impl's own "
                    f"default instead. Remove the default(s) from "
                    f"{sorted(dropped)!r} in this hookimpl's signature.",
                )


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

    _check_hook_arg_binding(pm)

    return pm


class CorvidaePlugin:
    """Optional base class for corvidae plugins.

    Stores pm and config as instance attributes in on_init, making them
    available before on_start is called. Plugins that need extra init
    should override on_init and call ``await super().on_init(pm, config)``
    first.

    Class attributes (override as class-level assignments, not by mutation):
        depends_on: frozenset of plugin names this plugin requires to be
            registered. Checked by validate_dependencies at startup.
        optional_depends_on: frozenset of plugin names this plugin uses
            when they are loaded and works without. Not checked; resolve
            with ``pm.get_plugin(name)`` and treat None as absent. Impls of
            an absent optional owner's hooks stay inactive and are logged at
            DEBUG by report_orphan_hookimpls.
        hookspecs: a class of ``@hookspec`` methods this plugin owns, or
            None. Installed by install_plugin_hookspecs at startup; names
            follow the hook grammar with the plugin name after the prefix
            (``on_<plugin>_<name>``).
    """

    depends_on: frozenset[str] = frozenset()
    optional_depends_on: frozenset[str] = frozenset()
    hookspecs: ClassVar[type | None] = None

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        """Store pm and config. Called after all plugins are registered."""
        self.pm = pm
        self.config = config


class TransportPlugin(CorvidaePlugin):
    """Base class for transports: the one outbound broadcast filter.

    The ``send`` hook reaches every transport. This class implements it
    once: it drops events for channels the transport does not own and
    dispatches the rest by type to ``send_message``, ``send_thinking``,
    ``send_tool_status`` and ``send_progress``. Those are plain methods,
    not hookimpls, and default to no-ops, so a transport overrides only
    what it can display.

    Class attributes:
        transport: The transport name, the ``transport`` half of the
            channel ids it owns (``"irc"`` owns ``irc:#lex``).
        depends_on: Every transport resolves channels through the
            registry.
    """

    transport: ClassVar[str]
    depends_on: frozenset[str] = frozenset({"registry"})

    def owns(self, channel: Channel) -> bool:
        """The broadcast filter: True if this transport delivers to ``channel``.

        Override to narrow it (WebPlugin also requires that it is
        enabled); never to widen it past ``transport``.
        """
        return channel.matches_transport(self.transport)

    @hookimpl
    async def send(self, channel: Channel, event: Outbound) -> None:
        """Filter, then dispatch the event to the matching send_* method."""
        # The one broadcast filter. Nothing below runs for a channel
        # another transport owns.
        if not self.owns(channel):
            return
        # Dispatch by event type; an unknown type is a programming error
        # in the sender, so log it rather than raise into the agent loop.
        match event:
            case OutboundMessage():
                await self.send_message(channel, event)
            case OutboundThinking():
                await self.send_thinking(channel, event)
            case OutboundToolStatus():
                await self.send_tool_status(channel, event)
            case OutboundProgress():
                await self.send_progress(channel, event)
            case _:
                _transport_logger.warning(
                    "%s: unknown outbound event %r",
                    self.transport, type(event).__name__,
                )

    async def send_message(self, channel: Channel, event: OutboundMessage) -> None:
        """Deliver the agent's reply. Default: no-op."""

    async def send_thinking(self, channel: Channel, event: OutboundThinking) -> None:
        """Show reasoning content. Default: no-op."""

    async def send_tool_status(self, channel: Channel, event: OutboundToolStatus) -> None:
        """Show a tool call's lifecycle. Default: no-op."""

    async def send_progress(self, channel: Channel, event: OutboundProgress) -> None:
        """Show intermediate assistant text. Default: no-op."""


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

        Grammar class: observe (broadcast; return value ignored).

        Use for storing pm, reading config values, and resolving references
        to other plugins. Do not create runtime resources here — use on_start.

        Args:
            pm: The plugin manager.
            config: The full parsed YAML config dict.
        """

    @hookspec
    async def on_start(self, config: dict) -> None:
        """Called once when the daemon starts, after config is loaded.

        Grammar class: observe (broadcast; return value ignored).

        Args:
            config: The full parsed YAML config dict.
        """

    @hookspec
    async def on_stop(self) -> None:
        """Called once when the daemon receives SIGINT or SIGTERM.

        Grammar class: observe (broadcast; return value ignored).
        """

    @hookspec
    async def on_message(
        self, channel: Channel, sender: str, text: str, trace_id: str
    ) -> None:
        """Called when an inbound message arrives on a channel.

        Grammar class: observe (broadcast; return value ignored).

        Args:
            channel: The Channel object for this conversation scope.
            sender: The user or entity that sent the message.
            text: The message content.
            trace_id: Id minted by the transport for this message
                (``corvidae.trace.mint_trace_id()``). Required,
                with no default: pluggy would silently drop a defaulted
                param. The Agent uses it verbatim for the gate, the
                admitted/rejected hooks and persistence, and raises
                ValueError if it is empty. admitted/rejected fire before
                this hook returns, so a transport can treat "returned
                without admission" as the message being dropped.
        """

    @hookspec
    async def send(self, channel: Channel, event: "Outbound") -> None:
        """Deliver an outbound event to the transport that owns ``channel``.

        Grammar class: bare imperative (broadcast; the caller decides what to do with the result list).

        Every transport receives every event. ``TransportPlugin.send``
        drops events for channels the transport does not own (its
        ``owns()`` check) and dispatches the rest by type to the
        transport's ``send_message``, ``send_thinking``,
        ``send_tool_status`` or ``send_progress`` method, which default to
        no-ops. Transports subclass ``TransportPlugin`` and never implement
        this hook directly, so none of them can forget the filter.

        Args:
            channel: The Channel the event is for.
            event: An ``OutboundMessage``, ``OutboundThinking``,
                ``OutboundToolStatus`` or ``OutboundProgress``
                (corvidae/outbound.py).
        """

    @hookspec
    def register_tools(self, tool_registry: list) -> None:
        """Called during startup so plugins can add tools to the agent loop.

        Grammar class: bare imperative (broadcast; the caller decides what to do with the result list).

        Args:
            tool_registry: A mutable list. Plugins append callable tool
                functions to this list; the agent loop will make them
                available to the LLM.
        """

    @hookspec
    async def on_agent_response(
        self,
        channel: Channel,
        request_text: str,
        response_text: str,
        trace_id: str | None,
        meta: dict,
        logprobs: dict | None,
    ) -> None:
        """Called after the agent loop produces a response to a message.

        Grammar class: observe (broadcast; return value ignored).

        All parameters are required (no defaults): a default on a spec
        param exempts it from the arg-binding guard, so an impl mirroring
        the defaulted signature would silently receive the default instead
        of the caller's value (the pluggy silent-drop trap). Impls that
        don't care about a param simply omit it — pluggy passes declared
        args only.

        Args:
            channel: The Channel where the conversation occurred.
            request_text: The original user message that triggered the loop.
                For a tool-using exchange this is the last tool-result
                turn's text, never the exchange's true originating text.
            response_text: The final text produced by the agent loop.
            trace_id: The trace id this response belongs to,
                or None.
            meta: The trace metadata dict; producers of notifications
                may stamp keys into it via ``on_notify`` meta, and the dict
                travels with the trace id through task cycles.
            logprobs: The response's logprobs envelope, or None.

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

        Grammar class: observe (broadcast; return value ignored).

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
        self, channel: Channel, sender: str, text: str, trace_id: str | None
    ) -> bool | None:
        """Gate hook: decide whether to process an incoming message.

        Grammar class: veto (broadcast; resolved with resolve_reject_wins).

        Broadcast hook. All implementations run. Any False vetoes (reject-wins).
        Return False to reject, True to accept, None for no opinion.

        Args:
            channel: The Channel object for this conversation scope.
            sender: The user or entity that sent the message.
            text: The message content.
            trace_id: The id minted by the transport for this message
                (passed to ``on_message``). Present for every USER message.

        Note:
            Extension point for message filtering (rate limiting, blocklists,
            channel muting). No built-in plugin implements this hook. Return
            False to reject, True to force-accept, None for no opinion.
            Resolved with resolve_reject_wins.
        """

    @hookspec
    async def should_send_response(
        self,
        channel: Channel,
        text: str,
        emission: str,
        trace_id: str | None,
        meta: dict,
    ) -> bool | None:
        """Gate hook: decide whether an outbound emission reaches the channel.

        Grammar class: veto (broadcast; resolved with resolve_reject_wins).

        The outbound mirror of ``should_process_message``. Any False vetoes;
        None is no opinion. An exception from any implementation fails open
        (the Agent logs it and sends), so an implementation that can fail
        should catch and log its own errors rather than discard its
        siblings' verdicts.

        Four firing sites, two modes:

        - ``emission="final"`` fires between generation and persistence,
          only when the result's tool calls will not dispatch (a text
          result, or the max-turns branch). It controls persistence: a veto
          stores the assistant message as ``MessageType.UNSENT``, skips the
          transport send, and appends a not-sent marker to the window. The
          window keeps the text, including across restarts;
          ``on_agent_response`` still fires.
        - ``emission`` in ``"progress"``, ``"thinking"``, ``"error"`` fires
          before that send. A veto suppresses that emission only: the
          message row is unchanged and tool dispatch proceeds. Progress and
          thinking vetoes append a not-sent marker; an error veto suppresses
          an unpersisted send and nothing else.

        All parameters are required (no defaults), so the arg-binding guard
        catches an impl that re-declares one with a default.

        Args:
            channel: The Channel the emission is for.
            text: Exactly the text that would be sent (for "final", the
                display text, or the max-turns fallback on that branch).
            emission: One of "final", "progress", "thinking", "error".
                Treat unknown values as no opinion.
            trace_id: The current turn's trace id, or None.
            meta: The trace metadata dict the item carries. Producers stamp
                plugin-side keys into it (e.g. ``meta["origin"]``); core
                never does.
        """

    @hookspec
    async def on_message_admitted(
        self, channel: Channel, trace_id: str, sender: str, text: str
    ) -> None:
        """Broadcast after should_process_message admits a USER message.

        Grammar class: observe (broadcast; return value ignored).

        Fired exactly once per admitted message, after reject-wins
        resolution. Side effects only (e.g. OutcomeLogPlugin's
        record_exchange).

        Args:
            channel: The Channel the message arrived on.
            trace_id: The id minted by the transport for this message.
            sender: The user or entity that sent the message.
            text: The message content.
        """

    @hookspec
    async def on_message_rejected(
        self, channel: Channel, trace_id: str, sender: str, text: str
    ) -> None:
        """Broadcast after should_process_message vetoes a USER message.

        Grammar class: observe (broadcast; return value ignored).

        Fired exactly once per rejected message.

        Args:
            channel: The Channel the message arrived on.
            trace_id: The id minted by the transport for this message.
            sender: The user or entity that sent the message.
            text: The message content.
        """

    @hookspec
    async def on_message_persisted(
        self,
        channel: Channel,
        trace_id: str,
        rowid: int | None,
        text: str,
        meta: dict,
    ) -> None:
        """Broadcast after the trace-originating message row is persisted.

        Grammar class: observe (broadcast; return value ignored).

        Fired only when the current queue item originates its trace
        (USER items, and notification items whose id was minted at
        dequeue) — never for mid-exchange tool-result rows, injected
        CONTEXT, or assistant rows (per-row firing under one id would
        overwrite the rowid with each successive row). Fires once per
        originating item regardless of persistence outcome, so ``rowid``
        may be None.

        Args:
            channel: The Channel the message was persisted on.
            trace_id: The trace id this row originates.
            rowid: The message_log rowid of the persisted row, or None.
            text: The originating row's content.
            meta: The trace metadata dict carried by the item.
        """

    @hookspec(firstresult=True)
    async def resolve_llm_error(
        self, channel: Channel, error: Exception
    ) -> str | None:
        """Resolve the text sent to the channel when run_agent_turn raises.

        Grammar class: resolve (firstresult=True). Return a string to replace
        the default error message, or None to defer to the next handler. The
        first non-None return value wins; subsequent handlers are not called.

        Who answers: LLMPlugin, which names the failure and the next step for
        LLM client errors, registered ``trylast`` so that a default-priority
        plugin that owns the error answers first.
        """

    @hookspec(firstresult=True)
    async def resolve_compaction(
        self, channel: "Channel", conversation: "ContextWindow", max_tokens: int
    ) -> "bool | None":
        """Resolve who compacts the conversation near the context limit.

        Grammar class: resolve (firstresult=True). Here "resolve" means
        "resolve who compacts": the core ignores the return value, but the
        chain stops at the first non-None one. Handlers are called in priority
        order. The first handler that returns a non-None value wins and the
        chain stops. Return True to signal that compaction was performed,
        or None to defer to the next handler.

        Ordering:
        - tryfirst handlers run first (side-effect handlers that return
          None and do not stop the chain).
        - Default-priority handlers run next (third-party replacements).
        - trylast handlers run last (CompactionPlugin as fallback).

        Who answers: CompactionPlugin, unless a default-priority replacement
        answers first (docs/plugin-guide.md, FancyCompactionPlugin).

        The kernel enforces the token budget after this hook whatever an
        implementation returns or raises: a window still over max_tokens
        loses its oldest rows from memory (ContextWindow.fit_to_budget).

        Args:
            channel: The Channel being compacted.
            conversation: The ContextWindow to compact.
            max_tokens: The channel's max_context_tokens limit.
        """

    @hookspec(firstresult=True)
    async def transform_tool_result(
        self, tool_name: str, result: str, channel: "Channel | None"
    ) -> str | None:
        """Transform a tool result before it enters the conversation.

        Grammar class: transform (firstresult=True wrapper chain with a
        trylast seed on _SeedHooksPlugin). Use ``@hookimpl(wrapper=True)``
        to wrap the chain result. The seed plugin returns the input unchanged;
        wrappers above it compose transforms in LIFO order.

        Example::

            @hookimpl(wrapper=True)
            def transform_tool_result(self, **kwargs):
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

        Grammar class: observe (broadcast; return value ignored).

        Broadcast hook (side effects only). Fires from the LLMClient
        chokepoint via an observer injected by LLMPlugin, so every LLM
        call in the system is covered — turn-loop, compaction, subagent,
        and future background calls alike.

        The request payload is summarized (counts), not shipped wholesale —
        full messages in a broadcast hook would copy the entire prompt per
        call. A debugging consumer needing full payloads is a new hook, not
        a widening of this one.

        No in-tree plugin implements this hook. It is kept deliberately:
        observability hooks come in request/response pairs (this one pairs
        with on_llm_response), and it exists for consumers outside the tree.

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

        Grammar class: observe (broadcast; return value ignored).

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

        Grammar class: observe (broadcast; return value ignored).

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

    @hookspec(firstresult=True)
    async def resolve_conversation(self, channel: "Channel") -> "list[dict] | None":
        """Resolve the persisted conversation history for a channel.

        Grammar class: resolve (firstresult=True). Return a list of tagged
        message dicts if this plugin has stored history, or None to defer to
        the next handler. The first non-None return value wins.

        Who answers: PersistencePlugin. Callers are Agent (lazy channel init)
        and AcpPlugin (``session/load``).

        Args:
            channel: The Channel to load history for.
        """

    @hookspec(firstresult=True)
    async def resolve_channel(self, channel_id: str) -> "Channel | None":
        """Return the live Channel for a persisted channel id, creating it
        with transport-specific config if needed.

        Grammar class: resolve (firstresult=True).

        Sequential hook (firstresult=True). Transports implement this for
        ids they own and return None for others. Callers holding only a
        persisted ``transport:scope`` id (e.g. SchedulerPlugin firing a row
        after a restart) use it so a transport can apply config a bare
        ``registry.get_or_create`` would bypass, such as Signal's E.164
        alias ``ChannelConfig``. ``channel_id`` has no default — see the
        arg-binding guard (tests/test_hook_arg_binding.py).

        Who answers: the transport whose name is the id's ``transport:``
        prefix (DiscordPlugin, SignalPlugin). The hook is keyed on that
        prefix and every implementation returns None for ids it does not
        own, so registration order cannot change the answer.

        A transport may also return None for an id it does own when it
        cannot apply that channel's config yet (PRD §3.2.7), e.g. Signal
        before its first E.164 alias resolution. Callers must then retry
        rather than fall back to ``registry.get_or_create``, which would
        cache a default-configured channel; SchedulerPlugin does, for any
        id whose prefix names a plugin (by its ``transport`` attribute)
        that implements this hook.

        Args:
            channel_id: The ``transport:scope`` persistence key.

        Returns:
            The Channel, or None when this transport does not own the id
            or cannot build it yet.
        """

    @hookspec(firstresult=True)
    async def resolve_register_prompt(self, channel: "Channel") -> "str | None":
        """Return the default register text for a channel (#316).

        Grammar class: resolve (firstresult=True).

        The register is appended after the agent-wide identity in the
        channel's system prompt. Agent asks for it once per conversation,
        only when no ``register_prompt`` is configured for the channel at
        any layer (exact entry, transport-wide entry, ``agent:``).

        Who answers: the transport that owns the channel (SignalPlugin,
        DiscordPlugin, IRCPlugin ship defaults). Every implementation
        returns None for channels it does not own. No answer means a blank
        register. ``channel`` has no default — see the arg-binding guard
        (tests/test_hook_arg_binding.py).

        Args:
            channel: The Channel whose conversation is starting.

        Returns:
            The register text, or None for no default.
        """

    @hookspec
    async def persist_message(
        self, channel: "Channel", message: dict, message_type: "MessageType"
    ) -> "int | None":
        """Persist a message appended to the conversation.

        Grammar class: bare imperative (broadcast; the caller decides what to
        do with the result list). Called after every conv.append(). Every
        implementation runs. PersistencePlugin returns the inserted
        ``message_log`` rowid; every other implementation (JsonlLogPlugin)
        returns None. Callers resolve the result list with
        ``resolve_single_result``, which logs an error and uses the first
        value if more than one implementation answers, and attach the rowid
        to the in-window message copy as ``_db_id`` (bootstrap-mapping §4.8).

        Args:
            channel: The Channel where the event occurred.
            message: The untagged message dict (no _message_type key).
            message_type: The MessageType of the message.

        Returns:
            The message_log rowid from the persistence implementation,
            None from every other implementation.
        """

    @hookspec
    async def before_compaction(
        self,
        channel: "Channel",
        compacted_ids: list[int],
        cut: int,
    ) -> None:
        """Fired just before compaction drops the head of the window.

        Grammar class: observe (broadcast; return value ignored).

        Broadcast, awaited (it runs on the response path). Fired by
        CompactionPlugin after the summary succeeded and before
        replace_with_summary, so the window still holds every message about
        to be dropped and the server still holds their rendered prefix. A
        plugin that wants a fork over that prefix runs it here, with
        ``Agent.fork(channel, instruction, cut=cut)`` (GitHub issue #51).
        Failures are logged and never block compaction.

        Args:
            channel: The Channel being compacted.
            compacted_ids: The ``message_log`` rowids of the messages about
                to be dropped, in window order (same list on_compaction
                receives). Empty when none were persisted.
            cut: Window index of the first retained message; the dropped
                messages are ``channel.conversation.messages[:cut]``.
                Normally no tool cycle straddles it; the exception is a
                long-span cycle whose result is near the tail, where
                compaction keeps its planned cut (CompactionPlugin
                ._clean_cut) and the fork prefix renders that call with a
                placeholder result.
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

        Grammar class: observe (broadcast; return value ignored).

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
    async def before_agent_turn(
        self,
        channel: "Channel",
        trace_id: str | None,
        meta: dict,
    ) -> None:
        """Called before each LLM invocation, after compaction.

        Grammar class: observe (broadcast; return value ignored).

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
            trace_id: The current turn's trace id, or None.
            meta: The trace metadata dict; producers of notifications
                may stamp keys into it via ``on_notify`` meta, and the dict
                travels with the trace id through task cycles.
        """

    @hookspec
    async def after_persist_assistant(self, channel: "Channel", message: dict) -> None:
        """Called after the assistant message has been persisted to the conversation log.

        Grammar class: observe (broadcast; return value ignored).

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

        Grammar class: transform (firstresult=True wrapper chain with a trylast seed on _SeedHooksPlugin).

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
    async def on_config_reload(self, config: dict) -> None:
        """Called when agent.yaml is reloaded from disk.

        Grammar class: observe (broadcast; return value ignored).

        Plugins should re-read their configuration from the new config dict.
        The config dict has already been merged with CLI overrides and validated.

        This hook fires on the event loop thread. Plugins must not block.
        Errors are caught per-plugin and logged; they do not prevent other
        plugins from receiving the update.

        Args:
            config: The full re-parsed and merged config dict.
        """
