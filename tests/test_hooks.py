import pytest
import apluggy as pluggy
from pluggy import HookimplMarker, HookspecMarker

from corvidae.hooks import AgentSpec, hookimpl, hookspec
from corvidae.hooks import create_plugin_manager
from corvidae.hooks import get_dependency, validate_dependencies
from corvidae.hooks import resolve_hook_results, HookStrategy


def test_hookspec_marker_exists():
    assert isinstance(hookspec, HookspecMarker)


def test_hookimpl_marker_exists():
    assert isinstance(hookimpl, HookimplMarker)


async def test_register_plugin_with_hookimpl():
    called_with = {}

    class MyPlugin:
        @hookimpl
        async def on_start(self, config):
            called_with["config"] = config

    pm = create_plugin_manager()
    pm.register(MyPlugin())
    await pm.ahook.on_start(config={"key": "value"})

    assert called_with == {"config": {"key": "value"}}


def test_sync_hook_register_tools():
    tool_registry = []

    class MyPlugin:
        @hookimpl
        def register_tools(self, tool_registry):
            tool_registry.append("my_tool")

    pm = create_plugin_manager()
    pm.register(MyPlugin())
    pm.hook.register_tools(tool_registry=tool_registry)

    assert tool_registry == ["my_tool"]


async def test_multiple_plugins_receive_hook():
    from corvidae.channel import Channel

    received = []

    class PluginA:
        @hookimpl
        async def on_message(self, channel, sender, text):
            received.append("A")

    class PluginB:
        @hookimpl
        async def on_message(self, channel, sender, text):
            received.append("B")

    pm = create_plugin_manager()
    pm.register(PluginA())
    pm.register(PluginB())
    ch = Channel(transport="test", scope="scope")
    await pm.ahook.on_message(channel=ch, sender="user", text="hi")

    assert sorted(received) == ["A", "B"]


# ---------------------------------------------------------------------------
# get_dependency
# ---------------------------------------------------------------------------

class _FakePlugin:
    """Minimal plugin class used as a type anchor in get_dependency tests."""


def test_get_dependency_returns_plugin_when_found_and_type_matches():
    pm = create_plugin_manager()
    plugin = _FakePlugin()
    pm.register(plugin, name="fake")

    result = get_dependency(pm, "fake", _FakePlugin)

    assert result is plugin


def test_get_dependency_raises_runtime_error_when_not_registered():
    pm = create_plugin_manager()

    with pytest.raises(RuntimeError):
        get_dependency(pm, "nonexistent", _FakePlugin)


def test_get_dependency_raises_type_error_when_wrong_type():
    class _OtherPlugin:
        pass

    pm = create_plugin_manager()
    pm.register(_OtherPlugin(), name="other")

    with pytest.raises(TypeError):
        get_dependency(pm, "other", _FakePlugin)


# ---------------------------------------------------------------------------
# validate_dependencies
# ---------------------------------------------------------------------------

def test_validate_dependencies_passes_when_all_declared_deps_are_registered():
    class _Dep:
        pass

    class _Consumer:
        depends_on = {"dep"}

    pm = create_plugin_manager()
    pm.register(_Dep(), name="dep")
    pm.register(_Consumer(), name="consumer")

    # Should not raise.
    validate_dependencies(pm)


def test_validate_dependencies_raises_runtime_error_when_dependency_missing():
    class _Consumer:
        depends_on = {"missing_dep"}

    pm = create_plugin_manager()
    pm.register(_Consumer(), name="consumer")

    with pytest.raises(RuntimeError):
        validate_dependencies(pm)


def test_validate_dependencies_passes_when_depends_on_is_empty_set():
    class _NoDeclaredDeps:
        depends_on: set[str] = set()

    pm = create_plugin_manager()
    pm.register(_NoDeclaredDeps(), name="nodeclared")

    # Should not raise — empty set means no dependencies.
    validate_dependencies(pm)


def test_validate_dependencies_skips_plugins_without_depends_on():
    class _NoDeps:
        pass

    pm = create_plugin_manager()
    pm.register(_NoDeps(), name="nodeps")

    # Should not raise.
    validate_dependencies(pm)


# ---------------------------------------------------------------------------
# New hookspecs on AgentSpec (red phase)
# ---------------------------------------------------------------------------


def test_agentspec_has_should_process_message():
    """should_process_message hookspec exists on AgentSpec."""
    assert hasattr(AgentSpec, "should_process_message"), (
        "AgentSpec must have a should_process_message hookspec"
    )


def test_agentspec_has_on_llm_error():
    """on_llm_error hookspec exists on AgentSpec."""
    assert hasattr(AgentSpec, "on_llm_error"), (
        "AgentSpec must have an on_llm_error hookspec"
    )


def test_agentspec_has_compact_conversation():
    """compact_conversation hookspec exists on AgentSpec."""
    assert hasattr(AgentSpec, "compact_conversation"), (
        "AgentSpec must have a compact_conversation hookspec"
    )


def test_agentspec_has_process_tool_result():
    """process_tool_result hookspec exists on AgentSpec."""
    assert hasattr(AgentSpec, "process_tool_result"), (
        "AgentSpec must have a process_tool_result hookspec"
    )


def test_agentspec_has_on_idle():
    """on_idle hookspec exists on AgentSpec."""
    assert hasattr(AgentSpec, "on_idle"), (
        "AgentSpec must have an on_idle hookspec"
    )


# ---------------------------------------------------------------------------
# resolve_hook_results (red phase)
# ---------------------------------------------------------------------------


def test_resolve_hook_results_reject_wins_false_vetoes():
    """REJECT_WINS: any False in results returns False."""
    result = resolve_hook_results(
        [None, True, False], "should_process_message", HookStrategy.REJECT_WINS
    )
    assert result is False


def test_resolve_hook_results_reject_wins_true_when_no_false():
    """REJECT_WINS: True when no False is present."""
    result = resolve_hook_results(
        [None, True, None], "should_process_message", HookStrategy.REJECT_WINS
    )
    assert result is True


def test_resolve_hook_results_reject_wins_none_when_all_none():
    """REJECT_WINS: None when all results are None."""
    result = resolve_hook_results(
        [None, None], "should_process_message", HookStrategy.REJECT_WINS
    )
    assert result is None


def test_resolve_hook_results_reject_wins_empty_list():
    """REJECT_WINS: empty list returns None."""
    result = resolve_hook_results(
        [], "should_process_message", HookStrategy.REJECT_WINS
    )
    assert result is None


def test_resolve_hook_results_accept_wins_true_when_any_true():
    """ACCEPT_WINS: True when any result is True."""
    result = resolve_hook_results(
        [None, True, None], "ensure_conversation", HookStrategy.ACCEPT_WINS
    )
    assert result is True


def test_resolve_hook_results_accept_wins_none_when_no_true():
    """ACCEPT_WINS: None when no True is present."""
    result = resolve_hook_results(
        [None, None], "ensure_conversation", HookStrategy.ACCEPT_WINS
    )
    assert result is None


def test_resolve_hook_results_accept_wins_ignores_false():
    """ACCEPT_WINS: False values are ignored; returns None if no True."""
    result = resolve_hook_results(
        [None, False, None], "ensure_conversation", HookStrategy.ACCEPT_WINS
    )
    assert result is None


def test_resolve_hook_results_value_first_empty_list():
    """VALUE_FIRST: empty list returns None."""
    result = resolve_hook_results(
        [], "transform_display_text", HookStrategy.VALUE_FIRST
    )
    assert result is None


def test_resolve_hook_results_value_first_all_none():
    """VALUE_FIRST: list of all None values returns None."""
    result = resolve_hook_results(
        [None, None], "transform_display_text", HookStrategy.VALUE_FIRST
    )
    assert result is None


def test_resolve_hook_results_value_first_single_non_none():
    """VALUE_FIRST: single non-None result is returned directly."""
    result = resolve_hook_results(
        [None, "hello", None], "transform_display_text", HookStrategy.VALUE_FIRST
    )
    assert result == "hello"


async def test_resolve_hook_results_value_first_warns_and_picks_alphabetically_first(
    caplog,
):
    """VALUE_FIRST: warns and picks alphabetically-first plugin name on conflict."""
    import logging

    class AlphaPlugin:
        @hookimpl
        async def transform_display_text(self, channel, text, result_message):
            return "alpha result"

    class ZebraPlugin:
        @hookimpl
        async def transform_display_text(self, channel, text, result_message):
            return "zebra result"

    pm = create_plugin_manager()
    pm.register(AlphaPlugin(), name="alpha")
    pm.register(ZebraPlugin(), name="zebra")

    results = await pm.ahook.transform_display_text(
        channel=None, text="hi", result_message={}
    )

    with caplog.at_level(logging.WARNING, logger="corvidae.hooks"):
        result = resolve_hook_results(
            results, "transform_display_text", HookStrategy.VALUE_FIRST, pm=pm
        )

    assert result == "alpha result"
    warning_records = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "transform_display_text" in r.getMessage()
    ]
    assert warning_records, "Expected a WARNING log naming the conflicting plugins"


def test_resolve_hook_results_value_first_no_pm_falls_back_to_first(caplog):
    """VALUE_FIRST: pm=None with multiple non-None results returns first, logs warning."""
    import logging

    with caplog.at_level(logging.WARNING, logger="corvidae.hooks"):
        result = resolve_hook_results(
            ["first_result", "second_result"],
            "transform_display_text",
            HookStrategy.VALUE_FIRST,
            pm=None,
        )

    assert result == "first_result"
    warning_records = [
        r for r in caplog.records
        if r.levelno == logging.WARNING
    ]
    assert warning_records, "Expected a WARNING log when pm is None and multiple results exist"


async def test_resolve_hook_results_value_first_result_plugin_correlation_correct(
    caplog,
):
    """VALUE_FIRST: result-to-plugin correlation is correct despite apluggy's reversed
    execution order. Register two plugins returning distinct values; verify that
    resolve_hook_results attributes each result to the correct plugin name.

    apluggy (like pluggy) executes hooks in reversed registration order, so the
    results list is in reversed order relative to get_hookimpls(). This test
    guards against a zip(impls, results) ordering mismatch.
    """
    import logging

    # "alpha" registered first, "zebra" registered second.
    # apluggy executes zebra first, then alpha (reversed order).
    # results list: [zebra_result, alpha_result]
    # get_hookimpls() returns [alpha_impl, zebra_impl] (registration order).
    # reversed(get_hookimpls()) == [zebra_impl, alpha_impl], matching results order.
    # Alphabetically: "alpha" < "zebra", so alpha's result wins.

    class AlphaPlugin:
        @hookimpl
        async def transform_display_text(self, channel, text, result_message):
            return "from alpha"

    class ZebraPlugin:
        @hookimpl
        async def transform_display_text(self, channel, text, result_message):
            return "from zebra"

    pm = create_plugin_manager()
    pm.register(AlphaPlugin(), name="alpha")
    pm.register(ZebraPlugin(), name="zebra")

    results = await pm.ahook.transform_display_text(
        channel=None, text="hi", result_message={}
    )

    with caplog.at_level(logging.WARNING, logger="corvidae.hooks"):
        result = resolve_hook_results(
            results, "transform_display_text", HookStrategy.VALUE_FIRST, pm=pm
        )

    # Alphabetically "alpha" < "zebra", so alpha's result must be selected.
    assert result == "from alpha", (
        f"Expected 'from alpha' (alphabetically first plugin), got {result!r}. "
        "This may indicate a zip(impls, results) ordering mismatch."
    )


# ---------------------------------------------------------------------------
# validate_dependencies — cycle detection (red phase)
# ---------------------------------------------------------------------------


def test_validate_dependencies_raises_on_direct_cycle():
    """A depends on B and B depends on A — a direct cycle must raise RuntimeError
    with the cycle path in the message."""

    class _PluginA:
        depends_on = {"b"}

    class _PluginB:
        depends_on = {"a"}

    pm = create_plugin_manager()
    pm.register(_PluginA(), name="a")
    pm.register(_PluginB(), name="b")

    with pytest.raises(RuntimeError, match=r"cycle"):
        validate_dependencies(pm)


def test_validate_dependencies_raises_on_indirect_cycle():
    """A → B → C → A — an indirect three-node cycle must raise RuntimeError."""

    class _PluginA:
        depends_on = {"b"}

    class _PluginB:
        depends_on = {"c"}

    class _PluginC:
        depends_on = {"a"}

    pm = create_plugin_manager()
    pm.register(_PluginA(), name="a")
    pm.register(_PluginB(), name="b")
    pm.register(_PluginC(), name="c")

    with pytest.raises(RuntimeError, match=r"cycle"):
        validate_dependencies(pm)


def test_validate_dependencies_raises_on_self_dependency():
    """A plugin that lists itself in depends_on must raise RuntimeError."""

    class _SelfDepPlugin:
        depends_on = {"self_dep"}

    pm = create_plugin_manager()
    pm.register(_SelfDepPlugin(), name="self_dep")

    with pytest.raises(RuntimeError, match=r"cycle"):
        validate_dependencies(pm)


def test_validate_dependencies_does_not_raise_on_valid_dag():
    """A → B → C and A → C (diamond-ish DAG) must NOT raise — no cycle present."""

    class _PluginA:
        depends_on = {"b", "c"}

    class _PluginB:
        depends_on = {"c"}

    class _PluginC:
        pass

    pm = create_plugin_manager()
    pm.register(_PluginA(), name="a")
    pm.register(_PluginB(), name="b")
    pm.register(_PluginC(), name="c")

    # Should not raise.
    validate_dependencies(pm)


