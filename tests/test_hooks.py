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


# ---------------------------------------------------------------------------
# Wrapper chain integration tests (Phase 4 red phase)
# ---------------------------------------------------------------------------


class TestWrapperChainIntegration:
    """Integration tests for wrapper chain semantics on transform_display_text
    and process_tool_result hooks.

    These tests encode the NEW behavior where:
    - Both hookspecs are firstresult=True
    - A _SeedHooksPlugin (trylast=True) is auto-registered by create_plugin_manager()
      and returns the input unchanged
    - Wrapper hookimpls wrap the chain result with @hookimpl(wrapper=True)

    All tests FAIL until Green phase is complete.
    """

    # ------------------------------------------------------------------
    # transform_display_text
    # ------------------------------------------------------------------

    async def test_transform_display_text_seed_only_returns_input(self):
        """Seed-only (no wrappers): result equals the input text unchanged."""
        pm = create_plugin_manager()
        # No additional plugins registered — only the seed.

        result = await pm.ahook.transform_display_text(
            channel=None, text="hello world", result_message={}
        )

        # firstresult=True returns a single value, not a list.
        assert result == "hello world"

    async def test_transform_display_text_single_wrapper_transforms_result(self):
        """A single wrapper plugin transforms the seed's result."""
        class UpperWrapper:
            @hookimpl(wrapper=True)
            def transform_display_text(self, **kwargs):
                result = yield
                if result is not None:
                    return result.upper()
                return result

        pm = create_plugin_manager()
        pm.register(UpperWrapper(), name="upper")

        result = await pm.ahook.transform_display_text(
            channel=None, text="hello", result_message={}
        )

        assert result == "HELLO"

    async def test_transform_display_text_two_wrappers_stack(self):
        """Two wrappers compose: both transforms apply in order."""
        class ExclaimWrapper:
            @hookimpl(wrapper=True)
            def transform_display_text(self, **kwargs):
                result = yield
                if result is not None:
                    return result + "!"
                return result

        class UpperWrapper:
            @hookimpl(wrapper=True)
            def transform_display_text(self, **kwargs):
                result = yield
                if result is not None:
                    return result.upper()
                return result

        pm = create_plugin_manager()
        pm.register(ExclaimWrapper(), name="exclaim")
        pm.register(UpperWrapper(), name="upper")

        result = await pm.ahook.transform_display_text(
            channel=None, text="hello", result_message={}
        )

        # Pluggy LIFO: UpperWrapper (registered last) is outermost.
        # Execution: seed returns "hello" → ExclaimWrapper returns "hello!" →
        # UpperWrapper returns "HELLO!"
        assert result == "HELLO!"

    async def test_transform_display_text_non_wrapper_coexists_with_wrapper(self):
        """A non-wrapper hookimpl coexists with a wrapper.

        The non-wrapper (regular hookimpl) returns a non-None value, short-circuiting
        the seed (firstresult). The wrapper above it still transforms that result.
        """
        class NonWrapperPlugin:
            @hookimpl
            async def transform_display_text(self, channel, text, result_message):
                return "from non-wrapper"

        class ExclaimWrapper:
            @hookimpl(wrapper=True)
            def transform_display_text(self, **kwargs):
                result = yield
                if result is not None:
                    return result + "!"
                return result

        pm = create_plugin_manager()
        pm.register(NonWrapperPlugin(), name="nonwrapper")
        pm.register(ExclaimWrapper(), name="exclaim")

        result = await pm.ahook.transform_display_text(
            channel=None, text="original", result_message={}
        )

        # The non-wrapper's result is the inner value; the wrapper transforms it.
        assert result == "from non-wrapper!"

    # ------------------------------------------------------------------
    # process_tool_result
    # ------------------------------------------------------------------

    async def test_process_tool_result_seed_only_returns_input(self):
        """Seed-only (no wrappers): result equals the input result unchanged."""
        pm = create_plugin_manager()
        # No additional plugins registered — only the seed.

        result = await pm.ahook.process_tool_result(
            tool_name="my_tool", result="tool output", channel=None
        )

        # firstresult=True returns a single value, not a list.
        assert result == "tool output"

    async def test_process_tool_result_wrapper_transforms_result(self):
        """A wrapper on process_tool_result transforms the seed's result."""
        class TagWrapper:
            @hookimpl(wrapper=True)
            def process_tool_result(self, **kwargs):
                result = yield
                if result is not None:
                    return f"[processed] {result}"
                return result

        pm = create_plugin_manager()
        pm.register(TagWrapper(), name="tagger")

        result = await pm.ahook.process_tool_result(
            tool_name="my_tool", result="raw output", channel=None
        )

        assert result == "[processed] raw output"


