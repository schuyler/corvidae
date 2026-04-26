import pytest
import apluggy as pluggy
from pluggy import HookimplMarker, HookspecMarker

from corvidae.hooks import AgentSpec, hookimpl, hookspec
from corvidae.hooks import create_plugin_manager
from corvidae.hooks import get_dependency, validate_dependencies


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
# call_firstresult_hook (red phase)
# ---------------------------------------------------------------------------


async def test_call_firstresult_hook_returns_none_when_no_impls():
    """Returns None when no implementations are registered."""
    from corvidae.hooks import call_firstresult_hook

    pm = create_plugin_manager()
    result = await call_firstresult_hook(pm, "should_process_message",
                                         channel=None, sender="user", text="hi")
    assert result is None


async def test_call_firstresult_hook_returns_first_non_none():
    """Returns the first non-None result from registered implementations,
    and does NOT call further implementations after the first non-None result."""
    from corvidae.hooks import call_firstresult_hook

    call_order = []

    class PluginA:
        @hookimpl
        async def should_process_message(self, channel, sender, text):
            call_order.append("A")
            return None  # no opinion

    class PluginB:
        @hookimpl
        async def should_process_message(self, channel, sender, text):
            call_order.append("B")
            return True  # accept — stops iteration

    class PluginC:
        @hookimpl
        async def should_process_message(self, channel, sender, text):
            call_order.append("C")
            return False  # would reject — must not be reached

    pm = create_plugin_manager()
    pm.register(PluginA(), name="a")
    pm.register(PluginB(), name="b")
    pm.register(PluginC(), name="c")

    result = await call_firstresult_hook(pm, "should_process_message",
                                          channel=None, sender="user", text="hi")
    # PluginB returns True — C must not run after a non-None result
    assert result is True
    assert call_order == ["A", "B"], (
        f"Expected only A and B to be called (short-circuit after B), "
        f"got: {call_order}"
    )


async def test_call_firstresult_hook_skips_none_results():
    """Skips None-returning implementations and continues to the next."""
    from corvidae.hooks import call_firstresult_hook

    call_order = []

    class PluginFirst:
        @hookimpl
        async def should_process_message(self, channel, sender, text):
            call_order.append("first")
            return None

    class PluginSecond:
        @hookimpl
        async def should_process_message(self, channel, sender, text):
            call_order.append("second")
            return False

    pm = create_plugin_manager()
    pm.register(PluginFirst(), name="first")
    pm.register(PluginSecond(), name="second")

    result = await call_firstresult_hook(pm, "should_process_message",
                                          channel=None, sender="user", text="hi")
    assert result is False
    assert "first" in call_order
    assert "second" in call_order


async def test_call_firstresult_hook_respects_tryfirst():
    """tryfirst implementations run before regular implementations."""
    from corvidae.hooks import call_firstresult_hook

    call_order = []

    class RegularPlugin:
        @hookimpl
        async def should_process_message(self, channel, sender, text):
            call_order.append("regular")
            return "regular_result"

    class FirstPlugin:
        @hookimpl(tryfirst=True)
        async def should_process_message(self, channel, sender, text):
            call_order.append("tryfirst")
            return "first_result"

    pm = create_plugin_manager()
    pm.register(RegularPlugin(), name="regular")
    pm.register(FirstPlugin(), name="first_plugin")

    result = await call_firstresult_hook(pm, "should_process_message",
                                          channel=None, sender="user", text="hi")
    # tryfirst must run first and short-circuit
    assert result == "first_result"
    assert call_order[0] == "tryfirst"
    assert "regular" not in call_order


async def test_call_firstresult_hook_respects_trylast():
    """trylast implementations run after regular implementations."""
    from corvidae.hooks import call_firstresult_hook

    call_order = []

    class LastPlugin:
        @hookimpl(trylast=True)
        async def should_process_message(self, channel, sender, text):
            call_order.append("trylast")
            return "last_result"

    class RegularPlugin:
        @hookimpl
        async def should_process_message(self, channel, sender, text):
            call_order.append("regular")
            return "regular_result"

    pm = create_plugin_manager()
    pm.register(LastPlugin(), name="last_plugin")
    pm.register(RegularPlugin(), name="regular")

    result = await call_firstresult_hook(pm, "should_process_message",
                                          channel=None, sender="user", text="hi")
    # Regular runs first and short-circuits; trylast is never reached
    assert result == "regular_result"
    assert call_order[0] == "regular"
    assert "trylast" not in call_order


async def test_call_firstresult_hook_skips_wrapper_impls():
    """Wrapper implementations are skipped (not direct result producers)."""
    from corvidae.hooks import call_firstresult_hook

    class WrapperPlugin:
        @hookimpl(wrapper=True)
        def should_process_message(self, channel, sender, text):
            # wrappers are generators; this should never be called as a result producer
            outcome = yield  # pragma: no cover
            return outcome  # pragma: no cover

    class NormalPlugin:
        @hookimpl
        async def should_process_message(self, channel, sender, text):
            return True

    pm = create_plugin_manager()
    pm.register(WrapperPlugin(), name="wrapper")
    pm.register(NormalPlugin(), name="normal")

    # Should return the normal plugin's result, not error on the wrapper
    result = await call_firstresult_hook(pm, "should_process_message",
                                          channel=None, sender="user", text="hi")
    assert result is True


async def test_call_firstresult_hook_filters_kwargs_to_impl_signature():
    """Filters kwargs to match each implementation's parameter list."""
    from corvidae.hooks import call_firstresult_hook

    received_kwargs = {}

    class SparsePlugin:
        @hookimpl
        async def should_process_message(self, sender):
            # Only accepts 'sender' — not channel or text
            received_kwargs["sender"] = sender
            return "ok"

    pm = create_plugin_manager()
    pm.register(SparsePlugin(), name="sparse")

    # Should not raise even though channel and text are not in SparsePlugin's signature
    result = await call_firstresult_hook(pm, "should_process_message",
                                          channel=object(), sender="alice", text="hi")
    assert result == "ok"
    assert received_kwargs == {"sender": "alice"}


async def test_call_firstresult_hook_returns_none_when_all_return_none():
    """Returns None when all implementations return None."""
    from corvidae.hooks import call_firstresult_hook

    class PluginA:
        @hookimpl
        async def should_process_message(self, channel, sender, text):
            return None

    class PluginB:
        @hookimpl
        async def should_process_message(self, channel, sender, text):
            return None

    pm = create_plugin_manager()
    pm.register(PluginA(), name="a")
    pm.register(PluginB(), name="b")

    result = await call_firstresult_hook(pm, "should_process_message",
                                          channel=None, sender="user", text="hi")
    assert result is None


async def test_call_firstresult_hook_unknown_hook_name_returns_none():
    """Returns None when the hook name does not exist on the spec."""
    from corvidae.hooks import call_firstresult_hook

    pm = create_plugin_manager()
    result = await call_firstresult_hook(pm, "nonexistent_hook_name",
                                          channel=None, sender="user", text="hi")
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


async def test_call_firstresult_hook_supports_sync_impls():
    """Sync implementations (non-async) are also supported."""
    from corvidae.hooks import call_firstresult_hook

    class SyncPlugin:
        @hookimpl
        def should_process_message(self, channel, sender, text):
            return "sync_result"

    pm = create_plugin_manager()
    pm.register(SyncPlugin(), name="sync")

    result = await call_firstresult_hook(pm, "should_process_message",
                                          channel=None, sender="user", text="hi")
    assert result == "sync_result"
