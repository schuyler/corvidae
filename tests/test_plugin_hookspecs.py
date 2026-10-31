"""Plugin-declared hookspecs (#229).

A plugin declares hooks it owns with a ``hookspecs`` class attribute; other
plugins implement them, with a hard (``depends_on``) or optional
(``optional_depends_on``) dependency on the owner. ``install_plugin_hookspecs``
validates and installs every declaration at startup, and
``report_orphan_hookimpls`` reports hookimpls that ended up with no spec.

Each test class covers one requirement from the issue; error tests assert
the hook, the plugin and the remedy appear in the message.
"""
from __future__ import annotations

import logging
import os
import tempfile
from unittest.mock import AsyncMock, patch

import apluggy as pluggy
import pytest

from corvidae.hooks import (
    AgentSpec,
    CorvidaePlugin,
    _check_hook_arg_binding,
    create_plugin_manager,
    hookimpl,
    hookspec,
    install_plugin_hookspecs,
    report_orphan_hookimpls,
    validate_dependencies,
)

_PM_LOGGER = "corvidae.plugin_manager"


# ---------------------------------------------------------------------------
# Fixtures: an owner ("web") and its spec, plus helpers
# ---------------------------------------------------------------------------


class WebSpec:
    """Hooks owned by the test plugin registered as ``web``."""

    @hookspec(firstresult=True)
    async def resolve_web_answer(self, value: int) -> int | None:
        """First non-None answer wins."""

    @hookspec
    async def on_web_console_init(self, panels: list) -> None:
        """Implementers append to ``panels``."""


class WebOwner(CorvidaePlugin):
    """Owner plugin: declares WebSpec and calls its hooks."""

    hookspecs = WebSpec


class Answerer(CorvidaePlugin):
    """Implements web's resolve hook; hard dependency on web."""

    depends_on = frozenset({"web"})

    @hookimpl
    async def resolve_web_answer(self, value: int) -> int:
        return value * 2


class PanelAdder(CorvidaePlugin):
    """Implements web's observe hook; optional dependency on web."""

    optional_depends_on = frozenset({"web"})

    def __init__(self) -> None:
        self.calls = 0

    @hookimpl
    async def on_web_console_init(self, panels: list) -> None:
        self.calls += 1
        panels.append("profiler")


def _pm(*plugins: tuple[str, object]) -> pluggy.PluginManager:
    """A fresh corvidae plugin manager with ``(name, plugin)`` pairs registered."""
    pm = create_plugin_manager()
    for name, plugin in plugins:
        pm.register(plugin, name=name)
    return pm


def _owner_with(spec_cls: type) -> CorvidaePlugin:
    """An owner plugin instance whose class declares ``spec_cls``."""
    cls = type("Owner", (CorvidaePlugin,), {"hookspecs": spec_cls})
    return cls()


def _install_error(pm: pluggy.PluginManager) -> str:
    """Run install_plugin_hookspecs, expecting failure; return the message."""
    with pytest.raises(pluggy.PluginValidationError) as excinfo:
        install_plugin_hookspecs(pm)
    return str(excinfo.value)


# ---------------------------------------------------------------------------
# R1: a plugin owns a hook that others implement and the owner calls
# ---------------------------------------------------------------------------


class TestOwnedHookRoundTrip:
    """R1: owner declares, implementer implements, owner's call reaches it."""

    @pytest.mark.parametrize("owner_first", [True, False], ids=["owner-first", "impl-first"])
    async def test_owner_call_reaches_implementer(self, owner_first: bool) -> None:
        pairs = [("web", WebOwner()), ("answerer", Answerer())]
        pm = _pm(*(pairs if owner_first else reversed(pairs)))

        install_plugin_hookspecs(pm)

        assert await pm.ahook.resolve_web_answer(value=21) == 42

    async def test_broadcast_hook_reaches_every_implementer(self) -> None:
        a, b = PanelAdder(), PanelAdder()
        pm = _pm(("web", WebOwner()), ("a", a), ("b", b))
        install_plugin_hookspecs(pm)

        panels: list = []
        await pm.ahook.on_web_console_init(panels=panels)

        assert panels == ["profiler", "profiler"]

    def test_agentspec_does_not_gain_plugin_hooks(self) -> None:
        pm = _pm(("web", WebOwner()))
        install_plugin_hookspecs(pm)

        assert not hasattr(AgentSpec, "resolve_web_answer")
        assert not hasattr(AgentSpec, "on_web_console_init")

    async def test_declaration_inherited_from_base_class_is_installed(self) -> None:
        class ConsoleBase(CorvidaePlugin):
            hookspecs = WebSpec

        class ConcreteWeb(ConsoleBase):
            pass

        pm = _pm(("web", ConcreteWeb()), ("answerer", Answerer()))
        install_plugin_hookspecs(pm)

        assert await pm.ahook.resolve_web_answer(value=1) == 2

    def test_plugin_without_hookspecs_is_ignored(self) -> None:
        """Plugins that declare nothing (the default) install nothing."""
        pm = _pm(("plain", CorvidaePlugin()))
        install_plugin_hookspecs(pm)  # must not raise

    def test_corvidae_plugin_defaults(self) -> None:
        assert CorvidaePlugin.hookspecs is None
        assert CorvidaePlugin.optional_depends_on == frozenset()


# ---------------------------------------------------------------------------
# R2: specs exist before the arg-binding guard and before on_init
# ---------------------------------------------------------------------------


class TestSpecsBeforeGuardAndInit:
    """R2."""

    def test_arg_binding_guard_covers_plugin_hooks(self) -> None:
        class DefaultsRequired(CorvidaePlugin):
            @hookimpl
            async def resolve_web_answer(self, value=None):
                return None

        pm = _pm(("web", WebOwner()), ("bad", DefaultsRequired()))
        install_plugin_hookspecs(pm)

        with pytest.raises(pluggy.PluginValidationError) as excinfo:
            _check_hook_arg_binding(pm)
        message = str(excinfo.value)
        assert "bad" in message
        assert "resolve_web_answer" in message
        assert "value" in message

    async def test_on_init_can_call_another_plugins_hook(self) -> None:
        seen: list = []

        class Caller(CorvidaePlugin):
            @hookimpl
            async def on_init(self, pm, config: dict) -> None:
                await super().on_init(pm, config)
                seen.append(await pm.ahook.resolve_web_answer(value=5))

        pm = _pm(("caller", Caller()), ("web", WebOwner()), ("answerer", Answerer()))
        install_plugin_hookspecs(pm)
        await pm.ahook.on_init(pm=pm, config={})

        assert seen == [10]

    async def test_runtime_installs_before_guard_and_on_init(self) -> None:
        """Runtime.start: install → arg-binding guard → dependency validation
        → orphan report → on_init."""
        import yaml

        from corvidae.runtime import Runtime

        order: list[str] = []

        def recorder(name: str):
            def record(*args, **kwargs):
                order.append(name)
            return record

        async def on_init(**kwargs):
            order.append("on_init")
            return []

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"llm": {"main": {"base_url": "http://x", "model": "m"}}}, f)
            config_path = f.name

        try:
            from unittest.mock import MagicMock

            pm = MagicMock()
            pm.ahook.on_init = AsyncMock(side_effect=on_init)
            pm.ahook.on_start = AsyncMock(return_value=[])
            pm.ahook.on_stop = AsyncMock(return_value=[])
            agent = MagicMock()
            agent.on_start = AsyncMock()
            pm.get_plugin.return_value = agent
            pm.load_setuptools_entrypoints.side_effect = recorder("load")

            with patch("corvidae.runtime.create_plugin_manager", return_value=pm), \
                 patch("corvidae.runtime.configure_logging"), \
                 patch("corvidae.runtime.install_plugin_hookspecs", side_effect=recorder("install")), \
                 patch("corvidae.runtime._check_hook_arg_binding", side_effect=recorder("guard")), \
                 patch("corvidae.runtime.validate_dependencies", side_effect=recorder("deps")), \
                 patch("corvidae.runtime.report_orphan_hookimpls", side_effect=recorder("orphans")):
                await Runtime(config_path=config_path).start()
        finally:
            os.unlink(config_path)

        assert order == ["load", "install", "guard", "deps", "orphans", "on_init"], order


# ---------------------------------------------------------------------------
# R3: a hard dependency on a missing owner still fails startup
# ---------------------------------------------------------------------------


class TestHardDependency:
    """R3."""

    def test_missing_owner_fails_dependency_validation(self) -> None:
        pm = _pm(("answerer", Answerer()))
        install_plugin_hookspecs(pm)

        with pytest.raises(RuntimeError) as excinfo:
            validate_dependencies(pm)
        assert "'web'" in str(excinfo.value)


# ---------------------------------------------------------------------------
# R4: optional dependency on an absent owner
# ---------------------------------------------------------------------------


class TestOptionalDependency:
    """R4."""

    async def test_absent_blocked_owner_does_not_stop_startup(self) -> None:
        adder = PanelAdder()
        pm = create_plugin_manager()
        pm.set_blocked("web")
        pm.register(WebOwner(), name="web")  # blocked: not registered
        pm.register(adder, name="adder")

        install_plugin_hookspecs(pm)
        _check_hook_arg_binding(pm)
        validate_dependencies(pm)
        report_orphan_hookimpls(pm)
        await pm.ahook.on_init(pm=pm, config={})

        assert pm.get_plugin("web") is None
        assert pm.get_plugin("adder") is adder

    def test_mutual_optional_dependencies_are_not_a_cycle(self) -> None:
        class A(CorvidaePlugin):
            optional_depends_on = frozenset({"b"})

        class B(CorvidaePlugin):
            optional_depends_on = frozenset({"a"})

        pm = _pm(("a", A()), ("b", B()))
        validate_dependencies(pm)  # must not raise


# ---------------------------------------------------------------------------
# R5 / R5a: orphan hookimpls
# ---------------------------------------------------------------------------


def _orphan_records(caplog: pytest.LogCaptureFixture, level: int) -> list[logging.LogRecord]:
    return [
        r for r in caplog.records
        if r.name == _PM_LOGGER and r.levelno == level
    ]


class TestOrphanReport:
    """R5 and R5a."""

    def test_typo_in_hook_name_warns_naming_plugin_and_hook(self, caplog) -> None:
        class Typo(CorvidaePlugin):
            depends_on = frozenset({"web"})

            @hookimpl
            async def on_web_console_inti(self, panels: list) -> None:
                pass

        pm = _pm(("web", WebOwner()), ("typo", Typo()))
        install_plugin_hookspecs(pm)

        with caplog.at_level(logging.DEBUG, logger=_PM_LOGGER):
            report_orphan_hookimpls(pm)

        warnings = _orphan_records(caplog, logging.WARNING)
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert "typo" in message
        assert "on_web_console_inti" in message
        assert "depends_on" in message  # remedy names the dependency fix

    def test_typo_in_grammar_prefix_warns(self, caplog) -> None:
        class BadPrefix(CorvidaePlugin):
            @hookimpl
            async def onweb_console_init(self, panels: list) -> None:
                pass

        pm = _pm(("badprefix", BadPrefix()))
        install_plugin_hookspecs(pm)

        with caplog.at_level(logging.DEBUG, logger=_PM_LOGGER):
            report_orphan_hookimpls(pm)

        warnings = _orphan_records(caplog, logging.WARNING)
        assert len(warnings) == 1
        assert "onweb_console_init" in warnings[0].getMessage()
        assert "badprefix" in warnings[0].getMessage()

    def test_absent_optional_owner_logs_debug_not_warning(self, caplog) -> None:
        pm = create_plugin_manager()
        pm.set_blocked("web")
        pm.register(PanelAdder(), name="adder")
        install_plugin_hookspecs(pm)

        with caplog.at_level(logging.DEBUG, logger=_PM_LOGGER):
            report_orphan_hookimpls(pm)

        assert _orphan_records(caplog, logging.WARNING) == []
        debug = [
            r for r in _orphan_records(caplog, logging.DEBUG)
            if "on_web_console_init" in r.getMessage()
        ]
        assert len(debug) == 1
        assert "'adder'" in debug[0].getMessage()
        assert "'web'" in debug[0].getMessage()
        assert "not loaded" in debug[0].getMessage()

    # R5a downgrades only when all three hold: the hook carries an owner
    # segment, that owner is a declared optional dependency, and it isn't
    # loaded. The next two tests flip the conditions the others don't.

    def test_typo_with_loaded_optional_owner_warns(self, caplog) -> None:
        class OptionalTypo(CorvidaePlugin):
            optional_depends_on = frozenset({"web"})

            @hookimpl
            async def on_web_console_inti(self, panels: list) -> None:
                pass

        pm = _pm(("web", WebOwner()), ("optyp", OptionalTypo()))
        install_plugin_hookspecs(pm)

        with caplog.at_level(logging.DEBUG, logger=_PM_LOGGER):
            report_orphan_hookimpls(pm)

        warnings = _orphan_records(caplog, logging.WARNING)
        assert len(warnings) == 1
        assert "'optyp'" in warnings[0].getMessage()
        assert "on_web_console_inti" in warnings[0].getMessage()
        assert not any(
            "not loaded" in r.getMessage()
            for r in _orphan_records(caplog, logging.DEBUG)
        )

    def test_orphan_not_named_for_absent_optional_owner_warns(self, caplog) -> None:
        class Mixed(CorvidaePlugin):
            optional_depends_on = frozenset({"web"})

            @hookimpl
            async def on_other_thing(self) -> None:
                pass

        pm = create_plugin_manager()
        pm.set_blocked("web")
        pm.register(Mixed(), name="mixed")
        install_plugin_hookspecs(pm)

        with caplog.at_level(logging.DEBUG, logger=_PM_LOGGER):
            report_orphan_hookimpls(pm)

        warnings = _orphan_records(caplog, logging.WARNING)
        assert len(warnings) == 1
        assert "on_other_thing" in warnings[0].getMessage()

    def test_absent_owner_without_optional_dependency_warns(self, caplog) -> None:
        """R5a only applies to a *declared* optional owner."""

        class Undeclared(CorvidaePlugin):
            @hookimpl
            async def on_web_console_init(self, panels: list) -> None:
                pass

        pm = _pm(("undeclared", Undeclared()))
        install_plugin_hookspecs(pm)

        with caplog.at_level(logging.DEBUG, logger=_PM_LOGGER):
            report_orphan_hookimpls(pm)

        assert len(_orphan_records(caplog, logging.WARNING)) == 1

    def test_in_tree_plugins_produce_no_orphan_warnings(self, caplog) -> None:
        pm = create_plugin_manager()
        pm.load_setuptools_entrypoints("corvidae")
        install_plugin_hookspecs(pm)

        with caplog.at_level(logging.DEBUG, logger=_PM_LOGGER):
            report_orphan_hookimpls(pm)

        assert _orphan_records(caplog, logging.WARNING) == []


# ---------------------------------------------------------------------------
# R6: duplicate declarations fail startup naming both owners
# ---------------------------------------------------------------------------


class TestDuplicateDeclaration:
    """R6."""

    @pytest.mark.parametrize("owner_first", [True, False], ids=["owner-first", "dup-first"])
    def test_second_plugin_redeclaring_a_hook_fails(self, owner_first: bool) -> None:
        pairs = [("web", WebOwner()), ("web2", _owner_with(WebSpec))]
        pm = _pm(*(pairs if owner_first else reversed(pairs)))

        message = _install_error(pm)

        assert "resolve_web_answer" in message or "on_web_console_init" in message
        assert "'web'" in message
        assert "'web2'" in message
        assert "depend on" in message
        assert "rename" in message

    def test_redeclaring_an_agentspec_hook_fails(self) -> None:
        class InitSpec:
            @hookspec
            async def on_init(self, pm, config: dict) -> None:
                pass

        pm = _pm(("rogue", _owner_with(InitSpec)))

        message = _install_error(pm)

        assert "on_init" in message
        assert "AgentSpec" in message
        assert "'rogue'" in message
        assert "rename" in message

    def test_subclass_registered_under_second_name_fails(self) -> None:
        class WebSubclass(WebOwner):
            pass

        pm = _pm(("web", WebOwner()), ("web_extra", WebSubclass()))

        message = _install_error(pm)

        assert "'web'" in message
        assert "'web_extra'" in message


# ---------------------------------------------------------------------------
# R7: owner segment and no bare imperatives
# ---------------------------------------------------------------------------


class TestOwnerSegment:
    """R7."""

    def test_wrong_owner_segment_fails(self) -> None:
        class OtherSpec:
            @hookspec
            async def on_other_thing(self) -> None:
                pass

        message = _install_error(_pm(("web", _owner_with(OtherSpec))))

        assert "on_other_thing" in message
        assert "'web'" in message
        assert "on_web_" in message  # the accepted form

    def test_empty_remainder_after_owner_fails(self) -> None:
        class EmptySpec:
            @hookspec
            async def on_web_(self) -> None:
                pass

        message = _install_error(_pm(("web", _owner_with(EmptySpec))))

        assert "'on_web_'" in message
        assert "'web'" in message
        assert "on_web_<name>" in message  # the accepted form

    def test_bare_imperative_name_fails(self) -> None:
        class ImperativeSpec:
            @hookspec
            async def register_web_panels(self, panels: list) -> None:
                pass

        message = _install_error(_pm(("web", _owner_with(ImperativeSpec))))

        assert "register_web_panels" in message
        assert "'web'" in message
        for prefix in ("on_", "should_", "resolve_", "transform_"):
            assert prefix in message  # accepted prefixes are listed

    @pytest.mark.parametrize(
        "name, opts",
        [
            ("on_web_x", {}),
            ("before_web_x", {}),
            ("after_web_x", {}),
            ("should_web_x", {}),
            ("resolve_web_x", {"firstresult": True}),
        ],
    )
    def test_every_prefixed_class_is_accepted(self, name: str, opts: dict) -> None:
        async def spec_method(self) -> None:
            pass

        spec_cls = type("Spec", (), {name: hookspec(**opts)(spec_method)})
        pm = _pm(("web", _owner_with(spec_cls)))

        install_plugin_hookspecs(pm)

        assert getattr(pm.hook, name).spec is not None


# ---------------------------------------------------------------------------
# R11: plugin specs are grammar-checked at startup, one shared rule table
# ---------------------------------------------------------------------------


class TestGrammarCheck:
    """R11."""

    @pytest.mark.parametrize(
        "name, firstresult",
        [
            ("on_web_thing", True),
            ("before_web_thing", True),
            ("after_web_thing", True),
            ("should_web_thing", True),
            ("resolve_web_thing", False),
            ("transform_web_thing", False),
        ],
    )
    def test_options_not_matching_prefix_fail(self, name: str, firstresult: bool) -> None:
        async def spec_method(self) -> None:
            pass

        spec_cls = type("Spec", (), {name: hookspec(firstresult=firstresult)(spec_method)})

        message = _install_error(_pm(("web", _owner_with(spec_cls))))

        assert name in message
        assert "'web'" in message
        assert "firstresult" in message

    def test_transform_without_owner_seed_fails(self) -> None:
        class Spec:
            @hookspec(firstresult=True)
            async def transform_web_text(self, text: str) -> str:
                pass

        message = _install_error(_pm(("web", _owner_with(Spec))))

        assert "transform_web_text" in message
        assert "'web'" in message
        assert "trylast" in message

    async def test_transform_with_owner_seed_is_accepted(self) -> None:
        class Spec:
            @hookspec(firstresult=True)
            async def transform_web_text(self, text: str) -> str:
                pass

        class SeededOwner(CorvidaePlugin):
            hookspecs = Spec

            @hookimpl(trylast=True)
            async def transform_web_text(self, text: str) -> str:
                return text

        pm = _pm(("web", SeededOwner()))
        install_plugin_hookspecs(pm)

        assert await pm.ahook.transform_web_text(text="hi") == "hi"

    def test_spec_class_without_hookspecs_fails(self) -> None:
        class EmptySpec:
            def not_a_hook(self) -> None:
                pass

        message = _install_error(_pm(("web", _owner_with(EmptySpec))))

        assert "'web'" in message
        assert "EmptySpec" in message

    def test_grammar_table_lives_in_the_kernel(self) -> None:
        """The runtime check and tests/test_hook_grammar.py share one table."""
        import tests.test_hook_grammar as grammar_test
        from corvidae.hooks import HOOK_GRAMMAR, IMPERATIVE_HOOKS, grammar_class

        assert grammar_test.GRAMMAR is HOOK_GRAMMAR
        assert grammar_test.IMPERATIVE is IMPERATIVE_HOOKS
        assert grammar_class("resolve_web_x") == "resolve"
        assert grammar_class("register_tools") is None
