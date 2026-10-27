"""The hook grammar: a hookspec's name tells you its resolution semantics.

Every hookspec name starts with a prefix that fixes its pluggy options, or
is a bare imperative verb on a short allowlist (plans/kernel-split.md,
Part 1 §1; docs/plugin-guide.md, "Hook grammar"). These tests check names
against pluggy options. Whether the core reads a hook's return value is not
something pluggy exposes, so that half of the grammar is enforced at review
time by the admission criterion, not here.

Every test is parametrized per hook name, so a failure names the hook.
"""

from collections.abc import Iterable

import pytest
from pluggy import HookCaller

from corvidae.hooks import AgentSpec, _SeedHooksPlugin, create_plugin_manager

# Prefix class -> (prefixes, the pluggy spec options every member must have).
GRAMMAR: dict[str, tuple[tuple[str, ...], dict[str, bool]]] = {
    "observe": (("on_", "before_", "after_"), {"firstresult": False}),
    "veto": (("should_",), {"firstresult": False}),
    "resolve": (("resolve_",), {"firstresult": True}),
    "transform": (("transform_",), {"firstresult": True}),
}

# Bare imperative names: broadcast, caller-defined use of the result list.
IMPERATIVE = {"send", "register_tools", "persist_message"}

# Names retired by renames or folds; none may come back as a hookspec.
RETIRED = {
    "on_conversation_event",  # -> persist_message
    "process_tool_result",  # -> transform_tool_result
    "load_conversation",  # -> resolve_conversation
    "on_llm_error",  # -> resolve_llm_error
    "compact_conversation",  # -> resolve_compaction
    "send_message",  # -> send(event=OutboundMessage)
    "send_thinking",  # -> send(event=OutboundThinking)
    "send_tool_status",  # -> send(event=OutboundToolStatus)
    "send_progress",  # -> send(event=OutboundProgress)
}


def _hookcallers() -> dict[str, HookCaller]:
    """Map hook name -> HookCaller for every hookspec on a fresh manager.

    Walks ``pm.hook.__dict__`` the way ``_check_hook_arg_binding`` does;
    callers without a spec (impl-only names) are not hookspecs and are
    skipped.
    """
    pm = create_plugin_manager()
    return {
        name: caller
        for name, caller in pm.hook.__dict__.items()
        if getattr(caller, "spec", None) is not None
    }


_CALLERS = _hookcallers()


def _grammar_class(name: str) -> str | None:
    """Return the grammar class whose prefix ``name`` starts with, or None."""
    for klass, (prefixes, _opts) in GRAMMAR.items():
        if name.startswith(prefixes):
            return klass
    return None


def _rows(names: Iterable[str]) -> list:
    """Parametrize rows, one per hook name."""
    return [pytest.param(n, id=n) for n in sorted(names)]


# --- Test 1: every name is in the grammar ------------------------------------


@pytest.mark.parametrize("name", _rows(_CALLERS))
def test_name_has_grammar_prefix_or_is_imperative(name: str) -> None:
    """Each hookspec name has a grammar prefix or is on the imperative list."""
    assert _grammar_class(name) is not None or name in IMPERATIVE, (
        f"hookspec {name!r} has no grammar prefix "
        f"({', '.join(p for ps, _ in GRAMMAR.values() for p in ps)}) "
        f"and is not on the imperative allowlist {sorted(IMPERATIVE)}"
    )


# --- Test 2: the prefix fixes the pluggy options ------------------------------


@pytest.mark.parametrize("name", _rows(_CALLERS))
def test_prefix_matches_pluggy_options(name: str) -> None:
    """A prefixed hookspec has exactly the pluggy options its class requires.

    A name with no prefix is test 1's failure, not this one's, so it skips
    rather than failing twice for one cause.
    """
    klass = _grammar_class(name)
    if klass is None:
        pytest.skip(f"{name!r} has no grammar prefix; test 1 covers it")
    _prefixes, expected = GRAMMAR[klass]
    opts = _CALLERS[name].spec.opts
    for option, value in expected.items():
        assert bool(opts.get(option)) is value, (
            f"{name!r} is in the {klass!r} class, which requires "
            f"{option}={value}, but its hookspec has {option}={opts.get(option)}"
        )


# --- Test 3: transform hooks have a seed --------------------------------------


@pytest.mark.parametrize(
    "name", _rows(n for n in _CALLERS if _grammar_class(n) == "transform")
)
def test_transform_hook_has_trylast_seed(name: str) -> None:
    """Each transform_ hook has a trylast seed on _SeedHooksPlugin.

    The seed returns its input unchanged, so a wrapper chain always has a
    value to transform even when no plugin implements the hook.
    """
    seeds = [
        impl
        for impl in _CALLERS[name].get_hookimpls()
        if isinstance(impl.plugin, _SeedHooksPlugin)
    ]
    assert seeds, f"transform hook {name!r} has no _SeedHooksPlugin implementation"
    assert all(impl.trylast for impl in seeds), (
        f"_SeedHooksPlugin.{name} must be registered with trylast=True"
    )


# --- Test 4: imperative hooks exist and are broadcast -------------------------


@pytest.mark.parametrize("name", _rows(IMPERATIVE))
def test_imperative_hook_exists_and_is_broadcast(name: str) -> None:
    """Each allowlisted imperative name is a broadcast hookspec."""
    assert name in _CALLERS, f"imperative hook {name!r} has no hookspec"
    assert not _CALLERS[name].spec.opts.get("firstresult"), (
        f"imperative hook {name!r} must be broadcast (firstresult=False)"
    )


# --- Test 5: retired names stay retired ----------------------------------------


@pytest.mark.parametrize("name", _rows(RETIRED))
def test_retired_name_is_not_a_hookspec(name: str) -> None:
    """A renamed or folded hook's old name is gone from AgentSpec.

    This is the only test that pins ``on_conversation_event`` ->
    ``persist_message``: the old name has a valid ``on_`` prefix and broadcast
    options, so tests 1 and 2 cannot see that the core reads its result.
    """
    assert not hasattr(AgentSpec, name), f"{name!r} is still a hookspec on AgentSpec"
