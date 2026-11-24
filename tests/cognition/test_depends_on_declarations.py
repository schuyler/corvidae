"""The cognition plugins follow the kernel's plugin conventions.

The cognition half of tests/test_depends_on_declarations.py's
``_REGISTERED_PLUGIN_CLASSES`` check: every plugin has a ``depends_on``
class attribute and accepts ``pm`` as its first positional argument.
"""

import importlib
import inspect

import pytest

# Cognition plugins that are constructed with pm, by dotted path.
_COGNITION_PLUGIN_CLASSES = [
    "corvidae_cognition.memory.MemoryPlugin",
    "corvidae_cognition.funnel.FunnelPlugin",
    "corvidae_cognition.gates.GatePlugin",
    "corvidae_cognition.heartbeat.HeartbeatPlugin",
    "corvidae_cognition.goals.GoalsPlugin",
]


def _load(dotted_name: str) -> type:
    """Import a class by its dotted path."""
    module_path, class_name = dotted_name.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), class_name)


@pytest.mark.parametrize("dotted_name", _COGNITION_PLUGIN_CLASSES)
def test_plugin_has_depends_on(dotted_name: str) -> None:
    """Every cognition plugin has a depends_on class attribute."""
    assert hasattr(_load(dotted_name), "depends_on"), (
        f"{dotted_name} must have a 'depends_on' class attribute"
    )


@pytest.mark.parametrize("dotted_name", _COGNITION_PLUGIN_CLASSES)
def test_plugin_init_accepts_pm_as_first_positional_arg(dotted_name: str) -> None:
    """Every cognition plugin accepts pm as its first positional argument."""
    params = list(inspect.signature(_load(dotted_name).__init__).parameters.values())
    # params[0] is self; params[1] must be a positional 'pm'.
    assert len(params) >= 2, f"{dotted_name}.__init__ takes no arguments besides self"
    assert params[1].name == "pm", (
        f"{dotted_name}.__init__ first non-self parameter must be 'pm', "
        f"got {params[1].name!r}"
    )
    assert params[1].kind in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ), f"{dotted_name}.__init__ 'pm' must be positional"


def test_memory_depends_on_kernel_clock_and_idle() -> None:
    """MemoryPlugin declares the kernel clock and idle tracker (R1, R4, R5).

    Production's validate_dependencies enforces these names even though
    memory resolves the objects leniently.
    """
    memory = _load("corvidae_cognition.memory.MemoryPlugin")
    assert memory.depends_on == frozenset({"persistence", "llm", "clock", "idle", "trace", "registry"})


def test_goals_no_longer_depends_on_the_funnel() -> None:
    """GoalsPlugin keeps memory and the clock but drops the funnel (#1176, R5.2b).

    Goals must load with the funnel switched off, so "funnel" cannot be a
    hard dependency; the held-text store is the kernel's.
    """
    goals = _load("corvidae_cognition.goals.GoalsPlugin")
    assert "funnel" not in goals.depends_on
    assert {"persistence", "memory", "clock"} <= goals.depends_on


@pytest.mark.parametrize(
    "dotted_name",
    [
        "corvidae_cognition.recall.RecallSourceMixin",
        "corvidae_cognition.markdown_recall.MarkdownRecallPlugin",
    ],
)
def test_recall_sources_no_longer_list_the_funnel_as_optional(dotted_name: str) -> None:
    """Recall sources contribute through before_compose, not the funnel (#1176)."""
    cls = _load(dotted_name)
    assert "funnel" not in getattr(cls, "optional_depends_on", frozenset())
    assert "funnel" not in getattr(cls, "depends_on", frozenset())


def test_critique_depends_on_held() -> None:
    """CritiquePlugin hands verdicts to the kernel's held store, so it declares it (#1176)."""
    critique = _load("corvidae_cognition.critique.CritiquePlugin")
    assert "held" in critique.depends_on
    assert "funnel" not in critique.depends_on
