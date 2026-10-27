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
