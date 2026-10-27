"""Cognition loggers keep their ``corvidae.*`` names after the move.

corvidae/logging.py configures the "corvidae" logger at the operator's
level and the root at WARNING. A cognition module using
``logging.getLogger(__name__)`` would log as ``corvidae_cognition.*`` and
lose its INFO and DEBUG lines (plans/kernel-split.md, Traps).
"""

import importlib
import pkgutil

import pytest

corvidae_cognition = pytest.importorskip("corvidae_cognition")

# Every module in the package except the package itself.
MODULES = sorted(
    info.name
    for info in pkgutil.iter_modules(corvidae_cognition.__path__, "corvidae_cognition.")
)


@pytest.mark.parametrize("name", MODULES)
def test_module_logger_is_in_the_corvidae_namespace(name: str) -> None:
    """Each module has a module-level ``logger`` named ``corvidae.<...>``."""
    # Import the module and read its module-level logger, if any.
    module = importlib.import_module(name)
    logger = getattr(module, "logger", None)
    assert logger is not None, f"{name} has no module-level logger"
    assert logger.name.startswith("corvidae."), (
        f"{name}'s logger is {logger.name!r}; it must start with 'corvidae.'"
    )
