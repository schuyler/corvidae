"""Shared test fixtures."""

import logging
import warnings

import pytest
import pytest_asyncio
import aiosqlite

# Safety net: suppress chromadb deprecation warning that triggers SIGINT in pytest async runner
# The pytest filterwarnings config also handles this, but conftest-level suppression
# ensures the filter is active before any test module imports happen.
warnings.filterwarnings("ignore", message=".*iscoroutinefunction.*deprecated.*")


def pytest_addoption(parser):
    parser.addoption(
        "--run-eval", action="store_true", default=False,
        help="Run live LLM evaluation tests (marked @pytest.mark.eval)",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-eval"):
        skip_eval = pytest.mark.skip(reason="Needs --run-eval option")
        for item in items:
            if "eval" in [m.name for m in item.iter_markers()]:
                item.add_marker(skip_eval)


@pytest_asyncio.fixture
async def db():
    from corvidae.persistence import init_db
    async with aiosqlite.connect(":memory:") as conn:
        await init_db(conn)
        yield conn


@pytest.fixture
def plugin_manager():
    from corvidae.hooks import create_plugin_manager
    return create_plugin_manager()


@pytest.fixture(autouse=True)
def _reset_corvidae_logger():
    """Ensure the corvidae logger propagates to root so caplog captures records.

    Other test modules (test_logging.py) may apply dictConfig with
    propagate=False on the corvidae logger. This fixture resets it.
    """
    corvidae_logger = logging.getLogger("corvidae")
    original_propagate = corvidae_logger.propagate
    original_handlers = corvidae_logger.handlers[:]
    corvidae_logger.propagate = True
    yield
    corvidae_logger.propagate = original_propagate
    corvidae_logger.handlers = original_handlers
