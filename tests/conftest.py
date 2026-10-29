"""Shared test fixtures."""

import logging
import warnings

import pytest
import pytest_asyncio
import aiosqlite

# Autouse: closes aiosqlite connections a failing test left open, which would
# otherwise hang pytest at exit (GitHub issue #64).
from aiosqlite_guard import close_leaked_aiosqlite_connections  # noqa: F401

# Suppress iscoroutinefunction deprecation warning that can trigger SIGINT in pytest async runner.
# conftest-level suppression ensures the filter is active before any test module imports happen.
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


@pytest.fixture
def pm():
    """A plugin manager for install_kernel_clock to register on."""
    from corvidae.hooks import create_plugin_manager

    return create_plugin_manager()


@pytest_asyncio.fixture
async def kernel_clock(pm):
    """A FakeClock registered as "clock" plus a started IdleTracker as "idle".

    Yields ``(clock, idle)``. Teardown stops the tracker and closes the
    clock so no timer task outlives the test. The import is deferred so a
    broken kernel clock fails the tests that use it, not collection of the
    whole suite.
    """
    from fake_clock import install_kernel_clock

    clock, idle = install_kernel_clock(pm)
    yield clock, idle
    # Stop the tracker first, then cancel any timer still live.
    idle.stop()
    await clock.close()
