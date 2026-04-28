"""Shared test fixtures."""

import logging

import pytest
import pytest_asyncio
import aiosqlite


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
