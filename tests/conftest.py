"""Shared test fixtures."""

import pytest
import pytest_asyncio
import aiosqlite


@pytest_asyncio.fixture
async def db():
    from corvidae.conversation import init_db
    async with aiosqlite.connect(":memory:") as conn:
        await init_db(conn)
        yield conn


@pytest.fixture
def plugin_manager():
    from corvidae.hooks import create_plugin_manager
    return create_plugin_manager()
