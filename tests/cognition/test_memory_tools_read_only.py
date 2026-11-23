"""Memory tools' read-only declarations (#1190 PR 1, criterion A2).

search_memory and search_transcripts only read, so a fork may call them;
recall_raw bumps access stats, so it is not declared.
"""

from .test_memory_tools import build_tools_env, get_tools


async def test_memory_tool_read_only_declarations():
    # Import inside the test so a missing symbol fails this test only.
    from corvidae.tool import declares_read_only

    _memory, tools_plugin, db = await build_tools_env()
    try:
        tools = get_tools(tools_plugin)
        assert declares_read_only(tools["search_memory"]) is True
        assert declares_read_only(tools["search_transcripts"]) is True
        assert declares_read_only(tools["recall_raw"]) is False
    finally:
        await db.close()
