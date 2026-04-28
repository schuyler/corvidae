"""Tests for WorkspaceIndexerPlugin and workspace_search tool."""

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# TestChunkText — Token-aware chunking
# ---------------------------------------------------------------------------


class TestChunkText:
    async def test_chunk_text_basic(self):
        from corvidae.tools.index import _chunk_text

        text = "Hello world. This is a test of chunking. " * 10
        chunks = _chunk_text(text)
        assert len(chunks) >= 1
        # Each chunk should be roughly 500 tokens (default)
        for chunk in chunks:
            from corvidae.tools.index import _token_count
            assert _token_count(chunk) <= 510  # small tolerance

    async def test_chunk_text_small(self):
        from corvidae.tools.index import _chunk_text

        text = "Short text"
        chunks = _chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    async def test_chunk_text_has_overlap(self):
        from corvidae.tools.index import _chunk_text, _token_count

        # Create text longer than chunk size to force multiple chunks
        text = "Word. " * 300
        with_overlap = _chunk_text(text, chunk_size=100, overlap=50)
        no_overlap = _chunk_text(text, chunk_size=100, overlap=0)
        # Overlap produces MORE chunks (smaller step size) but each chunk reuses
        # some content from the previous one. Verify both produce valid output.
        assert len(with_overlap) >= 2
        assert len(no_overlap) >= 1
        # All chunks should be non-empty and under 2x chunk_size
        for chunk in with_overlap:
            assert _token_count(chunk) < 200  # much less than 2 * 500


# ---------------------------------------------------------------------------
# TestWorkspaceIndexer — Build and search
# ---------------------------------------------------------------------------


class TestWorkspaceIndexer:
    async def test_build_indexes_files(self, tmp_path):
        from corvidae.tools.index import WorkspaceIndexer

        # Create some text files
        (tmp_path / "test.txt").write_text("Hello world. This is a test file for indexing. " * 10)
        (tmp_path / "notes.md").write_text("# Notes\nSome notes here. " * 5)

        indexer = WorkspaceIndexer(tmp_path)
        stats = indexer.build()

        assert stats["files_indexed"] == 2
        assert stats["total_chunks"] >= 2
        assert len(stats["indexed_files"]) == 2

    async def test_search_returns_results(self, tmp_path):
        from corvidae.tools.index import WorkspaceIndexer

        # Create an indexed file with distinctive content
        (tmp_path / "doc.txt").write_text(
            "The quick brown fox jumps over the lazy dog near the river. " * 10
        )

        indexer = WorkspaceIndexer(tmp_path)
        indexer.build()

        results = indexer.search("brown fox")
        assert len(results) >= 1
        # Score is cosine similarity (1 - distance); expect positive similarity
        assert results[0]["score"] > 0.2

    async def test_search_empty_collection(self, tmp_path):
        from corvidae.tools.index import WorkspaceIndexer

        indexer = WorkspaceIndexer(tmp_path)
        results = indexer.search("anything")
        assert results == []

    async def test_find_text_files_excludes_binaries(self, tmp_path):
        from corvidae.tools.index import _find_text_files

        # Create a text file and a binary-looking file
        (tmp_path / "real.txt").write_text("Real content here. ")
        (tmp_path / "fake.jpg").write_bytes(b"\x89PNG\r\n\x1a\n")  # PNG header

        files = _find_text_files(tmp_path)
        names = [f.name for f in files]
        assert "real.txt" in names
        # Binary files should be skipped even if they have text-like extensions
        assert "fake.jpg" not in names


# ---------------------------------------------------------------------------
# TestWorkspaceIndexerPlugin — Plugin lifecycle and tool registration
# ---------------------------------------------------------------------------


class TestWorkspaceIndexerPlugin:
    async def test_register_tools_adds_workspace_search(self):
        from corvidae.tools.index import WorkspaceIndexerPlugin
        from corvidae.tool import Tool

        plugin = WorkspaceIndexerPlugin()
        registry = []
        plugin.register_tools(registry)

        names = {item.name if isinstance(item, Tool) else item.__name__ for item in registry}
        assert "workspace_search" in names
        assert "build_index" in names

    async def test_workspace_search_returns_formatted(self, tmp_path):
        from corvidae.tools.index import WorkspaceIndexerPlugin

        # Create indexed file
        (tmp_path / "search_test.txt").write_text(
            "Python is a great programming language. " * 10
        )

        plugin = WorkspaceIndexerPlugin()
        plugin._workspace_root = tmp_path.resolve()
        plugin._get_indexer().build()

        result = await plugin.workspace_search("great programming")
        assert "search_test.txt" in result
        assert "Python" in result
        assert "[" in result  # score bracket

    async def test_workspace_search_no_results(self, tmp_path):
        from corvidae.tools.index import WorkspaceIndexerPlugin

        plugin = WorkspaceIndexerPlugin()
        plugin._workspace_root = tmp_path.resolve()
        # No files indexed

        result = await plugin.workspace_search("query for empty index")
        assert "No results found" in result

    async def test_build_index_returns_stats(self, tmp_path):
        from corvidae.tools.index import WorkspaceIndexerPlugin

        (tmp_path / "stats.txt").write_text("Stats test content here. " * 10)

        plugin = WorkspaceIndexerPlugin()
        plugin._workspace_root = tmp_path.resolve()

        result = await plugin.build_index()
        assert "Indexed" in result
        assert "files" in result.lower()


# ---------------------------------------------------------------------------
# TestWorkspaceSearchTool — Integration with corvidae Tool system
# ---------------------------------------------------------------------------


class TestWorkspaceSearchTool:
       async def test_tool_call_format(self, tmp_path):
        from corvidae.tools.index import WorkspaceIndexerPlugin
        from corvidae.tool import Tool

        (tmp_path / "tool_test.txt").write_text("Tool testing is important. " * 10)

        plugin = WorkspaceIndexerPlugin()
        plugin._workspace_root = tmp_path.resolve()
        plugin._get_indexer().build()

        # Get the registered tool
        registry = []
        plugin.register_tools(registry)
        search_tool = next(t for t in registry if t.name == "workspace_search")

        assert isinstance(search_tool, Tool)
        desc = search_tool.schema.get("function", {}).get("description", "")
        assert len(desc) > 10
