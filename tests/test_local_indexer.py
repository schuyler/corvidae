"""Tests for local_indexer plugin."""

import asyncio
from pathlib import Path

import pytest

from corvidae.tools.local_indexer import (
    LocalIndexer,
    LocalIndexerPlugin,
    _chunk_text,
    hamming_distance,
    simhash,
)


# ---------------------------------------------------------------------------
# Simhash tests
# ---------------------------------------------------------------------------


class TestSimhash:
    def test_simhash_deterministic(self):
        text = "The quick brown fox jumps over the lazy dog"
        sh1 = simhash(text)
        sh2 = simhash(text)
        assert sh1 == sh2

    def test_simhash_different_text_different_hash(self):
        sh1 = simhash("hello world")
        sh2 = simhash("goodbye universe")
        assert sh1 != sh2

    def test_simhash_similar_text_same_bits_more(self):
        """Similar text should share more bits in simhash."""
        base = "the quick brown fox"
        similar = "the quick brown cat"
        sh_base = simhash(base)
        sh_similar = simhash(similar)
        dist = hamming_distance(sh_base, sh_similar)

        # Unrelated text should have ~32 bit difference (half of 64)
        unrelated_sh = simhash("xyz random gibberish")
        dist_unrelated = hamming_distance(sh_base, unrelated_sh)

        assert dist < dist_unrelated

    def test_simhash_empty_string(self):
        sh = simhash("")
        assert isinstance(sh, int)
        assert sh >= 0


class TestHammingDistance:
    def test_identical_hashes(self):
        assert hamming_distance(0x1234, 0x1234) == 0

    def test_completely_different(self):
        # 0xFFFFFFFFFFFFFFFF vs 0x0000000000000000 should be 64
        assert hamming_distance(0xFFFFFFFFFFFFFFFF, 0x0) == 64

    def test_single_bit_difference(self):
        a = 0b1000
        b = 0b1001
        assert hamming_distance(a, b) == 1


# ---------------------------------------------------------------------------
# Chunking tests
# ---------------------------------------------------------------------------


class TestChunkText:
    def test_short_text_no_split(self):
        text = "Short text"
        chunks = _chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_splits(self):
        text = "word " * 200 + "end"  # ~800 chars
        chunks = _chunk_text(text)
        assert len(chunks) > 1

    def test_empty_text(self):
        chunks = _chunk_text("")
        assert chunks == []

    def test_only_whitespace(self):
        chunks = _chunk_text("   \n\n  ")
        assert chunks == []


# ---------------------------------------------------------------------------
# SQLite backend tests
# ---------------------------------------------------------------------------


@pytest.fixture
async def indexer(tmp_path):
    """Create a LocalIndexer instance with a temporary DB."""
    db_path = tmp_path / "test_index.db"
    idx = LocalIndexer(db_path)
    await idx.connect()
    yield idx
    await idx.close()


class TestLocalIndexer:
    async def test_index_file(self, indexer, tmp_path):
        """Indexing a file should create chunks."""
        test_file = tmp_path / "test.md"
        test_file.write_text("Hello world! This is a test file.")

        count = await indexer.index_file(str(test_file))
        assert count > 0

    async def test_index_empty_file(self, indexer, tmp_path):
        """Empty files should not create chunks."""
        test_file = tmp_path / "empty.md"
        test_file.write_text("")

        count = await indexer.index_file(str(test_file))
        assert count == 0

    async def test_index_skips_unchanged(self, indexer, tmp_path):
        """Re-indexing unchanged file should return 0."""
        test_file = tmp_path / "test.md"
        test_file.write_text("Stable content")

        first = await indexer.index_file(str(test_file))
        second = await indexer.index_file(str(test_file))
        assert second == 0

    async def test_index_updates_on_change(self, indexer, tmp_path):
        """Modifying file should re-index."""
        import time

        test_file = tmp_path / "test.md"
        test_file.write_text("Original content")

        await indexer.index_file(str(test_file))

        # Small delay to ensure mtime changes
        time.sleep(0.01)
        test_file.write_text("Modified content")

        count = await indexer.index_file(str(test_file))
        assert count > 0

    async def test_index_directory(self, indexer, tmp_path):
        """Indexing a directory should process all text files."""
        (tmp_path / "a.md").write_text("File A content")
        (tmp_path / "b.txt").write_text("File B content")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub/c.md").write_text("Sub file")

        count = await indexer.index_directory(tmp_path)
        assert count > 0


# ---------------------------------------------------------------------------
# Search tests
# ---------------------------------------------------------------------------


@pytest.fixture
async def indexed_indexer(tmp_path):
    """Create and populate an indexer with test files."""
    db_path = tmp_path / "search_test.db"
    idx = LocalIndexer(db_path)
    await idx.connect()

    # Create test files
    (tmp_path / "doc1.md").write_text(
        "Machine learning is a subset of artificial intelligence."
    )
    (tmp_path / "doc2.md").write_text(
        "Python is a popular programming language for data science."
    )
    (tmp_path / "doc3.md").write_text(
        "The quick brown fox jumps over the lazy dog."
    )

    await idx.index_directory(tmp_path)
    yield idx, tmp_path
    await idx.close()


class TestSearch:
    async def test_fts_search(self, indexed_indexer):
        """Full-text search should find exact keyword matches."""
        idx, _ = indexed_indexer
        results = await idx.search_text("machine learning")
        assert len(results) > 0
        # Should match doc1.md
        assert "doc1.md" in results[0].file_path

    async def test_similar_search(self, indexed_indexer):
        """Similarity search should find related content."""
        idx, _ = indexed_indexer
        results = await idx.search_similar("artificial intelligence")
        # Should find doc1 (machine learning/artificial intelligence related)
        assert len(results) > 0

    async def test_hybrid_search(self, indexed_indexer):
        """Hybrid search should combine FTS and similarity results."""
        idx, _ = indexed_indexer
        results = await idx.hybrid_search("programming language")
        # Should find doc2 (Python programming)
        assert len(results) > 0

    async def test_no_results(self, indexed_indexer):
        """Search with no matches should return empty list."""
        idx, _ = indexed_indexer
        results = await idx.search_text("xyznonexistent")
        assert len(results) == 0

    async def test_result_has_simhash_int(self, indexed_indexer):
        """Results should have simhash as integer (converted from hex)."""
        idx, _ = indexed_indexer
        results = await idx.search_text("machine")
        if results:
            assert isinstance(results[0].simhash, int)


# ---------------------------------------------------------------------------
# Plugin integration tests
# ---------------------------------------------------------------------------


class TestLocalIndexerPlugin:
    async def test_plugin_registers_tool(self, tmp_path):
        """Plugin should register search_workspace tool."""
        plugin = LocalIndexerPlugin()
        registry = []
        plugin.register_tools(tool_registry=registry)
        assert len(registry) == 1

    async def test_search_workspace_returns_string_no_indexer(self):
        """search_workspace should return error string when no indexer."""
        plugin = LocalIndexerPlugin()
        registry = []
        plugin.register_tools(tool_registry=registry)
        tool_fn = registry[0]

        result = await tool_fn("test")
        assert "Local indexer not initialized" in result

    async def test_search_workspace_returns_string_with_indexer(self, tmp_path):
        """search_workspace should return formatted results when indexed."""
        plugin = LocalIndexerPlugin()
        from corvidae.tools.local_indexer import LocalIndexer

        db_path = tmp_path / "test_plugin.db"
        plugin.indexer = LocalIndexer(db_path)
        await plugin.indexer.connect()
        plugin._workspace_root = tmp_path

        # Index a test file
        (tmp_path / "test.md").write_text("Test content for workspace search")
        await plugin.indexer.index_file(str(tmp_path / "test.md"))

        registry = []
        plugin.register_tools(tool_registry=registry)
        tool_fn = registry[0]

        result = await tool_fn("test content", mode="text")
        assert isinstance(result, str)
        assert "Test content" in result or "No results" not in result

        await plugin.indexer.close()
