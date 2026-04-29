"""Local workspace indexer plugin.

Uses simhash-based embeddings and SQLite FTS5 for fast full-text search
over the agent's workspace directory. Provides `search_workspace` tool.

No ML dependencies required — uses pure-Python simhash (Rabin fingerprint).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiosqlite
from corvidae.hooks import hookimpl
from corvidae.tool import Tool

logger = logging.getLogger(__name__)


def _text_hash(text: str) -> int:
    """Simple hash for change detection (fast, not cryptographic)."""
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16)


# ---------------------------------------------------------------------------
# Simhash (Rabin fingerprint-based, 64-bit)
# ---------------------------------------------------------------------------

def _rabin_hash(text: str, seed: int = 0x9E3779B97F4A7C15) -> int:
    """Simple Rabin-style rolling hash."""
    h = seed
    for ch in text:
        h = (h * 67 + ord(ch)) & 0xFFFFFFFFFFFFFFFF
    return h


def simhash(text: str, k: int = 3) -> int:
    """Compute a 64-bit simhash from text using character n-grams.

    The vector dimension is 64 (one bit per hash bucket). Each n-gram
    casts a vote for or against each bit position based on its Rabin hash.

    Args:
        text: Input text to embed.
        k: N-gram size (3 = trigram, good balance of speed/quality).

    Returns:
        64-bit integer simhash.
    """
    # Hash each n-gram
    ngrams = [text[i:i+k] for i in range(len(text) - k + 1)]
    weights = [0] * 64

    for ng in ngrams:
        h = _rabin_hash(ng)
        for bit in range(64):
            if h & (1 << bit):
                weights[bit] += 1
            else:
                weights[bit] -= 1

    # Build simhash from majority votes
    result = 0
    for i, w in enumerate(weights):
        if w > 0:
            result |= (1 << i)
    return result


def hamming_distance(a: int, b: int) -> int:
    """Count differing bits between two simhashes."""
    xor = a ^ b
    return bin(xor).count("1")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _chunk_text(text: str, max_chunk_chars: int = 500) -> list[str]:
    """Split text into overlapping chunks suitable for search indexing.

    Chunks are ~500 characters with a 20% overlap to ensure adjacent chunks
    share boundary content.

    Args:
        text: Text to split.
        max_chunk_chars: Maximum size of each chunk (soft limit).

    Returns:
        List of non-empty chunk strings.
    """
    if len(text) <= max_chunk_chars:
        return [text] if text.strip() else []

    chunks = []
    overlap = int(max_chunk_chars * 0.2)
    start = 0

    while start < len(text):
        end = start + max_chunk_chars
        # Try to break at word boundary
        if end < len(text):
            space = text.rfind(" ", start, end)
            if space > start:
                end = space

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Advance past overlap region
        next_start = min(end + 1, len(text))
        if next_start <= start + 1:
            break
        start = max(start + max_chunk_chars - overlap, next_start)

    return chunks


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """A single search result."""
    file_path: str
    chunk_text: str
    simhash: int
    relevance_score: float  # 0.0 to 1.0


class LocalIndexer:
    """SQLite-backed workspace indexer with simhash embeddings.

    Supports incremental indexing, full-text search via FTS5, and
    similarity search via hamming-distance on simhash vectors.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self._db: aiosqlite.Connection | None = None
        self._workspace_root: Path | None = None
        self._file_hashes: dict[str, int] = {}  # file -> mtime hash

    async def connect(self) -> None:
        """Open the SQLite database and create tables if needed."""
        self._db = await aiosqlite.connect(str(self.db_path))
        self._db.row_factory = aiosqlite.Row

        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                simhash_hex TEXT NOT NULL
            )
        """)

        # FTS5 virtual table for full-text search (separate from chunks)
        await self._db.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
            USING fts5(content, file_path, chunk_index)
        """)

        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)

        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    async def index_file(self, file_path: str) -> int:
        """Index a single file. Returns number of chunks indexed.

        Skips files that haven't changed since last indexed (mtime-based).
        """
        path = Path(file_path)
        if not path.exists():
            # Remove stale entries
            await self._remove_file(file_path)
            return 0

        try:
            mtime = int(path.stat().st_mtime * 1000)
        except OSError:
            return 0

        stored_mtime = self._file_hashes.get(file_path)
        if stored_mtime == mtime:
            return 0  # no change

        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.debug(f"Failed to read {file_path}: {e}")
            return 0

        if not text.strip():
            return 0

        chunks = _chunk_text(text)
        count = 0

        for idx, chunk in enumerate(chunks):
            sh = simhash(chunk)
            cursor = await self._db.execute(
                "INSERT OR REPLACE INTO chunks (file_path, chunk_index, content, simhash_hex) "
                "VALUES (?, ?, ?, ?)",
                (file_path, idx, chunk, format(sh, '016x')),
            )
            rowid = cursor.lastrowid or 1
            # Insert into FTS table with same rowid for join compatibility
            await self._db.execute(
                "INSERT OR REPLACE INTO chunks_fts (rowid, content, file_path, chunk_index) "
                "VALUES (?, ?, ?, ?)",
                (rowid, chunk, file_path, idx),
            )
            count += 1

        # Update mtime tracking
        async with self._db.execute(
            "SELECT value FROM metadata WHERE key=?", (f"mtime:{file_path}",)
        ) as cursor:
            stored = await cursor.fetchone()
        if stored:
            await self._db.execute(
                "UPDATE metadata SET value=? WHERE key=?", (str(mtime), f"mtime:{file_path}")
            )
        else:
            await self._db.execute(
                "INSERT INTO metadata (key, value) VALUES (?, ?)",
                (f"mtime:{file_path}", str(mtime)),
            )

        self._file_hashes[file_path] = mtime
        await self._db.commit()
        return count

    async def index_directory(self, root: Path, exclude_patterns: list[str] | None = None) -> int:
        """Index all text files under root. Returns total chunks indexed."""
        if exclude_patterns is None:
            exclude_patterns = [".git", "__pycache__", "*.pyc", "*.egg-info", ".claude"]

        count = 0
        for file_path in sorted(root.rglob("*")):
            if file_path.is_file():
                # Check exclusions
                rel = str(file_path.relative_to(root))
                skip = False
                for pat in exclude_patterns:
                    if re.match(pat.lstrip("*"), rel) or pat in rel.split("/"):
                        skip = True
                        break
                if skip:
                    continue

                # Skip binary files
                try:
                    raw = file_path.read_bytes()[:8192]
                    if b"\x00" in raw:
                        continue
                except Exception:
                    continue

                chunks = await self.index_file(str(file_path))
                count += chunks
                logger.debug(f"Indexed {file_path}: {chunks} chunks")

        return count

    async def _remove_file(self, file_path: str) -> None:
        """Remove all chunks for a deleted file."""
        await self._db.execute("DELETE FROM chunks WHERE file_path=?", (file_path,))
        await self._db.execute("DELETE FROM chunks_fts WHERE file_path=?", (file_path,))
        await self._db.commit()

    async def search_text(self, query: str, limit: int = 10) -> list[SearchResult]:
        """Full-text search using FTS5. Returns ranked results."""
        async with self._db.execute("""
            SELECT c.file_path, c.chunk_index, c.content, c.simhash_hex,
                   rank as fts_score
            FROM chunks_fts f
            JOIN chunks c ON c.id = f.rowid
            WHERE chunks_fts MATCH ?
            ORDER BY fts_score ASC
            LIMIT ?
        """, (query, limit)) as cursor:
            rows = await cursor.fetchall()
        results = []
        for row in rows:
            # Normalize FTS score (lower is better)
            score = 1.0 / (1.0 + abs(row["fts_score"]))
            simhash_int = int(row["simhash_hex"], 16)
            results.append(SearchResult(
                file_path=row["file_path"],
                chunk_text=row["content"],
                simhash=simhash_int,
                relevance_score=score,
            ))
        return results

    async def search_similar(self, query: str, limit: int = 10) -> list[SearchResult]:
        """Similarity search using simhash hamming distance."""
        query_sh = simhash(query)

        async with self._db.execute("SELECT * FROM chunks") as cursor:
            rows = await cursor.fetchall()

        scored = []
        for row in rows:
            dist = hamming_distance(query_sh, int(row["simhash_hex"], 16))
            # Convert distance to similarity score (max dist = 64)
            similarity = max(0, 1.0 - (dist / 64.0))
            scored.append((similarity, row))

        scored.sort(key=lambda x: -x[0])
        results = []
        for sim, row in scored[:limit]:
            results.append(SearchResult(
                file_path=row["file_path"],
                chunk_text=row["content"],
                simhash=int(row["simhash_hex"], 16),
                relevance_score=sim,
            ))
        return results

    async def hybrid_search(self, query: str, limit: int = 10) -> list[SearchResult]:
        """Combine FTS and similarity search with weighted scoring."""
        fts_results = await self.search_text(query, limit=limit * 2)
        sim_results = await self.search_similar(query, limit=limit * 2)

        # Build lookup by (file_path, chunk_index) to merge duplicates
        scored: dict[tuple[str, int], float] = {}
        for r in fts_results:
            key = (r.file_path, r.chunk_text)
            if key not in scored or r.relevance_score > scored[key][0]:
                scored[key] = (r.relevance_score, r)
        for r in sim_results:
            key = (r.file_path, r.chunk_text)
            existing = scored.get(key, (0, None))
            new_score = max(existing[0], r.relevance_score * 0.5)  # Lower weight for similarity
            if new_score > existing[0]:
                scored[key] = (new_score, r)

        ranked = sorted(scored.values(), key=lambda x: -x[0])
        return [r[1] for r in ranked[:limit]]


# ---------------------------------------------------------------------------
# Plugin hook implementation
# ---------------------------------------------------------------------------

class LocalIndexerPlugin:
    """Plugin that indexes the workspace directory and provides search tool.

    Hook methods:
        on_start(config): Initialize indexer, scan workspace, build index.
        on_stop(): Clean shutdown.
        on_idle(): Background re-indexing of changed files.
        register_tools(tool_registry): Expose search_workspace tool.
    """

    depends_on = set()

    def __init__(self, pm) -> None:
        self.pm = pm
        self.indexer: LocalIndexer | None = None
        self._workspace_root: Path | None = None

    @hookimpl
    async def on_start(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the workspace indexer."""
        if not config:
            return

        tools_config = config.get("tools", {})
        workspace_root = tools_config.get("workspace_path")
        if not workspace_root:
            logger.info("No workspace path configured; skipping local indexer")
            return

        self._workspace_root = Path(workspace_root).resolve()
        if not self._workspace_root.exists():
            logger.warning(f"Workspace directory does not exist: {self._workspace_root}")
            return

        # Initialize SQLite-backed indexer
        db_path = tools_config.get("indexer_db_path", str(self._workspace_root / ".corvidae_index.db"))
        self.indexer = LocalIndexer(db_path)
        await self.indexer.connect()
        self.indexer._workspace_root = self._workspace_root

        # Index all files initially
        count = await self.indexer.index_directory(self._workspace_root)
        logger.info(f"Indexed {count} chunks from workspace: {self._workspace_root}")

    @hookimpl
    async def on_stop(self) -> None:
        """Shut down the indexer cleanly."""
        if self.indexer:
            await self.indexer.close()
            self.indexer = None

    @hookimpl
    async def on_idle(self) -> None:
        """Perform background re-indexing when idle."""
        if not self.indexer or not self._workspace_root:
            return
        try:
            count = await self.indexer.index_directory(self._workspace_root)
            if count > 0:
                logger.info(f"Background index updated {count} chunks")
        except Exception as e:
            logger.debug(f"Indexing error in on_idle: {e}")

    @hookimpl
    def register_tools(self, tool_registry: list) -> None:
        """Register the search_workspace tool."""
        plugin = self  # capture for closure

        async def _search_workspace(query: str, mode: str = "hybrid") -> str:
            """Search local workspace files.

            Args:
                query: Search query (full-text, similarity, or hybrid).
                mode: Search mode — "text" (FTS only), "similar" (similarity), or "hybrid".

            Returns:
                Formatted search results as a string.
            """
            if not plugin.indexer:
                return "Local indexer not initialized."

            if mode == "text":
                results = await plugin.indexer.search_text(query)
            elif mode == "similar":
                results = await plugin.indexer.search_similar(query)
            else:
                results = await plugin.indexer.hybrid_search(query)

            if not results:
                return f"No results for '{query}'"

            lines = [f"Search results for '{query}' ({mode} mode):", ""]
            for i, r in enumerate(results, 1):
                score_pct = int(r.relevance_score * 100)
                lines.append(f"{i}. [{score_pct}%] {r.file_path}")
                # Show first 200 chars of matching content
                preview = r.chunk_text[:200] + ("..." if len(r.chunk_text) > 200 else "")
                lines.append(f"   {preview}")

            return "\n".join(lines)

        _search_workspace.__name__ = "search_workspace"
        tool_registry.append(Tool.from_function(_search_workspace))


# ---------------------------------------------------------------------------
# Entry point registration for entry-point-based plugin discovery
# ---------------------------------------------------------------------------

def get_local_indexer_plugin() -> LocalIndexerPlugin:
    """Return a new LocalIndexerPlugin instance.

    Used by entry-point-based plugin loading (see pyproject.toml [project.entry-points]).
    Note: pm is set to None here because entry-point loading does not provide a pm at
    factory call time. Callers that need pm should instantiate LocalIndexerPlugin directly.
    """
    return LocalIndexerPlugin(pm=None)
