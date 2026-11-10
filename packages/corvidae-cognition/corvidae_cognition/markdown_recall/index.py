"""The markdown-folder recall index (#719).

A SQLite file (sqlite-vec vec0 + FTS5) that mirrors a folder of markdown
files. The files are the source of truth; the index is a cheap-to-rebuild
cache.

``reconcile()`` walks the folder, hashes what changed, re-chunks and re-embeds
the new/edited files, and replaces each file's index rows in one transaction.
It is idempotent and safe: an unreadable root, or a walk that finds zero files
while the manifest holds many (an unmounted folder), aborts and keeps the old
index rather than looking like mass deletion.

``search()`` runs vector KNN over ``chunks_vec`` and a keyword search over
``chunks_fts``, fused by reciprocal rank (RRF, C=60) exactly as
``MemoryPlugin`` does, and degrades to FTS5-only when the vec extension cannot
load or the embedding stage is unavailable. Text is canonical; vectors are a
cache.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import aiosqlite

from corvidae_cognition.markdown_recall.chunking import chunk_page

logger = logging.getLogger("corvidae.markdown_recall.index")

# Reciprocal rank fusion constant: a candidate earns 1 / (RRF_C + rank) from
# each ranked list it is in. 60 is the value from the original RRF paper, the
# same as memory.py.
RRF_C = 60
# How many rows each side overfetches before fusion, so the fused top-k is
# not starved when the two rankings disagree.
OVERFETCH = 3

# The chunker version stamped into the manifest meta: a change here (a new
# vendored chunker) rebuilds the whole index, like an encoder change.
CHUNKER_VERSION = "otterwiki-1"

CHUNKS_DDL = """
CREATE TABLE IF NOT EXISTS chunks (
    rowid INTEGER PRIMARY KEY,
    path TEXT NOT NULL,          -- path relative to the folder root
    section_path TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content_hash TEXT NOT NULL,  -- sha256 of the source file
    text TEXT NOT NULL
)
"""

CHUNKS_FTS_DDL = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, content='chunks', content_rowid='rowid'
)
"""

# External-content FTS5 sync triggers. A chunk row is replaced (not updated)
# on a reconcile, so delete + insert, no update trigger.
CHUNKS_FTS_INSERT_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
END
"""

CHUNKS_FTS_DELETE_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
END
"""

MANIFEST_DDL = """
CREATE TABLE IF NOT EXISTS manifest (
    path TEXT PRIMARY KEY,
    size INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    chunk_count INTEGER NOT NULL
)
"""

EMBEDDING_META_DDL = """
CREATE TABLE IF NOT EXISTS embedding_meta (
    encoder TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    document_prefix TEXT NOT NULL DEFAULT '',
    query_prefix TEXT NOT NULL DEFAULT '',
    chunker_version TEXT NOT NULL DEFAULT ''
)
"""


@dataclass(frozen=True)
class Hit:
    """One ranked result: a chunk's identity, text, and fused rank score."""

    rowid: int
    path: str
    section_path: str
    chunk_index: int
    text: str
    score: float


@dataclass(frozen=True)
class _EncoderIdentity:
    """The embedding profile's identity; a change rebuilds the index."""

    model: str
    dimensions: int
    document_prefix: str
    query_prefix: str


class MarkdownIndex:
    """The SQLite index over a folder of markdown files.

    ``root`` is the folder to mirror; ``db_path`` is the index file;
    ``get_client`` returns the embedding client (the ``llm`` stanza's
    ``embedding`` stage); ``get_identity`` returns the encoder identity (or
    None when no embedding profile is configured — the index is then
    FTS5-only).
    """

    def __init__(
        self,
        root: Path,
        db_path: Path,
        get_client: Callable[[], Any],
        get_identity: Callable[[], _EncoderIdentity | None],
    ) -> None:
        self._root = Path(root)
        self._db_path = Path(db_path)
        self._get_client = get_client
        self._get_identity = get_identity
        self._db: aiosqlite.Connection | None = None
        self._schema_ready = False
        self._vec_available = False
        self._encoder_mismatch = False
        # A folder-level lock: the index is shared across channels, so
        # reconcile and search serialize on it (not the mixin's per-channel
        # single-flight).
        self._lock = asyncio.Lock()

    # -- connection + schema ------------------------------------------------

    async def _connect(self) -> aiosqlite.Connection | None:
        """Open the index connection (WAL) or None when unavailable.

        Re-opens a stale closed connection: in the daemon the connection lives
        for the process (one loop, never closed), but a closed handle must not
        poison every later query — the schema is in the file, so re-opening is
        cheap.
        """
        if self._db is not None:
            try:
                await self._db.execute("SELECT 1")
                return self._db
            except Exception:
                self._db = None  # stale/closed: re-open below
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            db = await aiosqlite.connect(self._db_path)
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")
            self._db = db
            return db
        except Exception:
            logger.warning(
                "markdown recall: could not open the index at %s",
                self._db_path, exc_info=True,
            )
            return None

    async def _ensure_schema(self) -> aiosqlite.Connection | None:
        """Create the index tables; the vector surface only when it can exist.

        Safe to call repeatedly — a no-op once the schema landed. Returns the
        connection, or None when the index file is unavailable.
        """
        db = await self._connect()
        if db is None:
            return None
        if self._schema_ready:
            return db

        await db.execute(CHUNKS_DDL)
        await db.execute(CHUNKS_FTS_DDL)
        await db.execute(CHUNKS_FTS_INSERT_TRIGGER)
        await db.execute(CHUNKS_FTS_DELETE_TRIGGER)
        await db.execute(MANIFEST_DDL)
        await db.execute(EMBEDDING_META_DDL)

        # The vector surface exists only with an embedding profile configured
        # and a loadable sqlite-vec extension; anything else degrades to
        # FTS5-only. Text is canonical; vectors are a cache.
        identity = self._get_identity()
        if identity is not None and await self._load_vec_extension(db):
            await self._check_embedding_meta(db, identity)
            if not self._encoder_mismatch:
                await db.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0("
                    f"rowid INTEGER PRIMARY KEY, embedding float[{identity.dimensions}])"
                )
                self._vec_available = True

        self._schema_ready = True
        return db

    async def _load_vec_extension(self, db: aiosqlite.Connection) -> bool:
        """Load sqlite-vec into the connection; False (never raises) on failure."""
        try:
            await db.enable_load_extension(True)
        except AttributeError:
            return False
        try:
            import sqlite_vec
            await db.load_extension(sqlite_vec.loadable_path())
            return True
        except Exception:
            logger.warning("markdown recall: sqlite-vec failed to load", exc_info=True)
            return False
        finally:
            try:
                await db.enable_load_extension(False)
            except Exception:
                logger.warning("markdown recall: could not re-disable extension loading")

    async def _check_embedding_meta(
        self, db: aiosqlite.Connection, identity: _EncoderIdentity
    ) -> None:
        """Set ``_encoder_mismatch`` when the stored identity differs from now.

        A mismatch means the vectors were built by a different encoder (or a
        different chunker); reconcile() rebuilds everything in that case.
        """
        async with db.execute(
            "SELECT encoder, dimensions, document_prefix, query_prefix, "
            "chunker_version FROM embedding_meta LIMIT 1"
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            await db.execute(
                "INSERT INTO embedding_meta (encoder, dimensions, document_prefix, "
                "query_prefix, chunker_version) VALUES (?, ?, ?, ?, ?)",
                (
                    identity.model, identity.dimensions, identity.document_prefix,
                    identity.query_prefix, CHUNKER_VERSION,
                ),
            )
            self._encoder_mismatch = False
            return
        stored = (
            row["encoder"], row["dimensions"], row["document_prefix"],
            row["query_prefix"], row["chunker_version"],
        )
        current = (
            identity.model, identity.dimensions, identity.document_prefix,
            identity.query_prefix, CHUNKER_VERSION,
        )
        self._encoder_mismatch = stored != current
        if self._encoder_mismatch:
            logger.warning(
                "markdown recall: encoder/chunker identity changed "
                "(%r -> %r); the index will be rebuilt", stored, current,
            )

    # -- reconcile ----------------------------------------------------------

    def _walk(self) -> dict[str, tuple[int, int]] | None:
        """Map relative path -> (size, mtime_ns) for every .md under the root.

        Returns None when the root is unreadable (the caller aborts). Never
        follows a symlink out of the folder.
        """
        if not self._root.is_dir():
            return None
        files: dict[str, tuple[int, int]] = {}
        try:
            for p in self._root.rglob("*.md"):
                if p.is_symlink():
                    continue  # do not follow symlinks out of the folder
                if not p.is_file():
                    continue
                try:
                    st = p.stat()
                except OSError:
                    continue
                files[p.relative_to(self._root).as_posix()] = (st.st_size, st.st_mtime_ns)
        except OSError:
            logger.warning("markdown recall: could not walk %s", self._root, exc_info=True)
            return None
        return files

    @staticmethod
    def _hash_bytes(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    async def _embed(
        self, texts: list[str], kind: str
    ) -> list[list[float] | None]:
        """Embed ``texts`` via the embedding stage; None on any failure."""
        if not texts:
            return []
        try:
            client = self._get_client()
            vectors = await client.embed(texts, kind=kind)
            return list(vectors)
        except Exception:
            logger.warning(
                "markdown recall: embedding failed (%d texts, kind=%s); "
                "the affected chunks stay un-embedded (FTS5-only for them)",
                len(texts), kind, exc_info=True,
            )
            return [None] * len(texts)

    async def reconcile(self) -> None:
        """Rebuild the index to match the folder. Idempotent and safe.

        The walk is the truth. Unchanged files are skipped; new/edited files
        are re-chunked, embedded outside the transaction, and their rows
        replaced in one transaction; missing files are deleted. An unreadable
        root, or a zero-file walk while the manifest holds many rows, aborts
        and keeps the old index (an unmounted folder must not look like mass
        deletion).
        """
        async with self._lock:
            db = await self._ensure_schema()
            if db is None:
                return
            files = self._walk()
            if files is None:
                logger.warning(
                    "markdown recall: root %s is unreadable; kept the old index",
                    self._root,
                )
                return
            async with db.execute("SELECT count(*) AS n FROM manifest") as cursor:
                row = await cursor.fetchone()
            manifest_count = row["n"] if row else 0
            # An unmounted folder reads as zero files; that is not mass deletion.
            if not files and manifest_count > 0:
                logger.warning(
                    "markdown recall: walk of %s found no files but the manifest "
                    "holds %d; kept the old index (unmounted folder?)",
                    self._root, manifest_count,
                )
                return

            # An encoder/chunker change rebuilds everything.
            if self._encoder_mismatch:
                await self._rebuild_all(db, files)
                return

            # Current manifest: path -> (size, mtime_ns, content_hash).
            current: dict[str, tuple[int, int, str]] = {}
            async with db.execute(
                "SELECT path, size, mtime_ns, content_hash FROM manifest"
            ) as cursor:
                async for r in cursor:
                    current[r["path"]] = (r["size"], r["mtime_ns"], r["content_hash"])

            for path, (size, mtime_ns) in files.items():
                prev = current.get(path)
                if prev is not None and prev[0] == size and prev[1] == mtime_ns:
                    continue  # unchanged: the stat check is a shortcut
                await self._reconcile_file(db, path, size, mtime_ns, prev)

            # Missing files: in the manifest, not in the walk.
            for path in set(current) - set(files):
                await self._delete_file(db, path)
            await db.commit()

    async def _rebuild_all(
        self, db: aiosqlite.Connection, files: dict[str, tuple[int, int]]
    ) -> None:
        """Drop all rows and reindex every file (encoder/chunker changed)."""
        logger.info("markdown recall: rebuilding the whole index (identity changed)")
        async with db.execute("DELETE FROM chunks") as _c:
            pass
        async with db.execute("DELETE FROM manifest") as _c:
            pass
        if self._vec_available:
            async with db.execute("DELETE FROM chunks_vec") as _c:
                pass
        # Re-stamp the identity so future reconciles see a match.
        identity = self._get_identity()
        if identity is not None:
            await db.execute(
                "UPDATE embedding_meta SET encoder=?, dimensions=?, "
                "document_prefix=?, query_prefix=?, chunker_version=?",
                (
                    identity.model, identity.dimensions, identity.document_prefix,
                    identity.query_prefix, CHUNKER_VERSION,
                ),
            )
            self._encoder_mismatch = False
        for path, (size, mtime_ns) in files.items():
            await self._reconcile_file(db, path, size, mtime_ns, None)
        await db.commit()

    async def _reconcile_file(
        self,
        db: aiosqlite.Connection,
        path: str,
        size: int,
        mtime_ns: int,
        prev: tuple[int, int, str] | None,
    ) -> None:
        """Index one file: hash, and re-chunk only when the hash changed."""
        full = self._root / path
        try:
            data = full.read_bytes()
        except OSError:
            logger.warning("markdown recall: could not read %s", full, exc_info=True)
            return
        # Re-stat after reading: skip a file written mid-read.
        try:
            st = full.stat()
        except OSError:
            return
        if st.st_mtime_ns != mtime_ns and st.st_size != size:
            # Changed while we read it; the next reconcile picks up the new state.
            return
        content_hash = self._hash_bytes(data)
        # Hash unchanged (an mtime-only touch): update the mtime, keep the rows.
        if prev is not None and prev[2] == content_hash:
            await db.execute(
                "UPDATE manifest SET mtime_ns=? WHERE path=?", (mtime_ns, path)
            )
            return
        text = data.decode("utf-8", errors="replace")
        chunks = chunk_page(path, text)
        vectors = await self._embed([c["text"] for c in chunks], "document")
        await self._replace_file_chunks(db, path, mtime_ns, content_hash, chunks, vectors)

    async def _replace_file_chunks(
        self,
        db: aiosqlite.Connection,
        path: str,
        mtime_ns: int,
        content_hash: str,
        chunks: list[dict],
        vectors: list[list[float] | None],
    ) -> None:
        """Replace one file's chunk + vec + manifest rows in one transaction."""
        try:
            if self._vec_available:
                import sqlite_vec
                async with db.execute(
                    "SELECT rowid FROM chunks WHERE path=?", (path,)
                ) as c:
                    old_rowids = [r[0] for r in await c.fetchall()]
                for rid in old_rowids:
                    await db.execute("DELETE FROM chunks_vec WHERE rowid=?", (rid,))
            await db.execute("DELETE FROM chunks WHERE path=?", (path,))
            rows: list[tuple] = []
            vec_rows: list[tuple] = []
            for i, c in enumerate(chunks):
                cur = await db.execute(
                    "INSERT INTO chunks (path, section_path, chunk_index, "
                    "content_hash, text) VALUES (?, ?, ?, ?, ?)",
                    (
                        path, c["metadata"]["section_path"],
                        c["metadata"]["chunk_index"], content_hash, c["text"],
                    ),
                )
                rowid = cur.lastrowid
                rows.append(rowid)
                v = vectors[i] if i < len(vectors) else None
                if self._vec_available and v is not None:
                    vec_rows.append((rowid, sqlite_vec.serialize_float32(v)))
            if vec_rows:
                await db.executemany(
                    "INSERT INTO chunks_vec (rowid, embedding) VALUES (?, ?)",
                    vec_rows,
                )
            await db.execute(
                "INSERT OR REPLACE INTO manifest (path, size, mtime_ns, "
                "content_hash, chunk_count) VALUES (?, 0, ?, ?, ?)",
                (path, mtime_ns, content_hash, len(chunks)),
            )
            await db.commit()
        except Exception:
            await db.rollback()
            raise

    async def _delete_file(self, db: aiosqlite.Connection, path: str) -> None:
        """Delete one file's chunk + vec + manifest rows."""
        if self._vec_available:
            async with db.execute(
                "SELECT rowid FROM chunks WHERE path=?", (path,)
            ) as c:
                rowids = [r[0] for r in await c.fetchall()]
            for rid in rowids:
                await db.execute("DELETE FROM chunks_vec WHERE rowid=?", (rid,))
        await db.execute("DELETE FROM chunks WHERE path=?", (path,))
        await db.execute("DELETE FROM manifest WHERE path=?", (path,))

    # -- search -------------------------------------------------------------

    async def search(self, query: str, k: int = 5) -> list[Hit]:
        """Return the top-``k`` chunks for ``query``, in fused rank order.

        Vector KNN + FTS5 fused by RRF when the vec surface is available and
        the query embeds; FTS5-only otherwise. An empty/absent index yields [].
        """
        async with self._lock:
            db = await self._ensure_schema()
            if db is None:
                return []
            vec_rank: dict[int, int] = {}
            fts_rank: dict[int, int] = {}
            fts_ids: list[int] = []
            # FTS side (always attempted; it is the degrade path).
            try:
                fts_ids = await self._fts_ids(db, query, k * OVERFETCH)
                fts_rank = {rid: i for i, rid in enumerate(fts_ids, start=1)}
            except Exception:
                logger.warning("markdown recall: FTS5 search failed", exc_info=True)
            # Vector side (only when the surface exists and the query embeds).
            if self._vec_available:
                qv = await self._embed([query], "query")
                if qv and qv[0] is not None:
                    try:
                        vec_rank = await self._vec_ids(db, qv[0], k * OVERFETCH)
                    except Exception:
                        logger.warning(
                            "markdown recall: vector search failed; "
                            "ranking the FTS side alone", exc_info=True,
                        )
            return await self._fuse(db, vec_rank, fts_rank, k)

    async def _fts_ids(
        self, db: aiosqlite.Connection, query: str, limit: int
    ) -> list[int]:
        """Rowids matching ``query`` in bm25 order (a sanitized MATCH)."""
        match = _fts_match(query)
        if not match:
            return []
        async with db.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? "
            "ORDER BY bm25(chunks_fts) LIMIT ?",
            (match, limit),
        ) as cursor:
            return [row[0] for row in await cursor.fetchall()]

    async def _vec_ids(
        self, db: aiosqlite.Connection, query_vector: list[float], limit: int
    ) -> dict[int, int]:
        """rowid -> 1-based KNN rank for the query vector."""
        import sqlite_vec

        serialized = sqlite_vec.serialize_float32(query_vector)
        async with db.execute(
            "SELECT rowid FROM chunks_vec "
            "ORDER BY vec_distance_cosine(embedding, ?) LIMIT ?",
            (serialized, limit),
        ) as cursor:
            rows = await cursor.fetchall()
        return {row[0]: i for i, row in enumerate(rows, start=1)}

    async def _fuse(
        self,
        db: aiosqlite.Connection,
        vec_rank: dict[int, int],
        fts_rank: dict[int, int],
        k: int,
    ) -> list[Hit]:
        """Fused RRF over the two rank maps, joined to chunk rows."""
        pool = set(vec_rank) | set(fts_rank)
        if not pool:
            return []
        scored = []
        for rid in pool:
            score = 0.0
            if rid in vec_rank:
                score += 1.0 / (RRF_C + vec_rank[rid])
            if rid in fts_rank:
                score += 1.0 / (RRF_C + fts_rank[rid])
            scored.append((score, rid))
        scored.sort(key=lambda t: (-t[0], t[1]))
        top = [rid for _score, rid in scored[:k]]
        if not top:
            return []
        marks = ",".join("?" for _ in top)
        async with db.execute(
            f"SELECT rowid, path, section_path, chunk_index, text FROM chunks "
            f"WHERE rowid IN ({marks})",
            top,
        ) as cursor:
            rows = {row["rowid"]: row for row in await cursor.fetchall()}
        hits = []
        for score, rid in scored[:k]:
            row = rows.get(rid)
            if row is None:
                continue
            hits.append(
                Hit(
                    rowid=rid, path=row["path"], section_path=row["section_path"],
                    chunk_index=row["chunk_index"], text=row["text"], score=score,
                )
            )
        return hits

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None
            self._schema_ready = False


def _fts_match(query: str) -> str:
    """Sanitize a free-text query into an FTS5 MATCH expression.

    Each token becomes a prefix term (``tok*``) so partial words match, and
    the tokens are OR-joined so one good term carries the query when another
    is a stopword. Returns "" when there is nothing to match.
    """
    tokens = [t for t in query.lower().split() if t]
    if not tokens:
        return ""
    return " OR ".join(f'"{t}"*' for t in tokens)
