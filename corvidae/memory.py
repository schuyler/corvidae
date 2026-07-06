"""MemoryPlugin — autobiographical memory for the Corvidae agent daemon.

Compaction becomes memory formation: dialog leaving the active window is
consolidated into first-person memory records (SQLite + sqlite-vec + FTS5),
and inbound messages trigger retrieval whose results enter the window
through the context-admission funnel (bootstrap-mapping §3.1).

This module owns the memory schema:
  - ``memory``: one row per consolidated first-person record, linked back
    to the raw dialog by a ``message_log`` id range. Rows are never
    deleted — demotion/redaction (Phase 1b) mutate columns.
  - ``consolidation_watermark``: per-channel last-consolidated
    ``message_log`` id; what makes the compaction/idle trigger overlap safe.
  - ``embedding_meta``: the encoder identity the vector cache was built
    with. Summaries are canonical text; vectors are a rebuildable cache.
  - ``memory_fts``: external-content FTS5 over summaries with content-sync
    triggers (§4.11) — the retrieval fallback when the encoder is down.
  - ``memory_vec``: sqlite-vec vec0 exact-KNN table, created only when the
    extension loads (§4.12). Some Python builds lack
    ``enable_load_extension``; the plugin then degrades to FTS5-only
    retrieval with one clear warning.

Config:
    memory:
      idle_consolidate_after: 1800   # seconds of channel inactivity before
                                     # idle-triggered consolidation
      half_life_days: 30             # retrieval recency half-life
      retrieval:
        k: 8                         # KNN candidate count
        bands:
          strong: 0.75               # score >= strong → asserted confidently
          moderate: 0.60             # score >= moderate → hedged
      channel_groups: {}             # {group_name: [channel ids]} shared memory
"""

from __future__ import annotations

import logging

import aiosqlite

from corvidae.hooks import CorvidaePlugin, get_dependency, hookimpl

logger = logging.getLogger("corvidae.memory")

# ---------------------------------------------------------------------------
# Schema DDL (bootstrap-mapping §4.11–4.12)
# ---------------------------------------------------------------------------

MEMORY_DDL = """
CREATE TABLE IF NOT EXISTS memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id TEXT NOT NULL,
    created_at REAL NOT NULL,
    summary TEXT NOT NULL,              -- first-person, epistemic framing preserved
    importance REAL NOT NULL,           -- the prior (WP1a.6); updated by use in Phase 1b
    valence REAL,                       -- NULL until Phase 2 appraisal
    topic_tags TEXT,                    -- JSON array of strings
    participants TEXT,                  -- JSON array of sender strings
    msg_id_start INTEGER NOT NULL,      -- message_log id range (raw-dialog link)
    msg_id_end INTEGER NOT NULL,
    retrieval_count INTEGER NOT NULL DEFAULT 0,
    last_retrieved_at REAL,
    indexed INTEGER NOT NULL DEFAULT 1, -- 0 = demoted out of retrieval (Phase 1b)
    superseded_by INTEGER,              -- near-dup merge target (Phase 1b)
    redacted INTEGER NOT NULL DEFAULT 0,-- redact tombstone flag (Phase 1b)
    embedded INTEGER NOT NULL DEFAULT 0 -- 0 = embedding pending/failed (backfillable)
)
"""

MEMORY_INDEX_DDL = (
    "CREATE INDEX IF NOT EXISTS idx_memory_channel ON memory (channel_id, created_at)"
)

WATERMARK_DDL = """
CREATE TABLE IF NOT EXISTS consolidation_watermark (
    channel_id TEXT PRIMARY KEY,
    last_message_id INTEGER NOT NULL
)
"""

EMBEDDING_META_DDL = """
CREATE TABLE IF NOT EXISTS embedding_meta (
    encoder TEXT NOT NULL,
    dimensions INTEGER NOT NULL
)
"""

MEMORY_FTS_DDL = """
CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    summary, content='memory', content_rowid='id'
)
"""

# Content-sync triggers for the external-content FTS5 table. No delete
# trigger: memory rows are never deleted (demotion/redaction mutate columns).
MEMORY_FTS_INSERT_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS memory_ai AFTER INSERT ON memory BEGIN
    INSERT INTO memory_fts(rowid, summary) VALUES (new.id, new.summary);
END
"""

MEMORY_FTS_UPDATE_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS memory_au AFTER UPDATE OF summary ON memory BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, summary) VALUES ('delete', old.id, old.summary);
    INSERT INTO memory_fts(rowid, summary) VALUES (new.id, new.summary);
END
"""


class MemoryPlugin(CorvidaePlugin):
    """Plugin that owns the memory store schema (consolidation and retrieval
    are layered on in later work packages of Phase 1a).

    Attributes:
        _vec_available: True when the sqlite-vec extension loaded and the
            memory_vec table exists. False degrades retrieval to FTS5.
        _encoder_mismatch: True when embedding_meta records a different
            encoder/dimensions than the current config — embedding writes
            and vector retrieval are disabled rather than silently mixing
            encoders.
        _schema_ready: schema creation ran to completion. on_start ordering
            across plugins is LIFO, so the persistence DB may not exist at
            our on_start; schema creation retries lazily on first use.
    """

    depends_on = frozenset({"persistence", "llm"})

    def __init__(self, pm=None) -> None:
        if pm is not None:
            self.pm = pm
        self._schema_ready = False
        self._vec_available = False
        self._encoder_mismatch = False
        self._embedding_cfg: dict | None = None
        self._memory_cfg: dict = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        """Read memory.* and llm.embedding config."""
        await super().on_init(pm, config)
        self._memory_cfg = config.get("memory", {}) or {}
        self._embedding_cfg = (config.get("llm", {}) or {}).get("embedding")

    @hookimpl
    async def on_start(self, config: dict) -> None:
        """Create the schema on the persistence connection.

        Deferred to first use when the persistence DB has not opened yet
        (hook-ordering dependent, same pattern as UsageLogPlugin).
        """
        try:
            await self._ensure_schema()
        except Exception:
            logger.warning("memory schema creation deferred", exc_info=True)

    # ------------------------------------------------------------------
    # Schema plumbing
    # ------------------------------------------------------------------

    def _resolve_db(self) -> aiosqlite.Connection | None:
        """Return the persistence plugin's DB connection, or None if not open yet."""
        from corvidae.persistence import PersistencePlugin
        persistence = get_dependency(self.pm, "persistence", PersistencePlugin)
        return persistence.db

    def _embedding_dimensions(self) -> int | None:
        """The configured embedding dimension, or None when unconfigured."""
        if self._embedding_cfg is None:
            return None
        dimensions = self._embedding_cfg.get("dimensions")
        return int(dimensions) if dimensions is not None else None

    async def _load_vec_extension(self, db: aiosqlite.Connection) -> bool:
        """Load the sqlite-vec extension into the connection.

        Returns False (never raises) when the Python build lacks
        ``enable_load_extension`` or the extension fails to load — the
        caller degrades to FTS5-only retrieval (§4.12). Kept as a method so
        tests can monkeypatch the failure mode.
        """
        try:
            await db.enable_load_extension(True)
        except AttributeError:
            # Some Python builds compile out extension loading entirely.
            return False
        try:
            import sqlite_vec
            await db.load_extension(sqlite_vec.loadable_path())
            return True
        except Exception:
            logger.warning("sqlite-vec extension failed to load", exc_info=True)
            return False
        finally:
            try:
                await db.enable_load_extension(False)
            except Exception:
                logger.warning(
                    "could not re-disable extension loading", exc_info=True
                )

    async def _ensure_schema(self) -> aiosqlite.Connection | None:
        """Create all memory tables; returns the DB or None when unavailable.

        Safe to call repeatedly — a no-op once the schema landed.
        """
        db = self._resolve_db()
        if db is None:
            return None
        if self._schema_ready:
            return db

        # Core tables and the FTS surface land unconditionally.
        await db.execute(MEMORY_DDL)
        await db.execute(MEMORY_INDEX_DDL)
        await db.execute(WATERMARK_DDL)
        await db.execute(EMBEDDING_META_DDL)
        await db.execute(MEMORY_FTS_DDL)
        await db.execute(MEMORY_FTS_INSERT_TRIGGER)
        await db.execute(MEMORY_FTS_UPDATE_TRIGGER)

        # The vector surface exists only with an embedding role configured
        # and a loadable sqlite-vec extension; anything else degrades to
        # FTS5-only retrieval. Text is canonical; vectors are a cache.
        dimensions = self._embedding_dimensions()
        encoder = (
            self._embedding_cfg.get("model") if self._embedding_cfg else None
        )
        if dimensions is not None and encoder is not None:
            await self._check_embedding_meta(db, encoder, dimensions)
            if not self._encoder_mismatch:
                if await self._load_vec_extension(db):
                    await db.execute(
                        "CREATE VIRTUAL TABLE IF NOT EXISTS memory_vec USING vec0("
                        "memory_id INTEGER PRIMARY KEY, "
                        f"embedding FLOAT[{dimensions}] distance_metric=cosine)"
                    )
                    self._vec_available = True
                else:
                    logger.warning(
                        "sqlite-vec unavailable: vector retrieval disabled, "
                        "degrading to FTS5"
                    )

        await db.commit()
        self._schema_ready = True
        return db

    async def _check_embedding_meta(
        self, db: aiosqlite.Connection, encoder: str, dimensions: int
    ) -> None:
        """Record or verify the encoder identity the vector cache was built with.

        A stored encoder differing from config is an operator ERROR — mixing
        encoders in one index silently corrupts similarity. The rebuild flow
        is Phase 1b territory; here we refuse to mix and disable embedding.
        """
        async with db.execute(
            "SELECT encoder, dimensions FROM embedding_meta LIMIT 1"
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            await db.execute(
                "INSERT INTO embedding_meta (encoder, dimensions) VALUES (?, ?)",
                (encoder, dimensions),
            )
            return
        if (row[0], row[1]) != (encoder, dimensions):
            self._encoder_mismatch = True
            logger.error(
                "embedding_meta records encoder %s/%dd but config wants %s/%dd; "
                "embeddings disabled — re-embed the memory store to switch "
                "encoders (rebuild flow lands in Phase 1b)",
                row[0], row[1], encoder, dimensions,
            )
