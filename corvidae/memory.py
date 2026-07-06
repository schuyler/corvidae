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

import asyncio
import json
import logging
import time
from collections.abc import Callable
from typing import Protocol

import aiosqlite

from corvidae.attribution import reset_attribution, set_attribution
from corvidae.hooks import CorvidaePlugin, get_dependency, hookimpl

logger = logging.getLogger("corvidae.memory")

# Seconds a channel must be inactive before the idle trigger consolidates
# its un-consolidated tail.
DEFAULT_IDLE_CONSOLIDATE_AFTER = 1800.0

# Default consolidation prompt. prompts/memory_consolidation.md is the
# documented copy of this text; memory.consolidation_prompt in config
# overrides it with a literal string.
DEFAULT_CONSOLIDATION_PROMPT = (
    "You are consolidating a conversation segment into the agent's "
    "long-term autobiographical memory. Write a FIRST-PERSON summary from "
    "the agent's point of view (the assistant is \"I\").\n\n"
    "Preserve epistemic framing: distinguish what I was told from what I "
    "inferred or speculated (\"Schuyler told me...\", \"I speculated "
    "that...\", \"I suggested...\"). Keep concrete details that would "
    "matter later: names, decisions, commitments, corrections, and "
    "outcomes. Omit filler.\n\n"
    "Respond with a single JSON object, nothing else:\n"
    "{\n"
    '  "summary": "<first-person summary, a few sentences>",\n'
    '  "topic_tags": ["<short topic tag>", ...],\n'
    '  "participants": ["<sender name>", ...]\n'
    "}"
)

# Rubric for the default importance prior (Persyn-style cheap-model rating).
RUBRIC_PROMPT = (
    "Rate the lasting importance of remembering this conversation segment "
    "for a personal agent, from 0.0 (trivial small talk, transient chatter) "
    "to 1.0 (identity-level facts, standing commitments, significant "
    "decisions, or emotionally weighty events).\n\n"
    'Respond with a single JSON object, nothing else: {"importance": <float>}'
)


def _parse_json_block(text: str) -> dict:
    """Extract the first JSON object from LLM output.

    Tolerates surrounding prose and markdown code fences. Raises ValueError
    when no parseable object is present — callers own their degradation.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"no JSON object in LLM output: {text[:200]!r}")
    return json.loads(text[start:end + 1])


def _dialog_transcript(messages: list[dict]) -> str:
    """Render dialog messages as 'role: content' lines for a prompt."""
    return "\n".join(
        f"{m.get('role', '?')}: {m.get('content', '')}" for m in messages
    )


class ImportancePrior(Protocol):
    """Pluggable supplier of the consolidation-time importance prior (§3.1).

    Held as ``MemoryPlugin.importance_prior`` and assignable by later
    phases: the Phase 2 appraisal replaces the default rubric; a Phase 6
    toggle adds a surprise term.
    """

    async def score(self, messages: list[dict]) -> float: ...   # 0.0–1.0


class RubricPrior:
    """Default importance prior: cheap-model rubric rating (§3.1).

    Returns 0.5 on any failure (logged) — a scoring outage must never
    block memory formation.
    """

    def __init__(self, get_client: Callable[[], object]) -> None:
        # A callable, not a client, so the current background client is
        # resolved at call time (clients can be swapped on config reload).
        self._get_client = get_client

    async def score(self, messages: list[dict]) -> float:
        try:
            client = self._get_client()
            response = await client.chat([
                {"role": "system", "content": RUBRIC_PROMPT},
                {"role": "user", "content": _dialog_transcript(messages)},
            ])
            text = response["choices"][0]["message"]["content"]
            importance = float(_parse_json_block(text)["importance"])
            return max(0.0, min(1.0, importance))
        except Exception:
            logger.warning(
                "importance rubric failed; defaulting to 0.5", exc_info=True
            )
            return 0.5

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
        self._idle_consolidate_after: float = DEFAULT_IDLE_CONSOLIDATE_AFTER
        self._consolidation_prompt: str = DEFAULT_CONSOLIDATION_PROMPT
        # Tracked background tasks (consolidation) — plugin-owned
        # asyncio.create_task, never the TaskQueue: every TaskQueue
        # completion triggers a full main-model turn (§2.3, trap #5).
        self._tasks: set[asyncio.Task] = set()
        # Serializes the check-and-insert commit section of consolidation
        # so the watermark compare-and-set is race-free on the shared
        # persistence connection.
        self._db_lock = asyncio.Lock()
        # Pluggable importance prior; replaced by later phases (§3.1).
        self.importance_prior: ImportancePrior = RubricPrior(
            lambda: self._llm().get_client("background")
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        """Read memory.* and llm.embedding config."""
        await super().on_init(pm, config)
        self._memory_cfg = config.get("memory", {}) or {}
        self._embedding_cfg = (config.get("llm", {}) or {}).get("embedding")
        self._idle_consolidate_after = self._memory_cfg.get(
            "idle_consolidate_after", DEFAULT_IDLE_CONSOLIDATE_AFTER
        )
        self._consolidation_prompt = self._memory_cfg.get(
            "consolidation_prompt", DEFAULT_CONSOLIDATION_PROMPT
        )

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

    @hookimpl
    async def on_stop(self) -> None:
        """Cancel tracked background tasks (consolidation in flight)."""
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)
        self._tasks.clear()

    # ------------------------------------------------------------------
    # Consolidation triggers (bootstrap-mapping §3.1)
    # ------------------------------------------------------------------

    @hookimpl
    async def on_compaction(
        self, channel, summary_msg: dict, retain_count: int, compacted_ids: list[int]
    ) -> None:
        """Trigger A: dialog leaving the active window enters long-term memory.

        Depends only on the hook payload — never on the summary row being
        in the DB (ordering against PersistencePlugin on this broadcast is
        just registration order; trap #4).

        Deliberate divergence note (OPT-1a-1): bootstrap-mapping §3.1
        prefers a single summarization call with two outputs (window
        summary + memory record). This phase ships consolidation as its own
        cheap-model call for plugin decoupling; Phase 0 metering makes the
        double-pay measurable, and merging the calls is a named follow-up
        once the cost is known.
        """
        if not compacted_ids:
            return
        self._spawn(self._consolidate_range(channel.id, max(compacted_ids)))

    @hookimpl
    async def on_idle(self) -> None:
        """Trigger B: consolidate the un-consolidated tail of inactive channels.

        Inactivity comes from the live Channel's last_active when the
        channel is registered, else from the newest message_log timestamp
        (covers backlog on channels this process has not seen traffic on).
        """
        try:
            db = await self._ensure_schema()
            if db is None:
                return
            now = time.time()
            registry = self.pm.get_plugin("registry")
            async with db.execute(
                "SELECT channel_id, MAX(id), MAX(timestamp) FROM message_log "
                "GROUP BY channel_id"
            ) as cursor:
                rows = await cursor.fetchall()
            for channel_id, max_id, max_ts in rows:
                watermark = await self._get_watermark(db, channel_id)
                if max_id <= watermark:
                    continue
                last_active = max_ts
                channel = registry.get(channel_id) if registry is not None else None
                if channel is not None:
                    last_active = channel.last_active
                if now - last_active < self._idle_consolidate_after:
                    continue
                self._spawn(self._consolidate_range(channel_id, max_id))
        except Exception:
            logger.warning("idle consolidation scan failed", exc_info=True)

    # ------------------------------------------------------------------
    # Background task tracking (§2.3 — silent, never the TaskQueue)
    # ------------------------------------------------------------------

    def _spawn(self, coro) -> asyncio.Task:
        """Run a coroutine as a tracked background task with full logging."""
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._on_task_done)
        return task

    def _on_task_done(self, task: asyncio.Task) -> None:
        """Drop the handle and surface any unhandled exception."""
        self._tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.warning("memory background task failed", exc_info=exc)

    async def wait_for_background_tasks(self) -> None:
        """Await all tracked tasks — a test/live-check convenience."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    # ------------------------------------------------------------------
    # The consolidation task (single code path for both triggers)
    # ------------------------------------------------------------------

    async def _consolidate_range(self, channel_id: str, range_end: int) -> None:
        """Consolidate message_log rows (watermark, range_end] into one record.

        Overlap safety (trap #6): both triggers WILL race to the same
        range. The watermark is read optimistically before the LLM work,
        then re-checked under the DB lock in the same atomic section as the
        insert + watermark advance; any concurrent advance discards this
        task's record.
        """
        attribution_token = set_attribution(
            stage="consolidation", channel_id=channel_id
        )
        try:
            db = await self._ensure_schema()
            if db is None:
                return
            watermark = await self._get_watermark(db, channel_id)
            if range_end <= watermark:
                return

            # Fetch the working range and keep only real dialog.
            async with db.execute(
                "SELECT id, message, message_type FROM message_log "
                "WHERE channel_id = ? AND id > ? AND id <= ? ORDER BY id",
                (channel_id, watermark, range_end),
            ) as cursor:
                rows = await cursor.fetchall()
            dialog = []
            for _rowid, message_json, message_type in rows:
                if message_type != "message":
                    continue
                message = json.loads(message_json)
                content = message.get("content")
                if message.get("role") in ("user", "assistant") and \
                        isinstance(content, str) and content.strip():
                    dialog.append(message)

            if not dialog:
                # Pure CONTEXT/system segments advance the watermark
                # without producing a record.
                async with self._db_lock:
                    current = await self._get_watermark(db, channel_id)
                    if current == watermark:
                        await self._set_watermark(db, channel_id, range_end)
                        await db.commit()
                return

            # One background-model call produces the first-person record.
            record = await self._summarize_range(dialog)
            importance = await self.importance_prior.score(dialog)

            # Embed the summary; failure stores embedded=0 for later
            # backfill (trap #10 — degrade, never block).
            embedding: list[float] | None = None
            if self._embedding_enabled():
                try:
                    client = self._llm().get_client("embedding")
                    embedding = (await client.embed([record["summary"]]))[0]
                except Exception:
                    logger.warning(
                        "memory embedding failed; storing record with "
                        "embedded=0 for backfill", exc_info=True,
                    )

            # Atomic commit section: insert + watermark advance together.
            async with self._db_lock:
                current = await self._get_watermark(db, channel_id)
                if current != watermark:
                    logger.debug(
                        "watermark moved during consolidation; discarding",
                        extra={"channel_id": channel_id},
                    )
                    return
                cursor = await db.execute(
                    "INSERT INTO memory (channel_id, created_at, summary, "
                    "importance, topic_tags, participants, msg_id_start, "
                    "msg_id_end, embedded) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        channel_id,
                        time.time(),
                        record["summary"],
                        importance,
                        json.dumps(record["topic_tags"]),
                        json.dumps(record["participants"]),
                        rows[0][0],
                        rows[-1][0],
                        1 if embedding is not None else 0,
                    ),
                )
                memory_id = cursor.lastrowid
                if embedding is not None and self._vec_available:
                    import sqlite_vec
                    await db.execute(
                        "INSERT INTO memory_vec (memory_id, embedding) VALUES (?, ?)",
                        (memory_id, sqlite_vec.serialize_float32(embedding)),
                    )
                await self._set_watermark(db, channel_id, range_end)
                await db.commit()
            logger.info(
                "memory consolidated",
                extra={
                    "channel_id": channel_id,
                    "memory_id": memory_id,
                    "msg_id_start": rows[0][0],
                    "msg_id_end": rows[-1][0],
                    "embedded": embedding is not None,
                },
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            # Fail-soft: the watermark did not advance, so a later trigger
            # retries this range.
            logger.warning(
                "consolidation failed for %s", channel_id, exc_info=True
            )
        finally:
            reset_attribution(attribution_token)

    async def _summarize_range(self, dialog: list[dict]) -> dict:
        """One llm.background call producing the first-person record.

        Returns {"summary": str, "topic_tags": list, "participants": list};
        raises on unusable output (the task logs and leaves the watermark).
        """
        client = self._llm().get_client("background")
        response = await client.chat([
            {"role": "system", "content": self._consolidation_prompt},
            {"role": "user", "content": _dialog_transcript(dialog)},
        ])
        text = response["choices"][0]["message"]["content"]
        data = _parse_json_block(text)
        summary = data.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("consolidation output has no usable summary")
        return {
            "summary": summary.strip(),
            "topic_tags": [str(t) for t in data.get("topic_tags") or []],
            "participants": [str(p) for p in data.get("participants") or []],
        }

    # ------------------------------------------------------------------
    # Watermark helpers
    # ------------------------------------------------------------------

    async def _get_watermark(self, db: aiosqlite.Connection, channel_id: str) -> int:
        """The last consolidated message_log id for a channel (0 when none)."""
        async with db.execute(
            "SELECT last_message_id FROM consolidation_watermark WHERE channel_id = ?",
            (channel_id,),
        ) as cursor:
            row = await cursor.fetchone()
        return row[0] if row else 0

    async def _set_watermark(
        self, db: aiosqlite.Connection, channel_id: str, message_id: int
    ) -> None:
        """Advance the per-channel watermark (caller commits)."""
        await db.execute(
            "INSERT INTO consolidation_watermark (channel_id, last_message_id) "
            "VALUES (?, ?) ON CONFLICT(channel_id) "
            "DO UPDATE SET last_message_id = excluded.last_message_id",
            (channel_id, message_id),
        )

    # ------------------------------------------------------------------
    # Schema plumbing
    # ------------------------------------------------------------------

    def _llm(self):
        """The LLMPlugin instance (role-client access)."""
        from corvidae.llm_plugin import LLMPlugin
        return get_dependency(self.pm, "llm", LLMPlugin)

    def _embedding_enabled(self) -> bool:
        """True when an embedding role is configured and encoders match."""
        return self._embedding_cfg is not None and not self._encoder_mismatch

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
