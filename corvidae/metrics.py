"""Metrics emission and sinks for LLM observability.

Three plugins:
  - MetricsPlugin: consumes on_llm_response and emits on_metrics events
    (token counts, latency, errors) with role/model/stage/channel tags.
  - UsageLogPlugin: consumes on_llm_response and writes one row per LLM
    call into the ``usage_log`` SQLite table (persistence plugin's DB).
  - MetricsJsonlPlugin: consumes on_metrics and appends one JSON line per
    event to a configurable file. Disabled when unconfigured.

All emission and sinking is fail-soft: a metering bug must never take
down the agent, so every failure is logged with a traceback and swallowed.

Config:
    daemon:
      metrics_jsonl: metrics.jsonl   # path for the JSONL sink; omit to disable
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import IO

import aiosqlite

from corvidae.hooks import CorvidaePlugin, get_dependency, hookimpl

logger = logging.getLogger(__name__)

# DDL for the per-call usage log. The exchange_key column stays NULL until
# Phase 2 mints exchange keys into the attribution context.
USAGE_LOG_DDL = """
CREATE TABLE IF NOT EXISTS usage_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    request_id TEXT NOT NULL,
    role TEXT NOT NULL,
    model TEXT NOT NULL,
    stage TEXT,
    channel_id TEXT,
    exchange_key TEXT,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    latency_ms REAL,
    error TEXT
)
"""
USAGE_LOG_INDEX_DDL = "CREATE INDEX IF NOT EXISTS idx_usage_ts ON usage_log (ts)"


class MetricsPlugin(CorvidaePlugin):
    """Emits on_metrics events from every LLM response.

    Emission happens from on_llm_response — never from inside an
    on_metrics implementation (that would recurse through pluggy).
    """

    depends_on = frozenset()

    def __init__(self, pm=None) -> None:
        if pm is not None:
            self.pm = pm

    async def _emit(self, name: str, value: float, tags: dict[str, str]) -> None:
        """Fire one on_metrics event, fail-soft."""
        try:
            await self.pm.ahook.on_metrics(name=name, value=value, tags=tags)
        except Exception:
            logger.warning("on_metrics emission failed for %s", name, exc_info=True)

    @hookimpl
    async def on_llm_response(
        self, role, model, request_id, usage, latency_ms, attribution, error
    ) -> None:
        """Translate one LLM response into token/latency/error metrics."""
        tags = {
            "role": role,
            "model": model,
            "stage": attribution.get("stage", ""),
            "channel": attribution.get("channel_id", ""),
        }
        # Token metrics: emit only the fields the usage dict actually has.
        if usage:
            for metric, key in (
                ("llm.tokens.prompt", "prompt_tokens"),
                ("llm.tokens.completion", "completion_tokens"),
                ("llm.tokens.total", "total_tokens"),
            ):
                if usage.get(key) is not None:
                    await self._emit(metric, float(usage[key]), tags)
        # Latency is always known — the client measured it.
        await self._emit("llm.latency_ms", float(latency_ms), tags)
        # Errors are counted as 1.0 events.
        if error is not None:
            await self._emit("llm.errors", 1.0, tags)


class UsageLogPlugin(CorvidaePlugin):
    """Writes one usage_log row per LLM call via the persistence DB."""

    depends_on = frozenset({"persistence"})

    def __init__(self, pm=None) -> None:
        if pm is not None:
            self.pm = pm
        self._table_ready = False

    def _resolve_db(self) -> aiosqlite.Connection | None:
        """Return the persistence plugin's DB connection, or None if not open yet."""
        from corvidae.persistence import PersistencePlugin
        persistence = get_dependency(self.pm, "persistence", PersistencePlugin)
        return persistence.db

    async def _ensure_table(self) -> aiosqlite.Connection | None:
        """Create the usage_log table on first use; returns the DB or None.

        on_start hook ordering is LIFO across plugins, so the persistence
        DB may not exist yet during our on_start — creation is retried
        lazily on the first write.
        """
        db = self._resolve_db()
        if db is None:
            return None
        if not self._table_ready:
            await db.execute(USAGE_LOG_DDL)
            await db.execute(USAGE_LOG_INDEX_DDL)
            await db.commit()
            self._table_ready = True
        return db

    @hookimpl
    async def on_start(self, config: dict) -> None:
        # Best-effort early DDL; falls back to lazy creation on first write
        # if the persistence DB hasn't opened yet (hook-ordering dependent).
        try:
            await self._ensure_table()
        except Exception:
            logger.warning("usage_log table creation deferred", exc_info=True)

    @hookimpl
    async def on_llm_response(
        self, role, model, request_id, usage, latency_ms, attribution, error
    ) -> None:
        """Insert one row for this LLM call. Fail-soft."""
        try:
            db = await self._ensure_table()
            if db is None:
                logger.warning(
                    "usage_log write skipped: persistence DB not available"
                )
                return
            usage = usage or {}
            await db.execute(
                "INSERT INTO usage_log (ts, request_id, role, model, stage, "
                "channel_id, exchange_key, prompt_tokens, completion_tokens, "
                "total_tokens, latency_ms, error) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    time.time(),
                    request_id,
                    role,
                    model,
                    attribution.get("stage"),
                    attribution.get("channel_id"),
                    attribution.get("exchange_key"),
                    usage.get("prompt_tokens"),
                    usage.get("completion_tokens"),
                    usage.get("total_tokens"),
                    latency_ms,
                    error,
                ),
            )
            await db.commit()
        except Exception:
            logger.warning("usage_log write failed", exc_info=True)


class MetricsJsonlPlugin(CorvidaePlugin):
    """Appends on_metrics events as JSON lines to a configurable file.

    Mirrors JsonlLogPlugin's open/fail-soft behavior: a single append-mode
    handle opened lazily, flushed per write, closed on stop. Disabled
    entirely when ``daemon.metrics_jsonl`` is unset.
    """

    depends_on = frozenset()

    def __init__(self, pm=None) -> None:
        if pm is not None:
            self.pm = pm
        self._path: Path | None = None
        self._handle: IO | None = None

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        await super().on_init(pm, config)
        path = config.get("daemon", {}).get("metrics_jsonl")
        if path is None:
            return
        base_dir = config.get("_base_dir", Path("."))
        self._path = Path(base_dir) / path

    @hookimpl
    async def on_start(self, config: dict) -> None:
        if self._path is None:
            return
        # Ensure the parent directory exists before the first append.
        await asyncio.to_thread(
            self._path.parent.mkdir, parents=True, exist_ok=True
        )
        logger.info("metrics JSONL sink: %s", self._path)

    @hookimpl
    async def on_metrics(self, name: str, value: float, tags: dict) -> None:
        """Append one JSON line per metric event. Fail-soft."""
        if self._path is None:
            return
        record = {"ts": time.time(), "name": name, "value": value, "tags": tags}
        try:
            await asyncio.to_thread(self._write_record, record)
        except Exception:
            logger.warning("metrics JSONL write failed", exc_info=True)

    def _write_record(self, record: dict) -> None:
        """Write a single JSON record (sync, for use with to_thread)."""
        if self._handle is None:
            self._handle = open(self._path, "a")
        self._handle.write(json.dumps(record) + "\n")
        self._handle.flush()

    @hookimpl
    async def on_stop(self) -> None:
        if self._handle is not None:
            await asyncio.to_thread(self._handle.close)
            self._handle = None
