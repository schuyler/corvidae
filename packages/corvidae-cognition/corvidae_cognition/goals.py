"""GoalsPlugin — the agent's open goals, kept in SQLite and shown in the window (#133).

Goals are agent-wide: every channel sees every goal. Open goals enter each
window as one CONTEXT entry admitted through the funnel; the entry carries
its own "replaces any earlier goals list" clause, so the latest one is
current. Retiring a goal writes a memory record of the conclusion first,
so no goal is retired without its record.

Tools:
  add_goal / update_goal / retire_goal — one per verb, so a channel can
  disable creation alone.

Operator: ``corvidae goals --db PATH seed TEXT`` inserts a goal from a
second process; the daemon reads the table every turn.

Heartbeat: ``retired_goals`` gives the heartbeat's review the newest
retirement records (agent-wide).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite
import click

from corvidae.hooks import CorvidaePlugin, get_dependency, hookimpl
from corvidae.tool import Tool, tool_to_schema
from corvidae_cognition.funnel import FunnelPlugin
from corvidae_cognition.memory import MemoryPlugin

logger = logging.getLogger("corvidae.goals")

# Importance of a retirement record; at 1.0 it stays above the retention
# floor, so a concluded purpose stays retrievable.
RETIRED_GOAL_IMPORTANCE = 1.0

GOALS_DDL = """
CREATE TABLE IF NOT EXISTS goals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    retired_at REAL,
    memory_id INTEGER
)
"""


@dataclass(frozen=True)
class Goal:
    """One row of the goals table."""

    id: int
    text: str
    created_at: float
    updated_at: float
    retired_at: float | None
    memory_id: int | None


def _collapse(text: str) -> str:
    """Collapse whitespace runs (newlines included) to one space."""
    return " ".join(text.split())


def _utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _day(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")


class GoalsPlugin(CorvidaePlugin):
    """Goals table, the three goal tools, and the per-turn goals entry."""

    depends_on = frozenset({"persistence", "memory", "funnel", "clock"})

    def __init__(self, pm=None) -> None:
        if pm is not None:
            self.pm = pm
        self._schema_ready = False

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        await super().on_init(pm, config)

    @hookimpl
    async def on_start(self, config: dict) -> None:
        """Create the table now, or on first use if the DB isn't open yet."""
        try:
            await self._ensure_schema()
        except Exception:
            logger.warning("goals schema creation deferred", exc_info=True)

    def _memory(self) -> MemoryPlugin:
        return get_dependency(self.pm, "memory", MemoryPlugin)

    async def _ensure_schema(self) -> aiosqlite.Connection | None:
        """Create the table; returns the DB, or None when it isn't open yet."""
        from corvidae.persistence import PersistencePlugin
        db = get_dependency(self.pm, "persistence", PersistencePlugin).db
        if db is None:
            return None
        if not self._schema_ready:
            await db.execute(GOALS_DDL)
            await db.commit()
            self._schema_ready = True
        return db

    async def _require_db(self) -> aiosqlite.Connection:
        db = await self._ensure_schema()
        if db is None:
            raise RuntimeError("goals: persistence database is not open")
        return db

    async def _all_goals(self) -> list[Goal]:
        """Every row, open and retired, in id order."""
        db = await self._require_db()
        async with db.execute(
            "SELECT id, text, created_at, updated_at, retired_at, memory_id "
            "FROM goals ORDER BY id"
        ) as cursor:
            return [Goal(*row) for row in await cursor.fetchall()]

    async def _get(self, db: aiosqlite.Connection, goal_id: int) -> Goal | None:
        async with db.execute(
            "SELECT id, text, created_at, updated_at, retired_at, memory_id "
            "FROM goals WHERE id = ?",
            (goal_id,),
        ) as cursor:
            row = await cursor.fetchone()
        return Goal(*row) if row else None

    async def _check_open(self, db: aiosqlite.Connection, goal_id: int) -> Goal | str:
        """The open goal, or the error text for an unknown or retired id."""
        goal = await self._get(db, goal_id)
        if goal is None:
            return (
                f"Error: no open goal #{goal_id}; "
                "the ids are in the latest goals entry"
            )
        if goal.retired_at is not None:
            return (
                f"Error: goal #{goal_id} is already retired; add a new goal instead"
            )
        return goal

    def _now(self) -> float:
        return self.pm.get_plugin("clock").now()

    async def retired_goals(self, limit: int) -> list[dict]:
        """The newest-retired goals' conclusions, for the heartbeat's review.

        Each is {"age_seconds", "summary"}: the age since retirement on the
        kernel clock, and the retirement record's summary (a superseded
        record keeps its summary; a redacted one shows its tombstone).
        Agent-wide, like the goals list (#220).
        """
        db = await self._require_db()
        now = self._now()
        async with db.execute(
            "SELECT g.retired_at, m.summary FROM goals g "
            "JOIN memory m ON m.id = g.memory_id "
            "WHERE g.retired_at IS NOT NULL "
            "ORDER BY g.retired_at DESC, g.id DESC LIMIT ?",
            (limit,),
        ) as cursor:
            rows = await cursor.fetchall()
        return [
            {"age_seconds": max(0.0, now - retired_at), "summary": summary}
            for retired_at, summary in rows
        ]

    @hookimpl
    def register_tools(self, tool_registry: list) -> None:
        plugin = self

        async def add_goal(text: str, _ctx=None) -> str:
            """Add an open goal. Open goals appear, with ids, in the latest goals
            CONTEXT entry in your window."""
            text = _collapse(text)
            if not text:
                return "Error: text is empty; say what the goal is"
            db = await plugin._require_db()
            now = plugin._now()
            async with plugin._memory()._db_lock:
                cursor = await db.execute(
                    "INSERT INTO goals (text, created_at, updated_at) VALUES (?, ?, ?)",
                    (text, now, now),
                )
                await db.commit()
            return f"Goal #{cursor.lastrowid} added."

        async def update_goal(id: int, text: str, _ctx=None) -> str:
            """Replace the text of an open goal. Open goals appear, with ids, in the
            latest goals CONTEXT entry in your window."""
            text = _collapse(text)
            if not text:
                return "Error: text is empty; say what the goal is now"
            db = await plugin._require_db()
            checked = await plugin._check_open(db, id)
            if isinstance(checked, str):
                return checked
            async with plugin._memory()._db_lock:
                cursor = await db.execute(
                    "UPDATE goals SET text = ?, updated_at = ? "
                    "WHERE id = ? AND retired_at IS NULL",
                    (text, plugin._now(), id),
                )
                await db.commit()
            # A concurrent retire between the check and the UPDATE.
            if cursor.rowcount == 0:
                return (
                    f"Error: goal #{id} is already retired; add a new goal instead"
                )
            return f"Goal #{id} updated."

        async def retire_goal(id: int, conclusion: str, _ctx=None) -> str:
            """Retire an open goal and record what you concluded. conclusion is what
            you concluded and why, in the first person: 'I concluded X, because Y'.
            Open goals appear, with ids, in the latest goals CONTEXT entry in your
            window."""
            if _ctx is None or _ctx.channel is None:
                return "Error: no channel context available"
            conclusion = conclusion.strip()
            if not conclusion:
                return (
                    "Error: conclusion is empty; say what you concluded and why"
                )
            db = await plugin._require_db()
            checked = await plugin._check_open(db, id)
            if isinstance(checked, str):
                return checked
            channel_id = _ctx.channel.id
            # Link the record to the newest real message on this channel.
            async with db.execute(
                "SELECT MAX(id) FROM message_log "
                "WHERE channel_id = ? AND message_type = 'message'",
                (channel_id,),
            ) as cursor:
                (msg_id,) = await cursor.fetchone()
            now = plugin._now()
            summary = (
                f'I retired my goal "{checked.text}" '
                f"(set {_day(checked.created_at)}). {conclusion}"
            )
            # The record goes first: no goal is retired without its record.
            try:
                memory_id = await plugin._memory().write_record(
                    channel_id=channel_id,
                    summary=summary,
                    created_at=now,
                    importance=RETIRED_GOAL_IMPORTANCE,
                    msg_id_start=msg_id,
                    msg_id_end=msg_id,
                )
            except Exception as exc:
                logger.warning(
                    "retirement record for goal %d failed", id, exc_info=True
                )
                return (
                    f"Error: couldn't record the conclusion for goal #{id} "
                    f"({type(exc).__name__}); the goal is still open, try again"
                )
            # write_record released the lock; take it for the state change only.
            async with plugin._memory()._db_lock:
                cursor = await db.execute(
                    "UPDATE goals SET retired_at = ?, updated_at = ?, memory_id = ? "
                    "WHERE id = ? AND retired_at IS NULL",
                    (now, now, memory_id, id),
                )
                await db.commit()
            if cursor.rowcount == 0:
                return (
                    f"Error: goal #{id} is already retired; add a new goal instead"
                )
            return (
                f"Goal #{id} retired; conclusion recorded as memory #{memory_id}."
            )

        for fn in (add_goal, update_goal, retire_goal):
            tool_registry.append(Tool(name=fn.__name__, fn=fn, schema=tool_to_schema(fn)))

    @hookimpl
    async def before_agent_turn(self, channel, trace_id, meta) -> None:
        """Admit the current goals entry; the funnel dedupes an unchanged one."""
        conv = getattr(channel, "conversation", None)
        if conv is None:
            return
        goals = await self._all_goals()
        if not goals:
            return
        # The last-change time, not "now": the entry changes only when a goal does.
        as_of = _utc(max(g.updated_at for g in goals))
        open_goals = [g for g in goals if g.retired_at is None]
        if open_goals:
            lines = [
                f"Open goals as of {as_of} (this list replaces any earlier goals list):"
            ]
            lines += [f"#{g.id} {g.text}" for g in open_goals]
            entry = "\n".join(lines)
        else:
            entry = (
                f"No open goals as of {as_of} "
                "(this replaces any earlier goals list)."
            )
        funnel = get_dependency(self.pm, "funnel", FunnelPlugin)
        await funnel.admit(channel, conv, "goals", [entry])


# ---------------------------------------------------------------------------
# Operator CLI
# ---------------------------------------------------------------------------


@click.group("goals")
@click.option("--db", "db_path", required=True, help="Path to sessions.db")
@click.pass_context
def goals_command(ctx, db_path):
    """Operator tool for the agent's goals."""
    # Check the path first: aiosqlite would silently create an empty database.
    if not Path(db_path).is_file():
        raise click.ClickException(f"Database not found: {db_path}")
    ctx.ensure_object(dict)
    ctx.obj["db_path"] = db_path


@goals_command.command("seed")
@click.argument("text")
@click.pass_context
def _seed_cmd(ctx, text):
    """Add an open goal; it appears on every channel at its next turn."""
    db_path = ctx.obj["db_path"]
    text = _collapse(text)
    if not text:
        raise click.ClickException("Goal text is empty; pass the goal as TEXT")

    async def _run():
        conn = await aiosqlite.connect(db_path)
        await conn.execute("PRAGMA busy_timeout = 5000")
        try:
            # A second process needs WAL to write beside the running daemon.
            async with conn.execute("PRAGMA journal_mode") as cursor:
                row = await cursor.fetchone()
            mode = (row[0] if row else "").lower()
            if mode != "wal":
                raise click.ClickException(
                    f"Database journal mode is '{mode}' — the goals CLI requires "
                    f"WAL mode; the corvidae daemon sets it on startup ({db_path})"
                )
            await conn.execute(GOALS_DDL)
            now = time.time()
            cursor = await conn.execute(
                "INSERT INTO goals (text, created_at, updated_at) VALUES (?, ?, ?)",
                (text, now, now),
            )
            await conn.commit()
            click.echo(
                f"Seeded goal #{cursor.lastrowid}; "
                "it appears on every channel at its next turn."
            )
        finally:
            await conn.close()

    asyncio.run(_run())
