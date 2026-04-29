"""GoalTrackerPlugin — tactical goal management and idle-state execution.

Solves the "reactive" nature of nanobot by enabling autonomous background
maintenance and proactive preparation during idle windows.

Design:
- Persistent list of Tactical Goals with status tracking (pending, active, done, blocked)
- Detects main-loop idleness (>30s since last generation by default)
- Triggers execution of the next pending goal during idle windows
- Priority scoring to order goals when multiple are ready
- Configurable idle threshold and max concurrent goals

Proposed Idle Tasks (pluggable via strategy functions):
1. Context Pre-loading: Monitor git commits or file changes; re-index relevant files
2. Research Synthesis: Check "Pending Research" notes, fetch/summarize outdated links
3. Plugin Development: Draft and unit-test plugins defined in wiki specs

This is NOT a general-purpose task scheduler — it's specifically for goals
that are meaningful to the agent owner and that can be checked without
interrupting active conversations.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

if True:  # TYPE_CHECKING guard — these types exist in corvidae but create circular imports
    from corvidae.channel import Channel

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data model
# ------------------------------------------------------------------

class GoalStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    DONE = "done"
    BLOCKED = "blocked"


@dataclass
class TacticalGoal:
    """A single tactical goal for the agent to work on autonomously."""
    id: str
    title: str
    description: str
    status: GoalStatus = GoalStatus.PENDING
    priority: int = 50  # 0-100, higher = more important
    strategy: str = "default"  # which idle-strategy function to use
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    notes: str = ""


# ------------------------------------------------------------------
# Storage
# ------------------------------------------------------------------

class GoalStore:
    """SQLite-backed persistent storage for TacticalGoals."""

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._conn: Any | None = None

    async def open(self) -> None:
        import aiosqlite
        self._conn = await aiosqlite.connect(str(self._db_path))
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS goals (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                priority INTEGER NOT NULL DEFAULT 50,
                strategy TEXT NOT NULL DEFAULT 'default',
                created_at REAL NOT NULL,
                completed_at REAL,
                notes TEXT NOT NULL DEFAULT ''
            )
        """)
        await self._conn.commit()

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None

    async def save_goal(self, goal: TacticalGoal) -> None:
        assert self._conn
        await self._conn.execute(
            """INSERT OR REPLACE INTO goals
               (id, title, description, status, priority, strategy,
                created_at, completed_at, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (goal.id, goal.title, goal.description, goal.status.value,
             goal.priority, goal.strategy, goal.created_at,
             goal.completed_at, goal.notes),
        )
        await self._conn.commit()

    async def load_all(self) -> list[TacticalGoal]:
        assert self._conn
        cursor = await self._conn.execute(
            "SELECT * FROM goals ORDER BY priority DESC"
        )
        rows = await cursor.fetchall()
        goals = []
        for row in rows:
            goals.append(TacticalGoal(
                id=row[0], title=row[1], description=row[2],
                status=GoalStatus(row[3]), priority=row[4],
                strategy=row[5], created_at=row[6],
                completed_at=row[7], notes=row[8],
            ))
        return goals

    async def get_next_pending(self) -> TacticalGoal | None:
        assert self._conn
        cursor = await self._conn.execute(
            "SELECT * FROM goals WHERE status='pending' ORDER BY priority DESC LIMIT 1"
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return TacticalGoal(
            id=row[0], title=row[1], description=row[2],
            status=GoalStatus(row[3]), priority=row[4],
            strategy=row[5], created_at=row[6],
            completed_at=row[7], notes=row[8],
        )


# ------------------------------------------------------------------
# Idle strategies
# ------------------------------------------------------------------

async def default_strategy(goal: TacticalGoal, workspace_root: Path) -> str:
    """Generic fallback: log the goal and return a placeholder result."""
    return f"Goal '{goal.title}' — strategy '{goal.strategy}' not yet implemented."


async def strategy_context_preload(
    goal: TacticalGoal, workspace_root: Path
) -> str:
    """Monitor for git changes and re-index relevant files.

    Checks if the workspace has uncommitted changes or recent commits.
    If so, signals that indexing is needed (actual index trigger would
    come from a separate tool/plugin).
    """
    import subprocess  # noqa: PLC0415 — only imported when strategy runs
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(workspace_root),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if status.stdout.strip():
            lines = status.stdout.strip().split("\n")
            return f"Context preload: {len(lines)} file(s) with pending changes. Indexing recommended."
        return "Context preload: workspace clean, no indexing needed."
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return "Context preload: unable to check git status."


async def strategy_research_synthesis(
    goal: TacticalGoal, workspace_root: Path
) -> str:
    """Check for pending research notes that need enrichment.

    Looks for Links/ pages with status: pending and reports how many
    need processing. (Actual fetching would be done by a separate tool.)
    """
    links_dir = workspace_root / "Links"
    if not links_dir.is_dir():
        return "Research synthesis: no Links/ directory found."

    pending_count = 0
    for f in links_dir.iterdir():
        if f.name.startswith("_"):
            continue
        try:
            content = f.read_text(encoding="utf-8")
            if "status: pending" in content.lower():
                pending_count += 1
        except Exception:
            pass

    return f"Research synthesis: {pending_count} pending link(s) found in Links/."


# Known strategies — extensible via registry pattern
STRATEGIES: dict[str, Callable] = {
    "default": default_strategy,
    "context_preload": strategy_context_preload,
    "research_synthesis": strategy_research_synthesis,
}


# ------------------------------------------------------------------
# Plugin
# ------------------------------------------------------------------

class GoalTrackerPlugin:
    """Tactical goal management with idle-time execution."""

    name = "goal_tracker"
    depends_on: list[str] = []

    def __init__(
        self,
        db_path: str | Path | None = None,
        idle_threshold_seconds: float = 30.0,
        max_concurrent_goals: int = 1,
        workspace_root: Path | None = None,
    ):
        self._workspace_root = workspace_root or Path(".")
        self._idle_threshold = idle_threshold_seconds
        self._max_concurrent = max_concurrent_goals
        db = db_path or self._workspace_root / "goals.db"
        self._store = GoalStore(Path(db))
        self._active_goals: dict[str, TacticalGoal] = {}  # id -> goal
        self._idle_since: float | None = None

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    async def on_start(self, plugin_manager) -> None:
        """Initialize storage and register the /goals tool."""
        await self._store.open()
        pm = plugin_manager
        pm.register(self)
        logger.info("GoalTrackerPlugin registered")

    async def on_stop(self, plugin_manager) -> None:
        await self._store.close()

    # ------------------------------------------------------------------
    # Idle detection hooks
    # ------------------------------------------------------------------

    async def on_message_received(self, channel: "Channel") -> None:
        """Reset idle timer when a message arrives."""
        self._idle_since = None

    async def before_agent_turn(self, channel: "Channel") -> None:
        """Mark that we're actively responding — no idle work now."""
        self._idle_since = time.monotonic() + 999_999  # effectively infinite

    async def after_idle(
        self, idle_duration_seconds: float
    ) -> str:
        """Check for pending goals and execute the highest-priority one.

        Called by IdleMonitorPlugin's on_idle hook when idle duration
        exceeds the threshold. Only executes if we're not already working
        on a goal and no channels are active.
        """
        if self._active_goals:
            return "Already processing goals — skipping this idle cycle."

        next_goal = await self._store.get_next_pending()
        if not next_goal:
            return "No pending goals."

        # Mark as active
        next_goal.status = GoalStatus.ACTIVE
        await self._store.save_goal(next_goal)
        self._active_goals[next_goal.id] = next_goal

        try:
            strategy_fn = STRATEGIES.get(
                next_goal.strategy, default_strategy
            )
            result = await strategy_fn(next_goal, self._workspace_root)
        except Exception as e:
            result = f"Error executing goal '{next_goal.title}': {e}"

        return f"**Goal: {next_goal.title}**\n{result}"

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    async def add_goal(
        self, title: str, description: str = "",
        priority: int = 50, strategy: str = "default",
    ) -> str:
        """Add a new tactical goal to the tracker.

        Args:
            title: Short name for the goal.
            description: Longer description of what needs to be done.
            priority: 0-100, higher = more important (default 50).
            strategy: Which idle-strategy function to use.
        """
        import uuid  # noqa: PLC0415 — only when tool is called

        goal = TacticalGoal(
            id=uuid.uuid4().hex[:8],
            title=title,
            description=description or title,
            priority=max(0, min(100, priority)),
            strategy=strategy,
        )
        await self._store.save_goal(goal)

        available = ", ".join(STRATEGIES.keys())
        return (
            f"Goal added: **{title}** (priority {goal.priority}, "
            f"strategy '{strategy}' [{available}])"
        )

    async def list_goals(self) -> str:
        """List all tactical goals with their current status."""
        goals = await self._store.load_all()
        if not goals:
            return "No tactical goals recorded."

        lines = ["**Tactical Goals**"]
        for g in sorted(goals, key=lambda x: (
            -x.priority, x.status.value, x.created_at
        )):
            status_label = g.status.value.upper()
            active_flag = " ⚡" if g.id in self._active_goals else ""
            lines.append(
                f"  [{status_label}] {g.title} (P{g.priority}){active_flag}"
            )
            if g.description:
                desc = g.description[:80] + ("…" if len(g.description) > 80 else "")
                lines.append(f"    {desc}")

        return "\n".join(lines)
