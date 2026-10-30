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
      idle_consolidate_after: 1800   # seconds a channel must be quiet (no
                                     # queued or in-flight work) before its
                                     # tail is consolidated; repeats at this
                                     # period while the quiet lasts
      retention:
        interval: 21600              # seconds between retention runs (clock
                                     # timer; last_run survives restarts)
        max_delay: 3600              # longest a due run waits for the system
                                     # to go quiet before running anyway
      half_life_days: 30             # retrieval recency decay (e-folding days)
      retrieval:
        k: 8                         # KNN candidate count
        band_on: score               # score | similarity — what bands compare
        recency_from: created        # created | last_used — ranking age origin
        bands:
          strong: 0.75               # value >= strong → asserted confidently
          moderate: 0.60             # value >= moderate → hedged
      channel_groups: {}             # {group_name: [channel ids or * ? globs]}
                                     # shared memory
      consolidation_fork_prompt: ... # instruction for forked consolidation
                                     # (agent.forks, #51); static config

Changes to ``idle_consolidate_after`` and ``retention.*`` take effect at
restart: the idle watch and retention timer are set when the plugin starts.
A channel that never goes quiet is consolidated through compaction.
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import TYPE_CHECKING, Protocol

import aiosqlite

from corvidae.attribution import reset_attribution, set_attribution
from corvidae.context import MessageType
from corvidae.fork import fork_agent, quote, window_db_ids
from corvidae.hooks import CorvidaePlugin, get_dependency, hookimpl
from corvidae.persistence import add_column_if_missing

if TYPE_CHECKING:
    from corvidae.clock import Clock, Timer
    from corvidae.idle import IdleTracker, IdleWatch

logger = logging.getLogger("corvidae.memory")

# Seconds a channel must be inactive before the idle trigger consolidates
# its un-consolidated tail.
DEFAULT_IDLE_CONSOLIDATE_AFTER = 1800.0

# Seconds between retention runs, and the longest a due run waits for the
# system to go quiet before running regardless.
DEFAULT_RETENTION_INTERVAL = 21600.0
DEFAULT_RETENTION_MAX_DELAY = 3600.0
# How long the system must be quiet for retention to prefer that moment.
RETENTION_QUIET_SECONDS = 60.0
# Minimum wait before retention is re-armed after a failed read, and the
# floor under the next due time so a failed last_run write cannot loop.
RETENTION_DB_RETRY_SECONDS = 5.0

# Retrieval scoring parameters. These are tuned parameters, not truths —
# the §6 eval benchmarks (tests/evals, scripts/eval_memory.py) govern
# changes to them.
DEFAULT_RETRIEVAL_K = 8
DEFAULT_HALF_LIFE_DAYS = 30.0
DEFAULT_BAND_STRONG = 0.75
DEFAULT_BAND_MODERATE = 0.60
# What the relevance band is computed from, and where ranking age is
# measured from (GitHub issue #52). The first entry is the default.
BAND_ON_CHOICES = ("score", "similarity")
RECENCY_FROM_CHOICES = ("created", "last_used")
# Honest similarity stand-in for FTS5-degraded retrieval, where cosine
# similarity cannot be measured: keyword hits enter at the moderate band.
FTS_SIMILARITY_PROXY = 0.6
# Vec KNN over-fetch factor: candidates are filtered by channel scope and
# indexed/redacted flags after the KNN scan, so fetch more than k.
VEC_OVERFETCH = 4

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

# Default fork instruction for consolidation (GitHub issue #51). With
# agent.forks on, the record is written by a fork of the live window: this
# text is appended to it as a user turn, followed by a sentence naming where
# the range starts. prompts/memory_consolidation_fork.md is the documented
# copy; memory.consolidation_fork_prompt (static config, not a runtime
# tunable) overrides it. The agent's own system prompt is above it, so it
# doesn't need to explain who "I" is.
DEFAULT_CONSOLIDATION_FORK_PROMPT = (
    "[Memory consolidation] Pause the conversation here and write a record "
    "of part of it for your long-term autobiographical memory, in the first "
    "person, as yourself.\n\n"
    "Keep what you were told apart from what you inferred or speculated "
    "(\"Schuyler told me...\", \"I speculated that...\", \"I suggested...\"). "
    "Keep concrete details that would matter later: names, decisions, "
    "commitments, corrections, and outcomes. Omit filler. Do not call tools.\n\n"
    "Respond with a single JSON object, nothing else:\n"
    "{\n"
    '  "summary": "<first-person summary, a few sentences>",\n'
    '  "topic_tags": ["<short topic tag>", ...],\n'
    '  "participants": ["<sender name>", ...]\n'
    "}"
)

# Bound on fork records waiting for their on_compaction task (keyed by
# (channel_id, range end)). An entry is consumed by the task that follows
# its compaction; the bound only matters if that task never runs.
FORK_STASH_MAXSIZE = 64

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
        raise ValueError(f"consolidation: no JSON object in LLM output: {text[:500]!r}")
    return json.loads(text[start:end + 1])


class _StubChannel:
    """Minimal duck-typed channel for consolidation when the registry misses.

    Exposes only ``runtime_overrides`` (empty) and ``id`` — enough for
    ``resolve_tunable`` to fall through to the config/default surfaces
    (sanctioned by WP2.3's duck-typed resolver).
    """

    def __init__(self, channel_id: str) -> None:
        self.id = channel_id
        self.runtime_overrides: dict = {}


def _dialog_transcript(messages: list[dict]) -> str:
    """Render dialog messages as 'role: content' lines for a prompt."""
    return "\n".join(
        f"{m.get('role', '?')}: {m.get('content', '')}" for m in messages
    )


def _record_from_text(text: str) -> dict:
    """Parse a consolidation reply into the record dict.

    Returns {"summary": str, "topic_tags": list, "participants": list};
    raises ValueError on unusable output (no JSON, or no summary).
    """
    data = _parse_json_block(text)
    summary = data.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("consolidation output has no usable summary")
    return {
        "summary": summary.strip(),
        "topic_tags": [str(t) for t in data.get("topic_tags") or []],
        "participants": [str(p) for p in data.get("participants") or []],
    }


def _fork_range_sentence(prefix: list[dict], dialog_ids: set[int]) -> str:
    """Name where a consolidation range starts, for a fork instruction.

    ``prefix`` is the window the fork sees; ``dialog_ids`` the rowids of the
    range's dialog rows. When nothing but a summary precedes the first of
    them, the range is everything after the summary (or everything). Else
    rows below the watermark (already consolidated by an earlier idle run)
    are in view, so the first new dialog message is quoted. Near-dup merge
    is the backstop if the model draws the line a little wrong.
    """
    first = next(
        (i for i, m in enumerate(prefix) if m.get("_db_id") in dialog_ids), len(prefix)
    )
    before = prefix[:first]
    if all(m.get("_message_type") == MessageType.SUMMARY for m in before):
        if before:
            return "Record the conversation above that follows the summary at its top."
        return "Record the whole conversation above."
    opener = next(
        (
            m for m in prefix[first:]
            if m.get("role") in ("user", "assistant")
            and isinstance(m.get("content"), str) and m["content"].strip()
        ),
        prefix[first] if first < len(prefix) else {},
    )
    return (
        "Record the conversation above from the message that begins "
        f"«{quote(opener.get('content'))}» onward. Earlier messages are "
        "already in your memory; do not restate them."
    )


async def fetch_range_rows(
    db: aiosqlite.Connection, channel_id: str, after_id: int, through_id: int
) -> list[tuple]:
    """Raw message_log rows for (after_id, through_id], id-ordered."""
    async with db.execute(
        "SELECT id, message, message_type FROM message_log "
        "WHERE channel_id = ? AND id > ? AND id <= ? ORDER BY id",
        (channel_id, after_id, through_id),
    ) as cursor:
        return await cursor.fetchall()


async def source_timestamp(db: aiosqlite.Connection, message_id: int) -> float:
    """When a message_log row was logged — the date a memory record built
    from a range ending at that row carries (GitHub issue #52). Falls back
    to now when the row is gone."""
    async with db.execute(
        "SELECT timestamp FROM message_log WHERE id = ?", (message_id,)
    ) as cursor:
        row = await cursor.fetchone()
    return row[0] if row is not None and row[0] is not None else time.time()


def dialog_from_rows(rows: list[tuple]) -> list[dict]:
    """The consolidation working set: real user/assistant dialog only."""
    dialog = []
    for _rowid, message_json, message_type in rows:
        if message_type != "message":
            continue
        message = json.loads(message_json)
        content = message.get("content")
        if message.get("role") in ("user", "assistant") and \
                isinstance(content, str) and content.strip():
            dialog.append(message)
    return dialog


def _retrieval_choice(retrieval_cfg: dict, key: str, choices: tuple[str, ...]) -> str:
    """Read an enum-valued memory.retrieval key, warning on and ignoring an
    unsupported value so a typo degrades to the default (choices[0])."""
    value = retrieval_cfg.get(key, choices[0])
    if value not in choices:
        logger.warning(
            "memory.retrieval.%s %r is not one of %s; using %r",
            key, value, ", ".join(choices), choices[0],
        )
        return choices[0]
    return value


# Wildcards a memory.channel_groups member may use. These are the only
# metacharacters fnmatchcase and SQLite GLOB agree on; '[' / ']' (character
# classes) differ between the two and are rejected in pattern members
# (GitHub issue #50). Wildcard-free members match by equality, so a
# literal id may contain brackets (IRC nicks can).
_GLOB_WILDCARDS = frozenset("*?")
_GLOB_REJECTED = frozenset("[]")


def _is_pattern(member: str) -> bool:
    """Whether a channel_groups member is a glob pattern, not a literal id."""
    return not _GLOB_WILDCARDS.isdisjoint(member)


def _validate_channel_groups(groups: object) -> dict[str, list[str]]:
    """Check memory.channel_groups is {name: [str]} with no bracket patterns.

    Raises ValueError rather than dropping a bad member: silently narrowing
    or reshaping a memory compartment is worse than refusing to start.
    """
    if not isinstance(groups, dict):
        raise ValueError(
            f"memory.channel_groups must be a mapping, got {type(groups).__name__}"
        )
    for name, members in groups.items():
        if not isinstance(members, list) or not all(
            isinstance(m, str) for m in members
        ):
            raise ValueError(
                f"memory.channel_groups.{name} must be a list of channel ids"
            )
        for member in members:
            if _is_pattern(member) and not _GLOB_REJECTED.isdisjoint(member):
                raise ValueError(
                    f"memory.channel_groups.{name} member {member!r}: "
                    "'[' and ']' are not supported in a '*' / '?' pattern"
                )
    return groups


@dataclass(frozen=True)
class ChannelScope:
    """The set of channels whose memories a calling channel may read.

    ``ids`` match by equality (the caller's own id and literal group
    members); ``patterns`` are '*' / '?' globs from group members, matched
    case-sensitively with the same semantics in Python and SQLite GLOB.
    """

    ids: frozenset[str]
    patterns: tuple[str, ...] = ()

    def contains(self, channel_id: str) -> bool:
        """Whether a memory owned by channel_id is visible in this scope."""
        return channel_id in self.ids or any(
            fnmatchcase(channel_id, p) for p in self.patterns
        )

    def sql(self, column: str) -> tuple[str, list[str]]:
        """A parenthesized WHERE fragment restricting column to this scope,
        with its bind parameters (ids first, then patterns). An empty scope
        yields "(0)", which matches nothing."""
        if not self.ids and not self.patterns:
            return "(0)", []
        clauses = []
        if self.ids:
            placeholders = ",".join("?" for _ in self.ids)
            clauses.append(f"{column} IN ({placeholders})")
        clauses.extend(f"{column} GLOB ?" for _ in self.patterns)
        return "(" + " OR ".join(clauses) + ")", [*sorted(self.ids), *self.patterns]

    def same_reach(self, other: "ChannelScope") -> bool:
        """Whether two scopes admit exactly the same channels.

        Ids already matched by one of the scope's own patterns are dropped
        before comparing, so the scopes of two members of a "*" group (each
        holding its own id plus "*") compare equal. Syntactically different
        patterns compare unequal even if they match the same channels; that
        errs toward "different", which only forgoes a merge.
        """
        def normal(scope: "ChannelScope") -> tuple[frozenset[str], frozenset[str]]:
            # Drop ids a pattern in the same scope already covers.
            ids = frozenset(
                i for i in scope.ids
                if not any(fnmatchcase(i, p) for p in scope.patterns)
            )
            return ids, frozenset(scope.patterns)

        return normal(self) == normal(other)

    def describe(self) -> str:
        """Human-readable listing for scope errors shown to the model."""
        return ", ".join([*sorted(self.ids), *self.patterns])


def _format_age(seconds: float) -> str:
    """Compact human age for retrieval lines: 'now', '12m', '5h', '3d'."""
    if seconds < 60:
        return "now"
    if seconds < 3600:
        return f"{int(seconds // 60)}m"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h"
    return f"{int(seconds // 86400)}d"


def _fts_match_query(text: str) -> str | None:
    """Build a safe FTS5 MATCH expression from free text.

    Raw user text can be FTS5 syntax-invalid; quote each word token and OR
    them. Returns None when the text has no word tokens.
    """
    tokens = re.findall(r"\w+", text)[:20]
    if not tokens:
        return None
    return " OR ".join(f'"{token}"' for token in tokens)


class ImportancePrior(Protocol):
    """Pluggable supplier of the consolidation-time importance prior (§3.1).

    Held as ``MemoryPlugin.importance_prior`` and assignable by later
    phases: the Phase 2 appraisal replaces the default rubric; a Phase 6
    toggle adds a surprise term.
    """

    async def score(
        self,
        messages: list[dict],
        msg_id_range: tuple[int, int] | None = None,
        channel=None,
    ) -> float: ...   # 0.0–1.0


class RubricPrior:
    """Default importance prior: cheap-model rubric rating (§3.1).

    Returns 0.5 on any failure (logged) — a scoring outage must never
    block memory formation.
    """

    def __init__(self, get_client: Callable[[], object]) -> None:
        # A callable, not a client, so the current background client is
        # resolved at call time (clients can be swapped on config reload).
        self._get_client = get_client

    async def score(
        self,
        messages: list[dict],
        msg_id_range: tuple[int, int] | None = None,
        channel=None,
    ) -> float:
        # msg_id_range / channel are accepted for protocol conformance
        # (WP2.5) but the rubric ignores both — it rates the dialog text.
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
    dimensions INTEGER NOT NULL,
    document_prefix TEXT NOT NULL DEFAULT '',
    query_prefix TEXT NOT NULL DEFAULT ''
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

# The per-retrieval raw profile stream (§3.7 wants this accumulating from
# day one). Each row is keyed by the turn's trace id, and the profile is
# copied into exchange_log under the same id.
RETRIEVAL_LOG_DDL = """
CREATE TABLE IF NOT EXISTS retrieval_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    channel_id TEXT NOT NULL,
    trace_id TEXT,                  -- NULL when the retrieval ran outside any trace
    top_score REAL,                 -- recency-decayed score of the top-ranked candidate
    top_similarity REAL,            -- raw similarity of that same candidate
    hit_count INTEGER NOT NULL,
    admitted_count INTEGER NOT NULL,
    degraded_to_fts INTEGER NOT NULL DEFAULT 0
)
"""

# Phase 1b: one-row table persisting the retention job's last-run timestamp
# so restart loops don't re-run it on every start (trap #5).
RETENTION_META_DDL = """
CREATE TABLE IF NOT EXISTS retention_meta (
    last_run REAL NOT NULL DEFAULT 0
)
"""

# Phase 1b: standalone (non-external-content) FTS5 over message_log content.
# Content-sync is done by triggers, NOT external-content; the update trigger
# uses plain DELETE because the FTS5 'delete' command is only valid for
# external-content tables (verified empirically, SQLite 3.47.1 — plain DELETE
# works correctly for regular FTS5 tables).
MESSAGE_FTS_DDL = """
CREATE VIRTUAL TABLE IF NOT EXISTS message_fts USING fts5(content_text)
"""

MESSAGE_FTS_INSERT_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS message_log_ai AFTER INSERT ON message_log BEGIN
    INSERT INTO message_fts(rowid, content_text)
    VALUES (new.id, coalesce(json_extract(new.message, '$.content'), ''));
END
"""

# Plain DELETE — not the FTS5 'delete' command — because message_fts is a
# regular (non-external-content) FTS5 table. Verified empirically: the 'delete'
# command aborts every UPDATE message_log SET message=... with "SQL logic error".
MESSAGE_FTS_UPDATE_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS message_log_au AFTER UPDATE OF message ON message_log BEGIN
    DELETE FROM message_fts WHERE rowid = old.id;
    INSERT INTO message_fts(rowid, content_text)
    VALUES (new.id, coalesce(json_extract(new.message, '$.content'), ''));
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

    depends_on = frozenset({"persistence", "llm", "clock", "idle"})

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
        self._consolidation_fork_prompt: str = DEFAULT_CONSOLIDATION_FORK_PROMPT
        # Fork records made in before_compaction, waiting for the
        # on_compaction task: (channel_id, range_end) -> (watermark, record).
        self._fork_stash: collections.OrderedDict[tuple[str, int], tuple[int, dict]] = (
            collections.OrderedDict()
        )
        self._retrieval_k: int = DEFAULT_RETRIEVAL_K
        self._band_strong: float = DEFAULT_BAND_STRONG
        self._band_moderate: float = DEFAULT_BAND_MODERATE
        self._half_life_days: float = DEFAULT_HALF_LIFE_DAYS
        self._band_on: str = BAND_ON_CHOICES[0]
        self._recency_from: str = RECENCY_FROM_CHOICES[0]
        self._channel_groups: dict[str, list[str]] = {}
        # Phase 1b: near-dup merge threshold (WP1b.2)
        self._dup_threshold: float = 0.95
        # Tracked background tasks (consolidation) — plugin-owned
        # asyncio.create_task, never the TaskQueue: every TaskQueue
        # completion triggers a full main-model turn (§2.3, trap #5).
        self._tasks: set[asyncio.Task] = set()
        # Channels with an idle consolidation in flight; a repeat firing
        # for one of them is skipped.
        self._idle_inflight: set[str] = set()
        # Kernel clock and idle tracker; both set in on_start, or both None.
        self._clock: Clock | None = None
        self._idle: IdleTracker | None = None
        self._retention_interval: float = DEFAULT_RETENTION_INTERVAL
        self._retention_max_delay: float = DEFAULT_RETENTION_MAX_DELAY
        # Per-channel consolidation watch.
        self._idle_watch: IdleWatch | None = None
        # Timer for the next retention due time (or the read-failure re-arm).
        self._retention_timer: Timer | None = None
        # Quiet watch and max-delay deadline, live from due until the run.
        self._retention_watch: IdleWatch | None = None
        self._retention_deadline: Timer | None = None
        # True from the due time until the run starts.
        self._retention_pending: bool = False
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
        self._idle_consolidate_after = float(self._memory_cfg.get(
            "idle_consolidate_after", DEFAULT_IDLE_CONSOLIDATE_AFTER
        ))
        retention_cfg = self._memory_cfg.get("retention", {}) or {}
        self._retention_interval = float(
            retention_cfg.get("interval", DEFAULT_RETENTION_INTERVAL)
        )
        self._retention_max_delay = float(
            retention_cfg.get("max_delay", DEFAULT_RETENTION_MAX_DELAY)
        )
        self._consolidation_prompt = self._memory_cfg.get(
            "consolidation_prompt", DEFAULT_CONSOLIDATION_PROMPT
        )
        self._consolidation_fork_prompt = self._memory_cfg.get(
            "consolidation_fork_prompt", DEFAULT_CONSOLIDATION_FORK_PROMPT
        )
        retrieval_cfg = self._memory_cfg.get("retrieval", {}) or {}
        self._retrieval_k = retrieval_cfg.get("k", DEFAULT_RETRIEVAL_K)
        bands = retrieval_cfg.get("bands", {}) or {}
        self._band_strong = bands.get("strong", DEFAULT_BAND_STRONG)
        self._band_moderate = bands.get("moderate", DEFAULT_BAND_MODERATE)
        self._half_life_days = self._memory_cfg.get(
            "half_life_days", DEFAULT_HALF_LIFE_DAYS
        )
        self._band_on = _retrieval_choice(retrieval_cfg, "band_on", BAND_ON_CHOICES)
        self._recency_from = _retrieval_choice(
            retrieval_cfg, "recency_from", RECENCY_FROM_CHOICES
        )
        self._channel_groups = _validate_channel_groups(
            self._memory_cfg.get("channel_groups", {}) or {}
        )
        # Phase 1b: near-dup merge threshold (WP1b.2)
        self._dup_threshold = float(self._memory_cfg.get("dup_threshold", 0.95))

    @hookimpl
    async def on_start(self, config: dict) -> None:
        """Create the schema, run the startup retention check, arm the clock.

        Schema creation is deferred to first use when the persistence DB has
        not opened yet (hook-ordering dependent, same pattern as
        UsageLogPlugin). With the schema ready, spawns the message_fts
        backfill and runs the retention check. Then, when the kernel clock
        and idle tracker are both registered, registers the per-channel
        consolidation watch and arms the retention timer.
        """
        # shortcut: hand-built test plugin managers omit the kernel clock; make it strict if a non-Runtime entry point appears
        clock = self.pm.get_plugin("clock")
        idle = self.pm.get_plugin("idle")
        if clock is None or idle is None:
            # Memory uses both objects or neither.
            logger.warning(
                "memory: no kernel clock/idle tracker registered; idle "
                "consolidation and periodic retention are off"
            )
            clock = idle = None
        self._clock = clock
        self._idle = idle
        try:
            db = await self._ensure_schema()
            if db is not None:
                self._spawn(self._backfill_message_fts())
                # Run synchronously so it completes before on_start returns,
                # preventing races with tests/callers that call run_retention_job
                # directly without waiting for background tasks first.
                await self._run_retention_if_due()
        except Exception:
            logger.warning("memory schema creation deferred", exc_info=True)
        if self._clock is None or self._idle is None:
            return
        # Consolidate each channel's tail once it has been quiet a period.
        self._idle_watch = self._idle.watch_channels(
            self._idle_consolidate_after,
            self._on_channel_idle,
            name="memory.consolidation",
        )
        # Arm even if the schema was not ready; the arm retries its read.
        await self._arm_retention()

    @hookimpl
    async def on_stop(self) -> None:
        """Cancel the watches and timers, then tracked background tasks."""
        # Stop the idle watches so no new consolidation or retention starts.
        for watch in (self._idle_watch, self._retention_watch):
            if watch is not None:
                watch.cancel()
        # Cancel the retention timer and its max_delay deadline.
        for timer in (self._retention_timer, self._retention_deadline):
            if timer is not None:
                timer.cancel()
        # Drop the handles and the pending flag so nothing re-arms.
        self._idle_watch = None
        self._retention_watch = None
        self._retention_timer = None
        self._retention_deadline = None
        self._retention_pending = False
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)
        self._tasks.clear()

    # ------------------------------------------------------------------
    # Consolidation triggers (bootstrap-mapping §3.1)
    # ------------------------------------------------------------------

    @hookimpl
    async def before_compaction(self, channel, compacted_ids: list[int], cut: int) -> None:
        """Fork the compaction-time record while the prefix is still live.

        With forks on for "consolidation", runs the record fork over the
        window's dropped prefix (``cut``) and stashes it under
        ``(channel_id, max(compacted_ids))`` for the on_compaction task.
        Runs inline (the hook is awaited before replace_with_summary), so
        it costs the response path one more generation. Fail-soft: any
        miss leaves no stash and the task makes the standalone call.
        """
        if not compacted_ids:
            return
        agent = fork_agent(self.pm, "consolidation")
        if agent is None:
            return
        range_end = max(compacted_ids)
        attribution_token = set_attribution(
            stage="consolidation", channel_id=channel.id, trigger="compaction"
        )
        try:
            db = await self._ensure_schema()
            if db is None:
                return
            watermark = await self._get_watermark(db, channel.id)
            if range_end <= watermark:
                return
            rows = await fetch_range_rows(db, channel.id, watermark, range_end)
            if not dialog_from_rows(rows):
                return
            record = await self._fork_record(agent, channel, rows, cut)
            if record is not None:
                self._stash_put((channel.id, range_end), (watermark, record))
        except Exception:
            logger.warning(
                "consolidation fork at compaction failed", exc_info=True,
                extra={"channel_id": channel.id},
            )
        finally:
            reset_attribution(attribution_token)

    def _stash_put(self, key: tuple[str, int], value: tuple[int, dict]) -> None:
        """Stash a fork record, evicting the oldest past the bound."""
        self._fork_stash[key] = value
        self._fork_stash.move_to_end(key)
        while len(self._fork_stash) > FORK_STASH_MAXSIZE:
            self._fork_stash.popitem(last=False)

    async def _fork_record(self, agent, channel, rows: list[tuple], cut: int | None) -> dict | None:
        """The memory record from a fork of ``channel``'s window, or None.

        Forks only when every dialog row of the range is in the window the
        fork sees (``messages[:cut]``): the record must be about rows the
        model can read. Summary rows are skipped (summaries are never
        consolidated, and a live window's summary has no rowid). Any fork
        failure or unusable reply is logged and returns None.
        """
        conv = getattr(channel, "conversation", None)
        if conv is None:
            return None
        prefix = conv.messages if cut is None else conv.messages[:cut]
        dialog_ids = {row_id for row_id, _msg, mtype in rows if mtype == "message"}
        if not dialog_ids <= window_db_ids(prefix):
            logger.debug(
                "consolidation range not in the window; using the standalone call",
                extra={"channel_id": channel.id},
            )
            return None
        instruction = (
            self._consolidation_fork_prompt + "\n\n"
            + _fork_range_sentence(prefix, dialog_ids)
        )
        try:
            result = await agent.fork(channel, instruction, cut=cut)
            return _record_from_text(result.text)
        except Exception:
            logger.warning(
                "consolidation fork failed; using the standalone call",
                exc_info=True, extra={"channel_id": channel.id},
            )
            return None

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
        self._spawn(self._consolidate_range(channel.id, max(compacted_ids), "compaction"))

    async def _on_channel_idle(self, channel_id: str) -> None:
        """Trigger B: consolidate a quiet channel's un-consolidated tail.

        Called by the idle tracker once the channel has been quiet for
        ``idle_consolidate_after`` seconds, and again each period while the
        quiet lasts. Never enqueues anything: the work is a plugin-owned task.
        """
        # A consolidation from an earlier firing is still running.
        if channel_id in self._idle_inflight:
            return
        try:
            db = await self._ensure_schema()
            if db is None:
                # The next repeat retries.
                return
            async with db.execute(
                "SELECT MAX(id) FROM message_log WHERE channel_id = ?",
                (channel_id,),
            ) as cursor:
                row = await cursor.fetchone()
            max_id = row[0] if row is not None else None
            if max_id is None:
                return
            watermark = await self._get_watermark(db, channel_id)
        except Exception:
            logger.warning(
                "idle consolidation scan failed for %s", channel_id, exc_info=True
            )
            return
        if max_id <= watermark:
            return
        self._idle_inflight.add(channel_id)
        task = self._spawn(self._consolidate_range(channel_id, max_id, "idle"))
        # Free the channel for the next firing however the task ends.
        task.add_done_callback(lambda _t: self._idle_inflight.discard(channel_id))

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
    # Phase 1b background tasks: message_fts backfill and retention job
    # ------------------------------------------------------------------

    async def _backfill_message_fts(self) -> None:
        """Index message_log rows absent from message_fts.

        Uses NOT IN (SELECT rowid FROM message_fts) — crash-safe: a rerun
        after a mid-backfill crash picks up only missing rows without hitting
        FTS5 rowid uniqueness on re-insert (verified empirically, SQLite 3.47.1).
        """
        try:
            db = await self._ensure_schema()
            if db is None:
                return
            async with db.execute(
                "SELECT id, message FROM message_log "
                "WHERE id NOT IN (SELECT rowid FROM message_fts)"
            ) as cursor:
                rows = await cursor.fetchall()
            if not rows:
                return
            for row_id, msg_json in rows:
                try:
                    content = json.loads(msg_json).get("content") or ""
                except Exception:
                    content = ""
                await db.execute(
                    "INSERT INTO message_fts(rowid, content_text) VALUES (?, ?)",
                    (row_id, content if isinstance(content, str) else ""),
                )
            await db.commit()
            logger.info("message_fts: backfilled %d rows", len(rows))
        except Exception:
            logger.warning("message_fts backfill failed", exc_info=True)

    def _now(self) -> float:
        """Wall time from the kernel clock, or ``time.time()`` without one."""
        return self._clock.now() if self._clock is not None else time.time()

    async def _run_retention_if_due(self) -> None:
        """Run the retention job if the interval has elapsed since last_run.

        Persisting last_run before the job prevents crash-looping daemons
        from re-running it on every restart (trap #5). It also preserves the
        zero-traffic guarantee: daemons idle longer than the interval run
        the job on next start. Failures are logged, not raised.
        """
        try:
            db = await self._ensure_schema()
            if db is None:
                return
            interval = self._retention_interval

            async with db.execute(
                "SELECT last_run FROM retention_meta LIMIT 1"
            ) as cursor:
                row = await cursor.fetchone()

            now = self._now()
            last_run = row[0] if row is not None else 0.0

            if now - last_run < interval:
                logger.debug("retention: skipping (interval not elapsed)")
                return

            # Update last_run before running (prevents double-run on crash restart)
            if row is None:
                await db.execute(
                    "INSERT INTO retention_meta (last_run) VALUES (?)", (now,)
                )
            else:
                await db.execute(
                    "UPDATE retention_meta SET last_run = ?", (now,)
                )
            await db.commit()

            from corvidae_cognition.retention import run_retention_job
            await run_retention_job(self, now=now)
        except Exception:
            logger.warning("retention startup trigger failed", exc_info=True)

    async def _arm_retention(self) -> None:
        """Arm the timer for the next retention due time. Never raises.

        The one place the retention timer is created. Reads last_run and arms
        at ``last_run + interval``, floored at ``now + RETENTION_DB_RETRY_SECONDS``
        so a last_run that failed to advance cannot arm in the past and loop
        at zero delay. If last_run cannot be read, re-arms itself shortly.
        """
        if self._clock is None:
            return
        try:
            db = await self._ensure_schema()
            if db is None:
                # Expected at boot: memory starts before persistence opens
                # the DB (on_start is LIFO). One plain line, no traceback (#175).
                logger.warning(
                    "retention: persistence DB not open yet; arming the "
                    "retention timer deferred, re-arming in %ss",
                    RETENTION_DB_RETRY_SECONDS,
                )
                self._schedule_retention_rearm()
                return
            async with db.execute(
                "SELECT last_run FROM retention_meta LIMIT 1"
            ) as cursor:
                row = await cursor.fetchone()
            last_run = row[0] if row is not None else 0.0
            when = max(
                last_run + self._retention_interval,
                self._now() + RETENTION_DB_RETRY_SECONDS,
            )
            self._retention_timer = self._clock.call_at(
                when, self._retention_due, name="memory.retention"
            )
        except Exception:
            logger.warning(
                "retention: cannot read retention_meta; re-arming in %ss",
                RETENTION_DB_RETRY_SECONDS,
                exc_info=True,
            )
            self._schedule_retention_rearm()

    def _schedule_retention_rearm(self) -> None:
        """Retry _arm_retention after RETENTION_DB_RETRY_SECONDS."""
        self._retention_timer = self._clock.call_later(
            RETENTION_DB_RETRY_SECONDS,
            self._arm_retention,
            name="memory.retention.rearm",
        )

    async def _retention_due(self) -> None:
        """Retention is due: run when the system is quiet, or at max_delay."""
        if self._clock is None or self._idle is None:
            return
        self._retention_pending = True
        # Prefer a quiet moment; the deadline bounds the wait under traffic.
        self._retention_watch = self._idle.watch_system(
            RETENTION_QUIET_SECONDS,
            self._run_retention_now,
            name="memory.retention.quiet",
        )
        self._retention_deadline = self._clock.call_later(
            self._retention_max_delay,
            self._run_retention_now,
            name="memory.retention.max_delay",
        )

    async def _run_retention_now(self) -> None:
        """Start the retention cycle once, from whichever trigger fires first."""
        if not self._retention_pending:
            return
        self._retention_pending = False
        # Cancelling the watch sets its flag, so it cannot re-arm even when
        # this runs inside its own callback; a started Timer's cancel is a no-op.
        if self._retention_watch is not None:
            self._retention_watch.cancel()
            self._retention_watch = None
        if self._retention_deadline is not None:
            self._retention_deadline.cancel()
            self._retention_deadline = None
        # Plugin-owned task: the clock callback returns at once, on_stop cancels it.
        self._spawn(self._retention_cycle())

    async def _retention_cycle(self) -> None:
        """Run retention if due, then arm the next due time."""
        # Logs its own failures, so the arm below always runs.
        await self._run_retention_if_due()
        await self._arm_retention()

    # ------------------------------------------------------------------
    # The consolidation task (single code path for both triggers)
    # ------------------------------------------------------------------

    async def _consolidate_range(self, channel_id: str, range_end: int, trigger: str) -> None:
        """Consolidate message_log rows (watermark, range_end] into one record.

        Overlap safety (trap #6): both triggers WILL race to the same
        range. The watermark is read optimistically before the LLM work,
        then re-checked under the DB lock in the same atomic section as the
        insert + watermark advance; any concurrent advance discards this
        task's record.
        """
        attribution_token = set_attribution(
            stage="consolidation", channel_id=channel_id, trigger=trigger
        )
        try:
            db = await self._ensure_schema()
            if db is None:
                return
            watermark = await self._get_watermark(db, channel_id)
            if range_end <= watermark:
                return

            # Fetch the working range and keep only real dialog.
            rows = await fetch_range_rows(db, channel_id, watermark, range_end)
            dialog = dialog_from_rows(rows)

            if not dialog:
                # Pure CONTEXT/system segments advance the watermark
                # without producing a record.
                async with self._db_lock:
                    current = await self._get_watermark(db, channel_id)
                    if current == watermark:
                        await self._set_watermark(db, channel_id, range_end)
                        await db.commit()
                return

            # The first-person record: a fork of the live window when one
            # is available (stashed at compaction, or run now for idle),
            # else one standalone background-model call.
            record = await self._consolidation_record(
                channel_id, rows, dialog, watermark, range_end, trigger
            )

            # The appraisal-driven prior (WP2.5) scores over the consolidated
            # message-id range; pass the real channel when the registry has
            # one, else a stub with empty runtime_overrides (sanctioned by
            # WP2.3's duck-typed resolver) so config/default surfaces apply.
            msg_id_range = (rows[0][0], rows[-1][0])
            registry = self.pm.get_plugin("registry")
            channel = registry.get(channel_id) if registry is not None else None
            if channel is None:
                channel = _StubChannel(channel_id)
            importance = await self.importance_prior.score(
                dialog, msg_id_range=msg_id_range, channel=channel
            )

            # Valence: mean stage-2 valence over the range (NULL when none) —
            # the §4.9-classifier deletion made real: affect is the appraisal
            # already computed. Fail-soft when appraisal is absent.
            valence: float | None = None
            appraisal_plugin = self.pm.get_plugin("appraisal")
            if appraisal_plugin is not None:
                try:
                    valence = await appraisal_plugin.mean_valence(msg_id_range)
                except Exception:
                    logger.warning(
                        "mean-valence lookup failed; storing NULL valence",
                        exc_info=True, extra={"channel_id": channel_id},
                    )

            # Date the record from when its source conversation happened,
            # not from when consolidation ran — imported history would
            # otherwise be dated at import time (issue #52).
            created_at = await source_timestamp(db, rows[-1][0])

            # Embed the summary; failure stores embedded=0 for later
            # backfill (trap #10 — degrade, never block).
            embedding: list[float] | None = None
            if self._embedding_enabled():
                try:
                    client = self._llm().get_client("embedding")
                    embedding = (await client.embed([record["summary"]], kind="document"))[0]
                except Exception:
                    logger.warning(
                        "memory embedding failed; storing record with "
                        "embedded=0 for backfill", exc_info=True,
                    )

            # Atomic commit section: insert + watermark advance together.
            async with self._db_lock:
                current = await self._get_watermark(db, channel_id)
                if current != watermark:
                    # Design (OPT-1a-2): a concurrent consolidation task won
                    # the compare-and-set and advanced the watermark.  Discarding
                    # here is intentional — attempting to consolidate the
                    # non-overlapping tail would require re-fetching messages and
                    # risk producing spurious records.  Any rows beyond the
                    # winner's range_end remain unconsolidated, but the idle
                    # trigger provides eventual coverage: the next idle cycle
                    # will see the updated watermark and pick them up.
                    logger.debug(
                        "watermark moved during consolidation; discarding",
                        extra={"channel_id": channel_id},
                    )
                    return
                cursor = await db.execute(
                    "INSERT INTO memory (channel_id, created_at, summary, "
                    "importance, valence, topic_tags, participants, "
                    "msg_id_start, msg_id_end, embedded) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        channel_id,
                        created_at,
                        record["summary"],
                        importance,
                        valence,
                        json.dumps(record["topic_tags"]),
                        json.dumps(record["participants"]),
                        rows[0][0],
                        rows[-1][0],
                        1 if embedding is not None else 0,
                    ),
                )
                memory_id = cursor.lastrowid

                # WP1b.2 — near-dup check: query BEFORE inserting the new vec row
                # so the new record is naturally excluded from the results.
                # Candidates come from this channel's retrieval scope (GitHub
                # issue #58), but a record on another channel merges only if
                # that channel's scope reaches exactly the same channels:
                # the survivor must stay visible to every channel that could
                # see the superseded record, and groups are not transitive.
                # Skip entirely when embedding is unavailable or the new record
                # has no vector (embed failed — no vector to compare).
                dup_old_id: int | None = None
                dup_cross_channel = False
                own_scope = self._channel_scope(channel_id)
                if (
                    embedding is not None
                    and self._vec_available
                    and not self._encoder_mismatch
                ):
                    try:
                        import sqlite_vec as _sv
                        scope_sql, scope_params = own_scope.sql("m.channel_id")
                        async with db.execute(
                            "SELECT v.memory_id, (1.0 - v.distance) as similarity, "
                            "m.channel_id "
                            "FROM (SELECT memory_id, distance FROM memory_vec "
                            "      WHERE embedding MATCH ? AND k = ?) AS v "
                            "JOIN memory m ON m.id = v.memory_id "
                            f"WHERE {scope_sql} AND m.indexed = 1 "
                            "ORDER BY v.distance",
                            (
                                _sv.serialize_float32(embedding),
                                VEC_OVERFETCH * 2,
                                *scope_params,
                            ),
                        ) as dup_cursor:
                            dup_rows = await dup_cursor.fetchall()
                        # Nearest first: take the first candidate over the
                        # threshold whose owner has the same reach.
                        for dup_candidate_id, dup_similarity, dup_channel in dup_rows:
                            if dup_similarity < self._dup_threshold:
                                break
                            if dup_channel == channel_id or own_scope.same_reach(
                                self._channel_scope(dup_channel)
                            ):
                                dup_old_id = dup_candidate_id
                                dup_cross_channel = dup_channel != channel_id
                                break
                    except Exception:
                        logger.debug(
                            "dup detection query failed; skipping for this record",
                            exc_info=True,
                        )

                if embedding is not None and self._vec_available:
                    import sqlite_vec
                    await db.execute(
                        "INSERT INTO memory_vec (memory_id, embedding) VALUES (?, ?)",
                        (memory_id, sqlite_vec.serialize_float32(embedding)),
                    )

                # If a near-dup was found, fold the old record into the new one.
                if dup_old_id is not None:
                    async with db.execute(
                        "SELECT retrieval_count, last_retrieved_at, importance, "
                        "msg_id_start, msg_id_end, created_at "
                        "FROM memory WHERE id = ?",
                        (dup_old_id,),
                    ) as old_cursor:
                        old_row = await old_cursor.fetchone()
                    if old_row is not None:
                        (old_rc, old_lra, old_importance,
                         old_start, old_end, old_created) = old_row
                        # Fold stats into the new record
                        inherited_rc = (old_rc or 0)  # old record's count to fold in
                        new_importance = max(importance, old_importance or 0.0)
                        new_lra = old_lra  # new record has None; old may have a value
                        # Fold the message range only within one channel:
                        # message_log ids are global, and recall_raw and
                        # /redact read a record's range on its own channel.
                        # A superseded record keeps its own range.
                        if dup_cross_channel:
                            new_start, new_end = rows[0][0], rows[-1][0]
                        else:
                            new_start = min(old_start, rows[0][0])
                            new_end = max(old_end, rows[-1][0])
                        # The merged record spans both conversations; date it
                        # from the later one. Imported history has newer ids
                        # but older timestamps, so this is not new_end's date.
                        new_created = max(created_at, old_created)
                        await db.execute(
                            "UPDATE memory SET "
                            "retrieval_count = retrieval_count + ?, "
                            "importance = ?, "
                            "last_retrieved_at = CASE "
                            "  WHEN last_retrieved_at IS NULL THEN ? "
                            "  WHEN ? IS NULL THEN last_retrieved_at "
                            "  ELSE MAX(last_retrieved_at, ?) "
                            "END, "
                            "msg_id_start = ?, msg_id_end = ?, created_at = ? "
                            "WHERE id = ?",
                            (inherited_rc, new_importance,
                             new_lra, new_lra, new_lra,
                             new_start, new_end, new_created, memory_id),
                        )
                        # Supersede the old record
                        await db.execute(
                            "UPDATE memory SET superseded_by = ?, indexed = 0, embedded = 0 "
                            "WHERE id = ?",
                            (memory_id, dup_old_id),
                        )
                        if self._vec_available:
                            await db.execute(
                                "DELETE FROM memory_vec WHERE memory_id = ?",
                                (dup_old_id,),
                            )
                        logger.info(
                            "memory dedup: superseded %d with %d (similarity=%.3f)",
                            dup_old_id, memory_id, dup_similarity,
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
                    "trigger": trigger,
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

    async def _consolidation_record(
        self, channel_id: str, rows: list[tuple], dialog: list[dict],
        watermark: int, range_end: int, trigger: str,
    ) -> dict:
        """Produce the record for (watermark, range_end].

        Compaction trigger: the record before_compaction stashed, if it was
        built against this same watermark (otherwise it covers a different
        range and is dropped). Idle trigger: a fork of the channel's live
        window when forks are on and the range is in it. Everything else,
        and every miss, is the standalone call.
        """
        record: dict | None = None
        if trigger == "compaction":
            stashed = self._fork_stash.pop((channel_id, range_end), None)
            if stashed is not None and stashed[0] == watermark:
                record = stashed[1]
        elif trigger == "idle":
            agent = fork_agent(self.pm, "consolidation")
            registry = self.pm.get_plugin("registry")
            channel = registry.get(channel_id) if registry is not None else None
            if agent is not None and channel is not None:
                record = await self._fork_record(agent, channel, rows, None)
        if record is not None:
            return record
        token = set_attribution(mode="standalone")
        try:
            return await self._summarize_range(dialog)
        finally:
            reset_attribution(token)

    async def _summarize_range(self, dialog: list[dict]) -> dict:
        """One llm.background call producing the first-person record.

        Returns {"summary": str, "topic_tags": list, "participants": list}.
        A malformed reply is retried once; if both attempts are unusable it
        degrades to the raw transcript rather than losing the range.
        """
        client = self._llm().get_client("background")
        messages = [
            {"role": "system", "content": self._consolidation_prompt},
            {"role": "user", "content": _dialog_transcript(dialog)},
        ]
        for attempt in range(2):
            response = await client.chat(messages)
            text = response["choices"][0]["message"]["content"]
            try:
                return _record_from_text(text)
            except ValueError as exc:
                logger.warning(
                    "consolidation reply unusable (attempt %d/2): %s",
                    attempt + 1, exc,
                )
        # The call is already paid for; keep the range as its raw transcript.
        return {
            "summary": _dialog_transcript(dialog),
            "topic_tags": [],
            "participants": [],
        }

    # ------------------------------------------------------------------
    # Retrieval read path (§3.1)
    # ------------------------------------------------------------------

    @hookimpl
    async def before_agent_turn(self, channel, trace_id, meta) -> None:
        """Retrieve memories for the inbound user text and admit them.

        Runs only when the window tail is a plain user MESSAGE — retrieval
        is skipped entirely on notification-triggered turns. Everything
        here is fail-soft: retrieval trouble must never break the turn.

        ``trace_id``: when present, the retrieval profile is written
        into retrieval_log.trace_id and copied into exchange_log via
        the OutcomeLogPlugin's update_exchange (retrieval_top_score,
        retrieval_top_similarity, retrieval_hit_count).
        """
        try:
            conv = getattr(channel, "conversation", None)
            if conv is None or not conv.messages:
                return
            tail = conv.messages[-1]
            from corvidae.context import MessageType
            if (
                tail.get("role") != "user"
                or tail.get("_message_type", MessageType.MESSAGE) != MessageType.MESSAGE
                or not isinstance(tail.get("content"), str)
                or not tail["content"].strip()
            ):
                return

            db = await self._ensure_schema()
            if db is None:
                return

            candidates, degraded = await self.retrieve(channel.id, tail["content"])

            # Drop weak-banded candidates unless nothing else matched.
            non_weak = [c for c in candidates if c["band"] != "weak"]
            selected = non_weak or candidates

            # One line per memory: "[band] (age) summary".
            lines = [
                f"[{c['band']}] ({_format_age(c['age_seconds'])}) {c['summary']}"
                for c in selected
            ]

            # Admit through the funnel — never append CONTEXT directly
            # (trap #8); the funnel dedupes against the window and budgets.
            admitted_ids: list[int] = []
            if lines:
                funnel = self.pm.get_plugin("funnel")
                if funnel is None:
                    logger.warning(
                        "funnel plugin not registered; memory retrieval "
                        "cannot admit context"
                    )
                else:
                    admitted_lines = await funnel.admit(channel, conv, "memory", lines)
                    if admitted_lines:
                        admitted_ids = [
                            c["id"] for c, line in zip(selected, lines)
                            if line in admitted_lines
                        ]

            # Access stats — the write half of reconsolidation — move only
            # for memories that actually entered the window.
            if admitted_ids:
                now = time.time()
                placeholders = ",".join("?" for _ in admitted_ids)
                await db.execute(
                    f"UPDATE memory SET retrieval_count = retrieval_count + 1, "
                    f"last_retrieved_at = ? WHERE id IN ({placeholders})",
                    (now, *admitted_ids),
                )

            # Persist the retrieval profile (§3.7's day-one raw stream).
            # Both values describe the top-ranked candidate: the decayed
            # score it was ranked by, and the raw similarity that
            # band_on: similarity compares against the band thresholds.
            top_score = candidates[0]["score"] if candidates else None
            top_similarity = candidates[0]["similarity"] if candidates else None
            hit_count = len(candidates)
            await db.execute(
                "INSERT INTO retrieval_log (ts, channel_id, trace_id, top_score, "
                "top_similarity, hit_count, admitted_count, degraded_to_fts) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    time.time(),
                    channel.id,
                    trace_id,
                    top_score,
                    top_similarity,
                    hit_count,
                    len(admitted_ids),
                    1 if degraded else 0,
                ),
            )
            await db.commit()

            # Copy the retrieval profile into exchange_log under the key —
            # best-effort; OutcomeLogPlugin may not be registered, or the
            # row may not exist yet.
            if trace_id is not None:
                outcome_log = self.pm.get_plugin("outcome_log")
                if outcome_log is not None:
                    try:
                        await outcome_log.update_exchange(
                            trace_id,
                            retrieval_top_score=top_score,
                            retrieval_top_similarity=top_similarity,
                            retrieval_hit_count=hit_count,
                        )
                    except Exception:
                        logger.warning(
                            "retrieval profile -> exchange_log copy failed",
                            exc_info=True, extra={"channel_id": channel.id},
                        )
        except Exception:
            logger.warning(
                "memory retrieval failed; turn proceeds without it",
                exc_info=True, extra={"channel_id": channel.id},
            )

    async def retrieve(
        self, channel_id: str, text: str, k: int | None = None
    ) -> tuple[list[dict], bool]:
        """Score memory candidates for a query text.

        Vector KNN over memory_vec when available; FTS5 keyword search
        otherwise or when the encoder fails (trap #10). Candidates are
        channel-compartmentalized: same channel, or same configured group.

        Returns:
            (candidates sorted by score descending, degraded_to_fts).
            Each candidate: {id, summary, created_at, similarity, score,
            band, age_seconds}.
        """
        db = await self._ensure_schema()
        if db is None:
            return [], False
        k = k or self._retrieval_k
        scope = self._channel_scope(channel_id)
        now = time.time()

        rows: list[tuple] = []
        degraded = False
        if self._vec_available and self._embedding_enabled():
            try:
                client = self._llm().get_client("embedding")
                query_vector = (await client.embed([text], kind="query"))[0]
                rows = await self._vec_candidates(db, query_vector, scope, k)
            except Exception:
                logger.warning(
                    "retrieval embedding failed; degrading to FTS5",
                    exc_info=True,
                )
                degraded = True
        else:
            degraded = True

        if degraded:
            rows = await self._fts_candidates(db, text, scope, k)

        candidates = []
        for memory_id, summary, created_at, last_retrieved_at, similarity in rows:
            # Displayed age is always when the memory was formed.
            age_seconds = max(0.0, now - created_at)
            # Ranking age: from formation, or (recency_from: last_used)
            # from the latest use, matching retention's last_activity.
            if self._recency_from == "last_used":
                last_activity = max(created_at, last_retrieved_at or 0.0)
            else:
                last_activity = created_at
            ranking_age = max(0.0, now - last_activity)
            # Relevance × recency decay; participant-match arrives with
            # Phase 4 (channel scope is the proxy until then, §3.1).
            score = similarity * math.exp(
                -(ranking_age / 86400.0) / self._half_life_days
            )
            # The band says how relevant a memory is: from the decayed
            # score (default) or, with band_on: similarity, from relevance
            # alone so an old, apt memory is not presented as a guess.
            band_value = similarity if self._band_on == "similarity" else score
            if band_value >= self._band_strong:
                band = "strong"
            elif band_value >= self._band_moderate:
                band = "moderate"
            else:
                band = "weak"
            candidates.append({
                "id": memory_id,
                "summary": summary,
                "created_at": created_at,
                "similarity": similarity,
                "score": score,
                "band": band,
                "age_seconds": age_seconds,
            })
        candidates.sort(key=lambda c: c["score"], reverse=True)
        return candidates[:k], degraded

    async def recent(self, channel_id: str, limit: int) -> list[dict]:
        """The newest memories of the other channels in ``channel_id``'s scope.

        Records owned by ``channel_id`` itself are left out, so a channel
        reviewing its group (the heartbeat) doesn't review its own earlier
        reviews. Same status filters as retrieval: indexed and not redacted.
        Returns up to ``limit`` dicts ``{summary, channel_id, age_seconds}``,
        newest first, aged on the kernel clock. Raises RuntimeError while
        the memory DB is not open.
        """
        db = await self._ensure_schema()
        if db is None:
            raise RuntimeError("memory DB is not open yet")
        scope_sql, scope_params = self._channel_scope(channel_id).sql("channel_id")
        async with db.execute(
            "SELECT summary, channel_id, created_at FROM memory "
            f"WHERE indexed = 1 AND redacted = 0 AND {scope_sql} AND channel_id != ? "
            "ORDER BY created_at DESC, id DESC LIMIT ?",
            (*scope_params, channel_id, limit),
        ) as cursor:
            rows = await cursor.fetchall()
        now = self._now()
        return [
            {"summary": summary, "channel_id": owner, "age_seconds": max(0.0, now - created_at)}
            for summary, owner, created_at in rows
        ]

    def _channel_scope(self, channel_id: str) -> ChannelScope:
        """The channels whose memories this channel may retrieve.

        The channel itself (by equality), plus every member of any
        configured group it belongs to (memory.channel_groups, §3.1
        compartmentalization). A group applies when the caller equals a
        literal member or matches a '*' / '?' pattern member; its literal
        members join the scope as ids and its patterns as patterns, so
        channels created at runtime are covered (GitHub issue #50).
        """
        ids = {channel_id}
        patterns: list[str] = []
        for members in self._channel_groups.values():
            # Does this group admit the caller?
            if not any(
                fnmatchcase(channel_id, m) if _is_pattern(m) else channel_id == m
                for m in members
            ):
                continue
            # Merge the group's members into the scope, deduplicated.
            for m in members:
                if not _is_pattern(m):
                    ids.add(m)
                elif m not in patterns:
                    patterns.append(m)
        return ChannelScope(frozenset(ids), tuple(patterns))

    async def _vec_candidates(
        self, db: aiosqlite.Connection, query_vector: list[float],
        scope: ChannelScope, k: int,
    ) -> list[tuple]:
        """Exact-KNN candidates:
        (id, summary, created_at, last_retrieved_at, cosine_similarity).

        sqlite-vec is brute-force exact KNN (no ANN index), so the KNN scan
        costs the same regardless of k; we over-fetch and filter by scope
        and flags in the join.
        """
        import sqlite_vec
        scope_sql, scope_params = scope.sql("m.channel_id")
        async with db.execute(
            "SELECT m.id, m.summary, m.created_at, m.last_retrieved_at, v.distance "
            "FROM (SELECT memory_id, distance FROM memory_vec "
            "      WHERE embedding MATCH ? AND k = ?) AS v "
            "JOIN memory m ON m.id = v.memory_id "
            f"WHERE m.indexed = 1 AND m.redacted = 0 "
            f"AND {scope_sql} "
            "ORDER BY v.distance",
            (
                sqlite_vec.serialize_float32(query_vector),
                k * VEC_OVERFETCH,
                *scope_params,
            ),
        ) as cursor:
            rows = await cursor.fetchall()
        # vec0 cosine distance = 1 - cosine similarity.
        return [
            (memory_id, summary, created_at, last_retrieved_at, 1.0 - distance)
            for memory_id, summary, created_at, last_retrieved_at, distance in rows
        ]

    async def _fts_candidates(
        self, db: aiosqlite.Connection, text: str, scope: ChannelScope, k: int
    ) -> list[tuple]:
        """FTS5 keyword candidates with a fixed similarity proxy:
        (id, summary, created_at, last_retrieved_at, FTS_SIMILARITY_PROXY)."""
        match = _fts_match_query(text)
        if match is None:
            return []
        scope_sql, scope_params = scope.sql("m.channel_id")
        async with db.execute(
            "SELECT m.id, m.summary, m.created_at, m.last_retrieved_at "
            "FROM memory_fts JOIN memory m ON m.id = memory_fts.rowid "
            f"WHERE memory_fts MATCH ? AND m.indexed = 1 AND m.redacted = 0 "
            f"AND {scope_sql} "
            "ORDER BY rank LIMIT ?",
            (match, *scope_params, k),
        ) as cursor:
            rows = await cursor.fetchall()
        return [
            (memory_id, summary, created_at, last_retrieved_at, FTS_SIMILARITY_PROXY)
            for memory_id, summary, created_at, last_retrieved_at in rows
        ]

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
        # Schema upgrade: pre-existing two-column tables (before this feature)
        # lack document_prefix/query_prefix. Add them with DEFAULT '' so they
        # compare equal to prefix-less config, and mismatch against configured
        # prefixes routes through the normal _encoder_mismatch path.
        async with db.execute("PRAGMA table_info(embedding_meta)") as cursor:
            existing_cols = {row[1] for row in await cursor.fetchall()}
        for col in ("document_prefix", "query_prefix"):
            if col not in existing_cols:
                await db.execute(
                    f"ALTER TABLE embedding_meta ADD COLUMN {col} "
                    f"TEXT NOT NULL DEFAULT ''"
                )
        await db.execute(MEMORY_FTS_DDL)
        await db.execute(MEMORY_FTS_INSERT_TRIGGER)
        await db.execute(MEMORY_FTS_UPDATE_TRIGGER)
        await db.execute(RETRIEVAL_LOG_DDL)
        # Schema upgrade: retrieval_log tables from before top_similarity
        # (#66) gain it as a nullable column; old rows read NULL.
        await add_column_if_missing(db, "retrieval_log", "top_similarity", "REAL")
        # Phase 1b: message_fts and retention_meta
        await db.execute(MESSAGE_FTS_DDL)
        await db.execute(MESSAGE_FTS_INSERT_TRIGGER)
        await db.execute(MESSAGE_FTS_UPDATE_TRIGGER)
        await db.execute(RETENTION_META_DDL)

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

        A stored encoder, dimensions, or prefix value differing from config is
        an operator ERROR — mixing encoders or prefixes silently corrupts
        similarity. Config revert is the only remediation; here we refuse to
        mix and disable embedding.
        """
        doc_prefix = (self._embedding_cfg or {}).get("document_prefix", "") or ""
        qry_prefix = (self._embedding_cfg or {}).get("query_prefix", "") or ""

        async with db.execute(
            "SELECT encoder, dimensions, document_prefix, query_prefix "
            "FROM embedding_meta LIMIT 1"
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            await db.execute(
                "INSERT INTO embedding_meta "
                "(encoder, dimensions, document_prefix, query_prefix) "
                "VALUES (?, ?, ?, ?)",
                (encoder, dimensions, doc_prefix, qry_prefix),
            )
            return
        # Normalize NULL (from ALTER TABLE DEFAULT '') to '' for comparison.
        stored = (row[0], row[1], row[2] or "", row[3] or "")
        if stored != (encoder, dimensions, doc_prefix, qry_prefix):
            self._encoder_mismatch = True
            logger.error(
                "embedding_meta records %s/%dd/%r/%r but config wants %s/%dd/%r/%r; "
                "embeddings disabled — revert config to restore vector retrieval",
                stored[0], stored[1], stored[2], stored[3],
                encoder, dimensions, doc_prefix, qry_prefix,
            )
