"""MemoryPlugin — autobiographical memory for the Corvidae agent daemon.

Compaction becomes memory formation: dialog leaving the active window is
consolidated into first-person memory records (SQLite + sqlite-vec + FTS5),
and inbound messages trigger retrieval whose results enter the window
through the context-admission funnel (bootstrap-mapping §3.1).

This module owns the memory schema:
  - ``memory``: one row per consolidated first-person record, linked back
    to the raw dialog by a ``message_log`` id range. Rows are never
    deleted — demotion/redaction (Phase 1b) mutate columns.
    Each row is stamped with how it was made (issue #299): ``prompt_hash``
    (sha256 of the consolidation instruction text), ``mode`` (``fork`` |
    ``standalone`` | ``direct``) and ``model`` (the model the call was sent
    with), plus ``trigger`` (issue #633: ``compaction`` | ``idle`` |
    ``turn`` | ``eval``, what started the consolidation; NULL for direct
    rows and rows from before the column). All NULL means the row predates stamping
    ("unknown"); ``direct`` rows (goal retirement) were written by no
    consolidation call.
  - ``memory_prompt``: ``hash`` -> instruction text, stored once per hash.
  - ``consolidation_watermark``: per-channel last-consolidated
    ``message_log`` id; what makes overlapping triggers
    (compaction, idle, turn) safe.
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
      consolidate_after_tokens: 4000 # window tokens of un-consolidated
                                     # conversation that start a turn
                                     # consolidation (forks only; positive int)
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
      consolidation_fork_prompt: ... # instruction for forked consolidation
                                     # (agent.forks, #51); static config

Changes to ``idle_consolidate_after``, ``consolidate_after_tokens`` and
``retention.*`` take effect at restart: the idle watch and retention timer
are set when the plugin starts, and ``consolidate_after_tokens`` is read
once at startup.
A channel that never goes quiet is consolidated at turn ends (forks on)
and through compaction.
"""

from __future__ import annotations

import asyncio
import collections
import dataclasses
import hashlib
import json
import logging
import math
import re
import sqlite3
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

import aiosqlite

from corvidae.attribution import reset_attribution, set_attribution
from corvidae.prompts import resolve_prompt
from corvidae.context import ContextWindow, MessageType, count_tokens, message_tokens
from corvidae.fork import (
    ForkRefused,
    ViewInputs,
    build_fork_prompt,
    fork_agent,
    quote,
    snap_cut_forward,
    snap_cut_to_tool_boundary,
    view_agent,
    view_call,
    window_db_ids,
)
from corvidae.hooks import CorvidaePlugin, get_dependency, hookimpl
from corvidae.llm_plugin import parse_llm_config
from corvidae_cognition.funnel import Admission
from corvidae.persistence import (
    add_column_if_missing,
    decode_message_row,
    decodable_summaries,
    parse_message_type,
    view_rows,
)

if TYPE_CHECKING:
    from corvidae.clock import Clock, Timer
    from corvidae.idle import IdleTracker, IdleWatch

logger = logging.getLogger("corvidae.memory")

# Seconds a channel must be inactive before the idle trigger consolidates
# its un-consolidated tail.
DEFAULT_IDLE_CONSOLIDATE_AFTER = 1800.0

# Window tokens of un-consolidated conversation that start a turn consolidation.
DEFAULT_CONSOLIDATE_AFTER_TOKENS: int = 4000

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
# What passive retrieval's query holds besides the unanswered user messages
# (#665): the agent's previous reply (variant A), or nothing (the pre-#665
# query). The first entry is the default.
QUERY_CONTEXT_CHOICES = ("previous_reply", "none")
# Where passive retrieval's candidates come from on the vector path (#666):
# vec KNN and FTS5 fused by reciprocal rank, or vec KNN alone. The first
# entry is the default.
CANDIDATES_CHOICES = ("hybrid", "vector")
# Reciprocal rank fusion constant: a candidate earns 1 / (RRF_C + rank) from
# each list it is ranked in. 60 is the value from the original RRF paper;
# it flattens the head so one list's #1 can't outweigh agreement.
# shortcut: a constant with equal list weights; make it (or per-list
# weights) a memory.retrieval key if the labelled set shows a need.
RRF_C = 60
# Common words the hybrid side's FTS query leaves out (#666), so a chatty
# message can't buy fusion rank on "what" or "the". Matched lowercased.
# shortcut: a small English list; a deployment conversing in another
# language gets no stopwording, so extend it if one shows up.
FTS_STOPWORDS = frozenset("""
a about above after again all also am an and any are as at be because been
before being both but by can could did do does doing done for from had has
have having he her here hers him his how i if in into is it its just me
more most my no not now of off on once only or other our ours out over own
same she should so some such than that the their theirs them then there
these they this those through to too under until up us very was we were
what when where which while who whom why will with would yes you your yours
ok okay oh hi hey thanks thank please
""".split())
# Honest similarity stand-in for FTS5-degraded retrieval, where cosine
# similarity cannot be measured: keyword hits enter at the moderate band.
FTS_SIMILARITY_PROXY = 0.6
# Vec KNN over-fetch factor: candidates are filtered by channel scope and
# indexed/redacted flags after the KNN scan, so fetch more than k.
VEC_OVERFETCH = 4


def _exclude_clause(
    exclude: tuple[str, int] | None,
) -> tuple[str, tuple]:
    """The window-exclusion SQL fragment for a candidate query (#634).

    Leaves out records of ``exclude``'s channel whose range starts at or past
    its ``min_msg_id``. Both conditions: message_log ids are global, so an id
    test alone would also hide other channels' records. Keyed on
    ``msg_id_start`` so a record straddling the window's oldest id stays.
    Returns the bare in-window condition and its parameters, or
    ``("0", ())`` (never true) when there is nothing to exclude. Callers
    negate it in WHERE or select it as a flag (#407).
    """
    if exclude is None:
        return "0", ()
    return "(m.channel_id = ? AND m.msg_id_start >= ?)", exclude


# One scored-pool input row from a candidate search:
# (memory_id, summary, created_at, last_retrieved_at, similarity, in_window).
CandidateRow = tuple[int, str, float, "float | None", float, bool]


@dataclass(frozen=True)
class RetrievalPool:
    """The whole scored pool of one retrieval (#407)."""

    # Not-excluded candidates, score descending: the whole pool, not just k.
    ranked: list[dict]
    # (would-be rank, candidate) for in-window records that would have made
    # the top k had they not been excluded. Vec path only.
    excluded: list[tuple[int, dict]]
    # In-window records the search matched; None when no exclusion applied.
    excluded_in_window: int | None
    degraded: bool
    k: int


# How long (ms) the dedicated memory write connection waits for SQLite's
# write lock before failing the write (the range is then retried).
# shortcut: a module constant, not an agent.yaml key; make it config if a
# deployment with slow disks or long retention runs needs a longer wait.
WRITE_BUSY_TIMEOUT_MS = 5000

# JSON schema for the consolidation reply (#199, #635). Mirrors what
# _consolidation_from_text reads: ``entries`` (each needing a narrative) and
# ``importance`` are required; ``topic_tags`` and ``participants`` are not,
# since the parser tolerates them missing and a grammar that forced the
# arrays would add empty-array noise on small models. ``importance`` is a
# scalar, so that reason doesn't apply: it is required so the rubric call
# can be skipped. Property order is the generation order of grammar
# backends: the rating comes after the narrative. Sent as response_format so
# a model that wants to continue the chat can only return a reply; providers
# that ignore it still get the prompt's JSON instruction.
CONSOLIDATION_SCHEMA = {
    "type": "object",
    "properties": {
        "entries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "narrative": {"type": "string"},
                    "topic_tags": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["narrative"],
            },
        },
        "participants": {"type": "array", "items": {"type": "string"}},
        "importance": {"type": "number"},
    },
    "required": ["entries", "importance"],
}
CONSOLIDATION_EXTRA_BODY = {
    "response_format": {
        "type": "json_schema",
        "json_schema": {"name": "memory_record", "schema": CONSOLIDATION_SCHEMA},
    }
}

# Default instruction for consolidation (GitHub issue #51). With agent.forks
# on, the record is written by a fork of the live window; a range that has
# left the window is written from a rebuilt view of it on the consolidation
# stage's client (#380). Either way this text is appended as a user turn, followed by a
# sentence naming where the range starts. prompts/memory_consolidation_fork.md
# is the documented copy; memory.consolidation_fork_prompt (static config,
# not a runtime tunable) overrides it. The agent's own system prompt is above
# it, so it doesn't need to explain who "I" is.
DEFAULT_CONSOLIDATION_FORK_PROMPT = (
    "[Memory consolidation] Pause the conversation here and write a record "
    "of part of it for your long-term autobiographical memory, in the first "
    "person, as yourself. This record is your private memory, not a message "
    "to anyone on the channel: no one there will read it, so write it for "
    "yourself.\n\n"
    "Narrate this stretch of the conversation from your own point of view, "
    "step by step, in your own words: what was said, what you did, what you "
    "learned, and what was decided or left open. Don't quote anyone "
    "verbatim, but keep facts and opinions exactly as they were expressed, "
    "and who expressed each (names, numbers, dates, identifiers, "
    "preferences, commitments, corrections), so you can recall them "
    "accurately later. Keep what you were told apart from what you "
    "inferred or speculated (\"so-and-so told me...\", \"I guessed "
    "that...\"). Say what you did with tools and what it told you, not "
    "their raw output. Bracketed CONTEXT blocks are things you recalled or "
    "were shown, not new events; don't record them as if they happened now. "
    "Skip pleasantries and chatter.\n\n"
    "Write it as separate entries, one for each distinct stretch or topic, "
    "in order. Each entry is a paragraph at most and must make sense read "
    "on its own, months from now, with nothing else: name the people, "
    "projects and things involved rather than \"he\" or \"it\". Don't "
    "attribute a fact from one entry's topic to another's.\n\n"
    "Then rate how important this stretch is to remember long-term, from 0 "
    "(routine) to 1 (essential). Do not call tools.\n\n"
    "Respond with a single JSON object, nothing else:\n"
    "{\n"
    '  "entries": [\n'
    '    {"narrative": "<first-person narrative of one stretch or topic>",\n'
    '     "topic_tags": ["<short topic tag>", "..."]}\n'
    "  ],\n"
    '  "participants": ["<sender name>", "..."],\n'
    '  "importance": "<number from 0 to 1>"\n'
    "}"
)

# How a consolidation range gets its record (MemoryPlugin._live_record).
LiveOutcome = Literal["record", "skip", "catch_up"]

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

# JSON schema for the rubric prior (#199). Mirrors RubricPrior.score, which
# reads a float ``importance``. Providers that ignore response_format still
# get the prompt's JSON instruction.
RUBRIC_SCHEMA = {
    "type": "object",
    "properties": {"importance": {"type": "number"}},
    "required": ["importance"],
}
RUBRIC_EXTRA_BODY = {
    "response_format": {
        "type": "json_schema",
        "json_schema": {"name": "importance_rating", "schema": RUBRIC_SCHEMA},
    }
}


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


@dataclass(frozen=True)
class ConsolidationEntry:
    """One narrative entry: becomes one memory row."""

    narrative: str              # stripped, non-empty; stored as memory.summary
    topic_tags: tuple[str, ...]


@dataclass(frozen=True)
class Consolidation:
    """A usable consolidation reply plus how it was made."""

    entries: tuple[ConsolidationEntry, ...]   # >= 1, reply order
    participants: tuple[str, ...]             # shared by every entry
    importance: float | None                  # the reply's rating; None = absent
    stamp: RecordStamp                        # trigger still None; _commit sets it


def _string_tuple(value: object, key: str) -> tuple[str, ...]:
    """Read ``value`` (absent, None, a string or a list) as a tuple of strings.

    A lone string reads as a one-item list (a small model's "participants":
    "kestrel"). Raises ValueError naming ``key`` for anything else, rather
    than escaping as a TypeError; the caller decides what that costs (a
    malformed entry is dropped, a malformed participants reads as none).
    """
    # Absent and null both mean "none given".
    if value is None:
        return ()
    # A bare string is one item, not a sequence of characters.
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, list):
        raise ValueError(
            f"{key} must be a list of strings or a string, "
            f"got {type(value).__name__}"
        )
    return tuple(str(v) for v in value)


def _coerce_importance(value: object) -> float | None:
    """The reply's importance as a finite float in [0, 1], else None (absent).

    A bool is not a number here (``true`` would become 1.0), a non-finite
    value is dropped (``nan`` would clamp to maximum), and so is one outside
    [0, 1] (#655): a rating on another scale (``7`` out of 10, ``"1e3"``)
    would otherwise clamp to 1.0 and exempt the record from retention
    demotion, so the rubric rates it instead.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        number = float(value)
    except (ValueError, OverflowError):
        # Unparseable text, or an integer too large for a float.
        return None
    return number if math.isfinite(number) and 0.0 <= number <= 1.0 else None


def _consolidation_from_text(text: str, stamp: RecordStamp) -> Consolidation:
    """Parse a consolidation reply into a :class:`Consolidation`.

    ``entries`` items with a non-blank string ``narrative`` become entries;
    any other item is dropped. Without an ``entries`` key, a legacy
    ``summary`` reads as the one entry. A wrong-typed ``participants`` reads
    as none, with a warning (#655). Raises ValueError on unusable output (no
    JSON, no usable entries), which each caller already handles.
    """
    data = _parse_json_block(text)
    entries: list[ConsolidationEntry] = []
    try:
        # New shape: keep each entry with a non-blank narrative, in order.
        if "entries" in data:
            items = data["entries"]
            for item in items if isinstance(items, list) else []:
                narrative = item.get("narrative") if isinstance(item, dict) else None
                if not isinstance(narrative, str) or not narrative.strip():
                    continue
                try:
                    tags = _string_tuple(item.get("topic_tags"), "topic_tags")
                except ValueError:
                    # A malformed entry is dropped like any other.
                    continue
                entries.append(ConsolidationEntry(narrative.strip(), tags))
        # Legacy shape (no "entries" key): the summary is the one entry.
        else:
            summary = data.get("summary")
            if isinstance(summary, str) and summary.strip():
                tags = _string_tuple(data.get("topic_tags"), "topic_tags")
                entries.append(ConsolidationEntry(summary.strip(), tags))
    except ValueError as exc:
        raise ValueError(f"consolidation output is malformed: {exc}") from exc
    if not entries:
        raise ValueError(
            "consolidation output has no usable entries "
            "(each needs a non-empty narrative)"
        )
    # Participants are shared by every entry of the consolidation. A
    # wrong-typed value loses only itself, not the entries (#655).
    try:
        participants = _string_tuple(data.get("participants"), "participants")
    except ValueError as exc:
        logger.warning(
            "consolidation reply's participants is unusable (%s); storing the "
            "entries with no participants", exc,
        )
        participants = ()
    return Consolidation(
        entries=tuple(entries),
        participants=participants,
        importance=_coerce_importance(data.get("importance")),
        stamp=stamp,
    )


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
    """Raw message_log rows for (after_id, through_id], id-ordered.

    The message column comes back as bytes (CAST AS BLOB) so a corrupt row
    fails in ``decode_message_row`` instead of inside the fetch.
    """
    async with db.execute(
        "SELECT id, CAST(message AS BLOB), message_type FROM message_log "
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


def _decodes_to_object(raw: bytes | str) -> bool:
    """True when a message_log value is a JSON object (quiet: no warning)."""
    try:
        return isinstance(json.loads(raw), dict)
    except (ValueError, RecursionError):
        # Corrupt value: the caller treats it as absent; the warning is
        # logged once by decode_message_row in dialog_from_rows.
        return False


def _dialog_ids(rows: list[tuple]) -> set[int]:
    """Row ids of the message-type rows that decode.

    A corrupt row never loads into a window, so a view or fork must not be
    required to find it there (dialog_from_rows warned about it).
    """
    return {
        row_id for row_id, raw, mtype in rows
        if mtype == "message" and _decodes_to_object(raw)
    }


def dialog_from_rows(rows: list[tuple], channel_id: str) -> list[dict]:
    """The consolidation working set: real user/assistant dialog only.

    A corrupt row, or one whose message_type is outside MessageType, counts
    as non-dialog: it is skipped with a warning naming the row and channel,
    so the watermark can still advance past it.
    """
    known = {t.value for t in MessageType}
    dialog = []
    for rowid, message_json, message_type in rows:
        # An unknown type is non-dialog, loudly
        if message_type not in known:
            parse_message_type(
                rowid, message_type, channel_id,
                consequence="consolidation treats it as non-dialog",
            )
            continue
        if message_type != "message":
            continue
        message = decode_message_row(
            rowid, message_json, channel_id,
            consequence="consolidation treats it as non-dialog and moves on",
        )
        if message is None:
            continue
        content = message.get("content")
        if message.get("role") in ("user", "assistant") and \
                isinstance(content, str) and content.strip():
            dialog.append(message)
    return dialog


@dataclass(frozen=True)
class CatchupSegment:
    """One window segment of a catch-up range: its rows, where its watermark
    lands, and the finished background request."""

    rows: list[tuple[int, bytes, str]]  # the segment's range rows (summary rows included), fetch_range_rows shape
    end_id: int                         # rows[-1][0]: the watermark after this segment commits
    messages: list[dict]                # the finished background request


def _segment_run(
    view: list[dict], own_ids: set[int], inputs: ViewInputs, fork_prompt: str
) -> list[list[int]]:
    """Cut one run's view into segments that each fit the channel's budget.

    ``view`` is the run's messages (the summary in force first, if any);
    ``own_ids`` the ids of the run's range rows. Returns, per segment, the
    view indices of its own rows. A candidate segment fits when the system
    prompt, the instruction on its untrimmed view and its own rows (plus a
    context tool call whose results open the segment) total at most the
    budget. A tool cycle straddling a cut goes whole into the next segment;
    a unit that alone overflows becomes its own segment.
    """
    own_pos = [k for k, m in enumerate(view) if m.get("_db_id") in own_ids]
    sys_tokens = count_tokens(inputs.system_prompt)
    toks = [message_tokens(m) for m in view]
    # Where each tool call sits in the view, to find calls below a segment's start.
    call_at = {c["id"]: k for k, m in enumerate(view) for c in m.get("tool_calls") or []}

    def fits(a: int, b: int) -> bool:
        """Whether own rows ``own_pos[a:b]`` fit the budget as one segment."""
        idx = own_pos[a:b]
        end = idx[-1] + 1
        # A call below the segment's start whose results are own rows counts
        # as an own row: the fit must keep the call and its results together.
        ctx_calls = {
            call_at[view[k]["tool_call_id"]] for k in idx
            if view[k].get("role") == "tool"
            and call_at.get(view[k].get("tool_call_id"), end) < idx[0]
        }
        own_tokens = sum(toks[k] for k in idx) + sum(toks[k] for k in ctx_calls)
        dialog = {
            view[k]["_db_id"] for k in idx if view[k].get("_message_type") == MessageType.MESSAGE
        }
        instruction = {
            "role": "user",
            "content": fork_prompt + "\n\n" + _fork_range_sentence(view[:end], dialog),
        }
        return sys_tokens + message_tokens(instruction) + own_tokens <= inputs.max_tokens

    segments: list[list[int]] = []
    a = 0
    while a < len(own_pos):
        # Grow the segment: ``b`` is the exclusive end (an index into
        # ``own_pos``) of the longest run of own rows from ``a`` that fits.
        b = a
        while b < len(own_pos) and fits(a, b + 1):
            b += 1
        if b == len(own_pos):
            segments.append(own_pos[a:])
            break
        # The cut falls before own row b: back up to a straddling tool
        # cycle's call (the cycle goes whole into the next segment).
        first = own_pos[a]
        sub = view[first:]
        count = 0
        if b > a:
            boundary = first + snap_cut_to_tool_boundary(sub, own_pos[b] - first)
            count = sum(1 for k in own_pos[a:b] if k < boundary)
        if count == 0:
            # One row or cycle alone overflows: it is its own segment.
            boundary = first + snap_cut_forward(sub, 1)
            count = sum(1 for k in own_pos[a:] if k < boundary)
        # ``count`` own rows leave as this segment; the next starts after them.
        segments.append(own_pos[a:a + count])
        a += count
    return segments


def _measured_size(messages: list[dict]) -> int:
    """A request's size as sent: the system prompt by ``count_tokens``, every
    later message by ``message_tokens``."""
    return count_tokens(messages[0]["content"]) + sum(message_tokens(m) for m in messages[1:])


async def build_catchup_segments(
    db: aiosqlite.Connection, channel_id: str, after_id: int, through_id: int,
    inputs: ViewInputs, fork_prompt: str,
) -> list[CatchupSegment]:
    """The background requests for a range that has left the live window.

    The range is (after_id, through_id]. Each non-summary range row has a
    summary in force (the newest decodable summary dated before it); a run of
    rows under the same summary is one window, rebuilt as that summary plus
    the rows reload would load under it, capped at the run's last row. A run
    is cut into segments that each fit ``inputs.max_tokens`` (see
    ``_segment_run``); each segment's view is trimmed oldest-first to the
    budget, put under ``inputs.apply_reasoning_history``, and ends with the
    consolidation instruction. The request body is always
    ``CONSOLIDATION_EXTRA_BODY`` with no tools. The daemon and the replay
    harness both call this, so they cannot diverge (R10).
    """
    # 1. Read once: the range rows with timestamps, every decodable summary,
    # then the view rows.
    async with db.execute(
        "SELECT id, CAST(message AS BLOB), message_type, timestamp FROM message_log "
        "WHERE channel_id = ? AND id > ? AND id <= ? ORDER BY id",
        (channel_id, after_id, through_id),
    ) as cursor:
        fetched = await cursor.fetchall()
    range_rows = [row[:3] for row in fetched]
    row_ts = {row[0]: row[3] for row in fetched}
    summaries = await decodable_summaries(db, channel_id)

    # 2-3. A run is a maximal sequence of non-summary range rows with the
    # same summary in force; summary rows attach to a segment by id span.
    runs: list[tuple[tuple | None, list[tuple]]] = []
    for row in range_rows:
        if row[2] == "summary":
            continue
        in_force = next((sm for sm in summaries if sm[1] < row_ts[row[0]]), None)
        if runs and (runs[-1][0] and runs[-1][0][0]) == (in_force and in_force[0]):
            runs[-1][1].append(row)
        else:
            runs.append((in_force, [row]))
    if not runs:
        return []
    after_ts = None if any(r[0] is None for r in runs) else min(r[0][1] for r in runs)
    view = await view_rows(db, channel_id, after_ts, through_id)

    # 4-5. Each run's view, then its segments (own-row view indices).
    plans: list[tuple[list[dict], list[int], set[int]]] = []
    for in_force, run_rows in runs:
        last_id = run_rows[-1][0]
        if in_force is not None:
            # Rows reload would load under the summary, in id order.
            body = sorted(
                (m for ts, m in view if ts > in_force[1] and m["_db_id"] <= last_id),
                key=lambda m: m["_db_id"],
            )
            run_view = [in_force[2]] + body
        else:
            run_view = [m for _, m in view if m["_db_id"] <= last_id]
        own_ids = {r[0] for r in run_rows}
        for own in _segment_run(run_view, own_ids, inputs, fork_prompt):
            plans.append((run_view, own, own_ids))

    # 6. Spans: segment k ends at the last range row before segment k+1's
    # first non-summary row; the last ends at the last range row.
    all_ids = [row[0] for row in range_rows]
    firsts = [min(run_view[k]["_db_id"] for k in own) for run_view, own, _ in plans]
    ends = [
        max(i for i in all_ids if i < nxt) for nxt in firsts[1:]
    ] + [all_ids[-1]]

    # 7. One request per segment.
    segments: list[CatchupSegment] = []
    start_after = after_id
    for (run_view, own, _), end_id in zip(plans, ends):
        seg_rows = [r for r in range_rows if start_after < r[0] <= end_id]
        start_after = end_id
        seg_ids = {r[0] for r in seg_rows if r[2] != "summary"}
        window = ContextWindow(channel_id)
        window.system_prompt = inputs.system_prompt
        window.fence_key = inputs.fence_key
        # Shallow copies: the policy's in-place stripping must not alter rows
        # a later segment reuses as context.
        window.messages = [dict(m) for m in run_view[:own[-1] + 1]]
        dialog_ids = _dialog_ids(seg_rows)

        def instruction() -> str:
            """The consolidation instruction over the window as it stands."""
            return fork_prompt + "\n\n" + _fork_range_sentence(window.messages, dialog_ids)

        # I0: the instruction's size on the untrimmed window, reserved by
        # fitting the window to what is left. Then the live loop's order:
        # fit, then the reasoning_history policy.
        reserve = message_tokens({"role": "user", "content": instruction()})
        dropped, truncated = window.fit_to_budget(inputs.max_tokens - reserve)
        inputs.apply_reasoning_history(window.messages)

        # 8. Warnings: rows the view lost, truncation, residual overage.
        lost = sorted(
            m["_db_id"] for m in dropped if m.get("_db_id") in seg_ids
        )
        if lost:
            logger.warning(
                "catch-up view for channel %s dropped its own rows (message ids %s) "
                "to fit max_context_tokens %d; they are committed without reaching a "
                "view — raise max_context_tokens or the consolidation profile's context",
                channel_id, ", ".join(map(str, lost)), inputs.max_tokens,
            )
        if truncated:
            logger.warning(
                "catch-up view for channel %s truncated message id %s to fit "
                "max_context_tokens %d — raise max_context_tokens or the "
                "consolidation profile's context",
                channel_id, window.messages[-1].get("_db_id"), inputs.max_tokens,
            )
        messages = build_fork_prompt(
            window, None, instruction(), frozenset(), inputs.mid_role
        )
        measured = _measured_size(messages)
        if measured > inputs.max_tokens:
            logger.warning(
                "catch-up view for channel %s, message ids %d-%d measures %d tokens, "
                "over max_context_tokens %d; sending it anyway — raise "
                "max_context_tokens or the consolidation profile's context",
                channel_id, seg_rows[0][0], end_id, measured, inputs.max_tokens,
            )
        segments.append(CatchupSegment(seg_rows, end_id, messages))
    return segments


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


def _previous_reply(messages: list[dict]) -> str | None:
    """The text of the window's last sent assistant reply, or None.

    The last assistant MESSAGE row with non-empty string content: tool-call-
    only rows carry no text to refer back to, and an UNSENT row (vetoed by a
    send gate) never reached the person, so neither can be what a follow-up
    means by "that".
    """
    # Walk back from the tail to the first qualifying assistant row.
    for msg in reversed(messages):
        if (
            msg.get("role") == "assistant"
            and msg.get("_message_type", MessageType.MESSAGE) == MessageType.MESSAGE
            and isinstance(msg.get("content"), str)
            and msg["content"].strip()
        ):
            return msg["content"]
    return None


def retrieval_query(messages: list[dict], query_context: str) -> str:
    """Build passive retrieval's query text from a window (#665).

    The unanswered user messages (a coalesced burst, #398), preceded with
    ``query_context: previous_reply`` by the agent's previous reply when the
    window holds one. Parts are joined by blank lines, so the FTS query's
    per-paragraph token cap gives each its own budget and a long reply cannot
    crowd out the user's words. Shared by ``before_agent_turn`` and
    ``scripts/eval_memory.py`` so the eval scores the query the daemon runs.
    shortcut: the reply is used whole, however long; trim or cap it if the
    labelled self-contained probes show it diluting short questions.
    """
    from corvidae.context import unanswered_user_messages

    # The user part: every unanswered user message with text.
    parts = [
        m["content"]
        for m in unanswered_user_messages(messages)
        if isinstance(m.get("content"), str)
    ]
    # The context part: the previous reply, ahead of the user's words.
    if query_context == "previous_reply":
        reply = _previous_reply(messages)
        if reply is not None:
            parts.insert(0, reply)
    return "\n\n".join(parts)


def _format_age(seconds: float) -> str:
    """Compact human age for retrieval lines: 'now', '12m', '5h', '3d'."""
    if seconds < 60:
        return "now"
    if seconds < 3600:
        return f"{int(seconds // 60)}m"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h"
    return f"{int(seconds // 86400)}d"


def _fts_match_query(text: str, stopwords: bool = False) -> str | None:
    """Build a safe FTS5 MATCH expression from free text.

    Raw user text can be FTS5 syntax-invalid; quote each word token and OR
    them. Returns None when the text has no word tokens. With ``stopwords``
    (the hybrid side, #666), words in ``FTS_STOPWORDS`` are left out before
    the cap, so the cap counts content words.

    The 20-token cap applies per blank-line-separated paragraph, so a
    coalesced burst (messages joined by blank lines, #398) keeps each
    message's tokens; single-paragraph text is capped at 20 as always.
    shortcut: a message or ``memory_search`` query that itself contains blank
    lines gets 20 tokens per paragraph, and there is no total cap (a 100-message
    burst gives ~2000 OR terms); add a cap if bursts that size ever appear.
    """
    # Cap each paragraph's tokens separately, then pool them all.
    tokens: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        words = re.findall(r"\w+", paragraph)
        if stopwords:
            words = [w for w in words if w.lower() not in FTS_STOPWORDS]
        tokens.extend(words[:20])
    if not tokens:
        return None
    return " OR ".join(f'"{token}"' for token in tokens)


def _fused_keys(
    pool_rows: list[dict], in_window_rows: list[dict],
    vec_keys: dict[int, float], fts_order: list[int],
) -> dict[int, tuple[float, float]]:
    """Reciprocal-rank-fusion sort keys for a hybrid pool (#666).

    Two ranked lists: every candidate with a measured cosine, by decayed
    cosine (so recency stays in the ranking, as before fusion), and every
    FTS-matched candidate, in bm25 order. A candidate earns
    ``1 / (RRF_C + rank)`` from each list it is in; missing from a list, it
    earns nothing there. The key is (fused, score), so the decayed score
    breaks ties. Pool ranks are counted among the pool alone; an in-window
    row gets the rank it would have had among them (one more than the pool
    rows ahead of it), so its would-be rank is judged as if it stood alone.
    """
    pool_ids = {c["id"] for c in pool_rows}
    # The vec list, best first, over the pool's measured candidates.
    vec_list = sorted(
        (i for i in pool_ids if i in vec_keys), key=lambda i: vec_keys[i], reverse=True,
    )
    vec_rank = {i: n for n, i in enumerate(vec_list, start=1)}
    # The FTS list keeps bm25 order, pool rows only.
    fts_list = [i for i in fts_order if i in pool_ids]
    fts_rank = {i: n for n, i in enumerate(fts_list, start=1)}

    def fused(vec_r: int | None, fts_r: int | None) -> float:
        """Sum of the reciprocal-rank terms the candidate earns."""
        total = 0.0
        if vec_r is not None:
            total += 1.0 / (RRF_C + vec_r)
        if fts_r is not None:
            total += 1.0 / (RRF_C + fts_r)
        return total

    keys: dict[int, tuple[float, float]] = {
        c["id"]: (fused(vec_rank.get(c["id"]), fts_rank.get(c["id"])), c["score"])
        for c in pool_rows
    }
    # An in-window row's hypothetical ranks among the pool rows.
    fts_position = {i: n for n, i in enumerate(fts_order)}
    for e in in_window_rows:
        vec_r = None
        if e["id"] in vec_keys:
            vec_r = 1 + sum(1 for i in vec_list if vec_keys[i] >= vec_keys[e["id"]])
        fts_r = None
        if e["id"] in fts_position:
            fts_r = 1 + sum(1 for i in fts_list if fts_position[i] < fts_position[e["id"]])
        keys[e["id"]] = (fused(vec_r, fts_r), e["score"])
    return keys


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
        *,
        hint: float | None = None,
    ) -> float: ...   # 0.0–1.0


class RubricPrior:
    """Default importance prior: cheap-model rubric rating (§3.1).

    Returns 0.5 on any failure (logged) — a scoring outage must never
    block memory formation.
    """

    def __init__(
        self,
        get_client: Callable[[], object],
        get_prompt: Callable[[], str] | None = None,
    ) -> None:
        # A callable, not a client, so the client is resolved at call time:
        # the plugin may not have started when this is constructed.
        self._get_client = get_client
        self._get_prompt = get_prompt or (lambda: RUBRIC_PROMPT)

    async def score(
        self,
        messages: list[dict],
        msg_id_range: tuple[int, int] | None = None,
        channel=None,
        *,
        hint: float | None = None,
    ) -> float:
        # msg_id_range / channel are accepted for protocol conformance
        # (WP2.5) but the rubric ignores both — it rates the dialog text.
        # The consolidation reply's own rating, when it gave one, stands in
        # for the model call.
        if hint is not None:
            return max(0.0, min(1.0, hint))
        try:
            client = self._get_client()
            response = await client.chat([
                {"role": "system", "content": self._get_prompt()},
                {"role": "user", "content": _dialog_transcript(messages)},
            ], extra_body=RUBRIC_EXTRA_BODY)
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
    embedded INTEGER NOT NULL DEFAULT 0,-- 0 = embedding pending/failed (backfillable)
    prompt_hash TEXT,                   -- sha256 of the instruction text; NULL = unknown/direct
    mode TEXT,                          -- fork | standalone | direct; NULL = unknown (pre-stamp)
    model TEXT,                         -- model the consolidation call was sent with
    trigger TEXT                        -- compaction | idle | turn | eval; NULL = direct/pre-#633
)
"""

# Instruction text of each stamp, stored once per hash (issue #299). Holds
# configured instructions only, never conversation text.
MEMORY_PROMPT_DDL = """
CREATE TABLE IF NOT EXISTS memory_prompt (
    hash TEXT PRIMARY KEY,
    text TEXT NOT NULL
)
"""


def prompt_hash(text: str) -> str:
    """The stamp hash of an instruction text: sha256 of its UTF-8 bytes."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RecordStamp:
    """How a memory record was made: instruction text, mode and model sent.

    ``prompt`` is None for ``direct`` records (no consolidation call);
    ``model`` is None when the client could not say.
    """

    prompt: str | None
    mode: str
    model: str | None
    # What started the consolidation (compaction | idle | turn | eval); set by
    # _commit, None for direct records (#633).
    trigger: str | None = None

    @property
    def hash(self) -> str | None:
        """The prompt's hash, or None when there is no prompt."""
        return None if self.prompt is None else prompt_hash(self.prompt)


# Records no consolidation call produced (goal retirement): mode only.
DIRECT_STAMP = RecordStamp(None, "direct", None)


def resolve_consolidation_fork_prompt(config: dict) -> str:
    """The consolidation instruction for a memory config.

    Routes through :func:`resolve_prompt` so the key accepts inline text,
    a file reference, or a list of files (PRD principle 15, #411).
    """
    return resolve_prompt(config, "memory.consolidation_fork_prompt",
                          DEFAULT_CONSOLIDATION_FORK_PROMPT)
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
    degraded_to_fts INTEGER NOT NULL DEFAULT 0,
    excluded_in_window INTEGER      -- in-window records matched before exclusion; NULL if none applied
)
"""

# One row per logged candidate of a passive retrieval (#407). Holds ids and
# numbers only, never text. outcome is one of: admitted, dropped_weak,
# dropped_budget, dropped_dup, cut, excluded_in_window, not_admitted (no
# funnel registered, #654).
RETRIEVAL_CANDIDATE_DDL = """
CREATE TABLE IF NOT EXISTS retrieval_candidate (
    retrieval_id INTEGER NOT NULL,   -- retrieval_log.id
    memory_id    INTEGER NOT NULL,   -- memory.id
    similarity   REAL NOT NULL,      -- cosine similarity, or FTS_SIMILARITY_PROXY when not measurable
    score        REAL NOT NULL,      -- recency-decayed score (banding input; the vec-only rank key)
    band         TEXT NOT NULL,      -- strong | moderate | weak
    rank         INTEGER NOT NULL,   -- 1-based; would-be rank for excluded_in_window rows
    outcome      TEXT NOT NULL,
    source       TEXT,               -- vec | fts | both (#666); NULL on rows from before it
    PRIMARY KEY (retrieval_id, memory_id)
) WITHOUT ROWID
"""

# Lookups by memory: "why didn't it remember X" and the redaction delete.
RETRIEVAL_CANDIDATE_INDEX_DDL = """
CREATE INDEX IF NOT EXISTS retrieval_candidate_memory
    ON retrieval_candidate (memory_id)
"""


async def redact_retrieval_candidates(
    db: aiosqlite.Connection,
    channel_id: str,
    msg_ids: list[int],
    dry_run: bool,
    tombstone: str,
) -> int | None:
    """Delete retrieval_candidate rows that reference a redacted memory (#407).

    Registered as the ``retrieval_candidate`` scrubber in the
    ``corvidae.redact`` entry-point group. The rows hold no text, but a row
    for a redacted memory still records that it was retrieved. Targets rows
    of memories in ``channel_id`` that are already flagged redacted or whose
    range intersects any of ``msg_ids`` (the cascade's own test), whichever
    channel the retrieval ran on. The predicate runs once per message id so
    a large range never exceeds SQLite's bound-variable limit. Returns the
    row count (deleted, or that would be in dry-run mode), or None when the
    table is absent. ``tombstone`` is unused.
    """
    # A DB whose daemon never ran memory has no table to scrub.
    async with db.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' "
        "AND name = 'retrieval_candidate'"
    ) as cursor:
        if await cursor.fetchone() is None:
            return None

    # Memories already flagged redacted (also covers a target whose range
    # has no message_log rows).
    memory_ids: set[int] = set()
    async with db.execute(
        "SELECT id FROM memory WHERE channel_id = ? AND redacted = 1",
        (channel_id,),
    ) as cursor:
        memory_ids.update(row[0] for row in await cursor.fetchall())
    # Memories the cascade flags (or would flag, in dry-run): range hits x.
    for msg_id in msg_ids:
        async with db.execute(
            "SELECT id FROM memory WHERE channel_id = ? "
            "AND msg_id_start <= ? AND msg_id_end >= ?",
            (channel_id, msg_id, msg_id),
        ) as cursor:
            memory_ids.update(row[0] for row in await cursor.fetchall())

    # Count or delete per memory id (one bound parameter each).
    total = 0
    for memory_id in sorted(memory_ids):
        if dry_run:
            async with db.execute(
                "SELECT COUNT(*) FROM retrieval_candidate WHERE memory_id = ?",
                (memory_id,),
            ) as cursor:
                total += (await cursor.fetchone())[0]
        else:
            cursor = await db.execute(
                "DELETE FROM retrieval_candidate WHERE memory_id = ?",
                (memory_id,),
            )
            total += cursor.rowcount
    if not dry_run:
        await db.commit()
    return total

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
        # Dedicated write connection (#221): opened in on_start once the
        # schema exists (or lazily on first write), closed in on_stop.
        # Only MemoryPlugin writes to it, so its transactions are atomic by
        # construction.
        self._write_db: aiosqlite.Connection | None = None
        # True only when memory opened _write_db itself (not the shared fallback).
        self._owns_write_db = False
        self._write_path: str = "<unknown>"
        self._vec_available = False
        self._encoder_mismatch = False
        self._embedding_cfg: dict | None = None
        self._memory_cfg: dict = {}
        self._idle_consolidate_after: float = DEFAULT_IDLE_CONSOLIDATE_AFTER
        self._consolidation_fork_prompt: str = DEFAULT_CONSOLIDATION_FORK_PROMPT
        # Fork records made in before_compaction, waiting for the
        # on_compaction task: (channel_id, range_end) -> (watermark, record).
        self._fork_stash: collections.OrderedDict[tuple[str, int], tuple[int, Consolidation]] = (
            collections.OrderedDict()
        )
        self._retrieval_k: int = DEFAULT_RETRIEVAL_K
        self._band_strong: float = DEFAULT_BAND_STRONG
        self._band_moderate: float = DEFAULT_BAND_MODERATE
        self._half_life_days: float = DEFAULT_HALF_LIFE_DAYS
        self._band_on: str = BAND_ON_CHOICES[0]
        self._recency_from: str = RECENCY_FROM_CHOICES[0]
        # Phase 1b: near-dup merge threshold (WP1b.2)
        self._dup_threshold: float = 0.95
        # Tracked background tasks (consolidation) — plugin-owned
        # asyncio.create_task, never the TaskQueue: every TaskQueue
        # completion triggers a full main-model turn (§2.3, trap #5).
        self._tasks: set[asyncio.Task] = set()
        # Window tokens that start a turn consolidation (on_init overrides).
        self._consolidate_after_tokens: int = DEFAULT_CONSOLIDATE_AFTER_TOKENS
        # Channels with a turn or idle consolidation task in flight; a repeat
        # trigger (turn or idle) for one of them is skipped.
        self._idle_inflight: set[str] = set()
        # Kernel clock and idle tracker; both set in on_start, or both None.
        self._clock: Clock | None = None
        self._idle: IdleTracker | None = None
        self._retention_interval: float = DEFAULT_RETENTION_INTERVAL
        self._retention_max_delay: float = DEFAULT_RETENTION_MAX_DELAY
        # Days after which retention prunes retrieval logs; None is off (#654).
        self._retrieval_log_max_age_days: float | None = None
        # Per-channel consolidation watch.
        self._idle_watch: IdleWatch | None = None
        # Timer for the next retention due time (or the read-failure re-arm).
        self._retention_timer: Timer | None = None
        # Quiet watch and max-delay deadline, live from due until the run.
        self._retention_watch: IdleWatch | None = None
        self._retention_deadline: Timer | None = None
        # True from the due time until the run starts.
        self._retention_pending: bool = False
        # Serializes MemoryPlugin's own writers (consolidation versus goal
        # retirement) so the watermark compare-and-set is race-free within
        # the process. It no longer carries cross-plugin atomicity: that is
        # the write connection's BEGIN IMMEDIATE transaction (#221).
        self._db_lock = asyncio.Lock()
        # Pluggable importance prior; replaced by later phases (§3.1).
        self.importance_prior: ImportancePrior = RubricPrior(
            lambda: self._llm().get_client("rubric"),
            lambda: self._rubric_prompt,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @hookimpl
    def register_llm_stages(self, stages: dict[str, str]) -> None:
        stages["consolidation"] = "chat"
        stages["rubric"] = "chat"
        stages["embedding"] = "embedding"

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        """Read memory.* and the embedding stage's resolved llm profile."""
        await super().on_init(pm, config)
        self._memory_cfg = config.get("memory", {}) or {}
        # The same parse LLMPlugin runs, so the encoder identity here is the
        # profile the embedding stage's client is built from (llm.stages,
        # else the default embedding profile; None when it has neither).
        llm_config = parse_llm_config(config.get("llm"))
        embedding_profile = llm_config.profile_for("embedding", "embedding")
        self._embedding_cfg = (
            llm_config.profiles[embedding_profile] if embedding_profile else None
        )
        raw_after = self._memory_cfg.get(
            "idle_consolidate_after", DEFAULT_IDLE_CONSOLIDATE_AFTER
        )
        self._idle_consolidate_after = float(raw_after)
        # Fail at startup, naming the key, rather than inside the idle tracker.
        if (
            not math.isfinite(self._idle_consolidate_after)
            or self._idle_consolidate_after <= 0
        ):
            raise ValueError(
                f"memory.idle_consolidate_after: got {raw_after!r}; expected "
                "a positive finite number of seconds — check agent.yaml"
            )
        # Fail at startup, naming the key; bool is an int subclass (YAML
        # ``true``), so it is rejected explicitly.
        raw_tokens = self._memory_cfg.get(
            "consolidate_after_tokens", DEFAULT_CONSOLIDATE_AFTER_TOKENS
        )
        if (
            isinstance(raw_tokens, bool)
            or not isinstance(raw_tokens, int)
            or raw_tokens <= 0
        ):
            raise ValueError(
                f"memory.consolidate_after_tokens: got {raw_tokens!r}; expected "
                "a positive integer number of window tokens "
                f"(default {DEFAULT_CONSOLIDATE_AFTER_TOKENS}) — check agent.yaml"
            )
        self._consolidate_after_tokens = raw_tokens
        retention_cfg = self._memory_cfg.get("retention", {}) or {}
        self._retention_interval = float(
            retention_cfg.get("interval", DEFAULT_RETENTION_INTERVAL)
        )
        self._retention_max_delay = float(
            retention_cfg.get("max_delay", DEFAULT_RETENTION_MAX_DELAY)
        )
        # Retrieval-log pruning age (#654): absent or null is off; anything
        # else must be a positive finite number of days. Fail at startup
        # naming the key; bool is an int subclass (YAML ``true``).
        raw_age = retention_cfg.get("retrieval_log_max_age_days")
        if raw_age is not None and (
            isinstance(raw_age, bool)
            or not isinstance(raw_age, (int, float))
            or not math.isfinite(raw_age)
            or raw_age <= 0
        ):
            raise ValueError(
                f"memory.retention.retrieval_log_max_age_days: got {raw_age!r}; "
                "expected a positive number of days, or leave it unset to keep "
                "every retrieval_log row — check agent.yaml"
            )
        self._retrieval_log_max_age_days = (
            float(raw_age) if raw_age is not None else None
        )
        # A non-str prompt is refused here (startup fails loudly).
        self._consolidation_fork_prompt = resolve_consolidation_fork_prompt(config)
        self._rubric_prompt = resolve_prompt(config, "memory.rubric_prompt", RUBRIC_PROMPT)
        # The standalone prompt is gone; say so once at startup (O2).
        if "consolidation_prompt" in self._memory_cfg:
            logger.warning(
                "memory.consolidation_prompt is no longer used and is ignored: "
                "catch-up records are now written from the agent's own view "
                "with memory.consolidation_fork_prompt. Move any custom "
                "instruction there (it is sent as a user instruction after "
                "the conversation, not as a system prompt) and remove "
                "memory.consolidation_prompt from agent.yaml."
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
        # What the passive query holds besides the user's words (#665).
        self._query_context = _retrieval_choice(
            retrieval_cfg, "query_context", QUERY_CONTEXT_CHOICES
        )
        # Where vector-path candidates come from (#666).
        self._candidates = _retrieval_choice(
            retrieval_cfg, "candidates", CANDIDATES_CHOICES
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
                # Schema exists: open the dedicated write connection.
                await self._write_conn()
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
        # No task is writing any more: close the dedicated write connection.
        if self._write_db is not None:
            write_db, self._write_db = self._write_db, None
            try:
                # The in-memory fallback is the persistence connection: not ours.
                if self._owns_write_db:
                    await write_db.close()
            except Exception:
                logger.warning(
                    "memory write connection failed to close cleanly",
                    exc_info=True,
                )

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
        miss (range not in the window, fork failed) leaves no stash and the
        on_compaction task uses the background view: after the drop the
        range has left the window, so it is rebuilt from message_log (#320).
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
            if not dialog_from_rows(rows, channel.id):
                return
            # The range must be wholly in the prefix the fork will see.
            conv = getattr(channel, "conversation", None)
            contained = self._range_in_window(conv, rows, cut) if conv is not None else None
            if contained is None:
                logger.debug(
                    "consolidation range not in the window",
                    extra={"channel_id": channel.id},
                )
                return
            record = await self._fork_record(agent, channel, contained, cut)
            self._stash_put((channel.id, range_end), (watermark, record))
        except Exception:
            logger.warning(
                "consolidation fork at compaction failed for channel %s; "
                "the on_compaction task uses the background view", channel.id,
                exc_info=True, extra={"channel_id": channel.id},
            )
        finally:
            reset_attribution(attribution_token)

    def _stash_put(self, key: tuple[str, int], value: tuple[int, Consolidation]) -> None:
        """Stash a fork record, evicting the oldest past the bound."""
        self._fork_stash[key] = value
        self._fork_stash.move_to_end(key)
        while len(self._fork_stash) > FORK_STASH_MAXSIZE:
            self._fork_stash.popitem(last=False)

    @staticmethod
    def _range_in_window(
        conv, rows: list[tuple], cut: int | None
    ) -> tuple[list[dict], set[int]] | None:
        """``(prefix, dialog_ids)`` when every dialog row of ``rows`` is in the
        window the fork would see (``messages[:cut]``), else None.

        The record must be about rows the model can read. Summary rows are
        skipped (summaries are never consolidated, and a live window's
        summary has no rowid). Pure: no await, so a caller can check and
        render with the window unchanged between them.
        """
        prefix = conv.messages if cut is None else conv.messages[:cut]
        dialog_ids = _dialog_ids(rows)
        if not dialog_ids <= window_db_ids(prefix):
            return None
        return prefix, dialog_ids

    async def _fork_record(
        self, agent, channel, contained: tuple[list[dict], set[int]], cut: int | None
    ) -> Consolidation:
        """The memory record from a fork of ``channel``'s window.

        ``contained`` is :meth:`_range_in_window`'s result, so the range is
        known to be in the window. A fork failure or an unusable reply
        raises: the caller decides between a skip (idle, in-window range)
        and the catch-up view (compaction).
        """
        prefix, dialog_ids = contained
        instruction = (
            self._consolidation_fork_prompt + "\n\n"
            + _fork_range_sentence(prefix, dialog_ids)
        )
        result = await agent.fork(
            channel, instruction, stage="consolidation", cut=cut,
            extra_body=CONSOLIDATION_EXTRA_BODY,
        )
        # Stamp the fork instruction (not the code-generated range
        # sentence, which quotes conversation text) and the model sent.
        return _consolidation_from_text(
            result.text,
            RecordStamp(self._consolidation_fork_prompt, "fork", result.model),
        )

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
        # A consolidation (turn or idle) is already running or being set up.
        if channel_id in self._idle_inflight:
            return
        # Claim the channel before the first await so an overlapping firing
        # sees it; released below unless a task takes ownership of the claim.
        self._idle_inflight.add(channel_id)
        spawned = False
        try:
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
            task = self._spawn(self._consolidate_range(channel_id, max_id, "idle"))
            spawned = True
            # The task's end frees the channel, however it ends.
            task.add_done_callback(lambda _t: self._idle_inflight.discard(channel_id))
        finally:
            # Every exit that spawned no task (including cancellation) releases
            # the claim.
            if not spawned:
                self._idle_inflight.discard(channel_id)

    @hookimpl
    async def on_agent_response(self, channel) -> None:
        """Turn trigger: a final reply may have left N window tokens of
        un-consolidated conversation above the watermark.

        Runs before the reply is sent, so it awaits nothing: in-memory checks
        and one spawn. The token sum needs the watermark (a DB read), so it
        lives in the spawned task; this only fixes ``range_end`` now.
        """
        # Forks off: nothing to do, and nothing to log (PRD §3.8.5).
        if fork_agent(self.pm, "consolidation") is None:
            return
        # Tool calls outstanding: this reply is not a turn boundary.
        if channel.pending_tool_call_ids:
            return
        # A consolidation is already in flight for this channel.
        if channel.id in self._idle_inflight:
            return
        # range_end is the newest rowid in the window right now.
        ids = window_db_ids(channel.conversation.messages)
        if not ids:
            return
        task = self._spawn(self._turn_consolidation(channel, max(ids)))
        # Claim the channel right after the spawn (no await between); the
        # task's end frees it, however it ends.
        self._idle_inflight.add(channel.id)
        task.add_done_callback(lambda _t: self._idle_inflight.discard(channel.id))

    async def _turn_consolidation(self, channel, range_end: int) -> None:
        """Fork-consolidate ``(watermark, range_end]`` once the window tokens
        of un-consolidated conversation in it reach N; otherwise do nothing."""
        try:
            db = await self._ensure_schema()
            # No database: the next turn retries, as _consolidate_range does.
            if db is None:
                return
            watermark = await self._get_watermark(db, channel.id)
        except Exception:
            logger.warning(
                "turn consolidation could not read the watermark for channel %s "
                "(schema setup or watermark query failed); "
                "the next turn trigger retries", channel.id, exc_info=True,
            )
            return
        # Sum new conversation tokens, stopping once N is reached.
        total = 0
        for m in channel.conversation.messages:
            db_id = m.get("_db_id")
            if not isinstance(db_id, int) or not watermark < db_id <= range_end:
                continue
            # Admitted memories, summaries and unsent replies are not new dialog.
            if m.get("_message_type") in (
                MessageType.CONTEXT, MessageType.SUMMARY, MessageType.UNSENT
            ):
                continue
            total += message_tokens(m)
            if total >= self._consolidate_after_tokens:
                await self._consolidate_range(channel.id, range_end, "turn")
                return

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
                "SELECT id, CAST(message AS BLOB), channel_id FROM message_log "
                "WHERE id NOT IN (SELECT rowid FROM message_fts)"
            ) as cursor:
                rows = await cursor.fetchall()
            if not rows:
                return
            for row_id, msg_json, row_channel in rows:
                # A corrupt row is indexed as empty text (with a warning) so
                # it is not retried on every start and the rest still index.
                msg = decode_message_row(
                    row_id, msg_json, row_channel,
                    consequence="it is indexed as empty text and the "
                                "backfill continues",
                )
                content = (msg.get("content") if msg is not None else "") or ""
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
    # The consolidation task (single code path for all three triggers: compaction, idle, turn)
    # ------------------------------------------------------------------

    async def _consolidate_range(self, channel_id: str, range_end: int, trigger: str) -> None:
        """Consolidate message_log rows (watermark, range_end] into records.

        A range wholly in the live window is one record (a fork, or a skip).
        A range that has left it (compaction and idle; a turn range that has
        left it is skipped, see ``_live_record``) is written from the agent's
        rebuilt view on
        the consolidation stage's client, one record per segment (split at summary boundaries
        and wherever a segment would exceed the channel's budget); each
        segment commits and moves the watermark by itself.

        Overlap safety (trap #6): the triggers (compaction, idle, turn)
        WILL race to the same range. The watermark is read optimistically before the LLM work,
        then re-checked under the DB lock in the same atomic section as the
        insert + watermark advance; any concurrent advance discards this
        task's record (and, for a multi-segment range, the rest of it).
        """
        attribution_token = set_attribution(
            stage="consolidation", channel_id=channel_id, trigger=trigger
        )
        # The message-id range and what is already stored, for the failure log.
        range_desc = f"up to {range_end}"
        stored = "nothing was stored"
        try:
            db = await self._ensure_schema()
            if db is None:
                return
            watermark = await self._get_watermark(db, channel_id)
            if range_end <= watermark:
                return

            # Fetch the working range and keep only real dialog.
            rows = await fetch_range_rows(db, channel_id, watermark, range_end)
            dialog = dialog_from_rows(rows, channel_id)
            if rows:
                range_desc = f"{rows[0][0]}-{rows[-1][0]}"

            if not dialog:
                # Pure CONTEXT/system segments advance the watermark
                # without producing a record.
                await self._commit(
                    channel_id, rows, dialog, watermark, range_end, None, trigger
                )
                return

            # The first-person record: a fork of the live window for a range
            # in it (stashed at compaction, or run now for idle and turn), a skip when
            # that fork misses, else the rebuilt view on the stage's client.
            outcome, record = await self._live_record(
                channel_id, rows, watermark, range_end, trigger
            )
            if outcome == "skip":
                return
            if outcome == "record":
                await self._commit(
                    channel_id, rows, dialog, watermark, range_end, record, trigger
                )
                return

            # Catch-up: one view, one record and one commit per segment. Each
            # commit expects the previous segment's end as the watermark.
            client, stamp, segments = await self._catch_up(channel_id, watermark, range_end)
            expected = watermark
            if not segments:
                # The range has dialog but yielded no request: nothing is
                # stored and the watermark stays, so say so rather than go quiet.
                logger.warning(
                    "catch-up built no segments for %s (message ids %s); "
                    "the range is retried on the next trigger",
                    channel_id, range_desc,
                )
            # Rows of no-dialog segments waiting to fold into the next commit.
            carried: list[tuple] = []
            for i, segment in enumerate(segments):
                seg_dialog = dialog_from_rows(segment.rows, channel_id)
                if not seg_dialog and i < len(segments) - 1:
                    # A no-dialog segment followed by another is not committed
                    # alone: its ids fold into the next record's span, so memory
                    # id spans tile (watermark, range_end] with no gap.
                    carried.extend(segment.rows)
                    continue
                commit_rows = carried + list(segment.rows)
                carried = []
                seg_record = (
                    await self._view_record(client, channel_id, segment, seg_dialog, stamp)
                    if seg_dialog else None
                )
                committed = await self._commit(
                    channel_id, commit_rows, seg_dialog, expected,
                    segment.end_id, seg_record, trigger,
                )
                if not committed:
                    # A concurrent task won; the rest is discarded (OPT-1a-2).
                    return
                expected = segment.end_id
                stored = f"message ids through {expected} are stored"
        except asyncio.CancelledError:
            raise
        except Exception:
            # Fail-soft: the write transaction rolled back and the watermark
            # did not advance past what is stored, so a later trigger retries.
            logger.warning(
                "consolidation failed for %s (trigger=%s, message ids %s); "
                "%s and the rest of the range will be retried on the "
                "next trigger",
                channel_id, trigger, range_desc, stored, exc_info=True,
            )
        finally:
            reset_attribution(attribution_token)

    async def _commit(
        self, channel_id: str, rows: list[tuple], dialog: list[dict],
        expected_wm: int, new_wm: int, record: Consolidation | None, trigger: str,
    ) -> bool:
        """Store ``record``'s entries for ``rows`` and move the watermark, atomically.

        Each entry becomes its own row sharing the range, importance,
        valence, participants and stamp. With no record only the watermark
        moves (a range with no dialog).
        The write is a compare-and-set on ``expected_wm``: False when another
        task already moved the watermark, True once committed.
        """
        db = await self._ensure_schema()
        if db is None:
            return False

        if record is None:
            # Pure CONTEXT/system segments advance the watermark
            # without producing a record.
            async with self._db_lock, self._write_txn() as wdb:
                current = await self._get_watermark(wdb, channel_id)
                if current != expected_wm:
                    return False
                await self._set_watermark(wdb, channel_id, new_wm)
            return True

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
            dialog, msg_id_range=msg_id_range, channel=channel,
            hint=record.importance,
        )

        # Valence: mean stage-2 valence over the range (NULL when none) —
        # the §4.9-classifier deletion made real: affect is the appraisal
        # already computed. Fail-soft when appraisal is absent.
        valence: float | None = None
        appraisal_plugin = self.pm.get_plugin("appraisal")
        if appraisal_plugin is not None:
            try:
                valence = await appraisal_plugin.mean_valence(
                    channel_id, msg_id_range
                )
            except Exception:
                logger.warning(
                    "mean-valence lookup failed; storing NULL valence",
                    exc_info=True, extra={"channel_id": channel_id},
                )

        # Date the record from when its source conversation happened,
        # not from when consolidation ran — imported history would
        # otherwise be dated at import time (issue #52).
        created_at = await source_timestamp(db, rows[-1][0])

        # Embed every entry's narrative (in halves if the server caps the
        # batch, #655); failure stores
        # embedded=0 for later backfill (trap #10 — degrade, never block).
        embeddings: list[list[float] | None] = [None] * len(record.entries)
        if self._embedding_enabled():
            try:
                client = self._llm().get_client("embedding")
                embeddings = await self._embed_in_chunks(
                    client, [e.narrative for e in record.entries], channel_id
                )
            except Exception:
                logger.warning(
                    "memory embedding failed for channel %s (%d entries); "
                    "storing them with embedded=0 for backfill",
                    channel_id, len(record.entries), exc_info=True,
                )
            # A vector of the wrong width loses only its own entry's vector.
            embeddings = [
                self._check_embedding_width(v) if v is not None else None
                for v in embeddings
            ]

        # Atomic commit section: insert, vec row, near-dup merge and
        # watermark advance are one transaction on the write connection,
        # so they land together or not at all (#221). The lock keeps
        # memory's own writers (goal retirement) out meanwhile.
        async with self._db_lock, self._write_txn() as wdb:
            current = await self._get_watermark(wdb, channel_id)
            if current != expected_wm:
                # Design (OPT-1a-2): a concurrent consolidation task won
                # the compare-and-set and advanced the watermark.  Discarding
                # here is intentional — attempting to consolidate the
                # non-overlapping tail would require re-fetching messages and
                # risk producing spurious records.  Any rows beyond the
                # winner's range_end remain unconsolidated, but the next
                # trigger (turn, idle or compaction) sees the updated
                # watermark and picks them up.
                logger.debug(
                    "watermark moved during consolidation; discarding",
                    extra={"channel_id": channel_id},
                )
                return False
            # Every consolidation record passes through here, so this is
            # where the stamp learns its trigger; catch-up keeps the one
            # that fell through to it (#633).
            stamp = dataclasses.replace(record.stamp, trigger=trigger)
            memory_ids: list[int] = []
            for entry, embedding in zip(record.entries, embeddings):
                # Siblings never merge into each other: their ids are
                # excluded from the near-dup check (by id, since a sibling
                # that merged with an older record has a widened range).
                memory_ids.append(await self._insert_record(
                    wdb,
                    channel_id=channel_id,
                    summary=entry.narrative,
                    created_at=created_at,
                    importance=importance,
                    valence=valence,
                    topic_tags=list(entry.topic_tags),
                    participants=list(record.participants),
                    msg_id_start=rows[0][0],
                    msg_id_end=rows[-1][0],
                    embedding=embedding,
                    stamp=stamp,
                    exclude_ids=frozenset(memory_ids),
                ))
            await self._set_watermark(wdb, channel_id, new_wm)
        logger.info(
            "memory consolidated",
            extra={
                "channel_id": channel_id,
                "memory_ids": memory_ids,
                "entries": len(memory_ids),
                "msg_id_start": rows[0][0],
                "msg_id_end": rows[-1][0],
                "embedded": sum(e is not None for e in embeddings),
                "trigger": trigger,
            },
        )
        return True

    async def _embed_in_chunks(
        self, client: Any, texts: list[str], channel_id: str,
    ) -> list[list[float] | None]:
        """Embed ``texts`` as documents, halving a batch the server rejects (#655).

        A server with a per-request cap rejects a large batch; the two halves
        are retried, recursively, so depth is bounded by log2 of the count.
        A single text that fails means the encoder itself is failing, not a
        cap: that raises, and the caller stores everything unembedded, as it
        did before the retry existed.
        """
        try:
            vectors = await client.embed(texts, kind="document")
            if len(vectors) != len(texts):
                raise ValueError(
                    f"embedding returned {len(vectors)} vectors for {len(texts)} texts"
                )
            return list(vectors)
        except Exception as exc:
            if len(texts) == 1:
                raise
            logger.warning(
                "memory embedding of %d entries failed for channel %s (%s); "
                "retrying in two smaller batches, in case the server caps the "
                "batch size",
                len(texts), channel_id, type(exc).__name__,
            )
        mid = len(texts) // 2
        return (
            await self._embed_in_chunks(client, texts[:mid], channel_id)
            + await self._embed_in_chunks(client, texts[mid:], channel_id)
        )

    async def _insert_record(
        self, db: aiosqlite.Connection, *, channel_id: str, summary: str,
        created_at: float, importance: float, valence: float | None,
        topic_tags: list[str], participants: list[str],
        msg_id_start: int, msg_id_end: int, embedding: list[float] | None,
        stamp: RecordStamp, exclude_ids: frozenset[int] = frozenset(),
    ) -> int:
        """Insert + vec row + near-dup merge on ``db``, the write connection.

        ``exclude_ids`` are records the near-dup check must skip (the
        siblings this consolidation already inserted).

        The caller holds ``_db_lock`` inside ``_write_txn`` and so owns the
        commit or rollback. ``stamp`` says how the record was made; its
        instruction text is stored once in ``memory_prompt``.
        """
        # The instruction text, once per hash (an empty prompt still counts).
        if stamp.prompt is not None:
            await db.execute(
                "INSERT OR IGNORE INTO memory_prompt (hash, text) VALUES (?, ?)",
                (stamp.hash, stamp.prompt),
            )
        cursor = await db.execute(
            "INSERT INTO memory (channel_id, created_at, summary, "
            "importance, valence, topic_tags, participants, "
            "msg_id_start, msg_id_end, embedded, prompt_hash, mode, model, trigger) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                channel_id,
                created_at,
                summary,
                importance,
                valence,
                json.dumps(topic_tags),
                json.dumps(participants),
                msg_id_start,
                msg_id_end,
                1 if embedding is not None else 0,
                stamp.hash,
                stamp.mode,
                stamp.model,
                stamp.trigger,
            ),
        )
        memory_id = cursor.lastrowid

        # WP1b.2 — near-dup check: query BEFORE inserting the new vec row
        # so the new record is naturally excluded from the results.
        # Candidates come from the agent's whole memory (PRD §3.7 Memory 3,
        # GitHub issue #168): the nearest indexed record over the threshold
        # merges, on whatever channel it lives.
        # Skip entirely when embedding is unavailable or the new record
        # has no vector (embed failed — no vector to compare).
        dup_old_id: int | None = None
        dup_cross_channel = False
        if (
            embedding is not None
            and self._vec_available
            and not self._encoder_mismatch
        ):
            try:
                import sqlite_vec as _sv
                async with db.execute(
                    "SELECT v.memory_id, (1.0 - v.distance) as similarity, "
                    "m.channel_id "
                    "FROM (SELECT memory_id, distance FROM memory_vec "
                    "      WHERE embedding MATCH ? AND k = ?) AS v "
                    "JOIN memory m ON m.id = v.memory_id "
                    "WHERE m.indexed = 1 "
                    "ORDER BY v.distance",
                    (
                        _sv.serialize_float32(embedding),
                        VEC_OVERFETCH * 2 + len(exclude_ids),
                    ),
                ) as dup_cursor:
                    dup_rows = [
                        r for r in await dup_cursor.fetchall()
                        if r[0] not in exclude_ids
                    ]
                # Rows are nearest first: the nearest non-sibling merges if
                # it is over the threshold, and none farther can be.
                if dup_rows and dup_rows[0][1] >= self._dup_threshold:
                    dup_old_id, dup_similarity, dup_channel = dup_rows[0]
                    dup_cross_channel = dup_channel != channel_id
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
                    new_start, new_end = msg_id_start, msg_id_end
                else:
                    new_start = min(old_start, msg_id_start)
                    new_end = max(old_end, msg_id_end)
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

        return memory_id

    async def write_record(
        self, *, channel_id: str, summary: str, created_at: float,
        importance: float, msg_id_start: int, msg_id_end: int,
    ) -> int:
        """Embed (fail soft to embedded=0), then _insert_record as one
        transaction on the write connection under _db_lock. Raises (leaving
        no row) if the DB isn't open or on DB failure; returns memory.id."""
        if await self._ensure_schema() is None:
            raise RuntimeError("memory database is not open")
        # Embed the summary; failure stores embedded=0 for later backfill.
        embedding: list[float] | None = None
        if self._embedding_enabled():
            try:
                client = self._llm().get_client("embedding")
                embedding = (await client.embed([summary], kind="document"))[0]
            except Exception:
                logger.warning(
                    "memory embedding failed; storing record with "
                    "embedded=0 for backfill", exc_info=True,
                )
            if embedding is not None:
                embedding = self._check_embedding_width(embedding)
        async with self._db_lock, self._write_txn() as wdb:
            memory_id = await self._insert_record(
                wdb, channel_id=channel_id, summary=summary,
                created_at=created_at, importance=importance, valence=None,
                topic_tags=[], participants=[],
                msg_id_start=msg_id_start, msg_id_end=msg_id_end,
                embedding=embedding, stamp=DIRECT_STAMP,
            )
        return memory_id

    def _skip_consolidation(
        self, channel_id: str, rows: list[tuple], watermark: int, reason: str,
        trigger: str, **kwargs,
    ) -> None:
        """Log that an in-window range was skipped: channel, message range and
        what was missing (consolidation has no trace id). The watermark is
        untouched; ``trigger`` names what retries (idle, turn, or idle or
        compaction for a turn range skipped with forks off)."""
        logger.warning(
            "consolidation skipped for channel %s, messages %d-%d: %s; "
            "the watermark stays at %d and the next %s trigger retries",
            channel_id, rows[0][0], rows[-1][0], reason, watermark, trigger,
            extra={"channel_id": channel_id}, **kwargs,
        )

    async def _live_record(
        self, channel_id: str, rows: list[tuple], watermark: int, range_end: int,
        trigger: str,
    ) -> tuple[LiveOutcome, Consolidation | None]:
        """How to produce the record for (watermark, range_end].

        Compaction trigger: the record before_compaction stashed, if it was
        built against this same watermark (otherwise it covers a different
        range and is dropped); no stash means ``catch_up``. Idle and turn
        triggers: a range wholly in the channel's live window forks it, and a
        miss (forks off, fork failed, unusable reply) is a ``skip`` with a
        warning (PRD §3.8.5, #320). A range not wholly in the window (no
        channel, no window, rows that left it) is ``catch_up``: the rebuilt
        view, for idle; for turn it is a ``skip`` with one DEBUG line (idle or
        compaction consolidates it).
        """
        if trigger == "compaction":
            stashed = self._fork_stash.pop((channel_id, range_end), None)
            if stashed is not None and stashed[0] == watermark:
                return "record", stashed[1]
        elif trigger in ("idle", "turn"):
            registry = self.pm.get_plugin("registry")
            channel = registry.get(channel_id) if registry is not None else None
            conv = getattr(channel, "conversation", None)
            contained = self._range_in_window(conv, rows, None) if conv is not None else None
            if contained is not None:
                # Live-window work: fork or skip, never the rebuilt view (PRD §3.8.5).
                agent = fork_agent(self.pm, "consolidation")
                if agent is None:
                    self._skip_consolidation(
                        channel_id, rows, watermark,
                        "forks are not enabled for consolidation "
                        "(agent.forks.enabled / agent.forks.stages)",
                        # While forks stay off the turn hook doesn't fire, so
                        # idle or compaction is what retries a turn range.
                        trigger if trigger == "idle" else "idle or compaction",
                    )
                    return "skip", None
                # No await between the containment check and the render.
                try:
                    return "record", await self._fork_record(agent, channel, contained, None)
                except Exception as exc:
                    self._skip_consolidation(
                        channel_id, rows, watermark,
                        f"the fork failed or its reply was unusable ({type(exc).__name__})",
                        trigger, exc_info=True,
                    )
                    return "skip", None
            if trigger == "turn":
                # Not in the window: idle or compaction consolidates it, so
                # this recurs each turn; DEBUG, never the catch-up view.
                logger.debug(
                    "turn consolidation skipped for channel %s, messages %d-%d: "
                    "the range is not wholly in the live window; idle or "
                    "compaction consolidates it; the watermark stays at %d",
                    channel_id, rows[0][0], rows[-1][0], watermark,
                    extra={"channel_id": channel_id},
                )
                return "skip", None
        return "catch_up", None

    async def _catch_up(
        self, channel_id: str, watermark: int, range_end: int
    ) -> tuple[object, RecordStamp, list[CatchupSegment]]:
        """The consolidation client, record stamp and view segments for a
        range that has left the window. Makes no commits.

        Raises:
            RuntimeError: no agent plugin can supply the channel's view
                inputs, or the database is not open.
            ValueError, ChannelNotReady: from ``view_inputs`` (a malformed
                channel id; a transport that cannot build the channel yet).
        """
        agent = view_agent(self.pm)
        if agent is None:
            raise RuntimeError(
                f"no agent plugin can supply view inputs for channel {channel_id}; "
                "the catch-up range stays unconsolidated until it is registered"
            )
        inputs = await agent.view_inputs(channel_id)
        # Bound once (R3): the stamp's model and the requests share a client.
        client = self._llm().get_client("consolidation")
        stamp = RecordStamp(
            self._consolidation_fork_prompt, "standalone",
            client.request_model(CONSOLIDATION_EXTRA_BODY),
        )
        db = await self._ensure_schema()
        if db is None:
            raise RuntimeError(f"memory database not open; cannot build views for channel {channel_id}")
        segments = await build_catchup_segments(
            db, channel_id, watermark, range_end, inputs, self._consolidation_fork_prompt
        )
        return client, stamp, segments

    async def _view_record(
        self, client: Any, channel_id: str, segment: CatchupSegment,
        seg_dialog: list[dict], stamp: RecordStamp,
    ) -> Consolidation:
        """One view call producing the first-person record for ``segment``.

        Returns the parsed :class:`Consolidation`. An
        unusable reply (unparseable, or a tool call) is retried once; if both
        attempts are unusable it degrades to the raw transcript rather than
        losing the range. Transport errors propagate.
        """
        span = f"{segment.rows[0][0]}-{segment.end_id}"
        for attempt in range(2):
            try:
                result = await view_call(
                    client, channel_id, segment.messages,
                    extra_body=CONSOLIDATION_EXTRA_BODY,
                )
                return _consolidation_from_text(result.text, stamp)
            except (ValueError, ForkRefused) as exc:
                logger.warning(
                    "consolidation reply unusable for channel %s, message ids %s "
                    "(attempt %d/2): %s",
                    channel_id, span, attempt + 1, exc,
                )
        # The calls are already paid for; keep the range as its raw transcript.
        logger.warning(
            "consolidation of channel %s, message ids %s: both replies unusable; "
            "storing the raw transcript as the record",
            channel_id, span,
        )
        return Consolidation(
            entries=(ConsolidationEntry(_dialog_transcript(seg_dialog), ()),),
            participants=(),
            importance=None,
            stamp=stamp,
        )

    # ------------------------------------------------------------------
    # Retrieval read path (§3.1)
    # ------------------------------------------------------------------

    @hookimpl
    async def before_agent_turn(self, channel, trace_id, annotations) -> None:
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

            # Query with every unanswered user message (a coalesced burst),
            # preceded by the agent's previous reply unless configured off
            # (#665), so a follow-up retrieves the topic it refers to.
            query = retrieval_query(conv.messages, self._query_context)
            # Leave out this channel's records whose whole range is still in
            # the window (#634): they would only echo what the model can see.
            # A window with no message_log ids applies no exclusion.
            window_ids = window_db_ids(conv.messages)
            exclude = (channel.id, min(window_ids)) if window_ids else None
            pool = await self._retrieve_pool(query, exclude=exclude)
            degraded = pool.degraded
            candidates = pool.ranked[: pool.k]

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
            admission = Admission(admitted=[], duplicate=[])
            # No funnel means nothing could be admitted (#654: logged as
            # not_admitted, not as a budget drop).
            no_funnel = False
            if lines:
                funnel = self.pm.get_plugin("funnel")
                no_funnel = funnel is None
                if funnel is None:
                    logger.warning(
                        "funnel plugin not registered; memory retrieval "
                        "cannot admit context"
                    )
                else:
                    admission = await funnel.admit_detailed(
                        channel, conv, "memory", lines
                    )
                    admitted_lines = admission.admitted
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
            # Both values describe the top-ranked candidate (by fused rank
            # on the hybrid path, by score otherwise): its decayed score,
            # and the raw similarity that band_on: similarity compares
            # against the band thresholds.
            top_score = candidates[0]["score"] if candidates else None
            top_similarity = candidates[0]["similarity"] if candidates else None
            hit_count = len(candidates)
            log_cursor = await db.execute(
                "INSERT INTO retrieval_log (ts, channel_id, trace_id, top_score, "
                "top_similarity, hit_count, admitted_count, degraded_to_fts, "
                "excluded_in_window) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    time.time(),
                    channel.id,
                    trace_id,
                    top_score,
                    top_similarity,
                    hit_count,
                    len(admitted_ids),
                    1 if degraded else 0,
                    pool.excluded_in_window,
                ),
            )
            retrieval_id = log_cursor.lastrowid

            # Log every pool candidate with its outcome (#407). A failure
            # here must not lose the retrieval_log row or the exchange
            # profile, so it is caught and warned about on its own.
            try:
                weak_dropped = bool(non_weak)
                rows = []
                for i, c in enumerate(pool.ranked):
                    if i >= pool.k:
                        outcome = "cut"
                    elif c["band"] == "weak" and weak_dropped:
                        outcome = "dropped_weak"
                    else:
                        line = lines[selected.index(c)]
                        if line in admission.admitted:
                            outcome = "admitted"
                        elif line in admission.duplicate:
                            outcome = "dropped_dup"
                        elif no_funnel:
                            outcome = "not_admitted"
                        else:
                            outcome = "dropped_budget"
                    rows.append((retrieval_id, c["id"], c["similarity"],
                                 c["score"], c["band"], i + 1, outcome,
                                 c["source"]))
                # Excluded in-window records carry their would-be rank.
                for would_be_rank, c in pool.excluded:
                    rows.append((retrieval_id, c["id"], c["similarity"],
                                 c["score"], c["band"], would_be_rank,
                                 "excluded_in_window", c["source"]))
                # A memory the redact CLI flagged after it was scored gets no
                # row (#654): the flag check is part of the insert itself, so
                # either the flag is visible here or the scrubber, which runs
                # after the flag commits, deletes the row.
                await db.executemany(
                    "INSERT INTO retrieval_candidate (retrieval_id, memory_id, "
                    "similarity, score, band, rank, outcome, source) "
                    "SELECT ?, ?, ?, ?, ?, ?, ?, ? "
                    "WHERE (SELECT redacted FROM memory WHERE id = ?) = 0",
                    [(*row, row[1]) for row in rows],
                )
            except Exception:
                logger.warning(
                    "retrieval candidate log failed for channel %s "
                    "(retrieval_log id %s); some or all of this retrieval's "
                    "candidate rows are missing, the retrieval and its "
                    "profile are logged",
                    channel.id, retrieval_id,
                    exc_info=True, extra={"channel_id": channel.id},
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
        self, text: str, k: int | None = None,
        exclude: tuple[str, int] | None = None,
    ) -> tuple[list[dict], bool]:
        """Score memory candidates for a query text.

        Vector KNN over memory_vec when available; FTS5 keyword search
        otherwise or when the encoder fails (trap #10). Candidates come from
        the agent's whole memory, every channel's (PRD principle 14).

        ``exclude``: optional ``(channel_id, min_msg_id)``. Records of that
        channel whose ``msg_id_start`` is at or past ``min_msg_id`` (their
        whole range still in the live window) are left out before the cut
        to k (#634). Passive retrieval passes it; this is not a scope.

        Returns:
            (candidates best first: by fused rank on the hybrid path, by
            score otherwise; degraded_to_fts). Each candidate: {id, summary,
            created_at, similarity, score, band, age_seconds, source}.
        """
        pool = await self._retrieve_pool(text, k, exclude)
        return pool.ranked[: pool.k], pool.degraded

    async def _retrieve_pool(
        self, text: str, k: int | None = None,
        exclude: tuple[str, int] | None = None,
    ) -> RetrievalPool:
        """Score the whole candidate pool for a query text (#407).

        ``retrieve`` returns the top k of ``ranked``; passive retrieval
        logs the whole pool. ``excluded`` lists in-window records whose
        score would have placed them in the top k (vec path only; the FTS
        path reports only their count).
        """
        db = await self._ensure_schema()
        k = k or self._retrieval_k
        if db is None:
            return RetrievalPool([], [], None, False, k)
        now = time.time()

        rows: list[CandidateRow] = []
        degraded = False
        # Hybrid bookkeeping (#666): FTS-matched ids in bm25 order, and the
        # ids that have a measured cosine (vec hits plus FTS hits with a
        # stored vector). Both stay empty on the vec-only and degraded paths.
        fts_order: list[int] = []
        measured: set[int] = set()
        vec_ids: set[int] = set()
        if self._vec_available and self._embedding_enabled():
            try:
                client = self._llm().get_client("embedding")
                query_vector = (await client.embed([text], kind="query"))[0]
                rows = await self._vec_candidates(db, query_vector, k, exclude)
            except Exception:
                logger.warning(
                    "retrieval embedding failed; degrading to FTS5",
                    exc_info=True,
                )
                degraded = True
            else:
                vec_ids = {row[0] for row in rows}
                measured = set(vec_ids)
                if self._candidates == "hybrid":
                    rows, fts_order, measured = await self._hybrid_rows(
                        db, text, query_vector, k, exclude, rows,
                    )
        else:
            degraded = True

        if degraded:
            rows = await self._fts_candidates(db, text, k, exclude)

        fts_ids = set(fts_order)
        candidates = []
        in_window_flags: list[bool] = []
        # Per-candidate ranking inputs for fusion: the decayed raw cosine.
        vec_keys: dict[int, float] = {}
        for (
            memory_id, summary, created_at, last_retrieved_at, similarity, in_window,
        ) in rows:
            in_window_flags.append(in_window)
            # Displayed age is always when the memory was formed.
            age_seconds = max(0.0, now - created_at)
            # Ranking age: from formation, or (recency_from: last_used)
            # from the latest use, matching retention's last_activity.
            if self._recency_from == "last_used":
                last_activity = max(created_at, last_retrieved_at or 0.0)
            else:
                last_activity = created_at
            ranking_age = max(0.0, now - last_activity)
            decay = math.exp(-(ranking_age / 86400.0) / self._half_life_days)
            if memory_id in measured:
                vec_keys[memory_id] = similarity * decay
            # Relevance × recency decay.
            score = similarity * decay
            # The band says how relevant a memory is: from the decayed
            # score (default) or, with band_on: similarity, from relevance
            # alone so an old, apt memory is not presented as a guess.
            band_value = similarity if self._band_on == "similarity" else score
            # A keyword match the vector search missed bands at no less than
            # the FTS proxy (#666), so a name the embedding doesn't place near
            # the query is not dropped as weak. Floored on the band value
            # itself (flooring similarity would still decay below the proxy),
            # and only for FTS-only hits: records the KNN also returned keep
            # their measured, decayed band. ``similarity`` and ``score`` stay
            # measured either way.
            if memory_id in fts_ids and memory_id not in vec_ids:
                band_value = max(band_value, FTS_SIMILARITY_PROXY)
            if band_value >= self._band_strong:
                band = "strong"
            elif band_value >= self._band_moderate:
                band = "moderate"
            else:
                band = "weak"
            # Which search found it: the degraded path is FTS by definition.
            if degraded:
                source = "fts"
            elif memory_id in vec_ids:
                source = "both" if memory_id in fts_ids else "vec"
            else:
                source = "fts"
            candidates.append({
                "id": memory_id,
                "summary": summary,
                "created_at": created_at,
                "similarity": similarity,
                "score": score,
                "band": band,
                "age_seconds": age_seconds,
                "source": source,
            })
        # Split off the in-window rows; the rest is the ranked pool.
        pool_rows = [c for c, w in zip(candidates, in_window_flags) if not w]
        in_window_rows = [c for c, w in zip(candidates, in_window_flags) if w]
        if fts_order:
            # Hybrid: rank by reciprocal rank fusion, score breaking ties.
            keys = _fused_keys(pool_rows, in_window_rows, vec_keys, fts_order)
        else:
            # Vec-only or degraded: rank by the decayed score, as before #666.
            keys = {c["id"]: (c["score"],) for c in candidates}
        ranked = sorted(pool_rows, key=lambda c: keys[c["id"]], reverse=True)
        # Each in-window record is judged alone: would it have made top k?
        excluded: list[tuple[int, dict]] = []
        for e in in_window_rows:
            would_be_rank = 1 + sum(1 for c in ranked if keys[c["id"]] >= keys[e["id"]])
            if would_be_rank <= k:
                excluded.append((would_be_rank, e))
        # The exclusion count: flagged rows (vec) or one COUNT query (FTS).
        excluded_in_window: int | None = None
        if exclude is not None:
            excluded_in_window = (
                await self._fts_in_window_count(db, text, exclude)
                if degraded else len(in_window_rows)
            )
        return RetrievalPool(ranked, excluded, excluded_in_window, degraded, k)

    async def recent(self, limit: int) -> list[dict]:
        """The newest memories of the whole memory, every channel's.

        Same status filters as retrieval: indexed and not redacted.
        Returns up to ``limit`` dicts ``{summary, channel_id, age_seconds}``,
        newest first, aged on the kernel clock. Raises RuntimeError while
        the memory DB is not open.
        """
        db = await self._ensure_schema()
        if db is None:
            raise RuntimeError("memory DB is not open yet")
        async with db.execute(
            "SELECT summary, channel_id, created_at FROM memory "
            "WHERE indexed = 1 AND redacted = 0 "
            "ORDER BY created_at DESC, id DESC LIMIT ?",
            (limit,),
        ) as cursor:
            rows = await cursor.fetchall()
        now = self._now()
        return [
            {"summary": summary, "channel_id": owner, "age_seconds": max(0.0, now - created_at)}
            for summary, owner, created_at in rows
        ]

    async def _vec_candidates(
        self, db: aiosqlite.Connection, query_vector: list[float], k: int,
        exclude: tuple[str, int] | None = None,
    ) -> list[CandidateRow]:
        """Exact-KNN candidates:
        (id, summary, created_at, last_retrieved_at, cosine_similarity,
        in_window). In-window rows are returned flagged, not filtered (#407).

        sqlite-vec is brute-force exact KNN (no ANN index), so the KNN scan
        costs the same regardless of k; we over-fetch and filter by status
        flags in the join, and select the window exclusion (see
        ``retrieve``) as the ``in_window`` flag.
        """
        import sqlite_vec
        knn_k = k * VEC_OVERFETCH
        exclude_sql, exclude_params = _exclude_clause(exclude)
        if exclude is not None:
            # The KNN cut happens before the join's filters, so fetch one
            # extra row per excluded record or in-window records could fill
            # every slot. The count ignores status flags: an upper bound.
            async with db.execute(
                "SELECT COUNT(*) FROM memory WHERE channel_id = ? AND msg_id_start >= ?",
                exclude,
            ) as cursor:
                (excluded_count,) = await cursor.fetchone()
            knn_k += excluded_count
        async with db.execute(
            "SELECT m.id, m.summary, m.created_at, m.last_retrieved_at, v.distance, "
            + exclude_sql + " AS in_window "
            "FROM (SELECT memory_id, distance FROM memory_vec "
            "      WHERE embedding MATCH ? AND k = ?) AS v "
            "JOIN memory m ON m.id = v.memory_id "
            "WHERE m.indexed = 1 AND m.redacted = 0 "
            "ORDER BY v.distance",
            (
                *exclude_params,
                sqlite_vec.serialize_float32(query_vector),
                knn_k,
            ),
        ) as cursor:
            rows = await cursor.fetchall()
        # vec0 cosine distance = 1 - cosine similarity.
        return [
            (memory_id, summary, created_at, last_retrieved_at,
             1.0 - distance, bool(in_window))
            for memory_id, summary, created_at, last_retrieved_at, distance, in_window
            in rows
        ]

    async def _hybrid_rows(
        self, db: aiosqlite.Connection, text: str, query_vector: list[float],
        k: int, exclude: tuple[str, int] | None, vec_rows: list[CandidateRow],
    ) -> tuple[list[CandidateRow], list[int], set[int]]:
        """Add the FTS side to vec candidates for fusion (#666).

        Runs the stopworded FTS search over the same records (``k ×
        VEC_OVERFETCH``, in-window rows flagged, not filtered), measures the
        real cosine of every FTS hit the KNN didn't return from its stored
        vector, and falls back to ``FTS_SIMILARITY_PROXY`` for a hit with no
        vector (one not yet embedded; demoted records aren't candidates).
        Returns (the union of rows, FTS ids in bm25 order, ids with a
        measured cosine). A failing FTS side
        is logged and leaves the vec rows alone: the vec pool is the rescue.
        """
        measured = {row[0] for row in vec_rows}
        try:
            fts_rows = await self._fts_candidates(
                db, text, k * VEC_OVERFETCH, exclude,
                flag_window=True, stopwords=True,
            )
            known = {row[0] for row in vec_rows}
            extra = [row for row in fts_rows if row[0] not in known]
            cosines = await self._vec_similarities(
                db, query_vector, [row[0] for row in extra],
            )
        except Exception:
            logger.warning(
                "hybrid retrieval: the FTS5 side failed; ranking the vector "
                "candidates alone for this query",
                exc_info=True,
            )
            return vec_rows, [], measured
        # FTS-only rows take their measured cosine, or the proxy.
        merged = list(vec_rows)
        for memory_id, summary, created_at, last_retrieved_at, _proxy, in_window in extra:
            similarity = cosines.get(memory_id)
            if similarity is None:
                similarity = FTS_SIMILARITY_PROXY
            else:
                measured.add(memory_id)
            merged.append(
                (memory_id, summary, created_at, last_retrieved_at, similarity, in_window)
            )
        return merged, [row[0] for row in fts_rows], measured

    async def _vec_similarities(
        self, db: aiosqlite.Connection, query_vector: list[float], ids: list[int],
    ) -> dict[int, float]:
        """Cosine similarity to the query of each id's stored vector (#666).

        One point lookup per id in ``memory_vec``; ids with no vector are
        absent from the result.
        """
        if not ids:
            return {}
        import sqlite_vec
        serialized = sqlite_vec.serialize_float32(query_vector)
        result: dict[int, float] = {}
        # One id per query keeps clear of SQLite's bound-variable limit.
        for memory_id in ids:
            async with db.execute(
                "SELECT vec_distance_cosine(embedding, ?) FROM memory_vec "
                "WHERE memory_id = ?",
                (serialized, memory_id),
            ) as cursor:
                row = await cursor.fetchone()
            if row is not None and row[0] is not None:
                # Cosine distance = 1 - cosine similarity.
                result[memory_id] = 1.0 - row[0]
        return result

    async def _fts_candidates(
        self, db: aiosqlite.Connection, text: str, k: int,
        exclude: tuple[str, int] | None = None,
        *, flag_window: bool = False, stopwords: bool = False,
    ) -> list[CandidateRow]:
        """FTS5 keyword candidates with a fixed similarity proxy:
        (id, summary, created_at, last_retrieved_at, FTS_SIMILARITY_PROXY,
        False).

        The window exclusion (see ``retrieve``) sits in WHERE, ahead of the
        LIMIT, so it applies before the cut to k. With ``flag_window`` (the
        hybrid side, #666) in-window rows are returned flagged instead, as
        the vec path does, with the LIMIT raised by their count so they
        can't take every slot; ``stopwords`` drops common words from the
        MATCH. Rows come back in bm25 order."""
        match = _fts_match_query(text, stopwords=stopwords)
        if match is None:
            return []
        exclude_sql, exclude_params = _exclude_clause(exclude)
        if flag_window:
            # Fetch past the in-window rows (an upper bound, as for vec).
            limit = k
            if exclude is not None:
                async with db.execute(
                    "SELECT COUNT(*) FROM memory WHERE channel_id = ? AND msg_id_start >= ?",
                    exclude,
                ) as cursor:
                    (excluded_count,) = await cursor.fetchone()
                limit += excluded_count
            async with db.execute(
                "SELECT m.id, m.summary, m.created_at, m.last_retrieved_at, "
                + exclude_sql + " AS in_window "
                "FROM memory_fts JOIN memory m ON m.id = memory_fts.rowid "
                "WHERE memory_fts MATCH ? AND m.indexed = 1 AND m.redacted = 0 "
                "ORDER BY rank LIMIT ?",
                (*exclude_params, match, limit),
            ) as cursor:
                flagged = await cursor.fetchall()
            return [
                (memory_id, summary, created_at, last_retrieved_at,
                 FTS_SIMILARITY_PROXY, bool(in_window))
                for memory_id, summary, created_at, last_retrieved_at, in_window
                in flagged
            ]
        async with db.execute(
            "SELECT m.id, m.summary, m.created_at, m.last_retrieved_at "
            "FROM memory_fts JOIN memory m ON m.id = memory_fts.rowid "
            "WHERE memory_fts MATCH ? AND m.indexed = 1 AND m.redacted = 0"
            " AND NOT " + exclude_sql + " ORDER BY rank LIMIT ?",
            (match, *exclude_params, k),
        ) as cursor:
            rows = await cursor.fetchall()
        return [
            (memory_id, summary, created_at, last_retrieved_at,
             FTS_SIMILARITY_PROXY, False)
            for memory_id, summary, created_at, last_retrieved_at in rows
        ]

    async def _fts_in_window_count(
        self, db: aiosqlite.Connection, text: str, exclude: tuple[str, int],
    ) -> int:
        """How many in-window records the FTS query matches (#407).

        Same MATCH and status filters as ``_fts_candidates``, with the
        window condition un-negated. Count only: no scores without rows.
        """
        match = _fts_match_query(text)
        if match is None:
            return 0
        exclude_sql, exclude_params = _exclude_clause(exclude)
        async with db.execute(
            "SELECT COUNT(*) FROM memory_fts JOIN memory m ON m.id = memory_fts.rowid "
            "WHERE memory_fts MATCH ? AND m.indexed = 1 AND m.redacted = 0 AND "
            + exclude_sql,
            (match, *exclude_params),
        ) as cursor:
            (count,) = await cursor.fetchone()
        return count

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
        """The LLMPlugin instance (profile-client access)."""
        from corvidae.llm_plugin import LLMPlugin
        return get_dependency(self.pm, "llm", LLMPlugin)

    def _embedding_enabled(self) -> bool:
        """True when an embedding profile is configured and encoders match."""
        return self._embedding_cfg is not None and not self._encoder_mismatch

    async def _open_write_db(self, shared: aiosqlite.Connection) -> aiosqlite.Connection:
        """Open the dedicated write connection on the persistence database file.

        Set up like the persistence connection: same file, the journal mode
        the persistence connection runs in (WAL by default), a bounded
        busy_timeout, and sqlite-vec when memory uses vectors. Autocommit
        (isolation_level=None) so transactions are exactly the explicit
        BEGIN IMMEDIATE .. COMMIT of ``_write_txn``. Returns ``shared`` itself
        for a database with no file (in-memory), which a second connection
        cannot open.
        """
        async with shared.execute("PRAGMA database_list") as cursor:
            files = [row[2] for row in await cursor.fetchall() if row[1] == "main"]
        path = files[0] if files else ""
        if not path:
            # shortcut: an in-memory persistence DB (tests) keeps memory writes
            # on the shared connection with no cross-plugin atomicity; a
            # daemon always has a file, so upgrade only if one ever doesn't.
            logger.warning(
                "memory: persistence database has no file; memory writes "
                "share the persistence connection (not atomic against "
                "other plugins' commits)"
            )
            return shared
        async with shared.execute("PRAGMA journal_mode") as cursor:
            row = await cursor.fetchone()
        journal_mode = str(row[0]) if row else "wal"
        conn = await aiosqlite.connect(path, isolation_level=None)
        try:
            await conn.execute(f"PRAGMA journal_mode={journal_mode}")
            await conn.execute(f"PRAGMA busy_timeout={int(WRITE_BUSY_TIMEOUT_MS)}")
            if self._vec_available and not await self._load_vec_extension(conn):
                logger.warning(
                    "memory: sqlite-vec failed to load on the write connection "
                    "for %s; vector inserts on it will fail until restart "
                    "(check the sqlite-vec install)", path,
                )
        except BaseException:
            # Don't leak the half-configured connection's worker thread.
            await conn.close()
            raise
        self._write_path = path
        return conn

    async def _write_conn(self) -> aiosqlite.Connection | None:
        """The connection memory record writes use; None when no DB is open yet.

        Opened lazily after the schema exists, because on_start order is
        LIFO and the persistence DB may not be open at memory's on_start.
        """
        if self._write_db is not None:
            return self._write_db
        shared = await self._ensure_schema()
        if shared is None:
            return None
        # Re-check: another task may have opened it while the schema ran.
        if self._write_db is None:
            self._write_db = await self._open_write_db(shared)
            self._owns_write_db = self._write_db is not shared
        return self._write_db

    @asynccontextmanager
    async def _write_txn(self) -> AsyncIterator[aiosqlite.Connection]:
        """One memory write as a real SQLite transaction on the write connection.

        ``BEGIN IMMEDIATE`` .. ``COMMIT``; any exception (and cancellation)
        rolls back, logs a failed rollback at WARNING, and propagates for the
        caller to report. A write blocked past the busy timeout logs a
        WARNING naming the database path and timeout, then propagates.
        Callers hold ``_db_lock`` so memory's own writers don't contend.
        """
        conn = await self._write_conn()
        if conn is None:
            raise RuntimeError("memory database is not open")
        shared = self._resolve_db()
        # A fallback to the shared connection has no transaction of its own.
        own_txn = conn is not shared
        path = self._write_path
        if own_txn:
            try:
                await conn.execute("BEGIN IMMEDIATE")
            except sqlite3.OperationalError as exc:
                if "locked" in str(exc) or "busy" in str(exc):
                    logger.warning(
                        "memory write blocked: database %s stayed locked past "
                        "the %d ms busy timeout; the write will be retried "
                        "on the next trigger",
                        path, WRITE_BUSY_TIMEOUT_MS,
                    )
                raise
        try:
            yield conn
            # COMMIT is inside the try: a failed commit (SQLITE_BUSY against a
            # long reader) must roll back rather than leave the lock held.
            await (conn.execute("COMMIT") if own_txn else conn.commit())
        except BaseException:
            # Undo every statement of the section, including cancellation.
            try:
                await (conn.execute("ROLLBACK") if own_txn else conn.rollback())
            except Exception:
                logger.warning(
                    "memory write rollback failed on %s; closing the write "
                    "connection so the next memory write reopens a clean one "
                    "(restart the daemon if writes keep failing)",
                    path, exc_info=True,
                )
                if own_txn:
                    # Drop the wedged connection; closing releases its lock.
                    self._write_db = None
                    try:
                        await conn.close()
                    except Exception:
                        logger.warning(
                            "memory write connection on %s failed to close "
                            "after a failed rollback", path, exc_info=True,
                        )
            raise

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

    def _check_embedding_width(self, embedding: list[float]) -> list[float] | None:
        """A wrong-width vector is an embed failure.

        vec0 rejects a wrong-width insert; letting one reach the vec INSERT
        inside a write transaction rolls the whole transaction back — on the
        consolidation path that is record, vec row and watermark, and every
        later trigger re-summarizes the same range and fails the same way,
        stalling the channel. Returning None routes the row through the
        existing embedded=0 backfill handling (degrade, never block).
        """
        declared = self._embedding_dimensions()
        if declared is not None and len(embedding) != declared:
            logger.warning(
                "memory embedding width mismatch (%d vs declared %d); "
                "storing record with embedded=0 for backfill",
                len(embedding), declared,
            )
            return None
        return embedding

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
        # Stamp columns (#299): tables from before stamping gain them as
        # nullable columns without rewriting any row; old rows read NULL.
        await add_column_if_missing(db, "memory", "prompt_hash", "TEXT")
        await add_column_if_missing(db, "memory", "mode", "TEXT")
        await add_column_if_missing(db, "memory", "model", "TEXT")
        # Trigger column (#633), likewise nullable; old rows read NULL.
        await add_column_if_missing(db, "memory", "trigger", "TEXT")
        await db.execute(MEMORY_PROMPT_DDL)
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
        # #407: the exclusion count and the per-candidate child table.
        await add_column_if_missing(db, "retrieval_log", "excluded_in_window", "INTEGER")
        await db.execute(RETRIEVAL_CANDIDATE_DDL)
        # #666: which search found each candidate; old rows read NULL.
        await add_column_if_missing(db, "retrieval_candidate", "source", "TEXT")
        await db.execute(RETRIEVAL_CANDIDATE_INDEX_DDL)
        # Phase 1b: message_fts and retention_meta
        await db.execute(MESSAGE_FTS_DDL)
        await db.execute(MESSAGE_FTS_INSERT_TRIGGER)
        await db.execute(MESSAGE_FTS_UPDATE_TRIGGER)
        await db.execute(RETENTION_META_DDL)

        # The vector surface exists only with an embedding profile configured
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
