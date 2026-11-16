"""MemoryToolsPlugin — search_memory, search_transcripts and recall_raw tools
(WP1b.3, #648).

Provides the agent's active memory surface (bootstrap-mapping §3.1 "remember
harder"). All three tools span the agent's whole memory, every channel's (PRD
principle 14, GitHub issue #168); they reach further than passive retrieval in
*status* (demoted and superseded records are visible).

Tool semantics:
  search_memory  — search by meaning and keywords (cosine scan fused with FTS5,
                   #667), keyword-only when the vector path is down; does NOT
                   update access stats.
  search_transcripts — keyword search over every channel's raw message_log
                   rows, so specifics consolidation dropped stay reachable;
                   does NOT update access stats.
  recall_raw     — verbatim dialog replay, by memory record (DOES update access
                   stats) or by channel and message-id range (does not).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from datetime import datetime, timezone

from corvidae.context import MessageType
from corvidae.hooks import CorvidaePlugin, get_dependency, hookimpl
from corvidae.persistence import decode_message_row, parse_message_type
from corvidae_cognition.memory import _format_age, _fts_match_query, _fused_keys
from corvidae.tool import Tool, tool_to_schema

logger = logging.getLogger("corvidae.tools.memory_tools")

# Rough characters-per-token approximation for the token cap.
_CHARS_PER_TOKEN = 4

# Most rows each search side contributes, as the keyword search always took.
_SEARCH_LIMIT = 50

# First line of a keyword-only result, so the agent knows a paraphrase may
# have been missed (#667).
KEYWORD_ONLY_NOTE = "(keyword search only: semantic search unavailable)"


def _semantic_unavailable_reason(memory) -> str | None:
    """Why the vector path can't serve a search, or None when it can."""
    if memory._embedding_cfg is None:
        return "no embedding profile configured"
    if memory._encoder_mismatch:
        return "stored encoder differs from the configured one"
    if not memory._vec_available:
        return "sqlite-vec not loaded"
    return None


def _parse_utc(name: str, value: str | None) -> float | str | None:
    """Parse an after/before filter to an epoch float (naive means UTC).

    None passes through; an unparseable value returns the error string the
    tool hands back, naming the argument and the accepted formats.
    """
    if value is None:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return (
            f"Error: invalid '{name}' value '{value}'. "
            "Use YYYY-MM-DD or ISO-8601 format."
        )
    # Naive values are read as UTC, never the host's local time.
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _cap_lines(lines: list[str], max_tokens: int) -> str:
    """Join lines, stopping before the one that would cross ``max_tokens``
    (approximate) and appending the truncation marker."""
    max_chars = max_tokens * _CHARS_PER_TOKEN
    kept: list[str] = []
    used_chars = 0
    truncated = False
    for line in lines:
        # Stop before the line that would cross the cap.
        if used_chars + len(line) + 1 > max_chars:
            truncated = True
            break
        kept.append(line)
        used_chars += len(line) + 1
    if truncated:
        kept.append("[truncated — token limit reached]")
    return "\n".join(kept)


def _row_role(
    row_id: int, msg_json: bytes, message_type: str, channel_id: str,
    consequence: str,
) -> tuple[str, dict] | None:
    """Decode a message_log row to (role label, message), or None to skip it.

    A corrupt blob or unknown type is skipped with the standard warning
    (``consequence`` says what the caller still returns). An unsent row's
    role is labelled so it is not read back as dialog that was said.
    """
    msg = decode_message_row(row_id, msg_json, channel_id, consequence=consequence)
    if msg is None:
        return None
    if message_type not in {t.value for t in MessageType}:
        parse_message_type(
            row_id, message_type, channel_id, consequence=consequence,
        )
        return None
    role = msg.get("role", "?")
    # A withheld response was never said; label it.
    if message_type == MessageType.UNSENT.value:
        role = f"{role} (withheld, not sent)"
    return role, msg


async def _replay(
    db, channel: str, start: int, end: int, header: str, max_tokens: int,
) -> tuple[str, int]:
    """Replay one channel's message_log ids [start, end] as ``role: content``
    lines under ``header``; returns (text, rows read)."""
    # message_log uses a global AUTOINCREMENT id shared by all channels;
    # the numeric range contains interleaved foreign-channel rows. The
    # channel_id predicate is load-bearing.
    async with db.execute(
        "SELECT id, CAST(message AS BLOB), message_type FROM message_log "
        "WHERE channel_id = ? AND id BETWEEN ? AND ? ORDER BY id",
        (channel, start, end),
    ) as cursor:
        msg_rows = await cursor.fetchall()

    # Build output with token cap (approximate: chars / 4); the header counts.
    max_chars = max_tokens * _CHARS_PER_TOKEN
    output_parts = [header]
    used_chars = len(header)
    truncated = False
    for row_id, msg_json, message_type in msg_rows:
        decoded = _row_role(
            row_id, msg_json, message_type, channel,
            consequence="the rest of the range is still returned",
        )
        if decoded is None:
            continue
        role, msg = decoded
        content = msg.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        line = f"{role}: {content}"
        if used_chars + len(line) + 1 > max_chars:
            truncated = True
            break
        output_parts.append(line)
        used_chars += len(line) + 1
    if truncated:
        output_parts.append("[truncated — token limit reached]")
    return "\n".join(output_parts), len(msg_rows)


def _format_results(
    entries: list[tuple[int, str, str, float, int, int | None]], max_tokens: int,
) -> str:
    """Render (id, score text, summary, created_at, indexed, superseded_by)
    entries as numbered lines, capped at roughly ``max_tokens``."""
    now = time.time()
    lines: list[str] = []
    for mid, score_text, summary, created_at, indexed, superseded_by in entries:
        # Status flags: superseded wins over demoted (both are indexed=0).
        flags = []
        if superseded_by is not None:
            flags.append("[superseded]")
        elif indexed == 0:
            flags.append("[demoted]")
        flag_str = (" " + " ".join(flags)) if flags else ""
        age_str = _format_age(max(0.0, now - created_at))
        lines.append(f"{mid}. {score_text} ({age_str}){flag_str} {summary}")
    return _cap_lines(lines, max_tokens)


# A None schema can mean another task is still creating it, so wait briefly
# before reporting it unavailable.
_SCHEMA_RETRY_ATTEMPTS = 3
_SCHEMA_RETRY_DELAY = 0.05


class MemoryToolsPlugin(CorvidaePlugin):
    """Plugin registering search_memory, search_transcripts and recall_raw tools.

    Depends on the memory plugin for DB access.
    """

    depends_on = frozenset({"memory"})

    def __init__(self, pm=None) -> None:
        if pm is not None:
            self.pm = pm

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        await super().on_init(pm, config)

    def _memory(self):
        from corvidae_cognition.memory import MemoryPlugin
        return get_dependency(self.pm, "memory", MemoryPlugin)

    async def _wait_for_schema(self, memory):
        """Return the memory DB, retrying briefly while schema setup finishes."""
        for attempt in range(_SCHEMA_RETRY_ATTEMPTS):
            db = await memory._ensure_schema()
            if db is not None:
                return db
            if attempt < _SCHEMA_RETRY_ATTEMPTS - 1:
                await asyncio.sleep(_SCHEMA_RETRY_DELAY)
        return None

    async def _keyword_rows(
        self, db, query: str, match: str, where: str, params: list,
    ) -> list[tuple] | None:
        """FTS5 keyword rows (id, summary, created_at, indexed, superseded_by,
        rank) best first, falling back to a LIKE scan when the FTS query
        fails. None when the scan fails too."""
        sql = (
            "SELECT m.id, m.summary, m.created_at, m.indexed, m.superseded_by, "
            "memory_fts.rank "
            "FROM memory_fts JOIN memory m ON m.id = memory_fts.rowid "
            f"WHERE memory_fts MATCH ? AND {where} "
            f"ORDER BY rank LIMIT {_SEARCH_LIMIT}"
        )
        try:
            async with db.execute(sql, (match, *params)) as cursor:
                return list(await cursor.fetchall())
        except Exception:
            logger.warning(
                "search_memory FTS query failed; falling back to summary scan",
                exc_info=True,
            )
        # Degrade to a plain LIKE scan over summaries using the same word
        # tokens, so a broken FTS index does not fail the search.
        tokens = re.findall(r"\w+", query)[:20]
        like_sql = (
            "SELECT m.id, m.summary, m.created_at, m.indexed, "
            "m.superseded_by, 0.0 FROM memory m "
            f"WHERE {where} AND ("
            + " OR ".join("m.summary LIKE ?" for _ in tokens)
            + f") ORDER BY m.created_at DESC LIMIT {_SEARCH_LIMIT}"
        )
        like_params = [f"%{token}%" for token in tokens]
        try:
            async with db.execute(like_sql, (*params, *like_params)) as cursor:
                return list(await cursor.fetchall())
        except Exception:
            logger.warning("search_memory fallback scan failed", exc_info=True)
            return None

    async def _semantic_entries(
        self, db, memory, query: str, query_vector: list[float],
        where: str, params: list,
    ) -> list[tuple[int, str, str, float, int, int | None]]:
        """Rank filtered records by meaning and keywords (#667).

        Two lists over the same filters: an exact cosine scan of the records
        that have a vector (filtered before the cut, which a KNN can't do),
        and the stopworded FTS5 search. They are fused by reciprocal rank as
        passive retrieval fuses them (#666), but on raw cosine with no
        recency decay: a deliberate search wants the best answer, and age is
        shown. A keyword hit outside the scan gets its cosine from its stored
        vector; one with no vector (demoted, unembedded) is ranked by FTS
        alone and shows ``kw``. A failing FTS side is logged and the scan's
        results stand alone. Returns display entries, best first.
        """
        import sqlite_vec

        # The cosine scan, filtered in WHERE so filters apply before LIMIT.
        async with db.execute(
            "SELECT m.id, m.summary, m.created_at, m.indexed, m.superseded_by, "
            "vec_distance_cosine(v.embedding, ?) AS distance "
            "FROM memory m JOIN memory_vec v ON v.memory_id = m.id "
            f"WHERE {where} ORDER BY distance LIMIT {_SEARCH_LIMIT}",
            (sqlite_vec.serialize_float32(query_vector), *params),
        ) as cursor:
            scanned = await cursor.fetchall()
        records: dict[int, tuple] = {}
        cosine: dict[int, float] = {}
        for mid, summary, created_at, indexed, superseded_by, distance in scanned:
            records[mid] = (summary, created_at, indexed, superseded_by)
            cosine[mid] = 1.0 - distance

        # The keyword side, in bm25 order; common words left out.
        fts_order: list[int] = []
        match = _fts_match_query(query, stopwords=True)
        if match is not None:
            try:
                async with db.execute(
                    "SELECT m.id, m.summary, m.created_at, m.indexed, m.superseded_by "
                    "FROM memory_fts JOIN memory m ON m.id = memory_fts.rowid "
                    f"WHERE memory_fts MATCH ? AND {where} "
                    f"ORDER BY rank LIMIT {_SEARCH_LIMIT}",
                    (match, *params),
                ) as cursor:
                    keyword = await cursor.fetchall()
            except Exception:
                logger.warning(
                    "search_memory FTS query failed; falling back to the "
                    "semantic results alone for %r", query, exc_info=True,
                )
                keyword = []
            for mid, summary, created_at, indexed, superseded_by in keyword:
                fts_order.append(mid)
                records.setdefault(mid, (summary, created_at, indexed, superseded_by))
            # Keyword hits the scan didn't reach: measure their cosine.
            missing = [mid for mid in fts_order if mid not in cosine]
            cosine.update(await memory._vec_similarities(db, query_vector, missing))

        # Fuse: cosine list over measured records, FTS list in bm25 order.
        pool = [{"id": mid, "score": cosine.get(mid, 0.0)} for mid in records]
        keys = _fused_keys(pool, [], cosine, fts_order)
        ranked = sorted(records, key=lambda mid: keys[mid], reverse=True)
        return [
            (
                mid,
                f"{cosine[mid]:.2f}" if mid in cosine else "kw",
                records[mid][0], records[mid][1], records[mid][2], records[mid][3],
            )
            for mid in ranked
        ]

    @hookimpl
    def register_tools(self, tool_registry: list) -> None:
        plugin = self

        async def search_memory(
            query: str,
            channel: str | None = None,
            tags: list[str] | None = None,
            after: str | None = None,
            before: str | None = None,
            include_demoted: bool = True,
            max_tokens: int = 2000,
        ) -> str:
            """Search memory records by meaning and keywords. Returns a numbered list
            led by each memory id, with score (cosine similarity to the query, or
            "kw" for a keyword hit with no embedding), age, flags
            ([demoted]/[superseded]), and summary; best match first, by relevance
            alone. Demoted and superseded records are found by keyword only. If semantic search
            is unavailable the first line says so and results are keyword matches.
            Does not update access stats. Use include_demoted=False to restrict to active (indexed=1)
            records. Use after/before (YYYY-MM-DD or ISO-8601, UTC) to filter by
            creation date. Use channel to restrict to one channel's records
            (an exact id such as irc:#lex). Use tags=[...] to filter by
            topic_tags. Results are token-capped at max_tokens (approximate)."""
            memory = plugin._memory()

            # Parse date filters (naive values treated as UTC); an error
            # string is returned as-is.
            after_ts = _parse_utc("after", after)
            if isinstance(after_ts, str):
                return after_ts
            before_ts = _parse_utc("before", before)
            if isinstance(before_ts, str):
                return before_ts

            db = await plugin._wait_for_schema(memory)
            if db is None:
                return "Error: memory schema not available"

            # Build FTS match expression
            match = _fts_match_query(query)
            if match is None:
                return "No results: query has no searchable terms."

            # The whole memory, narrowed to one channel only when asked. The
            # channel arg matches by equality, never as a glob.
            conditions: list[str] = []
            params: list = []
            if channel is not None:
                conditions.append("m.channel_id = ?")
                params.append(channel)

            if not include_demoted:
                # indexed=1 excludes demoted (indexed=0), superseded (indexed=0),
                # and redacted-tombstone records (indexed=0) alike — all are indexed=0
                conditions.append("m.indexed = 1")

            if after_ts is not None:
                conditions.append("m.created_at >= ?")
                params.append(after_ts)
            if before_ts is not None:
                conditions.append("m.created_at < ?")
                params.append(before_ts)

            # Tags filter: JSON array intersection on topic_tags.
            # Parameterized via IN clause over json_each — safe against injection.
            if tags:
                tag_placeholders = ",".join("?" for _ in tags)
                conditions.append(
                    f"m.topic_tags IS NOT NULL AND EXISTS ("
                    f"SELECT 1 FROM json_each(m.topic_tags) "
                    f"WHERE json_each.value IN ({tag_placeholders}))"
                )
                params.extend(tags)

            where = " AND ".join(conditions) if conditions else "1"

            # Embed the query when the vector path is up (#667); otherwise,
            # or when the encoder fails, search by keyword and say so.
            query_vector: list[float] | None = None
            reason = _semantic_unavailable_reason(memory)
            if reason is None:
                try:
                    client = memory._llm().get_client("embedding")
                    query_vector = (await client.embed([query], kind="query"))[0]
                except Exception:
                    logger.warning(
                        "search_memory: the query encoder failed; falling back "
                        "to keyword search for %r", query, exc_info=True,
                    )
            else:
                logger.warning(
                    "search_memory: semantic search unavailable (%s); using "
                    "keyword search for %r", reason, query,
                )

            if query_vector is not None:
                # A failing cosine scan (a damaged memory_vec) falls through
                # to the keyword search rather than failing the tool.
                try:
                    entries = await plugin._semantic_entries(
                        db, memory, query, query_vector, where, params,
                    )
                except Exception:
                    logger.warning(
                        "search_memory: the semantic scan failed; falling back "
                        "to keyword search for %r", query, exc_info=True,
                    )
                else:
                    if not entries:
                        return "No memories found matching your query."
                    return _format_results(entries, max_tokens)

            rows = await plugin._keyword_rows(db, query, match, where, params)
            if rows is None:
                return "Error: search failed internally"
            if not rows:
                return f"{KEYWORD_ONLY_NOTE}\nNo memories found matching your query."
            # FTS5 rank is negative (more negative = more relevant); display
            # as a positive band-less score, as the keyword search always has.
            entries = [
                (mid, f"{-(rank or 0.0):.4f}", summary, created_at, indexed, superseded_by)
                for mid, summary, created_at, indexed, superseded_by, rank in rows
            ]
            return f"{KEYWORD_ONLY_NOTE}\n" + _format_results(entries, max_tokens)

        async def search_transcripts(
            query: str,
            channel: str | None = None,
            after: str | None = None,
            before: str | None = None,
            max_tokens: int = 2000,
        ) -> str:
            """Search every channel's raw messages by keyword; each hit's leading number is a message_log id, widen it with recall_raw(channel, start_id, end_id); channel is an exact id and after/before are YYYY-MM-DD or ISO-8601 UTC (null for no filter); does not update access stats."""
            # Parse date filters; an error string is returned as-is.
            after_ts = _parse_utc("after", after)
            if isinstance(after_ts, str):
                return after_ts
            before_ts = _parse_utc("before", before)
            if isinstance(before_ts, str):
                return before_ts

            memory = plugin._memory()
            db = await plugin._wait_for_schema(memory)
            if db is None:
                return "Error: memory schema not available"

            match = _fts_match_query(query)
            if match is None:
                return "No results: query has no searchable terms."

            # Optional filters, all bound parameters (channel is equality).
            conditions = ["message_fts MATCH ?"]
            params: list = [match]
            if channel is not None:
                conditions.append("m.channel_id = ?")
                params.append(channel)
            if after_ts is not None:
                conditions.append("m.timestamp >= ?")
                params.append(after_ts)
            if before_ts is not None:
                conditions.append("m.timestamp < ?")
                params.append(before_ts)

            # One read-only query: bm25 order, a snippet window per hit.
            try:
                async with db.execute(
                    "SELECT m.id, m.channel_id, CAST(m.message AS BLOB), "
                    "m.message_type, m.timestamp, "
                    "snippet(message_fts, 0, '**', '**', '…', 16) "
                    "FROM message_fts JOIN message_log m ON m.id = message_fts.rowid "
                    f"WHERE {' AND '.join(conditions)} "
                    f"ORDER BY bm25(message_fts) LIMIT {_SEARCH_LIMIT}",
                    params,
                ) as cursor:
                    rows = await cursor.fetchall()
            except Exception as exc:
                logger.warning(
                    "search_transcripts query failed for %r", query, exc_info=True,
                )
                return (
                    f'Error: search_transcripts failed for "{query}" '
                    f"({type(exc).__name__}); retry with a simpler query."
                )

            lines: list[str] = []
            for row_id, channel_id, msg_json, message_type, ts, snippet in rows:
                # Corrupt rows and unknown types drop out with a warning.
                decoded = _row_role(
                    row_id, msg_json, message_type, channel_id,
                    consequence="it is left out of the search results",
                )
                if decoded is None:
                    continue
                role, _msg = decoded
                # UTC explicitly, never the host's local time.
                when = datetime.fromtimestamp(ts, timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                )
                lines.append(f"{row_id}. [{when}] {channel_id} {role}: {snippet}")

            if not lines:
                where = f" in channel {channel}" if channel is not None else ""
                return f"No transcript messages found matching your query{where}."
            return _cap_lines(lines, max_tokens)

        # The output is other channels' raw text: fence it (#648).
        search_transcripts.returns_untrusted = True  # type: ignore[attr-defined]

        async def recall_raw(
            memory_id: int | None = None,
            channel: str | None = None,
            start_id: int | None = None,
            end_id: int | None = None,
            max_tokens: int = 1000,
        ) -> str:
            """Replay raw dialog messages, either by memory_id (from a memory line's "(age, memory <id>)" or search_memory; updates access stats) or by channel with start_id and end_id (message ids from search_transcripts hits; no stat update); pass null for the arguments of the form you don't use."""
            # Classify the call: the record form (memory_id alone) or the
            # range form (channel, start_id and end_id together).
            forms = (
                "recall_raw takes either memory_id alone, or channel, "
                "start_id and end_id together"
            )
            record_form = (
                memory_id is not None
                and channel is None and start_id is None and end_id is None
            )
            range_form = (
                memory_id is None
                and channel is not None and start_id is not None
                and end_id is not None
            )
            # Exactly one form per call; name what was passed.
            if not record_form and not range_form:
                passed = [
                    name for name, value in (
                        ("memory_id", memory_id), ("channel", channel),
                        ("start_id", start_id), ("end_id", end_id),
                    ) if value is not None
                ]
                got = ", ".join(passed) if passed else "no arguments"
                return (
                    f"Error: {forms}; pass null for the arguments you don't "
                    f"use (got {got})."
                )
            if range_form and start_id > end_id:  # type: ignore[operator]
                return (
                    f"Error: recall_raw start_id {start_id} is after end_id "
                    f"{end_id}; pass start_id <= end_id. {forms}."
                )

            memory = plugin._memory()
            db = await plugin._wait_for_schema(memory)
            if db is None:
                return "Error: memory schema not available"

            if range_form:
                # Range form: replay only; no stat writes.
                header = f"[channel {channel}: messages {start_id}–{end_id}]"
                text, rows_read = await _replay(
                    db, channel, start_id, end_id, header, max_tokens,  # type: ignore[arg-type]
                )
                if rows_read == 0:
                    text += (
                        f"\nNo messages in channel {channel} between ids "
                        f"{start_id} and {end_id}."
                    )
                return text

            # Record form: fetch the record (no indexed/redacted filter:
            # "remember harder").
            async with db.execute(
                "SELECT id, channel_id, summary, msg_id_start, msg_id_end, participants "
                "FROM memory WHERE id = ?",
                (memory_id,),
            ) as cursor:
                row = await cursor.fetchone()

            if row is None:
                return f"Error: memory {memory_id} not found"

            mid, channel_id, _summary, msg_id_start, msg_id_end, participants_json = row

            # Parse participants for the header
            try:
                participants = json.loads(participants_json) if participants_json else []
            except Exception:
                participants = []

            participants_display = (
                "{" + ", ".join(sorted(participants)) + "}"
                if participants else "{}"
            )
            header = (
                f"[memory {mid}: messages {msg_id_start}–{msg_id_end}, "
                f"participants: {participants_display}]"
            )
            text, _rows_read = await _replay(
                db, channel_id, msg_id_start, msg_id_end, header, max_tokens,
            )

            # Bump access stats (WP1b.1.3: recall_raw bumps, search_memory does not)
            now = time.time()
            await db.execute(
                "UPDATE memory SET "
                "retrieval_count = retrieval_count + 1, "
                "last_retrieved_at = ? "
                "WHERE id = ?",
                (now, memory_id),
            )
            await db.commit()

            return text

        # recall_raw replays raw channel text in both forms: fence it (#648).
        recall_raw.returns_untrusted = True  # type: ignore[attr-defined]

        tool_registry.append(Tool(
            name="search_memory",
            fn=search_memory,
            schema=tool_to_schema(search_memory),
        ))
        tool_registry.append(Tool(
            name="search_transcripts",
            fn=search_transcripts,
            schema=tool_to_schema(search_transcripts),
        ))
        tool_registry.append(Tool(
            name="recall_raw",
            fn=recall_raw,
            schema=tool_to_schema(recall_raw),
        ))
