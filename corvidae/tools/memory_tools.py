"""MemoryToolsPlugin — search_memory and recall_raw tools (WP1b.3).

Provides the agent's active memory surface (bootstrap-mapping §3.1 "remember
harder"). Both tools enforce the calling channel's _channel_scope — they are
broader than passive retrieval in *status* (see demoted/superseded records)
but never broader in *compartment* (channel B cannot reach channel A's records).

Tool semantics:
  search_memory  — FTS keyword search; does NOT update access stats.
  recall_raw     — verbatim dialog replay; DOES update access stats.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone

from corvidae.hooks import CorvidaePlugin, get_dependency, hookimpl
from corvidae.memory import _format_age, _fts_match_query
from corvidae.tool import Tool, tool_to_schema

logger = logging.getLogger("corvidae.tools.memory_tools")

# Rough characters-per-token approximation for the token cap.
_CHARS_PER_TOKEN = 4


class MemoryToolsPlugin(CorvidaePlugin):
    """Plugin registering search_memory and recall_raw agent tools.

    Depends on the memory plugin for DB access and channel-scope resolution.
    """

    depends_on = frozenset({"memory"})

    def __init__(self, pm=None) -> None:
        if pm is not None:
            self.pm = pm

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        await super().on_init(pm, config)

    def _memory(self):
        from corvidae.memory import MemoryPlugin
        return get_dependency(self.pm, "memory", MemoryPlugin)

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
            _ctx=None,
        ) -> str:
            """Search memory records using keyword search. Returns a numbered list with
            score, age, flags ([demoted]/[superseded]), and summary. Does not update
            access stats. Use include_demoted=False to restrict to active (indexed=1)
            records. Use after/before (YYYY-MM-DD or ISO-8601, UTC) to filter by
            creation date. Use tags=[...] to filter by topic_tags. Results are
            token-capped at max_tokens (approximate)."""
            memory = plugin._memory()

            if _ctx is None or _ctx.channel is None:
                return "Error: no channel context available"

            calling_channel = _ctx.channel
            scope = memory._channel_scope(calling_channel.id)

            # Validate channel argument if supplied
            if channel is not None and channel not in scope:
                visible = ", ".join(sorted(scope))
                return (
                    f"Error: channel '{channel}' is outside your visible scope. "
                    f"Visible channels: {visible}"
                )

            # Parse date filters (naive values treated as UTC per plan)
            after_ts: float | None = None
            before_ts: float | None = None
            if after is not None:
                try:
                    dt = datetime.fromisoformat(after)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    after_ts = dt.timestamp()
                except ValueError:
                    return (
                        f"Error: invalid 'after' value '{after}'. "
                        "Use YYYY-MM-DD or ISO-8601 format."
                    )
            if before is not None:
                try:
                    dt = datetime.fromisoformat(before)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    before_ts = dt.timestamp()
                except ValueError:
                    return (
                        f"Error: invalid 'before' value '{before}'. "
                        "Use YYYY-MM-DD or ISO-8601 format."
                    )

            db = await memory._ensure_schema()
            if db is None:
                return "Error: memory schema not available"

            # Build FTS match expression
            match = _fts_match_query(query)
            if match is None:
                return "No results: query has no searchable terms."

            # Scope: restrict to channel arg (narrowing) or calling channel's full scope
            scope_to_use = [channel] if channel is not None else scope
            placeholders = ",".join("?" for _ in scope_to_use)

            conditions = [f"m.channel_id IN ({placeholders})"]
            params: list = list(scope_to_use)

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

            where = " AND ".join(conditions)
            sql = (
                "SELECT m.id, m.summary, m.created_at, m.indexed, m.superseded_by, "
                "memory_fts.rank "
                "FROM memory_fts JOIN memory m ON m.id = memory_fts.rowid "
                f"WHERE memory_fts MATCH ? AND {where} "
                "ORDER BY rank LIMIT 50"
            )

            try:
                async with db.execute(sql, (match, *params)) as cursor:
                    rows = await cursor.fetchall()
            except Exception:
                logger.warning("search_memory FTS query failed", exc_info=True)
                return "Error: search failed internally"

            if not rows:
                return "No memories found matching your query."

            now = time.time()
            max_chars = max_tokens * _CHARS_PER_TOKEN
            lines: list[str] = []
            used_chars = 0
            truncated = False
            for mid, summary, created_at, indexed, superseded_by, rank in rows:
                age = max(0.0, now - created_at)
                flags = []
                if superseded_by is not None:
                    flags.append("[superseded]")
                elif indexed == 0:
                    flags.append("[demoted]")
                flag_str = (" " + " ".join(flags)) if flags else ""
                age_str = _format_age(age)
                # FTS5 rank is negative (more negative = more relevant);
                # display as a positive band-less score.
                score = -(rank or 0.0)
                line = f"{mid}. {score:.4f} ({age_str}){flag_str} {summary}"
                if used_chars + len(line) + 1 > max_chars:
                    truncated = True
                    break
                lines.append(line)
                used_chars += len(line) + 1

            if truncated:
                lines.append("[truncated — token limit reached]")

            return "\n".join(lines)

        async def recall_raw(
            memory_id: int,
            max_tokens: int = 1000,
            _ctx=None,
        ) -> str:
            """Replay the raw dialog messages for a memory record (remember harder).
            Returns a participants header followed by verbatim role:content lines.
            Works on demoted, superseded, and tombstoned records. Updates access stats."""
            memory = plugin._memory()

            if _ctx is None or _ctx.channel is None:
                return "Error: no channel context available"

            calling_channel = _ctx.channel
            scope = memory._channel_scope(calling_channel.id)

            db = await memory._ensure_schema()
            if db is None:
                return "Error: memory schema not available"

            # Fetch the record (no indexed/redacted filter: "remember harder")
            async with db.execute(
                "SELECT id, channel_id, summary, msg_id_start, msg_id_end, participants "
                "FROM memory WHERE id = ?",
                (memory_id,),
            ) as cursor:
                row = await cursor.fetchone()

            if row is None:
                return f"Error: memory {memory_id} not found"

            mid, channel_id, _summary, msg_id_start, msg_id_end, participants_json = row

            # Scope check — no stat bump on rejection
            if channel_id not in scope:
                return (
                    f"Error: memory {memory_id} (channel '{channel_id}') is outside "
                    f"your scope. Visible channels: {', '.join(sorted(scope))}"
                )

            # Fetch raw messages with channel_id predicate.
            # message_log uses a global AUTOINCREMENT id shared by all channels;
            # the numeric range [msg_id_start, msg_id_end] contains interleaved
            # foreign-channel rows. The channel_id predicate is load-bearing.
            async with db.execute(
                "SELECT id, message FROM message_log "
                "WHERE channel_id = ? AND id BETWEEN ? AND ? ORDER BY id",
                (channel_id, msg_id_start, msg_id_end),
            ) as cursor:
                msg_rows = await cursor.fetchall()

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

            # Build output with token cap (approximate: chars / 4)
            max_chars = max_tokens * _CHARS_PER_TOKEN
            output_parts = [header]
            used_chars = len(header)
            truncated = False

            for _row_id, msg_json in msg_rows:
                try:
                    msg = json.loads(msg_json)
                except Exception:
                    continue
                role = msg.get("role", "?")
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

            return "\n".join(output_parts)

        tool_registry.append(Tool(
            name="search_memory",
            fn=search_memory,
            schema=tool_to_schema(search_memory),
        ))
        tool_registry.append(Tool(
            name="recall_raw",
            fn=recall_raw,
            schema=tool_to_schema(recall_raw),
        ))
