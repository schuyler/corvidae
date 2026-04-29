"""DreamPlugin — background memory consolidation for the Corvidae agent daemon.

Analogy: REM sleep for an AI agent. During idle periods the plugin queries
recent conversation history from SQLite, extracts facts worth persisting, and
writes them to MEMORY.md (the user's long-term memory file).

Design decisions:
- Uses the ``on_idle`` hook so it runs when no messages are pending.
- Queries the last N assistant messages per channel (default 20) — short enough
  to fit in a prompt but long enough to capture multi-turn conversations.
- Idempotent: checks existing MEMORY.md content before writing to avoid
  duplication.
- Configurable interval via ``config.dream.interval_seconds`` (default 300).

Future enhancements:
- LLM-assisted summarization instead of raw text extraction.
- Wiki-based memory consolidation (write to OtterWiki pages).
- Configurable fact categories (preferences, learnings, tool tips, etc.).
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)


class DreamPlugin:
    """Periodically reviews recent conversation history and updates MEMORY.md."""

    # How many raw rows to load per channel (buffer for non-assistant messages).
    MAX_ROWS = 40
    # How many assistant messages per channel to review after filtering.
    MAX_MESSAGES = 20
    # Maximum characters of text to process per channel (rough limit on prompt size).
    MAX_CHARS = 8000
    # Minimum seconds between dream cycles (configurable).
    DEFAULT_INTERVAL = 300

    def __init__(self, workspace_root: str | Path) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.interval_seconds: int = self.DEFAULT_INTERVAL
        self._last_dream_time: float = 0.0
        self._db_path: Path | None = None

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    async def on_start(self, config: dict) -> None:
        """Read dream config and locate the SQLite DB."""
        dream_cfg = config.get("dream", {})
        self.interval_seconds = dream_cfg.get("interval_seconds", self.DEFAULT_INTERVAL)
        self._db_path = self._locate_db()

    async def on_idle(self) -> None:
        """Trigger a dream cycle if enough time has elapsed and DB is ready."""
        now = time.time()
        if (now - self._last_dream_time) < self.interval_seconds:
            return
        self._last_dream_time = now

        if self._db_path is None or not self._db_path.exists():
            logger.debug("dream: no sessions.db found, skipping")
            return

        await self._dream_cycle()

    # ------------------------------------------------------------------
    # DB discovery
    # ------------------------------------------------------------------

    def _locate_db(self) -> Path | None:
        """Find sessions.db within the workspace tree.

        Search order:
        1. Direct paths (workspace/corvidae/sessions.db, workspace/sessions.db)
        2. Recursive search up to depth 3 from workspace_root
        """
        # Check direct locations first
        candidates = [
            self.workspace_root / "corvidae" / "sessions.db",
            self.workspace_root / "sessions.db",
        ]

        for path in candidates:
            if path.exists():
                logger.info("dream: located sessions.db at %s", path)
                return path

        # Recursive search
        for depth in range(1, 4):
            patterns = ["**/sessions.db"]
            for pattern in patterns:
                matches = list(self.workspace_root.glob(pattern))
                for match in matches:
                    if (match.resolve().relative_to(self.workspace_root)).parent.parts.count("/") <= depth:
                        logger.info("dream: located sessions.db at %s", match)
                        return match

        logger.debug(
            "dream: sessions.db not found under %s; dream cycle will be skipped",
            self.workspace_root,
        )
        return None

    # ------------------------------------------------------------------
    # Dream cycle
    # ------------------------------------------------------------------

    async def _dream_cycle(self) -> None:
        """Main dream cycle: load recent messages → extract facts → write to MEMORY.md."""
        if self._db_path is None:
            return

        channels = await self._load_recent_assistant_messages()

        if not channels:
            logger.debug("dream: no recent assistant messages found")
            return

        total_assistant_msgs = sum(len(v) for v in channels.values())
        logger.info(
            "dream: reviewing %d assistant messages across %d channel(s)",
            total_assistant_msgs, len(channels),
        )

        # Build per-channel text blocks (stripped of thinking blocks)
        facts_by_channel: dict[str, str] = {}
        for ch_id, msgs in channels.items():
            text_parts: list[str] = []
            for msg in msgs:
                content = msg.get("content", "") or ""
                if not content.strip():
                    continue
                # Strip <think>...</think> blocks to reduce token count
                cleaned = re.sub(r"</think>", "", content)
                cleaned = re.sub(
                    r"#[\s]*thinking.*?(?=\n|$)", "<thinking>", cleaned, flags=re.DOTALL
                )
                text_parts.append(cleaned)

            channel_text = "\n\n---\n\n".join(text_parts[:5])  # limit to 5 most recent
            if channel_text.strip():
                facts_by_channel[ch_id] = channel_text

        await self._write_new_facts(facts_by_channel)

    async def _load_recent_assistant_messages(self) -> dict[str, list[dict]]:
        """Load recent assistant messages grouped by channel.

        Returns:
            Mapping of channel_id → list of assistant message dicts (newest first).
        """
        async with aiosqlite.connect(str(self._db_path)) as db:
            # Load recent rows; we filter for assistant messages in Python
            cur = await db.execute(
                "SELECT channel_id, message FROM message_log ORDER BY timestamp DESC LIMIT ?",
                (self.MAX_ROWS,),
            )
            rows = await cur.fetchall()

        channels: dict[str, list[dict]] = {}
        for channel_id, msg_json in rows:
            try:
                msg = json.loads(msg_json)
            except (json.JSONDecodeError, TypeError):
                continue

            if msg.get("role") != "assistant":
                continue

            channels.setdefault(channel_id, []).append(msg)

        # Limit per channel and reverse so newest-first stays newest-first
        for ch in channels:
            channels[ch] = channels[ch][:self.MAX_MESSAGES]

        return channels

    # ------------------------------------------------------------------
    # Memory writing
    # ------------------------------------------------------------------

    async def _write_new_facts(self, facts_by_channel: dict[str, str]) -> None:
        """Check MEMORY.md and append new facts that aren't already present."""
        memory_path = self.workspace_root / "memory" / "MEMORY.md"

        existing_content = ""
        if memory_path.exists():
            existing_content = memory_path.read_text(encoding="utf-8")

        # Extract the Long-term Memory section for deduplication
        long_term_section = self._extract_long_term_memory(existing_content)

        # Build a set of existing facts (split on bullet points and sentences)
        existing_facts = set(self._flatten_facts(long_term_section))

        new_facts: list[str] = []
        for ch_id, text in facts_by_channel.items():
            extracted = self._extract_sentences(text)
            for sentence in extracted:
                if len(sentence) < 25:
                    continue
                # Normalize for deduplication check
                normalized = self._normalize_for_dedup(sentence)
                if normalized not in existing_facts:
                    new_facts.append(sentence)

        if not new_facts:
            logger.info("dream: no new facts to persist")
            return

        # Deduplicate and cap
        new_facts = list(dict.fromkeys(new_facts))[:20]

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())
        new_fact_line = "- " + " ".join(new_facts)

        updated_content = self._insert_into_memory(existing_content, new_fact_line)

        memory_path.parent.mkdir(parents=True, exist_ok=True)
        memory_path.write_text(updated_content, encoding="utf-8")
        logger.info("dream: persisted %d facts to MEMORY.md", len(new_facts))

    def _extract_long_term_memory(self, content: str) -> str:
        """Extract the Long-term Memory section from MEMORY.md content."""
        match = re.search(
            r"## Long-term Memory\n(.*?)(?=^##|\Z)",
            content,
            re.DOTALL | re.MULTILINE,
        )
        return match.group(1) if match else ""

    def _flatten_facts(self, section: str) -> set[str]:
        """Flatten a Long-term Memory section into normalized fact strings."""
        if not section.strip():
            return set()

        facts = set()
        for line in section.split("\n"):
            line = line.strip("- \t")
            if not line:
                continue
            # Normalize whitespace and punctuation
            normalized = re.sub(r"\s+", " ", line).strip().lower()
            if len(normalized) > 10:
                facts.add(normalized)
        return facts

    def _extract_sentences(self, text: str) -> list[str]:
        """Extract sentences from text, filtering out code blocks and links."""
        # Remove code fences
        text = re.sub(r"```[\s\S]*?```", "", text)
        # Split on sentence boundaries
        raw_sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in raw_sentences if len(s.strip()) > 0]

    def _normalize_for_dedup(self, sentence: str) -> str:
        """Normalize a sentence for deduplication comparison."""
        # Remove timestamps and special markers
        cleaned = re.sub(r"\[\d{4}-\d{2}-\d{2}.*?\]", "", sentence)
        cleaned = re.sub(r"#[a-z_]+", "", cleaned)  # remove thinking tags
        cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
        return cleaned

    def _insert_into_memory(self, content: str, new_fact_line: str) -> str:
        """Insert a new fact into MEMORY.md's Long-term Memory section."""
        match = re.search(
            r"(## Long-term Memory\n)",
            content,
            re.MULTILINE,
        )

        if match:
            # Insert after the header
            insert_pos = match.end()
            return (
                content[:insert_pos] + new_fact_line + "\n" + content[insert_pos:]
            )

        # No Long-term Memory section — create one before User Information
        user_info_marker = "## User Information"
        if user_info_marker in content:
            idx = content.index(user_info_marker)
            return (
                content[:idx]
                + f"## Long-term Memory\n{new_fact_line}\n\n"
                + content[idx:]
            )

        # Append at the end
        return content.rstrip() + "\n\n## Long-term Memory\n" + new_fact_line + "\n"
