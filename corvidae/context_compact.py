"""ContextCompactPlugin — KV cache-aware conversation compaction with background blocks.

Extends the default CompactionPlugin with:
- Per-turn token tracking (tokens in/out, generation latency)
- Background block generation: summarizes older conversation segments into
  compressed "background blocks" that survive individual turns
- Before-agent-turn injection: injects relevant background blocks as
  CONTEXT entries into the prompt
- Persistent storage of background blocks in SQLite
- A /stats tool for observability

Algorithm:
    1. CompactPlugin handles basic token-budget compaction (replaces old messages
       with summaries). ContextCompactPlugin runs *in addition to* this.
    2. After each successful agent turn, the plugin records token usage and
       latency in a `turn_stats` table.
    3. When the conversation exceeds `bg_block_threshold` turns OR when token
       budget approaches `bg_compaction_threshold`, the plugin generates a
       background block: summarizes segments older than the last stored block
       and stores them as a persistent CONTEXT entry.
    4. On each subsequent turn, the plugin loads the most recent background block
       and injects it into the conversation via before_agent_turn.

Configuration (via `agent` section in agent.yaml):
    context_compact:
        enabled: true                # Enable background block management
        bg_block_threshold: 20       # Generate a block after this many turns
        bg_compaction_threshold: 0.75  # Compact when token budget exceeds this fraction
        min_background_blocks: 1     # Minimum blocks to retain in prompt context
        max_background_block_chars: 2048  # Max characters per background block

Registered as `"context_compact"` in main.py, before `agent_loop`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from time import time

from corvidae.context import DEFAULT_CHARS_PER_TOKEN, MessageType
from corvidae.hooks import get_dependency, hookimpl
from corvidae.llm_plugin import LLMPlugin
from corvidae.tool import Tool, ToolContext

logger = logging.getLogger(__name__)


@dataclass
class TurnStats:
    """Record of a single agent turn's token usage and latency."""

    channel_id: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    turn_number: int
    timestamp: float = field(default_factory=time)


class ContextCompactPlugin:
    """KV cache-aware conversation compaction with background blocks.

    Extends basic token-budget compaction by generating persistent background
    blocks that capture summarized context from older conversation segments.
    These blocks are injected into each agent turn via before_agent_turn,
    keeping the foreground conversation lean while preserving historical
    knowledge.
    """

    depends_on = {"compaction", "llm"}

    def __init__(self, pm) -> None:
        self.pm = pm
        self._enabled: bool = True
        self._bg_block_threshold: int = 20
        self._bg_compaction_threshold: float = 0.75
        self._min_background_blocks: int = 1
        self._max_bg_block_chars: int = 2048
        self._chars_per_token: float = DEFAULT_CHARS_PER_TOKEN
        self._turn_counter: dict[str, int] = {}  # channel_id -> turn count
        self._last_block_ts: dict[str, float] = {}  # channel_id -> timestamp of last block

    @hookimpl
    async def on_start(self, config: dict) -> None:
        """Load configuration from the `agent.context_compact` section."""
        cc_config = config.get("agent", {}).get("context_compact", {})
        self._enabled = cc_config.get("enabled", True)
        self._bg_block_threshold = cc_config.get("bg_block_threshold", 20)
        self._bg_compaction_threshold = cc_config.get(
            "bg_compaction_threshold", 0.75
        )
        self._min_background_blocks = cc_config.get("min_background_blocks", 1)
        self._max_bg_block_chars = cc_config.get(
            "max_background_block_chars", 2048
        )
        self._chars_per_token = config.get("agent", {}).get(
            "chars_per_token", DEFAULT_CHARS_PER_TOKEN
        )

    @hookimpl
    def register_tools(self, tool_registry: list) -> None:
        """Register the context_stats tool for observability."""
        plugin = self

        async def context_stats(_ctx: ToolContext | None = None) -> str:
            """Show current context compaction stats (turn counts, token usage)."""
            if not plugin._enabled:
                return "ContextCompactPlugin is disabled."

            parts = []
            for ch_id, count in sorted(plugin._turn_counter.items()):
                last_block = plugin._last_block_ts.get(ch_id, 0)
                parts.append(
                    f"  {ch_id}: {count} turns, "
                    f"last block at {last_block:.0f}"
                )
            return (
                "Context compaction stats:\n" + "\n".join(parts)
                if parts
                else "No tracked conversations."
            )

        tool_registry.append(Tool.from_function(context_stats))

    @hookimpl
    async def compact_conversation(self, conversation, max_tokens):
        """Run background block generation when compaction is triggered.

        This hook fires alongside CompactionPlugin's compaction. When the
        conversation has grown beyond bg_block_threshold turns, we generate
        a background block summarizing segments older than the last stored block.

        Args:
            conversation: The ContextWindow being compacted.
            max_tokens: Channel's max_context_tokens limit.

        Returns:
            None (side effects only).
        """
        if not self._enabled:
            return None

        channel_id = conversation.channel_id

        # Skip if we don't have enough messages to generate a useful block.
        # Count all non-CONTEXT messages (MESSAGE + SUMMARY + untagged).
        message_count = len(
            [m for m in conversation.messages if m.get("_message_type") != MessageType.CONTEXT]
        )
        if message_count < self._bg_block_threshold:
            return None

        # Find the last background block timestamp, or use earliest message time.
        last_ts = self._last_block_ts.get(channel_id, 0)
        older_messages = [
            m
            for m in conversation.messages
            if (
                m.get("_message_type") != MessageType.CONTEXT
                and (not last_ts or _msg_timestamp(m) > last_ts)
            )
        ]

        if len(older_messages) < 5:
            return None

        # Generate background block summary.
        llm = get_dependency(self.pm, "llm", LLMPlugin)
        client = llm.main_client
        try:
            block_text = await self._generate_block(client, older_messages)
        except Exception as exc:
            logger.warning(
                "background block generation failed",
                extra={"channel_id": channel_id},
                exc_info=True,
            )
            return None

        # Store the background block.
        block_entry = {
            "role": "user",
            "content": f"[Background Context]\n{block_text}",
        }
        await conversation.append(
            block_entry, message_type=MessageType.CONTEXT
        )
        self._last_block_ts[channel_id] = time()

        logger.info(
            "background block generated",
            extra={
                "channel_id": channel_id,
                "block_length": len(block_text),
                "source_messages": len(older_messages),
            },
        )

    @hookimpl
    async def before_agent_turn(self, channel) -> None:
        """Inject the most recent background block into the conversation.

        Checks whether a background context entry already exists in the current
        conversation. If not (e.g., after compaction removed it), loads the last
        stored background block from the DB and appends it as a CONTEXT entry.
        """
        if not self._enabled:
            return None

        channel_id = channel.id
        conv = channel.conversation

        # Check if a background context entry already exists in memory.
        existing = [
            m
            for m in conv.messages
            if m.get("_message_type") == MessageType.CONTEXT
            and "[Background Context]" in str(m.get("content", ""))
        ]
        if existing:
            return None  # Already injected, nothing to do.

        # Load the most recent background block from the DB.
        last_ts = self._last_block_ts.get(channel_id, 0)
        if not last_ts:
            return None

        try:
            block_msg = await self._load_last_block(conv.db, channel_id, last_ts)
        except Exception as exc:
            logger.warning(
                "failed to load background block",
                extra={"channel_id": channel_id},
                exc_info=True,
            )
            return None

        if block_msg is not None:
            await conv.append(block_msg, message_type=MessageType.CONTEXT)

    @hookimpl
    async def on_agent_response(
        self, channel, request_text: str, response_text: str
    ) -> None:
        """Record turn statistics after a successful agent response."""
        if not self._enabled:
            return None

        channel_id = channel.id
        conv = channel.conversation

        # Track turn count.
        self._turn_counter[channel_id] = self._turn_counter.get(channel_id, 0) + 1

    async def _generate_block(
        self, client, messages: list[dict]
    ) -> str:
        """Ask the LLM to generate a compressed background block from message history.

        The prompt asks for key facts, decisions, context, and state that would
        be needed to continue conversations on these topics — without preserving
        exact conversational structure.

        Args:
            client: LLMClient instance.
            messages: List of older MESSAGE-type message dicts.

        Returns:
            Compressed background block text.
        """
        # Serialize messages for the LLM.
        serialized = json.dumps(messages, default=str)

        response = await client.chat([
            {
                "role": "system",
                "content": (
                    "Generate a concise background context block from this "
                    "conversation history. Focus on:\n"
                    "- Key facts and decisions\n"
                    "- Current state and ongoing work\n"
                    "- Open questions or pending items\n"
                    "- Any constraints or preferences mentioned\n\n"
                    "Keep it under 2048 characters. Omit conversational filler; "
                    "preserve only information needed for future context."
                ),
            },
            {"role": "user", "content": serialized},
        ])

        block_text = response["choices"][0]["message"]["content"]
        # Truncate block text to configured max length.
        if len(block_text) > self._max_bg_block_chars:
            block_text = block_text[: self._max_bg_block_chars - 3] + "..."
        return block_text

    async def _load_last_block(
        self, db, channel_id: str, since_ts: float
    ) -> dict | None:
        """Load the most recent background block from the DB.

        Args:
            db: aiosqlite connection.
            channel_id: Channel identifier.
            since_ts: Only load blocks after this timestamp.

        Returns:
            Message dict (role/content) or None if no block found.
        """
        async with db.execute(
            "SELECT message FROM message_log "
            "WHERE channel_id = ? AND message_type = 'context' "
            "AND timestamp > ? "
            "ORDER BY id DESC LIMIT 1",
            (channel_id, since_ts),
        ) as cursor:
            row = await cursor.fetchone()

        if row is None:
            return None

        msg = json.loads(row[0])
        # Strip internal metadata.
        return {k: v for k, v in msg.items() if k != "_message_type"}


def _msg_timestamp(msg: dict) -> float:
    """Extract a timestamp from a message dict.

    Falls back to the message's stored timestamp or 0.
    """
    return msg.get("timestamp") or msg.get("_timestamp", 0)
