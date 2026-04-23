"""AgentLoopPlugin — wires the agent loop into the hook system."""

import logging
from collections.abc import Callable
from pathlib import Path

import aiosqlite

from sherman.agent_loop import run_agent_loop, strip_reasoning_content, strip_thinking, tool_to_schema
from sherman.conversation import ConversationLog, init_db
from sherman.hooks import hookimpl
from sherman.llm import LLMClient
from sherman.prompt import resolve_system_prompt

logger = logging.getLogger(__name__)


class AgentLoopPlugin:
    """Plugin that wires the agent loop into the hook system."""

    def __init__(self, pm) -> None:  # pm is untyped: pluggy.PluginManager has no typed interface for .registry
        self.pm = pm
        self.client: LLMClient | None = None
        self.db: aiosqlite.Connection | None = None
        self.tools: dict[str, Callable] = {}
        self.tool_schemas: list[dict] = []
        self.base_dir: Path = Path(".")

    @hookimpl
    async def on_start(self, config: dict) -> None:
        # 1. Create and start LLM client.
        # Missing 'llm' key uses empty dict, causing KeyError on required
        # fields (base_url, model) — fail-fast on misconfiguration.
        self.tools = {}
        self.tool_schemas = []
        self.base_dir = config.get("_base_dir", Path("."))
        llm_config = config.get("llm", {})
        self.client = LLMClient(
            base_url=llm_config["base_url"],
            model=llm_config["model"],
            api_key=llm_config.get("api_key"),
        )
        await self.client.start()

        # 2. Open SQLite database (only if not already injected for testing)
        if self.db is None:
            db_path = config.get("daemon", {}).get("session_db", "sessions.db")
            self.db = await aiosqlite.connect(db_path)
            await init_db(self.db)

        # 3. Collect tools from all plugins via register_tools hook (sync).
        tool_fns: list = []
        self.pm.hook.register_tools(tool_registry=tool_fns)
        for fn in tool_fns:
            self.tools[fn.__name__] = fn
            self.tool_schemas.append(tool_to_schema(fn))

    @hookimpl
    async def on_message(self, channel, sender: str, text: str) -> None:
        if self.client is None:
            logger.error("on_message called but LLM client is not initialized")
            await self.pm.ahook.send_message(
                channel=channel,
                text="Sorry, the agent is not configured.",
            )
            return

        # 1. Lazy-initialize conversation on the channel
        await self._ensure_conversation(channel)
        conv = channel.conversation
        resolved = self.pm.registry.resolve_config(channel)

        # 2. Append user message to conversation log (persisted)
        await conv.append({"role": "user", "content": text})

        # 3. Compact if approaching context limit
        await conv.compact_if_needed(self.client, resolved["max_context_tokens"])

        # 4. Build prompt and run agent loop
        messages = conv.build_prompt()
        # NOTE: messages_before must use len(messages) from build_prompt(), NOT
        # len(conv.messages). build_prompt() prepends the system message, so
        # len(messages) == len(conv.messages) + 1. Using len(conv.messages) would
        # cause an off-by-one and double-persist the last user message.
        messages_before = len(messages)

        try:
            raw_response = await run_agent_loop(
                self.client, messages, self.tools, self.tool_schemas
            )
        except Exception:
            logger.exception("run_agent_loop failed for channel %s", channel.id)
            await self.pm.ahook.send_message(
                channel=channel,
                text="Sorry, I encountered an error and could not process your message.",
            )
            return

        # 5. Persist new messages appended by run_agent_loop
        new_messages = messages[messages_before:]
        # Persist BEFORE stripping — conv.append serializes the dict at call
        # time, so the full message (including reasoning_content) is saved to
        # disk before in-memory stripping occurs below.
        for msg in new_messages:
            await conv.append(msg)

        # 6. Thinking token handling for active history.
        # Only strip the newly appended messages — earlier messages were
        # already stripped on prior turns.
        if not resolved["keep_thinking_in_history"]:
            strip_reasoning_content(new_messages)

        # 7. Strip thinking for display and send response
        display_response = strip_thinking(raw_response)

        await self.pm.ahook.on_agent_response(
            channel=channel,
            request_text=text,
            response_text=display_response,
        )
        await self.pm.ahook.send_message(
            channel=channel,
            text=display_response,
        )

    @hookimpl
    async def on_stop(self) -> None:
        if self.client:
            await self.client.stop()
        if self.db:
            await self.db.close()

    async def _ensure_conversation(self, channel) -> None:
        """Lazy-initialize ConversationLog on a channel if not present."""
        if channel.conversation is not None:
            return
        conv = ConversationLog(self.db, channel.id)
        resolved = self.pm.registry.resolve_config(channel)
        conv.system_prompt = resolve_system_prompt(resolved["system_prompt"], self.base_dir)
        await conv.load()
        channel.conversation = conv
