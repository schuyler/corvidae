"""Shared async helper functions for integration tests."""

import asyncio
import contextlib

import logging
import re
import time

import pytest
import aiosqlite

from corvidae.hooks import hookimpl


def sent(pm, kind: type) -> list[tuple[object, object]]:
    """Return (channel, event) for every ``send`` of an event of ``kind``.

    Reads the AsyncMock that ``build_plugin_and_channel`` installs as
    ``pm.ahook.send``, in call order. ``kind`` is an outbound event class
    from corvidae.outbound, e.g. ``sent(pm, OutboundMessage)``.
    """
    return [
        (call.kwargs["channel"], call.kwargs["event"])
        for call in pm.ahook.send.await_args_list
        if isinstance(call.kwargs["event"], kind)
    ]


def sent_texts(pm, kind: type) -> list[str]:
    """Return the ``text`` of every sent event of ``kind``, in call order."""
    return [event.text for _channel, event in sent(pm, kind)]


async def build_plugin_and_channel(
    agent_defaults=None,
    channel_config=None,
    *,
    mock_send: bool = True,
    mock_on_agent_response: bool = True,
    db_path: str = ":memory:",
):
    """Assemble plugin graph with an in-memory (or ``db_path`` file) DB.

    Returns (plugin, channel, db). Callers are responsible for teardown:
    task_plugin.on_stop() and db.close().

    Args:
        agent_defaults: Override AGENT_DEFAULTS for channel registry.
        channel_config: Override default ChannelConfig.
        mock_send: If True, replace the outbound ``send`` hook with an
            AsyncMock; read what was sent with ``sent()``.
        mock_on_agent_response: If True, replace on_agent_response hook with AsyncMock.
        db_path: The session DB; a file path lets a second connection (e.g.
            a read-only probe) see what the turn persisted.
    """
    from unittest.mock import AsyncMock

    from corvidae.agent import Agent
    from corvidae.channel import ChannelConfig, ChannelRegistry
    from corvidae.persistence import PersistencePlugin, init_db
    from corvidae.hooks import create_plugin_manager
    from corvidae.task import TaskPlugin
    from corvidae.thinking import ThinkingPlugin

    _AGENT_DEFAULTS = {
        "system_prompt": "You are a test assistant.",
        "max_context_tokens": 8000,
        "reasoning_history": "drop",
    }

    if agent_defaults is None:
        agent_defaults = _AGENT_DEFAULTS

    db = await aiosqlite.connect(db_path)
    await init_db(db)

    pm = create_plugin_manager()
    registry = ChannelRegistry()
    registry.agent_defaults = agent_defaults
    pm.register(registry, name="registry")

    if mock_send:
        pm.ahook.send = AsyncMock()
    if mock_on_agent_response:
        pm.ahook.on_agent_response = AsyncMock()

    task_plugin = TaskPlugin()
    pm.register(task_plugin, name="task")
    await task_plugin.on_init(pm=pm, config={})
    await task_plugin.on_start(config={})

    persistence = PersistencePlugin()
    persistence.db = db
    persistence._registry = registry
    pm.register(persistence, name="persistence")

    thinking_plugin = ThinkingPlugin()
    pm.register(thinking_plugin, name="thinking")
    await thinking_plugin.on_init(pm=pm, config={})

    plugin = Agent()
    pm.register(plugin, name="agent")
    plugin.pm = pm
    plugin._registry = registry

    channel = registry.get_or_create(
        "test",
        "scope1",
        config=channel_config or ChannelConfig(),
    )

    return plugin, channel, db


async def drain(plugin, channel):
    """Drain the channel's SerialQueue."""
    if channel.id in plugin.queues:
        await plugin.queues[channel.id].drain()


async def drain_task_queue(plugin):
    """Wait for all TaskQueue tasks to complete including on_complete callbacks."""
    task_plugin = plugin.pm.get_plugin("task")
    if task_plugin and task_plugin.task_queue:
        await task_plugin.task_queue.queue.join()
        await asyncio.sleep(0)


class FakeForkAgent:
    """Stands in for Agent in fork and view tests (GitHub issues #51, #380, #388).

    Forks are enabled for ``stages``; each fork is recorded with its
    attribution and answers ``text``, or raises ``error``. ``fail_marker``
    limits the error to instructions containing it.

    It is also view-capable (``view_agent`` duck-types on ``view_inputs``):
    ``view_inputs`` returns ``ViewInputs("IDENTITY\n\nREGISTER", max_tokens,
    "system", <policy bound via apply_reasoning_history>)``. ``max_tokens``
    and ``reasoning_history`` ("drop" | "exchange" | "keep") are settable per
    test; ``mid_conversation_system_role`` is the role compaction renders
    mid-conversation system messages with. ``view_calls`` records the channel
    ids view_inputs was asked about.
    """

    #: The system prompt every fake view starts with.
    SYSTEM_PROMPT = "IDENTITY\n\nREGISTER"

    def __init__(self, stages, text="", error=None, fail_marker=None, model=None,
                 max_tokens=100_000, reasoning_history="drop"):
        self.stages = set(stages)
        self.model = model
        self.text = text
        self.error = error
        self.fail_marker = fail_marker
        self.calls = []
        self.max_tokens = max_tokens
        self.reasoning_history = reasoning_history
        self.mid_conversation_system_role = "system"
        self.view_calls: list[str] = []
        # Settable per test (a long prompt makes the system prompt alone
        # exceed a small budget).
        self.system_prompt = self.SYSTEM_PROMPT

    def fork_enabled(self, stage):
        return stage in self.stages

    async def view_inputs(self, channel_id):
        """The channel's ViewInputs: a fixed prompt, this agent's budget, the
        system mid role and the reasoning policy bound to ``reasoning_history``."""
        from corvidae.channel import ReasoningHistory
        from corvidae.fork import ViewInputs
        from corvidae.thinking import apply_reasoning_history

        self.view_calls.append(channel_id)
        policy = ReasoningHistory(self.reasoning_history)
        return ViewInputs(
            self.system_prompt,
            self.max_tokens,
            "system",
            lambda messages: apply_reasoning_history(messages, policy),
        )

    async def fork(self, channel, instruction, *, stage, cut=None, extra_body=None):
        from corvidae.attribution import get_attribution
        from corvidae.fork import ForkResult

        self.calls.append({
            "instruction": instruction, "stage": stage, "cut": cut, "extra_body": extra_body,
            "attribution": dict(get_attribution()),
        })
        if self.error is not None and (self.fail_marker is None or self.fail_marker in instruction):
            raise self.error
        # ``model`` (issue #299) is the model the fork was sent with; passed
        # only when set so tests that ignore it run on any ForkResult.
        extra = {"model": self.model} if self.model is not None else {}
        return ForkResult(
            text=self.text, message={"role": "assistant", "content": self.text}, **extra
        )


class RecordingClient:
    """An LLMClient stand-in for view calls: records every chat request with
    its attribution, answers ``text`` (or raises ``error``)."""

    def __init__(self, text="", error=None, model="stub-model"):
        self.text = text
        self.error = error
        self.model = model
        self.chat_calls: list[dict] = []

    def request_model(self, extra_body=None):
        """The model a call is sent with, as LLMClient.request_model does."""
        return (extra_body or {}).get("model", self.model)

    async def chat(self, messages, tools=None, extra_body=None):
        from corvidae.attribution import get_attribution

        self.chat_calls.append({
            "messages": messages, "tools": tools, "extra_body": extra_body,
            "attribution": dict(get_attribution()),
        })
        if self.error is not None:
            raise self.error
        return {"choices": [{"message": {"role": "assistant", "content": self.text}}]}

    async def embed(self, texts, kind=None):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


#: Stages the kernel plugins register (compaction, subagent), for fixtures that
#: build an LLMPlugin without on_start and so never run register_llm_stages.
CORE_LLM_STAGES = {"compaction": "chat", "subagent": "chat"}


#: A minimal llm: section: one chat profile, no embedding profile.
STUB_LLM_SECTION = {"main": {"base_url": "http://stub.test/v1", "model": "stub"}}


def with_chat_llm(config: dict | None = None) -> dict:
    """``config`` with an llm: section added when it has none, so a plugin
    that parses llm: in on_init gets the default chat profile every real
    config has."""
    config = dict(config or {})
    config.setdefault("llm", STUB_LLM_SECTION)
    return config


def stub_llm_config(plugin, llm_section: dict | None = None) -> None:
    """Give an LLMPlugin that never ran on_start its parsed llm: section, so
    get_client resolves the clients a test assigns to ``plugin._clients``."""
    from corvidae.llm_plugin import parse_llm_config

    plugin._config = parse_llm_config(llm_section or STUB_LLM_SECTION)


def collect_llm_stages(plugin, name):
    """The stages ``plugin`` registers through the register_llm_stages hook,
    collected on a real plugin manager as LLMPlugin.on_start does."""
    from corvidae.hooks import create_plugin_manager

    pm = create_plugin_manager()
    pm.register(plugin, name=name)
    stages: dict[str, str] = {}
    pm.hook.register_llm_stages(stages=stages)
    return stages


def recording_llm_plugin(main, background=None):
    """A real LLMPlugin whose ``get_client`` records the stage it is asked for
    (``plugin.stages_requested``) and serves ``main`` and, when given, a
    ``background`` client that no stage resolves to. The kernel stages are
    registered, as register_llm_stages would on start."""
    from corvidae.llm_plugin import LLMPlugin

    class _RecordingLLMPlugin(LLMPlugin):
        """LLMPlugin that logs each requested stage."""

        def __init__(self) -> None:
            super().__init__()
            self.stages_requested: list[str | None] = []

        def get_client(self, stage: str | None = None):
            self.stages_requested.append(stage)
            return super().get_client(stage)

    plugin = _RecordingLLMPlugin()
    plugin._stages = dict(CORE_LLM_STAGES)
    plugin._clients["main"] = main
    llm_section = dict(STUB_LLM_SECTION)
    if background is not None:
        plugin._clients["background"] = background
        llm_section["background"] = {"base_url": "http://background.test/v1", "model": "bg"}
    stub_llm_config(plugin, llm_section)
    return plugin


class LLMResponseRecorder:
    """Plugin recording the profile and attribution of every on_llm_response."""

    def __init__(self) -> None:
        self.responses: list[dict] = []

    @hookimpl
    async def on_llm_response(self, profile, model, attribution):
        self.responses.append({"profile": profile, "model": model, "attribution": attribution})


async def start_llm_plugin(monkeypatch, llm_config: dict, stages: dict, pm=None):
    """A real LLMPlugin, started from ``llm_config``, on ``pm`` (or a fresh one).

    ``stages`` is what an owner plugin registers through register_llm_stages.
    Every client is real, but ``aiohttp.ClientSession`` is replaced for the
    test by a fake that answers each POST with a canned text completion and
    records ``{"url", "json"}`` in the returned ``posts`` list, so a test sees
    which endpoint and body each call reached.

    Returns (plugin, posts). The caller stops the plugin.
    """
    import aiohttp

    from corvidae.hooks import create_plugin_manager
    from corvidae.llm_plugin import LLMPlugin
    from llm_response_fixtures import _make_text_response

    posts: list[dict] = []

    class _Response:
        status = 200
        headers: dict = {}

        def raise_for_status(self):
            pass

        async def json(self):
            return _make_text_response("ok")

        async def text(self):
            return ""

    class _Post:
        async def __aenter__(self):
            return _Response()

        async def __aexit__(self, *exc):
            return False

    class _Session:
        def __init__(self, headers=None):
            self.headers = headers or {}

        def post(self, url, json=None, timeout=None):
            posts.append({"url": url, "json": json})
            return _Post()

        async def close(self):
            pass

    monkeypatch.setattr(aiohttp, "ClientSession", _Session)

    table = dict(stages)

    class _StageOwner:
        @hookimpl
        def register_llm_stages(self, stages):
            stages.update(table)

    if pm is None:
        pm = create_plugin_manager()
    pm.register(_StageOwner(), name="stage_owner")
    plugin = LLMPlugin()
    pm.register(plugin, name="llm")
    config = {"llm": llm_config}
    await plugin.on_init(pm, config)
    await plugin.on_start(config)
    return plugin, posts


def window_of(channel_id, rowids, summary=False, contents=None):
    """A ContextWindow of alternating user/assistant MESSAGE rows tagged with
    ``rowids`` (contents default to "m<rowid>"), optionally after a summary."""
    from corvidae.context import ContextWindow, MessageType

    conv = ContextWindow(channel_id)
    conv.system_prompt = "You are Lex."
    if summary:
        conv.messages.append({"role": "assistant", "content": "[Summary of earlier conversation]\nold",
                              "_message_type": MessageType.SUMMARY})
    for i, rid in enumerate(rowids):
        conv.messages.append({
            "role": "user" if i % 2 == 0 else "assistant",
            "content": contents[i] if contents else f"m{rid}",
            "_db_id": rid, "_message_type": MessageType.MESSAGE,
        })
    return conv


@contextlib.asynccontextmanager
async def acp_agent_env(agent_defaults: dict):
    """An AcpPlugin and CorvidaeAcpAgent over an in-memory sessions DB, with
    a mocked client connection (``conn.session_update`` is an AsyncMock).

    Yields a SimpleNamespace with pm, registry, db, plugin, agent and conn;
    closes the DB on exit. Callers still call ``agent.initialize``.
    """
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    from corvidae.channel import ChannelRegistry
    from corvidae.channels.acp import AcpPlugin, CorvidaeAcpAgent
    from corvidae.hooks import create_plugin_manager
    from corvidae.persistence import PersistencePlugin, init_db

    pm = create_plugin_manager()
    registry = ChannelRegistry(agent_defaults)
    pm.register(registry, name="registry")
    persistence = PersistencePlugin(pm)
    db = await aiosqlite.connect(":memory:")
    await init_db(db)
    persistence.db = db
    pm.register(persistence, name="persistence")
    plugin = AcpPlugin()
    pm.register(plugin, name="acp")
    await plugin.on_init(pm=pm, config={"_acp_mode": True})
    agent = CorvidaeAcpAgent(
        agent_info={"name": "corvidae", "title": "Corvidae", "version": "0.0.1"},
        plugin=plugin,
    )
    conn = MagicMock()
    conn.session_update = AsyncMock()
    plugin._conn = conn
    agent.on_connect(conn)
    try:
        yield SimpleNamespace(pm=pm, registry=registry, db=db, plugin=plugin, agent=agent, conn=conn)
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# Corrupt message_log rows (#226 set, shared by every reader's tests, #232)
# ---------------------------------------------------------------------------

# (id, SQL expression for the message value, bound params). The expression
# lets a test store bytes no Python str can carry (invalid UTF-8 as TEXT).
CORRUPT_MESSAGES = [
    pytest.param("?", ("{not valid json",), id="invalid-json"),
    pytest.param("CAST(X'FF7B' AS TEXT)", (), id="invalid-utf8-text"),
    pytest.param("X'FF7B'", (), id="invalid-utf8-blob"),
    pytest.param("?", ("null",), id="json-null"),
    pytest.param("?", ("[1]",), id="json-array"),
    pytest.param("?", ("1" * 5000,), id="int-digit-limit"),
]


async def insert_raw_message(
    db, channel_id: str, value_sql: str, params: tuple = (),
    message_type: str = "message", ts: float | None = None,
    drop_fts_trigger: bool = False,
) -> int:
    """Insert a message_log row whose message is the SQL expression value_sql.

    The memory plugin's message_log_ai trigger runs json_extract on the new
    message, which rejects invalid JSON at INSERT time. A corrupt row can only
    reach such a database from outside (hand edits, an older version), so
    drop_fts_trigger=True removes that trigger first to simulate it.
    """
    if drop_fts_trigger:
        await db.execute("DROP TRIGGER IF EXISTS message_log_ai")
    cursor = await db.execute(
        "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
        f"VALUES (?, {value_sql}, ?, ?)",
        (channel_id, *params, ts if ts is not None else time.time(), message_type),
    )
    await db.commit()
    return cursor.lastrowid


def skip_warnings(caplog, row_id: int, channel_id: str) -> list[str]:
    """WARNING texts that name both this row id and this channel."""
    return [
        r.getMessage() for r in caplog.records
        if r.levelno == logging.WARNING
        and re.search(rf"\b{row_id}\b", r.getMessage())
        and channel_id in r.getMessage()
    ]
