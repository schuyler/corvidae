"""Lease scopes (#363): every model call names its queue.

Covers the kernel half of the acceptance criteria in
.claude/plans/363-lease-scopes.md: the ``lease_scope`` API and its
attribution field (AC1, AC2), propagation into TaskQueue workers (AC3), the
unleased warning (AC4, AC5), the usage_log column and the on_metrics tag
(AC6, AC7), and the kernel call sites of the AC8 sweep (turn, Agent.fork,
compaction, the fork and subagent tools). The cognition call sites live in
tests/cognition/test_lease_scopes.py (kernel tests must not import
corvidae_cognition) and the vision one in packages/corvidae-vision/tests.

Names that do not exist yet (``lease_scope``, ``lease_key``,
``_unleased_warned_stages``) are reached inside each test, so a test fails on
its own missing piece rather than all of them failing at collection.
"""

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable

import aiosqlite
import pytest
from helpers import (
    build_plugin_and_channel,
    drain,
    drain_task_queue,
    start_llm_plugin,
    window_of,
)
from llm_response_fixtures import (
    _make_text_response,
    _make_tool_call,
    _make_tool_call_response,
)

from corvidae.agent import QueueItem, QueueItemRole
from corvidae.attribution import get_attribution, reset_attribution, set_attribution
from corvidae.channel import Channel, ChannelConfig
from corvidae.context import ContextWindow
from corvidae.fork import ForksConfig
from corvidae.hooks import create_plugin_manager, hookimpl
from corvidae.metrics import MetricsPlugin, UsageLogPlugin
from corvidae.persistence import PersistencePlugin
from corvidae.task import Task

# The stages the stack's LLMPlugin resolves; every one maps to the main profile.
STAGES = {
    "compaction": "chat",
    "subagent": "chat",
    "fork": "chat",
    "consolidation": "chat",
    "appraisal": "chat",
}
LLM_SECTION = {
    "providers": {"main": {"base_url": "http://main.test/v1"}},
    "profiles": {"main": {"model": "main-m"}},
}

# Words in a user message that make the scripted model call a tool.
TURN_SPAWNS_FORK = "SPAWN_FORK_FROM_TURN"
TURN_SPAWNS_SUBAGENT = "SPAWN_SUBAGENT_FROM_TURN"
SUBAGENT_SPAWNS_FORK = "SUBAGENT_SPAWNS_FORK"


# ---------------------------------------------------------------------------
# Stub HTTP session and the stack under test
# ---------------------------------------------------------------------------


class _Response:
    """A 200 response carrying ``payload``."""

    status = 200
    headers: dict = {}

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        """Never raises: every stubbed response is a success."""

    async def json(self) -> dict:
        """The scripted payload."""
        return self._payload

    async def text(self) -> str:
        """No body text."""
        return ""


class _Post:
    """The async context manager ``session.post`` returns."""

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    async def __aenter__(self) -> _Response:
        return _Response(self._payload)

    async def __aexit__(self, *exc: object) -> bool:
        return False


class ScriptedSession:
    """An aiohttp session stand-in: records every POST body in ``posts`` and
    answers chat requests with ``respond(body)`` (text "ok" by default)."""

    def __init__(self) -> None:
        self.headers: dict = {}
        self.posts: list[dict] = []
        self.respond: Callable[[dict], dict] = lambda body: _make_text_response("ok")

    def post(self, url: str, json: dict | None = None, timeout: Any = None, **kw: Any) -> _Post:
        """Record the request and answer it."""
        self.posts.append({"url": url, "json": json})
        if url.endswith("/embeddings"):
            return _Post({"data": [{"index": 0, "embedding": [1.0, 0.0]}]})
        return _Post(self.respond(json or {}))

    async def close(self) -> None:
        """Nothing to release."""


@dataclass
class Stack:
    """A real Agent, LLMPlugin (stubbed HTTP) and UsageLogPlugin on one pm."""

    plugin: Any
    channel: Channel
    db: aiosqlite.Connection
    llm: Any
    session: ScriptedSession


async def build_stack(monkeypatch: pytest.MonkeyPatch) -> Stack:
    """Assemble the stack the way tests/test_phase0_integration.py does, over
    the real LLMClient with its HTTP session replaced by a ScriptedSession."""
    plugin, channel, db = await build_plugin_and_channel()
    pm = plugin.pm
    await plugin.on_init(pm, {"agent": {"forks": {"enabled": True}}})
    llm, _posts = await start_llm_plugin(monkeypatch, LLM_SECTION, STAGES, pm=pm)
    # No network: every client posts through the scripted session.
    session = ScriptedSession()
    for client in llm._clients.values():
        client.session = session
    usage_log = UsageLogPlugin()
    pm.register(usage_log, name="usage_log")
    await usage_log.on_init(pm, {})
    await usage_log.on_start({})
    plugin._client = llm.get_client()
    plugin._mark_started()
    return Stack(plugin, channel, db, llm, session)


@pytest.fixture
async def stack(monkeypatch: pytest.MonkeyPatch):
    """A built Stack, torn down after the test."""
    built = await build_stack(monkeypatch)
    yield built
    await built.plugin.pm.get_plugin("task").on_stop()
    await built.llm.on_stop()
    await built.db.close()


@pytest.fixture(autouse=True)
def fresh_unleased_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Forget which stages already warned, so each test sees a first call.

    ``raising=False``: before the change the set does not exist."""
    import corvidae.llm as llm_module

    monkeypatch.setattr(llm_module, "_unleased_warned_stages", set(), raising=False)


async def usage_rows(db: aiosqlite.Connection) -> list[tuple]:
    """(stage, lease_key, mode) of every usage_log row, in write order."""
    async with db.execute(
        "SELECT stage, lease_key, mode FROM usage_log ORDER BY id"
    ) as cursor:
        return await cursor.fetchall()


def keys_of(rows: list[tuple], stage: str) -> list[str | None]:
    """The lease keys of the rows written for ``stage``."""
    return [key for row_stage, key, _mode in rows if row_stage == stage]


def unleased_records(caplog: pytest.LogCaptureFixture, level: int | None = None) -> list[logging.LogRecord]:
    """Records the unleased check logged (their text says "unleased", per the designed format), at
    ``level`` when given."""
    return [
        r for r in caplog.records
        if "unleased" in r.getMessage().lower() and (level is None or r.levelno == level)
    ]


async def settle(stack: Stack) -> None:
    """Let a turn's tool tasks, their notifications and the follow-up turns finish."""
    for _ in range(4):
        await drain_task_queue(stack.plugin)
        await drain(stack.plugin, stack.channel)
        await asyncio.sleep(0.01)


def user_item(stack: Stack, text: str) -> QueueItem:
    """A user message on the stack's channel."""
    return QueueItem(
        role=QueueItemRole.USER, content=text, channel=stack.channel, sender="alice"
    )


# ---------------------------------------------------------------------------
# AC1, AC2: the scope API
# ---------------------------------------------------------------------------


def test_lease_scope_override_restores_parent_and_leaves_other_fields() -> None:
    """AC1: an explicit key shadows the enclosing one and restores it on exit;
    after the outermost scope the field is absent; other fields never move."""
    from corvidae.attribution import lease_scope

    token = set_attribution(stage="turn", channel_id="irc:#a")
    try:
        before = {"stage": "turn", "channel_id": "irc:#a"}
        with lease_scope("a"):
            assert get_attribution()["lease_key"] == "a"
            with lease_scope("b"):
                assert get_attribution()["lease_key"] == "b"
                assert {k: get_attribution()[k] for k in before} == before
            assert get_attribution()["lease_key"] == "a"
        assert "lease_key" not in get_attribution()
        assert get_attribution() == before
    finally:
        reset_attribution(token)


def test_lease_scope_without_key_inherits_enclosing_key() -> None:
    """AC2: a nested scope with no key sees the enclosing key, and that key is
    still current after it exits."""
    from corvidae.attribution import lease_scope

    with lease_scope("a"):
        with lease_scope():
            assert get_attribution()["lease_key"] == "a"
        assert get_attribution()["lease_key"] == "a"


# ---------------------------------------------------------------------------
# AC3: the key reaches TaskQueue workers
# ---------------------------------------------------------------------------


async def test_task_created_in_scope_carries_key_into_worker(stack: Stack) -> None:
    """AC3: a Task created inside lease_scope("a") makes a chat call, in a
    worker started outside any scope, whose usage_log row has lease_key "a"."""
    from corvidae.attribution import lease_scope

    task_queue = stack.plugin.pm.get_plugin("task").task_queue
    done = asyncio.Event()

    async def work() -> str:
        """The chat call, made in the worker long after the scope ended."""
        await stack.llm.get_client().chat([{"role": "user", "content": "bg"}])
        done.set()
        return "ok"

    # Create the Task inside the scope; enqueue it after the scope has exited.
    with lease_scope("a"):
        task = Task(work=work, channel=stack.channel)
    await task_queue.enqueue(task)
    await asyncio.wait_for(done.wait(), timeout=5.0)
    await task_queue.queue.join()

    rows = await usage_rows(stack.db)
    assert [key for _stage, key, _mode in rows] == ["a"]


# ---------------------------------------------------------------------------
# AC4, AC5: the unleased check
# ---------------------------------------------------------------------------


async def chat_in_stage(stack: Stack, stage: str) -> None:
    """One chat call attributed to ``stage`` with no lease key."""
    token = set_attribution(stage=stage)
    try:
        await stack.llm.get_client().chat([{"role": "user", "content": "hi"}])
    finally:
        reset_attribution(token)


async def test_unleased_call_warns_once_per_stage_then_debug(
    stack: Stack, caplog: pytest.LogCaptureFixture
) -> None:
    """AC4: the first unleased call for a stage logs one WARNING naming the
    stage and the lease_scope remedy; the second logs DEBUG; a new stage warns
    again; every call still reaches the server."""
    caplog.set_level(logging.DEBUG, logger="corvidae.llm")

    await chat_in_stage(stack, "x")
    await chat_in_stage(stack, "x")

    warnings = unleased_records(caplog, logging.WARNING)
    debugs = unleased_records(caplog, logging.DEBUG)
    assert len(warnings) == 1, [r.getMessage() for r in caplog.records]
    assert re.search(r"\bx\b", warnings[0].getMessage())
    assert "lease_scope" in warnings[0].getMessage()
    assert len(debugs) == 1

    await chat_in_stage(stack, "y")

    warnings = unleased_records(caplog, logging.WARNING)
    assert len(warnings) == 2
    assert re.search(r"\by\b", warnings[1].getMessage())
    # Unleased is not refused: all three requests were sent.
    assert len(stack.session.posts) == 3


async def test_embed_without_key_logs_no_unleased_record(
    stack: Stack, caplog: pytest.LogCaptureFixture
) -> None:
    """AC5: an embed() call with no key emits no unleased record at any level."""
    caplog.set_level(logging.DEBUG)

    await stack.llm.get_client().embed(["hello"], "document")

    assert unleased_records(caplog) == []


# ---------------------------------------------------------------------------
# AC6: usage_log and on_metrics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key,expected", [("k", "k"), (None, None)])
async def test_usage_row_records_lease_key(stack: Stack, key: str | None, expected: str | None) -> None:
    """AC6: a call inside lease_scope("k") writes lease_key 'k'; a call with
    no key writes NULL."""
    from corvidae.attribution import lease_scope

    if key is None:
        await chat_in_stage(stack, "turn")
    else:
        with lease_scope(key):
            await stack.llm.get_client().chat([{"role": "user", "content": "hi"}])

    rows = await usage_rows(stack.db)
    assert [row_key for _stage, row_key, _mode in rows] == [expected]


class _MetricsRecorder:
    """Consumer plugin recording every on_metrics event's tags."""

    def __init__(self) -> None:
        self.tags: list[dict] = []

    @hookimpl
    async def on_metrics(self, name: str, value: float, tags: dict) -> None:
        """Keep the tags of each event."""
        self.tags.append(dict(tags))


@pytest.mark.parametrize("key,expected", [("k", "k"), (None, "")])
async def test_metrics_events_tag_lease_key(stack: Stack, key: str | None, expected: str) -> None:
    """AC6: every on_metrics event for the call carries tags["lease_key"]:
    the key inside a scope, "" with none."""
    from corvidae.attribution import lease_scope

    pm = stack.plugin.pm
    recorder = _MetricsRecorder()
    pm.register(MetricsPlugin(pm), name="metrics")
    pm.register(recorder, name="metrics_recorder")

    if key is None:
        await chat_in_stage(stack, "turn")
    else:
        with lease_scope(key):
            await stack.llm.get_client().chat([{"role": "user", "content": "hi"}])

    assert recorder.tags, "MetricsPlugin emitted no events"
    assert [tags.get("lease_key") for tags in recorder.tags] == [expected] * len(recorder.tags)


# ---------------------------------------------------------------------------
# AC7: in-place migration
# ---------------------------------------------------------------------------

# usage_log as created before #363: no lease_key column.
PRE_363_USAGE_LOG_DDL = """
CREATE TABLE usage_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    request_id TEXT NOT NULL,
    profile TEXT NOT NULL,
    model TEXT NOT NULL,
    stage TEXT,
    channel_id TEXT,
    trace_id TEXT,
    trigger TEXT,
    mode TEXT,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    cached_tokens INTEGER,
    cache_n INTEGER,
    prompt_n INTEGER,
    latency_ms REAL,
    error TEXT,
    started_at REAL,
    ended_at REAL,
    total_ms REAL,
    attempts INTEGER,
    message_count INTEGER,
    tool_count INTEGER,
    max_context_tokens INTEGER,
    prompt_ms REAL,
    predicted_ms REAL
)
"""


async def test_pre_363_usage_log_gains_lease_key_on_start(tmp_path) -> None:
    """AC7: a usage_log without lease_key gains the column on UsageLogPlugin
    start, old rows read NULL, and a new write lands."""
    async with aiosqlite.connect(str(tmp_path / "old.db")) as db:
        await db.execute(PRE_363_USAGE_LOG_DDL)
        await db.execute(
            "INSERT INTO usage_log (ts, request_id, profile, model) "
            "VALUES (1.0, 'old', 'main', 'm')"
        )
        await db.commit()

        pm = create_plugin_manager()
        persistence = PersistencePlugin()
        persistence.db = db
        pm.register(persistence, name="persistence")
        usage_log = UsageLogPlugin(pm)
        await usage_log.on_start({})

        async with db.execute("PRAGMA table_info(usage_log)") as cursor:
            columns = {row[1] for row in await cursor.fetchall()}
        assert "lease_key" in columns

        # A new row carrying a key lands beside the old one.
        await usage_log.on_llm_response(
            profile="main", model="m", request_id="new", usage={}, latency_ms=1.0,
            attribution={"stage": "turn", "lease_key": "irc:#a"}, error=None,
            timings=None, started_at=2.0, total_ms=1.0, attempts=1,
            message_count=1, tool_count=0,
        )
        async with db.execute(
            "SELECT request_id, lease_key FROM usage_log ORDER BY id"
        ) as cursor:
            rows = await cursor.fetchall()
        assert rows == [("old", None), ("new", "irc:#a")]


# ---------------------------------------------------------------------------
# AC8: the call-site sweep, kernel paths
# ---------------------------------------------------------------------------


async def test_main_turn_is_keyed_by_channel(stack: Stack, caplog: pytest.LogCaptureFixture) -> None:
    """AC8: the main turn's call carries the channel id."""
    caplog.set_level(logging.WARNING)

    await stack.plugin._process_queue_item(user_item(stack, "hello"))

    assert keys_of(await usage_rows(stack.db), "turn") == [stack.channel.id]
    assert unleased_records(caplog, logging.WARNING) == []


async def test_fork_from_foreign_key_context_is_keyed_by_channel(
    stack: Stack, caplog: pytest.LogCaptureFixture
) -> None:
    """AC8: Agent.fork called while another key (subagent:x) is current still
    carries the channel id."""
    caplog.set_level(logging.WARNING)
    stack.channel.conversation = window_of(stack.channel.id, [1, 2])

    token = set_attribution(stage="consolidation", lease_key="subagent:x")
    try:
        await stack.plugin.fork(stack.channel, "Summarize.", stage="consolidation")
    finally:
        reset_attribution(token)

    rows = await usage_rows(stack.db)
    assert rows == [("consolidation", stack.channel.id, "fork")]
    assert unleased_records(caplog, logging.WARNING) == []


def long_window(channel_id: str) -> ContextWindow:
    """A window big enough to trip compaction at a small token budget."""
    conv = ContextWindow(channel_id)
    conv.system_prompt = "You are a test assistant."
    for i in range(12):
        conv.append({"role": "user", "content": f"message {i} " + "x " * 40})
    return conv


async def test_compaction_fork_is_keyed_by_channel(stack: Stack, caplog: pytest.LogCaptureFixture) -> None:
    """AC8: the compaction fork carries the channel id."""
    from corvidae.compaction import CompactionPlugin

    caplog.set_level(logging.WARNING)
    compaction = CompactionPlugin()
    stack.plugin.pm.register(compaction, name="compaction")
    await compaction.on_init(stack.plugin.pm, {})
    conv = long_window(stack.channel.id)
    stack.channel.conversation = conv

    assert await compaction.resolve_compaction(
        channel=stack.channel, conversation=conv, max_tokens=120
    )

    rows = await usage_rows(stack.db)
    assert rows == [("compaction", stack.channel.id, "fork")]
    assert unleased_records(caplog, logging.WARNING) == []


async def test_compaction_view_fallback_is_keyed_by_channel(
    stack: Stack, caplog: pytest.LogCaptureFixture
) -> None:
    """AC8: with forks off, compaction's view call inside a turn carries the
    channel id (and so does the turn's own call)."""
    from corvidae.compaction import CompactionPlugin

    caplog.set_level(logging.WARNING)
    # Forks off: compaction summarizes with the plain view call.
    stack.plugin._forks = ForksConfig(enabled=False)
    compaction = CompactionPlugin()
    stack.plugin.pm.register(compaction, name="compaction")
    await compaction.on_init(stack.plugin.pm, {})
    stack.channel.conversation = long_window(stack.channel.id)
    stack.channel.config = ChannelConfig(max_context_tokens=140)

    await stack.plugin._process_queue_item(user_item(stack, "hello again"))

    rows = await usage_rows(stack.db)
    assert rows == [
        ("compaction", stack.channel.id, "standalone"),
        ("turn", stack.channel.id, None),
    ]
    assert unleased_records(caplog, logging.WARNING) == []


async def install_spawn_tools(stack: Stack) -> None:
    """Register the real subagent and fork tools, and make them the Agent's
    tool set, so a scripted model can call them."""
    from corvidae.tool import ToolRegistry
    from corvidae.tool_collection import ToolCollectionPlugin
    from corvidae.tools.fork_tool import ForkToolPlugin
    from corvidae.tools.subagent import SubagentPlugin

    pm = stack.plugin.pm
    registry = ToolRegistry()
    tools_plugin = ToolCollectionPlugin(pm)
    tools_plugin.registry = registry
    pm.register(tools_plugin, name="tools")
    subagent = SubagentPlugin(pm)
    fork = ForkToolPlugin(pm)
    pm.register(subagent, name="subagent")
    pm.register(fork, name="fork_tool")
    await subagent.on_start(config={})
    await fork.on_start(config={})
    # Collect the tools the plugins declare, as the daemon does.
    declared: list = []
    subagent.register_tools(tool_registry=declared)
    fork.register_tools(tool_registry=declared)
    for tool in declared:
        registry.add(tool)
    stack.plugin._tools = registry.as_dict()
    stack.plugin._tool_schemas = registry.schemas()


def script_tool_calls(stack: Stack) -> None:
    """Script the model: the trigger words in the last user message pick the
    tool it calls; everything else gets a plain text reply."""

    def respond(body: dict) -> dict:
        """Answer one chat request."""
        last = (body.get("messages") or [{}])[-1]
        text = last.get("content") if isinstance(last.get("content"), str) else ""
        if last.get("role") != "user":
            return _make_text_response("ok")
        if TURN_SPAWNS_FORK in text:
            call = _make_tool_call("c-fork", "fork", {"task": "review it"})
        elif TURN_SPAWNS_SUBAGENT in text:
            call = _make_tool_call(
                "c-sub", "subagent", {"instructions": SUBAGENT_SPAWNS_FORK, "description": "nested"}
            )
        elif SUBAGENT_SPAWNS_FORK in text:
            call = _make_tool_call("c-nested-fork", "fork", {"task": "review it"})
        else:
            return _make_text_response("ok")
        return _make_tool_call_response([call])

    stack.session.respond = respond


def subagent_task_id(stack: Stack) -> str:
    """The id in the subagent tool's "Subagent task <id> enqueued" reply, read
    from the tool result the turn recorded."""
    replies = [
        m["content"] for m in stack.channel.conversation.messages
        if m.get("role") == "tool" and isinstance(m.get("content"), str)
    ]
    ids = [match.group(1) for r in replies if (match := re.search(r"Subagent task (\w+) enqueued", r))]
    assert len(ids) == 1, f"expected one subagent reply, saw {replies!r}"
    return ids[0]


async def test_fork_tool_from_turn_is_keyed_by_channel(
    stack: Stack, caplog: pytest.LogCaptureFixture
) -> None:
    """AC8: the fork tool's loop, run from a turn, carries the channel id."""
    caplog.set_level(logging.WARNING)
    await install_spawn_tools(stack)
    script_tool_calls(stack)

    await stack.plugin._process_queue_item(user_item(stack, TURN_SPAWNS_FORK))
    await settle(stack)

    rows = await usage_rows(stack.db)
    assert keys_of(rows, "fork") == [stack.channel.id]
    assert unleased_records(caplog, logging.WARNING) == []


async def test_fork_tool_from_subagent_is_keyed_by_channel_not_subagent(
    stack: Stack, caplog: pytest.LogCaptureFixture
) -> None:
    """AC8: the fork tool, run from inside a subagent, carries the channel id,
    not subagent:<id>; the subagent's own calls carry subagent:<id>."""
    caplog.set_level(logging.WARNING)
    await install_spawn_tools(stack)
    script_tool_calls(stack)

    await stack.plugin._process_queue_item(user_item(stack, TURN_SPAWNS_SUBAGENT))
    await settle(stack)

    rows = await usage_rows(stack.db)
    assert keys_of(rows, "fork") == [stack.channel.id]
    assert set(keys_of(rows, "subagent")) == {f"subagent:{subagent_task_id(stack)}"}
    assert unleased_records(caplog, logging.WARNING) == []


async def test_subagent_tool_is_keyed_by_its_task_id(
    stack: Stack, caplog: pytest.LogCaptureFixture
) -> None:
    """AC8: the subagent tool's loop carries subagent:<task_id>, the id in the
    tool's "Subagent task <id> enqueued" reply."""
    caplog.set_level(logging.WARNING)
    await install_spawn_tools(stack)
    script_tool_calls(stack)
    # No nested fork this time: the subagent just answers.
    base = stack.session.respond

    def respond(body: dict) -> dict:
        """The scripted model, minus the subagent's fork call."""
        last = (body.get("messages") or [{}])[-1]
        if isinstance(last.get("content"), str) and SUBAGENT_SPAWNS_FORK in last["content"]:
            return _make_text_response("ok")
        return base(body)

    stack.session.respond = respond

    await stack.plugin._process_queue_item(user_item(stack, TURN_SPAWNS_SUBAGENT))
    await settle(stack)

    rows = await usage_rows(stack.db)
    assert keys_of(rows, "subagent") == [f"subagent:{subagent_task_id(stack)}"]
    assert unleased_records(caplog, logging.WARNING) == []
