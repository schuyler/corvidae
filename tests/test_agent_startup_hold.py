"""#71: a message that reaches Agent.on_message before the Agent has started
is held until it has, not dropped (PRD 3.1.7).

Unit tests drive Agent directly (the held state is "not started yet", gated on `_started`);
the real-path test drives Runtime.start with a plugin that delivers a
message during the on_start broadcast, before step 12 starts the Agent.
"""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from corvidae.hooks import hookimpl
from corvidae.trace import mint_trace_id
from helpers import build_plugin_and_channel, drain
from llm_response_fixtures import _make_text_response

pytestmark = pytest.mark.timeout(15)


@pytest.fixture
async def env():
    """(plugin, channel, db) with an Agent that has not started."""
    plugin, channel, db = await build_plugin_and_channel()
    yield plugin, channel, db
    task_plugin = plugin.pm.get_plugin("task")
    if task_plugin:
        await task_plugin.on_stop()
    await db.close()


def _start(plugin, replies: list[str] | None = None) -> MagicMock:
    """Stand in for Agent.on_start: give it a client and mark it started."""
    client = MagicMock()
    client.chat = AsyncMock(
        side_effect=[_make_text_response(r) for r in (replies or ["ok"] * 5)]
    )
    plugin._client = client
    plugin._mark_started()
    return client


async def _settle() -> None:
    """Let ready tasks run."""
    for _ in range(5):
        await asyncio.sleep(0)


class TestHold:
    async def test_message_before_start_is_held_then_runs_a_turn(self, env, caplog):
        """R1: no early return, no 'not initialized' error, turn runs after start."""
        plugin, channel, _ = env
        with caplog.at_level(logging.ERROR, logger="corvidae.agent"):
            task = asyncio.create_task(
                plugin.on_message(channel, "u", "hello", mint_trace_id())
            )
            await _settle()
            assert not task.done(), "on_message must hold until the agent starts"
            client = _start(plugin, ["hi back"])
            await asyncio.wait_for(task, 2)
            await drain(plugin, channel)
        client.chat.assert_awaited()
        assert channel.conversation.messages[-1]["content"] == "hi back"

    async def test_arrival_order_kept(self, env):
        """R2: two held messages are enqueued in arrival order."""
        plugin, channel, _ = env
        t1 = asyncio.create_task(plugin.on_message(channel, "u", "first", mint_trace_id()))
        await _settle()
        t2 = asyncio.create_task(plugin.on_message(channel, "u", "second", mint_trace_id()))
        await _settle()
        _start(plugin)
        await asyncio.wait_for(asyncio.gather(t1, t2), 2)
        await drain(plugin, channel)
        users = [m["content"] for m in channel.conversation.messages if m["role"] == "user"]
        assert users == ["first", "second"]

    async def test_empty_trace_id_raises_before_hold(self, env):
        """R3: the trace-id check fires immediately, not after the hold."""
        plugin, channel, _ = env
        with pytest.raises(ValueError, match="trace_id"):
            await asyncio.wait_for(plugin.on_message(channel, "u", "x", ""), 1)

    async def test_admission_fires_before_on_message_returns(self, env):
        """Transports release a typing slot in a finally around on_message
        unless admission fired first: for a held message, admission must
        still happen before on_message returns (Signal #95, Discord #102)."""
        plugin, channel, _ = env
        events: list[str] = []

        class Recorder:
            @hookimpl
            async def on_message_admitted(self, channel, trace_id, sender, text):
                events.append("admitted")

        plugin.pm.register(Recorder(), name="recorder")
        task = asyncio.create_task(plugin.on_message(channel, "u", "hi", mint_trace_id()))
        await _settle()
        assert events == [], "no admission while held"
        _start(plugin)
        await asyncio.wait_for(task, 2)
        events.append("returned")
        assert events == ["admitted", "returned"]


class TestHoldGate:
    async def test_client_set_before_start_does_not_skip_hold(self, env):
        """ConfigWatcher can fire on_config_reload during the step-11 broadcast,
        which sets _client before the Agent has started. The hold must gate on
        _started, so the message waits and is processed in order after start."""
        plugin, channel, _ = env
        t1 = asyncio.create_task(plugin.on_message(channel, "u", "first", mint_trace_id()))
        await _settle()
        # Simulate a reload re-borrowing the client before start.
        plugin._client = MagicMock()
        t2 = asyncio.create_task(plugin.on_message(channel, "u", "second", mint_trace_id()))
        await _settle()
        assert not t1.done() and not t2.done(), "must stay held until _mark_started"
        _start(plugin)
        await asyncio.wait_for(asyncio.gather(t1, t2), 2)
        await drain(plugin, channel)
        users = [m["content"] for m in channel.conversation.messages if m["role"] == "user"]
        assert users == ["first", "second"]


class TestStopWhileHeld:
    async def test_stop_releases_held_callers_with_warning(self, env, caplog):
        """R4/R5: on_stop releases each held caller; one WARNING per message
        names channel and trace id; no admission happens."""
        plugin, channel, _ = env
        admitted: list[str] = []

        class Recorder:
            @hookimpl
            async def on_message_admitted(self, channel, trace_id, sender, text):
                admitted.append(trace_id)

        plugin.pm.register(Recorder(), name="recorder")
        ids = [mint_trace_id(), mint_trace_id()]
        with caplog.at_level(logging.WARNING, logger="corvidae.agent"):
            tasks = [
                asyncio.create_task(plugin.on_message(channel, "u", "m", t)) for t in ids
            ]
            await _settle()
            assert not any(t.done() for t in tasks)
            # on_stop without a prior on_start must not raise either.
            await asyncio.wait_for(plugin.on_stop(), 2)
            await asyncio.wait_for(asyncio.gather(*tasks), 2)
        assert admitted == []
        warns = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        for trace_id in ids:
            matching = [w for w in warns if trace_id in w]
            assert len(matching) == 1, warns
            assert channel.id in matching[0]
            assert "stopping or never started" in matching[0]
            assert "not processed" in matching[0]

    async def test_message_after_stop_is_not_held_forever(self, env, caplog):
        """A message arriving after on_stop returns with the same warning."""
        plugin, channel, _ = env
        await plugin.on_stop()
        with caplog.at_level(logging.WARNING, logger="corvidae.agent"):
            await asyncio.wait_for(plugin.on_message(channel, "u", "late", mint_trace_id()), 1)
        assert "stopping or never started" in caplog.text

    async def test_cleanup_after_failed_start_releases_held(self, env):
        """R4: startup aborting (Runtime skips agent.on_stop) also releases."""
        plugin, channel, _ = env
        task = asyncio.create_task(plugin.on_message(channel, "u", "m", mint_trace_id()))
        await _settle()
        plugin.release_held_messages()
        await asyncio.wait_for(task, 2)


class TestRuntimeStartup:
    @pytest.fixture(autouse=True)
    def _reset_logging(self):
        """Runtime.start reconfigures global logging (level, handlers and
        propagate on the corvidae and root loggers); restore all of it so
        later tests' caplog is unaffected."""
        saved = [
            (lg, lg.level, lg.propagate, list(lg.handlers))
            for lg in (logging.getLogger("corvidae"), logging.getLogger())
        ]
        yield
        for lg, level, propagate, handlers in saved:
            lg.setLevel(level)
            lg.propagate = propagate
            lg.handlers[:] = handlers

    async def test_message_delivered_during_on_start_broadcast(
        self, tmp_path, monkeypatch
    ):
        """Real path: a transport delivering from its on_start (step 11) is
        processed once the Agent starts (step 12); Runtime.stop stays in budget."""
        from corvidae.llm import LLMClient
        from corvidae.runtime import Runtime
        from corvidae.task import TaskPlugin
        from test_minimal_profile import _drain_all, _write_minimal_config

        monkeypatch.setattr(
            LLMClient, "chat",
            AsyncMock(return_value=_make_text_response("held reply")),
        )
        delivered: list[asyncio.Task] = []
        orig = TaskPlugin.on_start

        @hookimpl
        async def delivering_on_start(self, config):
            # Deliver as a transport would on connect: a fire-and-forget
            # task that reaches Agent.on_message mid-broadcast.
            await orig(self, config)
            registry = self.pm.get_plugin("registry")
            channel = registry.get_or_create("irc", "main")
            delivered.append(asyncio.create_task(
                self.pm.ahook.on_message(
                    channel=channel, sender="u", text="early", trace_id=mint_trace_id(),
                )
            ))

        monkeypatch.setattr(TaskPlugin, "on_start", delivering_on_start)
        rt = Runtime(config_path=_write_minimal_config(tmp_path, tmp_path / "s.db"))
        if True:
            await rt.start()
            try:
                assert delivered, "the delivering on_start never ran"
                await asyncio.wait_for(asyncio.gather(*delivered), 5)
                agent = rt.pm.get_plugin("agent")
                await _drain_all(agent)
                channel = rt.pm.get_plugin("registry").get_or_create("irc", "main")
                assert channel.conversation.messages[-1]["content"] == "held reply"
            finally:
                await asyncio.wait_for(rt.stop(), 5)

    async def test_failed_start_cleanup_releases_held_message(
        self, tmp_path, monkeypatch, caplog
    ):
        """Real path: step 12 raises after a transport delivered during step 11;
        Runtime._cleanup_failed_start must release the held on_message, whose
        WARNING names the channel and trace id."""
        from corvidae.agent import Agent
        from corvidae.runtime import Runtime
        from corvidae.task import TaskPlugin
        from test_minimal_profile import _write_minimal_config

        delivered: list[asyncio.Task] = []
        trace_id = mint_trace_id()
        orig = TaskPlugin.on_start

        @hookimpl
        async def delivering_on_start(self, config):
            # Deliver mid-broadcast, as a transport would on connect.
            await orig(self, config)
            channel = self.pm.get_plugin("registry").get_or_create("irc", "main")
            delivered.append(asyncio.create_task(
                self.pm.ahook.on_message(
                    channel=channel, sender="u", text="early", trace_id=trace_id,
                )
            ))

        async def boom(self, config):
            raise RuntimeError("step 12 failed")

        monkeypatch.setattr(TaskPlugin, "on_start", delivering_on_start)
        monkeypatch.setattr(Agent, "_start_plugin", boom)
        rt = Runtime(config_path=_write_minimal_config(tmp_path, tmp_path / "s.db"))
        # Runtime.start sets propagate=False on the corvidae logger, so caplog
        # sees nothing; capture with a handler attached after start.
        records: list[logging.LogRecord] = []

        class _Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        with pytest.raises(RuntimeError, match="step 12 failed"):
            await rt.start()
        handler = _Capture(level=logging.WARNING)
        logging.getLogger("corvidae").addHandler(handler)
        try:
            assert delivered, "the delivering on_start never ran"
            assert not delivered[0].done(), "message must be held before cleanup"
            await asyncio.wait_for(rt._cleanup_failed_start(), 5)
            await asyncio.wait_for(asyncio.gather(*delivered), 2)
        finally:
            logging.getLogger("corvidae").removeHandler(handler)
        warns = [r.getMessage() for r in records if r.levelno == logging.WARNING]
        matching = [w for w in warns if trace_id in w]
        assert len(matching) == 1, warns
        assert "irc:main" in matching[0]
        assert "resend" in matching[0]
