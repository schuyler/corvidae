"""Tests for retrying a main turn when the LLM backend is unreachable (#281).

A connection-class failure (or the router's "model is not loaded" after the
client's own wait) leaves the user's message unanswered. The Agent sends the
error fallback once, then re-runs the turn by itself on a capped exponential
backoff driven by the kernel clock, until the backend answers or any turn on
the channel succeeds. No test sleeps real time; FakeClock moves it.
"""

from __future__ import annotations

import json
import logging
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest
from fake_clock import install_kernel_clock
from helpers import build_plugin_and_channel, drain, sent_texts
from llm_response_fixtures import _make_text_response

from corvidae.hooks import hookimpl
from corvidae.llm import LLMHTTPError
from corvidae.outbound import OutboundMessage
from corvidae.trace import mint_trace_id

# Body of the router's "model is not loaded" 400 (see test_model_not_loaded.py).
NOT_LOADED_BODY = json.dumps({
    "error": {"code": 400, "message": "model is not loaded", "type": "invalid_request_error"}
})


def _http_error(body: str, status: int = 400) -> LLMHTTPError:
    """An LLMHTTPError as LLMClient raises it for a non-2xx response."""
    cause = aiohttp.ClientResponseError(
        request_info=MagicMock(), history=(), status=status, message="error"
    )
    return LLMHTTPError(cause, body=body, model="swift-1.5")


def _conn_error() -> Exception:
    """A connection-class failure, as LLMClient raises it after its retries."""
    return aiohttp.ClientConnectionError("refused")


def _retry_timers(clock) -> list[str]:
    """Names of live turn-retry timers on ``clock``."""
    return [n for n in clock.live_timer_names() if n.startswith("turn-retry:")]


async def _setup(side_effect, *, clock: bool = True):
    """Build an Agent with a fake clock and a scripted ``chat``.

    Returns (plugin, channel, db, clock). ``side_effect`` is the list of
    exceptions and responses ``chat`` yields, in call order.
    """
    plugin, channel, db = await build_plugin_and_channel()
    fake = install_kernel_clock(plugin.pm)[0] if clock else None
    plugin._client = MagicMock()
    plugin._mark_started()
    plugin._client.chat = AsyncMock(side_effect=side_effect)
    return plugin, channel, db, fake


async def _send(plugin, channel, text: str = "hi") -> str:
    """Deliver one inbound user message and drain the queue; return its trace id."""
    trace_id = mint_trace_id()
    await plugin.on_message(channel=channel, sender="user", text=text, trace_id=trace_id)
    await drain(plugin, channel)
    return trace_id


async def _advance(plugin, channel, clock, seconds: float) -> None:
    """Advance the fake clock, then drain whatever the timers enqueued."""
    await clock.advance(seconds)
    await drain(plugin, channel)


async def _count_user_rows(db, channel, text: str) -> int:
    """Count message_log rows for ``channel`` whose user content is ``text``."""
    async with db.execute(
        "SELECT message FROM message_log WHERE channel_id = ?", (channel.id,)
    ) as cur:
        rows = [json.loads(r[0]) async for r in cur]
    return sum(1 for m in rows if m.get("role") == "user" and m.get("content") == text)


# ---------------------------------------------------------------------------
# classifier
# ---------------------------------------------------------------------------


class TestIsBackendUnreachable:
    """is_backend_unreachable separates 'cannot reach it' from everything else."""

    @staticmethod
    def _fn():
        """Resolve the classifier lazily so each case fails on its own assertion."""
        from corvidae import llm

        return getattr(llm, "is_backend_unreachable", None)

    @pytest.mark.parametrize(
        "make",
        [
            lambda: aiohttp.ClientConnectionError("x"),
            lambda: aiohttp.ServerDisconnectedError(),
            lambda: aiohttp.ClientConnectorError(MagicMock(), OSError("refused")),
            lambda: _http_error(NOT_LOADED_BODY),
        ],
        ids=["connection", "disconnected", "connector", "not-loaded"],
    )
    def test_true_for_connection_class(self, make):
        """Connection failures and the not-loaded 400 are unreachable."""
        fn = self._fn()
        assert fn is not None, "corvidae.llm.is_backend_unreachable is missing"
        assert fn(make()) is True

    @pytest.mark.parametrize(
        "make",
        [
            lambda: aiohttp.ServerTimeoutError(),
            lambda: aiohttp.SocketTimeoutError(),
            lambda: aiohttp.ConnectionTimeoutError(),
            lambda: TimeoutError(),
            lambda: _http_error('{"error": {"message": "boom"}}', status=500),
            lambda: ValueError("bad"),
        ],
        ids=["server-timeout", "socket-timeout", "connect-timeout", "timeout",
             "http-500", "value-error"],
    )
    def test_false_for_everything_else(self, make):
        """Timeouts, server HTTP errors and other exceptions are not retried."""
        fn = self._fn()
        assert fn is not None, "corvidae.llm.is_backend_unreachable is missing"
        assert fn(make()) is False


# ---------------------------------------------------------------------------
# retry behaviour
# ---------------------------------------------------------------------------


async def test_unreachable_turn_retries_until_reply():
    """R1 AC: one fallback on the first failure, then the real reply, no new message."""
    plugin, channel, db, clock = await _setup(
        [_conn_error(), _conn_error(), _make_text_response("real reply")]
    )
    # Count on_message calls: a retry must not look like a new inbound message.
    on_message = AsyncMock(wraps=plugin.on_message)
    plugin.on_message = on_message

    await _send(plugin, channel)
    assert len(sent_texts(plugin.pm, OutboundMessage)) == 1  # the fallback

    await _advance(plugin, channel, clock, 5)
    assert len(sent_texts(plugin.pm, OutboundMessage)) == 1  # failed retry: silent

    await _advance(plugin, channel, clock, 10)
    texts = sent_texts(plugin.pm, OutboundMessage)
    assert len(texts) == 2 and texts[1] == "real reply"
    assert on_message.await_count == 1  # the retry was not a new inbound message
    # The user message appears once in the window and once in the log.
    window_users = [
        m for m in channel.conversation.messages
        if m.get("role") == "user" and m.get("content") == "hi"
    ]
    assert len(window_users) == 1
    assert await _count_user_rows(db, channel, "hi") == 1
    await db.close()


async def test_model_not_loaded_is_retried():
    """R1: the router's 'model is not loaded' is retried like a connection failure."""
    plugin, channel, db, clock = await _setup(
        [_http_error(NOT_LOADED_BODY), _make_text_response("real reply")]
    )
    await _send(plugin, channel)
    assert len(sent_texts(plugin.pm, OutboundMessage)) == 1
    await _advance(plugin, channel, clock, 5)
    assert sent_texts(plugin.pm, OutboundMessage)[-1] == "real reply"
    await db.close()


async def test_retry_keeps_trace_and_does_not_refire_persisted():
    """R5: same trace id, on_message_persisted and before_agent_turn fire once."""
    plugin, channel, db, clock = await _setup(
        [_conn_error(), _make_text_response("real reply")]
    )
    persisted: list[str] = []
    before: list[str] = []

    class Spy:
        """Records how often the once-per-exchange hooks fire."""

        @hookimpl
        async def on_message_persisted(self, channel, trace_id, rowid, text, meta):
            persisted.append(trace_id)

        @hookimpl
        async def before_agent_turn(self, channel, trace_id, meta):
            before.append(trace_id)

    plugin.pm.register(Spy(), name="spy")

    trace_id = await _send(plugin, channel)
    await _advance(plugin, channel, clock, 5)

    assert persisted == [trace_id]
    assert before == [trace_id]
    # The reply carries the inbound trace id.
    kwargs = plugin.pm.ahook.on_agent_response.await_args.kwargs
    assert kwargs["trace_id"] == trace_id
    assert kwargs["response_text"] == "real reply"
    await db.close()


async def test_retry_does_not_reset_turn_counter():
    """R5: a retry is not a new exchange, so max_turns is not refreshed."""
    plugin, channel, db, clock = await _setup(
        [_conn_error(), _make_text_response("real reply")]
    )
    await _send(plugin, channel)
    channel.turn_counter = 3
    await _advance(plugin, channel, clock, 5)
    assert channel.turn_counter == 4
    await db.close()


async def test_backoff_schedule_and_warning(caplog):
    """R2: delays 5,10,20,40,60,60; one WARNING per failed retry; one fallback."""
    plugin, channel, db, clock = await _setup([_conn_error() for _ in range(6)])
    delays: list[float] = []
    caplog.set_level(logging.WARNING, logger="corvidae.agent")

    await _send(plugin, channel)
    # Each round: read the pending retry's delay, fire it, and check the one
    # WARNING the failed retry logged names the next delay.
    for expected_next in [10, 20, 40, 60, 60]:
        delay = await _next_retry_delay(clock)
        delays.append(delay)
        caplog.clear()
        await _advance(plugin, channel, clock, delay)
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1, warnings
        assert channel.id in warnings[0]
        assert "ClientConnectionError" in warnings[0]
        assert f"retrying in {expected_next} s" in warnings[0]
        assert "llm.main.base_url" in warnings[0]
    delays.append(await _next_retry_delay(clock))

    assert delays == [5, 10, 20, 40, 60, 60]
    assert len(sent_texts(plugin.pm, OutboundMessage)) == 1
    await db.close()


async def _next_retry_delay(clock) -> float:
    """Delay of the pending turn-retry timer, read from the clock's schedule."""
    return await clock.next_sleep()


async def test_successful_new_message_drops_retry():
    """R3 AC: a later message whose turn succeeds cancels the pending retry."""
    plugin, channel, db, clock = await _setup(
        [_conn_error(), _make_text_response("answer to second")]
    )
    await _send(plugin, channel, "first")
    assert _retry_timers(clock)

    await _send(plugin, channel, "second")
    await _advance(plugin, channel, clock, 60)

    assert plugin._client.chat.await_count == 2
    texts = sent_texts(plugin.pm, OutboundMessage)
    assert texts.count("answer to second") == 1 and len(texts) == 2
    assert _retry_timers(clock) == []
    await db.close()


@pytest.mark.parametrize("planted", [True, False], ids=["other-pending", "no-pending"])
async def test_superseded_retry_item_is_dropped(planted):
    """R3: a retry item that is not the pending one (or has none) is ignored."""
    import dataclasses

    import corvidae.agent as agent_mod

    plugin, channel, db, _clock = await _setup([_make_text_response("must not run")])
    base = agent_mod.QueueItem(
        role=agent_mod.QueueItemRole.USER, content="old", channel=channel,
        sender="user", trace_id=mint_trace_id(),
    )
    item_a = dataclasses.replace(base, originating=False, retry=True)
    item_b = dataclasses.replace(base, originating=False, retry=True)
    if planted:
        # A pending state for a different item A; B is stale.
        plugin._turn_retries[channel.id] = agent_mod._TurnRetry(
            item=item_a, attempt=0, timer=MagicMock()
        )
    else:
        # A success already popped the state; the fired item is stale.
        assert channel.id not in plugin._turn_retries
    await plugin._process_queue_item(item_b)
    assert plugin._client.chat.await_count == 0
    await db.close()


@pytest.mark.parametrize(
    "make",
    [
        lambda: _http_error('{"error": {"message": "boom"}}', status=500),
        lambda: TimeoutError(),
        lambda: aiohttp.ServerTimeoutError(),
        lambda: ValueError("bad"),
    ],
    ids=["http-500", "timeout", "server-timeout", "value-error"],
)
async def test_other_failures_not_retried(make):
    """R4: only unreachable-backend failures schedule a retry."""
    plugin, channel, db, clock = await _setup([make()])
    await _send(plugin, channel)
    assert len(sent_texts(plugin.pm, OutboundMessage)) == 1
    assert _retry_timers(clock) == []
    await db.close()


async def test_retry_failing_otherwise_stops_silently(caplog):
    """R4/R1: a retry that fails for another reason sends nothing and stops."""
    plugin, channel, db, clock = await _setup(
        [_conn_error(), _http_error('{"error": {"message": "boom"}}', status=500)]
    )
    caplog.set_level(logging.WARNING, logger="corvidae.agent")
    await _send(plugin, channel)
    caplog.clear()
    await _advance(plugin, channel, clock, 5)

    assert len(sent_texts(plugin.pm, OutboundMessage)) == 1
    assert _retry_timers(clock) == []
    assert any(
        channel.id in r.getMessage() and "not retrying" in r.getMessage()
        for r in caplog.records if r.levelno == logging.WARNING
    )
    await db.close()


async def test_vetoed_error_fallback_still_retries():
    """A vetoed 'error' emission must not cancel the retry."""
    plugin, channel, db, clock = await _setup(
        [_conn_error(), _make_text_response("real reply")]
    )

    class Veto:
        """Vetoes only the error fallback."""

        @hookimpl
        async def should_send_response(self, channel, text, emission, trace_id, meta):
            return False if emission == "error" else None

    plugin.pm.register(Veto(), name="veto")

    await _send(plugin, channel)
    assert sent_texts(plugin.pm, OutboundMessage) == []
    assert _retry_timers(clock)
    await _advance(plugin, channel, clock, 5)
    assert sent_texts(plugin.pm, OutboundMessage) == ["real reply"]
    await db.close()


async def test_no_clock_warns_and_skips_retry(caplog):
    """Without a kernel clock the fallback is sent and the skip is logged."""
    plugin, channel, db, _ = await _setup([_conn_error()], clock=False)
    caplog.set_level(logging.WARNING, logger="corvidae.agent")
    await _send(plugin, channel)

    assert len(sent_texts(plugin.pm, OutboundMessage)) == 1
    assert any(
        channel.id in r.getMessage() and "will not be retried" in r.getMessage()
        for r in caplog.records if r.levelno == logging.WARNING
    )
    await db.close()


async def test_on_stop_cancels_pending_retry():
    """on_stop drops pending retries: a due timer enqueues nothing and calls nothing."""
    plugin, channel, db, clock = await _setup(
        [_conn_error(), _make_text_response("must not run")]
    )
    await _send(plugin, channel)
    assert _retry_timers(clock)

    # Keep the mock: on_stop clears the client reference.
    chat = plugin._client.chat
    await plugin.on_stop()
    await clock.advance(5)

    # No second LLM call, no retry timer, nothing left on the channel's queue.
    assert chat.await_count == 1
    assert _retry_timers(clock) == []
    queue = plugin.queues.get(channel.id)
    assert queue is None or queue._queue.empty()
    await db.close()
