"""Tests for issue #94: every tool call is bounded by ``tools.call_timeout``.

Red phase.  Covers R1-R5 at the ``dispatch_tool_call`` seam (R6 parser tests
live in ``tests/test_tool_collection.py``; the wiring tests live beside the
code they touch).  ``HANG`` is longer than the 15 s pytest timeout, so a
missing bound can only fail.  Each dispatch is also wrapped in a 5 s
``asyncio.wait_for`` so an unbounded call fails fast and clearly in red.
"""

import asyncio
import json
import logging
import time
from unittest.mock import AsyncMock

import pytest

import corvidae.tool as tool_mod
from corvidae.channel import Channel
from corvidae.tool import dispatch_tool_call

# Longer than the global pytest timeout: an unbounded tool cannot pass slowly.
HANG = 30.0
# Fail-fast guard around each dispatch under test.
GUARD = 5.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_own_timeout_warning_set(monkeypatch):
    """Give each test a fresh once-per-tool warning set.

    The throttle is module-level state; resetting it keeps the
    unreadable-own_timeout test independent of run order.  No-ops while the
    attribute does not exist yet (red phase).
    """
    if hasattr(tool_mod, "_warned_own_timeout"):
        monkeypatch.setattr(tool_mod, "_warned_own_timeout", set())


def _tt(default=600.0, per_tool=None):
    """Build a ToolTimeouts (imported lazily so each test fails on its own)."""
    from corvidae.tool import ToolTimeouts

    return ToolTimeouts(default, per_tool or {})


def _call(call_id: str, name: str, args: dict | None = None) -> dict:
    """An LLM tool-call dict."""
    return {
        "id": call_id,
        "function": {"name": name, "arguments": json.dumps(args or {})},
    }


async def _timed(coro):
    """Await ``coro`` under the fail-fast guard; return (result, elapsed s)."""
    start = time.monotonic()
    result = await asyncio.wait_for(coro, GUARD)
    return result, time.monotonic() - start


def _make_hang(flag: asyncio.Event | None = None):
    """A tool that sleeps HANG seconds; sets ``flag`` if cancelled."""

    async def hang(**kw):
        try:
            await asyncio.sleep(HANG)
        except asyncio.CancelledError:
            if flag is not None:
                flag.set()
            raise
        return "unreachable"

    return hang


def _timeout_records(caplog) -> list[logging.LogRecord]:
    """WARNING+ records whose message mentions a timeout."""
    return [
        r for r in caplog.records
        if r.levelno >= logging.WARNING and "timed out" in r.getMessage()
    ]


TIMEOUT_01 = "Error: tool 'hang' timed out after 0.1s"


# ---------------------------------------------------------------------------
# R1: the bound applies on every dispatch path
# ---------------------------------------------------------------------------


async def test_dispatch_bounds_inline_call():
    """A hanging inline tool returns the R2 error result within the bound."""
    r, elapsed = await _timed(
        dispatch_tool_call(_call("c1", "hang"), {"hang": _make_hang()}, call_timeouts=_tt(0.1))
    )
    assert elapsed < 1.0
    assert r.content == TIMEOUT_01
    assert r.error is True
    assert r.latency_ms is not None


class _HangBackend:
    """A ToolBackend whose run() hangs; records cancellation."""

    def __init__(self) -> None:
        self.cancelled = asyncio.Event()

    async def run(self, name, args, channel):
        try:
            await asyncio.sleep(HANG)
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


@pytest.mark.parametrize("kind", ["override", "acp-local"])
async def test_dispatch_bounds_tool_backend(kind):
    """The bound also covers the ToolBackend path (override and ACP default)."""
    channel = Channel(transport="acp", scope="s1")
    backend = None
    if kind == "override":
        # An explicit backend override whose run() hangs.
        backend = _HangBackend()
        channel.runtime_overrides["tool_backend"] = backend
    # For acp-local the default LocalToolBackend runs the hanging tool.
    r, elapsed = await _timed(
        dispatch_tool_call(
            _call("c1", "hang"), {"hang": _make_hang()},
            channel=channel, call_timeouts=_tt(0.1),
        )
    )
    assert elapsed < 1.0
    assert r.content == TIMEOUT_01
    assert r.error is True
    if backend is not None:
        assert backend.cancelled.is_set()


def test_default_bound_is_600():
    """The default bound is 600 s, both as a value type and as parsed config."""
    from corvidae.tool import ToolTimeouts
    from corvidae.tool_collection import parse_tool_timeouts

    assert ToolTimeouts().for_tool("x") == 600.0
    assert parse_tool_timeouts({}).default == 600.0


def test_entry_points_default_to_the_600s_bound():
    """R1: a caller that passes no call_timeouts is still bounded at 600 s,
    on the public dispatch function, the subagent loop and a fresh Agent."""
    import inspect

    from corvidae.agent import Agent
    from corvidae.tool import dispatch_tool_call
    from corvidae.tools.subagent import run_agent_loop

    # Each default is a ToolTimeouts bounding an arbitrary tool at 600 s,
    # never None (which would leave the call unbounded).
    for fn in (dispatch_tool_call, run_agent_loop):
        default = inspect.signature(fn).parameters["call_timeouts"].default
        assert default is not None
        assert default.for_tool("x") == 600.0
    assert Agent()._tool_call_timeouts.for_tool("x") == 600.0


@pytest.mark.parametrize("value", [None, 0])
async def test_null_and_zero_disable(value):
    """call_timeout null/0 parses to no bound; a slow tool then finishes."""
    from corvidae.tool_collection import parse_tool_timeouts

    t = parse_tool_timeouts({"call_timeout": value})
    assert t.default is None

    async def slow(**kw):
        await asyncio.sleep(0.2)
        return "ok"

    # If 0 were not normalised, asyncio.timeout(0) would fire at once.
    r, _ = await _timed(dispatch_tool_call(_call("c1", "slow"), {"slow": slow}, call_timeouts=t))
    assert r.content == "ok"


# ---------------------------------------------------------------------------
# R2: transform_tool_result still fires on a timeout
# ---------------------------------------------------------------------------


async def test_transform_tool_result_fires_on_timeout():
    """The hook sees the timeout string; a transform can replace it."""
    from corvidae.hooks import create_plugin_manager, hookimpl

    seen = []
    replacement = {"value": None}

    class RecordingPlugin:
        @hookimpl
        async def transform_tool_result(self, tool_name, result, channel):
            seen.append({"tool_name": tool_name, "result": result})
            return replacement["value"]

    pm = create_plugin_manager()
    pm.register(RecordingPlugin(), name="recorder")

    r, _ = await _timed(
        dispatch_tool_call(
            _call("c1", "hang"), {"hang": _make_hang()}, pm=pm, call_timeouts=_tt(0.1)
        )
    )
    assert seen == [{"tool_name": "hang", "result": TIMEOUT_01}]
    assert r.content == TIMEOUT_01

    # A transform that returns a value wins over the timeout string.
    replacement["value"] = "X"
    r, _ = await _timed(
        dispatch_tool_call(
            _call("c2", "hang"), {"hang": _make_hang()}, pm=pm, call_timeouts=_tt(0.1)
        )
    )
    assert r.content == "X"


# ---------------------------------------------------------------------------
# R3: per-tool overrides
# ---------------------------------------------------------------------------


async def test_per_tool_override():
    """An override bounds one tool; other tools keep the default."""
    t = _tt(600, {"slow_tool": 0.2})

    r, elapsed = await _timed(
        dispatch_tool_call(_call("c1", "slow_tool"), {"slow_tool": _make_hang()}, call_timeouts=t)
    )
    assert r.content == "Error: tool 'slow_tool' timed out after 0.2s"
    assert 0.15 < elapsed < 2.0

    async def other(**kw):
        await asyncio.sleep(0.05)
        return "ok"

    r, _ = await _timed(dispatch_tool_call(_call("c2", "other"), {"other": other}, call_timeouts=t))
    assert r.content == "ok"
    assert t.for_tool("other") == 600


async def test_override_null_disables_for_one_tool():
    """A None override unbounds that tool only."""
    t = _tt(0.05, {"long": None})

    async def long(**kw):
        await asyncio.sleep(0.2)
        return "ok"

    r, _ = await _timed(dispatch_tool_call(_call("c1", "long"), {"long": long}, call_timeouts=t))
    assert r.content == "ok"

    # The same object still bounds an unlisted tool at 0.05 s.
    r, elapsed = await _timed(
        dispatch_tool_call(_call("c2", "hang"), {"hang": _make_hang()}, call_timeouts=t)
    )
    assert r.content == "Error: tool 'hang' timed out after 0.05s"
    assert elapsed < 2.0


def test_parse_override_zero_is_none():
    """A 0 override normalises to None (unbounded)."""
    from corvidae.tool_collection import parse_tool_timeouts

    assert parse_tool_timeouts({"timeouts": {"a": 0}}).per_tool == {"a": None}


# ---------------------------------------------------------------------------
# R4: skip the outer bound when the tool's own timeout is shorter
# ---------------------------------------------------------------------------


async def test_own_timeout_shorter_skips_outer():
    """own_timeout <= bound: no outer timer; the tool's own error is delivered."""

    async def own(**kw):
        await asyncio.sleep(0.3)
        return "Error: own timeout"

    own.own_timeout = lambda: 0.1  # equal to the bound counts as <=

    r, elapsed = await _timed(
        dispatch_tool_call(_call("c1", "own"), {"own": own}, call_timeouts=_tt(0.1))
    )
    assert r.content == "Error: own timeout"
    assert r.error is False
    assert elapsed >= 0.25


async def test_own_timeout_longer_keeps_outer():
    """own_timeout (a number) above the bound: the outer bound still applies."""
    hang = _make_hang()
    hang.own_timeout = 5

    r, elapsed = await _timed(
        dispatch_tool_call(_call("c1", "hang"), {"hang": hang}, call_timeouts=_tt(0.1))
    )
    assert r.content == TIMEOUT_01
    assert elapsed < 2.0


async def test_own_timeout_read_at_dispatch():
    """The declared timeout is read on each dispatch, not captured."""
    state = {"t": 0.05}

    async def own(**kw):
        await asyncio.sleep(0.2)
        return "own"

    own.own_timeout = lambda: state["t"]

    # Own 0.05 <= bound 0.1: no outer timer, the tool finishes.
    r, _ = await _timed(dispatch_tool_call(_call("c1", "own"), {"own": own}, call_timeouts=_tt(0.1)))
    assert r.content == "own"

    # Raised to 5: now the outer 0.1 s bound applies to the same tool.
    state["t"] = 5
    r, _ = await _timed(dispatch_tool_call(_call("c2", "own"), {"own": own}, call_timeouts=_tt(0.1)))
    assert r.content == "Error: tool 'own' timed out after 0.1s"


async def test_own_timeouterror_is_generic_failure(caplog):
    """A tool's own TimeoutError (before the deadline) is a plain failure."""

    async def x(**kw):
        raise TimeoutError("inner")

    with caplog.at_level(logging.DEBUG, logger="corvidae.tool"):
        r, _ = await _timed(dispatch_tool_call(_call("c1", "x"), {"x": x}, call_timeouts=_tt(0.1)))
    assert r.content == "Error: tool 'x' failed"
    assert r.error is True
    assert not any("timed out after" in rec.getMessage() for rec in caplog.records)


async def test_unreadable_own_timeout_falls_back_and_warns_once(caplog):
    """A broken own_timeout falls back to the outer bound; one WARNING total."""
    hang = _make_hang()
    hang.own_timeout = lambda: 1 / 0

    with caplog.at_level(logging.WARNING, logger="corvidae.tool"):
        for i in range(2):
            r, _ = await _timed(
                dispatch_tool_call(_call(f"c{i}", "hang"), {"hang": hang}, call_timeouts=_tt(0.05))
            )
            assert r.content == "Error: tool 'hang' timed out after 0.05s"

    # The throttle: two dispatches, a single own_timeout warning.
    warns = [
        rec for rec in caplog.records
        if rec.levelno == logging.WARNING
        and "hang" in rec.getMessage()
        and "ZeroDivisionError" in (rec.getMessage() + (rec.exc_text or "") + str(rec.exc_info))
    ]
    own_warns = [w for w in warns if "timed out" not in w.getMessage()]
    assert len(own_warns) == 1


def test_own_timeout_on_bound_method():
    """A bound method reads own_timeout from its underlying function."""
    from corvidae.tool import resolve_own_timeout

    class Holder:
        async def run(self):
            return "x"

        run.own_timeout = 7

    assert resolve_own_timeout(Holder().run) == 7


def test_mock_tool_has_no_own_timeout(caplog):
    """An AsyncMock tool must not invent an own_timeout, nor warn."""
    from corvidae.tool import resolve_own_timeout

    with caplog.at_level(logging.WARNING):
        assert resolve_own_timeout(AsyncMock()) is None
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


# ---------------------------------------------------------------------------
# R5: cancellation and the WARNING text
# ---------------------------------------------------------------------------

CHANNEL = Channel(transport="test", scope="s")


async def test_timeout_cancels_and_warns_once(caplog):
    """The tool is cancelled; exactly one WARNING with the designed text."""
    flag = asyncio.Event()
    with caplog.at_level(logging.WARNING, logger="corvidae.tool"):
        await _timed(
            dispatch_tool_call(
                _call("call-9", "hang"), {"hang": _make_hang(flag)},
                channel=CHANNEL, call_timeouts=_tt(0.1),
            )
        )
    assert flag.is_set()
    recs = _timeout_records(caplog)
    assert len(recs) == 1
    assert recs[0].levelno == logging.WARNING
    assert recs[0].getMessage() == (
        "tool hang timed out after 0.1s on test:s (call call-9) and was cancelled; "
        "raise tools.timeouts.hang if this tool legitimately runs longer"
    )
    assert recs[0].exc_info


async def test_swallowed_cancel_is_discarded_and_warned(caplog):
    """A tool that swallows the cancel and returns: late result discarded."""

    async def hang(**kw):
        try:
            await asyncio.sleep(HANG)
        except asyncio.CancelledError:
            return "late"

    with caplog.at_level(logging.WARNING, logger="corvidae.tool"):
        r, elapsed = await _timed(
            dispatch_tool_call(
                _call("call-9", "hang"), {"hang": hang},
                channel=CHANNEL, call_timeouts=_tt(0.1),
            )
        )
    assert r.content == TIMEOUT_01
    assert r.error is True
    assert r.latency_ms is not None
    assert elapsed < 2.0
    recs = _timeout_records(caplog)
    assert len(recs) == 1
    assert recs[0].getMessage() == (
        "tool hang timed out after 0.1s on test:s (call call-9) but ignored "
        "cancellation; its late result was discarded — the tool must re-raise "
        "CancelledError; raise tools.timeouts.hang if this tool legitimately runs longer"
    )


async def test_swallowed_cancel_then_raise_is_timeout(caplog):
    """A tool that raises something else on cancel still yields the timeout."""

    async def hang(**kw):
        try:
            await asyncio.sleep(HANG)
        except asyncio.CancelledError:
            raise RuntimeError("x")

    with caplog.at_level(logging.WARNING, logger="corvidae.tool"):
        r, _ = await _timed(
            dispatch_tool_call(
                _call("call-9", "hang"), {"hang": hang},
                channel=CHANNEL, call_timeouts=_tt(0.1),
            )
        )
    assert r.content == TIMEOUT_01
    assert r.error is True
    ignored = [rec for rec in _timeout_records(caplog) if "ignored cancellation" in rec.getMessage()]
    assert len(ignored) == 1


# ---------------------------------------------------------------------------
# R2: through the Agent, a timeout frees the (single) task worker
# ---------------------------------------------------------------------------


async def test_agent_timeout_frees_worker_and_continues():
    """A timed-out call frees the only worker; the queued call and next turn run."""
    from helpers import build_plugin_and_channel, drain, drain_task_queue
    from llm_response_fixtures import (
        _make_text_response,
        _make_tool_call,
        _make_tool_call_response,
    )
    from corvidae.trace import mint_trace_id

    plugin, channel, db = await build_plugin_and_channel()
    pm = plugin.pm
    tp = pm.get_plugin("task")
    try:
        # Rebuild the task plugin with ONE worker (default is 4), so c2 can
        # only start after c1 has released the worker.
        await tp.on_stop()
        await tp.on_init(pm=pm, config={"daemon": {"max_task_workers": 1}})
        await tp.on_start(config={})
        assert tp.task_queue.max_workers == 1

        async def quick(**kw):
            return "quick-ok"

        plugin._tools = {"hang": _make_hang(), "quick": quick}
        # Set before the feature exists: the attribute is simply unused today.
        from corvidae.tool import ToolTimeouts

        plugin._tool_call_timeouts = ToolTimeouts(0.1)

        client = AsyncMock()
        client.chat = AsyncMock(
            side_effect=[
                _make_tool_call_response(
                    [_make_tool_call("c1", "hang", {}), _make_tool_call("c2", "quick", {})]
                ),
                _make_text_response("recovered"),
            ]
        )
        plugin._client = client
        plugin._mark_started()

        async def drive():
            await plugin.on_message(channel=channel, sender="user", text="go", trace_id=mint_trace_id())
            await drain(plugin, channel)
            await drain_task_queue(plugin)
            await drain(plugin, channel)

        start = time.monotonic()
        await asyncio.wait_for(drive(), GUARD)
        elapsed = time.monotonic() - start

        msgs = channel.conversation.messages
        # The Agent prefixes each task result with "[Task <id>] "; strip it.
        import re

        by_id = {
            m["tool_call_id"]: re.sub(r"^\[Task [0-9a-f]+\] ", "", m["content"])
            for m in msgs if m.get("role") == "tool"
        }
        assert by_id == {"c1": TIMEOUT_01, "c2": "quick-ok"}
        assert client.chat.await_count == 2
        assert msgs[-1]["role"] == "assistant"
        assert msgs[-1]["content"] == "recovered"
        assert elapsed < 2.0
    finally:
        await pm.get_plugin("task").on_stop()
        await db.close()
