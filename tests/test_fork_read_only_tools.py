"""Read-only tools in forks (#1190 PR 1, kernel half; criteria A1, A3-A7).

A fork may dispatch a tool only when the registering plugin declared it
read-only (``reads_only = True`` on the function), at most ``max_tool_calls``
times, with every result framed as untrusted data, and without firing hooks.
These tests drive the real ``Agent.fork`` with a scripted fake LLM client.
"""

import json
import logging
from unittest.mock import AsyncMock

import pytest

from corvidae.attribution import get_attribution, reset_attribution, set_attribution
from corvidae.fork import ForkRefused
from corvidae.hooks import hookimpl
from corvidae.provenance import PROVENANCE_KEY, ItemMetadata, encode, frame

# Reuse the fork suite's agent fixture and constants (same fake client wiring).
from test_fork import INSTRUCTION, forking_agent  # noqa: F401
from llm_response_fixtures import _make_text_response, _make_tool_call, _make_tool_call_response


def _schema(name: str) -> dict:
    """A parameterless function schema for ``name``."""
    return {"type": "function", "function": {"name": name, "parameters": {}}}


def _install_tools(plugin, **fns) -> dict[str, list]:
    """Register ``fns`` as the agent's tools; return a per-tool call log."""
    plugin._tools = dict(fns)
    plugin._tool_schemas = [_schema(n) for n in fns]
    return {n: [] for n in fns}


def _make_tools(log_attr: list):
    """A read-only tool ``lookup`` and an undeclared tool ``writer``.

    Each records the attribution seen inside its body in ``log_attr``.
    """

    async def lookup() -> str:
        """Read something."""
        log_attr.append(("lookup", dict(get_attribution())))
        return "RAW-LOOKUP-CONTENT"

    lookup.reads_only = True

    async def writer() -> str:
        """Write something (no read-only declaration)."""
        log_attr.append(("writer", dict(get_attribution())))
        return "WROTE"

    return lookup, writer


def _tool_msgs(messages: list[dict]) -> list[dict]:
    """The role=tool messages of one request."""
    return [m for m in messages if m.get("role") == "tool"]


def _script(client, replies: list[dict]):
    """Make ``client.chat`` return ``replies`` in order, recording each call."""
    seen: list[dict] = []
    it = iter(replies)

    async def chat(messages, **kwargs):
        # Snapshot the request and the attribution at the moment of the call.
        seen.append({
            "messages": [dict(m) for m in messages],
            "extra_body": dict(kwargs.get("extra_body") or {}),
            "attribution": dict(get_attribution()),
        })
        return next(it)

    client.chat = AsyncMock(side_effect=chat)
    return seen


def _call_reply(*calls):
    """A scripted model reply carrying ``calls`` as structured tool calls."""
    return _make_tool_call_response(list(calls))


# A1 -------------------------------------------------------------------------


def test_declares_read_only_is_true_only_for_exactly_true():
    """A1: only the exact ``True`` declaration counts; absent or truthy does not."""
    from corvidae.tool import declares_read_only

    async def plain():
        return ""

    async def yes():
        return ""

    yes.reads_only = True

    async def truthy():
        return ""

    truthy.reads_only = 1

    assert declares_read_only(plain) is False
    assert declares_read_only(yes) is True
    assert declares_read_only(truthy) is False


@pytest.fixture
def keyed_agent(forking_agent):
    """``forking_agent`` with a fence key on the window, as production sets one.

    The main loop frames tool results with the window's fence key; the shared
    fixture leaves it None, which ``provenance.frame`` cannot hash.
    """
    plugin, channel, client = forking_agent
    channel.conversation.fence_key = b"k" * 32
    return plugin, channel, client


def test_mcp_tool_is_never_declared_read_only():
    """A1: the MCP client builds tools without the read-only declaration."""
    from unittest.mock import MagicMock

    from corvidae.mcp_client import _make_tool, _McpServerState
    from corvidae.tool import declares_read_only

    # A minimal server state and MCP tool definition, as test_mcp_client does.
    server = _McpServerState(
        name="fs", session=MagicMock(), prefix="fs", timeout_seconds=30.0, mcp_tools=[]
    )
    mcp_tool = MagicMock()
    mcp_tool.name = "search"
    mcp_tool.description = ""
    mcp_tool.input_schema = {"type": "object", "properties": {}}

    tool = _make_tool(server, mcp_tool)

    assert declares_read_only(tool.fn) is False


# A3 -------------------------------------------------------------------------


async def test_fork_frames_every_result_as_untrusted_data(keyed_agent):
    plugin, channel, client = keyed_agent
    lookup, writer = _make_tools([])
    _install_tools(plugin, lookup=lookup, writer=writer)
    seen = _script(client, [
        _call_reply(_make_tool_call("c1", "lookup", {})),
        _make_text_response("verdict"),
    ])

    await plugin.fork(channel, INSTRUCTION, stage="consolidation", max_tool_calls=2)

    # lookup does not declare returns_untrusted, yet its result is framed.
    # The window already renders tool rows of its own; take the fork's by id.
    (tool_msg,) = [m for m in _tool_msgs(seen[1]["messages"]) if m["tool_call_id"] == "c1"]
    row = {
        "role": "tool", "tool_call_id": "c1", "content": "RAW-LOOKUP-CONTENT",
        PROVENANCE_KEY: encode(ItemMetadata(tool="lookup", untrusted=True), {}),
    }
    expected = frame(row, row, channel.id, channel.conversation.fence_key)[1]["content"]
    content = tool_msg["content"]
    assert content == expected
    assert "[metadata: result of your own lookup call c1;" in content
    assert "untrusted outside content" in content
    assert " begins]" in content and " ends]" in content
    inside = content.split(" begins]\n", 1)[1].rsplit("\n[payload", 1)[0]
    assert inside == "RAW-LOOKUP-CONTENT"


# A4 -------------------------------------------------------------------------


async def test_cap_limits_calls_and_the_next_request_sends_tool_choice_none(
    keyed_agent, caplog
):
    plugin, channel, client = keyed_agent
    calls: list = []
    lookup, writer = _make_tools(calls)
    _install_tools(plugin, lookup=lookup, writer=writer)
    seen = _script(client, [
        _call_reply(_make_tool_call("c1", "lookup", {})),
        # Two calls in one reply, one call of cap left: the second is over.
        _call_reply(_make_tool_call("c2", "lookup", {}), _make_tool_call("c3", "lookup", {})),
        _make_text_response("verdict"),
    ])
    token = set_attribution(trace_id="trace-1190")
    try:
        with caplog.at_level(logging.INFO, logger="corvidae.agent"):
            result = await plugin.fork(
                channel, INSTRUCTION, stage="consolidation", max_tool_calls=2
            )
    finally:
        reset_attribution(token)

    assert result.text == "verdict"
    # At most N=2 invocations across all requests.
    assert len(calls) == 2
    # Calls remain: auto. Cap reached: none.
    assert [s["extra_body"]["tool_choice"] for s in seen] == ["auto", "auto", "none"]
    # The over-cap call got an error result, not a dispatch.
    last_tools = {m["tool_call_id"]: m["content"] for m in _tool_msgs(seen[2]["messages"])}
    assert "tool-call limit reached" in last_tools["c3"]
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING
                and getattr(r, "stage", None) == "consolidation"
                and "limit" in r.getMessage()]
    assert len(warnings) == 1
    assert warnings[0].channel == channel.id
    assert warnings[0].trace_id == "trace-1190"


async def test_tool_call_in_a_reply_sent_with_tool_choice_none_is_refused(keyed_agent):
    plugin, channel, client = keyed_agent
    lookup, writer = _make_tools([])
    _install_tools(plugin, lookup=lookup, writer=writer)
    _script(client, [
        _call_reply(_make_tool_call("c1", "lookup", {})),
        # The request after the cap sent "none"; a call in its reply is refused.
        _call_reply(_make_tool_call("c2", "lookup", {})),
    ])
    with pytest.raises(ForkRefused):
        await plugin.fork(channel, INSTRUCTION, stage="consolidation", max_tool_calls=1)


# A5 -------------------------------------------------------------------------


async def test_cap_zero_tool_call_reply_raises_fork_refused(keyed_agent):
    plugin, channel, client = keyed_agent
    calls: list = []
    lookup, writer = _make_tools(calls)
    _install_tools(plugin, lookup=lookup, writer=writer)
    _script(client, [_call_reply(_make_tool_call("c1", "lookup", {}))])
    with pytest.raises(ForkRefused):
        await plugin.fork(channel, INSTRUCTION, stage="consolidation", max_tool_calls=0)
    assert calls == []


# A6 -------------------------------------------------------------------------


async def test_undeclared_tool_is_refused_by_result_and_not_invoked(keyed_agent):
    plugin, channel, client = keyed_agent
    calls: list = []
    lookup, writer = _make_tools(calls)
    _install_tools(plugin, lookup=lookup, writer=writer)
    seen = _script(client, [
        _call_reply(_make_tool_call("c1", "writer", {})),
        _make_text_response("verdict"),
    ])

    result = await plugin.fork(channel, INSTRUCTION, stage="consolidation", max_tool_calls=2)

    assert result.text == "verdict"
    assert [name for name, _ in calls] == []
    # The window already renders tool rows of its own; take the fork's by id.
    (tool_msg,) = [m for m in _tool_msgs(seen[1]["messages"]) if m["tool_call_id"] == "c1"]
    assert tool_msg["tool_call_id"] == "c1"
    assert "writer" in tool_msg["content"]
    assert "not available in a fork" in tool_msg["content"]
    # A refusal is a tool result too, so it reaches the model framed (A3).
    assert " begins]" in tool_msg["content"]


def _calls_with(**override):
    """One call dict, with ``override`` keys replacing the good ones."""
    good = {"id": "c1", "function": {"name": "lookup", "arguments": "{}"}}
    return {**good, **override}


@pytest.mark.parametrize("bad", [
    _calls_with(function={"arguments": "{}"}),
    _calls_with(function={"name": "lookup"}),
    _calls_with(function={"name": "lookup", "arguments": {"a": 1}}),
    _calls_with(function="lookup"),
    _calls_with(id=""),
    _calls_with(id=7),
    "not-a-call",
], ids=[
    "no-name", "no-arguments", "non-str-arguments", "function-not-dict",
    "empty-id", "non-str-id", "call-not-dict",
])
async def test_malformed_call_raises_fork_refused(keyed_agent, bad):
    plugin, channel, client = keyed_agent
    calls: list = []
    lookup, writer = _make_tools(calls)
    _install_tools(plugin, lookup=lookup, writer=writer)
    _script(client, [_call_reply(bad), _make_text_response("verdict")])
    with pytest.raises(ForkRefused):
        await plugin.fork(channel, INSTRUCTION, stage="consolidation", max_tool_calls=3)
    assert calls == []


async def test_id_duplicated_across_replies_raises_fork_refused(keyed_agent):
    plugin, channel, client = keyed_agent
    calls: list = []
    lookup, writer = _make_tools(calls)
    _install_tools(plugin, lookup=lookup, writer=writer)
    _script(client, [
        _call_reply(_make_tool_call("dup", "lookup", {})),
        _call_reply(_make_tool_call("dup", "lookup", {})),
        _make_text_response("verdict"),
    ])
    with pytest.raises(ForkRefused):
        await plugin.fork(channel, INSTRUCTION, stage="consolidation", max_tool_calls=3)


async def test_id_duplicated_within_a_reply_raises_fork_refused(keyed_agent):
    plugin, channel, client = keyed_agent
    lookup, writer = _make_tools([])
    _install_tools(plugin, lookup=lookup, writer=writer)
    _script(client, [
        _call_reply(_make_tool_call("dup", "lookup", {}), _make_tool_call("dup", "lookup", {})),
        _make_text_response("verdict"),
    ])
    with pytest.raises(ForkRefused):
        await plugin.fork(channel, INSTRUCTION, stage="consolidation", max_tool_calls=3)


# A7 -------------------------------------------------------------------------


class _Spy:
    """Counts transform_tool_result calls."""

    def __init__(self):
        self.calls = 0

    @hookimpl
    async def transform_tool_result(self, result, **kwargs):
        self.calls += 1
        return result


async def test_fork_tool_calls_fire_no_hooks_and_are_attributed(keyed_agent, caplog):
    plugin, channel, client = keyed_agent
    calls: list = []
    lookup, writer = _make_tools(calls)
    _install_tools(plugin, lookup=lookup, writer=writer)
    spy = _Spy()
    plugin.pm.register(spy, name="spy")
    seen = _script(client, [
        _call_reply(_make_tool_call("c1", "lookup", {})),
        _make_text_response("verdict"),
    ])
    token = set_attribution(trace_id="trace-a7", stage="consolidation")
    try:
        with caplog.at_level(logging.INFO, logger="corvidae.agent"):
            await plugin.fork(channel, INSTRUCTION, stage="consolidation", max_tool_calls=2)
    finally:
        reset_attribution(token)

    # No hook fired for the tool call.
    assert spy.calls == 0
    # Every model call and the tool body ran under mode=fork and the stage.
    assert len(seen) == 2
    for s in seen:
        assert s["attribution"]["mode"] == "fork"
        assert s["attribution"]["stage"] == "consolidation"
    assert [name for name, _ in calls] == ["lookup"]
    inside = calls[0][1]
    assert inside["mode"] == "fork" and inside["stage"] == "consolidation"
    # One INFO record per tool call naming stage, channel, trace and tool.
    records = [r for r in caplog.records if r.getMessage() == "fork tool call"]
    assert len(records) == 1
    rec = records[0]
    assert rec.levelno == logging.INFO
    assert (rec.stage, rec.channel, rec.trace_id, rec.tool) == (
        "consolidation", channel.id, "trace-a7", "lookup",
    )


# yield_to_caller (#1190 PR 2) ------------------------------------------------


async def test_yield_to_caller_runs_between_rounds_not_before_the_first_request(keyed_agent):
    """The callback is awaited after each round's tool results and before the
    next model request; never before the first request, and not after the
    final reply."""
    plugin, channel, client = keyed_agent
    lookup, writer = _make_tools([])
    _install_tools(plugin, lookup=lookup, writer=writer)
    seen = _script(client, [
        _call_reply(_make_tool_call("c1", "lookup", {})),
        _call_reply(_make_tool_call("c2", "lookup", {})),
        _make_text_response("verdict"),
    ])
    # How many requests had been sent each time the callback ran.
    sent_at_yield: list[int] = []

    async def pause() -> None:
        sent_at_yield.append(len(seen))

    result = await plugin.fork(
        channel, INSTRUCTION, stage="consolidation", max_tool_calls=2,
        yield_to_caller=pause)

    assert result.text == "verdict"
    # After request 1 and after request 2; not at 0 (before the first), not 3.
    assert sent_at_yield == [1, 2]


async def test_yield_to_caller_is_not_awaited_by_a_fork_without_tool_rounds(keyed_agent):
    """A fork that answers at once has no round boundary, so the callback never runs."""
    plugin, channel, client = keyed_agent
    _script(client, [_make_text_response("verdict")])
    calls: list[int] = []

    async def pause() -> None:
        calls.append(1)

    await plugin.fork(
        channel, INSTRUCTION, stage="consolidation", max_tool_calls=2,
        yield_to_caller=pause)
    assert calls == []


async def test_exception_from_yield_to_caller_propagates_out_of_fork(keyed_agent, caplog):
    """An exception from the callback ends the fork: it propagates, no further
    request is sent, and the stage's "stage done" line says ok=False."""
    plugin, channel, client = keyed_agent
    lookup, writer = _make_tools([])
    _install_tools(plugin, lookup=lookup, writer=writer)
    seen = _script(client, [
        _call_reply(_make_tool_call("c1", "lookup", {})),
        _make_text_response("never sent"),
    ])

    async def stale() -> None:
        raise RuntimeError("channel went stale")

    with caplog.at_level(logging.INFO, logger="corvidae.agent"):
        with pytest.raises(RuntimeError, match="channel went stale"):
            await plugin.fork(
                channel, INSTRUCTION, stage="consolidation", max_tool_calls=2,
                yield_to_caller=stale)

    assert len(seen) == 1
    [done] = [r for r in caplog.records if r.getMessage() == "stage done"]
    assert done.ok is False
