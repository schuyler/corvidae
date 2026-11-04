"""Prompt build cannot fail on any row the window can hold (PRD §3.1.18.1, #392).

Every step below runs unwrapped over the whole window on every turn, so an
exception here is a deterministic per-turn failure: the channel never answers
again. Each test is one of the cases the issue reproduces.

Log assertions check the identifiers and consequence, not just the absence of
a raise (AGENTS.md: test the message, not just the failure).
"""

from __future__ import annotations

import datetime
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from corvidae.channel import Channel  # noqa: F401 (fixture type hint)
from corvidae.context import (
    ORPHAN_RESULT_LABEL,
    ContextWindow,
    MessageType,
    message_tokens,
)
from corvidae.fork import snap_cut_forward, snap_cut_to_tool_boundary
from corvidae.outbound import OutboundMessage
from corvidae.trace import mint_trace_id

from helpers import build_plugin_and_channel, drain, sent_texts
from llm_response_fixtures import _make_text_response

pytestmark = pytest.mark.timeout(15)


def make_window(rows: list, system_prompt: str = "sys") -> ContextWindow:
    """A ContextWindow holding ``rows`` with _db_id 1..n and MESSAGE tags.

    Rows are placed directly rather than through ``append`` so a malformed
    entry can reach the window, which is what the issue describes.
    """
    conv = ContextWindow("test:malformed")
    conv.system_prompt = system_prompt
    for i, row in enumerate(rows, start=1):
        if isinstance(row, dict):
            conv.messages.append(
                {"_db_id": i, "_message_type": MessageType.MESSAGE, **row}
            )
        else:
            conv.messages.append(row)
    return conv


# A tool call with no "id": an OpenAI-compatible backend can omit it, and
# turn.py passes tool_calls through unnormalized.
CALL_NO_ID = {"type": "function", "function": {"name": "f", "arguments": "{}"}}


# ---------------------------------------------------------------------------
# R1: token estimation never raises on a row's content
# ---------------------------------------------------------------------------


def test_message_tokens_accepts_a_non_json_serializable_value() -> None:
    """A datetime in content is counted, not a TypeError."""
    assert message_tokens({"role": "system", "content": datetime.datetime.now()}) > 0


def test_token_estimate_accepts_a_non_json_serializable_value() -> None:
    """The issue's exact reproduction: token_estimate on a datetime row."""
    conv = make_window([{"role": "system", "content": datetime.datetime.now()}])
    assert conv.token_estimate() > 0


# ---------------------------------------------------------------------------
# R2: fit_to_budget and the cut snaps tolerate malformed tool_calls
# ---------------------------------------------------------------------------


def test_fit_to_budget_accepts_a_tool_call_with_no_id() -> None:
    """The issue's reproduction: fit_to_budget(1) raised KeyError: 'id'."""
    conv = make_window([
        {"role": "assistant", "content": "", "tool_calls": [CALL_NO_ID]},
        {"role": "user", "content": "hi"},
    ])
    dropped, truncated = conv.fit_to_budget(1)
    # The budget only fits the newest row, so the call row is the one dropped.
    assert [m["content"] for m in dropped] == [""]
    assert truncated is False
    assert [m["content"] for m in conv.messages] == ["hi"]


@pytest.mark.parametrize(
    "tool_calls",
    [
        pytest.param([CALL_NO_ID], id="call-without-id"),
        pytest.param("not-a-list", id="string"),
        pytest.param({"id": "call-1"}, id="single-dict"),
        pytest.param([None], id="none-entry"),
    ],
)
def test_snap_cuts_accept_malformed_tool_calls(tool_calls) -> None:
    """Both snaps return a usable cut instead of raising.

    A call with no usable id is not a call, so it never straddles a cut:
    the snaps leave the cut where it was rather than moving it around it.
    """
    messages = [
        {"role": "assistant", "content": "", "tool_calls": tool_calls},
        {"role": "tool", "tool_call_id": "call-1", "content": "r"},
        {"role": "user", "content": "hi"},
    ]
    assert snap_cut_forward(messages, 1) == 1
    assert snap_cut_to_tool_boundary(messages, 2) == 2


# ---------------------------------------------------------------------------
# R3/R4: build_prompt renders what it can and logs what it leaves out
# ---------------------------------------------------------------------------


def test_build_prompt_accepts_a_tool_call_with_no_id() -> None:
    """The issue's reproduction: build_prompt() raised KeyError: 'id'."""
    conv = make_window([
        {"role": "assistant", "content": "", "tool_calls": [CALL_NO_ID]},
        {"role": "user", "content": "hi"},
    ])
    prompt = conv.build_prompt()
    assert prompt[0]["role"] == "system"
    assert any(m.get("content") == "hi" for m in prompt)


def test_build_prompt_omits_an_unrenderable_row_and_names_it(caplog) -> None:
    """A row that is not a message object stays out of the prompt, and the
    warning names the channel and the row id."""
    conv = make_window([
        {"role": "user", "content": "before"},
        ["not", "a", "message"],
        {"role": "user", "content": "after"},
    ])
    with caplog.at_level(logging.WARNING, logger="corvidae.context"):
        prompt = conv.build_prompt()

    contents = [m.get("content") for m in prompt]
    assert "before" in contents
    assert "after" in contents

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    matching = [w for w in warnings if "position 1 (not persisted)" in w]
    assert matching, f"no warning naming the row; got {warnings}"
    message = matching[0]
    assert "test:malformed" in message
    assert "left out of the prompt" in message


def test_build_prompt_warns_when_it_drops_a_malformed_tool_call(caplog) -> None:
    """A row left with nothing renderable is omitted, and it says so, naming
    the channel and the row."""
    conv = make_window([
        {"role": "assistant", "content": "", "tool_calls": [CALL_NO_ID]},
        {"role": "user", "content": "hi"},
    ])
    with caplog.at_level(logging.WARNING, logger="corvidae.context"):
        conv.build_prompt()

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    matching = [w for w in warnings if "message_log id 1" in w]
    assert matching, f"no warning naming row 1; got {warnings}"
    assert "test:malformed" in matching[0]
    assert "left out of the prompt" in matching[0]


def test_build_prompt_keeps_a_row_that_still_has_content(caplog) -> None:
    """A row with a malformed call plus real text keeps the text, and the
    warning says only the call was dropped."""
    conv = make_window([
        {"role": "assistant", "content": "I will check.", "tool_calls": [CALL_NO_ID]},
        {"role": "user", "content": "hi"},
    ])
    with caplog.at_level(logging.WARNING, logger="corvidae.context"):
        prompt = conv.build_prompt()

    rendered = [m for m in prompt if m.get("content") == "I will check."]
    assert rendered, f"the row's text was dropped; prompt was {prompt}"
    assert "tool_calls" not in rendered[0]

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    matching = [w for w in warnings if "message_log id 1" in w]
    assert matching, f"no warning naming row 1; got {warnings}"
    assert "rendered without them" in matching[0]


def test_build_prompt_warns_for_a_malformed_call_on_an_anchored_row(caplog) -> None:
    """A tool-call row whose cycle completes renders on the deferred block
    path; its malformed sibling call is dropped there too, and said so."""
    good = {"id": "call-ok", "type": "function", "function": {"name": "f", "arguments": "{}"}}
    conv = make_window([
        {"role": "assistant", "content": "", "tool_calls": [good, CALL_NO_ID]},
        {"role": "tool", "tool_call_id": "call-ok", "content": "result"},
        {"role": "user", "content": "hi"},
    ])
    with caplog.at_level(logging.WARNING, logger="corvidae.context"):
        prompt = conv.build_prompt()

    rendered = [m for m in prompt if m.get("role") == "assistant"]
    assert rendered, f"the row went missing; prompt was {prompt}"
    assert [c["id"] for c in rendered[0]["tool_calls"]] == ["call-ok"]

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("message_log id 1" in w and "rendered without them" in w for w in warnings), warnings


def test_token_estimate_counts_a_row_with_a_non_serializable_value() -> None:
    """The degraded count still covers the row's other fields: a row holding a
    large plain string is not counted as if it were empty."""
    conv = make_window([
        {"role": "system", "content": datetime.datetime.now(), "extra": "word " * 500},
    ])
    empty = make_window([{"role": "system", "content": ""}])
    assert conv.token_estimate() > empty.token_estimate() + 100


# ---------------------------------------------------------------------------
# R5: a turn completes on a channel whose window holds such a row
# ---------------------------------------------------------------------------


async def test_turn_completes_with_a_malformed_row_in_the_window() -> None:
    """End to end through _process_queue_item: the agent still answers."""
    plugin, channel, db = await build_plugin_and_channel()
    try:
        conv = ContextWindow(channel.id)
        conv.system_prompt = "You are a test assistant."
        conv.messages.append(
            {
                "_db_id": 1,
                "_message_type": MessageType.MESSAGE,
                "role": "assistant",
                "content": "",
                "tool_calls": [CALL_NO_ID],
            }
        )
        channel.conversation = conv

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=_make_text_response("answered"))
        plugin._client = mock_client
        plugin._mark_started()

        await plugin.on_message(
            channel=channel, sender="user", text="hello", trace_id=mint_trace_id()
        )
        await drain(plugin, channel)

        assert sent_texts(plugin.pm, OutboundMessage) == ["answered"]
    finally:
        task_plugin = plugin.pm.get_plugin("task")
        if task_plugin:
            await task_plugin.on_stop()
        await db.close()


# ---------------------------------------------------------------------------
# Review follow-ups: the same row classes on the paths the first pass missed
# ---------------------------------------------------------------------------


def test_fit_to_budget_leaves_a_non_mapping_newest_row_alone() -> None:
    """The truncation branch must not assume the newest row has `.get`.

    It runs after the head has already been dropped, so raising here loses
    rows and then fails the same way on every later turn (#392 review).
    """
    conv = make_window([
        {"role": "user", "content": "a" * 5000},
        ["not", "a", "message"],
    ])
    dropped, truncated = conv.fit_to_budget(1)
    assert truncated is False
    assert conv.messages == [["not", "a", "message"]]


def test_build_prompt_accepts_an_unhashable_tool_call_id() -> None:
    """`tool_call_id` arrives from message_log unvalidated; a list cannot be
    a set member, and the membership test must not raise (#392 review)."""
    conv = make_window([
        {"role": "tool", "tool_call_id": ["a"], "content": "r"},
        {"role": "user", "content": "hi"},
    ])
    prompt = conv.build_prompt()
    # Its call is not in the window, so it renders as a framed orphan result.
    assert any(ORPHAN_RESULT_LABEL in m.get("content", "") for m in prompt)


def test_build_prompt_filters_a_falsy_non_list_tool_calls() -> None:
    """An empty-string `tool_calls` is still malformed: render without it and
    say so, rather than passing it to a chat template (#392 review)."""
    conv = make_window([
        {"role": "assistant", "content": "text", "tool_calls": ""},
        {"role": "user", "content": "hi"},
    ])
    prompt = conv.build_prompt()
    rendered = [m for m in prompt if m.get("content") == "text"]
    assert rendered, prompt
    assert "tool_calls" not in rendered[0]


def test_warn_over_budget_accepts_a_non_mapping_dropped_row() -> None:
    """The over-budget warning reads `_db_id` off every dropped row (#392)."""
    from corvidae.agent import Agent

    conv = make_window([
        ["not", "a", "message"],
        {"role": "user", "content": "x" * 4000},
    ])
    dropped, truncated = conv.fit_to_budget(50)

    channel = type("C", (), {"id": "test:malformed"})()
    # Must not raise; the non-mapping row simply carries no id.
    Agent._warn_over_budget(None, channel, conv, 15000, dropped, truncated)


def test_build_fork_prompt_accepts_a_tool_call_with_no_id() -> None:
    """A fork renders the same prefix the main turn does, so it needs the
    same tolerance: a compaction fork and the background view both go
    through it (#392 review)."""
    from corvidae.fork import build_fork_prompt

    conv = make_window([
        {"role": "assistant", "content": "", "tool_calls": [CALL_NO_ID]},
        {"role": "user", "content": "hi"},
    ])
    prompt = build_fork_prompt(conv, None, "summarize", frozenset(), "system")
    assert prompt[0]["role"] == "system"
    assert prompt[-1] == {"role": "user", "content": "summarize"}
