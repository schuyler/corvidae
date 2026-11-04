"""Every cut-off reply is logged where it happens (#381, PRD §3.10).

`LLMClient.chat` flags a cut-off with ``reasoning_unfinished``, but only the
main turn ever logged it. A cut-off on a fork, a subagent, compaction or a
cognition stage was flagged and silently discarded, so the one record of what
the model actually produced never reached the log.

Logging it in ``chat`` covers every caller. The warning names the channel when
there is an attribution and carries the raw completion, per the §3.10 rule
that an anomalous response is logged with the channel and its text.
"""

from __future__ import annotations

import copy
import logging

import pytest

from corvidae.attribution import reset_attribution, set_attribution
from corvidae.llm import LLMClient

from test_llm_hooks import BASE_URL, MODEL, _make_mock_response, _make_mock_session

pytestmark = pytest.mark.timeout(15)

CUT_OFF = {
    "id": "chatcmpl-cut",
    "object": "chat.completion",
    "choices": [{
        "index": 0,
        # No closing </think>: reasoning never finished.
        "message": {"role": "assistant", "content": "thinking about the answer"},
        "finish_reason": "length",
    }],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}


def _client(body: dict) -> LLMClient:
    client = LLMClient(base_url=BASE_URL, model=MODEL)
    client.raw_reasoning = True
    # chat() splits the think block in place, so each test needs its own copy.
    client.session = _make_mock_session(_make_mock_response(json_body=copy.deepcopy(body)))
    return client


async def test_a_cut_off_is_logged_by_chat_itself(caplog):
    """A caller that never inspects the flag still leaves a record."""
    client = _client(CUT_OFF)
    with caplog.at_level(logging.WARNING, logger="corvidae.llm"):
        await client.chat([{"role": "user", "content": "hi"}])

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("cut off" in w.lower() for w in warnings), warnings


async def test_the_cut_off_warning_names_the_channel(caplog):
    """§3.10: the warning carries the channel from the attribution."""
    client = _client(CUT_OFF)
    token = set_attribution(stage="appraisal", channel_id="test:cut")
    try:
        with caplog.at_level(logging.WARNING, logger="corvidae.llm"):
            await client.chat([{"role": "user", "content": "hi"}])
    finally:
        reset_attribution(token)

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    matching = [w for w in warnings if "cut off" in w.lower()]
    assert matching, warnings
    assert "test:cut" in matching[0]


async def test_the_cut_off_warning_carries_the_response_text(caplog):
    """§3.10: the anomalous response itself is logged, not just a flag."""
    client = _client(CUT_OFF)
    with caplog.at_level(logging.WARNING, logger="corvidae.llm"):
        await client.chat([{"role": "user", "content": "hi"}])

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    matching = [w for w in warnings if "cut off" in w.lower()]
    assert matching, warnings
    assert "thinking about the answer" in matching[0]


async def test_a_complete_reply_is_not_logged(caplog):
    """A normal reply must stay silent: this warning is for anomalies only."""
    body = {
        "id": "chatcmpl-ok",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "reasoning</think>the answer"},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    client = _client(body)
    with caplog.at_level(logging.WARNING, logger="corvidae.llm"):
        await client.chat([{"role": "user", "content": "hi"}])

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert not any("cut off" in w.lower() for w in warnings), warnings
