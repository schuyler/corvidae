"""Unit tests for ContextWindow.fit_to_budget (#308, PRD §3.3.2).

fit_to_budget is the pure, synchronous over-budget fallback: it drops the
oldest rows from the in-memory window (snapping tool cycles forward), and
truncates the newest row in memory only when that row alone exceeds the
budget. One test per case of the approved design's list (1-7).
"""

from __future__ import annotations

import re

import pytest

from corvidae.context import (
    ORPHAN_RESULT_LABEL,
    ContextWindow,
    MessageType,
    count_tokens,
    message_tokens,
)

pytestmark = pytest.mark.timeout(15)


def words(n: int) -> str:
    """Text of about n tokens (one 'word' per token)."""
    return " ".join(["word"] * n)


def make_window(rows: list[dict], system_prompt: str = "sys") -> ContextWindow:
    """A ContextWindow holding ``rows`` as MESSAGE rows, tagged with _db_id
    1..n unless a row already carries a _message_type or _db_id."""
    conv = ContextWindow("test:unit")
    conv.system_prompt = system_prompt
    for i, row in enumerate(rows, start=1):
        tagged = {"_db_id": i, "_message_type": MessageType.MESSAGE, **row}
        conv.messages.append(tagged)
    return conv


def tail_budget(conv: ContextWindow, keep: int) -> int:
    """The smallest budget that fits the system prompt plus the last
    ``keep`` rows exactly (so dropping fewer rows cannot fit)."""
    return count_tokens(conv.system_prompt) + sum(
        message_tokens(m) for m in conv.messages[len(conv.messages) - keep:]
    )


def test_within_budget_returns_nothing_and_leaves_window_identical() -> None:
    """1: a window already within budget is untouched: ([], False)."""
    conv = make_window([{"role": "user", "content": "hi"},
                        {"role": "assistant", "content": "hello"}])
    before = [dict(m) for m in conv.messages]
    dropped, truncated = conv.fit_to_budget(conv.token_estimate())
    assert (dropped, truncated) == ([], False)
    assert conv.messages == before


def test_over_budget_drops_the_minimum_oldest_rows() -> None:
    """2: over budget drops exactly the fewest oldest rows that fit."""
    conv = make_window([
        {"role": "user" if i % 2 == 0 else "assistant", "content": words(40)}
        for i in range(6)
    ])
    original = list(conv.messages)
    budget = tail_budget(conv, keep=4)
    dropped, truncated = conv.fit_to_budget(budget)
    assert truncated is False
    # Oldest first, and no more than necessary.
    assert dropped == original[:2]
    assert conv.messages == original[2:]
    assert conv.token_estimate() <= budget


@pytest.mark.parametrize("head_type", [MessageType.SUMMARY, MessageType.CONTEXT])
def test_summary_or_context_head_row_is_dropped_first(head_type) -> None:
    """3: a SUMMARY/CONTEXT row at the head has no special status."""
    conv = make_window([
        {"role": "assistant", "content": words(40), "_message_type": head_type},
        {"role": "user", "content": words(40)},
        {"role": "assistant", "content": words(40)},
    ])
    original = list(conv.messages)
    dropped, _ = conv.fit_to_budget(tail_budget(conv, keep=2))
    assert dropped == [original[0]]
    assert conv.messages == original[1:]


def test_newest_row_alone_over_budget_is_truncated_in_memory() -> None:
    """4: only the newest row remains, cut with a marker naming k and M;
    its tags survive and the original dict is not mutated."""
    huge = {"role": "user", "content": words(1000)}
    conv = make_window([{"role": "user", "content": "old"}, huge])
    newest = conv.messages[-1]
    original_content = newest["content"]
    budget = 200
    dropped, truncated = conv.fit_to_budget(budget)
    assert truncated is True
    assert len(dropped) == 1
    assert len(conv.messages) == 1
    kept = conv.messages[0]
    # The marker names how much was kept (k) of the original length (M).
    match = re.search(
        r"\[truncated to fit the context window: showing the first (\d+) of "
        r"(\d+) characters\]$",
        kept["content"],
    )
    assert match, kept["content"][-120:]
    k, m = int(match.group(1)), int(match.group(2))
    assert m == len(original_content)
    assert kept["content"].startswith(original_content[:k])
    assert conv.token_estimate() <= budget
    # Tags carried over; the original dict is a different, untouched object.
    assert kept["_db_id"] == 2
    assert kept["_message_type"] == MessageType.MESSAGE
    assert kept is not newest
    assert newest["content"] == original_content


def test_tool_cycle_straddling_the_cut_snaps_forward_past_the_result() -> None:
    """5: a cut between a tool_calls row and its result moves past the
    result: no kept tool row whose call was dropped."""
    call = {"id": "c1", "type": "function",
            "function": {"name": "t", "arguments": "{}"}}
    conv = make_window([
        {"role": "user", "content": words(30)},
        {"role": "assistant", "content": "", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "c1", "content": words(30)},
        {"role": "user", "content": words(30)},
        {"role": "assistant", "content": words(30)},
        {"role": "user", "content": words(30)},
    ])
    original = list(conv.messages)
    # Budget that fits rows 2.. exactly puts the raw cut at 2, between
    # the call (1) and its result (2).
    dropped, _ = conv.fit_to_budget(tail_budget(conv, keep=4))
    assert dropped == original[:3]
    assert conv.messages == original[3:]
    assert not [m for m in conv.messages if m["role"] == "tool"]


def test_snap_reaching_the_end_keeps_the_cut_and_renders_an_orphan() -> None:
    """6: when snapping would drop the newest row (the result), the raw cut
    stays; build_prompt frames the orphaned result as a notification."""
    call = {"id": "c1", "type": "function",
            "function": {"name": "t", "arguments": "{}"}}
    conv = make_window([
        {"role": "user", "content": words(30)},
        {"role": "assistant", "content": "", "tool_calls": [call]},
        {"role": "user", "content": words(30)},
        {"role": "tool", "tool_call_id": "c1", "content": words(30)},
    ])
    original = list(conv.messages)
    conv.fit_to_budget(tail_budget(conv, keep=2))
    # The newest row survives even though its call is gone.
    assert conv.messages == original[2:]
    prompt = conv.build_prompt()
    assert not [m for m in prompt if m["role"] == "tool"]
    assert any(ORPHAN_RESULT_LABEL in (m.get("content") or "") for m in prompt)


def test_system_prompt_leaving_no_room_keeps_newest_row_unchanged() -> None:
    """7: no room for even a marker: no exception, the newest row is left
    as is, truncated is False, and the list is mutated in place."""
    conv = make_window(
        [{"role": "user", "content": "first"},
         {"role": "user", "content": "newest question"}],
        system_prompt=words(200),
    )
    rows = conv.messages
    newest = conv.messages[-1]
    dropped, truncated = conv.fit_to_budget(100)
    assert truncated is False
    assert [m["content"] for m in dropped] == ["first"]
    # Same list object, same newest row, no marker.
    assert conv.messages is rows
    assert conv.messages == [newest]
    assert conv.messages[0]["content"] == "newest question"
