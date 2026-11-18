"""A catch-up segment whose newest row is one huge tool call fits (#1045).

The production row that showed the gap: empty content, a short
reasoning_content and a 351 KB tool call. fit_to_budget used to cut content
only, so the segment went out about 13x over budget with the step-8
"sending it anyway" warning. The builder now truncates the call's arguments.
"""

import json
import logging

import aiosqlite
import pytest
from corvidae_cognition.memory import (
    DEFAULT_CONSOLIDATION_FORK_PROMPT,
    _measured_size,
    build_catchup_segments,
    logger,
)

from corvidae.fork import ViewInputs
from corvidae.thinking import strip_reasoning_content

pytestmark = pytest.mark.timeout(15)

CHANNEL = "irc:#oversized"
BUDGET = 11_200


async def test_catchup_truncates_an_oversized_tool_call_instead_of_sending_it(caplog):
    """The segment holding the production-shaped row measures within the
    budget, the truncation warning names it, and no overage warning fires."""
    # Production shape: user asks, assistant answers with one huge call.
    rows = [
        {"role": "user", "content": "write the report out to disk"},
        {"role": "assistant", "content": "", "reasoning_content": "x" * 54,
         "tool_calls": [{"id": "c1", "type": "function", "function": {
             "name": "write_file",
             "arguments": json.dumps({"path": "r.md", "text": "data " * 70_000}),
         }}]},
    ]
    db = await aiosqlite.connect(":memory:")
    try:
        await db.execute(
            "CREATE TABLE message_log (id INTEGER PRIMARY KEY, channel_id TEXT, "
            "message TEXT, timestamp REAL, message_type TEXT)"
        )
        await db.executemany(
            "INSERT INTO message_log VALUES (?, ?, ?, ?, 'message')",
            [(i, CHANNEL, json.dumps(m), float(i)) for i, m in enumerate(rows, start=1)],
        )
        inputs = ViewInputs("IDENTITY", BUDGET, "system", strip_reasoning_content, b"k" * 32)
        with caplog.at_level(logging.WARNING, logger=logger.name):
            segments = await build_catchup_segments(
                db, CHANNEL, 0, len(rows), inputs, DEFAULT_CONSOLIDATION_FORK_PROMPT
            )
    finally:
        await db.close()

    # Every request fits; the warnings say truncated, not "sending it anyway".
    assert all(_measured_size(s.messages) <= BUDGET for s in segments)
    text = caplog.text
    assert "truncated message id 2" in text
    assert "sending it anyway" not in text
