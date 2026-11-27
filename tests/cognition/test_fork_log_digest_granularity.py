"""A block digest describes the rendered block, not just its text (#1204).

The bug this file was written against is closed: `digest_of` hashes the whole
block (role, content, tool_calls) and `record()` compares a supplied list
against the rendered blocks, refusing a mismatch. Both checks were reviewed
live at `e341e04` — narrowing `digest_of` to content-only turns five of these
tests red, which is the guard that matters.

What the file now pins, one test per shape: a content-only digest list is
refused for all four block shapes, and whole-block digests separate two sends
that differ only in a role or a tool call.

`context.py:433` (`mid_conversation_system_role`) renders non-leading system
messages under a configured role, which is a role-only difference at constant
content. No frequency is claimed: 0 message contents in 24h of the reviewer's
store were shared across two roles (probe 12:10:30Z), and system prompts are
not in `message_log`, so that probe cannot see this path either.
"""

from __future__ import annotations

import pytest

from tests.cognition.fork_log_store import close_log, open_log
from corvidae_cognition.fork_log import ForkLog, digest_of, prompt_sha256

pytestmark = pytest.mark.timeout(30)

# Same content, different send. A role flip and a tool-call are the two shapes
# this codebase produces.
SYSTEM_ROLE_FLIP = [
    {"role": "system", "content": "frame"},
    {"role": "user", "content": "same words"},
]
SYSTEM_ROLE_FLIPPED = [
    {"role": "system", "content": "frame"},
    {"role": "assistant", "content": "same words"},
]
TOOL_CALL_ONLY = [
    {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "name": "shell"}]},
]
NO_TOOL_CALL = [{"role": "assistant", "content": None}]


@pytest.fixture
async def log(tmp_path):
    log = await open_log(tmp_path / "sessions.db")
    await log.open()
    try:
        yield log
    finally:
        await close_log(log)


@pytest.mark.parametrize(
    "blocks",
    [SYSTEM_ROLE_FLIP, SYSTEM_ROLE_FLIPPED, TOOL_CALL_ONLY, NO_TOOL_CALL],
    ids=["role-flip", "role-flipped", "tool-call-only", "no-tool-call"],
)
async def test_a_content_only_digest_list_is_refused(log, blocks):
    """The finding: content alone cannot identify the send, so it is not taken.

    ``record()`` compares the list against the rendered blocks and refuses a
    mismatch, which is what stops a row describing a send that never happened.
    """
    with pytest.raises(ValueError, match="do not describe"):
        await log.record(
            stage="critique",
            channel_id="internal:test",
            messages=blocks,
            digests=[digest_of(block.get("content") or "") for block in blocks],
        )


@pytest.mark.parametrize(
    "blocks",
    [SYSTEM_ROLE_FLIP, SYSTEM_ROLE_FLIPPED, TOOL_CALL_ONLY, NO_TOOL_CALL],
    ids=["role-flip", "role-flipped", "tool-call-only", "no-tool-call"],
)
async def test_whole_block_digests_identify_the_send(log, blocks):
    fork_id = await log.record(
        stage="critique",
        channel_id="internal:test",
        messages=blocks,
        digests=[digest_of(block) for block in blocks],
    )
    chain = await log.resolve_chain(fork_id)
    assert chain == [digest_of(block) for block in blocks]


async def test_two_sends_differing_only_in_role_do_not_share_a_chain(log):
    left = await log.record(
        stage="critique",
        channel_id="internal:test",
        messages=SYSTEM_ROLE_FLIP,
        digests=[digest_of(b) for b in SYSTEM_ROLE_FLIP],
    )
    right = await log.record(
        stage="critique",
        channel_id="internal:test",
        messages=SYSTEM_ROLE_FLIPPED,
        digests=[digest_of(b) for b in SYSTEM_ROLE_FLIPPED],
    )
    assert await log.resolve_chain(left) != await log.resolve_chain(right)
