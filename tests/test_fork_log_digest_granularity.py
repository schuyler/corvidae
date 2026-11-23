"""A block digest has to describe the rendered block, not just its text (#1204).

`digest_of(text: str)` can only be fed `message["content"]`, so a call site has
no way to put `role` or `tool_calls` into the digest list. Two prompts that
differ only there therefore share a digest list and differ in `prompt_sha256` —
and nothing in the module compares a reconstructed chain against the row's
prompt hash, so no check fires: the row simply describes a send that was not
the one recorded.

The property under test is the one `fork_digest` exists for — that a resolved
chain identifies which prompt was sent — asserted through the public API rather
than against `digest_of`'s signature, so the test flips to passing whether the
fix lands in `digest_of` or at the call sites.

`context.py:433` (`mid_conversation_system_role`) renders non-leading system
messages under a configured role, which is a role-only difference at constant
content. I claim no frequency: 0 message contents in 24h of my own store were
shared across two roles (probe 12:10:30Z), and system prompts are not in
`message_log` at all, so that probe cannot see this path either.
"""

from __future__ import annotations

import pytest

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
    log = ForkLog(str(tmp_path / "sessions.db"))
    await log.open()
    try:
        yield log
    finally:
        await log.close()


async def _chain_for(log: ForkLog, messages) -> list[bytes]:
    """One fork, digests as a caller must compute them: content only.

    The `or ""` is load-bearing and part of the finding: a call site that passes
    `block["content"]` straight through raises on a tool-call block, whose
    content is None (measured: AttributeError from digest_of, 12:18:10Z). A
    caller that avoids the crash collapses it to the empty string, which is what
    the parametrized tool-call case below then pins — two different sends, the
    same digest list.
    """
    fork_id = await log.record(
        stage="critique",
        channel_id="internal:test",
        messages=messages,
        digests=[digest_of(block["content"] or "") for block in messages],
    )
    chain = await log.resolve_chain(fork_id)
    assert chain is not None, "a single-fork chain should always resolve"
    return chain


@pytest.mark.xfail(
    strict=True,
    reason="a digest list over content cannot distinguish two sends that differ in role or tool_calls",
)
@pytest.mark.parametrize(
    "left, right",
    [(SYSTEM_ROLE_FLIP, SYSTEM_ROLE_FLIPPED), (TOOL_CALL_ONLY, NO_TOOL_CALL)],
    ids=["role-flip", "tool-call-only"],
)
async def test_a_resolved_chain_identifies_which_prompt_was_sent(log, left, right):
    assert prompt_sha256(left) != prompt_sha256(right)
    assert await _chain_for(log, left) != await _chain_for(log, right)


async def test_the_prompt_hash_does_notice_both(log):
    assert prompt_sha256(SYSTEM_ROLE_FLIP) != prompt_sha256(SYSTEM_ROLE_FLIPPED)
    assert prompt_sha256(TOOL_CALL_ONLY) != prompt_sha256(NO_TOOL_CALL)
