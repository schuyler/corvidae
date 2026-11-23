"""Digest granularity: a block digest has to describe the rendered block, not
just its text (#1204 review follow-up).

`digest_of(text)` hashes a string, so a call site can only hand it
`message["content"]`. The chain it feeds is a list of rendered *messages*,
where `role` and `tool_calls` are part of what was sent — and `context.py`'s
`mid_conversation_system_role` makes a role-only difference a real occurrence
in this codebase, not a hypothetical. Two prompts differing only in a role then
yield identical digest lists while `prompt_sha256` differs, so a reconstructed
chain can describe a prompt whose hash does not match the row's: exactly the
mismatch this table exists to catch.

Test 1 is strict-xfail: it fails against the current implementation (the
function cannot take a rendered block at all) and flips to XPASS when
`digest_of` accepts one. Tests 2 and 3 pin the asymmetry that makes it a
correctness finding rather than a nit.
"""

import pytest

from corvidae_cognition.fork_log import digest_of, prompt_sha256

USER_X = [{"role": "system", "content": "S"}, {"role": "user", "content": "x"}]
ASSISTANT_X = [{"role": "system", "content": "S"}, {"role": "assistant", "content": "x"}]


@pytest.mark.xfail(
    strict=True,
    reason="digest_of(text) cannot take a rendered block, so role and tool_calls are invisible to the chain",
)
def test_a_block_digest_distinguishes_a_role_change():
    left = [digest_of(block) for block in USER_X]
    right = [digest_of(block) for block in ASSISTANT_X]
    assert left != right


def test_hashing_content_alone_collides_across_that_role_change():
    # What the current signature forces a call site to do, and the collision it
    # produces: identical block lists for two prompts that are not the same send.
    left = [digest_of(block["content"]) for block in USER_X]
    right = [digest_of(block["content"]) for block in ASSISTANT_X]
    assert left == right


def test_the_prompt_hash_does_notice_a_role_change():
    assert prompt_sha256(USER_X) != prompt_sha256(ASSISTANT_X)
