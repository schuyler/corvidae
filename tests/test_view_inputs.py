"""Agent.view_inputs: everything channel-specific a background view needs,
in one value (#380, #388, I2, D3, D5, D17).

A real Agent and registry; no LLM is involved.
"""

import pytest

from corvidae.channel import load_channel_config
from corvidae.context import ContextWindow

from helpers import build_plugin_and_channel

IDENTITY = "AGENT IDENTITY"
WIDE_REGISTER = "WIDE REGISTER"
AGENT_BUDGET = 4321
WIDE_BUDGET = 777


@pytest.mark.parametrize("case", ["registered", "after_restart", "live_conversation"])
async def test_view_inputs(case):
    """R1/I2: the Agent resolves the channel (live, or not yet in the
    registry after a restart) and returns its system prompt, budget and
    reasoning policy. A live conversation's own prompt wins (D17)."""
    agent, channel, db = await build_plugin_and_channel(agent_defaults={
        "system_prompt": IDENTITY,
        "max_context_tokens": AGENT_BUDGET,
        "reasoning_history": "drop",
    })
    try:
        # A transport-wide entry carries the register and a tighter budget.
        registry = agent._registry
        load_channel_config(
            {"channels": {"test:*": {
                "register_prompt": WIDE_REGISTER, "max_context_tokens": WIDE_BUDGET,
            }}},
            registry,
        )

        if case == "after_restart":
            # Absent from the registry, as after a daemon restart.
            channel_id = "test:never-seen"
            assert registry.get(channel_id) is None
        else:
            channel_id = channel.id
            # The registry entry predates the wide entry: re-matching it is
            # set_wide_entries' job, which load_channel_config did.
            assert registry.get(channel_id) is channel
        if case == "live_conversation":
            conv = ContextWindow(channel_id)
            conv.system_prompt = "OLD IDENTITY\n\nOLD REGISTER"
            channel.conversation = conv
            # The configured identity changed since the conversation began.
            registry.agent_defaults["system_prompt"] = "NEW IDENTITY"

        inputs = await agent.view_inputs(channel_id)

        resolved_channel = registry.get(channel_id)
        if case == "live_conversation":
            # D17: the prompt the agent had, not the rewritten config.
            assert inputs.system_prompt == channel.conversation.system_prompt
            assert inputs.system_prompt == "OLD IDENTITY\n\nOLD REGISTER"
        else:
            # Identity, a blank line, then the wide entry's register.
            assert inputs.system_prompt == f"{IDENTITY}\n\n{WIDE_REGISTER}"
            assert inputs.system_prompt == await agent.channel_system_prompt(resolved_channel)
        assert inputs.max_tokens == registry.resolve_config(resolved_channel)["max_context_tokens"]
        assert inputs.max_tokens == WIDE_BUDGET

        # The bound policy is the channel's: ``drop`` strips reasoning from
        # assistant messages in place.
        messages = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a", "reasoning_content": "thinking"},
        ]
        inputs.apply_reasoning_history(messages)
        assert "reasoning_content" not in messages[1]
        assert messages[1]["content"] == "a"
    finally:
        await db.close()
