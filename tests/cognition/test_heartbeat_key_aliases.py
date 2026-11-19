"""#1079 AC7, AC8: the retired heartbeat.* keys still work for one release.

Each old key is read by its new owner (memory or goals) through
corvidae_cognition.config_keys; these tests go through the plugins' on_init
and read the effect off the beat section each plugin contributes.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from corvidae.hooks import create_plugin_manager
from corvidae.llm_plugin import LLMPlugin
from corvidae_cognition.goals import GoalsPlugin
from corvidae_cognition.memory import MemoryPlugin
from helpers import with_chat_llm


async def _observe(kind: str, config: dict):
    """Init the owning plugin on ``config`` and return the setting's effect:
    the limit its beat section asks for, or the instruction it ends with."""
    if kind == "memory_limit":
        pm = create_plugin_manager()
        pm.register(LLMPlugin(), name="llm")
        plugin = MemoryPlugin()
        await plugin.on_init(pm=pm, config=with_chat_llm(config))
        seen: list[int] = []

        async def recent(limit):
            seen.append(limit)
            return []

        plugin.recent = recent
        await plugin.collect_heartbeat_prompts(channel=SimpleNamespace(id="x"))
        return seen[0]
    plugin = GoalsPlugin()
    await plugin.on_init(MagicMock(), config)
    seen = []

    async def retired(limit):
        seen.append(limit)
        return []

    plugin.retired_goals = retired
    section = await plugin.collect_heartbeat_prompts(channel=None)
    return seen[0] if kind == "retired_limit" else section


#: kind, (old block, old key), (new block, new key), old value, new value,
#: and what _observe returns for each.
CASES = [
    pytest.param(
        "memory_limit", ("heartbeat", "memory_limit"), ("memory", "heartbeat_limit"),
        3, 4, id="memory-limit",
    ),
    pytest.param(
        "retired_limit", ("heartbeat", "retired_goal_limit"),
        ("goals", "heartbeat_retired_limit"), 3, 4, id="retired-goal-limit",
    ),
    pytest.param(
        "instruction", ("heartbeat", "goals_instruction"),
        ("goals", "heartbeat_instruction"), "OLD PICK.", "NEW PICK.", id="goals-instruction",
    ),
]


def _cfg(*pairs) -> dict:
    """Build a config dict from ((block, key), value) pairs."""
    config: dict = {}
    for (block, key), value in pairs:
        config.setdefault(block, {})[key] = value
    return config


def _effect(kind: str, observed, value):
    """The observed effect matches ``value`` (an instruction is the section's tail)."""
    return observed.endswith(value) if kind == "instruction" else observed == value


@pytest.mark.parametrize("kind, old, new, old_value, new_value", CASES)
async def test_old_key_still_takes_effect_and_warns_naming_the_new_key(
    kind, old, new, old_value, new_value, caplog
):
    """AC7: the old key sets the value, and a WARNING names both the old key
    and the key to move it to."""
    with caplog.at_level(logging.WARNING):
        observed = await _observe(kind, _cfg((old, old_value)))
    assert _effect(kind, observed, old_value)
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(f"{old[0]}.{old[1]}" in m and f"{new[0]}.{new[1]}" in m for m in warnings)


@pytest.mark.parametrize("kind, old, new, old_value, new_value", CASES)
async def test_old_and_new_key_together_fail_startup_naming_both(
    kind, old, new, old_value, new_value
):
    """AC8: the same setting under both keys fails startup; the message names
    both keys and says which to remove."""
    with pytest.raises(ValueError) as exc:
        await _observe(kind, _cfg((old, old_value), (new, new_value)))
    message = str(exc.value)
    assert f"{old[0]}.{old[1]}" in message
    assert f"{new[0]}.{new[1]}" in message
    assert "remove" in message
    assert "agent.yaml" in message


@pytest.mark.parametrize("kind, old, new, old_value, new_value", CASES)
async def test_old_key_set_to_null_counts_as_absent(
    kind, old, new, old_value, new_value, caplog
):
    """AC8 edge: an old key set to null alongside the new key is absent, so it
    neither fails startup nor warns, and the new key's value wins."""
    with caplog.at_level(logging.WARNING):
        observed = await _observe(kind, _cfg((old, None), (new, new_value)))
    assert _effect(kind, observed, new_value)
    assert not [r for r in caplog.records if f"{old[0]}.{old[1]}" in r.getMessage()]
