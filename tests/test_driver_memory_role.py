"""Tests for the harness memory_role profile (harness/driver.py).

The probe checks that the model still answers the person when the memory
block is the last "user" message it sees (agent.mid_conversation_system_role:
user). The live run needs llama-server; these tests cover the pieces that
don't: the probe's warm-up → seed → ask sequence works against a daemon
started the real way, the profile template turns on what the probe needs,
and the verdict logic.
"""
from pathlib import Path

import pytest
import yaml

from corvidae.channel import Channel
from corvidae.context import ContextWindow
from corvidae.funnel import FunnelPlugin
from corvidae.hooks import create_plugin_manager
from corvidae.llm_plugin import LLMPlugin
from corvidae.memory import MemoryPlugin
from corvidae.persistence import PersistencePlugin

from harness.driver import (
    MEMROLE_CHANNEL,
    MEMROLE_FACT,
    MEMROLE_QUERY,
    MEMROLE_WARMUP,
    PROFILES,
    memory_role_verdict,
    memory_schema_ready,
    seed_memory_row,
)

REPO = Path(__file__).resolve().parent.parent


def _render(name: str) -> dict:
    """Load harness/<name>.yaml.in as run.sh would render it."""
    text = (REPO / "harness" / f"{name}.yaml.in").read_text()
    return yaml.safe_load(text.replace("@STATE_DIR@", "/state"))


async def _start_daemon_plugins(state_dir: Path):
    """persistence + llm + funnel + memory, started through pm.ahook.on_start
    with the rendered buster-memrole config, as the daemon starts them.

    on_start hooks run concurrently, so MemoryPlugin may find no database yet
    and defer its schema to first use; the probe has to cope with that.
    Returns (pm, memory, persistence).
    """
    config = _render("buster-memrole")
    config["_base_dir"] = state_dir
    pm = create_plugin_manager()
    persistence = PersistencePlugin()
    pm.register(persistence, name="persistence")
    llm = LLMPlugin()
    pm.register(llm, name="llm")
    funnel = FunnelPlugin()
    pm.register(funnel, name="funnel")
    memory = MemoryPlugin()
    pm.register(memory, name="memory")
    for plugin in (llm, funnel, memory):
        await plugin.on_init(pm=pm, config=config)
    await pm.ahook.on_start(config=config)
    return pm, memory, persistence


async def _turn(memory: MemoryPlugin, channel: Channel, text: str) -> None:
    """The turn-loop entry: append the user message, fire retrieval."""
    channel.conversation.append({"role": "user", "content": text})
    await memory.before_agent_turn(channel=channel, correlation_id=None, meta={})


class TestProbeSequence:
    async def test_warmup_seed_ask_puts_memory_block_last(self, tmp_path):
        """The probe's sequence against a really-started daemon: a warm-up
        turn makes the memory schema exist, the driver's seeded row is
        retrieved for the probe question, and with role: user the built
        prompt ends on the memory block as a user message."""
        pm, memory, persistence = await _start_daemon_plugins(tmp_path)
        db_path = tmp_path / "sessions.db"
        channel = Channel(transport="irc", scope=MEMROLE_CHANNEL)
        channel.conversation = ContextWindow(channel.id)
        channel.conversation.system_prompt = "You are Buster."
        try:
            await _turn(memory, channel, MEMROLE_WARMUP)
            assert memory_schema_ready(db_path)

            seed_memory_row(db_path, channel.id, MEMROLE_FACT)
            await _turn(memory, channel, MEMROLE_QUERY)
            prompt = channel.conversation.build_prompt(mid_conversation_system_role="user")
        finally:
            # Closes the DB and the LLM client sessions on_start opened.
            await pm.ahook.on_stop()

        assert prompt[-2] == {"role": "user", "content": MEMROLE_QUERY}
        assert prompt[-1]["role"] == "user"
        assert MEMROLE_FACT in prompt[-1]["content"]
        assert "[weak]" not in prompt[-1]["content"]


class TestProfileTemplate:
    def test_memrole_profile_enables_what_the_probe_needs(self):
        cfg = _render("buster-memrole")
        disabled = set(cfg.get("plugins", {}).get("disabled", []))
        assert {"memory", "funnel"}.isdisjoint(disabled)
        assert cfg["agent"]["mid_conversation_system_role"] == "user"
        assert "embedding" not in cfg["llm"]  # FTS-only, no embedding server
        assert MEMROLE_CHANNEL in cfg["irc"]["channels"]

    def test_default_profile_stays_minimal(self):
        cfg = _render("buster")
        assert {"memory", "funnel"} <= set(cfg["plugins"]["disabled"])

    def test_every_profile_has_a_template_and_channels(self):
        for name, (probes, channels) in PROFILES.items():
            assert (REPO / "harness" / f"{name}.yaml.in").is_file()
            assert probes and channels


class TestMemoryRoleVerdict:
    def test_pass(self):
        passed, _ = memory_role_verdict(answered=True, recalled=True, admitted=1)
        assert passed

    @pytest.mark.parametrize(
        "answered, recalled, admitted, needle",
        [
            (False, True, 1, "did not answer"),
            (True, False, 1, "did not recall"),
            (True, True, 0, "no memory block"),
            (True, True, None, "no memory block"),
        ],
    )
    def test_each_failure_is_named(self, answered, recalled, admitted, needle):
        passed, details = memory_role_verdict(
            answered=answered, recalled=recalled, admitted=admitted
        )
        assert not passed
        assert needle in details
