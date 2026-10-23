"""Tests for the harness memory_role profile (harness/driver.py).

The probe checks that the model still answers the person when the memory
block is the last "user" message it sees (agent.mid_conversation_system_role:
user). The live run needs llama-server; these tests cover the pieces that
don't: the seeded memory row is one real retrieval finds, the profile
template turns on what the probe needs, and the verdict logic.
"""
from pathlib import Path

import aiosqlite
import pytest
import yaml

from corvidae.hooks import create_plugin_manager
from corvidae.llm_plugin import LLMPlugin
from corvidae.memory import MemoryPlugin
from corvidae.persistence import PersistencePlugin, init_db

from harness.driver import (
    MEMROLE_CHANNEL,
    MEMROLE_FACT,
    MEMROLE_QUERY,
    PROFILES,
    memory_role_verdict,
    seed_memory_row,
)

REPO = Path(__file__).resolve().parent.parent


def _render(name: str) -> dict:
    """Load harness/<name>.yaml.in as run.sh would render it."""
    text = (REPO / "harness" / f"{name}.yaml.in").read_text()
    return yaml.safe_load(text.replace("@STATE_DIR@", "/state"))


# The profile's own llm and memory sections, so retrieval here runs with the
# settings the live probe does (FTS-only, band_on: similarity).
_MEMROLE = _render("buster-memrole")
FTS_ONLY_CONFIG = {"llm": _MEMROLE["llm"], "memory": _MEMROLE["memory"]}


async def _memory_on(db_path: Path) -> tuple[MemoryPlugin, aiosqlite.Connection]:
    """A MemoryPlugin with no embedding role (FTS-only, as in the profile)."""
    db = await aiosqlite.connect(db_path)
    await init_db(db)
    pm = create_plugin_manager()
    persistence = PersistencePlugin()
    persistence.db = db
    pm.register(persistence, name="persistence")
    llm = LLMPlugin()
    pm.register(llm, name="llm")
    await llm.on_init(pm=pm, config=FTS_ONLY_CONFIG)
    memory = MemoryPlugin()
    pm.register(memory, name="memory")
    await memory.on_init(pm=pm, config=FTS_ONLY_CONFIG)
    await memory.on_start(config=FTS_ONLY_CONFIG)
    return memory, db


class TestSeedMemoryRow:
    async def test_seeded_row_is_retrieved_for_the_probe_query(self, tmp_path):
        """The driver writes the row with stdlib sqlite3 while the daemon
        holds the DB; real FTS retrieval must find it for the probe query,
        or the probe would pass without a memory block ever being shown."""
        path = tmp_path / "sessions.db"
        memory, db = await _memory_on(path)
        try:
            seed_memory_row(path, f"irc:{MEMROLE_CHANNEL}", MEMROLE_FACT)
            candidates, degraded = await memory.retrieve(
                f"irc:{MEMROLE_CHANNEL}", MEMROLE_QUERY
            )
        finally:
            await db.close()
        assert degraded is True  # no embedding role: keyword retrieval
        assert [c["summary"] for c in candidates] == [MEMROLE_FACT]
        assert candidates[0]["band"] != "weak"  # admitted, not dropped


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
