"""A size target for consolidation entries (issue #686).

R1 the instruction asks for one entry per topic within a word target; R2 the
target is one config key read by both the instruction and the overrun check;
R3 an over-target entry is stored in full and warned about; R4 the concrete-
detail clauses survive; R5 a new target changes the stamp hash; R6 raw records
never warn; R8 an operator override gets the target too.
"""

import logging

import pytest
from corvidae_cognition.memory import (
    DEFAULT_CONSOLIDATION_FORK_PROMPT,
    DEFAULT_ENTRY_TARGET_TOKENS,
    RecordStamp,
    consolidation_instruction,
    entry_size_sentence,
    resolve_entry_target_tokens,
)
from helpers import FakeForkAgent, window_of

from corvidae.context import count_tokens

from .test_memory_consolidation import (
    EMBED_CONFIG,
    _SequenceChatClient,
    build_consolidation_env,
    drive_compaction_consolidation,
    drive_idle_consolidation,
    entries_reply,
    fetch_memories,
    seed_dialog,
)
from .test_memory_fork import DIALOG, _compaction, _Registry, _use
from .test_memory_stamps import config_with, prompt_text, stamps_of


def narrative_of(words: int) -> str:
    """A narrative of ``words`` common short words (about one token each)."""
    cycle = ["alpha", "beta", "gamma", "delta"]
    return " ".join(cycle[i % 4] for i in range(words))


def overrun_records(caplog) -> list[logging.LogRecord]:
    """WARNING records from the memory plugin that talk about the entry target."""
    return [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING
        and r.name == "corvidae.memory"
        and "target" in r.getMessage()
    ]


def overrun_warnings(caplog) -> list[str]:
    """The texts of :func:`overrun_records`."""
    return [r.getMessage() for r in overrun_records(caplog)]


async def fork_env(config, tmp_path, name="s.db"):
    """(memory, channel, db, ids): a plugin built from ``config`` with a live window."""
    memory, persistence, _stub, channel, db = await build_consolidation_env(
        config, db_path=str(tmp_path / name)
    )
    memory.pm.register(_Registry(channel), name="registry")
    ids = await seed_dialog(persistence, channel, DIALOG)
    channel.conversation = window_of(channel.id, ids, contents=[c for _, c in DIALOG])
    return memory, channel, db, ids


async def fork_one_entry(memory, channel, ids, narrative):
    """Run one compaction fork whose reply is a single entry of ``narrative``."""
    _use(memory, FakeForkAgent(
        {"consolidation"}, text=entries_reply([(narrative, ["t"])]), model="m"
    ))
    await _compaction(memory, channel, ids[:3], cut=3)


# ---------------------------------------------------------------------------
# R1: the instruction asks for one topic per entry within a word target
# ---------------------------------------------------------------------------


def test_default_instruction_asks_one_topic_per_entry_and_words():
    """R1: the composed default names the target in words and wants one topic
    per entry; the old 'a paragraph at most' wording is gone."""
    text = consolidation_instruction(DEFAULT_CONSOLIDATION_FORK_PROMPT, 200)

    assert "150 words" in text
    assert "one entry for each distinct stretch or topic" in " ".join(text.split())
    assert "a paragraph at most" not in DEFAULT_CONSOLIDATION_FORK_PROMPT


# ---------------------------------------------------------------------------
# R2: one config key drives instruction and threshold
# ---------------------------------------------------------------------------


def test_entry_target_default_is_200():
    """R2: absent key resolves to the default 200."""
    assert resolve_entry_target_tokens({"memory": {}}) == DEFAULT_ENTRY_TARGET_TOKENS == 200


def test_entry_target_set_changes_the_sentence():
    """R2: a configured 100 resolves to 100 and states 75 words."""
    assert resolve_entry_target_tokens({"memory": {"entry_target_tokens": 100}}) == 100
    assert "75 words" in entry_size_sentence(100)


@pytest.mark.parametrize("bad", [0, -5, True, "200", 1.5])
def test_entry_target_rejects_non_positive_integers(bad):
    """R2: the error names the key, the rejected value and what is accepted."""
    with pytest.raises(ValueError) as err:
        resolve_entry_target_tokens({"memory": {"entry_target_tokens": bad}})

    message = str(err.value)
    assert "memory.entry_target_tokens" in message
    assert repr(bad) in message
    assert "positive integer" in message
    # The remedy: the default and where to fix it.
    assert "default 200" in message
    assert "agent.yaml" in message


async def test_bad_entry_target_fails_plugin_startup():
    """R2: a bad key fails MemoryPlugin startup, naming the key; the plugin
    does not fall back to the default."""
    from corvidae_cognition.memory import MemoryPlugin

    from corvidae.hooks import create_plugin_manager

    memory = MemoryPlugin()
    config = config_with(EMBED_CONFIG, entry_target_tokens=0)
    with pytest.raises(ValueError) as err:
        await memory.on_init(pm=create_plugin_manager(), config=config)
    assert "memory.entry_target_tokens" in str(err.value)


async def test_configured_target_drives_instruction_and_threshold(tmp_path, caplog):
    """R2: target 100 sends '75 words' and warns for an entry between 100 and
    the default 200 tokens."""
    config = config_with(EMBED_CONFIG, entry_target_tokens=100)
    memory, channel, db, ids = await fork_env(config, tmp_path)
    try:
        narrative = narrative_of(130)
        # Over the configured 100 but under the default 200.
        assert 100 < count_tokens(narrative) <= 200
        agent = _use(memory, FakeForkAgent(
            {"consolidation"}, text=entries_reply([(narrative, ["t"])]), model="m"
        ))
        with caplog.at_level(logging.WARNING, logger="corvidae.memory"):
            await _compaction(memory, channel, ids[:3], cut=3)

        assert "75 words" in agent.calls[0]["instruction"]
        assert len(overrun_warnings(caplog)) == 1
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# R3: an over-target entry is stored in full and warned about, once
# ---------------------------------------------------------------------------


async def test_over_target_entry_is_stored_in_full_with_one_warning(tmp_path, caplog):
    """R3: full text stored; one warning naming channel, memory id, token
    count, target, mode fork and the prompt hash."""
    memory, channel, db, ids = await fork_env(EMBED_CONFIG, tmp_path)
    try:
        narrative = narrative_of(400)
        tokens = count_tokens(narrative)
        assert tokens > 200
        with caplog.at_level(logging.WARNING, logger="corvidae.memory"):
            await fork_one_entry(memory, channel, ids, narrative)

        [row] = await fetch_memories(db)
        # Stored unchanged: no truncation.
        assert row["summary"] == narrative
        [(_, stored_hash, _mode, _model)] = await stamps_of(db)
        [record] = overrun_records(caplog)
        text = record.getMessage()
        # The human-readable line names each identifier.
        for needle in (channel.id, "mode fork", stored_hash):
            assert needle in text
        for number in (f"memory {row['id']}", f"{tokens} tokens", "target 200"):
            assert number in text, (number, text)
        # The same values ride in extra for log search.
        assert (record.channel_id, record.memory_id, record.tokens, record.target,
                record.mode, record.prompt_hash) == (
            channel.id, row["id"], tokens, 200, "fork", stored_hash)
    finally:
        await db.close()


@pytest.mark.parametrize("words,expected", [(200, 0), (201, 1)])
async def test_warning_boundary_is_the_target(tmp_path, caplog, words, expected):
    """R3/R7: exactly at the target is no warning; one token over is one
    warning (no tolerance margin)."""
    memory, channel, db, ids = await fork_env(EMBED_CONFIG, tmp_path)
    try:
        narrative = narrative_of(words)
        assert count_tokens(narrative) == words
        with caplog.at_level(logging.WARNING, logger="corvidae.memory"):
            await fork_one_entry(memory, channel, ids, narrative)

        assert len(await fetch_memories(db)) == 1
        assert len(overrun_warnings(caplog)) == expected
    finally:
        await db.close()


async def test_only_the_over_target_entry_of_several_warns(tmp_path, caplog):
    """R3: in a multi-entry consolidation, the one over-target entry warns,
    naming its own record id; its siblings do not."""
    memory, channel, db, ids = await fork_env(EMBED_CONFIG, tmp_path)
    try:
        long_text = narrative_of(400)
        _use(memory, FakeForkAgent({"consolidation"}, text=entries_reply([
            ("first short entry about tea", ["a"]),
            (long_text, ["b"]),
            ("third short entry about bikes", ["c"]),
        ]), model="m"))
        with caplog.at_level(logging.WARNING, logger="corvidae.memory"):
            await _compaction(memory, channel, ids[:3], cut=3)

        rows = await fetch_memories(db)
        [long_row] = [r for r in rows if r["summary"] == long_text]
        [record] = overrun_records(caplog)
        assert record.memory_id == long_row["id"]
    finally:
        await db.close()


async def test_standalone_record_over_target_warns(tmp_path, caplog):
    """R3/R6: a standalone (catch-up view) record is written from the
    instruction too, so an over-target entry warns with mode standalone."""
    config = config_with(EMBED_CONFIG, entry_target_tokens=5)
    memory, persistence, _stub, channel, db = await build_consolidation_env(
        config, db_path=str(tmp_path / "s.db")
    )
    try:
        with caplog.at_level(logging.WARNING, logger="corvidae.memory"):
            await drive_idle_consolidation(memory, persistence, channel, db)

        [(_, _hash, mode, _model)] = await stamps_of(db)
        assert mode == "standalone"
        records = overrun_records(caplog)
        assert records and all(r.mode == "standalone" for r in records)
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# R4: the concrete-detail clauses stay
# ---------------------------------------------------------------------------


def test_default_prompt_keeps_detail_clauses():
    """R4: shorter does not mean less specific."""
    flat = " ".join(DEFAULT_CONSOLIDATION_FORK_PROMPT.split())

    assert "names, numbers, dates, identifiers, preferences, commitments, corrections" in flat
    assert "who expressed each" in flat
    assert "Keep what you were told apart from what you inferred or speculated" in flat


# ---------------------------------------------------------------------------
# R5: a new target is a new stamp hash
# ---------------------------------------------------------------------------


async def test_different_targets_stamp_different_hashes(tmp_path):
    """R5: targets 200 and 100 give different stored prompt hashes."""
    hashes = []
    for target in (200, 100):
        config = config_with(EMBED_CONFIG, entry_target_tokens=target)
        memory, persistence, _stub, channel, db = await build_consolidation_env(
            config, db_path=str(tmp_path / f"{target}.db")
        )
        try:
            await drive_idle_consolidation(memory, persistence, channel, db)
            [(_, stored, _mode, _model)] = await stamps_of(db)
            # The plugin's own stamp agrees with what was stored.
            assert stored == RecordStamp(memory._consolidation_fork_prompt, "standalone", None).hash
            hashes.append(stored)
        finally:
            await db.close()

    assert hashes[0] != hashes[1]


# ---------------------------------------------------------------------------
# R6: raw records never warn
# ---------------------------------------------------------------------------


async def test_raw_record_over_target_logs_no_warning(tmp_path, caplog):
    """R6: a raw-transcript fallback record longer than the target is the
    degraded path, not a model ignoring the target, so it does not warn."""
    config = config_with(EMBED_CONFIG, entry_target_tokens=5)
    memory, persistence, _stub, channel, db = await build_consolidation_env(
        config, db_path=str(tmp_path / "s.db")
    )
    try:
        memory._llm()._clients["main"] = _SequenceChatClient(["nope", "still not json"])
        with caplog.at_level(logging.WARNING, logger="corvidae.memory"):
            await drive_compaction_consolidation(memory, persistence, channel)

        [(_, _hash, mode, _model)] = await stamps_of(db)
        assert mode == "raw"
        [row] = await fetch_memories(db)
        assert count_tokens(row["summary"]) > 5
        assert overrun_warnings(caplog) == []
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# R8: an override gets the target too
# ---------------------------------------------------------------------------


async def test_override_gets_size_sentence_stamped_and_sent(tmp_path):
    """R8: the stored instruction ends with the size sentence, and the sent
    text puts it immediately before the range sentence."""
    config = config_with(EMBED_CONFIG, consolidation_fork_prompt="CUSTOM")
    memory, channel, db, ids = await fork_env(config, tmp_path)
    try:
        agent = _use(memory, FakeForkAgent(
            {"consolidation"}, text=entries_reply([("short entry", ["t"])]), model="m"
        ))
        await _compaction(memory, channel, ids[:3], cut=3)

        [(_, stored_hash, _mode, _model)] = await stamps_of(db)
        stored = await prompt_text(db, stored_hash)
        assert stored.startswith("CUSTOM")
        assert stored.endswith(entry_size_sentence(200))
        # Size sentence, blank line, then the range sentence.
        assert entry_size_sentence(200) + "\n\nRecord the" in agent.calls[0]["instruction"]
    finally:
        await db.close()
