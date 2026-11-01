"""Memory records stamped with the instructions that produced them (issue #299).

Covers the write side: R1 (every consolidation call stamps its record with the
instruction text, mode and model), R2 (write_record rows are mode ``direct``),
R3 (a pre-change database gains the columns without any row being rewritten)
and the config guard for non-string prompts.

Red phase: these fail today because ``memory`` has no ``prompt_hash``/``mode``/
``model`` columns, no ``memory_prompt`` table and ``corvidae_cognition.memory``
has no ``RecordStamp``/``DIRECT_STAMP``. New names are imported inside the test
bodies so the rest of the module (and the other test files) keeps collecting.
"""

import asyncio
import copy
import hashlib
import json
import time

import aiosqlite
import pytest

from corvidae_cognition.memory import MemoryPlugin

from helpers import FakeForkAgent, window_of
from .test_goals import add_message, ctx_for, env  # noqa: F401  (fixture reused)
from .test_goals import rows as goals_rows
from .test_memory_consolidation import (
    CONSOLIDATION_JSON,
    EMBED_CONFIG,
    StubClient,
    _SequenceChatClient,
    build_consolidation_env,
    drive_compaction_consolidation,
    drive_idle_consolidation,
    seed_dialog,
)
from .test_memory_dedup import EMBED_CONFIG_DEDUP, build_dedup_env
from .test_memory_dedup import seed_dialog as dedup_seed_dialog
from .test_memory_fork import DIALOG, FORK_JSON, _compaction, _Registry

# The memory table exactly as it was before issue #299 (no stamp columns).
PRE_CHANGE_MEMORY_DDL = """
CREATE TABLE memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id TEXT NOT NULL,
    created_at REAL NOT NULL,
    summary TEXT NOT NULL,
    importance REAL NOT NULL,
    valence REAL,
    topic_tags TEXT,
    participants TEXT,
    msg_id_start INTEGER NOT NULL,
    msg_id_end INTEGER NOT NULL,
    retrieval_count INTEGER NOT NULL DEFAULT 0,
    last_retrieved_at REAL,
    indexed INTEGER NOT NULL DEFAULT 1,
    superseded_by INTEGER,
    redacted INTEGER NOT NULL DEFAULT 0,
    embedded INTEGER NOT NULL DEFAULT 0
)
"""
PRE_CHANGE_COLUMNS = (
    "id, channel_id, created_at, summary, importance, valence, topic_tags, "
    "participants, msg_id_start, msg_id_end, retrieval_count, last_retrieved_at, "
    "indexed, superseded_by, redacted, embedded"
)


def sha(text: str) -> str:
    """The stamp hash the design specifies: sha256 of the UTF-8 instruction text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ModelStub(StubClient):
    """StubClient that also answers request_model, like a real LLMClient."""

    def __init__(self, model: str, chat_text: str = CONSOLIDATION_JSON,
                 sent_model: str | None = None):
        super().__init__(chat_text)
        # ``model`` is the configured name; ``sent_model`` (default: the same)
        # is what a call is actually sent with, e.g. via an extra_body override.
        self.model = model
        self.sent_model = sent_model or model

    def request_model(self, extra_body=None) -> str:
        """Mirror LLMClient.request_model: the extra_body override, else the sent model."""
        return (extra_body or {}).get("model", self.sent_model)


def config_with(base: dict, **memory) -> dict:
    """Copy ``base`` with extra ``memory.*`` keys merged in."""
    config = copy.deepcopy(base)
    config.setdefault("memory", {}).update(memory)
    return config


async def stamps_of(db) -> list[tuple]:
    """(id, prompt_hash, mode, model) of every memory row, oldest first."""
    async with db.execute(
        "SELECT id, prompt_hash, mode, model FROM memory ORDER BY id"
    ) as cursor:
        return await cursor.fetchall()


async def prompt_text(db, prompt_hash: str):
    """The memory_prompt text stored under ``prompt_hash``, or None if absent."""
    async with db.execute(
        "SELECT text FROM memory_prompt WHERE hash = ?", (prompt_hash,)
    ) as cursor:
        row = await cursor.fetchone()
    return row[0] if row else None


# ---------------------------------------------------------------------------
# R1: every consolidation call stamps its record
# ---------------------------------------------------------------------------


async def test_standalone_record_stamped_with_overridden_prompt(tmp_path):
    """R1: a standalone record carries mode, the model sent, and the exact prompt."""
    config = config_with(EMBED_CONFIG, consolidation_prompt="PROMPT-S")
    memory, persistence, stub, channel, db = await build_consolidation_env(
        config, db_path=str(tmp_path / "s.db")
    )
    try:
        # Background role answers with a distinct model name.
        memory._llm()._clients["background"] = ModelStub("configured", sent_model="bg-x")
        await drive_idle_consolidation(memory, persistence, channel, db)

        [(_, prompt_hash, mode, model)] = await stamps_of(db)
        assert mode == "standalone"
        assert model == "bg-x"
        assert prompt_hash == sha("PROMPT-S")
        # The side table resolves the hash to exactly the configured text.
        assert await prompt_text(db, prompt_hash) == "PROMPT-S"
    finally:
        await db.close()


@pytest.fixture
async def fork_env(tmp_path):
    """(memory, channel, db, rowids) with a fork prompt override and a live window."""
    config = config_with(EMBED_CONFIG, consolidation_fork_prompt="PROMPT-F")
    memory, persistence, stub, channel, db = await build_consolidation_env(
        config, db_path=str(tmp_path / "s.db")
    )
    memory.pm.register(_Registry(channel), name="registry")
    ids = await seed_dialog(persistence, channel, DIALOG)
    channel.conversation = window_of(channel.id, ids, contents=[c for _, c in DIALOG])
    yield memory, channel, db, ids
    await db.close()


async def test_fork_record_stamped_with_fork_prompt_only(fork_env):
    """R1: a compaction-fork record stamps the fork prompt, not the per-range sentence."""
    memory, channel, db, ids = fork_env
    memory.pm.register(
        FakeForkAgent({"consolidation"}, text=FORK_JSON, model="main-y"), name="agent"
    )
    await _compaction(memory, channel, ids[:3], cut=3)

    [(_, prompt_hash, mode, model)] = await stamps_of(db)
    assert mode == "fork"
    assert model == "main-y"
    text = await prompt_text(db, prompt_hash)
    # Exactly the fork prompt: no code-generated range sentence (it quotes chat text).
    assert text == "PROMPT-F"
    assert "Record the" not in text


async def test_idle_fork_record_stamped(fork_env):
    """R1: an idle-trigger fork of the live window gets the same fork stamp."""
    memory, channel, db, ids = fork_env
    memory.pm.register(
        FakeForkAgent({"consolidation"}, text=FORK_JSON, model="main-y"), name="agent"
    )
    await memory._consolidate_range(channel.id, ids[-1], "idle")

    [(_, prompt_hash, mode, model)] = await stamps_of(db)
    assert (mode, model) == ("fork", "main-y")
    assert prompt_hash == sha("PROMPT-F")
    assert await prompt_text(db, prompt_hash) == "PROMPT-F"


async def test_raw_transcript_fallback_is_stamped_standalone(tmp_path):
    """R1: the degraded raw-transcript record is still stamped as the standalone call."""
    config = config_with(EMBED_CONFIG, consolidation_prompt="PROMPT-S")
    memory, persistence, stub, channel, db = await build_consolidation_env(
        config, db_path=str(tmp_path / "s.db")
    )
    try:
        # Two malformed replies: the retry fails and the record degrades to the transcript.
        client = _SequenceChatClient(["nope", "still not json"])
        client.model = "configured"
        client.request_model = lambda extra_body=None: "bg-x"
        memory._llm()._clients["background"] = client
        await drive_compaction_consolidation(memory, persistence, channel)

        [(_, prompt_hash, mode, model)] = await stamps_of(db)
        assert (mode, model) == ("standalone", "bg-x")
        assert prompt_hash == sha("PROMPT-S")
    finally:
        await db.close()


async def test_prompt_text_stored_once(tmp_path):
    """R1: the instruction text is stored once however many records share it."""
    config = config_with(EMBED_CONFIG, consolidation_prompt="PROMPT-S")
    memory, persistence, stub, channel, db = await build_consolidation_env(
        config, db_path=str(tmp_path / "s.db")
    )
    try:
        memory._llm()._clients["background"] = ModelStub("bg-x")
        # Three separate ranges consolidated under one prompt.
        for _ in range(3):
            await drive_compaction_consolidation(memory, persistence, channel)

        stamps = await stamps_of(db)
        assert len(stamps) == 3
        assert {s[1] for s in stamps} == {sha("PROMPT-S")}
        async with db.execute("SELECT COUNT(*) FROM memory_prompt") as cursor:
            assert (await cursor.fetchone())[0] == 1
    finally:
        await db.close()


async def test_dedup_survivor_keeps_its_own_stamp():
    """R1: a near-dup merge leaves the superseded row's stamp and stamps the survivor anew."""
    config = config_with(EMBED_CONFIG_DEDUP, consolidation_prompt="P1")
    memory, persistence, stub, channel, db = await build_dedup_env(config)
    try:
        if not memory._vec_available:
            pytest.skip("sqlite-vec unavailable; dedup requires vec similarity")
        stub.chat_text = json.dumps({
            "summary": "I helped kestrel debug the ESP32 weather station wifi dropout",
            "topic_tags": [], "participants": [],
        })
        # Same fixed vector for both summaries: similarity 1.0, so the second merges.
        stub.fixed_embeddings["I helped kestrel"] = [1.0, 0.0, 0.0, 0.0]

        first = await dedup_seed_dialog(persistence, channel, [("user", "a"), ("assistant", "b")])
        await memory.on_compaction(channel=channel, summary_msg={}, retain_count=0, compacted_ids=first)
        await memory.wait_for_background_tasks()

        # The operator edits the prompt between the two consolidations.
        memory._consolidation_prompt = "P2"
        second = await dedup_seed_dialog(persistence, channel, [("user", "c"), ("assistant", "d")])
        await memory.on_compaction(channel=channel, summary_msg={}, retain_count=0, compacted_ids=second)
        await memory.wait_for_background_tasks()

        async with db.execute(
            "SELECT prompt_hash, superseded_by FROM memory ORDER BY id"
        ) as cursor:
            [(old_hash, old_sup), (new_hash, new_sup)] = await cursor.fetchall()
        assert old_sup is not None and new_sup is None
        assert old_hash == sha("P1")
        assert new_hash == sha("P2")
    finally:
        await db.close()


async def test_empty_prompt_is_stored(tmp_path):
    """R1: an empty configured prompt is a real prompt: hashed and stored as ''."""
    config = config_with(EMBED_CONFIG, consolidation_prompt="")
    memory, persistence, stub, channel, db = await build_consolidation_env(
        config, db_path=str(tmp_path / "s.db")
    )
    try:
        await drive_idle_consolidation(memory, persistence, channel, db)

        [(_, prompt_hash, mode, _model)] = await stamps_of(db)
        assert mode == "standalone"
        assert prompt_hash == sha("")
        assert await prompt_text(db, prompt_hash) == ""
    finally:
        await db.close()


@pytest.mark.parametrize("key,value", [
    ("consolidation_prompt", None),
    ("consolidation_fork_prompt", 3),
])
async def test_non_str_prompt_rejected_at_init(key, value):
    """A non-string prompt is refused at startup, naming the key and the value."""
    from corvidae.hooks import create_plugin_manager

    memory = MemoryPlugin()
    pm = create_plugin_manager()
    config = config_with(EMBED_CONFIG, **{key: value})
    with pytest.raises(ValueError) as excinfo:
        await memory.on_init(pm=pm, config=config)
    message = str(excinfo.value)
    # Identifier and the rejected value, per AGENTS.md error guidance.
    assert f"memory.{key}" in message
    assert f"got {value!r}" in message
    assert "string" in message


# ---------------------------------------------------------------------------
# R2: records no consolidation call produced are marked direct
# ---------------------------------------------------------------------------


async def test_write_record_is_direct(tmp_path):
    """R2: write_record rows have mode 'direct' and no hash, model or prompt row."""
    memory, persistence, stub, channel, db = await build_consolidation_env(
        db_path=str(tmp_path / "s.db")
    )
    try:
        ids = await seed_dialog(persistence, channel, [("user", "hi")])
        memory_id = await memory.write_record(
            channel_id=channel.id, summary="I retired a goal.", created_at=1.0,
            importance=1.0, msg_id_start=ids[0], msg_id_end=ids[0],
        )

        [(row_id, prompt_hash, mode, model)] = await stamps_of(db)
        assert row_id == memory_id
        assert mode == "direct"
        assert prompt_hash is None and model is None
        async with db.execute("SELECT COUNT(*) FROM memory_prompt") as cursor:
            assert (await cursor.fetchone())[0] == 0
    finally:
        await db.close()


async def test_retired_goal_record_is_direct(env):  # noqa: F811  (goals fixture)
    """R2: retiring a goal writes its memory record with mode 'direct'."""
    ctx = ctx_for(env.channel)
    await add_message(env, env.channel.id, "kestrel nest evidence")
    await env.tools["add_goal"]("Find the kestrel nest", _ctx=ctx)
    await env.tools["retire_goal"](1, "I concluded it is on Brannan Street, because twigs.", _ctx=ctx)

    (memory_id,) = (await goals_rows(env, "SELECT memory_id FROM goals WHERE id = 1"))[0]
    (mode,) = (await goals_rows(env, "SELECT mode FROM memory WHERE id = ?", (memory_id,)))[0]
    assert mode == "direct"


# ---------------------------------------------------------------------------
# R3: pre-change databases are upgraded in place without rewriting rows
# ---------------------------------------------------------------------------


async def test_pre_change_db_gains_columns_without_rewriting_rows(tmp_path):
    """R3: opening a pre-change DB adds the schema, leaves rows alone, lists them 'unknown'."""
    from click.testing import CliRunner

    path = str(tmp_path / "s.db")
    # Build a pre-change database by hand: old DDL plus two rows.
    async with aiosqlite.connect(path) as raw:
        await raw.execute("PRAGMA journal_mode=wal")
        await raw.execute(PRE_CHANGE_MEMORY_DDL)
        # A current retention run, so on_start's retention job doesn't demote
        # the old-looking seeded rows and change what R3 must leave alone.
        await raw.execute("CREATE TABLE retention_meta (last_run REAL NOT NULL DEFAULT 0)")
        await raw.execute("INSERT INTO retention_meta (last_run) VALUES (?)", (time.time(),))
        for i in (1, 2):
            await raw.execute(
                "INSERT INTO memory (channel_id, created_at, summary, importance, "
                "msg_id_start, msg_id_end, retrieval_count, indexed) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (f"irc:#c{i}", 100.0 + i, f"old summary {i}", 0.5, i, i + 1, i, 1),
            )
        await raw.commit()
        async with raw.execute(f"SELECT {PRE_CHANGE_COLUMNS} FROM memory ORDER BY id") as c:
            snapshot = await c.fetchall()

    memory, persistence, stub, channel, db = await build_consolidation_env(db_path=path)
    try:
        await memory._ensure_schema()

        async with db.execute("PRAGMA table_info(memory)") as c:
            columns = {row[1] for row in await c.fetchall()}
        assert {"prompt_hash", "mode", "model"} <= columns
        async with db.execute(
            "SELECT name FROM sqlite_master WHERE name = 'memory_prompt'"
        ) as c:
            assert await c.fetchone() is not None
        # Old columns are byte-for-byte what they were; the new ones are NULL.
        async with db.execute(f"SELECT {PRE_CHANGE_COLUMNS} FROM memory ORDER BY id") as c:
            assert await c.fetchall() == snapshot
        assert [r[1:] for r in await stamps_of(db)] == [(None, None, None)] * 2
    finally:
        await memory.on_stop()
        await db.close()

    # The listing groups both under 'unknown'.
    from corvidae_cognition.memory_command import memory_command

    result = await asyncio.to_thread(
        CliRunner().invoke, memory_command, ["--db", path, "stamps"]
    )
    assert result.exit_code == 0, (result.output, result.exception)
    unknown = [line for line in result.output.splitlines() if "unknown" in line]
    assert len(unknown) == 1 and "2 records" in unknown[0]
