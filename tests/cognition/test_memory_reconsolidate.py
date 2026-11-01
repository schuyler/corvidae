"""Operator CLI ``corvidae memory`` (stamps, reconsolidate) for issue #299.

Covers R2 (direct rows list apart and are never re-consolidated), R4 (list by
stamp), R5 (re-consolidation supersedes with the current stamp), R6 (dry run),
R7 (never embeds, deletes only the superseded vector), R8 (selection), R9
(per-record failure), R10 (operator-only, beside a daemon, error handling) and
R11 (the new record inherits the old one's history).

Red phase: ``corvidae_cognition.memory_command`` does not exist, ``memory`` has
no stamp columns and ``MemoryPlugin`` has no ``reconsolidate_record``. Every new
name is imported inside the test bodies so the module still collects. All LLM
clients are stubs; no test touches the network.
"""

import asyncio
import contextlib
import datetime
import functools
import importlib.metadata
import logging
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import aiosqlite
import pytest
import yaml

from corvidae.channel import Channel
from corvidae.llm_plugin import LLMPlugin
from corvidae.persistence import init_db
from corvidae_cognition.memory import (
    DEFAULT_CONSOLIDATION_PROMPT,
    MemoryPlugin,
)

from .test_memory_consolidation import (
    EMBED_CONFIG,
    build_consolidation_env,
    fetch_watermark,
    seed_dialog,
)
from .test_memory_stamps import (
    PRE_CHANGE_MEMORY_DDL,
    ModelStub,
    config_with,
    sha,
)

OLD = "OLD-PROMPT"
NEW = "NEW-PROMPT"

# Daemon-style config without an embedding role (the vector-free baseline).
NOEMBED_CONFIG = {
    "llm": {
        "main": {"base_url": "http://localhost:8080", "model": "chat"},
        "background": {"base_url": "http://localhost:8080", "model": "bg"},
    },
    "memory": {"idle_consolidate_after": 1800},
}


# ---------------------------------------------------------------------------
# Stubs and helpers
# ---------------------------------------------------------------------------


class ChatStub(ModelStub):
    """Background/main chat client stub for re-consolidation.

    A call is a consolidation call only if its system message is the
    standalone consolidation prompt. Everything else (the RubricPrior's
    request, appraisal) is the importance prior's and gets valid rubric JSON,
    so call counts see exactly the calls re-consolidation itself makes.
    Consolidation calls are recorded in ``attempts`` (including failures) and,
    when they succeed, in ``chat_calls``. ``fail_marker`` fails calls whose
    dialog contains it; ``before_reply`` is a one-shot async hook that can
    mutate the DB mid-call.
    """

    def __init__(self, model: str = "configured-bg", prompt: str = NEW):
        super().__init__(model, sent_model="bg-new")
        self.prompt = prompt
        self.fail_marker: str | None = None
        self.before_reply = None
        self.attempts: list[dict] = []

    @property
    def consolidation_calls(self) -> list[dict]:
        """Successful chat calls whose system message is the consolidation prompt."""
        return [c for c in self.chat_calls if self._is_consolidation(c["messages"])]

    def _is_consolidation(self, messages) -> bool:
        """True when the first system message is the standalone instruction text."""
        system = [m for m in messages if m.get("role") == "system"]
        return bool(system) and system[0].get("content") == self.prompt

    async def start(self):
        """LLMPlugin.on_start starts every client it builds."""

    async def stop(self):
        """LLMPlugin.on_stop stops every client."""

    async def chat(self, messages, tools=None, extra_body=None):
        # The prior's calls: answer with rubric JSON, record nothing.
        if not self._is_consolidation(messages):
            return {"choices": [{"message": {
                "role": "assistant", "content": '{"importance": 0.5}'}}]}
        self.attempts.append({"messages": messages})
        if self.before_reply is not None:
            hook, self.before_reply = self.before_reply, None  # one-shot
            await hook(messages)
        text = " ".join(str(m.get("content")) for m in messages)
        if self.fail_marker and self.fail_marker in text:
            raise RuntimeError("llm down for test")
        return await super().chat(messages, tools, extra_body)


async def make_env(path, *, prompt: str = NEW, embed: bool = False, chat=None):
    """Consolidation env on a WAL file DB whose current standalone prompt is ``prompt``."""
    config = config_with(EMBED_CONFIG if embed else NOEMBED_CONFIG, consolidation_prompt=prompt)
    memory, persistence, embed_stub, channel, db = await build_consolidation_env(
        config, db_path=str(path)
    )
    chat = chat or ChatStub()
    clients = memory._llm()._clients
    clients["main"] = chat
    clients["background"] = chat
    return SimpleNamespace(
        memory=memory, persistence=persistence, embed_stub=embed_stub, chat=chat,
        channel=channel, db=db, path=str(path), config=config,
    )


async def close(env) -> None:
    """Stop memory (closes its write connection) and the persistence connection."""
    await env.memory.on_stop()
    await env.db.close()


async def set_state(env, memory_id: int, **columns) -> None:
    """UPDATE memory columns on the shared connection (as retention or redact would)."""
    sets = ", ".join(f"{name} = ?" for name in columns)
    await env.db.execute(
        f"UPDATE memory SET {sets} WHERE id = ?", (*columns.values(), memory_id)
    )
    await env.db.commit()


async def add_record(env, *, n: int = 1, stamp=None, channel=None, vec=None,
                     created_at: float | None = None, importance: float = 0.5,
                     **state) -> int:
    """Seed a two-message dialog and a consolidated record over it.

    ``stamp`` defaults to OLD/standalone. ``vec`` stores an embedding (so a
    memory_vec row exists); ``state`` columns (indexed, retrieval_count, ...)
    are applied afterwards, as the daemon would have left them.
    """
    from corvidae_cognition.memory import RecordStamp

    channel = channel or env.channel
    ids = await seed_dialog(
        env.persistence, channel, [("user", f"talk {n}"), ("assistant", f"reply {n}")]
    )
    await env.memory._ensure_schema()
    stamp = stamp or RecordStamp(OLD, "standalone", "old-model")
    async with env.memory._db_lock, env.memory._write_txn() as wdb:
        memory_id = await env.memory._insert_record(
            wdb, channel_id=channel.id, summary=f"old summary {n}",
            created_at=created_at if created_at is not None else 1000.0 + n,
            importance=importance, valence=None, topic_tags=[], participants=[],
            msg_id_start=ids[0], msg_id_end=ids[-1], embedding=vec, stamp=stamp,
        )
    if state:
        await set_state(env, memory_id, **state)
    return memory_id


async def add_unstamped(env, *, n: int = 1) -> int:
    """A pre-change style row: raw INSERT leaving prompt_hash and mode NULL."""
    ids = await seed_dialog(
        env.persistence, env.channel, [("user", f"talk {n}"), ("assistant", f"reply {n}")]
    )
    await env.memory._ensure_schema()
    cursor = await env.db.execute(
        "INSERT INTO memory (channel_id, created_at, summary, importance, "
        "msg_id_start, msg_id_end) VALUES (?, ?, ?, ?, ?, ?)",
        (env.channel.id, 500.0 + n, f"unstamped {n}", 0.5, ids[0], ids[-1]),
    )
    await env.db.commit()
    return cursor.lastrowid


async def row_of(env, memory_id: int) -> dict:
    """Every memory column of one row, by name."""
    async with env.db.execute("SELECT * FROM memory WHERE id = ?", (memory_id,)) as cursor:
        row = await cursor.fetchone()
        return dict(zip([d[0] for d in cursor.description], row)) if row else {}


async def run_reconsolidate(env, stamp: str):
    """select_records then reconsolidate through the same API the CLI glue uses."""
    from corvidae_cognition.memory_command import (
        probe_memory_schema,
        reconsolidate,
        select_records,
    )

    schema = await probe_memory_schema(env.db)
    selection = await select_records(env.db, schema, stamp)
    return selection, await reconsolidate(env.memory, selection)


def sql(path: str, query: str, params=()) -> list[tuple]:
    """One-off read on a plain sqlite3 connection (closed on return)."""
    conn = sqlite3.connect(path)
    try:
        return conn.execute(query, params).fetchall()
    finally:
        conn.close()


async def vec_ids(path: str) -> set[int]:
    """memory_ids present in memory_vec, read on a fresh sqlite-vec-enabled connection."""
    async with aiosqlite.connect(path) as conn:
        # Load sqlite-vec directly, never through the class method tests patch.
        try:
            import sqlite_vec

            await conn.enable_load_extension(True)
            await conn.load_extension(sqlite_vec.loadable_path())
        except Exception:
            pytest.skip("sqlite-vec unavailable")
        async with conn.execute("SELECT memory_id FROM memory_vec") as cursor:
            return {r[0] for r in await cursor.fetchall()}


async def invoke(*args: str):
    """Run ``corvidae memory <args>`` through CliRunner, off the event loop's thread."""
    from click.testing import CliRunner
    from corvidae_cognition.memory_command import memory_command

    return await asyncio.to_thread(CliRunner().invoke, memory_command, list(args))


def write_config(tmp_path, **sections) -> str:
    """Write an agent.yaml with a daemon-like llm block and prompt NEW; sections replace keys."""
    config = {
        "llm": {
            "main": {"base_url": "http://localhost:8080", "model": "chat"},
            "background": {"base_url": "http://localhost:8080", "model": "bg"},
        },
        "memory": {"consolidation_prompt": NEW},
    }
    config.update(sections)
    path = tmp_path / "agent.yaml"
    path.write_text(yaml.safe_dump(config))
    return str(path)


def patch_clients(monkeypatch, chat) -> None:
    """Make the real offline wiring build ``chat`` for every llm role (no network)."""
    monkeypatch.setattr(LLMPlugin, "_create_client", lambda self, cfg, role="main": chat)


def patch_offline(monkeypatch, chat, *, prompt: str = NEW) -> None:
    """Replace _open_offline_memory with one yielding an in-loop env on the same file."""
    from corvidae_cognition import memory_command

    @contextlib.asynccontextmanager
    async def fake_open(db_path, config_path):
        env = await make_env(db_path, prompt=prompt, chat=chat)
        try:
            yield env.memory
        finally:
            await close(env)

    monkeypatch.setattr(memory_command, "_open_offline_memory", fake_open)


def assert_clean_exit(result, code: int) -> None:
    """Exit code matches and the failure was a handled exit, not a traceback."""
    assert result.exit_code == code, (result.output, repr(result.exception))
    assert result.exception is None or isinstance(result.exception, SystemExit), repr(result.exception)
    assert "Traceback" not in result.output


async def old_db(tmp_path, count: int = 2, **state) -> tuple[str, list[int]]:
    """A closed WAL file DB with ``count`` OLD-stamped live records on irc:#test."""
    env = await make_env(tmp_path / "s.db")
    ids = [await add_record(env, n=i + 1, **state) for i in range(count)]
    await close(env)
    return env.path, ids


async def make_pre_change_db(path: str, rows: int = 3, *, superseded: tuple[int, ...] = (),
                             partial: bool = False) -> list[int]:
    """A WAL DB written by the pre-change daemon: old memory DDL, real dialog rows.

    ``superseded`` lists 1-based row numbers marked superseded; ``partial``
    adds only prompt_hash (a migration interrupted after its first ALTER).
    """
    ids = []
    async with aiosqlite.connect(path) as raw:
        await init_db(raw)
        await raw.execute("PRAGMA journal_mode=wal")
        await raw.execute(PRE_CHANGE_MEMORY_DDL)
        if partial:
            await raw.execute("ALTER TABLE memory ADD COLUMN prompt_hash TEXT")
        for i in range(1, rows + 1):
            first = None
            for role in ("user", "assistant"):
                cursor = await raw.execute(
                    "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
                    "VALUES (?, ?, ?, 'message')",
                    ("irc:#test", f'{{"role": "{role}", "content": "talk {i}"}}', 100.0 + i),
                )
                first = first or cursor.lastrowid
            cursor = await raw.execute(
                "INSERT INTO memory (channel_id, created_at, summary, importance, "
                "msg_id_start, msg_id_end, superseded_by) VALUES (?, ?, ?, 0.5, ?, ?, ?)",
                ("irc:#test", 200.0 + i, f"pre-change summary {i}", first, first + 1,
                 999 if i in superseded else None),
            )
            ids.append(cursor.lastrowid)
        await raw.commit()
    return ids


@pytest.fixture(autouse=True)
def _restore_logging():
    """The CLI reconfigures logging; put the corvidae logger back after each test."""
    logger = logging.getLogger("corvidae")
    root = logging.getLogger()
    saved = (list(logger.handlers), logger.level, logger.propagate, list(root.handlers), root.level)
    yield
    logger.handlers[:] = saved[0]
    logger.setLevel(saved[1])
    logger.propagate = saved[2]
    root.handlers[:] = saved[3]
    root.setLevel(saved[4])


# ---------------------------------------------------------------------------
# R2: direct records list apart and are never re-consolidated
# ---------------------------------------------------------------------------


async def make_mixed_db(tmp_path) -> dict:
    """One direct row, two pre-change rows and one stamped row, closed on disk."""
    env = await make_env(tmp_path / "s.db")
    ids = await seed_dialog(env.persistence, env.channel, [("user", "goal talk")])
    direct = await env.memory.write_record(
        channel_id=env.channel.id, summary="I retired a goal.", created_at=1.0,
        importance=1.0, msg_id_start=ids[0], msg_id_end=ids[0],
    )
    unknown = [await add_unstamped(env, n=1), await add_unstamped(env, n=2)]
    stamped = await add_record(env, n=3)
    await close(env)
    return {"path": env.path, "direct": direct, "unknown": unknown, "stamped": stamped}


async def test_direct_lists_apart_from_unknown(tmp_path):
    """R2: 'direct' and 'unknown' are separate groups with their own counts."""
    db = await make_mixed_db(tmp_path)

    result = await invoke("--db", db["path"], "stamps")

    assert_clean_exit(result, 0)
    lines = result.output.splitlines()
    direct = [line for line in lines if line.lstrip().startswith("direct")]
    unknown = [line for line in lines if line.lstrip().startswith("unknown")]
    assert len(direct) == 1 and "1 records" in direct[0]
    assert len(unknown) == 1 and "2 records" in unknown[0]


async def test_reconsolidate_never_touches_direct(tmp_path, monkeypatch):
    """R2: no selector reaches a direct row, and reconsolidate_record refuses one outright."""
    from corvidae_cognition.memory import ReconsolidateSkipped

    db = await make_mixed_db(tmp_path)
    chat = ChatStub()
    patch_clients(monkeypatch, chat)
    config = write_config(tmp_path)

    # The 'unknown' selection itself excludes the direct row: a dry run never
    # lists it as something it would re-consolidate.
    dry = await invoke("--db", db["path"], "reconsolidate", "unknown", "--dry-run")
    assert_clean_exit(dry, 0)
    assert f"#{db['direct']} (" not in dry.output

    # 'unknown' rebuilds the two pre-change rows and leaves the direct one alone.
    result = await invoke("--db", db["path"], "reconsolidate", "unknown", "--config", config)
    assert_clean_exit(result, 0)
    assert sql(db["path"], "SELECT superseded_by FROM memory WHERE id = ?", (db["direct"],)) == [(None,)]
    [(start, end)] = sql(
        db["path"], "SELECT msg_id_start, msg_id_end FROM memory WHERE id = ?", (db["direct"],)
    )
    covering = sql(
        db["path"],
        "SELECT id FROM memory WHERE id != ? AND msg_id_start = ? AND msg_id_end = ?",
        (db["direct"], start, end),
    )
    assert covering == []
    assert len(chat.consolidation_calls) == 2

    # Asking for the direct group is refused with the reason and the way forward.
    result = await invoke("--db", db["path"], "reconsolidate", "direct", "--config", config)
    assert result.exit_code != 0
    assert "never re-consolidated" in result.output
    assert "corvidae memory --db" in result.output

    # And the plugin itself skips a direct row before any LLM call.
    env = await make_env(db["path"], chat=ChatStub())
    try:
        with pytest.raises(ReconsolidateSkipped, match="direct"):
            await env.memory.reconsolidate_record(db["direct"])
        assert env.chat.attempts == []
    finally:
        await close(env)


# ---------------------------------------------------------------------------
# R4: list by stamp
# ---------------------------------------------------------------------------


def _epoch(year: int, month: int, day: int) -> float:
    """Noon UTC on a date, so any display timezone shows the same day."""
    return datetime.datetime(year, month, day, 12, tzinfo=datetime.timezone.utc).timestamp()


async def test_stamps_lists_two_stamps_and_unknown(tmp_path):
    """R4: two stamps plus unstamped rows list as three groups with the right details."""
    from corvidae_cognition.memory import RecordStamp

    env = await make_env(tmp_path / "s.db")
    stamp_a = RecordStamp("PROMPT-A", "standalone", "model-a")
    stamp_b = RecordStamp("PROMPT-B", "fork", "model-b")
    await add_record(env, n=1, stamp=stamp_a, channel=Channel("irc", "#a"), created_at=_epoch(2025, 9, 1))
    await add_record(env, n=2, stamp=stamp_a, channel=Channel("irc", "#b"), created_at=_epoch(2025, 10, 30))
    await add_record(env, n=3, stamp=stamp_b, channel=Channel("irc", "#a"), created_at=_epoch(2025, 9, 15))
    for n in (4, 5, 6):
        await add_unstamped(env, n=n)
    await close(env)
    before = Path(env.path).read_bytes()

    result = await invoke("--db", env.path, "stamps")

    assert_clean_exit(result, 0)
    # Exactly three group lines, one per stamp plus the unknown group.
    groups = [line for line in result.output.splitlines() if " records" in line]
    assert len(groups) == 3, result.output
    [line_a] = [line for line in groups if sha("PROMPT-A")[:12] in line]
    for expected in ("2 records", "irc:#a", "irc:#b", "standalone", "model-a",
                     "2025-09-01..2025-10-30"):
        assert expected in line_a, (expected, line_a)
    [line_b] = [line for line in groups if sha("PROMPT-B")[:12] in line]
    assert "1 records" in line_b and "fork" in line_b and "model-b" in line_b
    [unknown] = [line for line in groups if line.lstrip().startswith("unknown")]
    assert "3 records" in unknown
    # Listing is read-only: the main database file is byte-identical.
    assert Path(env.path).read_bytes() == before


async def test_stamps_on_missing_db_names_path(tmp_path):
    """R4: a typo in --db names the path and creates nothing."""
    path = str(tmp_path / "typo.db")

    result = await invoke("--db", path, "stamps")

    assert result.exit_code != 0
    assert f"Database not found: {path}" in result.output
    assert not Path(path).exists()


# ---------------------------------------------------------------------------
# R5: re-consolidation supersedes with the current stamp
# ---------------------------------------------------------------------------


async def test_reconsolidate_supersedes_with_current_stamp(tmp_path):
    """R5: each old record is superseded by a new one stamped NEW; the old row and raw dialog stay."""
    env = await make_env(tmp_path / "s.db")
    try:
        old_ids = [await add_record(env, n=1), await add_record(env, n=2)]
        before = {i: await row_of(env, i) for i in old_ids}
        watermark = await fetch_watermark(env.db, env.channel.id)

        selection, report = await run_reconsolidate(env, sha(OLD))

        assert sorted(i for i, _ in selection.selected) == old_ids
        for old_id in old_ids:
            old = await row_of(env, old_id)
            # Demoted, not deleted: the row is intact and points at its replacement.
            assert old["superseded_by"] is not None and old["indexed"] == 0
            assert old["summary"] == before[old_id]["summary"]
            new = await row_of(env, old["superseded_by"])
            assert new["prompt_hash"] == sha(NEW)
            assert new["mode"] == "standalone"
            # Stamped with the model the call was sent with, not the configured name.
            assert new["model"] == "bg-new"
            # The superseded row's vector flag is cleared.
            assert old["embedded"] == 0
            assert (new["channel_id"], new["msg_id_start"], new["msg_id_end"]) == (
                old["channel_id"], old["msg_id_start"], old["msg_id_end"])
        async with env.db.execute(
            "SELECT text FROM memory_prompt WHERE hash = ?", (sha(NEW),)
        ) as cursor:
            assert (await cursor.fetchone())[0] == NEW
        # The consolidation watermark is never read or advanced.
        assert await fetch_watermark(env.db, env.channel.id) == watermark

        # recall_raw still reaches the superseded record's dialog.
        from corvidae_cognition.memory_tools import MemoryToolsPlugin

        tools_plugin = MemoryToolsPlugin()
        env.memory.pm.register(tools_plugin, name="memory_tools")
        await tools_plugin.on_init(pm=env.memory.pm, config=env.config)
        registered: list = []
        tools_plugin.register_tools(registered)
        ctx = MagicMock()
        ctx.channel = env.channel
        recalled = await {t.name: t.fn for t in registered}["recall_raw"](old_ids[0], _ctx=ctx)
        assert "user: talk 1" in recalled
    finally:
        await close(env)


# ---------------------------------------------------------------------------
# R6: dry run
# ---------------------------------------------------------------------------


async def test_dry_run_is_byte_identical_and_names_records(tmp_path, monkeypatch):
    """R6: --dry-run (no config) names ids and channels, skips counted, file untouched, no LLM."""
    env = await make_env(tmp_path / "s.db")
    live_a = await add_record(env, n=1, channel=Channel("irc", "#a"))
    live_b = await add_record(env, n=2, channel=Channel("irc", "#b"))
    await add_record(env, n=3, channel=Channel("irc", "#a"), superseded_by=live_a)
    await add_record(env, n=4, channel=Channel("irc", "#a"), redacted=1)
    await close(env)
    before = Path(env.path).read_bytes()

    # Starting the LLM plugin or building any client would fail the test; the
    # wraps keeps the pluggy hookimpl marker so the patched on_start would be called.
    def boom(*args, **kwargs):
        raise AssertionError("dry run must not start the LLM plugin or build a client")

    original = LLMPlugin.on_start
    monkeypatch.setattr(LLMPlugin, "on_start", functools.wraps(original)(boom))
    monkeypatch.setattr(LLMPlugin, "_create_client", boom)

    result = await invoke("--db", env.path, "reconsolidate", sha(OLD)[:12], "--dry-run")

    assert_clean_exit(result, 0)
    assert f"#{live_a} (irc:#a)" in result.output
    assert f"#{live_b} (irc:#b)" in result.output
    assert "would skip 1 superseded, 1 redacted" in result.output
    assert "Nothing was changed" in result.output
    # Which -wal/-shm files exist is not compared; the main file's bytes are.
    assert Path(env.path).read_bytes() == before


async def test_dry_run_without_config_lists_would_reconsolidate_count(tmp_path):
    """R6: without --config nothing can be checked against the current prompt, so the
    dry run reports the count (the design specifies no caveat text, so none is asserted)."""
    env = await make_env(tmp_path / "s.db")
    from corvidae_cognition.memory import RecordStamp

    stamp = RecordStamp(DEFAULT_CONSOLIDATION_PROMPT, "standalone", "m")
    for n in (1, 2):
        await add_record(env, n=n, stamp=stamp)
    await close(env)

    result = await invoke(
        "--db", env.path, "reconsolidate", sha(DEFAULT_CONSOLIDATION_PROMPT)[:12], "--dry-run"
    )

    assert_clean_exit(result, 0)
    assert "Would re-consolidate 2 records" in result.output


# ---------------------------------------------------------------------------
# R7: never embeds, deletes only the superseded vector
# ---------------------------------------------------------------------------


async def test_reconsolidated_record_is_unembedded_with_prior(tmp_path):
    """R7: new row is embedded=0 with the prior, no vec row; the old vector is gone; no embed call."""
    env = await make_env(tmp_path / "s.db", embed=True)
    try:
        if not env.memory._vec_available:
            pytest.skip("sqlite-vec unavailable")
        old_id = await add_record(env, vec=[1.0, 0.0, 0.0, 0.0], importance=0.5)
        env.memory.importance_prior.score.return_value = 0.7
        assert old_id in await vec_ids(env.path)

        await run_reconsolidate(env, sha(OLD))

        old = await row_of(env, old_id)
        new = await row_of(env, old["superseded_by"])
        assert new["embedded"] == 0
        # The old row started embedded=1; superseding clears it with its vector.
        assert old["embedded"] == 0
        assert new["importance"] == 0.7  # max(0.5, prior 0.7)
        present = await vec_ids(env.path)
        assert new["id"] not in present and old_id not in present
        # The command never embeds: the encoder stub was not called.
        assert env.embed_stub.embed_calls == []
        # One consolidation call, attributed to this operation.
        [call] = env.chat.consolidation_calls
        assert call["attribution"]["stage"] == "consolidation"
        assert env.chat._is_consolidation(call["messages"])
        assert call["attribution"]["trigger"] == "reconsolidate"
        assert call["attribution"]["mode"] == "standalone"
    finally:
        await close(env)


async def test_cli_never_writes_embedding_meta_or_creates_memory_vec(tmp_path, monkeypatch):
    """R7: a DB built without embeddings stays that way even with llm.embedding in the config."""
    path, _ = await old_db(tmp_path, count=1)
    chat = ChatStub()
    patch_clients(monkeypatch, chat)
    embedding = {"base_url": "http://localhost:8081", "model": "test-embedder", "dimensions": 4}
    config = write_config(tmp_path, llm={
        "main": {"base_url": "http://localhost:8080", "model": "chat"},
        "background": {"base_url": "http://localhost:8080", "model": "bg"},
        "embedding": embedding,
    })

    result = await invoke("--db", path, "reconsolidate", sha(OLD)[:12], "--config", config)

    assert_clean_exit(result, 0)
    assert sql(path, "SELECT COUNT(*) FROM embedding_meta") == [(0,)]
    assert sql(path, "SELECT name FROM sqlite_master WHERE name = 'memory_vec'") == []
    assert sql(path, "SELECT embedded FROM memory WHERE prompt_hash = ?", (sha(NEW),)) == [(0,)]


async def test_vec_load_failure_refuses(tmp_path, monkeypatch):
    """R7: memory_vec exists but sqlite-vec won't load: refuse before any LLM call, rows untouched."""
    env = await make_env(tmp_path / "s.db", embed=True)
    if not env.memory._vec_available:
        await close(env)
        pytest.skip("sqlite-vec unavailable")
    await add_record(env, vec=[1.0, 0.0, 0.0, 0.0])
    await close(env)
    snapshot = sql(env.path, "SELECT * FROM memory ORDER BY id")
    chat = ChatStub()
    patch_clients(monkeypatch, chat)

    async def refuse_load(self, db):
        return False

    monkeypatch.setattr(MemoryPlugin, "_load_vec_extension", refuse_load)

    result = await invoke(
        "--db", env.path, "reconsolidate", sha(OLD)[:12], "--config", write_config(tmp_path)
    )

    assert result.exit_code == 1, (result.output, repr(result.exception))
    for expected in (env.path, "sqlite-vec", "no records were changed", "uv sync --extra cognition"):
        assert expected in result.output
    assert "redact" not in result.output
    assert chat.attempts == []
    assert sql(env.path, "SELECT * FROM memory ORDER BY id") == snapshot


async def test_cli_deletes_superseded_vector(tmp_path, monkeypatch):
    """R7 through the real CLI path: the old vector is deleted, the new row is unembedded."""
    pytest.importorskip("sqlite_vec")
    env = await make_env(tmp_path / "s.db", embed=True)
    if not env.memory._vec_available:
        await close(env)
        pytest.skip("sqlite-vec unavailable")
    old_id = await add_record(env, vec=[1.0, 0.0, 0.0, 0.0])
    await close(env)
    assert old_id in await vec_ids(env.path)
    patch_clients(monkeypatch, ChatStub())

    # Record every connection sqlite-vec is loaded on.
    loaded: list[tuple[int, object]] = []
    original = MemoryPlugin._load_vec_extension

    async def recording(self, db):
        # Store the isolation level now: the connection is closed by the end.
        loaded.append((id(db), db.isolation_level))
        return await original(self, db)

    monkeypatch.setattr(MemoryPlugin, "_load_vec_extension", recording)

    result = await invoke(
        "--db", env.path, "reconsolidate", sha(OLD)[:12], "--config", write_config(tmp_path)
    )

    assert_clean_exit(result, 0)
    calls = list(loaded)
    present = await vec_ids(env.path)
    new_ids = [r[0] for r in sql(env.path, "SELECT id FROM memory WHERE prompt_hash = ?", (sha(NEW),))]
    assert len(new_ids) == 1
    assert old_id not in present and new_ids[0] not in present
    assert sql(env.path, "SELECT embedded FROM memory WHERE id = ?", (new_ids[0],)) == [(0,)]
    # The extension went onto the dedicated write connection (opened with
    # isolation_level=None), never the shared persistence connection.
    assert calls, "sqlite-vec was never loaded, so the vector could not have been deleted"
    assert all(level is None for _, level in calls)


async def test_write_connection_reopen_reloads_vec(tmp_path, monkeypatch):
    """R7: a write connection reopened mid-run gets sqlite-vec again (_load_vec_on_write)."""
    env = await make_env(tmp_path / "s.db", embed=True)
    try:
        if not env.memory._vec_available:
            pytest.skip("sqlite-vec unavailable")
        first = await add_record(env, n=1, vec=[1.0, 0.0, 0.0, 0.0])
        second = await add_record(env, n=2, vec=[0.0, 1.0, 0.0, 0.0])
        # Behave like the CLI process: vectors are loaded only by the flag.
        env.memory._vec_available = False
        env.memory._load_vec_on_write = True
        loaded: list = []
        original = env.memory._load_vec_extension

        async def recording(db):
            loaded.append(db)
            return await original(db)

        monkeypatch.setattr(env.memory, "_load_vec_extension", recording)

        await env.memory.reconsolidate_record(first)
        # Drop the write connection, as _write_txn does after a wedge.
        await env.memory._write_db.close()
        env.memory._write_db = None
        await env.memory.reconsolidate_record(second)

        assert loaded and loaded[-1] is env.memory._write_db
        present = await vec_ids(env.path)
        assert first not in present and second not in present
    finally:
        await close(env)


async def test_no_memory_vec_needs_no_extension(tmp_path, monkeypatch):
    """R7: without memory_vec the CLI never loads sqlite-vec and still supersedes."""
    path, ids = await old_db(tmp_path, count=1)
    patch_clients(monkeypatch, ChatStub())

    async def must_not_load(self, db):
        raise AssertionError("sqlite-vec must not be loaded when memory_vec is absent")

    monkeypatch.setattr(MemoryPlugin, "_load_vec_extension", must_not_load)

    result = await invoke(
        "--db", path, "reconsolidate", sha(OLD)[:12], "--config", write_config(tmp_path)
    )

    assert_clean_exit(result, 0)
    assert sql(path, "SELECT superseded_by FROM memory WHERE id = ?", (ids[0],))[0][0] is not None


@pytest.mark.parametrize("disabled,expected", [([], "AppraisalPrior"), (["appraisal"], "RubricPrior")])
async def test_offline_memory_wires_appraisal_prior(tmp_path, monkeypatch, disabled, expected):
    """R7: the offline wiring installs the daemon's importance prior (appraisal unless disabled)."""
    from corvidae_cognition.memory_command import _open_offline_memory

    path, _ = await old_db(tmp_path, count=1)
    patch_clients(monkeypatch, ChatStub())
    config = write_config(tmp_path, plugins={"disabled": disabled})

    async with _open_offline_memory(path, config) as memory:
        assert type(memory.importance_prior).__name__ == expected


# ---------------------------------------------------------------------------
# R11: the new record inherits the old one's history
# ---------------------------------------------------------------------------


async def test_reconsolidating_demoted_record_keeps_history(tmp_path):
    """R11 (the AC): a demoted record with retrieval_count=4 stays demoted, with its history."""
    env = await make_env(tmp_path / "s.db")
    try:
        old_id = await add_record(
            env, indexed=0, embedded=0, retrieval_count=4, last_retrieved_at=777.0,
            created_at=123.0, importance=0.9,
        )
        env.memory.importance_prior.score.return_value = 0.3

        await run_reconsolidate(env, sha(OLD))

        old = await row_of(env, old_id)
        new = await row_of(env, old["superseded_by"])
        assert (new["indexed"], new["retrieval_count"], new["last_retrieved_at"]) == (0, 4, 777.0)
        assert new["created_at"] == 123.0
        assert new["importance"] == 0.9  # max(0.9, prior 0.3)
        assert new["embedded"] == 0
    finally:
        await close(env)


async def test_reconsolidating_live_record_keeps_history(tmp_path):
    """R11: a live record keeps its stats and creation time; importance rises to the prior; no merge."""
    env = await make_env(tmp_path / "s.db")
    try:
        old_id = await add_record(env, importance=0.4, retrieval_count=3, created_at=321.0)
        env.memory.importance_prior.score.return_value = 0.6

        await run_reconsolidate(env, sha(OLD))

        old = await row_of(env, old_id)
        new = await row_of(env, old["superseded_by"])
        assert (new["indexed"], new["retrieval_count"], new["created_at"]) == (1, 3, 321.0)
        assert new["importance"] == 0.6
        # Only the old row was superseded: near-duplicate merge never runs.
        assert sql(env.path, "SELECT COUNT(*) FROM memory WHERE superseded_by IS NOT NULL") == [(1,)]
    finally:
        await close(env)


async def test_demoted_during_llm_call_inherits_current_indexed(tmp_path):
    """R11: retention demoting the record while the LLM call runs is honoured by the new row."""
    env = await make_env(tmp_path / "s.db")
    try:
        old_id = await add_record(env)

        async def demote(messages):
            await set_state(env, old_id, indexed=0, embedded=0)

        env.chat.before_reply = demote
        await run_reconsolidate(env, sha(OLD))

        new = await row_of(env, (await row_of(env, old_id))["superseded_by"])
        assert (new["indexed"], new["embedded"]) == (0, 0)
    finally:
        await close(env)


async def test_history_read_inside_transaction(tmp_path):
    """R11: a retrieval bump landing during the LLM call is carried over, not lost."""
    env = await make_env(tmp_path / "s.db")
    try:
        old_id = await add_record(env, retrieval_count=1)

        async def bump(messages):
            await set_state(env, old_id, retrieval_count=5)

        env.chat.before_reply = bump
        await run_reconsolidate(env, sha(OLD))

        new = await row_of(env, (await row_of(env, old_id))["superseded_by"])
        assert new["retrieval_count"] == 5
    finally:
        await close(env)


# ---------------------------------------------------------------------------
# R8: selection
# ---------------------------------------------------------------------------


async def test_selection_skips_superseded_and_redacted_includes_demoted(tmp_path):
    """R8: live and demoted records are rebuilt; superseded and redacted are skipped and counted."""
    env = await make_env(tmp_path / "s.db")
    try:
        live = await add_record(env, n=1)
        demoted = await add_record(env, n=2, indexed=0)
        await add_record(env, n=3, superseded_by=live)
        await add_record(env, n=4, redacted=1)

        selection, report = await run_reconsolidate(env, sha(OLD))

        assert sorted(i for i, _ in selection.selected) == [live, demoted]
        assert {k: v for k, v in report.skipped.items() if v} == {"superseded": 1, "redacted": 1}
        assert sql(env.path, "SELECT COUNT(*) FROM memory WHERE prompt_hash = ?", (sha(NEW),)) == [(2,)]
    finally:
        await close(env)


async def test_range_without_dialog_is_skipped(tmp_path):
    """R8: a record whose message range has no dialog left is skipped as 'no dialog'."""
    env = await make_env(tmp_path / "s.db")
    try:
        old_id = await add_record(env)
        old = await row_of(env, old_id)
        # Turn the range's rows into non-dialog (context) rows.
        await env.db.execute(
            "UPDATE message_log SET message_type = 'context' WHERE id BETWEEN ? AND ?",
            (old["msg_id_start"], old["msg_id_end"]),
        )
        await env.db.commit()

        _, report = await run_reconsolidate(env, sha(OLD))

        assert {k: v for k, v in report.skipped.items() if v} == {"no dialog": 1}
        assert env.chat.attempts == []
        assert (await row_of(env, old_id))["superseded_by"] is None
    finally:
        await close(env)


async def test_record_superseded_after_selection_is_skipped(tmp_path):
    """R8: a record superseded between selection and the call is skipped; nothing is inserted."""
    from corvidae_cognition.memory import ReconsolidateSkipped

    env = await make_env(tmp_path / "s.db")
    try:
        old_id = await add_record(env)
        await set_state(env, old_id, superseded_by=old_id + 100)
        rows_before = sql(env.path, "SELECT COUNT(*) FROM memory")

        with pytest.raises(ReconsolidateSkipped, match="superseded"):
            await env.memory.reconsolidate_record(old_id)

        assert sql(env.path, "SELECT COUNT(*) FROM memory") == rows_before
        assert env.chat.attempts == []
    finally:
        await close(env)


async def test_tombstoned_id_costs_no_llm_call(tmp_path):
    """R8: an id redacted after selection is reported skipped and never reaches the LLM."""
    from corvidae_cognition.memory_command import (
        probe_memory_schema,
        reconsolidate,
        select_records,
    )

    env = await make_env(tmp_path / "s.db")
    try:
        old_id = await add_record(env)
        selection = await select_records(env.db, await probe_memory_schema(env.db), sha(OLD))
        await set_state(env, old_id, redacted=1)

        report = await reconsolidate(env.memory, selection)

        assert report.skipped.get("redacted") == 1
        assert env.chat.attempts == []
        assert sql(env.path, "SELECT COUNT(*) FROM memory") == [(1,)]
    finally:
        await close(env)


# ---------------------------------------------------------------------------
# Retry and current instructions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dry_run", [False, True])
async def test_current_hash_is_refused(tmp_path, monkeypatch, dry_run):
    """Records already under the current prompt are refused, before any plugin starts or write."""
    from corvidae_cognition.memory import RecordStamp

    env = await make_env(tmp_path / "s.db")
    await add_record(env, stamp=RecordStamp(NEW, "standalone", "m"))
    await close(env)
    snapshot = (sql(env.path, "SELECT * FROM memory"), sql(env.path, "SELECT * FROM memory_prompt"))
    chat = ChatStub()
    patch_clients(monkeypatch, chat)
    started: list[bool] = []
    original = LLMPlugin.on_start

    @functools.wraps(original)
    async def spy(self, config):
        started.append(True)
        return await original(self, config)

    monkeypatch.setattr(LLMPlugin, "on_start", spy)
    config = write_config(tmp_path)

    args = ["--db", env.path, "reconsolidate", sha(NEW)[:12], "--config", config]
    result = await invoke(*args, *(["--dry-run"] if dry_run else []))

    assert result.exit_code == 1, (result.output, repr(result.exception))
    for expected in ("already under the current instructions",
                     "change memory.consolidation_prompt", config):
        assert expected in result.output
    assert chat.attempts == [] and started == []
    assert (sql(env.path, "SELECT * FROM memory"), sql(env.path, "SELECT * FROM memory_prompt")) == snapshot


async def test_dry_run_with_config_on_old_stamp_starts_no_plugins(tmp_path, monkeypatch):
    """R6: --dry-run --config on a non-current stamp reads config only; no plugin starts."""
    path, ids = await old_db(tmp_path, count=1)
    started: list[bool] = []

    def boom(*args, **kwargs):
        started.append(True)
        raise AssertionError("dry run must not start plugins or build clients")

    monkeypatch.setattr(LLMPlugin, "on_start", functools.wraps(LLMPlugin.on_start)(boom))
    monkeypatch.setattr(LLMPlugin, "_create_client", boom)

    result = await invoke("--db", path, "reconsolidate", sha(OLD)[:12], "--config",
                          write_config(tmp_path), "--dry-run")

    assert_clean_exit(result, 0)
    assert f"#{ids[0]}" in result.output and started == []


async def test_retry_selects_only_remaining_old_records(tmp_path, monkeypatch):
    """A failed record can be retried with the same command; finished ones are not redone."""
    from corvidae_cognition.memory_command import (
        probe_memory_schema,
        select_records,
    )

    env = await make_env(tmp_path / "s.db")
    ids = [await add_record(env, n=n) for n in (1, 2, 3)]
    await close(env)
    chat = ChatStub()
    patch_clients(monkeypatch, chat)
    config = write_config(tmp_path)
    command = ("--db", env.path, "reconsolidate", sha(OLD)[:12], "--config", config)

    # First run: record #2's call fails, the other two succeed.
    chat.fail_marker = "talk 2"
    first = await invoke(*command)
    assert first.exit_code == 1, (first.output, repr(first.exception))
    assert sha(OLD)[:12] in first.output and sha(NEW)[:12] in first.output
    first_new = sql(env.path, "SELECT id FROM memory WHERE prompt_hash = ? ORDER BY id", (sha(NEW),))
    assert len(first_new) == 2
    calls_before = len(chat.consolidation_calls)

    # Second run, same command, LLM healthy again: only #2 is redone.
    chat.fail_marker = None
    second = await invoke(*command)
    assert_clean_exit(second, 0)
    assert len(chat.consolidation_calls) == calls_before + 1
    assert sql(env.path, "SELECT superseded_by FROM memory WHERE id IN (?, ?)",
               (first_new[0][0], first_new[1][0])) == [(None,), (None,)]
    async with aiosqlite.connect(env.path) as conn:
        selection = await select_records(conn, await probe_memory_schema(conn), sha(OLD))
    assert not selection.selected
    assert all(sql(env.path, "SELECT superseded_by FROM memory WHERE id = ?", (i,))[0][0] is not None
               for i in ids)


# ---------------------------------------------------------------------------
# R9: a per-record failure is logged and skipped
# ---------------------------------------------------------------------------


async def test_one_failure_does_not_stop_the_rest(tmp_path, monkeypatch, caplog):
    """R9: record #2's LLM call fails; #1 and #3 complete; the report and log name the failure."""
    env = await make_env(tmp_path / "s.db")
    ids = [await add_record(env, n=n) for n in (1, 2, 3)]
    await close(env)
    chat = ChatStub()
    chat.fail_marker = "talk 2"
    patch_offline(monkeypatch, chat)
    config = write_config(tmp_path)

    with caplog.at_level(logging.WARNING):
        result = await invoke("--db", env.path, "reconsolidate", sha(OLD)[:12], "--config", config)

    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert f"memory #{ids[1]} (irc:#test) failed (RuntimeError)" in result.output
    assert "Re-run 'corvidae memory --db" in result.output
    superseded = dict(sql(env.path, "SELECT id, superseded_by FROM memory WHERE id IN (?, ?, ?)", tuple(ids)))
    assert superseded[ids[0]] is not None and superseded[ids[2]] is not None
    assert superseded[ids[1]] is None
    # The log line names the id and carries the traceback.
    warnings = [r for r in caplog.records
                if r.levelno == logging.WARNING and str(ids[1]) in r.getMessage()
                and "re-consolidation failed" in r.getMessage()]
    assert warnings and warnings[0].exc_info


# ---------------------------------------------------------------------------
# R10: operator-only, beside a daemon, and clean errors
# ---------------------------------------------------------------------------


def test_memory_command_registered():
    """R10: `corvidae memory` is a plugin subcommand (needs `uv sync` to refresh entry points)."""
    commands = {ep.name: ep.value for ep in importlib.metadata.entry_points(group="corvidae.commands")}
    assert commands.get("memory") == "corvidae_cognition.memory_command:memory_command"


async def test_not_a_tool(tmp_path):
    """Regression guard (passes today): re-consolidation is never an agent tool."""
    from corvidae_cognition.memory_tools import MemoryToolsPlugin

    env = await make_env(tmp_path / "s.db")
    try:
        plugin = MemoryToolsPlugin()
        env.memory.pm.register(plugin, name="memory_tools")
        await plugin.on_init(pm=env.memory.pm, config=env.config)
        registered: list = []
        plugin.register_tools(registered)
        assert registered and not [t.name for t in registered if "reconsolidate" in t.name]
    finally:
        await close(env)


async def test_reconsolidate_beside_daemon_connection(tmp_path):
    """R10: the commit succeeds while a daemon connection holds a read transaction."""
    daemon = await make_env(tmp_path / "s.db")
    cli = await make_env(tmp_path / "s.db")
    try:
        await add_record(daemon)
        # The daemon connection sits inside an open read snapshot.
        await daemon.db.execute("BEGIN")
        async with daemon.db.execute("SELECT COUNT(*) FROM memory") as cursor:
            await cursor.fetchone()

        await run_reconsolidate(cli, sha(OLD))

        await daemon.db.rollback()
        async with daemon.db.execute(
            "SELECT COUNT(*) FROM memory WHERE prompt_hash = ?", (sha(NEW),)
        ) as cursor:
            assert (await cursor.fetchone())[0] == 1
    finally:
        await close(cli)
        await close(daemon)


async def test_unknown_stamp_prefix_message(tmp_path):
    """A prefix matching no stamp says so and points at the listing command."""
    path, _ = await old_db(tmp_path, count=1)

    result = await invoke("--db", path, "reconsolidate", "abc", "--dry-run")

    assert result.exit_code == 1
    assert "No records under stamp 'abc'" in result.output
    assert "corvidae memory --db" in result.output


async def test_ambiguous_prefix_message(tmp_path):
    """A prefix matching two stamps names the count and asks for more characters."""
    from corvidae_cognition.memory import RecordStamp

    # Pigeonhole: among 17 prompts two share the first hex digit of their hash.
    by_digit: dict[str, str] = {}
    pair = None
    for i in range(17):
        prompt = f"P{i}"
        digit = sha(prompt)[0]
        if digit in by_digit:
            pair = (by_digit[digit], prompt)
            break
        by_digit[digit] = prompt
    assert pair is not None
    env = await make_env(tmp_path / "s.db")
    for n, prompt in enumerate(pair, 1):
        await add_record(env, n=n, stamp=RecordStamp(prompt, "standalone", "m"))
    await close(env)
    prefix = sha(pair[0])[0]

    result = await invoke("--db", env.path, "reconsolidate", prefix, "--dry-run")

    assert result.exit_code == 1
    assert f"Stamp prefix '{prefix}' matches 2 stamps" in result.output
    assert "give more characters" in result.output


async def test_dry_run_on_pre_change_db(tmp_path):
    """A pre-change DB is listed and previewed without migrating it or raising."""
    path = str(tmp_path / "s.db")
    ids = await make_pre_change_db(path, rows=3, superseded=(3,))
    before = Path(path).read_bytes()

    result = await invoke("--db", path, "reconsolidate", "unknown", "--dry-run")

    assert_clean_exit(result, 0)
    assert f"#{ids[0]}" in result.output and f"#{ids[1]}" in result.output
    assert f"#{ids[2]}" not in result.output
    assert "would skip 1 superseded" in result.output
    missing = await invoke("--db", path, "reconsolidate", "abc", "--dry-run")
    assert "No records under stamp 'abc'" in missing.output
    listing = await invoke("--db", path, "stamps")
    assert_clean_exit(listing, 0)
    [unknown] = [line for line in listing.output.splitlines() if line.lstrip().startswith("unknown")]
    assert "3 records" in unknown
    # Read-only throughout: no migration, no changed bytes.
    assert Path(path).read_bytes() == before
    columns = {row[1] for row in sql(path, "PRAGMA table_info(memory)")}
    assert "prompt_hash" not in columns


async def test_partial_migration_db(tmp_path):
    """Only prompt_hash was added before a migration stopped: reads treat the rest as NULL."""
    path = str(tmp_path / "s.db")
    await make_pre_change_db(path, rows=2, partial=True)

    listing = await invoke("--db", path, "stamps")
    dry = await invoke("--db", path, "reconsolidate", "unknown", "--dry-run")

    assert_clean_exit(listing, 0)
    assert_clean_exit(dry, 0)


@pytest.mark.parametrize("form", ["stamps", "dry", "live"])
async def test_no_memory_table(tmp_path, monkeypatch, form):
    """Without a memory table every form says so and creates no schema."""
    path = str(tmp_path / "s.db")
    async with aiosqlite.connect(path) as raw:
        await init_db(raw)
        await raw.execute("PRAGMA journal_mode=wal")
    chat = ChatStub()
    patch_clients(monkeypatch, chat)
    args = {
        "stamps": ["stamps"],
        "dry": ["reconsolidate", "unknown", "--dry-run"],
        "live": ["reconsolidate", "unknown", "--config", write_config(tmp_path)],
    }[form]

    result = await invoke("--db", path, *args)

    assert_clean_exit(result, 1)
    assert f"No memory table in {path}" in result.output
    assert sql(path, "SELECT name FROM sqlite_master WHERE name = 'memory'") == []
    assert chat.attempts == []


async def test_live_run_selects_after_migration(tmp_path, monkeypatch):
    """A live run on a pre-change DB migrates first, then rebuilds every unknown row."""
    path = str(tmp_path / "s.db")
    ids = await make_pre_change_db(path, rows=2)
    patch_clients(monkeypatch, ChatStub())

    result = await invoke("--db", path, "reconsolidate", "unknown", "--config", write_config(tmp_path))

    assert_clean_exit(result, 0)
    new_ids = {r[0] for r in sql(path, "SELECT id FROM memory WHERE prompt_hash = ?", (sha(NEW),))}
    assert len(new_ids) == 2
    for old_id in ids:
        [(superseded_by,)] = sql(path, "SELECT superseded_by FROM memory WHERE id = ?", (old_id,))
        assert superseded_by in new_ids


async def test_stamp_validation_message(tmp_path):
    """A STAMP that is not 'unknown', 'direct' or a hex prefix is a usage error naming the value."""
    path, _ = await old_db(tmp_path, count=1)
    before = Path(path).read_bytes()

    bad = await invoke("--db", path, "reconsolidate", "ab%", "--dry-run")
    other = await invoke("--db", path, "reconsolidate", "XYZ", "--dry-run")
    direct = await invoke("--db", path, "reconsolidate", "direct", "--dry-run")

    assert bad.exit_code == 2 and other.exit_code == 2
    assert "got 'ab%'" in bad.output
    for result in (bad, other):
        assert "'unknown'" in result.output and "0-9, a-f" in result.output
    assert direct.exit_code != 0 and "never re-consolidated" in direct.output
    assert Path(path).read_bytes() == before


async def test_config_errors_are_wrapped(tmp_path, monkeypatch):
    """Bad config is reported as 'Config error: ... - check <path>', never a traceback."""
    path, _ = await old_db(tmp_path, count=1)
    patch_clients(monkeypatch, ChatStub())
    good_llm = {
        "main": {"base_url": "http://localhost:8080", "model": "chat"},
        "background": {"base_url": "http://localhost:8080", "model": "bg"},
    }
    cases = [
        ({"llm": {"background": good_llm["background"]}}, "missing required key llm.main"),
        ({"llm": {"main": good_llm["main"], "background": {"model": "bg"}}},
         "missing required key llm.background.base_url"),
        ({"memory": {"consolidation_prompt": None}}, "memory.consolidation_prompt must be a string"),
        ({"plugins": {"disabled": 5}}, "plugins.disabled"),
        ({"plugins": 5}, "plugins must be a mapping"),
        # A non-mapping logging block is a config error too.
        ({"logging": "DEBUG"}, "logging must be a mapping"),
        ({"memory": {"channel_groups": "oops"}}, "memory.channel_groups"),
        # An unexpected exception from plugin init is wrapped, not a traceback.
        ({"memory": {"retrieval": 5}}, "Config error:"),
    ]
    for sections, key in cases:
        config = write_config(tmp_path, **sections)

        result = await invoke("--db", path, "reconsolidate", sha(OLD)[:12], "--config", config)

        assert_clean_exit(result, 1)
        assert "Config error:" in result.output, (sections, result.output)
        assert key in result.output, (sections, result.output)
        assert config in result.output, (sections, result.output)
    # Exact wording for the first case, per the design.
    first = await invoke("--db", path, "reconsolidate", sha(OLD)[:12], "--config",
                         write_config(tmp_path, **cases[0][0]))
    assert f"Config error: missing required key llm.main — check {tmp_path / 'agent.yaml'}" in first.output


async def test_non_utf8_config_is_a_config_error(tmp_path):
    """A config saved in another encoding is a Config error naming the file, not a traceback."""
    path, _ = await old_db(tmp_path, count=1)
    config = tmp_path / "bad.yaml"
    # Latin-1 bytes for an accented prompt: invalid UTF-8.
    config.write_bytes(
        b"llm:\n  main: {base_url: 'http://localhost:8080', model: chat}\n"
        b"memory:\n  consolidation_prompt: 'r\xe9sum\xe9'\n"
    )

    result = await invoke(
        "--db", path, "reconsolidate", sha(OLD)[:12], "--dry-run", "--config", str(config)
    )

    assert_clean_exit(result, 1)
    assert "Config error:" in result.output
    assert "not valid UTF-8" in result.output
    assert str(config) in result.output


async def test_null_llm_role_is_not_refused(tmp_path, monkeypatch):
    """A role set to null (llm.background: ~) is valid for the daemon, so the CLI accepts it."""
    path, _ = await old_db(tmp_path, count=1)
    patch_clients(monkeypatch, ChatStub())
    config = write_config(tmp_path, llm={
        "main": {"base_url": "http://localhost:8080", "model": "chat"},
        "background": None,
    })

    result = await invoke("--db", path, "reconsolidate", sha(OLD)[:12], "--config", config)

    assert_clean_exit(result, 0)
    assert "Config error" not in result.output


async def test_all_superseded_is_nothing_to_do(tmp_path, monkeypatch):
    """When every record under the stamp is already superseded or redacted, say so and exit 0."""
    env = await make_env(tmp_path / "s.db")
    await add_record(env, n=1, superseded_by=99)
    await add_record(env, n=2, redacted=1)
    await close(env)
    chat = ChatStub()
    patch_clients(monkeypatch, chat)
    stamp = sha(OLD)[:12]

    result = await invoke("--db", env.path, "reconsolidate", stamp, "--config", write_config(tmp_path))

    assert_clean_exit(result, 0)
    assert f"All 2 records under {stamp} are already superseded or redacted; nothing to do." in result.output
    assert chat.attempts == []


async def test_cli_logs_to_stderr(tmp_path, monkeypatch):
    """The CLI never writes into the daemon's log file; its warnings go to stderr."""
    env = await make_env(tmp_path / "s.db")
    for n in (1, 2):
        await add_record(env, n=n)
    await close(env)
    chat = ChatStub()
    chat.fail_marker = "talk 2"
    patch_clients(monkeypatch, chat)
    config = write_config(tmp_path, logging={"file": "daemon.log", "level": "INFO"})
    monkeypatch.chdir(tmp_path)

    result = await invoke("--db", env.path, "reconsolidate", sha(OLD)[:12], "--config", config)

    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert not (tmp_path / "daemon.log").exists()
    assert "re-consolidation failed for memory" in result.stderr


async def test_config_required_without_dry_run(tmp_path):
    """A live run needs --config; the error says so and offers --dry-run."""
    path, _ = await old_db(tmp_path, count=1)

    result = await invoke("--db", path, "reconsolidate", "abc")

    assert result.exit_code == 2
    assert "needs --config" in result.output and "--dry-run" in result.output


async def test_wal_message_names_mode_and_command(tmp_path, monkeypatch):
    """A non-WAL database is refused with its path, journal mode and this command's name."""
    path, _ = await old_db(tmp_path, count=1)
    # Flip the closed database out of WAL, as an unstarted daemon's DB would be.
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=delete")
    conn.close()
    snapshot = sql(path, "SELECT * FROM memory ORDER BY id")
    chat = ChatStub()
    patch_clients(monkeypatch, chat)

    result = await invoke(
        "--db", path, "reconsolidate", sha(OLD)[:12], "--config", write_config(tmp_path)
    )

    assert result.exit_code == 1, (result.output, repr(result.exception))
    for expected in (path, "'delete' journal mode", "corvidae memory reconsolidate",
                     "Start the daemon once on this database, then retry",
                     "Nothing was changed"):
        assert expected in result.output
    assert "redact" not in result.output
    assert chat.attempts == []
    assert sql(path, "SELECT * FROM memory ORDER BY id") == snapshot
