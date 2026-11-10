"""Tests for ``scripts/eval_memory.py --snapshot`` (#658).

Snapshot mode copies a deployment's ``sessions.db`` (never opening the
original for write), runs the real ``MemoryPlugin`` retrieval path over the
copy with the deployment's ``agent.yaml``, scores operator labels with
Recall@k and MRR, and writes every query's whole candidate pool to JSON.

No network: ``LLMClient.embed`` is replaced by the bag-of-words stub and
``LLMClient.chat`` records and raises. Everything lives under ``tmp_path``.
"""

import asyncio
import copy
import hashlib
import importlib.util
import json
import math
import re
import shutil
import sqlite3
import sys
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml
from evals.metrics import mrr, recall_at_k

from corvidae.llm import LLMClient

from .test_memory_retrieval import (
    EMBED_CONFIG,
    bow_embed,
    build_retrieval_env,
    seed_memory,
)

# Absolute path of the script under test; it is not an importable package.
SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "eval_memory.py"

# True when the sqlite-vec extension can be imported (the vec path exists).
HAS_VEC = importlib.util.find_spec("sqlite_vec") is not None

# Marker for tests that need the vector path (the FTS job runs the rest).
needs_vec = pytest.mark.skipif(not HAS_VEC, reason="sqlite-vec unavailable")

# Three summaries whose bag-of-words overlap with "alpha beta gamma delta"
# fixes the cosine order (1.0, 0.5, 0.25) so rankings are known in advance.
TEXT_A = "alpha beta gamma delta"
TEXT_B = "alpha beta epsilon zeta"
TEXT_C = "alpha eta theta iota"

# A chat-only config: no embedding profile, so retrieval is FTS5-only.
FTS_CONFIG: dict[str, Any] = {
    "llm": {"main": {"base_url": "http://localhost:8080", "model": "chat"}},
}


class StubCalls:
    """Records of the calls the stubbed ``LLMClient`` received."""

    def __init__(self) -> None:
        # Each embed call as (texts, kind).
        self.embeds: list[tuple[list[str], str]] = []
        # Each chat call's messages.
        self.chats: list[list[dict]] = []


@pytest.fixture
def stub_calls(monkeypatch: pytest.MonkeyPatch) -> StubCalls:
    """Replace LLMClient.embed/chat so no request ever leaves the process."""
    calls = StubCalls()

    # Embed returns deterministic bag-of-words vectors and records the call.
    async def fake_embed(self: LLMClient, texts: list[str], kind: str) -> list[list[float]]:
        calls.embeds.append((list(texts), kind))
        return [bow_embed(t) for t in texts]

    # Chat records the call and raises: retrieval must never reach it.
    async def fake_chat(self: LLMClient, messages: list[dict], *args: Any, **kwargs: Any) -> dict:
        calls.chats.append(messages)
        raise AssertionError("snapshot mode must never call chat()")

    monkeypatch.setattr(LLMClient, "embed", fake_embed)
    monkeypatch.setattr(LLMClient, "chat", fake_chat)
    return calls


@pytest.fixture
def script(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load scripts/eval_memory.py by path (it is not a package)."""
    name = "eval_memory_under_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered so dataclasses defined in the script resolve their module.
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def private_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point tempfile at a tmp_path subdirectory so work_dir stays inside it."""
    sub = tmp_path / "systmp"
    sub.mkdir()
    monkeypatch.setattr("tempfile.tempdir", str(sub))
    return sub


def sha256(path: Path) -> str:
    """Hex sha256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_config(path: Path, config: dict[str, Any], memory: dict[str, Any] | None = None) -> Path:
    """Write an agent.yaml (deep-copied config plus an optional memory: block)."""
    data = copy.deepcopy(config)
    if memory is not None:
        data["memory"] = memory
    path.write_text(yaml.safe_dump(data))
    return path


def write_labels(path: Path, queries: list[dict[str, Any]]) -> Path:
    """Write a labels file in the documented format."""
    path.write_text(json.dumps({"description": "test", "queries": queries}))
    return path


def build_source(
    db_path: Path,
    summaries: list[str],
    config: dict[str, Any] | None = None,
    created_at: float | None = None,
    embedded: bool = HAS_VEC,
    mutate: Callable[[Any, list[int]], Awaitable[None]] | None = None,
) -> list[int]:
    """Build a WAL-mode sessions.db with memory rows; return their ids.

    Rows are embedded by default only when sqlite-vec is present (an FTS-only
    install cannot write a vec row); pass embedded=False for a config with no
    embedding profile. ``mutate(db, ids)`` runs after the plugin has
    stopped and before the final commit, for per-test table edits. The
    database is closed afterwards, so its WAL is checkpointed.
    """

    async def build() -> list[int]:
        # Real schema via the plugin's own startup, on a file database.
        memory, _embedder, channel, _conv, db = await build_retrieval_env(
            config or EMBED_CONFIG, db_path=str(db_path)
        )
        await memory.wait_for_background_tasks()
        # Create the memory/vec/embedding_meta tables before seeding rows.
        await memory._ensure_schema()
        await db.execute("PRAGMA journal_mode=WAL")
        ids = [
            await seed_memory(db, channel.id, s, created_at=created_at, embedded=embedded)
            for s in summaries
        ]
        await memory.on_stop()
        if mutate is not None:
            await mutate(db, ids)
        await db.commit()
        await db.close()
        return ids

    return asyncio.run(build())


def read_pool(path: Path) -> dict[str, Any]:
    """Load a written pool JSON file."""
    return json.loads(path.read_text())


def test_snapshot_copy_leaves_source_bytes_and_includes_wal_rows(
    script: ModuleType, stub_calls: StubCalls, private_tmp: Path,
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """S1: the source is only opened read-only, and WAL-only rows are copied."""
    live_dir = tmp_path / "live"
    live_dir.mkdir()
    live_db = live_dir / "sessions.db"

    # Source with the memory schema but no embedding_meta row: a read-write
    # open by the plugin would insert one and change the file's bytes.
    async def drop_meta(db: Any, ids: list[int]) -> None:
        await db.execute("DELETE FROM embedding_meta")

    build_source(live_db, ["harbor lighthouse keeper"], mutate=drop_meta)

    # A still-open stdlib writer commits one more row that lives only in -wal.
    writer = sqlite3.connect(live_db)
    try:
        writer.execute("PRAGMA wal_autocheckpoint=0")
        if HAS_VEC:
            import sqlite_vec

            writer.enable_load_extension(True)
            sqlite_vec.load(writer)
        wal_text = "zanzibar quokka sighting"
        cursor = writer.execute(
            "INSERT INTO memory (channel_id, created_at, summary, importance, "
            "msg_id_start, msg_id_end, embedded) VALUES (?, ?, ?, 0.5, 1, 2, ?)",
            ("irc:#electronics", time.time(), wal_text, 1 if HAS_VEC else 0),
        )
        wal_id = cursor.lastrowid
        if HAS_VEC:
            writer.execute(
                "INSERT INTO memory_vec (memory_id, embedding) VALUES (?, ?)",
                (wal_id, sqlite_vec.serialize_float32(bow_embed(wal_text))),
            )
        writer.commit()

        # Crashed-daemon image: the db plus its -wal, copied while writer is open.
        image_dir = tmp_path / "image"
        image_dir.mkdir()
        image_db = image_dir / "sessions.db"
        shutil.copy(live_db, image_db)
        shutil.copy(Path(f"{live_db}-wal"), Path(f"{image_db}-wal"))
    finally:
        writer.close()
    image_wal = Path(f"{image_db}-wal")
    # Sanity: the committed row really is only in the WAL image.
    assert image_wal.stat().st_size > 0

    config = write_config(tmp_path / "agent.yaml", EMBED_CONFIG)
    labels = write_labels(
        tmp_path / "labels.json", [{"text": "zanzibar quokka", "relevant": [wal_id]}]
    )
    pool_out = tmp_path / "pool.json"
    before = (sha256(image_db), sha256(image_wal))

    script.main([
        "--snapshot", str(image_db), "--labels", str(labels),
        "--config", str(config), "--pool-out", str(pool_out),
    ])

    # Neither the db nor its -wal changed, and the WAL-only row was scored.
    assert (sha256(image_db), sha256(image_wal)) == before
    pool = read_pool(pool_out)
    assert wal_id in [c["id"] for c in pool["queries"][0]["candidates"]]

    # Writing the pool onto the source itself is refused before any work.
    with pytest.raises(SystemExit) as excinfo:
        script.main([
            "--snapshot", str(image_db), "--labels", str(labels),
            "--config", str(config), "--pool-out", str(image_db),
        ])
    err = capsys.readouterr().err
    assert excinfo.value.code == 2
    assert "--pool-out" in err and str(image_db.resolve()) in err
    assert (sha256(image_db), sha256(image_wal)) == before


@needs_vec
def test_snapshot_scores_known_ranking(
    script: ModuleType, stub_calls: StubCalls, private_tmp: Path,
    tmp_path: Path,
) -> None:
    """S2: per-query and mean Recall@k / MRR equal the hand-computed values."""
    source = tmp_path / "sessions.db"
    id_a, id_b, id_c = build_source(source, [TEXT_A, TEXT_B, TEXT_C])
    config = write_config(tmp_path / "agent.yaml", EMBED_CONFIG)
    # Query 1 ranks A, B, C (cosine 1.0, 0.5, 0.25); query 2 ranks C first,
    # then A and B tie at 0.25 and B (seeded later, so newer) wins the tie;
    # query 3's only relevant id (C) ranks third, below k=2.
    labels = write_labels(tmp_path / "labels.json", [
        {"text": TEXT_A, "relevant": [id_b, id_c]},
        {"text": TEXT_C, "relevant": [id_c]},
        {"text": TEXT_A, "relevant": [id_c]},
    ])
    pool_out = tmp_path / "pool.json"

    script.main([
        "--snapshot", str(source), "--labels", str(labels), "--config", str(config),
        "--pool-out", str(pool_out), "--k", "2",
    ])

    # Expected values come from the shared metric functions on the known order.
    exp1_r = recall_at_k([id_a, id_b], [id_b, id_c], 2)
    exp1_m = mrr([id_a, id_b], [id_b, id_c])
    exp2_r = recall_at_k([id_c, id_b], [id_c], 2)
    exp2_m = mrr([id_c, id_b], [id_c])
    exp3_r = recall_at_k([id_a, id_b], [id_c], 2)
    exp3_m = mrr([id_a, id_b], [id_c])
    pool = read_pool(pool_out)
    q1, q2, q3 = pool["queries"]
    assert q1["recall_at_k"] == pytest.approx(exp1_r) == pytest.approx(0.5)
    assert q1["mrr"] == pytest.approx(exp1_m) == pytest.approx(0.5)
    assert q2["recall_at_k"] == pytest.approx(exp2_r) == pytest.approx(1.0)
    assert q2["mrr"] == pytest.approx(exp2_m) == pytest.approx(1.0)
    # Below the cutoff: recall 0 and MRR@k 0.
    assert q3["recall_at_k"] == pytest.approx(exp3_r) == pytest.approx(0.0)
    assert q3["mrr"] == pytest.approx(exp3_m) == pytest.approx(0.0)
    assert pool["summary"]["mean_recall_at_k"] == pytest.approx(
        (exp1_r + exp2_r + exp3_r) / 3
    )
    assert pool["summary"]["mean_mrr"] == pytest.approx((exp1_m + exp2_m + exp3_m) / 3)
    assert pool["summary"]["scored"] == 3


def test_snapshot_reports_fts_fallback_on_encoder_mismatch(
    script: ModuleType, stub_calls: StubCalls, private_tmp: Path,
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """S3: an encoder mismatch degrades every query and the run says so."""
    # Source built with encoder "test-embedder"; the config names another.
    source = tmp_path / "sessions.db"
    ids = build_source(source, [TEXT_A, TEXT_B])
    other = copy.deepcopy(EMBED_CONFIG)
    other["llm"]["embedding"]["model"] = "other-embedder"
    config = write_config(tmp_path / "agent.yaml", other)
    labels = write_labels(tmp_path / "labels.json", [
        {"text": TEXT_A, "relevant": [ids[0]]},
        {"text": TEXT_B, "relevant": [ids[1]]},
    ])
    pool_out = tmp_path / "pool.json"

    script.main([
        "--snapshot", str(source), "--labels", str(labels),
        "--config", str(config), "--pool-out", str(pool_out),
    ])

    # Per-query flag, summary count, and the stdout sentence.
    pool = read_pool(pool_out)
    assert [q["degraded"] for q in pool["queries"]] == [True, True]
    assert pool["summary"]["degraded"] == 2
    assert "FTS5 fallback on 2 of 2 queries" in capsys.readouterr().out


def test_snapshot_names_unreachable_label_ids(
    script: ModuleType, stub_calls: StubCalls, private_tmp: Path,
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """S4: unreachable labelled ids are reported by id and status, not scored."""
    states: dict[str, int] = {}

    # Turn three rows into a redacted, a superseded and an unindexed record.
    async def mark(db: Any, ids: list[int]) -> None:
        reachable, redacted, superseded, not_indexed = ids
        states.update(
            reachable=reachable, redacted=redacted,
            superseded=superseded, not_indexed=not_indexed,
        )
        await db.execute("UPDATE memory SET redacted = 1 WHERE id = ?", (redacted,))
        await db.execute(
            "UPDATE memory SET indexed = 0, superseded_by = ? WHERE id = ?",
            (reachable, superseded),
        )
        await db.execute("UPDATE memory SET indexed = 0 WHERE id = ?", (not_indexed,))

    source = tmp_path / "sessions.db"
    build_source(source, [TEXT_A, TEXT_A, TEXT_A, TEXT_A], mutate=mark)
    absent = 9999
    config = write_config(tmp_path / "agent.yaml", EMBED_CONFIG)
    labels = write_labels(tmp_path / "labels.json", [
        {"text": TEXT_A, "relevant": [
            states["reachable"], absent, states["redacted"],
            states["superseded"], states["not_indexed"],
        ]},
        {"text": TEXT_A, "relevant": [absent, states["redacted"]]},
    ])
    pool_out = tmp_path / "pool.json"

    script.main([
        "--snapshot", str(source), "--labels", str(labels),
        "--config", str(config), "--pool-out", str(pool_out),
    ])

    pool = read_pool(pool_out)
    q0, q1 = pool["queries"]
    # Each unreachable id carries its status; the reachable one is scored.
    assert {u["id"]: u["status"] for u in q0["unscorable"]} == {
        absent: "absent",
        states["redacted"]: "redacted",
        states["superseded"]: "superseded",
        states["not_indexed"]: "not_indexed",
    }
    assert q0["recall_at_k"] == pytest.approx(1.0)
    # A query with no reachable id has null metrics and is not counted.
    assert q1["recall_at_k"] is None and q1["mrr"] is None
    assert pool["summary"]["scored"] == 1
    # Stdout names every unreachable id.
    out = capsys.readouterr().out
    unscorable_lines = [ln for ln in out.splitlines() if "unscorable" in ln]
    assert unscorable_lines
    for ident in (absent, states["redacted"], states["superseded"], states["not_indexed"]):
        # Word-boundary match so id 1 is not satisfied by id 10 or 9999.
        assert any(re.search(rf"\b{ident}\b", ln) for ln in unscorable_lines)
    for status in ("absent", "redacted", "superseded", "not_indexed"):
        assert status in out


def test_snapshot_never_calls_chat_or_embeds_documents(
    script: ModuleType, stub_calls: StubCalls, private_tmp: Path,
    tmp_path: Path,
) -> None:
    """S5: retrieval only: no chat call, no document embed, no retention."""

    # Leave work a wrongly-run on_start would do: unconsolidated messages,
    # an embedded=0 record, and no retention_meta row (so retention is due).
    async def leave_work(db: Any, ids: list[int]) -> None:
        for n in range(3):
            await db.execute(
                "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
                "VALUES (?, ?, ?, 'message')",
                ("irc:#electronics",
                 json.dumps({"role": "user", "content": f"pending message {n}"}),
                 time.time()),
            )
        await seed_memory(db, "irc:#electronics", TEXT_C, embedded=False)
        await db.execute("DELETE FROM retention_meta")

    source = tmp_path / "sessions.db"
    ids = build_source(source, [TEXT_A, TEXT_B], mutate=leave_work)
    config = write_config(tmp_path / "agent.yaml", EMBED_CONFIG)
    query_texts = [TEXT_A, TEXT_B]
    labels = write_labels(tmp_path / "labels.json", [
        {"text": query_texts[0], "relevant": [ids[0]]},
        {"text": query_texts[1], "relevant": [ids[1]]},
    ])

    script.main([
        "--snapshot", str(source), "--labels", str(labels), "--config", str(config),
        "--pool-out", str(tmp_path / "pool.json"),
    ])

    # Never a chat request.
    assert stub_calls.chats == []
    if HAS_VEC:
        # One query-kind embed per labels query, nothing else.
        assert len(stub_calls.embeds) == len(query_texts)
        assert all(kind == "query" for _texts, kind in stub_calls.embeds)
        assert sorted(texts[0] for texts, _kind in stub_calls.embeds) == sorted(query_texts)
    else:
        # The FTS path embeds nothing.
        assert stub_calls.embeds == []


def test_fixture_mode_unchanged_and_flag_conflicts_named(
    script: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """S6: fixture mode still works, and a missing --labels is named with its remedy."""
    # A tiny retriever module on sys.path stands in for a real retriever.
    module_name = "snapshot_test_reverse_retriever"
    (tmp_path / f"{module_name}.py").write_text(
        "def reverse_retriever(query_text, memories):\n"
        "    return [m['id'] for m in reversed(memories)]\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    # Driven through sys.argv: this half must hold on the pre-snapshot main().
    monkeypatch.setattr(
        sys, "argv",
        ["eval_memory.py", "--retriever", f"{module_name}:reverse_retriever"],
    )
    try:
        script.main()
    finally:
        sys.modules.pop(module_name, None)
    out = capsys.readouterr().out
    assert str(script.DEFAULT_FIXTURE) in out
    assert "recall@5" in out

    # --snapshot without --labels: exit 2, naming the flag and the remedy.
    with pytest.raises(SystemExit) as excinfo:
        script.main([
            "--snapshot", str(tmp_path / "sessions.db"),
            "--config", str(tmp_path / "agent.yaml"),
            "--pool-out", str(tmp_path / "pool.json"),
        ])
    err = capsys.readouterr().err
    assert excinfo.value.code == 2
    assert "--labels" in err
    assert "--labels PATH" in err


@pytest.mark.parametrize("pool_name, extra, named", [
    ("sessions.db-wal", [], "WAL file"),
    ("sessions.db-shm", [], "shared-memory file"),
    ("agent.yaml", [], "--config"),
    ("pool.json", ["--k", "0"], "--k must be a positive integer"),
])
def test_snapshot_rejects_unsafe_pool_out_and_k(
    script: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    pool_name: str, extra: list[str], named: str,
) -> None:
    """S1 guard (review fix): --pool-out may not overwrite the source's WAL/shm
    files or the config, and --k must be positive; each is named before any I/O."""
    # Argument errors exit 2 from parser.error, naming what was refused.
    with pytest.raises(SystemExit) as excinfo:
        script.main([
            "--snapshot", str(tmp_path / "sessions.db"),
            "--labels", str(tmp_path / "labels.json"),
            "--config", str(tmp_path / "agent.yaml"),
            "--pool-out", str(tmp_path / pool_name), *extra,
        ])
    assert excinfo.value.code == 2
    assert named in capsys.readouterr().err


@needs_vec
def test_snapshot_pool_json_holds_whole_pool(
    script: ModuleType, stub_calls: StubCalls, private_tmp: Path,
    tmp_path: Path,
) -> None:
    """S7: with k=1 the pool JSON still lists every scored candidate."""
    source = tmp_path / "sessions.db"
    ids = build_source(source, [TEXT_A, TEXT_B, TEXT_C])
    config = write_config(tmp_path / "agent.yaml", EMBED_CONFIG)
    labels = write_labels(tmp_path / "labels.json", [{"text": TEXT_A, "relevant": [ids[0]]}])
    pool_out = tmp_path / "pool.json"

    script.main([
        "--snapshot", str(source), "--labels", str(labels), "--config", str(config),
        "--pool-out", str(pool_out), "--k", "1",
    ])

    # All three candidates, ranked 1..3, each with the six documented fields.
    candidates = read_pool(pool_out)["queries"][0]["candidates"]
    assert [c["rank"] for c in candidates] == [1, 2, 3]
    assert [c["id"] for c in candidates] == ids
    for cand in candidates:
        assert set(cand) == {"id", "rank", "similarity", "score", "band", "source"}
        assert cand["band"] in {"strong", "moderate", "weak"}
    scores = [c["score"] for c in candidates]
    assert scores == sorted(scores, reverse=True)


def test_config_half_life_days_changes_scores(
    script: ModuleType, stub_calls: StubCalls, private_tmp: Path,
    tmp_path: Path,
) -> None:
    """S8: memory.half_life_days from --config reaches the scorer."""
    # FTS-path source (no embedding profile) with a ten-day-old row.
    source = tmp_path / "sessions.db"
    ids = build_source(
        source, [TEXT_A], config=FTS_CONFIG, created_at=time.time() - 10 * 86400,
        embedded=False,
    )
    labels = write_labels(tmp_path / "labels.json", [{"text": TEXT_A, "relevant": [ids[0]]}])

    # Two runs differing only in the config's half-life.
    scores: dict[float, float] = {}
    for half_life in (1, 1000):
        config = write_config(
            tmp_path / f"agent{half_life}.yaml", FTS_CONFIG,
            memory={"half_life_days": half_life},
        )
        pool_out = tmp_path / f"pool{half_life}.json"
        script.main([
            "--snapshot", str(source), "--labels", str(labels),
            "--config", str(config), "--pool-out", str(pool_out),
        ])
        candidates = read_pool(pool_out)["queries"][0]["candidates"]
        assert [c["id"] for c in candidates] == ids
        scores[half_life] = candidates[0]["score"]

    # score = 0.6 (FTS similarity proxy) * exp(-10 / half_life_days).
    for half_life, score in scores.items():
        assert score == pytest.approx(0.6 * math.exp(-10 / half_life), rel=1e-3)


def test_snapshot_not_a_database_reports_open_error(
    script: ModuleType, stub_calls: StubCalls, private_tmp: Path,
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """S1 edge: a non-database --snapshot gives one named line, no traceback."""
    bogus = tmp_path / "notadb.db"
    bogus.write_text("this is not a sqlite database\n" * 20)
    config = write_config(tmp_path / "agent.yaml", EMBED_CONFIG)
    labels = write_labels(tmp_path / "labels.json", [{"text": TEXT_A, "relevant": [1]}])

    with pytest.raises(SystemExit) as excinfo:
        script.main([
            "--snapshot", str(bogus), "--labels", str(labels),
            "--config", str(config), "--pool-out", str(tmp_path / "pool.json"),
        ])

    # Exit 1, exactly one eval_memory: line naming the path and the remedy.
    err = capsys.readouterr().err
    assert excinfo.value.code == 1
    lines = err.strip().splitlines()
    assert len(lines) == 1
    assert lines[0].startswith(f"eval_memory: cannot open snapshot {bogus.resolve()} read-only (")
    assert "check the --snapshot path" in lines[0]
    assert "Traceback" not in err


# ---------------------------------------------------------------------------
# Retrieval variants (#665 query context, #666 candidate sources): a probe can
# carry the reply it followed ("context") and a "subset" tag; --query-context
# and --candidates override the config so variants score against one snapshot.
# ---------------------------------------------------------------------------


def run(script: ModuleType, tmp_path: Path, labels: Path, *extra: str) -> dict:
    """Run snapshot mode over tmp_path/sessions.db and return the pool JSON."""
    pool_out = tmp_path / "pool.json"
    script.main([
        "--snapshot", str(tmp_path / "sessions.db"), "--labels", str(labels),
        "--config", str(tmp_path / "agent.yaml"), "--pool-out", str(pool_out),
        *extra,
    ])
    return read_pool(pool_out)


@needs_vec
def test_context_precedes_the_probe_text_in_the_query(
    script: ModuleType, stub_calls: StubCalls, private_tmp: Path, tmp_path: Path,
) -> None:
    """With the default query_context the embedded query is context + text."""
    build_source(tmp_path / "sessions.db", [TEXT_A])
    write_config(tmp_path / "agent.yaml", EMBED_CONFIG)
    labels = write_labels(tmp_path / "labels.json", [
        {"text": "thanks, that worked", "context": TEXT_A, "relevant": [1]},
    ])

    pool = run(script, tmp_path, labels)

    assert stub_calls.embeds[-1] == ([f"{TEXT_A}\n\nthanks, that worked"], "query")
    query = pool["queries"][0]
    assert query["query"] == f"{TEXT_A}\n\nthanks, that worked"
    assert query["context"] == TEXT_A
    assert pool["summary"]["query_context"] == "previous_reply"


@needs_vec
def test_query_context_none_flag_scores_the_baseline(
    script: ModuleType, stub_calls: StubCalls, private_tmp: Path, tmp_path: Path,
) -> None:
    """--query-context none overrides the config: the context is not sent."""
    build_source(tmp_path / "sessions.db", [TEXT_A])
    write_config(tmp_path / "agent.yaml", EMBED_CONFIG)
    labels = write_labels(tmp_path / "labels.json", [
        {"text": "thanks, that worked", "context": TEXT_A, "relevant": [1]},
    ])

    pool = run(script, tmp_path, labels, "--query-context", "none")

    assert stub_calls.embeds[-1] == (["thanks, that worked"], "query")
    assert pool["summary"]["query_context"] == "none"


@needs_vec
def test_follow_up_probe_recalls_only_with_context(
    script: ModuleType, stub_calls: StubCalls, private_tmp: Path, tmp_path: Path,
) -> None:
    """A follow-up probe finds its record with the reply as context, not without.

    Alone, "eta theta" ranks C first; with "beta gamma delta" before it, A
    (three shared words) outranks C (two)."""
    id_a, _id_b, _id_c = build_source(tmp_path / "sessions.db", [TEXT_A, TEXT_B, TEXT_C])
    write_config(tmp_path / "agent.yaml", EMBED_CONFIG)
    labels = write_labels(tmp_path / "labels.json", [
        {"text": "eta theta", "context": "beta gamma delta", "relevant": [id_a],
         "subset": "deictic"},
    ])

    with_context = run(script, tmp_path, labels, "--k", "1")
    baseline = run(script, tmp_path, labels, "--k", "1", "--query-context", "none")

    assert with_context["queries"][0]["recall_at_k"] == pytest.approx(1.0)
    assert baseline["queries"][0]["recall_at_k"] == pytest.approx(0.0)


@needs_vec
def test_summary_breaks_metrics_down_per_subset(
    script: ModuleType, stub_calls: StubCalls, private_tmp: Path, tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Each subset tag gets its own scored count and means; untagged probes
    are reported as their own group."""
    id_a, id_b, id_c = build_source(tmp_path / "sessions.db", [TEXT_A, TEXT_B, TEXT_C])
    write_config(tmp_path / "agent.yaml", EMBED_CONFIG)
    labels = write_labels(tmp_path / "labels.json", [
        {"text": TEXT_A, "relevant": [id_a], "subset": "self-contained"},
        {"text": TEXT_A, "relevant": [id_c], "subset": "self-contained"},
        {"text": TEXT_C, "relevant": [id_c], "subset": "deictic"},
        {"text": TEXT_B, "relevant": [id_b]},
    ])

    pool = run(script, tmp_path, labels, "--k", "1")

    subsets = pool["summary"]["subsets"]
    assert set(subsets) == {"self-contained", "deictic", "(untagged)"}
    assert subsets["self-contained"]["scored"] == 2
    assert subsets["self-contained"]["mean_recall_at_k"] == pytest.approx(0.5)
    assert subsets["deictic"]["mean_mrr"] == pytest.approx(1.0)
    assert subsets["(untagged)"]["scored"] == 1
    assert pool["queries"][0]["subset"] == "self-contained"
    out = capsys.readouterr().out
    assert "self-contained" in out and "deictic" in out


@pytest.mark.parametrize("field, value, expected", [
    ("context", 7, "queries[0].context: got 7; expected a string"),
    ("subset", ["a"], 'queries[0].subset: got ["a"]; expected a non-empty string'),
    ("subset", "", 'queries[0].subset: got ""; expected a non-empty string'),
])
def test_bad_context_or_subset_is_named(
    script: ModuleType, tmp_path: Path, field: str, value: object, expected: str,
) -> None:
    """A malformed optional field names the query, the value and the fix."""
    labels = write_labels(tmp_path / "labels.json", [
        {"text": "q", "relevant": [1], field: value},
    ])
    with pytest.raises(ValueError) as excinfo:
        script.load_labels(labels)
    message = str(excinfo.value)
    assert expected in message
    assert "fix the labels file" in message


def test_query_context_flag_needs_snapshot_and_a_known_value(
    script: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """--query-context outside snapshot mode, or with an unknown value, is refused."""
    with pytest.raises(SystemExit) as excinfo:
        script.main(["--query-context", "none"])
    assert excinfo.value.code == 2
    assert "--query-context is only used with --snapshot" in capsys.readouterr().err

    with pytest.raises(SystemExit) as excinfo:
        script.main([
            "--snapshot", str(tmp_path / "s.db"), "--labels", str(tmp_path / "l.json"),
            "--config", str(tmp_path / "a.yaml"), "--pool-out", str(tmp_path / "p.json"),
            "--query-context", "everything",
        ])
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "everything" in err and "previous_reply" in err


# #666: --candidates hybrid|vector scores fused vs vec-only on one snapshot.
NAMED = "Yoz prefers shipping releases on Thursdays"
DISTRACTORS = ["what about that one", "what about this", "what about it then",
               "and what about those", "so what about them"]


@needs_vec
def test_candidates_flag_scores_fused_against_vector_only(
    script: ModuleType, stub_calls: StubCalls, private_tmp: Path, tmp_path: Path,
) -> None:
    """A name probe the vector search misses at k=1 is found by the fused pool;
    each run records its candidate mode, and every candidate its source."""
    ids = build_source(tmp_path / "sessions.db", [*DISTRACTORS, NAMED])
    write_config(tmp_path / "agent.yaml", EMBED_CONFIG)
    labels = write_labels(tmp_path / "labels.json", [
        {"text": "what about Yoz", "relevant": [ids[-1]], "subset": "proper-noun"},
    ])

    fused = run(script, tmp_path, labels, "--k", "1")
    vector = run(script, tmp_path, labels, "--k", "1", "--candidates", "vector")

    assert fused["summary"]["candidates"] == "hybrid"
    assert vector["summary"]["candidates"] == "vector"
    assert fused["queries"][0]["recall_at_k"] == pytest.approx(1.0)
    assert vector["queries"][0]["recall_at_k"] == pytest.approx(0.0)
    assert fused["queries"][0]["candidates"][0]["source"] == "fts"
    assert {c["source"] for c in vector["queries"][0]["candidates"]} == {"vec"}


def test_candidates_flag_needs_snapshot(
    script: ModuleType, capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        script.main(["--candidates", "vector"])
    assert excinfo.value.code == 2
    assert "--candidates is only used with --snapshot" in capsys.readouterr().err
