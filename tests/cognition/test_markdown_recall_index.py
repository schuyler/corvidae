"""Tests for the markdown-folder recall index (#719).

Covers the schema (the vector surface only when an encoder + vec are
available), reconcile (new/unchanged/edited/missing files, the unmounted-folder
safety), and search (FTS5-only, and vec KNN + FTS5 fused by RRF). A fake
embedding client stands in for the ``llm`` stanza's ``embedding`` stage.
"""

import hashlib

import pytest
from corvidae_cognition.markdown_recall.index import (
    Hit,
    MarkdownIndex,
    _EncoderIdentity,
    _fts_match,
)


class FakeEmbed:
    """Deterministic stand-in for the embedding stage's client."""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self.calls = 0

    async def embed(self, texts: list[str], kind: str = "document") -> list[list[float]]:
        self.calls += 1
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            out.append([b / 255.0 for b in h[: self.dim]])
        return out


IDENTITY = _EncoderIdentity(
    model="fake-encoder", dimensions=8, document_prefix="", query_prefix=""
)


async def make_index(tmp_path, files, identity=IDENTITY):
    root = tmp_path / "notes"
    root.mkdir(parents=True)
    for name, content in files.items():
        (root / name).write_text(content)
    dim = identity.dimensions if identity else 8
    embed = FakeEmbed(dim=dim)
    index = MarkdownIndex(
        root=root,
        db_path=tmp_path / "index.db",
        get_client=lambda: embed,
        get_identity=lambda: identity,
    )
    return index, root, embed


async def _tables(db) -> set[str]:
    async with db.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
    ) as cursor:
        return {row[0] for row in await cursor.fetchall()}


def _vec_importable() -> bool:
    import importlib.util
    return importlib.util.find_spec("sqlite_vec") is not None


@pytest.mark.parametrize(
    "identity,expect_vec",
    [(IDENTITY, _vec_importable()), (None, False)],
)
async def test_schema_creates_the_tables(tmp_path, identity, expect_vec):
    index, _, _ = await make_index(tmp_path, {"a.md": "# A\n\nalpha content"}, identity)
    db = await index._ensure_schema()
    tables = await _tables(db)
    assert {"chunks", "chunks_fts", "manifest", "embedding_meta"} <= tables
    assert ("chunks_vec" in tables) == expect_vec
    await index.close()


async def test_reconcile_indexes_new_files(tmp_path):
    index, _, embed = await make_index(
        tmp_path, {"a.md": "# A\n\nalpha beta gamma", "b.md": "# B\n\nbravo charlie"}
    )
    await index.reconcile()
    db = index._db
    async with db.execute("SELECT count(*) AS n FROM manifest") as c:
        assert (await c.fetchone())["n"] == 2
    async with db.execute("SELECT count(*) AS n FROM chunks") as c:
        assert (await c.fetchone())["n"] >= 2
    # The manifest records the real file size (the stat shortcut depends on it).
    async with db.execute("SELECT size FROM manifest") as c:
        sizes = [r[0] for r in await c.fetchall()]
    assert all(sz > 0 for sz in sizes), f"expected non-zero sizes, got {sizes}"
    assert embed.calls >= 2
    await index.close()


async def test_reconcile_skips_unchanged_files(tmp_path):
    index, _, embed = await make_index(tmp_path, {"a.md": "# A\n\nalpha beta"})
    await index.reconcile()
    calls = embed.calls
    await index.reconcile()  # no change
    assert embed.calls == calls  # nothing re-embedded
    await index.close()


async def test_reconcile_rechunks_on_edit(tmp_path):
    index, root, embed = await make_index(tmp_path, {"a.md": "# A\n\nalpha beta"})
    await index.reconcile()
    root.joinpath("a.md").write_text("# A\n\ncompletely different delta content")
    await index.reconcile()
    db = index._db
    async with db.execute("SELECT text FROM chunks WHERE path='a.md'") as c:
        rows = [r[0] for r in await c.fetchall()]
    assert any("delta" in r for r in rows)
    assert embed.calls > 1
    await index.close()


async def test_reconcile_deletes_missing_files(tmp_path):
    index, root, _ = await make_index(
        tmp_path, {"a.md": "# A\n\nalpha", "b.md": "# B\n\nbravo"}
    )
    await index.reconcile()
    root.joinpath("b.md").unlink()
    await index.reconcile()
    db = index._db
    async with db.execute("SELECT path FROM manifest") as c:
        paths = {r[0] for r in await c.fetchall()}
    assert paths == {"a.md"}
    await index.close()


async def test_unmounted_folder_keeps_the_index(tmp_path):
    index, _, _ = await make_index(tmp_path, {"a.md": "# A\n\nalpha beta"})
    await index.reconcile()
    empty = tmp_path / "empty"
    empty.mkdir()
    index._root = empty  # simulate the folder vanishing
    await index.reconcile()
    db = index._db
    async with db.execute("SELECT count(*) AS n FROM manifest") as c:
        assert (await c.fetchone())["n"] == 1  # kept, not mass-deleted
    await index.close()


async def test_search_fts_only_ranks_the_match_first(tmp_path):
    index, _, _ = await make_index(
        tmp_path,
        {
            "a.md": "# Radio\n\nthe waterfall is a map of the spectrum",
            "b.md": "# Other\n\nunrelated content about cooking",
        },
        identity=None,  # FTS5-only
    )
    await index.reconcile()
    hits = await index.search("waterfall spectrum", k=3)
    assert hits, "expected at least one hit"
    assert hits[0].path == "a.md"
    await index.close()


async def test_search_vec_plus_fts_fuses_and_ranks(tmp_path):
    index, _, embed = await make_index(
        tmp_path,
        {
            "a.md": "# Radio\n\nthe waterfall is a map of the spectrum",
            "b.md": "# Other\n\nunrelated content about cooking",
        },
    )
    await index.reconcile()
    hits = await index.search("waterfall spectrum", k=3)
    assert hits
    assert hits[0].path == "a.md"
    assert isinstance(hits[0], Hit)
    assert hits[0].score > 0
    await index.close()


async def test_search_empty_index_returns_nothing(tmp_path):
    index, _, _ = await make_index(tmp_path, {})
    await index.reconcile()
    assert await index.search("anything", k=3) == []
    await index.close()


def test_fts_match_sanitizes_and_ors():
    assert _fts_match("hello world") == '"hello"* OR "world"*'
    assert _fts_match("") == ""
    assert _fts_match("   ") == ""
    assert _fts_match("spectrum") == '"spectrum"*'


def test_fts_match_escapes_double_quotes():
    # A `"` in a token is doubled so the MATCH stays valid FTS5.
    assert _fts_match('foo"bar') == '"foo""bar"*'
    # a" -> a"" (the quote doubles), wrapped -> "a"""*
    assert _fts_match('a" b"c') == '"a"""* OR "b""c"*'


class TestMultiRoot:
    """Multi-root: multiple directories, namespaced paths."""

    async def _make_multi(self, tmp_path, root_names, files_per_root):
        """Create a multi-root index with the given roots and files."""
        roots = []
        for name, files in zip(root_names, files_per_root):
            root = tmp_path / name
            root.mkdir(parents=True, exist_ok=True)
            for fname, fcontent in files.items():
                (root / fname).write_text(fcontent)
            roots.append(root)
        dim = IDENTITY.dimensions
        embed = FakeEmbed(dim=dim)
        index = MarkdownIndex(
            root=roots,
            db_path=tmp_path / "index.db",
            get_client=lambda: embed,
            get_identity=lambda: IDENTITY,
        )
        return index, roots, embed

    async def test_two_roots_namespaced(self, tmp_path):
        """Two roots: paths are prefixed with the root basename."""
        idx, roots, _ = await self._make_multi(
            tmp_path,
            ["values", "skills"],
            [{"radio.md": "# Radio\n\nSome content."},
             {"radio.md": "# Radio Skill\n\nDifferent content."}],
        )
        await idx.reconcile()
        files = idx._walk()
        assert "values/radio.md" in files
        assert "skills/radio.md" in files
        # Manifest has rows for both files (proves _resolve_path worked)
        async with idx._db.execute("SELECT count(*) FROM manifest") as cur:
            (count,) = await cur.fetchone()
        assert count == 2, f"expected 2 manifest rows, got {count}"
        await idx.close()

    async def test_single_root_unnamespaced(self, tmp_path):
        """Single root (list of one): paths are NOT prefixed (backward compat)."""
        idx, roots, _ = await self._make_multi(
            tmp_path,
            ["values"],
            [{"radio.md": "# Radio\n\nContent."}],
        )
        await idx.reconcile()
        files = idx._walk()
        assert "radio.md" in files
        assert "values/radio.md" not in files
        await idx.close()

    async def test_nonexistent_root_aborts_walk(self, tmp_path):
        """A non-existent root aborts the walk (keeps the old index)."""
        idx, roots, _ = await self._make_multi(
            tmp_path,
            ["values"],
            [{"radio.md": "# Radio\n\nContent."}],
        )
        # Add a non-existent root
        idx._roots.append(tmp_path / "nonexistent")
        files = idx._walk()
        assert files is None  # any root unreadable: keep the old index
        await idx.close()

    async def test_same_basename_rejected(self, tmp_path):
        """Two roots with the same basename are rejected at construction."""
        root_a = tmp_path / "work" / "values"
        root_b = tmp_path / "personal" / "values"
        root_a.mkdir(parents=True)
        root_b.mkdir(parents=True)
        (root_a / "a.md").write_text("# A")
        (root_b / "b.md").write_text("# B")
        import pytest
        with pytest.raises(ValueError, match="duplicate root basename"):
            MarkdownIndex(
                root=[root_a, root_b],
                db_path=tmp_path / "index.db",
                get_client=lambda: FakeEmbed(),
                get_identity=lambda: IDENTITY,
            )

    async def test_resolve_path_missing_prefix(self, tmp_path):
        """A namespaced path with no matching root prefix is rejected (multi-root)."""
        idx, roots, _ = await self._make_multi(
            tmp_path,
            ["values", "skills"],
            [{"radio.md": "# Radio\n\nContent."},
             {"skill.md": "# Skill\n\nContent."}],
        )
        import pytest
        with pytest.raises(FileNotFoundError, match="no root matches prefix"):
            idx._resolve_path("unknown/x.md")
        await idx.close()

    async def test_resolve_path_bare_name_multi_root(self, tmp_path):
        """A bare name (no slash) in multi-root mode is rejected."""
        idx, roots, _ = await self._make_multi(
            tmp_path,
            ["values", "skills"],
            [{"radio.md": "# Radio\n\nContent."},
             {"skill.md": "# Skill\n\nContent."}],
        )
        import pytest
        with pytest.raises(FileNotFoundError, match="has no root prefix"):
            idx._resolve_path("radio.md")
        await idx.close()

    async def test_present_but_empty_root_keeps_index(self, tmp_path):
        """A root that exists but is empty (transient unmount) keeps its index rows."""
        idx, roots, _ = await self._make_multi(
            tmp_path,
            ["values", "skills"],
            [{"radio.md": "# Radio\n\nContent."},
             {"skill.md": "# Skill\n\nContent."}],
        )
        await idx.reconcile()
        async with idx._db.execute("SELECT count(*) FROM manifest") as cur:
            (count1,) = await cur.fetchone()
        assert count1 == 2

        # Empty the skills root (directory still exists, no files)
        (roots[1] / "skill.md").unlink()
        # Reconcile should keep the old index (per-root emptiness check)
        await idx.reconcile()
        async with idx._db.execute("SELECT count(*) FROM manifest") as cur:
            (count2,) = await cur.fetchone()
        assert count2 == 2, "present-but-empty root should not prune its rows"
        await idx.close()

    async def test_overlapping_roots_rejected(self, tmp_path):
        """Nested/overlapping roots are rejected at construction."""
        import pytest
        a = tmp_path / "a"
        b = a / "b"
        a.mkdir()
        b.mkdir()
        with pytest.raises(ValueError, match="overlap"):
            MarkdownIndex(
                root=[a, b],
                db_path=tmp_path / "index.db",
                get_client=lambda: FakeEmbed(),
                get_identity=lambda: IDENTITY,
            )

    async def test_all_roots_unreadable(self, tmp_path):
        """All roots unreadable: _walk returns None."""
        idx = MarkdownIndex(
            root=[tmp_path / "a", tmp_path / "b"],
            db_path=tmp_path / "index.db",
            get_client=lambda: FakeEmbed(),
            get_identity=lambda: IDENTITY,
        )
        assert idx._walk() is None
        await idx.close()
    async def test_resolve_path_traversal_rejected(self, tmp_path):
        """_resolve_path rejects path traversal that escapes the matched root."""
        idx, roots, _ = await self._make_multi(
            tmp_path,
            ["values", "skills"],
            [{"radio.md": "# Radio\n\nContent."},
             {"secret.md": "# Secret\n\nHidden."}],
        )
        # values/../skills/secret.md resolves under skills/, but the
        # namespace says values/ — must be rejected.
        import pytest
        with pytest.raises(FileNotFoundError, match="escapes the root"):
            idx._resolve_path("values/../skills/secret.md")
        # Traversal through a non-existent intermediate directory
        with pytest.raises(FileNotFoundError, match="escapes the root"):
            idx._resolve_path("values/../tmp/skills/secret.md")
        await idx.close()

    async def test_nested_multi_root_paths(self, tmp_path):
        """Nested paths within a root are namespaced correctly."""
        root = tmp_path / "values"
        root.mkdir(parents=True)
        sub = root / "sub"
        sub.mkdir()
        (sub / "radio.md").write_text("# Nested\n\nContent.")
        (root / "top.md").write_text("# Top\n\nContent.")
        dim = IDENTITY.dimensions
        embed = FakeEmbed(dim=dim)
        idx = MarkdownIndex(
            root=[root, tmp_path / "skills"],
            db_path=tmp_path / "index.db",
            get_client=lambda: embed,
            get_identity=lambda: IDENTITY,
        )
        (tmp_path / "skills").mkdir()
        await idx.reconcile()
        files = idx._walk()
        assert "values/sub/radio.md" in files
        assert "values/top.md" in files
        # _resolve_path round-trips the nested path
        resolved = idx._resolve_path("values/sub/radio.md")
        assert resolved == sub / "radio.md"
        await idx.close()

    async def test_reconcile_missing_root_keeps_index(self, tmp_path):
        """reconcile() with a missing root keeps the old index (no pruning)."""
        idx, roots, _ = await self._make_multi(
            tmp_path,
            ["values", "skills"],
            [{"radio.md": "# Radio\n\nContent."},
             {"skill.md": "# Skill\n\nContent."}],
        )
        await idx.reconcile()
        # Verify both files are indexed
        async with idx._db.execute("SELECT count(*) FROM manifest") as cur:
            (count,) = await cur.fetchone()
        assert count == 2
        # Remove one root, then reconcile
        import shutil
        shutil.rmtree(tmp_path / "skills")
        await idx.reconcile()
        # The old index is kept (no pruning of the missing root's files)
        async with idx._db.execute("SELECT count(*) FROM manifest") as cur:
            (count,) = await cur.fetchone()
        assert count == 2, "missing root should not prune the old index"
        await idx.close()

    async def test_empty_string_root_rejected(self, tmp_path):
        """Empty-string roots are rejected at construction."""
        import pytest
        with pytest.raises(ValueError, match="is empty"):
            MarkdownIndex(
                root=["", "values"],
                db_path=tmp_path / "index.db",
                get_client=lambda: FakeEmbed(),
                get_identity=lambda: IDENTITY,
            )

    async def test_root_with_empty_basename_rejected(self, tmp_path):
        """A root that resolves to '/' (empty basename) is rejected."""
        import pytest
        with pytest.raises(ValueError, match="empty basename"):
            MarkdownIndex(
                root=["/", "values"],
                db_path=tmp_path / "index.db",
                get_client=lambda: FakeEmbed(),
                get_identity=lambda: IDENTITY,
            )



async def test_reconcile_does_not_hold_write_lock_across_embedding(tmp_path):
    """No embedding call may run while the index write lock is held (#1211).

    Reproduces the 2026-11-22 wedge in miniature. Measured on this host
    (Python 3.14.3 / sqlite 3.50.4, aiosqlite's default ``isolation_level=''``)
    a single DML statement opens an implicit write transaction that stays open
    until COMMIT: an external writer with ``timeout=0.0`` answers
    ``database is locked``. Two paths reach that state -- ``_ensure_schema``
    stamps ``embedding_meta`` and ``_rebuild_all`` deletes every row before its
    first embedding call, and ``_reconcile_file``'s mtime-only touch returns
    without committing, so the next file's embedding runs under that open
    transaction. Either one can hold the lock for the length of a model call
    (the live incident: 646 s for a 49-text embed), and every other writer to
    this database gives up at its busy timeout. Each embed probes the lock from
    a second connection: ``acquired`` is clean, ``locked`` is the defect.
    """
    import aiosqlite

    db_path = tmp_path / "index.db"
    root = tmp_path / "notes"
    root.mkdir()
    (root / "a.md").write_text("# A\n\nalpha beta gamma")
    (root / "b.md").write_text("# B\n\nbravo charlie delta echo")
    (root / "c.md").write_text("# C\n\ncharlie delta echo")
    (root / "d.md").write_text("# D\n\ndelta echo foxtrot")

    probe: list[str] = []

    async def _probe_write_lock(path) -> str:
        """'acquired' when no write lock is held; 'locked' when it is."""
        other = await aiosqlite.connect(str(path), timeout=0.0)
        try:
            try:
                await other.execute(
                    "UPDATE manifest SET mtime_ns = mtime_ns WHERE path = ?",
                    ("__probe__",),
                )
                await other.commit()
            except Exception as exc:  # noqa: BLE001 - recorded as the finding
                if "locked" in str(exc) or "busy" in str(exc):
                    return "locked"
                raise
        finally:
            await other.close()
        return "acquired"

    class ProbingEmbed(FakeEmbed):
        async def embed(self, texts: list[str], kind: str = "document"):
            outcome = await super().embed(texts, kind)
            probe.append(await _probe_write_lock(db_path))
            return outcome

    embed = ProbingEmbed()
    identity = _EncoderIdentity(
        model="fake-encoder", dimensions=8, document_prefix="", query_prefix=""
    )
    index = MarkdownIndex(
        root=root,
        db_path=db_path,
        get_client=lambda: embed,
        get_identity=lambda: identity,
    )
    await index.reconcile()  # pass 1: every file indexed
    await index.close()
    probe.clear()
    embed.calls = 0

    # Scenario 1, the rebuild path. _ensure_schema() latches the identity on
    # first connect, so an encoder change is seen only by a fresh open -- which
    # is also how it happens in the daemon, after a restart with a reconfigured
    # embedding profile.
    identity = _EncoderIdentity(
        model="other-encoder", dimensions=8, document_prefix="", query_prefix=""
    )
    index = MarkdownIndex(
        root=root,
        db_path=db_path,
        get_client=lambda: embed,
        get_identity=lambda: identity,
    )
    await index.reconcile()
    # Exactly one embed per file: the rebuild used to run its per-file loop
    # twice, so every file was embedded twice (#1211 review round 1). The
    # count is what catches that, not the row counts -- the second loop
    # rewrote the same rows.
    assert embed.calls == 4, f"expected one embed per file (4), calls={embed.calls}"
    assert probe and all(p == "acquired" for p in probe), (
        f"embedding held the write lock (rebuild path) in "
        f"{sum(1 for p in probe if p == 'locked')} of {len(probe)} calls: {probe}"
    )

    # Scenario 2, the incremental path: an mtime-only touch commits no work and
    # returns, so without a commit the next edited file's embedding call runs
    # inside that write transaction. Touched and edited files alternate so the
    # defect is caught whatever order the walk yields.
    probe.clear()
    embed.calls = 0
    for name, content in (
        ("a.md", "# A\n\nalpha beta gamma hotel"),
        ("b.md", "# B\n\nbravo charlie delta india"),
        ("c.md", None),
        ("d.md", "# D\n\ndelta echo foxtrot juliet"),
    ):
        if content is None:
            (root / name).touch()
        else:
            (root / name).write_text(content)
    await index.reconcile()
    await index.close()
    assert embed.calls >= 1, f"scenario 2 expected an embed, calls={embed.calls}"
    assert probe and all(p == "acquired" for p in probe), (
        f"embedding held the write lock (mtime-touch path) in "
        f"{sum(1 for p in probe if p == 'locked')} of {len(probe)} calls: {probe}"
    )


async def test_rebuild_keeps_existing_chunks_readable(tmp_path):
    """A rebuild must not leave readers on an empty index (corvidae#1211).

    The first shape of the fix committed ``DELETE FROM chunks`` before
    re-embedding, which is worse than the lock it removed: in WAL mode a
    read-only transaction keeps seeing the pre-transaction snapshot, so the old
    uncommitted wipe let readers see the whole previous index for the length of
    a rebuild, while a committed wipe shows them nothing. And ``_embed``
    swallows provider failures, so wiping first can empty the index with a
    broken model and nothing to rebuild from. Rows are now replaced per file by
    ``_replace_file_chunks``, so this asserts the previous index is still there
    mid-rebuild, counted from a second connection.
    """
    import aiosqlite

    db_path = tmp_path / "index.db"
    root = tmp_path / "notes"
    root.mkdir()
    for name in ("a.md", "b.md", "c.md"):
        (root / name).write_text(f"# {name}\n\n{name} content words here")

    state = {"identity": _EncoderIdentity(
        model="fake-encoder", dimensions=8, document_prefix="", query_prefix=""
    )}
    seen: list[int] = []

    class ReadingEmbed(FakeEmbed):
        async def embed(self, texts: list[str], kind: str = "document"):
            outcome = await super().embed(texts, kind)
            other = await aiosqlite.connect(str(db_path))
            try:
                async with other.execute("SELECT count(*) FROM chunks") as cursor:
                    seen.append((await cursor.fetchone())[0])
            finally:
                await other.close()
            return outcome

    embed = ReadingEmbed()
    index = MarkdownIndex(
        root=root,
        db_path=db_path,
        get_client=lambda: embed,
        get_identity=lambda: state["identity"],
    )
    await index.reconcile()
    await index.close()
    before = len(seen)
    assert before >= 3, f"pass 1 should have indexed every file, probes={seen}"

    seen.clear()
    state["identity"] = _EncoderIdentity(
        model="other-encoder", dimensions=8, document_prefix="", query_prefix=""
    )
    index = MarkdownIndex(
        root=root,
        db_path=db_path,
        get_client=lambda: embed,
        get_identity=lambda: state["identity"],
    )
    await index.reconcile()
    await index.close()
    assert seen and all(count >= 3 for count in seen), (
        f"a reader saw the index partly or fully wiped mid-rebuild: "
        f"{seen} (pass-1 probes={before})"
    )
