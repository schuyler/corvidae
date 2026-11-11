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

    async def test_nonexistent_root_skipped(self, tmp_path):
        """A non-existent root is skipped; the other root still works."""
        idx, roots, _ = await self._make_multi(
            tmp_path,
            ["values"],
            [{"radio.md": "# Radio\n\nContent."}],
        )
        # Add a non-existent root
        idx._roots.append(tmp_path / "nonexistent")
        files = idx._walk()
        assert files is not None
        assert "values/radio.md" in files  # multi-root mode (2 roots in list)
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
        await idx.close() if False else None

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
