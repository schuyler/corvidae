"""Tests for the markdown-folder recall source plugin (#719).

These tests go through the real plugin manager: a folder of markdown files and
a plain user message. The integration is exercised FTS5-only (no embedding
profile configured) — the vector path is covered at the index level; here the
point is the wiring (config, on_init, the startup reconcile, search/render, and
the ``before_compose`` contribution, which ``test_recall_contribution.py`` pins
for a folder with hits).
"""

from pathlib import Path
from types import SimpleNamespace

import aiosqlite
import pytest
from corvidae_cognition.markdown_recall import MarkdownRecallPlugin
from corvidae_cognition.markdown_recall.index import Hit

from corvidae.channel import Channel
from corvidae.context import ContextWindow
from corvidae.hooks import create_plugin_manager, install_plugin_hookspecs
from fake_clock import install_kernel_clock
from corvidae.persistence import PersistencePlugin, init_db


def write_folder(root: Path, files: dict[str, str]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (root / name).write_text(content)


async def make(config: dict, root: Path, tmp_path: Path) -> SimpleNamespace:
    """The markdown-recall source on a real plugin manager.

    ``db_path`` is always overridden to ``tmp_path`` so no test can write
    into the checkout's ``state/`` (the ``state_dir_guard`` autouse fixture
    fails on that).
    """
    config = dict(config)
    block = dict(config.get("markdown_recall", {}))
    block["db_path"] = str(tmp_path / "index.db")
    config["markdown_recall"] = block
    db = await aiosqlite.connect(":memory:")
    await init_db(db)
    pm = create_plugin_manager()
    persistence = PersistencePlugin()
    persistence.db = db
    pm.register(persistence, name="persistence")
    # Kernel clock and idle tracker MarkdownRecallPlugin.on_start resolves.
    install_kernel_clock(pm)
    source = MarkdownRecallPlugin()
    pm.register(source, name="markdown_recall")
    install_plugin_hookspecs(pm)
    await source.on_init(pm=pm, config=config)
    channel = Channel(transport="test", scope="recall")
    channel.conversation = ContextWindow(channel.id)
    return SimpleNamespace(
        pm=pm, source=source, channel=channel,
        conv=channel.conversation, db=db,
    )


@pytest.fixture
async def build(tmp_path):
    opened: list[aiosqlite.Connection] = []

    async def _make(config: dict, root: Path):
        env = await make(config, root, tmp_path)
        opened.append(env.db)
        return env

    yield _make
    for db in opened:
        await db.close()


async def turn(env: SimpleNamespace) -> object:
    """The turn's before_compose answer from the source."""
    return await env.source.before_compose(
        channel=env.channel, trace_id="t", annotations={"origin": "user"}
    )


def user(env: SimpleNamespace, text: str) -> None:
    env.conv.append({"role": "user", "content": text})


async def test_search_returns_ranked_hits_and_render_is_stable(build, tmp_path):
    # The before_compose hand-off is the mixin's behavior, covered by
    # test_recall_mixin.py. Here: the
    # wiring — the plugin's search against the index, and render's stable line.
    root = tmp_path / "notes"
    write_folder(root, {
        "radio.md": "# Waterfall\n\nThe waterfall display maps the spectrum over time.",
        "cooking.md": "# Other\n\nUnrelated content about fermentation and cooking.",
    })
    config = {"markdown_recall": {
        "root": str(root), "db_path": str(tmp_path / "index.db"),
    }}
    env = await build(config, root)
    await env.source._reconcile()  # build the index directly (deterministic)
    from corvidae_cognition.recall import RecallContext
    ctx = RecallContext(channel=env.channel, trace_id="t", annotations={})
    hits = await env.source.search("waterfall spectrum", ctx)
    assert hits, "expected at least one hit"
    assert any("waterfall" in h.text.lower() for h in hits)
    line = env.source.render(hits[0])
    assert line.startswith("[") and "waterfall" in line.lower()
    # Deterministic: a second identical search renders the same top line.
    again = await env.source.search("waterfall spectrum", ctx)
    assert env.source.render(again[0]) == line


async def test_reconcile_builds_the_index(build, tmp_path):
    root = tmp_path / "notes"
    write_folder(root, {"a.md": "# A\n\nalpha beta gamma content"})
    config = {"markdown_recall": {
        "root": str(root), "db_path": str(tmp_path / "index.db"),
    }}
    env = await build(config, root)
    await env.source.on_start(config)
    await env.source._startup_task
    db = env.source._index._db
    async with db.execute("SELECT count(*) AS n FROM manifest") as c:
        assert (await c.fetchone())["n"] == 1
    async with db.execute("SELECT count(*) AS n FROM chunks") as c:
        assert (await c.fetchone())["n"] >= 1


async def test_missing_folder_is_fail_soft(build, tmp_path):
    root = tmp_path / "does-not-exist"
    config = {"markdown_recall": {
        "root": str(root), "db_path": str(tmp_path / "index.db"),
    }}
    env = await build(config, root)
    await env.source.on_start(config)
    await env.source._startup_task  # must not raise
    user(env, "anything at all")
    assert await turn(env) is None  # no index, nothing contributed, no crash


async def test_config_validation_names_the_key(build, tmp_path):
    root = tmp_path / "notes"
    write_folder(root, {"a.md": "# A\n\ncontent"})
    config = {"markdown_recall": {
        "root": str(root), "db_path": str(tmp_path / "index.db"),
        "max_hits": "five",  # malformed
    }}
    with pytest.raises(ValueError) as exc:
        await make(config, root, tmp_path)
    msg = str(exc.value)
    assert "max_hits" in msg
    assert "five" in msg


def test_render_is_a_stable_line():
    plugin = MarkdownRecallPlugin()
    # The chunker embeds the section prefix into the text; render returns it
    # as-is (adding the prefix again would double it).
    text = "[Waterfall] The waterfall maps the spectrum."
    hit = Hit(
        rowid=1, path="a.md", section_path="Waterfall",
        chunk_index=0, text=text, score=1.0,
    )
    assert plugin.render(hit) == text


async def test_on_stop_closes_the_index(build, tmp_path):
    """#738: on_stop closes the aiosqlite connection so the worker thread
    can exit and the daemon doesn't hang at interpreter shutdown."""
    root = tmp_path / "notes"
    write_folder(root, {"a.md": "# A\n\nalpha content"})
    config = {"markdown_recall": {"root": str(root), "db_path": str(tmp_path / "index.db")}}
    env = await build(config, root)
    await env.source._reconcile()  # open the connection
    assert env.source._index._db is not None, "connection should be open"
    await env.source.on_stop()
    assert env.source._index._db is None, "connection should be closed"


async def test_unconfigured_source_is_inert_in_before_compose(tmp_path):
    """#739: an unconfigured plugin (no markdown_recall block) must not raise
    in before_compose. The gate is initialized, search returns None, and
    nothing is contributed.

    Bypasses the make() helper, which always injects a markdown_recall block
    for the db_path override.
    """
    db = await aiosqlite.connect(":memory:")
    await init_db(db)
    pm = create_plugin_manager()
    persistence = PersistencePlugin()
    persistence.db = db
    pm.register(persistence, name="persistence")
    # Kernel clock and idle tracker MarkdownRecallPlugin.on_start resolves.
    install_kernel_clock(pm)
    source = MarkdownRecallPlugin()
    pm.register(source, name="markdown_recall")
    install_plugin_hookspecs(pm)
    await source.on_init(pm=pm, config={})  # no markdown_recall block
    channel = Channel(transport="test", scope="recall")
    channel.conversation = ContextWindow(channel.id)

    # The gate must be initialized (not None), or before_compose raises.
    assert source._recall_gate is not None, "gate must be initialized"
    # The index must be None (inert).
    assert source._index is None, "index must be None when unconfigured"
    # before_compose must not raise, and contributes nothing.
    channel.conversation.append({"role": "user", "content": "hello"})
    assert await source.before_compose(
        channel=channel, trace_id="t", annotations={"origin": "user"}
    ) is None
    # on_stop on an inert plugin must not raise.
    await source.on_stop()
    await db.close()


async def test_on_stop_cancels_active_startup_task(build, tmp_path):
    """#738 review: on_stop must cancel AND await the startup task before
    closing the index. A cancel() without await leaves the task pending,
    and the reconcile can hit a closed connection."""
    root = tmp_path / "notes"
    write_folder(root, {"a.md": "# A\n\nalpha content"})
    config = {"markdown_recall": {"root": str(root), "db_path": str(tmp_path / "index.db")}}
    env = await build(config, root)
    # Start the plugin (creates the startup task).
    await env.source.on_start(config)
    assert env.source._startup_task is not None, "startup task should exist"
    # on_stop must cancel and await it.
    await env.source.on_stop()
    assert env.source._startup_task.done(), "startup task should be done"
    assert env.source._index._db is None, "connection should be closed"


async def test_roots_list_config(build, tmp_path):
    """markdown_recall.roots (list) is accepted, passed to the index,
    and reconciles correctly."""
    values = tmp_path / "values"
    skills = tmp_path / "skills"
    values.mkdir()
    skills.mkdir()
    (values / "test.md").write_text("# Test")
    (skills / "skill.md").write_text("# Skill")

    env = await build(
        {"markdown_recall": {"roots": [str(values), str(skills)]}},
        values,
    )
    assert isinstance(env.source._root, list)
    assert len(env.source._root) == 2
    # The index received the multi-root config
    assert len(env.source._index._roots) == 2
    # Reconcile works end-to-end through the plugin
    await env.source._index.reconcile()
    async with env.source._index._db.execute(
        "SELECT count(*) FROM manifest"
    ) as cur:
        (count,) = await cur.fetchone()
    assert count == 2
    await env.source.on_stop()


async def test_empty_roots_rejected(build, tmp_path):
    """markdown_recall.roots: [] is rejected with a clear error."""
    import pytest
    values = tmp_path / "values"
    values.mkdir()
    with pytest.raises(ValueError, match="non-empty"):
        await build({"markdown_recall": {"roots": []}}, values)


async def test_single_root_backward_compat(build, tmp_path):
    """markdown_recall.root (single string) still works after the config change."""
    values = tmp_path / "values"
    values.mkdir()
    (values / "test.md").write_text("# Test")

    env = await build(
        {"markdown_recall": {"root": str(values)}},
        values,
    )
    from pathlib import Path
    assert isinstance(env.source._root, Path)
    await env.source.on_stop()
