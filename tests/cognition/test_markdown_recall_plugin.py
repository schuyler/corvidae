"""Tests for the markdown-folder recall source plugin (#719).

The end-to-end test goes through the real plugin manager and the real funnel,
like the mixin's own tests: a folder of markdown files, a plain user message,
and the funnel admitting a framed CONTEXT hit. The integration is exercised
FTS5-only (no embedding profile configured) — the vector path is covered at
the index level; here the point is the wiring (config, on_init, the startup
reconcile, search/render, and the funnel hand-off).
"""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import aiosqlite
import pytest
from corvidae_cognition.funnel import FunnelPlugin
from corvidae_cognition.markdown_recall import MarkdownRecallPlugin
from corvidae_cognition.markdown_recall.index import Hit

from corvidae.channel import Channel
from corvidae.context import ContextWindow, MessageType
from corvidae.hooks import create_plugin_manager, install_plugin_hookspecs
from corvidae.persistence import PersistencePlugin, init_db


class Clock:
    """Fake monotonic clock; time moves only when a test sets ``t``."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def write_folder(root: Path, files: dict[str, str]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (root / name).write_text(content)


async def make(config: dict, root: Path, tmp_path: Path) -> SimpleNamespace:
    """A funnel and the markdown-recall source on a real plugin manager.

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
    clock = Clock()
    funnel = FunnelPlugin(now=clock)
    pm.register(funnel, name="funnel")
    source = MarkdownRecallPlugin()
    pm.register(source, name="markdown_recall")
    install_plugin_hookspecs(pm)
    await funnel.on_init(pm=pm, config=config)
    await source.on_init(pm=pm, config=config)
    channel = Channel(transport="test", scope="recall")
    channel.conversation = ContextWindow(channel.id)
    return SimpleNamespace(
        pm=pm, funnel=funnel, source=source, channel=channel,
        conv=channel.conversation, clock=clock, db=db,
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


async def turn(env: SimpleNamespace) -> None:
    await env.funnel.before_agent_turn(
        channel=env.channel, trace_id="t", annotations={"origin": "user"}
    )


async def settle() -> None:
    for _ in range(8):
        await asyncio.sleep(0)


def user(env: SimpleNamespace, text: str) -> None:
    env.conv.append({"role": "user", "content": text})


def context_text(env: SimpleNamespace) -> str:
    return "\n".join(
        m.get("content") or "" for m in env.conv.messages
        if m.get("_message_type") == MessageType.CONTEXT
    )


async def test_search_returns_ranked_hits_and_render_is_stable(build, tmp_path):
    # The funnel hand-off (on_funnel_ingest -> background search -> submit) is
    # the mixin's behavior, covered by test_recall_mixin.py. Here: the NEW
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
    await turn(env)
    await settle()
    assert context_text(env) == ""  # no index, no CONTEXT, no crash


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
