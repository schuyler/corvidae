"""markdown_recall's search hook (#1190 PR 1, criterion A8).

``pm.ahook.resolve_markdown_recall_search(query=...)`` answers a search over
the indexed roots for other plugins: rendered hits, or None when there is no
index or its schema is not built, in which case nothing is created.
"""

from pathlib import Path

from corvidae_cognition.markdown_recall import MarkdownRecallPlugin

from corvidae.hooks import create_plugin_manager, install_plugin_hookspecs

from .test_markdown_recall_plugin import make, write_folder


async def _env(tmp_path: Path, files: dict[str, str]):
    """An env over a notes folder."""
    root = tmp_path / "notes"
    write_folder(root, files)
    config = {"markdown_recall": {"root": str(root)}}
    return await make(config, root, tmp_path)


async def test_hook_returns_rendered_hits_once_reconciled(tmp_path):
    env = await _env(tmp_path, {
        "radio.md": "# Waterfall\n\nThe waterfall display maps the spectrum over time.",
        "cooking.md": "# Other\n\nUnrelated content about fermentation.",
    })
    await env.source._reconcile()

    result = await env.pm.ahook.resolve_markdown_recall_search(query="waterfall spectrum")

    assert isinstance(result, str)
    lines = result.splitlines()
    assert lines and all(": " in line for line in lines)
    assert any(line.split(": ", 1)[0].endswith("radio.md") for line in lines)
    assert any("waterfall" in line.lower() for line in lines)
    await env.source.on_stop()


async def test_hook_is_none_when_the_plugin_has_no_index():
    # make() always injects a markdown_recall block, so build the unconfigured
    # plugin by hand (as the #739 inert test does).
    pm = create_plugin_manager()
    source = MarkdownRecallPlugin()
    pm.register(source, name="markdown_recall")
    install_plugin_hookspecs(pm)
    await source.on_init(pm=pm, config={})
    assert source._index is None

    assert await pm.ahook.resolve_markdown_recall_search(query="anything") is None


async def test_hook_is_none_and_writes_nothing_when_schema_not_built(tmp_path):
    env = await _env(tmp_path, {"a.md": "# A\n\nalpha beta gamma"})
    db_file = tmp_path / "index.db"
    assert not db_file.exists()  # no reconcile has run

    result = await env.pm.ahook.resolve_markdown_recall_search(query="alpha")

    assert result is None
    # The call created no index file (and so no rows).
    assert not db_file.exists()
    await env.source.on_stop()
