"""RED tests for #1176: recall sources contribute through ``before_compose``.

Spec: docs/1176-gather-and-held-text-design.md, section 6 (callers,
RecallSourceMixin) and requirements R2.3, R3.1. ``on_funnel_ingest`` and its
promise go; the mixin's ``before_compose`` awaits the search itself (the
gather's allowance bounds the turn's wait) and returns
``Contribution(source=<source_name>, entries=<rendered hits>)``, or ``None``
when the gate says this is not a user turn. Markdown recall inherits it.

Criterion covered: R1 (this file's id; the cognition criteria list calls it R1).

Equivalence note: the kernel's gather calls ``before_compose`` on each
implementation with ``channel``, ``trace_id`` and ``annotations`` keywords and
reads the answer. These tests make that call on the real mixin and the real
markdown plugin with a window shaped as the loop leaves it (a plain user
message at the tail), so the source cannot tell them from the kernel.

``corvidae.held`` does not exist yet, so it is imported on use.
"""

import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import aiosqlite
import pytest
from corvidae_cognition.markdown_recall import MarkdownRecallPlugin
from corvidae_cognition.recall import RecallSourceMixin
from fake_clock import install_kernel_clock

from corvidae.channel import Channel
from corvidae.context import ContextWindow
from corvidae.hooks import CorvidaePlugin, create_plugin_manager, hookimpl
from corvidae.persistence import PersistencePlugin, init_db


def held_mod() -> Any:
    # Imported on use: corvidae.held is written in Green.
    import corvidae.held as held

    return held


class FakeSource(RecallSourceMixin, CorvidaePlugin):
    """A recall source on the mixin whose hits the test sets."""

    source_name = "fake"
    config_key = "fake_recall"

    def __init__(self) -> None:
        # Queries seen and the hits the next search returns.
        self.queries: list[str] = []
        self.hits: list[Any] | None = ["hit one", "hit two"]

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        # Store pm and config, then let the mixin read its settings and build its gate.
        await super().on_init(pm, config)
        self.init_recall(config)

    async def search(self, query: str, ctx: Any) -> list[Any] | None:
        # Record the query and answer with the configured hits.
        self.queries.append(query)
        return self.hits

    def render(self, hit: Any) -> str:
        # One stable line per hit.
        return f"rendered {hit}"


async def maybe_await(value: Any) -> Any:
    # A hookimpl may be sync or async; the kernel awaits it either way.
    if inspect.isawaitable(value):
        return await value
    return value


async def make_fake() -> SimpleNamespace:
    # The fake source on a real plugin manager, one channel, no funnel at all.
    pm = create_plugin_manager()
    source = FakeSource()
    pm.register(source, name="fake_recall")
    await source.on_init(pm=pm, config={})
    channel = Channel(transport="test", scope="recall")
    channel.conversation = ContextWindow(channel.id)
    return SimpleNamespace(pm=pm, source=source, channel=channel, conv=channel.conversation)


async def compose(env: SimpleNamespace) -> Any:
    # The kernel's before_compose call for the window as it now stands.
    return await maybe_await(
        env.source.before_compose(channel=env.channel, trace_id="trace-1", annotations={})
    )


class TestMixinContribution:
    """R1: the mixin answers the before-compose gather."""

    async def test_plain_user_turn_returns_the_rendered_hits(self):
        env = await make_fake()
        env.conv.append({"role": "user", "content": "what do my notes say?"})
        contribution = await compose(env)
        assert contribution == held_mod().Contribution(
            source="fake", entries=["rendered hit one", "rendered hit two"]
        )
        assert env.source.queries == ["what do my notes say?"]

    async def test_gated_turn_returns_none_and_does_not_search(self):
        env = await make_fake()
        # The window ends in an assistant reply, not a user message: gated.
        env.conv.append({"role": "user", "content": "what do my notes say?"})
        env.conv.append({"role": "assistant", "content": "they say nothing"})
        assert await compose(env) is None
        assert env.source.queries == []

    def test_on_funnel_ingest_is_gone_from_the_mixin_and_markdown_recall(self):
        assert not hasattr(RecallSourceMixin, "on_funnel_ingest")
        assert not hasattr(MarkdownRecallPlugin, "on_funnel_ingest")


def write_folder(root: Path, files: dict[str, str]) -> None:
    # Write ``files`` under ``root``.
    root.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (root / name).write_text(content)


class TestMarkdownRecall:
    """R1: markdown_recall works through the mixin's before_compose."""

    async def test_hit_from_the_folder_is_contributed_under_its_source_name(self, tmp_path):
        root = tmp_path / "notes"
        write_folder(root, {
            "radio.md": "# Waterfall\n\nThe waterfall display maps the spectrum over time.",
            "cooking.md": "# Other\n\nUnrelated content about fermentation and cooking.",
        })
        # db_path goes under tmp_path so nothing writes into the checkout's state/.
        config = {"markdown_recall": {"root": str(root), "db_path": str(tmp_path / "index.db")}}
        db = await aiosqlite.connect(":memory:")
        await init_db(db)
        try:
            pm = create_plugin_manager()
            persistence = PersistencePlugin()
            persistence.db = db
            pm.register(persistence, name="persistence")
            install_kernel_clock(pm)
            source = MarkdownRecallPlugin()
            pm.register(source, name="markdown_recall")
            await source.on_init(pm=pm, config=config)
            # Build the index directly, as the plugin's own tests do (deterministic).
            await source._reconcile()
            channel = Channel(transport="test", scope="recall")
            channel.conversation = ContextWindow(channel.id)
            channel.conversation.append({"role": "user", "content": "waterfall spectrum"})
            contribution = await maybe_await(
                source.before_compose(channel=channel, trace_id="trace-1", annotations={})
            )
            assert contribution is not None
            assert contribution.source == "markdown_recall"
            assert any("waterfall" in entry.lower() for entry in contribution.entries)
        finally:
            await db.close()
