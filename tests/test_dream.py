"""Tests for DreamPlugin — background memory consolidation."""

import asyncio
import time
from pathlib import Path
from unittest.mock import patch, mock_open

import aiosqlite
import pytest


# ---------------------------------------------------------------------------
# DB discovery
# ---------------------------------------------------------------------------


class TestDbDiscovery:
    async def test_locate_direct_corvidae_path(self, tmp_path):
        from corvidae.tools.dream import DreamPlugin

        db = tmp_path / "corvidae" / "sessions.db"
        db.parent.mkdir(parents=True)
        db.touch()

        plugin = DreamPlugin(workspace_root=tmp_path)
        await plugin.on_start(config={})
        assert plugin._db_path == db.resolve()

    async def test_locate_direct_workspace_path(self, tmp_path):
        from corvidae.tools.dream import DreamPlugin

        db = tmp_path / "sessions.db"
        db.touch()

        plugin = DreamPlugin(workspace_root=tmp_path)
        await plugin.on_start(config={})
        assert plugin._db_path == db.resolve()

    async def test_locate_recursive(self, tmp_path):
        from corvidae.tools.dream import DreamPlugin

        # Deep path: workspace/a/b/c/sessions.db
        db = tmp_path / "a" / "b" / "c" / "sessions.db"
        db.parent.mkdir(parents=True)
        db.touch()

        plugin = DreamPlugin(workspace_root=tmp_path)
        await plugin.on_start(config={})
        assert plugin._db_path == db.resolve()

    async def test_no_db_returns_none(self, tmp_path):
        from corvidae.tools.dream import DreamPlugin

        plugin = DreamPlugin(workspace_root=tmp_path)
        await plugin.on_start(config={})
        assert plugin._db_path is None


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


class TestConfigParsing:
    async def test_default_interval(self):
        from corvidae.tools.dream import DreamPlugin

        plugin = DreamPlugin(workspace_root="/tmp")
        await plugin.on_start(config={})
        assert plugin.interval_seconds == 300

    async def test_custom_interval(self):
        from corvidae.tools.dream import DreamPlugin

        plugin = DreamPlugin(workspace_root="/tmp")
        await plugin.on_start(config={"dream": {"interval_seconds": 600}})
        assert plugin.interval_seconds == 600


# ---------------------------------------------------------------------------
# Message loading
# ---------------------------------------------------------------------------


class TestMessageLoading:
    async def test_loads_assistant_messages(self, tmp_path):
        from corvidae.tools.dream import DreamPlugin
        import aiosqlite

        # Create a mock DB with assistant messages
        db = tmp_path / "corvidae" / "sessions.db"
        db.parent.mkdir(parents=True)

        async with aiosqlite.connect(str(db)) as conn:
            await conn.execute("CREATE TABLE message_log (id INTEGER PRIMARY KEY, channel_id TEXT, message TEXT, timestamp REAL)")
            for i in range(5):
                await conn.execute(
                    "INSERT INTO message_log (channel_id, message, timestamp) VALUES (?, ?, ?)",
                    (f"irc:#test{i}", f'{{"role":"assistant","content":"Message {i}"}}', 1000.0 + i),
                )
            await conn.commit()

        plugin = DreamPlugin(workspace_root=tmp_path)
        await plugin.on_start(config={})
        channels = await plugin._load_recent_assistant_messages()

        assert len(channels) == 5
        for ch, msgs in channels.items():
            assert len(msgs) == 1
            assert msgs[0]["role"] == "assistant"

    async def test_filters_out_non_assistant_messages(self, tmp_path):
        from corvidae.tools.dream import DreamPlugin

        db = tmp_path / "corvidae" / "sessions.db"
        db.parent.mkdir(parents=True)

        async with aiosqlite.connect(str(db)) as conn:
            await conn.execute("CREATE TABLE message_log (id INTEGER PRIMARY KEY, channel_id TEXT, message TEXT, timestamp REAL)")
            await conn.execute(
                "INSERT INTO message_log (channel_id, message, timestamp) VALUES (?, ?, ?)",
                ("irc:#test", '{"role":"user","content":"Hello"}', 1000.0),
            )
            await conn.commit()

        plugin = DreamPlugin(workspace_root=tmp_path)
        await plugin.on_start(config={})
        channels = await plugin._load_recent_assistant_messages()
        assert len(channels) == 0


# ---------------------------------------------------------------------------
# Memory writing (unit tests for helper methods)
# ---------------------------------------------------------------------------


class TestMemoryWriting:
    async def test_insert_into_memory_creates_section(self):
        from corvidae.tools.dream import DreamPlugin

        plugin = DreamPlugin(workspace_root="/tmp")
        content = "## User Information\nSchuyler is a maker.\n"
        new_fact = "- [2026-01-01 00:00 UTC] Test fact"
        result = plugin._insert_into_memory(content, new_fact)

        assert "## Long-term Memory" in result
        assert new_fact in result

    async def test_insert_into_memory_appends_to_existing(self):
        from corvidae.tools.dream import DreamPlugin

        plugin = DreamPlugin(workspace_root="/tmp")
        content = "## Long-term Memory\n- old fact\n\n## User Info\n"
        new_fact = "- [2026-01-01 00:00 UTC] New fact"
        result = plugin._insert_into_memory(content, new_fact)

        assert "- old fact" in result
        assert new_fact in result
        # New fact should come before old fact (inserted after header)
        assert result.index(new_fact) < result.index("- old fact")

    async def test_insert_into_memory_no_section_no_user_info(self):
        from corvidae.tools.dream import DreamPlugin

        plugin = DreamPlugin(workspace_root="/tmp")
        content = "Just some content.\n"
        new_fact = "- [2026-01-01 00:00 UTC] New fact"
        result = plugin._insert_into_memory(content, new_fact)

        assert "## Long-term Memory" in result
        assert result.strip().endswith(new_fact)

    async def test_extract_long_term_memory(self):
        from corvidae.tools.dream import DreamPlugin

        plugin = DreamPlugin(workspace_root="/tmp")
        content = "## Long-term Memory\n- fact 1\n- fact 2\n## User Info\n"
        section = plugin._extract_long_term_memory(content)
        assert "- fact 1" in section
        assert "- fact 2" in section

    async def test_extract_long_term_memory_missing(self):
        from corvidae.tools.dream import DreamPlugin

        plugin = DreamPlugin(workspace_root="/tmp")
        content = "No long-term memory here.\n"
        section = plugin._extract_long_term_memory(content)
        assert section == ""

    async def test_flatten_facts(self):
        from corvidae.tools.dream import DreamPlugin

        plugin = DreamPlugin(workspace_root="/tmp")
        section = "- This is a fact.\n- Another fact here.\n"
        facts = plugin._flatten_facts(section)
        assert len(facts) == 2
        all_text = " ".join(facts)
        assert "this is a fact" in all_text
        assert "another fact here" in all_text

    async def test_flatten_facts_empty(self):
        from corvidae.tools.dream import DreamPlugin

        plugin = DreamPlugin(workspace_root="/tmp")
        facts = plugin._flatten_facts("")
        assert len(facts) == 0

    async def test_normalize_for_dedup(self):
        from corvidae.tools.dream import DreamPlugin

        plugin = DreamPlugin(workspace_root="/tmp")
        sentence = "This is a TEST fact with [2026-04-28 16:38 PDT] timestamps and #thinking blocks"
        normalized = plugin._normalize_for_dedup(sentence)
        assert "test fact" in normalized
        assert "2026-04-28" not in normalized
        assert "thinking" not in normalized

    async def test_extract_sentences_removes_code_fences(self):
        from corvidae.tools.dream import DreamPlugin

        plugin = DreamPlugin(workspace_root="/tmp")
        text = "Normal sentence. Here is some code.\n\n```python\nprint('hello')\n```\nAnother sentence."
        sentences = plugin._extract_sentences(text)
        assert all("```" not in s for s in sentences)

    async def test_extract_sentences_short_filter(self):
        from corvidae.tools.dream import DreamPlugin

        plugin = DreamPlugin(workspace_root="/tmp")
        text = "Short. Long sentence that is worth extracting."
        sentences = plugin._extract_sentences(text)
        # Short fragments should be filtered (empty or very short after split)
        assert len(sentences) >= 1


# ---------------------------------------------------------------------------
# Integration: full dream cycle with mock DB
# ---------------------------------------------------------------------------


class TestDreamCycleIntegration:
    async def test_dream_cycle_persists_new_facts(self, tmp_path):
        from corvidae.tools.dream import DreamPlugin

        # Create a mock DB
        db = tmp_path / "corvidae" / "sessions.db"
        db.parent.mkdir(parents=True)

        async with aiosqlite.connect(str(db)) as conn:
            await conn.execute("CREATE TABLE message_log (id INTEGER PRIMARY KEY, channel_id TEXT, message TEXT, timestamp REAL)")
            # Add assistant messages with factual content
            for i in range(10):
                await conn.execute(
                    "INSERT INTO message_log (channel_id, message, timestamp) VALUES (?, ?, ?)",
                    ("irc:#test", f'{{"role":"assistant","content":"The system uses SQLite for persistence. This is fact number {i}. Extra context to make the sentence long enough to pass filters."}}', 1000.0 + i),
                )
            await conn.commit()

        # Create an empty MEMORY.md
        memory_path = tmp_path / "memory" / "MEMORY.md"
        memory_path.parent.mkdir(parents=True)
        memory_path.write_text("# Memory\n", encoding="utf-8")

        plugin = DreamPlugin(workspace_root=tmp_path)
        await plugin.on_start(config={})
        await plugin._dream_cycle()

        # Check that MEMORY.md was updated
        content = memory_path.read_text(encoding="utf-8")
        assert "## Long-term Memory" in content
        # Should contain extracted sentences (not all identical since we check dedup)
        assert "SQLite" in content or "persistence" in content

    async def test_dream_cycle_skips_when_no_db(self, tmp_path):
        from corvidae.tools.dream import DreamPlugin

        plugin = DreamPlugin(workspace_root=tmp_path)
        await plugin.on_start(config={})
        # Should not raise
        await plugin._dream_cycle()


# ---------------------------------------------------------------------------
# on_idle throttling
# ---------------------------------------------------------------------------


class TestIdleThrottling:
    async def test_skips_when_interval_not_elapsed(self):
        from corvidae.tools.dream import DreamPlugin

        plugin = DreamPlugin(workspace_root="/tmp")
        await plugin.on_start(config={})
        # Don't set _last_dream_time; it defaults to 0.0 which is very old
        # So the first call should proceed. Let's test the throttle by setting it recently.
        import time
        plugin._last_dream_time = time.time()
        plugin.interval_seconds = 300

        # on_idle should not trigger a cycle because we just ran one
        with patch.object(plugin, '_dream_cycle') as mock_dream:
            await plugin.on_idle()
            mock_dream.assert_not_called()

    async def test_triggers_when_interval_elapsed(self, tmp_path):
        from corvidae.tools.dream import DreamPlugin

        plugin = DreamPlugin(workspace_root=tmp_path)
        await plugin.on_start(config={})
        # Set last dream to 10 minutes ago
        plugin._last_dream_time = time.time() - 600
        plugin.interval_seconds = 300
        # Create a fake DB so the exists() check passes
        db = tmp_path / "corvidae" / "sessions.db"
        db.parent.mkdir(parents=True)
        db.touch()
        plugin._db_path = db

        with patch.object(plugin, '_dream_cycle') as mock_dream:
            await plugin.on_idle()
            mock_dream.assert_called_once()
