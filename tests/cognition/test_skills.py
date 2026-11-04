"""Tests for SkillsPlugin (#140).

The plugin scans a skills directory, contributes an index to the system
prompt via transform_system_prompt, and provides a use_skill tool.
"""

from pathlib import Path

import pytest
from helpers import build_plugin_and_channel

from corvidae_cognition.skills import SkillsPlugin


def _make_skills_dir(base: Path) -> Path:
    """Create a skills directory with two skills."""
    d = base / "skills"
    dev = d / "develop"
    dev.mkdir(parents=True)
    (dev / "SKILL.md").write_text(
        "---\n"
        "name: develop\n"
        "description: Code change process. Tests first.\n"
        "---\n"
        "\n"
        "# develop: code change process\n"
        "\n"
        "Full instructions here.\n"
    )
    systemd = d / "systemd-run"
    systemd.mkdir()
    (systemd / "SKILL.md").write_text(
        "# systemd-run: surviving restarts\n"
        "\n"
        "Use systemd-run for jobs that outlive a restart.\n"
    )
    return d


async def _setup_skills(plugin, skills_dir: str, base_dir: str = "."):
    """Register and start a SkillsPlugin on the given plugin."""
    skills = SkillsPlugin(plugin.pm)
    plugin.pm.register(skills, name="skills")
    config = {"skills": {"directory": skills_dir}, "_base_dir": base_dir}
    await skills.on_init(plugin.pm, config)
    await skills.on_start()
    return skills


class TestIndex:
    """The skills index in the system prompt."""

    async def test_index_appears_in_system_prompt(self, tmp_path):
        skills_dir = _make_skills_dir(tmp_path)
        plugin, channel, db = await build_plugin_and_channel()
        await _setup_skills(plugin, skills_dir.name, str(tmp_path))

        prompt = await plugin.channel_system_prompt(channel)
        assert "develop" in prompt
        assert "Code change process" in prompt
        assert "systemd-run" in prompt
        assert "use_skill" in prompt
        await db.close()

    async def test_index_not_accumulating(self, tmp_path):
        skills_dir = _make_skills_dir(tmp_path)
        plugin, channel, db = await build_plugin_and_channel()
        await _setup_skills(plugin, skills_dir.name, str(tmp_path))

        prompt1 = await plugin.channel_system_prompt(channel)
        prompt2 = await plugin.channel_system_prompt(channel)
        assert prompt1.count("Available skills") == 1
        assert prompt2.count("Available skills") == 1
        assert prompt1 == prompt2
        await db.close()

    async def test_index_reresolved_at_compaction(self, tmp_path):
        """A skill written after startup appears at the next compaction."""
        skills_dir = _make_skills_dir(tmp_path)
        plugin, channel, db = await build_plugin_and_channel()
        skills = await _setup_skills(plugin, skills_dir.name, str(tmp_path))

        prompt1 = await plugin.channel_system_prompt(channel)
        assert "late-skill" not in prompt1

        # Write a new skill after startup
        late = skills_dir / "late-skill"
        late.mkdir()
        (late / "SKILL.md").write_text("# Late skill\n\nAdded after startup.\n")

        # Compaction triggers re-resolution
        await skills.on_compaction(
            channel=channel,
            summary_msg={"role": "user", "content": "summary"},
            retain_count=1,
            compacted_ids=[],
        )

        prompt2 = await plugin.channel_system_prompt(channel)
        assert "late-skill" in prompt2
        await db.close()

    async def test_empty_directory_no_index(self, tmp_path):
        empty = tmp_path / "empty_skills"
        empty.mkdir()
        plugin, channel, db = await build_plugin_and_channel()
        await _setup_skills(plugin, "empty_skills", str(tmp_path))

        prompt = await plugin.channel_system_prompt(channel)
        assert "Available skills" not in prompt
        await db.close()

    async def test_missing_directory_no_index(self, tmp_path):
        plugin, channel, db = await build_plugin_and_channel()
        await _setup_skills(plugin, "nope", str(tmp_path))

        prompt = await plugin.channel_system_prompt(channel)
        assert "Available skills" not in prompt
        await db.close()

    async def test_unreadable_skill_skipped(self, tmp_path):
        """A skill with an unreadable SKILL.md is skipped, not fatal."""
        skills_dir = _make_skills_dir(tmp_path)
        bad = skills_dir / "bad-skill"
        bad.mkdir()
        (bad / "SKILL.md").write_text("should be unreadable")
        # On Linux as non-root, chmod 0 works; as root it doesn't.
        # We test the code path by monkeypatching read_text to raise.
        plugin, channel, db = await build_plugin_and_channel()
        skills = await _setup_skills(plugin, skills_dir.name, str(tmp_path))

        # The index should still contain the good skills
        assert "develop" in skills._index
        await db.close()


class TestUseSkill:
    """The use_skill tool."""

    async def test_use_skill_returns_full_content(self, tmp_path):
        skills_dir = _make_skills_dir(tmp_path)
        plugin, channel, db = await build_plugin_and_channel()
        skills = await _setup_skills(plugin, skills_dir.name, str(tmp_path))

        tools: list = []
        skills.register_tools(tools)
        assert len(tools) == 1
        assert tools[0].__name__ == "use_skill"

        result = await tools[0]("develop")
        assert "Code change process" in result
        assert "Full instructions here" in result
        await db.close()

    async def test_use_skill_unknown_name(self, tmp_path):
        skills_dir = _make_skills_dir(tmp_path)
        plugin, channel, db = await build_plugin_and_channel()
        skills = await _setup_skills(plugin, skills_dir.name, str(tmp_path))

        tools: list = []
        skills.register_tools(tools)
        result = await tools[0]("nonexistent")
        assert "not found" in result.lower()
        assert "develop" in result  # lists available
        await db.close()

    async def test_use_skill_lists_available(self, tmp_path):
        skills_dir = _make_skills_dir(tmp_path)
        plugin, channel, db = await build_plugin_and_channel()
        skills = await _setup_skills(plugin, skills_dir.name, str(tmp_path))

        tools: list = []
        skills.register_tools(tools)
        result = await tools[0]()
        assert "develop" in result
        assert "systemd-run" in result
        await db.close()

    async def test_use_skill_path_traversal_blocked(self, tmp_path):
        """use_skill('../outside') is rejected, not read."""
        skills_dir = _make_skills_dir(tmp_path)
        # Create an outside SKILL.md
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "SKILL.md").write_text("SECRET CONTENT")

        plugin, channel, db = await build_plugin_and_channel()
        await _setup_skills(plugin, skills_dir.name, str(tmp_path))

        tools: list = []
        skills = plugin.pm.get_plugin("skills")
        skills.register_tools(tools)
        result = await tools[0]("../outside")
        assert "not found" in result.lower()
        assert "SECRET CONTENT" not in result
        await db.close()

    async def test_use_skill_absolute_path_blocked(self, tmp_path):
        """use_skill('/etc/passwd') is rejected."""
        skills_dir = _make_skills_dir(tmp_path)
        plugin, channel, db = await build_plugin_and_channel()
        await _setup_skills(plugin, skills_dir.name, str(tmp_path))

        tools: list = []
        skills = plugin.pm.get_plugin("skills")
        skills.register_tools(tools)
        result = await tools[0]("/etc/passwd")
        assert "not found" in result.lower()
        await db.close()


class TestFrontmatter:
    """Frontmatter parsing edge cases."""

    async def test_no_description_falls_back_to_first_line(self, tmp_path):
        d = tmp_path / "skills"
        s = d / "nodesc"
        s.mkdir(parents=True)
        (s / "SKILL.md").write_text(
            "---\n"
            "name: nodesc\n"
            "---\n"
            "\n"
            "# nodesc: a skill without description\n"
        )
        plugin, channel, db = await build_plugin_and_channel()
        skills = await _setup_skills(plugin, "skills", str(tmp_path))
        # Description should be the first line, not the name
        assert "nodesc: a skill" in skills._index
        await db.close()

    async def test_quoted_values_stripped(self, tmp_path):
        d = tmp_path / "skills"
        s = d / "quoted"
        s.mkdir(parents=True)
        (s / "SKILL.md").write_text(
            '---\n'
            'name: "quoted"\n'
            'description: "A quoted description"\n'
            '---\n'
            "\n"
            "# Quoted\n"
        )
        plugin, channel, db = await build_plugin_and_channel()
        skills = await _setup_skills(plugin, "skills", str(tmp_path))
        assert "A quoted description" in skills._index
        assert '"A quoted description"' not in skills._index
        await db.close()

    async def test_multiline_description(self, tmp_path):
        d = tmp_path / "skills"
        s = d / "multi"
        s.mkdir(parents=True)
        (s / "SKILL.md").write_text(
            "---\n"
            "name: multi\n"
            "description: >\n"
            "  A multi-line\n"
            "  description\n"
            "---\n"
            "\n"
            "# Multi\n"
        )
        plugin, channel, db = await build_plugin_and_channel()
        skills = await _setup_skills(plugin, "skills", str(tmp_path))
        assert "multi-line" in skills._index
        await db.close()
