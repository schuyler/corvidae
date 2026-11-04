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


async def _setup_skills(plugin, skills_dir: str):
    """Register and start a SkillsPlugin on the given plugin."""
    skills = SkillsPlugin(plugin.pm)
    plugin.pm.register(skills, name="skills")
    await skills.on_init(plugin.pm, {"skills": {"directory": skills_dir}})
    await skills.on_start()
    return skills


class TestIndex:
    """The skills index in the system prompt."""

    async def test_index_appears_in_system_prompt(self, tmp_path):
        skills_dir = _make_skills_dir(tmp_path)
        plugin, channel, db = await build_plugin_and_channel()
        await _setup_skills(plugin, str(skills_dir))

        prompt = await plugin.channel_system_prompt(channel)
        assert "develop" in prompt
        assert "Code change process" in prompt
        assert "systemd-run" in prompt
        assert "use_skill" in prompt
        await db.close()

    async def test_index_not_accumulating(self, tmp_path):
        skills_dir = _make_skills_dir(tmp_path)
        plugin, channel, db = await build_plugin_and_channel()
        await _setup_skills(plugin, str(skills_dir))

        prompt1 = await plugin.channel_system_prompt(channel)
        prompt2 = await plugin.channel_system_prompt(channel)
        assert prompt1.count("Available skills") == 1
        assert prompt2.count("Available skills") == 1
        assert prompt1 == prompt2
        await db.close()

    async def test_empty_directory_no_index(self, tmp_path):
        empty = tmp_path / "empty_skills"
        empty.mkdir()
        plugin, channel, db = await build_plugin_and_channel()
        await _setup_skills(plugin, str(empty))

        prompt = await plugin.channel_system_prompt(channel)
        assert "Available skills" not in prompt
        await db.close()

    async def test_missing_directory_no_index(self, tmp_path):
        plugin, channel, db = await build_plugin_and_channel()
        await _setup_skills(plugin, str(tmp_path / "nope"))

        prompt = await plugin.channel_system_prompt(channel)
        assert "Available skills" not in prompt
        await db.close()


class TestUseSkill:
    """The use_skill tool."""

    async def test_use_skill_returns_full_content(self, tmp_path):
        skills_dir = _make_skills_dir(tmp_path)
        plugin, channel, db = await build_plugin_and_channel()
        skills = await _setup_skills(plugin, str(skills_dir))

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
        skills = await _setup_skills(plugin, str(skills_dir))

        tools: list = []
        skills.register_tools(tools)
        result = await tools[0]("nonexistent")
        assert "not found" in result.lower()
        await db.close()

    async def test_use_skill_lists_available(self, tmp_path):
        skills_dir = _make_skills_dir(tmp_path)
        plugin, channel, db = await build_plugin_and_channel()
        skills = await _setup_skills(plugin, str(skills_dir))

        tools: list = []
        skills.register_tools(tools)
        result = await tools[0]()
        assert "develop" in result
        assert "systemd-run" in result
        await db.close()
