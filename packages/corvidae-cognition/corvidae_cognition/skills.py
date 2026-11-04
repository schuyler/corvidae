"""SkillsPlugin — skills directory, index, and use_skill tool (#140).

Skills are a directory of name, description, instructions, and resources.
An index of names and descriptions sits in the stable prompt region,
contributed through the kernel's ``transform_system_prompt`` hook (#136).
A ``use_skill`` tool loads a skill's full instructions on demand.

Skills are writable by the agent (principle 15). Bundled copies protect
against loss: an overwritten skill can be restored from its bundled copy.
"""

import logging
from pathlib import Path

from corvidae.hooks import CorvidaePlugin, hookimpl

logger = logging.getLogger("corvidae.skills")


def _parse_frontmatter(text: str) -> dict:
    """Parse simple YAML frontmatter (key: value pairs) from a SKILL.md.

    Returns an empty dict when there is no frontmatter.
    """
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    block = text[3:end].strip()
    result = {}
    for line in block.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            result[key.strip()] = value.strip()
    return result


def _first_line(text: str) -> str:
    """The first non-empty, non-frontmatter line, stripped of # markers."""
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("---"):
            return line.lstrip("# ").strip()
    return ""


class SkillsPlugin(CorvidaePlugin):
    """Skills directory, index in the system prompt, and use_skill tool."""

    def __init__(self, pm=None) -> None:
        if pm is not None:
            self.pm = pm
        self._skills_dir: Path | None = None
        self._index: str = ""

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        skills_cfg = config.get("skills", {})
        self._skills_dir = Path(skills_cfg.get("directory", "skills"))

    @hookimpl
    async def on_start(self) -> None:
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        """Scan the skills directory and build the index string."""
        if self._skills_dir is None or not self._skills_dir.is_dir():
            self._index = ""
            return

        entries = []
        for entry in sorted(self._skills_dir.iterdir()):
            skill_file = entry / "SKILL.md"
            if not entry.is_dir() or not skill_file.is_file():
                continue
            text = skill_file.read_text(encoding="utf-8", errors="replace")
            fm = _parse_frontmatter(text)
            name = fm.get("name", entry.name)
            description = fm.get("description", _first_line(text))
            entries.append(f"- {name}: {description}")

        if not entries:
            self._index = ""
            return

        self._index = (
            "Available skills (use `use_skill` to load full instructions):\n"
            + "\n".join(entries)
        )

    @hookimpl(wrapper=True)
    def transform_system_prompt(self, channel, text):
        result = yield
        if self._index:
            return f"{result}\n\n{self._index}"
        return result

    @hookimpl
    def register_tools(self, tool_registry: list) -> None:
        plugin = self

        async def use_skill(name: str = "") -> str:
            """Load a skill's full instructions by name.
            With no argument, lists available skills."""
            if not name:
                if not plugin._index:
                    return "No skills available."
                return plugin._index
            if plugin._skills_dir is None:
                return "Skills directory not configured."
            skill_file = plugin._skills_dir / name / "SKILL.md"
            if not skill_file.is_file():
                available = ", ".join(
                    e.name for e in sorted(plugin._skills_dir.iterdir())
                    if e.is_dir() and (e / "SKILL.md").is_file()
                )
                return (
                    f"Skill '{name}' not found. "
                    f"Available: {available or 'none'}"
                )
            return skill_file.read_text(encoding="utf-8", errors="replace")

        tool_registry.append(use_skill)

