"""SkillsPlugin — skills directory, index, and use_skill tool (#140).

Skills are a directory of name, description, instructions, and resources.
An index of names and descriptions sits in the stable prompt region,
contributed through the kernel's ``transform_system_prompt`` hook (#136).
A ``use_skill`` tool loads a skill's full instructions on demand.

Skills are writable by the agent (principle 15). The index is re-resolved
at compaction boundaries so a skill written after startup appears in the
prompt at the next boundary.
"""

import asyncio
import logging
import re
from pathlib import Path

import yaml

from corvidae.hooks import CorvidaePlugin, hookimpl

logger = logging.getLogger("corvidae.skills")


def _parse_skill(text: str) -> tuple[str, str]:
    """Extract (name, description) from a SKILL.md file.

    Uses YAML frontmatter when present; falls back to the first content
    line. Returns (name, description) where name defaults to "" (the
    caller substitutes the directory name) and description defaults to
    the first non-empty line after any frontmatter.
    """
    # Strip BOM if present
    if text.startswith("\ufeff"):
        text = text[1:]

    fm: dict = {}
    fm_end = 0
    lines = text.splitlines(keepends=True)
    if lines and lines[0].strip() == "---":
        # Find the closing delimiter: a line that is exactly "---"
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                fm_end = sum(len(l) for l in lines[: i + 1])
                try:
                    loaded = yaml.safe_load("".join(lines[1:i]))
                    if isinstance(loaded, dict):
                        fm = loaded
                except yaml.YAMLError:
                    pass
                break

    name = fm.get("name")
    name = str(name) if name is not None else ""
    description = fm.get("description")
    description = str(description).strip() if description is not None else ""

    if not description:
        # Fall back to the first non-empty line after the frontmatter
        for line in text[fm_end:].splitlines():
            line = line.strip()
            if line:
                # Strip markdown heading markers
                description = re.sub(r"^\s*#+\s*", "", line).strip()
                break

    return name, description


class SkillsPlugin(CorvidaePlugin):
    """Skills directory, index in the system prompt, and use_skill tool."""

    def __init__(self, pm=None) -> None:
        if pm is not None:
            self.pm = pm
        self._skills_dir: Path | None = None
        self._index: str = ""
        self._skill_names: set[str] = set()
        self._name_to_dir: dict[str, str] = {}

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        await super().on_init(pm, config)
        skills_cfg = config.get("skills") or {}
        directory = skills_cfg.get("directory") or "skills"
        base = config.get("_base_dir") or "."
        self._skills_dir = Path(base) / directory

    @hookimpl
    async def on_start(self) -> None:
        await asyncio.to_thread(self._rebuild_index)

    @hookimpl
    async def on_compaction(self, channel, summary_msg, retain_count, compacted_ids):
        await asyncio.to_thread(self._rebuild_index)

    def _scan_skills(self) -> list[tuple[str, str, str]]:
        """Synchronous scan. Returns (dir_name, display_name, description) tuples."""
        results = []
        if self._skills_dir is None or not self._skills_dir.is_dir():
            return results

        try:
            children = sorted(self._skills_dir.iterdir(), key=lambda p: p.name)
        except OSError as e:
            logger.warning("Skills directory unreadable: %s", e)
            return results

        for entry in children:
            skill_file = entry / "SKILL.md"
            if not entry.is_dir() or not skill_file.is_file():
                continue
            try:
                text = skill_file.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                logger.warning("Skipping unreadable skill %s: %s", entry.name, e)
                continue
            name, description = _parse_skill(text)
            display_name = name or entry.name
            results.append((entry.name, display_name, description))
        return results

    def _rebuild_index(self) -> None:
        """Build the index string from a synchronous scan."""
        results = self._scan_skills()
        self._skill_names = {r[0] for r in results}
        # Map both directory name and display name to the directory name,
        # so use_skill accepts either.
        self._name_to_dir = {}
        for dir_name, display_name, _ in results:
            self._name_to_dir[dir_name] = dir_name
            self._name_to_dir[display_name] = dir_name
        entries = [f"- {r[1]}: {r[2]}" for r in results]
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
        if result is None:
            result = text
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
            # Validate against the scanned index to prevent path traversal.
            # Accept both directory name and frontmatter display name.
            if name not in plugin._name_to_dir:
                available = ", ".join(sorted(plugin._skill_names))
                return (
                    f"Skill '{name}' not found. "
                    f"Available: {available or 'none'}. "
                    f"Use use_skill with no argument to list skills."
                )
            dir_name = plugin._name_to_dir[name]
            skill_file = plugin._skills_dir / dir_name / "SKILL.md"

            def _read() -> str:
                return skill_file.read_text(encoding="utf-8", errors="replace")

            try:
                return await asyncio.to_thread(_read)
            except OSError as e:
                return f"Failed to read skill '{name}': {e}"

        tool_registry.append(use_skill)
