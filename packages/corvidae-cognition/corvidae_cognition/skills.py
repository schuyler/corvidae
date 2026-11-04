"""SkillsPlugin — skills directory, index, and use_skill tool (#140).

Skills are a directory of name, description, instructions, and resources.
An index of names and descriptions sits in the stable prompt region,
contributed through the kernel's ``transform_system_prompt`` hook (#136).
A ``use_skill`` tool loads a skill's full instructions on demand.

Skills are writable by the agent (principle 15). The index is re-resolved
at compaction boundaries so a skill written after startup appears in the
prompt at the next boundary.
"""

import logging
from pathlib import Path

import yaml

from corvidae.hooks import CorvidaePlugin, hookimpl

logger = logging.getLogger("corvidae.skills")


def _parse_skill(text: str) -> tuple[str, str]:
    """Extract (name, description) from a SKILL.md file.

    Uses YAML frontmatter when present; falls back to the first line.
    Returns (name, description) where name defaults to the directory
    name (set by the caller) and description defaults to the first line.
    """
    fm: dict = {}
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            try:
                loaded = yaml.safe_load(text[3:end])
                if isinstance(loaded, dict):
                    fm = loaded
            except yaml.YAMLError:
                pass

    name = str(fm.get("name", ""))
    description = str(fm.get("description", ""))

    if not description:
        # Fall back to the first non-empty line after the frontmatter block
        start = 0
        if text.startswith("---"):
            end = text.find("\n---", 3)
            if end != -1:
                start = end + 4  # skip past the closing ---
        for line in text[start:].splitlines():
            line = line.strip()
            if line:
                description = line.lstrip("# ").strip()
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

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        await super().on_init(pm, config)
        skills_cfg = config.get("skills", {})
        directory = skills_cfg.get("directory", "skills")
        base = config.get("_base_dir", ".")
        self._skills_dir = Path(base) / directory

    @hookimpl
    async def on_start(self) -> None:
        self._rebuild_index()

    @hookimpl
    async def on_compaction(self, channel, summary_msg, retain_count, compacted_ids):
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        """Scan the skills directory and build the index string."""
        self._skill_names = set()
        if self._skills_dir is None or not self._skills_dir.is_dir():
            self._index = ""
            return

        entries = []
        try:
            children = sorted(self._skills_dir.iterdir())
        except OSError as e:
            logger.warning("Skills directory unreadable: %s", e)
            self._index = ""
            return

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
            # The directory name is the canonical key; frontmatter name
            # is display-only.
            self._skill_names.add(entry.name)
            display_name = name or entry.name
            entries.append(f"- {display_name}: {description}")

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
            # Validate against the scanned index to prevent path traversal.
            if name not in plugin._skill_names:
                available = ", ".join(sorted(plugin._skill_names))
                return (
                    f"Skill '{name}' not found. "
                    f"Available: {available or 'none'}. "
                    f"Use use_skill with no argument to list skills."
                )
            skill_file = plugin._skills_dir / name / "SKILL.md"
            try:
                return skill_file.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                return f"Failed to read skill '{name}': {e}"

        tool_registry.append(use_skill)
