"""Integration tests: resolved prompt keys must reach the actual prompts.

tests/test_prompts.py proves resolve_prompt in isolation. The bug that
slipped past that coverage (RubricPrior referenced a prompt attribute that
did not exist; the exception was swallowed and every memory silently
scored 0.5) can only be caught by a test that follows the resolved value
to the messages the stage actually sends.

Kernel prompt keys (cognition plugin keys are in tests/cognition/):

- tools.subagent.system_prompt       -> SubagentPlugin._system_prompt
- agent.compaction_fork_prompt       -> CompactionPlugin._fork_prompt

Plus the reload contract: a bad reload value keeps the previous prompt
and leaves no partial state.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from corvidae.compaction import CompactionPlugin
from corvidae.tools.subagent import SUBAGENT_SYSTEM_PROMPT, SubagentPlugin


class _FakePM:
    """Duck-typed plugin manager: the base on_init only stores it."""

    def __init__(self, plugins: dict | None = None) -> None:
        self._plugins = plugins or {}

    def get_plugin(self, name: str):
        return self._plugins.get(name)


class TestSubagentPromptRouting:
    """tools.subagent.system_prompt at on_start."""

    async def test_inline_resolved_at_start(self):
        plugin = SubagentPlugin()
        cfg = {"tools": {"subagent": {"system_prompt": "SUB INLINE"}}}
        await plugin.on_start(cfg)
        assert plugin._system_prompt == "SUB INLINE"

    async def test_default_when_absent(self):
        plugin = SubagentPlugin()
        await plugin.on_start({})
        assert plugin._system_prompt == SUBAGENT_SYSTEM_PROMPT


class TestCompactionPromptRouting:
    """agent.compaction_fork_prompt at on_init / on_config_reload."""

    async def test_inline_resolved_at_init(self):
        plugin = CompactionPlugin()
        cfg = {"agent": {"compaction_fork_prompt": "COMPACT INLINE"}}
        await plugin.on_init(_FakePM(), cfg)
        assert plugin._fork_prompt == "COMPACT INLINE"

    async def test_file_resolved_at_init(self, tmp_path: Path):
        (tmp_path / "c.md").write_text("Compact from file.\n")
        plugin = CompactionPlugin()
        cfg = {
            "_base_dir": str(tmp_path),
            "agent": {"compaction_fork_prompt": {"file": "c.md"}},
        }
        await plugin.on_init(_FakePM(), cfg)
        assert plugin._fork_prompt == "Compact from file."

    async def test_bad_reload_value_leaves_no_partial_state(self):
        # Resolve-into-locals before commit: a bad prompt file must not
        # publish new thresholds without the prompt.
        plugin = CompactionPlugin()
        cfg = {"agent": {"compaction_threshold": 0.8, "compaction_fork_prompt": "FIRST"}}
        await plugin.on_init(_FakePM(), cfg)
        with pytest.raises(ValueError):
            await plugin.on_config_reload(
                {"agent": {"compaction_threshold": 0.9, "compaction_fork_prompt": 42}}
            )
        assert plugin._fork_prompt == "FIRST"
        assert plugin._compaction_threshold == 0.8
