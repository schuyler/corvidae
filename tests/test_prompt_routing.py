"""Integration tests: resolved prompt keys must reach the actual prompts.

tests/test_prompts.py proves resolve_prompt in isolation. The bug that
slipped past that coverage (RubricPrior referenced a prompt attribute that
did not exist; the exception was swallowed and every memory silently
scored 0.5) can only be caught by a test that follows the resolved value
to the messages the stage actually sends.

For every prompt key routed through resolve_prompt (#411/#412), pin the
wiring config -> resolve_prompt -> the prompt that goes out:

- memory.rubric_prompt               -> RubricPrior.score() system message
- heartbeat.prompt                   -> HeartbeatPlugin._prompt
- heartbeat.goals_instruction        -> HeartbeatPlugin._goals_instruction
- tools.subagent.system_prompt       -> SubagentPlugin._system_prompt
- agent.compaction_fork_prompt       -> CompactionPlugin._fork_prompt
- memory.consolidation_fork_prompt   -> resolve_consolidation_fork_prompt()
- appraisal.stage2_fork_prompt       -> AppraisalPlugin._fork_prompt
- critique.fork_prompt.<lens>        -> CritiquePlugin._fork_prompts

Plus the reload contract: a bad reload value keeps the previous prompt
and leaves no partial state.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from corvidae.compaction import CompactionPlugin
from corvidae.tools.subagent import SUBAGENT_SYSTEM_PROMPT, SubagentPlugin
from corvidae_cognition.appraisal import AppraisalPlugin
from corvidae_cognition.critique import DEFAULT_FORK_PROMPTS, CritiquePlugin
from corvidae_cognition.heartbeat import (
    DEFAULT_HEARTBEAT_PROMPT,
    GOALS_INSTRUCTION,
    HeartbeatPlugin,
)
from corvidae_cognition.memory import (
    DEFAULT_CONSOLIDATION_FORK_PROMPT,
    RUBRIC_PROMPT,
    RubricPrior,
    resolve_consolidation_fork_prompt,
)


class _FakePM:
    """Duck-typed plugin manager: the base on_init only stores it."""

    def __init__(self, plugins: dict | None = None) -> None:
        self._plugins = plugins or {}

    def get_plugin(self, name: str):
        return self._plugins.get(name)


class _CaptureClient:
    """Background LLM client that records calls and returns importance 0.9."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def chat(self, messages: list[dict], extra_body=None) -> dict:
        self.calls.append({"messages": messages, "extra_body": extra_body})
        return {"choices": [{"message": {"content": '{"importance": 0.9}'}}]}


class TestRubricPromptRouting:
    """memory.rubric_prompt must reach the scorer's system message."""

    async def test_resolved_prompt_reaches_scorer(self):
        client = _CaptureClient()
        prior = RubricPrior(lambda: client, lambda: "CUSTOM RUBRIC TEXT")
        score = await prior.score([{"role": "user", "content": "hi"}])
        assert score == 0.9
        sent = client.calls[0]["messages"]
        assert sent[0] == {"role": "system", "content": "CUSTOM RUBRIC TEXT"}

    async def test_prompt_is_read_at_call_time(self):
        # The exact shape the RubricPrior bug broke: the prior must see a
        # prompt swapped after construction (config reload), so the wiring
        # passes a callable, never a value.
        client = _CaptureClient()
        state = {"prompt": "FIRST"}
        prior = RubricPrior(lambda: client, lambda: state["prompt"])
        await prior.score([{"role": "user", "content": "a"}])
        state["prompt"] = "SECOND"
        await prior.score([{"role": "user", "content": "b"}])
        contents = [c["messages"][0]["content"] for c in client.calls]
        assert contents == ["FIRST", "SECOND"]

    async def test_default_prompt_when_no_get_prompt(self):
        client = _CaptureClient()
        prior = RubricPrior(lambda: client)
        await prior.score([{"role": "user", "content": "a"}])
        assert client.calls[0]["messages"][0]["content"] == RUBRIC_PROMPT


class TestHeartbeatPromptRouting:
    """heartbeat.prompt / heartbeat.goals_instruction at on_init."""

    async def test_inline_prompts_resolved_at_init(self):
        plugin = HeartbeatPlugin()
        cfg = {
            "heartbeat": {
                "after": 60,
                "prompt": "HB INLINE",
                "goals_instruction": "GOALS INLINE",
            }
        }
        await plugin.on_init(_FakePM({"memory": object()}), cfg)
        assert plugin._prompt == "HB INLINE"
        assert plugin._goals_instruction == "GOALS INLINE"

    async def test_file_prompt_resolved_at_init(self, tmp_path: Path):
        (tmp_path / "hb.md").write_text("From the heartbeat file.\n")
        plugin = HeartbeatPlugin()
        cfg = {
            "_base_dir": str(tmp_path),
            "heartbeat": {"after": 60, "prompt": {"file": "hb.md"}},
        }
        await plugin.on_init(_FakePM({"memory": object()}), cfg)
        assert plugin._prompt == "From the heartbeat file."

    async def test_defaults_when_keys_absent(self):
        plugin = HeartbeatPlugin()
        cfg = {"heartbeat": {"after": 60}}
        await plugin.on_init(_FakePM({"memory": object()}), cfg)
        assert plugin._prompt == DEFAULT_HEARTBEAT_PROMPT
        assert plugin._goals_instruction == GOALS_INSTRUCTION


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


class TestConsolidationPromptRouting:
    """memory.consolidation_fork_prompt via resolve_consolidation_fork_prompt."""

    def test_inline(self):
        cfg = {"memory": {"consolidation_fork_prompt": "CONSOL INLINE"}}
        assert resolve_consolidation_fork_prompt(cfg) == "CONSOL INLINE"

    def test_file(self, tmp_path: Path):
        (tmp_path / "m.md").write_text("Consolidate from file.\n")
        cfg = {
            "_base_dir": str(tmp_path),
            "memory": {"consolidation_fork_prompt": {"file": "m.md"}},
        }
        assert resolve_consolidation_fork_prompt(cfg) == "Consolidate from file."

    def test_default_when_absent(self):
        assert (
            resolve_consolidation_fork_prompt({"memory": {}})
            == DEFAULT_CONSOLIDATION_FORK_PROMPT
        )

    def test_bad_value_raises_value_error_naming_key(self):
        # The recorded contract decision (#412 review): the config-level
        # helper raises ValueError, not TypeError, for consistency with
        # config validation.
        cfg = {"memory": {"consolidation_fork_prompt": 42}}
        with pytest.raises(ValueError, match="consolidation_fork_prompt"):
            resolve_consolidation_fork_prompt(cfg)


class TestAppraisalPromptRouting:
    """appraisal.stage2_fork_prompt at on_init / on_config_reload."""

    async def test_inline_resolved_at_init(self):
        plugin = AppraisalPlugin()
        cfg = {"appraisal": {"stage2_fork_prompt": "APP INLINE"}}
        await plugin.on_init(_FakePM(), cfg)
        assert plugin._fork_prompt == "APP INLINE"

    async def test_bad_reload_value_keeps_previous(self):
        plugin = AppraisalPlugin()
        cfg = {"appraisal": {"stage2_fork_prompt": "FIRST"}}
        await plugin.on_init(_FakePM(), cfg)
        await plugin.on_config_reload({"appraisal": {"stage2_fork_prompt": 42}})
        assert plugin._fork_prompt == "FIRST"


class TestCritiquePromptRouting:
    """critique.fork_prompt.<lens> at on_init / on_config_reload."""

    async def test_lens_prompts_resolved_at_init(self):
        plugin = CritiquePlugin()
        cfg = {"critique": {"fork_prompt": {"provenance": "PROV INLINE"}}}
        await plugin.on_init(_FakePM(), cfg)
        assert plugin._fork_prompts["provenance"] == "PROV INLINE"
        assert plugin._fork_prompts["predictive"] == DEFAULT_FORK_PROMPTS["predictive"]

    async def test_bad_reload_value_keeps_previous(self):
        plugin = CritiquePlugin()
        await plugin.on_init(_FakePM(), {})
        first = dict(plugin._fork_prompts)
        await plugin.on_config_reload(
            {"critique": {"fork_prompt": {"provenance": 42}}}
        )
        assert plugin._fork_prompts == first
