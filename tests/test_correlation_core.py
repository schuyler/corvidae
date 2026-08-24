"""RED tests for the core-generalization work (disentangle-buster.md Design
§1 correlation mechanism, §2 deletions).

R1's acceptance made executable: a grep test asserting no core module
references the cognition vocabulary, plus structural assertions (QueueItem/
Task field names, the new hookspec signatures, absence of the deleted
surface); then behavioral round-trip tests for the generic correlation
mechanism (dequeue resolution, attribution, Task/TaskPlugin meta
propagation, on_message_persisted firing discipline).
"""

import dataclasses
import inspect
import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from helpers import build_plugin_and_channel, drain
from llm_response_fixtures import _make_text_response

CORVIDAE_ROOT = Path(__file__).resolve().parent.parent / "corvidae"

# R1: cognition plugins keep cognition vocabulary internally (plugin-side);
# only core modules must be vocabulary-free (Design §1, "Vocabulary rule").
COGNITION_MODULES = {
    "appraisal.py",
    "critique.py",
    "funnel.py",
    "memory.py",
    "retention.py",
    "outcome_log.py",
    "thinking.py",
    "tools/memory_tools.py",
}

VOCAB_PATTERN = re.compile(r"\b(appraisal|critique|funnel|salience|withheld|engagement)\b")
ORIGIN_TAXONOMY_LITERAL = "'user'|'reminder'|'critique'|'heartbeat'|'task'"


def _core_python_files():
    for path in sorted(CORVIDAE_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(CORVIDAE_ROOT).as_posix()
        if rel in COGNITION_MODULES:
            continue
        yield path, rel


# ---------------------------------------------------------------------------
# R1 vocabulary grep
# ---------------------------------------------------------------------------


class TestR1VocabularyGrep:
    def test_no_core_module_matches_cognition_vocabulary(self):
        offenders = []
        for path, rel in _core_python_files():
            text = path.read_text()
            for m in VOCAB_PATTERN.finditer(text):
                line_no = text.count("\n", 0, m.start()) + 1
                offenders.append(f"{rel}:{line_no}: {m.group(0)!r}")
        assert not offenders, (
            "core modules must not reference cognition vocabulary "
            "(appraisal/critique/funnel/salience/withheld/engagement):\n"
            + "\n".join(offenders)
        )

    def test_no_core_module_contains_origin_taxonomy_literal(self):
        offenders = [
            rel for path, rel in _core_python_files()
            if ORIGIN_TAXONOMY_LITERAL in path.read_text()
        ]
        assert not offenders, (
            f"core modules must not hardcode the origin taxonomy literal "
            f"{ORIGIN_TAXONOMY_LITERAL!r}; found in: {offenders}"
        )


# ---------------------------------------------------------------------------
# Structural assertions: QueueItem / Task field names
# ---------------------------------------------------------------------------


class TestQueueItemStructure:
    def test_fields_are_generic(self):
        from corvidae.agent import QueueItem

        names = {f.name for f in dataclasses.fields(QueueItem)}
        assert "correlation_id" in names
        assert "originating" in names
        assert "exchange_key" not in names
        assert "origin" not in names
        assert "originates_exchange" not in names


class TestTaskStructure:
    def test_fields_are_generic(self):
        from corvidae.task import Task

        names = {f.name for f in dataclasses.fields(Task)}
        assert "correlation_id" in names
        assert "meta" in names
        assert "exchange_key" not in names
        assert "origin" not in names


# ---------------------------------------------------------------------------
# Structural assertions: hookspec signatures
# ---------------------------------------------------------------------------


def _params(fn) -> list[str]:
    return [p for p in inspect.signature(fn).parameters if p != "self"]


class TestHookspecSignatures:
    def test_should_process_message(self):
        from corvidae.hooks import AgentSpec
        assert _params(AgentSpec.should_process_message) == [
            "channel", "sender", "text", "correlation_id",
        ]

    def test_on_message_admitted(self):
        from corvidae.hooks import AgentSpec
        assert _params(AgentSpec.on_message_admitted) == [
            "channel", "correlation_id", "sender", "text",
        ]

    def test_on_message_rejected(self):
        from corvidae.hooks import AgentSpec
        assert _params(AgentSpec.on_message_rejected) == [
            "channel", "correlation_id", "sender", "text",
        ]

    def test_on_message_persisted(self):
        from corvidae.hooks import AgentSpec
        assert _params(AgentSpec.on_message_persisted) == [
            "channel", "correlation_id", "rowid", "text", "meta",
        ]

    def test_before_agent_turn(self):
        from corvidae.hooks import AgentSpec
        assert _params(AgentSpec.before_agent_turn) == [
            "channel", "correlation_id", "meta",
        ]

    def test_on_agent_response(self):
        from corvidae.hooks import AgentSpec
        assert _params(AgentSpec.on_agent_response) == [
            "channel", "request_text", "response_text",
            "correlation_id", "meta", "logprobs",
        ]


# ---------------------------------------------------------------------------
# Deletion verification
# ---------------------------------------------------------------------------


class TestDeletedSurfaceAbsent:
    def test_hookstrategy_deleted(self):
        import corvidae.hooks as hooks_module
        assert not hasattr(hooks_module, "HookStrategy")

    def test_resolve_hook_results_deleted_and_replaced(self):
        import corvidae.hooks as hooks_module
        assert not hasattr(hooks_module, "resolve_hook_results")
        assert hasattr(hooks_module, "resolve_reject_wins")

    def test_on_plugin_added_removed_hookspecs_deleted(self):
        from corvidae.hooks import AgentSpec
        assert not hasattr(AgentSpec, "on_plugin_added")
        assert not hasattr(AgentSpec, "on_plugin_removed")

    def test_agent_refresh_tools_and_hookimpls_deleted(self):
        from corvidae.agent import Agent
        assert not hasattr(Agent, "refresh_tools")
        assert not hasattr(Agent, "on_plugin_added")
        assert not hasattr(Agent, "on_plugin_removed")

    def test_agentplugin_alias_deleted(self):
        import corvidae.agent as agent_module
        assert not hasattr(agent_module, "AgentPlugin")

    def test_tool_collection_plugin_added_removed_hookimpls_deleted(self):
        from corvidae.tool_collection import ToolCollectionPlugin
        assert not hasattr(ToolCollectionPlugin, "on_plugin_added")
        assert not hasattr(ToolCollectionPlugin, "on_plugin_removed")
        # rebuild_registry itself stays -- on_start still uses it.
        assert hasattr(ToolCollectionPlugin, "rebuild_registry")

    def test_mint_correlation_id_replaces_mint_exchange_key(self):
        import corvidae.agent as agent_module
        assert hasattr(agent_module, "mint_correlation_id")
        assert not hasattr(agent_module, "mint_exchange_key")


# ---------------------------------------------------------------------------
# Behavioral: dequeue correlation resolution
# ---------------------------------------------------------------------------


@pytest.fixture
async def plugin_and_channel():
    plugin, channel, db = await build_plugin_and_channel()
    yield plugin, channel, db
    task_plugin = plugin.pm.get_plugin("task")
    if task_plugin:
        await task_plugin.on_stop()
    await db.close()


class TestDequeueCorrelationResolution:
    async def test_inherits_correlation_id_from_meta(self, plugin_and_channel):
        plugin, channel, db = plugin_and_channel
        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=_make_text_response("hi"))
        plugin._client = mock_client

        await plugin.on_notify(
            channel=channel, source="task", text="standalone",
            tool_call_id=None, meta={"correlation_id": "inherited-123"},
        )
        await drain(plugin, channel)

        call_kwargs = plugin.pm.ahook.on_agent_response.call_args.kwargs
        assert call_kwargs["correlation_id"] == "inherited-123"
        assert call_kwargs["meta"] == {"correlation_id": "inherited-123"}

    async def test_mints_when_absent(self, plugin_and_channel):
        plugin, channel, db = plugin_and_channel
        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=_make_text_response("hi"))
        plugin._client = mock_client

        await plugin.on_notify(
            channel=channel, source="task", text="standalone",
            tool_call_id=None, meta=None,
        )
        await drain(plugin, channel)

        call_kwargs = plugin.pm.ahook.on_agent_response.call_args.kwargs
        assert call_kwargs["correlation_id"] is not None
        assert isinstance(call_kwargs["correlation_id"], str)
        # No origin field anywhere at the core level (R1).
        assert "origin" not in call_kwargs


class TestAttributionCarriesCorrelationAndMeta:
    async def test_attribution_snapshot(self, plugin_and_channel):
        from corvidae.attribution import get_attribution

        plugin, channel, db = plugin_and_channel
        captured = {}

        async def fake_chat(messages, **kwargs):
            captured.update(get_attribution())
            return _make_text_response("hi")

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(side_effect=fake_chat)
        plugin._client = mock_client

        await plugin.on_notify(
            channel=channel, source="task", text="hello",
            tool_call_id=None, meta={"task_id": "t-1"},
        )
        await drain(plugin, channel)

        assert captured.get("correlation_id") is not None
        assert captured.get("meta") == {"task_id": "t-1"}


# ---------------------------------------------------------------------------
# Behavioral: TaskPlugin meta round-trip
# ---------------------------------------------------------------------------


class TestTaskPluginMetaRoundTrip:
    async def test_on_task_complete_merges_task_meta_into_notify_meta(self):
        from corvidae.channel import Channel
        from corvidae.task import Task, TaskPlugin

        pm = MagicMock()
        pm.ahook.on_notify = AsyncMock()
        pm.ahook.send_tool_status = AsyncMock()
        plugin = TaskPlugin()
        plugin.pm = pm

        channel = Channel(transport="test", scope="round-trip")

        async def work():
            return "result"

        task = Task(
            work=work, channel=channel, correlation_id="cid-1",
            meta={"origin": "critique"}, tool_call_id=None,
        )
        await plugin._on_task_complete(task, "the result")

        call_kwargs = pm.ahook.on_notify.call_args.kwargs
        assert call_kwargs["meta"] == {
            "task_id": task.task_id,
            "correlation_id": "cid-1",
            "origin": "critique",
        }


# ---------------------------------------------------------------------------
# Behavioral: on_message_persisted firing discipline
# ---------------------------------------------------------------------------


class TestOnMessagePersistedFiringDiscipline:
    async def test_fires_with_text_and_nullable_rowid_for_originating_items(
        self, plugin_and_channel
    ):
        plugin, channel, db = plugin_and_channel
        # Force rowid=None: no on_conversation_event listener returns a value.
        plugin.pm.ahook.on_conversation_event = AsyncMock(return_value=[None])
        persisted = AsyncMock()
        plugin.pm.ahook.on_message_persisted = persisted

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=_make_text_response("hi"))
        plugin._client = mock_client

        await plugin.on_message(channel=channel, sender="user", text="hello")
        await drain(plugin, channel)

        persisted.assert_awaited_once()
        call_kwargs = persisted.call_args.kwargs
        assert call_kwargs["rowid"] is None
        assert call_kwargs["text"] == "hello"
        assert call_kwargs["correlation_id"] is not None
