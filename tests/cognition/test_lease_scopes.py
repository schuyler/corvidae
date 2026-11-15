"""Lease scopes (#363), AC8's cognition call sites.

Each path is driven through the real Agent, LLMPlugin and LLMClient (HTTP
stubbed) with UsageLogPlugin registered, and the test asserts the
``lease_key`` of the usage_log row the path wrote, plus that no unleased
warning was logged while it ran. The kernel paths are in
tests/test_lease_scopes.py; they live apart because kernel tests must not
import corvidae_cognition (tests/test_kernel_boundary.py).

``lease_key`` does not exist before the change, so every test fails on the
missing usage_log column until it does.
"""

import json
import logging
from dataclasses import dataclass
from typing import Any

import aiohttp
import aiosqlite
import pytest
from corvidae_cognition.appraisal import DEFAULT_STAGE2_FORK_PROMPT, AppraisalPlugin
from corvidae_cognition.critique import DEFAULT_FORK_PROMPTS, CritiquePlugin
from corvidae_cognition.memory import DEFAULT_CONSOLIDATION_FORK_PROMPT, MemoryPlugin
from corvidae_cognition.outcome_log import OutcomeLogPlugin
from helpers import build_plugin_and_channel, window_of
from test_lease_scopes import ScriptedSession, unleased_records, usage_rows

from corvidae.channel import Channel
from corvidae.llm_plugin import LLMPlugin
from corvidae.metrics import UsageLogPlugin

from .test_appraisal_stage2 import VALID_STAGE2_JSON
from .test_memory_consolidation import CONSOLIDATION_JSON, EMBED_CONFIG, seed_dialog

# The key AC8 names for catch-up consolidation and the rubric call.
UPKEEP_KEY = "upkeep:memory:consolidation"
CRITIQUE_VERDICT = json.dumps(
    {"objections": [{"claim": "it was Tuesday", "objection": "you said Monday"}]}
)
RUBRIC_REPLY = json.dumps({"importance": 0.6})
DIALOG = [
    ("user", "my weather station keeps dropping wifi"),
    ("assistant", "Does it happen when the heater kicks on?"),
    ("user", "yes, every time"),
    ("assistant", "That's the brownout detector resetting the radio."),
]


class _EmbedStub:
    """Embedding client stand-in: no HTTP, so no usage_log row."""

    async def stop(self) -> None:
        """LLMPlugin.on_stop closes every client; there is nothing to close."""

    def request_model(self, extra_body: dict | None = None) -> str:
        """The model name an embed call would be sent with."""
        return "embedder"

    async def embed(self, texts: list[str], kind: str | None = None) -> list[list[float]]:
        """One fixed vector per text."""
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


@dataclass
class Env:
    """The real kernel stack plus the cognition plugins under test."""

    plugin: Any
    channel: Channel
    db: aiosqlite.Connection
    session: ScriptedSession
    memory: MemoryPlugin
    appraisal: AppraisalPlugin
    critique: CritiquePlugin
    outcome: OutcomeLogPlugin
    llm: LLMPlugin


@pytest.fixture
async def env(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Agent, LLMPlugin (stubbed HTTP), UsageLogPlugin, memory, appraisal and
    critique on one pm, over a file DB under tmp_path. The scripted model
    answers each stage's prompt with a reply that stage can parse."""
    import corvidae.llm as llm_module

    # Forget which stages already warned, so the check sees a first call.
    monkeypatch.setattr(llm_module, "_unleased_warned_stages", set(), raising=False)

    db_path = str(tmp_path / "sessions.db")
    plugin, channel, db = await build_plugin_and_channel(db_path=db_path)
    # The memory plugin's dedicated writer needs the daemon's WAL mode.
    await db.execute("PRAGMA journal_mode=wal")
    pm = plugin.pm
    config = {
        **EMBED_CONFIG,
        "daemon": {"session_db": db_path},
        "agent": {"forks": {"enabled": True}},
    }
    await plugin.on_init(pm, config)

    # Every client posts through one scripted session; no network.
    session = ScriptedSession()
    monkeypatch.setattr(aiohttp, "ClientSession", lambda headers=None, **kw: session)
    llm = LLMPlugin()
    pm.register(llm, name="llm")
    await llm.on_init(pm, config)
    outcome = OutcomeLogPlugin()
    pm.register(outcome, name="outcome_log")
    memory = MemoryPlugin()
    pm.register(memory, name="memory")
    appraisal = AppraisalPlugin()
    pm.register(appraisal, name="appraisal")
    critique = CritiquePlugin()
    pm.register(critique, name="critique")
    usage_log = UsageLogPlugin()
    pm.register(usage_log, name="usage_log")
    await llm.on_start(config)
    for p in (outcome, memory, appraisal, critique, usage_log):
        await p.on_init(pm=pm, config=config)
        await p.on_start(config=config)
    # Embeddings are not under test: no HTTP and no usage row for them.
    llm._clients["embedding"] = _EmbedStub()
    plugin._client = llm.get_client()
    plugin._mark_started()

    def respond(body: dict) -> dict:
        """The reply each stage can parse, picked from its prompt."""
        from llm_response_fixtures import _make_text_response

        messages = body.get("messages") or [{}]
        last = messages[-1].get("content") or ""
        # The rubric call is the one headed by the rubric system prompt.
        if messages[0].get("content") == memory._rubric_prompt:
            return _make_text_response(RUBRIC_REPLY)
        for prompt, reply in (
            (DEFAULT_STAGE2_FORK_PROMPT, VALID_STAGE2_JSON),
            (DEFAULT_FORK_PROMPTS["predictive"], CRITIQUE_VERDICT),
            (DEFAULT_CONSOLIDATION_FORK_PROMPT, CONSOLIDATION_JSON),
        ):
            if isinstance(last, str) and last.startswith(prompt):
                return _make_text_response(reply)
        return _make_text_response("ok")

    session.respond = respond
    yield Env(plugin, channel, db, session, memory, appraisal, critique, outcome, llm)
    await memory.wait_for_background_tasks()
    await appraisal.on_stop()
    await critique.on_stop()
    await plugin.pm.get_plugin("task").on_stop()
    await llm.on_stop()
    await db.close()


def post_kinds(env: Env) -> list[str]:
    """"rubric" or "view" for each chat request the session saw, in order."""
    return [
        "rubric" if (p["json"]["messages"][0].get("content") == env.memory._rubric_prompt)
        else "view"
        for p in env.session.posts
    ]


async def keyed_posts(env: Env) -> list[tuple[str, str | None]]:
    """(kind, lease_key) per request, pairing each post with its usage row."""
    rows = await usage_rows(env.db)
    kinds = post_kinds(env)
    assert len(rows) == len(kinds), (rows, kinds)
    return [(kind, key) for kind, (_stage, key, _mode) in zip(kinds, rows)]


async def test_appraisal_stage2_is_keyed_by_channel(env: Env, caplog: pytest.LogCaptureFixture) -> None:
    """AC8: appraisal stage 2 (a fork) carries the channel id."""
    caplog.set_level(logging.WARNING)
    env.channel.conversation = window_of(env.channel.id, [41, 42])
    await env.outcome.record_exchange("k", env.channel.id, origin="user")
    await env.appraisal.on_message_persisted(
        channel=env.channel, trace_id="k", rowid=41, text="is the heater the culprit?", annotations={})

    await env.appraisal._run_stage2(env.channel, "k", "is the heater the culprit?", "yes, brownout", None)

    rows = await usage_rows(env.db)
    assert [key for stage, key, _m in rows if stage == "appraisal"] == [env.channel.id]
    assert unleased_records(caplog, logging.WARNING) == []


async def test_critique_lens_is_keyed_by_channel(env: Env, caplog: pytest.LogCaptureFixture) -> None:
    """AC8: a critique lens (a fork) carries the channel id."""
    caplog.set_level(logging.WARNING)
    env.channel.conversation = window_of(env.channel.id, [7, 8])
    await env.outcome.record_exchange("k", env.channel.id, origin="user")
    await env.critique.on_message_persisted(
        channel=env.channel, trace_id="k", rowid=7, text="when is the meeting?", annotations={})

    await env.critique._run_critique(
        env.channel, "k", "when is the meeting?", "It was Tuesday, as you said.",
        ["predictive"], False)

    rows = await usage_rows(env.db)
    assert [key for stage, key, _m in rows if stage == "critique"] == [env.channel.id]
    assert unleased_records(caplog, logging.WARNING) == []


async def test_compaction_consolidation_fork_is_keyed_by_channel(
    env: Env, caplog: pytest.LogCaptureFixture
) -> None:
    """AC8: the consolidation fork spawned at compaction (_compaction_fork)
    carries the channel id."""
    caplog.set_level(logging.WARNING)
    persistence = env.plugin.pm.get_plugin("persistence")
    ids = await seed_dialog(persistence, env.channel, DIALOG)
    env.channel.conversation = window_of(
        env.channel.id, ids, contents=[text for _role, text in DIALOG])

    await env.memory.before_compaction(
        channel=env.channel, compacted_ids=ids[:3], cut=3)
    await env.memory.wait_for_background_tasks()

    rows = await usage_rows(env.db)
    assert rows == [("consolidation", env.channel.id, "fork")]
    assert unleased_records(caplog, logging.WARNING) == []


@pytest.mark.parametrize("via_idle_watch", [False, True])
async def test_catchup_consolidation_view_is_keyed_by_upkeep(
    env: Env, caplog: pytest.LogCaptureFixture, via_idle_watch: bool
) -> None:
    """AC8: the catch-up view (a range out of the window) carries
    upkeep:memory:consolidation, also when the idle watch triggers it from an
    empty context."""
    caplog.set_level(logging.WARNING)
    persistence = env.plugin.pm.get_plugin("persistence")
    ids = await seed_dialog(persistence, env.channel, DIALOG)
    # No live window: the range has left it, so the view is rebuilt.
    env.channel.conversation = None

    if via_idle_watch:
        await env.memory._on_channel_idle(env.channel.id)
    else:
        await env.memory._consolidate_range(env.channel.id, ids[-1], "compaction")
    await env.memory.wait_for_background_tasks()

    pairs = await keyed_posts(env)
    assert [key for kind, key in pairs if kind == "view"] == [UPKEEP_KEY]
    assert unleased_records(caplog, logging.WARNING) == []


async def test_rubric_score_during_catchup_commit_is_keyed_by_upkeep(
    env: Env, caplog: pytest.LogCaptureFixture
) -> None:
    """AC8: the rubric score call made while a catch-up commit runs carries
    upkeep:memory:consolidation."""
    caplog.set_level(logging.WARNING)
    persistence = env.plugin.pm.get_plugin("persistence")
    ids = await seed_dialog(persistence, env.channel, DIALOG)
    env.channel.conversation = None

    await env.memory._consolidate_range(env.channel.id, ids[-1], "compaction")

    pairs = await keyed_posts(env)
    assert [key for kind, key in pairs if kind == "rubric"] == [UPKEEP_KEY]
    assert unleased_records(caplog, logging.WARNING) == []
