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
from tests.cognition.trace_helpers import add_trace
from fake_clock import install_kernel_clock
from helpers import build_plugin_and_channel, window_of
from test_lease_scopes import ScriptedSession, loud_unleased_records, usage_rows

from corvidae.attribution import reset_attribution, set_attribution
from corvidae.channel import Channel
from corvidae.llm_plugin import LLMPlugin
from corvidae.metrics import UsageLogPlugin

from .test_appraisal_stage2 import VALID_STAGE2_JSON
from .test_memory_consolidation import CONSOLIDATION_JSON, EMBED_CONFIG, seed_dialog

# The per-run consolidation lease key (#985, #1016):
# memory:consolidation:<channel id>:<id>, where <id> is the run's trace_id or a
# minted id (never None/empty). Two concurrent runs get distinct keys — by run,
# and by channel, so a trace shared across channels cannot merge two queues.
CONSOLIDATION_KEY_PREFIX = "memory:consolidation:"


def _is_consolidation_key(key: object) -> bool:
    """True for a per-run consolidation key with a non-empty id suffix."""
    return (
        isinstance(key, str)
        and key.startswith(CONSOLIDATION_KEY_PREFIX)
        and len(key) > len(CONSOLIDATION_KEY_PREFIX)
    )
# The combined critique fork answers one key per lens (#1190).
CRITIQUE_VERDICT = json.dumps({"predictive": {"objections": [
    {"claim": "it was Tuesday", "objection": "you said Monday"}]}})
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
    # memory's on_start resolves the kernel clock and idle tracker (#1180).
    install_kernel_clock(pm)
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
    # The reporters fire on_trace_record, so the trace plugin comes first.
    add_trace(pm)
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
        # Appraisal has no on_start of its own (#1189).
        if p is not appraisal:
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
            if isinstance(last, str) and prompt in last:
                return _make_text_response(reply)
        return _make_text_response("ok")

    session.respond = respond
    yield Env(plugin, channel, db, session, memory, appraisal, critique, outcome, llm)
    await memory.wait_for_background_tasks()
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
    await env.appraisal.on_message_persisted(
        channel=env.channel, trace_id="k", rowid=41, text="is the heater the culprit?", annotations={})

    await env.appraisal._run_stage2(env.channel, "k", "is the heater the culprit?", "yes, brownout", None)

    rows = await usage_rows(env.db)
    assert [key for stage, key, _m in rows if stage == "appraisal"] == [env.channel.id]
    assert loud_unleased_records(caplog) == []


async def test_critique_lens_is_keyed_by_channel(env: Env, caplog: pytest.LogCaptureFixture) -> None:
    """AC8: a critique lens (a fork) carries the channel id."""
    caplog.set_level(logging.WARNING)
    env.channel.conversation = window_of(env.channel.id, [7, 8])
    await env.critique.on_message_persisted(
        channel=env.channel, trace_id="k", rowid=7, text="when is the meeting?", annotations={})

    # The before-send point runs the lens fork inline through the public hook
    # (the idle runner this called before was folded into the one fork, #1190).
    await env.critique.on_appraisal_before_send(
        channel=env.channel, trace_id="k", text="It was Tuesday, as you said.",
        annotations={}, lenses={"predictive": 1.0})

    rows = await usage_rows(env.db)
    assert [key for stage, key, _m in rows if stage == "critique"] == [env.channel.id]
    assert loud_unleased_records(caplog) == []


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
    assert loud_unleased_records(caplog) == []


@pytest.mark.parametrize("via_idle_watch", [False, True])
async def test_catchup_consolidation_view_is_keyed_by_upkeep(
    env: Env, caplog: pytest.LogCaptureFixture, via_idle_watch: bool
) -> None:
    """The catch-up view (a range out of the window) carries a per-run
    memory:consolidation:<channel id>:<id> key, also when the idle watch triggers
    it from an empty context."""
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
    view_keys = [key for kind, key in pairs if kind == "view"]
    assert len(view_keys) == 1
    assert _is_consolidation_key(view_keys[0]), view_keys
    assert loud_unleased_records(caplog) == []


async def test_rubric_score_during_catchup_commit_is_keyed_by_upkeep(
    env: Env, caplog: pytest.LogCaptureFixture
) -> None:
    """The rubric score call made while a catch-up commit runs carries the
    run's memory:consolidation:<channel id>:<id> key — and the same key as the view that
    run produced (the id is minted once per run, not per call)."""
    caplog.set_level(logging.WARNING)
    persistence = env.plugin.pm.get_plugin("persistence")
    ids = await seed_dialog(persistence, env.channel, DIALOG)
    env.channel.conversation = None

    await env.memory._consolidate_range(env.channel.id, ids[-1], "compaction")

    pairs = await keyed_posts(env)
    view_keys = [key for kind, key in pairs if kind == "view"]
    rubric_keys = [key for kind, key in pairs if kind == "rubric"]
    assert len(view_keys) == 1 and len(rubric_keys) == 1, (view_keys, rubric_keys)
    assert _is_consolidation_key(view_keys[0]) and _is_consolidation_key(rubric_keys[0])
    # Intra-run stability: one id for the whole run, so the view and the rubric
    # it commits share a key.
    assert view_keys[0] == rubric_keys[0], (view_keys, rubric_keys)
    assert loud_unleased_records(caplog) == []


async def test_rubric_after_fork_uses_the_per_run_key(
    env: Env, caplog: pytest.LogCaptureFixture
) -> None:
    """After a fork record, the rubric (called in _commit) runs under the run's
    per-run key, not the channel key the fork re-keyed to (#985): the fork's
    lease scope restores the outer per-run key on exit. This is the trickiest
    path the PR touches — outer per-run scope, fork re-keys to the channel,
    rubric runs back under the per-run key."""
    caplog.set_level(logging.WARNING)
    persistence = env.plugin.pm.get_plugin("persistence")
    ids = await seed_dialog(persistence, env.channel, DIALOG)
    # The range is in the live window: _live_record takes the fork path
    # (outcome == "record"), which commits via _commit -> rubric (the fork
    # record has no importance, so the rubric rates the range).
    env.channel.conversation = window_of(
        env.channel.id, ids, contents=[text for _role, text in DIALOG])

    await env.memory._consolidate_range(env.channel.id, ids[-1], "idle")
    await env.memory.wait_for_background_tasks()

    rows = await usage_rows(env.db)
    fork_keys = [key for stage, key, mode in rows if stage == "consolidation" and mode == "fork"]
    rubric_keys = [key for stage, key, mode in rows if stage == "consolidation" and mode == "rubric"]
    assert len(fork_keys) == 1, (fork_keys, rows)
    # The fork carries the channel key (re-keyed inside _live_record).
    assert fork_keys[0] == env.channel.id, fork_keys
    # The rubric (after the fork) carries the run's per-run key.
    assert len(rubric_keys) == 1, (rubric_keys, rows)
    assert _is_consolidation_key(rubric_keys[0]), rubric_keys
    # The fork and rubric keys differ: the fork is the channel, the rubric is
    # the per-run key (the fork's lease scope restored the outer key).
    assert fork_keys[0] != rubric_keys[0], (fork_keys, rubric_keys)
    assert loud_unleased_records(caplog) == []


async def test_two_channels_sharing_a_trace_get_distinct_consolidation_keys(
    env: Env,
) -> None:
    """The reason the key carries the channel (#1016).

    Two runs that share a trace_id used to land on one queue. Nothing produces
    that today, but the key no longer depends on it not happening.
    """
    from corvidae.attribution import reset_attribution, set_attribution

    persistence = env.plugin.pm.get_plugin("persistence")
    ids = await seed_dialog(persistence, env.channel, DIALOG)
    env.channel.conversation = None
    registry = env.plugin.pm.get_plugin("registry")
    other = registry.get_or_create(env.channel.transport, "other-scope")

    other_ids = await seed_dialog(persistence, other, DIALOG)
    other.conversation = None
    token = set_attribution(trace_id="shared-trace-42")
    try:
        await env.memory._consolidate_range(env.channel.id, ids[-1], "compaction")
        await env.memory._consolidate_range(other.id, other_ids[-1], "compaction")
    finally:
        reset_attribution(token)
    await env.memory.wait_for_background_tasks()

    keys = [key for kind, key in await keyed_posts(env) if kind == "view"]
    assert keys == [
        f"memory:consolidation:{env.channel.id}:shared-trace-42",
        f"memory:consolidation:{other.id}:shared-trace-42",
    ], keys
    assert len(set(keys)) == 2


async def test_consolidation_with_trace_id_uses_it_as_the_key(
    env: Env, caplog: pytest.LogCaptureFixture
) -> None:
    """The run's trace_id, when attribution holds one, is the key suffix
    (#985, #1016): the key is exactly
    memory:consolidation:<channel-id>:<that-trace-id>. The
    trace_id is an attribution field independent of the trigger; a compaction
    range that has left the window is rebuilt as a view, so the per-run key is
    observable."""
    caplog.set_level(logging.WARNING)
    persistence = env.plugin.pm.get_plugin("persistence")
    ids = await seed_dialog(persistence, env.channel, DIALOG)
    env.channel.conversation = None

    # An outer attribution with a known trace_id (as a turn would set).
    token = set_attribution(trace_id="known-trace-123")
    try:
        await env.memory._consolidate_range(env.channel.id, ids[-1], "compaction")
    finally:
        reset_attribution(token)
    await env.memory.wait_for_background_tasks()

    pairs = await keyed_posts(env)
    view_keys = [key for kind, key in pairs if kind == "view"]
    assert view_keys == [
        f"memory:consolidation:{env.channel.id}:known-trace-123"
    ], view_keys
    assert loud_unleased_records(caplog) == []


async def test_consolidation_runs_have_distinct_keys(
    env: Env, caplog: pytest.LogCaptureFixture
) -> None:
    """Two non-fork consolidation runs get distinct per-run keys (#985), so the
    lease attribution (usage_log.lease_key) is per stream, not one shared
    upkeep queue. Both runs have no trace id (compaction trigger), so each
    mints its own id."""
    caplog.set_level(logging.WARNING)
    persistence = env.plugin.pm.get_plugin("persistence")
    env.channel.conversation = None

    ids1 = await seed_dialog(persistence, env.channel, DIALOG)
    await env.memory._consolidate_range(env.channel.id, ids1[-1], "compaction")
    await env.memory.wait_for_background_tasks()

    # A fresh range past the watermark: a second run, its own queue.
    second = [
        ("user", "the radio dropped again"),
        ("assistant", "same brownout pattern, likely the detector"),
        ("user", "yes, every time the heater kicks in"),
        ("assistant", "check the detector fuse and the brownout setting"),
    ]
    ids2 = await seed_dialog(persistence, env.channel, second)
    await env.memory._consolidate_range(env.channel.id, ids2[-1], "compaction")
    await env.memory.wait_for_background_tasks()

    pairs = await keyed_posts(env)
    view_keys = [key for kind, key in pairs if kind == "view"]
    # One view per run; both per-run consolidation keys, distinct.
    assert len(view_keys) == 2, view_keys
    assert all(_is_consolidation_key(k) for k in view_keys), view_keys
    assert view_keys[0] != view_keys[1], view_keys
    assert loud_unleased_records(caplog) == []
