"""RED tests for WP2.5 — AppraisalPlugin stage 2: full appraisal, importance
prior, valence.

Per plans/implementation/phase-2.md WP2.5:
- on_agent_response enqueues exactly one SILENT (deliver=False) stage-2 task,
  stamped with the exchange key/origin; no on_notify fires from it;
- the task body sets attribution stage="appraisal" with the exchange key, makes
  one tier-3 call (appraisal → background → main fallback), and MERGE-persists
  {"stage2": ..., "entropy": ...} into the appraisal envelope — never a full
  overwrite (stage1/stage1_out survive);
- malformed model output is logged and the row keeps stage 1 only; no exception
  escapes;
- no stage-2 task for origin="critique" exchanges;
- AppraisalPrior scores a covered msg-id range from stored appraisals and falls
  back to the wrapped prior when uncovered; consolidation writes mean stage-2
  valence into the memory record;
- entropy_summary summarizes an OpenAI-style logprobs envelope (topn + residual,
  NLL fallback, absent → None).
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import pytest

from corvidae.attribution import get_attribution
from corvidae.channel import Channel, ChannelConfig
from corvidae.hooks import create_plugin_manager
from corvidae.llm_plugin import LLMPlugin
from corvidae.outcome_log import OutcomeLogPlugin
from corvidae.persistence import PersistencePlugin, init_db
from corvidae.task import TaskPlugin


VALID_STAGE2_JSON = json.dumps({
    "valence": 0.8, "stakes": 0.9, "ambiguity": 0.2,
    "commitment_density": 0.5, "novelty": 0.7, "correction": True,
})


def _make_channel(scope="stage2") -> Channel:
    return Channel(transport="test", scope=scope, config=ChannelConfig())


class StubClient:
    """Minimal LLMClient stand-in recording chat calls + attribution."""

    def __init__(self, chat_text: str = VALID_STAGE2_JSON):
        self.chat_text = chat_text
        self.chat_calls: list[dict] = []

    async def chat(self, messages, tools=None, extra_body=None):
        self.chat_calls.append({
            "messages": messages,
            "extra_body": extra_body,
            "attribution": dict(get_attribution()),
        })
        return {"choices": [{"message": {"role": "assistant", "content": self.chat_text}}]}


async def build_env(tmp_path, chat_text=VALID_STAGE2_JSON, start_worker=False):
    """Wire persistence + outcome_log + llm(stub) + task + appraisal against a
    file-backed session DB. Returns a SimpleNamespace of the pieces."""
    from corvidae.appraisal import AppraisalPlugin

    db_path = str(tmp_path / "sessions.db")
    db = await aiosqlite.connect(db_path)
    await init_db(db)

    pm = create_plugin_manager()
    persistence = PersistencePlugin()
    persistence.db = db
    pm.register(persistence, name="persistence")

    outcome = OutcomeLogPlugin()
    pm.register(outcome, name="outcome_log")
    await outcome.on_init(pm=pm, config={})
    await outcome.on_start(config={})

    llm = LLMPlugin()
    pm.register(llm, name="llm")
    stub = StubClient(chat_text)
    llm._clients["main"] = stub
    llm._clients["background"] = stub

    task_plugin = TaskPlugin()
    pm.register(task_plugin, name="task")
    await task_plugin.on_init(pm=pm, config={})
    # Always create the queue so on_agent_response can enqueue; only spin the
    # worker when the test wants the silent-task delivery path exercised.
    from corvidae.task import TaskQueue
    task_plugin.task_queue = TaskQueue(max_workers=1)
    if start_worker:
        task_plugin._worker_task = asyncio.create_task(
            task_plugin.task_queue.run_worker(task_plugin._on_task_complete)
        )

    appraisal = AppraisalPlugin()
    pm.register(appraisal, name="appraisal")
    cfg = {"daemon": {"session_db": db_path}}
    await appraisal.on_init(pm=pm, config=cfg)
    await appraisal.on_start(config=cfg)

    channel = _make_channel()
    return SimpleNamespace(
        pm=pm, db=db, db_path=db_path, appraisal=appraisal, task=task_plugin,
        outcome=outcome, llm=llm, stub=stub, channel=channel,
    )


async def _teardown(env):
    await env.appraisal.on_stop()
    worker = getattr(env.task, "_worker_task", None)
    if worker is not None:
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass
    await env.db.close()


async def _run_next_task(task_plugin):
    """Pull the single queued task and run its work() inside its context."""
    queue = task_plugin.task_queue
    assert queue is not None
    task = queue.queue.get_nowait()
    result = await asyncio.create_task(task.work(), context=task.ctx)
    return task, result


async def _read_appraisal(db, key):
    async with db.execute(
        "SELECT appraisal FROM exchange_log WHERE exchange_key = ?", (key,)
    ) as cursor:
        row = await cursor.fetchone()
    return json.loads(row[0]) if row and row[0] else None


# ---------------------------------------------------------------------------
# 1. Trigger: exactly one silent task, stamped, attributed
# ---------------------------------------------------------------------------


class TestTrigger:
    async def test_one_silent_task_enqueued_and_stamped(self, tmp_path):
        env = await build_env(tmp_path)
        await env.appraisal.on_agent_response(
            channel=env.channel, request_text="req", response_text="resp",
            exchange_key="k1", origin="user", originating_text="hi there",
            logprobs=None, withheld=False,
        )
        queue = env.task.task_queue
        assert queue.queue.qsize() == 1
        task = queue.queue.get_nowait()
        assert task.deliver is False
        assert task.tool_call_id is None
        assert task.exchange_key == "k1"
        assert task.origin == "user"
        assert "appraisal" in task.description
        await _teardown(env)

    async def test_no_task_for_critique_origin(self, tmp_path):
        env = await build_env(tmp_path)
        await env.appraisal.on_agent_response(
            channel=env.channel, request_text="req", response_text="resp",
            exchange_key="kc", origin="critique", originating_text="hi",
            logprobs=None, withheld=False,
        )
        assert env.task.task_queue.queue.qsize() == 0
        await _teardown(env)

    async def test_silent_task_fires_no_on_notify(self, tmp_path):
        """Run the task through the real worker: deliver=False → no on_notify,
        no main-model turn (trap #10)."""
        env = await build_env(tmp_path, start_worker=True)
        notified = []

        from corvidae.hooks import CorvidaePlugin, hookimpl

        class NotifySpy(CorvidaePlugin):
            depends_on = frozenset()

            @hookimpl
            async def on_notify(self, channel, source, text, tool_call_id, meta):
                notified.append(source)

        env.pm.register(NotifySpy(), name="notify_spy")

        await env.appraisal.on_agent_response(
            channel=env.channel, request_text="req", response_text="resp",
            exchange_key="k2", origin="user", originating_text="hi",
            logprobs=None, withheld=False,
        )
        # Let the worker drain.
        import asyncio
        for _ in range(50):
            if env.task.task_queue.is_idle:
                break
            await asyncio.sleep(0.01)
        assert notified == []
        await _teardown(env)

    async def test_task_body_sets_appraisal_attribution(self, tmp_path):
        env = await build_env(tmp_path)
        # Seed the row so update_exchange lands.
        await env.outcome.record_exchange("k3", env.channel.id, origin="user")
        await env.appraisal.on_agent_response(
            channel=env.channel, request_text="req", response_text="resp",
            exchange_key="k3", origin="user", originating_text="hi",
            logprobs=None, withheld=False,
        )
        await _run_next_task(env.task)
        assert len(env.stub.chat_calls) == 1
        attr = env.stub.chat_calls[0]["attribution"]
        assert attr.get("stage") == "appraisal"
        assert attr.get("exchange_key") == "k3"
        await _teardown(env)


# ---------------------------------------------------------------------------
# 2. Persist: stage-2 JSON under the key; merge-not-overwrite; malformed safe
# ---------------------------------------------------------------------------


class TestStage2Persist:
    async def test_stage2_persisted_under_key(self, tmp_path):
        env = await build_env(tmp_path)
        await env.outcome.record_exchange("kp", env.channel.id, origin="user")
        await env.appraisal.on_agent_response(
            channel=env.channel, request_text="req", response_text="resp",
            exchange_key="kp", origin="user", originating_text="hi",
            logprobs=None, withheld=False,
        )
        await _run_next_task(env.task)
        env_appraisal = await _read_appraisal(env.db, "kp")
        assert env_appraisal["stage2"]["valence"] == 0.8
        assert env_appraisal["stage2"]["stakes"] == 0.9
        assert env_appraisal["stage2"]["correction"] is True
        # Advisory synchronous reader populated.
        assert env.appraisal.get_last_stage2(env.channel.id)["valence"] == 0.8
        await _teardown(env)

    async def test_merge_not_overwrite_keeps_stage1_and_stage1_out(self, tmp_path):
        """After stage 2 lands on an exchange already carrying stage1 +
        stage1_out, the envelope still contains all three."""
        env = await build_env(tmp_path)
        await env.outcome.upsert_exchange(
            "km", env.channel.id, "user",
            appraisal={"stage1": {"salience": 0.3}, "stage1_out": {"salience": 0.4}},
        )
        await env.appraisal.on_agent_response(
            channel=env.channel, request_text="req", response_text="resp",
            exchange_key="km", origin="user", originating_text="hi",
            logprobs=None, withheld=False,
        )
        await _run_next_task(env.task)
        envelope = await _read_appraisal(env.db, "km")
        assert envelope["stage1"] == {"salience": 0.3}
        assert envelope["stage1_out"] == {"salience": 0.4}
        assert envelope["stage2"]["valence"] == 0.8
        await _teardown(env)

    async def test_malformed_output_keeps_stage1_no_exception(self, tmp_path):
        env = await build_env(tmp_path, chat_text="not json at all")
        await env.outcome.upsert_exchange(
            "kbad", env.channel.id, "user", appraisal={"stage1": {"salience": 0.5}},
        )
        await env.appraisal.on_agent_response(
            channel=env.channel, request_text="req", response_text="resp",
            exchange_key="kbad", origin="user", originating_text="hi",
            logprobs=None, withheld=False,
        )
        # Must not raise.
        await _run_next_task(env.task)
        envelope = await _read_appraisal(env.db, "kbad")
        assert envelope == {"stage1": {"salience": 0.5}}  # stage2 absent
        assert env.appraisal.get_last_stage2(env.channel.id) is None
        await _teardown(env)

    async def test_entropy_persisted_when_logprobs_present(self, tmp_path):
        env = await build_env(tmp_path)
        await env.outcome.record_exchange("ke", env.channel.id, origin="user")
        logprobs = {"content": [
            {"token": "a", "logprob": -0.1,
             "top_logprobs": [{"token": "a", "logprob": -0.1},
                              {"token": "b", "logprob": -2.3}]},
            {"token": "c", "logprob": -0.5,
             "top_logprobs": [{"token": "c", "logprob": -0.5},
                              {"token": "d", "logprob": -1.0}]},
        ]}
        await env.appraisal.on_agent_response(
            channel=env.channel, request_text="req", response_text="resp",
            exchange_key="ke", origin="user", originating_text="hi",
            logprobs=logprobs, withheld=False,
        )
        await _run_next_task(env.task)
        envelope = await _read_appraisal(env.db, "ke")
        assert envelope["entropy"]["kind"] == "topn"
        assert envelope["entropy"]["n_tokens"] == 2
        assert envelope["entropy"]["mean"] >= 0.0
        await _teardown(env)


# ---------------------------------------------------------------------------
# 3. entropy_summary unit tests
# ---------------------------------------------------------------------------


class TestEntropySummary:
    def test_absent_logprobs_returns_none(self):
        from corvidae.appraisal import entropy_summary
        assert entropy_summary(None) is None
        assert entropy_summary({}) is None
        assert entropy_summary({"content": []}) is None

    def test_topn_with_residual(self):
        from corvidae.appraisal import entropy_summary
        env = entropy_summary({"content": [
            {"token": "x", "logprob": -0.05,
             "top_logprobs": [{"token": "x", "logprob": -0.05}]},
        ]})
        assert env["kind"] == "topn"
        assert env["n_tokens"] == 1
        assert env["mean"] > 0.0  # residual bucket contributes entropy

    def test_nll_fallback_when_no_top_logprobs(self):
        from corvidae.appraisal import entropy_summary
        env = entropy_summary({"content": [
            {"token": "x", "logprob": -1.0},
            {"token": "y", "logprob": -2.0},
        ]})
        assert env["kind"] == "nll"
        assert env["n_tokens"] == 2
        assert env["mean"] == pytest.approx(1.5)
        assert env["max"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# 4. AppraisalPrior + mean_valence over a msg-id range
# ---------------------------------------------------------------------------


class TestAppraisalPrior:
    async def _seed(self, env, key, rowid, stage1=None, stage2=None):
        cols = {}
        env_appraisal = {}
        if stage1 is not None:
            env_appraisal["stage1"] = stage1
        if stage2 is not None:
            env_appraisal["stage2"] = stage2
        await env.outcome.upsert_exchange(
            key, env.channel.id, "user", appraisal=env_appraisal,
        )
        await env.outcome.update_exchange(key, message_rowid=rowid)

    async def test_covered_range_returns_composite(self, tmp_path):
        env = await build_env(tmp_path)
        await self._seed(env, "e1", 10, stage1={"salience": 0.2},
                         stage2={"stakes": 1.0, "valence": 1.0, "novelty": 1.0})
        score = await env.appraisal.importance_over_range(env.channel, (5, 15))
        # composite = 0.4*1 + 0.3*|1-0.5|*2 + 0.3*1 = 1.0; max(0.2, 1.0) = 1.0
        assert score == pytest.approx(1.0)
        await _teardown(env)

    async def test_stage1_only_uses_salience(self, tmp_path):
        env = await build_env(tmp_path)
        await self._seed(env, "e2", 20, stage1={"salience": 0.42})
        score = await env.appraisal.importance_over_range(env.channel, (15, 25))
        assert score == pytest.approx(0.42)
        await _teardown(env)

    async def test_uncovered_range_returns_none(self, tmp_path):
        env = await build_env(tmp_path)
        await self._seed(env, "e3", 100, stage1={"salience": 0.9})
        assert await env.appraisal.importance_over_range(env.channel, (1, 10)) is None
        await _teardown(env)

    async def test_prior_falls_back_when_uncovered(self, tmp_path):
        from corvidae.appraisal import AppraisalPrior
        env = await build_env(tmp_path)
        fallback = MagicMock()
        fallback.score = AsyncMock(return_value=0.33)
        prior = AppraisalPrior(appraisal=env.appraisal, fallback=fallback)
        result = await prior.score([{"role": "user", "content": "x"}],
                                   msg_id_range=(1, 5), channel=env.channel)
        assert result == 0.33
        fallback.score.assert_awaited_once()
        await _teardown(env)

    async def test_prior_uses_appraisal_when_covered(self, tmp_path):
        from corvidae.appraisal import AppraisalPrior
        env = await build_env(tmp_path)
        await self._seed(env, "e4", 30, stage1={"salience": 0.55})
        fallback = MagicMock()
        fallback.score = AsyncMock(return_value=0.1)
        prior = AppraisalPrior(appraisal=env.appraisal, fallback=fallback)
        result = await prior.score([{"role": "user", "content": "x"}],
                                   msg_id_range=(25, 35), channel=env.channel)
        assert result == pytest.approx(0.55)
        fallback.score.assert_not_awaited()
        await _teardown(env)

    async def test_mean_valence_over_range(self, tmp_path):
        env = await build_env(tmp_path)
        await self._seed(env, "v1", 40, stage2={"valence": 0.2})
        await self._seed(env, "v2", 41, stage2={"valence": 0.8})
        await self._seed(env, "v3", 42, stage1={"salience": 0.5})  # no stage2
        mean = await env.appraisal.mean_valence((35, 45))
        assert mean == pytest.approx(0.5)
        # No stage-2 anywhere → None.
        assert await env.appraisal.mean_valence((100, 200)) is None
        await _teardown(env)

    async def test_consolidation_writes_mean_valence(self, tmp_path):
        """End-to-end: consolidation writes mean stage-2 valence into the
        memory record and uses the appraisal-driven importance prior."""
        from corvidae.appraisal import AppraisalPlugin
        from corvidae.channel import Channel, ChannelConfig
        from corvidae.context import MessageType
        from corvidae.memory import MemoryPlugin

        consolidation_json = json.dumps({
            "summary": "I discussed the plan with the user.",
            "topic_tags": ["plan"],
            "participants": ["user"],
        })

        db_path = str(tmp_path / "sessions.db")
        db = await aiosqlite.connect(db_path)
        await init_db(db)
        pm = create_plugin_manager()
        persistence = PersistencePlugin()
        persistence.db = db
        pm.register(persistence, name="persistence")

        outcome = OutcomeLogPlugin()
        pm.register(outcome, name="outcome_log")
        await outcome.on_init(pm=pm, config={})
        await outcome.on_start(config={})

        llm = LLMPlugin()
        pm.register(llm, name="llm")
        llm._clients["main"] = StubClient(consolidation_json)
        llm._clients["background"] = StubClient(consolidation_json)

        cfg = {"daemon": {"session_db": db_path}, "memory": {}}
        memory = MemoryPlugin()
        pm.register(memory, name="memory")
        await memory.on_init(pm=pm, config=cfg)
        await memory.on_start(config=cfg)

        appraisal = AppraisalPlugin()
        pm.register(appraisal, name="appraisal")
        await appraisal.on_init(pm=pm, config=cfg)
        await appraisal.on_start(config=cfg)  # installs AppraisalPrior on memory

        channel = Channel(transport="test", scope="conv", config=ChannelConfig())

        # Seed dialog into message_log.
        rowids = []
        for role, content in [("user", "here is the plan"),
                              ("assistant", "understood, I will do it")]:
            rowids.append(await persistence.on_conversation_event(
                channel=channel,
                message={"role": role, "content": content},
                message_type=MessageType.MESSAGE,
            ))

        # Seed exchange_log appraisals covering the range with stage-2 valence.
        await outcome.upsert_exchange(
            "x1", channel.id, "user",
            appraisal={"stage1": {"salience": 0.3},
                       "stage2": {"stakes": 0.9, "valence": 0.9, "novelty": 0.5}},
        )
        await outcome.update_exchange("x1", message_rowid=rowids[0])
        await outcome.upsert_exchange(
            "x2", channel.id, "user",
            appraisal={"stage2": {"stakes": 0.1, "valence": 0.7, "novelty": 0.1}},
        )
        await outcome.update_exchange("x2", message_rowid=rowids[1])

        await memory._consolidate_range(channel.id, rowids[-1])

        async with db.execute(
            "SELECT importance, valence FROM memory WHERE channel_id = ?",
            (channel.id,),
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None
        importance, valence = row
        # Mean stage-2 valence = (0.9 + 0.7) / 2 = 0.8.
        assert valence == pytest.approx(0.8)
        # Importance came from the appraisal prior (covered range), not the
        # rubric fallback: max composite over the two exchanges.
        assert 0.0 <= importance <= 1.0

        await appraisal.on_stop()
        await db.close()

    async def test_prior_installed_on_memory_at_start(self, tmp_path):
        """AppraisalPlugin.on_start wraps MemoryPlugin.importance_prior."""
        from corvidae.appraisal import AppraisalPlugin, AppraisalPrior

        db_path = str(tmp_path / "sessions.db")
        db = await aiosqlite.connect(db_path)
        await init_db(db)
        pm = create_plugin_manager()
        persistence = PersistencePlugin()
        persistence.db = db
        pm.register(persistence, name="persistence")

        from corvidae.hooks import CorvidaePlugin

        class FakeMemory(CorvidaePlugin):
            depends_on = frozenset()

        memory = FakeMemory()
        original_prior = object()
        memory.importance_prior = original_prior
        pm.register(memory, name="memory")

        appraisal = AppraisalPlugin()
        pm.register(appraisal, name="appraisal")
        cfg = {"daemon": {"session_db": db_path}}
        await appraisal.on_init(pm=pm, config=cfg)
        await appraisal.on_start(config=cfg)

        assert isinstance(memory.importance_prior, AppraisalPrior)
        assert memory.importance_prior._fallback is original_prior
        await appraisal.on_stop()
        await db.close()
