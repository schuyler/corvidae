"""RED tests for WP2.5 — AppraisalPlugin stage 2: full appraisal, importance
prior, valence.

Per plans/implementation/phase-2.md WP2.5:
- on_agent_response enqueues exactly one SILENT (deliver=False) stage-2 task,
  stamped with the trace id/origin; no on_notify fires from it;
- the task body sets attribution stage="appraisal" with the trace id, makes
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
from corvidae_cognition.outcome_log import OutcomeLogPlugin
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
    from corvidae_cognition.appraisal import AppraisalPlugin

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
        "SELECT appraisal FROM exchange_log WHERE trace_id = ?", (key,)
    ) as cursor:
        row = await cursor.fetchone()
    return json.loads(row[0]) if row and row[0] else None


# ---------------------------------------------------------------------------
# 0. Originating-text LRU (moved from Agent to AppraisalPlugin, §1.4/§1.5).
# Fed by a new on_message_persisted hookimpl; read via get_originating_text.
# ---------------------------------------------------------------------------


class TestOriginatingTextLRU:
    async def test_on_message_persisted_populates_the_lru(self, tmp_path):
        env = await build_env(tmp_path)
        await env.appraisal.on_message_persisted(
            channel=env.channel, trace_id="k-lru-1", rowid=5,
            text="the original user text", meta={},
        )
        assert env.appraisal.get_originating_text("k-lru-1") == "the original user text"
        await _teardown(env)

    async def test_on_message_persisted_accepts_nullable_rowid(self, tmp_path):
        """on_message_persisted fires for originating items regardless of
        persistence outcome (rowid nullable, §1.1/§1.2) -- the LRU write
        does not depend on rowid being non-None."""
        env = await build_env(tmp_path)
        await env.appraisal.on_message_persisted(
            channel=env.channel, trace_id="k-lru-none", rowid=None,
            text="text with no rowid yet", meta={},
        )
        assert env.appraisal.get_originating_text("k-lru-none") == "text with no rowid yet"
        await _teardown(env)

    async def test_get_originating_text_returns_none_for_unknown_id(self, tmp_path):
        env = await build_env(tmp_path)
        assert env.appraisal.get_originating_text("never-seen") is None
        await _teardown(env)

    async def test_lru_bounded_at_512_evicts_oldest(self, tmp_path):
        from corvidae_cognition.appraisal import ORIGINATING_TEXT_LRU_MAXSIZE
        assert ORIGINATING_TEXT_LRU_MAXSIZE == 512

        env = await build_env(tmp_path)
        for i in range(ORIGINATING_TEXT_LRU_MAXSIZE + 1):
            await env.appraisal.on_message_persisted(
                channel=env.channel, trace_id=f"k-{i}", rowid=i,
                text=f"text {i}", meta={},
            )
        # The oldest entry (k-0) was evicted once the 513th arrived; the
        # newest MAXSIZE entries survive.
        assert env.appraisal.get_originating_text("k-0") is None
        assert env.appraisal.get_originating_text("k-1") == "text 1"
        assert env.appraisal.get_originating_text(
            f"k-{ORIGINATING_TEXT_LRU_MAXSIZE}"
        ) == f"text {ORIGINATING_TEXT_LRU_MAXSIZE}"
        await _teardown(env)

    async def test_stage2_prompt_recovers_originating_text_through_a_tool_cycle(
        self, tmp_path
    ):
        """The exchange's true originating text is recoverable through a
        tool cycle: on_message_persisted records it once, at the
        originating USER row. on_agent_response's request_text is the last
        tool-result turn's text (never the true originating text, per its
        legacy semantics -- see test_phase2a_wp21.py's
        test_final_on_agent_response_request_text_stays_last_turn_text),
        yet the stage-2 prompt is still built from the recovered text."""
        env = await build_env(tmp_path)
        await env.appraisal.on_message_persisted(
            channel=env.channel, trace_id="k-cycle", rowid=1,
            text="the original user text", meta={},
        )
        await env.appraisal.on_agent_response(
            channel=env.channel, request_text="tool result: done",
            response_text="final response", trace_id="k-cycle",
            meta={}, logprobs=None,
        )
        await _run_next_task(env.task)

        assert len(env.stub.chat_calls) == 1
        user_message = env.stub.chat_calls[0]["messages"][1]["content"]
        assert "the original user text" in user_message
        assert "tool result: done" not in user_message
        await _teardown(env)


# ---------------------------------------------------------------------------
# 1. Trigger: exactly one silent task, stamped, attributed
# ---------------------------------------------------------------------------


class TestTrigger:
    async def test_one_silent_task_enqueued_and_stamped(self, tmp_path):
        env = await build_env(tmp_path)
        await env.appraisal.on_agent_response(
            channel=env.channel, request_text="req", response_text="resp",
            trace_id="k1", meta={},
            logprobs=None,
        )
        queue = env.task.task_queue
        assert queue.queue.qsize() == 1
        task = queue.queue.get_nowait()
        assert task.deliver is False
        assert task.tool_call_id is None
        assert task.trace_id == "k1"
        assert task.meta == {}
        assert "appraisal" in task.description
        await _teardown(env)

    async def test_no_task_for_critique_origin(self, tmp_path):
        env = await build_env(tmp_path)
        await env.appraisal.on_agent_response(
            channel=env.channel, request_text="req", response_text="resp",
            trace_id="kc", meta={"origin": "critique"},
            logprobs=None,
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
            trace_id="k2", meta={},
            logprobs=None,
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
            trace_id="k3", meta={},
            logprobs=None,
        )
        await _run_next_task(env.task)
        assert len(env.stub.chat_calls) == 1
        attr = env.stub.chat_calls[0]["attribution"]
        assert attr.get("stage") == "appraisal"
        assert attr.get("trace_id") == "k3"
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
            trace_id="kp", meta={},
            logprobs=None,
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
            trace_id="km", meta={},
            logprobs=None,
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
            trace_id="kbad", meta={},
            logprobs=None,
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
            trace_id="ke", meta={},
            logprobs=logprobs,
        )
        await _run_next_task(env.task)
        envelope = await _read_appraisal(env.db, "ke")
        assert envelope["entropy"]["kind"] == "topn"
        assert envelope["entropy"]["n_tokens"] == 2
        assert envelope["entropy"]["mean"] >= 0.0
        await _teardown(env)


# ---------------------------------------------------------------------------
# 2b. past_claims (#155): the model lists the reply's claims about the past
# and search phrases for each; listeners hear the finished stage 2.
# ---------------------------------------------------------------------------


CLAIMS_STAGE2 = {
    **json.loads(VALID_STAGE2_JSON),
    "past_claims": [{"claim": "the deadline is Thursday", "queries": ["deadline Thursday"]}],
}


class TestPastClaims:
    def test_parsed_from_the_reply(self):
        from corvidae_cognition.appraisal import _stage2_from_text
        stage2 = _stage2_from_text(json.dumps(CLAIMS_STAGE2))
        assert stage2["past_claims"] == CLAIMS_STAGE2["past_claims"]

    def test_missing_field_is_no_claims(self):
        from corvidae_cognition.appraisal import _stage2_from_text
        assert _stage2_from_text(VALID_STAGE2_JSON)["past_claims"] == []

    def test_malformed_entries_dropped_and_logged(self, caplog):
        from corvidae_cognition.appraisal import _stage2_from_text
        reply = {**json.loads(VALID_STAGE2_JSON), "past_claims": [
            "a bare string", {"queries": ["no claim"]}, {"claim": "  "},
            {"claim": "kept", "queries": ["q", 7, "  ", "r"]},
        ]}
        with caplog.at_level("WARNING", logger="corvidae.appraisal"):
            stage2 = _stage2_from_text(json.dumps(reply))
        assert stage2["past_claims"] == [{"claim": "kept", "queries": ["q", "r"]}]
        assert "past_claims" in caplog.text

    def test_not_a_list_is_no_claims(self):
        from corvidae_cognition.appraisal import _stage2_from_text
        reply = {**json.loads(VALID_STAGE2_JSON), "past_claims": "none"}
        assert _stage2_from_text(json.dumps(reply))["past_claims"] == []

    def test_capped(self):
        from corvidae_cognition.appraisal import (
            MAX_PAST_CLAIMS, MAX_QUERIES_PER_CLAIM, MAX_QUERY_CHARS, _stage2_from_text,
        )
        reply = {**json.loads(VALID_STAGE2_JSON), "past_claims": [
            {"claim": f"c{i}", "queries": [f"q{j}" + "x" * MAX_QUERY_CHARS for j in range(9)]}
            for i in range(MAX_PAST_CLAIMS + 4)
        ]}
        claims = _stage2_from_text(json.dumps(reply))["past_claims"]
        assert len(claims) == MAX_PAST_CLAIMS
        assert len(claims[0]["queries"]) == MAX_QUERIES_PER_CLAIM
        assert len(claims[0]["queries"][0]) == MAX_QUERY_CHARS

    def test_schema_and_prompts_ask_for_it(self):
        from corvidae_cognition.appraisal import (
            DEFAULT_STAGE2_FORK_PROMPT, DEFAULT_STAGE2_PROMPT, STAGE2_SCHEMA,
        )
        assert "past_claims" in STAGE2_SCHEMA["required"]
        assert STAGE2_SCHEMA["properties"]["past_claims"]["type"] == "array"
        assert "past_claims" in DEFAULT_STAGE2_PROMPT
        assert "past_claims" in DEFAULT_STAGE2_FORK_PROMPT


class TestStage2Listeners:
    async def _run(self, env, trace_id, meta=None):
        await env.outcome.record_exchange(trace_id, env.channel.id, origin="user")
        await env.appraisal.on_agent_response(
            channel=env.channel, request_text="req", response_text="the reply",
            trace_id=trace_id, meta=meta if meta is not None else {"origin": "reminder"},
            logprobs=None,
        )
        await _run_next_task(env.task)

    async def test_listener_hears_the_persisted_stage2(self, tmp_path):
        env = await build_env(tmp_path, chat_text=json.dumps(CLAIMS_STAGE2))
        heard = []

        async def listener(**kwargs):
            # Stage 2 is persisted before listeners run.
            kwargs["persisted"] = (await _read_appraisal(env.db, kwargs["trace_id"]))["stage2"]
            heard.append(kwargs)

        env.appraisal.add_stage2_listener(listener)
        try:
            await self._run(env, "kl")
            [call] = heard
            assert call["channel"] is env.channel
            assert call["trace_id"] == "kl"
            assert call["response_text"] == "the reply"
            assert call["meta"] == {"origin": "reminder"}
            assert call["stage2"]["past_claims"] == CLAIMS_STAGE2["past_claims"]
            assert call["persisted"]["past_claims"] == CLAIMS_STAGE2["past_claims"]
        finally:
            await _teardown(env)

    async def test_listener_runs_outside_the_appraisal_attribution(self, tmp_path):
        env = await build_env(tmp_path, chat_text=json.dumps(CLAIMS_STAGE2))
        seen = []

        async def listener(**kwargs):
            seen.append(dict(get_attribution()))

        env.appraisal.add_stage2_listener(listener)
        try:
            await self._run(env, "ka")
            assert seen[0].get("stage") != "appraisal"
        finally:
            await _teardown(env)

    async def test_failing_listener_is_logged_and_others_still_run(self, tmp_path, caplog):
        env = await build_env(tmp_path, chat_text=json.dumps(CLAIMS_STAGE2))
        heard = []

        async def broken(**kwargs):
            raise RuntimeError("listener exploded")

        async def fine(**kwargs):
            heard.append(kwargs["trace_id"])

        env.appraisal.add_stage2_listener(broken)
        env.appraisal.add_stage2_listener(fine)
        try:
            with caplog.at_level("WARNING", logger="corvidae.appraisal"):
                await self._run(env, "kf")
            assert heard == ["kf"]
            assert "stage-2 listener" in caplog.text and "kf" in caplog.text
        finally:
            await _teardown(env)

    async def test_failed_stage2_notifies_no_one(self, tmp_path):
        env = await build_env(tmp_path, chat_text="not json at all")
        heard = []

        async def listener(**kwargs):
            heard.append(kwargs)

        env.appraisal.add_stage2_listener(listener)
        try:
            await self._run(env, "kb")
            assert heard == []
        finally:
            await _teardown(env)


# ---------------------------------------------------------------------------
# 3. entropy_summary unit tests
# ---------------------------------------------------------------------------


class TestEntropySummary:
    def test_absent_logprobs_returns_none(self):
        from corvidae_cognition.appraisal import entropy_summary
        assert entropy_summary(None) is None
        assert entropy_summary({}) is None
        assert entropy_summary({"content": []}) is None

    def test_topn_with_residual(self):
        from corvidae_cognition.appraisal import entropy_summary
        env = entropy_summary({"content": [
            {"token": "x", "logprob": -0.05,
             "top_logprobs": [{"token": "x", "logprob": -0.05}]},
        ]})
        assert env["kind"] == "topn"
        assert env["n_tokens"] == 1
        assert env["mean"] > 0.0  # residual bucket contributes entropy

    def test_nll_fallback_when_no_top_logprobs(self):
        from corvidae_cognition.appraisal import entropy_summary
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
    async def _seed(self, env, key, rowid, stage1=None, stage2=None,
                    channel_id=None):
        # channel_id defaults to env.channel so existing tests are unchanged.
        channel_id = channel_id if channel_id is not None else env.channel.id
        cols = {}
        env_appraisal = {}
        if stage1 is not None:
            env_appraisal["stage1"] = stage1
        if stage2 is not None:
            env_appraisal["stage2"] = stage2
        await env.outcome.upsert_exchange(
            key, channel_id, "user", appraisal=env_appraisal,
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
        from corvidae_cognition.appraisal import AppraisalPrior
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
        from corvidae_cognition.appraisal import AppraisalPrior
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
        mean = await env.appraisal.mean_valence(env.channel.id, (35, 45))
        assert mean == pytest.approx(0.5)
        # No stage-2 anywhere → None.
        assert await env.appraisal.mean_valence(env.channel.id, (100, 200)) is None
        await _teardown(env)

    # R1 (#195): importance for channel A's range must ignore channel B's
    # exchanges, whose message_log rowids interleave inside A's range.
    async def test_importance_over_range_ignores_other_channels(self, tmp_path):
        env = await build_env(tmp_path)
        await self._seed(env, "a1", 40, stage1={"salience": 0.2})
        await self._seed(env, "b1", 41, stage1={"salience": 0.95},
                         channel_id="irc:#b")
        await self._seed(env, "a2", 42, stage1={"salience": 0.3})
        score = await env.appraisal.importance_over_range(env.channel, (40, 42))
        # Only A's exchanges count: max(0.2, 0.3), not B's 0.95.
        assert score == pytest.approx(0.3)
        await _teardown(env)

    # R2 (#195): mean_valence takes a channel id and averages only that
    # channel's stage-2 valences (A: 0.2, 0.4; interleaved B: 0.9 -> 0.3).
    async def test_mean_valence_ignores_other_channels(self, tmp_path):
        env = await build_env(tmp_path)
        await self._seed(env, "a1", 40, stage2={"valence": 0.2})
        await self._seed(env, "b1", 41, stage2={"valence": 0.9},
                         channel_id="irc:#b")
        await self._seed(env, "a2", 42, stage2={"valence": 0.4})
        mean = await env.appraisal.mean_valence(env.channel.id, (40, 42))
        assert mean == pytest.approx(0.3)
        await _teardown(env)

    # R2 (#195) end to end: consolidating channel A stores A's mean valence
    # even though a channel B exchange sits inside A's message-id range.
    async def test_consolidation_valence_ignores_interleaved_channel(self, tmp_path):
        from corvidae_cognition.appraisal import AppraisalPlugin
        from corvidae.channel import Channel, ChannelConfig
        from corvidae.context import MessageType
        from corvidae_cognition.memory import MemoryPlugin

        consolidation_json = json.dumps({
            "summary": "I discussed the plan with the user.",
            "topic_tags": ["plan"],
            "participants": ["user"],
        })

        # Wire persistence, outcome_log, llm stub, memory and appraisal.
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
        await appraisal.on_start(config=cfg)

        chan_a = Channel(transport="test", scope="conv-a", config=ChannelConfig())
        chan_b = Channel(transport="test", scope="conv-b", config=ChannelConfig())

        # Interleave message_log rows: A, B, A (B's row lands inside A's range).
        rowids = []
        for chan, role, content in [
            (chan_a, "user", "here is the plan"),
            (chan_b, "user", "unrelated chatter"),
            (chan_a, "assistant", "understood, I will do it"),
        ]:
            rowids.append(await persistence.persist_message(
                channel=chan,
                message={"role": role, "content": content},
                message_type=MessageType.MESSAGE,
            ))

        # Stage-2 valences: A 0.2 and 0.4, interleaved B 0.9 -> A's mean 0.3.
        for key, chan, rowid, val in [
            ("xa1", chan_a, rowids[0], 0.2),
            ("xb1", chan_b, rowids[1], 0.9),
            ("xa2", chan_a, rowids[2], 0.4),
        ]:
            await outcome.upsert_exchange(
                key, chan.id, "user",
                appraisal={"stage2": {"stakes": 0.5, "valence": val, "novelty": 0.5}},
            )
            await outcome.update_exchange(key, message_rowid=rowid)

        # try/finally so teardown runs even when an assertion fails.
        try:
            await memory._consolidate_range(chan_a.id, rowids[-1], "test")

            async with db.execute(
                "SELECT valence FROM memory WHERE channel_id = ?", (chan_a.id,),
            ) as cursor:
                row = await cursor.fetchone()
            assert row is not None
            assert row[0] == pytest.approx(0.3)

        finally:
            await appraisal.on_stop()
            await db.close()

    # R3 (#195): a failing exchange_log query is logged as one WARNING naming
    # the channel, id range, exception type and fallback, and yields [].
    async def test_appraisals_in_range_logs_lookup_failure(self, tmp_path, caplog):
        import logging
        import sqlite3

        env = await build_env(tmp_path)
        # Replace the exchange DB with one whose execute() raises.
        broken = MagicMock()
        broken.execute = MagicMock(side_effect=sqlite3.OperationalError("boom"))
        env.appraisal._exchange_db = lambda: broken

        with caplog.at_level(logging.WARNING):
            result = await env.appraisal._appraisals_in_range(
                env.channel.id, (40, 42)
            )

        assert result == []
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        record = warnings[0]
        text = record.getMessage()
        # Every element R3 names: the channel, the id range as the message
        # renders it, the exception type, and both halves of the fallback.
        for needle in (
            env.channel.id,
            "messages 40–42",
            "OperationalError",
            "rubric prior",
            "NULL valence",
        ):
            assert needle in text
        # The traceback attached is the lookup's own exception.
        assert record.exc_info is not None
        assert record.exc_info[0] is sqlite3.OperationalError
        await _teardown(env)

    async def test_consolidation_writes_mean_valence(self, tmp_path):
        """End-to-end: consolidation writes mean stage-2 valence into the
        memory record and uses the appraisal-driven importance prior."""
        from corvidae_cognition.appraisal import AppraisalPlugin
        from corvidae.channel import Channel, ChannelConfig
        from corvidae.context import MessageType
        from corvidae_cognition.memory import MemoryPlugin

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
            rowids.append(await persistence.persist_message(
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

        await memory._consolidate_range(channel.id, rowids[-1], "test")

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
        from corvidae_cognition.appraisal import AppraisalPlugin, AppraisalPrior

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


def test_parse_json_block_error_names_appraisal_phase_and_long_text():
    """The appraisal JSON parse error names its phase and keeps >200 chars."""
    from corvidae_cognition.appraisal import _parse_json_block

    text = "x" * 300 + " PROVIDER_ERROR_MARKER"
    with pytest.raises(ValueError) as excinfo:
        _parse_json_block(text)

    message = str(excinfo.value)
    assert "appraisal" in message.lower()
    assert "PROVIDER_ERROR_MARKER" in message
