"""RED tests for WP2.5 — AppraisalPlugin stage 2.

Per plans/implementation/phase-2.md WP2.5:
- on_agent_response enqueues exactly one SILENT (deliver=False) stage-2 task,
  stamped with the trace id/origin; no on_notify fires from it;
- the task body sets attribution stage="appraisal" with the trace id, makes
  one tier-3 call (appraisal → background → main fallback), and MERGE-persists
  {"stage2": ..., "entropy": ...} into the appraisal envelope — never a full
  overwrite (stage1/stage1_out survive);
- malformed model output is logged and the row gets no stage 2; no exception
  escapes;
- a stage-2 task for origin="critique" exchanges too (#1031);
- entropy_summary summarizes an OpenAI-style logprobs envelope (topn + residual,
  NLL fallback, absent → None).
"""

import asyncio
import json
from types import SimpleNamespace

import aiosqlite
import pytest

from corvidae.attribution import get_attribution
from corvidae.channel import Channel, ChannelConfig
from corvidae.hooks import CorvidaePlugin, create_plugin_manager, hookimpl
from corvidae.llm_plugin import LLMPlugin
from corvidae_cognition.outcome_log import OutcomeLogPlugin
from corvidae.persistence import PersistencePlugin, init_db
from corvidae.task import TaskPlugin

from helpers import attach_store, FakeForkAgent, window_of
from tests.cognition.trace_helpers import TraceRecorder, add_trace


# A stage-2 reply as the fork now asks for it: the reply's claims about the
# past (the appraisal vector was removed, #1189).
VALID_STAGE2_JSON = json.dumps({
    "past_claims": [{"claim": "we agreed on Tuesday", "queries": ["Tuesday"]}],
})


def _make_channel(scope="stage2") -> Channel:
    return Channel(transport="test", scope=scope, config=ChannelConfig())


class StubClient:
    """Minimal LLMClient stand-in recording chat calls + attribution."""

    def __init__(self, chat_text: str = VALID_STAGE2_JSON):
        self.chat_text = chat_text
        self.chat_calls: list[dict] = []

    def request_model(self, extra_body=None):
        """The model a call is sent with, as LLMClient.request_model does."""
        return (extra_body or {}).get("model", "stub-model")

    async def chat(self, messages, tools=None, extra_body=None):
        self.chat_calls.append({
            "messages": messages,
            "extra_body": extra_body,
            "attribution": dict(get_attribution()),
        })
        return {"choices": [{"message": {"role": "assistant", "content": self.chat_text}}]}


async def build_env(tmp_path, chat_text=VALID_STAGE2_JSON, start_worker=False):
    """Wire persistence + outcome_log + llm(stub) + task + appraisal against a
    file-backed session DB. Returns a SimpleNamespace of the pieces.

    Appraisal's own hookspecs (AppraisalSpec) are added, as add_trace adds the
    trace plugin's and startup's install_plugin_hookspecs would."""
    from corvidae_cognition.appraisal import AppraisalPlugin, AppraisalSpec

    db_path = str(tmp_path / "sessions.db")
    # ``db`` seeds and reads raw; the plugins write through the route.
    db = await aiosqlite.connect(db_path, isolation_level=None)
    await init_db(db)

    pm = create_plugin_manager()
    persistence = PersistencePlugin()
    persistence.db, persistence.writer = await attach_store(db)
    pm.register(persistence, name="persistence")

    # The trace protocol plugin and a listener, before any reporter.
    recorder = TraceRecorder()
    add_trace(pm, recorder)
    outcome = OutcomeLogPlugin()
    pm.register(outcome, name="outcome_log")
    await outcome.on_init(pm=pm, config={})
    await outcome.on_start(config={})

    llm = LLMPlugin()
    pm.register(llm, name="llm")
    stub = StubClient(chat_text)
    llm._clients["main"] = stub

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

    channel = _make_channel()
    # Stage 2 is a fork of the live window (#320): a fork agent answers it
    # with ``chat_text`` and a window holds every rowid these tests persist.
    # A miss skips, so without both the stage-2 vector would never exist.
    agent = FakeForkAgent({"appraisal"}, text=chat_text)
    pm.register(agent, name="agent")
    pm.add_hookspecs(AppraisalSpec)
    channel.conversation = window_of(channel.id, list(range(1, 200)))
    return SimpleNamespace(
        pm=pm, db=db, db_path=db_path, appraisal=appraisal, task=task_plugin,
        outcome=outcome, llm=llm, stub=stub, channel=channel, agent=agent,
        recorder=recorder,
    )


async def _teardown(env):
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


async def _originate(env, trace_id, rowid=1, text="the user asked"):
    """Record the exchange's originating row, as the Agent does when it
    persists the user message. Stage 2 forks only when this row is known
    and in the window; without it the stage skips (#320)."""
    await env.appraisal.on_message_persisted(
        channel=env.channel, trace_id=trace_id, rowid=rowid, text=text, annotations={},
    )


async def _read_appraisal(recorder, key):
    """The appraisal records fired for trace ``key``, merged in order with the
    last write per top-level key winning; None when it fired none."""
    records = [r["record"] for r in recorder.by_source("appraisal") if r["trace_id"] == key]
    if not records:
        return None
    merged: dict = {}
    for record in records:
        merged.update(record)
    return merged


# ---------------------------------------------------------------------------
# 0. Originating-text LRU (moved from Agent to AppraisalPlugin, §1.4/§1.5).
# Fed by a new on_message_persisted hookimpl; read from appraisal's own LRU.
# ---------------------------------------------------------------------------


class TestOriginatingTextLRU:
    async def test_on_message_persisted_populates_the_lru(self, tmp_path):
        env = await build_env(tmp_path)
        await env.appraisal.on_message_persisted(
            channel=env.channel, trace_id="k-lru-1", rowid=5,
            text="the original user text", annotations={},
        )
        assert env.appraisal._originating_text.get("k-lru-1") == "the original user text"
        await _teardown(env)

    async def test_on_message_persisted_accepts_nullable_rowid(self, tmp_path):
        """on_message_persisted fires for originating items regardless of
        persistence outcome (rowid nullable, §1.1/§1.2) -- the LRU write
        does not depend on rowid being non-None."""
        env = await build_env(tmp_path)
        await env.appraisal.on_message_persisted(
            channel=env.channel, trace_id="k-lru-none", rowid=None,
            text="text with no rowid yet", annotations={},
        )
        assert env.appraisal._originating_text.get("k-lru-none") == "text with no rowid yet"
        await _teardown(env)

    async def test_originating_text_lru_returns_none_for_unknown_id(self, tmp_path):
        env = await build_env(tmp_path)
        assert env.appraisal._originating_text.get("never-seen") is None
        await _teardown(env)

    async def test_lru_bounded_at_512_evicts_oldest(self, tmp_path):
        from corvidae_cognition.appraisal import ORIGINATING_TEXT_LRU_MAXSIZE
        assert ORIGINATING_TEXT_LRU_MAXSIZE == 512

        env = await build_env(tmp_path)
        for i in range(ORIGINATING_TEXT_LRU_MAXSIZE + 1):
            await env.appraisal.on_message_persisted(
                channel=env.channel, trace_id=f"k-{i}", rowid=i,
                text=f"text {i}", annotations={},
            )
        # The oldest entry (k-0) was evicted once the 513th arrived; the
        # newest MAXSIZE entries survive.
        assert env.appraisal._originating_text.get("k-0") is None
        assert env.appraisal._originating_text.get("k-1") == "text 1"
        assert env.appraisal._originating_text.get(
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
            text="the original user text", annotations={},
        )
        await env.appraisal.on_agent_response(
            channel=env.channel, request_text="tool result: done",
            response_text="final response", trace_id="k-cycle",
            annotations={}, logprobs=None,
        )
        await _run_next_task(env.task)

        # Stage 2 is a fork of the window (#320); the instruction quotes the
        # recovered originating text, not the last tool-result turn.
        assert env.stub.chat_calls == []
        assert len(env.agent.calls) == 1
        instruction = env.agent.calls[0]["instruction"]
        assert "«the original user text»" in instruction
        assert "tool result: done" not in instruction
        await _teardown(env)


# ---------------------------------------------------------------------------
# 1. Trigger: exactly one silent task, stamped, attributed
# ---------------------------------------------------------------------------


class TestTrigger:
    async def test_one_silent_task_enqueued_and_stamped(self, tmp_path):
        env = await build_env(tmp_path)
        await env.appraisal.on_agent_response(
            channel=env.channel, request_text="req", response_text="resp",
            trace_id="k1", annotations={},
            logprobs=None,
        )
        queue = env.task.task_queue
        assert queue.queue.qsize() == 1
        task = queue.queue.get_nowait()
        assert task.deliver is False
        assert task.tool_call_id is None
        assert task.trace_id == "k1"
        assert task.annotations == {}
        assert "appraisal" in task.description
        await _teardown(env)

    async def test_stage2_runs_for_critique_origin(self, tmp_path):
        """#1031: a correction is judged, so provenance needs its claims."""
        env = await build_env(tmp_path)
        await env.appraisal.on_agent_response(
            channel=env.channel, request_text="req", response_text="resp",
            trace_id="kc", annotations={"origin": "critique"},
            logprobs=None,
        )
        assert env.task.task_queue.queue.qsize() == 1
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
            async def on_notify(self, channel, source, text, tool_call_id, annotations):
                notified.append(source)

        env.pm.register(NotifySpy(), name="notify_spy")

        await env.appraisal.on_agent_response(
            channel=env.channel, request_text="req", response_text="resp",
            trace_id="k2", annotations={},
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
        await _originate(env, "k3")
        await env.appraisal.on_agent_response(
            channel=env.channel, request_text="req", response_text="resp",
            trace_id="k3", annotations={},
            logprobs=None,
        )
        await _run_next_task(env.task)
        assert env.stub.chat_calls == []
        assert len(env.agent.calls) == 1
        attr = env.agent.calls[0]["attribution"]
        assert attr.get("stage") == "appraisal"
        assert attr.get("trace_id") == "k3"
        await _teardown(env)


# ---------------------------------------------------------------------------
# 2. Persist: stage-2 JSON under the key; merge-not-overwrite; malformed safe
# ---------------------------------------------------------------------------


class TestStage2Persist:
    async def test_stage2_persisted_under_key(self, tmp_path):
        env = await build_env(tmp_path)
        await _originate(env, "kp")
        await env.appraisal.on_agent_response(
            channel=env.channel, request_text="req", response_text="resp",
            trace_id="kp", annotations={},
            logprobs=None,
        )
        await _run_next_task(env.task)
        env_appraisal = await _read_appraisal(env.recorder, "kp")
        assert env_appraisal["stage2"] == json.loads(VALID_STAGE2_JSON)
        await _teardown(env)

    async def test_the_fork_writes_a_fork_log_row(self, tmp_path):
        """#1204: the call site, not just record(), puts a row in the table."""
        import aiosqlite as _aio

        env = await build_env(tmp_path)
        await _originate(env, "kl")
        await env.appraisal.on_agent_response(
            channel=env.channel, request_text="req", response_text="resp",
            trace_id="kl", annotations={},
            logprobs=None,
        )
        await _run_next_task(env.task)
        db = await _aio.connect(env.db_path)
        try:
            async with db.execute(
                "SELECT stage, channel_id, prompt_sha256, instruction,"
                " system_sha256, reconstructable FROM fork_log"
            ) as cur:
                rows = await cur.fetchall()
            assert len(rows) == 1
            _stage, _channel, sha, instruction, system_sha, restorable = rows[0]
            assert sha, "the row names the prompt it rendered"
            assert instruction, "the instruction is the question side"
            assert system_sha, "the system prompt is stored once per version"
            assert restorable == 1
            # the window's identity is stored, and the system prompt with it
            async with db.execute("SELECT count(*) FROM fork_block") as cur:
                assert (await cur.fetchone())[0] >= 1
        finally:
            await db.close()
        await _teardown(env)

    async def test_malformed_output_records_no_stage2_no_exception(self, tmp_path):
        env = await build_env(tmp_path, chat_text="not json at all")
        await _originate(env, "kbad")
        await env.appraisal.on_agent_response(
            channel=env.channel, request_text="req", response_text="resp",
            trace_id="kbad", annotations={},
            logprobs=None,
        )
        # Must not raise.
        await _run_next_task(env.task)
        envelope = await _read_appraisal(env.recorder, "kbad")
        assert "stage2" not in (envelope or {})  # stage2 absent
        await _teardown(env)

    async def test_entropy_persisted_when_logprobs_present(self, tmp_path):
        env = await build_env(tmp_path)
        await _originate(env, "ke")
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
            trace_id="ke", annotations={},
            logprobs=logprobs,
        )
        await _run_next_task(env.task)
        envelope = await _read_appraisal(env.recorder, "ke")
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
        assert _stage2_from_text(json.dumps({}))["past_claims"] == []

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


class _IdlePeer(CorvidaePlugin):
    """A peer that implements on_appraisal_idle and hands every call, as
    keyword arguments, to ``callback`` (the hook that replaced the stage-2
    listener, #1169)."""

    depends_on = frozenset()

    def __init__(self, callback):
        self.callback = callback

    @hookimpl
    async def on_appraisal_idle(
        self, channel, trace_id, response_text, annotations, lenses, past_claims,
    ):
        await self.callback(
            channel=channel, trace_id=trace_id, response_text=response_text,
            annotations=annotations, lenses=lenses, past_claims=past_claims)


def _listen(env, callback, name="peer"):
    """Register an idle peer beside appraisal (build_env installed the
    hookspecs; pluggy checks a later hookimpl against them)."""
    env.pm.register(_IdlePeer(callback), name=name)


class TestStage2Listeners:
    async def _run(self, env, trace_id, annotations=None):
        await _originate(env, trace_id)
        await env.appraisal.on_agent_response(
            channel=env.channel, request_text="req", response_text="the reply",
            trace_id=trace_id, annotations=annotations if annotations is not None else {"origin": "reminder"},
            logprobs=None,
        )
        await _run_next_task(env.task)

    async def test_listener_hears_the_persisted_stage2(self, tmp_path):
        env = await build_env(tmp_path, chat_text=json.dumps(CLAIMS_STAGE2))
        heard = []

        async def listener(**kwargs):
            # Stage 2 is reported before peers are told.
            kwargs["persisted"] = (await _read_appraisal(env.recorder, kwargs["trace_id"]))["stage2"]
            heard.append(kwargs)

        _listen(env, listener)
        try:
            await self._run(env, "kl")
            [call] = heard
            assert call["channel"] is env.channel
            assert call["trace_id"] == "kl"
            assert call["response_text"] == "the reply"
            assert call["annotations"] == {"origin": "reminder"}
            assert call["past_claims"] == CLAIMS_STAGE2["past_claims"]
            assert call["persisted"]["past_claims"] == CLAIMS_STAGE2["past_claims"]
        finally:
            await _teardown(env)

    async def test_listener_runs_outside_the_appraisal_attribution(self, tmp_path):
        env = await build_env(tmp_path, chat_text=json.dumps(CLAIMS_STAGE2))
        seen = []

        async def listener(**kwargs):
            seen.append(dict(get_attribution()))

        _listen(env, listener)
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

        _listen(env, broken, name="broken")
        _listen(env, fine, name="fine")
        try:
            with caplog.at_level("WARNING", logger="corvidae"):
                await self._run(env, "kf")
            assert heard == ["kf"]
            # The hook dispatch logs the failure, naming the hook and the plugin.
            assert "on_appraisal_idle" in caplog.text and "broken" in caplog.text
        finally:
            await _teardown(env)

    async def test_failed_stage2_notifies_once_with_none(self, tmp_path):
        env = await build_env(tmp_path, chat_text="not json at all")
        heard = []

        async def listener(**kwargs):
            heard.append(kwargs)

        _listen(env, listener)
        try:
            await self._run(env, "kb")
            # A miss notifies exactly once, with no lenses and no claims (#1033 R5).
            assert len(heard) == 1
            assert heard[0]["lenses"] is None and heard[0]["past_claims"] is None
        finally:
            await _teardown(env)


    async def test_unexpected_exception_notifies_once_with_none(
        self, tmp_path, monkeypatch, caplog,
    ):
        """#1033 R5: a stage 2 that raises is a miss: logged at warning with
        the trace, and listeners hear stage2=None once."""
        env = await build_env(tmp_path, chat_text=json.dumps(CLAIMS_STAGE2))
        heard = []

        async def listener(**kwargs):
            heard.append(kwargs)

        async def explode(*args, **kwargs):
            raise RuntimeError("stage 2 exploded")

        _listen(env, listener)
        monkeypatch.setattr(env.appraisal, "_run_stage2", explode)
        try:
            with caplog.at_level("WARNING", logger="corvidae.appraisal"):
                await self._run(env, "kx")
            assert [(h["lenses"], h["past_claims"]) for h in heard] == [(None, None)]
            assert "kx" in caplog.text
        finally:
            await _teardown(env)

    async def test_cancelled_stage2_notifies_no_one(self, tmp_path, monkeypatch):
        """#1033: cancellation (shutdown) is not a miss; no listener hears it."""
        env = await build_env(tmp_path, chat_text=json.dumps(CLAIMS_STAGE2))
        heard = []

        async def listener(**kwargs):
            heard.append(kwargs)

        async def cancelled(*args, **kwargs):
            raise asyncio.CancelledError()

        _listen(env, listener)
        monkeypatch.setattr(env.appraisal, "_run_stage2", cancelled)
        try:
            with pytest.raises(asyncio.CancelledError):
                await self._run(env, "kc")
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


def test_parse_json_block_error_names_appraisal_phase_and_long_text():
    """The appraisal JSON parse error names its phase and keeps >200 chars."""
    from corvidae_cognition.appraisal import _parse_json_block

    text = "x" * 300 + " PROVIDER_ERROR_MARKER"
    with pytest.raises(ValueError) as excinfo:
        _parse_json_block(text)

    message = str(excinfo.value)
    assert "appraisal" in message.lower()
    assert "PROVIDER_ERROR_MARKER" in message
