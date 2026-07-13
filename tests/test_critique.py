"""RED tests for WP2.7 — CritiquePlugin.

Per plans/implementation/phase-2.md WP2.7:
- provenance snapshot of CONTEXT messages under the exchange key
  (before_agent_turn);
- eligibility by PROPAGATED origin (trap #3): user/reminder eligible;
  critique/heartbeat/task exempt (the recursion brake); a verdict-triggered
  tool-using turn ends exempt; a user exchange ending on a tool-result turn is
  critiqued against the original user text;
- lens selection from the appraisal vector (predictive/constrained/adversarial)
  with below-threshold sampling via an injectable RNG, marked in the outcome;
- the mechanical two-tier provenance gate (past-claim detector ∧ weak retrieval
  ∧ empty message_fts) firing independently of appraisal scores;
- silent execution: empty verdict → no on_notify, outcome recorded; non-empty →
  funnel registration + one stub, verdict as framed CONTEXT on the next turn;
- runtime-tunable thresholds via channel.runtime_overrides (directive 2).

Detector pattern spec (WP2.7 point 4): the past-claim detector fires on the
first-person-recall / past-assertion patterns in corvidae.critique.
PAST_CLAIM_PATTERNS; the message_fts key-term extraction caps at
critique.provenance.max_terms word tokens.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiosqlite
import pytest

from corvidae.attribution import get_attribution
from corvidae.channel import Channel, ChannelConfig
from corvidae.context import ContextWindow, MessageType
from corvidae.hooks import CorvidaePlugin, create_plugin_manager, hookimpl
from corvidae.outcome_log import OutcomeLogPlugin
from corvidae.persistence import PersistencePlugin, init_db
from corvidae.task import TaskPlugin, TaskQueue


EMPTY_VERDICT = json.dumps({"objections": []})
OBJECTION_VERDICT = json.dumps({"objections": [
    {"claim": "the deadline is Friday", "objection": "no record says Friday",
     "severity": 0.7},
]})


class StubClient:
    def __init__(self, chat_text=OBJECTION_VERDICT):
        self.chat_text = chat_text
        self.chat_calls: list[dict] = []

    async def chat(self, messages, tools=None, extra_body=None):
        self.chat_calls.append({
            "messages": messages,
            "attribution": dict(get_attribution()),
        })
        return {"choices": [{"message": {"role": "assistant", "content": self.chat_text}}]}


class FakeAppraisal(CorvidaePlugin):
    """Controllable appraisal store for lens-selection tests."""

    depends_on = frozenset()

    def __init__(self, stage1=None, stage2=None):
        self._stage1 = stage1
        self._stage2 = stage2

    async def get_appraisal(self, exchange_key):
        return self._stage1

    def get_last_stage2(self, channel_id):
        return self._stage2


class FixedRng:
    def __init__(self, value=0.99, choice_value="predictive"):
        self.value = value
        self.choice_value = choice_value

    def random(self):
        return self.value

    def choice(self, seq):
        return self.choice_value if self.choice_value in seq else seq[0]


def _make_channel(scope="critique") -> Channel:
    ch = Channel(transport="test", scope=scope, config=ChannelConfig())
    ch.conversation = ContextWindow(ch.id)
    return ch


async def _seed_fts(path, contents):
    db = await aiosqlite.connect(path)
    await db.execute("CREATE VIRTUAL TABLE message_fts USING fts5(content_text)")
    for c in contents:
        await db.execute("INSERT INTO message_fts(content_text) VALUES (?)", (c,))
    await db.commit()
    await db.close()


async def build_critique(
    tmp_path, appraisal=None, chat_text=OBJECTION_VERDICT, fts_contents=(),
    with_funnel=False, config=None,
):
    from corvidae.critique import CritiquePlugin

    db_path = str(tmp_path / "sessions.db")
    await _seed_fts(db_path, fts_contents)  # creates message_fts

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

    from corvidae.llm_plugin import LLMPlugin
    llm = LLMPlugin()
    pm.register(llm, name="llm")
    stub = StubClient(chat_text)
    llm._clients["main"] = stub

    task_plugin = TaskPlugin()
    pm.register(task_plugin, name="task")
    await task_plugin.on_init(pm=pm, config={})
    task_plugin.task_queue = TaskQueue(max_workers=1)

    funnel = None
    if with_funnel:
        from corvidae.funnel import FunnelPlugin
        funnel = FunnelPlugin()
        pm.register(funnel, name="funnel")
        await funnel.on_init(pm=pm, config={})

    if appraisal is not None:
        pm.register(appraisal, name="appraisal")

    plugin = CritiquePlugin()
    pm.register(plugin, name="critique")
    cfg = config or {}
    cfg.setdefault("daemon", {})["session_db"] = db_path
    await plugin.on_init(pm=pm, config=cfg)
    await plugin.on_start(config=cfg)

    channel = _make_channel()
    return SimpleNamespace(
        pm=pm, db=db, db_path=db_path, plugin=plugin, task=task_plugin,
        outcome=outcome, llm=llm, stub=stub, funnel=funnel, channel=channel,
    )


async def _teardown(env):
    await env.plugin.on_stop()
    await env.db.close()


async def _run_next_task(task_plugin):
    task = task_plugin.task_queue.queue.get_nowait()
    await asyncio.create_task(task.work(), context=task.ctx)
    return task


async def _read_outcomes(db, key):
    async with db.execute(
        "SELECT outcomes FROM exchange_log WHERE exchange_key = ?", (key,)
    ) as cursor:
        row = await cursor.fetchone()
    return json.loads(row[0]) if row and row[0] else None


async def _fire(env, exchange_key="k", origin="user", response_text="resp",
                originating_text="orig user text"):
    await env.outcome.record_exchange(exchange_key, env.channel.id, origin=origin)
    await env.plugin.on_agent_response(
        channel=env.channel, request_text="req", response_text=response_text,
        exchange_key=exchange_key, origin=origin,
        originating_text=originating_text, logprobs=None, withheld=False,
    )


# ---------------------------------------------------------------------------
# 0. Pure detector units (part of the red-test spec)
# ---------------------------------------------------------------------------


class TestPastClaimDetector:
    def test_fires_on_recall_patterns(self):
        from corvidae.critique import is_past_claim
        assert is_past_claim("You told me the deadline was Friday.")
        assert is_past_claim("I remember we discussed the budget.")
        assert is_past_claim("As I mentioned earlier, the plan changed.")

    def test_no_fire_on_plain_statement(self):
        from corvidae.critique import is_past_claim
        assert not is_past_claim("The weather is nice today.")
        assert not is_past_claim("")

    def test_extract_terms_caps_and_dedupes(self):
        from corvidae.critique import _extract_terms
        terms = _extract_terms("alpha beta alpha gamma delta epsilon zeta eta theta", 3)
        assert terms == ["alpha", "beta", "gamma"]


# ---------------------------------------------------------------------------
# 1. Provenance snapshot (before_agent_turn)
# ---------------------------------------------------------------------------


class TestProvenanceSnapshot:
    async def test_snapshots_context_messages(self, tmp_path):
        env = await build_critique(tmp_path)
        await env.outcome.record_exchange("ks", env.channel.id, origin="user")
        env.channel.conversation.append(
            {"role": "system", "content": "<<memory>> the plan is X <</memory>>"},
            MessageType.CONTEXT,
        )
        env.channel.conversation.append(
            {"role": "user", "content": "what is the plan?"}, MessageType.MESSAGE,
        )
        await env.plugin.before_agent_turn(env.channel, "ks", "user")
        async with env.db.execute(
            "SELECT provenance_snapshot FROM exchange_log WHERE exchange_key = ?",
            ("ks",),
        ) as cursor:
            row = await cursor.fetchone()
        snap = json.loads(row[0])
        assert len(snap) == 1
        assert "the plan is X" in snap[0]["content"]
        await _teardown(env)

    async def test_no_context_no_snapshot(self, tmp_path):
        env = await build_critique(tmp_path)
        await env.outcome.record_exchange("kn", env.channel.id, origin="user")
        env.channel.conversation.append(
            {"role": "user", "content": "hi"}, MessageType.MESSAGE,
        )
        await env.plugin.before_agent_turn(env.channel, "kn", "user")
        async with env.db.execute(
            "SELECT provenance_snapshot FROM exchange_log WHERE exchange_key = ?",
            ("kn",),
        ) as cursor:
            row = await cursor.fetchone()
        assert row[0] is None
        await _teardown(env)


# ---------------------------------------------------------------------------
# 2. Eligibility by origin (trap #3)
# ---------------------------------------------------------------------------


class TestEligibility:
    async def test_user_origin_eligible(self, tmp_path):
        appraisal = FakeAppraisal(stage2={"ambiguity": 0.9})
        env = await build_critique(tmp_path, appraisal=appraisal)
        await _fire(env, "ku", origin="user")
        assert env.task.task_queue.queue.qsize() == 1
        await _teardown(env)

    async def test_reminder_origin_eligible(self, tmp_path):
        appraisal = FakeAppraisal(stage2={"ambiguity": 0.9})
        env = await build_critique(tmp_path, appraisal=appraisal)
        await _fire(env, "kr", origin="reminder")
        assert env.task.task_queue.queue.qsize() == 1
        await _teardown(env)

    @pytest.mark.parametrize("origin", ["critique", "heartbeat", "task"])
    async def test_exempt_origins_produce_no_task(self, tmp_path, origin):
        appraisal = FakeAppraisal(stage2={"ambiguity": 0.9})
        env = await build_critique(tmp_path, appraisal=appraisal)
        await _fire(env, "ke", origin=origin)
        assert env.task.task_queue.queue.qsize() == 0
        await _teardown(env)

    async def test_user_exchange_critiqued_against_originating_text(self, tmp_path):
        """A user exchange ending on a tool-result turn is critiqued against
        the ORIGINAL user text (originating_text), not request_text."""
        appraisal = FakeAppraisal(stage2={"ambiguity": 0.9})
        env = await build_critique(tmp_path, appraisal=appraisal)
        await _fire(env, "kt", origin="user",
                    originating_text="the ORIGINAL user question")
        await _run_next_task(env.task)
        user_msg = env.stub.chat_calls[0]["messages"][1]["content"]
        assert "the ORIGINAL user question" in user_msg
        await _teardown(env)


# ---------------------------------------------------------------------------
# 3. Lens selection + below-threshold sampling
# ---------------------------------------------------------------------------


class TestLensSelection:
    async def test_high_ambiguity_selects_predictive(self, tmp_path):
        appraisal = FakeAppraisal(stage2={"ambiguity": 0.9})
        env = await build_critique(tmp_path, appraisal=appraisal,
                                   chat_text=EMPTY_VERDICT)
        await _fire(env, "kp", origin="user")
        await _run_next_task(env.task)
        outcomes = await _read_outcomes(env.db, "kp")
        assert "predictive" in outcomes["critique"]["lenses"]
        await _teardown(env)

    async def test_high_commitment_selects_constrained(self, tmp_path):
        appraisal = FakeAppraisal(stage1={"commitment_density": 0.9})
        env = await build_critique(tmp_path, appraisal=appraisal,
                                   chat_text=EMPTY_VERDICT)
        await _fire(env, "kc", origin="user")
        await _run_next_task(env.task)
        outcomes = await _read_outcomes(env.db, "kc")
        assert "constrained" in outcomes["critique"]["lenses"]
        await _teardown(env)

    async def test_neg_valence_and_disagreement_selects_adversarial(self, tmp_path):
        appraisal = FakeAppraisal(
            stage1={"disagreement": 0.9}, stage2={"valence": 0.1},
        )
        env = await build_critique(tmp_path, appraisal=appraisal,
                                   chat_text=EMPTY_VERDICT)
        await _fire(env, "kadv", origin="user")
        await _run_next_task(env.task)
        outcomes = await _read_outcomes(env.db, "kadv")
        assert "adversarial" in outcomes["critique"]["lenses"]
        await _teardown(env)

    async def test_below_threshold_no_task_when_not_sampled(self, tmp_path):
        appraisal = FakeAppraisal(
            stage1={"commitment_density": 0.0, "disagreement": 0.0},
            stage2={"ambiguity": 0.0, "valence": 0.5},
        )
        env = await build_critique(tmp_path, appraisal=appraisal)
        env.plugin._rng = FixedRng(value=0.99)  # never below the 0.05 rate
        await _fire(env, "kb", origin="user", response_text="no claims here")
        assert env.task.task_queue.queue.qsize() == 0
        await _teardown(env)

    async def test_below_threshold_sampled_marks_outcome(self, tmp_path):
        appraisal = FakeAppraisal(
            stage1={"commitment_density": 0.0, "disagreement": 0.0},
            stage2={"ambiguity": 0.0, "valence": 0.5},
        )
        env = await build_critique(tmp_path, appraisal=appraisal,
                                   chat_text=EMPTY_VERDICT)
        env.plugin._rng = FixedRng(value=0.01, choice_value="predictive")
        await _fire(env, "ksamp", origin="user", response_text="no claims here")
        assert env.task.task_queue.queue.qsize() == 1
        await _run_next_task(env.task)
        outcomes = await _read_outcomes(env.db, "ksamp")
        assert outcomes["critique"]["sampled_below_threshold"] is True
        await _teardown(env)

    async def test_no_appraisal_plugin_critiques_everything(self, tmp_path):
        env = await build_critique(tmp_path, appraisal=None,
                                   chat_text=EMPTY_VERDICT)
        await _fire(env, "kna", origin="user", response_text="anything")
        assert env.task.task_queue.queue.qsize() == 1
        await _teardown(env)

    async def test_threshold_override_takes_effect_without_reinit(self, tmp_path):
        """Raising critique.lens.ambiguity via runtime_overrides suppresses the
        lens on the next exchange with no re-init (directive 2)."""
        appraisal = FakeAppraisal(stage2={"ambiguity": 0.7})
        env = await build_critique(tmp_path, appraisal=appraisal,
                                   chat_text=EMPTY_VERDICT)
        # Default ambiguity threshold 0.6 → predictive fires.
        await _fire(env, "kd1", origin="user")
        assert env.task.task_queue.queue.qsize() == 1
        await _run_next_task(env.task)
        # Raise the bar above 0.7 → no stylistic lens; also disable sampling.
        env.channel.runtime_overrides["critique.lens.ambiguity"] = 0.95
        env.channel.runtime_overrides["critique.sample_below_rate"] = 0.0
        await _fire(env, "kd2", origin="user", response_text="no claim")
        assert env.task.task_queue.queue.qsize() == 0
        await _teardown(env)


# ---------------------------------------------------------------------------
# 4. Provenance gate (mechanical, two-tier)
# ---------------------------------------------------------------------------


async def _low_appraisal_env(tmp_path, fts_contents=(), chat_text=OBJECTION_VERDICT):
    appraisal = FakeAppraisal(
        stage1={"commitment_density": 0.0, "disagreement": 0.0},
        stage2={"ambiguity": 0.0, "valence": 0.5},
    )
    env = await build_critique(
        tmp_path, appraisal=appraisal, chat_text=chat_text,
        fts_contents=fts_contents,
        config={"critique": {"sample_below_rate": 0.0}},
    )
    return env


class TestProvenanceGate:
    async def test_past_claim_weak_retrieval_empty_fts_fires(self, tmp_path):
        env = await _low_appraisal_env(tmp_path, chat_text=EMPTY_VERDICT)
        await env.outcome.record_exchange("kpg", env.channel.id, origin="user")
        await env.outcome.update_exchange(
            "kpg", retrieval_top_score=0.1, retrieval_hit_count=1,
        )
        await env.plugin.on_agent_response(
            channel=env.channel, request_text="req",
            response_text="You told me the launch date was March.",
            exchange_key="kpg", origin="user",
            originating_text="when is launch?", logprobs=None, withheld=False,
        )
        assert env.task.task_queue.queue.qsize() == 1
        await _run_next_task(env.task)
        outcomes = await _read_outcomes(env.db, "kpg")
        assert "provenance" in outcomes["critique"]["lenses"]
        await _teardown(env)

    async def test_strong_retrieval_does_not_fire(self, tmp_path):
        env = await _low_appraisal_env(tmp_path)
        await env.outcome.record_exchange("kstrong", env.channel.id, origin="user")
        await env.outcome.update_exchange(
            "kstrong", retrieval_top_score=0.9, retrieval_hit_count=5,
        )
        await env.plugin.on_agent_response(
            channel=env.channel, request_text="req",
            response_text="You told me the launch date was March.",
            exchange_key="kstrong", origin="user",
            originating_text="when is launch?", logprobs=None, withheld=False,
        )
        assert env.task.task_queue.queue.qsize() == 0
        await _teardown(env)

    async def test_fts_hit_does_not_fire(self, tmp_path):
        # message_fts contains the response's key terms → tier 2 not empty.
        env = await _low_appraisal_env(
            tmp_path,
            fts_contents=["You told me the launch date was March next year"],
        )
        await env.outcome.record_exchange("kfts", env.channel.id, origin="user")
        await env.outcome.update_exchange(
            "kfts", retrieval_top_score=0.1, retrieval_hit_count=0,
        )
        await env.plugin.on_agent_response(
            channel=env.channel, request_text="req",
            response_text="You told me the launch date was March.",
            exchange_key="kfts", origin="user",
            originating_text="when is launch?", logprobs=None, withheld=False,
        )
        assert env.task.task_queue.queue.qsize() == 0
        await _teardown(env)

    async def test_no_past_claim_does_not_fire(self, tmp_path):
        env = await _low_appraisal_env(tmp_path)
        await env.outcome.record_exchange("knp", env.channel.id, origin="user")
        await env.outcome.update_exchange(
            "knp", retrieval_top_score=0.1, retrieval_hit_count=0,
        )
        await env.plugin.on_agent_response(
            channel=env.channel, request_text="req",
            response_text="The launch will be great.",
            exchange_key="knp", origin="user",
            originating_text="thoughts?", logprobs=None, withheld=False,
        )
        assert env.task.task_queue.queue.qsize() == 0
        await _teardown(env)

    async def test_provenance_disabled_does_not_fire(self, tmp_path):
        env = await _low_appraisal_env(tmp_path)
        env.channel.runtime_overrides["critique.provenance.enabled"] = False
        await env.outcome.record_exchange("kdis", env.channel.id, origin="user")
        await env.outcome.update_exchange(
            "kdis", retrieval_top_score=0.1, retrieval_hit_count=0,
        )
        await env.plugin.on_agent_response(
            channel=env.channel, request_text="req",
            response_text="You told me the answer was 42.",
            exchange_key="kdis", origin="user",
            originating_text="q?", logprobs=None, withheld=False,
        )
        assert env.task.task_queue.queue.qsize() == 0
        await _teardown(env)


# ---------------------------------------------------------------------------
# 5. Silent execution: empty vs non-empty verdict
# ---------------------------------------------------------------------------


class TestExecution:
    async def test_empty_verdict_no_notify_outcome_recorded(self, tmp_path):
        appraisal = FakeAppraisal(stage2={"ambiguity": 0.9})
        env = await build_critique(tmp_path, appraisal=appraisal,
                                   chat_text=EMPTY_VERDICT, with_funnel=True)
        env.pm.ahook.on_notify = AsyncMock()
        await _fire(env, "kempty", origin="user")
        await _run_next_task(env.task)
        env.pm.ahook.on_notify.assert_not_awaited()
        outcomes = await _read_outcomes(env.db, "kempty")
        assert outcomes["critique"]["objections"] == 0
        # The critic call carried the exchange attribution.
        attr = env.stub.chat_calls[0]["attribution"]
        assert attr.get("stage") == "critique"
        assert attr.get("exchange_key") == "kempty"
        await _teardown(env)

    async def test_nonempty_verdict_registers_and_delivers_context(self, tmp_path):
        appraisal = FakeAppraisal(stage2={"ambiguity": 0.9})
        env = await build_critique(tmp_path, appraisal=appraisal,
                                   chat_text=OBJECTION_VERDICT, with_funnel=True)
        notify = AsyncMock()
        env.pm.ahook.on_notify = notify
        await _fire(env, "kfull", origin="user")
        await _run_next_task(env.task)
        outcomes = await _read_outcomes(env.db, "kfull")
        assert outcomes["critique"]["objections"] == 1
        # One stub fired for the critique origin.
        notify.assert_awaited_once()
        assert notify.await_args.kwargs["meta"] == {"origin": "critique"}
        # Drain on a critique-origin turn → verdict appears as framed CONTEXT.
        await env.funnel.before_agent_turn(
            env.channel, exchange_key="kfull", origin="critique",
        )
        context_text = "\n".join(
            m.get("content") or ""
            for m in env.channel.conversation.messages
            if m.get("_message_type") == MessageType.CONTEXT
        )
        assert "no record says Friday" in context_text
        await _teardown(env)
