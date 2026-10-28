"""RED tests for WP2.7 — CritiquePlugin.

Per plans/implementation/phase-2.md WP2.7:
- provenance snapshot of CONTEXT messages under the trace id
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
from corvidae_cognition.outcome_log import OutcomeLogPlugin
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
        # trace_id -> originating text, standing in for the LRU
        # AppraisalPlugin owns (§1.4); CritiquePlugin reads it via
        # get_originating_text with a request_text fallback.
        self._originating_text: dict[str, str] = {}

    async def get_appraisal(self, trace_id):
        return self._stage1

    def get_last_stage2(self, channel_id):
        return self._stage2

    def get_originating_text(self, trace_id):
        return self._originating_text.get(trace_id)


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
    from corvidae_cognition.critique import CritiquePlugin

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
        from corvidae_cognition.funnel import FunnelPlugin
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
        appraisal=appraisal,
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
        "SELECT outcomes FROM exchange_log WHERE trace_id = ?", (key,)
    ) as cursor:
        row = await cursor.fetchone()
    return json.loads(row[0]) if row and row[0] else None


async def _fire(env, trace_id="k", origin="user", response_text="resp",
                originating_text="orig user text"):
    """Simulates the propagated-origin mechanics (§1.4/§1.5): "user" makes
    the trace_id admitted via on_message_admitted (meta carries no
    origin string -- eligibility is inferred from admitted-LRU membership);
    "reminder" stamps meta["origin"]; "critique"/"heartbeat" stamp their own
    exempt origin into meta; "task" is neither admitted nor stamped
    (background/task-origin default, per §1.4)."""
    await env.outcome.record_exchange(trace_id, env.channel.id, origin=origin)
    if env.appraisal is not None:
        env.appraisal._originating_text[trace_id] = originating_text
    if origin == "user":
        await env.plugin.on_message_admitted(
            channel=env.channel, trace_id=trace_id,
            sender="user", text=originating_text,
        )
        meta = {}
    elif origin == "task":
        meta = {}
    else:
        meta = {"origin": origin}
    await env.plugin.on_agent_response(
        channel=env.channel, request_text="req", response_text=response_text,
        trace_id=trace_id, meta=meta, logprobs=None,
    )


# ---------------------------------------------------------------------------
# Session-db base-dir resolution (config consistency regression)
# ---------------------------------------------------------------------------


class TestSessionDbBaseDirResolution:
    async def test_relative_session_db_resolves_against_base_dir(
        self, tmp_path, monkeypatch
    ):
        """A relative daemon.session_db resolves against config['_base_dir'],
        not the process cwd -- the same mechanism MetricsJsonlPlugin already
        uses (persistence.py resolves session_db the same way).

        Regression test: previously the raw relative path was passed
        straight to aiosqlite.connect(), so the message_fts probe pointed
        at a nonexistent file and silently degraded to no-probe.
        """
        from corvidae_cognition.critique import CritiquePlugin

        config_dir = tmp_path / "config_dir"
        config_dir.mkdir()
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()

        db_path = str(config_dir / "sessions.db")
        await _seed_fts(db_path, ["some raw dialog content"])

        monkeypatch.chdir(elsewhere)

        pm = create_plugin_manager()
        plugin = CritiquePlugin()
        pm.register(plugin, name="critique")
        cfg = {"_base_dir": config_dir, "daemon": {"session_db": "sessions.db"}}
        await plugin.on_init(pm=pm, config=cfg)
        await plugin.on_start(config=cfg)

        try:
            assert plugin._probe_db is not None, (
                "message_fts probe failed to open -- the relative "
                "session_db path did not resolve against _base_dir"
            )
        finally:
            await plugin.on_stop()


# ---------------------------------------------------------------------------
# 0. Pure detector units (part of the red-test spec)
# ---------------------------------------------------------------------------


class TestPastClaimDetector:
    def test_fires_on_recall_patterns(self):
        from corvidae_cognition.critique import is_past_claim
        assert is_past_claim("You told me the deadline was Friday.")
        assert is_past_claim("I remember we discussed the budget.")
        assert is_past_claim("As I mentioned earlier, the plan changed.")

    def test_no_fire_on_plain_statement(self):
        from corvidae_cognition.critique import is_past_claim
        assert not is_past_claim("The weather is nice today.")
        assert not is_past_claim("")

    def test_extract_terms_caps_and_dedupes(self):
        from corvidae_cognition.critique import _extract_terms
        terms = _extract_terms("alpha beta alpha gamma delta epsilon zeta eta theta", 3)
        assert terms == ["alpha", "beta", "gamma"]

    def test_extract_terms_drops_stopwords_and_short_tokens(self):
        # #155 cause 2: "I" and "the" match nearly every row, so they are
        # no evidence of a record.
        from corvidae_cognition.critique import _extract_terms
        assert _extract_terms("I like the weather today", 8) == ["weather", "today"]

    def test_extract_terms_keeps_short_tokens_with_digits(self):
        from corvidae_cognition.critique import _extract_terms
        assert _extract_terms("The meeting is at 3 in room 4b", 8) == [
            "meeting", "3", "room", "4b",
        ]

    def test_extract_terms_prefers_the_claim_sentence(self):
        # Terms come from the sentence is_past_claim matched, minus the
        # recall phrase and its framing words, not from the whole reply.
        from corvidae_cognition.critique import _extract_terms
        text = (
            "Sure thing. As I mentioned earlier, the quarterly audit deadline "
            "is Thursday. Let me know if anything changes."
        )
        assert _extract_terms(text, 8) == ["quarterly", "audit", "deadline", "Thursday"]

    def test_extract_terms_claim_before_the_recall_phrase(self):
        from corvidae_cognition.critique import _extract_terms
        assert _extract_terms("The deadline is Thursday, as I mentioned.", 8) == [
            "deadline", "Thursday",
        ]

    def test_extract_terms_falls_back_to_whole_text(self):
        # A claim sentence with no content words ("As I said.") leaves the
        # claim in a neighbouring sentence, so the whole reply is used.
        from corvidae_cognition.critique import _extract_terms
        assert _extract_terms("As I said. The budget is 5k.", 8) == ["budget", "5k"]

    def test_extract_terms_splits_on_underscores_like_fts5(self):
        # unicode61 treats "_" as a separator: "___" would be an empty
        # phrase that matches nothing and empties every AND probe.
        from corvidae_cognition.critique import _extract_terms
        assert _extract_terms("You told me the API_KEY field is ___ here.", 8) == [
            "API", "KEY", "field",
        ]

    def test_extract_terms_keeps_decimals_in_the_claim_sentence(self):
        from corvidae_cognition.critique import _extract_terms
        assert _extract_terms("You told me the price was 3.50 per unit.", 8) == [
            "price", "3", "50", "per", "unit",
        ]

    def test_extract_terms_zero_cap_yields_nothing(self):
        # max_terms 0 turns tier 2 off (docs/configuration.md).
        from corvidae_cognition.critique import _extract_terms
        assert _extract_terms("the deadline is Friday", 0) == []


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
        await env.plugin.before_agent_turn(env.channel, "ks", {})
        async with env.db.execute(
            "SELECT provenance_snapshot FROM exchange_log WHERE trace_id = ?",
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
        await env.plugin.before_agent_turn(env.channel, "kn", {})
        async with env.db.execute(
            "SELECT provenance_snapshot FROM exchange_log WHERE trace_id = ?",
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


# Above every row _seed_fts writes: the seeded rows predate the exchange.
ORIGIN_ROWID = 10_000


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


async def _persist_origin(env, trace_id, rowid=ORIGIN_ROWID):
    """Record the exchange's originating message_log rowid, as the Agent's
    on_message_persisted does. The probe only counts rows before it (#155)."""
    await env.plugin.on_message_persisted(
        channel=env.channel, trace_id=trace_id, rowid=rowid,
        text="orig", meta={},
    )


class TestProvenanceGate:
    async def test_past_claim_weak_retrieval_empty_fts_fires(self, tmp_path):
        env = await _low_appraisal_env(tmp_path, chat_text=EMPTY_VERDICT)
        await env.outcome.record_exchange("kpg", env.channel.id, origin="user")
        await env.outcome.update_exchange(
            "kpg", retrieval_top_score=0.1, retrieval_hit_count=1,
        )
        await env.plugin.on_message_admitted(
            channel=env.channel, trace_id="kpg",
            sender="user", text="when is launch?",
        )
        await _persist_origin(env, "kpg")
        await env.plugin.on_agent_response(
            channel=env.channel, request_text="req",
            response_text="You told me the launch date was March.",
            trace_id="kpg", meta={}, logprobs=None,
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
        await env.plugin.on_message_admitted(
            channel=env.channel, trace_id="kstrong",
            sender="user", text="when is launch?",
        )
        await _persist_origin(env, "kstrong")
        await env.plugin.on_agent_response(
            channel=env.channel, request_text="req",
            response_text="You told me the launch date was March.",
            trace_id="kstrong", meta={}, logprobs=None,
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
        await env.plugin.on_message_admitted(
            channel=env.channel, trace_id="kfts",
            sender="user", text="when is launch?",
        )
        await _persist_origin(env, "kfts")
        await env.plugin.on_agent_response(
            channel=env.channel, request_text="req",
            response_text="You told me the launch date was March.",
            trace_id="kfts", meta={}, logprobs=None,
        )
        assert env.task.task_queue.queue.qsize() == 0
        await _teardown(env)

    async def test_no_past_claim_does_not_fire(self, tmp_path):
        env = await _low_appraisal_env(tmp_path)
        await env.outcome.record_exchange("knp", env.channel.id, origin="user")
        await env.outcome.update_exchange(
            "knp", retrieval_top_score=0.1, retrieval_hit_count=0,
        )
        await env.plugin.on_message_admitted(
            channel=env.channel, trace_id="knp",
            sender="user", text="thoughts?",
        )
        await _persist_origin(env, "knp")
        await env.plugin.on_agent_response(
            channel=env.channel, request_text="req",
            response_text="The launch will be great.",
            trace_id="knp", meta={}, logprobs=None,
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
        await env.plugin.on_message_admitted(
            channel=env.channel, trace_id="kdis",
            sender="user", text="q?",
        )
        await _persist_origin(env, "kdis")
        await env.plugin.on_agent_response(
            channel=env.channel, request_text="req",
            response_text="You told me the answer was 42.",
            trace_id="kdis", meta={}, logprobs=None,
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
        assert attr.get("trace_id") == "kempty"
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
            env.channel, trace_id="kfull", meta={"origin": "critique"},
        )
        context_text = "\n".join(
            m.get("content") or ""
            for m in env.channel.conversation.messages
            if m.get("_message_type") == MessageType.CONTEXT
        )
        assert "no record says Friday" in context_text
        await _teardown(env)


def test_parse_json_block_error_names_critique_phase_and_long_text():
    """The critique JSON parse error names its phase and keeps >200 chars."""
    from corvidae_cognition.critique import _parse_json_block

    text = "x" * 300 + " PROVIDER_ERROR_MARKER"
    with pytest.raises(ValueError) as excinfo:
        _parse_json_block(text)

    message = str(excinfo.value)
    assert "critique" in message.lower()
    assert "PROVIDER_ERROR_MARKER" in message


class TestProvenanceProbeIgnoresUnsent:
    """WP2.8 follow-through: a withheld (UNSENT) row is no record that a
    claim was made, so the raw-log probe must not count it."""

    async def test_unsent_row_does_not_count_as_a_record(self, tmp_path):
        from corvidae_cognition.critique import CritiquePlugin

        db_path = str(tmp_path / "sessions.db")
        db = await aiosqlite.connect(db_path)
        await init_db(db)
        await db.execute("CREATE VIRTUAL TABLE message_fts USING fts5(content_text)")
        cursor = await db.execute(
            "INSERT INTO message_log (channel_id, message, timestamp, message_type) VALUES (?, ?, ?, ?)",
            ("test:x", json.dumps({"role": "assistant", "content": "the deadline is Friday"}), 1.0, "unsent"),
        )
        await db.execute(
            "INSERT INTO message_fts(rowid, content_text) VALUES (?, ?)",
            (cursor.lastrowid, "the deadline is Friday"),
        )
        await db.commit()

        plugin = CritiquePlugin()
        plugin._probe_db = db
        try:
            assert await plugin._message_fts_empty("deadline Friday", 8, before_rowid=ORIGIN_ROWID) is True
            await db.execute("UPDATE message_log SET message_type = 'message'")
            await db.commit()
            assert await plugin._message_fts_empty("deadline Friday", 8, before_rowid=ORIGIN_ROWID) is False
        finally:
            await db.close()


class TestProvenanceProbeCutoff:
    """#155 cause 1: the exchange's own rows (at or after its originating
    row) are never a record of its own claim."""

    async def _probe(self, tmp_path, rows):
        from corvidae_cognition.critique import CritiquePlugin

        db = await aiosqlite.connect(str(tmp_path / "sessions.db"))
        await init_db(db)
        await db.execute("CREATE VIRTUAL TABLE message_fts USING fts5(content_text)")
        for rowid, content in rows:
            await db.execute(
                "INSERT INTO message_fts(rowid, content_text) VALUES (?, ?)",
                (rowid, content),
            )
        await db.commit()
        plugin = CritiquePlugin()
        plugin._probe_db = db
        return plugin, db

    async def test_rows_at_or_after_the_cutoff_are_not_a_record(self, tmp_path):
        plugin, db = await self._probe(tmp_path, [(5, "the deadline is Friday")])
        try:
            assert await plugin._message_fts_empty("deadline Friday", 8, before_rowid=5) is True
            assert await plugin._message_fts_empty("deadline Friday", 8, before_rowid=6) is False
        finally:
            await db.close()

    async def test_all_claim_terms_must_appear_in_one_row(self, tmp_path):
        # AND, not OR: a row sharing one content word is no record.
        plugin, db = await self._probe(tmp_path, [(1, "the deadline moved"), (2, "Friday lunch")])
        try:
            assert await plugin._message_fts_empty("deadline Friday", 8, before_rowid=10) is True
        finally:
            await db.close()

    async def test_underscore_run_does_not_hide_a_record(self, tmp_path):
        plugin, db = await self._probe(
            tmp_path, [(1, "the API_KEY field is ___ here")],
        )
        try:
            assert await plugin._message_fts_empty(
                "You told me the API_KEY field is ___ here.", 8, before_rowid=10,
            ) is False
        finally:
            await db.close()

    async def test_no_originating_rowid_does_not_fire(self, tmp_path, caplog):
        # Without the cutoff the probe can't tell the exchange's own rows
        # from a record, so it stays inconclusive (conservative).
        env = await _low_appraisal_env(tmp_path)
        await env.outcome.record_exchange("knorow", env.channel.id, origin="user")
        await env.plugin.on_message_admitted(
            channel=env.channel, trace_id="knorow", sender="user", text="q?",
        )
        try:
            with caplog.at_level("WARNING", logger="corvidae.critique"):
                assert await env.plugin._provenance_should_fire(
                    env.channel, "knorow", "You told me the launch date was March.",
                ) is False
            assert "provenance probe skipped for trace knorow" in caplog.text
        finally:
            await _teardown(env)


# ---------------------------------------------------------------------------
# #155: the provenance gate on the production path — a real Agent turn on a
# file DB with MemoryPlugin's message_fts DDL and insert trigger, and the
# critique probe opened read-only on the same file.
# ---------------------------------------------------------------------------


CLAIM_REPLY = "As I mentioned earlier, the quarterly audit deadline is Thursday."


async def _real_path_env(tmp_path, replies):
    """Agent + persistence on a file DB with the real message_fts trigger,
    and a CritiquePlugin whose gate decisions are recorded per trace id."""
    from unittest.mock import MagicMock

    from corvidae_cognition.critique import CritiquePlugin
    from corvidae_cognition.memory import MESSAGE_FTS_DDL, MESSAGE_FTS_INSERT_TRIGGER
    from helpers import build_plugin_and_channel
    from llm_response_fixtures import _make_text_response

    db_path = str(tmp_path / "sessions.db")
    agent, channel, db = await build_plugin_and_channel(
        mock_on_agent_response=False, db_path=db_path,
    )
    await db.execute(MESSAGE_FTS_DDL)
    await db.execute(MESSAGE_FTS_INSERT_TRIGGER)
    await db.commit()

    pm = agent.pm
    critique = CritiquePlugin()
    pm.register(critique, name="critique")
    cfg = {"daemon": {"session_db": db_path}}
    await critique.on_init(pm=pm, config=cfg)
    await critique.on_start(config=cfg)
    assert critique._probe_db is not None
    # The critic itself is out of scope here: only the gate decision is.
    critique._run_critique = AsyncMock()

    decisions: dict[str, bool] = {}
    real_gate = critique._provenance_should_fire

    async def spy(channel, trace_id, response_text):
        decisions[trace_id] = await real_gate(channel, trace_id, response_text)
        return decisions[trace_id]

    critique._provenance_should_fire = spy

    client = MagicMock()
    client.chat = AsyncMock(side_effect=[_make_text_response(r) for r in replies])
    agent._client = client
    return SimpleNamespace(
        agent=agent, channel=channel, db=db, critique=critique,
        decisions=decisions,
    )


async def _real_path_teardown(env):
    from helpers import drain_task_queue

    await drain_task_queue(env.agent)
    await env.critique.on_stop()
    await env.agent.pm.get_plugin("task").on_stop()
    await env.db.close()


async def _say(env, text):
    from corvidae.trace import mint_trace_id
    from helpers import drain

    trace_id = mint_trace_id()
    await env.agent.on_message(channel=env.channel, sender="user", text=text, trace_id=trace_id)
    await drain(env.agent, env.channel)
    return trace_id


class TestProvenanceGateRealPath:
    async def test_unsupported_claim_fires(self, tmp_path):
        # The reply's own row is in message_fts by the time the gate runs;
        # it must not count as a record of itself.
        env = await _real_path_env(tmp_path, [CLAIM_REPLY])
        try:
            trace_id = await _say(env, "when is it due?")
            assert env.decisions[trace_id] is True
        finally:
            await _real_path_teardown(env)

    async def test_supported_claim_does_not_fire(self, tmp_path):
        env = await _real_path_env(tmp_path, ["Noted.", CLAIM_REPLY])
        try:
            await _say(env, "Heads up: the quarterly audit deadline is Thursday.")
            trace_id = await _say(env, "when is it due?")
            assert env.decisions[trace_id] is False
        finally:
            await _real_path_teardown(env)

    async def test_earlier_row_sharing_only_common_words_still_fires(self, tmp_path):
        env = await _real_path_env(tmp_path, ["Me too.", CLAIM_REPLY])
        try:
            await _say(env, "I like the weather today")
            trace_id = await _say(env, "when is it due?")
            assert env.decisions[trace_id] is True
        finally:
            await _real_path_teardown(env)
