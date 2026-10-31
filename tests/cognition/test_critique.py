"""Tests for CritiquePlugin (WP2.7; the provenance check redesigned in #155).

- provenance snapshot of CONTEXT messages under the trace id
  (before_agent_turn);
- eligibility by PROPAGATED origin (trap #3): user/reminder eligible;
  critique/heartbeat/task exempt (the recursion brake); a user exchange
  ending on a tool-result turn is critiqued against the original user text;
- stylistic lens selection from the appraisal vector
  (predictive/constrained/adversarial) with below-threshold sampling via an
  injectable RNG, marked in the outcome;
- the provenance check (#155): stage-2 appraisal (a model) lists the
  reply's claims about the past; critique fetches the closest earlier
  messages for each, within the channel's memory scope and before the
  exchange; the provenance critic (a model) judges support. No regex or
  word list judges language anywhere on that path;
- silent execution: empty verdict -> no on_notify, outcome recorded;
  non-empty stylistic -> funnel registration, no stub; non-empty
  provenance -> funnel registration + one stub carrying PROVENANCE_NOTE;
- runtime-tunable thresholds via channel.runtime_overrides (directive 2).
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import aiosqlite
import pytest

from corvidae.attribution import get_attribution
from corvidae.channel import Channel, ChannelConfig
from corvidae.context import ContextWindow, MessageType
from corvidae.hooks import CorvidaePlugin, create_plugin_manager, hookimpl, resolve_reject_wins
from corvidae_cognition.outcome_log import OutcomeLogPlugin
from corvidae.persistence import PersistencePlugin, init_db
from corvidae.task import TaskPlugin, TaskQueue

from helpers import FakeForkAgent


EMPTY_VERDICT = json.dumps({"objections": []})
OBJECTION_VERDICT = json.dumps({"objections": [
    {"claim": "the deadline is Friday", "objection": "no record says Friday",
     "severity": 0.7},
]})


def verdicts_text(*supported):
    """A provenance fork reply: one verdict per flag, claim index in order."""
    return json.dumps({"verdicts": [
        {"claim": i + 1, "reason": f"reason {i + 1}", "supported": flag}
        for i, flag in enumerate(supported)]})


SUPPORTED_VERDICT = verdicts_text(True)
UNSUPPORTED_VERDICT = verdicts_text(False)


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
        self.stage2_listeners: list = []

    async def get_appraisal(self, trace_id):
        return self._stage1

    def get_last_stage2(self, channel_id):
        return self._stage2

    def get_originating_text(self, trace_id):
        return self._originating_text.get(trace_id)

    def add_stage2_listener(self, callback):
        # CritiquePlugin registers here to hear the stage-2 past_claims.
        self.stage2_listeners.append(callback)


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
            assert await plugin._probe.get() is not None, (
                "message_fts probe failed to open -- the relative "
                "session_db path did not resolve against _base_dir"
            )
        finally:
            await plugin.on_stop()


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

    async def test_stylistic_objection_registers_without_waking(self, tmp_path):
        appraisal = FakeAppraisal(stage2={"ambiguity": 0.9})
        env = await build_critique(tmp_path, appraisal=appraisal,
                                   chat_text=OBJECTION_VERDICT, with_funnel=True)
        notify = AsyncMock()
        env.pm.ahook.on_notify = notify
        # Spy on both funnel entry points; register keeps its real behaviour.
        register = Mock(wraps=env.funnel.register)
        env.funnel.register = register
        env.funnel.register_and_wake = AsyncMock()
        await _fire(env, "kfull", origin="user")
        await _run_next_task(env.task)
        outcomes = await _read_outcomes(env.db, "kfull")
        assert outcomes["critique"]["objections"] == 1
        # Registered without a stub: no wake of any kind.
        register.assert_called_once()
        assert "no record says Friday" in "\n".join(register.call_args.kwargs["entries"])
        env.funnel.register_and_wake.assert_not_awaited()
        notify.assert_not_awaited()
        # A turn with no origin (a user turn) admits the verdict as framed CONTEXT.
        await env.funnel.before_agent_turn(env.channel, trace_id="kfull", meta={})
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


# ---------------------------------------------------------------------------
# 6. The provenance check (#155): model-detected claims, mechanically
# fetched evidence, a model judging support.
# ---------------------------------------------------------------------------


CLAIM = {"claim": "the quarterly audit deadline is Thursday",
         "queries": ["quarterly audit deadline", "audit Thursday"]}
CLAIM_REPLY = "As I mentioned earlier, the quarterly audit deadline is Thursday."
# No recall phrase at all: only stage 2's past_claims can say this reply
# makes a claim about the past, so no wording filter can be what triggers.
CUE_FREE_REPLY = "The quarterly audit deadline is Thursday."


async def _log(env, content, channel_id=None, message_type="message", role="user", ts=1.0):
    """Insert a message_log row and index it in message_fts under the same
    rowid, as MemoryPlugin's message_log_ai trigger would."""
    cursor = await env.db.execute(
        "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
        "VALUES (?, ?, ?, ?)",
        (channel_id or env.channel.id, json.dumps({"role": role, "content": content}),
         ts, message_type),
    )
    await env.db.execute(
        "INSERT INTO message_fts(rowid, content_text) VALUES (?, ?)",
        (cursor.lastrowid, content),
    )
    await env.db.commit()
    return cursor.lastrowid


async def _provenance_env(tmp_path, chat_text=OBJECTION_VERDICT, with_funnel=False):
    appraisal = FakeAppraisal()
    return await build_critique(
        tmp_path, appraisal=appraisal, chat_text=chat_text, with_funnel=with_funnel,
        config={"critique": {"sample_below_rate": 0.0}},
    )


async def _begin_exchange(env, trace_id, origin_rowid, origin="user"):
    """Admit (for user origin) and persist the exchange's originating row."""
    await env.outcome.record_exchange(trace_id, env.channel.id, origin=origin)
    if origin == "user":
        await env.plugin.on_message_admitted(
            channel=env.channel, trace_id=trace_id, sender="user", text="when is it due?",
        )
    await env.plugin.on_message_persisted(
        channel=env.channel, trace_id=trace_id, rowid=origin_rowid,
        text="when is it due?", meta={},
    )


async def _stage2(env, trace_id, claims, meta=None, response_text=CUE_FREE_REPLY):
    """Deliver a stage-2 result the way AppraisalPlugin's listener call does."""
    [listener] = env.appraisal.stage2_listeners
    await listener(
        channel=env.channel, trace_id=trace_id, response_text=response_text,
        meta=meta or {}, stage2={"valence": 0.5, "past_claims": claims},
    )


def _candidate_rowids(evidence):
    return [[c["rowid"] for c in e["candidates"]] for e in evidence]


SYSTEM_PROMPT = "You are Lex. Standing fact: the office is in Lisbon."
TOOL_TEXT = "lookup result: the quarterly audit deadline is Thursday"


class RenderingForkAgent(FakeForkAgent):
    """A fake fork agent that also records the messages the fork request
    would carry (build_fork_prompt on the channel's window and the cut)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.requests: list[list[dict]] = []

    async def fork(self, channel, instruction, *, cut=None, extra_body=None):
        from corvidae.fork import build_fork_prompt

        if channel.conversation is not None:
            self.requests.append(build_fork_prompt(
                channel.conversation, cut, instruction, frozenset(), "system"))
        return await super().fork(
            channel, instruction, cut=cut, extra_body=extra_body)


def _row(role, content, rowid, **extra):
    msg = {"role": role, "content": content, "_message_type": MessageType.MESSAGE, **extra}
    if rowid is not None:
        msg["_db_id"] = rowid
    return msg


async def _provenance_exchange(env, agent, trace_id="kp", reply_text=CUE_FREE_REPLY,
                               support=None, persist_reply=True):
    """A finished exchange in a live window: user message, a tool call and
    its result, then the reply. The rows are in message_log (and the FTS
    index) and in the channel's window; the exchange is begun and
    on_agent_response has fired. ``agent`` is registered as the Agent plugin."""
    from corvidae.context import ContextWindow

    env.pm.register(agent, name="agent")
    if support:
        await _log(env, support)
    origin = await _log(env, "when is it due?")
    call_row = await _log(env, "", role="assistant")
    tool_row = await _log(env, TOOL_TEXT, role="tool")
    reply_row = await _log(env, reply_text, role="assistant")
    conv = ContextWindow(env.channel.id)
    conv.system_prompt = SYSTEM_PROMPT
    conv.messages = [
        _row("user", "when is it due?", origin),
        _row("assistant", "", call_row, tool_calls=[{
            "id": "c1", "type": "function",
            "function": {"name": "lookup", "arguments": "{}"}}]),
        _row("tool", TOOL_TEXT, tool_row, tool_call_id="c1"),
        _row("assistant", reply_text, reply_row if persist_reply else None),
    ]
    env.channel.conversation = conv
    await _begin_exchange(env, trace_id, origin_rowid=origin)
    await env.plugin.on_agent_response(
        channel=env.channel, request_text="when is it due?", response_text=reply_text,
        trace_id=trace_id, meta={}, logprobs=None,
    )
    return SimpleNamespace(origin=origin, tool_row=tool_row, reply_row=reply_row,
                           conv=conv, trace_id=trace_id)


async def _run_provenance(env, trace_id="kp", claims=None):
    """Deliver stage 2's claims and run the provenance task they enqueue."""
    await _stage2(env, trace_id, [CLAIM] if claims is None else claims)
    await _run_next_task(env.task)


def _provenance_instructions(agent):
    return [c["instruction"] for c in agent.calls
            if c["instruction"].startswith(_provenance_fork_prompt())]


def _provenance_fork_prompt():
    from corvidae_cognition.critique import DEFAULT_FORK_PROMPTS

    return DEFAULT_FORK_PROMPTS["provenance"]


class TestProvenanceTrigger:
    async def test_registers_a_stage2_listener(self, tmp_path):
        env = await _provenance_env(tmp_path)
        try:
            assert len(env.appraisal.stage2_listeners) == 1
        finally:
            await _teardown(env)

    async def test_no_appraisal_warns_that_provenance_is_off(self, tmp_path, caplog):
        with caplog.at_level("WARNING", logger="corvidae.critique"):
            env = await build_critique(tmp_path)
        try:
            assert "provenance check disabled" in caplog.text
            assert "appraisal" in caplog.text
        finally:
            await _teardown(env)

    async def test_on_agent_response_never_selects_provenance(self, tmp_path):
        # Provenance waits for stage-2's claims; the reply's wording alone
        # triggers nothing.
        appraisal = FakeAppraisal(stage2={"ambiguity": 0.9})
        env = await build_critique(tmp_path, appraisal=appraisal, chat_text=EMPTY_VERDICT)
        try:
            # With an originating rowid and an empty message_fts, the old
            # wording-triggered gate would have added "provenance" here.
            await env.plugin.on_message_persisted(
                channel=env.channel, trace_id="kst", rowid=50, text="q?", meta={},
            )
            await _fire(env, "kst", response_text=CLAIM_REPLY)
            await _run_next_task(env.task)
            outcomes = await _read_outcomes(env.db, "kst")
            assert outcomes["critique"]["lenses"] == ["predictive"]
        finally:
            await _teardown(env)

    async def test_claims_enqueue_one_silent_task(self, tmp_path):
        env = await _provenance_env(tmp_path)
        try:
            await _begin_exchange(env, "kp", origin_rowid=50)
            await _stage2(env, "kp", [CLAIM])
            assert env.task.task_queue.queue.qsize() == 1
            task = env.task.task_queue.queue.get_nowait()
            assert task.deliver is False
        finally:
            await _teardown(env)

    async def test_no_claims_no_task(self, tmp_path):
        env = await _provenance_env(tmp_path)
        try:
            await _begin_exchange(env, "kn", origin_rowid=50)
            await _stage2(env, "kn", [])
            assert env.task.task_queue.queue.qsize() == 0
        finally:
            await _teardown(env)

    @pytest.mark.parametrize("origin", ["critique", "heartbeat", "task"])
    async def test_exempt_origins_no_task(self, tmp_path, origin):
        env = await _provenance_env(tmp_path)
        try:
            await _begin_exchange(env, "kx", origin_rowid=50, origin=origin)
            meta = {} if origin == "task" else {"origin": origin}
            await _stage2(env, "kx", [CLAIM], meta=meta)
            assert env.task.task_queue.queue.qsize() == 0
        finally:
            await _teardown(env)

    async def test_reminder_origin_eligible(self, tmp_path):
        env = await _provenance_env(tmp_path)
        try:
            await _begin_exchange(env, "kr", origin_rowid=50, origin="reminder")
            await _stage2(env, "kr", [CLAIM], meta={"origin": "reminder"})
            assert env.task.task_queue.queue.qsize() == 1
        finally:
            await _teardown(env)

    async def test_disabled_no_task(self, tmp_path):
        env = await _provenance_env(tmp_path)
        env.channel.runtime_overrides["critique.provenance.enabled"] = False
        try:
            await _begin_exchange(env, "kd", origin_rowid=50)
            await _stage2(env, "kd", [CLAIM])
            assert env.task.task_queue.queue.qsize() == 0
        finally:
            await _teardown(env)


class TestProvenanceEvidence:
    async def test_rows_before_the_exchange_are_candidates(self, tmp_path):
        env = await _provenance_env(tmp_path)
        try:
            earlier = await _log(env, "Heads up: the quarterly audit deadline is Thursday.")
            origin = await _log(env, "when is it due?")
            reply = await _log(env, CLAIM_REPLY, role="assistant")
            await _begin_exchange(env, "ke", origin_rowid=origin)
            evidence = await env.plugin._gather_evidence(env.channel, "ke", [CLAIM])
            assert _candidate_rowids(evidence) == [[earlier]]
            assert reply not in _candidate_rowids(evidence)[0]
            assert evidence[0]["claim"] == CLAIM["claim"]
            assert "quarterly audit deadline" in evidence[0]["candidates"][0]["content"]
            assert evidence[0]["candidates"][0]["role"] == "user"
        finally:
            await _teardown(env)

    async def test_unsent_and_context_rows_are_not_candidates(self, tmp_path):
        env = await _provenance_env(tmp_path)
        try:
            await _log(env, "the quarterly audit deadline is Thursday", message_type="unsent",
                       role="assistant")
            await _log(env, "[critique] the quarterly audit deadline is Thursday",
                       message_type="context", role="system")
            summary = await _log(env, "Summary: audit deadline set for Thursday",
                                 message_type="summary", role="system")
            origin = await _log(env, "when is it due?")
            await _begin_exchange(env, "ku", origin_rowid=origin)
            evidence = await env.plugin._gather_evidence(env.channel, "ku", [CLAIM])
            assert _candidate_rowids(evidence) == [[summary]]
        finally:
            await _teardown(env)

    async def test_only_channels_in_memory_scope(self, tmp_path):
        from corvidae_cognition.memory import ChannelScope

        env = await _provenance_env(tmp_path)

        class ScopeOnlyMemory(CorvidaePlugin):
            def _channel_scope(self, channel_id):
                return ChannelScope(frozenset({channel_id}), ("signal:*",))

        env.pm.register(ScopeOnlyMemory(), name="memory")
        try:
            other = await _log(env, "the quarterly audit deadline is Thursday",
                               channel_id="irc:#elsewhere")
            grouped = await _log(env, "the quarterly audit deadline is Thursday",
                                 channel_id="signal:alice")
            origin = await _log(env, "when is it due?")
            await _begin_exchange(env, "ks", origin_rowid=origin)
            evidence = await env.plugin._gather_evidence(env.channel, "ks", [CLAIM])
            assert _candidate_rowids(evidence) == [[grouped]]
            assert other not in _candidate_rowids(evidence)[0]
        finally:
            await _teardown(env)

    async def test_without_memory_only_the_channel_itself(self, tmp_path):
        env = await _provenance_env(tmp_path)
        try:
            await _log(env, "the quarterly audit deadline is Thursday", channel_id="irc:#elsewhere")
            own = await _log(env, "the quarterly audit deadline is Thursday")
            origin = await _log(env, "when is it due?")
            await _begin_exchange(env, "kw", origin_rowid=origin)
            evidence = await env.plugin._gather_evidence(env.channel, "kw", [CLAIM])
            assert _candidate_rowids(evidence) == [[own]]
        finally:
            await _teardown(env)

    async def test_closest_first_and_capped(self, tmp_path):
        from corvidae_cognition.critique import PROVENANCE_CANDIDATES

        env = await _provenance_env(tmp_path)
        try:
            for i in range(PROVENANCE_CANDIDATES + 3):
                await _log(env, f"the audit is on the calendar, note {i}")
            best = await _log(env, "quarterly audit deadline: Thursday")
            origin = await _log(env, "when is it due?")
            await _begin_exchange(env, "kc", origin_rowid=origin)
            [ev] = await env.plugin._gather_evidence(env.channel, "kc", [CLAIM])
            assert len(ev["candidates"]) == PROVENANCE_CANDIDATES
            assert ev["candidates"][0]["rowid"] == best
        finally:
            await _teardown(env)

    @pytest.mark.parametrize("query", [
        'NOT audit', 'say "hi"', "audit*", "content_text: audit", "^audit",
        "audit AND OR", "___", "(audit", "naïve café", "Thursday🤞",
    ])
    async def test_query_syntax_is_inert(self, tmp_path, query):
        # Model-written queries are data: FTS5 syntax in them is quoted,
        # never parsed, and never raises.
        env = await _provenance_env(tmp_path)
        try:
            await _log(env, "the audit is Thursday")
            origin = await _log(env, "when is it due?")
            await _begin_exchange(env, "kq", origin_rowid=origin)
            evidence = await env.plugin._gather_evidence(
                env.channel, "kq", [{"claim": "c", "queries": [query]}],
            )
            assert len(evidence) == 1
        finally:
            await _teardown(env)

    async def test_no_originating_rowid_gathers_nothing(self, tmp_path, caplog):
        # Without the cutoff the exchange's own rows can't be told apart.
        env = await _provenance_env(tmp_path)
        try:
            await env.plugin.on_message_admitted(
                channel=env.channel, trace_id="knr", sender="user", text="q?",
            )
            with caplog.at_level("WARNING", logger="corvidae.critique"):
                evidence = await env.plugin._gather_evidence(env.channel, "knr", [CLAIM])
            assert evidence is None
            assert "knr" in caplog.text
        finally:
            await _teardown(env)

    async def test_no_probe_gathers_nothing(self, tmp_path, caplog):
        env = await _provenance_env(tmp_path)
        try:
            await _begin_exchange(env, "kpr", origin_rowid=50)
            # A closed probe is the real disable: get() returns None.
            await env.plugin._probe.close()
            with caplog.at_level("WARNING", logger="corvidae.critique"):
                evidence = await env.plugin._gather_evidence(env.channel, "kpr", [CLAIM])
            assert evidence is None
            assert "kpr" in caplog.text
        finally:
            await _teardown(env)


class TestProvenanceCritic:
    """The provenance check runs as a fork of the window (#164)."""

    async def test_fork_sees_claims_evidence_and_snapshot(self, tmp_path):
        env = await _provenance_env(tmp_path)
        agent = RenderingForkAgent({"critique"}, text=UNSUPPORTED_VERDICT)
        try:
            ex = await _provenance_exchange(
                env, agent, support="Heads up: the quarterly audit deadline is Thursday.")
            await env.outcome.update_exchange(
                "kp", provenance_snapshot=json.dumps([{"content": "memory: audits are quarterly"}]),
            )
            await _run_provenance(env)
            [call] = agent.calls
            instruction = call["instruction"]
            assert instruction.startswith(_provenance_fork_prompt())
            assert "confabulated memory" in instruction
            assert CLAIM["claim"] in instruction
            assert "Heads up: the quarterly audit deadline is Thursday." in instruction
            assert "memory: audits are quarterly" in instruction
            assert call["attribution"]["stage"] == "critique"
            assert env.stub.chat_calls == []
            outcomes = await _read_outcomes(env.db, "kp")
            assert outcomes["provenance"] == {"claims": 1, "candidates": 1, "supported": 0, "objections": 1}
        finally:
            await _teardown(env)

    async def test_same_exchange_rows_are_not_candidates(self, tmp_path):
        # The tool row indexes the claim's words but sits inside the exchange
        # (and the window), so it is in the fork request, not the search.
        env = await _provenance_env(tmp_path)
        agent = RenderingForkAgent({"critique"}, text=SUPPORTED_VERDICT)
        try:
            ex = await _provenance_exchange(env, agent)
            evidence = await env.plugin._gather_evidence(env.channel, "kp", [CLAIM])
            assert _candidate_rowids(evidence) == [[]]
            await _run_provenance(env)
            [request] = agent.requests
            assert any(m.get("content") == TOOL_TEXT for m in request)
        finally:
            await _teardown(env)

    async def test_no_evidence_is_shown_as_none_found(self, tmp_path):
        env = await _provenance_env(tmp_path)
        agent = RenderingForkAgent({"critique"}, text=SUPPORTED_VERDICT)
        try:
            await _provenance_exchange(env, agent)
            await _run_provenance(env)
            [instruction] = _provenance_instructions(agent)
            assert "no earlier message matched" in instruction
        finally:
            await _teardown(env)

    async def test_objections_reach_the_funnel(self, tmp_path):
        from corvidae_cognition.critique import PROVENANCE_NOTE

        env = await _provenance_env(tmp_path, with_funnel=True)
        env.funnel.register_and_wake = AsyncMock()
        agent = RenderingForkAgent({"critique"}, text=UNSUPPORTED_VERDICT)
        try:
            await _provenance_exchange(env, agent, trace_id="kf")
            await _run_provenance(env, "kf")
            env.funnel.register_and_wake.assert_awaited_once()
            kwargs = env.funnel.register_and_wake.await_args.kwargs
            assert kwargs["origin"] == "critique"
            assert kwargs["entries"][0].startswith("[provenance]")
            assert kwargs["stub_text"] == PROVENANCE_NOTE
        finally:
            await _teardown(env)

    async def test_failed_search_never_reaches_the_critic_as_no_match(self, tmp_path, caplog):
        # A search that raised looked at nothing: telling the critic "no
        # earlier message matched" would turn a supported claim into an
        # objection. A lone surrogate (a truncated emoji escape in the
        # model's JSON) is one real input that makes sqlite's bind raise.
        env = await _provenance_env(tmp_path)
        agent = RenderingForkAgent({"critique"}, text=UNSUPPORTED_VERDICT)
        try:
            await _provenance_exchange(
                env, agent, trace_id="kfail",
                support="Heads up: the quarterly audit deadline is Thursday.")
            claims = [CLAIM, {"claim": "we agreed on Tuesday", "queries": ["\ud83d Tuesday"]}]
            with caplog.at_level("WARNING", logger="corvidae.critique"):
                assert await env.plugin._gather_evidence(env.channel, "kfail", claims) is None
                await _run_provenance(env, "kfail", claims)
            assert agent.calls == [] and env.stub.chat_calls == []
            assert "evidence search failed for trace kfail" in caplog.text
            assert "skipping the provenance check" in caplog.text
        finally:
            await _teardown(env)

    async def test_no_evidence_available_skips_the_critic(self, tmp_path, caplog):
        # An unconfirmed absence must not reach the critic as "no record":
        # with the fork available but no originating row to bound the
        # search, the check is skipped.
        env = await _provenance_env(tmp_path)
        agent = RenderingForkAgent({"critique"}, text=UNSUPPORTED_VERDICT)
        try:
            await _provenance_exchange(env, agent, trace_id="kne")
            env.plugin._originating_rowid.pop("kne")
            with caplog.at_level("WARNING", logger="corvidae.critique"):
                await _run_provenance(env, "kne")
            assert agent.calls == [] and env.stub.chat_calls == []
            assert "no originating message_log row" in caplog.text
        finally:
            await _teardown(env)

    async def test_long_hit_is_shown_around_the_match(self, tmp_path):
        # R7: the matched term sits past the 500-character excerpt limit.
        from corvidae_cognition.critique import PROVENANCE_EXCERPT_CHARS

        env = await _provenance_env(tmp_path)
        agent = RenderingForkAgent({"critique"}, text=SUPPORTED_VERDICT)
        long_text = ("filler words about nothing " * 30) + " the zebrafish permit lapses Friday " + (
            "more filler " * 60)
        assert long_text.index("zebrafish") > PROVENANCE_EXCERPT_CHARS
        # Neither the head nor the tail of the row reaches the match.
        assert len(long_text) - long_text.index("zebrafish") > PROVENANCE_EXCERPT_CHARS
        try:
            await _provenance_exchange(env, agent, support=long_text)
            claim = {"claim": "the permit lapses Friday", "queries": ["zebrafish"]}
            await _run_provenance(env, claims=[claim])
            [instruction] = _provenance_instructions(agent)
            assert "zebrafish permit lapses Friday" in instruction
            assert "……" not in instruction
            line = next(ln for ln in instruction.splitlines() if "zebrafish" in ln and "«" in ln)
            excerpt = line[line.index("«") + 1:line.rindex("»")]
            assert len(excerpt) <= PROVENANCE_EXCERPT_CHARS
        finally:
            await _teardown(env)

    async def test_short_hit_is_shown_whole(self, tmp_path):
        env = await _provenance_env(tmp_path)
        agent = RenderingForkAgent({"critique"}, text=SUPPORTED_VERDICT)
        try:
            await _provenance_exchange(
                env, agent, support="Heads up: the quarterly audit deadline is Thursday.")
            await _run_provenance(env)
            [instruction] = _provenance_instructions(agent)
            assert "«Heads up: the quarterly audit deadline is Thursday.»" in instruction
        finally:
            await _teardown(env)

    async def test_non_string_content_hit_is_shown_empty(self, tmp_path):
        env = await _provenance_env(tmp_path)
        agent = RenderingForkAgent({"critique"}, text=SUPPORTED_VERDICT)
        try:
            cursor = await env.db.execute(
                "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
                "VALUES (?, ?, ?, ?)",
                (env.channel.id, json.dumps({"role": "user", "content": [{"type": "text"}]}),
                 1.0, "message"),
            )
            await env.db.execute(
                "INSERT INTO message_fts(rowid, content_text) VALUES (?, ?)",
                (cursor.lastrowid, "zebrafish permit"),
            )
            await env.db.commit()
            await _provenance_exchange(env, agent)
            claim = {"claim": "the permit lapses Friday", "queries": ["zebrafish"]}
            await _run_provenance(env, claims=[claim])
            [instruction] = _provenance_instructions(agent)
            assert "«»" in instruction
        finally:
            await _teardown(env)


# ---------------------------------------------------------------------------
# #155 on the production path: a real Agent turn on a file DB with
# MemoryPlugin's message_fts DDL and insert trigger, real AppraisalPlugin
# stage 2 (stubbed model) and CritiquePlugin's read-only probe on that file.
# ---------------------------------------------------------------------------


class RoleStub:
    """An LLM client stand-in answering every call with fixed text."""

    def __init__(self, text):
        self.text = text
        self.calls: list[list[dict]] = []

    async def chat(self, messages, tools=None, extra_body=None):
        self.calls.append(messages)
        return {"choices": [{"message": {"role": "assistant", "content": self.text}}]}


def _stage2_reply(past_claims):
    return json.dumps({
        "valence": 0.5, "stakes": 0.5, "ambiguity": 0.0, "commitment_density": 0.0,
        "novelty": 0.5, "correction": False, "past_claims": past_claims,
    })


async def _real_path_env(tmp_path, replies, past_claims):
    from unittest.mock import MagicMock

    from corvidae.llm_plugin import LLMPlugin
    from corvidae_cognition.appraisal import AppraisalPlugin
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
    cfg = {"daemon": {"session_db": db_path}, "critique": {"sample_below_rate": 0.0}}
    outcome = OutcomeLogPlugin()
    pm.register(outcome, name="outcome_log")
    await outcome.on_init(pm=pm, config=cfg)
    await outcome.on_start(config=cfg)

    llm = LLMPlugin()
    pm.register(llm, name="llm")
    llm._clients["appraisal"] = RoleStub(_stage2_reply(past_claims))
    critic = RoleStub(OBJECTION_VERDICT)
    llm._clients["critic"] = critic

    plugins = []
    for name, cls in (("appraisal", AppraisalPlugin), ("critique", CritiquePlugin)):
        p = cls()
        pm.register(p, name=name)
        await p.on_init(pm=pm, config=cfg)
        plugins.append(p)
    for p in plugins:
        await p.on_start(config=cfg)
    critique = plugins[1]
    assert await critique._probe.get() is not None

    gathered: dict[str, list] = {}
    real_gather = critique._gather_evidence

    async def spy(channel, trace_id, claims):
        gathered[trace_id] = await real_gather(channel, trace_id, claims)
        return gathered[trace_id]

    critique._gather_evidence = spy

    # Forks on for critique; the fork request is recorded and answered with
    # an objection (the agent's own client is reserved for the turns).
    from corvidae.fork import ForkResult, ForksConfig, build_fork_prompt

    agent._forks = ForksConfig(enabled=True, stages=frozenset({"critique"}))
    forks: list[dict] = []

    async def fork(ch, instruction, *, cut=None, extra_body=None):
        forks.append({
            "instruction": instruction, "cut": cut,
            "request": build_fork_prompt(
                ch.conversation, cut, instruction, frozenset(), "system"),
        })
        return ForkResult(text=UNSUPPORTED_VERDICT, message={"role": "assistant"})

    agent.fork = fork

    client = MagicMock()
    client.chat = AsyncMock(side_effect=[_make_text_response(r) for r in replies])
    agent._client = client
    return SimpleNamespace(
        agent=agent, channel=channel, db=db, critic=critic, gathered=gathered,
        plugins=plugins, forks=forks,
    )


async def _real_path_teardown(env):
    for p in reversed(env.plugins):
        await p.on_stop()
    await env.agent.pm.get_plugin("task").on_stop()
    await env.db.close()


async def _say(env, text):
    from corvidae.trace import mint_trace_id
    from helpers import drain, drain_task_queue

    trace_id = mint_trace_id()
    await env.agent.on_message(channel=env.channel, sender="user", text=text, trace_id=trace_id)
    await drain(env.agent, env.channel)
    # Stage 2, then the provenance task stage 2's listener enqueued.
    await drain_task_queue(env.agent)
    await drain_task_queue(env.agent)
    return trace_id


async def _row_texts(db, rowids):
    texts = []
    for rowid in rowids:
        async with db.execute("SELECT message FROM message_log WHERE id = ?", (rowid,)) as cur:
            texts.append(json.loads((await cur.fetchone())[0])["content"])
    return texts


def _real_provenance_forks(env):
    prompt = _provenance_fork_prompt()
    return [f for f in env.forks if f["instruction"].startswith(prompt)]


class TestProvenanceRealPath:
    async def test_unsupported_claim_reaches_the_fork_with_no_evidence(self, tmp_path):
        # The reply's own row is already indexed when the check runs; it
        # must not be offered as a record of itself.
        env = await _real_path_env(tmp_path, [CUE_FREE_REPLY], [CLAIM])
        try:
            trace_id = await _say(env, "when is it due?")
            assert _candidate_rowids(env.gathered[trace_id]) == [[]]
            [fork] = _real_provenance_forks(env)
            assert CLAIM["claim"] in fork["instruction"]
            assert env.critic.calls == []
            outcomes = await _read_outcomes(env.db, trace_id)
            assert outcomes["provenance"]["objections"] == 1
        finally:
            await _real_path_teardown(env)

    async def test_supporting_earlier_row_is_put_in_front_of_the_fork(self, tmp_path):
        env = await _real_path_env(tmp_path, ["Noted.", CLAIM_REPLY], [CLAIM])
        try:
            await _say(env, "Heads up: the quarterly audit deadline is Thursday.")
            trace_id = await _say(env, "when is it due?")
            [rowids] = _candidate_rowids(env.gathered[trace_id])
            texts = await _row_texts(env.db, rowids)
            assert texts[0] == "Heads up: the quarterly audit deadline is Thursday."
            assert "when is it due?" not in texts and CLAIM_REPLY not in texts
            # Turn 1 is checked too: the stub claims on every stage 2.
            provenance = _real_provenance_forks(env)
            assert len(provenance) == 2
            assert texts[0] in provenance[-1]["instruction"]
            assert env.critic.calls == []
            # The fork ends at the reply of the exchange it audits.
            assert provenance[-1]["request"][-2]["content"] == CLAIM_REPLY
        finally:
            await _real_path_teardown(env)

    async def test_reply_without_past_claims_runs_no_provenance_fork(self, tmp_path):
        env = await _real_path_env(tmp_path, ["It is due Thursday."], [])
        try:
            await _say(env, "when is it due?")
            assert _real_provenance_forks(env) == []
            assert env.critic.calls == []
        finally:
            await _real_path_teardown(env)


class TestProvenancePrompts:
    """The critic must not take the agent's word for a fact: its own
    earlier statement shows only that it said so (PR #156 follow-up)."""

    def test_own_earlier_statement_is_not_a_record_of_a_fact(self):
        prompt = _provenance_fork_prompt()
        assert "shows only that it was said" in prompt
        assert "a user message, a tool result, a summary or the snapshot" in prompt

    def test_summary_rows_are_labelled_summary_not_assistant(self):
        # Compaction stores a summary as an assistant turn; shown to the
        # critic under that role it would read as the agent's own word.
        from corvidae_cognition.critique import CritiquePlugin, _candidate

        row = (9, "test:x", json.dumps({
            "role": "assistant", "content": "[Summary of earlier conversation]\naudit Thursday",
        }), "summary", 1.0)
        candidate = _candidate(row)
        assert candidate["role"] == "summary"
        rendered = CritiquePlugin._render_evidence([{"claim": "c", "candidates": [candidate]}])
        assert ", summary, test:x]" in rendered and "assistant" not in rendered

    def test_fork_prompt_names_the_summary_turn(self):
        from corvidae_cognition.critique import DEFAULT_FORK_PROMPTS

        assert "[Summary of earlier conversation]" in DEFAULT_FORK_PROMPTS["provenance"]

    def test_fork_prompt_excludes_the_audited_response(self):
        from corvidae_cognition.critique import DEFAULT_FORK_PROMPTS

        assert "not a record of itself" in DEFAULT_FORK_PROMPTS["provenance"]

    def test_provenance_has_no_standalone_prompt(self):
        from corvidae_cognition.critique import DEFAULT_PROMPTS

        assert "provenance" not in DEFAULT_PROMPTS

    def test_fork_prompt_counts_system_prompt_and_tool_results_as_records(self):
        prompt = _provenance_fork_prompt().lower()
        assert "your system prompt" in prompt
        assert "tool results from this exchange" in prompt


def test_provenance_note_names_the_no_reply_marker():
    from corvidae_cognition.critique import NO_REPLY_MARKER, PROVENANCE_NOTE

    assert NO_REPLY_MARKER in PROVENANCE_NOTE


class TestNoReplyVeto:
    """CritiquePlugin.should_send_response vetoes a critique turn's NO_REPLY."""

    async def _verdict(self, text="NO_REPLY", emission="final", origin="critique"):
        from corvidae_cognition.critique import CritiquePlugin

        # Dispatch through the plugin manager, as the kernel's _gate_send
        # does, so a method missing @hookimpl is never consulted.
        pm = create_plugin_manager()
        pm.register(CritiquePlugin(), name="critique")
        results = await pm.ahook.should_send_response(
            channel=Channel(transport="test", scope="veto"),
            text=text, emission=emission, trace_id="kveto", meta={"origin": origin},
        )
        return resolve_reject_wins(results)

    @pytest.mark.parametrize("text", ["NO_REPLY", "  NO_REPLY\n"])
    async def test_marker_on_a_critique_final_is_vetoed(self, text):
        assert await self._verdict(text=text) is False

    async def test_user_origin_has_no_opinion(self):
        assert await self._verdict(origin="user") is None

    async def test_progress_emission_has_no_opinion(self):
        assert await self._verdict(emission="progress") is None

    async def test_other_text_has_no_opinion(self):
        assert await self._verdict(text="The deadline is Thursday.") is None


# ---------------------------------------------------------------------------
# Lazy message_fts probe (#159)
# ---------------------------------------------------------------------------


async def _swap_in_empty_probe(env, tmp_path, clock):
    """Point the plugin's probe at a DB with no message_fts, with a fake clock."""
    from corvidae_cognition.probe import ReadOnlyProbe

    await env.plugin._probe.close()
    path = str(tmp_path / "late.db")
    async with aiosqlite.connect(path) as late:
        # message_log exists (kernel schema) but message_fts does not yet.
        await init_db(late)
    env.plugin._probe = ReadOnlyProbe(
        path, "message_fts", "critique provenance",
        "the provenance check stays off until message_fts exists",
        clock=clock,
    )
    return path


class TestLazyProbe:
    async def test_provenance_starts_working_when_table_appears(self, tmp_path, caplog):
        """#159 R1/R5: no restart; skip warning kept while absent; recovers."""
        from corvidae_cognition.probe import PROBE_RETRY_SECONDS

        env = await _provenance_env(tmp_path)
        try:
            now = [0.0]
            path = await _swap_in_empty_probe(env, tmp_path, lambda: now[0])
            await _begin_exchange(env, "klz", origin_rowid=50)
            with caplog.at_level("WARNING", logger="corvidae"):
                assert await env.plugin._gather_evidence(
                    env.channel, "klz", [CLAIM]) is None
            # The probe warning names the path; the per-trace skip stays.
            assert path in caplog.text
            assert "provenance check skipped for trace klz" in caplog.text

            # message_fts appears later; the interval elapses.
            await _seed_fts(path, ["the quarterly audit deadline is Thursday"])
            now[0] = PROBE_RETRY_SECONDS + 1
            evidence = await env.plugin._gather_evidence(env.channel, "klz", [CLAIM])
            assert evidence is not None
        finally:
            await _teardown(env)

    async def test_on_stop_closes_probe_connection(self, tmp_path):
        """#159 R6: on_stop leaves no open connection."""
        env = await _provenance_env(tmp_path)
        try:
            await _begin_exchange(env, "kst", origin_rowid=50)
            await env.plugin._gather_evidence(env.channel, "kst", [CLAIM])
            conn = env.plugin._probe.connection
            assert conn is not None
            await env.plugin.on_stop()
            assert env.plugin._probe.connection is None
            assert conn._connection is None
        finally:
            await _teardown(env)
