"""Tests for CritiquePlugin (WP2.7; the provenance check redesigned in #155).

- eligibility (#1031): every main-loop turn with a trace is judged,
  whatever its origin; findings on a ``critique``-origin turn (a correction)
  wait for the next turn and never wake one (the recursion brake); a user
  exchange ending on a tool-result turn is critiqued against the original
  user text;
- stylistic lens selection from the appraisal vector
  (predictive/constrained/adversarial) with below-threshold sampling via an
  injectable RNG, marked in the outcome;
- the provenance check (#155): stage-2 appraisal (a model) lists the
  reply's claims about the past; critique fetches the closest earlier
  messages for each, on every channel and before the
  exchange; the provenance critic (a model) judges support. No regex or
  word list judges language anywhere on that path;
- silent execution: empty verdict -> no on_notify, outcome recorded;
  non-empty stylistic -> held-store hold, no stub; non-empty
  provenance -> held-store hold_and_wake + one stub carrying PROVENANCE_NOTE;
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
from corvidae.hooks import (
    CorvidaePlugin,
    create_plugin_manager,
    hookimpl,
    install_plugin_hookspecs,
    resolve_reject_wins,
)
from corvidae_cognition.outcome_log import OutcomeLogPlugin
from corvidae.persistence import PersistencePlugin, init_db
from corvidae.task import TaskPlugin, TaskQueue

from tests.cognition.trace_helpers import TraceRecorder
from fake_clock import install_kernel_clock
from helpers import add_kernel_held, add_kernel_registry, CORRUPT_MESSAGES, FakeForkAgent, insert_raw_message, skip_warnings, window_of


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
    """Stands for a registered appraisal plugin. Critique decides presence
    with ``pm.has_plugin("appraisal")`` and acts through appraisal's hooks
    only (#1169), so this has no methods: tests call critique's
    ``on_appraisal_*`` hookimpls directly, as appraisal's task would."""

    depends_on = frozenset()


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


def _register_trace(pm) -> TraceRecorder:
    """Register the trace plugin and a recorder on ``pm``. The hookspecs are
    left to install_plugin_hookspecs, which every caller runs afterwards."""
    from corvidae_cognition.trace import TracePlugin

    recorder = TraceRecorder()
    pm.register(TracePlugin(), name="trace")
    pm.register(recorder, name="trace_recorder")
    return recorder


async def build_critique(
    tmp_path, appraisal=None, chat_text=OBJECTION_VERDICT, fts_contents=(),
    with_held=False, config=None, fork=True, extra_plugins=None,
):
    """Wire critique + outcome_log + llm(stub) + task over a file DB.

    ``fork`` registers a ``FakeForkAgent({"critique"})`` answering with
    ``chat_text`` and gives the channel a window holding rowids 1, 2 and 50,
    the rows these tests give their exchanges. The stylistic lenses are forks
    of the window (#320); a miss skips, so without both the lens never runs.
    Tests that register their own agent pass ``fork=False``. ``extra_plugins``
    (name -> plugin) are registered beside critique and their hookspecs
    installed before critique's ``on_init``, as the daemon does.
    """
    from corvidae_cognition.critique import CritiquePlugin

    db_path = str(tmp_path / "sessions.db")
    await _seed_fts(db_path, fts_contents)  # creates message_fts

    db = await aiosqlite.connect(db_path)
    await init_db(db)

    pm = create_plugin_manager()
    # outcome_log resolves the kernel registry with get_dependency (#1180).
    add_kernel_registry(pm)
    persistence = PersistencePlugin()
    persistence.db = db
    pm.register(persistence, name="persistence")

    # The trace protocol plugin and a listener on it; the specs themselves
    # come from install_plugin_hookspecs below, as in the daemon.
    recorder = _register_trace(pm)
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

    # The kernel's held-text store, which critique hands verdicts to (#1176);
    # it reads the kernel clock and registry when it holds an entry.
    held = None
    if with_held:
        install_kernel_clock(pm)
        held = await add_kernel_held(pm)

    if appraisal is not None:
        pm.register(appraisal, name="appraisal")

    plugin = CritiquePlugin()
    pm.register(plugin, name="critique")
    for extra_name, extra in (extra_plugins or {}).items():
        pm.register(extra, name=extra_name)
    # The daemon installs plugin-declared hookspecs before any on_init.
    install_plugin_hookspecs(pm)
    cfg = config or {}
    cfg.setdefault("daemon", {})["session_db"] = db_path
    await plugin.on_init(pm=pm, config=cfg)
    await plugin.on_start(config=cfg)

    channel = _make_channel()
    fork_agent = None
    if fork:
        # The default fork agent and a window containing the originating rows.
        fork_agent = FakeForkAgent({"critique"}, text=chat_text)
        pm.register(fork_agent, name="agent")
        channel.conversation = window_of(channel.id, [1, 2, 50])
    return SimpleNamespace(
        pm=pm, db=db, db_path=db_path, plugin=plugin, task=task_plugin,
        outcome=outcome, llm=llm, stub=stub, held=held, channel=channel,
        appraisal=appraisal, fork_agent=fork_agent, recorder=recorder,
    )


async def _teardown(env):
    await env.plugin.on_stop()
    await env.db.close()


async def _run_next_task(task_plugin):
    task = task_plugin.task_queue.queue.get_nowait()
    await asyncio.create_task(task.work(), context=task.ctx)
    return task


async def _read_outcomes(recorder, key):
    """The critique records fired for trace ``key``, merged in order with the
    last write per top-level key winning; None when it fired none."""
    records = [r["record"] for r in recorder.by_source("critique") if r["trace_id"] == key]
    if not records:
        return None
    merged: dict = {}
    for record in records:
        merged.update(record)
    return merged


async def _admit(env, trace_id, origin):
    """Admit the exchange the way a user message does; other origins have
    no admission to record."""
    if origin == "user":
        await env.outcome.on_message_admitted(
            channel=env.channel, trace_id=trace_id, sender="u", text="x",
        )


async def _fire(env, trace_id="k", origin="user", response_text="resp",
                originating_text="orig user text", vector=None):
    """Simulates the propagated-origin mechanics (§1.4/§1.5): "user" and
    "task" (a subagent or fork-tool result) carry no origin annotation; any
    other origin is stamped into annotations["origin"].

    ``vector`` (lens name -> share) is the critique vector stage 2 delivers
    for the trace right after the response (#1078): the appraisal listener
    runs with it, as the daemon's stage-2 task would. None: stage 2 never
    lands."""
    await _admit(env, trace_id, origin)
    if env.appraisal is not None:
        env.plugin._originating_texts[trace_id] = originating_text
    # A lens forks only while the exchange's originating row is known and in
    # the window (#320); persist row 1 unless the test already recorded one.
    if env.fork_agent is not None and trace_id not in env.plugin._originating_rowid:
        await env.plugin.on_message_persisted(
            channel=env.channel, trace_id=trace_id, rowid=1, text=originating_text, annotations={})
    if origin in ("user", "task"):
        annotations = {}
    else:
        annotations = {"origin": origin}
    await env.plugin.on_agent_response(
        channel=env.channel, request_text="req", response_text=response_text,
        trace_id=trace_id, annotations=annotations, logprobs=None,
    )
    # Stage 2 lands for the same trace, carrying the critique vector.
    if vector is not None:
        await env.plugin.on_appraisal_idle(
            channel=env.channel, trace_id=trace_id, response_text=response_text,
            annotations=annotations, lenses=vector, past_claims=[],
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
# 2. Eligibility by origin (trap #3)
# ---------------------------------------------------------------------------


class TestEligibility:
    async def test_user_origin_eligible(self, tmp_path):
        env = await build_critique(tmp_path, appraisal=FakeAppraisal())
        await _fire(env, "ku", origin="user", vector={"predictive": 1.0})
        assert env.task.task_queue.queue.qsize() == 1
        await _teardown(env)

    async def test_reminder_origin_eligible(self, tmp_path):
        env = await build_critique(tmp_path, appraisal=FakeAppraisal())
        await _fire(env, "kr", origin="reminder", vector={"predictive": 1.0})
        assert env.task.task_queue.queue.qsize() == 1
        await _teardown(env)

    @pytest.mark.parametrize("origin", ["critique", "heartbeat", "webhook:github", "task"])
    async def test_every_origin_is_judged(self, tmp_path, origin):
        """#1031: no origin is exempt. "task" is a turn with no origin that
        was never admitted (a subagent or fork-tool result landing)."""
        env = await build_critique(tmp_path, appraisal=FakeAppraisal())
        await _fire(env, "ke", origin=origin, vector={"predictive": 1.0})
        assert env.task.task_queue.queue.qsize() == 1
        await _teardown(env)

    async def test_a_turn_without_a_trace_is_not_judged(self, tmp_path):
        appraisal = FakeAppraisal()
        env = await build_critique(tmp_path, appraisal=appraisal)
        await env.plugin.on_agent_response(
            channel=env.channel, request_text="req", response_text="resp",
            trace_id=None, annotations={}, logprobs=None,
        )
        assert env.task.task_queue.queue.qsize() == 0
        await _teardown(env)

    async def test_critique_does_not_infer_user_from_admission(self, tmp_path):
        """#1031: eligibility no longer depends on admission, so critique
        keeps no admitted-trace list and does not subscribe to the hook."""
        env = await build_critique(tmp_path, appraisal=FakeAppraisal())
        impls = [i.plugin_name for i in env.pm.hook.on_message_admitted.get_hookimpls()]
        assert "critique" not in impls
        await _teardown(env)

    async def test_user_exchange_critiqued_against_originating_text(self, tmp_path):
        """A user exchange ending on a tool-result turn is critiqued against
        the ORIGINAL user text (originating_text), not request_text."""
        env = await build_critique(tmp_path, appraisal=FakeAppraisal())
        await _fire(env, "kt", origin="user",
                    originating_text="the ORIGINAL user question",
                    vector={"predictive": 1.0})
        await _run_next_task(env.task)
        # The lens is a fork (#320); its instruction quotes the original text.
        assert env.stub.chat_calls == []
        assert "«the ORIGINAL user question»" in env.fork_agent.calls[0]["instruction"]
        assert env.fork_agent.calls[0]["stage"] == "critique"
        await _teardown(env)


# ---------------------------------------------------------------------------
# 3. Lens selection + below-threshold sampling
# ---------------------------------------------------------------------------


class TestLensSelection:
    """Which lens runs is the critique vector's business (#1078,
    tests/cognition/test_critique_vector.py); these pin what the old
    selection tests also proved and still holds."""

    async def test_retention_prompt_asks_for_file_and_content(self):
        """The objection has to name where the fact goes and what to write."""
        from corvidae_cognition.critique import DEFAULT_FORK_PROMPTS, DEFAULT_PROMPTS

        for prompt in (DEFAULT_PROMPTS["retention"], DEFAULT_FORK_PROMPTS["retention"]):
            assert '"file": "..."' in prompt
            assert '"content": "..."' in prompt

    async def test_retention_fork_names_the_knowledge_base_root(self, tmp_path):
        """The lens can only name a file if it knows the directory."""
        env = await build_critique(
            tmp_path, appraisal=FakeAppraisal(), chat_text=EMPTY_VERDICT,
            config={"critique": {"retention": {"root": "values/"}}},
        )
        await _fire(env, "kroot", origin="user", vector={"retention": 1.0})
        await _run_next_task(env.task)
        instructions = [c["instruction"] for c in env.fork_agent.calls]
        assert any("values/" in text for text in instructions), instructions
        # The root is named, not implied: the model has to name a path under it.
        assert any("knowledge base root is" in text for text in instructions)
        await _teardown(env)

    async def test_retention_root_not_indexed_by_recall_warns(self, tmp_path, caplog):
        """A root recall does not index is a silent data-loss path, so say it."""
        appraisal = FakeAppraisal()
        env = await build_critique(
            tmp_path, appraisal=appraisal,
            config={
                "critique": {"retention": {"root": "notes/"}},
                "markdown_recall": {"roots": ["values/"]},
            },
        )
        warnings = [r.getMessage() for r in caplog.records
                    if r.levelname == "WARNING" and "critique.retention.root" in r.getMessage()]
        assert warnings, "expected a WARNING naming the misconfigured root"
        assert "notes" in warnings[0] and "values" in warnings[0]
        await _teardown(env)

    async def test_retention_root_matching_recall_is_quiet(self, tmp_path, caplog):
        """No warning when the two agree, or when recall is not configured."""
        appraisal = FakeAppraisal()
        env = await build_critique(
            tmp_path, appraisal=appraisal,
            config={
                "critique": {"retention": {"root": "values/"}},
                "markdown_recall": {"roots": ["values/"]},
            },
        )
        assert not [r for r in caplog.records
                    if r.levelname == "WARNING" and "critique.retention.root" in r.getMessage()]
        await _teardown(env)
        caplog.clear()
        # No recall config at all: nothing to drift from, so quiet.
        other = tmp_path / "norecall"
        other.mkdir()
        env = await build_critique(
            other, appraisal=appraisal,
            config={"critique": {"retention": {"root": "nobody-indexes-this/"}}},
        )
        assert not [r for r in caplog.records
                    if r.levelname == "WARNING" and "critique.retention.root" in r.getMessage()]
        await _teardown(env)

    async def test_none_winning_queues_no_task_when_not_sampled(self, tmp_path):
        env = await build_critique(tmp_path, appraisal=FakeAppraisal())
        env.plugin._rng = FixedRng(value=0.99)  # never below the 0.05 rate
        await _fire(env, "kb", origin="user", response_text="no claims here",
                    vector={"none": 1.0})
        assert env.task.task_queue.queue.qsize() == 0
        await _teardown(env)

    async def test_none_winning_sampled_marks_outcome(self, tmp_path):
        env = await build_critique(tmp_path, appraisal=FakeAppraisal(),
                                   chat_text=EMPTY_VERDICT)
        env.plugin._rng = FixedRng(value=0.01, choice_value="predictive")
        await _fire(env, "ksamp", origin="user", response_text="no claims here",
                    vector={"none": 1.0})
        assert env.task.task_queue.queue.qsize() == 1
        await _run_next_task(env.task)
        outcomes = await _read_outcomes(env.recorder, "ksamp")
        assert outcomes["critique"]["sampled_below_threshold"] is True
        await _teardown(env)

    async def test_no_appraisal_plugin_critiques_everything(self, tmp_path):
        """With no appraisal plugin a lens runs on the exchanges the fallback
        rate samples (here every one, critique.fallback_rate 1.0; the share
        itself is tested in test_critique_hooks.py)."""
        env = await build_critique(tmp_path, appraisal=None,
                                   chat_text=EMPTY_VERDICT,
                                   config={"critique": {"fallback_rate": 1.0}})
        env.plugin._rng = FixedRng(value=0.5)
        await _fire(env, "kna", origin="user", response_text="anything")
        assert env.task.task_queue.queue.qsize() == 1
        await _teardown(env)

    async def test_sample_rate_override_takes_effect_without_reinit(self, tmp_path):
        """Lowering critique.sample_below_rate via runtime_overrides stops the
        sampled lens on the next exchange with no re-init (directive 2)."""
        env = await build_critique(tmp_path, appraisal=FakeAppraisal(),
                                   chat_text=EMPTY_VERDICT)
        env.plugin._rng = FixedRng(value=0.01, choice_value="predictive")
        # "none" wins and the draw is below the default rate: a lens is sampled.
        await _fire(env, "kd1", origin="user", vector={"none": 1.0})
        assert env.task.task_queue.queue.qsize() == 1
        await _run_next_task(env.task)
        # Set the rate to 0: the same draw no longer samples.
        env.channel.runtime_overrides["critique.sample_below_rate"] = 0.0
        await _fire(env, "kd2", origin="user", response_text="no claim",
                    vector={"none": 1.0})
        assert env.task.task_queue.queue.qsize() == 0
        await _teardown(env)


# ---------------------------------------------------------------------------
# 5. Silent execution: empty vs non-empty verdict
# ---------------------------------------------------------------------------


class TestExecution:
    async def test_empty_verdict_no_notify_outcome_recorded(self, tmp_path):
        env = await build_critique(tmp_path, appraisal=FakeAppraisal(),
                                   chat_text=EMPTY_VERDICT, with_held=True)
        env.pm.ahook_raise.on_notify = AsyncMock()
        await _fire(env, "kempty", origin="user", vector={"predictive": 1.0})
        await _run_next_task(env.task)
        env.pm.ahook_raise.on_notify.assert_not_awaited()
        outcomes = await _read_outcomes(env.recorder, "kempty")
        assert outcomes["critique"]["objections"] == 0
        # The lens fork carried the exchange attribution.
        assert env.stub.chat_calls == []
        attr = env.fork_agent.calls[0]["attribution"]
        assert attr.get("stage") == "critique"
        assert attr.get("trace_id") == "kempty"
        await _teardown(env)

    async def test_stylistic_objection_registers_without_waking(self, tmp_path):
        env = await build_critique(tmp_path, appraisal=FakeAppraisal(),
                                   chat_text=OBJECTION_VERDICT, with_held=True)
        notify = AsyncMock()
        env.pm.ahook_raise.on_notify = notify
        # Spy on both store entry points; hold keeps its real behaviour.
        ingest = AsyncMock(wraps=env.held.hold)
        env.held.hold = ingest
        env.held.hold_and_wake = AsyncMock()
        await _fire(env, "kfull", origin="user", vector={"predictive": 1.0})
        await _run_next_task(env.task)
        outcomes = await _read_outcomes(env.recorder, "kfull")
        assert outcomes["critique"]["objections"] == 1
        # Registered without a stub: no wake of any kind.
        ingest.assert_called_once()
        assert "no record says Friday" in "\n".join(ingest.call_args.kwargs["entries"])
        env.held.hold_and_wake.assert_not_awaited()
        notify.assert_not_awaited()
        # A turn with no origin (a user turn) admits the verdict as framed CONTEXT.
        await env.held.admit_for_turn(
            env.channel, env.channel.conversation, trace_id="kfull", origin=None,
            contributions=[],
        )
        context_text = "\n".join(
            m.get("content") or ""
            for m in env.channel.conversation.messages
            if m.get("_message_type") == MessageType.CONTEXT
        )
        assert "no record says Friday" in context_text
        await _teardown(env)


RETENTION_VERDICT = json.dumps({"objections": [
    {"claim": "Yoz runs ARDC tech", "objection": "not filed",
     "file": "people/yoz.md", "content": "- runs ARDC tech", "severity": 0.5},
]})


class TestRetentionDelivery:
    """#742 via #857: a retention finding reaches the agent with the file and
    the content, and wakes the channel so the fact is filed now."""

    async def _run_retention(self, tmp_path, key, origin="user"):
        env = await build_critique(
            tmp_path, appraisal=FakeAppraisal(), chat_text=RETENTION_VERDICT,
            with_held=True,
            config={"critique": {"retention": {"root": "values/"}}},
        )
        env.pm.ahook_raise.on_notify = AsyncMock()
        env.held.hold = AsyncMock(wraps=env.held.hold)
        env.held.hold_and_wake = AsyncMock()
        await _fire(env, key, origin=origin, vector={"retention": 1.0})
        await _run_next_task(env.task)
        return env

    async def test_the_file_and_the_content_reach_the_agent(self, tmp_path):
        """AC1: the held entry, not just the outcome record, carries both."""
        env = await self._run_retention(tmp_path, "kdeliver")
        calls = env.held.hold_and_wake.await_args_list + env.held.hold.call_args_list
        entries = "\n".join(e for c in calls for e in c.kwargs["entries"])
        assert "people/yoz.md" in entries, entries
        assert "- runs ARDC tech" in entries, entries
        await _teardown(env)

    async def test_a_retention_finding_wakes_the_channel_to_file_it(self, tmp_path):
        """AC2a: an action turn, silent by default, that reads before it appends."""
        env = await self._run_retention(tmp_path, "kwake")
        env.held.hold_and_wake.assert_awaited_once()
        kwargs = env.held.hold_and_wake.await_args.kwargs
        # The NO_REPLY veto keys on this origin: anything else would post it.
        assert kwargs["origin"] == "critique"
        stub = kwargs["stub_text"]
        assert "NO_REPLY" in stub, stub
        assert "read" in stub.lower() and "append" in stub.lower(), stub
        assert "values/" in stub, "the filing turn must not guess the root"
        assert all("[retention]" in e for e in kwargs["entries"]), kwargs["entries"]
        await _teardown(env)

    async def test_with_a_root_a_stylistic_finding_still_waits(self, tmp_path):
        """AC2c (ported for #1078): a knowledge-base root makes retention a
        waking lens, not every built-in. A constrained finding on a channel
        with a root goes through ingest and never wakes."""
        env = await build_critique(
            tmp_path, appraisal=FakeAppraisal(), chat_text=OBJECTION_VERDICT,
            with_held=True,
            config={"critique": {"retention": {"root": "values/"}}},
        )
        env.pm.ahook_raise.on_notify = AsyncMock()
        env.held.hold = AsyncMock(wraps=env.held.hold)
        env.held.hold_and_wake = AsyncMock()
        # The vector picks constrained; its objection must wait.
        await _fire(env, "kmixed", vector={"constrained": 1.0})
        await _run_next_task(env.task)
        env.held.hold.assert_called_once()
        entries = env.held.hold.call_args.kwargs["entries"]
        assert entries and all("[constrained]" in e for e in entries), entries
        env.held.hold_and_wake.assert_not_awaited()
        await _teardown(env)


class TestRetentionRootFromRecall:
    """Owner, on #857: the retention root matches recall's root when the
    retention key is unset, so facts are filed where recall reads them."""

    async def _env(self, tmp_path, recall, chat_text=EMPTY_VERDICT, with_held=False):
        return await build_critique(
            tmp_path, appraisal=FakeAppraisal(), chat_text=chat_text,
            with_held=with_held, config={"markdown_recall": recall},
        )

    async def test_recalls_root_turns_the_lens_on_and_is_named(self, tmp_path):
        """AC9a: the lens runs, and the fork names recall's root."""
        env = await self._env(tmp_path, {"root": "values/"})
        await _fire(env, "krec", origin="user", vector={"retention": 1.0})
        await _run_next_task(env.task)
        outcomes = await _read_outcomes(env.recorder, "krec")
        assert "retention" in outcomes["critique"]["lenses"]
        instructions = [c["instruction"] for c in env.fork_agent.calls]
        assert any("knowledge base root is «values/»" in t for t in instructions), instructions
        await _teardown(env)

    async def test_the_lens_fork_writes_a_fork_log_row(self, tmp_path):
        """#1204: the lens point is recorded, not just the provenance one."""
        env = await self._env(tmp_path, {"root": "values/"})
        await _fire(env, "klens", origin="user", vector={"retention": 1.0})
        await _run_next_task(env.task)
        async with env.db.execute(
            "SELECT stage, point, instruction, lens_id FROM fork_log"
        ) as cur:
            rows = await cur.fetchall()
        assert any(
            stage == "critique" and point == "lens" and "knowledge base root" in instr
            and lens_id == "retention"
            for stage, point, instr, lens_id in rows
        ), rows
        await _teardown(env)

    async def test_two_lenses_in_one_fork_are_named_in_lens_id(self, tmp_path):
        """#1226 shape: one fork carries every lens; the row names them all."""
        env = await self._env(tmp_path, {"root": "values/"})
        await _fire(
            env, "ktwo", origin="user",
            vector={"retention": 1.0, "constrained": 1.0},
        )
        await _run_next_task(env.task)
        async with env.db.execute(
            "SELECT lens_id FROM fork_log WHERE point = 'lens'"
        ) as cur:
            ids = [r[0] for r in await cur.fetchall()]
        assert ids, "no lens row"
        assert all(set(i.split("+")) >= {"retention", "constrained"} for i in ids), ids
        assert len(env.fork_agent.calls) == 1, "one fork, not one per lens"
        await _teardown(env)

    async def test_the_wake_stub_names_recalls_root(self, tmp_path):
        """AC9a: the filing turn is told the same root."""
        env = await self._env(
            tmp_path, {"root": "values/"}, chat_text=RETENTION_VERDICT, with_held=True,
        )
        env.held.hold = AsyncMock(wraps=env.held.hold)
        env.held.hold_and_wake = AsyncMock()
        await _fire(env, "krecw", origin="user", vector={"retention": 1.0})
        await _run_next_task(env.task)
        env.held.hold_and_wake.assert_awaited_once()
        assert "values/" in env.held.hold_and_wake.await_args.kwargs["stub_text"]
        await _teardown(env)

    async def test_several_recall_roots_use_the_first(self, tmp_path):
        """AC9b: the first listed root is the one the lens files into."""
        env = await self._env(tmp_path, {"roots": ["a/", "b/"]})
        await _fire(env, "kroots", origin="user", vector={"retention": 1.0})
        await _run_next_task(env.task)
        instructions = [c["instruction"] for c in env.fork_agent.calls]
        assert any("knowledge base root is «a/»" in t for t in instructions), instructions
        await _teardown(env)

    async def test_a_derived_root_does_not_warn(self, tmp_path, caplog):
        """AC9e: nothing has drifted when the root came from recall."""
        env = await self._env(tmp_path, {"roots": ["a/", "b/"]})
        assert not [r for r in caplog.records
                    if r.levelname == "WARNING" and "critique.retention.root" in r.getMessage()]
        await _teardown(env)


def test_objection_schema_allows_file_and_content():
    """AC3: a server that honours the schema can return the retention fields."""
    from corvidae_cognition.critique import OBJECTION_SCHEMA

    item = OBJECTION_SCHEMA["properties"]["objections"]["items"]
    assert item["properties"]["file"] == {"type": "string"}
    assert item["properties"]["content"] == {"type": "string"}
    assert item["required"] == ["claim", "objection"]


async def test_retention_root_warns_against_recalls_default_root(tmp_path, caplog):
    """AC6: a markdown_recall block with neither roots nor root indexes
    state/markdown_recall, so a different retention root is drift too."""
    appraisal = FakeAppraisal()
    env = await build_critique(
        tmp_path, appraisal=appraisal,
        config={"critique": {"retention": {"root": "values/"}}, "markdown_recall": {}},
    )
    warnings = [r.getMessage() for r in caplog.records
                if r.levelname == "WARNING" and "critique.retention.root" in r.getMessage()]
    assert len(warnings) == 1, warnings
    assert "values" in warnings[0] and "state/markdown_recall" in warnings[0], warnings[0]
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


async def _provenance_env(tmp_path, chat_text=OBJECTION_VERDICT, with_held=False):
    appraisal = FakeAppraisal()
    return await build_critique(
        tmp_path, appraisal=appraisal, chat_text=chat_text, with_held=with_held,
        config={"critique": {"sample_below_rate": 0.0}}, fork=False,
    )


async def _begin_exchange(env, trace_id, origin_rowid, origin="user"):
    """Record the exchange and persist its originating row."""
    await _admit(env, trace_id, origin)
    await env.plugin.on_message_persisted(
        channel=env.channel, trace_id=trace_id, rowid=origin_rowid,
        text="when is it due?", annotations={},
    )


async def _stage2_hook(env, *, channel, trace_id, response_text, annotations, stage2):
    """Deliver a stage-2 dict (None: a miss) as appraisal's on_appraisal_idle
    does: its lens vector and past claims as the hook's two arguments."""
    await env.plugin.on_appraisal_idle(
        channel=channel, trace_id=trace_id, response_text=response_text,
        annotations=annotations,
        lenses=None if stage2 is None else stage2.get("lenses"),
        past_claims=None if stage2 is None else stage2.get("past_claims", []),
    )


async def _stage2(env, trace_id, claims, annotations=None, response_text=CUE_FREE_REPLY):
    """Deliver a stage-2 result the way AppraisalPlugin's on_appraisal_idle does."""
    await env.plugin.on_appraisal_idle(
        channel=env.channel, trace_id=trace_id, response_text=response_text,
        annotations=annotations or {}, lenses=None, past_claims=claims,
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

    async def fork(self, channel, instruction, *, stage, cut=None, extra_body=None,
                   **kwargs):
        from corvidae.fork import build_fork_prompt

        if channel.conversation is not None:
            self.requests.append(build_fork_prompt(
                channel.conversation, cut, instruction, frozenset(), "system"))
        # Whatever else the caller passes (max_tool_calls, yield_to_caller) goes on.
        return await super().fork(
            channel, instruction, stage=stage, cut=cut, extra_body=extra_body,
            **kwargs)


def _row(role, content, rowid, **extra):
    msg = {"role": role, "content": content, "_message_type": MessageType.MESSAGE, **extra}
    if rowid is not None:
        msg["_db_id"] = rowid
    return msg


async def _provenance_exchange(env, agent, trace_id="kp", reply_text=CUE_FREE_REPLY,
                               support=None, persist_reply=True, annotations=None):
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
        trace_id=trace_id, annotations=annotations or {}, logprobs=None,
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
    async def test_no_appraisal_warns_that_provenance_is_off(self, tmp_path, caplog):
        with caplog.at_level("INFO", logger="corvidae.critique"):
            env = await build_critique(tmp_path)
        try:
            assert "provenance check is off" in caplog.text
            assert "appraisal" in caplog.text
        finally:
            await _teardown(env)

    async def test_on_agent_response_never_selects_provenance(self, tmp_path):
        # Provenance waits for stage-2's claims; the reply's wording alone
        # triggers nothing.
        env = await build_critique(tmp_path, appraisal=FakeAppraisal(), chat_text=EMPTY_VERDICT)
        try:
            # With an originating rowid and an empty message_fts, the old
            # wording-triggered gate would have added "provenance" here.
            await env.plugin.on_message_persisted(
                channel=env.channel, trace_id="kst", rowid=50, text="q?", annotations={},
            )
            await _fire(env, "kst", response_text=CLAIM_REPLY, vector={"predictive": 1.0})
            await _run_next_task(env.task)
            outcomes = await _read_outcomes(env.recorder, "kst")
            assert outcomes["critique"]["lenses"] == ["predictive"]
            # Stage 2 listed no claims, so no provenance task was queued.
            assert env.task.task_queue.queue.empty()
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

    @pytest.mark.parametrize("origin", ["critique", "heartbeat", "webhook:github", "task"])
    async def test_every_origin_is_checked(self, tmp_path, origin):
        """#1031: no origin is exempt from the provenance check."""
        env = await _provenance_env(tmp_path)
        try:
            await _begin_exchange(env, "kx", origin_rowid=50, origin=origin)
            annotations = {} if origin == "task" else {"origin": origin}
            await _stage2(env, "kx", [CLAIM], annotations=annotations)
            assert env.task.task_queue.queue.qsize() == 1
        finally:
            await _teardown(env)

    async def test_reminder_origin_eligible(self, tmp_path):
        env = await _provenance_env(tmp_path)
        try:
            await _begin_exchange(env, "kr", origin_rowid=50, origin="reminder")
            await _stage2(env, "kr", [CLAIM], annotations={"origin": "reminder"})
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


async def _use_real_message_fts(env) -> None:
    """Replace the test's plain message_fts with the DDL the memory plugin
    creates in production (the tokenizer under test), before any row is logged."""
    from corvidae_cognition.memory import MESSAGE_FTS_DDL

    await env.db.execute("DROP TABLE message_fts")
    await env.db.execute(MESSAGE_FTS_DDL)
    await env.db.commit()


class TestProvenanceUnspacedText:
    """#1186 R2: critique's provenance search finds unspaced text through
    the shared keyword query (pieces, or LIKE for a 2-character term)."""

    @pytest.mark.parametrize("query", ["我也喜欢吃", "苹果"], ids=["containing", "short"])
    async def test_unspaced_claim_finds_the_earlier_message(self, tmp_path, query):
        env = await _provenance_env(tmp_path)
        try:
            await _use_real_message_fts(env)
            earlier = await _log(env, "我喜欢吃苹果")
            await _log(env, "kestrel rain gauge calibration")
            origin = await _log(env, "你喜欢什么？")
            await _begin_exchange(env, "kj", origin_rowid=origin)
            claim = {"claim": "你说过你喜欢吃苹果", "queries": [query]}
            evidence = await env.plugin._gather_evidence(env.channel, "kj", [claim])
            assert _candidate_rowids(evidence) == [[earlier]]
        finally:
            await _teardown(env)

    async def test_non_string_query_entries_are_skipped_not_fatal(self, tmp_path):
        """A model-written `queries` list can hold a non-string; the claim
        is still searched by its string entries (no TypeError abandons the
        provenance check)."""
        env = await _provenance_env(tmp_path)
        try:
            await _use_real_message_fts(env)
            earlier = await _log(env, "the quarterly audit deadline is Thursday")
            origin = await _log(env, "when is it due?")
            await _begin_exchange(env, "kn", origin_rowid=origin)
            claim = {"claim": "deadline", "queries": [None, 42, "quarterly audit deadline"]}
            evidence = await env.plugin._gather_evidence(env.channel, "kn", [claim])
            assert evidence is not None
            assert _candidate_rowids(evidence) == [[earlier]]
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

    @pytest.mark.parametrize(("value_sql", "params"), CORRUPT_MESSAGES)
    async def test_corrupt_row_is_skipped_with_a_warning(
        self, tmp_path, caplog, value_sql, params
    ):
        # A corrupt matching row (#260) is skipped, not fatal: the valid
        # candidate survives and the warning names the row and channel.
        env = await _provenance_env(tmp_path)
        try:
            text = "the quarterly audit deadline is Thursday"
            good = await _log(env, text)
            bad = await insert_raw_message(
                env.db, env.channel.id, value_sql, params, drop_fts_trigger=True)
            await env.db.execute(
                "INSERT INTO message_fts(rowid, content_text) VALUES (?, ?)", (bad, text))
            await env.db.commit()
            origin = await _log(env, "when is it due?")
            await _begin_exchange(env, "kc", origin_rowid=origin)
            with caplog.at_level("WARNING"):
                evidence = await env.plugin._gather_evidence(env.channel, "kc", [CLAIM])
            assert evidence is not None
            assert _candidate_rowids(evidence) == [[good]]
            assert skip_warnings(caplog, bad, env.channel.id)
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

    async def test_searches_every_channel(self, tmp_path):
        # Evidence is raw dialog on every channel (PRD §3.7 Critique 2), with
        # no memory plugin or scope needed to say which.
        env = await _provenance_env(tmp_path)
        try:
            other = await _log(env, "the quarterly audit deadline is Thursday",
                               channel_id="irc:#elsewhere")
            signal = await _log(env, "the quarterly audit deadline is Thursday",
                                channel_id="signal:alice")
            own = await _log(env, "the quarterly audit deadline is Thursday")
            origin = await _log(env, "when is it due?")
            await _begin_exchange(env, "ks", origin_rowid=origin)
            evidence = await env.plugin._gather_evidence(env.channel, "ks", [CLAIM])
            assert sorted(_candidate_rowids(evidence)[0]) == sorted([other, signal, own])
            channels = {c["channel_id"] for c in evidence[0]["candidates"]}
            assert channels == {"irc:#elsewhere", "signal:alice", env.channel.id}
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

    async def test_fork_sees_claims_and_evidence_not_window_context(self, tmp_path):
        # The window is already in the fork's prefix, so the instruction
        # carries the claim and the search evidence and no copy of the
        # window's CONTEXT rows (#1001).
        env = await _provenance_env(tmp_path)
        agent = RenderingForkAgent({"critique"}, text=UNSUPPORTED_VERDICT)
        try:
            ex = await _provenance_exchange(
                env, agent, support="Heads up: the quarterly audit deadline is Thursday.")
            # A CONTEXT row with text found nowhere else in the exchange.
            env.channel.conversation.messages.insert(0, {
                "role": "system", "content": "memory: zanzibar ledger is audited quarterly",
                "_message_type": MessageType.CONTEXT,
            })
            await env.plugin.before_agent_turn(env.channel, "kp", {})
            await _run_provenance(env)
            [call] = agent.calls
            instruction = call["instruction"]
            assert instruction.startswith(_provenance_fork_prompt())
            assert "confabulated memory" in instruction
            assert CLAIM["claim"] in instruction
            assert "Heads up: the quarterly audit deadline is Thursday." in instruction
            assert "zanzibar ledger" not in instruction
            assert "snapshot" not in instruction.lower()
            assert call["stage"] == "critique"
            assert call["attribution"]["stage"] == "critique"
            assert env.stub.chat_calls == []
            outcomes = await _read_outcomes(env.recorder, "kp")
            assert outcomes["provenance"] == {
                "claims": 1, "candidates": 1, "supported": 0, "objections": 1,
                "detail": [{"claim": "the quarterly audit deadline is Thursday",
                            "objection": "reason 1", "lens": "provenance"}]}
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

    async def test_objections_reach_the_held_store(self, tmp_path):
        from corvidae_cognition.critique import PROVENANCE_NOTE

        env = await _provenance_env(tmp_path, with_held=True)
        env.held.hold_and_wake = AsyncMock()
        agent = RenderingForkAgent({"critique"}, text=UNSUPPORTED_VERDICT)
        try:
            await _provenance_exchange(env, agent, trace_id="kf")
            await _run_provenance(env, "kf")
            env.held.hold_and_wake.assert_awaited_once()
            kwargs = env.held.hold_and_wake.await_args.kwargs
            assert kwargs["origin"] == "critique"
            assert "[provenance]" in kwargs["entries"][0]
            # Other waking lenses' sentences (proactivity's, #1126) follow it.
            assert kwargs["stub_text"].startswith(PROVENANCE_NOTE)
        finally:
            await _teardown(env)

    async def test_with_a_retention_root_the_provenance_stub_also_files(self, tmp_path):
        """AC2d: one stub governs a channel's critique drain, whichever wake
        fires first, so with the lens on it must cover retention filing too."""
        from corvidae_cognition.critique import PROVENANCE_NOTE

        env = await _provenance_env(tmp_path, with_held=True)
        env.plugin.config.setdefault("critique", {})["retention"] = {"root": "values/"}
        env.held.hold_and_wake = AsyncMock()
        agent = RenderingForkAgent({"critique"}, text=UNSUPPORTED_VERDICT)
        try:
            await _provenance_exchange(env, agent, trace_id="kfr")
            await _run_provenance(env, "kfr")
            stub = env.held.hold_and_wake.await_args.kwargs["stub_text"]
            assert stub != PROVENANCE_NOTE
            assert "values/" in stub and "append" in stub.lower(), stub
            assert "NO_REPLY" in stub
        finally:
            await _teardown(env)

    async def test_failed_search_never_reaches_the_critic_as_no_match(
        self, tmp_path, caplog, monkeypatch
    ):
        # A search that raised looked at nothing: telling the critic "no
        # earlier message matched" would turn a supported claim into an
        # objection. #1186: a lone surrogate no longer raises (it is not a
        # search term), so the failure is injected at the second claim's
        # search, the seam where sqlite's own error would surface.
        import sqlite3

        from corvidae_cognition import critique as critique_module

        real_where = critique_module.keyword_where
        calls = {"n": 0}

        def failing_where(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 2:
                raise sqlite3.OperationalError("injected search failure")
            return real_where(*args, **kwargs)

        monkeypatch.setattr(critique_module, "keyword_where", failing_where)
        env = await _provenance_env(tmp_path)
        agent = RenderingForkAgent({"critique"}, text=UNSUPPORTED_VERDICT)
        try:
            await _provenance_exchange(
                env, agent, trace_id="kfail",
                support="Heads up: the quarterly audit deadline is Thursday.")
            claims = [CLAIM, {"claim": "we agreed on Tuesday", "queries": ["Tuesday"]}]
            with caplog.at_level("WARNING", logger="corvidae.critique"):
                calls["n"] = 0
                assert await env.plugin._gather_evidence(env.channel, "kfail", claims) is None
                calls["n"] = 0
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


def _stage2_reply(past_claims):
    return json.dumps({"past_claims": past_claims})


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
    recorder = _register_trace(pm)
    outcome = OutcomeLogPlugin()
    pm.register(outcome, name="outcome_log")
    await outcome.on_init(pm=pm, config=cfg)
    await outcome.on_start(config=cfg)

    llm = LLMPlugin()
    pm.register(llm, name="llm")

    plugins = []
    for name, cls in (("appraisal", AppraisalPlugin), ("critique", CritiquePlugin)):
        p = cls()
        pm.register(p, name=name)
        plugins.append(p)
    # The daemon installs plugin-declared hookspecs (critique's among them)
    # after registering and before any on_init.
    install_plugin_hookspecs(pm)
    for p in plugins:
        await p.on_init(pm=pm, config=cfg)
    critique = plugins[1]
    # Appraisal has no on_start of its own (#1189).
    await critique.on_start(config=cfg)
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

    agent._forks = ForksConfig(enabled=True, stages=frozenset({"appraisal", "critique"}))
    forks: list[dict] = []

    async def fork(ch, instruction, *, stage, cut=None, extra_body=None, **_kw):
        forks.append({
            "instruction": instruction, "cut": cut,
            "request": build_fork_prompt(
                ch.conversation, cut, instruction, frozenset(), "system"),
        })
        # Stage 2 (a fork too, #320) answers with the claims; critique with
        # an unsupported verdict.
        from corvidae_cognition.appraisal import DEFAULT_STAGE2_FORK_PROMPT

        text = (_stage2_reply(past_claims) if instruction.startswith(DEFAULT_STAGE2_FORK_PROMPT)
                else UNSUPPORTED_VERDICT)
        return ForkResult(text=text, message={"role": "assistant"})

    agent.fork = fork

    client = MagicMock()
    client.chat = AsyncMock(side_effect=[_make_text_response(r) for r in replies])
    agent._client = client
    agent._mark_started()
    return SimpleNamespace(
        agent=agent, channel=channel, db=db, gathered=gathered,
        plugins=plugins, forks=forks, recorder=recorder,
    )


async def _real_path_teardown(env):
    for p in reversed(env.plugins):
        # Appraisal has nothing to stop since stage 1 went (#1185).
        if hasattr(p, "on_stop"):
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
            outcomes = await _read_outcomes(env.recorder, trace_id)
            assert outcomes["provenance"]["objections"] == 1
        finally:
            await _real_path_teardown(env)

    async def test_the_provenance_fork_writes_a_fork_log_row(self, tmp_path):
        """#1204: critique's capture lands through the real fork path."""
        env = await _real_path_env(tmp_path, [CUE_FREE_REPLY], [CLAIM])
        try:
            await _say(env, "when is it due?")
            async with env.db.execute(
                "SELECT stage, point, instruction FROM fork_log"
            ) as cur:
                rows = await cur.fetchall()
            points = {(r[0], r[1]) for r in rows}
            assert ("critique", "provenance") in points
            assert ("appraisal", "after-send") in points
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
            # The fork ends at the reply of the exchange it audits.
            assert provenance[-1]["request"][-2]["content"] == CLAIM_REPLY
        finally:
            await _real_path_teardown(env)

    async def test_reply_without_past_claims_runs_no_provenance_fork(self, tmp_path):
        env = await _real_path_env(tmp_path, ["It is due Thursday."], [])
        try:
            await _say(env, "when is it due?")
            assert _real_provenance_forks(env) == []
        finally:
            await _real_path_teardown(env)


class TestProvenancePrompts:
    """The critic must not take the agent's word for a fact: its own
    earlier statement shows only that it said so (PR #156 follow-up)."""

    def test_own_earlier_statement_is_not_a_record_of_a_fact(self):
        prompt = _provenance_fork_prompt()
        assert "shows only that it was said" in prompt
        assert "a user message, a tool result, a summary or retrieved CONTEXT" in prompt

    def test_summary_rows_are_labelled_summary_not_assistant(self):
        # Compaction stores a summary as an assistant turn; shown to the
        # critic under that role it would read as the agent's own word.
        from corvidae_cognition.critique import CritiquePlugin, _candidate

        row = (9, "test:x", {
            "role": "assistant", "content": "[Summary of earlier conversation]\naudit Thursday",
        }, "summary", 1.0)
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
            text=text, emission=emission, trace_id="kveto", annotations={"origin": origin},
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


async def _create_late_db(path, channel_id, content):
    """Create the session DB at path with message_log, message_fts, and one
    earlier row indexed under the same rowid (as MemoryPlugin's trigger would)."""
    async with aiosqlite.connect(path) as late:
        await init_db(late)
        cursor = await late.execute(
            "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
            "VALUES (?, ?, ?, ?)",
            (channel_id, json.dumps({"role": "user", "content": content}),
             1.0, "message"),
        )
        await late.execute("CREATE VIRTUAL TABLE message_fts USING fts5(content_text)")
        await late.execute(
            "INSERT INTO message_fts(rowid, content_text) VALUES (?, ?)",
            (cursor.lastrowid, content),
        )
        await late.commit()


class TestLazyProbe:
    async def test_provenance_starts_working_when_table_appears(self, tmp_path, caplog):
        """#159 R1/R5: the plugin's own probe, built by on_start against a
        DB file that doesn't exist yet, picks it up with no restart."""
        from corvidae_cognition.probe import PROBE_RETRY_SECONDS

        env = await _provenance_env(tmp_path)
        try:
            # Restart the plugin's probe against a path with no file at all.
            await env.plugin._probe.close()
            late = str(tmp_path / "late.db")
            cfg = {"daemon": {"session_db": late}}
            await env.plugin.on_start(config=cfg)
            now = [0.0]
            env.plugin._probe.clock = lambda: now[0]

            await _begin_exchange(env, "klz", origin_rowid=50)
            with caplog.at_level("WARNING", logger="corvidae"):
                assert await env.plugin._gather_evidence(
                    env.channel, "klz", [CLAIM]) is None
            # The probe warning names the path; the per-trace skip stays.
            assert late in caplog.text
            assert "provenance check skipped for trace klz" in caplog.text

            # The file appears with message_fts and a matching row; the
            # retry interval elapses.
            await _create_late_db(
                late, env.channel.id, "the quarterly audit deadline is Thursday")
            now[0] = PROBE_RETRY_SECONDS + 1
            evidence = await env.plugin._gather_evidence(env.channel, "klz", [CLAIM])
            assert evidence is not None
            assert evidence[0]["candidates"]
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


# ---------------------------------------------------------------------------
# #398 R13: a coalesced burst's own messages are never earlier evidence
# ---------------------------------------------------------------------------


class TestBurstProvenanceBoundary:
    """R13 (#398): the provenance search excludes every row of the burst."""

    async def test_claim_supported_only_by_an_earlier_burst_message_is_not_found(
        self, tmp_path
    ):
        """R13 (#398): the burst's last message owns the trace, but a claim
        supported only by an earlier message of the same burst is not counted
        as found in prior history; a message from before the burst still is."""
        env = await _provenance_env(tmp_path)
        try:
            # History from before the burst, then a three-message burst.
            earlier = await _log(env, "Ana said the quarterly audit deadline is Thursday.")
            m1 = await _log(env, "Heads up: the quarterly audit deadline is Thursday.")
            m2 = await _log(env, "unrelated chatter")
            m3 = await _log(env, "when is it due?")
            # The window holds the burst's three unanswered rows; the trace is
            # the last message's, so its recorded originating row is m3.
            conv = ContextWindow(env.channel.id)
            conv.messages = [
                _row("user", "Heads up: the quarterly audit deadline is Thursday.", m1),
                _row("user", "unrelated chatter", m2),
                _row("user", "when is it due?", m3),
            ]
            env.channel.conversation = conv
            await _begin_exchange(env, "kb", origin_rowid=m3)

            await env.plugin.before_agent_turn(env.channel, "kb", {})
            evidence = await env.plugin._gather_evidence(env.channel, "kb", [CLAIM])

            # Only the pre-burst message is evidence; m1 (same burst) is not.
            assert _candidate_rowids(evidence) == [[earlier]]
        finally:
            await _teardown(env)


class TestRetentionWakeNeedsARoot:
    """#742 via #857: a retention finding wakes only while there is a root.

    The lens cannot run without one, but the wake rule is written where it is
    read, so the pairing is pinned here rather than left to the lens's
    selection: a wake whose stub says "stay silent and file nothing" spends a
    turn to do nothing.
    """

    @staticmethod
    async def _register(env, *, root):
        if root is not None:
            env.plugin.config.setdefault("critique", {})["retention"] = {"root": root}
        env.held.hold = AsyncMock()
        env.held.hold_and_wake = AsyncMock()
        objection = {
            "claim": "Yoz runs ARDC tech", "objection": "not filed", "lens": "retention",
            "file": "people/yoz.md", "content": "- runs ARDC tech",
        }
        await env.plugin._record_and_register(
            env.channel, "kwr", {"critique": {"lenses": ["retention"], "objections": 1}},
            [objection], wake=False, origin=None, reply_ref="",
        )

    async def test_with_a_root_the_finding_wakes_the_channel(self, tmp_path):
        appraisal = FakeAppraisal()
        env = await build_critique(tmp_path, appraisal=appraisal, with_held=True)
        try:
            await self._register(env, root="values/")
            env.held.hold_and_wake.assert_awaited_once()
            assert "values/" in env.held.hold_and_wake.await_args.kwargs["stub_text"]
            env.held.hold.assert_not_called()
        finally:
            await _teardown(env)

    async def test_without_a_root_the_finding_waits_for_the_next_turn(self, tmp_path):
        appraisal = FakeAppraisal()
        env = await build_critique(tmp_path, appraisal=appraisal, with_held=True)
        try:
            await self._register(env, root=None)
            env.held.hold_and_wake.assert_not_awaited()
            [entry] = env.held.hold.call_args.kwargs["entries"]
            assert "[retention]" in entry and "people/yoz.md" in entry, entry
        finally:
            await _teardown(env)


class TestCorrectionFindingsWait:
    """#1031 / PRD §3.7 Critique 4: a turn whose origin is ``critique`` (a
    correction) is judged, but its non-empty findings never wake a turn; they
    wait as CONTEXT for the channel's next turn. Each path that can wake is
    driven through the hook that starts it, so the origin has to travel from
    the turn's annotations to the delivery."""

    async def test_provenance_findings_on_a_correction_wait(self, tmp_path):
        env = await _provenance_env(tmp_path, with_held=True)
        env.held.hold = AsyncMock(wraps=env.held.hold)
        env.held.hold_and_wake = AsyncMock()
        agent = RenderingForkAgent({"critique"}, text=UNSUPPORTED_VERDICT)
        correction = {"origin": "critique"}
        try:
            await _provenance_exchange(env, agent, trace_id="kc", annotations=correction)
            await _stage2(env, "kc", [CLAIM], annotations=correction)
            await _run_next_task(env.task)
            env.held.hold_and_wake.assert_not_awaited()
            [call] = env.held.hold.call_args_list
            assert "[provenance]" in call.kwargs["entries"][0], call
        finally:
            await _teardown(env)

    async def test_retention_findings_on_a_correction_wait(self, tmp_path):
        env = await TestRetentionDelivery._run_retention(
            TestRetentionDelivery(), tmp_path, "kcr", origin="critique")
        try:
            env.held.hold_and_wake.assert_not_awaited()
            entries = [e for c in env.held.hold.call_args_list for e in c.kwargs["entries"]]
            assert any("[retention]" in e for e in entries), entries
        finally:
            await _teardown(env)


def test_the_wake_stub_does_not_name_a_cause_it_cannot_know():
    """#1134: one stub per drain, so it must not claim provenance found anything."""
    from corvidae_cognition.critique import PROVENANCE_NOTE, RETENTION_WAKE_NOTE

    for note in (PROVENANCE_NOTE, RETENTION_WAKE_NOTE):
        opening = note.split(".")[0].lower()
        assert "provenance check found" not in opening
        assert "left findings in the critique context" in note
