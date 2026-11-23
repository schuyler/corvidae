"""Tests for GatePlugin — the decide gate (docs/prd.md §3.7 Output gate).

Interface pinned here:
  corvidae_cognition.gates.GatePlugin (entry point "gates",
    depends_on = frozenset()):
      should_send_response — the decide gate. A final, progress, thinking or
        error emission from an origin listed in gate.send.quiet_origins is
        withheld unless the turn called speak_up; gate.send.allow_progress /
        allow_thinking suppress those emissions, recorded as
        outcomes.suppressed. Any other final is sent: there is no salience
        floor and no engagement gate (#1185).

Origin is a plugin-side convention (#131): annotations["origin"] when present,
else "user" for a trace admitted through on_message_admitted, else the
origin ``resolve_trace_origin`` answers, else "task".

Integration tests (a real Agent, outcome log and appraisal) cover the
seams: the withheld path into WP2.8, origin through tool cycles. The
decisions, tunable typing and origin derivation are unit tests against stubs.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from corvidae.channel import Channel, ChannelConfig
from corvidae.hooks import create_plugin_manager, hookimpl
from corvidae.outbound import OutboundMessage, OutboundProgress
from corvidae.trace import mint_trace_id
from helpers import add_kernel_registry, build_plugin_and_channel, drain, drain_task_queue, sent_texts
from tests.cognition.trace_helpers import TraceRecorder, add_trace
from llm_response_fixtures import (
    _make_mixed_response,
    _make_text_response,
    _make_tool_call,
)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class Env:
    """A real Agent with outcome_log, appraisal and gates registered."""

    def __init__(self, plugin, channel, db, config):
        self.plugin = plugin
        self.channel = channel
        self.db = db
        self.config = config
        self.pm = plugin.pm

    @property
    def gates(self):
        return self.pm.get_plugin("gates")

    async def settle(self):
        """Wait for every fire-and-forget record task."""
        for _ in range(5):
            pending = set(self.gates._record_tasks)
            if not pending:
                return
            await asyncio.gather(*pending, return_exceptions=True)

    async def row(self, trace_id):
        """The trace's merged records, as the old exchange_log row held them.

        Records of one source merge in order, the last write per top-level
        key winning; origin comes from ``resolve_trace_origin``.
        """
        await self.settle()
        outcomes: dict = {}
        appraisal: dict = {}
        # Merge each source's fragments in the order they were fired.
        for rec in self.recorder.records:
            if rec["trace_id"] != trace_id:
                continue
            if rec["source"] == "appraisal":
                appraisal.update(rec["record"])
            elif rec["source"] in ("gates", "silence", "critique"):
                outcomes.update(rec["record"])
        origin = await self.pm.ahook.resolve_trace_origin(trace_id=trace_id)
        assert origin is not None or outcomes or appraisal, f"no records for {trace_id}"
        return {"origin": origin, "appraisal": appraisal, "outcomes": outcomes}

    async def rows_by_type(self):
        async with self.db.execute(
            "SELECT message, message_type FROM message_log WHERE channel_id = ? ORDER BY id",
            (self.channel.id,),
        ) as cursor:
            return [(json.loads(m), t) for m, t in await cursor.fetchall()]


@pytest.fixture
async def make_env(tmp_path):
    envs = []

    async def _make(config=None, responses=()):
        from corvidae_cognition.appraisal import AppraisalPlugin
        from corvidae_cognition.gates import GatePlugin
        from corvidae_cognition.outcome_log import OutcomeLogPlugin

        cfg = config if config is not None else {}
        plugin, channel, db = await build_plugin_and_channel(mock_on_agent_response=False)
        pm = plugin.pm
        # The trace protocol plugin and a listener on it, before any reporter.
        recorder = TraceRecorder()
        add_trace(pm, recorder)
        outcome = OutcomeLogPlugin()
        pm.register(outcome, name="outcome_log")
        await outcome.on_init(pm=pm, config=cfg)
        await outcome.on_start(config=cfg)
        # Appraisal is registered so a test sees that nothing it does
        # reaches the gate path any more (#1185).
        for name, p in (("appraisal", AppraisalPlugin()), ("gates", GatePlugin())):
            pm.register(p, name=name)
            await p.on_init(pm=pm, config=cfg)
            if hasattr(p, "on_start"):
                await p.on_start(config=cfg)
        # These tests are not about forks: switch them off so stage 2 skips
        # (forks are on by default, #320) instead of eating a scripted reply.
        from corvidae.fork import ForksConfig

        plugin._forks = ForksConfig(enabled=False)
        client = MagicMock()
        client.chat = AsyncMock(side_effect=list(responses))
        plugin._client = client
        plugin._mark_started()
        env = Env(plugin, channel, db, cfg)
        env.recorder = recorder
        env.persisted = PersistedTraceIds()
        pm.register(env.persisted, name="persisted_trace_ids")
        envs.append(env)
        return env

    yield _make

    for env in envs:
        await env.settle()
        await env.gates.on_stop()
        await env.pm.get_plugin("task").on_stop()
        await env.db.close()


async def _say(env, text):
    trace_id = mint_trace_id()
    await env.plugin.on_message(channel=env.channel, sender="user", text=text, trace_id=trace_id)
    await drain(env.plugin, env.channel)
    return trace_id


class PersistedTraceIds:
    """Records every originating trace id, in order (on_message_persisted)."""

    def __init__(self):
        self.trace_ids: list[str] = []

    @hookimpl
    async def on_message_persisted(self, channel, trace_id, rowid, text, annotations):
        self.trace_ids.append(trace_id)


async def _notify(env, text, annotations=None):
    """A notification-born exchange; returns its dequeue-minted trace id."""
    before = len(env.persisted.trace_ids)
    await env.plugin.on_notify(
        channel=env.channel, source="scheduler", text=text,
        tool_call_id=None, annotations=annotations,
    )
    await drain(env.plugin, env.channel)
    assert len(env.persisted.trace_ids) == before + 1
    return env.persisted.trace_ids[-1]


# ---------------------------------------------------------------------------
# Decide gate through a real Agent
# ---------------------------------------------------------------------------


class TestDecideGate:
    async def test_background_final_is_sent_and_nothing_scores_it(self, make_env):
        """#1185: with appraisal loaded and no quiet origin, a task-origin
        final reaches the channel, and neither a send verdict nor a stage-1
        vector is written."""
        env = await make_env(responses=[_make_text_response("nothing to add.")])
        trace_id = await _notify(env, "background check finished")

        assert sent_texts(env.pm, OutboundMessage) == ["nothing to add."]
        row = await env.row(trace_id)
        assert "send" not in row["outcomes"]
        assert "stage1_out" not in row["appraisal"]
        assert row["origin"] == "task"

    async def test_quiet_origin_final_is_withheld(self, make_env):
        env = await make_env(
            config={"gate": {"send": {"quiet_origins": ["task"]}}},
            responses=[_make_text_response("nothing to add.")],
        )
        trace_id = await _notify(env, "background check finished")

        assert sent_texts(env.pm, OutboundMessage) == []
        rows = await env.rows_by_type()
        assert [t for m, t in rows if m.get("role") == "assistant"] == ["unsent"]
        row = await env.row(trace_id)
        send = row["outcomes"]["send"]
        assert send["origin"] == "task"
        assert send["vetoed"] is True
        assert send["reason"] == "quiet_origin"
        # The upsert kept the row's origin (no relabel to "user").
        assert row["origin"] == "task"

    async def test_user_exchange_through_tool_cycle_is_never_withheld(self, make_env):
        """Origin survives a tool cycle: the final is the user's, so a quiet
        task origin doesn't withhold it."""
        env = await make_env(
            config={"gate": {"send": {"quiet_origins": ["task"]}}},
            responses=[
                _make_mixed_response("", [_make_tool_call("call-1", "my_tool", {"x": "a"})]),
                _make_text_response("ok."),
            ],
        )

        async def my_tool(x: str) -> str:
            """A test tool."""
            return f"result:{x}"

        env.plugin._tools = {"my_tool": my_tool}
        trace_id = await _say(env, "look it up")
        await drain_task_queue(env.plugin)
        await drain(env.plugin, env.channel)

        assert sent_texts(env.pm, OutboundMessage) == ["ok."]
        row = await env.row(trace_id)
        assert row["origin"] == "user"


class TestEmissionSuppression:
    async def test_progress_suppressed_and_recorded(self, make_env):
        env = await make_env(responses=[
            _make_mixed_response("Let me look that up", [_make_tool_call("call-1", "my_tool", {"x": "a"})]),
            _make_text_response("Found it"),
        ])
        env.channel.runtime_overrides["gate.send.allow_progress"] = False

        async def my_tool(x: str) -> str:
            """A test tool."""
            return f"result:{x}"

        env.plugin._tools = {"my_tool": my_tool}
        trace_id = await _say(env, "look it up")
        await drain_task_queue(env.plugin)
        await drain(env.plugin, env.channel)

        assert sent_texts(env.pm, OutboundProgress) == []
        assert sent_texts(env.pm, OutboundMessage) == ["Found it"]
        row = await env.row(trace_id)
        assert row["outcomes"]["suppressed"] == {"progress": True}


class TestWithheldStillCritiqued:
    """PRD §3.7 Critique 3: a response a gate withheld is still judged.

    Before #1031 no origin could be both withheld and eligible (critique judged
    only ``reminder`` and admitted ``user`` traces, and the gate exempts both).
    Now every main-loop turn with a trace is judged, so a withheld background
    final, a quiet heartbeat's for one, must stay eligible.
    """

    async def test_every_withheld_final_is_eligible(self, make_env):
        from corvidae_cognition.critique import CritiquePlugin

        env = await make_env(config={"gate": {"send": {"quiet_origins": ["heartbeat"]}}})
        critique = CritiquePlugin()
        env.pm.register(critique, name="critique")
        await critique.on_init(pm=env.pm, config=env.config)

        withheld: list[str] = []
        for origin in ("reminder", "user", "critique", "heartbeat", "task"):
            verdict = await env.gates.should_send_response(
                channel=env.channel, text="x", emission="final",
                trace_id="t1", annotations={"origin": origin},
            )
            if verdict is False:
                withheld.append(origin)
                assert critique._eligible("t1"), origin
        # The property is only worth something if the gate withholds at all
        # under this fixture, and withholds a background origin.
        assert "heartbeat" in withheld, withheld


# ---------------------------------------------------------------------------
# Unit tests against stubs
# ---------------------------------------------------------------------------


class _FakePM:
    """A pm stand-in for tests that only read config (no hooks fire)."""

    def get_plugin(self, name):
        return None


def _writes(recorder) -> list[dict]:
    """The gate's trace records, as ``{"trace_id", "origin", "outcomes"}`` dicts."""
    return [
        {"trace_id": r["trace_id"], "origin": r["origin"], "outcomes": r["record"]}
        for r in recorder.by_source("gates")
    ]


async def _gate(config=None, *, appraisal=None):
    """A GatePlugin on a real plugin manager with the trace plugin and a
    recorder; returns (gate, recorder, channel)."""
    from corvidae_cognition.gates import GatePlugin

    pm = create_plugin_manager()
    recorder = TraceRecorder()
    add_trace(pm, recorder)
    if appraisal is not None:
        pm.register(appraisal, name="appraisal")
    gate = GatePlugin(pm=pm)
    gate.config = config or {}
    channel = Channel(transport="test", scope="gates", config=ChannelConfig())
    return gate, recorder, channel


async def _final(gate, channel, annotations=None, trace_id="t1"):
    verdict = await gate.should_send_response(
        channel=channel, text="nothing to add.", emission="final",
        trace_id=trace_id, annotations=annotations or {},
    )
    await asyncio.gather(*set(gate._record_tasks))
    return verdict


class TestEmissions:
    @pytest.mark.parametrize("emission", ["error", "something-new"])
    async def test_other_emissions_have_no_opinion(self, emission):
        gate, log, channel = await _gate()
        verdict = await gate.should_send_response(
            channel=channel, text="x", emission=emission, trace_id="t1", annotations={"origin": "heartbeat"},
        )
        assert verdict is None
        assert _writes(log) == []

    @pytest.mark.parametrize("emission", ["progress", "thinking"])
    async def test_emission_switches(self, emission):
        gate, log, channel = await _gate()
        args = dict(channel=channel, text="x", emission=emission, trace_id="t1", annotations={})
        assert await gate.should_send_response(**args) is None
        channel.runtime_overrides[f"gate.send.allow_{emission}"] = False
        assert await gate.should_send_response(**args) is False
        await asyncio.gather(*set(gate._record_tasks))
        assert _writes(log) == [{"trace_id": "t1", "origin": None, "outcomes": {"suppressed": {emission: True}}}]


class TestTunableTypes:
    @pytest.mark.parametrize(
        "value, expected",
        [(True, True), (False, False), ("false", False), ("TRUE", True), ("off", False),
         (1, True), (0, False), ("maybe", True), ({"x": 1}, True)],
    )
    def test_bool_tunable(self, value, expected):
        from corvidae_cognition.gates import _bool_tunable

        channel = Channel(transport="test", scope="t", config=ChannelConfig())
        channel.runtime_overrides["gate.send.allow_progress"] = value
        assert _bool_tunable(channel, {}, "gate.send.allow_progress", True) is expected

    def test_bad_value_warning_names_key_and_value(self, caplog):
        from corvidae_cognition.gates import _bool_tunable

        channel = Channel(transport="test", scope="t", config=ChannelConfig())
        channel.runtime_overrides["gate.send.allow_progress"] = "maybe"
        _bool_tunable(channel, {}, "gate.send.allow_progress", True)
        assert any(
            "gate.send.allow_progress" in r.getMessage() and "'maybe'" in r.getMessage()
            and "boolean" in r.getMessage()
            for r in caplog.records
        )


class TestFailOpen:
    """Anything unexpected inside the gate sends; a raise in this broadcast
    would drop the reply."""

    async def test_decide_error_sends(self):
        gate, _log, channel = await _gate({"gate": {"send": {"quiet_origins": ["heartbeat"]}}})
        gate._origin = AsyncMock(side_effect=RuntimeError("origin lookup exploded"))
        assert await _final(gate, channel, {"origin": "heartbeat"}) is None


class TestOrigin:
    async def test_meta_then_admitted_then_task(self):
        gate, _log, _channel = await _gate()
        assert await gate._origin("t1", {"origin": "reminder"}) == "reminder"
        await gate.on_message_admitted(channel=None, trace_id="t1", sender="u", text="x")
        assert await gate._origin("t1", {}) == "user"
        assert await gate._origin("t2", {}) == "task"
        assert await gate._origin(None, {}) == "task"

    async def test_evicted_user_trace_resolves_through_the_hook(self, tmp_path):
        from corvidae_cognition.outcome_log import OutcomeLogPlugin
        from corvidae.persistence import PersistencePlugin, init_db
        import aiosqlite

        async with aiosqlite.connect(str(tmp_path / "sessions.db")) as db:
            await init_db(db)
            gate, _log, channel = await _gate()
            pm = gate.pm
            # outcome_log resolves the kernel registry (#1180) and records
            # only for channels it holds, so the registry holds this one.
            add_kernel_registry(pm).channels[channel.id] = channel
            # A real outcome_log on a file DB admits "old"; gates never saw it.
            persistence = PersistencePlugin()
            persistence.db = db
            pm.register(persistence, name="persistence")
            outcome = OutcomeLogPlugin()
            pm.register(outcome, name="outcome_log")
            await outcome.on_init(pm, {})
            await outcome.on_start({})
            await outcome.on_message_admitted(
                channel=channel, trace_id="old", sender="u", text="x",
            )
            assert await gate._origin("old", {}) == "user"
            assert await gate._origin("never-seen", {}) == "task"


# ---------------------------------------------------------------------------
# #790 — quiet origins (gate.send.quiet_origins) and the speak_up tool
# ---------------------------------------------------------------------------

_QUIET = {"gate": {"send": {"quiet_origins": ["webhook:github"]}}}
_GITHUB = {"origin": "webhook:github"}


def _quiet_config(**send):
    """A config naming webhook:github quiet, plus extra gate.send keys."""
    return {"gate": {"send": {"quiet_origins": ["webhook:github"], **send}}}


async def _gate_with_tools(config):
    """A gate on stubs after on_init, plus the tools register_tools offers."""
    gate, log, channel = await _gate(config)
    await gate.on_init(pm=gate.pm, config=config)
    registry: list = []
    gate.register_tools(tool_registry=registry)
    return gate, log, channel, {tool.name: tool for tool in registry}


async def _speak_up(tools, channel, trace_id="t1", **kwargs):
    """Call speak_up as the kernel's tool Task does: the turn's trace id in
    the attribution and a ToolContext for ``channel``."""
    from corvidae.attribution import reset_attribution, set_attribution
    from corvidae.tool import ToolContext

    ctx = ToolContext(channel=channel, tool_call_id="call-1", task_queue=None)
    token = set_attribution(trace_id=trace_id)
    try:
        return await tools["speak_up"].fn(_ctx=ctx, **kwargs)
    finally:
        reset_attribution(token)


async def _emit(gate, channel, emission, annotations=None, trace_id="t1"):
    verdict = await gate.should_send_response(
        channel=channel, text="x", emission=emission,
        trace_id=trace_id, annotations=annotations or {},
    )
    await asyncio.gather(*set(gate._record_tasks))
    return verdict


def _merged_outcomes(log) -> dict:
    """Every outcomes fragment the stub log received, merged by top-level key."""
    merged: dict = {}
    for write in _writes(log):
        merged.update(write.get("outcomes", {}))
    return merged


class TestQuietOrigins:
    async def test_q1_listed_origin_final_is_withheld(self):
        gate, log, channel = await _gate(_quiet_config())
        assert await _final(gate, channel, _GITHUB) is False
        send = _writes(log)[-1]["outcomes"]["send"]
        assert send["reason"] == "quiet_origin"
        assert send["vetoed"] is True
        assert send["surfaced"] is False

    @pytest.mark.parametrize("emission", ["progress", "thinking"])
    async def test_q2_progress_and_thinking_are_vetoed(self, emission):
        config = _quiet_config(allow_progress=True, allow_thinking=True)
        gate, log, channel = await _gate(config)
        assert await _emit(gate, channel, emission, _GITHUB) is False
        assert _writes(log)[-1]["outcomes"] == {
            "suppressed": {emission: True}, "quiet_origin": "webhook:github",
        }

    async def test_q2b_error_fallback_is_withheld_from_a_quiet_turn(self):
        """A model outage during a burst of notifications is the same flood in
        error text (#936): the fallback is withheld and the veto recorded, so
        the missing message ties back to the quiet origin."""
        gate, log, channel = await _gate(_QUIET)
        assert await _emit(gate, channel, "error", _GITHUB) is False
        assert _writes(log)[-1]["outcomes"] == {
            "suppressed": {"error": True}, "quiet_origin": "webhook:github",
        }

    async def test_q2b_error_fallback_after_speak_up_is_sent(self):
        gate, log, channel, tools = await _gate_with_tools(_QUIET)
        await _speak_up(tools, channel, trace_id="t1", reason="build is broken")
        assert await _emit(gate, channel, "error", _GITHUB, trace_id="t1") is None

    @pytest.mark.parametrize("origin", ["heartbeat", "user", "reminder"])
    async def test_q2b_error_fallback_from_a_speaking_origin_is_sent(self, origin):
        """Only a quiet origin loses its error fallback: every other origin,
        including a user or reminder turn, still hears what failed."""
        gate, _log, channel = await _gate(_QUIET)
        assert await _emit(gate, channel, "error", {"origin": origin}) is None

    async def test_q3_speak_up_lifts_the_veto_for_that_trace(self):
        gate, log, channel, tools = await _gate_with_tools(_QUIET)
        await _speak_up(tools, channel, trace_id="t1", reason="build is broken")
        assert await _final(gate, channel, _GITHUB, trace_id="t1") is None
        send = _merged_outcomes(log)["send"]
        assert send["surfaced"] is True
        assert send["vetoed"] is False
        assert send["reason"] == "surfaced"
        # Another trace from the same origin stays quiet.
        assert await _final(gate, channel, _GITHUB, trace_id="t2") is False

    async def test_q3_thinking_after_a_surfaced_final_is_not_vetoed(self):
        """The final is gated before the same response's thinking, so the
        surfaced mark must outlive the final's decision."""
        gate, _log, channel, tools = await _gate_with_tools(_QUIET)
        await _speak_up(tools, channel, trace_id="t1")
        assert await _final(gate, channel, _GITHUB, trace_id="t1") is None
        assert await _emit(gate, channel, "thinking", _GITHUB, trace_id="t1") is None

    @pytest.mark.parametrize("origin", ["user", "reminder"])
    async def test_q4_user_and_reminder_cannot_be_quiet(self, origin, caplog):
        config = {"gate": {"send": {"quiet_origins": ["user", "reminder"]}}}
        gate, _log, channel = await _gate(config)
        with caplog.at_level("WARNING", logger="corvidae.gates"):
            assert await _final(gate, channel, {"origin": origin}) is None
        text = " ".join(r.getMessage() for r in caplog.records)
        assert "gate.send.quiet_origins" in text
        assert origin in text

    async def test_q5_absent_key_changes_nothing_but_still_offers_the_tool(self):
        """speak_up is offered with no quiet origin at startup, so a reload
        that adds one never leaves a veto without its hatch (#925)."""
        gate, _log, channel, tools = await _gate_with_tools({})
        assert await _final(gate, channel, _GITHUB) is None
        assert "speak_up" in tools

    async def test_q5_only_never_quiet_origins_still_offers_the_tool(self):
        config = {"gate": {"send": {"quiet_origins": ["user"]}}}
        _gate_obj, _log, _channel, tools = await _gate_with_tools(config)
        assert "speak_up" in tools

    async def test_q5_reload_adding_an_origin_has_a_working_hatch(self):
        """A quiet origin added by hot reload is vetoed at once, and the tool
        registered at startup lifts that veto with no restart (#925)."""
        gate, _log, channel, tools = await _gate_with_tools({})
        await gate.on_config_reload(config=_QUIET)
        assert await _final(gate, channel, _GITHUB, trace_id="t1") is False
        await _speak_up(tools, channel, trace_id="t2")
        assert await _final(gate, channel, _GITHUB, trace_id="t2") is None

    # #928: gate.send.quiet_origins is a per-channel runtime tunable.
    async def test_q6_runtime_override_can_lift_the_veto(self):
        gate, _log, channel = await _gate(_QUIET)
        channel.runtime_overrides["gate.send.quiet_origins"] = []
        assert await _final(gate, channel, _GITHUB) is None

    async def test_q6_runtime_override_can_add_a_quiet_origin(self):
        gate, _log, channel = await _gate({})
        channel.runtime_overrides["gate.send.quiet_origins"] = ["webhook:github"]
        assert await _final(gate, channel, _GITHUB) is False

    async def test_q6_clearing_the_override_restores_the_config_list(self):
        """set_settings with null pops the override; YAML applies again."""
        gate, _log, channel = await _gate(_QUIET)
        channel.runtime_overrides["gate.send.quiet_origins"] = []
        assert await _final(gate, channel, _GITHUB, trace_id="t1") is None
        channel.runtime_overrides.pop("gate.send.quiet_origins")
        assert await _final(gate, channel, _GITHUB, trace_id="t2") is False

    async def test_q6_override_on_one_channel_leaves_another_on_config(self):
        gate, _log, channel = await _gate(_QUIET)
        other = Channel(transport="test", scope="other", config=ChannelConfig())
        channel.runtime_overrides["gate.send.quiet_origins"] = []
        assert await _final(gate, channel, _GITHUB, trace_id="t1") is None
        assert await _final(gate, other, _GITHUB, trace_id="t2") is False

    async def test_q6_override_listing_user_is_ignored_with_a_warning(self, caplog):
        gate, _log, channel = await _gate({})
        channel.runtime_overrides["gate.send.quiet_origins"] = ["user", "webhook:github"]
        with caplog.at_level("WARNING", logger="corvidae.gates"):
            assert await _final(gate, channel, _GITHUB) is False
        text = " ".join(r.getMessage() for r in caplog.records)
        assert "gate.send.quiet_origins" in text
        assert "'user'" in text

    async def test_q6_invalid_override_fails_open_naming_key_and_channel(self, caplog):
        gate, _log, channel = await _gate(_QUIET)
        channel.runtime_overrides["gate.send.quiet_origins"] = "webhook:github"
        with caplog.at_level("WARNING", logger="corvidae.gates"):
            assert await _final(gate, channel, _GITHUB) is None
        text = " ".join(r.getMessage() for r in caplog.records)
        assert "gate.send.quiet_origins" in text
        assert "'webhook:github'" in text
        assert channel.id in text

    @pytest.mark.parametrize("value", ["webhook:github", 5, ["webhook:github", 5]])
    async def test_q7_non_list_fails_open_with_a_warning(self, value, caplog):
        config = {"gate": {"send": {"quiet_origins": value}}}
        gate, _log, channel = await _gate(config)
        with caplog.at_level("WARNING", logger="corvidae.gates"):
            assert await _final(gate, channel, _GITHUB) is None
        text = " ".join(r.getMessage() for r in caplog.records)
        assert "gate.send.quiet_origins" in text
        assert str(value) in text
        assert "list" in text

    async def test_q8_speak_up_is_recorded_on_the_trace(self):
        gate, log, channel, tools = await _gate_with_tools(_QUIET)
        await _speak_up(tools, channel, trace_id="t1", reason="build is broken")
        await asyncio.gather(*set(gate._record_tasks))
        assert _writes(log)[-1]["trace_id"] == "t1"
        assert _writes(log)[-1]["outcomes"] == {"speak_up": "build is broken"}

    async def test_q8_speak_up_without_a_trace_says_the_reply_stays_withheld(self):
        gate, log, channel, tools = await _gate_with_tools(_QUIET)
        result = await _speak_up(tools, channel, trace_id=None)
        assert "trace" in result
        assert "withheld" in result
        assert _writes(log) == []

    async def test_q8_speak_up_without_a_channel_returns_error_text(self):
        from corvidae.tool import ToolContext

        _gate_obj, _log, _channel, tools = await _gate_with_tools(_QUIET)
        ctx = ToolContext(channel=None, tool_call_id="call-1", task_queue=None)
        result = await tools["speak_up"].fn(_ctx=ctx, reason="x")
        assert "channel" in result
        assert "withheld" in result


# ---------------------------------------------------------------------------
# #1185 — stage-1 appraisal and the decide gate's salience floor are removed
# ---------------------------------------------------------------------------


class _ScoringAppraisal:
    """An appraisal plugin that would score every text 0.0, and records
    whether anything asked it to."""

    def __init__(self):
        self.calls: list[str] = []

    async def get_or_compute(self, channel, trace_id, text):
        self.calls.append("in")
        return {"salience": 0.0}

    async def get_or_compute_out(self, channel, trace_id, text):
        self.calls.append("out")
        return {"salience": 0.0}

    async def get_appraisal(self, trace_id):
        self.calls.append("get_appraisal")
        return {"salience": 0.0}

    def get_last_stage2(self, channel_id):
        self.calls.append("get_last_stage2")
        return None


class TestNoSalienceFloor:
    @pytest.mark.parametrize("origin", ["heartbeat", "task", "critique", "webhook:github"])
    async def test_a_background_final_is_sent_and_nothing_scores_it(self, origin):
        """A final from an origin that is not quiet is never withheld on a
        score: the gate asks no appraisal plugin, and records nothing."""
        appraisal = _ScoringAppraisal()
        gate, log, channel = await _gate({}, appraisal=appraisal)
        assert await _final(gate, channel, {"origin": origin}) is None
        assert appraisal.calls == []
        assert _writes(log) == []

    def test_the_gate_has_no_inbound_hook(self):
        """The engagement gate is gone: nothing in GatePlugin judges an
        inbound message."""
        from corvidae_cognition.gates import GatePlugin

        assert not hasattr(GatePlugin, "should_process_message")


class TestRemovedGateKeys:
    _REMOVED = {
        "gate": {
            "engagement": {"enforce": True, "threshold": 0.2},
            "send": {"enforce": True, "min_salience": 0.15, "quiet_origins": []},
        }
    }

    @pytest.mark.parametrize("hook", ["on_init", "on_config_reload"])
    async def test_each_removed_key_is_warned_by_name(self, hook, caplog):
        from corvidae_cognition.gates import GatePlugin

        gate = GatePlugin(pm=_FakePM())
        with caplog.at_level("WARNING", logger="corvidae"):
            if hook == "on_init":
                await gate.on_init(pm=gate.pm, config=self._REMOVED)
            else:
                await gate.on_config_reload(config=self._REMOVED)
        messages = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
        for key in ("gate.engagement.enforce", "gate.engagement.threshold",
                    "gate.send.enforce", "gate.send.min_salience"):
            hits = [m for m in messages if key in m]
            assert hits, f"no warning names {key}: {messages}"
            assert all("delete" in m and "agent.yaml" in m for m in hits), hits
        # A key that still works is not reported as removed.
        assert not any("gate.send.quiet_origins was removed" in m for m in messages)

    @pytest.mark.parametrize("config", [
        {"gate": {"send": {"quiet_origins": []}}},
        # A removed key set to null is absent, at either level.
        {"gate": {"engagement": {"enforce": None}, "send": {"min_salience": None}}},
    ], ids=["no_removed_key", "null_removed_key"])
    async def test_a_config_without_removed_keys_warns_nothing(self, config, caplog):
        from corvidae_cognition.gates import GatePlugin

        gate = GatePlugin(pm=_FakePM())
        with caplog.at_level("WARNING", logger="corvidae"):
            await gate.on_init(pm=gate.pm, config=config)
        assert [r for r in caplog.records if r.levelname == "WARNING"] == []
