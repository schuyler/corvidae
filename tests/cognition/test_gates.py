"""RED tests for WP2.9 — the engagement and decide gates, and the outbound
appraisal vector (plans/implementation/phase-2.md WP2.9; docs/prd.md §3.7).

Interface pinned here:
  corvidae_cognition.gates.GatePlugin (entry point "gates",
    depends_on = frozenset()):
      should_process_message — the engagement gate; shadow mode by default
        (gate.engagement.enforce false, gate.engagement.threshold 0.2);
        records outcomes.engagement = {salience, threshold, would_reject,
        enforced} through upsert_exchange.
      should_send_response — the decide gate; never vetoes a user-origin
        final; vetoes a non-user final under gate.send.min_salience (0.15)
        when gate.send.enforce (default true); fails open on a missing
        vector; records outcomes.send; gate.send.allow_progress /
        allow_thinking suppress those emissions, recorded as
        outcomes.suppressed.
  AppraisalPlugin.get_or_compute_out(channel, trace_id, text) and a thin
    should_send_response impl that always returns None.

Origin is a plugin-side convention (#131): annotations["origin"] when present,
else "user" for a trace admitted through on_message_admitted, else the
origin exchange_log holds, else "task".

Integration tests (a real Agent, outcome log and appraisal) cover the
seams: upsert-before-admission, the withheld path into WP2.8, direction
distinctness, dispatch order, origin through tool cycles. The decision
table, tunable typing and origin derivation are unit tests against stubs.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import pytest

from corvidae.channel import Channel, ChannelConfig
from corvidae.hooks import hookimpl, resolve_reject_wins
from corvidae.outbound import OutboundMessage, OutboundProgress
from corvidae.trace import mint_trace_id
from helpers import build_plugin_and_channel, drain, drain_task_queue, sent_texts
from llm_response_fixtures import (
    _make_mixed_response,
    _make_text_response,
    _make_tool_call,
)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


async def _seed_probe_db(path, summaries=("we discussed the garden and the tomatoes",)):
    """A session DB file whose memory_fts the appraisal probe can query."""
    db = await aiosqlite.connect(path)
    await db.execute("CREATE VIRTUAL TABLE memory_fts USING fts5(summary)")
    for s in summaries:
        await db.execute("INSERT INTO memory_fts(summary) VALUES (?)", (s,))
    await db.commit()
    await db.close()


class Env:
    """A real Agent with outcome_log, appraisal and gates registered.

    ``order`` names the registration order of "appraisal" and "gates";
    leaving one out leaves that plugin unregistered.
    """

    def __init__(self, plugin, channel, db, config):
        self.plugin = plugin
        self.channel = channel
        self.db = db
        self.config = config
        self.pm = plugin.pm

    @property
    def appraisal(self):
        return self.pm.get_plugin("appraisal")

    @property
    def gates(self):
        return self.pm.get_plugin("gates")

    async def settle(self):
        """Wait for every fire-and-forget persist/record task."""
        for _ in range(5):
            pending = set()
            if self.appraisal is not None:
                pending |= set(self.appraisal._persist_tasks)
            if self.gates is not None:
                pending |= set(self.gates._record_tasks)
            if not pending:
                return
            await asyncio.gather(*pending, return_exceptions=True)

    async def row(self, trace_id):
        await self.settle()
        async with self.db.execute(
            "SELECT origin, message_rowid, appraisal, outcomes FROM exchange_log WHERE trace_id = ?",
            (trace_id,),
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None, f"no exchange_log row for {trace_id}"
        origin, rowid, appraisal, outcomes = row
        return {
            "origin": origin,
            "message_rowid": rowid,
            "appraisal": json.loads(appraisal) if appraisal else {},
            "outcomes": json.loads(outcomes) if outcomes else {},
        }

    async def rows_by_type(self):
        async with self.db.execute(
            "SELECT message, message_type FROM message_log WHERE channel_id = ? ORDER BY id",
            (self.channel.id,),
        ) as cursor:
            return [(json.loads(m), t) for m, t in await cursor.fetchall()]


@pytest.fixture
async def make_env(tmp_path):
    envs = []

    async def _make(order=("appraisal", "gates"), config=None, responses=(), probe=False):
        from corvidae_cognition.outcome_log import OutcomeLogPlugin

        # Without probe=True the session_db file doesn't exist, so the
        # appraisal probe is disabled and vectors use the no-probe novelty.
        probe_path = str(tmp_path / f"probe{len(envs)}.db")
        if probe:
            await _seed_probe_db(probe_path)
        cfg = config if config is not None else {}
        cfg.setdefault("daemon", {})["session_db"] = probe_path

        plugin, channel, db = await build_plugin_and_channel(mock_on_agent_response=False)
        pm = plugin.pm
        outcome = OutcomeLogPlugin()
        pm.register(outcome, name="outcome_log")
        await outcome.on_init(pm=pm, config=cfg)
        await outcome.on_start(config=cfg)
        for name in order:
            if name == "appraisal":
                from corvidae_cognition.appraisal import AppraisalPlugin

                p = AppraisalPlugin()
            else:
                from corvidae_cognition.gates import GatePlugin

                p = GatePlugin()
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
        env.recorder = TraceRecorder()
        pm.register(env.recorder, name="trace_recorder")
        envs.append(env)
        return env

    yield _make

    for env in envs:
        await env.settle()
        for name in ("gates", "appraisal"):
            p = env.pm.get_plugin(name)
            if p is not None:
                await p.on_stop()
        await env.pm.get_plugin("task").on_stop()
        await env.db.close()


async def _say(env, text):
    trace_id = mint_trace_id()
    await env.plugin.on_message(channel=env.channel, sender="user", text=text, trace_id=trace_id)
    await drain(env.plugin, env.channel)
    return trace_id


class TraceRecorder:
    """Records every originating trace id, in order (on_message_persisted)."""

    def __init__(self):
        self.trace_ids: list[str] = []

    @hookimpl
    async def on_message_persisted(self, channel, trace_id, rowid, text, annotations):
        self.trace_ids.append(trace_id)


async def _notify(env, text, annotations=None):
    """A notification-born exchange; returns its dequeue-minted trace id."""
    before = len(env.recorder.trace_ids)
    await env.plugin.on_notify(
        channel=env.channel, source="scheduler", text=text,
        tool_call_id=None, annotations=annotations,
    )
    await drain(env.plugin, env.channel)
    assert len(env.recorder.trace_ids) == before + 1
    return env.recorder.trace_ids[-1]


# ---------------------------------------------------------------------------
# Engagement gate (inbound)
# ---------------------------------------------------------------------------


class TestEngagementShadow:
    async def test_low_salience_still_processed_and_would_reject_recorded(self, make_env):
        env = await make_env(
            config={"gate": {"engagement": {"threshold": 1.0}}},
            responses=[_make_text_response("sure")],
        )
        trace_id = await _say(env, "ok")

        assert sent_texts(env.pm, OutboundMessage) == ["sure"]
        row = await env.row(trace_id)
        engagement = row["outcomes"]["engagement"]
        assert engagement["would_reject"] is True
        assert engagement["enforced"] is False
        assert engagement["threshold"] == 1.0
        assert 0.0 <= engagement["salience"] <= 1.0
        # The upsert regression: the gate write precedes the admission
        # insert, and the row still ends up with the admission fields.
        assert row["origin"] == "user"
        assert row["message_rowid"] is not None
        assert "stage1" in row["appraisal"]


class TestEngagementEnforced:
    async def test_enforced_rejects_below_threshold(self, make_env):
        env = await make_env(config={"gate": {"engagement": {"enforce": True, "threshold": 1.0}}})
        rejected = []
        original = env.pm.ahook.on_message_rejected

        async def spy(**kwargs):
            rejected.append(kwargs["trace_id"])
            return await original(**kwargs)

        env.pm.ahook.on_message_rejected = spy
        trace_id = await _say(env, "ok")

        assert rejected == [trace_id]
        assert sent_texts(env.pm, OutboundMessage) == []
        row = await env.row(trace_id)
        assert row["outcomes"]["engagement"]["enforced"] is True
        assert row["outcomes"]["engagement"]["would_reject"] is True
        assert row["outcomes"]["gate"] == "rejected"
        assert "stage1" in row["appraisal"]
        assert row["message_rowid"] is None

# ---------------------------------------------------------------------------
# Decide gate (outbound)
# ---------------------------------------------------------------------------


class TestDecideGate:
    async def test_user_final_never_vetoed_even_at_max_threshold(self, make_env):
        env = await make_env(
            config={"gate": {"send": {"min_salience": 1.0}}},
            responses=[_make_text_response("ok.")],
        )
        trace_id = await _say(env, "ok")

        assert sent_texts(env.pm, OutboundMessage) == ["ok."]
        row = await env.row(trace_id)
        send = row["outcomes"]["send"]
        assert send["origin"] == "user"
        assert send["vetoed"] is False
        assert "stage1_out" in row["appraisal"]

    async def test_task_origin_low_salience_final_is_withheld(self, make_env):
        env = await make_env(
            config={"gate": {"send": {"min_salience": 1.0}}},
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
        assert send["enforced"] is True
        assert send["salience"] < send["min_salience"]
        assert "stage1_out" in row["appraisal"]
        # The upsert kept the row's origin (no relabel to "user").
        assert row["origin"] == "task"

    async def test_user_exchange_through_tool_cycle_is_never_withheld(self, make_env):
        env = await make_env(
            config={"gate": {"send": {"min_salience": 1.0}}},
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
        assert row["outcomes"]["send"]["origin"] == "user"
        assert row["origin"] == "user"


class TestDirectionDistinctness:
    async def test_user_exchange_has_distinct_stage1_and_stage1_out(self, make_env):
        env = await make_env(probe=True, responses=[_make_text_response("Done. I will water the tomatoes on Friday.")])
        probes = []
        original = env.appraisal._probe_query

        async def spy(channel, text):
            probes.append(text)
            return await original(channel, text)

        env.appraisal._probe_query = spy
        trace_id = await _say(env, "Did you ever talk about the garden?")

        row = await env.row(trace_id)
        assert "stage1" in row["appraisal"] and "stage1_out" in row["appraisal"]
        assert row["appraisal"]["stage1"] != row["appraisal"]["stage1_out"]
        assert probes == [
            "Did you ever talk about the garden?",
            "Done. I will water the tomatoes on Friday.",
        ]


class TestComputeFailureFailsOpen:
    async def test_outbound_compute_raising_sends_final_and_records_miss(self, make_env):
        env = await make_env(
            config={"gate": {"send": {"min_salience": 1.0}}},
            responses=[_make_text_response("nothing to add.")],
        )

        async def boom(channel, text, direction):
            if direction == "out":
                raise RuntimeError("blend exploded")
            return await type(env.appraisal)._compute(env.appraisal, channel, text, direction)

        env.appraisal._compute = boom

        # The thin trigger alone returns None and swallows nothing silently.
        results = await env.appraisal.should_send_response(
            channel=env.channel, text="x", emission="final",
            trace_id=mint_trace_id(), annotations={},
        )
        assert results is None

        trace_id = await _notify(env, "background check finished")
        assert sent_texts(env.pm, OutboundMessage) == ["nothing to add."]
        send = (await env.row(trace_id))["outcomes"]["send"]
        assert send["vetoed"] is False
        assert send["reason"] == "no_vector"


class TestOrderingIndependence:
    @pytest.mark.parametrize("order", [("appraisal", "gates"), ("gates", "appraisal")])
    async def test_vectors_observed_and_probe_runs_once_per_direction(self, make_env, order):
        env = await make_env(order=order, config={
            "gate": {"engagement": {"threshold": 1.0}, "send": {"min_salience": 1.0}},
        })
        computes = []
        original = env.appraisal._compute

        async def spy(channel, text, direction):
            computes.append(direction)
            await asyncio.sleep(0.01)  # keep the compute in flight across callers
            return await original(channel, text, direction)

        env.appraisal._compute = spy
        trace_id = mint_trace_id()

        # One broadcast each: the gate and the appraisal trigger run
        # concurrently in a single asyncio.gather.
        inbound = await env.pm.ahook.should_process_message(
            channel=env.channel, sender="user", text="ok", trace_id=trace_id,
        )
        outbound = await env.pm.ahook.should_send_response(
            channel=env.channel, text="nothing to add.", emission="final",
            trace_id=trace_id, annotations={"origin": "heartbeat"},
        )

        assert resolve_reject_wins(inbound) is None  # shadow mode
        assert resolve_reject_wins(outbound) is False
        assert computes == ["in", "out"]
        row = await env.row(trace_id)
        assert row["outcomes"]["engagement"]["would_reject"] is True
        assert row["outcomes"]["send"]["vetoed"] is True


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

class TestTunableFlips:
    async def test_runtime_override_flips_next_exchange(self, make_env):
        env = await make_env(responses=[
            _make_text_response("nothing to add."),
            _make_text_response("nothing to add."),
        ])
        await _notify(env, "check one")
        assert sent_texts(env.pm, OutboundMessage) == ["nothing to add."]

        env.channel.runtime_overrides["gate.send.min_salience"] = 1.0
        await _notify(env, "check two")
        assert sent_texts(env.pm, OutboundMessage) == ["nothing to add."]

    async def test_config_reload_flips_next_exchange(self, make_env):
        env = await make_env(responses=[
            _make_text_response("nothing to add."),
            _make_text_response("nothing to add."),
        ])
        await _notify(env, "check one")
        assert len(sent_texts(env.pm, OutboundMessage)) == 1

        new_config = {**env.config, "gate": {"send": {"min_salience": 1.0}}}
        await env.gates.on_config_reload(config=new_config)
        await _notify(env, "check two")
        assert len(sent_texts(env.pm, OutboundMessage)) == 1


class TestWithheldStillCritiqued:
    """WP2.7's deferred assertion is now unreachable, and this says why.

    Critique is eligible only for ``reminder`` origin and for admitted ``user``
    traces; a user final is never withheld, so the one origin that could be both
    withheld and eligible was the reminder. Exempting reminders from the gate
    (the change above) removed the last vehicle, and a test that constructs an
    impossible combination would assert nothing. Kept as a class so the next
    reader finds the reasoning rather than a gap: if a new origin becomes
    eligible *and* withholdable, this is where its test goes.
    """

    async def test_no_origin_is_both_withheld_and_eligible(self, make_env):
        from corvidae_cognition.critique import CritiquePlugin

        env = await make_env(config={"gate": {"send": {"min_salience": 1.0}}})
        critique = CritiquePlugin()
        env.pm.register(critique, name="critique")
        await critique.on_init(pm=env.pm, config=env.config)

        withheld: list[str] = []
        for origin in ("reminder", "user", "critique", "heartbeat", "task"):
            eligible = critique._eligible("t1", {"origin": origin})
            verdict = await env.gates.should_send_response(
                channel=env.channel, text="x", emission="final",
                trace_id="t1", annotations={"origin": origin},
            )
            assert not (eligible and verdict is False), origin
            if verdict is False:
                withheld.append(origin)
        # The assertion above is only worth something if the gate can withhold
        # at all under this fixture: an exempt-everything gate would pass it
        # vacuously.
        assert withheld, "no origin was withheld, so the property is untested"
        # The same guard on the other half: a broken _eligible that returned
        # False for everything would satisfy the assertion above silently.
        assert any(
            critique._eligible("t1", {"origin": origin})
            for origin in ("reminder", "user", "critique", "heartbeat", "task")
        ), "no origin was eligible, so the property is untested"

class _FakePM:
    def __init__(self, **plugins):
        self.plugins = plugins

    def get_plugin(self, name):
        return self.plugins.get(name)


class _StubAppraisal:
    """Returns fixed vectors; ``salience=None`` makes the compute raise."""

    def __init__(self, salience=0.1, input_salience=0.5):
        self.salience = salience
        self.input_salience = input_salience

    async def _vector(self, channel, trace_id, text):
        if self.salience is None:
            raise RuntimeError("compute failed")
        return {"salience": self.salience}

    get_or_compute = _vector
    get_or_compute_out = _vector

    async def get_appraisal(self, trace_id):
        return {"salience": self.input_salience}

    def get_last_stage2(self, channel_id):
        return None


class _StubOutcomeLog:
    def __init__(self):
        self.writes: list[dict] = []

    async def upsert_exchange(self, trace_id, channel_id, origin=None, **columns):
        self.writes.append({"trace_id": trace_id, "origin": origin, **columns})


async def _gate(config=None, *, appraisal=None, persistence=None):
    """A GatePlugin wired to stubs; returns (gate, outcome_log, channel)."""
    from corvidae_cognition.gates import GatePlugin

    outcome_log = _StubOutcomeLog()
    plugins = {"outcome_log": outcome_log}
    if appraisal is not None:
        plugins["appraisal"] = appraisal
    if persistence is not None:
        plugins["persistence"] = persistence
    gate = GatePlugin(pm=_FakePM(**plugins))
    gate.config = config or {}
    channel = Channel(transport="test", scope="gates", config=ChannelConfig())
    return gate, outcome_log, channel


async def _final(gate, channel, annotations=None, trace_id="t1"):
    verdict = await gate.should_send_response(
        channel=channel, text="nothing to add.", emission="final",
        trace_id=trace_id, annotations=annotations or {},
    )
    await asyncio.gather(*set(gate._record_tasks))
    return verdict


class TestDecisionTable:
    @pytest.mark.parametrize(
        "annotations, salience, config, verdict, record",
        [
            # Non-user, below threshold, enforced: withheld.
            ({"origin": "heartbeat"}, 0.1, {}, False,
             {"vetoed": True, "would_veto": True, "reason": "below_threshold"}),
            # At or above threshold: sent.
            ({"origin": "heartbeat"}, 0.5, {}, None,
             {"vetoed": False, "would_veto": False, "reason": "at_or_above_threshold"}),
            # Shadow mode: would veto, sends.
            ({"origin": "heartbeat"}, 0.1, {"gate": {"send": {"enforce": False}}}, None,
             {"vetoed": False, "would_veto": True, "enforced": False}),
            # A reminder is text the operator scheduled: never withheld, and the
            # record says why rather than blaming the threshold -- while still
            # carrying the counterfactual the threshold fit needs.
            ({"origin": "reminder"}, 0.1, {}, None,
             {"origin": "reminder", "vetoed": False, "would_veto": True,
              "reason": "reminder_origin"}),
            # annotations["origin"] is used as-is, and every other origin still
            # takes the threshold.
            ({"origin": "critique"}, 0.1, {}, False, {"origin": "critique", "vetoed": True}),
            # User origin is never withheld, at any threshold.
            ({"origin": "user"}, 0.0, {"gate": {"send": {"min_salience": 1.0}}}, None,
             {"vetoed": False, "reason": "user_origin"}),
        ],
    )
    async def test_final_decisions(self, annotations, salience, config, verdict, record):
        gate, log, channel = await _gate(config, appraisal=_StubAppraisal(salience))
        assert await _final(gate, channel, annotations) is verdict
        send = log.writes[-1]["outcomes"]["send"]
        assert {k: send[k] for k in record} == record

    @pytest.mark.parametrize("salience", [0.0, 0.05, 0.147])
    async def test_a_reminder_is_never_withheld(self, salience):
        """38 reminder finals were dropped in three days, scored 0.05-0.15.

        Salience is a guess about whether text is worth saying; a reminder is
        text the operator asked for at a time they chose, so the gate has no
        business overruling it (#see PR)."""
        gate, log, channel = await _gate(
            {"gate": {"send": {"min_salience": 0.15}}}, appraisal=_StubAppraisal(salience)
        )
        assert await _final(gate, channel, {"origin": "reminder"}) is None
        send = log.writes[-1]["outcomes"]["send"]
        assert send["vetoed"] is False and send["reason"] == "reminder_origin"
        assert send["salience"] == salience
        # The counterfactual survives the exemption: this row would have been
        # withheld, so a threshold fit can still count it.
        assert send["would_veto"] is True

    async def test_user_final_skips_advisory_reads(self):
        gate, log, channel = await _gate(appraisal=_StubAppraisal(0.0))
        await _final(gate, channel, {"origin": "user"})
        assert "advisory" not in log.writes[-1]["outcomes"]["send"]

    async def test_non_user_final_records_advisory_without_nulls(self):
        gate, log, channel = await _gate(appraisal=_StubAppraisal(0.1, input_salience=0.4))
        await _final(gate, channel, {"origin": "heartbeat"})
        # previous_stage2 is None, so it is left out rather than nulled
        # (a null in a json_patch fragment deletes the key).
        assert log.writes[-1]["outcomes"]["send"]["advisory"] == {"input_salience": 0.4}

    @pytest.mark.parametrize("appraisal", [None, _StubAppraisal(salience=None)], ids=["absent", "raising"])
    async def test_missing_vector_fails_open_and_omits_salience(self, appraisal):
        gate, log, channel = await _gate(
            {"gate": {"send": {"min_salience": 1.0}}}, appraisal=appraisal,
        )
        assert await _final(gate, channel, {"origin": "heartbeat"}) is None
        send = log.writes[-1]["outcomes"]["send"]
        assert send["reason"] == "no_vector"
        assert "salience" not in send

    @pytest.mark.parametrize("emission", ["error", "something-new"])
    async def test_other_emissions_have_no_opinion(self, emission):
        gate, log, channel = await _gate(
            {"gate": {"send": {"min_salience": 1.0}}}, appraisal=_StubAppraisal(0.0),
        )
        verdict = await gate.should_send_response(
            channel=channel, text="x", emission=emission, trace_id="t1", annotations={"origin": "heartbeat"},
        )
        assert verdict is None
        assert log.writes == []

    @pytest.mark.parametrize("emission", ["progress", "thinking"])
    async def test_emission_switches(self, emission):
        gate, log, channel = await _gate()
        args = dict(channel=channel, text="x", emission=emission, trace_id="t1", annotations={})
        assert await gate.should_send_response(**args) is None
        channel.runtime_overrides[f"gate.send.allow_{emission}"] = False
        assert await gate.should_send_response(**args) is False
        await asyncio.gather(*set(gate._record_tasks))
        assert log.writes == [{"trace_id": "t1", "origin": None, "outcomes": {"suppressed": {emission: True}}}]


class TestEngagementDecisions:
    @pytest.mark.parametrize(
        "salience, config, verdict, would_reject",
        [
            (0.1, {}, None, True),  # shadow mode: records, admits
            (0.1, {"gate": {"engagement": {"enforce": True}}}, False, True),
            (0.5, {"gate": {"engagement": {"enforce": True}}}, None, False),
        ],
    )
    async def test_engagement(self, salience, config, verdict, would_reject):
        gate, log, channel = await _gate(config, appraisal=_StubAppraisal(salience))
        result = await gate.should_process_message(channel=channel, sender="u", text="ok", trace_id="t1")
        await asyncio.gather(*set(gate._record_tasks))
        assert result is verdict
        engagement = log.writes[-1]["outcomes"]["engagement"]
        assert engagement["would_reject"] is would_reject
        assert log.writes[-1]["origin"] == "user"

    async def test_no_appraisal_admits_and_records_nothing(self):
        gate, log, channel = await _gate({"gate": {"engagement": {"enforce": True, "threshold": 1.0}}})
        assert await gate.should_process_message(channel=channel, sender="u", text="ok", trace_id="t1") is None
        assert log.writes == []


class TestTunableTypes:
    @pytest.mark.parametrize(
        "value, expected",
        [(0.3, 0.3), ("0.3", 0.3), (1, 1.0), ("high", 0.2), (["x"], 0.2), (True, 0.2)],
    )
    def test_float_tunable(self, value, expected):
        from corvidae_cognition.gates import _float_tunable

        channel = Channel(transport="test", scope="t", config=ChannelConfig())
        channel.runtime_overrides["gate.engagement.threshold"] = value
        assert _float_tunable(channel, {}, "gate.engagement.threshold", 0.2) == expected

    @pytest.mark.parametrize(
        "value, expected",
        [(True, True), (False, False), ("false", False), ("TRUE", True), ("off", False),
         (1, True), (0, False), ("maybe", True), ({"x": 1}, True)],
    )
    def test_bool_tunable(self, value, expected):
        from corvidae_cognition.gates import _bool_tunable

        channel = Channel(transport="test", scope="t", config=ChannelConfig())
        channel.runtime_overrides["gate.send.enforce"] = value
        assert _bool_tunable(channel, {}, "gate.send.enforce", True) is expected

    def test_bad_value_warning_names_key_and_value(self, caplog):
        from corvidae_cognition.gates import _float_tunable

        channel = Channel(transport="test", scope="t", config=ChannelConfig())
        channel.runtime_overrides["gate.engagement.threshold"] = "high"
        _float_tunable(channel, {}, "gate.engagement.threshold", 0.2)
        assert any(
            "gate.engagement.threshold" in r.getMessage() and "'high'" in r.getMessage()
            and "number" in r.getMessage()
            for r in caplog.records
        )

    async def test_string_false_does_not_enforce_engagement(self):
        gate, log, channel = await _gate(appraisal=_StubAppraisal(0.0))
        channel.runtime_overrides["gate.engagement.enforce"] = "false"
        verdict = await gate.should_process_message(channel=channel, sender="u", text="ok", trace_id="t1")
        await asyncio.gather(*set(gate._record_tasks))
        assert verdict is None
        assert log.writes[-1]["outcomes"]["engagement"]["enforced"] is False


class TestFailOpen:
    """Anything unexpected inside either gate admits and sends; a raise in
    these broadcasts would drop the message or the reply."""

    async def test_engagement_error_admits(self):
        gate, _log, channel = await _gate(appraisal=_StubAppraisal(salience="not a number"))
        assert await gate.should_process_message(channel=channel, sender="u", text="ok", trace_id="t1") is None

    async def test_decide_error_sends(self):
        gate, _log, channel = await _gate(appraisal=_StubAppraisal(salience="not a number"))
        assert await _final(gate, channel, {"origin": "heartbeat"}) is None


class TestOrigin:
    async def test_meta_then_admitted_then_task(self):
        gate, _log, _channel = await _gate()
        assert await gate._origin("t1", {"origin": "reminder"}) == "reminder"
        await gate.on_message_admitted(channel=None, trace_id="t1", sender="u", text="x")
        assert await gate._origin("t1", {}) == "user"
        assert await gate._origin("t2", {}) == "task"
        assert await gate._origin(None, {}) == "task"

    async def test_evicted_user_trace_reads_exchange_log(self):
        from types import SimpleNamespace

        async with aiosqlite.connect(":memory:") as db:
            await db.execute("CREATE TABLE exchange_log (trace_id TEXT, origin TEXT)")
            await db.execute("INSERT INTO exchange_log VALUES ('old', 'user')")
            gate, log, channel = await _gate(
                {"gate": {"send": {"min_salience": 1.0}}},
                appraisal=_StubAppraisal(0.0), persistence=SimpleNamespace(db=db),
            )
            assert await gate._origin("old", {}) == "user"
            # So its final is never withheld, and the record doesn't relabel the row.
            assert await _final(gate, channel, trace_id="old") is None
            assert log.writes[-1]["origin"] is None


# ---------------------------------------------------------------------------
# #790 — quiet origins (gate.send.quiet_origins) and the speak_up tool
# ---------------------------------------------------------------------------

_QUIET = {"gate": {"send": {"quiet_origins": ["webhook:github"]}}}
_GITHUB = {"origin": "webhook:github"}


def _quiet_config(**send):
    """A config naming webhook:github quiet, plus extra gate.send keys."""
    return {"gate": {"send": {"quiet_origins": ["webhook:github"], **send}}}


async def _gate_with_tools(config, *, appraisal=None):
    """A gate on stubs after on_init, plus the tools register_tools offers."""
    gate, log, channel = await _gate(config, appraisal=appraisal)
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
    for write in log.writes:
        merged.update(write.get("outcomes", {}))
    return merged


class TestQuietOrigins:
    @pytest.mark.parametrize(
        "appraisal, config",
        [
            (_StubAppraisal(0.9), _quiet_config()),
            (None, _quiet_config()),
            (_StubAppraisal(0.9), _quiet_config(enforce=False)),
        ],
        ids=["high_salience", "no_appraisal", "enforce_off"],
    )
    async def test_q1_listed_origin_final_is_withheld(self, appraisal, config):
        gate, log, channel = await _gate(config, appraisal=appraisal)
        assert await _final(gate, channel, _GITHUB) is False
        send = log.writes[-1]["outcomes"]["send"]
        assert send["reason"] == "quiet_origin"
        assert send["vetoed"] is True
        assert send["surfaced"] is False

    @pytest.mark.parametrize("emission", ["progress", "thinking"])
    async def test_q2_progress_and_thinking_are_vetoed(self, emission):
        config = _quiet_config(allow_progress=True, allow_thinking=True)
        gate, log, channel = await _gate(config, appraisal=_StubAppraisal(0.9))
        assert await _emit(gate, channel, emission, _GITHUB) is False
        assert log.writes[-1]["outcomes"] == {
            "suppressed": {emission: True}, "quiet_origin": "webhook:github",
        }

    async def test_q2_error_emission_is_not_vetoed(self):
        gate, log, channel = await _gate(_QUIET, appraisal=_StubAppraisal(0.9))
        assert await _emit(gate, channel, "error", _GITHUB) is None

    async def test_q3_speak_up_lifts_the_veto_for_that_trace(self):
        gate, log, channel, tools = await _gate_with_tools(
            _QUIET, appraisal=_StubAppraisal(0.9)
        )
        await _speak_up(tools, channel, trace_id="t1", reason="build is broken")
        assert await _final(gate, channel, _GITHUB, trace_id="t1") is None
        send = _merged_outcomes(log)["send"]
        assert send["surfaced"] is True
        # Another trace from the same origin stays quiet.
        assert await _final(gate, channel, _GITHUB, trace_id="t2") is False

    async def test_q3_normal_policy_still_applies_after_speak_up(self):
        config = _quiet_config(min_salience=0.5)
        gate, log, channel, tools = await _gate_with_tools(
            config, appraisal=_StubAppraisal(0.05)
        )
        await _speak_up(tools, channel, trace_id="t1")
        assert await _final(gate, channel, _GITHUB, trace_id="t1") is False
        send = log.writes[-1]["outcomes"]["send"]
        assert send["reason"] == "below_threshold"
        assert send["surfaced"] is True

    async def test_q3_thinking_after_a_surfaced_final_is_not_vetoed(self):
        """The final is gated before the same response's thinking, so the
        surfaced mark must outlive the final's decision."""
        gate, _log, channel, tools = await _gate_with_tools(
            _QUIET, appraisal=_StubAppraisal(0.9)
        )
        await _speak_up(tools, channel, trace_id="t1")
        assert await _final(gate, channel, _GITHUB, trace_id="t1") is None
        assert await _emit(gate, channel, "thinking", _GITHUB, trace_id="t1") is None

    @pytest.mark.parametrize("origin", ["user", "reminder"])
    async def test_q4_user_and_reminder_cannot_be_quiet(self, origin, caplog):
        config = {"gate": {"send": {"quiet_origins": ["user", "reminder"]}}}
        gate, _log, channel = await _gate(config, appraisal=_StubAppraisal(0.9))
        with caplog.at_level("WARNING", logger="corvidae.gates"):
            assert await _final(gate, channel, {"origin": origin}) is None
        text = " ".join(r.getMessage() for r in caplog.records)
        assert "gate.send.quiet_origins" in text
        assert origin in text

    async def test_q5_absent_key_changes_nothing_but_still_offers_the_tool(self):
        """speak_up is offered with no quiet origin at startup, so a reload
        that adds one never leaves a veto without its hatch (#925)."""
        gate, _log, channel, tools = await _gate_with_tools(
            {}, appraisal=_StubAppraisal(0.9)
        )
        assert await _final(gate, channel, _GITHUB) is None
        assert "speak_up" in tools

    async def test_q5_only_never_quiet_origins_still_offers_the_tool(self):
        config = {"gate": {"send": {"quiet_origins": ["user"]}}}
        _gate_obj, _log, _channel, tools = await _gate_with_tools(config)
        assert "speak_up" in tools

    async def test_q5_speak_up_with_nothing_quiet_changes_no_decision(self):
        """With no quiet origin, a surfaced trace gets the same verdict and
        reason as one that never called speak_up."""
        config = {"gate": {"send": {"min_salience": 0.5}}}
        gate, log, channel, tools = await _gate_with_tools(
            config, appraisal=_StubAppraisal(0.05)
        )
        await _speak_up(tools, channel, trace_id="t1")
        assert await _final(gate, channel, _GITHUB, trace_id="t1") is False
        surfaced_reason = log.writes[-1]["outcomes"]["send"]["reason"]
        assert await _final(gate, channel, _GITHUB, trace_id="t2") is False
        assert log.writes[-1]["outcomes"]["send"]["reason"] == surfaced_reason

    async def test_q5_reload_adding_an_origin_has_a_working_hatch(self):
        """A quiet origin added by hot reload is vetoed at once, and the tool
        registered at startup lifts that veto with no restart (#925)."""
        gate, _log, channel, tools = await _gate_with_tools(
            {}, appraisal=_StubAppraisal(0.9)
        )
        await gate.on_config_reload(config=_QUIET)
        assert await _final(gate, channel, _GITHUB, trace_id="t1") is False
        await _speak_up(tools, channel, trace_id="t2")
        assert await _final(gate, channel, _GITHUB, trace_id="t2") is None

    # #928: gate.send.quiet_origins is a per-channel runtime tunable.
    async def test_q6_runtime_override_can_lift_the_veto(self):
        gate, _log, channel = await _gate(_QUIET, appraisal=_StubAppraisal(0.9))
        channel.runtime_overrides["gate.send.quiet_origins"] = []
        assert await _final(gate, channel, _GITHUB) is None

    async def test_q6_runtime_override_can_add_a_quiet_origin(self):
        gate, _log, channel = await _gate({}, appraisal=_StubAppraisal(0.9))
        channel.runtime_overrides["gate.send.quiet_origins"] = ["webhook:github"]
        assert await _final(gate, channel, _GITHUB) is False

    async def test_q6_clearing_the_override_restores_the_config_list(self):
        """set_settings with null pops the override; YAML applies again."""
        gate, _log, channel = await _gate(_QUIET, appraisal=_StubAppraisal(0.9))
        channel.runtime_overrides["gate.send.quiet_origins"] = []
        assert await _final(gate, channel, _GITHUB, trace_id="t1") is None
        channel.runtime_overrides.pop("gate.send.quiet_origins")
        assert await _final(gate, channel, _GITHUB, trace_id="t2") is False

    async def test_q6_override_on_one_channel_leaves_another_on_config(self):
        gate, _log, channel = await _gate(_QUIET, appraisal=_StubAppraisal(0.9))
        other = Channel(transport="test", scope="other", config=ChannelConfig())
        channel.runtime_overrides["gate.send.quiet_origins"] = []
        assert await _final(gate, channel, _GITHUB, trace_id="t1") is None
        assert await _final(gate, other, _GITHUB, trace_id="t2") is False

    async def test_q6_override_listing_user_is_ignored_with_a_warning(self, caplog):
        gate, _log, channel = await _gate({}, appraisal=_StubAppraisal(0.9))
        channel.runtime_overrides["gate.send.quiet_origins"] = ["user", "webhook:github"]
        with caplog.at_level("WARNING", logger="corvidae.gates"):
            assert await _final(gate, channel, _GITHUB) is False
        text = " ".join(r.getMessage() for r in caplog.records)
        assert "gate.send.quiet_origins" in text
        assert "'user'" in text

    async def test_q6_invalid_override_fails_open_naming_key_and_channel(self, caplog):
        gate, _log, channel = await _gate(_QUIET, appraisal=_StubAppraisal(0.9))
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
        gate, _log, channel = await _gate(config, appraisal=_StubAppraisal(0.9))
        with caplog.at_level("WARNING", logger="corvidae.gates"):
            assert await _final(gate, channel, _GITHUB) is None
        text = " ".join(r.getMessage() for r in caplog.records)
        assert "gate.send.quiet_origins" in text
        assert str(value) in text
        assert "list" in text

    async def test_q8_speak_up_is_recorded_on_the_trace(self):
        gate, log, channel, tools = await _gate_with_tools(_QUIET)
        result = await _speak_up(tools, channel, trace_id="t1", reason="build is broken")
        await asyncio.gather(*set(gate._record_tasks))
        assert log.writes[-1]["trace_id"] == "t1"
        assert log.writes[-1]["outcomes"] == {"speak_up": "build is broken"}

    async def test_q8_speak_up_without_a_trace_says_the_reply_stays_withheld(self):
        gate, log, channel, tools = await _gate_with_tools(_QUIET)
        result = await _speak_up(tools, channel, trace_id=None)
        assert "trace" in result
        assert "withheld" in result
        assert log.writes == []

    async def test_q8_speak_up_without_a_channel_returns_error_text(self):
        from corvidae.tool import ToolContext

        _gate_obj, _log, _channel, tools = await _gate_with_tools(_QUIET)
        ctx = ToolContext(channel=None, tool_call_id="call-1", task_queue=None)
        result = await tools["speak_up"].fn(_ctx=ctx, reason="x")
        assert "channel" in result
        assert "withheld" in result
