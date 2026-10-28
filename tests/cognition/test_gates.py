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

Origin is a plugin-side convention (#131): meta["origin"] when present,
else "user" for a trace admitted through on_message_admitted, else "task".
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import pytest

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

    async def _make(order=("appraisal", "gates"), config=None, responses=()):
        from corvidae_cognition.outcome_log import OutcomeLogPlugin

        probe_path = str(tmp_path / f"probe{len(envs)}.db")
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
        client = MagicMock()
        client.chat = AsyncMock(side_effect=list(responses))
        plugin._client = client
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
    async def on_message_persisted(self, channel, trace_id, rowid, text, meta):
        self.trace_ids.append(trace_id)


async def _notify(env, text, meta=None):
    """A notification-born exchange; returns its dequeue-minted trace id."""
    before = len(env.recorder.trace_ids)
    await env.plugin.on_notify(
        channel=env.channel, source="scheduler", text=text,
        tool_call_id=None, meta=meta,
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

    async def test_enforced_admits_above_threshold(self, make_env):
        env = await make_env(
            config={"gate": {"engagement": {"enforce": True, "threshold": 0.0}}},
            responses=[_make_text_response("sure")],
        )
        trace_id = await _say(env, "ok")
        assert sent_texts(env.pm, OutboundMessage) == ["sure"]
        row = await env.row(trace_id)
        assert row["outcomes"]["engagement"]["would_reject"] is False


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

    async def test_task_origin_above_threshold_is_sent_and_recorded(self, make_env):
        env = await make_env(
            config={"gate": {"send": {"min_salience": 0.0}}},
            responses=[_make_text_response("Heads up: the build broke.")],
        )
        trace_id = await _notify(env, "background check finished")

        assert sent_texts(env.pm, OutboundMessage) == ["Heads up: the build broke."]
        send = (await env.row(trace_id))["outcomes"]["send"]
        assert send["vetoed"] is False

    async def test_shadow_send_records_would_veto_but_sends(self, make_env):
        env = await make_env(
            config={"gate": {"send": {"enforce": False, "min_salience": 1.0}}},
            responses=[_make_text_response("nothing to add.")],
        )
        trace_id = await _notify(env, "background check finished")

        assert sent_texts(env.pm, OutboundMessage) == ["nothing to add."]
        send = (await env.row(trace_id))["outcomes"]["send"]
        assert send["vetoed"] is False
        assert send["would_veto"] is True
        assert send["enforced"] is False

    async def test_meta_origin_is_used(self, make_env):
        env = await make_env(
            config={"gate": {"send": {"min_salience": 1.0}}},
            responses=[_make_text_response("stretch time.")],
        )
        trace_id = await _notify(env, "time to stretch", meta={"origin": "reminder"})

        send = (await env.row(trace_id))["outcomes"]["send"]
        assert send["origin"] == "reminder"
        assert send["vetoed"] is True

    async def test_error_emission_is_never_vetoed(self, make_env):
        env = await make_env(config={"gate": {"send": {"min_salience": 1.0}}})
        results = await env.pm.ahook.should_send_response(
            channel=env.channel, text="Sorry — it broke", emission="error",
            trace_id=mint_trace_id(), meta={"origin": "heartbeat"},
        )
        assert resolve_reject_wins(results) is None


class TestDirectionDistinctness:
    async def test_user_exchange_has_distinct_stage1_and_stage1_out(self, make_env):
        env = await make_env(responses=[_make_text_response("Done. I will water the tomatoes on Friday.")])
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
            trace_id=mint_trace_id(), meta={},
        )
        assert results is None

        trace_id = await _notify(env, "background check finished")
        assert sent_texts(env.pm, OutboundMessage) == ["nothing to add."]
        send = (await env.row(trace_id))["outcomes"]["send"]
        assert send["vetoed"] is False
        assert send["reason"] == "no_vector"


class TestNoAppraisal:
    async def test_inbound_unchanged_and_outbound_fails_open(self, make_env):
        env = await make_env(
            order=("gates",),
            config={"gate": {
                "engagement": {"enforce": True, "threshold": 1.0},
                "send": {"min_salience": 1.0},
            }},
            responses=[_make_text_response("sure"), _make_text_response("nothing to add.")],
        )
        trace_id = await _say(env, "ok")
        assert sent_texts(env.pm, OutboundMessage) == ["sure"]

        task_trace = await _notify(env, "background check finished")
        assert sent_texts(env.pm, OutboundMessage) == ["sure", "nothing to add."]
        send = (await env.row(task_trace))["outcomes"]["send"]
        assert send["reason"] == "no_vector"
        assert send["vetoed"] is False
        assert "engagement" not in (await env.row(trace_id))["outcomes"]


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
            trace_id=trace_id, meta={"origin": "heartbeat"},
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

    async def test_thinking_suppressed_and_recorded(self, make_env):
        env = await make_env(
            config={"gate": {"send": {"allow_thinking": False}}},
            responses=[_make_text_response("answer", reasoning="pondering")],
        )
        trace_id = await _say(env, "think about it")

        row = await env.row(trace_id)
        assert row["outcomes"]["suppressed"] == {"thinking": True}
        assert sent_texts(env.pm, OutboundMessage) == ["answer"]


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


class TestBlocklist:
    async def test_immutable_gate_key_refused_by_set_settings(self, make_env):
        from corvidae.tool import Tool, ToolContext
        from corvidae.tools.settings import RuntimeSettingsPlugin

        env = await make_env()
        settings = RuntimeSettingsPlugin()
        await settings.on_init(pm=env.pm, config={"agent": {"immutable_settings": ["gate.send.enforce"]}})
        registry = []
        settings.register_tools(tool_registry=registry)
        tool = next(t for t in registry if (t.name if isinstance(t, Tool) else t.__name__) == "set_settings")
        fn = tool.fn if isinstance(tool, Tool) else tool

        reply = await fn({"gate.send.enforce": False}, ToolContext(
            channel=env.channel, tool_call_id="call_test", task_queue=None,
        ))

        assert "gate.send.enforce" in reply
        assert "immutable" in reply
        assert "gate.send.enforce" not in env.channel.runtime_overrides


class TestWithheldStillCritiqued:
    async def test_withheld_reminder_final_is_critiqued(self, make_env):
        """WP2.7's deferred assertion: a withheld response stays eligible."""
        from corvidae_cognition.critique import CritiquePlugin

        env = await make_env(
            config={"gate": {"send": {"min_salience": 1.0}}, "critique": {"sample_below_rate": 1.0}},
            responses=[_make_text_response("stretch time.")],
        )
        critique = CritiquePlugin()
        env.pm.register(critique, name="critique")
        await critique.on_init(pm=env.pm, config=env.config)
        critique._run_critique = AsyncMock()

        await _notify(env, "time to stretch", meta={"origin": "reminder"})
        await drain_task_queue(env.plugin)

        rows = await env.rows_by_type()
        assert [t for m, t in rows if m.get("role") == "assistant"] == ["unsent"]
        critique._run_critique.assert_awaited_once()
        args = critique._run_critique.await_args.args
        assert "stretch time." in args


class TestRegistration:
    def test_entry_point_registered(self):
        import tomllib
        from pathlib import Path

        pyproject = Path(__file__).resolve().parents[2] / "packages/corvidae-cognition/pyproject.toml"
        eps = tomllib.loads(pyproject.read_text())["project"]["entry-points"]["corvidae"]
        assert eps["gates"] == "corvidae_cognition.gates:GatePlugin"

    def test_depends_on_is_empty(self):
        from corvidae_cognition.gates import GatePlugin

        assert GatePlugin.depends_on == frozenset()


class TestReviewFindings:
    """WP2.9 review gate: bad tunable types, origin across tool cycles and
    LRU eviction, and records that survive json_patch's null-deletes."""

    async def test_unparseable_threshold_falls_back_and_admits(self, make_env, caplog):
        env = await make_env(responses=[_make_text_response("sure")])
        env.channel.runtime_overrides["gate.engagement.threshold"] = "high"
        env.channel.runtime_overrides["gate.engagement.enforce"] = "false"

        trace_id = await _say(env, "ok")

        assert sent_texts(env.pm, OutboundMessage) == ["sure"]
        engagement = (await env.row(trace_id))["outcomes"]["engagement"]
        assert engagement["threshold"] == 0.2  # the default
        assert engagement["enforced"] is False
        messages = [r.getMessage() for r in caplog.records]
        assert any("gate.engagement.threshold" in m and "'high'" in m and "number" in m for m in messages)

    async def test_string_false_does_not_enforce(self, make_env):
        env = await make_env(responses=[_make_text_response("sure")])
        env.channel.runtime_overrides["gate.engagement.enforce"] = "false"
        env.channel.runtime_overrides["gate.engagement.threshold"] = "1.0"

        trace_id = await _say(env, "ok")

        assert sent_texts(env.pm, OutboundMessage) == ["sure"]
        engagement = (await env.row(trace_id))["outcomes"]["engagement"]
        assert engagement["threshold"] == 1.0
        assert engagement["would_reject"] is True
        assert engagement["enforced"] is False

    async def test_bad_send_tunables_never_raise(self, make_env):
        env = await make_env(responses=[_make_text_response("nothing to add.")])
        env.channel.runtime_overrides["gate.send.min_salience"] = ["nope"]
        env.channel.runtime_overrides["gate.send.enforce"] = {"x": 1}

        trace_id = await _notify(env, "background check finished")

        # Defaults apply (0.15, enforce true); this text scores above 0.15.
        assert sent_texts(env.pm, OutboundMessage) == ["nothing to add."]
        send = (await env.row(trace_id))["outcomes"]["send"]
        assert send["min_salience"] == 0.15
        assert send["enforced"] is True

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

    async def test_evicted_user_trace_reads_origin_from_exchange_log(self, make_env):
        env = await make_env(config={"gate": {"send": {"min_salience": 1.0}}})
        trace_id = mint_trace_id()
        outcome = env.pm.get_plugin("outcome_log")
        await outcome.record_exchange(trace_id, env.channel.id, origin="user")
        assert trace_id not in env.gates._admitted  # as if evicted

        results = await env.pm.ahook.should_send_response(
            channel=env.channel, text="ok.", emission="final",
            trace_id=trace_id, meta={},
        )

        assert resolve_reject_wins(results) is None
        row = await env.row(trace_id)
        assert row["outcomes"]["send"]["origin"] == "user"
        assert row["origin"] == "user"  # not relabelled

    async def test_no_vector_record_omits_salience(self, make_env):
        env = await make_env(order=("gates",), responses=[_make_text_response("nothing to add.")])
        trace_id = await _notify(env, "background check finished")
        send = (await env.row(trace_id))["outcomes"]["send"]
        assert send["reason"] == "no_vector"
        assert "salience" not in send
