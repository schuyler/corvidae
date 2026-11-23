"""Red tests for WP2.1 — Exchange keys, origin stamping, and enriched hooks.

Phase 2A design (plans/implementation/phase-2.md, "Phase 2A — Design",
WP2.1 section, PASSED re-review #1 at HEAD bb03fa5). Covers:

  1. mint_trace_id() shape.
  2. USER message: the transport-supplied key reaches the gate hook,
     admitted -> outcome_log records origin='user', message_rowid populated
     after the turn.
  3. Gate plugin returns False -> on_message_rejected fires, outcome_log
     records with null message_rowid and outcomes {"gate": "rejected"}.
  4. Tool cycle: Task stamped with key+origin; tool-result notification
     turn inherits the same key (no second exchange row); final
     on_agent_response carries originating_text.
  5. Standalone notification (no tool_call_id, no annotations key): key minted
     at dequeue, one on_message_persisted firing, origin='task'.
  6. Mid-exchange tool-result rows never fire on_message_persisted (count
     firings) — folded into test 4's assertion and a dedicated multi-tool
     count test.
  7. before_agent_turn receives (channel, trace_id, origin); retrieval
     profile reaches the trace record under the key.
  8. usage_log rows carry the trace id (attribution wiring).

All tests are new and live only in this file, per the WP2.1 red-author's
instructions (no edits to shared conftest/other test files).
"""

import asyncio
import re
from unittest.mock import ANY, AsyncMock, MagicMock

import pytest

from corvidae.trace import mint_trace_id
from corvidae.hooks import CorvidaePlugin, create_plugin_manager, hookimpl

from corvidae.outbound import OutboundMessage
from helpers import build_plugin_and_channel, drain, drain_task_queue, sent
from tests.cognition.trace_helpers import add_trace, outcomes_file, read_outcomes
from llm_response_fixtures import (
    _make_text_response,
    _make_tool_call,
    _make_tool_call_response,
)


# ---------------------------------------------------------------------------
# Local helpers (do not import shared fixtures beyond tests/helpers.py)
# ---------------------------------------------------------------------------


async def _setup_outcome_log(pm, db):
    """Register a real OutcomeLogPlugin against the given db and pm.

    Mirrors tests/cognition/test_outcome_log.py's _setup, duplicated locally per
    the red-author's file-isolation instructions.
    """
    from corvidae_cognition.outcome_log import OutcomeLogPlugin

    # outcome_log depends on the trace protocol plugin.
    add_trace(pm)
    plugin = OutcomeLogPlugin()
    pm.register(plugin, name="outcome_log")
    await plugin.on_init(pm, {})
    await plugin.on_start({})
    # The writers skip channels the registry never held (#261), so the
    # channel these tests write rows for must be registered.
    registry = pm.get_plugin("registry")
    if registry is not None:
        registry.get_or_create("irc", "#general")
    return plugin


async def _setup_usage_log(pm):
    from corvidae.metrics import UsageLogPlugin

    plugin = UsageLogPlugin()
    pm.register(plugin, name="usage_log")
    await plugin.on_init(pm, {})
    await plugin.on_start({})
    return plugin


def _trace_lines(db_path, trace_id) -> list[dict]:
    """The outcomes-file lines written for ``trace_id``, in order."""
    return [
        line for line in read_outcomes(outcomes_file(db_path))
        if line["trace_id"] == trace_id
    ]


class _KeyCapturePlugin:
    """Gate plugin that records the trace_id it was handed and, when
    configured, vetoes the message."""

    def __init__(self, reject: bool = False):
        self.reject = reject
        self.seen_keys: list[str] = []
        self.seen_calls: list[dict] = []

    @hookimpl
    async def should_process_message(self, channel, sender, text, trace_id):
        self.seen_calls.append(
            {"channel": channel, "sender": sender, "text": text, "trace_id": trace_id}
        )
        if trace_id is not None:
            self.seen_keys.append(trace_id)
        return False if self.reject else None


class _AdmissionRecorderPlugin:
    """Records on_message_admitted / on_message_rejected / on_message_persisted firings."""

    def __init__(self):
        self.admitted: list[dict] = []
        self.rejected: list[dict] = []
        self.persisted: list[dict] = []

    @hookimpl
    async def on_message_admitted(self, channel, trace_id, sender, text):
        self.admitted.append(
            {"channel": channel, "trace_id": trace_id, "sender": sender, "text": text}
        )

    @hookimpl
    async def on_message_rejected(self, channel, trace_id, sender, text):
        self.rejected.append(
            {"channel": channel, "trace_id": trace_id, "sender": sender, "text": text}
        )

    @hookimpl
    async def on_message_persisted(self, channel, trace_id, rowid, text, annotations):
        self.persisted.append(
            {
                "channel": channel,
                "trace_id": trace_id,
                "rowid": rowid,
                "text": text,
                "annotations": annotations,
            }
        )


class _BeforeTurnRecorderPlugin:
    """Records the (channel, trace_id, annotations) tuple before_agent_turn receives."""

    def __init__(self):
        self.calls: list[dict] = []

    @hookimpl
    async def before_agent_turn(self, channel, trace_id, annotations):
        self.calls.append(
            {"channel": channel, "trace_id": trace_id, "annotations": annotations}
        )


class _AgentResponseRecorderPlugin:
    """Records the enriched on_agent_response call's kwargs."""

    def __init__(self):
        self.calls: list[dict] = []

    @hookimpl
    async def on_agent_response(
        self,
        channel,
        request_text,
        response_text,
        trace_id,
        annotations,
        logprobs,
    ):
        self.calls.append(
            {
                "channel": channel,
                "request_text": request_text,
                "response_text": response_text,
                "trace_id": trace_id,
                "annotations": annotations,
                "logprobs": logprobs,
            }
        )


# ---------------------------------------------------------------------------
# 1. mint_trace_id shape
# ---------------------------------------------------------------------------


class TestMintTraceId:
    def test_returns_hex_hex_shape(self):
        from corvidae.agent import mint_trace_id

        key = mint_trace_id()
        assert re.fullmatch(r"[0-9a-f]+-[0-9a-f]{12}", key), (
            f"expected '<hex-time>-<hex12>' shape, got {key!r}"
        )

    def test_time_prefix_is_monotone_nondecreasing(self):
        from corvidae.agent import mint_trace_id

        first = mint_trace_id()
        time_a = int(first.split("-")[0], 16)
        second = mint_trace_id()
        time_b = int(second.split("-")[0], 16)
        assert time_b >= time_a

    def test_two_calls_produce_distinct_keys(self):
        from corvidae.agent import mint_trace_id

        assert mint_trace_id() != mint_trace_id()


# ---------------------------------------------------------------------------
# 2. USER message admission: key minted before gate, outcome_log records,
#    message_rowid populated after the turn.
# ---------------------------------------------------------------------------


class TestUserMessageAdmission:
    async def test_gate_hook_receives_trace_id(self):
        """should_process_message must be called with the trace_id the
        transport minted and passed to on_message (the Agent no longer mints
        ids for USER messages)."""
        plugin, channel, db = await build_plugin_and_channel()

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=_make_text_response("hi"))
        plugin._client = mock_client
        plugin._mark_started()

        gate = _KeyCapturePlugin()
        plugin.pm.register(gate, name="key_capture")

        tid = mint_trace_id()
        await plugin.on_message(channel=channel, sender="user", text="hello", trace_id=tid)
        await drain(plugin, channel)

        assert len(gate.seen_calls) == 1
        assert gate.seen_calls[0]["trace_id"] == tid

        await db.close()

    async def test_admitted_message_is_recorded_with_user_origin(self, tmp_path):
        db_path = str(tmp_path / "sessions.db")
        plugin, channel, db = await build_plugin_and_channel(db_path=db_path)
        await _setup_outcome_log(plugin.pm, db)

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=_make_text_response("hi there"))
        plugin._client = mock_client
        plugin._mark_started()

        gate = _KeyCapturePlugin()
        plugin.pm.register(gate, name="key_capture")

        await plugin.on_message(channel=channel, sender="user", text="hello", trace_id=mint_trace_id())
        await drain(plugin, channel)

        assert len(gate.seen_keys) == 1
        key = gate.seen_keys[0]

        lines = _trace_lines(db_path, key)
        admitted = [l for l in lines if l["record"] == {"event": "admitted"}]
        assert len(admitted) == 1, "expected an admitted record for the message"
        assert admitted[0]["origin"] == "user"

        await db.close()

    async def test_message_rowid_populated_after_turn_matches_persisted_user_row(self, tmp_path):
        db_path = str(tmp_path / "sessions.db")
        plugin, channel, db = await build_plugin_and_channel(db_path=db_path)
        await _setup_outcome_log(plugin.pm, db)

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=_make_text_response("hi there"))
        plugin._client = mock_client
        plugin._mark_started()

        gate = _KeyCapturePlugin()
        plugin.pm.register(gate, name="key_capture")

        await plugin.on_message(channel=channel, sender="user", text="hello", trace_id=mint_trace_id())
        await drain(plugin, channel)

        key = gate.seen_keys[0]
        [persisted_line] = [
            l for l in _trace_lines(db_path, key) if l["record"].get("event") == "persisted"
        ]
        message_rowid = persisted_line["record"]["message_rowid"]
        assert message_rowid is not None

        async with db.execute(
            "SELECT id, message FROM message_log WHERE id = ?", (message_rowid,)
        ) as cursor:
            db_row = await cursor.fetchone()
        assert db_row is not None
        import json as _json
        persisted = _json.loads(db_row[1])
        assert persisted.get("role") == "user"
        assert persisted.get("content") == "hello"

        await db.close()


# ---------------------------------------------------------------------------
# 3. Gate rejection: on_message_rejected fires, outcome_log records the
#    rejection and never a message_rowid.
# ---------------------------------------------------------------------------


class TestGateRejection:
    async def test_rejected_message_fires_on_message_rejected(self):
        plugin, channel, db = await build_plugin_and_channel()

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=_make_text_response("should not run"))
        plugin._client = mock_client
        plugin._mark_started()

        gate = _KeyCapturePlugin(reject=True)
        plugin.pm.register(gate, name="key_capture")
        recorder = _AdmissionRecorderPlugin()
        plugin.pm.register(recorder, name="admission_recorder")

        tid = mint_trace_id()
        await plugin.on_message(channel=channel, sender="user", text="blocked", trace_id=tid)
        await drain(plugin, channel)

        assert len(recorder.rejected) == 1
        assert recorder.rejected[0]["trace_id"] == tid
        assert len(recorder.admitted) == 0
        assert sent(plugin.pm, OutboundMessage) == []

        await db.close()

    async def test_rejected_message_is_recorded_with_no_rowid(self, tmp_path):
        db_path = str(tmp_path / "sessions.db")
        plugin, channel, db = await build_plugin_and_channel(db_path=db_path)
        await _setup_outcome_log(plugin.pm, db)

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=_make_text_response("should not run"))
        plugin._client = mock_client
        plugin._mark_started()

        gate = _KeyCapturePlugin(reject=True)
        plugin.pm.register(gate, name="key_capture")

        await plugin.on_message(channel=channel, sender="user", text="blocked", trace_id=mint_trace_id())
        await drain(plugin, channel)

        assert len(gate.seen_keys) == 1
        key = gate.seen_keys[0]

        records = [l["record"] for l in _trace_lines(db_path, key)]
        assert {"event": "rejected"} in records, (
            "rejected exchange must still be recorded (§3.2 offline corpus)"
        )
        # A rejected message is never persisted, so no rowid is ever recorded.
        assert [r for r in records if r.get("event") == "persisted"] == []

        await db.close()


# ---------------------------------------------------------------------------
# 4. Tool cycle propagation
# ---------------------------------------------------------------------------


class TestToolCyclePropagation:
    async def test_task_stamped_with_trace_id_and_meta(self):
        """_dispatch_tool_calls must stamp Task.trace_id/annotations from the
        current item so the tool cycle carries the exchange forward."""
        plugin, channel, db = await build_plugin_and_channel()

        captured_tasks = []
        real_dispatch = plugin._dispatch_tool_calls

        async def spy_dispatch(tool_calls, channel):
            task_queue = plugin.pm.get_plugin("task").task_queue
            orig_enqueue = task_queue.enqueue

            async def capture_enqueue(task):
                captured_tasks.append(task)
                await orig_enqueue(task)

            task_queue.enqueue = capture_enqueue
            try:
                await real_dispatch(tool_calls, channel)
            finally:
                task_queue.enqueue = orig_enqueue

        plugin._dispatch_tool_calls = spy_dispatch

        async def my_plugin_tool(query: str) -> str:
            """A plugin-provided tool."""
            return "tool output"

        plugin._tools = {"my_plugin_tool": my_plugin_tool}
        from corvidae.tool import tool_to_schema

        plugin._tool_schemas = [tool_to_schema(my_plugin_tool)]

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(
            side_effect=[
                _make_tool_call_response(
                    [_make_tool_call("call_1", "my_plugin_tool", {"query": "q"})]
                ),
                _make_text_response("final response"),
            ]
        )
        plugin._client = mock_client
        plugin._mark_started()

        gate = _KeyCapturePlugin()
        plugin.pm.register(gate, name="key_capture")

        await plugin.on_message(channel=channel, sender="user", text="use the tool", trace_id=mint_trace_id())
        await drain(plugin, channel)
        task_plugin = plugin.pm.get_plugin("task")
        await task_plugin.task_queue.queue.join()
        await asyncio.sleep(0)
        await drain(plugin, channel)

        assert len(captured_tasks) == 1
        task = captured_tasks[0]
        assert task.trace_id is not None
        assert task.trace_id == gate.seen_keys[0]
        # Core stamps only the attribution annotations forward; it carries no
        # 'user' origin string itself (R1 — origin is a plugin-side
        # convention over annotations["origin"], never set by core for USER items).
        assert task.annotations == {}

        await task_plugin.on_stop()
        await db.close()

    async def test_tool_result_turn_inherits_same_key_no_second_exchange_record(self, tmp_path):
        db_path = str(tmp_path / "sessions.db")
        plugin, channel, db = await build_plugin_and_channel(db_path=db_path)
        outcome_log = await _setup_outcome_log(plugin.pm, db)

        async def my_plugin_tool(query: str) -> str:
            """A plugin-provided tool."""
            return "tool output"

        plugin._tools = {"my_plugin_tool": my_plugin_tool}
        from corvidae.tool import tool_to_schema

        plugin._tool_schemas = [tool_to_schema(my_plugin_tool)]

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(
            side_effect=[
                _make_tool_call_response(
                    [_make_tool_call("call_1", "my_plugin_tool", {"query": "q"})]
                ),
                _make_text_response("final response after tool"),
            ]
        )
        plugin._client = mock_client
        plugin._mark_started()

        gate = _KeyCapturePlugin()
        plugin.pm.register(gate, name="key_capture")

        await plugin.on_message(channel=channel, sender="user", text="use the tool", trace_id=mint_trace_id())
        await drain(plugin, channel)
        task_plugin = plugin.pm.get_plugin("task")
        await task_plugin.task_queue.queue.join()
        await asyncio.sleep(0)
        await drain(plugin, channel)

        key = gate.seen_keys[0]

        events = [l["record"].get("event") for l in _trace_lines(db_path, key)]
        assert events.count("admitted") == 1, (
            "tool-result turn must not admit a second exchange under the key"
        )
        assert events.count("persisted") == 1, (
            "tool-result turn must not persist a second originating row"
        )

        await task_plugin.on_stop()
        await db.close()

    async def test_final_on_agent_response_request_text_stays_last_turn_text(self):
        """request_text keeps its legacy semantics after the rename: the
        last tool-result turn's text, never the exchange's true originating
        text. (The positive half of this behavior -- that the true
        originating text IS recoverable through the same tool cycle -- now
        lives at its migrated location, AppraisalPlugin.get_originating_text;
        see test_appraisal_stage2.py::TestOriginatingTextLRU.)"""
        plugin, channel, db = await build_plugin_and_channel(mock_on_agent_response=False)

        recorder = _AgentResponseRecorderPlugin()
        plugin.pm.register(recorder, name="response_recorder")

        async def my_plugin_tool(query: str) -> str:
            """A plugin-provided tool."""
            return "tool output"

        plugin._tools = {"my_plugin_tool": my_plugin_tool}
        from corvidae.tool import tool_to_schema

        plugin._tool_schemas = [tool_to_schema(my_plugin_tool)]

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(
            side_effect=[
                _make_tool_call_response(
                    [_make_tool_call("call_1", "my_plugin_tool", {"query": "q"})]
                ),
                _make_text_response("final response after tool"),
            ]
        )
        plugin._client = mock_client
        plugin._mark_started()

        await plugin.on_message(channel=channel, sender="user", text="the original user text", trace_id=mint_trace_id())
        await drain(plugin, channel)
        task_plugin = plugin.pm.get_plugin("task")
        await task_plugin.task_queue.queue.join()
        await asyncio.sleep(0)
        await drain(plugin, channel)

        assert len(recorder.calls) == 1
        # request_text (legacy semantics) is the tool-result turn's text,
        # never the exchange's true originating text -- this is exactly the
        # mis-pairing bug the plugin-side originating-text LRU exists to fix.
        assert recorder.calls[0]["request_text"] != "the original user text"
        assert recorder.calls[0]["trace_id"] is not None

        await task_plugin.on_stop()
        await db.close()


# ---------------------------------------------------------------------------
# 5/6. Standalone notification + persisted-firing discipline
# ---------------------------------------------------------------------------


class TestStandaloneNotificationAndPersistedFiring:
    async def test_standalone_notification_mints_key_at_dequeue_with_task_origin(self):
        plugin, channel, db = await build_plugin_and_channel()

        recorder = _AdmissionRecorderPlugin()
        plugin.pm.register(recorder, name="admission_recorder")

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=_make_text_response("notified"))
        plugin._client = mock_client
        plugin._mark_started()

        await plugin.on_notify(
            channel=channel, source="task", text="standalone event",
            tool_call_id=None, annotations=None,
        )
        await drain(plugin, channel)

        assert len(recorder.persisted) == 1
        assert recorder.persisted[0]["trace_id"] is not None
        # origin='task' is recorded for this key by the test below.

        await db.close()

    async def test_standalone_notification_fires_on_message_persisted_exactly_once(self):
        plugin, channel, db = await build_plugin_and_channel()

        recorder = _AdmissionRecorderPlugin()
        plugin.pm.register(recorder, name="admission_recorder")

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=_make_text_response("notified"))
        plugin._client = mock_client
        plugin._mark_started()

        await plugin.on_notify(
            channel=channel, source="task", text="standalone event",
            tool_call_id=None, annotations=None,
        )
        await drain(plugin, channel)

        assert len(recorder.persisted) == 1

        await db.close()

    async def test_standalone_notification_is_recorded_with_task_origin(self, tmp_path):
        db_path = str(tmp_path / "sessions.db")
        plugin, channel, db = await build_plugin_and_channel(db_path=db_path)
        await _setup_outcome_log(plugin.pm, db)

        recorder = _AdmissionRecorderPlugin()
        plugin.pm.register(recorder, name="admission_recorder")

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=_make_text_response("notified"))
        plugin._client = mock_client
        plugin._mark_started()

        await plugin.on_notify(
            channel=channel, source="task", text="standalone event",
            tool_call_id=None, annotations=None,
        )
        await drain(plugin, channel)

        key = recorder.persisted[0]["trace_id"]
        [persisted_line] = [
            l for l in _trace_lines(db_path, key) if l["record"].get("event") == "persisted"
        ]
        assert persisted_line["origin"] == "task"

        await db.close()

    async def test_mid_exchange_tool_result_rows_never_fire_on_message_persisted(self):
        """A tool cycle with N tool results fires on_message_persisted exactly
        once total (on the originating USER row), never per tool-result row."""
        plugin, channel, db = await build_plugin_and_channel()

        recorder = _AdmissionRecorderPlugin()
        plugin.pm.register(recorder, name="admission_recorder")

        async def tool_one() -> str:
            """First tool."""
            return "result one"

        async def tool_two() -> str:
            """Second tool."""
            return "result two"

        plugin._tools = {"tool_one": tool_one, "tool_two": tool_two}
        from corvidae.tool import tool_to_schema

        plugin._tool_schemas = [tool_to_schema(tool_one), tool_to_schema(tool_two)]

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(
            side_effect=[
                _make_tool_call_response(
                    [
                        _make_tool_call("call_1", "tool_one", {}),
                        _make_tool_call("call_2", "tool_two", {}),
                    ]
                ),
                _make_text_response("final response"),
            ]
        )
        plugin._client = mock_client
        plugin._mark_started()

        await plugin.on_message(channel=channel, sender="user", text="use two tools", trace_id=mint_trace_id())
        await drain(plugin, channel)
        task_plugin = plugin.pm.get_plugin("task")
        await task_plugin.task_queue.queue.join()
        await asyncio.sleep(0)
        await drain(plugin, channel)

        # Exactly one on_message_persisted firing for the whole exchange,
        # despite two tool-result notifications passing through the queue.
        assert len(recorder.persisted) == 1

        await task_plugin.on_stop()
        await db.close()


# ---------------------------------------------------------------------------
# 7. before_agent_turn enrichment + retrieval profile persistence
# ---------------------------------------------------------------------------


class TestBeforeAgentTurnEnrichment:
    async def test_before_agent_turn_receives_channel_trace_id_meta(self):
        plugin, channel, db = await build_plugin_and_channel()

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=_make_text_response("hi"))
        plugin._client = mock_client
        plugin._mark_started()

        gate = _KeyCapturePlugin()
        plugin.pm.register(gate, name="key_capture")
        recorder = _BeforeTurnRecorderPlugin()
        plugin.pm.register(recorder, name="before_turn_recorder")

        await plugin.on_message(channel=channel, sender="user", text="hello", trace_id=mint_trace_id())
        await drain(plugin, channel)

        assert len(recorder.calls) == 1
        assert recorder.calls[0]["trace_id"] == gate.seen_keys[0]
        # Core carries no 'user' origin string for a USER item -- origin is
        # a plugin-side convention over annotations["origin"] (R1); core's annotations
        # for a directly-admitted USER message stays empty.
        assert recorder.calls[0]["annotations"] == {}

        await db.close()

    async def test_retrieval_profile_reaches_the_trace_record_under_the_key(self, tmp_path):
        """MemoryPlugin.before_agent_turn (updated for WP2.1 point 8) reports
        retrieval_top_score/retrieval_hit_count through on_trace_record under
        the trace id; outcome_log writes them to the outcomes file."""
        db_path = str(tmp_path / "sessions.db")
        plugin, channel, db = await build_plugin_and_channel(db_path=db_path)
        await _setup_outcome_log(plugin.pm, db)

        gate = _KeyCapturePlugin()
        plugin.pm.register(gate, name="key_capture")

        # A minimal before_agent_turn implementer standing in for MemoryPlugin,
        # firing the same on_trace_record call point-8 requires. This
        # avoids pulling in the full MemoryPlugin/embedding stack for this
        # narrow assertion (retrieval-profile persistence under the key),
        # while still proving the enriched hookspec carries what a real
        # consumer needs to perform the report.
        pm = plugin.pm

        class _StandInRetrievalProfiler:
            @hookimpl
            async def before_agent_turn(self, channel, trace_id, annotations):
                if trace_id is None:
                    return
                await pm.ahook.on_trace_record(
                    trace_id=trace_id, channel_id=channel.id, source="memory",
                    origin=None,
                    record={"retrieval_top_score": 0.75, "retrieval_hit_count": 2},
                )

        plugin.pm.register(_StandInRetrievalProfiler(), name="stand_in_profiler")

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=_make_text_response("hi"))
        plugin._client = mock_client
        plugin._mark_started()

        await plugin.on_message(channel=channel, sender="user", text="hello", trace_id=mint_trace_id())
        await drain(plugin, channel)

        key = gate.seen_keys[0]
        memory_records = [
            l["record"] for l in _trace_lines(db_path, key) if l["source"] == "memory"
        ]
        assert memory_records == [{"retrieval_top_score": 0.75, "retrieval_hit_count": 2}]

        await db.close()


# ---------------------------------------------------------------------------
# 8. usage_log carries the trace id (attribution wiring)
# ---------------------------------------------------------------------------


class TestUsageLogAttributionWiring:
    async def test_usage_log_row_carries_trace_id_via_attribution(self):
        """Every LLM call inside _process_queue_item_attributed must run
        under an attribution contextvar carrying trace_id, wired via
        the widened set_attribution(...) call ahead of the attributed body
        (design fix for the ordering finding)."""
        from corvidae.attribution import get_attribution

        plugin, channel, db = await build_plugin_and_channel()
        await _setup_usage_log(plugin.pm)

        gate = _KeyCapturePlugin()
        plugin.pm.register(gate, name="key_capture")

        captured_attribution = {}

        async def fake_chat(messages, **kwargs):
            captured_attribution.update(get_attribution())
            return _make_text_response("hi")["choices"][0]["message"] and _make_text_response("hi")

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(side_effect=fake_chat)
        plugin._client = mock_client
        plugin._mark_started()

        await plugin.on_message(channel=channel, sender="user", text="hello", trace_id=mint_trace_id())
        await drain(plugin, channel)

        assert captured_attribution.get("trace_id") == gate.seen_keys[0]

    async def test_usage_log_table_row_has_trace_id_column_populated(self):
        """End-to-end: firing on_llm_response with the real attribution
        snapshot writes a usage_log row whose trace_id column is set
        (renamed from trace_id via the lazy ALTER TABLE migration)."""
        from corvidae.attribution import get_attribution, reset_attribution, set_attribution

        plugin, channel, db = await build_plugin_and_channel()
        usage_log = await _setup_usage_log(plugin.pm)

        gate = _KeyCapturePlugin()
        plugin.pm.register(gate, name="key_capture")

        captured_attribution = {}

        async def fake_chat(messages, **kwargs):
            # Simulate the LLMClient observer firing on_llm_response with the
            # attribution snapshot at call time, as _HookObserver does.
            attribution = get_attribution()
            captured_attribution.update(attribution)
            await plugin.pm.ahook.on_llm_response(
                profile="main",
                model="test-model",
                request_id="req-1",
                usage={"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
                latency_ms=1.0,
                attribution=attribution,
                error=None,
                timings=None,
                started_at=1000.0,
                total_ms=1.0,
                attempts=1,
                message_count=1,
                tool_count=0,
                lease_wait_ms=None,
            )
            return _make_text_response("hi")

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(side_effect=fake_chat)
        plugin._client = mock_client
        plugin._mark_started()

        await plugin.on_message(channel=channel, sender="user", text="hello", trace_id=mint_trace_id())
        await drain(plugin, channel)

        assert captured_attribution.get("trace_id") == gate.seen_keys[0]

        async with db.execute(
            "SELECT trace_id FROM usage_log WHERE request_id = ?", ("req-1",)
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None
        assert row[0] == gate.seen_keys[0]

        await db.close()
