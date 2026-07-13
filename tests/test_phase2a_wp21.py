"""Red tests for WP2.1 — Exchange keys, origin stamping, and enriched hooks.

Phase 2A design (plans/implementation/phase-2.md, "Phase 2A — Design",
WP2.1 section, PASSED re-review #1 at HEAD bb03fa5). Covers:

  1. mint_exchange_key() shape.
  2. USER message: key minted before the gate, gate hook receives it,
     admitted -> exchange_log row origin='user', message_rowid populated
     after the turn.
  3. Gate plugin returns False -> on_message_rejected fires, exchange_log
     row exists with null message_rowid and outcomes {"gate": "rejected"}.
  4. Tool cycle: Task stamped with key+origin; tool-result notification
     turn inherits the same key (no second exchange row); final
     on_agent_response carries originating_text.
  5. Standalone notification (no tool_call_id, no meta key): key minted
     at dequeue, one on_message_persisted firing, origin='task'.
  6. Mid-exchange tool-result rows never fire on_message_persisted (count
     firings) — folded into test 4's assertion and a dedicated multi-tool
     count test.
  7. before_agent_turn receives (channel, exchange_key, origin); retrieval
     profile lands in exchange_log under the key.
  8. usage_log rows carry the exchange key (attribution wiring).
  9. upsert_exchange write-order independence.
  10. Atomic JSON merge for outcomes/appraisal columns (json_patch).

All tests are new and live only in this file, per the WP2.1 red-author's
instructions (no edits to shared conftest/other test files).
"""

import asyncio
import re
from unittest.mock import ANY, AsyncMock, MagicMock

import pytest

from corvidae.hooks import CorvidaePlugin, create_plugin_manager, hookimpl

from helpers import build_plugin_and_channel, drain, drain_task_queue
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

    Mirrors tests/test_outcome_log.py's _setup, duplicated locally per
    the red-author's file-isolation instructions.
    """
    from corvidae.outcome_log import OutcomeLogPlugin

    plugin = OutcomeLogPlugin()
    pm.register(plugin, name="outcome_log")
    await plugin.on_init(pm, {})
    await plugin.on_start({})
    return plugin


async def _setup_usage_log(pm):
    from corvidae.metrics import UsageLogPlugin

    plugin = UsageLogPlugin()
    pm.register(plugin, name="usage_log")
    await plugin.on_init(pm, {})
    await plugin.on_start({})
    return plugin


async def _fetch_exchange_row(db, exchange_key):
    async with db.execute(
        "SELECT channel_id, origin, message_rowid, outcomes, appraisal "
        "FROM exchange_log WHERE exchange_key = ?",
        (exchange_key,),
    ) as cursor:
        return await cursor.fetchone()


class _KeyCapturePlugin:
    """Gate plugin that records the exchange_key it was handed and, when
    configured, vetoes the message."""

    def __init__(self, reject: bool = False):
        self.reject = reject
        self.seen_keys: list[str] = []
        self.seen_calls: list[dict] = []

    @hookimpl
    async def should_process_message(self, channel, sender, text, exchange_key):
        self.seen_calls.append(
            {"channel": channel, "sender": sender, "text": text, "exchange_key": exchange_key}
        )
        if exchange_key is not None:
            self.seen_keys.append(exchange_key)
        return False if self.reject else None


class _AdmissionRecorderPlugin:
    """Records on_message_admitted / on_message_rejected / on_message_persisted firings."""

    def __init__(self):
        self.admitted: list[dict] = []
        self.rejected: list[dict] = []
        self.persisted: list[dict] = []

    @hookimpl
    async def on_message_admitted(self, channel, exchange_key, sender, text):
        self.admitted.append(
            {"channel": channel, "exchange_key": exchange_key, "sender": sender, "text": text}
        )

    @hookimpl
    async def on_message_rejected(self, channel, exchange_key, sender, text):
        self.rejected.append(
            {"channel": channel, "exchange_key": exchange_key, "sender": sender, "text": text}
        )

    @hookimpl
    async def on_message_persisted(self, channel, exchange_key, rowid):
        self.persisted.append(
            {"channel": channel, "exchange_key": exchange_key, "rowid": rowid}
        )


class _BeforeTurnRecorderPlugin:
    """Records the (channel, exchange_key, origin) tuple before_agent_turn receives."""

    def __init__(self):
        self.calls: list[dict] = []

    @hookimpl
    async def before_agent_turn(self, channel, exchange_key, origin):
        self.calls.append(
            {"channel": channel, "exchange_key": exchange_key, "origin": origin}
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
        exchange_key,
        origin,
        originating_text,
        logprobs,
        withheld,
    ):
        self.calls.append(
            {
                "channel": channel,
                "request_text": request_text,
                "response_text": response_text,
                "exchange_key": exchange_key,
                "origin": origin,
                "originating_text": originating_text,
                "logprobs": logprobs,
                "withheld": withheld,
            }
        )


# ---------------------------------------------------------------------------
# 1. mint_exchange_key shape
# ---------------------------------------------------------------------------


class TestMintExchangeKey:
    def test_returns_hex_hex_shape(self):
        from corvidae.agent import mint_exchange_key

        key = mint_exchange_key()
        assert re.fullmatch(r"[0-9a-f]+-[0-9a-f]{12}", key), (
            f"expected '<hex-time>-<hex12>' shape, got {key!r}"
        )

    def test_time_prefix_is_monotone_nondecreasing(self):
        from corvidae.agent import mint_exchange_key

        first = mint_exchange_key()
        time_a = int(first.split("-")[0], 16)
        second = mint_exchange_key()
        time_b = int(second.split("-")[0], 16)
        assert time_b >= time_a

    def test_two_calls_produce_distinct_keys(self):
        from corvidae.agent import mint_exchange_key

        assert mint_exchange_key() != mint_exchange_key()


# ---------------------------------------------------------------------------
# 2. USER message admission: key minted before gate, exchange_log row,
#    message_rowid populated after the turn.
# ---------------------------------------------------------------------------


class TestUserMessageAdmission:
    async def test_gate_hook_receives_exchange_key(self):
        """should_process_message must be called with a non-None exchange_key
        for a USER message, minted before the gate fires."""
        plugin, channel, db = await build_plugin_and_channel()

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=_make_text_response("hi"))
        plugin._client = mock_client

        gate = _KeyCapturePlugin()
        plugin.pm.register(gate, name="key_capture")

        await plugin.on_message(channel=channel, sender="user", text="hello")
        await drain(plugin, channel)

        assert len(gate.seen_calls) == 1
        assert gate.seen_calls[0]["exchange_key"] is not None
        assert isinstance(gate.seen_calls[0]["exchange_key"], str)

        await db.close()

    async def test_admitted_message_creates_exchange_log_row_with_user_origin(self):
        plugin, channel, db = await build_plugin_and_channel()
        await _setup_outcome_log(plugin.pm, db)

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=_make_text_response("hi there"))
        plugin._client = mock_client

        gate = _KeyCapturePlugin()
        plugin.pm.register(gate, name="key_capture")

        await plugin.on_message(channel=channel, sender="user", text="hello")
        await drain(plugin, channel)

        assert len(gate.seen_keys) == 1
        key = gate.seen_keys[0]

        row = await _fetch_exchange_row(db, key)
        assert row is not None, "expected an exchange_log row for the admitted message"
        channel_id, origin, message_rowid, outcomes, appraisal = row
        assert origin == "user"

        await db.close()

    async def test_message_rowid_populated_after_turn_matches_persisted_user_row(self):
        plugin, channel, db = await build_plugin_and_channel()
        await _setup_outcome_log(plugin.pm, db)

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=_make_text_response("hi there"))
        plugin._client = mock_client

        gate = _KeyCapturePlugin()
        plugin.pm.register(gate, name="key_capture")

        await plugin.on_message(channel=channel, sender="user", text="hello")
        await drain(plugin, channel)

        key = gate.seen_keys[0]
        row = await _fetch_exchange_row(db, key)
        message_rowid = row[2]
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
# 3. Gate rejection: on_message_rejected fires, exchange_log row exists
#    with null message_rowid and outcomes {"gate": "rejected"}.
# ---------------------------------------------------------------------------


class TestGateRejection:
    async def test_rejected_message_fires_on_message_rejected(self):
        plugin, channel, db = await build_plugin_and_channel()

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=_make_text_response("should not run"))
        plugin._client = mock_client

        gate = _KeyCapturePlugin(reject=True)
        plugin.pm.register(gate, name="key_capture")
        recorder = _AdmissionRecorderPlugin()
        plugin.pm.register(recorder, name="admission_recorder")

        await plugin.on_message(channel=channel, sender="user", text="blocked")
        await drain(plugin, channel)

        assert len(recorder.rejected) == 1
        assert recorder.rejected[0]["exchange_key"] is not None
        assert len(recorder.admitted) == 0
        plugin.pm.ahook.send_message.assert_not_awaited()

        await db.close()

    async def test_rejected_message_exchange_log_row_has_null_rowid_and_gate_outcome(self):
        plugin, channel, db = await build_plugin_and_channel()
        await _setup_outcome_log(plugin.pm, db)

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=_make_text_response("should not run"))
        plugin._client = mock_client

        gate = _KeyCapturePlugin(reject=True)
        plugin.pm.register(gate, name="key_capture")

        await plugin.on_message(channel=channel, sender="user", text="blocked")
        await drain(plugin, channel)

        assert len(gate.seen_keys) == 1
        key = gate.seen_keys[0]

        row = await _fetch_exchange_row(db, key)
        assert row is not None, "rejected exchange must still get a row (§3.2 offline corpus)"
        channel_id, origin, message_rowid, outcomes, appraisal = row
        assert message_rowid is None

        import json as _json
        outcomes_dict = _json.loads(outcomes) if outcomes else {}
        assert outcomes_dict.get("gate") == "rejected"

        await db.close()


# ---------------------------------------------------------------------------
# 4. Tool cycle propagation
# ---------------------------------------------------------------------------


class TestToolCyclePropagation:
    async def test_task_stamped_with_exchange_key_and_origin(self):
        """_dispatch_tool_calls must stamp Task.exchange_key/origin from the
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

        gate = _KeyCapturePlugin()
        plugin.pm.register(gate, name="key_capture")

        await plugin.on_message(channel=channel, sender="user", text="use the tool")
        await drain(plugin, channel)
        task_plugin = plugin.pm.get_plugin("task")
        await task_plugin.task_queue.queue.join()
        await asyncio.sleep(0)
        await drain(plugin, channel)

        assert len(captured_tasks) == 1
        task = captured_tasks[0]
        assert task.exchange_key is not None
        assert task.exchange_key == gate.seen_keys[0]
        assert task.origin == "user"

        await task_plugin.on_stop()
        await db.close()

    async def test_tool_result_turn_inherits_same_key_no_second_exchange_row(self):
        plugin, channel, db = await build_plugin_and_channel()
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

        gate = _KeyCapturePlugin()
        plugin.pm.register(gate, name="key_capture")

        await plugin.on_message(channel=channel, sender="user", text="use the tool")
        await drain(plugin, channel)
        task_plugin = plugin.pm.get_plugin("task")
        await task_plugin.task_queue.queue.join()
        await asyncio.sleep(0)
        await drain(plugin, channel)

        key = gate.seen_keys[0]

        async with db.execute(
            "SELECT COUNT(*) FROM exchange_log WHERE exchange_key = ?", (key,)
        ) as cursor:
            (count,) = await cursor.fetchone()
        assert count == 1, "tool-result turn must not mint/insert a second exchange row"

        await task_plugin.on_stop()
        await db.close()

    async def test_final_on_agent_response_carries_original_user_text_as_originating_text(self):
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

        await plugin.on_message(channel=channel, sender="user", text="the original user text")
        await drain(plugin, channel)
        task_plugin = plugin.pm.get_plugin("task")
        await task_plugin.task_queue.queue.join()
        await asyncio.sleep(0)
        await drain(plugin, channel)

        assert len(recorder.calls) == 1
        assert recorder.calls[0]["originating_text"] == "the original user text"
        # request_text (legacy semantics) is the tool-result turn's text, not
        # the originating user text -- this is exactly the mis-pairing bug
        # originating_text exists to fix.
        assert recorder.calls[0]["request_text"] != "the original user text"

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

        await plugin.on_notify(
            channel=channel, source="task", text="standalone event",
            tool_call_id=None, meta=None,
        )
        await drain(plugin, channel)

        assert len(recorder.persisted) == 1
        assert recorder.persisted[0]["exchange_key"] is not None
        # origin='task' must be recorded on the exchange_log row for this key.

        await db.close()

    async def test_standalone_notification_fires_on_message_persisted_exactly_once(self):
        plugin, channel, db = await build_plugin_and_channel()

        recorder = _AdmissionRecorderPlugin()
        plugin.pm.register(recorder, name="admission_recorder")

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=_make_text_response("notified"))
        plugin._client = mock_client

        await plugin.on_notify(
            channel=channel, source="task", text="standalone event",
            tool_call_id=None, meta=None,
        )
        await drain(plugin, channel)

        assert len(recorder.persisted) == 1

        await db.close()

    async def test_standalone_notification_exchange_log_row_has_task_origin(self):
        plugin, channel, db = await build_plugin_and_channel()
        await _setup_outcome_log(plugin.pm, db)

        recorder = _AdmissionRecorderPlugin()
        plugin.pm.register(recorder, name="admission_recorder")

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=_make_text_response("notified"))
        plugin._client = mock_client

        await plugin.on_notify(
            channel=channel, source="task", text="standalone event",
            tool_call_id=None, meta=None,
        )
        await drain(plugin, channel)

        key = recorder.persisted[0]["exchange_key"]
        row = await _fetch_exchange_row(db, key)
        assert row is not None
        assert row[1] == "task"

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

        await plugin.on_message(channel=channel, sender="user", text="use two tools")
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
    async def test_before_agent_turn_receives_channel_exchange_key_origin(self):
        plugin, channel, db = await build_plugin_and_channel()

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=_make_text_response("hi"))
        plugin._client = mock_client

        gate = _KeyCapturePlugin()
        plugin.pm.register(gate, name="key_capture")
        recorder = _BeforeTurnRecorderPlugin()
        plugin.pm.register(recorder, name="before_turn_recorder")

        await plugin.on_message(channel=channel, sender="user", text="hello")
        await drain(plugin, channel)

        assert len(recorder.calls) == 1
        assert recorder.calls[0]["exchange_key"] == gate.seen_keys[0]
        assert recorder.calls[0]["origin"] == "user"

        await db.close()

    async def test_retrieval_profile_lands_in_exchange_log_under_the_key(self):
        """MemoryPlugin.before_agent_turn (updated for WP2.1 point 8) writes
        retrieval_top_score/retrieval_hit_count into exchange_log under the
        exchange key via update_exchange."""
        plugin, channel, db = await build_plugin_and_channel()
        await _setup_outcome_log(plugin.pm, db)

        gate = _KeyCapturePlugin()
        plugin.pm.register(gate, name="key_capture")

        # A minimal before_agent_turn implementer standing in for MemoryPlugin,
        # exercising the same update_exchange call point-8 requires. This
        # avoids pulling in the full MemoryPlugin/embedding stack for this
        # narrow assertion (retrieval-profile persistence under the key),
        # while still proving the enriched hookspec carries what a real
        # consumer needs to perform the write.
        outcome_log = plugin.pm.get_plugin("outcome_log")

        class _StandInRetrievalProfiler:
            @hookimpl
            async def before_agent_turn(self, channel, exchange_key, origin):
                if exchange_key is None:
                    return
                await outcome_log.update_exchange(
                    exchange_key, retrieval_top_score=0.75, retrieval_hit_count=2,
                )

        plugin.pm.register(_StandInRetrievalProfiler(), name="stand_in_profiler")

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=_make_text_response("hi"))
        plugin._client = mock_client

        await plugin.on_message(channel=channel, sender="user", text="hello")
        await drain(plugin, channel)

        key = gate.seen_keys[0]
        async with db.execute(
            "SELECT retrieval_top_score, retrieval_hit_count FROM exchange_log "
            "WHERE exchange_key = ?",
            (key,),
        ) as cursor:
            row = await cursor.fetchone()
        assert row == (0.75, 2)

        await db.close()


# ---------------------------------------------------------------------------
# 8. usage_log carries the exchange key (attribution wiring)
# ---------------------------------------------------------------------------


class TestUsageLogAttributionWiring:
    async def test_usage_log_row_carries_exchange_key_via_attribution(self):
        """Every LLM call inside _process_queue_item_attributed must run
        under an attribution contextvar carrying exchange_key, wired via the
        widened set_attribution(...) call ahead of the attributed body
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

        await plugin.on_message(channel=channel, sender="user", text="hello")
        await drain(plugin, channel)

        assert captured_attribution.get("exchange_key") == gate.seen_keys[0]

    async def test_usage_log_table_row_has_exchange_key_column_populated(self):
        """End-to-end: firing on_llm_response with the real attribution
        snapshot writes a usage_log row whose exchange_key column is set."""
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
                role="main",
                model="test-model",
                request_id="req-1",
                usage={"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
                latency_ms=1.0,
                attribution=attribution,
                error=None,
            )
            return _make_text_response("hi")

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(side_effect=fake_chat)
        plugin._client = mock_client

        await plugin.on_message(channel=channel, sender="user", text="hello")
        await drain(plugin, channel)

        assert captured_attribution.get("exchange_key") == gate.seen_keys[0]

        async with db.execute(
            "SELECT exchange_key FROM usage_log WHERE request_id = ?", ("req-1",)
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None
        assert row[0] == gate.seen_keys[0]

        await db.close()


# ---------------------------------------------------------------------------
# 9. upsert_exchange write-order independence
# ---------------------------------------------------------------------------


class TestUpsertExchangeWriteOrderIndependence:
    async def test_upsert_before_insert_creates_row_with_columns(self):
        plugin, channel, db = await build_plugin_and_channel()
        outcome_log = await _setup_outcome_log(plugin.pm, db)

        await outcome_log.upsert_exchange(
            "ex-upsert-1", "irc:#general", origin="user", probe_score=0.42,
        )

        async with db.execute(
            "SELECT channel_id, origin, probe_score FROM exchange_log "
            "WHERE exchange_key = ?",
            ("ex-upsert-1",),
        ) as cursor:
            row = await cursor.fetchone()
        assert row == ("irc:#general", "user", 0.42)

        await db.close()

    async def test_later_record_exchange_insert_or_ignore_does_not_clobber_upsert(self):
        plugin, channel, db = await build_plugin_and_channel()
        outcome_log = await _setup_outcome_log(plugin.pm, db)

        await outcome_log.upsert_exchange(
            "ex-upsert-2", "irc:#general", origin="user", probe_score=0.9,
        )
        # A later hook-driven record_exchange for the same key must not
        # clobber the upsert-created row (INSERT OR IGNORE).
        await outcome_log.record_exchange("ex-upsert-2", "other-channel", origin="task")

        async with db.execute(
            "SELECT channel_id, origin, probe_score FROM exchange_log "
            "WHERE exchange_key = ?",
            ("ex-upsert-2",),
        ) as cursor:
            row = await cursor.fetchone()
        assert row == ("irc:#general", "user", 0.9)

        await db.close()

    async def test_reverse_order_also_converges(self):
        """record_exchange first, then upsert_exchange for the same key:
        the upsert's UPDATE half still applies its columns."""
        plugin, channel, db = await build_plugin_and_channel()
        outcome_log = await _setup_outcome_log(plugin.pm, db)

        await outcome_log.record_exchange("ex-upsert-3", "irc:#general", origin="user")
        await outcome_log.upsert_exchange(
            "ex-upsert-3", "irc:#general", origin="user", probe_score=0.5,
        )

        async with db.execute(
            "SELECT channel_id, probe_score FROM exchange_log WHERE exchange_key = ?",
            ("ex-upsert-3",),
        ) as cursor:
            row = await cursor.fetchone()
        assert row == ("irc:#general", 0.5)

        await db.close()


# ---------------------------------------------------------------------------
# 10. Atomic JSON merge (json_patch) for outcomes/appraisal columns
# ---------------------------------------------------------------------------


class TestAtomicJsonMerge:
    async def test_two_concurrent_outcomes_writers_both_keys_survive(self):
        plugin, channel, db = await build_plugin_and_channel()
        outcome_log = await _setup_outcome_log(plugin.pm, db)

        await outcome_log.record_exchange("ex-merge-1", "irc:#general", origin="user")

        # Two "concurrent" fire-and-forget writers merging disjoint
        # top-level keys into outcomes.
        await outcome_log.update_exchange(
            "ex-merge-1", outcomes={"engagement": {"salience": 0.6}},
        )
        await outcome_log.update_exchange(
            "ex-merge-1", outcomes={"gate": "rejected"},
        )

        import json as _json

        async with db.execute(
            "SELECT outcomes FROM exchange_log WHERE exchange_key = ?",
            ("ex-merge-1",),
        ) as cursor:
            (outcomes_raw,) = await cursor.fetchone()
        outcomes = _json.loads(outcomes_raw)
        assert outcomes.get("engagement") == {"salience": 0.6}
        assert outcomes.get("gate") == "rejected"

    async def test_two_appraisal_writers_stage1_out_not_erased_by_stage2(self):
        """First writer merges stage1_out; a later stage2 writer must not
        erase it (tranche-2 important 2 regression)."""
        plugin, channel, db = await build_plugin_and_channel()
        outcome_log = await _setup_outcome_log(plugin.pm, db)

        await outcome_log.record_exchange("ex-merge-2", "irc:#general", origin="user")

        await outcome_log.update_exchange(
            "ex-merge-2", appraisal={"stage1_out": {"novelty": 0.3}},
        )
        await outcome_log.update_exchange(
            "ex-merge-2", appraisal={"stage2": {"valence": -0.1}},
        )

        import json as _json

        async with db.execute(
            "SELECT appraisal FROM exchange_log WHERE exchange_key = ?",
            ("ex-merge-2",),
        ) as cursor:
            (appraisal_raw,) = await cursor.fetchone()
        appraisal = _json.loads(appraisal_raw)
        assert appraisal.get("stage1_out") == {"novelty": 0.3}
        assert appraisal.get("stage2") == {"valence": -0.1}

    async def test_merge_column_rejects_scalar_value(self):
        """A scalar for a merge column (outcomes/appraisal) is a ValueError —
        distinguishing merge-columns from plain-set columns."""
        plugin, channel, db = await build_plugin_and_channel()
        outcome_log = await _setup_outcome_log(plugin.pm, db)

        await outcome_log.record_exchange("ex-merge-3", "irc:#general", origin="user")

        with pytest.raises(ValueError):
            await outcome_log.update_exchange("ex-merge-3", outcomes="not-a-dict")

    async def test_plain_set_column_rejects_dict_value(self):
        """A dict for a non-merge column (e.g. origin) is a ValueError."""
        plugin, channel, db = await build_plugin_and_channel()
        outcome_log = await _setup_outcome_log(plugin.pm, db)

        await outcome_log.record_exchange("ex-merge-4", "irc:#general", origin="user")

        with pytest.raises(ValueError):
            await outcome_log.update_exchange("ex-merge-4", origin={"nested": "dict"})

    async def test_upsert_exchange_merge_columns_use_atomic_json_patch(self):
        """upsert_exchange's guarded UPDATE half also merges outcomes/
        appraisal atomically (not read-merge-write)."""
        plugin, channel, db = await build_plugin_and_channel()
        outcome_log = await _setup_outcome_log(plugin.pm, db)

        await outcome_log.upsert_exchange(
            "ex-merge-5", "irc:#general", origin="user",
            appraisal={"stage1": {"novelty": 0.2}},
        )
        await outcome_log.upsert_exchange(
            "ex-merge-5", "irc:#general", origin="user",
            appraisal={"stage1_out": {"novelty": 0.4}},
        )

        import json as _json

        async with db.execute(
            "SELECT appraisal FROM exchange_log WHERE exchange_key = ?",
            ("ex-merge-5",),
        ) as cursor:
            (appraisal_raw,) = await cursor.fetchone()
        appraisal = _json.loads(appraisal_raw)
        assert appraisal.get("stage1") == {"novelty": 0.2}
        assert appraisal.get("stage1_out") == {"novelty": 0.4}

    async def test_truly_concurrent_writers_via_gather_all_disjoint_keys_survive(self):
        """Fires many merges concurrently via asyncio.gather, each writing a
        DISTINCT top-level key into the SAME row's outcomes column, then
        asserts ALL keys survive.

        Why this discriminates atomic vs non-atomic implementations (Red TDD
        Review, "Important, WP2.1" finding: the sibling tests in this class
        only exercise json_patch merge *semantics* under sequential —
        individually awaited — calls, which a naive read-then-write
        implementation would also pass, because each call's read-modify-write
        cycle fully completes before the next call starts. No sequential test
        can distinguish "one atomic UPDATE ... SET x = json_patch(x, ?)" from
        "SELECT x; merge in Python; UPDATE x = ?" — both produce a correct
        final row when calls never overlap.

        A non-atomic implementation of update_exchange has (at minimum) two
        awaited round trips per call: an `await db.execute("SELECT
        outcomes...")` to read the current value, then Python-side dict merge,
        then `await db.execute("UPDATE ... SET outcomes = ?")` to write the
        merged result back. Each `await` is a point where the event loop may
        switch to another coroutine. When N such calls are launched together
        via asyncio.gather, their SELECT/merge/UPDATE sequences can interleave
        on the shared aiosqlite connection: writer B's SELECT can run before
        writer A's UPDATE has committed, so B's in-memory merge is built from
        a base that doesn't yet contain A's key. B's subsequent UPDATE then
        overwrites the row with a value that never included A's key — a
        classic lost update. With enough concurrent writers targeting the
        same row, this race becomes overwhelmingly likely to manifest at
        least once (this test uses 12 writers merging 12 distinct keys,
        chosen to keep the flake probability of a buggy implementation
        accidentally passing negligible while keeping runtime small).

        A genuinely atomic single-statement implementation
        (`UPDATE exchange_log SET outcomes = json_patch(COALESCE(outcomes,
        '{}'), ?) WHERE exchange_key = ?`) has no read-then-write gap for
        another coroutine to interleave into: the merge happens inside SQLite
        itself, in the single SQL statement dispatched to aiosqlite's
        executor thread. Regardless of what order the coroutines' `await
        db.execute(...)` calls actually reach the executor in, each statement
        fully reads-and-merges-and-writes atomically before the next one
        starts, so all N keys survive no matter the interleaving. This test
        therefore fails under a read-merge-write implementation (some keys
        missing) and passes under a single-statement json_patch
        implementation (all keys present) — the discrimination the sibling
        sequential tests in this class cannot provide.
        """
        plugin, channel, db = await build_plugin_and_channel()
        outcome_log = await _setup_outcome_log(plugin.pm, db)

        await outcome_log.record_exchange("ex-merge-gather", "irc:#general", origin="user")

        writer_count = 12
        await asyncio.gather(
            *(
                outcome_log.update_exchange(
                    "ex-merge-gather",
                    outcomes={f"writer_{i}": {"seq": i}},
                )
                for i in range(writer_count)
            )
        )

        import json as _json

        async with db.execute(
            "SELECT outcomes FROM exchange_log WHERE exchange_key = ?",
            ("ex-merge-gather",),
        ) as cursor:
            (outcomes_raw,) = await cursor.fetchone()
        outcomes = _json.loads(outcomes_raw)

        missing = [f"writer_{i}" for i in range(writer_count) if f"writer_{i}" not in outcomes]
        assert not missing, (
            f"lost update(s) detected under concurrent asyncio.gather writers: "
            f"missing keys {missing} — a non-atomic read-merge-write "
            f"update_exchange drops keys from writers whose SELECT raced "
            f"another writer's not-yet-committed UPDATE. Got: {outcomes!r}"
        )
        for i in range(writer_count):
            assert outcomes[f"writer_{i}"] == {"seq": i}

        await db.close()

    async def test_merge_uses_single_update_statement_no_select_then_update(self):
        """Structural proof of atomicity: spies on the real aiosqlite
        Connection.execute (the exact object OutcomeLogPlugin._resolve_db()
        returns -- persistence.db, the same connection instance the test
        fixture created and that update_exchange operates on) during a
        single update_exchange(..., outcomes={...}) call, and asserts the
        merge path issues exactly one write statement and never issues a
        SELECT against exchange_log first.

        Why this discriminates atomic vs non-atomic implementations: this
        complements the asyncio.gather race test above with a deterministic,
        non-probabilistic check that does not depend on the scheduler
        actually interleaving two coroutines' awaits in an unlucky order. A
        read-merge-write implementation of the json_patch merge for a
        merge-column kwarg MUST, by construction, issue a SELECT to fetch the
        current column value before it can merge in Python and issue the
        UPDATE -- there is no way to merge "the new fragment" into "the
        existing envelope" without first reading the existing envelope
        somehow, and the design explicitly forbids reading it into Python
        (WP2.1 point 7: "single atomic SQL statement... NOT read-merge-
        write"). A single-statement `UPDATE ... SET outcomes =
        json_patch(COALESCE(outcomes, '{}'), ?) WHERE exchange_key = ?`
        performs the read, merge, and write entirely inside SQLite as part of
        one `execute()` call from Python's perspective -- there is exactly
        one call to the connection's execute for the merge, and it is never
        preceded by a SELECT. Spying on `db.execute` and asserting "no SELECT
        text, exactly one execute call whose SQL contains UPDATE" therefore
        catches a naive implementation deterministically, independent of
        scheduling luck.
        """
        plugin, channel, db = await build_plugin_and_channel()
        outcome_log = await _setup_outcome_log(plugin.pm, db)

        await outcome_log.record_exchange("ex-merge-spy", "irc:#general", origin="user")

        executed_sql: list[str] = []
        real_execute = db.execute

        def spying_execute(sql, *args, **kwargs):
            executed_sql.append(sql)
            return real_execute(sql, *args, **kwargs)

        db.execute = spying_execute
        try:
            await outcome_log.update_exchange(
                "ex-merge-spy", outcomes={"engagement": {"salience": 0.9}},
            )
        finally:
            db.execute = real_execute

        selects = [sql for sql in executed_sql if sql.strip().upper().startswith("SELECT")]
        assert not selects, (
            f"update_exchange issued a SELECT before its UPDATE -- this is "
            f"the read-then-write shape a non-atomic implementation has; an "
            f"atomic json_patch UPDATE never reads the column into Python. "
            f"Statements executed: {executed_sql!r}"
        )

        write_statements = [
            sql for sql in executed_sql if sql.strip().upper().startswith(("UPDATE", "INSERT"))
        ]
        assert len(write_statements) == 1, (
            f"expected exactly one write statement for a single "
            f"update_exchange(..., outcomes=...) call (the atomic json_patch "
            f"merge is one UPDATE), got {len(write_statements)}: "
            f"{write_statements!r}"
        )
        assert "json_patch" in write_statements[0].lower(), (
            f"expected the merge-column UPDATE to use SQLite's json_patch() "
            f"for an atomic in-database merge, got: {write_statements[0]!r}"
        )

        await db.close()
