"""Tests for the LLMClient observer seam and on_llm_request/on_llm_response hooks."""

import asyncio
import logging
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from corvidae.hooks import create_plugin_manager, hookimpl
from corvidae.llm import LLMClient

BASE_URL = "http://localhost:8080"
MODEL = "test-model"
MESSAGES = [{"role": "user", "content": "hello"}]
TOOLS = [{"type": "function", "function": {"name": "my_tool", "parameters": {}}}]
USAGE = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
MOCK_COMPLETION = {
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "hello back"},
            "finish_reason": "stop",
        }
    ],
    "usage": USAGE,
}


def _make_mock_response(status: int = 200, json_body: dict | None = None):
    """Build a mock aiohttp response."""
    response = AsyncMock()
    response.status = status
    response.json = AsyncMock(return_value=json_body or MOCK_COMPLETION)
    response.raise_for_status = MagicMock()
    response.headers = {}
    if status >= 400:
        from aiohttp import ClientResponseError
        response.raise_for_status.side_effect = ClientResponseError(
            request_info=MagicMock(), history=(), status=status
        )
    return response


def _make_mock_session(*responses):
    """Build a mock aiohttp.ClientSession whose post() yields each response in turn."""
    session = AsyncMock()
    cms = []
    for response in responses:
        post_cm = AsyncMock()
        post_cm.__aenter__ = AsyncMock(return_value=response)
        post_cm.__aexit__ = AsyncMock(return_value=False)
        cms.append(post_cm)
    session.post = MagicMock(side_effect=cms)
    session.close = AsyncMock()
    return session


class _RecordingObserver:
    """Observer that records every request/response callback."""

    def __init__(self):
        self.requests: list[dict] = []
        self.responses: list[dict] = []

    async def request(self, **kwargs):
        self.requests.append(kwargs)

    async def response(self, **kwargs):
        self.responses.append(kwargs)


class _RaisingObserver:
    async def request(self, **kwargs):
        raise RuntimeError("observer request boom")

    async def response(self, **kwargs):
        raise RuntimeError("observer response boom")


class TestLLMClientObserver:
    async def test_request_then_response_fired_exactly_once(self):
        client = LLMClient(base_url=BASE_URL, model=MODEL)
        client.session = _make_mock_session(_make_mock_response())
        observer = _RecordingObserver()
        client.observer = observer

        result = await client.chat(MESSAGES, tools=TOOLS)

        assert result == MOCK_COMPLETION
        assert len(observer.requests) == 1
        assert len(observer.responses) == 1
        req = observer.requests[0]
        resp = observer.responses[0]
        # request_id pairs the request with its response
        assert req["request_id"] == resp["request_id"]
        assert req["message_count"] == len(MESSAGES)
        assert req["tool_count"] == len(TOOLS)
        assert req["model"] == MODEL
        # usage passes through verbatim; latency is measured
        assert resp["usage"] == USAGE
        assert resp["latency_ms"] > 0
        assert resp["error"] is None

    async def test_no_observer_behaves_as_before(self):
        client = LLMClient(base_url=BASE_URL, model=MODEL)
        client.session = _make_mock_session(_make_mock_response())
        result = await client.chat(MESSAGES)
        assert result == MOCK_COMPLETION

    async def test_raising_observer_does_not_break_chat(self, caplog):
        client = LLMClient(base_url=BASE_URL, model=MODEL)
        client.session = _make_mock_session(_make_mock_response())
        client.observer = _RaisingObserver()

        with caplog.at_level(logging.WARNING):
            result = await client.chat(MESSAGES)

        assert result == MOCK_COMPLETION
        # A warning was logged for the failing observer callbacks.
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    async def test_terminal_http_error_fires_one_response_with_error(self):
        from aiohttp import ClientResponseError

        client = LLMClient(base_url=BASE_URL, model=MODEL)
        client.session = _make_mock_session(_make_mock_response(status=400))
        observer = _RecordingObserver()
        client.observer = observer

        with pytest.raises(ClientResponseError):
            await client.chat(MESSAGES)

        assert len(observer.requests) == 1
        assert len(observer.responses) == 1
        resp = observer.responses[0]
        assert resp["usage"] is None
        assert resp["error"] is not None
        assert resp["request_id"] == observer.requests[0]["request_id"]

    async def test_transient_then_success_fires_one_response(self):
        # 429 (retried) then 200 → exactly one request and one response.
        client = LLMClient(base_url=BASE_URL, model=MODEL, retry_base_delay=0.0)
        client.session = _make_mock_session(
            _make_mock_response(status=429), _make_mock_response()
        )
        observer = _RecordingObserver()
        client.observer = observer

        result = await client.chat(MESSAGES)

        assert result == MOCK_COMPLETION
        assert len(observer.requests) == 1
        assert len(observer.responses) == 1
        assert observer.responses[0]["error"] is None

    async def test_retries_exhausted_fires_one_response_with_error(self):
        from aiohttp import ClientResponseError

        client = LLMClient(
            base_url=BASE_URL, model=MODEL, max_retries=1, retry_base_delay=0.0
        )
        client.session = _make_mock_session(
            _make_mock_response(status=503), _make_mock_response(status=503)
        )
        observer = _RecordingObserver()
        client.observer = observer

        with pytest.raises(ClientResponseError):
            await client.chat(MESSAGES)

        assert len(observer.requests) == 1
        assert len(observer.responses) == 1
        assert observer.responses[0]["error"] is not None


class _HookRecorder:
    """Plugin recording on_llm_request/on_llm_response hook invocations."""

    def __init__(self):
        self.requests: list[dict] = []
        self.responses: list[dict] = []

    @hookimpl
    async def on_llm_request(
        self, profile, model, request_id, message_count, tool_count, attribution
    ):
        self.requests.append(
            {
                "profile": profile,
                "model": model,
                "request_id": request_id,
                "message_count": message_count,
                "tool_count": tool_count,
                "attribution": attribution,
            }
        )

    @hookimpl
    async def on_llm_response(
        self, profile, model, request_id, usage, latency_ms, attribution, error
    ):
        self.responses.append(
            {
                "profile": profile,
                "model": model,
                "request_id": request_id,
                "usage": usage,
                "latency_ms": latency_ms,
                "attribution": attribution,
                "error": error,
            }
        )


class TestLLMPluginObserverWiring:
    async def _start_plugin(self, config):
        from corvidae.llm_plugin import LLMPlugin

        pm = create_plugin_manager()
        recorder = _HookRecorder()
        pm.register(recorder, name="recorder")
        plugin = LLMPlugin()
        pm.register(plugin, name="llm")
        await plugin.on_init(pm, config)
        await plugin.on_start(config)
        return plugin, recorder

    async def test_main_and_background_clients_get_observers_with_profiles(self):
        config = {
            "llm": {
                "main": {"base_url": BASE_URL, "model": "main-model"},
                "background": {"base_url": BASE_URL, "model": "bg-model"},
            }
        }
        plugin, recorder = await self._start_plugin(config)
        try:
            # Stub out the HTTP sessions so chat() never hits the network.
            plugin.main_client.session = _make_mock_session(_make_mock_response())
            plugin.background_client.session = _make_mock_session(
                _make_mock_response()
            )

            await plugin.main_client.chat(MESSAGES)
            await plugin.background_client.chat(MESSAGES)
        finally:
            await plugin.on_stop()

        profiles = [r["profile"] for r in recorder.requests]
        assert profiles == ["main", "background"]
        models = [r["model"] for r in recorder.requests]
        assert models == ["main-model", "bg-model"]
        assert [r["profile"] for r in recorder.responses] == ["main", "background"]

    async def test_hook_receives_attribution_snapshot(self):
        from corvidae.attribution import reset_attribution, set_attribution

        config = {"llm": {"main": {"base_url": BASE_URL, "model": MODEL}}}
        plugin, recorder = await self._start_plugin(config)
        try:
            plugin.main_client.session = _make_mock_session(_make_mock_response())
            token = set_attribution(stage="turn", channel_id="c1")
            try:
                await plugin.main_client.chat(MESSAGES)
            finally:
                reset_attribution(token)
        finally:
            await plugin.on_stop()

        assert recorder.requests[0]["attribution"] == {
            "stage": "turn",
            "channel_id": "c1",
        }
        assert recorder.responses[0]["attribution"] == {
            "stage": "turn",
            "channel_id": "c1",
        }

    async def test_config_reload_rewires_observer_on_new_client(self):
        config = {"llm": {"main": {"base_url": BASE_URL, "model": MODEL}}}
        plugin, recorder = await self._start_plugin(config)
        try:
            new_config = {
                "llm": {"main": {"base_url": BASE_URL, "model": "new-model"}}
            }
            await plugin.on_config_reload(new_config)
            plugin.main_client.session = _make_mock_session(_make_mock_response())
            await plugin.main_client.chat(MESSAGES)
        finally:
            await plugin.on_stop()
            # Allow the deferred old-client close task to be scheduled/cancelled.
            await asyncio.sleep(0)

        assert recorder.requests[0]["model"] == "new-model"
        assert recorder.requests[0]["profile"] == "main"


class TestObserverNamesModelSent:
    """#297 / PRD §3.13.10: the observer pair names the model each call was
    sent with, not the profile's configured model, and passes llama-server's
    ``timings`` through so cache hits are recordable."""

    async def _chat(self, client, **kwargs):
        # Run one chat call with a recording observer; return the observer.
        observer = _RecordingObserver()
        client.observer = observer
        await client.chat(MESSAGES, **kwargs)
        return observer

    async def test_instance_extra_body_model_reaches_observers(self):
        client = LLMClient(base_url=BASE_URL, model=MODEL, extra_body={"model": "instance-model"})
        client.session = _make_mock_session(_make_mock_response())
        observer = await self._chat(client)
        assert observer.requests[0]["model"] == "instance-model"
        assert observer.responses[0]["model"] == "instance-model"

    async def test_call_extra_body_model_overrides_instance(self):
        client = LLMClient(base_url=BASE_URL, model=MODEL, extra_body={"model": "instance-model"})
        client.session = _make_mock_session(_make_mock_response())
        observer = await self._chat(client, extra_body={"model": "call-model"})
        assert observer.requests[0]["model"] == "call-model"
        assert observer.responses[0]["model"] == "call-model"

    @pytest.mark.parametrize(
        "instance_body, call_body, expected",
        [
            ({"model": "instance-model"}, None, "instance-model"),
            (None, {"model": "call-model"}, "call-model"),
            (None, None, MODEL),
        ],
        ids=["instance-override", "call-override", "no-override"],
    )
    async def test_terminal_failure_names_model_sent(self, instance_body, call_body, expected):
        from aiohttp import ClientResponseError

        client = LLMClient(base_url=BASE_URL, model=MODEL, extra_body=instance_body)
        client.session = _make_mock_session(_make_mock_response(status=400))
        observer = _RecordingObserver()
        client.observer = observer
        with pytest.raises(ClientResponseError):
            await client.chat(MESSAGES, extra_body=call_body)
        assert observer.requests[0]["model"] == expected
        assert observer.responses[0]["model"] == expected
        assert observer.responses[0]["timings"] is None

    async def test_no_override_names_configured_model(self):
        client = LLMClient(base_url=BASE_URL, model=MODEL)
        client.session = _make_mock_session(_make_mock_response())
        observer = await self._chat(client)
        assert observer.requests[0]["model"] == MODEL
        assert observer.responses[0]["model"] == MODEL

    async def test_embed_names_configured_model(self):
        # embed takes no extra_body, so the configured model is the one sent.
        client = LLMClient(base_url=BASE_URL, model=MODEL)
        client.session = _make_mock_session(
            _make_mock_response(json_body={"data": [{"index": 0, "embedding": [0.1]}]})
        )
        observer = _RecordingObserver()
        client.observer = observer
        await client.embed(["text"], kind="document")
        assert observer.requests[0]["model"] == MODEL
        assert observer.responses[0]["model"] == MODEL
        assert observer.responses[0]["timings"] is None

    async def test_timings_pass_through_verbatim(self):
        timings = {"cache_n": 236, "prompt_n": 12, "predicted_n": 5}
        client = LLMClient(base_url=BASE_URL, model=MODEL)
        client.session = _make_mock_session(
            _make_mock_response(json_body={**MOCK_COMPLETION, "timings": timings})
        )
        observer = await self._chat(client)
        assert observer.responses[0]["timings"] == timings

    async def test_missing_timings_is_none(self):
        client = LLMClient(base_url=BASE_URL, model=MODEL)
        client.session = _make_mock_session(_make_mock_response())
        observer = await self._chat(client)
        assert observer.responses[0]["timings"] is None

    async def test_override_model_and_cache_counts_reach_usage_log(self, db):
        """End to end through the hook bridge: an extra_body model override
        and llama-server's timings land in usage_log."""
        from corvidae.llm_plugin import _HookObserver
        from corvidae.metrics import UsageLogPlugin
        from corvidae.persistence import PersistencePlugin

        pm = create_plugin_manager()
        persistence = PersistencePlugin()
        persistence.db = db
        pm.register(persistence, name="persistence")
        pm.register(UsageLogPlugin(pm), name="usage_log")

        timings = {"cache_n": 800, "prompt_n": 20}
        client = LLMClient(base_url=BASE_URL, model=MODEL)
        client.session = _make_mock_session(
            _make_mock_response(json_body={**MOCK_COMPLETION, "timings": timings})
        )
        client.observer = _HookObserver(pm, "main", MODEL)
        await client.chat(MESSAGES, extra_body={"model": "call-model"})

        async with db.execute("SELECT model, cache_n, prompt_n FROM usage_log") as cursor:
            assert await cursor.fetchall() == [("call-model", 800, 20)]

    @pytest.mark.parametrize(
        "instance_body, call_body, status, expected",
        [
            ({"model": "instance-model"}, None, 200, "instance-model"),
            (None, None, 200, MODEL),
            ({"model": "instance-model"}, None, 400, "instance-model"),
            (None, {"model": "call-model"}, 400, "call-model"),
        ],
        ids=["instance-ok", "no-override-ok", "instance-fail", "call-fail"],
    )
    async def test_usage_log_names_model_sent(self, db, instance_body, call_body, status, expected):
        """usage_log names the model sent for each override source, on
        success and on terminal failure, and the profile's model without one."""
        from aiohttp import ClientResponseError

        from corvidae.llm_plugin import _HookObserver
        from corvidae.metrics import UsageLogPlugin
        from corvidae.persistence import PersistencePlugin

        pm = create_plugin_manager()
        persistence = PersistencePlugin()
        persistence.db = db
        pm.register(persistence, name="persistence")
        pm.register(UsageLogPlugin(pm), name="usage_log")

        client = LLMClient(base_url=BASE_URL, model=MODEL, extra_body=instance_body)
        client.session = _make_mock_session(_make_mock_response(status=status))
        client.observer = _HookObserver(pm, "main", MODEL)
        if status >= 400:
            with pytest.raises(ClientResponseError):
                await client.chat(MESSAGES, extra_body=call_body)
        else:
            await client.chat(MESSAGES, extra_body=call_body)

        async with db.execute("SELECT model FROM usage_log") as cursor:
            assert await cursor.fetchall() == [(expected,)]


def _make_status_response(status: int, body: str = ""):
    """Build a mock aiohttp response with a status and a readable error body."""
    response = AsyncMock()
    response.status = status
    response.json = AsyncMock(return_value=MOCK_COMPLETION)
    response.text = AsyncMock(return_value=body)
    response.raise_for_status = MagicMock()
    response.headers = {}
    if status >= 400:
        from aiohttp import ClientResponseError
        response.raise_for_status.side_effect = ClientResponseError(
            request_info=MagicMock(), history=(), status=status
        )
    return response


class TestLLMCallTimings:
    """#350: every on_llm_response carries started_at, total_ms, attempts and the request's counts."""

    async def test_first_try_chat_reports_one_attempt_and_counts(self):
        client = LLMClient(base_url=BASE_URL, model=MODEL)
        client.session = _make_mock_session(_make_mock_response())
        observer = _RecordingObserver()
        client.observer = observer

        # Bracket the call with wall-clock reads so started_at can be checked against them.
        before = time.time()
        await client.chat(MESSAGES, tools=TOOLS)
        after = time.time()

        resp = observer.responses[0]
        assert resp["attempts"] == 1
        assert isinstance(resp["started_at"], float)
        assert before <= resp["started_at"] <= after
        assert resp["total_ms"] >= resp["latency_ms"]
        assert resp["message_count"] == len(MESSAGES)
        assert resp["tool_count"] == len(TOOLS)

    async def test_chat_without_tools_reports_zero_tool_count(self):
        client = LLMClient(base_url=BASE_URL, model=MODEL)
        client.session = _make_mock_session(_make_mock_response())
        observer = _RecordingObserver()
        client.observer = observer

        await client.chat(MESSAGES)

        resp = observer.responses[0]
        assert resp["message_count"] == len(MESSAGES)
        assert resp["tool_count"] == 0

    async def test_embed_reports_text_count_and_no_tools(self):
        client = LLMClient(base_url=BASE_URL, model=MODEL)
        embed_response = _make_status_response(200)
        embed_response.json = AsyncMock(return_value={
            "data": [
                {"index": 0, "embedding": [0.1, 0.2]},
                {"index": 1, "embedding": [0.3, 0.4]},
            ]
        })
        client.session = _make_mock_session(embed_response)
        observer = _RecordingObserver()
        client.observer = observer

        await client.embed(["first", "second"], kind="document")

        resp = observer.responses[0]
        # message_count is the number of texts; embeddings carry no tools.
        assert resp["message_count"] == 2
        assert resp["tool_count"] == 0
        assert resp["attempts"] == 1

    async def test_two_transient_failures_then_success_counts_all_attempts(self):
        # Small base delay: retry sleeps are 0.05 s then 0.10 s (no Retry-After header).
        client = LLMClient(base_url=BASE_URL, model=MODEL, retry_base_delay=0.05)
        client.session = _make_mock_session(
            _make_status_response(503),
            _make_status_response(503),
            _make_mock_response(),
        )
        observer = _RecordingObserver()
        client.observer = observer

        await client.chat(MESSAGES)

        resp = observer.responses[0]
        slept_ms = (0.05 + 0.10) * 1000
        assert resp["attempts"] == 3
        # total_ms spans the failed attempts and the sleeps between them.
        assert resp["total_ms"] >= slept_ms
        assert resp["total_ms"] > resp["latency_ms"]
        # latency_ms is the final attempt alone, so it excludes the sleeps.
        assert resp["latency_ms"] < slept_ms

    async def test_model_not_loaded_wait_counts_as_attempt(self, monkeypatch):
        # Shrink the 1 s minimum not-loaded delay so the test stays fast.
        monkeypatch.setattr("corvidae.llm.MODEL_NOT_LOADED_MIN_DELAY", 0.01)
        client = LLMClient(base_url=BASE_URL, model=MODEL, retry_base_delay=0.01)
        not_loaded_body = '{"error": {"message": "model is not loaded"}}'
        client.session = _make_mock_session(
            _make_status_response(400, not_loaded_body),
            _make_mock_response(),
        )
        observer = _RecordingObserver()
        client.observer = observer

        await client.chat(MESSAGES)

        # One not-loaded wait plus the successful attempt.
        assert observer.responses[0]["attempts"] == 2

    async def test_terminal_failure_fires_one_response_with_timing(self):
        from aiohttp import ClientResponseError

        client = LLMClient(base_url=BASE_URL, model=MODEL)
        client.session = _make_mock_session(
            _make_status_response(404, "no such model")
        )
        observer = _RecordingObserver()
        client.observer = observer

        with pytest.raises(ClientResponseError):
            await client.chat(MESSAGES)

        assert len(observer.responses) == 1
        resp = observer.responses[0]
        assert resp["error"] is not None
        assert resp["timings"] is None
        assert resp["attempts"] >= 1
        assert isinstance(resp["started_at"], float)
        assert isinstance(resp["total_ms"], float)
        assert resp["total_ms"] >= 0
