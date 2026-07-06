"""Tests for the LLMClient observer seam and on_llm_request/on_llm_response hooks."""

import asyncio
import logging
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
        self, role, model, request_id, message_count, tool_count, attribution
    ):
        self.requests.append(
            {
                "role": role,
                "model": model,
                "request_id": request_id,
                "message_count": message_count,
                "tool_count": tool_count,
                "attribution": attribution,
            }
        )

    @hookimpl
    async def on_llm_response(
        self, role, model, request_id, usage, latency_ms, attribution, error
    ):
        self.responses.append(
            {
                "role": role,
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

    async def test_main_and_background_clients_get_observers_with_roles(self):
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

        roles = [r["role"] for r in recorder.requests]
        assert roles == ["main", "background"]
        models = [r["model"] for r in recorder.requests]
        assert models == ["main-model", "bg-model"]
        assert [r["role"] for r in recorder.responses] == ["main", "background"]

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
        assert recorder.requests[0]["role"] == "main"
