"""Tests for corvidae.llm.LLMClient."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from corvidae.llm import LLMClient


BASE_URL = "http://localhost:8080"
MODEL = "test-model"
MESSAGES = [{"role": "user", "content": "hello"}]
TOOLS = [{"type": "function", "function": {"name": "my_tool", "parameters": {}}}]
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


def _make_mock_session(response):
    """Build a mock aiohttp.ClientSession whose post() returns response."""
    session = AsyncMock()
    # session.post() is used as an async context manager
    post_cm = AsyncMock()
    post_cm.__aenter__ = AsyncMock(return_value=response)
    post_cm.__aexit__ = AsyncMock(return_value=False)
    session.post = MagicMock(return_value=post_cm)
    session.close = AsyncMock()
    return session


class TestLLMClientInit:
    def test_strips_trailing_slash(self):
        client = LLMClient(base_url="http://localhost:8080/", model=MODEL)
        assert client.base_url == "http://localhost:8080"

    def test_no_trailing_slash_unchanged(self):
        client = LLMClient(base_url=BASE_URL, model=MODEL)
        assert client.base_url == BASE_URL

    def test_model_stored(self):
        client = LLMClient(base_url=BASE_URL, model=MODEL)
        assert client.model == MODEL


class TestLLMClientLifecycle:
    async def test_start_creates_session(self):
        client = LLMClient(base_url=BASE_URL, model=MODEL)
        with patch("aiohttp.ClientSession") as mock_cls:
            mock_cls.return_value = MagicMock()
            await client.start()
            assert client.session is not None

    async def test_stop_closes_session(self):
        client = LLMClient(base_url=BASE_URL, model=MODEL)
        mock_session = _make_mock_session(_make_mock_response())
        client.session = mock_session
        await client.stop()
        mock_session.close.assert_called_once()


class TestLLMClientChat:
    async def test_chat_simple_response(self):
        client = LLMClient(base_url=BASE_URL, model=MODEL)
        response = _make_mock_response(json_body=MOCK_COMPLETION)
        client.session = _make_mock_session(response)

        result = await client.chat(MESSAGES)

        assert result == MOCK_COMPLETION
        client.session.post.assert_called_once()
        call_args = client.session.post.call_args
        assert call_args[0][0] == f"{BASE_URL}/chat/completions"
        payload = call_args[1]["json"]
        assert payload["model"] == MODEL
        assert payload["messages"] == MESSAGES

    async def test_chat_with_tools(self):
        client = LLMClient(base_url=BASE_URL, model=MODEL)
        response = _make_mock_response(json_body=MOCK_COMPLETION)
        client.session = _make_mock_session(response)

        result = await client.chat(MESSAGES, tools=TOOLS)

        assert result == MOCK_COMPLETION
        call_args = client.session.post.call_args
        payload = call_args[1]["json"]
        assert payload["tools"] == TOOLS

    async def test_chat_no_tools_omits_key(self):
        client = LLMClient(base_url=BASE_URL, model=MODEL)
        response = _make_mock_response(json_body=MOCK_COMPLETION)
        client.session = _make_mock_session(response)

        await client.chat(MESSAGES, tools=None)

        call_args = client.session.post.call_args
        payload = call_args[1]["json"]
        assert "tools" not in payload

    async def test_chat_raises_on_http_error(self):
        from aiohttp import ClientResponseError

        client = LLMClient(base_url=BASE_URL, model=MODEL, max_retries=0)
        response = _make_mock_response(status=500)
        client.session = _make_mock_session(response)

        with pytest.raises(ClientResponseError):
            await client.chat(MESSAGES)


class TestLLMClientRetry:
    """Fix 3: Retry on transient LLM API errors."""

    async def test_chat_retries_on_transient_status(self):
        """First call returns 500, second returns 200. session.post called twice."""
        import aiohttp

        client = LLMClient(
            base_url=BASE_URL, model=MODEL, max_retries=1, retry_base_delay=1.0
        )
        resp_500 = _make_mock_response(status=500)
        resp_200 = _make_mock_response(status=200, json_body=MOCK_COMPLETION)

        session = AsyncMock()
        cm_500 = AsyncMock()
        cm_500.__aenter__ = AsyncMock(return_value=resp_500)
        cm_500.__aexit__ = AsyncMock(return_value=False)
        cm_200 = AsyncMock()
        cm_200.__aenter__ = AsyncMock(return_value=resp_200)
        cm_200.__aexit__ = AsyncMock(return_value=False)
        session.post = MagicMock(side_effect=[cm_500, cm_200])
        session.close = AsyncMock()
        client.session = session

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await client.chat(MESSAGES)

        assert result == MOCK_COMPLETION
        assert session.post.call_count == 2

    async def test_chat_retries_on_429_honors_retry_after(self):
        """429 with Retry-After: 5 header; asyncio.sleep called with 5.0."""
        import aiohttp

        client = LLMClient(
            base_url=BASE_URL, model=MODEL, max_retries=1, retry_base_delay=1.0
        )
        resp_429 = _make_mock_response(status=429)
        resp_429.headers = {"Retry-After": "5"}
        resp_200 = _make_mock_response(status=200, json_body=MOCK_COMPLETION)

        session = AsyncMock()
        cm_429 = AsyncMock()
        cm_429.__aenter__ = AsyncMock(return_value=resp_429)
        cm_429.__aexit__ = AsyncMock(return_value=False)
        cm_200 = AsyncMock()
        cm_200.__aenter__ = AsyncMock(return_value=resp_200)
        cm_200.__aexit__ = AsyncMock(return_value=False)
        session.post = MagicMock(side_effect=[cm_429, cm_200])
        session.close = AsyncMock()
        client.session = session

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await client.chat(MESSAGES)

        assert result == MOCK_COMPLETION
        mock_sleep.assert_called_once_with(5.0)

    async def test_chat_raises_after_exhausting_retries(self):
        """All attempts return 500 (max_retries=2 => 3 total). ClientResponseError raised."""
        import aiohttp

        client = LLMClient(
            base_url=BASE_URL, model=MODEL, max_retries=2, retry_base_delay=1.0
        )

        def make_cm(status):
            resp = _make_mock_response(status=status)
            cm = AsyncMock()
            cm.__aenter__ = AsyncMock(return_value=resp)
            cm.__aexit__ = AsyncMock(return_value=False)
            return cm

        session = AsyncMock()
        session.post = MagicMock(side_effect=[make_cm(500), make_cm(500), make_cm(500)])
        session.close = AsyncMock()
        client.session = session

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(aiohttp.ClientResponseError):
                await client.chat(MESSAGES)

        assert session.post.call_count == 3

    async def test_chat_no_retry_on_400(self):
        """400 is non-transient; ClientResponseError raised immediately, post called once."""
        import aiohttp

        client = LLMClient(
            base_url=BASE_URL, model=MODEL, max_retries=3, retry_base_delay=1.0
        )
        resp_400 = _make_mock_response(status=400)
        session = _make_mock_session(resp_400)
        client.session = session

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(aiohttp.ClientResponseError):
                await client.chat(MESSAGES)

        assert session.post.call_count == 1

    async def test_chat_retries_on_connection_error(self):
        """First call raises ClientConnectionError; second succeeds."""
        import aiohttp

        client = LLMClient(
            base_url=BASE_URL, model=MODEL, max_retries=1, retry_base_delay=1.0
        )
        resp_200 = _make_mock_response(status=200, json_body=MOCK_COMPLETION)

        session = AsyncMock()
        cm_err = AsyncMock()
        cm_err.__aenter__ = AsyncMock(
            side_effect=aiohttp.ClientConnectionError("connection refused")
        )
        cm_err.__aexit__ = AsyncMock(return_value=False)
        cm_200 = AsyncMock()
        cm_200.__aenter__ = AsyncMock(return_value=resp_200)
        cm_200.__aexit__ = AsyncMock(return_value=False)
        session.post = MagicMock(side_effect=[cm_err, cm_200])
        session.close = AsyncMock()
        client.session = session

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await client.chat(MESSAGES)

        assert result == MOCK_COMPLETION

    async def test_chat_raises_connection_error_after_retries(self):
        """All attempts raise ClientConnectionError (max_retries=1). Exception propagates."""
        import aiohttp

        client = LLMClient(
            base_url=BASE_URL, model=MODEL, max_retries=1, retry_base_delay=1.0
        )

        def make_error_cm():
            cm = AsyncMock()
            cm.__aenter__ = AsyncMock(
                side_effect=aiohttp.ClientConnectionError("refused")
            )
            cm.__aexit__ = AsyncMock(return_value=False)
            return cm

        session = AsyncMock()
        session.post = MagicMock(side_effect=[make_error_cm(), make_error_cm()])
        session.close = AsyncMock()
        client.session = session

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(aiohttp.ClientConnectionError):
                await client.chat(MESSAGES)

        assert session.post.call_count == 2

    def test_retry_delay_exponential_backoff(self):
        """attempt=0 returns base_delay; attempt=1 returns base_delay * 2."""
        client = LLMClient(
            base_url=BASE_URL,
            model=MODEL,
            retry_base_delay=2.0,
            retry_max_delay=60.0,
        )
        assert client._retry_delay(0) == 2.0
        assert client._retry_delay(1) == 4.0

    def test_retry_delay_honors_retry_after(self):
        """_retry_delay(0, retry_after='5') returns 5.0."""
        client = LLMClient(
            base_url=BASE_URL,
            model=MODEL,
            retry_base_delay=2.0,
            retry_max_delay=60.0,
        )
        assert client._retry_delay(0, retry_after="5") == 5.0

    def test_retry_delay_caps_at_max(self):
        """_retry_delay(10) returns retry_max_delay (60.0), not 2.0 * 2^10."""
        client = LLMClient(
            base_url=BASE_URL,
            model=MODEL,
            retry_base_delay=2.0,
            retry_max_delay=60.0,
        )
        assert client._retry_delay(10) == 60.0

    def test_retry_delay_retry_after_capped(self):
        """_retry_delay(0, retry_after='999') with retry_max_delay=60 returns 60.0."""
        client = LLMClient(
            base_url=BASE_URL,
            model=MODEL,
            retry_base_delay=2.0,
            retry_max_delay=60.0,
        )
        assert client._retry_delay(0, retry_after="999") == 60.0


class TestLLMClientTimeout:
    """Fix 4: Configurable timeout on LLM HTTP calls."""

    def test_timeout_stored_as_client_timeout(self):
        """LLMClient(timeout=30.0) stores aiohttp.ClientTimeout(total=30.0)."""
        import aiohttp

        client = LLMClient(base_url=BASE_URL, model=MODEL, timeout=30.0)
        assert client.timeout == aiohttp.ClientTimeout(total=30.0)

    def test_timeout_none_by_default(self):
        """Default LLMClient has self.timeout is None."""
        client = LLMClient(base_url=BASE_URL, model=MODEL)
        assert client.timeout is None

    async def test_chat_passes_timeout_to_post(self):
        """session.post() is called with timeout=self.timeout kwarg."""
        import aiohttp

        client = LLMClient(base_url=BASE_URL, model=MODEL, timeout=30.0)
        response = _make_mock_response(status=200, json_body=MOCK_COMPLETION)
        client.session = _make_mock_session(response)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await client.chat(MESSAGES)

        call_kwargs = client.session.post.call_args[1]
        assert "timeout" in call_kwargs
        assert call_kwargs["timeout"] == client.timeout


class TestLLMClientChatExtraBody:
    async def test_chat_with_extra_body_none(self):
        """extra_body=None should not add any extra fields to the payload."""
        client = LLMClient(base_url=BASE_URL, model=MODEL)
        response = _make_mock_response(json_body=MOCK_COMPLETION)
        client.session = _make_mock_session(response)

        await client.chat(MESSAGES, extra_body=None)

        call_args = client.session.post.call_args
        payload = call_args[1]["json"]
        assert payload.keys() == {"model", "messages"}

    async def test_chat_with_extra_body_empty_dict(self):
        """extra_body={} should not add any extra fields to the payload."""
        client = LLMClient(base_url=BASE_URL, model=MODEL)
        response = _make_mock_response(json_body=MOCK_COMPLETION)
        client.session = _make_mock_session(response)

        await client.chat(MESSAGES, extra_body={})

        call_args = client.session.post.call_args
        payload = call_args[1]["json"]
        assert payload.keys() == {"model", "messages"}

    async def test_chat_with_extra_body_fields(self):
        """extra_body with actual fields should merge them into the payload."""
        client = LLMClient(base_url=BASE_URL, model=MODEL)
        response = _make_mock_response(json_body=MOCK_COMPLETION)
        client.session = _make_mock_session(response)

        await client.chat(MESSAGES, extra_body={"id_slot": 1})

        call_args = client.session.post.call_args
        payload = call_args[1]["json"]
        assert payload["id_slot"] == 1
        assert payload["model"] == MODEL
        assert payload["messages"] == MESSAGES

    async def test_chat_with_extra_body_multiple_fields(self):
        """extra_body with multiple fields should merge all into the payload."""
        client = LLMClient(base_url=BASE_URL, model=MODEL)
        response = _make_mock_response(json_body=MOCK_COMPLETION)
        client.session = _make_mock_session(response)

        await client.chat(MESSAGES, extra_body={"id_slot": 1, "cache_prompt": True})

        call_args = client.session.post.call_args
        payload = call_args[1]["json"]
        assert payload["id_slot"] == 1
        assert payload["cache_prompt"] == True
