"""Tests for sherman.llm.LLMClient."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sherman.llm import LLMClient


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
        assert call_args[0][0] == f"{BASE_URL}/v1/chat/completions"
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

        client = LLMClient(base_url=BASE_URL, model=MODEL)
        response = _make_mock_response(status=500)
        client.session = _make_mock_session(response)

        with pytest.raises(ClientResponseError):
            await client.chat(MESSAGES)


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
