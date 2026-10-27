"""Tests for the actionable 'model is not loaded' error reply (GitHub issue #55).

A llama-server router running --no-models-autoload answers a request for an
unloaded model with HTTP 400 "model is not loaded". After LLMClient's wait
budget runs out, LLMPlugin's on_llm_error turns that into a reply naming
the model, instead of the generic error.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import aiohttp

from corvidae.hooks import hookimpl
from corvidae.llm import LLMHTTPError
from corvidae.llm_plugin import LLMPlugin

from helpers import build_plugin_and_channel, drain

NOT_LOADED_BODY = json.dumps({
    "error": {"code": 400, "message": "model is not loaded", "type": "invalid_request_error"}
})


def _http_error(body: str, model: str = "swift-1.5", status: int = 400) -> LLMHTTPError:
    """An LLMHTTPError as LLMClient raises it for a non-2xx response."""
    cause = aiohttp.ClientResponseError(
        request_info=MagicMock(), history=(), status=status, message="Bad Request"
    )
    return LLMHTTPError(cause, body=body, model=model)


class TestLLMPluginOnLlmError:
    async def test_names_unloaded_model(self):
        reply = await LLMPlugin().on_llm_error(channel=MagicMock(), error=_http_error(NOT_LOADED_BODY))
        assert reply is not None
        assert "swift-1.5" in reply
        assert "isn't loaded" in reply

    async def test_explains_other_http_errors(self):
        """Non-model-not-loaded HTTP failures get their own message."""
        error = _http_error('{"error": {"message": "context too long"}}')
        reply = await LLMPlugin().on_llm_error(channel=MagicMock(), error=error)
        assert reply is not None
        assert "400" in reply
        assert "context too long" in reply

    async def test_defers_on_non_http_errors(self):
        error = RuntimeError("model is not loaded")
        assert await LLMPlugin().on_llm_error(channel=MagicMock(), error=error) is None


class TestCreateClient:
    def test_model_not_loaded_wait_from_config(self):
        cfg = {"base_url": "http://x", "model": "m", "model_not_loaded_wait": 15}
        assert LLMPlugin()._create_client(cfg).model_not_loaded_wait == 15

    def test_model_not_loaded_wait_default(self):
        cfg = {"base_url": "http://x", "model": "m"}
        assert LLMPlugin()._create_client(cfg).model_not_loaded_wait == 60.0


class TestEndToEnd:
    async def test_turn_replies_with_actionable_message(self):
        """A turn failing with 'model is not loaded' sends LLMPlugin's message."""
        plugin, channel, db = await build_plugin_and_channel()
        plugin._client = MagicMock()
        plugin._client.chat = AsyncMock(side_effect=_http_error(NOT_LOADED_BODY))
        plugin.pm.register(LLMPlugin(), name="llm_for_error")

        await plugin.on_message(channel=channel, sender="user", text="hi")
        await drain(plugin, channel)

        sent = plugin.pm.ahook.send_message.call_args.kwargs["text"]
        assert "swift-1.5" in sent and "isn't loaded" in sent
        await db.close()

    async def test_other_plugin_overrides(self):
        """LLMPlugin's handler is trylast: another plugin's message wins."""
        plugin, channel, db = await build_plugin_and_channel()
        plugin._client = MagicMock()
        plugin._client.chat = AsyncMock(side_effect=_http_error(NOT_LOADED_BODY))
        plugin.pm.register(LLMPlugin(), name="llm_for_error")

        class Override:
            @hookimpl
            async def on_llm_error(self, channel, error):
                return "custom"

        plugin.pm.register(Override(), name="override")

        await plugin.on_message(channel=channel, sender="user", text="hi")
        await drain(plugin, channel)

        assert plugin.pm.ahook.send_message.call_args.kwargs["text"] == "custom"
        await db.close()


class TestWaitValidation:
    def test_rejects_non_numeric_wait(self):
        import pytest

        cfg = {"base_url": "http://x", "model": "m", "model_not_loaded_wait": None}
        with pytest.raises(ValueError, match="model_not_loaded_wait"):
            LLMPlugin()._create_client(cfg)

    def test_rejects_negative_wait(self):
        import pytest

        cfg = {"base_url": "http://x", "model": "m", "model_not_loaded_wait": -1}
        with pytest.raises(ValueError, match="model_not_loaded_wait"):
            LLMPlugin()._create_client(cfg)

    def test_rejects_non_finite_wait(self):
        import pytest

        for value in (float("inf"), float("nan")):
            cfg = {"base_url": "http://x", "model": "m", "model_not_loaded_wait": value}
            with pytest.raises(ValueError, match="model_not_loaded_wait"):
                LLMPlugin()._create_client(cfg)
