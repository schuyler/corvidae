"""Tests for the embeddings client surface and LLM role generalization
(Phase 1a WP1a.3, bootstrap-mapping §4.3).

- LLMClient.embed POSTs {base_url}/embeddings and unpacks vectors in order.
- LLMPlugin builds a client per role under llm:; get_client(role) falls
  back to main for unconfigured or unknown roles.
- llm.embedding requires a ``dimensions`` key, validated at startup.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from corvidae.llm import LLMClient
from corvidae.llm_plugin import LLMPlugin


BASE_URL = "http://localhost:8080"
MODEL = "embed-model"

EMBEDDING_RESPONSE = {
    "object": "list",
    "data": [
        {"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]},
        {"object": "embedding", "index": 1, "embedding": [0.4, 0.5, 0.6]},
    ],
    "model": MODEL,
    "usage": {"prompt_tokens": 8, "total_tokens": 8},
}


def _make_mock_response(status: int = 200, json_body: dict | None = None):
    """Build a mock aiohttp response."""
    response = AsyncMock()
    response.status = status
    response.json = AsyncMock(return_value=json_body or EMBEDDING_RESPONSE)
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
    post_cm = AsyncMock()
    post_cm.__aenter__ = AsyncMock(return_value=response)
    post_cm.__aexit__ = AsyncMock(return_value=False)
    session.post = MagicMock(return_value=post_cm)
    session.close = AsyncMock()
    return session


class TestLLMClientEmbed:
    async def test_embed_sends_payload_and_unpacks_vectors_in_order(self):
        client = LLMClient(base_url=BASE_URL, model=MODEL)
        client.session = _make_mock_session(_make_mock_response())

        vectors = await client.embed(["alpha", "beta"], kind="document")

        assert vectors == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        call_args = client.session.post.call_args
        assert call_args[0][0] == f"{BASE_URL}/embeddings"
        assert call_args[1]["json"] == {"model": MODEL, "input": ["alpha", "beta"]}

    async def test_embed_raises_on_terminal_failure(self):
        """Callers own their degradation — embed raises, never swallows."""
        from aiohttp import ClientResponseError

        client = LLMClient(base_url=BASE_URL, model=MODEL, max_retries=0)
        client.session = _make_mock_session(_make_mock_response(status=500))

        with pytest.raises(ClientResponseError):
            await client.embed(["alpha"], kind="document")

    async def test_embed_fires_observer_response_with_usage(self):
        """The Phase 0 observer sees exactly one response per embed call."""
        client = LLMClient(base_url=BASE_URL, model=MODEL)
        client.session = _make_mock_session(_make_mock_response())
        observer = MagicMock()
        observer.request = AsyncMock()
        observer.response = AsyncMock()
        client.observer = observer

        await client.embed(["alpha", "beta"], kind="document")

        observer.response.assert_called_once()
        kwargs = observer.response.call_args.kwargs
        assert kwargs["usage"] == {"prompt_tokens": 8, "total_tokens": 8}
        assert kwargs["error"] is None

    async def test_embed_retries_on_transient_status(self):
        """The retry logic is shared with chat(): 500 then 200 succeeds."""
        client = LLMClient(
            base_url=BASE_URL, model=MODEL, max_retries=1, retry_base_delay=1.0
        )
        resp_500 = _make_mock_response(status=500)
        resp_200 = _make_mock_response(status=200)

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
            vectors = await client.embed(["alpha", "beta"], kind="document")

        assert vectors == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        assert session.post.call_count == 2

    # --- RED: design items 1, 2, 3 ---

    async def test_embed_document_kind_prepends_document_prefix(self):
        """item 1: kind='document' prepends document_prefix to every batch element."""
        client = LLMClient(
            base_url=BASE_URL, model=MODEL,
            document_prefix="search_document: ",
            query_prefix="search_query: ",
        )
        client.session = _make_mock_session(_make_mock_response())
        await client.embed(["alpha", "beta"], kind="document")
        payload = client.session.post.call_args[1]["json"]
        assert payload["input"] == ["search_document: alpha", "search_document: beta"]

    async def test_embed_query_kind_prepends_query_prefix(self):
        """item 1: kind='query' prepends query_prefix to every batch element."""
        client = LLMClient(
            base_url=BASE_URL, model=MODEL,
            document_prefix="search_document: ",
            query_prefix="search_query: ",
        )
        client.session = _make_mock_session(_make_mock_response())
        await client.embed(["alpha", "beta"], kind="query")
        payload = client.session.post.call_args[1]["json"]
        assert payload["input"] == ["search_query: alpha", "search_query: beta"]

    async def test_embed_empty_prefix_sends_texts_unchanged(self):
        """item 2: absent/empty prefixes leave the wire payload byte-identical to today."""
        client = LLMClient(base_url=BASE_URL, model=MODEL)
        client.session = _make_mock_session(_make_mock_response())
        await client.embed(["alpha", "beta"], kind="document")
        payload = client.session.post.call_args[1]["json"]
        assert payload["input"] == ["alpha", "beta"]

    async def test_embed_unknown_kind_raises_value_error(self):
        """item 3: a kind value other than 'document'/'query' raises ValueError."""
        client = LLMClient(base_url=BASE_URL, model=MODEL)
        client.session = _make_mock_session(_make_mock_response())
        with pytest.raises(ValueError):
            await client.embed(["alpha"], kind="passage")


class TestLLMPluginRoles:
    def _plugin_with_clients(self, roles: dict) -> LLMPlugin:
        """Build a bare plugin with the given role->client mapping."""
        plugin = LLMPlugin()
        for role, client in roles.items():
            plugin._clients[role] = client
        return plugin

    def test_get_client_embedding_returns_embedding_client(self):
        main, embedding = MagicMock(name="main"), MagicMock(name="embedding")
        plugin = self._plugin_with_clients({"main": main, "embedding": embedding})
        assert plugin.get_client("embedding") is embedding

    def test_get_client_embedding_falls_back_to_main(self):
        main = MagicMock(name="main")
        plugin = self._plugin_with_clients({"main": main})
        assert plugin.get_client("embedding") is main

    def test_get_client_unknown_role_falls_back_to_main(self):
        main = MagicMock(name="main")
        plugin = self._plugin_with_clients({"main": main})
        assert plugin.get_client("no-such-role") is main

    async def test_on_start_builds_client_per_role(self):
        """Every key under llm: becomes a client; embedding keeps dimensions."""
        config = {
            "llm": {
                "main": {"base_url": BASE_URL, "model": "chat-model"},
                "embedding": {
                    "base_url": "http://localhost:8081",
                    "model": MODEL,
                    "dimensions": 3,
                },
            }
        }
        plugin = LLMPlugin()
        await plugin.on_init(pm=None, config=config)
        with patch.object(LLMClient, "start", new_callable=AsyncMock):
            await plugin.on_start(config=config)

        assert plugin.get_client("embedding").model == MODEL
        assert plugin.get_client("embedding") is not plugin.get_client("main")
        assert plugin.embedding_dimensions == 3
        # Legacy accessors still work over the role dict.
        assert plugin.main_client is plugin.get_client("main")

    async def test_missing_dimensions_raises_at_startup(self):
        config = {
            "llm": {
                "main": {"base_url": BASE_URL, "model": "chat-model"},
                "embedding": {"base_url": BASE_URL, "model": MODEL},
            }
        }
        plugin = LLMPlugin()
        await plugin.on_init(pm=None, config=config)
        with patch.object(LLMClient, "start", new_callable=AsyncMock):
            with pytest.raises((KeyError, ValueError)):
                await plugin.on_start(config=config)

    def test_create_client_passes_prefix_params_from_config(self):
        """item 7: document_prefix/query_prefix under llm.embedding reach the client."""
        cfg = {
            "base_url": BASE_URL,
            "model": MODEL,
            "dimensions": 3,
            "document_prefix": "search_document: ",
            "query_prefix": "search_query: ",
        }
        plugin = LLMPlugin()
        client = plugin._create_client(cfg, role="embedding")
        # Both prefix attributes must be set on the constructed client.
        assert client.document_prefix == "search_document: "
        assert client.query_prefix == "search_query: "
