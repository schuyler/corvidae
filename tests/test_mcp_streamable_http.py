"""McpClientPlugin over streamable HTTP, with request headers (#190).

These tests run a real MCP server — the SDK's ``MCPServer`` served by
uvicorn on an ephemeral localhost port — behind a small ASGI wrapper that
answers 401 unless the request carries ``Authorization: Bearer <token>``.
That is the shape of the server the issue names (otterwiki's ``/mcp``), and
it exercises the transport and the headers end to end rather than through
mocks.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import uvicorn
from mcp.server.mcpserver import MCPServer

from corvidae.mcp_client import McpClientPlugin

TOKEN = "s3cret-test-token"


def _bearer_app(token: str):
    """An ASGI app: the MCP streamable-HTTP endpoint at /mcp, refusing any
    HTTP request without the expected bearer token."""
    server = MCPServer("echo-http")

    # The one tool the tests call.
    @server.tool()
    def echo(text: str) -> str:
        return text

    inner = server.streamable_http_app()

    async def app(scope, receive, send):
        # Check the bearer token on HTTP requests; lifespan passes straight
        # through so the session manager starts and stops with uvicorn.
        if scope["type"] == "http":
            headers = dict(scope["headers"])
            if headers.get(b"authorization") != f"Bearer {token}".encode():
                await send({"type": "http.response.start", "status": 401, "headers": []})
                await send({"type": "http.response.body", "body": b""})
                return
        await inner(scope, receive, send)

    return app


@pytest.fixture
async def mcp_url() -> AsyncIterator[str]:
    """Serve the bearer-protected MCP app on a free port; yield its /mcp URL."""
    # Port 0: the OS picks a free port at bind time, so there is no race.
    server = uvicorn.Server(
        uvicorn.Config(_bearer_app(TOKEN), host="127.0.0.1", port=0,
                       log_level="warning", lifespan="on")
    )
    task = asyncio.create_task(server.serve())
    # Wait for the listener before the test connects.
    async with asyncio.timeout(5):
        while not server.started:
            await asyncio.sleep(0.02)
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        server.should_exit = True
        await task


async def _start(servers: dict) -> McpClientPlugin:
    """Init and start a plugin with the given server configs."""
    plugin = McpClientPlugin(None)
    config = {"mcp": {"servers": servers}}
    await plugin.on_init(pm=MagicMock(), config=config)
    await plugin.on_start(config=config)
    return plugin


class TestStreamableHttp:
    async def test_connects_with_env_bearer_token_and_calls_tool(
        self, mcp_url, monkeypatch
    ):
        # The token comes from the environment, never a literal in config.
        monkeypatch.setenv("TEST_MCP_TOKEN", TOKEN)
        plugin = await _start({
            "wiki": {
                "transport": "streamable_http",
                "url": mcp_url,
                "tool_prefix": "w",
                "headers": {"Authorization": "Bearer ${TEST_MCP_TOKEN}"},
                "connect_timeout_seconds": 5,
            }
        })
        try:
            assert [s.name for s in plugin._servers] == ["wiki"]
            tools = {t.name: t for t in plugin._cached_tools}
            assert list(tools) == ["w__echo"]
            assert await tools["w__echo"].fn(text="hello") == "hello"
        finally:
            await plugin.on_stop()

    async def test_without_headers_server_refuses_and_is_left_pending(self, mcp_url):
        # No Authorization header: the server's 401 fails the connect, and
        # the server stays queued for the lazy retry like any failed connect.
        plugin = await _start({
            "wiki": {"transport": "streamable_http", "url": mcp_url,
                     "connect_timeout_seconds": 5},
        })
        try:
            assert plugin._servers == []
            assert "wiki" in plugin._pending_servers
        finally:
            await plugin.on_stop()


class TestHeaderEnvironment:
    async def test_missing_env_var_names_server_header_and_variable(
        self, mcp_url, monkeypatch, caplog
    ):
        # An unset variable is a config error: exactly one warning saying what
        # to set, and no lazy retry, because the daemon's environment cannot
        # change. A second, correctly configured server still connects.
        monkeypatch.delenv("TEST_MCP_MISSING", raising=False)
        monkeypatch.setenv("TEST_MCP_TOKEN", TOKEN)
        with caplog.at_level(logging.WARNING, logger="corvidae.mcp_client"):
            plugin = await _start({
                "wiki": {
                    "transport": "streamable_http",
                    "url": "http://127.0.0.1:1/mcp",
                    "headers": {"Authorization": "Bearer ${TEST_MCP_MISSING}"},
                },
                "good": {
                    "transport": "streamable_http",
                    "url": mcp_url,
                    "headers": {"Authorization": "Bearer ${TEST_MCP_TOKEN}"},
                    "connect_timeout_seconds": 5,
                },
            })
        try:
            warnings = [r for r in caplog.records
                        if r.levelno >= logging.WARNING
                        and r.name == "corvidae.mcp_client"]
            assert len(warnings) == 1, [r.getMessage() for r in warnings]
            message = warnings[0].getMessage()
            for needle in ("wiki", "Authorization", "TEST_MCP_MISSING", "environment"):
                assert needle in message, message
            assert [s.name for s in plugin._servers] == ["good"]
            assert "wiki" not in plugin._pending_servers
        finally:
            await plugin.on_stop()

    @pytest.mark.parametrize("token_env", [TOKEN, "wrong-token"])
    async def test_header_values_are_not_logged(
        self, mcp_url, monkeypatch, caplog, token_env
    ):
        # Neither the accepted token (success path) nor a rejected one (the
        # 401 failure path, which logs with a traceback) appears anywhere in
        # a log record: message, extra fields or formatted traceback.
        monkeypatch.setenv("TEST_MCP_TOKEN", token_env)
        with caplog.at_level(logging.DEBUG):
            plugin = await _start({
                "wiki": {
                    "transport": "streamable_http",
                    "url": mcp_url,
                    "headers": {"Authorization": "Bearer ${TEST_MCP_TOKEN}"},
                    "connect_timeout_seconds": 5,
                }
            })
        try:
            assert bool(plugin._servers) == (token_env == TOKEN)
            formatter = logging.Formatter()
            for record in caplog.records:
                text = str(record.__dict__)
                if record.exc_info:
                    text += formatter.formatException(record.exc_info)
                assert token_env not in text, record.getMessage()
        finally:
            await plugin.on_stop()

    async def test_sse_transport_sends_headers(self, monkeypatch):
        # sse_client receives the resolved headers. Mocked: the SDK's
        # MCPServer has no bearer-protected SSE fixture worth standing up.

        monkeypatch.setenv("TEST_MCP_TOKEN", TOKEN)
        session = MagicMock()
        session.initialize = AsyncMock()
        session.list_tools = AsyncMock(return_value=MagicMock(tools=[]))
        session_cm = AsyncMock()
        session_cm.__aenter__ = AsyncMock(return_value=session)
        session_cm.__aexit__ = AsyncMock(return_value=False)
        sse_cm = AsyncMock()
        sse_cm.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
        sse_cm.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("mcp.client.sse.sse_client", return_value=sse_cm) as sse_client,
            patch("mcp.ClientSession", return_value=session_cm),
        ):
            plugin = await _start({
                "old": {
                    "transport": "sse",
                    "url": "http://127.0.0.1:1/sse",
                    "headers": {"Authorization": "Bearer ${TEST_MCP_TOKEN}"},
                }
            })
        try:
            assert sse_client.call_args.kwargs["headers"] == {
                "Authorization": f"Bearer {TOKEN}"
            }
        finally:
            await plugin.on_stop()
