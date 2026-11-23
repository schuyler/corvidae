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
import socket
from fake_clock import clock_pm
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from corvidae.mcp_client import McpClientPlugin
from mcp_http_server import TOKEN


async def _start(servers: dict) -> McpClientPlugin:
    """Init and start a plugin with the given server configs."""
    plugin = McpClientPlugin(None)
    config = {"mcp": {"servers": servers}}
    await plugin.on_init(pm=clock_pm(), config=config)
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

    async def test_control_characters_in_resolved_value_are_rejected_unlogged(
        self, monkeypatch, caplog
    ):
        # A token with a trailing newline would make httpx raise an error that
        # quotes the header value into a logged traceback. It is refused as a
        # config error instead, without the value, and not retried.
        monkeypatch.setenv("TEST_MCP_TOKEN", TOKEN + "\n")
        with caplog.at_level(logging.DEBUG):
            plugin = await _start({
                "wiki": {
                    "transport": "streamable_http",
                    "url": "http://127.0.0.1:1/mcp",
                    "headers": {"Authorization": "Bearer ${TEST_MCP_TOKEN}"},
                }
            })
        try:
            warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
            assert len(warnings) == 1, [r.getMessage() for r in warnings]
            assert "TEST_MCP_TOKEN" in warnings[0].getMessage()
            assert not any(TOKEN in str(r.__dict__) for r in caplog.records)
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


def _closed_port_url() -> str:
    """A localhost URL with nothing listening: bind port 0, then close it."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}/mcp"


def _two_servers(mcp_url: str, down_url: str) -> dict:
    """A working bearer-authenticated server plus one with nothing listening."""
    return {
        "down": {"transport": "streamable_http", "url": down_url,
                 "tool_prefix": "d", "connect_timeout_seconds": 2},
        "wiki": {"transport": "streamable_http", "url": mcp_url,
                 "tool_prefix": "w",
                 "headers": {"Authorization": f"Bearer {TOKEN}"},
                 "connect_timeout_seconds": 5},
    }


class TestUnreachableServer:
    async def test_unreachable_server_is_pending_and_others_connect(self, mcp_url):
        # R1: a closed-port server does not stop on_start; it is queued.
        plugin = await _start(_two_servers(mcp_url, _closed_port_url()))
        try:
            assert list(plugin._pending_servers) == ["down"]
            assert [s.name for s in plugin._servers] == ["wiki"]
        finally:
            await plugin.on_stop()

    async def test_retry_loop_survives_unreachable_server(
        self, mcp_url, pm, kernel_clock
    ):
        # R2 (loop): the loop makes a second attempt after the first fails.
        clock, _ = kernel_clock
        plugin = McpClientPlugin(None)
        config = {"mcp": {"servers": _two_servers(mcp_url, _closed_port_url())}}
        attempts: list[str] = []
        real_connect = plugin._connect_server

        async def counting(name, cfg):
            attempts.append(name)
            await real_connect(name, cfg)

        plugin._connect_server = counting
        await plugin.on_init(pm=pm, config=config)
        await plugin.on_start(config=config)
        try:
            assert attempts.count("down") == 1
            await clock.advance(10)
            assert attempts.count("down") == 2
            assert not plugin._retry_task.done()
        finally:
            await plugin.on_stop()

    async def test_tool_call_with_unreachable_server_pending(self, mcp_url):
        # R2 (fast path): the tool call retries the down server and still
        # returns the working server's result.
        plugin = await _start(_two_servers(mcp_url, _closed_port_url()))
        try:
            tools = {t.name: t for t in plugin._cached_tools}
            assert await tools["w__echo"].fn(text="hello") == "hello"
        finally:
            await plugin.on_stop()

    async def test_connect_warning_names_server_url_and_error_type(
        self, mcp_url, caplog
    ):
        # R4: the warning's message line names server, URL and ConnectError.
        down_url = _closed_port_url()
        with caplog.at_level(logging.WARNING, logger="corvidae.mcp_client"):
            plugin = await _start(_two_servers(mcp_url, down_url))
        try:
            messages = [r.getMessage() for r in caplog.records
                        if r.levelno == logging.WARNING]
            assert any(
                "'down'" in m and down_url in m and "ConnectError" in m
                for m in messages
            ), messages
        finally:
            await plugin.on_stop()


class TestBareCancelBackstop:
    async def test_on_start_queues_server_on_uncancelled_cancellederror(
        self, monkeypatch
    ):
        # Backstop (on_start): a CancelledError nobody asked for is a
        # per-server failure, not a reason to abort startup.
        async def boom(self, name, cfg):
            raise asyncio.CancelledError()

        monkeypatch.setattr(McpClientPlugin, "_connect_server", boom)
        plugin = await _start({
            "down": {"transport": "streamable_http",
                     "url": "http://127.0.0.1:1/mcp"},
        })
        try:
            assert list(plugin._pending_servers) == ["down"]
        finally:
            await plugin.on_stop()

    async def test_retry_round_keeps_server_pending_on_uncancelled_cancellederror(
        self, monkeypatch
    ):
        # Backstop (retry round): same bare CancelledError, server already
        # pending; the round returns and the server stays pending.
        async def boom(self, name, cfg):
            raise asyncio.CancelledError()

        plugin = McpClientPlugin(None)
        config = {"mcp": {"servers": {}}}
        await plugin.on_init(pm=clock_pm(), config=config)
        plugin._pending_servers = {
            "down": {"transport": "streamable_http",
                     "url": "http://127.0.0.1:1/mcp"},
        }
        monkeypatch.setattr(McpClientPlugin, "_connect_server", boom)
        assert await plugin._retry_round() is False
        assert list(plugin._pending_servers) == ["down"]


class TestThirdPartyLogsDoNotCarryTheCredential:
    """#595: the MCP SDK and httpx log the configured URL whole, so a userinfo
    password or a query token reaches the daemon log unless something redacts
    records corvidae does not emit."""

    PASSWORD = "hunter2-not-the-bearer"

    def _url_with_userinfo(self, mcp_url: str) -> str:
        scheme, _, rest = mcp_url.partition("://")
        return f"{scheme}://alfred:{self.PASSWORD}@{rest}"

    async def test_a_401_with_userinfo_never_logs_the_password(
        self, mcp_url, caplog
    ):
        # The server refuses the request (no bearer header) after accepting the
        # connection: the shape #595 was reported in.
        url = self._url_with_userinfo(mcp_url)
        with caplog.at_level(logging.DEBUG):
            plugin = await _start({
                "wiki": {"transport": "streamable_http", "url": url,
                         "connect_timeout_seconds": 5},
            })
            try:
                assert plugin._servers == [], "the connect should have failed"
            finally:
                await plugin.on_stop()

        assert caplog.records, "nothing was logged, so this proves nothing"
        leaks = [r.getMessage() for r in caplog.records if self.PASSWORD in r.getMessage()]
        assert leaks == [], f"a log record carries the password: {leaks[0][:120]}"
        # caplog.text renders tracebacks too, where the SDK quotes the URL.
        assert self.PASSWORD not in caplog.text

    async def test_a_401_with_a_query_token_never_logs_the_token(
        self, mcp_url, caplog
    ):
        # The other way a credential reaches a URL, and the reason the filter is
        # the fix rather than stripping userinfo alone.
        token = "query-token-9d1e"
        with caplog.at_level(logging.DEBUG):
            plugin = await _start({
                "wiki": {"transport": "streamable_http",
                         "url": f"{mcp_url}?token={token}",
                         "connect_timeout_seconds": 5},
            })
            try:
                assert plugin._servers == []
            finally:
                await plugin.on_stop()

        assert caplog.records, "nothing was logged, so this proves nothing"
        leaks = [r.getMessage() for r in caplog.records if token in r.getMessage()]
        assert leaks == [], f"a log record carries the query token: {leaks[0][:120]}"
        # caplog.text renders tracebacks too, where the SDK quotes the URL.
        assert token not in caplog.text

    async def test_an_sse_401_after_connect_never_logs_the_password(
        self, caplog
    ):
        # The #595 scenario proper: the SSE stream is accepted, then the POST
        # that follows is refused with 401. The SDK logs that at ERROR
        # ("Error in post_writer") with the credential in the traceback, not
        # the message, so the whole rendered log text is checked.
        import uvicorn
        from mcp.server.mcpserver import MCPServer

        inner = MCPServer("echo-sse").sse_app()

        async def refuse_posts(scope, receive, send):
            # GET (the SSE stream) and lifespan pass; any POST gets a 401.
            if scope["type"] == "http" and scope["method"] == "POST":
                await send({"type": "http.response.start", "status": 401,
                            "headers": []})
                await send({"type": "http.response.body", "body": b""})
                return
            await inner(scope, receive, send)

        # Port 0: the OS picks a free port at bind time.
        server = uvicorn.Server(uvicorn.Config(
            refuse_posts, host="127.0.0.1", port=0, log_level="warning",
            lifespan="on"))
        serve_task = asyncio.create_task(server.serve())
        async with asyncio.timeout(5):
            while not server.started:
                await asyncio.sleep(0.02)
        port = server.servers[0].sockets[0].getsockname()[1]
        url = f"http://alfred:{self.PASSWORD}@127.0.0.1:{port}/sse"
        try:
            with caplog.at_level(logging.DEBUG):
                plugin = await _start({
                    "old": {"transport": "sse", "url": url,
                            "connect_timeout_seconds": 3},
                })
                try:
                    assert plugin._servers == [], "the connect should have failed"
                finally:
                    await plugin.on_stop()
        finally:
            server.should_exit = True
            await serve_task

        assert caplog.records, "nothing was logged, so this proves nothing"
        assert self.PASSWORD not in caplog.text
