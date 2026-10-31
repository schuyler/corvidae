"""A real streamable-HTTP MCP server for tests, shared by several test files.

The SDK's ``MCPServer`` is served by uvicorn (see the ``mcp_url`` fixture in
``conftest.py``) behind a small ASGI wrapper that answers 401 unless the
request carries ``Authorization: Bearer <token>``, and records every HTTP
request it sees so tests can assert on the session lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mcp.server.mcpserver import MCPServer

TOKEN = "s3cret-test-token"


@dataclass
class RequestLog:
    """What the ASGI wrapper saw: (method, session id) per request, plus the
    session ids the server issued in responses."""

    requests: list[tuple[str, str | None]] = field(default_factory=list)
    issued_sessions: list[str] = field(default_factory=list)

    def deletes(self) -> list[str | None]:
        """Session ids of the DELETE (session termination) requests seen."""
        return [sid for method, sid in self.requests if method == "DELETE"]


def bearer_app(token: str, log: RequestLog | None = None):
    """An ASGI app: the MCP streamable-HTTP endpoint at /mcp, refusing any
    HTTP request without the expected bearer token and recording requests
    into ``log``."""
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
            sid = headers.get(b"mcp-session-id")
            if log is not None:
                log.requests.append(
                    (scope["method"], sid.decode() if sid else None)
                )
            if headers.get(b"authorization") != f"Bearer {token}".encode():
                await send({"type": "http.response.start", "status": 401, "headers": []})
                await send({"type": "http.response.body", "body": b""})
                return

            # Remember the session id the server issues, to match a later
            # DELETE against it.
            async def recording_send(message):
                if log is not None and message["type"] == "http.response.start":
                    for key, value in message.get("headers", []):
                        if key.lower() == b"mcp-session-id":
                            log.issued_sessions.append(value.decode())
                await send(message)

            await inner(scope, receive, recording_send)
            return
        await inner(scope, receive, send)

    return app
