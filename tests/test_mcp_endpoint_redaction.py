"""A failed MCP connect names the endpoint without its credentials (#386).

The connect-failure warnings print the server's endpoint so an operator can
tell which one is misconfigured. A server configured with a credential in the
URL -- userinfo or a query-string token -- would otherwise write it to the
daemon log at every failed connect and every retry.
"""

import logging

import pytest

import corvidae.mcp_client as mcp_client_mod
from corvidae.mcp_client import _server_endpoint
from fake_clock import clock_pm


# Secrets that must never reach a log line.
QUERY_SECRET = "SECRET1"
USERINFO_PASSWORD = "hunter2"


def test_query_string_token_is_not_logged():
    """A ?token= query in the endpoint never reaches the log."""
    endpoint = _server_endpoint(
        {"transport": "streamable_http",
         "url": f"https://wiki.example.com/mcp?token={QUERY_SECRET}"}
    )
    assert QUERY_SECRET not in endpoint
    assert "wiki.example.com/mcp" in endpoint


def test_userinfo_password_is_not_logged():
    """A user:pass@ userinfo section never reaches the log."""
    endpoint = _server_endpoint(
        {"transport": "sse",
         "url": f"https://user:{USERINFO_PASSWORD}@wiki.example.com/mcp"}
    )
    assert USERINFO_PASSWORD not in endpoint
    assert "wiki.example.com/mcp" in endpoint


def test_fragment_is_not_logged():
    """A fragment is dropped: it is free text the operator did not mean to log."""
    endpoint = _server_endpoint(
        {"transport": "streamable_http",
         "url": "https://wiki.example.com/mcp#section-secret"}
    )
    assert "section-secret" not in endpoint
    assert "wiki.example.com/mcp" in endpoint


def test_query_presence_is_still_visible():
    """Redaction keeps the fact that a query was configured, without its value."""
    endpoint = _server_endpoint(
        {"transport": "streamable_http",
         "url": f"https://wiki.example.com/mcp?token={QUERY_SECRET}"}
    )
    assert "?" in endpoint


def test_plain_url_is_unchanged():
    """A URL with no credentials is logged whole, as before."""
    endpoint = _server_endpoint(
        {"transport": "streamable_http", "url": "https://wiki.example.com/mcp"}
    )
    assert endpoint == "https://wiki.example.com/mcp"


def test_port_and_scheme_survive():
    """Scheme, host and port identify the endpoint; none is dropped."""
    endpoint = _server_endpoint(
        {"transport": "sse", "url": "http://127.0.0.1:8931/mcp?token=x"}
    )
    assert endpoint.startswith("http://127.0.0.1:8931")
    assert "x" not in endpoint.replace("mcp", "")


def test_stdio_command_is_unchanged():
    """A stdio server logs its command as before; only URLs are redacted."""
    endpoint = _server_endpoint(
        {"transport": "stdio", "command": "/usr/bin/mcp-server"}
    )
    assert endpoint == "/usr/bin/mcp-server"


def test_missing_key_does_not_raise():
    """A config that failed validation may lack the key; the endpoint is "?"."""
    assert _server_endpoint({}) == "?"


@pytest.mark.parametrize(
    "bad",
    [":::not a url:::", "", "http://[unclosed", None, 42],
)
def test_unparseable_value_does_not_raise(bad):
    """Redaction must never raise: it runs inside failure logging."""
    endpoint = _server_endpoint({"transport": "streamable_http", "url": bad})
    assert isinstance(endpoint, str)


# ---------------------------------------------------------------------------
# End-to-end: the two connect-failure warnings
# ---------------------------------------------------------------------------


@pytest.fixture
def secret_url_server():
    """A server config whose URL carries both a userinfo password and a token."""
    return {
        "transport": "streamable_http",
        "url": (
            f"https://user:{USERINFO_PASSWORD}@wiki.example.com/mcp"
            f"?token={QUERY_SECRET}"
        ),
    }


async def _plugin_with(servers: dict, failures: set[str]):
    """A started plugin whose named servers fail to connect."""
    plugin = mcp_client_mod.McpClientPlugin(None)
    config = {"mcp": {"servers": servers}}
    await plugin.on_init(pm=clock_pm(), config=config)

    async def connect(name, cfg):
        if name in failures:
            raise ConnectionError("refused")
        # A server that is not failing never registers, so it also stays
        # pending; the warning under test comes only from the failures.
        raise ConnectionError("refused")

    plugin._connect_server = connect
    return plugin


async def test_on_start_skip_line_hides_credentials(secret_url_server, caplog):
    """The on_start connect-failure warning carries no credential."""
    plugin = await _plugin_with({"wiki": secret_url_server}, {"wiki"})
    with caplog.at_level(logging.WARNING):
        await plugin.on_start({"mcp": {"servers": {"wiki": secret_url_server}}})

    joined = "\n".join(
        r.getMessage() for r in caplog.records if "wiki" in r.getMessage()
    )
    assert joined, "the connect failure must be logged"
    assert "wiki.example.com/mcp" in joined
    assert QUERY_SECRET not in joined
    assert USERINFO_PASSWORD not in joined


async def test_retry_round_line_hides_credentials(secret_url_server, caplog):
    """The retry-round reconnect failure warning carries no credential."""
    plugin = await _plugin_with({"wiki": secret_url_server}, {"wiki"})
    await plugin.on_start({"mcp": {"servers": {"wiki": secret_url_server}}})
    caplog.clear()

    with caplog.at_level(logging.WARNING):
        await plugin._retry_round(from_loop=True)

    joined = "\n".join(
        r.getMessage() for r in caplog.records if "wiki" in r.getMessage()
    )
    assert joined, "the reconnect failure must be logged"
    assert "wiki.example.com/mcp" in joined
    assert QUERY_SECRET not in joined
    assert USERINFO_PASSWORD not in joined


async def test_two_failing_servers_are_distinguishable(caplog):
    """R386.2: each failing server gets its own line naming its host."""
    servers = {
        "wiki": {"transport": "streamable_http", "url": "https://wiki.example.com/mcp"},
        "search": {"transport": "sse", "url": "https://search.example.com/sse"},
    }
    plugin = await _plugin_with(servers, {"wiki", "search"})
    with caplog.at_level(logging.WARNING):
        await plugin.on_start({"mcp": {"servers": servers}})

    messages = [r.getMessage() for r in caplog.records]
    assert any("wiki.example.com" in m for m in messages)
    assert any("search.example.com" in m for m in messages)
