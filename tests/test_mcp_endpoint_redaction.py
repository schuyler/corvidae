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
    # Exactly one marker: the path carries no "?" of its own, and a rebuilt
    # URL must not join its own delimiter to ours ("mcp??…").
    assert endpoint == "https://wiki.example.com/mcp?…"


def test_bad_port_does_not_raise():
    """A malformed authority is accepted by config validation and must not raise.

    `url: http://host:abc/mcp?token=S` passes the config check (a non-empty
    string) and is queued for retry. `parts.port` raises ValueError on it, and
    both call sites evaluate the endpoint inside an except handler, so a raise
    here turns a failed MCP connect into a daemon startup failure (#386 review).
    """
    endpoint = _server_endpoint(
        {"transport": "streamable_http", "url": "http://host:abc/mcp?token=S"}
    )
    assert endpoint == "(unparseable endpoint)"
    assert "S" not in endpoint.replace("unparseable", "")


def test_out_of_range_port_does_not_raise():
    """A port above 65535 raises from parts.port too."""
    endpoint = _server_endpoint(
        {"transport": "streamable_http", "url": "http://host:99999/mcp"}
    )
    assert isinstance(endpoint, str)
    assert endpoint == "(unparseable endpoint)"


def test_ipv6_literal_keeps_its_brackets():
    """An IPv6 host is re-bracketed, or the colons read as a port separator."""
    endpoint = _server_endpoint(
        {"transport": "sse", "url": "http://[::1]:8931/mcp?token=x"}
    )
    assert endpoint == "http://[::1]:8931/mcp?…"
    assert "x" not in endpoint


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


async def _plugin_with(servers: dict) -> "mcp_client_mod.McpClientPlugin":
    """A started plugin whose configured servers all fail to connect."""
    plugin = mcp_client_mod.McpClientPlugin(None)
    config = {"mcp": {"servers": servers}}
    await plugin.on_init(pm=clock_pm(), config=config)

    async def connect(name, cfg):
        raise ConnectionError("refused")

    plugin._connect_server = connect
    return plugin


async def test_on_start_skip_line_hides_credentials(secret_url_server, caplog):
    """The on_start connect-failure warning carries no credential."""
    plugin = await _plugin_with({"wiki": secret_url_server})
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
    plugin = await _plugin_with({"wiki": secret_url_server})
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
    plugin = await _plugin_with(servers)
    with caplog.at_level(logging.WARNING):
        await plugin.on_start({"mcp": {"servers": servers}})

    messages = [r.getMessage() for r in caplog.records]
    assert any("wiki.example.com" in m for m in messages)
    assert any("search.example.com" in m for m in messages)


# ---------------------------------------------------------------------------
# The config-error skip warning
#
# A server whose *config* is invalid is skipped before any transport opens,
# and that warning quotes the offending value. A credential can sit in a value
# the validator rejects -- a server entry that should be a mapping but is the
# URL itself, or a headers list holding a bearer token -- so the skip line must
# not quote it whole (#386 review).
# ---------------------------------------------------------------------------


async def test_config_error_skip_line_hides_a_url_with_credentials(caplog):
    """A server entry that is the URL itself is not quoted into the log.

    `mcp.servers.wiki` that is a bare string instead of a mapping is a config
    error, and the message used to interpolate that string verbatim.
    """
    secret_url = (
        f"https://user:{USERINFO_PASSWORD}@wiki.example.com/mcp"
        f"?token={QUERY_SECRET}"
    )
    plugin = await _plugin_with({"wiki": secret_url})
    with caplog.at_level(logging.WARNING):
        await plugin.on_start({"mcp": {"servers": {"wiki": secret_url}}})

    joined = "\n".join(
        r.getMessage() for r in caplog.records if "wiki" in r.getMessage()
    )
    assert joined, "the config-error skip must be logged"
    # The operator still learns which server and which key to fix.
    assert "wiki" in joined
    assert "mapping" in joined
    assert QUERY_SECRET not in joined
    assert USERINFO_PASSWORD not in joined


async def test_config_error_skip_line_hides_a_headers_bearer_token(caplog):
    """A rejected `headers` list holding a token is not quoted into the log."""
    headers_secret = "SECRET3"
    bad_headers = [f"Bearer {headers_secret}"]
    server = {"transport": "streamable_http", "url": "https://h.example.com/mcp",
              "headers": bad_headers}
    plugin = await _plugin_with({"wiki": server})
    with caplog.at_level(logging.WARNING):
        await plugin.on_start({"mcp": {"servers": {"wiki": server}}})

    joined = "\n".join(
        r.getMessage() for r in caplog.records if "wiki" in r.getMessage()
    )
    assert joined, "the config-error skip must be logged"
    assert "headers" in joined
    assert "wiki" in joined
    assert headers_secret not in joined

