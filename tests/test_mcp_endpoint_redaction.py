"""A failed MCP connect names the endpoint without its credentials (#386).

The connect-failure warnings print the server's endpoint so an operator can
tell which one is misconfigured. A server configured with a credential in the
URL -- userinfo or a query-string token -- would otherwise write it to the
daemon log at every failed connect and every retry.
"""

import io
import logging

import pytest
from fake_clock import clock_pm

import corvidae.mcp_client as mcp_client_mod
from corvidae.mcp_client import _server_endpoint

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


async def test_config_error_skip_line_hides_a_headers_bare_string(caplog):
    """A `headers` value given as a bare string is also a credential.

    A string is the most likely malformed shape for a mapping key, and
    `"Bearer SECRET"` in it is the token itself. The type test used to run
    before the key was consulted, so this shape echoed it whole (review).
    """
    headers_secret = "SECRET4"
    server = {"transport": "streamable_http", "url": "https://h.example.com/mcp",
              "headers": f"Bearer {headers_secret}"}
    plugin = await _plugin_with({"wiki": server})
    with caplog.at_level(logging.WARNING):
        await plugin.on_start({"mcp": {"servers": {"wiki": server}}})

    joined = "\n".join(
        r.getMessage() for r in caplog.records if "wiki" in r.getMessage()
    )
    assert joined, "the config-error skip must be logged"
    assert "headers" in joined
    assert headers_secret not in joined


async def test_config_error_skip_line_hides_a_url_inside_a_container(caplog):
    """A credential-bearing URL is redacted wherever the skip line echoes it.

    Both cases must be *rejected by validation*, or there is no skip line for
    the assertion to check and the test asserts nothing. `args: [endpoint]` is
    a list of strings and is accepted, so it belongs to the connect path,
    which `_server_endpoint` already covers (review).
    """
    endpoint = (
        f"https://user:{USERINFO_PASSWORD}@wiki.example.com/mcp"
        f"?token={QUERY_SECRET}"
    )
    cases = [
        {"transport": "streamable_http", "url": [endpoint]},
        # A mixed list is rejected; a list of only the URL is not.
        {"transport": "stdio", "command": "c", "args": [endpoint, 5]},
    ]
    for server in cases:
        plugin = await _plugin_with({"wiki": server})
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            await plugin.on_start({"mcp": {"servers": {"wiki": server}}})

        joined = "\n".join(
            r.getMessage() for r in caplog.records if "wiki" in r.getMessage()
        )
        assert "config error" in joined, (
            f"this case must be a config error, or it asserts nothing: {server!r}"
        )
        assert "wiki" in joined
        assert QUERY_SECRET not in joined, f"query token leaked via {server!r}"
        assert USERINFO_PASSWORD not in joined, f"userinfo leaked via {server!r}"


async def test_config_error_skip_line_hides_a_url_given_as_the_transport(caplog):
    """The `transport:` rejection names the key without quoting a secret.

    This was the one config-error site still interpolating its value raw
    (review).
    """
    endpoint = (
        f"https://user:{USERINFO_PASSWORD}@wiki.example.com/mcp"
        f"?token={QUERY_SECRET}"
    )
    server = {"transport": endpoint, "url": "https://h.example.com/mcp"}
    plugin = await _plugin_with({"wiki": server})
    with caplog.at_level(logging.WARNING):
        await plugin.on_start({"mcp": {"servers": {"wiki": server}}})

    joined = "\n".join(
        r.getMessage() for r in caplog.records if "wiki" in r.getMessage()
    )
    assert joined, "the config-error skip must be logged"
    assert "transport" in joined, "the message must still name the key"
    assert QUERY_SECRET not in joined
    assert USERINFO_PASSWORD not in joined



def test_validate_mcp_config_hides_a_url_that_is_the_whole_block():
    """`mcp:` given as the URL itself is reported without its credentials.

    `Runtime` prints this as `Config error: … — check agent.yaml`, so the
    value reaches stderr; it was interpolated raw (review).
    """
    from corvidae.mcp_client import validate_mcp_config

    endpoint = (
        f"https://user:{USERINFO_PASSWORD}@wiki.example.com/mcp"
        f"?token={QUERY_SECRET}"
    )
    with pytest.raises(ValueError) as excinfo:
        validate_mcp_config({"mcp": endpoint})

    message = str(excinfo.value)
    assert "mcp" in message
    assert QUERY_SECRET not in message
    assert USERINFO_PASSWORD not in message



def test_a_pathologically_nested_value_does_not_raise():
    """Describing a rejected value must not raise, however deep it nests.

    The recursion was unbounded, so a deeply nested or self-referential value
    raised RecursionError from inside config-error handling and replaced the
    error it was describing (review).
    """
    from corvidae.mcp_client import _describe_config_value

    deep: list = []
    cur = deep
    for _ in range(5000):
        nxt: list = []
        cur.append(nxt)
        cur = nxt
    assert isinstance(_describe_config_value(deep), str)

    self_ref: dict = {}
    self_ref["self"] = self_ref
    assert isinstance(_describe_config_value(self_ref), str)


# ---------------------------------------------------------------------------
# The invariant, as a property rather than a list of shapes
#
# Four rounds of this review each found a shape the previous round's fix did
# not cover: a list, then a bare string, then a URL inside a container, then
# a URL `urlsplit` rejects. Enumerating shapes is what kept failing. The
# property the helper must hold is narrower and total:
#
#   No string containing `@`, `?` or `://` is ever echoed into a message.
#
# Those three markers are userinfo, query and scheme. Anything without them
# is echoed, because a wrong value is the diagnostic the operator needs.
# ---------------------------------------------------------------------------


# Every one of these carries a credential and must never appear.
CREDENTIAL_STRINGS = [
    "https://user:hunter2@h.example.com/mcp?token=SECRET1",
    # shapes urlsplit rejects, one per reason
    "https://user:hunter2@h.example.com:99999/mcp?token=SECRET1",  # port range
    "https://user:hunter2@h.example.com:abc/mcp?token=SECRET1",    # bad port
    "https://user:p/ss@h.example.com/mcp?token=SECRET1",           # '/' in pw
    "https://user:hunter2@[unclosed/mcp?token=SECRET1",            # bad v6
    # shapes that are not parseable as URLs at all -- the plausible typos
    "wiki.example.com/mcp?token=SECRET1",                          # no scheme
    "user:hunter2@internal:8080/mcp",                              # no scheme
]

# Values that carry nothing secret and must stay visible.
INNOCUOUS_VALUES = ["abc", "[1]", 42, None, True, 1.5]


@pytest.mark.parametrize("secret", CREDENTIAL_STRINGS)
def test_no_credential_string_ever_reaches_a_description(secret):
    """The invariant, over the shapes urlsplit accepts and rejects."""
    from corvidae.mcp_client import _describe_config_value

    for value in (
        secret,
        [secret],
        (secret,),
        {"k": secret},
        {secret: 1},          # a URL as a dict *key*
        [[secret]],           # nested
        {"a": [{"b": secret}]},
    ):
        described = _describe_config_value(value)
        assert secret not in described, (
            f"credential echoed for {value!r}: {described!r}"
        )
        # The secret itself, not just the whole string, must be absent.
        assert "SECRET1" not in described, f"token echoed for {value!r}"
        assert "hunter2" not in described, f"password echoed for {value!r}"


@pytest.mark.parametrize("value", INNOCUOUS_VALUES)
def test_innocuous_values_are_still_shown(value):
    """The invariant must not swallow the diagnostic.

    A wrong value is what tells the operator what to fix, so anything without
    a credential marker stays visible.
    """
    from corvidae.mcp_client import _describe_config_value

    described = _describe_config_value(value)
    assert repr(value) in described or str(value) in described


def test_describing_never_raises_on_a_cycle():
    """No shape of input may raise from inside config-error handling."""
    from corvidae.mcp_client import _describe_config_value

    cyclic: list = []
    cyclic.append(cyclic)
    assert isinstance(_describe_config_value(cyclic), str)

    d: dict = {}
    d["self"] = d
    assert isinstance(_describe_config_value(d), str)


# ---------------------------------------------------------------------------
# The rendered record, not just the message
#
# Both warnings used to attach the exception, and an HTTP client quotes the URL
# in its errors: httpx2.HTTPStatusError reads `Client error '401 Unauthorized'
# for url '<the configured url>'`. So a server whose URL carries credentials
# gets no traceback; the warning carries the exception's text with every URL
# redacted instead. These tests render the whole record as a handler would.
# ---------------------------------------------------------------------------


def _rendered(records) -> str:
    """Everything a log handler would write, traceback included."""
    handler = logging.StreamHandler(io.StringIO())
    handler.setFormatter(logging.Formatter("%(message)s"))
    return "\n".join(handler.format(r) for r in records)


class _UrlBearingError(Exception):
    """Stands in for httpx2.HTTPStatusError, which formats the URL in."""

    def __init__(self, url: str) -> None:
        super().__init__(f"Client error '401 Unauthorized' for url '{url}'")


async def _failing_with(servers: dict, make_exc) -> "mcp_client_mod.McpClientPlugin":
    """A started plugin whose connects raise ``make_exc(cfg)``."""
    plugin = await _plugin_with(servers)

    async def connect(name, cfg):
        raise make_exc(cfg)

    plugin._connect_server = connect
    return plugin


async def _render_failure(servers: dict, make_exc, caplog, *, retry: bool) -> str:
    """Render the on_start warning, or the retry-round one if ``retry``."""
    plugin = await _failing_with(servers, make_exc)
    if retry:
        await plugin.on_start({"mcp": {"servers": servers}})
        caplog.clear()
    with caplog.at_level(logging.WARNING):
        if retry:
            await plugin._retry_round(from_loop=True)
        else:
            await plugin.on_start({"mcp": {"servers": servers}})
    return _rendered(caplog.records)


@pytest.mark.parametrize("retry", [False, True], ids=["on_start", "retry_round"])
async def test_the_rendered_record_hides_a_url_carrying_exception(
    secret_url_server, caplog, retry
):
    """No credential in the rendered record at either site, and the status
    the operator needs survives."""
    rendered = await _render_failure(
        {"wiki": secret_url_server},
        lambda cfg: _UrlBearingError(cfg["url"]),
        caplog,
        retry=retry,
    )
    assert "wiki.example.com/mcp" in rendered
    assert "401 Unauthorized" in rendered, "the status text was lost"
    assert "traceback omitted" in rendered
    assert "Traceback" not in rendered
    assert QUERY_SECRET not in rendered, "query token reached the record"
    assert USERINFO_PASSWORD not in rendered, "userinfo reached the record"


async def test_a_percent_encoded_token_does_not_reach_the_record(caplog):
    """httpx2 percent-encodes the URL it reports, so `SEC RET1` arrives as
    `SEC%20RET1`; the URL is redacted whatever its encoding (review)."""
    cfg = {
        "transport": "sse",
        "url": "https://wiki.example.com/mcp?token=SEC RET1&x=s\u00e9cret2",
    }
    encoded = "https://wiki.example.com/mcp?token=SEC%20RET1&x=s%C3%A9cret2"
    rendered = await _render_failure(
        {"wiki": cfg}, lambda _cfg: _UrlBearingError(encoded), caplog, retry=False
    )
    assert "failed to connect" in rendered
    assert "RET1" not in rendered, "the encoded token reached the record"
    assert "cret2" not in rendered, "the encoded non-ASCII token reached the record"


async def test_an_exception_group_is_read_through_to_its_errors(
    secret_url_server, caplog
):
    """streamable_http failures arrive as an ExceptionGroup, whose own text
    names nothing; the status inside is shown, the URL inside redacted."""
    rendered = await _render_failure(
        {"wiki": secret_url_server},
        lambda cfg: ExceptionGroup("unhandled errors", [_UrlBearingError(cfg["url"])]),
        caplog,
        retry=False,
    )
    assert "401 Unauthorized" in rendered
    assert QUERY_SECRET not in rendered
    assert USERINFO_PASSWORD not in rendered


async def test_a_credential_free_endpoint_keeps_its_traceback(caplog):
    """Withholding the traceback is only for URLs that carry credentials."""
    cfg = {"transport": "sse", "url": "https://wiki.example.com/mcp"}
    rendered = await _render_failure(
        {"wiki": cfg}, lambda c: _UrlBearingError(c["url"]), caplog, retry=False
    )
    assert "Traceback" in rendered
    assert "traceback omitted" not in rendered


class _UnreadableError(Exception):
    """An exception whose str() raises."""

    def __str__(self) -> str:
        raise RuntimeError("unreadable")


async def test_an_unreadable_exception_neither_raises_nor_stops_on_start(caplog):
    """Reading the failure's text runs inside the except handler; a failure
    there must not abort on_start and skip the later servers."""
    servers = {
        "wiki": {"transport": "sse", "url": f"https://wiki.example.com/mcp?token={QUERY_SECRET}"},
        "docs": {"transport": "sse", "url": f"https://docs.example.com/mcp?token={QUERY_SECRET}"},
    }
    rendered = await _render_failure(
        servers, lambda _cfg: _UnreadableError(), caplog, retry=False
    )
    assert "'wiki'" in rendered and "'docs'" in rendered
    assert "(message unavailable)" in rendered


# ---------------------------------------------------------------------------
# Config-error and header-env messages
# ---------------------------------------------------------------------------

def test_a_schemeless_endpoint_is_still_shown():
    """`localhost:8080` has no credential position, and is the typo to fix."""
    endpoint = _server_endpoint({"transport": "sse", "url": "localhost:8080"})
    assert "localhost:8080" in endpoint


def test_a_header_name_that_is_a_credential_url_is_not_echoed(monkeypatch):
    """The header-env error names the header without echoing a credential."""
    from corvidae.mcp_client import McpHeaderEnvError, _resolve_headers

    monkeypatch.delenv("NOPE_UNSET_VAR", raising=False)
    header = f"https://u:{USERINFO_PASSWORD}@h.example/?t={QUERY_SECRET}"
    with pytest.raises(McpHeaderEnvError) as info:
        _resolve_headers("wiki", {header: "${NOPE_UNSET_VAR}"})
    message = str(info.value)
    assert "'wiki'" in message and "NOPE_UNSET_VAR" in message
    assert QUERY_SECRET not in message
    assert USERINFO_PASSWORD not in message


async def test_config_error_skip_line_hides_headers_inside_a_list_shaped_entry(caplog):
    """A stray `-` makes the server entry a list; its nested `headers` value
    is still a credential and must not be quoted (review)."""
    headers_secret = "SECRET5"
    entry = [{"transport": "sse", "url": "https://wiki.example.com/sse",
              "headers": {"Authorization": f"Bearer {headers_secret}"}}]
    plugin = await _plugin_with({"wiki": entry})
    with caplog.at_level(logging.WARNING):
        await plugin.on_start({"mcp": {"servers": {"wiki": entry}}})

    joined = "\n".join(
        r.getMessage() for r in caplog.records if "wiki" in r.getMessage()
    )
    assert "mcp.servers.wiki must be a mapping" in joined
    assert headers_secret not in joined


def test_validate_mcp_config_hides_headers_in_a_list_shaped_servers_block():
    """`servers:` written as a list must not quote a nested headers value."""
    from corvidae.mcp_client import validate_mcp_config

    servers = [{"name": "wiki", "transport": "sse",
                "url": "https://wiki.example.com/sse",
                "headers": {"Authorization": "Bearer SECRET5"}}]
    with pytest.raises(ValueError) as info:
        validate_mcp_config({"mcp": {"servers": servers}})
    message = str(info.value)
    assert "mcp.servers must be a mapping" in message
    assert "SECRET5" not in message


