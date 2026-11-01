"""Tests for corvidae.llm_plugin.LLMPlugin.

All tests in this file are RED-phase tests — they verify interfaces
that do not exist yet. They are expected to FAIL against the current
codebase and pass after Part 3 of the agent-decomposition refactor is
implemented.

See plans/agent-decomposition-parts-3-4.md for the full specification.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# 1. LLMPlugin can be imported from corvidae.llm_plugin
# ---------------------------------------------------------------------------


def test_llm_plugin_importable():
    """LLMPlugin class exists and can be imported from corvidae.llm_plugin."""
    from corvidae.llm_plugin import LLMPlugin  # noqa: F401


# ---------------------------------------------------------------------------
# 2. LLMPlugin.depends_on is set() (no dependencies)
# ---------------------------------------------------------------------------


def test_llm_plugin_depends_on_is_empty_set():
    """LLMPlugin.depends_on must be an empty set — it has no upstream deps."""
    from corvidae.llm_plugin import LLMPlugin

    assert hasattr(LLMPlugin, "depends_on"), (
        "LLMPlugin must have a 'depends_on' class attribute"
    )
    assert LLMPlugin.depends_on == set(), (
        f"LLMPlugin.depends_on should be set() but got {LLMPlugin.depends_on!r}"
    )


# ---------------------------------------------------------------------------
# 3. LLMPlugin.__init__ accepts pm, initializes main_client and
#    background_client as None
# ---------------------------------------------------------------------------


def test_llm_plugin_init_stores_pm_and_initializes_clients_as_none():
    """LLMPlugin(pm) must store pm and initialize main/background client as None."""
    from corvidae.llm_plugin import LLMPlugin

    mock_pm = MagicMock()
    plugin = LLMPlugin(pm=mock_pm)

    assert plugin.pm is mock_pm, "LLMPlugin must store pm as self.pm"
    assert plugin.main_client is None, "main_client must be None before on_start"
    assert plugin.background_client is None, "background_client must be None before on_start"


# ---------------------------------------------------------------------------
# 4. LLMPlugin.get_client("main") returns main_client
# ---------------------------------------------------------------------------


def test_llm_plugin_get_client_main_returns_main_client():
    """get_client('main') must return main_client."""
    from corvidae.llm_plugin import LLMPlugin

    plugin = LLMPlugin(pm=None)
    mock_client = MagicMock()
    plugin.main_client = mock_client

    result = plugin.get_client("main")
    assert result is mock_client, (
        f"get_client('main') should return main_client, got {result!r}"
    )


# ---------------------------------------------------------------------------
# 5. LLMPlugin.get_client("background") returns background_client if set,
#    else falls back to main_client
# ---------------------------------------------------------------------------


def test_llm_plugin_get_client_background_returns_background_client_when_set():
    """get_client('background') returns background_client when configured."""
    from corvidae.llm_plugin import LLMPlugin

    plugin = LLMPlugin(pm=None)
    plugin.main_client = MagicMock(name="main")
    plugin.background_client = MagicMock(name="background")

    result = plugin.get_client("background")
    assert result is plugin.background_client, (
        "get_client('background') should return background_client when set"
    )


def test_llm_plugin_get_client_background_falls_back_to_main_when_not_set():
    """get_client('background') falls back to main_client when background is None."""
    from corvidae.llm_plugin import LLMPlugin

    plugin = LLMPlugin(pm=None)
    mock_main = MagicMock(name="main")
    plugin.main_client = mock_main
    plugin.background_client = None  # not configured

    result = plugin.get_client("background")
    assert result is mock_main, (
        "get_client('background') should fall back to main_client when background_client is None"
    )


# ---------------------------------------------------------------------------
# 6. LLMPlugin._create_client returns an LLMClient with correct params
# ---------------------------------------------------------------------------


def test_llm_plugin_create_client_returns_llm_client_with_correct_params():
    """_create_client(cfg, role) must return an LLMClient constructed from cfg."""
    from corvidae.llm_plugin import LLMPlugin
    from corvidae.llm import LLMClient

    cfg = {
        "base_url": "http://localhost:8080",
        "model": "test-model",
        "api_key": "sk-test",
        "extra_body": {"stream": False},
        "max_retries": 5,
        "retry_base_delay": 1.0,
        "retry_max_delay": 30.0,
        "timeout": 60.0,
    }

    client = LLMPlugin()._create_client(cfg, role="main")

    assert isinstance(client, LLMClient), (
        f"_create_client should return an LLMClient, got {type(client)!r}"
    )
    assert client.base_url == cfg["base_url"]
    assert client.model == cfg["model"]


def test_llm_plugin_create_client_uses_defaults_for_optional_params():
    """_create_client(cfg) uses sensible defaults when optional keys are absent."""
    from corvidae.llm_plugin import LLMPlugin
    from corvidae.llm import LLMClient

    cfg = {
        "base_url": "http://localhost:8080",
        "model": "my-model",
    }

    # Must not raise
    client = LLMPlugin()._create_client(cfg, role="main")
    assert isinstance(client, LLMClient)


def test_llm_plugin_create_client_rejects_non_bool_raw_reasoning():
    """raw_reasoning is passed through when boolean; a string is rejected
    naming the key, the value and the accepted values (#355)."""
    from corvidae.llm_plugin import LLMPlugin

    cfg = {"base_url": "http://localhost:8080", "model": "my-model"}
    assert LLMPlugin()._create_client({**cfg, "raw_reasoning": True}).raw_reasoning is True
    with pytest.raises(ValueError) as exc:
        LLMPlugin()._create_client({**cfg, "raw_reasoning": "false"}, role="background")
    assert "llm.background.raw_reasoning" in str(exc.value)
    assert "true or false" in str(exc.value)
    assert "'false'" in str(exc.value)


def test_llm_plugin_create_client_reasoning_reopened_default_and_rerun():
    """R3 (#348): key unset -> truncate; "rerun" is passed through."""
    from corvidae.llm_plugin import LLMPlugin

    # Unset key defaults to truncate; a valid value passes through.
    cfg = {"base_url": "http://localhost:8080", "model": "my-model"}
    assert LLMPlugin()._create_client(cfg).reasoning_reopened == "truncate"
    assert LLMPlugin()._create_client({**cfg, "reasoning_reopened": "rerun"}).reasoning_reopened == "rerun"


def test_llm_plugin_create_client_invalid_reasoning_reopened_warns_and_falls_back(caplog):
    """R7 (#348): an invalid value does not raise; the warning names the key,
    the rejected value and the accepted values; the client falls back to
    truncate."""
    from corvidae.llm_plugin import LLMPlugin

    # An invalid string: no raise, fallback, and a warning naming key, value
    # and accepted values.
    cfg = {"base_url": "http://localhost:8080", "model": "my-model"}
    with caplog.at_level("WARNING", logger="corvidae.llm_plugin"):
        client = LLMPlugin()._create_client({**cfg, "reasoning_reopened": "sometimes"}, role="background")
    assert client.reasoning_reopened == "truncate"
    text = " ".join(r.getMessage() for r in caplog.records if r.name == "corvidae.llm_plugin")
    for needle in ("llm.background.reasoning_reopened", "'sometimes'", "truncate", "rerun"):
        assert needle in text

    # A non-string value also falls back without raising.
    assert LLMPlugin()._create_client({**cfg, "reasoning_reopened": 3}).reasoning_reopened == "truncate"


# ---------------------------------------------------------------------------
# 7. LLMPlugin.on_start creates and starts main_client from config
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_plugin_on_start_creates_and_starts_main_client():
    """on_start must create main_client and call client.start()."""
    from corvidae.llm_plugin import LLMPlugin

    plugin = LLMPlugin()
    config = {
        "llm": {
            "main": {
                "base_url": "http://localhost:8080",
                "model": "test-model",
            }
        }
    }
    await plugin.on_init(pm=MagicMock(), config=config)

    mock_client = MagicMock()
    mock_client.start = AsyncMock()

    with patch.object(LLMPlugin, "_create_client", return_value=mock_client) as mock_create:
        await plugin.on_start(config=config)

    mock_create.assert_called_once_with(config["llm"]["main"], role="main")
    mock_client.start.assert_awaited_once()
    assert plugin.main_client is mock_client


@pytest.mark.asyncio
async def test_llm_plugin_on_start_creates_background_client_when_configured():
    """on_start creates background_client when llm.background is present."""
    from corvidae.llm_plugin import LLMPlugin

    plugin = LLMPlugin(pm=None)

    main_mock = MagicMock(name="main_client")
    main_mock.start = AsyncMock()
    bg_mock = MagicMock(name="background_client")
    bg_mock.start = AsyncMock()

    clients = [main_mock, bg_mock]

    config = {
        "llm": {
            "main": {"base_url": "http://localhost:8080", "model": "main-model"},
            "background": {"base_url": "http://localhost:9090", "model": "bg-model"},
        }
    }
    await plugin.on_init(pm=None, config=config)

    with patch.object(LLMPlugin, "_create_client", side_effect=clients):
        await plugin.on_start(config=config)

    main_mock.start.assert_awaited_once()
    bg_mock.start.assert_awaited_once()
    assert plugin.main_client is main_mock
    assert plugin.background_client is bg_mock


@pytest.mark.asyncio
async def test_llm_plugin_on_start_no_background_client_when_absent():
    """on_start leaves background_client as None when llm.background is absent."""
    from corvidae.llm_plugin import LLMPlugin

    plugin = LLMPlugin(pm=None)

    mock_client = MagicMock()
    mock_client.start = AsyncMock()

    with patch.object(LLMPlugin, "_create_client", return_value=mock_client):
        config = {
            "llm": {
                "main": {"base_url": "http://localhost:8080", "model": "test-model"},
            }
        }
        await plugin.on_start(config=config)

    assert plugin.background_client is None


# ---------------------------------------------------------------------------
# 8. LLMPlugin.on_stop stops both clients
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_plugin_on_stop_stops_main_and_background_clients():
    """on_stop must call stop() on both main and background clients when both exist."""
    from corvidae.llm_plugin import LLMPlugin

    plugin = LLMPlugin(pm=None)
    main_mock = MagicMock()
    main_mock.stop = AsyncMock()
    bg_mock = MagicMock()
    bg_mock.stop = AsyncMock()

    plugin.main_client = main_mock
    plugin.background_client = bg_mock

    await plugin.on_stop()

    main_mock.stop.assert_awaited_once()
    bg_mock.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_llm_plugin_on_stop_stops_only_main_when_no_background():
    """on_stop must stop main_client without error when background_client is None."""
    from corvidae.llm_plugin import LLMPlugin

    plugin = LLMPlugin(pm=None)
    main_mock = MagicMock()
    main_mock.stop = AsyncMock()
    plugin.main_client = main_mock
    plugin.background_client = None

    await plugin.on_stop()  # must not raise

    main_mock.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_llm_plugin_on_stop_safe_when_clients_are_none():
    """on_stop must not raise when no clients were created (e.g., start never called)."""
    from corvidae.llm_plugin import LLMPlugin

    plugin = LLMPlugin(pm=None)
    # Both clients are None — on_stop must complete without error
    await plugin.on_stop()


# ---------------------------------------------------------------------------
# resolve_llm_error — explain transport and provider failures, not just unloaded
# models. Without this every 401/429/connection failure reaches the user as
# the generic "Sorry, I encountered an error" message.
# ---------------------------------------------------------------------------


def _http_error(status: int, body: str = "") -> "object":
    """Build an LLMHTTPError with a synthetic ClientResponseError cause."""
    import aiohttp

    from corvidae.llm import LLMHTTPError

    cause = aiohttp.ClientResponseError(
        request_info=MagicMock(),
        history=(),
        status=status,
        message="error",
        headers={},
    )
    return LLMHTTPError(cause, body=body, model="test-model")


class TestResolveLlmErrorExplainsFailures:
    async def test_401_names_the_api_key(self):
        """A 401 must point at llm.main.api_key, not just say 'an error'."""
        from corvidae.llm_plugin import LLMPlugin

        plugin = LLMPlugin(pm=None)
        message = await plugin.resolve_llm_error(
            channel=None, error=_http_error(401, '{"error":"bad key"}')
        )

        assert message is not None
        assert "401" in message
        assert "api_key" in message.lower()

    async def test_429_mentions_rate_limit_or_quota(self):
        """A 429 must name rate limiting/quota and include the provider body."""
        from corvidae.llm_plugin import LLMPlugin

        plugin = LLMPlugin(pm=None)
        body = '{"error":{"message":"Insufficient balance"}}'
        message = await plugin.resolve_llm_error(channel=None, error=_http_error(429, body))

        assert message is not None
        assert "429" in message
        assert "Insufficient balance" in message

    async def test_unmapped_status_still_names_the_status(self):
        """Any other HTTP failure must at least name the status code."""
        from corvidae.llm_plugin import LLMPlugin

        plugin = LLMPlugin(pm=None)
        message = await plugin.resolve_llm_error(channel=None, error=_http_error(502, "bad gateway"))

        assert message is not None
        assert "502" in message

    async def test_connection_error_points_at_base_url(self):
        """A refused connection must point at llm.main.base_url."""
        import aiohttp

        from corvidae.llm_plugin import LLMPlugin

        plugin = LLMPlugin(pm=None)
        message = await plugin.resolve_llm_error(
            channel=None, error=aiohttp.ClientConnectionError("Connection refused")
        )

        assert message is not None
        assert "base_url" in message.lower()

    async def test_timeout_is_explained(self):
        """An asyncio timeout must be explained as a timeout."""
        from corvidae.llm_plugin import LLMPlugin

        plugin = LLMPlugin(pm=None)
        message = await plugin.resolve_llm_error(channel=None, error=asyncio.TimeoutError())

        assert message is not None
        assert "timed out" in message.lower() or "timeout" in message.lower()

    async def test_unrelated_error_defers_to_the_generic_message(self):
        """Non-LLM errors must still return None so the generic text applies."""
        from corvidae.llm_plugin import LLMPlugin

        plugin = LLMPlugin(pm=None)
        message = await plugin.resolve_llm_error(
            channel=None, error=RuntimeError("something else")
        )

        assert message is None


# ---------------------------------------------------------------------------
# api_key from the environment via ${VAR} (#217)
# ---------------------------------------------------------------------------

_KEY_VAR = "CORVIDAE_TEST_KEY"
_SECRET = "sk-secret-value-217"


def _role_cfg(api_key: str | None) -> dict:
    """Build a minimal role config carrying the given api_key."""
    cfg = {
        "base_url": "http://localhost:9",
        "model": "m",
        "dimensions": 8,
    }
    if api_key is not None:
        cfg["api_key"] = api_key
    return cfg


async def _start_plugin(llm: dict):
    """Run on_start for an LLMPlugin with the given llm: config."""
    from corvidae.llm_plugin import LLMPlugin

    plugin = LLMPlugin()
    plugin._role_configs = dict(llm)
    await plugin.on_start(config={"llm": llm})
    return plugin


class TestApiKeyFromEnvironment:
    @pytest.mark.parametrize("role", ["main", "background", "embedding"])
    async def test_env_ref_resolves_to_bearer_token(self, role, monkeypatch):
        """K1: ${VAR} in api_key is sent as the Bearer token, for every role."""
        monkeypatch.setenv(_KEY_VAR, _SECRET)
        llm = {"main": _role_cfg("sk-main")}
        llm[role] = _role_cfg("${" + _KEY_VAR + "}")
        plugin = await _start_plugin(llm)
        try:
            client = plugin.get_client(role)
            assert client.api_key == _SECRET
            assert client.session.headers["Authorization"] == f"Bearer {_SECRET}"
            # The raw config keeps the reference, never the resolved key.
            assert plugin._role_configs[role]["api_key"] == "${" + _KEY_VAR + "}"
        finally:
            await plugin.on_stop()

    @pytest.mark.parametrize("value", [None, "", "k\n"], ids=["unset", "empty", "newline"])
    async def test_bad_variable_fails_startup_with_clear_message(
        self, value, monkeypatch, caplog
    ):
        """K2/K3: unset, empty or multi-line variable is a startup error."""
        if value is None:
            monkeypatch.delenv(_KEY_VAR, raising=False)
        else:
            monkeypatch.setenv(_KEY_VAR, value)
        llm = {
            "main": _role_cfg("sk-main"),
            "background": _role_cfg("${" + _KEY_VAR + "}"),
        }
        from corvidae.llm_plugin import LLMPlugin

        # Build the plugin here (not via _start_plugin) so teardown can stop
        # the main client that on_start started before background failed.
        plugin = LLMPlugin()
        plugin._role_configs = dict(llm)
        try:
            with caplog.at_level("DEBUG"):
                with pytest.raises(ValueError) as excinfo:
                    await plugin.on_start(config={"llm": llm})
        finally:
            await plugin.on_stop()
        msg = str(excinfo.value)
        assert "llm.background.api_key" in msg
        assert _KEY_VAR in msg
        if value == "k\n":
            assert "newline" in msg
        else:
            assert "not set in the daemon's environment" in msg and "restart" in msg
        # K3: the value never reaches the error or the logs.
        if value:
            assert value not in msg
            for rec in caplog.records:
                assert value not in rec.getMessage()
                assert value not in (rec.exc_text or "")

    async def test_secret_with_newline_absent_from_error_and_logs(
        self, monkeypatch, caplog
    ):
        """K3: a multi-line secret is not echoed anywhere."""
        secret = "topsecret217\nsecond"
        monkeypatch.setenv(_KEY_VAR, secret)
        llm = {"main": _role_cfg("${" + _KEY_VAR + "}")}
        with caplog.at_level("DEBUG"):
            with pytest.raises(ValueError) as excinfo:
                await _start_plugin(llm)
        assert "topsecret217" not in str(excinfo.value)
        for rec in caplog.records:
            assert "topsecret217" not in rec.getMessage()
            assert "topsecret217" not in (rec.exc_text or "")

    async def test_401_reply_does_not_contain_key(self, monkeypatch, caplog):
        """K3: the 401 reply names the key but never its value."""
        from corvidae.llm_plugin import LLMPlugin

        monkeypatch.setenv(_KEY_VAR, _SECRET)
        plugin = await _start_plugin({"main": _role_cfg("${" + _KEY_VAR + "}")})
        try:
            with caplog.at_level("DEBUG"):
                reply = await plugin.resolve_llm_error(
                    channel=None, error=_http_error(401, '{"error":"bad key"}')
                )
            assert reply is not None and "api_key" in reply
            assert _SECRET not in reply
            for rec in caplog.records:
                assert _SECRET not in rec.getMessage()
                assert _SECRET not in (rec.exc_text or "")
        finally:
            await plugin.on_stop()

    async def test_literal_key_still_works(self):
        """K4: a literal api_key behaves as before."""
        plugin = await _start_plugin({"main": _role_cfg("sk-literal")})
        try:
            assert plugin.get_client("main").session.headers[
                "Authorization"
            ] == "Bearer sk-literal"
        finally:
            await plugin.on_stop()

    def test_literal_with_dollar_but_no_ref_is_left_alone(self):
        """K4: '$' without a ${NAME} is not an env reference."""
        from corvidae.llm_plugin import LLMPlugin

        client = LLMPlugin()._create_client(_role_cfg("pa$$word$"), role="main")
        assert client.api_key == "pa$$word$"

    async def test_reload_to_unset_variable_keeps_old_client(self, monkeypatch):
        """Reload: an unset variable raises and the old main client stays."""
        monkeypatch.delenv(_KEY_VAR, raising=False)
        plugin = await _start_plugin({"main": _role_cfg("sk-old")})
        try:
            old = plugin.get_client("main")
            with pytest.raises(ValueError) as excinfo:
                await plugin.on_config_reload(
                    config={"llm": {"main": _role_cfg("${" + _KEY_VAR + "}")}}
                )
            assert "llm.main.api_key" in str(excinfo.value)
            assert plugin.get_client("main") is old
        finally:
            await plugin.on_stop()


# ---------------------------------------------------------------------------
# #238: a role per model-calling stage, fallback chains in LLMPlugin
# ---------------------------------------------------------------------------


def test_fallback_table_pinned():
    """R2: the chain table is the single source of fallback truth — pinned
    verbatim so a change is a deliberate, visible edit."""
    from corvidae.llm_plugin import LLMPlugin

    assert LLMPlugin.FALLBACK_CHAINS == {
        "subagent": ("subagent", "background", "main"),
        "consolidation": ("consolidation", "background", "main"),
        "rubric": ("rubric", "background", "main"),
        "critic": ("critic", "background", "main"),
        "appraisal": ("appraisal", "background", "main"),
        "compaction": ("compaction", "background", "main"),
    }


@pytest.mark.parametrize(
    "role,configured,expected",
    [
        # its own role wins when configured
        ("subagent", {"main": 1, "background": 2, "subagent": 3}, "subagent"),
        ("consolidation", {"main": 1, "background": 2, "consolidation": 3}, "consolidation"),
        ("rubric", {"main": 1, "background": 2, "rubric": 3}, "rubric"),
        ("critic", {"main": 1, "background": 2, "critic": 3}, "critic"),
        ("appraisal", {"main": 1, "background": 2, "appraisal": 3}, "appraisal"),
        ("compaction", {"main": 1, "compaction": 3}, "compaction"),
        # chain falls through to background
        ("subagent", {"main": 1, "background": 2}, "background"),
        ("consolidation", {"main": 1, "background": 2}, "background"),
        ("rubric", {"main": 1, "background": 2}, "background"),
        ("critic", {"main": 1, "background": 2}, "background"),
        ("appraisal", {"main": 1, "background": 2}, "background"),
        # compaction falls through to background, like the other stages
        ("compaction", {"main": 1, "background": 2}, "background"),
        # unknown role: straight to main
        ("something-new", {"main": 1, "background": 2}, "main"),
        # nothing but main
        ("critic", {"main": 1}, "main"),
    ],
)
def test_get_client_walks_chain(role, configured, expected):
    """R2: get_client walks the declared chain to the first configured role."""
    from corvidae.llm_plugin import LLMPlugin

    plugin = LLMPlugin(pm=None)
    clients = {r: MagicMock(name=r) for r, _ in configured.items()}
    for r, c in clients.items():
        plugin._clients[r] = c
    assert plugin.get_client(role) is clients[expected]


@pytest.mark.parametrize(
    "role,expected",
    [
        # R4: a config with only main/background/embedding routes every
        # stage to the same client it used before #238.
        ("subagent", "background"),       # was get_client("background")
        ("consolidation", "background"),  # was get_client("background")
        ("rubric", "background"),         # was get_client("background")
        ("compaction", "background"),     # was llm.main_client; main's view path runs on background
        ("critic", "background"),         # was the manual critic->bg->main loop
        ("appraisal", "background"),      # was the manual appraisal->bg->main loop
        ("main", "main"),
        ("embedding", "embedding"),
    ],
)
def test_legacy_config_routes_every_stage_as_before(role, expected):
    from corvidae.llm_plugin import LLMPlugin

    plugin = LLMPlugin(pm=None)
    for r in ("main", "background", "embedding"):
        plugin._clients[r] = MagicMock(name=r)
    assert plugin.get_client(role) is plugin._clients[expected]


async def test_on_start_logs_resolved_role_map(caplog):
    """R5: one INFO line at startup shows where each stage's traffic goes."""
    import logging

    from corvidae.llm_plugin import LLMPlugin

    plugin = LLMPlugin(pm=None)

    class FakeClient:
        def __init__(self, role):
            self.role = role

        async def start(self):
            pass

        async def stop(self):
            pass

    with patch.object(
        plugin, "_create_client", lambda cfg, role="main": FakeClient(role)
    ):
        with caplog.at_level(logging.INFO):
            await plugin.on_start(
                {"llm": {"main": {"model": "m"}, "background": {"model": "b"}}}
            )
    line = next(
        (r.message for r in caplog.records
         if r.message.startswith("llm roles resolved:")),
        None,
    )
    assert line is not None, "no 'llm roles resolved:' INFO line at startup"
    assert "consolidation→background" in line
    assert "subagent→background" in line
    assert "compaction→background" in line
    assert "main→main" in line


def test_no_client_reads_llm_private_dict_outside_llm_plugin():
    """R3: no consumer walks LLMPlugin._clients — the grep is the test.

    web.py keeps its own unrelated _clients fan-out set; the patterns
    below target the llm plugin's dict specifically.
    """
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parent.parent
    pattern = re.compile(r'getattr\(\s*llm\s*,\s*"_clients"|llm\._clients')
    offenders = []
    for path in list(root.glob("corvidae/**/*.py")) + list(root.glob("packages/**/*.py")):
        if "test" in path.parts:
            continue
        if path.name == "llm_plugin.py":
            continue
        if pattern.search(path.read_text()):
            offenders.append(str(path))
    assert not offenders, f"reads of LLMPlugin._clients outside llm_plugin.py: {offenders}"


def _fake_ep(name, dist_name):
    import types

    return types.SimpleNamespace(name=name, dist=types.SimpleNamespace(name=dist_name))


def test_entry_point_duplicates_fail_loudly(monkeypatch):
    """R6: two distributions claiming one entry-point name is a Config
    error naming the point, both distributions, and the remedy."""
    from corvidae import runtime

    import importlib.metadata

    monkeypatch.setattr(
        importlib.metadata, "entry_points", lambda group="corvidae": [
            _fake_ep("dup", "dist-a"),
            _fake_ep("dup", "dist-b"),
            _fake_ep("solo", "dist-a"),
        ]
    )
    with pytest.raises(ValueError) as exc:
        runtime.check_entry_point_duplicates("corvidae")
    message = str(exc.value)
    assert "dup" in message
    assert "dist-a" in message and "dist-b" in message
    assert "uninstall one" in message


def test_entry_point_no_duplicates_passes(monkeypatch):
    from corvidae import runtime

    import importlib.metadata

    monkeypatch.setattr(
        importlib.metadata, "entry_points", lambda group="corvidae": [
            _fake_ep("one", "dist-a"),
            _fake_ep("two", "dist-b"),
        ]
    )
    runtime.check_entry_point_duplicates("corvidae")  # no raise
