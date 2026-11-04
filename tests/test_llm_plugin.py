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

from helpers import stub_llm_config


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
# 3. LLMPlugin.__init__ accepts pm and starts with no clients
# ---------------------------------------------------------------------------


def test_llm_plugin_init_stores_pm_and_initializes_clients_as_none():
    """LLMPlugin(pm) must store pm and hold no clients before on_start."""
    from corvidae.llm_plugin import LLMPlugin

    mock_pm = MagicMock()
    plugin = LLMPlugin(pm=mock_pm)

    assert plugin.pm is mock_pm, "LLMPlugin must store pm as self.pm"
    assert plugin.get_client() is None, "the main-turn client must be None before on_start"
    assert plugin._clients == {}, "no client may exist before on_start"


# ---------------------------------------------------------------------------
# 4. LLMPlugin.get_client() with no stage returns the default chat client (main turns)
# ---------------------------------------------------------------------------


def test_llm_plugin_get_client_without_stage_returns_main_client():
    """R2: get_client() is the main-turn client."""
    from corvidae.llm_plugin import LLMPlugin

    plugin = LLMPlugin(pm=None)
    mock_client = MagicMock()
    plugin._clients["main"] = mock_client
    stub_llm_config(plugin)

    result = plugin.get_client()
    assert result is mock_client, (
        f"get_client() should return the main profile's client, got {result!r}"
    )


# ---------------------------------------------------------------------------
# 6. LLMPlugin._create_client returns an LLMClient with correct params
# ---------------------------------------------------------------------------


def test_llm_plugin_create_client_returns_llm_client_with_correct_params():
    """_create_client(cfg, profile) must return an LLMClient constructed from cfg."""
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

    client = LLMPlugin()._create_client(cfg, profile="main")

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
    client = LLMPlugin()._create_client(cfg, profile="main")
    assert isinstance(client, LLMClient)


def test_llm_plugin_create_client_rejects_non_bool_raw_reasoning():
    """raw_reasoning is passed through when boolean; a string is rejected
    naming the key, the value and the accepted values (#355)."""
    from corvidae.llm_plugin import LLMPlugin

    cfg = {"base_url": "http://localhost:8080", "model": "my-model"}
    assert LLMPlugin()._create_client({**cfg, "raw_reasoning": True}).raw_reasoning is True
    with pytest.raises(ValueError) as exc:
        LLMPlugin()._create_client({**cfg, "raw_reasoning": "false"}, profile="background")
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
        client = LLMPlugin()._create_client({**cfg, "reasoning_reopened": "sometimes"}, profile="background")
    assert client.reasoning_reopened == "truncate"
    text = " ".join(r.getMessage() for r in caplog.records if r.name == "corvidae.llm_plugin")
    for needle in ("llm.background.reasoning_reopened", "'sometimes'", "truncate", "rerun"):
        assert needle in text

    # A non-string value also falls back without raising.
    assert LLMPlugin()._create_client({**cfg, "reasoning_reopened": 3}).reasoning_reopened == "truncate"


# ---------------------------------------------------------------------------
# 7. LLMPlugin.on_start creates and starts the main client from config
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_plugin_on_start_creates_and_starts_main_client():
    """on_start must create the main profile's client and call client.start()."""
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

    # The client is built from the resolved profile, which carries its kind.
    mock_create.assert_called_once_with(
        {**config["llm"]["main"], "kind": "chat"}, profile="main"
    )
    mock_client.start.assert_awaited_once()
    assert plugin._clients["main"] is mock_client


# ---------------------------------------------------------------------------
# 8. LLMPlugin.on_stop stops every client
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_plugin_on_stop_stops_every_client():
    """on_stop must call stop() on the client of every profile."""
    from corvidae.llm_plugin import LLMPlugin

    plugin = LLMPlugin(pm=None)
    main_mock = MagicMock()
    main_mock.stop = AsyncMock()
    other_mock = MagicMock()
    other_mock.stop = AsyncMock()

    plugin._clients["main"] = main_mock
    plugin._clients["other"] = other_mock

    await plugin.on_stop()

    main_mock.stop.assert_awaited_once()
    other_mock.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_llm_plugin_on_stop_safe_when_clients_are_none():
    """on_stop must not raise when no clients were created (e.g., start never called)."""
    from corvidae.llm_plugin import LLMPlugin

    plugin = LLMPlugin(pm=None)
    # No clients exist — on_stop must complete without error
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


def _profile_cfg(api_key: str | None) -> dict:
    """Build a minimal profile config carrying the given api_key."""
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
    await plugin.on_start(config={"llm": llm})
    return plugin


class TestApiKeyFromEnvironment:
    @pytest.mark.parametrize("profile", ["main", "background", "embedding"])
    async def test_env_ref_resolves_to_bearer_token(self, profile, monkeypatch):
        """K1: ${VAR} in api_key is sent as the Bearer token, for every profile."""
        monkeypatch.setenv(_KEY_VAR, _SECRET)
        llm = {"main": _profile_cfg("sk-main")}
        llm[profile] = _profile_cfg("${" + _KEY_VAR + "}")
        if profile == "embedding":
            llm[profile]["kind"] = "embedding"
        plugin = await _start_plugin(llm)
        try:
            client = plugin._clients[profile]
            assert client.api_key == _SECRET
            assert client.session.headers["Authorization"] == f"Bearer {_SECRET}"
            # The raw config keeps the reference, never the resolved key.
            assert plugin._config.profiles[profile]["api_key"] == "${" + _KEY_VAR + "}"
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
            "main": _profile_cfg("sk-main"),
            "background": _profile_cfg("${" + _KEY_VAR + "}"),
        }
        from corvidae.llm_plugin import LLMPlugin

        # Build the plugin here (not via _start_plugin) so teardown can stop
        # the main client that on_start started before background failed.
        plugin = LLMPlugin()
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
        llm = {"main": _profile_cfg("${" + _KEY_VAR + "}")}
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
        plugin = await _start_plugin({"main": _profile_cfg("${" + _KEY_VAR + "}")})
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
        plugin = await _start_plugin({"main": _profile_cfg("sk-literal")})
        try:
            assert plugin.get_client().session.headers[
                "Authorization"
            ] == "Bearer sk-literal"
        finally:
            await plugin.on_stop()

    def test_literal_with_dollar_but_no_ref_is_left_alone(self):
        """K4: '$' without a ${NAME} is not an env reference."""
        from corvidae.llm_plugin import LLMPlugin

        client = LLMPlugin()._create_client(_profile_cfg("pa$$word$"), profile="main")
        assert client.api_key == "pa$$word$"

    async def test_reload_to_unset_variable_warns_and_keeps_old_client(
        self, monkeypatch, caplog
    ):
        """Reload: llm is restart-only, so a changed api_key (here an unset
        variable) warns, is not resolved, and the old client stays."""
        monkeypatch.delenv(_KEY_VAR, raising=False)
        plugin = await _start_plugin({"main": _profile_cfg("sk-old")})
        try:
            old = plugin.get_client()
            with caplog.at_level("WARNING", logger="corvidae.llm_plugin"):
                await plugin.on_config_reload(
                    config={"llm": {"main": _profile_cfg("${" + _KEY_VAR + "}")}}
                )
            assert "apply on restart" in caplog.text
            assert plugin.get_client() is old
        finally:
            await plugin.on_stop()


# ---------------------------------------------------------------------------
# #431: a name per model-calling stage; a stage resolves to its kind's
# default profile; stages are registered by their owning plugins
# ---------------------------------------------------------------------------

_ALL_STAGES = {
    "compaction": "chat",
    "consolidation": "chat",
    "embedding": "embedding",
    "rubric": "chat",
    "subagent": "chat",
}


def _plugin_with_profiles(stages: dict, *profiles: str):
    """A bare LLMPlugin holding ``stages`` and one stub client per profile."""
    from corvidae.llm_plugin import LLMPlugin

    plugin = LLMPlugin(pm=None)
    plugin._stages = dict(stages)
    llm_section = {}
    for profile in profiles:
        plugin._clients[profile] = MagicMock(name=profile)
        llm_section[profile] = {"base_url": f"http://{profile}.test", "model": profile}
    if "embedding" in llm_section:
        llm_section["embedding"].update(kind="embedding", dimensions=4)
    stub_llm_config(plugin, llm_section)
    return plugin


@pytest.mark.parametrize("stage", ["compaction", "consolidation", "rubric", "subagent"])
def test_chat_stage_resolves_to_main_with_background_configured(stage):
    """R2: a chat stage gets the main client; nothing resolves through
    background."""
    plugin = _plugin_with_profiles(_ALL_STAGES, "main", "background", "embedding")
    assert plugin.get_client(stage) is plugin._clients["main"]


def test_embedding_stage_resolves_to_embedding_client():
    """R2: the embedding stage gets the embedding client."""
    plugin = _plugin_with_profiles(_ALL_STAGES, "main", "background", "embedding")
    assert plugin.get_client("embedding") is plugin._clients["embedding"]


def test_embedding_stage_without_embedding_profile_is_none():
    """R2: an unconfigured default is None; the embedding stage does not fall
    back to main."""
    plugin = _plugin_with_profiles(_ALL_STAGES, "main", "background")
    assert plugin.get_client("embedding") is None


def test_fallback_chain_surface_is_gone():
    """R2: FALLBACK_CHAINS, fallback_chain and resolve_role are removed."""
    from corvidae.llm_plugin import LLMPlugin

    for name in ("FALLBACK_CHAINS", "fallback_chain", "resolve_role"):
        assert not hasattr(LLMPlugin, name), f"LLMPlugin.{name} must be removed"


@pytest.mark.parametrize("stage", ["unregistered", "main"])
def test_get_client_rejects_a_name_that_is_not_a_registered_stage(stage):
    """R3: an unregistered stage fails loudly, and a profile name is not a
    stage (callers wanting main use get_client())."""
    plugin = _plugin_with_profiles(_ALL_STAGES, "main", "background")
    with pytest.raises(ValueError) as excinfo:
        plugin.get_client(stage)
    message = str(excinfo.value)
    assert stage in message
    assert "compaction" in message, "the error should name the registered stages"


async def test_on_start_collects_stages_through_register_llm_stages():
    """R3: LLMPlugin.on_start collects stages from every plugin's
    register_llm_stages hookimpl, before any client starts."""
    from corvidae.hooks import create_plugin_manager, hookimpl
    from corvidae.llm_plugin import LLMPlugin

    class _Owner:
        @hookimpl
        def register_llm_stages(self, stages):
            stages["owned"] = "chat"

    class _OtherOwner:
        @hookimpl
        def register_llm_stages(self, stages):
            stages["vectors"] = "embedding"

    class _BlockedClient:
        async def start(self):
            raise RuntimeError("client start blocked")

    pm = create_plugin_manager()
    pm.register(_Owner(), name="owner")
    pm.register(_OtherOwner(), name="other_owner")
    plugin = LLMPlugin()
    pm.register(plugin, name="llm")
    config = {"llm": {"main": {"base_url": "http://m", "model": "m"}}}
    await plugin.on_init(pm, config)

    with patch.object(plugin, "_create_client", lambda cfg, **kw: _BlockedClient()):
        with pytest.raises(RuntimeError, match="blocked"):
            await plugin.on_start(config)

    assert plugin._stages == {"owned": "chat", "vectors": "embedding"}


_STAGE_LOG_PREFIX = "llm stages resolved:"


@pytest.mark.parametrize(
    "llm_config,expected",
    [
        (
            {"main": {"base_url": "http://m", "model": "m"},
             "background": {"base_url": "http://b", "model": "b"},
             "embedding": {"base_url": "http://e", "model": "e", "kind": "embedding",
                           "dimensions": 4}},
            "llm stages resolved: compaction→main, consolidation→main, "
            "embedding→embedding, rubric→main, subagent→main",
        ),
        (
            {"main": {"base_url": "http://m", "model": "m"},
             "background": {"base_url": "http://b", "model": "b"}},
            "llm stages resolved: compaction→main, consolidation→main, "
            "embedding→(none), rubric→main, subagent→main",
        ),
    ],
    ids=["embedding_configured", "embedding_unconfigured"],
)
async def test_on_start_logs_each_stage_resolution(caplog, llm_config, expected):
    """R5: one INFO line at startup lists each registered stage's profile,
    sorted by stage name, with (none) for an unconfigured default."""
    import logging

    from corvidae.hooks import create_plugin_manager, hookimpl
    from corvidae.llm_plugin import LLMPlugin

    class _Owner:
        @hookimpl
        def register_llm_stages(self, stages):
            # Deliberately unsorted: the line sorts them.
            for name in ("subagent", "rubric", "embedding", "consolidation", "compaction"):
                stages[name] = _ALL_STAGES[name]

    class FakeClient:
        async def start(self):
            pass

        async def stop(self):
            pass

    pm = create_plugin_manager()
    pm.register(_Owner(), name="owner")
    plugin = LLMPlugin()
    pm.register(plugin, name="llm")
    config = {"llm": llm_config}
    await plugin.on_init(pm, config)

    with patch.object(plugin, "_create_client", lambda cfg, **kw: FakeClient()):
        with caplog.at_level(logging.INFO, logger="corvidae.llm_plugin"):
            await plugin.on_start(config)

    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith(_STAGE_LOG_PREFIX)]
    assert lines == [expected]


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
