"""Tests for corvidae.llm_plugin.LLMPlugin.

All tests in this file are RED-phase tests — they verify interfaces
that do not exist yet. They are expected to FAIL against the current
codebase and pass after Part 3 of the agent-decomposition refactor is
implemented.

See plans/agent-decomposition-parts-3-4.md for the full specification.
"""

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
    """_create_client(cfg) must return an LLMClient constructed from cfg."""
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

    client = LLMPlugin._create_client(cfg)

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
    client = LLMPlugin._create_client(cfg)
    assert isinstance(client, LLMClient)


# ---------------------------------------------------------------------------
# 7. LLMPlugin.on_start creates and starts main_client from config
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_plugin_on_start_creates_and_starts_main_client():
    """on_start must create main_client and call client.start()."""
    from corvidae.llm_plugin import LLMPlugin

    plugin = LLMPlugin(pm=None)

    mock_client = MagicMock()
    mock_client.start = AsyncMock()

    with patch.object(LLMPlugin, "_create_client", return_value=mock_client) as mock_create:
        config = {
            "llm": {
                "main": {
                    "base_url": "http://localhost:8080",
                    "model": "test-model",
                }
            }
        }
        await plugin.on_start(config=config)

    mock_create.assert_called_once_with(config["llm"]["main"])
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

    with patch.object(LLMPlugin, "_create_client", side_effect=clients):
        config = {
            "llm": {
                "main": {"base_url": "http://localhost:8080", "model": "main-model"},
                "background": {"base_url": "http://localhost:9090", "model": "bg-model"},
            }
        }
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
