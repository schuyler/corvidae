"""RED tests for corvidae.shutdown and its wiring into Runtime (#746).

Covers R6 (daemon.exit_timeout) and R6b (daemon.shutdown_timeout) parsing,
the startup Config error for a bad value, and Runtime.stop() using the
configured graceful bound.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Values rejected for exit_timeout: (id, value).
BAD_EXIT = [
    ("negative", -1),
    ("bool", True),
    ("string", "soon"),
    ("inf", float("inf")),
    ("nan", float("nan")),
]
# shutdown_timeout rejects the same plus 0 and null (positive only).
BAD_GRACEFUL = BAD_EXIT + [("zero", 0), ("null", None)]


def _ids(cases: list[tuple[str, object]]) -> list[str]:
    """pytest ids for a (id, value) case list."""
    return [name for name, _ in cases]


def test_defaults_when_keys_absent():
    """R6/R6b: absent keys give graceful 3 s and exit 5 s."""
    from corvidae.shutdown import parse_shutdown_timeouts

    parsed = parse_shutdown_timeouts({})
    assert parsed.graceful == 3.0
    assert parsed.exit == 5.0


@pytest.mark.parametrize("value", [0, None], ids=["zero", "null"])
def test_exit_timeout_zero_or_null_disables(value):
    """R6: 0 and null mean no watchdog (catches an `or 5` default)."""
    from corvidae.shutdown import parse_shutdown_timeouts

    assert parse_shutdown_timeouts({"exit_timeout": value}).exit is None


def test_arm_exit_watchdog_none_starts_no_thread():
    """R6: a disabled watchdog (None) arms nothing, so it can never fire.

    Catches an arming side that falls back to the 5 s default for None.
    """
    import threading

    from corvidae.shutdown import arm_exit_watchdog

    before = set(threading.enumerate())
    # Patched so a wrong implementation's thread can't kill the pytest run.
    with patch("os._exit"):
        arm_exit_watchdog(None)
    # Any thread started by the call would show up here.
    assert set(threading.enumerate()) - before == set()


@pytest.mark.parametrize(
    "value", [v for _, v in BAD_EXIT], ids=_ids(BAD_EXIT)
)
def test_bad_exit_timeout_message_names_key_value_and_rule(value):
    """R6: the error names daemon.exit_timeout, the value and the accepted values."""
    from corvidae.shutdown import parse_shutdown_timeouts

    with pytest.raises(ValueError) as excinfo:
        parse_shutdown_timeouts({"exit_timeout": value})
    message = str(excinfo.value)
    assert "daemon.exit_timeout" in message
    assert repr(value) in message
    # Accepted values: a number of seconds >= 0, with 0/null disabling.
    assert ">= 0" in message and "null" in message


@pytest.mark.parametrize(
    "value", [v for _, v in BAD_GRACEFUL], ids=_ids(BAD_GRACEFUL)
)
def test_bad_shutdown_timeout_message_names_key_value_and_rule(value):
    """R6b: the error names daemon.shutdown_timeout, the value and 'seconds > 0'."""
    from corvidae.shutdown import parse_shutdown_timeouts

    with pytest.raises(ValueError) as excinfo:
        parse_shutdown_timeouts({"shutdown_timeout": value})
    message = str(excinfo.value)
    assert "daemon.shutdown_timeout" in message
    assert repr(value) in message
    assert "seconds > 0" in message


def test_non_mapping_daemon_message_names_section():
    """R6: a non-mapping daemon section is rejected, naming `daemon` and the value."""
    from corvidae.shutdown import parse_shutdown_timeouts

    with pytest.raises(ValueError) as excinfo:
        parse_shutdown_timeouts(5)
    message = str(excinfo.value)
    assert "daemon" in message
    assert "5" in message


@pytest.mark.parametrize(
    "key,value",
    [("exit_timeout", -1), ("shutdown_timeout", 0)],
    ids=["exit_timeout", "shutdown_timeout"],
)
async def test_startup_with_bad_value_prints_config_error_and_exits(
    tmp_path, capsys, key, value
):
    """R6/R6b: Runtime startup prints `Config error` naming key and path, exits 1."""
    from corvidae.runtime import Runtime

    config_path = tmp_path / "agent.yaml"
    config_path.write_text(
        "llm:\n  main: {base_url: http://x, model: m}\n"
        f"daemon:\n  {key}: {value}\n"
    )
    # Fully mocked pm, so that without the validation start() runs to the
    # end and the failure is the missing SystemExit, not a mock misuse.
    mock_pm = MagicMock()
    mock_pm.ahook.on_init = AsyncMock(return_value=[])
    mock_pm.ahook.on_start = AsyncMock(return_value=[])
    mock_agent = MagicMock()
    mock_agent.on_start = AsyncMock()
    mock_pm.get_plugin.return_value = mock_agent
    with patch(
        "corvidae.runtime.create_plugin_manager", return_value=mock_pm
    ), patch("corvidae.runtime.validate_dependencies"), patch(
        "corvidae.runtime.configure_logging"
    ):
        rt = Runtime(config_path=str(config_path))
        with pytest.raises(SystemExit) as excinfo:
            await rt.start()

    assert excinfo.value.code == 1
    stderr = capsys.readouterr().err
    assert "Config error:" in stderr
    assert f"daemon.{key}" in stderr
    assert repr(value) in stderr
    assert f"check {config_path}" in stderr


async def test_stop_uses_configured_graceful_bound(tmp_path, caplog):
    """R6b wiring: a configured small shutdown_timeout bounds stop().

    on_stop sleeps 1 s, longer than the configured 0.2 s but shorter than
    the old hard-coded 3 s, so only a stop() reading the key forces the
    (patched) os._exit(1); its warning names daemon.shutdown_timeout.
    """
    from corvidae.runtime import Runtime

    config_path = tmp_path / "agent.yaml"
    config_path.write_text(
        "llm:\n  main: {base_url: http://x, model: m}\n"
        "daemon:\n  shutdown_timeout: 0.2\n"
    )

    async def slow_on_stop() -> list:
        # Longer than the configured bound, shorter than the old 3 s.
        await asyncio.sleep(1.0)
        return []

    mock_pm = MagicMock()
    mock_pm.ahook.on_init = AsyncMock(return_value=[])
    mock_pm.ahook.on_start = AsyncMock(return_value=[])
    mock_pm.ahook.on_stop = AsyncMock(side_effect=slow_on_stop)
    mock_agent = MagicMock()
    mock_agent.on_start = AsyncMock()
    mock_agent.on_stop = AsyncMock()
    mock_pm.get_plugin.return_value = mock_agent

    with patch(
        "corvidae.runtime.create_plugin_manager", return_value=mock_pm
    ), patch("corvidae.runtime.validate_dependencies"), patch(
        "corvidae.runtime.configure_logging"
    ), patch("os._exit") as mock_os_exit, caplog.at_level(logging.WARNING):
        rt = Runtime(config_path=str(config_path))
        await rt.start()
        await rt.stop()

    mock_os_exit.assert_called_once_with(1)
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("daemon.shutdown_timeout" in m for m in warnings), warnings
