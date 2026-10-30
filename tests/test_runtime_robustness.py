"""Runtime startup robustness: null plugin config (#176), cleanup after a
failed start (#177), and an honest "plugins loaded" line (#178)."""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from corvidae.runtime import Runtime

_LLM = {"main": {"base_url": "http://localhost:8080", "model": "m"}}


def _mock_pm():
    """A mock plugin manager whose hooks succeed and whose agent is a mock."""
    pm = MagicMock()
    pm.ahook.on_init = AsyncMock(return_value=[])
    pm.ahook.on_start = AsyncMock(return_value=[])
    pm.ahook.on_stop = AsyncMock(return_value=[])
    pm.list_name_plugin.return_value = []
    agent = MagicMock()
    agent.on_start = AsyncMock()
    agent.on_stop = AsyncMock()
    pm.get_plugin.return_value = agent
    return pm, agent


@pytest.fixture
def config_file():
    """Write a config dict to a temp YAML file; yield a writer, clean up after."""
    paths: list[str] = []

    def write(data: dict) -> str:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            paths.append(f.name)
            return f.name

    yield write
    for path in paths:
        os.unlink(path)


def _patched(pm):
    """Patch Runtime's collaborators so start() uses ``pm``."""
    return (
        patch("corvidae.runtime.create_plugin_manager", return_value=pm),
        patch("corvidae.runtime.validate_dependencies"),
        patch("corvidae.runtime.configure_logging"),
    )


async def _start(config_path: str, pm) -> Runtime:
    """Run Runtime.start() against ``pm``."""
    a, b, c = _patched(pm)
    with a, b, c:
        rt = Runtime(config_path=config_path)
        await rt.start()
    return rt


class TestNullPluginsSection:
    @pytest.mark.parametrize("plugins", [None, {"disabled": None}, {}])
    async def test_null_plugins_boot_with_nothing_blocked(self, config_file, plugins):
        # plugins: (null) and plugins.disabled: (null) mean "none disabled".
        pm, agent = _mock_pm()
        await _start(config_file({"llm": _LLM, "plugins": plugins}), pm)
        agent.on_start.assert_awaited_once()
        pm.set_blocked.assert_not_called()

    @pytest.mark.parametrize(
        "plugins, needle",
        [
            ({"disabled": "memory"}, "plugins.disabled"),
            ({"disabled": ["memory", 3]}, "plugins.disabled"),
            ("memory", "plugins"),
        ],
    )
    async def test_malformed_plugins_is_a_config_error(
        self, config_file, capsys, plugins, needle
    ):
        # A wrong type is the one-line Config error, naming key and value,
        # not a traceback and not a letter-by-letter block list.
        pm, _agent = _mock_pm()
        path = config_file({"llm": _LLM, "plugins": plugins})
        with pytest.raises(SystemExit) as excinfo:
            await _start(path, pm)
        assert excinfo.value.code == 1
        # Pick the Config error line itself: an earlier test can leave a
        # logging handler that writes "--- Logging error ---" to stderr.
        lines = [line for line in capsys.readouterr().err.splitlines()
                 if line.startswith("Config error:")]
        assert len(lines) == 1, lines
        err = lines[0]
        assert needle in err and "memory" in err and path in err, err
        assert "list" in err, err  # the accepted form
        pm.set_blocked.assert_not_called()


class TestLoadedExcludesBlocked:
    async def test_blocked_names_are_not_listed_as_loaded(self, config_file, caplog):
        # pluggy keeps a blocked name as (name, None); it is not loaded.
        pm, _agent = _mock_pm()
        pm.list_name_plugin.return_value = [
            ("agent", MagicMock()), ("scheduler", None), ("registry", MagicMock()),
        ]
        with caplog.at_level(logging.INFO, logger="corvidae.runtime"):
            await _start(
                config_file({"llm": _LLM, "plugins": {"disabled": ["scheduler"]}}), pm
            )
        record = next(r for r in caplog.records if r.getMessage() == "plugins loaded")
        assert list(record.loaded) == ["agent", "registry"]
        assert list(record.blocked) == ["scheduler"]


class TestFailedStartCleansUp:
    async def _run(self, config_path: str, pm) -> None:
        """Run Runtime.run() against ``pm``."""
        a, b, c = _patched(pm)
        with a, b, c:
            await Runtime(config_path=config_path).run()

    async def test_raising_on_start_stops_plugins_then_reraises(self, config_file):
        # The plugins that started (persistence's DB) get on_stop, then the
        # original exception propagates so the process exits non-zero.
        pm, _agent = _mock_pm()
        order: list[str] = []

        async def failing_start(**kwargs):
            order.append("on_start")
            raise RuntimeError("plugin start explosion")

        async def record_stop(**kwargs):
            order.append("on_stop")
            return []

        pm.ahook.on_start = AsyncMock(side_effect=failing_start)
        pm.ahook.on_stop = AsyncMock(side_effect=record_stop)

        with pytest.raises(RuntimeError, match="plugin start explosion"):
            await self._run(config_file({"llm": _LLM}), pm)
        assert order == ["on_start", "on_stop"]

    async def test_cleanup_failure_does_not_mask_start_failure(
        self, config_file, caplog
    ):
        # A raising on_stop is logged; the start exception is what propagates.
        pm, _agent = _mock_pm()
        pm.ahook.on_start = AsyncMock(side_effect=RuntimeError("start explosion"))
        pm.ahook.on_stop = AsyncMock(side_effect=ValueError("stop explosion"))

        with caplog.at_level(logging.WARNING, logger="corvidae.runtime"):
            with pytest.raises(RuntimeError, match="start explosion"):
                await self._run(config_file({"llm": _LLM}), pm)
        assert any(
            r.levelno == logging.WARNING and r.exc_info
            and "stop explosion" in str(r.exc_info[1])
            for r in caplog.records
        ), [r.getMessage() for r in caplog.records]

    async def test_hanging_cleanup_is_bounded(self, config_file):
        # An on_stop that never returns cannot recreate the hang.
        pm, _agent = _mock_pm()
        pm.ahook.on_start = AsyncMock(side_effect=RuntimeError("start explosion"))

        async def hang(**kwargs):
            await asyncio.Event().wait()

        pm.ahook.on_stop = AsyncMock(side_effect=hang)

        async with asyncio.timeout(10):
            with pytest.raises(RuntimeError, match="start explosion"):
                await self._run(config_file({"llm": _LLM}), pm)

    async def test_real_aiosqlite_connection_is_closed(self, config_file, tmp_path):
        # End to end with the real PersistencePlugin: its DB opens in on_start,
        # a later on_start raises, and the connection's non-daemon worker
        # thread is gone afterwards, so interpreter shutdown cannot hang.
        import aiosqlite

        from corvidae.persistence import PersistencePlugin

        opened: list[aiosqlite.Connection] = []
        pm, _agent = _mock_pm()
        persistence = PersistencePlugin()

        async def start(**kwargs):
            persistence.db = await aiosqlite.connect(str(tmp_path / "s.db"))
            opened.append(persistence.db)
            raise RuntimeError("later plugin explosion")

        async def stop(**kwargs):
            await persistence.on_stop()
            return []

        pm.ahook.on_start = AsyncMock(side_effect=start)
        pm.ahook.on_stop = AsyncMock(side_effect=stop)

        with pytest.raises(RuntimeError, match="later plugin explosion"):
            await self._run(config_file({"llm": _LLM}), pm)
        # aiosqlite clears _connection when close() completes (see
        # tests/aiosqlite_guard.py), stopping its worker thread.
        assert opened and opened[0]._connection is None
