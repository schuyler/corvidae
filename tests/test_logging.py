"""Failing tests for the logging feature described in plans/logging-design.md.

These tests verify log calls in each module. The tests fail because the
implementation does not yet exist — module-level loggers are absent in most
modules and the specific log calls have not been added.

Convention: Use `caplog` (pytest's built-in fixture) to assert on log records.
"""

import logging
import logging.config
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from corvidae.channel import ChannelConfig, ChannelRegistry, load_channel_config
from corvidae.context import ContextWindow
from corvidae.persistence import init_db
from corvidae.channel import resolve_system_prompt


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_logging():
    """Reset global logging state after each test so caplog is clean."""
    yield
    logging.config.dictConfig({"version": 1, "disable_existing_loggers": False})


# ---------------------------------------------------------------------------
# Section 1: Logger naming convention
# ---------------------------------------------------------------------------


class TestLoggerNamingConvention:
    """Every module must have a module-level logger named after __name__."""

    def test_agent_loop_has_module_logger(self):
        """corvidae.agent_loop must expose a module-level `logger` attribute
        whose name matches the module __name__."""
        import corvidae.agent_loop as mod
        assert hasattr(mod, "logger"), "agent_loop.py must define module-level `logger`"
        assert mod.logger.name == "corvidae.agent_loop"

    def test_llm_has_module_logger(self):
        """corvidae.llm must expose a module-level `logger` attribute."""
        import corvidae.llm as mod
        assert hasattr(mod, "logger"), "llm.py must define module-level `logger`"
        assert mod.logger.name == "corvidae.llm"

    def test_context_has_module_logger(self):
        """corvidae.context must expose a module-level `logger` attribute."""
        import corvidae.context as mod
        assert hasattr(mod, "logger"), "context.py must define module-level `logger`"
        assert mod.logger.name == "corvidae.context"

    def test_channel_has_module_logger(self):
        """corvidae.channel must expose a module-level `logger` attribute."""
        import corvidae.channel as mod
        assert hasattr(mod, "logger"), "channel.py must define module-level `logger`"
        assert mod.logger.name == "corvidae.channel"

    def test_prompt_has_module_logger(self):
        """resolve_system_prompt logs under 'corvidae.prompt' (canonical: corvidae.channel)."""
        import corvidae.channel as mod
        assert hasattr(mod, "_prompt_logger"), "channel.py must define module-level `_prompt_logger`"
        assert mod._prompt_logger.name == "corvidae.prompt"

    def test_main_has_module_logger(self):
        """corvidae.main must expose a module-level `logger` attribute."""
        import corvidae.main as mod
        assert hasattr(mod, "logger"), "main.py must define module-level `logger`"
        assert mod.logger.name == "corvidae.main"

    def test_agent_loop_plugin_has_module_logger(self):
        """Agent logs under 'corvidae.agent' (canonical: corvidae.agent)."""
        import corvidae.agent as mod
        assert hasattr(mod, "logger")
        assert mod.logger.name == "corvidae.agent"

    def test_plugin_manager_has_module_logger(self):
        """create_plugin_manager logs under 'corvidae.plugin_manager' (canonical: corvidae.hooks)."""
        import corvidae.hooks as mod
        assert hasattr(mod, "_pm_logger"), "hooks.py must define module-level `_pm_logger`"
        assert mod._pm_logger.name == "corvidae.plugin_manager"


# ---------------------------------------------------------------------------
# Section 2: configure_logging() — unit tests
# ---------------------------------------------------------------------------


class TestConfigureLogging:
    """Unit tests for the configure_logging() function in corvidae.logging."""

    def test_defaults_to_stderr(self):
        """configure_logging() with no args installs a StreamHandler on stderr."""
        from corvidae.logging import configure_logging

        configure_logging()
        corvidae_logger = logging.getLogger("corvidae")
        handlers = corvidae_logger.handlers
        assert handlers, "corvidae logger must have at least one handler"
        assert any(
            isinstance(h, logging.StreamHandler)
            and getattr(h, "stream", None) is not None
            and h.stream is sys.stderr
            for h in handlers
        ), f"Expected a StreamHandler on stderr, got: {handlers}"

    def test_file_creates_rotating_handler(self, tmp_path):
        """configure_logging(file=...) installs a RotatingFileHandler."""
        import logging.handlers
        from corvidae.logging import configure_logging

        log_file = str(tmp_path / "test.log")
        configure_logging(file=log_file)
        corvidae_logger = logging.getLogger("corvidae")
        handlers = corvidae_logger.handlers
        assert any(
            isinstance(h, logging.handlers.RotatingFileHandler)
            for h in handlers
        ), f"Expected a RotatingFileHandler, got: {handlers}"

    def test_level_applied(self):
        """configure_logging(level='DEBUG') sets corvidae logger to DEBUG."""
        from corvidae.logging import configure_logging

        configure_logging(level="DEBUG")
        corvidae_logger = logging.getLogger("corvidae")
        assert corvidae_logger.level == logging.DEBUG, (
            f"Expected DEBUG ({logging.DEBUG}), got {corvidae_logger.level}"
        )

    def test_invalid_level_raises(self):
        """configure_logging with an unrecognized level raises ValueError."""
        from corvidae.logging import configure_logging

        with pytest.raises(ValueError, match="Invalid log level"):
            configure_logging(level="VERBOSE")

    def test_structured_formatter_used(self):
        """The handler installed by configure_logging uses StructuredFormatter."""
        from corvidae.logging import configure_logging, StructuredFormatter

        configure_logging()
        corvidae_logger = logging.getLogger("corvidae")
        formatters = [h.formatter for h in corvidae_logger.handlers if h.formatter]
        assert any(
            isinstance(f, StructuredFormatter) for f in formatters
        ), f"Expected StructuredFormatter on at least one handler, got: {formatters}"

    def test_root_logger_at_warning(self):
        """configure_logging() leaves the root logger at WARNING."""
        from corvidae.logging import configure_logging

        configure_logging()
        root = logging.getLogger()
        assert root.level == logging.WARNING, (
            f"Expected root logger at WARNING ({logging.WARNING}), got {root.level}"
        )

    def test_level_case_insensitive(self):
        """configure_logging(level='debug') is normalized via .upper() and works."""
        from corvidae.logging import configure_logging

        configure_logging(level="debug")
        corvidae_logger = logging.getLogger("corvidae")
        assert corvidae_logger.level == logging.DEBUG, (
            f"Expected DEBUG ({logging.DEBUG}) after level='debug', got {corvidae_logger.level}"
        )


# ---------------------------------------------------------------------------
# Section 3: main() integration — logging configuration
# ---------------------------------------------------------------------------


def _make_minimal_config(tmp_path, extra=None):
    """Write a minimal agent.yaml for integration tests and return its path."""
    data = {"llm": {"base_url": "http://localhost:8080", "model": "test"}}
    if extra:
        data.update(extra)
    config_file = tmp_path / "agent.yaml"
    config_file.write_text(yaml.dump(data))
    return str(config_file)


def _make_mock_pm():
    mock_pm = MagicMock()
    mock_pm.ahook.on_init = AsyncMock(return_value=[])
    mock_pm.ahook.on_start = AsyncMock(return_value=[])
    mock_pm.ahook.on_stop = AsyncMock(return_value=[])
    mock_agent = MagicMock()
    mock_agent.on_start = AsyncMock()
    mock_agent.on_stop = AsyncMock()
    mock_pm.get_plugin.return_value = mock_agent
    return mock_pm


def _make_mock_agent():
    mock_agent = MagicMock()
    mock_agent.on_start = AsyncMock()
    mock_agent.on_stop = AsyncMock()
    return mock_agent


class TestMainLoggingConfiguration:
    """Integration tests for main()'s logging configuration via configure_logging()."""

    async def _run_main(self, config_path, **main_kwargs):
        """Run main() with a SIGINT scheduled after 50ms and the standard plugin mocks."""
        import asyncio
        import signal

        with patch("corvidae.main.configure_logging") as mock_configure, \
             patch("corvidae.main.create_plugin_manager") as mock_pm_factory:
            mock_pm_factory.return_value = _make_mock_pm()

            async def run():
                asyncio.get_running_loop().call_later(
                    0.05, os.kill, os.getpid(), signal.SIGINT
                )
                from corvidae.main import main
                await main(config_path, **main_kwargs)

            await run()

        return mock_configure

    async def test_programmatic_default_is_stderr(self, tmp_path):
        """main() without cli_mode calls configure_logging with file=None (stderr)."""
        config_path = _make_minimal_config(tmp_path)
        mock_configure = await self._run_main(config_path)

        mock_configure.assert_called_once()
        _, kwargs = mock_configure.call_args
        assert kwargs.get("file") is None, (
            f"Expected file=None in programmatic mode, got: {kwargs}"
        )

    async def test_cli_mode_default_is_file(self, tmp_path):
        """main(cli_mode=True) defaults file to 'corvidae.log'."""
        config_path = _make_minimal_config(tmp_path)
        mock_configure = await self._run_main(config_path, cli_mode=True)

        mock_configure.assert_called_once()
        _, kwargs = mock_configure.call_args
        assert kwargs.get("file") == "corvidae.log", (
            f"Expected file='corvidae.log' in cli_mode, got: {kwargs}"
        )

    async def test_yaml_file_overrides_cli_default(self, tmp_path):
        """YAML logging.file='custom.log' overrides the cli_mode default of 'corvidae.log'."""
        config_path = _make_minimal_config(
            tmp_path, extra={"logging": {"file": "custom.log"}}
        )
        mock_configure = await self._run_main(config_path, cli_mode=True)

        mock_configure.assert_called_once()
        _, kwargs = mock_configure.call_args
        assert kwargs.get("file") == "custom.log", (
            f"Expected file='custom.log' from YAML override, got: {kwargs}"
        )

    async def test_yaml_level_passed_through(self, tmp_path):
        """YAML logging.level='DEBUG' is forwarded to configure_logging."""
        config_path = _make_minimal_config(
            tmp_path, extra={"logging": {"level": "DEBUG"}}
        )
        mock_configure = await self._run_main(config_path)

        mock_configure.assert_called_once()
        _, kwargs = mock_configure.call_args
        assert kwargs.get("level") == "DEBUG", (
            f"Expected level='DEBUG' from YAML, got: {kwargs}"
        )

    async def test_yaml_file_passed_through(self, tmp_path):
        """YAML logging.file is forwarded as the file kwarg to configure_logging."""
        log_file = str(tmp_path / "custom.log")
        config_path = _make_minimal_config(
            tmp_path, extra={"logging": {"file": log_file}}
        )
        mock_configure = await self._run_main(config_path)

        mock_configure.assert_called_once()
        _, kwargs = mock_configure.call_args
        assert kwargs.get("file") == log_file, (
            f"Expected file={log_file!r} from YAML, got: {kwargs}"
        )

    async def test_main_logs_startup_info(self, tmp_path, caplog):
        """main() must emit an INFO log for 'logging configured' (startup message)."""
        config_path = _make_minimal_config(tmp_path)

        import asyncio
        import signal

        with caplog.at_level(logging.INFO, logger="corvidae.main"), \
             patch("corvidae.main.configure_logging"), \
             patch("corvidae.main.create_plugin_manager") as mock_pm_factory:
            mock_pm_factory.return_value = _make_mock_pm()

            async def run():
                asyncio.get_running_loop().call_later(
                    0.05, os.kill, os.getpid(), signal.SIGINT
                )
                from corvidae.main import main
                await main(config_path)

            await run()

        log_messages = [r.message for r in caplog.records if r.name == "corvidae.main"]
        assert any(
            "logging" in m.lower() or "starting" in m.lower() or "configured" in m.lower()
            for m in log_messages
        ), f"Expected startup INFO log in corvidae.main, got: {log_messages}"

    async def test_main_logs_shutdown_info(self, tmp_path, caplog):
        """main() must emit an INFO log when shutdown signal is received."""
        config_path = _make_minimal_config(tmp_path)

        import asyncio
        import signal

        with caplog.at_level(logging.INFO, logger="corvidae.main"), \
             patch("corvidae.main.configure_logging"), \
             patch("corvidae.main.create_plugin_manager") as mock_pm_factory:
            mock_pm_factory.return_value = _make_mock_pm()

            async def run():
                asyncio.get_running_loop().call_later(
                    0.05, os.kill, os.getpid(), signal.SIGINT
                )
                from corvidae.main import main
                await main(config_path)

            await run()

        log_messages = [r.message for r in caplog.records if r.name == "corvidae.main"]
        assert any(
            "shutdown" in m.lower() or "stopping" in m.lower() or "signal" in m.lower()
            for m in log_messages
        ), f"Expected shutdown INFO log in corvidae.main, got: {log_messages}"


# ---------------------------------------------------------------------------
# Section 3: llm.py — INFO logs on start, stop, chat completion
# ---------------------------------------------------------------------------


class TestLLMLogging:
    """LLMClient must log INFO on start/stop and chat completion with token usage."""

    async def test_llm_start_logs_info(self, caplog):
        """LLMClient.start() must emit an INFO log containing base_url and model."""
        from corvidae.llm import LLMClient
        client = LLMClient(base_url="http://localhost:8080", model="test-model")

        with caplog.at_level(logging.INFO, logger="corvidae.llm"), \
             patch("aiohttp.ClientSession") as mock_session_cls:
            mock_session_cls.return_value = MagicMock()
            await client.start()

        records = [r for r in caplog.records if r.name == "corvidae.llm"]
        assert records, "LLMClient.start() must emit at least one log record"
        assert any(r.levelno == logging.INFO for r in records), (
            "LLMClient.start() must emit an INFO record"
        )

    async def test_llm_stop_logs_info(self, caplog):
        """LLMClient.stop() must emit an INFO log."""
        from corvidae.llm import LLMClient
        client = LLMClient(base_url="http://localhost:8080", model="test-model")
        mock_session = AsyncMock()
        mock_session.close = AsyncMock()
        client.session = mock_session

        with caplog.at_level(logging.INFO, logger="corvidae.llm"):
            await client.stop()

        records = [r for r in caplog.records if r.name == "corvidae.llm"]
        assert records, "LLMClient.stop() must emit at least one log record"

    async def test_llm_chat_logs_info_with_token_usage(self, caplog):
        """LLMClient.chat() must emit an INFO log after success including
        model, latency_ms, and token usage fields."""
        from aiohttp import ClientResponseError
        from corvidae.llm import LLMClient

        mock_completion = {
            "choices": [{"message": {"role": "assistant", "content": "hi"}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_completion)
        mock_response.raise_for_status = MagicMock()

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_cm)

        client = LLMClient(base_url="http://localhost:8080", model="test-model")
        client.session = mock_session

        with caplog.at_level(logging.INFO, logger="corvidae.llm"):
            await client.chat([{"role": "user", "content": "hi"}])

        records = [r for r in caplog.records if r.name == "corvidae.llm"]
        info_records = [r for r in records if r.levelno == logging.INFO]
        assert info_records, "LLMClient.chat() must emit an INFO log after completion"

        # Check that the log record carries structured extra fields
        completion_record = info_records[-1]
        assert hasattr(completion_record, "latency_ms"), (
            "chat completion INFO log must include latency_ms as structured field"
        )

    async def test_llm_chat_logs_error_on_http_failure(self, caplog):
        """LLMClient.chat() must emit an ERROR log when the HTTP call fails."""
        from aiohttp import ClientResponseError
        from corvidae.llm import LLMClient

        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.raise_for_status = MagicMock(
            side_effect=ClientResponseError(
                request_info=MagicMock(), history=(), status=500
            )
        )

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_cm)

        client = LLMClient(base_url="http://localhost:8080", model="test-model", max_retries=0)
        client.session = mock_session

        with caplog.at_level(logging.ERROR, logger="corvidae.llm"), \
             pytest.raises(ClientResponseError):
            await client.chat([{"role": "user", "content": "hi"}])

        records = [r for r in caplog.records if r.name == "corvidae.llm"]
        error_records = [r for r in records if r.levelno == logging.ERROR]
        assert error_records, "LLMClient.chat() must emit an ERROR log on HTTP failure"

    async def test_llm_chat_logs_debug_request(self, caplog):
        """LLMClient.chat() must emit a DEBUG log before the request with
        message count and tool count (not content)."""
        from corvidae.llm import LLMClient

        mock_completion = {
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        }
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_completion)
        mock_response.raise_for_status = MagicMock()

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_cm)

        client = LLMClient(base_url="http://localhost:8080", model="test-model")
        client.session = mock_session

        with caplog.at_level(logging.DEBUG, logger="corvidae.llm"):
            await client.chat([{"role": "user", "content": "test"}])

        records = [r for r in caplog.records if r.name == "corvidae.llm"]
        debug_records = [r for r in records if r.levelno == logging.DEBUG]
        assert debug_records, "LLMClient.chat() must emit a DEBUG log before the request"


# ---------------------------------------------------------------------------
# Section 4: agent_loop.py — WARNING and DEBUG logs
# ---------------------------------------------------------------------------


class TestAgentLoopLogging:
    """run_agent_loop must log WARNING for edge cases and INFO per turn."""

    async def test_max_turns_logs_warning(self, caplog):
        """When max turns are exhausted, agent_loop must emit a WARNING log."""
        from corvidae.agent_loop import run_agent_loop

        client = MagicMock()
        client.chat = AsyncMock(
            return_value={
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"id": "c1", "function": {"name": "noop", "arguments": "{}"}}],
                    }
                }]
            }
        )
        noop = AsyncMock(return_value="result")

        with caplog.at_level(logging.WARNING, logger="corvidae.agent_loop"):
            await run_agent_loop(
                client,
                [{"role": "user", "content": "go"}],
                tools={"noop": noop},
                tool_schemas=[],
                max_turns=2,
            )

        records = [r for r in caplog.records if r.name == "corvidae.agent_loop"]
        warning_records = [r for r in records if r.levelno == logging.WARNING]
        assert warning_records, (
            "run_agent_loop must emit a WARNING when max_turns is reached"
        )

    async def test_unknown_tool_logs_warning(self, caplog):
        """When LLM calls an unknown tool, agent_loop must emit a WARNING log."""
        from corvidae.agent_loop import run_agent_loop

        client = MagicMock()
        client.chat = AsyncMock(
            side_effect=[
                {"choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"id": "c1", "function": {"name": "ghost_tool", "arguments": "{}"}}],
                    }
                }]},
                {"choices": [{"message": {"role": "assistant", "content": "done"}}]},
            ]
        )

        with caplog.at_level(logging.WARNING, logger="corvidae.tool"):
            await run_agent_loop(
                client,
                [{"role": "user", "content": "go"}],
                tools={},
                tool_schemas=[],
            )

        records = [r for r in caplog.records if r.name == "corvidae.tool"]
        warning_records = [r for r in records if r.levelno == logging.WARNING]
        assert warning_records, (
            "run_agent_loop must emit a WARNING when an unknown tool is called"
        )

    async def test_tool_exception_logs_warning(self, caplog):
        """When a tool raises, agent_loop must emit a WARNING log with exc_info."""
        from corvidae.agent_loop import run_agent_loop

        async def bad_tool(**kwargs):
            raise ValueError("something broke")

        client = MagicMock()
        client.chat = AsyncMock(
            side_effect=[
                {"choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"id": "c1", "function": {"name": "bad_tool", "arguments": "{}"}}],
                    }
                }]},
                {"choices": [{"message": {"role": "assistant", "content": "recovered"}}]},
            ]
        )

        with caplog.at_level(logging.WARNING, logger="corvidae.tool"):
            await run_agent_loop(
                client,
                [{"role": "user", "content": "go"}],
                tools={"bad_tool": bad_tool},
                tool_schemas=[],
            )

        records = [r for r in caplog.records if r.name == "corvidae.tool"]
        warning_records = [r for r in records if r.levelno == logging.WARNING]
        assert warning_records, (
            "run_agent_loop must emit a WARNING when a tool raises an exception"
        )

    async def test_turn_info_log(self, caplog):
        """Each turn in run_agent_loop must emit at least one INFO log, and at
        least one INFO record must carry a latency_ms attribute."""
        from corvidae.agent_loop import run_agent_loop

        client = MagicMock()
        client.chat = AsyncMock(
            return_value={
                "choices": [{"message": {"role": "assistant", "content": "hello"}}]
            }
        )

        with caplog.at_level(logging.INFO, logger="corvidae.agent_loop"):
            await run_agent_loop(
                client,
                [{"role": "user", "content": "hi"}],
                tools={},
                tool_schemas=[],
            )

        records = [r for r in caplog.records if r.name == "corvidae.agent_loop"]
        info_records = [r for r in records if r.levelno == logging.INFO]
        assert info_records, (
            "run_agent_loop must emit at least one INFO log per turn"
        )
        assert any(hasattr(r, "latency_ms") for r in info_records), (
            "at least one INFO record must carry a latency_ms attribute"
        )


# ---------------------------------------------------------------------------
# Section 5: compaction.py — WARNING for compaction, INFO, DEBUG logs
# ---------------------------------------------------------------------------


class TestCompactionLogging:
    """CompactionPlugin must log WARNING before compaction."""

    async def test_compaction_triggered_logs_warning(self, caplog):
        """CompactionPlugin.compact_conversation must log WARNING when compaction is triggered."""
        from corvidae.compaction import CompactionPlugin

        conv = ContextWindow("test:chan1")
        conv.system_prompt = ""
        # 25 messages × 100 chars = 2500 chars; token_estimate ~714; 80% of 100 = 80 → triggers
        conv.messages = [{"role": "user", "content": "x" * 100} for _ in range(25)]

        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(
            return_value={"choices": [{"message": {"content": "summary"}}]}
        )

        mock_channel = MagicMock()
        mock_channel.id = "test:chan1"

        plugin = CompactionPlugin(None)
        plugin._llm_client = mock_client
        with caplog.at_level(logging.WARNING, logger="corvidae.compaction"):
            await plugin.compact_conversation(
                channel=mock_channel, conversation=conv, max_tokens=100
            )

        records = [r for r in caplog.records if r.name == "corvidae.compaction"]
        warning_records = [r for r in records if r.levelno == logging.WARNING]
        assert warning_records, (
            "CompactionPlugin.compact_conversation must emit WARNING when compaction is triggered"
        )


# ---------------------------------------------------------------------------
# Section 6: channel.py — INFO per registered channel, WARNING on invalid key
# ---------------------------------------------------------------------------


class TestChannelLogging:
    """load_channel_config must log INFO per channel and WARNING on bad key format.
    ChannelRegistry.resolve_config must log DEBUG."""

    def test_load_channel_config_logs_info_per_channel(self, caplog):
        """load_channel_config must emit one INFO log per registered channel."""
        config = {
            "channels": {
                "irc:#lex": {"system_prompt": "IRC prompt."},
                "signal:+15551234567": {},
            }
        }
        registry = ChannelRegistry({})

        with caplog.at_level(logging.INFO, logger="corvidae.channel"):
            load_channel_config(config, registry)

        records = [r for r in caplog.records if r.name == "corvidae.channel"]
        info_records = [r for r in records if r.levelno == logging.INFO]
        assert len(info_records) >= 2, (
            "load_channel_config must emit at least one INFO log per registered channel"
        )

    def test_load_channel_config_invalid_key_logs_warning(self, caplog):
        """load_channel_config must emit a WARNING before raising ValueError for
        a key with no colon."""
        config = {"channels": {"nocolon": {}}}
        registry = ChannelRegistry({})

        with caplog.at_level(logging.WARNING, logger="corvidae.channel"), \
             pytest.raises(ValueError):
            load_channel_config(config, registry)

        records = [r for r in caplog.records if r.name == "corvidae.channel"]
        warning_records = [r for r in records if r.levelno == logging.WARNING]
        assert warning_records, (
            "load_channel_config must emit a WARNING for invalid channel key format"
        )

    def test_resolve_config_logs_debug(self, caplog):
        """ChannelRegistry.resolve_config must emit a DEBUG log."""
        from corvidae.channel import ChannelConfig, ChannelRegistry

        registry = ChannelRegistry({"system_prompt": "Agent.", "max_context_tokens": 8000})
        channel = registry.get_or_create("irc", "#lex", config=ChannelConfig())

        with caplog.at_level(logging.DEBUG, logger="corvidae.channel"):
            registry.resolve_config(channel)

        records = [r for r in caplog.records if r.name == "corvidae.channel"]
        debug_records = [r for r in records if r.levelno == logging.DEBUG]
        assert debug_records, (
            "ChannelRegistry.resolve_config must emit a DEBUG log"
        )


# ---------------------------------------------------------------------------
# Section 7: prompt.py — WARNING on empty list, DEBUG on resolution
# ---------------------------------------------------------------------------


class TestPromptLogging:
    """resolve_system_prompt must warn on empty list and debug on resolution."""

    def test_empty_list_logs_warning(self, caplog):
        """resolve_system_prompt with an empty list must emit a WARNING."""
        with caplog.at_level(logging.WARNING, logger="corvidae.prompt"):
            result = resolve_system_prompt([], Path("/tmp"))

        assert result == ""
        records = [r for r in caplog.records if r.name == "corvidae.prompt"]
        warning_records = [r for r in records if r.levelno == logging.WARNING]
        assert warning_records, (
            "resolve_system_prompt must emit a WARNING when value is an empty list"
        )

    def test_string_prompt_logs_debug(self, caplog):
        """resolve_system_prompt with a string must emit a DEBUG log."""
        with caplog.at_level(logging.DEBUG, logger="corvidae.prompt"):
            resolve_system_prompt("You are helpful.", Path("/tmp"))

        records = [r for r in caplog.records if r.name == "corvidae.prompt"]
        debug_records = [r for r in records if r.levelno == logging.DEBUG]
        assert debug_records, (
            "resolve_system_prompt must emit a DEBUG log for string input"
        )

    def test_file_list_prompt_logs_debug(self, tmp_path, caplog):
        """resolve_system_prompt with a file list must emit a DEBUG log."""
        f = tmp_path / "prompt.md"
        f.write_text("You are a bot.")

        with caplog.at_level(logging.DEBUG, logger="corvidae.prompt"):
            resolve_system_prompt([str(f)], tmp_path)

        records = [r for r in caplog.records if r.name == "corvidae.prompt"]
        debug_records = [r for r in records if r.levelno == logging.DEBUG]
        assert debug_records, (
            "resolve_system_prompt must emit a DEBUG log for file list input"
        )


# ---------------------------------------------------------------------------
# Section 8: plugin_manager.py — DEBUG log on create_plugin_manager
# ---------------------------------------------------------------------------


class TestPluginManagerLogging:
    """plugin_manager must log DEBUG when created."""

    def test_create_plugin_manager_logs_debug(self, caplog):
        """create_plugin_manager() must emit a DEBUG log."""
        from corvidae.hooks import create_plugin_manager

        with caplog.at_level(logging.DEBUG, logger="corvidae.plugin_manager"):
            create_plugin_manager()

        records = [r for r in caplog.records if r.name == "corvidae.plugin_manager"]
        debug_records = [r for r in records if r.levelno == logging.DEBUG]
        assert debug_records, (
            "create_plugin_manager() must emit a DEBUG log"
        )


# ---------------------------------------------------------------------------
# Section 8: agent_loop_plugin.py — INFO logs in on_start and on_message
# ---------------------------------------------------------------------------


class TestAgentLogging:
    """Agent must log INFO on_start (tool count, channel count) and
    on_message (channel, sender, latency)."""

    async def test_on_start_logs_info_tool_and_channel_count(self, caplog):
        """on_start must emit an INFO log with tool_count and channel_count."""
        from corvidae.agent import Agent
        from corvidae.hooks import create_plugin_manager

        pm = create_plugin_manager()
        registry = ChannelRegistry({"system_prompt": "Test.", "max_context_tokens": 8000})
        pm.register(registry, name="registry")
        pm.ahook.send_message = AsyncMock()
        pm.ahook.on_agent_response = AsyncMock()

        from corvidae.llm_plugin import LLMPlugin
        from corvidae.tool_collection import ToolCollectionPlugin
        from corvidae.tool import ToolRegistry
        mock_llm = LLMPlugin(pm)
        mock_llm.main_client = MagicMock()
        pm.register(mock_llm, name="llm")

        tools_plugin = ToolCollectionPlugin(pm)
        tools_plugin.registry = ToolRegistry()
        pm.register(tools_plugin, name="tools")

        plugin = Agent(pm)
        pm.register(plugin, name="agent")

        with caplog.at_level(logging.INFO, logger="corvidae.agent"):
            await plugin.on_start(config={
                "llm": {"main": {"base_url": "http://localhost:8080", "model": "test"}},
                "daemon": {"session_db": ":memory:"},
            })

        records = [r for r in caplog.records if r.name == "corvidae.agent"]
        info_records = [r for r in records if r.levelno == logging.INFO]
        assert info_records, (
            "on_start must emit at least one INFO log (tool_count, channel_count)"
        )
        # At least one record should have tool_count extra field
        assert any(hasattr(r, "tool_count") for r in info_records), (
            "on_start INFO log must include tool_count as structured field"
        )

    async def _make_on_message_plugin(self, db):
        """Shared setup for on_message tests: returns (plugin, channel)."""
        from corvidae.agent import Agent
        from corvidae.hooks import create_plugin_manager
        from corvidae.persistence import PersistencePlugin

        pm = create_plugin_manager()
        registry = ChannelRegistry({"system_prompt": "Test.", "max_context_tokens": 8000,
                                     "keep_thinking_in_history": False})
        pm.register(registry, name="registry")
        pm.ahook.send_message = AsyncMock()
        pm.ahook.on_agent_response = AsyncMock()

        # Register PersistencePlugin with injected in-memory DB
        persistence = PersistencePlugin(pm)
        persistence.db = db
        persistence._registry = registry
        pm.register(persistence, name="persistence")

        plugin = Agent(pm)
        pm.register(plugin, name="agent")
        plugin._registry = registry

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(
            return_value={"choices": [{"message": {"role": "assistant", "content": "hi"}}]}
        )
        plugin._client = mock_client

        channel = registry.get_or_create("test", "scope1")
        return plugin, channel

    async def test_on_message_logs_info_channel_and_sender(self, db, caplog):
        """on_message must emit an INFO log with channel and sender."""
        plugin, channel = await self._make_on_message_plugin(db)

        with caplog.at_level(logging.INFO, logger="corvidae.agent"):
            await plugin.on_message(channel=channel, sender="alice", text="hello")

        records = [r for r in caplog.records if r.name == "corvidae.agent"]
        info_records = [r for r in records if r.levelno == logging.INFO]
        assert info_records, (
            "on_message must emit at least one INFO log (channel, sender)"
        )

    async def test_on_message_logs_response_with_latency(self, db, caplog):
        """on_message must emit an INFO log after response with latency_ms field."""
        plugin, channel = await self._make_on_message_plugin(db)

        with caplog.at_level(logging.INFO, logger="corvidae.agent"):
            await plugin.on_message(channel=channel, sender="alice", text="hello")
            # Drain the channel queue so the consumer (which emits latency log)
            # runs before we check caplog records.
            if channel.id in plugin.queues:
                await plugin.queues[channel.id].drain()

        records = [r for r in caplog.records if r.name == "corvidae.agent"]
        info_records = [r for r in records if r.levelno == logging.INFO]
        # At least one must have latency_ms
        assert any(hasattr(r, "latency_ms") for r in info_records), (
            "on_message must emit an INFO log with latency_ms structured field after response"
        )

    async def test_load_conversation_logs_info(self, db, caplog):
        """PersistencePlugin.load_conversation must emit an INFO log when loading
        an existing conversation for a channel."""
        import json
        import time
        from corvidae.hooks import create_plugin_manager
        from corvidae.persistence import PersistencePlugin

        pm = create_plugin_manager()
        registry = ChannelRegistry({"system_prompt": "Test.", "max_context_tokens": 8000,
                                     "keep_thinking_in_history": False})
        pm.register(registry, name="registry")

        persistence = PersistencePlugin(pm)
        persistence.db = db
        persistence._registry = registry
        pm.register(persistence, name="persistence")

        channel = registry.get_or_create("test", "scope1")

        # Insert a row so load_conversation returns results and emits the INFO log
        await db.execute(
            "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
            "VALUES (?, ?, ?, 'message')",
            (channel.id, json.dumps({"role": "user", "content": "hi"}), time.time()),
        )
        await db.commit()

        with caplog.at_level(logging.INFO, logger="corvidae.persistence"):
            await persistence.load_conversation(channel=channel)

        records = [r for r in caplog.records if r.name == "corvidae.persistence"]
        info_records = [r for r in records if r.levelno == logging.INFO]
        assert info_records, (
            "PersistencePlugin.load_conversation must emit INFO log when loading an existing conversation"
        )
