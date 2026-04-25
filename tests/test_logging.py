"""Failing tests for the logging feature described in plans/logging-design.md.

These tests verify log calls in each module. The tests fail because the
implementation does not yet exist — module-level loggers are absent in most
modules and the specific log calls have not been added.

Convention: Use `caplog` (pytest's built-in fixture) to assert on log records.
"""

import logging
import logging.config
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from sherman.channel import ChannelConfig, ChannelRegistry, load_channel_config
from sherman.conversation import ConversationLog, init_db
from sherman.conversation import resolve_system_prompt


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
        """sherman.agent_loop must expose a module-level `logger` attribute
        whose name matches the module __name__."""
        import sherman.agent_loop as mod
        assert hasattr(mod, "logger"), "agent_loop.py must define module-level `logger`"
        assert mod.logger.name == "sherman.agent_loop"

    def test_llm_has_module_logger(self):
        """sherman.llm must expose a module-level `logger` attribute."""
        import sherman.llm as mod
        assert hasattr(mod, "logger"), "llm.py must define module-level `logger`"
        assert mod.logger.name == "sherman.llm"

    def test_conversation_has_module_logger(self):
        """sherman.conversation must expose a module-level `logger` attribute."""
        import sherman.conversation as mod
        assert hasattr(mod, "logger"), "conversation.py must define module-level `logger`"
        assert mod.logger.name == "sherman.conversation"

    def test_channel_has_module_logger(self):
        """sherman.channel must expose a module-level `logger` attribute."""
        import sherman.channel as mod
        assert hasattr(mod, "logger"), "channel.py must define module-level `logger`"
        assert mod.logger.name == "sherman.channel"

    def test_prompt_has_module_logger(self):
        """resolve_system_prompt logs under 'sherman.prompt' (canonical: sherman.conversation)."""
        import sherman.conversation as mod
        assert hasattr(mod, "_prompt_logger"), "conversation.py must define module-level `_prompt_logger`"
        assert mod._prompt_logger.name == "sherman.prompt"

    def test_main_has_module_logger(self):
        """sherman.main must expose a module-level `logger` attribute."""
        import sherman.main as mod
        assert hasattr(mod, "logger"), "main.py must define module-level `logger`"
        assert mod.logger.name == "sherman.main"

    def test_agent_loop_plugin_has_module_logger(self):
        """AgentPlugin logs under 'sherman.agent' (canonical: sherman.agent)."""
        import sherman.agent as mod
        assert hasattr(mod, "logger")
        assert mod.logger.name == "sherman.agent"

    def test_plugin_manager_has_module_logger(self):
        """create_plugin_manager logs under 'sherman.plugin_manager' (canonical: sherman.hooks)."""
        import sherman.hooks as mod
        assert hasattr(mod, "_pm_logger"), "hooks.py must define module-level `_pm_logger`"
        assert mod._pm_logger.name == "sherman.plugin_manager"


# ---------------------------------------------------------------------------
# Section 2: main.py — _DEFAULT_LOGGING and dictConfig
# ---------------------------------------------------------------------------


class TestMainLogging:
    """main.py must define _DEFAULT_LOGGING and call dictConfig on startup."""

    def test_default_logging_constant_exists(self):
        """main.py must define _DEFAULT_LOGGING as a module-level constant."""
        import sherman.main as mod
        assert hasattr(mod, "_DEFAULT_LOGGING"), "main.py must define _DEFAULT_LOGGING"

    def test_default_logging_constant_has_required_keys(self):
        """_DEFAULT_LOGGING must have version, formatters, handlers, loggers, root."""
        import sherman.main as mod
        d = mod._DEFAULT_LOGGING
        assert d.get("version") == 1
        assert "formatters" in d
        assert "handlers" in d
        assert "loggers" in d
        assert "root" in d

    def test_default_logging_sherman_level_is_info(self):
        """_DEFAULT_LOGGING must configure sherman logger at INFO."""
        import sherman.main as mod
        sherman_cfg = mod._DEFAULT_LOGGING["loggers"].get("sherman", {})
        assert sherman_cfg.get("level") == "INFO"

    def test_default_logging_disable_existing_is_false(self):
        """_DEFAULT_LOGGING must set disable_existing_loggers to False."""
        import sherman.main as mod
        assert mod._DEFAULT_LOGGING.get("disable_existing_loggers") is False

    async def test_main_calls_dictconfig_with_defaults(self):
        """When no 'logging' key in config, main() must call dictConfig with
        _DEFAULT_LOGGING."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(
                {"llm": {"base_url": "http://localhost:8080", "model": "test"}},
                f,
            )
            config_path = f.name

        try:
            import asyncio
            import signal

            with patch("logging.config.dictConfig") as mock_dictconfig, \
                 patch("sherman.main.create_plugin_manager") as mock_pm_factory:
                mock_pm = MagicMock()
                mock_pm.ahook.on_start = AsyncMock(return_value=[])
                mock_pm.ahook.on_stop = AsyncMock(return_value=[])
                mock_pm_factory.return_value = mock_pm

                async def run():
                    asyncio.get_running_loop().call_later(0.05, os.kill, os.getpid(), signal.SIGINT)
                    from sherman.main import main
                    await main(config_path)

                await run()

            mock_dictconfig.assert_called()
            from sherman.main import _DEFAULT_LOGGING
            first_call_arg = mock_dictconfig.call_args_list[0][0][0]
            assert first_call_arg == _DEFAULT_LOGGING, (
                "dictConfig should be called with _DEFAULT_LOGGING when no logging key in config"
            )
        finally:
            os.unlink(config_path)

    async def test_main_calls_dictconfig_with_yaml_logging(self):
        """When 'logging' key is present in config, main() must call dictConfig
        with that config (not _DEFAULT_LOGGING)."""
        custom_logging = {
            "version": 1,
            "disable_existing_loggers": False,
            "handlers": {"null": {"class": "logging.NullHandler"}},
            "loggers": {"sherman": {"level": "DEBUG", "handlers": ["null"], "propagate": False}},
            "root": {"level": "ERROR", "handlers": ["null"]},
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(
                {
                    "llm": {"base_url": "http://localhost:8080", "model": "test"},
                    "logging": custom_logging,
                },
                f,
            )
            config_path = f.name

        try:
            import asyncio
            import signal

            with patch("logging.config.dictConfig") as mock_dictconfig, \
                 patch("sherman.main.create_plugin_manager") as mock_pm_factory:
                mock_pm = MagicMock()
                mock_pm.ahook.on_start = AsyncMock(return_value=[])
                mock_pm.ahook.on_stop = AsyncMock(return_value=[])
                mock_pm_factory.return_value = mock_pm

                async def run():
                    asyncio.get_running_loop().call_later(0.05, os.kill, os.getpid(), signal.SIGINT)
                    from sherman.main import main
                    await main(config_path)

                await run()

            mock_dictconfig.assert_called()
            first_call_arg = mock_dictconfig.call_args_list[0][0][0]
            assert first_call_arg == custom_logging, (
                "dictConfig should be called with YAML logging config when present"
            )
        finally:
            os.unlink(config_path)

    async def test_main_logs_startup_info(self, caplog):
        """main() must emit an INFO log for 'logging configured' and 'daemon starting'
        (or similar startup message)."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(
                {"llm": {"base_url": "http://localhost:8080", "model": "test"}},
                f,
            )
            config_path = f.name

        try:
            import asyncio
            import signal

            with caplog.at_level(logging.INFO, logger="sherman.main"), \
                 patch("logging.config.dictConfig"), \
                 patch("sherman.main.create_plugin_manager") as mock_pm_factory:
                mock_pm = MagicMock()
                mock_pm.ahook.on_start = AsyncMock(return_value=[])
                mock_pm.ahook.on_stop = AsyncMock(return_value=[])
                mock_pm_factory.return_value = mock_pm

                async def run():
                    asyncio.get_running_loop().call_later(0.05, os.kill, os.getpid(), signal.SIGINT)
                    from sherman.main import main
                    await main(config_path)

                await run()

            log_messages = [r.message for r in caplog.records if r.name == "sherman.main"]
            assert any("logging" in m.lower() or "starting" in m.lower() or "configured" in m.lower()
                       for m in log_messages), (
                f"Expected startup INFO log in sherman.main, got: {log_messages}"
            )
        finally:
            os.unlink(config_path)

    async def test_main_logs_shutdown_info(self, caplog):
        """main() must emit an INFO log after stop_event.wait() returns when
        shutdown signal is received (e.g., 'shutdown signal received, stopping')."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(
                {"llm": {"base_url": "http://localhost:8080", "model": "test"}},
                f,
            )
            config_path = f.name

        try:
            import asyncio
            import signal

            with caplog.at_level(logging.INFO, logger="sherman.main"), \
                 patch("logging.config.dictConfig"), \
                 patch("sherman.main.create_plugin_manager") as mock_pm_factory:
                mock_pm = MagicMock()
                mock_pm.ahook.on_start = AsyncMock(return_value=[])
                mock_pm.ahook.on_stop = AsyncMock(return_value=[])
                mock_pm_factory.return_value = mock_pm

                async def run():
                    asyncio.get_running_loop().call_later(0.05, os.kill, os.getpid(), signal.SIGINT)
                    from sherman.main import main
                    await main(config_path)

                await run()

            log_messages = [r.message for r in caplog.records if r.name == "sherman.main"]
            assert any("shutdown" in m.lower() or "stopping" in m.lower() or "signal" in m.lower()
                       for m in log_messages), (
                f"Expected shutdown INFO log in sherman.main, got: {log_messages}"
            )
        finally:
            os.unlink(config_path)


# ---------------------------------------------------------------------------
# Section 3: llm.py — INFO logs on start, stop, chat completion
# ---------------------------------------------------------------------------


class TestLLMLogging:
    """LLMClient must log INFO on start/stop and chat completion with token usage."""

    async def test_llm_start_logs_info(self, caplog):
        """LLMClient.start() must emit an INFO log containing base_url and model."""
        from sherman.llm import LLMClient
        client = LLMClient(base_url="http://localhost:8080", model="test-model")

        with caplog.at_level(logging.INFO, logger="sherman.llm"), \
             patch("aiohttp.ClientSession") as mock_session_cls:
            mock_session_cls.return_value = MagicMock()
            await client.start()

        records = [r for r in caplog.records if r.name == "sherman.llm"]
        assert records, "LLMClient.start() must emit at least one log record"
        assert any(r.levelno == logging.INFO for r in records), (
            "LLMClient.start() must emit an INFO record"
        )

    async def test_llm_stop_logs_info(self, caplog):
        """LLMClient.stop() must emit an INFO log."""
        from sherman.llm import LLMClient
        client = LLMClient(base_url="http://localhost:8080", model="test-model")
        mock_session = AsyncMock()
        mock_session.close = AsyncMock()
        client.session = mock_session

        with caplog.at_level(logging.INFO, logger="sherman.llm"):
            await client.stop()

        records = [r for r in caplog.records if r.name == "sherman.llm"]
        assert records, "LLMClient.stop() must emit at least one log record"

    async def test_llm_chat_logs_info_with_token_usage(self, caplog):
        """LLMClient.chat() must emit an INFO log after success including
        model, latency_ms, and token usage fields."""
        from aiohttp import ClientResponseError
        from sherman.llm import LLMClient

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

        with caplog.at_level(logging.INFO, logger="sherman.llm"):
            await client.chat([{"role": "user", "content": "hi"}])

        records = [r for r in caplog.records if r.name == "sherman.llm"]
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
        from sherman.llm import LLMClient

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

        client = LLMClient(base_url="http://localhost:8080", model="test-model")
        client.session = mock_session

        with caplog.at_level(logging.ERROR, logger="sherman.llm"), \
             pytest.raises(ClientResponseError):
            await client.chat([{"role": "user", "content": "hi"}])

        records = [r for r in caplog.records if r.name == "sherman.llm"]
        error_records = [r for r in records if r.levelno == logging.ERROR]
        assert error_records, "LLMClient.chat() must emit an ERROR log on HTTP failure"

    async def test_llm_chat_logs_debug_request(self, caplog):
        """LLMClient.chat() must emit a DEBUG log before the request with
        message count and tool count (not content)."""
        from sherman.llm import LLMClient

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

        with caplog.at_level(logging.DEBUG, logger="sherman.llm"):
            await client.chat([{"role": "user", "content": "test"}])

        records = [r for r in caplog.records if r.name == "sherman.llm"]
        debug_records = [r for r in records if r.levelno == logging.DEBUG]
        assert debug_records, "LLMClient.chat() must emit a DEBUG log before the request"


# ---------------------------------------------------------------------------
# Section 4: agent_loop.py — WARNING and DEBUG logs
# ---------------------------------------------------------------------------


class TestAgentLoopLogging:
    """run_agent_loop must log WARNING for edge cases and INFO per turn."""

    async def test_max_turns_logs_warning(self, caplog):
        """When max turns are exhausted, agent_loop must emit a WARNING log."""
        from sherman.agent_loop import run_agent_loop

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

        with caplog.at_level(logging.WARNING, logger="sherman.agent_loop"):
            await run_agent_loop(
                client,
                [{"role": "user", "content": "go"}],
                tools={"noop": noop},
                tool_schemas=[],
                max_turns=2,
            )

        records = [r for r in caplog.records if r.name == "sherman.agent_loop"]
        warning_records = [r for r in records if r.levelno == logging.WARNING]
        assert warning_records, (
            "run_agent_loop must emit a WARNING when max_turns is reached"
        )

    async def test_unknown_tool_logs_warning(self, caplog):
        """When LLM calls an unknown tool, agent_loop must emit a WARNING log."""
        from sherman.agent_loop import run_agent_loop

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

        with caplog.at_level(logging.WARNING, logger="sherman.agent_loop"):
            await run_agent_loop(
                client,
                [{"role": "user", "content": "go"}],
                tools={},
                tool_schemas=[],
            )

        records = [r for r in caplog.records if r.name == "sherman.agent_loop"]
        warning_records = [r for r in records if r.levelno == logging.WARNING]
        assert warning_records, (
            "run_agent_loop must emit a WARNING when an unknown tool is called"
        )

    async def test_tool_exception_logs_warning(self, caplog):
        """When a tool raises, agent_loop must emit a WARNING log with exc_info."""
        from sherman.agent_loop import run_agent_loop

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

        with caplog.at_level(logging.WARNING, logger="sherman.agent_loop"):
            await run_agent_loop(
                client,
                [{"role": "user", "content": "go"}],
                tools={"bad_tool": bad_tool},
                tool_schemas=[],
            )

        records = [r for r in caplog.records if r.name == "sherman.agent_loop"]
        warning_records = [r for r in records if r.levelno == logging.WARNING]
        assert warning_records, (
            "run_agent_loop must emit a WARNING when a tool raises an exception"
        )

    async def test_turn_info_log(self, caplog):
        """Each turn in run_agent_loop must emit at least one INFO log, and at
        least one INFO record must carry a latency_ms attribute."""
        from sherman.agent_loop import run_agent_loop

        client = MagicMock()
        client.chat = AsyncMock(
            return_value={
                "choices": [{"message": {"role": "assistant", "content": "hello"}}]
            }
        )

        with caplog.at_level(logging.INFO, logger="sherman.agent_loop"):
            await run_agent_loop(
                client,
                [{"role": "user", "content": "hi"}],
                tools={},
                tool_schemas=[],
            )

        records = [r for r in caplog.records if r.name == "sherman.agent_loop"]
        info_records = [r for r in records if r.levelno == logging.INFO]
        assert info_records, (
            "run_agent_loop must emit at least one INFO log per turn"
        )
        assert any(hasattr(r, "latency_ms") for r in info_records), (
            "at least one INFO record must carry a latency_ms attribute"
        )


# ---------------------------------------------------------------------------
# Section 5: conversation.py — WARNING for compaction, INFO, DEBUG logs
# ---------------------------------------------------------------------------


class TestConversationLogging:
    """ConversationLog must log WARNING before compaction and INFO after."""

    async def test_compaction_triggered_logs_warning(self, db, caplog):
        """compact_if_needed must log WARNING when compaction is triggered."""
        conv = ConversationLog(db, channel_id="test:chan1")
        conv.system_prompt = ""
        # 25 messages × 100 chars = 2500 chars; token_estimate ~714; 80% of 100 = 80 → triggers
        conv.messages = [{"role": "user", "content": "x" * 100} for _ in range(25)]

        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(
            return_value={"choices": [{"message": {"content": "summary"}}]}
        )

        with caplog.at_level(logging.WARNING, logger="sherman.conversation"):
            await conv.compact_if_needed(mock_client, max_tokens=100)

        records = [r for r in caplog.records if r.name == "sherman.conversation"]
        warning_records = [r for r in records if r.levelno == logging.WARNING]
        assert warning_records, (
            "compact_if_needed must emit WARNING when compaction is triggered"
        )

    async def test_compaction_completed_logs_info(self, db, caplog):
        """compact_if_needed must log INFO after compaction with messages_before
        and messages_after counts."""
        conv = ConversationLog(db, channel_id="test:chan1")
        conv.system_prompt = ""
        conv.messages = [{"role": "user", "content": "x" * 100} for _ in range(25)]

        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(
            return_value={"choices": [{"message": {"content": "summary"}}]}
        )

        with caplog.at_level(logging.INFO, logger="sherman.conversation"):
            await conv.compact_if_needed(mock_client, max_tokens=100)

        records = [r for r in caplog.records if r.name == "sherman.conversation"]
        info_records = [r for r in records if r.levelno == logging.INFO]
        assert info_records, (
            "compact_if_needed must emit INFO when compaction completes"
        )

    async def test_load_debug_log(self, db, caplog):
        """ConversationLog.load() must emit a DEBUG log with the count of
        messages loaded."""
        conv = ConversationLog(db, channel_id="test:chan1")

        with caplog.at_level(logging.DEBUG, logger="sherman.conversation"):
            await conv.load()

        records = [r for r in caplog.records if r.name == "sherman.conversation"]
        debug_records = [r for r in records if r.levelno == logging.DEBUG]
        assert debug_records, (
            "ConversationLog.load() must emit a DEBUG log with message count"
        )

    async def test_append_debug_log(self, db, caplog):
        """ConversationLog.append() must emit a DEBUG log with role and content
        length."""
        conv = ConversationLog(db, channel_id="test:chan1")

        with caplog.at_level(logging.DEBUG, logger="sherman.conversation"):
            await conv.append({"role": "user", "content": "hello"})

        records = [r for r in caplog.records if r.name == "sherman.conversation"]
        debug_records = [r for r in records if r.levelno == logging.DEBUG]
        assert debug_records, (
            "ConversationLog.append() must emit a DEBUG log with role and content length"
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

        with caplog.at_level(logging.INFO, logger="sherman.channel"):
            load_channel_config(config, registry)

        records = [r for r in caplog.records if r.name == "sherman.channel"]
        info_records = [r for r in records if r.levelno == logging.INFO]
        assert len(info_records) >= 2, (
            "load_channel_config must emit at least one INFO log per registered channel"
        )

    def test_load_channel_config_invalid_key_logs_warning(self, caplog):
        """load_channel_config must emit a WARNING before raising ValueError for
        a key with no colon."""
        config = {"channels": {"nocolon": {}}}
        registry = ChannelRegistry({})

        with caplog.at_level(logging.WARNING, logger="sherman.channel"), \
             pytest.raises(ValueError):
            load_channel_config(config, registry)

        records = [r for r in caplog.records if r.name == "sherman.channel"]
        warning_records = [r for r in records if r.levelno == logging.WARNING]
        assert warning_records, (
            "load_channel_config must emit a WARNING for invalid channel key format"
        )

    def test_resolve_config_logs_debug(self, caplog):
        """ChannelRegistry.resolve_config must emit a DEBUG log."""
        from sherman.channel import ChannelConfig, ChannelRegistry

        registry = ChannelRegistry({"system_prompt": "Agent.", "max_context_tokens": 8000})
        channel = registry.get_or_create("irc", "#lex", config=ChannelConfig())

        with caplog.at_level(logging.DEBUG, logger="sherman.channel"):
            registry.resolve_config(channel)

        records = [r for r in caplog.records if r.name == "sherman.channel"]
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
        with caplog.at_level(logging.WARNING, logger="sherman.prompt"):
            result = resolve_system_prompt([], Path("/tmp"))

        assert result == ""
        records = [r for r in caplog.records if r.name == "sherman.prompt"]
        warning_records = [r for r in records if r.levelno == logging.WARNING]
        assert warning_records, (
            "resolve_system_prompt must emit a WARNING when value is an empty list"
        )

    def test_string_prompt_logs_debug(self, caplog):
        """resolve_system_prompt with a string must emit a DEBUG log."""
        with caplog.at_level(logging.DEBUG, logger="sherman.prompt"):
            resolve_system_prompt("You are helpful.", Path("/tmp"))

        records = [r for r in caplog.records if r.name == "sherman.prompt"]
        debug_records = [r for r in records if r.levelno == logging.DEBUG]
        assert debug_records, (
            "resolve_system_prompt must emit a DEBUG log for string input"
        )

    def test_file_list_prompt_logs_debug(self, tmp_path, caplog):
        """resolve_system_prompt with a file list must emit a DEBUG log."""
        f = tmp_path / "prompt.md"
        f.write_text("You are a bot.")

        with caplog.at_level(logging.DEBUG, logger="sherman.prompt"):
            resolve_system_prompt([str(f)], tmp_path)

        records = [r for r in caplog.records if r.name == "sherman.prompt"]
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
        from sherman.hooks import create_plugin_manager

        with caplog.at_level(logging.DEBUG, logger="sherman.plugin_manager"):
            create_plugin_manager()

        records = [r for r in caplog.records if r.name == "sherman.plugin_manager"]
        debug_records = [r for r in records if r.levelno == logging.DEBUG]
        assert debug_records, (
            "create_plugin_manager() must emit a DEBUG log"
        )


# ---------------------------------------------------------------------------
# Section 8: agent_loop_plugin.py — INFO logs in on_start and on_message
# ---------------------------------------------------------------------------


class TestAgentPluginLogging:
    """AgentPlugin must log INFO on_start (tool count, channel count) and
    on_message (channel, sender, latency)."""

    async def test_on_start_logs_info_tool_and_channel_count(self, caplog):
        """on_start must emit an INFO log with tool_count and channel_count."""
        from sherman.agent import AgentPlugin
        from sherman.hooks import create_plugin_manager

        pm = create_plugin_manager()
        registry = ChannelRegistry({"system_prompt": "Test.", "max_context_tokens": 8000})
        pm.register(registry, name="registry")
        pm.ahook.send_message = AsyncMock()
        pm.ahook.on_agent_response = AsyncMock()

        plugin = AgentPlugin(pm)
        pm.register(plugin, name="agent_loop")

        mock_client = MagicMock()
        mock_client.start = AsyncMock()

        with caplog.at_level(logging.INFO, logger="sherman.agent"), \
             patch("sherman.agent.LLMClient", return_value=mock_client), \
             patch("sherman.agent.aiosqlite.connect", new_callable=AsyncMock) as mock_connect, \
             patch("sherman.agent.init_db", new_callable=AsyncMock):
            mock_connect.return_value = MagicMock()
            await plugin.on_start(config={
                "llm": {"main": {"base_url": "http://localhost:8080", "model": "test"}},
                "daemon": {"session_db": ":memory:"},
            })

        records = [r for r in caplog.records if r.name == "sherman.agent"]
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
        from sherman.agent import AgentPlugin
        from sherman.hooks import create_plugin_manager

        pm = create_plugin_manager()
        registry = ChannelRegistry({"system_prompt": "Test.", "max_context_tokens": 8000,
                                     "keep_thinking_in_history": False})
        pm.register(registry, name="registry")
        pm.ahook.send_message = AsyncMock()
        pm.ahook.on_agent_response = AsyncMock()

        plugin = AgentPlugin(pm)
        pm.register(plugin, name="agent_loop")
        plugin.db = db
        plugin._registry = registry

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(
            return_value={"choices": [{"message": {"role": "assistant", "content": "hi"}}]}
        )
        plugin.client = mock_client

        channel = registry.get_or_create("test", "scope1")
        return plugin, channel

    async def test_on_message_logs_info_channel_and_sender(self, db, caplog):
        """on_message must emit an INFO log with channel and sender."""
        plugin, channel = await self._make_on_message_plugin(db)

        with caplog.at_level(logging.INFO, logger="sherman.agent"):
            await plugin.on_message(channel=channel, sender="alice", text="hello")

        records = [r for r in caplog.records if r.name == "sherman.agent"]
        info_records = [r for r in records if r.levelno == logging.INFO]
        assert info_records, (
            "on_message must emit at least one INFO log (channel, sender)"
        )

    async def test_on_message_logs_response_with_latency(self, db, caplog):
        """on_message must emit an INFO log after response with latency_ms field."""
        plugin, channel = await self._make_on_message_plugin(db)

        with caplog.at_level(logging.INFO, logger="sherman.agent"):
            await plugin.on_message(channel=channel, sender="alice", text="hello")
            # Drain the channel queue so the consumer (which emits latency log)
            # runs before we check caplog records.
            if channel.id in plugin._queues:
                await plugin._queues[channel.id].drain()

        records = [r for r in caplog.records if r.name == "sherman.agent"]
        info_records = [r for r in records if r.levelno == logging.INFO]
        # At least one must have latency_ms
        assert any(hasattr(r, "latency_ms") for r in info_records), (
            "on_message must emit an INFO log with latency_ms structured field after response"
        )

    async def test_ensure_conversation_logs_info(self, db, caplog):
        """_ensure_conversation must emit an INFO log when initializing a new
        conversation for a channel."""
        from sherman.agent import AgentPlugin
        from sherman.hooks import create_plugin_manager

        pm = create_plugin_manager()
        registry = ChannelRegistry({"system_prompt": "Test.", "max_context_tokens": 8000,
                                     "keep_thinking_in_history": False})
        pm.register(registry, name="registry")
        pm.ahook.send_message = AsyncMock()
        pm.ahook.on_agent_response = AsyncMock()

        plugin = AgentPlugin(pm)
        pm.register(plugin, name="agent_loop")
        plugin.db = db
        plugin._registry = registry

        channel = registry.get_or_create("test", "scope1")
        assert channel.conversation is None

        with caplog.at_level(logging.INFO, logger="sherman.agent"):
            await plugin._ensure_conversation(channel)

        records = [r for r in caplog.records if r.name == "sherman.agent"]
        info_records = [r for r in records if r.levelno == logging.INFO]
        assert info_records, (
            "_ensure_conversation must emit INFO log when initializing a new conversation"
        )
