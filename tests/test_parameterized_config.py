"""RED TDD tests for parameterize-hardcoded-values design.

These tests FAIL because production code has not been changed yet.
They prove the specification is testable before implementation begins.

Design: /Users/sderle/code/sherman/plans/parameterize-hardcoded-values.md
"""

import asyncio
import collections
import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.timeout(30)


# ===========================================================================
# Group A — Tool functions
# ===========================================================================


class TestShellTimeout:
    async def test_shell_timeout_default(self):
        """shell() uses 30s timeout by default."""
        from corvidae.tools.shell import shell

        sig = inspect.signature(shell)
        assert "timeout" in sig.parameters, (
            "shell() must accept a 'timeout' parameter"
        )
        param = sig.parameters["timeout"]
        assert param.default == 30, (
            f"shell() default timeout must be 30, got {param.default!r}"
        )

    async def test_shell_timeout_override(self):
        """shell(timeout=10) passes 10 to asyncio.wait_for."""
        from corvidae.tools.shell import shell

        captured_timeout = []

        original_wait_for = asyncio.wait_for

        async def mock_wait_for(coro, timeout=None):
            captured_timeout.append(timeout)
            # Cancel the coroutine cleanly so the subprocess doesn't actually run.
            coro.close()
            raise asyncio.TimeoutError()

        with patch("asyncio.wait_for", side_effect=mock_wait_for):
            # Suppress the TimeoutError from shell's own handling.
            result = await shell("echo hi", timeout=10)

        assert captured_timeout, "asyncio.wait_for was never called"
        assert captured_timeout[0] == 10, (
            f"shell(timeout=10) must pass timeout=10 to wait_for, got {captured_timeout[0]!r}"
        )

    def test_shell_timeout_error_template(self):
        """TIMEOUT_ERROR_TEMPLATE constant exists in shell module."""
        import corvidae.tools.shell as shell_mod

        assert hasattr(shell_mod, "TIMEOUT_ERROR_TEMPLATE"), (
            "corvidae.tools.shell must define TIMEOUT_ERROR_TEMPLATE"
        )
        template = shell_mod.TIMEOUT_ERROR_TEMPLATE
        assert isinstance(template, str), "TIMEOUT_ERROR_TEMPLATE must be a str"
        assert "{timeout}" in template, (
            "TIMEOUT_ERROR_TEMPLATE must contain '{timeout}' placeholder"
        )


class TestWebFetchMaxResponse:
    async def test_web_fetch_max_response_default(self):
        """web_fetch truncates at 50000 by default."""
        from corvidae.tools.web import web_fetch

        sig = inspect.signature(web_fetch)
        assert "max_response_bytes" in sig.parameters, (
            "web_fetch() must accept a 'max_response_bytes' parameter"
        )
        param = sig.parameters["max_response_bytes"]
        assert param.default == 50_000, (
            f"web_fetch() default max_response_bytes must be 50000, got {param.default!r}"
        )

    async def test_web_fetch_max_response_override(self):
        """web_fetch(max_response_bytes=100) truncates at 100 bytes."""
        from corvidae.tools.web import web_fetch

        long_body = "x" * 500

        mock_response = AsyncMock()
        mock_response.status = 200
        # Mock the new API: content.readexactly() and get_encoding()
        mock_response.content = AsyncMock()
        mock_response.content.readexactly = AsyncMock(return_value=long_body[:100].encode())
        mock_response.get_encoding = MagicMock(return_value="utf-8")

        mock_response_ctx = AsyncMock()
        mock_response_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response_ctx)

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            result = await web_fetch("http://example.com", max_response_bytes=100)

        # Result must be no more than 100 chars of body plus any truncation indicator.
        assert result.startswith("x" * 100), (
            f"web_fetch(max_response_bytes=100) must keep first 100 chars, got: {result[:120]!r}"
        )
        assert len(result) < 500, (
            "web_fetch(max_response_bytes=100) must truncate, not return full body"
        )

    def test_web_fetch_timeout_param_exists(self):
        """web_fetch() has a timeout parameter with default 15."""
        from corvidae.tools.web import web_fetch

        sig = inspect.signature(web_fetch)
        assert "timeout" in sig.parameters, (
            "web_fetch() must accept a 'timeout' parameter"
        )
        param = sig.parameters["timeout"]
        assert param.default == 15, (
            f"web_fetch() default timeout must be 15, got {param.default!r}"
        )

    def test_web_fetch_with_session_timeout_param_exists(self):
        """web_fetch_with_session() has a timeout parameter with default 15."""
        from corvidae.tools.web import web_fetch_with_session

        sig = inspect.signature(web_fetch_with_session)
        assert "timeout" in sig.parameters, (
            "web_fetch_with_session() must accept a 'timeout' parameter"
        )
        param = sig.parameters["timeout"]
        assert param.default == 15, (
            f"web_fetch_with_session() default timeout must be 15, got {param.default!r}"
        )

    def test_web_constants(self):
        """TRUNCATION_INDICATOR and TIMEOUT_ERROR_TEMPLATE exist in web module."""
        import corvidae.tools.web as web_mod

        assert hasattr(web_mod, "TRUNCATION_INDICATOR"), (
            "corvidae.tools.web must define TRUNCATION_INDICATOR"
        )
        assert isinstance(web_mod.TRUNCATION_INDICATOR, str)

        assert hasattr(web_mod, "TIMEOUT_ERROR_TEMPLATE"), (
            "corvidae.tools.web must define TIMEOUT_ERROR_TEMPLATE"
        )
        assert isinstance(web_mod.TIMEOUT_ERROR_TEMPLATE, str)
        assert "{timeout}" in web_mod.TIMEOUT_ERROR_TEMPLATE, (
            "TIMEOUT_ERROR_TEMPLATE must contain '{timeout}' placeholder"
        )


class TestReadFileMaxSize:
    async def test_read_file_max_size_default(self):
        """read_file rejects files >1MB by default."""
        from corvidae.tools.files import read_file

        sig = inspect.signature(read_file)
        assert "max_size" in sig.parameters, (
            "read_file() must accept a 'max_size' parameter"
        )
        param = sig.parameters["max_size"]
        assert param.default == 1024 * 1024, (
            f"read_file() default max_size must be 1048576, got {param.default!r}"
        )

    async def test_read_file_max_size_override(self, tmp_path):
        """read_file(max_size=100) rejects files larger than 100 bytes."""
        from corvidae.tools.files import read_file

        f = tmp_path / "medium.txt"
        f.write_text("y" * 200)

        result = await read_file(str(f), max_size=100)

        assert "error" in result.lower(), (
            f"read_file(max_size=100) must return an error for a 200-byte file, got: {result!r}"
        )


class TestCoreToolsReadsConfig:
    async def test_core_tools_reads_config(self):
        """CoreToolsPlugin.on_start reads tools config section."""
        from corvidae.tools import CoreToolsPlugin

        plugin = CoreToolsPlugin()

        config = {
            "tools": {
                "shell_timeout": 99,
                "web_fetch_timeout": 5,
                "web_max_response_bytes": 1234,
                "max_file_read_bytes": 512,
            }
        }

        # Patch aiohttp.ClientSession to avoid a real HTTP connection.
        mock_session = MagicMock()
        mock_session.close = AsyncMock()
        with patch("aiohttp.ClientSession", return_value=mock_session):
            await plugin.on_start(config=config)

        assert plugin._shell_timeout == 99, (
            f"Expected _shell_timeout=99 from config, got {plugin._shell_timeout!r}"
        )
        assert plugin._web_fetch_timeout == 5, (
            f"Expected _web_fetch_timeout=5 from config, got {plugin._web_fetch_timeout!r}"
        )
        assert plugin._web_max_response_bytes == 1234, (
            f"Expected _web_max_response_bytes=1234 from config, got {plugin._web_max_response_bytes!r}"
        )
        assert plugin._max_file_read_bytes == 512, (
            f"Expected _max_file_read_bytes=512 from config, got {plugin._max_file_read_bytes!r}"
        )

        # Cleanup.
        await plugin.on_stop()


# ===========================================================================
# Group B — Compaction / conversation
# ===========================================================================


class TestCompactionPlugin:
    def test_compaction_plugin_has_init(self):
        """CompactionPlugin() sets default attrs."""
        from corvidae.compaction import CompactionPlugin

        plugin = CompactionPlugin()

        assert hasattr(plugin, "_compaction_threshold"), (
            "CompactionPlugin must have _compaction_threshold after __init__"
        )
        assert plugin._compaction_threshold == 0.8, (
            f"Default _compaction_threshold must be 0.8, got {plugin._compaction_threshold!r}"
        )

        assert hasattr(plugin, "_compaction_retention"), (
            "CompactionPlugin must have _compaction_retention after __init__"
        )
        assert plugin._compaction_retention == 0.5, (
            f"Default _compaction_retention must be 0.5, got {plugin._compaction_retention!r}"
        )

        assert hasattr(plugin, "_min_messages"), (
            "CompactionPlugin must have _min_messages after __init__"
        )
        assert plugin._min_messages == 5, (
            f"Default _min_messages must be 5, got {plugin._min_messages!r}"
        )

        assert hasattr(plugin, "_chars_per_token"), (
            "CompactionPlugin must have _chars_per_token after __init__"
        )
        assert plugin._chars_per_token == 3.5, (
            f"Default _chars_per_token must be 3.5, got {plugin._chars_per_token!r}"
        )

    async def test_compaction_plugin_reads_config(self):
        """CompactionPlugin.on_start reads agent config."""
        from corvidae.compaction import CompactionPlugin

        plugin = CompactionPlugin()
        config = {
            "agent": {
                "compaction_threshold": 0.9,
                "compaction_retention": 0.4,
                "min_messages_to_compact": 10,
                "chars_per_token": 4.0,
            }
        }

        await plugin.on_start(config=config)

        assert plugin._compaction_threshold == 0.9, (
            f"Expected _compaction_threshold=0.9, got {plugin._compaction_threshold!r}"
        )
        assert plugin._compaction_retention == 0.4, (
            f"Expected _compaction_retention=0.4, got {plugin._compaction_retention!r}"
        )
        assert plugin._min_messages == 10, (
            f"Expected _min_messages=10, got {plugin._min_messages!r}"
        )
        assert plugin._chars_per_token == 4.0, (
            f"Expected _chars_per_token=4.0, got {plugin._chars_per_token!r}"
        )


class TestContextWindowCharsPerToken:
    def test_context_window_chars_per_token_default(self):
        """ContextWindow uses 3.5 chars_per_token by default."""
        from corvidae.context import ContextWindow

        sig = inspect.signature(ContextWindow.__init__)
        assert "chars_per_token" in sig.parameters, (
            "ContextWindow.__init__ must accept 'chars_per_token' parameter"
        )
        param = sig.parameters["chars_per_token"]
        assert param.default == 3.5, (
            f"ContextWindow default chars_per_token must be 3.5, got {param.default!r}"
        )

    def test_context_window_chars_per_token_override(self):
        """ContextWindow(chars_per_token=4.0) uses 4.0 in token_estimate."""
        from corvidae.context import ContextWindow

        conv = ContextWindow("chan1", chars_per_token=4.0)

        assert conv.chars_per_token == 4.0, (
            f"Expected chars_per_token=4.0, got {conv.chars_per_token!r}"
        )

        # Verify it actually affects token_estimate.
        conv.system_prompt = ""
        conv.messages = [{"role": "user", "content": "a" * 40}]

        estimate = conv.token_estimate()
        # 40 chars / 4.0 = 10
        assert estimate == 10, (
            f"token_estimate with chars_per_token=4.0 and 40 chars should be 10, got {estimate}"
        )



# ===========================================================================
# Group C — Tool result truncation
# ===========================================================================


class TestToolTruncationTemplate:
    def test_tool_truncation_template_constant(self):
        """TOOL_TRUNCATION_TEMPLATE exists in corvidae.tool with em-dash."""
        import corvidae.tool as tool_mod

        assert hasattr(tool_mod, "TOOL_TRUNCATION_TEMPLATE"), (
            "corvidae.tool must define TOOL_TRUNCATION_TEMPLATE"
        )
        template = tool_mod.TOOL_TRUNCATION_TEMPLATE
        assert isinstance(template, str), "TOOL_TRUNCATION_TEMPLATE must be a str"
        assert "\u2014" in template, (
            "TOOL_TRUNCATION_TEMPLATE must contain an em-dash (—)"
        )
        assert "{original_len}" in template, (
            "TOOL_TRUNCATION_TEMPLATE must contain '{original_len}' placeholder"
        )


class TestExecuteToolCallMaxResultChars:
    async def test_execute_tool_call_max_result_chars_param(self):
        """execute_tool_call accepts max_result_chars kwarg with default 100_000."""
        from corvidae.tool import execute_tool_call

        sig = inspect.signature(execute_tool_call)
        assert "max_result_chars" in sig.parameters, (
            "execute_tool_call() must accept a 'max_result_chars' keyword parameter"
        )
        param = sig.parameters["max_result_chars"]
        assert param.default == 100_000, (
            f"execute_tool_call() default max_result_chars must be 100000, got {param.default!r}"
        )

    async def test_execute_tool_call_max_result_chars_override(self):
        """execute_tool_call truncates at max_result_chars when provided."""
        from corvidae.tool import execute_tool_call

        async def big_tool():
            return "z" * 500

        result = await execute_tool_call(
            big_tool,
            {},
            tool_call_id="call1",
            max_result_chars=50,
        )

        # Must start with 50 'z' chars and then be truncated.
        assert result.startswith("z" * 50), (
            f"execute_tool_call with max_result_chars=50 must keep first 50 chars"
        )
        assert len(result) < 500, (
            "execute_tool_call with max_result_chars=50 must truncate a 500-char result"
        )


class TestRunAgentLoopMaxResultCharsParam:
    def test_agent_loop_max_result_chars_param(self):
        """run_agent_loop accepts max_result_chars kwarg."""
        from corvidae.agent_loop import run_agent_loop

        sig = inspect.signature(run_agent_loop)
        assert "max_result_chars" in sig.parameters, (
            "run_agent_loop() must accept a 'max_result_chars' keyword parameter"
        )


# ===========================================================================
# Group D — Task / IRC
# ===========================================================================


class TestTaskQueueCompletedBuffer:
    def test_task_queue_completed_buffer_default(self):
        """TaskQueue uses maxlen=100 for the completed deque by default."""
        from corvidae.task import TaskQueue

        queue = TaskQueue()
        assert queue.completed.maxlen == 100, (
            f"TaskQueue().completed.maxlen must be 100, got {queue.completed.maxlen!r}"
        )

    def test_task_queue_completed_buffer_override(self):
        """TaskQueue(completed_buffer=50) uses maxlen=50."""
        from corvidae.task import TaskQueue

        sig = inspect.signature(TaskQueue.__init__)
        assert "completed_buffer" in sig.parameters, (
            "TaskQueue.__init__ must accept 'completed_buffer' parameter"
        )

        queue = TaskQueue(completed_buffer=50)
        assert queue.completed.maxlen == 50, (
            f"TaskQueue(completed_buffer=50).completed.maxlen must be 50, got {queue.completed.maxlen!r}"
        )


class TestTaskConstants:
    def test_task_failure_template_constant(self):
        """TASK_FAILURE_TEMPLATE exists in corvidae.task."""
        import corvidae.task as task_mod

        assert hasattr(task_mod, "TASK_FAILURE_TEMPLATE"), (
            "corvidae.task must define TASK_FAILURE_TEMPLATE"
        )
        template = task_mod.TASK_FAILURE_TEMPLATE
        assert isinstance(template, str), "TASK_FAILURE_TEMPLATE must be a str"
        assert "{task_id}" in template, (
            "TASK_FAILURE_TEMPLATE must contain '{task_id}' placeholder"
        )
        assert "{error}" in template, (
            "TASK_FAILURE_TEMPLATE must contain '{error}' placeholder"
        )

    def test_status_history_count_constant(self):
        """STATUS_HISTORY_COUNT exists in corvidae.task with value 3."""
        import corvidae.task as task_mod

        assert hasattr(task_mod, "STATUS_HISTORY_COUNT"), (
            "corvidae.task must define STATUS_HISTORY_COUNT"
        )
        assert task_mod.STATUS_HISTORY_COUNT == 3, (
            f"STATUS_HISTORY_COUNT must be 3, got {task_mod.STATUS_HISTORY_COUNT!r}"
        )


class TestIRCMessageChunkSizeFromConfig:
    async def test_irc_message_chunk_size_from_config(self):
        """IRCPlugin reads irc.message_chunk_size in on_start."""
        # Import IRCPlugin here so corvidae.channels.irc is cached in sys.modules
        # before any patching; this ensures patch("corvidae.channels.irc.get_dependency")
        # targets the same module instance that IRCPlugin closes over.
        from corvidae.channels.irc import IRCPlugin
        from corvidae.channel import ChannelRegistry

        pm_mock = MagicMock()
        registry_mock = MagicMock(spec=ChannelRegistry)

        with patch("corvidae.channels.irc.get_dependency", return_value=registry_mock), \
             patch("asyncio.create_task"):
            plugin = IRCPlugin(pm_mock)
            await plugin.on_start(config={
                "irc": {
                    "host": "localhost",
                    "nick": "bot",
                    "message_chunk_size": 200,
                }
            })

        assert hasattr(plugin, "_message_chunk_size"), (
            "IRCPlugin must store _message_chunk_size after on_start"
        )
        assert plugin._message_chunk_size == 200, (
            f"Expected _message_chunk_size=200 from config, got {plugin._message_chunk_size!r}"
        )


class TestAgentMaxToolResultChars:
    async def test_agent_plugin_reads_max_tool_result_chars(self):
        """Agent._start_plugin reads max_result_chars from ToolCollectionPlugin.

        After Part 4, Agent borrows _max_tool_result_chars from
        ToolCollectionPlugin rather than reading agent.max_tool_result_chars directly.
        """
        from corvidae.agent import Agent
        from corvidae.channel import ChannelRegistry
        from corvidae.hooks import create_plugin_manager
        from corvidae.tool_collection import ToolCollectionPlugin
        from corvidae.tool import ToolRegistry

        pm = create_plugin_manager()
        registry = ChannelRegistry({
            "system_prompt": "test",
            "max_context_tokens": 8000,
            "keep_thinking_in_history": False,
        })
        pm.register(registry, name="registry")

        from corvidae.llm_plugin import LLMPlugin
        mock_llm = LLMPlugin(pm)
        mock_llm.main_client = MagicMock()
        pm.register(mock_llm, name="llm")

        # Configure ToolCollectionPlugin with the custom max_result_chars value
        tools_plugin = ToolCollectionPlugin(pm)
        tools_plugin.registry = ToolRegistry()
        tools_plugin.max_result_chars = 9999
        pm.register(tools_plugin, name="tools")

        plugin = Agent(pm)
        pm.register(plugin, name="agent")

        config = {
            "llm": {
                "main": {
                    "base_url": "http://fake",
                    "model": "test",
                }
            },
        }

        await plugin._start_plugin(config)

        assert hasattr(plugin, "_max_tool_result_chars"), (
            "Agent must store _max_tool_result_chars after _start_plugin"
        )
        assert plugin._max_tool_result_chars == 9999, (
            f"Expected _max_tool_result_chars=9999 from ToolCollectionPlugin, got {plugin._max_tool_result_chars!r}"
        )


class TestSubagentPluginMaxToolResultChars:
    async def test_subagent_plugin_does_not_read_max_tool_result_chars_from_config(self):
        """SubagentPlugin.on_start no longer reads max_tool_result_chars from config.

        After consolidation (Item 3), SubagentPlugin reads this value from
        Agent at _launch time instead of independently from config.
        """
        from corvidae.tools.subagent import SubagentPlugin

        pm_mock = MagicMock()
        plugin = SubagentPlugin(pm_mock)

        config = {
            "agent": {
                "max_tool_result_chars": 8888,
            },
            "llm": {
                "main": {
                    "base_url": "http://fake",
                    "model": "test",
                }
            },
        }

        await plugin.on_start(config=config)

        assert not hasattr(plugin, "_max_tool_result_chars"), (
            "SubagentPlugin should not store _max_tool_result_chars after on_start; "
            "it now reads from Agent at _launch time"
        )


class TestTaskPluginCompletedBuffer:
    async def test_task_plugin_reads_completed_task_buffer(self):
        """TaskPlugin.on_start reads daemon.completed_task_buffer from config."""
        from corvidae.task import TaskPlugin

        pm_mock = MagicMock()
        plugin = TaskPlugin(pm_mock)

        with patch("asyncio.create_task"):
            await plugin.on_start(config={"daemon": {"completed_task_buffer": 42}})

        assert plugin.task_queue is not None, "TaskPlugin.task_queue must be set after on_start"
        assert plugin.task_queue.completed.maxlen == 42, (
            f"Expected task_queue.completed.maxlen=42 from config, got {plugin.task_queue.completed.maxlen!r}"
        )


# ===========================================================================
# Tier 2 constants
# ===========================================================================


class TestAgentConstants:
    def test_agent_constants(self):
        """DEFAULT_LLM_ERROR_MESSAGE and MAX_TURNS_FALLBACK_MESSAGE exist in agent.py."""
        import corvidae.agent as agent_mod

        assert hasattr(agent_mod, "DEFAULT_LLM_ERROR_MESSAGE"), (
            "corvidae.agent must define DEFAULT_LLM_ERROR_MESSAGE"
        )
        assert isinstance(agent_mod.DEFAULT_LLM_ERROR_MESSAGE, str)
        assert len(agent_mod.DEFAULT_LLM_ERROR_MESSAGE) > 0

        assert hasattr(agent_mod, "MAX_TURNS_FALLBACK_MESSAGE"), (
            "corvidae.agent must define MAX_TURNS_FALLBACK_MESSAGE"
        )
        assert isinstance(agent_mod.MAX_TURNS_FALLBACK_MESSAGE, str)
        assert len(agent_mod.MAX_TURNS_FALLBACK_MESSAGE) > 0


class TestAgentLoopConstants:
    def test_agent_loop_constants(self):
        """LOG_TRUNCATION_LENGTH and MAX_ROUNDS_REACHED_MESSAGE exist in agent_loop.py."""
        import corvidae.agent_loop as agent_loop_mod

        assert hasattr(agent_loop_mod, "LOG_TRUNCATION_LENGTH"), (
            "corvidae.agent_loop must define LOG_TRUNCATION_LENGTH"
        )
        assert isinstance(agent_loop_mod.LOG_TRUNCATION_LENGTH, int)
        assert agent_loop_mod.LOG_TRUNCATION_LENGTH == 200, (
            f"LOG_TRUNCATION_LENGTH must be 200, got {agent_loop_mod.LOG_TRUNCATION_LENGTH!r}"
        )

        assert hasattr(agent_loop_mod, "MAX_ROUNDS_REACHED_MESSAGE"), (
            "corvidae.agent_loop must define MAX_ROUNDS_REACHED_MESSAGE"
        )
        assert isinstance(agent_loop_mod.MAX_ROUNDS_REACHED_MESSAGE, str)
        assert len(agent_loop_mod.MAX_ROUNDS_REACHED_MESSAGE) > 0
