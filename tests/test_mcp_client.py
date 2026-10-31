"""Tests for corvidae.mcp_client — McpClientPlugin.

All tests are expected to FAIL until corvidae/mcp_client.py is implemented.

Design assumptions (from plans/mcp-client-plugin.md):
- McpClientPlugin.on_start(config) is an async hookimpl.
- McpClientPlugin.register_tools(tool_registry) is a sync hookimpl.
- McpClientPlugin.on_stop() is an async hookimpl.
- Tool naming: "{prefix}__{mcp_tool_name}" when prefix is set, else just "{mcp_tool_name}".
- prefix defaults to the server name if tool_prefix not specified in config.
- Schema translation: wraps mcp_tool.input_schema in the OpenAI function-call envelope.
- Unsafe keys ($schema, $id, $comment, $defs, definitions) are stripped from input_schema.
- Tool call success: calls session.call_tool, joins TextContent blocks with newlines.
- Tool call timeout: returns error string including tool name, server name, and timeout.
- Tool call exception: returns error string including original exception message.
- result.is_error: prepends "Error: " to the joined text.
- Duplicate tool names: first registration wins; warning logged, duplicate skipped.
- Non-text content blocks: dropped (logged at DEBUG), text blocks only are joined.
- on_stop: signals every server's owner task to close its session.
- on_start with no config: returns early, no sessions connected.
- Empty result: returns "(no output)" when no text blocks are present.
"""

import asyncio
import logging
from contextlib import AsyncExitStack
from fake_clock import clock_pm
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# The import under test — will raise ImportError until the file exists.
import corvidae.mcp_client as mcp_client_mod
from corvidae.mcp_client import McpClientPlugin, _McpServerState, _ServerOwner, _call_mcp_tool, _make_tool, _mcp_tool_to_schema


# ---------------------------------------------------------------------------
# Helpers for building mock MCP objects
# ---------------------------------------------------------------------------


def _make_mcp_tool(name: str, description: str = "", input_schema: dict | None = None):
    """Build a minimal mock MCP tool definition."""
    tool = MagicMock()
    tool.name = name
    tool.description = description
    tool.input_schema = input_schema or {"type": "object", "properties": {}}
    return tool


def _make_text_block(text: str):
    """Build a mock MCP TextContent block."""
    block = MagicMock()
    block.text = text
    return block


def _make_image_block():
    """Build a mock MCP ImageContent block (no .text attribute)."""
    block = MagicMock(spec=[])  # no attributes by default
    return block


def _make_tool_result(content: list, is_error: bool = False):
    """Build a mock MCP call_tool result."""
    result = MagicMock()
    result.content = content
    result.is_error = is_error
    return result


def _make_server_state(
    name: str = "myserver",
    session: object = None,
    prefix: str = "myserver",
    timeout_seconds: float = 30.0,
    mcp_tools: list | None = None,
) -> _McpServerState:
    """Build a _McpServerState for use in unit tests."""
    return _McpServerState(
        name=name,
        session=session or MagicMock(),
        prefix=prefix,
        timeout_seconds=timeout_seconds,
        mcp_tools=mcp_tools or [],
    )


# ---------------------------------------------------------------------------
# TestOnStartNoConfig
# ---------------------------------------------------------------------------


class TestOnStartNoConfig:
    """on_start returns early when no MCP servers are configured."""

    async def test_no_mcp_key_in_config(self):
        """Config with no 'mcp' key: no servers connected, _cached_tools empty."""
        plugin = McpClientPlugin(None)
        await plugin.on_start(config={})
        assert plugin._servers == []
        assert plugin._cached_tools == []
        assert plugin._owners == {}

    async def test_empty_servers_dict(self):
        """Config with 'mcp.servers: {}': no servers connected, _cached_tools empty."""
        plugin = McpClientPlugin(None)
        await plugin.on_start(config={"mcp": {"servers": {}}})
        assert plugin._servers == []
        assert plugin._cached_tools == []
        assert plugin._owners == {}

    async def test_missing_servers_key(self):
        """Config with 'mcp: {}': no servers connected, _cached_tools empty."""
        plugin = McpClientPlugin(None)
        await plugin.on_start(config={"mcp": {}})
        assert plugin._servers == []
        assert plugin._cached_tools == []
        assert plugin._owners == {}


# ---------------------------------------------------------------------------
# TestOnStartWithStdioServer
# ---------------------------------------------------------------------------


class TestOnStartWithStdioServer:
    """on_start connects, initializes session, fetches tools."""

    async def test_connects_stdio_server_and_fetches_tools(self):
        """A configured stdio server results in a populated _servers list."""
        plugin = McpClientPlugin(None)

        # Build mock objects for the MCP plumbing.
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        list_tools_result = MagicMock()
        list_tools_result.tools = [_make_mcp_tool("read_file")]
        mock_session.list_tools = AsyncMock(return_value=list_tools_result)

        # stdio_client returns an (read, write) async context manager pair.
        mock_read = MagicMock()
        mock_write = MagicMock()

        # ClientSession is also an async context manager.
        mock_client_session_cm = AsyncMock()
        mock_client_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_client_session_cm.__aexit__ = AsyncMock(return_value=False)

        mock_stdio_cm = AsyncMock()
        mock_stdio_cm.__aenter__ = AsyncMock(return_value=(mock_read, mock_write))
        mock_stdio_cm.__aexit__ = AsyncMock(return_value=False)

        config = {
            "mcp": {
                "servers": {
                    "fs": {
                        "transport": "stdio",
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                    }
                }
            }
        }

        await plugin.on_init(pm=clock_pm(), config=config)
        with (
            patch("mcp.client.stdio.stdio_client", return_value=mock_stdio_cm),
            patch("mcp.client.stdio.StdioServerParameters"),
            patch("mcp.ClientSession", return_value=mock_client_session_cm),
        ):
            await plugin.on_start(config=config)

        assert len(plugin._servers) == 1
        assert plugin._servers[0].name == "fs"

    async def test_session_initialize_called(self):
        """session.initialize() is called after entering the session context."""
        plugin = McpClientPlugin(None)

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        list_tools_result = MagicMock()
        list_tools_result.tools = []
        mock_session.list_tools = AsyncMock(return_value=list_tools_result)

        mock_stdio_cm = AsyncMock()
        mock_stdio_cm.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
        mock_stdio_cm.__aexit__ = AsyncMock(return_value=False)

        mock_client_session_cm = AsyncMock()
        mock_client_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_client_session_cm.__aexit__ = AsyncMock(return_value=False)

        config = {
            "mcp": {
                "servers": {
                    "git": {
                        "transport": "stdio",
                        "command": "uvx",
                        "args": ["mcp-server-git"],
                    }
                }
            }
        }

        await plugin.on_init(pm=clock_pm(), config=config)
        with (
            patch("mcp.client.stdio.stdio_client", return_value=mock_stdio_cm),
            patch("mcp.client.stdio.StdioServerParameters"),
            patch("mcp.ClientSession", return_value=mock_client_session_cm),
        ):
            await plugin.on_start(config=config)

        mock_session.initialize.assert_called_once()

    async def test_cached_tools_populated(self):
        """_cached_tools is populated from list_tools result."""
        plugin = McpClientPlugin(None)

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        list_tools_result = MagicMock()
        list_tools_result.tools = [
            _make_mcp_tool("read_file", "Read a file"),
            _make_mcp_tool("write_file", "Write a file"),
        ]
        mock_session.list_tools = AsyncMock(return_value=list_tools_result)

        mock_stdio_cm = AsyncMock()
        mock_stdio_cm.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
        mock_stdio_cm.__aexit__ = AsyncMock(return_value=False)

        mock_client_session_cm = AsyncMock()
        mock_client_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_client_session_cm.__aexit__ = AsyncMock(return_value=False)

        config = {
            "mcp": {
                "servers": {
                    "fs": {
                        "transport": "stdio",
                        "command": "npx",
                        "args": [],
                    }
                }
            }
        }

        await plugin.on_init(pm=clock_pm(), config=config)
        with (
            patch("mcp.client.stdio.stdio_client", return_value=mock_stdio_cm),
            patch("mcp.client.stdio.StdioServerParameters"),
            patch("mcp.ClientSession", return_value=mock_client_session_cm),
        ):
            await plugin.on_start(config=config)

        assert len(plugin._cached_tools) == 2

    async def test_failed_server_skipped(self):
        """A server that raises during connection is skipped; daemon continues."""
        plugin = McpClientPlugin(None)

        mock_stdio_cm = AsyncMock()
        mock_stdio_cm.__aenter__ = AsyncMock(side_effect=ConnectionRefusedError("no server"))
        mock_stdio_cm.__aexit__ = AsyncMock(return_value=False)

        config = {
            "mcp": {
                "servers": {
                    "broken": {
                        "transport": "stdio",
                        "command": "nonexistent",
                        "args": [],
                    }
                }
            }
        }

        with (
            patch("mcp.client.stdio.stdio_client", return_value=mock_stdio_cm),
            patch("mcp.client.stdio.StdioServerParameters"),
        ):
            # Should not raise.
            await plugin.on_start(config=config)

        assert plugin._servers == []


# ---------------------------------------------------------------------------
# TestOnStartWithSseServer
# ---------------------------------------------------------------------------


class TestOnStartWithSseServer:
    """on_start connects via SSE transport when url config is given."""

    async def test_connects_sse_server_and_fetches_tools(self):
        """A configured SSE server results in a populated _servers list."""
        plugin = McpClientPlugin(None)

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        list_tools_result = MagicMock()
        list_tools_result.tools = [_make_mcp_tool("search")]
        mock_session.list_tools = AsyncMock(return_value=list_tools_result)

        mock_read = MagicMock()
        mock_write = MagicMock()

        mock_client_session_cm = AsyncMock()
        mock_client_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_client_session_cm.__aexit__ = AsyncMock(return_value=False)

        mock_sse_cm = AsyncMock()
        mock_sse_cm.__aenter__ = AsyncMock(return_value=(mock_read, mock_write))
        mock_sse_cm.__aexit__ = AsyncMock(return_value=False)

        config = {
            "mcp": {
                "servers": {
                    "remote_api": {
                        "transport": "sse",
                        "url": "http://localhost:8001/sse",
                    }
                }
            }
        }

        await plugin.on_init(pm=clock_pm(), config=config)
        with (
            patch("mcp.client.sse.sse_client", return_value=mock_sse_cm),
            patch("mcp.ClientSession", return_value=mock_client_session_cm),
        ):
            await plugin.on_start(config=config)

        assert len(plugin._servers) == 1
        assert plugin._servers[0].name == "remote_api"
        assert len(plugin._cached_tools) == 1


# ---------------------------------------------------------------------------
# TestOnStartUnknownTransport
# ---------------------------------------------------------------------------


class TestOnStartUnknownTransport:
    """An unknown transport type logs a warning and skips the server."""

    async def test_unknown_transport_skipped_with_warning(self, caplog):
        """A server with an unknown transport type is skipped; daemon continues."""
        plugin = McpClientPlugin(None)

        config = {
            "mcp": {
                "servers": {
                    "bad_server": {
                        "transport": "grpc",
                        "url": "http://localhost:9000",
                    }
                }
            }
        }

        await plugin.on_init(pm=clock_pm(), config=config)
        with caplog.at_level(logging.WARNING, logger="corvidae.mcp_client"):
            # Should not raise.
            await plugin.on_start(config=config)

        assert plugin._servers == []
        assert any(record.levelno == logging.WARNING for record in caplog.records)


# ---------------------------------------------------------------------------
# TestRegisterTools
# ---------------------------------------------------------------------------


class TestRegisterTools:
    """register_tools extends tool_registry with cached tools."""

    def test_extends_registry_with_cached_tools(self):
        """Cached tools are appended to the provided tool_registry list."""
        from corvidae.tool import Tool

        plugin = McpClientPlugin(None)

        # Build a fake cached tool.
        async def dummy_fn(**kwargs):
            return "ok"

        fake_tool = Tool(name="server__read", fn=dummy_fn, schema={"type": "function", "function": {}})
        plugin._cached_tools = [fake_tool]

        registry = []
        plugin.register_tools(tool_registry=registry)

        assert len(registry) == 1
        assert registry[0] is fake_tool

    def test_empty_cached_tools_extends_nothing(self):
        """When no servers are configured, register_tools is a no-op."""
        plugin = McpClientPlugin(None)
        registry = []
        plugin.register_tools(tool_registry=registry)
        assert registry == []

    def test_does_not_replace_existing_registry_items(self):
        """register_tools extends (not replaces) an existing registry."""
        from corvidae.tool import Tool

        plugin = McpClientPlugin(None)

        async def existing_fn(**kwargs):
            return "existing"

        async def mcp_fn(**kwargs):
            return "mcp"

        existing_tool = Tool(name="existing", fn=existing_fn, schema={})
        mcp_tool = Tool(name="mcp__thing", fn=mcp_fn, schema={})
        plugin._cached_tools = [mcp_tool]

        registry = [existing_tool]
        plugin.register_tools(tool_registry=registry)

        assert len(registry) == 2
        assert registry[0] is existing_tool
        assert registry[1] is mcp_tool


# ---------------------------------------------------------------------------
# TestOnStop
# ---------------------------------------------------------------------------


class TestOnStop:
    """on_stop signals each server's owner task and clears plugin state."""

    @staticmethod
    def _owner(name: str, body) -> "_ServerOwner":
        """An owner whose task runs ``body(owner)`` once stop is signalled."""
        owner = _ServerOwner(name=name)

        async def run() -> None:
            await owner.stop.wait()
            await body(owner)

        owner.task = asyncio.create_task(run())
        return owner

    async def test_on_stop_signals_owner_and_awaits_it(self):
        """on_stop sets the owner's stop event and waits for its task."""
        plugin = McpClientPlugin(None)
        closed = []

        async def body(owner):
            closed.append(owner.name)

        plugin._owners["a"] = self._owner("a", body)

        await plugin.on_stop()

        assert closed == ["a"]
        assert plugin._owners == {}

    async def test_on_stop_clears_state(self):
        """on_stop empties _owners, _servers and _cached_tools."""
        plugin = McpClientPlugin(None)

        async def body(owner):
            return None

        plugin._owners["a"] = self._owner("a", body)
        plugin._servers.append(_make_server_state(name="a"))
        plugin._cached_tools = [object()]

        await plugin.on_stop()

        assert plugin._owners == {}
        assert plugin._servers == []
        assert plugin._cached_tools == []

    async def test_on_stop_no_owners_is_safe(self):
        """on_stop is a no-op when no server connected."""
        plugin = McpClientPlugin(None)
        assert plugin._owners == {}
        # Must not raise.
        await plugin.on_stop()

    async def test_on_stop_owner_task_raises_logs_and_clears(self, caplog):
        """An owner task that dies with an exception is logged by name and
        type; on_stop still completes and clears state."""
        plugin = McpClientPlugin(None)

        async def body(owner):
            raise RuntimeError("transport closed unexpectedly")

        plugin._owners["srv"] = self._owner("srv", body)

        with caplog.at_level(logging.WARNING, logger="corvidae.mcp_client"):
            # Must NOT raise: the exception is logged, not swallowed.
            await plugin.on_stop()

        assert plugin._owners == {}
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 1
        assert "srv" in warnings[0].getMessage()
        assert "RuntimeError" in warnings[0].getMessage()
        assert warnings[0].exc_info is not None


# ---------------------------------------------------------------------------
# TestToolNaming
# ---------------------------------------------------------------------------


class TestToolNaming:
    """Tool name = prefix + '__' + mcp_tool_name when prefix is set."""

    def test_prefix_and_tool_name_joined_with_double_underscore(self):
        """Tool name is '{prefix}__{mcp_tool_name}'."""
        server = _make_server_state(name="fs", prefix="fs")
        mcp_tool = _make_mcp_tool("read_file")
        tool = _make_tool(server, mcp_tool)
        assert tool.name == "fs__read_file"

    def test_prefix_defaults_to_server_name(self):
        """When tool_prefix is not set, prefix equals the server name."""
        # This is enforced in _connect_server; we verify _make_tool behavior directly.
        server = _make_server_state(name="git", prefix="git")
        mcp_tool = _make_mcp_tool("clone")
        tool = _make_tool(server, mcp_tool)
        assert tool.name == "git__clone"

    def test_custom_prefix(self):
        """Explicit tool_prefix overrides server name in tool naming."""
        server = _make_server_state(name="filesystem", prefix="fs")
        mcp_tool = _make_mcp_tool("list_directory")
        tool = _make_tool(server, mcp_tool)
        assert tool.name == "fs__list_directory"


# ---------------------------------------------------------------------------
# TestToolNamingEmptyPrefix
# ---------------------------------------------------------------------------


class TestToolNamingEmptyPrefix:
    """When prefix is empty string, tool name is just the MCP tool name."""

    def test_empty_prefix_no_separator(self):
        """tool_prefix: '' produces tool name without prefix or separator."""
        server = _make_server_state(name="myserver", prefix="")
        mcp_tool = _make_mcp_tool("query")
        tool = _make_tool(server, mcp_tool)
        assert tool.name == "query"

    def test_empty_prefix_no_double_underscore(self):
        """Empty prefix must not produce a leading '__' in the tool name."""
        server = _make_server_state(name="myserver", prefix="")
        mcp_tool = _make_mcp_tool("search")
        tool = _make_tool(server, mcp_tool)
        assert not tool.name.startswith("__")


# ---------------------------------------------------------------------------
# TestSchemaTranslation
# ---------------------------------------------------------------------------


class TestSchemaTranslation:
    """MCP input_schema is wrapped in the OpenAI function-calling envelope."""

    def test_schema_wrapped_in_openai_envelope(self):
        """_mcp_tool_to_schema returns a dict with 'type': 'function' and 'function' key."""
        mcp_tool = _make_mcp_tool(
            "read_file",
            description="Read a file from disk",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        )
        schema = _mcp_tool_to_schema("myserver__read_file", mcp_tool)
        assert schema["type"] == "function"
        assert "function" in schema
        assert schema["function"]["name"] == "myserver__read_file"
        assert schema["function"]["description"] == "Read a file from disk"
        assert "parameters" in schema["function"]

    def test_parameters_match_input_schema(self):
        """parameters in the OpenAI envelope equals the MCP input_schema (after sanitization)."""
        input_schema = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }
        mcp_tool = _make_mcp_tool("read_file", input_schema=input_schema)
        schema = _mcp_tool_to_schema("server__read_file", mcp_tool)
        assert schema["function"]["parameters"]["type"] == "object"
        assert "path" in schema["function"]["parameters"]["properties"]

    def test_null_input_schema_produces_empty_object_schema(self):
        """When input_schema is None, parameters defaults to empty object schema."""
        mcp_tool = _make_mcp_tool("ping")
        mcp_tool.input_schema = None
        schema = _mcp_tool_to_schema("server__ping", mcp_tool)
        params = schema["function"]["parameters"]
        assert params.get("type") == "object"

    def test_empty_description_allowed(self):
        """An empty MCP description produces an empty string in the schema (not None)."""
        mcp_tool = _make_mcp_tool("ping", description="")
        schema = _mcp_tool_to_schema("server__ping", mcp_tool)
        assert schema["function"]["description"] == ""


# ---------------------------------------------------------------------------
# TestSchemaSanitization
# ---------------------------------------------------------------------------


class TestSchemaSanitization:
    """Unsafe keys are stripped from the top level of input_schema."""

    def test_dollar_schema_stripped(self):
        """'$schema' key is removed from the parameters."""
        mcp_tool = _make_mcp_tool(
            "read_file",
            input_schema={
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {},
            },
        )
        schema = _mcp_tool_to_schema("server__read_file", mcp_tool)
        assert "$schema" not in schema["function"]["parameters"]

    def test_dollar_id_stripped(self):
        """'$id' key is removed from the parameters."""
        mcp_tool = _make_mcp_tool(
            "tool",
            input_schema={"$id": "some-id", "type": "object", "properties": {}},
        )
        schema = _mcp_tool_to_schema("server__tool", mcp_tool)
        assert "$id" not in schema["function"]["parameters"]

    def test_dollar_comment_stripped(self):
        """'$comment' key is removed from the parameters."""
        mcp_tool = _make_mcp_tool(
            "tool",
            input_schema={"$comment": "a comment", "type": "object", "properties": {}},
        )
        schema = _mcp_tool_to_schema("server__tool", mcp_tool)
        assert "$comment" not in schema["function"]["parameters"]

    def test_dollar_defs_stripped(self):
        """'$defs' key is removed from the parameters."""
        mcp_tool = _make_mcp_tool(
            "tool",
            input_schema={
                "$defs": {"MyType": {"type": "string"}},
                "type": "object",
                "properties": {},
            },
        )
        schema = _mcp_tool_to_schema("server__tool", mcp_tool)
        assert "$defs" not in schema["function"]["parameters"]

    def test_definitions_stripped(self):
        """'definitions' key is removed from the parameters."""
        mcp_tool = _make_mcp_tool(
            "tool",
            input_schema={
                "definitions": {"MyType": {"type": "string"}},
                "type": "object",
                "properties": {},
            },
        )
        schema = _mcp_tool_to_schema("server__tool", mcp_tool)
        assert "definitions" not in schema["function"]["parameters"]

    def test_safe_keys_preserved(self):
        """Safe keys like 'type', 'properties', 'required' are not stripped."""
        input_schema = {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        }
        mcp_tool = _make_mcp_tool("tool", input_schema=input_schema)
        schema = _mcp_tool_to_schema("server__tool", mcp_tool)
        params = schema["function"]["parameters"]
        assert params["type"] == "object"
        assert "properties" in params
        assert "required" in params

    def test_multiple_unsafe_keys_all_stripped(self):
        """Multiple unsafe keys are all stripped in a single call."""
        mcp_tool = _make_mcp_tool(
            "tool",
            input_schema={
                "$schema": "...",
                "$id": "...",
                "$comment": "...",
                "type": "object",
                "properties": {},
            },
        )
        schema = _mcp_tool_to_schema("server__tool", mcp_tool)
        params = schema["function"]["parameters"]
        assert "$schema" not in params
        assert "$id" not in params
        assert "$comment" not in params


# ---------------------------------------------------------------------------
# TestToolExecutionSuccess
# ---------------------------------------------------------------------------


class TestToolExecutionSuccess:
    """Successful tool call returns joined text content."""

    async def test_single_text_block_returned(self):
        """Single text block: result is the block's text."""
        mock_session = AsyncMock()
        text_block = _make_text_block("file contents here")
        tool_result = _make_tool_result([text_block])
        mock_session.call_tool = AsyncMock(return_value=tool_result)

        result = await _call_mcp_tool(mock_session, "read_file", {"path": "/tmp/x"}, 30.0, "myserver")
        assert result == "file contents here"

    async def test_multiple_text_blocks_joined_with_newlines(self):
        """Multiple text blocks are joined with newlines."""
        mock_session = AsyncMock()
        blocks = [_make_text_block("line1"), _make_text_block("line2")]
        tool_result = _make_tool_result(blocks)
        mock_session.call_tool = AsyncMock(return_value=tool_result)

        result = await _call_mcp_tool(mock_session, "read_file", {}, 30.0, "myserver")
        assert result == "line1\nline2"

    async def test_empty_content_returns_no_output(self):
        """Empty content list returns '(no output)'."""
        mock_session = AsyncMock()
        tool_result = _make_tool_result([])
        mock_session.call_tool = AsyncMock(return_value=tool_result)

        result = await _call_mcp_tool(mock_session, "ping", {}, 30.0, "myserver")
        assert result == "(no output)"

    async def test_call_tool_receives_correct_arguments(self):
        """session.call_tool is called with the original tool name and kwargs."""
        mock_session = AsyncMock()
        tool_result = _make_tool_result([_make_text_block("ok")])
        mock_session.call_tool = AsyncMock(return_value=tool_result)

        await _call_mcp_tool(mock_session, "read_file", {"path": "/etc/hosts"}, 30.0, "myserver")

        mock_session.call_tool.assert_called_once_with("read_file", arguments={"path": "/etc/hosts"})


# ---------------------------------------------------------------------------
# TestToolExecutionTimeout
# ---------------------------------------------------------------------------


class TestToolExecutionTimeout:
    """Tool call timeout returns an error string (does not raise)."""

    async def test_timeout_returns_error_string(self):
        """asyncio.TimeoutError produces an error string mentioning the tool and server."""
        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(side_effect=asyncio.TimeoutError)

        result = await _call_mcp_tool(mock_session, "slow_tool", {}, 5.0, "myserver")

        assert "slow_tool" in result
        assert "myserver" in result
        # The design says: "timed out after {timeout}s"
        assert "5" in result or "timed out" in result.lower() or "timeout" in result.lower()

    async def test_timeout_does_not_raise(self):
        """TimeoutError is caught; function returns a string rather than raising."""
        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(side_effect=asyncio.TimeoutError)

        # Must not raise.
        result = await _call_mcp_tool(mock_session, "slow_tool", {}, 30.0, "myserver")
        assert isinstance(result, str)

    async def test_timeout_uses_wait_for(self):
        """_call_mcp_tool wraps call_tool in asyncio.wait_for."""
        mock_session = AsyncMock()
        tool_result = _make_tool_result([_make_text_block("ok")])
        mock_session.call_tool = AsyncMock(return_value=tool_result)

        with patch("asyncio.wait_for", wraps=asyncio.wait_for) as mock_wait_for:
            await _call_mcp_tool(mock_session, "tool", {}, 30.0, "myserver")

        mock_wait_for.assert_called_once()


# ---------------------------------------------------------------------------
# TestToolExecutionException
# ---------------------------------------------------------------------------


class TestToolExecutionException:
    """Tool call exception returns an error string (does not raise)."""

    async def test_exception_returns_error_string(self):
        """Generic exception produces an error string mentioning the exception message."""
        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(side_effect=RuntimeError("protocol error"))

        result = await _call_mcp_tool(mock_session, "read_file", {}, 30.0, "myserver")

        assert "protocol error" in result

    async def test_exception_does_not_raise(self):
        """Exception is caught; function returns a string rather than raising."""
        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(side_effect=ValueError("bad input"))

        result = await _call_mcp_tool(mock_session, "tool", {}, 30.0, "myserver")
        assert isinstance(result, str)

    async def test_exception_string_contains_tool_name(self):
        """Error string mentions the original tool name."""
        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(side_effect=Exception("oops"))

        result = await _call_mcp_tool(mock_session, "my_tool", {}, 30.0, "myserver")

        assert "my_tool" in result


# ---------------------------------------------------------------------------
# TestToolExecutionIsError
# ---------------------------------------------------------------------------


class TestToolExecutionIsError:
    """result.is_error causes the result to be prefixed with 'Error: '."""

    async def test_is_error_prefixes_with_error(self):
        """When result.is_error is True, output is prefixed with 'Error: '."""
        mock_session = AsyncMock()
        blocks = [_make_text_block("something went wrong")]
        tool_result = _make_tool_result(blocks, is_error=True)
        mock_session.call_tool = AsyncMock(return_value=tool_result)

        result = await _call_mcp_tool(mock_session, "tool", {}, 30.0, "myserver")

        assert result.startswith("Error: ")
        assert "something went wrong" in result

    async def test_is_error_false_no_prefix(self):
        """When result.is_error is False, output has no 'Error: ' prefix."""
        mock_session = AsyncMock()
        blocks = [_make_text_block("all good")]
        tool_result = _make_tool_result(blocks, is_error=False)
        mock_session.call_tool = AsyncMock(return_value=tool_result)

        result = await _call_mcp_tool(mock_session, "tool", {}, 30.0, "myserver")

        assert not result.startswith("Error: ")
        assert result == "all good"


# ---------------------------------------------------------------------------
# TestToolNameCollision
# ---------------------------------------------------------------------------


class TestToolNameCollision:
    """First tool registration wins; duplicates are skipped with a warning."""

    def test_first_tool_wins_on_collision(self):
        """When two servers expose the same tool name, only the first is kept."""
        plugin = McpClientPlugin(None)

        mcp_tool_a = _make_mcp_tool("read_file", "Server A read_file")
        mcp_tool_b = _make_mcp_tool("read_file", "Server B read_file")

        server_a = _make_server_state(name="server_a", prefix="")
        server_a.mcp_tools = [mcp_tool_a]

        server_b = _make_server_state(name="server_b", prefix="")
        server_b.mcp_tools = [mcp_tool_b]

        plugin._servers = [server_a, server_b]
        tools = plugin._build_tool_list()

        # Only one tool named "read_file" should appear.
        names = [t.name for t in tools]
        assert names.count("read_file") == 1

    def test_duplicate_skipped_warning_logged(self, caplog):
        """A WARNING is logged when a duplicate tool name is encountered."""
        plugin = McpClientPlugin(None)

        mcp_tool_a = _make_mcp_tool("query")
        mcp_tool_b = _make_mcp_tool("query")

        server_a = _make_server_state(name="server_a", prefix="")
        server_a.mcp_tools = [mcp_tool_a]

        server_b = _make_server_state(name="server_b", prefix="")
        server_b.mcp_tools = [mcp_tool_b]

        plugin._servers = [server_a, server_b]

        with caplog.at_level(logging.WARNING, logger="corvidae.mcp_client"):
            plugin._build_tool_list()

        assert any("collision" in record.message.lower() or "duplicate" in record.message.lower()
                   for record in caplog.records)

    def test_no_collision_both_tools_registered(self):
        """Tools with different names from different servers are both registered."""
        plugin = McpClientPlugin(None)

        mcp_tool_a = _make_mcp_tool("read_file")
        mcp_tool_b = _make_mcp_tool("write_file")

        server_a = _make_server_state(name="server_a", prefix="")
        server_a.mcp_tools = [mcp_tool_a]

        server_b = _make_server_state(name="server_b", prefix="")
        server_b.mcp_tools = [mcp_tool_b]

        plugin._servers = [server_a, server_b]
        tools = plugin._build_tool_list()

        names = {t.name for t in tools}
        assert names == {"read_file", "write_file"}

    def test_prefix_prevents_collision(self):
        """With different prefixes, same MCP tool name from two servers does not collide."""
        plugin = McpClientPlugin(None)

        mcp_tool_a = _make_mcp_tool("read_file")
        mcp_tool_b = _make_mcp_tool("read_file")

        server_a = _make_server_state(name="server_a", prefix="a")
        server_a.mcp_tools = [mcp_tool_a]

        server_b = _make_server_state(name="server_b", prefix="b")
        server_b.mcp_tools = [mcp_tool_b]

        plugin._servers = [server_a, server_b]
        tools = plugin._build_tool_list()

        names = {t.name for t in tools}
        assert names == {"a__read_file", "b__read_file"}


# ---------------------------------------------------------------------------
# TestNonTextContentBlocks
# ---------------------------------------------------------------------------


class TestNonTextContentBlocks:
    """Non-text content blocks are dropped; only text blocks are joined."""

    async def test_non_text_blocks_dropped(self):
        """ImageContent blocks (no .text) are ignored in the result."""
        mock_session = AsyncMock()
        image_block = _make_image_block()
        text_block = _make_text_block("some text")
        tool_result = _make_tool_result([image_block, text_block])
        mock_session.call_tool = AsyncMock(return_value=tool_result)

        result = await _call_mcp_tool(mock_session, "tool", {}, 30.0, "myserver")

        assert result == "some text"

    async def test_all_non_text_blocks_returns_no_output(self):
        """When all blocks are non-text, result is '(no output)'."""
        mock_session = AsyncMock()
        image_block = _make_image_block()
        tool_result = _make_tool_result([image_block])
        mock_session.call_tool = AsyncMock(return_value=tool_result)

        result = await _call_mcp_tool(mock_session, "tool", {}, 30.0, "myserver")

        assert result == "(no output)"

    async def test_non_text_blocks_logged_at_debug(self, caplog):
        """Non-text blocks are logged at DEBUG level."""
        mock_session = AsyncMock()
        image_block = _make_image_block()
        tool_result = _make_tool_result([image_block])
        mock_session.call_tool = AsyncMock(return_value=tool_result)

        with caplog.at_level(logging.DEBUG, logger="corvidae.mcp_client"):
            await _call_mcp_tool(mock_session, "tool", {}, 30.0, "myserver")

        assert any(record.levelno == logging.DEBUG for record in caplog.records)

    async def test_mixed_blocks_only_text_in_result(self):
        """Mixed text and non-text blocks: only text blocks appear in the result."""
        mock_session = AsyncMock()
        block1 = _make_text_block("first")
        block2 = _make_image_block()
        block3 = _make_text_block("second")
        tool_result = _make_tool_result([block1, block2, block3])
        mock_session.call_tool = AsyncMock(return_value=tool_result)

        result = await _call_mcp_tool(mock_session, "tool", {}, 30.0, "myserver")

        assert result == "first\nsecond"


async def test_generic_exception_names_the_server():
    """A generic tool-call failure names the MCP server, like the timeout branch."""
    mock_session = AsyncMock()
    mock_session.call_tool = AsyncMock(side_effect=RuntimeError("protocol error"))

    result = await _call_mcp_tool(mock_session, "read_file", {}, 30.0, "myserver")

    assert "myserver" in result

# ---------------------------------------------------------------------------
# Lazy reconnect for a server that failed at startup
# ---------------------------------------------------------------------------


def _working_session(tool_names: list[str]):
    """Return (session, session context manager) exposing the named tools."""
    session = AsyncMock()
    session.initialize = AsyncMock()
    result = MagicMock()
    result.tools = [_make_mcp_tool(name) for name in tool_names]
    session.list_tools = AsyncMock(return_value=result)
    session.call_tool = AsyncMock(
        return_value=_make_tool_result([_make_text_block("ok")])
    )
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=session)
    cm.__aexit__ = AsyncMock(return_value=False)
    return session, cm


def _working_stdio_cm():
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


def _failing_cm(error: Exception):
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(side_effect=error)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


_MCP_CONFIG = {
    "mcp": {
        "servers": {
            "broken": {"transport": "stdio", "command": "broken", "args": []},
            "good": {"transport": "stdio", "command": "good", "args": []},
        }
    }
}


class TestLazyReconnect:
    async def test_failed_server_reconnects_on_next_tool_call(self):
        """A server skipped at startup is retried when an MCP tool is called."""
        plugin = McpClientPlugin(None)
        await plugin.on_init(pm=clock_pm(), config=_MCP_CONFIG)

        good_session, good_session_cm = _working_session(["ok"])
        with (
            patch(
                "mcp.client.stdio.stdio_client",
                side_effect=[
                    _failing_cm(ConnectionRefusedError("no server")),
                    _working_stdio_cm(),
                ],
            ),
            patch("mcp.client.stdio.StdioServerParameters"),
            patch("mcp.ClientSession", return_value=good_session_cm),
        ):
            await plugin.on_start(config=_MCP_CONFIG)

        assert [s.name for s in plugin._servers] == ["good"]
        assert [t.name for t in plugin._cached_tools] == ["good__ok"]

        # The broken server is healthy by the time the first MCP tool runs.
        recovered_session, recovered_session_cm = _working_session(["recovered"])
        with (
            patch(
                "mcp.client.stdio.stdio_client",
                return_value=_working_stdio_cm(),
            ),
            patch("mcp.client.stdio.StdioServerParameters"),
            patch("mcp.ClientSession", return_value=recovered_session_cm),
        ):
            (good_tool,) = plugin._cached_tools
            result = await good_tool.fn()

        assert result == "ok"
        assert [s.name for s in plugin._servers] == ["good", "broken"]
        assert "broken__recovered" in [t.name for t in plugin._cached_tools]

    async def test_failed_reconnect_does_not_break_the_tool_call(self, caplog):
        """A still-broken server logs and leaves the current call working."""
        plugin = McpClientPlugin(None)
        await plugin.on_init(pm=clock_pm(), config=_MCP_CONFIG)

        good_session, good_session_cm = _working_session(["ok"])
        with (
            patch(
                "mcp.client.stdio.stdio_client",
                side_effect=[
                    _failing_cm(ConnectionRefusedError("no server")),
                    _working_stdio_cm(),
                ],
            ),
            patch("mcp.client.stdio.StdioServerParameters"),
            patch("mcp.ClientSession", return_value=good_session_cm),
        ):
            await plugin.on_start(config=_MCP_CONFIG)

        with caplog.at_level(logging.WARNING, logger="corvidae.mcp_client"), \
                patch(
                    "mcp.client.stdio.stdio_client",
                    return_value=_failing_cm(ConnectionRefusedError("still down")),
                ), \
                patch("mcp.client.stdio.StdioServerParameters"):
            (good_tool,) = plugin._cached_tools
            result = await good_tool.fn()

        assert result == "ok"
        assert any(
            "reconnect failed" in r.getMessage()
            for r in caplog.records if r.levelno >= logging.WARNING
        ), [r.getMessage() for r in caplog.records]


# ---------------------------------------------------------------------------
# Background retry loop for servers that failed at startup (#107)
# ---------------------------------------------------------------------------


class _Sdk:
    """A scripted MCP SDK: which servers answer, plus a log of every attempt.

    ``up`` names the servers that currently connect; ``gate`` (when set)
    holds every connect open until the test sets it; ``attempts`` records
    (monotonic time, server) per connect attempt.
    """

    def __init__(self, clock) -> None:
        self.clock = clock
        self.up: set[str] = set()
        self.gate: asyncio.Event | None = None
        self.entered = asyncio.Event()
        self.attempts: list[tuple[float, str]] = []

    def stdio_client(self, command: str):
        """Return a transport cm that connects only when ``command`` is up."""
        self.attempts.append((self.clock.monotonic(), command))
        sdk = self

        async def _enter(*_a):
            sdk.entered.set()
            if sdk.gate is not None:
                await sdk.gate.wait()
            if command not in sdk.up:
                raise ConnectionRefusedError(f"{command} is down")
            return (MagicMock(), MagicMock())

        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(side_effect=_enter)
        cm.__aexit__ = AsyncMock(return_value=False)
        return cm

    def client_session(self, *_a, **_k):
        """Return a session cm exposing one tool named ``t``."""
        _, cm = _working_session(["t"])
        return cm

    def attempts_for(self, name: str) -> list[float]:
        """Seconds (from the first recorded monotonic) of each attempt on ``name``."""
        return [t for t, n in self.attempts if n == name]


def _servers(*names: str) -> dict:
    """Config for stdio servers whose command is their own name."""
    return {
        "mcp": {
            "servers": {
                n: {"transport": "stdio", "command": n, "args": []} for n in names
            }
        }
    }


@pytest.fixture
def sdk(kernel_clock):
    """Patch the MCP SDK entry points with a scripted _Sdk for the test."""
    clock, _ = kernel_clock
    fake = _Sdk(clock)
    with (
        patch("mcp.client.stdio.stdio_client", side_effect=fake.stdio_client),
        patch(
            "mcp.client.stdio.StdioServerParameters",
            side_effect=lambda command, args, env: command,
        ),
        patch("mcp.ClientSession", side_effect=fake.client_session),
    ):
        yield fake


class _FakeTools:
    """Stand-in for ToolCollectionPlugin that counts registry rebuilds."""

    def __init__(self) -> None:
        self.rebuilds = 0

    async def rebuild_registry(self) -> None:
        self.rebuilds += 1


async def _start_plugin(pm, config: dict) -> tuple[McpClientPlugin, _FakeTools]:
    """Register a fake tools plugin, then init and start an McpClientPlugin."""
    tools = _FakeTools()
    pm.register(tools, name="tools")
    plugin = McpClientPlugin(None)
    await plugin.on_init(pm=pm, config=config)
    await plugin.on_start(config=config)
    return plugin, tools


def _owner_tasks() -> list[asyncio.Task]:
    """Live mcp-owner tasks in the running loop."""
    return [
        t for t in asyncio.all_tasks()
        if t.get_name().startswith("mcp-owner-") and not t.done()
    ]


class TestRetryLoop:
    async def test_all_down_server_recovers_without_a_tool_call(
        self, pm, kernel_clock, sdk
    ):
        """R1: with every server down, the loop reconnects past the first delay."""
        clock, _ = kernel_clock
        plugin, tools = await _start_plugin(pm, _servers("a", "b"))
        try:
            assert plugin._servers == []
            assert plugin._retry_task is not None
            sdk.up.add("a")

            await clock.advance(10)

            assert [s.name for s in plugin._servers] == ["a"]
            assert [t.name for t in plugin._cached_tools] == ["a__t"]
            assert tools.rebuilds == 1
            assert list(plugin._pending_servers) == ["b"]

            # b recovers later; once nothing is pending the loop exits.
            sdk.up.add("b")
            await clock.advance(20)
            assert plugin._pending_servers == {}
            assert plugin._retry_task.done()
            assert sorted(t.name for t in plugin._cached_tools) == ["a__t", "b__t"]
        finally:
            await plugin.on_stop()

    async def test_loop_not_started_when_nothing_is_pending(
        self, pm, kernel_clock, sdk
    ):
        """R2: a clean startup starts no retry task."""
        sdk.up.add("a")
        plugin, _ = await _start_plugin(pm, _servers("a"))
        try:
            assert plugin._retry_task is None
        finally:
            await plugin.on_stop()

    async def test_backoff_ladder(self, pm, kernel_clock, sdk):
        """R2: attempts land at 10, 30, 70, 150, 310, 610, 910 s."""
        clock, _ = kernel_clock
        plugin, _ = await _start_plugin(pm, _servers("a"))
        try:
            start = clock.monotonic()
            sdk.attempts.clear()
            await clock.advance(910)
            offsets = [t - start for t in sdk.attempts_for("a")]
            assert offsets == [10, 30, 70, 150, 310, 610, 910]
        finally:
            await plugin.on_stop()

    async def test_backoff_resets_after_a_reconnect(self, pm, kernel_clock, sdk):
        """R2: a round that reconnects something resets the delay to 10 s."""
        clock, _ = kernel_clock
        plugin, _ = await _start_plugin(pm, _servers("a", "b"))
        try:
            await clock.advance(10)  # both fail; next delay 20
            sdk.up.add("a")
            await clock.advance(20)  # a reconnects; b still pending
            start = clock.monotonic()
            sdk.attempts.clear()
            await clock.advance(10)
            assert [t - start for t in sdk.attempts_for("b")] == [10]
        finally:
            await plugin.on_stop()

    async def test_failed_retry_logs_warning_with_server_type_and_delay(
        self, pm, kernel_clock, sdk, caplog
    ):
        """R3: the WARNING text names server, exception type and next delay."""
        clock, _ = kernel_clock
        plugin, _ = await _start_plugin(pm, _servers("a"))
        try:
            caplog.clear()
            with caplog.at_level(logging.WARNING, logger="corvidae.mcp_client"):
                await clock.advance(10)
            records = [
                r for r in caplog.records
                if r.levelno == logging.WARNING and "retrying in" in r.getMessage()
            ]
            assert len(records) == 1, [r.getMessage() for r in caplog.records]
            msg = records[0].getMessage()
            assert "'a'" in msg
            assert "ConnectionRefusedError" in msg
            assert "retrying in 20s" in msg
            assert records[0].exc_info is not None
        finally:
            await plugin.on_stop()

    async def test_tool_call_retry_does_not_race_the_loop(
        self, pm, kernel_clock, sdk
    ):
        """R4: a tool call during an in-flight loop reconnect starts no second connect."""
        clock, _ = kernel_clock
        sdk.up.add("good")
        plugin, _ = await _start_plugin(pm, _servers("good", "a"))
        try:
            (good_tool,) = plugin._cached_tools
            sdk.up.add("a")
            sdk.gate = asyncio.Event()
            sdk.entered.clear()
            sdk.attempts.clear()
            advancing = asyncio.create_task(clock.advance(10))
            await asyncio.wait_for(sdk.entered.wait(), 2)

            # The loop's connect for "a" is held open; the tool call must skip.
            assert await good_tool.fn() == "ok"
            assert len(sdk.attempts_for("a")) == 1

            sdk.gate.set()
            await advancing
            assert len(sdk.attempts_for("a")) == 1
            assert [s.name for s in plugin._servers].count("a") == 1
            assert len(plugin._owners) == 2
            assert sorted(t.name for t in plugin._cached_tools) == ["a__t", "good__t"]
        finally:
            await plugin.on_stop()

    async def test_on_stop_while_loop_is_sleeping(self, pm, kernel_clock, sdk):
        """R5: on_stop wakes and finishes a sleeping loop."""
        plugin, _ = await _start_plugin(pm, _servers("a"))
        task = plugin._retry_task
        await plugin.on_stop()
        assert task.done()
        assert _owner_tasks() == []

    async def test_on_stop_while_loop_is_connecting(self, pm, kernel_clock, sdk):
        """R5: on_stop cancels a loop that is mid-connect, leaving no owner."""
        clock, _ = kernel_clock
        plugin, _ = await _start_plugin(pm, _servers("a"))
        task = plugin._retry_task
        sdk.gate = asyncio.Event()  # never set: the connect hangs
        sdk.entered.clear()
        advancing = asyncio.create_task(clock.advance(10))
        await asyncio.wait_for(sdk.entered.wait(), 2)

        await plugin.on_stop()

        assert task.done()
        assert _owner_tasks() == []
        advancing.cancel()
        await asyncio.gather(advancing, return_exceptions=True)

    async def test_connect_finishing_after_stop_is_torn_down(
        self, pm, kernel_clock, sdk
    ):
        """A tool-call retry still connecting when on_stop runs registers no owner."""
        sdk.up.update({"good", "a"})
        plugin, _ = await _start_plugin(pm, _servers("good"))
        # Queue "a" as pending without the loop, and call a tool to retry it.
        plugin._pending_servers["a"] = _servers("a")["mcp"]["servers"]["a"]
        plugin._servers_config = {}
        plugin._cached_tools = plugin._build_tool_list()
        sdk.gate = asyncio.Event()
        sdk.entered.clear()
        (good_tool,) = plugin._cached_tools
        call = asyncio.create_task(good_tool.fn())
        await asyncio.wait_for(sdk.entered.wait(), 2)

        await plugin.on_stop()
        sdk.gate.set()
        await asyncio.wait_for(call, 2)

        assert plugin._owners == {}
        assert plugin._servers == []
        assert _owner_tasks() == []

    async def test_config_errors_are_not_retried(
        self, pm, kernel_clock, sdk, caplog, monkeypatch
    ):
        """R6: config-error servers are skipped once, naming the key to fix."""
        clock, _ = kernel_clock
        monkeypatch.delenv("NOPE_TOKEN", raising=False)
        config = {"mcp": {"servers": {
            "down": {"transport": "stdio", "command": "down"},
            "hdr": {"transport": "sse", "url": "http://x",
                    "headers": {"Authorization": "Bearer ${NOPE_TOKEN}"}},
            "badtransport": {"transport": "carrier-pigeon", "command": "badtransport"},
            "nocommand": {"transport": "stdio"},
            "badtimeout": {"transport": "stdio", "command": "badtimeout",
                           "connect_timeout_seconds": -1},
        }}}
        with caplog.at_level(logging.WARNING, logger="corvidae.mcp_client"):
            plugin, _ = await _start_plugin(pm, config)
        try:
            assert list(plugin._pending_servers) == ["down"]
            texts = [r.getMessage() for r in caplog.records
                     if r.levelno == logging.WARNING]
            for server, key in [
                ("hdr", "NOPE_TOKEN"),
                ("badtransport", "transport"),
                ("nocommand", "command"),
                ("badtimeout", "connect_timeout_seconds"),
            ]:
                hits = [t for t in texts if server in t and key in t]
                assert len(hits) == 1, (server, key, texts)

            sdk.attempts.clear()
            await clock.advance(10)
            assert {n for _, n in sdk.attempts} == {"down"}
        finally:
            await plugin.on_stop()


class TestConfigTypeErrors:
    """Wrongly typed config values skip that server only (#107 review)."""

    @pytest.mark.parametrize("bad, key", [
        (None, "mcp.servers.bad"),
        ({"transport": "stdio", "command": "bad", "timeout_seconds": [1]},
         "timeout_seconds"),
        ({"transport": "stdio", "command": "bad",
          "connect_timeout_seconds": {"x": 1}}, "connect_timeout_seconds"),
        ({"transport": "sse", "url": "http://x", "headers": ["h"]},
         "mcp.servers.bad.headers"),
    ])
    async def test_bad_type_is_skipped_and_later_server_connects(
        self, pm, kernel_clock, sdk, caplog, bad, key
    ):
        sdk.up.add("good")
        config = _servers("good")
        config["mcp"]["servers"] = {"bad": bad, **config["mcp"]["servers"]}
        caplog.set_level(logging.WARNING, logger="corvidae.mcp_client")
        plugin, _ = await _start_plugin(pm, config)
        try:
            assert [s.name for s in plugin._servers] == ["good"]
            assert plugin._pending_servers == {}
            skips = [r for r in caplog.records if r.levelno == logging.WARNING
                     and r.__dict__.get("server") == "bad"]
            assert len(skips) == 1
            msg = skips[0].getMessage()
            assert "bad" in msg and key in msg
        finally:
            await plugin.on_stop()

    async def test_unexpected_parse_exception_is_skipped(
        self, pm, kernel_clock, sdk, caplog, monkeypatch
    ):
        sdk.up.add("good")
        config = _servers("bad", "good")
        real = mcp_client_mod._parse_server_config

        def boom(name, cfg):
            if name == "bad":
                raise RuntimeError("surprise")
            return real(name, cfg)

        monkeypatch.setattr(mcp_client_mod, "_parse_server_config", boom)
        caplog.set_level(logging.WARNING, logger="corvidae.mcp_client")
        plugin, _ = await _start_plugin(pm, config)
        try:
            assert [s.name for s in plugin._servers] == ["good"]
            assert plugin._pending_servers == {}
            skips = [r for r in caplog.records if r.levelno == logging.WARNING
                     and r.__dict__.get("server") == "bad"]
            assert len(skips) == 1 and skips[0].exc_info is not None
            assert "'bad'" in skips[0].getMessage()
        finally:
            await plugin.on_stop()


class TestConfigShapeErrors:
    """Remaining shape/value gaps are rejected once, never retried (#265)."""

    @pytest.mark.parametrize("bad, key, shown", [
        ({"transport": "stdio", "command": None}, "mcp.servers.bad.command", "None"),
        ({"transport": "stdio", "command": ["x"]}, "mcp.servers.bad.command", "['x']"),
        ({"transport": "sse", "url": None}, "mcp.servers.bad.url", "None"),
        ({"transport": "stdio", "command": "c", "args": "x"},
         "mcp.servers.bad.args", "'x'"),
        ({"transport": "stdio", "command": "c", "args": [1]},
         "mcp.servers.bad.args", "[1]"),
        ({"transport": "stdio", "command": "c", "env": ["A=1"]},
         "mcp.servers.bad.env", "['A=1']"),
        ({"transport": "stdio", "command": "c",
          "connect_timeout_seconds": float("nan")},
         "mcp.servers.bad.connect_timeout_seconds", "nan"),
        ({"transport": "stdio", "command": "c",
          "connect_timeout_seconds": float("inf")},
         "mcp.servers.bad.connect_timeout_seconds", "inf"),
        ({"transport": "stdio", "command": "c", "timeout_seconds": float("nan")},
         "mcp.servers.bad.timeout_seconds", "nan"),
    ])
    async def test_bad_shape_skipped_once_and_not_retried(
        self, pm, kernel_clock, sdk, caplog, bad, key, shown
    ):
        sdk.up.add("good")
        config = _servers("good")
        config["mcp"]["servers"] = {"bad": bad, **config["mcp"]["servers"]}
        caplog.set_level(logging.WARNING, logger="corvidae.mcp_client")
        plugin, _ = await _start_plugin(pm, config)
        try:
            assert [s.name for s in plugin._servers] == ["good"]
            assert plugin._pending_servers == {}
            skips = [r for r in caplog.records if r.levelno == logging.WARNING
                     and r.__dict__.get("server") == "bad"]
            assert len(skips) == 1
            msg = skips[0].getMessage()
            assert key in msg and shown in msg
            assert "failed to connect" not in msg
        finally:
            await plugin.on_stop()

    @pytest.mark.parametrize("mcp, key, shown", [
        (["x"], "mcp", "['x']"),
        ({"servers": ["a"]}, "mcp.servers", "['a']"),
        ({"servers": "a"}, "mcp.servers", "'a'"),
    ])
    async def test_non_mapping_block_raises_named_value_error(
        self, pm, mcp, key, shown
    ):
        from corvidae.mcp_client import McpClientPlugin
        with pytest.raises(ValueError) as excinfo:
            await McpClientPlugin().on_init(pm=pm, config={"mcp": mcp})
        assert key in str(excinfo.value) and shown in str(excinfo.value)

    @pytest.mark.parametrize("mcp", [None, {}, {"servers": None}])
    async def test_null_or_empty_block_means_no_servers(self, pm, mcp):
        from corvidae.mcp_client import McpClientPlugin
        plugin = McpClientPlugin()
        await plugin.on_init(pm=pm, config={"mcp": mcp})
        assert plugin._servers_config == {}


class TestRetryLoopSurvivesRebuildError:
    async def test_rebuild_registry_error_is_logged_and_loop_continues(
        self, pm, kernel_clock, sdk, caplog
    ):
        clock, _ = kernel_clock
        plugin, tools = await _start_plugin(pm, _servers("a", "b"))
        try:
            async def boom() -> None:
                raise RuntimeError("rebuild broke")

            tools.rebuild_registry = boom
            sdk.up.add("a")
            caplog.set_level(logging.WARNING, logger="corvidae.mcp_client")
            await clock.advance(10)

            assert not plugin._retry_task.done()
            msgs = [r.getMessage() for r in caplog.records
                    if r.levelno == logging.WARNING and "rebuild" in r.getMessage()]
            assert len(msgs) == 1 and "'a'" in msgs[0]

            # The loop keeps retrying: b recovers on the next round.
            sdk.up.add("b")
            await clock.advance(10)
            assert sorted(s.name for s in plugin._servers) == ["a", "b"]
        finally:
            await plugin.on_stop()

    async def test_tool_call_retry_message_promises_no_delay(
        self, pm, kernel_clock, sdk, caplog
    ):
        sdk.up.add("good")
        plugin, _ = await _start_plugin(pm, _servers("good", "late"))
        try:
            tool = next(t for t in plugin._cached_tools if t.name == "good__t")
            caplog.set_level(logging.WARNING, logger="corvidae.mcp_client")
            caplog.clear()
            await tool.fn()
            msgs = [r.getMessage() for r in caplog.records
                    if "reconnect failed" in r.getMessage()]
            assert msgs and all("retrying in" not in m for m in msgs)
            assert all("background retry" in m for m in msgs)
        finally:
            await plugin.on_stop()
