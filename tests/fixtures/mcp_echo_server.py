"""Minimal stdio MCP server with one `echo` tool, for real-transport tests."""

from mcp.server.mcpserver import MCPServer

server = MCPServer("echo")


@server.tool()
def echo(text: str) -> str:
    """Echo text back."""
    return text


if __name__ == "__main__":
    server.run()
