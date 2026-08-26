"""WP-A1.3: swappable ToolBackend seam for ACP channels."""

from __future__ import annotations

from typing import Any

import pytest

from corvidae.channel import Channel
from corvidae.tool import dispatch_tool_call


class _FakeBackend:
    """Records run() calls and returns a fixed string."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict, Channel]] = []

    async def run(self, name: str, args: dict, channel: Channel) -> str:
        self.calls.append((name, args, channel))
        return "fake-backend-result"


@pytest.mark.asyncio
async def test_local_tool_backend_runs_callable() -> None:
    """LocalToolBackend.run invokes the named tool and returns its result."""
    from corvidae.tools.backends import LocalToolBackend

    async def echo(text: str) -> str:
        return f"echo:{text}"

    backend = LocalToolBackend(tools={"echo": echo})
    channel = Channel(transport="acp", scope="s1")
    result = await backend.run("echo", {"text": "hi"}, channel)
    assert result == "echo:hi"


@pytest.mark.asyncio
async def test_acp_channel_uses_tool_backend_seam() -> None:
    """ACP channels honor channel.runtime_overrides['tool_backend']."""
    fake = _FakeBackend()
    channel = Channel(transport="acp", scope="s1")
    channel.runtime_overrides["tool_backend"] = fake

    async def unused() -> str:
        return "should-not-run"

    call = {
        "id": "tc1",
        "function": {"name": "unused", "arguments": "{}"},
    }
    result = await dispatch_tool_call(call, {"unused": unused}, channel=channel)
    assert result.content == "fake-backend-result"
    assert result.error is False
    assert len(fake.calls) == 1
    assert fake.calls[0][0] == "unused"


@pytest.mark.asyncio
async def test_acp_channel_defaults_to_local_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without an override, ACP channels construct LocalToolBackend for dispatch."""
    from corvidae.tools import backends

    constructed: list[Any] = []
    RealLocal = backends.LocalToolBackend

    class SpyLocal(RealLocal):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            constructed.append(True)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(backends, "LocalToolBackend", SpyLocal)

    channel = Channel(transport="acp", scope="s1")

    async def ping() -> str:
        return "pong"

    call = {
        "id": "tc1",
        "function": {"name": "ping", "arguments": "{}"},
    }
    result = await dispatch_tool_call(call, {"ping": ping}, channel=channel)
    assert result.content == "pong"
    assert constructed, "ACP dispatch must construct LocalToolBackend when no override"


@pytest.mark.asyncio
async def test_non_acp_channel_unaffected() -> None:
    """CLI (and other) channels keep calling tools directly when no backend set."""
    channel = Channel(transport="cli", scope="main")

    async def greet() -> str:
        return "hello-cli"

    call = {
        "id": "tc1",
        "function": {"name": "greet", "arguments": "{}"},
    }
    result = await dispatch_tool_call(call, {"greet": greet}, channel=channel)
    assert result.content == "hello-cli"
    assert "tool_backend" not in channel.runtime_overrides
