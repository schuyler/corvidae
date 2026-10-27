"""Subprocess ACP smoke: real stdio + Runtime, mocked LLM (no live API).

Complements Gate B (in-process). This boots ``corvidae acp``, speaks ACP over
pipes via the official client, and proves stdout stays JSON-RPC (no CLI banner).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from aiohttp import web

pytest.importorskip("acp")

pytestmark = pytest.mark.timeout(60)

REPO_ROOT = Path(__file__).resolve().parents[1]


class _RecordingClient:
    """Minimal ACP client that records session updates."""

    def __init__(self) -> None:
        self.updates: list[tuple[str, object]] = []

    async def session_update(self, session_id: str, update, **kwargs) -> None:
        self.updates.append((session_id, update))

    async def request_permission(self, session_id, tool_call, options, **kwargs):
        from acp.schema import DeniedOutcome, RequestPermissionResponse

        return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))

    async def write_text_file(self, *args, **kwargs):
        return None

    async def read_text_file(self, *args, **kwargs):
        from acp.schema import ReadTextFileResponse

        return ReadTextFileResponse(content="")

    async def create_terminal(self, *args, **kwargs):
        raise NotImplementedError("terminals not used in smoke")

    async def terminal_output(self, *args, **kwargs):
        raise NotImplementedError("terminals not used in smoke")

    async def release_terminal(self, *args, **kwargs):
        return None

    async def wait_for_terminal_exit(self, *args, **kwargs):
        raise NotImplementedError("terminals not used in smoke")

    async def kill_terminal(self, *args, **kwargs):
        return None

    async def create_elicitation(self, *args, **kwargs):
        raise NotImplementedError("elicitation not used in smoke")

    async def complete_elicitation(self, *args, **kwargs) -> None:
        return None

    async def ext_method(self, method: str, params: dict) -> dict:
        return {}

    async def ext_notification(self, method: str, params: dict) -> None:
        return None

    def on_connect(self, conn) -> None:
        return None


async def _start_mock_llm(response_text: str) -> tuple[web.AppRunner, str]:
    """Serve one OpenAI-shaped chat completion; return (runner, base_url)."""

    async def chat_completions(request: web.Request) -> web.Response:
        # Ignore body; always return the canned assistant message.
        await request.read()
        body = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": response_text,
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        return web.json_response(body)

    app = web.Application()
    app.router.add_post("/chat/completions", chat_completions)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    sockets = site._server.sockets  # noqa: SLF001 — need bound port
    port = sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}"


def _write_smoke_config(path: Path, *, base_url: str, session_db: Path) -> None:
    """Minimal agent.yaml: mocked LLM, cognition plugins off, local session DB."""
    config = {
        "llm": {
            "main": {
                "base_url": base_url,
                "model": "mock-model",
            }
        },
        "daemon": {"session_db": str(session_db)},
        "agent": {
            "system_prompt": "You are a test assistant.",
            "max_context_tokens": 4096,
            "max_turns": 3,
        },
        "channels": {
            "cli:local": {
                "system_prompt": "CLI channel present so config looks real; ACP mode must ignore it.",
            }
        },
        "plugins": {
            "disabled": [
                "memory",
                "memory_tools",
                "funnel",
                "appraisal",
                "critique",
                "outcome_log",
            ]
        },
        "logging": {
            "level": "WARNING",
            "file": str(path.parent / "corvidae-acp-smoke.log"),
        },
    }
    path.write_text(yaml.safe_dump(config), encoding="utf-8")


@pytest.mark.asyncio
async def test_acp_stdio_subprocess_initialize_session_prompt(tmp_path: Path) -> None:
    """Spawn ``corvidae acp``, drive ACP over stdio, get a mocked LLM reply."""
    from acp import PROTOCOL_VERSION, text_block
    from acp.stdio import spawn_agent_process

    canned = "STDIO_SMOKE_OK"
    runner, base_url = await _start_mock_llm(canned)
    try:
        config_path = tmp_path / "agent.yaml"
        _write_smoke_config(
            config_path,
            base_url=base_url,
            session_db=tmp_path / "sessions.db",
        )
        client = _RecordingClient()

        # Prefer the project venv's uv/python entry — matches local and CI.
        command = "uv"
        args = [
            "run",
            "--directory",
            str(REPO_ROOT),
            "--extra",
            "acp",
            "corvidae",
            "acp",
            "--config",
            str(config_path),
        ]

        async with spawn_agent_process(
            client,
            command,
            *args,
            cwd=str(REPO_ROOT),
        ) as (conn, process):
            init = await conn.initialize(protocol_version=PROTOCOL_VERSION)
            assert init.protocol_version == PROTOCOL_VERSION
            assert init.agent_info is not None
            assert init.agent_info.name == "corvidae"

            created = await conn.new_session(cwd=str(tmp_path))
            assert created.session_id

            prompt_resp = await conn.prompt(
                session_id=created.session_id,
                prompt=[text_block("say the smoke token")],
            )
            assert prompt_resp.stop_reason == "end_turn"

            # At least one agent_message_chunk with the canned LLM text.
            texts: list[str] = []
            for _sid, update in client.updates:
                session_update = getattr(update, "session_update", None)
                if session_update == "agent_message_chunk":
                    content = getattr(update, "content", None)
                    text = getattr(content, "text", None) if content is not None else None
                    if text:
                        texts.append(text)
            assert any(canned in t for t in texts), (
                f"expected canned LLM text in session updates, got {texts!r}"
            )

            # Process must still be running mid-session (stdio ownership OK).
            assert process.returncode is None
    finally:
        await runner.cleanup()
