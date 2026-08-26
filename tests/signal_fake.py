"""A fake signal-cli JSON-RPC daemon, for testing corvidae's Signal transport.

signal-cli's `daemon` mode speaks JSON-RPC 2.0, newline-delimited, over a
unix socket: inbound Signal messages arrive as unsolicited notifications
(method "receive"), and the client sends requests (e.g. a "send" request)
that get a matching JSON-RPC response keyed by "id". This module is a
minimal stdlib-asyncio double of that wire protocol -- no signal-cli
process, no library mocking, just the same bytes on the same kind of
socket. Behavior that only manifests against the real daemon -- receive-mode
semantics, typing-indicator expiry, reconnection -- belongs to the live
shakedown; see docs/signal-ops.md.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("tests.signal_fake")


def frame_contains(obj: Any, value: Any) -> bool:
    """Recursively search a decoded JSON-RPC frame for a value.

    Used instead of asserting exact field names for pieces of the wire
    protocol the design leaves unresolved (e.g. which param carries a
    timestamp or a recipient) -- the tests care that the value made it
    into the outbound request somewhere, not the exact key it travels
    under.
    """
    if obj == value:
        return True
    if isinstance(obj, dict):
        return any(frame_contains(v, value) for v in obj.values())
    if isinstance(obj, list):
        return any(frame_contains(v, value) for v in obj)
    return False


class FakeSignalServer:
    """A single-client fake signal-cli JSON-RPC daemon on a unix socket.

    Typical use:

        server = FakeSignalServer(tmp_path / "signal.sock")
        await server.start()
        # ... start the plugin pointed at server.socket_path ...
        await server.wait_for_connection()
        await server.push_notification("receive", {...envelope...})
        request = await server.next_request()
        await server.stop()
    """

    def __init__(self, socket_path: Path) -> None:
        self.socket_path = socket_path
        self._server: asyncio.AbstractServer | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._reader: asyncio.StreamReader | None = None
        self._connected = asyncio.Event()
        self._requests: asyncio.Queue[dict] = asyncio.Queue()
        self.received: list[dict] = []
        self._read_task: asyncio.Task | None = None
        self._pending_notifications: list[tuple[str, dict]] = []
        # method-substring -> error message, consumed once, for forcing an
        # RPC-level error response to the next matching request.
        self._fail_next: dict[str, str] = {}
        # value -> canned result, consumed once, for the next request whose
        # frame contains that value anywhere (used for contact-lookup-style
        # round trips where the exact method name is unresolved by design).
        self._respond_next: list[tuple[Any, Any]] = []
        self.connection_count = 0
        # Optional synchronous callback invoked with each request frame the
        # instant it is read, for tests that need to assert ordering against
        # some other event (e.g. "typing fires before on_message is called").
        self.on_request: Any = None

    async def start(self) -> None:
        """Start listening on the unix socket."""
        self._server = await asyncio.start_unix_server(
            self._on_client_connected, path=str(self.socket_path)
        )

    async def _on_client_connected(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Accept a client connection and (re)start the read loop.

        Only one client is expected at a time, mirroring signal-cli's
        daemon; a reconnect after a drop replaces the prior reader/writer.
        """
        self._reader = reader
        self._writer = writer
        self.connection_count += 1
        self._connected.set()
        # Flush anything queued before this connection existed -- models
        # --receive-mode=on-connection delivering the backlog the instant
        # a JSON-RPC client attaches.
        for method, params in self._pending_notifications:
            await self._write_frame({"jsonrpc": "2.0", "method": method, "params": params})
        self._pending_notifications = []
        self._read_task = asyncio.create_task(self._read_loop())

    async def _read_loop(self) -> None:
        """Read newline-delimited JSON-RPC frames the client sends and record them."""
        assert self._reader is not None
        try:
            while True:
                line = await self._reader.readline()
                if not line:
                    return
                stripped = line.strip()
                if not stripped:
                    continue
                frame = json.loads(stripped.decode("utf-8"))
                self.received.append(frame)
                if self.on_request is not None:
                    self.on_request(frame)
                await self._requests.put(frame)
                await self._maybe_auto_respond(frame)
        except asyncio.CancelledError:
            raise
        except (OSError, ConnectionError):
            # Peer went away mid-read; not an error in the test harness.
            logger.debug("fake signal server read loop ended", exc_info=True)

    async def _maybe_auto_respond(self, frame: dict) -> None:
        """Reply to a request frame: a canned error or result if one was
        armed for it, else a generic success -- signal-cli's daemon acks
        every request, so a client that awaits a response for calls it
        does not otherwise care about (e.g. typing, receipts) must not
        hang against this double by default."""
        method = str(frame.get("method", "")).lower()
        req_id = frame.get("id")
        if req_id is None:
            return
        for substr, message in list(self._fail_next.items()):
            if substr.lower() in method:
                del self._fail_next[substr]
                await self._write_frame(
                    {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": -1, "message": message},
                    }
                )
                return
        for i, (value, result) in enumerate(self._respond_next):
            if frame_contains(frame, value):
                del self._respond_next[i]
                await self._write_frame({"jsonrpc": "2.0", "id": req_id, "result": result})
                return
        await self._write_frame({"jsonrpc": "2.0", "id": req_id, "result": None})

    async def _write_frame(self, frame: dict) -> None:
        assert self._writer is not None
        self._writer.write((json.dumps(frame) + "\n").encode("utf-8"))
        await self._writer.drain()

    async def wait_for_connection(self, timeout: float = 5.0) -> None:
        """Block until a client has connected at least once."""
        await asyncio.wait_for(self._connected.wait(), timeout=timeout)

    async def push_notification(self, method: str, params: dict) -> None:
        """Send an unsolicited JSON-RPC notification to the connected client."""
        await self.wait_for_connection()
        await self._write_frame({"jsonrpc": "2.0", "method": method, "params": params})

    def queue_before_connect(self, method: str, params: dict) -> None:
        """Queue a notification to be delivered the instant a client connects.

        Models signal-cli's --receive-mode=on-connection: messages that
        arrived while no client was attached are flushed on attach.
        """
        self._pending_notifications.append((method, params))

    async def push_raw_line(self, line: str) -> None:
        """Send a raw (possibly malformed) line, for malformed-frame tests."""
        await self.wait_for_connection()
        assert self._writer is not None
        self._writer.write((line + "\n").encode("utf-8"))
        await self._writer.drain()

    def fail_next_matching(self, method_substring: str, message: str = "boom") -> None:
        """Arm a one-shot JSON-RPC error response for the next request whose
        method contains ``method_substring``."""
        self._fail_next[method_substring] = message

    def respond_when_contains(self, value: Any, result: Any) -> None:
        """Arm a one-shot JSON-RPC success response for the next request whose
        frame contains ``value`` anywhere (any key), replying with ``result``.

        Matches by content rather than method name, since the exact JSON-RPC
        method for e.g. a contact lookup is left unresolved by the design
        ("listContacts or the installed version's equivalent; verify the
        method name").
        """
        self._respond_next.append((value, result))

    async def next_request(self, timeout: float = 5.0) -> dict:
        """Wait for and return the next JSON-RPC frame sent by the client."""
        return await asyncio.wait_for(self._requests.get(), timeout=timeout)

    def requests_matching(self, method_substring: str) -> list[dict]:
        """Return already-received frames whose method contains the substring
        (case-insensitive -- signal-cli's methods are camelCase)."""
        return [
            f for f in self.received
            if method_substring.lower() in str(f.get("method", "")).lower()
        ]

    async def drop_connection(self) -> None:
        """Simulate the daemon dropping the connection (e.g. a restart)."""
        if self._read_task is not None:
            self._read_task.cancel()
            try:
                await self._read_task
            except (asyncio.CancelledError, Exception):
                logger.debug("exception awaiting cancelled read task", exc_info=True)
            self._read_task = None
        if self._writer is not None:
            self._writer.close()
        self._connected.clear()
        self._reader = None
        self._writer = None

    async def stop(self) -> None:
        """Tear down the server and any open connection."""
        await self.drop_connection()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
