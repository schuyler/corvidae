"""JSON-RPC 2.0 client for the WhatsApp sidecar's unix socket (#1220).

The sidecar is one connection, newline-delimited JSON-RPC 2.0. Requests carry
an id and wait for the reply with that id; notifications are messages the
sidecar sends unprompted (``message``, ``session``) and take no reply.

Two error codes matter to the operator, and they are read from
``error.data.state`` rather than by matching prose:

* ``-32001`` NOT_LINKED -- WhatsApp logged the device out; re-pair.
* ``-32002`` NOT_CONNECTED -- the link is fine but the socket is not ready;
  wait.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
from typing import Any, Awaitable, Callable

logger = logging.getLogger("corvidae.whatsapp")

#: Error codes the plugin translates into an operator sentence.
NOT_LINKED = -32001
NOT_CONNECTED = -32002
BATCH_REFUSED = -32600


class SidecarError(RuntimeError):
    """A JSON-RPC error reply.

    ``state`` is the machine-readable half (``error.data.state``); ``fix`` is
    the sentence to show a person.
    """

    def __init__(self, code: int, message: str, state: str | None, fix: str | None):
        super().__init__(f"{message} (code {code}, state {state})")
        self.code = code
        self.state = state
        self.fix = fix


def _error_from_reply(error: dict) -> SidecarError:
    """One reply's ``error`` object as a SidecarError, naming the fix."""
    code = int(error.get("code", 0))
    data = error.get("data") or {}
    state = data.get("state") if isinstance(data, dict) else None
    fix = data.get("fix") if isinstance(data, dict) else None
    if fix is None:
        fix = {
            NOT_LINKED: "the WhatsApp link was logged out; run the pair tool to relink",
            NOT_CONNECTED: "the sidecar is not connected yet; wait and try again",
            BATCH_REFUSED: "send one request per write, never a batch",
        }.get(code)
    return SidecarError(code, str(error.get("message", "sidecar error")), state, fix)


class SidecarClient:
    """One connection to the sidecar, with requests and notifications.

    ``on_notification(method, params)`` is awaited for every unprompted
    message; it must not block the read loop for long, since replies queue
    behind it.
    """

    def __init__(
        self,
        path: str,
        *,
        on_notification: Callable[[str, dict], Awaitable[None]],
        call_timeout: float = 30.0,
    ) -> None:
        self.path = path
        self._on_notification = on_notification
        self._call_timeout = call_timeout
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._pending: dict[int, asyncio.Future] = {}
        self._ids = itertools.count(1)
        self._read_task: asyncio.Task | None = None

    @property
    def connected(self) -> bool:
        return self._writer is not None and not self._writer.is_closing()

    async def connect(self) -> None:
        """Open the socket and start the read loop."""
        self._reader, self._writer = await asyncio.open_unix_connection(self.path)
        self._read_task = asyncio.create_task(self._read_loop())

    async def close(self) -> None:
        """Stop the read loop and drop the connection."""
        task, self._read_task = self._read_task, None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        writer, self._writer = self._writer, None
        if writer is not None:
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass
        for future in self._pending.values():
            if not future.done():
                future.set_exception(ConnectionError("the sidecar connection closed"))
        self._pending.clear()

    async def _read_loop(self) -> None:
        """Answer replies by id, hand notifications to the plugin, one line each."""
        assert self._reader is not None
        while True:
            line = await self._reader.readline()
            if not line:
                raise ConnectionError("the sidecar closed the socket")
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("whatsapp: sidecar sent a line that is not JSON; skipping")
                continue
            if not isinstance(payload, dict):
                logger.warning("whatsapp: sidecar sent a JSON value that is not an object")
                continue
            if "id" in payload and ("result" in payload or "error" in payload):
                self._settle(payload)
            elif "method" in payload:
                try:
                    await self._on_notification(payload["method"], payload.get("params") or {})
                except Exception:
                    # A plugin failure must not stop the read loop: the next
                    # reply would otherwise hang behind it.
                    logger.warning(
                        "whatsapp: notification %s was not handled", payload.get("method"),
                        exc_info=True,
                    )
            else:
                logger.warning("whatsapp: sidecar sent an object that is neither reply nor notification")

    def _settle(self, payload: dict) -> None:
        """Resolve or reject the request waiting on this reply."""
        future = self._pending.pop(payload.get("id"), None)
        if future is None or future.done():
            logger.debug("whatsapp: reply for an unknown request id; dropping")
            return
        if "error" in payload:
            future.set_exception(_error_from_reply(payload["error"]))
        else:
            future.set_result(payload.get("result"))

    async def call(self, method: str, params: dict | None = None) -> Any:
        """Send one request and wait for its reply, or raise SidecarError."""
        if self._writer is None or self._writer.is_closing():
            raise SidecarError(NOT_CONNECTED, "the sidecar socket is not connected", "not_connected",
                               "the sidecar is not connected; start it and try again")
        request_id = next(self._ids)
        payload = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}}
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        self._writer.write(json.dumps(payload, separators=(",", ":")).encode() + b"\n")
        await self._writer.drain()
        try:
            return await asyncio.wait_for(future, timeout=self._call_timeout)
        except asyncio.TimeoutError:
            self._pending.pop(request_id, None)
            raise SidecarError(
                NOT_CONNECTED, f"the sidecar did not answer {method}", "timeout",
                f"the sidecar did not answer {method} within {self._call_timeout:.0f}s;"
                " check that it is running",
            ) from None
