"""Context leases: which key may have a model call in flight on a bounded provider.

A provider that declares ``heads`` can serve that many keys at once (a key is
the queue a call belongs to, see ``corvidae.attribution.lease_scope``). This
module holds a lease table per bounded provider (holders, FIFO waiters, stop
flags, in-call counts) and the default policy: sticky until contention (#364,
#367). A holder keeps its head, idle included, while no other key waits; when
one does, an idle holder gives way at a call boundary.

The policy is one function, ``ProviderLeases._settle``, re-run after every
state change. Nothing here calls a model. With no bounded chat-used provider
there is no manager and clients never touch this code.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

#: Why a holder gave up its head while it still had (or may have) work.
YieldReason = Literal["contention", "cooperative", "stop"]


@dataclass(frozen=True)
class HolderView:
    """A holder in a snapshot."""

    key: str
    held_s: float
    in_call: bool


@dataclass(frozen=True)
class WaiterView:
    """A waiter in a snapshot."""

    key: str
    waited_s: float


@dataclass(frozen=True)
class ProviderView:
    """The state of one provider's table: heads, holders, FIFO waiters."""

    name: str
    heads: int
    holders: tuple[HolderView, ...]
    waiters: tuple[WaiterView, ...]


@dataclass
class _Lease:
    """A key's head on one provider."""

    since: float
    # Calls in flight (or reserved by a grant) under this lease.
    in_call: int
    # When the last call under the lease ended; orders idle holders (LRU).
    last_end: float
    # Attribution of the last call that entered; tags yield/release events.
    attribution: dict
    # A cooperative yield waiting for the key's calls to end.
    yield_pending: bool = False


@dataclass
class _Waiter:
    """A key waiting for a head, FIFO ordered; concurrent calls share a future."""

    key: str
    since: float
    attribution: dict
    future: asyncio.Future
    calls: int = 0


class ProviderLeases:
    """The lease table of one bounded provider (see the module docstring)."""

    def __init__(self, manager: LeaseManager, name: str, heads: int) -> None:
        self._mgr = manager
        self.name = name
        self.heads = heads
        # key -> its lease; insertion order is grant order.
        self._leases: dict[str, _Lease] = {}
        # FIFO waiters, one entry per key.
        self._waiters: list[_Waiter] = []
        # Keys stopped here: never granted or passed through until resumed.
        self._stopped: set[str] = set()
        self._closed = False

    # -- the call path ------------------------------------------------------

    @contextlib.asynccontextmanager
    async def call(self, key: str, attribution: dict) -> AsyncIterator[float]:
        """Hold ``key``'s lease for one call; yields the wait in seconds.

        On exit (normal, raise or cancel) the key's in-call count drops and
        the policy runs: the call boundary.
        """
        started = self._mgr.monotonic()
        # Step 0: a closed provider takes no new lease shutdown would never release.
        if self._closed:
            raise self._shutdown_error()
        await self._acquire(key, attribution)
        try:
            yield self._mgr.monotonic() - started
        finally:
            self._end_call(key)
            self._settle()

    async def _acquire(self, key: str, attribution: dict) -> None:
        """Pass through, take a free head, or wait in the FIFO."""
        lease = self._leases.get(key)
        if lease is not None:
            # Step 1: a holder passes through while no other key waits.
            if key not in self._stopped and not self._eligible_waiter():
                lease.in_call += 1
                lease.attribution = attribution
                return
        elif key not in self._stopped and len(self._leases) < self.heads:
            # Step 2: a free head is taken without waiting.
            self._leases[key] = _Lease(
                self._mgr.monotonic(), 1, self._mgr.monotonic(), attribution
            )
            self._emit("lease.grant", 1.0, key, attribution)
            return

        # Step 3: join the key's shared waiter entry (or the FIFO tail).
        waiter = self._waiter(key)
        if waiter is None:
            waiter = _Waiter(
                key, self._mgr.monotonic(), attribution,
                asyncio.get_running_loop().create_future(),
            )
            self._waiters.append(waiter)
        waiter.calls += 1
        self._settle()
        try:
            # Shielded so one call's cancellation never cancels the key's others.
            await asyncio.shield(waiter.future)
        except BaseException:
            # Own cancel or shutdown failure: undo this call's part (R11).
            future = waiter.future
            if future.done() and not future.cancelled() and future.exception() is None:
                # Granted but never resumed: give the reservation back.
                self._end_call(key)
            else:
                waiter.calls -= 1
                if waiter.calls <= 0 and waiter in self._waiters:
                    self._waiters.remove(waiter)
            self._settle()
            raise

    def _end_call(self, key: str) -> None:
        """One call of ``key`` is over: drop its in-call count if the lease lives."""
        lease = self._leases.get(key)
        if lease is not None:
            lease.in_call = max(0, lease.in_call - 1)
            lease.last_end = self._mgr.monotonic()

    # -- the default policy -------------------------------------------------

    def _eligible_waiter(self) -> bool:
        """Whether a waiter exists that is not stopped and holds nothing here."""
        return any(
            w.key not in self._stopped and w.key not in self._leases
            for w in self._waiters
        )

    def _settle(self) -> None:
        """Re-run the policy until the table is stable (no-op once closed)."""
        if self._closed:
            return
        changed = True
        while changed:
            changed = False
            # Stop / cooperative: idle holders that must or chose to give way.
            for key, lease in list(self._leases.items()):
                if lease.in_call:
                    continue
                if key in self._stopped:
                    self._remove(key, "stop")
                    changed = True
                elif lease.yield_pending:
                    if self._eligible_waiter():
                        self._remove(key, "cooperative")
                        changed = True
                    else:
                        # Nobody is behind the yielder: the yield is moot.
                        lease.yield_pending = False
            # Free heads go to eligible waiters in FIFO order.
            changed |= self._grant()
            # Contention: an eligible waiter and no free head: the least
            # recently used idle holder gives way.
            if self._eligible_waiter():
                idle = [
                    (held.last_end, k)
                    for k, held in self._leases.items()
                    if not held.in_call
                ]
                if idle:
                    self._remove(min(idle)[1], "contention")
                    changed = True
            # Admission: a holder's parked calls run once pass-through holds.
            if not self._eligible_waiter():
                for waiter in list(self._waiters):
                    lease = self._leases.get(waiter.key)
                    if lease is not None and waiter.key not in self._stopped:
                        self._admit(waiter, lease)
                        changed = True

    def _grant(self) -> bool:
        """Give each free head to the first eligible waiter; True if any."""
        granted = False
        while len(self._leases) < self.heads:
            waiter = next(
                (w for w in self._waiters
                 if w.key not in self._stopped and w.key not in self._leases),
                None,
            )
            if waiter is None:
                break
            now = self._mgr.monotonic()
            self._waiters.remove(waiter)
            # Every call of the key waiting here is reserved as in-call at
            # once, so the policy cannot take the head back before they resume.
            self._leases[waiter.key] = _Lease(
                now, waiter.calls, now, waiter.attribution
            )
            self._emit("lease.grant", 1.0, waiter.key, waiter.attribution)
            self._emit_wait(waiter, now)
            waiter.future.set_result(None)
            granted = True
        return granted

    def _admit(self, waiter: _Waiter, lease: _Lease) -> None:
        """Let a holder's parked calls proceed on its existing lease."""
        self._waiters.remove(waiter)
        lease.in_call += waiter.calls
        lease.attribution = waiter.attribution
        self._emit_wait(waiter, self._mgr.monotonic())
        waiter.future.set_result(None)

    def _remove(self, key: str, reason: YieldReason) -> None:
        """Take ``key``'s head away; its waiter entry (if any) keeps its place."""
        lease = self._leases.pop(key)
        self._emit("lease.yield", 1.0, key, lease.attribution, reason=reason)

    # -- views and helpers --------------------------------------------------

    def _waiter(self, key: str) -> _Waiter | None:
        """The waiter entry of ``key``, if it has one."""
        return next((w for w in self._waiters if w.key == key), None)

    def _shutdown_error(self) -> RuntimeError:
        """The failure a call gets when the provider is shutting down."""
        return RuntimeError(
            f"LLM provider {self.name} is shutting down; the call was not sent"
        )

    def _emit_wait(self, waiter: _Waiter, now: float) -> None:
        """Emit ``lease.wait_ms`` for a waiter entry that was just resolved."""
        self._emit(
            "lease.wait_ms", (now - waiter.since) * 1000.0,
            waiter.key, waiter.attribution,
        )

    def _emit(
        self, name: str, value: float, key: str, attribution: dict, **extra: str
    ) -> None:
        """Emit one event tagged like the call it affects."""
        tags = {
            "provider": self.name,
            "lease_key": key,
            "stage": attribution.get("stage") or "",
            "channel": attribution.get("channel_id") or "",
            "trigger": attribution.get("trigger") or "",
            **extra,
        }
        self._mgr.emit(name, value, tags)

    def snapshot(self) -> ProviderView:
        """Heads, holders (idle included) and FIFO waiters with ages; no I/O."""
        now = self._mgr.monotonic()
        return ProviderView(
            name=self.name,
            heads=self.heads,
            holders=tuple(
                HolderView(k, now - lease.since, lease.in_call > 0)
                for k, lease in self._leases.items()
            ),
            waiters=tuple(WaiterView(w.key, now - w.since) for w in self._waiters),
        )

    def close(self) -> None:
        """Release every lease (reason shutdown) and fail every waiting call.

        Safe to call twice: the second call finds nothing left.
        """
        self._closed = True
        for key, lease in list(self._leases.items()):
            del self._leases[key]
            held_ms = (self._mgr.monotonic() - lease.since) * 1000.0
            self._emit(
                "lease.release", held_ms, key, lease.attribution, reason="shutdown"
            )
        for waiter in self._waiters:
            if not waiter.future.done():
                waiter.future.set_exception(self._shutdown_error())
        self._waiters.clear()


class LeaseManager:
    """The lease tables of the bounded providers and the stop/resume/yield controls."""

    def __init__(
        self,
        heads: Mapping[str, int],
        *,
        monotonic: Callable[[], float],
        emit: Callable[[str, float, dict[str, str]], None] | None,
    ) -> None:
        self.monotonic = monotonic
        self._emit_fn = emit
        self._providers = {
            name: ProviderLeases(self, name, count) for name, count in heads.items()
        }

    def provider(self, name: str) -> ProviderLeases | None:
        """The table of provider ``name``; None means unbounded."""
        return self._providers.get(name)

    def emit(self, name: str, value: float, tags: dict[str, str]) -> None:
        """Pass one event to the emitter; a failing emitter never breaks leasing."""
        if self._emit_fn is None:
            return
        try:
            self._emit_fn(name, value, tags)
        except Exception:
            logger.warning(
                "lease metric %s for key %s on provider %s was not emitted",
                name, tags.get("lease_key"), tags.get("provider"), exc_info=True,
            )

    def _bounded(self, provider: str) -> ProviderLeases:
        """The table of ``provider``, or a ValueError naming the accepted ones."""
        table = self._providers.get(provider)
        if table is None:
            raise ValueError(
                f"provider {provider!r} has no heads, so it has no leases; "
                f"bounded providers: {', '.join(sorted(self._providers)) or '(none)'}"
            )
        return table

    def snapshot(self, provider: str) -> ProviderView:
        """The view of ``provider``'s table."""
        return self._bounded(provider).snapshot()

    def stop(self, provider: str, key: str) -> None:
        """Stop ``key`` on ``provider``: it gives up its head at a call boundary
        and its work waits until ``resume`` (R8)."""
        table = self._bounded(provider)
        table._stopped.add(key)
        table._settle()

    def resume(self, provider: str, key: str) -> None:
        """Undo ``stop``: the key's parked work is admitted or granted."""
        table = self._bounded(provider)
        table._stopped.discard(key)
        table._settle()

    def yield_key(self, key: str) -> None:
        """Cooperative yield on every provider where ``key`` holds a head."""
        for table in self._providers.values():
            lease = table._leases.get(key)
            if lease is None:
                continue
            lease.yield_pending = True
            # The yielder's parked calls go to the end of the line (R9m).
            waiter = table._waiter(key)
            if waiter is not None:
                table._waiters.remove(waiter)
                table._waiters.append(waiter)
            table._settle()

    def close(self) -> None:
        """Release every lease and fail every waiter (shutdown, R11)."""
        for table in self._providers.values():
            table.close()
