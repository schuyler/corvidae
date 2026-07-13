"""RED tests for WP2.6 — Funnel deferred registration + per-origin stub coalescing.

The §2.2 routing rule (plans/implementation/phase-2.md WP2.6): non-tool_call_id
notification payloads register with the funnel per (channel.id, origin); one
stub notification wakes the channel per pending pair; the drain in
FunnelPlugin.before_agent_turn admits everything queued for the triggering
exchange's origin. Per-origin coalescing is a correctness point, not an
optimization — coalescing a critique verdict into another origin's stub would
make the verdict-responding turn critique-eligible (recursion reopened).

Covers the plan's red-test list:
- Three registrations before any drain → exactly one stub (spy on on_notify),
  one drain admits all three.
- Payloads of origin A do not drain on an origin-B turn or a user turn.
- Admission failure (monkeypatched admit raising) → payloads still
  registered; next register_and_wake fires a fresh stub.
- Budget-dropped entries survive to the next drain.
"""

from unittest.mock import AsyncMock

import aiosqlite
import pytest

from corvidae.channel import Channel
from corvidae.context import ContextWindow, MessageType
from corvidae.funnel import FunnelPlugin
from corvidae.hooks import create_plugin_manager
from corvidae.persistence import PersistencePlugin, init_db


async def build_funnel(config: dict | None = None):
    """Register persistence + funnel on a fresh pm with an in-memory DB.

    Returns (funnel, channel, conv, db). on_notify is replaced with an
    AsyncMock spy so stub firings are observable without an Agent.
    """
    db = await aiosqlite.connect(":memory:")
    await init_db(db)

    pm = create_plugin_manager()
    persistence = PersistencePlugin()
    persistence.db = db
    pm.register(persistence, name="persistence")

    funnel = FunnelPlugin()
    pm.register(funnel, name="funnel")
    await funnel.on_init(pm=pm, config=config or {})

    pm.ahook.on_notify = AsyncMock()

    channel = Channel(transport="test", scope="funnel-deferred")
    conv = ContextWindow(channel.id)
    channel.conversation = conv
    return funnel, channel, conv, db


def _window_context_text(conv: ContextWindow) -> str:
    """All CONTEXT message content concatenated, for containment asserts."""
    return "\n".join(
        msg.get("content") or ""
        for msg in conv.messages
        if msg.get("_message_type") == MessageType.CONTEXT
    )


class TestStubCoalescing:
    async def test_three_registrations_one_stub_one_drain_admits_all(self):
        """Multiple registrations for one (channel, origin) pair before any
        drain coalesce into exactly one stub; the drain admits all queued
        payloads."""
        funnel, channel, conv, db = await build_funnel()
        pm = funnel.pm

        await funnel.register_and_wake(channel, origin="critique", source="critique", entries=["verdict one"])
        await funnel.register_and_wake(channel, origin="critique", source="critique", entries=["verdict two"])
        await funnel.register_and_wake(channel, origin="critique", source="critique", entries=["verdict three"])

        # Exactly one stub for the pair — later registrations only queue.
        pm.ahook.on_notify.assert_awaited_once()
        kwargs = pm.ahook.on_notify.await_args.kwargs
        assert kwargs["channel"] is channel
        assert kwargs["source"] == "critique"
        assert kwargs["tool_call_id"] is None
        assert kwargs["meta"] == {"origin": "critique"}

        # The drain on a critique-origin turn admits all three.
        await funnel.before_agent_turn(channel=channel, exchange_key="k1", origin="critique")
        window = _window_context_text(conv)
        assert "verdict one" in window
        assert "verdict two" in window
        assert "verdict three" in window

        # Registry drained: a subsequent same-origin turn admits nothing new.
        msg_count = len(conv.messages)
        await funnel.before_agent_turn(channel=channel, exchange_key="k2", origin="critique")
        assert len(conv.messages) == msg_count

        await db.close()

    async def test_stub_refires_after_drain(self):
        """Once drained, the pending flag is cleared — a new registration
        fires a fresh stub."""
        funnel, channel, conv, db = await build_funnel()
        pm = funnel.pm

        await funnel.register_and_wake(channel, origin="critique", source="critique", entries=["first"])
        await funnel.before_agent_turn(channel=channel, exchange_key="k1", origin="critique")

        await funnel.register_and_wake(channel, origin="critique", source="critique", entries=["second"])
        assert pm.ahook.on_notify.await_count == 2

        await db.close()

    async def test_stub_text_carries_pending_count(self):
        funnel, channel, conv, db = await build_funnel()
        pm = funnel.pm

        await funnel.register_and_wake(
            channel, origin="critique", source="critique", entries=["a", "b"]
        )
        text = pm.ahook.on_notify.await_args.kwargs["text"]
        assert "2" in text
        assert "critique" in text

        await db.close()

    async def test_empty_entries_is_a_noop(self):
        """Registering nothing queues nothing and wakes nobody."""
        funnel, channel, conv, db = await build_funnel()
        pm = funnel.pm

        await funnel.register_and_wake(channel, origin="critique", source="critique", entries=[])
        pm.ahook.on_notify.assert_not_awaited()

        await funnel.before_agent_turn(channel=channel, exchange_key="k1", origin="critique")
        assert _window_context_text(conv) == ""

        await db.close()


class TestPerOriginIsolation:
    async def test_origin_a_payloads_do_not_drain_on_origin_b_turn(self):
        """The recursion brake: critique payloads must not drain into a
        task-origin turn (§2.2 correctness point)."""
        funnel, channel, conv, db = await build_funnel()

        await funnel.register_and_wake(channel, origin="critique", source="critique", entries=["a verdict"])

        await funnel.before_agent_turn(channel=channel, exchange_key="k1", origin="task")
        assert "a verdict" not in _window_context_text(conv)

        # Still registered — the matching-origin turn drains it.
        await funnel.before_agent_turn(channel=channel, exchange_key="k2", origin="critique")
        assert "a verdict" in _window_context_text(conv)

        await db.close()

    async def test_origin_payloads_do_not_drain_on_user_turn(self):
        funnel, channel, conv, db = await build_funnel()

        await funnel.register_and_wake(channel, origin="critique", source="critique", entries=["a verdict"])

        await funnel.before_agent_turn(channel=channel, exchange_key="k1", origin="user")
        assert "a verdict" not in _window_context_text(conv)

        await db.close()

    async def test_channels_are_isolated(self):
        """Payloads queued on one channel never drain on another channel's
        same-origin turn."""
        funnel, channel_a, conv_a, db = await build_funnel()
        channel_b = Channel(transport="test", scope="funnel-deferred-b")
        conv_b = ContextWindow(channel_b.id)
        channel_b.conversation = conv_b

        await funnel.register_and_wake(channel_a, origin="critique", source="critique", entries=["a verdict"])

        await funnel.before_agent_turn(channel=channel_b, exchange_key="k1", origin="critique")
        assert "a verdict" not in _window_context_text(conv_b)
        # Still available for channel A.
        await funnel.before_agent_turn(channel=channel_a, exchange_key="k2", origin="critique")
        assert "a verdict" in _window_context_text(conv_a)

        await db.close()


class TestFailureAndBudget:
    async def test_admission_failure_keeps_payloads_and_rearms_stub(self, monkeypatch):
        """A failure inside admission leaves payloads registered; because
        the pending flag was cleared FIRST, the next producer's stub
        re-arms the channel instead of wedging it."""
        funnel, channel, conv, db = await build_funnel()
        pm = funnel.pm

        await funnel.register_and_wake(channel, origin="critique", source="critique", entries=["kept verdict"])
        assert pm.ahook.on_notify.await_count == 1

        async def boom(*args, **kwargs):
            raise RuntimeError("admission exploded")

        monkeypatch.setattr(funnel, "admit", boom)
        # Drain attempt fails inside admit — must not raise out of the hook,
        # and must not lose the payload.
        await funnel.before_agent_turn(channel=channel, exchange_key="k1", origin="critique")
        monkeypatch.undo()

        # A fresh registration fires a fresh stub (flag was cleared first).
        await funnel.register_and_wake(channel, origin="critique", source="critique", entries=["second verdict"])
        assert pm.ahook.on_notify.await_count == 2

        # The next successful drain admits BOTH the kept and the new payload.
        await funnel.before_agent_turn(channel=channel, exchange_key="k2", origin="critique")
        window = _window_context_text(conv)
        assert "kept verdict" in window
        assert "second verdict" in window

        await db.close()

    async def test_budget_dropped_entries_survive_to_next_drain(self):
        """Entries the budget dropped stay registered for the next stub
        (§2.2); entries admitted unregister."""
        # Budget sized so only the first entry fits per drain: each entry
        # counts 10 tokens ("alpha" is 1 token/word, "bravo" is 2), so the
        # 12-token budget admits one entry per admit() call but never both.
        funnel, channel, conv, db = await build_funnel(
            config={"funnel": {"budgets": {"critique": 12}}}
        )

        long_a = "alpha " * 10
        long_b = "bravo " * 5
        await funnel.register_and_wake(
            channel, origin="critique", source="critique", entries=[long_a.strip(), long_b.strip()]
        )

        await funnel.before_agent_turn(channel=channel, exchange_key="k1", origin="critique")
        window = _window_context_text(conv)
        assert "alpha" in window
        assert "bravo" not in window

        # The dropped entry drains on the next same-origin turn.
        await funnel.before_agent_turn(channel=channel, exchange_key="k2", origin="critique")
        window = _window_context_text(conv)
        assert "bravo" in window

        await db.close()


class TestMidDrainRegistrationWedge:
    """2B review-gate MF-1 regression: a payload registered MID-DRAIN
    (after the pending flag was discarded) is admitted by the in-progress
    drain, so its own stub's turn arrives with an empty registry. That turn
    must still clear the pending flag — otherwise the pair is wedged: every
    later register_and_wake sees the stale flag and never fires a stub, and
    critique-origin turns only originate from stubs."""

    async def test_mid_drain_registration_does_not_wedge_the_pair(self):
        funnel, channel, conv, db = await build_funnel()
        pm = funnel.pm

        # Interpose on the persistence hook admit() awaits, so a second
        # registration lands exactly mid-drain (flag already discarded).
        orig_event = pm.ahook.on_conversation_event
        fired = {"done": False}

        async def registering_event(**kwargs):
            if not fired["done"]:
                fired["done"] = True
                await funnel.register_and_wake(
                    channel, origin="critique", source="critique", entries=["payload B"]
                )
            return await orig_event(**kwargs)

        pm.ahook.on_conversation_event = registering_event

        # Stub 1 fires for A.
        await funnel.register_and_wake(
            channel, origin="critique", source="critique", entries=["payload A"]
        )
        assert pm.ahook.on_notify.await_count == 1

        # Stub 1's turn drains; B registers mid-drain (stub 2 fires) and is
        # consumed by this same drain.
        await funnel.before_agent_turn(channel=channel, exchange_key="k1", origin="critique")
        assert pm.ahook.on_notify.await_count == 2
        window = _window_context_text(conv)
        assert "payload A" in window
        assert "payload B" in window

        pm.ahook.on_conversation_event = orig_event

        # Stub 2's turn arrives with an empty registry — it must clear the
        # stale flag, not early-return past it.
        await funnel.before_agent_turn(channel=channel, exchange_key="k2", origin="critique")

        # A fresh registration must fire a fresh stub (count 3). Before the
        # fix this stayed at 2 and payload C sat undeliverable forever.
        await funnel.register_and_wake(
            channel, origin="critique", source="critique", entries=["payload C"]
        )
        assert pm.ahook.on_notify.await_count == 3
        await funnel.before_agent_turn(channel=channel, exchange_key="k3", origin="critique")
        assert "payload C" in _window_context_text(conv)

        await db.close()
