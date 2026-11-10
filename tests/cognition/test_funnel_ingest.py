"""RED tests for #717: funnel ``ingest``/``ingest_and_wake``, the ``on_funnel_ingest``
promise hook, ``funnel.ingest_ttl`` expiry, shutdown and TTL config.

Spec: .claude/plans/717-funnel-ingest.md, Test plan R1-R8. ``Ingest`` and
``FunnelSpec`` are reached through ``funnel_mod`` so this module imports before
they exist; R6's absent-funnel test passes at Red.
"""

import asyncio
import logging
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiosqlite
import pytest

import corvidae_cognition.funnel as funnel_mod
from corvidae.channel import Channel
from corvidae.context import ContextWindow, MessageType
from corvidae.hooks import (
    CorvidaePlugin, create_plugin_manager, hookimpl,
    install_plugin_hookspecs, report_orphan_hookimpls,
)
from corvidae.persistence import PersistencePlugin, init_db
from corvidae_cognition.funnel import Admission, FunnelPlugin

# ALPHA and BRAVO are 10 tokens each (count_tokens): under a 12-token budget only one fits per drain.
ALPHA = ("alpha " * 10).strip()
BRAVO = ("bravo " * 5).strip()


class Clock:
    """Fake monotonic clock; time moves only when a test sets ``t``."""

    def __init__(self) -> None:
        # Current fake monotonic time, in seconds.
        self.t = 0.0

    def __call__(self) -> float:
        # The funnel reads time by calling the clock.
        return self.t


class Provider(CorvidaePlugin):
    """Test implementer of ``on_funnel_ingest``: submits each queued offer once."""

    # Lists the funnel as optional, as the plugin guide asks of implementers.
    optional_depends_on = frozenset({"funnel"})

    def __init__(self) -> None:
        # Queued (channel id, source, future) offers, and how often collected.
        self.offers: list[tuple[str, str, asyncio.Future]] = []
        self.calls = 0
        # Raise after submitting, to test a sibling implementation's submission.
        self.fail_after_submit = False

    def offer(self, channel_id: str, source: str, future: asyncio.Future) -> None:
        # Queue ``future`` for ``source``; it is submitted on the channel's next collect.
        self.offers.append((channel_id, source, future))

    @hookimpl
    async def on_funnel_ingest(self, channel, trace_id, annotations, submit) -> None:
        # Submit this channel's offers, then drop them, so each future goes in once.
        self.calls += 1
        mine = [o for o in self.offers if o[0] == channel.id]
        self.offers = [o for o in self.offers if o[0] != channel.id]
        for _, source, future in mine:
            submit(source, future)
        if self.fail_after_submit:
            raise RuntimeError("provider exploded after submitting")


def pending() -> asyncio.Future:
    # A promise nobody has resolved yet.
    return asyncio.get_running_loop().create_future()


def resolved(*entries: str, on_admitted=None) -> asyncio.Future:
    # A promise already resolved to an Ingest of ``entries``.
    future = pending()
    future.set_result(funnel_mod.Ingest(entries=list(entries), on_admitted=on_admitted))
    return future


def text_of(conv: ContextWindow) -> str:
    # All CONTEXT content in ``conv``, for containment asserts.
    return "\n".join(
        m.get("content") or "" for m in conv.messages
        if m.get("_message_type") == MessageType.CONTEXT
    )


def logged(caplog, level: int = logging.WARNING) -> list[str]:
    # Log messages at ``level`` or above.
    return [r.getMessage() for r in caplog.records if r.levelno >= level]


async def make(config: dict | None = None, clock: Clock | None = None) -> SimpleNamespace:
    # A funnel on a real plugin manager, built as Runtime builds it: persistence,
    # funnel and provider registered, specs installed, one channel, in-memory DB.
    db = await aiosqlite.connect(":memory:")
    await init_db(db)
    pm = create_plugin_manager()
    persistence = PersistencePlugin()
    persistence.db = db
    pm.register(persistence, name="persistence")
    clock = clock or Clock()
    funnel = FunnelPlugin(now=clock)
    pm.register(funnel, name="funnel")
    provider = Provider()
    pm.register(provider, name="provider")
    install_plugin_hookspecs(pm)
    await funnel.on_init(pm=pm, config=config or {})
    pm.ahook.on_notify = AsyncMock()
    channel = Channel(transport="test", scope="funnel-ingest")
    channel.conversation = ContextWindow(channel.id)
    return SimpleNamespace(
        pm=pm, funnel=funnel, provider=provider, channel=channel,
        conv=channel.conversation, clock=clock, db=db,
    )


async def turn(env: SimpleNamespace, channel=None, origin: str = "user", annotations=None) -> None:
    # One agent turn the funnel must collect on; ``annotations`` overrides the origin.
    channel = channel or env.channel
    before = env.provider.calls
    notes = {"origin": origin} if annotations is None else annotations
    await env.funnel.before_agent_turn(channel=channel, trace_id="t", annotations=notes)
    assert env.provider.calls == before + 1, "on_funnel_ingest was not called on this turn"


@pytest.fixture
async def build():
    # Factory for make(); closes the databases it opened.
    opened: list[aiosqlite.Connection] = []

    async def _build(config: dict | None = None, clock: Clock | None = None):
        env = await make(config, clock)
        opened.append(env.db)
        return env

    yield _build
    for db in opened:
        await db.close()


class TestRename:

    def test_old_names_are_removed_and_new_names_exist(self):
        assert not hasattr(FunnelPlugin, "register")
        assert not hasattr(FunnelPlugin, "register_and_wake")
        assert callable(FunnelPlugin.ingest) and callable(FunnelPlugin.ingest_and_wake)


class TestPromiseAdmission:

    async def test_promise_is_admitted_framed_at_the_first_turn_after_it_resolves(self, build):
        env = await build()
        future = pending()
        env.provider.offer(env.channel.id, "memory", future)
        await turn(env)
        await turn(env, origin="user")
        assert not future.done() and "late entry" not in text_of(env.conv)
        future.set_result(funnel_mod.Ingest(entries=["late entry"]))
        # A turn with no origin annotation still sweeps and admits the ready result.
        await turn(env, annotations={})
        assert funnel_mod.FRAME_HEADER.format(source="memory") in text_of(env.conv) and "late entry" in text_of(env.conv)

    async def test_none_result_admits_nothing(self, build):
        env = await build()
        future = pending()
        env.provider.offer(env.channel.id, "memory", future)
        await turn(env)
        future.set_result(None)
        await turn(env)
        assert text_of(env.conv) == ""

    @pytest.mark.parametrize("how", ["swept", "expired"])
    async def test_raising_promise_is_logged_with_source_and_channel_and_dropped(self, build, caplog, how):
        env = await build(config={"funnel": {"ingest_ttl": 10}})
        future = pending()
        env.provider.offer(env.channel.id, "memory", future)
        await turn(env)
        future.set_exception(RuntimeError("provider exploded"))
        if how == "expired":
            env.clock.t = 20.0
        with caplog.at_level(logging.WARNING, logger="corvidae.funnel"):
            await turn(env)
        assert text_of(env.conv) == ""
        warned = [r for r in caplog.records if r.levelno == logging.WARNING and "memory" in r.getMessage() and env.channel.id in r.getMessage()]
        assert warned and warned[0].exc_info and warned[0].exc_info[0] is RuntimeError

    async def test_a_raising_implementation_does_not_lose_a_sibling_submission(self, build, caplog):
        env = await build()
        raiser = Provider()
        raiser.fail_after_submit = True
        env.pm.register(raiser, name="raiser")
        raiser.offer(env.channel.id, "broken", pending())
        env.provider.offer(env.channel.id, "memory", resolved("sibling entry"))
        with caplog.at_level(logging.WARNING, logger="corvidae.funnel"):
            await turn(env)
        await turn(env)
        assert "sibling entry" in text_of(env.conv)
        assert any(env.channel.id in m for m in logged(caplog))

    async def test_oversize_promise_entry_is_dropped_with_warning_naming_the_budget_key(self, build, caplog):
        env = await build(config={"funnel": {"budgets": {"memory": 12}}})
        env.provider.offer(env.channel.id, "memory", resolved(("charlie " * 30).strip(), "small entry"))
        with caplog.at_level(logging.WARNING, logger="corvidae.funnel"):
            await turn(env)
            await turn(env)
        assert "small entry" in text_of(env.conv) and "charlie" not in text_of(env.conv)
        assert any(
            env.channel.id in m and "memory" in m and "funnel.budgets.memory" in m
            for m in logged(caplog)
        )

    async def test_callback_gets_the_admission_including_duplicates(self, build):
        env = await build()
        env.funnel.ingest(env.channel, "memory", ["alpha"])
        await turn(env)
        seen: list[Admission] = []
        env.provider.offer(env.channel.id, "memory", resolved("alpha", "bravo", on_admitted=seen.append))
        await turn(env)
        await turn(env)
        assert seen == [Admission(admitted=["bravo"], duplicate=["alpha"])]

    async def test_raising_callback_is_logged_and_the_drain_and_collect_go_on(self, build, caplog):
        env = await build()
        calls: list[Admission] = []

        def boom(admission) -> None:
            # Record each invocation, then fail.
            calls.append(admission)
            raise RuntimeError("callback exploded")

        env.provider.offer(env.channel.id, "memory", resolved("entry one", on_admitted=boom))
        env.provider.offer(env.channel.id, "grounding", resolved("entry two"))
        await turn(env)
        with caplog.at_level(logging.WARNING, logger="corvidae.funnel"):
            await turn(env)
        await turn(env)
        # Both entries admitted; the callback ran once, so the payload was not re-drained.
        assert "entry one" in text_of(env.conv) and "entry two" in text_of(env.conv) and len(calls) == 1
        assert any(env.channel.id in m and "memory" in m for m in logged(caplog))


class TestOnePendingPerSource:

    async def test_newer_submit_cancels_the_older_pending_one(self, build):
        env = await build()
        older, newer = pending(), pending()
        env.provider.offer(env.channel.id, "memory", older)
        await turn(env)
        env.provider.offer(env.channel.id, "memory", newer)
        await turn(env)
        assert older.cancelled()
        newer.set_result(funnel_mod.Ingest(entries=["newer entry"]))
        await turn(env)
        assert "newer entry" in text_of(env.conv)

    async def test_different_sources_and_channels_are_kept_apart(self, build):
        env = await build()
        other = Channel(transport="test", scope="funnel-ingest-b")
        other.conversation = ContextWindow(other.id)
        mem, grd, other_mem = pending(), pending(), pending()
        env.provider.offer(env.channel.id, "memory", mem)
        env.provider.offer(env.channel.id, "grounding", grd)
        env.provider.offer(other.id, "memory", other_mem)
        await turn(env)
        await turn(env, channel=other)
        assert not (mem.cancelled() or grd.cancelled() or other_mem.cancelled())
        mem.set_result(funnel_mod.Ingest(entries=["mem entry"]))
        grd.set_result(funnel_mod.Ingest(entries=["grd entry"]))
        other_mem.set_result(funnel_mod.Ingest(entries=["other entry"]))
        await turn(env)
        await turn(env, channel=other)
        assert "mem entry" in text_of(env.conv) and "grd entry" in text_of(env.conv)
        assert "other entry" in text_of(other.conversation)
        assert "other entry" not in text_of(env.conv)

    @pytest.mark.parametrize("how", ["superseded", "expired", "swept", "stopped"])
    async def test_a_promise_its_provider_cancelled_is_dropped_without_raising(self, build, how):
        env = await build(config={"funnel": {"ingest_ttl": 10}})
        cid = env.channel.id
        first = pending()
        env.provider.offer(cid, "memory", first)
        await turn(env)
        first.cancel()
        if how == "superseded":
            # A newer submit for the same source takes the cancelled one's slot.
            env.provider.offer(cid, "memory", pending())
            await turn(env)
        elif how == "expired":
            env.clock.t = 20.0
            await turn(env)
        elif how == "swept":
            await turn(env)
        else:
            await env.funnel.on_stop()
        if how != "stopped":
            # The funnel still admits a fresh promise afterwards.
            probe = pending()
            env.provider.offer(cid, "probe", probe)
            await turn(env)
            probe.set_result(funnel_mod.Ingest(entries=["probe entry"]))
            await turn(env)
            assert "probe entry" in text_of(env.conv)


class TestIngestTtl:

    async def test_unset_ttl_expires_nothing(self, build):
        env = await build()
        waiting = pending()
        env.provider.offer(env.channel.id, "grounding", waiting)
        await turn(env)
        env.funnel.ingest(env.channel, "memory", ["queued entry"])
        env.clock.t = 1_000_000.0
        await turn(env)
        assert "queued entry" in text_of(env.conv) and not waiting.done()

    async def test_expired_queued_and_pending_entries_are_dropped_and_logged_with_age(self, build, caplog):
        # Slow (submitted at t=0) and stale (queued at t=0) expire at t=17; fresh (queued at t=15) stays.
        env = await build(config={"funnel": {"ingest_ttl": 10}})
        slow = pending()
        env.provider.offer(env.channel.id, "grounding", slow)
        await turn(env)
        env.funnel.ingest(env.channel, "memory", ["stale entry"])
        env.clock.t = 15.0
        env.funnel.ingest(env.channel, "memory", ["fresh entry"])
        env.clock.t = 17.0
        with caplog.at_level(logging.INFO, logger="corvidae.funnel"):
            await turn(env)
        text = text_of(env.conv)
        assert "stale entry" not in text and "fresh entry" in text and slow.cancelled()
        # One expiry line per dropped item, each naming its own source, the channel, the age and the key.
        expiry = [m for m in logged(caplog, logging.INFO) if "funnel.ingest_ttl" in m]
        assert len(expiry) == 2, expiry
        for source in ("memory", "grounding"):
            assert sum(source in m and env.channel.id in m and re.search(r"(?<![\d.])17(\.\d+)?(?!\d)", m) is not None for m in expiry) == 1, expiry

    async def test_promise_finished_while_idle_past_ttl_is_dropped_not_admitted(self, build):
        # Submitted at t=0, resolved at t=5, swept at t=12: age from submit is 12.
        env = await build(config={"funnel": {"ingest_ttl": 10}})
        late = pending()
        env.provider.offer(env.channel.id, "memory", late)
        await turn(env)
        env.clock.t = 5.0
        late.set_result(funnel_mod.Ingest(entries=["idle entry"]))
        env.clock.t = 12.0
        await turn(env)
        assert "idle entry" not in text_of(env.conv)

    async def test_promise_payload_requeued_by_the_budget_keeps_its_submit_age(self, build):
        # BRAVO is re-queued at t=6 by the budget; it must still expire at t=12.
        env = await build(config={"funnel": {"ingest_ttl": 10, "budgets": {"memory": 12}}})
        slow = pending()
        env.provider.offer(env.channel.id, "memory", slow)
        await turn(env)
        env.clock.t = 5.0
        slow.set_result(funnel_mod.Ingest(entries=[ALPHA, BRAVO]))
        env.clock.t = 6.0
        await turn(env)
        assert "alpha" in text_of(env.conv) and "bravo" not in text_of(env.conv)
        env.clock.t = 12.0
        await turn(env)
        assert "bravo" not in text_of(env.conv)

    async def test_ingest_and_wake_payload_never_expires(self, build):
        env = await build(config={"funnel": {"ingest_ttl": 10}})
        await env.funnel.ingest_and_wake(
            env.channel, origin="critique", source="critique", entries=["verdict"],
        )
        env.clock.t = 100.0
        await turn(env, origin="critique")
        assert "verdict" in text_of(env.conv)


class TestShutdown:

    async def test_on_stop_cancels_pending_promises(self, build):
        env = await build()
        waiting = pending()
        env.provider.offer(env.channel.id, "memory", waiting)
        await turn(env)
        await env.funnel.on_stop()
        assert waiting.cancelled()


class TestHookSpec:

    async def test_registered_funnel_installs_the_spec_and_collects_the_implementer(self, build):
        env = await build()
        assert getattr(env.pm.hook, "on_funnel_ingest").spec is not None
        await turn(env)

    async def test_absent_funnel_leaves_the_implementer_inert_and_startup_clean(self, caplog):
        # Characterises kernel behaviour that already holds: passes at Red.
        pm = create_plugin_manager()
        pm.register(Provider(), name="provider")
        with caplog.at_level(logging.DEBUG):
            install_plugin_hookspecs(pm)
            report_orphan_hookimpls(pm)
        # Reported at DEBUG only, because the implementer lists funnel as optional.
        mentions = [r for r in caplog.records if "on_funnel_ingest" in r.getMessage()]
        assert mentions and all(r.levelno == logging.DEBUG for r in mentions)


class TestIngestTtlConfig:

    @pytest.mark.parametrize("bad", [0, -1, True, "5"])
    async def test_invalid_value_warns_and_leaves_expiry_off(self, build, caplog, bad):
        with caplog.at_level(logging.WARNING, logger="corvidae.funnel"):
            env = await build(config={"funnel": {"ingest_ttl": bad}})
        # The warning names the key, the rejected value and the accepted form.
        assert any(
            "funnel.ingest_ttl" in m and repr(bad) in m and re.search(r"greater than 0|positive", m)
            for m in logged(caplog)
        ), logged(caplog)
        # Expiry is off: a payload queued long before the turn is still admitted.
        env.funnel.ingest(env.channel, "memory", ["old entry"])
        env.clock.t = 1000.0
        await turn(env)
        assert "old entry" in text_of(env.conv)

    async def test_ingest_ttl_is_reloadable_and_a_bad_reload_turns_expiry_off(self, build):
        assert "funnel.ingest_ttl" in FunnelPlugin.reloadable_keys
        env = await build()
        await env.funnel.on_config_reload(config={"funnel": {"ingest_ttl": 10}})
        env.funnel.ingest(env.channel, "memory", ["first"])
        env.clock.t = 20.0
        await turn(env)
        assert "first" not in text_of(env.conv)
        await env.funnel.on_config_reload(config={"funnel": {"ingest_ttl": 0}})
        env.funnel.ingest(env.channel, "memory", ["second"])
        env.clock.t = 1000.0
        await turn(env)
        assert "second" in text_of(env.conv)
