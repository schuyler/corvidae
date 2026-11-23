"""Tests for #718 / #1176: the shared recall-source mixin (``corvidae_cognition.recall``).

Covers ``SearchGate`` (cooldown and single-flight on an injected clock),
``parse_recall_settings`` (rejected-value messages), and the mixin's
``before_compose`` on a real plugin manager with a fake source. The mixin
answers the kernel's before-compose gather with a ``Contribution`` (what that
contribution holds for a real folder is pinned by ``test_recall_contribution.py``);
these tests call ``before_compose`` as the gather does, with ``channel``,
``trace_id`` and ``annotations`` keywords. ``recall`` is imported lazily through
``recall_mod``.
"""

import asyncio
import logging
from types import SimpleNamespace
from typing import Any

import pytest

from corvidae.channel import Channel
from corvidae.context import ContextWindow, MessageType
from corvidae.held import Contribution
from corvidae.hooks import (
    CorvidaePlugin, create_plugin_manager, hookimpl,
    install_plugin_hookspecs,
)


def recall_mod():
    # Imported on use, so a missing module fails each test rather than collection.
    import corvidae_cognition.recall as recall
    return recall


class Clock:
    """Fake monotonic clock; time moves only when a test sets ``t``."""

    def __init__(self) -> None:
        # Current fake time, in seconds.
        self.t = 0.0

    def __call__(self) -> float:
        # Callers read time by calling the clock.
        return self.t


class FakeSource:
    """A recall source built on the mixin; its search is whatever the test sets."""

    def __new__(cls, *args: Any, **kwargs: Any):
        # Build the real class lazily, because the mixin only exists after Green.
        recall = recall_mod()

        class _Source(recall.RecallSourceMixin, CorvidaePlugin):
            source_name = "fake"
            config_key = "fake_recall"

            def __init__(self, clock: Clock) -> None:
                # Queries seen, the hits to return, and what to do instead of returning.
                self.clock = clock
                self.queries: list[str] = []
                self.hits: list[Any] | None = ["hit one"]
                self.raises: Exception | None = None
                self.gate_event: asyncio.Event | None = None
                self.sleep_forever = False
                self.cancelled = False

            @hookimpl
            async def on_init(self, pm, config: dict) -> None:
                # Store pm and config, then let the mixin read its settings and build its gate.
                await super().on_init(pm, config)
                self.init_recall(config, now=self.clock)

            async def search(self, query: str, ctx: Any) -> list[Any] | None:
                # Record the query, then block, hang, raise or answer as the test asked.
                self.queries.append(query)
                try:
                    if self.gate_event is not None:
                        await self.gate_event.wait()
                    if self.sleep_forever:
                        await asyncio.sleep(3600)
                except asyncio.CancelledError:
                    self.cancelled = True
                    raise
                if self.raises is not None:
                    raise self.raises
                return self.hits

            def render(self, hit: Any) -> str:
                # One line per hit.
                return f"rendered {hit}"

        return _Source(*args, **kwargs)


async def make(config: dict | None = None) -> SimpleNamespace:
    # A fake source on a real plugin manager, one channel.
    pm = create_plugin_manager()
    clock = Clock()
    source = FakeSource(clock)
    pm.register(source, name="fake_recall")
    install_plugin_hookspecs(pm)
    await source.on_init(pm=pm, config=config or {})
    channel = Channel(transport="test", scope="recall")
    channel.conversation = ContextWindow(channel.id)
    return SimpleNamespace(
        pm=pm, source=source, channel=channel, conv=channel.conversation, clock=clock,
    )


@pytest.fixture
async def build():
    # Factory for make().
    async def _build(config: dict | None = None):
        return await make(config)

    return _build


async def compose(env: SimpleNamespace) -> Contribution | None:
    # One turn's before_compose answer from the source, as the gather calls it.
    return await env.source.before_compose(
        channel=env.channel, trace_id="t", annotations={"origin": "user"}
    )


async def settle() -> None:
    # Let a started search run up to its first await.
    for _ in range(5):
        await asyncio.sleep(0)


def user(env: SimpleNamespace, text: str) -> None:
    # Append a user message to the channel's window.
    env.conv.append({"role": "user", "content": text})


def warnings(caplog) -> list[logging.LogRecord]:
    # Records at WARNING or above.
    return [r for r in caplog.records if r.levelno >= logging.WARNING]


class TestSearchGate:

    def test_second_start_is_refused_while_one_is_in_flight(self):
        gate = recall_mod().SearchGate(cooldown=0, now=Clock())
        assert gate.try_start("c1") is True
        assert gate.try_start("c1") is False

    def test_cooldown_blocks_until_it_has_elapsed_then_allows(self):
        clock = Clock()
        gate = recall_mod().SearchGate(cooldown=5, now=clock)
        assert gate.try_start("c1")
        gate.finish("c1")
        clock.t = 4.9
        assert gate.try_start("c1") is False
        clock.t = 5.0
        assert gate.try_start("c1") is True

    def test_channels_are_independent(self):
        gate = recall_mod().SearchGate(cooldown=5, now=Clock())
        assert gate.try_start("c1")
        assert gate.try_start("c2")

    def test_refused_start_does_not_restart_the_cooldown(self):
        clock = Clock()
        gate = recall_mod().SearchGate(cooldown=5, now=clock)
        gate.try_start("c1")
        gate.finish("c1")
        clock.t = 3.0
        assert gate.try_start("c1") is False
        clock.t = 5.0
        assert gate.try_start("c1") is True

    def test_finish_of_an_unknown_channel_is_harmless(self):
        recall_mod().SearchGate(cooldown=0, now=Clock()).finish("never-started")


class TestSettings:

    def test_no_block_gives_the_defaults(self):
        recall = recall_mod()
        s = recall.parse_recall_settings({}, "fake_recall")
        assert (s.cooldown, s.timeout) == (recall.DEFAULT_COOLDOWN, recall.DEFAULT_TIMEOUT)

    def test_values_are_read_from_the_named_block_only(self):
        # A top-level key of the same name is the whole-dict trap: it must be ignored.
        cfg = {"cooldown": 99, "timeout": 99, "fake_recall": {"cooldown": 2, "timeout": 3.5}}
        s = recall_mod().parse_recall_settings(cfg, "fake_recall")
        assert (s.cooldown, s.timeout) == (2.0, 3.5)
        top = recall_mod().parse_recall_settings({"cooldown": 99}, "fake_recall")
        assert top.cooldown == recall_mod().DEFAULT_COOLDOWN

    def test_zero_cooldown_is_allowed(self):
        assert recall_mod().parse_recall_settings({"x": {"cooldown": 0}}, "x").cooldown == 0

    @pytest.mark.parametrize("field,bad", [
        ("cooldown", -1), ("cooldown", "5"), ("cooldown", True), ("cooldown", None),
        ("timeout", 0), ("timeout", -2), ("timeout", "10"), ("timeout", False),
        ("timeout", None),
    ])
    def test_rejected_value_names_key_value_and_accepted_values(self, field, bad):
        with pytest.raises(ValueError) as err:
            recall_mod().parse_recall_settings({"fake_recall": {field: bad}}, "fake_recall")
        msg = str(err.value)
        # Key, rejected value, accepted values, and where to fix it.
        assert f"fake_recall.{field}" in msg and repr(bad) in msg
        assert "seconds" in msg and ("greater than 0" in msg or "0 or more" in msg)
        assert "agent.yaml" in msg

    def test_a_block_that_is_not_a_mapping_is_rejected_naming_the_key(self):
        with pytest.raises(ValueError) as err:
            recall_mod().parse_recall_settings({"fake_recall": ["cooldown"]}, "fake_recall")
        assert "fake_recall" in str(err.value) and "mapping" in str(err.value)
        assert "agent.yaml" in str(err.value)

    async def test_a_malformed_block_aborts_the_host_in_on_init(self):
        with pytest.raises(ValueError, match="fake_recall.timeout"):
            await make(config={"fake_recall": {"timeout": -1}})


class TestGate:

    async def test_a_plain_user_message_tail_is_searched_with_the_retrieval_query(self, build):
        env = await build({"fake_recall": {"cooldown": 0}})
        user(env, "first question")
        env.conv.append({"role": "assistant", "content": "an answer"})
        user(env, "and a follow up")
        await compose(env)
        # The previous reply precedes the unanswered user message, as memory builds it.
        assert env.source.queries == ["an answer\n\nand a follow up"]

    @pytest.mark.parametrize("tail", [
        {"role": "assistant", "content": "a reply"},
        {"role": "tool", "content": "tool output", "tool_call_id": "x"},
        {"role": "user", "content": "   "},
        {"role": "user", "content": None},
    ])
    async def test_other_tails_are_not_searched(self, build, tail):
        env = await build({"fake_recall": {"cooldown": 0}})
        env.conv.append(dict(tail))
        assert await compose(env) is None
        assert env.source.queries == []

    async def test_a_summary_row_at_the_tail_is_not_searched(self, build):
        env = await build({"fake_recall": {"cooldown": 0}})
        user(env, "real question")
        env.conv.append({"role": "user", "content": "a summary"}, MessageType.SUMMARY)
        assert await compose(env) is None
        assert env.source.queries == []

    async def test_context_rows_after_the_user_message_are_looked_through(self, build):
        # before_compose runs after every before_agent_turn, which may have
        # appended CONTEXT below the user message; the turn is still a user turn.
        env = await build({"fake_recall": {"cooldown": 0}})
        user(env, "real question")
        env.conv.append({"role": "user", "content": "ctx"}, MessageType.CONTEXT)
        await compose(env)
        assert env.source.queries == ["real question"]

    async def test_context_rows_after_an_assistant_reply_do_not_make_it_a_user_turn(self, build):
        env = await build({"fake_recall": {"cooldown": 0}})
        user(env, "question")
        env.conv.append({"role": "assistant", "content": "a reply"})
        env.conv.append({"role": "user", "content": "ctx"}, MessageType.CONTEXT)
        assert await compose(env) is None
        assert env.source.queries == []

    async def test_a_skipped_turn_does_not_use_up_the_cooldown(self, build):
        env = await build({"fake_recall": {"cooldown": 100}})
        env.conv.append({"role": "assistant", "content": "a reply"})
        await compose(env)
        user(env, "question")
        await compose(env)
        assert len(env.source.queries) == 1


class TestContribution:

    @pytest.mark.parametrize("hits", [None, []])
    async def test_no_hits_contributes_nothing(self, build, hits):
        env = await build({"fake_recall": {"cooldown": 0}})
        env.source.hits = hits
        user(env, "question")
        assert await compose(env) is None

    async def test_a_hit_that_cannot_render_is_dropped_and_logged_the_rest_kept(self, build, caplog):
        env = await build({"fake_recall": {"cooldown": 0}})
        env.source.hits = ["good", "bad", "also good"]
        original = env.source.render

        def render(hit: Any) -> str:
            # Fail on one hit only.
            if hit == "bad":
                raise KeyError("missing field")
            return original(hit)

        env.source.render = render
        user(env, "question")
        with caplog.at_level(logging.WARNING):
            contribution = await compose(env)
        assert contribution == Contribution("fake", ["rendered good", "rendered also good"])
        warned = warnings(caplog)
        assert warned and "fake" in warned[0].getMessage() and env.channel.id in warned[0].getMessage()
        assert warned[0].exc_info and warned[0].exc_info[0] is KeyError


class TestFailSoft:

    async def test_a_raising_search_is_logged_with_source_channel_and_traceback_and_the_turn_proceeds(self, build, caplog):
        env = await build({"fake_recall": {"cooldown": 0}})
        env.source.raises = ConnectionError("wiki is down")
        user(env, "question")
        with caplog.at_level(logging.WARNING):
            assert await compose(env) is None
        msgs = [r for r in warnings(caplog) if "fake" in r.getMessage()]
        assert msgs, [r.getMessage() for r in caplog.records]
        text = msgs[0].getMessage()
        # Names the source, the channel and the exception type; the remedy says the turn went on.
        assert env.channel.id in text and "ConnectionError" in text and "without" in text
        assert msgs[0].exc_info and msgs[0].exc_info[0] is ConnectionError

    async def test_a_failed_search_counts_as_a_search_for_the_cooldown(self, build):
        env = await build({"fake_recall": {"cooldown": 100}})
        env.source.raises = ConnectionError("down")
        user(env, "question")
        await compose(env)
        user(env, "retry")
        await compose(env)
        assert len(env.source.queries) == 1
        env.clock.t = 100.0
        user(env, "after the cooldown")
        await compose(env)
        assert len(env.source.queries) == 2

    async def test_a_search_over_its_timeout_is_cancelled_logged_and_counted(self, build, caplog):
        env = await build({"fake_recall": {"cooldown": 100, "timeout": 0.05}})
        env.source.sleep_forever = True
        user(env, "question")
        with caplog.at_level(logging.WARNING):
            assert await compose(env) is None
        assert env.source.cancelled
        msgs = [r.getMessage() for r in warnings(caplog) if "fake" in r.getMessage()]
        # Names the source, the channel and the setting to raise.
        assert msgs and env.channel.id in msgs[0] and "fake_recall.timeout" in msgs[0]
        user(env, "next")
        await compose(env)
        assert len(env.source.queries) == 1

    async def test_a_host_that_never_called_init_recall_fails_with_a_clear_message(self):
        recall = recall_mod()

        class Bare(recall.RecallSourceMixin, CorvidaePlugin):
            source_name = "bare"
            config_key = "bare_recall"

        channel = Channel(transport="test", scope="bare")
        channel.conversation = ContextWindow(channel.id)
        with pytest.raises(RuntimeError, match="init_recall"):
            await Bare().before_compose(channel=channel, trace_id="t", annotations={})


class TestSingleFlight:

    async def test_no_second_search_while_one_is_in_flight(self, build):
        env = await build({"fake_recall": {"cooldown": 0}})
        env.source.gate_event = asyncio.Event()
        user(env, "question")
        first = asyncio.create_task(compose(env))
        await settle()
        user(env, "second")
        # The second turn's gather finds the channel's slot taken and contributes nothing.
        assert await compose(env) is None
        assert len(env.source.queries) == 1
        # The first search was not superseded: it still lands once released.
        env.source.gate_event.set()
        assert await first == Contribution("fake", ["rendered hit one"])

    async def test_the_next_search_starts_after_the_first_has_finished(self, build):
        env = await build({"fake_recall": {"cooldown": 0}})
        user(env, "question")
        await compose(env)
        user(env, "second")
        await compose(env)
        assert len(env.source.queries) == 2

    async def test_a_cancelled_gather_releases_the_slot_and_is_not_logged_as_a_failure(self, build, caplog):
        env = await build({"fake_recall": {"cooldown": 0}})
        env.source.gate_event = asyncio.Event()
        user(env, "question")
        task = asyncio.create_task(compose(env))
        await settle()
        with caplog.at_level(logging.WARNING):
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert not [r for r in warnings(caplog) if "fake" in r.getMessage() and r.exc_info]
        # The cancelled search freed its slot, so the next turn searches again.
        env.source.gate_event.set()
        user(env, "after the cancel")
        await compose(env)
        assert env.source.cancelled and len(env.source.queries) == 2
