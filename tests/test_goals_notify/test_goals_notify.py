"""Tests for the goals_notify plugin (#841).

A real plugin manager with the goals hookspec (GoalsSpec) installed, the real
CrossChannelSendPlugin, and the subscriber. The hook is fired through the
plugin manager inside a turn's attribution, the way GoalsPlugin fires it after
a state change commits.
"""

from __future__ import annotations

import logging
from importlib.metadata import entry_points
from unittest.mock import AsyncMock

import pytest
from corvidae_cognition.goals import GoalsSpec
from cross_channel_send import CrossChannelSendPlugin

from corvidae.attribution import _attribution
from corvidae.channel import ChannelConfig, ChannelRegistry
from corvidae.hooks import create_plugin_manager

from goals_notify import GoalsNotifyPlugin

ORIGIN = "signal:alice"
ACTOR = "irc:#ops"


@pytest.fixture(autouse=True)
def _reset_attribution():
    """Keep attribution clean between tests."""
    token = _attribution.set({})
    yield
    _attribution.reset(token)


async def _build():
    """PM with GoalsSpec, a registry holding both channels, cross_channel_send
    and the subscriber, all started; returns (pm, registry, xcs)."""
    pm = create_plugin_manager()
    pm.add_hookspecs(GoalsSpec)
    registry = ChannelRegistry()
    pm.register(registry, name="registry")
    registry.get_or_create("signal", "alice", config=ChannelConfig())
    registry.get_or_create("irc", "#ops", config=ChannelConfig())
    xcs = CrossChannelSendPlugin()
    pm.register(xcs, name="cross_channel_send")
    subscriber = GoalsNotifyPlugin()
    pm.register(subscriber, name="goals_notify")
    # Start the way the runtime does: broadcast on_init, then on_start.
    await pm.ahook.on_init(pm=pm, config={})
    await pm.ahook.on_start(config={})
    return pm, registry, xcs


def _in_turn(channel_id: str, hop_count: int | None = None) -> None:
    """Set the attribution a turn on ``channel_id`` would carry."""
    annotations = {} if hop_count is None else {"hop_count": hop_count}
    _attribution.set(
        {"channel_id": channel_id, "trace_id": "trace-841", "annotations": annotations}
    )


async def _fire(pm, *, kind="status", old="open", new="blocked",
                origin=ORIGIN, actor=ACTOR, conclusion=None, goal_id=7):
    """Fire on_goals_state_changed through the plugin manager."""
    await pm.ahook.on_goals_state_changed(
        goal_id=goal_id, kind=kind, old_status=old, new_status=new,
        origin_channel=origin, actor_channel=actor, conclusion=conclusion,
    )


async def test_status_change_notifies_origin_with_the_send_fields():
    # B1 + B2: the real notify_other_channel builds the notification from the
    # changing turn's attribution; on_notify sees every field #808 names.
    pm, registry, xcs = await _build()
    on_notify = AsyncMock()
    pm.ahook_raise.on_notify = on_notify
    _in_turn(ACTOR, hop_count=1)

    await _fire(pm, old="open", new="blocked")

    assert on_notify.call_count == 1
    kwargs = on_notify.call_args.kwargs
    assert kwargs["channel"] is registry.get(ORIGIN)
    assert kwargs["source"] == f"agent_send from {ACTOR}"
    # The changing turn carries no origin, so neither does the note (#1034).
    assert kwargs["annotations"] == {
        "source_channel": ACTOR,
        "source_trace_id": "trace-841",
        "hop_count": 2,
    }
    assert "#7" in kwargs["text"]
    assert "open → blocked" in kwargs["text"]


async def test_retirement_notifies_origin_with_conclusion():
    # B1 + B2: one call to the origin naming the old status and the conclusion.
    pm, registry, xcs = await _build()
    xcs.notify_other_channel = AsyncMock(return_value=f"Queued on {ORIGIN}.")
    _in_turn(ACTOR)

    await _fire(pm, kind="retired", old="in-progress", new="retired",
                conclusion="Shipped the fix; the bug is gone.")

    xcs.notify_other_channel.assert_awaited_once()
    channel_id, text = xcs.notify_other_channel.call_args.args
    assert channel_id == ORIGIN
    assert "#7" in text
    assert "retired" in text
    assert "in-progress" in text
    assert "Shipped the fix; the bug is gone." in text


@pytest.mark.parametrize("origin", [None, ACTOR])
async def test_no_send_without_origin_or_from_origin(origin):
    # B3: a seeded goal (origin None) and a change made on the origin channel
    # itself are skipped before notify_other_channel is called.
    pm, registry, xcs = await _build()
    xcs.notify_other_channel = AsyncMock(return_value="Queued on x.")
    _in_turn(ACTOR)

    await _fire(pm, origin=origin, actor=ACTOR)

    xcs.notify_other_channel.assert_not_awaited()


async def test_refusal_is_logged_at_warning(caplog):
    # B4: notify_other_channel reports refusals as strings; the subscriber logs
    # them naming the goal, the channel and the reason.
    pm, registry, xcs = await _build()
    # The refusal text names no channel, so ORIGIN in the log must come from
    # the subscriber itself.
    refusal = "Could not send: unexpected error."
    xcs.notify_other_channel = AsyncMock(return_value=refusal)
    _in_turn(ACTOR)

    with caplog.at_level(logging.WARNING):
        await _fire(pm, goal_id=42)

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("42" in m and ORIGIN in m and refusal in m for m in warnings)


async def test_success_is_not_logged_at_warning(caplog):
    # B4's other side: a queued send is not reported as a refusal.
    pm, registry, xcs = await _build()
    xcs.notify_other_channel = AsyncMock(return_value=f"Queued on {ORIGIN}.")
    _in_turn(ACTOR)

    with caplog.at_level(logging.WARNING):
        await _fire(pm)

    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_entry_point_registered():
    # B5: without the entry point the plugin silently never loads.
    (ep,) = entry_points(group="corvidae", name="goals_notify")
    assert ep.value == "goals_notify:GoalsNotifyPlugin"


def test_declared_dependencies():
    # B5: cross_channel_send is required; goals is optional.
    assert GoalsNotifyPlugin.depends_on == frozenset({"cross_channel_send"})
    assert GoalsNotifyPlugin.optional_depends_on == frozenset({"goals"})

