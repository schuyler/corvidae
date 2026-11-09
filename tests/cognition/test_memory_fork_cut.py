"""Turn and idle consolidation fork with a cut just past range_end (issue
#622), so a message that reaches the window between the trigger and the
fork's render is not narrated by this consolidation and again by the next."""

from helpers import FakeForkAgent

from corvidae.context import MessageType
from corvidae.fork import build_fork_prompt

from .test_memory_consolidation import fetch_stamped_memories, fetch_watermark, seed_dialog
from .test_memory_fork import DIALOG
from .test_memory_turn_trigger import REPLY, make_env, tokens_of  # noqa: F401 (fixture)

#: The message that arrives after the trigger fixed range_end.
LATE = ("user", "also the porch light flickers")


async def test_turn_fork_does_not_see_a_message_appended_after_the_hook(make_env):  # noqa: F811
    """The turn hook fixes range_end; a message appended to the window before
    the task forks lies past the cut. The next consolidation starts at it."""
    agent = FakeForkAgent({"consolidation"}, text=REPLY)
    env = await make_env(tokens_of(DIALOG), agent=agent)
    memory, channel, ids = env.memory, env.channel, env.ids
    # The late message is already in message_log, so only the window order
    # decides what the fork sees.
    [late_id] = await seed_dialog(env.persistence, channel, [LATE])

    # The hook fixes range_end at ids[-1]; the late message lands right after.
    await memory.on_agent_response(channel=channel)
    channel.conversation.messages.append({
        "role": LATE[0], "content": LATE[1],
        "_db_id": late_id, "_message_type": MessageType.MESSAGE,
    })
    await memory.wait_for_background_tasks()

    # One fork, cut just past the range: the late message isn't rendered.
    [call] = agent.calls
    assert call["cut"] == len(DIALOG)
    prompt = build_fork_prompt(channel.conversation, call["cut"], "X", frozenset(), "system")
    assert all(LATE[1] not in str(m.get("content")) for m in prompt)
    # The record and the watermark stop at range_end.
    [record] = await fetch_stamped_memories(env.db)
    assert (record["msg_id_start"], record["msg_id_end"]) == (ids[0], ids[-1])
    assert await fetch_watermark(env.db, channel.id) == ids[-1]

    # The next consolidation narrates the late message, starting from it.
    await memory._consolidate_range(channel.id, late_id, "idle")
    assert len(agent.calls) == 2
    assert LATE[1] in agent.calls[1]["instruction"]
    assert agent.calls[1]["cut"] == len(DIALOG) + 1


async def test_idle_fork_cuts_at_range_end(make_env):  # noqa: F811
    """Idle consolidation of (watermark, range_end] forks with the cut just
    past range_end, even though the window holds later rows."""
    agent = FakeForkAgent({"consolidation"}, text=REPLY)
    env = await make_env(tokens_of(DIALOG), agent=agent)

    await env.memory._consolidate_range(env.channel.id, env.ids[1], "idle")

    [call] = agent.calls
    assert call["cut"] == 2
    [record] = await fetch_stamped_memories(env.db)
    assert (record["msg_id_start"], record["msg_id_end"]) == (env.ids[0], env.ids[1])
