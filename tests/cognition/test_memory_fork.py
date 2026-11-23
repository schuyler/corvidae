"""Consolidation as a fork of the window (GitHub issue #51, phase 3).

Compaction trigger: before_compaction forks the record from the dropped
prefix and stashes it; the on_compaction task consumes it instead of the
view call, and a fork miss there is the main-profile view call (the range has
left the window, #380, #388). Idle trigger: a range wholly in the live window
forks the full window, and a miss skips (watermark unchanged, retried on the
next trigger, #320, PRD §3.8.5); a range not wholly in a live window is
written from the rendered view on llm.main. Everything after the LLM
call is unchanged.
"""

import json
import logging

import pytest

from corvidae.fork import ForkRefused
from corvidae_cognition.memory import DEFAULT_CONSOLIDATION_FORK_PROMPT

from helpers import FakeForkAgent, window_of
from .test_memory_consolidation import (
    THREE_ENTRIES, assert_shared_columns_equal, build_consolidation_env,
    entries_reply, fetch_memories, fetch_stamped_memories, fetch_watermark,
    seed_dialog,
)

FORK_JSON = json.dumps({"summary": "I told kestrel it was the brownout detector.",
                        "topic_tags": ["wifi"], "participants": ["kestrel"]})
DIALOG = [
    ("user", "my weather station keeps dropping wifi"),
    ("assistant", "Does it happen when the heater kicks on?"),
    ("user", "yes, every time"),
    ("assistant", "That's the brownout detector resetting the radio."),
]


class _Registry:
    def __init__(self, channel):
        self.channel = channel

    def get(self, channel_id):
        return self.channel if channel_id == self.channel.id else None


@pytest.fixture
async def env():
    """(memory, stub client, channel, db, rowids) with DIALOG persisted and
    a live window holding it."""
    memory, persistence, stub, channel, db = await build_consolidation_env()
    memory.pm.register(_Registry(channel), name="registry")
    ids = await seed_dialog(persistence, channel, DIALOG)
    channel.conversation = window_of(channel.id, ids, contents=[c for _, c in DIALOG])
    yield memory, stub, channel, db, ids
    await db.close()


def _use(memory, agent):
    # build_consolidation_env registers a default forks-off view-capable
    # agent under "agent"; swap it for this test's.
    if memory.pm.get_plugin("agent") is not None:
        memory.pm.unregister(name="agent")
    memory.pm.register(agent, name="agent")
    return agent


def _view_requests(stub):
    """The background view requests the consolidation stage sent."""
    return [c for c in stub.chat_calls if c["attribution"].get("stage") == "consolidation"]


async def _compaction(memory, channel, compacted, cut):
    await memory.before_compaction(channel=channel, compacted_ids=compacted, cut=cut)
    await memory.on_compaction(channel=channel, summary_msg={}, retain_count=1, compacted_ids=compacted)
    await memory.wait_for_background_tasks()


async def test_compaction_fork_record_is_stashed_and_consumed(env):
    memory, stub, channel, db, ids = env
    agent = _use(memory, FakeForkAgent({"consolidation"}, text=FORK_JSON))
    await _compaction(memory, channel, ids[:3], cut=3)

    call = agent.calls[0]
    assert call["cut"] == 3
    assert call["stage"] == "consolidation"
    assert call["instruction"].startswith(DEFAULT_CONSOLIDATION_FORK_PROMPT)
    assert (call["attribution"]["stage"], call["attribution"]["trigger"]) == ("consolidation", "compaction")
    assert _view_requests(stub) == []
    [record] = await fetch_memories(db)
    assert record["summary"] == json.loads(FORK_JSON)["summary"]
    assert (record["msg_id_start"], record["msg_id_end"]) == (ids[0], ids[2])
    assert memory._fork_stash == {}


@pytest.mark.parametrize("agent,drop_first_row", [
    (FakeForkAgent({"consolidation"}, error=ForkRefused("tool call")), False),
    (FakeForkAgent({"consolidation"}, error=RuntimeError("down")), False),
    (FakeForkAgent({"consolidation"}, text="not json"), False),
    (FakeForkAgent({"consolidation"}, text=json.dumps({"summary": " "})), False),
    (FakeForkAgent(set(), text=FORK_JSON), False),
    # The window lost a row of the range (an earlier compaction whose
    # consolidation failed): the fork can't see it all.
    (FakeForkAgent({"consolidation"}, text=FORK_JSON), True),
])
async def test_compaction_fork_miss_uses_the_view(env, agent, drop_first_row):
    memory, stub, channel, db, ids = env
    _use(memory, agent)
    if drop_first_row:
        channel.conversation.messages.pop(0)
    await _compaction(memory, channel, ids[:3], cut=2 if drop_first_row else 3)
    assert memory._fork_stash == {}
    # One view request headed by the system prompt, attributed standalone.
    [call] = _view_requests(stub)
    assert call["messages"][0] == {"role": "system", "content": FakeForkAgent.SYSTEM_PROMPT}
    assert call["attribution"]["mode"] == "standalone"
    assert len(await fetch_memories(db)) == 1


async def test_stash_from_a_different_watermark_is_discarded(env):
    """The watermark moved between fork and task: the stashed record covers
    the wrong range, so the task makes the view call."""
    memory, stub, channel, db, ids = env
    _use(memory, FakeForkAgent({"consolidation"}, text=FORK_JSON))
    await memory.before_compaction(channel=channel, compacted_ids=ids[:3], cut=3)
    # Let the background fork (#628) read the watermark and finish first.
    await memory.wait_for_background_tasks()
    mem_db = await memory._ensure_schema()
    await memory._set_watermark(mem_db, channel.id, ids[0])
    await mem_db.commit()
    await memory.on_compaction(channel=channel, summary_msg={}, retain_count=1, compacted_ids=ids[:3])
    await memory.wait_for_background_tasks()
    assert len(_view_requests(stub)) == 1
    assert (await fetch_memories(db))[0]["msg_id_start"] == ids[1]


async def test_summary_rows_in_range_do_not_block_the_fork(env):
    """A live summary has no _db_id but its message_log row can sit inside
    the range; only dialog rows count for containment."""
    memory, stub, channel, db, ids = env
    agent = _use(memory, FakeForkAgent({"consolidation"}, text=FORK_JSON))
    await db.execute(
        "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
        "VALUES (?, '{}', 0.0, 'summary')", (channel.id,))
    await db.commit()
    later = ids + await seed_dialog(memory.pm.get_plugin("persistence"), channel, [("user", "more")])
    channel.conversation = window_of(channel.id, later, summary=True)
    await memory.before_compaction(channel=channel, compacted_ids=later[:5], cut=6)
    # The compaction fork runs in the background (#628).
    await memory.wait_for_background_tasks()
    assert len(agent.calls) == 1


@pytest.mark.parametrize("live_window,error,forked,view", [
    (True, None, True, 0),
    # No live window: the range is written from the view on llm.main.
    (False, None, False, 1),
    # A fork failure on an in-window range skips: no view call and no
    # memory row (#320); test_idle_in_window_miss_skips covers the details.
    (True, RuntimeError("down"), False, 0),
])
async def test_idle_trigger_forks_the_full_window(env, live_window, error, forked, view):
    memory, stub, channel, db, ids = env
    agent = _use(memory, FakeForkAgent({"consolidation"}, text=FORK_JSON, error=error))
    if not live_window:
        channel.conversation = None
    await memory._consolidate_range(channel.id, ids[-1], "idle")
    if live_window:
        # The cut sits just past range_end (#622); range_end is the window's
        # last row here, so the fork still renders the full window.
        assert (agent.calls[0]["cut"], agent.calls[0]["attribution"]["trigger"]) == (len(ids), "idle")
    assert len(_view_requests(stub)) == view
    assert len(await fetch_memories(db)) == (1 if (forked or view) else 0)


@pytest.mark.parametrize("agent,reason", [
    (FakeForkAgent({"consolidation"}, error=RuntimeError("down")), "RuntimeError"),
    (FakeForkAgent({"consolidation"}, text="not json"), "the fork failed or its reply was unusable"),
    (FakeForkAgent(set(), text=FORK_JSON), "forks are not enabled for consolidation"),
    (None, "forks are not enabled for consolidation"),  # no agent plugin
])
async def test_idle_in_window_miss_skips(env, caplog, agent, reason):
    """#320: an in-window range that cannot fork is skipped, not written from
    the background view; the watermark stays and a warning names the range."""
    memory, stub, channel, db, ids = env
    if agent is not None:
        _use(memory, agent)
    before = await fetch_watermark(db, channel.id)
    with caplog.at_level(logging.WARNING, logger="corvidae.memory"):
        await memory._consolidate_range(channel.id, ids[-1], "idle")
    # No background-role call, no record, watermark unmoved.
    assert stub.chat_calls == []
    assert await fetch_memories(db) == []
    assert await fetch_watermark(db, channel.id) == before
    # One warning with the channel, the message range, the reason, the retry.
    [text] = [r.getMessage() for r in caplog.records
              if r.levelno == logging.WARNING and "consolidation skipped for channel" in r.getMessage()]
    assert channel.id in text and f"{ids[0]}-{ids[-1]}" in text
    assert reason in text and "watermark stays at" in text
    # An idle skip is never retried by the next idle firing (#342), so its line
    # names what does -- including when the cause is forks being off, which
    # rules out the turn trigger as well (#1002 review).
    if "forks are not enabled" in reason:
        assert "a new message or compaction retries" in text, text
    else:
        assert "a new message, the turn trigger or compaction retries" in text, text


async def test_idle_skip_is_retried(env):
    """#320: a skipped range stays unconsolidated; the next idle run with a
    working fork consolidates it."""
    memory, stub, channel, db, ids = env
    _use(memory, FakeForkAgent({"consolidation"}, error=RuntimeError("down")))
    await memory._consolidate_range(channel.id, ids[-1], "idle")
    assert await fetch_memories(db) == []
    # Swap in a working fork and run again.
    agent = _use(memory, FakeForkAgent({"consolidation"}, text=FORK_JSON))
    await memory._consolidate_range(channel.id, ids[-1], "idle")
    assert len(agent.calls) == 1 and stub.chat_calls == []
    [record] = await fetch_memories(db)
    assert (record["msg_id_start"], record["msg_id_end"]) == (ids[0], ids[-1])


def _no_window(memory, channel, ids):
    channel.conversation = None


def _no_registry_channel(memory, channel, ids):
    memory.pm.unregister(name="registry")


def _range_partly_trimmed(memory, channel, ids):
    channel.conversation = window_of(channel.id, ids[2:], contents=[c for _, c in DIALOG][2:])


@pytest.mark.parametrize("break_window", [_no_window, _no_registry_channel, _range_partly_trimmed])
async def test_idle_out_of_window_uses_the_view(env, break_window):
    """#320 R5, #380 R1: a range that is not wholly in a live window is
    windowless work: no fork even with a working agent, one view request
    (system prompt first) on llm.main, a record."""
    memory, stub, channel, db, ids = env
    agent = _use(memory, FakeForkAgent({"consolidation"}, text=FORK_JSON))
    break_window(memory, channel, ids)
    await memory._consolidate_range(channel.id, ids[-1], "idle")
    assert agent.calls == []
    [call] = _view_requests(stub)
    assert call["messages"][0] == {"role": "system", "content": FakeForkAgent.SYSTEM_PROMPT}
    assert len(await fetch_memories(db)) == 1


async def test_compaction_trigger_without_stash_uses_the_view(env, caplog):
    """#320 R3, #380 R1: the fork at compaction fails, so nothing is stashed
    and the on_compaction task makes the one background view call, with a
    warning that says so."""
    memory, stub, channel, db, ids = env
    _use(memory, FakeForkAgent({"consolidation"}, error=RuntimeError("down")))
    with caplog.at_level(logging.WARNING, logger="corvidae.memory"):
        await _compaction(memory, channel, ids[:3], cut=3)
    assert memory._fork_stash == {}
    # One request: headed by the system prompt, ending in the instruction.
    [call] = _view_requests(stub)
    assert call["messages"][0] == {"role": "system", "content": FakeForkAgent.SYSTEM_PROMPT}
    assert call["messages"][-1]["content"].startswith(DEFAULT_CONSOLIDATION_FORK_PROMPT)
    assert len(await fetch_memories(db)) == 1
    assert any(r.levelno == logging.WARNING and channel.id in r.getMessage()
               and "background view" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("summary,watermark_index,expected", [
    (False, None, "the whole conversation above"),
    (True, None, "follows the summary"),
    # Rows below the watermark are still in view: name the first new one.
    (True, 1, "«yes, every time»"),
])
async def test_range_boundary_sentence(env, summary, watermark_index, expected):
    memory, stub, channel, db, ids = env
    agent = _use(memory, FakeForkAgent({"consolidation"}, text=FORK_JSON))
    channel.conversation = window_of(channel.id, ids, summary=summary, contents=[c for _, c in DIALOG])
    if watermark_index is not None:
        mem_db = await memory._ensure_schema()
        await memory._set_watermark(mem_db, channel.id, ids[watermark_index])
        await mem_db.commit()
    await memory._consolidate_range(channel.id, ids[-1], "idle")
    assert expected in agent.calls[0]["instruction"]


async def test_forks_send_the_record_schema(env):
    """#199: compaction and idle forks carry the same response_format as the
    background view call."""
    memory, stub, channel, db, ids = env
    agent = _use(memory, FakeForkAgent({"consolidation"}, text=FORK_JSON))
    await _compaction(memory, channel, ids[:3], cut=3)
    await memory._consolidate_range(channel.id, ids[-1], "idle")
    assert len(agent.calls) == 2
    for call in agent.calls:
        schema = call["extra_body"]["response_format"]["json_schema"]["schema"]
        assert schema["required"] == ["entries", "importance"]
        assert set(schema["properties"]) == {"entries", "participants", "importance"}


# ---------------------------------------------------------------------------
# Several narrative entries per consolidation (#635)
# ---------------------------------------------------------------------------

#: A reply whose entries are all blank, beside a stray top-level summary that
#: must not be read once "entries" is present (C2).
BLANK_ENTRIES_JSON = json.dumps({
    "entries": [{"narrative": "  "}, {"narrative": ""}],
    "summary": "A stray summary that must not become a record.",
    "importance": 0.5,
})


async def test_fork_reply_with_three_entries_stores_three_rows(env):
    """C1: a 3-entry fork reply stores 3 rows that share range, created_at,
    importance, valence, participants and stamp, and each carries its own
    summary, tags (in reply order) and embedding."""
    memory, stub, channel, db, ids = env
    _use(memory, FakeForkAgent({"consolidation"}, text=entries_reply(THREE_ENTRIES), model="fork-model"))
    # One distinct vector per input text.
    stub.embed_result = [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0]]
    await memory._consolidate_range(channel.id, ids[-1], "idle")

    rows = await fetch_stamped_memories(db)
    assert len(rows) == 3
    assert_shared_columns_equal(rows)
    assert (rows[0]["msg_id_start"], rows[0]["msg_id_end"]) == (ids[0], ids[-1])
    assert rows[0]["participants"] == json.dumps(["kestrel"])
    assert (rows[0]["mode"], rows[0]["model"], rows[0]["trigger"]) == ("fork", "fork-model", "idle")
    assert rows[0]["prompt_hash"] is not None
    assert [r["summary"] for r in rows] == [n for n, _ in THREE_ENTRIES]
    assert [json.loads(r["topic_tags"]) for r in rows] == [t for _, t in THREE_ENTRIES]
    assert [r["embedded"] for r in rows] == [1, 1, 1]
    # Each row's stored vector is the one its own narrative was embedded to
    # (C1); only checkable when the vec table exists.
    if memory._vec_available:
        import struct
        expected = [(1.0, 0, 0, 0), (0, 1.0, 0, 0), (0, 0, 1.0, 0)]
        for row, vec in zip(rows, expected):
            async with db.execute(
                "SELECT embedding FROM memory_vec WHERE memory_id = ?", (row["id"],)
            ) as cursor:
                blob = (await cursor.fetchone())[0]
            assert struct.unpack("4f", blob) == vec


async def test_idle_fork_with_only_blank_entries_skips(env, caplog):
    """C2 (idle): a reply whose entries all have blank narratives is
    unusable: no row, watermark unchanged, and the skip warning names the
    channel and range."""
    memory, stub, channel, db, ids = env
    _use(memory, FakeForkAgent({"consolidation"}, text=BLANK_ENTRIES_JSON))
    before = await fetch_watermark(db, channel.id)
    with caplog.at_level(logging.WARNING, logger="corvidae.memory"):
        await memory._consolidate_range(channel.id, ids[-1], "idle")

    assert await fetch_memories(db) == []
    assert await fetch_watermark(db, channel.id) == before
    [text] = [r.getMessage() for r in caplog.records
              if r.levelno == logging.WARNING and "consolidation skipped for channel" in r.getMessage()]
    assert channel.id in text and f"{ids[0]}-{ids[-1]}" in text


async def test_compaction_fork_with_only_blank_entries_falls_to_the_view(env):
    """C2 (compaction): the same unusable reply leaves no stash, so the
    on_compaction task sends the catch-up view request."""
    memory, stub, channel, db, ids = env
    _use(memory, FakeForkAgent({"consolidation"}, text=BLANK_ENTRIES_JSON))
    await _compaction(memory, channel, ids[:3], cut=3)

    assert memory._fork_stash == {}
    assert len(_view_requests(stub)) == 1
    rows = await fetch_memories(db)
    assert len(rows) == 1
    assert "stray summary" not in rows[0]["summary"]


async def test_legacy_summary_reply_reads_as_one_entry(env):
    """C3: a legacy {"summary", "topic_tags"} reply is one entry: one row
    with that summary and those tags."""
    memory, stub, channel, db, ids = env
    reply = json.dumps({"summary": "I fixed kestrel's wifi.", "topic_tags": ["wifi", "esp32"]})
    _use(memory, FakeForkAgent({"consolidation"}, text=reply))
    await memory._consolidate_range(channel.id, ids[-1], "idle")

    [row] = await fetch_memories(db)
    assert row["summary"] == "I fixed kestrel's wifi."
    assert json.loads(row["topic_tags"]) == ["wifi", "esp32"]


# --- #342: an unchanged idle skip is not retried every idle period ----------


@pytest.fixture
async def env_with_persistence():
    """The ``env`` fixture plus the persistence plugin, for tests that add a
    message after the channel has already skipped."""
    memory, persistence, stub, channel, db = await build_consolidation_env()
    memory.pm.register(_Registry(channel), name="registry")
    ids = await seed_dialog(persistence, channel, DIALOG)
    channel.conversation = window_of(
        channel.id, ids, contents=[c for _, c in DIALOG]
    )
    yield memory, persistence, stub, channel, db, ids
    await db.close()


async def _idle_firing(memory, channel_id):
    await memory._on_channel_idle(channel_id)
    await memory.wait_for_background_tasks()


def _skip_warnings(caplog):
    return [
        r for r in caplog.records
        if "consolidation skipped for channel" in r.getMessage()
    ]


async def test_an_unchanged_range_skips_once(env_with_persistence, caplog):
    """#342 option B: the second idle firing on an unchanged channel makes no
    fork attempt and logs nothing. Delete the early return in
    ``_on_channel_idle`` and this is the test that fails."""
    memory, persistence, stub, channel, db, ids = env_with_persistence
    agent = _use(memory, FakeForkAgent({"consolidation"}, error=RuntimeError("down")))

    with caplog.at_level(logging.WARNING, logger="corvidae.memory"):
        await _idle_firing(memory, channel.id)
        await _idle_firing(memory, channel.id)

    assert len(agent.calls) == 1, agent.calls
    assert len(_skip_warnings(caplog)) == 1
    assert await fetch_memories(db) == []


async def test_a_new_message_makes_the_next_idle_firing_retry(
    env_with_persistence, caplog,
):
    """Nothing has changed is the condition, and a new message changes it.
    Compare the watermark alone (drop ``MAX(id)`` from the tuple) and this is
    the test that fails: the second firing stays silent and never retries."""
    memory, persistence, stub, channel, db, ids = env_with_persistence
    agent = _use(memory, FakeForkAgent({"consolidation"}, error=RuntimeError("down")))

    with caplog.at_level(logging.WARNING, logger="corvidae.memory"):
        await _idle_firing(memory, channel.id)
        assert len(agent.calls) == 1

        more = await seed_dialog(persistence, channel, [("user", "any news?")])
        channel.conversation = window_of(
            channel.id, ids + more,
            contents=[c for _, c in DIALOG] + ["any news?"],
        )
        await _idle_firing(memory, channel.id)

    assert len(agent.calls) == 2, agent.calls
    assert len(_skip_warnings(caplog)) == 2


async def test_the_suppression_holds_for_every_later_firing(
    env_with_persistence, caplog,
):
    """#342 option B: quiet means quiet, not one firing in two.

    ``pop``ing the record suppresses exactly one firing and then retries, so a
    quiet channel alternates skip/suppress: half the wasted forks and half the
    repeated warnings, which is not what the ruling says and not what the title
    says. Reading it instead holds for every firing until something changes.
    """
    memory, persistence, stub, channel, db, ids = env_with_persistence
    agent = _use(memory, FakeForkAgent({"consolidation"}, error=RuntimeError("down")))

    with caplog.at_level(logging.WARNING, logger="corvidae.memory"):
        for _ in range(4):
            await _idle_firing(memory, channel.id)

    assert len(agent.calls) == 1, agent.calls
    assert len(_skip_warnings(caplog)) == 1
# --- #929: the fork paths must name the range the reply was rated over ------


def _unusable_importance_reply() -> str:
    return json.dumps({
        "summary": "I told kestrel it was the brownout detector.",
        "topic_tags": ["wifi"], "participants": ["kestrel"],
        # 7 is a rating on another scale: dropped, so the rubric rates it.
        "importance": 7,
    })


async def _importance_warning(caplog, run) -> str:
    with caplog.at_level(logging.WARNING, logger="corvidae.memory"):
        await run()
    [text] = [r.getMessage() for r in caplog.records
              if "importance is unusable" in r.getMessage()]
    return text


async def test_compaction_fork_warns_with_the_real_range(env, caplog):
    """Removing ``msg_id_range=(rows[0][0], range_end)`` from the compaction
    fork's ``_fork_record`` call leaves this the only failing test: the warning
    then says "message ids unknown" and an operator cannot find the range."""
    memory, stub, channel, db, ids = env
    _use(memory, FakeForkAgent({"consolidation"}, text=_unusable_importance_reply()))

    text = await _importance_warning(
        caplog, lambda: _compaction(memory, channel, ids[:3], cut=3),
    )

    assert channel.id in text
    assert f"{ids[0]}-{ids[2]}" in text, text
    assert "unknown" not in text


async def test_idle_fork_warns_with_the_real_range(env, caplog):
    """Same for the live/idle fork's ``_fork_record`` call."""
    memory, stub, channel, db, ids = env
    _use(memory, FakeForkAgent({"consolidation"}, text=_unusable_importance_reply()))

    text = await _importance_warning(
        caplog, lambda: memory._consolidate_range(channel.id, ids[-1], "idle"),
    )

    assert channel.id in text
    assert f"{ids[0]}-{ids[-1]}" in text, text
    assert "unknown" not in text


async def test_a_moved_watermark_makes_the_next_idle_firing_retry(
    env_with_persistence, caplog,
):
    """#1002: the pair is (watermark, MAX(id)), and the watermark half is real.

    A partial advance — a catch-up segment leaving the watermark short of the
    last row — means work was done and the tail is still un-consolidated, so the
    idle trigger must try again. Compare MAX(id) alone and this test fails,
    because nothing about the row count changed.
    """
    memory, persistence, stub, channel, db, ids = env_with_persistence
    agent = _use(memory, FakeForkAgent({"consolidation"}, error=RuntimeError("down")))

    with caplog.at_level(logging.WARNING, logger="corvidae.memory"):
        await _idle_firing(memory, channel.id)
        assert len(agent.calls) == 1

        await db.execute(
            "INSERT OR REPLACE INTO consolidation_watermark "
            "(channel_id, last_message_id) VALUES (?, ?)",
            (channel.id, ids[1]),
        )
        await db.commit()

        await _idle_firing(memory, channel.id)

    assert len(agent.calls) == 2, agent.calls
    assert len(_skip_warnings(caplog)) == 2


async def test_a_turn_skip_does_not_suppress_the_next_idle_firing(
    env_with_persistence, caplog,
):
    """#1002: only an idle skip records the pair.

    A turn skip's line promises that the idle trigger retries, so a turn skip
    must not leave a record that makes the next idle firing return silently.
    Drop the ``trigger == "idle"`` guard in ``_skip_consolidation`` and this
    test fails.
    """
    memory, persistence, stub, channel, db, ids = env_with_persistence
    agent = _use(memory, FakeForkAgent({"consolidation"}, error=RuntimeError("down")))

    with caplog.at_level(logging.WARNING, logger="corvidae.memory"):
        await memory._consolidate_range(channel.id, ids[-1], "turn")
        assert len(agent.calls) == 1

        await _idle_firing(memory, channel.id)

    assert len(agent.calls) == 2, agent.calls
    assert len(_skip_warnings(caplog)) == 2


@pytest.mark.parametrize("agent", [
    FakeForkAgent(set(), text=FORK_JSON), None,  # no agent plugin
])
async def test_a_forks_off_turn_skip_names_idle_or_compaction(env, caplog, agent):
    """#1002: forks off means the turn hook never fires, so a turn range's line
    must name what does retry it — idle or compaction.

    Moving this wording out of ``_skip_consolidation``'s shared branch and into
    the call site took it out of every test's reach; nothing asserted it until
    here (#1002 review).
    """
    memory, stub, channel, db, ids = env
    if agent is not None:
        _use(memory, agent)
    with caplog.at_level(logging.WARNING, logger="corvidae.memory"):
        await memory._consolidate_range(channel.id, ids[-1], "turn")
    [text] = [
        r.getMessage() for r in caplog.records
        if r.levelno == logging.WARNING
        and "consolidation skipped for channel" in r.getMessage()
    ]
    assert "the next idle or compaction trigger retries" in text, text


async def test_a_forks_off_idle_skip_is_suppressed_too(env_with_persistence, caplog):
    """#1002 review: #342's suppression holds when the cause is forks off.

    Every other suppression test drives a *failing fork*, so the case #342 is
    titled after — forks off, nothing will change until a message arrives or
    someone enables forks — had no test. Two idle firings, one warning.
    """
    memory, persistence, stub, channel, db, ids = env_with_persistence
    _use(memory, FakeForkAgent(set(), text=FORK_JSON))

    with caplog.at_level(logging.WARNING, logger="corvidae.memory"):
        await _idle_firing(memory, channel.id)
        await _idle_firing(memory, channel.id)

    assert len(_skip_warnings(caplog)) == 1
    assert "a new message or compaction retries" in _skip_warnings(caplog)[0].getMessage()
