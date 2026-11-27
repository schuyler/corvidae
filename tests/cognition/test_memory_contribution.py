"""RED tests for #1176: memory's retrieval as a before-compose contribution.

Spec: the #1176 design doc in PR #1212's history, section 6 (callers, memory)
and requirements R2.2, R3.9, R5.4. Memory's retrieval moves out of
``before_agent_turn`` into ``before_compose``, which returns a
``Contribution("memory", lines, keys)``. What the store then admitted is
announced on ``on_held_admitted``; memory writes access stats, ``retrieval_log``,
``retrieval_candidate`` and the ``on_trace_record`` profile from that event.
Raw-transcript pieces are sized to ``memory.entry_target_tokens``.

Criteria covered: M1 contribution, M2 gate past trailing CONTEXT, M3 the
admitted event writes the records, M4 an empty contribution still logs, M5
piece size.

Equivalence note: the kernel gathers ``before_compose`` by calling each
implementation with ``channel``, ``trace_id`` and ``annotations`` keywords,
and fires ``on_held_admitted`` with ``channel``, ``source``, ``trace_id``,
``admitted_keys`` and ``duplicate_keys`` after appending. These tests make
those same calls on the real plugin, on a real database, with keys shaped as
memory produced them in the gather (``memory <id>)``), so memory cannot tell
them from the kernel's.

``corvidae.held`` does not exist yet, so it is imported on use.
"""

import copy
import inspect
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import aiosqlite
import pytest
from corvidae_cognition.memory import WORST_MEMORY_LINE_PREFIX, MemoryPlugin
from fake_clock import install_kernel_clock
from helpers import attach_store, fresh_session_db_path, stub_llm_config, with_chat_llm

from corvidae.channel import Channel
from corvidae.context import ContextWindow, MessageType, count_tokens
from corvidae.hooks import create_plugin_manager
from corvidae.llm_plugin import LLMPlugin
from corvidae.persistence import PersistencePlugin, init_db

from .test_memory_consolidation import (
    EMBED_CONFIG as CONSOLIDATION_CONFIG,
)
from .test_memory_consolidation import (
    _SequenceChatClient,
    build_consolidation_env,
    fetch_stamped_memories,
    seed_dialog,
)
from .test_memory_retrieval import EMBED_CONFIG, StubEmbedClient, gate_open, seed_memory
from .trace_helpers import TraceRecorder, add_trace


def held_mod() -> Any:
    # Imported on use: corvidae.held is written in Green.
    import corvidae.held as held

    return held


async def build_env(config: dict | None = None, *, with_spy_funnel: bool = False) -> SimpleNamespace:
    # Memory on a real plugin manager as Runtime builds it: persistence, kernel
    # clock and idle tracker, the trace protocol plugin with a recorder, llm
    # with a stub embedder. No funnel: memory must not need one. With
    # ``with_spy_funnel`` an AsyncMock-backed plugin named "funnel" is
    # registered, to prove memory no longer calls it.
    config = with_chat_llm(config or EMBED_CONFIG)
    db = await aiosqlite.connect(fresh_session_db_path(), isolation_level=None)
    await init_db(db)
    pm = create_plugin_manager()
    install_kernel_clock(pm)
    persistence = PersistencePlugin()
    persistence.db, persistence.writer = await attach_store(db)
    pm.register(persistence, name="persistence")
    recorder = add_trace(pm, TraceRecorder())
    llm = LLMPlugin()
    pm.register(llm, name="llm")
    await llm.on_init(pm=pm, config=config)
    embedder = StubEmbedClient()
    llm._clients["main"] = embedder
    llm._clients["embedding"] = embedder
    stub_llm_config(llm, config.get("llm"))
    spy = None
    if with_spy_funnel:
        spy = SimpleNamespace(admit_detailed=AsyncMock(), admit=AsyncMock())
        pm.register(spy, name="funnel")
    memory = MemoryPlugin()
    pm.register(memory, name="memory")
    await memory.on_init(pm=pm, config=config)
    await memory.on_start(config=config)
    channel = Channel(transport="irc", scope="#electronics")
    conv = ContextWindow(channel.id)
    channel.conversation = conv
    return SimpleNamespace(
        memory=memory, channel=channel, conv=conv, db=db, recorder=recorder, spy=spy,
        pm=pm,
    )


async def maybe_await(value: Any) -> Any:
    # A hookimpl may be sync or async; the kernel awaits it either way.
    if inspect.isawaitable(value):
        return await value
    return value


async def compose(env: SimpleNamespace, text: str, trace_id: str = "trace-1") -> Any:
    # One turn's before_compose for a new user message.
    env.conv.append({"role": "user", "content": text})
    return await maybe_await(
        env.memory.before_compose(
            channel=env.channel, trace_id=trace_id, annotations={}
        )
    )


async def admitted(
    env: SimpleNamespace,
    *,
    admitted_keys: list[str],
    duplicate_keys: list[str],
    trace_id: str = "trace-1",
    source: str = "memory",
) -> None:
    # The kernel's announcement after it appended (or did not append) the row.
    await maybe_await(
        env.memory.on_held_admitted(
            channel=env.channel,
            source=source,
            trace_id=trace_id,
            admitted_keys=admitted_keys,
            duplicate_keys=duplicate_keys,
        )
    )


async def fetch_all(db: aiosqlite.Connection, sql: str) -> list[tuple]:
    # Every row of a query, as tuples.
    async with db.execute(sql) as cursor:
        return list(await cursor.fetchall())


class TestContribution:
    """M1: memory contributes lines and keys, and no longer appends CONTEXT itself."""

    async def test_before_compose_returns_memory_lines_and_keys_for_a_user_turn(self):
        env = await build_env(gate_open())
        rowid = await seed_memory(
            env.db, env.channel.id, "kestrel moved the rain gauge to the north fence"
        )
        contribution = await compose(env, "where is the rain gauge?")
        assert isinstance(contribution, held_mod().Contribution)
        assert contribution.source == "memory"
        assert len(contribution.entries) == 1
        assert f"memory {rowid})" in contribution.entries[0]
        assert "north fence" in contribution.entries[0]
        # One key per entry: the record id the store dedupes on.
        assert contribution.keys == [f"memory {rowid})"]
        await env.db.close()

    async def test_before_agent_turn_no_longer_retrieves_or_appends_context(self):
        env = await build_env(gate_open(), with_spy_funnel=True)
        await seed_memory(
            env.db, env.channel.id, "kestrel moved the rain gauge to the north fence"
        )
        env.conv.append({"role": "user", "content": "where is the rain gauge?"})
        # Through the hook, not the method: memory may drop before_agent_turn
        # entirely once retrieval moves; the criterion is only that the turn's
        # before_agent_turn appends and offers nothing.
        await env.pm.ahook.before_agent_turn(
            channel=env.channel, trace_id="trace-1", annotations={}
        )
        # Nothing in the window, nothing offered to a funnel, nothing logged:
        # the whole retrieval now runs in before_compose.
        context = [m for m in env.conv.messages if m.get("_message_type") == MessageType.CONTEXT]
        assert context == []
        env.spy.admit_detailed.assert_not_awaited()
        assert await fetch_all(env.db, "SELECT * FROM retrieval_log") == []
        await env.db.close()


class TestGate:
    """M2: the gate looks past trailing CONTEXT rows for the user message."""

    async def test_window_ending_user_message_then_context_stamp_still_retrieves(self):
        env = await build_env(gate_open())
        rowid = await seed_memory(
            env.db, env.channel.id, "kestrel moved the rain gauge to the north fence"
        )
        env.conv.append({"role": "user", "content": "where is the rain gauge?"})
        # What timepiece's stamp or signal's pending row leave in before_agent_turn.
        env.conv.append(
            {"role": "system", "content": "[time] 2026-11-23 09:00"}, MessageType.CONTEXT
        )
        contribution = await maybe_await(
            env.memory.before_compose(channel=env.channel, trace_id="trace-1", annotations={})
        )
        assert contribution is not None
        assert contribution.keys == [f"memory {rowid})"]
        await env.db.close()


async def seed_three(env: SimpleNamespace) -> list[int]:
    # Three records that all match the query below and pass the open gate.
    return [
        await seed_memory(env.db, env.channel.id, f"kestrel {name} the rain gauge fence")
        for name in ("moved", "painted", "removed")
    ]


QUERY = "where is the rain gauge fence?"


class TestAdmittedEvent:
    """M3: on_held_admitted writes the retrieval records, for the matching trace."""

    async def test_event_writes_stats_log_candidates_and_trace_profile(self):
        env = await build_env(gate_open())
        a, b, c = await seed_three(env)
        contribution = await compose(env, QUERY)
        # Precondition: all three were offered, so each outcome below is a choice
        # the event made, not a gate result.
        assert sorted(contribution.keys) == sorted(f"memory {i})" for i in (a, b, c))

        await admitted(
            env, admitted_keys=[f"memory {a})"], duplicate_keys=[f"memory {b})"]
        )

        # Access stats moved for the admitted id only.
        stats = {
            row[0]: row[1:]
            for row in await fetch_all(
                env.db, "SELECT id, retrieval_count, last_retrieved_at FROM memory"
            )
        }
        assert stats[a][0] == 1 and stats[a][1] is not None
        assert stats[b] == (0, None)
        assert stats[c] == (0, None)

        # One retrieval_log row with the trace and the admitted count.
        log = await fetch_all(
            env.db,
            "SELECT channel_id, trace_id, hit_count, admitted_count FROM retrieval_log",
        )
        assert log == [(env.channel.id, "trace-1", 3, 1)]

        # Candidate outcomes: admitted, duplicate, and what neither list names
        # (dropped for budget).
        outcomes = dict(
            await fetch_all(
                env.db, "SELECT memory_id, outcome FROM retrieval_candidate"
            )
        )
        assert outcomes == {a: "admitted", b: "dropped_dup", c: "dropped_budget"}

        # The retrieval profile reaches the trace once, from source memory.
        records = env.recorder.by_source("memory")
        assert len(records) == 1
        assert records[0]["trace_id"] == "trace-1"
        assert records[0]["channel_id"] == env.channel.id
        assert records[0]["record"]["retrieval_hit_count"] == 3
        await env.db.close()

    async def test_event_for_another_trace_or_source_writes_nothing(self):
        env = await build_env(gate_open())
        a, _b, _c = await seed_three(env)
        await compose(env, QUERY, trace_id="trace-1")

        # A different trace, and a different source on the same trace: neither
        # is memory's record to close, and neither may consume the stash.
        await admitted(env, admitted_keys=[f"memory {a})"], duplicate_keys=[], trace_id="other")
        await admitted(
            env, admitted_keys=[f"memory {a})"], duplicate_keys=[], source="goals"
        )
        assert await fetch_all(env.db, "SELECT * FROM retrieval_log") == []
        assert await fetch_all(env.db, "SELECT retrieval_count FROM memory WHERE id = %d" % a) == [(0,)]
        assert env.recorder.by_source("memory") == []

        # The matching event still finds its stash.
        await admitted(env, admitted_keys=[f"memory {a})"], duplicate_keys=[])
        assert len(await fetch_all(env.db, "SELECT * FROM retrieval_log")) == 1
        await env.db.close()


class TestEmptyContribution:
    """M4: nothing selected still closes the record when the event arrives."""

    async def test_all_weak_candidates_still_write_a_retrieval_log_row(self):
        # Default gate: the stub embedder's cosine puts an unrelated record
        # below the weak band, so the contribution is empty.
        env = await build_env()
        weak = await seed_memory(
            env.db, env.channel.id, "the greenhouse humidity sensor reads four percent high"
        )
        contribution = await compose(env, "qwertyuiop zxcvbnm asdfghjkl")
        assert isinstance(contribution, held_mod().Contribution)
        assert contribution.entries == []

        # Nothing was admitted; the event still arrives (the store offers
        # empty contributions to the event) and closes the records.
        await admitted(env, admitted_keys=[], duplicate_keys=[])
        log = await fetch_all(
            env.db, "SELECT trace_id, admitted_count FROM retrieval_log"
        )
        assert log == [("trace-1", 0)]
        outcomes = dict(
            await fetch_all(env.db, "SELECT memory_id, outcome FROM retrieval_candidate")
        )
        assert outcomes.get(weak) == "dropped_weak"
        await env.db.close()


LONG_DIALOG = [
    ("user", "the weather station keeps dropping off wifi every night"),
    ("assistant", "try WIFI_PS_NONE and check the brownout detector"),
    ("user", " ".join(f"reading{i}=21.{i}C" for i in range(60))),
    ("assistant", "the readings look fine; it is the radio resetting"),
]

#: Small enough to force several raw pieces from LONG_DIALOG.
TARGET = 40


class TestPieceSize:
    """M5: raw-transcript pieces fit memory.entry_target_tokens, whatever funnel says."""

    @pytest.mark.parametrize(
        "funnel",
        [
            None,
            {"budgets": {"memory": 4096}},
            {"default_budget": 4096},
        ],
        ids=["no-funnel-config", "funnel.budgets.memory", "funnel.default_budget"],
    )
    async def test_pieces_are_sized_to_the_entry_target_not_the_funnel_budget(self, funnel):
        config = copy.deepcopy(CONSOLIDATION_CONFIG)
        config["memory"]["entry_target_tokens"] = TARGET
        if funnel is not None:
            config["funnel"] = funnel
        memory, persistence, _stub, channel, db = await build_consolidation_env(config)
        try:
            memory._llm()._clients["main"] = _SequenceChatClient(["nope", "still not json"])
            rowids = await seed_dialog(persistence, channel, LONG_DIALOG)
            await memory.on_compaction(
                channel=channel, summary_msg={}, retain_count=0, compacted_ids=rowids,
            )
            await memory.wait_for_background_tasks()
            rows = await fetch_stamped_memories(db)
            # Several pieces, each a memory line that fits the target.
            assert len(rows) > 1
            for r in rows:
                assert count_tokens(WORST_MEMORY_LINE_PREFIX + r["summary"]) <= TARGET, r["summary"]
        finally:
            await db.close()

    def test_memory_module_reads_no_funnel_config(self):
        # R5.4 as a mechanical check: memory.py has no funnel config read left
        # (the plugin's docstrings may still mention the funnel by name).
        import corvidae_cognition.memory as memory_mod

        source = inspect.getsource(memory_mod)
        assert 'config.get("funnel"' not in source
        assert "funnel_budgets" not in source
