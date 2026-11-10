"""Funnel key dedupe and per-window source caps (#624, #632).

Key dedupe: ``admit_detailed(..., keys=[...])`` takes an optional list
parallel to ``entries``. An entry is a duplicate when its key appears in a
CONTEXT row framed for the same source, so a source can dedupe on a stable
identity (memory's record id) instead of the rendered line, which carries a
changing age. Sources that pass no keys keep whole-line, window-wide dedupe.

Window cap: ``funnel.window_budgets.<source>`` bounds the tokens of that
source's CONTEXT one window holds. An admission fits only what is left under
the cap, and compaction frees what it drops.
"""

import logging

import aiosqlite
from corvidae_cognition.funnel import FRAME_HEADER, FunnelPlugin

from corvidae.channel import Channel
from corvidae.context import ContextWindow, MessageType, count_tokens
from corvidae.hooks import create_plugin_manager
from corvidae.persistence import PersistencePlugin, init_db


async def build_funnel(config: dict | None = None):
    """Register persistence + funnel on a fresh pm with an in-memory DB.

    Returns (funnel, channel, conv, db).
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

    channel = Channel(transport="test", scope="keys")
    conv = ContextWindow(channel.id)
    channel.conversation = conv
    return funnel, channel, conv, db


def source_rows(conv: ContextWindow, source: str) -> list[str]:
    """Contents of the window's CONTEXT rows framed for ``source``."""
    header = FRAME_HEADER.format(source=source)
    return [
        m["content"] for m in conv.messages
        if m.get("_message_type") == MessageType.CONTEXT
        and m["content"].startswith(header)
    ]


class TestKeyDedupe:
    async def test_same_key_with_different_text_is_a_duplicate(self):
        """The age (and band) on a line change; the key does not (#624)."""
        funnel, channel, conv, db = await build_funnel()
        await funnel.admit_detailed(
            channel, conv, "memory", ["(5h, memory 7) a fact"],
            keys=["memory 7)"],
        )
        result = await funnel.admit_detailed(
            channel, conv, "memory", ["[marginal] (6h, memory 7) a fact"],
            keys=["memory 7)"],
        )
        # Nothing new is appended, and the caller is told it was a duplicate.
        assert result.admitted == []
        assert result.duplicate == ["[marginal] (6h, memory 7) a fact"]
        assert len(conv.messages) == 1
        await db.close()

    async def test_shorter_id_is_not_inside_a_longer_one(self):
        """Key "memory 12)" must not match a window holding "memory 123)",
        nor the reverse."""
        funnel, channel, conv, db = await build_funnel()
        await funnel.admit_detailed(
            channel, conv, "memory", ["(1h, memory 123) long id"],
            keys=["memory 123)"],
        )
        short = await funnel.admit_detailed(
            channel, conv, "memory", ["(1h, memory 12) short id"],
            keys=["memory 12)"],
        )
        assert short.admitted == ["(1h, memory 12) short id"]

        # The reverse: a window holding id 12 doesn't block id 123.
        funnel2, channel2, conv2, db2 = await build_funnel()
        await funnel2.admit_detailed(
            channel2, conv2, "memory", ["(1h, memory 12) short id"],
            keys=["memory 12)"],
        )
        long = await funnel2.admit_detailed(
            channel2, conv2, "memory", ["(1h, memory 123) long id"],
            keys=["memory 123)"],
        )
        assert long.admitted == ["(1h, memory 123) long id"]
        await db.close()
        await db2.close()

    async def test_key_in_another_sources_context_does_not_dedupe(self):
        """A key matches only CONTEXT framed for the same source, so another
        source quoting a memory line can't suppress the memory."""
        funnel, channel, conv, db = await build_funnel()
        await funnel.admit(
            channel, conv, "goals", ["goal note quoting (2h, memory 9) x"],
        )
        result = await funnel.admit_detailed(
            channel, conv, "memory", ["(2h, memory 9) x"], keys=["memory 9)"],
        )
        assert result.admitted == ["(2h, memory 9) x"]
        assert len(source_rows(conv, "memory")) == 1
        await db.close()

    async def test_keyless_entries_keep_window_wide_line_dedupe(self):
        """Without keys, an entry whose exact text is in any CONTEXT row,
        whatever its source, is a duplicate — as before."""
        funnel, channel, conv, db = await build_funnel()
        await funnel.admit(channel, conv, "goals", ["shared exact line"])
        result = await funnel.admit_detailed(
            channel, conv, "memory", ["shared exact line"],
        )
        assert result.admitted == []
        assert result.duplicate == ["shared exact line"]
        await db.close()

    async def test_keys_length_mismatch_raises_naming_both_lengths(self):
        funnel, channel, conv, db = await build_funnel()
        try:
            await funnel.admit_detailed(
                channel, conv, "memory", ["one", "two"], keys=["k1"],
            )
        except ValueError as exc:
            message = str(exc)
        else:
            raise AssertionError("a keys/entries length mismatch must raise")
        # The message names both lengths and the source.
        assert "1 keys" in message and "2 entries" in message
        assert "memory" in message
        # Nothing was appended.
        assert conv.messages == []
        await db.close()

    async def test_admit_passes_keys_through(self):
        """admit() takes the same keys keyword as admit_detailed()."""
        funnel, channel, conv, db = await build_funnel()
        await funnel.admit(channel, conv, "memory", ["(1h, memory 3) y"],
                           keys=["memory 3)"])
        again = await funnel.admit(channel, conv, "memory", ["(2h, memory 3) y"],
                                   keys=["memory 3)"])
        assert again == []
        await db.close()


# An entry of a known size for cap arithmetic.
ENTRY = "alpha bravo charlie delta echo"


class TestWindowCap:
    async def test_cap_admits_up_to_the_ceiling_then_nothing(self, caplog):
        """Entries are admitted while the source's CONTEXT in the window plus
        the new entries stay within the cap; at the cap, nothing is, and the
        drop is logged with the channel, source and config key."""
        funnel, channel, conv, db = await build_funnel()
        # Size the cap to hold exactly one framed row of one entry.
        await funnel.admit(channel, conv, "memory", [ENTRY])
        one_row = count_tokens(conv.messages[0]["content"])

        funnel, channel, conv, db2 = await build_funnel(
            {"funnel": {"window_budgets": {"memory": one_row}}}
        )
        first = await funnel.admit(channel, conv, "memory", [ENTRY])
        assert first == [ENTRY]

        with caplog.at_level(logging.INFO, logger="corvidae.funnel"):
            second = await funnel.admit(channel, conv, "memory", ["foxtrot"])
        assert second == []
        assert len(source_rows(conv, "memory")) == 1
        messages = [r.getMessage() for r in caplog.records]
        assert any(
            channel.id in m and "memory" in m
            and "funnel.window_budgets.memory" in m
            for m in messages
        ), messages
        await db.close()
        await db2.close()

    async def test_cap_skips_entries_that_do_not_fit_in_order(self):
        """Under the cap, an entry that would overrun what's left is skipped
        and a smaller one after it still fits."""
        long_entry = " ".join(["word"] * 60)
        cap = count_tokens("short one") + count_tokens("short two") + 5
        funnel, channel, conv, db = await build_funnel(
            {"funnel": {"window_budgets": {"memory": cap}}}
        )
        admitted = await funnel.admit(
            channel, conv, "memory", ["short one", long_entry, "short two"],
        )
        assert admitted == ["short one", "short two"]
        await db.close()

    async def test_cap_counts_only_that_sources_rows(self):
        """Other sources' CONTEXT neither counts toward the memory cap nor is
        capped by it."""
        funnel, channel, conv, db = await build_funnel(
            {"funnel": {"window_budgets": {"memory": 200}}}
        )
        big = " ".join(["goalword"] * 150)
        assert await funnel.admit(channel, conv, "goals", [big]) == [big]
        # The goals row is far over 200 tokens, yet memory still has room.
        assert await funnel.admit(channel, conv, "memory", [ENTRY]) == [ENTRY]
        await db.close()

    async def test_unset_cap_means_no_ceiling(self):
        """With no cap configured, the window holds well past any plausible
        built-in default (the example's 4096 included)."""
        funnel, channel, conv, db = await build_funnel()
        big = " ".join(["word"] * 400)
        for i in range(15):
            admitted = await funnel.admit(
                channel, conv, "memory", [f"{big} number {i}"],
            )
            assert admitted == [f"{big} number {i}"]
        held = sum(count_tokens(c) for c in source_rows(conv, "memory"))
        assert held > 4096
        await db.close()

    async def test_compaction_frees_the_cap(self):
        """Once compaction drops the source's rows from the window, the
        source admits again."""
        funnel, channel, conv, db = await build_funnel()
        await funnel.admit(channel, conv, "memory", [ENTRY])
        one_row = count_tokens(conv.messages[0]["content"])
        await db.close()

        funnel, channel, conv, db = await build_funnel(
            {"funnel": {"window_budgets": {"memory": one_row}}}
        )
        assert await funnel.admit(channel, conv, "memory", [ENTRY]) == [ENTRY]
        assert await funnel.admit(channel, conv, "memory", ["foxtrot"]) == []

        # Compaction replaces everything but the retained tail with a summary.
        conv.append({"role": "user", "content": "later message"})
        conv.replace_with_summary({"role": "assistant", "content": "summary"}, 1)

        assert await funnel.admit(channel, conv, "memory", ["foxtrot"]) == ["foxtrot"]
        await db.close()

    async def test_cap_applies_with_an_explicit_budget(self):
        funnel, channel, conv, db = await build_funnel(
            {"funnel": {"window_budgets": {"memory": 3}}}
        )
        admitted = await funnel.admit(
            channel, conv, "memory", [ENTRY], budget_tokens=10_000,
        )
        assert admitted == []
        await db.close()

    async def test_zero_cap_disables_the_source(self):
        funnel, channel, conv, db = await build_funnel(
            {"funnel": {"window_budgets": {"memory": 0}}}
        )
        assert await funnel.admit(channel, conv, "memory", ["x"]) == []
        assert conv.messages == []
        await db.close()


class TestWindowCapConfig:
    async def test_invalid_values_warn_and_are_ignored(self, caplog):
        """A non-integer, bool or negative cap warns naming the key, the value
        and what is accepted, and leaves that source uncapped."""
        bad = {"memory": "lots", "goals": True, "critique": -1, "notes": 1.5}
        with caplog.at_level(logging.WARNING, logger="corvidae.funnel"):
            funnel, channel, conv, db = await build_funnel(
                {"funnel": {"window_budgets": bad}}
            )
        warnings = [r.getMessage() for r in caplog.records
                    if r.levelno == logging.WARNING]
        for source, value in bad.items():
            assert any(
                f"funnel.window_budgets.{source}" in m and repr(value) in m
                and "non-negative integer" in m
                for m in warnings
            ), (source, warnings)
        # Each source stays uncapped.
        big = " ".join(["word"] * 300)
        for source in bad:
            assert await funnel.admit(channel, conv, source, [f"{source} {big}"]) != []
        await db.close()

    async def test_non_mapping_window_budgets_warns_and_is_ignored(self, caplog):
        with caplog.at_level(logging.WARNING, logger="corvidae.funnel"):
            funnel, channel, conv, db = await build_funnel(
                {"funnel": {"window_budgets": [4096]}}
            )
        warnings = [r.getMessage() for r in caplog.records
                    if r.levelno == logging.WARNING]
        assert any(
            "funnel.window_budgets" in m and "[4096]" in m and "mapping" in m
            for m in warnings
        ), warnings
        big = " ".join(["word"] * 300)
        assert await funnel.admit(channel, conv, "memory", [big]) == [big]
        await db.close()

    async def test_window_budgets_is_reloadable(self):
        funnel, channel, conv, db = await build_funnel()
        assert "funnel.window_budgets" in FunnelPlugin.reloadable_keys
        await funnel.on_config_reload(
            {"funnel": {"window_budgets": {"memory": 0}}}
        )
        assert await funnel.admit(channel, conv, "memory", ["x"]) == []
        await db.close()


class TestDeferredOverCap:
    async def test_entry_larger_than_cap_is_dropped_with_warning(self, caplog):
        """A deferred entry larger than its source's cap can never be
        admitted, so the drain drops it with a warning naming the key instead
        of re-queueing it forever (#623 shape)."""
        funnel, channel, conv, db = await build_funnel(
            {"funnel": {"window_budgets": {"critique": 60}}}
        )
        oversized = ("charlie " * 100).strip()
        funnel.register(channel, source="critique",
                              entries=[oversized, "small verdict"])

        with caplog.at_level(logging.WARNING, logger="corvidae.funnel"):
            await funnel.before_agent_turn(
                channel=channel, trace_id="t1", annotations={"origin": "user"}
            )

        window = "\n".join(source_rows(conv, "critique"))
        assert "small verdict" in window
        assert "charlie" not in window
        warnings = [r.getMessage() for r in caplog.records
                    if r.levelno == logging.WARNING]
        assert any(
            channel.id in m and "funnel.window_budgets.critique" in m
            for m in warnings
        ), warnings
        # Nothing stays queued for the channel.
        assert not funnel._next_turn.get(channel.id)
        await db.close()
