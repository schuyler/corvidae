"""The Zulip catch-up cursor (#644): what survives a restart, and what a bad
file costs.

The cursor is the transport's memory of how far it has read in each channel.
It is the only thing standing between a restart and a silently dropped
request, so its failure modes matter more than its happy path: a file that
cannot be read must cost one gap, never a failed start, and a write must never
leave half a file behind.
"""

import json
import logging

import pytest

from corvidae.channels.zulip import CURSOR_FILE, ZulipPlugin


@pytest.fixture
def plugin(tmp_path):
    p = ZulipPlugin()
    p._cursor_dir = tmp_path
    return p


def _write(tmp_path, text):
    (tmp_path / CURSOR_FILE).write_text(text)
    return tmp_path / CURSOR_FILE


async def test_a_missing_file_is_a_first_run(plugin, caplog):
    with caplog.at_level(logging.DEBUG, logger="corvidae.zulip"):
        await plugin._load_cursor()

    assert plugin._cursor == {}
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


async def test_save_then_load_round_trips(plugin, tmp_path):
    plugin._remember_seen("zulip:7", [["stream", "7"]], 42)
    await plugin._save_cursor()

    fresh = ZulipPlugin()
    fresh._cursor_dir = tmp_path
    await fresh._load_cursor()

    assert fresh._cursor == {"zulip:7": {"last": 42, "narrow": [["stream", "7"]]}}


async def test_the_save_leaves_no_temporary_file(plugin, tmp_path):
    plugin._remember_seen("zulip:7", [["stream", "7"]], 42)
    await plugin._save_cursor()

    assert sorted(p.name for p in tmp_path.iterdir()) == [CURSOR_FILE]


async def test_a_corrupt_file_costs_one_gap_not_a_failed_start(plugin, tmp_path, caplog):
    path = _write(tmp_path, "{not json")
    with caplog.at_level(logging.WARNING, logger="corvidae.zulip"):
        with caplog.at_level(logging.WARNING, logger="corvidae.zulip"):
            await plugin._load_cursor()

    assert plugin._cursor == {}
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any(str(path) in m and "not a JSON object" in m for m in warnings), warnings


async def test_a_bad_entry_is_dropped_and_named(plugin, tmp_path, caplog):
    _write(
        tmp_path,
        json.dumps(
            {
                "zulip:7": {"last": 5, "narrow": [["stream", "7"]]},
                "zulip:dm:1": {"last": "later", "narrow": []},
                "zulip:dm:2": {"narrow": []},
            }
        ),
    )
    with caplog.at_level(logging.WARNING, logger="corvidae.zulip"):
        await plugin._load_cursor()

    assert plugin._cursor == {"zulip:7": {"last": 5, "narrow": [["stream", "7"]]}}
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("2 unreadable entries" in m for m in warnings), warnings


async def test_a_repeat_does_not_move_the_cursor_backwards(plugin):
    assert plugin._remember_seen("zulip:7", [["stream", "7"]], 42) is True
    assert plugin._remember_seen("zulip:7", [["stream", "7"]], 42) is False
    assert plugin._remember_seen("zulip:7", [["stream", "7"]], 41) is False
    assert plugin._cursor["zulip:7"]["last"] == 42


async def test_the_narrow_is_stored_with_the_id(plugin):
    plugin._remember_seen("zulip:dm:1,2", [["dm", "1,2"]], 7)
    assert plugin._cursor["zulip:dm:1,2"]["narrow"] == [["dm", "1,2"]]


async def test_a_failed_save_warns_and_does_not_raise(plugin, tmp_path, caplog):
    plugin._cursor_dir = tmp_path / "gone"
    plugin._remember_seen("zulip:7", [["stream", "7"]], 42)
    with caplog.at_level(logging.WARNING, logger="corvidae.zulip"):
        await plugin._save_cursor()

    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("could not save the catch-up cursor" in m for m in warnings), warnings
